"""
agent.py
--------
A ReAct agent for the Career Intelligence Platform: Think -> Act ->
Observe -> Repeat, using Groq's native tool calling.

Tools (3, so this clears the "agent uses multiple tools" bar with room
to spare):
  - search_career_corpus : wraps your existing hybrid vector+graph
                            retrieval (retrieval_core.retrieve_and_rank)
  - compare_roles         : finds shared vs. unique skills/tools between
                            two roles in the corpus
  - calculate_skill_gap   : deterministic, no-LLM-call gap analysis
                            between a user's stated skills and a target
                            role's required skills/tools

Planning containment (so this isn't just "an agent" but a *contained*
agent, matching the curriculum's Week 10 intent):
  - max_steps          : hard ceiling on Think/Act cycles (default 5)
  - max_tool_retries   : how many times the exact same tool+args
                          combination is allowed before being blocked
  - loop detection      : an identical (tool, arguments) signature
                          repeated beyond max_tool_retries gets blocked
                          with an explicit observation telling the model
                          to try something else or finalize -- and the
                          loop hard-stops right after, it does not just
                          warn and keep spinning
  - no-progress cutoff  : if two consecutive steps produce zero NEW
                          tool-call signatures (i.e. the model is only
                          re-running things it already tried, even if
                          individually still under the retry limit),
                          the agent stops itself rather than burning the
                          remaining step budget

Every run returns the final answer AND a full trace of what the agent
tried, which step it happened on, and why it stopped -- useful both for
your own debugging and as a thing to literally show in a demo.

This module does NOT import anything from app_main.py (and app_main.py
does not need to be running for you to test this file directly -- see
the __main__ block at the bottom).
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from retrieval_core import retrieve_and_rank

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def tool_search_career_corpus(args):
    query = args.get("query", "")
    limit = int(args.get("limit", 5))
    results = retrieve_and_rank(query, limit=limit, min_trust_score=0.1)

    if not results:
        return {"matches": [], "summary": "No matching roles found in the corpus for this query."}

    summary_lines = [
        f"{r['title']} (match_score={r['match_score']}, trust={r['trust_score']}): "
        f"{r['context_description'][:200]}"
        for r in results
    ]
    return {
        "matches": [
            {"title": r["title"], "skills": r["skills"], "tools": r["tools"], "match_score": r["match_score"]}
            for r in results
        ],
        "summary": "\n".join(summary_lines),
    }


def tool_compare_roles(args):
    role_a_query = args.get("role_a", "")
    role_b_query = args.get("role_b", "")

    match_a = retrieve_and_rank(role_a_query, limit=1, min_trust_score=0.0)
    match_b = retrieve_and_rank(role_b_query, limit=1, min_trust_score=0.0)

    if not match_a or not match_b:
        missing = role_a_query if not match_a else role_b_query
        return {"error": f"Could not find a corpus entry closely matching '{missing}'."}

    a, b = match_a[0], match_b[0]
    skills_a, skills_b = set(a["skills"]), set(b["skills"])
    tools_a, tools_b = set(a["tools"]), set(b["tools"])

    return {
        "role_a": a["title"],
        "role_b": b["title"],
        "shared_skills": sorted(skills_a & skills_b),
        "unique_to_role_a": sorted(skills_a - skills_b),
        "unique_to_role_b": sorted(skills_b - skills_a),
        "shared_tools": sorted(tools_a & tools_b),
    }


def tool_calculate_skill_gap(args):
    current_skills = [s.strip().lower() for s in args.get("current_skills", []) if s.strip()]
    target_role_query = args.get("target_role", "")

    match = retrieve_and_rank(target_role_query, limit=1, min_trust_score=0.0)
    if not match:
        return {"error": f"Could not find a corpus entry closely matching '{target_role_query}'."}

    role = match[0]
    required = [s for s in (role["skills"] + role["tools"]) if s.strip()]
    required_lookup = {s.lower(): s for s in required}  # de-dupes case-insensitively, keeps original casing

    matched = [orig for lower, orig in required_lookup.items() if lower in current_skills]
    missing = [orig for lower, orig in required_lookup.items() if lower not in current_skills]
    readiness = round(100 * len(matched) / len(required_lookup), 1) if required_lookup else 0.0

    return {
        "target_role": role["title"],
        "matched_skills": matched,
        "missing_skills": missing,
        "readiness_percent": readiness,
    }


TOOL_DISPATCH = {
    "search_career_corpus": tool_search_career_corpus,
    "compare_roles": tool_compare_roles,
    "calculate_skill_gap": tool_calculate_skill_gap,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_career_corpus",
            "description": "Search the career role corpus (hybrid vector + graph retrieval) for roles matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. a role name, skill, or tool."},
                    "limit": {"type": "integer", "description": "Max number of roles to return.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_roles",
            "description": "Compare two roles in the corpus and return their shared and unique skills/tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role_a": {"type": "string", "description": "First role name."},
                    "role_b": {"type": "string", "description": "Second role name."},
                },
                "required": ["role_a", "role_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_skill_gap",
            "description": "Compute which of a target role's required skills/tools the user already has and which are missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills/tools the user already knows.",
                    },
                    "target_role": {"type": "string", "description": "The role to evaluate readiness for."},
                },
                "required": ["current_skills", "target_role"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a Career Intelligence agent for a student-facing dashboard. "
    "You have tools to search a career role corpus, compare two roles, and "
    "calculate a skill gap against a target role. Use tools whenever you need "
    "real data about roles, skills, or tools -- do not invent role details "
    "from general knowledge. Once you have enough information, give a clear, "
    "specific final answer directly to the user without calling any more tools."
)


# --------------------------------------------------------------------------
# Planning containment helpers
# --------------------------------------------------------------------------
def _tool_call_signature(tool_call):
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}
    return f"{tool_call.function.name}:{json.dumps(args, sort_keys=True)}"


def _tool_message(tool_call, payload):
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(payload),
    }


def _execute_tool(tool_call, max_tool_retries):
    """Runs a tool, retrying on exceptions up to max_tool_retries times."""
    tool_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    handler = TOOL_DISPATCH.get(tool_name)
    attempts = 0
    error = None
    result = None

    while attempts <= max_tool_retries:
        attempts += 1
        try:
            if handler is None:
                raise ValueError(f"Unknown tool requested: {tool_name}")
            result = handler(args)
            error = None
            break
        except Exception as exc:
            error = str(exc)
            continue

    return {
        "tool": tool_name,
        "arguments": args,
        "attempts": attempts,
        "result": result if error is None else None,
        "error": error,
    }


# --------------------------------------------------------------------------
# Main ReAct loop
# --------------------------------------------------------------------------
def run_agent(query, session_id=None, current_skills=None, max_steps=5, max_tool_retries=2):
    user_content = query
    if current_skills:
        user_content += f"\n\n(The user's current skills: {', '.join(current_skills)})"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    trace = []
    seen_signatures = {}
    final_answer = None
    terminated_reason = None
    loop_detected = False
    no_progress_streak = 0
    step = 0

    while step < max_steps and final_answer is None and terminated_reason is None:
        step += 1
        try:
            response = groq_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=900,
            )
        except Exception as exc:
            terminated_reason = f"llm_error: {exc}"
            break

        message = response.choices[0].message
        messages.append(message)

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            final_answer = message.content
            terminated_reason = "completed"
            break

        # --- classify each requested tool call before running anything ---
        blocked_calls, runnable_calls, new_signatures_this_step = [], [], 0
        for tool_call in tool_calls:
            signature = _tool_call_signature(tool_call)
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            if seen_signatures[signature] == 1:
                new_signatures_this_step += 1
            if seen_signatures[signature] > max_tool_retries:
                blocked_calls.append(tool_call)
            else:
                runnable_calls.append(tool_call)

        # --- no-progress cutoff ---
        if new_signatures_this_step == 0:
            no_progress_streak += 1
        else:
            no_progress_streak = 0

        # --- loop detection: block repeats beyond the retry limit ---
        if blocked_calls:
            loop_detected = True
            for tool_call in blocked_calls:
                trace.append({
                    "step": step,
                    "event": "loop_blocked",
                    "tool": tool_call.function.name,
                    "detail": "Identical tool+arguments already attempted beyond the retry limit.",
                })
                messages.append(_tool_message(tool_call, {
                    "error": "This exact tool call has already been attempted. "
                             "Use a different tool, different arguments, or give your final answer now."
                }))

        for tool_call in runnable_calls:
            execution = _execute_tool(tool_call, max_tool_retries)
            trace.append({"step": step, **execution})
            payload = execution["result"] if execution["error"] is None else {"error": execution["error"]}
            messages.append(_tool_message(tool_call, payload))

        if blocked_calls:
            # Hard stop right here -- don't let it spin for the remaining steps.
            terminated_reason = "loop_detected"
            break

        if no_progress_streak >= 2:
            terminated_reason = "no_progress"
            break

    if terminated_reason is None and final_answer is None:
        terminated_reason = "max_steps_reached"

    if final_answer is None:
        # Forced wrap-up: ask for a best-effort answer using whatever was gathered.
        messages.append({
            "role": "user",
            "content": "Stop using tools. Give your best final answer now, based only on what you've already gathered.",
        })
        try:
            response = groq_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=900,
            )
            final_answer = response.choices[0].message.content
        except Exception as exc:
            final_answer = (
                "I wasn't able to safely complete this request within the step limit. "
                "Try a more specific query, or fewer combined questions at once."
            )
            terminated_reason = f"{terminated_reason}+forced_wrapup_failed"

    return {
        "final_answer": final_answer,
        "trace": trace,
        "containment": {
            "steps_used": step,
            "max_steps": max_steps,
            "terminated_reason": terminated_reason,
            "loop_detected": loop_detected,
        },
    }


if __name__ == "__main__":
    # Quick standalone smoke test -- does NOT require the FastAPI server
    # to be running, just a working .env (Groq key + Chroma/Neo4j access).
    import sys

    test_query = " ".join(sys.argv[1:]) or (
        "I know PostgreSQL and React. How close am I to being a Full Stack Developer, "
        "and how does that role compare to a Python Backend Engineer?"
    )

    print(f"Query: {test_query}\n")
    outcome = run_agent(test_query, current_skills=["PostgreSQL", "React"])

    print("--- TRACE ---")
    for entry in outcome["trace"]:
        print(entry)

    print("\n--- CONTAINMENT ---")
    print(outcome["containment"])

    print("\n--- FINAL ANSWER ---")
    print(outcome["final_answer"])

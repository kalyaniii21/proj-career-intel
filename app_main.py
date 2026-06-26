import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

from memory_manager import UpstashSessionMemory
from retrieval_core import collection, close_connections, retrieve_and_rank, format_results_as_context
from agent import run_agent

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

session_memory = None
try:
    session_memory = UpstashSessionMemory()
except Exception as exc:
    print(f"Session memory disabled: {exc}")

app = FastAPI(title="Agentic Career Intelligence Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_fallback_insight(user_query: str, processed_results: list):
    if not processed_results:
        return (
            f"Career Intelligence Brief: {user_query}\n\n"
            "Overview\n"
            "No close matches were found in the local vector store for this query. "
            "Try using a broader role name, a core skill, or a tool-stack phrase such as "
            "'backend developer APIs databases' or 'data analyst SQL dashboards'.\n\n"
            "How to Improve the Search\n"
            "- Use a target role instead of a full sentence.\n"
            "- Include 2-3 important tools or skills.\n"
            "- Lower the trust threshold if you want exploratory results.\n\n"
            "Next Step\n"
            "Run another query with a clearer role and skill combination so the system can retrieve stronger matches."
        )

    role_sections = []
    responsibility_lines = []
    project_ideas = []
    skill_keywords = []

    for index, item in enumerate(processed_results[:3], start=1):
        description = item["context_description"]
        role_sections.append(
            f"{index}. {item['title']} "
            f"(trust score: {item['trust_score']}, type: {item['type']})\n"
            f"   - What it involves: {description}\n"
            "   - Why it matters: This match shows the practical responsibilities and systems you should be ready to discuss."
        )
        responsibility_lines.append(f"- Understand and explain: {description}")

        lowered = description.lower()
        for keyword in [
            "python", "database", "api", "server", "frontend",
            "cloud", "devops", "ci/cd", "analytics", "dashboard",
        ]:
            if keyword in lowered and keyword not in skill_keywords:
                skill_keywords.append(keyword)

    if "database" in skill_keywords or "server" in skill_keywords or "api" in skill_keywords:
        project_ideas.append(
            "1. Build a backend API for a job tracker with authentication, database tables, validation, and clean error handling."
        )
    if "frontend" in skill_keywords or "dashboard" in skill_keywords:
        project_ideas.append(
            "2. Create a dashboard that consumes your API, displays search results, loading states, and useful empty/error states."
        )
    if "devops" in skill_keywords or "ci/cd" in skill_keywords or "cloud" in skill_keywords:
        project_ideas.append(
            "3. Add a deployment pipeline with environment variables, basic tests, and a short README explaining the architecture."
        )
    if not project_ideas:
        project_ideas = [
            "1. Build a small end-to-end project related to the target role and document the architecture.",
            "2. Add one measurable improvement such as faster search, cleaner data validation, or better user feedback.",
            "3. Prepare a short demo script that explains the problem, design, tradeoffs, and result.",
        ]

    skills_text = ", ".join(skill_keywords[:8]) if skill_keywords else "role fundamentals, project experience, communication, problem solving"

    return (
        f"Career Intelligence Brief: {user_query}\n\n"
        "Executive Summary\n"
        "The local retrieval engine found relevant career-role context, so this brief turns those matches into preparation guidance. "
        "The strongest signals should be treated as practical role expectations: what systems you may build, what tools you should explain, "
        "and what interview examples you need ready. Use this answer as a study map for responsibilities, skills, projects, and interview talking points.\n\n"
        "Best Matching Role Signals\n"
        + "\n\n".join(role_sections)
        + "\n\nCore Responsibilities\n"
        + "\n".join(responsibility_lines[:7])
        + "\n- Translate requirements into working features and explain your design choices clearly.\n"
        "- Debug issues methodically, read logs, test assumptions, and communicate what changed.\n"
        "- Think about reliability, maintainability, performance, and user impact while building.\n\n"
        "Skills to Prioritize\n"
        f"Technical Skills: {skills_text}.\n"
        "Tools and Platforms: databases, APIs, version control, deployment basics, debugging tools, and any framework used in your project.\n"
        "Professional Skills: clear communication, ownership, structured problem solving, documentation, and explaining tradeoffs.\n\n"
        "Project and Portfolio Ideas\n"
        + "\n".join(project_ideas)
        + "\n\nInterview Preparation\n"
        "Prepare 3 stories: one about building a feature, one about fixing a difficult bug, and one about improving performance or usability. "
        "For each story, explain the problem, your approach, the technical choices, the result, and what you learned. "
        "When asked about a skill, connect it to a project instead of only defining it.\n\n"
        "Next 7-Day Action Plan\n"
        "Day 1: Review the retrieved role descriptions and list the top responsibilities.\n"
        "Day 2: Choose one project that proves the most important responsibility.\n"
        "Day 3: Improve the project with validation, logging, or better error handling.\n"
        "Day 4: Write a short architecture note explaining data flow and tradeoffs.\n"
        "Day 5: Practice explaining one bug or challenge you solved.\n"
        "Day 6: Prepare answers for tools, databases, APIs, and deployment basics.\n"
        "Day 7: Do a mock interview and refine weak answers.\n\n"
        "Interview Talking Points\n"
        "- Start with the business or user problem.\n"
        "- Describe the technical design and why you chose it.\n"
        "- Mention one obstacle and how you solved it.\n"
        "- End with the outcome, metric, or learning."
    )


class QueryRequest(BaseModel):
    session_id: str
    query: str
    limit: int = 3
    min_trust_score: float = 0.1


class AgentQueryRequest(BaseModel):
    session_id: str
    query: str
    current_skills: list[str] = []
    max_steps: int = 5


@app.get("/api/session/{session_id}")
async def get_session_history(session_id: str):
    if not session_memory:
        return {"history": []}
    try:
        return {"history": session_memory.get_session_history(session_id)}
    except Exception as e:
        print(f"Session history unavailable: {str(e)}")
        return {"history": []}


@app.post("/api/search")
async def semantic_and_graph_search(request: QueryRequest):
    try:
        user_query = request.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if collection.count() == 0:
            return {
                "results": [],
                "ai_response": build_fallback_insight(user_query, []),
            }

        processed_results = retrieve_and_rank(user_query, request.limit, request.min_trust_score)
        unified_context = format_results_as_context(processed_results)

        system_prompt = f"""
You are an advanced Career Intelligence AI Advisor for a student-facing dashboard.
Write a detailed, practical career guidance report based only on the retrieved context below.

User Query:
{user_query}

Retrieved Hybrid Knowledge Context:
{unified_context}

Output requirements:
- Use clear section headings.
- Write 500-800 words when enough context is available.
- Keep the answer specific to the user's query and retrieved roles.
- Do not say "based on the context" repeatedly.
- Avoid generic motivational filler.
- Use plain text formatting that renders well in a dashboard.

Required structure:
Career Intelligence Brief
1. Executive Summary
Give a 3-5 sentence summary of what the query indicates and which career direction it maps to.
2. Role Fit Analysis
Explain the best matching role signals from the retrieved context and what each role usually expects.
3. Core Responsibilities
List 5-7 concrete responsibilities the student should understand.
4. Skills to Prioritize
Group skills into Technical Skills, Tools/Platforms, and Professional Skills.
5. Project and Portfolio Ideas
Recommend 2-3 specific project ideas the student can build or describe in interviews.
6. Interview Preparation
Give practical talking points and example themes the student should prepare.
7. Next 7-Day Action Plan
Provide a day-by-day preparation plan.
"""

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": system_prompt}],
                model=os.getenv("GROQ_SEARCH_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.2,
                max_tokens=1200,
            )
            ai_insight = chat_completion.choices[0].message.content
        except Exception as exc:
            print(f"Groq insight generation skipped: {exc}")
            ai_insight = build_fallback_insight(user_query, processed_results)

        if session_memory:
            try:
                session_memory.add_message_to_session(request.session_id, "user", user_query)
                session_memory.add_message_to_session(request.session_id, "assistant", ai_insight)
            except Exception as exc:
                print(f"Session memory write failed: {exc}")

        return {
            "results": processed_results,
            "ai_response": ai_insight,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"API Endpoint Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent")
async def agent_search(request: AgentQueryRequest):
    """ReAct agent endpoint: Think -> Act -> Observe -> Repeat, with
    planning containment (max steps, retry limits, loop detection).
    See agent.py for the implementation and containment.terminated_reason
    in the response for why the agent stopped."""
    try:
        user_query = request.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        agent_result = run_agent(
            query=user_query,
            session_id=request.session_id,
            current_skills=request.current_skills or None,
            max_steps=request.max_steps,
        )

        if session_memory:
            try:
                session_memory.add_message_to_session(request.session_id, "user", user_query)
                session_memory.add_message_to_session(request.session_id, "assistant", agent_result["final_answer"])
            except Exception as exc:
                print(f"Session memory write failed: {exc}")

        return agent_result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Agent Endpoint Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    close_connections()
    print("Connections flushed safely.")

"""
generate_benchmark.py
----------------------
Builds a real evaluation benchmark from data/career_intelligence_corpus.json,
split into dev / val / hidden sets, instead of 20 randomly-sampled
synthetic Q&A pairs from an unrelated Chroma collection (which is what the
original script did).

Three query types, mirroring the curriculum's "lookup / synthesis /
multi-step reasoning" categories:

  - lookup    : template-based, directly grounded in one role's own
                skills/tools/responsibilities. Zero hallucination risk,
                zero API cost, works even with no GROQ_API_KEY.
  - synthesis : compares two roles that share at least one skill or tool.
                Template-based for the same reason.
  - reasoning : "given these skills, what role/path fits" style questions,
                phrased by Groq for natural variety. Falls back to a
                template version if the API key is missing or the call
                fails, so the script always completes.

Every generated item carries `expected_entities` (which role(s) the
retrieval layer should surface) and `expected_keywords` (terms pulled
directly from the corpus, used by evaluate.py to check the generated
answer actually mentions real, grounded content).

What this does NOT do: the curriculum's 3-human-annotator + Cohen's kappa
process. That's a human process, not a script. What this DOES do instead:
an automated groundedness check that drops any LLM-generated item whose
expected_keywords don't actually appear in the source entity's text --
a cheap proxy for "is this question/answer actually grounded", not a
replacement for human review.
"""

import argparse
import json
import os
import random

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CORPUS_PATH = os.path.join("data", "career_intelligence_corpus.json")

LOOKUP_TEMPLATES = [
    "What does a {title} actually do day-to-day?",
    "What core skills should someone targeting a {title} role focus on?",
    "Which tools or technologies does a {title} typically use?",
    "What's a good portfolio project to demonstrate {title} skills?",
    "What are the main responsibilities of a {title}?",
    "What interview topics should I prepare for a {title} position?",
]

SYNTHESIS_TEMPLATES = [
    "How does a {title_a} differ from a {title_b} in day-to-day work?",
    "If I already know {shared_term}, should I lean toward {title_a} or {title_b}?",
    "What's the overlap and difference in tools used by a {title_a} versus a {title_b}?",
]

REASONING_FALLBACK_TEMPLATE = (
    "I'm comfortable with {shared_term} and want to grow my career. "
    "Based on a {title} role, what should I learn next and why?"
)


def as_list(value):
    return value if isinstance(value, list) else []


def normalize(text):
    return " ".join(text.lower().split())


def token_overlap(a, b):
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def is_near_duplicate(query, seen_queries, threshold=0.85):
    normalized = normalize(query)
    if normalized in seen_queries:
        return True
    return any(token_overlap(normalized, existing) >= threshold for existing in seen_queries)


def sample_keywords(item, max_count=4):
    pool = as_list(item.get("skills")) + as_list(item.get("tools"))
    pool = [p for p in pool if p.strip()]
    random.shuffle(pool)
    return pool[:max_count] if pool else []


# --------------------------------------------------------------------------
# Template-based generation (deterministic, fully grounded, free)
# --------------------------------------------------------------------------
def generate_lookup_items(corpus_data, count, seen_queries):
    items = []
    attempts = 0
    while len(items) < count and attempts < count * 5:
        attempts += 1
        item = random.choice(corpus_data)
        title = item.get("title", "Unknown Role")
        template = random.choice(LOOKUP_TEMPLATES)
        query = template.format(title=title)

        if is_near_duplicate(query, seen_queries):
            continue

        keywords = sample_keywords(item)
        if not keywords:
            continue

        seen_queries.add(normalize(query))
        items.append({
            "id": f"lookup_{len(items):04d}",
            "query": query,
            "type": "lookup",
            "expected_entities": [title],
            "expected_keywords": keywords,
            "source_entity_ids": [item.get("entity_id", title)],
            "generated_by": "template",
        })
    return items


def find_shared_term(item_a, item_b):
    pool_a = set(as_list(item_a.get("skills")) + as_list(item_a.get("tools")))
    pool_b = set(as_list(item_b.get("skills")) + as_list(item_b.get("tools")))
    shared = [t for t in (pool_a & pool_b) if t.strip()]
    return shared


def generate_synthesis_items(corpus_data, count, seen_queries):
    items = []
    attempts = 0
    if len(corpus_data) < 2:
        return items

    while len(items) < count and attempts < count * 8:
        attempts += 1
        item_a, item_b = random.sample(corpus_data, 2)
        shared_terms = find_shared_term(item_a, item_b)
        if not shared_terms:
            continue

        title_a, title_b = item_a.get("title", "Role A"), item_b.get("title", "Role B")
        shared_term = random.choice(shared_terms)
        template = random.choice(SYNTHESIS_TEMPLATES)
        query = template.format(title_a=title_a, title_b=title_b, shared_term=shared_term)

        if is_near_duplicate(query, seen_queries):
            continue

        keywords = list({shared_term, *sample_keywords(item_a, 2), *sample_keywords(item_b, 2)})
        seen_queries.add(normalize(query))
        items.append({
            "id": f"synthesis_{len(items):04d}",
            "query": query,
            "type": "synthesis",
            "expected_entities": [title_a, title_b],
            "expected_keywords": keywords[:5],
            "source_entity_ids": [
                item_a.get("entity_id", title_a),
                item_b.get("entity_id", title_b),
            ],
            "generated_by": "template",
        })
    return items


# --------------------------------------------------------------------------
# LLM-based generation for reasoning questions (with template fallback)
# --------------------------------------------------------------------------
def try_build_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY found -- reasoning questions will use the template fallback only.")
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as exc:
        print(f"Could not initialize Groq client ({exc}) -- using template fallback only.")
        return None


def llm_generate_reasoning_query(client, model, item, shared_term):
    title = item.get("title", "Unknown Role")
    responsibilities = " ".join(as_list(item.get("responsibilities"))[:3])

    prompt = f"""You are creating ONE evaluation question for a career-advice RAG system.

Role: {title}
Known responsibilities: {responsibilities}
A skill/tool the asker already has: {shared_term}

Write a natural first-person question a student might ask, where the
correct answer requires reasoning about this role (not just a fact lookup).
Return JSON ONLY, no markdown fences, in this exact shape:
{{"question": "..."}}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.4,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    question = parsed["question"].strip()
    if not question:
        raise ValueError("Empty question returned")
    return question


def generate_reasoning_items(corpus_data, count, seen_queries, groq_client, groq_model):
    items = []
    attempts = 0
    while len(items) < count and attempts < count * 5:
        attempts += 1
        item = random.choice(corpus_data)
        title = item.get("title", "Unknown Role")
        keywords = sample_keywords(item, 3)
        if not keywords:
            continue
        shared_term = keywords[0]

        query = None
        generated_by = "template"
        if groq_client:
            try:
                query = llm_generate_reasoning_query(groq_client, groq_model, item, shared_term)
                generated_by = "llm"
            except Exception as exc:
                print(f"  LLM reasoning generation failed ({exc}); using template fallback.")

        if not query:
            query = REASONING_FALLBACK_TEMPLATE.format(shared_term=shared_term, title=title)

        if is_near_duplicate(query, seen_queries):
            continue

        # Groundedness check: every keyword we claim is "expected" must
        # actually appear in this entity's own corpus text.
        full_text = json.dumps(item).lower()
        keywords = [k for k in keywords if k.lower() in full_text]
        if not keywords:
            continue

        seen_queries.add(normalize(query))
        items.append({
            "id": f"reasoning_{len(items):04d}",
            "query": query,
            "type": "reasoning",
            "expected_entities": [title],
            "expected_keywords": keywords,
            "source_entity_ids": [item.get("entity_id", title)],
            "generated_by": generated_by,
        })
    return items


# --------------------------------------------------------------------------
# Split + save
# --------------------------------------------------------------------------
def split_dataset(items, dev_ratio, val_ratio, hidden_ratio, seed):
    assert abs(dev_ratio + val_ratio + hidden_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    dev_end = int(n * dev_ratio)
    val_end = dev_end + int(n * val_ratio)

    return {
        "dev": shuffled[:dev_end],
        "val": shuffled[dev_end:val_end],
        "hidden": shuffled[val_end:],
    }


def save_split(name, items, output_dir):
    path = os.path.join(output_dir, f"{name}_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {len(items)} items to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate dev/val/hidden RAG evaluation benchmarks.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--count", type=int, default=150, help="Total number of queries to generate.")
    parser.add_argument("--lookup-ratio", type=float, default=0.5)
    parser.add_argument("--synthesis-ratio", type=float, default=0.3)
    parser.add_argument("--reasoning-ratio", type=float, default=0.2)
    parser.add_argument("--dev-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groq-model", default="llama-3.1-8b-instant")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Cannot find corpus at {args.corpus}")

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    if len(corpus_data) == 0:
        raise ValueError("Corpus is empty.")

    print(f"Loaded {len(corpus_data)} corpus entries.")
    random.seed(args.seed)

    lookup_count = round(args.count * args.lookup_ratio)
    synthesis_count = round(args.count * args.synthesis_ratio)
    reasoning_count = args.count - lookup_count - synthesis_count

    seen_queries = set()

    print(f"Generating {lookup_count} lookup questions...")
    lookup_items = generate_lookup_items(corpus_data, lookup_count, seen_queries)

    print(f"Generating {synthesis_count} synthesis questions...")
    synthesis_items = generate_synthesis_items(corpus_data, synthesis_count, seen_queries)

    print(f"Generating {reasoning_count} reasoning questions...")
    groq_client = try_build_groq_client()
    reasoning_items = generate_reasoning_items(
        corpus_data, reasoning_count, seen_queries, groq_client, args.groq_model
    )

    all_items = lookup_items + synthesis_items + reasoning_items
    print(f"\nTotal generated: {len(all_items)} "
          f"(lookup={len(lookup_items)}, synthesis={len(synthesis_items)}, "
          f"reasoning={len(reasoning_items)})")

    if len(all_items) < args.count * 0.5:
        print("WARNING: generated far fewer items than requested -- your corpus may be too "
              "small/homogeneous for this --count. Consider lowering --count or growing the corpus.")

    splits = split_dataset(all_items, args.dev_ratio, args.val_ratio, args.hidden_ratio, args.seed)
    for name, items in splits.items():
        save_split(name, items, args.output_dir)

    llm_generated = sum(1 for item in all_items if item.get("generated_by") == "llm")
    print(f"\n{llm_generated}/{len(all_items)} items were LLM-phrased; the rest are "
          "template-generated and fully deterministic.")
    print("Reminder: this replaces the 3-annotator/kappa process with an automated "
          "groundedness check, not human review. Spot-check a sample before treating "
          "this as a trusted gold-standard benchmark.")


if __name__ == "__main__":
    main()

import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from neo4j import GraphDatabase
from pydantic import BaseModel

from memory_manager import UpstashSessionMemory


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

CHROMA_DATA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="career_intelligence_vectors",
    embedding_function=embedding_func,
)

raw_uri = os.getenv("NEO4J_URI", "")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD")

if "+s://" in raw_uri:
    neo4j_uri = raw_uri.replace("+s://", "+ssc://")
else:
    neo4j_uri = raw_uri

graph_driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password),
)


def get_graph_relationships(entity_title: str):
    try:
        with graph_driver.session() as session:
            cypher_query = """
            MATCH (a:Entity)-[r:RELATED_TO]-(b:Entity)
            WHERE toLower(a.title) CONTAINS toLower($title)
            RETURN DISTINCT b.title AS connected_title, b.type AS connected_type
            LIMIT 3
            """
            graph_res = session.run(cypher_query, title=entity_title)
            return [
                f"{record['connected_title']} ({record['connected_type']})"
                for record in graph_res
            ]
    except Exception as exc:
        print(f"Graph enrichment skipped for '{entity_title}': {exc}")
        return []


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
            "python",
            "database",
            "api",
            "server",
            "frontend",
            "cloud",
            "devops",
            "ci/cd",
            "analytics",
            "dashboard",
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


def split_metadata_list(value: str):
    if not value:
        return []
    return [item.strip() for item in re.split(r",|\|", value) if item.strip()]


def calculate_semantic_score(distance):
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))


def calculate_keyword_boost(query, metadata):
    query_lower = query.lower()
    boost = 0.0
    reasons = []

    title = metadata.get("title", "")
    if title and title.lower() in query_lower:
        boost += 0.12
        reasons.append(f"title match: {title}")

    for alias in split_metadata_list(metadata.get("aliases", "")):
        if alias.lower() in query_lower:
            boost += 0.08
            reasons.append(f"alias match: {alias}")
            break

    matched_skills = [
        skill for skill in split_metadata_list(metadata.get("skills", ""))
        if skill.lower() in query_lower
    ]
    if matched_skills:
        boost += min(0.18, 0.05 * len(matched_skills))
        reasons.append("skill match: " + ", ".join(matched_skills[:4]))

    matched_tools = [
        tool for tool in split_metadata_list(metadata.get("tools", ""))
        if tool.lower() in query_lower
    ]
    if matched_tools:
        boost += min(0.12, 0.04 * len(matched_tools))
        reasons.append("tool match: " + ", ".join(matched_tools[:4]))

    return min(boost, 0.3), reasons


def confidence_label(score):
    if score >= 0.78:
        return "Strong Match"
    if score >= 0.62:
        return "Good Match"
    if score >= 0.45:
        return "Exploratory Match"
    return "Weak Match"


def build_context_block(entity_title, metadata, graph_context):
    return "\n".join([
        f"Role: {entity_title}",
        f"Description: {metadata.get('description', '')}{graph_context}",
        f"Aliases: {metadata.get('aliases', '')}",
        f"Skills: {metadata.get('skills', '')}",
        f"Tools: {metadata.get('tools', '')}",
        f"Responsibilities: {metadata.get('responsibilities', '')}",
        f"Project Ideas: {metadata.get('project_ideas', '')}",
        f"Interview Questions: {metadata.get('interview_questions', '')}",
    ])


class QueryRequest(BaseModel):
    session_id: str
    query: str
    limit: int = 3
    min_trust_score: float = 0.1


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

        total_records = collection.count()
        if total_records == 0:
            return {
                "results": [],
                "ai_response": build_fallback_insight(user_query, []),
            }

        candidate_count = min(total_records, max(request.limit * 4, request.limit))
        vector_results = collection.query(
            query_texts=[user_query],
            n_results=candidate_count,
        )

        processed_results = []
        context_blocks = []

        if vector_results and vector_results["documents"] and vector_results["documents"][0]:
            for i in range(len(vector_results["documents"][0])):
                doc_text = vector_results["documents"][0][i]
                metadata = vector_results["metadatas"][0][i]
                distance = vector_results["distances"][0][i]

                semantic_score = calculate_semantic_score(distance)
                source_trust = float(metadata.get("base_trust_score", 0.85))
                keyword_boost, match_reasons = calculate_keyword_boost(user_query, metadata)
                final_score = min(1.0, (semantic_score * 0.65) + (source_trust * 0.25) + keyword_boost)

                if final_score >= request.min_trust_score:
                    entity_title = metadata.get("title", "Unknown Entity")
                    graph_relationships = get_graph_relationships(entity_title)

                    graph_context = ""
                    if graph_relationships:
                        graph_context = f" | Adjacent Paths: {', '.join(graph_relationships)}."
                        match_reasons.append("graph relationship found")

                    if not match_reasons:
                        match_reasons.append("semantic similarity to role profile")

                    context_blocks.append(build_context_block(entity_title, metadata, graph_context))
                    processed_results.append({
                        "title": entity_title,
                        "type": metadata.get("type", "General Context"),
                        "match_score": round(final_score, 2),
                        "semantic_score": round(semantic_score, 2),
                        "source_trust": round(source_trust, 2),
                        "trust_score": round(final_score, 2),
                        "confidence_label": confidence_label(final_score),
                        "match_reasons": match_reasons[:4],
                        "context_description": metadata.get("description", doc_text) + graph_context,
                        "aliases": split_metadata_list(metadata.get("aliases", "")),
                        "skills": split_metadata_list(metadata.get("skills", "")),
                        "tools": split_metadata_list(metadata.get("tools", "")),
                        "responsibilities": split_metadata_list(metadata.get("responsibilities", "")),
                        "project_ideas": split_metadata_list(metadata.get("project_ideas", "")),
                        "interview_questions": split_metadata_list(metadata.get("interview_questions", "")),
                    })

        processed_results = sorted(
            processed_results,
            key=lambda item: item["match_score"],
            reverse=True,
        )[:request.limit]
        context_blocks = [
            "\n".join([
                f"Role: {item['title']}",
                f"Description: {item['context_description']}",
                f"Skills: {', '.join(item['skills'])}",
                f"Tools: {', '.join(item['tools'])}",
                f"Responsibilities: {' | '.join(item['responsibilities'])}",
                f"Project Ideas: {' | '.join(item['project_ideas'])}",
                f"Interview Questions: {' | '.join(item['interview_questions'])}",
                f"Match Score: {item['match_score']}",
                f"Source Trust: {item['source_trust']}",
            ])
            for item in processed_results
        ]

        unified_context = "\n\n".join(context_blocks) if context_blocks else "No corporate matches found in vector-graph layers."

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
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
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

    except Exception as e:
        print(f"API Endpoint Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    graph_driver.close()
    print("Connections flushed safely.")

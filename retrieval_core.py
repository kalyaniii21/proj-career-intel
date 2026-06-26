"""
retrieval_core.py
------------------
Hybrid vector + graph retrieval and trust-scoring logic, extracted out of
app_main.py so it can be shared by both the existing /api/search endpoint
and the new ReAct agent (agent.py) without a circular import.

Nothing here changes the *behavior* of your original app_main.py -- the
scoring formula, the graph enrichment, the sorting, all identical. This
file just gives that logic a name (`retrieve_and_rank`) and a home outside
the FastAPI endpoint function so other code can call it directly.
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

CHROMA_DATA_PATH = "chroma_db"
COLLECTION_NAME = "career_intelligence_vectors"

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
)

_raw_uri = os.getenv("NEO4J_URI", "")
_neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
_neo4j_password = os.getenv("NEO4J_PASSWORD")

if "+s://" in _raw_uri:
    _neo4j_uri = _raw_uri.replace("+s://", "+ssc://")
else:
    _neo4j_uri = _raw_uri

graph_driver = GraphDatabase.driver(_neo4j_uri, auth=(_neo4j_user, _neo4j_password))


def close_connections():
    graph_driver.close()


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


def retrieve_and_rank(query: str, limit: int = 3, min_trust_score: float = 0.1):
    """The exact scoring/ranking logic that used to live inline inside
    app_main.py's /api/search handler. Returns a list of result dicts,
    sorted by match_score descending, capped at `limit`. Returns []
    if the collection is empty or nothing clears min_trust_score."""
    total_records = collection.count()
    if total_records == 0:
        return []

    candidate_count = min(total_records, max(limit * 4, limit))
    vector_results = collection.query(query_texts=[query], n_results=candidate_count)

    processed_results = []

    if vector_results and vector_results["documents"] and vector_results["documents"][0]:
        for i in range(len(vector_results["documents"][0])):
            doc_text = vector_results["documents"][0][i]
            metadata = vector_results["metadatas"][0][i]
            distance = vector_results["distances"][0][i]

            semantic_score = calculate_semantic_score(distance)
            source_trust = float(metadata.get("base_trust_score", 0.85))
            keyword_boost, match_reasons = calculate_keyword_boost(query, metadata)
            final_score = min(1.0, (semantic_score * 0.65) + (source_trust * 0.25) + keyword_boost)

            if final_score >= min_trust_score:
                entity_title = metadata.get("title", "Unknown Entity")
                graph_relationships = get_graph_relationships(entity_title)
                graph_context = ""
                if graph_relationships:
                    graph_context = f" | Adjacent Paths: {', '.join(graph_relationships)}."
                    match_reasons.append("graph relationship found")
                if not match_reasons:
                    match_reasons.append("semantic similarity to role profile")

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

    return sorted(processed_results, key=lambda item: item["match_score"], reverse=True)[:limit]


def format_results_as_context(processed_results):
    """Builds the same per-result text block app_main.py feeds to Groq.
    Reused by the agent's search tool so the model gets human-readable
    observations, not raw JSON."""
    if not processed_results:
        return "No corpus matches found in vector-graph layers."

    blocks = [
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
    return "\n\n".join(blocks)

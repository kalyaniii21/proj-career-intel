import json
import os

import chromadb
from chromadb.utils import embedding_functions


COLLECTION_NAME = "career_intelligence_vectors"
CHROMA_DATA_PATH = "chroma_db"


def as_text_list(values):
    return values if isinstance(values, list) else []


def build_search_document(item):
    source = item.get("source_metadata", {})
    sections = [
        f"Title: {item.get('title', '')}",
        f"Type: {item.get('type', '')}",
        f"Aliases: {', '.join(as_text_list(item.get('aliases')))}",
        f"Description: {item.get('description', '')}",
        f"Skills: {', '.join(as_text_list(item.get('skills')))}",
        f"Tools: {', '.join(as_text_list(item.get('tools')))}",
        "Responsibilities: " + " ".join(as_text_list(item.get("responsibilities"))),
        "Project Ideas: " + " ".join(as_text_list(item.get("project_ideas"))),
        "Interview Questions: " + " ".join(as_text_list(item.get("interview_questions"))),
        f"Authority: {source.get('authority_type', 'curated_role_framework')}",
    ]
    return "\n".join(section for section in sections if section.strip())


def build_metadata(item):
    source = item.get("source_metadata", {})
    return {
        "title": item.get("title", "Unknown Role"),
        "type": item.get("type", "Role"),
        "description": item.get("description", ""),
        "aliases": ", ".join(as_text_list(item.get("aliases"))),
        "skills": ", ".join(as_text_list(item.get("skills"))),
        "tools": ", ".join(as_text_list(item.get("tools"))),
        "responsibilities": " | ".join(as_text_list(item.get("responsibilities"))),
        "project_ideas": " | ".join(as_text_list(item.get("project_ideas"))),
        "interview_questions": " | ".join(as_text_list(item.get("interview_questions"))),
        "authority": source.get("authority_type", "curated_role_framework"),
        "base_trust_score": float(source.get("base_trust_score", 0.85)),
    }


def rebuild_vector_db():
    corpus_path = os.path.join("data", "career_intelligence_corpus.json")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Cannot find dataset at {corpus_path}.")

    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus_data = json.load(file)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Cleared old vector cache.")
    except Exception:
        pass

    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

    documents = [build_search_document(item) for item in corpus_data]
    ids = [item.get("entity_id", f"career_role_{index}") for index, item in enumerate(corpus_data)]
    metadatas = [build_metadata(item) for item in corpus_data]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"Success. ChromaDB vector space populated with {len(corpus_data)} rich career records.")


if __name__ == "__main__":
    rebuild_vector_db()

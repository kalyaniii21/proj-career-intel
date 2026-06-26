"""
create_database.py
-------------------
Rebuilds the Chroma vector store from data/career_intelligence_corpus.json.

Changes vs. the original version:
1. Real token-aware chunking (default 512 tokens, 75 overlap) instead of
   one giant document per corpus entry. Short entries (the common case for
   this corpus) still end up as a single chunk -- chunking only kicks in
   when an entry's text actually exceeds chunk_size tokens.
2. Near-duplicate chunk removal via a from-scratch MinHash implementation
   (no extra dependency). This is O(n^2) chunk comparisons, which is fine
   for a corpus of a few hundred-thousand chunks at most. At real scale
   you'd want LSH banding instead -- out of scope here, flagged honestly.
3. A post-build validation report (chunk counts, token-length stats,
   how many entities got split) so you have *something* to point to for
   the "validate the corpus" step. This is NOT a substitute for the
   100-page human-reviewed boundary check the original curriculum spec
   calls for -- it's an automated sanity check, not a quality judgment.

Schema compatibility: every chunk's metadata still contains the exact same
keys app_main.py reads (title, type, description, aliases, skills, tools,
responsibilities, project_ideas, interview_questions, authority,
base_trust_score) plus three new fields: chunk_index, total_chunks,
parent_entity_id. app_main.py needs zero changes to keep working.
"""

import argparse
import hashlib
import json
import os
import statistics

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "career_intelligence_vectors"
CHROMA_DATA_PATH = "chroma_db"
DEFAULT_CORPUS_PATH = os.path.join("data", "career_intelligence_corpus.json")


# --------------------------------------------------------------------------
# Tokenizer (uses tiktoken if installed, falls back to whitespace tokens)
# --------------------------------------------------------------------------
class Tokenizer:
    def __init__(self):
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self.backend = "tiktoken"
        except ImportError:
            self._enc = None
            self.backend = "whitespace (install tiktoken for real token counts)"

    def encode(self, text):
        if self._enc:
            return self._enc.encode(text)
        return text.split()

    def decode(self, tokens):
        if self._enc:
            return self._enc.decode(tokens)
        return " ".join(tokens)

    def count(self, text):
        return len(self.encode(text))


def chunk_text(text, tokenizer, chunk_size=512, overlap=75):
    """Sliding-window token chunking. Returns [text] unchanged if it
    already fits in one chunk."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return chunks


# --------------------------------------------------------------------------
# MinHash near-duplicate detection (from scratch, no extra dependency)
# --------------------------------------------------------------------------
def _shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _hash_shingle(shingle, seed):
    return int(hashlib.md5(f"{seed}:{shingle}".encode("utf-8")).hexdigest(), 16)


def minhash_signature(text, num_hashes=32, k=5):
    shingles = _shingles(text, k=k)
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_hash_shingle(s, seed) for s in shingles) for seed in range(num_hashes))


def estimated_jaccard(sig_a, sig_b):
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def deduplicate_chunks(chunk_records, threshold=0.85, num_hashes=32, shingle_size=5):
    kept, kept_signatures = [], []
    removed = 0
    for record in chunk_records:
        sig = minhash_signature(record["text"], num_hashes=num_hashes, k=shingle_size)
        if any(estimated_jaccard(sig, existing) >= threshold for existing in kept_signatures):
            removed += 1
            continue
        kept.append(record)
        kept_signatures.append(sig)
    return kept, removed


# --------------------------------------------------------------------------
# Corpus -> chunk records
# --------------------------------------------------------------------------
def as_text_list(values):
    return values if isinstance(values, list) else []


def build_full_text(item):
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


def build_base_metadata(item):
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


def build_chunk_records(corpus_data, tokenizer, chunk_size, overlap):
    records = []
    for index, item in enumerate(corpus_data):
        entity_id = item.get("entity_id", f"career_role_{index}")
        full_text = build_full_text(item)
        base_metadata = build_base_metadata(item)

        text_chunks = chunk_text(full_text, tokenizer, chunk_size=chunk_size, overlap=overlap)
        total_chunks = len(text_chunks)

        for chunk_index, chunk_text_value in enumerate(text_chunks):
            metadata = dict(base_metadata)
            metadata["chunk_index"] = chunk_index
            metadata["total_chunks"] = total_chunks
            metadata["parent_entity_id"] = entity_id

            chunk_id = entity_id if total_chunks == 1 else f"{entity_id}__chunk{chunk_index}"
            records.append({
                "id": chunk_id,
                "text": chunk_text_value,
                "metadata": metadata,
            })
    return records


def print_validation_report(records, tokenizer, removed_duplicates):
    token_counts = [tokenizer.count(r["text"]) for r in records]
    multi_chunk_entities = {
        r["metadata"]["parent_entity_id"]
        for r in records
        if r["metadata"]["total_chunks"] > 1
    }

    print("\n--- Corpus Validation Report ---")
    print(f"Tokenizer backend:        {tokenizer.backend}")
    print(f"Total chunks (post-dedup): {len(records)}")
    print(f"Duplicate chunks removed:  {removed_duplicates}")
    print(f"Entities split into >1 chunk: {len(multi_chunk_entities)}")
    if token_counts:
        print(f"Chunk token length -- min/avg/max: "
              f"{min(token_counts)} / {round(statistics.mean(token_counts), 1)} / {max(token_counts)}")
    print("NOTE: this is an automated structural sanity check, not a human-reviewed")
    print("boundary validation. Spot-check a sample of chunks manually before treating")
    print("this corpus as production-ready.")
    print("---------------------------------\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def rebuild_vector_db(corpus_path, chroma_path, collection_name, chunk_size, overlap,
                       dedup_threshold, skip_dedup):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Cannot find dataset at {corpus_path}.")

    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus_data = json.load(file)

    tokenizer = Tokenizer()
    print(f"Loaded {len(corpus_data)} corpus entries. Tokenizer backend: {tokenizer.backend}")

    chunk_records = build_chunk_records(corpus_data, tokenizer, chunk_size, overlap)
    print(f"Built {len(chunk_records)} chunks before dedup "
          f"(chunk_size={chunk_size}, overlap={overlap}).")

    removed_duplicates = 0
    if not skip_dedup:
        chunk_records, removed_duplicates = deduplicate_chunks(
            chunk_records, threshold=dedup_threshold
        )
        print(f"Removed {removed_duplicates} near-duplicate chunks "
              f"(MinHash threshold={dedup_threshold}).")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma_client.delete_collection(collection_name)
        print("Cleared old vector cache.")
    except Exception:
        pass

    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
    )

    # Batch inserts to stay safe on larger corpora.
    batch_size = 200
    for start in range(0, len(chunk_records), batch_size):
        batch = chunk_records[start:start + batch_size]
        collection.add(
            documents=[r["text"] for r in batch],
            ids=[r["id"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
        )

    print(f"Success. ChromaDB populated with {len(chunk_records)} chunk records "
          f"from {len(corpus_data)} corpus entries.")

    print_validation_report(chunk_records, tokenizer, removed_duplicates)


def parse_args():
    parser = argparse.ArgumentParser(description="Build/rebuild the career intelligence vector DB.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--chroma-path", default=CHROMA_DATA_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=75)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--no-dedup", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rebuild_vector_db(
        corpus_path=args.corpus,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        dedup_threshold=args.dedup_threshold,
        skip_dedup=args.no_dedup,
    )
# Career Intelligence Platform

Lightweight RAG (retrieval-augmented generation) demo using LangChain-style components, Chroma for vector storage, and local embedding models. This repo contains scripts to build a Chroma database from text, query it, and generate small synthetic benchmarks.

*Agentic Career Intelligence Platform:* 

A high-performance GraphRAG pipeline that unifies semantic vector spaces with structured knowledge graphs to deliver context-aware tech career trajectories and engineering insights.

#Core Architecture Flow: 

-Semantic Indexing: Ingests Kaggle job frameworks into a local high-dimensional vector space.
-Knowledge Graphing: Establishes structural RELATED_TO dependencies across clusters in the cloud.
-Hybrid Retrieval: Queries vector weights and graph nodes simultaneously to build deep context.
-Agentic Synthesis: Dispatches unified context payloads to Groq for precise insight generation.

#Tech Stack & Tools:

1. Frontend: React.js & JavaScript (ES6+) — Dark-mode analytics dashboard with trust scoring.
2. Backend: FastAPI (Python) - Asynchronous engine managing the core retrieval pipeline.
3. Vector DB: ChromaDB - Persistent vector store for semantic proximity mapping.
4. Graph DB: Neo4j Aura Cloud - Cloud-managed native graph database mapping entity paths.
5. LLM Engine: Groq API (Llama 3.3) - Ultra-low latency inference engine driving agentic synthesis.
6. Memory Layer: Upstash Redis - Serverless layer managing real-time session history tracking.
7. Environment: Dotenv & Virtualenv — Secure configuration separating tokens from runtime code.
## Features
- Build a Chroma vector DB from local text chunks
- Query the DB with simple retrieval workflows
- Generate a small synthetic benchmark using an LLM provider

## Requirements
- Python 3.10+ recommended
- `requirements.txt` lists necessary Python packages
- For `chromadb` you may need `onnxruntime` installed separately on some platforms

## Quickstart

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install "unstructured[md]"
```

3. Create the Chroma DB (this reads your source corpus and writes vectors):

```bash
python create_database.py
```

4. Query the DB:

```bash
python query_data.py "Your question here"
```

5. (Optional) Generate a small synthetic benchmark using your configured Groq/LLM key:

```bash
python generate_benchmark.py
```

## Notes on large files
- `data/job_descriptions.csv` is intentionally excluded from the repository (it is large). Keep large datasets out of Git — use Git LFS, releases, or external storage and add a small sample instead.

## Useful project files
- `create_database.py` — build and persist Chroma embeddings
- `query_data.py` — run a retrieval + answer flow against the DB
- `generate_benchmark.py` — synthesize evaluation Q/A pairs (requires an LLM key)
- `rag_engine.py`, `app_main.py` — higher-level orchestration and examples



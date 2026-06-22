import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# --- RELIABILITY FIX: Explicit Path Loading ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# DEBUG PRINT: This will confirm if the key is actually loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    print(f"Checking directory: {os.getcwd()}")
else:
    print(f"✅ API Key detected (starts with: {api_key[:6]}...)")

CHROMA_PATH = "chroma"
def main():
    # 1. Load your chunks
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma", embedding_function=embedding_function)
    data = db.get()
    chunks = data['documents']
    
    # Use the API key loaded from the environment instead of hard-coding secrets
    model = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    
    benchmark = []
    print(f"🧬 Starting Synthetic Mining from {len(chunks)} chunks...")

    # For this lab, let's generate a 20-query "Mini-Benchmark"
    for i in range(20):
        # Pick a random chunk for variety
        chunk = random.choice(chunks)
        
        prompt = f"""
        TASK: Generate a high-quality RAG evaluation pair.
        TEXT: {chunk}
        
        REQUIREMENTS:
        1. Question: Must be answerable ONLY using the text above.
        2. Answer: Must be concise and factually grounded in the text.
        
        FORMAT: JSON ONLY.
        {{ "question": "...", "answer": "..." }}
        """
        
        try:
            response = model.invoke(prompt)
            # Clean the response to ensure it's valid JSON
            clean_json = response.content.strip().replace("```json", "").replace("```", "")
            qa_pair = json.loads(clean_json)
            benchmark.append(qa_pair)
            print(f"✅ Generated Q{i+1}")
        except Exception as e:
            print(f"⚠️ Skip chunk {i}: {e}")

    # Save to file
    with open("dev_benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=4)
    
    print("\n🏁 Gate G1 Phase 1 Complete: 'dev_benchmark.json' created.")

if __name__ == "__main__":
    main()
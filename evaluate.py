import json
import time
import requests

def run_evaluation():
    # 1. Load your newly structured dev benchmark dataset
    try:
        with open("dev_benchmark.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Could not find 'dev_benchmark.json'. Please check your filename!")
        return

    print(f"📋 Loaded evaluation dataset with {len(dataset)} benchmark queries.")
    print("🚀 Starting RAG Pipeline Evaluation against live FastAPI Server...")
    
    results = []
    success_count = 0
    session_id = "eval_session_baseline"

    for i, item in enumerate(dataset, 1):
        # Gracefully handle both formats ('query' or old 'question')
        query_text = item.get("query") or item.get("question")
        expected = item.get("expected_answer") or item.get("answer")
        query_type = item.get("type", "unknown")

        print(f"\n({i}/{len(dataset)}) Evaluating Type: [{query_type}]")
        print(f"❓ Query: {query_text}")
        
        start_time = time.time()
        
        try:
            # 2. Make an HTTP POST request to your running FastAPI server
            response = requests.post(
                "http://127.0.0.1:8001/ask",
                json={"message": query_text, "session_id": session_id},
                timeout=15.0
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                ai_answer = data.get("answer", "")
                sources = data.get("sources", [])
                
                print(f"⏱️ Latency: {round(latency, 2)}s")
                print(f"🤖 AI Answer: {ai_answer[:80]}...")
                
                results.append({
                    "query": query_text,
                    "expected": expected,
                    "actual": ai_answer,
                    "sources": sources,
                    "latency": latency,
                    "type": query_type,
                    "status": "SUCCESS"
                })
                success_count += 1
            else:
                print(f"❌ Server Error: Status Code {response.status_code}")
                results.append({"query": query_text, "status": "SERVER_ERROR", "error": response.text})
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is your 'python api.py' server running on port 8001?")
            return
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"query": query_text, "status": "ERROR", "error": str(e)})
            
        # 3. Pause briefly to honor your backend's 6-second rate limiter middleware
        if i < len(dataset):
            print("⏳ Cool-down: Sleeping 6.5 seconds to bypass rate limiter...")
            time.sleep(6.5)

    # 4. Save clean metrics log to your workspace
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*50)
    print(f"✅ Benchmark Complete! Processed: {success_count}/{len(dataset)} queries.")
    print("📝 Full analytics output generated in 'evaluation_results.json'.")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
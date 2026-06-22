import requests
import json

API_URL = "http://localhost:8001/ask"
HEADERS = {"X-API-Key": "your_secret_admin_key"}

def run_evaluation():
    print("Loading test cases...")

    with open("eval_data.json", "r") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases\n")

    passed = 0

    for i, case in enumerate(test_cases, start=1):

        print("=" * 50)
        print(f"Test #{i}")
        print(f"Question: {case['question']}")

        try:
            print("Sending request to API...")

            response = requests.post(
                API_URL,
                json={"question": case["question"]},
                headers=HEADERS,
                timeout=30
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:

                data = response.json()
                answer = data.get("answer", "")

                print(f"Answer: {answer[:100]}")

                if case["expected"].lower() in answer.lower():
                    print("✅ PASS")
                    passed += 1
                else:
                    print(f"❌ FAIL")
                    print(f"Expected keyword: {case['expected']}")

            else:
                print(f"⚠️ API Error: {response.text}")

        except requests.exceptions.Timeout:
            print("⏰ Request timed out")

        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to FastAPI server")
            print("Make sure uvicorn server is running on port 8000")

        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\n" + "=" * 50)
    print(f"Final Score: {passed}/{len(test_cases)}")


if __name__ == "__main__":
    run_evaluation()
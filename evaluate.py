"""
evaluate.py
------------
Runs a benchmark (dev_benchmark.json / val_benchmark.json / hidden_benchmark.json,
as produced by generate_benchmark.py) against the LIVE running server.

Fixes vs. the original script:
1. Contract mismatch: the old version POSTed {"message", "session_id"} to
   /ask on port 8001. app_main.py actually exposes POST /api/search
   expecting {"session_id", "query", "limit", "min_trust_score"} and
   returns {"results": [...], "ai_response": "..."}. This version matches
   that contract exactly.
2. Real scoring instead of just logging latency:
   - retrieval_hit@1   : did the correct role appear as the #1 result?
   - retrieval_hit@k   : did the correct role appear anywhere in the
                         top-k results returned (k = the `limit` you pass)?
   - keyword_coverage  : what fraction of the benchmark item's
                         expected_keywords (real skills/tools pulled from
                         the corpus) actually show up in the generated
                         ai_response text?
   These are reported overall and broken out by query type (lookup /
   synthesis / reasoning), plus a pass/fail check against thresholds you
   can tune with --hit1-threshold / --hitk-threshold / --keyword-threshold.

Usage:
    python evaluate.py --dataset dev_benchmark.json
    python evaluate.py --dataset hidden_benchmark.json --base-url http://127.0.0.1:8001
"""

import argparse
import json
import time

import requests


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_search_api(base_url, query, session_id, limit, min_trust_score, timeout=20.0):
    response = requests.post(
        f"{base_url}/api/search",
        json={
            "session_id": session_id,
            "query": query,
            "limit": limit,
            "min_trust_score": min_trust_score,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def score_item(item, response_json):
    expected_entities = {e.lower() for e in item.get("expected_entities", [])}
    expected_keywords = item.get("expected_keywords", [])

    results = response_json.get("results", [])
    ai_response = (response_json.get("ai_response") or "").lower()

    result_titles = [r.get("title", "").lower() for r in results]

    hit_at_1 = bool(result_titles) and result_titles[0] in expected_entities
    hit_at_k = any(title in expected_entities for title in result_titles)

    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in ai_response)
        keyword_coverage = matched / len(expected_keywords)
    else:
        keyword_coverage = None

    return {
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "keyword_coverage": keyword_coverage,
        "num_results_returned": len(results),
    }


def summarize(results, group_key=None):
    """Aggregate metrics overall, or within a single type bucket if
    group_key is provided (e.g. 'lookup')."""
    subset = [r for r in results if r["status"] == "SUCCESS"]
    if group_key:
        subset = [r for r in subset if r["type"] == group_key]

    if not subset:
        return None

    hit1_rate = sum(1 for r in subset if r["hit_at_1"]) / len(subset)
    hitk_rate = sum(1 for r in subset if r["hit_at_k"]) / len(subset)

    coverages = [r["keyword_coverage"] for r in subset if r["keyword_coverage"] is not None]
    avg_keyword_coverage = sum(coverages) / len(coverages) if coverages else None

    avg_latency = sum(r["latency"] for r in subset) / len(subset)

    return {
        "count": len(subset),
        "hit_at_1_rate": round(hit1_rate, 3),
        "hit_at_k_rate": round(hitk_rate, 3),
        "avg_keyword_coverage": round(avg_keyword_coverage, 3) if avg_keyword_coverage is not None else None,
        "avg_latency_sec": round(avg_latency, 3),
    }


def print_summary_block(label, summary):
    if summary is None:
        print(f"{label}: no successful queries to score.")
        return
    print(f"{label} (n={summary['count']}):")
    print(f"  hit@1            = {summary['hit_at_1_rate']}")
    print(f"  hit@k            = {summary['hit_at_k_rate']}")
    print(f"  keyword coverage = {summary['avg_keyword_coverage']}")
    print(f"  avg latency (s)  = {summary['avg_latency_sec']}")


def run_evaluation(args):
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} benchmark queries from {args.dataset}")
    print(f"Targeting {args.base_url}/api/search\n")

    session_id = f"eval_session_{int(time.time())}"
    results = []

    for i, item in enumerate(dataset, 1):
        query = item.get("query")
        query_type = item.get("type", "unknown")
        print(f"({i}/{len(dataset)}) [{query_type}] {query}")

        start_time = time.time()
        try:
            response_json = call_search_api(
                args.base_url, query, session_id, args.limit, args.min_trust_score, args.timeout
            )
            latency = time.time() - start_time
            scored = score_item(item, response_json)
            results.append({
                "id": item.get("id"),
                "query": query,
                "type": query_type,
                "status": "SUCCESS",
                "latency": latency,
                **scored,
            })
            print(f"  hit@1={scored['hit_at_1']} hit@k={scored['hit_at_k']} "
                  f"keyword_coverage={scored['keyword_coverage']} latency={round(latency, 2)}s")
        except requests.exceptions.ConnectionError:
            print(f"  Connection error -- is your server running at {args.base_url}?")
            results.append({"id": item.get("id"), "query": query, "type": query_type,
                             "status": "CONNECTION_ERROR"})
        except requests.exceptions.HTTPError as exc:
            print(f"  HTTP error: {exc}")
            results.append({"id": item.get("id"), "query": query, "type": query_type,
                             "status": "HTTP_ERROR", "error": str(exc)})
        except Exception as exc:
            print(f"  Error: {exc}")
            results.append({"id": item.get("id"), "query": query, "type": query_type,
                             "status": "ERROR", "error": str(exc)})

        if args.delay > 0 and i < len(dataset):
            time.sleep(args.delay)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print("\n" + "=" * 60)
    print(f"Completed {success_count}/{len(dataset)} queries successfully.")
    print(f"Full per-query results written to {args.output}")
    print("=" * 60)

    overall = summarize(results)
    print_summary_block("\nOVERALL", overall)

    query_types = sorted({item.get("type", "unknown") for item in dataset})
    for qtype in query_types:
        print()
        print_summary_block(f"TYPE = {qtype}", summarize(results, group_key=qtype))

    print("\n--- Gate Check (tune thresholds with --hit1-threshold etc.) ---")
    if overall:
        checks = [
            ("hit@1", overall["hit_at_1_rate"], args.hit1_threshold),
            ("hit@k", overall["hit_at_k_rate"], args.hitk_threshold),
        ]
        if overall["avg_keyword_coverage"] is not None:
            checks.append(("keyword_coverage", overall["avg_keyword_coverage"], args.keyword_threshold))

        all_passed = True
        for name, value, threshold in checks:
            passed = value >= threshold
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name} = {value} (threshold {threshold})")
        print(f"\nOverall gate: {'PASS' if all_passed else 'FAIL'}")
    else:
        print("  No successful queries -- cannot evaluate gate.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Career Intelligence RAG API.")
    parser.add_argument("--dataset", default="dev_benchmark.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=5, help="Matches QueryRequest.limit")
    parser.add_argument("--min-trust-score", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to sleep between requests. Set to 0 if you have no rate limiter.")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--hit1-threshold", type=float, default=0.6)
    parser.add_argument("--hitk-threshold", type=float, default=0.85)
    parser.add_argument("--keyword-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
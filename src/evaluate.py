import json
import statistics
import time

import requests

from compute_score import compute_score

TIMEOUT_S = 10
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0
F1_PASS_THRESHOLD = 0.7  # tune against a hand-checked sample of your own data


def load_dataset(dataset_path):
    with open(dataset_path, "r") as file:
        return [json.loads(line) for line in file]


def call_with_retries(endpoint, payload):
    """
    POSTs payload to endpoint, retrying transient failures (timeouts,
    connection errors, 5xx) with exponential backoff. Does not retry
    4xx client errors, since retrying won't fix a bad request.
    Returns (parsed_json, latency_seconds).
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            response = requests.post(endpoint, json=payload, timeout=TIMEOUT_S)
            latency = time.time() - start
            if 400 <= response.status_code < 500:
                response.raise_for_status()  # client error - don't retry
            response.raise_for_status()      # 5xx - falls into except below, retried
            return response.json(), latency
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_exc = e
            if status is not None and 400 <= status < 500:
                raise  # client error, fail immediately
            # else: 5xx, fall through to retry
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
        except ValueError as e:
            # response.json() failed to parse - treat as non-retryable,
            # since a malformed body usually won't fix itself on retry
            raise
 
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE_S * (2 ** attempt))
 
    raise last_exc


def run_requests(endpoint, dataset_path):
    """
    Runs each test case against the endpoint. Returns (predictions, errors, latencies).
    A failure on one case does not stop the run for the rest.
    """
    cases = load_dataset(dataset_path)
    predictions = {}
    errors = []
    latencies = {}
 
    for case in cases:
        try:
            data, latency = call_with_retries(endpoint, {"id": case["id"], "input": case["input"]})
            content = data["choices"][0]["message"]["content"]
            predictions[case["id"]] = content
            latencies[case["id"]] = latency
        except Exception as e:
            errors.append({
                "id": case["id"],
                "input": case["input"],
                "error_type": type(e).__name__,
                "detail": str(e),
            })
 
    return predictions, errors, latencies


def summarize(dataset_path, predictions, errors):
    scores = compute_score(dataset_path, predictions)
 
    cases_by_id = {c["id"]: c for c in load_dataset(dataset_path)}
 
    failures = []
    passed_count = 0
    for case_id, score in scores.items():
        passed = score["exact_match"] or score["f1"] >= F1_PASS_THRESHOLD
        if passed:
            passed_count += 1
        else:
            failures.append({
                "id": case_id,
                "input": cases_by_id[case_id]["input"],
                "expected": cases_by_id[case_id]["expected"],
                "prediction": predictions[case_id],
                "f1": round(score["f1"], 4),
                "reason": f"low overlap (f1={score['f1']:.2f}) vs expected '{cases_by_id[case_id]['expected']}'",
            })
 
    total_cases = len(cases_by_id)
 
    return {
        "run_metadata": {
            "total_cases": total_cases,
            "completed": len(scores),
            "errored": len(errors),
        },
        "summary": {
            "pass_rate": round(passed_count / len(scores), 4) if scores else None,
            "error_rate": round(len(errors) / total_cases, 4) if total_cases else None,
            "mean_f1": round(statistics.mean(s["f1"] for s in scores.values()), 4) if scores else None,
        },
        "failures": failures,
        "errors": errors,
    }


if __name__ == "__main__":
    predictions, errors, latencies = run_requests("http://localhost:8000/v1/chat/completions", "src/test.jsonl")
    summary = summarize("src/test.jsonl", predictions, errors)
    print(json.dumps(summary, indent=2))

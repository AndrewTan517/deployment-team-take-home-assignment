import json

from src.evaluate import run_requests, summarize


def main():
    predictions, errors = run_requests("http://localhost:8000/v1/chat/completions", "src/test.jsonl")
    summary = summarize("src/test.jsonl", predictions, errors)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

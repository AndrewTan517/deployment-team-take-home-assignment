# Deployment Team Take-Home Assignment

## Parts A and C

You can find the answers to parts A and C in their respective PDF files in the root directory (`part-a.pdf` and `part-c.pdf`). Additionally, the architecture diagram used in `part-a.pdf` can be found at `images/architecture-diagram.png`.

## Part B

### Overview

The LLM evaluation harness sends a dataset of questions to a running LLM endpoint, collects its responses, and scores them against expected answers using exact match and F1 metrics. It produces a structured JSON report summarising pass rates, mean F1, per-case failures, and any request errors.

### Components

| File | Role |
|---|---|
| `main.py` | Entry point — wires together the endpoint URL, dataset path, and output |
| `src/evaluate.py` | Loads the dataset, calls the endpoint, and builds the summary report |
| `src/compute_score.py` | Scores individual predictions (exact match + token-level F1) |
| `src/endpoint.py` | Example FastAPI endpoint that the harness can be run against |

### Dataset format

The dataset is a JSONL file where each line is a JSON object with three fields:

```jsonl
{"id": "q1", "input": "What is the leave policy?", "expected": "14 days annual leave"}
{"id": "q2", "input": "Who approves travel claims?", "expected": "Direct manager"}
```

- `id` — unique identifier for the question
- `input` — the question text sent to the endpoint
- `expected` — the ground-truth answer used for scoring

### Endpoint contract

The harness POSTs each case to the endpoint as JSON:

```json
{ "id": "q1", "input": "What is the leave policy?" }
```

The endpoint must return an OpenAI-compatible chat completions response. The harness reads the answer from `choices[0].message.content`:

```json
{
  "choices": [
    { "message": { "content": "14 days annual leave" } }
  ]
}
```

### Running the harness

1. Start the endpoint (the bundled example uses `uvicorn`):

   ```bash
   uvicorn src.endpoint:app --reload
   ```

2. Run the harness:

   ```bash
   python main.py
   ```

   By default `main.py` points at `http://localhost:8000/v1/chat/completions` with `src/test.jsonl` as the dataset. Edit those values directly in `main.py` to target a different endpoint or dataset.

### Output

The harness prints a JSON report to stdout:

```json
{
  "run_metadata": {
    "total_cases": 2,
    "completed": 2,
    "errored": 0
  },
  "summary": {
    "pass_rate": 1.0,
    "error_rate": 0.0,
    "mean_f1": 1.0
  },
  "failures": [],
  "errors": []
}
```

- `pass_rate` — fraction of completed cases where `exact_match` is true **or** F1 ≥ 0.7
- `error_rate` — fraction of total cases that failed to reach the endpoint
- `mean_f1` — average token-level F1 across all scored cases
- `failures` — cases that were reached but scored below the pass threshold, including the prediction and expected answer for inspection
- `errors` — cases where the HTTP request itself failed (timeout, connection error, 5xx after retries)

### Scoring

Each prediction is normalised before comparison: lowercased, punctuation stripped, and articles (*a*, *an*, *the*) removed.

- **Exact match** — `1` if the normalised prediction equals the normalised expected answer, `0` otherwise.
- **F1** — token-level overlap between prediction and expected answer, computed as the harmonic mean of precision and recall over the shared token multiset.

A case **passes** if either its exact match score is `1` or its F1 score is ≥ 0.7.

### Retry behaviour

Transient failures (timeouts, connection errors, HTTP 5xx) are retried up to 3 times with exponential backoff (1 s, 2 s). HTTP 4xx errors and JSON parse failures are not retried, as they indicate a problem with the request itself.

### Future work

If time permits, possible future developments could include:
- Additional scoring metrics (BLEU, ROUGE, etc.)
- LLM-as-a-judge scoring
- Scoring against other tasks (e.g. hellaswag, GSM8k)
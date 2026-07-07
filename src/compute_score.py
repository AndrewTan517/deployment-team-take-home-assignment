# Adapted from huggingface/evaluate/metrics/squad/compute_score.py
import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_score(dataset, predictions):
    scores = {}
    with open(dataset, "r") as file:
        for line in file:
            line = json.loads(line)
            if line["id"] not in predictions:
                message = "Unanswered question " + line["id"] + " will receive score 0."
                print(message, file=sys.stderr)
                continue
            ground_truth = line["expected"]
            prediction = predictions[line["id"]]
            exact_match = exact_match_score(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)
            scores[line["id"]] = {"exact_match": exact_match, "f1": f1}

    return scores

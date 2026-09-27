"""
End-to-end RAG evaluation with DeepEval, one result file per generator model.

    python evaluation/run_deepeval.py                              # default model from app/config.py
    python evaluation/run_deepeval.py --model gemma3:4b
    python evaluation/run_deepeval.py --model llama3.2:3b --judge qwen3:4b-instruct
    python evaluation/run_deepeval.py --compare                    # table of every saved result

Results: evaluation/results/<model>.json  (":" -> "_" because Windows filenames can't contain ":")

--model  the model that ANSWERS the questions (the thing you are comparing)
--judge  the model that SCORES the answers. Keep it the SAME across runs:
         if the judge changes too, you can't tell whether a score moved
         because of the answer or because of a stricter/looser grader.

Note: this file used to be called deepeval.py. A script with the same name
as a package shadows it: `python evaluation/deepeval.py` puts evaluation/
first on sys.path, so `from deepeval.metrics import ...` found THIS file
instead of the installed library.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
APP_DIR = REPO_ROOT / "app"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_JUDGE = "qwen3:4b-instruct"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]


def safe_name(model):
    return model.replace(":", "_").replace("/", "_")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="generator model (default: LOCAL_MODEL from app/config.py)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help=f"evaluator model (default: {DEFAULT_JUDGE})")
    ap.add_argument("--compare", action="store_true", help="print a comparison of all saved results and exit")
    return ap.parse_args()


def compare_results():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No results in {RESULTS_DIR} yet.")
    cols = ["model", "judge", *METRIC_NAMES, "retrieval_hit", "latency_ms"]
    print("  ".join(f"{c:>22}" for c in cols))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        m = data["mean_scores"]
        vals = [data["model"], data["judge"]] + [m.get(c) for c in cols[2:]]
        print("  ".join(f"{('-' if v is None else round(v, 3) if isinstance(v, float) else v):>22}" for v in vals))


def model_installed(base_url, name):
    import httpx

    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        return False
    names = {t["name"] for t in tags}
    return name in names or f"{name}:latest" in names


def build_metrics(evaluator_model):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    kwargs = {"model": evaluator_model, "threshold": 0.5, "include_reason": True}
    return {
        "faithfulness": FaithfulnessMetric(**kwargs),
        "answer_relevancy": AnswerRelevancyMetric(**kwargs),
        "contextual_precision": ContextualPrecisionMetric(**kwargs),
        "contextual_recall": ContextualRecallMetric(**kwargs),
    }


def score_rows(eval_rows, judge, base_url):
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    evaluator = OllamaModel(model=judge, base_url=base_url, temperature=0)
    metrics = build_metrics(evaluator)

    results = []
    for row in eval_rows:
        print(f"Scoring {row['qid']}: {row['user_input']}")
        test_case = LLMTestCase(
            input=row["user_input"],
            actual_output=row["response"],
            expected_output=row["reference"],
            retrieval_context=row["retrieved_contexts"],
        )
        scores, reasons = {}, {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                scores[name] = float(metric.score)
                reasons[name] = metric.reason
            except Exception as e:  # noqa: BLE001 - one metric failure must not stop the batch
                print(f"  {name} failed: {e}")
                scores[name] = None
                reasons[name] = f"failed: {e}"
        results.append({**row, "scores": scores, "reasons": reasons})
    return results


def mean_scores(results):
    out = {}
    for name in METRIC_NAMES:
        vals = [r["scores"][name] for r in results if r["scores"].get(name) is not None]
        out[name] = round(statistics.mean(vals), 3) if vals else None
    out["retrieval_hit"] = round(statistics.mean(r["retrieval_hit"] for r in results), 3)
    out["latency_ms"] = round(statistics.median(r["latency_ms"] for r in results), 1)
    return out


def main():
    args = parse_args()
    if args.compare:
        compare_results()
        return

    # Must happen BEFORE importing anything from app/: config.py reads
    # LOCAL_MODEL once at import time and llm.py builds the client from it.
    if args.model:
        os.environ["LOCAL_MODEL"] = args.model
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    os.chdir(REPO_ROOT)  # app/ uses relative data paths

    from config import LOCAL_MODEL, OLLAMA_BASE_URL
    from eval_set import EVAL_SET, build_eval_rows, validate_eval_set

    problems = validate_eval_set()
    if problems:
        sys.exit("Eval set does not match chunks.jsonl:\n" + "\n".join(problems))

    for name in {LOCAL_MODEL, args.judge}:
        if not model_installed(OLLAMA_BASE_URL, name):
            sys.exit(f"Model '{name}' not found at {OLLAMA_BASE_URL}. Run `ollama pull {name}` (and make sure Ollama is running).")

    print(f"Generator: {LOCAL_MODEL} | Judge: {args.judge} | Questions: {len(EVAL_SET)}")

    eval_rows = build_eval_rows()
    if not eval_rows:
        sys.exit("No eval rows were produced - check the RAG pipeline.")

    results = score_rows(eval_rows, args.judge, OLLAMA_BASE_URL)
    summary = mean_scores(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{safe_name(LOCAL_MODEL)}.json"
    payload = {
        "model": LOCAL_MODEL,
        "judge": args.judge,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_questions": len(results),
        "mean_scores": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {out_path.relative_to(REPO_ROOT)}")
    print("\nMean scores:")
    for k, v in summary.items():
        print(f"  {k:>22}: {v}")


if __name__ == "__main__":
    main()

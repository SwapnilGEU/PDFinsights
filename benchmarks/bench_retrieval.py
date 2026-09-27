"""
Retrieval benchmark: how often does each method put the right chunk near the top?

Usage (from repo root or benchmarks/):
    python benchmarks/bench_retrieval.py                       # methods from bench_config
    python benchmarks/bench_retrieval.py --methods bm25,chroma
    python benchmarks/bench_retrieval.py --methods bm25,chroma,hybrid,rerank
    python benchmarks/bench_retrieval.py --refresh          # ignore cache, recompute

Metrics (all averaged over the golden questions):
    Hit@k     1 if ANY relevant chunk is in the top k             -> "did we find it at all?"
    Recall@k  fraction of the relevant chunks found in the top k  -> matters for multi-chunk answers
    MRR@k     1/rank of the first relevant chunk (0 if not in k)  -> "how high did it rank?"
    nDCG@k    like MRR but credits every relevant chunk by position
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bench_config import (
    CHUNKS_PATH,
    ENABLED_METHODS,
    FETCH_K,
    K_VALUES,
    RERANK_CANDIDATES,
    RESULTS_DIR,
)
from retrievers import BUILDERS
from validate_golden import load_golden, validate


def hit_at_k(ranked, relevant, k):
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked, relevant, k):
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


METRICS = {"hit": hit_at_k, "recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}


def first_relevant_rank(ranked, relevant):
    for rank, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return rank
    return None  # not found within FETCH_K


def percentile(values, p):
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


CACHE_DIR = Path(__file__).resolve().parent / "cache"


def fingerprint(name):
    """Anything that could change a method's rankings goes into this hash.
    Change chunks, app config (embedding/reranker model), retriever code or
    FETCH_K -> new fingerprint -> cache is ignored and rebuilt automatically."""
    bench = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in [CHUNKS_PATH, APP_DIR / "config.py", APP_DIR / "chunking.py", APP_DIR / "vectorstore.py", APP_DIR / "retrieval.py", APP_DIR / "reranker.py", bench / "retrievers.py"]:
        if f.exists():
            h.update(f.read_bytes())
    h.update(f"{name}|{FETCH_K}|{RERANK_CANDIDATES}".encode())
    return h.hexdigest()[:16]


def get_rankings(name, golden, use_cache=True):
    """Returns {question: ranked chunk_ids}, {question: latency_ms}, build_s.
    Models are only loaded if some question is missing from the cache."""
    cache_file = CACHE_DIR / f"{name}.json"
    fp = fingerprint(name)
    cache = {"fingerprint": fp, "build_s": 0.0, "rankings": {}, "latency_ms": {}}
    if use_cache and cache_file.exists():
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fp:
            cache = saved
        else:
            print("  cache is stale (chunks/config/code changed) -> recomputing")

    missing = [r["question"] for r in golden if r["question"] not in cache["rankings"]]
    if not missing:
        print(f"  all {len(golden)} rankings loaded from cache/{name}.json (no models loaded)")
        return cache["rankings"], cache["latency_ms"], cache["build_s"]

    print(f"  computing {len(missing)} rankings (loading models)...")
    t0 = time.perf_counter()
    search = BUILDERS[name]()
    cache["build_s"] = round(time.perf_counter() - t0, 2)
    search("warm up query")  # first call can include lazy model loading

    for q in missing:
        t = time.perf_counter()
        cache["rankings"][q] = [int(c) for c in search(q, k=FETCH_K)]
        cache["latency_ms"][q] = (time.perf_counter() - t) * 1000

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return cache["rankings"], cache["latency_ms"], cache["build_s"]


def dedupe(ranked):
    """Keep the first occurrence of each chunk_id. A retriever returning the same
    chunk twice (e.g. a vector store with duplicate entries) must not be credited
    twice - that is how nDCG once came out as 1.78."""
    seen = set()
    return [cid for cid in ranked if not (cid in seen or seen.add(cid))]


def run_method(name, golden, use_cache=True):
    rankings, lat_by_q, build_s = get_rankings(name, golden, use_cache)

    per_q, latencies = [], []
    for row in golden:
        relevant = set(row["relevant_chunk_ids"])
        ranked = dedupe(rankings[row["question"]])
        latencies.append(lat_by_q[row["question"]])

        rec = {
            "method": name,
            "qid": row["qid"],
            "qtype": row["qtype"],
            "question": row["question"],
            "relevant": row["relevant_chunk_ids"],
            "first_relevant_rank": first_relevant_rank(ranked, relevant),
            "top5": ranked[:5],
        }
        for k in K_VALUES:
            for m, fn in METRICS.items():
                rec[f"{m}@{k}"] = fn(ranked, relevant, k)
        per_q.append(rec)

    summary = {"method": name, "n": len(per_q), "build_s": round(build_s, 2)}
    for k in K_VALUES:
        for m in METRICS:
            summary[f"{m}@{k}"] = round(statistics.mean(r[f"{m}@{k}"] for r in per_q), 3)
    summary["lat_p50_ms"] = round(percentile(latencies, 0.5), 1)
    summary["lat_p95_ms"] = round(percentile(latencies, 0.95), 1)
    return summary, per_q


def print_table(summaries):
    cols = ["method"] + [f"hit@{k}" for k in K_VALUES] + [f"mrr@{K_VALUES[-1]}", f"ndcg@{K_VALUES[-1]}", "lat_p50_ms"]
    print("\n" + "  ".join(f"{c:>12}" for c in cols))
    for s in summaries:
        print("  ".join(f"{s[c]:>12}" for c in cols))


def print_by_qtype(per_q_all, k):
    by = defaultdict(lambda: defaultdict(list))
    for r in per_q_all:
        by[r["method"]][r["qtype"]].append(r[f"hit@{k}"])
    qtypes = sorted({r["qtype"] for r in per_q_all})
    print(f"\nHit@{k} by question type")
    print(f"{'method':>14}" + "".join(f"{q:>12}" for q in qtypes))
    for method, d in by.items():
        print(f"{method:>14}" + "".join(f"{statistics.mean(d[q]):>12.2f}" for q in qtypes))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(ENABLED_METHODS))
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and recompute everything")
    ap.add_argument("--show-misses", type=int, default=5, help="k used for the miss list")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in BUILDERS]
    if unknown:
        sys.exit(f"Unknown methods {unknown}. Available: {list(BUILDERS)}")

    golden = load_golden()
    problems = validate(golden)
    if problems:
        print("\n".join(problems))
        sys.exit("Golden set does not match chunks.jsonl (did you re-chunk?). Fix before benchmarking.")
    print(f"Golden set OK: {len(golden)} questions")

    summaries, per_q_all = [], []
    for m in methods:
        print(f"\n>>> {m}")
        s, pq = run_method(m, golden, use_cache=not args.refresh)
        summaries.append(s)
        per_q_all.extend(pq)

    print_table(summaries)
    print_by_qtype(per_q_all, args.show_misses)

    k = args.show_misses
    for m in methods:
        misses = [r for r in per_q_all if r["method"] == m and r[f"hit@{k}"] == 0]
        print(f"\n{m}: {len(misses)} misses at k={k}")
        for r in misses[:10]:
            print(f"  {r['qid']} rank={r['first_relevant_rank']} want={r['relevant']} got={r['top5']}  {r['question'][:60]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    write_csv(RESULTS_DIR / f"retrieval_summary_{ts}.csv", summaries)
    write_csv(RESULTS_DIR / f"retrieval_per_question_{ts}.csv", per_q_all)
    print(f"\nSaved results/retrieval_summary_{ts}.csv and retrieval_per_question_{ts}.csv")


if __name__ == "__main__":
    main()

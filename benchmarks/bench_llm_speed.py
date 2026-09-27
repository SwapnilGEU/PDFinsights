"""
LLM speed benchmark against the local Ollama server.

Every model gets the SAME RAG-style prompts (question + its golden context
chunks), the same options (temperature 0, seed, num_ctx, num_predict), a
warm-up call, and several repeats. We report medians.

    python benchmarks/bench_llm_speed.py
    python benchmarks/bench_llm_speed.py --models qwen3:4b-instruct,gemma3:4b --repeats 5

What each number means
    ttft_ms      time to first token, measured on the client via streaming.
                 In RAG this is dominated by prefill of the long context.
    prefill_tps  prompt tokens processed per second (prompt_eval_count / prompt_eval_duration)
    decode_tps   generated tokens per second (eval_count / eval_duration)  <- the usual "tok/s"
    total_ms     full request time on the client
    load_ms      model load time reported by Ollama (should be ~0 after warm-up)
    vram_gb      how much of the model sits on the GPU (from /api/ps). If this is
                 less than size_gb the model spilled to CPU RAM and will be slow.
"""

import argparse
import csv
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_config import (
    CHUNKS_PATH,
    CONTEXT_CHUNKS,
    GEN_OPTIONS,
    MODELS,
    N_PROMPTS,
    OLLAMA_URL,
    REPEATS,
    RESULTS_DIR,
)
from validate_golden import load_golden

SYSTEM_PROMPT = (
    "You are a helpful question-answering assistant for machine learning. "
    "Answer the question using ONLY the supplied context. "
    "If the answer is not present in the context, say: "
    "\"I don't know based on the provided context.\""
)

NS = 1e9  # Ollama reports durations in nanoseconds


def build_prompts(n):
    """Question + CONTEXT_CHUNKS chunks: the golden chunks first, padded with
    neighbouring chunks so every prompt has a realistic, similar length."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    golden = load_golden()
    step = max(1, len(golden) // n)  # spread prompts across the whole book
    prompts = []
    for row in golden[::step][:n]:
        ids = list(row["relevant_chunk_ids"])
        nxt = row["primary_chunk_id"] + 1
        while len(ids) < CONTEXT_CHUNKS and nxt in chunks:
            ids.append(nxt)
            nxt += 1
        context = "\n\n".join(
            f"[{chunks[i]['heading_path']}, p.{chunks[i]['page']}]\n{chunks[i]['text']}" for i in ids[:CONTEXT_CHUNKS]
        )
        user = f"Context:\n{context}\n\nQuestion:\n{row['question']}"
        prompts.append({"qid": row["qid"], "user": user})
    return prompts


def available_models(client):
    tags = client.get(f"{OLLAMA_URL}/api/tags").json().get("models", [])
    return {m["name"] for m in tags}


def chat_stream(client, model_cfg, user_msg, options, run_tag=""):
    """One streamed /api/chat call. Returns a dict of timings + token counts."""
    body = {
        "model": model_cfg["name"],
        "messages": [
            # run_tag goes FIRST so every request has a unique prefix. Ollama keeps
            # recent prompts in its KV cache; without this, repeat 2+ of the same
            # prompt skips prefill entirely and TTFT/prefill_tps look ~10x too good.
            {"role": "system", "content": f"[run {run_tag}] {SYSTEM_PROMPT}"},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "options": options,
    }
    if model_cfg.get("think") is not None:
        body["think"] = model_cfg["think"]

    t0 = time.perf_counter()
    ttft = None
    text = []
    thinking = []  # reasoning models stream their hidden thoughts here
    final = {}
    with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            thinking.append(msg.get("message", {}).get("thinking", "") or "")
            piece = msg.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            text.append(piece)
            if msg.get("done"):
                final = msg
    total_ms = (time.perf_counter() - t0) * 1000

    p_tok = final.get("prompt_eval_count", 0)
    p_dur = final.get("prompt_eval_duration", 0) / NS
    g_tok = final.get("eval_count", 0)
    g_dur = final.get("eval_duration", 0) / NS
    return {
        "ttft_ms": round(ttft or total_ms, 1),
        "total_ms": round(total_ms, 1),
        "load_ms": round(final.get("load_duration", 0) / 1e6, 1),
        "prompt_tokens": p_tok,
        "prefill_ms": round(p_dur * 1000, 1),
        "prefill_tps": round(p_tok / p_dur, 1) if p_dur else 0.0,
        "gen_tokens": g_tok,
        "decode_ms": round(g_dur * 1000, 1),
        "decode_tps": round(g_tok / g_dur, 1) if g_dur else 0.0,
        "total_tokens": p_tok + g_tok,
        "thinking_chars": len("".join(thinking)),
        "answer": "".join(text).strip(),
    }


def vram_usage(client, model_name):
    for m in client.get(f"{OLLAMA_URL}/api/ps").json().get("models", []):
        if m["name"] == model_name or m.get("model") == model_name:
            return round(m.get("size", 0) / 1e9, 2), round(m.get("size_vram", 0) / 1e9, 2)
    return None, None


def unload(client, model_name):
    """Free VRAM before the next model so models don't compete for the GPU."""
    client.post(f"{OLLAMA_URL}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=60)


def median(rows, key):
    return round(statistics.median(r[key] for r in rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma list, overrides bench_config.MODELS")
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    models = MODELS
    if args.models:
        models = [{"name": n.strip(), "think": None} for n in args.models.split(",")]

    client = httpx.Client()
    try:
        installed = available_models(client)
    except httpx.ConnectError:
        sys.exit(f"Ollama not reachable at {OLLAMA_URL}. Start it with `ollama serve`.")

    prompts = build_prompts(args.prompts)
    print(f"{len(prompts)} prompts x {args.repeats} repeats, options={GEN_OPTIONS}")

    runs, summaries = [], []
    for cfg in models:
        name = cfg["name"]
        if name not in installed and f"{name}:latest" not in installed:
            print(f"\n--- skipping {name}: not installed (run `ollama pull {name}`)")
            continue

        print(f"\n=== {name}")
        chat_stream(client, cfg, "Say OK.", GEN_OPTIONS)  # warm-up: loads weights, allocates KV cache
        size_gb, vram_gb = vram_usage(client, name)
        print(f"loaded: size={size_gb} GB, on GPU={vram_gb} GB")

        model_runs = []
        for rep in range(args.repeats):
            for p in prompts:
                res = chat_stream(client, cfg, p["user"], GEN_OPTIONS, run_tag=uuid.uuid4().hex[:8])
                res.update({"model": name, "qid": p["qid"], "repeat": rep})
                model_runs.append(res)
                if res["thinking_chars"]:
                    print(f"  WARNING: {name} is emitting reasoning tokens - speed is not comparable to instruct models")
                print(
                    f"  {p['qid']} r{rep}: ttft={res['ttft_ms']:.0f}ms "
                    f"prefill={res['prefill_tps']:.0f}t/s decode={res['decode_tps']:.1f}t/s "
                    f"({res['prompt_tokens']}+{res['gen_tokens']} tok)"
                )
        runs.extend(model_runs)

        summaries.append(
            {
                "model": name,
                "size_gb": size_gb,
                "vram_gb": vram_gb,
                "ttft_ms_p50": median(model_runs, "ttft_ms"),
                "total_ms_p50": median(model_runs, "total_ms"),
                "prefill_tps_p50": median(model_runs, "prefill_tps"),
                "decode_tps_p50": median(model_runs, "decode_tps"),
                "prompt_tokens_p50": median(model_runs, "prompt_tokens"),
                "gen_tokens_p50": median(model_runs, "gen_tokens"),
                "total_tokens_sum": sum(r["total_tokens"] for r in model_runs),
            }
        )
        unload(client, name)

    if not summaries:
        sys.exit("No models were benchmarked.")

    cols = ["model", "vram_gb", "ttft_ms_p50", "prefill_tps_p50", "decode_tps_p50", "total_ms_p50", "gen_tokens_p50"]
    print("\n" + "  ".join(f"{c:>18}" for c in cols))
    for s in summaries:
        print("  ".join(f"{s[c]!s:>18}" for c in cols))
    print("\nNote: prompt_tokens differ per model because each tokenizer splits text differently.")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for fname, rows in [(f"llm_speed_summary_{ts}.csv", summaries), (f"llm_speed_runs_{ts}.csv", runs)]:
        with open(RESULTS_DIR / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved results/llm_speed_summary_{ts}.csv (answers are in the runs CSV for later quality eval)")


if __name__ == "__main__":
    main()

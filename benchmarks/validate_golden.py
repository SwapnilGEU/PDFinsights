"""
Checks that golden_set.jsonl still lines up with data/processed/chunks.jsonl.

Every golden row stores a verbatim `evidence` snippet from its primary chunk.
If you re-chunk (new chunk size, new splitter...) the chunk_ids shift and the
golden labels silently become wrong. This catches that.

    python benchmarks/validate_golden.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_config import CHUNKS_PATH, GOLDEN_PATH


def load_golden(path=GOLDEN_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(golden):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    problems = []
    for row in golden:
        for cid in row["relevant_chunk_ids"]:
            if cid not in chunks:
                problems.append(f"{row['qid']}: chunk {cid} does not exist")
        primary = chunks.get(row["primary_chunk_id"])
        if primary and row["evidence"].lower() not in primary["text"].lower():
            problems.append(f"{row['qid']}: evidence not found in chunk {row['primary_chunk_id']}")
    return problems


if __name__ == "__main__":
    golden = load_golden()
    problems = validate(golden)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK - {len(golden)} golden questions match chunks.jsonl")

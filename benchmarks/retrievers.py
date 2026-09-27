"""
Every retriever here has the same shape:

    search(query: str, k: int) -> list[int]      # ranked chunk_ids, best first

That common shape is the whole trick: bench_retrieval.py doesn't care HOW a
method works, only which chunk_ids it returns and in what order. To add a new
method, write a builder that returns a search function and register it in
BUILDERS at the bottom.

Methods (in the order you build them):
    bm25        keyword search, same tokenizer as app/retrieval.py
    chroma      dense vector search (vectorstore.similarity_search). This is
                the same search as vectorstore.as_retriever("similarity") -
                the retriever is only a LangChain wrapper around it.
    chroma_mmr  dense search with MMR (relevant AND diverse results)
    hybrid      app/retrieval.hybrid_search: bm25 + chroma fused with RRF
    rerank      hybrid top-20 re-scored by the cross-encoder in app/reranker.py

Builders are lazy: `bm25` never imports torch/chroma, so you can benchmark it
before the vector store exists.
"""

import json
import os
import re
import sys

from bench_config import APP_DIR, CHUNKS_PATH, FETCH_K, REPO_ROOT, RERANK_CANDIDATES

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _use_app_modules():
    """app/*.py uses bare imports + relative data paths, so mimic running from repo root."""
    os.chdir(REPO_ROOT)
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))


def _load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- stage 1: BM25
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    """Must match app/retrieval.py tokenize(). Copied (not imported) because
    importing app/retrieval.py would also load the embedding model + Chroma."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25():
    from rank_bm25 import BM25Okapi

    chunks = _load_chunks()
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def search(query, k=FETCH_K):
        scores = bm25.get_scores(_tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [ids[i] for i in top]

    return search


# ---------------------------------------------------------------- stage 2: Chroma (dense)
def build_chroma():
    """Dense search: embed the query with the same model as the chunks, return nearest vectors."""
    _use_app_modules()
    from vectorstore import vectorstore

    def search(query, k=FETCH_K):
        docs = vectorstore.similarity_search(query, k=k)
        return [d.metadata["chunk_id"] for d in docs]

    return search


def build_chroma_mmr():
    """Optional: MMR trades a bit of relevance for diversity among results."""
    _use_app_modules()
    from vectorstore import vectorstore

    def search(query, k=FETCH_K):
        docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
        return [d.metadata["chunk_id"] for d in docs]

    return search


# ---------------------------------------------------------------- stage 3: hybrid (BM25 + dense, RRF)
def build_hybrid():
    _use_app_modules()
    from retrieval import hybrid_search

    def search(query, k=FETCH_K):
        docs = hybrid_search(query, k=k, fetch_k=FETCH_K)
        return [d.metadata["chunk_id"] for d in docs]

    return search


# ---------------------------------------------------------------- stage 4: reranker on top of hybrid
def build_rerank():
    _use_app_modules()
    from reranker import rerank
    from retrieval import hybrid_search

    def search(query, k=FETCH_K):
        candidates = hybrid_search(query, k=RERANK_CANDIDATES, fetch_k=FETCH_K)
        docs = rerank(query, candidates, top_k=k)
        return [d.metadata["chunk_id"] for d in docs]

    return search


BUILDERS = {
    "bm25": build_bm25,
    "chroma": build_chroma,
    "chroma_mmr": build_chroma_mmr,
    "hybrid": build_hybrid,
    "rerank": build_rerank,
}

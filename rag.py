"""
Simple, solid RAG — distilled from jxnl/systematically-improving-rag.

The lesson of that course in one sentence: RAG is a *retrieval* system you
improve with measurement, not a magic pipeline. So the method that matters
here is .evaluate() — it generates synthetic questions from your own chunks
and measures recall@k. That number is your flywheel. Every upgrade below
(persistence, reranking, routing) is opt-in and only earns its place when
evaluate() proves it moves the number.

No vector DB server, no new deps beyond openai + numpy. numpy cosine over a
matrix is exact and fast to ~100k chunks; .save()/.load() make it persistent;
the reranker and router reuse the OpenAI client you already have.

Deps: openai, numpy.  Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class RAG:
    def __init__(self, name: str = "default", description: str = "",
                 embed_model: str = EMBED_MODEL, chat_model: str = CHAT_MODEL):
        self.client = OpenAI()
        self.name = name                 # used by Router
        self.description = description    # what this collection is about (for routing)
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.chunks: list[str] = []
        self._emb: np.ndarray | None = None  # (n, d), L2-normalized rows

    # --- ingest -----------------------------------------------------------
    def add(self, docs: list[str], chunk_chars: int = 800, overlap: int = 100):
        """Chunk docs and embed. Call once per corpus, or repeatedly to append."""
        new = [c for d in docs for c in _chunk(d, chunk_chars, overlap)]
        if not new:
            return self
        vecs = self._embed(new)
        self.chunks += new
        self._emb = vecs if self._emb is None else np.vstack([self._emb, vecs])
        return self

    # --- retrieve ---------------------------------------------------------
    def search(self, query: str, k: int = 5, fetch: int = 20,
               rerank: bool = False) -> list[tuple[str, float]]:
        """Cosine top-k. With rerank=True, pull `fetch` by cosine then let an
        LLM reorder them (the usual next win once recall is your ceiling)."""
        q = self._embed([query])[0]
        scores = self._emb @ q
        idx = list(self._topk(q, fetch if rerank else k))
        if rerank:
            order = self._rerank_order(query, [self.chunks[i] for i in idx])
            idx = [idx[o] for o in order]
        idx = idx[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def answer(self, query: str, k: int = 5, rerank: bool = False) -> str:
        hits = self.search(query, k, rerank=rerank)
        context = "\n\n".join(f"[{i+1}] {c}" for i, (c, _) in enumerate(hits))
        msg = [
            {"role": "system", "content":
             "Answer using ONLY the numbered context. Cite sources like [1]. "
             "If the context doesn't contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        r = self.client.chat.completions.create(model=self.chat_model, messages=msg)
        return r.choices[0].message.content

    # --- the point: measure retrieval ------------------------------------
    def evaluate(self, k: int = 5, sample: int | None = None,
                 fetch: int = 20, rerank: bool = False) -> dict:
        """The flywheel. For each chunk, generate a question it should answer,
        then check whether that chunk lands in the top-k for its own question.

        Compare evaluate() vs evaluate(rerank=True) to PROVE the reranker helps
        before you ship it. Returns recall@k, MRR, and the failing chunk indices
        — those failures are your to-do list, not a score to admire."""
        n = len(self.chunks)
        targets = range(n) if sample is None else _spread(n, sample)
        hits, ranks, failures = 0, [], []
        for i in targets:
            q = self._question_for(self.chunks[i])
            qvec = self._embed([q])[0]
            idx = list(self._topk(qvec, max(fetch, k)))
            if rerank:
                order = self._rerank_order(q, [self.chunks[j] for j in idx])
                idx = [idx[o] for o in order]
            top = idx[:k]
            if i in top:
                hits += 1
                ranks.append(1 / (top.index(i) + 1))
            else:
                ranks.append(0.0)
                failures.append(i)
        m = len(ranks)
        return {"recall@k": hits / m, "mrr": sum(ranks) / m, "k": k,
                "evaluated": m, "rerank": rerank, "failures": failures}

    # --- persistence (the honest "vector DB") ----------------------------
    def save(self, path: str):
        """Persist to {path}.npz + {path}.json. Exact search, zero infra.
        Swap in pgvector/LanceDB here when the matrix stops fitting in RAM."""
        np.savez_compressed(f"{path}.npz", emb=self._emb)
        Path(f"{path}.json").write_text(json.dumps(
            {"chunks": self.chunks, "name": self.name,
             "description": self.description, "embed_model": self.embed_model,
             "chat_model": self.chat_model}))

    @classmethod
    def load(cls, path: str):
        meta = json.loads(Path(f"{path}.json").read_text())
        self = cls(meta["name"], meta["description"],
                   meta["embed_model"], meta["chat_model"])
        self.chunks = meta["chunks"]
        self._emb = np.load(f"{path}.npz")["emb"]
        return self

    # --- internals --------------------------------------------------------
    def _topk(self, q: np.ndarray, k: int) -> np.ndarray:
        """Pure cosine top-k over normalized rows. The one bit worth testing."""
        scores = self._emb @ q
        return np.argsort(-scores)[:k]

    def _rerank_order(self, query: str, texts: list[str]) -> list[int]:
        numbered = "\n".join(f"{i}: {t[:400]}" for i, t in enumerate(texts))
        r = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content":
                       f"Query: {query}\n\nPassages:\n{numbered}\n\nReturn the "
                       "passage numbers ordered MOST to LEAST relevant, as a "
                       "JSON array of integers. Only the array."}])
        return _parse_order(r.choices[0].message.content, len(texts))

    def _embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.embeddings.create(model=self.embed_model, input=texts)
        v = np.array([d.embedding for d in r.data], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v

    def _question_for(self, chunk: str) -> str:
        r = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content":
                       "Write one short, specific question a user would ask "
                       "that THIS passage answers. Question only:\n\n" + chunk}])
        return r.choices[0].message.content.strip()


class Router:
    """Send each query to the right collection. The course's two-level metric:
    a wrong answer is either a routing miss OR a retrieval miss — this class
    lets you separate the two. Give each RAG a clear .description and the LLM
    picks; upgrade to a fine-tuned classifier only when routing accuracy caps
    your numbers."""
    def __init__(self, rags: list[RAG]):
        self.rags = {r.name: r for r in rags}

    def route(self, query: str) -> RAG:
        menu = "\n".join(f"- {r.name}: {r.description}" for r in self.rags.values())
        any_rag = next(iter(self.rags.values()))
        r = any_rag.client.chat.completions.create(
            model=any_rag.chat_model,
            messages=[{"role": "user", "content":
                       f"Collections:\n{menu}\n\nQuery: {query}\n\nWhich single "
                       "collection best answers it? Reply with the name only."}])
        return self.rags[_pick(r.choices[0].message.content, list(self.rags))]

    def answer(self, query: str, **kw) -> str:
        return self.route(query).answer(query, **kw)


# --- free functions (naive on purpose, pure, testable) --------------------
def _chunk(text: str, size: int, overlap: int) -> list[str]:
    # ponytail: fixed-char sliding window. Good enough to start; upgrade to
    # sentence/heading-aware splitting only after evaluate() shows it hurts recall.
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)]


def _spread(n: int, k: int) -> list[int]:
    # evenly sample k of n indices without RNG (deterministic evals > flaky ones)
    if k >= n:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _parse_order(text: str, n: int) -> list[int]:
    """Extract a valid permutation of 0..n-1 from an LLM reply; missing indices
    are appended in original order so a bad reply degrades to no-op, never crash."""
    seen, out = set(), []
    for m in re.findall(r"\d+", text):
        i = int(m)
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    out += [i for i in range(n) if i not in seen]
    return out


def _pick(reply: str, names: list[str]) -> str:
    reply = reply.strip().lower()
    for name in names:
        if name.lower() in reply:
            return name
    return names[0]


# --- one runnable check, no API needed ------------------------------------
def _selfcheck():
    import tempfile
    r = RAG.__new__(RAG)                        # skip __init__/OpenAI
    r.chunks = ["cat", "dog", "fish"]
    r.name, r.description = "animals", "pets"
    r.embed_model, r.chat_model = EMBED_MODEL, CHAT_MODEL
    r._emb = np.array([[1, 0], [0, 1], [1, 1]], np.float32)
    r._emb /= np.linalg.norm(r._emb, axis=1, keepdims=True)
    q = np.array([1, 0], np.float32)            # closest to "cat"
    assert list(r._topk(q, 2)) == [0, 2], "cosine top-k order wrong"
    assert _chunk("abcdefghij", 4, 1) == ["abcd", "defg", "ghij", "j"]
    assert _spread(10, 3) == [0, 4, 9]
    assert _parse_order("[2, 0]", 3) == [2, 0, 1], "rerank parse/degrade wrong"
    assert _parse_order("garbage", 2) == [0, 1], "bad reply must no-op"
    assert _pick("use the ANIMALS one", ["docs", "animals"]) == "animals"
    assert _pick("???", ["docs", "animals"]) == "docs", "router must fall back"
    with tempfile.TemporaryDirectory() as d:    # save/load round-trip, no API
        p = f"{d}/store"
        r.save(p)
        r2 = RAG.load(p)
        assert r2.chunks == r.chunks and np.allclose(r2._emb, r._emb)
        assert r2.name == "animals"
    print("ok")


if __name__ == "__main__":
    _selfcheck()

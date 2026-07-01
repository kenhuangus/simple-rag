"""
Simple, solid RAG — distilled from jxnl/systematically-improving-rag.

The lesson of that course in one sentence: RAG is a *retrieval* system you
improve with measurement, not a magic pipeline. So the interesting method
here is not .answer() (every tutorial has that) — it's .evaluate(), which
generates synthetic questions from your own chunks and measures recall@k.
That number is your flywheel. Watch it go up when you change chunking,
add a reranker, or swap the embedding model. If you aren't measuring it,
you are guessing.

No vector DB. numpy cosine over a matrix is exact and fast to ~100k chunks.
Add a real DB (pgvector, LanceDB) only when the matrix stops fitting in RAM.

Deps: openai, numpy.  Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class RAG:
    def __init__(self, embed_model: str = EMBED_MODEL, chat_model: str = CHAT_MODEL):
        self.client = OpenAI()
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
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = self._embed([query])[0]
        idx = self._topk(q, k)
        scores = self._emb @ q
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def answer(self, query: str, k: int = 5) -> str:
        hits = self.search(query, k)
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
    def evaluate(self, k: int = 5, sample: int | None = None) -> dict:
        """The flywheel. For each chunk, generate a question it should answer,
        then check whether that chunk lands in the top-k for its own question.

        Returns recall@k, MRR, and the chunk indices that FAILED — those
        failures are your to-do list, not a score to admire."""
        n = len(self.chunks)
        targets = range(n) if sample is None else _spread(n, sample)
        hits, ranks, failures = 0, [], []
        for i in targets:
            qvec = self._embed([self._question_for(self.chunks[i])])[0]
            ranked = list(self._topk(qvec, max(k, 10)))
            if i in ranked[:k]:
                hits += 1
                ranks.append(1 / (ranked.index(i) + 1))
            else:
                ranks.append(0.0)
                failures.append(i)
        m = len(ranks)
        return {
            "recall@k": hits / m, "mrr": sum(ranks) / m, "k": k,
            "evaluated": m, "failures": failures,
        }

    # --- internals --------------------------------------------------------
    def _topk(self, q: np.ndarray, k: int) -> np.ndarray:
        """Pure cosine top-k over normalized rows. The one bit worth testing."""
        scores = self._emb @ q
        return np.argsort(-scores)[:k]

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
                       "that THIS passage answers. Question only:\n\n" + chunk}],
        )
        return r.choices[0].message.content.strip()


# --- free functions (naive on purpose) ------------------------------------
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


# --- one runnable check, no API needed ------------------------------------
def _selfcheck():
    r = RAG.__new__(RAG)                       # skip __init__/OpenAI
    r.chunks = ["cat", "dog", "fish"]
    r._emb = np.array([[1, 0], [0, 1], [1, 1]], np.float32)
    r._emb /= np.linalg.norm(r._emb, axis=1, keepdims=True)
    q = np.array([1, 0], np.float32)           # closest to "cat"
    assert list(r._topk(q, 2)) == [0, 2], "cosine top-k order wrong"
    assert _chunk("abcdefghij", 4, 1) == ["abcd", "defg", "ghij", "j"]
    assert _spread(10, 3) == [0, 4, 9]
    print("ok")


if __name__ == "__main__":
    _selfcheck()

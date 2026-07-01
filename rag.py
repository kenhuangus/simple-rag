"""
Simple, solid RAG — a pipeline you can actually improve.

The idea in one sentence: RAG is a *retrieval* system you improve with
measurement, not a magic pipeline. So the method that matters here is
.evaluate() — it generates synthetic questions from your own chunks and
measures recall@k. That number is your flywheel.

Real parts, not toys:
  - Vector DB: ChromaDB (HNSW ANN + on-disk persistence, embedded, no server).
  - Reranker: FlashRank cross-encoder (real relevance model, ~3MB, no GPU/API).
  - Embeddings + generation: OpenAI.

Deps: openai, chromadb, flashrank.  Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import re
from functools import lru_cache
import chromadb

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class RAG:
    def __init__(self, name: str = "default", description: str = "",
                 path: str = "./rag_db", embed_model: str = EMBED_MODEL,
                 chat_model: str = CHAT_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.name = _valid_name(name)
        self.embed_model, self.chat_model = embed_model, chat_model
        self.db = chromadb.PersistentClient(path=path)   # persists automatically
        self.col = self.db.get_or_create_collection(
            self.name, metadata={"description": description, "hnsw:space": "cosine"})

    @property
    def description(self) -> str:            # stored in the collection, survives reload
        return (self.col.metadata or {}).get("description", "")

    # --- ingest -----------------------------------------------------------
    def add(self, docs: list[str], chunk_chars: int = 800, overlap: int = 100):
        """Chunk, embed, and store in Chroma. Persists to disk under `path`;
        reopen with RAG(name, path=...) — no re-embedding, no save() needed."""
        chunks = [c for d in docs for c in _chunk(d, chunk_chars, overlap)]
        if not chunks:
            return self
        base = self.col.count()
        self.col.add(ids=[str(base + i) for i in range(len(chunks))],
                     embeddings=self._embed(chunks), documents=chunks)
        return self

    # --- retrieve ---------------------------------------------------------
    def search(self, query: str, k: int = 5, fetch: int = 30,
               rerank: bool = False) -> list[tuple[str, float]]:
        """ANN top-k from Chroma. With rerank=True, pull `fetch` candidates then
        reorder with the FlashRank cross-encoder (the usual next win once recall
        is your ceiling). Returns (chunk, score) — cosine sim, or rerank score."""
        res = self.col.query(query_embeddings=self._embed([query]),
                             n_results=max(fetch, k) if rerank else k)
        docs, dists = res["documents"][0], res["distances"][0]
        if not docs:
            return []
        if rerank:
            return [(docs[i], s) for i, s in _rank(query, docs)[:k]]
        return [(d, 1 - float(dist)) for d, dist in zip(docs, dists)]

    def answer(self, query: str, k: int = 5, rerank: bool = False) -> str:
        hits = self.search(query, k, rerank=rerank)
        context = "\n\n".join(f"[{i+1}] {c}" for i, (c, _) in enumerate(hits))
        r = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content":
                 "Answer using ONLY the numbered context. Cite sources like [1]. "
                 "If the context doesn't contain the answer, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])
        return r.choices[0].message.content

    # --- the point: measure retrieval ------------------------------------
    def evaluate(self, k: int = 5, sample: int | None = None,
                 fetch: int = 30, rerank: bool = False) -> dict:
        """The flywheel. For each chunk, generate a question it should answer,
        then check whether that chunk lands in the top-k for its own question.

        Compare evaluate() vs evaluate(rerank=True) to PROVE the reranker helps
        before you ship it. Returns recall@k, MRR, and the failing chunk ids —
        those failures are your to-do list, not a score to admire."""
        data = self.col.get()
        ids, docs = data["ids"], data["documents"]
        n = len(ids)
        targets = range(n) if sample is None else _spread(n, sample)
        hits, ranks, failures = 0, [], []
        for t in targets:
            q = self._question_for(docs[t])
            res = self.col.query(query_embeddings=self._embed([q]),
                                 n_results=max(fetch, k))
            rids, rdocs = res["ids"][0], res["documents"][0]
            if rerank:
                rids = [rids[i] for i, _ in _rank(q, rdocs)]
            top = rids[:k]
            if ids[t] in top:
                hits += 1
                ranks.append(1 / (top.index(ids[t]) + 1))
            else:
                ranks.append(0.0)
                failures.append(ids[t])
        m = len(ranks) or 1
        return {"recall@k": hits / m, "mrr": sum(ranks) / m, "k": k,
                "evaluated": len(ranks), "rerank": rerank, "failures": failures}

    # --- internals --------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in r.data]

    def _question_for(self, chunk: str) -> str:
        r = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content":
                       "Write one short, specific question a user would ask "
                       "that THIS passage answers. Question only:\n\n" + chunk}])
        return r.choices[0].message.content.strip()


class Router:
    """Send each query to the right collection. Think in two levels: a wrong
    answer is either a routing miss OR a retrieval miss — this class lets you
    separate the two. Give each RAG a clear .description and the LLM picks;
    upgrade to a fine-tuned classifier only when routing accuracy caps your
    numbers."""
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


# --- reranker (real cross-encoder, lazy-loaded) ---------------------------
@lru_cache(maxsize=1)
def _ranker(max_length: int = 256):
    from flashrank import Ranker            # ~3MB model, downloaded once, cached
    return Ranker(max_length=max_length)


def _rank(query: str, docs: list[str]) -> list[tuple[int, float]]:
    """Cross-encoder rerank. Returns (index_into_docs, score), best first."""
    from flashrank import RerankRequest
    req = RerankRequest(query=query,
                        passages=[{"id": i, "text": d} for i, d in enumerate(docs)])
    return [(o["id"], float(o["score"])) for o in _ranker().rerank(req)]


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


def _pick(reply: str, names: list[str]) -> str:
    reply = reply.strip().lower()
    for name in names:
        if name.lower() in reply:
            return name
    return names[0]


def _valid_name(name: str) -> str:
    # Chroma requires 3-512 chars from [a-zA-Z0-9._-], start/end alphanumeric.
    s = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-._") or "col"
    if len(s) < 3:
        s += "-col"
    return s[:512]


# --- one runnable check: real Chroma ANN, no API/network needed -----------
def _selfcheck():
    assert _chunk("abcdefghij", 4, 1) == ["abcd", "defg", "ghij", "j"]
    assert _spread(10, 3) == [0, 4, 9]
    assert _pick("use the ANIMALS one", ["docs", "animals"]) == "animals"
    assert _pick("???", ["docs", "animals"]) == "docs", "router must fall back"
    assert _valid_name("a") == "a-col" and _valid_name("hi there!") == "hi-there"

    c = chromadb.EphemeralClient()           # in-memory, exercises the real ANN
    col = c.get_or_create_collection("selfcheck", metadata={"hnsw:space": "cosine"})
    col.add(ids=["0", "1", "2"], embeddings=[[1, 0], [0, 1], [1, 1]],
            documents=["cat", "dog", "fish"])
    res = col.query(query_embeddings=[[1, 0]], n_results=2)   # closest to "cat"
    assert res["ids"][0] == ["0", "2"], f"ANN order wrong: {res['ids'][0]}"
    print("ok")


if __name__ == "__main__":
    _selfcheck()

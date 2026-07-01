"""
Simple, solid RAG — a pipeline you can actually improve.

The idea: retrieval quality is the foundation of RAG. If the right chunk never
makes it into the context, no prompt-tuning fixes the answer. So measure
retrieval first. .evaluate() does that by generating a synthetic question from
each chunk and checking whether that chunk comes back for its own question.

What .evaluate() measures — and what it does NOT:
  - It reports hit-rate@k (== recall@k when each question has one gold chunk)
    and MRR@k over *synthetic* questions. It is a fast, self-serve sanity check
    on retrieval, not a production quality score.
  - Synthetic questions are written FROM the chunk, so they share vocabulary
    with it. That makes retrieval easier than real user queries — treat the
    number as an optimistic upper bound / regression signal, not ground truth.
  - Chunk overlap causes false misses: a question from chunk i can be answered
    by a near-duplicate neighbor, which the metric counts as a miss. Inspect
    failures; some are artifacts, not real gaps.
  - It says nothing about generation faithfulness (hallucination), answer
    relevance, or multi-hop questions that need several chunks. Add a
    generation eval (e.g. an LLM-judge for groundedness) for those.
  The real win is the loop: measure on a held-out set of *real* user queries,
  read the failures, change one thing, re-measure.

Real components (not toys), but this is a minimal teaching scaffold — no
batching, retries, or auth. Harden before production.
  - Vector DB: ChromaDB (HNSW ANN + on-disk persistence, embedded, no server).
  - Reranker: FlashRank cross-encoder (real relevance model, local, no GPU/API).
  - Embeddings + generation: OpenAI.

Deps: openai, chromadb, flashrank.  Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import re
from functools import lru_cache
import chromadb

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"   # solid cross-encoder (~34MB). Swap
                                           # "ms-marco-TinyBERT-L-2-v2" for tiny/fast.


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
        is your ceiling). Returns (chunk, score): score is cosine similarity
        (0..1) for the ANN path, or the reranker's relevance score (not a
        probability, only meaningful for ordering) for the rerank path."""
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
            model=self.chat_model, temperature=0,
            messages=[
                {"role": "system", "content":
                 "Answer using ONLY the numbered context. Cite sources like [1]. "
                 "If the context doesn't contain the answer, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])
        return r.choices[0].message.content

    # --- the point: measure retrieval ------------------------------------
    def evaluate(self, k: int = 5, sample: int | None = None,
                 fetch: int = 30, rerank: bool = False) -> dict:
        """Measure retrieval with synthetic questions. For each chunk, generate a
        question from it, then check whether that chunk returns in the top-k.

        Because each question has exactly one gold chunk, `recall@k` here is
        identical to hit-rate@k, and `mrr` is truncated at k (MRR@k). These are
        proxy metrics on synthetic queries — see the module docstring for what
        they do and do NOT tell you. Use `sample=N` to spot-check a big corpus
        (N API calls); results are ~reproducible (question gen is temperature 0).

        Compare evaluate() vs evaluate(rerank=True) to check the reranker's lift
        before shipping it. `failures` are the gold chunk ids that missed —
        candidates to inspect (some are overlap artifacts, not real gaps)."""
        data = self.col.get()
        ids, docs = data["ids"], data["documents"]
        n = len(ids)
        targets = range(n) if sample is None else _spread(n, sample)
        hits, ranks, failures = 0, [], []
        for t in targets:
            q = self._question_for(docs[t])
            res = self.col.query(query_embeddings=self._embed([q]),
                                 n_results=max(fetch, k) if rerank else k)
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

    def score(self, golden: list[dict], k: int = 5, fetch: int = 30,
              rerank: bool = False) -> dict:
        """Benchmark retrieval against a GOLDEN DATASET of real, labeled queries
        — the number you actually trust (evaluate() is only a synthetic warm-up).

        golden: list of {"query": str, "relevant": str | list[str]}, where each
        `relevant` is a text snippet a correct chunk must contain. Matching by
        content (not chunk id) keeps your labels valid when you re-chunk — the
        single most useful property of a durable RAG benchmark.

        Returns recall@k (share of queries whose relevant chunk was retrieved),
        MRR@k, and the `misses` (queries that failed — your to-do list)."""
        hits, ranks, misses = 0, [], []
        for item in golden:
            rel = item["relevant"]
            rel = [rel] if isinstance(rel, str) else rel
            docs = [c for c, _ in self.search(item["query"], k=k, fetch=fetch,
                                              rerank=rerank)]
            rank = _first_hit(docs, rel)
            if rank:
                hits += 1
                ranks.append(1 / rank)
            else:
                ranks.append(0.0)
                misses.append(item["query"])
        m = len(golden) or 1
        return {"recall@k": hits / m, "mrr": sum(ranks) / m, "k": k,
                "evaluated": len(golden), "rerank": rerank, "misses": misses}

    # --- internals --------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in r.data]

    def _question_for(self, chunk: str) -> str:
        # temperature=0 so evaluate() is ~reproducible run-to-run
        r = self.client.chat.completions.create(
            model=self.chat_model, temperature=0,
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
def _ranker(model: str = RERANK_MODEL, max_length: int = 512):
    from flashrank import Ranker            # model downloaded once, then cached
    return Ranker(model_name=model, max_length=max_length)


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


def _first_hit(docs: list[str], relevant: list[str]) -> int:
    """1-based rank of the first retrieved doc containing any relevant snippet,
    or 0 if none. Content match, so it survives re-chunking. The one bit of
    benchmark logic worth a test."""
    for i, d in enumerate(docs):
        low = d.lower()
        if any(s.lower() in low for s in relevant):
            return i + 1
    return 0


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
    assert _first_hit(["a cat sat", "a dog ran"], ["DOG"]) == 2, "content match rank"
    assert _first_hit(["nope"], ["missing"]) == 0, "no hit -> 0"

    c = chromadb.EphemeralClient()           # in-memory, exercises the real ANN
    col = c.get_or_create_collection("selfcheck", metadata={"hnsw:space": "cosine"})
    col.add(ids=["0", "1", "2"], embeddings=[[1, 0], [0, 1], [1, 1]],
            documents=["cat", "dog", "fish"])
    res = col.query(query_embeddings=[[1, 0]], n_results=2)   # closest to "cat"
    assert res["ids"][0] == ["0", "2"], f"ANN order wrong: {res['ids'][0]}"
    print("ok")


if __name__ == "__main__":
    _selfcheck()

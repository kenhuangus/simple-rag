# simple-rag

A RAG pipeline in one file, distilled from
[jxnl/systematically-improving-rag](https://github.com/jxnl/systematically-improving-rag).

**📖 Tutorial site (animated data-flow + copy-paste code): [`docs/index.html`](docs/index.html)**
— enable GitHub Pages on `/docs` to publish it.

That course's real lesson, cutting through the noise:

> **RAG is a retrieval system you improve by measuring it — not a pipeline you
> build once.** The flywheel is measure → analyze → improve → iterate.

Every RAG tutorial gives you `answer()`. Almost none give you `evaluate()`.
Here the eval loop is the centerpiece, because recall@k is the only thing that
tells you whether your next change helped.

Real parts, not toys: **ChromaDB** (HNSW vector DB, embedded, auto-persists to
disk), **FlashRank** (real cross-encoder reranker, ~3MB, no GPU/API key), OpenAI
for embeddings + generation.

## Use

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

```python
from rag import RAG

r = RAG().add(open("docs.txt").read().split("\n\n"))
print(r.evaluate(k=5))      # <- start here, ALWAYS: recall@k, mrr, failures
print(r.answer("your question"))
```

## The flywheel, in code

1. **Measure** — `evaluate()` generates a question from each chunk and checks
   whether that chunk is retrieved for its own question. Gives `recall@k`, `mrr`,
   and the list of chunks that **failed**.
2. **Analyze** — read `result["failures"]`. Bad chunking? Wrong embedding? Two
   chunks that need joining?
3. **Improve** — change one thing: `chunk_chars`, embedding model, add rerank.
4. **Iterate** — re-run `evaluate()`. Moved up? Keep it. Didn't? Revert. No vibes.

## Upgrades — add each only when recall@k says so

**1 · Persist** — Chroma auto-persists to disk; just point at a `path`:
```python
RAG("docs", path="./rag_db").add(chunks)   # writes to ./rag_db
r = RAG("docs", path="./rag_db")           # next run: reopens, no re-embedding
```

**2 · Rerank** — real cross-encoder, and prove it helped before shipping:
```python
r.evaluate(k=5)                # baseline recall@k
r.evaluate(k=5, rerank=True)   # pull 30 from Chroma, FlashRank reorders; ship only if higher
r.answer("your question", rerank=True)
```

**3 · Route** across collections (separate routing misses from retrieval misses):
```python
from rag import RAG, Router
billing = RAG("billing", "invoices, refunds, payment errors").add(billing_docs)
api     = RAG("api", "endpoints, auth, rate limits").add(api_docs)
Router([billing, api]).answer("why was my card declined?")   # -> routed, then answered
```

## When to reach for what

| Symptom | Fix |
|---|---|
| recall@k low, one topic | tune `chunk_chars` / `overlap`, re-`evaluate()` |
| right chunks fetched but ranked low | `rerank=True` (FlashRank cross-encoder) |
| mixed topics, wrong docs entirely | `Router` across collections |
| millions of vectors / multi-node | swap Chroma for hosted Qdrant/pgvector (same query shape) |
| domain jargon, recall stuck | fine-tune embeddings (course, later ch.) — 6–10% typical |

Run `python rag.py` for a no-API self-check of the retrieval math.

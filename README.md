# simple-rag

A RAG pipeline in one file — small enough to read in one sitting, with real
components. It's a teaching scaffold, not a hardened production system (no
batching, retries, or auth — add those before you ship).

**📖 Tutorial site (animated data-flow + copy-paste code): [`docs/index.html`](docs/index.html)**
— enable GitHub Pages on `/docs` to publish it.

The one idea most RAG tutorials skip:

> **Retrieval quality is the foundation.** If the right chunk never reaches the
> context, no prompt-tuning fixes the answer. So measure retrieval first, then
> iterate: measure → analyze → improve → re-measure.

Every RAG tutorial gives you `answer()`. Almost none give you `evaluate()`.
Here the eval loop is the centerpiece — because retrieval is the part that
silently caps your quality, and the only way to know it improved is to measure.

Real components: **ChromaDB** (HNSW vector DB, embedded, auto-persists to disk),
**FlashRank** (cross-encoder reranker, `ms-marco-MiniLM-L-12-v2` ≈ 34 MB, runs
locally, no GPU/API key), OpenAI for embeddings + generation.

## What `evaluate()` measures — and what it does not

`evaluate()` writes a synthetic question from each chunk and checks whether that
chunk comes back for its own question. Be precise about what that gives you:

- It reports **hit-rate@k** (identical to recall@k when each question has one
  gold chunk) and **MRR@k** on *synthetic* queries. It's a fast, self-serve
  retrieval sanity check and a regression signal — **not** a production score.
- Questions are written *from* the chunk, so they share its vocabulary. Real
  user queries are harder, so treat the number as an **optimistic upper bound**.
- Chunk `overlap` causes **false misses**: a question from chunk *i* can be
  answered by a near-duplicate neighbor, counted as a miss. Inspect failures.
- It says nothing about **generation faithfulness**, answer relevance, or
  multi-hop questions. Add a generation eval (e.g. an LLM-judge for
  groundedness) for those.

The real workflow: build a held-out set of **real** user queries, measure on
that, read the failures, change one thing, re-measure.

## Use

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

```python
from rag import RAG

r = RAG().add(open("docs.txt").read().split("\n\n"))
print(r.evaluate(k=5))      # <- start here: hit-rate@k (=recall@k), MRR@k, failures
print(r.answer("your question"))
```

## The flywheel, in code

1. **Measure** — `evaluate()` generates a question from each chunk and checks
   whether that chunk is retrieved for its own question. Gives `recall@k`, `mrr`,
   and the list of chunks that **failed** (see the caveats above).
2. **Analyze** — read `result["failures"]`. Bad chunking? Wrong embedding? Two
   chunks that need joining? A false miss from overlap?
3. **Improve** — change one thing: `chunk_chars`, embedding model, add rerank.
4. **Iterate** — re-run `evaluate()`. Moved up? Keep it. Didn't? Revert.
   (Question generation runs at temperature 0, so the number is reproducible.)

## Build a real benchmark (golden dataset)

`evaluate()` is a *synthetic warm-up*. The number you actually trust comes from a
**benchmark**: a **golden dataset** of real, labeled queries, plus your metrics,
run the same way every time. (They're not the same thing — the *golden dataset*
is the labeled data; the *benchmark* is that data + metrics + a repeatable
harness.)

**How to build the golden dataset**

1. **Collect real queries.** Pull from logs, support tickets, or a domain expert.
   Cold start? Have `evaluate()` draft synthetic questions, then have a **human
   edit and keep the good ones** — LLM drafts, human owns the label.
2. **Label what "relevant" means.** For each query, record a text snippet that a
   correct chunk must contain. Labeling by *content*, not chunk id, keeps the
   dataset valid when you re-chunk — the single most useful property here.
3. **Cover the query types** you actually get: easy and hard, single-hop and
   multi-hop, plus a few unanswerable ones (the right answer is "I don't know").
4. **Size it** for a stable signal — ~50–100 queries to start, more for
   confidence. **Freeze and version it** so scores compare across runs; grow it
   by adding every real production failure.
5. **Pick metrics.** Retrieval: recall@k, MRR/nDCG. Generation (next layer):
   faithfulness/groundedness and answer correctness via an LLM-judge.

**Store it as JSONL** (`golden.jsonl`):

```json
{"query": "what city is the capital of France?", "relevant": "capital of France"}
{"query": "which organelle makes energy in cells?", "relevant": "powerhouse of the cell"}
{"query": "how high is Everest?", "relevant": "8,849 meters"}
```

**Run the benchmark:**

```python
import json
golden = [json.loads(l) for l in open("golden.jsonl")]

print(r.score(golden, k=5))                 # recall@k + MRR@k over REAL queries
print(r.score(golden, k=5, rerank=True))    # does the reranker help on YOUR data?
# result["misses"] = the queries that failed — your to-do list
```

`score()` counts a hit when any retrieved chunk contains a labeled snippet, so
your labels survive re-chunking. This is the number to report and track over
time — not the synthetic one.

## Upgrades — add each only when the metric says so

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
| millions of vectors / multi-node | move to a hosted store (Qdrant, pgvector, Pinecone) — same idea (embed query → nearest neighbors), different API |
| domain jargon, recall stuck | try a bigger embedding model (`text-embedding-3-large`); to *fine-tune* embeddings, switch to an open model (e.g. BGE) — OpenAI embeddings aren't fine-tunable |

Run `python rag.py` for a no-API self-check of the retrieval math (real Chroma
ANN, offline).

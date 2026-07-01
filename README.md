# simple-rag

A RAG pipeline in one file, distilled from
[jxnl/systematically-improving-rag](https://github.com/jxnl/systematically-improving-rag).

That course's real lesson, cutting through the noise:

> **RAG is a retrieval system you improve by measuring it — not a pipeline you
> build once.** The flywheel is measure → analyze → improve → iterate.

Every RAG tutorial gives you `answer()`. Almost none give you `evaluate()`.
This repo makes the eval loop the centerpiece, because that number — recall@k —
is the only thing that tells you whether your next change helped.

## Use

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python -c "
from rag import RAG
r = RAG().add(open('yourdocs.txt').read().splitlines())
print(r.evaluate(k=5))      # <- start here, ALWAYS
print(r.answer('your question'))
"
```

## The flywheel, in code

1. **Measure** — `evaluate()` generates a synthetic question from each chunk and
   checks whether that chunk is retrieved for its own question. Gives you
   `recall@k`, `mrr`, and the list of chunks that *failed*.
2. **Analyze** — read the `failures`. Bad chunking? Wrong embedding for the
   domain? Questions that need two chunks joined?
3. **Improve** — change one thing: `chunk_chars`, the embedding model, add a
   reranker over `search()` results.
4. **Iterate** — re-run `evaluate()`. If recall@k didn't move, you learned
   nothing was wrong there. If it moved, you have proof. No vibes.

## Deliberately skipped (add when the number says to)

- **Vector DB** — numpy cosine is exact to ~100k chunks. Add pgvector/LanceDB
  when the matrix stops fitting in RAM.
- **Reranker** — a cross-encoder over `search(k=20)` is the usual next win.
  Add it after `evaluate()` shows retrieval recall is your ceiling.
- **Query routing / fine-tuned embeddings** — the course's later chapters.
  They're improvements you *justify with the metric*, not defaults.

Run `python rag.py` for a no-API self-check of the retrieval math.

# Ask My Docs

A production-shaped RAG system over a domain document set: **hybrid retrieval**
(BM25 + dense vectors fused with reciprocal rank fusion), **cross-encoder
reranking**, **enforced citations** that can veto an answer, and a
**CI-gated evaluation pipeline** with Ragas-style metrics.

It ships as two implementations of one specification:

| | what it is | how you run it |
| --- | --- | --- |
| **`web/`** | The whole pipeline in the browser. No build step, no server, no API key. | Double-click `web/index.html` |
| **`service/`** | The Python service: same algorithms, FastAPI, pluggable OpenAI / Cohere / sentence-transformers providers, pytest suite. | `pip install -e "service[api]"` |

[`ALGORITHMS.md`](ALGORITHMS.md) is the contract both obey — every constant,
formula, and enforcement rule, plus the measurements behind the ones that were
tuned. If the two implementations ever disagree, that file is right and the code
is wrong.

---

## Start here (nothing to install)

Open **`web/index.html`**. It indexes eight internal handbooks (57 chunks) in
about 30 ms and answers questions with citations you can click back to the
source passage.

Try, in order:

1. *"What are the default API rate limits?"* — a list answer with two figures,
   each traced to the line it came from.
2. *"Who approves a 60,000 EUR vendor contract?"* — the corpus never mentions
   60,000; the answer comes from the `25,001 to 100,000 EUR` band.
3. *"How much is the CEO's annual bonus?"* — a refusal, with the passages it
   inspected still shown so you can see it searched properly.
4. Turn **cross-encoder reranking off** and ask #3 again. It now answers, wrongly
   and with citations. That is the single most useful thing in this repo.

Then open **`web/evaluate.html`** and press **Run evaluation** to execute the
same 33-case gate that CI runs, followed by **Run ablation matrix** to see what
each retrieval stage is actually worth.

If you want a real `http://` origin (devtools, or talking to the Python
service), this repo includes a dependency-free static server for machines with
no Python or Node:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
# http://localhost:8123/index.html
```

---

## The pipeline

```
question
   │
   ├── BM25 (Okapi, k1=1.2, b=0.75) ────────► top 20    ← raw query only
   │                                                       (lexical precision)
   └── dense vectors (cosine) ──────────────► top 20    ← alias-expanded
   │
   ├── reciprocal rank fusion (k=60) ───────► top 12
   │
   ├── cross-encoder rerank ────────────────► top 5 with score ≥ 0.55
   │        │
   │        └── nothing clears the floor? ──► REFUSE (generator never runs)
   │
   ├── extractive synthesis ────────────────► ≤ 3 claims, each bound to a chunk
   │
   └── citation enforcement (CE1–CE4) ──────► answer + citations
            │                                 or REFUSE
            └── one repair retry
```

### Retrieval

BM25 and dense search run independently and are merged by RRF, so neither needs
its score calibrated against the other. Retrieval mode is switchable — `bm25`,
`vector`, `hybrid` — which is what the ablation compares.

The alias-expanded query (`spec/aliases.json`) goes to the dense branch, the
reranker, and the generator; **BM25 alone keeps the raw words**, so expansion can
never dilute lexical precision. Expanding the reranker mattered more than
expanding retrieval: with expansion on the dense branch only, the worst
answerable question scored 0.385 against a 0.55 floor and was refused, because
the corpus says "paid annual leave" and the user typed "vacation". Expanding the
reranker's view too moved that to 0.852 while the highest-scoring unanswerable
question stayed at 0.517. Across the whole golden set the worst answerable
question scores 0.613, so the gap the floor has to sit in is `0.517 … 0.613` —
narrow, which is why the `near-miss` suite exists to keep expansion from buying
that recall at the cost of false positives.

The default embedder needs no model download: a deterministic signed random
projection of an idf-weighted bag of words and bigrams. It is a TF-IDF sketch,
not a learned semantic model, and the README says so because the difference
matters — see [Limitations](#limitations). Set `OPENAI_API_KEY` and the service
uses real embeddings instead.

### Reranking

A cross-encoder scores the (query, passage) pair *jointly* instead of comparing
two independent vectors, which is why it fixes precision that retrieval cannot.
The service uses Cohere `rerank-v3.5` or a local
`cross-encoder/ms-marco-MiniLM-L-6-v2` when available, and otherwise a
deterministic `lexical-ce` fallback that scores the pair on eight explicit
features (idf-weighted coverage, proximity, phrase match, numeric density,
numeric-range bracketing, heading match…) through a logistic function.

Its score does double duty: ranking, and the **refusal decision**. A passage
must clear `RELEVANCE_FLOOR = 0.55`, a value calibrated against the golden set
rather than guessed (§5.2 of `ALGORITHMS.md`).

### Citation enforcement

The generator is not trusted, whether it is a hosted LLM or the local
extractive path. Every claim must pass:

| rule | check |
| --- | --- |
| **CE1** | its `[n]` marker resolves to a passage actually in the context |
| **CE2** | every declarative sentence carries a marker |
| **CE3a** | idf-weighted containment in the cited passage ≥ 0.62 |
| **CE3b** | **every number in the claim appears in the cited passage** — no threshold, one invented figure fails the claim |
| **CE4** | at least one claim survives, or the whole answer becomes a refusal |

Failures trigger one repair pass, then get dropped. The API guarantees the
answer contains **no digit absent from its cited quotes** — the check that most
directly catches the failure mode people actually fear, a fluent answer with a
fabricated number and a citation that looks legitimate.

---

## Evaluation and the CI gate

`eval/goldens.json` holds 33 cases — 23 answerable, 10 that must be refused —
in three suites:

- **full-sentence** (21, untagged) — the well-formed case: 18 answerable
  questions plus 3 that are plainly outside the corpus (`"What is our stock
  price?"`), which any sane system refuses,
- **short-form** (5) — `"Rate limits?"`, `"GPU quota?"`, the terse phrasings
  people actually type, which a system tuned only on verbose questions refuses,
- **near-miss** (7, all unanswerable) — questions that share vocabulary with a
  real section but have no answer there. Each one is a regression guard for a
  failure that was measured and fixed: `"Which airline should I book for
  international travel?"` is in the set because *travel* and *books* both appear
  in the learning-budget section, and an earlier build answered it confidently.

Metrics are deterministic re-implementations of the Ragas definitions —
`faithfulness`, `answer_relevancy`, `context_precision` (average precision),
`context_recall` — plus `answer_correctness`, `citation_validity`, and
`refusal_accuracy`. No LLM judge, so the gate is reproducible, free, and fast
enough to run on every pull request. `run_eval.py --judge llm` swaps in a real
judge when a key is available; the deterministic numbers are the ones that gate.

Measured baseline (hybrid + `lexical-ce`, all 33 cases):

| metric | measured | gate floor |
| --- | --- | --- |
| faithfulness | 1.000 | 0.90 |
| context_precision | 1.000 | 0.85 |
| context_recall | 1.000 | 0.90 |
| answer_relevancy | 0.585 | 0.50 |
| answer_correctness | 1.000 | 0.85 |
| citation_validity | 1.000 | 1.00 |
| refusal_accuracy | 1.000 | 1.00 |

`answer_relevancy` is 0.585 by construction: answers are verbatim policy
sentences, and a TF-IDF sketch of a document sentence is only ever moderately
similar to a question in a user's own words. It is a tripwire, not a target.

```bash
python eval/run_eval.py                 # writes eval/report.md + report.json, exits 1 on regression
python eval/run_eval.py --ablate        # the retrieval-mode comparison
```

CI runs ruff, pytest, then the gate, uploads the report as an artifact, and
comments the summary table on the pull request. A second job installs the
package with **no optional extras** to prove the zero-dependency path works.

### What the ablation actually showed

| configuration | ctx precision | refusal accuracy | gate |
| --- | --- | --- | --- |
| bm25 only, no rerank | 0.951 | 0.100 | FAIL |
| vector only, no rerank | 1.000 | 0.300 | FAIL |
| hybrid (RRF), no rerank | 0.980 | 0.100 | FAIL |
| hybrid (RRF) + rerank | 1.000 | 1.000 | pass |

Two findings worth stating plainly, because neither is the usual pitch:

- **The cross-encoder earns its place through refusal, not precision.** Remove
  it and 9 of 10 unanswerable questions get answered confidently. Context
  precision — the thing reranking is normally sold on — moves only 0.980 → 1.000.
- **Fusion is a robustness win here, not a precision win.** With reranking on,
  all three modes tie, and dense-only already scores 1.000 without it. Fusion
  removes the dependence on which branch happens to suit a query; on 57 chunks
  of tight policy prose that is worth less than it would be on a big, noisy
  corpus. Claiming otherwise would not be supported by the numbers.

---

## Providers

Everything works with **no API keys**. Keys upgrade components in place; the
enforcement and evaluation layers are unchanged.

| stage | zero-key default | with keys |
| --- | --- | --- |
| embeddings | `hashing` (deterministic TF-IDF sketch) | `OPENAI_API_KEY` → `text-embedding-3-small`; or install `sentence-transformers` → `all-MiniLM-L6-v2` |
| reranking | `lexical-ce` (explicit pair features) | `COHERE_API_KEY` → `rerank-v3.5`; or `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| generation | `extractive` (quotes real sentences) | `OPENAI_API_KEY` → LLM at `temperature=0`, then the same CE1–CE4 gauntlet |
| eval judge | deterministic | `--judge llm` |

The browser build is extractive-only and calls nothing over the network by
design.

---

## Layout

```
ALGORITHMS.md          the specification both implementations obey
spec/                  stopwords + query-expansion aliases (canonical)
data/corpus/           8 markdown documents with YAML front matter
eval/
  goldens.json         33 golden cases in 3 suites
  thresholds.json      the CI gate
  metrics.py           deterministic Ragas-style metrics
  run_eval.py          runner, report writer, exit-code gate
service/
  askmydocs/           text, chunking, embeddings, retrieval, rerank,
                       generate, citations, pipeline, api, cli
  tests/               pytest suite incl. a fabricated-number test
web/
  index.html           ask + retrieval funnel inspector
  evaluate.html        the gate and ablation dashboard
  js/                  the same pipeline, one file per stage
scripts/
  serve.ps1            dependency-free static server (Windows)
  build_web_corpus.py  regenerates the browser's inlined corpus
.github/workflows/     CI: lint, tests, eval gate, PR comment
```

Corpus and goldens are canonical in `data/` and `eval/`. The browser cannot
`fetch()` from a `file://` origin, so `web/js/corpus.data.js` and
`web/js/goldens.data.js` are generated mirrors — run
`python scripts/build_web_corpus.py` after editing the originals.

---

## Limitations

Worth knowing before you point this at your own documents.

- **The zero-key retrieval stack is lexical, not semantic.** The hashing
  embedder and `lexical-ce` reranker match words, not meaning. The alias table
  papers over the gap for known vocabulary — the corpus says "paid annual
  leave", users say "vacation" — but a paraphrase nobody anticipated will be
  refused rather than answered. Real embeddings and a real cross-encoder fix
  this; that is what the provider slots are for.
- **`RELEVANCE_FLOOR` is corpus-specific.** 0.55 sits in a measured gap on
  *this* corpus. Recalibrate on your own goldens before trusting the refusal
  behaviour, and expect the gap to narrow as the corpus grows.
- **Extractive answers are quotes, not prose.** They are always grounded and
  never fluent. Add an LLM generator for readability; it still has to pass
  CE1–CE4.
- **33 golden cases is a starting point.** Enough to catch the regressions
  documented in `ALGORITHMS.md`, not enough to characterise a production
  system. Grow the set every time you find a failure — that is the workflow the
  three suites are shaped around.
- **The browser build keeps user-uploaded documents in `localStorage`**, capped
  at 1.5 MB, and re-embeds the whole corpus on every change. It is a demo and an
  inspector, not the deployment target.

---

## Reference material

The design follows the sources this project was built against:

- **[LangChain RAG tutorial](https://docs.langchain.com/oss/python/langchain/rag)** —
  the indexing → retrieval → generation structure, and the discipline of
  passing retrieved documents to generation as an explicit, inspectable list.
  Implemented directly rather than through the framework, so the retrieval
  funnel stays visible end to end and the default install has no dependencies;
  the stage boundaries are the same ones the tutorial draws.
- **[Cohere Rerank](https://docs.cohere.com/docs/rerank-guide)** — why a
  cross-encoder that reads the query and passage together beats comparing two
  independently-computed embeddings. `rerank-v3.5` is a first-class provider;
  `lexical-ce` is the offline stand-in, built to the same interface.
- **[Ragas](https://docs.ragas.io/)** — the metric definitions. Faithfulness,
  answer relevance, context precision, and context recall are implemented to
  Ragas's semantics but with deterministic scorers instead of an LLM judge, so
  they can gate CI on every commit at zero cost.

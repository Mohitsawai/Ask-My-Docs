# Ask My Docs — Algorithm Specification

This document is the **single source of truth** for the retrieval and evaluation
maths. Two implementations must agree with it:

| Implementation | Path | Runtime |
| --- | --- | --- |
| Production service | `service/` | Python 3.11+ stdlib, FastAPI optional |
| Zero-install demo | `web/` | Browser, classic scripts, no build step |

The browser build deliberately uses classic `<script>` tags rather than ES
modules: modules are blocked by CORS under the `file://` origin, and
double-clicking `web/index.html` has to work.

Any change to a constant here must be applied to both sides in the same change.

---

## 0. Constants

```
CHUNK_TARGET_WORDS      = 180
CHUNK_OVERLAP_WORDS     = 40
CHUNK_MIN_WORDS         = 25

EMBED_DIM               = 384
EMBED_HASHES            = 3          # signed hashes per feature
EMBED_BIGRAM_WEIGHT     = 0.55

BM25_K1                 = 1.2
BM25_B                  = 0.75

BRANCH_TOP_K            = 20         # candidates taken from each retriever
RRF_K                   = 60
FUSED_TOP_K             = 12         # candidates handed to the reranker
FINAL_TOP_K             = 5          # passages handed to the generator
RELEVANCE_FLOOR         = 0.55       # min rerank score to enter the context
CLAIM_PASSAGE_RATIO     = 0.75       # passage must be this close to the best
                                     #   one before it may source a claim
SENTENCE_FLOOR          = 0.15       # absolute min selection score
SENTENCE_RATIO          = 0.40       # relative to the best sentence
COVERAGE_BONUS          = 0.60       # reward for covering new query terms
DEDUPE_SIM              = 0.90       # cosine above which two claims duplicate
MAX_CLAIMS              = 3
GROUNDING_FLOOR         = 0.62       # idf-weighted containment for CE3
```

`RELEVANCE_FLOOR` is the only constant here that is *calibrated* rather than
chosen, because it alone decides whether the system answers or refuses. See
§5.2 for the measurement it comes from and how to recalibrate it for a
different corpus.

---

## 1. Text normalisation and tokenisation

Identical on both sides. Order matters.

1. Lowercase (`toLowerCase` / `str.lower`).
2. Replace unicode dashes `– —` with `-`, curly quotes with ASCII quotes.
3. Strip thousands separators inside numbers: `1,500 -> 1500` (regex
   `(\d),(\d{3})\b` applied repeatedly).
4. Replace every character that is not `[a-z0-9%.\-]` with a space.
5. Collapse whitespace, split on space.
6. Trim leading/trailing `.` and `-` from each token, but keep an internal
   `.` (so `1.2`, `v2.1`, `sev-1`, `0.90`, `50%` survive intact).
7. Drop any token that is now empty, or that is a bare `-`, `.`, or `%`.

Steps 6 and 7 are in that order deliberately: trimming first is what turns
`"..."` into an empty token, so a single drop pass afterwards catches both the
tokens that started out as bare punctuation and the ones that became empty.
Dropping first would leave `"..."` in the list as an empty string. The bare `%`
case exists because step 4 turns `50 % of budget` into a standalone `%`.

`content_tokens(text)` additionally removes stopwords (`spec/stopwords.json`,
129 words, mirrored in `service/askmydocs/text.py` and `web/js/spec.data.js`)
and applies light suffix folding. Exactly one rule fires, tested in this order:

1. `ies` → `y`, when length > 4 (`policies` → `policy`)
2. `ing` → ``,  when length > 6 (`reviewing` → `review`)
3. `ed`  → ``,  when length > 5 (`reviewed` → `review`)
4. `s`   → ``,  when length > 3 and the token does not end in `ss`, `us`,
   or `is` (`minutes` → `minute`, `days` → `day`)

There is deliberately **no bare `es` → `` rule**: it would fold `minutes` to
`minut` while leaving `minute` alone, so the singular and plural of the most
common unit in this corpus would stop matching each other.

A token is never folded if it contains a digit, `%`, `.`, or `-`, so `sev-1`,
`0.90`, `50%`, `tier-1`, and `v2.1` survive.

### 1.1 Which idf?

Two idf functions exist and must not be confused:

- **`idf_bm25(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`** — used by BM25 (§4.1),
  by the reranker's `f_idf` and proximity weighting (§5.1), and by the CE3a
  grounding check (§7).
- **`idf_embed(f) = ln(1 + (N + 1) / (df + 1))`** — used only to weight
  features inside the hashing embedder (§3.1).

`N` and `df` are always counted over chunks, not documents.

---

## 2. Chunking

Markdown-aware, deterministic:

0. **Normalise line endings to `\n` before computing any offset.** `char_start`
   and `char_end` index into the document text, and both implementations rebuild
   those offsets a line at a time; a two-character CRLF separator makes every
   subsequent chunk slide one byte earlier per preceding line. Normalise once at
   ingest rather than tracking the separator width. The repository corpus is
   `\n`-only, so this only bites on pasted or uploaded input — which the browser
   corpus panel accepts.
1. Parse YAML front matter into document metadata
   (`title`, `source_id`, `owner`, `version`, `effective`).
2. Split the body on `##` headings. The `#` title block becomes the preamble
   section with heading = document title.

   The `##` line itself is **not** part of the section body it introduces: the
   heading text is captured as the section's `heading` and the body starts at
   the next line. This is load-bearing rather than cosmetic — it shifts every
   `char_start`/`char_end` in every citation and changes the word count that
   step 3 windows on, so an implementation that leaves the heading inline will
   disagree with this document on every offset it emits. The preamble is the
   exception: its `# Title` line is its only content, so it is kept.
3. For each section, if its word count exceeds `CHUNK_TARGET_WORDS`, cut it
   into windows of `CHUNK_TARGET_WORDS` words with `CHUNK_OVERLAP_WORDS`
   overlap; otherwise keep it whole.
4. Discard windows shorter than `CHUNK_MIN_WORDS` unless the section produced
   only one window.
5. Every chunk carries: `chunk_id` (`{source_id}#{section_index}.{window_index}`),
   `source_id`, `doc_title`, `heading`, `text`, `char_start`, `char_end`,
   `word_count`.

The chunk's **indexed text** is `heading + "\n" + text` — the heading is
repeated into the body so that both retrievers can match on it.

---

## 3. Embeddings

The embedding provider is pluggable. Resolution order at startup:

1. `openai` — used when `OPENAI_API_KEY` is set (`text-embedding-3-small`, 1536d).
2. `sentence-transformers` — used when the package is installed
   (`all-MiniLM-L6-v2`, 384d).
3. `hashing` — **the default, zero-dependency, deterministic fallback.** This is
   the only provider the browser implements, and the only one whose numbers are
   pinned by this spec.

### 3.1 Hashing embedder (`hashing`)

A signed random projection of an idf-weighted bag of words and bigrams — i.e. a
TF-IDF sketch, not a learned semantic model. It is exact, reproducible, and
needs no network.

For text `t`:

1. `toks = content_tokens(t)`.
2. Features = unigrams (weight `1.0`) and adjacent bigrams `a_b`
   (weight `EMBED_BIGRAM_WEIGHT`).
3. Feature weight `w(f) = (1 + ln(tf(f))) * idf(f) * feature_weight`, where
   `idf(f) = ln(1 + (N + 1) / (df(f) + 1))` over the chunk corpus.
   Unseen features use `df = 0`.
4. For `s` in `0..EMBED_HASHES-1`: `h = fnv1a32(f + "#" + s)`;
   `bucket = h % EMBED_DIM`; `sign = (h >> 31) & 1 ? -1 : +1`;
   `v[bucket] += sign * w(f) * rsqrt`, where `rsqrt = 1 / sqrt(EMBED_HASHES)` is
   computed **once** and multiplied in. Mathematically this is division by
   `sqrt(EMBED_HASHES)`; in floating point it is not always the same last bit,
   and this is the largest accumulation in the pipeline. Multiply by the
   precomputed reciprocal so both implementations round identically.
5. L2-normalise `v`. An all-zero vector stays all-zero and scores 0.

`fnv1a32` is the standard 32-bit FNV-1a over UTF-8 bytes with offset basis
`0x811c9dc5` and prime `0x01000193`, kept in unsigned 32-bit arithmetic.

Similarity is cosine. Because step 5 leaves every vector at unit length, the
browser reference takes a **plain dot product** and skips the denominator,
while the Python service divides by `sqrt(norm_a * norm_b)` so that `cosine`
stays correct for a hosted embedder that does not normalise. The two agree to
about `1e-16`, which is below every threshold in this document except one: the
`DEDUPE_SIM` comparison in §6.1 step 5 is an inequality, so two candidate
sentences sitting within a rounding error of `0.90` similarity could in
principle be deduplicated by one implementation and not the other. Accepted
knowingly — at that separation either verdict is defensible — but do not
introduce a second inequality against a raw similarity without revisiting it.

### 3.2 Query expansion (vector branch, reranker, generator)

Before embedding a **query**, the alias table in `spec/aliases.json` is applied:
each matched alias appends its expansion to the query text (the original text is
kept). A trigger matches on whole-word/phrase containment in the normalised
query.

Expansion applies to the **vector branch, the reranker, and the generator's
sentence scorer** — never to BM25, which keeps the raw query so its lexical
precision is untouched.

Extending expansion past the vector branch was originally rejected on the
theory that padding the query dilutes the reranker's coverage features (they
divide by `|Q|`). Measurement said otherwise, because the aliases are curated
to use the corpus's own vocabulary, so the added terms hit the correct passage
and miss the wrong ones. On a 22-question probe set the worst answerable
question rose from **0.385 to 0.852** while every unanswerable one stayed at or
below **0.517**:

| query | without expansion | with expansion |
| --- | --- | --- |
| "How many vacation days?" (answerable) | 0.385 | 0.911 |
| "How long are backups kept?" (answerable) | 0.533 | 0.888 |
| "Rate limits?" (answerable) | 0.778 | 0.947 |
| "Which airline should I book…?" (unanswerable) | 0.517 | 0.517 |

The corpus never says "vacation"; it says "paid annual leave". Without
expansion a user who types their own words instead of the document's words gets
refused, which is the single most common complaint about internal-docs search.

An alias trigger is therefore a precision-sensitive asset, not a free win. A
bare `holiday` trigger in the paid-leave entry fired on *"What is the company
holiday party budget?"* and lifted an unanswerable question to **0.894** — a
confident, well-cited, wrong answer. The trigger was removed and the question
is now golden case `g28`.

---

## 4. Retrievers

### 4.1 BM25 (lexical)

Okapi BM25 over `content_tokens(indexed_text)`:

```
idf(q)   = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
score(D) = Σ_q idf(q) * f(q,D) * (k1 + 1)
                       / (f(q,D) + k1 * (1 - b + b * |D| / avgdl))
```

with `k1 = BM25_K1`, `b = BM25_B`. Query terms are `content_tokens(query)`
with duplicates kept (a repeated term contributes twice).

### 4.2 Vector (dense)

Cosine between the expanded-query embedding and each chunk embedding.

### 4.3 Fusion — Reciprocal Rank Fusion

Each branch returns its top `BRANCH_TOP_K` with 1-based ranks. Chunks scoring
exactly 0 in a branch are excluded from that branch.

```
rrf(D) = Σ_branches 1 / (RRF_K + rank_branch(D))
```

Ties break on `(higher bm25 rank, then chunk_id)` for determinism. The top
`FUSED_TOP_K` fused chunks go to the reranker.

Retrieval mode is configurable — `bm25`, `vector`, `hybrid` (default) — which
is what the ablation table in the UI and in `eval/report.md` compares.

---

## 5. Cross-encoder reranking

Provider resolution order:

1. `cohere` — `rerank-v3.5` when `COHERE_API_KEY` is set.
2. `cross-encoder` — `cross-encoder/ms-marco-MiniLM-L-6-v2` when
   sentence-transformers is installed.
3. `lexical-ce` — **the default deterministic fallback**, specified below and
   implemented identically in the browser.

Providers 1 and 2 return their own scores, which are min-max normalised into
`[0,1]` per query before the `RELEVANCE_FLOOR` is applied.

### 5.1 `lexical-ce` feature scoring

A cross-encoder looks at the pair jointly; this fallback does the same with
explicit pair features rather than learned attention. For query `q` and
passage `p`, let `Q = content_tokens(q)` (unique), `P = content_tokens(p)`:

| Feature | Definition |
| --- | --- |
| `f_cov` | `|Q ∩ P| / |Q|` |
| `f_idf` | `Σ_{t ∈ Q ∩ P} idf(t) / Σ_{t ∈ Q} idf(t)` |
| `f_prox` | proximity, see below |
| `f_phrase` | `matched_query_bigrams / max(1, |Q| - 1)` |
| `f_num` | numeric density, see below |
| `f_range` | numeric bracket match, see below |
| `f_head` | `|Q ∩ content_tokens(heading)| / |Q|` |
| `f_short` | `1` if `p` has fewer than `CHUNK_MIN_WORDS` words, else `0` |

**Proximity** `f_prox`: find the shortest span in `P` (by token index) that
contains the maximum number of distinct terms of `Q`. With `m` distinct terms
matched in a span of `L` tokens, `f_prox = (m / |Q|) * 1 / (1 + ln(1 + L / max(1, m)))`.
If `m < 2`, `f_prox = 0`.

**Numeric density** `f_num`: `0` unless the query asks a quantity — one of
`how many`, `how much`, `how long`, `how often`, `how quickly`, `how soon`,
`what is the`, `what are the`, `limit`, `rate`, `budget`, `quota`,
`retention`, `sla` — and then `min(1, count_of_numbers(p) / 3)`.

This started as "1 if the passage contains a digit". In a policy corpus almost
every passage contains a digit, so the binary form handed the same `+0.70` to
everything and carried no information at all; worse, it was enough to push
unanswerable questions over the relevance floor. Counting figures instead makes
it discriminative: for "what are the canary traffic steps", the sentence listing
`1% / 5% / 25% / 100%` scores 1.0 while the prose around it scores 0–0.33.

**Numeric bracket** `f_range`: `1` when a number in the query falls inside a
numeric range stated on a *single line* of the passage, else `0`. Only values
`>= 100` on either side are considered, so section numbers, `SEV-1` and
`5 minutes` cannot form spurious brackets.

Threshold tables are everywhere in policy documents, and no lexical feature can
connect *"Who approves a 60,000 EUR vendor contract?"* to the band
`25,001 to 100,000 EUR: department head plus Finance Operations`, which never
mentions 60,000. `f_range` is what makes that lookup work.

```
logit = -1.85
      + 2.20 * f_idf
      + 1.40 * f_cov
      + 1.00 * f_prox
      + 0.90 * f_phrase
      + 0.70 * f_num
      + 1.00 * f_range
      + 0.80 * f_head
      - 0.70 * f_short
score = 1 / (1 + exp(-logit))
```

Passages are sorted by score descending; the top `FINAL_TOP_K` with
`score >= RELEVANCE_FLOOR` become the answer context. If none qualify, the
pipeline short-circuits to a refusal and never calls the generator.

### 5.2 Calibrating `RELEVANCE_FLOOR`

This one number is the refusal policy, so it is measured rather than guessed.
Over the full golden set, with expansion enabled, the best passage for a
question the corpus **can** answer never scores below **0.613**, and the best
passage for a question it **cannot** answer never exceeds **0.517**. `0.55`
sits inside that gap, nearer the unanswerable ceiling so that genuine questions
are not lost.

Do not confuse this with the **0.852** in §3.2. That figure is the worst
answerable question in the 22-question expansion probe, and it is higher only
because that probe is a subset. The gap that the floor actually has to survive
is the one measured over every case, which is the narrower `0.517 … 0.613`
quoted here — about six points of headroom on each side, not thirty.

Two things are worth knowing before reusing this number:

1. **It is corpus-specific.** Recalibrate on your own goldens — dump the top
   passage score per question, then pick a value inside the gap. The floor
   slider in the UI and `--floor` on the CLI exist for this.
2. **An earlier attempt to gate on evidence instead of score failed**, and the
   record is useful. Requiring `f_idf >= 0.45` looked more principled but
   refused three answerable questions, while `0.30` let *"Which airline should
   I book for international travel?"* through at `f_idf = 0.435` — it matched
   the learning-budget section on `travel` and on `books`. The evidence share
   simply does not separate the two classes; the calibrated score does.

Without a reranker (`--no-rerank`) there is no calibrated score to threshold,
so nothing can be refused. That, and not context precision, is what the
ablation in §8.1 shows the cross-encoder buying.

---

## 6. Answer synthesis

### 6.1 Extractive generator (default, no API key)

1. **Restrict the sources.** Only passages with
   `rerank_score >= CLAIM_PASSAGE_RATIO * best_passage_score` may contribute a
   claim. A passage that merely cleared `RELEVANCE_FLOOR` is relevant enough to
   show but not to quote: without this rule, *"How long do we have to complete
   a customer deletion request?"* answered correctly from the retention policy
   and then appended a true, well-cited, entirely irrelevant sentence about API
   rate limits.
2. **Split into sentences** on `(?<=[.!?])\s+` plus markdown list-item and
   table-row boundaries. Table rows are rendered as `column | column | column`
   so numeric tables stay quotable, and are kept atomic — splitting a rendered
   row on its internal periods destroys that shape. Heading lines are
   **dropped** rather than emitted as sentences: a heading asserts nothing, so
   it must never become a claim, and §2 has already repeated it into both the
   indexed text and the step 4 scorer.
3. **Drop stubs**: a sentence ending in `:` or carrying fewer than 5 content
   tokens. `Default limits per API token:` is grammatically a sentence, scores
   well on coverage, and asserts nothing.
4. **Score** each sentence with the §5.1 pair scorer, against
   `heading + "\n" + sentence` rather than the bare sentence, then multiply by
   `0.6 + 0.4 * passage_rerank_score`.

   The heading matters more than it looks. A list item such as
   `Read endpoints: **600 requests per minute**` inherits its subject from the
   section title and repeats none of the question's words, so scoring it alone
   gives coverage 0 — and the answer to "what are the default API rate limits"
   came back quoting the HTTP 429 error text instead of the limits.
5. **Select greedily on relevance plus marginal query coverage.** With
   `covered` = query terms already covered by selected claims,

   ```
   gain(c) = c.score + COVERAGE_BONUS * (Σ idf of c's query terms not in covered)
                                        / Σ idf of all query terms
   ```

   Take the highest `gain` each round, stop at `MAX_CLAIMS` or when the best
   `gain < max(SENTENCE_FLOOR, SENTENCE_RATIO * best_sentence_score)`. Skip a
   candidate whose embedding is within `DEDUPE_SIM` cosine of an already
   selected claim, or whose normalised text is identical.

   A skipped candidate is **removed from the pool, not reconsidered** in a
   later round. It is drawn from the pool before the duplication test, so once
   rejected it cannot come back — otherwise the loop could stall, re-picking
   the same highest-gain duplicate every round until `MAX_CLAIMS` iterations
   were spent selecting nothing.

   Two simpler designs were measured and rejected:

   - **Top-k by score** answers only the loudest half of a two-part question.
     *"How many days of paid annual leave … and how much can be carried over?"*
     returned the 25-days sentence at 0.649 and dropped the 5-carried-over
     sentence at 0.223.
   - **MMR** (`λ * score - (1-λ) * max_similarity`) is worse. When the answer
     is a list of parallel facts, the second fact is the *most* similar
     remaining sentence, so diversity evicts it in favour of something
     off-topic: asking for read and write rate limits returned the write limit
     and then an unrelated sentence, never the read limit.

   Rewarding uncovered query terms fixes both, because that is what a
   multi-part question actually needs. The similarity check survives only to
   drop true duplicates, which chunk overlap guarantees will exist.
6. Each selected sentence becomes a **claim** bound to the `chunk_id` it came
   from. Claims are re-ordered by source passage rank, then by position in the
   passage, so the answer reads in document order.
7. Rendered answer: each claim's text, trimmed, ending in `[n]` where `n` is
   the 1-based index of its passage in the final context list.

### 6.2 LLM generator (when `OPENAI_API_KEY` is set)

Same context, `temperature = 0`, and a system prompt that requires: answer only
from the numbered context, one or more `[n]` markers on **every** sentence, no
outside knowledge, and the exact string `INSUFFICIENT_CONTEXT` when the context
does not contain the answer. The output then goes through §7 exactly like the
extractive output — the enforcement layer does not trust the model.

---

## 7. Citation enforcement

Applied to every answer regardless of generator. A claim is the unit of check.

| Rule | Check | Failure action |
| --- | --- | --- |
| **CE1** | every `[n]` resolves to a passage in the final context, and that passage's `chunk_id` matches the claim's | strip the marker, claim becomes uncited |
| **CE2** | every declarative sentence carries ≥ 1 marker | claim marked `uncited` |
| **CE3a** | idf-weighted containment of the claim's content tokens in its cited passage ≥ `GROUNDING_FLOOR` | claim marked `ungrounded` |
| **CE3b** | every number in the claim appears in the cited passage (after normalisation, `%` and units included) | claim marked `number-hallucination` — hard failure, no threshold |
| **CE4** | at least one claim survives | whole answer becomes a refusal |

Repair loop: on any CE2/CE3 failure the pipeline **retries once** — the LLM path
regenerates with the failed claim quoted back and a stricter instruction, the
extractive path re-selects excluding the failed sentence. Claims that still fail
are dropped. If every claim is dropped, the response is:

```
I couldn't find that in the indexed documents.
```

with `refused = true`, a `refusal_reason` distinguishing the two ways to get
here (`no passage reached the relevance floor` versus `every candidate claim
failed citation enforcement`), and the inspected passages still returned as
`considered` so a user can see what was searched.

Every surviving claim yields a citation record: `{ index, chunk_id, source_id,
doc_title, heading, char_start, char_end, quote, quotes, quote_exact,
grounding }`.

`quote_exact` is false when the quote had to be located by a fallback rather
than found verbatim — a sentence that wrapped across two source lines is
re-joined with a single space before the search, so the span is correct but no
longer a byte-for-byte substring of the original. Consumers that highlight the
source should treat a false value as "approximate span".

There is **one record per marker number, not per claim**, and it must carry
*every* quote drawn from that passage in `quotes` (with `quote` as the joined
form). Keeping only the first quote per passage is a real bug that was shipped
and caught: three claims citing `[1]` produced one citation holding one
sentence, so figures from the other two claims were no longer backed by any
stored quote and the digit contract below failed on 9 of 22 golden cases.

The API contract guarantees `answer_text` contains no digit that is absent from
its cited quotes. Marker digits are stripped before the check, otherwise `[1]`
would count as an uncited `1`.

---

## 8. Evaluation metrics

Deterministic re-implementations of the Ragas metric definitions, so CI needs no
LLM judge and no network. `eval/run_eval.py --judge llm` swaps in an LLM judge
when a key is present; the deterministic numbers are the ones that gate CI.

Let `C` = retrieved contexts in final rank order, `A` = the answer's claims,
`F` = `expected_facts`, `S` = `expected_sources`.

**`faithfulness`** = supported claims / total claims, where "supported" means
the claim passed CE3a and CE3b. A refusal scores 1.0: it asserts nothing, so
nothing in it can be unfaithful. Whether the refusal was *appropriate* is
measured by `refusal_accuracy` and `answer_correctness`, not here.

**`answer_relevancy`** =
`0.5 * cos(embed(expand(question)), embed(answer)) + 0.5 * cov`, where
`cov = |content_tokens(question) ∩ (content_tokens(answer) ∪ content_tokens(cited headings))| / |content_tokens(question)|`.
Refusals are **excluded from the mean** rather than scored 0 — the refusal text
is deliberately unrelated to the question, so scoring it would just measure how
many questions were refused, which `refusal_accuracy` already reports.

**`context_precision`** = average precision over `C`:
`rel_i = 1` iff `C_i.source_id ∈ S` **and** (`F` is empty or `C_i` contains ≥ 1
fact of `F`). `AP = Σ_i (P@i * rel_i) / max(1, Σ_i rel_i)`.

**`context_recall`** = `|{f ∈ F : f appears in some C_i}| / |F|`.
Undefined (excluded from the mean) when `F` is empty.

**`answer_correctness`** = `|{f ∈ F : f appears in answer_text}| / |F|`.
Excluded when `F` is empty.

**`citation_validity`** = 1 iff every claim in the rendered answer has a marker
that resolves and passed CE3; refusals score 1.

**`refusal_accuracy`** — computed over `answerable = false` cases only:
1 if the system refused, 0 if it answered.

Fact matching uses the §1 normalisation on both sides, then substring
containment, with numeric tokens additionally compared after stripping
thousands separators.

### 8.1 Measured baseline and CI gate

`eval/thresholds.json` holds the gate. `run_eval.py` exits non-zero when any
mean metric is below its threshold, **or** when any `answerable: false` case was
answered, and writes `eval/report.md` + `eval/report.json`. The GitHub Actions
workflow runs unit tests, then the gate, and uploads the report as an artifact
on every pull request.

Baseline over all 33 golden cases (`mode=hybrid`, `reranker=lexical-ce`,
`embedder=hashing`, `generator=extractive`, deterministic judge):

| metric | measured | gate floor |
| --- | --- | --- |
| faithfulness | 1.000 | 0.90 |
| context_precision | 1.000 | 0.85 |
| context_recall | 1.000 | 0.90 |
| answer_relevancy | 0.585 | 0.50 |
| answer_correctness | 1.000 | 0.85 |
| citation_validity | 1.000 | 1.00 |
| refusal_accuracy | 1.000 | 1.00 |

All 23 answerable cases produce every expected fact with valid citations; all
10 unanswerable cases refuse.

`answer_relevancy` sits at 0.585 by construction, not by weakness: the answers
are verbatim policy sentences, and a hashed TF-IDF sketch of a document
sentence is only ever moderately similar to a question phrased in a user's own
words. It is included as a regression tripwire, not as a quality target.

### 8.2 Ablation

Same 33 cases, varying retrieval mode and the reranker:

| configuration | context precision | correctness | relevancy | refusal accuracy | gate |
| --- | --- | --- | --- | --- | --- |
| bm25 only, no rerank | 0.951 | 1.000 | 0.506 | 0.100 | FAIL |
| bm25 + rerank | 1.000 | 1.000 | 0.585 | 1.000 | pass |
| vector only, no rerank | 1.000 | 1.000 | 0.515 | 0.300 | FAIL |
| vector + rerank | 1.000 | 1.000 | 0.585 | 1.000 | pass |
| hybrid (RRF), no rerank | 0.980 | 1.000 | 0.512 | 0.100 | FAIL |
| **hybrid (RRF) + rerank** | **1.000** | **1.000** | **0.585** | **1.000** | **pass** |

The honest reading, which is not the usual sales pitch for either stage:

- **The cross-encoder is load-bearing, for refusal rather than precision.** Its
  calibrated score is the only signal thresholdable into a refusal policy.
  Without it, refusal accuracy collapses from 1.000 to 0.100 — 9 of 10
  unanswerable questions get confidently answered — and every such
  configuration fails the gate outright. Context precision, the thing
  reranking is usually sold on, moves only from 0.980 to 1.000.
- **Hybrid fusion is a robustness win, not a precision win on this corpus.**
  With the reranker on, all three retrieval modes reach identical numbers,
  because 57 chunks of tightly-worded policy text is a corpus where either
  branch alone usually surfaces the right passage. Fusion earns its place by
  removing the *dependence* on which branch happens to work for a given query
  — visible in the no-rerank rows, where BM25 alone drops to 0.951 while
  fusion holds 0.980 — and it would matter more on a larger or noisier corpus.
  Note that dense retrieval alone scores 1.000 there, so claiming a precision
  win for fusion on this corpus would be unsupported by the measurement.

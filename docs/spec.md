# Parsica Recall 1.0 spec

Spec `recall/1.0`. This file is the specification of record for Parsica Recall. Changes to it arrive by pull request, like the code.

Parsica Recall measures whether a memory system surfaces the right record for a question, with a judge that never calls a model, a retrieval window and corpus size that are fixed by the benchmark rather than chosen by the system under test, and results that anyone can re-derive from the per-question records. This spec defines 1.0: the data formats, the adapter contract, the scoring, the standard report, and the rules that make two results comparable.

## 1. Purpose, scope and the three promises

Recall answers one question: given a store of memories and a question whose answer lives in one or a few of them, does the system put those records in the window an agent would actually read? It scores retrieval, not answering. The answer model is a separate, optional stage (fullstack mode) and never the judge.

**Promise 1, the judge never calls a model.** Every score is a function of the question file, the corpus and the returned records: id match, normalized substring, token overlap. Two people running the same inputs get the same bytes. An LLM judge is not permitted in any mode of this benchmark; the only model allowed anywhere is the answer model in fullstack mode, and it is the thing under test, not the referee.

**Promise 2, k and N are set by the benchmark.** The retrieval window k is a property of the consumer (how many records an agent prompt can carry), and the realistic window grows with the store, so the benchmark fixes k per pool-size band and every system at that size reports the same k. The corpus size N is a property of the task, so it is either a published standard size (padded with a published distractor set) or stated in large print next to every number. A system does not get to pick the k that flatters it or the N that makes chance look like skill.

**Promise 3, every result can be re-derived.** A run writes the first-hit rank of every question, the full returned window, and a manifest with content hashes of the question file, the corpus, the harness and the adapter. Recall@k for any k up to the window, MRR, the chance line and the skill number all follow from the per-question records alone.

In scope for 1.0: retrieval mode against a file corpus, retrieval mode against a live store, the standard report, the lint and gates that protect the question set, the generation protocol for new question sets. Out of scope for 1.0, specified but deferred: fullstack mode. Not in scope at all: comparisons with memory products as products (end-to-end chat scores).

**Open source.** Recall is MIT and stays MIT. It is meant as a public answer to benchmark numbers that move when the k, the judge or the answer model moves. The repository holds no engine code, the reference adapters are textbook and written from scratch, and nothing ships in a public question set or distractor file until it has passed the secrets-and-PII rule in section 9. The October 2026 audit that reshaped the benchmark, including the finding that the earlier question set v9 flattered the system it was first built to test, is part of the public record. v9 and its cleaned successor v10 are named in this document as history; they are not published.

## 2. Pipeline

1. Generate candidate questions from a memory source (the one step that may use a model; section 9).
2. Lint against the whole pool and run the release gates.
3. Freeze: `questions.json` with `gold_ids` and `gold_hashes`, `corpus.jsonl`, and a manifest with their hashes.
4. A run reads the frozen set and picks the band from N (section 6).
5. File mode: ingest the corpus into a fresh workspace. Live mode: open the store read-only and take the pool from `list_ids()`.
6. One search per question at the band's window.
7. Score every question on its own records (section 5) and aggregate: gold, skill, chance, MRR, breakdowns.
8. Verify with a second run, write the results file with its manifest, print the one-line result.

Questions are generated once from a memory source and frozen with the ids and content hashes of their gold records; the lint and gates decide what ships. A run reads that frozen file and either ingests the corpus into a fresh workspace (file mode) or searches a store that already holds the records (live mode). Both produce the same results shape, and a result is verified when a second run is byte-identical and the manifest hashes match.

## 3. Data formats

The corpus and the questions are separate files, joined by record ids. An id is the SHA-256 of the record's normalized text (lowercase, punctuation stripped, whitespace collapsed), truncated to 16 hex characters, so the same text always gets the same id in any store and a padded or reordered corpus does not change it.

**corpus.jsonl**, one record per line:

```json
{"id": "3f9a1c0e7b2d4a61", "text": "The author lost 22 pounds in 4 months.", "source": "session_0412", "kind": "user", "ts": "2026-03-10T00:00:00Z", "meta": {}}
```

`id` and `text` are required; `source`, `kind`, `ts` and `meta` are optional and pass through to the adapter's `ingest`. In live mode there is no corpus file: the store is the corpus, and `list_ids()` on the adapter enumerates it. A private set generated from a live store may keep the store's own stable ids in file mode too (--store-ids); such a result is never a standard result, and public sets always use content ids.

**questions.json**, a list:

```json
{
  "id": "q_0011",
  "question": "How many pounds did the author lose in four months?",
  "category": "direct_recall_user",
  "difficulty": "easy",
  "answer_type": "quantity_with_unit",
  "answer": "22 pounds",
  "acceptable_answers": ["22 lbs", "twenty-two pounds"],
  "gold_ids": ["3f9a1c0e7b2d4a61"],
  "must_rank_first": null,
  "gold_hashes": {"3f9a1c0e7b2d4a61": "sha256:..."},
  "review_status": "publish",
  "review_notes": ""
}
```

Rules for the fields that carry the score:

- `gold_ids` lists every record that counts as a correct retrieval. One id for single-fact questions; several for multi-record categories (cross\_session, knowledge\_update, correction\_handling, recency\_preference). The runner refuses a question whose gold ids are not all present in the pool.
- `must_rank_first`, when set, names the gold record that has to be the first gold in the window for the question to score on the version tier (the current version of a changed fact). It may be a list when the current version exists as several verbatim copies in the pool; any of them first scores. Null elsewhere. The version tier counts an item only when every one of its gold records carries a `ts` (from the corpus in file mode, from `record_ts()` in live mode), because nothing else in the pool can tell a system which record is current; with no such item the tier is n/a, and the harness never invents a timestamp (amended October 19, 2026).
- `gold_hashes` pins the gold text at generation time. A live store that has rewritten or enriched the record still matches by id; the hash is what tells you the text drifted, and the run reports the count.
- `answer` and `acceptable_answers` are used only by the exact tier and by fullstack mode. They are not how retrieval is judged.
- A record's text is not in the question file; it lives in the corpus once.

**Categories.** There are thirteen (`recall_common.CATEGORIES`). Abstention questions carry `gold_ids: []` and are scored by the abstention rule (section 5), and the four multi-record categories must have at least two gold ids or the linter reclassifies them. Minimum counts per category for the standard track are in section 9.

**Pool definition.** The pool is the set of record ids the retriever can return: the corpus file in file mode, `list_ids()` in live mode. N is its size. Every chance and ceiling number is a function of N and of each question's G = `len(gold_ids)`, and nothing else.

## 4. Adapter contract v2

The v1 five-method shape stays and every v1 adapter keeps working. v2 adds ids and scores to `search`, four optional methods, and a stricter rule on what the runner does with the window.

```python
class BaseMemoryAdapter(ABC):
    def setup(self, workspace: str, mode: str) -> None: ...   # mode is "file" or "live"
    def ingest(self, record: dict) -> None: ...                # file mode only; record has id, text, and the optional fields
    def save(self) -> None: ...
    def search(self, query: str, limit: int) -> list[Hit]: ...  # Hit = (id, text, score) or a bare str (v1)
    def teardown(self) -> None: ...

    # optional
    def list_ids(self) -> list[str]: ...                 # the pool, live mode; default: unknown
    def record_ts(self, ids: list[str]) -> dict[str, str]: ...  # live mode, read-only: the store's own ts per id; the version tier needs it
    def memory_count(self) -> int: ...                   # default -1
    def version(self) -> dict: ...                       # the engine declaration below; default unknown
```

- **Ids.** A hit's id is matched against `gold_ids`. An adapter that cannot return ids returns text, and the runner derives the id by hashing the normalized text, which only works when the store returns records verbatim. Adapters that chunk, merge or enrich on ingest should keep the record id and return it, or they will be scored as if they lost the record.
- **Scores.** Optional, any monotone real number (higher is better). When present they feed the calibration metrics and the abstention rule. When absent those are reported as not available, never estimated.
- **The window.** The runner requests `limit = window` once per question at the band's window (10, 15, 20 or 25, section 6) and truncates to that window whatever comes back. An adapter that returns more does not get credit for it; an adapter that returns fewer is scored on what it returned. Every smaller k in the band's set is derived from the stored ranks of that one call, so there is exactly one search per question per run.
- **Ordering is the system's claim.** Ties are the adapter's problem to break deterministically; the runner never re-sorts.
- **No peeking.** The adapter sees the query string and nothing else: not the category, not the gold ids, not the question id. A reference adapter in the repo (plain BM25, 40 lines) exists so a new adapter author has a number to beat and a shape to copy.
- **Live mode.** `setup(workspace, "live")` opens a connection; `ingest` is never called; `list_ids()` is required for the chance line and the runner refuses to produce a standard report without it (it still produces recall numbers, labeled as pool size unknown, with `pool=unknown` on the result line). `verified` keeps its one meaning, two identical runs, whether or not the pool was fingerprinted; the pool's state is its own field.

**Duplicate ids burn their slot.** If the window contains the same record id twice (chunking, or a store that returns a record and its enriched copy), the second occurrence is dropped and the window is not refilled. A system that fills a window of 15 with five copies of three records has returned three records.

**Declare the engine.** `version()` returns a small dict, not just a string: `{"version": ..., "engine_class": "lexical" | "embedding" | "hybrid" | "llm_assisted", "models": [...], "network": true | false}`. The manifest carries it, the report prints it, and leaderboards group by engine class, so a lexical engine is never silently ranked against one that calls a frontier model at query time. Using a model at search time is allowed; hiding it is not.

## 5. Scoring definitions

Every question produces a small set of per-question facts, and every reported number is an aggregate of those facts. Nothing is scored on the concatenated window; each returned record is judged on its own, which is what makes ranks well defined.

| Tier | Per-question fact | Rule | Role |
| --- | --- | --- | --- |
| Gold | `gold_rank` | Position (0-based) of the first returned record whose id is in `gold_ids`; null if none in the window | The retrieval headline |
| Gold, all | `gold_all_rank` | Position by which every id in `gold_ids` has appeared; null if not all appear | Multi-record categories |
| Version | `version_ok` | True when the first gold in the window is one of the `must_rank_first` records; only for questions that set it, and counted only where every gold record carries a `ts` | The "current record first" number |
| Exact | `exact_rank` | Position of the first returned record whose normalized text contains the normalized answer or an acceptable variant | Diagnostic; the fullstack scorer's input |
| Family | `family_rank` | Position of the first record with token overlap >= 0.6 against any gold text, by multiplicity, gold as denominator | Diagnostic only, never a headline |
| Abstention | `abstained` | See below | The abstention table |

**Normalization** is one function shared by the runner, the linter and the generator: lowercase, strip everything that is not a letter, digit or space, collapse whitespace. Number words are not expanded in retrieval mode; acceptable answers cover that.

**Abstention** questions have no gold. In retrieval mode they are scored only when the adapter returns scores: the question abstains correctly if the top score is below the adapter's declared threshold (reported in `version()` metadata) or the window is empty. Without scores, abstention is reported as not available and excluded from every aggregate. An earlier rule (penalize a window that happens to contain the phrase "not mentioned") is withdrawn.

**Multi-record categories.** knowledge\_update, correction\_handling and recency\_preference always set `must_rank_first` to the current version and list the stale version among `gold_ids`. The gold tier credits either version in the window (the system found the thread); the version tier credits only a window that puts the current one first. cross\_session lists all its records and is scored on `gold_all_rank`.

**Drifted gold.** In live mode, if a gold record's stored text no longer matches `gold_hashes`, the question still scores by id, and the run reports `drifted_gold_count`. A store that rewrites records is not penalized for rewriting; it is penalized only if it cannot find them.

## 6. Standard reporting: k, N, chance, skill and tracks

The rule that stops k-shopping: **k is fixed and N is standardized, and both are printed next to every number.**

**k is dictated by pool size.** k belongs to the consumer, and the consumer's realistic window grows with the store: an agent on a 500-record store reads five, and one on a store of 50,000 to 75,000 records reads twenty. So the benchmark fixes k per pool-size band, roughly five more per decade of N, and every system at a given N reports the same nested set, the same headline k and the same window. One search per question at the band's window yields every k in the set.

| Pool size N | Band | k set reported | Headline k | Window requested |
| --- | --- | --- | --- | --- |
| under 100 | none | 1 | 1 | 5 (labeled small pool, no standard report) |
| 100 to 999 | S | 1, 5, 10 | 5 | 10 |
| 1,000 to 9,999 | M | 1, 5, 10, 15 | 10 | 15 |
| 10,000 to 49,999 | L | 1, 5, 10, 15, 20 | 15 | 20 |
| 50,000 and up | XL | 1, 5, 10, 15, 20, 25 | 20 | 25 |

What a system may not do is choose: at its N the headline k is the band's, the whole set is printed, and a figure at any k outside the band's set is not a standard result. Within a band every system is judged at the same k. Across bands you compare on skill at k = 10, which every band reports, and on the scaling curve (section 7), never on a headline from one band against a headline from another.

**N is the difficulty knob, and chance makes it visible.** For a question with G gold records in a pool of N, a random retriever's hit rate in a window of k is the hypergeometric 1 minus C(N-G, k) / C(N, k); for G = 1 that is k / N. The runner computes the mean over answerable questions using each question's own G (this is the G-aware chance line already in the harness since July 3, 2026, now a required output) and prints it beside every recall figure.

| Pool N | chance@5 | chance@10 | chance@15 | chance@20 | chance@25 | Largest k with chance <= 5% |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 5.0% | 10.0% | 15.0% | 20.0% | 25.0% | 5 |
| 200 | 2.5% | 5.0% | 7.5% | 10.0% | 12.5% | 10 |
| 500 | 1.0% | 2.0% | 3.0% | 4.0% | 5.0% | 25 |
| 1,000 | 0.5% | 1.0% | 1.5% | 2.0% | 2.5% | 50 |
| 10,000 | 0.05% | 0.1% | 0.15% | 0.2% | 0.25% | 500 |
| 50,000 | 0.01% | 0.02% | 0.03% | 0.04% | 0.05% | 2,500 |

Values for G = 1; the runner uses measured G. The earlier set v9, a pool of 457 with a measured mean G of 3.68, had a 5% ceiling of k = 7, and its band under this spec is S with a window of 10, so its k = 15 window flattered everyone twice over.

**Skill is the cross-pool number.** Skill@k = (Gold@k minus chance@k) / (1 minus chance@k): 100 is perfect, 0 is no better than random, negative is worse than random, reported unclamped. At N = 10,000 skill and recall are the same to one decimal; at N = 100 they are not, and that is the point. Two systems on different pools are compared on skill, with N in the label, and never on raw recall.

**Validity of a k for a pool.** A k is valid when mean chance@k is at or under 5%. The band floors are set so the band's headline k is always valid at G = 1 (5% at N = 100, 1% at 1,000, 0.15% at 10,000, 0.04% at 50,000). A k in the set that is invalid for the actual N and measured G (the window at the bottom of band S, or any k when G runs high) is still printed, marked above ceiling, and excluded from the headline line. Under 100 records only Gold@1 is reported and the run is labeled small pool.

**Standard pool sizes.** Skill corrects for chance but not for distractor difficulty, so systems on different N are still not equals. The public tracks fix N by padding the question corpus with a published distractor file, taken in file order, deterministic:

| Track | N | Band | Headline k | Meant to stand in for | Status |
| --- | --- | --- | --- | --- | --- |
| Smoke | 250 | S | 5 | a unit-test store | tests and CI only, never reported |
| 1k | 1,000 | M | 10 | a few weeks of agent memory | required |
| 10k | 10,000 | L | 15 | months of memory | required |
| 30k | 30,000 | L | 15 | about a year of agent memory | required |
| 50k | 50,000 | XL | 20 | an everyday durable store for an agent that has run a while; the headline track, and Gold@20 here is the Recall number | required |
| 100k | 100,000 | XL | 20 | a long-lived agent store | optional until a public distractor file that size exists |

The public pool, `data/pool-v1/corpus.jsonl`, is a synthetic agent-memory store whose every track is the first N records of that one file ([docs/pool.md](pool.md); amended October 20, 2026). Its first 1,000 records, the core, are the 1k track by themselves.

**The gold window** (amended October 21, 2026). A public question set may take its gold from anywhere in the first 10,000 records of the pool, not only the core; those 10,000 records are the question corpus, and the records past them are the distractors of the larger tracks. So a set's questions are valid on the 10k, 30k and 50k tracks, where every gold record is in the pool. The 1k track holds only the core, so it reports the subset of the set whose gold records all sit in the first 1,000 (for an abstention question, whose topic records do); the set's manifest names that subset. The generator draws **core first, with a quota**: each category takes at most a set number of its questions from the core (40 for the first pool-v1 set) before it fills from the rest of the window, so the 1k track keeps about 400 to 600 questions of the set and the rest of its questions sit past the core. **The version categories take their units from the pool's own record of its planted changes** (amended October 22, 2026): a synthetic pool whose generator recorded the changes it planted supplies them as pairs of record ids (`data/pool-v1/planted_pairs.jsonl`, written from the pool's own generation record by `parsica-recall export-planted-pairs`, never from the persona files), and the generator builds knowledge_update, correction_handling and recency_preference units from those pairs by type, each pair's later record being the current one. Corrections and preference changes come from that record alone, so their count in a set is bounded by what the pool planted; knowledge_update goes on with pairs found by topic similarity past the core, since the tail's updates are not recorded. A store with no such record keeps the similarity pairs of docs/generation.md section 5. The lint's pool-aware rules still run against the whole file, the largest track the set ships with. When the pool grows past the records a set was linted against, the lint reruns against the new records, and an item that now leaks or stops abstaining is retired, never edited; the larger pool's generator keeps the set's answers out of the records it adds (docs/generation.md section 17).

Distractors are linted as part of the pool, and the lint is rerun whenever the question set changes: `GOLD_MULTIPLICITY` fails a question whose answer or alternate appears in any distractor, `NO_SECRETS_OR_PII_POOL` covers every distractor, and `GATE_DISTRACTOR_MATCH` compares the distractors' token-length and timestamp distributions with the corpus's (two-sample KS, tolerance 0.1). No lint rule measures overlap between a distractor and a gold record; the public pool's generator screens its tail against its core when it writes it (docs/pool.md section 5), and the family tier (section 5) reports any overlap that remains (amended October 21, 2026). Live-mode results carry their own N and are comparable to the tracks only through skill, with that caveat printed in the report.

A private corpus can stand in for the distractor file in a private track: real agent memories make far better distractors than forum text. Such a track stays private. The public distractor file for the 30k and 50k tracks has to come from public text, screened by the lint.

**Distractors carry no signal.** Three ways a padded corpus can leak which records are gold, and the rule that closes each: position (gold records would all be ingested first), so file mode ingests a deterministic shuffle of corpus plus distractors with the seed in the manifest, and the order-sensitivity pass uses a second seed; timestamps (gold from one period, distractors from another), so distractor `ts` values are drawn to match the corpus's distribution, or `ts` is omitted on public tracks; length (gold shorter or longer than the filler), so the lint reports the token-length distribution of gold and distractors and a track is not valid if they differ beyond a published tolerance. A ranker that learns "short, early, recent" should gain nothing from it.

**What a standard result must contain.** The full question set (no subsetting; a subset run is labeled partial), every category in the per-category table, every k in the band's set with chance and skill for each, MRR at the band's window (gold tier), the version-tier rate, the k ceiling at 5% and 10%, N and G statistics, and the manifest (section 8). The one-line form for leaderboards and journals is fixed too:

```
recall/1.0 track=10k band=L N=10000 k=15 window=20 qset=v11@a1b2c3d4 gold@15=89.3 skill@15=89.2 gold@1=61.0 gold@5=78.1 gold@10=85.0 gold@20=91.2 mrr@20=0.712 version=80.0 chance@15=0.15 ceiling5=500 adapter=my-store@0.3.1 verified=yes standard=yes
```

A line missing any field is not a Recall result. Three fields appear only when they apply: `spread@k=` after an unstable double run, `above_ceiling=k` when the headline k is invalid for the measured N and G, and `pool=unknown` or `pool=changed` on a live result. A live result without a pool size prints no k, chance or ceiling and lists gold at every k of the widest band instead.

## 7. Everything a run collects

The headline is a handful of numbers; the value of a run is in the rest. All of it is derived from one search per question plus two cheap extra passes, and all of it is written to the results file so nobody has to rerun to ask a new question of the data.

| Group | What | Why it is worth having |
| --- | --- | --- |
| Ranks | `gold_rank`, `gold_all_rank`, `exact_rank`, `family_rank`, `version_ok` per question; the full returned window with ids and scores | Re-derive any recall@k, audit any miss |
| Headline | Gold and Skill at every k in the band's set, MRR at the window, version-tier rate, chance curve for k = 1..N, k ceilings at 5% and 10% | The standard report |
| Breakdowns | Every headline number per category and per difficulty; per category the first-wrong rank (where the first non-gold record lands) | Shows which categories carry the score and where the ranker loses to a distractor |
| Pool | N, G mean/median/max, zero-gold count, drifted-gold count, `memory_count()` and store amplification (memories per record, which exposes chunking) | Tells a reader whether the number is comparable to another run |
| Latency | Search time per question; p50, p95, max; ingest records per second in file mode | A system that is right but slow is a different product |
| Calibration (scores present) | AUROC of score against gold hit, and the hit rate in each score decile | Makes abstention scorable, and tells an agent whether it can trust the score for a confidence threshold |
| Redundancy | Share of the window that overlaps another returned record at 0.9 or more | Wasted window slots; a store that returns five copies of one memory is not giving the agent five memories |
| Order sensitivity (extra pass, file mode) | Ingest the corpus in a second fixed shuffled order, rerun, report the share of questions whose `gold_rank` changed and the headline delta | Catches ingest-order leakage in rankers and stores; an engine whose ranking does not depend on ingest order scores zero here, and one that breaks ties by ingest order shows exactly those ties |
| Query robustness (extra pass) | Each question also run in two deterministic variants, casefolded with punctuation stripped and with its stopwords removed; report agreement of `gold_rank` with the original | Cheap, model-free paraphrase pressure; a lexical engine that collapses here is telling you something |
| Scaling curve (tracks) | Gold@10 and Skill@10 (the k every band shares) plus each track's own headline k, at N = 1,000, 10,000 and 50,000 from the same question set | The one picture that says how a system degrades as a store grows, which is the question a buyer actually has |
| Age (live mode) | Hit rate by record age bucket (the store's `ts`) | Recency bias, in either direction |
| Verification | Manifest hashes, double-run status, harness and adapter versions | Section 8 |

Two things on purpose not collected: anything that needs a model (no LLM-judged relevance, no embedding-based similarity of returned records to the question), and anything about the answer model in retrieval mode. The scaling curve and the extra passes are optional in a local run and required for a standard-track result.

## 8. Determinism and verification

A Recall result is a claim about a system; verification is what lets someone else check the claim without trusting the person who made it.

**The manifest** is written into every results file and hashed as a whole:

| Field | Value |
| --- | --- |
| `harness_version` | Recall version, the git commit of the runner when it ran from a checkout, and `source_sha256`, the SHA-256 of the harness's own source files |
| `qset` | question-set version plus SHA-256 of the file |
| `corpus` | SHA-256 of corpus.jsonl, or `live` plus N and the SHA-256 of the sorted id list; `id_scheme`, `content` or `store` |
| `distractors` | SHA-256 of the distractor file and how many were taken, or none |
| `adapter` | module path, class, `version()` string |
| `runtime` | Python version, platform, `PYTHONHASHSEED` |
| `params` | mode, track, band and window, extra passes run |
| `records_hash` | SHA-256 of the per-question records with timing fields removed |

**The double run.** `--verify` runs the whole benchmark twice in one invocation and compares `records_hash`. Equal means the system is deterministic under this harness and the result is marked verified. Unequal means the result is marked unstable, the differing question ids are listed, and the headline is reported as the mean of the two runs with the spread beside it. Non-determinism is not a failure of the benchmark; hiding it would be. The double run covers the base run. The extra passes (section 7) run once, on a fresh adapter after the base run, and carry the base run's verification status; when the base run is unstable the pass differences include run-to-run noise and the report says so. A standard result needs a verified base run with both passes run.

**What the harness controls.** Hash-seed randomization is pinned (the existing re-exec), ingest order is the corpus file order on `track=none` and the deterministic shuffle of section 6 (seed in the manifest) on a track, question order is the file order, the workspace is fresh per run, and the runner never re-sorts the adapter's results. Everything else that could vary lives inside the adapter, which is the thing being measured.

**What verified means and does not mean.** Verified says: this result can be reproduced from these inputs by this code. It does not say the inputs are good; that is the job of the lint and gates (section 9), whose pass is recorded in the question set's own manifest. A standard-track result is one that is verified, run on the full question set, on a standard track, with the extra passes.

**Comparing two results.** Two results are comparable when `qset`, `corpus` (or track and distractors) and `harness_version` match. `harness_version` is the Recall version plus the commit, so results from two commits never diff, and a result from a tree with uncommitted changes is not pinned to its commit and is refused the same way; `--allow-unpinned` downgrades both refusals to printed notes for local work and marks the summary line unpinned. A copy installed from a package has no commit. It is pinned by `source_sha256`, which every result records: where either result has no commit, the two compare when their source hashes are equal, and a result with neither a commit nor a source hash is pinned to nothing (amended November 21, 2026). A live result whose pool changed during its run has no single pool and is refused too. The comparison tool refuses to diff results whose manifests disagree on any of this, names what differs, and prints the per-question diff (questions won, lost, rank moved) when they agree. This is the tool for a before-and-after number around a change to a store, such as a backfill or a new enrichment pass.

**Overfitting and leaderboard entries.** Verified means reproducible, not honest: a retriever tuned on a set will score well on that set. Two levers keep that in check. The generation protocol lets anyone regenerate a fresh question set from the same corpus with a new seed, and question sets get a scheduled regeneration (a new version each quarter, or whenever a result looks too good); results always name their version. And a leaderboard entry requires the adapter source, the manifest, and a rerun by someone other than the submitter; an entry whose adapter reads anything but the query string (the question file is on the same disk, after all) is removed when found. A sealed result, run by the maintainer on a set the submitter has never seen, is the strongest badge and the only one worth arguing about. From the first pool-v1 set on, a public set is split by seed into a public half and a sealed half; the public half's manifest carries only the SHA-256 of the sealed half's question file, as a commitment, and the sealed half is never published: the maintainers keep it outside the public repository (amended October 21, 2026; docs/generation.md section 9).

**Results files.** A system may also be scored from a results file it writes itself, with no adapter (docs/adapting.md): one line per question with its hits, best first. The scorer is the same code. What the harness cannot vouch for is marked on the result: it is one run made outside the harness, so the result says `verified=no`, the adapter is named `file:<name>`, the manifest holds the file's SHA-256, the extra passes do not run, no search time is reported unless the file reports its own, and it is never a standard result. Every question needs a line; a file that leaves one out is refused (amended November 21, 2026).

## 9. Question set hygiene and generation

An audit of the earlier set v9 showed that a question set can pass every schema rule and still hand out 50 free hits. The 1.0 linter therefore sees the pool, not just the item, and every gate is code.

**Pool-aware lint rules** (all run against the corpus plus the distractor file of the largest track):

| Rule | Severity | Test |
| --- | --- | --- |
| gold\_present | error | every `gold_ids` entry exists in the pool and its hash matches |
| gold\_multiplicity | error | the exact-tier G over the pool is 1 for single-record questions (the answer text appears in no other record); multi-record questions have G equal to `len(gold_ids)`; role labels excepted |
| answer\_not\_in\_question | error | neither the answer nor any alternate appears in the normalized question |
| key\_edges | error | no answer or alternate begins or ends on a function word, or is a single function word |
| alternate\_specificity | error | no alternate matches more than 1% of pool records |
| duplicate\_fact | warn, then reclassify | at most one question per (gold set, normalized answer); the duplicates become a paraphrase\_recall item or are dropped |
| source\_length | error | every gold record has at least 6 tokens |
| entity\_coherence | warn | a capitalized entity in the question that appears in no gold record is flagged for human review |
| multi\_record\_shape | error | the four multi-record categories carry at least two gold ids and, where required, `must_rank_first` |
| category\_minimums | error, release gate | each category has at least 25 publishable items in the standard set |

Two more rules apply to anything that ships in the public repo, written fresh for Recall and not borrowed from any other codebase: **no\_secrets\_or\_pii** (error): no record, question, answer or distractor matches a credential shape (API-key prefixes, long high-entropy tokens, connection strings) or a personal-data shape (email, phone number, street address, government id); and **distractor\_match** (error): the distractor file's token-length and timestamp distributions sit within the published tolerance of the corpus's (section 6).

**Gates are executable.** Every gate is a check with a pass/fail result in the lint report, and the question set's manifest records the lint report hash. A question set with any failing gate cannot be used for a standard-track result; the runner checks the manifest.

**Severity and status vocabulary** is one enum (`error`, `warn`, `info`; `publish`, `rewrite`, `reclassify`, `drop`, `hold_experimental`), checked against the lint spec on every lint run. There is no separate public and experimental split: whether a question set can back a standard result comes only from its gates and its manifest. A mismatch between two files that once silenced a warning threshold is the kind of bug one enum rules out.

**Generation protocol (the doc2query path).** Generating questions is the one step that uses a model, so it is fenced:

1. Sample records from the source, stratified by `kind` and by age, with a fixed seed recorded in the manifest; never the whole store.
2. For each category, a frozen prompt (its hash in the manifest) produces candidate questions with answers, alternates and the ids of the records they came from. Multi-record categories are generated from record pairs the sampler builds (a fact and its later update, two sessions on one topic).
3. Run the full linter including the pool-aware rules against the real pool. Everything that fails is held or dropped; nothing is hand-edited into passing without a note in `review_notes`.
4. Freeze: write the question file with `gold_ids` and `gold_hashes`, version it, record the generator model and prompt hashes in the manifest. From here on the set is data; regenerating is a new version.
5. Run the reference BM25 adapter and the oracle adapter. The oracle must score 100 on the gold tier (or the set is broken), and the BM25 number is published as the floor that a system has to clear to be interesting.

How the generator carries out these steps is in [docs/generation.md](generation.md), and its frozen prompts are in [prompts/](../prompts/).

**Versioning.** Question sets are immutable once published. A fix is a new version; results name the version; the comparison tool refuses to compare across versions. v11 is the first published set under this spec.

## 10. Live mode and privacy

Live mode is the benchmark against a store that exists. The question set was generated from that store (section 9), the runner skips ingest, and `search` goes to the real retrieval path, enrichment and all.

- **Read only.** The adapter opens the store read-only or uses a token scoped to search. The runner never writes to a live store.
- **Fingerprint the pool.** `list_ids()` is called at the start and again at the end of every run, and the SHA-256 of the sorted id list goes into the manifest. If the two differ (an agent wrote or deleted a memory), or the end call returns nothing, the run is marked `pool=changed` and `verified=no`; rerun on a quiet store or a snapshot copy. The fingerprint covers ids, not texts, so a record rewritten in place mid-run shows up only through `drifted_gold_count`. A store that cannot list its ids gets recall numbers labeled pool size unknown and never a standard result.
- **Gold drift is reported, not punished.** A store that enriches or rewrites records still matches by id; `drifted_gold_count` says how many gold texts no longer match their generation-time hash.
- **N is whatever the store has.** The report prints it, prints chance and skill, and labels the result live rather than a track. Comparing a live result to a track result is done on skill with N stated, and never as a leaderboard entry.
- **Results stay local by default.** Questions generated from a personal store contain personal facts, and the returned windows contain store content. Live results are written next to the question file by default and never under the Recall repository, whatever path is passed; the runner resolves symlinks before it checks. Whoever keeps a question set inside another repository keeps the result files out of it. `--redact` replaces every returned record text with its `text_hash` (so gold drift can still be checked) and drops the question, answer and alternate fields, which a redacted file has no use for and which are short enough to guess from a hash; ids, ranks and scores stay, which is enough for the standard report, the comparison tool and a journal note. Shared results (the journal, a PR, a leaderboard) are redacted results.
- **What goes in the journal.** The one-line result string from section 6 plus the manifest hash. Never the per-question records from a live store.

## 11. Fullstack mode (deferred)

Fullstack mode asks the second question: given the window the system retrieved, does an answer model produce the right answer? It is deferred past 1.0 and is not in the harness today. It is on the roadmap, and its rules are fixed now so nobody designs around their absence.

**There is no judge model.** The judge in fullstack mode is the same deterministic matcher as everywhere else, with the three fixes from the audit: word-boundary matching instead of bidirectional substring, a minimum generated-answer length of one content word, and an abstention phrase list without "there is no". What gets standardized is the answer model: it is the only model in the pipeline, and swapping it is the easiest way to move a fullstack number.

**The standard answer model is pinned open weights.** Criteria: open weights with a permissive license, small enough to run on a laptop, instruction-tuned, and middle of the road on purpose: a mid-sized instruct model whose extraction ceiling on the oracle window clears 95%, since a frontier model would only be measuring itself; greedy decoding (temperature 0), served by a pinned backend version. The weights hash, backend and version go in the manifest, and the double run applies. An API model may be reported as a second line, labeled with its exact model id and the date of the run, and is never the standard line, because API models change under the same name.

**How to pick it, rather than whose name is biggest:** run three candidates on the oracle window (gold records only) for a whole question set and keep the one with the highest agreement with the matcher, that is, the model that most reliably extracts the answer when handed the right record. That number is the model's extraction ceiling, and it is published with the model choice.

**Which era of model.** A modern one. Benchmarks that test reasoning over long context with a model as the judge lean on strong models, so a stronger model saturates them and the questions have to get harder. Recall's answer model only reads a window of five to twenty records and extracts; any modern mid-sized model does that at 95% or better on the oracle window, and that is the intended result, because fullstack is meant to measure retrieval through the model rather than the model. Difficulty in Recall comes from N, from distractors and from the multi-record categories, not from question wording, so moving to a current model does not force a harder question set. A current API model (a small or mid-sized model of the day) is a good second line, pinned by exact model id and run date; it cannot be the standard line because it cannot be hashed or run locally and will one day be retired under the same name.

**Scoring.** Context is the band's headline window (5, 10, 15 or 20 records), joined in rank order with a record separator; the system prompt is frozen and its hash is in the manifest. The report prints three numbers: the retrieval headline (Gold@5), the fullstack headline (matcher pass rate on generated answers), and the extraction ceiling, so the reader can see how much of the gap between retrieval and fullstack belongs to the model. Abstention lives here: an abstention question passes when the model answers with an abstention phrase, and fails when it invents an answer from a window that does not contain one.

**Cost.** Input and output tokens per question are recorded, and no price table, since prices go stale and tokens do not.

**Not allowed.** Any retrieval-time use of the answer model (query rewriting, reranking) counts as part of the system under test and must happen inside the adapter, where it is scored as retrieval. An LLM used to judge answers is not a Recall result.

## 12. Decisions

Every decision that shaped 1.0, with the reasoning, so nobody reopens one without knowing why it was made. All were settled on October 19, 2026.

**Settled:**

- k is dictated by pool size (section 6).
- Validity threshold: chance at or under 5%; N = 100 is the floor for a standard report.
- Tracks: 1k, 10k, 30k and 50k, all required; 30k stands in for about a year of agent memory.
- Family tier stays as a diagnostic; the data is worth keeping.
- Answer model: middle of the road and modern (section 11); the bake-off picks among mid-sized open-weight candidates.
- Order of work: the gold tier, the lint and the headline first, then live mode, then tracks and distractors, then fullstack.

**Settled, with the reasoning on record:**

1. **Headline tier.** Settled: gold by id at the band's k is the retrieval headline; exact and family stay as diagnostics. The reasoning: id is the record's identity, so it cannot credit the wrong record, cannot be echoed, survives enrichment and rewriting, makes G explicit, and is the only tier that can tell a current version from a stale one.
2. **Which track is quoted as "the" Recall number.** Settled: the 50k track at Gold@20. The reasoning: about 50,000 memories with a window of 20 is the everyday case for anyone running an agent with durable memory, high scores at 20 and above on a large pool are where a number starts to mean something, and what matters is recalling the right memory when it matters most. 100k is the next track up.

What is built and what is not is tracked in the README, not here. This document changes only when a decision changes, and a changed decision is a new spec version, named on every result line.

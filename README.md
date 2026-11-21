# Parsica Recall

A deterministic benchmark for the memory of AI agents. It asks one question: when an agent needs a memory, does the memory system put the right record in front of it?

Recall scores retrieval only. The judge never calls a model. The window it scores is set by the size of the store, not by the system being tested. Every number can be re-derived from the per-question records a run writes.

- Version: `1.0.0b1`, a beta. Recall stays in beta until it has been adequately used or peer reviewed.
- Spec: `recall/1.0`, in [docs/spec.md](docs/spec.md). Where the spec and the code disagree, the spec wins.
- Current question set: `v11`, interim, over a public synthetic pool of 10,000 records. The schema has 13 categories.
- License: MIT.

## Why it exists

Benchmark numbers are easy to move. Pick a friendlier k, a smaller pool, a kinder judge model or a stronger answer model, and the same system looks better. Recall is built so that none of those choices belong to the system under test:

1. **The judge never calls a model.** Every score is an id match, a normalized phrase match or a token overlap. Two people running the same inputs get the same bytes.
2. **k and N are set by the benchmark.** The window comes from the size of the pool, so every system at the same size is judged on the same window, and the chance line is printed beside every number.
3. **Every result can be re-derived.** A run writes each question's full window, with ids, ranks and scores, and a manifest with the hashes of everything that went in.

Recall publishes no system's numbers, its maintainers' own included. The only numbers in this repository are calibration: plain BM25 and an oracle, so a reader knows where the floor and the ceiling sit.

## Quick start

Python 3.11 or later. Standard library only.

```bash
pip install parsica-recall

# Plain BM25 on the current question set, run twice and checked for determinism
parsica-recall run --adapter bm25 --verify

# The ceiling (reads the question file, which no real system may do)
parsica-recall run --adapter oracle --verify
```

From a clone, with nothing installed, `python -m parsica_recall run --adapter bm25 --verify` does the same.

A run prints a report and writes `results/<qset>_<adapter>.json` in the folder you ran it from, plus two CSVs beside it: the recall curve, and the per-category and per-difficulty breakdowns. The last line of the report is the one-line result:

```
recall/1.0 track=none band=L N=10000 k=15 window=20 qset=v11@2e5b2c03 gold@15=87.58 skill@15=87.56 gold@1=59.66 gold@5=82.36 gold@10=85.89 gold@20=88.5 mrr@20=0.69 version=48.48 chance@15=0.1449 ceiling5=516 adapter=bm25@1.0 verified=yes standard=no
```

`parsica-recall` with no arguments lists every command, and `parsica-recall <command> --help` prints its options.

## Run it on your memory system

There are three ways in, and all three end in the same scorer. [docs/adapting.md](docs/adapting.md) is the full guide.

**1. A results file, with no code from this repository.** For a system in any language, or a hosted service.

```bash
parsica-recall export-task --out ~/recall-task     # the queries (ids and text only), the corpus, the window
# load corpus.jsonl into your system, run every query, write one JSON line per query:
#   {"qid": "q_0001", "hits": [{"id": "3f9a1c0e7b2d4a61", "score": 12.5}, ...]}
parsica-recall score ~/recall-task/results.jsonl
```

A scored file says `verified=no`, because the harness did not run the system itself. The score is exact all the same: it follows from the file, and the file's hash is in the result.

**2. A Python adapter.** One class with five methods. The harness runs the system itself, so the run can be verified and timed.

```python
from parsica_recall.adapters import BaseMemoryAdapter

class MyAdapter(BaseMemoryAdapter):
    def setup(self, workspace, mode="file"):
        self.store = MyStore(workspace)            # a fresh, empty directory in file mode

    def ingest(self, record):
        # record = {"id": ..., "text": ..., and maybe "kind", "ts", "source", "meta"}
        self.store.add(record["id"], record["text"])

    def save(self):
        self.store.flush()

    def search(self, query, limit):
        # best first; return (id, text, score) tuples
        return [(h.id, h.text, h.score) for h in self.store.search(query, top_k=limit)]

    def teardown(self):
        self.store.close()

    def version(self):
        return {"version": "0.1", "engine_class": "hybrid", "models": ["my-embedder"], "network": False}
```

```bash
parsica-recall run --adapter custom --custom-adapter my_module.MyAdapter --verify --passes order,query
```

**3. Live mode.** For a store that already holds its records, measured as it stands, read only. See [Live mode](#live-mode).

The six adapters that ship here (`empty`, `random`, `echo`, `bm25`, `bm25stop`, `oracle`) are reference adapters: yardsticks, not memory systems. No adapter for any product ships with Recall.

## Build a benchmark from your own memories

The public set measures a system on a shared pool. The question that usually matters more is how a system does on your memories. `generate-questions` writes a question set from any store's export, in the doc2query style: it reads records and writes the questions an agent would later ask about them, with the gold records pinned by id. [docs/doc2query.md](docs/doc2query.md) is the guide, end to end.

```bash
pip install "parsica-recall[generate]"     # adds the anthropic SDK, needed only for a live generation run

parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 \
    --seed 7 --store-ids --estimate          # what it is likely to cost; no call is made
parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 \
    --seed 7 --store-ids --max-cost 10       # a live run needs a spending limit
```

Three things to know before you run it:

- **It is the one step that uses a model.** Scoring never does. Today the question builder runs on Claude through the Anthropic API, with your own key, and a set records which model wrote it.
- **It sends some of your records to that API.** The frozen prompt and the text of the sampled records go out, never more than half the store. A record with the shape of a credential or of personal data (keys, tokens, email addresses, phone numbers, street addresses, card numbers) is never sent. That screen reads shapes, not meaning: a personal fact written in plain words can be sent.
- **A set built from your store restates your memories.** Keep the set, its folder and its results out of public places. `--redact` makes results you can share.

## How scoring works

**Records and ids.** The corpus is a JSONL file of records. A record's id is the first 16 hex characters of the SHA-256 of its normalized text, so the same text gets the same id in any store. Normalization is one function (`recall_common.normalize`) shared by the runner, the linter and the generator: Unicode NFKC, lowercase, strip everything that is not a letter, digit or space, collapse whitespace.

**Gold tier (the headline).** Every question lists `gold_ids`, the records that count as a correct retrieval. A question scores at k when a gold record sits in the top k of the window. Questions whose answer lives in several records list them all.

**Version tier.** For a question about a fact that changed, `must_rank_first` names the current record. The version tier asks whether the first gold record in the window is a current one. It counts an item only when every one of its gold records carries a `ts`, because without one nothing in the pool tells any system which record is current. The harness never invents a timestamp.

**Diagnostics.** Two string tiers stay in the output and are never the headline: exact (a returned record contains the answer as whole tokens) and family (60% token overlap with a gold record).

**Bands.** The window and the reported k come from the pool size N, so every system at the same N is judged the same way:

| Pool size N | Band | k reported | Headline k | Window |
| --- | --- | --- | --- | --- |
| under 100 | none | 1 | 1 | 5 (small pool, no standard report) |
| 100 to 999 | S | 1, 5, 10 | 5 | 10 |
| 1,000 to 9,999 | M | 1, 5, 10, 15 | 10 | 15 |
| 10,000 to 49,999 | L | 1, 5, 10, 15, 20 | 15 | 20 |
| 50,000 and up | XL | 1, 5, 10, 15, 20, 25 | 20 | 25 |

**Chance and skill.** Beside every recall figure the runner prints the chance a random retriever would score, using each question's own number of gold records (hypergeometric, 1 minus C(N-G, k) / C(N, k)), and skill = (recall - chance) / (1 - chance). A k is valid while mean chance is at or under 5%.

**Abstention.** Abstention questions have no gold. They are scored only when a system returns scores and declares a threshold: the question abstains correctly when the top score is under the threshold or the window is empty. Otherwise abstention is reported as not available and left out of every aggregate.

**Also in every results file:** per-category and per-difficulty tables, latency (p50, p95, max), calibration (AUROC of score against gold) when scores exist, how much of the window repeats an earlier hit, and every question's full window with ids, scores and ranks.

**Private sets with store ids.** A set built from a live store may keep the store's own stable ids: run and lint with `--store-ids`. The ids are then taken as given, the manifest records `id_scheme: store`, and the result is never a standard result. Public sets always use content ids.

## Verification and the manifest

Every results file carries a manifest: the harness version, its git commit and the SHA-256 of its own source files, the SHA-256 of the question file and the corpus, the adapter and its `version()`, the runtime, the band and window, and a hash of the per-question records with timing removed. `--verify` runs the whole benchmark twice in one invocation and compares those hashes. Equal means `verified=yes`. Unequal means `verified=unstable`, with the differing questions listed and the headline reported as a mean with its spread.

A result is a standard result only on a standard track, with a question set whose gates all pass, verified, with both extra passes run. The standard tracks are not built yet, so every result today prints `standard=no`, with the reasons.

### Comparing two results

```bash
parsica-recall compare before.json after.json
```

Two results are comparable only when they ran the same question set, on the same pool, under the same harness. Anything else is refused: the tool names what differs and exits 2. When the results match it prints the headline deltas, the per-question diff on the gold tier (won, lost, moved up, moved down, by question id), and a one-line summary:

```
compare recall/1.0 qset=v11@2e5b2c03 pool=file@cea80a56 N=10000 band=L k=15 a=bm25@1.0 b=bm25stop@1.0 gold@15=87.58->88.19(+0.61) skill@15=87.56->88.17 won=9 lost=5 up=65 down=50 same=523
```

A harness is pinned by its commit in a checkout, and by the hash of its source files in an installed copy, which has no commit. A result from a tree with uncommitted changes is not pinned and is refused; `--allow-unpinned` turns that into a printed note for local work and appends `unpinned=yes` to the summary line. The diff needs only ids and ranks, so redacted results compare the same way.

## Extra passes

Two cheap passes, optional in a local run and required for a standard result (spec section 7). Turn them on with `--passes order,query`. Each runs once, on a fresh adapter after the base run, so the base run's records do not change.

- **order** (file mode): the corpus is ingested a second time in a fixed shuffled order. It reports the share of answerable questions whose gold rank changed. An engine whose ranking does not depend on ingest order scores zero.
- **query** (file and live mode): every answerable question is searched in two deterministic variants. `casefold_nopunct` is the shared normalizer; `no_stopwords` is the question with a fixed list of stopwords taken out. It reports how often the gold rank agrees with the original.

On v11, plain BM25 changes none of 652 answerable questions on the order pass. On the query pass it agrees with itself on 89.11% of questions with `casefold_nopunct` and 81.44% with `no_stopwords`.

## Live mode

Live mode runs Recall against a store that already holds its records, through its real retrieval path. Nothing is ingested and nothing is written to the store.

```bash
parsica-recall run --live --adapter custom --custom-adapter my_store.MyStoreAdapter \
    --questions ~/recall-sets/my-store-v1/questions.json --verify --redact
```

- `setup(workspace, "live")` opens the store read only, and `ingest` is never called.
- `list_ids()` is the pool. N, the band, the window and the chance line all come from it, and the runner refuses a question set whose gold ids are not all in it. Without `list_ids()` the run still prints recall, labeled pool size unknown, and is never a standard result.
- The sorted id list is hashed at the start and the end of every run. If the hashes differ, records were added or removed during the run, and the result is marked `pool=changed` and `verified=no`. Run on a quiet store or a snapshot.
- Gold drift is reported, not punished: a gold record whose returned text no longer matches its `gold_hashes` entry still scores by id, and the run reports `drifted_gold_count`.
- The version tier reads the store's own timestamps through the optional `record_ts(ids)`. The same timestamps give the age breakdown: the gold hit rate by the age of each gold record.
- Results go to `recall_live_<adapter>.json` next to the question file, and the runner refuses to write live results anywhere under the folder the harness sits in, because they hold store content. `--redact` replaces every returned record text with its hash and drops the question, answer and alternate fields. Share only redacted results.
- A live result says `track=live`. It is compared with a track result only on skill, with N stated.

[docs/live_http_adapter.md](docs/live_http_adapter.md) is a complete read-only adapter for a store behind HTTP.

## The question set: v11

v11 is the first set written by the generation protocol (spec section 9) on the public pool: `generate-questions` over pool v1's 10,000 records, a model writing the questions, the lint deciding what freezes, no item edited by hand. Its frozen items were split by seed into two halves of 727 questions each. The public half is in `data/v11/`. The sealed half is not published.

`data/v11/` holds:

| File | What it is |
| --- | --- |
| `corpus.jsonl` | A byte copy of pool v1's 10,000 records, the pool the set was linted against |
| `questions.json` | 727 questions (652 answerable, 75 abstention) with `gold_ids`, `must_rank_first` and `gold_hashes` |
| `manifest.json` | File hashes, counts, the model's token counts and cost, the sampler and generation record, the lint result, the status, and the SHA-256 of the sealed half's `questions.json` |
| `lint_report.json` | The full lint report |
| `reference.json` | The oracle and BM25 result lines |
| `generation/prompts/` | The prompt files as sent, one per category |

How it was built, on October 22, 2026: targets of 150 per category, `claude-opus-5-5` as the writer, 2,431 calls, $38.02 by the call log. The lint found 0 items with an error, and the oracle scores 100.

Questions per category in the public half. The sealed half holds the other half of each, give or take one item:

| Category | Questions | Note |
| --- | --- | --- |
| direct_recall_user | 75 | |
| direct_recall_agent | 75 | |
| entity_attribution | 75 | |
| temporal_awareness | 60 | |
| paraphrase_recall | 75 | |
| cross_session | 58 | |
| long_horizon | 75 | |
| vocabulary_gap | 75 | |
| knowledge_update | 11 | under 25, not reported alone |
| correction_handling | 7 | under 25, not reported alone |
| implicit_preference | 51 | |
| abstention | 75 | |
| recency_preference | 15 | under 25, not reported alone |
| all | 727 | 232 with every gold record in the first 1,000 records |

**Status: interim.** One gate fails, `GATE_CATEGORY_MINIMUMS`, on the three thin categories. They count in every total and are not reported alone. A larger pool with more planted changes is what fills them.

**A known soft spot.** The pool file is ingested in file order, and its first 1,000 records come first. 232 of the 727 questions have all their gold there, so a system that favors what it ingested first has a small edge until track mode shuffles the order. The order pass shows whether a system depends on ingest order.

### Calibration on v11

Two reference adapters on the public half (N = 10,000, band L, window 20, headline k = 15), each run twice with identical records. They are calibration, not a leaderboard: they show where the floor and the ceiling sit.

| Adapter | gold@1 | gold@5 | gold@10 | gold@15 | skill@15 | MRR@20 | version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 59.66 | 82.36 | 85.89 | 87.58 | 87.56 | 0.690 | 48.48 |
| oracle | 91.10 | 100.0 | 100.0 | 100.0 | 100.0 | 0.956 | 100.0 |

Chance at k = 15 is 0.14%. The oracle is 100 from k = 5 on, the check that the set and the scorer agree; its gold@1 is under 100 because an item with more than one gold record cannot be complete at k = 1. Plain BM25 at 87.58 gold@15 is the floor a system has to clear to be interesting on this set.

### The sealed half

A sealed half exists: 727 questions over the same pool, split from the same build by seed. It is never published. The public manifest commits to it by hash, so when it is used, anyone holding the public half can check that it is the half that was committed to:

```
sha256 of the sealed questions.json: df4a9b7c596eeb5569355546e7e29681e458dc3cafddea95608d420cc11096b4
```

No sealed scoring is offered yet.

## The public pool

Pool v1 (`data/pool-v1/`) is a synthetic agent-memory store under MIT: 10,000 records. Eight invented people each live 6 to 18 months of sessions with their own AI agent, and the pool is the records an agent-memory store would keep from those sessions. Every person in it is invented. The design is in [docs/pool.md](docs/pool.md), the persona files are in [scenarios/](scenarios/), and the frozen prompts are in [prompts/pool/](prompts/pool/).

Every record has `id` (its content id), `text`, `kind` (`user`, `agent` or `entry`), `ts` and `source` (a session id). The first 1,000 records, the core, carry planted material for every category: version chains, corrections, preference changes, threads across sessions and long-horizon facts.

The pool was written by `claude-sonnet-5-5` in 1,090 calls, and every record was screened: credential and personal-data shapes, records too short to be gold, duplicates and near copies. A failing record was dropped, never edited. `data/pool-v1/generation/` holds the plan, where each record came from, every dropped record with its reason, and the prompt and persona files used. The two raw call logs of the build (the model's replies before screening) are not published; the pool's manifest keeps their SHA-256.

`parsica-recall generate-pool` writes a pool of your own from persona files, and `parsica-recall show-pool` prints one for reading.

## Lint and gates

```bash
parsica-recall lint --output lint_report.json
```

The linter judges each answer key against the whole pool, not just its own question. Error rules include: gold records present and unchanged, no record outside the gold set contains the answer, the answer is not in the question, no key starts or ends on a function word, no alternate matches more than 1% of the pool, gold records of at least 6 tokens, multi-record categories with at least two gold records, and no credential or personal-data shapes anywhere in the records or questions. A lint report says where such a shape sits and what kind it is, and never repeats the text.

Every release gate is code with a pass or a fail. Whether a question set may back a standard result comes from its gates and its manifest.

## What is not in this release

- **Standard tracks** (1k, 10k, 30k, 50k). The pool's first 10,000 records are here; a larger pool and the runner's track mode come later. Until then every result prints `standard=no`.
- **Sealed scoring.** The sealed half exists and is committed to by hash. Nobody is scoring against it yet.
- **Fullstack mode**, where an answer model reads the window. It is on the roadmap, and spec section 11 fixes its rules.
- **Other model providers for the question builder.** Today it runs on Claude through the Anthropic API.

## Files

- `parsica_recall/`: the harness. `run_benchmark.py` (the runner), `results_file.py` (the no-code path), `compare_results.py`, `adapters.py` (the adapter contract), `reference_adapters.py`, `recall_common.py` (the normalizer, record ids, bands and the chance model, shared by everything), `lint_validator.py`, `relint_set.py`, `generate_questions.py`, `generate_pool.py`, `export_planted_pairs.py`, `show_questions.py`, `show_pool.py`, and `cli.py` (the `parsica-recall` command)
- `parsica_recall_lint_spec.json`: the lint's rules and gates
- `data/v11/`: the current question set, public half
- `data/pool-v1/`: the public pool and its generation record
- `prompts/`: the question builder's frozen prompts, one per category; `prompts/pool/`: the pool generator's
- `scenarios/`: the pool's persona files
- `docs/spec.md`: the Recall 1.0 specification
- `docs/adapting.md`: running Recall on any memory system
- `docs/doc2query.md`: building a question set from your own store
- `docs/generation.md` and `docs/pool.md`: the design of the two generators
- `docs/live_http_adapter.md`: a read-only HTTP adapter for live mode
- `tests/`: `python -m unittest discover -s tests`
- `tools/check_dist.py` and `tools/dist_files.txt`: the check a built package has to pass before it is uploaded, against a list of every file it may hold

## License and contact

MIT. Copyright (c) 2026 Antaris Analytics LLC. See [LICENSE](LICENSE).

Questions and reports: Info@Parsica.ai, or an issue on this repository.

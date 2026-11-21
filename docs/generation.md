# Question generation

This is the design of the question generator (`parsica-recall generate-questions`, the module `parsica_recall/generate_questions.py`), the generator that spec section 9 describes as the generation protocol (the doc2query path). It takes any `corpus.jsonl`, whether an export of your own memory store or the public pool, and writes a frozen, linted question set with every prompt, seed and model response recorded. One tool serves a private set over a person's own store and a public set over the public pool alike.

The protocol itself does not change. This document says how the generator carries out its five steps. The prompts it sends are in [`prompts/`](../prompts/), one file per category.

## 1. What a run does

```
parsica-recall generate-questions --corpus export/corpus.jsonl --out ~/recall-sets/mystore-v1 \
    --seed 7 --targets 25 --max-cost 10 [--model claude-sonnet-5-5] [--store-ids]
```

The public pool-v1 set adds the gold window, the core and the split:

```
parsica-recall generate-questions --corpus data/pool-v1/corpus.jsonl --out data/v11 --release \
    --sealed-out sealed/v11-sealed --split-seed 2 --seed 2 --targets 150 --core-quota 40 \
    --gold-window 10000 --core 1000 --model claude-opus-5-5 --api-key-env MY_API_KEY --max-cost USD
```

`USD` stands for the full run's limit, which the maintainer sets after reading the pilot and the estimate made from it (section 14, decision 14); the command refuses to run until it is a number.

1. **Read and check the corpus.** It is loaded with the runner's own loader, so a corpus the runner would refuse is refused here too. Without `--store-ids` every id must be the content id of its text; with it, ids are taken as given (spec section 3). Two more checks come before any call: every text is a non-empty string, and every field the model would see is valid Unicode (a lone surrogate, often half of a cut emoji, cannot be sent).
2. **Sample** records per category, stratified by kind and by age, from a fixed seed, and never the whole store (section 3 below).
3. **Generate** candidate items per category: the category's frozen prompt plus a small batch of sampled records (or record pairs) goes to the model, which returns items in a fixed JSON schema (section 6).
4. **Check and lint.** Each candidate is checked against what the model was shown, then linted with `parsica_recall/lint_validator.py` against the whole corpus as the pool. Every candidate that fails is held or dropped. Nothing is edited into passing (sections 6 to 8).
5. **Freeze** `questions.json` with `gold_ids` and `gold_hashes`, copy the corpus beside it, lint the frozen file again and write `lint_report.json` and `manifest.json` (sections 8 and 9). With `--split-seed`, the frozen items are first split into a public half and a sealed half, and each half is written, linted and checked on its own (section 9).
6. **Reference runs.** The oracle and the reference BM25 adapter run in file mode with `--verify`, and `reference.json` records both. The oracle must score 100 on the gold tier at the band's window. If it does not, the run reports the set broken, names the questions the oracle missed, and exits 1 (section 10).

Before step 3 a live run needs `--max-cost`, and `--estimate` prints what the run is likely to cost without making any call (section 16).

The command line:

| Flag | Meaning |
| --- | --- |
| `--corpus` | The corpus file. Required. |
| `--out` | The directory the set is written to. Required. It must not exist yet, or must be empty, because a frozen set is never overwritten. Where it may be is in section 11. Its last path component becomes the set's version name (`qset_version`), so `--out ~/recall-sets/my-store-qset-v1` writes a set named `my-store-qset-v1`. |
| `--seed` | The sampler seed, an integer. Required, so every set's seed is a choice someone made. |
| `--targets` | Publishable items wanted per category: one integer for every category, or a list such as `direct_recall_user=5,abstention=3` (a category left out gets 0). Default 25, the standard minimum. |
| `--provider` | Who runs the model: `anthropic` (the default), `openai` or `google`; `openai` with `--base-url` is any endpoint that speaks OpenAI's chat format (section 18). |
| `--model` | The generation model. Default `claude-sonnet-5-5` on Anthropic, `gpt-6.2-sol` on OpenAI and `gemini-4-hydrogen` on Google; required with `--base-url`. The pool-v1 set is written by `claude-opus-5-5`, named on its command line. |
| `--base-url URL` | With `--provider openai`: the endpoint the calls go to in place of OpenAI's (section 18). Recorded. |
| `--price-in USD`, `--price-out USD` | The model's prices per million tokens, in place of the table's or for a model it does not list; cache reads and writes are counted at the input price (section 18). Recorded. |
| `--max-output-tokens N` | With `--base-url`: the cap sent as `max_tokens`; by default none is sent (section 18). |
| `--store-ids` | The corpus keeps its store's own ids (spec section 3). The lint, the manifest and the reference runs all follow it. |
| `--release` | Allows an `--out` under `data/`, for a public set the maintainers name, and a `--sealed-out` under `sealed/` (section 11). |
| `--replay` | Rebuilds a set from a recorded `generation/calls.jsonl` instead of calling the API (section 12). |
| `--gold-window N` | Gold comes only from the corpus file's first N records; default, the whole file. The lint and the abstention screen still read every record (section 3). |
| `--core N` | The corpus file's first N records are the core: every category takes its units from the core first, then from the rest of the gold window, and the manifest names the questions whose gold is all in the core (section 3). |
| `--core-quota N` | With `--core`: at most N items per category freeze from the core. A category's core stage ends once N of its core items have frozen, once it has shown half its unit cap, or once its core units run out, whichever comes first; then the rest of the window (section 3). |
| `--unit-cap CATEGORY=N,...` | A category's unit cap in place of 4 times its target: `knowledge_update=1200,implicit_preference=900` (section 3). |
| `--planted-pairs FILE` | A pool's record of its planted changes, as `parsica-recall export-planted-pairs` writes it: the version categories build their core units from these pairs of record ids, by type; correction_handling and recency_preference take nothing else, and knowledge_update goes on with similarity pairs past the core (section 5). |
| `--split-seed N` | Splits the frozen items, seeded and stratified, into a public half (`--out`) and a sealed half (`--sealed-out`) (section 9). |
| `--sealed-out` | The folder of the sealed half, which also keeps the generation record. Its last path component is the sealed half's own set name. Under `sealed/` with `--release`; never published (section 11). |
| `--max-cost USD` | The most the set may cost, calls a `--resume` replays included. The run stops before its next new call once the cost passes it, with exit 3. Required for a live run (section 16). |
| `--estimate` | Prints, per category and per stage, the items the run is expected to freeze, its calls and cost, and the worst case, and stops before any call (section 16). |
| `--estimate-from DIR [DIR ...]` | With `--estimate`: uses the tokens and yields one or more finished pilots measured, stage by stage, from set folders that hold their generation record; a later folder's measurements replace an earlier one's, category by category (section 16). |
| `--resume` | Picks up a stopped run from its own call log; the calls it holds replay free (section 12). |
| `--check-api` | Makes one tiny call to prove the key and the network, and prints `ok` with the model that answered, or the error's class name; nothing else. Needs no other flag, except `--api-key-env` when the key is in another variable. |
| `--api-key-env NAME` | Reads the API key from this one environment variable instead of the provider's own (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`), and sends every call to the provider's official endpoint whatever its base-URL variable says (sections 2 and 18). With `--base-url`, the only way to give that endpoint a key. Not recorded. |

The exit status is 0 when the set was written and the oracle scored 100 on it (or `--estimate` printed, or `--check-api` said ok), 1 when the oracle check failed, 2 on input the generator refuses or a model call that failed, and 3 when `--max-cost` stopped the run, which a `--resume` continues.

## 2. What leaves the machine

Generation is the one step that uses a model, and the model is a remote API: Anthropic's by default, or OpenAI's, Google's Gemini API, or an endpoint named with `--base-url` (section 18, which also covers each provider's key). What is sent: the frozen prompt and the text of the sampled records, with their `id`, `kind`, `ts` and `source` when present. Nothing else from the store is sent, and the run never sends the whole store (section 3). For a private store, that still means personal records go to the API, and the README says so plainly.

Records that match a credential or personal-data shape (the lint's own scanner: API keys, tokens, connection strings, email addresses, phone numbers, street addresses, government ids, card numbers) are never sampled and never sent. The scan covers every field the model would see: the text, `kind` and `source`, a `ts` that is not a readable timestamp, and the `id`. An id may look random, because store ids often do, so the scanner's random-token shape alone does not exclude a record by its id; every other shape does. These records stay in the pool, where they are distractors like any other record.

The API key comes from `ANTHROPIC_API_KEY` through the official `anthropic` Python SDK. The generator never reads it itself, and it never appears in output, logs, the manifest, the generation record or an error message. Or, with `--api-key-env NAME`, it comes from the one variable `NAME`, exactly as for the pool generator (docs/pool.md section 2), whose code it shares: the generator reads it when it makes the client, hands it to the SDK as the key, pins the API address to `https://api.anthropic.com` whatever `ANTHROPIC_BASE_URL` says, refuses a value with a space, a line break or a quote inside, and takes the key out of every error it reports, dropping the exception it came from. The flag is not recorded, so a resume may name another variable. The SDK is needed only for a live run. A live run loads it and makes the client before writing anything, and nothing is written until the first reply is in hand. So a missing package, a missing key, or a key the API rejects on the first call leaves no folder behind, and neither does a run in which no category had anything to send. The tests, `--replay` and the rest of the harness never load the SDK. If a later call fails partway through a run, the folder keeps the calls made so far, and the error says it is not a question set and has to be removed before the next run.

## 3. The sampler

**Eligible records.** A record can be shown to the model when its text has at least 6 normalized tokens (the lint's `source_length` minimum, so every gold record will pass it) and no field the model would see matches a credential or personal-data shape (section 2). Ineligible records stay in the pool.

**The gold window.** With `--gold-window N`, only the corpus file's first N records can be shown, so gold comes only from them (spec section 6: the pool-v1 set takes its gold from the first 10,000 records). A near-copy cluster with a member past the window is out whole, as one with an ineligible member is. Records past the window are never shown and stay in the pool: the lint, the abstention screen, difficulty and the reference runs read every record of the file. The age bands, the strata and the sample cap are taken over the window. The manifest records the window and how many records sit past it.

**Core first.** With `--core N`, the corpus file's first N records are the core (pool-v1's 1,000, the 1k track). Every category's units come in two stages, each built from its own records alone: first the core, then the rest of the gold window. Records, pairs and topics of the first stage are all core records, and those of the second are all past the core, so every unit sits wholly in one stage. A category takes units from the core until the core runs out or its target is met, and only then reaches past it, so the core gives each category what it can before the rest of the window does. The unit cap covers both stages, and `max_sample_share` 0.5 applies to the whole window.

**The core quota.** With `--core-quota N` (40 for the pool-v1 set, October 21, 2026, after a review), a category's core stage ends at the first of three points: N of its items from the core have frozen, it has shown half its unit cap, or its core units run out. Then it fills from the rest of the window, up to the whole unit cap. A core item that passes the checks and the lint after the quota is met is recorded as surplus (`the core quota is met`) and never freezes, and a held core item gets no retry once the quota is met, so at most N items per category come from the core. The core stage's units are drawn up to half the cap, and the tail stage's up to the whole cap, since the core stage may end early. Without a quota, the reach of the core is only its own units and the target, which on pool v1 would give the core nearly every item: the review measured that every single-record category would plan all 600 of its units in the core. The quota keeps the 1k track at about 400 to 600 questions (13 categories at 40) and puts the rest of the set past the core. `run.json` and the manifest record it. The second stage's draw orders are seeded like the first, with ` after the core` added to the stream's name. The manifest counts, per category, the units shown from the core and the frozen items whose gold is all in the core or not (`frozen_core`, `frozen_tail`), and its `core_subset` names the questions whose gold (for abstention, whose topic) is all in the core: the set's part on the 1k track. Without `--core` there is one stage and the orders are the ones a set made before it would draw.

**Strata.** Every eligible record falls in one stratum, the pair (kind, age band):

- **Kind** is the record's `kind` field, or `none` when it has none.
- **Age band** comes from `ts`, read with the harness's own `parse_ts`. The records with a readable `ts` are sorted by time (ties by id) and cut into four bands of equal size: oldest, older, newer, newest. Records without one form a fifth band, `no ts`. The bands come from the times in the file, never from the clock, so the same corpus gives the same bands on any day. A corpus with no `ts` at all has one age band, and its strata are its kinds.

**Seed and order.** Each category draws from its own order of the eligible records. That order is a seeded permutation, made with Python's `random.Random`, seeded with a string built from `--seed`, the corpus file's SHA-256 and the category name. The same seed on the same corpus always gives the same orders. Two categories draw in different orders, and the same seed on another corpus draws differently. Within a stratum the ids are sorted before they are shuffled. The strata are interleaved in proportion to their sizes: at each step the stratum furthest behind its share goes next (ties by the stratum's name, in integer arithmetic), so every stretch of the order stays close to the store's mix of kinds and ages.

**Batches and stopping.** The generator takes units from a category's order in batches, sends each batch to the model, checks and lints what comes back, and stops when the category has its target of publishable items, or reaches one of these caps:

- **Units per category:** at most 4 times the target, and at least 12, or the cap `--unit-cap` names for the category (`knowledge_update=1200,implicit_preference=900`), recorded in `run.json`, so a resume keeps it, and in the manifest. A unit is a record, a pair or a topic, depending on the category. The sample cap below still holds whatever the unit caps say, and the estimate prints how many records the planned units would show against it (section 16).
- **Never the whole store:** across all categories together, at most half of the eligible memories are ever shown to the model (`max_sample_share` 0.5). A memory is one near-copy cluster (below), since a copy is never shown on its own, so a store that holds every note twice still shows at most half of its notes. The categories take one batch each per round. At the start of each round, what is left of the cap is split evenly over the categories still running, and any remainder goes one record each to the first of them in the harness's category order. That way a small corpus with big targets cuts every category short alike, not just the ones last in line. A record another category already showed costs nothing.
- **Waiting and the last of the cap:** a category whose next unit needs more new records than its share of the round (a pair needs two, a topic up to four) waits for a later round instead of ending. After a round in which no category could take anything, the next round offers all of what is left to each category in turn, first come first served, so the last records of the cap go to a category that can use them. If that round moves nothing either, the run stops, and the categories still short are listed in the manifest as stopped by the sample cap.

Batch sizes are 6 records for single-record categories, 3 pairs for pair categories, and one topic of up to 4 records for abstention.

**Recorded.** `generation/sample.json` lists every unit shown to the model, per category and in order, with each record's stratum. The manifest records the seed, the sample size (distinct records shown to the model), the eligible and excluded counts, the strata sizes, the caps, and the file's SHA-256.

**Near copies.** The same memory often sits in a store twice: a comment and its quoted copy, a note and its enriched version. Two records are near copies when all three of these hold:

- each covers at least 90% of the other's tokens, which is the lint's `gold_near_copy` threshold;
- they carry the same set of digit tokens;
- they carry the same set of words written with a capital first letter.

A copy keeps the numbers and names, and an update changes them. So a long record in which one number or name changed is a candidate for a version pair, not a copy.

The sampler groups near copies into clusters, the connected groups of the relation, and shows the model one record per cluster: the one that comes first in the corpus file. Every item from that record lists the whole cluster in `gold_ids`, by a rule fixed before any model call. That is what an earlier set's curation did by hand ("the same calendar entry with its prefix added to gold"). In the version categories the current record's copies all go into `must_rank_first`, which may be a list for exactly this case (spec section 3). A cluster with one ineligible member is left out whole, so nothing ineligible becomes gold. This rule belongs to the sampler; the lint's `gold_near_copy` warning is unchanged.

## 4. Which records each category sees

| Category | Unit | Drawn from | Needs |
| --- | --- | --- | --- |
| direct_recall_user | one record | all eligible records | |
| direct_recall_agent | one record | all eligible records | |
| entity_attribution | one record | records that hold a rare name (below) | |
| temporal_awareness | one record | all eligible records | |
| paraphrase_recall | one record | all eligible records | |
| vocabulary_gap | one record | all eligible records | |
| implicit_preference | one record | all eligible records | |
| long_horizon | one record | the oldest age band only | `ts` on at least 8 records |
| knowledge_update | an earlier and a later record | version pairs | `ts` on both records |
| correction_handling | an earlier and a later record | version pairs | `ts` on both records |
| recency_preference | an earlier and a later record | version pairs | `ts` on both records |
| cross_session | two records from different sessions | session pairs | `source` on both records |
| abstention | a topic: one record and up to 3 similar ones | all eligible records | |

The single-record categories all draw from the whole store, except entity_attribution. Their prompts tell the model to skip a record that does not fit, so a forum post produces no direct_recall_agent item, for example. Skipping is normal and is counted.

**Rare names.** entity_attribution draws only from records that hold a rare name (October 22, 2026, after the v11 pilot, whose entity answers were recurring names, Kroger in 192 other records among them, so 19 of its 23 items leaked). A name is a capitalized word of three or more characters that does not start a sentence, or a run of two or more capitalized words ("Dana Whitfield", "Rillito Valley Vet"; a run may start a sentence, and then drops a leading function word such as "Then"), read from the record's text before normalization; a possessive counts as its stem ("Kennelbook's" is Kennelbook, rare only when Kennelbook is). It is rare when, as whole tokens, it appears in no other record of the whole pool outside the record's own near-copy cluster (`RARE_NAME_MAX_OTHERS`, 0). On pool v1, 890 of the 9,996 eligible memories hold a rare name, 213 of them in the core; with one other record allowed it would be 1,546 and 303. The manifest records the rule and the count, and a corpus in which no record qualifies gives the category no units and says why. The lint is unchanged: `GOLD_MULTIPLICITY` still decides whether the answer the model picked leaks, and `entity_coherence` still warns on a capitalized word in the question that no gold record holds.

A corpus without `ts` cannot tell an earlier record from a later one, or an old one from a new one, so it gets no long_horizon and no version-category items. A corpus without `source` cannot tell two sessions apart, so it gets no cross_session items. The manifest gives the reason for every category the generator could not build, and the target counts as unmet. The harness never invents a timestamp, and neither does the generator.

## 5. Record pairs and topics

**Topic similarity** is one fixed, model-free measure:

- A record's content tokens are its normalized tokens minus the harness's `STOPWORDS`.
- Each token is weighted by its inverse document frequency over the whole pool, with the reference BM25's formula: idf = ln(1 + (N - df + 0.5) / (df + 0.5)).
- The similarity of two records is the summed weight of the content tokens they share, divided by the smaller of their two total weights. That gives 0 for no shared words and 1 when one record's words are all in the other.
- Two records are on one topic when their similarity is at least 0.3 (`pair_min_similarity`, recorded) and they are not near copies, since a copy is the same memory, not an update.
- To find candidates quickly, words that appear in more than 5% of the pool (and in at least 100 records) are skipped when gathering them. The similarity itself still counts every shared word.

**Version pairs** (knowledge_update, correction_handling, recency_preference) are built as follows.

- **Finding a partner.** For each anchor in the shared pair order, the partner is the most similar eligible record that is on the same topic and has a readable `ts` different from the anchor's (ties broken by id). The record with the earlier `ts` is the earlier record.
- **Dealing pairs.** One seeded pair order serves all three categories. Pairs are dealt to them in turn, so a pair is offered to one category only and never yields two items.
- **The model's call.** The model sees the pair as "earlier" and "later" with both times. It writes an item only when the later record updates (knowledge_update), corrects (correction_handling) or changes a preference in (recency_preference) the earlier one, and skips the pair otherwise.
- **Checks on the item.** `gold_ids` must list both records, and `must_rank_first` must name the later one. An item that says otherwise is dropped.

**Planted pairs** (`--planted-pairs FILE`; October 22, 2026). A synthetic pool whose generator recorded the changes it planted may supply them, and then the version categories take their units from that record instead of from similarity. On the v11 pilot the similarity partner was the planted partner for only 40% of the corrections, 12% of the preference changes and 44% of the version steps, and the pairs were dealt to the three categories in turn, so two thirds of the real changes landed in the wrong category. For pool v1 the file is `data/pool-v1/planted_pairs.jsonl`, written by `parsica-recall export-planted-pairs` from the pool's own generation record alone: the call log's `state_update` events (one per planted step, with its label and the position of the record that carries it) mapped to record ids through `generation/records.jsonl`, under the rules the pool generator used to fold a step into its state (the label was asked for in that session, the record was kept by the screens, the text holds no credential or personal-data shape, and the first event per label in a session counts; docs/pool.md section 4). It never reads the persona files. One line per planted item, `{"type", "label", "persona", "earlier", "later"}`, holding only the item's last transition (the step before the last, and the last), so the later record carries the current value and nothing later in the pool changes it; an item with one kept step has no transition and is left out. On pool v1 that is 137 pairs, 50 version chains, 48 corrections and 39 preference changes, every one in the core. The exporter is deterministic, and a test checks the committed file against the pool's own final state (`generation/states.json`).

The generator routes the pairs by type: `version_chain` to knowledge_update, `correction` to correction_handling and `preference_change` to recency_preference, each category's pairs in a seeded order drawn like records by their later records. A pair is used when both records are eligible memories of the gold window (each standing for its near-copy cluster), are two different memories, carry a `ts` with the earlier one before the later, and no earlier line gave the same pair or the same later record; the rest are counted by reason. With a core, the core stage takes planted pairs only. Past the core, knowledge_update alone goes on with similarity pairs, since the tail's updates are not recorded, while correction_handling and recency_preference take nothing else: a set holds at most as many of those as the pool planted and the core quota allows (about 40 each for pool v1, since every planted pair sits in the core), and more needs planted changes in the pool itself, which a later pool round adds. Without a core there is one stage, treated like the one past the core. The manifest records the file's name and SHA-256, the pairs read per type, used per category and left out by reason, and `run.json` records the hash, so a resume keeps the file. A user's store has no such record and keeps the similarity pairs above.

**Session pairs** (cross_session) are built the same way, from records whose `source` values differ. The spec's corpus format has `source` as the place a record came from, and the generator reads it as the session. A store that keeps its session id somewhere else maps it to `source` when it exports. The model writes an item only when one question naturally needs both records: the question names the subject through a detail found in only one record, and the answer is a span found in only the other. The question is scored on `gold_all_rank` (spec section 5).

**Topics** (abstention) are an anchor plus its up to three most similar eligible records on the same topic, so the model sees what the store holds on a subject before it writes a question the store cannot answer.

## 6. Generation calls

- **Model.** `--model`, default `claude-sonnet-5-5`. That is the model's full API id; current ids carry no date suffix, so there is no dated snapshot to pin instead. The manifest records three things: the id requested, the id the API reports serving each call, and when the first request went out and the last reply arrived, in UTC, since an API model can change under one name (spec section 11 says the same of the answer model). Server-side fallbacks are off, so every item comes from the pinned model. A refusal yields no items and is counted.
- **Request.** The system prompt is the category's prompt file, byte for byte, with its line endings read as LF (section 12). The user message is the batch, one JSON object per record, holding the fields the record has of `id`, `kind`, `ts`, `source` and `text`. Pairs are labeled earlier and later, or first and second for cross_session. The response format is the category's JSON schema, passed as structured output (`output_config.format`), so the reply is always one JSON object of that shape. The schema is the block between `BEGIN SCHEMA` and `END SCHEMA` in the prompt file, so the file's hash covers both the prompt and the schema. The hash is the SHA-256 of the file's bytes with CRLF line endings read as LF, so a checkout that stores the file with CRLF gives the same hash.
- **Settings.** Effort is set to `high` on every call, so the manifest can say what ran (Sonnet 5.5's default; Opus 5.5's own default is `medium`).
- **Prompt caching.** The system prompt goes as one block with `cache_control` `ephemeral` and a `ttl` of `1h`, the 1-hour cache (`SYSTEM_CACHE` in the code). A category's prompt is the same on every one of its calls, a few thousand tokens (about 3,300 at the 2.36 characters per token pool v1's calls measured, which may count prose high; the pilot measures it): a category's first call writes its prompt to the cache, at the 1-hour write price, and its later calls and retries read it, at the read price (for `claude-opus-5-5`, $8 and $0.20 per million tokens, against $4 for plain input). A read starts the entry's hour again, counted from the start of the request. The calls go one at a time and the categories take turns, so a prompt waits a whole round, every other category's call and retries, for its category's next turn. With 13 categories that round can run past five minutes, and a 5-minute entry would then be gone at nearly every turn, each call paying a new write at $5 where a read costs $0.20, more than no cache at all. The hour costs at most about 13 cents more over a run (13 first writes at $8 instead of $5) and holds however long a round takes (October 21, 2026). Opus 5.5 caches a prefix of 512 tokens or more. The request's SHA-256 covers the cache setting, its `ttl` included, so a set made with another setting never replays as one made with this one. Every call records `cache_creation_input_tokens`, the part of it the API reports writing to the 1-hour cache (`cache_creation_1h_input_tokens`; a reply that reports no split wrote what the request asked for), and `cache_read_input_tokens`; the progress line prints the call's cache read, and the manifest's cost counts each at its price. A run whose cache reads stay at 0 from a category's second call on is paying for every prompt in full; the bill shows it, and so does the progress line. Thinking is left at the model's default and `max_tokens` is 16000. There is no temperature setting, because current models reject sampling settings. The model's answers therefore cannot be made repeatable, and reproducibility comes from recording them instead (section 12).
- **Record of every call.** Each call goes into `generation/calls.jsonl`: category, the candidate a retry is for (`retry_of`, null on a first call), the unit ids, the user message, the request's SHA-256, the response text, the stop reason, the served model id, the token counts (input, output, prompt-cache writes with their 1-hour part, and reads), and the times the request went out and the reply arrived. A call that ends in a refusal, at `max_tokens` or with unreadable JSON yields no items and is counted. The SDK retries rate limits and server errors on its own.
- **One retry for a `GOLD_MULTIPLICITY` hold.** When the lint holds an item and every error it found is a `GOLD_MULTIPLICITY` leak (the answer or an alternate also appears in a record outside gold), the item goes back to the model once, in a call of its own, while its category is still short of its target. The user message starts with the line `RETRY`, then `held:` and the item as the model returned it, then `why:` and the words that leak with how many other records hold each (`the answer "stays in cabinet" also appears in 11 other records of the store`), then the record or pair the item came from, exactly as it was first shown. No other record is shown. The category's prompt says what to do with it (its section A SECOND TRY): write about a different, more distinctive fact of the same record or pair, never a longer quote of the held answer, since adding the words around it does not make it pick out the record; in the version categories the answer must still be the new value, so the retry writes it only when the later record gives that value in another form that does not contain the held answer (a different spelling, an abbreviation, a number written out), and returns nothing when the new value cannot stand on its own; it never invites a fuller name or an added unit, which the padded-retry check below would drop. The prompts' "specific enough" line says the same: pick a distinctive fact, never lengthen a span with neighbouring words. What comes back is checked and linted like any item, with one check of its own first: a retried item whose answer holds the held answer as a strict sub-span is dropped as a padded retry (`$12.95` padded to `$12.95 each, which I just confirmed`), and the manifest counts them. On the v11 pilot 31 of 93 items froze after a retry, and about 14 of those had padded the held answer, which the old line about "a longer span" invited (October 22, 2026). One that is held again is not retried again, and an item held for any other error gets no retry. The retry is recorded like any call, and the manifest counts retry calls and the items they froze.
- **Checks before the lint.** These are mechanical checks on what came back:
  - the item has exactly the fields of its category's schema;
  - every gold id is one of the records shown in that call;
  - an abstention item has no gold ids, and every other item has at least one;
  - a pair item lists exactly one pair's two records, and in the version categories its `must_rank_first` names the later record;
  - `review_notes` is empty;
  - there is at most one item per record or pair, and two per abstention topic;
  - the question is not the same text as a frozen item's, since the oracle looks questions up by text (a held or dropped item never ships, so its text stays free);
  - a retried item's answer does not hold the held item's answer as a strict sub-span (a padded retry, above).

  Then the prompts' own rules that a machine can check:
  - the question ends with a question mark (the full-width and Arabic marks count) and has at most 30 words;
  - one question per item: no second question mark, and no second question joined on with "and who", "and what", "and which", "and when", "and where", "and why" or "and how";
  - outside abstention, the answer has 1 to 8 words (1 to 6 for vocabulary_gap, 2 to 8 for implicit_preference), and no acceptable answer is longer than that limit;
  - outside abstention, when the answer is a number with a unit (one of the lint's unit words or symbols, or a unit of time), every acceptable answer that has a number keeps a unit: "22 lbs" for "22 pounds" passes, "22" does not;
  - in the version categories, neither the answer nor an acceptable answer appears in the earlier record or its near copies;
  - in cross_session, the answer does not appear in both records, since one record would then answer the question alone;
  - in cross_session, the gold records, near copies included, all fit the band's window, since an item scored on `gold_all_rank` that cannot fit is one even the oracle would miss.

  A candidate that fails a check is dropped, with the reason recorded.

**Fields the model does not return.** The generator fills them in by fixed rules:

- `id`: `q_0001` onward, in the harness's category order, then in generation order.
- `category`: from the prompt.
- `answer_type`: from the answer and the category. Abstention items get `abstention`. An entity_attribution answer gets `role` when it is one of the lint's role labels and `entity` otherwise. Elsewhere the first rule that fits wins, in this order:
  1. an answer that starts with `=` is a `formula`;
  2. a version number with a `v` or at least two dots (`v3.11`, `1.2.3`) is a `version`, while a plain decimal such as `3.11` is not;
  3. for an answer with a digit: a recognizable date (a year, a numeric date, or a month with a number) is a `date`; a time of day such as `10:30` is a `phrase`; a number right before a time unit is a `duration` (`4 months`, `30-minute`; `second transect` is not); digits with one of the lint's unit words are `quantity_with_unit`; and an answer that is all digits once punctuation is removed is a `number`;
  4. code-shaped text (brackets, an underscore between words, a leading flag dash, dots between words) is a `code_symbol`;
  5. everything else is a `phrase`.
- `difficulty`: from where the reference BM25 adapter ranks the gold over the whole pool, searched with the question at the band's window:
  - first is `easy`;
  - inside the window is `medium`;
  - outside it is `hard`;
  - abstention items are `medium`.

  This makes difficulty a measured, model-free property of the item and the pool. It is computed at the freeze. Until then a candidate carries a placeholder, and no lint rule reads difficulty beyond checking its value.
- `gold_hashes`: the `text_hash` of each gold record's exact text.
- `review_status` is `publish`, `review_notes` is empty, and `must_rank_first` is null outside the version categories.

## 7. Abstention items

The model sees a topic and writes up to two questions that a person using this store could plausibly ask on that subject, whose answer none of the records holds. It asks for a detail the records leave out: a brand, a number, a date, a name, a reason. In `answer` it names the specific value it would expect such an answer to be, and in `acceptable_answers` up to five other likely values or spellings. `gold_ids` is empty.

**The screen** runs against the whole pool, every record in the corpus file, with the same token-boundary match the lint and the exact tier use. A candidate is dropped when:

- the answer or any alternate appears in any record, since the store might hold the answer after all;
- the answer or an alternate is longer than 6 normalized tokens, so a long, oddly specific value cannot pass the screen just by being long;
- the question shares fewer than two content tokens with the topic's records, so a system cannot abstain just because the question is off topic.

An empty acceptable answer says nothing and is not screened.

The screen proves that the values the model expected are absent from the store. It cannot prove that no answer in any words exists, so the prompt shows the model the topic's records and asks it to list the values a real answer would most likely contain.

**What freezes.** The frozen item takes a fixed form: answer `not mentioned`, acceptable answers `not mentioned`, `not specified`, `no information`, `not discussed` and `unknown`, `answer_type` `abstention`, `gold_ids` empty. The screened values stay in `generation/candidates.jsonl` with the screen's result, where anyone can rerun the check.

## 8. Lint, hold and drop

1. **First pass.** Every candidate is linted against the whole corpus with `lint_validator.lint`, using the set's own version name, the repository README and `--store-ids` when set.
2. **Publishable items.** A candidate is publishable when the lint says so: its proposed status is `publish` and it carries no error.
3. **Everything else** is held when the lint proposes `rewrite`, `reclassify` or `hold_experimental`, and dropped when it proposes `drop`. Held and dropped items both go to `generation/candidates.jsonl` with their findings, and neither ships. No item is edited, and `review_notes` stays empty on every generated item, so a note in a question set always means a person wrote it. An item held only on `GOLD_MULTIPLICITY` leaks gets its one retry (section 6): the model writes a new item, which is a new candidate (its `retry_of` names the held one), and the held item stays held.
4. **What freezes.** Per category, the first `target` publishable items in generation order are frozen. Any further publishable items are recorded as surplus. No item is picked by hand. When a publishable candidate has the same gold set and answer as a frozen item, the frozen item keeps the fact and the later one is held for reclassification, as the lint's `duplicate_fact` rule does over a whole file. Only frozen items count here: a surplus, held or dropped item never ships, so it takes no fact from a later candidate.
5. **Second pass.** The frozen `questions.json` is linted again, and that report ships as `lint_report.json`. Dropping items cannot create a new item finding or a new duplicate, so every frozen item stays publishable, while the gates now describe the frozen set.

The manifest gives per category the target and, in order: units shown, candidates returned, records the model skipped, candidates dropped by the checks or the abstention screen, held by the lint, dropped by the lint, publishable, frozen and surplus.

## 9. The frozen set and its manifest

`--out` holds:

| File | What |
| --- | --- |
| `questions.json` | The frozen question set. |
| `corpus.jsonl` | A byte copy of `--corpus`, so the set is self-contained and its hash in the manifest is the hash of the corpus it was linted against. |
| `lint_report.json` | The lint of the frozen set. |
| `manifest.json` | Below. |
| `reference.json` | The oracle and BM25 runs. |
| `generation/prompts/` | The prompt files used, as hashed and sent, with LF line endings. |
| `generation/sample.json` | Every unit shown to the model, per category, in order, and the units each retry showed again. |
| `generation/calls.jsonl` | Every model call, retries included. |
| `generation/candidates.jsonl` | Every candidate with its fate and the reason: frozen, surplus, held, dropped; for a split set, which half each frozen item went to. |
| `generation/run.json` | The settings a `--resume` checks (section 12). |

`manifest.json` keeps the shape the runner reads (`qset_version`, `status`, `files`, `lint`) and adds the generation record:

| Field | Value |
| --- | --- |
| `qset_version`, `spec`, `id_scheme` | The set's name, `recall/1.0`, and `content` or `store`. |
| `status`, `standard_eligible`, `status_reason` | Below. |
| `files` | SHA-256 of `questions.json` and `corpus.jsonl`. The runner checks both. |
| `counts` | Questions, answerable, abstention, records, and the count per category. |
| `core_subset` | With `--core`: the core's size, the rule, and the questions whose gold (for abstention, whose topic) is all in the core, by count, by category and by id. These are the set's questions on the track of the core's size. |
| `targets` | The target per category, as asked. |
| `model` | The id requested, the ids served, effort, `max_tokens`, the prompt cache setting (`system_cache`), server-side fallbacks (off), the SDK version, when the first request went out and the last reply arrived (UTC), the number of calls and of retry calls, total input, output, cache-write (with its 1-hour part) and cache-read tokens, and the cost at the prices in `PRICES` (section 16), with a note that the bill is the authority. A set built on OpenAI, on Google or at an endpoint of your own adds `provider`, `endpoint`, `reasoning_tokens` and `prices`; a set priced with `--price-in` and `--price-out` adds `prices` (section 18). |
| `prompts` | Per category, the prompt file and its SHA-256. |
| `sampler` | Seed; sample size (distinct records shown); pool size; the gold window (its size, how many records sit past it); with `--core`, the core's size and its eligible memories; eligible memories and records, and excluded counts by reason; the near-copy clusters; strata and their sizes; how many records carry `ts` and `source`; the caps, with what the sample cap counts; `pair_min_similarity`; the near-copy rule; batch sizes; and `sample.json` with its SHA-256. |
| `generation` | Per category, the counts listed in section 8, the retry calls and the items they froze, and with `--core` the units shown from the core and the frozen items whose gold is all in it or not, plus the reason when a category could not be built. Also the retry rule, and `calls.jsonl`, `candidates.jsonl` and `run.json` with their SHA-256. |
| `split` | With `--split-seed`: below. |
| `lint` | The report file and its SHA-256, the lint spec version, items with errors, `gates_passed`, `failed_gates`, and every gate with its result. |
| `tool` | The generator file's SHA-256, the harness commit, and whether the checkout had uncommitted changes. |
| `provenance` | One sentence: who wrote the questions (the model id), from what (the seed and the prompts), and that no item was edited by hand. |

**The split and the sealed half.** With `--split-seed` and `--sealed-out`, the frozen items are split once the lint has decided what freezes and the difficulties are set. The strata are the pairs (category, core or tail), where core means every gold record (for abstention, every topic record) sits in the core; without `--core` the strata are the categories. Within each stratum the frozen items, in generation order, are shuffled by a generator seeded with the split seed and the stratum's name, and the first half is public. A stratum with an odd count gives its extra item to the public half and the sealed half in turn, so each stratum's halves differ by at most one item and the two halves by at most one overall. The same frozen items and the same split seed always give the same halves. Each half is a set of its own: its own `questions.json`, numbered from `q_0001`, a byte copy of the corpus, its own lint report and manifest, its own reference runs and oracle check, and its own `core_subset`. The sealed half's set name is its folder's name, which must differ from the public half's.

The generation record (`sample.json`, `calls.jsonl`, `candidates.jsonl`, `run.json`) would show the sealed questions, so it lives in the sealed half's folder only; the call log is written there from the first reply. The public half carries the prompt files and no generation record, and its manifest says where the record is held. In the public manifest the `split` block gives the seed, the strata and the rule, and of the sealed half only the SHA-256 of its `questions.json`, as a commitment: when the sealed half is used, anyone holding the public half can check that it is the half the commitment named. The sealed manifest's `split` block names the public half and the SHA-256 of its `questions.json`. A replay of the sealed half's call log rebuilds both halves byte for byte.

**The sealed half is never published.** The maintainers keep it outside the public repository, and it never appears in a public file.

**Status.** A set is `interim` when any category has fewer than 25 publishable items or any gate fails. That covers every pilot and every set with small targets. It is `standard` only when every gate passes. One gate, `GATE_VERSIONS_MATCH`, compares the set's version with the harness's current set (`v11` since October 22, 2026; a set's sealed half, named `<set>-sealed`, counts as the same set). So a new public set is `interim` until the release that makes it current, which re-runs the lint on both halves with `parsica-recall relint` and rewrites their lint reports and the lint block and status of their manifests in the same change. A private set keeps failing that gate, which is right, because a private set is never a standard result (spec section 3). If the oracle check fails, the status is `broken`.

## 10. Reference runs and the oracle check

With the manifest written, the oracle and the reference BM25 adapter run through `run_benchmark` in file mode with `--verify` (and `--store-ids` when set), the oracle reading the new `questions.json`. `reference.json` holds each run's result line, verification, `records_hash`, and gold, skill and chance at every k.

A split set runs both on each half, and each half's folder gets its own `reference.json`; the run exits 1 when the oracle misses in either.

The oracle must score 100 on the gold tier at the band's window, the largest k, where every gold record of a multi-record item fits. (At k = 1 a cross_session item with two gold records cannot be complete even for the oracle, since it is scored on `gold_all_rank`.) If it does not, the run prints `set broken`, names every question the oracle missed and why (a gold id missing from the pool, a duplicated question text), rewrites the manifest's status to `broken` with those ids, and exits 1. The BM25 result line is printed as the floor a system has to clear.

## 11. Where a set may be written

A question set from a personal store holds personal facts, so `--out` may not be inside this repository, wherever the path points. The path is resolved, symlinks included, before the check, as the runner does for live results. The one exception is a public set: a path under `data/` together with `--release`, for a set the maintainers name. `--release` with a path outside the repository is refused, so the flag only ever means that. The same rule holds for `--sealed-out`, whose one place inside the repository is under `sealed/`, with `--release`; with `--release` it may not be anywhere else. The two folders must be separate, neither inside the other.

## 12. What is deterministic

Everything except the model's answers. Given the same corpus, seed, targets, prompt files, lint spec, generator and set name, and the same model responses, the generator writes the same files byte for byte, the manifest included, with one exception: the manifest's `tool.harness_commit` and `tool.dirty` describe the checkout that ran it. The same generator on another commit, or in a checkout with uncommitted changes, writes those two values differently and everything else the same. The manifest's times and token counts come from the call log, not from the clock. The responses are all in `generation/calls.jsonl`, so `--replay generation/calls.jsonl` rebuilds a set from its recorded responses without calling the API, and anyone can check that a frozen set follows from its record. The tests drive the same path with canned responses. Times and token counts never appear in the question file or the lint report.

**`--resume`** picks up a run that stopped (at `--max-cost`, or a failed call, or a closed window), in the same folder, with the same command plus `--resume`. Every recorded response answers the request that produced it, for free, and only the calls the log does not hold go to the API. The settings in `generation/run.json` must match: the set's name and the sealed half's, the corpus's SHA-256, the seed, the targets, the model, `--store-ids`, the gold window, the core, the core quota, the split seed, the lint spec and the prompts' hashes. `--max-cost` and `--api-key-env` are not among them, so a resume may raise the one and name another variable for the other. A last line cut short by a stopped run is taken out before any new call is added. While a run is going the log is only appended to; once the set freezes, the log is rewritten, whole, through a temporary file moved into place, with exactly the calls the set was written from. A resumed run ends with the same files a run that never stopped would have written. A folder that holds a frozen set is never resumed.

**Line endings never make two runs differ.** Git's `core.autocrlf` checks text files out with CRLF line endings on Windows, which would change a prompt's bytes, and so its hash, and stop its schema from being found. Three things prevent that:
- the repository's `.gitattributes` has git write `prompts/`, the top-level `docs/` files, `data/` and `sealed/` with LF line endings (files a clone checked out before this rule keep the endings they have);
- the generator reads prompt files with CRLF read as LF before it parses or hashes them, and hashes its own file for the manifest the same way;
- every file the generator writes has LF line endings on every operating system, so a set made on Windows replays byte for byte on Linux, with the one exception above.

A corpus is hashed as given, since `corpus.jsonl` is the user's own file: an export saved with CRLF is a different file from the same export saved with LF, and the same seed draws a different sample from it.

## 13. Tests

All of these run with a faked model and no network:

- the sampler is deterministic from the seed, and differs across seeds and categories;
- the strata and age bands, including a corpus with no `ts`;
- the sample cap: half the memories, split evenly, a category that waits, the last of the cap going to one that can use it, and the unit cap;
- eligibility: every field the model sees is scanned, a random-looking store id is allowed, and a cluster with one ineligible member is out whole;
- the near-copy clusters, including the overlap both ways and the digit and capital rule;
- version and session pairs, including the earlier and later order, the rule that a pair goes to one category, no pairs from equal `ts` or one session, and `must_rank_first` as a list when the later record has copies;
- the user message, with its pair labels;
- the abstention screen, both ways, with empty values skipped;
- the checks before the lint and the prompts' rules, case by case, including a cross_session item too big for the window;
- the lint integration, with a failing item held or dropped and never frozen, and duplicates checked against frozen items only;
- the freeze and the manifest, field by field, the freeze order, `answer_type` and difficulty;
- `--store-ids`;
- the `--out` location rule and the set name;
- the records checked before any call, and no folder left by a failed first call;
- the API request's shape, with a fake SDK;
- replay, with the call times;
- line endings: a CRLF copy of every prompt loads, parses and hashes like the original, every file of a set is written with LF line endings, and the manifest hashes the generator with LF line endings;
- the oracle check failing on a planted bad gold id, and naming it, and `RECALL_QFILE` left as it was.

Generator v2 adds `tests/test_generate_v2.py`, with the same faked model and temporary folders:

- the cost guard: `--max-cost` stopping before the call after the reply that passed it, with every call recorded and no set written; a `--resume` with the same cap making no new call; a `--resume` with a higher cap making only the calls the log does not hold and ending byte for byte like a run that never stopped; a cut last line dropped; other settings, a frozen set and an empty folder refused; exit 3 from the command line; a live run without `--max-cost` refused before anything is written; a model with no price refused; one price table for both generators;
- `--estimate`: no model made, no key read, nothing written; expected and worst case from the plan; a finished pilot's measured yields and tokens used by `--estimate-from`;
- `--api-key-env` and `--check-api` with a fake SDK: the client made from the named variable with the base URL pinned, the key in no output or file, taken out of an SDK error, and `--check-api` printing ok or the error's class (the providers' own tests are listed in section 18);
- the gold window: nothing past it shown or gold, the lint still holding an answer that leaks past it, a near copy past it keeping its record out;
- core first: the plans, pairs wholly in one stage, a target the core meets never reaching past it, a target past it filling from the rest, and the core subset;
- the core quota: the core stage ending at the quota, at half the unit cap and when its units run out, a core item past the quota held back as surplus and given no retry, the plans under a quota, and the quota required with `--core` and kept by a resume;
- the estimate by stage: a pilot measured stage by stage, each stage's yield applied to its own units with thin stages named, a stage with no call of its own taking the pilot's tokens over all its calls, a category the cap leaves short marked, and the totals of questions and of the core subset;
- prompt caching: the request's hash covering the cache setting and its hour, the system block marked for the 1-hour cache, cache tokens recorded with the API's split of its writes (and without it), priced at the 1-hour and 5-minute rates, counted the same by a resume and shown on the progress line, and the estimate pricing the cache;
- the version categories' retry keeping the new value, and the added ways of stating a preference;
- the split: seeded, stratified, complete and disjoint, the same files from the same seed, the public half holding the sealed half's hash and no generation record and no sealed question, replay of both halves, and where the sealed half may go;
- the retry: one call naming the leaking words with the record alone, the fix freezing, no retry once the target is met, none twice, none for other errors;
- the new checks (two questions in one, an alternate dropping the unit), prompts v2 (the full function-word list, the retry section, the temporal and implicit-preference changes), lint spec 3 (a time of day, a short initialism, a shown and a stated preference), and `parsica-recall show-questions`.

The changes of October 22, 2026 add `tests/test_generate_v3.py`, with the same faked model and temporary folders; one test reads the committed pool's generation record:

- the exporter: the committed pool exports deterministically to the committed `planted_pairs.jsonl` (137 pairs, 50 version chains, 48 corrections, 39 preference changes), every pair is an item's last transition and agrees with the pool's own final state, sits in the core and has its earlier `ts` before its later; the fold rules on a small pool (a label not asked for, a dropped record, a second event for one label, a step with a personal-data shape, a tail call, an item with one step), the command line, and the file read strictly;
- `--planted-pairs`: units routed by type, the core stage taking planted pairs only, knowledge_update going on with similarity pairs past the core and never a planted or core pair, correction_handling and recency_preference never taking similarity pairs, the one stage without a core, pairs the corpus cannot use counted by reason, a run freezing the planted pairs with the later record first, the manifest's file, hash and counts, the core quota holding, a resume keeping the file, and the estimate naming it;
- rare names: `record_names` case by case, a name found in no other record against one found in others, a name shared only with a near copy, the one-other-record threshold, entity_attribution drawing only from such records with the manifest's count, and a corpus with none;
- the retry: `padded_retry` case by case, a padded retry dropped and counted while a retry about another fact freezes, every prompt's A SECOND TRY asking for a different fact and never a longer quote (the version prompts keeping the new value), and the nine "specific enough" lines;
- `--unit-cap`: parsing, a cap replacing a category's own in the plans, the run and the estimate's worst case, the manifest and `run.json`, a resume keeping it, and bad input refused;
- the estimate: several pilots merged with a later one's measurements replacing an earlier one's per category, the records-shown line against the sample cap in its three forms, and a planted stage guessed at the planted yield;
- `answer_type`: a time unit needs a number right before it.

## 14. Decisions

Settled on October 20, 2026, after the design review. The sections above are written to match.

1. **answer_type and difficulty** come from fixed rules, not from the model: answer_type from the answer's form, and difficulty from where BM25 ranks the gold over the pool (section 6). Hard items will lean toward vocabulary_gap and implicit_preference, which is true of those items.
2. **Abstention items** freeze in a fixed form. The values the model expected are a screen, never answers, and they stay in the generation record (section 7).
3. **Corpora without `ts` or `source`** get no long_horizon or version-category items without `ts`, and no cross_session items without `source`. The manifest says why (section 4).
4. **Records with credential or personal-data shapes** are never sampled, so they are never sent to the API (section 2).
5. **`--out`** is refused anywhere in the repository, except under `data/` with `--release` (section 11).
6. **Near copies** join `gold_ids` by a fixed rule. A review added that they must also share their digit tokens and capitalized words, so an update is never mistaken for a copy (section 3).
7. **`--replay`** rebuilds a set from its recorded responses (section 12).

Also settled: the generation model is pinned to `claude-sonnet-5-5`, and the thirteen prompts were approved as drafted. Their SHA-256 is in every manifest, so a changed prompt is a new prompt version, and a set made with it is a new set.

Settled on October 21, 2026, for the first set over the public pool:

8. **Writer** `claude-opus-5-5`, after a three-writer comparison on an earlier set; named with `--model`, while the default stays `claude-sonnet-5-5`.
9. **The gold window** is the first 10,000 records of `data/pool-v1/corpus.jsonl`, and the set is valid on the 10k, 30k and 50k tracks; the 1k track reports the core subset (spec section 6).
10. **Core first**: each category takes what the core can support, then fills from the rest of the window (decision 15 caps what it takes from the core).
11. **Size**: about 150 per category, about 1,950 in all, split by seed into a public half and a sealed half of about 75 per category each.
12. **The sealed half** is never published: the maintainers keep it outside the public repository.
13. **An Opus pilot first**, 8 per category over the window with the v2 prompts, which the maintainer reads before the full run.
14. **A cost guard before any paid call**: `--estimate`, `--max-cost` and `--resume` (section 16). The full run never starts before the maintainer has seen its estimate and set `--max-cost`.

Settled on October 21, 2026, after a review:

15. **A core quota**: at most 40 questions per category from the core, then the tail (`--core-quota 40`, section 3).
16. **The hard categories** (knowledge_update, correction_handling, recency_preference, cross_session and implicit_preference, which the unit cap leaves short of 150 at the yields assumed) are decided after the pilot: the pilot measures their yields, and what a higher cap would cost is shown before the full run.
17. **Prompt caching** of the system prompt, in generator v2, with the 1-hour cache (section 6). The hour was chosen once it was worked out that a round of calls can run past five minutes.
18. **The pilot's `--max-cost`** is the reworked pilot's worst case, rounded up to the next dollar.

Settled on October 22, 2026, after reading the v11 pilot:

19. **Planted pairs.** The version categories take their units from the pool's record of its planted changes, as pairs of record ids, never from the persona files (section 5). correction_handling and recency_preference come from that record alone; knowledge_update goes on with similarity pairs past the core.
20. **Entity attribution: rare names only** (section 4). The lint rule stays.
21. **The retry asks for a different, more distinctive fact, never a longer quote**, and a retried answer that wraps the held one is dropped (section 6).
22. **Raise the unit caps for implicit_preference and knowledge_update** (`--unit-cap`, section 3), after a mini pilot of the changed categories (about $2) and before the full run; the full run is estimated from both pilots (section 16).

## 15. Generator v2

Recorded after a pilot on an earlier set on October 20, 2026, as suggestions, and built on October 21, 2026, with the changes the three-writer comparison asked for. The prompt changes make new prompt hashes, so a set made with them is a new set.

1. **A bounded retry for `GOLD_MULTIPLICITY` holds** (section 6). One retry per item at most, with the leaking words named and no other records shown, recorded in `generation/calls.jsonl` like any other call.
2. **The prompts' function-word list is `recall_common.FUNCTION_WORDS`.** Every prompt with an answer key now names all 112 words the lint's `KEY_EDGES` rule checks, where it named 38; a test keeps the two equal.
3. **A pre-lint check that an alternate keeps the answer's unit** (section 6), and a line in the prompts saying so.
4. **A pre-lint check for two questions in one** (section 6), and a line in every prompt.
5. **A line in the direct_recall and entity_attribution prompts against restating most of the record in the question.**
6. **temporal_awareness** no longer allows a question about "a record that contains a number": the answer is a time expression, never a plain number or amount ("1000 words", "34.6%").
7. **implicit_preference** leaves out advice (someone else's view, even when the person follows it) and habits or preferences the record states outright, and says the preferred thing's own name is a fine answer. The lint follows (spec 3): `IMPLICIT_PREFERENCE_MUST_NOT_BE_EXPLICIT` fires only when the gold record states a preference outright, with one of `stated_preference_terms`.
8. **A time of day such as "9PM" passes the length rule**: the lint's `short_answer_time_regex` (spec 3), and the prompts' length line.
9. **Alternates meet the answer's length rules**, the four-letter minimum included, so an initialism such as "OSU" is not written as an alternate.
10. **After a review**: the core quota (section 3), prompt caching for an hour (section 6), the estimate by stage (section 16); the retry line of knowledge_update, correction_handling and recency_preference asks for a longer span of the later record that still gives the new value (no longer "or another fact it states", since the answer must stay the new value); and `stated_preference_terms` gains go-to, swear by, stick with, from now on, only buy, only use, my pick and big fan.
11. **After the v11 pilot (October 22, 2026)**: planted pairs for the version categories (section 5), rare names for entity_attribution (section 4), the retry rewritten in every prompt to ask for a different fact and never a longer quote, with the padded-retry drop (section 6), `--unit-cap` (section 3), `--estimate-from` over several pilots with the records-shown line (section 16), and a time unit in `answer_type` needing a number before it. The prompt changes make new prompt hashes again.
12. **The v11 full run (October 22, 2026).** The first set from the protocol on pool v1, from the decisions made after the mini pilot: targets of 150, the core first with a quota of 40, the planted pairs, unit caps for the four categories the pilots showed short, Opus 5.5, in chunks under `--max-cost`:

    ```bash
    parsica-recall generate-questions --corpus data/pool-v1/corpus.jsonl --out data/v11 --release \
        --sealed-out sealed/v11-sealed --split-seed 2 --seed 2 --targets 150 --core-quota 40 \
        --gold-window 10000 --core 1000 --planted-pairs data/pool-v1/planted_pairs.jsonl \
        --unit-cap knowledge_update=100,correction_handling=96,recency_preference=78,implicit_preference=1200 \
        --model claude-opus-5-5 --api-key-env MY_API_KEY --max-cost 15
    ```

    then the same command with `--resume --max-cost 30`, then `--resume --max-cost 45`. What the run showed:
    - **The chunks and the resume.** The first chunk was cut by a two-hour limit on the run at $14.92, eight cents short of its cap, with 943 complete calls in the log; the resume took the log as it stood and went on. The second chunk stopped at its cap ($30.01, exit 3). The third finished the set at $38.02. Every call after a resume read its system prompt from the cache. The manifest's `tool` block says `dirty: true` and names the commit the last chunk started from, because the call log was being appended in the checkout while the run went.
    - **The estimate against the result.** From both pilots the estimate said about 1,485 questions, $31.12 expected and $50.92 at the worst case; the run gave 1,454 questions for $38.02 over 2,431 calls, with 466 in the 1k subset against 468 expected. The count was close. The cost ran 22% over the expected figure and under the worst case because 1,234 calls were retries, 51% against the 37% the pilots measured: at the full targets more items were held on `GOLD_MULTIPLICITY` (entity_attribution retried 262 times over 570 units, temporal_awareness 249 over 600).
    - **Where the categories stopped.** Seven met 150. temporal_awareness (120) and cross_session (115) reached the 600-unit cap; implicit_preference reached 102 at its 1,200-unit cap (0.085 frozen per unit); knowledge_update 22 at its 100-unit cap (21 from the 50 planted pairs, 1 from the 50 similarity pairs of the tail); correction_handling (15) and recency_preference (30) used every planted pair. Those three are under 25 per half, so `GATE_CATEGORY_MINIMUMS` fails on both halves and the set is interim; the 30k tail with planted changes is what fills them.
    - **Two refusals.** Two of the 2,431 replies came back with the API's `refusal` stop reason and no text (one entity_attribution call, one vocabulary_gap call); each was counted an unusable call and the run went on. Two retries were dropped as padded, both in vocabulary_gap.
    - **The replay.** `--replay` on the sealed half's call log into a scratch folder outside the repository rebuilt both halves without a call: the same `questions.json` and `lint_report.json` bytes as the frozen halves, and manifests equal except for `tool` (section 12).

## 16. The cost guard

Generation is the only step that spends money, so a live run cannot start without a limit, and what it will cost is printed before it starts.

**Prices.** `PRICES` in `parsica_recall/generate_questions.py` holds Anthropic's first-party rates per million tokens (input, output, 5-minute cache write, 1-hour cache write, cache read), one table for both generators; `parsica_recall/generate_pool.py` reads it from there, and its calls use the 5-minute cache. On October 21, 2026 the pricing page gave `claude-opus-5-5` $4 in and $20 out, `claude-sonnet-5-5` $2 and $10, `claude-fable-5-1` $10 and $50. A cache write costs 1.25 times the input price for five minutes and 2 times for an hour, and a cache read on Opus 5.5 or Sonnet 5.5 costs $0.20 (on Fable 5.1, $0.25). A call's cost is its own token counts from the reply's usage at the price of the model the run asked for. The bill is the authority.

**`--estimate`** prints one row per category, then stops: no model is made, no key is read and nothing is written. With `--core`, a row has two stages, the core and then the rest of the window, and for each the units the run is expected to show, the yield it assumes and the items it expects to freeze; then the category's frozen items in all, its calls and their cost, and the worst case. Input tokens come from the planned messages themselves, the system prompt and the records each batch would send, at 2.3 characters per token (pool v1's 1,090 recorded calls measured 2.36). With prompt caching on (section 6), each category's system prompt is priced as one 1-hour cache write on its first call and a cache read on every call after, at the cache prices in `PRICES`; the records each batch sends are plain input. Output tokens per call and per retry, retries per call and each category's publishable items per unit shown are the `EST_*` figures in the code, rough until a pilot measures them, and a yield the estimate guessed is marked `*`.

**Expected** walks each category's core stage at the core's yield until the core quota, the target or the core's units run out (the core's units stop at half the unit cap under a quota, section 3), then the rest of the window at its own yield until the target or the unit cap. A category the unit cap leaves short of its target is marked `short` and listed under the table with the items it is expected to reach. The totals line gives the questions the set is expected to hold, how many of them have every gold record in the core (the 1k subset), and the calls, retries, tokens and cost. **Worst case** is every category running to its unit cap, at the same tokens and retries per call; where batches rounding at the stage boundary would make it cost less than expected, it is the expected cost. A reply can run longer than these averages, and the cache saves only while it holds, so `--max-cost` is the guard. On a stopped run, `--estimate` also says what the log already holds, which a resume replays free.

**`--estimate-from DIR`** takes what a finished set measured, stage by stage: units shown, publishable items and their yield per unit, output tokens per call and per retry, retries per call, and the characters per input token over all its calls. The set is the pilot, or the sealed half of a split set, which keeps the generation record. A call counts in the stage of its first unit, and a retry in the stage of the item it retries; an item is publishable when it froze or was surplus. A stage the finished set never reached takes its rates over all its units, and a stage that holds units but no call of its own (a batch that ran on from the core) takes its tokens and retries per call from all of the set's calls. A stage it measured on fewer than 12 units (`EST_TRUST_UNITS`) is marked `?` and named in a line of its own, `too few units to trust`, since a yield over so few units can be far off. So the full run's estimate comes from the pilot, and says where the pilot showed too little to go on. It takes more than one folder (`--estimate-from ~/recall-sets/pilot-1 ~/recall-sets/pilot-2`): a later folder's measurements replace an earlier folder's, category by category, and so do its characters per input token, so the full run is estimated from the v11 pilot plus the mini pilot of the categories that changed after it, whose measurements in the v11 pilot (similarity pairs, any record for entity_attribution) no longer apply. The line under the table names every folder.

**Two more lines.** A stage made of planted pairs alone (section 5) is guessed at `EST_YIELD_PLANTED` (0.5), not at the category's similarity-pair yield, until a pilot measures it. And the estimate prints how many distinct records the planned units would show, expected and at the worst case, against the sample cap (half the eligible memories, 4,998 on pool v1), since a raised unit cap can run into it: it says when the expected run would pass the cap, and when only the worst case would.

**`--max-cost USD`** is the most the set may cost, counted over every call of the set, the ones a resume replays included. Before each new call, the run stops when the cost so far has passed the limit: the reply that passed it is recorded in full, nothing is written mid-way, and the run exits 3 with what it spent and how to go on. A resume with the same limit replays the log free and stops again before its first new call; a resume with a higher limit goes on from there. So `--max-cost` is a ceiling on the set, not on one run, and raising it is how more is spent. The last reply can pass the limit by at most one call's cost. A live run without `--max-cost` is refused, and so is a limit for a model with no price on file.

**The manifest** reports the set's cost from its call log, with the token counts it came from.

## 17. When the pool grows

Not built in generator v2; recorded so the next pool round keeps the set sound. When the pool grows past the records a set was linted against (pool v1 is 10,000 records today; the 30k and 50k tracks need more), the lint reruns against the new records. An item that now leaks (`GOLD_MULTIPLICITY` finds its answer in a new record) or stops abstaining (the abstention screen finds a screened value in one) is retired, never edited, and the set's next release says how many. The writer of the larger tail (a block-structured tail, from a writer that is not built yet) should put this set's answers and screened abstention values on each new block's avoid list, so few items are retired in the first place.

## 18. Other providers

Added November 21, 2026. The question builder can call OpenAI's API (`--provider openai`), Google's Gemini API (`--provider google`), or any endpoint that speaks OpenAI's chat format (`--provider openai --base-url URL`), beside Anthropic's, which stays the default. The pool generator calls Anthropic only.

**What stays as it was.** A set built on Anthropic is requested, recorded and described exactly as before: the request's SHA-256, the call log, `run.json` and the manifest have the same fields and bytes, so v11 and every set built before still replay byte for byte. The only addition an Anthropic set can carry is the price flags' record, when they are given.

**One choice, checked once.** The provider, the model, the endpoint, the variable the key is read from, the output cap, the effort and the price are settled before anything is written (`ModelChoice` in the code). It refuses `--base-url` without `--provider openai` or without `--model`; a URL that is not `http` or `https`, or that holds a user name, a password, a query or a fragment; a key sent over plain `http` to anything but this machine; `--max-output-tokens` without `--base-url`; one price flag without the other; and a model whose name belongs to another provider (`gpt-` to OpenAI, `gemini-` to Google, `claude-` to Anthropic).

**The requests.** Every provider gets the category's prompt as the system message, the batch as the user message, and the category's JSON schema as the reply's format.

| | OpenAI | Google | An endpoint of your own |
| --- | --- | --- | --- |
| SDK | `openai`, Chat Completions | `google-genai`, generateContent, the Gemini Developer API (never Vertex AI) | `openai`, Chat Completions |
| Schema | `response_format`, `json_schema`, `strict` | `response_mime_type` JSON with `response_json_schema` | `response_format`, `json_schema`, `strict` |
| Effort | `reasoning_effort` `high` | `thinking_level` `HIGH` | none |
| Output cap | `max_completion_tokens` 32000 | `max_output_tokens` 32000 | `max_tokens` only with `--max-output-tokens` |
| Also | `store` false; the SDK's two retries | the SDK's automatic continuation off, three attempts | nothing a server may not know |
| Endpoint | `https://api.openai.com/v1`, whatever `OPENAI_BASE_URL` says | `https://generativelanguage.googleapis.com/`, whatever `GOOGLE_GEMINI_BASE_URL` says | the URL given |
| Key | `OPENAI_API_KEY`, or `--api-key-env` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` (the SDK reads `GOOGLE_API_KEY` first), or `--api-key-env` | `--api-key-env` only, never `OPENAI_API_KEY`; without it, a placeholder |

32,000 leaves room for reasoning tokens, which count against the cap on both providers (OpenAI's reasoning guide advises room for them), and stays under the 32,768 tokens one Gemini call returns. Automatic continuation is off so a call is one request: a reply cut at the cap is recorded as cut, and the SDK never makes a second, uncounted request. The key handling is AnthropicModel's: the value is never printed, logged or written, and every error has it taken out and drops the exception it came from.

**What a reply becomes.** OpenAI's `prompt_tokens` holds the cached and the cache-written tokens, and Google's `prompt_token_count` the cached ones, so `input_tokens` in the call log is the rest, as Anthropic counts it. OpenAI's cache writes (`cache_write_tokens`) and reads (`cached_tokens`), and Google's cached tokens, go to the same fields as Anthropic's. Output is what each provider bills as output: OpenAI's `completion_tokens`, which hold the reasoning, and Google's reply tokens plus its thought tokens; `reasoning_tokens` records the reasoning part. Google's thought parts are left out of the reply's text. The stop reason is recorded in the provider's own word; a refusal (OpenAI's `refusal` field, `content_filter`, Google's `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `LANGUAGE`, a blocked prompt) yields no items as a declined call, and a cut reply (`length`, `MAX_TOKENS`, `CONTINUATION`) as one that hit its cap.

**What the set records.** For a provider other than Anthropic, each call adds `provider`, `endpoint` and `reasoning_tokens`; `run.json` adds the provider, the endpoint, the effort and the cap, so a `--resume` with another provider is refused; the manifest's `model` adds the same with `reasoning_tokens` and `prices`, and its `system_cache` and `server_side_fallbacks` are null, since those are Anthropic's settings. The request's SHA-256 covers the provider, the endpoint, the effort and the cap, so a log replays only for the choice that made it: `--replay` takes the same `--provider`, `--model` and `--base-url`.

**Prices.** The table holds the models listed in docs/doc2query.md, at each provider's standard prices as of the date beside them in the code: OpenAI's cache writes at 1.25 times input for the default 30-minute cache, and Google's implicit cache with no write charge. `--price-in` and `--price-out` replace the table for one run, with cache reads and writes counted at the input price. Each call then carries the price it was counted at (`price_usd_per_million`), and so does every call at an endpoint of your own, null when no price was given, so the cost a log reports never comes from a table that has since changed. `--max-cost` needs a price from one or the other.

**The estimate.** For OpenAI, Google and an endpoint of your own, every prompt is counted uncached, which can only overstate, and the estimate says its tokens per call were measured on Claude and asks for a pilot with `--estimate-from` before a full run.

**Tests.** `tests/test_providers.py`, with each SDK faked: the choice's defaults and what it refuses; an Anthropic request hashing as before, with and without the price flags, and another provider or endpoint hashing apart; each request's exact shape; replies parsed into the call log's fields, refusals and cut replies included; the key read from the named variable and taken out of an error; an endpoint of your own never sent `OPENAI_API_KEY`; a missing package naming its extra; `--check-api` printing ok or the error's class; records priced by their own price or the table; a whole set built through each adapter, recorded, priced and replayed byte for byte without the SDK; an Anthropic set unchanged unless the flags price it; and the command line. Outside the suite, the adapters were run with the real `openai` (3.57.0) and `google-genai` (3.6.1) SDKs against a local stub server, which showed the requests on the wire and the replies read back.

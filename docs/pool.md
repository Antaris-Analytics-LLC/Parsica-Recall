# The public pool

This is the design of the pool generator (`parsica-recall generate-pool`, the module `parsica_recall/generate_pool.py`), the generator that writes Recall's public pool: a synthetic agent-memory store, released under MIT, that carries material for every category the benchmark asks about. Invented people each live 6 to 18 months of sessions with their own AI agent, and the pool is the records an agent-memory store would keep from those sessions. The question is always the one Recall asks: does the system recall the correct memory, at the correct time, when it matters.

The persona files are in [`scenarios/`](../scenarios/), one per person, and the frozen prompts are in [`prompts/pool/`](../prompts/pool/). The question generator ([generation.md](generation.md)) never reads the persona files; it sees only the pool.

## 1. What a run does

```
parsica-recall generate-pool --scenarios scenarios --out ~/recall-pools/pool-v1 \
    --seed 1 --core 1000 --total 50000 [--model claude-sonnet-5-5] [--personas p1,p2]
```

1. **Read the persona files and the prompts.** Every persona file is checked before anything else happens (section 8), and both prompts must carry their schema.
2. **Plan** every persona's sessions from the seed: how many, on which days and at what times, how many records each, and what each core session must include (section 3).
3. **Write the core.** One model call writes one session of one persona. The call receives the persona file, the persona's running state and the session's plan, and returns the session's records and a state update (section 4). A persona's sessions run in order; personas run in parallel, `--workers` at a time.
4. **Screen** every record the moment it comes back, then screen the whole core again in a fixed order, and add make-up sessions until every persona holds exactly its share (section 5).
5. **Write the tail** the same way, screened against the finished core as well.
6. **Assemble** `corpus.jsonl`: the core in time order, then the tail in a seeded order (section 6). Write the manifest and the generation record (section 7).

The command line:

| Flag | Meaning |
| --- | --- |
| `--scenarios` | The folder of persona files. Default `scenarios/`. |
| `--personas` | Only these persona ids, comma separated, such as `p1,p2`. Default: every file in the folder. |
| `--out` | The folder the pool is written to. Required. It must not exist yet, or must be empty, unless `--resume`. Where it may be is in section 9. Its last path component becomes the pool's name. |
| `--seed` | The plan seed, an integer. Required. |
| `--core` | Records in the core. Default 1,000: the 1k track, which a question set draws from first (spec section 6). |
| `--total` | All records, core included. Default 50,000, the largest standard track. |
| `--model` | The generation model. Default `claude-sonnet-5-5`. |
| `--workers` | Personas written at once. Default 4. It changes how long a run takes, never what it writes. |
| `--release` | Allows an `--out` under `data/`, for the public pool. |
| `--replay` | Rebuilds a pool from a recorded call log (a `generation/` folder, or any of its `calls-NNNN.jsonl` parts) into a new folder, without the API (section 10). |
| `--resume` | Picks up a run that stopped, from the call log in `--out`; only the calls it does not hold go to the API (section 10). |
| `--effort` | Effort for the calls: one level for both phases, or `core=LEVEL,tail=LEVEL`. Default `high`. Recorded on every call. |
| `--max-calls` | Stops the run before its next new API call once this many were made in this run, with every call recorded; exit 3. Calls a `--resume` replays do not count. |
| `--estimate` | Prints the plan's sessions, calls, tokens and cost estimate, then stops before any call. |
| `--check-api` | Makes one tiny call to prove the key and the network, and prints `ok` with the model that answered, or the error's class name; nothing else. Needs no other flag, except `--api-key-env` when the key is in another variable. |
| `--api-key-env` | Reads the API key from this one environment variable instead of `ANTHROPIC_API_KEY`, and sends every call to `https://api.anthropic.com` whatever `ANTHROPIC_BASE_URL` says (section 2). Not recorded. |

The exit status is 0 when the pool was written (or `--estimate` printed, or `--check-api` said ok), 2 on input the generator refuses or a model call that failed, and 3 when `--max-calls` stopped the run, which a `--resume` continues.

**Before any call** a run prints its plan and an estimate: core sessions, the tail's sessions at the least over-ask, the calls each is likely to take (plus 8% for retries and make-ups), how many the log already holds (replayed free on a resume), the input, cached and output tokens, and the cost at the model's price. The per-call and per-record figures are the pilots' averages at effort high (`EST_*` in the code), so the estimate is rough, and lower efforts write less. The prices are Anthropic's first-party rates in `PRICES`, one table for both generators, kept in `parsica_recall/generate_questions.py`; a model not listed gets token counts only. The manifest reports the cost from the call log's own token counts the same way; the bill is the authority.

## 2. What leaves the machine

The pool is invented, so nothing of anyone's own store is sent. Each call sends the session prompt, the persona file's descriptive fields (name, situation, domains, agent role, voice, agent voice, people, things), the persona's running state (which the model itself wrote in earlier calls) and the session's plan. The persona file's numbers (timeline, cadence, sizes, quotas) stay on the machine and shape the plan.

The API key comes from `ANTHROPIC_API_KEY` through the official `anthropic` Python SDK, exactly as for the question generator, and the generator never reads it. Or, with `--api-key-env NAME`, it comes from the one variable `NAME`, handled by the same code as the question generator's (the pool's model extends its `AnthropicModel`, and only streams its calls): the generator reads it when it makes the client, hands it to the SDK as the key, and pins the API address to `https://api.anthropic.com`, whatever `ANTHROPIC_BASE_URL` says. A key handed over this way also stops the SDK from reading any other credential in the environment, such as `ANTHROPIC_AUTH_TOKEN` (checked against anthropic 1.17.0). Spaces and line breaks around the value are let go; a value with a space, a line break or a quote inside it is refused, never sent. Either way the key never appears in output, logs, the manifest, the generation record or an error message: with `--api-key-env`, every error the generator reports has the key taken out of the SDK's text and drops the exception it came from, so not even a traceback can show it. The flag itself is not recorded, so a resume may name another variable. A live run loads the SDK and makes the client before writing anything, and nothing is written until the first reply is in hand, so a missing package, a missing key or a key the API rejects on the first call leaves no folder behind. The tests, `--replay` and the rest of the harness never load the SDK.

## 3. The plan

**Shares.** `--core` and `--total - --core` are each split evenly over the personas; a remainder goes one record each to the first personas in id order. Eight personas and a core of 1,000 give each persona 125 core records.

**Quotas.** A persona file says how much category material its core carries, written for a core share of 125 records:

| Quota | Label | What it plants | Feeds |
| --- | --- | --- | --- |
| `version_chains` | V | a fact stated, then changed once or twice, each value in its own record and session | knowledge_update |
| `preference_changes` | R | a preference stated, then changed, each in its own record and session | recency_preference |
| `corrections` | C | a wrong statement, then a later record that names it and corrects it | correction_handling |
| `threads` | T | one topic across two to four sessions, each part adding a new detail | cross_session |
| `long_horizon_facts` | L | a fact said once, early, and never again | long_horizon |
| `shown_preferences` | S | a preference shown by behavior in two or three sessions, never stated | implicit_preference |

The other categories (direct recall, entity attribution, temporal awareness, paraphrase, vocabulary gap, abstention) draw on every record, since every record carries a specific fact. The generator scales each quota to the persona's actual share in integer arithmetic, half up, and keeps at least one of anything the file asks for: a pilot share of 150 turns a quota of 6 into 7.

The shipped quotas are sized so eight personas can feed the standard minimum of 25 items per category: about six of each per persona gives 48 units per category across the core, which leaves room for what the question generator's checks and lint hold back (a pilot on an earlier set froze about 60% of what the model wrote).

**Sessions.** Each phase of each persona has its own seeded generator, made from the seed, a hash of the persona file's planning fields (`id`, `timeline`, the numbers in `cadence`, `records_per_session` and `core_quotas`) and the phase name. Rewording a persona (its situation, voices, people or the cadence's description) never moves its sessions; changing a number does.
- **Sizes** are drawn from the file's `records_per_session` range until they reach the share; the last is cut to make the sum exact, and a last session that would fall under the range is spread over the others.
- **Days** are drawn from the timeline by the file's weekday weights, without repeats while the timeline has enough days. Integer weights and integer draws keep the plan the same on every platform.
- **Times** fall within the file's local hours, unique to the minute and, when the day allows it, three hours from the persona's other sessions that day. The model sees local time; records carry UTC, from the file's `utc_offset_hours`.
- **Session ids** look like `p3-s4f1a2c`: the persona, then six hex digits of a hash of the seed, the persona, the phase and the session's place in the plan. They are opaque on purpose. A core session and a tail session look alike, and an id says nothing about where a session sits in time, so `source` never tells a ranker which records are core.

The core's plan depends only on the seed, the persona file and the core share, so it is the same whatever `--total` is. The tail's sessions are planned after it, over the same timeline, avoiding the core's minutes.

**What each core session must include.** Every quota item is placed on sessions by fixed rules:
- a long-horizon fact goes in the first quarter of the persona's sessions;
- a version chain or preference change has two steps, or three (about a third of them, when there are at least five sessions); its first step falls in the first half, and later steps follow at least two sessions apart where they fit, one apart where they do not;
- a correction's error goes anywhere but the last session, and its correction one to three sessions later;
- a thread runs over two to four distinct sessions, and a shown preference over two or three;
- each step is one record, and a session never carries more steps than records: when one would, a record moves to it from the session with the most room.

`generation/plan.json` lists every session with its date, size and the steps it carries.

**Entries.** Each session has 15% entries, rounded half up, and at least one; across a persona that lands between 10 and 20 percent, with the exact share in the manifest.

## 4. One session call

- **Model and settings.** `--model` (default `claude-sonnet-5-5`), the effort `--effort` gives the call's phase, thinking at the model's default, no temperature, server-side fallbacks off, as for the question generator. Pool calls stream, with `max_tokens` 32000: pilot 2 had one reply cut at 16000, and the SDK refuses a non-streaming request it expects to run past about ten minutes, so the pool streams and takes the final message. The manifest records the model asked for, the models that answered, the effort per phase and how many calls ran at each, the SDK version, the first and last call times, and the input, output, cache-write and cache-read tokens.
- **Prompt caching.** The fixed prefix is cached: the system prompt, and the start of the user message up to `cache_split`, which is the persona (in the core) or the persona and the two core lists that open the state (in the tail). Those bytes are the same for every session of a persona in a phase, so each session after the first reads them from the cache. The message is still one JSON object; the split falls between two keys. Each call records its `cache_split`, and the request hash covers it, the effort and the streaming.
- **Request.** The system prompt is the phase's prompt file, `core_session.md` or `tail_session.md`, byte for byte with its line endings read as LF. Its schema, between `BEGIN SCHEMA` and `END SCHEMA`, is the structured output format, so one hash covers both. The user message is one JSON object:
  - `persona`: the file's descriptive fields;
  - `session`: its id, local time, weekday and the persona's whole timeline;
  - `write`: how many records, how many of them entries;
  - `must_include`: one instruction per planted step, written from the state, such as "V2, step 2 of 3: the fact about "the launch date" has changed since it was last given as "March 3" (2026-01-14). In its own record, the agent brings up the new fact as forced by circumstances: a delay, a price, a broken part, someone else's choice ... do not repeat the old value.";
  - `state`: the running state (below).
- **Response.** One JSON object: `records`, a list of `{kind, text}` in session order, and `state_update`, with `new_people`, `facts` (one line per fact the session stated) and `events` (one per planted step, with its label, the position of the record that carries it, its subject, its value and a note, which for a correction's first part holds the true value).
- **The state** is what keeps months of sessions coherent. In the core it shows the people so far, the newest 200 facts, every started item with its subject and values, and a list of long-horizon facts never to mention again. In the tail it shows the people, every core fact and item value (as a list the tail must never give a value for), the subjects of the planted items (matters the tail never touches) and the newest 200 tail facts. The tail may stay in the same projects, places and people, and it carries small changes and corrections of its own, worded as freely as the core's, so change words never mark a core record. Each session sees only what was established on or before its own date, so a make-up dated earlier than later sessions never sees their facts; the tail's list of core facts to avoid is the one exception. The state is rebuilt from the replies alone, so the call log holds everything it ever was: each call's user message carries the state it was given, and its response carries the update.
- **What enters the state.** A planted step counts only when the record it names was kept by the screens, so a later step never builds on a value the pool does not hold. A fact line, a new person or a step whose text has a secret or personal-data shape is left out of the state, so it never reaches `states.json` or a later request. A record dropped later, by the whole-pool pass or by trimming, has already been folded in; that is rare, and the manifest counts drops by reason.
- **Instructions follow the state, not the plan.** A step is written from what the state holds, so if the model did not report a chain's first value, or its record was dropped, the next step asks for a first value again instead of changing a value that was never stated.
- **How changes and corrections are worded.** Pilot 1 showed one template for every correction and change ("I need to correct what I told you on ... That was wrong. The true X is Y"), which would let the word "wrong" mark correction records. So each step draws its wording from its own seeded generator (seed, planning hash, label, step), never from the plan's: a change gets a speaker (the person, or for a version chain either one) and one of seven ways of bringing it up (`CHANGE_STYLES`); a correction's first part gets who makes the error, and its second part one of the ways that fit that speaker (`CORRECTION_STYLES`), many of which never use the words wrong, mistake or correct. Both prompts forbid set phrases and openings on a date.
- **Times.** A record's `ts` is its session's start plus a seeded 40 to 420 seconds per record, in reply order, as ISO UTC. A record's text is stripped of leading and trailing blank space before its id is computed, so no record ends in blank lines.
- **Unusable replies.** A refusal, a reply cut at `max_tokens` or one that is not the schema's JSON is asked again, at most twice, with the same request (each attempt is recorded). A session whose three replies all fail writes nothing, and its planted steps move to the persona's next session; steps still unplaced when the core ends are counted as lost and never move into the tail.
- **Record of every call.** The call log holds, per call: the persona, phase, session and attempt, the request's SHA-256, the effort and cache split, the user message, the response text, the stop reason, the served model, the token counts (cache writes and reads included) and the times the request went out and the reply arrived. It is split into numbered parts, `generation/calls-0001.jsonl`, `calls-0002.jsonl` and on, each under 45 MB, because the full log of a large pool passes GitHub's 100 MB limit for one file. A running log rolls to the next part when the current one would pass the limit; the finished log is rewritten in order across parts, and the manifest lists the parts and their hashes. `--replay` and `--resume` read the parts in order.

## 5. Screens

Every record is screened when its reply comes back, in this order. A record that fails is dropped and logged in `generation/dropped.jsonl` with the reason; nothing is edited into passing.

| Reason | Rule |
| --- | --- |
| `bad_record_shape` | the record is not `{kind, text}` with a kind of `user`, `agent` or `entry` and a non-empty text |
| `not_unicode` | the text holds a lone surrogate |
| `too_short` | fewer than 6 normalized tokens, the lint's `min_gold_tokens`, so every record can be gold |
| `entry_length` | an entry under 100 or over 800 words |
| `secret_or_pii` | the text matches a shape of the lint's own scanner (`NO_SECRETS_OR_PII_POOL`): API keys, tokens, connection strings, email addresses, phone numbers, street addresses, government ids, card numbers. The dropped log keeps the shapes found and never the text, and the state never takes it in. The raw reply in the call log still holds it, because replay needs the reply as it came; the manifest counts such replies (`screens.calls_with_secret_or_pii_shapes`). |
| `duplicate_id` | the same content id as a record already kept |
| `near_copy` | a near copy of a record already kept, by the question generator's rule: each covers at least 90% of the other's tokens, with the same digit tokens and the same capitalized words |
| `restates_core` | a tail record that restates a core record: on content tokens (the harness's `STOPWORDS` left out), each covers 0.6 or more of the other's (`token_overlap` both ways), and the core record has at least 8 content tokens. That is the near-copy rule's shape at a lower bar: a paraphrase of similar length. Pilot 2's first rule (all tokens, the core record as the only reference) dropped twelve records, every one a long entry of 266 to 457 tokens holding 60% of a short core record's words through stopwords and common words, with content overlap of only 0.15 to 0.55; those are kept now. |
| `over_budget` | a persona over its exact share drops the surplus: the core its newest records, the tail an even, seeded spread (below) |

**The family tier still sees what the screen allows.** The runner's family tier keeps the spec's measure (all tokens, the gold record as the reference), so a long tail entry the screen keeps can still register as a family hit for a short gold record. That is a diagnostic, never the headline, and it is the honest reading of such an entry.

**Order, so threads never decide what is kept.** While personas run in parallel, the duplicate and near-copy screens look only within the persona. When every persona has finished a phase, one whole-pool pass runs in a fixed order (persona, then session, then position; in the tail, after every core record) and drops a record whose copy came earlier in that order. A later round screens only the records added since against everything kept, so a record kept once is never dropped later. The result is the same with one worker or eight.

**Exact counts.** In the core, a session that keeps fewer records than it was asked for passes the shortfall on, and one that keeps more asks the next for fewer. No session is ever asked for more than the persona's largest session size, so a model that writes short never snowballs into one huge request. The tail is over-asked instead: it is planned once the core is done, for its share plus the core's measured drop rate and 5 points, at least 10% (`tail_overask_per_mille` in the manifest), and each tail session asks for its planned size. After its last planned session a short persona adds make-up sessions until it holds its share. Each is at least the persona's smallest session size and at most its largest, so one record short asks for a whole small session and the surplus is trimmed (in pilot 2, one record short became seven one-record calls, each dropped by the same screen). Each is dated on a seeded day of the whole timeline, drawn by the weekday weights like every other session, so make-ups never gather at the end of the timeline and give core records away by their dates. (A make-up that carries the planted steps of a lost session is dated the day after the persona's last session, so the steps stay in order.) One round adds enough make-ups for the shortfall plus three, and there are at most three rounds after the whole-pool pass. A persona over its share drops the surplus: in the core its newest records (a session's worth at most), in the tail a seeded even spread across all its tail records, since dropping only the newest would thin the tail's last months and its `ts` distribution has to match the core's (`GATE_DISTRACTOR_MATCH`). The core holds exactly `--core` records or the run stops and says why (sessions lost, unusable replies, records dropped); so does every persona's tail share.

**Checked at assembly.** Every record's `id` is the content id of its text, its `ts` is an ISO UTC time, its `source` is a session id, its `kind` is one of the three, and no id appears twice; the written `corpus.jsonl` is then loaded with the runner's own loader. A failure there is a bug, and the run stops.

## 6. The file order and the tracks

`corpus.jsonl` holds one record per line with exactly the spec's fields: `id`, `text`, `source` (the session id), `kind` and `ts`. No `meta`, so nothing passes to an adapter that a real store would not have.

- **The core** comes first, in `ts` order (ties by persona, session and position). It is, by itself, the 1k track. A question set takes its gold from the first 10,000 records, the core first, and marks the questions whose gold is all in the core (spec section 6, amended October 21, 2026).
- **The tail** follows. Its records are grouped by persona and month of `ts`; each group is sorted and then shuffled by a seeded generator, and the groups are interleaved in proportion to their sizes, so every prefix of the tail is a representative slice of personas and months.
- **Tracks are prefixes.** The 10k, 30k and 50k tracks are the first 10,000, 30,000 and 50,000 lines of the file. The tail is the spec's distractor file by another name (spec section 6).
- **No signal in time.** Tail sessions are dated over the same timelines as the core, so the core and the tail share one distribution of `ts`, as spec section 6 asks of distractors.
- **Position is a signal until the runner shuffles.** In the file, the core comes first and in time order. Spec section 6 has file mode ingest a seeded shuffle of the corpus and its distractors, but track mode in the runner, which does that, is not built yet. Until it is, a prefix of the file used as a plain corpus is ingested in file order, which hands a ranker "ingested first" as a sign of a core record. Shuffle a prefix before using it as a corpus by hand.

## 7. The pool folder and its manifest

`--out` holds:

| File | What |
| --- | --- |
| `corpus.jsonl` | The pool. |
| `manifest.json` | Below. |
| `generation/calls-0001.jsonl`, ... | Every model call, sorted by phase, persona, session and attempt, in parts under 45 MB, read in order. |
| `generation/records.jsonl` | For every line of the corpus: its persona, phase, session, position in its session and the call that wrote it. |
| `generation/dropped.jsonl` | Every dropped record with its reason (texts of `secret_or_pii` drops left out). |
| `generation/plan.json` | Every session of every persona: id, phase, local and UTC start, size, entries and planted steps. |
| `generation/states.json` | Every persona's final state. |
| `generation/run.json` | The settings `--resume` checks. |
| `generation/prompts/`, `generation/scenarios/` | The prompt and persona files used, as hashed and sent, with LF line endings. |
| `planted_pairs.jsonl` | Written afterwards by `parsica-recall export-planted-pairs` from the call log and `records.jsonl`: the pool's planted version chains, corrections and preference changes as pairs of record ids (each item's last transition), for the question generator's `--planted-pairs` (docs/generation.md section 5). Not part of the manifest. |

`manifest.json` holds:

| Field | Value |
| --- | --- |
| `pool`, `spec`, `id_scheme`, `license` | The pool's name, `recall/1.0`, `content`, `MIT`. |
| `status`, `status_reason` | `complete` for every persona file, the standard core and the standard total; `partial` otherwise, with the reason. Every pilot is partial. |
| `files` | SHA-256 of the corpus and of every generation file. |
| `layout` | Where the core and the tail sit in the file, how each is ordered, and which tracks the file holds. |
| `model` | As in section 4, plus unusable replies and lost sessions. |
| `prompts`, `scenarios` | Each file and its SHA-256. |
| `plan` | Seed, shares, scaled quotas, items that could not be placed, sessions planned and made up, planted steps asked for and reported, and every constant of section 3. |
| `screens` | The rules of section 5. |
| `statistics` | Below. |
| `tool` | The generator file's SHA-256 (LF), the harness commit and whether the checkout had uncommitted changes. |
| `provenance` | Who wrote the pool, from what, and that no record was edited by hand and every person in it is invented. |

**Statistics.** The manifest reports the pool the way a real store can be measured, so a real store's statistics (lengths, kinds, entries per day, sessions, restatement rate; never its text) can sit beside them as a realism yardstick:
- records by kind, for the whole pool, the core and the tail, and the entry share of each;
- per persona: records, core, tail, kinds, sessions, active days, first and last `ts`;
- words per record (all, and per kind) and characters per record: min, the 5th, 25th, 50th, 75th and 95th percentiles (nearest rank), max and mean;
- sessions (core and tail) and records per session;
- active days (a persona's distinct UTC dates of `ts`), records, entries and sessions per active day;
- records by month;
- near-copy clusters left in the pool (0 after the screens; the count is an independent check);
- the restatement rate: the share of records that restate an earlier record by the `restates_core` rule (content tokens, 0.6 both ways, the earlier record with at least 8 content tokens), for the whole pool, for core records against the whole pool, and for the core against itself;
- dropped records, by reason and by phase, and each phase's drop rate (dropped over returned).

## 8. Persona files

One JSON file per persona in `scenarios/`, authored and reviewed with the code. The people are invented; real products, tools, places and public organizations are fine, and no real person appears. Nothing is modeled on any real store.

| Field | What |
| --- | --- |
| `id` | `p1`, `p2`, ... |
| `name`, `situation` | Who they are and what is going on in their life. |
| `domains` | What their life and their agent sessions are about. |
| `agent_role` | What their agent does for them. |
| `voice` | How they write. |
| `agent_voice` | How their agent writes, so a coding agent and a kitchen assistant do not sound alike. |
| `people` | Invented people in their life, with relations. |
| `things` | Real products, tools, places and organizations they use. |
| `timeline` | `start` (a date) and `months` (6 to 18). |
| `cadence` | A description, `weekday_weights` (seven whole numbers, Monday first), `local_hours` (first and last hour) and `utc_offset_hours`. |
| `records_per_session` | The fewest and most records in a session, within 3 to 30. |
| `core_quotas` | The six quotas of section 3, for a core share of 125. |

The eight: a founder building software with agent sessions, a home cook and DIYer, a graduate student, a small-business operator, a researcher, a frequent traveler, a parent running a household, and a hobbyist with a technical project.

## 9. Where a pool may be written

As for a question set: `--out` may not be inside this repository, wherever the path points, after resolving symlinks. The one exception is the public pool, under `data/` with `--release` (`data/pool-v1/` for v1). `--release` with a path outside the repository is refused.

## 10. What is deterministic

Everything except the model's answers. Given the same persona files, seed, sizes, prompts and generator, and the same model responses, the generator writes the same files byte for byte, the manifest included, with any `--workers`. The one exception is the manifest's `tool.harness_commit` and `tool.dirty`, which describe the checkout that ran it. The manifest's times and token counts come from the call log, never from the clock.

- **`--replay generation/`** (a pool's generation folder, or any part of its call log) rebuilds a pool into a new folder from its recorded responses, without the API, which is how anyone checks that a pool follows from its record. A request the record does not hold is an error that says so.
- **`--resume`** picks up a run that stopped, in the same folder, with the same command plus `--resume`. Every recorded response is reused for the request that produced it, and only the calls the log does not hold go to the API. The settings in `generation/run.json` must match (seed, model, core, persona files, prompts). A last line cut short by a stopped run is taken out before any new call is added. While a run is going the log is only ever appended to, so a resume that fails or is stopped in its turn loses nothing; the log is rewritten in order only once the pool is written, whole and then moved into place. A resumed run ends with the same files a run that never stopped would have written.
- **`--total` may grow on a resume**, so a pool can be written core first and its tail added later, with every core call reused. The tail's plan depends on the tail's size, though, so tail calls recorded under another total are not reused, and the final log keeps only the calls the pool was written from. The tail's effort may change on a resume too; the core's may not, since its recorded calls were made at it.
- **`--max-calls`** ends a run cleanly before the call that would pass the limit: every call made is in the log, and exit 3 says to resume. That is how a large build runs in chunks (section 13): a chunk is one run with `--max-calls`, then the same command with `--resume`.
- **Line endings never make two runs differ.** `.gitattributes` has git write `prompts/pool/`, `scenarios/`, the top-level `docs/` files and `data/` with LF line endings; the generator reads prompt and persona files with CRLF read as LF before it parses or hashes them; and every file it writes is LF on every operating system.

If a call fails partway through a run, the folder keeps the calls made so far, and the error says to run the same command again with `--resume`.

## 11. Tests

`tests/test_pool.py`, with a faked model and no network:

- the plan: sizes, determinism from the seed, the placement rules for every quota, weighted days inside the timeline, opaque unique session ids, the core plan independent of the tail, quota scaling, quotas too large for a share;
- persona files: the eight shipped ones load, plan without losing an item and carry no secret or personal-data shape; bad files are refused with their name; duplicate and unknown ids; CRLF copies of persona and prompt files hash like the LF files;
- a run: an exact core then the tail, the record shape and the runner's loader, the tail's proportional seeded order, the core independent of `--total`, the same bytes with one worker or four, another seed giving another pool, LF output, the request's fields, the state and the instructions it drives, no folder after a failed first call;
- replay and resume: the same bytes, a request the record does not hold, a stopped run resumed (with a cut last line) ending identical to one that never stopped, other settings refused, a growing total;
- every screen, each with a planted record: secret and personal data (text not kept), too short, entry length, bad shape, a duplicate and a near copy within a persona, a near copy across personas (the first persona keeps it), a tail record restating the core, make-up sessions for short sessions, trimming over the share, a retried reply, a lost session passing its plan on;
- the review fixes: a resume that fails in its turn keeps every recorded call; a cut line, resumed twice over, ends identical to a run that never stopped; no request larger than the largest session size; make-ups spread over the timeline, inside the persona's hours, seeing only their past; secret text in a state update kept out of the state and every later request; a planted step whose record was dropped asked again; planted steps of a lost last session kept in the core; the same bytes with one, four or eight workers when there are copies across personas and make-ups;
- the `--out` rules and the pool's name;
- the manifest statistics against the corpus, the percentiles, the restatement and near-copy counts on a small pool, and the cover index against brute force;
- the build: the restates screen on records written to the shape of pilot 2's twelve drops (the long entry kept, the same-length paraphrase dropped, a short reference never firing) alone and in a run; the tail over-asked by the core's drop rate and trimmed evenly; no make-up under the smallest size; `--max-calls` pausing (exit 3 from the command line) and a chain of resumes ending identical to one run; the call log split into parts, each under the limit, replayed in order, and resumed across parts with a cut last line; the estimate printed before any call, with recorded calls counted on a resume; `--effort` read, recorded per call and checked on a resume;
- the API request, with a fake SDK: streamed, `max_tokens` 32000, the effort per phase, cache control on the system prompt and on the fixed user prefix (never the session), one prefix for every tail session of a persona, cache tokens and cost in the manifest; `--check-api` printing ok, or only the error's class;
- `--api-key-env`, with a fake SDK: the client made from the named variable alone, with the base URL pinned, and the same files as a run without the flag; the key in no output, log record, file or error text, on a run that succeeds and when the SDK raises (making the client, an API error, a `TypeError`, anything else), not even in a printed traceback; a name that is not a variable name (the key itself, say), an unset or empty variable, and a value with a space, a line break or a quote inside, all refused before anything is written; `--estimate` never reading the key; `--check-api` with it;
- `parsica-recall show-pool`.

## 13. Building pool v1

Pool v1 was written by `claude-sonnet-5-5`, the core at effort high and the tail at effort medium, with 8 workers, in chunks: each chunk was one run stopped by `--max-calls` and continued with `--resume`. The folder is `data/pool-v1`, written with `--release`.

**The order:**
1. The core of 1,000 first: `--core 1000 --total 1000`. Its manifest and records were read before the tail ran.
2. Then the tail to 10,000: the same command with `--resume --total 10000`.

The estimate and the run for the core's first chunk:

```bash
parsica-recall generate-pool --out data/pool-v1 --release --seed 1 --core 1000 --total 1000 \
    --effort core=high,tail=medium --workers 8 --max-calls 150 --api-key-env MY_API_KEY --estimate
parsica-recall generate-pool --out data/pool-v1 --release --seed 1 --core 1000 --total 1000 \
    --effort core=high,tail=medium --workers 8 --max-calls 150 --api-key-env MY_API_KEY
```

The later tail, to 30,000 and 50,000 records, is not written yet.

The two raw call logs of pool v1's build (`generation/calls-0001.jsonl` and `calls-0002.jsonl`, the model's replies before screening) are not published. The pool's manifest keeps their SHA-256. So `--replay` cannot rebuild pool v1 from this repository alone; it rebuilds any pool whose call log you hold, your own included.

## 12. Decisions

Settled on October 20, 2026:

1. **One pool, one file order.** `data/pool-v1/corpus.jsonl`: a core of exactly 1,000 records first, then a tail up to 50,000. The core is the question corpus and the 1k track; the larger tracks are prefixes; the core is in `ts` order and the tail in a seeded order. (Amended October 21, 2026: a question set's gold may sit anywhere in the first 10,000 records, the core first; the core stays the 1k track. Spec section 6.)
2. **Eight invented personas**, each with their own agent over 6 to 18 months of sessions. Invented names only, nothing modeled on a real store or a real life.
3. **Every record** has `id` (the content id), `text`, `kind` (`user`, `agent` or `entry`), `ts` (ISO UTC) and `source` (a session id). Entries are 100 to 800 words and 10 to 20 percent of records.
4. **Material for every category in the core**, each core fact stated once, and a tail that continues the same lives and never restates a core fact.
5. **Model** `claude-sonnet-5-5` by default, `--model` as a flag, recorded in the manifest.

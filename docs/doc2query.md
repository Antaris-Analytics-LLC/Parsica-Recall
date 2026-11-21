# Building a benchmark from your own memories

The public question set tells you how a memory system does on a shared pool. This guide is for the other question: how does it do on your memories?

`parsica-recall generate-questions` reads an export of any memory store and writes a question set over it. This is the doc2query direction: it starts from the records and writes the questions an agent would later ask about them, instead of starting from questions and hoping the store holds the answers. Every question is pinned to the records that answer it by id, so scoring stays an id match with no model in it.

The path has four steps:

1. [Export your store](#1-export-your-store) to one file.
2. [Estimate](#2-estimate-the-cost) the cost. No call is made.
3. [Build](#3-build-the-set) the set, with your own API key and a spending limit, on [the provider you choose](#choose-a-provider).
4. [Run the benchmark](#4-run-the-benchmark-on-it) on it.

Read [What leaves your machine](#what-leaves-your-machine) before step 3. Commands are written as `parsica-recall ...`; from a clone with nothing installed, `python -m parsica_recall ...` does the same.

## 1. Export your store

Write your records to a file named `corpus.jsonl`, one JSON object per line, with these five fields:

```
{"id": "mem_00412", "text": "I moved the weekly review to Thursday at 4 pm.", "kind": "user", "ts": "2026-03-10T16:20:00Z", "source": "session_0188"}
```

| Field | Required | What it is | What it unlocks |
| --- | --- | --- | --- |
| `id` | yes | The record's own stable id in your store. Unique. | Scoring. The benchmark matches on it. |
| `text` | yes | The record's text, as the store holds it. A non-empty string. | Everything. |
| `kind` | no | What sort of record it is, in your own words: `user`, `agent`, `note`. | The sample is spread across kinds. |
| `ts` | no | When the record was written: ISO 8601, or epoch seconds. | long_horizon, and the three categories about facts that changed. |
| `source` | no | The session or conversation the record came from. | cross_session. |

Three notes on the export:

- **Export the text your store searches.** If the store rewrites or enriches records, export the original text and keep the id. That is how the benchmark tells whether a question written from a record's own words still reaches the record.
- **Keep old versions in.** If your store keeps a record that was later replaced, export both, each with its own `ts`. Questions about a fact that changed need the old record and the new one.
- **Without `ts` and `source` you still get a set.** It covers eight of the thirteen categories, and the manifest says why the rest are empty.

Nothing about the export is specific to any store. For a store in SQLite it can be this short:

```python
import json, sqlite3

db = sqlite3.connect("file:my_store.db?mode=ro", uri=True)          # read only
with open("corpus.jsonl", "w", encoding="utf-8") as out:
    for rid, text, kind, ts, session in db.execute("SELECT id, text, kind, created_at, session_id FROM memories"):
        out.write(json.dumps({"id": rid, "text": text, "kind": kind, "ts": ts, "source": session}, ensure_ascii=False) + "\n")
```

Every command below takes `--store-ids`. It says the ids in the file are your store's own, and they are kept as given. (A public set uses content ids, the hash of each record's text, and does not take the flag.)

## 2. Estimate the cost

```bash
parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 \
    --seed 7 --store-ids --estimate
```

`--estimate` reads the corpus, plans the run and prints what it expects, per category: the questions it should freeze, the calls, the cost, and the worst case. It makes no call, reads no key and writes nothing.

What to look at:

- **The expected cost and the worst case.** With the default of 25 questions per category, the estimate for a store with `ts` and `source` comes to about $4.50 on `claude-sonnet-5-5` and about $8.80 on `claude-opus-5-5`, with a worst case about a third higher. The cost follows the number of questions, not the size of the store.
- **On another provider.** The estimate's guesses for tokens per call were measured on Claude, and a reasoning model also spends output tokens thinking, so for an OpenAI, Gemini or other model build a small pilot first (`--targets 2`), then estimate again with `--estimate-from` and the pilot's folder. The estimate says so when you run it.
- **Categories marked `short`.** These are expected to end under their target, most often the categories about facts that changed, which need pairs of records.
- **The sample cap.** The generator never shows the model more than half of the eligible records. On a small store that cap is reached first, and the set is smaller.

The estimate is rough until a run has measured the yields. The spending limit in the next step is the guard, and your bill is the authority.

Options that change the cost:

| Flag | What it does |
| --- | --- |
| `--targets` | Questions per category: one number for all, or pairs such as `direct_recall_user=10,abstention=5`. The default is 25. Start small. |
| `--provider` | Who runs the model: `anthropic` (the default), `openai` or `google`. See [Choose a provider](#choose-a-provider). |
| `--model` | The model that writes the questions. Each provider has a default; the models with a price on file are listed under [Choose a provider](#choose-a-provider). A live run needs a price, so the limit can be counted. |
| `--price-in`, `--price-out` | The model's prices in US dollars per million tokens, for a model with no price on file or a price that has gone stale. Recorded in the set. |

## 3. Build the set

The question builder is the one step of Recall that uses a model. It runs on Claude through the Anthropic API by default, on an OpenAI or Gemini model, or on any model behind an endpoint that speaks OpenAI's chat format, a model on your own machine included. Scoring never calls a model.

```bash
pip install "parsica-recall[generate]"          # adds the anthropic SDK
export ANTHROPIC_API_KEY=...                    # your own key; never put it in a file or on a command line

parsica-recall generate-questions --check-api   # optional: one tiny call that proves the key and the network

parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 \
    --seed 7 --store-ids --max-cost 10
```

- `--out` is a folder that does not exist yet, outside the folder the harness sits in. Its name becomes the set's name.
- `--seed` fixes which records are sampled. Another seed gives another set.
- `--max-cost` is required for a live run: the most the whole set may cost, in US dollars. When the cost passes it, the run stops cleanly before its next call and exits 3.
- `--api-key-env NAME` reads the key from another environment variable, and sends every call to the provider's official endpoint only.

### Choose a provider

| Provider | Flags | Install | Key | Default model |
| --- | --- | --- | --- | --- |
| Anthropic | none (the default) | `pip install "parsica-recall[generate]"` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| OpenAI | `--provider openai` | `pip install "parsica-recall[openai]"` | `OPENAI_API_KEY` | `gpt-6.2-sol` |
| Google (the Gemini API) | `--provider google` | `pip install "parsica-recall[google]"` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-4-hydrogen` |
| An endpoint of your own | `--provider openai --base-url URL --model NAME` | `pip install "parsica-recall[openai]"` | none, or the variable you name with `--api-key-env` | none: `--model` is required |

```bash
pip install "parsica-recall[openai]"
export OPENAI_API_KEY=...
parsica-recall generate-questions --provider openai --check-api
parsica-recall generate-questions --provider openai --corpus ~/exports/my-store/corpus.jsonl \
    --out ~/recall-sets/my-store-v1 --seed 7 --store-ids --max-cost 10
```

For Google, the same with `[google]`, `GEMINI_API_KEY` and `--provider google`. Read [what Google's terms say](#what-leaves-your-machine) about its unpaid tier before you send it your store.

`--model` picks another model of the same provider. Prices are on file for:

- Anthropic: `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1`
- OpenAI: `gpt-6.2-astra`, `gpt-6.2-sol`, `gpt-6.1-terra`, `gpt-6-luna`
- Google: `gemini-4-carbon`, `gemini-4-argon`, `gemini-4-neon`, `gemini-4-hydrogen`, `gemini-3.5-flash-lite`

They were taken from each provider's price list in October and November 2026 (Gemini 4 Carbon and Argon at the prices they move to on December 1, 2026), and prices change. For any other model, or when one of these has gone stale, give the prices yourself with `--price-in` and `--price-out`, in US dollars per million tokens. The set records which prices its cost was counted at.

Every provider gets the same request: the category's frozen prompt as the system message, the sampled records as the user message, and the JSON schema the reply must follow (in strict mode on OpenAI). OpenAI and Gemini models are asked to reason at high effort, with room for 32,000 output tokens; their reasoning tokens are billed as output, and the call log and the manifest count them. The prompts were written while building sets with Claude. Every other model follows the same prompts, and every item it writes goes through the same checks and the same lint, so a model that writes worse questions freezes fewer of them; it never lowers the bar.

### A model on your own machine, or any compatible endpoint

`--base-url` sends the calls to any endpoint that speaks OpenAI's chat format: a local server such as Ollama, vLLM or LM Studio, or a hosted service with an OpenAI-compatible API.

```bash
# a local Ollama server, which answers at http://localhost:11434/v1
parsica-recall generate-questions --provider openai --base-url http://localhost:11434/v1 --model MODEL \
    --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 --seed 7 --store-ids \
    --price-in 0 --price-out 0 --max-cost 0
```

- **The model's name.** `--model` takes the name the server knows the model by.
- **The key.** The endpoint gets a key only from the variable you name with `--api-key-env`, never your `OPENAI_API_KEY`, so an OpenAI key cannot reach another service by accident. A key goes over `https://` only, unless the endpoint is on your own machine.
- **The price.** No price is on file for an endpoint of your own. A local model costs nothing per token, so `--price-in 0 --price-out 0` lets `--max-cost 0` pass. For a hosted service, give its prices.
- **What it is sent.** The prompt, the records and the JSON schema, and nothing else: no reasoning setting, and no output cap unless you set one with `--max-output-tokens`.
- **The JSON schema.** The server has to support `response_format` with a JSON schema (structured output); check its documentation. A reply that does not follow the schema yields no items and is counted, so try `--targets 1` first.
- **The context window.** A request runs to about 3,600 input tokens and to 6,200 at most (measured over the 2,431 calls that built v11), and a reply is usually under 1,200 tokens, more for a model that reasons. Give the server a context window of 16,000 tokens or more; some servers start with a smaller one and cut the prompt short.

What happens in the run:

1. Records are sampled per category, spread across kinds and across the age of the store.
2. Each batch goes to the model with the category's frozen prompt (in [prompts/](../prompts/)), and the model answers in a fixed JSON schema.
3. Every item is checked against what the model was shown, then linted against the whole corpus. An item whose answer also appears in another record, or sits in the question, or is cut on a function word, is held or dropped. Nothing is edited into passing.
4. The frozen set is written, with a copy of the corpus, the lint report and the manifest.
5. The oracle and plain BM25 run on the new set. The oracle must score 100, or the set is marked broken.

**If the run stops.** A run that the spending limit stopped, or that a failed call cut short, goes on with the same command plus `--resume`. The calls it already made are replayed from its own log for free. A higher `--max-cost` is how a stopped run spends more.

**Exit status.** 0: the set was written and the oracle scored 100. 1: the oracle check failed. 2: the input was refused or a call failed. 3: the spending limit stopped the run.

### What you get

| File | What it is |
| --- | --- |
| `questions.json` | The questions, each with its answer, its `gold_ids` and the hashes of its gold records |
| `corpus.jsonl` | A byte copy of your export, so the set is self-contained |
| `manifest.json` | The model, the prompt hashes, the seed, the counts, the cost, and every gate's result |
| `lint_report.json` | The lint of the frozen set |
| `reference.json` | The oracle and BM25 result lines on this set |
| `generation/` | The prompts used, every record shown to the model, every call and reply, and every item with its fate |

```bash
parsica-recall show-questions questions ~/recall-sets/my-store-v1      # each question with its gold records
parsica-recall show-questions summary ~/recall-sets/my-store-v1        # the counts and the cost
parsica-recall show-questions held ~/recall-sets/my-store-v1           # what did not freeze, and why
```

**The set says `interim`. That is expected.** A private set is never a standard result: it keeps store ids, and its name is not the harness's current public set. Thin categories fail the 25-per-category gate as well. None of that stops you from using it.

**The secrets gate.** `GATE_NO_SECRETS_OR_PII` fails when a record of your corpus has the shape of a credential or of personal data. It tells you so in the lint report, by record id and kind of shape, without repeating the text. It does not stop the run, and those records were never sent to the model.

### The thirteen categories

| Category | What the question asks |
| --- | --- |
| direct_recall_user | What a person said about themselves or their situation |
| direct_recall_agent | What an assistant or agent said, recommended or decided |
| entity_attribution | Who said, did, made or owns something |
| temporal_awareness | When something happened, how long it took, what came first |
| paraphrase_recall | A fact, asked in clearly different words from the record's |
| vocabulary_gap | Something described without the record's own term for it |
| implicit_preference | A preference a record shows without stating it |
| long_horizon | A lasting fact from the oldest quarter of the store (needs `ts`) |
| cross_session | Something that needs two records from different sessions (needs `source`) |
| knowledge_update | A fact that changed; the newer record should come first (needs `ts`) |
| correction_handling | A fact stated wrongly and then corrected (needs `ts`) |
| recency_preference | A preference that changed (needs `ts`) |
| abstention | A plausible question the store cannot answer |

### A tuning half and a held-out half

If you plan to tune your system against the set, split it so you do not tune on what you test on:

```bash
parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1 \
    --sealed-out ~/recall-sets/my-store-v1-sealed --split-seed 1 --seed 7 --store-ids --max-cost 10
```

`--out` is the half you tune on. `--sealed-out` is a second set of the same build, stratified by category, that you run only at the end.

## 4. Run the benchmark on it

Pick the way that fits your system. [docs/adapting.md](adapting.md) explains each.

**Through a Python adapter, loading the export into a fresh store:**

```bash
parsica-recall run --adapter custom --custom-adapter my_module.MyAdapter --store-ids --verify \
    --questions ~/recall-sets/my-store-v1/questions.json --corpus ~/recall-sets/my-store-v1/corpus.jsonl
```

**Against the store as it stands (live mode):**

```bash
parsica-recall run --live --adapter custom --custom-adapter my_store.MyStoreAdapter --verify --redact \
    --questions ~/recall-sets/my-store-v1/questions.json
```

**With no adapter, from a results file:**

```bash
parsica-recall export-task --out ~/my-task --store-ids \
    --questions ~/recall-sets/my-store-v1/questions.json --corpus ~/recall-sets/my-store-v1/corpus.jsonl
# run ~/my-task/queries.jsonl against your store and write ~/my-task/results.jsonl with your store's ids
parsica-recall score ~/my-task/results.jsonl --store-ids --redact \
    --questions ~/recall-sets/my-store-v1/questions.json --corpus ~/recall-sets/my-store-v1/corpus.jsonl
```

**The floor on your own set.** `reference.json` already holds plain BM25 on your set. It is the number your system has to clear on your own records.

**Before and after a change.** Run the same set before and after you change the store's retrieval, then:

```bash
parsica-recall compare before.json after.json
```

It prints which questions were won and lost. The two runs have to share the question set, the pool and the harness, or the tool refuses and says what differs.

## What leaves your machine

One step sends data anywhere: building the set (step 3). It goes to the provider you chose, under your own key: the Anthropic API, the OpenAI API, Google's Gemini API, or the endpoint you named with `--base-url`. An endpoint on your own machine keeps everything on it.

**Each provider's terms apply to what it is sent.** Read them before you send your store:

- **Google.** Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms) let it use what is sent to its unpaid tier, the free one, to improve its products, and people may read it; they ask you not to send personal or confidential information there. The paid terms do not allow that use, and they apply when the key belongs to a Google Cloud project with billing turned on. Build a set from your own store with such a key only.
- **OpenAI.** The request asks OpenAI not to keep a stored copy of the reply (`store=false`). OpenAI's [policy for API data](https://openai.com/enterprise-privacy/) says how long it keeps requests and whether they train its models.
- **Anthropic.** Anthropic's commercial terms for the API apply.
- **An endpoint of your own.** Whoever runs it.

**What is sent.** The category's frozen prompt, and the text of the sampled records with their `id`, `kind`, `ts` and `source`. Never the whole store: at most half of the eligible records are ever shown. A retry names the words of an answer that leaked and shows no other record.

**What is never sent.** A record with the shape of a credential or of personal data in any field that would be sent: API keys, tokens, connection strings, email addresses, phone numbers, street addresses, government ids, card numbers. Such a record stays in the pool as a distractor.

**The limit of that screen.** It reads shapes, not meaning. A record that says, in plain words, something private about you or someone else has no shape to catch and can be sent. If your store holds records you would not send to an API, take them out of the export first.

**Your key.** It is read from the environment by the provider's SDK (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`), or from the one variable you name with `--api-key-env`. An endpoint of your own gets a key only from that variable. The key is never printed, logged or written, and it is taken out of every error message.

**What is written to disk.** Only the `--out` folder (and `--sealed-out`): the set, a full copy of your export, every record shown to the model, and every call with its reply. Nothing is written until the first reply arrives.

Scoring, linting, comparing and everything else run on your machine and make no network call.

## Keep it private

A set built from your store restates your memories. Its questions quote them and its corpus is a copy of them.

- Keep the set folder, the task folder and the results out of public repositories, issues and chats.
- The generator and the runner refuse to write a private set or live results under the folder the harness sits in, so they cannot be committed with it by accident.
- To share a result, run with `--redact`. It replaces each returned text with its hash and drops the question and answer fields. Ids, ranks and scores stay, which is enough to compare.
- What you can always share from a private run: the one-line result.

## Repeating a build

Current models take no temperature setting, so a model's answers cannot be repeated. Everything else is fixed by the corpus, the seed, the targets and the prompts. Every reply is recorded, and `--replay` rebuilds the set from that record byte for byte, without the API and without a key. Give it the same `--provider`, `--model` and `--base-url` the set was built with:

```bash
parsica-recall generate-questions --corpus ~/exports/my-store/corpus.jsonl --out ~/recall-sets/my-store-v1-check \
    --seed 7 --store-ids --replay ~/recall-sets/my-store-v1/generation/calls.jsonl
```

A set is never edited after it is frozen. A fix or a new sample is a new set with a new name, and every result names the set it ran on.

The design of the generator, with every rule it applies, is in [docs/generation.md](generation.md).

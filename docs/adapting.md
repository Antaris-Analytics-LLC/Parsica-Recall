# Running Recall on any memory system

Recall does not care what a memory system is made of. It hands the system records, asks it questions, and looks at which records come back. There are three ways to connect a system. All three end in the same scorer and the same one-line result.

| Way | What you write | Use it when |
| --- | --- | --- |
| [A results file](#1-a-results-file-no-code-from-this-repository) | A small script in any language | The system is not in Python, is a hosted service, or you want nothing from this repository inside your stack |
| [A Python adapter](#2-a-python-adapter) | One class with five methods | The system has a Python client and you want the harness to run it: verified runs, the extra passes, timing |
| [A live adapter](#3-live-mode-a-store-that-already-holds-its-records) | The same class, read only | The store already holds its records and you want to measure it as it stands |

The six adapters that ship with Recall (`empty`, `random`, `echo`, `bm25`, `bm25stop`, `oracle`) are reference adapters. They are yardsticks, not memory systems: a floor, a check on the chance model, a plain lexical baseline, and a ceiling. No adapter for any product ships here, and that is on purpose. A benchmark that carries its own adapter for a product owns the number that adapter produces.

Commands below are written as `parsica-recall ...`. From a clone with nothing installed, `python -m parsica_recall ...` does the same.

## 1. A results file: no code from this repository

Three steps.

**Step 1. Get the task.**

```bash
parsica-recall export-task --out ~/recall-task
```

That writes three files:

| File | What it is |
| --- | --- |
| `queries.jsonl` | One line per question: `{"qid": "q_0001", "query": "..."}`. The question id and the question text, and nothing else. No answers, no categories, no gold records. |
| `corpus.jsonl` | The records to load into your system, one per line: `id` and `text` on every record, and `kind`, `ts` and `source` where the corpus has them. |
| `task.json` | The hashes of both files, the size of the pool, and the window: how many hits to return per query. |

The window comes from the size of the pool and not from your system. For the current set, 10,000 records, it is 20.

**Step 2. Run your system and write one line per query.**

Load `corpus.jsonl` into your system however it loads things. Keep each record's `id`. Then run every query and write the hits, best first:

```
{"recall_results": 1, "system": {"name": "my-memory", "version": "0.3", "engine_class": "hybrid", "models": ["my-embedder"], "network": false}}
{"qid": "q_0001", "hits": [{"id": "3f9a1c0e7b2d4a61", "score": 12.5}, {"id": "0b72a04ed6f91453", "score": 9.1}]}
{"qid": "q_0002", "hits": ["7eb662465738d319", "8a4a728188fb589a"]}
{"qid": "q_0003", "hits": []}
```

The rules of the file:

- **The first line is optional.** It says what the system is. `engine_class` is one of `lexical`, `embedding`, `hybrid`, `llm_assisted`. Without the line, the result says `unknown`. Say honestly whether a model runs at search time: using one is allowed, hiding it is not.
- **Every question needs exactly one line.** An empty `hits` list says the system returned nothing. A missing line is refused, so a half-finished run can never pass for a whole one.
- **A hit** is a record id as a string, or an object with `id` and, when the system has one, `score` (any number, higher is better).
- **If your system cannot keep record ids,** give `{"text": "..."}` with the record's text word for word. The scorer finds the record from the text. This only works when the text comes back unchanged.
- **Hits past the window are not read.** Returning more earns nothing. A record returned twice burns its second slot.
- **`"ms"` is optional** on a result line: what your system says the search took, in milliseconds. The harness did not time it, and the report says whose number it is.
- **To have abstention scored,** put a `score` on every hit and `"abstain_threshold"` in the `system` block. A question with no answer in the pool is then scored as a correct abstention when the list is empty or the top score is under the threshold.

Any language can write this. In Python, with `my_search` standing in for your system:

```python
import json

with open("queries.jsonl", encoding="utf-8") as queries, open("results.jsonl", "w", encoding="utf-8") as out:
    out.write(json.dumps({"recall_results": 1, "system": {"name": "my-memory", "version": "0.3"}}) + "\n")
    for line in queries:
        q = json.loads(line)
        hits = my_search(q["query"], limit=20)          # your system: returns (record id, score) pairs, best first
        out.write(json.dumps({"qid": q["qid"], "hits": [{"id": i, "score": s} for i, s in hits]}) + "\n")
```

**Step 3. Score it.**

```bash
parsica-recall score ~/recall-task/results.jsonl
```

The report and the results file are the ones an adapter run produces, from the same scorer. The last line is the one-line result:

```
recall/1.0 track=none band=L N=10000 k=15 window=20 qset=v11@2e5b2c03 gold@15=... skill@15=... adapter=file:my-memory@0.3 verified=no standard=no
```

**What a results file cannot carry.** It is one run, made outside the harness. So three things are always true of a scored file, and the result says each of them:

- `verified=no`. Verified means the harness ran the system twice and got the same records. Here the harness did not run the system at all.
- The extra passes (ingest order and query variants) do not run. They need a system to ask.
- `adapter=file:...` marks the result line itself, so nobody mistakes it for a harness run.

The score is as exact as any other. It follows from the file alone, and the file's SHA-256 is in the result's manifest, so anyone holding the file gets the same numbers.

**A set built from your own store** is scored the same way. Name the set's files, and add `--store-ids` when the set keeps your store's own ids:

```bash
parsica-recall export-task --out ~/my-task --questions ~/recall-sets/my-store-v1/questions.json \
    --corpus ~/recall-sets/my-store-v1/corpus.jsonl --store-ids
parsica-recall score ~/my-task/results.jsonl --questions ~/recall-sets/my-store-v1/questions.json \
    --corpus ~/recall-sets/my-store-v1/corpus.jsonl --store-ids --redact
```

If your store already holds those records, skip the loading: run the queries against the store as it is and return its own ids. [docs/doc2query.md](doc2query.md) shows how to build such a set.

## 2. A Python adapter

An adapter is one class. The harness creates it, hands it the records, asks the questions, and scores what comes back. Because the harness runs the system itself, an adapter run can be verified (`--verify` runs it twice and compares), can take the extra passes, and is timed.

### The contract

```python
from parsica_recall.adapters import BaseMemoryAdapter

class MyAdapter(BaseMemoryAdapter):
    def setup(self, workspace, mode="file"): ...   # open the store; file mode gets a fresh, empty folder
    def ingest(self, record): ...                  # one record: {"id", "text", and maybe "kind", "ts", "source", "meta"}
    def save(self): ...                            # called once after the last ingest
    def search(self, query, limit): ...            # up to `limit` hits, best first
    def teardown(self): ...                        # close connections, remove temp files

    # optional
    def version(self): ...                         # what the engine is (below)
    def memory_count(self): ...                    # how many memories the store holds, -1 if unknown
    def list_ids(self): ...                        # live mode: every id search() can return
    def record_ts(self, ids): ...                  # live mode: {id: timestamp} for those ids
```

A hit is `(id, text, score)`, `(id, text)`, or the bare text.

### The worked example: plain BM25

This is a whole adapter. It is the same textbook Okapi BM25 as the built-in `bm25` reference adapter, written out as a file you can copy. Save it as `my_bm25.py`:

```python
import math
import re
from collections import Counter

from parsica_recall.adapters import BaseMemoryAdapter

TOKEN = re.compile(r"[a-z0-9]+")


class MyBM25(BaseMemoryAdapter):
    """Plain Okapi BM25 (k1 = 1.2, b = 0.75), no stemming, no stopwords."""

    K1, B = 1.2, 0.75

    def setup(self, workspace, mode="file"):
        # File mode hands you a fresh, empty folder. This adapter keeps everything in memory.
        if mode != "file":
            raise ValueError("MyBM25 only runs in file mode")
        self.ids, self.texts = [], []

    def ingest(self, record):
        # One call per record. Keep the id you are given: scoring matches on it.
        self.ids.append(record["id"])
        self.texts.append(record["text"])

    def save(self):
        # Called once, after the last ingest. Build the index here.
        self.tfs, df = [], Counter()
        for text in self.texts:
            tf = Counter(TOKEN.findall(text.lower()))
            self.tfs.append((tf, sum(tf.values())))
            df.update(tf.keys())
        n = len(self.texts)
        self.avgdl = sum(dl for _, dl in self.tfs) / n if n else 0.0
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query, limit):
        # The query string is all you see. Return up to `limit` hits, best first.
        terms = sorted(set(TOKEN.findall(query.lower())))
        scored = []
        for i, (tf, dl) in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = self.K1 * (1 - self.B + self.B * dl / self.avgdl) if self.avgdl else self.K1
                    s += self.idf[t] * f * (self.K1 + 1) / (f + norm)
            if s > 0:
                scored.append((-s, i))   # a tie breaks on ingest order, the same way every run
        scored.sort()
        return [(self.ids[i], self.texts[i], -neg) for neg, i in scored[:limit]]

    def teardown(self):
        pass

    def version(self):
        return {"version": "0.1", "engine_class": "lexical", "models": [], "network": False}
```

Run it from the folder that holds the file:

```bash
parsica-recall run --adapter custom --custom-adapter my_bm25.MyBM25 --verify
```

On the current set it scores exactly what the built-in adapter scores, `gold@15=87.58`, with the same per-question records. That is the number a real system has to clear to be interesting.

To wrap a real system, keep the shape and change the insides: `setup` opens your store in the fresh folder, `ingest` adds one record under the id it was given, `save` flushes, and `search` calls your retrieval and returns ids.

### Rules worth knowing before you run

- **Keep the record id and return it.** If you return bare text, the runner hashes the text to find the id, which only works when your store returns records word for word. A store that chunks, merges or rewrites records should carry the id through, or it is scored as if it lost the record.
- **`search` sees the query string and nothing else.** Not the category, not the question id, not the gold ids. An adapter that reads the question file is not a result.
- **The runner asks for `limit = window` once per question** and cuts to the window. Returning more earns nothing.
- **A repeated id burns its slot.** The second copy is dropped and the slot is not refilled.
- **Break ties the same way every time**, by id or by ingest order. Otherwise `--verify` reports the run as unstable, which is not a failure but is printed.
- **Declare the engine.** `version()` returns `{"version", "engine_class", "models", "network"}`. `engine_class` is one of `lexical`, `embedding`, `hybrid`, `llm_assisted`.
- **Scores are optional.** With scores, the report adds calibration. With scores and `"abstain_threshold"` in `version()`, abstention is scored.
- **The module has to be importable.** The folder you run the command from is searched, so `my_bm25.py` next to you is found. For a module elsewhere, put its folder on `PYTHONPATH`.
- **An older adapter still runs.** A class whose `setup` takes only the workspace and whose `ingest` takes `(text, source, category)` is called that way, and its text hits are matched by content id.

### The full run

```bash
parsica-recall run --adapter custom --custom-adapter my_module.MyAdapter --verify --passes order,query
```

`--verify` runs the benchmark twice and compares the per-question records. `--passes order,query` adds two cheap checks: the corpus ingested in a second, shuffled order, and every question asked in two variants. The README says what each reports.

## 3. Live mode: a store that already holds its records

Live mode measures a store as it stands, through its real retrieval path. Nothing is ingested and nothing is written to the store. The question set has to be one that was built from that store, since its gold ids are the store's own ids ([docs/doc2query.md](doc2query.md)).

```bash
parsica-recall run --live --adapter custom --custom-adapter my_store.MyStoreAdapter \
    --questions ~/recall-sets/my-store-v1/questions.json --verify --redact
```

A live adapter is the same class with three differences:

- `setup(workspace, "live")` opens the store read only. `ingest` is never called. Make it raise, so a mistake is loud.
- `list_ids()` returns every id `search` can return. That list is the pool: it gives the run its N, its band, its window and its chance line. Without it the run still prints recall, labeled pool size unknown.
- `record_ts(ids)`, if the store keeps timestamps, returns `{id: timestamp}`. The version tier and the age breakdown need it.

[docs/live_http_adapter.md](live_http_adapter.md) is a complete read-only adapter for a store behind HTTP, to copy and fill in.

Live results hold store content. They are written next to the question file and never under the folder the harness sits in, and `--redact` replaces each returned text with its hash and drops the question and answer fields. Share only redacted results.

If you would rather not write a live adapter, the results file covers this case too: build the set from your store, run the exported queries against the store as it is, and score the file with `--store-ids` (section 1).

## Reading the result

Every run ends with one line:

```
recall/1.0 track=none band=L N=10000 k=15 window=20 qset=v11@2e5b2c03 gold@15=87.58 skill@15=87.56 gold@1=59.66 gold@5=82.36 gold@10=85.89 gold@20=88.5 mrr@20=0.69 version=48.48 chance@15=0.1449 ceiling5=516 adapter=bm25@1.0 verified=yes standard=no
```

- `gold@15` is the headline: the share of answerable questions with a gold record in the top 15. The k comes from the band, and the band from N.
- `skill@15` is the same number with chance taken out. `chance@15` is what a random retriever scores.
- `version` is how often, for a fact that changed, the current record came first.
- `verified` is `yes` only when the harness ran the system twice and got identical records.
- `standard=no` today, for every result. The standard tracks are not built yet, and the current set is interim. The report lists the reasons.

When you share a number, share the whole line. It carries everything a reader needs to know what was measured.

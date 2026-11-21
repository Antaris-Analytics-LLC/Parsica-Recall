# A read-only HTTP search adapter for live mode

Live mode (`parsica-recall run --live`) runs Recall against a memory store that already holds its records. The runner never ingests and never writes. This page shows the shape of an adapter for a store you reach over HTTP. It is a sketch to copy and fill in, not a client for any particular service: the endpoint paths and JSON field names below are placeholders for whatever your service exposes.

## What the adapter has to do

| Method | Live mode |
| --- | --- |
| `setup(workspace, mode)` | Read the base URL and the token from the environment. Refuse any mode but `"live"`. Open nothing that can write. |
| `ingest(record)` | Never called in live mode. Raise, so a mistake is loud. |
| `save()` | Nothing to do. |
| `search(query, limit)` | One read call. Return up to `limit` hits, best first, as `(id, text, score)`. |
| `list_ids()` | Every record id that `search` can return. This is the pool: it gives the run its N, its band and its chance line. Without it the run is labeled pool size unknown and is never a standard result. |
| `record_ts(ids)` | Optional. The store's own timestamp for each of these ids, as ISO 8601 strings (or epoch seconds). The version tier and the age breakdown need it; without it both are `n/a`. The runner calls it once, after the searches. |
| `version()` | What the engine is, honestly: `engine_class`, `models`, and `network: True`. |
| `teardown()` | Close the session. |

## Rules that matter

- **Read only.** Use a token that can search and list and nothing else. The adapter has no code path that writes to the store.
- **The token comes from an environment variable.** Never from code, a command-line flag or a config file in the repository, and never into a log line, an exception message, `version()` or the results file.
- **HTTPS**, except to a server on your own machine.
- **Stable ids.** `gold_ids` in your question set, `list_ids()` and the ids in your hits must all be the same kind of id: the store's stable record id. If the store rewrites or enriches a record, its id must not change. That is how Recall tells a rewritten record (reported as `drifted_gold_count`) from a lost one (a miss).
- **Deterministic order.** Break score ties the same way every time (by id, for example), or `--verify` will report the run as unstable.
- **Timeouts** on every call, so a hung server fails the run instead of stalling it.
- **A quiet store.** The runner fingerprints `list_ids()` at the start and the end of the run. If records are added or removed in between, the run is marked `pool=changed` and `verified=no`. The fingerprint covers ids, not texts, so a record rewritten in place during the run is not caught, except through `drifted_gold_count` for gold records the store returns. Run against a quiet store or a snapshot copy.

## The shape

Standard library only. The three endpoints (`/search`, `/ids`, `/timestamps`) and their JSON fields are placeholders; map them onto your service.

```python
import json
import os
import urllib.parse
import urllib.request

from parsica_recall.adapters import BaseMemoryAdapter


class HttpSearchAdapter(BaseMemoryAdapter):
    """Read-only search over HTTP, for parsica-recall run --live."""

    URL_VAR = "RECALL_STORE_URL"      # e.g. https://memory.example.com/api
    TOKEN_VAR = "RECALL_STORE_TOKEN"  # a token scoped to search and list only
    TIMEOUT_S = 30

    def setup(self, workspace, mode="file"):
        if mode != "live":
            raise ValueError("HttpSearchAdapter only runs in live mode (--live)")
        self.base = os.environ[self.URL_VAR].rstrip("/")
        parts = urllib.parse.urlsplit(self.base)
        if parts.scheme != "https" and parts.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError(f"{self.URL_VAR} must be an https URL")
        token = os.environ.get(self.TOKEN_VAR)
        if not token:
            raise ValueError(f"set {self.TOKEN_VAR} to a read-only token")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=self._headers)
        with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def ingest(self, record):
        raise RuntimeError("live mode never ingests; this adapter is read-only")

    def save(self):
        pass

    def search(self, query, limit):
        out = self._call("POST", "/search", {"query": query, "limit": limit})
        hits = [(h["id"], h.get("text", ""), h.get("score")) for h in out.get("hits", [])]
        return hits[:limit]

    def list_ids(self):
        ids, cursor = [], None
        while True:
            path = "/ids" + ("?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else "")
            page = self._call("GET", path)
            ids.extend(page.get("ids", []))
            cursor = page.get("next")
            if not cursor:
                return ids

    def record_ts(self, ids):
        out = self._call("POST", "/timestamps", {"ids": list(ids)})
        return {i: ts for i, ts in out.get("ts", {}).items() if ts}

    def teardown(self):
        self._headers = {}

    def version(self):
        return {"version": "0.1", "engine_class": "hybrid", "models": ["the embedder your service uses"],
                "network": True}
```

## Running it

Keep the question set and the results outside this repository. Live results default to a file next to the question file, and the runner refuses to write them anywhere under the folder the harness sits in, because they hold store content.

```bash
export RECALL_STORE_URL=https://memory.example.com/api
export RECALL_STORE_TOKEN=...   # read-only; never commit it

parsica-recall run --live --adapter custom --custom-adapter my_http_adapter.HttpSearchAdapter \
    --questions ~/recall/my-store-qset-v1/questions.json --verify --redact
```

`--redact` replaces every returned record text with its `text_hash`, so gold drift can still be checked, and drops the question, answer and alternate fields, which are short enough to guess from a hash. Ids, ranks and scores stay. Share only redacted results, and from a live store share only the one-line result and the manifest hash in places like a ticket or a chat.

"""Reference adapters for Parsica Recall.

Written from scratch for the October 2026 audit of the benchmark's earlier question
set, and kept as the repository's reference adapters: textbook Okapi BM25 (k1 = 1.2,
b = 0.75) plus five baselines that bracket every real result. They are yardsticks, not
memory systems, and no engine code from any product is in them. All of them are
deterministic: ties break on ingest order and the random adapter seeds itself from the
query string.

    empty     returns nothing; the floor
    random    k records drawn uniformly, seeded per query; checks the chance model
    echo      returns the question text itself; shows what echoing the query earns
    bm25      plain Okapi BM25, no stemming, no stopwords
    bm25stop  the same BM25 with a 60-word English stopword list
    oracle    returns the question's gold records; the ceiling. It reads the
              question file (RECALL_QFILE, default the current set's questions.json
              under data/), which no real adapter may do.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List

from .adapters import BaseMemoryAdapter
from .recall_common import CURRENT_QSET, ROOT_DIR, STOPWORDS, current_ids, record_id

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = STOPWORDS  # one list, shared with the query-robustness pass

REFERENCE_VERSION = "1.0"


def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class _Collect(BaseMemoryAdapter):
    """Keeps ingested records in order. Subclasses only define search()."""

    def setup(self, workspace: str, mode: str = "file") -> None:
        if mode != "file":
            raise ValueError(f"{type(self).__name__} only runs in file mode")
        self.ids: List[str] = []
        self.texts: List[str] = []

    def ingest(self, record: Dict[str, Any]) -> None:
        self.ids.append(record["id"])
        self.texts.append(record["text"])

    def save(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def memory_count(self) -> int:
        return len(self.ids)

    def version(self) -> Dict[str, Any]:
        return {"version": REFERENCE_VERSION, "engine_class": "baseline", "models": [],
                "network": False}


class EmptyAdapter(_Collect):
    """Retrieves nothing. Its score is the benchmark's floor."""

    def search(self, query: str, limit: int) -> List[Any]:
        return []


class RandomAdapter(_Collect):
    """Uniform random records, seeded from the query string (independent of hash seed)."""

    def search(self, query: str, limit: int) -> List[Any]:
        rng = random.Random("recall-random|" + query)
        k = min(limit, len(self.ids))
        return [(self.ids[i], self.texts[i], None) for i in rng.sample(range(len(self.ids)), k)]


class EchoAdapter(_Collect):
    """Returns the query text itself and nothing from the store.

    It can never score on the gold tier; on the exact tier it shows how many
    questions carry their own answer."""

    def search(self, query: str, limit: int) -> List[Any]:
        return [(record_id(query), query, None)]


class BM25Adapter(_Collect):
    """Plain Okapi BM25 (k1 = 1.2, b = 0.75), no stemming, no stopwords."""

    k1 = 1.2
    b = 0.75
    use_stop = False

    def _terms(self, text: str) -> List[str]:
        return [t for t in _tok(text) if not (self.use_stop and t in _STOP)]

    def save(self) -> None:
        self.tfs = []
        df: Counter = Counter()
        for text in self.texts:
            toks = self._terms(text)
            tf = Counter(toks)
            self.tfs.append((tf, len(toks)))
            df.update(tf.keys())
        n = len(self.texts)
        self.avgdl = (sum(dl for _, dl in self.tfs) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, limit: int) -> List[Any]:
        terms = sorted(set(self._terms(query)))
        scored = []
        for i, (tf, dl) in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((-s, i))
        scored.sort()
        return [(self.ids[i], self.texts[i], -neg) for neg, i in scored[:limit]]

    def version(self) -> Dict[str, Any]:
        return {"version": REFERENCE_VERSION, "engine_class": "lexical", "models": [],
                "network": False,
                "params": {"k1": self.k1, "b": self.b, "stopwords": self.use_stop}}


class BM25StopAdapter(BM25Adapter):
    """The same BM25 with a small English stopword list."""

    use_stop = True


class OracleAdapter(_Collect):
    """Returns each question's gold records, current version first. The ceiling.

    It looks the question up in the question file, which is exactly what the
    contract forbids a real adapter to do; it exists to prove the question set and
    the scorer agree (it has to score 100 on the gold tier)."""

    def setup(self, workspace: str, mode: str = "file") -> None:
        super().setup(workspace, mode)
        default = os.path.join(ROOT_DIR, "data", CURRENT_QSET, "questions.json")
        path = os.environ.get("RECALL_QFILE", default)
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)
        self.gold: Dict[str, List[str]] = {}
        for q in questions:
            ids = list(q.get("gold_ids") or [])
            current = [c for c in current_ids(q.get("must_rank_first")) if c in ids]
            self.gold[q["question"]] = current + [i for i in ids if i not in current]

    def save(self) -> None:
        self.by_id = dict(zip(self.ids, self.texts))

    def search(self, query: str, limit: int) -> List[Any]:
        return [(i, self.by_id[i], 1.0) for i in self.gold.get(query, []) if i in self.by_id][:limit]

    def version(self) -> Dict[str, Any]:
        v = super().version()
        v["note"] = "reads the question file; a ceiling, not a system"
        return v

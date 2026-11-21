#!/usr/bin/env python3
"""
Parsica Recall question generator (spec recall/1.0, section 9). The design is in
docs/generation.md, and the frozen prompts are in prompts/, one file per category.

Takes any corpus.jsonl (an export of your own store, or the public pool) and writes a
frozen, linted question set with every prompt, seed and model response recorded:

    parsica-recall generate-questions --corpus export/corpus.jsonl --out ~/recall-sets/mystore-v1 \\
        --seed 7 --targets 25 --max-cost 5 [--model claude-sonnet-5-5] [--store-ids]

Generation is the one step of the benchmark that uses a model. Everything else here is
deterministic: the same corpus, seed, targets, prompts and model responses give the
same files byte for byte, and --replay rebuilds a set from its generation/calls.jsonl
without calling the API. A run that --max-cost stopped picks up from its own call log
with --resume, and the calls it holds cost nothing the second time. By default the
anthropic SDK reads the API key from ANTHROPIC_API_KEY and this file never reads it.
With --api-key-env NAME this file reads that one variable, hands it to the SDK with the
base URL pinned to api.anthropic.com, and takes it out of every error it reports. Either
way it never prints, logs or writes the key.

Exit status: 0 when the set was written and the oracle scored 100 on it (or --estimate
printed, or --check-api said ok); 1 when the oracle check failed (the set is marked
broken and its questions are named); 2 on input the generator refuses, or a model call
that failed; 3 when --max-cost stopped the run, which a --resume continues.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import lint_validator as lv
from . import recall_common as rc
from . import run_benchmark as rb
from .adapters import get_adapter
from .reference_adapters import BM25Adapter

REPO_DIR = rc.ROOT_DIR  # the repository's root in a checkout, the package folder in an installed copy
PROMPTS_DIR = os.path.join(REPO_DIR, "prompts")
DATA_DIR = os.path.join(REPO_DIR, "data")
README = os.path.join(REPO_DIR, "README.md")

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_TARGET = 25          # the standard minimum per category (the lint spec's min_category_items)
EFFORT = "high"              # set on every call, so the manifest can say what ran (Opus 5.5's own default is medium)
MAX_TOKENS = 16000
# A category's system prompt is the same on every one of its calls, so it goes as one block
# marked for prompt caching (docs/generation.md section 6): the first call writes it to the
# cache and the next calls read it, at the cache prices in PRICES. The calls go one at a
# time and the categories take turns, so a prompt can wait more than five minutes for its
# category's next turn; the 1-hour cache keeps it (decided October 21, 2026). None would send
# it uncached.
SYSTEM_CACHE: Optional[Dict[str, str]] = {"type": "ephemeral", "ttl": "1h"}
API_URL = "https://api.anthropic.com"  # with --api-key-env, every call goes here, whatever ANTHROPIC_BASE_URL says
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# US dollars per million tokens: input, output, 5-minute cache write, 1-hour cache write,
# cache read (Anthropic's first-party prices as of October 2026). One table for both
# generators (generate_pool.py reads it from here). Used only for the estimate printed
# before a run, the --max-cost guard and the cost a manifest reports; a model not listed
# gets token counts alone.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 4.00, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 8.00, 0.20),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 20.00, 0.25),
}
# For --estimate only (docs/generation.md section 16). Characters per input token, measured
# over pool v1's 1,090 recorded calls (2.36; the request is JSON lines like this one's).
EST_CHARS_PER_TOKEN = 2.3
# Output tokens per call and per retry: rough, from a three-writer comparison on an earlier set
# (Opus 5.5 at effort high wrote about 350 output tokens per item). --estimate-from a
# finished pilot replaces these, and the yields below, with what the pilot measured.
EST_OUTPUT_PER_CALL = {"record": 1500, "pair": 800, "topic": 600}
EST_OUTPUT_PER_RETRY = 500
EST_RETRIES_PER_CALL = 0.3
EST_TRUST_UNITS = 12          # a stage's measured yield over fewer units than this is flagged as too thin to trust
# Publishable items per unit shown: rough guesses until a pilot measures them.
EST_YIELD = {
    "direct_recall_user": 0.5, "direct_recall_agent": 0.3, "entity_attribution": 0.35,
    "temporal_awareness": 0.3, "paraphrase_recall": 0.5, "cross_session": 0.1, "long_horizon": 0.4,
    "vocabulary_gap": 0.35, "knowledge_update": 0.08, "correction_handling": 0.06, "implicit_preference": 0.15,
    "abstention": 0.9, "recency_preference": 0.06,
}
# A planted pair holds a change by construction (docs/generation.md section 5), so a stage
# made of them is guessed at this yield, not a similarity pair's, until a pilot measures it.
EST_YIELD_PLANTED = 0.5

# The sampler (docs/generation.md sections 3 to 5). Every value is recorded in the manifest.
MAX_SAMPLE_SHARE = 0.5       # never the whole store: at most half the eligible records are shown
UNITS_PER_TARGET = 4         # a category may use 4 units per item it needs, and at least MIN_UNITS
MIN_UNITS = 12
BATCH = {"record": 6, "pair": 3, "topic": 1}
ITEMS_PER_UNIT = {"record": 1, "pair": 1, "topic": 2}
TOPIC_NEIGHBORS = 3
PAIR_MIN_SIMILARITY = 0.3
NEAR_COPY_OVERLAP = 0.9      # the lint's gold_near_copy threshold, both ways
LONG_HORIZON_MIN_TS = 8
COMMON_TOKEN_SHARE = 0.05    # tokens in more of the pool than this are skipped when gathering candidates
COMMON_TOKEN_FLOOR = 100

# The abstention screen and the form an abstention item freezes in (section 7).
PROBE_MAX_TOKENS = 6
TOPIC_MIN_SHARED = 2
ABSTENTION_ANSWER = "not mentioned"
ABSTENTION_ALTERNATES = ["not mentioned", "not specified", "no information", "not discussed", "unknown"]

AGE_BANDS = ("oldest", "older", "newer", "newest")
NO_TS = "no ts"

RECORD_CATEGORIES = ("direct_recall_user", "direct_recall_agent", "entity_attribution", "temporal_awareness",
                     "paraphrase_recall", "long_horizon", "vocabulary_gap", "implicit_preference")
PAIR_CATEGORIES = rc.VERSION_CATEGORIES + rc.ALL_GOLD_CATEGORIES
UNIT_KIND = {**{c: "record" for c in RECORD_CATEGORIES}, **{c: "pair" for c in PAIR_CATEGORIES},
             "abstention": "topic"}
PAIR_LABELS = {**{c: ("earlier", "later") for c in rc.VERSION_CATEGORIES}, "cross_session": ("first", "second")}
assert set(UNIT_KIND) == set(rc.CATEGORIES)

# A pool's record of its planted changes (--planted-pairs, written by export_planted_pairs.py;
# docs/generation.md section 5): the item types it holds and the version category each feeds.
PLANTED_CATEGORY = {"version_chain": "knowledge_update", "correction": "correction_handling",
                    "preference_change": "recency_preference"}
PLANTED_FIELDS = ("type", "label", "persona", "earlier", "later")
PADDED_RETRY = "padded retry"  # the reason a retried item is dropped for lengthening the held answer

FATES = ("frozen", "surplus", "held_by_lint", "dropped_by_lint", "dropped_by_checks", "dropped_by_screen")


class GenerateError(Exception):
    """Input the generator refuses, or a model call that failed. Exit 2."""


def _dump(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every operating system
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _lf(data: bytes) -> bytes:
    """Bytes with CRLF line endings read as LF. A checkout can store a file either way
    (git's core.autocrlf turns LF into CRLF on Windows); its hash must not depend on that."""
    return data.replace(b"\r\n", b"\n")


def _lf_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(_lf(f.read())).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scalar(value: Any) -> Optional[str]:
    """A record field as a string, or None when it is missing, blank or not a plain value."""
    if value is None or isinstance(value, (bool, dict, list)):
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SCHEMA_BLOCK = re.compile(r"^BEGIN SCHEMA\n(.*?)\nEND SCHEMA$", re.S | re.M)


class Prompt:
    """One category's frozen prompt: the file's text with its line endings read as LF,
    which is the system prompt, the SHA-256 of those LF bytes, and the JSON schema between
    BEGIN SCHEMA and END SCHEMA, which is the response format. One hash covers both, and a
    checkout that stores the file with CRLF gives the same text, schema and hash."""

    def __init__(self, category: str, path: str):
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as exc:
            raise GenerateError(f"cannot read the {category} prompt: {exc}")
        self.category = category
        self.path = path
        self.data = _lf(raw)  # the bytes that are hashed, sent and copied into the set
        self.text = self.data.decode("utf-8")
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        m = _SCHEMA_BLOCK.search(self.text)
        if not m:
            raise GenerateError(f"{path}: no JSON schema between BEGIN SCHEMA and END SCHEMA")
        try:
            self.schema = json.loads(m.group(1))
        except ValueError as exc:
            raise GenerateError(f"{path}: the schema is not valid JSON ({exc})")
        self.fields = set(self.schema["properties"]["items"]["items"]["properties"])


def load_prompts(categories: Sequence[str], prompts_dir: str = PROMPTS_DIR) -> Dict[str, Prompt]:
    return {c: Prompt(c, os.path.join(prompts_dir, f"{c}.md")) for c in categories}


# ---------------------------------------------------------------------------
# The corpus: similarity, near copies, eligibility, age bands
# ---------------------------------------------------------------------------

_FIRST_WORD_CHAR = re.compile(r"[^\W_]")


def capital_tokens(text: str) -> frozenset:
    """The normalized forms of the words written with a capital first letter."""
    out = set()
    for piece in unicodedata.normalize("NFKC", str(text)).split():
        m = _FIRST_WORD_CHAR.search(piece)
        if m and m.group(0).isupper():
            norm = rc.normalize(piece)
            if norm:
                out.add(norm)
    return frozenset(out)


def digit_tokens(tokens: Sequence[str]) -> frozenset:
    """The normalized tokens that contain a digit."""
    return frozenset(t for t in tokens if any(ch.isdigit() for ch in t))


# entity_attribution draws only from records that hold a rare name (docs/generation.md
# section 4): a capitalized word, or a run of capitalized words, found in at most
# RARE_NAME_MAX_OTHERS other records of the whole pool outside the record's near-copy
# cluster. Decided October 22, 2026, after the v11 pilot: its entity answers were recurring
# names (Kroger in 192 other records), so 19 of 23 items leaked. The lint is unchanged.
RARE_NAME_MAX_OTHERS = 0
RARE_NAME_MIN_CHARS = 3      # a name's letters and digits, once normalized
_NAME_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_SENTENCE_END = ".?!:\n\"'“‘)]-*|"


_POSSESSIVE = re.compile(r"['’]s\b")


def _stem(word: str) -> str:
    """A name word without its possessive: "Kennelbook's" is Kennelbook, so a possessive
    counts as its stem and is rare only when the stem is (a review on October 22, 2026
    found that 27 of 956 records qualified only through a possessive)."""
    return word[:-2] if word.endswith(("'s", "’s")) else word


def stemmed_text(text: str) -> str:
    """A record's normalized text with every possessive stemmed the way record_names stems
    a name word, so a name is counted in the records that write it as a possessive too."""
    return rc.normalize(_POSSESSIVE.sub("", unicodedata.normalize("NFKC", str(text))))


def record_names(text: str) -> List[str]:
    """The names a record writes, as normalized phrases, in order and without repeats: each
    run of two or more capitalized words separated by spaces ("Dana Whitfield", "Rillito
    Valley Vet"), and each capitalized word of two or more characters on its own. A capital
    that starts a sentence is not a name on its own ("Picked up a skillet"), and a run that
    starts a sentence drops a leading function word ("Then Marcus called" gives marcus) but
    keeps a name ("Dana Whitfield fixed the valve" gives dana whitfield). A possessive
    counts as its stem ("Kennelbook's" gives kennelbook)."""
    text = unicodedata.normalize("NFKC", str(text))
    words = [(m.start(), m.end(), _stem(m.group(0))) for m in _NAME_WORD.finditer(text)]
    capital = [w[0].isupper() and len(w) >= 2 for _, _, w in words]
    names: List[str] = []
    i = 0
    while i < len(words):
        if not capital[i]:
            i += 1
            continue
        j = i + 1
        while j < len(words) and capital[j] and text[words[j - 1][1]:words[j][0]].strip(" \t") == "":
            j += 1
        before = text[:words[i][0]].rstrip(" \t")
        start = i
        if not before or before[-1] in _SENTENCE_END:  # the run starts a sentence
            if j - i < 2 or rc.normalize(words[i][2]) in rc.FUNCTION_WORDS or rc.normalize(words[i][2]) in rc.STOPWORDS:
                start = i + 1
        alone = j - start == 1 and len(words[start][2]) >= 3  # a word on its own needs three characters ("Dr" is not a name)
        if j - start >= 2 or (alone and (start > i or (before and before[-1] not in _SENTENCE_END))):
            norm = rc.normalize(" ".join(w for _, _, w in words[start:j]))
            if norm and norm not in names:
                names.append(norm)
        i = j
    return names


def rare_name_records(corpus: "Corpus", reps: Sequence[str], max_others: int = RARE_NAME_MAX_OTHERS
                      ) -> Dict[str, List[str]]:
    """The eligible records (by their representatives) that hold a rare name, each with the
    rare names it holds: a name of record_names found, as whole tokens, in at most
    max_others records of the whole pool outside the record's near-copy cluster, and at
    least RARE_NAME_MIN_CHARS letters and digits long. The texts are searched with their
    possessives stemmed (stemmed_text), as the names are, so "Kennelbook's" and
    "Kennelbook" count as one name."""
    index = rc.PhraseIndex([stemmed_text(corpus.by_id[rid]["text"]) for rid in corpus.ids])
    holders: Dict[str, set] = {}  # a name -> the positions of the records that hold it, computed once

    def records_holding(norm: str) -> set:
        found = holders.get(norm)
        if found is None:
            sets = [index.postings.get(t) for t in norm.split()]
            if any(s is None for s in sets):
                found = set()
            elif len(sets) == 1:
                found = sets[0]
            else:
                found = {i for i in set.intersection(*sorted(sets, key=len)) if rc.contains_phrase(index.norm[i], norm)}
            holders[norm] = found
        return found

    out: Dict[str, List[str]] = {}
    for rid in reps:
        cluster = {corpus.pos[m] for m in corpus.cluster[rid]}
        rare = []
        for name in record_names(corpus.by_id[rid]["text"]):
            if len(name.replace(" ", "")) < RARE_NAME_MIN_CHARS:
                continue
            if len(records_holding(name) - cluster) <= max_others:
                rare.append(name)
        if rare:
            out[rid] = rare
    return out


def _union_find(ids: Sequence[str], pos: Dict[str, int]) -> Tuple[Callable[[str], str], Callable[[str, str], None]]:
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb_ = find(a), find(b)
        if ra != rb_:  # the record earliest in the corpus file stays the root
            parent[max(ra, rb_, key=pos.get)] = min(ra, rb_, key=pos.get)

    return find, union


class Corpus:
    """The pool, with what the sampler, the screen and the lint need computed once."""

    def __init__(self, records: List[Dict[str, Any]], spec: Dict[str, Any], store_ids: bool):
        self.records = records
        self.ctx = lv.Context(records, spec, store_ids=store_ids)  # the lint's own view of the pool
        self.ids = self.ctx.ids
        self.by_id = self.ctx.by_id
        self.pos = {rid: i for i, rid in enumerate(self.ids)}
        self.N = len(self.ids)
        self.tokens = self.ctx.tokens_by_id
        self.content = {rid: frozenset(t for t in self.tokens[rid] if t not in rc.STOPWORDS) for rid in self.ids}
        df = Counter(t for rid in self.ids for t in self.content[rid])
        # The reference BM25's idf. math.fsum is exact, so no sum depends on set order.
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.weight = {rid: math.fsum(self.idf[t] for t in self.content[rid]) for rid in self.ids}
        self.postings: Dict[str, List[str]] = defaultdict(list)
        for rid in self.ids:
            for t in self.content[rid]:
                self.postings[t].append(rid)
        self.common = max(COMMON_TOKEN_FLOOR, int(COMMON_TOKEN_SHARE * self.N))
        self.ts = {rid: rc.parse_ts(self.by_id[rid].get("ts")) for rid in self.ids}
        self.source = {rid: _scalar(self.by_id[rid].get("source")) for rid in self.ids}
        self.kind = {rid: _scalar(self.by_id[rid].get("kind")) or "none" for rid in self.ids}
        self.cluster = self._near_copy_clusters()

    def similarity(self, a: str, b: str) -> float:
        """The idf weight of the content tokens two records share, over the smaller of
        their two weights: 0 for no shared word, 1 when one record's words are all in
        the other."""
        shared = self.content[a] & self.content[b]
        denom = min(self.weight[a], self.weight[b])
        if not shared or denom <= 0:
            return 0.0
        return math.fsum(self.idf[t] for t in shared) / denom

    def related(self, rid: str, keep: Callable[[str], bool]) -> List[Tuple[float, str]]:
        """Records on one topic with rid (similarity at least PAIR_MIN_SIMILARITY) that
        keep() accepts, most similar first, ties by id. The candidates are the records
        sharing one of rid's less common words; words in more than a few percent of the
        pool weigh little and gather nothing, though the similarity counts every shared
        word. The shared weight of the less common words is summed through the postings,
        and only a candidate whose upper bound (that weight plus all of rid's common
        words) can reach the threshold gets the exact similarity, so a large store stays
        fast and the result is the same."""
        common_weight = math.fsum(self.idf[t] for t in self.content[rid] if len(self.postings[t]) > self.common)
        score: Dict[str, float] = defaultdict(float)
        for t in sorted(self.content[rid]):
            post = self.postings[t]
            if len(post) <= self.common:
                w = self.idf[t]
                for c in post:
                    score[c] += w
        score.pop(rid, None)
        own = self.weight[rid]
        out = []
        for c, s in score.items():
            if s + common_weight + 1e-9 < PAIR_MIN_SIMILARITY * min(own, self.weight[c]) or not keep(c):
                continue
            exact = self.similarity(rid, c)
            if exact >= PAIR_MIN_SIMILARITY:
                out.append((exact, c))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def _near_copy_clusters(self) -> Dict[str, List[str]]:
        """Each record's cluster of near copies, in corpus order. Two records are near
        copies when each covers at least NEAR_COPY_OVERLAP of the other's tokens and
        they carry the same digit tokens and the same capitalized words: a copy keeps
        the numbers and names, an update changes them. Clusters are the connected
        components, so the first member in the corpus file is the representative."""
        digits = {rid: digit_tokens(self.tokens[rid]) for rid in self.ids}
        caps = {rid: capital_tokens(self.by_id[rid]["text"]) for rid in self.ids}
        postings: Dict[str, List[str]] = defaultdict(list)
        for rid in self.ids:
            for t in set(self.tokens[rid]):
                postings[t].append(rid)
        find, union = _union_find(self.ids, self.pos)
        for a in self.ids:
            toks = self.tokens[a]
            n = len(toks)
            if not n:
                continue
            # A record missing k of a's distinct tokens misses at least k of its token
            # occurrences, so every near copy holds one of a's few rarest tokens (one
            # spare in k against float rounding).
            k = int(n * (1 - NEAR_COPY_OVERLAP)) + 2
            rare = sorted(set(toks), key=lambda t: (len(postings[t]), t))[:k]
            cands = set()
            for t in rare:
                cands.update(postings[t])
            for b in cands:
                if self.pos[b] <= self.pos[a] or digits[a] != digits[b] or caps[a] != caps[b]:
                    continue
                tb = self.tokens[b]
                if min(rc.token_overlap(toks, tb), rc.token_overlap(tb, toks)) >= NEAR_COPY_OVERLAP:
                    union(a, b)
        groups: Dict[str, List[str]] = defaultdict(list)
        for rid in self.ids:
            groups[find(rid)].append(rid)
        return {rid: groups[find(rid)] for rid in self.ids}

    def has_shape(self, rid: str) -> bool:
        """Whether a field the model would see has a credential or personal-data shape: the
        text, the kind and the source, a ts that is not a readable timestamp, and the id.
        An id may look random, since store ids often do; every other shape counts there."""
        rec = self.by_id[rid]
        fields = [rec["text"], rec.get("kind"), rec.get("source")]
        if self.ts[rid] is None:
            fields.append(rec.get("ts"))
        if any(lv.scan_secrets_pii(_scalar(v)) for v in fields if _scalar(v)):
            return True
        return any(hit["shape"] != "high_entropy_token" for hit in lv.scan_secrets_pii(rid))

    def window_size(self, window: Optional[int]) -> int:
        """How many leading records gold may come from: window, or the whole corpus."""
        return self.N if window is None else min(window, self.N)

    def eligibility(self, min_tokens: int, window: Optional[int] = None) -> Tuple[List[str], Dict[str, str]]:
        """The representatives of clusters whose records may all be shown to the model,
        in corpus order, and why each other record of the gold window may not. A record is
        eligible when it sits in the window (the first window records of the corpus file),
        with at least min_tokens tokens and no credential or personal-data shape in any
        field the model would see; a cluster with one ineligible member, or one member past
        the window, is out whole, so nothing ineligible and nothing past the window becomes
        gold. Records past the window are never shown; they stay in the pool."""
        limit = self.window_size(window)
        why: Dict[str, str] = {}
        for rid in self.ids[:limit]:
            if len(self.tokens[rid]) < min_tokens:
                why[rid] = "too_short"
            elif self.has_shape(rid):
                why[rid] = "secret_or_pii"
        reps = []
        for rid in self.ids[:limit]:
            members = self.cluster[rid]
            if members[0] != rid:
                continue
            if any(m in why for m in members):
                for m in members:
                    if self.pos[m] < limit:
                        why.setdefault(m, "near_copy_of_an_excluded_record")
                continue
            if any(self.pos[m] >= limit for m in members):
                for m in members:
                    if self.pos[m] < limit:
                        why.setdefault(m, "near_copy_past_the_gold_window")
                continue
            reps.append(rid)
        return reps, why

    def age_bands(self, window: Optional[int] = None) -> Dict[str, str]:
        """Quartiles of the gold window's own ts (ties by id), never the clock; no ts is its
        own band. Without a window, the window is the whole corpus."""
        limit = self.window_size(window)
        timed = sorted((self.ts[rid], rid) for rid in self.ids[:limit] if self.ts[rid] is not None)
        bands = {rid: NO_TS for rid in self.ids}
        for i, (_, rid) in enumerate(timed):
            bands[rid] = AGE_BANDS[(4 * i) // len(timed)]
        return bands


# ---------------------------------------------------------------------------
# Draw orders, pairs and topics
# ---------------------------------------------------------------------------

def proportional_order(groups: Dict[str, List[str]]) -> Iterator[str]:
    """Interleaves the groups so every stretch of the result keeps their proportions: at
    each step the group furthest behind its share goes next, ties by group name. Integer
    arithmetic, so the order never depends on rounding."""
    keys = sorted(k for k, v in groups.items() if v)
    total = sum(len(groups[k]) for k in keys)
    taken = dict.fromkeys(keys, 0)
    for step in range(1, total + 1):
        best, best_deficit = None, None
        for k in keys:
            size = len(groups[k])
            if taken[k] < size:
                deficit = step * size - taken[k] * total
                if best is None or deficit > best_deficit:
                    best, best_deficit = k, deficit
        yield groups[best][taken[best]]
        taken[best] += 1


def seeded_order(ids: Sequence[str], stratum: Dict[str, str], seed: int, corpus_sha: str,
                 stream: str) -> Iterator[str]:
    """One category's (or one pair builder's) order: within each stratum the ids are
    sorted and then shuffled by a generator seeded with the seed, the corpus hash and
    the stream's name; the strata are interleaved in proportion to their sizes."""
    rng = random.Random(f"recall/1.0 generate_questions|{seed}|{corpus_sha}|{stream}")
    groups: Dict[str, List[str]] = defaultdict(list)
    for rid in ids:
        groups[stratum[rid]].append(rid)
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        groups[key] = members
    return proportional_order(groups)


class Plan:
    """A category's units in the order they are offered, its unit cap, and the reason
    when the corpus cannot give it any. With a core, the first n_core units are drawn
    from the core and the rest from the remainder of the gold window."""

    def __init__(self, category: str, units: List[Tuple[str, ...]], cap: int, reason: Optional[str] = None,
                 n_core: Optional[int] = None, planted_stages: Sequence[str] = ()):
        self.category = category
        self.kind = UNIT_KIND[category]
        self.units = units
        self.cap = cap
        self.reason = reason
        self.n_core = n_core
        self.planted_stages = frozenset(planted_stages)  # the stages whose units are all planted pairs


def unit_cap(target: int) -> int:
    return max(UNITS_PER_TARGET * target, MIN_UNITS) if target > 0 else 0


def parse_unit_caps(text: Optional[str]) -> Dict[str, int]:
    """--unit-cap: category=N pairs; a category left out keeps its cap of UNITS_PER_TARGET
    times its target (at least MIN_UNITS)."""
    out: Dict[str, int] = {}
    if text is None or not str(text).strip():
        return out
    for part in str(text).split(","):
        cat, sep, num = part.strip().partition("=")
        if not sep or cat.strip() not in rc.CATEGORIES or not re.fullmatch(r"[1-9]\d*", num.strip()):
            raise GenerateError(f"--unit-cap: cannot read {part.strip()!r}; give category=number pairs such as "
                                "knowledge_update=1200,implicit_preference=900, each number 1 or more")
        out[cat.strip()] = int(num.strip())
    return out


def read_planted_pairs(path: str) -> List[Dict[str, str]]:
    """A pool's record of its planted changes, as export_planted_pairs.py writes it: one
    JSON object per line with exactly PLANTED_FIELDS, each a non-empty string, and a type
    in PLANTED_CATEGORY. Anything else is refused, since the file is the pool's own record."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().split("\n") if line.strip()]
    except OSError as exc:
        raise GenerateError(f"--planted-pairs: cannot read {path}: {exc}")
    rows = []
    for n, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise GenerateError(f"--planted-pairs {path}: line {n} is not JSON ({exc})")
        if not isinstance(row, dict) or set(row) != set(PLANTED_FIELDS) \
                or not all(isinstance(row[k], str) and row[k].strip() for k in PLANTED_FIELDS):
            raise GenerateError(f"--planted-pairs {path}: line {n} is not an object with the fields "
                                f"{', '.join(PLANTED_FIELDS)}, each a non-empty string")
        if row["type"] not in PLANTED_CATEGORY:
            raise GenerateError(f"--planted-pairs {path}: line {n}: unknown type {row['type']!r}; one of "
                                f"{', '.join(PLANTED_CATEGORY)}")
        rows.append(row)
    if not rows:
        raise GenerateError(f"--planted-pairs {path}: the file holds no pairs")
    return rows


def route_planted_pairs(rows: Sequence[Dict[str, str]], corpus: Corpus, reps: Sequence[str],
                        window: Optional[int]) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Any]]:
    """The planted pairs each version category may use, in file order, and what the
    manifest records of them: pairs read per type, used per category, left out by reason.
    Each record stands for its near-copy cluster (its representative, which is what the
    model is shown). A pair is used when both memories are eligible in the gold window, are
    two different memories, carry a ts with the earlier one before the later, and no earlier
    line gave the same pair or the same later memory."""
    eligible = set(reps)
    limit = corpus.window_size(window)
    routed: Dict[str, List[Tuple[str, str]]] = {c: [] for c in rc.VERSION_CATEGORIES}
    read: Counter = Counter()
    skipped: Counter = Counter()
    seen_pairs: set = set()
    seen_later: set = set()
    for row in rows:
        read[row["type"]] += 1
        ids = [row["earlier"], row["later"]]
        reason = None
        if any(r not in corpus.by_id for r in ids):
            reason = "a record not in the corpus"
        elif any(corpus.pos[r] >= limit for r in ids):
            reason = "a record past the gold window"
        else:
            earlier, later = (corpus.cluster[r][0] for r in ids)
            if earlier == later:
                reason = "the same memory twice"
            elif earlier not in eligible or later not in eligible:
                reason = "a memory that is not eligible"
            elif corpus.ts[earlier] is None or corpus.ts[later] is None:
                reason = "a record without ts"
            elif not corpus.ts[earlier] < corpus.ts[later]:
                reason = "the later record's ts is not after the earlier's"
            elif (earlier, later) in seen_pairs or later in seen_later:
                reason = "a pair or later record an earlier line gave"
        if reason:
            skipped[reason] += 1
            continue
        seen_pairs.add((earlier, later))
        seen_later.add(later)
        routed[PLANTED_CATEGORY[row["type"]]].append((earlier, later))
    return routed, {"read": dict(sorted(read.items())), "used": {c: len(v) for c, v in routed.items()},
                    "skipped": dict(sorted(skipped.items()))}


def planted_order(pairs: Sequence[Tuple[str, str]], stratum: Dict[str, str], seed: int, corpus_sha: str,
                  stream: str) -> List[Tuple[str, str]]:
    """Planted pairs in one seeded order, drawn like records by their later records."""
    by_later: Dict[str, Tuple[str, str]] = {}
    for earlier, later in pairs:
        by_later.setdefault(later, (earlier, later))
    return [by_later[r] for r in seeded_order(list(by_later), stratum, seed, corpus_sha, stream)]


def version_pairs(corpus: Corpus, reps: List[str], stratum: Dict[str, str], seed: int, corpus_sha: str,
                  caps: Dict[str, int], stream: str = "version pairs") -> Dict[str, List[Tuple[str, str]]]:
    """Pairs of records on one topic with different ts, as (earlier, later), dealt in turn
    to the version categories that want them, so a pair goes to one category only. Each
    anchor, in one seeded order, pairs with its most similar record among reps."""
    active = [c for c in rc.VERSION_CATEGORIES if caps.get(c, 0) > 0]
    out: Dict[str, List[Tuple[str, str]]] = {c: [] for c in rc.VERSION_CATEGORIES}
    timed = [r for r in reps if corpus.ts[r] is not None]
    timed_set = set(timed)
    seen = set()
    turn = 0
    for anchor in seeded_order(timed, stratum, seed, corpus_sha, stream):
        if all(len(out[c]) >= caps[c] for c in active):
            break
        ts = corpus.ts[anchor]
        near = corpus.related(anchor, lambda c: c in timed_set and corpus.ts[c] != ts)
        if not near:
            continue
        earlier, later = sorted((anchor, near[0][1]), key=lambda r: (corpus.ts[r], r))
        if (earlier, later) in seen:
            continue
        seen.add((earlier, later))
        for _ in range(len(active)):
            cat = active[turn % len(active)]
            turn += 1
            if len(out[cat]) < caps[cat]:
                out[cat].append((earlier, later))
                break
    return out


def session_pairs(corpus: Corpus, reps: List[str], stratum: Dict[str, str], seed: int, corpus_sha: str,
                  cap: int, stream: str = "cross_session") -> List[Tuple[str, str]]:
    """Pairs of records on one topic from different sessions (source values), in corpus order."""
    sourced = [r for r in reps if corpus.source[r] is not None]
    sourced_set = set(sourced)
    out: List[Tuple[str, str]] = []
    seen = set()
    for anchor in seeded_order(sourced, stratum, seed, corpus_sha, stream):
        if len(out) >= cap:
            break
        src = corpus.source[anchor]
        near = corpus.related(anchor, lambda c: c in sourced_set and corpus.source[c] != src)
        if not near:
            continue
        pair = tuple(sorted((anchor, near[0][1]), key=corpus.pos.get))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


AFTER_THE_CORE = " after the core"  # the stream-name suffix of the second stage's draw orders


def build_plans(corpus: Corpus, reps: List[str], bands: Dict[str, str], targets: Dict[str, int], seed: int,
                corpus_sha: str, core: Optional[int] = None, core_quota: Optional[int] = None,
                unit_caps: Optional[Dict[str, int]] = None, planted: Optional[Dict[str, List[Tuple[str, str]]]] = None,
                rare_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Plan]:
    """Every category's units in the order they are offered. With core, the units come in
    two stages, each built from its own records alone: first the core (the first core
    records of the corpus), then the rest of the gold window, and every unit's records sit
    wholly in one stage. The unit cap covers both stages. Without core_quota a category
    takes what the core can give before it reaches past it; with one, its core stage holds
    at most half the unit cap (the run ends the stage sooner once core_quota items from
    the core have frozen), and the tail stage holds up to the whole cap, since the core
    stage may end early. unit_caps replaces a category's cap (--unit-cap); planted is a
    pool's record of its planted changes, routed by route_planted_pairs; rare_names is
    what rare_name_records found, computed here when entity_attribution needs it."""
    stratum = {rid: f"{corpus.kind[rid]} / {bands[rid]}" for rid in reps}
    caps = {}
    for c in rc.CATEGORIES:
        target = targets.get(c, 0)
        caps[c] = ((unit_caps or {}).get(c) or unit_cap(target)) if target > 0 else 0
    n_timed = sum(1 for rid in corpus.ids if corpus.ts[rid] is not None)
    timed_reps = sum(1 for r in reps if corpus.ts[r] is not None)
    sources = {corpus.source[r] for r in reps if corpus.source[r] is not None}
    if core is None:
        stages = [("", reps)]
    else:
        stages = [("", [r for r in reps if corpus.pos[r] < core]),
                  (AFTER_THE_CORE, [r for r in reps if corpus.pos[r] >= core])]

    def stage_left(cap: int, first: bool, used: int) -> int:
        """The units a stage may add, after the used units of the stages before it."""
        if core_quota is None:
            return cap - used
        return cap // 2 if first else cap

    pairs: Dict[str, List[Tuple[str, str]]] = {c: [] for c in rc.VERSION_CATEGORIES}
    planted_in: Dict[str, set] = {c: set() for c in rc.VERSION_CATEGORIES}  # the stages made of planted pairs alone
    core_pairs: Dict[str, int] = {}
    for suffix, stage in stages:
        left = {c: stage_left(caps[c], not suffix, len(pairs[c])) for c in rc.VERSION_CATEGORIES}
        stage_name = "all" if core is None else ("tail" if suffix else "core")
        if planted is not None:
            # The pool's own record of its planted changes (docs/generation.md section 5):
            # each version category takes the pairs of its type that sit wholly in this
            # stage, in seeded order. The core stage takes nothing else. Past the core (or
            # in the one stage without a core) knowledge_update goes on with similarity
            # pairs, since the tail's updates are not recorded, while corrections and
            # preference changes come from the record alone.
            here = set(stage)
            for c in rc.VERSION_CATEGORIES:
                mine = [p for p in planted.get(c, []) if p[0] in here and p[1] in here]
                take = planted_order(mine, stratum, seed, corpus_sha, c + " planted" + suffix)[:max(left[c], 0)]
                pairs[c] += take
                left[c] -= len(take)
                if take:
                    planted_in[c].add(stage_name)
            similar = {c: (max(left[c], 0) if c == "knowledge_update" and stage_name != "core" else 0)
                       for c in rc.VERSION_CATEGORIES}
        else:
            similar = left
        if any(n > 0 for n in similar.values()):
            seen = {p for c in rc.VERSION_CATEGORIES for p in pairs[c]}
            for c, more in version_pairs(corpus, stage, stratum, seed, corpus_sha, similar,
                                         "version pairs" + suffix).items():
                more = [p for p in more if p not in seen]
                pairs[c] += more
                if more:
                    planted_in[c].discard(stage_name)
        if not suffix:
            core_pairs = {c: len(v) for c, v in pairs.items()}

    def staged(make: Callable[[str, List[str], int], List[Tuple[str, ...]]], cap: int) -> Tuple[list, int]:
        units: List[Tuple[str, ...]] = []
        n_first = 0
        for suffix, stage in stages:
            units += make(suffix, stage, stage_left(cap, not suffix, len(units)))
            if not suffix:
                n_first = len(units)
        return units, n_first

    def records_of(cat: str, keep: Callable[[str], bool] = lambda r: True):
        return lambda suffix, stage, left: [(r,) for r in itertools.islice(
            seeded_order([r for r in stage if keep(r)], stratum, seed, corpus_sha, cat + suffix), max(left, 0))]

    def topics(suffix: str, stage: List[str], left: int) -> List[Tuple[str, ...]]:
        here = set(stage)
        out: List[Tuple[str, ...]] = []
        for anchor in seeded_order(stage, stratum, seed, corpus_sha, "abstention" + suffix):
            if len(out) >= left:
                break
            near = corpus.related(anchor, lambda c: c in here)[:TOPIC_NEIGHBORS]
            out.append((anchor,) + tuple(c for _, c in near))
        return out

    plans = {}
    for cat in rc.CATEGORIES:
        cap = caps[cat]
        units: List[Tuple[str, ...]] = []
        n_first = 0
        reason = None
        if not cap:
            pass
        elif cat == "long_horizon":
            if n_timed < LONG_HORIZON_MIN_TS:
                reason = f"needs ts on at least {LONG_HORIZON_MIN_TS} records; the corpus has {n_timed}"
            else:
                units, n_first = staged(records_of(cat, lambda r: bands[r] == AGE_BANDS[0]), cap)
        elif cat in rc.VERSION_CATEGORIES:
            if timed_reps < 2:
                reason = f"needs ts on its records; {timed_reps} eligible records carry one"
            else:
                units, n_first = list(pairs[cat]), core_pairs.get(cat, 0)
                if not units:
                    reason = (f"no two eligible records on one topic (similarity {PAIR_MIN_SIMILARITY}) with different ts"
                              if planted is None or cat == "knowledge_update" else
                              "no planted pair of its type among the eligible records (--planted-pairs)")
        elif cat == "entity_attribution":
            # Only records that hold a rare name, so the answer can pick out the record.
            if rare_names is None:
                rare_names = rare_name_records(corpus, reps)
            units, n_first = staged(records_of(cat, lambda r: r in rare_names), cap)
            if not units:
                reason = (f"no eligible record holds a rare name (a capitalized name found in at most "
                          f"{RARE_NAME_MAX_OTHERS} other record of the pool)")
        elif cat == "cross_session":
            if len(sources) < 2:
                reason = "needs source on its records, with at least two different sessions"
            else:
                units, n_first = staged(lambda suffix, stage, left: session_pairs(
                    corpus, stage, stratum, seed, corpus_sha, max(left, 0), "cross_session" + suffix), cap)
                if not units:
                    reason = f"no two eligible records on one topic (similarity {PAIR_MIN_SIMILARITY}) from different sessions"
        elif cat == "abstention":
            units, n_first = staged(topics, cap)
        else:
            units, n_first = staged(records_of(cat), cap)
        plans[cat] = Plan(cat, units, cap, reason, n_first if core is not None else None,
                          planted_in.get(cat, ()))
    return plans


# ---------------------------------------------------------------------------
# Model calls: the API, a recorded run, or a test's fake
# ---------------------------------------------------------------------------

class Reply:
    """What one call returned, as calls.jsonl records it."""

    def __init__(self, text: str, model: str, stop_reason: Optional[str], input_tokens: int = 0,
                 output_tokens: int = 0, time: Optional[str] = None, sdk: Optional[str] = None,
                 started: Optional[str] = None):
        self.text = text
        self.model = model
        self.stop_reason = stop_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.time = time or _utc_now()  # when the reply arrived
        self.started = started or self.time  # when the request went out
        self.sdk = sdk
        self.cache_write = 0  # prompt-cache tokens, when the API reports them
        self.cache_write_1h = 0  # the part of cache_write written with the 1-hour cache
        self.cache_read = 0
        self.in_log = False  # under --resume: the call is already in this folder's log


class AnthropicModel:
    """The Messages API through the official anthropic SDK, imported on the first call so
    the tests and the rest of the harness never need it. Server-side fallbacks are off, so
    every item comes from the pinned model; a refusal yields no items and is counted.

    The key: by default the SDK reads ANTHROPIC_API_KEY and nothing here reads it. With
    api_key_env (--api-key-env NAME) the client is made from that one variable, with the
    base URL pinned to API_URL; a key passed in also stops the SDK reading any other
    credential from the environment. The value is never printed, logged or written, and
    every error raised here has it taken out and drops the exception it came from. The
    pool generator's model (generate_pool.PoolAnthropicModel) is this class with a
    streamed call, so both generators handle the key with the same code."""

    def __init__(self, model_id: str, api_key_env: Optional[str] = None):
        if api_key_env is not None and not _ENV_NAME.fullmatch(api_key_env):
            raise GenerateError("--api-key-env takes the name of an environment variable, such as MY_API_KEY "
                                "(letters, digits and underscores), never the key itself")
        self.model_id = model_id
        self.api_key_env = api_key_env
        self._client = None
        self._key: Optional[str] = None

    def connect(self) -> None:
        """Loads the SDK and makes the client, with no network call. generate_set calls it
        before writing anything, so a missing package or key leaves no folder behind."""
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        if self.api_key_env is not None:
            self._connect_with_key()
            return
        try:
            import anthropic
        except ImportError:
            raise GenerateError("a live run needs the anthropic package: pip install anthropic "
                                "(the tests and --replay do not)")
        self._anthropic = anthropic
        self._sdk = f"anthropic {getattr(anthropic, '__version__', 'unknown')}"
        try:
            self._client = anthropic.Anthropic()
        except (anthropic.AnthropicError, TypeError) as exc:  # no credentials it can find, most often
            raise GenerateError(f"could not create the API client ({type(exc).__name__}: {exc}); check that "
                                "ANTHROPIC_API_KEY is set")

    def _connect_with_key(self) -> None:
        name = self.api_key_env
        key = os.environ.get(name, "").strip()
        if not key:
            raise GenerateError(f"--api-key-env: {name} is not set in this environment, or is empty")
        if any(not 33 <= ord(c) <= 126 or c in "\"'\\" for c in key):  # so no error can quote it escaped
            raise GenerateError(f"--api-key-env: {name} holds a space, a line break, a quote or another character "
                                "an API key does not have; check its value in the environment's settings")
        try:
            import anthropic
        except ImportError:
            raise GenerateError("a live run needs the anthropic package: pip install anthropic "
                                "(the tests and --replay do not)")
        self._anthropic = anthropic
        self._sdk = f"anthropic {getattr(anthropic, '__version__', 'unknown')}"
        self._key = key
        try:
            self._client = anthropic.Anthropic(api_key=key, base_url=API_URL)
        except Exception as exc:
            raise GenerateError(self._scrub(f"could not create the API client ({type(exc).__name__}: {exc})")) from None

    def _scrub(self, text: str) -> str:
        """text with the key taken out, when this model holds one."""
        return text.replace(self._key, "[the key]") if self._key else text

    def call(self, category: str, prompt: Prompt, user: str, request_sha: str) -> Reply:
        if self._client is None:
            self._connect()
        started = _utc_now()
        try:
            resp = self._client.messages.create(
                model=self.model_id,
                max_tokens=MAX_TOKENS,
                system=system_blocks(prompt),
                messages=[{"role": "user", "content": user}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": prompt.schema}},
            )
        except self._anthropic.APIError as exc:
            raise GenerateError(self._scrub(f"the {category} call failed: {type(exc).__name__}: {exc}")) from None
        except TypeError as exc:  # an SDK too old for this request, or no credentials it can find
            raise GenerateError(self._scrub(f"the anthropic SDK refused the {category} request ({exc}); upgrade "
                                            "it with pip install -U anthropic and check the API key is set")) from None
        except Exception as exc:  # anything else the SDK raises: a message, never a traceback that could hold the key
            raise GenerateError(self._scrub(f"the {category} call failed: {type(exc).__name__}: {exc}")) from None
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        usage = getattr(resp, "usage", None)
        reply = Reply(text, resp.model, resp.stop_reason, getattr(usage, "input_tokens", 0) or 0,
                      getattr(usage, "output_tokens", 0) or 0, _utc_now(), self._sdk, started)
        reply.cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        reply.cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        # The API splits its cache writes by how long they are kept; a reply without the
        # split wrote what the request asked for.
        split = getattr(getattr(usage, "cache_creation", None), "ephemeral_1h_input_tokens", None)
        reply.cache_write_1h = int(split) if split is not None else (
            reply.cache_write if (SYSTEM_CACHE or {}).get("ttl") == "1h" else 0)
        return reply

    def check(self) -> str:
        """One tiny call that proves the key and the network: the served model's id, or
        GenerateError naming only the error's class."""
        try:
            self._connect()
            resp = self._client.messages.create(model=self.model_id, max_tokens=16,
                                                messages=[{"role": "user", "content": "Reply with the word ok."}])
        except GenerateError:
            raise
        except Exception as exc:  # the class only: an API error's text can quote the request
            raise GenerateError(type(exc).__name__) from None
        return resp.model


def check_api(model_id: str = DEFAULT_MODEL, api_key_env: Optional[str] = None) -> str:
    """--check-api: one tiny call. Returns the served model's id; raises GenerateError
    with only the error's class name, never its text and never the key."""
    return AnthropicModel(model_id, api_key_env).check()


class ReplayModel:
    """Answers each request with the response a recorded generation/calls.jsonl holds for
    the same request (its SHA-256), in order, so a set rebuilds without the API (--replay).
    With a live model behind it (--resume), a request the log does not hold goes to the
    API, and a last line cut short by a stopped run is skipped (tolerate_cut, which an
    estimate on a stopped run sets too)."""

    def __init__(self, path: str, live: Any = None, tolerate_cut: Optional[bool] = None):
        self.path = path
        self.live = live
        self.recorded: Dict[str, deque] = defaultdict(deque)
        self.good_lines: List[str] = []  # the log's readable lines, as written
        self.skipped_lines = 0
        tolerate_cut = live is not None if tolerate_cut is None else tolerate_cut
        what = "--replay" if live is None and not tolerate_cut else "--resume"
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().split("\n")
        except OSError as exc:
            raise GenerateError(f"{what}: cannot read {path}: {exc}")
        filled = [i for i, line in enumerate(lines) if line.strip()]
        for i in filled:
            try:
                rec = json.loads(lines[i])
                self.recorded[rec["request_sha256"]].append(rec)
                self.good_lines.append(lines[i])
            except (ValueError, KeyError, TypeError) as exc:
                if tolerate_cut and i == filled[-1]:  # the line a stopped run was writing
                    self.skipped_lines += 1
                    continue
                raise GenerateError(f"{what}: cannot read {path}: line {i + 1}: {exc}")
        self.cost = math.fsum(record_cost(rec) or 0.0 for rec in map(json.loads, self.good_lines))

    def connect(self) -> None:
        if self.live is not None and hasattr(self.live, "connect"):
            self.live.connect()

    def holds(self, request_sha: str) -> bool:
        """Whether the next call with this request is answered from the log, for free."""
        return bool(self.recorded.get(request_sha))

    def unused(self) -> int:
        return sum(len(q) for q in self.recorded.values())

    def call(self, category: str, prompt: Prompt, user: str, request_sha: str) -> Reply:
        queue = self.recorded.get(request_sha)
        if not queue:
            if self.live is not None:
                return self.live.call(category, prompt, user, request_sha)
            raise GenerateError(f"--replay: {self.path} holds no response for this {category} request; the corpus, "
                                "seed, targets, window, model or prompts differ from the recorded run")
        rec = queue.popleft()
        reply = Reply(rec["response"], rec["served_model"], rec["stop_reason"], rec["input_tokens"],
                      rec["output_tokens"], rec["time"], rec.get("sdk"), rec.get("started"))
        reply.cache_write = rec.get("cache_creation_input_tokens", 0)
        reply.cache_write_1h = rec.get("cache_creation_1h_input_tokens", 0)
        reply.cache_read = rec.get("cache_read_input_tokens", 0)
        reply.in_log = self.live is not None
        return reply


def cost_exact(model_id: str, uncached_in: int, out: int, cache_write: int = 0, cache_read: int = 0,
               cache_write_1h: int = 0) -> Optional[float]:
    """US dollars at the prices in PRICES, unrounded; None for a model not listed.
    cache_write is written with the 5-minute cache, cache_write_1h with the 1-hour cache."""
    price = PRICES.get(model_id)
    if price is None:
        return None
    p_in, p_out, p_write, p_write_1h, p_read = price
    return (uncached_in * p_in + out * p_out + cache_write * p_write + cache_write_1h * p_write_1h
            + cache_read * p_read) / 1e6


def cost_usd(model_id: str, uncached_in: int, out: int, cache_write: int, cache_read: int) -> Optional[float]:
    """cost_exact rounded to the cent, as a manifest reports it (5-minute cache writes)."""
    cost = cost_exact(model_id, uncached_in, out, cache_write, cache_read)
    return None if cost is None else round(cost, 2)


def record_cost(rec: Dict[str, Any]) -> Optional[float]:
    """The cost of one recorded call, at the price of the model it asked for. Its cache
    writes are 1-hour ones as far as cache_creation_1h_input_tokens says, 5-minute ones for
    the rest."""
    hour = rec.get("cache_creation_1h_input_tokens", 0)
    return cost_exact(rec.get("model"), rec.get("input_tokens", 0), rec.get("output_tokens", 0),
                      rec.get("cache_creation_input_tokens", 0) - hour, rec.get("cache_read_input_tokens", 0), hour)


def system_blocks(prompt: Prompt) -> Any:
    """The request's system prompt: the prompt file's text as one block marked for the
    cache, or the bare text when SYSTEM_CACHE is None."""
    if SYSTEM_CACHE is None:
        return prompt.text
    return [{"type": "text", "text": prompt.text, "cache_control": dict(SYSTEM_CACHE)}]


def request_sha256(model_id: str, prompt: Prompt, user: str) -> str:
    blob = json.dumps({"model": model_id, "max_tokens": MAX_TOKENS, "effort": EFFORT,
                       "system_sha256": prompt.sha256, "system_cache": SYSTEM_CACHE, "user": user},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def record_json(rec: Dict[str, Any]) -> str:
    """A record as the model sees it: id, then kind, ts and source when the record has
    them, then the text."""
    out = {"id": rec["id"]}
    for field in ("kind", "ts", "source"):
        value = rec.get(field)
        if value not in (None, "") and not isinstance(value, (bool, dict, list)):
            out[field] = value
    out["text"] = rec["text"]
    return json.dumps(out, ensure_ascii=False)


def user_message(category: str, batch: Sequence[Tuple[str, ...]], corpus: Corpus) -> str:
    """The user message for one call: one record per line, or labeled pairs."""
    lines = []
    if UNIT_KIND[category] == "pair":
        first, second = PAIR_LABELS[category]
        for i, (a, b) in enumerate(batch, 1):
            lines += [f"PAIR {i}", f"{first}: {record_json(corpus.by_id[a])}",
                      f"{second}: {record_json(corpus.by_id[b])}"]
    else:
        for unit in batch:
            lines += [record_json(corpus.by_id[rid]) for rid in unit]
    return "\n".join(lines)


RETRY_HEAD = "RETRY"


def leaks(item: Dict[str, Any], gold: Sequence[str], corpus: Corpus) -> List[Tuple[str, str, int]]:
    """(field, value, n) for every key value that n records outside gold also contain:
    what the lint's GOLD_MULTIPLICITY rule holds an item for."""
    out = []
    gold = set(gold)
    for field, values in (("answer", [item["answer"]]), ("acceptable_answers", item["acceptable_answers"])):
        for value in values:
            hits = {corpus.ids[i] for i in corpus.ctx.index.matching(rc.normalize(value))}
            n = len(hits - gold)
            if n:
                out.append((field, value, n))
    return out


def retry_message(category: str, item: Dict[str, Any], unit: Tuple[str, ...], corpus: Corpus,
                  conflicts: Sequence[Tuple[str, str, int]]) -> str:
    """The user message of a GOLD_MULTIPLICITY retry (docs/generation.md section 6): the
    held item as the model returned it, the words that leak and how many other records
    hold them, then the unit exactly as it was first shown. No other record is shown."""
    why = "; ".join(f"the {'answer' if field == 'answer' else 'acceptable answer'} {json.dumps(value, ensure_ascii=False)} "
                    f"also appears in {n} other record{'s' if n != 1 else ''} of the store"
                    for field, value, n in conflicts)
    return "\n".join([RETRY_HEAD, "held: " + json.dumps(item, ensure_ascii=False), "why: " + why,
                      user_message(category, [unit], corpus)])


def parse_retry_message(text: str) -> Tuple[Dict[str, Any], str, List[Any]]:
    """The held item, the reason and the unit of a retry message, for tests and audits."""
    lines = text.split("\n")
    return (json.loads(lines[1].split(": ", 1)[1]), lines[2].split(": ", 1)[1],
            parse_user_message("\n".join(lines[3:])))


def parse_user_message(text: str) -> List[Any]:
    """The inverse of user_message, for tests and audits: a list of records, or of
    (first, second) record pairs for a pair message. A retry message gives the unit it
    shows again."""
    lines = text.split("\n")
    if lines and lines[0] == RETRY_HEAD:
        lines = lines[3:]
    if lines and lines[0].startswith("PAIR "):
        out = []
        for i in range(0, len(lines), 3):
            out.append(tuple(json.loads(line.split(": ", 1)[1]) for line in lines[i + 1:i + 3]))
        return out
    return [json.loads(line) for line in lines if line.strip()]


def parse_reply(reply: Reply) -> Tuple[List[Any], Optional[str]]:
    if reply.stop_reason == "refusal":
        return [], "the model declined the request"
    if reply.stop_reason == "max_tokens":
        return [], "the reply hit max_tokens"
    try:
        data = json.loads(reply.text)
    except ValueError:
        return [], "the reply is not JSON"
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return [], "the reply has no items list"
    return data["items"], None


# ---------------------------------------------------------------------------
# Candidates: the checks, the abstention screen, the fields the model does not return
# ---------------------------------------------------------------------------

def check_item(category: str, item: Any, prompt: Prompt, batch: Sequence[Tuple[str, ...]]
               ) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    """The mechanical checks before the lint (docs/generation.md section 6). Returns the
    unit the item belongs to, or the reason it is dropped."""
    if not isinstance(item, dict) or set(item) != prompt.fields:
        return None, "the item's fields do not match the schema"
    if not isinstance(item["question"], str) or not item["question"].strip():
        return None, "empty question"
    if not isinstance(item["answer"], str) or not item["answer"].strip():
        return None, "empty answer"
    alts, gold = item["acceptable_answers"], item["gold_ids"]
    if not isinstance(alts, list) or not all(isinstance(a, str) for a in alts):
        return None, "acceptable_answers is not a list of strings"
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold) or len(set(gold)) != len(gold):
        return None, "gold_ids is not a list of distinct strings"
    if item["review_notes"] != "":
        return None, "review_notes is not empty"
    kind = UNIT_KIND[category]
    if kind == "topic":
        if gold:
            return None, "an abstention item lists gold ids"
        return batch[0], None
    if not gold:
        return None, "no gold ids"
    shown = {rid for unit in batch for rid in unit}
    if any(g not in shown for g in gold):
        return None, "a gold id is not one of the records shown in this call"
    if kind == "record":
        return next(unit for unit in batch if unit[0] in gold), None
    unit = next((u for u in batch if set(u) == set(gold)), None)
    if unit is None:
        return None, "gold_ids is not exactly one pair's two records"
    if category in rc.VERSION_CATEGORIES and item["must_rank_first"] != unit[1]:
        return None, "must_rank_first is not the later record"
    return unit, None


QUESTION_MAX_WORDS = 30
ANSWER_WORDS = {"vocabulary_gap": (1, 6), "implicit_preference": (2, 8)}  # every other answerable category: 1 to 8
_QUESTION_MARKS = ("?", "？", "؟")  # and the full-width and Arabic question marks
# A second question joined to the first ("... and who ...?"): two questions in one item.
_SECOND_QUESTION = re.compile(r"\band\s+(who|whom|whose|what|which|when|where|why|how)\b", re.I)


def has_unit(value: str, unit_terms: Sequence[str]) -> bool:
    """Whether a key carries a unit: one of the lint's unit words or symbols (read on the
    raw lowercased text, as the lint's QUANTITY_REQUIRES_UNIT does, since normalizing
    strips $ and %), or a unit of time."""
    low = str(value).lower()
    return any(u.lower() in low for u in unit_terms) or bool(_TIME_UNIT.search(low))


def rule_checks(category: str, item: Dict[str, Any], unit: Tuple[str, ...], corpus: Corpus) -> Optional[str]:
    """The prompts' own rules that a machine can check (section 6): the question's shape
    and length, one question per item, the answer's length, a number's unit kept by every
    alternate, and in the pair categories where the answer may sit. Returns the reason an
    item is dropped, or None."""
    question = item["question"].strip()
    if not question.endswith(_QUESTION_MARKS):
        return "the question does not end in a question mark"
    if len(question.split()) > QUESTION_MAX_WORDS:
        return f"the question is longer than {QUESTION_MAX_WORDS} words"
    if sum(question.count(m) for m in _QUESTION_MARKS) > 1:
        return "two questions in one: the item holds more than one question mark"
    second = _SECOND_QUESTION.search(question)
    if second:
        return f"two questions in one: the question goes on with {second.group(0).lower()!r}"
    if category == "abstention":
        return None  # its values are screened instead
    low, high = ANSWER_WORDS.get(category, (1, 8))
    if not low <= len(rc.tokenize(item["answer"])) <= high:
        return f"the answer is not {low} to {high} words"
    if any(len(rc.tokenize(a)) > high for a in item["acceptable_answers"]):
        return f"an acceptable answer is longer than {high} words"
    units = corpus.ctx.const.get("unit_terms", [])
    if re.search(r"\d", item["answer"]) and has_unit(item["answer"], units):
        for alt in item["acceptable_answers"]:
            if re.search(r"\d", alt) and not has_unit(alt, units):
                return f"an acceptable answer drops the unit the answer has: {alt!r}"

    def holds(rid: str, value: str) -> bool:  # the record or one of its near copies
        norm = rc.normalize(value)
        return any(rc.contains_phrase(corpus.ctx.norm_by_id[m], norm) for m in corpus.cluster[rid])

    if category in rc.VERSION_CATEGORIES:
        earlier = unit[0]
        if any(holds(earlier, v) for v in [item["answer"]] + list(item["acceptable_answers"]) if v.strip()):
            return "the answer or an acceptable answer also appears in the earlier record"
    if category in rc.ALL_GOLD_CATEGORIES and all(holds(rid, item["answer"]) for rid in unit):
        return "the answer appears in both records, so one record alone answers the question"
    return None


def screen_abstention(item: Dict[str, Any], topic: Tuple[str, ...], corpus: Corpus
                      ) -> Tuple[Optional[str], Dict[str, Any]]:
    """The abstention screen, against every record of the pool (section 7). Empty values
    say nothing and are skipped; the answer itself is never empty (check_item)."""
    values = [v for v in [item["answer"]] + list(item["acceptable_answers"]) if v.strip()]
    detail: Dict[str, Any] = {"values": values, "found_in": {}, "shared_with_topic": []}
    for value in values:
        if len(rc.tokenize(value)) > PROBE_MAX_TOKENS:
            return f"a screened value is longer than {PROBE_MAX_TOKENS} words: {value!r}", detail
    for value in values:
        hits = [corpus.ids[i] for i in corpus.ctx.index.matching(rc.normalize(value))]
        if hits:
            detail["found_in"][value] = hits[:5]
    q_words = {t for t in rc.tokenize(item["question"]) if t not in rc.STOPWORDS}
    detail["shared_with_topic"] = sorted(q_words & set().union(*(corpus.content[r] for r in topic)))
    if detail["found_in"]:
        value, hits = next(iter(detail["found_in"].items()))
        return f"the screened value {value!r} appears in record {hits[0]}", detail
    if len(detail["shared_with_topic"]) < TOPIC_MIN_SHARED:
        return f"the question shares fewer than {TOPIC_MIN_SHARED} words with its topic", detail
    return None, detail


_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december"
           "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_MONTH_DATE = re.compile(rf"\b(?:{_MONTHS})\s+\d|\b\d+(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b")
_YEAR = re.compile(r"^(?:1[89]|20)\d\d$")
_NUMERIC_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}\b|^\d{1,2}/\d{1,2}/\d{2,4}$")
_TIME_OF_DAY = re.compile(r"^\d{1,2}:\d{2}\b")
# A unit of time counts only with a number right before it: "4 months" and "30-minute" are
# durations, "second transect" is not (the v11 pilot's reading, October 22, 2026).
_TIME_UNIT = re.compile(r"\d\s*-?\s*(?:second|minute|hour|day|week|fortnight|month|year|decade)s?\b", re.I)
_VERSION = re.compile(r"^v\d+(?:\.\d+)+\S*$|^\d+(?:\.\d+){2,}\S*$", re.I)
_CODE = re.compile(r"[()\[\]{}<>`]|\w_\w|::|->|^--?[A-Za-z]|^\S*[A-Za-z]\.[A-Za-z]\S*$")


def answer_type(category: str, answer: str, spec: Dict[str, Any]) -> str:
    """The answer_type of a generated item, from the answer's form (section 6). The first
    rule that fits wins, in this order: formula, version, then for an answer with a digit
    date, time of day (a phrase), duration, quantity with unit and number; then code
    symbol; else phrase."""
    if category == "abstention":
        return "abstention"
    const = spec.get("constants", {})
    raw = str(answer).strip()
    norm = rc.normalize(raw)
    if category == "entity_attribution":
        roles = {rc.normalize(v) for v in const.get("role_answer_whitelist", [])}
        return "role" if norm in roles else "entity"
    if raw.startswith("="):
        return "formula"
    if _VERSION.match(raw):  # v1.2 or 1.2.3; a plain decimal such as 3.11 is a number
        return "version"
    if any(ch.isdigit() for ch in raw):
        if _YEAR.match(norm) or _NUMERIC_DATE.match(raw) or _MONTH_DATE.search(norm):
            return "date"
        if _TIME_OF_DAY.match(raw):
            return "phrase"
        if _TIME_UNIT.search(raw):
            return "duration"
        if any(u.lower() in raw.lower() for u in const.get("unit_terms", [])):
            return "quantity_with_unit"
        if norm.replace(" ", "").isdigit():
            return "number"
    if _CODE.search(raw):
        return "code_symbol"
    return "phrase"


def assemble(category: str, item: Dict[str, Any], unit: Tuple[str, ...], corpus: Corpus, spec: Dict[str, Any],
             cand_id: str) -> Dict[str, Any]:
    """The question as it would freeze, in the lint spec's field order. Gold grows to
    each shown record's near copies, by the rule fixed before any call; difficulty is a
    placeholder until the freeze ranks it."""
    if category == "abstention":
        answer, alts, gold, current, atype = ABSTENTION_ANSWER, list(ABSTENTION_ALTERNATES), [], None, "abstention"
    else:
        gold = []
        for g in item["gold_ids"]:
            gold += [m for m in corpus.cluster[g] if m not in gold]
        current = None
        if category in rc.VERSION_CATEGORIES:
            copies = corpus.cluster[unit[1]]
            current = copies[0] if len(copies) == 1 else list(copies)
        answer, alts = item["answer"], list(item["acceptable_answers"])
        atype = answer_type(category, answer, spec)
    return {
        "id": cand_id, "question": item["question"], "category": category, "difficulty": "medium",
        "answer_type": atype, "answer": answer, "acceptable_answers": alts, "gold_ids": gold,
        "must_rank_first": current, "gold_hashes": {g: rc.text_hash(corpus.by_id[g]["text"]) for g in gold},
        "review_status": "publish", "review_notes": "",
    }


def difficulties(questions: List[Dict[str, Any]], records: List[Dict[str, Any]], window: int) -> List[str]:
    """Where the reference BM25 ranks each question's gold over the whole pool at the
    band's window: first is easy, inside the window medium, outside hard. Abstention
    items are medium."""
    bm25 = BM25Adapter()
    bm25.setup("", "file")
    for rec in records:
        bm25.ingest(rec)
    bm25.save()
    out = []
    for q in questions:
        gold = set(q["gold_ids"])
        if not gold:
            out.append("medium")
            continue
        ranks = [i for i, hit in enumerate(bm25.search(q["question"], window)) if hit[0] in gold]
        out.append("easy" if ranks and ranks[0] == 0 else "medium" if ranks else "hard")
    return out


# ---------------------------------------------------------------------------
# The oracle check and the reference runs
# ---------------------------------------------------------------------------

def _run_reference(q_path: str, c_path: str, store_ids: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prev = os.environ.get("RECALL_QFILE")
    os.environ["RECALL_QFILE"] = q_path  # the oracle reads the question file, as no real adapter may
    runs, results = {}, {}
    try:
        for name in ("oracle", "bm25"):
            res = rb.run_benchmark(lambda n=name: get_adapter(n), name, q_path, c_path, verify=True, quiet=True,
                                   store_ids=store_ids)
            h = res["headline"]
            runs[name] = {
                "result_line": res["result_line"], "verified": res["verification"]["status"],
                "records_hash": res["manifest"]["records_hash"], "gold": h["recall"], "skill": h["skill"],
                "chance": h["chance"], "mrr": h["mrr"], "version_pct": h["version_pct"],
                "exact": res["diagnostics"]["exact"]["recall"], "family": res["diagnostics"]["family"]["recall"],
            }
            results[name] = res
    finally:
        if prev is None:
            os.environ.pop("RECALL_QFILE", None)
        else:
            os.environ["RECALL_QFILE"] = prev
    return runs, results


def oracle_check(q_path: str, c_path: str, store_ids: bool) -> Dict[str, Any]:
    """The oracle must score 100 on the gold tier at the band's window (section 10).
    Returns whether it did, every question it missed and why, and the reference runs."""
    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)
    pool_ids = {r["id"] for r in rb.load_corpus(c_path, store_ids=store_ids)}
    texts = Counter(q.get("question") for q in questions)
    missed = []
    for q in questions:
        absent = [g for g in q.get("gold_ids") or [] if g not in pool_ids]
        if absent:
            missed.append({"id": q.get("id"), "reason": f"gold ids not in the pool: {', '.join(absent)}"})
        elif q.get("gold_ids") and texts[q.get("question")] > 1:
            missed.append({"id": q.get("id"), "reason": "another question has the same text, and the oracle "
                                                        "looks questions up by text"})
    if missed:  # the runner would refuse the set, so there is nothing to run
        return {"ok": False, "missed": missed, "gold_at_window": None, "verified": None, "reference": None}
    runs, results = _run_reference(q_path, c_path, store_ids)
    pq = [r for r in results["oracle"]["per_question"] if r["scored_on"] != "abstention"]
    for r in pq:
        if r["headline_rank"] is None:
            missed.append({"id": r["id"], "reason": "not every gold record is in the oracle's window"
                           if r["scored_on"] == "gold_all" else "no gold record in the oracle's window"})
    pct = round(100.0 * (len(pq) - len(missed)) / len(pq), 2) if pq else None
    return {"ok": not missed, "missed": missed, "gold_at_window": pct, "verified": runs["oracle"]["verified"],
            "reference": runs}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

_SET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # used with fullmatch, so no newline slips through
SEALED_DIR = os.path.join(REPO_DIR, "sealed")


class CostStop(Exception):
    """--max-cost was passed: the run stops before its next new API call, with every call
    recorded, and a --resume continues it. Exit 3."""


def check_out(out: str, release: bool, resume: bool = False, sealed: bool = False) -> Tuple[str, str]:
    """Where a set may be written (section 11). Returns the resolved folder and the set's
    name, which is the folder's last path component. Inside the repository a set may sit
    only under data/ (a public set) or, for the sealed half of a split set, under sealed/,
    and only with --release. The folder must be new or empty, except the one a --resume
    picks up from (checked by the caller)."""
    flag = "--sealed-out" if sealed else "--out"
    real = os.path.realpath(out)
    name = os.path.basename(real)
    if not _SET_NAME.fullmatch(name):
        raise GenerateError(f"{flag} {out}: the folder's name becomes the set's version name, so it takes letters, "
                            "digits, dots, dashes and underscores")
    root, what = (SEALED_DIR, "the sealed half of a set under sealed/") if sealed else (DATA_DIR, "a public set under data/")
    if rb.inside_repo(real):
        if not (release and real.startswith(os.path.realpath(root) + os.sep)):
            raise GenerateError(f"{flag} {out} is inside the repository. A question set holds the facts it was written "
                                f"from, so it is written outside; the one exception is {what}, with --release")
    elif release:
        raise GenerateError(f"--release is for {what} in this repository; this {flag} is outside it")
    if not resume and os.path.exists(real) and (not os.path.isdir(real) or os.listdir(real)):
        raise GenerateError(f"{flag} {out} is not an empty folder; a frozen set is never overwritten")
    return real, name


def parse_targets(text: str) -> Dict[str, int]:
    """One number for every category, or category=number pairs (a category left out gets 0)."""
    text = str(text).strip()
    if re.fullmatch(r"\d+", text):
        return {c: int(text) for c in rc.CATEGORIES}
    targets = dict.fromkeys(rc.CATEGORIES, 0)
    for part in text.split(","):
        cat, sep, num = part.strip().partition("=")
        if not sep or cat.strip() not in targets or not re.fullmatch(r"\d+", num.strip()):
            raise GenerateError(f"--targets: cannot read {part.strip()!r}; give one number for every category, "
                                "or category=number pairs such as direct_recall_user=5,abstention=3")
        targets[cat.strip()] = int(num)
    return targets


class _Category:
    """One category's progress through a run."""

    def __init__(self, plan: Plan, target: int, core_quota: Optional[int] = None):
        self.plan = plan
        self.target = target
        self.core_quota = core_quota  # at most this many items from the core freeze (--core-quota)
        self.pos = 0
        self.counts: Counter = Counter()
        self.stopped = (plan.reason or (None if plan.units else "no eligible records")) if target else None
        self.done = not target or not plan.units
        self.allowance = 0  # new records this category may still show this round

    def core_quota_met(self) -> bool:
        return self.core_quota is not None and self.counts["frozen_core"] >= self.core_quota

    def settle(self, cand: Dict[str, Any], fate: str, reason: Optional[str] = None) -> None:
        """A candidate's fate, counted the moment it is decided, so a batch never freezes
        more than the category still needs."""
        cand["fate"], cand["reason"] = fate, reason
        self.counts[fate] += 1

    def take(self, shown: set, sample_cap: int) -> List[Tuple[str, ...]]:
        """The next batch, within the unit cap, the sample cap and this round's share of
        it. A record another category already showed costs nothing. When only this
        round's share stops the next unit, the category waits for a later round instead
        of ending."""
        batch: List[Tuple[str, ...]] = []
        stop = None
        while len(batch) < BATCH[self.plan.kind]:
            if self.plan.n_core is not None and self.pos < self.plan.n_core and self.core_quota_met():
                self.pos = self.plan.n_core  # the core gave its quota, so the core stage ends here
            if self.counts["units_shown"] >= self.plan.cap:
                stop = "unit cap reached"
                break
            if self.pos >= len(self.plan.units):
                stop = "ran out of units"
                break
            unit = self.plan.units[self.pos]
            new = set(unit) - shown
            if len(shown) + len(new) > sample_cap:
                stop = "sample cap reached"
                break
            if len(new) > self.allowance:
                break  # this round's share is spent
            batch.append(unit)
            shown |= new
            self.allowance -= len(new)
            if self.plan.n_core is not None and self.pos < self.plan.n_core:
                self.counts["units_shown_core"] += 1
            self.pos += 1
            self.counts["units_shown"] += 1
        if stop:
            self.stopped = stop
            if not batch:
                self.done = True
        return batch


def check_records(records: List[Dict[str, Any]]) -> None:
    """What the runner's loader does not check but generation needs, before any call: a
    text that is a non-empty string, and fields that are valid Unicode (a lone surrogate,
    often half of a cut emoji, cannot be sent)."""
    for n, rec in enumerate(records, 1):
        if not isinstance(rec["text"], str) or not rec["text"].strip():
            raise GenerateError(f"record {n} ({rec['id']!r}): text must be a non-empty string")
        for field in ("id", "text", "kind", "ts", "source"):
            value = rec.get(field)
            if isinstance(value, str):
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    raise GenerateError(f"record {n} ({rec['id']!r}): {field} is not valid Unicode (a lone "
                                        "surrogate, often half of a cut emoji)")


def padded_retry(item: Dict[str, Any], held: Dict[str, Any]) -> bool:
    """Whether a retried item's answer holds the held item's answer as a strict sub-span:
    the model lengthened the held answer with the words around it ("$12.95" became "$12.95
    each, which I just confirmed") instead of picking another fact, so the item is dropped
    (docs/generation.md section 6)."""
    new, old = rc.normalize(item.get("answer")), rc.normalize(held.get("answer"))
    return bool(old) and new != old and rc.contains_phrase(new, old)


def retryable(findings: List[Dict[str, Any]]) -> bool:
    """Whether a held item gets its one retry: every error the lint found is a
    GOLD_MULTIPLICITY leak (the answer or an alternate in a record outside gold), so the
    retry has one job, which the message names."""
    errors = [f for f in findings if f["severity"] == "error"]
    return bool(errors) and all(f["rule_id"] == "GOLD_MULTIPLICITY" and f["code"] == "answer_in_other_record"
                                for f in errors)


def split_halves(frozen: List[Dict[str, Any]], split_seed: int) -> Dict[int, str]:
    """Each frozen candidate's half, public or sealed (docs/generation.md section 9). The
    strata are (category, core or tail); within each, the candidates in generation order
    are shuffled by a generator seeded with the split seed and the stratum, and the first
    half is public. A stratum with an odd count gives its extra item to the public and the
    sealed half in turn, so the halves differ by at most one item overall."""
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for c in frozen:
        strata[(c["category"], c["stage"] or "all")].append(c["n"])
    side: Dict[int, str] = {}
    extra_public = True
    for key in sorted(strata, key=lambda k: (rc.CATEGORIES.index(k[0]), k[1])):
        ns = sorted(strata[key])
        random.Random(f"recall/1.0 split|{split_seed}|{key[0]}|{key[1]}").shuffle(ns)
        public = len(ns) // 2
        if len(ns) % 2:
            public += 1 if extra_public else 0
            extra_public = not extra_public
        for i, n in enumerate(ns):
            side[n] = "public" if i < public else "sealed"
    return side


def _settings(**k: Any) -> Dict[str, Any]:
    """What generation/run.json records and a --resume checks: everything that shapes the
    requests and the freeze. --max-cost and --api-key-env are not in it, so a resume may
    raise the one and name another variable for the other."""
    return {"set": k["qset_version"], "sealed_set": k["sealed_name"], "corpus_sha256": k["corpus_sha"],
            "seed": k["seed"], "targets": {c: k["targets"].get(c, 0) for c in rc.CATEGORIES}, "model": k["model_id"],
            "store_ids": k["store_ids"], "gold_window": k["window"], "core": k["core"], "core_quota": k["core_quota"],
            "unit_caps": dict(sorted(k["unit_caps"].items())), "planted_pairs": k["planted_sha"],
            "split_seed": k["split_seed"], "lint_spec": k["spec"].get("spec_version"),
            "prompts": {c: p.sha256 for c, p in k["prompts"].items()}}


def generate_set(corpus_path: str, out: str, seed: int, targets: Dict[str, int], model_id: str = DEFAULT_MODEL,
                 store_ids: bool = False, release: bool = False, model: Any = None, replay: Optional[str] = None,
                 quiet: bool = False, resume: bool = False, max_cost: Optional[float] = None,
                 gold_window: Optional[int] = None, core: Optional[int] = None, split_seed: Optional[int] = None,
                 sealed_out: Optional[str] = None, api_key_env: Optional[str] = None, estimate_only: bool = False,
                 estimate_from: Optional[Any] = None, core_quota: Optional[int] = None,
                 planted_pairs: Optional[str] = None, unit_caps: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Generate, lint and freeze one question set into out (docs/generation.md). model is
    anything with call(category, prompt, user, request_sha) -> Reply; by default the API
    (with the key from api_key_env when given), the recorded responses when replay names a
    calls.jsonl, or both under resume. max_cost stops the run before its next new call once
    the set's cost passes it (CostStop). gold_window limits gold to the corpus's first
    records, core puts the corpus's first core records first (core_quota: at most that
    many items per category from the core, then the rest of the window), and split_seed with
    sealed_out splits the frozen items into a public half (out) and a sealed half
    (sealed_out), which keeps the generation record. planted_pairs names a pool's record of
    its planted changes, which the version categories build their units from; unit_caps
    replaces categories' unit caps. estimate_only prints the estimate and returns before
    any call; estimate_from is a finished set's folder, or a list of them, that measured
    its yields and tokens."""
    say = (lambda *a, **k: None) if quiet else print
    split = split_seed is not None
    if split != (sealed_out is not None):
        raise GenerateError("--split-seed and --sealed-out go together: the split needs a folder for its sealed half")
    if replay and resume:
        raise GenerateError("--replay builds a new set from a recorded run and --resume continues a stopped one; "
                            "give one of them")
    if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
        raise GenerateError("--max-cost takes a number of US dollars, 0 or more")
    if max_cost is not None and model_id not in PRICES:
        raise GenerateError(f"--max-cost: no price on file for {model_id} (PRICES), so the guard cannot count its cost")
    out_dir, qset_version = check_out(out, release, resume=resume and not split)
    sealed_dir = sealed_name = None
    if split:
        sealed_dir, sealed_name = check_out(sealed_out, release, resume=resume, sealed=True)
        if sealed_dir == out_dir or sealed_dir.startswith(out_dir + os.sep) or out_dir.startswith(sealed_dir + os.sep):
            raise GenerateError("--out and --sealed-out must be two separate folders, neither inside the other")
        if sealed_name == qset_version:
            raise GenerateError("--sealed-out: the sealed half needs a name of its own (its folder's name), "
                                f"not {qset_version}")
    work_dir = sealed_dir if split else out_dir  # where the call log lives while the run goes on
    calls_path = os.path.join(work_dir, "generation", "calls.jsonl")
    run_path = os.path.join(work_dir, "generation", "run.json")
    if resume:
        if os.path.exists(os.path.join(work_dir, "manifest.json")):
            raise GenerateError(f"--resume: {work_dir} holds a frozen set, which is never written again")
        if not (os.path.isfile(calls_path) and os.path.isfile(run_path)):
            raise GenerateError(f"--resume: {work_dir} holds no stopped run (generation/calls.jsonl and run.json)")
    if not any(targets.get(c, 0) for c in rc.CATEGORIES):
        raise GenerateError("--targets asks for no items")
    try:
        records = rb.load_corpus(corpus_path, store_ids=store_ids)
    except (rb.InputError, OSError, ValueError) as exc:
        raise GenerateError(str(exc))
    check_records(records)
    if gold_window is not None and not 1 <= gold_window <= len(records):
        raise GenerateError(f"--gold-window {gold_window}: the corpus holds {len(records)} records")
    if core is not None and not 1 <= core <= (gold_window or len(records)):
        raise GenerateError(f"--core {core}: the core must sit inside the gold window "
                            f"({gold_window or len(records)} records)")
    if core_quota is not None and (core is None or core_quota < 0):
        raise GenerateError("--core-quota takes a number of items, 0 or more, and goes with --core")
    unit_caps = dict(unit_caps or {})
    bad_caps = [c for c, n in unit_caps.items()
                if c not in rc.CATEGORIES or isinstance(n, bool) or not isinstance(n, int) or n < 1]
    if bad_caps:
        raise GenerateError(f"--unit-cap: {', '.join(map(str, bad_caps))}: each cap is a category and a number, 1 or more")
    planted_rows = read_planted_pairs(planted_pairs) if planted_pairs is not None else None
    planted_sha = rc.file_sha256(planted_pairs) if planted_pairs is not None else None
    with open(lv.DEFAULT_SPEC, encoding="utf-8") as f:
        spec = json.load(f)
    readme = None
    if os.path.exists(README):
        with open(README, encoding="utf-8") as f:
            readme = f.read()
    used = [c for c in rc.CATEGORIES if targets.get(c, 0) > 0]
    prompts = load_prompts(used)

    corpus_sha = rc.file_sha256(corpus_path)
    corpus = Corpus(records, spec, store_ids)
    window = corpus.window_size(gold_window)
    reps, why = corpus.eligibility(int(spec.get("constants", {}).get("min_gold_tokens", 6)), gold_window)
    eligible_records = sum(len(corpus.cluster[r]) for r in reps)
    sample_cap = int(MAX_SAMPLE_SHARE * len(reps))  # distinct memories: a near copy is never shown on its own
    band_window = rc.band_for(corpus.N)["window"]
    bands = corpus.age_bands(gold_window)
    planted = planted_info = None
    if planted_rows is not None:
        planted, planted_info = route_planted_pairs(planted_rows, corpus, reps, gold_window)
        planted_info = {"file": os.path.basename(planted_pairs), "sha256": planted_sha, **planted_info}
    rare = rare_name_records(corpus, reps) if targets.get("entity_attribution", 0) > 0 else None
    plans = build_plans(corpus, reps, bands, targets, seed, corpus_sha, core, core_quota, unit_caps, planted, rare)
    states = {c: _Category(plans[c], targets.get(c, 0), core_quota) for c in rc.CATEGORIES}
    settings = _settings(qset_version=qset_version, sealed_name=sealed_name, corpus_sha=corpus_sha, seed=seed,
                         targets=targets, model_id=model_id, store_ids=store_ids, window=window, core=core,
                         core_quota=core_quota, split_seed=split_seed, spec=spec, prompts=prompts,
                         unit_caps=unit_caps, planted_sha=planted_sha)
    say(f"Corpus {os.path.basename(corpus_path)}: {corpus.N} records, {len(reps)} eligible memories "
        f"({eligible_records} records with their near copies)"
        + (f" in the gold window of {window}" if window < corpus.N else "")
        + (f", core first ({core} records)" if core is not None else "")
        + f"; at most {sample_cap} will be shown to {model_id}")
    if planted_info is not None:
        say(f"Planted pairs {planted_info['file']}: "
            + ", ".join(f"{n} for {c}" for c, n in planted_info["used"].items())
            + (f"; {sum(planted_info['skipped'].values())} left out ("
               + "; ".join(f"{n} with {why}" for why, n in planted_info["skipped"].items()) + ")"
               if planted_info["skipped"] else ""))
    if rare is not None:
        say(f"entity_attribution draws from the {len(rare)} eligible memories that hold a rare name (one found in "
            f"at most {RARE_NAME_MAX_OTHERS} other record of the pool)")
    if resume:
        with open(run_path, encoding="utf-8") as f:
            started_with = json.load(f)
        differ = [k for k in settings if started_with.get(k) != settings[k]]
        if differ:
            raise GenerateError(f"--resume: {work_dir} was started with other settings ({', '.join(differ)}); run "
                                "the command it was started with, plus --resume")

    if estimate_only:
        recorded = ReplayModel(calls_path, tolerate_cut=True) if resume else None
        folders = [estimate_from] if isinstance(estimate_from, str) else list(estimate_from or [])
        calibration = read_calibrations(folders) if folders else None
        est = estimate(plans, targets, corpus, prompts, model_id, recorded, calibration, core_quota, sample_cap)
        print_estimate(est, say)
        return {"estimate": est, "out": out_dir, "qset_version": qset_version}

    if model is None:
        if replay:
            model = ReplayModel(replay)
        else:
            model = AnthropicModel(model_id, api_key_env)
    if resume:
        model = ReplayModel(calls_path, live=model)
        if model.skipped_lines:  # the line a stopped run was cut off writing, taken out before any new call
            _write_lines(calls_path, model.good_lines)
        say(f"Resuming: the log holds {len(model.good_lines)} calls (${model.cost:.2f}), replayed free")
    if hasattr(model, "connect"):
        model.connect()

    out_ready = resume  # nothing is written until the first reply is in hand
    shown: set = set()
    calls: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    sample: Dict[str, List[Dict[str, Any]]] = {c: [] for c in used}
    frozen_questions: set = set()
    fact_keys: Dict[Any, int] = {}
    spent = [0.0, 0.0]  # the set's cost so far, and the part of it this run paid for

    def stage_of(ids: Sequence[str]) -> Optional[str]:
        if core is None:
            return None
        return "core" if all(corpus.pos[r] < core for r in ids) else "tail"

    def do_call(cat: str, prompt: Prompt, user: str, units: Sequence[Tuple[str, ...]], retry_of: Optional[int]):
        nonlocal out_ready
        req = request_sha256(model_id, prompt, user)
        free = hasattr(model, "holds") and model.holds(req)
        if max_cost is not None and spent[0] > max_cost and not free:
            new = sum(1 for c in calls if c["new"])
            raise CostStop(f"--max-cost ${max_cost:.2f} passed: the set has cost ${spent[0]:.2f} over {len(calls)} "
                           f"calls (${spent[1]:.2f} on {new} new calls in this run), every one recorded in "
                           f"{work_dir}. To go on, run the same command with --resume and a higher --max-cost; the "
                           "calls it holds replay free")
        try:
            reply = model.call(cat, prompt, user, req)
        except GenerateError as exc:
            if not out_ready:
                raise
            raise GenerateError(f"{exc}. {work_dir} holds the calls made so far and is not a question set; run the "
                                "same command with --resume to go on from them (they replay free), or remove it")
        if not out_ready:
            os.makedirs(os.path.join(work_dir, "generation", "prompts"), exist_ok=True)
            for c in used:
                with open(os.path.join(work_dir, "generation", "prompts", f"{c}.md"), "wb") as f:
                    f.write(prompts[c].data)
            _dump(run_path, settings)
            open(calls_path, "w", encoding="utf-8", newline="\n").close()
            out_ready = True
        call = {"n": len(calls) + 1, "category": cat, "retry_of": retry_of, "units": [list(u) for u in units],
                "request_sha256": req, "model": model_id, "served_model": reply.model, "stop_reason": reply.stop_reason,
                "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                "cache_creation_input_tokens": reply.cache_write,
                "cache_creation_1h_input_tokens": reply.cache_write_1h, "cache_read_input_tokens": reply.cache_read,
                "started": reply.started, "time": reply.time, "sdk": reply.sdk, "user": user, "response": reply.text}
        line = json.dumps(call, ensure_ascii=False)
        if not reply.in_log:
            with open(calls_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        cost = record_cost(call) or 0.0
        spent[0] += cost
        if not reply.in_log:
            spent[1] += cost
        calls.append(dict(call, new=not reply.in_log, line=line))
        entry = {"call": call["n"], "units": [list(u) for u in units]}
        if retry_of is not None:
            entry["retry_of"] = retry_of
        sample[cat].append(entry)
        return call, reply

    def process(cat: str, st: _Category, prompt: Prompt, items: List[Any], batch: Sequence[Tuple[str, ...]],
                call_n: int, retry_of: Optional[int], held: Optional[Dict[str, Any]] = None
                ) -> Tuple[Counter, List[Tuple]]:
        """Checks, screens and lints one reply's items, in order, and settles each one.
        Returns the items per unit and the held items that get a retry. held is the item a
        retry was for, so a retried answer that only lengthens it is dropped."""
        per_unit: Counter = Counter()
        retries = []
        for item in items:
            cand = {"n": len(candidates) + 1, "category": cat, "call": call_n, "retry_of": retry_of, "unit": None,
                    "stage": None, "returned": item, "fate": None, "reason": None, "lint_status": None, "findings": [],
                    "screen": None, "question_id": None, "half": None}
            candidates.append(cand)
            st.counts["candidates"] += 1
            unit, reason = check_item(cat, item, prompt, batch)
            if unit is not None:
                cand["unit"] = list(unit)
                cand["stage"] = stage_of(unit)
                per_unit[unit] += 1
                if per_unit[unit] > ITEMS_PER_UNIT[st.plan.kind]:
                    reason = f"more than {ITEMS_PER_UNIT[st.plan.kind]} item(s) for one {st.plan.kind}"
                elif rc.normalize(item["question"]) in frozen_questions:  # the oracle looks questions up by text
                    reason = "the same question text as a frozen item"
                elif held is not None and padded_retry(item, held):
                    reason = f"{PADDED_RETRY}: the answer only adds words around the held answer"
                else:
                    reason = rule_checks(cat, item, unit, corpus)
            if reason:
                st.settle(cand, "dropped_by_checks", reason)
                if reason.startswith(PADDED_RETRY):
                    st.counts["padded_retries"] += 1
                continue
            if cat == "abstention":
                reason, cand["screen"] = screen_abstention(item, unit, corpus)
                if reason:
                    st.settle(cand, "dropped_by_screen", reason)
                    continue
            q = assemble(cat, item, unit, corpus, spec, f"cand_{cand['n']:05d}")
            if q["gold_ids"]:
                cand["stage"] = stage_of(q["gold_ids"])
            if cat in rc.ALL_GOLD_CATEGORIES and len(q["gold_ids"]) > band_window:
                st.settle(cand, "dropped_by_checks", f"its {len(q['gold_ids'])} gold records, near copies "
                                                     f"included, cannot all fit the window of {band_window}")
                continue
            res = lv.lint_item(q, cand["n"], corpus.ctx)
            status = res["proposed_review_status"]
            cand["lint_status"] = status
            cand["findings"] = [{k: f[k] for k in ("rule_id", "severity", "code", "message")}
                                for f in res["findings"]]
            key = (tuple(sorted(q["gold_ids"])), rc.normalize(q["answer"])) if q["gold_ids"] else None
            if status == "drop":
                st.settle(cand, "dropped_by_lint", "the lint proposes drop")
            elif not res["publishable"]:
                st.settle(cand, "held_by_lint", f"the lint proposes {status}" if status != "publish"
                          else "the lint found an error")
                if retry_of is None and retryable(cand["findings"]):
                    conflicts = leaks(item, q["gold_ids"], corpus)
                    if conflicts:
                        retries.append((cand, unit, item, conflicts))
            elif key in fact_keys:  # the lint's duplicate_fact rule, against what will ship
                cand["lint_status"] = "reclassify"
                st.settle(cand, "held_by_lint", f"duplicate_fact: same gold and answer as candidate "
                                                f"{fact_keys[key]}")
            else:
                st.counts["publishable"] += 1
                if cand["stage"] == "core" and st.core_quota_met():
                    st.settle(cand, "surplus", "the core quota is met")  # publishable, but the core gave its share
                elif st.counts["frozen"] < st.target:
                    cand["question"] = q
                    frozen_questions.add(rc.normalize(item["question"]))
                    if key:
                        fact_keys[key] = cand["n"]
                    st.settle(cand, "frozen")
                    if cand["stage"] == "core":
                        st.counts["frozen_core"] += 1
                    if retry_of is not None:
                        st.counts["frozen_after_retry"] += 1
                else:
                    st.settle(cand, "surplus")
        return per_unit, retries

    last_chance = False
    while True:  # one batch per unfinished category per round
        unfinished = [c for c in rc.CATEGORIES if not states[c].done]
        if not unfinished:
            break
        remaining = sample_cap - len(shown)
        for i, c in enumerate(unfinished):
            # What is left of the cap, split evenly so it cuts every category alike; after a
            # round in which no category could move, the whole of it, first come first served.
            states[c].allowance = remaining if last_chance else (
                remaining // len(unfinished) + (1 if i < remaining % len(unfinished) else 0))
        progressed = False
        for cat in rc.CATEGORIES:
            st = states[cat]
            if st.done:
                continue
            batch = st.take(shown, sample_cap)
            if not batch:
                continue
            progressed = True
            prompt = prompts[cat]
            call, reply = do_call(cat, prompt, user_message(cat, batch, corpus), batch, None)
            cache_read = call["cache_read_input_tokens"]
            st.counts["calls"] += 1
            items, problem = parse_reply(reply)
            if problem:
                st.counts["unusable_calls"] += 1
                say(f"  call {call['n']:>4} {cat}: {problem}")
                continue
            per_unit, retries = process(cat, st, prompt, items, batch, call["n"], None)
            st.counts["skipped"] += sum(1 for u in batch if not per_unit[u])
            for cand, unit, item, conflicts in retries:
                if st.counts["frozen"] >= st.target:
                    break  # the category has what it needs, so a retry would buy nothing
                if cand["stage"] == "core" and st.core_quota_met():
                    continue  # a core item could not freeze past the core quota
                retry, reply = do_call(cat, prompt, retry_message(cat, item, unit, corpus, conflicts), [unit],
                                       cand["n"])
                st.counts["calls"] += 1
                st.counts["retry_calls"] += 1
                again, problem = parse_reply(reply)
                if problem:
                    st.counts["unusable_calls"] += 1
                    continue
                process(cat, st, prompt, again, [unit], retry["n"], cand["n"], item)
            say(f"  call {call['n']:>4} {cat}: {len(batch)} {st.plan.kind}(s), {len(items)} item(s), "
                f"{st.counts['frozen']}/{st.target} frozen"
                + (f" ({st.counts['frozen_core']} from the core)" if core is not None else "")
                + f"; cache read {cache_read:,}; ${spent[0]:.2f} so far")
            if st.counts["frozen"] >= st.target:
                st.done, st.stopped = True, "target met"
        if progressed:
            last_chance = False
        elif last_chance:
            for c in unfinished:
                if not states[c].done:
                    states[c].done, states[c].stopped = True, "sample cap reached"
            break
        else:
            last_chance = True

    unused = model.unused() if resume else 0
    if unused:
        say(f"  note: {unused} recorded call(s) were not asked for again; the generator or its inputs changed since "
            "the run stopped, so the final log leaves them out")
    sample_path = os.path.join(work_dir, "generation", "sample.json")
    cands_path = os.path.join(work_dir, "generation", "candidates.jsonl")
    if out_ready:  # the final log holds exactly the calls the set was written from, in order
        _write_lines(calls_path, [c["line"] for c in calls])
    calls = [{k: v for k, v in c.items() if k not in ("new", "line")} for c in calls]

    # The freeze: category order, then generation order.
    frozen = sorted((c for c in candidates if c["fate"] == "frozen"),
                    key=lambda c: (rc.CATEGORIES.index(c["category"]), c["n"]))
    if not frozen:
        stops = "; ".join(f"{c}: {states[c].stopped}" for c in used)
        if not out_ready:
            raise GenerateError(f"nothing was sent to the model, so there is no set ({stops})")
        _dump(sample_path, {"seed": seed, "categories": sample})
        _write_jsonl(cands_path, [_public(c) for c in candidates])
        raise GenerateError(f"no item survived the checks and the lint, so there is no set to freeze; {work_dir} "
                            "holds the generation record for a look")
    levels = difficulties([c["question"] for c in frozen], records, band_window)
    for cand, level in zip(frozen, levels):
        cand["question"]["difficulty"] = level

    if split:
        side = split_halves(frozen, split_seed)
        halves = [("sealed", sealed_dir, sealed_name, [c for c in frozen if side[c["n"]] == "sealed"]),
                  ("public", out_dir, qset_version, [c for c in frozen if side[c["n"]] == "public"])]
    else:
        halves = [(None, out_dir, qset_version, frozen)]
    written = {}
    for half, folder, name, cands in halves:
        questions = []
        for i, cand in enumerate(cands, 1):
            q = dict(cand["question"])
            q["id"] = cand["question_id"] = f"q_{i:04d}"
            cand["half"] = half
            questions.append(q)
        os.makedirs(folder, exist_ok=True)
        q_path, c_path = os.path.join(folder, "questions.json"), os.path.join(folder, "corpus.jsonl")
        _dump(q_path, questions)
        shutil.copyfile(corpus_path, c_path)
        written[half] = {"folder": folder, "name": name, "questions": questions, "q_path": q_path, "c_path": c_path,
                         "cands": cands}
    for cand in candidates:
        cand.pop("question", None)
    for half, w in written.items():
        report = lv.lint(w["questions"], records, spec, readme_text=readme, qset_version=w["name"], store_ids=store_ids)
        report["questions_file"] = "questions.json"
        report["questions_sha256"] = rc.file_sha256(w["q_path"])
        report["corpus_file"] = "corpus.jsonl"
        report["corpus_sha256"] = rc.file_sha256(w["c_path"])
        w["lint_path"] = os.path.join(w["folder"], "lint_report.json")
        _dump(w["lint_path"], report)
        w["report"] = report
    if split:  # the public half carries the prompts, and never the generation record
        os.makedirs(os.path.join(out_dir, "generation", "prompts"), exist_ok=True)
        for c in used:
            with open(os.path.join(out_dir, "generation", "prompts", f"{c}.md"), "wb") as f:
                f.write(prompts[c].data)

    _dump(sample_path, {"seed": seed, "corpus_sha256": corpus_sha, "sample_size": len(shown),
                        "strata": {rid: f"{corpus.kind[rid]} / {bands[rid]}" for rid in sorted(shown)},
                        "categories": sample})
    _write_jsonl(cands_path, [_public(c) for c in candidates])

    common = dict(store_ids=store_ids, corpus=corpus, targets=targets, model_id=model_id, calls=calls,
                  prompts=prompts, seed=seed, shown=shown, reps=reps, why=why, bands=bands,
                  eligible_records=eligible_records, sample_cap=sample_cap, sample_path=sample_path,
                  states=states, calls_path=calls_path, cands_path=cands_path, run_path=run_path, spec=spec,
                  window=window, core=core, core_quota=core_quota, split_seed=split_seed, frozen=frozen,
                  unit_caps=unit_caps, planted=planted_info, rare=rare)
    for half, w in written.items():
        other = written.get({"public": "sealed", "sealed": "public"}.get(half))
        w["manifest"] = _manifest(qset_version=w["name"], q_path=w["q_path"], c_path=w["c_path"],
                                  questions=w["questions"], report=w["report"], lint_path=w["lint_path"],
                                  cands=w["cands"], half=half, other=other, **common)
        _dump(os.path.join(w["folder"], "manifest.json"), w["manifest"])

    checks = {}
    for half, w in written.items():
        check = oracle_check(w["q_path"], w["c_path"], store_ids)
        if check["reference"] is not None:
            _dump(os.path.join(w["folder"], "reference.json"), {"qset_version": w["name"], "runs": check["reference"]})
        if not check["ok"]:
            w["manifest"]["status"], w["manifest"]["standard_eligible"] = "broken", False
            w["manifest"]["status_reason"] = "broken: the oracle did not score 100 on the gold tier; missed " + ", ".join(
                f"{m['id']} ({m['reason']})" for m in check["missed"])
            _dump(os.path.join(w["folder"], "manifest.json"), w["manifest"])
        checks[half] = check
    main_half = "public" if split else None
    check = dict(checks[main_half])
    check["ok"] = all(c["ok"] for c in checks.values())
    return {"out": out_dir, "qset_version": qset_version, "manifest": written[main_half]["manifest"], "check": check,
            "report": written[main_half]["report"], "questions": written[main_half]["questions"],
            "candidates": candidates, "halves": written if split else None, "checks": checks,
            "sealed_out": sealed_dir,
            "cost_usd": round(math.fsum(record_cost(c) or 0.0 for c in calls), 2) if model_id in PRICES else None}


def _public(cand: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cand.items() if k != "question"}


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lines(path: str, lines: List[str]) -> None:
    """Rewrites a log whole, through a temporary file moved into place, so a stop partway
    never leaves it half written."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp, path)


def _manifest(**k: Any) -> Dict[str, Any]:
    questions, corpus, report, states = k["questions"], k["corpus"], k["report"], k["states"]
    calls, core, half = k["calls"], k["core"], k["half"]
    reasons = []
    if k["store_ids"]:
        reasons.append("store ids (private set)")
    if not report["gates_passed"]:
        reasons.append("gates failed: " + ", ".join(report["failed_gates"]))
    commit = rb.harness_commit()
    served = sorted({c["served_model"] for c in calls if c["served_model"]})
    frozen_by: Dict[str, Counter] = defaultdict(Counter)
    for cand in k["frozen"]:
        frozen_by[cand["category"]][cand["stage"]] += 1
    per_category = {}
    for cat in rc.CATEGORIES:
        st = states[cat]
        if not st.target:
            continue
        n = st.counts
        row = {
            "target": st.target, "units_shown": n["units_shown"], "calls": n["calls"],
            "unusable_calls": n["unusable_calls"], "candidates": n["candidates"], "skipped": n["skipped"],
            "dropped_by_checks": n["dropped_by_checks"], "dropped_by_screen": n["dropped_by_screen"],
            "held_by_lint": n["held_by_lint"], "dropped_by_lint": n["dropped_by_lint"],
            "publishable": n["publishable"], "frozen": n["frozen"], "surplus": n["surplus"],
            "retry_calls": n["retry_calls"], "frozen_after_retry": n["frozen_after_retry"],
            "padded_retries": n["padded_retries"], "stopped": st.stopped,
        }
        if core is not None:
            row.update({"units_shown_core": n["units_shown_core"], "frozen_core": frozen_by[cat]["core"],
                        "frozen_tail": frozen_by[cat]["tail"]})
        per_category[cat] = row
    by_cat = Counter(q["category"] for q in questions)
    total = math.fsum(record_cost(c) or 0.0 for c in calls)
    priced = k["model_id"] in PRICES
    cost_of = lambda f: sum(c[f] for c in calls)  # noqa: E731
    sampler = {
        "seed": k["seed"], "sample_size": len(k["shown"]), "pool": corpus.N,
        "gold_window": {"records": k["window"], "past_the_window": corpus.N - k["window"],
                        "rule": "gold comes only from the first records of the corpus file; the lint and the "
                                "abstention screen still read every record"},
        "eligible_records": k["eligible_records"], "eligible_units": len(k["reps"]),
        "excluded": dict(sorted(Counter(k["why"].values()).items())),
        "near_copy_clusters": sum(1 for rid in corpus.ids
                                  if corpus.cluster[rid][0] == rid and len(corpus.cluster[rid]) > 1),
        "records_with_ts": sum(1 for rid in corpus.ids if corpus.ts[rid] is not None),
        "records_with_source": sum(1 for rid in corpus.ids if corpus.source[rid] is not None),
        "strata": dict(sorted(Counter(f"{corpus.kind[r]} / {k['bands'][r]}" for r in k["reps"]).items())),
        "age_bands": "quartiles of the gold window's own ts, plus a no-ts band",
        "max_sample_share": MAX_SAMPLE_SHARE, "sample_cap": k["sample_cap"],
        "sample_cap_of": "eligible memories in the gold window, one per near-copy cluster; split evenly over the "
                         "running categories each round",
        "units_per_target": UNITS_PER_TARGET, "min_units": MIN_UNITS, "batch": BATCH,
        "topic_neighbors": TOPIC_NEIGHBORS, "pair_min_similarity": PAIR_MIN_SIMILARITY,
        "near_copy_rule": f"each covers at least {NEAR_COPY_OVERLAP} of the other's tokens, with the same digit "
                          "tokens and the same capitalized words",
        "long_horizon_min_ts": LONG_HORIZON_MIN_TS,
    }
    if k["unit_caps"]:
        sampler["unit_caps"] = {"rule": "--unit-cap: these categories' caps, in place of units_per_target times "
                                        "their targets", **dict(sorted(k["unit_caps"].items()))}
    if k["rare"] is not None:
        sampler["entity_attribution"] = {
            "rule": "units only from records that hold a rare name: a capitalized word, or a run of capitalized "
                    f"words, found in at most {RARE_NAME_MAX_OTHERS} other record of the whole pool outside the "
                    "record's near-copy cluster",
            "max_other_records": RARE_NAME_MAX_OTHERS, "records_with_a_rare_name": len(k["rare"]),
            "in_the_core": sum(1 for r in k["rare"] if corpus.pos[r] < core) if core is not None else None}
    if k["planted"] is not None:
        sampler["planted_pairs"] = {
            **k["planted"],
            "rule": "the pool's record of its planted changes, by type: version_chain pairs to knowledge_update, "
                    "correction to correction_handling, preference_change to recency_preference, in seeded order; "
                    "the core stage takes planted pairs only, and past the core knowledge_update alone goes on "
                    "with similarity pairs"}
    if core is not None:
        quota = k["core_quota"]
        sampler["core"] = {"records": core, "eligible_units": sum(1 for r in k["reps"] if corpus.pos[r] < core),
                           "quota": quota,
                           "rule": ("core first: each category's units come from the core until it runs out or the "
                                    "target is met, then from the rest of the gold window" if quota is None else
                                    f"core first, with a quota: each category's core stage ends once {quota} of its "
                                    "items from the core have frozen, once it has shown half its unit cap, or once "
                                    "its core units run out, whichever comes first; a core item past the quota is "
                                    "surplus; then the rest of the gold window")
                                   + "; each unit's records sit wholly in one of the two"}
    generation: Dict[str, Any] = {
        "per_category": per_category,
        "abstention_screen": {"max_words": PROBE_MAX_TOKENS, "min_words_shared_with_topic": TOPIC_MIN_SHARED,
                              "frozen_answer": ABSTENTION_ANSWER, "frozen_alternates": ABSTENTION_ALTERNATES},
        "retry": {"rule": "an item the lint holds only on GOLD_MULTIPLICITY leaks goes back to the model once, with "
                          "the leaking words named and no other record shown, while its category is short of its "
                          "target; the prompt asks for a different, more distinctive fact, and a retried item whose "
                          "answer only adds words around the held answer is dropped (padded retry)",
                  "calls": sum(n["retry_calls"] for n in per_category.values()),
                  "padded_retries": sum(n["padded_retries"] for n in per_category.values())},
    }
    if half == "public":  # the generation record would show the sealed half, so it stays with it
        generation["record"] = ("generation/sample.json, calls.jsonl, candidates.jsonl and run.json are held with "
                                "the sealed half and never published")
    else:
        sampler.update({"file": "generation/sample.json", "sha256": rc.file_sha256(k["sample_path"])})
        generation.update({
            "calls": {"file": "generation/calls.jsonl", "sha256": rc.file_sha256(k["calls_path"])},
            "candidates": {"file": "generation/candidates.jsonl", "sha256": rc.file_sha256(k["cands_path"])},
            "run": {"file": "generation/run.json", "sha256": rc.file_sha256(k["run_path"])},
        })
    m = {
        "qset_version": k["qset_version"],
        "spec": rc.SPEC_VERSION,
        "status": "interim" if reasons else "standard",
        "standard_eligible": not reasons,
        "status_reason": ("interim: " + "; ".join(reasons)) if reasons else "standard: every gate passes",
        "id_scheme": "store" if k["store_ids"] else "content",
        "files": {"questions.json": rc.file_sha256(k["q_path"]), "corpus.jsonl": rc.file_sha256(k["c_path"])},
        "counts": {"questions": len(questions),
                   "answerable": sum(1 for q in questions if q["gold_ids"]),
                   "abstention": sum(1 for q in questions if not q["gold_ids"]),
                   "records": corpus.N, "by_category": dict(sorted(by_cat.items()))},
    }
    if core is not None:
        ids = [q["id"] for q, cand in zip(questions, k["cands"]) if cand["stage"] == "core"]
        in_core = set(ids)
        m["core_subset"] = {
            "records": core,
            "rule": f"every gold record (for abstention, every record of its topic) is among the first {core} records "
                    "of the corpus; these questions are the set's part on the track of that size",
            "questions": len(ids),
            "by_category": dict(sorted(Counter(q["category"] for q in questions if q["id"] in in_core).items())),
            "ids": ids,
        }
    m.update({
        "targets": {c: k["targets"].get(c, 0) for c in rc.CATEGORIES},
        "model": {"requested": k["model_id"], "served": served, "effort": EFFORT, "max_tokens": MAX_TOKENS,
                  "system_cache": SYSTEM_CACHE,
                  "server_side_fallbacks": False, "sdk": next((c["sdk"] for c in calls if c["sdk"]), None),
                  "started": calls[0]["started"] if calls else None, "finished": calls[-1]["time"] if calls else None,
                  "calls": len(calls), "retry_calls": sum(1 for c in calls if c.get("retry_of") is not None),
                  "input_tokens": cost_of("input_tokens"), "output_tokens": cost_of("output_tokens"),
                  "cache_creation_input_tokens": cost_of("cache_creation_input_tokens"),
                  "cache_creation_1h_input_tokens": sum(c.get("cache_creation_1h_input_tokens", 0) for c in calls),
                  "cache_read_input_tokens": cost_of("cache_read_input_tokens"),
                  "cost_usd": round(total, 2) if priced else None,
                  "cost_note": "from the call log's token counts at the prices in PRICES; the bill is the authority"},
        "prompts": {c: {"file": f"prompts/{c}.md", "sha256": p.sha256} for c, p in k["prompts"].items()},
        "sampler": sampler,
        "generation": generation,
    })
    if half is not None:
        other = k["other"]
        rule = ["category"] + (["core or tail"] if core is not None else [])
        block: Dict[str, Any] = {"seed": k["split_seed"], "stratified_by": rule, "half": half,
                                 "how": "within each stratum, the frozen items in generation order are shuffled with "
                                        "the split seed and the first half is public; an odd stratum's extra item "
                                        "goes to the public and the sealed half in turn"}
        if half == "public":
            block["sealed"] = {"questions_sha256": rc.file_sha256(other["q_path"]),
                               "note": "the sealed half is never published; this hash commits to its questions.json"}
        else:
            block["public"] = {"qset_version": other["name"], "questions_sha256": rc.file_sha256(other["q_path"])}
        m["split"] = block
    m.update({
        "lint": {"report": "lint_report.json", "sha256": rc.file_sha256(k["lint_path"]),
                 "spec_version": k["spec"].get("spec_version"),
                 "items_with_errors": report["summary"]["items_with_errors"],
                 "gates_passed": report["gates_passed"], "failed_gates": report["failed_gates"],
                 "gates": [{"id": g["id"], "passed": g["passed"], "applicable": g["applicable"]}
                           for g in report["gates"]]},
        "tool": {"file": "generate_questions.py", "sha256": _lf_sha256(os.path.abspath(__file__)),
                 "harness_commit": commit["commit"], "dirty": commit["dirty"]},
        "provenance": (f"Questions written by {', '.join(served) or k['model_id']} from records sampled with seed "
                       f"{k['seed']}, with the frozen prompts listed above; every model response is in "
                       "generation/calls.jsonl" + (" (held with the sealed half)" if half == "public" else "")
                       + ", the lint decided what ships, and no item was edited by hand. No model is called by the "
                       "scorer."),
    })
    return m


# ---------------------------------------------------------------------------
# The estimate (docs/generation.md section 16)
# ---------------------------------------------------------------------------

def read_calibration(folder: str) -> Dict[str, Any]:
    """What a finished run measured, for --estimate-from, per category and per stage (the
    core, the rest of the window, or all for a run without a core): units shown,
    publishable items and their yield per unit, output tokens per call and per retry,
    retries per call, and whether the stage showed too few units to trust (thin); and the
    characters per input token over its calls. folder is a set that holds its generation
    record (a pilot, or the sealed half of a split set)."""
    m_path = os.path.join(folder, "manifest.json")
    gen = os.path.join(folder, "generation")
    try:
        with open(m_path, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(os.path.join(gen, "calls.jsonl"), encoding="utf-8") as f:
            calls = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(gen, "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
            pos = {json.loads(line)["id"]: i for i, line in enumerate(line for line in f if line.strip())}
        per_category = manifest["generation"]["per_category"]
    except (OSError, ValueError, KeyError) as exc:
        raise GenerateError(f"--estimate-from {folder}: needs a finished set with its generation record "
                            f"(manifest.json, corpus.jsonl, generation/calls.jsonl and candidates.jsonl): {exc}")
    core = (manifest.get("sampler", {}).get("core") or {}).get("records")

    def stage_of(unit: Sequence[str]) -> str:
        if core is None:
            return "all"
        return "core" if all(pos.get(r, core) < core for r in unit) else "tail"

    prompt_chars = {}
    for cat in {c["category"] for c in calls}:
        path = os.path.join(gen, "prompts", f"{cat}.md")
        if os.path.exists(path):
            with open(path, "rb") as f:
                prompt_chars[cat] = len(_lf(f.read()).decode("utf-8"))
    out: Dict[str, Any] = {"from": folder, "model": manifest.get("model", {}).get("requested"), "core": core,
                           "categories": {}}
    chars = tokens = 0
    for cat in per_category:
        buckets: Dict[str, Counter] = defaultdict(Counter)
        cand_stage = {c["n"]: c.get("stage") or "all" for c in cands if c["category"] == cat}
        for c in (c for c in calls if c["category"] == cat):
            if c.get("retry_of") is None:
                stages = [stage_of(u) for u in c["units"]]
                for s in stages:
                    buckets[s]["units"] += 1
                first = buckets[stages[0] if stages else "all"]  # a call is counted with its first unit's stage
                first["calls"] += 1
                first["output"] += c["output_tokens"]
            else:
                again = buckets[cand_stage.get(c["retry_of"], "all")]
                again["retries"] += 1
                again["retry_output"] += c["output_tokens"]
            if cat in prompt_chars:
                chars += prompt_chars[cat] + len(c["user"])
                tokens += c["input_tokens"] + c.get("cache_creation_input_tokens", 0) + c.get("cache_read_input_tokens", 0)
        for c in cands:
            if c["category"] == cat and c["fate"] in ("frozen", "surplus"):
                buckets[cand_stage[c["n"]]]["publishable"] += 1
        if core is not None:
            buckets["all"] = sum((buckets[s] for s in ("core", "tail")), Counter())

        def rates(b: Counter) -> Dict[str, Any]:
            return {"units": b["units"], "publishable": b["publishable"], "calls": b["calls"],
                    "retries": b["retries"], "thin": b["units"] < EST_TRUST_UNITS,
                    "yield": b["publishable"] / b["units"] if b["units"] else None,
                    "output_per_call": b["output"] / b["calls"] if b["calls"] else None,
                    "output_per_retry": b["retry_output"] / b["retries"] if b["retries"] else None,
                    "retries_per_call": b["retries"] / b["calls"] if b["calls"] else None}

        stages = {s: rates(b) for s, b in sorted(buckets.items())}
        whole = stages.get("all") or rates(Counter())
        out["categories"][cat] = {"stages": stages, "yield": whole["yield"], "output_per_call": whole["output_per_call"],
                                  "output_per_retry": whole["output_per_retry"],
                                  "retries_per_call": whole["retries_per_call"]}
    out["chars_per_token"] = chars / tokens if tokens else None
    return out


def read_calibrations(folders: Sequence[str]) -> Dict[str, Any]:
    """What several finished runs measured, as one calibration (--estimate-from with more
    than one folder): a later folder's measurements replace an earlier folder's, category by
    category, and so do its characters per input token. So the full run is estimated from
    the v11 pilot plus the mini pilot of the categories that changed after it."""
    out: Dict[str, Any] = {"from": [], "model": None, "core": None, "categories": {}, "chars_per_token": None}
    for folder in folders:
        cal = read_calibration(folder)
        out["from"].append(folder)
        out["categories"].update(cal["categories"])
        for key in ("model", "core", "chars_per_token"):
            if cal.get(key) is not None:
                out[key] = cal[key]
    return out


def estimate(plans: Dict[str, Plan], targets: Dict[str, int], corpus: Corpus, prompts: Dict[str, Prompt],
             model_id: str, recorded: Optional[ReplayModel] = None, calibration: Optional[Dict[str, Any]] = None,
             core_quota: Optional[int] = None, sample_cap: Optional[int] = None) -> Dict[str, Any]:
    """The questions, calls, tokens and cost a run is likely to come to, before any call
    (section 16). Input tokens come from the planned messages themselves, with each
    category's system prompt written to the cache on its first call and read from it after
    (SYSTEM_CACHE). Yields, output tokens and retries are the EST_* figures, or what a
    finished pilot measured (calibration), stage by stage. Expected: each category walks its
    core stage until the core quota, its target or the core's units, then the rest of the
    window until its target or the unit cap, at each stage's yield; a category the cap
    leaves short of its target is marked. Worst case: every category runs to its unit cap,
    at the same tokens and retries per call."""
    cal = (calibration or {}).get("categories", {})
    ratio = (calibration or {}).get("chars_per_token") or EST_CHARS_PER_TOKEN
    rows = {}
    shown_expected: set = set()  # the distinct records the planned units would show, against the sample cap
    shown_worst: set = set()
    for cat in rc.CATEGORIES:
        target = targets.get(cat, 0)
        plan = plans[cat]
        if not target or not plan.units:
            continue
        kind, batch, cap = UNIT_KIND[cat], BATCH[UNIT_KIND[cat]], plan.cap
        measured = cal.get(cat, {}).get("stages", {})

        def rates(stage: str) -> Dict[str, Any]:
            whole = measured.get("all") or {}
            m = measured.get(stage) or {}
            if not m.get("units"):
                m = whole  # a pilot without this stage: its rates over all its units

            def pick(field: str, guess: float) -> float:
                # A stage can hold units but no call of its own (a batch that ran on from the
                # core): then its tokens per call are the pilot's over all its calls.
                return next((s[field] for s in (m, whole) if s.get(field) is not None), guess)

            known = bool(m.get("units"))
            guess = EST_YIELD_PLANTED if stage in plan.planted_stages else EST_YIELD[cat]
            return {"yield": m["yield"] if known else guess, "measured": known,
                    "thin": known and m["units"] < EST_TRUST_UNITS, "units_measured": m.get("units", 0),
                    "publishable_measured": m.get("publishable", 0),
                    "output_per_call": pick("output_per_call", EST_OUTPUT_PER_CALL[kind]),
                    "output_per_retry": pick("output_per_retry", EST_OUTPUT_PER_RETRY),
                    "retries_per_call": 0.0 if kind == "topic" else pick("retries_per_call", EST_RETRIES_PER_CALL)}

        if plan.n_core is None:
            lists = [("all", plan.units)]
        else:
            lists = [("core", plan.units[:plan.n_core]), ("tail", plan.units[plan.n_core:])]
        r = {stage: rates(stage) for stage, _ in lists}

        def walk(stage_target: float, units: List[Tuple[str, ...]], y: float, room: int) -> Tuple[int, float]:
            limit = max(0, min(len(units), room))
            if stage_target <= 0 or not limit:
                return 0, 0.0
            used = limit if y <= 0 else min(limit, math.ceil(stage_target / y))
            return used, min(stage_target, y * used)

        expected: Dict[str, Tuple[int, float]] = {}
        worst: Dict[str, int] = {}
        left, room = float(target), cap
        for stage, units in lists:
            goal = min(left, core_quota) if stage == "core" and core_quota is not None else left
            expected[stage] = walk(goal, units, r[stage]["yield"], room)
            worst[stage] = max(0, min(len(units), cap - sum(worst.values())))
            left -= expected[stage][1]
            room -= expected[stage][0]
            for unit in units[:expected[stage][0]]:
                shown_expected.update(unit)
            for unit in units[:worst[stage]]:
                shown_worst.update(unit)

        def cost(used: Dict[str, int]) -> Dict[str, Any]:
            calls = retries = 0
            user_chars = out_tokens = 0.0
            for stage, units in lists:
                n = used[stage]
                chunks = [units[i:i + batch] for i in range(0, n, batch)]
                more = round(len(chunks) * r[stage]["retries_per_call"])
                chars = sum(len(user_message(cat, c, corpus)) for c in chunks)
                user_chars += chars + more * (chars / max(n, 1) + 400)  # a retry shows one unit and the held item
                out_tokens += len(chunks) * r[stage]["output_per_call"] + more * r[stage]["output_per_retry"]
                calls, retries = calls + len(chunks), retries + more
            system = len(prompts[cat].text) / ratio
            sent = calls + retries
            if SYSTEM_CACHE is not None and sent:
                uncached, write, read = user_chars / ratio, system, system * (sent - 1)
            else:
                uncached, write, read = user_chars / ratio + system * sent, 0.0, 0.0
            hour = write if (SYSTEM_CACHE or {}).get("ttl") == "1h" else 0.0
            return {"calls": calls, "retries": retries, "input_tokens": int(uncached), "cache_write_tokens": int(write),
                    "cache_read_tokens": int(read), "output_tokens": int(out_tokens),
                    "cost_usd": cost_exact(model_id, int(uncached), int(out_tokens), int(write - hour), int(read),
                                           int(hour))}

        frozen = math.fsum(f for _, f in expected.values())
        spend, most = cost({s: u for s, (u, _) in expected.items()}), cost(worst)
        if (most["cost_usd"] or 0) < (spend["cost_usd"] or 0):  # batches round differently at the stage boundary
            most = dict(spend)
        rows[cat] = {
            "target": target, "unit_cap": cap,
            "stages": {stage: {"units_available": len(units), "units": expected[stage][0],
                               "frozen": round(expected[stage][1], 1), "yield": round(r[stage]["yield"], 3),
                               "measured": r[stage]["measured"], "thin": r[stage]["thin"],
                               "units_measured": r[stage]["units_measured"],
                               "publishable_measured": r[stage]["publishable_measured"]}
                       for stage, units in lists},
            "frozen": round(frozen, 1), "frozen_core": round(expected["core"][1], 1) if "core" in expected else None,
            "short": frozen < target - 1e-9,
            "expected": spend, "worst": most, "worst_units": sum(worst.values()),
        }

    def total(which: str) -> Dict[str, Any]:
        t = {f: sum(row[which][f] for row in rows.values())
             for f in ("calls", "retries", "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")}
        costs = [row[which]["cost_usd"] for row in rows.values()]
        t["cost_usd"] = None if any(c is None for c in costs) else math.fsum(costs)
        return t

    cores = [row["frozen_core"] for row in rows.values() if row["frozen_core"] is not None]
    out = {"model": model_id, "categories": rows, "expected": total("expected"), "worst": total("worst"),
           "questions": round(math.fsum(row["frozen"] for row in rows.values()), 1),
           "core_subset": round(math.fsum(cores), 1) if cores else None, "core_quota": core_quota,
           "short": [cat for cat, row in rows.items() if row["short"]],
           "thin": [(cat, stage, s["units_measured"], s["publishable_measured"]) for cat, row in rows.items()
                    for stage, s in row["stages"].items() if s["thin"]],
           "cached": None if SYSTEM_CACHE is None else SYSTEM_CACHE.get("ttl", "5m"), "chars_per_token": ratio,
           "calibration": (calibration or {}).get("from"),
           "records_shown": {"expected": len(shown_expected), "worst": len(shown_worst), "sample_cap": sample_cap}}
    if recorded is not None:
        out["recorded"] = {"calls": len(recorded.good_lines), "cost_usd": recorded.cost}
    return out


def print_estimate(est: Dict[str, Any], say: Callable[..., None]) -> None:
    money = lambda v: "n/a" if v is None else f"${v:.2f}"  # noqa: E731
    measured = est["calibration"]
    if isinstance(measured, list):
        measured = ", ".join(measured)
    say(f"Estimate for {est['model']}, before any call ("
        + (f"yields and tokens measured by {measured}" if measured else
           "rough: yields and output tokens are the EST_* figures until a pilot measures them")
        + f"; {est['chars_per_token']:.2f} characters per input token; system prompts "
        + ({"1h": "cached for an hour", "5m": "cached for five minutes"}.get(est["cached"], "cached")
           if est["cached"] else "not cached")
        + (f"; at most {est['core_quota']} items per category from the core" if est["core_quota"] is not None else "")
        + "):")
    staged = any("core" in row["stages"] for row in est["categories"].values())

    def cells(s: Optional[Dict[str, Any]]) -> str:
        if s is None:
            return " " * 28
        mark = ("" if s["measured"] else "*") + ("?" if s["thin"] else "")
        return f"{s['units']:>11}{s['yield']:>7.2f}{mark:<2}{s['frozen']:>8.1f}"

    first, second = ("core", "tail") if staged else ("all", None)
    label = lambda stage: f"{stage + ' units' if staged else 'units':>11}{'yield':>7}{'':2}{'frozen':>8}"  # noqa: E731
    head = (f"  {'category':22}{'target':>7}" + label(first) + (label(second) if staged else "")
            + f"{'frozen':>8}{'calls':>7}{'expected':>10}{'worst calls':>13}{'worst':>9}")
    say(head)
    for cat, row in est["categories"].items():
        e, w, st = row["expected"], row["worst"], row["stages"]
        line = (f"  {cat:22}{row['target']:>7}" + cells(st.get(first))
                + (cells(st.get(second)) if staged else "")
                + f"{row['frozen']:>8.1f}{e['calls'] + e['retries']:>7}{money(e['cost_usd']):>10}"
                + f"{w['calls'] + w['retries']:>13}{money(w['cost_usd']):>9}" + ("  short" if row["short"] else ""))
        say(line)
    e, w = est["expected"], est["worst"]
    subset = (f" ({est['core_subset']:.0f} of them with every gold record in the core: the 1k subset)"
              if est["core_subset"] is not None else "")
    say(f"  expected: about {est['questions']:.0f} questions{subset}; about {e['calls'] + e['retries']:,} calls "
        f"({e['retries']:,} of them retries), {e['input_tokens']:,} input tokens, {e['cache_write_tokens']:,} written "
        f"to and {e['cache_read_tokens']:,} read from the cache, {e['output_tokens']:,} output tokens; about "
        f"{money(e['cost_usd'])}")
    say(f"  worst case: every category runs to its unit cap: about {w['calls'] + w['retries']:,} calls, "
        f"{w['input_tokens']:,} input tokens, {w['cache_write_tokens']:,} written to and {w['cache_read_tokens']:,} "
        f"read from the cache, {w['output_tokens']:,} output tokens; about {money(w['cost_usd'])}")
    shown = est.get("records_shown")
    if shown and shown["sample_cap"] is not None:
        say(f"  records shown to the model: about {shown['expected']:,} expected and {shown['worst']:,} at the worst "
            f"case, against the sample cap of {shown['sample_cap']:,} (half the eligible memories)"
            + ("; the sample cap would stop the run before the unit caps" if shown["expected"] > shown["sample_cap"]
               else "; the worst case passes the sample cap, which then ends the categories still running"
               if shown["worst"] > shown["sample_cap"] else ""))
    if est["short"]:
        say("  short of target at the unit cap: " + ", ".join(
            f"{cat} (about {est['categories'][cat]['frozen']:.0f} of {est['categories'][cat]['target']})"
            for cat in est["short"]))
    if est["thin"]:
        say("  too few units to trust: " + ", ".join(
            f"{cat} {stage} ({units} units, {items} publishable)" for cat, stage, units, items in est["thin"]))
    if "recorded" in est:
        say(f"  the log already holds {est['recorded']['calls']} calls ({money(est['recorded']['cost_usd'])}); "
            "a --resume replays them free")
    say(f"  (* a guessed yield; ? measured on fewer than {EST_TRUST_UNITS} units. A reply can run longer than these "
        "averages, and the cache saves only while it holds; --max-cost is the guard, and the bill is the authority.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any]) -> None:
    m = result["manifest"]
    halves = result.get("halves")
    if halves:
        print(f"\nWrote {result['out']} ({m['qset_version']}, the public half, status {m['status']}) and "
              f"{result['sealed_out']} ({halves['sealed']['manifest']['qset_version']}, the sealed half, status "
              f"{halves['sealed']['manifest']['status']})")
    else:
        print(f"\nWrote {result['out']} ({m['qset_version']}, status {m['status']})")
    header = ("category", "target", "units", "cand", "skip", "check", "screen", "held", "retry", "drop", "frozen",
              "surplus")
    print("  " + f"{header[0]:22}" + "".join(f"{h:>8}" for h in header[1:]) + "  stopped")
    for cat, n in m["generation"]["per_category"].items():
        cells = (n["target"], n["units_shown"], n["candidates"], n["skipped"], n["dropped_by_checks"],
                 n["dropped_by_screen"], n["held_by_lint"], n["retry_calls"], n["dropped_by_lint"], n["frozen"],
                 n["surplus"])
        print("  " + f"{cat:22}" + "".join(f"{c:>8}" for c in cells) + f"  {n['stopped'] or ''}")
    model = m["model"]
    cost = f"${model['cost_usd']:.2f}" if model["cost_usd"] is not None else "no price on file"
    print(f"  model: {model['calls']} calls ({model['retry_calls']} retries), {model['input_tokens']:,} input and "
          f"{model['output_tokens']:,} output tokens, {cost}; {model['started']} to {model['finished']} (UTC)")
    for half, w in (halves or {None: {"manifest": m}}).items():
        lint = w["manifest"]["lint"]
        label = f"{half} half: " if half else ""
        subset = w["manifest"].get("core_subset")
        print(f"  {label}lint: {w['manifest']['counts']['questions']} questions, {lint['items_with_errors']} with "
              "errors; gates " + ("all pass" if lint["gates_passed"] else "failed: " + ", ".join(lint["failed_gates"]))
              + (f"; {subset['questions']} with every gold record in the first {subset['records']}" if subset else ""))
    for half, check in result["checks"].items():
        label = f"{half} half: " if half else ""
        if check["ok"]:
            print(f"  {label}oracle: {check['gold_at_window']} on the gold tier at the window "
                  f"(verified={check['verified']})")
        else:
            print(f"  {label}set broken: the oracle missed "
                  + "; ".join(f"{x['id']} ({x['reason']})" for x in check["missed"]))
        if check["reference"]:
            print(f"  {label}bm25 floor: {check['reference']['bm25']['result_line']}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"Parsica Recall question generator ({rc.SPEC_VERSION}); "
                                            "design in docs/generation.md")
    p.add_argument("--corpus", default=None, help="corpus.jsonl to write questions from. Required, except with "
                                                  "--check-api")
    p.add_argument("--out", default=None,
                   help="Folder for the set (the public half of a split set); must not exist yet or be empty, and "
                        "must be outside this repository (a public set under data/ needs --release). Its name "
                        "becomes the set's version name. Required, except with --check-api")
    p.add_argument("--seed", type=int, default=None, help="Sampler seed. Required, except with --check-api")
    p.add_argument("--targets", default=str(DEFAULT_TARGET),
                   help="Publishable items per category: one number for every category, or category=number pairs "
                        f"(default {DEFAULT_TARGET}, the standard minimum)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Generation model (default {DEFAULT_MODEL})")
    p.add_argument("--store-ids", action="store_true",
                   help="The corpus keeps its store's own ids (a private set); never a standard result")
    p.add_argument("--release", action="store_true",
                   help="Allow an --out under data/ for a public set, and a --sealed-out under sealed/")
    p.add_argument("--replay", default=None,
                   help="Rebuild from a recorded generation/calls.jsonl instead of calling the API")
    p.add_argument("--gold-window", type=int, default=None, metavar="N",
                   help="Gold comes only from the corpus's first N records (default: the whole corpus); the lint "
                        "still reads every record")
    p.add_argument("--core", type=int, default=None, metavar="N",
                   help="The corpus's first N records are the core: every category draws its units from the core "
                        "first, then from the rest of the gold window, and the manifest marks the questions whose "
                        "gold is all in the core")
    p.add_argument("--core-quota", type=int, default=None, metavar="N",
                   help="With --core: at most N items per category come from the core. A category's core stage ends "
                        "once N of its core items have frozen, once it has shown half its unit cap, or once its core "
                        "units run out, whichever comes first; then the rest of the gold window")
    p.add_argument("--split-seed", type=int, default=None, metavar="N",
                   help="Split the frozen items, seeded and stratified, into a public half (--out) and a sealed half "
                        "(--sealed-out)")
    p.add_argument("--sealed-out", default=None, metavar="DIR",
                   help="Folder for the sealed half, which keeps the generation record; under sealed/ with --release. "
                        "Never published")
    p.add_argument("--max-cost", type=float, default=None, metavar="USD",
                   help="The most the set may cost, recorded calls included: the run stops before its next new call "
                        "once the cost passes it (exit 3), and --resume with a higher number goes on. Required for a "
                        "live run")
    p.add_argument("--estimate", action="store_true",
                   help="Print the expected cost and the worst case, then stop before any call")
    p.add_argument("--estimate-from", default=None, metavar="DIR", nargs="+", action="extend",
                   help="With --estimate: use the tokens and yields a finished pilot measured (a set folder that "
                        "holds its generation record). Several folders may be given; a later one's measurements "
                        "replace an earlier one's, category by category")
    p.add_argument("--planted-pairs", default=None, metavar="FILE",
                   help="A pool's record of its planted changes (export_planted_pairs.py): the version categories "
                        "build their core units from these pairs of record ids by type, correction_handling and "
                        "recency_preference from them alone, and knowledge_update goes on with similarity pairs "
                        "past the core")
    p.add_argument("--unit-cap", default=None, metavar="CATEGORY=N,...",
                   help="A category's unit cap in place of 4 times its target (at least 12), such as "
                        "knowledge_update=1200,implicit_preference=900")
    p.add_argument("--resume", action="store_true",
                   help="Pick up a stopped run from its own call log; the calls it holds replay free")
    p.add_argument("--check-api", action="store_true",
                   help="One tiny call that proves the key and the network: ok and the model, or the error's class")
    p.add_argument("--api-key-env", default=None, metavar="NAME",
                   help="Read the API key from this one environment variable instead of ANTHROPIC_API_KEY, and send "
                        "every call to api.anthropic.com whatever ANTHROPIC_BASE_URL says. Not recorded")
    args = p.parse_args(argv)
    if args.check_api:
        try:
            served = check_api(args.model, args.api_key_env)
        except GenerateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"ok: {served} answered")
        return 0
    missing = [flag for flag, value in (("--corpus", args.corpus), ("--out", args.out), ("--seed", args.seed))
               if value is None]
    if missing:
        print(f"ERROR: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required", file=sys.stderr)
        return 2
    if args.estimate_from and not args.estimate:
        print("ERROR: --estimate-from goes with --estimate", file=sys.stderr)
        return 2
    try:
        targets = parse_targets(args.targets)
        unit_caps = parse_unit_caps(args.unit_cap)
        if not (args.replay or args.estimate) and args.max_cost is None:
            raise GenerateError("a live run needs --max-cost USD, the most the set may cost; run with --estimate "
                                "first to see what it is likely to cost")
        result = generate_set(args.corpus, args.out, args.seed, targets, model_id=args.model,
                              store_ids=args.store_ids, release=args.release, replay=args.replay,
                              resume=args.resume, max_cost=args.max_cost, gold_window=args.gold_window,
                              core=args.core, split_seed=args.split_seed, sealed_out=args.sealed_out,
                              api_key_env=args.api_key_env, estimate_only=args.estimate,
                              estimate_from=args.estimate_from, core_quota=args.core_quota,
                              planted_pairs=args.planted_pairs, unit_caps=unit_caps)
    except CostStop as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 3
    except (GenerateError, rb.InputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.estimate:
        return 0
    print_summary(result)
    return 0 if result["check"]["ok"] else 1


if __name__ == "__main__":
    # Determinism: pin the hash seed and run again, as the runner does.
    _status = rc.pin_hash_seed(f"{rc.PACKAGE}.generate_questions", sys.argv[1:])
    sys.exit(main() if _status is None else _status)

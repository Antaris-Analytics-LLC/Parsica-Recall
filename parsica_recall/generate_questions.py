#!/usr/bin/env python3
"""
Parsica Recall question generator (spec recall/1.0, section 9). The design is in
docs/generation.md, and the frozen prompts are in prompts/, one file per category.

Takes any corpus.jsonl (an export of your own store, or the public pool) and writes a
frozen, linted question set with every prompt, seed and model response recorded:

    parsica-recall generate-questions --corpus export/corpus.jsonl --out ~/recall-sets/mystore-v1 \\
        --seed 7 --targets 25 --max-cost 5 [--provider openai] [--model MODEL] [--store-ids]

Generation is the one step of the benchmark that uses a model. Everything else here is
deterministic: the same corpus, seed, targets, prompts and model responses give the
same files byte for byte, and --replay rebuilds a set from its generation/calls.jsonl
without calling the API. A run that --max-cost stopped picks up from its own call log
with --resume, and the calls it holds cost nothing the second time.

The model is Claude through the Anthropic API (the default), an OpenAI model through
OpenAI's API (--provider openai), a Gemini model through Google's Gemini API (--provider
google), or any model behind an endpoint that speaks OpenAI's chat format (--provider
openai --base-url URL). By default each provider's SDK reads its own key variable
(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY) and this file never
reads it. With --api-key-env NAME this file reads that one variable, hands it to the SDK
with the provider's official endpoint pinned (or the --base-url endpoint), and takes it out
of every error it reports. --base-url never sends OPENAI_API_KEY: an endpoint of your own
gets a key only from --api-key-env. Either way it never prints, logs or writes the key.

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
import urllib.parse
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

# The providers the question builder can call (the pool generator calls Anthropic only).
# A model with no --model given is its provider's balanced model; an endpoint of your own
# (--base-url) has no default model. OpenAI's and Google's calls always go to the official
# endpoint below, whatever OPENAI_BASE_URL or GOOGLE_GEMINI_BASE_URL says; only --base-url
# sends them anywhere else.
PROVIDERS = ("anthropic", "openai", "google", "azure", "bedrock")
# Azure has no default: --model names a deployment, which only its owner knows.
DEFAULT_MODELS = {"anthropic": DEFAULT_MODEL, "openai": "gpt-6.2-sol", "google": "gemini-4-hydrogen",
                  "bedrock": "anthropic." + DEFAULT_MODEL}
# Azure OpenAI: an Azure resource's v1 API (https://NAME.openai.azure.com/openai/v1), which takes
# OpenAI's own request through OpenAI's own client. Its key comes from AZURE_KEY_ENV unless
# --api-key-env names another variable, and goes only to an Azure host, over https.
AZURE_HOSTS = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")
AZURE_KEY_ENV = "AZURE_OPENAI_API_KEY"
# Amazon Bedrock: Claude through the anthropic SDK's Bedrock client, which sends a Bedrock API
# key (--api-key-env's, else AWS_BEARER_TOKEN_BEDROCK's) or signs each call with your AWS
# credentials, to this address in the run's region.
BEDROCK_URL = "https://bedrock-mantle.{region}.api.aws/anthropic"
_AWS_REGION = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d{1,2}$")
OPENAI_URL = rc.OPENAI_URL  # the providers' own addresses live in recall_common, which the runner shares
GOOGLE_URL = rc.GOOGLE_URL
# OpenAI's and Google's reasoning tokens count against the output cap, and OpenAI's
# reasoning guide advises room for them; 32,000 also stays under the 32,768 tokens a
# single Gemini call returns. EFFORT ("high") is sent as reasoning_effort to OpenAI and as
# thinking_level to Google. An endpoint of your own gets neither unless --max-output-tokens
# sets the cap, so a server that knows no such field is not refused.
REASONING_MAX_TOKENS = 32000

# US dollars per million tokens: input, output, cache write, long cache write, cache read.
# One table for both generators (generate_pool.py reads it from here). Used only for the
# estimate printed before a run, the --max-cost guard and the cost a manifest reports; a
# model not listed gets token counts alone, unless --price-in and --price-out give its prices.
# - Anthropic, first-party prices as of November 22, 2026: the writes are the 5-minute and
#   the 1-hour cache (the question builder writes the 1-hour one). Cache reads are 0.05 times
#   input on Sonnet 5.5 and Opus 5.5 and 0.025 times on Fable 5.2. Haiku 5.5's prices are for
#   prompts up to 100,000 tokens, five times these above that; the builder's prompts are far
#   smaller.
# - OpenAI, standard prices for prompts up to 272,000 tokens as of November 21, 2026: the
#   write is the default 30-minute cache, at 1.25 times input (both write columns; the
#   question builder never asks for the 12-hour one).
# - Google, paid-tier prices as of November 21, 2026, with Gemini 4 Carbon and Argon at the
#   prices they move to on December 1, 2026. Gemini caches implicitly with no write charge,
#   so both write columns are the input price; thinking tokens are billed as output.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.20, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 4.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 8.00, 0.20),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 20.00, 0.25),
    "claude-fable-5-2": (8.00, 40.00, 10.00, 16.00, 0.20),
    "gpt-6.2-astra": (8.00, 40.00, 10.00, 10.00, 0.40),
    "gpt-6.2-sol": (2.00, 10.00, 2.50, 2.50, 0.10),
    "gpt-6.1-terra": (1.00, 5.00, 1.25, 1.25, 0.05),
    "gpt-6-luna": (0.10, 0.50, 0.125, 0.125, 0.01),
    "gemini-4-carbon": (4.00, 20.00, 4.00, 4.00, 0.20),
    "gemini-4-argon": (4.00, 20.00, 4.00, 4.00, 0.20),
    "gemini-4-neon": (3.00, 15.00, 3.00, 3.00, 0.15),
    "gemini-4-hydrogen": (1.50, 7.50, 1.50, 1.50, 0.075),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.30, 0.30, 0.03),
}
PRICES_AS_OF = {"anthropic": "2026-11-22", "openai": "2026-11-21", "google": "2026-11-21"}
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
        # OpenAI and Google only: the part of output_tokens the model spent reasoning, as
        # the API reports it. None for Anthropic, whose call records leave it out.
        self.reasoning_tokens: Optional[int] = None
        self.in_log = False  # under --resume: the call is already in this folder's log


def _check_env_name(name: Optional[str]) -> None:
    """--api-key-env takes a variable's name, never a key."""
    if name is not None and not _ENV_NAME.fullmatch(name):
        raise GenerateError("--api-key-env takes the name of an environment variable, such as MY_API_KEY "
                            "(letters, digits and underscores), never the key itself")


def _key_from_env(name: str) -> str:
    """The key in the one variable --api-key-env names, for every provider. Refused when
    the variable is unset or empty, or holds a character no API key has, so no error can
    ever quote the key escaped."""
    key = os.environ.get(name, "").strip()
    if not key:
        raise GenerateError(f"--api-key-env: {name} is not set in this environment, or is empty")
    if any(not 33 <= ord(c) <= 126 or c in "\"'\\" for c in key):  # so no error can quote it escaped
        raise GenerateError(f"--api-key-env: {name} holds a space, a line break, a quote or another character "
                            "an API key does not have; check its value in the environment's settings")
    return key


def _int(value: Any) -> int:
    """A token count an SDK reports, or 0 when it reports none."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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
        _check_env_name(api_key_env)
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
        key = _key_from_env(self.api_key_env)
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


class BedrockModel(AnthropicModel):
    """Claude on Amazon Bedrock: AnthropicModel's request, unchanged, through the anthropic
    SDK's Bedrock client (AnthropicBedrockMantle) to BEDROCK_URL in the run's region, the
    address ModelChoice recorded. The client sends a Bedrock API key, from the one variable
    --api-key-env names or else from AWS_BEARER_TOKEN_BEDROCK (or ANTHROPIC_AWS_API_KEY), which
    the SDK reads itself; without one, it signs each call with your AWS credentials as the AWS
    tools find them (the environment's keys, a profile), which needs the SDK's bedrock extra.
    The key or the credentials are never printed, logged or written."""

    def __init__(self, choice: ModelChoice):
        super().__init__(choice.model_id, choice.api_key_env)
        self.choice = choice

    def _connect(self) -> None:
        try:
            import anthropic
        except ImportError:
            raise GenerateError('a live run on Bedrock needs the anthropic package with its AWS extra: pip install '
                                '"parsica-recall[bedrock]" (the tests and --replay do not)')
        if not hasattr(anthropic, "AnthropicBedrockMantle"):
            raise GenerateError('this anthropic package has no Bedrock client; upgrade it: pip install -U '
                                '"parsica-recall[bedrock]"')
        self._anthropic = anthropic
        self._sdk = f"anthropic {getattr(anthropic, '__version__', 'unknown')}"
        kwargs: Dict[str, Any] = {"aws_region": self.choice.region, "base_url": self.choice.endpoint}
        if self.api_key_env is not None:
            self._key = _key_from_env(self.api_key_env)
            kwargs["api_key"] = self._key
        try:
            self._client = anthropic.AnthropicBedrockMantle(**kwargs)
        except Exception as exc:  # no AWS credentials it can find, most often
            raise GenerateError(self._scrub(f"could not create the Bedrock client ({type(exc).__name__}: {exc}); check "
                                            "the AWS credentials and the region, or name a Bedrock API key's "
                                            "variable with --api-key-env")) from None


def _provider_of(model_id: str) -> Optional[str]:
    """The provider a model's name belongs to, when the name says so."""
    if model_id.startswith("claude-"):
        return "anthropic"
    if model_id.startswith("gpt-") or re.match(r"o\d", model_id):
        return "openai"
    if model_id.startswith("gemini-"):
        return "google"
    return None


def _check_base_url(url: str, with_key: bool) -> str:
    """--base-url: an http or https URL with a host and nothing a log could leak (no user
    name, password, query or fragment). A key goes over https only, or to this machine."""
    text = url.strip()
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname or any(c.isspace() for c in text):
        raise GenerateError("--base-url takes an http:// or https:// URL, such as http://localhost:11434/v1")
    if parts.username is not None or parts.password is not None:
        raise GenerateError("--base-url must not hold a user name or password; give the endpoint's key with "
                            "--api-key-env")
    if parts.query or parts.fragment:
        raise GenerateError("--base-url takes the endpoint's base URL alone, with no ? or # part")
    local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if with_key and parts.scheme == "http" and not local:
        raise GenerateError("--base-url: a key goes to an endpoint over https only, unless the endpoint is on this "
                            "machine (localhost)")
    return text.rstrip("/")


def _check_azure_url(url: str) -> str:
    """--base-url with --provider azure: the v1 address of an Azure resource, over https, on
    an Azure host, so the key it is sent can go nowhere else."""
    text = _check_base_url(url, with_key=True)
    parts = urllib.parse.urlsplit(text)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host.endswith(AZURE_HOSTS):
        raise GenerateError("--provider azure takes your Azure resource's v1 address in --base-url, over https on an "
                            "Azure host, such as https://NAME.openai.azure.com/openai/v1")
    if not parts.path.rstrip("/").endswith("/openai/v1"):
        raise GenerateError("--base-url for Azure ends in /openai/v1, the v1 API, which takes OpenAI's own requests "
                            "(such as https://NAME.openai.azure.com/openai/v1)")
    return text


class ModelChoice:
    """Which model writes the questions and where its calls go, checked once before
    anything is written: the provider, the model, the endpoint, the variable the key is
    read from, the output cap, the effort and the price.

    Anthropic is the choice that shapes its requests, call log and manifest exactly as
    before OpenAI and Google were added, so a set built before still replays byte for
    byte; only the price flags add to its log and manifest. OpenAI and Google go to their
    official endpoints with reasoning at EFFORT and REASONING_MAX_TOKENS, priced from
    PRICES. An endpoint of your own (--base-url, provider openai) is sent neither field
    unless --max-output-tokens sets the cap, and is priced only by --price-in and
    --price-out. Azure (provider azure, --base-url your resource's v1 address, --model a
    deployment) is sent OpenAI's own request, and Bedrock (provider bedrock, a region)
    Anthropic's; both are priced only by --price-in and --price-out, since each platform
    sets its own prices by deployment and region."""

    def __init__(self, provider: str = "anthropic", model_id: Optional[str] = None, base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None, price_in: Optional[float] = None,
                 price_out: Optional[float] = None, max_output_tokens: Optional[int] = None,
                 aws_region: Optional[str] = None):
        if provider not in PROVIDERS:
            raise GenerateError(f"--provider takes one of {', '.join(PROVIDERS)}, not {provider!r}")
        _check_env_name(api_key_env)
        if base_url is not None:
            if provider not in ("openai", "azure"):
                raise GenerateError("--base-url goes with --provider openai, for an endpoint that speaks OpenAI's chat "
                                    "format, or with --provider azure, for your Azure resource")
            if provider == "azure":
                base_url = _check_azure_url(base_url)
                if not model_id:
                    raise GenerateError("--provider azure needs --model, the name of your deployment")
            else:
                base_url = _check_base_url(base_url, with_key=api_key_env is not None)
                if not model_id:
                    raise GenerateError("--base-url needs --model, the name the endpoint knows the model by")
        elif provider == "azure":
            raise GenerateError("--provider azure needs --base-url, your Azure resource's v1 address (such as "
                                "https://NAME.openai.azure.com/openai/v1), and --model, the name of your deployment")
        if aws_region is not None and provider != "bedrock":
            raise GenerateError("--aws-region goes with --provider bedrock")
        region = None
        if provider == "bedrock":
            region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise GenerateError("--provider bedrock needs a region: --aws-region, such as us-east-1, or AWS_REGION "
                                    "in the environment")
            if not _AWS_REGION.match(region):
                raise GenerateError(f"--aws-region takes an AWS region's name, such as us-east-1, not {region!r}")
        custom = provider == "openai" and base_url is not None
        if max_output_tokens is not None:
            if not custom:
                raise GenerateError(f"--max-output-tokens goes with --base-url on --provider openai; the providers' own "
                                    f"endpoints get {MAX_TOKENS} (Anthropic, Bedrock) or {REASONING_MAX_TOKENS} "
                                    "(OpenAI, Google, Azure)")
            if max_output_tokens < 1:
                raise GenerateError("--max-output-tokens takes a whole number of tokens, 1 or more")
        if (price_in is None) != (price_out is None):
            raise GenerateError("--price-in and --price-out go together: the input and the output price, in US "
                                "dollars per million tokens")
        for flag, value in (("--price-in", price_in), ("--price-out", price_out)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise GenerateError(f"{flag} takes a number of US dollars per million tokens, 0 or more")
        self.provider = provider
        self.model_id = model_id or DEFAULT_MODELS[provider]
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.custom = custom
        self.region = region
        if provider in ("anthropic", "openai", "google") and not custom:
            # Azure's models are deployments and Bedrock's carry Bedrock's own ids, so only the
            # providers' own endpoints check that a model's name is theirs.
            owner = _provider_of(self.model_id)
            if owner is not None and owner != provider:
                raise GenerateError(f"{self.model_id} is a model of {owner}'s; run it with --provider {owner}")
        if provider == "anthropic":
            self.endpoint, self.max_tokens, self.effort = None, MAX_TOKENS, EFFORT
        elif custom:
            self.endpoint, self.max_tokens, self.effort = base_url, max_output_tokens, None
        elif provider == "azure":
            self.endpoint, self.max_tokens, self.effort = base_url, REASONING_MAX_TOKENS, EFFORT
        elif provider == "bedrock":
            self.endpoint, self.max_tokens, self.effort = BEDROCK_URL.format(region=region), MAX_TOKENS, EFFORT
        else:
            self.endpoint = OPENAI_URL if provider == "openai" else GOOGLE_URL
            self.max_tokens, self.effort = REASONING_MAX_TOKENS, EFFORT
        self.flag_price = price_in is not None
        if self.flag_price:
            self.price: Optional[Tuple[float, ...]] = (price_in, price_out, price_in, price_in, price_in)
            self.price_source: Optional[str] = "--price-in and --price-out (cache reads and writes at the input price)"
        elif custom or provider in ("azure", "bedrock"):
            self.price, self.price_source = None, None
        else:
            self.price = PRICES.get(self.model_id)
            self.price_source = f"the price table as of {PRICES_AS_OF[provider]}" if self.price else None

    @property
    def claude_request(self) -> bool:
        """Anthropic's request shape: Anthropic's API itself, or Claude on Bedrock. Such a
        request marks its system prompt for the cache, and its tokens are counted as the
        estimate's guesses were measured."""
        return self.provider in ("anthropic", "bedrock")

    @property
    def other(self) -> bool:
        """A provider other than Anthropic: its requests, run.json and manifest say which."""
        return self.provider != "anthropic"

    def fields(self) -> Dict[str, Any]:
        """What a request hash, run.json, the call log and the manifest add for a provider
        other than Anthropic."""
        return {"provider": self.provider, "endpoint": self.endpoint, "effort": self.effort,
                "max_tokens": self.max_tokens}

    def priced_record(self) -> bool:
        """Whether each call record carries the price it was counted at: when the flags set
        it, and always for an endpoint of your own, which the table never prices."""
        return self.flag_price or self.custom

    def prices(self) -> Optional[Dict[str, Any]]:
        """The prices a manifest reports the cost at, by name."""
        if self.price is None:
            return None
        names = ("input", "output", "cache_write", "cache_write_long", "cache_read")
        return {"usd_per_million_tokens": dict(zip(names, self.price)), "source": self.price_source}


class OpenAIModel:
    """Chat Completions through the official openai SDK, imported on the first call, for
    OpenAI's API or an endpoint of your own that speaks the same format (--base-url). The
    reply's shape is set by the prompt's JSON schema in strict mode. To OpenAI the request
    also sends reasoning_effort, max_completion_tokens and store=false, so OpenAI keeps no
    stored copy of the completion; an endpoint of your own is sent the schema and the
    messages, and max_tokens only when --max-output-tokens sets it.

    The key: for OpenAI's API the SDK reads OPENAI_API_KEY unless --api-key-env names
    another variable, and the base URL is pinned to OPENAI_URL either way. An endpoint of
    your own gets the key in the --api-key-env variable, or a placeholder, never
    OPENAI_API_KEY. As with AnthropicModel, the key is never printed, logged or written,
    and every error raised here has it taken out and drops the exception it came from."""

    def __init__(self, choice: ModelChoice):
        self.choice = choice
        self.model_id = choice.model_id
        self._client = None
        self._key: Optional[str] = None

    def connect(self) -> None:
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        try:
            import openai
        except ImportError:
            raise GenerateError('a live run on OpenAI or Azure needs the openai package: pip install '
                                '"parsica-recall[openai]" (the tests and --replay do not)')
        self._openai = openai
        self._sdk = f"openai {getattr(openai, '__version__', 'unknown')}"
        c = self.choice
        if c.provider == "azure":
            # Azure's v1 API takes OpenAI's own client. The key goes in both headers Azure
            # reads (api-key, and the bearer token the client sends), to the resource's
            # address only, which ModelChoice has already held to https and an Azure host.
            self._key = _key_from_env(c.api_key_env or AZURE_KEY_ENV)
            try:
                self._client = openai.OpenAI(base_url=c.base_url, api_key=self._key, max_retries=2,
                                             default_headers={"api-key": self._key})
            except Exception as exc:
                raise GenerateError(self._scrub(f"could not create the client for Azure ({type(exc).__name__}: "
                                                f"{exc})")) from None
            return
        if c.api_key_env is not None:
            self._key = _key_from_env(c.api_key_env)
        kwargs: Dict[str, Any] = {"base_url": c.base_url or OPENAI_URL, "max_retries": 2}
        if self._key is not None:
            kwargs["api_key"] = self._key
        elif c.custom:
            kwargs["api_key"] = "none"  # an endpoint of your own is never sent OPENAI_API_KEY
        try:
            self._client = openai.OpenAI(**kwargs)
        except Exception as exc:  # no OPENAI_API_KEY in the environment, most often
            raise GenerateError(self._scrub(f"could not create the OpenAI client ({type(exc).__name__}: {exc}); check "
                                            "that OPENAI_API_KEY is set, or name the key's variable with "
                                            "--api-key-env")) from None

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "[the key]") if self._key else text

    def request(self, category: str, prompt: Prompt, user: str) -> Dict[str, Any]:
        """The keyword arguments of one Chat Completions call."""
        c = self.choice
        req: Dict[str, Any] = {
            "model": c.model_id,
            "messages": [{"role": "system", "content": prompt.text}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": f"recall_{category}", "schema": prompt.schema, "strict": True}},
        }
        if c.custom:
            if c.max_tokens is not None:
                req["max_tokens"] = c.max_tokens
        else:
            req.update(reasoning_effort=c.effort, max_completion_tokens=c.max_tokens, store=False)
        return req

    def call(self, category: str, prompt: Prompt, user: str, request_sha: str) -> Reply:
        if self._client is None:
            self._connect()
        started = _utc_now()
        try:
            resp = self._client.chat.completions.create(**self.request(category, prompt, user))
        except TypeError as exc:  # an SDK too old for this request
            raise GenerateError(self._scrub(f"the openai SDK refused the {category} request ({exc}); upgrade it with "
                                            "pip install -U openai")) from None
        except Exception as exc:  # an API error or anything else: a message, never a traceback that could hold the key
            raise GenerateError(self._scrub(f"the {category} call failed: {type(exc).__name__}: {exc}")) from None
        return self.reply(resp, started)

    def reply(self, resp: Any, started: str) -> Reply:
        """A Chat Completions response as calls.jsonl records it. OpenAI's prompt_tokens
        holds the cached and the cache-written tokens; input_tokens here is the rest, as
        Anthropic counts it. A refusal is recorded as the stop reason refusal."""
        choices = getattr(resp, "choices", None) or []
        first = choices[0] if choices else None
        message = getattr(first, "message", None)
        stop = "refusal" if getattr(message, "refusal", None) else getattr(first, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        read = _int(getattr(details, "cached_tokens", 0))
        write = _int(getattr(details, "cache_write_tokens", 0))
        reply = Reply(getattr(message, "content", None) or "", getattr(resp, "model", None) or self.model_id, stop,
                      max(_int(getattr(usage, "prompt_tokens", 0)) - read - write, 0),
                      _int(getattr(usage, "completion_tokens", 0)), _utc_now(), self._sdk, started)
        reply.cache_write, reply.cache_read = write, read
        reply.reasoning_tokens = _int(getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0))
        return reply

    def check(self) -> str:
        """One tiny call that proves the key, the network and the model: the served model's
        id, or GenerateError naming only the error's class."""
        try:
            self._connect()
            req: Dict[str, Any] = {"model": self.model_id,
                                   "messages": [{"role": "user", "content": "Reply with the word ok."}]}
            if self.choice.custom:
                req["max_tokens"] = 16
            else:
                req.update(max_completion_tokens=256, reasoning_effort="low", store=False)
            resp = self._client.chat.completions.create(**req)
        except GenerateError:
            raise
        except Exception as exc:  # the class only: an API error's text can quote the request
            raise GenerateError(type(exc).__name__) from None
        return getattr(resp, "model", None) or self.model_id


def _enum_text(value: Any) -> Optional[str]:
    """An SDK enum (or plain string) as its text: FinishReason.STOP is STOP."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


class GoogleModel:
    """generateContent through Google's official google-genai SDK, imported on the first
    call, on the Gemini Developer API (never Vertex AI). The reply's shape is set by the
    prompt's JSON schema (response_json_schema), the thinking by thinking_level at EFFORT,
    and the SDK's automatic continuation is off, so a call is one request and its tokens
    are the ones it reports.

    The key: the SDK reads GOOGLE_API_KEY or GEMINI_API_KEY (GOOGLE_API_KEY first) unless
    --api-key-env names another variable, and the base URL is pinned to GOOGLE_URL either
    way, whatever GOOGLE_GEMINI_BASE_URL says. As with AnthropicModel, the key is never
    printed, logged or written, and every error raised here has it taken out."""

    def __init__(self, choice: ModelChoice):
        self.choice = choice
        self.model_id = choice.model_id
        self._client = None
        self._key: Optional[str] = None

    def connect(self) -> None:
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        try:
            from google import genai
        except ImportError:
            raise GenerateError('a live run on Google needs the google-genai package: pip install '
                                '"parsica-recall[google]" (the tests and --replay do not)')
        self._sdk = f"google-genai {getattr(genai, '__version__', 'unknown')}"
        if self.choice.api_key_env is not None:
            self._key = _key_from_env(self.choice.api_key_env)
        kwargs: Dict[str, Any] = {"vertexai": False,
                                  "http_options": {"base_url": GOOGLE_URL, "retry_options": {"attempts": 3}}}
        if self._key is not None:
            kwargs["api_key"] = self._key
        try:
            self._client = genai.Client(**kwargs)
        except Exception as exc:  # no GEMINI_API_KEY or GOOGLE_API_KEY in the environment, most often
            raise GenerateError(self._scrub(f"could not create the Google client ({type(exc).__name__}: {exc}); check "
                                            "that GEMINI_API_KEY or GOOGLE_API_KEY is set, or name the key's variable "
                                            "with --api-key-env")) from None

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "[the key]") if self._key else text

    def config(self, prompt: Prompt) -> Dict[str, Any]:
        """The generateContent config of one call."""
        return {"system_instruction": prompt.text, "response_mime_type": "application/json",
                "response_json_schema": prompt.schema, "max_output_tokens": self.choice.max_tokens,
                "thinking_config": {"thinking_level": self.choice.effort.upper()},
                "automatic_continuation": False}

    def call(self, category: str, prompt: Prompt, user: str, request_sha: str) -> Reply:
        if self._client is None:
            self._connect()
        started = _utc_now()
        try:
            resp = self._client.models.generate_content(model=self.model_id, contents=user, config=self.config(prompt))
        except TypeError as exc:  # an SDK too old for this request
            raise GenerateError(self._scrub(f"the google-genai SDK refused the {category} request ({exc}); upgrade it "
                                            "with pip install -U google-genai")) from None
        except Exception as exc:  # an API error or anything else: a message, never a traceback that could hold the key
            raise GenerateError(self._scrub(f"the {category} call failed: {type(exc).__name__}: {exc}")) from None
        return self.reply(resp, started)

    def reply(self, resp: Any, started: str) -> Reply:
        """A generateContent response as calls.jsonl records it: the text parts without the
        thought parts; a prompt Google blocked as the stop reason PROMPT_BLOCKED_<reason>.
        Google's prompt_token_count holds the cached tokens, and the thinking tokens are
        billed as output, so output_tokens is the reply's tokens plus the thoughts."""
        candidates = getattr(resp, "candidates", None) or []
        first = candidates[0] if candidates else None
        blocked = _enum_text(getattr(getattr(resp, "prompt_feedback", None), "block_reason", None))
        if first is None and blocked:
            stop, text = f"PROMPT_BLOCKED_{blocked}", ""
        else:
            stop = _enum_text(getattr(first, "finish_reason", None))
            parts = getattr(getattr(first, "content", None), "parts", None) or []
            text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        usage = getattr(resp, "usage_metadata", None)
        read = _int(getattr(usage, "cached_content_token_count", 0))
        thoughts = _int(getattr(usage, "thoughts_token_count", 0))
        reply = Reply(text, getattr(resp, "model_version", None) or self.model_id, stop,
                      max(_int(getattr(usage, "prompt_token_count", 0)) - read, 0),
                      _int(getattr(usage, "candidates_token_count", 0)) + thoughts, _utc_now(), self._sdk, started)
        reply.cache_read = read
        reply.reasoning_tokens = thoughts
        return reply

    def check(self) -> str:
        """One tiny call that proves the key, the network and the model: the served model's
        id, or GenerateError naming only the error's class."""
        try:
            self._connect()
            resp = self._client.models.generate_content(
                model=self.model_id, contents="Reply with the word ok.",
                config={"max_output_tokens": 256, "thinking_config": {"thinking_level": "LOW"},
                        "automatic_continuation": False})
        except GenerateError:
            raise
        except Exception as exc:  # the class only: an API error's text can quote the request
            raise GenerateError(type(exc).__name__) from None
        return getattr(resp, "model_version", None) or self.model_id


def make_model(choice: ModelChoice) -> Any:
    """The live model for a choice: the Anthropic, OpenAI (OpenAI and Azure), Google or
    Bedrock adapter."""
    if choice.provider in ("openai", "azure"):
        return OpenAIModel(choice)
    if choice.provider == "google":
        return GoogleModel(choice)
    if choice.provider == "bedrock":
        return BedrockModel(choice)
    return AnthropicModel(choice.model_id, choice.api_key_env)


def check_api(model_id: str = DEFAULT_MODEL, api_key_env: Optional[str] = None,
              choice: Optional[ModelChoice] = None) -> str:
    """--check-api: one tiny call. Returns the served model's id; raises GenerateError
    with only the error's class name, never its text and never the key."""
    return make_model(choice or ModelChoice("anthropic", model_id, api_key_env=api_key_env)).check()


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
                                "seed, targets, window, provider, model or prompts differ from the recorded run")
        rec = queue.popleft()
        reply = Reply(rec["response"], rec["served_model"], rec["stop_reason"], rec["input_tokens"],
                      rec["output_tokens"], rec["time"], rec.get("sdk"), rec.get("started"))
        reply.cache_write = rec.get("cache_creation_input_tokens", 0)
        reply.cache_write_1h = rec.get("cache_creation_1h_input_tokens", 0)
        reply.cache_read = rec.get("cache_read_input_tokens", 0)
        reply.reasoning_tokens = rec.get("reasoning_tokens")  # OpenAI's and Google's records only
        reply.in_log = self.live is not None
        return reply


def cost_at(price: Optional[Sequence[float]], uncached_in: int, out: int, cache_write: int = 0, cache_read: int = 0,
            cache_write_1h: int = 0) -> Optional[float]:
    """US dollars at one price row (input, output, cache write, long cache write, cache
    read, per million tokens), unrounded; None with no price."""
    if price is None:
        return None
    p_in, p_out, p_write, p_write_1h, p_read = price
    return (uncached_in * p_in + out * p_out + cache_write * p_write + cache_write_1h * p_write_1h
            + cache_read * p_read) / 1e6


def cost_exact(model_id: str, uncached_in: int, out: int, cache_write: int = 0, cache_read: int = 0,
               cache_write_1h: int = 0) -> Optional[float]:
    """US dollars at the prices in PRICES, unrounded; None for a model not listed.
    cache_write is written with the 5-minute cache, cache_write_1h with the 1-hour cache."""
    return cost_at(PRICES.get(model_id), uncached_in, out, cache_write, cache_read, cache_write_1h)


def cost_usd(model_id: str, uncached_in: int, out: int, cache_write: int, cache_read: int) -> Optional[float]:
    """cost_exact rounded to the cent, as a manifest reports it (5-minute cache writes)."""
    cost = cost_exact(model_id, uncached_in, out, cache_write, cache_read)
    return None if cost is None else round(cost, 2)


def record_cost(rec: Dict[str, Any]) -> Optional[float]:
    """The cost of one recorded call, at the price the record carries (price_usd_per_million,
    set by --price-in and --price-out or held empty for an endpoint of your own), else at
    the table's price for the model it asked for. Its cache writes are 1-hour ones as far
    as cache_creation_1h_input_tokens says, the default ones for the rest."""
    hour = rec.get("cache_creation_1h_input_tokens", 0)
    price = rec["price_usd_per_million"] if "price_usd_per_million" in rec else PRICES.get(rec.get("model"))
    return cost_at(price, rec.get("input_tokens", 0), rec.get("output_tokens", 0),
                   rec.get("cache_creation_input_tokens", 0) - hour, rec.get("cache_read_input_tokens", 0), hour)


def system_blocks(prompt: Prompt) -> Any:
    """The request's system prompt: the prompt file's text as one block marked for the
    cache, or the bare text when SYSTEM_CACHE is None."""
    if SYSTEM_CACHE is None:
        return prompt.text
    return [{"type": "text", "text": prompt.text, "cache_control": dict(SYSTEM_CACHE)}]


def request_sha256(model_id: str, prompt: Prompt, user: str, choice: Optional[ModelChoice] = None) -> str:
    """The hash a call log keys each response by, so --replay and --resume find it again.
    An Anthropic request hashes as it always has; another provider's request adds the
    provider and the endpoint, and its own cap and effort in place of Anthropic's."""
    if choice is not None and choice.other:
        fields: Dict[str, Any] = {"model": model_id, "system_sha256": prompt.sha256, "user": user, **choice.fields()}
    else:
        fields = {"model": model_id, "max_tokens": MAX_TOKENS, "effort": EFFORT,
                  "system_sha256": prompt.sha256, "system_cache": SYSTEM_CACHE, "user": user}
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
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


# Voices (November 23, 2026). With --people, each unit's question is asked in a seeded voice
# and shape, written on an ASK line just before the unit, with the first name of the person
# whose record it is; prompts written for it read that line. Without --people
# none of this runs, and every request is what it was before, byte for byte.
ASK_HEAD = "ASK "
VOICES = (("person", 1), ("assistant", 1))
SHAPES = (("plain", 8), ("casual", 5), ("request", 3), ("terse", 4))
# Role words in a question, which the voices replace with a person or a pronoun.
_ROLE_WORDS = re.compile(r"\bthe\s+(?:user|assistant|agent|ai|individual)\b", re.I)
# The owner asks (November 23, 2026). With --owner beside --people, the store is treated as
# having an owner, the person whose records a unit holds, and every question is that owner's
# own message to their assistant. The ASK line then gives the owner's first name, so the
# writer knows who "I" is, and a seeded shape; prompts written for it read it.
# A question that names its owner is dropped, as one that names a role is; a topic shows only
# its owner's records; and every candidate records how it was asked ("ask"), so a set can be
# scored against its owner's records alone, questions with no gold record included.
OWNER_SHAPES = (("plain", 6), ("casual", 6), ("request", 3), ("fragment", 3), ("thinking", 2))
# --forms: beside its shape, each unit's question gets a form, the kind of question the
# owner asks, and one unit in LONG_ONE_IN is asked as a long message. Both come from streams of their
# own, so a unit keeps the shape --owner gives it unless it becomes long. Abstention is always "open".
OWNER_FORMS = (("open", 7), ("how", 4), ("when", 3), ("choice", 2), ("two-part", 4))
LONG_ONE_IN = 10
LONG_MAX_WORDS = 80
TWO_PART_MAX_WORDS = 35
# --form-scheme 2 (prompts/v7, November 24, 2026), from a read of trial sets drawn by the first scheme.
# The choice form is drawn less often (about 7.5% of units overall), and the side its right answer goes
# on is drawn too and written on the ASK line ("right": "first" or "second"): in those sets the right
# answer came first in 93% of choice questions, and second, after the old value, in every one about
# something that changed, so its place gave it away. The when form goes only to a unit whose shown
# records hold a time, a date, a day or a length of time, and to every temporal_awareness unit, whose
# questions ask when a thing was said; in those sets most when questions fell back to another kind, for
# want of a time to ask about.
# It weighs more where it may go (OWNER_FORMS_2), so it keeps about its share overall on pool v1.
FORM_SCHEMES = (1, 2, 3)
OWNER_FORMS_2 = (("open", 15), ("how", 8), ("when", 9), ("choice", 3), ("two-part", 8))
CHOICE_SIDES = ("first", "second")
# --form-scheme 3 (prompts/v7, November 24, 2026): scheme 2, with the when form kept to the categories
# whose answer can be a time. In a trial run of scheme 2 the draw gave the when form only to
# entity_attribution and vocabulary_gap units, whose answers are a name or a term, and none of the 15 when
# questions asked about a time, though every record held one. So under scheme 3 the when form goes only
# to a unit of WHEN_CATEGORIES whose shown records hold a time (temporal_awareness always), and weighs
# more there (OWNER_FORMS_3), so it keeps about its share on pool v1. Everything else is scheme 2.
OWNER_FORMS_3 = (("open", 15), ("how", 8), ("when", 15), ("choice", 3), ("two-part", 8))
WHEN_CATEGORIES = ("direct_recall_user", "direct_recall_agent", "temporal_awareness", "paraphrase_recall",
                   "cross_session", "long_horizon", "knowledge_update", "correction_handling")
_MONTHS = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
           r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)")
_TIME_IN_TEXT = re.compile(
    r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)(?![a-z])|\b(?:noon|midnight)\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\b|\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
    rf"|\b{_MONTHS}\s+\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
    r"|\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b|\b\d{4}/\d{1,2}/\d{1,2}\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty"
    r"|sixty|ninety|half an?)\s*-?\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?"
    r"|years?)\b",
    re.I)


def has_time(text: str) -> bool:
    """Whether a record's text holds a time of day, a date, a day of the week or a length of time:
    something a when question can ask about (--form-scheme 2). A clock time has to be one (23:59 is,
    99:99 is not); a date written with slashes needs its year, since 1/2 or 3/4 is as often a fraction.
    The record's own timestamp does not count, and neither do words such as "yesterday", which only
    the timestamp could place."""
    return bool(_TIME_IN_TEXT.search(text or ""))


def _weighted_name(rng: random.Random, pairs: Sequence[Tuple[str, int]]) -> str:
    """One name drawn by integer weights, in integer arithmetic, the same on every platform."""
    r = rng.randrange(sum(w for _, w in pairs))
    for name, w in pairs:
        if r < w:
            return name
        r -= w
    raise AssertionError("unreachable")


class Voices:
    """Who asks each unit's question and how. people maps a record's source, or the part of
    it before the first '-' (a pool's persona, p1 to p8 in pool v1), or '*' for every record,
    to a first name. The voice and the shape are drawn from the seed, the category and the
    unit, so the same unit is always asked the same way."""

    def __init__(self, seed: int, people: Dict[str, str], sha256: Optional[str] = None, owner: bool = False,
                 forms: bool = False, force_form: Optional[str] = None, form_scheme: int = 1):
        self.seed = seed
        self.people = dict(people)
        self.sha256 = sha256
        self.owner = owner  # every question is its owner's own message (--owner)
        self.forms = forms  # and each one also gets a form, and now and then the long shape (--forms)
        self.force_form = force_form  # every unit but abstention gets this form (--force-form)
        self.form_scheme = form_scheme  # 2 and 3: the weights, the choice side and the when rule of prompts/v7

    def timed(self, category: str, unit: Sequence[str], corpus: "Corpus") -> bool:
        """Whether the when form may go to this unit: a temporal_awareness unit always, any other when
        a record the writer is shown holds a time (has_time); under --form-scheme 3 only in
        WHEN_CATEGORIES, whose answer can be a time."""
        if category == "temporal_awareness":
            return True
        if self.form_scheme == 3 and category not in WHEN_CATEGORIES:
            return False
        return any(has_time(str(corpus.by_id[rid].get("text") or "")) for rid in self.owned(unit, corpus))

    def _form(self, category: str, unit: Sequence[str], corpus: "Corpus", key: str) -> str:
        if self.force_form:
            return self.force_form
        if self.form_scheme in (2, 3):
            weights, stream = (OWNER_FORMS_3, "form3") if self.form_scheme == 3 else (OWNER_FORMS_2, "form2")
            pairs = (weights if self.timed(category, unit, corpus)
                     else tuple(p for p in weights if p[0] != "when"))
            return _weighted_name(random.Random(f"recall/1.0 {stream}|{key}"), pairs)
        return _weighted_name(random.Random(f"recall/1.0 form|{key}"), OWNER_FORMS)

    def key_of(self, record: Dict[str, Any]) -> Optional[str]:
        """The people file's key a record falls under: its source, the part of it before the
        first '-', or '*'. Records under one key have one owner."""
        src = str(record.get("source") or "")
        return next((key for key in (src, src.split("-", 1)[0], "*") if self.people.get(key)), None)

    def person_of(self, record: Dict[str, Any]) -> Optional[str]:
        src = str(record.get("source") or "")
        return self.people.get(src) or self.people.get(src.split("-", 1)[0]) or self.people.get("*")

    def ask(self, category: str, unit: Sequence[str], corpus: "Corpus") -> Dict[str, str]:
        if self.owner:
            rng = random.Random(f"recall/1.0 owner|{self.seed}|{category}|{'+'.join(unit)}")
            out = {"shape": _weighted_name(rng, OWNER_SHAPES)}
            if self.forms:
                key = f"{self.seed}|{category}|{'+'.join(unit)}"
                out["form"] = "open" if category == "abstention" else self._form(category, unit, corpus, key)
                if self.form_scheme in (2, 3) and out["form"] == "choice":
                    out["right"] = CHOICE_SIDES[random.Random(f"recall/1.0 right|{key}").randrange(len(CHOICE_SIDES))]
                if random.Random(f"recall/1.0 long|{key}").randrange(LONG_ONE_IN) == 0:
                    out["shape"] = "long"
            name = self.person_of(corpus.by_id[unit[0]])
            if name:
                out["owner"] = name
            return out
        rng = random.Random(f"recall/1.0 voices|{self.seed}|{category}|{'+'.join(unit)}")
        out = {"voice": _weighted_name(rng, VOICES), "shape": _weighted_name(rng, SHAPES)}
        name = self.person_of(corpus.by_id[unit[0]])
        if name:
            out["person"] = name
        return out

    def owned(self, unit: Sequence[str], corpus: "Corpus") -> List[str]:
        """The unit's records that belong to its owner, the person of its first record. A
        record or a pair has one owner; a topic can gather records of several."""
        key = self.key_of(corpus.by_id[unit[0]])
        return [rid for rid in unit if self.key_of(corpus.by_id[rid]) == key]

    def record(self, category: str, unit: Sequence[str], corpus: "Corpus") -> Dict[str, Any]:
        """What a candidate keeps of how it was asked: the ASK line's fields and the people
        file's key of its owner."""
        return dict(self.ask(category, unit, corpus), key=self.key_of(corpus.by_id[unit[0]]))

    def describe(self) -> Dict[str, Any]:
        if self.owner and self.forms:
            out = {"people_sha256": self.sha256, "asker": "owner", "shapes": dict(OWNER_SHAPES),
                   "forms": dict(OWNER_FORMS), "long": f"1 in {LONG_ONE_IN}"}
            if self.form_scheme == 2:
                out.update(form_scheme=2, forms=dict(OWNER_FORMS_2), choice_right=list(CHOICE_SIDES),
                           when="only where a shown record holds a time, a date, a day or a length of time, "
                                "and in temporal_awareness")
            elif self.form_scheme == 3:
                out.update(form_scheme=3, forms=dict(OWNER_FORMS_3), choice_right=list(CHOICE_SIDES),
                           when="only in " + ", ".join(WHEN_CATEGORIES) + ", where a shown record holds a time, "
                                "a date, a day or a length of time (temporal_awareness always)")
            if self.force_form:
                out["form_forced"] = self.force_form
            return out
        if self.owner:
            return {"people_sha256": self.sha256, "asker": "owner", "shapes": dict(OWNER_SHAPES)}
        return {"people_sha256": self.sha256, "voices": dict(VOICES), "shapes": dict(SHAPES)}


def read_people(path: str) -> Dict[str, str]:
    """--people: a JSON object from a source (or a source's prefix, or '*') to a first name. A
    name goes to the model on every ASK line and into the call log, so it passes the same screen
    as a record's fields: a value with the shape of a credential or of personal data (an email
    address, a phone number and so on) is refused, and the error names its key, not the value."""
    try:
        with open(path, encoding="utf-8") as f:
            people = json.load(f)
    except (OSError, ValueError) as exc:
        raise GenerateError(f"--people: cannot read {path}: {exc}")
    if (not isinstance(people, dict) or not people
            or not all(isinstance(k, str) and k and isinstance(v, str) and v.strip() for k, v in people.items())):
        raise GenerateError("--people takes a JSON object from sources (or a source's prefix, or '*') to first names")
    for key, name in people.items():
        if lv.scan_secrets_pii(name.strip()):
            raise GenerateError(f"--people: the name for {key!r} has the shape of a credential or personal data; "
                                "it would be sent to the model on every ASK line, so give a first name")
    return {k: v.strip() for k, v in people.items()}


def _ask_line(voices: Optional["Voices"], category: str, unit: Sequence[str], corpus: "Corpus") -> List[str]:
    if voices is None:
        return []
    return [ASK_HEAD + json.dumps(voices.ask(category, unit, corpus), ensure_ascii=False, sort_keys=True)]


def user_message(category: str, batch: Sequence[Tuple[str, ...]], corpus: Corpus,
                 voices: Optional[Voices] = None) -> str:
    """The user message for one call: one record per line, or labeled pairs. With voices,
    an ASK line goes before each unit (a record, a pair or a topic's records). When the
    owner asks, a topic shows only its owner's records."""
    lines = []
    if UNIT_KIND[category] == "pair":
        first, second = PAIR_LABELS[category]
        for i, (a, b) in enumerate(batch, 1):
            lines += [f"PAIR {i}"] + _ask_line(voices, category, (a, b), corpus)
            lines += [f"{first}: {record_json(corpus.by_id[a])}", f"{second}: {record_json(corpus.by_id[b])}"]
    else:
        for unit in batch:
            lines += _ask_line(voices, category, unit, corpus)
            shown = voices.owned(unit, corpus) if voices is not None and voices.owner else unit
            lines += [record_json(corpus.by_id[rid]) for rid in shown]
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


COMMON_MODES = ("conversation", "owner")


def source_of(record: Dict[str, Any]) -> str:
    """A record's source as one string: the loader takes any JSON value there, and a list or
    a number must still compare and hash."""
    value = record.get("source")
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def gold_or_common(category: str, q: Dict[str, Any], corpus: Corpus, voices: Optional[Voices], mode: str,
                   window: int, confirm: Optional[Callable[[str], Optional[bool]]] = None,
                   eligible: Optional[set] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """--common-answers (docs/generation.md section 6): what an item the lint holds only for a
    GOLD_MULTIPLICITY leak may become instead. Returns the question to lint again, or None
    when it stays as it is, and what was found, for candidates.jsonl.

    A record of a gold record's conversation (the same source) that holds the answer or an
    alternate may say the fact again: the owner's turn, the assistant's reply and the note of
    the session are one conversation. It may also only share the words ("the east cabinet's
    hinges squeak" beside "the toolbox is in the east cabinet"), so each such record is put to
    confirm, which asks the writer whether the record, on its own, gives the answer to the
    question. Only a yes makes it gold, with its near copies. A record is put to the question
    only when its whole near-copy cluster could have been shown (eligible: inside the gold
    window and past the screen) and, when the owner asks, every member is that owner's; without
    confirm nothing is promoted. This is not done for the categories whose gold is several
    records by design (the version categories and cross_session), where a new gold record would
    change what the question asks of a system.

    Under mode "owner", an item whose answer is otherwise found only in other owners' records
    is kept and carries common_answer: another person's record cannot be what the owner asks
    about, so the question is sound by record id, and its answer text proves nothing. A record
    of the same owner in another conversation may say the same fact again, and nothing here
    can tell, so that item stays held, and so does one where a record's owner is not known."""
    gold = list(q["gold_ids"])
    held = set()
    for value in [q["answer"]] + list(q["acceptable_answers"]):
        held.update(corpus.ctx.index.matching(rc.normalize(value)))
    others = [corpus.ids[i] for i in sorted(held) if corpus.ids[i] not in gold]
    owned = voices is not None and voices.owner
    mine = voices.key_of(corpus.by_id[gold[0]]) if voices is not None else None
    sources = {source_of(corpus.by_id[g]) for g in gold} - {""}
    grown: List[str] = []
    checked: List[Dict[str, Any]] = []
    if category != "abstention" and category not in rc.MULTI_RECORD_CATEGORIES:
        asked = set()
        for rid in others:
            if source_of(corpus.by_id[rid]) not in sources:
                continue
            members = corpus.cluster[rid]
            if members[0] in asked:
                continue
            asked.add(members[0])
            if any(corpus.pos[m] >= window or (eligible is not None and m not in eligible) for m in members):
                checked.append({"id": rid, "answers": None, "why": "not eligible to be shown"})
                continue
            if owned and (mine is None or any(voices.key_of(corpus.by_id[m]) != mine for m in members)):
                checked.append({"id": rid, "answers": None, "why": "not the owner's record"})
                continue
            verdict = confirm(rid) if confirm is not None else None
            checked.append({"id": rid, "answers": verdict})
            if verdict is True:
                grown += [m for m in members if m not in gold and m not in grown]
    rest = [rid for rid in others if rid not in grown]
    note: Dict[str, Any] = {"gold_added": grown, "checked": checked, "elsewhere": len(rest), "outcome": "held"}
    q2 = None
    if grown:
        q2 = dict(q)
        q2["gold_ids"] = gold + grown
        q2["gold_hashes"] = {g: rc.text_hash(corpus.by_id[g]["text"]) for g in q2["gold_ids"]}
    if not rest:
        note["outcome"] = "gold" if grown else "held"
        return q2, note
    if mode != "owner" or voices is None:
        return None, note
    if mine is None or any(voices.key_of(corpus.by_id[rid]) is None for rid in rest):
        note["outcome"] = "owner_unknown"  # nothing shows the other record is another person's
        return None, note
    same = sum(1 for rid in rest if voices.key_of(corpus.by_id[rid]) == mine)
    note["same_owner"] = same
    if same:
        return None, note
    q2 = dict(q2 or q)
    q2[rc.COMMON_ANSWER] = True
    note["outcome"] = rc.COMMON_ANSWER
    return q2, note


_CHOICE_NEAR = 8  # an option sits within this many words of its "or"
_NOT_AN_OPTION = frozenset({"something", "anything", "not", "so", "else", "whatever", "what", "maybe"})


def _holds(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    return n > 0 and any(list(tokens[i:i + n]) == list(phrase) for i in range(len(tokens) - n + 1))


def choice_side(q: Dict[str, Any]) -> Optional[str]:
    """Where a choice question offers its answer (--forms): "first" when the answer, or an
    acceptable answer, is the option before an "or", "second" when it is the one after, and
    None when the question offers no two options with the answer as one of them, so the
    lint holds an answer written in the question as it always has. "cabinet xqad or something"
    is not a choice: the other side has to be an option too."""
    toks = rc.normalize(q["question"]).split()
    variants = [rc.normalize(v).split() for v in [q["answer"]] + list(q.get("acceptable_answers") or [])]
    for j, tok in enumerate(toks):
        if tok != "or":
            continue
        left, right = toks[max(0, j - _CHOICE_NEAR):j], toks[j + 1:j + 1 + _CHOICE_NEAR]
        if not left or not right or left[-1] in _NOT_AN_OPTION or right[0] in _NOT_AN_OPTION:
            continue
        for v in variants:
            if _holds(left, v) and not _holds(right, v):
                return "first"
            if _holds(right, v) and not _holds(left, v):
                return "second"
    return None


# The promotion check (--common-answers): one short call per record the conversation step would
# make gold, recorded in calls.jsonl under this category, which is not a question category.
CHECK_CATEGORY = "common_check"
CHECK_PROMPT = os.path.join(PROMPTS_DIR, "checks", "common_answer.md")


def check_message(q: Dict[str, Any], record: Dict[str, Any]) -> str:
    """The user message of a promotion check: the question, its answer and the one record."""
    return "\n".join(["QUESTION: " + q["question"], "ANSWER: " + q["answer"], "RECORD: " + record_json(record)])


def check_verdict(text: str) -> bool:
    """A check's reply as a verdict: yes only for exactly {"items": [{"answers": true}]}."""
    try:
        items = json.loads(text).get("items")
    except (ValueError, AttributeError):
        return False
    return isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict) and items[0].get("answers") is True


def retry_message(category: str, item: Dict[str, Any], unit: Tuple[str, ...], corpus: Corpus,
                  conflicts: Sequence[Tuple[str, str, int]], voices: Optional[Voices] = None) -> str:
    """The user message of a GOLD_MULTIPLICITY retry (docs/generation.md section 6): the
    held item as the model returned it, the words that leak and how many other records
    hold them, then the unit exactly as it was first shown. No other record is shown."""
    why = "; ".join(f"the {'answer' if field == 'answer' else 'acceptable answer'} {json.dumps(value, ensure_ascii=False)} "
                    f"also appears in {n} other record{'s' if n != 1 else ''} of the store"
                    for field, value, n in conflicts)
    return "\n".join([RETRY_HEAD, "held: " + json.dumps(item, ensure_ascii=False), "why: " + why,
                      user_message(category, [unit], corpus, voices)])


def parse_retry_message(text: str) -> Tuple[Dict[str, Any], str, List[Any]]:
    """The held item, the reason and the unit of a retry message, for tests and audits."""
    lines = text.split("\n")
    return (json.loads(lines[1].split(": ", 1)[1]), lines[2].split(": ", 1)[1],
            parse_user_message("\n".join(lines[3:])))


def parse_user_message(text: str) -> List[Any]:
    """The inverse of user_message, for tests and audits: a list of records, or of
    (first, second) record pairs for a pair message. A retry message gives the unit it
    shows again. ASK lines (voices) are left out."""
    lines = [line for line in text.split("\n") if not line.startswith(ASK_HEAD)]
    if lines and lines[0] == RETRY_HEAD:
        lines = lines[3:]
    if lines and lines[0].startswith("PAIR "):
        out = []
        for i in range(0, len(lines), 3):
            out.append(tuple(json.loads(line.split(": ", 1)[1]) for line in lines[i + 1:i + 3]))
        return out
    return [json.loads(line) for line in lines if line.strip()]


# The stop reasons that mean the model declined or was stopped by a filter, and those that
# mean the reply ran out of room, in each provider's words (Anthropic, OpenAI, Google). The
# call log keeps the provider's own word; parse_reply reads it through these.
DECLINED_STOPS = {"refusal", "content_filter", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                  "LANGUAGE", "IMAGE_SAFETY", "MODEL_ARMOR"}
CUT_STOPS = {"max_tokens", "length", "MAX_TOKENS", "CONTINUATION"}


def parse_reply(reply: Reply) -> Tuple[List[Any], Optional[str]]:
    stop = reply.stop_reason or ""
    if stop in DECLINED_STOPS or stop.startswith("PROMPT_BLOCKED_"):
        return [], "the model declined the request"
    if stop in CUT_STOPS:
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


def rule_checks(category: str, item: Dict[str, Any], unit: Tuple[str, ...], corpus: Corpus,
                roles: bool = False, owner: Optional[str] = None, ask: Optional[Dict[str, Any]] = None
                ) -> Optional[str]:
    """The prompts' own rules that a machine can check (section 6): the question's shape
    and length, one question per item, the answer's length, a number's unit kept by every
    alternate, and in the pair categories where the answer may sit. With roles (a run with
    voices), a question that names a role ("the user", "the assistant") is dropped too, and
    with owner (the owner's first name, when the owner asks) one that names its owner. With
    ask (--forms), a long message may run to LONG_MAX_WORDS with its question mark anywhere in it,
    and a two-part message to TWO_PART_MAX_WORDS with a question mark at the end of each part;
    both may join a second question to the first.
    Returns the reason an item is dropped, or None."""
    question = item["question"].strip()
    shape, form = (ask or {}).get("shape"), (ask or {}).get("form")
    long_message, two_part = shape == "long", form == "two-part"
    marks = sum(question.count(m) for m in _QUESTION_MARKS)
    if long_message and not marks:
        return "the message holds no question mark"
    if not long_message and not question.endswith(_QUESTION_MARKS):
        return "the question does not end in a question mark"
    if roles:
        role = _ROLE_WORDS.search(question)
        if role:
            return f"the question names a role ({role.group(0).lower()!r}) instead of a person or a pronoun"
    if owner and re.search(rf"(?<![\w-]){re.escape(owner)}(?![\w-])", question, re.I):
        return f"the question names its owner ({owner!r}), who asks as I"
    most = LONG_MAX_WORDS if long_message else TWO_PART_MAX_WORDS if two_part else QUESTION_MAX_WORDS
    if len(question.split()) > most:
        return f"the question is longer than {most} words"
    if not (long_message or two_part) and marks > 1:  # worded as before --forms, so earlier sets replay byte for byte
        return "two questions in one: the item holds more than one question mark"
    if marks > 2:
        return f"the message holds {marks} question marks"
    second = _SECOND_QUESTION.search(question)
    if second and not (long_message or two_part):
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
             cand_id: str, keep: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    """The question as it would freeze, in the lint spec's field order. Gold grows to
    each shown record's near copies, by the rule fixed before any call; difficulty is a
    placeholder until the freeze ranks it. keep, when the owner asks, takes only the near
    copies that are the owner's: the same words in another person's records are not the
    owner's memory."""
    if category == "abstention":
        answer, alts, gold, current, atype = ABSTENTION_ANSWER, list(ABSTENTION_ALTERNATES), [], None, "abstention"
    else:
        gold = []
        for g in item["gold_ids"]:
            gold += [m for m in corpus.cluster[g] if m not in gold and (keep is None or m == g or keep(m))]
        current = None
        if category in rc.VERSION_CATEGORIES:
            copies = [m for m in corpus.cluster[unit[1]] if keep is None or m == unit[1] or keep(m)]
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
                # Per category and per difficulty: n and gold at each k, so a reader sees where
                # the floor sits inside the set, not only its average.
                "by_category": {c: {"n": t["n"], "gold": t["gold"]} for c, t in res["by_category"].items()},
                "by_difficulty": {d: {"n": t["n"], "gold": t["gold"]} for d, t in res["by_difficulty"].items()},
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
    requests and the freeze. --max-cost, --api-key-env and the price flags are not in it, so
    a resume may raise the one, name another variable or price the model again. A provider
    other than Anthropic adds itself, its endpoint, its effort and its output cap."""
    out = {"set": k["qset_version"], "sealed_set": k["sealed_name"], "corpus_sha256": k["corpus_sha"],
           "seed": k["seed"], "targets": {c: k["targets"].get(c, 0) for c in rc.CATEGORIES}, "model": k["model_id"],
           "store_ids": k["store_ids"], "gold_window": k["window"], "core": k["core"], "core_quota": k["core_quota"],
           "unit_caps": dict(sorted(k["unit_caps"].items())), "planted_pairs": k["planted_sha"],
           "split_seed": k["split_seed"], "lint_spec": k["spec"].get("spec_version"),
           "prompts": {c: p.sha256 for c, p in k["prompts"].items()}}
    choice = k.get("choice")
    if choice is not None and choice.other:
        out.update(choice.fields())
    if k.get("prompts_rel") is not None:
        out["prompts_dir"] = k["prompts_rel"]
    if k.get("voices") is not None:
        out["voices"] = k["voices"].describe()
    if k.get("common_answers") is not None:
        out["common_answers"] = k["common_answers"]
    if k.get("check_prompt") is not None:
        out["check_prompt"] = k["check_prompt"].sha256  # the promotion check's prompt, generation/prompts/common_check.md
    return out


def generate_set(corpus_path: str, out: str, seed: int, targets: Dict[str, int], model_id: str = DEFAULT_MODEL,
                 store_ids: bool = False, release: bool = False, model: Any = None, replay: Optional[str] = None,
                 quiet: bool = False, resume: bool = False, max_cost: Optional[float] = None,
                 gold_window: Optional[int] = None, core: Optional[int] = None, split_seed: Optional[int] = None,
                 sealed_out: Optional[str] = None, api_key_env: Optional[str] = None, estimate_only: bool = False,
                 estimate_from: Optional[Any] = None, core_quota: Optional[int] = None,
                 planted_pairs: Optional[str] = None, unit_caps: Optional[Dict[str, int]] = None,
                 choice: Optional[ModelChoice] = None, prompts_dir: Optional[str] = None,
                 people: Optional[str] = None, owner: bool = False,
                 common_answers: Optional[str] = None, forms: bool = False,
                 force_form: Optional[str] = None, form_scheme: Optional[int] = None) -> Dict[str, Any]:
    """Generate, lint and freeze one question set into out (docs/generation.md). choice is
    the provider, model, endpoint and price (ModelChoice; by default Anthropic with
    model_id and api_key_env). model is anything with call(category, prompt, user,
    request_sha) -> Reply; by default the choice's API (with the key from api_key_env when
    given), the recorded responses when replay names a calls.jsonl, or both under resume.
    max_cost stops the run before its next new call once
    the set's cost passes it (CostStop). gold_window limits gold to the corpus's first
    records, core puts the corpus's first core records first (core_quota: at most that
    many items per category from the core, then the rest of the window), and split_seed with
    sealed_out splits the frozen items into a public half (out) and a sealed half
    (sealed_out), which keeps the generation record. planted_pairs names a pool's record of
    its planted changes, which the version categories build their units from; unit_caps
    replaces categories' unit caps. estimate_only prints the estimate and returns before
    any call; estimate_from is a finished set's folder, or a list of them, that measured
    its yields and tokens. prompts_dir takes the prompts from another folder (prompts/v7), and
    people (a --people file) turns on the voices: an ASK line before each unit, and the
    check that drops a question naming a role. owner, with people, makes every question its
    owner's own message: the ASK line gives the owner and a shape, a question
    naming its owner is dropped, a topic shows only its owner's records, and each candidate
    keeps how it was asked. common_answers ("conversation" or "owner", docs/generation.md
    section 6) changes what becomes of an item held only because its answer is in records
    outside gold: a record of the gold record's conversation that holds the answer becomes gold
    too, and under "owner" an item whose answer is otherwise only in other owners' records is
    kept and carries common_answer. forms, with owner, gives each question a form as
    well as a shape (open, how, when, choice, two-part), asks one unit in ten as a long message, lets
    those through the checks they need, and marks a choice question that offers its answer. None of
    these is recorded when it is not given, so a set built without them replays as before.
    force_form, with forms, gives every unit that form instead of a drawn one (abstention stays
    open): a set that tries one form on purpose. form_scheme 2, with forms (prompts/v7), draws the
    forms by OWNER_FORMS_2, gives the when form only to a unit whose shown records hold a time, and
    draws the side of a choice question's right answer onto the ASK line; 3 is 2 with the when form
    kept to WHEN_CATEGORIES and weighed by OWNER_FORMS_3; 1, or not given, is the first scheme, and a
    set built with any of them replays byte for byte."""
    say = (lambda *a, **k: None) if quiet else print
    if choice is None:
        choice = ModelChoice("anthropic", model_id, api_key_env=api_key_env)
    model_id = choice.model_id
    split = split_seed is not None
    if split != (sealed_out is not None):
        raise GenerateError("--split-seed and --sealed-out go together: the split needs a folder for its sealed half")
    if replay and resume:
        raise GenerateError("--replay builds a new set from a recorded run and --resume continues a stopped one; "
                            "give one of them")
    if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
        raise GenerateError("--max-cost takes a number of US dollars, 0 or more")
    if max_cost is not None and choice.price is None:
        if choice.custom:
            raise GenerateError("--max-cost: an endpoint of your own (--base-url) has no price on file, so the guard "
                                "cannot count its cost; give its prices with --price-in and --price-out (0 and 0 for "
                                "a model that costs nothing to run)")
        if choice.provider in ("azure", "bedrock"):
            platform = "Azure" if choice.provider == "azure" else "Bedrock"
            raise GenerateError(f"--max-cost: {platform} sets its own prices by deployment and region, and none is on "
                                "file, so the guard cannot count the cost; give your prices with --price-in and "
                                "--price-out")
        raise GenerateError(f"--max-cost: no price on file for {model_id} (PRICES), so the guard cannot count its cost; "
                            "give its prices with --price-in and --price-out")
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
    prompts_rel = None
    if prompts_dir is not None:
        if not os.path.isdir(prompts_dir):
            raise GenerateError(f"--prompts: {prompts_dir} is not a folder")
        here = os.path.realpath(prompts_dir)
        root = os.path.realpath(REPO_DIR)
        prompts_rel = (os.path.relpath(here, root).replace(os.sep, "/") if here.startswith(root + os.sep)
                       else os.path.basename(here.rstrip(os.sep)))
        prompts = load_prompts(used, prompts_dir)
    else:
        prompts = load_prompts(used)
    # The promotion check's prompt ships with the builder; --prompts does not move it.
    check_prompt = Prompt(CHECK_CATEGORY, CHECK_PROMPT) if common_answers is not None else None
    if owner and people is None:
        raise GenerateError("--owner needs --people: the file that says whose records are whose")
    if forms and not (owner and people is not None):
        raise GenerateError("--forms needs --people and --owner: a form is part of how the owner asks")
    if force_form is not None and (not forms or force_form not in dict(OWNER_FORMS)):
        raise GenerateError(f"--force-form {force_form}: needs --forms, and one of {', '.join(dict(OWNER_FORMS))}")
    if form_scheme is not None and (isinstance(form_scheme, bool) or form_scheme not in FORM_SCHEMES or not forms):
        raise GenerateError(f"--form-scheme {form_scheme}: needs --forms, and one of "
                            f"{', '.join(str(s) for s in FORM_SCHEMES)}")
    if common_answers is not None and common_answers not in COMMON_MODES:
        raise GenerateError(f"--common-answers {common_answers}: one of {', '.join(COMMON_MODES)}")
    if common_answers == "owner" and not (owner and people is not None):
        raise GenerateError("--common-answers owner needs --people and --owner: it keeps an item whose answer is "
                            "otherwise only in other owners' records, so it has to know whose records are whose")
    voices = (Voices(seed, read_people(people), rc.file_sha256(people), owner=owner, forms=forms,
                     force_form=force_form, form_scheme=form_scheme or 1) if people is not None else None)
    owner_asks = voices is not None and voices.owner

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
    if owner_asks:
        # When the owner asks, a pair is one owner's: a pair across two people would make one person's
        # question need another person's record. The order is kept, and the core's count with it.
        for plan in plans.values():
            if plan.kind == "pair":
                ok = [voices.key_of(corpus.by_id[a]) is not None
                      and voices.key_of(corpus.by_id[a]) == voices.key_of(corpus.by_id[b]) for a, b in plan.units]
                if plan.n_core is not None:
                    plan.n_core = sum(ok[:plan.n_core])
                plan.units = [u for u, k in zip(plan.units, ok) if k]
    eligible_ids = {m for r in reps for m in corpus.cluster[r]}  # what may be shown: whole clusters only
    states = {c: _Category(plans[c], targets.get(c, 0), core_quota) for c in rc.CATEGORIES}
    settings = _settings(qset_version=qset_version, sealed_name=sealed_name, corpus_sha=corpus_sha, seed=seed,
                         targets=targets, model_id=model_id, store_ids=store_ids, window=window, core=core,
                         core_quota=core_quota, split_seed=split_seed, spec=spec, prompts=prompts,
                         unit_caps=unit_caps, planted_sha=planted_sha, choice=choice, prompts_rel=prompts_rel,
                         voices=voices, common_answers=common_answers, check_prompt=check_prompt)
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
        est = estimate(plans, targets, corpus, prompts, model_id, recorded, calibration, core_quota, sample_cap,
                       choice=choice, voices=voices)
        print_estimate(est, say)
        return {"estimate": est, "out": out_dir, "qset_version": qset_version}

    if model is None:
        if replay:
            model = ReplayModel(replay)
        else:
            model = make_model(choice)
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

    def do_call(cat: str, prompt: Prompt, user: str, units: Sequence[Tuple[str, ...]], retry_of: Optional[int],
                check_of: Optional[int] = None):
        """One recorded call. check_of names the candidate a promotion check is for: it is
        recorded under CHECK_CATEGORY and is not a unit of any category's sample."""
        nonlocal out_ready
        req = request_sha256(model_id, prompt, user, choice)
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
            if check_prompt is not None:
                with open(os.path.join(work_dir, "generation", "prompts", f"{CHECK_CATEGORY}.md"), "wb") as f:
                    f.write(check_prompt.data)
            _dump(run_path, settings)
            open(calls_path, "w", encoding="utf-8", newline="\n").close()
            out_ready = True
        # An Anthropic call at the table's price is recorded exactly as before; another
        # provider adds itself and its endpoint, a reasoning model its reasoning tokens, and
        # a price from the flags (or an endpoint of your own) the price it was counted at.
        provider = {"provider": choice.provider, "endpoint": choice.endpoint} if choice.other else {}
        reasoning = {} if reply.reasoning_tokens is None else {"reasoning_tokens": reply.reasoning_tokens}
        priced_at = ({"price_usd_per_million": list(choice.price) if choice.price else None}
                     if choice.priced_record() else {})
        check = {} if check_of is None else {"check_of": check_of}
        call = {"n": len(calls) + 1, "category": cat, "retry_of": retry_of, **check, "units": [list(u) for u in units],
                "request_sha256": req, "model": model_id, **provider, "served_model": reply.model,
                "stop_reason": reply.stop_reason, "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens, **reasoning,
                "cache_creation_input_tokens": reply.cache_write,
                "cache_creation_1h_input_tokens": reply.cache_write_1h, "cache_read_input_tokens": reply.cache_read,
                **priced_at, "started": reply.started, "time": reply.time, "sdk": reply.sdk, "user": user,
                "response": reply.text}
        line = json.dumps(call, ensure_ascii=False)
        if not reply.in_log:
            with open(calls_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        cost = record_cost(call) or 0.0
        spent[0] += cost
        if not reply.in_log:
            spent[1] += cost
        calls.append(dict(call, new=not reply.in_log, line=line))
        if check_of is None:
            entry = {"call": call["n"], "units": [list(u) for u in units]}
            if retry_of is not None:
                entry["retry_of"] = retry_of
            sample[cat].append(entry)
        return call, reply

    def confirm_for(cand: Dict[str, Any], q: Dict[str, Any]) -> Callable[[str], Optional[bool]]:
        """The promotion check for one candidate: asks the writer whether a record gives the
        answer to the question (docs/generation.md section 6). The record counts as shown, so
        a check never takes the run past the sample cap; when it would, nothing is asked."""
        def confirm(rid: str) -> Optional[bool]:
            rep = corpus.cluster[rid][0]
            if rep not in shown and len(shown) >= sample_cap:
                return None
            shown.add(rep)
            _, reply = do_call(CHECK_CATEGORY, check_prompt, check_message(q, corpus.by_id[rid]), [(rid,)], None,
                               check_of=cand["n"])
            return check_verdict(reply.text)
        return confirm

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
                if owner_asks:
                    cand["ask"] = voices.record(cat, unit, corpus)
                per_unit[unit] += 1
                if per_unit[unit] > ITEMS_PER_UNIT[st.plan.kind]:
                    reason = f"more than {ITEMS_PER_UNIT[st.plan.kind]} item(s) for one {st.plan.kind}"
                elif rc.normalize(item["question"]) in frozen_questions:  # the oracle looks questions up by text
                    reason = "the same question text as a frozen item"
                elif held is not None and padded_retry(item, held):
                    reason = f"{PADDED_RETRY}: the answer only adds words around the held answer"
                else:
                    reason = rule_checks(cat, item, unit, corpus, roles=voices is not None,
                                         owner=cand["ask"].get("owner") if owner_asks else None,
                                         ask=cand["ask"] if owner_asks and voices.forms else None)
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
            owner_key = cand["ask"].get("key") if owner_asks else None
            keep = (lambda m: voices.key_of(corpus.by_id[m]) == owner_key) if owner_asks else None
            q = assemble(cat, item, unit, corpus, spec, f"cand_{cand['n']:05d}", keep=keep)
            if owner_asks and q["gold_ids"]:
                # One owner's question needs that owner's records only, the current version included.
                keys = {voices.key_of(corpus.by_id[g]) for g in q["gold_ids"] + rc.current_ids(q.get("must_rank_first"))}
                if owner_key is None or keys != {owner_key}:
                    st.settle(cand, "dropped_by_checks", "its gold records are not all its owner's")
                    continue
            if owner_asks and voices.forms and cand["ask"].get("form") == "choice" and q["gold_ids"]:
                side = choice_side(q)
                if side is not None:  # it offers its answer as one of two options joined by "or"
                    q[rc.CHOICE] = True
                    cand["choice_side"] = side
            if q["gold_ids"]:
                cand["stage"] = stage_of(q["gold_ids"])
            if cat in rc.ALL_GOLD_CATEGORIES and len(q["gold_ids"]) > band_window:
                st.settle(cand, "dropped_by_checks", f"its {len(q['gold_ids'])} gold records, near copies "
                                                     f"included, cannot all fit the window of {band_window}")
                continue
            res = lv.lint_item(q, cand["n"], corpus.ctx)
            if common_answers is not None and q["gold_ids"] and not res["publishable"] and retryable(res["findings"]):
                q2, cand["common"] = gold_or_common(cat, q, corpus, voices, common_answers, window,
                                                    confirm=confirm_for(cand, q), eligible=eligible_ids)
                res2 = lv.lint_item(q2, cand["n"], corpus.ctx) if q2 is not None else None
                if res2 is not None and res2["publishable"]:
                    q, res = q2, res2
                    cand["stage"] = stage_of(q["gold_ids"])
                else:
                    cand["common"]["outcome"] = "held"  # another rule holds the changed item: it stays as written
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
            call, reply = do_call(cat, prompt, user_message(cat, batch, corpus, voices), batch, None)
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
                retry, reply = do_call(cat, prompt, retry_message(cat, item, unit, corpus, conflicts, voices),
                                       [unit], cand["n"])
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

    common = dict(store_ids=store_ids, corpus=corpus, targets=targets, model_id=model_id, choice=choice, calls=calls,
                  prompts=prompts, seed=seed, shown=shown, reps=reps, why=why, bands=bands,
                  eligible_records=eligible_records, sample_cap=sample_cap, sample_path=sample_path,
                  states=states, calls_path=calls_path, cands_path=cands_path, run_path=run_path, spec=spec,
                  window=window, core=core, core_quota=core_quota, split_seed=split_seed, frozen=frozen,
                  unit_caps=unit_caps, planted=planted_info, rare=rare, prompts_rel=prompts_rel, voices=voices,
                  common_answers=common_answers)
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
            "cost_usd": round(math.fsum(record_cost(c) or 0.0 for c in calls), 2) if choice.price is not None else None}


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
    choice = k.get("choice")
    priced = choice.price is not None if choice is not None else k["model_id"] in PRICES
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
    model_info: Dict[str, Any] = {
        "requested": k["model_id"], "served": served, "effort": EFFORT, "max_tokens": MAX_TOKENS,
        "system_cache": SYSTEM_CACHE,
        "server_side_fallbacks": False, "sdk": next((c["sdk"] for c in calls if c["sdk"]), None),
        "started": calls[0]["started"] if calls else None, "finished": calls[-1]["time"] if calls else None,
        "calls": len(calls), "retry_calls": sum(1 for c in calls if c.get("retry_of") is not None),
        **({"check_calls": checks} if (checks := sum(1 for c in calls if c.get("check_of") is not None)) else {}),
        "input_tokens": cost_of("input_tokens"), "output_tokens": cost_of("output_tokens"),
        "cache_creation_input_tokens": cost_of("cache_creation_input_tokens"),
        "cache_creation_1h_input_tokens": sum(c.get("cache_creation_1h_input_tokens", 0) for c in calls),
        "cache_read_input_tokens": cost_of("cache_read_input_tokens"),
        "cost_usd": round(total, 2) if priced else None,
        "cost_note": "from the call log's token counts at the prices in PRICES; the bill is the authority"}
    if choice is not None and choice.other and choice.claude_request:
        # Claude on Bedrock: Anthropic's request, so Anthropic's cache marking and no
        # server-side fallback, with the provider and the endpoint it went to.
        model_info = {"provider": choice.provider, "endpoint": choice.endpoint, **model_info,
                      "effort": choice.effort, "max_tokens": choice.max_tokens, "system_cache": SYSTEM_CACHE,
                      "server_side_fallbacks": False}
    elif choice is not None and choice.other:
        # Another provider's own cap and effort, its endpoint, and no Anthropic cache marking
        # (OpenAI, Azure and Google cache on their own; the tokens they read from it are counted).
        model_info = {"provider": choice.provider, "endpoint": choice.endpoint, **model_info,
                      "effort": choice.effort, "max_tokens": choice.max_tokens, "system_cache": None,
                      "server_side_fallbacks": None,
                      "reasoning_tokens": sum(c.get("reasoning_tokens", 0) for c in calls)}
    if choice is not None and (choice.other or choice.flag_price):
        model_info["prices"] = choice.prices()
        model_info["cost_note"] = (f"from the call log's token counts at {choice.price_source}; the bill is the "
                                   "authority" if choice.price_source else "no price on file: token counts only")
    m.update({
        "targets": {c: k["targets"].get(c, 0) for c in rc.CATEGORIES},
        "model": model_info,
        "prompts": {c: {"file": f"{k.get('prompts_rel') or 'prompts'}/{c}.md", "sha256": p.sha256}
                    for c, p in k["prompts"].items()},
        "sampler": sampler,
        "generation": generation,
    })
    if k.get("voices") is not None:
        m["voices"] = k["voices"].describe()
    if k.get("common_answers") is not None:
        kept = [c for c in k["cands"] if c.get("question_id") and c.get("common")]
        m["common_answers"] = {
            "mode": k["common_answers"],
            "gold_grown": sum(1 for c in kept if c["common"]["gold_added"]),
            "records_added": sum(len(c["common"]["gold_added"]) for c in kept),
            rc.COMMON_ANSWER: sum(1 for c in kept if c["common"]["outcome"] == rc.COMMON_ANSWER),
            "note": "gold_grown questions took the records of the gold record's conversation that hold the answer "
                    "as gold; common_answer questions are scored by record id alone and are not in the exact tier",
        }
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
             core_quota: Optional[int] = None, sample_cap: Optional[int] = None,
             choice: Optional[ModelChoice] = None, voices: Optional[Voices] = None) -> Dict[str, Any]:
    """The questions, calls, tokens and cost a run is likely to come to, before any call
    (section 16). Input tokens come from the planned messages themselves, with each
    category's system prompt written to the cache on its first call and read from it after
    (SYSTEM_CACHE); for OpenAI and Google, which cache on their own terms, every prompt is
    counted uncached, which can only overstate. The price is the choice's (the table's, the
    flags', or none). Yields, output tokens and retries are the EST_* figures, or what a
    finished pilot measured (calibration), stage by stage. Expected: each category walks its
    core stage until the core quota, its target or the core's units, then the rest of the
    window until its target or the unit cap, at each stage's yield; a category the cap
    leaves short of its target is marked. Worst case: every category runs to its unit cap,
    at the same tokens and retries per call."""
    cal = (calibration or {}).get("categories", {})
    ratio = (calibration or {}).get("chars_per_token") or EST_CHARS_PER_TOKEN
    price = choice.price if choice is not None else PRICES.get(model_id)
    cache = None if choice is not None and not choice.claude_request else SYSTEM_CACHE
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
                chars = sum(len(user_message(cat, c, corpus, voices)) for c in chunks)
                user_chars += chars + more * (chars / max(n, 1) + 400)  # a retry shows one unit and the held item
                out_tokens += len(chunks) * r[stage]["output_per_call"] + more * r[stage]["output_per_retry"]
                calls, retries = calls + len(chunks), retries + more
            system = len(prompts[cat].text) / ratio
            sent = calls + retries
            if cache is not None and sent:
                uncached, write, read = user_chars / ratio, system, system * (sent - 1)
            else:
                uncached, write, read = user_chars / ratio + system * sent, 0.0, 0.0
            hour = write if (cache or {}).get("ttl") == "1h" else 0.0
            return {"calls": calls, "retries": retries, "input_tokens": int(uncached), "cache_write_tokens": int(write),
                    "cache_read_tokens": int(read), "output_tokens": int(out_tokens),
                    "cost_usd": cost_at(price, int(uncached), int(out_tokens), int(write - hour), int(read),
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
           "cached": None if cache is None else cache.get("ttl", "5m"), "chars_per_token": ratio,
           "calibration": (calibration or {}).get("from"),
           "records_shown": {"expected": len(shown_expected), "worst": len(shown_worst), "sample_cap": sample_cap}}
    if choice is not None and choice.other:
        out["provider"] = choice.provider
        out["endpoint"] = choice.endpoint
    if choice is not None and choice.flag_price:
        out["price_source"] = choice.price_source
    if recorded is not None:
        out["recorded"] = {"calls": len(recorded.good_lines), "cost_usd": recorded.cost}
    return out


def print_estimate(est: Dict[str, Any], say: Callable[..., None]) -> None:
    money = lambda v: "n/a" if v is None else f"${v:.2f}"  # noqa: E731
    measured = est["calibration"]
    if isinstance(measured, list):
        measured = ", ".join(measured)
    say(f"Estimate for {est['model']}" + (f" on {est['provider']}" if "provider" in est else "") + ", before any call ("
        + (f"yields and tokens measured by {measured}" if measured else
           "rough: yields and output tokens are the EST_* figures until a pilot measures them")
        + f"; {est['chars_per_token']:.2f} characters per input token; system prompts "
        + ({"1h": "cached for an hour", "5m": "cached for five minutes"}.get(est["cached"], "cached")
           if est["cached"] else "counted uncached" if "provider" in est else "not cached")
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
    if "provider" in est and est["provider"] == "bedrock":
        say(f"  (bedrock at {est['endpoint']}: Claude with Anthropic's own request, so the guesses and the cache "
            "apply as they do on Anthropic's API.)")
    elif "provider" in est:
        say(f"  ({est['provider']} at {est['endpoint']}: the guessed tokens per call were measured on Claude, and "
            "another model counts tokens its own way and spends output tokens reasoning; every prompt is counted "
            "uncached. Before a full run, build a small pilot set and pass its folder with --estimate-from.)")
    if est.get("price_source"):
        say(f"  (priced at {est['price_source']})")
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
    where = f" ({model['provider']})" if model.get("provider") else ""
    # Input is the part neither written to nor read from the cache, as every provider's
    # call log counts it, so the cache tokens are printed beside it.
    reasoning = (f" ({model['reasoning_tokens']:,} of them reasoning)" if model.get("reasoning_tokens") is not None
                 else "")
    print(f"  model{where}: {model['calls']} calls ({model['retry_calls']} retries), {model['input_tokens']:,} input, "
          f"{model['cache_creation_input_tokens']:,} written to and {model['cache_read_input_tokens']:,} read from the "
          f"cache, {model['output_tokens']:,} output tokens{reasoning}, {cost}; {model['started']} to "
          f"{model['finished']} (UTC)")
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
    p.add_argument("--provider", default="anthropic", choices=PROVIDERS,
                   help="Who runs the model: anthropic (the default), openai, google, azure (your Azure resource, "
                        "with --base-url and --model) or bedrock (Claude on Amazon Bedrock, with a region). openai "
                        "with --base-url is any endpoint that speaks OpenAI's chat format")
    p.add_argument("--model", default=None,
                   help="Generation model (default: " + ", ".join(f"{name} on {who}" for who, name in DEFAULT_MODELS.items())
                        + "; required with --base-url, and on azure, where it names your deployment)")
    p.add_argument("--aws-region", default=None, metavar="REGION",
                   help="With --provider bedrock: the AWS region whose Bedrock runs the calls, such as us-east-1 "
                        "(default: AWS_REGION in the environment). Recorded in the set as Bedrock's address there")
    p.add_argument("--base-url", default=None, metavar="URL",
                   help="With --provider openai: send the calls to this endpoint instead of OpenAI's, such as "
                        "http://localhost:11434/v1 for a local Ollama server. It gets a key only from --api-key-env, "
                        "never OPENAI_API_KEY. With --provider azure: your Azure resource's v1 address, such as "
                        "https://NAME.openai.azure.com/openai/v1. Recorded in the set")
    p.add_argument("--price-in", type=float, default=None, metavar="USD",
                   help="With --price-out: the model's input price in US dollars per million tokens, in place of the "
                        "price table (which can go stale) or for a model it does not list, so --max-cost can count. "
                        "Cache reads and writes are counted at this price too. Recorded in the set")
    p.add_argument("--price-out", type=float, default=None, metavar="USD",
                   help="With --price-in: the model's output price in US dollars per million tokens")
    p.add_argument("--max-output-tokens", type=int, default=None, metavar="N",
                   help="With --base-url: the most tokens a reply may use, sent as max_tokens (by default the endpoint "
                        "decides)")
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
                   help="Read the API key from this one environment variable instead of the provider's own "
                        "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY; AZURE_OPENAI_API_KEY on "
                        "azure), and send every call to the provider's official endpoint whatever its base-URL variable "
                        "says. With --base-url on openai, the only way to give that endpoint a key. On bedrock, a "
                        "Bedrock API key, used before AWS_BEARER_TOKEN_BEDROCK and your AWS credentials. Not recorded")
    p.add_argument("--prompts", default=None, metavar="DIR",
                   help="Take the category prompts from this folder (such as prompts/v7) instead of prompts/. Its "
                        "path and every prompt's SHA-256 are recorded")
    p.add_argument("--people", default=None, metavar="FILE",
                   help="Turn on the voices: a JSON object from a record's source (or the part before its first '-', "
                        "or '*') to the first name of the person whose records they are. Each unit's question is "
                        "then asked in a seeded voice and shape, written on an ASK line, and a question that names a "
                        "role (the user, the assistant) is dropped. Needs prompts that read the ASK line")
    p.add_argument("--owner", action="store_true",
                   help="With --people: treat the store as its owner's. Every question is the owner's own message to "
                        "their assistant (I, my, you), in a seeded shape; the ASK line gives the owner's first name; "
                        "a question that names its owner is dropped; a topic shows only its owner's records; and each "
                        "candidate records its owner. prompts/v7 is written for it, with --forms")
    p.add_argument("--forms", action="store_true",
                   help="With --people and --owner: each question also gets a form on its ASK line (open, how, "
                        "when, choice, two-part), one unit in ten is asked as a long message, and the checks let "
                        "each through what it needs. prompts/v7 is written for it, with --form-scheme 2 or 3")
    p.add_argument("--force-form", choices=[name for name, _ in OWNER_FORMS], default=None,
                   help="With --forms: every unit but abstention gets this form instead of a drawn one, to try one "
                        "form on purpose")
    p.add_argument("--form-scheme", type=int, choices=list(FORM_SCHEMES), default=None,
                   help="With --forms: 2 draws the forms by OWNER_FORMS_2 (choice about 7.5%%), gives the "
                        "when form only to a temporal_awareness unit or a unit whose shown records hold a time, "
                        "a date, a day or a length of time, and draws the side a choice question's right answer "
                        "goes on onto the ASK line (right: first or second). 3 is 2 with the when form kept to "
                        "the categories whose answer can be a time (WHEN_CATEGORIES; temporal_awareness always). "
                        "1, or not given, is the first scheme. prompts/v7 is written for 2 and 3")
    p.add_argument("--common-answers", choices=COMMON_MODES, default=None,
                   help="What becomes of an item held only because its answer is also in records outside gold. "
                        "conversation: a record of the gold record's conversation (the same source) that holds the "
                        "answer is gold too. owner (with --people and --owner): that, and an item whose answer is "
                        "otherwise only in other owners' records is kept, carries common_answer and is scored by "
                        "record id alone. Not set: such an item is held and retried, as before")
    args = p.parse_args(argv)

    def choose() -> ModelChoice:
        return ModelChoice(args.provider, args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                           price_in=args.price_in, price_out=args.price_out, max_output_tokens=args.max_output_tokens,
                           aws_region=args.aws_region)

    if args.check_api:
        try:
            served = check_api(choice=choose())
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
        choice = choose()
        if not (args.replay or args.estimate) and args.max_cost is None:
            raise GenerateError("a live run needs --max-cost USD, the most the set may cost; run with --estimate "
                                "first to see what it is likely to cost")
        result = generate_set(args.corpus, args.out, args.seed, targets, model_id=choice.model_id, choice=choice,
                              store_ids=args.store_ids, release=args.release, replay=args.replay,
                              resume=args.resume, max_cost=args.max_cost, gold_window=args.gold_window,
                              core=args.core, split_seed=args.split_seed, sealed_out=args.sealed_out,
                              api_key_env=args.api_key_env, estimate_only=args.estimate,
                              estimate_from=args.estimate_from, core_quota=args.core_quota,
                              planted_pairs=args.planted_pairs, unit_caps=unit_caps, prompts_dir=args.prompts,
                              people=args.people, owner=args.owner, common_answers=args.common_answers,
                              forms=args.forms, force_form=args.force_form, form_scheme=args.form_scheme)
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

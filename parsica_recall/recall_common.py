"""Shared definitions for Parsica Recall.

The runner, the linter and the question-set builder all import from here, so the
string the linter checks is the string the scorer matches (spec section 5), and
every chance and ceiling number comes from one implementation.

Contents: version strings, the one normalizer, record ids and text hashes, the
token-boundary phrase match, token overlap, the function-word list used by the
key_edges rule, the severity and status vocabulary, the band schedule, and the
G-aware chance model.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

SPEC_VERSION = "recall/1.0"
HARNESS_VERSION = "1.0.0b2"
CURRENT_QSET = "v11"          # since October 22, 2026
SEALED_SUFFIX = "-sealed"     # a split set's sealed half is named after its public half: v11 and v11-sealed


def is_current_qset(version) -> bool:
    """Whether a question-set version is the harness's current set: the set itself, or its
    sealed half, which the generator names with SEALED_SUFFIX (docs/generation.md section 9).
    The two halves are one set, split by seed, so GATE_VERSIONS_MATCH takes either."""
    return version in (CURRENT_QSET, CURRENT_QSET + SEALED_SUFFIX)


# ---------------------------------------------------------------------------
# Where the files that ship with the harness are
# ---------------------------------------------------------------------------

PACKAGE = "parsica_recall"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _root_dir() -> str:
    """The folder that holds the current question set (data/), the prompts, the persona
    files, the lint spec and the README. In a checkout that is the repository's root, one
    level above this package. An installed copy carries the same files, in the same
    layout, inside the package folder."""
    if os.path.isfile(os.path.join(PKG_DIR, "data", CURRENT_QSET, "questions.json")):
        return PKG_DIR
    return os.path.dirname(PKG_DIR)


ROOT_DIR = _root_dir()
IS_CHECKOUT = ROOT_DIR != PKG_DIR
LINT_SPEC = "parsica_recall_lint_spec.json"

_SOURCE_SHA256: Optional[str] = None


def harness_source_sha256() -> str:
    """One hash for the harness's own code: the SHA-256 of every .py file of this package
    and of the lint spec, each by name, with line endings read as LF. A checkout on any
    operating system and an installed copy of the same release give the same hash, so it
    pins a result to its harness where there is no git commit to pin it to (an installed
    copy), and says so when the code was changed (spec section 8)."""
    global _SOURCE_SHA256
    if _SOURCE_SHA256 is None:
        files = [(name, os.path.join(PKG_DIR, name)) for name in sorted(os.listdir(PKG_DIR)) if name.endswith(".py")]
        files.append((LINT_SPEC, os.path.join(ROOT_DIR, LINT_SPEC)))
        outer = hashlib.sha256()
        for name, path in files:
            try:
                with open(path, "rb") as f:
                    digest = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
            except OSError:
                digest = "missing"
            outer.update(f"{name}\0{digest}\n".encode("utf-8"))
        _SOURCE_SHA256 = "sha256:" + outer.hexdigest()
    return _SOURCE_SHA256


def pin_hash_seed(module: str, argv: Sequence[str]) -> Optional[int]:
    """Determinism: Python's per-process string-hash randomization can leak into an
    adapter's tie-breaking. When the hash seed is not pinned yet, this pins it, runs the
    same command again in a child process (python -m module argv) and returns the child's
    exit status. It returns None when the seed is already pinned, and the caller goes on.
    So every run is bit-identical without the caller setting an environment variable."""
    if os.environ.get("PYTHONHASHSEED") == "0":
        return None
    os.environ["PYTHONHASHSEED"] = "0"
    env = dict(os.environ)
    if IS_CHECKOUT:  # the child must find this same checkout, wherever it was started from
        env["PYTHONPATH"] = os.pathsep.join([ROOT_DIR] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return subprocess.call([sys.executable, "-m", module] + list(argv), env=env)


# ---------------------------------------------------------------------------
# Normalization, ids, hashes
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def normalize(text) -> str:
    """The one normalizer: lowercase, strip everything that is not a letter, digit
    or space, collapse whitespace.

    Text is put in Unicode NFKC form first so the same visible text always gets the
    same id (a composed and a decomposed "e" with an accent, full-width digits).
    Letters are any Unicode letter, digits are decimal digits, so text in other
    scripts keeps its words instead of normalizing to nothing. Punctuation is
    removed, not replaced: "J-channel" becomes "jchannel" on both sides of a match.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(" ")
        else:
            cat = unicodedata.category(ch)
            if cat[0] == "L" or cat == "Nd":
                kept.append(ch)
    return _WHITESPACE.sub(" ", "".join(kept)).strip()


def tokenize(text) -> List[str]:
    """Normalized tokens."""
    norm = normalize(text)
    return norm.split() if norm else []


def record_id(text) -> str:
    """Record id: SHA-256 of the normalized text, first 16 hex characters."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    """Pins the exact text (not normalized), so any edit to a record shows as drift."""
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def has_ts(value) -> bool:
    """Whether a record's ts is a real timestamp: a non-blank string that is not a null
    spelling, or a positive finite number (epoch seconds). 0, NaN, booleans, blanks
    and containers do not count. File mode and live mode use the same test."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value) and value > 0
    if isinstance(value, str):
        v = value.strip().lower()
        return bool(v) and v not in ("none", "null", "nan", "nat", "0")
    return False


def parse_ts(value) -> Optional[datetime]:
    """A ts as an aware UTC datetime: an ISO 8601 string (a trailing Z is UTC, no offset
    is read as UTC) or epoch seconds (milliseconds when the number is too large to be
    seconds). None when it is not a real timestamp or cannot be read."""
    if not has_ts(value):
        return None
    try:
        if isinstance(value, (int, float)):
            seconds = value / 1000.0 if value > 1e11 else float(value)
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        text = value.strip()
        if text[-1:] in ("Z", "z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


# Record age buckets for the live-mode age breakdown (spec section 7): name, last day.
AGE_BUCKETS = (("0-7d", 7), ("8-30d", 30), ("31-90d", 90), ("91-365d", 365), ("over a year", None))
AGE_UNKNOWN = "unknown"


def age_bucket(value, now: datetime) -> str:
    """The age bucket of a record whose ts is value, at the moment now. A ts in the
    future (clock skew) counts as new; no readable ts is unknown."""
    when = parse_ts(value)
    if when is None:
        return AGE_UNKNOWN
    days = max(0, (now - when).days)
    for name, last_day in AGE_BUCKETS:
        if last_day is None or days <= last_day:
            return name
    return AGE_UNKNOWN


def pool_hash(ids: Iterable[str]) -> str:
    """Fingerprint of a live pool: SHA-256 of its sorted, de-duplicated record ids,
    one per line. Taken at the start and the end of a live run (spec section 10)."""
    blob = "\n".join(sorted(set(str(i) for i in ids)))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def contains_phrase(norm_haystack: str, norm_needle: str) -> bool:
    """True when the needle's tokens appear as a contiguous run of whole tokens.

    Both arguments must already be normalized. Whole-token matching is what keeps
    an answer key of "int" from matching "print" and "into".
    """
    if not norm_needle or not norm_haystack:
        return False
    return f" {norm_needle} " in f" {norm_haystack} "


def answer_variants(answer, acceptable: Optional[Iterable]) -> List[str]:
    """Normalized, de-duplicated, non-empty answer plus acceptable answers, in order."""
    out: List[str] = []
    for value in [answer] + list(acceptable or []):
        norm = normalize(value)
        if norm and norm not in out:
            out.append(norm)
    return out


def matches_any(norm_text: str, variants: Sequence[str]) -> bool:
    return any(contains_phrase(norm_text, v) for v in variants)


class PhraseIndex:
    """Finds the records whose normalized text contains a phrase, without scanning
    every record: candidates come from the posting lists of the phrase's tokens and
    are then confirmed with contains_phrase."""

    def __init__(self, norm_texts: Sequence[str]):
        self.norm = list(norm_texts)
        self.postings: Dict[str, set] = {}
        for i, text in enumerate(self.norm):
            for tok in set(text.split()):
                self.postings.setdefault(tok, set()).add(i)

    def matching(self, norm_phrase: str) -> List[int]:
        toks = norm_phrase.split()
        if not toks:
            return []
        sets = [self.postings.get(t) for t in toks]
        if any(s is None for s in sets):
            return []
        cand = set.intersection(*sorted(sets, key=len))
        return sorted(i for i in cand if contains_phrase(self.norm[i], norm_phrase))

    def matching_any(self, norm_phrases: Iterable[str]) -> List[int]:
        out: set = set()
        for p in norm_phrases:
            out.update(self.matching(p))
        return sorted(out)


def token_overlap(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    """Share of the reference's tokens found in the candidate, by multiplicity.

    The reference is the denominator (the gold text, for the family tier).
    """
    if not reference_tokens or not candidate_tokens:
        return 0.0
    ref = Counter(reference_tokens)
    cand = Counter(candidate_tokens)
    hit = sum(min(c, cand[t]) for t, c in ref.items())
    return hit / len(reference_tokens)


FAMILY_THRESHOLD = 0.6


# Closed-class words that mark an answer key cut mid-phrase when they sit at
# either edge ("to measure", "button on the"). Left out on purpose: personal
# pronouns ("help you"), negation ("not just"), phrasal-verb particles ("set up"),
# "may" (also a month) and "am" (also a time of day, "5:00 AM"). Keys naturally
# start or end on those.
FUNCTION_WORDS = frozenset("""
a an the
about above across after against along among as at before behind below beneath
beside besides between beyond by despite during except for from in inside into
near of on onto outside per since than through throughout till to toward towards
under underneath until upon via with within without
and or but nor so yet because although though while whereas unless whether if
that which who whom whose what when where why how
is are was were be been being do does did has have had
will would shall should can could might must
this these those each every either neither such
my your his its our their
then there
""".split())


def edge_function_word(norm_key: str) -> Optional[str]:
    """The function word at the start or end of a normalized key, else None."""
    toks = norm_key.split()
    if not toks:
        return None
    if toks[0] in FUNCTION_WORDS:
        return toks[0]
    if toks[-1] in FUNCTION_WORDS:
        return toks[-1]
    return None


# ---------------------------------------------------------------------------
# Stopwords and the query-robustness variants (spec section 7)
# ---------------------------------------------------------------------------

# One fixed list, shared by the bm25stop reference adapter and the query pass.
STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been do does did what which who "
    "whom whose when where why how i you he she it we they my your his her its our their me "
    "him them this that these those with from by about as into than then so if not no can "
    "could would should will may might must have has had".split()
)

QUERY_VARIANTS = ("casefold_nopunct", "no_stopwords")
_WORD = re.compile(r"\w+")


def query_variant(question: str, name: str) -> str:
    """The query pass runs each question in two deterministic variants, each changing
    one thing: casefold_nopunct is the shared normalizer (lowercase, punctuation
    stripped, whitespace collapsed); no_stopwords is the question as written with
    every STOPWORDS word taken out."""
    if name == "casefold_nopunct":
        return normalize(question)
    if name == "no_stopwords":
        kept = _WORD.sub(lambda m: "" if m.group(0).casefold() in STOPWORDS else m.group(0), str(question))
        return " ".join(kept.split())
    raise ValueError(f"unknown query variant {name!r}")


# ---------------------------------------------------------------------------
# Lint vocabulary: one enum, checked against the lint spec on every lint run
# ---------------------------------------------------------------------------

SEVERITIES = ("error", "warn", "info")
STATUSES = ("publish", "rewrite", "reclassify", "drop", "hold_experimental")


CATEGORIES = (
    "direct_recall_user",
    "direct_recall_agent",
    "entity_attribution",
    "temporal_awareness",
    "paraphrase_recall",
    "cross_session",
    "long_horizon",
    "vocabulary_gap",
    "knowledge_update",
    "correction_handling",
    "implicit_preference",
    "abstention",
    "recency_preference",
)

# Categories whose gold set is several records (spec sections 3 and 5).
VERSION_CATEGORIES = ("knowledge_update", "correction_handling", "recency_preference")
ALL_GOLD_CATEGORIES = ("cross_session",)
MULTI_RECORD_CATEGORIES = VERSION_CATEGORIES + ALL_GOLD_CATEGORIES

# Categories whose answer is, by design, not written verbatim in the gold record.
NON_VERBATIM_CATEGORIES = ("vocabulary_gap", "implicit_preference")


def current_ids(must_rank_first) -> List[Any]:
    """The records a question names as the current version: none, one id, or a list
    of records that each carry the current version word for word (spec section 5).
    Any one of them first in the window scores on the version tier."""
    if not must_rank_first:
        return []
    if isinstance(must_rank_first, list):
        return list(must_rank_first)
    return [must_rank_first]


# ---------------------------------------------------------------------------
# Bands (spec section 6)
# ---------------------------------------------------------------------------

# (name, smallest N, largest N or None, k set, headline k, window)
BANDS = (
    ("none", 0, 99, (1,), 1, 5),
    ("S", 100, 999, (1, 5, 10), 5, 10),
    ("M", 1000, 9999, (1, 5, 10, 15), 10, 15),
    ("L", 10000, 49999, (1, 5, 10, 15, 20), 15, 20),
    ("XL", 50000, None, (1, 5, 10, 15, 20, 25), 20, 25),
)

VALID_CHANCE = 0.05


def band_for(n: int) -> Dict:
    for name, lo, hi, k_set, headline, window in BANDS:
        if n >= lo and (hi is None or n <= hi):
            return {"band": name, "k_set": list(k_set), "headline_k": headline,
                    "window": window, "small_pool": name == "none"}
    raise ValueError(f"no band for pool size {n}")


def band_unknown() -> Dict:
    """Live mode without list_ids(): N is unknown, so there is no band. The runner asks
    for the widest window and reports gold at every k any band reports, with no
    headline k and no chance."""
    name, lo, hi, k_set, headline, window = BANDS[-1]
    return {"band": "unknown", "k_set": list(k_set), "headline_k": None, "window": window, "small_pool": False}


# ---------------------------------------------------------------------------
# Chance model
# ---------------------------------------------------------------------------

def _miss_curve_any(n: int, g: int, k_max: int) -> List[float]:
    """P(no gold in a random k-subset) for k = 1..k_max: C(N-G, k) / C(N, k)."""
    out = []
    miss = 1.0
    for k in range(1, k_max + 1):
        den = n - k + 1
        num = den - g
        miss = miss * (num / den) if num > 0 else 0.0
        out.append(miss)
    return out


def _hit_curve_all(n: int, g: int, k_max: int) -> List[float]:
    """P(every gold in a random k-subset) for k = 1..k_max: C(N-G, k-G) / C(N, k)."""
    out = []
    for k in range(1, k_max + 1):
        if k < g:
            out.append(0.0)
            continue
        p = 1.0
        for j in range(g):
            p *= (k - j) / (n - j)
        out.append(p)
    return out


def mean_chance_curve(n: int, golds: Sequence[Tuple[int, str]], k_max: Optional[int] = None) -> List[float]:
    """Mean chance that a random retriever scores, for k = 1..k_max (index k-1).

    golds holds one (G, rule) pair per scored question: rule "any" scores a hit when
    any gold lands in the window (hypergeometric 1 - C(N-G, k) / C(N, k)), rule
    "all" only when every gold does. Questions are grouped by (G, rule), so the
    cost is one curve per distinct pair rather than one per question.
    """
    if n <= 0 or not golds:
        return []
    k_max = n if k_max is None else min(k_max, n)
    groups = Counter((min(max(int(g), 0), n), rule) for g, rule in golds)
    total = [0.0] * k_max
    for (g, rule), count in groups.items():
        if g == 0:
            continue
        if rule == "all":
            curve = _hit_curve_all(n, g, k_max)
        else:
            curve = [1.0 - m for m in _miss_curve_any(n, g, k_max)]
        for i, p in enumerate(curve):
            total[i] += p * count
    m = len(golds)
    return [t / m for t in total]


def chance_at(curve: Sequence[float], k: int) -> float:
    if not curve or k <= 0:
        return 0.0
    return curve[min(k, len(curve)) - 1]


def skill(observed: float, chance: float) -> float:
    """(observed - chance) / (1 - chance), as fractions, unclamped."""
    if chance >= 1.0:
        return 0.0
    return (observed - chance) / (1.0 - chance)


def k_ceiling(curve: Sequence[float], threshold: float) -> int:
    """Largest k whose mean chance is at or under the threshold (chance rises with k)."""
    ceiling = 0
    for i, p in enumerate(curve):
        if p <= threshold + 1e-12:
            ceiling = i + 1
        else:
            break
    return ceiling


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(s)))
    return s[min(rank, len(s)) - 1]

#!/usr/bin/env python3
"""
Parsica Recall public pool generator. The design is in docs/pool.md, the persona files
are in scenarios/, and the frozen prompts are in prompts/pool/.

Writes a synthetic agent-memory store: invented people, each with their own agent over
months of sessions, as one corpus.jsonl in a fixed order. The first --core records are
the question corpus and the 1k track; every larger track is a prefix of the file:

    parsica-recall generate-pool --scenarios scenarios --out ~/recall-pools/pool-v1 \\
        --seed 1 --core 1000 --total 50000 [--model claude-sonnet-5-5] [--personas p1,p2]

One model call writes one session of one persona, from the persona file, the persona's
running state and the session's plan. Everything else is deterministic: the same persona
files, seed, sizes, prompts and model responses give the same files byte for byte, and
--replay rebuilds a pool from its call log (generation/calls-NNNN.jsonl, in order) without
calling the API. A run that stops partway, or at --max-calls, picks up from its own call
log with --resume. By default the anthropic SDK reads the API key from ANTHROPIC_API_KEY
and this file never reads it. With --api-key-env NAME this file reads that one variable,
hands it to the SDK with the base URL pinned to api.anthropic.com, and takes it out of
every error it reports. Either way it never prints, logs or writes the key.

Exit status: 0 when the pool was written (or --estimate, or --check-api said ok); 2 on
input the generator refuses, or a model call that failed; 3 when --max-calls stopped the
run, which a --resume continues.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import generate_questions as gq
from . import lint_validator as lv
from . import recall_common as rc
from . import run_benchmark as rb

REPO_DIR = rc.ROOT_DIR  # the repository's root in a checkout, the package folder in an installed copy
SCENARIOS_DIR = os.path.join(REPO_DIR, "scenarios")
PROMPTS_DIR = os.path.join(REPO_DIR, "prompts", "pool")
DATA_DIR = os.path.join(REPO_DIR, "data")

GenerateError = gq.GenerateError  # input refused or a model call failed: exit 2

DEFAULT_MODEL = gq.DEFAULT_MODEL
DEFAULT_CORE = 1000           # the question corpus and the 1k track
DEFAULT_TOTAL = 50000         # the largest standard track; the 10k and 30k tracks are prefixes
DEFAULT_WORKERS = 4           # personas written at once; a persona's own sessions always run in order

# The plan (docs/pool.md section 3). Every value is recorded in the manifest.
QUOTA_REFERENCE = 125         # persona quotas are written for a core share of 125 records (1,000 over eight)
ENTRY_SHARE = 0.15            # entries per session: 15% of its records, at least one
ENTRY_WORDS = (100, 800)
MIN_TOKENS = 6                # the lint's min_gold_tokens, so every record can be gold
NEAR_COPY_OVERLAP = gq.NEAR_COPY_OVERLAP
RESTATE_OVERLAP = rc.FAMILY_THRESHOLD   # the lint's family threshold: no tail record covers a core record this much
SESSION_GAP_MINUTES = 180     # two sessions of one persona on one day start at least this far apart, when they can
RECORD_GAP_SECONDS = (40, 420)
MAX_ATTEMPTS = 3              # a session whose reply is unusable is asked again, at most twice
MAKEUP_SESSIONS = 3           # make-up sessions a persona may add in one round to reach its exact share
MAKEUP_ROUNDS = 3             # rounds of make-ups after the whole-pool screen
FACTS_IN_STATE = 200          # the newest facts the state shows the model
RESTATE_MIN_CONTENT = 8       # a core record needs this many content tokens before a tail record can restate it
TAIL_OVERASK_FLOOR = 100      # the tail is planned this many per mille over its share, at least ...
TAIL_OVERASK_MARGIN = 50      # ... and the core's measured drop rate plus this, then trimmed to the exact share
POOL_MAX_TOKENS = 32000       # pool calls stream, so a long reply never meets the SDK's non-streaming limit
API_URL = gq.API_URL          # with --api-key-env, every call goes here, whatever ANTHROPIC_BASE_URL says
CALL_PART_BYTES = 45_000_000  # the call log rolls to a new part before this size (GitHub refuses files over 100 MB)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "high"
# US dollars per million tokens: input, output, 5-minute cache write, cache read. One table
# for both generators, kept in generate_questions.py. Used only for the estimate printed
# before a run and the cost the manifest reports; a model not listed gets token counts alone.
PRICES = gq.PRICES
# Per call and per record, from pilots 1 and 2 (Sonnet 5.5, effort high), for the estimate only.
EST_INPUT_PER_CALL = {"core": 13500, "tail": 16000}
EST_CACHED_PER_CALL = {"core": 2500, "tail": 6000}
EST_OUTPUT_PER_RECORD = 1000
EST_EXTRA_CALLS = 0.08        # retries and make-up sessions, as a share of planned sessions
# How a change or a correction is worded, drawn per step from its own seeded generator, so
# the records never follow one template and no single word marks a correction (pilot 1).
CHANGE_STYLES = (
    "in passing, inside a sentence that is mostly about something else",
    "as a decision just made, with the reason for it",
    "as the result of something just done or found",
    "as forced by circumstances: a delay, a price, a broken part, someone else's choice",
    "in one short line, without explanation",
    "by comparing it with how things were, without restating the old value",
    "while planning something that depends on it",
)
CORRECTION_STYLES = {
    "the person": (
        "the person catches their own slip in passing and simply gives the right value next to the old one, "
        "without the words wrong, mistake or correct",
        "the person says they mixed two things up and says which is which",
        "someone in the person's life pointed it out, and the person passes that on",
        "the agent notices it does not match a receipt, message, document or log and quietly updates it",
        "the person says plainly that they had it wrong and gives the right value",
        "the person finds out while doing something that depends on it, and says what it actually is",
    ),
    "the agent": (
        "the agent rechecks its source and gives the right value against the earlier one, briefly and without fuss",
        "the person notices the agent's figure does not match what they see, and says what it is",
        "the agent owns its slip in a few words and gives the right value",
        "the agent finds the real value while doing a related task and updates it in passing",
        "the agent asks the person to confirm, and the record states the confirmed value against the earlier one",
    ),
}
QUOTA_KEYS = ("version_chains", "preference_changes", "corrections", "threads", "long_horizon_facts",
              "shown_preferences")
LABELS = {"version_chains": "V", "preference_changes": "R", "corrections": "C", "threads": "T",
          "long_horizon_facts": "L", "shown_preferences": "S"}
KINDS = ("user", "agent", "entry")
PHASES = ("core", "tail")
PROMPT_FILES = {"core": "core_session.md", "tail": "tail_session.md"}
TRACKS = (("1k", 1000), ("10k", 10000), ("30k", 30000), ("50k", 50000))
PERCENTILES = (5, 25, 50, 75, 95)
DROP_REASONS = ("bad_record_shape", "not_unicode", "too_short", "entry_length", "secret_or_pii", "duplicate_id",
                "near_copy", "restates_core", "over_budget")

_PERSONA_ID = re.compile(r"p[1-9][0-9]*")
_SESSION_ID = re.compile(r"p[1-9][0-9]*-s[0-9a-f]{6}")
_POOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ENV_NAME = gq._ENV_NAME
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random("recall/1.0 generate_pool|" + "|".join(str(p) for p in [seed, *parts]))


def _add_months(start: date, months: int) -> date:
    y, m = divmod(start.month - 1 + months, 12)
    year, month = start.year + y, m + 1
    days_in_month = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(start.day, days_in_month))


def _weighted_pick(rng: random.Random, items: List[Any], weights: List[int]) -> int:
    """The index of one item drawn by integer weights, in integer arithmetic, so the draw
    is the same on every platform."""
    r = rng.randrange(sum(weights))
    for i, w in enumerate(weights):
        if r < w:
            return i
        r -= w
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Prompts and persona files
# ---------------------------------------------------------------------------

class PoolPrompt:
    """One frozen prompt: the file's text with its line endings read as LF, the SHA-256 of
    those bytes, and the JSON schema between BEGIN SCHEMA and END SCHEMA."""

    def __init__(self, phase: str, path: str):
        try:
            with open(path, "rb") as f:
                self.data = gq._lf(f.read())
        except OSError as exc:
            raise GenerateError(f"cannot read the {phase} session prompt: {exc}")
        self.phase = phase
        self.path = path
        self.name = os.path.basename(path)
        self.text = self.data.decode("utf-8")
        self.sha256 = _sha256(self.data)
        m = gq._SCHEMA_BLOCK.search(self.text)
        if not m:
            raise GenerateError(f"{path}: no JSON schema between BEGIN SCHEMA and END SCHEMA")
        try:
            self.schema = json.loads(m.group(1))
        except ValueError as exc:
            raise GenerateError(f"{path}: the schema is not valid JSON ({exc})")


def load_prompts(prompts_dir: str = PROMPTS_DIR) -> Dict[str, PoolPrompt]:
    return {phase: PoolPrompt(phase, os.path.join(prompts_dir, PROMPT_FILES[phase])) for phase in PHASES}


# The fields the model sees, in this order. The rest of a persona file is for the plan.
PUBLIC_FIELDS = ("name", "situation", "domains", "agent_role", "voice", "agent_voice", "people", "things")
# The fields that shape the plan. The plan is seeded from these alone, so rewording a
# persona (its situation, voice or people) never moves its sessions.
PLAN_FIELDS = ("id", "timeline", "cadence", "records_per_session", "core_quotas")


class Persona:
    """One persona file, read with its line endings as LF and hashed, then checked."""

    def __init__(self, path: str):
        self.path = path
        self.file = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                self.data = gq._lf(f.read())
            d = json.loads(self.data.decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise GenerateError(f"{path}: cannot read the persona file ({exc})")
        self.sha256 = _sha256(self.data)
        self.raw = d

        def need(cond: bool, what: str) -> None:
            if not cond:
                raise GenerateError(f"{path}: {what}")

        need(isinstance(d, dict), "a persona file is one JSON object")
        need(isinstance(d.get("id"), str) and bool(_PERSONA_ID.fullmatch(d["id"])), "id must look like p1, p2, ...")
        for key in ("name", "situation", "agent_role", "voice", "agent_voice"):
            need(isinstance(d.get(key), str) and bool(d[key].strip()), f"{key} must be a non-empty string")
        for key in ("domains", "things"):
            need(isinstance(d.get(key), list) and d[key] and all(isinstance(x, str) and x.strip() for x in d[key]),
                 f"{key} must be a non-empty list of strings")
        need(isinstance(d.get("people"), list) and all(
            isinstance(p, dict) and set(p) == {"name", "relation"} and all(isinstance(v, str) and v.strip()
                                                                          for v in p.values())
            for p in d["people"]), "people must be a list of {name, relation}")
        tl = d.get("timeline")
        need(isinstance(tl, dict) and set(tl) == {"start", "months"}, "timeline must be {start, months}")
        try:
            self.start = date.fromisoformat(tl["start"])
        except (TypeError, ValueError):
            raise GenerateError(f"{path}: timeline.start must be a date such as 2025-09-01")
        need(isinstance(tl["months"], int) and 6 <= tl["months"] <= 18, "timeline.months must be 6 to 18")
        self.months = tl["months"]
        self.end = _add_months(self.start, self.months)  # the first day after the timeline
        cad = d.get("cadence")
        need(isinstance(cad, dict) and set(cad) == {"description", "weekday_weights", "local_hours",
                                                    "utc_offset_hours"},
             "cadence must be {description, weekday_weights, local_hours, utc_offset_hours}")
        w = cad["weekday_weights"]
        need(isinstance(w, list) and len(w) == 7 and all(isinstance(x, int) and x >= 0 for x in w) and sum(w) > 0,
             "cadence.weekday_weights must be seven whole numbers, Monday first, not all 0")
        hours = cad["local_hours"]
        need(isinstance(hours, list) and len(hours) == 2 and all(isinstance(h, int) for h in hours)
             and 0 <= hours[0] < hours[1] <= 24, "cadence.local_hours must be [first hour, last hour] within 0 to 24")
        need(isinstance(cad["utc_offset_hours"], int) and -12 <= cad["utc_offset_hours"] <= 14,
             "cadence.utc_offset_hours must be a whole number from -12 to 14")
        self.weekday_weights, self.local_hours, self.utc_offset = w, hours, cad["utc_offset_hours"]
        rps = d.get("records_per_session")
        need(isinstance(rps, list) and len(rps) == 2 and all(isinstance(x, int) for x in rps)
             and 3 <= rps[0] <= rps[1] <= 30, "records_per_session must be [fewest, most] within 3 to 30")
        self.size_range = (rps[0], rps[1])
        q = d.get("core_quotas")
        need(isinstance(q, dict) and set(q) == set(QUOTA_KEYS) and all(isinstance(v, int) and v >= 0
                                                                      for v in q.values()),
             "core_quotas must give a whole number for each of " + ", ".join(QUOTA_KEYS))
        self.quotas = q
        self.id = d["id"]
        self.n = int(self.id[1:])
        self.name = d["name"]
        self.public = {k: d[k] for k in PUBLIC_FIELDS}
        plan = {k: d[k] for k in PLAN_FIELDS}
        plan["cadence"] = {k: v for k, v in d["cadence"].items() if k != "description"}
        self.plan_key = _sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_personas(folder: str, wanted: Optional[Sequence[str]] = None) -> List[Persona]:
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
    except OSError as exc:
        raise GenerateError(f"--scenarios {folder}: {exc}")
    personas = [Persona(os.path.join(folder, n)) for n in names]
    seen: Dict[str, str] = {}
    for p in personas:
        if p.id in seen:
            raise GenerateError(f"--scenarios: {p.file} and {seen[p.id]} both use the id {p.id}")
        seen[p.id] = p.file
    if wanted:
        unknown = [w for w in wanted if w not in seen]
        if unknown:
            raise GenerateError(f"--personas: no persona file in {folder} has the id {', '.join(unknown)}")
        personas = [p for p in personas if p.id in set(wanted)]
    if not personas:
        raise GenerateError(f"--scenarios {folder}: no persona files")
    return sorted(personas, key=lambda p: p.n)


def split_evenly(total: int, n: int) -> List[int]:
    """total over n shares; the remainder goes one each to the first shares."""
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def scaled_quotas(persona: Persona, share: int) -> Dict[str, int]:
    """The persona's quotas, written for a core share of QUOTA_REFERENCE, scaled to its
    actual share in integer arithmetic (half up), and at least 1 where the file asks for any."""
    out = {}
    for key in QUOTA_KEYS:
        q = persona.quotas[key]
        out[key] = 0 if not q or not share else max(1, (q * share * 2 + QUOTA_REFERENCE) // (2 * QUOTA_REFERENCE))
    return out


# ---------------------------------------------------------------------------
# The plan: sessions, dates, sizes and what each core session must include
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, persona: Persona, phase: str, seq: int, sid: str, local: datetime, records: int,
                 directives: List[Dict[str, Any]], makeup: bool = False):
        self.persona = persona
        self.phase = phase
        self.seq = seq                  # order within the persona's phase: planned sessions by time, then make-ups
        self.id = sid
        self.local = local              # the persona's local time, which the model sees
        self.start = local - timedelta(hours=persona.utc_offset)  # UTC, which the records carry
        self.records = records
        self.directives = directives
        self.makeup = makeup

    def plan_json(self) -> Dict[str, Any]:
        return {"session": self.id, "phase": self.phase, "seq": self.seq, "makeup": self.makeup,
                "local": self.local.strftime("%Y-%m-%d %H:%M"), "start": self.start.strftime(_TS_FORMAT),
                "records": self.records, "entries": entries_for(self.records),
                "directives": [f"{d['label']} {d['step']}/{d['of']}" for d in self.directives]}


def entries_for(records: int) -> int:
    return max(1, int(ENTRY_SHARE * records + 0.5)) if records else 0


def session_sizes(rng: random.Random, budget: int, lo: int, hi: int) -> List[int]:
    """Session sizes drawn from [lo, hi] that sum to budget exactly. A last session that
    would fall under lo is spread one record at a time over the sessions before it that
    are under hi; if they have no room for it, it stays as a short session. No size is
    ever over hi."""
    sizes: List[int] = []
    while sum(sizes) < budget:
        sizes.append(rng.randint(lo, hi))
    if sizes:
        sizes[-1] -= sum(sizes) - budget
        room = sum(hi - s for s in sizes[:-1])
        if sizes[-1] < lo and len(sizes) > 1 and room >= sizes[-1]:
            extra = sizes.pop()
            i = 0
            while extra:  # one record at a time, to the sessions that are not yet at the largest size
                if sizes[i % len(sizes)] < hi:
                    sizes[i % len(sizes)] += 1
                    extra -= 1
                i += 1
    return sizes


def session_id(seed: int, persona: Persona, tag: str, used: set) -> str:
    """An opaque session id: nothing in it says which phase, or which place in time, a
    session has, so a source never tells a ranker which records are core."""
    k = 0
    while True:
        sid = f"{persona.id}-s{_sha256(f'recall/1.0 generate_pool|{seed}|{persona.id}|{tag}|{k}'.encode())[:6]}"
        if sid not in used:
            used.add(sid)
            return sid
        k += 1


def pick_minute(rng: random.Random, persona: Persona, day: date, taken: Dict[date, List[int]]) -> datetime:
    """A local start time on day, within the persona's hours, unique to the minute and,
    when the day allows it, SESSION_GAP_MINUTES from the persona's other sessions that day."""
    lo, hi = persona.local_hours[0] * 60, persona.local_hours[1] * 60
    others = taken.setdefault(day, [])
    minute = rng.randrange(lo, hi)
    for _ in range(50):
        if all(abs(minute - o) >= SESSION_GAP_MINUTES for o in others):
            break
        minute = rng.randrange(lo, hi)
    while minute in others:
        minute = minute + 1 if minute + 1 < hi else lo
    others.append(minute)
    return datetime(day.year, day.month, day.day) + timedelta(minutes=minute)


def pick_days(rng: random.Random, persona: Persona, n: int) -> List[date]:
    """n days of the timeline drawn by the weekday weights, without repeats while the
    timeline has enough days, sorted."""
    days = [persona.start + timedelta(days=i) for i in range((persona.end - persona.start).days)]
    days = [d for d in days if persona.weekday_weights[d.weekday()] > 0]
    weights = [persona.weekday_weights[d.weekday()] for d in days]
    out: List[date] = []
    pool, pool_w = list(days), list(weights)
    while len(out) < n:
        if not pool:
            pool, pool_w = list(days), list(weights)
        i = _weighted_pick(rng, pool, pool_w)
        out.append(pool.pop(i))
        pool_w.pop(i)
    return sorted(out)


def place_directives(rng: random.Random, n: int, quotas: Dict[str, int]) -> Tuple[List[List[Dict[str, Any]]],
                                                                                   List[str]]:
    """Which session carries each step of each quota item: long-horizon facts early,
    version chains and preference changes spread with gaps, a correction one to three
    sessions after its error, threads and shown preferences across the timeline. Returns
    the directives per session and the items that could not be placed."""
    per: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    lost: List[str] = []

    def put(label: str, typ: str, positions: List[int]) -> None:
        for step, pos in enumerate(positions, 1):
            per[pos].append({"label": label, "type": typ, "step": step, "of": len(positions)})

    def chain_positions(steps: int) -> Optional[List[int]]:
        for gap in (2, 1):
            out: List[int] = []
            prev = None
            for j in range(steps):
                remaining = steps - 1 - j
                lo = 0 if prev is None else prev + gap
                hi = (max(0, n // 2 - 1) if prev is None else n - 1) if remaining else n - 1
                hi = min(hi, n - 1 - remaining * gap)
                if lo > hi:
                    break
                prev = rng.randint(lo, hi)
                out.append(prev)
            if len(out) == steps:
                return out
        return None

    for key in QUOTA_KEYS:
        letter = LABELS[key]
        for i in range(1, quotas[key] + 1):
            label = f"{letter}{i}"
            if key == "long_horizon_facts":
                if n < 1:
                    lost.append(label)
                    continue
                put(label, "long_horizon", [rng.randrange(max(1, (n + 3) // 4))])
            elif key in ("version_chains", "preference_changes"):
                typ = "version_chain" if key == "version_chains" else "preference_change"
                steps = 3 if n >= 5 and rng.randrange(100) < 35 else 2
                pos = chain_positions(steps) or chain_positions(2)
                if pos is None:
                    lost.append(label)
                    continue
                put(label, typ, pos)
            elif key == "corrections":
                if n < 2:
                    lost.append(label)
                    continue
                a = rng.randint(0, n - 2)
                put(label, "correction", [a, rng.randint(a + 1, min(a + 3, n - 1))])
            else:
                typ = "thread" if key == "threads" else "shown_preference"
                k = min(n, rng.choice((2, 3, 3, 4)) if key == "threads" else rng.choice((2, 3)))
                if k < 2:
                    lost.append(label)
                    continue
                put(label, typ, sorted(rng.sample(range(n), k)))
    return per, lost


def rebalance(sizes: List[int], loads: List[int], lo: int) -> None:
    """Moves records into sessions that carry more directives than records, from the
    session with the most room (ties by order), never taking one under max(load, lo)."""
    while True:
        over = [i for i in range(len(sizes)) if loads[i] > sizes[i]]
        if not over:
            return
        i = over[0]
        room = [(sizes[j] - max(loads[j], lo), -j) for j in range(len(sizes)) if j != i]
        best = max(room, default=(0, 0))
        if best[0] <= 0:
            raise GenerateError("the persona's quotas need more records than its core share holds; lower the quotas "
                                "or raise --core")
        j = -best[1]
        sizes[j] -= 1
        sizes[i] += 1


def plan_phase(seed: int, persona: Persona, phase: str, budget: int, used_ids: set,
               taken: Dict[date, List[int]], quotas: Optional[Dict[str, int]] = None
               ) -> Tuple[List[Session], List[str]]:
    """One persona's planned sessions for a phase, in time order. The core's plan comes
    from the seed, the persona file and the core share alone, so it is the same whatever
    --total is; the tail's sessions are dated over the same timeline."""
    rng = _rng(seed, persona.plan_key, phase)
    lo, hi = persona.size_range
    sizes = session_sizes(rng, budget, lo, hi)
    n = len(sizes)
    directives: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    lost: List[str] = []
    if phase == "core" and quotas:
        directives, lost = place_directives(rng, n, quotas)
        rebalance(sizes, [len(d) for d in directives], lo)
    days = pick_days(rng, persona, n)
    starts = sorted(pick_minute(rng, persona, d, taken) for d in days)
    sessions = []
    for seq, (local, size, dirs) in enumerate(zip(starts, sizes, directives)):
        sid = session_id(seed, persona, f"{phase}|{seq}", used_ids)
        sessions.append(Session(persona, phase, seq, sid, local, size, dirs))
    return sessions, lost


# ---------------------------------------------------------------------------
# Screens: near copies, restating the core
# ---------------------------------------------------------------------------

class NearCopyIndex:
    """The question generator's near-copy rule, kept incrementally: each record covers at
    least NEAR_COPY_OVERLAP of the other's tokens, and both carry the same digit tokens and
    the same capitalized words."""

    def __init__(self) -> None:
        self.postings: Dict[str, List[str]] = defaultdict(list)
        self.info: Dict[str, Tuple[List[str], frozenset, frozenset]] = {}
        self.order: Dict[str, int] = {}

    def find(self, toks: List[str], digits: frozenset, caps: frozenset) -> Optional[str]:
        if not toks:
            return None
        # A near copy misses at most a tenth of toks's occurrences, so it holds one of any
        # int(n / 10) + 2 of toks's distinct tokens; the rarest gather the fewest candidates.
        k = int(len(toks) * (1 - NEAR_COPY_OVERLAP)) + 2
        rare = heapq.nsmallest(k, set(toks), key=lambda t: (len(self.postings.get(t, ())), t))
        cands = sorted({c for t in rare for c in self.postings.get(t, ())}, key=self.order.__getitem__)
        for c in cands:
            ct, cd, cc = self.info[c]
            if cd == digits and cc == caps and min(rc.token_overlap(toks, ct), rc.token_overlap(ct, toks)) \
                    >= NEAR_COPY_OVERLAP:
                return c
        return None

    def add(self, rid: str, toks: List[str], digits: frozenset, caps: frozenset) -> None:
        self.info[rid] = (toks, digits, caps)
        self.order[rid] = len(self.order)
        for t in set(toks):
            self.postings[t].append(rid)


class CoverIndex:
    """Finds an indexed record whose tokens a new text covers at threshold or more, by
    rc.token_overlap with the indexed record as the reference (the family tier's
    direction). Each record is indexed under its rarest distinct tokens until they hold
    more than (1 - threshold) of its occurrences, so any text that covers it holds one."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.postings: Dict[str, List[str]] = defaultdict(list)
        self.refs: Dict[str, List[str]] = {}
        self.order: Dict[str, int] = {}

    def add(self, rid: str, toks: List[str]) -> None:
        if not toks:
            return
        self.refs[rid] = toks
        self.order[rid] = len(self.order)
        counts = Counter(toks)
        need = int((1 - self.threshold) * len(toks)) + 1
        held = 0
        for t in sorted(counts, key=lambda t: (len(self.postings.get(t, ())), t)):
            self.postings[t].append(rid)
            held += counts[t]
            if held > need:
                break

    def find(self, toks: List[str]) -> Optional[str]:
        cands = {c for t in set(toks) for c in self.postings.get(t, ())}
        for c in sorted(cands, key=self.order.__getitem__):
            if rc.token_overlap(self.refs[c], toks) >= self.threshold:
                return c
        return None


def record_features(text: str) -> Tuple[List[str], frozenset, frozenset]:
    toks = rc.tokenize(text)
    return toks, gq.digit_tokens(toks), gq.capital_tokens(text)


def features(rec: Dict[str, Any]) -> Tuple[List[str], frozenset, frozenset]:
    """A kept record's tokens, digit tokens and capitalized words, computed once."""
    if "_features" not in rec:
        rec["_features"] = record_features(rec["text"])
    return rec["_features"]


# ---------------------------------------------------------------------------
# Model calls: recorded, replayed or live
# ---------------------------------------------------------------------------

CALL_PART = re.compile(r"calls-(\d{4})\.jsonl")


def call_parts(gen_dir: str) -> List[str]:
    """The call log's parts in a generation folder, in order: calls-0001.jsonl, ...
    A log written before parts existed (calls.jsonl) is read first."""
    try:
        names = os.listdir(gen_dir)
    except OSError:
        return []
    parts = sorted(n for n in names if CALL_PART.fullmatch(n))
    legacy = ["calls.jsonl"] if "calls.jsonl" in names else []
    return [os.path.join(gen_dir, n) for n in legacy + parts]


class Paused(Exception):
    """--max-calls was reached: the run stops before its next API call, resumable."""


class RecordedModel:
    """Answers each request with the response the recorded call log holds for the same
    request (its SHA-256), in order. The log is the numbered parts in a generation
    folder, read in order; path may name the folder or any part. With a live model
    behind it (--resume), a request the log does not hold goes to the API; without one
    (--replay) it is an error. A last line cut short by a stopped run is skipped."""

    def __init__(self, path: str, live: Any = None):
        self.path = path
        self.live = live
        self.recorded: Dict[str, deque] = defaultdict(deque)
        self.skipped_lines = 0
        self.last_part: Optional[str] = None
        self.last_lines: List[str] = []  # the last part's readable lines, as written
        self._lock = threading.Lock()
        if os.path.isdir(path):
            files = call_parts(path)
        elif CALL_PART.fullmatch(os.path.basename(path)):
            files = call_parts(os.path.dirname(path))
        else:
            files = [path]
        if not files:
            raise GenerateError(f"no call log in {path}")
        for n, f_path in enumerate(files):
            try:
                with open(f_path, encoding="utf-8") as f:
                    lines = f.read().split("\n")
            except OSError as exc:
                raise GenerateError(f"cannot read the call log {f_path}: {exc}")
            last_file = n == len(files) - 1
            filled = [i for i, line in enumerate(lines) if line.strip()]
            good = []
            for i in filled:
                try:
                    rec = json.loads(lines[i])
                    self.recorded[rec["request_sha256"]].append(rec)
                    good.append(lines[i])
                except (ValueError, KeyError, TypeError):
                    if last_file and i == filled[-1]:  # the line a stopped run was writing
                        self.skipped_lines += 1
                        continue
                    raise GenerateError(f"{f_path}: line {i + 1} is not a recorded call")
            if last_file:
                self.last_part, self.last_lines = f_path, good

    def connect(self) -> None:
        if self.live is not None and hasattr(self.live, "connect"):
            self.live.connect()

    def call(self, label: str, prompt: Any, user: str, request_sha: str, **kw: Any) -> gq.Reply:
        with self._lock:
            queue = self.recorded.get(request_sha)
            rec = queue.popleft() if queue else None
        if rec is not None:
            reply = gq.Reply(rec["response"], rec["served_model"], rec["stop_reason"], rec["input_tokens"],
                             rec["output_tokens"], rec["time"], rec.get("sdk"), rec.get("started"))
            reply.cache_write = rec.get("cache_creation_input_tokens", 0)
            reply.cache_read = rec.get("cache_read_input_tokens", 0)
            reply.in_log = self.live is not None  # under --resume the call is already in this folder's log
            return reply
        if self.live is None:
            raise GenerateError(f"--replay: {self.path} holds no response for the {label} request; the persona "
                                "files, seed, sizes, model, effort or prompts differ from the recorded run")
        return self.live.call(label, prompt, user, request_sha, **kw)


class LimitedModel:
    """The live model behind a --max-calls guard: the call that would pass the limit is
    never made, and the run stops cleanly so a resume can continue it."""

    def __init__(self, live: Any, max_calls: int):
        self.live = live
        self.max_calls = max_calls
        self.made = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if hasattr(self.live, "connect"):
            self.live.connect()

    def call(self, label: str, prompt: Any, user: str, request_sha: str, **kw: Any) -> gq.Reply:
        with self._lock:
            if self.made >= self.max_calls:
                raise Paused(f"--max-calls {self.max_calls} reached")
            self.made += 1
        return self.live.call(label, prompt, user, request_sha, **kw)


class PoolAnthropicModel(gq.AnthropicModel):
    """The Messages API for pool sessions: streamed (a session with long entries can run
    past the SDK's limit for a non-streaming request), max_tokens POOL_MAX_TOKENS, the
    effort each call asks for, and prompt caching on the fixed prefix: the system prompt,
    and the start of the user message up to split (the persona, and in the tail the core
    lists, which stay the same for every session of a persona). Server-side fallbacks
    are off, as for the question generator, so every record comes from the pinned model.

    The key is handled by the question generator's AnthropicModel, which this class
    extends: by default the SDK reads ANTHROPIC_API_KEY and nothing here reads it; with
    api_key_env (--api-key-env NAME) the client is made from that one variable, with the
    base URL pinned to API_URL. The value is never printed, logged or written, and every
    error raised here has it taken out (self._scrub) and drops the exception it came from.
    check(), the --check-api call, is the parent's too."""

    def call(self, label: str, prompt: Any, user: str, request_sha: str, effort: str = "high",
             split: int = 0) -> gq.Reply:
        if self._client is None:
            self._connect()
        content: List[Dict[str, Any]] = []
        if 0 < split < len(user):
            content.append({"type": "text", "text": user[:split], "cache_control": {"type": "ephemeral"}})
            content.append({"type": "text", "text": user[split:]})
        else:
            content.append({"type": "text", "text": user})
        started = gq._utc_now()
        try:
            with self._client.messages.stream(
                model=self.model_id,
                max_tokens=POOL_MAX_TOKENS,
                system=[{"type": "text", "text": prompt.text, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": prompt.schema}},
            ) as stream:
                resp = stream.get_final_message()
        except self._anthropic.APIError as exc:
            raise GenerateError(self._scrub(f"the {label} call failed: {type(exc).__name__}: {exc}")) from None
        except TypeError as exc:  # an SDK too old for this request, or no credentials it can find
            raise GenerateError(self._scrub(f"the anthropic SDK refused the {label} request ({exc}); upgrade it "
                                            "with pip install -U anthropic and check the API key is set")) from None
        except Exception as exc:  # anything else the SDK raises: a message, never a traceback that could hold the key
            raise GenerateError(self._scrub(f"the {label} call failed: {type(exc).__name__}: {exc}")) from None
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        usage = getattr(resp, "usage", None)
        reply = gq.Reply(text, resp.model, resp.stop_reason, getattr(usage, "input_tokens", 0) or 0,
                         getattr(usage, "output_tokens", 0) or 0, gq._utc_now(), self._sdk, started)
        reply.cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        reply.cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        return reply


def pool_request_sha256(model_id: str, prompt: Any, user: str, effort: str, split: int) -> str:
    blob = json.dumps({"model": model_id, "max_tokens": POOL_MAX_TOKENS, "effort": effort, "stream": True,
                       "system_sha256": prompt.sha256, "cache_split": split, "user": user},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_tokens(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t not in rc.STOPWORDS]


class RestateIndex:
    """The restates-core screen and the restatement rate. A record restates an indexed
    one when, on content tokens (rc.STOPWORDS left out), each covers at least threshold
    of the other's tokens, and the indexed one has at least RESTATE_MIN_CONTENT of them.
    Both directions, so a long entry that happens to hold most of a short record's words
    is not a restatement, and a paraphrase of similar length is (pilot 2's twelve drops
    were all the first kind)."""

    def __init__(self, threshold: float = RESTATE_OVERLAP, min_content: int = RESTATE_MIN_CONTENT):
        self.cover = CoverIndex(threshold)
        self.threshold = threshold
        self.min_content = min_content

    def add(self, rid: str, toks: Sequence[str]) -> None:
        content = content_tokens(toks)
        if len(content) >= self.min_content:
            self.cover.add(rid, content)

    def find(self, toks: Sequence[str]) -> Optional[str]:
        content = content_tokens(toks)
        if not content:
            return None
        cands = {c for t in set(content) for c in self.cover.postings.get(t, ())}
        for c in sorted(cands, key=self.cover.order.__getitem__):
            ref = self.cover.refs[c]
            if rc.token_overlap(ref, content) >= self.threshold and rc.token_overlap(content, ref) >= self.threshold:
                return c
        return None


def parse_session_reply(reply: gq.Reply) -> Tuple[List[Any], Dict[str, Any], Optional[str]]:
    if reply.stop_reason == "refusal":
        return [], {}, "the model declined the request"
    if reply.stop_reason == "max_tokens":
        return [], {}, "the reply hit max_tokens"
    try:
        data = json.loads(reply.text)
    except ValueError:
        return [], {}, "the reply is not JSON"
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        return [], {}, "the reply has no records list"
    update = data.get("state_update")
    return data["records"], update if isinstance(update, dict) else {}, None


# ---------------------------------------------------------------------------
# A persona's state and its sessions
# ---------------------------------------------------------------------------

class PersonaRun:
    """One persona through a run: its plan, its running state, its records and drops."""

    def __init__(self, ctx: "Run", persona: Persona, core_share: int, tail_share: int):
        self.ctx = ctx
        self.persona = persona
        self.budget = {"core": core_share, "tail": tail_share}
        self.quotas = scaled_quotas(persona, core_share)
        self.used_ids: set = set()
        self.taken: Dict[date, List[int]] = {}
        self.sessions: Dict[str, List[Session]] = {}
        self.lost: List[str] = []
        self.sessions["core"], self.lost = plan_phase(ctx.seed, persona, "core", core_share, self.used_ids,
                                                      self.taken, self.quotas)
        self.sessions["tail"] = []  # planned when the core is done, over-asked by its measured drop rate
        self.tail_planned = 0
        self.state: Dict[str, Any] = {"people": [dict(p) for p in persona.raw["people"]], "facts": [],
                                      "tail_facts": [], "items": {}}
        for sessions in self.sessions.values():
            for s in sessions:
                for d in s.directives:
                    self.state["items"].setdefault(d["label"], {"type": d["type"], "subject": "", "values": []})
        self.records: Dict[str, List[Dict[str, Any]]] = {"core": [], "tail": []}
        self.dropped: List[Dict[str, Any]] = []
        self.counts: Counter = Counter()
        self.carried: List[Dict[str, Any]] = []  # directives of a session whose replies were all unusable
        self.makeups = 0

    def plan_tail(self, overask_pm: int) -> None:
        """The tail's sessions, planned for its share plus overask_pm per mille (rounded up),
        so the screens' drops rarely leave it short; trim() then takes it to the exact share."""
        share = self.budget["tail"]
        self.tail_planned = share + -(-share * overask_pm // 1000) if share else 0
        self.sessions["tail"], _ = plan_phase(self.ctx.seed, self.persona, "tail", self.tail_planned, self.used_ids,
                                              self.taken)

    # -- the user message ---------------------------------------------------

    def directive_text(self, d: Dict[str, Any]) -> str:
        item = self.state["items"][d["label"]]
        vals = item["values"]
        label, typ = d["label"], d["type"]
        last = vals[-1] if vals else None
        k = len(vals) + 1
        of = max(d["of"], k)
        if typ == "long_horizon":
            return (f"{label}: state one specific fact about this person's life that matters but will never come up "
                    "again (a long-horizon fact), such as a date, a model or serial name, a measurement, a size, a "
                    f"name or a place. Put it in its own record. Report it under {label}.")
        rng = _rng(self.ctx.seed, self.persona.plan_key, "style", label, k)  # never draws from the plan's generator
        if typ in ("version_chain", "preference_change"):
            if last is None:
                if typ == "version_chain":
                    return (f"{label}, step 1 of {of}: state a specific current fact that will change later in this "
                            "life: a value such as a count, a price, a date or deadline, a tool, a place, a plan or "
                            "who is responsible. Pick a subject not used before. Put it in its own record. Report the "
                            f"subject and the value under {label}.")
                return (f"{label}, step 1 of {of}: have the person state a clear preference in their own words: a "
                        "favorite, a brand or tool they choose, how they like something done. Pick a subject not used "
                        f"before. Put it in its own record. Report the subject and the preference under {label}.")
            what = "fact" if typ == "version_chain" else "preference"
            who = "the person" if typ == "preference_change" else rng.choice(("the person", "the agent"))
            style = rng.choice(CHANGE_STYLES)
            return (f"{label}, step {k} of {of}: the {what} about \"{item['subject']}\" has changed since it was last "
                    f"given as \"{last['value']}\" ({last['date']}). In its own record, {who} brings up the new {what} "
                    f"{style}. Name the subject in words that make clear it is the same thing, and do not repeat the "
                    f"old value. Report the new value under {label}.")
        if typ == "correction":
            if last is None:
                who = rng.choice(("the person", "the agent"))
                return (f"{label}, part 1 of 2: in its own record, {who} states a specific fact that is not right: a "
                        "misremembered date, number, name, place or amount, said with ordinary confidence. Nobody "
                        f"notices yet. Report under {label} the subject, the value as stated, and the true value in "
                        "the note.")
            erred = self.erred(label)
            style = rng.choice(CORRECTION_STYLES[erred])
            note = f" The true value is \"{last['note']}\"." if last.get("note") else ""
            return (f"{label}, part 2 of 2: on {last['date']}, {erred} gave {item['subject']} as \"{last['value']}\", "
                    f"which is not right.{note} In its own record, set it straight this way: {style}. The record names "
                    "the earlier value, so a reader can tell what it replaces, and gives the right one. Do not open "
                    f"with the date. Report the right value under {label}.")
        if typ == "thread":
            if last is None:
                return (f"{label}, part 1 of {of}: start a topic that will continue in later sessions (a project, a "
                        "problem, a plan, a purchase, a situation). Give one specific detail in its own record. Report "
                        f"the topic and the detail under {label}.")
            so_far = "; ".join(f"{v['date']}: {v['value']}" for v in vals)
            return (f"{label}, part {k} of {of}: continue the topic \"{item['subject']}\" (so far: {so_far}). Add one "
                    "new specific detail that builds on what came before, in its own record, without repeating an "
                    f"earlier detail. Report the new detail under {label}.")
        if last is None:
            return (f"{label}, part 1 of {of}: show a consistent preference through what the person does or chooses "
                    "(what they order, buy, pick, skip or redo), in its own record, without ever stating it as a "
                    "preference and without words such as prefer, favorite, like or love about it. Report the "
                    f"preference in plain words under {label}; it is never shown to anyone.")
        return (f"{label}, part {k} of {of}: show the same unstated preference (\"{item['subject']}\") again through "
                f"a different action or choice, in its own record, never stating it. Report what was shown under "
                f"{label}.")

    def erred(self, label: str) -> str:
        """Who made a correction's error: the same seeded draw its part 1 made."""
        return _rng(self.ctx.seed, self.persona.plan_key, "style", label, 1).choice(("the person", "the agent"))

    def state_view(self, phase: str, upto: str) -> Dict[str, Any]:
        """What the model sees of the state for a session on the local date upto: only what
        was established on or before that date, so a make-up session dated earlier than
        later sessions never sees their facts. The tail's list of core facts to avoid is
        the exception: it holds every core fact, whatever its date."""
        st = self.state
        people = [{"name": p["name"], "relation": p["relation"]} for p in st["people"]
                  if p.get("since", "") <= upto]
        facts = [f for f in st["facts"] if f["date"] <= upto]
        if phase == "tail":
            avoid = [f["fact"] for f in st["facts"]]
            matters = []
            for label, item in sorted(st["items"].items(), key=lambda kv: _label_key(kv[0])):
                avoid += [f"{item['subject']}: {v['value']}" for v in item["values"]]
                if item["values"] and item["subject"]:
                    matters.append(item["subject"])
            tail = [f for f in st["tail_facts"] if f["date"] <= upto]
            # The two core lists first: they are the same for every tail session, so they sit in the cached prefix.
            return {"core_facts_never_restate": avoid, "core_matters_never_touch": matters, "people": people,
                    "tail_facts_so_far": [f"{f['date']}: {f['fact']}" for f in tail[-FACTS_IN_STATE:]]}
        open_items, never = {}, []
        for label, item in sorted(st["items"].items(), key=lambda kv: _label_key(kv[0])):
            values = [v for v in item["values"] if v["date"] <= upto]
            if not values:
                continue
            if item["type"] == "long_horizon":
                never.append(f"{item['subject']}: {values[0]['value']}")
                continue
            open_items[label] = {"type": item["type"], "subject": item["subject"],
                                 "values": [{k: v[k] for k in ("date", "value", "note") if v.get(k)} for v in values]}
        return {"people": people,
                "facts_so_far": [f"{f['date']}: {f['fact']}" for f in facts[-FACTS_IN_STATE:]],
                "open_items": open_items, "never_mention_again": never}

    def user_message(self, session: Session, want: int, directives: List[Dict[str, Any]]) -> Tuple[str, int]:
        """The user message and where its fixed prefix ends, for prompt caching. The prefix
        is what every session of this persona in this phase shares: the persona, and in the
        tail also the core lists that open the state. The message stays one JSON object; the
        split falls between two keys, so each piece the API caches is the same bytes each call."""
        entries = entries_for(want)
        sess = {"id": session.id, "local_time": session.local.strftime("%Y-%m-%d %H:%M"),
                "weekday": _WEEKDAYS[session.local.weekday()],
                "timeline": f"{self.persona.start.isoformat()} to {self.persona.end.isoformat()}"}
        write = {"records": want, "entries": entries, "user_and_agent_records": want - entries}
        must = [self.directive_text(d) for d in directives]
        state = self.state_view(session.phase, session.local.strftime("%Y-%m-%d"))
        if session.phase == "tail":
            msg = {"persona": self.persona.public, "state": state, "session": sess, "write": write,
                   "must_include": must}
            after, marker = '\n  "state": ', '\n    "people": '  # the state's people, not the persona's
        else:
            msg = {"persona": self.persona.public, "session": sess, "write": write, "must_include": must,
                   "state": state}
            after, marker = '\n  "persona": ', '\n  "session": '
        text = json.dumps(msg, ensure_ascii=False, indent=2)
        # A newline cannot sit inside a JSON string, so each marker is a key at its indent.
        return text, text.index(marker, text.index(after))

    # -- one session --------------------------------------------------------

    def run_session(self, session: Session, want: int) -> int:
        """Writes one session and returns how many of its records were kept."""
        dirs: List[Dict[str, Any]] = []
        seen = set()
        for d in self.carried + session.directives:
            if d["label"] not in seen:
                seen.add(d["label"])
                dirs.append(d)
        self.carried = []
        # A carried shortfall never takes a session past the largest size (or its own planned
        # size, which spreading a short last session can put one over).
        want = max(min(want, max(self.persona.size_range[1], session.records)), len(dirs), 1)
        prompt = self.ctx.prompts[session.phase]
        user, split = self.user_message(session, want, dirs)
        effort = self.ctx.effort[session.phase]
        req = pool_request_sha256(self.ctx.model_id, prompt, user, effort, split)
        label = f"{session.phase} session {session.id} ({self.persona.id})"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            reply = self.ctx.call(self, session, attempt, label, prompt, user, req, effort, split)
            raw, update, problem = parse_session_reply(reply)
            if not problem:
                break
            self.counts["unusable_calls"] += 1
            self.ctx.say(f"  {self.persona.id} {session.phase} {session.id}: {problem} (attempt {attempt})")
        if problem:
            self.counts["sessions_lost"] += 1
            self.carried = dirs
            return 0
        kept_positions = self.take_records(session, raw)
        kept = len(kept_positions)
        self.merge(session, update, {d["label"] for d in dirs}, kept_positions)
        self.counts["sessions"] += 1
        self.ctx.say(f"  {self.persona.id} {session.phase:4} {session.id} {session.local:%Y-%m-%d}: asked {want}, "
                     f"returned {len(raw)}, kept {kept}; {self.persona.id} {session.phase} "
                     f"{len(self.records[session.phase])}/{self.budget[session.phase]}")
        return kept

    def take_records(self, session: Session, raw: List[Any]) -> set:
        """Screens a reply's records and keeps the ones that pass; returns their positions."""
        rng = _rng(self.ctx.seed, session.id, "ts")
        t = session.start
        kept = set()
        for pos, item in enumerate(raw):
            if pos:
                t += timedelta(seconds=rng.randint(*RECORD_GAP_SECONDS))
            rec = {"persona": self.persona.id, "phase": session.phase, "session": session.id, "seq": session.seq,
                   "position": pos, "ts": t.strftime(_TS_FORMAT)}
            reason, detail = None, None
            if not isinstance(item, dict) or set(item) != {"kind", "text"} or item["kind"] not in KINDS \
                    or not isinstance(item["text"], str) or not item["text"].strip():
                reason = "bad_record_shape"
                rec.update(kind=item.get("kind") if isinstance(item, dict) else None,
                           text=item.get("text") if isinstance(item, dict) else None)
            else:
                rec.update(kind=item["kind"], text=item["text"].strip())  # no record ends in blank lines
                reason, detail = self.ctx.screen(self, rec)
            if reason:
                self.drop(rec, reason, detail)
            else:
                self.records[session.phase].append(rec)
                kept.add(pos)
        return kept

    def drop(self, rec: Dict[str, Any], reason: str, detail: Any = None) -> None:
        row = {k: rec.get(k) for k in ("persona", "phase", "session", "position", "ts", "kind", "text")}
        row["id"] = rec.get("id")
        row["reason"], row["detail"] = reason, detail
        if reason == "secret_or_pii":  # never written out, even in the generation record
            row["text"] = None
            row["id"] = None
        self.dropped.append(row)
        self.counts[f"dropped_{reason}"] += 1

    def clean(self, *texts: Any) -> bool:
        """Whether state text may be kept: a line with a secret or personal-data shape is
        left out of the state, so it never reaches states.json or a later request."""
        if any(isinstance(t, str) and lv.scan_secrets_pii(t) for t in texts):
            self.counts["state_lines_screened"] += 1
            return False
        return True

    def merge(self, session: Session, update: Dict[str, Any], labels: set, kept: set) -> None:
        """Folds a reply's state update into the state. A planted step counts only when the
        record it names (by position) was kept, so a later step never builds on a value the
        pool does not hold."""
        when = session.local.strftime("%Y-%m-%d")
        names = {p["name"] for p in self.state["people"]}
        for p in update.get("new_people") or []:
            if isinstance(p, dict) and isinstance(p.get("name"), str) and p["name"].strip() \
                    and p["name"] not in names and isinstance(p.get("relation"), str) \
                    and self.clean(p["name"], p["relation"]):
                self.state["people"].append({"name": p["name"], "relation": p["relation"], "since": when})
                names.add(p["name"])
        facts = self.state["facts" if session.phase == "core" else "tail_facts"]
        for f in update.get("facts") or []:
            if isinstance(f, str) and f.strip() and self.clean(f):
                facts.append({"date": when, "session": session.id, "fact": f.strip()})
        reported = set()
        for ev in update.get("events") or []:
            label = ev.get("label") if isinstance(ev, dict) else None
            if not isinstance(label, str) or label not in labels:
                self.counts["events_not_asked_for"] += 1
                continue
            if label in reported:
                continue
            pos = ev.get("record")
            if isinstance(pos, bool) or not isinstance(pos, int) or pos not in kept:
                self.counts["events_without_a_kept_record"] += 1
                continue
            if not self.clean(ev.get("subject"), ev.get("value"), ev.get("note")):
                continue
            item = self.state["items"][label]
            if not item["subject"] and isinstance(ev.get("subject"), str):
                item["subject"] = ev["subject"].strip()
            item["values"].append({"date": when, "session": session.id, "position": pos,
                                   "value": str(ev.get("value") or "").strip(), "note": str(ev.get("note") or "").strip()})
            reported.add(label)
        self.counts["directives_asked"] += len(labels)
        self.counts["directives_reported"] += len(reported)

    # -- a phase ------------------------------------------------------------

    def run_phase(self, phase: str) -> None:
        """The core carries each session's shortfall to the next; the tail is over-asked in
        its plan instead (plan_tail), so each tail session asks for its planned size."""
        self.carried = []
        carry = 0
        for s in self.sessions[phase]:
            want = s.records + carry
            kept = self.run_session(s, max(want, 1))
            if phase == "core":
                carry = want - kept  # a session asks for at most the largest size, so the rest rides on
        self.make_up(phase)
        if self.carried:  # a lost last session: its planted steps end with the phase, never in the tail
            self.counts["directives_lost"] += len(self.carried)
            self.carried = []

    def make_up(self, phase: str) -> None:
        """Adds ordinary sessions until the persona's share is met: each at most the largest
        session size, dated on a seeded day of the whole timeline (drawn by the weekday
        weights, like every other session), so make-ups never gather at the end and tell a
        ranker which records are core. One round adds at most enough sessions for the
        shortfall, plus MAKEUP_SESSIONS. The one exception: a make-up that carries the
        planted steps of a lost session is dated the day after the persona's last session.
        A make-up is never smaller than the persona's smallest session: one record short
        asks for a whole small session and the surplus is trimmed, so a single record that
        keeps failing a screen cannot cost a call each time (pilot 2's fiftieth record)."""
        lo, hi = self.persona.size_range
        limit = MAKEUP_SESSIONS + -(-max(0, self.budget[phase] - len(self.records[phase])) // hi)
        for _ in range(limit):
            short = self.budget[phase] - len(self.records[phase])
            if short <= 0 or self.ctx.abort.is_set():
                return
            self.makeups += 1
            rng = _rng(self.ctx.seed, self.persona.plan_key, phase, "makeup", self.makeups)
            if self.carried:  # planted steps from a lost session stay after the steps they build on
                last = max(x.local for x in self.sessions[phase])
                day = min(last.date() + timedelta(days=1), self.persona.end - timedelta(days=1))
            else:
                day = pick_days(rng, self.persona, 1)[0]
            local = pick_minute(rng, self.persona, day, self.taken)
            sid = session_id(self.ctx.seed, self.persona, f"{phase}|makeup|{self.makeups}", self.used_ids)
            size = max(lo, min(short, hi))
            s = Session(self.persona, phase, len(self.sessions[phase]), sid, local, size, [], makeup=True)
            self.sessions[phase].append(s)
            self.run_session(s, size)

    def trim(self, phase: str) -> None:
        """A persona over its share drops the surplus. The core drops its newest records,
        newest session first, as the surplus there is a session's worth at most. The tail
        was over-asked on purpose, so its surplus is drawn evenly across its records by a
        seeded generator: dropping only the newest would thin the tail's last months, and
        its ts distribution has to match the core's (GATE_DISTRACTOR_MATCH)."""
        recs = self.records[phase]
        surplus = len(recs) - self.budget[phase]
        if surplus <= 0:
            return
        if phase == "core":
            for _ in range(surplus):
                self.drop(recs.pop(), "over_budget")
            return
        rng = _rng(self.ctx.seed, self.persona.plan_key, "tail trim")
        gone = set(rng.sample(range(len(recs)), surplus))
        for i in sorted(gone):
            self.drop(recs[i], "over_budget")
        self.records[phase] = [r for i, r in enumerate(recs) if i not in gone]


def _label_key(label: str) -> Tuple[int, int]:
    return ("VRCTLS".index(label[0]), int(label[1:]))


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def check_out(out: str, release: bool, resume: bool) -> Tuple[str, str]:
    """Where a pool may be written (docs/pool.md section 9). Returns the resolved folder
    and the pool's name, which is the folder's last path component."""
    real = os.path.realpath(out)
    name = os.path.basename(real)
    if not _POOL_NAME.fullmatch(name):
        raise GenerateError(f"--out {out}: the folder's name becomes the pool's name, so it takes letters, digits, "
                            "dots, dashes and underscores")
    if rb.inside_repo(real):
        if not (release and real.startswith(os.path.realpath(DATA_DIR) + os.sep)):
            raise GenerateError(f"--out {out} is inside the repository. A pool is written outside it; the one "
                                "exception is the public pool under data/, with --release")
    elif release:
        raise GenerateError("--release is for the public pool under data/ in this repository; this --out is outside it")
    run_json = os.path.join(real, "generation", "run.json")
    if resume:
        if not os.path.isfile(run_json):
            raise GenerateError(f"--resume: {out} holds no generation/run.json, so there is no run to pick up")
    elif os.path.exists(real) and (not os.path.isdir(real) or os.listdir(real)):
        raise GenerateError(f"--out {out} is not an empty folder; a pool is never overwritten (to pick up a run that "
                            "stopped, add --resume)")
    return real, name


class _Aborted(Exception):
    """Another persona's call failed; this one stops before its next call."""


class _Stuck(Exception):
    """A persona is still short of its share after every round of make-ups."""


class Run:
    """The shared parts of one run: the model, the call log, the screens."""

    def __init__(self, seed: int, model_id: str, model: Any, prompts: Dict[str, PoolPrompt], out_dir: str,
                 run_info: Dict[str, Any], personas: List[Persona], quiet: bool, resumed: bool):
        self.seed = seed
        self.model_id = model_id
        self.model = model
        self.prompts = prompts
        self.out_dir = out_dir
        self.run_info = run_info
        self.personas = personas
        self.quiet = quiet
        self.calls: List[Dict[str, Any]] = []
        self.gen_dir = os.path.join(out_dir, "generation")
        parts = call_parts(self.gen_dir) if resumed else []
        self.part_path = parts[-1] if parts and CALL_PART.fullmatch(os.path.basename(parts[-1])) else None
        self.out_ready = resumed
        self.lock = threading.Lock()
        self.abort = threading.Event()
        self.effort = {"core": DEFAULT_EFFORT, "tail": DEFAULT_EFFORT}
        self.core_drop_pm = 0                 # the core's drop rate, per mille, measured before the tail
        self.overask_pm = TAIL_OVERASK_FLOOR  # how far over its share the tail is planned
        self.cover: Optional[RestateIndex] = None   # the final core, in the tail phase
        self.near: Dict[str, NearCopyIndex] = {}  # per persona while it runs
        self.ids: Dict[str, set] = {}
        self.settled: Dict[str, Tuple[set, NearCopyIndex]] = {}
        self.scenarios = SCENARIOS_DIR

    def say(self, *a: Any) -> None:
        if not self.quiet:
            with self.lock:
                print(*a, flush=True)

    def call(self, run: PersonaRun, session: Session, attempt: int, label: str, prompt: PoolPrompt, user: str,
             req: str, effort: str = DEFAULT_EFFORT, split: int = 0) -> gq.Reply:
        if self.abort.is_set():
            raise _Aborted()
        try:
            reply = self.model.call(label, prompt, user, req, effort=effort, split=split)
        except (GenerateError, Paused):
            self.abort.set()
            raise
        call = {"key": [PHASES.index(session.phase), run.persona.n, session.seq, attempt],
                "persona": run.persona.id, "phase": session.phase, "session": session.id, "attempt": attempt,
                "request_sha256": req, "model": self.model_id, "effort": effort, "cache_split": split,
                "served_model": reply.model, "stop_reason": reply.stop_reason, "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cache_creation_input_tokens": getattr(reply, "cache_write", 0) or 0,
                "cache_read_input_tokens": getattr(reply, "cache_read", 0) or 0,
                "started": reply.started, "time": reply.time, "sdk": reply.sdk, "user": user, "response": reply.text}
        with self.lock:
            if not self.out_ready:  # nothing is written until the first reply is in hand
                self.prepare_out()
            self.calls.append(call)
            if not getattr(reply, "in_log", False):  # the log is only ever appended to while a run is going
                self.append_call(json.dumps(call, ensure_ascii=False) + "\n")
        return reply

    def append_call(self, line: str) -> None:
        """Appends one call to the log's current part, starting the next part first when
        this line would take the current one past CALL_PART_BYTES."""
        size = len(line.encode("utf-8"))
        if self.part_path is None or (os.path.getsize(self.part_path) + size > CALL_PART_BYTES
                                      and os.path.getsize(self.part_path) > 0):
            n = int(CALL_PART.fullmatch(os.path.basename(self.part_path)).group(1)) + 1 if self.part_path else 1
            self.part_path = os.path.join(self.gen_dir, f"calls-{n:04d}.jsonl")
        with open(self.part_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)

    def prepare_out(self) -> None:
        gen = self.gen_dir
        for sub in ("prompts", "scenarios"):
            os.makedirs(os.path.join(gen, sub), exist_ok=True)
        for p in self.prompts.values():
            with open(os.path.join(gen, "prompts", p.name), "wb") as f:
                f.write(p.data)
        for p in self.personas:
            with open(os.path.join(gen, "scenarios", p.file), "wb") as f:
                f.write(p.data)
        gq._dump(os.path.join(gen, "run.json"), self.run_info)
        self.out_ready = True

    def screen(self, run: PersonaRun, rec: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        """The screens on one record while its persona runs. Cross-persona copies are
        caught by settle(), in a fixed order, so threads never decide what is kept."""
        text = rec["text"]
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return "not_unicode", None
        toks, digits, caps = features(rec)
        if len(toks) < MIN_TOKENS:
            return "too_short", len(toks)
        if rec["kind"] == "entry" and not ENTRY_WORDS[0] <= len(text.split()) <= ENTRY_WORDS[1]:
            return "entry_length", len(text.split())
        hits = lv.scan_secrets_pii(text)
        if hits:
            return "secret_or_pii", sorted({h["shape"] for h in hits})
        rid = rc.record_id(text)
        rec["id"] = rid
        ids = self.ids.setdefault(run.persona.id, set())
        if rid in ids:
            return "duplicate_id", rid
        near = self.near.setdefault(run.persona.id, NearCopyIndex())
        hit = near.find(toks, digits, caps)
        if hit:
            return "near_copy", hit
        if rec["phase"] == "tail" and self.cover is not None:
            hit = self.cover.find(toks)  # content tokens, both directions (RestateIndex)
            if hit:
                return "restates_core", hit
        ids.add(rid)
        near.add(rid, toks, digits, caps)
        return None, None

    def parallel(self, runs: List[PersonaRun], job: Callable[[PersonaRun], None], workers: int) -> None:
        errors: List[BaseException] = []

        def wrap(run: PersonaRun) -> None:
            try:
                job(run)
            except _Aborted:
                pass
            except BaseException as exc:  # the first failure stops the rest at their next call
                self.abort.set()
                errors.append(exc)

        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            pending = {pool.submit(wrap, r) for r in runs}
            while pending:  # a wait with a timeout, so Ctrl+C is seen on Windows too
                _, pending = wait(pending, timeout=0.5)
        except KeyboardInterrupt:
            self.abort.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        if errors:
            raise errors[0]

    def settle(self, runs: List[PersonaRun], phase: str, seeded: Iterable[Dict[str, Any]]) -> None:
        """The whole-pool screen, in a fixed order (persona, then session, then position):
        a record whose id or near copy came earlier in that order is dropped. A later round
        screens only the records added since (the make-ups) against everything kept, so a
        record kept in one round is never dropped in the next."""
        if phase not in self.settled:
            index = NearCopyIndex()
            ids: set = set()
            for rec in seeded:
                toks, digits, caps = features(rec)
                ids.add(rec["id"])
                index.add(rec["id"], toks, digits, caps)
            self.settled[phase] = (ids, index)
        ids, index = self.settled[phase]
        for run in runs:
            keep = []
            for rec in run.records[phase]:
                if rec.get("_settled"):
                    keep.append(rec)
                    continue
                rec["_settled"] = True
                toks, digits, caps = features(rec)
                if rec["id"] in ids:
                    run.drop(rec, "duplicate_id", rec["id"])
                    continue
                hit = index.find(toks, digits, caps)
                if hit:
                    run.drop(rec, "near_copy", hit)
                    continue
                ids.add(rec["id"])
                index.add(rec["id"], toks, digits, caps)
                keep.append(rec)
            run.records[phase] = keep


def parse_effort(text: str) -> Dict[str, str]:
    """--effort: one level for both phases, or core=LEVEL,tail=LEVEL (a phase left out gets the default)."""
    text = str(text).strip()
    if text in EFFORTS:
        return {"core": text, "tail": text}
    out = {"core": DEFAULT_EFFORT, "tail": DEFAULT_EFFORT}
    for part in text.split(","):
        phase, sep, level = part.strip().partition("=")
        if not sep or phase.strip() not in PHASES or level.strip() not in EFFORTS:
            raise GenerateError(f"--effort: cannot read {part.strip()!r}; give one of {', '.join(EFFORTS)}, or "
                                "core=LEVEL,tail=LEVEL such as core=high,tail=medium")
        out[phase.strip()] = level.strip()
    return out


def estimate(runs: List["PersonaRun"], model: Any, model_id: str, effort: Dict[str, str]) -> Dict[str, Any]:
    """The calls, tokens and cost a run is likely to need, from the plan and the pilots'
    averages (EST_*), before any call. Calls the log already holds replay for free."""
    recorded = Counter()
    for queue in getattr(model, "recorded", {}).values():
        for rec in queue:
            recorded[rec.get("phase", "core")] += 1
    out: Dict[str, Any] = {"phases": {}, "model": model_id}
    total_in = total_cached = total_out = 0
    for phase in PHASES:
        if phase == "core":
            sessions = sum(len(r.sessions["core"]) for r in runs)
            records = sum(r.budget["core"] for r in runs)
        else:  # planned after the core; sized here at the least over-ask
            records = sum(r.budget["tail"] + -(-r.budget["tail"] * TAIL_OVERASK_FLOOR // 1000) for r in runs)
            sessions = sum(-(-(r.budget["tail"] + -(-r.budget["tail"] * TAIL_OVERASK_FLOOR // 1000)) * 2
                             // sum(r.persona.size_range)) for r in runs if r.budget["tail"])
        calls = sessions + -(-sessions * int(EST_EXTRA_CALLS * 100) // 100)
        new_calls = max(0, calls - recorded[phase])
        share_new = new_calls / calls if calls else 0.0
        inp = int(new_calls * EST_INPUT_PER_CALL[phase])
        cached = int(new_calls * EST_CACHED_PER_CALL[phase])
        outp = int(records * EST_OUTPUT_PER_RECORD * share_new)
        out["phases"][phase] = {"sessions": sessions, "calls": calls, "recorded": recorded[phase],
                                "new_calls": new_calls, "effort": effort[phase], "input_tokens": inp,
                                "cached_input_tokens": cached, "output_tokens": outp}
        total_in, total_cached, total_out = total_in + inp, total_cached + cached, total_out + outp
    out.update(input_tokens=total_in, cached_input_tokens=total_cached, output_tokens=total_out,
               cost_usd=cost_usd(model_id, total_in - total_cached, total_out, 0, total_cached))
    return out


cost_usd = gq.cost_usd  # at the prices in PRICES, rounded to the cent


def print_estimate(est: Dict[str, Any], say: Callable[..., None]) -> None:
    for phase, e in est["phases"].items():
        say(f"  estimate, {phase}: {e['sessions']} sessions, about {e['calls']} calls at effort {e['effort']}"
            + (f" ({e['recorded']} already recorded, replayed free)" if e["recorded"] else ""))
    cost = f"about ${est['cost_usd']:.2f}" if est["cost_usd"] is not None else "no price on file for this model"
    say(f"  estimate, all: about {est['input_tokens']:,} input tokens ({est['cached_input_tokens']:,} of them read "
        f"from the cache) and {est['output_tokens']:,} output tokens; {cost} on {est['model']}. Rough: from the "
        "pilots' averages at effort high; medium writes less.")


def check_api(model_id: str = DEFAULT_MODEL, api_key_env: Optional[str] = None) -> str:
    """--check-api: one tiny call. Returns the served model's id; raises GenerateError
    with only the error's class name, never its text and never the key. api_key_env
    names the variable holding the key, as for a run."""
    return PoolAnthropicModel(model_id, api_key_env).check()


def generate_pool(scenarios: str, out: str, seed: int, core: int = DEFAULT_CORE, total: int = DEFAULT_TOTAL,
                  model_id: str = DEFAULT_MODEL, personas: Optional[Sequence[str]] = None, workers: int = DEFAULT_WORKERS,
                  release: bool = False, model: Any = None, replay: Optional[str] = None, resume: bool = False,
                  prompts_dir: str = PROMPTS_DIR, quiet: bool = False, effort: Optional[Dict[str, str]] = None,
                  max_calls: Optional[int] = None, estimate_only: bool = False,
                  api_key_env: Optional[str] = None) -> Dict[str, Any]:
    """Plan, write, screen and assemble one pool into out (docs/pool.md). model is anything
    with call(label, prompt, user, request_sha, effort=, split=) -> Reply; by default the
    API, the recorded responses when replay names a call log, or both under resume.
    max_calls stops the run before its next new call once that many were made (Paused);
    estimate_only prints the estimate and returns before any call. api_key_env names the
    variable the API key is read from (PoolAnthropicModel); it is never recorded."""
    say = (lambda *a, **k: None) if quiet else print
    effort = dict(effort or {"core": DEFAULT_EFFORT, "tail": DEFAULT_EFFORT})
    if max_calls is not None and max_calls < 1:
        raise GenerateError("--max-calls must be at least 1")
    if replay and resume:
        raise GenerateError("--replay and --resume do not go together: replay rebuilds into a new folder, resume "
                            "continues a folder")
    if core < 1:
        raise GenerateError("--core must be at least 1")
    if total < core:
        raise GenerateError(f"--total {total} is smaller than --core {core}; the core is the first part of the pool")
    out_dir, pool_name = check_out(out, release, resume)
    chosen = load_personas(scenarios, personas)
    prompts = load_prompts(prompts_dir)
    core_shares = split_evenly(core, len(chosen))
    tail_shares = split_evenly(total - core, len(chosen))
    run_info = {"pool": pool_name, "seed": seed, "model": model_id, "core": core, "total": total,
                "effort": effort, "personas": [p.id for p in chosen], "scenarios": {p.id: p.sha256 for p in chosen},
                "prompts": {p.name: p.sha256 for p in prompts.values()}}
    gen_dir = os.path.join(out_dir, "generation")
    if resume:
        with open(os.path.join(gen_dir, "run.json"), encoding="utf-8") as f:
            before = json.load(f)
        differ = [k for k in ("seed", "model", "core", "personas", "scenarios", "prompts") if before.get(k) != run_info[k]]
        if (before.get("effort") or {}).get("core", DEFAULT_EFFORT) != effort["core"]:
            differ.append("core effort")
        if differ:
            raise GenerateError(f"--resume: this run's {', '.join(differ)} differ from the recorded run in {out}; "
                                "resume with the same settings, or start a new --out")
    if model is None:
        live = None if replay else PoolAnthropicModel(model_id, api_key_env)
        if live is not None and max_calls is not None:
            live = LimitedModel(live, max_calls)
        if replay:
            model = RecordedModel(replay)
        elif resume:
            model = RecordedModel(gen_dir, live)
        else:
            model = live
    else:
        if max_calls is not None:
            model = LimitedModel(model, max_calls)
        if resume:
            model = RecordedModel(gen_dir, model)

    ctx = Run(seed, model_id, model, prompts, out_dir, run_info, chosen, quiet, resume)
    ctx.scenarios = scenarios
    ctx.effort = effort
    runs = [PersonaRun(ctx, p, c, t) for p, c, t in zip(chosen, core_shares, tail_shares)]
    est = estimate(runs, model, model_id, effort)
    say(f"Pool {pool_name}: {len(runs)} personas ({', '.join(p.id for p in chosen)}), {core} core records and "
        f"{total - core} tail records over about {sum(e['sessions'] for e in est['phases'].values())} sessions, "
        f"written by {model_id}" + (f"; this run stops after {max_calls} new calls" if max_calls else ""))
    print_estimate(est, say)
    if estimate_only:
        return {"out": out_dir, "pool": pool_name, "estimate": est}
    if hasattr(model, "connect"):
        model.connect()
    if resume and model.skipped_lines:  # cut the line a stopped run was writing, so new calls start clean
        _write_lines(model.last_part, model.last_lines)
    if resume:
        say(f"  resuming from {sum(len(q) for q in model.recorded.values())} recorded calls")

    try:
        for phase in PHASES:
            todo = [r for r in runs if r.budget[phase]]
            if not todo:
                continue
            if phase == "tail":
                core_recs = [rec for r in runs for rec in r.records["core"]]
                dropped = sum(1 for r in runs for d in r.dropped if d["phase"] == "core")
                ctx.core_drop_pm = dropped * 1000 // max(1, dropped + len(core_recs))
                ctx.overask_pm = max(TAIL_OVERASK_FLOOR, ctx.core_drop_pm + TAIL_OVERASK_MARGIN)
                for r in todo:
                    r.plan_tail(ctx.overask_pm)
                say(f"  tail: core drop rate {ctx.core_drop_pm / 10:.1f}%, so the tail is planned "
                    f"{ctx.overask_pm / 10:.1f}% over its share ({sum(r.tail_planned for r in todo)} records for "
                    f"{sum(r.budget['tail'] for r in todo)}) and trimmed to it")
                ctx.cover = RestateIndex()
                for rec in core_recs:
                    ctx.cover.add(rec["id"], features(rec)[0])
                ctx.near, ctx.ids = {}, {}
            ctx.parallel(todo, lambda r: r.run_phase(phase), workers)
            for rnd in range(MAKEUP_ROUNDS + 1):
                ctx.settle(runs, phase, [rec for r in runs for rec in r.records["core"]] if phase == "tail" else [])
                short = [r for r in todo if len(r.records[phase]) < r.budget[phase]]
                if not short:
                    break
                if rnd == MAKEUP_ROUNDS:
                    raise _Stuck(", ".join(
                        f"{r.persona.id} has {len(r.records[phase])} of its {r.budget[phase]} {phase} records "
                        f"({r.counts['sessions_lost']} sessions lost, {r.counts['unusable_calls']} unusable replies, "
                        f"{sum(v for k, v in r.counts.items() if k.startswith('dropped_'))} records dropped)"
                        for r in short))
                ctx.parallel(short, lambda r: r.make_up(phase), workers)
            for r in todo:
                r.trim(phase)
    except _Stuck as exc:
        raise GenerateError(f"a persona could not reach its share after {MAKEUP_ROUNDS} rounds of make-up sessions: "
                            f"{exc}. {out_dir} holds every call; a --resume would replay the same replies and stop "
                            "the same way, so look at the call log in generation/ first")
    except Paused as exc:
        m, made = ctx.model, None
        while m is not None and made is None:  # the LimitedModel sits under a RecordedModel on a resume
            made = m.made if isinstance(m, LimitedModel) else None
            m = getattr(m, "live", None)
        raise Paused(f"{exc}: {made if made is not None else max_calls} new calls made this run, every one recorded "
                     f"in {gen_dir}. Commit and push the folder, then run the same command with --resume to go on")
    except (GenerateError, KeyboardInterrupt) as exc:
        if not ctx.out_ready:
            raise
        what = "stopped" if isinstance(exc, KeyboardInterrupt) else str(exc)
        raise GenerateError(f"{what}. {out_dir} holds the calls made so far and is not a pool yet; run the same "
                            "command again with --resume to pick up from them")

    return assemble(ctx, runs, out_dir, pool_name, seed, core, total, model_id, release)


# ---------------------------------------------------------------------------
# Assembly: the file order, the record checks, the manifest
# ---------------------------------------------------------------------------

def _public_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": rec["id"], "text": rec["text"], "source": rec["session"], "kind": rec["kind"], "ts": rec["ts"]}


def tail_order(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """The tail's file order: records grouped by (persona, month of ts), each group sorted
    and then shuffled by a seeded generator, and the groups interleaved in proportion to
    their sizes, so every prefix is a representative slice of personas and months."""
    rng = _rng(seed, "tail order")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[f"{rec['persona']}|{rec['ts'][:7]}"].append(rec)
    keyed: Dict[str, List[int]] = {}
    flat: List[Dict[str, Any]] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: (r["ts"], r["seq"], r["position"]))
        rng.shuffle(members)
        keyed[key] = list(range(len(flat), len(flat) + len(members)))
        flat.extend(members)
    return [flat[i] for i in gq.proportional_order(keyed)]


def check_shapes(ordered: List[Dict[str, Any]]) -> None:
    """Every record's id is its content id, its ts a readable ISO UTC time, its source a
    session id, its kind one of the three; no id twice. A failure here is a bug."""
    seen = set()
    for n, rec in enumerate(ordered, 1):
        problems = []
        if rec["id"] != rc.record_id(rec["text"]):
            problems.append("id is not the content id of its text")
        if rec["id"] in seen:
            problems.append("id appears twice")
        if rc.parse_ts(rec["ts"]) is None or not rec["ts"].endswith("Z"):
            problems.append("ts is not an ISO UTC time")
        if not _SESSION_ID.fullmatch(rec["session"]):
            problems.append("source is not a session id")
        if rec["kind"] not in KINDS:
            problems.append("kind is not user, agent or entry")
        if problems:
            raise GenerateError(f"record {n} ({rec['id']}): " + "; ".join(problems))
        seen.add(rec["id"])


def _write_lines(path: str, lines: Iterable[str]) -> None:
    """Writes a file whole and then puts it in place, so a stop partway never leaves a
    half-written file where the only copy of the calls was."""
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp, path)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    _write_lines(path, (json.dumps(row, ensure_ascii=False) for row in rows))


def write_call_parts(gen: str, calls: List[Dict[str, Any]]) -> List[str]:
    """The finished call log, in order, as calls-0001.jsonl, calls-0002.jsonl, ... each
    under CALL_PART_BYTES, written whole and then moved into place. Parts left over from
    the running log (or a log from before parts) are removed only after the new ones are
    all in place, so a stop partway never loses a call."""
    old = set(call_parts(gen))
    parts: List[List[str]] = [[]]
    size = 0
    for c in calls:
        line = json.dumps(c, ensure_ascii=False)
        n = len(line.encode("utf-8")) + 1
        if parts[-1] and size + n > CALL_PART_BYTES:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += n
    paths = []
    for i, lines in enumerate(parts, 1):
        path = os.path.join(gen, f"calls-{i:04d}.jsonl")
        _write_lines(path, lines)
        paths.append(path)
    for path in sorted(old - set(paths)):
        os.remove(path)
    return paths


def assemble(ctx: Run, runs: List[PersonaRun], out_dir: str, pool_name: str, seed: int, core: int, total: int,
             model_id: str, release: bool) -> Dict[str, Any]:
    core_recs = sorted((rec for r in runs for rec in r.records["core"]),
                       key=lambda rec: (rec["ts"], int(rec["persona"][1:]), rec["seq"], rec["position"]))
    if len(core_recs) != core:
        raise GenerateError(f"the core holds {len(core_recs)} records, not {core}")
    tail_recs = tail_order([rec for r in runs for rec in r.records["tail"]], seed)
    ordered = core_recs + tail_recs
    check_shapes(ordered)

    gen = os.path.join(out_dir, "generation")
    if not ctx.out_ready:
        ctx.prepare_out()
    gq._dump(os.path.join(gen, "run.json"), ctx.run_info)  # a resume may have grown the total
    calls = sorted(ctx.calls, key=lambda c: c["key"])
    for i, c in enumerate(calls, 1):
        c["n"] = i
    part_paths = write_call_parts(gen, calls)
    call_n = {(c["persona"], c["session"]): c["n"] for c in calls}  # the last attempt is the one that counted
    c_path = os.path.join(out_dir, "corpus.jsonl")
    _write_jsonl(c_path, (_public_record(rec) for rec in ordered))
    rb.load_corpus(c_path)  # the runner's own loader takes it, or this raises

    r_path = os.path.join(gen, "records.jsonl")
    _write_jsonl(r_path, ({"line": i, "id": rec["id"], "persona": rec["persona"], "phase": rec["phase"],
                           "session": rec["session"], "position": rec["position"],
                           "call": call_n.get((rec["persona"], rec["session"]))}
                          for i, rec in enumerate(ordered, 1)))
    dropped = [row for r in runs for row in r.dropped]
    d_path = os.path.join(gen, "dropped.jsonl")
    _write_jsonl(d_path, dropped)
    plan = {"seed": seed, "personas": {}}
    for r in runs:
        plan["personas"][r.persona.id] = {
            "name": r.persona.name, "file": r.persona.file, "core_share": r.budget["core"],
            "tail_share": r.budget["tail"], "quotas": r.quotas, "not_placed": r.lost,
            "sessions": [s.plan_json() for ph in PHASES for s in r.sessions[ph]]}
    p_path = os.path.join(gen, "plan.json")
    gq._dump(p_path, plan)
    s_path = os.path.join(gen, "states.json")
    gq._dump(s_path, {r.persona.id: r.state for r in runs})

    files = {"corpus.jsonl": c_path, "generation/records.jsonl": r_path, "generation/dropped.jsonl": d_path,
             "generation/plan.json": p_path, "generation/states.json": s_path}
    files.update({f"generation/{os.path.basename(p)}": p for p in part_paths})
    manifest = build_manifest(ctx, runs, ordered, core_recs, tail_recs, dropped, calls, pool_name, seed, core, total,
                              model_id, release, files)
    manifest["generation"] = {"call_log": [f"generation/{os.path.basename(p)}" for p in part_paths],
                              "call_log_part_bytes": CALL_PART_BYTES,
                              "call_log_note": "read the parts in order; --replay and --resume take the folder"}
    gq._dump(os.path.join(out_dir, "manifest.json"), manifest)
    return {"out": out_dir, "pool": pool_name, "manifest": manifest, "records": ordered}


def percentiles(values: Sequence[float]) -> Dict[str, Any]:
    """min, the nearest-rank percentiles, max and the mean (4 decimals)."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    out: Dict[str, Any] = {"n": n, "min": v[0]}
    for p in PERCENTILES:
        out[f"p{p}"] = v[max(0, -(-p * n // 100) - 1)]
    out["max"] = v[-1]
    out["mean"] = round(sum(v) / n, 4)
    return out


def _share(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def pool_statistics(ordered: List[Dict[str, Any]], core_recs: List[Dict[str, Any]],
                    dropped: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the manifest reports about the pool, so a real store's statistics can sit
    beside it: kinds, lengths, sessions, active days, entries per active day, months,
    near copies and the restatement rate."""
    def kinds(recs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        c = Counter(r["kind"] for r in recs)
        return {k: c.get(k, 0) for k in KINDS}

    core_ids = {r["id"] for r in core_recs}
    tail_recs = [r for r in ordered if r["id"] not in core_ids]
    words = {"all": [len(r["text"].split()) for r in ordered]}
    for k in KINDS:
        words[k] = [len(r["text"].split()) for r in ordered if r["kind"] == k]
    sessions: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    days: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in ordered:
        sessions[(r["persona"], r["session"])].append(r)
        days[(r["persona"], r["ts"][:10])].append(r)
    by_persona: Dict[str, Any] = {}
    for pid in sorted({r["persona"] for r in ordered}, key=lambda p: int(p[1:])):
        recs = [r for r in ordered if r["persona"] == pid]
        ts = sorted(r["ts"] for r in recs)
        by_persona[pid] = {"records": len(recs), "core": sum(1 for r in recs if r["id"] in core_ids),
                           "tail": sum(1 for r in recs if r["id"] not in core_ids), **kinds(recs),
                           "sessions": sum(1 for (p, _) in sessions if p == pid),
                           "active_days": sum(1 for (p, _) in days if p == pid), "first_ts": ts[0], "last_ts": ts[-1]}

    near = NearCopyIndex()
    clustered: Dict[str, str] = {}
    cover = RestateIndex()
    restating = {"all": 0, "core": 0}
    for r in sorted(ordered, key=lambda r: (r["ts"], int(r["persona"][1:]), r["seq"], r["position"])):
        toks, digits, caps = features(r)
        hit = near.find(toks, digits, caps)
        if hit:
            clustered[r["id"]] = clustered.get(hit, hit)
        near.add(r["id"], toks, digits, caps)
        if cover.find(toks):
            restating["all"] += 1
            restating["core"] += r["id"] in core_ids
        cover.add(r["id"], toks)
    core_cover = RestateIndex()
    core_restating = 0
    for r in core_recs:
        toks = features(r)[0]
        core_restating += core_cover.find(toks) is not None
        core_cover.add(r["id"], toks)

    reasons = Counter(d["reason"] for d in dropped)
    return {
        "records": len(ordered), "core": len(core_recs), "tail": len(tail_recs),
        "by_kind": {"all": kinds(ordered), "core": kinds(core_recs), "tail": kinds(tail_recs)},
        "entry_share": {"all": _share(kinds(ordered)["entry"], len(ordered)),
                        "core": _share(kinds(core_recs)["entry"], len(core_recs)),
                        "tail": _share(kinds(tail_recs)["entry"], len(tail_recs)),
                        "in_range": 0.10 <= (_share(kinds(ordered)["entry"], len(ordered)) or 0) <= 0.20,
                        "range": [0.10, 0.20]},
        "by_persona": by_persona,
        "words": {k: percentiles(v) for k, v in words.items()},
        "characters": percentiles([len(r["text"]) for r in ordered]),
        "sessions": {"count": len(sessions),
                     "core": len({s for s, recs in sessions.items() if recs[0]["id"] in core_ids}),
                     "tail": len({s for s, recs in sessions.items() if recs[0]["id"] not in core_ids}),
                     "records_per_session": percentiles([len(v) for v in sessions.values()])},
        "active_days": {"count": len(days), "records_per_active_day": percentiles([len(v) for v in days.values()]),
                        "entries_per_active_day": percentiles([sum(1 for r in v if r["kind"] == "entry")
                                                               for v in days.values()]),
                        "sessions_per_active_day": percentiles([len({r["session"] for r in v})
                                                                for v in days.values()]),
                        "day": "the UTC date of ts, per persona"},
        "records_by_month": dict(sorted(Counter(r["ts"][:7] for r in ordered).items())),
        "near_copy_clusters": len(set(clustered.values())),
        "restatement_rate": {
            "all": _share(restating["all"], len(ordered)),
            "core_against_the_whole_pool": _share(restating["core"], len(core_recs)),
            "core_against_the_core": _share(core_restating, len(core_recs)),
            "definition": f"the share of records that restate an earlier record (earlier by ts): on content tokens "
                          f"(rc.STOPWORDS left out), each covers at least {RESTATE_OVERLAP} of the other's "
                          f"(rc.token_overlap both ways), and the earlier record has at least {RESTATE_MIN_CONTENT} "
                          "content tokens; the same rule as the restates_core screen"},
        "dropped": {"total": len(dropped), "by_reason": {k: reasons.get(k, 0) for k in DROP_REASONS},
                    "by_phase": {ph: sum(1 for d in dropped if d["phase"] == ph) for ph in PHASES},
                    "rate_by_phase": {ph: _share(sum(1 for d in dropped if d["phase"] == ph),
                                                 sum(1 for d in dropped if d["phase"] == ph)
                                                 + (len(core_recs) if ph == "core" else len(tail_recs)))
                                      for ph in PHASES},
                    "rate": "dropped over returned (kept plus dropped)"},
    }


def build_manifest(ctx: Run, runs: List[PersonaRun], ordered: List[Dict[str, Any]], core_recs: List[Dict[str, Any]],
                   tail_recs: List[Dict[str, Any]], dropped: List[Dict[str, Any]], calls: List[Dict[str, Any]],
                   pool_name: str, seed: int, core: int, total: int, model_id: str, release: bool,
                   files: Dict[str, str]) -> Dict[str, Any]:
    commit = rb.harness_commit()
    served = sorted({c["served_model"] for c in calls if c["served_model"]})
    all_personas = sorted(n for n in os.listdir(ctx.scenarios) if n.endswith(".json"))
    reasons = []
    if core != DEFAULT_CORE:
        reasons.append(f"core {core}, not {DEFAULT_CORE}")
    if total != DEFAULT_TOTAL:
        reasons.append(f"total {total}, not {DEFAULT_TOTAL}")
    if len(runs) != len(all_personas):
        reasons.append(f"{len(runs)} of the {len(all_personas)} persona files in scenarios/")
    counts = Counter()
    for r in runs:
        counts.update(r.counts)
    return {
        "pool": pool_name,
        "spec": rc.SPEC_VERSION,
        "status": "partial" if reasons else "complete",
        "status_reason": ("partial: " + "; ".join(reasons)) if reasons else "complete: every persona, the standard "
                                                                            "core and the standard total",
        "id_scheme": "content",
        "license": "MIT",
        "files": {k: rc.file_sha256(v) for k, v in files.items()},
        "layout": {"core": {"first_line": 1, "last_line": core, "order": "ts, then persona, session, position"},
                   "tail": {"first_line": core + 1, "last_line": len(ordered),
                            "order": "seeded: (persona, month) groups, each shuffled, interleaved by size"},
                   "tracks": {name: {"N": n, "available": len(ordered) >= n and core <= n,
                                     "records": "the first N lines of corpus.jsonl"} for name, n in TRACKS}},
        "model": {"requested": model_id, "served": served, "effort": ctx.effort,
                  "effort_by_call": dict(sorted(Counter(c.get("effort", DEFAULT_EFFORT) for c in calls).items())),
                  "max_tokens": POOL_MAX_TOKENS, "streaming": True,
                  "prompt_caching": "the system prompt, and the user message up to cache_split (the persona; in the "
                                    "tail also the core lists)",
                  "server_side_fallbacks": False, "sdk": next((c["sdk"] for c in calls if c["sdk"]), None),
                  "started": min((c["started"] for c in calls), default=None),
                  "finished": max((c["time"] for c in calls), default=None),
                  "calls": len(calls), "input_tokens": sum(c["input_tokens"] for c in calls),
                  "output_tokens": sum(c["output_tokens"] for c in calls),
                  "cache_creation_input_tokens": sum(c.get("cache_creation_input_tokens", 0) for c in calls),
                  "cache_read_input_tokens": sum(c.get("cache_read_input_tokens", 0) for c in calls),
                  "cost_usd": cost_usd(model_id, sum(c["input_tokens"] for c in calls),
                                       sum(c["output_tokens"] for c in calls),
                                       sum(c.get("cache_creation_input_tokens", 0) for c in calls),
                                       sum(c.get("cache_read_input_tokens", 0) for c in calls)),
                  "cost_note": "from the call log's token counts at the prices in PRICES; the bill is the authority",
                  "unusable_calls": counts["unusable_calls"], "sessions_lost": counts["sessions_lost"]},
        "prompts": {p.name: {"file": f"prompts/pool/{p.name}", "sha256": p.sha256} for p in ctx.prompts.values()},
        "scenarios": {r.persona.id: {"file": f"scenarios/{r.persona.file}", "sha256": r.persona.sha256,
                                     "name": r.persona.name} for r in runs},
        "plan": {"seed": seed, "core": core, "total": total, "personas": [r.persona.id for r in runs],
                 "core_shares": {r.persona.id: r.budget["core"] for r in runs},
                 "tail_shares": {r.persona.id: r.budget["tail"] for r in runs},
                 "quotas": {r.persona.id: r.quotas for r in runs},
                 "quota_reference": QUOTA_REFERENCE, "not_placed": {r.persona.id: r.lost for r in runs if r.lost},
                 "sessions_planned": sum(1 for r in runs for ph in PHASES for s in r.sessions[ph] if not s.makeup),
                 "makeup_sessions": sum(1 for r in runs for ph in PHASES for s in r.sessions[ph] if s.makeup),
                 "directives_asked": counts["directives_asked"], "directives_reported": counts["directives_reported"],
                 "events_without_a_kept_record": counts["events_without_a_kept_record"],
                 "events_not_asked_for": counts["events_not_asked_for"], "directives_lost": counts["directives_lost"],
                 "entry_share": ENTRY_SHARE, "entry_words": list(ENTRY_WORDS), "record_gap_seconds":
                     list(RECORD_GAP_SECONDS), "session_gap_minutes": SESSION_GAP_MINUTES,
                 "max_attempts": MAX_ATTEMPTS, "makeup_sessions_per_round": MAKEUP_SESSIONS,
                 "makeup_rounds": MAKEUP_ROUNDS, "facts_in_state": FACTS_IN_STATE,
                 "core_drop_rate_per_mille": ctx.core_drop_pm, "tail_overask_per_mille": ctx.overask_pm,
                 "tail_planned": {r.persona.id: r.tail_planned for r in runs},
                 "tail_overask_rule": f"max({TAIL_OVERASK_FLOOR}, the core's drop rate + {TAIL_OVERASK_MARGIN}) per "
                                      "mille; the surplus is trimmed by a seeded even spread",
                 "makeup_minimum": "a make-up session is never smaller than the persona's smallest session"},
        "screens": {"min_tokens": MIN_TOKENS, "entry_words": list(ENTRY_WORDS),
                    "secrets_and_pii": "lint_validator.scan_secrets_pii, the NO_SECRETS_OR_PII_POOL shapes; a record "
                                       "that matches is dropped, never edited; its text is left out of the corpus, "
                                       "dropped.jsonl and the state (and so of every later request), and stays only "
                                       "in the raw reply in the call log, which replay needs",
                    "state_lines_screened": counts["state_lines_screened"],
                    "calls_with_secret_or_pii_shapes": sum(1 for c in calls if lv.scan_secrets_pii(c["response"])),
                    "near_copy_rule": f"each covers at least {NEAR_COPY_OVERLAP} of the other's tokens, with the same "
                                      "digit tokens and the same capitalized words, against every earlier record of "
                                      "the pool",
                    "restates_core": f"no tail record restates a core record: on content tokens (rc.STOPWORDS left "
                                     f"out), neither may cover {RESTATE_OVERLAP} or more of the other's in both "
                                     f"directions when the core record has at least {RESTATE_MIN_CONTENT} content "
                                     "tokens, so a long entry over a short record is kept and a same-length "
                                     "paraphrase is dropped",
                    "core_exact": f"the core holds exactly {core} records"},
        "statistics": pool_statistics(ordered, core_recs, dropped),
        "tool": {"file": "generate_pool.py", "sha256": gq._lf_sha256(os.path.abspath(__file__)),
                 "harness_commit": commit["commit"], "dirty": commit["dirty"]},
        "provenance": (f"Written by {', '.join(served) or model_id} from the persona files and prompts listed above, "
                       f"with seed {seed}; every model response is in the call log (generation/calls-*.jsonl, in "
                       "order), the screens decided "
                       "what was kept, and no record was edited by hand. Every person in it is invented."),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any]) -> None:
    m = result["manifest"]
    st = m["statistics"]
    print(f"\nWrote {result['out']} ({m['pool']}, status {m['status']})")
    print(f"  {st['records']} records: {st['core']} core, {st['tail']} tail; kinds "
          + ", ".join(f"{k} {v}" for k, v in st["by_kind"]["all"].items())
          + f"; entry share {st['entry_share']['all']}")
    print(f"  {'persona':8}{'records':>9}{'core':>7}{'tail':>7}{'user':>7}{'agent':>7}{'entry':>7}{'sessions':>10}"
          f"{'days':>7}")
    for pid, p in st["by_persona"].items():
        print(f"  {pid:8}{p['records']:>9}{p['core']:>7}{p['tail']:>7}{p['user']:>7}{p['agent']:>7}{p['entry']:>7}"
              f"{p['sessions']:>10}{p['active_days']:>7}")
    drops = {k: v for k, v in st["dropped"]["by_reason"].items() if v}
    print(f"  dropped: {st['dropped']['total']}" + (" (" + ", ".join(f"{k} {v}" for k, v in drops.items()) + ")"
                                                   if drops else ""))
    md = m["model"]
    cost = f", about ${md['cost_usd']:.2f}" if md.get("cost_usd") is not None else ""
    print(f"  model: {md['calls']} calls, {md['input_tokens']} input, {md['cache_read_input_tokens']} cache-read, "
          f"{md['cache_creation_input_tokens']} cache-write and {md['output_tokens']} output tokens{cost}, served by "
          f"{', '.join(md['served']) or 'nothing'}")
    if not st["entry_share"]["in_range"]:
        print(f"  WARNING: the entry share {st['entry_share']['all']} is outside 0.10 to 0.20")
    if m["screens"]["calls_with_secret_or_pii_shapes"]:
        print(f"  note: {m['screens']['calls_with_secret_or_pii_shapes']} raw replies in the call log hold a "
              "secret or personal-data shape (those records were dropped from the pool)")
    print(f"  directives reported: {m['plan']['directives_reported']} of {m['plan']['directives_asked']}; "
          f"near-copy clusters {st['near_copy_clusters']}; restatement rate {st['restatement_rate']['all']}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"Parsica Recall public pool generator ({rc.SPEC_VERSION}); "
                                            "design in docs/pool.md")
    p.add_argument("--scenarios", default=SCENARIOS_DIR, help="Folder of persona files (default: scenarios/)")
    p.add_argument("--personas", default=None,
                   help="Only these persona ids, comma separated, such as p1,p2 (default: every file in the folder)")
    p.add_argument("--out", default=None,
                   help="Folder for the pool; must not exist yet or be empty (unless --resume), and must be outside "
                        "this repository (the public pool under data/ needs --release). Its name becomes the pool's "
                        "name. Required, except with --check-api")
    p.add_argument("--seed", type=int, default=None, help="Plan seed. Required, except with --check-api")
    p.add_argument("--core", type=int, default=DEFAULT_CORE, help=f"Core records (default {DEFAULT_CORE})")
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL, help=f"All records, core included (default "
                                                                     f"{DEFAULT_TOTAL})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Generation model (default {DEFAULT_MODEL})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Personas written at once (default {DEFAULT_WORKERS})")
    p.add_argument("--release", action="store_true", help="Allow an --out under data/, for the public pool")
    p.add_argument("--replay", default=None,
                   help="Rebuild from a recorded call log (a generation/ folder, or one of its calls-NNNN.jsonl "
                        "parts), without the API")
    p.add_argument("--resume", action="store_true",
                   help="Pick up a run that stopped, from the call log in --out; only new calls go to the API")
    p.add_argument("--effort", default=DEFAULT_EFFORT,
                   help=f"Effort for the calls: one of {', '.join(EFFORTS)}, or core=LEVEL,tail=LEVEL such as "
                        f"core=high,tail=medium (default {DEFAULT_EFFORT})")
    p.add_argument("--max-calls", type=int, default=None,
                   help="Stop before the next new API call once this many were made in this run (exit 3); recorded "
                        "calls replayed by --resume do not count. The folder is left ready for --resume")
    p.add_argument("--estimate", action="store_true",
                   help="Print the plan's call count and cost estimate, then stop before any call")
    p.add_argument("--check-api", action="store_true",
                   help="Make one tiny call to prove the key and the network: prints ok and the model that answered, "
                        "or the error's class name, and nothing else")
    p.add_argument("--api-key-env", default=None, metavar="NAME",
                   help="Read the API key from this environment variable instead of ANTHROPIC_API_KEY, and send "
                        "every call to api.anthropic.com whatever ANTHROPIC_BASE_URL says. The key is never "
                        "printed, logged or written, and is taken out of every error")
    args = p.parse_args(argv)
    if args.check_api:
        try:
            served = check_api(args.model, args.api_key_env)
        except GenerateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"ok: {served} answered")
        return 0
    if args.out is None or args.seed is None:
        p.error("--out and --seed are required")
    try:
        wanted = [x.strip() for x in args.personas.split(",") if x.strip()] if args.personas else None
        result = generate_pool(args.scenarios, args.out, args.seed, core=args.core, total=args.total,
                               model_id=args.model, personas=wanted, workers=args.workers, release=args.release,
                               replay=args.replay, resume=args.resume, effort=parse_effort(args.effort),
                               max_calls=args.max_calls, estimate_only=args.estimate,
                               api_key_env=args.api_key_env)
    except Paused as exc:
        print(f"PAUSED: {exc}")
        return 3
    except (GenerateError, rb.InputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.estimate:
        return 0
    print_summary(result)
    return 0


if __name__ == "__main__":
    # Determinism: pin the hash seed and run again, as the runner and the question generator do.
    _status = rc.pin_hash_seed(f"{rc.PACKAGE}.generate_pool", sys.argv[1:])
    sys.exit(main() if _status is None else _status)

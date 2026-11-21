#!/usr/bin/env python3
"""
Parsica Recall runner (spec recall/1.0).

Measures whether a memory system puts the right record in the window an agent
reads. The judge never calls a model: every score is an id match, a normalized
phrase match or a token overlap, so two runs on the same inputs give the same
bytes. The window and the reported k come from the pool size (the band), never
from the system under test.

Usage:
    # A reference adapter on the current question set, run twice and compared
    parsica-recall run --adapter bm25 --verify

    # Your own adapter
    parsica-recall run --adapter custom --custom-adapter my_module.MyAdapter \\
        --corpus data/v11/corpus.jsonl --questions data/v11/questions.json \\
        --output results/my_adapter.json --verify

    # Live mode: a store that already holds its records (spec section 10). Nothing is
    # ingested; results land next to the question file, never in this repository.
    parsica-recall run --live --adapter custom --custom-adapter my_module.MyStore \\
        --questions /path/to/my-qset/questions.json --verify --redact

Scoring tiers, per question (spec section 5):
    gold     rank of the first returned record whose id is a gold id (the headline)
    gold_all rank by which every gold id has appeared (cross_session)
    version  the first gold in the window is a current version (must_rank_first); counted
             only where every gold record carries a ts, otherwise n/a
    exact    first record whose normalized text contains the answer (diagnostic)
    family   first record with token overlap >= 0.6 against a gold text (diagnostic)

Fullstack mode (an answer model reads the window) is not part of this release. It is
on the roadmap, and docs/spec.md section 11 fixes its rules.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import recall_common as rc
from .adapters import (BaseMemoryAdapter, call_ingest, call_setup, describe_version,
                      get_adapter, get_custom_adapter, is_v1_ingest)

REPO_DIR = rc.ROOT_DIR  # the repository's root in a checkout, the package folder in an installed copy
DEFAULT_QSET_DIR = os.path.join(REPO_DIR, "data", rc.CURRENT_QSET)

TIMING_FIELDS = ("search_time_ms",)

# Who ran the system: the harness, through an adapter, or the system's owner, who hands
# over a results file (results_file.py, docs/adapting.md).
SOURCE_ADAPTER = "adapter"
SOURCE_RESULTS_FILE = "results_file"
CONTRACT_RESULTS_FILE = "results-file"


# ---------------------------------------------------------------------------
# Loading and checking inputs
# ---------------------------------------------------------------------------

class InputError(Exception):
    pass


def load_corpus(path: str, store_ids: bool = False) -> List[Dict[str, Any]]:
    """corpus.jsonl: one record per line, all ids unique. Ids are content hashes, unless
    store_ids: a private set generated from a live store may keep the store's own
    stable ids (spec section 3), taken as given."""
    records = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "id" not in rec or "text" not in rec:
                raise InputError(f"{path}:{lineno}: record needs 'id' and 'text'")
            if store_ids:
                if not isinstance(rec["id"], str) or not rec["id"].strip():
                    raise InputError(f"{path}:{lineno}: a store id must be a non-empty string, got {rec['id']!r}")
            elif rec["id"] != rc.record_id(rec["text"]):
                raise InputError(
                    f"{path}:{lineno}: id {rec['id']} is not the record id of its text "
                    f"(expected {rc.record_id(rec['text'])}); a private set that keeps its store's own ids "
                    "runs with --store-ids")
            if rec["id"] in seen:
                raise InputError(f"{path}:{lineno}: duplicate id {rec['id']}")
            seen.add(rec["id"])
            records.append(rec)
    if not records:
        raise InputError(f"{path}: corpus is empty")
    return records


def refuse(source: str, problems: List[str]) -> None:
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise InputError(f"{source}: refusing the question set:\n  {shown}{more}")


def missing_gold(questions: List[Dict[str, Any]], pool_ids: set) -> List[str]:
    out = []
    for q in questions:
        missing = [g for g in q.get("gold_ids") or [] if g not in pool_ids]
        if missing:
            out.append(f"{q.get('id')}: gold ids not in the pool: {missing}")
    return out


def load_questions(path: str, pool_ids: Optional[set]) -> List[Dict[str, Any]]:
    """Reads and checks a question file. With pool_ids it also refuses any question
    whose gold ids are not all in the pool; live mode passes None here and checks
    once list_ids() has answered."""
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise InputError(f"{path}: questions file must be a JSON list")
    problems = []
    seen = set()
    for q in questions:
        qid = q.get("id")
        if "gold_ids" not in q:
            problems.append(f"{qid}: no gold_ids (not a Recall 1.0 question file)")
            continue
        if qid in seen:
            problems.append(f"{qid}: duplicate question id")
        seen.add(qid)
        missing = [g for g in q["gold_ids"] if pool_ids is not None and g not in pool_ids]
        if missing:
            problems.append(f"{qid}: gold ids not in the pool: {missing}")
        first = q.get("must_rank_first")
        if first is not None:
            current = rc.current_ids(first)
            if not current or not all(isinstance(c, str) for c in current):
                problems.append(f"{qid}: must_rank_first must be a record id or a non-empty list of record ids")
            elif len(set(current)) != len(current):
                problems.append(f"{qid}: must_rank_first lists a record twice")
            elif any(c not in q["gold_ids"] for c in current):
                problems.append(f"{qid}: must_rank_first names a record that is not one of its gold ids")
    refuse(path, problems)
    return questions


def load_qset_manifest(questions_path: str, corpus_path: Optional[str]) -> Dict[str, Any]:
    """Reads manifest.json next to the questions file, if there is one, and checks
    that it describes these exact files (just the questions in live mode)."""
    path = os.path.join(os.path.dirname(os.path.abspath(questions_path)), "manifest.json")
    q_sha = rc.file_sha256(questions_path)
    info = {"version": "unknown", "sha256": q_sha, "manifest": None, "matches": False,
            "gates_passed": None, "status": "unknown", "failed_gates": []}
    if not os.path.exists(path):
        return info
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    files = m.get("files", {})
    matches = (files.get("questions.json") == q_sha
               and (corpus_path is None or files.get("corpus.jsonl") == rc.file_sha256(corpus_path)))
    lint = m.get("lint", {})
    info.update({
        "version": m.get("qset_version", "unknown"),
        "manifest": "manifest.json",
        "manifest_sha256": rc.file_sha256(path),
        "matches": matches,
        "gates_passed": lint.get("gates_passed"),
        "failed_gates": lint.get("failed_gates", []),
        "status": m.get("status", "unknown"),
    })
    return info


def harness_commit() -> Dict[str, Any]:
    """The commit the harness ran from, and whether the tree had uncommitted changes. Git
    is asked only in a checkout of this project, with its own .git at the root. An
    installed copy has no commit: asking git there would report whatever repository
    happens to hold the folder it was installed into. A result from an installed copy is
    pinned by rc.harness_source_sha256() instead."""
    if not rc.IS_CHECKOUT or not os.path.exists(os.path.join(REPO_DIR, ".git")):
        return {"commit": "unknown", "dirty": None}

    def git(*args):
        return subprocess.run(["git", "-C", REPO_DIR] + list(args), capture_output=True,
                              text=True, timeout=10)
    try:
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": "unknown", "dirty": None}
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


# ---------------------------------------------------------------------------
# Hits and the window
# ---------------------------------------------------------------------------

def to_hit(raw: Any) -> Tuple[str, str, Optional[float]]:
    """(id, text, score) from whatever the adapter returned for one hit."""
    if isinstance(raw, str):
        return rc.record_id(raw), raw, None
    if isinstance(raw, dict):
        text = raw.get("text", "")
        hid = raw.get("id") or rc.record_id(text)
        score = raw.get("score")
    elif isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
        hid, text = raw[0], raw[1]
        score = raw[2] if len(raw) == 3 else None
        hid = hid or rc.record_id(text)
    else:
        raise TypeError(f"a hit must be a text string, an (id, text, score) tuple or a dict; got {raw!r:.80}")
    if score is not None:
        score = float(score)
    return str(hid), "" if text is None else str(text), score


def build_window(raw_hits: List[Any], window: int) -> Tuple[List[Dict[str, Any]], int]:
    """Truncate to the window, then drop repeated ids. A repeat burns its slot: the
    window is not refilled and every kept hit keeps its original position."""
    entries = []
    seen = set()
    burned = 0
    for pos, raw in enumerate(list(raw_hits or [])[:window]):
        hid, text, score = to_hit(raw)
        if hid in seen:
            burned += 1
            continue
        seen.add(hid)
        entries.append({"rank": pos, "id": hid, "score": score, "text": text})
    return entries, burned


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

class Pool:
    """The corpus, with what scoring needs precomputed."""

    live = False
    ts_note: Optional[str] = None

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.ids = [r["id"] for r in records]
        self.id_set = set(self.ids)
        self.text = {r["id"]: r["text"] for r in records}
        self.norm = {r["id"]: rc.normalize(r["text"]) for r in records}
        self.tokens = {i: n.split() for i, n in self.norm.items()}
        self.index = rc.PhraseIndex([self.norm[i] for i in self.ids])
        self.N = len(records)
        # Record timestamps as the corpus gives them; the version tier needs them.
        self.ts = {r["id"]: r["ts"] for r in records if rc.has_ts(r.get("ts"))}

    def norm_of(self, hid: str, text: str) -> str:
        if hid in self.text and self.text[hid] == text:
            return self.norm[hid]
        return rc.normalize(text)

    def tokens_of(self, hid: str, text: str) -> List[str]:
        if hid in self.text and self.text[hid] == text:
            return self.tokens[hid]
        return rc.tokenize(text)


class LivePool:
    """Live mode: the store is the pool. Its ids come from list_ids() (None when the
    adapter cannot list them) and its timestamps from record_ts(). Record texts are
    known only through what search returns, so nothing about them is precomputed."""

    live = True

    def __init__(self, ids: Optional[List[str]]):
        self.ids = sorted(set(ids)) if ids is not None else None
        self.id_set = set(self.ids) if self.ids is not None else None
        self.N = len(self.ids) if self.ids is not None else None
        self.text: Dict[str, str] = {}
        self.ts: Dict[str, str] = {}
        self.ts_note: Optional[str] = None
        self.ts_read_at: Optional[datetime] = None

    def norm_of(self, hid: str, text: str) -> str:
        return rc.normalize(text)

    def tokens_of(self, hid: str, text: str) -> List[str]:
        return rc.tokenize(text)


def scoring_rule(q: Dict[str, Any]) -> str:
    if q.get("category") == "abstention" or not q.get("gold_ids"):
        return "abstention"
    if q.get("category") in rc.ALL_GOLD_CATEGORIES:
        return "gold_all"
    return "gold_any"


def question_constants(q: Dict[str, Any], pool: Pool) -> Dict[str, Any]:
    """Facts about a question that do not depend on the adapter: its G per tier."""
    gold = q.get("gold_ids") or []
    if not gold:
        return {"variants": [], "gold_tokens": [], "G": 0, "G_exact": None, "G_family": None}
    variants = rc.answer_variants(q.get("answer"), q.get("acceptable_answers"))
    gold_tokens = [pool.tokens[g] for g in gold]
    g_exact = len(pool.index.matching_any(variants))
    g_family = sum(1 for i in pool.ids
                   if any(rc.token_overlap(gt, pool.tokens[i]) >= rc.FAMILY_THRESHOLD for gt in gold_tokens))
    return {"variants": variants, "gold_tokens": gold_tokens, "G": len(gold),
            "G_exact": g_exact, "G_family": g_family}


def live_question_constants(q: Dict[str, Any]) -> Dict[str, Any]:
    """Live mode: the pool's texts are unknown, so the exact and family G are not
    available, and neither is the family tier (it needs the gold texts)."""
    gold = q.get("gold_ids") or []
    variants = rc.answer_variants(q.get("answer"), q.get("acceptable_answers")) if gold else []
    return {"variants": variants, "gold_tokens": [], "G": len(gold), "G_exact": None, "G_family": None}


def score_question(q: Dict[str, Any], const: Dict[str, Any], entries: List[Dict[str, Any]],
                   pool: Pool) -> Dict[str, Any]:
    gold = list(q.get("gold_ids") or [])
    gold_set = set(gold)
    rule = scoring_rule(q)
    for e in entries:
        e["gold"] = e["id"] in gold_set

    gold_rank = next((e["rank"] for e in entries if e["gold"]), None)
    first_pos = {}
    for e in entries:
        if e["gold"] and e["id"] not in first_pos:
            first_pos[e["id"]] = e["rank"]
    gold_all_rank = max(first_pos.values()) if gold and len(first_pos) == len(gold_set) else None
    first_nongold = next((e["rank"] for e in entries if not e["gold"]), None)

    version_ok = None
    current = rc.current_ids(q.get("must_rank_first"))
    if current:
        first_gold = next((e["id"] for e in entries if e["gold"]), None)
        version_ok = first_gold in current

    exact_rank = family_rank = None
    if rule != "abstention":
        for e in entries:
            if exact_rank is None and rc.matches_any(pool.norm_of(e["id"], e["text"]), const["variants"]):
                exact_rank = e["rank"]
            if family_rank is None:
                toks = pool.tokens_of(e["id"], e["text"])
                if any(rc.token_overlap(gt, toks) >= rc.FAMILY_THRESHOLD for gt in const["gold_tokens"]):
                    family_rank = e["rank"]
            if exact_rank is not None and family_rank is not None:
                break

    headline_rank = None
    if rule == "gold_any":
        headline_rank = gold_rank
    elif rule == "gold_all":
        headline_rank = gold_all_rank

    return {
        "id": q.get("id"),
        "question": q.get("question"),
        "category": q.get("category"),
        "difficulty": q.get("difficulty"),
        "answer": q.get("answer"),
        "acceptable_answers": q.get("acceptable_answers", []),
        "gold_ids": gold,
        "must_rank_first": q.get("must_rank_first"),
        "scored_on": rule,
        "G": const["G"],
        "G_exact": const["G_exact"],
        "G_family": const["G_family"],
        "headline_rank": headline_rank,
        "gold_rank": gold_rank,
        "gold_all_rank": gold_all_rank,
        "version_ok": version_ok,
        "exact_rank": exact_rank,
        "family_rank": family_rank,
        "first_nongold_rank": first_nongold,
        "abstained": None,
        "window": [{"rank": e["rank"], "id": e["id"], "gold": e["gold"], "score": e["score"],
                    "text": e["text"]} for e in entries],
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_once(adapter: BaseMemoryAdapter, pool: Pool, questions: List[Dict[str, Any]],
             consts: List[Dict[str, Any]], window: int, quiet: bool = False) -> Dict[str, Any]:
    workspace = tempfile.mkdtemp(prefix="parsica_recall_")
    log = (lambda *a: None) if quiet else print
    v1 = is_v1_ingest(adapter)
    try:
        call_setup(adapter, workspace, "file")
        log(f"Ingesting {pool.N} records ({'v1' if v1 else 'v2'} adapter contract)...")
        t0 = time.perf_counter()
        for rec in pool.records:
            call_ingest(adapter, rec, v1)
        adapter.save()
        ingest_s = time.perf_counter() - t0
        version = describe_version(adapter)
        memory_count = adapter.memory_count()
        searched = search_all(adapter, pool, questions, consts, window, log)
        return {
            "per_question": searched["per_question"],
            "version": version,
            "memory_count": memory_count,
            "ingest_s": ingest_s,
            "scores_present": searched["scores_present"],
            "text_only_hits": searched["text_only_hits"],
            "contract": "v1" if v1 else "v2",
        }
    finally:
        try:
            adapter.teardown()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def has_own_id(raw: Any) -> bool:
    """Whether a hit carries the adapter's own id, rather than leaving the runner to hash its text."""
    if isinstance(raw, dict):
        return bool(raw.get("id"))
    if isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
        return bool(raw[0])
    return False


def search_all(adapter: BaseMemoryAdapter, pool: Any, questions: List[Dict[str, Any]],
               consts: List[Dict[str, Any]], window: int, log: Callable[..., None]) -> Dict[str, Any]:
    """One search per question at the window, each scored. File and live mode share it."""
    log(f"Searching {len(questions)} questions (window {window})...")
    per_question = []
    scores_seen = scores_missing = 0
    text_only = 0
    for i, (q, const) in enumerate(zip(questions, consts)):
        t1 = time.perf_counter()
        raw = adapter.search(q["question"], window)
        elapsed_ms = (time.perf_counter() - t1) * 1000.0
        text_only += sum(1 for hit in list(raw or [])[:window] if not has_own_id(hit))
        entries, burned = build_window(raw, window)
        for e in entries:
            if e["score"] is None:
                scores_missing += 1
            else:
                scores_seen += 1
        rec = score_question(q, const, entries, pool)
        rec["returned_count"] = len(raw or [])
        rec["burned_duplicates"] = burned
        rec["search_time_ms"] = round(elapsed_ms, 3)
        per_question.append(rec)
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(questions)}")
    return {
        "per_question": per_question,
        "scores_present": scores_seen > 0 and scores_missing == 0,
        "text_only_hits": text_only,
    }


def describe_error(where: str, exc: BaseException, redact: bool) -> str:
    """An exception as the results file records it. A store's message can echo
    anything, a token or record text included, so under --redact only the exception
    type and the message's text_hash are kept."""
    if redact:
        return f"{where} raised {type(exc).__name__} (message withheld by --redact, {rc.text_hash(str(exc))})"
    return f"{where} raised {type(exc).__name__}: {exc}"


def list_pool_ids(adapter: BaseMemoryAdapter) -> Optional[List[str]]:
    ids = adapter.list_ids()
    return None if ids is None else [str(i) for i in ids]


def read_store_ts(adapter: BaseMemoryAdapter, questions: List[Dict[str, Any]], pool: LivePool) -> None:
    """The store's own timestamps for every gold record: the version tier needs them,
    and so does the age breakdown. The runner asks once, read-only, after the
    searches, and never guesses one."""
    wanted = sorted({g for q in questions for g in q.get("gold_ids") or []})
    if not wanted:
        return
    got = adapter.record_ts(wanted)
    if got is None:
        pool.ts_note = "the adapter has no record_ts(), so the runner cannot see the store's timestamps"
        return
    if not isinstance(got, dict):
        raise TypeError("record_ts() must return a dict of record id to ts")
    asked = set(wanted)
    pool.ts = {str(k): v for k, v in got.items() if str(k) in asked and rc.has_ts(v)}
    pool.ts_read_at = datetime.now(timezone.utc)


def run_live_once(adapter: BaseMemoryAdapter, questions: List[Dict[str, Any]], questions_path: str,
                  fixed: Optional[Dict[str, Any]] = None, quiet: bool = False, redact: bool = False) -> Dict[str, Any]:
    """One live run (spec section 10): open the store read-only, fingerprint the pool,
    search, fingerprint it again. Nothing is ingested and nothing is written. The
    first run fixes the pool, band and constants; a --verify run reuses them so
    both runs search the same window."""
    workspace = tempfile.mkdtemp(prefix="parsica_recall_live_")
    log = (lambda *a: None) if quiet else print
    try:
        call_setup(adapter, workspace, "live")
        ids_start = list_pool_ids(adapter)
        if fixed is None:
            pool = LivePool(ids_start)
            if pool.id_set is not None:
                refuse(questions_path, missing_gold(questions, pool.id_set))
            band = rc.band_for(pool.N) if pool.N is not None else rc.band_unknown()
            fixed = {"pool": pool, "band": band, "consts": [live_question_constants(q) for q in questions],
                     "ts_read": False}
        pool, band = fixed["pool"], fixed["band"]
        log(f"Live store: N={'unknown (no list_ids())' if pool.N is None else pool.N}, "
            f"band {band['band']}, window {band['window']}; no ingest")
        version = describe_version(adapter)
        memory_count = adapter.memory_count()
        searched = search_all(adapter, pool, questions, fixed["consts"], band["window"], log)
        # A store that fails at the end must not cost the run: the searches are kept and
        # every error is recorded. Only list_ids() speaks for the pool: one that fails is
        # a lost end fingerprint, so the run is marked (spec section 10). A record_ts()
        # that fails leaves the pool and the verification alone; the version tier and
        # the age breakdown go without timestamps, with the error in the version note.
        end_errors, fingerprint_errors = [], []
        try:
            ids_end = list_pool_ids(adapter)
        except Exception as exc:
            ids_end = None
            fingerprint_errors.append(describe_error("list_ids()", exc, redact))
            end_errors.append(fingerprint_errors[-1])
        if not fixed["ts_read"]:
            fixed["ts_read"] = True
            try:
                read_store_ts(adapter, questions, pool)
            except Exception as exc:
                pool.ts = {}
                pool.ts_note = describe_error("record_ts()", exc, redact)
                end_errors.append(pool.ts_note)
        return {
            "per_question": searched["per_question"],
            "version": version,
            "memory_count": memory_count,
            "ingest_s": None,
            "scores_present": searched["scores_present"],
            "text_only_hits": searched["text_only_hits"],
            "contract": "v1" if is_v1_ingest(adapter) else "v2",
            "pool_hashes": [None if ids is None else rc.pool_hash(ids) for ids in (ids_start, ids_end)],
            "end_errors": end_errors,
            "fingerprint_errors": fingerprint_errors,
            "fixed": fixed,
        }
    finally:
        try:
            adapter.teardown()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def redact_records(per_question: List[Dict[str, Any]]) -> None:
    """--redact (spec section 10): every returned record text becomes its text_hash,
    the same form as a gold_hashes entry, so gold drift can still be checked from the
    file. The question, the answer and its alternates are dropped: a redacted file has
    no use for them, and they are short enough to guess from a hash. Ids, ranks and
    scores stay."""
    for r in per_question:
        r["question"] = None
        r["answer"] = None
        r["acceptable_answers"] = []
        for e in r["window"]:
            e["text"] = rc.text_hash(e["text"])


def live_drift(per_question: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Tuple[set, set]:
    """Gold drift in live mode, read from what the store returned: a gold record whose
    returned text no longer hashes to its gold_hashes entry has drifted. Gold records
    the store never returned, or returned without text, cannot be checked."""
    pinned: Dict[str, str] = {}
    for q in questions:
        pinned.update(q.get("gold_hashes") or {})
    seen, drifted = set(), set()
    for r in per_question:
        for e in r["window"]:
            if e["gold"] and e["text"] and e["id"] in pinned:
                seen.add(e["id"])
                if rc.text_hash(e["text"]) != pinned[e["id"]]:
                    drifted.add(e["id"])
    return drifted, seen


def apply_abstention(run: Dict[str, Any]) -> Dict[str, Any]:
    """Abstention is scored only from scores (spec section 5): correct when the
    window is empty or its top score is under the adapter's declared threshold."""
    threshold = run["version"].get("abstain_threshold")
    available = run["scores_present"] and isinstance(threshold, (int, float))
    items = [r for r in run["per_question"] if r["scored_on"] == "abstention"]
    if not available:
        reason = ("adapter returns no scores" if not run["scores_present"]
                  else "adapter declares no abstain_threshold in version()")
        return {"available": False, "reason": reason, "total": len(items)}
    correct = 0
    for r in items:
        w = r["window"]
        r["abstained"] = (not w) or (w[0]["score"] is not None and w[0]["score"] < threshold)
        correct += int(r["abstained"])
    return {"available": True, "threshold": threshold, "correct": correct, "total": len(items),
            "pct": round(100.0 * correct / len(items), 2) if items else None}


def records_hash(per_question: List[Dict[str, Any]]) -> str:
    clean = [{k: v for k, v in r.items() if k not in TIMING_FIELDS} for r in per_question]
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _pct(hits: int, n: int) -> Optional[float]:
    return round(100.0 * hits / n, 2) if n else None


def tier_table(ranks: List[Optional[int]], curve: Optional[List[float]], k_set: List[int],
               window: int) -> Dict[str, Any]:
    """Recall at each k, with chance, skill and validity beside it. With no chance
    curve (pool size unknown, or a live pool whose texts the runner cannot see)
    those three are None, never estimated."""
    n = len(ranks)
    out = {"n": n, "recall": {}, "chance": {}, "skill": {}, "valid": {}}
    for k in k_set:
        hits = sum(1 for r in ranks if r is not None and r < k)
        obs = hits / n if n else 0.0
        out["recall"][str(k)] = _pct(hits, n)
        if curve is None:
            out["chance"][str(k)] = out["skill"][str(k)] = out["valid"][str(k)] = None
            continue
        ch = rc.chance_at(curve, k)
        out["chance"][str(k)] = round(100.0 * ch, 4)
        out["skill"][str(k)] = round(100.0 * rc.skill(obs, ch), 2) if n else None
        out["valid"][str(k)] = ch <= rc.VALID_CHANCE + 1e-12
    out["mrr"] = round(sum(1.0 / (r + 1) for r in ranks if r is not None and r < window) / n, 4) if n else None
    return out


def _g_stats(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}
    s = sorted(values)
    return {"mean": round(sum(s) / len(s), 3), "median": s[len(s) // 2], "max": s[-1],
            "zero": sum(1 for g in s if g == 0), "multi_pct": round(100.0 * sum(1 for g in s if g > 1) / len(s), 2)}


def auroc(pairs: List[Tuple[float, bool]]) -> Optional[float]:
    """Probability a random gold hit outscores a random non-gold hit (ties count half)."""
    pos = sum(1 for _, y in pairs if y)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    ordered = sorted(pairs, key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for p in ordered[i:j] if p[1])
        i = j
    return round((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg), 4)


def timed_version_item(rec: Dict[str, Any], pool: "Pool") -> bool:
    """The version tier counts an item only when every one of its gold records carries
    a ts. Without one, nothing in the pool can tell any system which record is
    current, and a version score would be a coin flip (spec section 5). The harness
    never invents a timestamp."""
    return bool(rec["gold_ids"]) and all(g in pool.ts for g in rec["gold_ids"])


def version_note(items: int, timed: int, no_ts_reason: Optional[str] = None) -> str:
    if not items:
        return "no version items"
    if not timed and no_ts_reason:
        return f"none of the {items} version items can be scored: {no_ts_reason}"
    if not timed:
        return (f"none of the {items} version items has a ts on every gold record, "
                "so no system can tell which record is current")
    return f"scored on {timed} of {items} version items (those with a ts on every gold record)"


def age_breakdown(pq: List[Dict[str, Any]], pool: Any, band: Dict[str, Any]) -> Dict[str, Any]:
    """Live mode (spec section 7): the gold hit rate at the headline k by the age of the
    gold record, from the store's own ts as record_ts() gave it, counted once per gold
    record per question. Ages are measured from the moment record_ts() answered."""
    if not pool.live:
        return {"available": False, "reason": "live mode only: it reads the store's own timestamps"}
    if pool.ts_read_at is None:
        return {"available": False, "reason": pool.ts_note or "record_ts() did not answer"}
    k = band["headline_k"] or band["window"]
    counts = {name: [0, 0] for name in [b for b, _ in rc.AGE_BUCKETS] + [rc.AGE_UNKNOWN]}
    for r in pq:
        if r["scored_on"] == "abstention":
            continue
        top = {e["id"] for e in r["window"] if e["rank"] < k}
        for g in r["gold_ids"]:
            bucket = counts[rc.age_bucket(pool.ts.get(g), pool.ts_read_at)]
            bucket[0] += 1
            bucket[1] += int(g in top)
    return {"available": True, "k": k, "measured_at": pool.ts_read_at.isoformat(timespec="seconds"),
            "buckets": {name: {"gold": c[0], "hit": c[1], "pct": _pct(c[1], c[0])} for name, c in counts.items()}}


def gold_chance_curve(rows: List[Dict[str, Any]], n: Optional[int], k_max: Optional[int] = None) -> Optional[List[float]]:
    """The gold tier's chance model (spec section 6) for a set of scored questions: each
    one's own G and scoring rule over a pool of n. The headline and every breakdown
    group use it, so a group holding every question gets the headline's numbers."""
    if n is None:
        return None
    return rc.mean_chance_curve(n, [(r["G"], "all" if r["scored_on"] == "gold_all" else "any") for r in rows],
                                k_max=k_max)


def aggregate(run: Dict[str, Any], pool: Pool, band: Dict[str, Any]) -> Dict[str, Any]:
    pq = run["per_question"]
    k_set, window, hk = band["k_set"], band["window"], band["headline_k"]
    answerable = [r for r in pq if r["scored_on"] != "abstention"]

    # Live mode knows N only through list_ids() and never sees the pool's texts, so the
    # gold chance line needs N, and the exact and family lines (and the family tier,
    # which needs the gold texts) are file mode only.
    live = pool.live
    gold_curve = gold_chance_curve(answerable, pool.N)
    exact_curve = None if live else rc.mean_chance_curve(
        pool.N, [(r["G_exact"], "any") for r in answerable], k_max=window)
    family_curve = None if live else rc.mean_chance_curve(
        pool.N, [(r["G_family"], "any") for r in answerable], k_max=window)

    headline = tier_table([r["headline_rank"] for r in answerable], gold_curve, k_set, window)
    version_items = [r for r in answerable if r["must_rank_first"]]
    timed = [r for r in version_items if timed_version_item(r, pool)]
    timed_ok = sum(1 for r in timed if r["version_ok"])
    headline.update({
        "tier": "gold", "band": band["band"], "k": hk, "window": window, "k_set": k_set,
        "version_items": len(version_items),
        "version_n": len(timed),
        "version_ok": timed_ok if timed else None,
        "version_pct": _pct(timed_ok, len(timed)),
        "version_note": version_note(len(version_items), len(timed), pool.ts_note),
        "ceiling": ({"0.05": rc.k_ceiling(gold_curve, 0.05), "0.10": rc.k_ceiling(gold_curve, 0.10)}
                    if gold_curve is not None else {"0.05": None, "0.10": None}),
        "headline_valid": (rc.chance_at(gold_curve, hk) <= rc.VALID_CHANCE + 1e-12
                           if gold_curve is not None and hk is not None else None),
    })

    all_k = list(range(1, window + 1))
    curves = {
        "gold": tier_table([r["headline_rank"] for r in answerable], gold_curve, all_k, window),
        "exact": tier_table([r["exact_rank"] for r in answerable], exact_curve, all_k, window),
        "family": None if live else tier_table([r["family_rank"] for r in answerable], family_curve, all_k, window),
    }
    diagnostics = {
        "note": "exact and family tiers are diagnostics, never the headline",
        "exact": tier_table([r["exact_rank"] for r in answerable], exact_curve, k_set, window),
        "family": None if live else tier_table([r["family_rank"] for r in answerable], family_curve, k_set, window),
    }
    if live:
        diagnostics["family_note"] = ("not available in live mode: the family tier needs the gold texts, "
                                      "which the store does not hand the runner")

    def breakdown(key: str) -> Dict[str, Any]:
        groups = defaultdict(list)
        for r in answerable:
            groups[r[key] or "unknown"].append(r)
        out = {}
        for name in sorted(groups):
            rows = groups[name]
            ranks = [r["headline_rank"] for r in rows]
            # Chance and skill from the group's own G values and scoring rules, with the
            # headline's chance model over the same pool N (spec section 7).
            table = tier_table(ranks, gold_chance_curve(rows, pool.N, k_max=window), k_set, window)
            t = {"n": len(rows),
                 "gold": {str(k): _pct(sum(1 for x in ranks if x is not None and x < k), len(rows)) for k in k_set},
                 "chance": table["chance"],
                 "skill": table["skill"],
                 "mrr": round(sum(1.0 / (x + 1) for x in ranks if x is not None) / len(rows), 4)}
            vrows = [r for r in rows if r["must_rank_first"]]
            if vrows:
                trows = [r for r in vrows if timed_version_item(r, pool)]
                t["version_pct"] = _pct(sum(1 for r in trows if r["version_ok"]), len(trows))
            fn = sorted(r["first_nongold_rank"] for r in rows if r["first_nongold_rank"] is not None)
            t["first_nongold_rank_median"] = fn[len(fn) // 2] if fn else None
            if hk is None:
                t["exact"] = t["family"] = None
            else:
                t["exact"] = _pct(sum(1 for r in rows if r["exact_rank"] is not None and r["exact_rank"] < hk),
                                  len(rows))
                t["family"] = None if live else _pct(
                    sum(1 for r in rows if r["family_rank"] is not None and r["family_rank"] < hk), len(rows))
            out[name] = t
        return out

    abstention = run["abstention"] if "abstention" in run else apply_abstention(run)

    times = [r["search_time_ms"] for r in pq if r["search_time_ms"] is not None]
    latency = {"search_ms_p50": rc.percentile(times, 50), "search_ms_p95": rc.percentile(times, 95),
               "search_ms_max": max(times) if times else None,
               "search_ms_mean": round(sum(times) / len(times), 3) if times else None,
               "ingest_records_per_s": (round(pool.N / run["ingest_s"], 1)
                                        if run["ingest_s"] is not None and run["ingest_s"] > 0 else None)}
    if run.get("contract") == CONTRACT_RESULTS_FILE:
        latency["measured_by"] = "the results file" if times else None
        latency["questions_timed"] = len(times)

    calibration: Dict[str, Any] = {"available": run["scores_present"]}
    if run["scores_present"]:
        pairs = [(e["score"], e["gold"]) for r in answerable for e in r["window"]]
        calibration["auroc"] = auroc(pairs)
        ordered = sorted(pairs, key=lambda p: p[0])
        if len(ordered) >= 10:
            deciles = []
            for d in range(10):
                chunk = ordered[d * len(ordered) // 10:(d + 1) * len(ordered) // 10]
                deciles.append(_pct(sum(1 for _, y in chunk if y), len(chunk)))
            calibration["gold_rate_by_score_decile"] = deciles
    else:
        calibration["reason"] = "adapter returns no scores"

    slots = redundant = 0
    for r in pq:
        toks = [pool.tokens_of(e["id"], e["text"]) for e in r["window"]]
        for j in range(1, len(toks)):
            slots += 1
            if any(rc.token_overlap(toks[j], toks[i]) >= 0.9 for i in range(j)):
                redundant += 1
    redundancy = {"slots_after_first": slots, "redundant": redundant, "pct": _pct(redundant, slots)}

    gold_g = [r["G"] for r in answerable]
    if live:
        drifted, drift_seen = live_drift(pq, run.get("questions", []))
    else:
        drifted = set()
        for q in run.get("questions", []):
            for gid, h in (q.get("gold_hashes") or {}).items():
                if gid in pool.text and rc.text_hash(pool.text[gid]) != h:
                    drifted.add(gid)
    mc = run["memory_count"]
    pool_stats = {"N": pool.N, "answerable": len(answerable), "abstention": len(pq) - len(answerable),
                  "G": _g_stats(gold_g),
                  "G_exact": {} if live else _g_stats([r["G_exact"] for r in answerable]),
                  "G_family": {} if live else _g_stats([r["G_family"] for r in answerable]),
                  "drifted_gold_count": len(drifted), "memory_count": mc,
                  "store_amplification": round(mc / pool.N, 3) if mc is not None and mc >= 0 and pool.N else None,
                  "burned_duplicates": sum(r["burned_duplicates"] for r in pq),
                  "over_returned": sum(1 for r in pq if r["returned_count"] > window)}
    if live:
        pool_stats["drift_checked"] = len(drift_seen)
        pool_stats["drift_note"] = ("checked on the gold records the store returned with text; "
                                    "gold records it never returned cannot be checked")

    out = {
        "headline": headline,
        "diagnostics": diagnostics,
        "abstention": abstention,
        "by_category": breakdown("category"),
        "by_difficulty": breakdown("difficulty"),
        "pool": pool_stats,
        "latency": latency,
        "calibration": calibration,
        "redundancy": redundancy,
        "curves": curves,
        "chance_curve_gold": [round(p, 6) for p in gold_curve] if gold_curve is not None else None,
        "age": age_breakdown(pq, pool, band),
    }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def result_line(agg: Dict[str, Any], manifest: Dict[str, Any], verified: str, standard: bool) -> str:
    """The one-line result (spec section 6). With the pool size unknown (live mode
    without list_ids()) there is no band, headline k, chance or ceiling, so the line
    carries gold at every k instead, says N=unknown and pool=unknown, and is not a
    standard result."""
    h = agg["headline"]
    k, w = h["k"], h["window"]
    n = agg["pool"]["N"]
    parts = [rc.SPEC_VERSION, f"track={manifest['params']['track']}", f"band={h['band']}",
             f"N={'unknown' if n is None else n}"]
    if k is not None:
        parts.append(f"k={k}")
    parts += [f"window={w}", f"qset={manifest['qset']['version']}@{manifest['qset']['sha256'][:8]}"]
    if k is not None:
        parts += [f"gold@{k}={h['recall'][str(k)]}", f"skill@{k}={h['skill'][str(k)]}"]
        for kk in h["k_set"]:
            if kk != k and h["valid"][str(kk)]:
                parts.append(f"gold@{kk}={h['recall'][str(kk)]}")
    else:
        parts += [f"gold@{kk}={h['recall'][str(kk)]}" for kk in h["k_set"]]
    parts.append(f"mrr@{w}={h['mrr']}")
    parts.append(f"version={h['version_pct'] if h['version_n'] else 'n/a'}")
    if k is not None:
        parts.append(f"chance@{k}={h['chance'][str(k)]}")
        parts.append(f"ceiling5={h['ceiling']['0.05']}")
    a = manifest["adapter"]
    parts.append(f"adapter={a['name']}@{a['version'].get('version', 'unknown')}")
    parts.append(f"verified={verified}")
    if "spread" in h:
        sk = k if k is not None else w
        parts.append(f"spread@{sk}={h['spread'][str(sk)]}")
    if h["headline_valid"] is False:
        parts.append(f"above_ceiling=k{k}")
    corpus = manifest["corpus"]
    if corpus.get("mode") == "live":
        if n is None:
            parts.append("pool=unknown")
        elif corpus.get("pool_changed"):
            parts.append("pool=changed")
    parts.append(f"standard={'yes' if standard else 'no'}")
    return " ".join(parts)


def _cell(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _signed(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _num(value: Any, width: int, places: int, suffix: str = "") -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width - len(suffix)}.{places}f}{suffix}"


def print_report(agg: Dict[str, Any], manifest: Dict[str, Any], verification: Dict[str, Any],
                 standard: Dict[str, Any], line: str) -> None:
    h, pool = agg["headline"], agg["pool"]
    a = manifest["adapter"]
    v = a["version"]
    q = manifest["qset"]
    bar = "=" * 70
    params = manifest["params"]
    live = params["mode"] == "live"
    print(f"\n{bar}\nPARSICA RECALL {rc.SPEC_VERSION} (harness {rc.HARNESS_VERSION}), retrieval, "
          f"{params['mode']} mode\n{bar}")
    net = {True: "yes", False: "no"}.get(v.get("network"), "unknown")
    print(f"Adapter:      {a['name']} ({v.get('engine_class')}, version {v.get('version')}, "
          f"network {net}, contract {a['contract']})")
    gates = "gates passed" if q.get("gates_passed") else (
        f"gates failed: {', '.join(q.get('failed_gates') or []) or 'unknown'}" if q.get("gates_passed") is False
        else "no question-set manifest")
    print(f"Question set: {q['version']}@{q['sha256'][:8]} ({q.get('status')}; {gates})")
    if pool["N"] is None:
        print(f"Pool:         N unknown (the adapter has no list_ids()); window {h['window']}, gold at k "
              f"{h['k_set']}, no chance, skill or ceiling")
    else:
        print(f"Pool:         N={pool['N']}, band {h['band']}, k set {h['k_set']}, headline k={h['k']}, "
              f"window {h['window']}" + (" (small pool: no standard report)" if h['band'] == 'none' else ""))
    if live:
        c = manifest["corpus"]
        if c.get("pool_sha256"):
            state = ("CHANGED during the run, so it cannot be verified" if c.get("pool_changed")
                     else "unchanged from start to end")
            print(f"Live pool:    sorted id list {c['pool_sha256'][:19]}..., {state}")
        print(f"Gold drift:   {pool['drifted_gold_count']} of the {pool['drift_checked']} gold records the store "
              f"returned no longer match gold_hashes")
    if params.get("redacted"):
        print("Texts:        redacted (record texts are their text_hash; questions and answers are dropped)")
    g = pool["G"]
    print(f"Questions:    {pool['answerable']} answerable, {pool['abstention']} abstention; "
          f"gold per question mean {g.get('mean')}, max {g.get('max')}")

    print(f"\nGold tier (headline), scored by record id")
    print(f"  {'k':>4} {'gold':>8} {'chance':>9} {'skill':>7}")
    for k in h["k_set"]:
        ks = str(k)
        mark = "*" if k == h["k"] else " "
        flag = "  above ceiling" if h["valid"][ks] is False else ""
        print(f" {mark}{k:>4} {_num(h['recall'][ks], 8, 2, '%')} {_num(h['chance'][ks], 9, 3, '%')} "
              f"{_num(h['skill'][ks], 7, 2)}{flag}")
    vtxt = (f"{h['version_ok']}/{h['version_n']} ({h['version_pct']}%)" if h["version_n"] else "n/a")
    print(f"  MRR@{h['window']}: {h['mrr']}   version tier: {vtxt}")
    if h["version_items"]:
        print(f"    {h['version_note']}")
    if h["ceiling"]["0.05"] is None:
        print("  k ceiling: n/a (pool size unknown)")
    else:
        print(f"  k ceiling: chance <= 5% up to k={h['ceiling']['0.05']}, <= 10% up to k={h['ceiling']['0.10']}")

    d = agg["diagnostics"]
    fam = d["family"]
    print(f"\nDiagnostics (string tiers, not the headline)")
    print(f"  {'k':>4} {'exact':>8} {'chance':>9} {'family':>8} {'chance':>9}")
    for k in h["k_set"]:
        ks = str(k)
        print(f"  {k:>4} {_num(d['exact']['recall'][ks], 8, 2, '%')} {_num(d['exact']['chance'][ks], 9, 3, '%')} "
              f"{_num(fam['recall'][ks] if fam else None, 8, 2, '%')} "
              f"{_num(fam['chance'][ks] if fam else None, 9, 3, '%')}")
    if "family_note" in d:
        print(f"  family tier {d['family_note']}")

    ab = agg["abstention"]
    if ab["available"]:
        print(f"\nAbstention: {ab['correct']}/{ab['total']} ({ab['pct']}%), threshold {ab['threshold']}")
    else:
        print(f"\nAbstention: not available ({ab['reason']}); excluded from every aggregate")

    print(f"\nBy category (gold tier)")
    head = "  ".join(f"{'@' + str(k):>6}" for k in h["k_set"])
    print(f"  {'category':24} {'n':>4}  {head}  {'MRR':>6}  version")
    for cat, t in agg["by_category"].items():
        cells = "  ".join(_num(t['gold'][str(k)], 6, 1) for k in h["k_set"])
        vcell = "" if "version_pct" not in t else ("n/a" if t["version_pct"] is None else t["version_pct"])
        print(f"  {cat:24} {t['n']:>4}  {cells}  {_num(t['mrr'], 6, 3)}  {vcell}")
    print(f"\nBy difficulty (gold tier)")
    for diff, t in agg["by_difficulty"].items():
        cells = "  ".join(_num(t['gold'][str(k)], 6, 1) for k in h["k_set"])
        print(f"  {diff:24} {t['n']:>4}  {cells}  {_num(t['mrr'], 6, 3)}")
    if any(v is not None for v in h["chance"].values()):  # no chance when the pool size is unknown
        for title, groups in (("By category", agg["by_category"]), ("By difficulty", agg["by_difficulty"])):
            print(f"\n{title}, skill and (chance) at k, from each group's own gold")
            head = "  ".join(f"{'@' + str(k):>15}" for k in h["k_set"])
            print(f"  {'':24} {'n':>4}  {head}")
            for name, t in groups.items():
                cells = "  ".join(f"{_num(t['skill'][str(k)], 6, 2)} ({_num(t['chance'][str(k)], 5, 2)}%)"
                                  for k in h["k_set"])
                print(f"  {name:24} {t['n']:>4}  {cells}")

    lat = agg["latency"]
    ingest = "no ingest (live mode)" if live else f"ingest {lat['ingest_records_per_s']} records/s"
    if "measured_by" not in lat:
        print(f"\nLatency: search p50 {lat['search_ms_p50']} ms, p95 {lat['search_ms_p95']} ms, "
              f"max {lat['search_ms_max']} ms; {ingest}")
    elif lat["measured_by"] is None:
        print("\nLatency: not measured (scored from a results file that reports no search times)")
    else:
        print(f"\nLatency, as the results file reports it ({lat['questions_timed']} questions): search p50 "
              f"{lat['search_ms_p50']} ms, p95 {lat['search_ms_p95']} ms, max {lat['search_ms_max']} ms")
    cal = agg["calibration"]
    print(f"Calibration: " + (f"AUROC {cal.get('auroc')}" if cal["available"] else f"not available ({cal['reason']})"))
    red = agg["redundancy"]
    print(f"Redundancy: {red['redundant']} of {red['slots_after_first']} window slots after the first "
          f"repeat an earlier hit at >= 0.9 overlap")
    if pool["burned_duplicates"]:
        print(f"Duplicate ids burned: {pool['burned_duplicates']}")

    passes = agg.get("extra_passes") or {}
    if passes:
        print("\nExtra passes")
    order = passes.get("order")
    if order and order["available"]:
        k = str(order["k"])
        g = order["gold"]
        print(f"  Order: {order['changed']} of {order['answerable']} answerable questions changed gold_rank "
              f"({order['changed_pct']}%); gold@{k} {_cell(g['base'][k])} -> {_cell(g['pass'][k])} "
              f"({_signed(g['delta'][k])}); seed '{order['seed']}'")
    elif order:
        print(f"  Order: not available ({order['reason']})")
    query = passes.get("query")
    if query and query["available"]:
        k = str(query["k"])
        cells = [f"{name} agrees on {_cell(v['agree_pct'])}% of gold_rank, gold@{k} {_cell(v['gold']['base'][k])} -> "
                 f"{_cell(v['gold']['pass'][k])} ({_signed(v['gold']['delta'][k])})"
                 for name, v in query["variants"].items()]
        print("  Query: " + "; ".join(cells))
        if query.get("pool_same_as_base") is False:
            print("  Query: the store's pool differed from the base run's when this pass ran, so its numbers "
                  "mix in that change")
    elif query:
        print(f"  Query: not available ({query['reason']})")
    if any(block.get("note") for block in passes.values()):
        print("  Note: the base run was unstable, so the pass differences include run-to-run noise")
    age = agg.get("age") or {}
    if live and age.get("available"):
        cells = [f"{name} {_cell(b['pct'])}% of {b['gold']}" for name, b in age["buckets"].items() if b["gold"]]
        print(f"\nAge, gold hit rate at k={age['k']} by the store's ts: " + ", ".join(cells))
    elif live:
        print(f"\nAge: not available ({age.get('reason')})")

    for note in agg.get("notes") or []:
        print(f"\nNote: {note}")

    label = {"yes": "verified", "no": "not verified", "unstable": "UNSTABLE"}[verification["status"]]
    print(f"\nVerification: {label}"
          + (f" ({verification['detail']})" if verification.get("detail") else ""))
    print(f"Standard result: {'yes' if standard['eligible'] else 'no'}"
          + ("" if standard["eligible"] else f" ({'; '.join(standard['reasons'])})"))
    print(f"\n{line}")


def write_csv(path: str, agg: Dict[str, Any]) -> None:
    c = agg["curves"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "gold_pct", "chance_pct", "skill_pct", "valid", "exact_pct",
                    "exact_chance_pct", "family_pct", "family_chance_pct"])
        fam = c["family"]
        for ks in c["gold"]["recall"]:
            valid = c["gold"]["valid"][ks]
            w.writerow([ks, c["gold"]["recall"][ks], c["gold"]["chance"][ks], c["gold"]["skill"][ks],
                        "" if valid is None else int(valid), c["exact"]["recall"][ks], c["exact"]["chance"][ks],
                        fam["recall"][ks] if fam else None, fam["chance"][ks] if fam else None])


def breakdown_csv_path(curve_csv: str) -> str:
    """The breakdown CSV sits beside the recall-curve CSV: x_recall_curve.csv gives
    x_breakdowns.csv, and any other name y.csv gives y_breakdowns.csv."""
    base = os.path.splitext(curve_csv)[0]
    if base.endswith("_recall_curve"):
        base = base[:-len("_recall_curve")]
    return base + "_breakdowns.csv"


def write_breakdown_csv(path: str, agg: Dict[str, Any]) -> None:
    """One row per group and k: the per-category and per-difficulty gold, chance and
    skill at every k in the band's set (spec section 7)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["breakdown", "group", "n", "k", "gold_pct", "chance_pct", "skill_pct"])
        for breakdown, groups in (("category", agg["by_category"]), ("difficulty", agg["by_difficulty"])):
            for name, t in groups.items():
                for ks in t["gold"]:
                    w.writerow([breakdown, name, t["n"], ks, t["gold"][ks], t["chance"][ks], t["skill"][ks]])


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Extra passes (spec section 7)
# ---------------------------------------------------------------------------

PASSES = ("order", "query")
ORDER_PASS_SEED = "recall/1.0 order pass 1"


def shuffled_records(records: List[Dict[str, Any]], seed: str) -> List[Dict[str, Any]]:
    """The order pass's ingest order: records sorted by the SHA-256 of seed:id. Anyone
    can rebuild it from the seed without this harness's random number generator."""
    return sorted(records, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode("utf-8")).hexdigest())


def pass_search(adapter: BaseMemoryAdapter, pool: Any, questions: List[Dict[str, Any]],
                consts: List[Dict[str, Any]], window: int, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
    """One search per answerable question with the given query text; only ranks are
    kept. A question is skipped (None) when its text comes out empty."""
    out: List[Optional[Dict[str, Any]]] = []
    for q, const, text in zip(questions, consts, texts):
        if scoring_rule(q) == "abstention" or not text:
            out.append(None)
            continue
        entries, _ = build_window(adapter.search(text, window), window)
        rec = score_question(q, const, entries, pool)
        out.append({"gold_rank": rec["gold_rank"], "headline_rank": rec["headline_rank"]})
    return out


def run_pass(adapter_factory: Callable[[], BaseMemoryAdapter], records: Optional[List[Dict[str, Any]]],
             live: bool, work: Callable[[BaseMemoryAdapter], Any]) -> Tuple[Any, Optional[str]]:
    """Runs work(adapter) on a fresh adapter: in file mode a new workspace with the
    records ingested in the order given, in live mode the same store opened again
    (and its pool fingerprinted, if it can list its ids)."""
    adapter = adapter_factory()
    workspace = tempfile.mkdtemp(prefix="parsica_recall_pass_")
    pool_fingerprint = None
    try:
        if live:
            call_setup(adapter, workspace, "live")
            try:
                ids = list_pool_ids(adapter)
                pool_fingerprint = None if ids is None else rc.pool_hash(ids)
            except Exception:
                pool_fingerprint = None
        else:
            v1 = is_v1_ingest(adapter)
            call_setup(adapter, workspace, "file")
            for rec in records or []:
                call_ingest(adapter, rec, v1)
            adapter.save()
        return work(adapter), pool_fingerprint
    finally:
        try:
            adapter.teardown()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def _gold_at(ranks: List[Optional[int]], k_set: List[int]) -> Dict[str, Optional[float]]:
    return {str(k): _pct(sum(1 for r in ranks if r is not None and r < k), len(ranks)) for k in k_set}


def _gold_change(base: List[Optional[int]], after: List[Optional[int]], k_set: List[int]) -> Dict[str, Any]:
    b, a = _gold_at(base, k_set), _gold_at(after, k_set)
    return {"base": b, "pass": a,
            "delta": {k: None if b[k] is None or a[k] is None else round(a[k] - b[k], 2) for k in b}}


def summarize_order(base: List[Dict[str, Any]], ranks: List[Optional[Dict[str, Any]]],
                    band: Dict[str, Any]) -> Dict[str, Any]:
    pairs = [(b, r) for b, r in zip(base, ranks) if r is not None]
    changed = [{"id": b["id"], "base": b["gold_rank"], "pass": r["gold_rank"]}
               for b, r in pairs if b["gold_rank"] != r["gold_rank"]]
    return {"available": True, "seed": ORDER_PASS_SEED, "answerable": len(pairs), "changed": len(changed),
            "changed_pct": _pct(len(changed), len(pairs)), "changed_questions": changed,
            "k": band["headline_k"] or band["window"],
            "gold": _gold_change([b["headline_rank"] for b, _ in pairs], [r["headline_rank"] for _, r in pairs],
                                 band["k_set"]),
            "ranks": {b["id"]: [r["gold_rank"], r["headline_rank"]] for b, r in pairs}}


def summarize_query(base: List[Dict[str, Any]], by_variant: Dict[str, List[Optional[Dict[str, Any]]]],
                    band: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": True, "k": band["headline_k"] or band["window"], "variants": {}}
    for name, ranks in by_variant.items():
        answerable = [(b, r) for b, r in zip(base, ranks) if b["scored_on"] != "abstention"]
        pairs = [(b, r) for b, r in answerable if r is not None]
        cats: Dict[str, List[bool]] = defaultdict(list)
        for b, r in pairs:
            cats[b["category"] or "unknown"].append(b["gold_rank"] == r["gold_rank"])
        agree = sum(1 for b, r in pairs if b["gold_rank"] == r["gold_rank"])
        out["variants"][name] = {
            "n": len(pairs), "skipped_empty": len(answerable) - len(pairs),
            "agree": agree, "agree_pct": _pct(agree, len(pairs)),
            "gold": _gold_change([b["headline_rank"] for b, _ in pairs], [r["headline_rank"] for _, r in pairs],
                                 band["k_set"]),
            "by_category": {c: {"n": len(v), "agree_pct": _pct(sum(v), len(v))} for c, v in sorted(cats.items())},
            "ranks": {b["id"]: None if r is None else [r["gold_rank"], r["headline_rank"]] for b, r in answerable},
        }
    return out


def run_extra_passes(names: List[str], adapter_factory: Callable[[], BaseMemoryAdapter], pool: Any,
                     questions: List[Dict[str, Any]], consts: List[Dict[str, Any]], band: Dict[str, Any],
                     live: bool, base: List[Dict[str, Any]], quiet: bool,
                     redact: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The two cheap extra passes (spec section 7). Each runs on its own fresh adapter
    after the base run, so the base run's records and records_hash never change; the
    passes keep question ids and ranks, never query or record text. A pass that fails
    is reported as not run; it never costs the base run its results file."""
    log = (lambda *a: None) if quiet else print
    window = band["window"]
    passes: Dict[str, Any] = {}
    ran: List[Dict[str, Any]] = []
    if "order" in names:
        if live:
            passes["order"] = {"available": False,
                               "reason": "live mode does not ingest, so there is no ingest order to vary"}
        else:
            log("\nExtra pass: order sensitivity (the corpus ingested in a second, shuffled order)...")
            try:
                ranks, _ = run_pass(adapter_factory, shuffled_records(pool.records, ORDER_PASS_SEED), False,
                                    lambda ad: pass_search(ad, pool, questions, consts, window,
                                                           [q["question"] for q in questions]))
                passes["order"] = summarize_order(base, ranks, band)
                ran.append({"name": "order", "seed": ORDER_PASS_SEED, "ingest_order": "sorted by sha256(seed:id)"})
            except Exception as exc:
                passes["order"] = {"available": False, "reason": "the pass failed: " +
                                   describe_error("the order pass", exc, redact)}
    if "query" in names:
        log("\nExtra pass: query robustness (two variants of every question)...")
        try:
            by_variant, fingerprint = run_pass(
                adapter_factory, None if live else pool.records, live,
                lambda ad: {v: pass_search(ad, pool, questions, consts, window,
                                           [rc.query_variant(q["question"], v) for q in questions])
                            for v in rc.QUERY_VARIANTS})
            passes["query"] = summarize_query(base, by_variant, band)
            if live:
                passes["query"]["pool_sha256"] = fingerprint
            stop = "\n".join(sorted(rc.STOPWORDS)).encode("utf-8")
            ran.append({"name": "query", "variants": list(rc.QUERY_VARIANTS), "stopwords": len(rc.STOPWORDS),
                        "stopwords_sha256": "sha256:" + hashlib.sha256(stop).hexdigest()})
        except Exception as exc:
            passes["query"] = {"available": False, "reason": "the pass failed: " +
                               describe_error("the query pass", exc, redact)}
    return passes, ran


def inside_repo(path: str) -> bool:
    """path is already resolved with os.path.realpath on the raw string: abspath first
    would fold "link/.." away before the link is followed, and open() would not."""
    p = os.path.realpath(path)
    root = os.path.realpath(REPO_DIR)
    return p == root or p.startswith(root + os.sep)


def finish_run(run: Dict[str, Any], adapter: BaseMemoryAdapter, questions: List[Dict[str, Any]], pool: Any,
               band: Dict[str, Any], redact: bool) -> Dict[str, Any]:
    run["questions"] = questions
    run["adapter_class"] = type(adapter)
    run["abstention"] = apply_abstention(run)  # sets each record's 'abstained' before hashing
    run["agg"] = aggregate(run, pool, band)    # from the real texts, before any redaction
    if redact:
        redact_records(run["per_question"])
    run["records_hash"] = records_hash(run["per_question"])  # of the records as written
    return run


def run_benchmark(
    adapter_factory: Callable[[], BaseMemoryAdapter],
    adapter_name: str,
    questions_path: str,
    corpus_path: Optional[str] = None,
    output_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    verify: bool = False,
    quiet: bool = False,
    live: bool = False,
    redact: bool = False,
    passes: Optional[List[str]] = None,
    store_ids: bool = False,
    source: str = SOURCE_ADAPTER,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the benchmark once (or twice with verify) and return the result dict.

    File mode ingests corpus_path into a fresh store. Live mode (spec section 10)
    searches a store that already holds its records: no corpus, no ingest, the pool
    is whatever list_ids() says, and the results never go under this repository.

    source says who ran the system. SOURCE_ADAPTER: the harness did, through an adapter.
    SOURCE_RESULTS_FILE: the system ran somewhere else and its hits were read from a
    results file (results_file.py). Scoring is the same code either way. What changes is
    what the harness can vouch for: a results file is one run the harness did not make,
    so the result is never verified, the extra passes cannot run, and no search time is
    reported unless the file reports its own. notes are added to the result's notes."""
    if source not in (SOURCE_ADAPTER, SOURCE_RESULTS_FILE):
        raise InputError(f"unknown source {source!r}")
    from_file = source == SOURCE_RESULTS_FILE
    if from_file and (live or verify or passes):
        raise InputError("a results file is scored in file mode, once: it holds one run, so there is nothing "
                         "to run twice and no system to ask the extra passes' questions")
    runs = []
    csv_target = csv_path or (os.path.splitext(output_path)[0] + "_recall_curve.csv" if output_path else None)
    breakdown_target = breakdown_csv_path(csv_target) if csv_target else None
    if live:
        if store_ids:
            raise InputError("--store-ids is for file mode; live mode always uses the store's own ids")
        output_path = os.path.realpath(output_path) if output_path else None
        csv_target = os.path.realpath(csv_target) if csv_target else None
        breakdown_target = os.path.realpath(breakdown_target) if breakdown_target else None
        for path in (output_path, csv_target, breakdown_target):
            if path and inside_repo(path):
                raise InputError(f"live results never go under the repository ({path}); they hold store "
                                 "content. Write them next to the question file or pass --output elsewhere")
        questions = load_questions(questions_path, None)
        qset = load_qset_manifest(questions_path, None)
        fixed = None
        for attempt in range(2 if verify else 1):
            if verify and not quiet:
                print(f"\nRun {attempt + 1} of 2")
            adapter = adapter_factory()
            run = run_live_once(adapter, questions, questions_path, fixed=fixed, quiet=quiet, redact=redact)
            fixed = run.pop("fixed")
            runs.append(finish_run(run, adapter, questions, fixed["pool"], fixed["band"], redact))
        pool, band, consts = fixed["pool"], fixed["band"], fixed["consts"]
    else:
        records = load_corpus(corpus_path, store_ids=store_ids)
        pool = Pool(records)
        questions = load_questions(questions_path, pool.id_set)
        qset = load_qset_manifest(questions_path, corpus_path)
        band = rc.band_for(pool.N)
        consts = [question_constants(q, pool) for q in questions]
        if not quiet:
            print(f"Loaded {len(questions)} questions and {pool.N} records; band {band['band']}, "
                  f"window {band['window']}")
        for attempt in range(2 if verify else 1):
            if verify and not quiet:
                print(f"\nRun {attempt + 1} of 2")
            adapter = adapter_factory()
            run = run_once(adapter, pool, questions, consts, band["window"], quiet=quiet)
            if from_file:
                # The harness did not time the system, so it reports no time of its own: only
                # what the results file says each search took, where it says so.
                reported = getattr(adapter, "reported_ms", {})
                for rec in run["per_question"]:
                    rec["search_time_ms"] = reported.get(rec["id"])
                run["ingest_s"] = None
                run["contract"] = CONTRACT_RESULTS_FILE
            runs.append(finish_run(run, adapter, questions, pool, band, redact))

    first = runs[0]
    aggs = [r.pop("agg") for r in runs]
    agg = aggs[0]
    extra, passes_ran = run_extra_passes(passes or [], adapter_factory, pool, questions, consts, band, live,
                                         first["per_question"], quiet, redact=redact)

    if from_file:
        verification = {"status": "no", "runs": 1, "records_hash": [first["records_hash"]],
                        "detail": "scored from a results file: the system ran outside the harness, so the "
                                  "harness cannot check that two runs agree. The score itself follows from the "
                                  "results file, whose SHA-256 is in the manifest"}
    elif not verify:
        verification = {"status": "no", "runs": 1, "records_hash": [first["records_hash"]],
                        "detail": "single run; pass --verify to check determinism"}
    elif runs[0]["records_hash"] == runs[1]["records_hash"]:
        verification = {"status": "yes", "runs": 2, "records_hash": [r["records_hash"] for r in runs],
                        "detail": "two runs, per-question records byte-identical"}
    else:
        diff = [a["id"] for a, b in zip(runs[0]["per_question"], runs[1]["per_question"])
                if records_hash([a]) != records_hash([b])]
        h0, h1 = aggs[0]["headline"], aggs[1]["headline"]
        def both(x: Optional[float], y: Optional[float]) -> bool:
            return x is not None and y is not None

        spread = {ks: {"mean": round((h0["recall"][ks] + h1["recall"][ks]) / 2, 2)
                       if both(h0["recall"][ks], h1["recall"][ks]) else None,
                       "spread": round(abs(h0["recall"][ks] - h1["recall"][ks]), 2)
                       if both(h0["recall"][ks], h1["recall"][ks]) else None}
                  for ks in h0["recall"]}
        verification = {"status": "unstable", "runs": 2, "records_hash": [r["records_hash"] for r in runs],
                        "differing_questions": diff, "headline_mean_and_spread": spread,
                        "detail": f"{len(diff)} questions differ between the two runs"}
        # The reported headline is the mean of the two runs, with the spread beside it (spec section 8).
        mean_h = dict(h0)
        mean_h["recall"] = {ks: v["mean"] for ks, v in spread.items()}
        mean_h["skill"] = {ks: (None if h0["chance"][ks] is None or mean_h["recall"][ks] is None else
                                round(100.0 * rc.skill(mean_h["recall"][ks] / 100.0, h0["chance"][ks] / 100.0), 2))
                           for ks in spread}
        mean_h["mrr"] = round((h0["mrr"] + h1["mrr"]) / 2, 4) if both(h0["mrr"], h1["mrr"]) else None
        mean_h["spread"] = {ks: v["spread"] for ks, v in spread.items()}
        agg = dict(agg, headline=mean_h)

    if verification["status"] == "unstable":
        for block in extra.values():
            if block.get("available"):
                block["note"] = ("the base run was unstable, so these differences include run-to-run noise; "
                                 "the passes compare against its first run")

    if live:
        # The pool is fingerprinted at the start and the end of every run; any
        # difference means the store changed under the run (spec section 10).
        hashes = [h for r in runs for h in r["pool_hashes"]]
        end_errors = [e for r in runs for e in r["end_errors"]]
        fingerprint_errors = [e for r in runs for e in r["fingerprint_errors"]]
        changed = None if all(h is None for h in hashes) else len(set(hashes)) > 1
        if fingerprint_errors and changed is not None:
            changed = True  # a list_ids() that raised at the end is a lost fingerprint
        corpus_info = {"mode": "live", "N": pool.N, "list_ids": pool.N is not None,
                       "pool_sha256": hashes[0], "pool_sha256_end": hashes[-1], "pool_changed": changed,
                       "id_scheme": "store"}
        if fingerprint_errors:
            verification = dict(verification, status="no", pool_hashes=hashes,
                                detail="list_ids() failed at the end of the run (" + "; ".join(fingerprint_errors) +
                                       "), so the pool cannot be confirmed unchanged and the result cannot be "
                                       "verified; the searches were kept")
        elif changed:
            verification = dict(verification, status="no", pool_hashes=hashes,
                                detail="pool changed during the run: the sorted id list at the end differs from "
                                       "the start, so the result cannot be verified; rerun on a quiet store or "
                                       "a snapshot")
        elif pool.N is None:
            verification = dict(verification, detail=verification["detail"] + "; the pool is not "
                                "fingerprinted, since the adapter has no list_ids()")
        if end_errors:
            verification = dict(verification, end_errors=end_errors)  # kept on record either way
        if extra.get("query", {}).get("available"):
            fp = extra["query"].get("pool_sha256")
            extra["query"]["pool_same_as_base"] = None if fp is None or hashes[0] is None else fp == hashes[0]
    else:
        corpus_info = {"file": os.path.basename(corpus_path), "sha256": rc.file_sha256(corpus_path), "N": pool.N,
                       "id_scheme": "store" if store_ids else "content"}

    # The double run covers the base run; a pass runs once and carries the base run's
    # final verification status (spec section 8).
    for block in extra.values():
        if block.get("available"):
            block["base_verification"] = verification["status"]

    cls = first["adapter_class"]
    manifest = {
        "spec": rc.SPEC_VERSION,
        "harness_version": {"recall": rc.HARNESS_VERSION, **harness_commit(),
                            "source_sha256": rc.harness_source_sha256()},
        "qset": {"version": qset["version"], "file": os.path.basename(questions_path), "sha256": qset["sha256"],
                 "manifest_sha256": qset.get("manifest_sha256"), "manifest_matches": qset["matches"],
                 "status": qset["status"], "gates_passed": qset["gates_passed"],
                 "failed_gates": qset["failed_gates"]},
        "corpus": corpus_info,
        "distractors": None,
        "adapter": {"name": adapter_name, "module": cls.__module__, "class": cls.__name__,
                    "contract": first["contract"], "version": first["version"]},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "params": {"mode": "live" if live else "file", "scoring": "retrieval",
                   "track": "live" if live else "none", "band": band["band"], "window": band["window"],
                   "k_set": band["k_set"], "headline_k": band["headline_k"], "extra_passes": passes_ran,
                   "redacted": redact},
        "records_hash": first["records_hash"],
    }
    manifest_sha = "sha256:" + hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    if live:
        reasons = ["track=live (a live result is compared to a track only on skill, with N stated, "
                   "and is never a leaderboard entry)"]
        if pool.N is None:
            reasons.append("pool size unknown: the adapter has no list_ids()")
        elif band["band"] == "none":
            reasons.append("small pool")
        if corpus_info["pool_changed"]:
            reasons.append("pool changed during the run")
    else:
        reasons = ["track=none (standard tracks need the distractor files, a later release)"]
        if store_ids:
            reasons.append("store ids (private set)")
        if band["band"] == "none":
            reasons.append("small pool")
    if not qset["matches"]:
        reasons.append("question-set manifest missing or does not match the files")
    elif not qset["gates_passed"]:
        reasons.append(f"question set {qset['version']} is {qset['status']}: gates failed "
                       f"({', '.join(qset['failed_gates'])})")
    if from_file:
        reasons.append("scored from a results file (the system ran outside the harness)")
    if verification["status"] != "yes":
        reasons.append("not verified" if verification["status"] == "no" else "unstable")
    missing = [label for name, label in (("order", "order sensitivity"), ("query", "query robustness"))
               if name not in {p["name"] for p in passes_ran}]
    if missing:
        reasons.append(f"extra passes not run ({', '.join(missing)})")
    standard = {"eligible": False, "reasons": reasons}

    result_notes = list(notes or [])
    if verification["status"] == "unstable":
        result_notes.append("the per-category and per-difficulty tables come from the first run, like every per-question "
                     "record; the headline is the mean of the two runs")
    if store_ids and first.get("text_only_hits"):
        result_notes.append(f"{first['text_only_hits']} hits came back as bare text, which the runner hashes to a content "
                     "id that never matches a store id; with --store-ids an adapter must return the ids it was given")
    agg = dict(agg, extra_passes=extra, notes=result_notes)
    line = result_line(agg, manifest, verification["status"], standard["eligible"])
    result = {
        "recall": rc.SPEC_VERSION,
        "result_line": line,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "verification": verification,
        "standard": standard,
        **agg,
        "per_question": first["per_question"],
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    if not quiet:
        print_report(agg, manifest, verification, standard, line)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if not quiet:
            print(f"\nResults saved to {output_path}")
        write_csv(csv_target, agg)
        write_breakdown_csv(breakdown_target, agg)
        if not quiet:
            print(f"Recall curve CSV saved to {csv_target}")
            print(f"Breakdown CSV saved to {breakdown_target}")
    elif csv_target:
        write_csv(csv_target, agg)
        write_breakdown_csv(breakdown_target, agg)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f'Parsica Recall ({rc.SPEC_VERSION}): retrieval benchmark for agent memory systems')
    parser.add_argument('--questions', default=os.path.join(DEFAULT_QSET_DIR, 'questions.json'),
                        help='questions.json (default: the current question set)')
    parser.add_argument('--corpus', default=None,
                        help='corpus.jsonl, file mode only (default: the current question set\'s corpus)')
    parser.add_argument('--live', action='store_true',
                        help='Live mode: search a store that already holds its records (needs --adapter custom). '
                             'No ingest; the pool comes from the adapter\'s list_ids(); results go next to '
                             '--questions and never under the folder the harness sits in')
    parser.add_argument('--passes', default='',
                        help='Extra passes, comma-separated: order (file mode: the corpus ingested in a second, '
                             'shuffled order) and query (two variants of every question). Example: --passes order,query')
    parser.add_argument('--store-ids', action='store_true',
                        help='File mode, private sets: keep the corpus ids as given (a live store\'s own stable '
                             'ids) instead of requiring content ids; never a standard result')
    parser.add_argument('--redact', action='store_true',
                        help='Replace each returned record text with its text_hash and drop the question, '
                             'answer and alternate fields; ids, ranks and scores stay (use it for any results '
                             'you share)')
    parser.add_argument('--adapter', default='bm25',
                        help='A reference adapter (empty, random, echo, bm25, bm25stop, oracle), or "custom" '
                             'with --custom-adapter for a memory system of your own (default: bm25)')
    parser.add_argument('--custom-adapter', default=None,
                        help='Dotted path to your adapter class, e.g. my_module.MyAdapter (with --adapter custom)')
    parser.add_argument('--verify', action='store_true',
                        help='Run twice in one invocation and compare the per-question records')
    parser.add_argument('--output', default=None,
                        help='Results JSON path (default: results/<qset>_<adapter>.json; in live mode '
                             'recall_live_<adapter>.json next to --questions)')
    parser.add_argument('--csv', default=None,
                        help='Recall curve CSV path (default: next to --output); the breakdown CSV goes beside it, '
                             'x_breakdowns.csv for x_recall_curve.csv')
    parser.add_argument('--search-limit', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.search_limit is not None:
        parser.error("--search-limit is gone: the window is set by the pool size (spec section 6), "
                     "so every system at the same N is judged on the same window")

    passes = [p.strip() for p in args.passes.split(',') if p.strip()]
    unknown = [p for p in passes if p not in PASSES]
    if unknown:
        parser.error(f"unknown pass {unknown[0]!r}: the passes are {', '.join(PASSES)}")

    if args.live:
        if args.adapter != 'custom':
            parser.error("--live searches a store that already holds its records: pass --adapter custom "
                         "--custom-adapter your_module.YourStoreAdapter; the reference adapters are file mode")
        if args.corpus:
            parser.error("--corpus has no meaning with --live: the store is the pool")
        if args.store_ids:
            parser.error("--store-ids is for file mode; live mode always uses the store's own ids")

    if args.adapter == 'custom':
        if not args.custom_adapter:
            parser.error("--custom-adapter is required with --adapter custom")
        adapter_name = args.custom_adapter
        factory = lambda: get_custom_adapter(args.custom_adapter)
    else:
        adapter_name = args.adapter
        try:
            get_adapter(args.adapter)  # fail fast on an unknown name
        except ValueError as exc:
            parser.error(str(exc))
        factory = lambda: get_adapter(args.adapter)

    if args.adapter == 'oracle':
        os.environ['RECALL_QFILE'] = os.path.abspath(args.questions)

    output = args.output
    safe = adapter_name.replace('/', '_').replace('.', '_')
    if output is None and args.live:
        output = os.path.join(os.path.dirname(os.path.abspath(args.questions)), f'recall_live_{safe}.json')
    elif output is None:
        qset_name = os.path.basename(os.path.dirname(os.path.abspath(args.questions))) or "qset"
        output = os.path.join(os.getcwd(), 'results', f'{qset_name}_{safe}.json')
    corpus = None if args.live else (args.corpus or os.path.join(DEFAULT_QSET_DIR, 'corpus.jsonl'))

    try:
        run_benchmark(factory, adapter_name, args.questions, corpus, output_path=output,
                      csv_path=args.csv, verify=args.verify, live=args.live,
                      redact=args.redact, passes=passes, store_ids=args.store_ids)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    # Determinism: Python's per-process string-hash randomization can leak into an
    # adapter's tie-breaking. Pin the hash seed and re-exec so every run is
    # bit-identical without the caller having to set an environment variable.
    _status = rc.pin_hash_seed(f"{rc.PACKAGE}.run_benchmark", sys.argv[1:])
    sys.exit(main() if _status is None else _status)

#!/usr/bin/env python3
"""
The no-code path: Parsica Recall for a memory system that has no Python adapter.

Any system, in any language, takes the benchmark in three steps:

    parsica-recall export-task --out ~/recall-task
        writes queries.jsonl (question ids and question text, nothing else), a copy of
        the corpus, and task.json (how many hits to return, and the hashes of both files)

    load corpus.jsonl into your system, run every query, and write one line per query
    to a results file:
        {"qid": "q_0001", "hits": [{"id": "3f9a1c0e7b2d4a61", "score": 12.5}, ...]}

    parsica-recall score ~/recall-task/results.jsonl
        scores the file with the scorer an adapter run uses, and prints the same report

The results file (format 1) is JSON lines in UTF-8:

    {"recall_results": 1, "system": {"name": "my-memory", "version": "0.3", "engine_class": "hybrid",
                                     "models": ["my-embedder"], "network": false}}
    {"qid": "q_0001", "hits": ["3f9a1c0e7b2d4a61", "0b72a04ed6f91453"]}
    {"qid": "q_0002", "hits": [{"id": "7eb662465738d319", "score": 0.83}, {"text": "a record, word for word"}]}
    {"qid": "q_0003", "hits": [], "ms": 41.7}

- The first line is optional. It says what the system is; without it the result says
  unknown. engine_class is one of lexical, embedding, hybrid, llm_assisted. With
  "abstain_threshold" and a score on every hit, abstention is scored (spec section 5).
- Every question of the set needs exactly one line. hits is the system's answer, best
  first. A hit is a record id (a string), or an object with "id" and, when the system has
  one, "score" (any number, higher is better). A system that cannot keep record ids gives
  {"text": ...} with the record's text word for word, and the scorer finds the record from
  the text, as it does for an adapter that returns bare text.
- An empty list says the system returned nothing. Hits past the window are not read, as
  for any adapter, and a record returned twice burns its second slot.
- "ms" is optional: what the system says the search took, in milliseconds. The harness
  did not time it, and the report says whose number it is.

What a results file cannot carry. It is one run, made outside the harness. So the result
always says verified=no, the extra passes do not run, and it is not a standard result
(spec sections 7 and 8). The score itself is as exact as any other: it follows from the
file alone, and the file's SHA-256 is in the result's manifest.

Exit status: 0 when the task or the result was written, 2 on input that is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

from . import recall_common as rc
from . import run_benchmark as rb
from .adapters import BaseMemoryAdapter

FORMAT = 1
ENGINE_CLASSES = ("lexical", "embedding", "hybrid", "llm_assisted")
HEADER_KEYS = ("recall_results", "system")
SYSTEM_KEYS = ("name", "version", "engine_class", "models", "network", "abstain_threshold")
LINE_KEYS = ("qid", "hits", "ms")
HIT_KEYS = ("id", "text", "score")
TASK_FILES = ("queries.jsonl", "corpus.jsonl", "task.json")
UNNAMED = "unnamed"
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _no_constant(name: str):
    raise ValueError(f"{name} is not a JSON number")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and value not in (float("inf"), float("-inf"))


def _extra_keys(obj: Dict[str, Any], allowed) -> List[str]:
    return sorted(k for k in obj if k not in allowed)


def _read_system(system: Any, where: str, problems: List[str]) -> Dict[str, Any]:
    """The header's "system" block, checked: what the file says the system is."""
    if not isinstance(system, dict):
        problems.append(f'{where}: "system" is an object, such as {{"name": "my-memory", "version": "0.3"}}')
        return {}
    extra = _extra_keys(system, SYSTEM_KEYS)
    if extra:
        problems.append(f'{where}: unknown key in "system": {", ".join(extra)} (the keys are {", ".join(SYSTEM_KEYS)})')
    out: Dict[str, Any] = {}
    for key in ("name", "version"):
        if key in system:
            if isinstance(system[key], bool) or not isinstance(system[key], (str, int, float)) or not str(system[key]).strip():
                problems.append(f'{where}: "{key}" is a short string')
            else:
                out[key] = str(system[key]).strip()
    if "engine_class" in system:
        if system["engine_class"] not in ENGINE_CLASSES:
            problems.append(f'{where}: "engine_class" is one of {", ".join(ENGINE_CLASSES)}')
        else:
            out["engine_class"] = system["engine_class"]
    if "models" in system:
        if not isinstance(system["models"], list) or not all(isinstance(m, str) and m.strip() for m in system["models"]):
            problems.append(f'{where}: "models" is a list of model names, or an empty list')
        else:
            out["models"] = [m.strip() for m in system["models"]]
    if "network" in system:
        if not isinstance(system["network"], bool):
            problems.append(f'{where}: "network" is true or false')
        else:
            out["network"] = system["network"]
    if "abstain_threshold" in system:
        if not _is_number(system["abstain_threshold"]):
            problems.append(f'{where}: "abstain_threshold" is a number')
        else:
            out["abstain_threshold"] = float(system["abstain_threshold"])
    return out


def _read_hit(raw: Any, where: str, problems: List[str]) -> Optional[Dict[str, Any]]:
    """One hit as {"id", "text", "score"}, or None when it is refused."""
    if isinstance(raw, str):
        if not raw.strip():
            problems.append(f"{where}: an empty string is not a record id")
            return None
        return {"id": raw, "text": None, "score": None}
    if not isinstance(raw, dict):
        problems.append(f'{where}: a hit is a record id (a string) or an object with "id" or "text"')
        return None
    extra = _extra_keys(raw, HIT_KEYS)
    if extra:
        problems.append(f"{where}: unknown key in a hit: {', '.join(extra)} (the keys are {', '.join(HIT_KEYS)})")
        return None
    hid, text, score = raw.get("id"), raw.get("text"), raw.get("score")
    if hid is None and text is None:
        problems.append(f'{where}: a hit needs "id", or "text" when the system keeps no ids')
        return None
    if hid is not None and (not isinstance(hid, str) or not hid.strip()):
        problems.append(f'{where}: "id" is the record id, a non-empty string')
        return None
    if text is not None and not isinstance(text, str):
        problems.append(f'{where}: "text" is a string')
        return None
    if score is not None and not _is_number(score):
        problems.append(f'{where}: "score" is a number (higher is better), or left out')
        return None
    return {"id": hid, "text": text, "score": None if score is None else float(score)}


def read_results(path: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reads and checks a results file against the question set it answers. Returns the
    system block, the hits per question id, the search times the file reports, and the
    file's SHA-256. Anything wrong is refused with its line number: a score is only worth
    printing when the file says exactly one thing."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise rb.InputError(f"{path}: cannot read the results file: {exc}")
    known = [q["id"] for q in questions]
    known_set = set(known)
    problems: List[str] = []
    system: Dict[str, Any] = {}
    hits: Dict[str, List[Dict[str, Any]]] = {}
    reported: Dict[str, float] = {}
    first_content = True
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {lineno}"
        is_first, first_content = first_content, False
        try:
            obj = json.loads(line, parse_constant=_no_constant)
        except ValueError as exc:
            problems.append(f"{where}: not a JSON line ({exc})")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{where}: each line is one JSON object")
            continue
        if "qid" not in obj:  # the header
            if not is_first:
                problems.append(f'{where}: no "qid"; only the first line may be the header '
                                '("recall_results" and "system")')
                continue
            extra = _extra_keys(obj, HEADER_KEYS)
            if extra:
                problems.append(f"{where}: unknown key in the header: {', '.join(extra)} "
                                f"(the keys are {', '.join(HEADER_KEYS)}; a result line has \"qid\")")
            if "recall_results" in obj and obj["recall_results"] != FORMAT:
                problems.append(f'{where}: "recall_results" is {obj["recall_results"]!r}; this harness reads '
                                f"format {FORMAT}")
            if "system" in obj:
                system = _read_system(obj["system"], where, problems)
            continue
        extra = _extra_keys(obj, LINE_KEYS)
        if extra:
            problems.append(f"{where}: unknown key: {', '.join(extra)} (the keys are {', '.join(LINE_KEYS)})")
        qid = obj["qid"]
        if not isinstance(qid, str) or qid not in known_set:
            problems.append(f"{where}: qid {qid!r} is not a question of this set")
            continue
        if qid in hits:
            problems.append(f"{where}: qid {qid} has a line already; one line per question")
            continue
        raw_hits = obj.get("hits")
        if not isinstance(raw_hits, list):
            problems.append(f'{where}: "hits" is a list, best first; an empty list says the system returned nothing')
            hits[qid] = []
            continue
        parsed = [_read_hit(raw, f"{where}, hit {i + 1}", problems) for i, raw in enumerate(raw_hits)]
        hits[qid] = [h for h in parsed if h is not None]
        if "ms" in obj:
            if not _is_number(obj["ms"]) or obj["ms"] < 0:
                problems.append(f'{where}: "ms" is the search time in milliseconds, a number of 0 or more')
            else:
                reported[qid] = round(float(obj["ms"]), 3)
    missing = [qid for qid in known if qid not in hits]
    if missing:
        shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
        problems.append(f"no line for {len(missing)} of the set's {len(known)} questions ({shown}); every question "
                        "needs a line, with an empty hits list when the system returned nothing")
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise rb.InputError(f"{path}: refusing the results file:\n  {shown}{more}")
    return {"path": path, "sha256": rc.file_sha256(path), "format": FORMAT, "system": system, "hits": hits,
            "reported_ms": reported}


def system_name(results: Dict[str, Any]) -> str:
    """The name a result line carries for a scored file: file:<the system's name>, with
    anything that could break the line (spaces, =) written as a dash."""
    name = _SLUG.sub("-", str(results["system"].get("name") or UNNAMED)).strip("-") or UNNAMED
    return f"file:{name}"


def describe(results: Dict[str, Any], record_ids: set, window: int) -> List[str]:
    """What a person should know about the file before reading its score: notes that go
    into the result and the report."""
    notes = []
    unknown = Counter()
    over = 0
    as_text = 0
    for qid, hits in results["hits"].items():
        if len(hits) > window:
            over += 1
        for h in hits[:window]:
            if h["id"] is None:
                as_text += 1
            elif h["id"] not in record_ids:
                unknown[qid] += 1
    if unknown:
        notes.append(f"{sum(unknown.values())} returned ids, in {len(unknown)} questions, are not record ids of this "
                     "corpus, so they can never be gold: a system has to return the ids it was given (or the "
                     "record's text)")
    if as_text:
        notes.append(f"{as_text} hits came as text and were matched to a record by the content id of that text, "
                     "which only works when the text is the record word for word")
    if over:
        notes.append(f"{over} questions came with more than the window of {window} hits; the scorer read the "
                     f"first {window} of each")
    return notes


class ResultsFileAdapter(BaseMemoryAdapter):
    """Answers each question with the hits a results file holds for it. The system ran
    somewhere else; this class only hands its answers to the scorer, through the same
    door an adapter uses, so a results file and an adapter run are scored by one code
    path. An id hit takes its text from the corpus, so the diagnostic tiers that read
    text (exact, family, redundancy) see what an adapter that kept the records would
    have returned; an id the corpus does not hold has no text and is never gold."""

    def __init__(self, results: Dict[str, Any], questions: List[Dict[str, Any]]):
        self.results = results
        self.reported_ms = dict(results["reported_ms"])
        self._qids: Dict[str, List[str]] = {}
        for q in questions:
            self._qids.setdefault(q["question"], []).append(q["id"])
        self._asked: Counter = Counter()
        self._text: Dict[str, str] = {}

    def setup(self, workspace: str, mode: str = "file") -> None:
        if mode != "file":
            raise ValueError("a results file is scored in file mode: the corpus file is the pool")
        self._asked.clear()
        self._text = {}

    def ingest(self, record: Dict[str, Any]) -> None:
        self._text[record["id"]] = record["text"]

    def save(self) -> None:
        pass

    def search(self, query: str, limit: int) -> List[Any]:
        qids = self._qids.get(query)
        if not qids:
            return []
        # Two questions with the same text are asked in file order, so each gets its own line.
        qid = qids[self._asked[query] % len(qids)]
        self._asked[query] += 1
        out: List[Any] = []
        for h in self.results["hits"][qid]:
            if h["id"] is None:
                out.append({"text": h["text"], "score": h["score"]})  # the scorer hashes the text to an id
            else:
                text = h["text"] if h["text"] is not None else self._text.get(h["id"], "")
                out.append((h["id"], text, h["score"]))
        return out

    def teardown(self) -> None:
        self._text = {}

    def version(self) -> Dict[str, Any]:
        system = self.results["system"]
        info: Dict[str, Any] = {
            "version": system.get("version", "unknown"),
            "engine_class": system.get("engine_class", "unknown"),
            "models": list(system.get("models", [])),
            "network": system.get("network"),
            "name": system.get("name", UNNAMED),
            "results_format": self.results["format"],
            "results_sha256": self.results["sha256"],
            "note": "scored from a results file: the system ran outside the harness, and what it is comes from "
                    "the file's own first line",
        }
        if "abstain_threshold" in system:
            info["abstain_threshold"] = system["abstain_threshold"]
        return info


# ---------------------------------------------------------------------------
# export-task
# ---------------------------------------------------------------------------

def export_task(questions_path: str, corpus_path: str, out: str, store_ids: bool = False) -> Dict[str, Any]:
    """Writes the task folder a system needs to take the benchmark without an adapter:
    queries.jsonl (question ids and question text, and nothing else of the question file),
    a byte copy of the corpus, and task.json. Returns task.json's contents."""
    out_real = os.path.realpath(out)
    if rb.inside_repo(out_real):
        raise rb.InputError(f"--out may not be under the folder the harness sits in ({out_real}); a task folder "
                            "holds a copy of the corpus, and for a set built from your own store that is your "
                            "own records. Name a folder somewhere else, such as ~/recall-task")
    if os.path.exists(out_real) and (not os.path.isdir(out_real) or os.listdir(out_real)):
        raise rb.InputError(f"--out {out_real} exists and is not an empty folder; a task folder is never overwritten")
    records = rb.load_corpus(corpus_path, store_ids=store_ids)
    questions = rb.load_questions(questions_path, {r["id"] for r in records})
    band = rc.band_for(len(records))
    qset = rb.load_qset_manifest(questions_path, corpus_path)
    os.makedirs(out_real, exist_ok=True)
    q_out = os.path.join(out_real, "queries.jsonl")
    with open(q_out, "w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps({"qid": q["id"], "query": q["question"]}, ensure_ascii=False) + "\n")
    c_out = os.path.join(out_real, "corpus.jsonl")
    shutil.copyfile(corpus_path, c_out)
    task = {
        "recall_task": 1,
        "spec": rc.SPEC_VERSION,
        "harness": rc.HARNESS_VERSION,
        "qset": {"version": qset["version"], "questions": len(questions), "questions_sha256": qset["sha256"],
                 "status": qset["status"]},
        "queries": {"file": "queries.jsonl", "sha256": rc.file_sha256(q_out), "fields": ["qid", "query"]},
        "corpus": {"file": "corpus.jsonl", "sha256": rc.file_sha256(c_out), "records": len(records),
                   "id_scheme": "store" if store_ids else "content",
                   "fields": "id and text on every record; kind, ts and source where the corpus has them"},
        "band": band["band"],
        "window": band["window"],
        "k_set": band["k_set"],
        "headline_k": band["headline_k"],
        "results": {
            "format": FORMAT,
            "how": f"one JSON line per query, best hit first, at most {band['window']} hits: "
                   '{"qid": "q_0001", "hits": [{"id": "<record id>", "score": 1.0}]}. An optional first line says '
                   'what the system is: {"recall_results": 1, "system": {"name": "...", "version": "..."}}. '
                   "Every query needs a line; an empty hits list says the system returned nothing.",
            "score_with": "parsica-recall score results.jsonl" + ("" if rc.is_current_qset(qset["version"]) and not store_ids
                                                                  else " --questions <questions.json> --corpus <corpus.jsonl>"
                                                                  + (" --store-ids" if store_ids else "")),
        },
    }
    with open(os.path.join(out_real, "task.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return task


def main_export(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Write the task folder a memory system needs to take Parsica Recall with no adapter: the "
                    "queries (question ids and question text only), a copy of the corpus, and task.json")
    p.add_argument("--out", required=True, help="A new or empty folder, outside the folder the harness sits in")
    p.add_argument("--questions", default=os.path.join(rb.DEFAULT_QSET_DIR, "questions.json"),
                   help="questions.json (default: the current question set)")
    p.add_argument("--corpus", default=None, help="corpus.jsonl (default: the current question set's corpus)")
    p.add_argument("--store-ids", action="store_true",
                   help="A private set that keeps its store's own ids (see parsica-recall run --store-ids)")
    args = p.parse_args(argv)
    corpus = args.corpus or os.path.join(rb.DEFAULT_QSET_DIR, "corpus.jsonl")
    try:
        task = export_task(args.questions, corpus, args.out, store_ids=args.store_ids)
    except (rb.InputError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    out = os.path.realpath(args.out)
    print(f"Wrote the task for question set {task['qset']['version']} to {out}")
    print(f"  queries.jsonl   {task['qset']['questions']} queries: the question id and the question text, nothing else")
    print(f"  corpus.jsonl    {task['corpus']['records']} records to load into your system")
    print("  task.json       the hashes of both, and the window")
    print(f"For every query, return your system's top {task['window']} records, best first (band {task['band']}: "
          f"the window comes from the pool's size, not from the system).")
    print("Write one JSON line per query to a results file, for example:")
    print('  {"qid": "q_0001", "hits": [{"id": "<record id>", "score": 1.0}, {"id": "<record id>", "score": 0.7}]}')
    print(f"Then: {task['results']['score_with']}")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def main_score(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Score a results file that any memory system can write (see parsica-recall export-task). "
                    "The scorer is the one an adapter run uses; the result says verified=no, because the "
                    "harness did not run the system")
    p.add_argument("results", help="The results file: one JSON line per question, {\"qid\": ..., \"hits\": [...]}")
    p.add_argument("--questions", default=os.path.join(rb.DEFAULT_QSET_DIR, "questions.json"),
                   help="questions.json (default: the current question set)")
    p.add_argument("--corpus", default=None, help="corpus.jsonl (default: the current question set's corpus)")
    p.add_argument("--store-ids", action="store_true",
                   help="A private set that keeps its store's own ids; never a standard result")
    p.add_argument("--redact", action="store_true",
                   help="Replace each returned record text with its text_hash and drop the question, answer and "
                        "alternate fields (use it for any results you share from a private set)")
    p.add_argument("--output", default=None, help="Results JSON path (default: results/<qset>_file_<system>.json)")
    p.add_argument("--csv", default=None, help="Recall curve CSV path (default: next to --output)")
    args = p.parse_args(argv)
    corpus = args.corpus or os.path.join(rb.DEFAULT_QSET_DIR, "corpus.jsonl")
    try:
        records = rb.load_corpus(corpus, store_ids=args.store_ids)
        questions = rb.load_questions(args.questions, {r["id"] for r in records})
        results = read_results(args.results, questions)
        name = system_name(results)
        notes = describe(results, {r["id"] for r in records}, rc.band_for(len(records))["window"])
        output = args.output
        if output is None:
            qset_name = os.path.basename(os.path.dirname(os.path.abspath(args.questions))) or "qset"
            output = os.path.join(os.getcwd(), "results", f"{qset_name}_{name.replace(':', '_').replace('.', '_')}.json")
        rb.run_benchmark(lambda: ResultsFileAdapter(results, questions), name, args.questions, corpus,
                         output_path=output, csv_path=args.csv, redact=args.redact, store_ids=args.store_ids,
                         source=rb.SOURCE_RESULTS_FILE, notes=notes)
    except (rb.InputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    print("use: parsica-recall export-task ...   or   parsica-recall score ...", file=sys.stderr)
    sys.exit(2)

#!/usr/bin/env python3
"""
Reads a question set written by generate_questions.py and prints it for a person to read.
Standard library only, and it never calls a model:

    parsica-recall show-questions questions ~/recall-sets/my-store-v1 [--category temporal_awareness] [--full]
    parsica-recall show-questions summary ~/recall-sets/my-store-v1
    parsica-recall show-questions held ~/recall-sets/my-store-v1 [--category implicit_preference]

questions: every frozen question by category, with its answer, alternates, difficulty,
whether its gold sits in the core, and the text of each gold record (cut at 400
characters unless --full). summary: the manifest's counts per category, the cost, the
lint, the oracle and the split. held: every candidate that did not freeze, with its fate
and the reason, from generation/candidates.jsonl (a public half keeps no generation
record, so point it at the sealed half).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

TEXT_LIMIT = 400


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cut(text: str, full: bool) -> str:
    text = " ".join(str(text).split())
    return text if full or len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + " [...]"


def show_questions(qset: str, category: Optional[str] = None, full: bool = False) -> None:
    questions = _load(os.path.join(qset, "questions.json"))
    corpus = {r["id"]: r for r in _load_jsonl(os.path.join(qset, "corpus.jsonl"))}
    manifest = _load(os.path.join(qset, "manifest.json"))
    subset = manifest.get("core_subset")
    in_core = set(subset["ids"]) if subset else None
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in questions:
        if not category or q["category"] == category:
            by_cat[q["category"]].append(q)
    print(f"Set {manifest['qset_version']} ({manifest['status_reason']}): {len(questions)} questions"
          + (f", {subset['questions']} with every gold record in the first {subset['records']}" if subset else ""))
    for cat, qs in by_cat.items():
        print(f"\n######## {cat}: {len(qs)}")
        for q in qs:
            where = "" if in_core is None else (", core" if q["id"] in in_core else ", tail")
            print(f"\n{q['id']} ({q['difficulty']}, {q['answer_type']}{where})")
            print(f"  Q: {q['question']}")
            alts = "; ".join(q["acceptable_answers"]) or "none"
            print(f"  A: {q['answer']}  (alternates: {alts})")
            current = q.get("must_rank_first")
            current = set(current if isinstance(current, list) else [current] if current else [])
            for g in q["gold_ids"]:
                rec = corpus.get(g)
                if rec is None:
                    print(f"  gold {g}: not in this corpus")
                    continue
                tag = " (current)" if g in current else ""
                when = f" {rec['ts']}" if rec.get("ts") else ""
                print(f"  gold {g}{tag} [{rec.get('kind', 'none')}{when}]: {_cut(rec['text'], full)}")


def show_summary(qset: str) -> None:
    m = _load(os.path.join(qset, "manifest.json"))
    print(f"Set {m['qset_version']}: {m['status_reason']}")
    model = m["model"]
    cost = f"${model['cost_usd']:.2f}" if model.get("cost_usd") is not None else "no price on file"
    print(f"Model: asked for {model['requested']}, served by {', '.join(model['served']) or 'none'}; "
          f"{model['calls']} calls ({model.get('retry_calls', 0)} retries), {model['input_tokens']:,} input and "
          f"{model['output_tokens']:,} output tokens; {cost} ({model.get('cost_note', '')})")
    print(f"Time: {model['started']} to {model['finished']} (UTC)")
    per = m["generation"]["per_category"]
    head = ("category", "target", "units", "core", "cand", "skip", "check", "screen", "held", "retry", "drop",
            "frozen", "surplus")
    print("\n  " + f"{head[0]:22}" + "".join(f"{h:>8}" for h in head[1:]) + "  stopped")
    for cat, n in per.items():
        cells = (n["target"], n["units_shown"], n.get("units_shown_core", "-"), n["candidates"], n["skipped"],
                 n["dropped_by_checks"], n["dropped_by_screen"], n["held_by_lint"], n.get("retry_calls", 0),
                 n["dropped_by_lint"], n["frozen"], n["surplus"])
        print("  " + f"{cat:22}" + "".join(f"{c:>8}" for c in cells) + f"  {n['stopped'] or ''}")
    print(f"\nThis folder: {m['counts']['questions']} questions; by category "
          + ", ".join(f"{k} {v}" for k, v in m["counts"]["by_category"].items()))
    if m.get("core_subset"):
        s = m["core_subset"]
        print(f"Core subset: {s['questions']} questions with every gold record in the first {s['records']}; "
              + ", ".join(f"{k} {v}" for k, v in s["by_category"].items()))
    sampler = m.get("sampler", {})
    if sampler.get("planted_pairs"):
        p = sampler["planted_pairs"]
        print(f"Planted pairs ({p['file']}): used "
              + ", ".join(f"{v} for {k}" for k, v in p["used"].items())
              + ("; left out " + ", ".join(f"{v} with {k}" for k, v in p["skipped"].items()) if p["skipped"] else ""))
    if sampler.get("entity_attribution"):
        ea = sampler["entity_attribution"]
        print(f"Rare names: entity_attribution drew from {ea['records_with_a_rare_name']} eligible memories with a "
              f"name found in at most {ea['max_other_records']} other record"
              + (f" ({ea['in_the_core']} in the core)" if ea.get("in_the_core") is not None else ""))
    retry = m["generation"].get("retry", {})
    if retry.get("padded_retries"):
        print(f"Padded retries dropped: {retry['padded_retries']} of {retry['calls']} retry calls")
    lint = m["lint"]
    print(f"Lint ({lint['spec_version']}): {lint['items_with_errors']} items with errors; gates "
          + ("all pass" if lint["gates_passed"] else "failed: " + ", ".join(lint["failed_gates"])))
    ref_path = os.path.join(qset, "reference.json")
    if os.path.exists(ref_path):
        for name, run in _load(ref_path)["runs"].items():
            print(f"{name}: {run['result_line']}")
    if m.get("split"):
        s = m["split"]
        other = s.get("sealed") or s.get("public")
        print(f"Split: the {s['half']} half, seed {s['seed']}, stratified by {', '.join(s['stratified_by'])}; the "
              f"other half's questions.json has SHA-256 {other['questions_sha256']}")


def show_held(qset: str, category: Optional[str] = None) -> int:
    path = os.path.join(qset, "generation", "candidates.jsonl")
    if not os.path.exists(path):
        print(f"ERROR: {qset} holds no generation/candidates.jsonl; the public half of a split set keeps no "
              "generation record, so point at the sealed half", file=sys.stderr)
        return 2
    rows = [c for c in _load_jsonl(path) if c["fate"] != "frozen" and (not category or c["category"] == category)]
    print(f"{len(rows)} candidates that did not freeze")
    for c in rows:
        item = c["returned"]
        question = item.get("question") if isinstance(item, dict) else item
        answer = item.get("answer") if isinstance(item, dict) else None
        retry = f", retry of {c['retry_of']}" if c.get("retry_of") else ""
        print(f"\n-- {c['category']} candidate {c['n']} (call {c['call']}{retry}): {c['fate']}: {c['reason']}")
        print(f"   Q: {question}")
        if answer is not None:
            alts = "; ".join(item.get("acceptable_answers") or []) or "none"
            print(f"   A: {answer}  (alternates: {alts})")
        for f in c.get("findings") or []:
            if f["severity"] == "error" or f["rule_id"] == "DUPLICATE_FACT":
                print(f"   {f['rule_id']}: {f['message']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:  # Windows consoles default to a code page that cannot print every character
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description="Print a question set written by generate_questions.py")
    p.add_argument("what", choices=("questions", "summary", "held"))
    p.add_argument("qset", help="The set folder (the --out or --sealed-out of generate_questions.py)")
    p.add_argument("--category", default=None, help="questions and held: only this category")
    p.add_argument("--full", action="store_true", help="questions: print every gold record whole")
    args = p.parse_args(argv)
    try:
        if args.what == "questions":
            show_questions(args.qset, args.category, args.full)
        elif args.what == "summary":
            show_summary(args.qset)
        else:
            return show_held(args.qset, args.category)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot read the set in {args.qset}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Reads a pool written by generate_pool.py and prints it for a person to read. Standard
library only, and it never calls a model:

    parsica-recall show-pool records ~/recall-pilot/pool-pilot-1 [--persona p1] [--phase core] [--only C]
    parsica-recall show-pool stats ~/recall-pilot/pool-pilot-1
    parsica-recall show-pool dropped ~/recall-pilot/pool-pilot-1

records: every record, by persona and then by session in time order, with each session's
date and what its plan asked it to include; a record that carries a planted step is tagged
with it, such as {C3 2} (C3, step 2), and --only C shows just those. stats: the manifest's statistics. dropped:
every record the screens dropped, with the reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def planted_tags(pool: str) -> Dict[Any, str]:
    """(session, position) -> the planted step a record carries, such as "C3 2/2", from the
    final states (each step's value keeps the session and position of its record)."""
    tags: Dict[Any, str] = {}
    for st in _load(os.path.join(pool, "generation", "states.json")).values():
        for label, item in st["items"].items():
            for k, v in enumerate(item["values"], 1):
                if v.get("position") is not None:
                    tags[(v["session"], v["position"])] = f"{label} {k}"
    return tags


def show_records(pool: str, persona: Optional[str] = None, phase: Optional[str] = None,
                 only: Optional[str] = None) -> None:
    corpus = {r["id"]: r for r in _load_jsonl(os.path.join(pool, "corpus.jsonl"))}
    where = _load_jsonl(os.path.join(pool, "generation", "records.jsonl"))
    plan = _load(os.path.join(pool, "generation", "plan.json"))
    tags = planted_tags(pool)
    if only:  # only the records that carry a planted step whose label starts with one of these letters
        where = [w for w in where if tags.get((w["session"], w["position"]), " ")[0] in only.upper()]
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for w in where:
        by_session[w["session"]].append(w)
    for pid, p in plan["personas"].items():
        if persona and pid != persona:
            continue
        print(f"\n######## {pid} {p['name']} ({p['file']}): core {p['core_share']}, tail {p['tail_share']}")
        sessions = sorted((s for s in p["sessions"] if s["session"] in by_session and (not phase or s["phase"] == phase)),
                          key=lambda s: s["start"])
        for n, s in enumerate(sessions, 1):
            recs = sorted(by_session[s["session"]], key=lambda w: w["position"])
            print(f"\n=== {pid} {s['phase']} session {n} of {len(sessions)}: {s['session']}, {s['local']} local "
                  f"({s['start']}), {len(recs)} records" + (" (make-up)" if s["makeup"] else ""))
            if s["directives"]:
                print("    plan asked for: " + ", ".join(s["directives"]))
            for w in recs:
                r = corpus[w["id"]]
                tag = tags.get((w["session"], w["position"]))
                print(f"  [{r['kind']}]" + (f" {{{tag}}}" if tag else "") + f" {r['ts']}  {r['text']}")


def show_stats(pool: str) -> None:
    m = _load(os.path.join(pool, "manifest.json"))
    st = m["statistics"]
    print(f"Pool {m['pool']} ({m['status_reason']})")
    print(f"Model: asked for {m['model']['requested']}, served by {', '.join(m['model']['served'])}; "
          f"{m['model']['calls']} calls, {m['model']['input_tokens']} input and {m['model']['output_tokens']} output "
          f"tokens; {m['model']['unusable_calls']} unusable replies, {m['model']['sessions_lost']} sessions lost")
    print(f"Time: {m['model']['started']} to {m['model']['finished']} (UTC)")
    print(f"\nRecords: {st['records']} ({st['core']} core, {st['tail']} tail)")
    for part in ("all", "core", "tail"):
        k = st["by_kind"][part]
        print(f"  {part:5} user {k['user']}, agent {k['agent']}, entry {k['entry']}; entry share {st['entry_share'][part]}")
    print("\nBy persona:")
    print(f"  {'id':5}{'records':>8}{'core':>6}{'tail':>6}{'user':>6}{'agent':>7}{'entry':>7}{'sessions':>10}{'days':>6}"
          "  first ts              last ts")
    for pid, p in st["by_persona"].items():
        print(f"  {pid:5}{p['records']:>8}{p['core']:>6}{p['tail']:>6}{p['user']:>6}{p['agent']:>7}{p['entry']:>7}"
              f"{p['sessions']:>10}{p['active_days']:>6}  {p['first_ts']}  {p['last_ts']}")

    def row(name: str, d: Dict[str, Any]) -> None:
        if not d.get("n"):
            print(f"  {name:28} none")
            return
        print(f"  {name:28} n {d['n']:>6}  min {d['min']:>5}  p5 {d['p5']:>5}  p25 {d['p25']:>5}  p50 {d['p50']:>5}  "
              f"p75 {d['p75']:>5}  p95 {d['p95']:>5}  max {d['max']:>5}  mean {d['mean']}")

    print("\nLengths (words; characters last):")
    for k, d in st["words"].items():
        row(f"words, {k}", d)
    row("characters, all", st["characters"])
    print("\nSessions and days:")
    print(f"  sessions {st['sessions']['count']} ({st['sessions']['core']} core, {st['sessions']['tail']} tail); "
          f"active days {st['active_days']['count']} ({st['active_days']['day']})")
    row("records per session", st["sessions"]["records_per_session"])
    row("records per active day", st["active_days"]["records_per_active_day"])
    row("entries per active day", st["active_days"]["entries_per_active_day"])
    row("sessions per active day", st["active_days"]["sessions_per_active_day"])
    print("\nRecords by month: " + ", ".join(f"{k} {v}" for k, v in st["records_by_month"].items()))
    rr = st["restatement_rate"]
    print(f"\nNear-copy clusters left in the pool: {st['near_copy_clusters']}")
    print(f"Restatement rate: all {rr['all']}; core records against the whole pool {rr['core_against_the_whole_pool']}; "
          f"core against the core {rr['core_against_the_core']}")
    print(f"  ({rr['definition']})")
    print(f"\nDropped: {st['dropped']['total']}; " + ", ".join(f"{k} {v}" for k, v in st["dropped"]["by_reason"].items()))
    d = st["dropped"]
    print("  by phase: " + "; ".join(f"{ph} {d['by_phase'].get(ph, 0)} dropped, drop rate {d['rate_by_phase'].get(ph)}"
                                     for ph in ("core", "tail")) + f" ({d.get('rate', '')})")
    pl = m["plan"]
    print(f"\nPlan: seed {pl['seed']}, {pl['sessions_planned']} sessions planned, {pl['makeup_sessions']} make-up "
          f"sessions; planted items reported by the model: {pl['directives_reported']} of {pl['directives_asked']}")
    for pid, q in pl["quotas"].items():
        print(f"  {pid} quotas: " + ", ".join(f"{k} {v}" for k, v in q.items())
              + (f"; not placed: {', '.join(pl['not_placed'][pid])}" if pid in pl["not_placed"] else ""))


def show_dropped(pool: str) -> None:
    rows = _load_jsonl(os.path.join(pool, "generation", "dropped.jsonl"))
    print(f"{len(rows)} dropped records")
    for d in rows:
        text = d["text"] if d["text"] is not None else "(text not kept: it matched a secret or personal-data shape)"
        print(f"\n-- {d['reason']} ({json.dumps(d['detail'])}): {d['persona']} {d['phase']} {d['session']} "
              f"position {d['position']} [{d['kind']}]")
        print(f"   {text}")


def main(argv: Optional[List[str]] = None) -> int:
    try:  # Windows consoles default to a code page that cannot print every character
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description="Print a pool written by generate_pool.py")
    p.add_argument("what", choices=("records", "stats", "dropped"))
    p.add_argument("pool", help="The pool folder (the --out of generate_pool.py)")
    p.add_argument("--persona", default=None, help="records: only this persona id, such as p1")
    p.add_argument("--phase", default=None, choices=("core", "tail"), help="records: only this phase")
    p.add_argument("--only", default=None,
                   help="records: only the records that carry a planted step of these kinds, such as C (corrections) "
                        "or VRC (changes and corrections)")
    args = p.parse_args(argv)
    try:
        if args.what == "records":
            show_records(args.pool, args.persona, args.phase, args.only)
        elif args.what == "stats":
            show_stats(args.pool)
        else:
            show_dropped(args.pool)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot read the pool in {args.pool}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

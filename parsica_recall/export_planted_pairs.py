#!/usr/bin/env python3
"""
Exports a pool's planted changes as pairs of record ids, for the question generator's
--planted-pairs (docs/generation.md section 5). Standard library only, no model call:

    parsica-recall export-planted-pairs data/pool-v1                 # writes data/pool-v1/planted_pairs.jsonl
    parsica-recall export-planted-pairs data/pool-v1 --out FILE      # writes it elsewhere

It reads the committed pool files alone: the call log's state_update events (one per
planted step the model reported, naming the step's label and the position of the record
that carries it) mapped to record ids through generation/records.jsonl, under the same
rules the pool generator used to fold a step into its state (docs/pool.md section 4): the
step's label was asked for in that session, its record was kept by the screens, its text
holds no credential or personal-data shape, and the first event per label in a session
counts. The persona files are never read.

One line per planted item of the three version types, in persona order and then label
order: {"type", "label", "persona", "earlier", "later"}, where type is version_chain
(knowledge_update), correction (correction_handling) or preference_change
(recency_preference), and earlier and later are the ids of the records that carry the
item's LAST transition: the step before the last, and the last. So the later record holds
the item's current value and nothing later in the pool changes it. An item the pool holds
only one step of has no transition and is left out. The output is deterministic: the same
pool gives the same file byte for byte.

Exit status: 0 when the file was written, 2 on a folder that is not a pool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import generate_pool as gp
from . import generate_questions as gq
from . import lint_validator as lv

# The planted item types the question generator takes, by the letter their labels carry
# (generate_pool.LABELS); the category each feeds is generate_questions.PLANTED_CATEGORY.
PLANTED_TYPES = {"V": "version_chain", "R": "preference_change", "C": "correction"}
_LABEL = re.compile(r"([VRCTLS][1-9][0-9]*)[,:]")


class ExportError(Exception):
    """A folder that is not a pool with its generation record. Exit 2."""


def asked_labels(user: str) -> List[str]:
    """The labels a session's call asked for: the leading label of each must_include line
    of its user message, as the pool generator wrote them (PersonaRun.directive_text)."""
    try:
        lines = json.loads(user).get("must_include") or []
    except (ValueError, AttributeError):
        return []
    out = []
    for line in lines:
        m = _LABEL.match(line) if isinstance(line, str) else None
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _clean(*texts: Any) -> bool:
    """Whether a step's text may count: the pool generator left a step with a credential or
    personal-data shape out of its state (PersonaRun.clean), so it is no planted step here."""
    return not any(isinstance(t, str) and lv.scan_secrets_pii(t) for t in texts)


def planted_steps(folder: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Every planted item's kept steps, in the order the pool folded them into its state:
    (persona, label) -> {"type", "steps": [{"call", "position", "id", "session"}, ...]}."""
    gen = os.path.join(folder, "generation")
    parts = gp.call_parts(gen)
    r_path = os.path.join(gen, "records.jsonl")
    if not parts or not os.path.isfile(r_path):
        raise ExportError(f"{folder} is not a pool with its generation record (generation/calls-NNNN.jsonl and "
                          "records.jsonl)")
    kept: Dict[Tuple[int, int], str] = {}
    try:
        with open(r_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    kept[(row["call"], row["position"])] = row["id"]
        calls = []
        for part in parts:
            with open(part, encoding="utf-8") as f:
                calls += [json.loads(line) for line in f if line.strip()]
        calls.sort(key=lambda c: c["n"])  # the order the sessions ran, persona by persona, phase by phase
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ExportError(f"cannot read the generation record in {folder}: {exc}")
    items: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for call in calls:
        try:
            data = json.loads(call["response"])
        except (ValueError, KeyError, TypeError):
            continue
        update = data.get("state_update") if isinstance(data, dict) else None
        if not isinstance(update, dict):
            continue
        asked = asked_labels(call.get("user", ""))
        reported = set()
        for ev in update.get("events") or []:
            label = ev.get("label") if isinstance(ev, dict) else None
            if not isinstance(label, str) or label not in asked or label in reported:
                continue
            pos = ev.get("record")
            if isinstance(pos, bool) or not isinstance(pos, int):
                continue
            rid = kept.get((call["n"], pos))
            if rid is None:  # the record was dropped by a screen, so the step never entered the state
                continue
            if not _clean(ev.get("subject"), ev.get("value"), ev.get("note")):
                continue
            item = items.setdefault((call["persona"], label), {"type": PLANTED_TYPES.get(label[0]), "steps": []})
            item["steps"].append({"call": call["n"], "position": pos, "id": rid, "session": call["session"]})
            reported.add(label)
    return items


def planted_pairs(folder: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """The pairs to write, in persona and label order, and per type how many items the
    pool reported, how many have a transition, and how many were left out with one step."""
    items = planted_steps(folder)
    rows = []
    counts: Dict[str, Dict[str, int]] = {t: {"items": 0, "pairs": 0, "one_step": 0}
                                         for t in PLANTED_TYPES.values()}
    for (persona, label), item in items.items():
        typ = item["type"]
        if typ is None:
            continue
        counts[typ]["items"] += 1
        steps = item["steps"]
        if len(steps) < 2:
            counts[typ]["one_step"] += 1
            continue
        counts[typ]["pairs"] += 1
        rows.append({"type": typ, "label": label, "persona": persona, "earlier": steps[-2]["id"],
                     "later": steps[-1]["id"]})
    rows.sort(key=lambda r: (int(r["persona"][1:]), gp._label_key(r["label"])))
    return rows, counts


def write_pairs(rows: List[Dict[str, Any]], path: str) -> bytes:
    """Writes the pairs as the generator reads them (generate_questions.read_planted_pairs),
    one object per line with the fields in PLANTED_FIELDS order, LF on every system."""
    data = "".join(json.dumps({k: row[k] for k in gq.PLANTED_FIELDS}, ensure_ascii=False) + "\n"
                   for row in rows).encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return data


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export a pool's planted changes as pairs of record ids")
    p.add_argument("pool", help="The pool folder (data/pool-v1): corpus.jsonl beside a generation/ folder")
    p.add_argument("--out", default=None, help="Where to write (default: planted_pairs.jsonl in the pool folder)")
    args = p.parse_args(argv)
    out = args.out or os.path.join(args.pool, "planted_pairs.jsonl")
    try:
        rows, counts = planted_pairs(args.pool)
        before = None
        if os.path.exists(out):
            with open(out, "rb") as f:
                before = f.read()
        data = write_pairs(rows, out)
    except (ExportError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for typ, n in counts.items():
        print(f"  {typ} ({gq.PLANTED_CATEGORY[typ]}): {n['pairs']} pairs from {n['items']} planted items"
              + (f" ({n['one_step']} with one step, no transition)" if n["one_step"] else ""))
    state = "unchanged" if before == data else ("rewritten" if before is not None else "written")
    print(f"Wrote {len(rows)} planted pairs to {out} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

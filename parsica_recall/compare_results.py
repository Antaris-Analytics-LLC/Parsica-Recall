#!/usr/bin/env python3
"""
Compare two Parsica Recall results (spec section 8).

    parsica-recall compare before.json after.json

Two results are comparable only when they ran the same question set (the SHA-256
of its file), on the same pool (the SHA-256 of the corpus file, or of a live
store's sorted id list, plus the track and distractors), under the same harness
version and commit. A live result whose pool changed during its run has no single
pool and is refused too. Otherwise the tool refuses, says what differs, and exits 2.

A result is pinned to its harness when it records a commit and ran on a tree without
uncommitted changes, or, from an installed copy, which has no commit, when it records
the SHA-256 of the harness's own source files. Two results from different commits,
two whose source hashes differ where either has no commit, or either one unpinned,
are refused the same way; --allow-unpinned turns those refusals into printed notes
for local work and appends unpinned=yes to the summary line.

When they match it prints the headline deltas, the per-question diff on the gold
tier at the band's headline k (won, lost, moved up, moved down, by question id)
and a one-line summary. Redacted results compare the same way: the diff needs
only ids and ranks.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import recall_common as rc


class Refused(Exception):
    pass


def load(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError) as exc:
        raise Refused(f"{path}: cannot read it as JSON ({exc})")
    if not isinstance(res, dict) or not str(res.get("recall", "")).startswith("recall/") \
            or "manifest" not in res or "per_question" not in res:
        raise Refused(f"{path}: not a Recall 1.0 results file (no recall/manifest/per_question)")
    return res


def pool_identity(res: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """(kind, hash) of the pool a result ran on."""
    c = res["manifest"].get("corpus") or {}
    if c.get("mode") == "live":
        return "live", c.get("pool_sha256")
    return "file", c.get("sha256")


def mismatches(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """What stops two results from being compared (spec section 8)."""
    ma, mb = a["manifest"], b["manifest"]
    out = []
    qa, qb = ma.get("qset", {}), mb.get("qset", {})
    if qa.get("sha256") != qb.get("sha256"):
        out.append(f"question set differs: {qa.get('version')}@{str(qa.get('sha256'))[:8]} vs "
                   f"{qb.get('version')}@{str(qb.get('sha256'))[:8]}")
    (ka, ha), (kb, hb) = pool_identity(a), pool_identity(b)
    for name, res, kind, h in (("a", a, ka, ha), ("b", b, kb, hb)):
        if h is None:
            out.append(f"{name} has no pool fingerprint ({kind} pool; a live run needs list_ids())")
        elif (res["manifest"].get("corpus") or {}).get("pool_changed"):
            out.append(f"{name}'s live pool changed during its run, so it has no single pool to compare")
    if ha is not None and hb is not None and (ka, ha) != (kb, hb):
        out.append(f"pool differs: {ka} {ha[:15]}... vs {kb} {hb[:15]}...")
    pa, pb = ma.get("params", {}), mb.get("params", {})
    if pa.get("track") != pb.get("track"):
        out.append(f"track differs: {pa.get('track')} vs {pb.get('track')}")
    if ma.get("distractors") != mb.get("distractors"):
        out.append("distractors differ")
    va, vb = ma.get("harness_version", {}), mb.get("harness_version", {})
    if va.get("recall") != vb.get("recall"):
        out.append(f"harness version differs: {va.get('recall')} vs {vb.get('recall')}")
    ids_a = [r["id"] for r in a["per_question"]]
    ids_b = [r["id"] for r in b["per_question"]]
    if not out and ids_a != ids_b:
        out.append("per-question records do not line up (a partial run?)")
    return out


def _commit(hv: Dict[str, Any]) -> Optional[str]:
    """A result's harness commit, or None when it records none (an installed copy)."""
    commit = hv.get("commit")
    return None if commit in (None, "unknown") else str(commit)


def unpinned(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """What keeps two results from being pinned to one harness (spec section 8). A
    checkout is pinned by its commit, on a tree without uncommitted changes. An installed
    copy has no commit and is pinned by source_sha256, the hash of the harness's own
    source files. Two checkouts must share the commit; where either result has no commit,
    the two must share the source hash. Anything else refuses a comparison unless
    --allow-unpinned turns it into a note."""
    va, vb = a["manifest"].get("harness_version", {}), b["manifest"].get("harness_version", {})
    ca, cb = _commit(va), _commit(vb)
    sa, sb = va.get("source_sha256"), vb.get("source_sha256")
    out = []
    if ca is not None and cb is not None:
        if ca != cb:
            out.append(f"harness commit differs: {ca[:7]} vs {cb[:7]}")
    elif sa and sb and sa != sb:
        out.append(f"harness source differs: {str(sa)[:15]}... vs {str(sb)[:15]}... (a result with no commit is "
                   "pinned by the hash of the harness's source files)")
    for name, hv, commit in (("a", va, ca), ("b", vb, cb)):
        if hv.get("dirty"):
            out.append(f"{name} ran on a tree with uncommitted changes, so it is not pinned to its commit")
        elif commit is None and not hv.get("source_sha256"):
            out.append(f"{name} records no harness commit and no source hash, so it is not pinned to a harness")
    return out


def _harness_label(hv: Dict[str, Any]) -> str:
    """recall version @ commit, or @ the source hash where there is no commit."""
    commit = _commit(hv)
    pin = commit[:7] if commit else ("source " + str(hv.get("source_sha256") or "unknown").replace("sha256:", "")[:8])
    return f"{hv.get('recall')}@{pin}"


def warnings(a: Dict[str, Any], b: Dict[str, Any], paths: Tuple[str, str]) -> List[str]:
    out = []
    for res, path in zip((a, b), paths):
        status = res.get("verification", {}).get("status")
        if status == "unstable":
            out.append(f"{path} is unstable: its per-question records are its first run, its headline the "
                       "mean of two runs")
        elif status != "yes":
            out.append(f"{path} is not verified ({status})")
    return out


def _rank(r: Optional[int]) -> float:
    return float("inf") if r is None else r


def per_question_diff(a: Dict[str, Any], b: Dict[str, Any], k: int) -> Dict[str, List[Tuple[str, Any, Any]]]:
    """Each answerable question on the gold tier: won (hit at k only in b), lost (only
    in a), moved up or down (same side of k, rank changed), or the same."""
    diff: Dict[str, List[Tuple[str, Any, Any]]] = {"won": [], "lost": [], "moved_up": [], "moved_down": [],
                                                   "same": []}
    for ra, rb_ in zip(a["per_question"], b["per_question"]):
        if ra.get("scored_on") == "abstention":
            continue
        x, y = ra.get("headline_rank"), rb_.get("headline_rank")
        hit_a, hit_b = x is not None and x < k, y is not None and y < k
        if hit_b and not hit_a:
            key = "won"
        elif hit_a and not hit_b:
            key = "lost"
        elif _rank(y) < _rank(x):
            key = "moved_up"
        elif _rank(y) > _rank(x):
            key = "moved_down"
        else:
            key = "same"
        diff[key].append((ra["id"], x, y))
    return diff


def abstention_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    if not (a.get("abstention", {}).get("available") and b.get("abstention", {}).get("available")):
        return None
    out: Dict[str, List[str]] = {"won": [], "lost": []}
    for ra, rb_ in zip(a["per_question"], b["per_question"]):
        if ra.get("scored_on") != "abstention":
            continue
        if rb_.get("abstained") and not ra.get("abstained"):
            out["won"].append(ra["id"])
        elif ra.get("abstained") and not rb_.get("abstained"):
            out["lost"].append(ra["id"])
    return out


def _delta(x: Any, y: Any, places: int) -> str:
    if x is None or y is None:
        return ""
    return f"{y - x:+.{places}f}"


def _cell(x: Any) -> str:
    return "n/a" if x is None else str(x)


def headline_rows(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    ha, hb = a["headline"], b["headline"]
    rows = []
    for kk in ha["k_set"]:
        ks = str(kk)
        rows.append((f"gold@{kk}", ha["recall"].get(ks), hb["recall"].get(ks),
                     _delta(ha["recall"].get(ks), hb["recall"].get(ks), 2)))
    for kk in ha["k_set"]:
        ks = str(kk)
        rows.append((f"skill@{kk}", ha["skill"].get(ks), hb["skill"].get(ks),
                     _delta(ha["skill"].get(ks), hb["skill"].get(ks), 2)))
    rows.append((f"mrr@{ha['window']}", ha.get("mrr"), hb.get("mrr"), _delta(ha.get("mrr"), hb.get("mrr"), 4)))
    va = ha.get("version_pct") if ha.get("version_n") else None
    vb = hb.get("version_pct") if hb.get("version_n") else None
    rows.append(("version", va, vb, _delta(va, vb, 2)))
    aa, ab = a.get("abstention", {}), b.get("abstention", {})
    xa = aa.get("pct") if aa.get("available") else None
    xb = ab.get("pct") if ab.get("available") else None
    rows.append(("abstention", xa, xb, _delta(xa, xb, 2)))
    return rows


def _pos(r: Optional[int]) -> str:
    return "-" if r is None else str(r + 1)


def _label(res: Dict[str, Any]) -> str:
    ad = res["manifest"]["adapter"]
    return f"{ad.get('name')}@{(ad.get('version') or {}).get('version', 'unknown')}"


def summary_line(a: Dict[str, Any], b: Dict[str, Any], diff: Dict[str, List], k: int,
                 unpinned_ok: bool = False) -> str:
    m = a["manifest"]
    kind, h = pool_identity(a)
    ha, hb = a["headline"], b["headline"]
    ks = str(k)
    parts = [f"compare {rc.SPEC_VERSION}", f"qset={m['qset'].get('version')}@{str(m['qset'].get('sha256'))[:8]}",
             f"pool={kind}@{str(h).replace('sha256:', '')[:8]}", f"N={a['pool'].get('N')}", f"band={ha['band']}",
             f"k={k}", f"a={_label(a)}", f"b={_label(b)}",
             f"gold@{k}={_cell(ha['recall'].get(ks))}->{_cell(hb['recall'].get(ks))}"
             f"({_delta(ha['recall'].get(ks), hb['recall'].get(ks), 2) or 'n/a'})",
             f"skill@{k}={_cell(ha['skill'].get(ks))}->{_cell(hb['skill'].get(ks))}",
             f"won={len(diff['won'])}", f"lost={len(diff['lost'])}", f"up={len(diff['moved_up'])}",
             f"down={len(diff['moved_down'])}", f"same={len(diff['same'])}"]
    if unpinned_ok:
        parts.append("unpinned=yes")
    return " ".join(parts)


def compare(path_a: str, path_b: str, allow_unpinned: bool = False) -> Dict[str, Any]:
    """Loads and compares two results; raises Refused when they are not comparable."""
    a, b = load(path_a), load(path_b)
    problems = mismatches(a, b)
    loose = unpinned(a, b)
    if problems or (loose and not allow_unpinned):
        hint = ("\n(--allow-unpinned compares them anyway, for local work, and marks the summary unpinned)"
                if loose and not problems else "")
        raise Refused("these results are not comparable:\n  " + "\n  ".join(problems + loose) + hint)
    k = a["headline"].get("k")
    if k is None:
        raise Refused("no headline k: the pool size is unknown")
    diff = per_question_diff(a, b, k)
    notes = [f"unpinned, compared because of --allow-unpinned: {p}" for p in loose]
    return {"a": a, "b": b, "k": k, "headline": headline_rows(a, b), "diff": diff,
            "abstention": abstention_diff(a, b), "warnings": notes + warnings(a, b, (path_a, path_b)),
            "line": summary_line(a, b, diff, k, unpinned_ok=bool(loose))}


def print_comparison(c: Dict[str, Any], path_a: str, path_b: str) -> None:
    a, b, k = c["a"], c["b"], c["k"]
    m, h = a["manifest"], a["headline"]
    kind, ph = pool_identity(a)
    hv = m.get("harness_version", {})
    print(f"a: {path_a}\n   {_label(a)}, verified={a.get('verification', {}).get('status')}")
    print(f"b: {path_b}\n   {_label(b)}, verified={b.get('verification', {}).get('status')}")
    hb = b["manifest"].get("harness_version", {})
    harness = (f"same harness {_harness_label(hv)}" if _harness_label(hv) == _harness_label(hb) else
               f"harness {_harness_label(hv)} vs {_harness_label(hb)}")
    print(f"Same question set {m['qset'].get('version')}@{str(m['qset'].get('sha256'))[:8]}, same {kind} pool "
          f"{str(ph)[:15]}... (N={a['pool'].get('N')}, band {h['band']}, window {h['window']}), {harness}")
    for w in c["warnings"]:
        print(f"Note: {w}")

    print(f"\nHeadline (gold tier, scored by record id)")
    print(f"  {'':12} {'a':>9} {'b':>9} {'delta':>9}")
    for name, x, y, d in c["headline"]:
        print(f"  {name:12} {_cell(x):>9} {_cell(y):>9} {d:>9}")

    d = c["diff"]
    print(f"\nPer question, gold tier at k={k} (positions count from 1; - is not in the window)")
    for key, title in (("won", "won"), ("lost", "lost"), ("moved_up", "moved up"), ("moved_down", "moved down")):
        rows = sorted(d[key])
        print(f"  {title} ({len(rows)})")
        for qid, x, y in rows:
            print(f"    {qid:16} {_pos(x):>3} -> {_pos(y)}")
    print(f"  unchanged ({len(d['same'])})")
    ab = c["abstention"]
    if ab is not None:
        print(f"  abstention: won {len(ab['won'])} {sorted(ab['won'])}, lost {len(ab['lost'])} {sorted(ab['lost'])}")
    print(f"\n{c['line']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Parsica Recall results: headline deltas and the per-question diff")
    parser.add_argument("a", help="the first results JSON (before)")
    parser.add_argument("b", help="the second results JSON (after)")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="compare results from different harness commits or from a tree with uncommitted "
                             "changes anyway (local work only); prints notes and marks the summary unpinned=yes")
    args = parser.parse_args(argv)
    try:
        c = compare(args.a, args.b, allow_unpinned=args.allow_unpinned)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print_comparison(c, args.a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main())

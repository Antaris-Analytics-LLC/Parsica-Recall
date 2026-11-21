#!/usr/bin/env python3
"""
Re-lint a frozen question set where it sits: the release step that makes a set current
(docs/generation.md section 9).

    parsica-recall relint data/v11 sealed/v11-sealed

For each folder it lints questions.json against corpus.jsonl with the lint spec and the
README, exactly as the generator did when it froze the set, and rewrites lint_report.json
and the manifest's lint block and status. Nothing else in the folder changes: not the
questions, not the corpus, not the reference runs, not the generation record, and no other
field of the manifest. The one gate whose result depends on the harness rather than on the
set is GATE_VERSIONS_MATCH, which compares the set's version with recall_common.CURRENT_QSET;
that is why making a set current re-runs the lint in the same pull request.

Exit status: 0 when every folder was rewritten, 2 when a folder could not be read, or is
marked broken (a set the oracle check failed is not released; fix it first).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import lint_validator as lv
from . import recall_common as rc

REPO_DIR = rc.ROOT_DIR  # the repository's root in a checkout, the package folder in an installed copy
DEFAULT_README = os.path.join(REPO_DIR, "README.md")


class RelintError(Exception):
    """A folder that is not a frozen set, or one that must not be relinted."""


def _dump(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every operating system, as the generator writes
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def lint_block(report: Dict[str, Any], lint_path: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """The manifest's lint block, in the shape generate_questions.py writes it."""
    return {"report": "lint_report.json", "sha256": rc.file_sha256(lint_path),
            "spec_version": spec.get("spec_version"),
            "items_with_errors": report["summary"]["items_with_errors"],
            "gates_passed": report["gates_passed"], "failed_gates": report["failed_gates"],
            "gates": [{"id": g["id"], "passed": g["passed"], "applicable": g["applicable"]} for g in report["gates"]]}


def status_fields(report: Dict[str, Any], store_ids: bool) -> Dict[str, Any]:
    """status, standard_eligible and status_reason, by the generator's rule: a private set
    (store ids) or any failed gate makes a set interim; otherwise it is standard."""
    reasons: List[str] = []
    if store_ids:
        reasons.append("store ids (private set)")
    if not report["gates_passed"]:
        reasons.append("gates failed: " + ", ".join(report["failed_gates"]))
    return {"status": "interim" if reasons else "standard",
            "standard_eligible": not reasons,
            "status_reason": ("interim: " + "; ".join(reasons)) if reasons else "standard: every gate passes"}


def relint(folder: str, spec_path: str = lv.DEFAULT_SPEC, readme_path: Optional[str] = DEFAULT_README) -> Dict[str, Any]:
    """Re-lint the set in folder and rewrite its lint_report.json and manifest.json.
    Returns the set's name, its status and the gates that failed."""
    q_path = os.path.join(folder, "questions.json")
    c_path = os.path.join(folder, "corpus.jsonl")
    m_path = os.path.join(folder, "manifest.json")
    lint_path = os.path.join(folder, "lint_report.json")
    for path in (q_path, c_path, m_path):
        if not os.path.isfile(path):
            raise RelintError(f"{folder} is not a frozen set: {os.path.basename(path)} is missing")
    with open(m_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("status") == "broken":
        raise RelintError(f"{folder} is marked broken ({manifest.get('status_reason')}); a broken set is not released")
    version = manifest.get("qset_version")
    if not version:
        raise RelintError(f"{folder}: manifest.json has no qset_version")
    files = manifest.get("files") or {}
    if files.get("questions.json") != rc.file_sha256(q_path) or files.get("corpus.jsonl") != rc.file_sha256(c_path):
        raise RelintError(f"{folder}: manifest.json does not describe these questions.json and corpus.jsonl")
    store_ids = manifest.get("id_scheme") == "store"
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)
    records = lv.load_jsonl(c_path)
    readme = None
    if readme_path and os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()

    report = lv.lint(questions, records, spec, readme_text=readme, qset_version=version, store_ids=store_ids)
    report["questions_file"] = "questions.json"
    report["questions_sha256"] = files["questions.json"]
    report["corpus_file"] = "corpus.jsonl"
    report["corpus_sha256"] = files["corpus.jsonl"]
    _dump(lint_path, report)

    manifest.update(status_fields(report, store_ids))
    manifest["lint"] = lint_block(report, lint_path, spec)
    _dump(m_path, manifest)
    return {"folder": folder, "qset_version": version, "status": manifest["status"],
            "failed_gates": report["failed_gates"], "items_with_errors": report["summary"]["items_with_errors"]}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Re-lint frozen question sets in place and rewrite their lint report "
                                            "and manifest status (the release step)")
    p.add_argument("folders", nargs="+", help="Question-set folders, such as data/v11 sealed/v11-sealed")
    p.add_argument("--spec", default=lv.DEFAULT_SPEC, help="Lint spec JSON")
    p.add_argument("--readme", default=DEFAULT_README, help="README checked by GATE_VERSIONS_MATCH")
    args = p.parse_args(argv)
    try:
        results = [relint(folder, args.spec, args.readme) for folder in args.folders]
    except (OSError, ValueError, RelintError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for r in results:
        failed = ", ".join(r["failed_gates"]) or "none"
        print(f"{r['folder']}: {r['qset_version']} is {r['status']}; items with errors {r['items_with_errors']}; "
              f"gates failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

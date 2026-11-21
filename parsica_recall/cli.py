"""The parsica-recall command: one entry point for every tool in the harness.

    parsica-recall run --adapter bm25 --verify
    parsica-recall score my_results.jsonl
    parsica-recall compare before.json after.json
    parsica-recall generate-questions --corpus export/corpus.jsonl --out ~/recall-sets/my-store-v1 --seed 7 --estimate

python -m parsica_recall <command> does the same from a checkout with nothing installed.
Each command is one module of this package with its own options; `parsica-recall <command>
--help` prints them. Standard library only.
"""

from __future__ import annotations

import importlib
import sys
from typing import List, Optional

from . import recall_common as rc

# command: (module, its entry function, what it does, whether it runs with a pinned hash seed)
COMMANDS = {
    "run": ("run_benchmark", "main", "run the benchmark on a memory system through an adapter", True),
    "export-task": ("results_file", "main_export",
                    "write the queries and the corpus for a system that has no adapter", False),
    "score": ("results_file", "main_score", "score a results file that any system can write", True),
    "compare": ("compare_results", "main", "compare two results files, or say why they are not comparable", False),
    "generate-questions": ("generate_questions", "main",
                           "write a question set from any corpus.jsonl (the doc2query path)", True),
    "show-questions": ("show_questions", "main", "print a question set for reading", False),
    "lint": ("lint_validator", "main", "lint a question set against its pool and run the gates", False),
    "relint": ("relint_set", "main", "re-lint a frozen set in place (the release step that makes a set current)",
               False),
    "generate-pool": ("generate_pool", "main", "write a synthetic agent-memory pool", True),
    "show-pool": ("show_pool", "main", "print a pool for reading", False),
    "export-planted-pairs": ("export_planted_pairs", "main", "write a pool's planted changes as pairs of record ids",
                             False),
}

PROG = "parsica-recall"


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"usage: {PROG} <command> [options]", "",
             f"Parsica Recall {rc.HARNESS_VERSION} (spec {rc.SPEC_VERSION}): a deterministic retrieval benchmark for "
             "agent memory systems.", "", "commands:"]
    lines += [f"  {name.ljust(width)}  {what}" for name, (_, _, what, _) in COMMANDS.items()]
    lines += ["", f"{PROG} <command> --help prints a command's options. {PROG} --version prints the version."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(usage())
        return 0 if args else 2
    if args[0] in ("-V", "--version", "version"):
        print(f"{PROG} {rc.HARNESS_VERSION} (spec {rc.SPEC_VERSION})")
        return 0
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"{PROG}: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module_name, entry, _, pinned = COMMANDS[command]
    if pinned:
        status = rc.pin_hash_seed(rc.PACKAGE, [command] + rest)
        if status is not None:
            return status
    module = importlib.import_module(f"{rc.PACKAGE}.{module_name}")
    saved = sys.argv[0]
    sys.argv[0] = f"{PROG} {command}"  # so a command's own usage and error lines name it
    try:
        return int(getattr(module, entry)(rest) or 0)
    finally:
        sys.argv[0] = saved


if __name__ == "__main__":
    sys.exit(main())

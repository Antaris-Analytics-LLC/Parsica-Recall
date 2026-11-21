"""The guides' own code is run, so an example a reader copies keeps working. Standard
library only, temporary folders only."""

import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsica_recall import run_benchmark as rb  # noqa: E402
from parsica_recall.adapters import BaseMemoryAdapter, get_adapter, get_custom_adapter  # noqa: E402
from test_recall import make_question, make_record  # noqa: E402

GUIDE = os.path.join(REPO, "docs", "adapting.md")


def python_blocks(path):
    with open(path, encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


class AdaptingGuideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recall-docs-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        texts = [f"record number {i} about topic{i} with filler words here" for i in range(150)]
        texts[7] = "the orchard harvest yielded seven hundred bushels of apples"
        texts[8] = "the orchard was replanted with pear trees after the harvest"
        texts[21] = "the ferry to the island leaves at half past six every morning"
        texts[40] = "the old barn roof was patched with cedar shingles last autumn"
        self.cpath = os.path.join(self.tmp, "corpus.jsonl")
        with open(self.cpath, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(make_record(t)) + "\n")
        questions = [
            make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels", [texts[7]]),
            make_question("q2", "when does the ferry to the island leave", "half past six", [texts[21]]),
            make_question("q3", "what was the barn roof patched with", "cedar shingles", [texts[40]]),
        ]
        self.qpath = os.path.join(self.tmp, "questions.json")
        with open(self.qpath, "w", encoding="utf-8") as f:
            json.dump(questions, f)

    def test_the_worked_example_is_the_reference_bm25(self):
        """docs/adapting.md prints a whole adapter and says it scores what the built-in bm25
        scores. Here the printed code is run, and its records are the built-in adapter's."""
        block = next(b for b in python_blocks(GUIDE) if "class MyBM25(BaseMemoryAdapter)" in b)
        module = types.ModuleType("my_bm25")
        exec(compile(block, "docs/adapting.md", "exec"), module.__dict__)
        self.assertTrue(issubclass(module.MyBM25, BaseMemoryAdapter))
        mine = rb.run_benchmark(module.MyBM25, "my_bm25", self.qpath, self.cpath, verify=True, quiet=True)
        reference = rb.run_benchmark(lambda: get_adapter("bm25"), "bm25", self.qpath, self.cpath, verify=True, quiet=True)
        self.assertEqual(mine["verification"]["status"], "yes")
        self.assertEqual(mine["manifest"]["records_hash"], reference["manifest"]["records_hash"])
        self.assertEqual(mine["headline"]["recall"], reference["headline"]["recall"])

    def test_every_python_block_of_the_guide_compiles(self):
        blocks = python_blocks(GUIDE)
        self.assertGreaterEqual(len(blocks), 3)
        for block in blocks:
            compile(block, "docs/adapting.md", "exec")

    def test_a_custom_adapter_in_the_folder_the_command_runs_from_is_found(self):
        """The guide says: run it from the folder that holds the file."""
        with open(os.path.join(self.tmp, "local_adapter_for_test.py"), "w", encoding="utf-8") as f:
            f.write("from parsica_recall.reference_adapters import BM25Adapter\n\n\nclass Local(BM25Adapter):\n    pass\n")
        here, path = os.getcwd(), list(sys.path)
        self.addCleanup(os.chdir, here)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), path))
        self.addCleanup(sys.modules.pop, "local_adapter_for_test", None)
        sys.path[:] = [p for p in sys.path if p not in ("", self.tmp)]
        os.chdir(self.tmp)
        adapter = get_custom_adapter("local_adapter_for_test.Local")
        self.assertEqual(type(adapter).__name__, "Local")
        self.assertEqual(sys.path[-1], os.getcwd())  # searched last, so it never shadows the standard library


if __name__ == "__main__":
    unittest.main()

"""Tests for the release step: which question set is current, how the versions gate reads a
set's sealed half, and relint_set.py, which re-lints a frozen set in place when a release
makes it current. Standard library only, like the rest of the suite."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsica_recall import lint_validator as lv  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import reference_adapters as ra  # noqa: E402
from parsica_recall import relint_set as rs  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
import test_generate as tg  # noqa: E402  (the generator's fake model and toolbox corpus)
from test_recall import make_question, make_record  # noqa: E402

SET_FILES = ("questions.json", "corpus.jsonl", "manifest.json", "lint_report.json")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CurrentSetTests(unittest.TestCase):
    def test_v11_is_the_current_set_and_its_sealed_half_counts_as_it(self):
        self.assertEqual(rc.CURRENT_QSET, "v11")
        self.assertTrue(rc.is_current_qset("v11"))
        self.assertTrue(rc.is_current_qset("v11" + rc.SEALED_SUFFIX))
        for other in ("v10", "v11-sealed-2", "v11x", "V11", "", None):
            self.assertFalse(rc.is_current_qset(other), other)

    def test_the_runner_and_the_lint_default_to_the_current_set(self):
        self.assertEqual(rb.DEFAULT_QSET_DIR, os.path.join(REPO, "data", "v11"))
        self.assertEqual(lv.DEFAULT_QSET_DIR, os.path.join(REPO, "data", "v11"))
        self.assertTrue(os.path.isfile(os.path.join(rb.DEFAULT_QSET_DIR, "questions.json")))

    def test_the_oracle_reads_the_current_set_when_no_question_file_is_named(self):
        saved = os.environ.pop("RECALL_QFILE", None)
        self.addCleanup(lambda: os.environ.__setitem__("RECALL_QFILE", saved) if saved is not None else None)
        workspace = tempfile.mkdtemp(prefix="recall-oracle-")
        self.addCleanup(shutil.rmtree, workspace, True)
        oracle = ra.OracleAdapter()
        oracle.setup(workspace, "file")
        questions = read_json(os.path.join(REPO, "data", rc.CURRENT_QSET, "questions.json"))
        self.assertEqual(set(oracle.gold), {q["question"] for q in questions})


class VersionsGateTests(unittest.TestCase):
    """GATE_VERSIONS_MATCH takes the current set and its sealed half, and nothing else."""

    README = (f"Current question set: {rc.CURRENT_QSET}. Harness {rc.HARNESS_VERSION}. "
              f"The schema has {len(rc.CATEGORIES)} categories.")

    def setUp(self):
        with open(lv.DEFAULT_SPEC, encoding="utf-8") as f:
            self.spec = json.load(f)
        texts = [f"memory number {i} about subject{i} and nothing else much" for i in range(12)]
        texts[3] = "the backup job runs every night at two in the morning"
        self.records = [make_record(t) for t in texts]
        self.question = make_question("q1", "when does the backup job run", "two in the morning", [texts[3]])

    def gate(self, version, readme=README):
        rep = lv.lint([self.question], self.records, self.spec, readme_text=readme, qset_version=version)
        return next(g for g in rep["gates"] if g["id"] == "GATE_VERSIONS_MATCH")

    def test_the_current_set_passes(self):
        self.assertTrue(self.gate(rc.CURRENT_QSET)["passed"])

    def test_the_sealed_half_of_the_current_set_passes(self):
        self.assertTrue(self.gate(rc.CURRENT_QSET + rc.SEALED_SUFFIX)["passed"])

    def test_another_set_fails_and_the_message_names_both_forms(self):
        gate = self.gate("v10")
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["detail"]["problems"],
                         [f"question set is v10, harness expects {rc.CURRENT_QSET} "
                          f"(or its sealed half, {rc.CURRENT_QSET}{rc.SEALED_SUFFIX})"])
        self.assertFalse(self.gate(rc.CURRENT_QSET + rc.SEALED_SUFFIX + "-2")["passed"])

    def test_a_readme_that_does_not_name_the_current_set_fails(self):
        gate = self.gate(rc.CURRENT_QSET, readme=f"Harness {rc.HARNESS_VERSION}. {len(rc.CATEGORIES)} categories.")
        self.assertFalse(gate["passed"])
        self.assertIn("README does not state the question-set version", gate["detail"]["problems"][0])

    def test_the_repository_readme_states_the_current_set(self):
        with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        self.assertTrue(self.gate(rc.CURRENT_QSET, readme=readme)["passed"])


class RelintShippedTests(unittest.TestCase):
    """The committed v11 halves are exactly what relint_set.py writes: the release re-ran the
    lint, rewrote each half's lint report and the lint block and status of its manifest, and
    changed nothing else. The byte-for-byte check lints 727 questions against 10,000 records
    twice, about 14 minutes, so it runs only with RECALL_SLOW_TESTS=1; the other checks read
    the committed files and run every time."""

    PUBLIC, SEALED = os.path.join("data", "v11"), os.path.join("sealed", "v11-sealed")

    def setUp(self):
        if not os.path.isfile(os.path.join(REPO, self.PUBLIC, "questions.json")):
            self.skipTest(f"{self.PUBLIC} not built")
        # The sealed half is never published: a tree without it checks the public half alone.
        self.sealed_held = os.path.isfile(os.path.join(REPO, self.SEALED, "questions.json"))
        self.HALVES = (self.PUBLIC, self.SEALED) if self.sealed_held else (self.PUBLIC,)
        self.tmp = tempfile.mkdtemp(prefix="recall-relint-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def copy_half(self, half):
        src = os.path.join(REPO, half)
        dst = os.path.join(self.tmp, os.path.basename(half))
        os.makedirs(dst)
        for name in SET_FILES:
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
        return src, dst

    @unittest.skipUnless(os.environ.get("RECALL_SLOW_TESTS"),
                         "set RECALL_SLOW_TESTS=1 to re-lint both v11 halves (about 7 minutes each)")
    def test_relint_reproduces_both_halves_byte_for_byte(self):
        for half in self.HALVES:
            src, dst = self.copy_half(half)
            before = {name: read_bytes(os.path.join(dst, name)) for name in SET_FILES}
            out = rs.relint(dst)
            self.assertEqual(out["status"], "interim", half)
            self.assertEqual(out["failed_gates"], ["GATE_CATEGORY_MINIMUMS"], half)
            self.assertEqual(out["items_with_errors"], 0, half)
            for name in SET_FILES:
                self.assertEqual(read_bytes(os.path.join(dst, name)), read_bytes(os.path.join(src, name)), (half, name))
                self.assertEqual(read_bytes(os.path.join(dst, name)), before[name], (half, name))

    def test_both_manifests_say_the_versions_gate_passes_and_why_the_set_is_interim(self):
        for half in self.HALVES:
            m = read_json(os.path.join(REPO, half, "manifest.json"))
            gates = {g["id"]: g["passed"] for g in m["lint"]["gates"]}
            self.assertTrue(gates["GATE_VERSIONS_MATCH"], half)
            self.assertFalse(gates["GATE_CATEGORY_MINIMUMS"], half)
            self.assertEqual(m["lint"]["failed_gates"], ["GATE_CATEGORY_MINIMUMS"], half)
            self.assertEqual((m["status"], m["standard_eligible"]), ("interim", False), half)
            self.assertEqual(m["status_reason"], "interim: gates failed: GATE_CATEGORY_MINIMUMS", half)
            self.assertEqual(m["lint"]["sha256"], rc.file_sha256(os.path.join(REPO, half, "lint_report.json")), half)
            report = read_json(os.path.join(REPO, half, "lint_report.json"))
            self.assertEqual(report["failed_gates"], ["GATE_CATEGORY_MINIMUMS"], half)
            self.assertEqual(report["questions_sha256"], m["files"]["questions.json"], half)
            info = rb.load_qset_manifest(os.path.join(REPO, half, "questions.json"),
                                         os.path.join(REPO, half, "corpus.jsonl"))
            self.assertTrue(info["matches"], half)

    def test_the_public_half_commits_to_the_sealed_half_by_hash_and_nothing_else(self):
        public = read_json(os.path.join(REPO, self.PUBLIC, "manifest.json"))
        self.assertEqual(public["qset_version"], rc.CURRENT_QSET)
        self.assertEqual(public["split"]["half"], "public")
        self.assertEqual(set(public["split"]["sealed"]), {"questions_sha256", "note"})
        self.assertRegex(public["split"]["sealed"]["questions_sha256"], r"^[0-9a-f]{64}$")
        if self.sealed_held:
            self.assertEqual(public["split"]["sealed"]["questions_sha256"],
                             rc.file_sha256(os.path.join(REPO, self.SEALED, "questions.json")))

    def test_the_sealed_half_is_named_after_the_public_one(self):
        if not self.sealed_held:
            self.skipTest("the sealed half is not in this tree")
        public = read_json(os.path.join(REPO, self.PUBLIC, "manifest.json"))
        sealed = read_json(os.path.join(REPO, self.SEALED, "manifest.json"))
        self.assertEqual(public["qset_version"], rc.CURRENT_QSET)
        self.assertEqual(sealed["qset_version"], rc.CURRENT_QSET + rc.SEALED_SUFFIX)
        self.assertEqual(sealed["split"]["public"]["qset_version"], public["qset_version"])


class RelintFreshSetTests(tg.GenTestCase):
    """relint_set.py on a set the generator just froze writes the generator's own bytes, and
    refuses what it must not touch."""

    def fresh(self):
        res = self.gen(tg.toolboxes(24), "direct_recall_user=2,vocabulary_gap=2,abstention=1")
        self.assertTrue(res["check"]["ok"])
        return res["out"]

    def test_relint_matches_what_the_generator_wrote(self):
        out = self.fresh()
        before = {name: read_bytes(os.path.join(out, name)) for name in SET_FILES}
        result = rs.relint(out)
        after = {name: read_bytes(os.path.join(out, name)) for name in SET_FILES}
        self.assertEqual(after, before)
        self.assertEqual(result["qset_version"], "set-a")
        self.assertEqual(result["status"], "interim")
        self.assertIn("GATE_VERSIONS_MATCH", result["failed_gates"])  # a set that is not current keeps failing it

    def test_relint_rewrites_the_lint_block_and_status_from_a_fresh_lint(self):
        out = self.fresh()
        m_path = os.path.join(out, "manifest.json")
        m = read_json(m_path)
        m["status"], m["standard_eligible"], m["status_reason"] = "standard", True, "stale"
        m["lint"] = {"report": "lint_report.json", "sha256": "stale", "gates": []}
        rs._dump(m_path, m)
        os.remove(os.path.join(out, "lint_report.json"))
        rs.relint(out)
        fresh = read_json(m_path)
        self.assertEqual((fresh["status"], fresh["standard_eligible"]), ("interim", False))
        self.assertTrue(fresh["status_reason"].startswith("interim: gates failed: "))
        self.assertEqual(fresh["lint"]["sha256"], rc.file_sha256(os.path.join(out, "lint_report.json")))
        self.assertEqual(list(fresh), list(m))  # every other field stays where it was

    def test_relint_refuses_a_broken_set_and_files_the_manifest_does_not_describe(self):
        out = self.fresh()
        m_path = os.path.join(out, "manifest.json")
        m = read_json(m_path)
        broken = dict(m, status="broken", status_reason="broken: the oracle did not score 100 on the gold tier")
        rs._dump(m_path, broken)
        with self.assertRaises(rs.RelintError):
            rs.relint(out)
        rs._dump(m_path, m)
        q_path = os.path.join(out, "questions.json")
        questions = read_json(q_path)
        questions[0]["question"] = questions[0]["question"] + " (edited)"
        rs._dump(q_path, questions)
        with self.assertRaises(rs.RelintError):
            rs.relint(out)
        with self.assertRaises(rs.RelintError):
            rs.relint(self.tmp)  # not a set at all

    def test_cli(self):
        out = self.fresh()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(rs.main([out]), 0)
            self.assertEqual(rs.main([os.path.join(self.tmp, "nowhere")]), 2)
        self.assertIn("set-a is interim", stdout.getvalue())
        self.assertIn("GATE_VERSIONS_MATCH", stdout.getvalue())
        self.assertIn("ERROR:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Tests for the no-code path (parsica_recall/results_file.py): the task folder a system
with no adapter is handed, and the scoring of the results file it writes back. Standard
library only, temporary folders only."""

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

from parsica_recall import cli  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import results_file as rf  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
from parsica_recall.adapters import get_adapter  # noqa: E402
from test_recall import make_question, make_record  # noqa: E402


class ResultsFileCase(unittest.TestCase):
    """A pool of 120 records (band S, window 10), three answerable questions and one
    abstention question, in a temporary folder."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recall-results-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        texts = [f"record number {i} about topic{i} with filler words here" for i in range(120)]
        texts[7] = "the orchard harvest yielded seven hundred bushels of apples"
        texts[21] = "the ferry to the island leaves at half past six every morning"
        texts[40] = "the old barn roof was patched with cedar shingles last autumn"
        self.texts = texts
        self.records = [make_record(t) for t in texts]
        self.cpath = os.path.join(self.tmp, "corpus.jsonl")
        with open(self.cpath, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
        self.questions = [
            make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels", [texts[7]]),
            make_question("q2", "when does the ferry to the island leave", "half past six", [texts[21]]),
            make_question("q3", "what was the barn roof patched with", "cedar shingles", [texts[40]]),
            {**make_question("q4", "what colour was the lighthouse door", "not mentioned", [], category="abstention",
                             answer_type="abstention"), "gold_ids": [], "gold_hashes": {}},
        ]
        self.qpath = os.path.join(self.tmp, "questions.json")
        with open(self.qpath, "w", encoding="utf-8") as f:
            json.dump(self.questions, f)
        self.rpath = os.path.join(self.tmp, "results.jsonl")
        self.gold = {q["id"]: (q["gold_ids"] or [None])[0] for q in self.questions}

    def write(self, lines):
        with open(self.rpath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
        return self.rpath

    def all_lines(self, **over):
        """One valid line per question: the gold record first, where there is one."""
        lines = []
        for q in self.questions:
            hits = [self.gold[q["id"]]] if self.gold[q["id"]] else []
            lines.append({"qid": q["id"], "hits": over.get(q["id"], hits)})
        return lines

    def read(self, lines):
        return rf.read_results(self.write(lines), self.questions)

    def score(self, lines, **kw):
        results = self.read(lines)
        window = rc.band_for(len(self.records))["window"]
        notes = rf.describe(results, {r["id"] for r in self.records}, window)
        return rb.run_benchmark(lambda: rf.ResultsFileAdapter(results, self.questions), rf.system_name(results),
                                self.qpath, self.cpath, quiet=True, source=rb.SOURCE_RESULTS_FILE, notes=notes, **kw)

    def refused(self, lines):
        with self.assertRaises(rb.InputError) as caught:
            self.read(lines)
        return str(caught.exception)


class ScoreTests(ResultsFileCase):
    def test_a_file_made_from_an_adapter_run_scores_to_the_same_records(self):
        ran = rb.run_benchmark(lambda: get_adapter("bm25"), "bm25", self.qpath, self.cpath, verify=True, quiet=True)
        self.assertEqual(ran["verification"]["status"], "yes")
        lines = [{"recall_results": 1, "system": {"name": "bm25 copy", "version": "1.0", "engine_class": "lexical",
                                                   "models": [], "network": False}}]
        lines += [{"qid": q["id"], "hits": [{"id": e["id"], "score": e["score"]} for e in q["window"]]}
                  for q in ran["per_question"]]
        res = self.score(lines)
        # the same per-question records, byte for byte, so every number that follows from them is the same
        self.assertEqual(res["verification"]["records_hash"], [ran["verification"]["records_hash"][0]])
        self.assertEqual(res["headline"]["recall"], ran["headline"]["recall"])
        self.assertEqual(res["headline"]["mrr"], ran["headline"]["mrr"])
        # and the labels say who ran the system
        self.assertEqual(res["verification"]["status"], "no")
        self.assertIn("results file", res["verification"]["detail"])
        self.assertFalse(res["standard"]["eligible"])
        self.assertIn("scored from a results file (the system ran outside the harness)", res["standard"]["reasons"])
        adapter = res["manifest"]["adapter"]
        self.assertEqual((adapter["name"], adapter["contract"]), ("file:bm25-copy", rb.CONTRACT_RESULTS_FILE))
        self.assertEqual(adapter["version"]["results_sha256"], rc.file_sha256(self.rpath))
        self.assertEqual((adapter["version"]["version"], adapter["version"]["engine_class"]), ("1.0", "lexical"))
        self.assertIn("adapter=file:bm25-copy@1.0 verified=no standard=no", res["result_line"])

    def test_plain_ids_object_hits_and_text_hits_all_score(self):
        res = self.score(self.all_lines(q2=[{"id": self.gold["q2"]}], q3=[{"text": self.texts[40]}]))
        self.assertEqual([q["gold_rank"] for q in res["per_question"]][:3], [0, 0, 0])
        self.assertEqual(res["headline"]["recall"]["1"], 100.0)
        self.assertTrue(any("came as text" in n for n in res["notes"]), res["notes"])

    def test_without_a_header_the_system_is_unknown(self):
        res = self.score(self.all_lines())
        version = res["manifest"]["adapter"]["version"]
        self.assertEqual(res["manifest"]["adapter"]["name"], "file:unnamed")
        self.assertEqual((version["version"], version["engine_class"], version["network"]), ("unknown", "unknown", None))

    def test_a_miss_is_a_miss_and_an_unknown_id_is_noted(self):
        other = self.records[3]["id"]
        res = self.score(self.all_lines(q1=[other], q2=["feedfacefeedface"]))
        self.assertEqual([q["gold_rank"] for q in res["per_question"]][:3], [None, None, 0])
        self.assertTrue(any("1 returned ids, in 1 questions, are not record ids" in n for n in res["notes"]), res["notes"])

    def test_hits_past_the_window_are_not_read(self):
        window = rc.band_for(len(self.records))["window"]
        filler = [r["id"] for r in self.records[50:50 + window]]
        res = self.score(self.all_lines(q1=filler + [self.gold["q1"]]))
        q1 = res["per_question"][0]
        self.assertIsNone(q1["gold_rank"])  # the gold record sits one past the window
        self.assertEqual((len(q1["window"]), q1["returned_count"]), (window, window + 1))
        self.assertTrue(any("more than the window" in n for n in res["notes"]), res["notes"])

    def test_a_record_returned_twice_burns_its_second_slot(self):
        g = self.gold["q1"]
        res = self.score(self.all_lines(q1=[g, g, self.records[3]["id"]]))
        q1 = res["per_question"][0]
        self.assertEqual((q1["burned_duplicates"], [e["rank"] for e in q1["window"]]), (1, [0, 2]))

    def test_no_search_time_is_reported_unless_the_file_reports_one(self):
        res = self.score(self.all_lines())
        self.assertIsNone(res["latency"]["search_ms_p50"])
        self.assertEqual((res["latency"]["measured_by"], res["latency"]["questions_timed"]), (None, 0))
        self.assertIsNone(res["latency"]["ingest_records_per_s"])
        lines = self.all_lines()
        for line, ms in zip(lines, (10, 20.5, 30, 40)):
            line["ms"] = ms
        res = self.score(lines)
        self.assertEqual((res["latency"]["measured_by"], res["latency"]["questions_timed"]), ("the results file", 4))
        self.assertEqual((res["latency"]["search_ms_p50"], res["latency"]["search_ms_max"]), (20.5, 40.0))
        self.assertEqual([q["search_time_ms"] for q in res["per_question"]], [10.0, 20.5, 30.0, 40.0])

    def test_reported_times_never_change_the_records_hash(self):
        plain = self.score(self.all_lines())
        timed = self.all_lines()
        timed[0]["ms"] = 99
        self.assertEqual(self.score(timed)["manifest"]["records_hash"], plain["manifest"]["records_hash"])

    def test_abstention_is_scored_from_scores_and_a_declared_threshold(self):
        header = {"recall_results": 1, "system": {"name": "s", "abstain_threshold": 0.5}}
        scored = [{"qid": q["id"], "hits": [{"id": self.gold[q["id"]], "score": 0.9}] if self.gold[q["id"]]
                   else [{"id": self.records[3]["id"], "score": 0.2}]} for q in self.questions]
        res = self.score([header] + scored)
        self.assertEqual((res["abstention"]["available"], res["abstention"]["correct"]), (True, 1))
        res = self.score(scored)  # no threshold declared
        self.assertFalse(res["abstention"]["available"])

    def test_a_results_file_is_one_run_so_verify_and_the_passes_are_refused(self):
        for kw in ({"verify": True}, {"passes": ["query"]}, {"live": True}):
            with self.assertRaises(rb.InputError, msg=kw):
                self.score(self.all_lines(), **kw)

    def test_the_report_names_the_file_and_prints_no_harness_timing(self):
        results = self.read(self.all_lines())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rb.run_benchmark(lambda: rf.ResultsFileAdapter(results, self.questions), rf.system_name(results),
                             self.qpath, self.cpath, source=rb.SOURCE_RESULTS_FILE)
        text = out.getvalue()
        self.assertIn("Latency: not measured (scored from a results file", text)
        self.assertIn("Verification: not verified (scored from a results file", text)
        self.assertNotIn("records/s", text)

    def test_store_ids_and_redaction_work_as_for_an_adapter(self):
        res = self.score(self.all_lines(), redact=True)
        self.assertTrue(all(e["text"].startswith("sha256:") for q in res["per_question"] for e in q["window"]))
        self.assertTrue(all(q["question"] is None for q in res["per_question"]))

    def test_the_system_name_cannot_break_the_result_line(self):
        self.assertEqual(rf.system_name({"system": {"name": "My Store = v2 (beta)"}}), "file:My-Store-v2-beta")
        self.assertEqual(rf.system_name({"system": {"name": "  "}}), "file:unnamed")
        self.assertEqual(rf.system_name({"system": {}}), "file:unnamed")


class RefusalTests(ResultsFileCase):
    def test_every_question_needs_a_line(self):
        message = self.refused(self.all_lines()[:-1])
        self.assertIn("no line for 1 of the set's 4 questions (q4)", message)
        self.assertIn("refusing the results file", message)

    def test_an_empty_list_is_an_answer(self):
        results = self.read(self.all_lines(q1=[]))
        self.assertEqual(results["hits"]["q1"], [])

    def test_bad_lines_are_named_by_line(self):
        ok = self.all_lines()
        cases = [
            ([*ok[:1], "{not json", *ok[1:]], "line 2: not a JSON line"),
            ([*ok, {"qid": "q9", "hits": []}], "line 5: qid 'q9' is not a question of this set"),
            ([*ok, ok[0]], "line 5: qid q1 has a line already"),
            ([{"qid": "q1", "hits": "abc"}, *ok[1:]], 'line 1: "hits" is a list'),
            ([{"qid": "q1", "hits": [{"score": 1}]}, *ok[1:]], 'line 1, hit 1: a hit needs "id"'),
            ([{"qid": "q1", "hits": [{"id": "a", "score": "high"}]}, *ok[1:]], 'line 1, hit 1: "score" is a number'),
            ([{"qid": "q1", "hits": [{"id": "a", "rank": 1}]}, *ok[1:]], "line 1, hit 1: unknown key in a hit: rank"),
            ([{"qid": "q1", "hits": [7]}, *ok[1:]], "line 1, hit 1: a hit is a record id"),
            ([{"qid": "q1", "hits": [""]}, *ok[1:]], "line 1, hit 1: an empty string is not a record id"),
            ([{"qid": "q1", "hits": [], "hit": []}, *ok[1:]], "line 1: unknown key: hit"),
            ([{"qid": "q1", "hits": [], "ms": -1}, *ok[1:]], 'line 1: "ms" is the search time'),
            (['{"qid": "q1", "hits": [{"id": "a", "score": NaN}]}', *ok[1:]], "line 1: not a JSON line"),
            (["[1, 2]", *ok], "line 1: each line is one JSON object"),
        ]
        for lines, expected in cases:
            self.assertIn(expected, self.refused(lines), expected)

    def test_the_header_is_checked_and_comes_first(self):
        ok = self.all_lines()
        cases = [
            ([{"recall_results": 2}, *ok], '"recall_results" is 2; this harness reads format 1'),
            ([{"system": {"name": "s", "engine": "hybrid"}}, *ok], 'unknown key in "system": engine'),
            ([{"system": {"engine_class": "magic"}}, *ok], '"engine_class" is one of lexical, embedding, hybrid, llm_assisted'),
            ([{"system": {"models": "one"}}, *ok], '"models" is a list'),
            ([{"system": {"network": "no"}}, *ok], '"network" is true or false'),
            ([{"system": {"abstain_threshold": "low"}}, *ok], '"abstain_threshold" is a number'),
            ([{"system": ["s"]}, *ok], '"system" is an object'),
            ([{"recall_results": 1, "note": "x"}, *ok], "unknown key in the header: note"),
            ([*ok, {"system": {"name": "late"}}], 'line 5: no "qid"; only the first line may be the header'),
        ]
        for lines, expected in cases:
            self.assertIn(expected, self.refused(lines), expected)

    def test_a_missing_file_is_an_input_error(self):
        with self.assertRaises(rb.InputError):
            rf.read_results(os.path.join(self.tmp, "nowhere.jsonl"), self.questions)


class ExportTaskTests(ResultsFileCase):
    def test_the_task_folder_holds_the_queries_the_corpus_and_the_window(self):
        out = os.path.join(self.tmp, "task")
        task = rf.export_task(self.qpath, self.cpath, out)
        self.assertEqual(sorted(os.listdir(out)), sorted(rf.TASK_FILES))
        with open(os.path.join(out, "queries.jsonl"), encoding="utf-8") as f:
            queries = [json.loads(line) for line in f]
        # the question id and the question text, and nothing else of the question file
        self.assertEqual(queries, [{"qid": q["id"], "query": q["question"]} for q in self.questions])
        with open(os.path.join(out, "queries.jsonl"), encoding="utf-8") as f:
            raw = f.read()
        for q in self.questions:
            for secret in [q["answer"], q["category"], *q["gold_ids"]]:
                self.assertNotIn(secret, raw)
        with open(self.cpath, "rb") as a, open(os.path.join(out, "corpus.jsonl"), "rb") as b:
            self.assertEqual(a.read(), b.read())
        with open(os.path.join(out, "task.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), task)
        self.assertEqual((task["band"], task["window"], task["corpus"]["records"]), ("S", 10, 120))
        self.assertEqual(task["corpus"]["sha256"], rc.file_sha256(self.cpath))
        self.assertEqual(task["queries"]["sha256"], rc.file_sha256(os.path.join(out, "queries.jsonl")))
        self.assertEqual((task["qset"]["questions"], task["corpus"]["id_scheme"]), (4, "content"))

    def test_a_task_folder_is_never_overwritten_and_never_inside_the_harness(self):
        out = os.path.join(self.tmp, "task")
        rf.export_task(self.qpath, self.cpath, out)
        with self.assertRaises(rb.InputError):
            rf.export_task(self.qpath, self.cpath, out)
        inside = os.path.join(rc.ROOT_DIR, "task-should-not-exist")
        with self.assertRaises(rb.InputError):
            rf.export_task(self.qpath, self.cpath, inside)
        self.assertFalse(os.path.exists(inside))

    def test_a_question_set_that_does_not_match_the_corpus_is_refused(self):
        other = os.path.join(self.tmp, "other.jsonl")
        with open(other, "w", encoding="utf-8") as f:
            for rec in self.records[50:]:
                f.write(json.dumps(rec) + "\n")
        with self.assertRaises(rb.InputError):
            rf.export_task(self.qpath, other, os.path.join(self.tmp, "task"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "task")))


class CommandTests(ResultsFileCase):
    """parsica-recall export-task and parsica-recall score, through the command's own door."""

    def setUp(self):
        super().setUp()
        saved = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "0"  # as the command pins it; here it keeps the command in this process
        self.addCleanup(lambda: os.environ.pop("PYTHONHASHSEED", None) if saved is None
                        else os.environ.__setitem__("PYTHONHASHSEED", saved))

    def command(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_export_task_then_score(self):
        task = os.path.join(self.tmp, "task")
        code, out, _ = self.command("export-task", "--out", task, "--questions", self.qpath, "--corpus", self.cpath)
        self.assertEqual(code, 0)
        self.assertIn("return your system's top 10 records", out)
        self.write(self.all_lines())
        output = os.path.join(self.tmp, "scored.json")
        code, out, _ = self.command("score", self.rpath, "--questions", self.qpath, "--corpus", self.cpath,
                                    "--output", output)
        self.assertEqual(code, 0)
        self.assertIn("adapter=file:unnamed@unknown verified=no standard=no", out)
        with open(output, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["headline"]["recall"]["1"], 100.0)

    def test_a_refused_file_exits_2_with_its_problems(self):
        self.write(self.all_lines()[:-1])
        code, _, err = self.command("score", self.rpath, "--questions", self.qpath, "--corpus", self.cpath,
                                    "--output", os.path.join(self.tmp, "scored.json"))
        self.assertEqual(code, 2)
        self.assertIn("refusing the results file", err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "scored.json")))

    def test_the_command_lists_itself_and_refuses_what_it_does_not_know(self):
        code, out, _ = self.command("--help")
        self.assertEqual(code, 0)
        for name in cli.COMMANDS:
            self.assertIn(f"  {name} ", out)
        self.assertEqual(self.command()[0], 2)
        code, _, err = self.command("no-such-command")
        self.assertEqual(code, 2)
        self.assertIn("unknown command 'no-such-command'", err)
        code, out, _ = self.command("--version")
        self.assertEqual((code, out.strip()), (0, f"parsica-recall {rc.HARNESS_VERSION} (spec {rc.SPEC_VERSION})"))


if __name__ == "__main__":
    unittest.main()

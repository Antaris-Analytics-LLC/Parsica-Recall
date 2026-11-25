"""Tests for the question builder's --common-answers switch (docs/generation.md section 6): a
record of the gold record's conversation that holds the answer becomes gold too, and under
"owner" an item whose answer is otherwise only in other owners' records is kept, carries
common_answer, passes the lint with a note and is left out of the runner's exact tier. A faked
model; no network and no SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_common_answers
"""

import json
import os
import sys
import tempfile
import shutil
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from parsica_recall import generate_questions as gq  # noqa: E402
from parsica_recall import lint_validator as lv  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
from parsica_recall.adapters import get_adapter  # noqa: E402
import test_generate as tg  # noqa: E402
import test_recall as tr  # noqa: E402
import test_voices as tv  # noqa: E402

V7 = os.path.join(REPO, "prompts", "v7")
PEOPLE = {"p1": "Maren", "p2": "Odile"}
YES = lambda rid: True  # noqa: E731  the writer says the record gives the answer
NO = lambda rid: False  # noqa: E731


def said_and_echo(i, owner="p1"):
    """An owner's turn about toolbox i and, in the same conversation, the assistant saying it back."""
    said = tg.toolbox(i, source=f"{owner}-s{i}")
    num, cab = tg.TOOLBOX_RE.search(said["text"]).groups()
    echo = tg.rec(f"Noted for the weekend list: the box numbered {num} belongs in cabinet {cab}, as you said.",
                  source=f"{owner}-s{i}")
    return said, echo


def elsewhere(i, owner, session):
    """Another conversation's record that happens to hold toolbox i's answer."""
    cab = tg.TOOLBOX_RE.search(tg.toolbox(i)["text"]).group(2)
    return tg.rec(f"The lock on cabinet {cab} was changed in March after the office move, as file {session} says.",
                  source=f"{owner}-s{session}")


def errors(res):
    return [f["code"] for f in res["findings"] if f["severity"] == "error"]


class GoldOrCommonTests(unittest.TestCase):
    def build(self, extra=(), source=True):
        said, echo = said_and_echo(1)
        if not source:
            said, echo = ({k: v for k, v in r.items() if k != "source"} for r in (said, echo))
        self.said, self.echo = said["id"], echo["id"]
        self.corpus = gq.Corpus([said, echo, *extra, *tg.toolboxes(8, start=20)], tg.spec(), False)
        num, cab = tg.TOOLBOX_RE.search(said["text"]).groups()
        it = tg.item(f"Where do I keep toolbox number {num}?", f"cabinet {cab}", [said["id"]])
        return gq.assemble("direct_recall_user", it, (said["id"],), self.corpus, tg.spec(), "cand_00001")

    def voices(self):
        return gq.Voices(7, PEOPLE, owner=True)

    def test_the_lint_holds_the_item_as_written(self):
        q = self.build()
        res = lv.lint_item(q, 1, self.corpus.ctx)
        self.assertFalse(res["publishable"])
        self.assertEqual(errors(res), ["answer_in_other_record"])
        self.assertTrue(gq.retryable(res["findings"]))

    def test_a_record_of_the_gold_conversation_that_holds_the_answer_becomes_gold(self):
        q = self.build()
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N,
                                     confirm=YES)
        self.assertEqual(q2["gold_ids"], [self.said, self.echo])
        self.assertEqual(list(q2["gold_hashes"]), [self.said, self.echo])
        self.assertEqual(q2["gold_hashes"][self.echo], rc.text_hash(self.corpus.by_id[self.echo]["text"]))
        self.assertNotIn(rc.COMMON_ANSWER, q2)
        self.assertEqual(note, {"gold_added": [self.echo], "checked": [{"id": self.echo, "answers": True}],
                                "elsewhere": 0, "outcome": "gold"})
        self.assertEqual(q["gold_ids"], [self.said])  # the item as written is not touched
        self.assertTrue(lv.lint_item(q2, 1, self.corpus.ctx)["publishable"])

    def test_another_owners_record_makes_it_a_common_answer_under_owner_only(self):
        q = self.build([elsewhere(1, "p2", 900)])
        checked = [{"id": self.echo, "answers": True}]
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N,
                                     confirm=YES)
        self.assertIsNone(q2)
        self.assertEqual(note, {"gold_added": [self.echo], "checked": checked, "elsewhere": 1, "outcome": "held"})
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, self.voices(), "owner", self.corpus.N,
                                     confirm=YES)
        self.assertEqual(q2["gold_ids"], [self.said, self.echo])
        self.assertIs(q2[rc.COMMON_ANSWER], True)
        self.assertEqual(note, {"gold_added": [self.echo], "checked": checked, "elsewhere": 1,
                                "outcome": rc.COMMON_ANSWER, "same_owner": 0})
        res = lv.lint_item(q2, 1, self.corpus.ctx)
        self.assertTrue(res["publishable"])
        noted = [f for f in res["findings"] if f["code"] == "common_answer"]
        self.assertEqual([(f["rule_id"], f["severity"]) for f in noted], [("GOLD_MULTIPLICITY", "info")])
        unmarked = {k: v for k, v in q2.items() if k != rc.COMMON_ANSWER}
        self.assertEqual(errors(lv.lint_item(unmarked, 1, self.corpus.ctx)), ["answer_in_other_record"])

    def test_the_same_owners_other_conversation_keeps_it_held(self):
        q = self.build([elsewhere(1, "p1", 700), elsewhere(1, "p2", 900)])
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, self.voices(), "owner", self.corpus.N,
                                     confirm=YES)
        self.assertIsNone(q2)
        self.assertEqual((note["elsewhere"], note["same_owner"], note["outcome"]), (2, 1, "held"))

    def test_the_categories_with_several_gold_records_keep_their_gold(self):
        q = self.build()
        for cat in rc.MULTI_RECORD_CATEGORIES:
            q2, note = gq.gold_or_common(cat, q, self.corpus, self.voices(), "owner", self.corpus.N)
            self.assertIsNone(q2, cat)  # the echo is the same owner's, and it is not made gold
            self.assertEqual((note["gold_added"], note["same_owner"]), ([], 1), cat)

    def test_a_record_with_no_source_has_no_conversation(self):
        q = self.build(source=False)
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N)
        self.assertIsNone(q2)
        self.assertEqual(note, {"gold_added": [], "checked": [], "elsewhere": 1, "outcome": "held"})

    def test_a_no_from_the_check_leaves_the_record_out_and_the_item_held(self):
        # A record that only shares the answer's words: the check says no, and it stays a leak.
        q = self.build()
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N,
                                     confirm=NO)
        self.assertIsNone(q2)
        self.assertEqual(note, {"gold_added": [], "checked": [{"id": self.echo, "answers": False}],
                                "elsewhere": 1, "outcome": "held"})

    def test_nothing_is_promoted_without_a_check(self):
        q = self.build()
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N)
        self.assertIsNone(q2)
        self.assertEqual(note["checked"], [{"id": self.echo, "answers": None}])

    def test_only_a_record_that_could_have_been_shown_is_put_to_the_check(self):
        q = self.build()
        asked = []
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation", self.corpus.N,
                                     confirm=lambda rid: asked.append(rid) or True, eligible={self.said})
        self.assertIsNone(q2)
        self.assertEqual(asked, [])
        self.assertEqual(note["checked"], [{"id": self.echo, "answers": None, "why": "not eligible to be shown"}])

    def test_another_owners_near_copy_keeps_the_echo_out(self):
        # The echo's word-for-word copy sits in another person's records: the cluster is not the owner's.
        said, echo = said_and_echo(1)
        copy = dict(echo, id="copy-of-echo", source="p2-s900")
        corpus = gq.Corpus([said, echo, copy, *tg.toolboxes(8, start=20)], tg.spec(), True)
        num, cab = tg.TOOLBOX_RE.search(said["text"]).groups()
        it = tg.item(f"Where do I keep toolbox number {num}?", f"cabinet {cab}", [said["id"]])
        q = gq.assemble("direct_recall_user", it, (said["id"],), corpus, tg.spec(), "cand_00001")
        q2, note = gq.gold_or_common("direct_recall_user", q, corpus, self.voices(), "owner", corpus.N,
                                     confirm=YES)
        self.assertNotIn(echo["id"], (q2 or {}).get("gold_ids", []))
        self.assertEqual(note["checked"][0]["why"], "not the owner's record")

    def test_an_owner_who_is_not_known_keeps_it_held(self):
        q = self.build([elsewhere(1, "p3", 900)])  # p3 is in no people file key
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, self.voices(), "owner", self.corpus.N,
                                     confirm=YES)
        self.assertIsNone(q2)
        self.assertEqual(note["outcome"], "owner_unknown")

    def test_a_source_that_is_not_a_string_still_compares(self):
        said, echo = said_and_echo(1)
        said["source"], echo["source"] = ["p1-s1"], ["p1-s1"]
        corpus = gq.Corpus([said, echo, *tg.toolboxes(8, start=20)], tg.spec(), False)
        num, cab = tg.TOOLBOX_RE.search(said["text"]).groups()
        it = tg.item(f"Where do I keep toolbox number {num}?", f"cabinet {cab}", [said["id"]])
        q = gq.assemble("direct_recall_user", it, (said["id"],), corpus, tg.spec(), "cand_00001")
        q2, note = gq.gold_or_common("direct_recall_user", q, corpus, None, "conversation", corpus.N,
                                     confirm=YES)
        self.assertEqual(q2["gold_ids"], [said["id"], echo["id"]])

    def test_a_record_past_the_gold_window_is_not_made_gold(self):
        q = self.build()
        q2, note = gq.gold_or_common("direct_recall_user", q, self.corpus, None, "conversation",
                                     self.corpus.pos[self.echo])
        self.assertIsNone(q2)
        self.assertEqual(note["gold_added"], [])


class CommonAnswerRunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=3"

    def records(self, leak_owner=None):
        out = []
        for i in range(24):
            out += said_and_echo(i)
            if leak_owner:
                out.append(elsewhere(i, leak_owner, 900 + i))
        return out

    def people(self):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(PEOPLE, f)
        return path

    def read(self, result, *parts):
        with open(os.path.join(result["out"], *parts), encoding="utf-8") as f:
            return f.read()

    def lines(self, result, name):
        return [json.loads(line) for line in self.read(result, "generation", name).splitlines()]

    def test_without_the_switch_the_item_is_held_and_tried_again(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(self.records(), self.TARGETS)
        self.assertIn("no item survived", str(cm.exception))
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(cands and all(c["fate"] != "frozen" and "common" not in c for c in cands))
        self.assertTrue(any(c["retry_of"] is not None for c in cands))

    def test_conversation_takes_the_echo_as_gold_and_asks_for_no_second_try(self):
        result = self.gen(self.records(), self.TARGETS, common_answers="conversation")
        questions = json.loads(self.read(result, "questions.json"))
        self.assertEqual(len(questions), 3)
        for q in questions:
            said, echo = (self._by_id(g) for g in q["gold_ids"])
            self.assertEqual(said["source"], echo["source"])
            self.assertIn(q["answer"], echo["text"])
            self.assertEqual(set(q["gold_hashes"]), set(q["gold_ids"]))
            self.assertNotIn(rc.COMMON_ANSWER, q)
        cands = self.lines(result, "candidates.jsonl")
        frozen = [c for c in cands if c["question_id"]]
        self.assertTrue(all(c["common"]["outcome"] == "gold" and len(c["common"]["gold_added"]) == 1 for c in frozen))
        self.assertTrue(all(c["retry_of"] is None for c in cands))
        self.assertEqual(json.loads(self.read(result, "generation", "run.json"))["common_answers"], "conversation")
        self.assertEqual(result["manifest"]["common_answers"]["mode"], "conversation")
        self.assertEqual((result["manifest"]["common_answers"]["gold_grown"],
                          result["manifest"]["common_answers"]["records_added"],
                          result["manifest"]["common_answers"][rc.COMMON_ANSWER]), (3, 3, 0))
        report = json.loads(self.read(result, "lint_report.json"))
        self.assertEqual(report["summary"]["items_with_errors"], 0)

    def _by_id(self, rid):
        return next(r for r in self.records("p2") if r["id"] == rid)

    def test_owner_keeps_an_answer_that_only_other_owners_records_share(self):
        result = self.gen(self.records("p2"), self.TARGETS, model=tg.FakeModel(tv.person_writer),
                          people=self.people(), prompts_dir=V7, owner=True, common_answers="owner")
        questions = json.loads(self.read(result, "questions.json"))
        n = len(questions)  # the sample also shows the other owner's records, which the writer skips
        self.assertGreaterEqual(n, 2)
        self.assertTrue(all(q[rc.COMMON_ANSWER] is True and len(q["gold_ids"]) == 2 for q in questions))
        self.assertEqual(list(questions[0])[-1], rc.COMMON_ANSWER)  # after the lint spec's own fields
        self.assertEqual(result["manifest"]["common_answers"],
                         dict(result["manifest"]["common_answers"], mode="owner", gold_grown=n, records_added=n,
                              common_answer=n))
        report = json.loads(self.read(result, "lint_report.json"))
        self.assertEqual(report["summary"]["items_with_errors"], 0)
        codes = [f["code"] for it in report["items"] for f in it["findings"]]
        self.assertEqual(codes.count("common_answer"), n)
        self.assertTrue(all(c["retry_of"] is None for c in self.lines(result, "candidates.jsonl")))

    def test_owner_leaves_the_item_held_when_the_owner_has_the_answer_elsewhere(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(self.records("p1"), self.TARGETS, model=tg.FakeModel(tv.person_writer),
                     people=self.people(), prompts_dir=V7, owner=True, common_answers="owner")
        self.assertIn("no item survived", str(cm.exception))
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        first = [c for c in cands if c["retry_of"] is None and c.get("common")]
        self.assertTrue(first and all(c["common"]["outcome"] == "held" and c["common"]["same_owner"] == 1
                                      and c["fate"] == "held_by_lint" for c in first))

    def test_a_replay_with_the_switch_rebuilds_the_set_byte_for_byte(self):
        corpus = self.corpus(self.records("p2"))
        kw = dict(quiet=True, people=self.people(), prompts_dir=V7, owner=True, common_answers="owner")
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-o"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(tv.person_writer), **kw)
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-o"), 7, gq.parse_targets(self.TARGETS),
                                 replay=os.path.join(first["out"], "generation", "calls.jsonl"), **kw)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        self.assertIn("questions.json", files)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_switch_is_not_recorded_when_it_is_not_given(self):
        result = self.gen(tg.toolboxes(24), self.TARGETS)
        self.assertNotIn("common_answers", json.loads(self.read(result, "generation", "run.json")))
        self.assertNotIn("common_answers", result["manifest"])
        self.assertTrue(all("common" not in c for c in self.lines(result, "candidates.jsonl")))

    def test_owner_mode_needs_the_owner_and_the_mode_must_be_known(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), self.TARGETS, common_answers="owner")
        self.assertIn("--common-answers owner needs --people and --owner", str(cm.exception))
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), self.TARGETS, common_answers="everything")
        self.assertIn("--common-answers everything", str(cm.exception))

    def test_the_command_line_takes_the_switch(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        base = ["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7", "--targets",
                "direct_recall_user=1", "--estimate"]
        self.assertEqual(gq.main(base + ["--common-answers", "conversation"]), 0)
        self.assertEqual(gq.main(base + ["--common-answers", "owner"]), 2)
        self.assertEqual(gq.main(base + ["--common-answers", "owner", "--people", self.people(), "--owner",
                                         "--prompts", V7]), 0)


class CommonAnswerScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recall-common-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        texts = [f"record number {i} about topic{i} with filler words here" for i in range(120)]
        texts[7] = "the orchard harvest yielded seven hundred bushels of apples"
        texts[8] = "a neighbour also counted seven hundred bushels of pears that year"
        self.texts = texts
        with open(os.path.join(self.tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(tr.make_record(t)) + "\n")

    def run_set(self, questions):
        path = os.path.join(self.tmp, "questions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(questions, f)
        return rb.run_benchmark(lambda: get_adapter("bm25"), "test", path, os.path.join(self.tmp, "corpus.jsonl"),
                                verify=False, quiet=True)

    def questions(self, mark):
        q1 = tr.make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels",
                              [self.texts[7]])
        if mark:
            q1[rc.COMMON_ANSWER] = True
        return [q1, tr.make_question("q2", "what is record number 3 about", "topic3", [self.texts[3]])]

    def test_a_common_answer_is_in_the_headline_and_not_in_the_exact_tier(self):
        res = self.run_set(self.questions(mark=True))
        q1, q2 = res["per_question"]
        self.assertEqual((q1["headline_rank"], q1["exact_rank"], q1[rc.COMMON_ANSWER]), (0, None, True))
        self.assertNotIn(rc.COMMON_ANSWER, q2)
        self.assertEqual(res["headline"]["n"], 2)
        self.assertEqual(res["diagnostics"]["exact"]["n"], 1)
        self.assertEqual(res["curves"]["exact"]["n"], 1)
        self.assertEqual(res["diagnostics"]["family"]["n"], 2)
        self.assertIn("1 of 2 answerable questions carry common_answer", res["diagnostics"]["exact_note"])
        self.assertEqual(res["pool"]["G_exact"]["max"], 1)  # q1's two holders are not counted
        self.assertEqual(res["by_category"]["direct_recall_user"]["exact"], 100.0)

    def test_without_the_mark_the_exact_tier_counts_it_as_before(self):
        res = self.run_set(self.questions(mark=False))
        q1 = res["per_question"][0]
        self.assertEqual(q1["exact_rank"], 0)
        self.assertNotIn(rc.COMMON_ANSWER, q1)
        self.assertEqual(res["diagnostics"]["exact"]["n"], 2)
        self.assertNotIn("exact_note", res["diagnostics"])
        self.assertEqual(res["pool"]["G_exact"]["max"], 2)


if __name__ == "__main__":
    unittest.main()

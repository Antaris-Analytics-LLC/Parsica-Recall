"""Tests for the question builder's --forms switch and the prompts written for it: a form on the ASK line beside the
shape (open, how, when, choice, two-part), one unit in ten asked as a long message, the checks each
needs relaxed for it alone, and a choice question that offers its answer marked so the lint notes it
instead of holding it. A faked model; no network and no SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_forms
"""

import json
import os
import sys
import types
import unittest
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from parsica_recall import generate_questions as gq  # noqa: E402
from parsica_recall import lint_validator as lv  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
import test_generate as tg  # noqa: E402
import test_voices as tv  # noqa: E402

V7 = os.path.join(REPO, "prompts", "v7")


def text_of(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def lines_of(*parts):
    return [json.loads(line) for line in text_of(*parts).splitlines() if line.strip()]


class FormVoicesTests(unittest.TestCase):
    def setUp(self):
        self.corpus = types.SimpleNamespace(by_id={f"r{i}": {"id": f"r{i}", "source": "p3-s1"} for i in range(1000)})

    def asks(self, forms, category="direct_recall_user", seed=5):
        v = gq.Voices(seed, {"p3": "Anjali"}, owner=True, forms=forms)
        return [v.ask(category, (f"r{i}",), self.corpus) for i in range(1000)]

    def test_each_ask_gets_a_seeded_form_and_one_in_ten_is_long(self):
        a = self.asks(True)
        self.assertTrue(all(set(x) == {"owner", "shape", "form"} for x in a))
        self.assertEqual(a, self.asks(True))
        self.assertNotEqual(a, self.asks(True, seed=6))
        forms = Counter(x["form"] for x in a)
        self.assertEqual(set(forms), {name for name, _ in gq.OWNER_FORMS})
        self.assertGreater(forms["open"], forms["choice"])  # 7 to 2 by weight
        long_share = sum(x["shape"] == "long" for x in a) / len(a)
        self.assertTrue(0.06 < long_share < 0.14, long_share)

    def test_a_unit_keeps_its_owner_shape_unless_it_becomes_long(self):
        plain, formed = self.asks(False), self.asks(True)
        for p, f in zip(plain, formed):
            self.assertEqual(f["owner"], p["owner"])
            if f["shape"] != "long":
                self.assertEqual(f["shape"], p["shape"])
        self.assertTrue(all("form" not in p and p["shape"] != "long" for p in plain))

    def test_abstention_is_always_open(self):
        self.assertTrue(all(x["form"] == "open" for x in self.asks(True, category="abstention")))

    def test_a_forced_form_goes_to_every_unit_but_abstention(self):
        v = gq.Voices(5, {"p3": "Anjali"}, owner=True, forms=True, force_form="choice")
        a = [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(200)]
        self.assertTrue(all(x["form"] == "choice" for x in a))
        self.assertEqual([x["shape"] for x in a], [x["shape"] for x in self.asks(True)[:200]])  # shapes as drawn
        self.assertEqual(v.ask("abstention", ("r1",), self.corpus)["form"], "open")
        self.assertEqual(v.describe()["form_forced"], "choice")

    def test_the_run_records_the_forms(self):
        self.assertEqual(gq.Voices(5, {"p3": "Anjali"}, owner=True, forms=True).describe(),
                         {"people_sha256": None, "asker": "owner", "shapes": dict(gq.OWNER_SHAPES),
                          "forms": dict(gq.OWNER_FORMS), "long": f"1 in {gq.LONG_ONE_IN}"})


class FormCheckTests(unittest.TestCase):
    def setUp(self):
        self.records = tg.toolboxes(4)
        self.corpus = gq.Corpus(self.records, tg.spec(), False)
        self.unit = (self.records[0]["id"],)
        num, cab = tg.TOOLBOX_RE.search(self.records[0]["text"]).groups()
        self.answer, self.num = f"cabinet {cab}", num

    def check(self, question, shape="plain", form="open"):
        it = tg.item(question, self.answer, [self.unit[0]])
        return gq.rule_checks("direct_recall_user", it, self.unit, self.corpus, roles=True, owner="Maren",
                              ask={"owner": "Maren", "shape": shape, "form": form})

    def test_plain_messages_keep_every_rule(self):
        self.assertIsNone(self.check(f"Where do I keep toolbox {self.num}?"))
        self.assertIn("more than one question mark", self.check(f"Toolbox {self.num}? Where is it?"))
        self.assertIn("goes on with", self.check(f"Where is toolbox {self.num} and what colour is it?"))
        self.assertIn("longer than 30", self.check("Where " + "really " * 30 + f"is toolbox {self.num}?"))
        self.assertIn("does not end", self.check(f"Where is toolbox {self.num}? Thanks"))

    def test_a_two_part_message_may_hold_two_questions(self):
        lead = f"Do you remember when we sorted out toolbox {self.num}? Where did we put it?"
        self.assertIsNone(self.check(lead, form="two-part"))
        self.assertIsNone(self.check(f"Where is toolbox {self.num} and what did we decide about it?", form="two-part"))
        self.assertIn("longer than 35", self.check("Where " + "really " * 35 + f"is toolbox {self.num}?",
                                                   form="two-part"))
        self.assertIn("3 question marks", self.check(f"Toolbox? {self.num}? Where?", form="two-part"))
        self.assertIn("does not end", self.check(f"Where is toolbox {self.num}? Thanks", form="two-part"))

    def test_a_long_message_may_put_its_question_anywhere(self):
        body = ("Long day here, the kids had swimming and the car made that noise again on the way home. "
                f"Anyway, where did I put toolbox {self.num}? I need the wrench before the weekend.")
        self.assertIsNone(self.check(body, shape="long"))
        self.assertIn("no question mark", self.check(body.replace("?", "."), shape="long"))
        self.assertIn("longer than 80", self.check("word " * 81 + "where?", shape="long"))


class ChoiceTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=6"

    def people(self):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"*": "Maren"}, f)
        return path

    def test_a_choice_question_is_marked_and_kept_and_any_other_is_held(self):
        def chooser(category, units):  # every item offers its answer, whatever form it was asked in
            out = []
            for it in tv.person_writer(category, units):
                num = it["question"].split("number ")[-1].rstrip("?")
                out.append(dict(it, question=f"Do I keep toolbox number {num} in {it['answer']} or the garage?"))
            return out

        result = self.gen(tg.toolboxes(60), self.TARGETS, model=tg.FakeModel(chooser), people=self.people(),
                          prompts_dir=V7, owner=True, forms=True)
        questions = json.loads(text_of(result["out"], "questions.json"))
        self.assertTrue(questions and all(q.get(rc.CHOICE) is True for q in questions))
        cands = lines_of(result["out"], "generation", "candidates.jsonl")
        kept = [c for c in cands if c["question_id"]]
        self.assertTrue(all(c["ask"]["form"] == "choice" for c in kept))
        others = [c for c in cands if c["unit"] and c["ask"]["form"] != "choice" and c["retry_of"] is None]
        self.assertTrue(others and all(c["fate"] != "frozen" for c in others))
        self.assertTrue(all("answer_in_question" in [f["code"] for f in c["findings"]] for c in others
                            if c["fate"] == "held_by_lint"))
        report = json.loads(text_of(result["out"], "lint_report.json"))
        self.assertEqual(report["summary"]["items_with_errors"], 0)

    def test_an_alternate_answer_offered_in_a_choice_marks_it_too(self):
        def chooser(category, units):  # offers the alternate spelling, not the answer as written
            out = []
            for it in tv.person_writer(category, units):
                num = it["question"].split("number ")[-1].rstrip("?")
                cab = it["answer"].split()[-1]
                # "cabinet-xqab" normalizes to one word that no record holds, so the small test pool does
                # not count it too generic, and only the alternate, not the answer, is in the question
                out.append(dict(it, answer=f"cabinet {cab}", acceptable_answers=[f"cabinet-{cab}"],
                                question=f"Is toolbox number {num} in cabinet-{cab} or in the garage?"))
            return out

        result = self.gen(tg.toolboxes(60), self.TARGETS, model=tg.FakeModel(chooser), people=self.people(),
                          prompts_dir=V7, owner=True, forms=True, force_form="choice")
        questions = json.loads(text_of(result["out"], "questions.json"))
        self.assertTrue(questions and all(q.get(rc.CHOICE) is True for q in questions))
        self.assertEqual(result["manifest"]["voices"]["form_forced"], "choice")

    def test_the_lint_holds_an_unmarked_answer_in_the_question(self):
        records = tg.toolboxes(4)
        corpus = gq.Corpus(records, tg.spec(), False)
        num, cab = tg.TOOLBOX_RE.search(records[0]["text"]).groups()
        it = tg.item(f"Is toolbox {num} in cabinet {cab} or the garage?", f"cabinet {cab}", [records[0]["id"]])
        q = gq.assemble("direct_recall_user", it, (records[0]["id"],), corpus, tg.spec(), "cand_00001")
        res = lv.lint_item(q, 1, corpus.ctx)
        self.assertIn("answer_in_question", [f["code"] for f in res["findings"] if f["severity"] == "error"])
        res = lv.lint_item(dict(q, **{rc.CHOICE: True}), 1, corpus.ctx)
        self.assertTrue(res["publishable"])
        self.assertEqual([(f["code"], f["severity"]) for f in res["findings"] if f["rule_id"] == "ANSWER_NOT_IN_QUESTION"],
                         [("choice", "info")])


class FormRunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=2,direct_recall_agent=1,abstention=1"

    def people(self):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"*": "Maren"}, f)
        return path

    def test_a_forced_form_needs_forms_and_a_known_form(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", people=self.people(), owner=True, force_form="choice")
        self.assertIn("--force-form choice: needs --forms", str(cm.exception))
        with self.assertRaises(gq.GenerateError):
            self.gen(tg.toolboxes(24), "direct_recall_user=2", people=self.people(), owner=True, forms=True,
                     force_form="riddle")

    def test_forms_needs_the_owner(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", forms=True)
        self.assertIn("--forms needs --people and --owner", str(cm.exception))
        with self.assertRaises(gq.GenerateError):
            self.gen(tg.toolboxes(24), "direct_recall_user=2", forms=True, people=self.people())

    def test_a_forms_run_records_the_forms_and_replays_byte_for_byte(self):
        corpus = self.corpus(tg.toolboxes(24))
        kw = dict(quiet=True, people=self.people(), prompts_dir=V7, owner=True, forms=True)
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-f"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(tv.person_writer), **kw)
        run = json.loads(text_of(first["out"], "generation", "run.json"))
        self.assertEqual(run["voices"]["forms"], dict(gq.OWNER_FORMS))
        self.assertEqual(first["manifest"]["voices"], run["voices"])
        calls = lines_of(first["out"], "generation", "calls.jsonl")
        self.assertTrue(calls and all('"form": "' in c["user"] for c in calls))
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-f"), 7, gq.parse_targets(self.TARGETS),
                                 replay=os.path.join(first["out"], "generation", "calls.jsonl"), **kw)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_command_line_takes_forms_with_the_owner(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        base = ["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7", "--targets",
                "direct_recall_user=1", "--estimate", "--prompts", V7]
        self.assertEqual(gq.main(base + ["--people", self.people(), "--owner", "--forms"]), 0)
        self.assertEqual(gq.main(base + ["--forms"]), 2)


if __name__ == "__main__":
    unittest.main()

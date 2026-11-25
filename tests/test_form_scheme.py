"""Tests for the question builder's --form-scheme 2 and prompts/v7: the forms drawn by OWNER_FORMS_2,
the when form only for a unit whose shown records hold a time (or a temporal_awareness unit), and the
side of a choice question's right answer drawn onto the ASK line. Scheme 1, or none, draws
as before. A faked model; no network and no SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_form_scheme
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
from parsica_recall import recall_common as rc  # noqa: E402
import test_generate as tg  # noqa: E402
import test_voices as tv  # noqa: E402

V7 = os.path.join(REPO, "prompts", "v7")


def text_of(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def lines_of(*parts):
    return [json.loads(line) for line in text_of(*parts).splitlines() if line.strip()]


class HasTimeTests(unittest.TestCase):
    def test_times_dates_days_and_lengths_of_time_count(self):
        for text in ("We meet at 7:30 tonight.", "The train leaves at 9pm.", "Back at 6 a.m. sharp.",
                     "Lunch at noon.", "The visit is on 14 June.", "It moved to June 14th.",
                     "She started in March 2016.", "Due 2026-10-30.", "Every Tuesday we swim.",
                     "The bake took 45 minutes.", "Give it two weeks.", "About half an hour, then rest.",
                     "A 30-minute walk."):
            self.assertTrue(gq.has_time(text), text)

    def test_other_words_and_numbers_do_not(self):
        for text in ("The sun was out.", "I sat down by the sea.", "Clamp it 1/4 inch from the edge.",
                     "Yesterday we talked about the shed.", "May I ask about the paint?", "A 4 amp fuse.",
                     "The 3rd shelf from the top.", "We wed in the garden.", "", None):
            self.assertFalse(gq.has_time(text), text)


class SchemeTwoVoicesTests(unittest.TestCase):
    def setUp(self):
        # even records hold a time, odd ones do not
        self.corpus = types.SimpleNamespace(by_id={
            f"r{i}": {"id": f"r{i}", "source": "p3-s1",
                      "text": "We talked at 7 pm about the shed." if i % 2 == 0 else "We talked about the shed."}
            for i in range(2000)})

    def voices(self, scheme=2, **kw):
        return gq.Voices(5, {"p3": "Anjali"}, owner=True, forms=True, form_scheme=scheme, **kw)

    def test_when_goes_only_to_a_unit_whose_records_hold_a_time(self):
        v = self.voices()
        asks = {i: v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)}
        self.assertFalse([i for i, a in asks.items() if i % 2 and a["form"] == "when"])
        self.assertGreater(sum(a["form"] == "when" for i, a in asks.items() if i % 2 == 0), 100)
        self.assertTrue(v.timed("temporal_awareness", ("r1",), self.corpus))
        self.assertFalse(v.timed("direct_recall_user", ("r1",), self.corpus))
        temporal = [v.ask("temporal_awareness", (f"r{i}",), self.corpus) for i in range(1, 2000, 2)]
        self.assertGreater(sum(a["form"] == "when" for a in temporal), 50)  # untimed text, yet when

    def test_a_choice_carries_a_drawn_side_and_nothing_else_does(self):
        v = self.voices()
        asks = [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)]
        sides = Counter(a["right"] for a in asks if a["form"] == "choice")
        self.assertEqual(set(sides), set(gq.CHOICE_SIDES))
        self.assertTrue(0.35 < sides["first"] / sum(sides.values()) < 0.65, sides)
        self.assertTrue(all("right" not in a for a in asks if a["form"] != "choice"))
        self.assertEqual(asks, [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)])
        one = [self.voices(scheme=1).ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)]
        self.assertTrue(all("right" not in a for a in one))

    def test_choice_comes_to_about_one_unit_in_thirteen(self):
        v = self.voices()
        asks = [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)]
        share = sum(a["form"] == "choice" for a in asks) / len(asks)
        self.assertTrue(0.05 < share < 0.11, share)

    def test_shapes_and_the_long_shape_are_drawn_as_in_scheme_one(self):
        two = [self.voices().ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(500)]
        one = [self.voices(scheme=1).ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(500)]
        self.assertEqual([a["shape"] for a in two], [a["shape"] for a in one])

    def test_a_forced_choice_still_gets_its_side_and_abstention_stays_open(self):
        v = self.voices(force_form="choice")
        asks = [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(200)]
        self.assertTrue(all(a["form"] == "choice" and a["right"] in gq.CHOICE_SIDES for a in asks))
        self.assertEqual(v.ask("abstention", ("r1",), self.corpus), {"shape": v.ask("abstention", ("r1",),
                                                                                         self.corpus)["shape"],
                                                                      "form": "open", "owner": "Anjali"})

    def test_the_run_records_scheme_two_and_scheme_one_stays_as_it_was(self):
        two = self.voices().describe()
        self.assertEqual(two["form_scheme"], 2)
        self.assertEqual(two["forms"], dict(gq.OWNER_FORMS_2))
        self.assertEqual(two["choice_right"], list(gq.CHOICE_SIDES))
        self.assertIn("time", two["when"])
        self.assertEqual(self.voices(scheme=1).describe(),
                         {"people_sha256": None, "asker": "owner", "shapes": dict(gq.OWNER_SHAPES),
                          "forms": dict(gq.OWNER_FORMS), "long": f"1 in {gq.LONG_ONE_IN}"})

    def test_the_ask_line_shows_the_side(self):
        v = self.voices(force_form="choice")
        corpus = gq.Corpus(tg.toolboxes(4), tg.spec(), False)
        rid = corpus.ids[0]
        line = gq.user_message("direct_recall_user", [(rid,)], corpus, v).split("\n")[0]
        self.assertTrue(line.startswith(gq.ASK_HEAD))
        ask = json.loads(line[len(gq.ASK_HEAD):])
        self.assertEqual(ask["form"], "choice")
        self.assertIn(ask["right"], gq.CHOICE_SIDES)


class SchemeThreeVoicesTests(unittest.TestCase):
    def setUp(self):
        # every record holds a time, so only the category can keep the when form away
        self.corpus = types.SimpleNamespace(by_id={
            f"r{i}": {"id": f"r{i}", "source": "p3-s1", "text": "We talked at 7 pm about the shed."} for i in range(2000)})

    def voices(self, scheme=3, **kw):
        return gq.Voices(5, {"p3": "Anjali"}, owner=True, forms=True, form_scheme=scheme, **kw)

    def forms_in(self, category, scheme=3):
        v = self.voices(scheme)
        return Counter(v.ask(category, (f"r{i}",), self.corpus)["form"] for i in range(2000))

    def test_when_stays_out_of_the_categories_whose_answer_cannot_be_a_time(self):
        for cat in ("entity_attribution", "vocabulary_gap", "implicit_preference", "recency_preference"):
            self.assertEqual(self.forms_in(cat)["when"], 0, cat)
            self.assertGreater(self.forms_in(cat, scheme=2)["when"], 0, cat)  # scheme 2 let it in
        for cat in gq.WHEN_CATEGORIES:
            self.assertGreater(self.forms_in(cat)["when"], 200, cat)

    def test_a_when_category_still_needs_a_time_in_its_records(self):
        corpus = types.SimpleNamespace(by_id={f"r{i}": {"id": f"r{i}", "source": "p3-s1", "text": "The shed."}
                                              for i in range(500)})
        v = self.voices()
        self.assertFalse(v.timed("direct_recall_user", ("r1",), corpus))
        self.assertTrue(v.timed("temporal_awareness", ("r1",), corpus))
        self.assertEqual(sum(v.ask("direct_recall_user", (f"r{i}",), corpus)["form"] == "when" for i in range(500)), 0)

    def test_choice_keeps_its_side_and_the_run_records_scheme_three(self):
        asks = [self.voices().ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(2000)]
        self.assertTrue(all(a["right"] in gq.CHOICE_SIDES for a in asks if a["form"] == "choice"))
        self.assertTrue(all("right" not in a for a in asks if a["form"] != "choice"))
        d = self.voices().describe()
        self.assertEqual((d["form_scheme"], d["forms"]), (3, dict(gq.OWNER_FORMS_3)))
        self.assertIn("knowledge_update", d["when"])
        self.assertEqual(self.voices(scheme=2).describe()["forms"], dict(gq.OWNER_FORMS_2))

    def test_scheme_two_draws_as_it_did(self):
        # the v7 check run's sets were built with scheme 2: its draw must not move
        two = [self.voices(scheme=2).ask("entity_attribution", (f"r{i}",), self.corpus) for i in range(300)]
        again = [gq.Voices(5, {"p3": "Anjali"}, owner=True, forms=True, form_scheme=2).ask(
            "entity_attribution", (f"r{i}",), self.corpus) for i in range(300)]
        self.assertEqual(two, again)
        self.assertGreater(sum(a["form"] == "when" for a in two), 0)


class SchemeTwoRunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=2,direct_recall_agent=1,abstention=1"

    def people(self):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"*": "Maren"}, f)
        return path

    def test_scheme_two_needs_forms_and_a_known_scheme(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", people=self.people(), owner=True, form_scheme=2)
        self.assertIn("--form-scheme 2: needs --forms", str(cm.exception))
        for bad in (4, 0, True):
            with self.assertRaises(gq.GenerateError):
                self.gen(tg.toolboxes(24), "direct_recall_user=2", people=self.people(), owner=True, forms=True,
                         form_scheme=bad)

    def test_a_scheme_two_run_records_it_shows_the_side_and_replays_byte_for_byte(self):
        corpus = self.corpus(tg.toolboxes(24))
        kw = dict(quiet=True, people=self.people(), prompts_dir=V7, owner=True, forms=True, form_scheme=2,
                  force_form="choice")
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-s"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(tv.person_writer), **kw)
        run = json.loads(text_of(first["out"], "generation", "run.json"))
        self.assertEqual(run["voices"]["form_scheme"], 2)
        self.assertEqual(first["manifest"]["voices"], run["voices"])
        calls = lines_of(first["out"], "generation", "calls.jsonl")
        self.assertTrue(calls and any('"right": "' in c["user"] for c in calls))
        cands = [c for c in lines_of(first["out"], "generation", "candidates.jsonl") if c["unit"]]
        self.assertTrue(all(c["ask"]["right"] in gq.CHOICE_SIDES for c in cands if c["ask"]["form"] == "choice"))
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-s"), 7, gq.parse_targets(self.TARGETS),
                                 replay=os.path.join(first["out"], "generation", "calls.jsonl"), **kw)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_without_the_scheme_a_forms_run_draws_by_scheme_one(self):
        corpus = self.corpus(tg.toolboxes(24))
        kw = dict(quiet=True, people=self.people(), prompts_dir=V7, owner=True, forms=True)
        a = gq.generate_set(corpus, os.path.join(self.tmp, "c", "set-o"), 7, gq.parse_targets(self.TARGETS),
                            model=tg.FakeModel(tv.person_writer), **kw)
        b = gq.generate_set(corpus, os.path.join(self.tmp, "d", "set-o"), 7, gq.parse_targets(self.TARGETS),
                            model=tg.FakeModel(tv.person_writer), form_scheme=1, **kw)
        self.assertNotIn("form_scheme", a["manifest"]["voices"])
        for name in ("questions.json", os.path.join("generation", "calls.jsonl")):
            self.assertEqual(text_of(a["out"], name), text_of(b["out"], name), name)

    def test_the_command_line_takes_the_scheme_with_forms(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        base = ["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7", "--targets",
                "direct_recall_user=1", "--estimate", "--prompts", V7, "--people", self.people(), "--owner"]
        self.assertEqual(gq.main(base + ["--forms", "--form-scheme", "2"]), 0)
        self.assertEqual(gq.main(base + ["--forms", "--form-scheme", "3"]), 0)
        self.assertEqual(gq.main(base + ["--form-scheme", "2"]), 2)

    def test_a_scheme_three_run_replays_byte_for_byte(self):
        corpus = self.corpus(tg.toolboxes(24))
        kw = dict(quiet=True, people=self.people(), prompts_dir=V7, owner=True, forms=True, form_scheme=3)
        first = gq.generate_set(corpus, os.path.join(self.tmp, "e", "set-t"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(tv.person_writer), **kw)
        self.assertEqual(json.loads(text_of(first["out"], "generation", "run.json"))["voices"]["form_scheme"], 3)
        second = gq.generate_set(corpus, os.path.join(self.tmp, "f", "set-t"), 7, gq.parse_targets(self.TARGETS),
                                 replay=os.path.join(first["out"], "generation", "calls.jsonl"), **kw)
        for name in ("questions.json", "manifest.json", os.path.join("generation", "candidates.jsonl")):
            self.assertEqual(text_of(first["out"], name), text_of(second["out"], name), name)


class PromptV7Tests(unittest.TestCase):
    def test_every_v7_prompt_keeps_the_schema_and_explains_the_ask_line(self):
        old = gq.load_prompts(rc.CATEGORIES)
        new = gq.load_prompts(rc.CATEGORIES, V7)
        for cat in rc.CATEGORIES:
            text = new[cat].text
            self.assertEqual(new[cat].schema, old[cat].schema, cat)
            self.assertIn("(prompts v7: the owner asks)", text.split("\n")[0], cat)
            self.assertNotIn(chr(0x2014), text, cat)  # no em dash
            self.assertEqual(text.count('The ASK line\'s "right" says where the right one goes'), 1, cat)
            self.assertEqual(text.count("not with a stock lead-in"), 1, cat)
            self.assertEqual(text.count("which side its right answer goes on"), 1, cat)
            for name, _ in gq.OWNER_FORMS + gq.OWNER_SHAPES + (("long", 0),):
                self.assertEqual(text.count(f'- "{name}": '), 1, (cat, name))
            self.assertEqual(text.count("A two-part or a long message follows its own line above."), 1, cat)
            self.assertIn("except in a two-part message", text, cat)


if __name__ == "__main__":
    unittest.main()

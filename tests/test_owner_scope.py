"""Tests for the question builder's limits in owner mode and on the forms: a pair, a near copy and a
promoted record are the owner's own; a promotion is checked and recorded; a choice offers two options;
times are read as times; and the checks keep their wording. A faked model; no network and no SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_owner_scope
"""

import json
import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from parsica_recall import generate_questions as gq  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
import test_common_answers as tca  # noqa: E402
import test_generate as tg  # noqa: E402

V7 = os.path.join(REPO, "prompts", "v7")


def owner_pair_writer(category, units):
    """pair_writer's items, asked by the owner: no role words."""
    return [dict(it, question=it["question"].replace("Where does squad", "Where did my squad"))
            for it in tg.pair_writer(category, units)]


class OwnerPairTests(tg.GenTestCase):
    def people(self, mapping):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
        return path

    def test_a_pair_across_two_owners_is_never_asked(self):
        recs = []
        for p in range(4):
            early, late = tg.standup_pair(p)
            first, second = ("p1", "p1") if p % 2 == 0 else ("p1", "p2")
            recs += [dict(early, source=f"{first}-{early['source']}"), dict(late, source=f"{second}-{late['source']}")]
        recs += tg.toolboxes(4)
        people = {"p1": "Maren", "p2": "Odile", "*": "Robin"}
        by_id = {r["id"]: r for r in recs}
        voices = gq.Voices(7, people, owner=True)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "recency_preference=2", model=tg.FakeModel(owner_pair_writer),
                           people=self.people(people), prompts_dir=V7, owner=True)
        pairs = [c["unit"] for c in res["candidates"] if c["category"] == "recency_preference" and c["unit"]]
        self.assertTrue(pairs)
        for unit in pairs:
            self.assertEqual(len({voices.key_of(by_id[r]) for r in unit}), 1, unit)
        for q in res["questions"]:
            self.assertEqual(len({voices.key_of(by_id[g]) for g in q["gold_ids"]}), 1, q["gold_ids"])

    def test_a_near_copy_in_another_owners_records_is_not_gold(self):
        said, echo = tca.said_and_echo(1)
        copy = dict(said, id="copy-of-said", source="p2-s900")
        corpus = gq.Corpus([said, copy, *tg.toolboxes(8, start=20)], tg.spec(), True)
        voices = gq.Voices(7, {"p1": "Maren", "p2": "Odile"}, owner=True)
        num, cab = tg.TOOLBOX_RE.search(said["text"]).groups()
        it = tg.item(f"Where do I keep toolbox number {num}?", f"cabinet {cab}", [said["id"]])
        self.assertEqual(corpus.cluster[said["id"]], [said["id"], copy["id"]])
        mine = gq.assemble("direct_recall_user", it, (said["id"],), corpus, tg.spec(), "c1",
                           keep=lambda m: voices.key_of(corpus.by_id[m]) == "p1")
        self.assertEqual(mine["gold_ids"], [said["id"]])
        everyone = gq.assemble("direct_recall_user", it, (said["id"],), corpus, tg.spec(), "c1")
        self.assertEqual(everyone["gold_ids"], [said["id"], copy["id"]])  # without an owner, as before


class PromotionCheckRunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=3"

    def records(self):
        return [r for i in range(24) for r in tca.said_and_echo(i)]

    def lines(self, result, name):
        with open(os.path.join(result["out"], "generation", name), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_each_promotion_is_a_recorded_check_call(self):
        model = tg.FakeModel()
        result = self.gen(self.records(), self.TARGETS, model=model, common_answers="conversation")
        checks = [c for c in self.lines(result, "calls.jsonl") if c["category"] == gq.CHECK_CATEGORY]
        self.assertTrue(checks)
        self.assertEqual(len(checks), len(model.checks))
        self.assertTrue(all(c["check_of"] and c["user"].startswith("QUESTION: ") and len(c["units"]) == 1
                            for c in checks))
        self.assertEqual(result["manifest"]["model"]["check_calls"], len(checks))
        run = json.loads(open(os.path.join(result["out"], "generation", "run.json"), encoding="utf-8").read())
        prompt = gq.Prompt(gq.CHECK_CATEGORY, gq.CHECK_PROMPT)
        self.assertEqual(run["check_prompt"], prompt.sha256)
        with open(os.path.join(result["out"], "generation", "prompts", f"{gq.CHECK_CATEGORY}.md"), "rb") as f:
            self.assertEqual(f.read(), prompt.data)
        cands = [c for c in self.lines(result, "candidates.jsonl") if c.get("question_id")]
        self.assertTrue(all(c["common"]["checked"][0]["answers"] is True for c in cands))

    def test_a_no_keeps_the_echo_out_and_the_item_goes_to_its_retry(self):
        model = tg.FakeModel(checker=lambda user: False)
        with self.assertRaises(gq.GenerateError):
            self.gen(self.records(), self.TARGETS, model=model, common_answers="conversation")
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        firsts = [c for c in cands if c.get("common")]
        self.assertTrue(firsts and all(c["common"]["gold_added"] == [] for c in firsts))
        self.assertTrue(any(c["retry_of"] is not None for c in cands))

    def test_a_run_without_the_switch_makes_no_check_and_records_nothing_new(self):
        result = self.gen(tg.toolboxes(24), "direct_recall_user=2")
        self.assertNotIn("check_calls", result["manifest"]["model"])
        run = json.loads(open(os.path.join(result["out"], "generation", "run.json"), encoding="utf-8").read())
        self.assertNotIn("check_prompt", run)

    def test_the_check_prompt_loads_and_asks_for_one_yes_or_no(self):
        prompt = gq.Prompt(gq.CHECK_CATEGORY, gq.CHECK_PROMPT)
        self.assertEqual(prompt.fields, {"answers"})
        self.assertNotIn(chr(0x2014), prompt.text)  # no em dash
        self.assertTrue(gq.check_verdict('{"items": [{"answers": true}]}'))
        for text in ('{"items": [{"answers": false}]}', '{"items": []}', "yes", '{"items": [{"answers": "true"}]}',
                     '{"items": [{"answers": true}, {"answers": true}]}'):
            self.assertFalse(gq.check_verdict(text), text)


class ChoiceSideTests(unittest.TestCase):
    def side(self, question, answer, alts=()):
        return gq.choice_side({"question": question, "answer": answer, "acceptable_answers": list(alts)})

    def test_the_side_the_answer_is_offered_on(self):
        self.assertEqual(self.side("Did I say left or right?", "left"), "first")
        self.assertEqual(self.side("Was it the 8:10 or the 7:40 ferry I switched to?", "7:40 ferry"), "second")
        self.assertEqual(self.side("Is it in cabinet xqaa or cabinet xqab?", "cabinet xqab"), "second")

    def test_no_choice_without_two_options(self):
        self.assertIsNone(self.side("Could you remind me about cabinet xqad?", "cabinet xqad"))
        self.assertIsNone(self.side("Was it cabinet xqad or something?", "cabinet xqad"))
        self.assertIsNone(self.side("Was it left or not?", "left"))
        self.assertIsNone(self.side("Was it left or left again?", "left"))


class CheckWordingTests(unittest.TestCase):
    def check(self, question, owner=None, ask=None):
        recs = tg.toolboxes(4)
        corpus = gq.Corpus(recs, tg.spec(), False)
        return gq.rule_checks("direct_recall_user", tg.item(question, "cabinet xqab", [recs[1]["id"]]),
                              (recs[1]["id"],), corpus, roles=True, owner=owner, ask=ask)

    def test_an_owner_named_in_any_case_is_dropped(self):
        for q in ("What did Robin say about the cabinet?", "what did robin say about the cabinet?",
                  "What did ROBIN say about the cabinet?"):
            self.assertIn("names its owner", self.check(q, owner="Robin"), q)
        self.assertNotIn("names its owner", self.check("What did Robinson say about the cabinet?", owner="Robin") or "")

    def test_an_ordinary_message_keeps_its_old_reason_for_extra_question_marks(self):
        self.assertEqual(self.check("Why? How? Which?"),
                         "two questions in one: the item holds more than one question mark")
        self.assertEqual(self.check("Why? How? Which cabinet did I pick?", ask={"shape": "long"}),
                         "the message holds 3 question marks")


class TimeTests(unittest.TestCase):
    def test_dates_with_slashes_and_seconds_count(self):
        for text in ("filed on 10/31/2026", "due 2026/11/04", "it took 45 seconds", "a 30 sec wait"):
            self.assertTrue(gq.has_time(text), text)

    def test_what_is_not_a_time_does_not(self):
        for text in ("code 99:99 on the panel", "add 1/2 cup of flour", "3/4 of the team", "score 25:61"):
            self.assertFalse(gq.has_time(text), text)
        for text in ("at 23:59 sharp", "by 9:05", "at 0:30"):
            self.assertTrue(gq.has_time(text), text)


class PeopleScreenTests(tg.GenTestCase):
    def test_a_name_with_the_shape_of_personal_data_is_refused(self):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"p1": "Maren", "p2": "someone" + "@" + "example.com"}, f)
        with self.assertRaises(gq.GenerateError) as cm:
            gq.read_people(path)
        self.assertIn("'p2'", str(cm.exception))
        self.assertNotIn("example", str(cm.exception))


class EmptyExactTierTests(unittest.TestCase):
    def test_no_question_to_score_gives_no_chance_and_no_validity(self):
        t = rb.tier_table([], [0.1] * 20, [1, 5, 15], 20)
        self.assertEqual(t["n"], 0)
        for k in ("1", "5", "15"):
            self.assertIsNone(t["recall"][k])
            self.assertIsNone(t["chance"][k])
            self.assertIsNone(t["skill"][k])
            self.assertIsNone(t["valid"][k])


if __name__ == "__main__":
    unittest.main()

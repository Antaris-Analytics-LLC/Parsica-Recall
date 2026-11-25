"""Tests for the question builder's owner mode (--owner beside --people) and prompts/v7: the ASK
line that names the owner, the check that drops a question naming its owner, a topic that shows
only its owner's records, and what a candidate and a set record. A faked model; no network and no
SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_owner
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


def naming_writer(category, units):
    """person_writer's items, with the owner named instead of speaking as I."""
    return [dict(it, question=it["question"].replace(" I ", " Maren ")) for it in tv.person_writer(category, units)]


class OwnerVoicesTests(unittest.TestCase):
    def setUp(self):
        self.corpus = types.SimpleNamespace(by_id={f"r{i}": {"id": f"r{i}", "source": "p3-s1"} for i in range(400)})

    def test_the_ask_gives_the_owner_and_a_seeded_shape_and_no_voice(self):
        v = gq.Voices(5, {"p3": "Anjali"}, owner=True)
        a = [v.ask("direct_recall_user", (f"r{i}",), self.corpus) for i in range(400)]
        self.assertTrue(all(set(x) == {"owner", "shape"} and x["owner"] == "Anjali" for x in a))
        self.assertEqual(a, [gq.Voices(5, {"p3": "Anjali"}, owner=True).ask("direct_recall_user", (f"r{i}",), self.corpus)
                             for i in range(400)])
        self.assertNotEqual(a, [gq.Voices(6, {"p3": "Anjali"}, owner=True).ask("direct_recall_user", (f"r{i}",),
                                                                               self.corpus) for i in range(400)])
        shapes = Counter(x["shape"] for x in a)
        self.assertEqual(set(shapes), {name for name, _ in gq.OWNER_SHAPES})
        self.assertGreater(shapes["plain"], shapes["thinking"])  # 6 to 2 by weight

    def test_without_owner_the_ask_is_what_it_was(self):
        mixed = gq.Voices(5, {"p3": "Anjali"}).ask("direct_recall_user", ("r1",), self.corpus)
        self.assertEqual(set(mixed), {"person", "shape", "voice"})
        self.assertEqual(gq.Voices(5, {"p3": "Anjali"}).describe(),
                         {"people_sha256": None, "voices": dict(gq.VOICES), "shapes": dict(gq.SHAPES)})
        self.assertEqual(gq.Voices(5, {"p3": "Anjali"}, owner=True).describe(),
                         {"people_sha256": None, "asker": "owner", "shapes": dict(gq.OWNER_SHAPES)})

    def test_a_record_falls_under_its_source_its_prefix_or_the_default(self):
        v = gq.Voices(1, {"p3": "Anjali", "s-exact": "Exact", "*": "Robin"}, owner=True)
        self.assertEqual(v.key_of({"source": "p3-s4f1a2c"}), "p3")
        self.assertEqual(v.key_of({"source": "s-exact"}), "s-exact")
        self.assertEqual(v.key_of({"source": "other"}), "*")
        self.assertIsNone(gq.Voices(1, {"p1": "Teo"}, owner=True).key_of({"source": "zz-1"}))

    def test_a_candidate_keeps_the_ask_and_its_owners_key(self):
        v = gq.Voices(5, {"p3": "Anjali"}, owner=True)
        kept = v.record("long_horizon", ("r7",), self.corpus)
        self.assertEqual(kept, dict(v.ask("long_horizon", ("r7",), self.corpus), key="p3"))


class OwnerMessageTests(unittest.TestCase):
    def setUp(self):
        self.records = [dict(r, source=f"p{1 + i % 2}-s{i}") for i, r in enumerate(tg.toolboxes(6))]
        self.corpus = gq.Corpus(self.records, tg.spec(), False)
        self.voices = gq.Voices(7, {"p1": "Maren", "p2": "Odile"}, owner=True)

    def test_each_record_gets_its_owner_on_the_ask_line(self):
        batch = [(r["id"],) for r in self.records[:4]]
        plain = gq.user_message("direct_recall_user", batch, self.corpus)
        owned = gq.user_message("direct_recall_user", batch, self.corpus, self.voices)
        asks = [json.loads(line[len(gq.ASK_HEAD):]) for line in owned.split("\n") if line.startswith(gq.ASK_HEAD)]
        self.assertEqual([a["owner"] for a in asks], ["Maren", "Odile", "Maren", "Odile"])
        self.assertTrue(all("voice" not in a and "person" not in a for a in asks))
        self.assertEqual(gq.parse_user_message(owned), gq.parse_user_message(plain))

    def test_a_topic_shows_only_its_owners_records(self):
        topic = tuple(r["id"] for r in self.records[:4])  # p1, p2, p1, p2
        owned = gq.user_message("abstention", [topic], self.corpus, self.voices)
        shown = [r["id"] for r in gq.parse_user_message(owned)]
        self.assertEqual(shown, [topic[0], topic[2]])
        self.assertEqual(json.loads(owned.split("\n")[0][len(gq.ASK_HEAD):])["owner"], "Maren")
        mixed = gq.user_message("abstention", [topic], self.corpus, gq.Voices(7, {"p1": "Maren", "p2": "Odile"}))
        self.assertEqual([r["id"] for r in gq.parse_user_message(mixed)], list(topic))  # without --owner, all of it


class OwnerCheckTests(unittest.TestCase):
    def setUp(self):
        self.records = tg.toolboxes(4)
        self.corpus = gq.Corpus(self.records, tg.spec(), False)
        self.unit = (self.records[0]["id"],)
        num, cab = tg.TOOLBOX_RE.search(self.records[0]["text"]).groups()
        self.answer, self.num = f"cabinet {cab}", num

    def check(self, question, owner):
        it = tg.item(question, self.answer, [self.unit[0]])
        return gq.rule_checks("direct_recall_user", it, self.unit, self.corpus, roles=True, owner=owner)

    def test_a_question_that_names_its_owner_is_dropped(self):
        self.assertIsNone(self.check(f"Where do I keep toolbox number {self.num}?", "Maren"))
        self.assertIn("names its owner", self.check(f"Where does Maren keep toolbox number {self.num}?", "Maren"))
        self.assertIn("names its owner", self.check(f"Where is Maren's toolbox number {self.num}?", "Maren"))
        self.assertIsNone(self.check(f"Where does Marengo keep toolbox number {self.num}?", "Maren"))
        self.assertIsNone(self.check(f"Where does Maren keep toolbox number {self.num}?", None))


class OwnerRunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=2,direct_recall_agent=1,abstention=1"

    def people(self, mapping=None):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping or {"*": "Maren"}, f)
        return path

    def read(self, result, *parts):
        with open(os.path.join(result["out"], *parts), encoding="utf-8") as f:
            return f.read()

    def test_an_owner_run_records_the_owner_everywhere(self):
        result = self.gen(tg.toolboxes(24), self.TARGETS, model=tg.FakeModel(tv.person_writer),
                          people=self.people(), prompts_dir=V7, owner=True)
        run = json.loads(self.read(result, "generation", "run.json"))
        self.assertEqual(run["prompts_dir"], "prompts/v7")
        self.assertEqual(run["voices"], {"people_sha256": rc.file_sha256(self.people()), "asker": "owner",
                                         "shapes": dict(gq.OWNER_SHAPES)})
        self.assertEqual(result["manifest"]["voices"], run["voices"])
        calls = [json.loads(line) for line in self.read(result, "generation", "calls.jsonl").splitlines()]
        self.assertTrue(calls and all('"owner": "Maren"' in c["user"] and '"voice"' not in c["user"] for c in calls))
        cands = [json.loads(line) for line in self.read(result, "generation", "candidates.jsonl").splitlines()]
        placed = [c for c in cands if c["unit"]]
        self.assertTrue(placed and all(c["ask"]["owner"] == "Maren" and c["ask"]["key"] == "*"
                                       and c["ask"]["shape"] in dict(gq.OWNER_SHAPES) for c in placed))
        frozen = [c for c in cands if c["question_id"]]
        self.assertEqual(len(frozen), len(json.loads(self.read(result, "questions.json"))))
        self.assertTrue(any(c["category"] == "abstention" for c in frozen))  # no gold record, and still an owner

    def test_people_without_owner_keeps_the_mixed_voices_and_no_ask_on_candidates(self):
        result = self.gen(tg.toolboxes(24), self.TARGETS, model=tg.FakeModel(tv.person_writer),
                          people=self.people(), prompts_dir=V7)
        self.assertEqual(json.loads(self.read(result, "generation", "run.json"))["voices"]["voices"], dict(gq.VOICES))
        cands = [json.loads(line) for line in self.read(result, "generation", "candidates.jsonl").splitlines()]
        self.assertTrue(cands and all("ask" not in c for c in cands))

    def test_owner_needs_people(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", owner=True)
        self.assertIn("--owner needs --people", str(cm.exception))

    def test_a_question_that_names_its_owner_is_dropped_in_a_run(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", model=tg.FakeModel(naming_writer),
                     people=self.people(), prompts_dir=V7, owner=True)
        self.assertIn("no item survived", str(cm.exception))
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(cands and all("names its owner" in (c.get("reason") or "") for c in cands), cands[:2])

    def test_replay_of_an_owner_run_rebuilds_it_byte_for_byte(self):
        corpus = self.corpus(tg.toolboxes(24))
        people = self.people()
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-o"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(tv.person_writer), quiet=True, people=people, prompts_dir=V7,
                                owner=True)
        calls = os.path.join(first["out"], "generation", "calls.jsonl")
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-o"), 7, gq.parse_targets(self.TARGETS),
                                 replay=calls, quiet=True, people=people, prompts_dir=V7, owner=True)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_command_line_takes_owner_with_people(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        base = ["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7", "--targets",
                "direct_recall_user=1", "--estimate", "--prompts", V7]
        self.assertEqual(gq.main(base + ["--people", self.people(), "--owner"]), 0)
        self.assertEqual(gq.main(base + ["--owner"]), 2)


if __name__ == "__main__":
    unittest.main()

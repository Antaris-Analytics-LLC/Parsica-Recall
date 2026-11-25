"""Tests for the question builder's voices (--people) and prompt folder (--prompts): the ASK
lines, the role check, and what a set records. A faked model; no network and no SDK:

    PYTHONPATH=tests python3 -m unittest tests.test_voices
"""

import json
import os
import random
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

V7 = os.path.join(REPO, "prompts", "v7")


def person_writer(category, units):
    """toolbox_writer's items, asked in the person's own voice: no role words."""
    out = []
    for it in tg.toolbox_writer(category, units):
        it = dict(it, question=it["question"].replace("the user", "I").replace("does I", "do I")
                  .replace("did the assistant", "did you").replace("the owner", "I"))
        out.append(it)
    return out


class VoicesTests(unittest.TestCase):
    def test_person_of_reads_the_source_its_prefix_or_the_default(self):
        v = gq.Voices(1, {"p3": "Anjali", "s-exact": "Exact", "*": "Robin"})
        self.assertEqual(v.person_of({"source": "p3-s4f1a2c"}), "Anjali")
        self.assertEqual(v.person_of({"source": "s-exact"}), "Exact")
        self.assertEqual(v.person_of({"source": "other"}), "Robin")
        self.assertEqual(v.person_of({}), "Robin")
        self.assertIsNone(gq.Voices(1, {"p1": "Teo"}).person_of({"source": "zz-1"}))

    def test_ask_is_drawn_from_the_seed_the_category_and_the_unit(self):
        corpus = types.SimpleNamespace(by_id={f"r{i}": {"id": f"r{i}", "source": "p3-s1"} for i in range(400)})
        v = gq.Voices(5, {"p3": "Anjali"})
        a = [v.ask("direct_recall_user", (f"r{i}",), corpus) for i in range(400)]
        self.assertEqual(a, [gq.Voices(5, {"p3": "Anjali"}).ask("direct_recall_user", (f"r{i}",), corpus)
                             for i in range(400)])
        self.assertNotEqual(a, [gq.Voices(6, {"p3": "Anjali"}).ask("direct_recall_user", (f"r{i}",), corpus)
                                for i in range(400)])
        self.assertNotEqual(a, [v.ask("long_horizon", (f"r{i}",), corpus) for i in range(400)])
        self.assertTrue(all(x["person"] == "Anjali" for x in a))
        voices = Counter(x["voice"] for x in a)
        shapes = Counter(x["shape"] for x in a)
        self.assertEqual(set(voices), {name for name, _ in gq.VOICES})
        self.assertEqual(set(shapes), {name for name, _ in gq.SHAPES})
        self.assertGreater(shapes["plain"], shapes["request"])  # 8 to 3 by weight

    def test_no_person_field_without_a_name(self):
        corpus = types.SimpleNamespace(by_id={"r1": {"id": "r1", "source": "zz"}})
        self.assertNotIn("person", gq.Voices(1, {"p1": "Teo"}).ask("abstention", ("r1",), corpus))

    def test_weighted_name_is_the_same_on_every_run(self):
        draws = [gq._weighted_name(random.Random("x"), gq.SHAPES) for _ in range(3)]
        self.assertEqual(len(set(draws)), 1)

    def test_read_people_refuses_what_is_not_a_map_of_names(self):
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")  # never written to
        with self.assertRaises(gq.GenerateError):
            gq.read_people(os.path.join(tmp, "no-such-people-file.json"))
        import tempfile
        d = tempfile.mkdtemp(prefix="recall-people-")
        self.addCleanup(__import__("shutil").rmtree, d, True)
        for bad in ([], {}, {"p1": ""}, {"p1": 3}, "Robin"):
            path = os.path.join(d, "people.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bad, f)
            with self.assertRaises(gq.GenerateError):
                gq.read_people(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"p1": " Teodora "}, f)
        self.assertEqual(gq.read_people(path), {"p1": "Teodora"})


class MessageTests(unittest.TestCase):
    def setUp(self):
        pairs = [r for p in range(3) for r in tg.standup_pair(p)]
        self.records = tg.toolboxes(6) + pairs
        self.corpus = gq.Corpus(self.records, tg.spec(), False)
        self.voices = gq.Voices(7, {"*": "Maren"})

    def test_records_get_an_ask_line_each_and_parse_back_the_same(self):
        batch = [(r["id"],) for r in self.records[:3]]
        plain = gq.user_message("direct_recall_user", batch, self.corpus)
        voiced = gq.user_message("direct_recall_user", batch, self.corpus, self.voices)
        self.assertNotIn(gq.ASK_HEAD, plain)
        lines = voiced.split("\n")
        self.assertEqual(len(lines), 6)
        self.assertTrue(all(lines[i].startswith(gq.ASK_HEAD) for i in (0, 2, 4)))
        ask = json.loads(lines[0][len(gq.ASK_HEAD):])
        self.assertEqual(ask["person"], "Maren")
        self.assertEqual(gq.parse_user_message(voiced), gq.parse_user_message(plain))

    def test_a_pair_gets_its_ask_line_after_the_pair_line(self):
        pairs = [tuple(r["id"] for r in tg.standup_pair(p)) for p in range(2)]
        plain = gq.user_message("knowledge_update", pairs, self.corpus)
        voiced = gq.user_message("knowledge_update", pairs, self.corpus, self.voices)
        lines = voiced.split("\n")
        self.assertEqual(lines[0], "PAIR 1")
        self.assertTrue(lines[1].startswith(gq.ASK_HEAD))
        self.assertTrue(lines[2].startswith("earlier: ") and lines[3].startswith("later: "))
        self.assertEqual(lines[4], "PAIR 2")
        self.assertEqual(gq.parse_user_message(voiced), gq.parse_user_message(plain))

    def test_a_topic_gets_one_ask_line(self):
        topic = tuple(r["id"] for r in self.records[:3])
        voiced = gq.user_message("abstention", [topic], self.corpus, self.voices)
        self.assertEqual(sum(1 for line in voiced.split("\n") if line.startswith(gq.ASK_HEAD)), 1)
        self.assertTrue(voiced.startswith(gq.ASK_HEAD))

    def test_a_retry_shows_the_ask_line_again_and_parses_back(self):
        unit = (self.records[0]["id"],)
        held = tg.item("Where do I keep it?", "cabinet x", [unit[0]])
        conflicts = [("answer", "cabinet x", 2)]
        plain = gq.retry_message("direct_recall_user", held, unit, self.corpus, conflicts)
        voiced = gq.retry_message("direct_recall_user", held, unit, self.corpus, conflicts, self.voices)
        self.assertIn("\n" + gq.ASK_HEAD, voiced)
        self.assertEqual(gq.parse_retry_message(voiced), gq.parse_retry_message(plain))


class RoleCheckTests(unittest.TestCase):
    def setUp(self):
        self.records = tg.toolboxes(4)
        self.corpus = gq.Corpus(self.records, tg.spec(), False)
        self.unit = (self.records[0]["id"],)
        num, cab = tg.TOOLBOX_RE.search(self.records[0]["text"]).groups()
        self.answer = f"cabinet {cab}"
        self.num = num

    def check(self, question, roles):
        it = tg.item(question, self.answer, [self.unit[0]])
        return gq.rule_checks("direct_recall_user", it, self.unit, self.corpus, roles=roles)

    def test_a_role_word_is_dropped_only_with_voices(self):
        q = f"Where does the user keep toolbox number {self.num}?"
        self.assertIsNone(self.check(q, roles=False))
        self.assertIn("names a role", self.check(q, roles=True))
        self.assertIn("names a role", self.check(f"Which cabinet did The Assistant pick for toolbox {self.num}?", True))
        self.assertIsNone(self.check(f"Where do I keep toolbox number {self.num}?", roles=True))


class RunTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=2,direct_recall_agent=1,abstention=1"

    def people(self, mapping=None):
        path = os.path.join(self.tmp, "people.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping or {"*": "Maren"}, f)
        return path

    def read(self, result, *parts):
        with open(os.path.join(result["out"], *parts), encoding="utf-8") as f:
            return f.read()

    def test_a_run_with_voices_records_them_and_sends_ask_lines(self):
        result = self.gen(tg.toolboxes(24), self.TARGETS, model=tg.FakeModel(person_writer),
                          people=self.people(), prompts_dir=V7)
        run = json.loads(self.read(result, "generation", "run.json"))
        self.assertEqual(run["prompts_dir"], "prompts/v7")
        self.assertEqual(run["voices"]["voices"], dict(gq.VOICES))
        self.assertEqual(run["voices"]["shapes"], dict(gq.SHAPES))
        self.assertEqual(run["voices"]["people_sha256"], rc.file_sha256(self.people()))
        m = result["manifest"]
        self.assertEqual(m["voices"], run["voices"])
        self.assertTrue(all(p["file"].startswith("prompts/v7/") for p in m["prompts"].values()))
        v7 = gq.load_prompts(["direct_recall_user"], V7)["direct_recall_user"]
        self.assertEqual(m["prompts"]["direct_recall_user"]["sha256"], v7.sha256)
        self.assertEqual(run["prompts"]["direct_recall_user"], v7.sha256)
        calls = [json.loads(line) for line in self.read(result, "generation", "calls.jsonl").splitlines()]
        self.assertTrue(calls and all(gq.ASK_HEAD + "{" in c["user"] for c in calls))
        self.assertTrue(all('"person": "Maren"' in c["user"] for c in calls))
        self.assertTrue(json.loads(self.read(result, "questions.json")))

    def test_a_run_without_them_records_nothing_new(self):
        result = self.gen(tg.toolboxes(24), self.TARGETS)
        run = json.loads(self.read(result, "generation", "run.json"))
        self.assertNotIn("prompts_dir", run)
        self.assertNotIn("voices", run)
        self.assertNotIn("voices", result["manifest"])
        self.assertTrue(all(p["file"] == f"prompts/{c}.md" for c, p in result["manifest"]["prompts"].items()))
        calls = self.read(result, "generation", "calls.jsonl")
        self.assertNotIn(gq.ASK_HEAD + "{", calls)

    def test_a_question_that_names_a_role_is_dropped_with_voices(self):
        # Every question toolbox_writer writes says "the user", so with voices none survives
        # and there is no set to freeze; the generation record says why each was dropped.
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(tg.toolboxes(24), "direct_recall_user=2", model=tg.FakeModel(), people=self.people(),
                     prompts_dir=V7)
        self.assertIn("no item survived", str(cm.exception))
        path = os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl")
        with open(path, encoding="utf-8") as f:
            cands = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(cands)
        self.assertTrue(all("names a role" in (c.get("reason") or "") for c in cands), cands[:2])
        # The same run without voices freezes them.
        self.assertTrue(json.loads(self.read(self.gen(tg.toolboxes(24), "direct_recall_user=2", out="run2/set-b"),
                                             "questions.json")))

    def test_replay_of_a_voiced_run_rebuilds_it_byte_for_byte(self):
        corpus = self.corpus(tg.toolboxes(24))
        people = self.people()
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-v"), 7, gq.parse_targets(self.TARGETS),
                                model=tg.FakeModel(person_writer), quiet=True, people=people, prompts_dir=V7)
        calls = os.path.join(first["out"], "generation", "calls.jsonl")
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-v"), 7, gq.parse_targets(self.TARGETS),
                                 replay=calls, quiet=True, people=people, prompts_dir=V7)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_command_line_takes_both_flags(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        code = gq.main(["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7",
                        "--targets", "direct_recall_user=1", "--estimate", "--prompts", V7, "--people", self.people()])
        self.assertEqual(code, 0)
        code = gq.main(["--corpus", self.corpus(tg.toolboxes(24)), "--out", out, "--seed", "7",
                        "--targets", "direct_recall_user=1", "--estimate", "--prompts", os.path.join(self.tmp, "nope")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

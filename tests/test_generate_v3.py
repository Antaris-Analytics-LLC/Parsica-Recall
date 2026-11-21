"""Tests for generator v2's later changes (docs/generation.md sections 4 to 6 and 16):
the planted-pairs exporter and --planted-pairs, the rare-name rule for entity_attribution,
the retry that asks for a different fact and the padded-retry drop, --unit-cap, the
estimate from several pilots with its records-shown line, and the time-unit rule of
answer_type. A faked model and temporary folders only; no network and no anthropic
package. One test reads the committed pool's generation record, which is in the repo:

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import re
import sys
import unittest
from collections import Counter
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_generate as tg  # noqa: E402  (the shared helpers: toolboxes, standup pairs, the fake writers)
import test_generate_v2 as v2  # noqa: E402  (RetryModel, PricedModel, leaking and fixed)

gq, lv, rc = tg.gq, tg.lv, tg.rc
REPO = tg.REPO
item, toolbox, toolboxes, rec, spec = tg.item, tg.toolbox, tg.toolboxes, tg.rec, tg.spec

from parsica_recall import export_planted_pairs as ep  # noqa: E402
from parsica_recall import show_questions as sq  # noqa: E402

POOL = os.path.join(REPO, "data", "pool-v1")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def standups(n):
    """n standup pairs as a flat record list: pair k is records 2k and 2k + 1."""
    return [r for k in range(n) for r in tg.standup_pair(k)]


# ---------------------------------------------------------------------------
# A. The exporter
# ---------------------------------------------------------------------------

def needs_the_call_log(test):
    """Pool v1's raw call log (generation/calls-NNNN.jsonl) is not published; its hashes are
    in the pool's manifest. A test that re-derives something from it runs where it is held."""
    if not any(name.startswith("calls-") for name in os.listdir(os.path.join(POOL, "generation"))):
        test.skipTest("pool v1's raw call log is not in this tree")


class ExporterTests(tg.GenTestCase):
    def test_the_committed_pairs_point_into_the_core_and_forward_in_time(self):
        """What the published files alone can show of planted_pairs.jsonl."""
        with open(os.path.join(POOL, "planted_pairs.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(Counter(r["type"] for r in rows),
                         {"version_chain": 50, "preference_change": 39, "correction": 48})
        self.assertTrue(all(list(r) == list(gq.PLANTED_FIELDS) for r in rows))
        pos, ts = {}, {}
        with open(os.path.join(POOL, "corpus.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                r = json.loads(line)
                pos[r["id"]], ts[r["id"]] = i, r["ts"]
        for r in rows:
            self.assertLessEqual(max(pos[r["earlier"]], pos[r["later"]]), 1000)  # the core plants every change
            self.assertLess(ts[r["earlier"]], ts[r["later"]])
        self.assertEqual(len({r["later"] for r in rows}), len(rows))

    def test_the_committed_pool_exports_deterministically_to_the_committed_file(self):
        needs_the_call_log(self)
        rows, counts = ep.planted_pairs(POOL)
        again, _ = ep.planted_pairs(POOL)
        self.assertEqual(rows, again)
        self.assertEqual(counts, {"version_chain": {"items": 50, "pairs": 50, "one_step": 0},
                                  "preference_change": {"items": 39, "pairs": 39, "one_step": 0},
                                  "correction": {"items": 48, "pairs": 48, "one_step": 0}})
        self.assertEqual(len(rows), 137)
        out = os.path.join(self.tmp, "pairs.jsonl")
        data = ep.write_pairs(rows, out)
        with open(os.path.join(POOL, "planted_pairs.jsonl"), "rb") as f:
            self.assertEqual(f.read(), data)  # the committed file is what the exporter writes
        self.assertNotIn(b"\r", data)
        self.assertEqual([list(json.loads(line)) for line in data.decode().split("\n") if line][:1],
                         [list(gq.PLANTED_FIELDS)])
        personas = [int(r["persona"][1:]) for r in rows]
        self.assertEqual(personas, sorted(personas))  # persona order, then label order within a persona

    def test_the_committed_pairs_are_the_last_transition_and_match_the_pools_state(self):
        needs_the_call_log(self)
        items = ep.planted_steps(POOL)
        rows, _ = ep.planted_pairs(POOL)
        by_label = {(r["persona"], r["label"]): r for r in rows}
        with open(os.path.join(POOL, "generation", "states.json"), encoding="utf-8") as f:
            states = json.load(f)
        three = 0
        for persona, st in states.items():
            for label, it in st["items"].items():  # the pool's own record of every planted step
                want = [(v["session"], v["position"]) for v in it["values"]]
                got = [(s["session"], s["position"]) for s in items.get((persona, label), {"steps": []})["steps"]]
                self.assertEqual(got, want, (persona, label))
                if it["type"] in ep.PLANTED_TYPES.values() and len(want) >= 2:
                    steps = items[(persona, label)]["steps"]
                    pair = by_label[(persona, label)]
                    self.assertEqual((pair["earlier"], pair["later"]), (steps[-2]["id"], steps[-1]["id"]))
                    three += len(want) == 3
        self.assertGreater(three, 20)  # the three-step chains, where the last transition is not the first
        pos, ts = {}, {}
        with open(os.path.join(POOL, "corpus.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                r = json.loads(line)
                pos[r["id"]], ts[r["id"]] = i, r["ts"]
        for r in rows:
            self.assertLessEqual(max(pos[r["earlier"]], pos[r["later"]]), 1000)  # the core plants every change
            self.assertLess(ts[r["earlier"]], ts[r["later"]])
        self.assertEqual(len({r["later"] for r in rows}), len(rows))

    def small_pool(self):
        """A pool of one persona and four calls, with the cases the fold rules decide."""
        email = "dana" + "@" + "example.org"
        folder = os.path.join(self.tmp, "pool")
        gen = os.path.join(folder, "generation")

        def call(n, session, must, events, phase="core"):
            user = json.dumps({"persona": {}, "must_include": must, "state": {}})
            resp = json.dumps({"records": [{"kind": "user", "text": f"record {i} of call {n}"} for i in range(5)],
                               "state_update": {"new_people": [], "facts": [], "events": events}})
            return {"n": n, "persona": "p1", "phase": phase, "session": session, "attempt": 1, "user": user,
                    "response": resp}

        def ev(label, record, value="a value"):
            return {"label": label, "record": record, "subject": "the thing", "value": value, "note": ""}

        calls = [
            call(1, "p1-s000001", ["V1, step 1 of 3: state a fact.", "C1, part 1 of 2: err.", "R1, step 1 of 2: x"],
                 [ev("V1", 0), ev("C1", 1), ev("R1", 2), ev("V9", 3), ev("V1", 4, "a second V1 event")]),
            call(2, "p1-s000002", ["V1, step 2 of 3: change it.", "C1, part 2 of 2: fix it.", "R1, step 1 of 2: x"],
                 [ev("V1", 0, f"mail {email}"), ev("C1", 1), ev("R1", 2)]),
            call(3, "p1-s000003", ["V1, step 3 of 3: change it again.", "R1, step 2 of 2: change it."],
                 [ev("V1", 0), ev("R1", "1")]),
            call(4, "p1-s000004", [], [ev("V1", 0)], phase="tail"),
        ]
        write_jsonl(os.path.join(gen, "calls-0001.jsonl"), calls)
        kept = [(1, 0), (1, 1), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0)]  # call 1's record 2 dropped
        write_jsonl(os.path.join(gen, "records.jsonl"),
                    [{"line": i, "id": f"id-{n}-{p}", "persona": "p1", "phase": "core", "session": f"p1-s00000{n}",
                      "position": p, "call": n} for i, (n, p) in enumerate(kept, 1)])
        return folder

    def test_the_fold_rules_on_a_small_pool(self):
        folder = self.small_pool()
        items = ep.planted_steps(folder)
        self.assertEqual({k: [s["id"] for s in v["steps"]] for k, v in items.items()}, {
            ("p1", "V1"): ["id-1-0", "id-3-0"],  # call 2's step held an email, so it never entered the state
            ("p1", "C1"): ["id-1-1", "id-2-1"],
            ("p1", "R1"): ["id-2-2"],  # call 1's record was dropped, and call 3 named no record position
        })  # V9 was never asked for, a second V1 event in call 1 is ignored, and the tail call asked for nothing
        rows, counts = ep.planted_pairs(folder)
        self.assertEqual(rows, [
            {"type": "version_chain", "label": "V1", "persona": "p1", "earlier": "id-1-0", "later": "id-3-0"},
            {"type": "correction", "label": "C1", "persona": "p1", "earlier": "id-1-1", "later": "id-2-1"}])
        self.assertEqual(counts["preference_change"], {"items": 1, "pairs": 0, "one_step": 1})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(ep.main([folder]), 0)
            self.assertEqual(ep.main([folder]), 0)
            self.assertEqual(ep.main([os.path.join(self.tmp, "nowhere")]), 2)
        self.assertIn("Wrote 2 planted pairs", out.getvalue())
        self.assertIn("(written)", out.getvalue())
        self.assertIn("(unchanged)", out.getvalue())
        self.assertIn("1 with one step, no transition", out.getvalue())
        self.assertIn("not a pool", err.getvalue())
        self.assertEqual(gq.read_planted_pairs(os.path.join(folder, "planted_pairs.jsonl")), rows)

    def test_the_file_is_read_strictly(self):
        path = os.path.join(self.tmp, "pairs.jsonl")
        good = {"type": "correction", "label": "C1", "persona": "p1", "earlier": "aaaa", "later": "bbbb"}
        for bad, why in [(dict(good, type="thread"), "unknown type"), ({k: v for k, v in good.items() if k != "label"}, "fields"),
                         (dict(good, later=""), "non-empty"), (dict(good, extra=1), "fields")]:
            write_jsonl(path, [good, bad])
            with self.assertRaises(gq.GenerateError) as cm:
                gq.read_planted_pairs(path)
            self.assertIn(why, str(cm.exception))
            self.assertIn("line 2", str(cm.exception))
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json\n")
        with self.assertRaises(gq.GenerateError):
            gq.read_planted_pairs(path)
        open(path, "w").close()
        with self.assertRaises(gq.GenerateError) as cm:
            gq.read_planted_pairs(path)
        self.assertIn("holds no pairs", str(cm.exception))
        with self.assertRaises(gq.GenerateError):
            gq.read_planted_pairs(os.path.join(self.tmp, "missing.jsonl"))


# ---------------------------------------------------------------------------
# B. --planted-pairs in the generator
# ---------------------------------------------------------------------------

class PlantedPairsTests(tg.GenTestCase):
    """12 standup pairs, the first 4 in a core of 8 records; pairs 0 and 1 are planted
    version chains, 2 and 3 corrections, 4 (in the tail) a preference change."""

    def setUp(self):
        super().setUp()
        self.recs = standups(12)
        self.pairs = [(self.recs[2 * k]["id"], self.recs[2 * k + 1]["id"]) for k in range(12)]

    def plant(self, rows=None, name="pairs.jsonl"):
        def row(k, typ, label):
            return {"type": typ, "label": label, "persona": "p1", "earlier": self.pairs[k][0], "later": self.pairs[k][1]}

        rows = rows if rows is not None else [row(0, "version_chain", "V1"), row(1, "version_chain", "V2"),
                                              row(2, "correction", "C1"), row(3, "correction", "C2"),
                                              row(4, "preference_change", "R1")]
        path = os.path.join(self.tmp, name)
        write_jsonl(path, rows)
        return path

    def plans(self, targets, core=8, **kw):
        corpus = gq.Corpus(self.recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        rows = gq.read_planted_pairs(self.plant())
        planted, info = gq.route_planted_pairs(rows, corpus, reps, None)
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), gq.parse_targets(targets), 7, "abc", core=core,
                               planted=planted, **kw)
        return corpus, plans, info

    def test_planted_units_are_routed_by_type_and_the_core_takes_nothing_else(self):
        targets = "knowledge_update=4,correction_handling=4,recency_preference=4"
        corpus, plans, info = self.plans(targets)
        self.assertEqual(info["used"], {"knowledge_update": 2, "correction_handling": 2, "recency_preference": 1})
        ku, ch, rp = (plans[c] for c in rc.VERSION_CATEGORIES)
        self.assertEqual((sorted(ku.units[:ku.n_core]), ku.n_core), (sorted(self.pairs[:2]), 2))
        self.assertEqual((sorted(ch.units), ch.n_core), (sorted(self.pairs[2:4]), 2))  # the record alone, no tail
        self.assertEqual((rp.units, rp.n_core), ([self.pairs[4]], 0))  # its one pair sits in the tail
        tail = ku.units[ku.n_core:]  # knowledge_update goes on with similarity pairs past the core
        self.assertTrue(tail)
        self.assertLessEqual(set(tail), set(self.pairs[5:]))  # never a planted pair, never the core's
        self.assertEqual((ku.planted_stages, ch.planted_stages, rp.planted_stages),
                         ({"core"}, {"core"}, {"tail"}))
        plain = gq.build_plans(corpus, corpus.eligibility(6)[0], corpus.age_bands(), gq.parse_targets(targets), 7,
                               "abc", core=8)
        self.assertEqual(plain["correction_handling"].planted_stages, frozenset())
        self.assertTrue(plain["correction_handling"].units)  # without the record: similarity pairs, dealt in turn

    def test_without_a_core_the_one_stage_takes_planted_pairs_first(self):
        corpus, plans, _ = self.plans("knowledge_update=4,correction_handling=4", core=None)
        ku = plans["knowledge_update"]
        self.assertEqual(sorted(ku.units[:2]), sorted(self.pairs[:2]))
        self.assertTrue(ku.units[2:])
        self.assertEqual(ku.planted_stages, frozenset())  # mixed: planted, then similarity
        self.assertEqual(sorted(plans["correction_handling"].units), sorted(self.pairs[2:4]))
        self.assertEqual(plans["correction_handling"].planted_stages, {"all"})

    def test_pairs_the_corpus_cannot_use_are_left_out_with_a_reason(self):
        corpus = gq.Corpus(toolboxes(2) + self.recs, spec(), False)  # a window of 20: pairs 0 to 8 sit in it
        reps, _ = corpus.eligibility(6, 20)
        box = toolboxes(2)[0]["id"]  # eligible, in the window, and without ts
        e, l = self.pairs[0]
        rows = [{"type": "correction", "label": "C1", "persona": "p1", "earlier": e, "later": l},
                {"type": "correction", "label": "C2", "persona": "p1", "earlier": e, "later": l},
                {"type": "correction", "label": "C3", "persona": "p1", "earlier": l, "later": e},
                {"type": "correction", "label": "C4", "persona": "p1", "earlier": e, "later": e},
                {"type": "correction", "label": "C5", "persona": "p1", "earlier": e, "later": "0123456789abcdef"},
                {"type": "correction", "label": "C6", "persona": "p1", "earlier": self.pairs[11][0], "later": self.pairs[11][1]},
                {"type": "correction", "label": "C7", "persona": "p1", "earlier": box, "later": l},
                {"type": "version_chain", "label": "V1", "persona": "p1", "earlier": self.pairs[1][0], "later": self.pairs[1][1]}]
        planted, info = gq.route_planted_pairs(rows, corpus, reps, 20)
        self.assertEqual(planted["correction_handling"], [(e, l)])
        self.assertEqual(planted["knowledge_update"], [self.pairs[1]])
        self.assertEqual(info["read"], {"correction": 7, "version_chain": 1})
        self.assertEqual(info["skipped"], {"a pair or later record an earlier line gave": 1,
                                           "the later record's ts is not after the earlier's": 1,
                                           "the same memory twice": 1, "a record not in the corpus": 1,
                                           "a record past the gold window": 1, "a record without ts": 1})

    def test_a_run_freezes_planted_pairs_and_records_the_file(self):
        path = self.plant()
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(self.recs, "knowledge_update=3,correction_handling=3,recency_preference=3",
                           model=tg.FakeModel(tg.pair_writer), core=8, planted_pairs=path)
        qs = {c: [tuple(q["gold_ids"]) for q in res["questions"] if q["category"] == c] for c in rc.VERSION_CATEGORIES}
        self.assertEqual(sorted(qs["correction_handling"]), sorted(self.pairs[2:4]))
        self.assertEqual(qs["recency_preference"], [self.pairs[4]])
        self.assertEqual(len(qs["knowledge_update"]), 3)  # two planted in the core, then a similarity pair
        for q in res["questions"]:
            self.assertEqual(q["must_rank_first"], q["gold_ids"][1])
        per = res["manifest"]["generation"]["per_category"]
        self.assertEqual((per["correction_handling"]["frozen_core"], per["correction_handling"]["stopped"]),
                         (2, "ran out of units"))
        p = res["manifest"]["sampler"]["planted_pairs"]
        self.assertEqual((p["file"], p["sha256"]), ("pairs.jsonl", rc.file_sha256(path)))
        self.assertEqual(p["read"], {"correction": 2, "preference_change": 1, "version_chain": 2})
        self.assertEqual(p["used"], {"knowledge_update": 2, "correction_handling": 2, "recency_preference": 1})
        self.assertEqual(p["skipped"], {})
        self.assertIn("planted pairs only", p["rule"])
        with open(os.path.join(res["out"], "generation", "run.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["planted_pairs"], rc.file_sha256(path))
        self.assertTrue(res["check"]["ok"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(sq.main(["summary", res["out"]]), 0)
        self.assertIn("Planted pairs (pairs.jsonl): used 2 for knowledge_update, 2 for correction_handling, "
                      "1 for recency_preference", buf.getvalue())

    def test_the_core_quota_holds_for_planted_pairs(self):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(self.recs, "correction_handling=3", model=tg.FakeModel(tg.pair_writer), core=8,
                           core_quota=1, planted_pairs=self.plant())
        n = res["manifest"]["generation"]["per_category"]["correction_handling"]
        self.assertEqual((n["frozen"], n["frozen_core"], n["surplus"]), (1, 1, 1))  # the other core pair is surplus
        self.assertEqual([c["reason"] for c in res["candidates"] if c["fate"] == "surplus"], ["the core quota is met"])

    def test_a_resume_keeps_the_file_and_the_estimate_names_it(self):
        path = self.plant()
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.CostStop):
            self.gen(self.recs, "knowledge_update=4", model=v2.PricedModel(tg.pair_writer), core=8,
                     planted_pairs=path, max_cost=0.1)  # the core's two planted pairs, then a second call
        other = self.plant(name="other.jsonl", rows=[{"type": "correction", "label": "C1", "persona": "p1",
                                                      "earlier": self.pairs[5][0], "later": self.pairs[5][1]}])
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError) as cm:
            self.gen(self.recs, "knowledge_update=4", model=v2.PricedModel(tg.pair_writer), core=8,
                     planted_pairs=other, resume=True)
        self.assertIn("other settings (planted_pairs)", str(cm.exception))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gq.main(["--corpus", os.path.join(self.tmp, "corpus.jsonl"), "--out",
                            os.path.join(self.tmp, "est", "set-p"), "--seed", "7", "--targets",
                            "correction_handling=4", "--core", "8", "--estimate", "--planted-pairs", path])
        self.assertEqual(code, 0)
        self.assertIn("Planted pairs pairs.jsonl: 2 for knowledge_update, 2 for correction_handling, "
                      "1 for recency_preference", buf.getvalue())


# ---------------------------------------------------------------------------
# C. Rare names for entity_attribution
# ---------------------------------------------------------------------------

def kroger(i):
    return rec(f"Bought eggs at Kroger again with the usual coupon on trip number {i} of the month.")


class RareNameTests(tg.GenTestCase):
    def test_record_names(self):
        self.assertEqual(gq.record_names("Dana Whitfield fixed the valve. Then Marcus called Dr. Samir Haddad "
                                         "about Rillito Valley Vet in Tucson."),
                         ["dana whitfield", "marcus", "samir haddad", "rillito valley vet", "tucson"])
        self.assertEqual(gq.record_names("Picked up a cast iron skillet at Lodge. The SMS went to Saguaro Paws "
                                         "at 8am on Monday; I asked Teo, and O'Brien agreed."),
                         ["lodge", "sms", "saguaro paws", "monday", "teo", "obrien"])
        self.assertEqual(gq.record_names("Ran the test.\nResults were fine, said Kim."), ["kim"])
        self.assertEqual(gq.record_names("no capitals here at all"), [])
        # A possessive counts as its stem, with either apostrophe; O'Brien keeps its letters.
        self.assertEqual(gq.record_names("The reminders from Kennelbook's queue reached Teo’s Compaq Presario."),
                         ["kennelbook", "teo compaq presario"])
        self.assertEqual(gq.record_names("Sent the file to O'Brien's desk at Kennelbook."), ["obrien", "kennelbook"])

    def test_a_possessive_is_rare_only_when_its_stem_is(self):
        owner = rec("The reminders went out from Kennelbook's queue an hour early on the first morning.")
        recs = [owner] + [rec(f"Kennelbook sent the clinic reminder batch number {i} on time this week.") for i in range(3)]
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        self.assertEqual(gq.rare_name_records(corpus, reps), {})  # kennelbooks would be rare; kennelbook is not

    def test_only_a_name_found_in_no_other_record_counts(self):
        dana = rec("Dana Whitfield fixed the leaking valve on the third floor the night before the inspection.")
        plain = rec("Picked up a cast iron skillet at the flea market and it needs seasoning before use.")
        a = rec(tg.QUARTERLY.format(n=42, tail=""))
        copy = rec(tg.QUARTERLY.format(n=42, tail=" overall"))  # a near copy: Denver sits in its own cluster only
        shared = [rec(f"Talked to Quellmark at the depot about the shelf labels on day {i} of the move.") for i in (1, 2)]
        recs = [dana, plain, a, copy] + shared + [kroger(i) for i in range(6)]
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        self.assertEqual(gq.rare_name_records(corpus, reps), {dana["id"]: ["dana whitfield"], a["id"]: ["denver"]})
        one_other = gq.rare_name_records(corpus, reps, 1)
        self.assertEqual(set(one_other), {dana["id"], a["id"]} | {r["id"] for r in shared})
        self.assertEqual(one_other[shared[0]["id"]], ["quellmark"])
        self.assertEqual(gq.RARE_NAME_MAX_OTHERS, 0)

    def test_entity_attribution_draws_only_from_records_with_a_rare_name(self):
        dana = rec("Dana Whitfield fixed the leaking valve on the third floor the night before the inspection.")
        recs = [dana] + [kroger(i) for i in range(6)] + toolboxes(6)

        def writer(category, units):
            return [item("Who fixed the leaking valve on the third floor before the inspection?", "Dana Whitfield",
                         [r["id"]]) for r in units if r["id"] == dana["id"]]

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "entity_attribution=2", model=tg.FakeModel(writer))
        self.assertEqual(self.shown(res), {dana["id"]})
        n = res["manifest"]["generation"]["per_category"]["entity_attribution"]
        self.assertEqual((n["units_shown"], n["frozen"], n["stopped"]), (1, 1, "ran out of units"))
        ea = res["manifest"]["sampler"]["entity_attribution"]
        self.assertEqual((ea["max_other_records"], ea["records_with_a_rare_name"], ea["in_the_core"]), (0, 1, None))
        self.assertIn("rare name", ea["rule"])
        self.assertEqual(res["questions"][0]["answer_type"], "entity")
        self.assertTrue(res["check"]["ok"])
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), gq.parse_targets("entity_attribution=2"), 7, "abc",
                               core=4)
        self.assertEqual(plans["entity_attribution"].units, [(dana["id"],)])
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "entity_attribution=1", out="none/set-e")
        self.assertIn("no eligible record holds a rare name", str(cm.exception))


# ---------------------------------------------------------------------------
# D. The retry: a different fact, and the padded-retry drop
# ---------------------------------------------------------------------------

def padded_or_fixed(category, held, why, units):
    """A retry that pads the held answer for even toolbox numbers and picks another fact for odd ones."""
    num, cab = tg.TOOLBOX_RE.search(units[0]["text"]).groups()
    if int(num) % 2 == 0:
        return [item(held["question"], f"{held['answer']} {cab}", held["gold_ids"])]  # "stays in cabinet xqaa"
    return [item(held["question"], f"cabinet {cab}", held["gold_ids"])]


class PaddedRetryTests(tg.GenTestCase):
    def test_padded_retry(self):
        held = {"answer": "stays in cabinet"}
        self.assertTrue(gq.padded_retry({"answer": "stays in cabinet xqaa"}, held))
        self.assertTrue(gq.padded_retry({"answer": "It stays in cabinet, I checked"}, held))
        self.assertFalse(gq.padded_retry({"answer": "stays in cabinet"}, held))  # unchanged: the lint holds it again
        self.assertFalse(gq.padded_retry({"answer": "cabinet xqaa"}, held))
        self.assertFalse(gq.padded_retry({"answer": "stays in cabinets"}, held))  # whole tokens only
        self.assertFalse(gq.padded_retry({"answer": "anything"}, {"answer": ""}))

    def test_a_padded_retry_is_dropped_and_counted(self):
        model = v2.RetryModel(v2.leaking, padded_or_fixed)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(12), "direct_recall_user=2", model=model)
        retried = [c for c in res["candidates"] if c["retry_of"]]
        padded = [c for c in retried if c["reason"] and c["reason"].startswith(gq.PADDED_RETRY)]
        self.assertTrue(padded)
        for c in padded:
            self.assertEqual(c["fate"], "dropped_by_checks")
            self.assertEqual(c["reason"], "padded retry: the answer only adds words around the held answer")
            self.assertTrue(c["returned"]["answer"].startswith("stays in cabinet "))
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["frozen"], n["padded_retries"], n["frozen_after_retry"]), (2, len(padded), 2))
        self.assertEqual(n["retry_calls"], len(retried))
        self.assertEqual(res["manifest"]["generation"]["retry"]["padded_retries"], len(padded))
        self.assertIn("padded retry", res["manifest"]["generation"]["retry"]["rule"])
        self.assertEqual([q["answer"].startswith("cabinet ") for q in res["questions"]], [True, True])

    def test_every_retry_section_asks_for_a_different_fact(self):
        prompts = gq.load_prompts(rc.CATEGORIES)
        for cat, p in prompts.items():
            if cat == "abstention":
                self.assertNotIn("A SECOND TRY", p.text)  # abstention items are screened, never held, never retried
                continue
            self.assertIn("A SECOND TRY", p.text, cat)
            self.assertIn("Never a longer quote of the held answer: adding the words around it does not make it "
                          "pick out the record.", p.text, cat)
            self.assertNotIn("a longer or more distinctive span", p.text, cat)
            self.assertNotIn("or another fact it states", p.text, cat)
            if cat in rc.VERSION_CATEGORIES:
                self.assertIn("The answer must still be the", p.text, cat)
                self.assertIn("another form that does not contain the held answer", p.text, cat)
                self.assertIn("When the new value cannot stand on its own as an answer no other record holds, "
                              'return {"items": []}', p.text, cat)
                # the retry never invites what the padded-retry check drops
                self.assertNotIn("unit or qualifier", p.text, cat)
                self.assertNotIn("fuller name", p.text, cat)
            elif cat == "cross_session":
                self.assertIn("a different, more distinctive fact of the same pair", p.text)
            elif cat in ("implicit_preference", "vocabulary_gap"):
                self.assertIn("a different, more distinctive", p.text, cat)
            else:
                self.assertIn("about a different, more distinctive fact of the same record", p.text, cat)
            self.assertNotIn(chr(0x2014), p.text)
        specific = [cat for cat, p in prompts.items() if "- Make it specific enough" in p.text]
        self.assertEqual(len(specific), 9)
        for cat in specific:
            line = next(ln for ln in prompts[cat].text.split("\n") if ln.startswith("- Make it specific enough"))
            self.assertTrue(line.endswith("Pick a distinctive fact rather than a longer span: never lengthen an "
                                          "answer with the words around it to make it unique."), cat)


# ---------------------------------------------------------------------------
# E. --unit-cap, the estimate from several pilots, the records-shown line, the time unit
# ---------------------------------------------------------------------------

class UnitCapTests(tg.GenTestCase):
    def test_parse(self):
        self.assertEqual(gq.parse_unit_caps("knowledge_update=1200, implicit_preference=900"),
                         {"knowledge_update": 1200, "implicit_preference": 900})
        self.assertEqual(gq.parse_unit_caps(None), {})
        self.assertEqual(gq.parse_unit_caps(" "), {})
        for bad in ("nonsense=3", "knowledge_update=0", "knowledge_update", "knowledge_update=x", "=5"):
            with self.assertRaises(gq.GenerateError):
                gq.parse_unit_caps(bad)
        corpus = self.corpus(toolboxes(4))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gq.main(["--corpus", corpus, "--out", os.path.join(self.tmp, "o"), "--seed", "1",
                                      "--targets", "2", "--unit-cap", "nonsense=3", "--estimate"]), 2)

    def test_a_cap_replaces_the_category_s_own(self):
        recs = toolboxes(40)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=2,paraphrase_recall=2")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", unit_caps={"direct_recall_user": 20})
        self.assertEqual((plans["direct_recall_user"].cap, len(plans["direct_recall_user"].units)), (20, 20))
        self.assertEqual((plans["paraphrase_recall"].cap, len(plans["paraphrase_recall"].units)), (12, 12))
        est = gq.estimate(plans, targets, corpus, gq.load_prompts(list(targets)), "claude-opus-5-5")
        self.assertEqual(est["categories"]["direct_recall_user"]["worst_units"], 20)
        calls = []

        def writer(category, units):  # nothing usable, so the category runs to its cap
            calls.append(len(units))
            return []

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError):
            self.gen(recs, "direct_recall_user=2", model=tg.FakeModel(writer), unit_caps={"direct_recall_user": 20})
        self.assertEqual(sum(calls), 20)

    def test_the_run_records_the_caps_and_a_resume_keeps_them(self):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(40), "direct_recall_user=2", unit_caps={"direct_recall_user": 20})
        self.assertEqual(res["manifest"]["sampler"]["unit_caps"]["direct_recall_user"], 20)
        self.assertIn("--unit-cap", res["manifest"]["sampler"]["unit_caps"]["rule"])
        with open(os.path.join(res["out"], "generation", "run.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["unit_caps"], {"direct_recall_user": 20})
        plain = self.gen(toolboxes(40), "direct_recall_user=2", out="plain/set-u")
        self.assertNotIn("unit_caps", plain["manifest"]["sampler"])
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.CostStop):
            self.gen(toolboxes(80), "direct_recall_user=4", model=v2.PricedModel(), max_cost=0.1,
                     unit_caps={"direct_recall_user": 20}, out="r/set-u")
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(80), "direct_recall_user=4", model=v2.PricedModel(), resume=True,
                     unit_caps={"direct_recall_user": 24}, out="r/set-u")
        self.assertIn("other settings (unit_caps)", str(cm.exception))
        with self.assertRaises(gq.GenerateError):
            self.gen(toolboxes(12), "direct_recall_user=1", unit_caps={"nonsense": 3}, out="bad/set-u")


class EstimateV3Tests(tg.GenTestCase):
    def test_a_later_pilot_replaces_an_earlier_one_per_category(self):
        class Bigger(v2.PricedModel):
            def call(self, category, prompt, user, request_sha):
                reply = super().call(category, prompt, user, request_sha)
                reply.output_tokens = 20_000
                return reply

        first = self.gen(toolboxes(30), "direct_recall_user=2,vocabulary_gap=2", model=v2.PricedModel(), out="a/pilot")
        second = self.gen(toolboxes(30), "direct_recall_user=2", model=Bigger(), out="b/pilot")
        cal = gq.read_calibrations([first["out"], second["out"]])
        self.assertEqual(cal["from"], [first["out"], second["out"]])
        self.assertEqual(cal["categories"]["direct_recall_user"]["output_per_call"], 20_000)  # the later pilot's
        self.assertEqual(cal["categories"]["vocabulary_gap"]["output_per_call"], 10_000)  # only the first measured it
        self.assertEqual(cal["chars_per_token"], gq.read_calibration(second["out"])["chars_per_token"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gq.main(["--corpus", os.path.join(self.tmp, "corpus.jsonl"), "--out", os.path.join(self.tmp, "f", "set"),
                            "--seed", "7", "--targets", "direct_recall_user=10,vocabulary_gap=4", "--estimate",
                            "--estimate-from", first["out"], second["out"]])
        self.assertEqual(code, 0)
        self.assertIn(f"measured by {first['out']}, {second['out']}", buf.getvalue())
        self.assertEqual(gq.read_calibrations([first["out"]])["from"], [first["out"]])

    def test_the_estimate_prints_the_records_shown_against_the_sample_cap(self):
        recs = toolboxes(40)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=8")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc")
        prompts = gq.load_prompts(["direct_recall_user"])
        est = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5", sample_cap=10)
        self.assertEqual(est["records_shown"], {"expected": 16, "worst": 32, "sample_cap": 10})  # 8 at a yield of 0.5

        def printed(sample_cap):
            buf = io.StringIO()
            gq.print_estimate(gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5", sample_cap=sample_cap),
                              lambda *a: print(*a, file=buf))
            return buf.getvalue()

        self.assertIn("records shown to the model: about 16 expected and 32 at the worst case, against the sample "
                      "cap of 10 (half the eligible memories); the sample cap would stop the run before the unit "
                      "caps", printed(10))
        self.assertIn("against the sample cap of 20 (half the eligible memories); the worst case passes the sample "
                      "cap, which then ends the categories still running", printed(20))
        self.assertTrue(re.search(r"sample cap of 40 \(half the eligible memories\)\n", printed(40)))
        self.assertNotIn("records shown", printed(None))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gq.main(["--corpus", self.corpus(recs), "--out", os.path.join(self.tmp, "e", "set"), "--seed", "7",
                            "--targets", "direct_recall_user=8", "--estimate"])
        self.assertEqual(code, 0)
        self.assertIn("against the sample cap of 20", buf.getvalue())  # half of 40 eligible memories

    def test_a_planted_stage_is_guessed_at_the_planted_yield(self):
        recs = standups(12)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        pairs = [(recs[2 * k]["id"], recs[2 * k + 1]["id"]) for k in range(4)]
        planted = {"knowledge_update": pairs[:2], "correction_handling": pairs[2:], "recency_preference": []}
        targets = gq.parse_targets("knowledge_update=4,correction_handling=4")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", core=8, planted=planted)
        est = gq.estimate(plans, targets, corpus, gq.load_prompts(list(targets)), "claude-opus-5-5")
        ku = est["categories"]["knowledge_update"]["stages"]
        self.assertEqual((ku["core"]["yield"], ku["tail"]["yield"]), (gq.EST_YIELD_PLANTED, gq.EST_YIELD["knowledge_update"]))
        self.assertEqual(est["categories"]["correction_handling"]["stages"]["core"]["yield"], gq.EST_YIELD_PLANTED)

    def test_a_time_unit_needs_a_number_before_it(self):
        s = spec()
        self.assertEqual(gq.answer_type("direct_recall_user", "second transect of 40 bolts", s), "phrase")
        self.assertEqual(gq.answer_type("direct_recall_user", "40 second hold", s), "duration")
        self.assertEqual(gq.answer_type("direct_recall_user", "30-minute walk", s), "duration")
        self.assertEqual(gq.answer_type("direct_recall_user", "4 months", s), "duration")
        units = s["constants"]["unit_terms"]
        self.assertFalse(gq.has_unit("second transect 40", units))
        self.assertTrue(gq.has_unit("3 weeks", units))
        self.assertFalse(gq.has_unit("week 3", units))


if __name__ == "__main__":
    unittest.main()

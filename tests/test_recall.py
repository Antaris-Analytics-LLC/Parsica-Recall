"""Unit tests for Parsica Recall. Standard library only:

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from parsica_recall import compare_results as cr  # noqa: E402
from parsica_recall import lint_validator as lv  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
from parsica_recall.adapters import BaseMemoryAdapter  # noqa: E402


def make_record(text, kind="note"):
    return {"id": rc.record_id(text), "text": text, "kind": kind}


def make_question(qid, question, answer, gold_texts, category="direct_recall_user", alts=None,
                  must_rank_first_text=None, answer_type="phrase"):
    gold = [rc.record_id(t) for t in gold_texts]
    return {
        "id": qid, "question": question, "category": category, "difficulty": "easy",
        "answer_type": answer_type, "answer": answer, "acceptable_answers": alts or [],
        "gold_ids": gold,
        "must_rank_first": ([rc.record_id(t) for t in must_rank_first_text] if isinstance(must_rank_first_text, list)
                            else rc.record_id(must_rank_first_text) if must_rank_first_text else None),
        "gold_hashes": {rc.record_id(t): rc.text_hash(t) for t in gold_texts},
        "review_status": "publish", "review_notes": "",
    }


class NormalizeTests(unittest.TestCase):
    def test_lowercase_strip_collapse(self):
        self.assertEqual(rc.normalize("  The  J-channel, rot/photodegrade!  "), "the jchannel rotphotodegrade")

    def test_unicode_forms_get_one_id(self):
        composed = "café au lait"
        decomposed = "café au lait"
        self.assertEqual(rc.record_id(composed), rc.record_id(decomposed))
        self.assertEqual(rc.normalize("１２"), "12")  # full-width digits

    def test_other_scripts_keep_their_words(self):
        self.assertEqual(rc.normalize("你好 世界"), "你好 世界")

    def test_record_id_shape(self):
        rid = rc.record_id("The author lost 22 pounds in 4 months.")
        self.assertEqual(len(rid), 16)
        self.assertEqual(rid, rc.record_id("the author lost 22 pounds in 4 months"))

    def test_whole_token_match(self):
        hay = rc.normalize("Use print() and step into the loop")
        self.assertFalse(rc.contains_phrase(hay, rc.normalize("int")))
        self.assertTrue(rc.contains_phrase(hay, rc.normalize("step into")))

    def test_edge_function_words(self):
        self.assertEqual(rc.edge_function_word("button on the"), "the")
        self.assertEqual(rc.edge_function_word("to measure"), "to")
        self.assertIsNone(rc.edge_function_word("help you"))
        self.assertIsNone(rc.edge_function_word("500 am"))
        self.assertIsNone(rc.edge_function_word("may 2026"))


class BandAndChanceTests(unittest.TestCase):
    def test_band_edges(self):
        self.assertEqual(rc.band_for(99)["band"], "none")
        self.assertEqual(rc.band_for(100)["band"], "S")
        self.assertEqual(rc.band_for(999)["window"], 10)
        self.assertEqual(rc.band_for(1000)["headline_k"], 10)
        self.assertEqual(rc.band_for(49999)["band"], "L")
        self.assertEqual(rc.band_for(50000)["window"], 25)

    def test_single_gold_chance_is_k_over_n(self):
        curve = rc.mean_chance_curve(200, [(1, "any")])
        for k in (1, 5, 10, 50):
            self.assertAlmostEqual(rc.chance_at(curve, k), k / 200, places=12)

    def test_all_rule_chance(self):
        curve = rc.mean_chance_curve(100, [(2, "all")])
        self.assertAlmostEqual(rc.chance_at(curve, 10), (10 * 9) / (100 * 99), places=12)
        self.assertEqual(rc.chance_at(curve, 1), 0.0)

    def test_mean_over_questions(self):
        curve = rc.mean_chance_curve(100, [(1, "any"), (0, "any")])
        self.assertAlmostEqual(rc.chance_at(curve, 10), 0.05, places=12)

    def test_ceiling_and_skill(self):
        curve = rc.mean_chance_curve(100, [(1, "any")])
        self.assertEqual(rc.k_ceiling(curve, 0.05), 5)
        self.assertAlmostEqual(rc.skill(0.5, 0.0), 0.5)
        self.assertLess(rc.skill(0.0, 0.1), 0.0)


class WindowAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.texts = ["alpha record about apples and orchards", "beta record about bread and baking",
                      "gamma record about cars and engines", "delta record about apples in winter"]
        self.pool = rb.Pool([make_record(t) for t in self.texts])

    def test_duplicates_burn_their_slot(self):
        a, b = self.texts[0], self.texts[1]
        hits = [(rc.record_id(a), a, 2.0), (rc.record_id(a), a, 1.9), (rc.record_id(b), b, 1.0)]
        entries, burned = rb.build_window(hits, 3)
        self.assertEqual(burned, 1)
        self.assertEqual([e["rank"] for e in entries], [0, 2])

    def test_window_truncates(self):
        entries, _ = rb.build_window(self.texts, 2)
        self.assertEqual(len(entries), 2)

    def test_bare_text_hits_get_record_ids(self):
        self.assertEqual(rb.to_hit(self.texts[2])[0], rc.record_id(self.texts[2]))

    def test_gold_ranks_and_version(self):
        q = make_question("q1", "what about apples?", "apples in winter", [self.texts[0], self.texts[3]],
                          category="knowledge_update", must_rank_first_text=self.texts[3])
        const = rb.question_constants(q, self.pool)
        entries, _ = rb.build_window([self.texts[2], self.texts[0], self.texts[3]], 10)
        rec = rb.score_question(q, const, entries, self.pool)
        self.assertEqual(rec["gold_rank"], 1)
        self.assertEqual(rec["gold_all_rank"], 2)
        self.assertFalse(rec["version_ok"])
        self.assertEqual(rec["exact_rank"], 2)
        self.assertEqual(rec["first_nongold_rank"], 0)

    def test_version_list_any_current_copy_first_scores(self):
        stale, copy_a, copy_b = self.texts[0], self.texts[3], self.texts[1]
        q = make_question("q3", "what about apples?", "apples", [stale, copy_a, copy_b],
                          category="knowledge_update", must_rank_first_text=[copy_a, copy_b])
        const = rb.question_constants(q, self.pool)
        for window, expected in (([copy_b, stale], True), ([copy_a], True), ([self.texts[2], stale, copy_a], False)):
            entries, _ = rb.build_window(window, 10)
            self.assertEqual(rb.score_question(q, const, entries, self.pool)["version_ok"], expected, window)

    def test_what_counts_as_a_ts(self):
        for value in ("2026-03-10T00:00:00Z", 1760000000, 1.5):
            self.assertTrue(rc.has_ts(value), value)
        for value in (None, "", "  ", "null", "None", "NaN", "0", 0, 0.0, -5, float("nan"), True, False, [], {}):
            self.assertFalse(rc.has_ts(value), value)

    def test_query_variants(self):
        q = "What is the name of the dog's owner, again?"
        self.assertEqual(rc.query_variant(q, "casefold_nopunct"), "what is the name of the dogs owner again")
        self.assertEqual(rc.query_variant(q, "no_stopwords"), "name dog's owner, again?")
        self.assertEqual(rc.query_variant("what is it", "no_stopwords"), "")

    def test_age_buckets(self):
        from datetime import datetime, timezone
        now = datetime(2026, 10, 19, tzinfo=timezone.utc)
        cases = {"2026-10-15T00:00:00Z": "0-7d", "2026-10-01": "8-30d", "2026-09-01T12:00:00+02:00": "31-90d",
                 "2026-01-05": "91-365d", 1700000000: "over a year", 1700000000000: "over a year",
                 "2026-12-01T00:00:00Z": "0-7d", "null": "unknown", 0: "unknown", "not a date": "unknown",
                 "0001-01-01T00:00:00+01:00": "unknown", "9999-12-31T23:59:59-05:00": "unknown"}
        for value, bucket in cases.items():
            self.assertEqual(rc.age_bucket(value, now), bucket, value)

    def test_cross_session_scores_on_all(self):
        q = make_question("q2", "what about apples?", "apples", [self.texts[0], self.texts[3]],
                          category="cross_session")
        const = rb.question_constants(q, self.pool)
        entries, _ = rb.build_window([self.texts[0], self.texts[1]], 10)
        rec = rb.score_question(q, const, entries, self.pool)
        self.assertEqual(rec["scored_on"], "gold_all")
        self.assertIsNone(rec["headline_rank"])


class _V1Adapter(BaseMemoryAdapter):
    """Old-style adapter: setup(workspace), ingest(text, source, category), text hits."""

    def setup(self, workspace):
        self.docs = []

    def ingest(self, text, source="", category="general"):
        self.docs.append(text)

    def save(self):
        pass

    def search(self, query, limit=15):
        words = set(query.lower().split())
        return [d for d in self.docs if words & set(d.lower().split())][:limit]

    def teardown(self):
        pass


class _ScoredAdapter(_V1Adapter):
    """v2-style adapter with scores and an abstain threshold."""

    def setup(self, workspace, mode="file"):
        self.recs = []

    def ingest(self, record):
        self.recs.append(record)

    def search(self, query, limit):
        words = set(rc.tokenize(query))
        out = []
        for r in self.recs:
            score = len(words & set(rc.tokenize(r["text"])))
            if score:
                out.append((r["id"], r["text"], float(score)))
        return sorted(out, key=lambda h: -h[2])[:limit]

    def version(self):
        return {"version": "t", "engine_class": "lexical", "models": [], "network": False,
                "abstain_threshold": 2.0}


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        texts = [f"record number {i} about topic{i} with filler words here" for i in range(120)]
        texts[7] = "the orchard harvest yielded seven hundred bushels of apples"
        self.texts = texts
        with open(os.path.join(self.tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(make_record(t)) + "\n")
        questions = [
            make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels",
                          [texts[7]]),
            {**make_question("q2", "what colour was the barn", "not mentioned", [], category="abstention",
                             answer_type="abstention"), "gold_ids": [], "gold_hashes": {}},
        ]
        with open(os.path.join(self.tmp, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(questions, f)

    def run_with(self, factory, verify=True):
        return rb.run_benchmark(factory, "test", os.path.join(self.tmp, "questions.json"),
                                os.path.join(self.tmp, "corpus.jsonl"), verify=verify, quiet=True)

    def run_oracle(self, questions):
        from parsica_recall.adapters import get_adapter
        path = os.path.join(self.tmp, "questions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(questions, f)
        saved = os.environ.get("RECALL_QFILE")
        os.environ["RECALL_QFILE"] = path
        try:
            return self.run_with(lambda: get_adapter("oracle"), verify=False)
        finally:
            if saved is None:
                os.environ.pop("RECALL_QFILE", None)
            else:
                os.environ["RECALL_QFILE"] = saved

    def write_corpus(self, ts_for=()):
        with open(os.path.join(self.tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for i, t in enumerate(self.texts):
                rec = make_record(t)
                if i in ts_for:
                    rec["ts"] = f"2026-01-{i:02d}T00:00:00Z"
                f.write(json.dumps(rec) + "\n")

    def test_v1_adapter_still_runs(self):
        res = self.run_with(_V1Adapter)
        self.assertEqual(res["manifest"]["adapter"]["contract"], "v1")
        self.assertEqual(res["headline"]["band"], "S")
        self.assertEqual(res["headline"]["recall"]["1"], 100.0)
        self.assertEqual(res["verification"]["status"], "yes")
        self.assertFalse(res["abstention"]["available"])
        self.assertEqual(res["manifest"]["corpus"]["id_scheme"], "content")
        self.assertNotIn("store ids (private set)", res["standard"]["reasons"])

    def test_abstention_from_scores(self):
        res = self.run_with(_ScoredAdapter, verify=False)
        self.assertTrue(res["abstention"]["available"])
        self.assertEqual(res["abstention"]["total"], 1)
        self.assertTrue(res["calibration"]["available"])

    def test_records_hash_matches_written_records(self):
        res = self.run_with(_ScoredAdapter, verify=False)
        self.assertIsNotNone(res["per_question"][1]["abstained"])
        self.assertEqual(rb.records_hash(res["per_question"]), res["manifest"]["records_hash"])

    def test_one_group_gets_the_headline_chance_and_skill(self):
        res = self.run_with(_ScoredAdapter, verify=False)  # one answerable question: one category, one difficulty
        h = res["headline"]
        for group in (res["by_category"]["direct_recall_user"], res["by_difficulty"]["easy"]):
            self.assertEqual(group["skill"], h["skill"])
            self.assertEqual(group["chance"], h["chance"])
            self.assertEqual(set(group["skill"]), {str(k) for k in h["k_set"]})

    def test_each_group_gets_chance_from_its_own_gold(self):
        t = self.texts
        questions = [make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels",
                                   [t[7]]),
                     make_question("q3", "record about topic11 and topic12", "topic11", [t[11], t[12]],
                                   category="paraphrase_recall"),
                     make_question("q4", "record about topic21 and topic22", "topic21", [t[21], t[22]],
                                   category="cross_session")]
        with open(os.path.join(self.tmp, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(questions, f)
        out = os.path.join(self.tmp, "groups.json")
        res = rb.run_benchmark(_ScoredAdapter, "test", os.path.join(self.tmp, "questions.json"),
                               os.path.join(self.tmp, "corpus.jsonl"), output_path=out, quiet=True)
        cats = res["by_category"]
        for cat, golds in (("direct_recall_user", [(1, "any")]), ("paraphrase_recall", [(2, "any")]),
                           ("cross_session", [(2, "all")])):
            curve = rc.mean_chance_curve(120, golds)
            for k in (1, 5, 10):
                self.assertEqual(cats[cat]["chance"][str(k)], round(100.0 * rc.chance_at(curve, k), 4), (cat, k))
        self.assertGreater(cats["paraphrase_recall"]["chance"]["5"], cats["direct_recall_user"]["chance"]["5"])
        self.assertEqual(cats["cross_session"]["chance"]["1"], 0.0)  # both golds cannot fit in a window of 1
        with open(os.path.join(self.tmp, "groups_breakdowns.csv"), encoding="utf-8") as f:
            rows = f.read().splitlines()
        self.assertEqual(rows[0], "breakdown,group,n,k,gold_pct,chance_pct,skill_pct")
        self.assertEqual(len(rows), 1 + 3 * 3 + 1 * 3)  # three categories and one difficulty, three k each
        self.assertIn(f"category,paraphrase_recall,1,5,{cats['paraphrase_recall']['gold']['5']},"
                      f"{cats['paraphrase_recall']['chance']['5']},{cats['paraphrase_recall']['skill']['5']}", rows)

    def test_result_line_has_every_field(self):
        res = self.run_with(_ScoredAdapter, verify=False)
        for field in ("track=", "band=", "N=", "k=", "window=", "qset=", "gold@5=", "skill@5=", "mrr@10=",
                      "version=", "chance@5=", "ceiling5=", "adapter=", "verified="):
            self.assertIn(field, res["result_line"])

    def test_loader_takes_a_list_of_current_records(self):
        pool_ids = {rc.record_id(t) for t in self.texts}
        q = make_question("q5", "which plan is current", "the plan", [self.texts[1], self.texts[2], self.texts[3]],
                          category="knowledge_update", must_rank_first_text=[self.texts[2], self.texts[3]])
        path = os.path.join(self.tmp, "list_questions.json")
        cases = ((q["must_rank_first"], True), ([], False), ([rc.record_id(self.texts[4])], False),
                 ([q["gold_ids"][1], q["gold_ids"][1]], False), ([7], False))
        for value, ok in cases:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{**q, "must_rank_first": value}], f)
            if ok:
                self.assertEqual(rb.load_questions(path, pool_ids)[0]["must_rank_first"], value)
            else:
                with self.assertRaises(rb.InputError, msg=str(value)):
                    rb.load_questions(path, pool_ids)

    def test_oracle_puts_every_current_copy_first(self):
        q = make_question("q6", "which plan is current", "the plan", [self.texts[1], self.texts[2], self.texts[3]],
                          category="knowledge_update", must_rank_first_text=[self.texts[3], self.texts[2]])
        res = self.run_oracle([q])
        rec = res["per_question"][0]
        self.assertEqual([e["id"] for e in rec["window"][:2]], q["must_rank_first"])
        self.assertTrue(rec["version_ok"])

    def test_version_tier_needs_a_ts_on_every_gold_record(self):
        t = self.texts
        timed = make_question("qa", "plan for topic10", "topic11", [t[10], t[11]], category="knowledge_update",
                              must_rank_first_text=t[11])
        half = make_question("qb", "plan for topic12", "topic13", [t[12], t[13]], category="knowledge_update",
                             must_rank_first_text=t[13])
        self.write_corpus(ts_for=(10, 11, 12))
        h = self.run_oracle([timed, half])["headline"]
        self.assertEqual((h["version_items"], h["version_n"], h["version_pct"]), (2, 1, 100.0))
        self.write_corpus()
        res = self.run_oracle([timed, half])
        h = res["headline"]
        self.assertEqual((h["version_items"], h["version_n"], h["version_ok"], h["version_pct"]), (2, 0, None, None))
        self.assertIn("version=n/a", res["result_line"])
        self.assertTrue(res["per_question"][0]["version_ok"])  # the per-question fact is still recorded

    def test_unstable_double_run_with_no_answerable_questions(self):
        class Flaky(_ScoredAdapter):
            calls = 0

            def search(self, query, limit):
                Flaky.calls += 1
                return super().search(query, limit) if Flaky.calls == 1 else []

        with open(os.path.join(self.tmp, "questions.json"), encoding="utf-8") as f:
            only_abstention = [q for q in json.load(f) if q["category"] == "abstention"]
        with open(os.path.join(self.tmp, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(only_abstention, f)
        res = self.run_with(Flaky, verify=True)
        self.assertEqual(res["verification"]["status"], "unstable")
        self.assertIsNone(res["headline"]["recall"]["5"])

    def test_passes_leave_the_base_run_alone(self):
        plain = self.run_with(_ScoredAdapter, verify=False)
        both = rb.run_benchmark(_ScoredAdapter, "test", os.path.join(self.tmp, "questions.json"),
                                os.path.join(self.tmp, "corpus.jsonl"), quiet=True, passes=["order", "query"])
        untimed = lambda pq: [{k: v for k, v in r.items() if k not in rb.TIMING_FIELDS} for r in pq]
        self.assertEqual(untimed(both["per_question"]), untimed(plain["per_question"]))
        self.assertEqual(both["manifest"]["records_hash"], plain["manifest"]["records_hash"])
        self.assertEqual(both["result_line"].replace("verified=no", ""), plain["result_line"].replace("verified=no", ""))
        self.assertEqual([p["name"] for p in both["manifest"]["params"]["extra_passes"]], ["order", "query"])
        self.assertEqual(both["manifest"]["params"]["extra_passes"][0]["seed"], rb.ORDER_PASS_SEED)
        self.assertFalse(any("extra passes" in r for r in both["standard"]["reasons"]))
        for name in ("order", "query"):  # a single base run is not verified, and the passes carry that
            self.assertEqual(both["extra_passes"][name]["base_verification"], "no", name)
        self.assertTrue(any("extra passes not run (order sensitivity, query robustness)" in r
                            for r in plain["standard"]["reasons"]))
        only_query = rb.run_benchmark(_ScoredAdapter, "test", os.path.join(self.tmp, "questions.json"),
                                      os.path.join(self.tmp, "corpus.jsonl"), quiet=True, passes=["query"])
        self.assertIn("extra passes not run (order sensitivity)", only_query["standard"]["reasons"])
        q = both["extra_passes"]["query"]["variants"]
        self.assertEqual(set(q), set(rc.QUERY_VARIANTS))
        self.assertEqual(q["casefold_nopunct"]["n"], 1)  # the one answerable question

    def test_passes_after_an_unstable_base_run_say_so(self):
        class SecondRunEmpty(_ScoredAdapter):
            made = 0

            def setup(self, workspace, mode="file"):
                super().setup(workspace, mode)
                SecondRunEmpty.made += 1
                self.empty = SecondRunEmpty.made == 2  # base run 2 of --verify finds nothing

            def search(self, query, limit):
                return [] if self.empty else super().search(query, limit)

        res = rb.run_benchmark(SecondRunEmpty, "test", os.path.join(self.tmp, "questions.json"),
                               os.path.join(self.tmp, "corpus.jsonl"), verify=True, quiet=True, passes=["order"])
        self.assertEqual(res["verification"]["status"], "unstable")
        self.assertIn("run-to-run noise", res["extra_passes"]["order"]["note"])
        self.assertTrue(any("tables come from the first run" in n for n in res["notes"]))
        self.assertEqual(res["extra_passes"]["order"]["base_verification"], "unstable")

    def test_order_pass(self):
        from parsica_recall.adapters import get_adapter
        q = make_question("q1", "how many bushels did the orchard harvest yield", "seven hundred bushels",
                          [self.texts[7]])
        path = os.path.join(self.tmp, "questions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([q], f)
        saved = os.environ.get("RECALL_QFILE")
        os.environ["RECALL_QFILE"] = path
        try:
            res = rb.run_benchmark(lambda: get_adapter("oracle"), "oracle", path, os.path.join(self.tmp, "corpus.jsonl"),
                                   quiet=True, passes=["order"])
        finally:
            if saved is None:
                os.environ.pop("RECALL_QFILE", None)
            else:
                os.environ["RECALL_QFILE"] = saved
        order = res["extra_passes"]["order"]
        self.assertEqual((order["answerable"], order["changed"]), (1, 0))  # the oracle ignores ingest order
        self.assertEqual(order["ranks"], {"q1": [0, 0]})
        self.assertEqual(order["gold"]["delta"]["5"], 0.0)

    def test_refuses_missing_gold(self):
        with open(os.path.join(self.tmp, "questions.json"), "w", encoding="utf-8") as f:
            json.dump([{**make_question("q9", "x", "y", ["not in the corpus at all"])}], f)
        with self.assertRaises(rb.InputError):
            self.run_with(_V1Adapter, verify=False)


class LintTests(unittest.TestCase):
    def setUp(self):
        with open(lv.DEFAULT_SPEC, encoding="utf-8") as f:
            self.spec = json.load(f)
        self.gold = "the dryer duct needs a proper metal cap attached before the siding goes on"
        self.other = "a metal cap is a good idea on any roof vent"
        self.records = [make_record(self.gold), make_record(self.other)] + [
            make_record(f"unrelated filler record number {i} about nothing much") for i in range(200)]

    def codes(self, q):
        rep = lv.lint([q], self.records, self.spec)
        return {f["code"] for f in rep["items"][0]["findings"]}, rep

    def test_clean_item_passes(self):
        q = make_question("q1", "what should have been attached first?", "proper metal cap attached before the siding",
                          [self.gold])
        codes, rep = self.codes(q)
        self.assertEqual(rep["summary"]["items_with_errors"], 0, codes)

    def test_key_edges_and_leak_and_question(self):
        codes, _ = self.codes(make_question("q1", "what metal cap was attached?", "metal cap", [self.gold],
                                            alts=["attached before the"]))
        self.assertIn("answer_in_other_record", codes)
        self.assertIn("answer_in_question", codes)
        self.assertIn("key_edge_function_word", codes)

    def test_must_rank_first_list_of_current_copies(self):
        stale = "the old plan was to paint the fence green before the spring rains arrive"
        cur_a = "the new plan is to stain the fence cedar brown before the spring rains arrive"
        cur_b = "update on the yard: the new plan is to stain the fence cedar brown this summer"
        self.records += [make_record(stale), make_record(cur_a), make_record(cur_b)]
        q = make_question("q1", "what is the plan for the yard now?", "stain the fence cedar brown",
                          [stale, cur_a, cur_b], category="knowledge_update", must_rank_first_text=[cur_a, cur_b])
        codes, rep = self.codes(q)
        self.assertEqual(rep["summary"]["items_with_errors"], 0, codes)
        codes, _ = self.codes({**q, "must_rank_first": [rc.record_id(cur_a), rc.record_id(stale)]})
        self.assertIn("answer_not_in_gold", codes)
        codes, _ = self.codes({**q, "must_rank_first": [rc.record_id(cur_a), rc.record_id(self.other)]})
        self.assertIn("must_rank_first_not_gold", codes)
        codes, _ = self.codes({**q, "must_rank_first": []})
        self.assertIn("version_category_without_current", codes)

    def role_item(self):
        """A post that names no one, in a pool where two other records carry the label itself."""
        post = "lost 22 pounds in four months by cutting sugar and walking every evening after dinner"
        self.records += [make_record(post), make_record("the post author said the plan was hard at first"),
                         make_record("the post author shared a photo of the scale this morning")]
        return make_question("q1", "who lost 22 pounds in four months by cutting sugar?", "post author", [post],
                             category="entity_attribution", answer_type="role", alts=["original poster"])

    def test_a_role_label_is_exempt_from_gold_multiplicity(self):
        role = self.role_item()
        codes, rep = self.codes(role)
        self.assertEqual(rep["summary"]["items_with_errors"], 0, codes)
        self.assertFalse(codes & {"answer_in_other_record", "answer_not_in_gold"}, codes)
        codes, _ = self.codes({**role, "answer_type": "phrase"})  # the same item typed phrase, as before
        self.assertLessEqual({"answer_in_other_record", "answer_not_in_gold"}, codes)

    def test_every_alternate_of_a_role_item_must_be_a_label(self):
        role = {**self.role_item(), "acceptable_answers": ["original poster", "the writer"]}
        _, rep = self.codes(role)
        found = [(f["field"], f["value"]) for f in rep["items"][0]["findings"]
                 if f["rule_id"] == "ROLE_ANSWER_MUST_BE_STABLE_LABEL"]
        self.assertEqual(found, [("acceptable_answers", "the writer")])
        _, rep = self.codes({**role, "answer": "the writer", "acceptable_answers": []})
        self.assertEqual([f["field"] for f in rep["items"][0]["findings"]
                          if f["rule_id"] == "ROLE_ANSWER_MUST_BE_STABLE_LABEL"], ["answer"])

    def test_gates_are_all_executable(self):
        q = make_question("q1", "what should have been attached first?", "proper metal cap attached before the siding",
                          [self.gold])
        rep = lv.lint([q], self.records, self.spec, readme_text=None, qset_version=None)
        self.assertTrue(all(isinstance(g["passed"], bool) for g in rep["gates"]))
        failed = set(rep["failed_gates"])
        self.assertIn("GATE_CATEGORY_MINIMUMS", failed)
        self.assertIn("GATE_VERSIONS_MATCH", failed)

    def test_secrets_and_pii_shapes(self):
        # Assembled from parts, so no whole credential or personal-data shape sits in this file.
        planted = ("mail me at someone" + "@" + "example.com, key " + "AKIA" + "ABCDEFGHIJKLMNOP" + ", call 415-555" +
                   "-0132, postgres://admin" + ":hunter2" + "@" + "db.internal/app, lives at 12" + " Baker Street")
        shapes = {h["shape"] for h in lv.scan_secrets_pii(planted)}
        self.assertTrue({"email", "aws_access_key", "phone_nanp", "connection_string_credentials",
                         "street_address"} <= shapes, shapes)

    def test_the_report_says_where_a_shape_sits_and_never_repeats_it(self):
        """A lint report is the kind of file people paste into an issue: it names the record or
        the field and the shape, and never the text that matched."""
        address = "someone" + "@" + "example.com"
        phone = "415-555" + "-0132"
        leaky = make_record(f"write to {address} or call {phone} about the invoice for the cedar deck")
        q = make_question("q1", "what should have been attached first?", "proper metal cap attached before the siding",
                          [self.gold])
        q["review_notes"] = f"asked by {address}"
        rep = lv.lint([q], self.records + [leaky], self.spec)
        pool = [f for f in rep["pool_findings"] if f["record"] == leaky["id"]]
        self.assertEqual({f["code"] for f in pool}, {"pii_email", "pii_phone_nanp"})
        self.assertTrue(all(set(f) == {"rule_id", "severity", "code", "record"} for f in pool), pool)
        item = [f for f in rep["items"][0]["findings"] if f["rule_id"] == "NO_SECRETS_OR_PII"]
        self.assertEqual([f["message"] for f in item], ["'review_notes' matches a pii shape (email)."])
        text = json.dumps(rep)
        for planted in (address, phone, "example.com", "0132"):
            self.assertNotIn(planted, text)
        self.assertEqual(set(lv.scan_secrets_pii(address)[0]), {"kind", "shape"})

    def test_no_false_positives_on_ordinary_text(self):
        for text in ["The post was submitted on March 10, 2026.", "run make -j 4 after cmake ..",
                     "In the racecode 'BE_20230112_8' Admirer finished in 3rd place.",
                     "Use 100 grams of 00 flour per egg.", "gopkg.in/yaml.v3"]:
            self.assertEqual(lv.scan_secrets_pii(text), [], text)


# A store that already holds its records, for live mode. Module level so the runner can
# import the adapter by dotted path, the way it imports any custom adapter.
_LIVE_STORE = {"records": {}, "ts": {}, "grow_on_search": False}


class _FakeLiveAdapter(BaseMemoryAdapter):
    """In-memory live store: it holds its records already and never ingests."""

    def setup(self, workspace, mode="file"):
        if mode != "live":
            raise ValueError("this store only runs live")
        self.store = _LIVE_STORE

    def ingest(self, record):
        raise AssertionError("live mode never ingests")

    def save(self):
        pass

    def search(self, query, limit):
        if self.store["grow_on_search"]:  # an agent writes a memory mid-run
            text = f"a memory written during the run, number {len(self.store['records'])}"
            self.store["records"][rc.record_id(text)] = text
            self.store["grow_on_search"] = False
        words = set(rc.tokenize(query))
        hits = [(rid, text, float(len(words & set(rc.tokenize(text)))))
                for rid, text in self.store["records"].items()]
        hits = [h for h in hits if h[2] > 0]
        hits.sort(key=lambda h: (-h[2], h[0]))
        return hits[:limit]

    def teardown(self):
        pass

    def list_ids(self):
        return list(self.store["records"])

    def record_ts(self, ids):
        return {i: self.store["ts"][i] for i in ids if i in self.store["ts"]}

    def version(self):
        return {"version": "t", "engine_class": "lexical", "models": [], "network": False}


class _FakeLiveNoList(_FakeLiveAdapter):
    """A live store that cannot list its ids or say when records were written."""

    def list_ids(self):
        return None

    def record_ts(self, ids):
        return None


class _FakeLiveLosesList(_FakeLiveAdapter):
    """A live store whose list_ids() stops answering partway through the run."""

    def list_ids(self):
        self.listed = getattr(self, "listed", 0) + 1
        return super().list_ids() if self.listed == 1 else None


class _FakeLiveRaisesAtEnd(_FakeLiveAdapter):
    """A live store whose list_ids() raises on its second call, at the end of the run."""

    def list_ids(self):
        self.listed = getattr(self, "listed", 0) + 1
        if self.listed > 1:
            raise RuntimeError("store went away")
        return super().list_ids()


class _FakeLiveFailsInPass(_FakeLiveAdapter):
    """A live store that answers the base run and then fails every search."""

    made = 0

    def setup(self, workspace, mode="file"):
        super().setup(workspace, mode)
        _FakeLiveFailsInPass.made += 1
        self.down = _FakeLiveFailsInPass.made > 1

    def search(self, query, limit):
        if self.down:
            raise RuntimeError("search backend down")
        return super().search(query, limit)


class _FakeLiveTsRaises(_FakeLiveAdapter):
    """A live store whose record_ts() raises."""

    def record_ts(self, ids):
        raise ConnectionError("timestamps unavailable")


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()  # outside the repository, as live results must be
        texts = [f"memory number {i} about subject{i} and nothing else much" for i in range(150)]
        texts[3] = "the backup job runs every night at two in the morning"
        texts[4] = "the backup job now runs at four in the morning instead"
        texts[5] = "my sister lives in lisbon and works as a nurse there"
        self.texts = texts
        _LIVE_STORE.update(records={rc.record_id(t): t for t in texts},
                           ts={rc.record_id(texts[3]): "2026-01-05T00:00:00Z",
                               rc.record_id(texts[4]): "2026-03-02T00:00:00Z"},
                           grow_on_search=False)
        questions = [
            make_question("q1", "where does my sister live", "lisbon", [texts[5]]),
            make_question("q2", "when does the backup job run now", "four in the morning", [texts[3], texts[4]],
                          category="knowledge_update", must_rank_first_text=texts[4]),
            {**make_question("q3", "what colour is the garage door", "not mentioned", [], category="abstention",
                             answer_type="abstention"), "gold_ids": [], "gold_hashes": {}},
        ]
        self.qpath = os.path.join(self.tmp, "questions.json")
        with open(self.qpath, "w", encoding="utf-8") as f:
            json.dump(questions, f)

    def run_cli(self, *extra, adapter="_FakeLiveAdapter", questions=None):
        argv = ["--live", "--adapter", "custom", "--custom-adapter", f"{__name__}.{adapter}",
                "--questions", questions or self.qpath, *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return rb.main(argv)

    def result(self, adapter="_FakeLiveAdapter"):
        safe = f"{__name__}.{adapter}".replace("/", "_").replace(".", "_")
        with open(os.path.join(self.tmp, f"recall_live_{safe}.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_live_end_to_end(self):
        self.assertEqual(self.run_cli("--verify"), 0)
        res = self.result()
        m, h = res["manifest"], res["headline"]
        self.assertEqual((m["params"]["mode"], m["params"]["track"]), ("live", "live"))
        self.assertEqual(m["corpus"]["N"], 150)
        self.assertEqual(m["corpus"]["id_scheme"], "store")  # a live store's ids are its own
        self.assertEqual(m["corpus"]["pool_sha256"], rc.pool_hash(rc.record_id(t) for t in self.texts))
        self.assertFalse(m["corpus"]["pool_changed"])
        self.assertEqual((h["band"], h["window"]), ("S", 10))
        self.assertEqual(res["verification"]["status"], "yes")
        self.assertEqual(res["per_question"][0]["gold_rank"], 0)
        self.assertEqual((h["version_n"], h["version_pct"]), (1, 100.0))  # record_ts gave both gold records a ts
        self.assertIsNotNone(h["chance"]["5"])
        for field in ("track=live", "band=S", "N=150", "k=5", "skill@5=", "version=100.0", "verified=yes"):
            self.assertIn(field, res["result_line"])
        self.assertNotIn("pool=", res["result_line"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, f"recall_live_{__name__.replace('.', '_')}"
                                                              f"__FakeLiveAdapter_recall_curve.csv")))

    def test_drift_is_reported_not_punished(self):
        gid = rc.record_id(self.texts[5])
        _LIVE_STORE["records"][gid] = self.texts[5] + ", enriched by the store"
        self.assertEqual(self.run_cli(), 0)
        res = self.result()
        self.assertEqual(res["pool"]["drifted_gold_count"], 1)
        self.assertEqual(res["per_question"][0]["gold_rank"], 0)

    def test_pool_size_unknown_without_list_ids(self):
        self.assertEqual(self.run_cli(adapter="_FakeLiveNoList"), 0)
        res = self.result("_FakeLiveNoList")
        line, h = res["result_line"], res["headline"]
        for field in ("N=unknown", "band=unknown", "window=25", "gold@1=", "gold@25=", "pool=unknown",
                      "version=n/a"):
            self.assertIn(field, line)
        self.assertNotIn("skill@", line)
        self.assertIsNone(h["chance"]["5"])
        self.assertIn("pool size unknown: the adapter has no list_ids()", res["standard"]["reasons"])
        for group in list(res["by_category"].values()) + list(res["by_difficulty"].values()):
            self.assertEqual(set(group["chance"]), {str(k) for k in h["k_set"]})  # every k, no chance at any
            self.assertTrue(all(v is None for v in list(group["chance"].values()) + list(group["skill"].values())))
        safe = f"{__name__}.{'_FakeLiveNoList'}".replace(".", "_")
        with open(os.path.join(self.tmp, f"recall_live_{safe}_breakdowns.csv"), encoding="utf-8") as f:
            self.assertIn("category,direct_recall_user,1,25,100.0,,", f.read())

    def test_pool_changed_is_not_verified(self):
        _LIVE_STORE["grow_on_search"] = True
        self.assertEqual(self.run_cli("--verify"), 0)
        res = self.result()
        self.assertTrue(res["manifest"]["corpus"]["pool_changed"])
        self.assertEqual(res["verification"]["status"], "no")
        self.assertIn("pool=changed", res["result_line"])
        self.assertIn("verified=no", res["result_line"])

    def test_redact_keeps_ids_ranks_and_scores(self):
        self.assertEqual(self.run_cli("--redact"), 0)
        res = self.result()
        q1 = res["per_question"][0]
        self.assertEqual((q1["question"], q1["answer"], q1["acceptable_answers"]), (None, None, []))
        top = q1["window"][0]
        self.assertEqual((top["id"], top["rank"], top["score"]), (rc.record_id(self.texts[5]), 0, 2.0))
        self.assertEqual(top["text"], rc.text_hash(self.texts[5]))
        self.assertEqual(rb.records_hash(res["per_question"]), res["manifest"]["records_hash"])
        self.assertTrue(res["manifest"]["params"]["redacted"])
        with open(os.path.join(self.tmp, f"recall_live_{__name__.replace('.', '_')}__FakeLiveAdapter.json"),
                  encoding="utf-8") as f:
            raw = f.read()
        # no question or answer text, and no hash of one either (a short answer is guessable from its hash)
        for secret in ("lisbon", "backup job", "garage door", "where does my sister live", "four in the morning",
                       rc.text_hash("lisbon"), rc.text_hash("where does my sister live")):
            self.assertNotIn(secret, raw)

    def test_live_results_never_go_under_the_repo(self):
        inside = os.path.join(rb.REPO_DIR, "results", "live_should_not_exist.json")
        self.assertEqual(self.run_cli("--output", inside), 2)
        self.assertFalse(os.path.exists(inside))
        default_qset = os.path.join(rb.DEFAULT_QSET_DIR, "questions.json")
        self.assertEqual(self.run_cli(questions=default_qset), 2)  # the default path would be in the repo

    def test_odd_timestamps_do_not_count(self):
        _LIVE_STORE["ts"][rc.record_id(self.texts[3])] = 0
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.result()["headline"]["version_n"], 0)

    def test_a_fingerprint_lost_mid_run_counts_as_a_change(self):
        self.assertEqual(self.run_cli(adapter="_FakeLiveLosesList"), 0)
        res = self.result("_FakeLiveLosesList")
        self.assertTrue(res["manifest"]["corpus"]["pool_changed"])
        self.assertEqual(res["verification"]["status"], "no")

    def test_list_ids_raising_at_the_end_keeps_the_run(self):
        self.assertEqual(self.run_cli("--verify", adapter="_FakeLiveRaisesAtEnd"), 0)
        res = self.result("_FakeLiveRaisesAtEnd")  # the results file was still written
        self.assertTrue(res["manifest"]["corpus"]["pool_changed"])
        self.assertEqual(res["verification"]["status"], "no")
        self.assertIn("store went away", res["verification"]["detail"])
        self.assertTrue(any("store went away" in e for e in res["verification"]["end_errors"]))
        self.assertIn("pool=changed", res["result_line"])
        self.assertIn("verified=no", res["result_line"])
        self.assertEqual(res["per_question"][0]["gold_rank"], 0)  # the searches were kept

    def test_record_ts_raising_leaves_the_pool_and_verification_alone(self):
        self.assertEqual(self.run_cli("--verify", adapter="_FakeLiveTsRaises"), 0)
        res = self.result("_FakeLiveTsRaises")
        corpus = res["manifest"]["corpus"]
        self.assertEqual(corpus["pool_sha256"], corpus["pool_sha256_end"])
        self.assertFalse(corpus["pool_changed"])
        self.assertEqual(res["verification"]["status"], "yes")
        self.assertNotIn("timestamps unavailable", res["verification"]["detail"])
        self.assertTrue(any("timestamps unavailable" in e for e in res["verification"]["end_errors"]))
        h = res["headline"]
        self.assertIn("record_ts() raised", h["version_note"])
        self.assertEqual((h["version_items"], h["version_n"], h["version_pct"]), (1, 0, None))
        self.assertFalse(res["age"]["available"])
        self.assertIn("timestamps unavailable", res["age"]["reason"])
        for field in ("verified=yes", "version=n/a"):
            self.assertIn(field, res["result_line"])
        self.assertNotIn("pool=", res["result_line"])
        self.assertEqual(cr.mismatches(res, res), [])  # comparable: its pool did not change
        self.assertEqual(self.run_cli("--redact", adapter="_FakeLiveTsRaises"), 0)
        res = self.result("_FakeLiveTsRaises")
        self.assertNotIn("timestamps unavailable", json.dumps(res))
        self.assertIn("message withheld by --redact", res["headline"]["version_note"])

    def test_symlinks_cannot_lead_live_results_into_the_repo(self):
        link = os.path.join(self.tmp, "linkdir")
        try:
            os.symlink(os.path.join(rb.REPO_DIR, "tests"), link)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        leaked_json = os.path.join(rb.REPO_DIR, "live_leak_test.json")
        leaked_csv = os.path.join(rb.REPO_DIR, "tests", "live_leak_test.csv")
        try:
            # linkdir/.. folds to the temp dir on paper, but open() would follow the link into the repo
            self.assertEqual(self.run_cli("--output", os.path.join(link, "..", "live_leak_test.json")), 2)
            self.assertFalse(os.path.exists(leaked_json))
            # a symlink sitting where the CSV next to --output would be written
            os.symlink(leaked_csv, os.path.join(self.tmp, "out_recall_curve.csv"))
            self.assertEqual(self.run_cli("--output", os.path.join(self.tmp, "out.json")), 2)
            self.assertFalse(os.path.exists(leaked_csv))
            # and one where the breakdown CSV would be written
            os.symlink(leaked_csv, os.path.join(self.tmp, "second_breakdowns.csv"))
            self.assertEqual(self.run_cli("--output", os.path.join(self.tmp, "second.json")), 2)
            self.assertFalse(os.path.exists(leaked_csv))
        finally:
            for path in (leaked_json, leaked_csv):
                if os.path.exists(path):
                    os.remove(path)

    def test_query_pass_runs_live_and_order_says_unavailable(self):
        self.assertEqual(self.run_cli("--verify", "--passes", "order,query"), 0)
        res = self.result()
        passes = res["extra_passes"]
        self.assertFalse(passes["order"]["available"])
        self.assertIn("live mode", passes["order"]["reason"])
        self.assertNotIn("base_verification", passes["order"])  # a pass that did not run carries nothing
        query = passes["query"]
        self.assertTrue(query["available"])
        self.assertTrue(query["pool_same_as_base"])
        self.assertEqual(query["base_verification"], "yes")  # the double run covers the base run only
        for name in rc.QUERY_VARIANTS:
            self.assertEqual(query["variants"][name]["n"], 2, name)  # both answerable questions
        self.assertEqual(query["variants"]["casefold_nopunct"]["agree_pct"], 100.0)
        self.assertEqual(set(query["variants"]["casefold_nopunct"]["ranks"]), {"q1", "q2"})
        self.assertEqual([p["name"] for p in res["manifest"]["params"]["extra_passes"]], ["query"])
        self.assertIn("extra passes not run (order sensitivity)", res["standard"]["reasons"])

    def test_a_failing_pass_keeps_the_run(self):
        for flags, shown in ((("--passes", "query"), True), (("--passes", "query", "--redact"), False)):
            _FakeLiveFailsInPass.made = 0
            self.assertEqual(self.run_cli(*flags, adapter="_FakeLiveFailsInPass"), 0, flags)
            res = self.result("_FakeLiveFailsInPass")  # the base run's results file was still written
            query = res["extra_passes"]["query"]
            self.assertFalse(query["available"])
            self.assertEqual("search backend down" in query["reason"], shown, flags)
            self.assertEqual(res["per_question"][0]["gold_rank"], 0)
            self.assertIn("extra passes not run (order sensitivity, query robustness)", res["standard"]["reasons"])

    def test_redact_withholds_exception_text(self):
        self.assertEqual(self.run_cli("--redact", adapter="_FakeLiveRaisesAtEnd"), 0)
        res = self.result("_FakeLiveRaisesAtEnd")
        self.assertNotIn("store went away", json.dumps(res))
        self.assertIn("message withheld by --redact", res["verification"]["detail"])
        self.assertIn(rc.text_hash("store went away"), res["verification"]["detail"])

    def test_age_breakdown(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        _LIVE_STORE["ts"] = {rc.record_id(self.texts[3]): (now - timedelta(days=40)).isoformat(),
                             rc.record_id(self.texts[4]): (now - timedelta(days=3)).isoformat()}
        self.assertEqual(self.run_cli(), 0)
        age = self.result()["age"]
        self.assertTrue(age["available"])
        b = age["buckets"]
        self.assertEqual((b["0-7d"]["gold"], b["31-90d"]["gold"], b["unknown"]["gold"]), (1, 1, 1))
        self.assertEqual(b["0-7d"]["pct"], 100.0)
        self.assertEqual(self.run_cli(adapter="_FakeLiveNoList"), 0)
        self.assertFalse(self.result("_FakeLiveNoList")["age"]["available"])  # record_ts() did not answer

    def test_unknown_pass_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rb.main(["--passes", "order,typo", "--questions", self.qpath])

    def test_live_refuses_missing_gold(self):
        del _LIVE_STORE["records"][rc.record_id(self.texts[5])]
        self.assertEqual(self.run_cli(), 2)

    def test_live_needs_a_custom_adapter(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rb.main(["--live", "--adapter", "bm25", "--questions", self.qpath])


class CompareTests(unittest.TestCase):
    """compare_results.py on two real results from one small file-mode set."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        texts = [f"record number {i} about topic{i} with filler words here" for i in range(120)]
        for i in range(1, 7):  # ingested before the gold, one shared word each
            texts[i] = f"an orchard note number {i} with nothing else useful"
        texts[10] = "the kettle whistled while grandma baked the bread"
        for i in range(11, 17):  # ingested after the gold, every question word
            texts[i] = f"what did grandma bake note {i}"
        texts[50] = "the orchard harvest yielded seven hundred bushels of apples"
        with open(os.path.join(self.tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(make_record(t)) + "\n")
        questions = [make_question("q1", "which orchard harvest yielded bushels", "seven hundred bushels", [texts[50]]),
                     make_question("q2", "what did grandma bake", "the bread", [texts[10]])]
        with open(os.path.join(self.tmp, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(questions, f)

    PINNED = {"recall": rc.HARNESS_VERSION, "commit": "c" * 40, "dirty": False}

    def write_result(self, factory, name, harness=None, **kw):
        out = os.path.join(self.tmp, f"{name}.json")
        rb.run_benchmark(factory, name, os.path.join(self.tmp, "questions.json"), os.path.join(self.tmp, "corpus.jsonl"),
                         output_path=out, quiet=True, **kw)
        # Pin the harness fields, so the tests do not depend on the state of this checkout.
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        res["manifest"]["harness_version"] = dict(harness or self.PINNED)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return out

    def compare(self, a, b, *flags):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cr.main([a, b, *flags])
        return code, out.getvalue(), err.getvalue()

    def test_matched_pair_is_diffed(self):
        a = self.write_result(_V1Adapter, "ingest_order")
        b = self.write_result(_ScoredAdapter, "overlap", redact=True)  # redacted results compare the same way
        code, out, _ = self.compare(a, b)
        self.assertEqual(code, 0)
        c = cr.compare(a, b)
        self.assertEqual([q for q, _, _ in c["diff"]["won"]], ["q1"])
        self.assertEqual([q for q, _, _ in c["diff"]["lost"]], ["q2"])
        self.assertEqual(c["diff"]["won"][0][1:], (6, 0))
        self.assertIn("won (1)", out)
        self.assertIn("compare recall/1.0 ", out)
        self.assertIn("won=1 lost=1 up=0 down=0 same=0", c["line"])
        self.assertNotIn("unpinned", c["line"])
        self.assertNotIn("unpinned", self.compare(a, b, "--allow-unpinned")[1])  # nothing was unpinned

    def test_mismatched_pairs_are_refused(self):
        a = self.write_result(_V1Adapter, "ingest_order")
        with open(a, encoding="utf-8") as f:
            base = json.load(f)
        cases = {
            "question set differs": lambda m: m["qset"].update(sha256="0" * 64),
            "pool differs": lambda m: m["corpus"].update(sha256="0" * 64),
            "harness version differs": lambda m: m["harness_version"].update(recall="9.9.9"),
            "harness commit differs": lambda m: m["harness_version"].update(commit="0" * 40),
        }
        for expected, change in cases.items():
            other = json.loads(json.dumps(base))
            change(other["manifest"])
            path = os.path.join(self.tmp, "other.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(other, f)
            code, out, err = self.compare(a, path)
            self.assertEqual(code, 2, expected)
            self.assertIn(expected, err)
            self.assertEqual(out, "")
        live = json.loads(json.dumps(base))
        live["manifest"]["corpus"] = {"mode": "live", "N": 120, "pool_sha256": "sha256:" + "1" * 64}
        path = os.path.join(self.tmp, "live.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(live, f)
        self.assertEqual(self.compare(a, path)[0], 2)
        live["manifest"]["corpus"].update(pool_sha256_end="sha256:" + "2" * 64, pool_changed=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(live, f)
        code, _, err = self.compare(path, path)
        self.assertEqual(code, 2)
        self.assertIn("changed during its run", err)

    def test_unpinned_pairs_are_refused_unless_allowed(self):
        a = self.write_result(_V1Adapter, "ingest_order")
        cases = {
            "uncommitted changes": dict(self.PINNED, dirty=True),
            "harness commit differs": dict(self.PINNED, commit="d" * 40),
            "records no harness commit": dict(self.PINNED, commit="unknown", dirty=None),
        }
        for expected, harness in cases.items():
            b = self.write_result(_ScoredAdapter, "other", harness=harness)
            code, out, err = self.compare(a, b)
            self.assertEqual(code, 2, expected)
            self.assertIn(expected, err)
            self.assertIn("--allow-unpinned", err)
            code, out, err = self.compare(a, b, "--allow-unpinned")
            self.assertEqual(code, 0, expected)
            self.assertIn(expected, out)  # printed as a note
            self.assertTrue(out.rstrip().endswith("unpinned=yes"), out[-200:])
        # a different Recall version is a different harness, not an unpinned one
        b = self.write_result(_ScoredAdapter, "other", harness=dict(self.PINNED, recall="9.9.9"))
        self.assertEqual(self.compare(a, b, "--allow-unpinned")[0], 2)

    def test_installed_copies_are_pinned_by_their_source_hash(self):
        """A copy installed from a package has no commit. It is pinned by source_sha256, the
        hash of the harness's own source files, which every result records (spec section 8)."""
        installed = {"recall": rc.HARNESS_VERSION, "commit": "unknown", "dirty": None,
                     "source_sha256": "sha256:" + "a" * 64}
        a = self.write_result(_V1Adapter, "ingest_order", harness=installed)
        b = self.write_result(_ScoredAdapter, "other", harness=installed)
        code, out, _ = self.compare(a, b)
        self.assertEqual(code, 0)
        self.assertNotIn("unpinned", out)
        self.assertIn(f"same harness {rc.HARNESS_VERSION}@source aaaaaaaa", out)
        # a clean checkout of the same code compares with an installed copy
        checkout = dict(self.PINNED, source_sha256=installed["source_sha256"])
        b = self.write_result(_ScoredAdapter, "other", harness=checkout)
        self.assertEqual(self.compare(a, b)[0], 0)
        # other source and no commit to go by: refused, or a printed note when allowed
        b = self.write_result(_ScoredAdapter, "other", harness=dict(installed, source_sha256="sha256:" + "b" * 64))
        code, _, err = self.compare(a, b)
        self.assertEqual(code, 2)
        self.assertIn("harness source differs", err)
        code, out, _ = self.compare(a, b, "--allow-unpinned")
        self.assertEqual(code, 0)
        self.assertTrue(out.rstrip().endswith("unpinned=yes"), out[-200:])
        # a checkout with uncommitted changes is not pinned, whatever its source hash says
        b = self.write_result(_ScoredAdapter, "other", harness=dict(checkout, dirty=True))
        self.assertEqual(self.compare(a, b)[0], 2)
        # and a result with neither a commit nor a source hash is pinned to nothing
        b = self.write_result(_ScoredAdapter, "other", harness={"recall": rc.HARNESS_VERSION, "commit": "unknown"})
        code, _, err = self.compare(a, b)
        self.assertEqual(code, 2)
        self.assertIn("records no harness commit and no source hash", err)

    def test_every_result_records_the_source_hash_and_git_is_asked_only_in_a_checkout(self):
        out = os.path.join(self.tmp, "plain.json")
        res = rb.run_benchmark(_V1Adapter, "plain", os.path.join(self.tmp, "questions.json"),
                               os.path.join(self.tmp, "corpus.jsonl"), output_path=out, quiet=True)
        source = res["manifest"]["harness_version"]["source_sha256"]
        self.assertEqual(source, rc.harness_source_sha256())
        self.assertRegex(source, r"^sha256:[0-9a-f]{64}$")
        with mock.patch.object(rc, "IS_CHECKOUT", False):  # an installed copy: no commit, whatever git would say
            self.assertEqual(rb.harness_commit(), {"commit": "unknown", "dirty": None})

    def test_classification(self):
        def res(ranks):
            return {"per_question": [{"id": f"q{i}", "scored_on": "gold_any", "headline_rank": r}
                                     for i, r in enumerate(ranks)]}
        d = cr.per_question_diff(res([None, 2, 7, 1, 3, None, 6]), res([4, None, 2, 0, 3, 9, None]), 5)
        names = {key: [q for q, _, _ in rows] for key, rows in d.items()}
        self.assertEqual(names, {"won": ["q0", "q2"], "lost": ["q1"], "moved_up": ["q3", "q5"],
                                 "moved_down": ["q6"], "same": ["q4"]})


class StoreIdTests(unittest.TestCase):
    """File mode for a private set that keeps its store's own ids (spec section 3)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        saved = os.environ.get("RECALL_QFILE")
        self.addCleanup(lambda: os.environ.pop("RECALL_QFILE", None) if saved is None
                        else os.environ.__setitem__("RECALL_QFILE", saved))
        texts = [f"memory number {i} about subject{i} and nothing else much" for i in range(150)]
        texts[3] = "the backup job runs every night at two in the morning"
        texts[5] = "my sister lives in lisbon and works as a nurse there"
        self.texts, self.ids = texts, [f"r_{i:04d}" for i in range(150)]
        self.cpath = os.path.join(self.tmp, "corpus.jsonl")
        with open(self.cpath, "w", encoding="utf-8") as f:
            for rid, t in zip(self.ids, texts):
                f.write(json.dumps({"id": rid, "text": t}) + "\n")

        def question(qid, text, answer, i):
            return {"id": qid, "question": text, "category": "direct_recall_user", "difficulty": "easy",
                    "answer_type": "phrase", "answer": answer, "acceptable_answers": [], "gold_ids": [self.ids[i]],
                    "must_rank_first": None, "gold_hashes": {self.ids[i]: rc.text_hash(texts[i])},
                    "review_status": "publish", "review_notes": ""}

        self.questions = [question("q1", "where does my sister live", "lisbon", 5),
                          question("q2", "when does the backup job run", "two in the morning", 3)]
        self.qpath = os.path.join(self.tmp, "questions.json")
        with open(self.qpath, "w", encoding="utf-8") as f:
            json.dump(self.questions, f)

    def cli(self, *extra):
        argv = ["--adapter", "oracle", "--questions", self.qpath, "--corpus", self.cpath,
                "--output", os.path.join(self.tmp, "out.json"), *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return rb.main(argv)

    def test_oracle_scores_100_with_store_ids_and_the_same_files_are_refused_without(self):
        self.assertEqual(self.cli("--verify", "--store-ids"), 0)
        with open(os.path.join(self.tmp, "out.json"), encoding="utf-8") as f:
            res = json.load(f)
        for k, v in res["headline"]["recall"].items():
            self.assertEqual(v, 100.0, k)
        self.assertEqual(res["verification"]["status"], "yes")
        self.assertEqual(res["manifest"]["corpus"]["id_scheme"], "store")
        self.assertIn("store ids (private set)", res["standard"]["reasons"])
        self.assertEqual(res["per_question"][0]["window"][0]["id"], "r_0005")
        os.remove(os.path.join(self.tmp, "out.json"))
        self.assertEqual(self.cli("--verify"), 2)  # the same files, without the flag
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out.json")))
        with self.assertRaises(rb.InputError):
            rb.load_corpus(self.cpath)

    def test_bm25_floor_runs_on_store_ids(self):
        from parsica_recall.adapters import get_adapter
        res = rb.run_benchmark(lambda: get_adapter("bm25"), "bm25", self.qpath, self.cpath, quiet=True,
                               store_ids=True)
        self.assertEqual(res["headline"]["recall"]["1"], 100.0)

    def test_store_ids_must_be_unique_non_empty_strings(self):
        for bad in ({"id": "  ", "text": "a record with a blank id"}, {"id": 7, "text": "a record with a number"},
                    {"id": "r_0003", "text": "a second record under an id already used"}):
            path = os.path.join(self.tmp, "bad.jsonl")
            with open(self.cpath, encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
                dst.write(src.read() + json.dumps(bad) + "\n")
            with self.assertRaises(rb.InputError, msg=str(bad)):
                rb.load_corpus(path, store_ids=True)

    def test_lint_takes_store_ids_with_its_flag(self):
        with open(lv.DEFAULT_SPEC, encoding="utf-8") as f:
            spec = json.load(f)
        records = [{"id": rid, "text": t} for rid, t in zip(self.ids, self.texts)]
        rep = lv.lint(self.questions, records, spec, store_ids=True)
        self.assertEqual(rep["summary"]["items_with_errors"], 0, rep["summary"]["finding_code_counts"])
        self.assertEqual(rep["id_scheme"], "store")
        rep = lv.lint(self.questions, records, spec)
        self.assertIn("gold_ids_malformed", {f["code"] for f in rep["items"][0]["findings"]})
        self.assertNotIn("id_scheme", rep)

    def test_store_ids_is_file_mode_only(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rb.main(["--live", "--adapter", "custom", "--custom-adapter", "x.Y", "--store-ids",
                     "--questions", self.qpath])
        with self.assertRaises(rb.InputError):
            rb.run_benchmark(_V1Adapter, "test", self.qpath, None, quiet=True, live=True, store_ids=True)

    def test_bare_text_hits_under_store_ids_are_named(self):
        res = rb.run_benchmark(_V1Adapter, "v1", self.qpath, self.cpath, quiet=True, store_ids=True)
        self.assertEqual(res["headline"]["recall"]["5"], 0.0)  # its hashed ids never match a store id
        self.assertTrue(any("came back as bare text" in n for n in res["notes"]), res["notes"])
        from parsica_recall.adapters import get_adapter
        res = rb.run_benchmark(lambda: get_adapter("bm25"), "bm25", self.qpath, self.cpath, quiet=True,
                               store_ids=True)
        self.assertEqual(res["notes"], [])

    def test_breakdown_csv_sits_beside_the_curve(self):
        self.assertEqual(rb.breakdown_csv_path("/x/run_recall_curve.csv"), "/x/run_breakdowns.csv")
        self.assertEqual(rb.breakdown_csv_path("/x/curve.csv"), "/x/curve_breakdowns.csv")


class ShippedQuestionSetTests(unittest.TestCase):
    """Checks on the current question set, data/<CURRENT_QSET>, as committed (skipped if it has not been built)."""

    QDIR = os.path.join(REPO, "data", rc.CURRENT_QSET)

    def setUp(self):
        if not os.path.exists(os.path.join(self.QDIR, "questions.json")):
            self.skipTest(f"data/{rc.CURRENT_QSET} not built")

    def test_manifest_matches_files(self):
        info = rb.load_qset_manifest(os.path.join(self.QDIR, "questions.json"),
                                     os.path.join(self.QDIR, "corpus.jsonl"))
        self.assertTrue(info["matches"])
        self.assertEqual(info["version"], rc.CURRENT_QSET)

    def test_oracle_is_perfect_and_deterministic(self):
        from parsica_recall.adapters import get_adapter
        os.environ["RECALL_QFILE"] = os.path.join(self.QDIR, "questions.json")
        res = rb.run_benchmark(lambda: get_adapter("oracle"), "oracle", os.path.join(self.QDIR, "questions.json"),
                               os.path.join(self.QDIR, "corpus.jsonl"), verify=True, quiet=True)
        self.assertEqual(res["verification"]["status"], "yes")
        with open(os.path.join(self.QDIR, "questions.json"), encoding="utf-8") as f:
            answerable = [q for q in json.load(f) if q.get("gold_ids")]
        # The oracle returns every gold record first, so a question is complete at k unless it needs
        # all its gold records (cross_session) and has more than k of them; at the headline k that
        # is every question (v11's public half: 91.1 at k = 1, 100 from k = 5 on).
        for k, v in res["headline"]["recall"].items():
            complete = sum(1 for q in answerable
                           if rb.scoring_rule(q) != "gold_all" or len(q["gold_ids"]) <= int(k))
            self.assertEqual(v, round(100.0 * complete / len(answerable), 2), k)
        self.assertEqual(res["headline"]["recall"][str(res["headline"]["k"])], 100.0)
        if res["headline"]["version_n"]:
            self.assertEqual(res["headline"]["version_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()

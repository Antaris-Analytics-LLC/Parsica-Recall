"""Tests for generator v2: the cost guard (--estimate, --max-cost, --resume),
the shared key handling (--api-key-env, --check-api), the gold window, core first, the
split into a public and a sealed half, the GOLD_MULTIPLICITY retry, the new checks before
the lint, prompts v2, lint spec 3 and show_questions.py. A faked model and temporary
folders only; no network and no anthropic package:

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import re
import sys
import types
import unittest
from collections import Counter
from typing import Any, Dict, List, Tuple
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_generate as tg  # noqa: E402  (the shared helpers: toolboxes, the fake writers, GenTestCase)

gq, lv, rc = tg.gq, tg.lv, tg.rc
REPO, TIME = tg.REPO, tg.TIME
item, toolbox, toolboxes, rec, code, spec = tg.item, tg.toolbox, tg.toolboxes, tg.rec, tg.code, tg.spec

from parsica_recall import show_questions as sq  # noqa: E402

# A key-shaped value for the tests below, assembled from parts so no whole key sits in this file.
KEY_VAR = "QUESTIONS_TEST_KEY"
FAKE_KEY = "-".join(("sk", "ant", "api03", "qtest", "Rk3Lw8Tv1Bn6Yc2M" + "d9Kf4Hj7Gs0Pz5Qe")) + "AA"


def one_per_call(category, units):
    """A writer that gives one item per call, so a run takes several calls per category."""
    return tg.toolbox_writer(category, units)[:1]


class PricedModel(tg.FakeModel):
    """The fake, with usage large enough to price: 100,000 input and 10,000 output tokens a
    call, $0.30 at claude-sonnet-5-5's prices."""

    def __init__(self, writer=one_per_call):
        super().__init__(writer)
        self.live_calls = 0

    def call(self, category, prompt, user, request_sha):
        self.live_calls += 1
        reply = super().call(category, prompt, user, request_sha)
        reply.input_tokens, reply.output_tokens = 100_000, 10_000
        return reply


CALL_COST = (100_000 * 2.0 + 10_000 * 10.0) / 1e6


def files_of(folder):
    out = {}
    for d, _, names in os.walk(folder):
        for n in names:
            path = os.path.join(d, n)
            with open(path, "rb") as f:
                out[os.path.relpath(path, folder)] = f.read()
    return out


class CostTests(tg.GenTestCase):
    # A unit cap of 16 is three calls each (6, 6 and 4 records) at one item a call: six calls in all.
    TARGETS = "direct_recall_user=4,paraphrase_recall=4"

    def run_set(self, out, model, **kw):
        return gq.generate_set(self.corpus_path, os.path.join(self.tmp, out), 7, gq.parse_targets(self.TARGETS),
                               model=model, quiet=True, **kw)

    def setUp(self):
        super().setUp()
        self.corpus_path = self.corpus(toolboxes(80))  # big enough that the sample cap never stops these runs

    def test_max_cost_stops_at_the_right_reply_and_a_resume_costs_nothing_for_what_it_holds(self):
        whole = self.run_set("whole/set-c", PricedModel())
        self.assertEqual(whole["manifest"]["model"]["calls"], 6)
        self.assertEqual(whole["manifest"]["model"]["cost_usd"], round(6 * CALL_COST, 2))
        with self.assertRaises(gq.CostStop) as cm:
            self.run_set("part/set-c", PricedModel(), max_cost=1.0)  # $0.90 after 3 calls, $1.20 after the 4th
        self.assertIn("$1.20 over 4 calls", str(cm.exception))
        folder = os.path.join(self.tmp, "part", "set-c")
        with open(os.path.join(folder, "generation", "calls.jsonl"), encoding="utf-8") as f:
            calls = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(calls), 4)  # the 4th reply passed $1.00, so the 5th call was never made
        self.assertFalse(os.path.exists(os.path.join(folder, "questions.json")))
        again = PricedModel()
        with self.assertRaises(gq.CostStop):  # the same cap: the log replays free, then it stops at once
            self.run_set("part/set-c", again, max_cost=1.0, resume=True)
        self.assertEqual(again.live_calls, 0)
        more = PricedModel()
        res = self.run_set("part/set-c", more, max_cost=5.0, resume=True)
        self.assertEqual(more.live_calls, 2)  # only the two calls the log did not hold
        self.assertEqual(files_of(res["out"]), files_of(whole["out"]))  # as if it never stopped

    def test_a_cut_last_line_is_dropped_before_the_resume_goes_on(self):
        whole = self.run_set("whole/set-c", PricedModel())
        with self.assertRaises(gq.CostStop):
            self.run_set("part/set-c", PricedModel(), max_cost=0.5)
        log = os.path.join(self.tmp, "part", "set-c", "generation", "calls.jsonl")
        with open(log, "a", encoding="utf-8") as f:
            f.write('{"n": 3, "category": "direct_rec')  # what a killed run leaves
        res = self.run_set("part/set-c", PricedModel(), resume=True)
        self.assertEqual(files_of(res["out"]), files_of(whole["out"]))

    def test_a_resume_refuses_other_settings_and_a_frozen_set(self):
        with self.assertRaises(gq.CostStop):
            self.run_set("part/set-c", PricedModel(), max_cost=0.5)
        with self.assertRaises(gq.GenerateError) as cm:
            gq.generate_set(self.corpus_path, os.path.join(self.tmp, "part", "set-c"), 8,
                            gq.parse_targets(self.TARGETS), model=PricedModel(), quiet=True, resume=True)
        self.assertIn("other settings (seed)", str(cm.exception))
        done = self.run_set("whole/set-c", PricedModel())
        with self.assertRaises(gq.GenerateError) as cm:
            self.run_set("whole/set-c", PricedModel(), resume=True)
        self.assertIn("holds a frozen set", str(cm.exception))
        self.assertTrue(done["check"]["ok"])
        with self.assertRaises(gq.GenerateError) as cm:
            self.run_set("nothing/set-c", PricedModel(), resume=True)
        self.assertIn("holds no stopped run", str(cm.exception))

    def test_the_command_line_exits_3_and_a_live_run_needs_max_cost(self):
        argv = ["--corpus", self.corpus_path, "--out", os.path.join(self.tmp, "cli", "set-c"), "--seed", "7",
                "--targets", self.TARGETS]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(gq.main(argv), 2)
        self.assertIn("needs --max-cost", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "cli")))
        err = io.StringIO()
        with mock.patch.object(gq, "AnthropicModel", lambda model_id, api_key_env=None: PricedModel()), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gq.main(argv + ["--max-cost", "1"]), 3)
            self.assertIn("STOPPED: --max-cost $1.00 passed", err.getvalue())
            self.assertEqual(gq.main(argv + ["--max-cost", "5", "--resume"]), 0)

    def test_a_model_with_no_price_cannot_be_guarded(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.run_set("x/set-c", PricedModel(), model_id="claude-unknown-1", max_cost=1.0)
        self.assertIn("no price on file", str(cm.exception))

    def test_prices_are_one_table_for_both_generators(self):
        from parsica_recall import generate_pool as gp
        self.assertIs(gp.PRICES, gq.PRICES)
        self.assertEqual(gq.PRICES["claude-opus-5-5"], (4.00, 20.00, 5.00, 8.00, 0.20))
        self.assertEqual(gq.cost_usd("claude-opus-5-5", 1_000_000, 100_000, 0, 0), 6.0)
        # 5-minute writes (the pool's) at $5, 1-hour writes (the question generator's) at $8
        self.assertEqual(gq.cost_usd("claude-opus-5-5", 0, 0, 1_000_000, 0), 5.0)
        self.assertEqual(gq.cost_exact("claude-opus-5-5", 0, 0, 0, 0, 1_000_000), 8.0)
        rec = {"model": "claude-opus-5-5", "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
               "cache_creation_input_tokens": 1_000_000, "cache_creation_1h_input_tokens": 250_000}
        self.assertAlmostEqual(gq.record_cost(rec), 0.75 * 5.0 + 0.25 * 8.0)
        del rec["cache_creation_1h_input_tokens"]  # a record from before the split: 5-minute writes
        self.assertAlmostEqual(gq.record_cost(rec), 5.0)


class EstimateTests(tg.GenTestCase):
    def test_the_estimate_makes_no_call_reads_no_key_and_writes_nothing(self):
        corpus = self.corpus(toolboxes(30))
        out = os.path.join(self.tmp, "est", "set-e")

        def refuse(*a, **k):
            raise AssertionError("the estimate made a model")

        buf = io.StringIO()
        with mock.patch.object(gq, "AnthropicModel", refuse), mock.patch.dict(sys.modules, {"anthropic": None}), \
                mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(buf):
            os.environ.pop(KEY_VAR, None)
            code_ = gq.main(["--corpus", corpus, "--out", out, "--seed", "7", "--targets", "4", "--estimate",
                             "--api-key-env", KEY_VAR, "--model", "claude-opus-5-5"])
        self.assertEqual(code_, 0, buf.getvalue())
        text = buf.getvalue()
        self.assertIn("Estimate for claude-opus-5-5, before any call", text)
        self.assertIn("worst case: every category runs to its unit cap", text)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "est")))

    def test_expected_and_worst_follow_the_plan(self):
        recs = toolboxes(40)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=4,abstention=2")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc")
        prompts = gq.load_prompts(["direct_recall_user", "abstention"])
        est = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5")
        row = est["categories"]["direct_recall_user"]
        self.assertEqual(row["worst_units"], len(plans["direct_recall_user"].units))
        self.assertEqual(row["worst_units"], gq.unit_cap(4))
        self.assertEqual(row["stages"]["all"]["units"], 8)  # 4 items at the guessed yield of 0.5
        self.assertEqual((row["frozen"], row["frozen_core"], row["short"]), (4.0, None, False))
        self.assertEqual(row["expected"]["calls"], 2)
        self.assertLessEqual(est["expected"]["cost_usd"], est["worst"]["cost_usd"])
        self.assertEqual(est["categories"]["abstention"]["expected"]["retries"], 0)  # abstention never retries
        self.assertEqual((est["questions"], est["core_subset"]), (6.0, None))
        user = len(gq.user_message("direct_recall_user", plans["direct_recall_user"].units[:6], corpus))
        self.assertGreaterEqual(row["expected"]["input_tokens"], int(user / gq.EST_CHARS_PER_TOKEN))

    def test_a_finished_pilot_calibrates_the_estimate(self):
        res = self.gen(toolboxes(30), "direct_recall_user=2,paraphrase_recall=2", model=PricedModel(), out="p/pilot")
        cal = gq.read_calibration(res["out"])
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual(cal["categories"]["direct_recall_user"]["yield"], n["publishable"] / n["units_shown"])
        self.assertEqual(cal["categories"]["direct_recall_user"]["output_per_call"], 10_000)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code_ = gq.main(["--corpus", os.path.join(self.tmp, "corpus.jsonl"), "--out",
                             os.path.join(self.tmp, "full", "set-f"), "--seed", "7", "--targets",
                             "direct_recall_user=20", "--estimate", "--estimate-from", res["out"]])
        self.assertEqual(code_, 0)
        self.assertIn(f"measured by {res['out']}", buf.getvalue())
        with self.assertRaises(gq.GenerateError):
            gq.read_calibration(os.path.join(self.tmp, "nowhere"))


def fake_sdk(clients, writer=tg.toolbox_writer, create_fail=None, client_fail=None, cache=(0, 0), sent=None,
             cache_1h=None):
    """A stand-in for the anthropic package: Anthropic(**kw) records its arguments, and
    messages.create answers as the toolbox writer would (or as --check-api expects).
    cache is the (written, read) cache tokens per call; with cache_1h, the usage also
    splits the writes by how long they are kept, as the API does."""

    class Messages:
        def create(self, **kw):
            if create_fail:
                raise create_fail
            user = kw["messages"][0]["content"]
            if sent is not None:
                sent.append(kw)
            if kw["max_tokens"] == 16:  # --check-api
                return types.SimpleNamespace(model="claude-sonnet-5-5", content=[], stop_reason="end_turn")
            category = kw["system"][0]["text"].split("category: ", 1)[1].split("\n", 1)[0]
            text = json.dumps({"items": writer(category, gq.parse_user_message(user))})
            usage = types.SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=cache[0],
                                          cache_read_input_tokens=cache[1])
            if cache_1h is not None:
                usage.cache_creation = types.SimpleNamespace(ephemeral_5m_input_tokens=cache[0] - cache_1h,
                                                             ephemeral_1h_input_tokens=cache_1h)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)],
                                         model="claude-sonnet-5-5", stop_reason="end_turn", usage=usage)

    class Client:
        def __init__(self, **kw):
            clients.append(kw)
            if client_fail:
                raise client_fail
            self.messages = Messages()

    fake = types.ModuleType("anthropic")
    fake.__version__ = "0.99.0"
    fake.Anthropic = Client
    fake.AnthropicError = type("AnthropicError", (Exception,), {})
    fake.APIError = type("APIError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (fake.APIError,), {})
    return fake


class ApiKeyEnvTests(tg.GenTestCase):
    """--api-key-env for the question generator: the pool's handling, shared."""

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {KEY_VAR: FAKE_KEY, "ANTHROPIC_BASE_URL": "https://proxy.example.invalid"})
        env.start()
        self.addCleanup(env.stop)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertNoLogs(level="DEBUG"):
            code_ = gq.main(list(argv))
        return code_, out.getvalue() + err.getvalue()

    def test_the_key_comes_from_the_named_variable_and_never_shows(self):
        clients = []
        out = os.path.join(self.tmp, "keyed", "set-k")
        corpus = self.corpus(toolboxes(24))
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(clients)}):
            code_, text = self.run_main("--corpus", corpus, "--out", out, "--seed", "7", "--targets",
                                        "direct_recall_user=2", "--max-cost", "1", "--api-key-env", KEY_VAR)
        self.assertEqual(code_, 0, text)
        self.assertEqual(clients, [{"api_key": FAKE_KEY, "base_url": "https://api.anthropic.com"}])
        self.assertNotIn(FAKE_KEY, text)
        for name, data in files_of(out).items():
            self.assertNotIn(FAKE_KEY.encode(), data, name)
            self.assertNotIn(KEY_VAR.encode(), data, name)  # the flag is not recorded either

    def test_an_sdk_error_has_the_key_taken_out(self):
        fake = fake_sdk([])
        fail = fake.AuthenticationError(f"invalid x-api-key {FAKE_KEY}")
        corpus = self.corpus(toolboxes(24))
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], create_fail=fail)}):
            code_, text = self.run_main("--corpus", corpus, "--out", os.path.join(self.tmp, "bad", "set-k"), "--seed",
                                        "7", "--targets", "direct_recall_user=2", "--max-cost", "1",
                                        "--api-key-env", KEY_VAR)
        self.assertEqual(code_, 2, text)
        self.assertIn("[the key]", text)
        self.assertNotIn(FAKE_KEY, text)

    def test_check_api_says_ok_or_the_error_class_only(self):
        clients = []
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(clients)}):
            code_, text = self.run_main("--check-api", "--api-key-env", KEY_VAR)
        self.assertEqual((code_, text.strip()), (0, "ok: claude-sonnet-5-5 answered"))
        self.assertEqual(clients, [{"api_key": FAKE_KEY, "base_url": gq.API_URL}])
        fake = fake_sdk([])
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(
                [], create_fail=fake.AuthenticationError(f"invalid x-api-key {FAKE_KEY}"))}):
            code_, text = self.run_main("--check-api", "--api-key-env", KEY_VAR)
        self.assertEqual((code_, text.strip()), (2, "error: AuthenticationError"))
        code_, text = self.run_main("--check-api", "--api-key-env", FAKE_KEY)
        self.assertEqual(code_, 2)
        self.assertIn("never the key itself", text)
        self.assertNotIn(FAKE_KEY, text)


class WindowAndCoreTests(tg.GenTestCase):
    def test_gold_comes_only_from_the_window_and_the_lint_still_reads_past_it(self):
        recs = toolboxes(20) + toolboxes(20, start=20)
        leak = rec(f"Someone at the depot also wrote cabinet {code(3)} on the shelf label.")
        recs.append(leak)  # past the window: never shown, but it holds toolbox 3's answer
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "direct_recall_user=19", gold_window=20)
        first = {r["id"] for r in recs[:20]}
        self.assertLessEqual(self.shown(res), first)
        self.assertTrue(all(g in first for q in res["questions"] for g in q["gold_ids"]))
        held = [c for c in res["candidates"] if c["returned"]["gold_ids"] == [recs[3]["id"]]]
        self.assertEqual(held[0]["fate"], "held_by_lint")
        self.assertIn("GOLD_MULTIPLICITY", {f["rule_id"] for f in held[0]["findings"]})
        window = res["manifest"]["sampler"]["gold_window"]
        self.assertEqual((window["records"], window["past_the_window"]), (20, 21))
        with self.assertRaises(gq.GenerateError):
            self.gen(recs, "direct_recall_user=1", gold_window=500, out="big/set-w")
        with self.assertRaises(gq.GenerateError):
            self.gen(recs, "direct_recall_user=1", gold_window=20, core=30, out="core/set-w")

    def test_a_near_copy_past_the_window_keeps_its_record_out(self):
        recs = toolboxes(10)
        copy = rec(recs[2]["text"][:-1] + " today.")
        corpus = gq.Corpus(recs + toolboxes(5, start=10) + [copy], spec(), False)
        reps, why = corpus.eligibility(6, 10)
        self.assertNotIn(recs[2]["id"], reps)
        self.assertEqual(why[recs[2]["id"]], "near_copy_past_the_gold_window")
        self.assertNotIn(copy["id"], why)  # past the window: not counted, never shown

    def test_core_first_plans_take_the_core_before_the_rest_of_the_window(self):
        recs = toolboxes(40)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6, 30)
        plans = gq.build_plans(corpus, reps, corpus.age_bands(30), gq.parse_targets("direct_recall_user=10"), 7,
                               "abc", core=10)
        p = plans["direct_recall_user"]
        self.assertEqual((p.n_core, len(p.units)), (10, 30))
        self.assertTrue(all(corpus.pos[u[0]] < 10 for u in p.units[:10]))
        self.assertTrue(all(10 <= corpus.pos[u[0]] < 30 for u in p.units[10:]))
        flat = gq.build_plans(corpus, reps, corpus.age_bands(30), gq.parse_targets("direct_recall_user=10"), 7, "abc")
        self.assertIsNone(flat["direct_recall_user"].n_core)
        self.assertEqual(sorted(flat["direct_recall_user"].units), sorted(p.units))

    def test_core_first_pairs_sit_wholly_in_one_stage(self):
        recs = [r for k in range(8) for r in tg.standup_pair(k)]  # 16 records: pairs 0 to 3 in the core
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), gq.parse_targets("knowledge_update=10,cross_session=10"),
                               7, "abc", core=8)
        for cat in ("knowledge_update", "cross_session"):
            p = plans[cat]
            self.assertTrue(p.units, cat)
            for i, unit in enumerate(p.units):
                stages = {corpus.pos[r] < 8 for r in unit}
                self.assertEqual(stages, {i < p.n_core}, (cat, i))

    def test_a_target_the_core_meets_never_reaches_past_it(self):
        recs = toolboxes(40)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "direct_recall_user=3", core=10, gold_window=30)
        self.assertLessEqual(self.shown(res), {r["id"] for r in recs[:10]})
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["frozen_core"], n["frozen_tail"], n["units_shown_core"]), (3, 0, 6))
        subset = res["manifest"]["core_subset"]
        self.assertEqual((subset["records"], subset["questions"]), (10, 3))
        self.assertEqual(subset["ids"], [q["id"] for q in res["questions"]])

    def test_a_target_past_what_the_core_holds_fills_from_the_rest(self):
        recs = toolboxes(40)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "direct_recall_user=12", core=10, gold_window=30)
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["frozen_core"], n["frozen_tail"], n["units_shown_core"]), (10, 2, 10))
        self.assertEqual(res["manifest"]["core_subset"]["questions"], 10)
        stages = [c["stage"] for c in res["candidates"] if c["fate"] == "frozen"]
        self.assertEqual(stages, ["core"] * 10 + ["tail"] * 2)  # the core first, in generation order


class SplitTests(tg.GenTestCase):
    TARGETS = "direct_recall_user=7,paraphrase_recall=6,abstention=2"

    def split_run(self, name="a", split_seed=1, **kw):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            return self.gen(toolboxes(40), self.TARGETS, out=f"{name}/set-s", core=20, split_seed=split_seed,
                            sealed_out=os.path.join(self.tmp, name, "set-s-sealed"), **kw)

    def test_the_split_is_seeded_stratified_and_complete(self):
        res = self.split_run()
        halves = res["halves"]
        pub, sealed = halves["public"], halves["sealed"]
        frozen = [c for c in res["candidates"] if c["fate"] == "frozen"]
        self.assertEqual(sorted(c["half"] for c in frozen), sorted(["public"] * len(pub["cands"])
                                                                    + ["sealed"] * len(sealed["cands"])))
        self.assertEqual(len(pub["questions"]) + len(sealed["questions"]), len(frozen))
        texts = [q["question"] for q in pub["questions"]] + [q["question"] for q in sealed["questions"]]
        self.assertEqual(len(set(texts)), len(frozen))  # disjoint, and together every frozen item
        strata = Counter((c["category"], c["stage"], c["half"]) for c in frozen)
        for cat, stage in {(c["category"], c["stage"]) for c in frozen}:
            self.assertLessEqual(abs(strata[(cat, stage, "public")] - strata[(cat, stage, "sealed")]), 1)
        self.assertLessEqual(abs(len(pub["questions"]) - len(sealed["questions"])), 1)
        self.assertTrue(res["check"]["ok"])
        for half in (pub, sealed):
            self.assertEqual([q["id"] for q in half["questions"]],
                             [f"q_{i:04d}" for i in range(1, len(half["questions"]) + 1)])
            self.assertEqual(half["report"]["summary"]["items_with_errors"], 0)
            self.assertIn("core_subset", half["manifest"])
        again = self.split_run("b")
        for half in ("public", "sealed"):
            self.assertEqual(files_of(again["halves"][half]["folder"]), files_of(res["halves"][half]["folder"]))
        other = self.split_run("c", split_seed=2)
        self.assertNotEqual([q["question"] for q in other["halves"]["public"]["questions"]],
                            [q["question"] for q in pub["questions"]])

    def test_the_public_half_commits_to_the_sealed_one_and_never_holds_the_record(self):
        res = self.split_run()
        pub, sealed = res["halves"]["public"], res["halves"]["sealed"]
        m = pub["manifest"]
        self.assertEqual(m["split"]["sealed"]["questions_sha256"], rc.file_sha256(sealed["q_path"]))
        self.assertEqual(sealed["manifest"]["split"]["public"]["questions_sha256"], rc.file_sha256(pub["q_path"]))
        self.assertEqual(set(m["split"]["sealed"]), {"questions_sha256", "note"})  # nothing else of the sealed half
        public_files = set(files_of(pub["folder"]))
        self.assertEqual({f for f in public_files if not f.startswith(os.path.join("generation", "prompts"))},
                         {"questions.json", "corpus.jsonl", "lint_report.json", "manifest.json", "reference.json"})
        for name in ("sample.json", "calls.jsonl", "candidates.jsonl", "run.json"):
            self.assertTrue(os.path.isfile(os.path.join(sealed["folder"], "generation", name)), name)
        self.assertNotIn("calls", m["generation"])
        self.assertIn("held with the sealed half", m["generation"]["record"])
        self.assertEqual(sealed["manifest"]["generation"]["calls"]["sha256"],
                         rc.file_sha256(os.path.join(sealed["folder"], "generation", "calls.jsonl")))
        for q in sealed["questions"]:  # no sealed question text reaches any public file
            for name, data in files_of(pub["folder"]).items():
                self.assertNotIn(q["question"].encode(), data, name)
        info = tg.rb.load_qset_manifest(pub["q_path"], pub["c_path"])
        self.assertTrue(info["matches"])

    def test_replay_rebuilds_both_halves(self):
        res = self.split_run()
        calls = os.path.join(res["halves"]["sealed"]["folder"], "generation", "calls.jsonl")
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            again = gq.generate_set(os.path.join(self.tmp, "corpus.jsonl"), os.path.join(self.tmp, "r", "set-s"), 7,
                                    gq.parse_targets(self.TARGETS), core=20, split_seed=1,
                                    sealed_out=os.path.join(self.tmp, "r", "set-s-sealed"), replay=calls, quiet=True)
        for half in ("public", "sealed"):
            self.assertEqual(files_of(again["halves"][half]["folder"]), files_of(res["halves"][half]["folder"]))

    def test_where_the_sealed_half_may_go(self):
        inside = os.path.join(REPO, "sealed", "v99-test-sealed")
        self.assertEqual(gq.check_out(inside, True, sealed=True), (os.path.realpath(inside), "v99-test-sealed"))
        self.assertFalse(os.path.exists(inside))
        for path, release in ((inside, False), (os.path.join(REPO, "data", "v99-x"), True),
                              (os.path.join(self.tmp, "outside"), True)):
            with self.assertRaises(gq.GenerateError):
                gq.check_out(path, release, sealed=True)
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "direct_recall_user=1", split_seed=1)
        self.assertIn("go together", str(cm.exception))
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "direct_recall_user=1", split_seed=1,
                     sealed_out=os.path.join(self.tmp, "run1", "set-a", "inner"))
        self.assertIn("two separate folders", str(cm.exception))
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "direct_recall_user=1", split_seed=1,
                     sealed_out=os.path.join(self.tmp, "other", "set-a"))
        self.assertIn("a name of its own", str(cm.exception))


class RetryModel:
    """A fake that answers first calls with first() and retry calls with again()."""

    def __init__(self, first, again):
        self.first, self.again, self.users = first, again, []

    def call(self, category, prompt, user, request_sha):
        self.users.append(user)
        if user.startswith(gq.RETRY_HEAD):
            held, why, units = gq.parse_retry_message(user)
            items = self.again(category, held, why, units)
        else:
            items = self.first(category, gq.parse_user_message(user))
        return gq.Reply(json.dumps({"items": items}), "fake-model", "end_turn", 7, 3, TIME, "fake")


def leaking(category, units):
    """Every toolbox record says "stays in cabinet", so that answer leaks into all the others."""
    out = []
    for r in units:
        num = tg.TOOLBOX_RE.search(r["text"]).group(1)
        out.append(item(f"Where does the user keep toolbox number {num}?", "stays in cabinet", [r["id"]]))
    return out


def fixed(category, held, why, units):
    cab = tg.TOOLBOX_RE.search(units[0]["text"]).group(2)
    return [item(held["question"], f"cabinet {cab}", held["gold_ids"])]


class RetryTests(tg.GenTestCase):
    def test_a_leak_goes_back_once_with_the_words_named_and_the_fix_freezes(self):
        model = RetryModel(leaking, fixed)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(12), "direct_recall_user=2", model=model)
        retries = [u for u in model.users if u.startswith("RETRY")]
        self.assertEqual(len(retries), 2)  # two fixes meet the target, so the other four holds are not retried
        held, why, units = gq.parse_retry_message(retries[0])
        self.assertEqual(held["answer"], "stays in cabinet")
        self.assertEqual(why, 'the answer "stays in cabinet" also appears in 11 other records of the store')
        self.assertEqual(len(units), 1)  # the record again, and no other
        frozen = [c for c in res["candidates"] if c["fate"] == "frozen"]
        self.assertEqual(len(frozen), 2)
        self.assertTrue(all(c["retry_of"] for c in frozen))
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["retry_calls"], n["frozen_after_retry"], n["calls"]), (2, 2, 3))
        with open(os.path.join(res["out"], "generation", "calls.jsonl"), encoding="utf-8") as f:
            calls = [json.loads(line) for line in f]
        self.assertEqual([c["retry_of"] is not None for c in calls], [False, True, True])
        self.assertEqual(res["manifest"]["model"]["retry_calls"], 2)

    def test_an_item_held_again_is_not_retried_twice(self):
        model = RetryModel(leaking, lambda category, held, why, units: [dict(held)])
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError):
            self.gen(toolboxes(12), "direct_recall_user=2", model=model)  # nothing ever freezes
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "candidates.jsonl"), encoding="utf-8") as f:
            cands = [json.loads(line) for line in f]
        firsts = {c["n"] for c in cands if c["retry_of"] is None}
        retried = Counter(c["retry_of"] for c in cands if c["retry_of"] is not None)
        self.assertEqual(set(retried), firsts)  # every first-pass hold had its one retry
        self.assertEqual(set(retried.values()), {1})  # and only one

    def test_other_errors_get_no_retry(self):
        def edges(category, units):
            return [dict(it, acceptable_answers=["in cabinet"]) for it in leaking(category, units)]  # KEY_EDGES too

        model = RetryModel(edges, fixed)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError):
            self.gen(toolboxes(12), "direct_recall_user=2", model=model)
        self.assertFalse([u for u in model.users if u.startswith("RETRY")])


class ChecksV2Tests(unittest.TestCase):
    def setUp(self):
        self.box = toolbox(0)
        self.corpus = gq.Corpus([self.box] + toolboxes(4, start=1), spec(), False)
        self.unit = (self.box["id"],)
        self.ok = item("Where does the user keep toolbox number 1000?", "cabinet xqaa", [self.box["id"]])

    def check(self, **change):
        return gq.rule_checks("direct_recall_user", dict(self.ok, **change), self.unit, self.corpus)

    def test_two_questions_in_one_are_dropped(self):
        self.assertIsNone(self.check())
        self.assertEqual(self.check(question="Where is toolbox number 1000 kept and who moved it there?"),
                         "two questions in one: the question goes on with 'and who'")
        self.assertEqual(self.check(question="Where is toolbox 1000? And when was it moved?"),
                         "two questions in one: the item holds more than one question mark")
        self.assertIsNone(self.check(question="Which cabinet holds the red and blue toolbox number 1000?"))

    def test_an_alternate_keeps_the_answers_unit(self):
        self.assertEqual(self.check(answer="22 pounds", acceptable_answers=["22"]),
                         "an acceptable answer drops the unit the answer has: '22'")
        self.assertEqual(self.check(answer="4 months", acceptable_answers=["4"]),
                         "an acceptable answer drops the unit the answer has: '4'")
        for alts in (["22 lbs"], ["twenty-two pounds"], []):
            self.assertIsNone(self.check(answer="22 pounds", acceptable_answers=alts), alts)
        self.assertIsNone(self.check(acceptable_answers=["cabinet 1000"]))  # the answer has no number

    def test_prompts_v2(self):
        prompts = gq.load_prompts(rc.CATEGORIES)
        for cat, p in prompts.items():
            self.assertIn("- Ask one thing.", p.text, cat)
            if cat == "abstention":
                continue
            m = re.search(r"small words: (.*?)\.(?: |\n)", p.text)
            self.assertIsNotNone(m, cat)
            self.assertEqual(set(m.group(1).split(", ")), set(rc.FUNCTION_WORDS), cat)
            self.assertIn("A SECOND TRY", p.text, cat)
            self.assertIn("starts with the line RETRY", p.text, cat)
            self.assertIn("OSU", p.text, cat)
        self.assertNotIn("record that contains a number", prompts["temporal_awareness"].text)
        self.assertIn('"9PM"', prompts["temporal_awareness"].text)
        self.assertIn("Advice is not a preference", prompts["implicit_preference"].text)
        self.assertIn("preferred thing's own name", prompts["implicit_preference"].text)
        for cat in ("direct_recall_user", "direct_recall_agent", "entity_attribution"):
            self.assertIn("Do not restate most of the record", prompts[cat].text, cat)


def lint_item(answer, gold_text, category="direct_recall_user", alts=(), answer_type="phrase"):
    records = [rec(gold_text)] + [rec(f"Unrelated filler record number {i} about nothing much at all") for i in range(20)]
    gold = records[0]["id"]
    q = {"id": "q_0001", "question": "What did the record say about it?", "category": category,
         "difficulty": "medium", "answer_type": answer_type, "answer": answer, "acceptable_answers": list(alts),
         "gold_ids": [gold], "must_rank_first": None, "gold_hashes": {gold: rc.text_hash(gold_text)},
         "review_status": "publish", "review_notes": ""}
    rep = lv.lint([q], records, spec())
    return {f["rule_id"] for f in rep["items"][0]["findings"]}


class LintSpec3Tests(unittest.TestCase):
    def test_a_time_of_day_passes_the_length_rule_and_a_short_initialism_does_not(self):
        self.assertEqual(spec()["spec_version"], "v10-lint-spec-3")
        found = lint_item("9PM", "The night shift handover moved to 9PM starting next week for everyone.")
        self.assertNotIn("PRIMARY_ANSWER_TOO_SHORT", found)
        found = lint_item("Ohio State University", "She finished her degree at Ohio State University last spring.",
                          category="entity_attribution", alts=["OSU"], answer_type="entity")
        self.assertIn("ACCEPTABLE_ANSWER_TOO_SHORT", found)

    def test_a_shown_preference_may_be_named_and_a_stated_one_may_not(self):
        shown = lint_item("oat milk latte", "Ordered the oat milk latte again, third time this week, though it costs "
                                            "more.", category="implicit_preference")
        self.assertNotIn("IMPLICIT_PREFERENCE_MUST_NOT_BE_EXPLICIT", shown)
        stated = lint_item("oat milk latte", "I always order the oat milk latte, even though it costs more.",
                           category="implicit_preference")
        self.assertIn("IMPLICIT_PREFERENCE_MUST_NOT_BE_EXPLICIT", stated)


class ShowQuestionsTests(tg.GenTestCase):
    def capture(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code_ = sq.main(list(argv))
        return code_, out.getvalue() + err.getvalue()

    def test_questions_summary_and_held_print(self):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(40), "direct_recall_user=3,paraphrase_recall=2", core=10,
                           model=RetryModel(leaking, fixed))
        code_, text = self.capture("questions", res["out"])
        self.assertEqual(code_, 0)
        self.assertIn("######## direct_recall_user: 3", text)
        self.assertIn("q_0001 (", text)
        self.assertIn(", core)", text)
        self.assertIn("  gold ", text)
        self.assertNotIn(chr(0x2014), text)
        code_, text = self.capture("summary", res["out"])
        self.assertEqual(code_, 0)
        self.assertIn("Core subset: 5 questions with every gold record in the first 10", text)
        self.assertIn("oracle:", text)
        code_, text = self.capture("held", res["out"], "--category", "direct_recall_user")
        self.assertEqual(code_, 0)
        self.assertIn("held_by_lint", text)
        self.assertIn("GOLD_MULTIPLICITY", text)

    def test_held_on_a_public_half_points_at_the_sealed_one(self):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(30), "direct_recall_user=4", split_seed=1,
                           sealed_out=os.path.join(self.tmp, "run1", "set-a-sealed"))
        code_, text = self.capture("held", res["out"])
        self.assertEqual(code_, 2)
        self.assertIn("point at the sealed half", text)
        code_, text = self.capture("held", res["sealed_out"])
        self.assertEqual(code_, 0)
        code_, text = self.capture("summary", res["out"])
        self.assertIn("Split: the public half, seed 1", text)


# ---------------------------------------------------------------------------
# After review: the core quota, the estimate by stage,
# prompt caching, and the two small changes
# ---------------------------------------------------------------------------

def tail_only(category, units):
    """A writer that gives nothing for the core's records (toolboxes 1000 to 1019)."""
    return [it for it in tg.toolbox_writer(category, units)
            if int(re.search(r"number (\d+)", it["question"]).group(1)) >= 1020]


class CoreQuotaTests(tg.GenTestCase):
    def run_quota(self, targets="direct_recall_user=6", quota=2, model=None, core=10, out="run1/set-a", **kw):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            return self.gen(toolboxes(40), targets, core=core, core_quota=quota, model=model, out=out, **kw)

    def test_the_core_stage_ends_once_the_quota_has_frozen(self):
        res = self.run_quota()
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["frozen_core"], n["frozen_tail"], n["units_shown_core"]), (2, 4, 6))
        cands = [c for c in res["candidates"] if c["category"] == "direct_recall_user"]
        past = [c for c in cands if c["stage"] == "core" and c["fate"] == "surplus"]
        self.assertEqual([c["reason"] for c in past], ["the core quota is met"] * 4)  # publishable, past the quota
        self.assertEqual([c["stage"] for c in cands if c["fate"] == "frozen"], ["core"] * 2 + ["tail"] * 4)
        self.assertEqual(res["manifest"]["core_subset"]["questions"], 2)
        core = res["manifest"]["sampler"]["core"]
        self.assertEqual(core["quota"], 2)
        self.assertIn("ends once 2 of its items from the core have frozen", core["rule"])
        with open(os.path.join(res["out"], "generation", "run.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["core_quota"], 2)

    def test_the_core_stage_ends_at_half_the_unit_cap(self):
        res = self.run_quota(targets="direct_recall_user=3", quota=3, model=tg.FakeModel(tail_only), core=20)
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual(n["units_shown_core"], gq.unit_cap(3) // 2)  # the core gave nothing, so half the cap
        self.assertEqual((n["frozen_core"], n["frozen_tail"]), (0, 3))

    def test_the_core_stage_ends_when_its_units_run_out(self):
        res = self.run_quota(targets="direct_recall_user=8", quota=6, core=4)
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["units_shown_core"], n["frozen_core"], n["frozen_tail"]), (4, 4, 4))

    def test_plans_under_a_quota(self):
        recs = toolboxes(60)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=5")
        p = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", core=30, core_quota=2)["direct_recall_user"]
        cap = gq.unit_cap(5)
        self.assertEqual(p.n_core, cap // 2)  # the core stage holds at most half the cap
        self.assertEqual(len(p.units) - p.n_core, cap)  # the tail up to the whole cap, since the core may stop early
        flat = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", core=30)["direct_recall_user"]
        self.assertEqual((flat.n_core, len(flat.units)), (cap, cap))  # no quota: the core takes up to the cap

    def test_a_core_item_past_the_quota_gets_no_retry(self):
        model = RetryModel(leaking, fixed)
        res = self.run_quota(targets="direct_recall_user=3", quota=1, model=model)
        retried = [c for c in res["candidates"] if c["retry_of"]]
        self.assertEqual([c["stage"] for c in retried], ["core", "tail", "tail"])  # one core fix, then the tail
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["frozen_core"], n["frozen_tail"], n["retry_calls"]), (1, 2, 3))

    def test_the_quota_needs_a_core_and_a_resume_keeps_it(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "direct_recall_user=1", core_quota=2)
        self.assertIn("goes with --core", str(cm.exception))
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.CostStop):
            self.gen(toolboxes(40), "direct_recall_user=6", core=10, core_quota=2, model=PricedModel(), max_cost=0.1)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0), self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(40), "direct_recall_user=6", core=10, core_quota=3, model=PricedModel(), resume=True)
        self.assertIn("other settings (core_quota)", str(cm.exception))


class EstimateByStageTests(tg.GenTestCase):
    def pilot(self):
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            return self.gen(toolboxes(40), "direct_recall_user=6", core=10, core_quota=2,
                            model=PricedModel(tg.toolbox_writer), out="p/pilot")  # an item for every record

    def test_the_pilot_is_measured_stage_by_stage(self):
        res = self.pilot()
        cal = gq.read_calibration(res["out"])
        stages = cal["categories"]["direct_recall_user"]["stages"]
        self.assertEqual({s: (v["units"], v["publishable"], v["thin"]) for s, v in stages.items()},
                         {"core": (6, 6, True), "tail": (6, 6, True), "all": (12, 12, False)})
        self.assertEqual(stages["core"]["output_per_call"], 10_000)
        self.assertEqual(cal["core"], 10)

    def test_the_estimate_applies_each_stages_yield(self):
        recs = toolboxes(80)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=10")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", core=40, core_quota=3)
        prompts = gq.load_prompts(["direct_recall_user"])
        cal = {"from": "a pilot", "chars_per_token": 2.5, "categories": {"direct_recall_user": {"stages": {
            "core": {"units": 20, "publishable": 20, "yield": 1.0, "output_per_call": 900, "retries_per_call": 0.0},
            "tail": {"units": 8, "publishable": 2, "yield": 0.25, "output_per_call": 700, "retries_per_call": 0.5}}}}}
        est = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5", calibration=cal, core_quota=3)
        row = est["categories"]["direct_recall_user"]
        self.assertEqual((row["stages"]["core"]["units"], row["stages"]["core"]["frozen"]), (3, 3.0))  # the quota
        self.assertEqual((row["stages"]["tail"]["units"], row["stages"]["tail"]["frozen"]), (28, 7.0))  # 7 at 0.25
        self.assertEqual((row["frozen"], row["frozen_core"], row["short"]), (10.0, 3.0, False))
        self.assertEqual(est["thin"], [("direct_recall_user", "tail", 8, 2)])  # 8 units: too few to trust
        self.assertEqual(est["core_subset"], 3.0)
        buf = io.StringIO()
        gq.print_estimate(est, lambda *a: print(*a, file=buf))
        self.assertIn("expected: about 10 questions (3 of them with every gold record in the core: the 1k subset)",
                      buf.getvalue())
        self.assertIn("too few units to trust: direct_recall_user tail (8 units, 2 publishable)", buf.getvalue())

    def test_a_stage_with_no_call_of_its_own_takes_the_pilots_tokens_over_all_its_calls(self):
        recs = toolboxes(80)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=10")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc", core=40, core_quota=3)
        prompts = gq.load_prompts(["direct_recall_user"])
        core = {"units": 20, "publishable": 20, "yield": 1.0, "output_per_call": 900, "retries_per_call": 0.0}

        def est(tail: Dict[str, Any], whole: Dict[str, Any]) -> Dict[str, Any]:
            stages = {"core": core, "tail": {"units": 2, "publishable": 1, "yield": 0.5, **tail}}
            if whole:
                stages["all"] = {"units": 22, "publishable": 21, "yield": 21 / 22, **whole}
            cal = {"from": "a pilot", "categories": {"direct_recall_user": {"stages": stages}}}
            return gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5", calibration=cal,
                               core_quota=3)["categories"]["direct_recall_user"]["expected"]

        # the tail's 2 units came in a batch that ran on from the core, so it has no call of its own
        none_of_its_own = {"output_per_call": None, "output_per_retry": None, "retries_per_call": None}
        over_all = {"output_per_call": 800, "output_per_retry": 300, "retries_per_call": 0.25}
        self.assertEqual(est(none_of_its_own, over_all), est(over_all, {}))  # the pilot's rates over all its calls
        self.assertNotEqual(est(none_of_its_own, over_all), est(none_of_its_own, {}))  # not the EST_* guesses

    def test_a_category_the_cap_leaves_short_is_marked(self):
        recs = [r for k in range(20) for r in tg.standup_pair(k)]
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("knowledge_update=8")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc")
        est = gq.estimate(plans, targets, corpus, gq.load_prompts(["knowledge_update"]), "claude-opus-5-5")
        row = est["categories"]["knowledge_update"]
        self.assertTrue(row["short"])
        self.assertLess(row["frozen"], 8)
        self.assertEqual(est["short"], ["knowledge_update"])
        buf = io.StringIO()
        gq.print_estimate(est, lambda *a: print(*a, file=buf))
        self.assertIn("short of target at the unit cap: knowledge_update (about", buf.getvalue())
        self.assertIn("  short", buf.getvalue())


class CachingTests(tg.GenTestCase):
    def test_the_hash_covers_the_cache_setting(self):
        prompt = gq.load_prompts(["direct_recall_user"])["direct_recall_user"]
        cached = gq.request_sha256("claude-opus-5-5", prompt, "x")
        with mock.patch.object(gq, "SYSTEM_CACHE", None):
            self.assertNotEqual(gq.request_sha256("claude-opus-5-5", prompt, "x"), cached)
            self.assertEqual(gq.system_blocks(prompt), prompt.text)
        with mock.patch.object(gq, "SYSTEM_CACHE", {"type": "ephemeral"}):  # the 5-minute cache
            self.assertNotEqual(gq.request_sha256("claude-opus-5-5", prompt, "x"), cached)
        self.assertEqual(gq.system_blocks(prompt),
                         [{"type": "text", "text": prompt.text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}])

    def run_cached(self, name: str, **sdk: Any) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], str]:
        sent: List[Dict[str, Any]] = []
        corpus = self.corpus(toolboxes(24))
        out = os.path.join(self.tmp, "cached", name)
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], sent=sent, **sdk)}), \
                contextlib.redirect_stdout(buf):
            code_ = gq.main(["--corpus", corpus, "--out", out, "--seed", "7", "--targets", "direct_recall_user=2",
                             "--model", "claude-opus-5-5", "--max-cost", "1"])
        self.assertEqual(code_, 0, buf.getvalue())
        self.assertTrue(sent)
        for kw in sent:
            self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            model = json.load(f)["model"]
        with open(os.path.join(out, "generation", "calls.jsonl"), encoding="utf-8") as f:
            calls = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(model["system_cache"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(model["calls"], len(calls))
        return out, model, calls, buf.getvalue()

    def test_cache_tokens_are_recorded_and_priced_and_shown(self):
        # A reply that does not split its writes wrote what the request asked for: the 1-hour cache.
        _, model, calls, text = self.run_cached("set-c", cache=(100, 1000))
        n = len(calls)
        self.assertEqual([c["cache_creation_1h_input_tokens"] for c in calls], [100] * n)
        self.assertEqual((model["cache_creation_input_tokens"], model["cache_creation_1h_input_tokens"],
                          model["cache_read_input_tokens"]), (100 * n, 100 * n, 1000 * n))
        self.assertEqual(model["cost_usd"], round(n * (10 * 4.0 + 5 * 20.0 + 100 * 8.0 + 1000 * 0.2) / 1e6, 2))
        self.assertIn("cache read 1,000", text)  # the progress line shows it, for the pilot's check

    def test_the_api_split_of_cache_writes_is_recorded_and_priced(self):
        out, model, calls, _ = self.run_cached("set-s", cache=(100_000, 0), cache_1h=60_000)
        n = len(calls)
        self.assertEqual({(c["cache_creation_input_tokens"], c["cache_creation_1h_input_tokens"]) for c in calls},
                         {(100_000, 60_000)})
        self.assertEqual(model["cache_creation_1h_input_tokens"], 60_000 * n)
        each = (10 * 4.0 + 5 * 20.0 + 40_000 * 5.0 + 60_000 * 8.0) / 1e6
        self.assertAlmostEqual(gq.record_cost(calls[0]), each)
        self.assertEqual(model["cost_usd"], round(n * each, 2))
        replayed = gq.ReplayModel(os.path.join(out, "generation", "calls.jsonl"))  # a resume counts the same cost
        self.assertAlmostEqual(replayed.cost, n * each)

    def test_the_estimate_prices_the_cache(self):
        recs = toolboxes(40)
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        targets = gq.parse_targets("direct_recall_user=8")
        plans = gq.build_plans(corpus, reps, corpus.age_bands(), targets, 7, "abc")
        prompts = gq.load_prompts(["direct_recall_user"])
        cached = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5")
        with mock.patch.object(gq, "SYSTEM_CACHE", None):
            plain = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5")
        with mock.patch.object(gq, "SYSTEM_CACHE", {"type": "ephemeral"}):
            short = gq.estimate(plans, targets, corpus, prompts, "claude-opus-5-5")
        e = cached["expected"]
        system = len(prompts["direct_recall_user"].text) / gq.EST_CHARS_PER_TOKEN
        self.assertEqual(e["cache_write_tokens"], int(system))  # written once, by the category's first call
        self.assertEqual(e["cache_read_tokens"], int(system * (e["calls"] + e["retries"] - 1)))
        self.assertEqual((plain["expected"]["cache_write_tokens"], plain["expected"]["cache_read_tokens"]), (0, 0))
        self.assertLess(e["cost_usd"], plain["expected"]["cost_usd"])
        # the write at the 1-hour price, $8 a million on Opus 5.5, where the 5-minute cache's is $5
        self.assertAlmostEqual(e["cost_usd"] - short["expected"]["cost_usd"], int(system) * 3.0 / 1e6)
        self.assertEqual((cached["cached"], short["cached"], plain["cached"]), ("1h", "5m", None))
        buf = io.StringIO()
        gq.print_estimate(cached, lambda *a: print(*a, file=buf))
        self.assertIn("system prompts cached for an hour", buf.getvalue())


class ReviewSmallChangesTests(unittest.TestCase):
    def test_a_version_retry_keeps_the_new_value(self):
        prompts = gq.load_prompts(rc.VERSION_CATEGORIES)
        for cat, p in prompts.items():
            self.assertNotIn("or another fact it states", p.text, cat)
            self.assertIn("The answer must still be the", p.text, cat)  # the retry was rewritten later (test_generate_v3)
            self.assertNotIn("a longer or more distinctive span", p.text, cat)

    def test_more_ways_of_stating_a_preference(self):
        terms = spec()["constants"]["stated_preference_terms"]
        for term in ("go-to", "swear by", "stick with", "from now on", "only buy", "only use", "my pick", "big fan"):
            self.assertIn(term, terms)
        stated = lint_item("oat milk latte", "The oat milk latte is my go-to order at the corner cafe these days.",
                           category="implicit_preference")
        self.assertIn("IMPLICIT_PREFERENCE_MUST_NOT_BE_EXPLICIT", stated)
        shown = lint_item("oat milk latte", "Got the oat milk latte to go and walked to the corner cafe after.",
                          category="implicit_preference")
        self.assertNotIn("IMPLICIT_PREFERENCE_MUST_NOT_BE_EXPLICIT", shown)  # "to go" is not "go-to"


if __name__ == "__main__":
    unittest.main()

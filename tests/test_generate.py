"""Tests for generate_questions.py with a faked model. Standard library only; no network
and no anthropic package:

    python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from collections import Counter
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from parsica_recall import generate_questions as gq  # noqa: E402
from parsica_recall import lint_validator as lv  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402

TIME = "2026-10-20T00:00:00Z"
ADJ = ("red", "blue", "green", "steel", "plastic")
PLACE = ("garage", "attic", "basement", "porch")
TOOLBOX_RE = re.compile(r"toolbox number (\d+) stays in cabinet (\w+)")
QUARTERLY = ("The quarterly report for the Denver office lists {n} open tickets and 7 closed tickets this week "
             "across the support queue and the billing queue together{tail}.")


def spec():
    with open(lv.DEFAULT_SPEC, encoding="utf-8") as f:
        return json.load(f)


def code(i):
    """A unique word per record, with no digits in it."""
    return "xq" + chr(97 + i // 26) + chr(97 + i % 26)


def rec(text, **fields):
    return {"id": rc.record_id(text), "text": text, "kind": "note", **fields}


def toolbox(i, **fields):
    text = (f"Entry {i} says the {ADJ[i % 5]} toolbox number {1000 + i} stays in cabinet {code(i)} "
            f"near the {PLACE[i % 4]}.")
    return {"id": rc.record_id(text), "text": text, "kind": ("note", "chat")[i % 2], **fields}


def toolboxes(n, start=0):
    return [toolbox(i) for i in range(start, start + n)]


def standup_pair(p):
    """An earlier and a later record about one squad's standup, from two sessions."""
    c = code(100 + p)
    early = rec(f"The weekly standup for squad {c} meets in the west{c} room on mornings.",
                ts="2026-02-01T00:00:00Z", source=f"s{p}a")
    late = rec(f"The weekly standup for squad {c} now meets in the east{c} room on afternoons.",
               ts="2026-05-01T00:00:00Z", source=f"s{p}b")
    return early, late


def item(question, answer, gold, alts=(), **extra):
    return {"question": question, "answer": answer, "acceptable_answers": list(alts), "gold_ids": list(gold),
            "review_notes": "", **extra}


QUESTIONS = {
    "direct_recall_user": "Where does the user keep toolbox number {num}?",
    "direct_recall_agent": "Which cabinet did the assistant name for toolbox number {num}?",
    "entity_attribution": "Which storage spot holds toolbox number {num}?",
    "temporal_awareness": "Where was toolbox number {num} kept last week?",
    "paraphrase_recall": "In which storage spot does toolbox {num} sit?",
    "long_horizon": "Where has toolbox number {num} always been kept?",
    "vocabulary_gap": "Where is toolbox number {num} kept?",
    "implicit_preference": "Where does the owner seem to favor storing toolbox {num}?",
}
ANSWERS = {"vocabulary_gap": "storage locker {code}", "implicit_preference": "prefers locker {code}"}


def toolbox_writer(category, units):
    """What a careful model would write for the toolbox records, and nothing for others."""
    if category == "abstention":
        m = TOOLBOX_RE.search(units[0]["text"])
        return [item(f"What brand is toolbox number {m.group(1)}?", "Zorvex", [], ["Quellmark"])] if m else []
    if gq.UNIT_KIND[category] == "pair":
        return []
    out = []
    for r in units:
        m = TOOLBOX_RE.search(r["text"])
        if m:
            num, cab = m.groups()
            out.append(item(QUESTIONS[category].format(num=num),
                            ANSWERS.get(category, "cabinet {code}").format(code=cab), [r["id"]]))
    return out


def pair_writer(category, units):
    if gq.UNIT_KIND[category] != "pair":
        return toolbox_writer(category, units)
    out = []
    for first, second in units:
        m = re.search(r"squad (\w+)", first["text"])
        if not m or m.group(1) not in second["text"]:
            continue  # not one squad's two standup records: a careful model skips the pair
        c = m.group(1)
        extra = {"must_rank_first": second["id"]} if category in rc.VERSION_CATEGORIES else {}
        out.append(item(f"Where does squad {c} hold its standup now?", f"east{c} room", [first["id"], second["id"]],
                        **extra))
    return out


class FakeModel:
    """Stands in for the API: a writer turns each call's records into items."""

    def __init__(self, writer=toolbox_writer, stop_reason="end_turn", text=None):
        self.writer, self.stop_reason, self.text = writer, stop_reason, text

    def call(self, category, prompt, user, request_sha):
        units = gq.parse_user_message(user)
        text = self.text if self.text is not None else json.dumps({"items": self.writer(category, units)})
        return gq.Reply(text, "fake-model", self.stop_reason, 7, 3, TIME, "fake")


class GenTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recall-gen-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def corpus(self, records, name="corpus.jsonl"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def gen(self, records, targets, model=None, seed=7, out="run1/set-a", **kw):
        return gq.generate_set(self.corpus(records), os.path.join(self.tmp, out), seed, gq.parse_targets(targets),
                               model=model or FakeModel(), quiet=True, **kw)

    def shown(self, result):
        ids = set()
        with open(os.path.join(result["out"], "generation", "calls.jsonl"), encoding="utf-8") as f:
            for line in f:
                for unit in gq.parse_user_message(json.loads(line)["user"]):
                    ids |= {r["id"] for r in (unit if isinstance(unit, tuple) else (unit,))}
        return ids


class PromptTests(unittest.TestCase):
    def test_every_category_has_a_prompt_with_its_schema(self):
        prompts = gq.load_prompts(rc.CATEGORIES)
        for cat, p in prompts.items():
            self.assertTrue(p.text.startswith(f"Parsica Recall question writer, category: {cat}\n"), cat)
            self.assertLessEqual({"question", "answer", "acceptable_answers", "gold_ids", "review_notes"}, p.fields)
            self.assertEqual("must_rank_first" in p.fields, cat in rc.VERSION_CATEGORIES, cat)
            self.assertNotIn(chr(0x2014), p.text)  # no em dash

    def test_the_model_sees_records_as_stored(self):
        line = gq.record_json(toolbox(1, ts="2026-01-01T00:00:00Z", source="s1", meta={"private": "x"}))
        self.assertEqual(list(json.loads(line)), ["id", "kind", "ts", "source", "text"])
        self.assertEqual(list(json.loads(gq.record_json(toolbox(2)))), ["id", "kind", "text"])

    def test_pairs_are_labeled_by_their_category(self):
        early, late = standup_pair(0)
        corpus = gq.Corpus([early, late] + toolboxes(2), spec(), False)
        pair = [(early["id"], late["id"])]
        msg = gq.user_message("knowledge_update", pair, corpus)
        self.assertEqual(msg.split("\n"), ["PAIR 1", "earlier: " + gq.record_json(early), "later: " + gq.record_json(late)])
        self.assertEqual(gq.parse_user_message(msg), [(json.loads(gq.record_json(early)), json.loads(gq.record_json(late)))])
        msg = gq.user_message("cross_session", pair, corpus)
        self.assertEqual([line.split(": ", 1)[0] for line in msg.split("\n")], ["PAIR 1", "first", "second"])
        msg = gq.user_message("abstention", pair, corpus)  # a topic is its records, one per line
        self.assertEqual(msg.split("\n"), [gq.record_json(early), gq.record_json(late)])


class LineEndingTests(GenTestCase):
    """A checkout can store text files with CRLF (git's core.autocrlf on Windows)."""

    def crlf_copy(self, path, folder):
        os.makedirs(folder, exist_ok=True)
        with open(path, "rb") as f:
            data = gq._lf(f.read())  # the checkout may hold it either way; generate_questions.py is not pinned to LF
        copy = os.path.join(folder, os.path.basename(path))
        with open(copy, "wb") as f:
            f.write(data.replace(b"\n", b"\r\n"))
        return copy

    def test_a_crlf_checkout_of_a_prompt_loads_parses_and_hashes_like_the_original(self):
        folder = os.path.join(self.tmp, "prompts")
        for cat in rc.CATEGORIES:
            self.crlf_copy(os.path.join(gq.PROMPTS_DIR, f"{cat}.md"), folder)
        lf, crlf = gq.load_prompts(rc.CATEGORIES), gq.load_prompts(rc.CATEGORIES, folder)
        for cat in rc.CATEGORIES:
            self.assertEqual((crlf[cat].text, crlf[cat].schema, crlf[cat].fields, crlf[cat].sha256),
                             (lf[cat].text, lf[cat].schema, lf[cat].fields, lf[cat].sha256), cat)
            self.assertEqual(lf[cat].sha256, rc.file_sha256(lf[cat].path), cat)  # an LF file hashes as before

    def test_a_set_from_a_crlf_checkout_is_written_with_lf_line_endings(self):
        folder = os.path.join(self.tmp, "prompts")
        for cat in ("abstention", "direct_recall_user"):
            self.crlf_copy(os.path.join(gq.PROMPTS_DIR, f"{cat}.md"), folder)
        load, real_open, writes = gq.load_prompts, open, []

        def spy(file, mode="r", *args, **kwargs):
            if set(mode) & set("wax"):
                writes.append((os.path.realpath(file), mode, kwargs.get("newline")))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch.object(gq, "load_prompts", lambda cats: load(cats, folder)), \
                mock.patch("builtins.open", spy):
            res = self.gen(toolboxes(24), "direct_recall_user=2,abstention=1")
        out = os.path.realpath(res["out"])
        mine = [(os.path.relpath(p, out), mode, newline) for p, mode, newline in writes if p.startswith(out + os.sep)]
        self.assertEqual({p for p, _, _ in mine}, {
            "questions.json", "corpus.jsonl", "lint_report.json", "manifest.json", "reference.json",
            os.path.join("generation", "sample.json"), os.path.join("generation", "calls.jsonl"),
            os.path.join("generation", "calls.jsonl.tmp"),  # the final log is rewritten through it, then moved
            os.path.join("generation", "run.json"),
            os.path.join("generation", "candidates.jsonl"), os.path.join("generation", "prompts", "abstention.md"),
            os.path.join("generation", "prompts", "direct_recall_user.md")})
        self.assertFalse(os.path.exists(os.path.join(out, "generation", "calls.jsonl.tmp")))
        for path, mode, newline in mine:
            self.assertTrue("b" in mode or newline == "\n", (path, mode, newline))  # never the OS's own ending
        for cat in ("abstention", "direct_recall_user"):  # the copies and hashes are the LF originals'
            original = os.path.join(gq.PROMPTS_DIR, f"{cat}.md")
            with open(os.path.join(out, "generation", "prompts", f"{cat}.md"), "rb") as f, open(original, "rb") as g:
                self.assertEqual(f.read(), g.read())
            self.assertEqual(res["manifest"]["prompts"][cat]["sha256"], rc.file_sha256(original))

    def test_the_manifest_hashes_the_generator_with_lf_line_endings(self):
        crlf = self.crlf_copy(gq.__file__, os.path.join(self.tmp, "crlf-checkout"))
        with mock.patch.object(gq, "__file__", crlf):
            res = self.gen(toolboxes(12), "direct_recall_user=1")
        self.assertEqual(res["manifest"]["tool"]["sha256"], gq._lf_sha256(gq.__file__))  # the LF bytes, whatever the checkout


class SamplerTests(GenTestCase):
    def test_same_seed_same_set_and_a_new_seed_samples_differently(self):
        recs = toolboxes(30)
        a = self.gen(recs, "direct_recall_user=2", out="a/set-x")
        b = self.gen(recs, "direct_recall_user=2", out="b/set-x")
        c = self.gen(recs, "direct_recall_user=2", seed=8, out="c/set-x")

        def read(result, name):
            with open(os.path.join(result["out"], name), "rb") as f:
                return f.read()

        for name in ("questions.json", "generation/sample.json", "generation/calls.jsonl", "manifest.json"):
            self.assertEqual(read(a, name), read(b, name), name)
        self.assertNotEqual(read(a, "generation/sample.json"), read(c, "generation/sample.json"))

    def test_each_category_draws_its_own_order(self):
        ids = [f"r{i:02d}" for i in range(20)]
        stratum = dict.fromkeys(ids, "s")
        one = list(gq.seeded_order(ids, stratum, 7, "abc", "direct_recall_user"))
        self.assertEqual(sorted(one), ids)
        self.assertNotEqual(one, list(gq.seeded_order(ids, stratum, 7, "abc", "paraphrase_recall")))
        self.assertEqual(one, list(gq.seeded_order(ids[::-1], stratum, 7, "abc", "direct_recall_user")))

    def test_strata_interleave_in_proportion(self):
        order = list(gq.proportional_order({"a": [f"a{i}" for i in range(6)], "b": ["b0", "b1"]}))
        self.assertEqual(order, ["a0", "a1", "b0", "a2", "a3", "a4", "b1", "a5"])

    def test_age_bands_are_quartiles_of_the_corpus_ts(self):
        recs = [toolbox(i, ts=f"2026-0{1 + i}-01T00:00:00Z") for i in range(8)] + toolboxes(2, start=8)
        bands = gq.Corpus(recs, spec(), False).age_bands()
        self.assertEqual(Counter(bands.values()), {"oldest": 2, "older": 2, "newer": 2, "newest": 2, "no ts": 2})
        self.assertEqual({bands[recs[0]["id"]], bands[recs[1]["id"]]}, {"oldest"})
        self.assertEqual(bands[recs[7]["id"]], "newest")

    def test_never_more_than_half_the_eligible_records_are_shown(self):
        res = self.gen(toolboxes(10), "8")
        m = res["manifest"]
        self.assertEqual(m["sampler"]["sample_cap"], 5)
        self.assertLessEqual(m["sampler"]["sample_size"], 5)
        self.assertLessEqual(len(self.shown(res)), 5)
        self.assertIn("sample cap reached", {n["stopped"] for n in m["generation"]["per_category"].values()})

    def test_the_cap_is_split_evenly_so_no_category_is_starved(self):
        res = self.gen(toolboxes(24), "8")  # 7 categories can run here, and the cap is 12 records
        m = res["manifest"]
        per = m["generation"]["per_category"]
        running = [c for c, n in per.items() if not (n["stopped"] or "").startswith(("needs", "no eligible record"))]
        self.assertEqual(len(running), 7)  # entity_attribution needs a rare name, which no toolbox record holds
        self.assertIn("rare name", per["entity_attribution"]["stopped"])
        self.assertEqual(per["entity_attribution"]["units_shown"], 0)
        for cat in running:
            self.assertGreater(per[cat]["units_shown"], 0, cat)
        self.assertEqual(m["sampler"]["sample_size"], 12)
        introduced, seen = Counter(), set()  # who brought each record to the model first
        with open(os.path.join(res["out"], "generation", "calls.jsonl"), encoding="utf-8") as f:
            for line in f:
                call = json.loads(line)
                for unit in call["units"]:
                    for rid in unit:
                        if rid not in seen:
                            seen.add(rid)
                            introduced[call["category"]] += 1
        self.assertLessEqual(max(introduced.values()), 2)  # 12 records over 7 categories, never 6 to one

    def test_ineligible_records_are_never_shown(self):
        short = rec("Too short to ask about.")
        email = rec("Reach Dana at " + "dana.whitfield" + "@" + "example.org" + " about the cabinet keys today.")
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(12) + [short, email], "direct_recall_user=12")
        shown = self.shown(res)
        self.assertEqual(len(shown), 12)
        self.assertNotIn(short["id"], shown)
        self.assertNotIn(email["id"], shown)
        self.assertEqual(res["manifest"]["sampler"]["excluded"], {"secret_or_pii": 1, "too_short": 1})

    def test_every_field_the_model_sees_is_scanned_but_a_random_store_id_is_allowed(self):
        email = "dana.whitfield" + "@" + "example.org"
        key = "sk" + "-ant-" + "abc" * 8
        store_id = "m-" + "a8F3kQ9zX2pL7vR4" + "tY6wB1nC5dE0gHjU"
        self.assertEqual({h["shape"] for h in lv.scan_secrets_pii(store_id)}, {"high_entropy_token"})
        recs = [dict(toolbox(0), id=store_id), dict(toolbox(1), id="m-1", source=email),
                dict(toolbox(2), id="m-2", kind=key), dict(toolbox(3), id="m-3", ts="ask " + email),
                dict(toolbox(4), id="m-4", ts="2026-05-01T00:00:00Z"), dict(toolbox(5), id=email)]
        reps, why = gq.Corpus(recs, spec(), True).eligibility(6)
        self.assertEqual(reps, [store_id, "m-4"])
        self.assertEqual(why, dict.fromkeys(["m-1", "m-2", "m-3", email], "secret_or_pii"))

    def test_the_sample_cap_counts_memories_not_copies(self):
        recs = []
        for t in toolboxes(12):
            recs += [t, rec(t["text"][:-1] + " today.")]  # every record has a near copy
        res = self.gen(recs, "direct_recall_user=2")
        s = res["manifest"]["sampler"]
        self.assertEqual((s["eligible_units"], s["eligible_records"], s["near_copy_clusters"]), (12, 24, 12))
        self.assertEqual(s["sample_cap"], 6)  # half the 12 memories, not half the 24 records
        self.assertLessEqual(s["sample_size"], 6)

    def test_a_category_stops_at_its_unit_cap(self):
        self.assertEqual([gq.unit_cap(t) for t in (0, 1, 3, 4, 5)], [0, 12, 12, 16, 20])
        calls = []

        def writer(category, units):  # one usable item in the first call, then none
            calls.append(len(units))
            return toolbox_writer(category, units)[:1] if len(calls) == 1 else []

        res = self.gen(toolboxes(30), "direct_recall_user=2", model=FakeModel(writer))
        n = res["manifest"]["generation"]["per_category"]["direct_recall_user"]
        self.assertEqual((n["units_shown"], n["frozen"], n["stopped"]), (gq.MIN_UNITS, 1, "unit cap reached"))
        self.assertEqual(calls, [6, 6])
        self.assertEqual(res["manifest"]["sampler"]["sample_cap"], 15)  # the unit cap stopped it, not the sample cap

    def test_a_category_short_of_its_share_waits_and_one_at_the_cap_stops(self):
        st = gq._Category(gq.Plan("direct_recall_user", [("a",), ("b",), ("c",)], 12), 2)
        shown = set()
        st.allowance = 1
        self.assertEqual(st.take(shown, 10), [("a",)])
        self.assertEqual(st.take(shown, 10), [])  # this round's share is spent, so it waits for the next round
        self.assertEqual((st.done, st.stopped), (False, None))
        shown.add("b")  # another category showed b, so it costs nothing
        self.assertEqual(st.take(shown, 10), [("b",)])
        self.assertEqual(st.take(shown, 2), [])  # the whole cap is spent, so it stops
        self.assertEqual((st.done, st.stopped), (True, "sample cap reached"))

    def test_the_last_of_the_cap_goes_to_a_category_that_can_use_it(self):
        recs = [r for p in range(3) for r in standup_pair(p)]
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 0.34):  # a cap of 2 records, split 1 and 1; a pair needs 2
            res = self.gen(recs, "knowledge_update=1,recency_preference=1", model=FakeModel(pair_writer))
        m = res["manifest"]
        per = m["generation"]["per_category"]
        self.assertEqual((m["sampler"]["sample_cap"], m["sampler"]["sample_size"]), (2, 2))
        self.assertEqual((per["knowledge_update"]["units_shown"], per["knowledge_update"]["frozen"]), (1, 1))
        self.assertEqual((per["recency_preference"]["units_shown"], per["recency_preference"]["stopped"]),
                         (0, "sample cap reached"))

    def test_one_item_per_record_and_two_per_abstention_topic(self):
        def writer(category, units):
            out = []
            for it in toolbox_writer(category, units):
                out += [it, dict(it, question=it["question"][:-1] + " today?")]
            if category == "abstention" and out:
                out.append(dict(out[0], question=out[0]["question"][:-1] + " this year?"))
            return out

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(8), "direct_recall_user=2,abstention=1", model=FakeModel(writer))
        over = Counter((c["category"], c["reason"]) for c in res["candidates"]
                       if (c["reason"] or "").startswith("more than"))
        units = res["manifest"]["generation"]["per_category"]["direct_recall_user"]["units_shown"]
        self.assertEqual(over, {("direct_recall_user", "more than 1 item(s) for one record"): units,
                                ("abstention", "more than 2 item(s) for one topic"): 1})


class NearCopyTests(GenTestCase):
    def test_near_copies_cluster_and_join_gold(self):
        a = rec(QUARTERLY.format(n=42, tail=""))
        copy = rec(QUARTERLY.format(n=42, tail=" overall"))
        recs = [a, copy] + toolboxes(4)
        self.assertEqual(gq.Corpus(recs, spec(), False).cluster[copy["id"]], [a["id"], copy["id"]])

        def writer(category, units):
            out = toolbox_writer(category, units)
            out += [item("How many open tickets did the Denver office report?", "42 open tickets", [r["id"]])
                    for r in units if "quarterly" in r["text"]]
            return out

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "direct_recall_user=6", model=FakeModel(writer))
        shown = self.shown(res)
        self.assertIn(a["id"], shown)
        self.assertNotIn(copy["id"], shown)  # one record per cluster is shown
        q = next(q for q in res["questions"] if q["answer"] == "42 open tickets")
        self.assertEqual(q["gold_ids"], [a["id"], copy["id"]])

    def test_a_pair_differing_in_one_number_is_a_version_candidate_not_a_near_copy(self):
        old = rec(QUARTERLY.format(n=42, tail=""), ts="2026-03-01T00:00:00Z")
        new = rec(QUARTERLY.format(n=47, tail=""), ts="2026-06-01T00:00:00Z")
        self.assertGreaterEqual(rc.token_overlap(rc.tokenize(old["text"]), rc.tokenize(new["text"])), 0.9)
        corpus = gq.Corpus([old, new] + toolboxes(6), spec(), False)
        self.assertEqual(corpus.cluster[new["id"]], [new["id"]])
        reps, _ = corpus.eligibility(6)
        pairs = gq.version_pairs(corpus, reps, dict.fromkeys(reps, "s"), 7, "abc",
                                 {"knowledge_update": 12, "correction_handling": 0, "recency_preference": 0})
        self.assertEqual(pairs["knowledge_update"], [(old["id"], new["id"])])

    def test_different_capitalized_words_are_not_near_copies(self):
        a = rec(QUARTERLY.format(n=42, tail=""))
        b = rec(QUARTERLY.format(n=42, tail="").replace("Denver", "Boston"))
        self.assertEqual(gq.Corpus([a, b], spec(), False).cluster[b["id"]], [b["id"]])

    def test_near_copies_need_the_overlap_both_ways(self):
        a = rec(QUARTERLY.format(n=42, tail=""))
        longer = rec(QUARTERLY.format(n=42, tail=" and the night desk logged slow replies on the weekend shift"))
        self.assertEqual(rc.token_overlap(rc.tokenize(a["text"]), rc.tokenize(longer["text"])), 1.0)
        self.assertLess(rc.token_overlap(rc.tokenize(longer["text"]), rc.tokenize(a["text"])), 0.9)
        self.assertEqual(gq.Corpus([a, longer], spec(), False).cluster[longer["id"]], [longer["id"]])

    def test_a_cluster_with_one_ineligible_member_is_out_whole(self):
        a = rec(QUARTERLY.format(n=42, tail=""))
        copy = rec(QUARTERLY.format(n=42, tail=" via " + "dana.whitfield" + "@" + "example.org"))
        corpus = gq.Corpus([a, copy] + toolboxes(4), spec(), False)
        self.assertEqual(corpus.cluster[a["id"]], [a["id"], copy["id"]])
        reps, why = corpus.eligibility(6)
        self.assertNotIn(a["id"], reps)  # else the copy, which may not be shown, would become gold
        self.assertEqual(why, {copy["id"]: "secret_or_pii", a["id"]: "near_copy_of_an_excluded_record"})


class PairTests(GenTestCase):
    def plans(self, recs, targets):
        corpus = gq.Corpus(recs, spec(), False)
        reps, _ = corpus.eligibility(6)
        return corpus, gq.build_plans(corpus, reps, corpus.age_bands(), gq.parse_targets(targets), 7, "abc")

    def test_version_pairs_are_earlier_then_later_and_go_to_one_category(self):
        recs = [r for p in range(6) for r in standup_pair(p)] + toolboxes(6)
        corpus, plans = self.plans(recs, "knowledge_update=1,correction_handling=1,recency_preference=1")
        seen = []
        for cat in rc.VERSION_CATEGORIES:
            self.assertTrue(plans[cat].units, cat)
            for early, late in plans[cat].units:
                self.assertLess(corpus.ts[early], corpus.ts[late])
                seen.append((early, late))
        self.assertEqual(len(seen), 6)
        self.assertEqual(len(set(seen)), 6)  # every pair dealt once

    def test_must_rank_first_is_the_later_record(self):
        recs = [r for p in range(4) for r in standup_pair(p)] + toolboxes(4)

        def writer(category, units):
            out = pair_writer(category, units)
            if out and category in rc.VERSION_CATEGORIES:
                out[0]["must_rank_first"] = units[0][0]["id"]  # names the earlier record
            return out

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "knowledge_update=2", model=FakeModel(writer))
        self.assertTrue([c for c in res["candidates"] if c["reason"] == "must_rank_first is not the later record"])
        self.assertEqual(len(res["questions"]), 2)
        for q in res["questions"]:
            self.assertEqual(q["must_rank_first"], q["gold_ids"][1])
            self.assertIn("now meets", next(r["text"] for r in recs if r["id"] == q["must_rank_first"]))

    def test_cross_session_needs_two_sessions(self):
        recs = [r for p in range(3) for r in standup_pair(p)] + toolboxes(4)
        stripped = [{k: v for k, v in r.items() if k != "source"} for r in recs]
        _, plans = self.plans(stripped, "cross_session=1")
        self.assertEqual(plans["cross_session"].units, [])
        self.assertIn("source", plans["cross_session"].reason)
        corpus, plans = self.plans(recs, "cross_session=1")
        self.assertTrue(plans["cross_session"].units)
        for a, b in plans["cross_session"].units:
            self.assertNotEqual(corpus.source[a], corpus.source[b])

    def test_without_ts_or_source_the_categories_that_need_them_are_not_built(self):
        res = self.gen(toolboxes(20), "direct_recall_user=1,long_horizon=1,knowledge_update=1,correction_handling=1,"
                                      "recency_preference=1,cross_session=1")
        per = res["manifest"]["generation"]["per_category"]
        for cat in ("long_horizon",) + rc.VERSION_CATEGORIES:
            self.assertIn("needs ts", per[cat]["stopped"], cat)
            self.assertEqual(per[cat]["units_shown"], 0)
        self.assertIn("needs source", per["cross_session"]["stopped"])

    def test_long_horizon_draws_from_the_oldest_quarter(self):
        recs = [toolbox(i, ts=f"2026-01-{i + 1:02d}T00:00:00Z") for i in range(16)]
        _, plans = self.plans(recs, "long_horizon=1")
        self.assertEqual(sorted(u[0] for u in plans["long_horizon"].units), sorted(r["id"] for r in recs[:4]))

    def test_records_with_equal_ts_or_one_session_do_not_pair(self):
        early, late = standup_pair(0)
        _, plans = self.plans([early, dict(late, ts=early["ts"])] + toolboxes(4), "knowledge_update=1")
        self.assertEqual(plans["knowledge_update"].units, [])
        self.assertIn("with different ts", plans["knowledge_update"].reason)
        others = [toolbox(i, source="s9") for i in range(4)]
        _, plans = self.plans([early, dict(late, source=early["source"])] + others, "cross_session=1")
        self.assertEqual(plans["cross_session"].units, [])
        self.assertIn("from different sessions", plans["cross_session"].reason)
        _, plans = self.plans([early, late] + others, "cross_session=1")
        self.assertEqual(plans["cross_session"].units, [(early["id"], late["id"])])

    def test_must_rank_first_lists_the_later_record_and_its_near_copies(self):
        early, late = standup_pair(0)
        copy = rec(late["text"][:-1] + " today.", ts=late["ts"], source=late["source"])
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen([early, late, copy] + toolboxes(4), "knowledge_update=1", model=FakeModel(pair_writer))
        q = res["questions"][0]
        self.assertEqual(q["gold_ids"], [early["id"], late["id"], copy["id"]])
        self.assertEqual(q["must_rank_first"], [late["id"], copy["id"]])
        self.assertTrue(res["check"]["ok"])

    def test_cross_session_gold_that_cannot_fit_the_window_is_dropped(self):
        early, late = standup_pair(0)
        copies = [rec(r["text"][:-1] + f" {w}.", ts=r["ts"], source=r["source"])
                  for r in (early, late) for w in ("today", "again")]
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen([early, late] + copies + toolboxes(6), "direct_recall_user=1,cross_session=1",
                           model=FakeModel(pair_writer))
        cand = next(c for c in res["candidates"] if c["category"] == "cross_session")
        self.assertEqual(cand["unit"], [early["id"], late["id"]])
        self.assertEqual((cand["fate"], cand["reason"]),
                         ("dropped_by_checks", "its 6 gold records, near copies included, cannot all fit the window of 5"))
        self.assertTrue(res["check"]["ok"])

    def test_version_items_freeze_with_both_records_and_the_later_first(self):
        recs = [r for p in range(4) for r in standup_pair(p)] + toolboxes(4)
        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(recs, "recency_preference=2", model=FakeModel(pair_writer))
        self.assertEqual(len(res["questions"]), 2)
        self.assertTrue(res["check"]["ok"])
        for q in res["questions"]:
            self.assertEqual(len(q["gold_ids"]), 2)
            self.assertEqual(q["category"], "recency_preference")


class AbstentionTests(GenTestCase):
    def run_with(self, values, question="What brand is toolbox number {num}?", out="run1/set-a"):
        def writer(category, units):
            if category != "abstention":
                return toolbox_writer(category, units)
            num = TOOLBOX_RE.search(units[0]["text"]).group(1)
            return [item(question.format(num=num), values[0], [], values[1:])]

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            return self.gen(toolboxes(8), "direct_recall_user=1,abstention=1", model=FakeModel(writer), out=out)

    def screened(self, res):
        return [c for c in res["candidates"] if c["category"] == "abstention"]

    def test_a_value_the_store_holds_drops_the_item(self):
        res = self.run_with(["garage", "Zorvex"])
        cands = self.screened(res)
        self.assertTrue(cands)
        for c in cands:
            self.assertEqual(c["fate"], "dropped_by_screen")
            self.assertIn("'garage' appears in record", c["reason"])
        self.assertNotIn("abstention", [q["category"] for q in res["questions"]])

    def test_long_values_and_off_topic_questions_drop(self):
        res = self.run_with(["the brand with the very long made up name"])
        self.assertIn("longer than 6 words", self.screened(res)[0]["reason"])
        res = self.run_with(["Zorvex"], question="What kayak paddle did they buy in spring?", out="run2/set-a")
        self.assertIn("fewer than 2 words with its topic", self.screened(res)[0]["reason"])

    def test_a_clean_item_freezes_in_v10_form_with_its_screen_recorded(self):
        res = self.run_with(["Zorvex", "Quellmark"])
        q = next(q for q in res["questions"] if q["category"] == "abstention")
        self.assertEqual((q["answer"], q["acceptable_answers"], q["answer_type"], q["gold_ids"]),
                         ("not mentioned", gq.ABSTENTION_ALTERNATES, "abstention", []))
        frozen = next(c for c in self.screened(res) if c["fate"] == "frozen")
        self.assertEqual(frozen["screen"]["values"], ["Zorvex", "Quellmark"])
        self.assertEqual(frozen["screen"]["found_in"], {})

    def test_an_empty_alternate_says_nothing_and_is_not_screened(self):
        res = self.run_with(["Zorvex", " "])
        frozen = next(c for c in self.screened(res) if c["fate"] == "frozen")
        self.assertEqual(frozen["screen"]["values"], ["Zorvex"])


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.prompts = gq.load_prompts(rc.CATEGORIES)

    def check(self, cat, it, batch):
        return gq.check_item(cat, it, self.prompts[cat], batch)

    def test_checks_before_the_lint(self):
        batch = [("aaaa",), ("bbbb",)]
        ok = item("Where is it kept?", "cabinet xqaa", ["aaaa"])
        self.assertEqual(self.check("direct_recall_user", ok, batch), (("aaaa",), None))
        for bad, reason in [
            (dict(ok, gold_ids=["zzzz"]), "a gold id is not one of the records shown in this call"),
            (dict(ok, review_notes="looks fine"), "review_notes is not empty"),
            (dict(ok, extra=1), "the item's fields do not match the schema"),
            (dict(ok, gold_ids=[]), "no gold ids"),
            (dict(ok, answer=" "), "empty answer"),
            (dict(ok, gold_ids=["aaaa", "aaaa"]), "gold_ids is not a list of distinct strings"),
        ]:
            self.assertEqual(self.check("direct_recall_user", bad, batch), (None, reason))
        self.assertEqual(self.check("abstention", ok, [("aaaa",)])[1], "an abstention item lists gold ids")
        pairs = [("e1", "l1"), ("e2", "l2")]
        good = item("Where is it now?", "east room", ["e2", "l2"], must_rank_first="l2")
        self.assertEqual(self.check("knowledge_update", good, pairs), (("e2", "l2"), None))
        self.assertEqual(self.check("knowledge_update", dict(good, gold_ids=["e1", "l2"]), pairs)[1],
                         "gold_ids is not exactly one pair's two records")
        self.assertEqual(self.check("knowledge_update", dict(good, must_rank_first="e2"), pairs)[1],
                         "must_rank_first is not the later record")
        self.assertEqual(self.check("knowledge_update", ok, pairs)[1], "the item's fields do not match the schema")

    def test_the_prompt_rules_a_machine_can_check(self):
        early, late = standup_pair(0)
        box = toolbox(0)
        corpus = gq.Corpus([early, late, box], spec(), False)
        unit, pair = (box["id"],), (early["id"], late["id"])
        east, west = f"east{code(100)} room", f"west{code(100)} room"
        ok = item("Where does the user keep toolbox number 1000?", "cabinet xqaa", [box["id"]])
        ten_words = "the cabinet in the garage by the old blue door"
        for cat, it, u, want in [
            ("direct_recall_user", ok, unit, None),
            ("direct_recall_user", dict(ok, question="Where is toolbox number 1000 kept？"), unit, None),
            ("direct_recall_user", dict(ok, question="Where is toolbox number 1000 kept؟ "), unit, None),
            ("direct_recall_user", dict(ok, question="Where is toolbox number 1000 kept."), unit,
             "the question does not end in a question mark"),
            ("direct_recall_user", dict(ok, question=" ".join(["word"] * 30) + "?"), unit, None),
            ("direct_recall_user", dict(ok, question=" ".join(["word"] * 31) + "?"), unit,
             "the question is longer than 30 words"),
            ("direct_recall_user", dict(ok, answer=ten_words), unit, "the answer is not 1 to 8 words"),
            ("direct_recall_user", dict(ok, acceptable_answers=[ten_words]), unit,
             "an acceptable answer is longer than 8 words"),
            ("vocabulary_gap", dict(ok, answer="a locker at the far back wall"), unit, "the answer is not 1 to 6 words"),
            ("implicit_preference", dict(ok, answer="locker"), unit, "the answer is not 2 to 8 words"),
            ("abstention", dict(ok, answer=ten_words), unit, None),  # its values are screened instead
            ("knowledge_update", dict(ok, answer=east), pair, None),
            ("knowledge_update", dict(ok, answer=west), pair,
             "the answer or an acceptable answer also appears in the earlier record"),
            ("recency_preference", dict(ok, answer=east, acceptable_answers=[west]), pair,
             "the answer or an acceptable answer also appears in the earlier record"),
            ("cross_session", dict(ok, answer=east), pair, None),
            ("cross_session", dict(ok, answer="weekly standup"), pair,
             "the answer appears in both records, so one record alone answers the question"),
        ]:
            self.assertEqual(gq.rule_checks(cat, it, u, corpus), want, (cat, it["question"], it["answer"]))


class LintIntegrationTests(GenTestCase):
    def test_failing_items_are_held_or_dropped_and_never_frozen(self):
        email = "dana" + "@" + "example.org"

        def writer(category, units):
            out = []
            for i, r in enumerate(units):
                num, cab = TOOLBOX_RE.search(r["text"]).groups()
                if i == 0:  # every record holds this answer: the lint holds it for rewrite
                    out.append(item(f"What happens to toolbox number {num}?", "stays in cabinet", [r["id"]]))
                elif i == 1:  # a personal-data shape: the lint drops it
                    out.append(item(f"Who is the contact for toolbox number {num}?", f"cabinet {cab} {email}",
                                    [r["id"]]))
                elif i == 3:  # the same question text as a frozen item: dropped before the lint
                    out.append(item(out[2]["question"], f"cabinet {cab}", [r["id"]]))
                elif i == 4:  # the same text as the held item, which never ships: this one may
                    out.append(item(out[0]["question"], f"cabinet {cab}", [r["id"]]))
                else:
                    out.append(item(f"Where does the user keep toolbox number {num}?", f"cabinet {cab}", [r["id"]]))
            return out

        res = self.gen(toolboxes(30), "direct_recall_user=3", model=FakeModel(writer))
        fates = {c["n"]: (c["fate"], c["lint_status"], c["reason"]) for c in res["candidates"]}
        self.assertEqual(fates[1][:2], ("held_by_lint", "rewrite"))
        self.assertEqual(fates[2][:2], ("dropped_by_lint", "drop"))
        self.assertEqual(fates[3][0], "frozen")
        self.assertEqual(fates[4], ("dropped_by_checks", None, "the same question text as a frozen item"))
        self.assertEqual((fates[5][0], fates[6][0]), ("frozen", "frozen"))
        self.assertEqual([q["answer"].split()[0] for q in res["questions"]], ["cabinet"] * 3)
        self.assertEqual(res["report"]["summary"]["items_with_errors"], 0)

    def test_a_role_label_answer_freezes(self):
        # The record holds a rare name (Quellmark Park), so entity_attribution samples it; the role label is the answer.
        post = rec("Lost 22 pounds in four months by cutting sugar and walking every evening around Quellmark Park.")
        carriers = [rec("The post author said the plan was hard at first."),  # the label itself, outside gold
                    rec("The post author shared a photo of the scale this morning.")]

        def writer(category, units):
            return [item("Who lost 22 pounds in four months by cutting sugar?", "post author", [r["id"]],
                         ["original poster"]) for r in units if r["id"] == post["id"]]

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen([post] + carriers + toolboxes(6), "entity_attribution=1", model=FakeModel(writer))
        q = res["questions"][0]
        self.assertEqual((q["answer"], q["answer_type"], q["gold_ids"]), ("post author", "role", [post["id"]]))
        self.assertEqual(res["report"]["summary"]["items_with_errors"], 0)
        self.assertTrue(res["check"]["ok"])

    def test_a_repeated_fact_is_held(self):
        def writer(category, units):
            return [item(f"Which cabinet holds toolbox {TOOLBOX_RE.search(r['text']).group(1)} for the {category} "
                         "check?", "cabinet " + TOOLBOX_RE.search(r["text"]).group(2), [r["id"]]) for r in units]

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(3), "direct_recall_user=3,paraphrase_recall=3", model=FakeModel(writer))
        by_record = {}
        for c in res["candidates"]:
            by_record.setdefault(tuple(c["returned"]["gold_ids"]), []).append(c)
        self.assertEqual(sorted(len(v) for v in by_record.values()), [2, 2, 2])  # both categories saw every record
        for cands in by_record.values():
            first, later = sorted(cands, key=lambda c: c["n"])
            self.assertEqual(first["fate"], "frozen")  # the first to state a fact keeps it
            self.assertEqual((later["fate"], later["lint_status"]), ("held_by_lint", "reclassify"))
            self.assertTrue(later["reason"].startswith(f"duplicate_fact: same gold and answer as candidate {first['n']}"))


class FreezeTests(GenTestCase):
    def test_freeze_and_manifest(self):
        recs = toolboxes(24)
        res = self.gen(recs, "direct_recall_user=2,vocabulary_gap=2,abstention=1")
        out, m = res["out"], res["manifest"]
        path = lambda *p: os.path.join(out, *p)  # noqa: E731
        with open(path("questions.json"), encoding="utf-8") as f:
            qs = json.load(f)
        self.assertEqual([q["id"] for q in qs], [f"q_{i:04d}" for i in range(1, 6)])
        self.assertEqual([q["category"] for q in qs], ["direct_recall_user"] * 2 + ["vocabulary_gap"] * 2 + ["abstention"])
        by_id = {r["id"]: r for r in recs}
        for q in qs:
            self.assertEqual(list(q), spec()["required_fields"])
            self.assertIn(q["difficulty"], ("easy", "medium", "hard"))
            for g in q["gold_ids"]:
                self.assertEqual(q["gold_hashes"][g], rc.text_hash(by_id[g]["text"]))
        with open(path("corpus.jsonl"), "rb") as a, open(os.path.join(self.tmp, "corpus.jsonl"), "rb") as b:
            self.assertEqual(a.read(), b.read())
        self.assertEqual(m["files"], {"questions.json": rc.file_sha256(path("questions.json")),
                                      "corpus.jsonl": rc.file_sha256(path("corpus.jsonl"))})
        self.assertEqual(m["lint"]["sha256"], rc.file_sha256(path("lint_report.json")))
        self.assertEqual(m["sampler"]["sha256"], rc.file_sha256(path("generation", "sample.json")))
        self.assertEqual(m["generation"]["calls"]["sha256"], rc.file_sha256(path("generation", "calls.jsonl")))
        self.assertEqual(m["generation"]["candidates"]["sha256"], rc.file_sha256(path("generation", "candidates.jsonl")))
        self.assertEqual((m["model"]["requested"], m["model"]["served"]), ("claude-sonnet-5-5", ["fake-model"]))
        self.assertFalse(m["model"]["server_side_fallbacks"])
        self.assertEqual(set(m["prompts"]), {"direct_recall_user", "vocabulary_gap", "abstention"})
        for p in m["prompts"].values():
            self.assertEqual(p["sha256"], rc.file_sha256(os.path.join(REPO, p["file"])))
        self.assertEqual(sorted(os.listdir(path("generation", "prompts"))),
                         ["abstention.md", "direct_recall_user.md", "vocabulary_gap.md"])
        self.assertEqual((m["status"], m["standard_eligible"], m["id_scheme"]), ("interim", False, "content"))
        self.assertIn("GATE_CATEGORY_MINIMUMS", m["lint"]["failed_gates"])
        self.assertEqual(m["sampler"]["seed"], 7)
        self.assertEqual({c: n["frozen"] for c, n in m["generation"]["per_category"].items()},
                         m["counts"]["by_category"])
        info = rb.load_qset_manifest(path("questions.json"), path("corpus.jsonl"))
        self.assertTrue(info["matches"])
        self.assertEqual(info["status"], "interim")
        with open(path("lint_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["questions_sha256"], m["files"]["questions.json"])
        self.assertEqual(report["summary"]["items_with_errors"], 0)
        with open(path("reference.json"), encoding="utf-8") as f:
            self.assertEqual(set(json.load(f)["runs"]), {"oracle", "bm25"})
        self.assertTrue(res["check"]["ok"])
        self.assertEqual((res["check"]["gold_at_window"], res["check"]["verified"]), (100.0, "yes"))

    def test_answer_types_follow_the_answer(self):
        s = spec()
        for cat, answer, want in [
            ("direct_recall_user", "22 pounds", "quantity_with_unit"), ("direct_recall_user", "4 months", "duration"),
            ("temporal_awareness", "August 15, 2026", "date"), ("direct_recall_user", "2026", "date"),
            ("direct_recall_user", "3300", "number"), ("direct_recall_agent", "=IF(B2=B1, A1, A1+1)", "formula"),
            ("direct_recall_agent", "v3.11", "version"), ("direct_recall_agent", "1.2.3", "version"),
            ("direct_recall_agent", "3.11", "number"), ("direct_recall_user", "3.5%", "quantity_with_unit"),
            ("direct_recall_user", "2.5kg", "quantity_with_unit"), ("temporal_awareness", "10:30", "phrase"),
            ("direct_recall_agent", "--incremental flag", "code_symbol"),
            ("direct_recall_agent", "os.path.join", "code_symbol"), ("direct_recall_user", "7:40 ferry", "phrase"),
            ("entity_attribution", "Dana Whitfield", "entity"), ("entity_attribution", "post author", "role"),
            ("abstention", "anything", "abstention"), ("paraphrase_recall", "dropped by half", "phrase"),
        ]:
            self.assertEqual(gq.answer_type(cat, answer, s), want, answer)

    def test_difficulty_comes_from_where_bm25_ranks_the_gold(self):
        recs = toolboxes(20)
        qs = [{"question": "Where does toolbox number 1003 stay?", "gold_ids": [recs[3]["id"]]},
              {"question": "Which colour suits dawn skies?", "gold_ids": [recs[4]["id"]]},
              {"question": "Is anything known?", "gold_ids": []}]
        self.assertEqual(gq.difficulties(qs, recs, 5), ["easy", "hard", "medium"])
        second = {"question": "Is toolbox number 1005 in cabinet xqaf, or toolbox 1003?", "gold_ids": [recs[3]["id"]]}
        self.assertEqual(gq.difficulties([second], recs, 5), ["medium"])  # in the window, but not first

    def test_the_freeze_orders_by_category_then_by_generation(self):
        dru_calls = []

        def writer(category, units):
            if category == "direct_recall_user":
                dru_calls.append(1)
                if len(dru_calls) == 1:
                    return []  # its first call gives nothing, so vocabulary_gap freezes first
            return sorted(toolbox_writer(category, units), key=lambda it: it["question"], reverse=True)

        with mock.patch.object(gq, "MAX_SAMPLE_SHARE", 1.0):
            res = self.gen(toolboxes(24), "direct_recall_user=2,vocabulary_gap=2", model=FakeModel(writer))
        frozen = {c["question_id"]: c for c in res["candidates"] if c["fate"] == "frozen"}
        order = [frozen[q["id"]] for q in res["questions"]]
        self.assertEqual([c["category"] for c in order], ["direct_recall_user"] * 2 + ["vocabulary_gap"] * 2)
        self.assertGreater(order[0]["n"], order[3]["n"])  # generated later, yet first: category order wins
        for a, b in (order[0:2], order[2:4]):
            self.assertLess(a["n"], b["n"])  # then generation order,
            self.assertGreater(a["returned"]["question"], b["returned"]["question"])  # not the question's text


class StoreIdTests(GenTestCase):
    def test_store_ids_generate_with_the_flag_and_are_refused_without(self):
        recs = [dict(toolbox(i), id=f"m-{i:04d}") for i in range(12)]
        res = self.gen(recs, "direct_recall_user=2", store_ids=True)
        m = res["manifest"]
        self.assertEqual(m["id_scheme"], "store")
        self.assertIn("store ids (private set)", m["status_reason"])
        self.assertTrue(res["check"]["ok"])
        self.assertTrue(all(g.startswith("m-") for q in res["questions"] for g in q["gold_ids"]))
        self.assertEqual(res["report"]["id_scheme"], "store")
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(recs, "direct_recall_user=2", out="run2/set-b")
        self.assertIn("--store-ids", str(cm.exception))


class OutTests(GenTestCase):
    def test_where_a_set_may_be_written(self):
        with self.assertRaises(gq.GenerateError) as cm:
            gq.check_out(os.path.join(REPO, "generated-set"), False)
        self.assertIn("inside the repository", str(cm.exception))
        data_path = os.path.join(REPO, "data", "v99-test")
        with self.assertRaises(gq.GenerateError):
            gq.check_out(data_path, False)
        self.assertEqual(gq.check_out(data_path, True), (os.path.realpath(data_path), "v99-test"))
        self.assertFalse(os.path.exists(data_path))
        with self.assertRaises(gq.GenerateError):
            gq.check_out(os.path.join(self.tmp, "elsewhere"), True)  # --release is only for data/
        full = os.path.join(self.tmp, "full")
        os.makedirs(full)
        open(os.path.join(full, "x"), "w").close()
        with self.assertRaises(gq.GenerateError) as cm:
            gq.check_out(full, False)
        self.assertIn("not an empty folder", str(cm.exception))
        with self.assertRaises(gq.GenerateError):
            gq.check_out(os.path.join(self.tmp, "bad name"), False)
        link = os.path.join(self.tmp, "link")
        try:
            os.symlink(REPO, link)
        except (OSError, NotImplementedError):  # Windows refuses symlinks without developer mode or admin rights
            self.skipTest("no symlinks here")
        with self.assertRaises(gq.GenerateError):
            gq.check_out(os.path.join(link, "through-a-link"), False)


class ReplayTests(GenTestCase):
    TARGETS = "direct_recall_user=2,vocabulary_gap=1,abstention=1"

    def test_replay_rebuilds_the_set_byte_for_byte(self):
        corpus = self.corpus(toolboxes(24))
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-r"), 7, gq.parse_targets(self.TARGETS),
                                model=FakeModel(), quiet=True)
        calls = os.path.join(first["out"], "generation", "calls.jsonl")
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-r"), 7, gq.parse_targets(self.TARGETS),
                                 replay=calls, quiet=True)
        files = sorted(os.path.relpath(os.path.join(d, f), first["out"])
                       for d, _, names in os.walk(first["out"]) for f in names)
        self.assertIn("reference.json", files)
        for name in files:
            with open(os.path.join(first["out"], name), "rb") as a, open(os.path.join(second["out"], name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_the_manifest_says_when_the_calls_started_and_finished_and_replay_keeps_both(self):
        class Clocked(FakeModel):
            def __init__(self):
                super().__init__()
                self.n = 0

            def call(self, category, prompt, user, request_sha):
                reply = super().call(category, prompt, user, request_sha)
                self.n += 1
                reply.started, reply.time = f"2026-10-20T00:00:{2 * self.n:02d}Z", f"2026-10-20T00:00:{2 * self.n + 1:02d}Z"
                return reply

        corpus = self.corpus(toolboxes(24))
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-r"), 7, gq.parse_targets(self.TARGETS),
                                model=Clocked(), quiet=True)
        calls = os.path.join(first["out"], "generation", "calls.jsonl")
        with open(calls, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        m = first["manifest"]["model"]
        self.assertEqual((m["started"], m["finished"]), (rows[0]["started"], rows[-1]["time"]))
        self.assertEqual(m["started"], "2026-10-20T00:00:02Z")
        self.assertLess(m["started"], m["finished"])
        second = gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-r"), 7, gq.parse_targets(self.TARGETS),
                                 replay=calls, quiet=True)
        self.assertEqual(second["manifest"]["model"], m)

    def test_replay_refuses_a_different_run(self):
        corpus = self.corpus(toolboxes(24))
        first = gq.generate_set(corpus, os.path.join(self.tmp, "a", "set-r"), 7, gq.parse_targets(self.TARGETS),
                                model=FakeModel(), quiet=True)
        with self.assertRaises(gq.GenerateError) as cm:
            gq.generate_set(corpus, os.path.join(self.tmp, "b", "set-r"), 8, gq.parse_targets(self.TARGETS),
                            replay=os.path.join(first["out"], "generation", "calls.jsonl"), quiet=True)
        self.assertIn("holds no response", str(cm.exception))


class OracleTests(GenTestCase):
    def frozen_set(self):
        res = self.gen(toolboxes(20), "direct_recall_user=2")
        return res, os.path.join(res["out"], "questions.json"), os.path.join(res["out"], "corpus.jsonl")

    def rewrite(self, q_path, change):
        with open(q_path, encoding="utf-8") as f:
            qs = json.load(f)
        change(qs)
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(qs, f)
        return qs

    def test_a_planted_bad_gold_id_breaks_the_set_and_is_named(self):
        _, q_path, c_path = self.frozen_set()

        def plant(qs):
            qs[1]["gold_ids"] = ["0123456789abcdef"]
            qs[1]["gold_hashes"] = {"0123456789abcdef": "sha256:" + "0" * 64}

        qs = self.rewrite(q_path, plant)
        check = gq.oracle_check(q_path, c_path, False)
        self.assertFalse(check["ok"])
        self.assertEqual([m["id"] for m in check["missed"]], [qs[1]["id"]])
        self.assertIn("not in the pool", check["missed"][0]["reason"])

    def test_the_reference_runs_leave_recall_qfile_as_they_found_it(self):
        _, q_path, c_path = self.frozen_set()
        with mock.patch.dict(os.environ, {"RECALL_QFILE": "before"}):
            self.assertTrue(gq.oracle_check(q_path, c_path, False)["ok"])
            self.assertEqual(os.environ["RECALL_QFILE"], "before")
        with mock.patch.dict(os.environ):
            os.environ.pop("RECALL_QFILE", None)
            self.assertTrue(gq.oracle_check(q_path, c_path, False)["ok"])
            self.assertNotIn("RECALL_QFILE", os.environ)

    def test_a_repeated_question_text_is_named(self):
        _, q_path, c_path = self.frozen_set()
        qs = self.rewrite(q_path, lambda qs: qs[1].__setitem__("question", qs[0]["question"]))
        check = gq.oracle_check(q_path, c_path, False)
        self.assertFalse(check["ok"])
        self.assertEqual({m["id"] for m in check["missed"]}, {qs[0]["id"], qs[1]["id"]})

    def test_a_failed_check_marks_the_set_broken_and_exits_1(self):
        res, _, _ = self.frozen_set()
        calls = os.path.join(res["out"], "generation", "calls.jsonl")
        out = os.path.join(self.tmp, "run2", "set-a")
        failed = {"ok": False, "missed": [{"id": "q_0002", "reason": "planted"}], "gold_at_window": 50.0,
                  "verified": "yes", "reference": None}
        with mock.patch.object(gq, "oracle_check", return_value=failed), \
                contextlib.redirect_stdout(io.StringIO()):
            code = gq.main(["--corpus", os.path.join(self.tmp, "corpus.jsonl"), "--out", out, "--seed", "7",
                            "--targets", "direct_recall_user=2", "--replay", calls])
        self.assertEqual(code, 1)
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["status"], "broken")
        self.assertIn("q_0002 (planted)", m["status_reason"])


class CLITests(GenTestCase):
    def test_targets(self):
        self.assertEqual(set(gq.parse_targets("3").values()), {3})
        t = gq.parse_targets("direct_recall_user=5, abstention=2")
        self.assertEqual((t["direct_recall_user"], t["abstention"], t["vocabulary_gap"]), (5, 2, 0))
        for bad in ("five", "direct_recall_user=x", "nonsense=3", "direct_recall_user"):
            with self.assertRaises(gq.GenerateError):
                gq.parse_targets(bad)
        corpus = self.corpus(toolboxes(4))
        for targets in ("nonsense=1", "0"):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gq.main(["--corpus", corpus, "--out", os.path.join(self.tmp, "o"), "--seed", "1",
                                          "--targets", targets]), 2)

    def test_a_live_run_needs_the_sdk_only_when_it_calls(self):
        model = gq.AnthropicModel(gq.DEFAULT_MODEL)  # constructing it imports nothing
        prompt = gq.load_prompts(["direct_recall_user"])["direct_recall_user"]
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(gq.GenerateError) as cm:
                model.call("direct_recall_user", prompt, "{}", "sha256:x")
        self.assertIn("pip install anthropic", str(cm.exception))
        self.assertEqual(gq.DEFAULT_MODEL, "claude-sonnet-5-5")

    def test_a_missing_sdk_stops_a_live_run_before_anything_is_written(self):
        out = os.path.join(self.tmp, "live", "set-l")
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(gq.GenerateError) as cm:
                gq.generate_set(self.corpus(toolboxes(8)), out, 1, gq.parse_targets("direct_recall_user=1"), quiet=True)
        self.assertIn("pip install anthropic", str(cm.exception))
        self.assertFalse(os.path.exists(out))

    def test_a_failed_first_call_leaves_no_folder_and_a_later_one_says_it_is_not_a_set(self):
        class Failing(FakeModel):
            def __init__(self, fail_at):
                super().__init__()
                self.fail_at, self.n = fail_at, 0

            def call(self, category, prompt, user, request_sha):
                self.n += 1
                if self.n == self.fail_at:  # a missing key, say, which the SDK reports on the first request
                    raise gq.GenerateError(f"the {category} call failed: AuthenticationError: invalid key")
                return super().call(category, prompt, user, request_sha)

        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(30), "direct_recall_user=12,paraphrase_recall=12", model=Failing(1))
        self.assertNotIn("is not a question set", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "run1", "set-a")))
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(30), "direct_recall_user=12,paraphrase_recall=12", model=Failing(2), out="run2/set-a")
        self.assertIn("is not a question set", str(cm.exception))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "run2", "set-a", "generation", "calls.jsonl")))

    def test_the_api_request_has_the_documented_shape(self):
        sent = {}

        class APIError(Exception):
            pass

        class Messages:
            def create(self, **kwargs):
                sent.update(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="thinking", thinking=""),
                             types.SimpleNamespace(type="text", text='{"items": []}')],
                    model="claude-sonnet-5-5", stop_reason="end_turn",
                    usage=types.SimpleNamespace(input_tokens=11, output_tokens=5))

        class Client:
            def __init__(self):
                self.messages = Messages()

        sdk = types.SimpleNamespace(Anthropic=Client, APIError=APIError, AnthropicError=Exception, __version__="9.9.9")
        prompt = gq.load_prompts(["knowledge_update"])["knowledge_update"]
        with mock.patch.dict(sys.modules, {"anthropic": sdk}):
            model = gq.AnthropicModel("claude-sonnet-5-5")
            reply = model.call("knowledge_update", prompt, "PAIR 1", "sha256:x")
            self.assertEqual(set(sent), {"model", "max_tokens", "system", "messages", "output_config"})  # no sampling
            self.assertEqual((sent["model"], sent["max_tokens"], sent["messages"]),
                             ("claude-sonnet-5-5", 16000, [{"role": "user", "content": "PAIR 1"}]))
            # the system prompt is one block, cached for an hour (generator v2)
            self.assertEqual(sent["system"], [{"type": "text", "text": prompt.text,
                                               "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
            self.assertEqual(sent["output_config"],
                             {"effort": "high", "format": {"type": "json_schema", "schema": prompt.schema}})
            self.assertEqual((reply.text, reply.model, reply.stop_reason, reply.input_tokens, reply.output_tokens,
                              reply.sdk), ('{"items": []}', "claude-sonnet-5-5", "end_turn", 11, 5, "anthropic 9.9.9"))

            def fails(**kwargs):
                raise APIError("overloaded")

            model._client.messages.create = fails
            with self.assertRaises(gq.GenerateError) as cm:
                model.call("knowledge_update", prompt, "PAIR 1", "sha256:x")
            self.assertIn("APIError: overloaded", str(cm.exception))

    def test_records_need_a_text_and_valid_unicode_before_any_call(self):
        model = FakeModel()
        model.call = mock.Mock(side_effect=AssertionError("no call may be made"))
        recs = [dict(toolbox(i), id=f"m-{i}") for i in range(6)] + [{"id": "m-x", "text": None}]
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(recs, "direct_recall_user=1", model=model, store_ids=True)
        self.assertIn("text must be a non-empty string", str(cm.exception))
        broken = os.path.join(self.tmp, "broken.jsonl")
        with open(broken, "w", encoding="utf-8") as f:
            for r in toolboxes(6) + [rec("A cut emoji \ud83d sits in this otherwise ordinary note")]:
                f.write(json.dumps(r) + "\n")  # ensure_ascii keeps the lone surrogate as an escape
        with self.assertRaises(gq.GenerateError) as cm:
            gq.generate_set(broken, os.path.join(self.tmp, "o", "set"), 1, gq.parse_targets("direct_recall_user=1"),
                            model=model, quiet=True)
        self.assertIn("not valid Unicode", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))

    def test_a_set_name_with_a_newline_is_refused(self):
        with self.assertRaises(gq.GenerateError):
            gq.check_out(os.path.join(self.tmp, "set-a\n"), False)

    def test_refusals_and_unreadable_replies_give_no_items(self):
        with self.assertRaises(gq.GenerateError) as cm:
            self.gen(toolboxes(12), "direct_recall_user=1", model=FakeModel(stop_reason="refusal"))
        self.assertIn("no item survived", str(cm.exception))
        with open(os.path.join(self.tmp, "run1", "set-a", "generation", "calls.jsonl"), encoding="utf-8") as f:
            self.assertEqual({json.loads(line)["stop_reason"] for line in f}, {"refusal"})
        with self.assertRaises(gq.GenerateError):
            self.gen(toolboxes(12), "direct_recall_user=1", model=FakeModel(text="not json"), out="run2/set-a")


if __name__ == "__main__":
    unittest.main()

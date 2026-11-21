"""Tests for generate_pool.py and show_pool.py with a faked model. Standard library only;
no network and no anthropic package:

    python3 -m unittest discover -s tests
"""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import types
import unittest
from collections import Counter
from datetime import date, datetime
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from parsica_recall import generate_pool as gp  # noqa: E402
from parsica_recall import generate_questions as gq  # noqa: E402
from parsica_recall import recall_common as rc  # noqa: E402
from parsica_recall import run_benchmark as rb  # noqa: E402
from parsica_recall import show_pool as sp  # noqa: E402

TIME = "2026-10-20T00:00:00Z"
# A made-up address for the screen tests, assembled from parts so no whole one sits in this file.
BILLING_MAIL = "billing.team" + "@" + "example.com"
LETTERS = "bcdfghjklmnpqrstvwxz"


def code(*parts):
    """A word of letters only, unique to its parts."""
    h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)
    out = ""
    for _ in range(7):
        h, r = divmod(h, len(LETTERS))
        out += LETTERS[r]
    return "q" + out


def persona_json(pid, **over):
    d = {
        "id": pid, "name": f"Person {pid.upper()}", "situation": f"An invented person {pid}.",
        "domains": ["cooking", "work"], "agent_role": "A helpful assistant.", "voice": "Plain.",
        "agent_voice": "Brief.",
        "people": [{"name": f"Friend {pid.upper()}", "relation": "friend"}],
        "things": ["Duluth"],
        "timeline": {"start": "2025-09-01", "months": 6},
        "cadence": {"description": "Weekdays.", "weekday_weights": [1, 1, 1, 1, 1, 0, 0], "local_hours": [8, 20],
                    "utc_offset_hours": -5},
        "records_per_session": [4, 8],
        "core_quotas": {"version_chains": 2, "preference_changes": 1, "corrections": 1, "threads": 1,
                        "long_horizon_facts": 1, "shown_preferences": 1},
    }
    for k, v in over.items():
        d[k] = v
    return d


def write_scenarios(folder, n=2, **over):
    os.makedirs(folder, exist_ok=True)
    for i in range(1, n + 1):
        with open(os.path.join(folder, f"p{i}_test.json"), "w", encoding="utf-8") as f:
            json.dump(persona_json(f"p{i}", **over), f, indent=2)
    return folder


class FakeModel:
    """Writes each session from its own request: the asked number of records and
    entries, with words unique to the session, so nothing is a near copy and the tail
    shares no vocabulary with the core. A subclass changes records() to plant one."""

    def __init__(self):
        self.calls = 0
        self.users = []

    def records(self, msg, phase):
        sid = msg["session"]["id"]
        n, e = msg["write"]["records"], msg["write"]["entries"]
        out = []
        for i in range(n - e):
            a, b, c, d, e_ = (code(sid, i, x) for x in "abcde")
            if phase == "core":  # under 60% template words, so no record restates another
                text = f"Logged {a} {b} beside {c} {d} today with {i + 3} {e_}."
            else:
                text = f"Tracked {a} {b} through {c} {d} yesterday {e_}."
            out.append({"kind": ("user", "agent")[i % 2], "text": text})
        for j in range(e):
            words = " ".join(code(sid, "entry", j, w) for w in range(110))
            out.append({"kind": "entry", "text": f"Notes {words}."})
        return out

    def reply(self, msg, phase):
        events = []
        for i, d in enumerate(msg["must_include"]):  # step i rides on record i
            label = re.match(r"[A-Z]\d+", d).group(0)
            events.append({"label": label, "record": i, "subject": f"subject of {label}",
                           "value": f"value in {msg['session']['id']}", "note": ""})
        recs = self.records(msg, phase)
        return {"records": recs, "state_update": {"new_people": [], "facts": [f"fact {i}" for i in range(len(recs))],
                                                  "events": events}}

    def text(self, msg, phase):
        return json.dumps(self.reply(msg, phase))

    def call(self, label, prompt, user, request_sha, **kw):
        self.calls += 1
        self.users.append(user)
        msg = json.loads(user)
        return gq.Reply(self.text(msg, prompt.phase), "claude-sonnet-5-5", "end_turn", len(user) // 4, 100, TIME,
                        None, TIME)


class FailAfter(FakeModel):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def call(self, label, prompt, user, request_sha, **kw):
        if self.calls >= self.n:
            raise gq.GenerateError(f"the {label} call failed: APIConnectionError: simulated")
        return super().call(label, prompt, user, request_sha, **kw)


def files_of(folder):
    out = {}
    for root, _, names in os.walk(folder):
        for n in names:
            p = os.path.join(root, n)
            with open(p, "rb") as f:
                out[os.path.relpath(p, folder)] = f.read()
    return out


def corpus_of(folder):
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def calls_of(out):
    """Every recorded call of a pool folder, all parts in order."""
    rows = []
    for path in gp.call_parts(os.path.join(out, "generation")):
        rows += jsonl(path)
    return rows


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pool-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.scen = write_scenarios(os.path.join(self.tmp, "scenarios"))

    def run_pool(self, name="pool-a", parent=None, model=None, core=40, total=100, scen=None, **kw):
        out = os.path.join(parent or self.tmp, name)
        model = model if model is not None else FakeModel()
        res = gp.generate_pool(scen or self.scen, out, 7, core=core, total=total, model=model, quiet=True, **kw)
        return res, out, model


# ---------------------------------------------------------------------------

class PlanTests(PoolTest):
    def persona(self, **over):
        path = os.path.join(self.tmp, "one.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(persona_json("p1", **over), f)
        return gp.Persona(path)

    def test_sizes_sum_to_the_budget_and_stay_in_range(self):
        for budget in (1, 5, 37, 125, 300):
            rng = gp._rng(1, "x", budget)
            sizes = gp.session_sizes(rng, budget, 6, 14)
            self.assertEqual(sum(sizes), budget)
            self.assertTrue(all(s <= 14 for s in sizes), sizes)
            if budget >= 6:
                self.assertTrue(all(s >= 6 for s in sizes), sizes)

    def test_the_plan_is_deterministic_and_differs_by_seed(self):
        p = self.persona(timeline={"start": "2025-09-01", "months": 12}, records_per_session=[6, 14])
        q = gp.scaled_quotas(p, 125)

        def plan(seed):
            s, _ = gp.plan_phase(seed, p, "core", 125, set(), {}, q)
            return [x.plan_json() for x in s]

        self.assertEqual(plan(1), plan(1))
        self.assertNotEqual(plan(1), plan(2))

    def test_directives_follow_their_rules(self):
        p = self.persona(timeline={"start": "2025-09-01", "months": 12}, records_per_session=[6, 14],
                         core_quotas={"version_chains": 6, "preference_changes": 5, "corrections": 6, "threads": 5,
                                      "long_horizon_facts": 5, "shown_preferences": 5})
        q = gp.scaled_quotas(p, 125)
        for seed in range(20):
            sessions, lost = gp.plan_phase(seed, p, "core", 125, set(), {}, q)
            self.assertEqual(lost, [])
            n = len(sessions)
            where = {}
            for i, s in enumerate(sessions):
                self.assertLessEqual(len(s.directives), s.records)
                for d in s.directives:
                    where.setdefault(d["label"], []).append((d["step"], i, d["of"]))
            self.assertEqual(sum(1 for lab in where if lab[0] == "V"), 6)
            for label, steps in where.items():
                steps.sort()
                positions = [i for _, i, _ in steps]
                self.assertEqual([s for s, _, _ in steps], list(range(1, len(steps) + 1)))
                self.assertEqual(positions, sorted(set(positions)), label)  # one step per session, in order
                if label[0] == "L":
                    self.assertEqual(len(steps), 1)
                    self.assertLess(positions[0], max(1, (n + 3) // 4))
                elif label[0] == "C":
                    self.assertEqual(len(steps), 2)
                    self.assertIn(positions[1] - positions[0], (1, 2, 3))
                else:
                    self.assertGreaterEqual(len(steps), 2)
                if label[0] in "VR":
                    self.assertLess(positions[0], max(1, n // 2))

    def test_sessions_fall_on_weighted_days_inside_the_timeline(self):
        p = self.persona(cadence={"description": "x", "weekday_weights": [0, 0, 1, 0, 0, 0, 0], "local_hours": [8, 9],
                                  "utc_offset_hours": -5})
        sessions, _ = gp.plan_phase(3, p, "core", 60, set(), {})
        for s in sessions:
            self.assertEqual(s.local.weekday(), 2)
            self.assertTrue(p.start <= s.local.date() < p.end)
            self.assertEqual(s.local.hour, 8)
            self.assertEqual(s.start, s.local.replace() + (datetime(2000, 1, 1, 5) - datetime(2000, 1, 1)))

    def test_session_ids_are_opaque_and_unique(self):
        p = self.persona()
        used = set()
        core, _ = gp.plan_phase(1, p, "core", 60, used, {})
        tail, _ = gp.plan_phase(1, p, "tail", 200, used, {})
        ids = [s.id for s in core + tail]
        self.assertEqual(len(ids), len(set(ids)))
        for sid in ids:
            self.assertRegex(sid, r"^p1-s[0-9a-f]{6}$")

    def test_the_core_plan_does_not_depend_on_the_tail(self):
        p = self.persona()
        a, _ = gp.plan_phase(1, p, "core", 60, set(), {})
        used, taken = set(), {}
        b, _ = gp.plan_phase(1, p, "core", 60, used, taken)
        gp.plan_phase(1, p, "tail", 500, used, taken)
        self.assertEqual([s.plan_json() for s in a], [s.plan_json() for s in b])

    def test_quotas_scale_with_the_share(self):
        p = self.persona(core_quotas={"version_chains": 6, "preference_changes": 5, "corrections": 0, "threads": 5,
                                      "long_horizon_facts": 5, "shown_preferences": 5})
        self.assertEqual(gp.scaled_quotas(p, 125)["version_chains"], 6)
        self.assertEqual(gp.scaled_quotas(p, 150)["version_chains"], 7)   # 7.2
        self.assertEqual(gp.scaled_quotas(p, 150)["preference_changes"], 6)  # 6.0
        self.assertEqual(gp.scaled_quotas(p, 10)["version_chains"], 1)    # at least one
        self.assertEqual(gp.scaled_quotas(p, 150)["corrections"], 0)

    def test_quotas_too_big_for_the_share_are_refused(self):
        p = self.persona(core_quotas={"version_chains": 40, "preference_changes": 0, "corrections": 0, "threads": 0,
                                      "long_horizon_facts": 0, "shown_preferences": 0})
        with self.assertRaisesRegex(gq.GenerateError, "quotas need more records"):
            gp.plan_phase(1, p, "core", 30, set(), {}, {**gp.scaled_quotas(p, 125), "version_chains": 40})

    def test_split_evenly(self):
        self.assertEqual(gp.split_evenly(1000, 8), [125] * 8)
        self.assertEqual(gp.split_evenly(10, 3), [4, 3, 3])
        self.assertEqual(gp.split_evenly(0, 2), [0, 0])


    def test_rewording_a_persona_never_moves_its_sessions(self):
        a = self.persona()
        b = self.persona(situation="Someone else entirely.", voice="Loud.", agent_voice="Wordy.",
                         people=[{"name": "Other", "relation": "cousin"}],
                         cadence={**persona_json("p1")["cadence"], "description": "Another description."})
        self.assertNotEqual(a.sha256, b.sha256)
        self.assertEqual(a.plan_key, b.plan_key)
        plan = lambda p: [s.plan_json() for s in gp.plan_phase(1, p, "core", 60, set(), {}, gp.scaled_quotas(p, 60))[0]]  # noqa: E731
        self.assertEqual(plan(a), plan(b))
        c = self.persona(records_per_session=[5, 9])
        self.assertNotEqual(a.plan_key, c.plan_key)


class PersonaFileTests(PoolTest):
    def test_the_eight_shipped_personas_load(self):
        ps = gp.load_personas(gp.SCENARIOS_DIR)
        self.assertEqual([p.id for p in ps], [f"p{i}" for i in range(1, 9)])
        self.assertEqual(len({p.name for p in ps}), 8)
        for p in ps:
            self.assertTrue(6 <= p.months <= 18)
            self.assertLessEqual(p.end, date(2026, 10, 20))
            sessions, lost = gp.plan_phase(1, p, "core", 125, set(), {}, gp.scaled_quotas(p, 125))
            self.assertEqual(lost, [])
            self.assertEqual(sum(s.records for s in sessions), 125)
            sessions, lost = gp.plan_phase(1, p, "core", 150, set(), {}, gp.scaled_quotas(p, 150))
            self.assertEqual(lost, [])

    def test_the_shipped_files_have_no_secret_or_personal_data_shapes(self):
        from parsica_recall import lint_validator as lv
        for name in sorted(os.listdir(gp.SCENARIOS_DIR)):
            with open(os.path.join(gp.SCENARIOS_DIR, name), encoding="utf-8") as f:
                self.assertEqual(lv.scan_secrets_pii(f.read()), [], name)

    def test_a_bad_persona_file_is_refused_with_its_name(self):
        bad = [({"id": "x1"}, "id must look like"), ({"timeline": {"start": "2025-09-01", "months": 3}}, "6 to 18"),
               ({"records_per_session": [2, 8]}, "records_per_session"),
               ({"core_quotas": {"version_chains": 1}}, "core_quotas"),
               ({"people": [{"name": "A"}]}, "people")]
        for over, msg in bad:
            folder = os.path.join(self.tmp, "bad" + str(abs(hash(msg))))
            write_scenarios(folder, 1, **over)
            with self.assertRaisesRegex(gq.GenerateError, msg):
                gp.load_personas(folder)

    def test_duplicate_and_unknown_ids_are_refused(self):
        folder = write_scenarios(os.path.join(self.tmp, "dup"), 1)
        shutil.copy(os.path.join(folder, "p1_test.json"), os.path.join(folder, "p1_copy.json"))
        with self.assertRaisesRegex(gq.GenerateError, "both use the id p1"):
            gp.load_personas(folder)
        with self.assertRaisesRegex(gq.GenerateError, "p9"):
            gp.load_personas(self.scen, ["p9"])
        self.assertEqual([p.id for p in gp.load_personas(self.scen, ["p2"])], ["p2"])

    def test_crlf_persona_and_prompt_files_hash_like_lf(self):
        lf = gp.Persona(os.path.join(self.scen, "p1_test.json"))
        crlf_path = os.path.join(self.tmp, "crlf.json")
        with open(crlf_path, "wb") as f:
            f.write(lf.data.replace(b"\n", b"\r\n"))
        self.assertEqual(gp.Persona(crlf_path).sha256, lf.sha256)
        folder = os.path.join(self.tmp, "prompts-crlf")
        os.makedirs(folder)
        for phase, name in gp.PROMPT_FILES.items():
            with open(os.path.join(gp.PROMPTS_DIR, name), "rb") as f:
                data = gq._lf(f.read())
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data.replace(b"\n", b"\r\n"))
        a, b = gp.load_prompts(), gp.load_prompts(folder)
        for phase in gp.PHASES:
            self.assertEqual(a[phase].sha256, b[phase].sha256)
            self.assertEqual(a[phase].schema, b[phase].schema)

    def test_the_shipped_prompts_carry_their_schema(self):
        for phase, p in gp.load_prompts().items():
            self.assertEqual(set(p.schema["properties"]), {"records", "state_update"})
            self.assertEqual(p.schema["properties"]["records"]["items"]["properties"]["kind"]["enum"],
                             list(gp.KINDS))


class RunTests(PoolTest):
    def test_a_run_writes_an_exact_core_then_the_tail(self):
        res, out, model = self.run_pool()
        recs = corpus_of(out)
        self.assertEqual(len(recs), 100)
        where = jsonl(os.path.join(out, "generation", "records.jsonl"))
        self.assertEqual([w["phase"] for w in where], ["core"] * 40 + ["tail"] * 60)
        self.assertEqual([r["ts"] for r in recs[:40]], sorted(r["ts"] for r in recs[:40]))
        self.assertEqual(Counter(w["persona"] for w in where[:40]), {"p1": 20, "p2": 20})
        self.assertEqual(Counter(w["persona"] for w in where[40:]), {"p1": 30, "p2": 30})
        for r in recs:
            self.assertEqual(list(r), ["id", "text", "source", "kind", "ts"])
            self.assertEqual(r["id"], rc.record_id(r["text"]))
            self.assertRegex(r["source"], r"^p[12]-s[0-9a-f]{6}$")
            self.assertRegex(r["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            self.assertIn(r["kind"], gp.KINDS)
        self.assertEqual(len(rb.load_corpus(os.path.join(out, "corpus.jsonl"))), 100)
        m = res["manifest"]
        self.assertEqual(m["status"], "partial")
        self.assertEqual(m["layout"]["core"], {"first_line": 1, "last_line": 40,
                                               "order": "ts, then persona, session, position"})
        self.assertEqual(m["model"]["calls"], model.calls)

    def test_the_tail_order_is_a_proportional_seeded_interleave(self):
        res, out, _ = self.run_pool(core=40, total=240)
        where = jsonl(os.path.join(out, "generation", "records.jsonl"))[40:]
        recs = corpus_of(out)[40:]
        self.assertNotEqual([r["ts"] for r in recs], sorted(r["ts"] for r in recs))  # not time order
        groups = Counter(f"{w['persona']}|{r['ts'][:7]}" for w, r in zip(where, recs))
        seen = Counter()
        for i, (w, r) in enumerate(zip(where, recs), 1):
            seen[f"{w['persona']}|{r['ts'][:7]}"] += 1
            for g, size in groups.items():  # every prefix keeps each group's share, within one record
                self.assertLessEqual(abs(seen[g] - i * size / len(recs)), 1.0 + 1e-9)

    def test_the_core_does_not_depend_on_the_total(self):
        _, a, _ = self.run_pool("pool-x", parent=os.path.join(self.tmp, "a"), core=40, total=40)
        _, b, _ = self.run_pool("pool-x", parent=os.path.join(self.tmp, "b"), core=40, total=140)
        self.assertEqual(corpus_of(a), corpus_of(b)[:40])

    def test_same_seed_and_replies_give_the_same_bytes_with_any_workers(self):
        _, a, _ = self.run_pool("same", parent=os.path.join(self.tmp, "a"), workers=1)
        _, b, _ = self.run_pool("same", parent=os.path.join(self.tmp, "b"), workers=4)
        self.assertEqual(files_of(a), files_of(b))

    def test_another_seed_gives_another_pool(self):
        _, a, _ = self.run_pool("a")
        out = os.path.join(self.tmp, "b")
        gp.generate_pool(self.scen, out, 8, core=40, total=100, model=FakeModel(), quiet=True)
        self.assertNotEqual(corpus_of(a), corpus_of(out))

    def test_every_written_file_has_lf_line_endings(self):
        _, out, _ = self.run_pool()
        for name, data in files_of(out).items():
            self.assertNotIn(b"\r\n", data, name)

    def test_the_request_holds_persona_state_and_plan(self):
        _, _, model = self.run_pool(core=40, total=40)
        first = json.loads(model.users[0])
        self.assertEqual(list(first), ["persona", "session", "write", "must_include", "state"])
        self.assertEqual(set(first["persona"]), set(gp.PUBLIC_FIELDS))
        self.assertNotIn("core_quotas", json.dumps(first))
        self.assertEqual(first["state"]["facts_so_far"], [])
        later = [json.loads(u) for u in model.users[1:]]
        self.assertTrue(any(m["state"]["facts_so_far"] for m in later))
        self.assertTrue(any(m["state"]["open_items"] for m in later))
        step2 = [d for m in later for d in m["must_include"] if re.match(r"V\d+, step 2", d)]
        self.assertTrue(step2)
        self.assertIn("has changed since it was last given as", step2[0])

    def test_a_long_horizon_fact_goes_to_never_mention_again(self):
        _, _, model = self.run_pool(core=40, total=40)
        msgs = [json.loads(u) for u in model.users]
        self.assertTrue(any(m["state"]["never_mention_again"] for m in msgs))
        for m in msgs:
            self.assertFalse(any(k.startswith("L") for k in m["state"]["open_items"]))

    def test_the_tail_sees_the_core_facts_to_avoid(self):
        _, _, model = self.run_pool(core=40, total=60)
        tail = [json.loads(u) for u in model.users if '"core_facts_never_restate"' in u]
        self.assertTrue(tail)
        self.assertTrue(all(m["must_include"] == [] for m in tail))
        self.assertIn("fact 0", tail[0]["state"]["core_facts_never_restate"])

    def test_the_manifest_records_model_prompts_scenarios_and_plan(self):
        res, out, _ = self.run_pool()
        m = res["manifest"]
        self.assertEqual(m["model"]["requested"], gp.DEFAULT_MODEL)
        self.assertEqual(m["model"]["served"], ["claude-sonnet-5-5"])
        self.assertEqual(m["model"]["started"], TIME)
        self.assertEqual(set(m["prompts"]), {"core_session.md", "tail_session.md"})
        self.assertEqual(m["scenarios"]["p1"]["sha256"], gp.Persona(os.path.join(self.scen, "p1_test.json")).sha256)
        self.assertEqual(m["plan"]["core_shares"], {"p1": 20, "p2": 20})
        self.assertEqual(m["files"]["corpus.jsonl"], rc.file_sha256(os.path.join(out, "corpus.jsonl")))
        self.assertEqual(m["layout"]["tracks"]["1k"]["available"], False)
        self.assertEqual(m["tool"]["file"], "generate_pool.py")
        self.assertIn("Every person in it is invented", m["provenance"])
        with open(os.path.join(out, "generation", "prompts", "core_session.md"), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), m["prompts"]["core_session.md"]["sha256"])
        self.assertTrue(os.path.isfile(os.path.join(out, "generation", "scenarios", "p1_test.json")))
        with open(os.path.join(out, "generation", "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
        self.assertEqual(set(plan["personas"]), {"p1", "p2"})

    def test_a_failed_first_call_leaves_no_folder(self):
        out = os.path.join(self.tmp, "never")
        with self.assertRaisesRegex(gq.GenerateError, "simulated"):
            gp.generate_pool(self.scen, out, 7, core=40, total=40, model=FailAfter(0), quiet=True)
        self.assertFalse(os.path.exists(out))

    def test_core_and_total_are_checked(self):
        with self.assertRaisesRegex(gq.GenerateError, "smaller than --core"):
            gp.generate_pool(self.scen, os.path.join(self.tmp, "x"), 1, core=50, total=40, model=FakeModel())
        with self.assertRaisesRegex(gq.GenerateError, "at least 1"):
            gp.generate_pool(self.scen, os.path.join(self.tmp, "x"), 1, core=0, total=40, model=FakeModel())


    def test_record_text_is_stripped_before_its_id(self):
        class Padded(FakeModel):
            def records(self, msg, phase):
                return [{"kind": r["kind"], "text": "  " + r["text"] + "\n\n"} for r in super().records(msg, phase)]
        _, out, _ = self.run_pool(model=Padded(), core=40, total=40)
        for r in corpus_of(out):
            self.assertEqual(r["text"], r["text"].strip())
            self.assertEqual(r["id"], rc.record_id(r["text"]))

    def test_changes_and_corrections_draw_a_seeded_style_and_speaker(self):
        q = {"version_chains": 6, "preference_changes": 5, "corrections": 6, "threads": 0, "long_horizon_facts": 0,
             "shown_preferences": 0}
        scen = write_scenarios(os.path.join(self.tmp, "big"), 1, timeline={"start": "2025-09-01", "months": 12},
                               records_per_session=[6, 14], core_quotas=q)
        _, _, model = self.run_pool(core=125, total=125, scen=scen)
        steps = [d for u in model.users for d in json.loads(u)["must_include"]]
        corrections = [d for d in steps if re.match(r"C\d+, part 2", d)]
        changes = [d for d in steps if re.match(r"[VR]\d+, step [23]", d)]
        self.assertTrue(corrections and changes)
        used = {st for d in corrections for who in gp.CORRECTION_STYLES for st in gp.CORRECTION_STYLES[who] if st in d}
        self.assertGreater(len(used), 1)  # not one template
        self.assertGreater(len({st for d in changes for st in gp.CHANGE_STYLES if st in d}), 1)
        for d in corrections:  # the style fits whoever made the error in part 1
            erred = "the agent" if ", the agent gave " in d else "the person"
            self.assertTrue(any(st in d for st in gp.CORRECTION_STYLES[erred]), d)
            self.assertNotIn("which was wrong", d)
        part1 = {d.split(",")[0]: d for d in steps if re.match(r"C\d+, part 1", d)}
        for d in corrections:
            label = d.split(",")[0]
            speaker = "the agent" if "the agent states" in part1[label] else "the person"
            self.assertIn(f", {speaker} gave ", d)
        _, _, again = self.run_pool("again", core=125, total=125, scen=scen)
        self.assertEqual(model.users, again.users)  # the same seed draws the same styles

    def test_the_tail_sees_the_core_matters_to_keep_out_of(self):
        _, _, model = self.run_pool(core=40, total=60)
        tail = [json.loads(u) for u in model.users if '"core_matters_never_touch"' in u]
        self.assertTrue(tail)
        self.assertIn("subject of V1", tail[-1]["state"]["core_matters_never_touch"])


class ReplayResumeTests(PoolTest):
    def test_replay_rebuilds_the_same_bytes(self):
        _, a, model = self.run_pool("same", parent=os.path.join(self.tmp, "a"))
        b = os.path.join(self.tmp, "b", "same")
        gp.generate_pool(self.scen, b, 7, core=40, total=100, replay=os.path.join(a, "generation"),
                         quiet=True)
        self.assertEqual(files_of(a), files_of(b))

    def test_replay_names_a_request_it_does_not_hold(self):
        _, a, _ = self.run_pool("a")
        with self.assertRaisesRegex(gq.GenerateError, "holds no response"):
            gp.generate_pool(self.scen, os.path.join(self.tmp, "b"), 8, core=40, total=100,
                             replay=os.path.join(a, "generation"), quiet=True)

    def test_resume_picks_up_from_the_call_log_and_ends_the_same(self):
        _, full, whole = self.run_pool("same", parent=os.path.join(self.tmp, "full"), workers=1)
        out = os.path.join(self.tmp, "cut", "same")
        with self.assertRaisesRegex(gq.GenerateError, "--resume"):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(5), workers=1, quiet=True)
        self.assertEqual(len(calls_of(out)), 5)
        with open(os.path.join(out, "generation", "calls-0001.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"n": 6, "request_sha256": "sha256:cut sh')  # the line a stopped run was writing
        live = FakeModel()
        gp.generate_pool(self.scen, out, 7, core=40, total=100, model=live, workers=1, resume=True, quiet=True)
        self.assertEqual(live.calls, whole.calls - 5)
        self.assertEqual(files_of(full), files_of(out))

    def test_resume_refuses_other_settings_and_a_folder_with_no_run(self):
        out = os.path.join(self.tmp, "cut")
        with self.assertRaises(gq.GenerateError):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(3), quiet=True)
        with self.assertRaisesRegex(gq.GenerateError, "seed differ"):
            gp.generate_pool(self.scen, out, 8, core=40, total=100, model=FakeModel(), resume=True, quiet=True)
        with self.assertRaisesRegex(gq.GenerateError, "no generation/run.json"):
            gp.generate_pool(self.scen, os.path.join(self.tmp, "empty"), 7, model=FakeModel(), resume=True)

    def test_resume_may_grow_the_total(self):
        _, out, first = self.run_pool("grow", core=40, total=40)
        live = FakeModel()
        res = gp.generate_pool(self.scen, out, 7, core=40, total=100, model=live, resume=True, quiet=True)
        self.assertEqual(res["manifest"]["statistics"]["records"], 100)
        calls = calls_of(out)
        self.assertEqual(live.calls, sum(1 for c in calls if c["phase"] == "tail"))  # the core came from the log
        self.assertEqual(sum(1 for c in calls if c["phase"] == "core"), first.calls)
        self.assertEqual(corpus_of(out)[:40], res["records"] and corpus_of(out)[:40])


    def test_a_resume_that_fails_too_keeps_every_recorded_call(self):
        out = os.path.join(self.tmp, "cut")
        with self.assertRaises(gq.GenerateError):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(9), workers=4, quiet=True)
        log = os.path.join(out, "generation", "calls-0001.jsonl")
        before = jsonl(log)
        with self.assertRaisesRegex(gq.GenerateError, "--resume"):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(2), workers=1, resume=True,
                             quiet=True)
        after = jsonl(log)
        self.assertEqual(after[:len(before)], before)  # nothing recorded was lost
        self.assertEqual(len(after), len(before) + 2)

    def test_resume_after_a_cut_line_twice_over_ends_the_same(self):
        _, full, whole = self.run_pool("same", parent=os.path.join(self.tmp, "full"), workers=4)
        out = os.path.join(self.tmp, "cut", "same")
        log = os.path.join(out, "generation", "calls-0001.jsonl")
        with self.assertRaises(gq.GenerateError):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(4), workers=4, quiet=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write('{"n": 9, "request_sha256": "sha2')
        with self.assertRaises(gq.GenerateError):
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FailAfter(3), workers=4, resume=True,
                             quiet=True)
        jsonl(log)  # every line reads: the cut line was taken out before new calls were added
        gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FakeModel(), workers=4, resume=True,
                         quiet=True)
        self.assertEqual(files_of(full), files_of(out))


class Planted(FakeModel):
    """Plants records in chosen sessions: plant(msg, phase, records) returns the new list."""

    def __init__(self, plant):
        super().__init__()
        self.plant = plant
        self.seen = Counter()

    def records(self, msg, phase):
        recs = super().records(msg, phase)
        self.seen[(msg["persona"]["name"], phase)] += 1
        return self.plant(self, msg, phase, recs, self.seen[(msg["persona"]["name"], phase)])


def dropped_of(out):
    return jsonl(os.path.join(out, "generation", "dropped.jsonl"))


class ScreenTests(PoolTest):
    def run_planted(self, plant, **kw):
        kw.setdefault("core", 40)
        kw.setdefault("total", 40)
        res, out, model = self.run_pool(model=Planted(plant), workers=1, **kw)
        return res, out, dropped_of(out)

    def assert_exact(self, res, out, core, total):
        self.assertEqual(len(corpus_of(out)), total)
        self.assertEqual(res["manifest"]["statistics"]["core"], core)

    def test_a_secret_or_personal_data_record_is_dropped_and_its_text_not_kept(self):
        def plant(m, msg, phase, recs, k):
            if k == 1 and msg["persona"]["name"] == "Person P1":
                recs.insert(0, {"kind": "user", "text": "Send the invoice to " + BILLING_MAIL + " before Friday please."})
            return recs
        res, out, dropped = self.run_planted(plant)
        rows = [d for d in dropped if d["reason"] == "secret_or_pii"]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["text"])
        self.assertIsNone(rows[0]["id"])
        self.assertEqual(rows[0]["detail"], ["email"])
        for name, data in files_of(out).items():
            if not re.match(r"generation.calls-\d{4}\.jsonl$", name):  # the call log keeps the raw reply, as sent
                self.assertNotIn(BILLING_MAIL.encode("utf-8"), data, name)
        self.assert_exact(res, out, 40, 40)

    def test_short_records_and_entries_out_of_range_are_dropped(self):
        def plant(m, msg, phase, recs, k):
            if k == 1:
                recs.insert(0, {"kind": "agent", "text": "Okay, done."})
                recs.append({"kind": "entry", "text": "Short entry " + " ".join(code(msg["session"]["id"], "s", i)
                                                                             for i in range(20))})
            return recs
        res, out, dropped = self.run_planted(plant)
        reasons = Counter(d["reason"] for d in dropped)
        self.assertEqual(reasons["too_short"], 2)
        self.assertEqual(reasons["entry_length"], 2)
        self.assert_exact(res, out, 40, 40)

    def test_a_bad_record_shape_is_dropped(self):
        def plant(m, msg, phase, recs, k):
            if k == 1 and msg["persona"]["name"] == "Person P2":
                recs.insert(0, {"kind": "note", "text": "A record of the wrong kind with enough words in it."})
            return recs
        _, _, dropped = self.run_planted(plant)
        self.assertEqual([d["reason"] for d in dropped], ["bad_record_shape"])

    def test_a_duplicate_and_a_near_copy_within_a_persona_are_dropped(self):
        def plant(m, msg, phase, recs, k):
            if msg["persona"]["name"] == "Person P1":
                if k == 1:
                    m.keep = recs[0]["text"]
                elif k == 2:
                    recs.insert(0, {"kind": "agent", "text": m.keep.lower().replace("logged", "Logged")})
                    recs.insert(1, {"kind": "agent", "text": m.keep.replace("today", "today again")})
            return recs
        res, out, dropped = self.run_planted(plant)
        reasons = sorted(d["reason"] for d in dropped)
        self.assertEqual(reasons, ["duplicate_id", "near_copy"])
        self.assert_exact(res, out, 40, 40)

    def test_a_near_copy_across_personas_keeps_the_first_persona_only(self):
        shared = "Booked the qzzvwxkp ferry to Bainbridge for Saturday with 4 bikes and the cooler."

        def plant(m, msg, phase, recs, k):
            if k == 1:
                recs.insert(0, {"kind": "user", "text": shared})
            return recs
        res, out, dropped = self.run_planted(plant)
        kept = [r for r in corpus_of(out) if r["text"] == shared]
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0]["source"].startswith("p1-"))
        self.assertEqual([(d["reason"], d["persona"]) for d in dropped if d["reason"] != "over_budget"],
                         [("duplicate_id", "p2")])  # p2's make-up asks a whole small session; the rest is trimmed
        self.assert_exact(res, out, 40, 40)

    def test_a_tail_record_that_restates_a_core_record_is_dropped(self):
        def plant(m, msg, phase, recs, k):
            if phase == "core" and k == 1 and msg["persona"]["name"] == "Person P1":
                m.core_text = recs[0]["text"]
            if phase == "tail" and k == 2 and msg["persona"]["name"] == "Person P2":
                recs.insert(0, {"kind": "agent", "text": "Earlier note: " + m.core_text + " Also checked the "
                                + code("x") + " shelf."})
            return recs
        res, out, dropped = self.run_planted(plant, core=40, total=100)
        rows = [d for d in dropped if d["reason"] == "restates_core"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "tail")
        self.assert_exact(res, out, 40, 100)

    def test_a_short_session_is_made_up_and_the_core_stays_exact(self):
        def plant(m, msg, phase, recs, k):
            return recs[:max(1, len(recs) - 3)]  # every session comes back three records short
        res, out, _ = self.run_planted(plant, core=40, total=80)
        self.assert_exact(res, out, 40, 80)
        self.assertGreater(res["manifest"]["plan"]["makeup_sessions"], 0)

    def test_a_persona_over_its_share_drops_its_last_records(self):
        def plant(m, msg, phase, recs, k):
            return recs + [{"kind": "agent", "text": f"Extra record {code(msg['session']['id'], 'x', i)} filed "
                                                     f"under {code(msg['session']['id'], 'y', i)} for later."}
                           for i in range(3)]
        res, out, dropped = self.run_planted(plant)
        self.assert_exact(res, out, 40, 40)
        self.assertGreater(Counter(d["reason"] for d in dropped)["over_budget"], 0)

    def test_an_unusable_reply_is_asked_again(self):
        class Flaky(FakeModel):
            def text(self, msg, phase):
                if self.calls == 1:
                    return "not json"
                return super().text(msg, phase)
        res, out, model = self.run_pool(model=Flaky(), workers=1, core=40, total=40)
        calls = calls_of(out)
        self.assertEqual([c["attempt"] for c in calls[:2]], [1, 2])
        self.assertEqual(res["manifest"]["model"]["unusable_calls"], 1)
        self.assert_core(out, 40)

    def assert_core(self, out, n):
        self.assertEqual(len(corpus_of(out)), n)

    def test_a_session_whose_replies_all_fail_passes_its_plan_on(self):
        class Lost(FakeModel):
            def call(self, label, prompt, user, request_sha, **kw):
                self.calls += 1
                self.users.append(user)
                if self.calls <= gp.MAX_ATTEMPTS:
                    return gq.Reply("", "claude-sonnet-5-5", "refusal", 1, 0, TIME, None, TIME)
                return gq.Reply(self.text(json.loads(user), prompt.phase), "claude-sonnet-5-5", "end_turn", 1, 1,
                                TIME, None, TIME)
        res, out, model = self.run_pool(model=Lost(), workers=1, core=40, total=40)
        first = json.loads(model.users[0])["must_include"]
        nxt = json.loads(model.users[gp.MAX_ATTEMPTS])["must_include"]
        labels = lambda xs: [re.match(r"[A-Z]\d+", x).group(0) for x in xs]  # noqa: E731
        self.assertTrue(set(labels(first)) <= set(labels(nxt)))
        self.assertEqual(res["manifest"]["model"]["sessions_lost"], 1)
        self.assert_core(out, 40)


    def test_no_session_asks_for_more_than_the_largest_size(self):
        class Stingy(FakeModel):  # writes at most three records, whatever it is asked
            def records(self, msg, phase):
                return super().records(msg, phase)[:3]
        res, out, model = self.run_pool(model=Stingy(), workers=1, core=40, total=80)
        asks = [json.loads(u)["write"]["records"] for u in model.users]
        self.assertLessEqual(max(asks), 8)  # records_per_session [4, 8]
        self.assertEqual(len(corpus_of(out)), 80)

    def test_make_ups_spread_over_the_timeline_and_see_only_their_past(self):
        class Stingy(FakeModel):
            def records(self, msg, phase):
                return super().records(msg, phase)[:3]
        res, out, model = self.run_pool(model=Stingy(), workers=1, core=40, total=40)
        with open(os.path.join(out, "generation", "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
        for pid, p in plan["personas"].items():
            planned = [x["local"] for x in p["sessions"] if not x["makeup"]]
            makeups = [x["local"] for x in p["sessions"] if x["makeup"]]
            self.assertTrue(makeups)
            self.assertTrue(any(m < max(planned) for m in makeups), pid)  # not all after the last session
            for m in makeups:
                hour = int(m[11:13])
                self.assertTrue(8 <= hour < 20, m)  # inside the persona's local hours
        for u in model.users:
            msg = json.loads(u)
            day = msg["session"]["local_time"][:10]
            for line in msg["state"]["facts_so_far"]:
                self.assertLessEqual(line[:10], day)

    def test_secret_text_in_a_state_update_never_reaches_the_state(self):
        class Leaky(FakeModel):
            def reply(self, msg, phase):
                out = super().reply(msg, phase)
                out["state_update"]["facts"].append("Her email is maren.kowal" + "@" + "example.org")
                out["state_update"]["new_people"].append({"name": "Pat", "relation": "reach at pat" + "@" + "example.org"})
                return out
        res, out, model = self.run_pool(model=Leaky(), workers=1, core=40, total=40)
        with open(os.path.join(out, "generation", "states.json"), encoding="utf-8") as f:
            self.assertNotIn("example.org", f.read())
        self.assertFalse(any("example.org" in u for u in model.users))
        self.assertGreater(res["manifest"]["screens"]["state_lines_screened"], 0)
        self.assertGreater(res["manifest"]["screens"]["calls_with_secret_or_pii_shapes"], 0)

    def test_a_planted_step_whose_record_was_dropped_is_asked_again(self):
        def plant(m, msg, phase, recs, k):
            if phase == "core" and k == 1 and msg["persona"]["name"] == "Person P1":
                recs[0] = {"kind": "user", "text": "Mail the receipts to ops.desk" + "@" + "example.com by noon tomorrow."}
            return recs
        res, out, dropped = self.run_planted(plant)
        with open(os.path.join(out, "generation", "states.json"), encoding="utf-8") as f:
            state = json.load(f)["p1"]
        with open(os.path.join(out, "generation", "plan.json"), encoding="utf-8") as f:
            sessions = json.load(f)["personas"]["p1"]["sessions"]
        label = sessions[0]["directives"][0].split()[0]
        values = state["items"][label]["values"]
        self.assertNotIn(sessions[0]["session"], [v["session"] for v in values])
        self.assertGreater(res["manifest"]["plan"]["events_without_a_kept_record"], 0)

    def test_planted_steps_lost_at_the_end_of_the_core_never_reach_the_tail(self):
        class LastLost(FakeModel):
            def call(self, label, prompt, user, request_sha, **kw):
                msg = json.loads(user)
                if prompt.phase == "core" and msg["must_include"] and getattr(self, "lose", None) == msg["session"]["id"]:
                    self.calls += 1
                    self.users.append(user)
                    return gq.Reply("", "claude-sonnet-5-5", "refusal", 1, 0, TIME, None, TIME)
                return super().call(label, prompt, user, request_sha, **kw)
        probe = os.path.join(self.tmp, "probe")
        gp.generate_pool(self.scen, probe, 7, core=40, total=40, model=FakeModel(), quiet=True, personas=["p1"])
        with open(os.path.join(probe, "generation", "plan.json"), encoding="utf-8") as f:
            sessions = [x for x in json.load(f)["personas"]["p1"]["sessions"] if x["directives"]]
        model = LastLost()
        model.lose = sessions[-1]["session"]
        res = gp.generate_pool(self.scen, os.path.join(self.tmp, "lost"), 7, core=40, total=80, model=model,
                               quiet=True, personas=["p1"])
        msgs = [json.loads(u) for u in model.users]
        lost = msgs[[m["session"]["id"] for m in msgs].index(model.lose)]
        after = [m for m in msgs if m["session"]["local_time"] > lost["session"]["local_time"]
                 and '"core_facts_never_restate"' not in json.dumps(m)]
        self.assertTrue(after)  # the steps went to a core make-up dated after the lost session
        self.assertEqual([x[:2] for x in after[0]["must_include"]], [x[:2] for x in lost["must_include"]])
        tail = [m for m in msgs if "core_facts_never_restate" in m["state"]]
        self.assertTrue(tail and all(m["must_include"] == [] for m in tail))
        self.assertEqual(res["manifest"]["plan"]["directives_lost"], 0)

    def test_workers_never_change_the_bytes_with_copies_and_make_ups(self):
        shared = "Booked the qzzvwxkp ferry to Bainbridge for Saturday with 4 bikes and the cooler."

        def plant(m, msg, phase, recs, k):
            recs = recs[:max(1, len(recs) - 2)]
            if k == 1:
                recs.insert(0, {"kind": "user", "text": shared})
            return recs
        outs = []
        for w in (1, 4, 8):
            _, out, _ = self.run_pool("same", parent=os.path.join(self.tmp, f"w{w}"), model=Planted(plant), workers=w,
                                      scen=write_scenarios(os.path.join(self.tmp, f"s{w}"), 3))
            outs.append(files_of(out))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])


# Pilot 2's twelve restates_core drops were all one shape: a long tail entry (266 to 457
# tokens) covering 60% of a short core record (30 to 58 tokens) through stopwords and common
# words, with content-only overlap of 0.15 to 0.55. The zip never uploaded, so these are
# written to that shape, with the measured numbers checked in the test.
CORE_FURNACE = ("Swapped the furnace filter in the basement for a pleated MERV 11 one from Menards this morning, "
                "since the old filter was gray after only three months, and I told Ingrid we should check it again "
                "at the start of March.")
LONG_ENTRY = ("Session note. This morning was mostly about the kitchen, but a few things from the week are worth "
              "keeping in one place so that they do not get lost. Desmond started with the sourdough: the starter "
              "was sluggish after the cold snap, so we moved the jar to the top of the refrigerator where the "
              "compressor keeps it a little warmer, and he fed it twice at a one to five to five ratio. He also "
              "asked about the basement, where the dehumidifier has been running more than it should, and we "
              "looked at the drain line together; it was kinked behind the shelf, which explains the puddle he "
              "found on Sunday. After that we went back to the pantry inventory. The Ball jars from the tomato "
              "run are now on the lower shelf, sorted by date, and the three that did not seal went into the "
              "freezer for chili later in the month. He wants to try a rye loaf for the weekend, so I put the "
              "flour order on the list for Sunniva's shop, two bags of King Arthur bread flour and one of dark "
              "rye. We also talked about the porch light, which flickers again, and he thinks it is the fixture "
              "rather than the bulb; Wendell offered to look at it on Thursday. Before we finished he mentioned "
              "that Pieter is coming up from Minneapolis at the start of the month, so the guest room needs the "
              "space heater back from the garage, and the air mattress needs a new valve cap. I noted that the "
              "old filter question can wait, since that was handled, and that the next big job is the bathroom "
              "tile, which he wants to start after the holidays once the weather turns and the store has the "
              "grout he likes in stock again.")
PARAPHRASE = ("This morning I replaced the basement furnace filter with a pleated MERV 11 from Menards, because "
              "the old filter had turned gray after three months, and I told Ingrid we ought to check it again "
              "when March starts.")
SHORT_CORE = "Ordered the oak shelf from IKEA yesterday."


class RestateTests(PoolTest):
    def test_the_twelve_drop_shape_is_kept_and_a_paraphrase_is_dropped(self):
        core, entry, para = rc.tokenize(CORE_FURNACE), rc.tokenize(LONG_ENTRY), rc.tokenize(PARAPHRASE)
        self.assertTrue(30 <= len(core) <= 58 and 266 <= len(entry) <= 457)
        self.assertGreaterEqual(rc.token_overlap(core, entry), 0.6)  # the old screen dropped it
        content_cover = rc.token_overlap(gp.content_tokens(core), gp.content_tokens(entry))
        self.assertTrue(0.15 <= content_cover <= 0.55, content_cover)
        old = gp.CoverIndex(0.6)
        old.add("c", core)
        self.assertEqual(old.find(entry), "c")
        idx = gp.RestateIndex()
        idx.add("c", core)
        self.assertIsNone(idx.find(entry))            # a long entry over a short record is kept
        self.assertEqual(idx.find(para), "c")         # a same-length paraphrase is dropped
        short = gp.RestateIndex()
        short.add("s", rc.tokenize(SHORT_CORE))       # under 8 content tokens: never a reference
        self.assertIsNone(short.find(rc.tokenize(SHORT_CORE + " Again.")))

    def test_the_screen_keeps_the_entry_and_drops_the_paraphrase_in_a_run(self):
        def plant(m, msg, phase, recs, k):
            if phase == "core" and k == 1 and msg["persona"]["name"] == "Person P1":
                recs.insert(0, {"kind": "user", "text": CORE_FURNACE})
            if phase == "tail" and k == 1 and msg["persona"]["name"] == "Person P2":
                recs.append({"kind": "entry", "text": LONG_ENTRY})
                recs.insert(0, {"kind": "agent", "text": PARAPHRASE})
            return recs
        res, out, _ = self.run_pool(model=Planted(plant), workers=1, core=40, total=100)
        rows = [d for d in dropped_of(out) if d["reason"] == "restates_core"]
        self.assertEqual([r["text"] for r in rows], [PARAPHRASE])
        self.assertIn(LONG_ENTRY, [r["text"] for r in corpus_of(out)] + [d["text"] for d in dropped_of(out)
                                                                       if d["reason"] == "over_budget"])


class BudgetTests(PoolTest):
    def test_the_tail_is_over_asked_and_trimmed_evenly(self):
        res, out, _ = self.run_pool(core=40, total=100, workers=1)
        plan = res["manifest"]["plan"]
        self.assertEqual(plan["core_drop_rate_per_mille"], 0)
        self.assertEqual(plan["tail_overask_per_mille"], gp.TAIL_OVERASK_FLOOR)
        self.assertEqual(plan["tail_planned"], {"p1": 33, "p2": 33})  # 30 and 10% rounded up
        self.assertEqual(Counter(w["persona"] for w in jsonl(os.path.join(out, "generation", "records.jsonl"))
                                 if w["phase"] == "tail"), {"p1": 30, "p2": 30})
        trimmed = [d for d in dropped_of(out) if d["reason"] == "over_budget"]
        self.assertEqual(len(trimmed), 6)
        with open(os.path.join(out, "generation", "plan.json"), encoding="utf-8") as f:
            tail_sessions = [s["session"] for s in json.load(f)["personas"]["p1"]["sessions"] if s["phase"] == "tail"]
        p1 = [d["session"] for d in trimmed if d["persona"] == "p1"]
        self.assertGreater(len(set(p1)), 1 if len(tail_sessions) > 1 else 0)  # not just the newest session

    def test_the_over_ask_follows_the_core_drop_rate(self):
        def plant(m, msg, phase, recs, k):
            if phase == "core":  # one short record per core session, dropped by the screens
                recs.insert(0, {"kind": "agent", "text": "Okay, done."})
            return recs
        res, out, _ = self.run_planted(plant, core=40, total=100)
        plan = res["manifest"]["plan"]
        self.assertGreater(plan["core_drop_rate_per_mille"], 0)
        self.assertEqual(plan["tail_overask_per_mille"],
                         max(gp.TAIL_OVERASK_FLOOR, plan["core_drop_rate_per_mille"] + gp.TAIL_OVERASK_MARGIN))

    def run_planted(self, plant, **kw):
        res, out, _ = self.run_pool(model=Planted(plant), workers=1, **kw)
        return res, out, dropped_of(out)

    def test_no_make_up_is_smaller_than_the_smallest_session(self):
        class OneShort(FakeModel):
            def records(self, msg, phase):
                recs = super().records(msg, phase)
                return recs[:-1] if len(recs) > 4 else recs
        _, out, model = self.run_pool(model=OneShort(), workers=1, core=40, total=40)
        with open(os.path.join(out, "generation", "plan.json"), encoding="utf-8") as f:
            sessions = [s for p in json.load(f)["personas"].values() for s in p["sessions"]]
        makeups = [s for s in sessions if s["makeup"]]
        self.assertTrue(makeups)
        self.assertTrue(all(s["records"] >= 4 for s in makeups), makeups)  # records_per_session [4, 8]
        self.assertEqual(len(corpus_of(out)), 40)

    def test_max_calls_pauses_and_resume_ends_the_same(self):
        _, full, whole = self.run_pool("same", parent=os.path.join(self.tmp, "full"), workers=1)
        out = os.path.join(self.tmp, "cut", "same")
        rounds = 0
        while True:
            rounds += 1
            try:
                gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FakeModel(), workers=1, quiet=True,
                                 max_calls=6, resume=rounds > 1)
                break
            except gp.Paused as exc:
                self.assertIn("--resume", str(exc))
                self.assertEqual(len(calls_of(out)), 6 * rounds)
        self.assertEqual(rounds, -(-whole.calls // 6))
        self.assertEqual(files_of(full), files_of(out))

    def test_max_calls_exits_3_from_the_command_line(self):
        out = os.path.join(self.tmp, "cli")
        buf = io.StringIO()
        with mock.patch.object(gp, "PoolAnthropicModel", lambda model_id, api_key_env=None: FakeModel()), \
                contextlib.redirect_stdout(buf):
            code = gp.main(["--scenarios", self.scen, "--out", out, "--seed", "7", "--core", "40", "--total", "40",
                            "--max-calls", "2"])
        self.assertEqual(code, 3)
        self.assertIn("PAUSED: --max-calls 2 reached", buf.getvalue())
        self.assertEqual(len(calls_of(out)), 2)

    def test_the_call_log_splits_into_parts_and_replays_in_order(self):
        with mock.patch.object(gp, "CALL_PART_BYTES", 20000):
            res, a, model = self.run_pool("same", parent=os.path.join(self.tmp, "a"), workers=4)
            parts = gp.call_parts(os.path.join(a, "generation"))
            self.assertGreater(len(parts), 2)
            for p in parts:
                with open(p, "rb") as f:
                    lines = f.read().split(b"\n")
                self.assertTrue(os.path.getsize(p) <= 20000 or len([x for x in lines if x]) == 1, p)
            self.assertEqual(res["manifest"]["generation"]["call_log"],
                             [f"generation/{os.path.basename(p)}" for p in parts])
            self.assertEqual(len(calls_of(a)), model.calls)
            self.assertEqual([c["n"] for c in calls_of(a)], list(range(1, model.calls + 1)))
            b = os.path.join(self.tmp, "b", "same")
            gp.generate_pool(self.scen, b, 7, core=40, total=100, replay=parts[1], quiet=True)
        self.assertEqual(files_of(a), files_of(b))

    def test_a_paused_run_with_parts_resumes_across_them(self):
        _, full, _ = self.run_pool("same", parent=os.path.join(self.tmp, "full"), workers=1)
        with mock.patch.object(gp, "CALL_PART_BYTES", 15000):
            out = os.path.join(self.tmp, "cut", "same")
            with self.assertRaises(gp.Paused):
                gp.generate_pool(self.scen, out, 7, core=40, total=100, model=FakeModel(), workers=1, quiet=True,
                                 max_calls=9)
            parts = gp.call_parts(os.path.join(out, "generation"))
            self.assertGreater(len(parts), 1)
            with open(parts[-1], "a", encoding="utf-8") as f:
                f.write('{"n": 99, "request_sha256": "cut')
            live = FakeModel()
            gp.generate_pool(self.scen, out, 7, core=40, total=100, model=live, workers=1, quiet=True, resume=True)
        self.assertEqual(corpus_of(full), corpus_of(out))

    def test_the_estimate_comes_before_any_call(self):
        model = FakeModel()
        buf = io.StringIO()
        out = os.path.join(self.tmp, "est")
        with contextlib.redirect_stdout(buf):
            res = gp.generate_pool(self.scen, out, 7, core=40, total=100, model=model, estimate_only=True,
                                   effort={"core": "high", "tail": "medium"})
        self.assertEqual(model.calls, 0)
        self.assertFalse(os.path.exists(out))
        text = buf.getvalue()
        self.assertIn("estimate, core:", text)
        self.assertIn("at effort medium", text)
        self.assertRegex(text, r"about \$\d+\.\d\d on claude-sonnet-5-5")
        est = res["estimate"]
        self.assertEqual(est["phases"]["core"]["sessions"],
                         sum(len(gp.plan_phase(7, p, "core", 20, set(), {}, gp.scaled_quotas(p, 20))[0])
                             for p in gp.load_personas(self.scen)))
        self.assertGreater(est["cost_usd"], 0)
        _, done, first = self.run_pool("done", core=40, total=40)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = gp.generate_pool(self.scen, done, 7, core=40, total=40, model=FakeModel(), resume=True,
                                   estimate_only=True)
        self.assertEqual(res["estimate"]["phases"]["core"]["recorded"], first.calls)
        self.assertIn("already recorded, replayed free", buf.getvalue())

    def test_effort_is_read_recorded_and_checked_on_resume(self):
        self.assertEqual(gp.parse_effort("medium"), {"core": "medium", "tail": "medium"})
        self.assertEqual(gp.parse_effort("core=high,tail=low"), {"core": "high", "tail": "low"})
        self.assertEqual(gp.parse_effort("tail=medium"), {"core": "high", "tail": "medium"})
        for bad in ("fast", "core=fast", "middle=high"):
            with self.assertRaisesRegex(gq.GenerateError, "--effort"):
                gp.parse_effort(bad)
        _, out, _ = self.run_pool("eff", core=40, total=40, effort={"core": "high", "tail": "medium"})
        self.assertEqual({c["effort"] for c in calls_of(out)}, {"high"})
        with self.assertRaisesRegex(gq.GenerateError, "core effort"):
            gp.generate_pool(self.scen, out, 7, core=40, total=40, model=FakeModel(), resume=True, quiet=True,
                             effort={"core": "medium", "tail": "medium"})
        res = gp.generate_pool(self.scen, out, 7, core=40, total=80, model=FakeModel(), resume=True, quiet=True,
                               effort={"core": "high", "tail": "low"})
        self.assertEqual(res["manifest"]["model"]["effort"], {"core": "high", "tail": "low"})
        self.assertEqual({c["effort"] for c in calls_of(out) if c["phase"] == "tail"}, {"low"})


class OutTests(PoolTest):
    def test_out_inside_the_repo_is_refused(self):
        with self.assertRaisesRegex(gq.GenerateError, "inside the repository"):
            gp.check_out(os.path.join(REPO, "pool-here"), False, False)
        with self.assertRaisesRegex(gq.GenerateError, "inside the repository"):
            gp.check_out(os.path.join(REPO, "scenarios", "x"), True, False)

    def test_release_is_only_for_data(self):
        real, name = gp.check_out(os.path.join(REPO, "data", "pool-test-never-made"), True, False)
        self.assertEqual(name, "pool-test-never-made")
        self.assertFalse(os.path.exists(real))
        with self.assertRaisesRegex(gq.GenerateError, "outside it"):
            gp.check_out(os.path.join(self.tmp, "x"), True, False)

    def test_a_folder_that_is_not_empty_is_refused_without_resume(self):
        out = os.path.join(self.tmp, "full")
        os.makedirs(out)
        open(os.path.join(out, "x"), "w").close()
        with self.assertRaisesRegex(gq.GenerateError, "not an empty folder"):
            gp.check_out(out, False, False)

    def test_the_folder_name_becomes_the_pool_name(self):
        with self.assertRaisesRegex(gq.GenerateError, "letters, digits"):
            gp.check_out(os.path.join(self.tmp, "bad name"), False, False)
        res, _, _ = self.run_pool("pool-pilot-1", core=40, total=40)
        self.assertEqual(res["manifest"]["pool"], "pool-pilot-1")

    def test_replay_and_resume_together_are_refused(self):
        with self.assertRaisesRegex(gq.GenerateError, "do not go together"):
            gp.generate_pool(self.scen, os.path.join(self.tmp, "x"), 1, replay="x", resume=True)


class StatisticsTests(PoolTest):
    def test_percentiles_are_nearest_rank(self):
        p = gp.percentiles(list(range(1, 101)))
        self.assertEqual((p["min"], p["p5"], p["p25"], p["p50"], p["p75"], p["p95"], p["max"]),
                         (1, 5, 25, 50, 75, 95, 100))
        self.assertEqual(p["mean"], 50.5)
        self.assertEqual(gp.percentiles([7]), {"n": 1, "min": 7, "p5": 7, "p25": 7, "p50": 7, "p75": 7, "p95": 7,
                                               "max": 7, "mean": 7.0})
        self.assertEqual(gp.percentiles([]), {"n": 0})

    def test_the_manifest_statistics_match_the_corpus(self):
        res, out, _ = self.run_pool(core=40, total=100)
        st = res["manifest"]["statistics"]
        recs = corpus_of(out)
        self.assertEqual(st["records"], 100)
        self.assertEqual(st["by_kind"]["all"], {k: sum(1 for r in recs if r["kind"] == k) for k in gp.KINDS})
        self.assertEqual(st["by_kind"]["core"], {k: sum(1 for r in recs[:40] if r["kind"] == k) for k in gp.KINDS})
        self.assertEqual(st["entry_share"]["all"], round(st["by_kind"]["all"]["entry"] / 100, 4))
        self.assertEqual(st["words"]["all"]["max"], max(len(r["text"].split()) for r in recs))
        self.assertEqual(st["characters"]["min"], min(len(r["text"]) for r in recs))
        self.assertEqual(st["sessions"]["count"], len({r["source"] for r in recs}))
        days = Counter((r["source"][:2], r["ts"][:10]) for r in recs)
        self.assertEqual(st["active_days"]["count"], len(days))
        entries = Counter((r["source"][:2], r["ts"][:10]) for r in recs if r["kind"] == "entry")
        self.assertEqual(st["active_days"]["entries_per_active_day"]["mean"],
                         round(sum(entries.values()) / len(days), 4))
        per_session = Counter(r["source"] for r in recs)
        self.assertEqual(st["sessions"]["records_per_session"]["max"], max(per_session.values()))
        self.assertEqual(sum(st["records_by_month"].values()), 100)
        self.assertEqual(st["by_persona"]["p1"]["records"], 50)
        self.assertEqual(st["near_copy_clusters"], 0)
        ordered = sorted(enumerate(recs), key=lambda x: (x[1]["ts"], x[0]))
        content = lambda t: gp.content_tokens(rc.tokenize(t))  # noqa: E731
        brute = sum(1 for j, (_, r) in enumerate(ordered)
                    if any(len(content(e["text"])) >= gp.RESTATE_MIN_CONTENT
                           and rc.token_overlap(content(e["text"]), content(r["text"])) >= 0.6
                           and rc.token_overlap(content(r["text"]), content(e["text"])) >= 0.6
                           for _, e in ordered[:j]))
        self.assertEqual(st["restatement_rate"]["all"], round(brute / 100, 4))
        self.assertEqual({d["reason"] for d in dropped_of(out)} - {"over_budget"}, set())  # only the tail's trim
        self.assertEqual(st["dropped"]["by_phase"]["core"], 0)
        self.assertEqual(st["dropped"]["total"], st["dropped"]["by_phase"]["tail"])

    def test_restatement_and_near_copy_counts_on_a_small_pool(self):
        def rec(text, ts, i):
            return {"id": rc.record_id(text), "text": text, "kind": "user", "ts": ts, "persona": "p1",
                    "session": "p1-s000000", "seq": 0, "position": i}
        a = rec("Moved the red enamel kettle onto the third pantry shelf beside Grandma's cast iron skillet.",
                "2026-01-01T00:00:00Z", 0)
        b = rec("Reminder: the red enamel kettle now lives on the third pantry shelf, beside Grandma's cast iron "
                "skillet.", "2026-01-02T00:00:00Z", 1)
        c = rec("Bought 2 lemons at the corner shop for the tart.", "2026-01-03T00:00:00Z", 2)
        d = rec("Bought 2 lemons at the corner shop for the tart!", "2026-01-04T00:00:00Z", 3)
        st = gp.pool_statistics([a, b, c, d], [a, b, c, d], [])
        self.assertEqual(st["near_copy_clusters"], 1)          # c and d
        self.assertEqual(st["restatement_rate"]["all"], 0.25)  # b restates a; c has under 8 content tokens

    def test_the_cover_index_finds_what_token_overlap_finds(self):
        idx = gp.CoverIndex(0.6)
        texts = ["the cat sat on the mat today", "a dog ran in the park at noon", "seven blue boats in the harbor"]
        for i, t in enumerate(texts):
            idx.add(str(i), rc.tokenize(t))
        probes = ["the cat sat on a mat", "dog ran park noon quickly", "boats blue seven harbor", "nothing alike here"]
        for p in probes:
            brute = next((str(i) for i, t in enumerate(texts)
                          if rc.token_overlap(rc.tokenize(t), rc.tokenize(p)) >= 0.6), None)
            self.assertEqual(idx.find(rc.tokenize(p)), brute, p)


def fake_sdk(seen, created=None, fail=None, clients=None):
    """A stand-in for the anthropic package: messages.stream records each request and
    answers it with FakeModel's reply; messages.create answers --check-api. clients
    records each client's keyword arguments. To make it raise, set fake.client_fail
    (making the client), fake.stream_fail to (n, error) (every stream after the first
    n) or fake.create_fail (--check-api); fail is an error class create raises."""

    class Stream:
        def __init__(self, kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            user = "".join(b["text"] for b in self.kw["messages"][0]["content"])
            phase = "tail" if '"core_facts_never_restate"' in user else "core"
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=FakeModel().text(json.loads(user), phase))],
                model="claude-sonnet-5-5", stop_reason="end_turn",
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=20, cache_creation_input_tokens=5,
                                            cache_read_input_tokens=7))

    class Messages:
        def stream(self, **kw):
            if fake.stream_fail and len(seen) >= fake.stream_fail[0]:
                raise fake.stream_fail[1]
            seen.append(kw)
            return Stream(kw)

        def create(self, **kw):
            if fake.create_fail:
                raise fake.create_fail
            if fail:
                raise fail("Invalid x-api-key sk-" + "ant-secret-should-never-print")
            if created is not None:
                created.append(kw)
            return types.SimpleNamespace(model="claude-sonnet-5-5", content=[], stop_reason="end_turn")

    class Client:
        def __init__(self, **kw):
            if clients is not None:
                clients.append(kw)
            if fake.client_fail:
                raise fake.client_fail
            self.messages = Messages()

    fake = types.ModuleType("anthropic")
    fake.__version__ = "0.99.0"
    fake.Anthropic = Client
    fake.AnthropicError = type("AnthropicError", (Exception,), {})
    fake.APIError = type("APIError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (fake.APIError,), {})
    fake.client_fail = fake.stream_fail = fake.create_fail = None
    return fake


class ApiTests(PoolTest):
    def test_the_api_request_streams_with_caching_and_the_effort_asked(self):
        seen = []
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(seen)}):
            res = gp.generate_pool(self.scen, os.path.join(self.tmp, "api"), 7, core=40, total=60, personas=["p1"],
                                   quiet=True, effort={"core": "high", "tail": "medium"})
        prompts = gp.load_prompts()
        core = [kw for kw in seen if '"core_facts_never_restate"' not in kw["messages"][0]["content"][0]["text"]]
        tail = [kw for kw in seen if kw not in core]
        self.assertTrue(core and tail)
        for kw in seen:
            self.assertEqual(set(kw), {"model", "max_tokens", "system", "messages", "output_config"})
            self.assertEqual(kw["max_tokens"], gp.POOL_MAX_TOKENS)
            self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
            first, rest = kw["messages"][0]["content"]
            self.assertEqual(first["cache_control"], {"type": "ephemeral"})
            self.assertNotIn("cache_control", rest)
            self.assertNotIn('"session"', first["text"])  # the session changes each call, so never in the prefix
            json.loads(first["text"] + rest["text"])
        self.assertEqual(core[0]["system"][0]["text"], prompts["core"].text)
        self.assertEqual(core[0]["output_config"], {"effort": "high", "format": {"type": "json_schema",
                                                                                 "schema": prompts["core"].schema}})
        self.assertEqual(tail[0]["output_config"]["effort"], "medium")
        self.assertIn('"core_matters_never_touch"', tail[0]["messages"][0]["content"][0]["text"])
        self.assertNotIn('"people"', tail[0]["messages"][0]["content"][0]["text"].split('"state"')[1])  # can grow
        # every tail session of a persona shares one cached prefix
        self.assertEqual(len({kw["messages"][0]["content"][0]["text"] for kw in tail}), 1)
        m = res["manifest"]["model"]
        self.assertEqual(m["sdk"], "anthropic 0.99.0")
        self.assertEqual((m["cache_read_input_tokens"], m["cache_creation_input_tokens"]), (7 * len(seen), 5 * len(seen)))
        self.assertEqual(m["effort_by_call"], {"high": len(core), "medium": len(tail)})
        self.assertEqual(m["cost_usd"], round((10 * 2.0 + 20 * 10.0 + 5 * 2.5 + 7 * 0.2) * len(seen) / 1e6, 2))
        calls = calls_of(os.path.join(self.tmp, "api"))
        self.assertEqual({c["effort"] for c in calls if c["phase"] == "tail"}, {"medium"})

    def test_check_api_prints_ok_or_the_error_class_only(self):
        created = []
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], created)}), contextlib.redirect_stdout(out):
            self.assertEqual(gp.main(["--check-api"]), 0)
        self.assertEqual(out.getvalue().strip(), "ok: claude-sonnet-5-5 answered")
        self.assertEqual(created[0]["max_tokens"], 16)
        fake = fake_sdk([])
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], fail=fake.AuthenticationError)}), \
                contextlib.redirect_stderr(err):
            self.assertEqual(gp.main(["--check-api"]), 2)
        self.assertEqual(err.getvalue().strip(), "error: AuthenticationError")
        self.assertNotIn("sk-ant", err.getvalue())


# A key-shaped value for the tests below, assembled from parts so no whole key sits in this file.
KEY_VAR = "POOL_TEST_KEY"
FAKE_KEY = "-".join(("sk", "ant", "api03", "pooltest", "Qx7Lw2Rv9Bn4Tc6M" + "d1Kf8Hj3Gs5Pz0Ye")) + "AA"


class ApiKeyEnvTests(PoolTest):
    """--api-key-env: the key comes from the named variable alone, every call goes to
    api.anthropic.com, and the key is in no output, log, file or error text, even when
    the SDK raises with the key in its message."""

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {KEY_VAR: FAKE_KEY, "ANTHROPIC_BASE_URL": "https://proxy.example.invalid",
                                           "ANTHROPIC_API_KEY": "-".join(("sk", "ant", "not", "this", "one"))})
        env.start()
        self.addCleanup(env.stop)

    def run_main(self, *argv):
        """gp.main with both streams captured and not one log record allowed: (exit code, text)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertNoLogs(level="DEBUG"):
            code_ = gp.main(list(argv))
        return code_, out.getvalue() + err.getvalue()

    def args(self, out, *extra):
        return ["--scenarios", self.scen, "--out", out, "--seed", "7", "--core", "40", "--total", "60",
                "--personas", "p1", *extra]

    def assert_no_key(self, text, out):
        self.assertNotIn(FAKE_KEY, text)
        for name, data in files_of(out).items():  # the call log, the manifest, every file the run left
            self.assertNotIn(FAKE_KEY.encode(), data, name)
            self.assertNotIn(KEY_VAR.encode(), data, name)  # the flag is never recorded either

    def test_the_key_comes_from_the_named_variable_and_every_call_goes_to_the_api(self):
        seen, clients = [], []
        keyed, plain = os.path.join(self.tmp, "a", "pool"), os.path.join(self.tmp, "b", "pool")
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(seen, clients=clients)}), \
                mock.patch.object(gq, "_utc_now", lambda: TIME):
            code_, text = self.run_main(*self.args(keyed, "--api-key-env", KEY_VAR))
            self.assertEqual(code_, 0, text)
            self.assertEqual(self.run_main(*self.args(plain))[0], 0)
        self.assertEqual(clients, [{"api_key": FAKE_KEY, "base_url": "https://api.anthropic.com"}, {}])
        self.assertTrue(seen)
        self.assertTrue(os.path.isfile(os.path.join(keyed, "manifest.json")))
        self.assert_no_key(text, keyed)
        # the flag changes how the client is made and nothing else: the same files as a run without it
        self.assertEqual(files_of(keyed), files_of(plain))

    def test_the_key_is_in_no_output_file_or_error_when_the_sdk_raises(self):
        cases = {
            "making the client": ("client_fail", lambda f: TypeError(f"cannot use the key {FAKE_KEY}")),
            "an API error": ("stream_fail", lambda f: (2, f.AuthenticationError(f"invalid x-api-key {FAKE_KEY}"))),
            "a TypeError": ("stream_fail", lambda f: (2, TypeError(f"bad header X-Api-Key: {FAKE_KEY}"))),
            "anything else": ("stream_fail", lambda f: (2, RuntimeError(f"the request with {FAKE_KEY} broke"))),
        }
        for n, (what, (attr, error)) in enumerate(cases.items()):
            with self.subTest(what):
                fake = fake_sdk([])
                setattr(fake, attr, error(fake))
                out = os.path.join(self.tmp, f"raised{n}")
                with mock.patch.dict(sys.modules, {"anthropic": fake}):
                    code_, text = self.run_main(*self.args(out, "--api-key-env", KEY_VAR))
                self.assertEqual(code_, 2, text)
                self.assertIn("[the key]", text)  # the error keeps its words, with the key taken out
                self.assert_no_key(text, out)
                recorded = calls_of(out) if os.path.isdir(out) else []
                self.assertEqual(len(recorded), 2 if attr == "stream_fail" else 0)  # the files checked are real
                # even a traceback of the error, printed whole, would not hold the key
                fake = fake_sdk([])
                setattr(fake, attr, error(fake))
                with mock.patch.dict(sys.modules, {"anthropic": fake}), self.assertRaises(gp.GenerateError) as cm:
                    gp.generate_pool(self.scen, os.path.join(self.tmp, f"traced{n}"), 7, core=40, total=60,
                                     personas=["p1"], quiet=True, api_key_env=KEY_VAR)
                self.assertNotIn(FAKE_KEY, "".join(traceback.format_exception(cm.exception)))

    def test_check_api_reads_the_named_variable_and_prints_no_key(self):
        created, clients = [], []
        os.environ[KEY_VAR] = f"  {FAKE_KEY}\n"  # spaces and a line break around the key are let go
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], created, clients=clients)}):
            code_, text = self.run_main("--check-api", "--api-key-env", KEY_VAR)
        self.assertEqual((code_, text.strip()), (0, "ok: claude-sonnet-5-5 answered"))
        self.assertEqual(clients, [{"api_key": FAKE_KEY, "base_url": gp.API_URL}])
        fake = fake_sdk([])
        fake.create_fail = fake.AuthenticationError(f"invalid x-api-key {FAKE_KEY}")
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            code_, text = self.run_main("--check-api", "--api-key-env", KEY_VAR)
        self.assertEqual((code_, text.strip()), (2, "error: AuthenticationError"))
        fake = fake_sdk([])
        fake.client_fail = TypeError(f"cannot use the key {FAKE_KEY}")
        with mock.patch.dict(sys.modules, {"anthropic": fake}):
            code_, text = self.run_main("--check-api", "--api-key-env", KEY_VAR)
        self.assertEqual((code_, text.strip()),
                         (2, "error: could not create the API client (TypeError: cannot use the key [the key])"))

    def test_a_bad_name_or_value_is_refused_before_anything_is_written(self):
        half = len(FAKE_KEY) // 2
        cases = {
            "the key itself as the name": (FAKE_KEY, FAKE_KEY, "never the key itself"),
            "an unset variable": (KEY_VAR, None, f"{KEY_VAR} is not set"),
            "an empty variable": (KEY_VAR, " \n", f"{KEY_VAR} is not set"),
            "a space inside": (KEY_VAR, FAKE_KEY[:half] + " " + FAKE_KEY[half:], f"{KEY_VAR} holds a space"),
            "a line break inside": (KEY_VAR, FAKE_KEY[:half] + "\n" + FAKE_KEY[half:], f"{KEY_VAR} holds a space"),
            "a quote inside": (KEY_VAR, FAKE_KEY[:half] + '"' + FAKE_KEY[half:], f"{KEY_VAR} holds a space"),
        }
        for n, (what, (name, value, said)) in enumerate(cases.items()):
            with self.subTest(what):
                if value is None:
                    os.environ.pop(KEY_VAR, None)
                else:
                    os.environ[KEY_VAR] = value
                clients = []
                out = os.path.join(self.tmp, f"refused{n}")
                for argv in (self.args(out, "--api-key-env", name), ["--check-api", "--api-key-env", name]):
                    with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], clients=clients)}):
                        code_, text = self.run_main(*argv)
                    self.assertEqual(code_, 2, text)
                    self.assertIn(said, text)
                    self.assertNotIn(FAKE_KEY[:half], text)
                    self.assertNotIn(FAKE_KEY[half:], text)
                self.assertEqual(clients, [])
                self.assertFalse(os.path.exists(out))

    def test_the_estimate_never_reads_the_key(self):
        os.environ.pop(KEY_VAR)
        clients = []
        out = os.path.join(self.tmp, "estimated")
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk([], clients=clients)}):
            code_, text = self.run_main(*self.args(out, "--api-key-env", KEY_VAR, "--estimate"))
        self.assertEqual(code_, 0, text)
        self.assertIn("estimate, core:", text)
        self.assertEqual(clients, [])
        self.assertFalse(os.path.exists(out))


class ShowPoolTests(PoolTest):
    def capture(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code_ = sp.main(list(argv))
        return code_, buf.getvalue()

    def test_records_stats_and_dropped_print(self):
        def plant(m, msg, phase, recs, k):
            if k == 1 and msg["persona"]["name"] == "Person P1":
                recs.insert(0, {"kind": "agent", "text": "Okay, done."})
            return recs
        _, out, _ = self.run_pool(model=Planted(plant), workers=1)
        rc_, text = self.capture("records", out)
        self.assertEqual(rc_, 0)
        self.assertIn("######## p1 Person P1", text)
        self.assertIn("core session 1 of", text)
        self.assertIn("plan asked for:", text)
        self.assertEqual(text.count("\n  ["), 100)
        _, only = self.capture("records", out, "--persona", "p2", "--phase", "core")
        self.assertNotIn("p1 Person", only)
        self.assertEqual(only.count("\n  ["), 20)
        rc_, stats = self.capture("stats", out)
        self.assertEqual(rc_, 0)
        self.assertIn("Records: 100 (40 core, 60 tail)", stats)
        self.assertIn("entries per active day", stats)
        self.assertIn("too_short 2", stats)  # the planted record, in the core and in the tail
        rc_, dropped = self.capture("dropped", out)
        self.assertIn("-- too_short", dropped)
        self.assertIn("Okay, done.", dropped)

    def test_planted_records_are_tagged_and_can_be_shown_alone(self):
        _, out, _ = self.run_pool(core=40, total=60)
        _, text = self.capture("records", out)
        self.assertRegex(text, r"\n  \[(user|agent)\] \{V1 1\} ")
        _, only = self.capture("records", out, "--only", "C")
        lines = [x for x in only.split("\n") if x.startswith("  [")]
        self.assertTrue(lines)
        self.assertTrue(all(re.match(r"  \[\w+\] \{C\d+ [12]\} ", x) for x in lines), lines)
        _, stats = self.capture("stats", out)
        self.assertRegex(stats, r"by phase: core 0 dropped, drop rate 0\.0; tail \d+ dropped, drop rate ")

    def test_a_missing_pool_exits_2(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(sp.main(["stats", os.path.join(self.tmp, "nothing")]), 2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Parsica Recall lint validator (spec recall/1.0, section 9).

Checks a question set against its pool, not just item by item: an answer key is
judged against every record a retriever can return. Rules and gates are declared
in parsica_recall_lint_spec.json; this file implements them. Every gate returns a
pass or a fail.

    parsica-recall lint --questions data/v11/questions.json \\
        --corpus data/v11/corpus.jsonl --output data/v11/lint_report.json

Exit status: 0 when the report was written, 2 on unreadable input. Use
--fail-on-errors to exit 1 when any item carries an error finding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import recall_common as rc

REPO_DIR = rc.ROOT_DIR  # the repository's root in a checkout, the package folder in an installed copy
DEFAULT_SPEC = os.path.join(REPO_DIR, "parsica_recall_lint_spec.json")
DEFAULT_QSET_DIR = os.path.join(REPO_DIR, "data", rc.CURRENT_QSET)

STATUS_PRECEDENCE = {s: i for i, s in enumerate(("publish", "hold_experimental", "reclassify", "rewrite", "drop"))}
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_SHA_VALUE = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Secrets and personal data (written fresh for Recall)
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [
    ("provider_key", re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}|sk_(?:live|test)_[A-Za-z0-9]{16,}"
        r"|rk_(?:live|test)_[A-Za-z0-9]{16,}|pk_live_[A-Za-z0-9]{16,})")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("connection_string_credentials", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@[^\s/]+")),
]

PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone_nanp", re.compile(r"(?<![\d\-])(?:\+?1[\s.\-]?)?\(?[2-9]\d{2}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\d\-])")),
    ("phone_international", re.compile(r"(?<![\w+])\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){2,5}\d{2,4}(?!\d)")),
    ("street_address", re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\.?\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|"
        r"Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Circle|Parkway|Pkwy|Highway|Hwy)\b")),
    ("us_ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("uk_ni_number", re.compile(r"\b(?![DFIQUV])[A-CEGHJ-PR-TW-Z](?![DFIOQUV])[A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")),
]

_CARD = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+=_\-]{32,}")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def scan_secrets_pii(text: Any) -> List[Dict[str, str]]:
    """Credential and personal-data shapes found in one string: the kind and the shape of
    each, and never the text that matched. A lint report is the kind of file people paste
    into an issue, so it says where a shape sits and what it is, and does not repeat it."""
    if not text:
        return []
    s = str(text)
    hits = []
    for name, pat in SECRET_PATTERNS:
        for m in pat.finditer(s):
            hits.append({"kind": "secret", "shape": name})
    for m in _LONG_TOKEN.finditer(s):
        tok = m.group(0)
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        is_hex = all(c in "0123456789abcdefABCDEF" for c in tok)
        if has_digit and has_alpha and _entropy(tok) >= (3.5 if is_hex else 4.2):
            hits.append({"kind": "secret", "shape": "high_entropy_token"})
    for name, pat in PII_PATTERNS:
        for m in pat.finditer(s):
            hits.append({"kind": "pii", "shape": name})
    for m in _CARD.finditer(s):
        digits = re.sub(r"[ \-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits.append({"kind": "pii", "shape": "card_number"})
    return hits


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    rule_id: str
    severity: str
    code: str
    message: str
    field: Optional[str] = None
    value: Any = None
    suggested_review_status: Optional[str] = None


class Context:
    """The pool and everything rules need precomputed."""

    def __init__(self, records: List[Dict[str, Any]], spec: Dict[str, Any],
                 distractors: Optional[List[Dict[str, Any]]] = None, store_ids: bool = False):
        self.spec = spec
        self.store_ids = store_ids  # a private set keeping its store's own ids (spec section 3)
        self.const = spec.get("constants", {})
        self.records = list(records) + list(distractors or [])
        self.n_corpus = len(records)
        self.by_id = {r["id"]: r for r in self.records}
        self.ids = [r["id"] for r in self.records]
        self.norm = [rc.normalize(r["text"]) for r in self.records]
        self.norm_by_id = dict(zip(self.ids, self.norm))
        self.index = rc.PhraseIndex(self.norm)
        self.tokens_by_id = {i: n.split() for i, n in self.norm_by_id.items()}
        self.N = len(self.records)

    def candidates_sharing_tokens(self, record_ids: Sequence[str]) -> List[str]:
        """Records sharing at least one token with any of the given records."""
        found: set = set()
        for rid in record_ids:
            for tok in set(self.tokens_by_id.get(rid, [])):
                found.update(self.index.postings.get(tok, ()))
        return [self.ids[i] for i in sorted(found)]


def _finding(rule: Dict[str, Any], code: str, message: str, field: Optional[str] = None,
             value: Any = None) -> Finding:
    action = rule.get("action") or {}
    return Finding(rule_id=rule["id"], severity=rule.get("severity", "warn"), code=code, message=message,
                   field=field, value=value, suggested_review_status=action.get("set_review_status"))


def _is_abstention(item: Dict[str, Any]) -> bool:
    return item.get("category") == "abstention"


def _keys(item: Dict[str, Any], field: str) -> List[str]:
    if field == "acceptable_answers":
        return [str(v) for v in (item.get(field) or [])]
    value = item.get(field)
    return [] if value is None else [str(value)]


def _gold_norms(item: Dict[str, Any], ctx: Context) -> List[str]:
    return [ctx.norm_by_id[g] for g in (item.get("gold_ids") or []) if g in ctx.norm_by_id]


# ---------------------------------------------------------------------------
# Item rules
# ---------------------------------------------------------------------------

def r_required_fields(item, rule, ctx):
    return [_finding(rule, "missing_required_field", f"Missing required field '{f}'.", field=f)
            for f in ctx.spec.get("required_fields", []) if f not in item]


def r_enum_membership(item, rule, ctx):
    enums = ctx.spec.get("enums", {})
    out = []
    for f in rule["match"]["fields"]:
        if f in item and item[f] not in enums.get(f, []):
            out.append(_finding(rule, "invalid_enum_value", f"Invalid value for '{f}': {item[f]!r}", f, item[f]))
    return out


def r_gold_shape(item, rule, ctx):
    out = []
    gold = item.get("gold_ids")
    if ctx.store_ids:  # store ids are the store's own: any non-empty string
        well_formed = lambda g: isinstance(g, str) and bool(g.strip())  # noqa: E731
        what = "non-empty store ids"
    else:
        well_formed = lambda g: isinstance(g, str) and bool(_HEX16.match(g))  # noqa: E731
        what = "16-hex record ids"
    if not isinstance(gold, list) or not all(well_formed(g) for g in gold):
        return [_finding(rule, "gold_ids_malformed", f"gold_ids must be a list of {what}.", "gold_ids", gold)]
    if len(set(gold)) != len(gold):
        out.append(_finding(rule, "gold_ids_repeat", "gold_ids lists a record twice.", "gold_ids", gold))
    first = item.get("must_rank_first")
    if first is not None:
        current = rc.current_ids(first)
        if not current or not all(isinstance(c, str) for c in current):
            out.append(_finding(rule, "must_rank_first_malformed",
                                "must_rank_first must be a record id or a non-empty list of record ids.",
                                "must_rank_first", first))
        elif len(set(current)) != len(current):
            out.append(_finding(rule, "must_rank_first_repeat", "must_rank_first lists a record twice.",
                                "must_rank_first", first))
        elif any(c not in gold for c in current):
            out.append(_finding(rule, "must_rank_first_not_gold", "must_rank_first is not one of the gold ids.",
                                "must_rank_first", first))
    if first is not None and item.get("category") not in rc.VERSION_CATEGORIES:
        out.append(_finding(rule, "must_rank_first_outside_version_category",
                            "must_rank_first is only set for knowledge_update, correction_handling and "
                            "recency_preference.", "must_rank_first", first))
    hashes = item.get("gold_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(gold):
        out.append(_finding(rule, "gold_hashes_mismatch", "gold_hashes must have exactly one entry per gold id.",
                            "gold_hashes", hashes))
    elif not all(isinstance(v, str) and _SHA_VALUE.match(v) for v in hashes.values()):
        out.append(_finding(rule, "gold_hashes_malformed", "gold_hashes values must be 'sha256:<64 hex>'.",
                            "gold_hashes", hashes))
    return out


def r_gold_present(item, rule, ctx):
    out = []
    hashes = item.get("gold_hashes") or {}
    for g in item.get("gold_ids") or []:
        rec = ctx.by_id.get(g)
        if rec is None:
            out.append(_finding(rule, "gold_not_in_pool", f"Gold record {g} is not in the pool.", "gold_ids", g))
        elif isinstance(hashes, dict) and hashes.get(g) not in (None, rc.text_hash(rec["text"])):
            out.append(_finding(rule, "gold_hash_mismatch", f"Gold record {g} text does not match its gold_hashes entry.",
                                "gold_hashes", g))
    return out


def r_abstention_shape(item, rule, ctx):
    gold = item.get("gold_ids") or []
    if _is_abstention(item):
        out = []
        if gold:
            out.append(_finding(rule, "abstention_has_gold", "Abstention items carry no gold ids.", "gold_ids", gold))
        if item.get("answer_type") != "abstention":
            out.append(_finding(rule, "abstention_answer_type", "Abstention items use answer_type 'abstention'.",
                                "answer_type", item.get("answer_type")))
        return out
    if not gold:
        return [_finding(rule, "answerable_without_gold", "An answerable item needs at least one gold id.", "gold_ids", gold)]
    if item.get("answer_type") == "abstention":
        return [_finding(rule, "abstention_answer_type_outside_category",
                         "answer_type 'abstention' is only for the abstention category.", "answer_type", "abstention")]
    return []


def r_multi_record_shape(item, rule, ctx):
    cat = item.get("category")
    gold = item.get("gold_ids") or []
    out = []
    if cat in rc.MULTI_RECORD_CATEGORIES and len(gold) < 2:
        out.append(_finding(rule, "multi_record_category_single_gold",
                            f"{cat} needs at least two gold records; this item has {len(gold)}.", "gold_ids", gold))
    if cat in rc.VERSION_CATEGORIES and not item.get("must_rank_first"):
        out.append(_finding(rule, "version_category_without_current",
                            f"{cat} must name the current version in must_rank_first.", "must_rank_first", None))
    return out


def r_source_length(item, rule, ctx):
    need = int(ctx.const.get("min_gold_tokens", 6))
    out = []
    for g in item.get("gold_ids") or []:
        n = len(ctx.norm_by_id.get(g, "").split())
        if g in ctx.norm_by_id and n < need:
            out.append(_finding(rule, "gold_too_short", f"Gold record {g} has {n} tokens (minimum {need}).", "gold_ids", g))
    return out


def _short_allowed(item, norm_value: str, ctx) -> bool:
    if item.get("answer_type") in set(ctx.const.get("short_answer_type_whitelist", [])):
        return True
    for name in ("short_answer_numeric_regex", "short_answer_time_regex"):  # all digits; a time of day, "9pm"
        pattern = ctx.const.get(name)
        if pattern and re.fullmatch(pattern, norm_value):
            return True
    return False


def r_min_length(item, rule, ctx):
    if _is_abstention(item):
        return []
    field = rule["match"]["field"]
    need = int(ctx.const.get("min_answer_chars", 4))
    out = []
    for value in _keys(item, field):
        norm = rc.normalize(value)
        if len(norm) < need and not _short_allowed(item, norm, ctx):
            out.append(_finding(rule, "answer_too_short", f"'{field}' value too short after normalization: {value!r}",
                                field, value))
    return out


def r_prohibited_values(item, rule, ctx):
    if _is_abstention(item):
        return []
    field = rule["match"]["field"]
    banned = {rc.normalize(v) for v in ctx.const.get(rule["match"]["values"], [])}
    return [_finding(rule, "prohibited_generic_key", f"'{field}' value is a prohibited generic key: {v!r}", field, v)
            for v in _keys(item, field) if rc.normalize(v) in banned]


def r_role_label(item, rule, ctx):
    if item.get("answer_type") != "role":
        return []
    allowed = {rc.normalize(v) for v in ctx.const.get("role_answer_whitelist", [])}
    out = []
    if rc.normalize(item.get("answer")) not in allowed:
        out.append(_finding(rule, "role_answer_not_stable_label", f"Role answer is not a stable label: {item.get('answer')!r}",
                            "answer", item.get("answer")))
    # A role item is exempt from GOLD_MULTIPLICITY, so every alternate has to be a label too.
    for value in _keys(item, "acceptable_answers"):
        if rc.normalize(value) not in allowed:
            out.append(_finding(rule, "role_answer_not_stable_label", f"Role alternate is not a stable label: {value!r}",
                                "acceptable_answers", value))
    return out


def r_entity_not_pronoun(item, rule, ctx):
    if item.get("answer_type") != "entity":
        return []
    banned = {rc.normalize(v) for v in ctx.const.get("pronoun_answers", [])}
    if rc.normalize(item.get("answer")) in banned:
        return [_finding(rule, "entity_answer_is_pronoun", f"Entity answer is a pronoun or artifact: {item.get('answer')!r}",
                         "answer", item.get("answer"))]
    return []


def r_quantity_unit(item, rule, ctx):
    if item.get("answer_type") != "quantity_with_unit":
        return []
    field = rule["match"]["field"]
    units = [u.lower() for u in ctx.const.get("unit_terms", [])]
    out = []
    for value in _keys(item, field):
        raw = value.lower()
        if re.search(r"\d", raw) and not any(u in raw for u in units):
            out.append(_finding(rule, "quantity_missing_unit", f"Numeric quantity lacks a unit in '{field}': {value!r}",
                                field, value))
    return out


def r_key_edges(item, rule, ctx):
    if _is_abstention(item):
        return []
    out = []
    for field in ("answer", "acceptable_answers"):
        for value in _keys(item, field):
            word = rc.edge_function_word(rc.normalize(value))
            if word:
                out.append(_finding(rule, "key_edge_function_word",
                                    f"'{field}' value starts or ends on the function word '{word}': {value!r}", field, value))
    return out


def r_answer_not_in_question(item, rule, ctx):
    if _is_abstention(item):
        return []
    nq = rc.normalize(item.get("question"))
    choice = item.get(rc.CHOICE) is True  # it offers its answer as one of two possibilities, on purpose
    out = []
    for field in ("answer", "acceptable_answers"):
        for value in _keys(item, field):
            if rc.contains_phrase(nq, rc.normalize(value)):
                if choice:
                    out.append(Finding(rule_id=rule["id"], severity="info", code="choice",
                                       message=f"The question offers its answer ('{field}') as one of two "
                                               f"possibilities: {value!r}", field=field, value=value))
                else:
                    out.append(_finding(rule, "answer_in_question",
                                        f"The question contains its own answer ('{field}'): {value!r}", field, value))
    return out


def r_gold_multiplicity(item, rule, ctx):
    if _is_abstention(item):
        return []
    if item.get("answer_type") == "role":
        # A role label says who acted; it is not a span of the record, so a verbatim count over
        # the pool says nothing about the item. ROLE_ANSWER_MUST_BE_STABLE_LABEL checks it instead.
        return []
    gold = set(item.get("gold_ids") or [])
    out = []
    reached = set()
    current = [c for c in rc.current_ids(item.get("must_rank_first")) if isinstance(c, str)]
    # An item that carries common_answer says its answer is also found outside gold and that
    # it is scored by record id alone: the runner leaves it out of the exact tier, so the leak
    # is noted here and holds nothing (docs/generation.md section 6).
    common = item.get(rc.COMMON_ANSWER) is True
    for field in ("answer", "acceptable_answers"):
        for value in _keys(item, field):
            norm = rc.normalize(value)
            hits = {ctx.ids[i] for i in ctx.index.matching(norm)}
            leaks = sorted(hits - gold)
            if leaks and common:
                out.append(Finding(rule_id=rule["id"], severity="info", code="common_answer",
                                   message=f"'{field}' value {value!r} also appears in {len(leaks)} non-gold "
                                           f"record(s); the item is scored by record id alone",
                                   field=field, value=value))
            elif leaks:
                out.append(_finding(rule, "answer_in_other_record",
                                    f"'{field}' value {value!r} also appears in {len(leaks)} non-gold record(s): "
                                    f"{', '.join(leaks[:5])}", field, value))
            reached |= hits
    if item.get("category") not in rc.NON_VERBATIM_CATEGORIES:
        # A list in must_rank_first names copies of the current version, so each one
        # has to carry the answer; otherwise any gold record will do.
        if current:
            unreached = [c for c in current if c not in reached]
            where = ("the must_rank_first record" if len(current) == 1
                     else f"must_rank_first record(s) {', '.join(unreached)}")
        else:
            unreached = [] if reached & gold else ["gold"]
            where = "any gold record"
        if unreached:
            out.append(_finding(rule, "answer_not_in_gold",
                                f"Neither the answer nor an alternate appears in {where}, so the exact tier cannot reach it.",
                                "answer", item.get("answer")))
    return out


def r_alternate_specificity(item, rule, ctx):
    if _is_abstention(item):
        return []
    limit = float(ctx.const.get("max_alternate_pool_share", 0.01)) * ctx.N
    out = []
    for value in _keys(item, "acceptable_answers"):
        n = len(ctx.index.matching(rc.normalize(value)))
        if n > limit:
            out.append(_finding(rule, "alternate_too_generic",
                                f"Alternate {value!r} matches {n} pool records (limit {limit:.1f}).", "acceptable_answers", value))
    return out


def r_vocab_gap_verbatim(item, rule, ctx):
    if item.get("category") != "vocabulary_gap":
        return []
    na = rc.normalize(item.get("answer"))
    if any(rc.contains_phrase(g, na) for g in _gold_norms(item, ctx)):
        return [_finding(rule, "vocab_gap_answer_verbatim_in_gold",
                         "vocabulary_gap answer appears verbatim in its gold record.", "answer", item.get("answer"))]
    return []


def r_vocab_gap_overlap(item, rule, ctx):
    if item.get("category") != "vocabulary_gap":
        return []
    qt = set(rc.tokenize(item.get("question")))
    gt = set(" ".join(_gold_norms(item, ctx)).split())
    ratio = len(qt & gt) / len(qt) if qt else 0.0
    if ratio > rule["match"].get("max_overlap", 0.55):
        return [_finding(rule, "vocab_gap_overlap_too_high", f"Question/gold token overlap too high: {ratio:.2f}",
                         value=round(ratio, 3))]
    return []


def r_temporal_signal(item, rule, ctx):
    if item.get("category") != "temporal_awareness":
        return []
    terms = [rc.normalize(t) for t in ctx.const.get("temporal_terms", [])]
    hay = rc.normalize(item.get("question")) + " " + " ".join(_gold_norms(item, ctx))
    if not any(rc.contains_phrase(hay, t) for t in terms) and not re.search(r"\d", hay):
        return [_finding(rule, "temporal_category_without_temporal_signal",
                         "temporal_awareness item has no temporal word or number in the question or gold record.")]
    return []


def r_implicit_explicitness(item, rule, ctx):
    if item.get("category") != "implicit_preference":
        return []
    at = set(rc.tokenize(item.get("answer")))
    if not at:
        return []
    gold_norms = _gold_norms(item, ctx)
    gt = set(" ".join(gold_norms).split())
    ratio = len(at & gt) / len(at)
    stated = ctx.const.get("stated_preference_terms")
    if stated is not None and not any(rc.contains_phrase(g, rc.normalize(t)) for g in gold_norms for t in stated):
        return []  # the record shows the preference without stating it, so the thing's own name is a fine answer
    if ratio >= rule["match"].get("max_ratio", 0.85):
        return [_finding(rule, "implicit_preference_is_explicit",
                         f"Answer appears too explicitly in the gold record for implicit_preference: {ratio:.2f}",
                         value=round(ratio, 3))]
    return []


_CAP = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_SENTENCE_START = re.compile(r"(?:^|[.?!:]\s+)$")


def r_entity_coherence(item, rule, ctx):
    gold_text = " ".join(_gold_norms(item, ctx))
    if not gold_text:
        return []
    skip = {w.lower() for w in ctx.const.get("question_words", [])}
    question = str(item.get("question", ""))
    missing = []
    for m in _CAP.finditer(question):
        if _SENTENCE_START.search(question[:m.start()]):
            continue  # a capital at the start of a sentence is not a name
        low = rc.normalize(m.group(0))
        if low in skip or low in missing:
            continue
        if not rc.contains_phrase(gold_text, low):
            missing.append(low)
    if missing:
        return [_finding(rule, "entity_not_in_gold",
                         f"Capitalized term(s) in the question appear in no gold record: {', '.join(missing)}",
                         "question", missing)]
    return []


def r_gold_near_copy(item, rule, ctx):
    """A record outside the gold set that is a near copy of a gold record (the same
    comment with a username prefix, reordered sentences) probably holds the same
    memory and belongs in gold_ids; flagged for review, not decided here."""
    gold = item.get("gold_ids") or []
    if not gold:
        return []
    threshold = rule["match"].get("min_overlap", 0.9)
    gold_tokens = [ctx.tokens_by_id[g] for g in gold if g in ctx.tokens_by_id]
    near = []
    for rid in ctx.candidates_sharing_tokens(gold):
        if rid in gold:
            continue
        toks = ctx.tokens_by_id[rid]
        for gt in gold_tokens:
            if min(rc.token_overlap(gt, toks), rc.token_overlap(toks, gt)) >= threshold:
                near.append(rid)
                break
    if near:
        return [_finding(rule, "gold_near_copy_outside_gold",
                         f"Record(s) {', '.join(near[:5])} are near copies of a gold record but not gold.",
                         "gold_ids", near)]
    return []


def r_secrets_pii_item(item, rule, ctx):
    out = []
    for field in ("question", "answer", "acceptable_answers", "review_notes"):
        for value in _keys(item, field):
            for hit in scan_secrets_pii(value):
                out.append(_finding(rule, f"{hit['kind']}_{hit['shape']}",
                                    f"'{field}' matches a {hit['kind']} shape ({hit['shape']}).", field))
    return out


ITEM_HANDLERS: Dict[str, Callable] = {
    "required_fields": r_required_fields,
    "enum_membership": r_enum_membership,
    "gold_shape": r_gold_shape,
    "gold_present": r_gold_present,
    "abstention_shape": r_abstention_shape,
    "multi_record_shape": r_multi_record_shape,
    "source_length": r_source_length,
    "min_length": r_min_length,
    "prohibited_values": r_prohibited_values,
    "role_label": r_role_label,
    "entity_not_pronoun": r_entity_not_pronoun,
    "quantity_unit": r_quantity_unit,
    "key_edges": r_key_edges,
    "answer_not_in_question": r_answer_not_in_question,
    "gold_multiplicity": r_gold_multiplicity,
    "alternate_specificity": r_alternate_specificity,
    "vocab_gap_verbatim": r_vocab_gap_verbatim,
    "vocab_gap_overlap": r_vocab_gap_overlap,
    "temporal_signal": r_temporal_signal,
    "implicit_explicitness": r_implicit_explicitness,
    "entity_coherence": r_entity_coherence,
    "secrets_pii_item": r_secrets_pii_item,
    "gold_near_copy": r_gold_near_copy,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _update_status(current: str, new: Optional[str]) -> str:
    if new and STATUS_PRECEDENCE.get(new, -1) > STATUS_PRECEDENCE.get(current, -1):
        return new
    return current


def lint_item(item: Dict[str, Any], idx: int, ctx: Context) -> Dict[str, Any]:
    findings: List[Finding] = []
    status = item.get("review_status") or "publish"
    for rule in ctx.spec.get("rules", []):
        handler = ITEM_HANDLERS.get(rule["match"]["type"])
        if handler is None:
            raise ValueError(f"lint spec rule {rule['id']} has no handler for type {rule['match']['type']!r}")
        for f in handler(item, rule, ctx):
            findings.append(f)
            status = _update_status(status, f.suggested_review_status)
    has_error = any(f.severity == "error" for f in findings)
    return {
        "index": idx,
        "id": item.get("id"),
        "category": item.get("category"),
        "answer_type": item.get("answer_type"),
        "current_review_status": item.get("review_status"),
        "proposed_review_status": status,
        "findings": [asdict(f) for f in findings],
        "publishable": status == "publish" and not has_error,
    }


def set_findings(questions: List[Dict[str, Any]], ctx: Context) -> List[Dict[str, Any]]:
    out = []
    for rule in ctx.spec.get("set_rules", []):
        t = rule["type"]
        groups: Dict[Any, List[str]] = defaultdict(list)
        if t == "unique_field":
            for q in questions:
                groups[q.get(rule["field"])].append(q.get("id"))
        elif t == "unique_question_text":
            for q in questions:
                groups[rc.normalize(q.get("question"))].append(q.get("id"))
        elif t == "duplicate_fact":
            for q in questions:
                if q.get("gold_ids"):
                    groups[(tuple(sorted(q["gold_ids"])), rc.normalize(q.get("answer")))].append(q.get("id"))
        else:
            raise ValueError(f"lint spec set rule {rule['id']} has unknown type {t!r}")
        for key, ids in groups.items():
            if len(ids) > 1:
                out.append({"rule_id": rule["id"], "severity": rule["severity"], "code": t,
                            "message": f"{len(ids)} items share the same {'fact' if t == 'duplicate_fact' else 'value'}",
                            "ids": ids})
    return out


def pool_findings(ctx: Context) -> List[Dict[str, Any]]:
    out = []
    for rule in ctx.spec.get("pool_rules", []):
        if rule["type"] != "secrets_pii_pool":
            raise ValueError(f"lint spec pool rule {rule['id']} has unknown type {rule['type']!r}")
        for rec in ctx.records:
            for hit in scan_secrets_pii(rec.get("text")):
                out.append({"rule_id": rule["id"], "severity": rule["severity"], "code": f"{hit['kind']}_{hit['shape']}",
                            "record": rec["id"]})
    return out


def _ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def run_gates(questions, items, sets, pools, ctx: Context, readme_text: Optional[str],
              qset_version: Optional[str], distractors) -> List[Dict[str, Any]]:
    by_rule = Counter()
    for it in items:
        for f in it["findings"]:
            by_rule[f["rule_id"]] += 1
    for f in sets + pools:
        by_rule[f["rule_id"]] += 1
    results = []
    for gate in ctx.spec.get("gates", []):
        t = gate["type"]
        applicable = True
        if t == "no_findings":
            counts = {r: by_rule.get(r, 0) for r in gate["rules"]}
            passed = sum(counts.values()) == 0
            detail = {"findings": counts}
        elif t == "no_severity":
            offenders = [it["id"] for it in items if any(f["severity"] == gate["severity"] for f in it["findings"])]
            passed = not offenders
            detail = {"items": len(offenders), "ids": offenders[:50]}
        elif t == "category_clean":
            offenders = [it["id"] for it in items if it["category"] == gate["category"] and (
                any(f["severity"] == "error" for f in it["findings"])
                or it["proposed_review_status"] not in gate["allowed_statuses"])]
            passed = not offenders
            detail = {"items": len(offenders), "ids": offenders[:50]}
        elif t == "entity_attribution_resolves":
            offenders = [it["id"] for it in items if it["category"] == "entity_attribution" and (
                it["answer_type"] not in gate["answer_types"]
                or any(f["rule_id"] in gate["rules"] for f in it["findings"]))]
            passed = not offenders
            detail = {"items": len(offenders), "ids": offenders[:50]}
        elif t == "versions_match":
            problems = []
            cats = ctx.spec.get("enums", {}).get("category", [])
            if tuple(cats) != rc.CATEGORIES:
                problems.append("lint spec categories differ from recall_common.CATEGORIES")
            unknown = sorted({q.get("category") for q in questions} - set(cats))
            if unknown:
                problems.append(f"categories outside the enum: {unknown}")
            if qset_version is None:
                problems.append("no question-set version given (pass --qset-version or a manifest.json)")
            elif not rc.is_current_qset(qset_version):
                problems.append(f"question set is {qset_version}, harness expects {rc.CURRENT_QSET} "
                                f"(or its sealed half, {rc.CURRENT_QSET}{rc.SEALED_SUFFIX})")
            if readme_text is None:
                problems.append("README not found")
            else:
                for needle, what in ((rc.CURRENT_QSET, "question-set version"), (rc.HARNESS_VERSION, "harness version"),
                                     (f"{len(rc.CATEGORIES)} categories", "category count")):
                    if needle not in readme_text:
                        problems.append(f"README does not state the {what} ({needle!r})")
            passed = not problems
            detail = {"problems": problems}
        elif t == "publish_only":
            offenders = [q.get("id") for q in questions if q.get("review_status") != "publish"]
            passed = not offenders
            detail = {"items": len(offenders), "ids": offenders[:50]}
        elif t == "category_minimums":
            need = int(ctx.const.get("min_category_items", 25))
            counts = Counter(it["category"] for it in items if it["publishable"])
            short = {c: counts.get(c, 0) for c in ctx.spec["enums"]["category"] if counts.get(c, 0) < need}
            passed = not short
            detail = {"minimum": need, "below_minimum": short, "counts": dict(sorted(counts.items()))}
        elif t == "distractor_match":
            if not distractors:
                applicable, passed = False, True
                detail = {"note": "no distractor file given; nothing to compare"}
            else:
                tol = float(ctx.const.get("distractor_ks_tolerance", 0.1))
                corpus = ctx.records[:ctx.n_corpus]
                ks_len = _ks_statistic([len(rc.tokenize(r["text"])) for r in corpus],
                                       [len(rc.tokenize(r["text"])) for r in distractors])
                c_ts = sorted(r["ts"] for r in corpus if r.get("ts"))
                d_ts = sorted(r["ts"] for r in distractors if r.get("ts"))
                ts_ok, ks_ts = True, None
                if c_ts or d_ts:
                    if not c_ts or not d_ts:
                        ts_ok = False
                    else:
                        rank = {v: i for i, v in enumerate(sorted(set(c_ts + d_ts)))}
                        ks_ts = _ks_statistic([rank[v] for v in c_ts], [rank[v] for v in d_ts])
                        ts_ok = ks_ts <= tol
                passed = ks_len <= tol and ts_ok
                detail = {"tolerance": tol, "ks_token_length": round(ks_len, 4),
                          "ks_ts": None if ks_ts is None else round(ks_ts, 4), "ts_present_on_one_side_only": not ts_ok and ks_ts is None}
        else:
            raise ValueError(f"lint spec gate {gate['id']} has unknown type {t!r}")
        results.append({"id": gate["id"], "passed": bool(passed), "applicable": applicable,
                        "prose": gate.get("prose"), "detail": detail})
    return results


def lint(questions: List[Dict[str, Any]], records: List[Dict[str, Any]], spec: Dict[str, Any],
         distractors: Optional[List[Dict[str, Any]]] = None, readme_text: Optional[str] = None,
         qset_version: Optional[str] = None, store_ids: bool = False) -> Dict[str, Any]:
    """Lint a question set against its pool. Returns the report dict (no timestamps,
    so the same inputs always give the same bytes). store_ids lints a private set that
    keeps its store's own ids; the report then says so, and nothing else changes."""
    if tuple(spec.get("enums", {}).get("severity", rc.SEVERITIES)) != rc.SEVERITIES:
        raise ValueError("lint spec severity enum differs from recall_common.SEVERITIES")
    if tuple(spec.get("enums", {}).get("review_status", rc.STATUSES)) != rc.STATUSES:
        raise ValueError("lint spec review_status enum differs from recall_common.STATUSES")
    ctx = Context(records, spec, distractors, store_ids=store_ids)
    items = [lint_item(q, i, ctx) for i, q in enumerate(questions)]
    sets = set_findings(questions, ctx)
    pools = pool_findings(ctx)
    dup_ids = {i for f in sets if f["code"] == "duplicate_fact" for i in f["ids"][1:]}
    for it in items:
        if it["id"] in dup_ids:
            it["proposed_review_status"] = _update_status(it["proposed_review_status"], "reclassify")
            it["publishable"] = False
    gates = run_gates(questions, items, sets, pools, ctx, readme_text, qset_version, distractors)

    severity = Counter()
    codes = Counter()
    for it in items:
        for f in it["findings"]:
            severity[f["severity"]] += 1
            codes[f["code"]] += 1
    for f in sets + pools:
        severity[f["severity"]] += 1
        codes[f["code"]] += 1
    failed = [g["id"] for g in gates if not g["passed"]]
    report = {
        "report_name": "Parsica Recall lint report",
        "spec_name": spec.get("spec_name"),
        "spec_version": spec.get("spec_version"),
        "recall_spec": rc.SPEC_VERSION,
        "pool_size": ctx.N,
        "summary": {
            "total_items": len(items),
            "publishable_items": sum(1 for it in items if it["publishable"]),
            "items_with_errors": sum(1 for it in items if any(f["severity"] == "error" for f in it["findings"])),
            "severity_counts": dict(sorted(severity.items())),
            "finding_code_counts": dict(sorted(codes.items())),
            "category_counts": dict(sorted(Counter(it["category"] for it in items).items())),
            "proposed_review_status_counts": dict(sorted(Counter(it["proposed_review_status"] for it in items).items())),
        },
        "gates": gates,
        "gates_passed": not failed,
        "failed_gates": failed,
        "set_findings": sets,
        "pool_findings": pools,
        "items": items,
    }
    if store_ids:
        report["id_scheme"] = "store"  # absent means content ids, so a content-id set's report keeps its bytes
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"Parsica Recall lint validator ({rc.SPEC_VERSION})")
    p.add_argument("--spec", default=DEFAULT_SPEC, help="Lint spec JSON")
    p.add_argument("--questions", default=os.path.join(DEFAULT_QSET_DIR, "questions.json"), help="questions.json")
    p.add_argument("--corpus", default=os.path.join(DEFAULT_QSET_DIR, "corpus.jsonl"), help="corpus.jsonl")
    p.add_argument("--distractors", default=None, help="Optional distractor JSONL, screened with the pool")
    p.add_argument("--readme", default=os.path.join(REPO_DIR, "README.md"), help="README checked by GATE_VERSIONS_MATCH")
    p.add_argument("--qset-version", default=None,
                   help="Question-set version (default: read from manifest.json next to the questions)")
    p.add_argument("--output", required=True, help="Lint report JSON to write")
    p.add_argument("--fail-on-errors", action="store_true", help="Exit 1 when any item carries an error finding")
    p.add_argument("--store-ids", action="store_true",
                   help="A private set that keeps its store's own ids: gold ids need only be non-empty strings")
    args = p.parse_args(argv)

    try:
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
        records = load_jsonl(args.corpus)
        distractors = load_jsonl(args.distractors) if args.distractors else None
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not isinstance(questions, list):
        print("ERROR: questions file must be a JSON list", file=sys.stderr)
        return 2
    readme = open(args.readme, encoding="utf-8").read() if os.path.exists(args.readme) else None
    version = args.qset_version
    if version is None:
        mpath = os.path.join(os.path.dirname(os.path.abspath(args.questions)), "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                version = json.load(f).get("qset_version")

    report = lint(questions, records, spec, distractors=distractors, readme_text=readme, qset_version=version,
                  store_ids=args.store_ids)
    report["questions_file"] = os.path.basename(args.questions)
    report["questions_sha256"] = rc.file_sha256(args.questions)
    report["corpus_file"] = os.path.basename(args.corpus)
    report["corpus_sha256"] = rc.file_sha256(args.corpus)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    s = report["summary"]
    print(f"Wrote lint report to {args.output}")
    print(f"  items {s['total_items']}, publishable {s['publishable_items']}, with errors {s['items_with_errors']}")
    print(f"  findings by severity: {s['severity_counts']}")
    for g in report["gates"]:
        state = "pass" if g["passed"] else "FAIL"
        if not g["applicable"]:
            state += " (not applicable)"
        print(f"  {g['id']:34} {state}")
    if args.fail_on_errors and s["items_with_errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

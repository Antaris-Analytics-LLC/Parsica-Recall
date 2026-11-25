"""Tests for the question builder's providers: OpenAI, Google and an endpoint of your own
(--base-url) beside Anthropic. The SDKs are faked, so no network, no key and no openai or
google-genai package is needed:

    python3 -m unittest discover -s tests
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_generate as tg  # noqa: E402  (the shared helpers: toolboxes, the fake writers, GenTestCase)

gq = tg.gq
from parsica_recall import recall_common as rc  # noqa: E402
ns = types.SimpleNamespace

# Key-shaped values, assembled from parts so no whole key sits in this file.
KEY_VAR = "PROVIDER_TEST_KEY"
OPENAI_KEY = "-".join(("sk", "proj", "ptest", "Qm4Xr7Vb2Nc8Wd5F" + "h1Kj6Lp3Ts9Ya0Ze"))
GOOGLE_KEY = "AIza" + "Gt7Lm2Qx9Rv4Bn6Y" + "c1Kd8Hj3Fs5Pw0Ze"
TARGETS = "direct_recall_user=4,paraphrase_recall=4"


def prompt(category="direct_recall_user"):
    return gq.load_prompts([category])[category]


def chat_response(text='{"items": []}', model="gpt-6.2-sol-2026-11-01", finish="stop", refusal=None,
                  prompt_tokens=1000, cached=600, written=100, completion=500, reasoning=200):
    """A Chat Completions response as the openai SDK returns it."""
    return ns(model=model, choices=[ns(finish_reason=finish, message=ns(content=text, refusal=refusal))],
              usage=ns(prompt_tokens=prompt_tokens, completion_tokens=completion,
                       prompt_tokens_details=ns(cached_tokens=cached, cache_write_tokens=written),
                       completion_tokens_details=ns(reasoning_tokens=reasoning)))


def enum(value):
    return ns(value=value)  # an SDK enum: its .value is its text


def gemini_response(parts=None, finish="STOP", blocked=None, model_version="gemini-4-hydrogen-001",
                    prompt_tokens=1000, cached=400, candidates=300, thoughts=200):
    """A generateContent response as the google-genai SDK returns it."""
    usage = ns(prompt_token_count=prompt_tokens, cached_content_token_count=cached,
               candidates_token_count=candidates, thoughts_token_count=thoughts)
    if blocked:
        return ns(candidates=[], prompt_feedback=ns(block_reason=enum(blocked)), model_version=model_version,
                  usage_metadata=ns(prompt_token_count=prompt_tokens, cached_content_token_count=None,
                                    candidates_token_count=None, thoughts_token_count=None))
    if parts is None:
        parts = [ns(text="weighing the records", thought=True), ns(text='{"items": ', thought=None),
                 ns(text="[]}", thought=None)]
    return ns(candidates=[ns(finish_reason=enum(finish), content=ns(parts=parts))], prompt_feedback=None,
              model_version=model_version, usage_metadata=usage)


def fake_openai(clients, sent, respond=None, create_fail=None, client_fail=None):
    """A stand-in for the openai package: OpenAI(**kw) records its arguments, and
    chat.completions.create(**kw) records the request and answers with respond(kw)."""
    mod = types.ModuleType("openai")
    mod.__version__ = "3.57.0"

    class OpenAIError(Exception):
        pass

    class AuthenticationError(OpenAIError):
        pass

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            if create_fail is not None:
                raise create_fail
            return respond(kw) if respond else chat_response()

    class OpenAI:
        def __init__(self, **kw):
            if client_fail is not None:
                raise client_fail
            clients.append(kw)
            self.chat = ns(completions=Completions())

    mod.OpenAI, mod.OpenAIError, mod.AuthenticationError = OpenAI, OpenAIError, AuthenticationError
    return mod


def fake_google(clients, sent, respond=None, create_fail=None, client_fail=None):
    """A stand-in for the google-genai package, as the modules `from google import genai`
    reads: Client(**kw) records its arguments, and models.generate_content(**kw) records
    the request and answers with respond(kw)."""
    genai = types.ModuleType("google.genai")
    genai.__version__ = "3.6.1"

    class Models:
        def generate_content(self, **kw):
            sent.append(kw)
            if create_fail is not None:
                raise create_fail
            return respond(kw) if respond else gemini_response()

    class Client:
        def __init__(self, **kw):
            if client_fail is not None:
                raise client_fail
            clients.append(kw)
            self.models = Models()

    genai.Client = Client
    google = types.ModuleType("google")
    google.genai = genai
    return {"google": google, "google.genai": genai}


class ChoiceTests(unittest.TestCase):
    def test_each_provider_has_its_default_model_endpoint_cap_and_effort(self):
        a, o, g = gq.ModelChoice(), gq.ModelChoice("openai"), gq.ModelChoice("google")
        self.assertEqual((a.model_id, a.endpoint, a.max_tokens, a.effort), ("claude-sonnet-5-5", None, 16000, "high"))
        self.assertEqual((o.model_id, o.endpoint, o.max_tokens, o.effort),
                         ("gpt-6.2-sol", "https://api.openai.com/v1", 32000, "high"))
        self.assertEqual((g.model_id, g.endpoint, g.max_tokens, g.effort),
                         ("gemini-4-hydrogen", "https://generativelanguage.googleapis.com/", 32000, "high"))
        self.assertEqual(o.price, (2.00, 10.00, 2.50, 2.50, 0.10))
        self.assertEqual(g.price, (1.50, 7.50, 1.50, 1.50, 0.075))
        self.assertIn("2026-11-21", o.price_source)
        self.assertFalse(a.other)
        self.assertTrue(o.other and g.other)

    def test_an_endpoint_of_your_own(self):
        c = gq.ModelChoice("openai", "llama3.3", base_url=" http://localhost:11434/v1/ ")
        self.assertEqual((c.endpoint, c.max_tokens, c.effort, c.price), ("http://localhost:11434/v1", None, None, None))
        self.assertTrue(c.custom and c.priced_record())
        c = gq.ModelChoice("openai", "gpt-6.2-sol", base_url="https://proxy.example.com/v1")
        self.assertIsNone(c.price)  # the table never prices an endpoint of your own, whatever the model is called
        c = gq.ModelChoice("openai", "m", base_url="https://proxy.example.com/v1", price_in=0, price_out=0,
                           max_output_tokens=4000)
        self.assertEqual((c.price, c.max_tokens), ((0, 0, 0, 0, 0), 4000))
        c = gq.ModelChoice("openai", "m", base_url="http://127.0.0.1:8000/v1", api_key_env=KEY_VAR)
        self.assertEqual(c.endpoint, "http://127.0.0.1:8000/v1")  # a key may go over http to this machine

    def test_what_a_choice_refuses(self):
        cases = [
            (dict(provider="mistral"), "--provider takes one of"),
            (dict(provider="google", model_id="m", base_url="https://x.example.com"), "goes with --provider openai"),
            (dict(provider="openai", base_url="https://x.example.com"), "--base-url needs --model"),
            (dict(provider="openai", model_id="m", base_url="localhost:11434"), "an http:// or https:// URL"),
            (dict(provider="openai", model_id="m", base_url="https://" + "user:pw" + "@" + "x.example.com"),
             "user name or password"),  # assembled from parts, as a credential-shaped fixture must be
            (dict(provider="openai", model_id="m", base_url="https://x.example.com/v1?k=1"), "no ? or # part"),
            (dict(provider="openai", model_id="m", base_url="http://x.example.com/v1", api_key_env=KEY_VAR),
             "over https only"),
            (dict(provider="openai", max_output_tokens=100), "--max-output-tokens goes with --base-url"),
            (dict(provider="openai", model_id="m", base_url="https://x.example.com", max_output_tokens=0), "1 or more"),
            (dict(provider="openai", price_in=1.0), "--price-in and --price-out go together"),
            (dict(provider="openai", price_in=-1.0, price_out=1.0), "--price-in takes a number"),
            (dict(provider="openai", price_in=1.0, price_out=float("nan")), "--price-out takes a number"),
            (dict(provider="google", model_id="gpt-6.2-sol"), "a model of openai's; run it with --provider openai"),
            (dict(provider="anthropic", model_id="gemini-4-carbon"), "--provider google"),
            (dict(provider="openai", api_key_env=OPENAI_KEY), "never the key itself"),
        ]
        for kw, message in cases:
            with self.subTest(kw=kw), self.assertRaises(gq.GenerateError) as cm:
                gq.ModelChoice(**kw)
            self.assertIn(message, str(cm.exception))
            self.assertNotIn(OPENAI_KEY, str(cm.exception))

    def test_prices_from_the_flags_replace_the_table(self):
        c = gq.ModelChoice("openai", price_in=3.0, price_out=12.0)
        self.assertEqual(c.price, (3.0, 12.0, 3.0, 3.0, 3.0))  # cache reads and writes at the input price
        self.assertTrue(c.flag_price and c.priced_record())
        self.assertEqual(c.prices()["usd_per_million_tokens"]["output"], 12.0)
        self.assertIn("--price-in", c.prices()["source"])


class RequestHashTests(unittest.TestCase):
    def test_an_anthropic_request_hashes_as_it_always_has(self):
        p = prompt()
        blob = json.dumps({"model": "claude-opus-5-5", "max_tokens": gq.MAX_TOKENS, "effort": gq.EFFORT,
                           "system_sha256": p.sha256, "system_cache": gq.SYSTEM_CACHE, "user": "x"},
                          sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        before = "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()
        self.assertEqual(gq.request_sha256("claude-opus-5-5", p, "x"), before)
        for choice in (gq.ModelChoice("anthropic", "claude-opus-5-5"),
                       gq.ModelChoice("anthropic", "claude-opus-5-5", price_in=1.0, price_out=2.0)):
            self.assertEqual(gq.request_sha256("claude-opus-5-5", p, "x", choice), before)

    def test_another_provider_or_endpoint_is_another_request(self):
        p = prompt()
        hashes = {gq.request_sha256("m", p, "x", c) for c in (
            gq.ModelChoice("anthropic", "m"), gq.ModelChoice("openai", "m"), gq.ModelChoice("google", "m"),
            gq.ModelChoice("openai", "m", base_url="http://localhost:11434/v1"),
            gq.ModelChoice("openai", "m", base_url="http://localhost:8000/v1"),
            gq.ModelChoice("openai", "m", base_url="http://localhost:8000/v1", max_output_tokens=900))}
        self.assertEqual(len(hashes), 6)
        # the price is not part of the request
        self.assertEqual(gq.request_sha256("m", p, "x", gq.ModelChoice("openai", "m")),
                         gq.request_sha256("m", p, "x", gq.ModelChoice("openai", "m", price_in=1.0, price_out=1.0)))


class OpenAIModelTests(unittest.TestCase):
    def test_the_request_has_the_documented_shape(self):
        clients, sent = [], []
        p = prompt()
        with mock.patch.dict(sys.modules, {"openai": fake_openai(clients, sent)}):
            model = gq.make_model(gq.ModelChoice("openai"))
            self.assertIsInstance(model, gq.OpenAIModel)
            reply = model.call("direct_recall_user", p, "the records", "sha256:x")
        self.assertEqual(clients, [{"base_url": "https://api.openai.com/v1", "max_retries": 2}])  # the SDK reads its own key
        self.assertEqual(sent, [{
            "model": "gpt-6.2-sol",
            "messages": [{"role": "system", "content": p.text}, {"role": "user", "content": "the records"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "recall_direct_recall_user", "schema": p.schema, "strict": True}},
            "reasoning_effort": "high", "max_completion_tokens": 32000, "store": False}])
        # prompt_tokens holds the cached and the cache-written tokens; input_tokens is the rest
        self.assertEqual((reply.text, reply.model, reply.stop_reason, reply.input_tokens, reply.output_tokens,
                          reply.cache_write, reply.cache_read, reply.reasoning_tokens, reply.sdk),
                         ('{"items": []}', "gpt-6.2-sol-2026-11-01", "stop", 300, 500, 100, 600, 200, "openai 3.57.0"))
        self.assertEqual(gq.parse_reply(reply), ([], None))

    def test_a_refusal_and_a_cut_reply_yield_no_items(self):
        for resp, why in ((chat_response(text=None, refusal="I can't help with that."), "declined"),
                          (chat_response(finish="content_filter"), "declined"),
                          (chat_response(finish="length", text='{"items": [{"quest'), "max_tokens")):
            with mock.patch.dict(sys.modules, {"openai": fake_openai([], [], respond=lambda kw, r=resp: r)}):
                reply = gq.OpenAIModel(gq.ModelChoice("openai")).call("direct_recall_user", prompt(), "u", "s")
            items, reason = gq.parse_reply(reply)
            self.assertEqual(items, [])
            self.assertIn(why, reason)

    def test_an_endpoint_of_your_own_is_never_sent_openai_api_key(self):
        clients, sent = [], []
        p = prompt()
        with mock.patch.dict(sys.modules, {"openai": fake_openai(clients, sent)}), \
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": OPENAI_KEY}):
            gq.make_model(gq.ModelChoice("openai", "llama3.3", base_url="http://localhost:11434/v1")).call(
                "direct_recall_user", p, "u", "s")
            gq.make_model(gq.ModelChoice("openai", "llama3.3", base_url="http://localhost:11434/v1",
                                         max_output_tokens=4000)).call("direct_recall_user", p, "u", "s")
        self.assertEqual(clients[0], {"base_url": "http://localhost:11434/v1", "max_retries": 2, "api_key": "none"})
        self.assertNotIn(OPENAI_KEY, json.dumps(clients))
        self.assertEqual(set(sent[0]), {"model", "messages", "response_format"})  # no field a server may not know
        self.assertEqual(sent[1]["max_tokens"], 4000)

    def test_the_named_variable_gives_the_key_and_no_error_quotes_it(self):
        clients = []
        with mock.patch.dict(os.environ, {KEY_VAR: OPENAI_KEY}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai(clients, [])}):
                gq.make_model(gq.ModelChoice("openai", api_key_env=KEY_VAR)).connect()
                gq.make_model(gq.ModelChoice("openai", "m", base_url="https://proxy.example.com/v1",
                                             api_key_env=KEY_VAR)).connect()
            self.assertEqual(clients, [{"base_url": "https://api.openai.com/v1", "max_retries": 2, "api_key": OPENAI_KEY},
                                       {"base_url": "https://proxy.example.com/v1", "max_retries": 2,
                                        "api_key": OPENAI_KEY}])
            fail = RuntimeError(f"401 Unauthorized: key {OPENAI_KEY} is not valid")
            with mock.patch.dict(sys.modules, {"openai": fake_openai([], [], create_fail=fail)}), \
                    self.assertRaises(gq.GenerateError) as cm:
                gq.make_model(gq.ModelChoice("openai", api_key_env=KEY_VAR)).call("direct_recall_user", prompt(), "u", "s")
        self.assertIn("[the key]", str(cm.exception))
        self.assertNotIn(OPENAI_KEY, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)

    def test_a_missing_package_or_key_says_what_to_do(self):
        with mock.patch.dict(sys.modules, {"openai": None}), self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("openai")).connect()
        self.assertIn('pip install "parsica-recall[openai]"', str(cm.exception))
        nokey = Exception("The api_key client option must be set")
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], [], client_fail=nokey)}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("openai")).connect()
        self.assertIn("check that OPENAI_API_KEY is set", str(cm.exception))

    def test_check_api_says_ok_or_the_error_class_only(self):
        sent = []
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], sent)}):
            self.assertEqual(gq.check_api(choice=gq.ModelChoice("openai")), "gpt-6.2-sol-2026-11-01")
        self.assertEqual(sent[0]["max_completion_tokens"], 256)
        self.assertEqual((sent[0]["reasoning_effort"], sent[0]["store"]), ("low", False))
        fail = fake_openai([], []).AuthenticationError(f"bad key {OPENAI_KEY}")
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], [], create_fail=fail)}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.check_api(choice=gq.ModelChoice("openai"))
        self.assertEqual(str(cm.exception), "AuthenticationError")


class GoogleModelTests(unittest.TestCase):
    def test_the_request_has_the_documented_shape(self):
        clients, sent = [], []
        p = prompt()
        with mock.patch.dict(sys.modules, fake_google(clients, sent)):
            model = gq.make_model(gq.ModelChoice("google"))
            self.assertIsInstance(model, gq.GoogleModel)
            reply = model.call("direct_recall_user", p, "the records", "sha256:x")
        self.assertEqual(clients, [{"vertexai": False, "http_options": {
            "base_url": "https://generativelanguage.googleapis.com/", "retry_options": {"attempts": 3}}}])
        self.assertEqual(sent, [{"model": "gemini-4-hydrogen", "contents": "the records", "config": {
            "system_instruction": p.text, "response_mime_type": "application/json", "response_json_schema": p.schema,
            "max_output_tokens": 32000, "thinking_config": {"thinking_level": "HIGH"},
            "automatic_continuation": False}}])
        # the thought part is left out; the thoughts are billed as output; the cached tokens are a part of the prompt
        self.assertEqual((reply.text, reply.model, reply.stop_reason, reply.input_tokens, reply.output_tokens,
                          reply.cache_write, reply.cache_read, reply.reasoning_tokens, reply.sdk),
                         ('{"items": []}', "gemini-4-hydrogen-001", "STOP", 600, 500, 0, 400, 200, "google-genai 3.6.1"))
        self.assertEqual(gq.parse_reply(reply), ([], None))

    def test_a_blocked_prompt_a_filter_and_a_cut_reply_yield_no_items(self):
        for resp, why in ((gemini_response(blocked="SAFETY"), "declined"),
                          (gemini_response(finish="PROHIBITED_CONTENT"), "declined"),
                          (gemini_response(finish="MAX_TOKENS"), "max_tokens")):
            with mock.patch.dict(sys.modules, fake_google([], [], respond=lambda kw, r=resp: r)):
                reply = gq.GoogleModel(gq.ModelChoice("google")).call("direct_recall_user", prompt(), "u", "s")
            items, reason = gq.parse_reply(reply)
            self.assertEqual(items, [])
            self.assertIn(why, reason)
        self.assertEqual(reply.stop_reason, "MAX_TOKENS")

    def test_the_named_variable_gives_the_key_and_no_error_quotes_it(self):
        clients = []
        with mock.patch.dict(os.environ, {KEY_VAR: GOOGLE_KEY}):
            with mock.patch.dict(sys.modules, fake_google(clients, [])):
                gq.make_model(gq.ModelChoice("google", api_key_env=KEY_VAR)).connect()
            self.assertEqual(clients[0]["api_key"], GOOGLE_KEY)
            fail = RuntimeError(f"400 API key not valid: {GOOGLE_KEY}")
            with mock.patch.dict(sys.modules, fake_google([], [], create_fail=fail)), \
                    self.assertRaises(gq.GenerateError) as cm:
                gq.make_model(gq.ModelChoice("google", api_key_env=KEY_VAR)).call("direct_recall_user", prompt(), "u", "s")
        self.assertIn("[the key]", str(cm.exception))
        self.assertNotIn(GOOGLE_KEY, str(cm.exception))

    def test_a_missing_package_says_what_to_do_and_check_api_names_the_class_only(self):
        with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("google")).connect()
        self.assertIn('pip install "parsica-recall[google]"', str(cm.exception))
        sent = []
        with mock.patch.dict(sys.modules, fake_google([], sent)):
            self.assertEqual(gq.check_api(choice=gq.ModelChoice("google")), "gemini-4-hydrogen-001")
        self.assertEqual(sent[0]["config"]["thinking_config"], {"thinking_level": "LOW"})

        class ClientError(Exception):
            pass

        with mock.patch.dict(sys.modules, fake_google([], [], create_fail=ClientError(GOOGLE_KEY))), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.check_api(choice=gq.ModelChoice("google"))
        self.assertEqual(str(cm.exception), "ClientError")


class AzureAndBedrockChoiceTests(unittest.TestCase):
    def test_azure_is_openais_request_at_your_resource_and_bedrock_anthropics_in_your_region(self):
        a = gq.ModelChoice("azure", "astra-deploy", base_url=AZURE_URL + "/")
        self.assertEqual((a.model_id, a.endpoint, a.max_tokens, a.effort, a.custom, a.claude_request, a.price),
                         ("astra-deploy", AZURE_URL, 32000, "high", False, False, None))
        self.assertEqual(a.fields(), {"provider": "azure", "endpoint": AZURE_URL, "effort": "high", "max_tokens": 32000})
        b = gq.ModelChoice("bedrock", aws_region="us-west-2", price_in=2, price_out=10)
        self.assertEqual((b.model_id, b.endpoint, b.max_tokens, b.effort, b.claude_request),
                         ("anthropic.claude-sonnet-5-5", "https://bedrock-mantle.us-west-2.api.aws/anthropic", 16000,
                          "high", True))
        self.assertEqual(b.price, (2, 10, 2, 2, 2))
        # a deployment or a Bedrock id may carry another provider's name, which is no mistake there
        self.assertEqual(gq.ModelChoice("azure", "gpt-6.2-astra", base_url=AZURE_URL).model_id, "gpt-6.2-astra")
        self.assertEqual(gq.ModelChoice("bedrock", "us.anthropic.claude-fable-5-2", aws_region="us-west-1").model_id,
                         "us.anthropic.claude-fable-5-2")

    def test_bedrock_takes_its_region_from_the_flag_or_the_environment(self):
        with mock.patch.dict(os.environ, {"AWS_REGION": "eu-central-1"}):
            self.assertEqual(gq.ModelChoice("bedrock").region, "eu-central-1")
            self.assertEqual(gq.ModelChoice("bedrock", aws_region="us-east-1").region, "us-east-1")
        bare = {k: v for k, v in os.environ.items() if k not in ("AWS_REGION", "AWS_DEFAULT_REGION")}
        with mock.patch.dict(os.environ, bare, clear=True), self.assertRaises(gq.GenerateError) as cm:
            gq.ModelChoice("bedrock")
        self.assertIn("--aws-region", str(cm.exception))

    def test_what_they_refuse(self):
        cases = [
            (dict(provider="azure", model_id="x", base_url="https://" + "evil.example.com/openai/v1"), "Azure host"),
            (dict(provider="azure", model_id="x", base_url="http://" + "jr.openai.azure.com/openai/v1"), "https only"),
            (dict(provider="azure", model_id="x", base_url="https://" + "jr.openai.azure.com/v2"), "/openai/v1"),
            (dict(provider="azure", model_id="x"), "needs --base-url"),
            (dict(provider="azure", base_url=AZURE_URL), "name of your deployment"),
            (dict(provider="azure", model_id="x", base_url=AZURE_URL, max_output_tokens=5), "--max-output-tokens"),
            (dict(provider="bedrock", aws_region="West 2"), "region's name"),
            (dict(provider="bedrock", aws_region="us-west-2", base_url="https://" + "x.example.com/v1"), "--base-url"),
            (dict(provider="openai", aws_region="us-west-2"), "--aws-region goes with --provider bedrock"),
        ]
        for kw, words in cases:
            with self.subTest(kw=kw), self.assertRaises(gq.GenerateError) as cm:
                gq.ModelChoice(**kw)
            self.assertIn(words, str(cm.exception))


class AzureModelTests(unittest.TestCase):
    def test_the_request_is_openais_and_goes_to_your_resource_with_the_key(self):
        clients, sent = [], []
        p = prompt()
        with mock.patch.dict(sys.modules, {"openai": fake_openai(clients, sent)}), \
                mock.patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": AZURE_KEY}):
            model = gq.make_model(gq.ModelChoice("azure", "astra-deploy", base_url=AZURE_URL))
            self.assertIsInstance(model, gq.OpenAIModel)
            model.call("direct_recall_user", p, "the records", "sha256:x")
        self.assertEqual(clients, [{"base_url": AZURE_URL, "api_key": AZURE_KEY, "max_retries": 2,
                                    "default_headers": {"api-key": AZURE_KEY}}])
        self.assertEqual(sent, [{
            "model": "astra-deploy",
            "messages": [{"role": "system", "content": p.text}, {"role": "user", "content": "the records"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "recall_direct_recall_user", "schema": p.schema, "strict": True}},
            "reasoning_effort": "high", "max_completion_tokens": 32000, "store": False}])

    def test_the_named_variable_gives_the_key_and_no_error_quotes_it(self):
        clients = []
        with mock.patch.dict(sys.modules, {"openai": fake_openai(clients, [])}), \
                mock.patch.dict(os.environ, {KEY_VAR: AZURE_KEY}):
            gq.make_model(gq.ModelChoice("azure", "d", base_url=AZURE_URL, api_key_env=KEY_VAR)).connect()
        self.assertEqual(clients[0]["api_key"], AZURE_KEY)
        fail = RuntimeError(f"401 for key {AZURE_KEY}")
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], [], create_fail=fail)}), \
                mock.patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": AZURE_KEY}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("azure", "d", base_url=AZURE_URL)).call("direct_recall_user", prompt(), "u", "s")
        self.assertNotIn(AZURE_KEY, str(cm.exception))
        self.assertIn("[the key]", str(cm.exception))

    def test_a_missing_key_says_which_variable(self):
        bare = {k: v for k, v in os.environ.items() if k != "AZURE_OPENAI_API_KEY"}
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], [])}), \
                mock.patch.dict(os.environ, bare, clear=True), self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("azure", "d", base_url=AZURE_URL)).connect()
        self.assertIn("AZURE_OPENAI_API_KEY", str(cm.exception))


class BedrockModelTests(unittest.TestCase):
    def test_the_request_is_anthropics_through_the_bedrock_client(self):
        clients, sent = [], []
        p = prompt()
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic(clients, sent)}):
            model = gq.make_model(gq.ModelChoice("bedrock", aws_region="us-west-2"))
            self.assertIsInstance(model, gq.BedrockModel)
            reply = model.call("direct_recall_user", p, "the records", "sha256:x")
        # the SDK finds the AWS credentials itself; the address is pinned to the one recorded
        self.assertEqual(clients, [{"aws_region": "us-west-2",
                                    "base_url": "https://bedrock-mantle.us-west-2.api.aws/anthropic"}])
        self.assertEqual(sent, [{
            "model": "anthropic.claude-sonnet-5-5", "max_tokens": 16000, "system": gq.system_blocks(p),
            "messages": [{"role": "user", "content": "the records"}],
            "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": p.schema}}}])
        self.assertEqual((reply.model, reply.stop_reason, reply.cache_write, reply.cache_write_1h, reply.sdk),
                         ("anthropic.claude-sonnet-5-5", "end_turn", 3000, 3000, "anthropic 2.12.0"))

    def test_a_bedrock_api_key_from_the_named_variable_and_no_error_quotes_it(self):
        clients = []
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic(clients, [])}), \
                mock.patch.dict(os.environ, {KEY_VAR: BEDROCK_KEY}):
            gq.make_model(gq.ModelChoice("bedrock", aws_region="us-west-2", api_key_env=KEY_VAR)).connect()
        self.assertEqual(clients[0]["api_key"], BEDROCK_KEY)
        fail = RuntimeError(f"denied for {BEDROCK_KEY}")
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic([], [], client_fail=fail)}), \
                mock.patch.dict(os.environ, {KEY_VAR: BEDROCK_KEY}), self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("bedrock", aws_region="us-west-2", api_key_env=KEY_VAR)).connect()
        self.assertNotIn(BEDROCK_KEY, str(cm.exception))

    def test_a_missing_package_or_an_sdk_too_old_says_what_to_do(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}), self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("bedrock", aws_region="us-west-2")).connect()
        self.assertIn('"parsica-recall[bedrock]"', str(cm.exception))
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic([], [], mantle=False)}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.make_model(gq.ModelChoice("bedrock", aws_region="us-west-2")).connect()
        self.assertIn("no Bedrock client", str(cm.exception))

    def test_check_api_says_ok_or_the_error_class_only(self):
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic([], [])}):
            self.assertEqual(gq.check_api(choice=gq.ModelChoice("bedrock", aws_region="us-west-2")),
                             "anthropic.claude-sonnet-5-5")
        fail = RuntimeError(f"AccessDenied with {BEDROCK_KEY}")
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic([], [], create_fail=fail)}), \
                self.assertRaises(gq.GenerateError) as cm:
            gq.check_api(choice=gq.ModelChoice("bedrock", aws_region="us-west-2"))
        self.assertEqual(str(cm.exception), "RuntimeError")


class CostTests(unittest.TestCase):
    def test_a_record_is_priced_by_its_own_price_or_the_table(self):
        rec = {"model": "gpt-6.2-sol", "input_tokens": 300, "output_tokens": 500, "cache_creation_input_tokens": 100,
               "cache_creation_1h_input_tokens": 0, "cache_read_input_tokens": 600}
        self.assertAlmostEqual(gq.record_cost(rec), (300 * 2.0 + 500 * 10.0 + 100 * 2.5 + 600 * 0.10) / 1e6)
        self.assertAlmostEqual(gq.record_cost(dict(rec, price_usd_per_million=[1, 4, 1, 1, 1])),
                               (300 + 500 * 4 + 100 + 600) / 1e6)
        self.assertIsNone(gq.record_cost(dict(rec, price_usd_per_million=None)))  # an endpoint of your own, unpriced
        self.assertAlmostEqual(gq.cost_exact("gemini-4-carbon", 1_000_000, 1_000_000), 24.0)  # December's prices


def sdk_writer(prompts):
    """What a provider would return for each request: the items the shared fake writer
    gives for the records in the user message, as a JSON reply."""
    category_of = {p.text: c for c, p in prompts.items()}

    def items(category, user):
        return json.dumps({"items": tg.toolbox_writer(category, gq.parse_user_message(user))[:1]})

    def openai(kw):
        category = kw["response_format"]["json_schema"]["name"][len("recall_"):]
        return chat_response(text=items(category, kw["messages"][1]["content"]), prompt_tokens=20_000, cached=5_000,
                             written=2_000, completion=3_000, reasoning=1_000)

    def google(kw):
        category = category_of[kw["config"]["system_instruction"]]
        return gemini_response(parts=[ns(text=items(category, kw["contents"]), thought=None)], prompt_tokens=20_000,
                               cached=0, candidates=2_000, thoughts=1_000)

    def bedrock(kw):
        category = category_of[kw["system"][0]["text"]]
        return claude_response(text=items(category, kw["messages"][0]["content"]), model=kw["model"])

    return {"openai": openai, "google": google, "bedrock": bedrock}


def claude_response(text='{"items": []}', model="anthropic.claude-sonnet-5-5", stop="end_turn", input_tokens=900,
                    output_tokens=700, written=3_000, read=0):
    """A Messages API response as the anthropic SDK returns it, on Bedrock as anywhere."""
    usage = ns(input_tokens=input_tokens, output_tokens=output_tokens, cache_creation_input_tokens=written,
               cache_read_input_tokens=read, cache_creation=ns(ephemeral_1h_input_tokens=written))
    return ns(model=model, stop_reason=stop, content=[ns(type="text", text=text)], usage=usage)


def fake_anthropic(clients, sent, respond=None, create_fail=None, client_fail=None, mantle=True):
    """A stand-in for the anthropic package as the Bedrock adapter reads it:
    AnthropicBedrockMantle(**kw) records its arguments, and messages.create(**kw) records the
    request and answers with respond(kw). mantle=False is an SDK too old for Bedrock's client."""
    mod = types.ModuleType("anthropic")
    mod.__version__ = "2.12.0"

    class AnthropicError(Exception):
        pass

    class APIError(AnthropicError):
        pass

    class Messages:
        def create(self, **kw):
            sent.append(kw)
            if create_fail is not None:
                raise create_fail
            return respond(kw) if respond else claude_response(model=kw["model"])

    class AnthropicBedrockMantle:
        def __init__(self, **kw):
            if client_fail is not None:
                raise client_fail
            clients.append(kw)
            self.messages = Messages()

    mod.AnthropicError, mod.APIError = AnthropicError, APIError
    if mantle:
        mod.AnthropicBedrockMantle = AnthropicBedrockMantle
    return mod


# Key-shaped values for Azure and Bedrock, assembled from parts like the others.
AZURE_KEY = "".join(("4f1c", "9e2b7a6d", "3c8e5f0a", "1b4d7e9c", "2a6f8b3d"))
BEDROCK_KEY = "ABSK" + "QmVkcm9ja0Fw" + "aUtleS10ZXN0"
AZURE_URL = "https://" + "jr-test" + ".openai.azure.com/openai/v1"


class BuildTests(tg.GenTestCase):
    """A whole set built through each adapter, with the SDK faked: the call log, run.json
    and the manifest record the provider, and the set replays byte for byte."""

    def setUp(self):
        super().setUp()
        self.corpus_path = self.corpus(tg.toolboxes(80))
        self.respond = sdk_writer(gq.load_prompts(gq.parse_targets(TARGETS)))

    def build(self, out, choice, **kw):
        return gq.generate_set(self.corpus_path, os.path.join(self.tmp, out), 7, gq.parse_targets(TARGETS),
                               model_id=choice.model_id, choice=choice, quiet=True, **kw)

    def sdk(self, provider):
        if provider in ("openai", "azure"):
            return {"openai": fake_openai([], [], respond=self.respond["openai"])}
        if provider == "bedrock":
            return {"anthropic": fake_anthropic([], [], respond=self.respond["bedrock"])}
        return fake_google([], [], respond=self.respond["google"])

    def test_a_set_built_on_azure_and_on_bedrock_records_them_and_replays(self):
        bedrock_url = "https://bedrock-mantle.us-west-2.api.aws/anthropic"
        for provider, choice, endpoint, cache in (
                ("azure", gq.ModelChoice("azure", "astra-deploy", base_url=AZURE_URL, price_in=8, price_out=40),
                 AZURE_URL, None),
                ("bedrock", gq.ModelChoice("bedrock", aws_region="us-west-2", price_in=2, price_out=10),
                 bedrock_url, gq.SYSTEM_CACHE)):
            with self.subTest(provider=provider):
                with mock.patch.dict(sys.modules, self.sdk(provider)), \
                        mock.patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": AZURE_KEY}):
                    res = self.build(f"{provider}/set-p", choice, max_cost=5.0)
                self.assertTrue(res["check"]["ok"])
                calls = self.read(res, "calls.jsonl")
                self.assertTrue(calls)
                self.assertEqual({(c["provider"], c["endpoint"], c["model"]) for c in calls},
                                 {(provider, endpoint, choice.model_id)})
                self.assertTrue(all(c["price_usd_per_million"] == list(choice.price) for c in calls))
                run = self.read(res, "run.json")
                self.assertEqual((run["provider"], run["endpoint"], run["effort"], run["max_tokens"]),
                                 (provider, endpoint, "high", choice.max_tokens))
                m = res["manifest"]["model"]
                self.assertEqual((m["provider"], m["endpoint"], m["effort"], m["max_tokens"], m["system_cache"]),
                                 (provider, endpoint, "high", choice.max_tokens, cache))
                self.assertEqual(rc.writer_label(m), f"{provider}/{choice.model_id}")
                log = os.path.join(res["out"], "generation", "calls.jsonl")
                with mock.patch.dict(sys.modules, {"openai": None, "anthropic": None}):
                    again = self.build(f"{provider}/set-r", choice, replay=log)
                self.assertEqual(self.read(again, "calls.jsonl"), calls)
                self.assertEqual(again["manifest"]["model"], m)

    def test_azure_and_bedrock_need_prices_for_the_guard(self):
        for choice, platform in ((gq.ModelChoice("azure", "d", base_url=AZURE_URL), "Azure"),
                                 (gq.ModelChoice("bedrock", aws_region="us-west-2"), "Bedrock")):
            with self.subTest(platform=platform), self.assertRaises(gq.GenerateError) as cm:
                self.build(f"{platform}/set-n", choice, max_cost=1.0)
            self.assertIn(f"{platform} sets its own prices", str(cm.exception))

    def read(self, res, name):
        with open(os.path.join(res["out"], "generation", name), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()] if name.endswith(".jsonl") else json.load(f)

    def test_a_set_built_on_openai_and_on_google_records_them_and_replays(self):
        for provider, model_id, endpoint in (("openai", "gpt-6.2-sol", "https://api.openai.com/v1"),
                                             ("google", "gemini-4-hydrogen",
                                              "https://generativelanguage.googleapis.com/")):
            with self.subTest(provider=provider):
                choice = gq.ModelChoice(provider)
                with mock.patch.dict(sys.modules, self.sdk(provider)):
                    res = self.build(f"{provider}/set-p", choice, max_cost=5.0)
                self.assertTrue(res["check"]["ok"])
                calls = self.read(res, "calls.jsonl")
                self.assertTrue(calls)
                self.assertEqual({(c["provider"], c["endpoint"], c["model"]) for c in calls},
                                 {(provider, endpoint, model_id)})
                self.assertTrue(all(c["reasoning_tokens"] == 1_000 for c in calls))
                self.assertTrue(all("price_usd_per_million" not in c for c in calls))  # the table prices them
                run = self.read(res, "run.json")
                self.assertEqual((run["provider"], run["endpoint"], run["effort"], run["max_tokens"]),
                                 (provider, endpoint, "high", 32000))
                m = res["manifest"]["model"]
                self.assertEqual((m["provider"], m["endpoint"], m["effort"], m["max_tokens"], m["system_cache"]),
                                 (provider, endpoint, "high", 32000, None))
                self.assertEqual(m["reasoning_tokens"], 1_000 * len(calls))
                self.assertEqual(m["prices"]["usd_per_million_tokens"]["input"], gq.PRICES[model_id][0])
                self.assertEqual(m["cost_usd"], round(sum(gq.record_cost(c) for c in calls), 2))
                self.assertGreater(m["cost_usd"], 0)
                # a replay with the same choice rebuilds the set byte for byte, with no SDK at all
                log = os.path.join(res["out"], "generation", "calls.jsonl")
                with mock.patch.dict(sys.modules, {"openai": None, "google": None, "google.genai": None}):
                    again = self.build(f"{provider}/set-r", choice, replay=log)
                self.assertEqual(self.read(again, "calls.jsonl"), calls)
                self.assertEqual(again["manifest"]["model"], m)
                # another provider's requests are not in the log
                other = gq.ModelChoice("anthropic")
                with self.assertRaises(gq.GenerateError) as cm:
                    self.build(f"{provider}/set-x", other, replay=log)
                self.assertIn("holds no response", str(cm.exception))

    def test_an_endpoint_of_your_own_needs_prices_for_the_guard_and_records_them(self):
        choice = gq.ModelChoice("openai", "llama3.3", base_url="http://localhost:11434/v1")
        with self.assertRaises(gq.GenerateError) as cm:
            self.build("local/set-n", choice, max_cost=0.0)
        self.assertIn("--price-in and --price-out (0 and 0", str(cm.exception))
        choice = gq.ModelChoice("openai", "llama3.3", base_url="http://localhost:11434/v1", price_in=0, price_out=0)
        with mock.patch.dict(sys.modules, self.sdk("openai")):
            res = self.build("local/set-l", choice, max_cost=0.0)
        calls = self.read(res, "calls.jsonl")
        self.assertTrue(all(c["price_usd_per_million"] == [0, 0, 0, 0, 0] for c in calls))
        m = res["manifest"]["model"]
        self.assertEqual((m["endpoint"], m["effort"], m["max_tokens"], m["cost_usd"]),
                         ("http://localhost:11434/v1", None, None, 0.0))
        self.assertIn("--price-in and --price-out", m["cost_note"])

    def test_an_anthropic_set_is_recorded_as_before_unless_the_flags_price_it(self):
        plain = self.build("a/set-a", gq.ModelChoice("anthropic"), model=tg.FakeModel(tg.toolbox_writer))
        calls = self.read(plain, "calls.jsonl")
        self.assertFalse(any(k in c for c in calls for k in ("provider", "endpoint", "reasoning_tokens",
                                                             "price_usd_per_million")))
        self.assertFalse(any(k in self.read(plain, "run.json") for k in ("provider", "endpoint")))
        self.assertFalse(any(k in plain["manifest"]["model"] for k in ("provider", "prices", "reasoning_tokens")))
        self.assertEqual(plain["manifest"]["model"]["server_side_fallbacks"], False)
        priced = self.build("a/set-b", gq.ModelChoice("anthropic", "fake-model", price_in=1.0, price_out=2.0),
                            model=tg.FakeModel(tg.toolbox_writer))
        calls = self.read(priced, "calls.jsonl")
        self.assertTrue(all(c["price_usd_per_million"] == [1.0, 2.0, 1.0, 1.0, 1.0] for c in calls))
        self.assertEqual(priced["manifest"]["model"]["prices"]["source"],
                         "--price-in and --price-out (cache reads and writes at the input price)")
        self.assertNotIn("provider", priced["manifest"]["model"])

    def test_the_command_line(self):
        out = os.path.join(self.tmp, "cli", "set-c")
        argv = ["--corpus", self.corpus_path, "--out", out, "--seed", "7", "--targets", TARGETS]
        buf, err = io.StringIO(), io.StringIO()

        def refuse(*a, **k):
            raise AssertionError("the estimate made a client")

        with mock.patch.object(gq, "OpenAIModel", refuse), contextlib.redirect_stdout(buf):
            self.assertEqual(gq.main(argv + ["--provider", "openai", "--estimate"]), 0)
        self.assertIn("Estimate for gpt-6.2-sol on openai, before any call", buf.getvalue())
        self.assertIn("build a small pilot set and pass its folder with --estimate-from", buf.getvalue())
        with contextlib.redirect_stderr(err):
            self.assertEqual(gq.main(argv + ["--provider", "google", "--base-url", "https://x.example.com",
                                             "--model", "m", "--max-cost", "1"]), 2)
        self.assertIn("--base-url goes with --provider openai", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "cli")))
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": fake_openai([], [])}), contextlib.redirect_stdout(buf):
            self.assertEqual(gq.main(["--check-api", "--provider", "openai"]), 0)
        self.assertEqual(buf.getvalue().strip(), "ok: gpt-6.2-sol-2026-11-01 answered")
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, self.sdk("google")), contextlib.redirect_stdout(buf):
            self.assertEqual(gq.main(argv + ["--provider", "google", "--max-cost", "5"]), 0)
        self.assertIn("model (google):", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

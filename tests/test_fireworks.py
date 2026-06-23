"""Offline tests for the resilience layer and the Fireworks client.

No network: the resilience helpers are exercised with synthetic functions and a
fake ``requests`` session, mirroring how hl-read's ``test_info.py`` patches the
SDK. ``time.sleep`` is patched so retries don't actually wait.
"""
from __future__ import annotations

import unittest
from unittest import mock

from token_router.models._http import HttpError, ResilientClient, _is_transient
from token_router.models.base import ModelError
from token_router.models.fireworks import FireworksModel


class TestIsTransient(unittest.TestCase):
    def test_name_based(self):
        class Timeout(Exception):  # class name is in _TRANSIENT_NAMES
            pass

        self.assertTrue(_is_transient(ConnectionError("dropped")))
        self.assertTrue(_is_transient(Timeout("slow")))

    def test_status_based(self):
        self.assertTrue(_is_transient(HttpError("x", 429)))
        self.assertTrue(_is_transient(HttpError("x", 503)))
        self.assertFalse(_is_transient(HttpError("bad request", 400)))
        self.assertFalse(_is_transient(HttpError("unauthorized", 401)))

    def test_message_based(self):
        self.assertTrue(_is_transient(Exception("429 Too Many Requests")))
        self.assertTrue(_is_transient(Exception("rate limit exceeded")))

    def test_non_transient(self):
        self.assertFalse(_is_transient(ValueError("nope")))


class _Flaky:
    """Raises ``exc`` the first ``fails`` times, then returns ``ok``."""

    def __init__(self, fails, exc):
        self.fails = fails
        self.exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return "ok"


class TestResilientCall(unittest.TestCase):
    def test_retries_then_succeeds(self):
        client = ResilientClient(max_retries=3, backoff_base=0.0)
        flaky = _Flaky(2, ConnectionError("blip"))
        with mock.patch("token_router.models._http.time.sleep"):
            self.assertEqual(client._call(flaky), "ok")
        self.assertEqual(flaky.calls, 3)

    def test_non_transient_raises_immediately(self):
        client = ResilientClient(max_retries=3, backoff_base=0.0)

        def boom():
            raise ValueError("real bug")

        with mock.patch("token_router.models._http.time.sleep"):
            with self.assertRaises(ValueError):
                client._call(boom)

    def test_exhausts_to_model_error(self):
        client = ResilientClient(max_retries=2, backoff_base=0.0)
        flaky = _Flaky(99, ConnectionError("down"))
        with mock.patch("token_router.models._http.time.sleep"):
            with self.assertRaises(ModelError):
                client._call(flaky)
        self.assertEqual(flaky.calls, 3)  # initial + 2 retries


class TestCache(unittest.TestCase):
    def test_caches_within_ttl(self):
        client = ResilientClient()
        calls = []

        def producer():
            calls.append(1)
            return len(calls)

        self.assertEqual(client._cached("k", 10, producer), 1)
        self.assertEqual(client._cached("k", 10, producer), 1)
        self.assertEqual(len(calls), 1)

    def test_ttl_zero_always_fetches(self):
        client = ResilientClient()
        calls = []
        client._cached("k", 0, lambda: calls.append(1))
        client._cached("k", 0, lambda: calls.append(1))
        self.assertEqual(len(calls), 2)


class _FakeResp:
    def __init__(self, status, payload, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeSession:
    """Returns/raises a scripted sequence of responses for ``.post``."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


_OK_PAYLOAD = {
    "choices": [{"message": {"content": "Paris"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
}


class TestFireworksModel(unittest.TestCase):
    def test_parses_text_and_usage(self):
        sess = _FakeSession([_FakeResp(200, _OK_PAYLOAD)])
        m = FireworksModel("model-x", api_key="k", session=sess)
        text, usage = m.complete("capital of France?")
        self.assertEqual(text, "Paris")
        self.assertEqual(usage.backend, "fireworks")
        self.assertEqual(usage.prompt_tokens, 5)
        self.assertEqual(usage.completion_tokens, 1)
        self.assertEqual(usage.total_tokens, 6)
        self.assertFalse(usage.estimated)

    def test_retries_transient_then_succeeds(self):
        sess = _FakeSession([ConnectionError("blip"), _FakeResp(200, _OK_PAYLOAD)])
        m = FireworksModel("model-x", api_key="k", session=sess)
        with mock.patch("token_router.models._http.time.sleep"):
            text, _ = m.complete("q?")
        self.assertEqual(text, "Paris")
        self.assertEqual(sess.calls, 2)

    def test_http_400_becomes_model_error(self):
        sess = _FakeSession([_FakeResp(400, {}, text="bad request")])
        m = FireworksModel("model-x", api_key="k", session=sess)
        with mock.patch("token_router.models._http.time.sleep"):
            with self.assertRaises(ModelError):
                m.complete("q?")
        self.assertEqual(sess.calls, 1)  # 400 is non-transient: no retry

    def test_missing_api_key_errors(self):
        sess = _FakeSession([_FakeResp(200, _OK_PAYLOAD)])
        m = FireworksModel("model-x", api_key="", session=sess)
        with self.assertRaises(ModelError):
            m.complete("q?")

    def test_usage_fallback_when_absent(self):
        payload = {"choices": [{"message": {"content": "hello world"}}]}  # no usage block
        sess = _FakeSession([_FakeResp(200, payload)])
        m = FireworksModel("model-x", api_key="k", session=sess)
        _, usage = m.complete("hi")
        self.assertTrue(usage.estimated)
        self.assertGreater(usage.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()

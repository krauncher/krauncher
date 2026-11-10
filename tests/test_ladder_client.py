"""KrauncherClient.ladder() / estimate(): broker /v1/ladder and /v1/assay/estimate, 403 -> KrauncherError."""
import asyncio

import httpx
import pytest

from krauncher import KrauncherClient, KrauncherError


def _client(monkeypatch, status, body):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"] = str(request.url), request.headers.get("X-API-Key")
        return httpx.Response(status, json=body)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler), **kw))
    return KrauncherClient(api_key="k", broker_url="https://broker.test/api"), seen


def test_ok(monkeypatch):
    c, seen = _client(monkeypatch, 200, {"estimate": {}, "ladder": {"rows": [{"gpu_id": "x"}]}})
    assert asyncio.run(c.ladder({"meta": {}}))["ladder"]["rows"] == [{"gpu_id": "x"}]
    assert seen == {"url": "https://broker.test/api/v1/ladder", "key": "k"}


def test_estimate(monkeypatch):
    c, seen = _client(monkeypatch, 200, {"meta": {"schema_version": "estimate/1"}})
    assert asyncio.run(c.estimate({"meta": {}}))["meta"]["schema_version"] == "estimate/1"
    assert seen == {"url": "https://broker.test/api/v1/assay/estimate", "key": "k"}


def test_no_access(monkeypatch):
    c, _ = _client(monkeypatch, 403, {"detail": "Ladder access is not enabled for this account"})
    with pytest.raises(KrauncherError):
        asyncio.run(c.ladder({}))

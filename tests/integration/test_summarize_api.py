"""Read-only summarize API surface: capabilities / candidates / staged.

Core scan + staging are exercised by their own unit tests; here we prove the
routes wire to core, shape the payloads, and (capabilities) reflect the host
honestly. list_candidates / session are monkeypatched for determinism.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.summarize.candidates import Candidate, ScanResult


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    # storage.path's parent drives session.summary_root → point it at tmp so
    # /staged reads an empty temp dir, never the real ~/.tj/summary.
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def client(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_capabilities_manual_always_on_dead_paths_flagged(client, monkeypatch):
    monkeypatch.delenv("TJ_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)   # no `claude` on host
    caps = (await client.get("/api/v1/summarize/capabilities")).json()
    assert caps["manual"]["available"] is True
    assert caps["api"]["available"] is False and caps["api"]["reason"]
    assert caps["claude_p"]["available"] is False and caps["claude_p"]["reason"]


async def test_capabilities_api_enabled_when_key_present(client, monkeypatch):
    monkeypatch.setenv("TJ_ANTHROPIC_API_KEY", "sk-test")
    caps = (await client.get("/api/v1/summarize/capabilities")).json()
    assert caps["api"]["available"] is True and caps["api"]["reason"] == ""


async def test_candidates_returns_scan_dict(client, monkeypatch):
    fake = ScanResult(
        candidates=[Candidate(
            path="./CLAUDE.md", prose_words=1000, total_chars=4000, protected_blocks=1,
            est_tokens_saved=410, pricing_mode="api", scope="repo", is_prompt=True,
        )],
        root=".", recursive=False, globals_checked=0, walk_capped=False, note="",
    )
    monkeypatch.setattr("tokenjam.core.summarize.candidates.list_candidates", lambda **kw: fake)
    body = (await client.get("/api/v1/summarize/candidates")).json()
    assert body["count"] == 1
    c = body["candidates"][0]
    assert c["path"] == "./CLAUDE.md" and c["kind"] == "prompt" and c["est_tokens_saved"] == 410


async def test_staged_empty_by_default(client):
    r = await client.get("/api/v1/summarize/staged")
    assert r.status_code == 200
    assert r.json() == {"staged": []}


async def test_staged_lists_and_reads_one(client, monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.summarize.session.list_staged",
        lambda config: [{"path": "./CLAUDE.md", "est_tokens_saved": 410}],
    )
    monkeypatch.setattr(
        "tokenjam.core.summarize.session.read_staged",
        lambda config, path: {"path": path, "diff": "@@ ... @@"} if path == "./CLAUDE.md" else None,
    )
    listed = (await client.get("/api/v1/summarize/staged")).json()
    assert listed["staged"][0]["path"] == "./CLAUDE.md"
    one = (await client.get("/api/v1/summarize/staged", params={"path": "./CLAUDE.md"})).json()
    assert one["staged"][0]["diff"] == "@@ ... @@"
    miss = (await client.get("/api/v1/summarize/staged", params={"path": "./nope.md"})).json()
    assert miss["staged"] == []


async def test_apply_defaults_dry_run_and_passes_go_through(client, monkeypatch):
    seen: dict = {}

    def fake_apply(config, path=None, *, go=False):
        seen["path"], seen["go"] = path, go
        return {"applied": [path] if (go and path) else [], "skipped": [], "dry_run": not go}

    monkeypatch.setattr("tokenjam.core.summarize.apply.apply_staged", fake_apply)
    dry = (await client.post("/api/v1/summarize/apply", json={"path": "./CLAUDE.md"})).json()
    assert seen == {"path": "./CLAUDE.md", "go": False} and dry["dry_run"] is True
    wrote = (await client.post("/api/v1/summarize/apply", json={"path": "./CLAUDE.md", "go": True})).json()
    assert seen["go"] is True and wrote["applied"] == ["./CLAUDE.md"]


async def test_apply_all_when_path_omitted(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "tokenjam.core.summarize.apply.apply_staged",
        lambda config, path=None, *, go=False: captured.update(path=path) or {"applied": [], "skipped": [], "dry_run": not go},
    )
    await client.post("/api/v1/summarize/apply", json={})
    assert captured["path"] is None   # omitted → apply all staged


async def test_undo_ok_and_drift_returns_409(client, monkeypatch):
    from tokenjam.core.summarize.session import SummarizeRefused

    def fake_undo(config, path, *, go=False):
        if path == "./drifted.md":
            raise SummarizeRefused("file changed since backup")
        return {"path": path, "restored": go, "dry_run": not go}

    monkeypatch.setattr("tokenjam.core.summarize.apply.undo", fake_undo)
    ok = await client.post("/api/v1/summarize/undo", json={"path": "./CLAUDE.md", "go": True})
    assert ok.status_code == 200 and ok.json()["restored"] is True
    bad = await client.post("/api/v1/summarize/undo", json={"path": "./drifted.md", "go": True})
    assert bad.status_code == 409


# ---- run / prep / check (mocked — no real outbound calls) ----

def _verdict(path="./CLAUDE.md", ok=True):
    from tokenjam.core.summarize.session import CheckVerdict
    return CheckVerdict(
        path=path, structure_ok=ok, reason="", integrity={}, words_before=1000,
        words_after=550, est_tokens_saved=410, must_keep_removed=[], must_keep_added=[],
        diff="@@ ... @@", restored="...", staged=ok, produced_by="api", note="",
    )


def _prep(path="./CLAUDE.md", wrapped="<wrapped>", note=""):
    from tokenjam.core.summarize.session import PrepResult
    return PrepResult(
        path=path, source_sha256="abc", wrapped_prompt=wrapped, system_rules="rules",
        prose_words=200, target_prose_words=100, protected_blocks=1, plan=[], note=note,
    )


async def test_run_normalizes_claude_p_and_serializes_verdict(client, monkeypatch):
    from tokenjam.core.summarize.delivery import RunResult
    seen: dict = {}

    def fake_via(config, path, mode, *, ratio=0.5):
        seen["mode"], seen["path"] = mode, path
        return RunResult(verdict=_verdict(path), amortization=None, skipped_note=None, cost_unknown=False)

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", fake_via)
    r = await client.post("/api/v1/summarize/run", json={"path": "./CLAUDE.md", "mode": "claude_p"})
    assert r.status_code == 200
    assert seen["mode"] == "claude-p"                      # underscore normalized to hyphen
    assert r.json()["verdict"]["est_tokens_saved"] == 410


async def test_run_serializes_amortization(client, monkeypatch):
    from tokenjam.core.summarize.delivery import Amortization, RunResult
    am = Amortization(model="claude-x", rewrite_usd=0.01, saving_usd_per_call=0.002,
                      break_even_calls=5, rates_known=True)
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.summarize_via",
        lambda config, path, mode, *, ratio=0.5: RunResult(verdict=_verdict(path), amortization=am),
    )
    body = (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).json()
    assert body["amortization"]["break_even_calls"] == 5 and body["amortization"]["rates_known"] is True


async def test_run_skipped_below_gate(client, monkeypatch):
    from tokenjam.core.summarize.delivery import RunResult
    monkeypatch.setattr(
        "tokenjam.core.summarize.delivery.summarize_via",
        lambda config, path, mode, *, ratio=0.5: RunResult(verdict=None, skipped_note="too short"),
    )
    body = (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).json()
    assert body["verdict"] is None and body["skipped_note"] == "too short"


async def test_run_rejects_manual_and_unknown_mode(client):
    for m in ("manual", "bogus"):
        r = await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": m})
        assert r.status_code == 400


async def test_run_maps_refuse_409_and_delivery_502(client, monkeypatch):
    from tokenjam.core.summarize.delivery import DeliveryError
    from tokenjam.core.summarize.session import SummarizeRefused

    def refuse(config, path, mode, *, ratio=0.5):
        raise SummarizeRefused("changed since prep")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", refuse)
    assert (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).status_code == 409

    def boom(config, path, mode, *, ratio=0.5):
        raise DeliveryError("model down")

    monkeypatch.setattr("tokenjam.core.summarize.delivery.summarize_via", boom)
    assert (await client.post("/api/v1/summarize/run", json={"path": "./x.md", "mode": "api"})).status_code == 502


async def test_prep_returns_wrapped_prompt_and_404_on_missing(client, monkeypatch):
    monkeypatch.setattr("tokenjam.core.summarize.session.prepare",
                        lambda *, path, ratio=0.5: _prep(path))
    ok = (await client.post("/api/v1/summarize/prep", json={"path": "./CLAUDE.md"})).json()
    assert ok["wrapped_prompt"] == "<wrapped>" and ok["source_sha256"] == "abc"

    def missing(*, path, ratio=0.5):
        raise FileNotFoundError(path)

    monkeypatch.setattr("tokenjam.core.summarize.session.prepare", missing)
    assert (await client.post("/api/v1/summarize/prep", json={"path": "./nope.md"})).status_code == 404


async def test_check_stages_and_maps_drift_409(client, monkeypatch):
    monkeypatch.setattr("tokenjam.core.summarize.session.check",
                        lambda config, path, summary, source_hash, **kw: _verdict(path, ok=True))
    ok = (await client.post("/api/v1/summarize/check",
                            json={"path": "./CLAUDE.md", "summary": "s", "source_hash": "abc"})).json()
    assert ok["structure_ok"] is True and ok["staged"] is True

    from tokenjam.core.summarize.session import SummarizeRefused

    def refuse(config, path, summary, source_hash, **kw):
        raise SummarizeRefused("changed")

    monkeypatch.setattr("tokenjam.core.summarize.session.check", refuse)
    assert (await client.post("/api/v1/summarize/check",
                              json={"path": "./x.md", "summary": "s", "source_hash": "abc"})).status_code == 409

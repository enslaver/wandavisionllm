"""pytest setup: keep unit tests out of the live logs in ~/.litellm (Wanda's LOOPS tab reads
loop-breaker.jsonl; every pytest run used to add ~6 fake warn/force/stop entries). Not deployed."""

import pytest

import loop_breaker


@pytest.fixture(autouse=True)
def _tmp_loop_breaker_log(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_breaker, "LOG_PATH", str(tmp_path / "loop-breaker.jsonl"))


@pytest.fixture(autouse=True)
def _no_live_inflight(tmp_path, monkeypatch):
    """The live ~/.litellm/stats-live.json (requests in flight) must not decide unit-test session ids."""
    import ultron_admit
    monkeypatch.setattr(ultron_admit, "STATS_LIVE", str(tmp_path / "stats-live.json"))


@pytest.fixture(autouse=True)
def _hindsight_headroom_off(tmp_path, monkeypatch):
    """The live ~/.ultron/hindsight-mode / headroom-mode must not touch unit-test requests (or the live notes db)."""
    import ultron_headroom
    import ultron_hindsight
    monkeypatch.setattr(ultron_headroom, "MODE_FILE", str(tmp_path / "headroom-mode"))
    monkeypatch.setattr(ultron_headroom, "LOG_PATH", str(tmp_path / "headroom.jsonl"))
    monkeypatch.setattr(ultron_hindsight, "MODE_FILE", str(tmp_path / "hindsight-mode"))
    monkeypatch.setattr(ultron_hindsight, "NOTES_DB", str(tmp_path / "hindsight-notes.sqlite"))
    monkeypatch.setattr(ultron_hindsight, "LOG_PATH", str(tmp_path / "hindsight.jsonl"))


@pytest.fixture(autouse=True)
def _no_live_tier_lock(tmp_path, monkeypatch):
    """A live ~/.ultron/tier-lock (e.g. `fable`) re-maps every opus/sonnet request in the admission tests onto the
    locked tier or to cloud; no file here = no lock. test_tier_lock.py sets its own."""
    import ultron_admit
    monkeypatch.setattr(ultron_admit, "TIER_LOCK_FILE", str(tmp_path / "tier-lock"))
    monkeypatch.setattr(ultron_admit, "TIER_SWAP_FILE", str(tmp_path / "tier-lock-swap"))

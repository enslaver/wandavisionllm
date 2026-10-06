"""pytest setup: keep unit tests out of the live logs in ~/.litellm (Wanda's LOOPS tab reads
loop-breaker.jsonl; every pytest run used to add ~6 fake warn/force/stop entries). Not deployed."""

import pytest

import loop_breaker


@pytest.fixture(autouse=True)
def _tmp_loop_breaker_log(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_breaker, "LOG_PATH", str(tmp_path / "loop-breaker.jsonl"))


@pytest.fixture(autouse=True)
def _bili_off(tmp_path, monkeypatch):
    """A live ~/.ultron/bili-mode must not reroute unit-test requests through bili."""
    import ultron_admit
    monkeypatch.setattr(ultron_admit, "BILI_MODE_FILE", str(tmp_path / "bili-mode"))

"""pytest setup for the media hook: ultron_media imports loop_breaker and ultron_tiers, which live in
../../litellm (all three land flat in ~/.litellm on the Mac), and unit tests must stay out of the live
loop-breaker log. Not deployed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "litellm"))

import loop_breaker  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_loop_breaker_log(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_breaker, "LOG_PATH", str(tmp_path / "loop-breaker.jsonl"))

import ultron_admit as ua

def test_lock_collapses_fable_opus(tmp_path, monkeypatch):
    f = tmp_path / "lock"
    monkeypatch.setattr(ua, "TIER_LOCK_FILE", str(f))
    assert ua.locked_tier("fable") == "fable" and ua.tier_lock() is None
    f.write_text("opus\n")
    assert ua.locked_tier("fable") == "opus" and ua.locked_tier("opus") == "opus"
    assert ua.locked_tier("sonnet") == "sonnet" and ua.locked_tier(None) is None

def test_swap_marker(tmp_path, monkeypatch):
    f = tmp_path / "swap"
    monkeypatch.setattr(ua, "TIER_SWAP_FILE", str(f))
    assert not ua.lock_swapping("opus")
    f.write_text("fable\n")
    assert ua.lock_swapping("opus") and ua.lock_swapping("fable") and not ua.lock_swapping("sonnet")

"""tiers.conf: the cloud-only tier (local = no)."""
import ultron_tiers

CONF = """[routing]
[fable]
local = no
cloud = anthropic/claude-fable-5.1
match = claude-fable-*
advertise = claude-fable-5-1
timeout = 1800
[opus]
cloud = anthropic/claude-opus-5.5
match = claude-opus-*
[haiku]
"""


def test_cloud_only_tier_is_not_local():
    t = ultron_tiers.Tiers(CONF)
    assert not t.problems, t.problems
    assert t.cloud_only() == ["fable"] and t.local_names() == ["opus", "haiku"]
    assert t.default == "opus" and t.helper == "haiku" and t.resident == "opus | haiku"
    for m in ("ultron/fable", "fable", "claude-fable-5-1", "claude-fable-9"):
        assert t.tier_for(m) is None  # nothing local to admit
    assert t.tier_for("claude-opus-5-5") == "opus" and t.tier_for("other") == "opus"


def test_cloud_only_blocks():
    t = ultron_tiers.Tiers(CONF)
    b = ultron_tiers.blocks(t, home="/h")
    sw = b["__TIERS_SWAP_MODELS__"]
    assert "  fable:" not in sw and "  opus:" in sw
    assert "fable" not in b["__TIERS_SWAP_ROUTING__"] and "fable" not in b["__TIERS_SWAP_HOOKS__"]
    lm = b["__TIERS_MODEL_LIST__"]
    fable = lm.split('- model_name: "ultron/fable"')[1].split("- model_name:")[0]
    assert 'model: "openai/anthropic/claude-fable-5.1"' in fable and "OMNIROUTE_BASE" in fable and "hosted_vllm" not in fable
    assert '"claude-fable-5-1"' in lm and '"claude-fable-*"' in lm and '"cloud/fable"' not in lm
    assert '"cloud/opus"' in lm and "ultron/fable" not in b["__TIERS_FALLBACKS__"]
    assert "ultron/fable" not in b["__TIERS_RETRY_POLICY__"]   # its timeout is a local-server setting
    assert 'fable: {model: "ultron/fable"' in b["__TIERS_ALIASES__"]


def test_cloud_only_rules():
    bad = ultron_tiers.Tiers("[routing]\ndefault = f\nhelper = f\nvision = f\nresident = f | big\n"
                             "[f]\nlocal = no\nttl = 5\n[big]\nsubstitute = f\n")
    assert set(bad.problems) == {
        "[f] local = no: needs a cloud = <model id> to answer with",
        "[f] ttl: not allowed with local = no",
        "[routing] default = f: that tier has local = no (cloud only)",
        "[routing] vision = f: that tier has local = no (cloud only)",
        "[routing] helper = f: that tier has local = no (cloud only)",
        "[routing] resident: f has local = no (cloud only)",
        "[big] substitute = f: that tier has local = no",
    }

#!/usr/bin/env python3
"""check_repo.py — static checks CI runs on every push (also: `make check`).

  1. every file deploy.py deploys exists in the repo
  2. deploy.py render fills every placeholder
  3. rendered YAML parses (LiteLLM, llama-swap) and has the expected shape
  4. rendered plists parse
  5. JSON examples parse; rendered bili/config.json is loopback-only and lists the routed tiers
  6. deployed Python imports only the standard library (plus litellm/yaml inside the hooks)

Needs pyyaml. Exits non-zero on the first failing group, after printing every failure in it.
"""

import ast
import glob
import importlib.util
import json
import os
import plistlib
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("FAIL", msg)


def load_deploy():
    spec = importlib.util.spec_from_file_location("deploy", os.path.join(ROOT, "deploy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_components(deploy):
    for name, c in deploy.COMPONENTS.items():
        src = os.path.join(ROOT, c.get("src", name))
        for f in c.get("files", []):
            rel = f[0] if isinstance(f, tuple) else os.path.join(c.get("src", name), f)
            if not os.path.isfile(os.path.join(ROOT, rel)):
                fail(f"{name}: {rel} is listed in COMPONENTS but missing")
        if c.get("tree") and not os.path.isdir(src):
            fail(f"{name}: tree {src} missing")


def render(out):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "deploy.py"), "render", out],
                       capture_output=True, text=True, env={**os.environ, "WANDAVISION_HOST": ""})
    if r.returncode:
        fail(f"deploy.py render failed: {r.stderr.strip()}")
    for d, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(d, f)
            try:
                text = open(p, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            for token in ("__HOME__", "__HOSTNAME__"):
                if token in text:
                    fail(f"{os.path.relpath(p, out)}: {token} left after render")


def check_yaml(out, deploy):
    import yaml
    tiers = deploy.TIERS
    cfg = yaml.safe_load(open(os.path.join(out, "litellm", "config.yaml")))
    names = [m.get("model_name") for m in cfg.get("model_list") or []]
    for need in [f"ultron/{n}" for n in tiers.names] + [f"cloud/{n}" for n in tiers.names if tiers.tier[n]["cloud"]]:
        if need not in names:
            fail(f"litellm/config.yaml: model_name {need} missing")
    callbacks = (cfg.get("litellm_settings") or {}).get("callbacks") or []
    order = [c.split(".")[0] for c in callbacks if "." in c]
    if order[:5] != ["ultron_stats", "loop_breaker", "ultron_media", "ultron_admit", "ultron_rescue"]:
        fail(f"litellm/config.yaml: hook order is {order}, expected stats, loop_breaker, media, admit, rescue")
    if (cfg.get("general_settings") or {}).get("master_key") != "os.environ/LITELLM_MASTER_KEY":
        fail("litellm/config.yaml: master_key must come from os.environ/LITELLM_MASTER_KEY")
    swap = yaml.safe_load(open(os.path.join(out, "llama-swap", "config.yaml")))
    swap_tiers = list((swap.get("models") or {}).keys())
    if swap_tiers != tiers.names:
        fail(f"llama-swap/config.yaml: tiers are {swap_tiers}, tiers.conf has {tiers.names}")
    for t, m in (swap.get("models") or {}).items():
        if not str(m.get("cmd", "")).startswith("/"):
            fail(f"llama-swap/config.yaml: {t}.cmd must be an absolute path after render")


def check_plists(out):
    for d, _, fs in os.walk(out):
        for f in fs:
            if f.endswith(".plist"):
                try:
                    with open(os.path.join(d, f), "rb") as fh:
                        pl = plistlib.load(fh)
                    if not pl.get("Label") or not pl.get("ProgramArguments"):
                        fail(f"{f}: Label/ProgramArguments missing")
                except Exception as e:  # noqa: BLE001 — report any parse error
                    fail(f"{f}: {e}")


def check_json():
    comfy = os.path.join(ROOT, "Vision", "comfyui")
    workflows = sorted(glob.glob(os.path.join(comfy, "workflows", "*.json")))
    for p in ["wanda/services.example.json", "Vision/comfyui/nodes.json", "Vision/comfyui/models.json",
              "lora/recipe-4b-vision.json"] + [os.path.relpath(w, ROOT) for w in workflows]:
        try:
            json.load(open(os.path.join(ROOT, p)))
        except Exception as e:  # noqa: BLE001
            fail(f"{p}: {e}")


def check_bili(out, deploy):
    """bili sits behind LiteLLM with no auth of its own: loopback only, never self-updating, and its
    providers and timeout must follow tiers.conf (the routing is unit-tested in test_ultron_admit.py)."""
    tiers = deploy.TIERS
    try:
        cfg = json.load(open(os.path.join(out, "bili", "config.json")))
    except Exception as e:  # noqa: BLE001
        fail(f"bili/config.json (rendered): {e}")
        return
    if cfg.get("host") != "127.0.0.1":
        fail(f"bili/config.json: host is {cfg.get('host')!r}; bili's /bili/ proxy has no auth, keep it on 127.0.0.1")
    for k in ("autoUpdate", "advisoryCheck", "releaseNotesCheck"):
        if cfg.get(k) is not False:
            fail(f"bili/config.json: {k} must be false")
    local = ((cfg.get("providers") or {}).get("http://127.0.0.1:8001") or {}).get("models") or {}
    if list(local) != tiers.routed():
        fail(f"bili/config.json: llama-swap models are {list(local)}, routed tiers are {tiers.routed()}")
    longest = max([tiers.tier[n]["timeout"] for n in tiers.routed()] + [0])
    if ((cfg.get("network") or {}).get("upstreamTimeoutMs") or 0) < longest * 1000:
        fail(f"bili/config.json: network.upstreamTimeoutMs is below the longest tier timeout ({longest} s)")
    label = "com.billion-context.bili"
    with open(os.path.join(out, "launchd", label + ".plist"), "rb") as fh:
        pl = plistlib.load(fh)
    if pl.get("Label") != label or not str(pl.get("ProgramArguments", [""])[-1]).endswith("/.bili/start.sh"):
        fail(f"{label}.plist: Label or ProgramArguments don't match bili/start.sh")


DEPLOYED_PY = {
    "deploy.py": {"ultron_tiers"},
    "wanda/server.py": set(),
    "litellm/loop_breaker.py": {"litellm"},
    "litellm/ultron_admit.py": {"litellm", "loop_breaker", "yaml", "ultron_tiers"},
    "Vision/media/ultron_media.py": {"litellm", "loop_breaker", "ultron_tiers"},
    "litellm/ultron_stats.py": {"litellm", "yaml"},
    "litellm/ultron_rescue.py": {"litellm"},
    "litellm/ultron_tiers.py": set(),
    "gpu-box/deploy.py": set(),        # deployed on the GPU box
    "Vision/judge/rank.py": set(),     # a client; runs anywhere
}


def check_stdlib_only():
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        print("skip stdlib-only check (needs Python 3.10+ for sys.stdlib_module_names)")
        return
    for rel, allowed in DEPLOYED_PY.items():
        tree = ast.parse(open(os.path.join(ROOT, rel)).read(), rel)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top not in stdlib and top not in allowed and top != "__future__":
                    fail(f"{rel}: imports {m} (deployed files are standard library only)")


def check_personal():
    """Catch obvious leftovers from a personal deployment: absolute home paths in tracked files."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    pat = re.compile(r"/Users/(?!you\b|x\b|<)[A-Za-z0-9._-]+/")
    for rel in files:
        if rel.startswith("scripts/"):
            continue
        try:
            text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for m in pat.finditer(text):
            fail(f"{rel}: hard-coded home path {m.group(0)} (use __HOME__ or $HOME)")


def main():
    deploy = load_deploy()
    groups = [("components", lambda: check_components(deploy))]
    with tempfile.TemporaryDirectory() as out:
        groups += [("render", lambda: render(out)), ("yaml", lambda: check_yaml(out, deploy)),
                   ("plists", lambda: check_plists(out)), ("json", check_json), ("bili", lambda: check_bili(out, deploy)),
                   ("stdlib-only", check_stdlib_only), ("home paths", check_personal)]
        for name, fn in groups:
            before = len(FAILS)
            fn()
            print(("ok  " if len(FAILS) == before else "BAD ") + name)
            if len(FAILS) > before and name in ("components", "render"):
                break
    if FAILS:
        print(f"{len(FAILS)} problem(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

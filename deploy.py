#!/usr/bin/env python3
"""deploy.py — the one way to change the stack. This repo is the source of truth.

    ./deploy.py status [component ...]                 what differs between the repo and this Mac
    ./deploy.py push   [component ...] [--dry-run] [--force] [--no-wait]
    ./deploy.py pull   [component ...]                 copy files edited in place back into the repo
    ./deploy.py render DIR [component ...]             write the filled-in files to DIR (no install)

Runs on the Mac that serves the models. To deploy from another machine, set WANDAVISION_HOST (an
ssh host) and WANDAVISION_REMOTE_ROOT (this repo's path there, e.g. a shared mount or a clone);
deploy.py then reruns itself on that host. Each folder mirrors a place on the Mac (COMPONENTS
below). Files are copied into place, not symlinked, so the services still start without the repo.

Placeholders: deployed text files may contain __HOME__ (your home directory), __HOSTNAME__
(WANDAVISION_HOSTNAME, the name Caddy serves HTTPS on), __BREW__ (Homebrew's prefix: /opt/homebrew
on Apple Silicon, /usr/local on Intel) and __LITELLM_PORT__ (LITELLM_PORT from ~/.litellm/env,
else 4000). Settings come from the environment, then from wandavision.conf next to this script
(KEY=VALUE lines; copy wandavision.conf.example).

Tiers: litellm/tiers.conf lists the local model tiers. A line `# __TIERS_<PART>__` in a config
file is replaced by that part generated from it (ultron_tiers.blocks()); the tier scripts to
deploy are the ones it names. A tiers.conf with problems is never rendered or pushed.

push copies only files whose content differs, then runs that component's reload step. LiteLLM and
llama-swap changes first wait until LiteLLM has no request in flight, so nobody's request is cut off
(--no-wait restarts anyway: for a fix a busy agent needs now; Claude Code retries cut requests).

Guard: every push records checksums in ~/.wandavision/deployed.json. A file edited in place since
the last push (or never pushed and different) is not overwritten: push lists it and stops. `pull`
brings those edits into the repo (review with git diff); --force discards them.

Secrets never live here. They stay on the Mac (~/.litellm/env, ~/.wanda/token) and the configs
reference them (os.environ/...). Runtime state (logs, pins.sqlite, ~/.ultron/*-mode) isn't managed.
Python 3.9 (macOS's /usr/bin/python3), standard library only.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "litellm"))
import ultron_tiers  # noqa: E402  (litellm/ultron_tiers.py, standard library only)

HOME = os.path.expanduser("~")
MANIFEST = os.path.join(HOME, ".wandavision", "deployed.json")
STATS_LIVE = os.path.join(HOME, ".litellm", "stats-live.json")
IDLE_WAIT_S = 300


def load_conf(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


CONF = load_conf(os.path.join(ROOT, "wandavision.conf"))


def setting(key, default=""):
    return os.environ.get(key) or CONF.get(key) or default


def env_file(key, path=None):
    if path is None:
        path = os.path.join(HOME, ".litellm", ".env") if False else os.path.join(HOME, ".litellm", "env")
    """A value from ~/.litellm/env (where LiteLLM's own settings live), or ''."""
    return load_conf(path).get(key, "")


def brew_prefix():
    try:
        out = subprocess.run(["brew", "--prefix"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out or os.environ.get("HOMEBREW_PREFIX") or ("/usr/local" if os.path.isdir("/usr/local/Homebrew") else "/opt/homebrew")


HOST = setting("WANDAVISION_HOST")                      # empty: deploy on this machine
REMOTE_ROOT = setting("WANDAVISION_REMOTE_ROOT", ROOT)  # the repo's path on HOST
BREW = brew_prefix()
PLACEHOLDERS = {
    "__HOME__": HOME,
    "__HOSTNAME__": setting("WANDAVISION_HOSTNAME", "localhost"),
    "__BREW__": BREW,
    "__LITELLM_PORT__": env_file("LITELLM_PORT") or "4000",
}
TIERS_CONF = os.path.join(ROOT, "litellm", "tiers.conf")
try:
    with open(TIERS_CONF) as _f:
        TIERS = ultron_tiers.Tiers(_f.read())
except Exception as e:  # noqa: BLE001 — unreadable or not INI: nothing below can work
    sys.exit(f"litellm/tiers.conf: {e}")
BLOCK_RE = re.compile(r"^([ \t]*)# (__TIERS_[A-Z_]+__)[ \t]*$", re.M)

# component -> where it lives on the Mac and how it's reloaded. Listed in first-install order.
#   files: listed files in the folder; tree: the whole folder (files removed here are removed there)
#   idle: wait for LiteLLM to have no request in flight before touching it
#   validate: run on the new file ({tmp}) before it replaces the live one
#   after: shell run once the component's files are in place ("litellm"/"launchd": built-in steps)
COMPONENTS = {
    "mtplx": {
        "src": "mtplx/bin", "dst": "~/.mtplx/bin", "files": sorted({t["script"] for t in TIERS.tier.values()}),
        "note": "tier scripts apply the next time llama-swap loads that tier",
    },
    "llama-swap": {
        "dst": "~/.llama-swap", "files": ["config.yaml"], "idle": True,
        "note": "llama-swap reloads its config itself (-watch-config)",
    },
    "litellm": {
        "dst": "~/.litellm", "files": ["config.yaml", "tiers.conf", "agents.conf", "start.sh", "loop_breaker.py", "ultron_tiers.py",
                                       "ultron_admit.py", "ultron_stats.py", "ultron_media.py", "ultron_rescue.py"],
        "idle": True, "after": "litellm",
    },
    "launchd": {
        "dst": "~/Library/LaunchAgents", "files": ["com.litellm.proxy.plist", "com.llama-swap.plist"],
        "validate": "plutil -lint {tmp} >/dev/null", "idle": True, "after": "launchd",
    },
    "wanda": {
        "dst": "~/wanda", "tree": True, "exclude": ["README.md", "services.example.json"],
        "after": "bash ~/wanda/install.sh",
    },
    "caddy": {
        "dst": BREW + "/etc", "files": ["Caddyfile"],
        "validate": BREW + "/opt/caddy/bin/caddy validate --config {tmp} --adapter caddyfile >/dev/null",
        "after": "caddy",
    },
    "bin": {"dst": "~/bin", "files": ["backup-stack.sh"]},
}
SKIP_DIRS = {"__pycache__", ".pytest_cache"}

LITELLM_AFTER = r'''
set -e
export PATH=__BREW__/bin:$HOME/.local/bin:$PATH
mkdir -p ~/.ultron ~/.ultron/media
[ -f ~/.ultron/route-mode ] || echo auto > ~/.ultron/route-mode
touch ~/.litellm/loop-breaker.jsonl ~/.litellm/ultron-admit.jsonl ~/.litellm/ultron-stats.jsonl ~/.litellm/media.jsonl
[ -f ~/.litellm/env ] || echo "litellm: ~/.litellm/env is missing; copy litellm/env.example there and set LITELLM_MASTER_KEY"
PY=~/.local/share/uv/tools/litellm/bin/python
if [ ! -x "$PY" ]; then
  echo "litellm: not installed as a uv tool; run: uv tool install 'litellm[proxy]' --with prometheus_client"
  exit 0
fi
if ! $PY -c "import prometheus_client" 2>/dev/null; then  # the stock prometheus callback needs it
  V=$($PY -c "import importlib.metadata as m;print(m.version('litellm'))")
  uv tool install --quiet --force "litellm[proxy]==$V" --with prometheus_client
  echo "litellm: added prometheus_client (litellm $V pinned)"
fi
# after the block above (its --force reinstall drops this). Without prisma a missing key is a 500, not a
# 401: the auth error path imports it even with no DB. uv pip adds only prisma + nodeenv; a fresh
# `uv tool install --with prisma` would re-resolve and upgrade fastapi, uvloop, ...
if ! $PY -c "import prisma" 2>/dev/null; then
  uv pip install --quiet --python $PY "prisma>=0.11.0,<1.0"
  echo "litellm: added prisma"
fi
if ! launchctl print "gui/$(id -u)/com.litellm.proxy" >/dev/null 2>&1; then
  echo "litellm: LaunchAgent not loaded yet; ./deploy.py push launchd starts it"
  exit 0
fi
launchctl kickstart -k "gui/$(id -u)/com.litellm.proxy"
P=$(sed -n 's/^LITELLM_PORT=//p' ~/.litellm/env 2>/dev/null | tr -d '"'); P=${P:-4000}
for i in $(seq 1 60); do curl -fsS -m 2 127.0.0.1:$P/health/liveliness >/dev/null 2>&1 && break; sleep 2; done
curl -fsS -m 2 127.0.0.1:$P/health/liveliness >/dev/null && echo "litellm: up" || { echo "litellm: NOT answering"; tail -30 ~/.litellm/litellm.log; exit 1; }
'''.replace("__BREW__", BREW)

CADDY_AFTER = r'''
export PATH=__BREW__/bin:$PATH
CF=__BREW__/etc/Caddyfile
if curl -fsS -m 2 localhost:2019/config/ >/dev/null 2>&1; then
  caddy reload --config "$CF" --adapter caddyfile && echo "caddy: reloaded"
else
  brew services start caddy && echo "caddy: started"
fi
'''.replace("__BREW__", BREW)


def live(p):
    return os.path.join(HOME, p[2:]) if p.startswith("~/") else p


def tier_blocks():
    if TIERS.problems:
        sys.exit("litellm/tiers.conf has problems; fix them first:\n  " + "\n  ".join(TIERS.problems))
    return ultron_tiers.blocks(TIERS)


def _block(indent, name):
    """The generated text for one `# __TIERS_<PART>__` line, indented like that line."""
    blocks = tier_blocks()
    if name not in blocks:
        sys.exit(f"unknown tiers block {name} (known: {', '.join(blocks)})")
    return "\n".join(indent + line if line else line for line in blocks[name].splitlines())


def render(data):
    """Fill placeholders and tier blocks in a text file; binary files pass through unchanged."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if "__TIERS_" in text:
        text = BLOCK_RE.sub(lambda m: _block(m.group(1), m.group(2)), text)
    for token, value in PLACEHOLDERS.items():
        text = text.replace(token, value)
    return text.encode("utf-8")


def unrender(data, template):
    """Put back the placeholders and tier-block markers the repo version uses, so a pull doesn't
    hard-code this Mac's values or copy generated tier entries into the template."""
    try:
        text, tmpl = data.decode("utf-8"), template.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for m in BLOCK_RE.finditer(tmpl):
        filled = _block(m.group(1), m.group(2))
        for token, value in PLACEHOLDERS.items():
            filled = filled.replace(token, value)
        if filled:
            text = text.replace(filled, m.group(0), 1)
    for token, value in PLACEHOLDERS.items():
        if token in tmpl and value:
            text = re.sub(r"(?<![\w.-])" + re.escape(value) + r"(?![\w.-])", token, text)
    return text.encode("utf-8")


def read(p):
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return None


def md5(data):
    return None if data is None else hashlib.md5(data).hexdigest()


def repo_md5(p):
    data = read(p)
    return None if data is None else md5(render(data))


def sh(cmd):
    r = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def mapping(names):
    """[(component, repo path, live path)] for the chosen components, plus live-only files in trees."""
    out = []
    for name in names:
        c = COMPONENTS[name]
        src = os.path.join(ROOT, c.get("src", name))
        if not c.get("tree"):
            out += [(name, os.path.join(src, f), live(c["dst"] + "/" + f)) for f in c["files"]]
            continue
        rels = set()
        for base in (src, live(c["dst"])):
            for d, dirs, fs in os.walk(base):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                for f in fs:
                    rel = os.path.relpath(os.path.join(d, f), base)
                    if f == ".DS_Store" or f.endswith((".pyc", ".deploy-tmp")) or rel in c.get("exclude", ()):
                        continue
                    rels.add(rel)
        out += [(name, os.path.join(src, r), live(c["dst"] + "/" + r)) for r in sorted(rels)]
    return out


def load_manifest():
    try:
        with open(MANIFEST) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_manifest(m):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST + ".tmp", "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(MANIFEST + ".tmp", MANIFEST)


def state(L, R, M):
    """L repo (filled in), R live, M last pushed (md5 or None)."""
    if L == R:
        return "same"
    if R == M or (R is None and M is None):
        return "delete" if L is None else "push"
    if M is None:
        return "untracked-in-place"   # differs and was never pushed from here
    return "edited-in-place" if R is not None else "deleted-in-place"


BLOCKED = ("untracked-in-place", "edited-in-place", "deleted-in-place")


def classify(names):
    man = load_manifest()
    return [(n, loc, dst, state(repo_md5(loc), md5(read(dst)), man.get(dst))) for n, loc, dst in mapping(names)], man


def inflight():
    try:
        with open(STATS_LIVE) as f:
            return len(json.load(f).get("inflight") or [])
    except (OSError, ValueError):
        return 0


def short(p):
    return "~" + p[len(HOME):] if p.startswith(HOME) else p


def cmd_status(names):
    rows, _ = classify(names)
    for name in names:
        mine = [r for r in rows if r[0] == name]
        odd = [r for r in mine if r[3] != "same"]
        print(f"{name:11} {COMPONENTS[name]['dst']:28} " + ("in sync" if not odd else f"{len(odd)} of {len(mine)} differ"))
        for _, _, dst, st in odd:
            print(f"    {st:20} {short(dst)}")
    n = inflight()
    if n:
        print(f"(LiteLLM has {n} request(s) in flight)")
    return 0


def cmd_pull(names):
    """Adopt the live version of every differing file: copy it into the repo and record it as pushed."""
    rows, man = classify(names)
    n = 0
    for _, loc, dst, st in rows:
        if st == "same":
            continue
        data = read(dst)
        if data is None:
            print(f"  skip   {short(dst)} (not installed)")
            continue
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        tmpl = read(loc)
        with open(loc, "wb") as f:
            f.write(unrender(data, tmpl) if tmpl is not None else data)
        man[dst] = md5(data)
        print(f"  pulled {short(dst)} -> {os.path.relpath(loc, ROOT)}")
        n += 1
    save_manifest(man)
    print(f"{n} file(s) pulled; review with git diff, then push." if n else "nothing to pull")
    return 0


def cmd_render(out_dir, names):
    """Write every component's filled-in files under out_dir/<component>/ (for review or CI checks)."""
    for name, loc, dst, _ in classify(names)[0]:
        data = read(loc)
        if data is None:
            continue
        rel = os.path.relpath(loc, os.path.join(ROOT, COMPONENTS[name].get("src", name)))
        p = os.path.join(out_dir, name, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(render(data))
    print(f"rendered {', '.join(names)} into {out_dir}")
    return 0


def install(loc, dst, validate):
    """Write the filled-in repo file over dst via a temp file; keeps dst's mode if it exists."""
    mode = os.stat(dst).st_mode & 0o777 if os.path.exists(dst) else (0o755 if dst.endswith(".sh") else 0o644)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".deploy-tmp"
    with open(tmp, "wb") as f:
        f.write(render(read(loc)))
    os.chmod(tmp, mode)
    if validate:
        rc, out = sh(validate.replace("{tmp}", "'" + tmp + "'"))
        if rc:
            os.remove(tmp)
            return out or f"validation failed ({rc})"
    os.replace(tmp, dst)
    return None


def cmd_push(names, dry, force, no_wait=False):
    rows, man = classify(names)
    blocked = [r for r in rows if r[3] in BLOCKED]
    if blocked and not force:
        print("Not pushing: these were changed in place since the last push from this repo:", file=sys.stderr)
        for _, _, dst, st in blocked:
            print(f"    {st:20} {short(dst)}", file=sys.stderr)
        print("Run ./deploy.py pull " + " ".join(sorted({b[0] for b in blocked})) +
              " to bring them into the repo, or push --force to overwrite them.", file=sys.stderr)
        return 1
    todo = [r for r in rows if r[3] != "same"]
    for name in names:
        mine = [r for r in todo if r[0] == name]
        if mine:
            print(f"{name}: " + ", ".join(f"{'delete' if not os.path.exists(loc) else 'write'} {short(dst)}"
                                          for _, loc, dst, _ in mine))
    if not todo:
        print("already matches the repo")
    if dry:
        return 0
    for _, loc, dst, st in rows:  # files that already match count as pushed
        if st == "same" and os.path.exists(loc):
            man[dst] = repo_md5(loc)
    save_manifest(man)
    for name in names:
        mine = [r for r in todo if r[0] == name]
        if not mine:
            continue
        c = COMPONENTS[name]
        if c.get("idle") and no_wait and inflight():
            print(f"{name}: --no-wait: cutting {inflight()} request(s) in flight (clients retry)")
        elif c.get("idle"):
            deadline = time.time() + IDLE_WAIT_S
            while inflight() and time.time() < deadline:
                time.sleep(2)
            if inflight():
                print(f"{name}: LiteLLM still has {inflight()} request(s) in flight after {IDLE_WAIT_S}s; "
                      f"not touching it. Rerun when it's idle.", file=sys.stderr)
                return 1
        for _, loc, dst, _ in mine:
            if not os.path.exists(loc):
                os.remove(dst)
                man.pop(dst, None)
                print(f"  deleted {short(dst)}")
                continue
            err = install(loc, dst, c.get("validate"))
            if err:
                save_manifest(man)
                print(f"{name}: {short(dst)} not replaced: {err}", file=sys.stderr)
                return 1
            man[dst] = repo_md5(loc)
            print(f"  wrote {short(dst)}")
        save_manifest(man)
        after = c.get("after")
        if after == "litellm":
            rc, out = sh(LITELLM_AFTER)
        elif after == "caddy":
            rc, out = sh(CADDY_AFTER)
        elif after == "launchd":
            rc, out = 0, ""
            for _, _, dst, _ in mine:
                label = os.path.basename(dst)[:-len(".plist")]
                r, o = sh(f'U=$(id -u); launchctl bootout gui/$U/{label} 2>/dev/null; '
                          f'for i in $(seq 1 20); do launchctl print gui/$U/{label} >/dev/null 2>&1 || break; sleep 0.25; done; '
                          f"launchctl bootstrap gui/$U '{dst}' && echo 'reloaded {label}'")
                rc, out = rc or r, (out + "\n" + o).strip()
        elif after:
            rc, out = sh(after)
        else:
            rc, out = 0, ""
        if out:
            print("  " + out.replace("\n", "\n  "))
        if rc:
            print(f"{name}: reload step failed ({rc})", file=sys.stderr)
            return 1
        if c.get("note"):
            print(f"  note: {c['note']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["status", "push", "pull", "render"])
    ap.add_argument("components", nargs="*", help=", ".join(COMPONENTS) + " (default: all); render takes DIR first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="push over files edited in place")
    ap.add_argument("--no-wait", action="store_true", help="restart LiteLLM/llama-swap even with requests in flight")
    a = ap.parse_args()
    if a.action == "render":
        if not a.components:
            ap.error("render needs an output directory")
        out_dir, a.components = a.components[0], a.components[1:]
    if HOST and a.action != "render" and not os.environ.get("WANDAVISION_ON_TARGET"):
        # run it where the files go; the repo must exist there at REMOTE_ROOT
        rel = os.path.relpath(os.path.abspath(__file__), ROOT)
        cmd = (f"cd {shlex.quote(REMOTE_ROOT)} && WANDAVISION_ON_TARGET=1 /usr/bin/python3 {shlex.quote(rel)} "
               + " ".join(shlex.quote(x) for x in sys.argv[1:]))
        return subprocess.call(["ssh", HOST, cmd])
    bad = [c for c in a.components if c not in COMPONENTS]
    if bad:
        ap.error(f"unknown component(s): {', '.join(bad)}")
    names = a.components or list(COMPONENTS)
    if a.action == "render":
        return cmd_render(out_dir, names)
    if a.action == "status":
        return cmd_status(names)
    if a.action == "pull":
        return cmd_pull(names)
    return cmd_push(names, a.dry_run, a.force, a.no_wait)


if __name__ == "__main__":
    sys.exit(main())

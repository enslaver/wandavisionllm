#!/usr/bin/env python3
"""gpu-box/deploy.py — install and refresh everything the GPU box needs. This repo is the source of truth.

    python gpu-box\\deploy.py status [component ...]
    python gpu-box\\deploy.py push   [component ...] [--dry-run] [--all] [--group image,video] [--restart]

Runs on the GPU box (Windows 11 with an NVIDIA GPU), from a clone of this repo. Stdlib only, Python 3.9+.
It's the GPU box's twin of ../deploy.py: files are copied into place, each component is checked
(`status`) and then brought in line (`push`). The services run from the copies, not from the clone.

Components, in install order:

  prereq   check only: NVIDIA driver, git, python (Tailscale reported, optional)
  caddy    ~\\caddy: Caddyfile, start_caddy.bat, caddy.exe (downloaded when missing); validated, then reloaded
  comfyui  ~\\ComfyUI\\ComfyUI-Easy-Install: run_service.bat, comfy_idle_unload.ps1, extra_model_paths.yaml
  nodes    ComfyUI custom_nodes cloned and pinned to the commits in Vision/comfyui/nodes.json
  tasks    scheduled tasks: Caddy-GPU-Box, ComfyUI-GPU-Box, ComfyUI Idle Unload
  models   the model files in Vision/comfyui/models.json (76 GB for both groups)
  unsloth  Unsloth Studio (the venv for training image LoRAs) and ~\\lora

A bare `push` runs prereq, caddy, comfyui, nodes and tasks. `models` and `unsloth` move gigabytes, so name
them: `push models --group image`. `--all` also installs the `extra` custom nodes. `--restart` restarts
ComfyUI once its queue is empty (needed after run_service.bat or extra_model_paths.yaml changes).

Placeholders: __HOME__ (your profile folder) and __HOSTNAME__ (GPU_BOX_HOSTNAME, the name Caddy serves
HTTPS on; default localhost) are filled in when a file is copied. GPU_BOX_HOSTNAME comes from the
environment, then from wandavision.conf at the repo root (KEY=VALUE lines, the file ../deploy.py reads).

ComfyUI itself comes from the ComfyUI-Easy-Install package (interactive; see gpu-box/README.md): push
stops with instructions when that tree is missing, and manages everything after it.
Secrets stay out of the repo: gated Hugging Face downloads read HF_TOKEN or ~\\.cache\\huggingface\\token.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
VISION = ROOT / "Vision" / "comfyui"
CADDY_DIR = HOME / "caddy"
CEI = HOME / "ComfyUI" / "ComfyUI-Easy-Install"
COMFY = CEI / "ComfyUI"
PY = CEI / "python_embeded" / "python.exe"
UNSLOTH_PY = HOME / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"
COMFY_URL = "http://127.0.0.1:8188"
ORDER = ["prereq", "caddy", "comfyui", "nodes", "tasks", "models", "unsloth"]
DEFAULT = ["prereq", "caddy", "comfyui", "nodes", "tasks"]
GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")


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


def setting(key, default=""):
    return os.environ.get(key) or load_conf(ROOT / "wandavision.conf").get(key) or default


PLACEHOLDERS = {
    "__HOME__": str(HOME),
    "__HOSTNAME__": setting("GPU_BOX_HOSTNAME", "localhost"),
}

# (component, repo file, live file). Placeholders inside a file are filled when it is copied.
FILES = [
    ("caddy", ROOT / "gpu-box/caddy/Caddyfile", CADDY_DIR / "Caddyfile"),
    ("caddy", ROOT / "gpu-box/caddy/start_caddy.bat", CADDY_DIR / "start_caddy.bat"),
    ("comfyui", VISION / "run_service.bat", CEI / "run_service.bat"),
    ("comfyui", VISION / "comfy_idle_unload.ps1", CEI / "comfy_idle_unload.ps1"),
    ("comfyui", VISION / "extra_model_paths.yaml", COMFY / "extra_model_paths.yaml"),
]

TASKS = [  # name, trigger, logon type, restart interval, description, command, arguments
    ("Caddy-GPU-Box", "boot", "S4U", "PT1M", "", "cmd.exe", f'/c "{CADDY_DIR / "start_caddy.bat"}"'),
    ("ComfyUI-GPU-Box", "boot", "S4U", "PT2M", "", "cmd.exe", f'/c "{CEI / "run_service.bat"}"'),
    ("ComfyUI Idle Unload", "5min", "InteractiveToken", None, "Unloads ComfyUI models after 15 min idle", "conhost.exe",
     f'--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{CEI / "comfy_idle_unload.ps1"}"'),
]

# ---------------------------------------------------------------- helpers


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)
    return r.returncode, (r.stdout + r.stderr).strip()


def norm(b):
    return b.replace(b"\r\n", b"\n")


def rendered(src):
    data = norm(src.read_bytes())
    for k, v in PLACEHOLDERS.items():
        data = data.replace(k.encode(), v.encode())
    return data


def file_state(src, dst):
    if not dst.exists():
        return "missing"
    return "same" if norm(dst.read_bytes()) == rendered(src) else "differs"


def install_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = rendered(src)
    if src.suffix in (".bat", ".ps1"):
        data = data.replace(b"\n", b"\r\n")  # cmd needs CRLF
    tmp = dst.with_name(dst.name + ".deploy-tmp")
    tmp.write_bytes(data)
    return tmp


def say(component, text):
    print(f"{component:8} {text}")


def gb(n):
    return f"{n / 2**30:.1f} GB"


def load(name):
    return json.loads((VISION / name).read_text(encoding="utf-8"))


def short(p):
    s = str(p)
    return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


# ---------------------------------------------------------------- prereq


def prereq(push, a):
    ok = True
    rc, out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else (1, "")
    say("prereq", f"GPU      {out if rc == 0 else 'nvidia-smi not found: install the NVIDIA driver'}")
    ok &= rc == 0
    git = shutil.which("git")
    say("prereq", f"git      {git or 'MISSING: winget install Git.Git'}")
    ok &= bool(git)
    ts = shutil.which("tailscale")
    say("prereq", f"tailscale {ts or 'not found (optional: only for a *.ts.net name)'}")
    say("prereq", f"repo     {ROOT}")
    host = PLACEHOLDERS["__HOSTNAME__"]
    say("prereq", f"hostname {host}" + (" (set GPU_BOX_HOSTNAME for HTTPS on a real name)" if host == "localhost" else ""))
    if host.endswith(".ts.net") and not ts:
        say("prereq", "         MISSING: a *.ts.net name needs Tailscale on this box (Caddy gets its certificate from it)")
        ok = False
    say("prereq", f"python   {sys.version.split()[0]}")
    return ok


# ---------------------------------------------------------------- caddy


def caddy_exe():
    return CADDY_DIR / "caddy.exe"


def caddy(push, a):
    rows = [(s, d, file_state(s, d)) for c, s, d in FILES if c == "caddy"]
    exe = caddy_exe().exists()
    for s, d, st in rows:
        if st != "same":
            say("caddy", f"{st:8} {short(d)}")
    if not exe:
        say("caddy", f"missing  {short(caddy_exe())} (downloaded from caddyserver.com on push)")
    if not push:
        if all(st == "same" for _, _, st in rows) and exe:
            say("caddy", "in sync")
        return True
    todo = [r for r in rows if r[2] != "same"]
    if not todo and exe:
        say("caddy", "in sync")
        return True
    if a.dry_run:
        return True
    CADDY_DIR.mkdir(parents=True, exist_ok=True)
    if not exe:
        say("caddy", "downloading caddy.exe")
        urllib.request.urlretrieve("https://caddyserver.com/api/download?os=windows&arch=amd64", caddy_exe())
    for s, d, st in todo:
        tmp = install_file(s, d)
        if d.name == "Caddyfile":
            rc, out = run([str(caddy_exe()), "validate", "--config", str(tmp), "--adapter", "caddyfile"])
            if rc:
                tmp.unlink()
                say("caddy", f"Caddyfile not replaced, validation failed:\n{out}")
                return False
        os.replace(tmp, d)
        say("caddy", f"wrote    {short(d)}")
    listening = run(["powershell", "-NoProfile", "-Command", "(Get-Process caddy -ErrorAction SilentlyContinue | Measure-Object).Count"])[1].strip() != "0"
    if listening:
        rc, out = run([str(caddy_exe()), "reload", "--config", str(CADDY_DIR / "Caddyfile"), "--adapter", "caddyfile"])
        say("caddy", "reloaded" if rc == 0 else f"reload failed: {out}")
        return rc == 0
    run(["schtasks", "/Run", "/TN", "Caddy-GPU-Box"])
    say("caddy", "started via the Caddy-GPU-Box task")
    return True


# ---------------------------------------------------------------- comfyui


def comfy_up():
    try:
        urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3)
        return True
    except OSError:
        return False


def comfy_queue_len():
    try:
        q = json.load(urllib.request.urlopen(COMFY_URL + "/queue", timeout=5))
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    except (OSError, ValueError):
        return 0


def restart_comfy():
    deadline = time.time() + 600
    while comfy_queue_len() and time.time() < deadline:
        time.sleep(5)
    if comfy_queue_len():
        say("comfyui", "queue still busy after 10 min; not restarting")
        return False
    # run_service.bat loops, so killing the listener restarts it 10 s later with the new files
    run(["powershell", "-NoProfile", "-Command",
         "Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue | "
         "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"])
    for _ in range(60):
        time.sleep(3)
        if comfy_up():
            say("comfyui", "restarted")
            return True
    run(["schtasks", "/Run", "/TN", "ComfyUI-GPU-Box"])
    say("comfyui", "did not come back in 3 min; started the ComfyUI-GPU-Box task")
    return False


def comfyui(push, a):
    if not PY.exists():
        say("comfyui", f"MISSING  {short(CEI)}: install ComfyUI-Easy-Install first (gpu-box/README.md, 'Fresh GPU box')")
        return False
    rows = [(s, d, file_state(s, d)) for c, s, d in FILES if c == "comfyui"]
    for s, d, st in rows:
        if st != "same":
            say("comfyui", f"{st:8} {short(d)}")
    rc, ver = run(["git", "-C", str(COMFY), "describe", "--tags"])
    say("comfyui", f"version  {ver if rc == 0 else '?'}; python {run([str(PY), '-c', 'import torch;print(torch.__version__)'])[1]}; "
                   f"{'answering' if comfy_up() else 'NOT answering'} on :8188")
    todo = [r for r in rows if r[2] != "same"]
    if not todo:
        say("comfyui", "files in sync")
        return True
    if not push or a.dry_run:
        return True
    for s, d, st in todo:
        os.replace(install_file(s, d), d)
        say("comfyui", f"wrote    {short(d)}")
    if a.restart:
        return restart_comfy()
    say("comfyui", "new run_service.bat / extra_model_paths.yaml apply on the next ComfyUI restart (push --restart)")
    return True


# ---------------------------------------------------------------- nodes


def node_rows(a):
    want = [n for n in load("nodes.json")["nodes"] if a.all or n["group"] == "core"]
    out = []
    for n in want:
        d = COMFY / "custom_nodes" / n["name"]
        if not (d / ".git").exists():
            out.append((n, d, "missing"))
            continue
        rc, head = run(["git", "-C", str(d), "rev-parse", "HEAD"])
        out.append((n, d, "same" if head == n["commit"] else f"at {head[:8]}"))
    return out


def nodes(push, a):
    if not COMFY.is_dir():
        say("nodes", "ComfyUI is not installed")
        return False
    rows = node_rows(a)
    bad = [r for r in rows if r[2] != "same"]
    for n, d, st in bad:
        say("nodes", f"{st:12} {n['name']} (want {n['commit'][:8]})")
    if not bad:
        say("nodes", f"{len(rows)} node(s) at their pinned commits")
        return True
    if not push or a.dry_run:
        return True
    ok = True
    for n, d, st in bad:
        steps = []
        if st == "missing":
            steps.append(["git", "clone", n["url"], str(d)])
        else:
            if run(["git", "-C", str(d), "status", "--porcelain"])[1]:
                say("nodes", f"skip {n['name']}: local changes in {short(d)}")
                continue
            steps.append(["git", "-C", str(d), "fetch", "origin"])
        steps.append(["git", "-C", str(d), "checkout", "--quiet", n["commit"]])
        for cmd in steps:
            rc, out = run(cmd, env=GIT_ENV)
            if rc:
                say("nodes", f"FAILED {n['name']}: {' '.join(cmd[:4])}: {out[-300:]}")
                ok = False
                break
        else:
            req = d / "requirements.txt"
            if req.exists():
                rc, out = run([str(PY), "-m", "pip", "install", "--no-warn-script-location", "-r", str(req)])
                if rc:
                    say("nodes", f"pip install -r failed for {n['name']}: {out[-300:]}")
                    ok = False
                    continue
            say("nodes", f"pinned   {n['name']} -> {n['commit'][:8]}")
    return ok


# ---------------------------------------------------------------- models


def models_base():
    m = re.search(r"base_path:\s*(\S+)", (VISION / "extra_model_paths.yaml").read_text(encoding="utf-8"))
    return Path(m.group(1).replace("__HOME__", str(HOME))) / "models"


def model_path(m):
    """Where the file is, checking ComfyUI's default models dir as well as the extra base_path."""
    for base in (models_base(), COMFY / "models"):
        p = base / m["dir"] / m["name"]
        if p.exists():
            return p
    return None


def hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        f = HOME / ".cache" / "huggingface" / "token"
        t = f.read_text().strip() if f.exists() else None
    return t


def download(m, token):
    dst = models_base() / m["dir"] / m["name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(m["url"], headers={"User-Agent": "gpu-box-deploy"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have == m["size"]:  # finished earlier, only the rename was missing
            os.replace(part, dst)
            return True
        say("models", f"FAILED   {m['name']}: HTTP {e.code}" + (" (gated: accept the licence on Hugging Face and set HF_TOKEN)" if e.code in (401, 403) else ""))
        return False
    mode = "ab" if have and resp.status == 206 else "wb"
    done, t0, last = (have if mode == "ab" else 0), time.time(), 0
    with open(part, mode) as f:
        while True:
            chunk = resp.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 15:
                last = time.time()
                print(f"         {m['name']}: {gb(done)} / {gb(m['size'])}", flush=True)
    if part.stat().st_size != m["size"]:
        say("models", f"FAILED   {m['name']}: got {part.stat().st_size} bytes, want {m['size']} (rerun to resume)")
        return False
    os.replace(part, dst)
    say("models", f"fetched  {m['name']} ({gb(m['size'])} in {time.time() - t0:.0f}s)")
    return True


def models(push, a):
    groups = set(a.group.split(","))
    want = [m for m in load("models.json")["models"] if m["group"] in groups]
    ok = True
    missing, wrong = [], []
    for m in want:
        p = model_path(m)
        if p is None:
            missing.append(m)
        elif p.stat().st_size != m["size"]:
            wrong.append(m)
    for m in missing:
        say("models", f"missing  {m['dir']}/{m['name']} ({gb(m['size'])}, {m['group']}{', gated' if m.get('gated') else ''})")
    for m in wrong:
        say("models", f"SIZE     {m['dir']}/{m['name']} differs from the pinned size (partial or replaced file)")
    if not missing and not wrong:
        say("models", f"{len(want)} model(s) present ({', '.join(sorted(groups))})")
        return True
    if not push or a.dry_run:
        return not wrong
    need = sum(m["size"] for m in missing)
    free = shutil.disk_usage(models_base().anchor or str(HOME)).free
    if need > free:
        say("models", f"need {gb(need)}, only {gb(free)} free on {models_base().anchor}")
        return False
    token = hf_token()
    for m in missing:
        ok &= download(m, token)
    return ok and not wrong


# ---------------------------------------------------------------- tasks


def task_xml(user, trigger, logon, restart, desc, cmd, args):
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    trig = ("<BootTrigger />" if trigger == "boot" else
            "<TimeTrigger><StartBoundary>2026-01-01T00:00:00</StartBoundary>"
            "<Repetition><Interval>PT5M</Interval><StopAtDurationEnd>true</StopAtDurationEnd></Repetition></TimeTrigger>")
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>{f'<Description>{esc(desc)}</Description>' if desc else ''}</RegistrationInfo>
  <Principals><Principal id="Author"><UserId>{esc(user)}</UserId><LogonType>{logon}</LogonType></Principal></Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>{'PT0S' if restart else 'PT2M'}</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    {'<StartWhenAvailable>true</StartWhenAvailable>' if not restart else ''}
    {f'<RestartOnFailure><Count>5</Count><Interval>{restart}</Interval></RestartOnFailure>' if restart else ''}
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
  </Settings>
  <Triggers>{trig}</Triggers>
  <Actions Context="Author"><Exec><Command>{esc(cmd)}</Command><Arguments>{esc(args)}</Arguments></Exec></Actions>
</Task>
"""


def task_live(name):
    rc, out = run(["schtasks", "/Query", "/TN", name, "/XML"])
    if rc:
        return None
    root = ET.fromstring(out.split("?>", 1)[1])
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    ex = root.find(".//t:Exec", ns)
    return (ex.findtext("t:Command", "", ns), ex.findtext("t:Arguments", "", ns),
            "boot" if root.find(".//t:BootTrigger", ns) is not None else "time")


def tasks(push, a):
    user = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}"
    ok, todo = True, []
    for name, trigger, logon, restart, desc, cmd, args in TASKS:
        live = task_live(name)
        want = (cmd, args, "boot" if trigger == "boot" else "time")
        if live == want:
            continue
        todo.append((name, trigger, logon, restart, desc, cmd, args))
        say("tasks", f"{'missing' if live is None else 'differs':8} {name}" + (f" (live: {live[0]} {live[1]})" if live else ""))
    if not todo:
        say("tasks", f"{len(TASKS)} task(s) in sync")
        return True
    if not push or a.dry_run:
        return True
    for t in todo:
        tmp = Path(os.environ.get("TEMP", ".")) / f"{re.sub(r'[^A-Za-z0-9]+', '_', t[0])}.xml"
        tmp.write_bytes(b"\xff\xfe" + task_xml(user, *t[1:]).encode("utf-16-le"))
        rc, out = run(["schtasks", "/Create", "/TN", t[0], "/XML", str(tmp), "/F"])
        tmp.unlink()
        if rc:
            say("tasks", f"FAILED   {t[0]}: {out}" + ("\n         boot triggers need an elevated shell: rerun as Administrator" if "denied" in out.lower() else ""))
            ok = False
        else:
            say("tasks", f"created  {t[0]}")
    return ok


# ---------------------------------------------------------------- unsloth


def unsloth(push, a):
    lora = HOME / "lora"
    have = UNSLOTH_PY.exists()
    if have:
        rc, out = run([str(UNSLOTH_PY), "-c", "import unsloth;print('version', unsloth.__version__)"])  # import itself prints banners
        m = re.search(r"^version (\S+)", out, re.M)
        ver = m.group(1) if m else "installed but `import unsloth` fails"
        say("unsloth", f"studio   {ver}")
    else:
        say("unsloth", f"missing  {short(UNSLOTH_PY)}")
    say("unsloth", f"~/lora   {'ok' if lora.is_dir() else 'missing (created on push)'}")
    if not push or a.dry_run:
        return True
    lora.mkdir(exist_ok=True)
    if have:
        return True
    say("unsloth", "running Unsloth's installer: irm https://unsloth.ai/install.ps1 | iex")
    rc, out = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://unsloth.ai/install.ps1 | iex"])
    print(out[-1500:])
    return rc == 0 and UNSLOTH_PY.exists()


# ---------------------------------------------------------------- main

COMPONENTS = {"prereq": prereq, "caddy": caddy, "comfyui": comfyui, "nodes": nodes, "tasks": tasks, "models": models, "unsloth": unsloth}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["status", "push"])
    ap.add_argument("components", nargs="*", help=", ".join(ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true", help="also the `extra` custom nodes")
    ap.add_argument("--group", default="image,video", help="model groups (default: both)")
    ap.add_argument("--restart", action="store_true", help="restart ComfyUI (when idle) after changing its files")
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("gpu-box/deploy.py runs on the GPU box (Windows); ../deploy.py is the Mac's")
    bad = [c for c in a.components if c not in COMPONENTS]
    if bad:
        ap.error(f"unknown component(s): {', '.join(bad)}")
    names = a.components or (ORDER if a.action == "status" else DEFAULT)
    push = a.action == "push"
    failed = [n for n in ORDER if n in names and not COMPONENTS[n](push, a)]
    if failed:
        print(f"\nneeds attention: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

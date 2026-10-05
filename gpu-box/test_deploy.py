"""Tests for gpu-box/deploy.py: the parts that don't touch the GPU box's services. Not deployed.
Run: cd gpu-box && uvx pytest -q"""

import http.server
import json
import os
import threading
import xml.etree.ElementTree as ET

import pytest

import deploy

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


@pytest.mark.parametrize("task", deploy.TASKS, ids=lambda t: t[0])
def test_task_xml_is_well_formed_and_carries_the_action(task):
    root = ET.fromstring(deploy.task_xml("GPU-BOX\\you", *task[1:]).split("?>", 1)[1])
    ex = root.find(".//t:Exec", NS)
    assert ex.findtext("t:Command", "", NS) == task[5] and ex.findtext("t:Arguments", "", NS) == task[6]
    assert (root.find(".//t:BootTrigger", NS) is not None) == (task[1] == "boot")
    assert root.findtext(".//t:LogonType", "", NS) == task[2]


def test_files_get_placeholders_filled_and_compare_ignoring_line_endings(tmp_path, monkeypatch):
    monkeypatch.setitem(deploy.PLACEHOLDERS, "__HOSTNAME__", "gpu-box.example.ts.net")
    src, dst = tmp_path / "a.yaml", tmp_path / "live.yaml"
    src.write_bytes(b"base_path: __HOME__\\models\r\nsite: __HOSTNAME__\r\n")
    assert deploy.file_state(src, dst) == "missing"
    dst.write_bytes(deploy.rendered(src).replace(b"\n", b"\r\n"))
    live = dst.read_bytes()
    assert str(deploy.HOME).encode() in live and b"gpu-box.example.ts.net" in live and b"__" not in live
    assert deploy.file_state(src, dst) == "same"
    dst.write_bytes(b"other\n")
    assert deploy.file_state(src, dst) == "differs"


def test_every_managed_file_exists_and_renders_completely():
    for _, src, _ in deploy.FILES:
        assert src.is_file(), src
        assert b"__HOME__" not in deploy.rendered(src) and b"__HOSTNAME__" not in deploy.rendered(src)


def test_hostname_comes_from_wandavision_conf_unless_the_environment_sets_it(tmp_path, monkeypatch):
    (tmp_path / "wandavision.conf").write_text("# comment\nGPU_BOX_HOSTNAME = gpu-box.example.ts.net\n")
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    monkeypatch.delenv("GPU_BOX_HOSTNAME", raising=False)
    assert deploy.setting("GPU_BOX_HOSTNAME", "localhost") == "gpu-box.example.ts.net"
    monkeypatch.setenv("GPU_BOX_HOSTNAME", "gpu-box.local")
    assert deploy.setting("GPU_BOX_HOSTNAME", "localhost") == "gpu-box.local"
    (tmp_path / "wandavision.conf").unlink()
    monkeypatch.delenv("GPU_BOX_HOSTNAME")
    assert deploy.setting("GPU_BOX_HOSTNAME", "localhost") == "localhost"


def test_models_live_under_the_profile_folder():
    base = str(deploy.models_base())
    assert base.startswith(str(deploy.HOME)) and "__HOME__" not in base and base.endswith("models")


def test_bat_and_ps1_are_written_with_crlf(tmp_path):
    src, dst = tmp_path / "x.bat", tmp_path / "out.bat"
    src.write_bytes(b"@echo off\nexit\n")
    tmp = deploy.install_file(src, dst)
    assert tmp.read_bytes() == b"@echo off\r\nexit\r\n"


def test_manifests_are_consistent():
    nodes = deploy.load("nodes.json")["nodes"]
    assert {n["group"] for n in nodes} <= {"core", "extra"} and all(len(n["commit"]) == 40 for n in nodes)
    assert len({n["name"] for n in nodes}) == len(nodes)
    models = deploy.load("models.json")["models"]
    assert {m["group"] for m in models} == {"image", "video"}
    assert all(m["size"] > 0 and m["url"].startswith("https://huggingface.co/") for m in models)
    wf = {p.name.split(".")[0] for p in (deploy.VISION / "workflows").glob("*.api.json")}
    assert wf == {m["for"] for m in models}, "every Vision workflow lists the models it needs, and every model is for one"


def test_workflows_name_only_models_in_the_manifest():
    """A workflow that loads a file models.json doesn't fetch would fail on a fresh GPU box."""
    names = {m["name"] for m in deploy.load("models.json")["models"]}
    keys = ("unet_name", "clip_name", "vae_name", "lora_name", "ckpt_name")
    for p in (deploy.VISION / "workflows").glob("*.api.json"):
        for node in json.loads(p.read_text(encoding="utf-8")).values():
            for k in keys:
                if k in node["inputs"]:
                    assert node["inputs"][k] in names, f"{p.name}: {node['inputs'][k]}"


class RangeHandler(http.server.BaseHTTPRequestHandler):
    data = os.urandom(300_000)

    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        if start >= len(self.data):
            self.send_response(416)
            self.end_headers()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(self.data) - start))
        self.end_headers()
        self.wfile.write(self.data[start:])

    def log_message(self, *a):
        pass


def test_download_resumes_a_partial_file(tmp_path, monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(deploy, "models_base", lambda: tmp_path)
    m = {"dir": "vae", "name": "x.safetensors", "size": len(RangeHandler.data), "url": f"http://127.0.0.1:{srv.server_port}/x"}
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "x.safetensors.part").write_bytes(RangeHandler.data[:100_000])
    assert deploy.download(m, None)
    assert (tmp_path / "vae" / "x.safetensors").read_bytes() == RangeHandler.data
    assert not (tmp_path / "vae" / "x.safetensors.part").exists()
    srv.shutdown()

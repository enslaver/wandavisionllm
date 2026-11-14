"""search/server.py against a fake Firecrawl + Context7. Run: cd search && uvx pytest -q (not deployed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import server

SEEN = []
MODE = {"docs": "ok", "web": "ok"}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(("POST", self.path, self.headers.get("Authorization"), body))
        if MODE["web"] == "down":
            return self.reply(500, {"error": "boom"})
        hit = {"url": "https://example.com/a", "title": "A", "description": "about a"}
        if "scrapeOptions" in body:
            hit["markdown"] = "x" * 9000
        self.reply(200, {"success": True, "data": {"web": [hit, dict(hit, url="https://example.com/b")]}})

    def do_GET(self):
        SEEN.append(("GET", self.path, self.headers.get("Authorization"), None))
        if MODE["docs"] == "boom":
            return self.reply(500, {"error": "boom"})
        if MODE["docs"] == "none":
            return self.reply(404, {"error": "no_documentation_found"})
        self.reply(200, {"codeSnippets": [{"libraryId": "/vercel/next.js", "codeTitle": "Route handler",
                                           "codeList": [{"language": "ts", "code": "export async function GET() {}"}]}],
                         "infoSnippets": [{"libraryId": "/vercel/next.js", "content": "Route handlers live in app/."}]})


@pytest.fixture(autouse=True)
def fake(tmp_path, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    env = tmp_path / "env"
    env.write_text("FIRECRAWL_API_KEY=fc-test\nCONTEXT7_API_KEY=\"c7-test\"\n")
    monkeypatch.setattr(server, "FIRECRAWL", base)
    monkeypatch.setattr(server, "CONTEXT7", base)
    monkeypatch.setattr(server, "ENV", str(env))
    SEEN.clear()
    MODE.update(docs="ok", web="ok")
    yield
    srv.shutdown()


def test_query_required_and_bad_source():
    assert server.run({})[0] == 400
    assert server.run({"query": "x", "source": "bogus"})[0] == 400
    assert server.run({"query": "x", "limit": "many"})[0] == 400


def test_auto_goes_web_for_a_plain_question_with_the_firecrawl_key():
    status, r = server.run({"query": "who won the 2026 world cup"})
    assert status == 200 and r["source"] == "web" and len(r["results"]) == 2
    assert SEEN[0][1] == "/v2/search" and SEEN[0][2] == "Bearer fc-test" and SEEN[0][3]["limit"] == 5


def test_auto_goes_docs_for_a_library_hint_or_a_docs_question():
    for body in ({"query": "streaming responses", "library": "next.js"},
                 {"query": "next.js documentation for route handlers"}):
        SEEN.clear()
        status, r = server.run(body)
        assert status == 200 and r["source"] == "docs"
        assert SEEN[0][0] == "GET" and SEEN[0][1].startswith("/api/v3/search?") and SEEN[0][2] == "Bearer c7-test"
        assert {x["type"] for x in r["results"]} == {"code", "info"}


def test_docs_hint_is_passed_as_library_params():
    server.run({"query": "q", "library": ["next.js", "openai"], "version": "15", "language": "TypeScript"})
    path = SEEN[0][1]
    assert path.count("library=") == 2 and "version=15" in path and "language=TypeScript" in path and "type=json" in path


def test_auto_falls_back_to_web_when_docs_find_nothing_but_explicit_docs_does_not():
    MODE["docs"] = "none"
    status, r = server.run({"query": "obscurelib documentation"})
    assert status == 200 and r["source"] == "web" and r["fell_back_from"] == "docs"
    status, r = server.run({"query": "obscurelib documentation", "source": "docs"})
    assert status == 404 and r["upstream_status"] == 404


def test_auto_falls_back_to_web_when_docs_error_but_explicit_docs_reports_it():
    MODE["docs"] = "boom"
    status, r = server.run({"query": "obscurelib documentation"})
    assert status == 200 and r["source"] == "web" and r["fell_back_from"] == "docs"
    status, r = server.run({"query": "obscurelib documentation", "source": "docs"})
    assert status == 502 and r["upstream_status"] == 500


def test_scrape_returns_trimmed_markdown():
    status, r = server.run({"query": "q", "source": "web", "scrape": True, "limit": 1})
    assert status == 200 and len(r["results"]) == 1 and r["results"][0]["markdown"].endswith("[truncated]")
    assert len(r["results"][0]["markdown"]) < 6100 and "scrapeOptions" in SEEN[0][3]


def test_limit_is_capped_and_upstream_failure_is_a_502():
    server.run({"query": "q", "source": "web", "limit": 99})
    assert SEEN[0][3]["limit"] == server.MAX_LIMIT
    MODE["web"] = "down"
    status, r = server.run({"query": "q", "source": "web"})
    assert status == 502 and r["upstream_status"] == 500


def test_no_key_still_sends_the_request(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ENV", str(tmp_path / "missing"))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert server.run({"query": "q", "source": "web"})[0] == 200
    assert SEEN[0][2] is None

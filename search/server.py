#!/usr/bin/env python3
"""search service: POST /search, behind LiteLLM's pass-through (:4000/ultron/search, LiteLLM key required).

One route that picks its source:
  web   Firecrawl v2 search (api.firecrawl.dev), optionally with each page's markdown
  docs  Context7 v3 search (context7.com): picks the libraries itself, returns code + info snippets
  auto  docs when a `library` hint is given or the query reads like a documentation question, else web;
        a docs search that finds nothing or fails falls back to web

Request  {"query": "...", "source": "auto|web|docs", "limit": 5, "scrape": false,
          "library": "next.js" | ["next.js", "openai"], "version": "15.4.0", "language": "Python"}
Response {"source": "web|docs", "query": "...", "results": [...], "fell_back_from": "docs"?}

Keys (FIRECRAWL_API_KEY, CONTEXT7_API_KEY) are read from ~/.litellm/env on every request, so adding one needs no
restart; both APIs also answer without a key at a lower rate limit. Listens on loopback only: LiteLLM's key check
(auth: true on the pass-through) is the gate. Python 3.9, stdlib only. Log: ~/.search/search.log (queries, never keys).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = os.environ.get("SEARCH_LISTEN", "127.0.0.1:8792")
ENV = os.path.expanduser(os.environ.get("SEARCH_ENV_FILE", "~/.litellm/env"))
LOG = os.path.expanduser(os.environ.get("SEARCH_LOG", "~/.search/search.log"))
FIRECRAWL = os.environ.get("FIRECRAWL_BASE", "https://api.firecrawl.dev")
CONTEXT7 = os.environ.get("CONTEXT7_BASE", "https://context7.com")
TIMEOUT_S = 40   # Context7 answers in 4-9 s; one 60 s hang was seen, so cut it short and retry the GET once
MAX_LIMIT = 10
MAX_PAGE_CHARS = 6000   # markdown kept per scraped page
MAX_SNIPPET_CHARS = 2500
MAX_BODY = 64 * 1024

DOCS_HINT = re.compile(
    r"\b(docs?|documentation|api reference|sdk|library|framework|package|how (?:do|can|to)\b.*\b(?:use|install|configure|set ?up)"
    r"|(?:import|install|configure|migrate)\b|usehook|use[A-Z]\w+|config(?:uration)? option)", re.I)


def key(name):
    """Env file value (re-read each call), else the process environment."""
    try:
        for line in open(ENV):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return os.environ.get(name, "")


class Upstream(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def fetch(url, body=None, token="", timeout=TIMEOUT_S, retries=0):
    """JSON from url. GET (body None) retries on a network error/timeout; HTTP errors are never retried."""
    for attempt in range(retries + 1):
        try:
            return fetch_once(url, body, token, timeout)
        except Upstream as e:
            if e.status != 502 or attempt == retries:
                raise


def fetch_once(url, body=None, token="", timeout=TIMEOUT_S):
    headers = {"Accept": "application/json", "User-Agent": "ultron-search/1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()[:400].decode("utf-8", "replace")
        raise Upstream(e.code, raw)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise Upstream(502, repr(e))


def trim(text, n):
    text = text or ""
    return text if len(text) <= n else text[:n] + " …[truncated]"


def search_web(q, limit, scrape):
    body = {"query": q, "limit": limit}
    if scrape:
        body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
    data = fetch(FIRECRAWL + "/v2/search", body, key("FIRECRAWL_API_KEY"))
    if isinstance(data, dict) and data.get("success") is False:
        raise Upstream(502, str(data.get("error") or data)[:300])
    hits = ((data or {}).get("data") or {}) if isinstance(data, dict) else {}
    web = hits.get("web") if isinstance(hits, dict) else hits   # v2 buckets by source; v1 returned a flat list
    out = []
    for h in (web or [])[:limit]:
        meta = h.get("metadata") or {}
        row = {"title": h.get("title") or meta.get("title") or "", "url": h.get("url") or meta.get("sourceURL") or "",
               "description": h.get("description") or meta.get("description") or ""}
        if scrape and h.get("markdown"):
            row["markdown"] = trim(h["markdown"], MAX_PAGE_CHARS)
        out.append(row)
    return out


def as_list(v):
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(x) for x in v if str(x).strip()] if isinstance(v, list) else []


def search_docs(q, limit, libraries, version, language):
    params = [("query", q), ("type", "json")]
    params += [("library", lib) for lib in libraries[:4]]
    if version and libraries:
        params.append(("version", version))
    if language:
        params.append(("language", language))
    data = fetch(CONTEXT7 + "/api/v3/search?" + urllib.parse.urlencode(params), token=key("CONTEXT7_API_KEY"), retries=1)
    out = []
    for s in ((data or {}).get("codeSnippets") or [])[:limit]:
        code = "\n\n".join(c.get("code", "") for c in (s.get("codeList") or []) if isinstance(c, dict))
        out.append({"type": "code", "library": s.get("libraryId"), "title": s.get("codeTitle"),
                    "content": trim(code, MAX_SNIPPET_CHARS)})
    for s in ((data or {}).get("infoSnippets") or [])[:limit]:
        out.append({"type": "info", "library": s.get("libraryId"), "title": s.get("breadcrumb") or "",
                    "content": trim(s.get("content"), MAX_SNIPPET_CHARS)})
    return out


def run(body):
    """(status, response dict)."""
    q = str(body.get("query") or "").strip()
    if not q:
        return 400, {"error": "query is required"}
    source = str(body.get("source") or "auto").lower()
    if source not in ("auto", "web", "docs"):
        return 400, {"error": "source must be auto, web or docs"}
    try:
        limit = max(1, min(int(body.get("limit") or 5), MAX_LIMIT))
    except (TypeError, ValueError):
        return 400, {"error": "limit must be an integer"}
    libraries = as_list(body.get("library"))
    chosen = source if source != "auto" else ("docs" if libraries or DOCS_HINT.search(q) else "web")
    resp = {"source": chosen, "query": q}
    try:
        if chosen == "docs":
            try:
                resp["results"] = search_docs(q, limit, libraries, str(body.get("version") or ""), str(body.get("language") or ""))
            except Upstream:
                if source != "auto":   # auto never fails on a docs hiccup: web still answers
                    raise
                resp["fell_back_from"] = "docs"
                resp["source"] = "web"
                resp["results"] = search_web(q, limit, bool(body.get("scrape")))
            if source == "auto" and not resp["results"] and "fell_back_from" not in resp:
                resp["fell_back_from"], resp["source"] = "docs", "web"
                resp["results"] = search_web(q, limit, bool(body.get("scrape")))
        else:
            resp["results"] = search_web(q, limit, bool(body.get("scrape")))
    except Upstream as e:
        return (502 if e.status >= 500 or e.status == 0 else e.status), {
            "error": f"{chosen} search failed upstream", "upstream_status": e.status, "detail": e.message}
    return 200, resp


def log(row):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "ultron-search/1"

    def log_message(self, *a):
        pass

    def send(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/healthz":
            self.send(200, {"ok": True, "firecrawl_key": bool(key("FIRECRAWL_API_KEY")), "context7_key": bool(key("CONTEXT7_API_KEY"))})
        else:
            self.send(404, {"error": "POST /search"})

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/search":
            return self.send(404, {"error": "POST /search"})
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self.send(413, {"error": "body too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as e:
            return self.send(400, {"error": "bad JSON: " + str(e)})
        status, resp = run(body)
        log({"ts": round(t0, 3), "status": status, "source": resp.get("source"), "asked": body.get("source") or "auto",
             "query": str(body.get("query") or "")[:200], "results": len(resp.get("results") or []),
             "fell_back_from": resp.get("fell_back_from"), "ms": int((time.time() - t0) * 1000)})
        self.send(status, resp)


def main():
    host, port = LISTEN.rsplit(":", 1)
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())

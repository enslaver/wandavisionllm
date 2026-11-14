# search

`POST <your-mac>:4000/ultron/search` - one route for web search (Firecrawl) and library docs (Context7), behind the LiteLLM key.

```
LiteLLM :4000  /ultron/search  (pass_through_endpoints, auth: true)  ->  search/server.py 127.0.0.1:8792  ->  Firecrawl v2 search | Context7 v3 search
```

```bash
curl -s http://127.0.0.1:4000/ultron/search -H "Authorization: Bearer $LITELLM_KEY" -H 'content-type: application/json' \
  -d '{"query": "next.js route handler streaming", "source": "auto", "limit": 5}'
```

| Field | Meaning |
|---|---|
| `query` | required |
| `source` | `auto` (default), `web`, `docs`. Auto picks docs when `library` is set or the query reads like a docs question, else web; a docs search that finds nothing falls back to web (`fell_back_from: docs`) |
| `limit` | 1-10, default 5 |
| `scrape` | web only: include each page's markdown (6,000 chars max each) |
| `library` | docs hint: a name (`next.js`) or Context7 id (`/vercel/next.js`), or a list of up to four |
| `version`, `language` | docs only; `version` needs a `library` |

Response: `{"source": "web|docs", "query", "results": [...]}`. Web rows: `title`, `url`, `description`, `markdown?`. Docs rows: `type` (`code`/`info`), `library`, `title`, `content`. (Not `/search`: LiteLLM's own `/search` route takes that path and wants a `search_tool_name`.) Errors: 400 bad request, upstream status passed through (404 = no docs) or 502 with `detail`.

Keys: `FIRECRAWL_API_KEY` and `CONTEXT7_API_KEY` in `~/.litellm/env`, re-read on every request (no restart). Both APIs also answer without a key at a lower rate limit. `curl 127.0.0.1:8792/healthz` shows which keys are set. Log: `~/.search/search.log` (queries and counts, never keys).

Deploy (optional; nothing runs until you do): `./deploy.py push search` (copies to `~/search/`, installs LaunchAgent `com.ultron.search`). The route itself is in `litellm/config.yaml`, so a first deploy also needs `./deploy.py push litellm`. Tests: `cd search && uvx pytest -q` (fake upstreams).


#!/usr/bin/env python3
"""Per-tier benchmark: reasoning, tool calling, coding, plus speed. Not deployed; run on ultron.

    python3 bench.py ultron/fable --out /tmp/bench-fable.json
    python3 bench.py ultron/opus ultron/sonnet ultron/haiku --out-dir ./bench-results

Every request goes through LiteLLM with `x-route: private` (local only, no cloud fallback) at
reasoning_effort=medium, so the hooks, defaults and thinking budgets are the ones real clients get.
Reasoning and coding are graded by machine (exact answer / unit tests in a subprocess); tool
calling is graded on the call itself (name, argument values and types). stdlib only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

LITELLM = os.environ.get("ULTRON_LITELLM", "http://127.0.0.1:4000")
ENV = os.path.expanduser("~/.litellm/env")
HEADERS = {"X-Claude-Code-Agent-Id": "bench", "x-route": "private", "x-hindsight": "off"}
EFFORT = "medium"


def key(name):
    try:
        for line in open(ENV):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.environ.get(name, "")


def call(model, messages, tools=None, max_tokens=8000, timeout=900):
    h = {"Authorization": "Bearer " + key("LITELLM_MASTER_KEY"), "Content-Type": "application/json",
         "X-Claude-Code-Session-Id": "bench-" + uuid.uuid4().hex[:8], **HEADERS}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "reasoning_effort": EFFORT}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(LITELLM + "/v1/chat/completions", json.dumps(body).encode(), h, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            status = r.status
    except urllib.error.HTTPError as e:
        data, status = {"error": e.read()[:300].decode("utf-8", "replace")}, e.code
    except Exception as e:  # timeout, reset
        data, status = {"error": repr(e)}, 0
    dt = time.time() - t0
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    msg = choice.get("message") or {}
    usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
    ct = usage.get("completion_tokens") or 0
    return {"status": status, "elapsed": dt, "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "", "tool_calls": msg.get("tool_calls") or [],
            "finish": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens") or 0,
            "completion_tokens": ct, "tok_s": ct / dt if dt and ct else 0.0,
            "error": data.get("error") if isinstance(data, dict) else None}


# ----------------------------------------------------------------------------- reasoning

SUFFIX = "\n\nThink it through, then end your reply with a final line of exactly `ANSWER: <value>` (just the value)."


def _sum_not_35():
    return sum(n for n in range(1, 101) if n % 3 and n % 5)


REASONING = [
    ("bat-and-ball", "A notebook and a pen cost $2.10 together. The notebook costs $2.00 more than the pen. "
     "How many cents does the pen cost?", "5"),
    ("bird-and-trains", "Two trains start 300 km apart and drive toward each other at 70 km/h and 80 km/h. "
     "A bird flies back and forth between their fronts at 120 km/h from the moment they start until they meet. "
     "How many km does the bird fly in total?", "240"),
    ("inclusion-exclusion", "What is the sum of the integers from 1 to 100 that are divisible by neither 3 nor 5?",
     str(_sum_not_35())),
    ("anagrams", "How many distinct arrangements of the letters of BANANA are there?", "60"),
    ("modpow", "What is the remainder when 7^2026 is divided by 13?", str(pow(7, 2026, 13))),
    ("water-jugs", "You have an unlimited water supply and two jugs of 3 and 5 litres with no markings. "
     "What is the minimum number of fill/empty/pour actions needed to measure exactly 4 litres?", "6"),
    ("monty-100", "There are 100 doors, one hides a car. You pick door 1. The host, who knows where the car is, "
     "opens 98 of the other doors, all empty. Give the probability of winning if you switch to the one remaining "
     "door, as a decimal.", "0.99"),
    ("age-order", "Alice is older than Bob. Carol is younger than Bob. Dave is older than Alice but younger than "
     "Eve. Who is the third oldest of the five? Answer with the name.", "alice"),
]


def norm(s):
    s = s.strip().strip("`*$ .").lower()
    s = re.sub(r"[,\s]", "", s)
    return s


def grade_reasoning(r, want):
    found = re.findall(r"ANSWER:\s*(.+)", r["content"])
    if not found:
        return False, "no ANSWER line"
    got = norm(found[-1])
    ok = got == norm(want) or (re.fullmatch(r"-?\d+(\.\d+)?", got) is not None
                               and re.fullmatch(r"-?\d+(\.\d+)?", norm(want)) is not None
                               and abs(float(got) - float(norm(want))) < 1e-9)
    return ok, f"got {found[-1].strip()[:40]!r} want {want}"


# ----------------------------------------------------------------------------- tool calling

def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool("get_weather", "Get the current weather for a city",
          {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    _tool("search_flights", "Search flights between airports",
          {"origin": {"type": "string", "description": "IATA code"}, "destination": {"type": "string", "description": "IATA code"},
           "date": {"type": "string", "description": "YYYY-MM-DD"}, "passengers": {"type": "integer"}},
          ["origin", "destination", "date"]),
    _tool("calculator", "Evaluate an arithmetic expression", {"expression": {"type": "string"}}, ["expression"]),
    _tool("read_file", "Read a text file from disk", {"path": {"type": "string"}}, ["path"]),
    _tool("create_event", "Create a calendar event",
          {"title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}},
           "duration_minutes": {"type": "integer"}}, ["title", "attendees", "duration_minutes"]),
]


def parse_calls(r):
    out = []
    for tc in r["tool_calls"]:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = None
        out.append((fn.get("name"), args))
    return out


def t_single(r):
    c = parse_calls(r)
    ok = len(c) == 1 and c[0][0] == "get_weather" and c[0][1] and str(c[0][1].get("city", "")).lower() == "tokyo" \
        and c[0][1].get("unit") == "fahrenheit"
    return ok, str(c)[:120]


def t_flights(r):
    c = parse_calls(r)
    a = c[0][1] if c and c[0][1] else {}
    ok = bool(c) and c[0][0] == "search_flights" and a.get("origin") == "SFO" and a.get("destination") == "JFK" \
        and a.get("date") == "2026-11-02" and a.get("passengers") == 3
    return ok, str(c)[:140]


def t_parallel(r):
    c = parse_calls(r)
    cities = sorted(str((a or {}).get("city", "")).lower() for n, a in c if n == "get_weather")
    return cities == ["berlin", "paris"], str(c)[:140]


def t_notool(r):
    return not r["tool_calls"] and "paris" in r["content"].lower(), (r["content"][:80] or str(parse_calls(r)))


def t_calc(r):
    c = parse_calls(r)
    ok = False
    if c and c[0][0] == "calculator" and c[0][1]:
        e = str(c[0][1].get("expression", ""))
        ok = bool(re.fullmatch(r"[\d\s*x×().+\-/]+", e)) and "1234" in e and "5678" in e
    return ok, str(c)[:100]


def t_readfile(r):
    c = parse_calls(r)
    return bool(c) and c[0][0] == "read_file" and (c[0][1] or {}).get("path") == "/etc/hostname", str(c)[:100]


def t_event(r):
    c = parse_calls(r)
    a = c[0][1] if c and c[0][1] else {}
    ok = bool(c) and c[0][0] == "create_event" and a.get("title") == "Roadmap" \
        and sorted(a.get("attendees") or []) == ["ann@example.com", "bob@example.com"] and a.get("duration_minutes") == 45
    return ok, str(c)[:160]


def t_followup(r):
    return not r["tool_calls"] and "18" in r["content"], r["content"][:100]


FOLLOWUP = [
    {"role": "user", "content": "What's the temperature in Oslo right now?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo", "unit": "celsius"})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Oslo", "temp_c": 18, "sky": "cloudy"})},
]

TOOL_CASES = [
    ("single-call-enum", [{"role": "user", "content": "What's the weather in Tokyo, in fahrenheit?"}], t_single),
    ("typed-args", [{"role": "user", "content": "Find flights for 3 people from SFO to JFK on 2026-11-02."}], t_flights),
    ("parallel-calls", [{"role": "user", "content": "Get the weather in Paris and in Berlin, celsius."}], t_parallel),
    ("no-tool-needed", [{"role": "user", "content": "What is the capital of France? Answer from memory."}], t_notool),
    ("pick-right-tool", [{"role": "user", "content": "Use a tool to compute 1234 * 5678."}], t_calc),
    ("exact-path", [{"role": "user", "content": "Show me the contents of /etc/hostname."}], t_readfile),
    ("array-args", [{"role": "user", "content": "Schedule a 45 minute sync titled 'Roadmap' with ann@example.com and bob@example.com."}], t_event),
    ("use-tool-result", FOLLOWUP, t_followup),
]


# ----------------------------------------------------------------------------- coding

CODING = [
    ("merge_intervals", "Write `merge_intervals(intervals)`: given a list of [start, end] pairs (unsorted, may overlap "
     "or touch), return the merged list of [start, end] lists sorted by start. Touching intervals (end == next start) merge.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\n"
     "assert merge_intervals([[5,6],[1,2],[2,3]]) == [[1,3],[5,6]]\nassert merge_intervals([[1,10],[2,3]]) == [[1,10]]"),
    ("LRUCache", "Write a class `LRUCache(capacity)` with `get(key)` (returns -1 if missing) and `put(key, value)`; "
     "both mark the key most recently used; put beyond capacity evicts the least recently used key. O(1) per op.",
     "c = LRUCache(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1) == 1\nc.put(3,3)\nassert c.get(2) == -1\n"
     "c.put(4,4)\nassert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4\nc.put(3,30)\nassert c.get(3) == 30"),
    ("roman_to_int", "Write `roman_to_int(s)` converting a valid Roman numeral string (1..3999) to an int.",
     "assert roman_to_int('III') == 3\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994\n"
     "assert roman_to_int('IX') == 9\nassert roman_to_int('MMMCMXCIX') == 3999"),
    ("parse_duration", "Write `parse_duration(s)` returning total seconds for strings like '1h30m15s', '45s', '2h', "
     "'3m5s', made of optional integer parts with units h, m, s in that order. Empty string returns 0; "
     "raise ValueError for anything that does not match (e.g. '5x', '10', '1s2h').",
     "assert parse_duration('1h30m15s') == 5415\nassert parse_duration('45s') == 45\nassert parse_duration('2h') == 7200\n"
     "assert parse_duration('3m5s') == 185\nassert parse_duration('') == 0\n"
     "for bad in ('5x', '10', '1s2h', 'h', '1h1h'):\n    try:\n        parse_duration(bad)\n    except ValueError:\n        pass\n"
     "    else:\n        raise AssertionError(bad)"),
    ("flatten", "Write `flatten(x)`: flatten arbitrarily nested lists/tuples into one flat list, preserving order. "
     "Strings and bytes are atoms (not split). Must handle nesting depth 5000 without RecursionError.",
     "assert flatten([1,[2,[3,(4,5)]],6]) == [1,2,3,4,5,6]\nassert flatten(['ab',['cd']]) == ['ab','cd']\n"
     "assert flatten([]) == []\nd = []\nfor _ in range(5000):\n    d = [d, 1]\nassert flatten(d) == [1]*5000"),
    ("lcs_length", "Write `lcs_length(a, b)` returning the length of the longest common subsequence of two strings.",
     "assert lcs_length('abcde','ace') == 3\nassert lcs_length('abc','def') == 0\nassert lcs_length('','abc') == 0\n"
     "assert lcs_length('AGGTAB','GXTXAYB') == 4\nassert lcs_length('a'*500,'a'*500) == 500"),
    ("topo_sort", "Write `topo_sort(graph)` where graph is a dict node -> list of nodes it depends on (every node "
     "appears as a key). Return a list with each node after all its dependencies, or None if there is a cycle.",
     "g = {'a':['b','c'],'b':['c'],'c':[],'d':['a']}\no = topo_sort(g)\nassert sorted(o) == sorted(g)\n"
     "assert all(o.index(d) < o.index(n) for n in g for d in g[n])\nassert topo_sort({'a':['b'],'b':['a']}) is None\n"
     "assert topo_sort({'a':['a']}) is None\nassert topo_sort({}) == []"),
    ("word_wrap", "Write `word_wrap(text, width)` returning a list of lines: greedily pack whitespace-separated words "
     "into lines of at most `width` characters, single spaces between words; a word longer than width gets its own line "
     "unbroken.",
     "assert word_wrap('the quick brown fox', 9) == ['the quick','brown fox']\n"
     "assert word_wrap('a bb ccc', 3) == ['a','bb','ccc']\nassert word_wrap('supercalifragilistic hi', 5) == ['supercalifragilistic','hi']\n"
     "assert word_wrap('', 10) == []\nassert word_wrap('  a   b ', 10) == ['a b']"),
]


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(code, tests):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(code + "\n\n" + tests + "\nprint('PASS')\n")
        try:
            out = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=20, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if out.returncode == 0 and "PASS" in out.stdout:
            return True, "tests pass"
        return False, (out.stderr.strip().splitlines() or ["failed"])[-1][:120]


# ----------------------------------------------------------------------------- driver

def bench(model, sections):
    res = {"model": model, "effort": EFFORT, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "reasoning": [],
           "tools": [], "coding": []}
    # warm (excluded from stats): loads the tier
    call(model, [{"role": "user", "content": "hi"}], max_tokens=16, timeout=600)

    def record(bucket, name, r, ok, detail):
        row = {"name": name, "ok": bool(ok), "detail": detail, "elapsed": round(r["elapsed"], 1),
               "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"],
               "tok_s": round(r["tok_s"], 1), "finish": r["finish"], "status": r["status"],
               "reasoning_chars": len(r["reasoning"]), "error": str(r["error"])[:160] if r["error"] else None}
        res[bucket].append(row)
        print(f"  [{model}] {bucket:9} {'PASS' if ok else 'FAIL'} {name:22} {row['elapsed']:6.1f}s "
              f"{row['completion_tokens']:6}tok {row['tok_s']:6.1f}/s  {detail[:70]}", flush=True)

    if "reasoning" in sections:
        for name, q, want in REASONING:
            r = call(model, [{"role": "user", "content": q + SUFFIX}], max_tokens=16000)
            ok, d = grade_reasoning(r, want) if r["status"] == 200 else (False, f"http {r['status']} {r['error']}")
            record("reasoning", name, r, ok, d)
    if "tools" in sections:
        for name, msgs, grader in TOOL_CASES:
            r = call(model, msgs, tools=TOOLS, max_tokens=6000)
            ok, d = grader(r) if r["status"] == 200 else (False, f"http {r['status']} {r['error']}")
            record("tools", name, r, ok, d)
    if "coding" in sections:
        for name, spec, tests in CODING:
            r = call(model, [{"role": "user", "content": spec + "\n\nReply with one ```python block holding only the "
                              "implementation (no tests, no example usage)."}], max_tokens=12000)
            ok, d = run_tests(extract_code(r["content"]), tests) if r["status"] == 200 else (False, f"http {r['status']} {r['error']}")
            record("coding", name, r, ok, d)

    allrows = res["reasoning"] + res["tools"] + res["coding"]
    toks = [x["tok_s"] for x in allrows if x["tok_s"]]
    res["summary"] = {k: {"passed": sum(x["ok"] for x in res[k]), "total": len(res[k])} for k in ("reasoning", "tools", "coding")}
    res["summary"]["median_tok_s"] = round(sorted(toks)[len(toks) // 2], 1) if toks else 0
    res["summary"]["total_completion_tokens"] = sum(x["completion_tokens"] for x in allrows)
    res["summary"]["total_seconds"] = round(sum(x["elapsed"] for x in allrows), 1)
    print(f"  [{model}] SUMMARY {json.dumps(res['summary'])}", flush=True)
    return res


def main():
    global EFFORT
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--effort", default=EFFORT, help="reasoning_effort for every request")
    ap.add_argument("--sections", default="reasoning,tools,coding")
    a = ap.parse_args()
    EFFORT = a.effort
    os.makedirs(a.out_dir, exist_ok=True)
    for m in a.models:
        res = bench(m, set(a.sections.split(",")))
        path = os.path.join(a.out_dir, "bench-" + m.replace("/", "-") + ("" if EFFORT == "medium" else "-" + EFFORT) + ".json")
        json.dump(res, open(path, "w"), indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()

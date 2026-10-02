#!/usr/bin/env python3
"""make_dataset.py — recovery examples for a LoRA, from agent traces (rejection sampling).

A trace is one request as the tier saw it: {"tier": ..., "messages": [...], "tools": [...]} in OpenAI chat
format (after ultron_admit's normalize_history). ultron_admit's trace tap writes one per conversation to
~/.ultron/traces/; anthropic_to_trace.py converts a captured /v1/messages body (no "tier": always kept).
Only traces of --tier (default sonnet) are used, and that tier samples and judges.

For every failure point in a trace (a tool result that is an error, or the same call returning the
same result again), whatever the tool:
  1. compress the history before it to fit training (system + goal + recent turns, long results cut)
  2. sample K next turns from the tier itself (llama-swap :8001)
  3. keep a candidate only if it is a well-formed tool call, repeats none of the failed calls, and
     the tier, asked as a judge, says it responds to what the error says
  4. write {"messages": prompt + [chosen turn], "tools": [...]} (mlx_lm chat+tools format)

    ~/lora/.venv/bin/python make_dataset.py ~/lora/traces/*.json --out ~/lora/data/v0 [--tier sonnet] [--k 6]

Samples and verdicts are cached in <out>/cache.jsonl, so a rerun resumes.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litellm"))
import ultron_admit as ua  # noqa: E402  (repair_split_tool_calls; stdlib only, no side effects)

API = "http://127.0.0.1:8001/v1/chat/completions"
TIER = "sonnet"  # llama-swap model that samples and judges; main() sets it from --tier
# Claude Code's record of a tool call split by the LiteLLM stream bug (fixed 2026-10-01, see litellm/README.md).
# repair_split_tool_calls rejoins most of them; a point whose prompt still holds one is skipped.
POISON = "__unparsedToolInput"

ERR = re.compile(r"^(Exit code [1-9]|<tool_use_error>|Error|error:|fatal:|Traceback \(most recent)"
                 r"|Usage: |Invalid option|Unknown command|No such file|does not exist|command not found"
                 r"|Permission denied|not recognized as", re.M)

SYS_CHARS, GOAL_CHARS, RESULT_CHARS, DESC_CHARS = 10000, 3000, 2000, 800
TAIL = 12  # messages of recent history kept before the failure point
CORE_TOOLS = {"Bash", "Read", "Edit", "Write", "Grep", "Glob"}  # offered even if the trace never used them


def text(content):
    if isinstance(content, list):
        return "\n".join(b.get("text", "[image]") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def cut(s, n):
    return s if len(s) <= n else s[: n * 2 // 3] + f"\n…[{len(s) - n} chars cut]…\n" + s[-n // 3:]


def args_of(call):
    a = call.get("function", {}).get("arguments")
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except ValueError:
        return {"_raw": a}


def sig(call):
    return call.get("function", {}).get("name", "") + " " + json.dumps(args_of(call), sort_keys=True)


def plain(m):
    """A message the chat template renders: text content, dict tool-call arguments, no images."""
    out = {"role": m["role"], "content": text(m.get("content"))}
    if m["role"] == "tool":
        out["content"] = cut(out["content"], RESULT_CHARS)
        if m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
    if m.get("tool_calls"):
        out["tool_calls"] = [{"id": c.get("id"), "type": "function",
                              "function": {"name": c["function"]["name"], "arguments": args_of(c)}}
                             for c in m["tool_calls"]]
    return out


def failure_points(msgs):
    """(i, failed, batch_bad): i is the last tool message of a batch where a result is an error or a
    repeat; failed holds every call that failed or repeated up to i, batch_bad those of this batch."""
    seen, failed, points = {}, set(), []
    calls = {c["id"]: c for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or []}
    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        c = calls.get(m.get("tool_call_id"))
        if not c:
            continue
        s, r = sig(c), text(m.get("content")).strip()
        bad = bool(ERR.search(r[:400])) or seen.get(s) == r
        seen[s] = r
        if bad:
            failed.add(s)
        last_of_batch = i + 1 >= len(msgs) or msgs[i + 1].get("role") != "tool"
        if last_of_batch:
            batch_bad = frozenset(sig(c2) for c2 in batch_calls(msgs, i, calls) if sig(c2) in failed)
            if bad or batch_bad:
                points.append((i, set(failed), batch_bad))
    return points


def batch_calls(msgs, i, calls):
    j = i
    while j >= 0 and msgs[j].get("role") == "tool":
        j -= 1
    return [calls[m["tool_call_id"]] for m in msgs[j + 1:i + 1] if m.get("tool_call_id") in calls]


def compress(msgs, i, tools):
    head = []
    if msgs and msgs[0].get("role") == "system":
        head.append({"role": "system", "content": cut(text(msgs[0]["content"]), SYS_CHARS)})
    goal = next((k for k, m in enumerate(msgs) if m.get("role") == "user"), None)
    if goal is not None:
        head.append({"role": "user", "content": cut(text(msgs[goal]["content"]), GOAL_CHARS)})
    s = max(i + 1 - TAIL, (goal or 0) + 1)
    while s <= i and msgs[s].get("role") != "assistant":
        s += 1
    tail = [plain(m) for m in msgs[s:i + 1]]
    used = {c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []}
    tl = []
    for t in tools or []:
        f = t.get("function", t)
        name = f.get("name", "")
        if name in used or name in CORE_TOOLS:
            tl.append({"type": "function", "function": {"name": name, "description": cut(f.get("description", ""), DESC_CHARS),
                                                         "parameters": f.get("parameters", {})}})
    return head + tail, tl


def seq_limit():
    """(tokenizer, max_seq_length) from <TIER>.yaml, or (None, 0) for a tier not set up for training.
    mlx_lm truncates longer examples from the end, which cuts the trained turn: a valid row whose
    turn is gone entirely makes its val loss nan (v0: prompt 8245 tokens, limit 8192)."""
    cfg_p = Path(__file__).with_name(f"{TIER}.yaml")
    if not cfg_p.exists():
        return None, 0
    import yaml
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(open(cfg_p))
    base = os.path.expanduser(os.environ.get("LORA_BASE") or f"~/lora/base/{TIER}-4bit")  # the base view run.sh trains
    return AutoTokenizer.from_pretrained(base), int(cfg.get("max_seq_length") or 0)


def chat(messages, tools, max_tokens, temperature):
    body = {"model": TIER, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(API, json.dumps(body).encode(), {"Content-Type": "application/json"})
    think, content, calls = "", "", {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            for ch in json.loads(line[6:]).get("choices") or []:
                d = ch.get("delta") or {}
                think += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
    return think.strip(), content.strip(), [calls[k] for k in sorted(calls)]


JUDGE = ("An agent's tool call failed or returned the same thing again. Decide whether its next step "
         "responds to what the result says, or blindly retries / ignores it.\n\n"
         "Last calls and results:\n{last}\n\nNext step it proposes:\n{nxt}\n\n"
         "Answer with one word on the last line: YES (it responds to the result: different arguments, a "
         "different command or tool, or a check of why it failed) or NO.")


def judge(prompt_msgs, cand):
    last = "\n".join(f"[{m['role']}] {cut(text(m.get('content')) or json.dumps(m.get('tool_calls'), default=str), 1500)}"
                     for m in prompt_msgs[-4:])
    nxt = "\n".join(f"{c['name']} {c['arguments']}" for c in cand["calls"])
    _, out, _ = chat([{"role": "user", "content": JUDGE.format(last=last, nxt=nxt)}], None, 1500, 0.2)
    return out.strip().split()[-1].strip(".*").upper() == "YES" if out.strip() else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="sonnet", help="use only traces of this tier; it also samples and judges")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-per-sig", type=int, default=2, help="failure points kept per identical failed call")
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--no-sample", action="store_true", help="only use points already in the cache")
    a = ap.parse_args()
    global TIER
    TIER = a.tier
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    cache_p = out / "cache.jsonl"
    cache = {}
    if cache_p.exists():
        for l in cache_p.open():
            d = json.loads(l); cache[d["key"]] = d
    examples, used = [], set()
    stats = {"traces": 0, "other_tier": 0, "points": 0, "kept": 0, "no_pass": 0, "poisoned": 0, "duplicate": 0,
             "too_long": 0}
    tok, limit = seq_limit()
    for tp in a.traces:
        tr = json.load(open(tp))
        if tr.get("tier", a.tier) != a.tier:
            stats["other_tier"] += 1
            continue
        stats["traces"] += 1
        msgs, tools = ua.repair_split_tool_calls(tr["messages"]), tr.get("tools")
        per_sig = {}
        for i, failed, batch_bad in failure_points(msgs):
            per_sig[batch_bad] = per_sig.get(batch_bad, 0) + 1
            if per_sig[batch_bad] > a.max_per_sig:
                continue
            stats["points"] += 1
            prompt, tl = compress(msgs, i, tools)
            names = {t["function"]["name"] for t in tl}
            if POISON in json.dumps(prompt, default=str):
                stats["poisoned"] += 1
                continue
            key = hashlib.sha1(json.dumps([prompt, sorted(names)], sort_keys=True).encode()).hexdigest()
            if key in used:  # subagents and resumed sessions share history: same point, other trace file
                stats["duplicate"] += 1
                continue
            used.add(key)
            if key not in cache and a.no_sample:
                continue
            if key not in cache:
                cands = []
                for _ in range(a.k):
                    th, ct, calls = chat(prompt, tl, 2048, a.temperature)
                    cands.append({"think": th, "content": ct, "calls": calls})
                prev = {sig({"function": c["function"]}) for m in prompt[-6:] for c in m.get("tool_calls") or []}
                for c in cands:
                    ok = bool(c["calls"]) and all(x["name"] in names for x in c["calls"])
                    try:
                        sigs = {sig({"function": {"name": x["name"], "arguments": json.loads(x["arguments"] or "{}")}}) for x in c["calls"]}
                    except ValueError:
                        ok, sigs = False, set()
                    c["valid"] = ok
                    c["repeat"] = bool(sigs & (failed | prev))
                    c["judge"] = judge(prompt, c) if ok and not c["repeat"] else None
                cache[key] = {"key": key, "trace": str(tp), "i": i, "cands": cands}
                with cache_p.open("a") as f:
                    f.write(json.dumps(cache[key]) + "\n")
            good = [c for c in cache[key]["cands"] if c["valid"] and not c["repeat"] and c["judge"]]
            print(f"{Path(tp).name} @{i}: {len(good)}/{len(cache[key]['cands'])} pass "
                  f"(repeat {sum(1 for c in cache[key]['cands'] if c['repeat'])})", flush=True)
            if not good:
                stats["no_pass"] += 1
                continue
            c = random.Random(key).choice(good)
            turn = {"role": "assistant",
                    "content": (f"<think>\n{c['think']}\n</think>\n\n" if c["think"] else "") + c["content"],
                    "tool_calls": [{"id": f"call_{n}", "type": "function",
                                    "function": {"name": x["name"], "arguments": json.loads(x["arguments"] or "{}")}}
                                   for n, x in enumerate(c["calls"])]}
            if tok and limit and len(tok(tok.apply_chat_template(prompt + [turn], tools=tl, tokenize=False))["input_ids"]) > limit:
                stats["too_long"] += 1
                continue
            examples.append({"messages": prompt + [turn], "tools": tl})
            stats["kept"] += 1
    random.Random(0).shuffle(examples)
    nv = max(1, int(len(examples) * a.valid)) if len(examples) > 1 else 0
    for name, rows in (("valid", examples[:nv]), ("train", examples[nv:])):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(stats), f"-> {out}/train.jsonl ({len(examples) - nv}), valid.jsonl ({nv})")


if __name__ == "__main__":
    sys.exit(main())

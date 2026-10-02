#!/usr/bin/env python3
"""eval.py — does an adapter stop repeating failed calls? Base vs adapter on held-out failure points.

For each example in a dataset file (its prompt = messages[:-1]), sample N next turns and count:
  repeat  the call is one that already failed or repeated in the prompt (what loop_breaker catches)
  valid   a parseable tool call to a tool that exists
Runs through mlx_lm (no MTP), so it measures the weights, not the server.

    ~/lora/.venv/bin/python eval.py ~/lora/data/v0/valid.jsonl --adapter ~/lora/runs/v0 [--n 4]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset as md  # noqa: E402
from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

BASE = os.path.expanduser(os.environ.get("LORA_BASE", "~/lora/base/sonnet-4bit"))  # text-only view of the pack (README)
CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    out = []
    for name, body in CALL.findall(text.split("</think>")[-1]):
        args = {}
        for k, v in PARAM.findall(body):
            try:
                args[k] = json.loads(v)
            except ValueError:
                args[k] = v
        out.append({"function": {"name": name, "arguments": args}})
    return out


def score(model, tok, rows, n, temp, max_tokens):
    sampler = make_sampler(temp=temp, top_p=0.95, top_k=20)
    tot = {"samples": 0, "repeat": 0, "valid": 0, "no_call": 0}
    for r in rows:
        prompt_msgs, tools = r["messages"][:-1], r.get("tools")
        names = {t["function"]["name"] for t in tools or []}
        failed = set().union(*[f for _, f, _ in md.failure_points(prompt_msgs)] or [set()])
        prompt = tok.apply_chat_template(prompt_msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        for _ in range(n):
            calls = parse_calls(generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler))
            tot["samples"] += 1
            if not calls:
                tot["no_call"] += 1
                continue
            tot["valid"] += all(c["function"]["name"] in names for c in calls)
            tot["repeat"] += any(md.sig(c) in failed for c in calls)
    s = tot["samples"] or 1
    return {k: v if k == "samples" else round(v / s, 3) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--adapter")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=1536)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.data).expanduser())]
    for label, adapter in (("base", None), ("adapter", a.adapter)) if a.adapter else (("base", None),):
        model, tok = load(a.base, adapter_path=adapter)
        print(label, json.dumps(score(model, tok, rows, a.n, a.temp, a.max_tokens)), flush=True)
        del model


if __name__ == "__main__":
    main()

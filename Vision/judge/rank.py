#!/usr/bin/env python3
"""rank.py PROMPT IMAGE IMAGE [IMAGE ...] — rank generated images with the stack's image judge (ultron/judge, SkyJM-Gen-4B).

  rank.py "a red fox sitting in snow, golden hour" take1.png take2.png take3.png
  rank.py --base https://your-mac.example.ts.net/llm -v "prompt" a.jpg b.jpg   # from another machine
  rank.py --json "prompt" *.png                                                 # machine-readable result

Every pair is judged twice, in both orders (A/B then B/A), because pairwise judges lean toward one
position; SkyJM's own benchmark runs do the same. A pair counts as a win only when both orders pick
the same image; otherwise it is a tie. Score = wins + ties/2. N images cost N*(N-1) calls, one at a
time (one GPU).

The model is SkyJM-Gen-4B (skylenage-ai, Apache-2.0), a Qwen3.5-4B fine-tuned to judge text-to-image
pairs: it writes a weighted rubric for the prompt, scores both images, and ends with \\boxed{A} or
\\boxed{B}. It is not a tier: it is `[judge]` in litellm/tiers.conf with `routed = no`, so LiteLLM serves it
only as ultron/judge. llama-swap's `judge` (mlx_vlm.server) takes the big tier's place (fable or opus) while it
runs, and ultron_admit holds each request until that tier is idle (see Vision/judge/README.md).

Images over --max-edge px are shrunk with macOS `sips` first (the processor would otherwise feed a
4K image as ~16k tokens; without sips they go as they are). Base URL: --base, else $ULTRON_BASE, else
http://127.0.0.1:4000 (LiteLLM on this Mac). Key: $LITELLM_KEY, else LITELLM_MASTER_KEY from ~/.litellm/env.
Stdlib only (macOS's /usr/bin/python3 is 3.9).
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

MODEL = "ultron/judge"

# SkyJM-RM's GEN_TEMPLATE, verbatim (skylenage-ai/SkyJM-RM judges/skyjm_rm/templates.py).
TEMPLATE = """\
# Role
You are an expert evaluator for text-to-image (T2I) generation, specializing in multi-dimensional image assessment. You are skilled at identifying the most relevant evaluation dimensions, assigning appropriate weights, and delivering precise, logically grounded comparisons.

# Workflow
1. Intent Mining: Analyze the prompt to identify its core objective, including required content, key attributes, stylistic constraints, compositional requirements, and any other critical conditions, while also taking into account general image evaluation principles.
2. Dimension Selection & Weighting: Select 3–5 evaluation dimensions that are most relevant to the task. All dimensions must be atomic: each dimension should assess exactly one distinct aspect and must not combine multiple criteria into a single dimension. Assign weights dynamically based on their importance. The total weight must sum to 100%.
3. Dimension-based Scoring: Evaluate each image on every selected dimension using the following 0–4 rubric. Scores must be assigned relative to the specific dimension being evaluated.
   0: Failed: Does not satisfy the dimension; severe errors or breakdowns are present.
   1: Poor: Satisfies the dimension only weakly; major deviations, omissions, or artifacts are present.
   2: Fair: Partially satisfies the dimension; the intended quality is present, but notable issues remain.
   3: Good: Satisfies the dimension well; only minor flaws are present.
   4: Excellent: Fully satisfies the dimension; highly consistent and essentially free of noticeable flaws.

# Output Format
## [Thinking Process]
- Task Analysis: Systematically analyze the prompt.
- Selected Dimensions & Weights: List the chosen dimensions and explain why each one is important, including the rationale for its assigned weight.

## [Detailed Evaluation]
- [Dimension Name] ([Weight]%)
  - Image A: [brief analysis] → Score: X/4
  - Image B: [brief analysis] → Score: X/4

## [Final Conclusion]
- Weighted Total Score: For each image, show the weighted score calculation and provide the final result rounded to 2 decimal places.
- Summary: Briefly summarize the key reasons behind the evaluation.
- Preference: You **must** output exactly one answer — \\boxed{{A}} or \\boxed{{B}}. A definitive choice is required even when scores are tied.

# Prompt
{prompt}

# Images
- Image A: <image>
- Image B: <image>"""

MAGIC = ((b"\x89PNG", "png"), (b"\xff\xd8\xff", "jpeg"), (b"RIFF", "webp"), (b"GIF8", "gif"))


def key() -> str:
    k = os.environ.get("LITELLM_KEY")
    if k:
        return k
    try:
        for line in open(os.path.expanduser("~/.litellm/env")):
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    except OSError:
        pass
    sys.exit("no key: set LITELLM_KEY (or run on the Mac, where ~/.litellm/env has LITELLM_MASTER_KEY)")


def load(path: str, max_edge: int, tmp: str) -> str:
    """data: URL for the image, shrunk to max_edge on its longest side when sips is around."""
    if max_edge and shutil.which("sips"):
        out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", path], capture_output=True, text=True).stdout
        dims = [int(n) for n in re.findall(r"pixel(?:Width|Height): (\d+)", out)]
        if dims and max(dims) > max_edge:
            small = os.path.join(tmp, f"{len(os.listdir(tmp))}-{os.path.basename(path)}")
            subprocess.run(["sips", "-Z", str(max_edge), path, "--out", small], capture_output=True, check=True)
            path = small
    raw = open(path, "rb").read()
    kind = next((k for m, k in MAGIC if raw.startswith(m)), None)
    if not kind:
        sys.exit(f"{path}: not a png/jpeg/webp/gif")
    return f"data:image/{kind};base64," + base64.b64encode(raw).decode()


def content(prompt: str, a: str, b: str) -> list[dict]:
    """The template with Image A / Image B at its two <image> markers, as chat content parts."""
    parts, imgs = [], iter((a, b))
    for seg in re.split(r"(<image>)", TEMPLATE.format(prompt=prompt)):
        if seg == "<image>":
            parts.append({"type": "image_url", "image_url": {"url": next(imgs)}})
        elif seg:
            parts.append({"type": "text", "text": seg})
    return parts


def judge(base: str, k: str, prompt: str, a: str, b: str, max_tokens: int, tries: int = 3) -> tuple[str | None, str, float]:
    """('A' | 'B' | None, judge's text, seconds). Retries a dropped or 5xx call (the judge can be
    unloaded mid-request when a tier needs its memory back)."""
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content(prompt, a, b)}]}).encode()
    for attempt in range(tries):
        t0 = time.time()
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", body,
                                     {"Authorization": "Bearer " + k, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:  # admission may hold it up to ~10 min
                text = json.load(r)["choices"][0]["message"].get("content") or ""
            picks = re.findall(r"\\boxed\{\s*\\?(?:text\{)?([AB])\}?\s*\}", text)
            return (picks[-1] if picks else None), text, time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries - 1:
                sys.exit(f"judge call failed: {e.code} {e.read()[:300].decode('utf-8', 'replace')}")
        except (urllib.error.URLError, OSError) as e:
            if attempt == tries - 1:
                sys.exit(f"judge call failed: {e}")
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def conclusion(text: str) -> str:
    i = text.find("[Final Conclusion]")
    return (text[i:] if i >= 0 else text[-600:]).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prompt", help="the prompt the images were generated from")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--base", default=os.environ.get("ULTRON_BASE", "http://127.0.0.1:4000"), help="LiteLLM base URL (default: $ULTRON_BASE, else http://127.0.0.1:4000)")
    ap.add_argument("--max-edge", type=int, default=1024, help="shrink longer images to this many px (0: never)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("-v", "--verbose", action="store_true", help="print each verdict's conclusion")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if len(args.images) < 2:
        ap.error("need at least two images")
    names = args.images
    if len(set(names)) != len(names):
        ap.error("an image is listed twice")
    k = key()
    tmp = tempfile.mkdtemp(prefix="rank-")
    try:
        urls = {n: load(n, args.max_edge, tmp) for n in names}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    calls = len(names) * (len(names) - 1)
    if not args.json:
        print(f"{len(names)} images, {calls} judge calls", file=sys.stderr)
    score = {n: {"wins": 0, "ties": 0, "losses": 0} for n in names}
    pairs = []
    for x, y in itertools.combinations(names, 2):
        p1, t1, s1 = judge(args.base, k, args.prompt, urls[x], urls[y], args.max_tokens)  # x is A
        p2, t2, s2 = judge(args.base, k, args.prompt, urls[y], urls[x], args.max_tokens)  # y is A
        first = {"A": x, "B": y}.get(p1 or "")
        second = {"A": y, "B": x}.get(p2 or "")
        winner = first if first and first == second else None
        if winner:
            loser = y if winner == x else x
            score[winner]["wins"] += 1
            score[loser]["losses"] += 1
        else:
            score[x]["ties"] += 1
            score[y]["ties"] += 1
        pairs.append({"a": x, "b": y, "winner": winner, "orders": [first, second], "seconds": [round(s1, 1), round(s2, 1)],
                      "conclusions": [conclusion(t1), conclusion(t2)]})
        if not args.json:
            verdict = f"{winner} wins" if winner else f"tie (orders picked {first or '?'} / {second or '?'})"
            print(f"  {x}  vs  {y}:  {verdict}   [{s1:.0f}s + {s2:.0f}s]", file=sys.stderr)
            if args.verbose:
                for c in pairs[-1]["conclusions"]:
                    print("    " + c.replace("\n", "\n    "), file=sys.stderr)
    ranked = sorted(names, key=lambda n: (-(score[n]["wins"] + score[n]["ties"] / 2), names.index(n)))
    if args.json:
        print(json.dumps({"prompt": args.prompt, "ranking": [{"image": n, "score": score[n]["wins"] + score[n]["ties"] / 2,
                          **score[n]} for n in ranked], "pairs": pairs}, indent=2))
    else:
        for i, n in enumerate(ranked, 1):
            s = score[n]
            print(f"{i}. {n}   score {s['wins'] + s['ties'] / 2:g}  ({s['wins']}W {s['ties']}T {s['losses']}L)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

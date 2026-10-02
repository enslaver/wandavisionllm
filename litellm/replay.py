"""Replay saved agent transcripts through the detector, as if each request went via ultron.

    python3 replay.py FILE_OR_DIR [...]     pi (*.jsonl) and Claude Code (*.jsonl) sessions

After every assistant tool step (once its results are in) the detector sees the history the
client would have sent next. Compactions reset the history, as they do for the real proxy.
Prints one line per run that reached warn or higher, then totals. A healthy corpus should
show no force/stop; known loops should reach stop.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import loop_breaker as lb


def _pi_text(content):
    return lb._content_text([{"type": "text", "text": c.get("text", "")} if c.get("type") == "text" else c
                             for c in content] if isinstance(content, list) else content)


def iter_events(path):
    """Yield ('step', Step) when a step's results are complete, ('reset', None) on compaction."""
    pending: dict[str, lb.Step] = {}
    cc_steps: dict[str, lb.Step] = {}  # Claude Code splits parallel tool_use over lines per message id
    for line in open(path, errors="ignore"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        t = o.get("type")
        if t == "compaction" or o.get("isCompactSummary") or t == "summary":
            pending.clear()
            cc_steps.clear()
            yield "reset", None
            continue
        m = o.get("message") if isinstance(o.get("message"), dict) else None
        if m is None:
            continue
        role = m.get("role")
        content = m.get("content")
        if t == "message" and role == "assistant":  # pi
            calls = [c for c in content or [] if isinstance(c, dict) and c.get("type") == "toolCall"]
            if calls:
                s = lb.Step([(c.get("id"), c.get("name", "?"), lb.canonical_args(c.get("name", "?"), c.get("arguments", {})))
                                   for c in calls])
                for c in calls:
                    pending[c.get("id")] = s
        elif t == "message" and role == "toolResult":  # pi
            s = pending.pop(m.get("toolCallId"), None)
            if s is not None:
                s.results[m.get("toolCallId")] = (_pi_text(m.get("content")), bool(m.get("isError")))
                if s.complete():
                    yield "step", s
        elif t == "assistant" and isinstance(content, list):  # Claude Code
            uses = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
            if uses:
                mid = m.get("id") or o.get("uuid")
                s = cc_steps.setdefault(mid, lb.Step())
                for c in uses:
                    s.calls.append((c.get("id"), c.get("name", "?"), lb.canonical_args(c.get("name", "?"), c.get("input", {}))))
                    pending[c.get("id")] = s
        elif t == "user" and isinstance(content, list):  # Claude Code
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    s = pending.pop(c.get("tool_use_id"), None)
                    if s is not None:
                        s.results[c.get("tool_use_id")] = (lb._content_text(c.get("content")), bool(c.get("is_error")))
                        if s.complete():
                            yield "step", s


def replay(path):
    steps: list[lb.Step] = []
    runs = []  # (peak level, rule, run, tools, first step index)
    cur = None
    for kind, s in iter_events(path):
        if kind == "reset":
            steps = []
            cur = None
            continue
        # Claude Code interleaves parallel tool_use lines with their results, so the same step
        # can complete more than once as it grows; a live request always carries it whole.
        s._rkey = None
        if not steps or steps[-1] is not s:
            steps.append(s)
        if len(steps) > 2 * lb.MAX_STEPS:
            steps = steps[-lb.MAX_STEPS:]
        v = lb.detect(steps)
        if v.level == "none":
            cur = None
            continue
        if cur is None or cur["rule"] != v.rule or v.run < cur["run"]:
            cur = {"level": v.level, "rule": v.rule, "run": v.run, "tools": v.tools, "preview": v.result_preview[:90]}
            runs.append(cur)
        else:
            cur["run"] = v.run
            if lb.LEVELS.index(v.level) > lb.LEVELS.index(cur["level"]):
                cur["level"] = v.level
    return runs


def main(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                files += [os.path.join(root, n) for n in names if n.endswith(".jsonl")]
        else:
            files.append(a)
    total = collections.Counter()
    for f in sorted(files):
        try:
            runs = replay(f)
        except Exception as exc:  # a bad transcript shouldn't stop the survey
            print(f"ERR {f}: {exc!r}", file=sys.stderr)
            continue
        for r in runs:
            total[r["level"]] += 1
            print(f"{r['level']:5} {r['rule']:8} run={r['run']:<5} {','.join(dict.fromkeys(r['tools']))[:60]:60} "
                  f"{os.path.basename(f)[:40]}  | {r['preview']}")
    print(f"files={len(files)} runs_reaching: warn={total['warn']} force={total['force']} stop={total['stop']}")


if __name__ == "__main__":
    main(sys.argv[1:])

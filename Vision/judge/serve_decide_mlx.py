#!/usr/bin/env python3
"""JEV-27B-VL decision server on MLX: POST /v1/decide for a llama-swap model or a standalone server.

A port of the model's own serve_decide.py (vLLM) to mlx_vlm. Same request, response and protocol (jev27-bare-v1),
so a caller of the hosted JEV API or of vLLM's serve_decide works unchanged:
  System 1  the decision LoRA (backbone LoRA + decision head as an lm_head LoRA): one forward pass, the next-token
            logits at the option verbalizers + the head's bias, softmax at the per-kind temperature (calibration.json)
  System 2  the base model (LoRA off) thinks in its chat template; when </think> comes, the "Answer: (X)" letter
            distribution is read in one step and folded in: p = (1 - w) p1 + w p2, w = 0.5 (thinking "auto" / "on")

The base is 4-bit MLX; the adapter stays bf16 beside each quantized linear (y = Wx + 2 * B(Ax), alpha/r = 64/32),
on for System 1 and off for System 2, so one copy of the weights serves both.

Differs from the vLLM server: one request at a time (one worker thread), prompts capped at --max-context tokens and
--max-images images, images only from http(s) or data:image URLs, System 2 reads its answer from the tokens it
generated (vLLM re-tokenizes the decoded text), and tournament / permute passes run in turn.

POST /v1/decide   {"kind": "noul"|"score"|"choice", "state": str|dict|[str, {"image": url}, ...], "question": str,
                   "options": [...] (choice), "strategy", "thinking", "threshold", "think_budget", "reasoning_effort",
                   "chat_template_kwargs", "return_reasoning", "debug", "system2_only"}   (see serve_decide.py)
GET  /v1/decide/info, /health

Run with mlx-vlm's python (fastapi, uvicorn, mlx_vlm):
  python serve_decide_mlx.py --model ~/models/LLM/JEV-27B-VL-mlx-4bit --port 8003
The adapter is read from <model>/adapter_vllm (decision_head.json beside it), calibration from <model>/calibration.json.
"""
import argparse
import json
import math
import os
import random
import string
import sys
import time
import zlib

MAX_OPTIONS = 256
PROTOCOL = "jev27-bare-v1"
PLACEHOLDER = "XQXCONTENTXQX"
PREFILL_STEP = 2048
# The model card's Qwen profile (serve_decide.PROFILES["qwen"]).
PROFILE = {"prefix": "", "image": "<|vision_start|><|image_pad|><|vision_end|>", "end_think": "</think>",
           "after_think": "\n\n", "strategy": "single", "threshold": 0.8, "mix": 0.5, "thinking": "off"}
LABELS = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


class BadRequest(Exception):
    pass


# ------------------------------------------------------------------ pure helpers (no MLX: unit-tested)
def lora_target(key):
    """PEFT adapter key -> (mlx_vlm module path, "a" | "b"), or None for a key that isn't a LoRA matrix."""
    if not key.startswith("base_model.model.") or not key.endswith(".weight"):
        return None
    path, _, which = key[len("base_model.model."):-len(".weight")].rpartition(".")
    if which not in ("lora_A", "lora_B"):
        return None
    if path.startswith("model.language_model."):      # mlx_vlm qwen3_5 sanitize_key
        path = "language_model.model." + path[len("model.language_model."):]
    elif path == "lm_head":
        path = "language_model.lm_head"
    else:
        raise ValueError(f"adapter key outside the language model: {key}")
    return path, which[-1].lower()


def parts_of(state):
    """serve_decide._parts: state -> (chat content parts, has image)."""
    if isinstance(state, str):
        return [{"type": "text", "text": state}], False
    if isinstance(state, dict):
        return [{"type": "text", "text": json.dumps(state, ensure_ascii=False)}], False
    out, img = [], False
    for p in state:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
        elif isinstance(p, dict) and "image" in p:
            out.append({"type": "image_url", "image_url": {"url": p["image"]}}); img = True
        elif isinstance(p, dict) and p.get("type") == "image_url":
            out.append(p); img = True
        elif isinstance(p, dict) and p.get("type") == "text":
            out.append(p)
        else:
            out.append({"type": "text", "text": json.dumps(p, ensure_ascii=False)})
    return out, img


def render(content, image_token):
    """The raw template of serve_decide: text parts verbatim, each image as the vision placeholder. -> (text, urls)"""
    text, urls = [], []
    for c in content:
        if c.get("type") == "text":
            text.append(c["text"])
        else:
            url = (c.get("image_url") or {}).get("url") if isinstance(c.get("image_url"), dict) else c.get("image_url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://", "data:image/")):
                raise BadRequest("images must be http(s) or data:image URLs")
            text.append(image_token); urls.append(url)
    return "".join(text), urls


def softmax(z):
    m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return [x / s for x in e]


def groups_of(n, k=16):
    g = math.ceil(n / k); base, extra = divmod(n, g); out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0); out.append(list(range(i, i + size))); i += size
    return out


def fold(p1, p2, w):
    return [(1 - w) * x + w * y for x, y in zip(p1, p2)]


class Ctx:
    def __init__(self):
        self.requests, self.prompt_tokens, self.completion_tokens, self.images = 0, 0, 0, {}


# ------------------------------------------------------------------ System 1 / System 2 (engine-agnostic)
def s1_pass(eng, ctx, kind, parts, question, opts):
    """One System 1 pass over `opts` (<=256). Returns probabilities aligned with opts."""
    head, temps = eng.head, eng.temps
    lo, hi = head["slots"]["ranges"][kind]
    if kind == "choice":
        n = len(opts); ids = eng.label_ids[:n]
        bias = [head["bias"][lo + i] if lo + i < hi else 0.0 for i in range(n)]
        lines = [f"{eng.labels[i]}) {o}" for i, o in enumerate(opts)]
    else:
        ids = head["verbalizer_ids"][lo:hi]; bias = head["bias"][lo:hi]; lines = opts
    content = ([{"type": "text", "text": f"{PROFILE['prefix']}[kind] {kind}\n[state] "}] + parts +
               [{"type": "text", "text": f"\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"}])
    lp = eng.readout(ctx, content, ids)
    return softmax([(z + b) / temps[kind] for z, b in zip(lp, bias)])


def s1_dist(eng, ctx, kind, parts, question, opts, strategy):
    if kind != "choice" or len(opts) <= 16 or strategy == "single":
        return s1_pass(eng, ctx, kind, parts, question, opts)
    n = len(opts)
    if strategy == "permute":
        rng = random.Random(zlib.crc32(question.encode()))
        orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(3)]
        p = [0.0] * n
        for o in orders:
            for i, v in zip(o, s1_pass(eng, ctx, kind, parts, question, [opts[i] for i in o])):
                p[i] += v / len(orders)
        return p
    groups = groups_of(n)  # tournament: groups of <=16 in the given order, then a final of 16
    in_group = {}
    for g in groups:
        in_group.update(zip(g, s1_pass(eng, ctx, kind, parts, question, [opts[i] for i in g])))
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted((o for g in groups for o in g if o not in set(chosen)), key=lambda o: (-in_group[o], o))
    fin = sorted(chosen + rest[: max(0, 16 - len(chosen))])
    final = dict(zip(fin, s1_pass(eng, ctx, kind, parts, question, [opts[i] for i in fin])))
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share = [0.0] * len(groups); cap = [0.0] * len(groups)
    for f in fin:
        share[group_of[f]] += final[f]; cap[group_of[f]] += in_group[f]
    among = sum(a * b for a, b in zip(share, cap))
    p = [final[o] * among if o in final else share[group_of[o]] * in_group[o] for o in range(n)]
    s = sum(p)
    return [x / s for x in p]


def s2_dist(eng, ctx, kind, parts, question, opts, budget, want_text, tmpl_kwargs=None):
    """The base model thinks over the same input; the answer-letter distribution is read after the thinking channel."""
    shown = ["Yes (true)", "No (false)"] if kind == "noul" else opts
    labs = eng.ans_labels[: len(shown)]
    body = "\n".join(f"({l}) {o}" for (l, _), o in zip(labs, shown))
    tail = (f"\n\nQuestion: {question}\n\nOptions:\n{body}\n\n"
            "Think it through carefully, then give your final answer on the last line in the form: Answer: (X)")
    pre, post = eng.think_frame(tmpl_kwargs)
    user = [{"type": "text", "text": pre}] + parts + [{"type": "text", "text": tail + post}]
    seed = zlib.crc32((question + body).encode()) & 0x7FFFFFFF
    t0 = time.time()
    thought, finished, ntok, lp = eng.think(ctx, user, budget, seed, PROFILE["end_think"] + PROFILE["after_think"] + "Answer: (",
                                            [t for _, t in labs])
    p = softmax(lp)
    if kind == "noul":
        p = [p[1], p[0]]  # (A) yes / (B) no -> [P(false), P(true)]
    info = {"think_tokens": ntok, "think_seconds": round(time.time() - t0, 3), "finished_within_budget": finished}
    if want_text:
        info["reasoning"] = thought.strip()
    return p, info


def decide(eng, req):
    """POST /v1/decide body (validated dict) -> (status, response). serve_decide.decide without the vLLM plumbing."""
    t0 = time.time()
    kind = req["kind"]
    if kind == "choice":
        opts = req.get("options") or []
        if not 2 <= len(opts) <= len(eng.labels):
            return 400, err(f"choice needs 2-{len(eng.labels)} options, got {len(opts)}")
    else:
        opts = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
    req_thinking = req.get("thinking") or "default"
    thinking = PROFILE["thinking"] if req_thinking == "default" else req_thinking
    if kind == "score":
        if req_thinking in ("auto", "on"):
            return 400, err("thinking is supported for noul and choice")
        thinking = "off"
    tmpl = dict(req.get("chat_template_kwargs") or {})
    if req.get("reasoning_effort") is not None:
        if not eng.has_effort:
            return 400, err("this base model has no reasoning_effort setting; control the reasoning length with think_budget")
        tmpl["reasoning_effort"] = req["reasoning_effort"]
    strategy = PROFILE["strategy"] if (req.get("strategy") or "auto") == "auto" else req["strategy"]
    tau = PROFILE["threshold"] if req.get("threshold") is None else req["threshold"]
    parts, _ = parts_of(req.get("state", ""))
    question, budget = req["question"], req.get("think_budget")
    ctx = Ctx()
    try:
        if req.get("system2_only"):
            if kind == "score":
                return 400, err("system2_only is supported for noul and choice")
            p2, think = s2_dist(eng, ctx, kind, parts, question, opts, budget, req.get("return_reasoning"), tmpl)
            k = max(range(len(p2)), key=p2.__getitem__)
            return 200, {"kind": kind, "options": opts, "probabilities": p2, "choice_index": k, "choice": opts[k], "system": 2,
                         "model": eng.name, "thinking": {"used": True, "budget": budget, **think},
                         "usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens},
                         "elapsed_seconds": time.time() - t0, "num_model_requests": ctx.requests}
        p1 = s1_dist(eng, ctx, kind, parts, question, opts, strategy)
        probs, think = p1, None
        if thinking == "on" or (thinking == "auto" and max(p1) < tau):
            p2, think = s2_dist(eng, ctx, kind, parts, question, opts, budget, req.get("return_reasoning"), tmpl)
            probs = fold(p1, p2, PROFILE["mix"])
            think = {"used": True, **think}
            if req.get("debug"):
                think.update(system1=p1, system2=p2)
        elif thinking != "off":
            think = {"used": False}
            if req.get("debug"):
                think["system1"] = p1
    except BadRequest as e:
        return 400, err(str(e))
    k = max(range(len(probs)), key=probs.__getitem__)
    native = kind != "choice" or len(opts) <= 16
    resp = {"kind": kind, "effective_kind": kind, "options": opts, "probabilities": probs, "choice_index": k,
            "choice": opts[k], "adaptation": "native" if native else f"{strategy}", "protocol": PROTOCOL, "model": eng.name,
            "usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
                      "total_tokens": ctx.prompt_tokens + ctx.completion_tokens},
            "elapsed_seconds": time.time() - t0, "num_model_requests": ctx.requests}
    if think is not None:
        resp["thinking"] = {"mode": thinking, "threshold": tau, "budget": budget, **({"chat_template_kwargs": tmpl} if tmpl else {}), **think}
    return 200, resp


def err(msg, code=400):
    return {"error": {"message": msg, "type": "BadRequestError", "code": code}}


def info(eng):
    return {"protocol": PROTOCOL, "model": eng.name, "model_type": eng.model_type, "max_options": len(eng.labels),
            "native_choice_options": len(eng.head["slots"]["verbalizers"]) - 8, "temperatures": eng.temps,
            "defaults": {"strategy": PROFILE["strategy"], "thinking": PROFILE["thinking"], "threshold": PROFILE["threshold"],
                         "mix": PROFILE["mix"], "think_budget": None},
            "reasoning_controls": {"enable_thinking": True, "reasoning_effort": ["xhigh", "medium", "low"] if eng.has_effort else None,
                                   "think_budget": "max thinking tokens (default: up to the context cap)"},
            "limits": {"max_context": eng.max_context, "max_images": eng.max_images}}


# ------------------------------------------------------------------ MLX engine
class Engine:
    def __init__(self, model_dir, name, max_context, max_images, adapter_dir=None, calibration=None):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten
        from mlx_vlm import load
        from mlx_vlm.utils import load_image

        self.mx, self.load_image = mx, load_image
        self.name, self.max_context, self.max_images = name, max_context, max_images
        self.model, self.processor = load(model_dir)
        self.tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.lm = self.model.language_model
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.model_type = cfg.get("model_type")
        gen = {}
        if os.path.exists(os.path.join(model_dir, "generation_config.json")):
            gen = json.load(open(os.path.join(model_dir, "generation_config.json")))
        self.sampling = (float(gen.get("temperature", 1.0)), int(gen.get("top_k", 0) or 0), float(gen.get("top_p", 1.0)))

        adapter_dir = adapter_dir or os.path.join(model_dir, "adapter_vllm")
        self.head = json.load(open(os.path.join(adapter_dir, "decision_head.json")))
        calibration = calibration or os.path.join(model_dir, "calibration.json")
        self.temps = json.load(open(calibration))["per_kind"] if os.path.exists(calibration) else {"noul": 1.0, "score": 1.0, "choice": 1.0}
        self.temps.setdefault("score", 1.0)

        # LoRA beside the quantized linears. A plain Module (not nn.QuantizedLinear / nn.Linear) keeps mlx_vlm's fused
        # decode and target-verify paths off these layers, so every call goes through __call__.
        acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        if acfg.get("use_rslora") or acfg.get("use_dora"):
            raise SystemExit("serve_decide_mlx: rslora / dora adapters aren't supported")
        scale = float(acfg["lora_alpha"]) / float(acfg["r"])

        class LoRA(nn.Module):
            on = True

            def __init__(self, base, a, b):
                super().__init__()
                self.base, self.lora_a, self.lora_b = base, a.T, (b * scale).T   # (in, r), (r, out)

            def __call__(self, x):
                y = self.base(x)
                if LoRA.on:
                    y = y + ((x.astype(self.lora_a.dtype) @ self.lora_a) @ self.lora_b).astype(y.dtype)
                return y

        self.LoRA = LoRA
        weights = mx.load(os.path.join(adapter_dir, "adapter_model.safetensors"))
        pairs = {}
        for k, v in weights.items():
            t = lora_target(k)
            if t:
                pairs.setdefault(t[0], {})[t[1]] = v
        mods = dict(self.model.named_modules())
        new = []
        for path, ab in sorted(pairs.items()):
            base = mods.get(path)
            if base is None or set(ab) != {"a", "b"}:
                raise SystemExit(f"serve_decide_mlx: adapter target {path} not found in the model")
            out_dim = base.weight.shape[0]
            if ab["b"].shape[0] != out_dim:
                raise SystemExit(f"serve_decide_mlx: {path}: lora_B rows {ab['b'].shape[0]} != {out_dim}")
            new.append((path, LoRA(base, ab["a"], ab["b"])))
        self.model.update_modules(tree_unflatten(new))
        mx.eval(self.model.parameters())
        self.lora_layers = len(new)

        # Labels (serve_decide.setup): System 1 option-line labels, System 2 answer labels.
        def single_token_labels(context):
            out = []
            for lab in LABELS:
                t = self.tok.encode(lab, add_special_tokens=False)
                if len(t) == 1 and t[0] in self.tok.encode(context.format(lab), add_special_tokens=False):
                    out.append((lab, t[0]))
                if len(out) == MAX_OPTIONS:
                    break
            return out

        dec = single_token_labels("x\n{}) y")
        lo, hi = self.head["slots"]["ranges"]["choice"]
        if [t for _, t in dec[: hi - lo]] != self.head["verbalizer_ids"][lo:hi]:
            raise SystemExit("serve_decide_mlx: first labels must be the trained A-P head")
        self.labels, self.label_ids = [l for l, _ in dec], [t for _, t in dec]
        base_ids = self.tok.encode("Answer: (", add_special_tokens=False)
        self.ans_labels = []
        for lab in LABELS:
            ids = self.tok.encode(f"Answer: ({lab})", add_special_tokens=False)
            if ids[: len(base_ids)] == base_ids and len(ids) == len(base_ids) + 2:
                self.ans_labels.append((lab, ids[len(base_ids)]))
            if len(self.ans_labels) == MAX_OPTIONS:
                break
        self.end_think_id = self.tok.convert_tokens_to_ids(PROFILE["end_think"])
        self.has_effort = "reasoning_effort" in (getattr(self.tok, "chat_template", None) or "")
        self.frames = {}

    def think_frame(self, kwargs):
        key = json.dumps(kwargs or {}, sort_keys=True)
        if key not in self.frames:
            kw = {k: v for k, v in (kwargs or {}).items() if k != "enable_thinking"}
            chat = self.tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True, **kw)
            self.frames[key] = tuple(chat.split(PLACEHOLDER))
        return self.frames[key]

    # -- model passes
    def encode(self, ctx, content):
        mx = self.mx
        text, urls = render(content, PROFILE["image"])
        if len(urls) > self.max_images:
            raise BadRequest(f"{len(urls)} images; this judge takes up to {self.max_images}")
        if not urls:
            ids = self.tok.encode(text, add_special_tokens=False)
            inputs = {"input_ids": mx.array([ids])}
        else:
            imgs = []
            for u in urls:
                if u not in ctx.images:
                    try:
                        ctx.images[u] = self.load_image(u)
                    except Exception as e:
                        raise BadRequest(f"image: {e}")
                imgs.append(ctx.images[u])
            enc = self.processor(text=[text], images=imgs, padding=False, return_tensors="np")
            inputs = {"input_ids": mx.array(enc["input_ids"]), "pixel_values": mx.array(enc["pixel_values"]),
                      "image_grid_thw": mx.array(enc["image_grid_thw"])}
        n = inputs["input_ids"].shape[1]
        if n > self.max_context:
            raise BadRequest(f"the prompt is {n} tokens; this judge takes up to {self.max_context}")
        return inputs

    def prefill(self, inputs, lora):
        """Chunked prefill into a fresh cache. -> (cache, last hidden state, rope delta, prompt tokens)"""
        mx = self.mx
        self.LoRA.on = lora
        ids = inputs["input_ids"]
        emb = self.model.get_input_embeddings(ids, inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw"))
        x, pos, rd = emb.inputs_embeds, emb.position_ids, emb.rope_deltas
        mx.eval([a for a in (x, pos, rd) if a is not None])
        self.lm._position_ids = self.lm._rope_deltas = None
        cache = self.lm.make_cache()
        n, h = ids.shape[1], None
        for s in range(0, n, PREFILL_STEP):
            e = min(n, s + PREFILL_STEP)
            out = self.lm(ids[:, s:e], inputs_embeds=x[:, s:e], cache=cache, position_ids=pos, rope_deltas=rd,
                          skip_logits=True, return_hidden=True)
            h = out.hidden_states[-1][:, -1:, :]
            mx.eval(h, [c.state for c in cache])
        return cache, h, rd, n

    def step(self, tokens, cache, rd):
        """Feed `tokens` after the cache. -> last hidden state"""
        mx = self.mx
        off = cache[self.lm.model.fa_idx].offset
        pos = mx.arange(off, off + len(tokens)).reshape(1, -1) + rd.reshape(-1, 1)
        pos = mx.broadcast_to(pos[None], (3, 1, len(tokens)))
        out = self.lm(mx.array([tokens]), cache=cache, position_ids=pos, rope_deltas=rd, skip_logits=True, return_hidden=True)
        return out.hidden_states[-1][:, -1:, :]

    def logits(self, h):
        return self.lm.lm_head(h)[0, -1].astype(self.mx.float32)

    def readout(self, ctx, content, ids):
        """System 1: one pass with the LoRA on, the logits at `ids` after the content."""
        inputs = self.encode(ctx, content)
        _, h, _, n = self.prefill(inputs, lora=True)
        z = self.logits(h)[self.mx.array(ids)].tolist()
        ctx.requests += 1; ctx.prompt_tokens += n; ctx.completion_tokens += 1   # as vLLM counts its 1-token read-out
        return z

    def sample(self, h):
        mx = self.mx
        temp, top_k, top_p = self.sampling
        z = self.logits(h)
        if temp <= 0:
            return int(mx.argmax(z).item())
        z = z / temp
        idx = None
        if 0 < top_k < z.shape[0]:
            idx = mx.argpartition(-z, top_k - 1)[:top_k]
            z = z[idx]
        if top_p < 1.0:
            order = mx.argsort(-z)
            sz = z[order]
            pr = mx.softmax(sz)
            keep = (mx.cumsum(pr) - pr) < top_p
            k = order[mx.random.categorical(mx.where(keep, sz, -mx.inf))]
        else:
            k = mx.random.categorical(z)
        return int((idx[k] if idx is not None else k).item())

    def think(self, ctx, content, budget, seed, tail, ids):
        """System 2: the base model (LoRA off) thinks until </think> or the budget, then the logits at `ids` after
        `tail`. -> (thought text, finished, completion tokens, logits)"""
        mx = self.mx
        inputs = self.encode(ctx, content)
        cache, h, rd, n = self.prefill(inputs, lora=False)
        tail_ids = self.tok.encode(tail, add_special_tokens=False)
        room = self.max_context - n - len(tail_ids)
        if room <= 0:
            raise BadRequest(f"the thinking prompt is {n} tokens; no room left under {self.max_context}")
        limit = room if budget is None else max(0, min(int(budget), room))
        mx.random.seed(seed)
        out, finished = [], False
        y = self.sample(h)
        while len(out) < limit:
            if y == self.end_think_id:
                finished = True
                break
            out.append(y)
            h = self.step([y], cache, rd)
            y = self.sample(h)
        h = self.step(tail_ids, cache, rd)
        z = self.logits(h)[mx.array(ids)].tolist()
        ctx.requests += 2
        ctx.prompt_tokens += n + n + len(out) + len(tail_ids)   # as vLLM counts: the generation, then the read-out
        ctx.completion_tokens += len(out) + int(finished) + 1
        return self.tok.decode(out, skip_special_tokens=False), finished, len(out) + int(finished), z


# ------------------------------------------------------------------ HTTP
def serve(args):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from typing import Literal

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    pool = ThreadPoolExecutor(max_workers=1)   # MLX work stays on one thread: one request at a time
    t0 = time.time()
    eng = pool.submit(Engine, os.path.expanduser(args.model), args.served_model_name, args.max_context, args.max_images,
                      args.adapter and os.path.expanduser(args.adapter), args.calibration and os.path.expanduser(args.calibration)).result()
    print(f"serve_decide_mlx: {eng.name} loaded in {time.time() - t0:.1f}s, {eng.lora_layers} LoRA layers, "
          f"temps {eng.temps}, max_context {eng.max_context}, max_images {eng.max_images}", file=sys.stderr, flush=True)

    class DecideRequest(BaseModel):
        kind: Literal["noul", "score", "choice"]
        state: str | dict | list = ""
        question: str
        options: list[str] | None = None
        strategy: Literal["auto", "single", "tournament", "permute"] = "auto"
        thinking: Literal["default", "off", "auto", "on"] = "default"
        threshold: float | None = None
        think_budget: int | None = None
        reasoning_effort: str | None = None
        chat_template_kwargs: dict | None = None
        return_reasoning: bool = False
        debug: bool = False
        system2_only: bool = False

    def run(body):
        try:
            return decide(eng, body)
        finally:
            eng.mx.clear_cache()   # hand Metal buffers back: the box is shared with the tiers

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": eng.name}

    @app.get("/v1/decide/info")
    async def decide_info():
        return info(eng)

    @app.post("/v1/decide")
    async def decide_route(req: DecideRequest):
        t = time.time()
        body = req.model_dump()
        status, resp = await asyncio.get_running_loop().run_in_executor(pool, run, body)
        print(f"serve_decide_mlx: {body['kind']} n={len(resp.get('options') or [])} -> {status} "
              f"{resp.get('choice', resp.get('error', {}).get('message'))!r} "
              f"tokens={resp.get('usage', {}).get('prompt_tokens')}/{resp.get('usage', {}).get('completion_tokens')} "
              f"{time.time() - t:.2f}s", file=sys.stderr, flush=True)
        return JSONResponse(resp, status_code=status)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="MLX model dir (adapter_vllm/ and calibration.json inside)")
    ap.add_argument("--adapter", help="adapter dir (default <model>/adapter_vllm)")
    ap.add_argument("--calibration", help="calibration.json (default <model>/calibration.json)")
    ap.add_argument("--served-model-name", default="JEV-27B-VL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--max-context", type=int, default=32768, help="prompt + thinking tokens per pass")
    ap.add_argument("--max-images", type=int, default=8)
    serve(ap.parse_args())


if __name__ == "__main__":
    main()

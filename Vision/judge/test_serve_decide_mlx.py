"""serve_decide_mlx without MLX: adapter key mapping, prompt rendering, strategies and the response, against a fake
engine. cd Vision/judge && uvx pytest -q"""
import math

import pytest

import serve_decide_mlx as sd


class FakeEngine:
    """Logits that favour one option by text; records every prompt it was given."""
    name, model_type, max_context, max_images, has_effort = "JEV-27B-VL", "qwen3_5", 32768, 8, True
    head = {"verbalizer_ids": [3721, 1802] + list(range(15, 21)) + list(range(32, 48)),
            "bias": [0.0] * 24, "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                                          "verbalizers": ["false", "true"] + [str(i) for i in range(6)] + list("ABCDEFGHIJKLMNOP")}}
    temps = {"noul": 1.0, "score": 1.0, "choice": 1.0}
    labels = sd.LABELS[:256]
    label_ids = list(range(32, 32 + 256))
    ans_labels = [(l, 1000 + i) for i, l in enumerate(sd.LABELS[:256])]

    def __init__(self, favour="opt7", think_answer=0):
        self.favour, self.think_answer, self.prompts = favour, think_answer, []

    def think_frame(self, kwargs):
        return ("<pre>", "<post>")

    def readout(self, ctx, content, ids):
        text, _ = sd.render(content, "<img>")
        self.prompts.append(text)
        ctx.requests += 1
        lines = text.split("[options]\n")[1].split("\n[decision]:")[0].split("\n")
        return [5.0 if l.split(") ", 1)[-1] == self.favour else 0.0 for l in lines][: len(ids)]

    def think(self, ctx, content, budget, seed, tail, ids):
        ctx.requests += 2
        return "thought", True, 3, [9.0 if i == self.think_answer else 0.0 for i in range(len(ids))]


def test_adapter_keys_map_onto_mlx_vlm_modules():
    assert sd.lora_target("base_model.model.model.language_model.layers.3.mlp.down_proj.lora_A.weight") == \
        ("language_model.model.layers.3.mlp.down_proj", "a")
    assert sd.lora_target("base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight") == \
        ("language_model.model.layers.0.linear_attn.in_proj_qkv", "b")
    assert sd.lora_target("base_model.model.lm_head.lora_B.weight") == ("language_model.lm_head", "b")
    assert sd.lora_target("base_model.model.lm_head.base_layer.weight") is None
    with pytest.raises(ValueError):
        sd.lora_target("base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight")


def test_state_parts_render_like_the_raw_template():
    parts, img = sd.parts_of(["frame:", {"image": "data:image/png;base64,AAAA"}, {"x": 1}])
    assert img
    text, urls = sd.render(parts, "<img>")
    assert text == 'frame:<img>{"x": 1}' and urls == ["data:image/png;base64,AAAA"]
    assert sd.parts_of({"a": "é"})[0][0]["text"] == '{"a": "é"}'
    for bad in ("/etc/passwd", "file:///etc/passwd"):
        with pytest.raises(sd.BadRequest):
            sd.render(sd.parts_of([{"image": bad}])[0], "<img>")


def test_system1_prompt_is_the_trained_bare_v1_layout():
    eng = FakeEngine(favour="red")
    status, r = sd.decide(eng, {"kind": "choice", "state": "s", "question": "q?", "options": ["red", "blue"]})
    assert status == 200 and r["choice"] == "red" and r["adaptation"] == "native" and r["protocol"] == "jev27-bare-v1"
    assert eng.prompts == ["[kind] choice\n[state] s\n[question] q?\n[options]\nA) red\nB) blue\n[decision]:"]
    assert math.isclose(sum(r["probabilities"]), 1.0)
    status, r = sd.decide(eng, {"kind": "noul", "state": "s", "question": "q?"})
    assert eng.prompts[-1].endswith("[options]\nfalse\ntrue\n[decision]:") and r["options"] == ["false", "true"]


def test_large_choices_run_a_tournament_and_find_the_favourite():
    opts = [f"opt{i}" for i in range(40)]
    eng = FakeEngine(favour="opt37")
    status, r = sd.decide(eng, {"kind": "choice", "question": "q", "options": opts, "strategy": "tournament"})
    assert status == 200 and r["choice"] == "opt37" and r["adaptation"] == "tournament"
    assert r["num_model_requests"] == 4    # 3 groups of <=16 + the final
    assert math.isclose(sum(r["probabilities"]), 1.0)
    status, r = sd.decide(eng, {"kind": "choice", "question": "q", "options": opts, "strategy": "permute"})
    assert r["choice"] == "opt37" and r["num_model_requests"] == 4


def test_thinking_folds_system2_and_noul_maps_yes_no_back():
    eng = FakeEngine(favour="true", think_answer=1)   # System 2 answers (B) "No (false)"
    status, r = sd.decide(eng, {"kind": "noul", "question": "q", "thinking": "on", "debug": True})
    assert status == 200 and r["thinking"]["used"] and r["num_model_requests"] == 3
    p1, p2 = r["thinking"]["system1"], r["thinking"]["system2"]
    assert p2[0] > 0.99                                # B -> P(false)
    assert r["probabilities"] == sd.fold(p1, p2, 0.5)


def test_requests_the_model_card_refuses():
    eng = FakeEngine()
    assert sd.decide(eng, {"kind": "choice", "question": "q", "options": ["one"]})[0] == 400
    assert sd.decide(eng, {"kind": "score", "question": "q", "thinking": "on"})[0] == 400
    assert sd.decide(eng, {"kind": "score", "question": "q", "system2_only": True})[0] == 400
    eng.has_effort = False
    assert sd.decide(eng, {"kind": "noul", "question": "q", "reasoning_effort": "low"})[0] == 400

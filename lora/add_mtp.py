#!/usr/bin/env python3
"""add_mtp.py MERGED_DIR BASE_DIR — copy the base checkpoint's MTP head into a fine-tuned one.

Qwen3.5 checkpoints carry a multi-token-prediction head (`mtp.*`, 15 tensors in Qwen3.5-4B) that
transformers doesn't model. A fine-tune saved through transformers (PEFT `merge_and_unload`, Unsloth
before 2026.9) keeps `mtp_num_hidden_layers: 1` in config.json but has no `mtp.*` tensors, and
`mtplx forge build` refuses it (`no_mtp_heads`). This copies them byte for byte from BASE_DIR (the
checkpoint the fine-tune started from, e.g. Qwen/Qwen3.5-4B) into MERGED_DIR/mtp.safetensors and adds
them to MERGED_DIR/model.safetensors.index.json. A merge that already has them is left alone.
The head was trained on the base's hidden states: it still works on a LoRA'd trunk, MTP acceptance
(speed) may drop a little, and the output doesn't change.
Python 3.9, standard library only, so it runs on the Mac's /usr/bin/python3 and on the GPU box.
"""
import json
import os
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors") and f != "mtp.safetensors")
    return {"metadata": {}, "weight_map": {k: f for f in files for k in header(os.path.join(d, f))[0] if k != "__metadata__"}}


def main(merged, base):
    midx = index(merged)
    have = [k for k in midx["weight_map"] if k.startswith("mtp.")]
    if have:
        print(f"{merged} already has {len(have)} mtp.* tensors; nothing to do")
        return 0
    bidx = index(base)
    want = sorted(k for k in bidx["weight_map"] if k.startswith("mtp."))
    if not want:
        sys.exit(f"{base} has no mtp.* tensors either")
    hdr, blobs, off = {}, [], 0
    for k in want:
        src = os.path.join(base, bidx["weight_map"][k])
        h, start = header(src)
        a, b = h[k]["data_offsets"]
        with open(src, "rb") as f:
            f.seek(start + a)
            blob = f.read(b - a)
        hdr[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + len(blob)]}
        blobs.append(blob)
        off += len(blob)
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # safetensors pads the header to 8 bytes
    with open(os.path.join(merged, "mtp.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for blob in blobs:
            f.write(blob)
    midx["weight_map"].update({k: "mtp.safetensors" for k in want})
    with open(os.path.join(merged, "model.safetensors.index.json"), "w") as f:
        json.dump(midx, f, indent=2)
    cfg_p = os.path.join(merged, "config.json")
    with open(cfg_p) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    if not tc.get("mtp_num_hidden_layers"):
        with open(os.path.join(base, "config.json")) as f:
            btc = json.load(f).get("text_config", {})
        tc["mtp_num_hidden_layers"] = btc.get("mtp_num_hidden_layers", 1)
        with open(cfg_p, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"added {len(want)} mtp.* tensors ({off / 2**20:.0f} MiB) to {merged}/mtp.safetensors")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

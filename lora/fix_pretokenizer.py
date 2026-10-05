#!/usr/bin/env python3
"""Restore the Qwen3.5 pre-tokenizer in an MTPLX pack's tokenizer.json.

transformers 5.x (inside `mtplx forge build`, `mlx_lm.convert`, Unsloth `save_pretrained`) re-serializes
Qwen3.5/3.6/3.8 tokenizers with the older Qwen2 split regex: `\\p{L}+` instead of `[\\p{L}\\p{M}]+`,
plus ByteLevel `trim_offsets: true`. Vocab and merges are unchanged, so English and code tokenize the
same, but every combining mark (Devanagari/Bengali/Tamil vowel signs, Thai vowels and tone marks,
Arabic/Hebrew pointing, VS16 emoji) becomes a pre-token boundary: Hindi +43%, Thai +92% tokens, and
token sequences the model never saw in training. See lora/README.md, "Pre-tokenizer regex".

Fixing tokenizer.json is not enough: transformers 5.14's `Qwen2Tokenizer` class rebuilds the Qwen2 regex
at load time whatever the file says. So tokenizer_config.json's `tokenizer_class` also becomes
`TokenizersBackend` (Qwen2Tokenizer's base class, and what Unsloth ships for Qwen3.6), which loads
tokenizer.json as written.

The right regex is still in the pack's own tokenizer_config.json (`pretokenize_regex`), so no source
checkpoint is needed; `--source DIR` takes pre_tokenizer + decoder from that tokenizer.json instead.

  fix_pretokenizer.py PACK --check            # report only (exit 1 if it needs the fix)
  fix_pretokenizer.py PACK --out NEWDIR       # NEWDIR = symlinks to PACK + fixed tokenizer files
  fix_pretokenizer.py PACK --in-place         # rewrite PACK's tokenizer files (keeps *.orig copies)

Python 3.9, standard library only (runs on the Mac's /usr/bin/python3).
"""
import argparse
import json
import os
import shutil
import sys

QWEN35_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
BYTELEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_regex(tok):
    pt = tok.get("pre_tokenizer") or {}
    for p in pt.get("pretokenizers", [pt]):
        if p.get("type") == "Split":
            return (p.get("pattern") or {}).get("Regex")
    return None


def merges(tok):
    return [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in tok["model"]["merges"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--source", help="checkpoint dir whose tokenizer.json has the right pre_tokenizer")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--out")
    g.add_argument("--in-place", action="store_true")
    a = ap.parse_args()

    tj = os.path.join(a.pack, "tokenizer.json")
    tok = load(tj)
    if a.source:
        src = load(os.path.join(a.source, "tokenizer.json"))
        if src["model"]["vocab"] != tok["model"]["vocab"] or merges(src) != merges(tok):
            sys.exit("vocab/merges differ from --source; not the same tokenizer, refusing")
        want_pt, want_dec = src["pre_tokenizer"], src.get("decoder")
    cj = os.path.join(a.pack, "tokenizer_config.json")
    cfg = load(cj)
    if not a.source:
        rx = cfg.get("pretokenize_regex") or QWEN35_REGEX
        if "\\p{M}" not in rx:
            sys.exit("tokenizer_config.json pretokenize_regex has no \\p{M}; pass --source")
        want_pt = {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": rx}, "behavior": "Isolated", "invert": False},
            dict(BYTELEVEL)]}
        want_dec = dict(BYTELEVEL)

    bad_file = tok.get("pre_tokenizer") != want_pt
    bad_class = cfg.get("tokenizer_class") == "Qwen2Tokenizer"
    broken = bad_file or bad_class
    print("%s: %s  (split regex has \\p{M}: %s; tokenizer_class: %s)" % (
        a.pack, "NEEDS FIX" if broken else "ok", "\\p{M}" in (split_regex(tok) or ""), cfg.get("tokenizer_class")))
    if a.check:
        sys.exit(1 if broken else 0)
    if not broken and not a.out:
        return
    tok["pre_tokenizer"] = want_pt
    if want_dec is not None:
        tok["decoder"] = want_dec
    if bad_class:
        cfg["tokenizer_class"] = "TokenizersBackend"

    if a.out:
        os.makedirs(a.out, exist_ok=False)
        for n in sorted(os.listdir(a.pack)):
            if n not in ("tokenizer.json", "tokenizer_config.json"):
                os.symlink(os.path.abspath(os.path.join(a.pack, n)), os.path.join(a.out, n))
        root = a.out
    else:
        for f in (tj, cj):
            if not os.path.exists(f + ".orig"):
                shutil.copy2(f, f + ".orig")
        root = a.pack
    for name, obj in (("tokenizer.json", tok), ("tokenizer_config.json", cfg)):
        dst = os.path.join(root, name)
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(dst + ".tmp", dst)
        print("wrote", dst)


if __name__ == "__main__":
    main()

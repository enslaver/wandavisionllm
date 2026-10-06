#!/usr/bin/env python3
"""unsloth_vision.py OUT [--data ROWS.jsonl] [--steps N | --epochs X] [--lr X] [--4bit] — haiku (Qwen3.5-4B) image LoRA
with Unsloth, on the GPU box (a PC with an NVIDIA GPU; not the Mac, not deployed).

Runs in Unsloth Studio's venv, from the repo clone, with Studio's chat model unloaded. On Windows:
  %USERPROFILE%\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe lora\\unsloth_vision.py %USERPROFILE%\\lora\\haiku-smoke --4bit
Writes OUT/adapter (the LoRA) and OUT/merged (merged_16bit safetensors plus the base's image preprocessor configs).
The Mac copies OUT/merged to ~/lora/merges/<name> and forges it into an MTPLX pack: lora/README.md, "Ship it on the Mac".

Data: unsloth/LaTeX_OCR by default (--rows of it), a pipeline smoke set: it proves train -> merge -> forge -> serve,
it isn't meant to ship. --data FILE.jsonl trains on your own rows, one JSON object per line:
  {"image": "img/0001.jpg", "prompt": "Is the gauge in the green?", "answer": "No: it reads 92, in the red."}
`image` is relative to the JSONL file (or absolute). An optional `think` holds a short reasoning block for a
thinking-on answer; without it the row trains a thinking-off answer (empty <think>).

Vision layers stay frozen (the tower haiku ships is kept). No modules_to_save: the embeddings are tied and feed the
MTP head. --4bit: QLoRA on a 4-bit load, for a 12 GB GPU (the bf16 weights are 9.3 GB). The merge still goes onto
the original 16-bit shards (unsloth_zoo merge_and_overwrite_lora), so nothing is lost there.
"""
import argparse
import json
import os
import shutil
import sys
import time

# Unsloth writes its compiled-module cache to the cwd by default, which would be the repo clone.
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.join(os.path.expanduser("~"), "lora", "unsloth_compiled_cache"))

from unsloth import FastVisionModel  # noqa: E402  (first: Unsloth patches transformers/trl on import)
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

BASE = "unsloth/Qwen3.5-4B"
PROMPT = "Write the LaTeX representation for this image."
LOCK = os.path.join(os.path.expanduser("~"), "lora", "gpu.lock")
GB = 2 ** 30


def pid_alive(pid):
    if sys.platform == "win32":
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return f" {pid} " in out
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock():
    """True once this process holds LOCK (it holds the pid). A lock whose pid is gone, from a crashed run, is taken over."""
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with open(LOCK) as f:
                pid = f.read().strip()
            if not pid.isdigit() or pid_alive(pid):
                return False
            os.remove(LOCK)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def release_lock():
    try:
        with open(LOCK) as f:
            mine = f.read().strip() == str(os.getpid())
        if mine:
            os.remove(LOCK)
    except OSError:
        pass


def gpu_guard(min_free_gb, headroom_gb, wait):
    """One run at a time, only with min_free_gb of VRAM free, then cap PyTorch below what's free.
    Free VRAM drops when Unsloth Studio, ComfyUI or Ollama holds a model. The cap makes an overrun raise CUDA OOM
    instead of letting the Windows driver spill into shared system RAM, which crawls and can stall the desktop.
    With wait, poll every minute instead of exiting."""
    import atexit
    import torch
    said = None
    while True:
        if not take_lock():
            kind, why = "busy", f"GPU busy: another run holds {LOCK}"
        else:
            free, total = torch.cuda.mem_get_info()
            if free >= min_free_gb * GB:
                break
            release_lock()
            kind, why = "vram", (f"only {free / GB:.1f} of {total / GB:.1f} GB VRAM free, need {min_free_gb}: unload the "
                                 "model in Unsloth Studio / ComfyUI / Ollama first (nvidia-smi shows who holds it)")
        if not wait:
            sys.exit(why)
        if kind != said:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {why}; waiting", flush=True)
            said = kind
        time.sleep(60)
    atexit.register(release_lock)
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - headroom_gb * GB) / total))
    print(f"gpu guard: {free / GB:.1f} GB free of {total / GB:.1f}, PyTorch capped at {(free - headroom_gb * GB) / GB:.1f} GB")


def chat(image, question, answer, think=""):
    """One row. Without think, the template renders an empty <think></think> (a thinking-off answer); with it,
    <think>\\n{think}\\n</think> first, exactly what a thinking-on request generates after its '<think>\\n' prompt."""
    # Inline <think> rather than a reasoning_content key: the template renders both the same, and Unsloth's vision
    # collator rebuilds message content, which could drop an extra key.
    text = f"<think>\n{think}\n</think>\n\n{answer}" if think else answer
    assistant = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    user = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": question}]
    return {"messages": [{"role": "user", "content": user}, assistant]}   # image None: a text-only row


def smoke_rows(n):
    ds = load_dataset("unsloth/LaTeX_OCR", split=f"train[:{n}]")
    return [chat(r["image"], PROMPT, r["text"]) for r in ds]


def jsonl_rows(path):
    """--data rows. Images open per row: decoding a few thousand 1280 px images up front takes ~13 GB of RAM.
    A row without "image" is text-only: mix some in to keep a text skill the tier already has (a yes/no gate, say)
    from fading while the adapter learns images."""
    import torch
    from PIL import Image

    root = os.path.dirname(os.path.abspath(path))
    specs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in ("prompt", "answer") if not r.get(k)]
            if missing:
                sys.exit(f"{path}:{n}: missing {', '.join(missing)}")
            image = os.path.join(root, os.path.expanduser(r["image"])) if r.get("image") else None
            specs.append((image, r["prompt"], r["answer"], r.get("think", "")))

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(specs)

        def __getitem__(self, i):
            image, q, ans, think = specs[i]
            return chat(Image.open(image).convert("RGB") if image else None, q, ans, think)

    print(f"{sum(bool(s[3]) for s in specs)} of {len(specs)} rows carry reasoning; "
          f"{sum(s[0] is None for s in specs)} text-only")
    return Rows()


def keep_awake():
    """Ask Windows not to sleep while this process runs (cleared when it exits; no settings change)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", help="JSONL of {[image,] prompt, answer[, think]} rows instead of the LaTeX_OCR smoke set")
    ap.add_argument("--rows", type=int, default=200, help="LaTeX_OCR rows for the smoke run")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--epochs", type=float, help="set --steps from the row count (effective batch 4)")
    ap.add_argument("--lr", type=float, help="default 2e-5 for the smoke set, 1e-4 with --data")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=7.0, help="refuse to start with less VRAM free")
    ap.add_argument("--headroom-gb", type=float, default=0.5, help="keep this much of the free VRAM unused")
    ap.add_argument("--wait-gpu", action="store_true", help="wait for the GPU (lock and free VRAM) instead of exiting")
    ap.add_argument("--max-hours", type=float, help="cap --steps to fit this budget, using --sec-per-step")
    ap.add_argument("--sec-per-step", type=float, help="measured by --probe")
    ap.add_argument("--probe", type=int, help="train this many steps, print sec/step and peak VRAM, save nothing")
    ap.add_argument("--save-every", type=int, default=200, help="checkpoint every N steps (adapter + optimizer)")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in OUT/ckpt")
    a = ap.parse_args()
    gpu_guard(a.min_free_gb, a.headroom_gb, a.wait_gpu)
    keep_awake()
    if a.lr is None:
        a.lr = 1e-4 if a.data else 2e-5
    data = jsonl_rows(a.data) if a.data else smoke_rows(a.rows)
    if a.epochs:
        a.steps = max(1, round(len(data) * a.epochs / 4))
    if a.max_hours and a.sec_per_step:
        a.steps = min(a.steps, int(a.max_hours * 3600 / a.sec_per_step))
    if a.probe:
        a.steps = a.probe
    print(f"{len(data)} rows, {a.steps} steps, lr {a.lr}")

    model, processor = FastVisionModel.from_pretrained(
        BASE, load_in_4bit=a.four_bit, load_in_16bit=not a.four_bit, max_seq_length=2048,
        use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model, processing_class=processor, train_dataset=data,
        data_collator=UnslothVisionDataCollator(
            model, processor, train_on_responses_only=True,
            instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=4, max_steps=a.steps,
            warmup_steps=5, learning_rate=a.lr, lr_scheduler_type="linear", optim="adamw_8bit",
            weight_decay=0.001, logging_steps=5, seed=3407, output_dir=os.path.join(a.out, "ckpt"),
            save_strategy="no" if a.probe else "steps", save_steps=a.save_every, save_total_limit=2,
            report_to="none", bf16=True,
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048, dataloader_num_workers=0))
    stats = trainer.train(resume_from_checkpoint=True if a.resume else None)
    import torch
    print(f"train_loss {stats.training_loss:.4f}  runtime {stats.metrics['train_runtime']:.0f}s  "
          f"peak VRAM {torch.cuda.max_memory_allocated() / GB:.1f} GB")
    if a.probe:
        print(f"probe_sec_per_step {stats.metrics['train_runtime'] / a.steps:.2f}")
        return

    model.save_pretrained(os.path.join(a.out, "adapter"))
    processor.save_pretrained(os.path.join(a.out, "adapter"))
    merged = os.path.join(a.out, "merged")
    model.save_pretrained_merged(merged, processor, save_method="merged_16bit")
    # The merge writes only processor_config.json; mtplx forge's vision graft needs these from the base.
    from huggingface_hub import snapshot_download
    base = snapshot_download(BASE, allow_patterns=["*preprocessor_config.json"])
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        if os.path.exists(os.path.join(base, f)) and not os.path.exists(os.path.join(merged, f)):
            shutil.copyfile(os.path.join(base, f), os.path.join(merged, f))
    print(f"merged -> {merged}  (next: copy it to the Mac's ~/lora/merges/<name>; lora/README.md, 'Ship it on the Mac')")


if __name__ == "__main__":
    main()

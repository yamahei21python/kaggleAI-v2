#!/usr/bin/env python3
"""CPU 網羅テスト 追加分。

  1. step2-vllm-vramcalc  … config.json から重み/KV を計算し、T4x2 で 32768/16384 が載るか判定
  2. step2-vllm-flagspec  … wheel から「使うフラグの値域」と MTP 検証条件を静的に確認

どちらも GPU 不要。1〜2 分。

    python3 kaggle/build_cpu_probe4.py
    cd kaggle/step2-vllm-vramcalc && kaggle kernels push -p .
    cd kaggle/step2-vllm-flagspec && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"


def _nb(md, code):
    cells = []
    for i, (t, src) in enumerate((("markdown", md), ("code", code))):
        d = {"cell_type": t, "id": f"cell{i:02d}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if t == "code":
            d["execution_count"] = None
            d["outputs"] = []
        cells.append(d)
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.13"}},
            "nbformat": 4, "nbformat_minor": 5}


def _write(slug, title, nb, datasets):
    d = HERE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": title,
            "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "keywords": [],
            "dataset_sources": datasets, "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {d}")


VRAMCALC = '''# T4 x2 で max_model_len が載るかを config から計算（GPU 不要）
import json
import os

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
cfg = json.load(open(os.path.join(ROOT, "config.json")))

print("architectures        :", cfg.get("architectures"))
print("model_type           :", cfg.get("model_type"))
print("num_hidden_layers    :", cfg.get("num_hidden_layers"))
print("hidden_size          :", cfg.get("hidden_size"))
print("num_attention_heads  :", cfg.get("num_attention_heads"))
print("num_key_value_heads  :", cfg.get("num_key_value_heads"))
print("head_dim             :", cfg.get("head_dim"))
print("max_position_emb     :", cfg.get("max_position_embeddings"))
print("torch_dtype          :", cfg.get("torch_dtype"))
print("quantization_config  :", json.dumps(cfg.get("quantization_config"), ensure_ascii=False))
print("rope_scaling         :", cfg.get("rope_scaling"))
print("tie_word_embeddings  :", cfg.get("tie_word_embeddings"))

L = cfg["num_hidden_layers"]
H = cfg["hidden_size"]
NH = cfg["num_attention_heads"]
NKV = cfg.get("num_key_value_heads") or NH
HD = cfg.get("head_dim") or (H // NH)
VOCAB = cfg.get("vocab_size")
print("\\nderived: head_dim", HD, "kv_heads", NKV, "vocab", VOCAB)

# --- 重みサイズ（実ファイル）---
def sz(name):
    p = os.path.join(ROOT, name)
    return os.path.getsize(p) if os.path.isfile(p) else 0

main = sz("model.safetensors")
mtp = sz("model-mtp.safetensors")
print(f"\\nweights: main {main/1e9:.2f} GB / mtp {mtp/1e9:.2f} GB")

# --- KV cache（fp16）---
kv_per_token = 2 * L * NKV * HD * 2          # K+V, 2 bytes
print(f"KV per token (fp16): {kv_per_token/1024**2:.3f} MiB")
print(f"KV for 32768 tokens: {kv_per_token*32768/1e9:.2f} GB (TP=2 で /2)")
print(f"KV for 16384 tokens: {kv_per_token*16384/1e9:.2f} GB (TP=2 で /2)")

# --- T4 x2 の見積り ---
GPU_MEM = 15.36           # GiB (15360 MiB)
UTIL = 0.90
N = 2
budget = GPU_MEM * UTIL * N
weight_per_gpu = (main + mtp) / N
print(f"\\nTP=2: budget {budget:.2f} GiB / weights {weight_per_gpu:.2f} GiB "
      f"→ KV+act {budget - weight_per_gpu:.2f} GiB")
for ml in (32768, 16384, 8192, 4096):
    need = kv_per_token * ml / N
    free = budget - weight_per_gpu - need
    verdict = "OK" if free > 1.0 else "NG"
    print(f"  max_model_len={ml:6d}: KV {need:6.2f} GiB, 残 {free:6.2f} GiB -> {verdict}")

# --- 量子化の妥当性（vLLM の awq は 4bit / group 128 / zp 有無）---
q = cfg.get("quantization_config") or {}
print("\\nquant check: bits=", q.get("bits"), "group_size=", q.get("group_size"),
      "zero_point=", q.get("zero_point"), "sym=", q.get("sym"))

# --- 理論重みサイズとの比較（AWQ 4bit なら ~14-15GB 期待）---
if VOCAB:
    print(f"embed/lm_head fp16 概算: {2*VOCAB*H*2/1e9:.2f} GB (tie={cfg.get('tie_word_embeddings')})")
'''

FLAGSPEC = '''# 使うフラグの「値域」と MTP の検証条件を wheel から確認（install 不要）
import glob
import os
import re
import subprocess
import sys
import zipfile

r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
                    "--no-deps", "vllm==0.32.0", "-d", "/tmp/w"],
                   capture_output=True, text=True)
whl = glob.glob("/tmp/w/*.whl")
print("download rc:", r.returncode, [os.path.basename(w) for w in whl])
z = zipfile.ZipFile(whl[0])
z.extractall("/tmp/vl")
import pathlib

root = pathlib.Path("/tmp/vl/vllm")

TARGET = ["--host", "--port", "--quantization", "--kv-cache-dtype",
          "--dtype", "--load-format", "--speculative-config",
          "--reasoning-parser", "--max-num-seqs", "--gpu-memory-utilization",
          "--tensor-parallel-size", "--max-model-len", "--served-model-name",
          "--enable-log-requests", "--enforce-eager"]

# add_argument のブロックを雑に集める
hits = {t: [] for t in TARGET}
for f in root.rglob("*.py"):
    try:
        lines = f.read_text(errors="replace").splitlines()
    except Exception:
        continue
    for i, line in enumerate(lines):
        for t in TARGET:
            if f'"{t}"' in line or f"'{t}'" in line:
                ctx = " | ".join(x.strip() for x in lines[i:i + 4])
                hits[t].append(f"{f.relative_to(root)}:{i+1}: {ctx[:220]}")

print("\\n=== フラグ定義（値域・choices）===")
for t in TARGET:
    print(f"\\n-- {t}: {len(hits[t])} hits")
    for h in hits[t][:4]:
        print("   ", h)

# choices の実値（quantization / kv-cache-dtype / load-format）
for name, rel in (("quantization", "model_executor/layers/quantization/__init__.py"),
                  ("kv", "config/cache.py"),
                  ("load", "config/load.py")):
    f = root / rel
    if not f.exists():
        print("missing:", rel)
        continue
    txt = f.read_text(errors="replace")
    print(f"\\n==== {rel} ====")
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if any(k in s for k in ("awq", "auto", "fp8", "CLASS", "Literal",
                                "safetensors", "pt", "dummy", "bitsandbytes")):
            print(f"  {i}: {s[:160]}")

# MTP の検証条件
sp = root / "config/speculative.py"
txt = sp.read_text(errors="replace")
print("\\n==== speculative.py: mtp 関連の検証 ====")
for i, line in enumerate(txt.splitlines(), 1):
    s = line.strip()
    if ("mtp" in s.lower() or "raise ValueError" in s
            or "num_speculative_tokens" in s):
        print(f"  {i}: {s[:170]}")
'''

if __name__ == "__main__":
    _write("step2-vllm-vramcalc", "Step2 vLLM vramcalc",
           _nb("# 🧮 T4 x2 で max_model_len が載るかの計算（GPU 不要）\n\n"
               "config.json と実ファイルサイズから KV cache と残 VRAM を見積る。\n",
               VRAMCALC), [DATASET_ID])
    _write("step2-vllm-flagspec", "Step2 vLLM flagspec",
           _nb("# 🧷 フラグの値域と MTP 検証条件（wheel 静的解析）\n\n"
               "`--quantization awq` / `--kv-cache-dtype auto` が通るか、"
               "`method=mtp` に何が要求されるか。\n",
               FLAGSPEC), [])

#!/usr/bin/env python3
"""vLLM wheel を「展開せず中身だけ見る」軽量プローブ（1〜2 分）。

理由: フル install（5 分）を待たずに、次が分かる。
  - `method=mtp` が 0.32.0 に存在するか（MTP 段を組む価値の有無）
  - CPU で `vllm serve --help` が落ちた原因（`get_kwargs(VllmConfig)["logging_config"]`）
  - wheel サイズ（= 転送時間の下限。uv / Dataset 施策の判断材料）

    python3 kaggle/build_wheel_inspect.py
    cd kaggle/step2-vllm-wheelinspect && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
SLUG = "step2-vllm-wheelinspect"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / SLUG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE = '''# vLLM wheel の中身だけ見る（install しない・1〜2 分）
import glob
import os
import subprocess
import sys
import zipfile

SPEC = "vllm==0.32.0"

r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
                    "--no-deps", SPEC, "-d", "/tmp/w"],
                   capture_output=True, text=True)
print("download rc:", r.returncode)
print(((r.stdout or "") + (r.stderr or ""))[-800:])

whls = glob.glob("/tmp/w/*.whl")
print("wheels:", [os.path.basename(w) for w in whls])
if not whls:
    raise SystemExit("wheel が取れない（index 到達不可）")
whl = whls[0]
size = os.path.getsize(whl)
print(f"wheel: {os.path.basename(whl)}  {size/1e6:.1f} MB")

z = zipfile.ZipFile(whl)
names = z.namelist()
print("entries:", len(names))

# --- 1. mtp の有無（speculative 系のみ）---
mtp_lines = []
for n in names:
    if not n.endswith(".py"):
        continue
    if "speculative" not in n and "arg_utils" not in n:
        continue
    try:
        txt = z.read(n).decode("utf-8", "replace")
    except Exception:
        continue
    if "mtp" not in txt.lower():
        continue
    for i, line in enumerate(txt.splitlines(), 1):
        if "mtp" in line.lower():
            mtp_lines.append(f"{n}:{i}: {line.strip()[:170]}")
print(f"\\nmtp 出現: {len(mtp_lines)}")
print("\\n".join(mtp_lines[:60]))

# --- 2. logging_config が VllmConfig にあるか（CPU 失敗の原因）---
try:
    cfg = z.read("vllm/config/__init__.py").decode("utf-8", "replace")
except KeyError:
    cfg = ""
print("\\nvllm/config/__init__.py に logging_config:",
      "logging_config" in cfg)
for i, line in enumerate(cfg.splitlines(), 1):
    if "logging_config" in line or "class VllmConfig" in line:
        print(f"  __init__.py:{i}: {line.strip()[:150]}")

# --- 3. _compute_kwargs / get_kwargs の該当箇所 ---
try:
    au = z.read("vllm/engine/arg_utils.py").decode("utf-8", "replace")
except KeyError:
    au = ""
for i, line in enumerate(au.splitlines(), 1):
    s = line.strip()
    if s.startswith("def _compute_kwargs") or s.startswith("def get_kwargs"):
        print(f"  arg_utils.py:{i}: {s}")
    if '"logging_config"' in s:
        print(f"  arg_utils.py:{i}: {s[:150]}")

# --- 3.5 CLI フラグをソースから静的に列挙（help は GPU 無しで動かない）---
import re
import pathlib

z.extractall("/tmp/vl")
root = pathlib.Path("/tmp/vl")
flags = set()
# 全 .py を走査（0.32 は CLI 定義の置き場所が変わっている）
for f in (root / "vllm").rglob("*.py"):
    try:
        txt = f.read_text(errors="replace")
    except Exception:
        continue
    if "add_argument" not in txt and "add_cli_args" not in txt:
        continue
    for m in re.finditer(r"""["'](--[a-z0-9][a-z0-9-]*)["']""", txt):
        flags.add(m.group(1))
print(f"フラグ総数: {len(flags)}")
NEED = ["--speculative-config", "--kv-cache-dtype", "--trust-remote-code",
        "--reasoning-parser", "--enable-auto-tool-choice", "--tool-call-parser",
        "--disable-log-requests", "--enforce-eager", "--max-num-seqs",
        "--served-model-name", "--tensor-parallel-size",
        "--gpu-memory-utilization", "--max-model-len", "--load-format",
        "--quantization", "--dtype", "--host", "--port"]
for f in NEED:
    print(("OK  " if f in flags else "MISS"), f)
print("\\n".join(sorted(flags)))

# --- 3.6 CPU で落ちた device 推論の条件 ---
try:
    dev = (root / "vllm/config/device.py").read_text(errors="replace")
    for i, line in enumerate(dev.splitlines(), 1):
        if 40 <= i <= 70:
            print(f"device.py:{i}: {line[:150]}")
except Exception as e:
    print("device.py NG:", e)

# --- 4. 量子化の対応表（T4=sm75 で awq_marlin が選ばれるか）---
for cand in ("vllm/model_executor/layers/quantization/__init__.py",
             "vllm/config/cache.py"):
    try:
        txt = z.read(cand).decode("utf-8", "replace")
    except KeyError:
        continue
    print(f"\\n--- {cand} ---")
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if "marlin" in s.lower() or "capability" in s.lower():
            print(f"  {cand}:{i}: {s[:150]}")

# awq_marlin が要求する sm を直接見る（T4=sm75 で使えるか）
for rel in ("vllm/model_executor/layers/quantization/awq_marlin.py",
            "vllm/platforms/cuda.py",
            "vllm/model_executor/layers/quantization/utils/marlin_utils.py"):
    f = root / rel
    if not f.exists():
        print("missing:", rel)
        continue
    print(f"\\n==== {rel} ====")
    for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
        s = line.strip()
        if ("capability" in s.lower() or "sm7" in s.lower()
                or "75" in s):
            print(f"  {rel}:{i}: {s[:160]}")
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "id": "cell00", "metadata": {},
         "source": ["# 🔎 vLLM wheel の中身だけ確認（install なし）\n",
                    "\n",
                    "- `method=mtp` の有無\n",
                    "- CPU で help が落ちた原因（`logging_config`）\n",
                    "- T4(sm75) で marlin が選ばれるか\n"]},
        {"cell_type": "code", "id": "cell01", "metadata": {},
         "execution_count": None, "outputs": [],
         "source": CODE.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT_DIR / f"{SLUG}.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = {
    "id": f"{OWNER}/{SLUG}",
    "title": "Step2 vLLM wheelinspect",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT_DIR}")

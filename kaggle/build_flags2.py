#!/usr/bin/env python3
"""フラグ第 2 弾: chat-template / language-model-only / reasoning parser 名 / 量子化方式。

wheel を落として中身を見るだけ（install なし・1〜2 分）。

    python3 kaggle/build_flags2.py
    cd kaggle/step2-vllm-flags2 && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
SLUG = "step2-vllm-flags2"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / SLUG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE = '''# chat-template / reasoning parser 名 / 量子化方式を wheel から確認
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile

r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
                    "--no-deps", "vllm==0.32.0", "-d", "/tmp/w"],
                   capture_output=True, text=True)
whl = glob.glob("/tmp/w/*.whl")
print("download rc:", r.returncode)
if not whl:
    raise SystemExit("wheel 取得失敗")
z = zipfile.ZipFile(whl[0])
z.extractall("/tmp/vl")
root = pathlib.Path("/tmp/vl/vllm")

# --- 1. 探したいフラグ ---
NEED = ["--chat-template", "--language-model-only", "--limit-mm-per-prompt",
        "--mm-processor-kwargs", "--reasoning-parser", "--structured-outputs-config",
        "--enable-prefix-caching", "--disable-mm-preprocessor-cache",
        "--quantization", "--spec-method", "--hf-overrides"]
print("\\n=== add_argument の存在 ===")
found = {f: [] for f in NEED}
for f in root.rglob("*.py"):
    try:
        txt = f.read_text(errors="replace")
    except Exception:
        continue
    for n in NEED:
        if f'"{n}"' in txt or f"'{n}'" in txt:
            for i, line in enumerate(txt.splitlines(), 1):
                if n in line:
                    found[n].append(f"{f.relative_to(root)}:{i}: {line.strip()[:150]}")
for n in NEED:
    print(f"\\n-- {n}: {len(found[n])}")
    for h in found[n][:3]:
        print("   ", h)

# --- 2. reasoning parser の登録名（--reasoning-parser に渡す文字列）---
print("\\n=== reasoning parser 登録名 ===")
reg = root / "reasoning"
print("dir exists:", reg.exists())
names = set()
for f in (reg.rglob("*.py") if reg.exists() else []):
    txt = f.read_text(errors="replace")
    for m in re.finditer(r'register_module\\(\\s*ReasoningParser\\s*,\\s*["\\']([^"\\']+)',
                         txt):
        names.add(m.group(1))
    # 旧式 dict 登録
    for m in re.finditer(r'^\\s*["\\']([a-z0-9_]+)["\\']\\s*:\\s*["\\'][A-Za-z0-9_]+Reasoning',
                         txt, re.M):
        names.add(m.group(1))
print(sorted(names))
for f in (reg.rglob("*.py") if reg.exists() else []):
    n = f.name
    if "qwen" in n.lower():
        print("  file:", n)

# qwen3 系の登録状況をクラス名から推測
if reg.exists():
    for f in reg.rglob("*.py"):
        txt = f.read_text(errors="replace")
        if "qwen" not in f.name.lower() and "qwen" not in txt.lower():
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if "register_module" in line or "ReasoningParser," in line:
                print(f"   {f.name}:{i}: {line.strip()[:150]}")

# --- 3. reasoning_parser の選択肢 ---
print("\\n=== reasoning_parser の choices ===")
au = (root / "engine/arg_utils.py").read_text(errors="replace")
i = au.find("--reasoning-parser")
print(au[max(0, i - 400):i + 500] if i > 0 else "not found")

# --- 4. compressed-tensors 対応 ---
print("\\n=== 量子化方式 ===")
q = (root / "model_executor/layers/quantization/__init__.py").read_text(errors="replace")
m = re.search(r"QuantizationMethods\\s*=\\s*Literal\\[(.*?)\\]", q, re.S)
print(m.group(1)[:1500] if m else "not found")
print("compressed-tensors を含む:", "compressed-tensors" in q)
print("compressed_tensors を含む:", "compressed_tensors" in q)

# --- 5. chat template の解決（tokenizer_config に無い場合）---
print("\\n=== chat template 解決 ===")
for f in root.rglob("chat_template*.py"):
    print("  file:", f.relative_to(root))
for f in root.rglob("*.py"):
    txt = f.read_text(errors="replace")
    if "chat_template" in txt and "tokenizer_config" in txt:
        for i, line in enumerate(txt.splitlines(), 1):
            if "chat_template" in line and ("tokenizer_config" in line
                                            or "get_chat_template" in line
                                            or "jinja" in line):
                print(f"  {f.relative_to(root)}:{i}: {line.strip()[:150]}")
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "id": "cell00", "metadata": {},
         "source": ["# 🧷 フラグ第 2 弾（wheel 静的解析）\n",
                    "\n",
                    "`--chat-template` / `--language-model-only` / reasoning parser 名 /"
                    " compressed-tensors 対応。\n"]},
        {"cell_type": "code", "id": "cell01", "metadata": {},
         "execution_count": None, "outputs": [],
         "source": CODE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.13"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(OUT_DIR / f"{SLUG}.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

meta = {"id": f"{OWNER}/{SLUG}", "title": "Step2 vLLM flags2",
        "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": []}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT_DIR}")

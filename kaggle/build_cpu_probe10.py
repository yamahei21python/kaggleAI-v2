#!/usr/bin/env python3
"""CPU 網羅テスト 第 10 弾 — 4 本。

    python3 kaggle/build_cpu_probe10.py
    bash kaggle/push_probe10.sh
"""
import json
import pathlib
import textwrap

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent


def _nb(title, code):
    return {
        "cells": [
            {"cell_type": "markdown", "id": "md01", "metadata": {}, "source": [f"# {title}\n\n"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "code01",
                "metadata": {},
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(slug, title, code):
    d = HERE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(json.dumps(_nb(title, code), ensure_ascii=False, indent=1))
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
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
    (d / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {d / slug}.ipynb")


# --------------------------------------------------------------------------
# 1. Python 3.13 compileall (構文・future文法)
# --------------------------------------------------------------------------
COMPILEALL = textwrap.dedent(
    r"""
import compileall
import sys
import os

dist = "/usr/local/lib/python3.13/dist-packages"
print(f"Python {sys.version} で compileall 実行中...")
result = compileall.compile_dir(dist, quiet=1, force=True)
print(f"compileall result: {'OK' if result is not None else 'None返す'}")

# vllm と torch のみ抜粋
for pkg in ("vllm", "torch", "triton"):
    p = os.path.join(dist, pkg)
    if os.path.isdir(p):
        r = os.popen(f"python -m compileall '{p}' 2>&1 | head -n 20").read()
        print(f"\n--- {pkg} ---\n{r.strip()}")
    else:
        print(f"\n--- {pkg} --- (not found in dist)")
"""
)
_write("cpu-probe10-compileall", "Python 3.13 compileall", COMPILEALL)


# --------------------------------------------------------------------------
# 2. ldd 依存確認 (未解決共有ライブラリ・RPATH)
# --------------------------------------------------------------------------
LDD = textwrap.dedent(
    r"""
import subprocess
import glob
import re

dist = "/usr/local/lib/python3.13/dist-packages"
libs = glob.glob(f"{dist}/**/*.so", recursive=True)[:200]
print(f"対象 .so: {len(libs)} 件")

found_bad = []
found_rpath = []
gpu_deps = []

for lib in libs:
    try:
        r = subprocess.run(["ldd", lib], capture_output=True, text=True, timeout=5)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        out = str(e)

    bad = re.findall(r"not found", out, re.IGNORECASE)
    rpath = re.findall(r"(?:RPATH|RUNPATH)=(.+?)\n", out)

    if bad:
        found_bad.append((lib, bad))
    if rpath:
        found_rpath.append((lib, rpath))
    if any(dep in out.lower() for dep in ["libcud", "libnvidia"]):
        gpu_deps.append(lib)

print(f"\n=== not found: {len(found_bad)} ===")
for lib, msgs in found_bad[:10]:
    print(f"  {lib}: {msgs}")

print(f"\n=== RPATH/RUNPATH: {len(found_rpath)} ===")
for lib, paths in found_rpath[:5]:
    print(f"  {lib}: {paths}")

print(f"\n=== GPU lib 依存: {len(gpu_deps)} ===")
for lib in gpu_deps[:10]:
    print(f"  {lib}")

print("\n=== ldd 完了 ===")
"""
)
_write("cpu-probe10-ldd", "ldd 依存チェック", LDD)


# --------------------------------------------------------------------------
# 3. Offline Restart (カーネル再起動後の import 確認)
# --------------------------------------------------------------------------
OFFLINE = textwrap.dedent(
    r"""
# 本セルはカーネル再起動後に実行すること
import sys

def check_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"[OK] {name} {ver}")
        return True
    except Exception as e:
        print(f"[NG] {name}: {type(e).__name__}: {str(e)[:200]}")
        return False

print("=== Python 環境 ===")
print(f"Python {sys.version}")

print("\n=== 主要パッケージ ===")
for pkg in ("torch", "transformers", "triton", "vllm"):
    check_import(pkg)

print("\n=== import vllm の詳細 ===")
try:
    import vllm
    print(f"[OK] vllm {vllm.__version__}")
    print(f"  path: {vllm.__file__}")
    # vllm.env_override が torch を import している
    print(f"  env_override ok: {hasattr(vllm, 'env_override')}")
except ImportError as e:
    print(f"[NG] vllm import error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== pip check ===")
import subprocess
r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
out = r.stdout or r.stderr
if out.strip():
    lines = out.strip().splitlines()
    print(f"{len(lines)} warnings/errors:")
    for line in lines[:15]:
        print(f"  {line}")
else:
    print("  (clean)")
"""
)
_write("cpu-probe10-offline", "Offline Restart 試験", OFFLINE)


# --------------------------------------------------------------------------
# 4. モデルファイル検査 (config.json, safetensors, tokenizer)
# --------------------------------------------------------------------------
MODEL = textwrap.dedent(
    r"""
import json
import glob
import os
import hashlib

# safetensors index 検索
index_cands = glob.glob("/kaggle/input/*/model.safetensors.index.json")
print("safetensors index:", index_cands[:3])

if not index_cands:
    print("⚠ index.json なし。safetensors ファイル一覧:")
    for f in glob.glob("/kaggle/input/**/*.safetensors"):
        print(f"  {f}")
    print("⚠ モデルファイル検査はスキップ (dataset 未アタッチ)")
    raise SystemExit()

index_path = index_cands[0]
print(f"index: {index_path}")
idx = json.load(open(index_path))

metadata = idx.get("metadata", {})
weight_map = idx.get("weight_map", {})
print(f"total params: {metadata.get('total', 'unknown')}")

# shard ファイル検証
missing = []
bad_sha = []
for fname in weight_map.values():
    fpath = os.path.join(os.path.dirname(index_path), fname)
    if not os.path.exists(fpath):
        missing.append(fname)
        continue
    # SHA256 チェック (先頭 1MB のみ)
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        h.update(f.read(1024 * 1024))
    bad_sha.append((fname, h.hexdigest()[:12]))

print(f"\ntotal shards: {len(weight_map)}")
print(f"missing: {len(missing)}")
if missing:
    print("  missing:", missing[:5])
print(f"sha256 (sample): {len(bad_sha)}")
for name, sha in bad_sha[:5]:
    print(f"  {name}: {sha}")

# tokenizer 確認
tok_paths = glob.glob("/kaggle/input/**/tokenizer*.json", recursive=True) + \
            glob.glob("/kaggle/input/**/tokenizer*.model", recursive=True)
print(f"\ntokenizer files: {len(tok_paths)}")
for t in tok_paths[:3]:
    print(f"  {t} ({os.path.getsize(t) / 1024 / 1024:.1f} MB)")

# config.json
cfg_matches = glob.glob("/kaggle/input/*/config.json")
if not cfg_matches:
    raise SystemExit("config.json が見つからない")
cfg_path = cfg_matches[0]
cfg = json.load(open(cfg_path))
print(f"\nconfig.json:")
print(f"  model_type: {cfg.get('model_type')}")
print(f"  architectures: {cfg.get('architectures')}")
print(f"  num_attention_heads: {cfg.get('num_attention_heads')}")
print(f"  num_key_value_heads: {cfg.get('num_key_value_heads')}")
print(f"  hidden_size: {cfg.get('hidden_size')}")

print("\n=== モデルファイル検査完了 ===")
"""
)
_write("cpu-probe10-model", "モデルファイル検査", MODEL)

print("ALL CELLS WRITTEN")
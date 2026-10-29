#!/usr/bin/env python3
"""`vllm serve --help` を CPU で通す方法を探す（= フラグを正式に確定）。

CPU では `Failed to infer device type` で help が落ちる。プラットフォーム解決を
env で固定できれば通るはず。候補を順に試して、通ったらフラグ全文を保存する。

    python3 kaggle/build_helpsolve.py
    cd kaggle/step2-vllm-helpsolve && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
SLUG = "step2-vllm-helpsolve"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / SLUG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE = '''# vllm serve --help を CPU で通す（install あり・約 6 分）
import json
import os
import subprocess
import sys
import time

r = subprocess.run([sys.executable, "-c", "import vllm"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("pip install vllm==0.32.0")
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--no-input", "vllm==0.32.0"],
                       capture_output=True, text=True, timeout=3600)
    print(p.returncode, ((p.stdout or "") + (p.stderr or ""))[-800:])

# torchaudio 不整合（GPU と同じ罠）を先に潰す
p = subprocess.run(["vllm", "serve", "--help"], capture_output=True, text=True)
if "torchaudio" in ((p.stdout or "") + (p.stderr or "")):
    print("torchaudio 不整合 → uninstall")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
                   capture_output=True, text=True, timeout=600)

CANDIDATES = [
    ("baseline", {}),
    ("VLLM_PLATFORM=cpu", {"VLLM_PLATFORM": "cpu"}),
    ("VLLM_TARGET_DEVICE=cpu", {"VLLM_TARGET_DEVICE": "cpu"}),
    ("VLLM_PLATFORM=cpu+VLLM_TARGET_DEVICE=cpu",
     {"VLLM_PLATFORM": "cpu", "VLLM_TARGET_DEVICE": "cpu"}),
    ("VLLM_USE_V1=0", {"VLLM_USE_V1": "0"}),
]

best = None
for label, env_extra in CANDIDATES:
    env = {**os.environ, **env_extra}
    t0 = time.time()
    p = subprocess.run(["vllm", "serve", "--help"], capture_output=True,
                       text=True, env=env, timeout=600)
    out = (p.stdout or "") + (p.stderr or "")
    ok = (p.returncode == 0 and "Traceback" not in out and len(out) > 500
          and "--tensor-parallel-size" in out)
    print(f"[{label}] rc={p.returncode} len={len(out)} ok={ok} "
          f"{time.time()-t0:.1f}s")
    if not ok:
        tail = [l for l in out.splitlines() if l.strip()][-3:]
        print("   ", " / ".join(tail)[:300])
    elif best is None:
        best = (label, env_extra, out)

if best:
    label, env_extra, out = best
    print(f"\\n✅ help 成功: {label}")
    open("/kaggle/working/vllm_help.txt", "w").write(out[:60000])
    NEED = ["--host", "--port", "--speculative-config", "--spec-method",
            "--kv-cache-dtype", "--quantization", "--dtype", "--load-format",
            "--reasoning-parser", "--enforce-eager", "--max-num-seqs",
            "--served-model-name", "--tensor-parallel-size",
            "--gpu-memory-utilization", "--max-model-len",
            "--disable-log-requests", "--enable-log-requests",
            "--enable-auto-tool-choice", "--tool-call-parser"]
    res = {"label": label, "env": env_extra,
           "flags": {f: (f in out) for f in NEED},
           "chars": len(out)}
    print(json.dumps(res["flags"], indent=1))
    with open("/kaggle/working/helpsolve.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
else:
    print("\\n❌ どの env でも help は通らなかった")
    # 最後の手段: プラットフォーム解決の中身を見る
    p = subprocess.run([sys.executable, "-c",
                        "import vllm.platforms as P;"
                        "import inspect;"
                        "print([n for n in dir(P) if 'platform' in n.lower()]);"
                        "print(inspect.getsource(P.resolve_current_platform))"],
                       capture_output=True, text=True)
    print(((p.stdout or "") + (p.stderr or ""))[-2500:])
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "id": "cell00", "metadata": {},
         "source": ["# 🛠 `vllm serve --help` を CPU で通す\n",
                    "\n",
                    "プラットフォーム解決を env で固定し、フラグを正式確定する。\n"]},
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

meta = {"id": f"{OWNER}/{SLUG}", "title": "Step2 vLLM helpsolve",
        "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": []}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT_DIR}")

#!/usr/bin/env python3
"""CPU 並列プローブ 2 本を生成（GPU クォータ消費ゼロ）。

作る kernel:
  1. step2-vllm-uvbench   … install の内訳測定（pip / uv / download 量）
  2. step2-vllm-mtpprobe  … MTP 対応の静的確認 + torchaudio 修復の実証

どちらも GPU 不要。dryrun と同時に走らせて待ち時間を潰す。

    python3 kaggle/build_cpu_probe.py
    cd kaggle/step2-vllm-uvbench && kaggle kernels push -p .
    cd kaggle/step2-vllm-mtpprobe && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
VLLM_SPEC = "vllm==0.32.0"


def _nb(cells):
    out = []
    for i, (t, src) in enumerate(cells):
        d = {"cell_type": t, "id": f"cell{i:02d}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if t == "code":
            d["execution_count"] = None
            d["outputs"] = []
        out.append(d)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(slug, title, nb):
    d = HERE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
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
    (d / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {d}")


# ── 1: uvbench ──────────────────────────────────────────────
UVBENCH = f'''# vLLM install の内訳測定（GPU 不要）
# 目的: 起動短縮 A（uv）/ B（wheel Dataset）のどちらが効くかを実測で決める。
#   1. pip install {VLLM_SPEC}         … 現行の実測基準
#   2. pip download --no-deps         … 純粋な転送+書込（wheel 1 個）
#   3. uv pip install --system        … 解決+展開の短縮幅
#   4. pip download（依存込み）        … wheel Dataset 化した場合の総量
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time

SPEC = "{VLLM_SPEC}"


def run(cmd, label, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[{{label}}] {{dt:.1f}}s  rc={{r.returncode}}")
    if r.returncode != 0:
        print(((r.stdout or "") + (r.stderr or ""))[-1500:])
    return dt, r


def du(path):
    total = 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n


def vers():
    out = {{}}
    for p in ("torch", "torchaudio", "transformers", "vllm"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


res = {{"spec": SPEC, "before": vers()}}
print("before:", res["before"])
print("python:", sys.version.split()[0])

# 1. 現行と同じ pip install
res["pip_install_sec"], _ = run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-input", SPEC],
    "pip install")
res["after_pip"] = vers()

# 2. wheel 1 個の転送時間（依存解決を除く）
dt, _ = run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
             "--no-deps", SPEC, "-d", "/tmp/w1"], "pip download no-deps")
res["pip_download_nodeps_sec"] = dt
res["pip_download_nodeps_bytes"], _ = du("/tmp/w1")

# 3. uv（vllm だけ外して、依存は既に入っている状態で測る）
run([sys.executable, "-m", "pip", "uninstall", "-y", "vllm"], "uninstall vllm")
run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "uv"],
    "install uv")
dt, r = run(["uv", "pip", "install", "--system", "-q", SPEC], "uv install")
res["uv_install_sec"] = dt
res["uv_ok"] = r.returncode == 0
res["uv_out"] = ((r.stdout or "") + (r.stderr or ""))[-800:]
res["after_uv"] = vers()

# 4. 依存込みの総 wheel サイズ（Dataset 化の実現性）
dt, _ = run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
             SPEC, "-d", "/tmp/wf"], "pip download all", timeout=5400)
res["pip_download_all_sec"] = dt
res["pip_download_all_bytes"], res["pip_download_all_files"] = du("/tmp/wf")
print("all wheels:", res["pip_download_all_bytes"] / 1e9, "GB",
      res["pip_download_all_files"], "files")
print(sorted(os.listdir("/tmp/wf"))[:20])

with open("/kaggle/working/uvbench.json", "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])
'''

# ── 2: mtpprobe ─────────────────────────────────────────────
MTPPROBE = f'''# MTP 対応の静的確認 + torchaudio 修復の実証（GPU 不要）
# 理由: MTP が vLLM 0.32 に無いなら、GPU ラダーの mtp 段は無駄。先に潰す。
import json
import os
import pathlib
import subprocess
import sys
import time

SPEC = "{VLLM_SPEC}"


def sh(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# install（dryrun と同じ手順）
t0 = time.time()
r = sh([sys.executable, "-c", "import vllm"])
if r.returncode != 0:
    print("pip install", SPEC)
    p = sh([sys.executable, "-m", "pip", "install", "-q", "--no-input", SPEC])
    print(p.returncode, ((p.stdout or "") + (p.stderr or ""))[-1200:])
print(f"install/確認: {{time.time()-t0:.0f}}s")


def cli_ok():
    r = sh(["vllm", "serve", "--help"], 300)
    out = (r.stdout or "") + (r.stderr or "")
    ok = (r.returncode == 0 and "Traceback" not in out and len(out) > 500)
    return ok, out


ok, out = cli_ok()
print("CLI OK:", ok, len(out), "文字")
fix = "none"
if not ok:
    print("⚠ traceback → torchaudio を外す")
    print(out[-800:])
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], 600)
    ok, out = cli_ok()
    fix = "uninstall"
print("CLI OK after:", ok, len(out), "文字 / fix:", fix)
if ok:
    open("/kaggle/working/vllm_help.txt", "w").write(out[:20000])

import vllm
print("vllm", vllm.__version__)
root = pathlib.Path(vllm.__file__).parent
print("root:", root)

# --- MTP の静的探索 ---
hits = []
for p in root.rglob("*.py"):
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        continue
    low = txt.lower()
    if "mtp" not in low and "speculative" not in low:
        continue
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if "mtp" in s.lower():
            hits.append(f"{{p.relative_to(root)}}:{{i}}: {{s[:160]}}")
print(f"mtp 出現: {{len(hits)}}")
print("\\n".join(hits[:80]))
open("/kaggle/working/mtp_hits.txt", "w").write("\\n".join(hits))

# --- SpeculativeConfig のスキーマ ---
report = {{"cli_ok": ok, "fix": fix, "mtp_hits": len(hits)}}
try:
    from vllm.config.speculative import SpeculativeConfig
    fields = getattr(SpeculativeConfig, "model_fields", None) or {{}}
    report["spec_fields"] = list(fields)
    print("fields:", list(fields))
    import inspect
    src = inspect.getsource(SpeculativeConfig)
    report["spec_src_head"] = src[:2500]
    print(src[:2500])
except Exception as e:
    report["spec_err"] = f"{{type(e).__name__}}: {{e}}"
    print("spec introspect NG:", report["spec_err"])

# --- 起動ラダーで使うフラグの実在確認 ---
NEED = ["--speculative-config", "--kv-cache-dtype", "--trust-remote-code",
        "--reasoning-parser", "--enable-auto-tool-choice", "--tool-call-parser",
        "--disable-log-requests", "--enforce-eager", "--max-num-seqs",
        "--served-model-name", "--tensor-parallel-size",
        "--gpu-memory-utilization", "--max-model-len", "--load-format"]
report["flags"] = {{f: (f in out) for f in NEED}}
print(json.dumps(report["flags"], indent=1))

with open("/kaggle/working/mtpprobe.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
'''


def main():
    _write("step2-vllm-uvbench", "Step2 vLLM uvbench",
           _nb([("markdown", "# 📦 vLLM install 内訳の実測（CPU）\n\n"
                            "uv / wheel Dataset の効果を数字で決める。\n"),
                ("code", UVBENCH)]))
    _write("step2-vllm-mtpprobe", "Step2 vLLM mtpprobe",
           _nb([("markdown", "# 🔍 MTP 対応とフラグの静的確認（CPU）\n\n"
                            "vLLM 0.32 が `method=mtp` を受けるか、"
                            "torchaudio 修復手順が効くかを先に確かめる。\n"),
                ("code", MTPPROBE)]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""wheelhouse Dataset の CPU 検証ノートを作る。

GPU クォータを使わずに、次を実測する:
  1. `--no-index --find-links` で 183 wheel から vllm が入るか
  2. install にかかる秒数（GPU 版 install 504 秒の短縮幅）
  3. torch / torchaudio の CUDA 版数が揃うか（cu130 同士か）
  4. `pip check` で依存破損が出ないか
  5. `vllm serve --help=all` が CPU でどこまで動くか

    python3 kaggle/build_whl_probe.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "whl-probe"
SLUG = "yamahei21python/whl-probe"
WH = "yamahei21python/vllm-wheelhouse"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n")})


md("""
# wheelhouse の CPU 検証

GPU クォータを使わずに、`yamahei21python/vllm-wheelhouse`（183 wheel / 1.9 GB）から
`vllm==0.32.0` を **ネットワーク無し** で入れられるかを確かめる。

測るもの:
- install 秒数（GPU 版は 504 秒だった）
- torch / torchaudio の CUDA 版数一致
- `pip check` の結果
- `vllm serve --help=all` の可否（CPU では失敗する可能性がある）
""")

code(r"""
import os
import shutil
import subprocess
import sys
import time

t_start = time.time()
print("python :", sys.version.split()[0])
print("pip    :", subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True).stdout.strip())

# Dataset のマウント先を探す（/kaggle/input/<slug> と /datasets/<owner>/<slug> の両方）
import glob
import zipfile

cands = []
for pat in ("/kaggle/input/vllm-wheelhouse", "/kaggle/input/*/vllm-wheelhouse",
            "/kaggle/input/*/*/vllm-wheelhouse"):
    cands += [p for p in glob.glob(pat) if os.path.isdir(p)]
WHL = sorted(cands)[0] if cands else ""
print("wheelhouse:", WHL or "(見つからない)")
if not WHL:
    raise SystemExit("Dataset がマウントされていません")

# --dir-mode zip で作ると /kaggle/input には *.zip が 1 個だけ置かれる。
# その場合は /kaggle/tmp へ 1 回だけ展開する（install より速い）。
def _wheels(root):
    w = glob.glob(os.path.join(root, "*.whl"))
    if w:
        return w
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(sub):
            w = glob.glob(os.path.join(sub, "*.whl"))
            if w:
                return w
    return []

wheels = _wheels(WHL)
if not wheels:
    zips = glob.glob(os.path.join(WHL, "*.zip")) + glob.glob(os.path.join(WHL, "*", "*.zip"))
    print("zip を検出:", [os.path.basename(z) for z in zips])
    dst = "/kaggle/tmp/whl"
    os.makedirs(dst, exist_ok=True)
    t_unzip = time.time()
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
    print(f"展開完了 {time.time()-t_unzip:.1f}s -> {dst}")
    wheels = _wheels(dst) or glob.glob(os.path.join(dst, "**", "*.whl"), recursive=True)

if not wheels:
    raise SystemExit("wheel が見つかりません（Dataset の中身を確認）")
WHL = os.path.dirname(wheels[0])
n_whl = len(wheels)
print("wheel 置き場:", WHL)
print("wheel 数   :", n_whl)
print("サイズ     :", round(sum(os.path.getsize(p) for p in wheels) / 1e9, 2), "GB")
u = shutil.disk_usage("/kaggle/working")
print("working 空き:", round(u.free / 1e9, 1), "GB")
""")

code(r"""
# 依存が閉じているかを先に確認する（インストールせずに解決だけ）
t0 = time.time()
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index",
     "--find-links", WHL, "vllm==0.32.0"],
    capture_output=True, text=True, timeout=1800,
)
print("dry-run exit:", r.returncode, f"({time.time()-t0:.1f}s)")
DRY_RC = r.returncode
DRY_SEC = round(time.time() - t0, 1)
tail = ((r.stdout or "") + (r.stderr or ""))[-2500:]
print(tail)
if r.returncode != 0:
    print("!!! 解決に失敗（wheelhouse 不足）")
""")

code(r"""
# 本番インストール（GPU 版と同じコマンド + --no-index）
t0 = time.time()
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-input",
     "--no-deps", "--no-index", "--find-links", WHL,
     "vllm==0.32.0"] + glob.glob(os.path.join(WHL, "*.whl")),
    capture_output=True, text=True, timeout=3600,
)
INSTALL_SEC = time.time() - t0
INSTALL_RC = r.returncode
print(f"install exit: {r.returncode}  所要 {INSTALL_SEC:.1f} 秒 "
      f"({INSTALL_SEC/60:.1f} 分)")
print(((r.stdout or "") + (r.stderr or ""))[-2000:])
""")

code(r"""
# 検証: import / 版数 / torchaudio 整合 / pip check
import subprocess, sys

def run(code, tmo=300):
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=tmo)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()

rc_imp, out_imp = run("import vllm; print('vllm', vllm.__version__)", 600)
print("import vllm:", rc_imp, out_imp[-300:])

rc_t, out_t = run("import torch; print('torch', torch.__version__, "
                  "'cuda', torch.version.cuda)")
print(rc_t, out_t)

# 本番と同じ torchaudio 整合チェック（PyPI 版から torch が差し替わるため）
rc_a, out_a = run("import torchaudio; print('torchaudio', torchaudio.__version__)")
print("torchaudio:", rc_a, out_a[-400:])
TA_FIX_SEC = 0.0
if rc_a != 0:
    t1 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "--no-index", "--find-links", WHL,
                    "--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
                   capture_output=True, text=True, timeout=1800)
    TA_FIX_SEC = round(time.time() - t1, 1)
    rc_a, out_a = run("import torchaudio; print('torchaudio', torchaudio.__version__)")
    print(f"torchaudio 再導入 {TA_FIX_SEC}s ->", rc_a, out_a[-300:])

rc_v, out_v = run("import torch, triton; print('torch', torch.__version__, "
                  "'cuda', torch.version.cuda, 'triton', triton.__version__)")
print("versions:", rc_v, out_v)

rc_c, out_c = run("import subprocess,sys;"
                  "print(subprocess.run([sys.executable,'-m','pip','check'],"
                  "capture_output=True,text=True).stdout)")
print("pip check:\n", out_c[:2000])
""")

code(r"""
# vllm CLI（CPU では失敗する想定。どこで落ちるか記録する）
import subprocess, sys, time

import shutil as _sh
_bin = _sh.which("vllm")
print("vllm bin:", _bin or "(PATH に無い → -m で代替)")
_exe = [_bin] if _bin else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
t0 = time.time()
p = subprocess.run(_exe + ["serve", "--help=all"], capture_output=True,
                   text=True, timeout=600)
HELP = (p.stdout or "") + (p.stderr or "")
print(f"--help=all: exit={p.returncode} {len(HELP)} 文字 "
      f"({time.time()-t0:.1f}s)")
print(HELP[:800])
complete = (len(HELP) > 5000 and "--tensor-parallel-size" in HELP
            and "--max-model-len" in HELP)
print("complete(個別フラグ入り):", complete)
""")

code(r"""
# 結果を保存
import json, time

res = {
    "wheel_count": n_whl,
    "wheelhouse": WHL,
    "dry_run_rc": DRY_RC,
    "dry_run_sec": DRY_SEC,
    "install_rc": INSTALL_RC,
    "install_sec": round(INSTALL_SEC, 1),
    "import_rc": rc_imp,
    "torchaudio_rc": rc_a,
    "torchaudio_fix_sec": TA_FIX_SEC,
    "versions": out_v,
    "help_rc": p.returncode,
    "help_chars": len(HELP),
    "help_complete": complete,
    "total_sec": round(time.time() - t_start, 1),
}
txt = json.dumps(res, ensure_ascii=False, indent=2)
for d in ("/kaggle/working", "/output"):
    try:
        with open(f"{d}/whl_result.json", "w") as f:
            f.write(txt + "\n")
    except Exception as e:
        print("保存失敗", d, e)
print(txt)
print()
print(f"総所要 {res['total_sec']}s / install {res['install_sec']}s"
      f" → GPU 版 504s と比較")
print("WHL_PROBE_COMPLETE")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "whl-probe.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
meta = {
    "id": SLUG,
    "title": "whl probe",
    "code_file": "whl-probe.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [WH],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", OUT / "whl-probe.ipynb")

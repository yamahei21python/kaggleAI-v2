#!/usr/bin/env python3
"""CPU 網羅検証 修正版 (cpu-probe13) — 全データセットパス自動探索 & 強制検出付き。"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
SLUG = "cpu-probe13"

CODE = r'''import os
import sys
import subprocess
import time
import glob
import compileall

print("============================================================")
print(" CPU 網羅検証 13 (データセット全探索 & 10項目網羅)")
print("============================================================")
print(f"Python: {sys.version}")

def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0.0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

site_pkgs = "/usr/local/lib/python3.13/site-packages"
dist_pkgs = "/usr/local/lib/python3.13/dist-packages"
target_dir = site_pkgs if os.path.isdir(site_pkgs) else dist_pkgs

print(f"Target site-packages: {target_dir}")
init_size = get_dir_size(target_dir)
print(f"1. インストール前 site-packages 容量: {init_size:.2f} MB")

print("\n=== データセット全階層探索 == /kaggle/input ===")
all_whls = glob.glob("/kaggle/input/**/*.whl", recursive=True)
print(f"発見された全 .whl ファイル数: {len(all_whls)}")
for w in all_whls[:10]:
    print("  ", w)

whl_dirs = sorted(list(set(os.path.dirname(w) for w in all_whls)))
print("wheelhouse ディレクトリ一覧:", whl_dirs)

print("\n=== 2. 二回連続インストール & 所要時間計測 ===")
for trial in (1, 2):
    t0 = time.time()
    if all_whls:
        # ディレクトリごと -f 指定、または直接全 whl ファイルを指定
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-U"] + all_whls
    else:
        print("⚠ whl ファイル未発見のため PyPI 試行 (オフライン時失敗想定)")
        cmd = [sys.executable, "-m", "pip", "install", "vllm"]
    
    print(f"--- インストール試行 {trial} ---")
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    print(f"  所要時間: {dt:.2f} 秒 (rc={r.returncode})")
    if r.returncode != 0:
        print("  Stderr 抜粋:", "\n".join((r.stderr or "").strip().splitlines()[-10:]))
    else:
        print("  Stdout 抜粋:", "\n".join((r.stdout or "").strip().splitlines()[-5:]))

post_size = get_dir_size(target_dir)
print(f"9. インストール後容量: {post_size:.2f} MB (差分: +{post_size - init_size:.2f} MB)")

print("\n=== 5. パッケージ出所・バージョン確認 ===")
r_list = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True)
lines = [line for line in (r_list.stdout or "").splitlines() if any(k in line.lower() for k in ("vllm", "torch", "triton", "transformers"))]
print("\n".join(lines) if lines else "(該当パッケージなし)")

print("\n=== 3. インストール済みパッケージ限定 compileall ===")
for pkg in ("vllm", "torch", "triton", "transformers"):
    p = os.path.join(target_dir, pkg)
    if os.path.isdir(p):
        res = compileall.compile_dir(p, quiet=1, force=True)
        print(f"  {pkg}: compileall {'OK' if res else 'NG'}")
    else:
        print(f"  {pkg}: site-packages に存在しません")

print("\n=== 4 & 7. インストール済み全 .so の ldd チェック ===")
all_sos = glob.glob(os.path.join(target_dir, "**/*.so"), recursive=True)
print(f"対象 .so 総数: {len(all_sos)}")
missing_libs = set()
for so in all_sos:
    try:
        res_ldd = subprocess.run(["ldd", so], capture_output=True, text=True, timeout=3)
        for line in (res_ldd.stdout or "").splitlines():
            if "not found" in line:
                lib_name = line.split("=>")[0].strip()
                missing_libs.add((os.path.basename(so), lib_name, line.strip()))
    except Exception:
        pass

print(f"未解決共有ライブラリ (not found): {len(missing_libs)} 件")
cufile_only = True
for so_name, lib_name, detail in list(missing_libs):
    print(f"  [{so_name}] -> {detail}")
    if "libcufile" not in detail:
        cufile_only = False

print(f"7. .so 不足は libcufile のみか: {'YES' if cufile_only and missing_libs else ('NO' if missing_libs else 'N/A (未解決なし)')}")

print("\n=== 6. 別 Python プロセスでの import 検証 ===")
test_code = [
    "import sys",
    "for m in ('torch', 'transformers', 'triton', 'vllm'):",
    "    try:",
    "        mod = __import__(m)",
    "        print(f'IMPORT_OK:{m} ({getattr(mod, \"__version__\", \"unknown\")})')",
    "    except Exception as e:",
    "        print(f'IMPORT_NG:{m}: {type(e).__name__}: {e}')"
]
r_imp = subprocess.run([sys.executable, "-c", "\n".join(test_code)], capture_output=True, text=True)
print(r_imp.stdout.strip())
if r_imp.stderr.strip():
    print("Stderr:", r_imp.stderr.strip()[:500])

print("\n=== 10. pip check 新規差分 ===")
r_chk = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
print(r_chk.stdout.strip() or "(pip check clean)")

print("\n============================================================")
print(" 全 10 項目検証完了")
print("============================================================")
'''

def _write():
    d = HERE / SLUG
    d.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "m1", "metadata": {}, "source": ["# CPU 網羅検証 13 (データセット全探索 & 10項目網羅)\n"]},
            {"cell_type": "code", "execution_count": None, "id": "c1", "metadata": {}, "source": CODE.splitlines(keepends=True)}
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    (d / f"{SLUG}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    
    meta = {
        "id": f"{OWNER}/cpu-12-10",
        "title": "CPU 網羅検証 13 (データセット全探索 & 10項目網羅)",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [
            "yamahei21python/vllm-wheelhouse",
            "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {d}")

if __name__ == "__main__":
    _write()

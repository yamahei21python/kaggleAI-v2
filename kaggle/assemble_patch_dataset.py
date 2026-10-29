#!/usr/bin/env python3
"""capture v2 の tar.gz から patch Dataset ソースを組み立てる。

入力: /tmp/cap2/ollama_patch.tar.gz  (bin/ollama, lib/ollama/**, VERSION.txt)
既存: /tmp/patchold/blobs/sha256-cbe6ad4...  (config blob 214B, model dataset に無い)
出力: <repo>/kaggle-dataset-orcarouter-patch/{bin,lib,blobs,dataset-metadata.json}

安全策: /tmp に組み立てて成功後に置換（失敗で旧資産を消さない）。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile

REPO = "/Users/kohei/Myproject/kaggleAI"
DST = os.path.join(REPO, "kaggle-dataset-orcarouter-patch")
TAR = "/tmp/cap2/ollama_patch.tar.gz"
OLD = "/tmp/patchold"
BUILD = "/tmp/patch_build"
CONFIG_BLOB_PREFIX = "sha256-cbe6ad4"  # manifest が要求する config blob（214B）

if not os.path.isfile(TAR):
    sys.exit(f"missing {TAR}")

# --- tar.gz 展開（traversal 検査つき）---
tmp = "/tmp/cap2x"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
with tarfile.open(TAR) as t:
    for m in t.getmembers():
        p = os.path.realpath(os.path.join(tmp, m.name))
        if not p.startswith(os.path.realpath(tmp) + os.sep):
            sys.exit(f"unsafe tar member: {m.name}")
        if m.issym() or m.islnk():
            sys.exit(f"unexpected link in tar: {m.name}")
    t.extractall(tmp)

root = os.path.join(tmp, "ollama_patch")
if not os.path.isdir(root):
    cand = [d for d in glob.glob(os.path.join(tmp, "*")) if os.path.isdir(d)]
    if len(cand) != 1:
        sys.exit(f"cannot locate extracted root: {cand}")
    root = cand[0]

ver_file = os.path.join(root, "VERSION.txt")
if not os.path.isfile(ver_file):
    sys.exit("VERSION.txt missing in archive")
ver = open(ver_file).read().strip()
# `ollama --version` はサーバ未起動だと警告2行を出す。version 行だけ残す。
ver = next((ln.strip() for ln in ver.splitlines() if "version" in ln.lower()
            and "client" in ln.lower()), ver.splitlines()[-1].strip())
# "Warning: client version is 0.40.12" → "0.40.12"
if " is " in ver:
    ver = ver.rsplit(" is ", 1)[1].strip()

# --- 既存 config blob（manifest が要求する 214B）---
if not os.path.isdir(OLD):
    os.makedirs(OLD, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch",
                    "-p", OLD, "--unzip"], check=True)
blobs = [b for b in glob.glob(os.path.join(OLD, "**", "blobs", "*"), recursive=True)
         if os.path.isfile(b)
         and os.path.basename(b).startswith(CONFIG_BLOB_PREFIX)
         and os.path.getsize(b) < 1024]
if len(blobs) != 1:
    sys.exit(f"config blob not uniquely found: {blobs}")

# --- /tmp に組み立て ---
shutil.rmtree(BUILD, ignore_errors=True)
os.makedirs(os.path.join(BUILD, "bin"))
os.makedirs(os.path.join(BUILD, "blobs"))
shutil.copy2(os.path.join(root, "bin", "ollama"), os.path.join(BUILD, "bin", "ollama"))
shutil.copytree(os.path.join(root, "lib", "ollama"), os.path.join(BUILD, "lib", "ollama"))
shutil.copy2(blobs[0], os.path.join(BUILD, "blobs", os.path.basename(blobs[0])))

# llama-server は lib/ollama 直下（install.sh と同じ配置）。build_step2 は
# lib 優先で探索し、lib walk がそのまま /usr/local/lib/ollama へ複製する。
ls = os.path.join(BUILD, "lib", "ollama", "llama-server")
cuda = os.path.join(BUILD, "lib", "ollama", "cuda_v12", "libggml-cuda.so")
if not os.path.isfile(ls):
    sys.exit("lib/ollama/llama-server missing in archive")
if not os.path.isfile(cuda):
    sys.exit("lib/ollama/cuda_v12/libggml-cuda.so missing in archive")

# cuda_v12 の完全版番号ライブラリ（libcublasLt.so.12.8.5.5 等）は install.sh では
# ハードリンク/シンボリックリンク。展開で実体が二重化するので落とす（約870MB）。
# ローダは SONAME（.so.12）を見るため実害なし。
pruned = []
cud = os.path.join(BUILD, "lib", "ollama", "cuda_v12")
for fn in sorted(os.listdir(cud)):
    if fn.startswith(("libcublasLt.so.", "libcublas.so.", "libcudart.so.")) and fn.count(".") >= 3:
        p = os.path.join(cud, fn)
        pruned.append(f"{fn} ({os.path.getsize(p):,} B)")
        os.remove(p)

meta = {
    "title": "Ollama Qwen3.8-27B Runtime Patch",
    "id": "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch",
    "subtitle": "ollama + llama-server + CUDA libs + config blob",
    "description": (
        "Ollama runtime for Step2 (Qwen3.8-27B). Contents: bin/ollama, "
        "lib/ollama/ (llama-server, *.so, cuda_v12/*, licenses), "
        "blobs/sha256-cbe6ad4... (config blob required by the model manifest; not "
        "present in the model dataset). With llama-server included, Step2 no longer "
        "runs install.sh, so the ollama version is pinned to this capture. "
        f"Version: {ver}"
    ),
    "licenses": [{"name": "other"}],
}
with open(os.path.join(BUILD, "dataset-metadata.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# --- 成功後に置換 ---
old_dst = DST + ".old"
shutil.rmtree(old_dst, ignore_errors=True)
if os.path.isdir(DST):
    os.replace(DST, old_dst)
os.replace(BUILD, DST)
shutil.rmtree(old_dst, ignore_errors=True)

total = sum(os.path.getsize(os.path.join(r, fn))
            for r, _d, fs in os.walk(DST) for fn in fs)
ls = os.path.join(DST, "lib", "ollama", "llama-server")
cuda = os.path.join(DST, "lib", "ollama", "cuda_v12", "libggml-cuda.so")
print(f"assembled {DST}")
print(f"  bin: {sorted(os.listdir(os.path.join(DST, 'bin')))}")
print(f"  blobs: {sorted(os.listdir(os.path.join(DST, 'blobs')))}")
print(f"  llama-server: {os.path.getsize(ls):,} B")
print(f"  cuda: {os.path.getsize(cuda):,} B")
print(f"  total: {total/1e9:.2f} GB")
print(f"  pruned: {len(pruned)} files")
for x in pruned:
    print(f"    - {x}")
print(f"  version: {ver}")

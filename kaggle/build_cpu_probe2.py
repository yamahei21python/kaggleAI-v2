#!/usr/bin/env python3
"""CPU 空き枠で回す追加テスト 2 本。

  1. step2-vllm-wheelset  … wheels を落として **オフライン install の所要**を測る（施策 B の実証）
  2. step2-vllm-mtpshape  … Dataset の config.json / safetensors ヘッダから MTP 重みの形を見る

どちらも GPU 不要・クォータ消費ゼロ。

    python3 kaggle/build_cpu_probe2.py
    cd kaggle/step2-vllm-wheelset && kaggle kernels push -p .
    cd kaggle/step2-vllm-mtpshape && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
VLLM_SPEC = "vllm==0.32.0"


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


WHEELSET = '''# 施策 B の実証: wheels を落として **オフライン install** を測る
# GPU 不要。Kaggle は 1 セッション使い捨てなので、torch を外して cold を再現する。
import glob
import json
import os
import subprocess
import sys
import time

SPEC = "vllm==0.32.0"
WD = "/kaggle/working/wheels"
os.makedirs(WD, exist_ok=True)


def run(cmd, label, timeout=5400):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[{label}] {dt:.1f}s rc={r.returncode}")
    if r.returncode != 0:
        print(((r.stdout or "") + (r.stderr or ""))[-1200:])
    return dt, r


res = {}

# 1. 取得（Dataset 化する中身）
dt, _ = run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
             SPEC, "-d", WD], "download")
files = glob.glob(WD + "/*")
res["download_sec"] = dt
res["files"] = len(files)
res["bytes"] = sum(os.path.getsize(f) for f in files)
print(f"  {res['bytes']/1e9:.2f} GB / {res['files']} files")
print(sorted(os.path.basename(f) for f in files)[:15])

# 2. cold 再現のため重い所を外す（このセッションは捨てる前提）
run([sys.executable, "-m", "pip", "uninstall", "-y", "vllm", "torch"],
    "uninstall")

# 3. pip オフライン install
dt, r = run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
             "--no-index", "--find-links", WD, SPEC], "pip offline")
res["pip_offline_sec"] = dt
res["pip_offline_ok"] = r.returncode == 0

# 4. uv オフライン install（比較）
run([sys.executable, "-m", "pip", "uninstall", "-y", "vllm"], "uninstall vllm")
run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "uv"],
    "install uv")
dt, r = run(["uv", "pip", "install", "--system", "--no-index",
             "--find-links", WD, "-q", SPEC], "uv offline")
res["uv_offline_sec"] = dt
res["uv_offline_ok"] = r.returncode == 0

# 5. 復元確認
r = subprocess.run([sys.executable, "-c",
                    "import vllm, torch;"
                    "print(vllm.__version__, torch.__version__)"],
                   capture_output=True, text=True)
print("import:", r.returncode, (r.stdout or "").strip(),
      (r.stderr or "")[-300:])
res["import_ok"] = r.returncode == 0

# 成果物に 4 GB を含めない（取得に時間がかかるだけ）。数値だけ残す。
import shutil
shutil.rmtree(WD, ignore_errors=True)

with open("/kaggle/working/wheelset.json", "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
'''

MTPSHAPE = '''# Dataset の中身から MTP ヘッドの形を確認（10 秒）
# `method=mtp` が汎用指定で拾えるか、draft model 指定が要るかの判断材料。
import json
import os
import struct
import sys

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
print("exists:", os.path.isdir(ROOT))
if not os.path.isdir(ROOT):
    cands = [p for p in ("/kaggle/input/qwen38-27b-uncensored-hauhaucs-awq-mtp",
                         "/kaggle/input/datasets/yamahei21python/"
                         "qwen38-27b-uncensored-hauhaucs-awq-mtp") if os.path.isdir(p)]
    print("cands:", cands)
    if cands:
        ROOT = cands[0]

cfg = json.load(open(os.path.join(ROOT, "config.json")))
print("architectures:", cfg.get("architectures"))
print("model_type   :", cfg.get("model_type"))
print("layers       :", cfg.get("num_hidden_layers"))
print("hidden       :", cfg.get("hidden_size"))
print("quantization :", cfg.get("quantization_config"))
print("max_pos      :", cfg.get("max_position_embeddings"))
print("tie_embed    :", cfg.get("tie_word_embeddings"))


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace"))


for name in ("model-mtp.safetensors", "model.safetensors"):
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        print("missing:", name)
        continue
    h = header(p)
    keys = [k for k in h if k != "__metadata__"]
    print(f"\\n== {name}: {len(keys)} tensors ==")
    print("metadata:", h.get("__metadata__"))
    # 層番号の分布（MTP が何層目に足されているか）
    layers = sorted({int(k.split(".")[2]) for k in keys
                     if k.startswith("model.layers.") and k.split(".")[2].isdigit()})
    print("layer ids:", layers[:5], "...", layers[-5:] if layers else None,
          "count:", len(layers))
    print("first keys:", keys[:12])
    if "mtp" in name:
        print("ALL MTP KEYS:")
        for k in keys:
            print("  ", k, h[k].get("shape"), h[k].get("dtype"))

gen = os.path.join(ROOT, "generation_config.json")
if os.path.isfile(gen):
    print("\\ngeneration_config:", json.load(open(gen)))
'''

if __name__ == "__main__":
    _write("step2-vllm-wheelset", "Step2 vLLM wheelset",
           _nb("# 📦 wheels を落としてオフライン install を測る（施策 B の実証）\n\n"
               "3.94 GB / 198 files を Dataset 化した場合の install 所要を実測。\n",
               WHEELSET), [])
    _write("step2-vllm-mtpshape", "Step2 vLLM mtpshape",
           _nb("# 🧬 MTP ヘッドの形を safetensors ヘッダから確認\n\n"
               "`method=mtp` の汎用指定で拾えるかの判断材料。\n",
               MTPSHAPE), [DATASET_ID])

#!/usr/bin/env python3
"""CPU 網羅テスト 追加分（その 5）。

  1. step2-vllm-compat  … vLLM wheel と「このモデル/設定」の互換性を静的に確認
                          （依存ピン・arch 対応・reasoning parser 名・kv-cache-dtype 値）
  2. step2-vllm-cfgdeep … Dataset の設定ファイルを深掘り（recipe.yaml / template 差分 / auto_map）

どちらも GPU 不要・数分。

    python3 kaggle/build_cpu_probe5.py
    cd kaggle/step2-vllm-compat && kaggle kernels push -p .
    cd kaggle/step2-vllm-cfgdeep && kaggle kernels push -p .
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


COMPAT = '''# vLLM 0.32 wheel × このモデル/設定 の互換性を静的確認（install 不要）
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
cfg = json.load(open(os.path.join(ROOT, "config.json")))
ARCH = (cfg.get("architectures") or ["<none>"])[0]
MODEL_TYPE = cfg.get("model_type")
print("architectures:", ARCH, "| model_type:", MODEL_TYPE)

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

# --- 1. 依存ピン（transformers / torch の要求）---
meta_name = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]
meta = z.read(meta_name).decode("utf-8", "replace")
print("\\n=== Requires-Dist（主要のみ）===")
for line in meta.splitlines():
    if line.startswith("Requires-Dist:"):
        low = line.lower()
        if any(k in low for k in ("torch", "transformers", "tokenizers",
                                  "safetensors", "numpy", "huggingface")):
            print("  ", line[len("Requires-Dist: "):][:140])
print("Requires-Python:", [l for l in meta.splitlines()
                           if l.startswith("Requires-Python")])

# --- 2. モデルアーキ対応 ---
reg = root / "model_executor/models/registry.py"
txt = reg.read_text(errors="replace") if reg.exists() else ""
print("\\n=== arch 対応 ===")
print("registry exists:", reg.exists())
print("arch in registry:", ARCH in txt)
for m in re.finditer(r'"([A-Za-z0-9_]+ForCausalLM)"\\s*:\\s*"([^"]+)"', txt):
    if "Qwen" in m.group(1):
        print("  ", m.group(1), "->", m.group(2))

# --- 3. reasoning parser 名（--reasoning-parser qwen3 が有効か）---
print("\\n=== reasoning parser 名 ===")
rp = list(root.rglob("reasoning/*.py")) + list(root.rglob("reasoning*.py"))
names = set()
for f in rp:
    t = f.read_text(errors="replace")
    for m in re.finditer(r'register_reasoning_parser\\(["\\']([^"\\']+)', t):
        names.add(m.group(1))
print("registered:", sorted(names))
print("qwen3 available:", "qwen3" in names)
for f in rp:
    if "qwen3" in f.name:
        print("  file:", f.relative_to(root))

# --- 4. kv-cache-dtype の取り得る値 ---
print("\\n=== kv-cache-dtype choices ===")
cache = (root / "config/cache.py").read_text(errors="replace")
m = re.search(r"CacheDType\\s*=\\s*Literal\\[(.*?)\\]", cache, re.S)
print(m.group(0)[:400] if m else "not found")

# --- 5. speculative の mpt 検証（method=mtp に model が要るか）---
print("\\n=== speculative: mtp 分岐 ===")
sp = (root / "config/speculative.py").read_text(errors="replace")
for i, line in enumerate(sp.splitlines(), 1):
    s = line.strip()
    if "mtp" in s.lower() and ("method" in s or "if " in s or "assert" in s
                               or "raise" in s):
        print(f"  {i}: {s[:170]}")

# --- 6. 量子化 awq のサポート条件（group_size 等）---
print("\\n=== awq 条件 ===")
for f in root.rglob("quantization/awq*.py"):
    t = f.read_text(errors="replace")
    for i, line in enumerate(t.splitlines(), 1):
        s = line.strip()
        if ("group_size" in s or "bits" in s or "zero_point" in s) and (
                "if " in s or "raise" in s or "==" in s):
            print(f"  {f.name}:{i}: {s[:150]}")
'''


CFGDEEP = '''# Dataset の設定ファイル深掘り（30 秒）
import json
import os

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
cfg = json.load(open(os.path.join(ROOT, "config.json")))
print("keys:", sorted(cfg.keys()))
print("\\nquantization_config:")
print(json.dumps(cfg.get("quantization_config"), indent=2, ensure_ascii=False))
print("\\nauto_map:", cfg.get("auto_map"))
print("rope:", cfg.get("rope_scaling"), "| rope_theta:", cfg.get("rope_theta"))
print("sliding_window:", cfg.get("sliding_window"),
      "| layer_types:", (cfg.get("layer_types") or [])[:6])
print("num_experts:", cfg.get("num_experts"),
      "| moe_intermediate:", cfg.get("moe_intermediate_size"))
print("intermediate_size:", cfg.get("intermediate_size"))

# tokenizer_config の chat_template と jinja ファイルの一致
tk = json.load(open(os.path.join(ROOT, "tokenizer_config.json")))
embedded = tk.get("chat_template")
print("\\ntokenizer_config chat_template:", type(embedded).__name__,
      len(embedded) if isinstance(embedded, str) else embedded)
tpl_files = ["chat_template.jinja", "chat_template_medium.jinja"]
for name in tpl_files:
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        continue
    body = open(p, encoding="utf-8").read()
    print(f"{name}: {len(body)} chars | == tokenizer_config: "
          f"{body.strip() == (embedded or '').strip()}")
    for kw in ("reasoning_effort", "thinking", "tool_call", "xhigh"):
        print(f"   contains {kw}:", kw in body)

print("\\ntokenizer_config keys:", sorted(tk.keys()))
print("bos/eos:", tk.get("bos_token"), tk.get("eos_token"),
      "| model_max_length:", tk.get("model_max_length"))

# recipe.yaml（vendor の変換情報）
p = os.path.join(ROOT, "recipe.yaml")
if os.path.isfile(p):
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    print(f"\\nrecipe.yaml: {len(lines)} lines")
    for line in lines[:40]:
        print("  ", line[:150])

# 余計なファイルが vLLM を混乱させないか
print("\\nall files:")
for f in sorted(os.listdir(ROOT)):
    fp = os.path.join(ROOT, f)
    print(f"  {f:35s} {'DIR' if os.path.isdir(fp) else os.path.getsize(fp)}")
'''

if __name__ == "__main__":
    _write("step2-vllm-compat", "Step2 vLLM compat",
           _nb("# 🧩 vLLM 0.32 × モデル/設定の互換性（install 不要）\n\n"
               "依存ピン・arch 対応・reasoning parser 名・kv-cache 値域・awq 条件。\n",
               COMPAT), [DATASET_ID])
    _write("step2-vllm-cfgdeep", "Step2 vLLM cfgdeep",
           _nb("# 🗂 Dataset 設定ファイルの深掘り\n\n"
               "量子化設定・template の実体・auto_map・recipe.yaml。\n",
               CFGDEEP), [DATASET_ID])

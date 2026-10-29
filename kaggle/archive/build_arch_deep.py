#!/usr/bin/env python3
"""`architectures` と text_config を fs から読むだけ（10 秒・dataset のみ）。

- vLLM が対応する arch か
- ハイブリッド線形注意（linear_attn）か
- MTP 層数（mtp_num_hidden_layers）が config にあるか

    python3 kaggle/build_arch_deep.py
    cd kaggle/step2-vllm-archdeep && kaggle kernels push -p .
"""
import json
import pathlib

OWNER = "yamahei21python"
SLUG = "step2-vllm-archdeep"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / SLUG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE = '''# architectures / text_config の中身（データセットのみ・10 秒）
import json
import os

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
cfg = json.load(open(os.path.join(ROOT, "config.json")))

print("architectures        :", cfg.get("architectures"))
print("model_type           :", cfg.get("model_type"))
print("language_model_only  :", cfg.get("language_model_only"))
print("dtype                :", cfg.get("dtype"))
print("transformers_version :", cfg.get("transformers_version"))
print("vision_config?       :", bool(cfg.get("vision_config")))
print("tie_word_embeddings  :", cfg.get("tie_word_embeddings"))

tc = cfg.get("text_config") or {}
print("\\n=== text_config keys ===")
print(sorted(tc.keys()))
for k in ("model_type", "num_hidden_layers", "hidden_size",
          "num_attention_heads", "num_key_value_heads", "head_dim",
          "num_experts", "n_routed_experts", "moe_intermediate_size",
          "intermediate_size", "max_position_embeddings",
          "mtp_num_hidden_layers", "full_attention_interval",
          "linear_attn_config", "layer_types", "mlp_only_layers",
          "shared_expert_intermediate_size", "num_nextn_predict_layers"):
    if k in tc:
        v = tc[k]
        if isinstance(v, (list, dict)):
            print(f"  {k}: {str(v)[:300]}")
        else:
            print(f"  {k}: {v}")

vc = cfg.get("vision_config") or {}
print("\\n=== vision_config (抜粋) ===")
for k, v in list(vc.items())[:15]:
    print(f"  {k}: {str(v)[:120]}")

print("\\nimage_token_id:", cfg.get("image_token_id"),
      "| vision_start:", cfg.get("vision_start_token_id"),
      "| vision_end:", cfg.get("vision_end_token_id"))

# safetensors の先頭キーから「実際の層構成」を見る（10 秒）
import struct


def head(path, n=40):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(size).decode("utf-8", "replace"))
    keys = [k for k in h if k != "__metadata__"]
    return keys

keys = head(os.path.join(ROOT, "model-mtp.safetensors"))
print("\\nMTP tensor count:", len(keys))
for k in keys[:40]:
    print("  ", k)
print("MTP prefixes:", sorted({k.split(".")[0] + "." + (k.split(".")[1] if "." in k else "") for k in keys})[:10])
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "id": "cell00", "metadata": {},
         "source": ["# 🧬 architectures / text_config / MTP の実体\n",
                    "\n",
                    "vLLM の対応 arch と、linear_attn / MoE / MTP の構成を確認。\n"]},
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

meta = {"id": f"{OWNER}/{SLUG}", "title": "Step2 vLLM archdeep",
        "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [],
        "dataset_sources": ["yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT_DIR}")

#!/usr/bin/env python3
"""CPU 網羅テスト 第 11 弾 — draft 専用ディレクトリ作戦の検証（GPU 不使用）。

背景: draft の `ModelConfig.model` は target と同じディレクトリになり、
loader は `*.safetensors` を glob するため **17.41 GiB の本体まで読む**（§27）。
`--speculative-config` の `model` に「model-mtp.safetensors + config 類だけ」を
置いた専用ディレクトリを渡せば 0.79 GiB で済む。成立するかを実測する。

  1. step2-vllm-draftdir … symlink で draft dir を作り、AutoConfig / ModelConfig /
                           SpeculativeConfig が通るか、glob が 1 ファイルになるか

    python3 kaggle/build_cpu_probe11.py
    bash kaggle/push_probe11.sh
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
SLUGS = ["draftdir"]

ROOT_FINDER = '''def find_root():
    import glob
    slug = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
    ordered = []
    for pat in ("/kaggle/input/*/config.json", "/kaggle/input/*/*/config.json",
                "/kaggle/input/*/*/*/config.json"):
        ordered += glob.glob(pat)
    cands = ([c for c in ordered if slug in c]
             or [c for c in ordered if "qwen" in c.lower()]
             or ordered)
    print("config.json 候補:", cands[:3])
    if not cands:
        print("/kaggle/input 直下:", glob.glob("/kaggle/input/*")[:10])
        raise SystemExit("dataset が見つからない")
    return cands[0].rsplit("/", 1)[0]
'''

INSTALL_BOOT = '''import subprocess
import sys

print("=== pip install vllm==0.32.0 ===")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "vllm==0.32.0"], capture_output=True, text=True, timeout=5400)
tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
print("   rc:", r.returncode, "|", " / ".join(tail[-3:])[:300])
'''


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


def _write(slug, title, md, code, datasets):
    d = HERE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(
        json.dumps(_nb(md, code), ensure_ascii=False, indent=1), encoding="utf-8")
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


DRAFTDIR = INSTALL_BOOT + ROOT_FINDER + r'''
import glob
import os
import shutil

ROOT = find_root()
DRAFT = "/kaggle/working/draft"
MTP_FILE = "model-mtp.safetensors"

print("=== dataset の中身（safetensors と config 類）===")
for fn in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, fn)
    if os.path.isfile(p) and (fn.endswith(".safetensors")
                              or fn.startswith(("config", "tokenizer",
                                                "chat_template", "generation",
                                                "special_tokens", "vocab",
                                                "merges", "added_tokens"))):
        print(f"  {fn}: {os.path.getsize(p) / 2**20:.1f} MiB")

print("\n=== draft dir を作る（symlink のみ）===")
if os.path.isdir(DRAFT):
    shutil.rmtree(DRAFT)
os.makedirs(DRAFT, exist_ok=True)
KEEP = ("config", "tokenizer", "chat_template", "generation_config",
        "special_tokens_map", "vocab", "merges", "added_tokens")
n = 0
for fn in sorted(os.listdir(ROOT)):
    if fn.startswith("."):
        continue
    if fn != MTP_FILE and not fn.startswith(KEEP):
        continue
    src = os.path.join(ROOT, fn)
    if not os.path.isfile(src):
        continue
    os.symlink(src, os.path.join(DRAFT, fn))
    n += 1
print(f"  symlink {n} 個")

print("\n=== glob の結果（ここが 1 ファイルなら狙い通り）===")
found = sorted(glob.glob(os.path.join(DRAFT, "*.safetensors")))
print("  *.safetensors:", [os.path.basename(x) for x in found])
tot = sum(os.path.getsize(x) for x in found)
print(f"  合計 {tot / 2**30:.2f} GiB")
print("  期待: 0.79 GiB（本体 17.41 GiB を読まない）")

print("\n=== AutoConfig（draft dir から）===")
from transformers import AutoConfig
try:
    hf = AutoConfig.from_pretrained(DRAFT, trust_remote_code=True)
    print("  model_type:", getattr(hf, "model_type", None))
    print("  architectures:", getattr(hf, "architectures", None))
    print("  mtp_num_hidden_layers:", getattr(hf, "mtp_num_hidden_layers", None))
    tc = getattr(hf, "text_config", None)
    print("  text_config.mtp_num_hidden_layers:",
          getattr(tc, "mtp_num_hidden_layers", None) if tc else None)
except Exception as e:
    print("  [NG]", type(e).__name__, str(e)[:300])

print("\n=== ModelConfig(draft dir) ===")
from vllm.config import ModelConfig, SpeculativeConfig
from vllm.config.parallel import ParallelConfig

d = None
try:
    d = ModelConfig(model=DRAFT, dtype="half", max_model_len=8192,
                    quantization="compressed-tensors",
                    trust_remote_code=True)
    print("  [OK] quant:", d.quantization, "dtype:", d.dtype)
    print("  arch:", d.hf_config.architectures)
    print("  model_type:", d.hf_config.model_type)
except Exception as e:
    print("  [NG]", type(e).__name__, str(e)[:400])

print("\n=== 本命: target は dataset、draft は draft dir ===")
try:
    tgt = ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                      trust_remote_code=True)
    par = ParallelConfig(tensor_parallel_size=2)
    sp = SpeculativeConfig(
        method="mtp", num_speculative_tokens=2,
        model=DRAFT, target_model_config=tgt, target_parallel_config=par)
    print("  [OK] method:", sp.method, "n:", sp.num_speculative_tokens)
    print("  draft.model:", sp.draft_model_config.model)
    print("  draft.arch:", sp.draft_model_config.hf_config.architectures)
    print("  draft.model_type:", sp.draft_model_config.hf_config.model_type)
    print("  draft.n_predict:",
          getattr(sp.draft_model_config.hf_config, "n_predict", None))
    print("  draft.quant:", sp.draft_model_config.quantization)
    print("  draft_tp:", sp.draft_parallel_config.tensor_parallel_size)
except Exception as e:
    print("  [NG]", type(e).__name__, str(e)[:500])

print("\n=== model 省略時（現行）との比較 ===")
try:
    tgt2 = ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                       trust_remote_code=True)
    sp2 = SpeculativeConfig(method="mtp", num_speculative_tokens=2,
                            target_model_config=tgt2,
                            target_parallel_config=par)
    print("  draft.model:", sp2.draft_model_config.model)
    print("  → この dir の *.safetensors:",
          len(glob.glob(os.path.join(sp2.draft_model_config.model,
                                     "*.safetensors"))), "ファイル")
except Exception as e:
    print("  [NG]", type(e).__name__, str(e)[:300])

print("\n=== vLLM の loader が使う allow_patterns で絞れるか（実コード確認）===")
import inspect
from vllm.model_executor.model_loader import default_loader
src = inspect.getsource(default_loader.DefaultModelLoader._prepare_weights)
for line in src.splitlines():
    if "allow_patterns" in line or "glob" in line:
        print("  ", line.strip()[:150])
'''


if __name__ == "__main__":
    _write("step2-vllm-draftdir", "Step2 vLLM draftdir",
           "# 📁 draft 専用ディレクトリ\n\n"
           "draft が 17.41 GiB を読む問題を、`--speculative-config` の `model` に\n"
           "最小ディレクトリを渡して回避できるか検証する。\n",
           DRAFTDIR, [DATASET_ID])

    script = HERE / "push_probe11.sh"
    script.write_text(
        "#!/bin/bash\n"
        "cd \"$(dirname \"$0\")\"\n"
        "for k in " + " ".join(SLUGS) + "; do\n"
        "  for i in $(seq 1 40); do\n"
        "    out=$(cd \"step2-vllm-$k\" && kaggle kernels push -p . 2>&1 | tail -1)\n"
        "    if echo \"$out\" | grep -q 'successfully pushed'; then\n"
        "      echo \"$k: pushed\"; break\n"
        "    fi\n"
        "    echo \"$k: 待機 ($i) $out\"; sleep 60\n"
        "  done\n"
        "done\n", encoding="utf-8")
    script.chmod(0o755)
    print(f"wrote {script}")

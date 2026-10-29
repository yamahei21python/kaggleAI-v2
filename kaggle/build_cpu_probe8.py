#!/usr/bin/env python3
"""CPU 網羅テスト 第 8 弾 — 5 本（GPU 不使用）。

  1. step2-vllm-cfgresolve … 実 pip install 後に ModelConfig / SpeculativeConfig /
                             実 argparse（フラグ実在を「parse できるか」で判定）
  2. step2-vllm-keymap     … checkpoint 全キー × WeightsMapper（target/draft）適用
                             + weight_shape から TP=2 分割可否
  3. step2-vllm-enviro     … VLLM_* 環境変数と validate_environ（即死する env）
  4. step2-vllm-mtpmat     … mtp 分岐の要求（num tokens / draft config / ファイル探索）
  5. step2-vllm-genconf    … generation_config の sampling 既定値がどう効くか

CPU 上限 5 のため push は 1 本ずつ:

    python3 kaggle/build_cpu_probe8.py
    bash kaggle/push_probe8.sh
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
SLUGS = ["cfgresolve", "keymap", "enviro", "mtpmat", "genconf"]

ROOT_FINDER = '''
def find_root():
    """dataset のマウント位置は 2 通りあるので config.json を探す。"""
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

WHEEL_BOOT = '''import glob
import pathlib
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "download", "--no-cache-dir",
                "--no-deps", "vllm==0.32.0", "-d", "/tmp/w"],
               capture_output=True, text=True)
whl = glob.glob("/tmp/w/*.whl")
print("wheel:", whl)
if not whl:
    raise SystemExit("wheel 取得失敗")
zipfile.ZipFile(whl[0]).extractall("/tmp/vl")
root = pathlib.Path("/tmp/vl/vllm")
print("py files:", len(list(root.rglob("*.py"))))
'''

INSTALL_BOOT = '''import subprocess
import sys

print("=== 素の環境 ===")
r = subprocess.run([sys.executable, "-c",
                    "import torch, transformers; print(torch.__version__, transformers.__version__)"],
                   capture_output=True, text=True)
print("  ", (r.stdout or r.stderr).strip()[:200])

print("=== pip install vllm==0.32.0 ===")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "vllm==0.32.0"], capture_output=True, text=True, timeout=5400)
tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
print("   rc:", r.returncode, "|", " / ".join(tail[-3:])[:300])
r = subprocess.run([sys.executable, "-c",
                    "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"],
                   capture_output=True, text=True)
print("   versions:", (r.stdout or r.stderr).strip()[:300])
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


# --------------------------------------------------------------------------
# 1. 構成解決（実 install → 実際の argparse でフラグ実在を判定）
# --------------------------------------------------------------------------
CFGRESOLVE = INSTALL_BOOT + ROOT_FINDER + '''
ROOT = find_root()
TPL = ROOT + "/chat_template.jinja"
print("ROOT:", ROOT)
print("template exists:", __import__("os").path.isfile(TPL))

CASE = r"""
import json, sys, traceback
ROOT, TPL = sys.argv[1], sys.argv[2]
from vllm.config import ModelConfig, SpeculativeConfig


def show(tag, fn):
    try:
        v = fn()
        print(f"[OK] {tag}")
        return v
    except Exception as e:
        print(f"[NG] {tag}: {type(e).__name__}: {str(e)[:600]}")
        return None


def dumpmc(tag, mc):
    hf = getattr(mc, "hf_config", None)
    qc = getattr(hf, "quantization_config", None)
    qc = qc if isinstance(qc, dict) else {}
    mm = getattr(mc, "multimodal_config", None)
    print(f"    quant={getattr(mc, 'quantization', None)!r} "
          f"model_type={getattr(hf, 'model_type', None)!r} dtype={mc.dtype}")
    print(f"    arch={getattr(hf, 'architectures', None)}")
    print(f"    qc.method={qc.get('quant_method')!r} qc.format={qc.get('format')!r}")
    print(f"    mm={type(mm).__name__ if mm else None} "
          f"lang_only={getattr(mm, 'language_model_only', None)}")
    print(f"    gen_cfg={getattr(mc, 'generation_config', None)!r}")


print("=== ModelConfig ===")
base = show("base", lambda: ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                                        trust_remote_code=True))
if base:
    dumpmc("base", base)
    print("    diff_sampling:", base.get_diff_sampling_param())
    print("    hf mtp_num_hidden_layers:",
          getattr(base.hf_config, "mtp_num_hidden_layers", None))
    print("    hf num_hidden_layers:", getattr(base.hf_config, "num_hidden_layers", None))

for q in ("compressed-tensors", "awq", "auto"):
    mc = show(f"quant={q}", lambda q=q: ModelConfig(
        model=ROOT, dtype="half", max_model_len=8192, quantization=q,
        trust_remote_code=True))
    if mc:
        dumpmc(f"quant={q}", mc)

mc = show("language_model_only", lambda: ModelConfig(
    model=ROOT, dtype="half", max_model_len=8192, language_model_only=True,
    trust_remote_code=True))
if mc:
    dumpmc("lang_only", mc)
    mm = getattr(mc, "multimodal_config", None)
    print("    limit_per_prompt:", getattr(mm, "limit_per_prompt", None))

mc = show("generation_config=vllm", lambda: ModelConfig(
    model=ROOT, dtype="half", max_model_len=8192, generation_config="vllm",
    trust_remote_code=True))
if mc:
    print("    diff_sampling(vllm):", mc.get_diff_sampling_param())

mc = show("override_generation_config", lambda: ModelConfig(
    model=ROOT, dtype="half", max_model_len=8192,
    override_generation_config={"repetition_penalty": 1.1},
    trust_remote_code=True))
if mc:
    print("    diff_sampling(override):", mc.get_diff_sampling_param())

print("\\n=== SpeculativeConfig（method 別）===")
if base:
    cases = [("mtp", dict(num_speculative_tokens=1)),
             ("mtp", dict(num_speculative_tokens=3)),
             ("mtp", dict(num_speculative_tokens=1, model=ROOT)),
             ("qwen3_5_mtp", dict(num_speculative_tokens=1)),
             ("mtp", dict()),
             ("mtp", dict(num_speculative_tokens=0)),
             ("mtp", dict(num_speculative_tokens=-1))]
    for meth, kw in cases:
        tag = f"{meth} {kw}"
        sp = show(tag, lambda meth=meth, kw=kw: SpeculativeConfig(
            method=meth, target_model_config=base, **kw))
        if not sp:
            continue
        dm = getattr(sp, "draft_model_config", None)
        print(f"    -> n={sp.num_speculative_tokens} model={sp.model!r} "
              f"quant={getattr(sp, 'quantization', None)!r}")
        if dm is not None:
            print(f"    draft: model={dm.model!r} "
                  f"model_type={getattr(dm.hf_config, 'model_type', None)!r} "
                  f"quant={dm.quantization!r}")
            print(f"    draft arch={getattr(dm.hf_config, 'architectures', None)}")
            print(f"    draft mtp_num_hidden_layers="
                  f"{getattr(dm.hf_config, 'mtp_num_hidden_layers', None)}")

print("\\n=== chat template ===")
from vllm.renderers.chat_utils import validate_chat_template
show("validate(file)", lambda: validate_chat_template(TPL) or "ok")
show("validate(nonexistent)", lambda: validate_chat_template("/nope.jinja"))

from vllm.transformers_utils.tokenizer import get_tokenizer
tok = show("get_tokenizer", lambda: get_tokenizer(ROOT, trust_remote_code=True))
if tok:
    t = tok.get_chat_template(None, tools=None)
    print("    template len:", len(t) if t else None, "| <think>:", "<think>" in (t or ""))

print("\\n=== 実 argparse（フラグ実在の本判定）===")
from vllm.entrypoints.launchers.cli_args import (
    make_arg_parser, validate_parsed_serve_args)
from vllm.utils.argparse_utils import FlexibleArgumentParser

p = make_arg_parser(FlexibleArgumentParser())

SPEC_JSON = json.dumps({"method": "mtp", "num_speculative_tokens": 1})
FLAGS = [
    ("--host", ["127.0.0.1"]),
    ("--port", ["8000"]),
    ("--served-model-name", ["m"]),
    ("--tensor-parallel-size", ["2"]),
    ("--dtype", ["half"]),
    ("--quantization", ["compressed-tensors"]),
    ("--max-model-len", ["8192"]),
    ("--max-num-seqs", ["4"]),
    ("--max-num-batched-tokens", ["4096"]),
    ("--gpu-memory-utilization", ["0.9"]),
    ("--kv-cache-dtype", ["fp8"]),
    ("--swap-space", ["0"]),
    ("--cpu-offload-gb", ["0"]),
    ("--trust-remote-code", []),
    ("--reasoning-parser", ["qwen3"]),
    ("--enforce-eager", []),
    ("--enable-prefix-caching", []),
    ("--enable-auto-tool-choice", []),
    ("--tool-call-parser", ["qwen3_xml"]),
    ("--speculative-config", [SPEC_JSON]),
    ("--spec-method", ["mtp"]),
    ("--language-model-only", []),
    ("--limit-mm-per-prompt", ['{"image":0}']),
    ("--chat-template", [TPL]),
    ("--chat-template-content-format", ["string"]),
    ("--default-chat-template-kwargs", ['{"reasoning_effort":"medium"}']),
    ("--trust-request-chat-template", []),
    ("--disable-log-requests", []),
    ("--enable-log-requests", []),
    ("--enable-chunked-prefill", []),
    ("--compilation-config", ['{"level":0}']),
    ("--load-format", ["safetensors"]),
    ("--mm-processor-kwargs", ["{}"]),
    ("--disable-mm-preprocessor-cache", []),
    ("--uvicorn-log-level", ["warning"]),
    ("--api-key", ["k"]),
    ("--allowed-origins", ['["*"]']),
    ("--tokenizer", [ROOT]),
    ("--seed", ["0"]),
    ("--distributed-executor-backend", ["mp"]),
    ("--worker-cls", [""]),
    ("--disable-custom-all-reduce", []),
    ("--generation-config", ["auto"]),
    ("--override-generation-config", ['{"repetition_penalty":1.05}']),
    ("--enforce-sampling", []),
    ("--served-model-name-extra", ["x"]),
]

ok_flags, ng_flags = [], []
print("結果は「parse できるか」で判定。--help の grep より確実。")
for flag, vals in FLAGS:
    try:
        p.parse_args([ROOT] + [flag] + vals)
        ok_flags.append(flag)
        print(f"  [OK] {flag}")
    except SystemExit as e:
        ng_flags.append(flag)
        print(f"  [NG] {flag} (SystemExit {e.code})")
    except Exception as e:
        ng_flags.append(flag)
        print(f"  [NG] {flag} ({type(e).__name__}: {str(e)[:160]})")

print("\\n--- まとめ ---")
print("OK:", len(ok_flags), "NG:", len(ng_flags))
print("OK_FLAGS =", json.dumps(ok_flags, ensure_ascii=False))
print("NG_FLAGS =", json.dumps(ng_flags, ensure_ascii=False))

print("\\n=== 全部盛り 1 回の parse ===")
ALL = [ROOT, "--served-model-name", "m", "--host", "127.0.0.1", "--port", "8000",
       "--tensor-parallel-size", "2", "--dtype", "half", "--max-model-len", "8192",
       "--max-num-seqs", "4", "--gpu-memory-utilization", "0.9",
       "--kv-cache-dtype", "fp8", "--trust-remote-code",
       "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
       "--tool-call-parser", "qwen3_xml", "--speculative-config", SPEC_JSON,
       "--language-model-only", "--enforce-eager", "--chat-template", TPL,
       "--chat-template-content-format", "string",
       "--default-chat-template-kwargs", '{"reasoning_effort":"medium"}',
       "--generation-config", "auto", "--disable-log-requests"]
ns = show("parse(ALL)", lambda: p.parse_args(ALL))
if ns:
    for k in ("chat_template", "chat_template_content_format",
              "default_chat_template_kwargs", "language_model_only",
              "speculative_config", "reasoning_parser", "tool_call_parser",
              "generation_config", "host", "port"):
        print(f"    {k} = {getattr(ns, k, '<missing>')!r}")
    show("validate_parsed_serve_args", lambda: validate_parsed_serve_args(ns) or "ok")
"""

import os
import subprocess
import sys
import tempfile

case = os.path.join(tempfile.gettempdir(), "cfgcase.py")
open(case, "w").write(CASE)
r = subprocess.run([sys.executable, case, ROOT, TPL],
                   capture_output=True, text=True, timeout=3600)
print((r.stdout or "")[-12000:])
if r.returncode != 0:
    print("=== stderr ===")
    print((r.stderr or "")[-4000:])
'''


# --------------------------------------------------------------------------
# 2. 重みキー × mapper（target / draft）と TP=2 分割可否
# --------------------------------------------------------------------------
KEYMAP = WHEEL_BOOT + ROOT_FINDER + '''
import json
import struct

ROOT = find_root()
TP = 2

print("\\n=== どのクラスが使われるか（config.architectures）===")
cfg = json.load(open(ROOT + "/config.json"))
print("  architectures:", cfg.get("architectures"), "| model_type:", cfg.get("model_type"))

print("\\n=== mapper 定義（実クラスごと）===")
for f in ("qwen3_5.py", "qwen3_vl.py", "qwen3_5_mtp.py"):
    t = (root / "model_executor/models" / f).read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        if "hf_to_vllm_mapper" in l or "orig_to_new_" in l:
            print(f"  {f}:{i}: {l.strip()[:200]}")

print("\\n=== 生キーの取得 ===")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace")), 8 + n


h_main, base_main = header(ROOT + "/model.safetensors")
h_mtp, _ = header(ROOT + "/model-mtp.safetensors")
keys_main = [k for k in h_main if k != "__metadata__"]
keys_mtp = [k for k in h_mtp if k != "__metadata__"]
print(f"  model.safetensors: {len(keys_main)} tensors")
print(f"  model-mtp.safetensors: {len(keys_mtp)} tensors")
print("  dtype 内訳:", {d: sum(1 for v in h_main.values()
                              if isinstance(v, dict) and v.get("dtype") == d)
                       for d in {v.get("dtype") for v in h_main.values()
                                 if isinstance(v, dict)}})


def apply_map(keys, prefix_map):
    out, dropped = [], []
    for k in keys:
        for pre, new in prefix_map.items():
            if k.startswith(pre):
                if new is None:
                    dropped.append(k)
                else:
                    out.append(new + k[len(pre):])
                break
        else:
            out.append(k)          # 一致なしはそのまま
    return out, dropped


TEXT_MAP = {"model.language_model.": "model.", "mtp.": None}
VL_MAP = {"model.visual.": "visual.", "lm_head.": "language_model.lm_head.",
          "model.language_model.": "language_model.model.", "mtp.": None}
DRAFT_MAP = {"mtp.": "model."}

for name, m in (("text-only クラス", TEXT_MAP), ("VL クラス(実使用)", VL_MAP)):
    out, dropped = apply_map(keys_main, m)
    pref = sorted({".".join(k.split(".")[:2]) for k in out})
    print(f"\\n--- {name}: {len(out)} 変換 / {len(dropped)} 破棄 ---")
    print("  prefix:", pref[:8])
    bad = [k for k in out if k.startswith(("model.visual.", "model.language_model.",
                                           "lm_head."))]
    print("  変換漏れ:", len(bad), bad[:3])

out_vl, dropped_vl = apply_map(keys_main, VL_MAP)
print("\\n--- VL mapper の内訳 ---")
for pre in ("visual.", "language_model.model.", "language_model.lm_head."):
    n = sum(1 for k in out_vl if k.startswith(pre))
    print(f"  {pre:26s} {n}")
print("  想定外:", [k for k in out_vl if not k.startswith(
    ("visual.", "language_model.model.", "language_model.lm_head."))][:4])
print("  dropped:", dropped_vl, "（main に mtp.* は無いのが正）")

print("\\n=== vision の stacked mapping（q/k/v -> qkv）===")
st = [k for k in out_vl if k.startswith("visual.") and ".attn.q." in k]
print("  visual attn.q の例:", st[:2], "件数:", len(st))
print("  → vllm は attn.q/k/v を qkv にまとめる（orig_to_new_stacked）")

print("\\n=== ignore 305 件との突き合わせ（変換後）===")
qc = cfg.get("quantization_config") or {}
ignore = list(qc.get("ignore") or [])
allk = out_vl
hit = [k for k in allk if any(k.startswith(i) for i in ignore)]
print("  ignore:", len(ignore), "| 一致した重み:", len(hit), hit[:3])
quant = [k for k in allk if k.endswith(("weight_packed", "weight_scale",
                                         "weight_shape", "weight_zero_point"))]
print("  量子化系キー:", len(quant))
print("  量子化系なのに ignore 一致:", len([k for k in quant
                                     if any(k.startswith(i) for i in ignore)]))
print("  ignore されるのに量子化系:", len([k for k in hit if k in quant]))

print("\\n=== draft mapper ===")
k2, d2 = apply_map(keys_mtp, DRAFT_MAP)
print(f"  {len(k2)} 変換 / {len(d2)} 破棄")
print("  例:", k2[:4])
print("  想定外（model. で始まらない）:", [k for k in k2 if not k.startswith("model.")][:3])

print("\\n=== weight_shape の実体（dtype で unpack を変える）===")
FMT = {"I32": ("<2i", 8), "I64": ("<2q", 16), "U32": ("<2I", 8), "U64": ("<2Q", 16)}
shapes = {}
with open(ROOT + "/model.safetensors", "rb") as f:
    for k in keys_main:
        if not k.endswith("weight_shape"):
            continue
        meta = h_main[k]
        a, b = meta["data_offsets"]
        fmt, need = FMT.get(meta.get("dtype"), ("<2i", 8))
        if b - a < need:
            print("  短すぎる:", k, meta.get("dtype"), b - a)
            continue
        f.seek(base_main + a)
        shapes[k] = struct.unpack(fmt, f.read(need))
print("  dtype:", {h_main[k].get("dtype") for k in shapes}, "| 件数:", len(shapes))
print("  例:", list(shapes.items())[:3])

odd = [(k, s) for k, s in shapes.items() if s[0] % TP or s[1] % TP]
print(f"  TP={TP} で割れない shape:", len(odd), odd[:5])
mx = max(shapes.values(), key=lambda s: s[0] * s[1]) if shapes else None
print("  最大 shape:", mx)

print("\\n=== 代表レイヤ ===")
for k in sorted(shapes):
    if "layers.0." in k and any(x in k for x in ("qkv", "z.weight", "o_proj",
                                                 "gate_up", "down_proj")):
        print(f"  {k.split('.language_model.')[-1]}: {shapes[k]}")
'''


# --------------------------------------------------------------------------
# 3. 環境変数と validate_environ
# --------------------------------------------------------------------------
ENVIRO = WHEEL_BOOT + '''
import os
import re

envs = (root / "envs.py").read_text(errors="replace")
names = sorted(set(re.findall(r'"(VLLM_[A-Z0-9_]+)"', envs)))
print("=== envs.py が読む環境変数:", len(names), "件 ===")

KEY = ("GDN", "ATTENTION_BACKEND", "FLASH", "TRITON", "USE_V1", "COMPILE",
       "CAPTURE", "WORKER", "MAX_JOBS", "CACHE", "TORCH", "DEVICE", "CUDA",
       "MULTIPROC", "SLEEP", "KV", "NIXL", "S3", "LOGGING", "VERBOSE")
hits = [n for n in names if any(k in n for k in KEY)]
for n in hits:
    m = re.search(r'"' + n + r'"\\s*:\\s*lambda[^\\n]*', envs)
    line = m.group(0) if m else ""
    cur = os.environ.get(n)
    print(f"  {n:44s} set={cur!r}  {line[:90]}")

print("\\n=== validate_environ（即死する env）===")
for f in ("platforms/cuda.py", "platforms/interface.py"):
    t = (root / f).read_text(errors="replace")
    i = t.find("def validate_environ")
    if i < 0:
        continue
    print(f"--- {f} ---")
    print(t[i:i + 2200])

print("\\n=== capability で分岐するフラグ候補 ===")
for f in ("platforms/cuda.py", "platforms/interface.py",
          "v1/attention/backends/registry.py"):
    t = (root / f).read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        if "has_device_capability" in l or "is_device_capability" in l:
            print(f"  {f}:{i}: {l.strip()[:150]}")

print("\\n=== 実行環境の env（VLLM_/CUDA_/TORCH_ のみ）===")
for k in sorted(os.environ):
    if k.startswith(("VLLM_", "CUDA_", "TORCH_", "NCCL_")):
        print(f"  {k}={os.environ[k][:80]}")
'''


# --------------------------------------------------------------------------
# 4. mtp 分岐の要求（draft なしで成立するか / ファイル探索）
# --------------------------------------------------------------------------
MTPMAT = WHEEL_BOOT + ROOT_FINDER + '''
import json

ROOT = find_root()

t = (root / "config/speculative.py").read_text(errors="replace")
lines = t.splitlines()

print("=== __post_init__ の mtp 分岐（前後）===")
for i, l in enumerate(lines, 1):
    if '"mtp"' in l and ("method" in l or "==" in l):
        a, b = max(0, i - 20), min(len(lines), i + 90)
        for j in range(a, b):
            print(f"  {j + 1}: {lines[j][:170]}")
        break

print("\\n=== qwen3_5 固有の分岐 ===")
for i, l in enumerate(lines, 1):
    if "qwen3_5" in l:
        a, b = max(0, i - 8), min(len(lines), i + 30)
        print(f"  --- line {i} ---")
        for j in range(a, b):
            print(f"  {j + 1}: {lines[j][:170]}")
        break

print("\\n=== 必須項目の検証 ===")
for i, l in enumerate(lines, 1):
    s = l.strip()
    if ("raise" in s or "assert" in s) and ("spec" in s.lower() or "draft" in s.lower()
                                            or "mtp" in s.lower() or "model" in s.lower()):
        print(f"  {i}: {s[:170]}")

print("\\n=== safetensors の探索（loader）===")
for f in ("model_executor/model_loader/weight_utils.py",
          "model_executor/model_loader/default_loader.py",
          "model_executor/model_loader/loader.py"):
    p = root / f
    if not p.exists():
        continue
    tt = p.read_text(errors="replace")
    for i, l in enumerate(tt.splitlines(), 1):
        s = l.strip()
        if "safetensors" in s and ("glob" in s or "endswith" in s or "iter" in s):
            print(f"  {f}:{i}: {s[:160]}")

print("\\n=== config.json の mtp 関連 ===")
cfg = json.load(open(ROOT + "/config.json"))
for k in sorted(cfg):
    if "mtp" in k.lower() or "layer" in k.lower() or "hidden" in k.lower():
        v = cfg[k]
        print(f"  {k}: {v if not isinstance(v, (dict, list)) else type(v).__name__}")
tc = cfg.get("text_config") or {}
for k in sorted(tc):
    if "mtp" in k.lower() or "layer" in k.lower():
        print(f"  text_config.{k}: {tc[k]}")

print("\\n=== draft クラス（Qwen3_5MTP）の前提 ===")
m = (root / "model_executor/models/qwen3_5_mtp.py").read_text(errors="replace")
for i, l in enumerate(m.splitlines(), 1):
    s = l.strip()
    if s.startswith("class ") or "num_mtp_layers" in s or "mtp_start_layer_idx" in s:
        print(f"  {i}: {s[:160]}")
'''


# --------------------------------------------------------------------------
# 5. generation_config の効き方
# --------------------------------------------------------------------------
GENCONF = INSTALL_BOOT + ROOT_FINDER + '''
ROOT = find_root()

CASE = r"""
import inspect
import json
import sys

ROOT = sys.argv[1]
from vllm.config import ModelConfig
from vllm.sampling_params import SamplingParams

mc = ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                 trust_remote_code=True)
print("generation_config 設定:", mc.generation_config)
diff = mc.get_diff_sampling_param()
print("diff_sampling_param:", diff)

print("\\n=== API 既定値との関係（serving 側の使い方）===")
src = inspect.getsource(SamplingParams.update_from_generation_config)
print(src[:1500])

print("\\n=== SamplingParams の既定値 ===")
sp = SamplingParams()
for k in ("temperature", "top_p", "top_k", "repetition_penalty",
          "presence_penalty", "frequency_penalty", "seed", "stop_token_ids"):
    print(f"  {k} = {getattr(sp, k, '<missing>')!r}")

print("\\n=== temperature=0 の扱い ===")
for t in (0.0, 0.01, None, 1.0):
    try:
        s = SamplingParams(temperature=t)
        print(f"  temperature={t!r} -> {s.temperature!r} greedy={getattr(s,'_greedy',None)}")
    except Exception as e:
        print(f"  temperature={t!r} -> {type(e).__name__}: {str(e)[:200]}")

print("\\n=== generation_config を反映した SamplingParams ===")
sp = SamplingParams()
sp.update_from_generation_config(diff, eos_token_id=None)
for k in ("temperature", "top_p", "top_k", "repetition_penalty", "stop_token_ids"):
    print(f"  {k} = {getattr(sp, k, '<missing>')!r}")

print("\\n=== eos 2 個の扱い ===")
gc = json.load(open(ROOT + "/generation_config.json"))
print("  generation_config:", gc)
sp = SamplingParams()
sp.update_from_generation_config(gc, eos_token_id=248046)
print("  stop_token_ids:", sp.stop_token_ids)
print("  all_stop_token_ids:", getattr(sp, "_all_stop_token_ids", None))
"""

import os
import subprocess
import sys
import tempfile

case = os.path.join(tempfile.gettempdir(), "gencase.py")
open(case, "w").write(CASE)
r = subprocess.run([sys.executable, case, ROOT], capture_output=True, text=True,
                   timeout=1800)
print((r.stdout or "")[-12000:])
if r.returncode != 0:
    print("=== stderr ===")
    print((r.stderr or "")[-3000:])
'''


if __name__ == "__main__":
    _write("step2-vllm-cfgresolve", "Step2 vLLM cfgresolve",
           "# 🧩 構成解決とフラグ実在\n\n"
           "実 install 後に ModelConfig / SpeculativeConfig / argparse を通す。\n",
           CFGRESOLVE, [DATASET_ID])
    _write("step2-vllm-keymap", "Step2 vLLM keymap",
           "# 🔑 重みキー × mapper\n\n"
           "全キーに mapper を適用し、未変換・TP=2 分割可否を出す。\n",
           KEYMAP, [DATASET_ID])
    _write("step2-vllm-enviro", "Step2 vLLM enviro",
           "# 🌱 環境変数\n\n"
           "VLLM_* と validate_environ。設定すると即死する env を特定。\n",
           ENVIRO, [])
    _write("step2-vllm-mtpmat", "Step2 vLLM mtpmat",
           "# 🧬 MTP の要求\n\n"
           "draft model なしで mtp が成立する条件とファイル探索。\n",
           MTPMAT, [DATASET_ID])
    _write("step2-vllm-genconf", "Step2 vLLM genconf",
           "# 🎲 generation_config\n\n"
           "既定 sampling 値が API にどう効くか、temperature=0 の扱い。\n",
           GENCONF, [DATASET_ID])

    script = HERE / "push_probe8.sh"
    script.write_text(
        "#!/bin/bash\n"
        "# CPU セッション上限 5。空きが出るたびに 1 本ずつ push する。\n"
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

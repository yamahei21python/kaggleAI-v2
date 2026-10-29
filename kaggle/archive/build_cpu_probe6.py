#!/usr/bin/env python3
"""CPU 網羅テスト 第 6 弾 — 5 本同時投入（GPU 不使用）。

  1. step2-vllm-qwen35src  … vLLM の qwen3_5 実装・線形注意(GDN)が sm75 で動くか
  2. step2-vllm-mtpload    … MTP 重みの実体と vLLM ローダの期待キーが一致するか
  3. step2-vllm-quantschema… pack-quantized(asymmetric 4bit) を sm75 で扱えるか
  4. step2-vllm-t4matrix   … wheel 全体の device_capability ガードを一覧化
  5. step2-vllm-tplplus    … template の tools / vision / reasoning_effort 描画

すべて CPU・数分。wheel は download するだけで install しない。

    python3 kaggle/build_cpu_probe6.py
    for k in qwen35src mtpload quantschema t4matrix tplplus; do
      (cd kaggle/step2-vllm-$k && kaggle kernels push -p .)
    done
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
MODEL_ROOT = ("/kaggle/input/datasets/yamahei21python/"
              "qwen38-27b-uncensored-hauhaucs-awq-mtp")

# dataset のマウント位置は環境で揺れる（実測: /kaggle/input/<slug> のことも
# /kaggle/input/datasets/<slug> のこともある）。config.json を探して決める。
ROOT_FINDER = '''
def find_root():
    import glob
    slug = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
    ordered = []
    for pat in ("/kaggle/input/*/config.json", "/kaggle/input/*/*/config.json",
                "/kaggle/input/*/*/*/config.json"):
        ordered += glob.glob(pat)
    # slug 一致を最優先。無ければ qwen を含むもの、最後に全候補。
    cands = ([c for c in ordered if slug in c]
             or [c for c in ordered if "qwen" in c.lower()]
             or ordered)
    print("config.json 候補:", cands[:3])
    if not cands:
        print("/kaggle/input 直下:", glob.glob("/kaggle/input/*")[:10])
        raise SystemExit("dataset が見つからない")
    return cands[0].rsplit("/", 1)[0]


ROOT = find_root()
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
# 1. qwen3_5 実装 × sm75（線形注意が最大の地雷）
# --------------------------------------------------------------------------
QWEN35SRC = WHEEL_BOOT + '''
print("\\n=== qwen3_5 関連ファイル ===")
targets = []
for f in sorted(root.rglob("qwen3_5*.py")):
    print("  ", f.relative_to(root), f.stat().st_size)
    targets.append(f)
for pat in ("*mamba*.py", "*linear_attn*.py", "*gdn*.py"):
    for f in sorted(root.rglob(pat)):
        print("  ", f.relative_to(root), f.stat().st_size)
        targets.append(f)

mf = root / "model_executor/models/qwen3_5.py"
txt = mf.read_text(errors="replace") if mf.exists() else ""
print("\\n=== qwen3_5.py の import ===")
for i, l in enumerate(txt.splitlines(), 1):
    s = l.strip()
    if s.startswith("import ") or s.startswith("from "):
        print(f"  {i}: {s[:150]}")

print("\\n=== qwen3_5.py の層構築（linear_attn / MTP）===")
for i, l in enumerate(txt.splitlines(), 1):
    if any(k in l for k in ("linear_attn", "LinearAttention", "mamba",
                            "mtp", "MTP", "gdn", "conv1d")):
        print(f"  {i}: {l.strip()[:160]}")

print("\\n=== 危険シンボル（sm75 で落ちうる）===")
DANGER = ["flash_attn", "causal_conv1d", "fla.ops", "from fla", "triton",
          "chunk_gated", "fused_recurrent", "selective_scan", "cutlass",
          "flashinfer"]
for f in sorted(set(targets)):
    t = f.read_text(errors="replace")
    hits = [k for k in DANGER if k in t]
    if hits:
        print(f"  {f.relative_to(root)}: {hits}")

print("\\n=== capability / sm 判定がある行（対象ファイル内）===")
for f in sorted(set(targets)):
    t = f.read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if ("capability" in s or "is_sm" in s or "major" in s
                or "75" in s or "80" in s or "90" in s) and (
                ">=" in s or "<" in s or "==" in s or "if " in s):
            print(f"  {f.relative_to(root)}:{i}: {s[:150]}")

print("\\n=== attention backend の優先順位（T4 はここで決まる）===")
sel = list(root.rglob("attention/selector.py")) + list(root.rglob("attention/backends/registry.py"))
for f in sel:
    print(f"--- {f.relative_to(root)} ---")
    t = f.read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if any(k in s for k in ("priority", "backend", "capability", "major",
                                "def get_attn_backend")) and (
                "=" in s or "def " in s or "if " in s):
            print(f"  {i}: {s[:150]}")

cu = root / "platforms/cuda.py"
if cu.exists():
    t = cu.read_text(errors="replace")
    i = t.find("def _get_backend_priority")
    print("\\n=== cuda._get_backend_priority ===")
    print(t[i:i + 2200] if i > 0 else "関数が見つからない")
'''


# --------------------------------------------------------------------------
# 2. MTP 重みの実体 × vLLM ローダの期待キー
# --------------------------------------------------------------------------
MTPLOAD = WHEEL_BOOT + '''
import json
import os
import struct

''' + ROOT_FINDER + '''


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace"))


print("=== Dataset 内のファイル ===")
for name in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, name)
    size = "DIR" if os.path.isdir(p) else os.path.getsize(p)
    print(f"  {name:32s} {size}")

print("\\n=== safetensors ヘッダ ===")
allkeys = {}
for name in ("model.safetensors", "model-mtp.safetensors"):
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        print("  missing:", name)
        continue
    h = header(p)
    meta = h.pop("__metadata__", None)
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    print(f"  {name}: tensors={len(h)} bytes={total / 2**30:.2f}GiB meta={meta}")
    dts = {}
    for v in h.values():
        dts[v["dtype"]] = dts.get(v["dtype"], 0) + 1
    print("    dtypes:", dts)
    ks = sorted(h)
    allkeys[name] = ks
    print("    first 6:", ks[:6])
    pref = {}
    for k in ks:
        parts = k.split(".")
        p3 = ".".join(parts[:3])
        pref[p3] = pref.get(p3, 0) + 1
    top = sorted(pref.items(), key=lambda x: -x[1])[:12]
    print("    prefixes:", top)
    packed = [k for k in ks if "packed" in k][:4]
    print("    weight_packed 例:", packed)
    print("    scale 例:", [k for k in ks if k.endswith("weight_scale")][:3])

print("\\n=== index.json ===")
ip = os.path.join(ROOT, "model.safetensors.index.json")
if os.path.isfile(ip):
    idx = json.load(open(ip))
    wm = idx.get("weight_map", {})
    print("  entries:", len(wm), "| files:", sorted(set(wm.values())))
    mtp_keys = [k for k in wm if "mtp" in k.lower()]
    print("  mtp keys in index:", len(mtp_keys), mtp_keys[:6])
else:
    print("  index.json なし（単一シャード）")

print("\\n=== vLLM が MTP をどう読むか ===")
cand = [f for f in root.rglob("*.py")
        if "mtp" in f.name.lower() or "spec_decode" in str(f)]
for f in sorted(cand)[:12]:
    t = f.read_text(errors="replace")
    lines = t.splitlines()
    hits = [i for i, l in enumerate(lines) if "mtp" in l.lower()]
    if not hits:
        continue
    print(f"--- {f.relative_to(root)} ({len(hits)} hits) ---")
    for i in hits[:10]:
        print(f"  {i + 1}: {lines[i].strip()[:150]}")

qf = root / "model_executor/models/qwen3_5_mtp.py"
if qf.exists():
    print("\\n=== qwen3_5_mtp.py の重み名 ===")
    t = qf.read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if any(k in s for k in ("weight", "fc", "norm", "prefix", "load",
                                "class ", "def ")):
            print(f"  {i}: {s[:150]}")

print("\\n=== 実キーと期待プレフィックスの照合 ===")
actual = allkeys.get("model-mtp.safetensors", [])
print("actual sample:", actual[:10])
for want in ("mtp.", "model.mtp.", "model.layers.64.", "language_model.mtp."):
    hit = [k for k in actual if k.startswith(want)]
    print(f"  {want:22s} -> {len(hit)}")
'''


# --------------------------------------------------------------------------
# 3. compressed-tensors(pack-quantized) を sm75 で扱えるか
# --------------------------------------------------------------------------
QUANTSCHEMA = WHEEL_BOOT + '''
import json
import os

ct = root / "model_executor/layers/quantization/compressed_tensors"
print("=== compressed_tensors ディレクトリ ===")
print("exists:", ct.exists())
if ct.exists():
    for f in sorted(ct.rglob("*.py")):
        print("  ", f.relative_to(ct), f.stat().st_size)

qinit = root / "model_executor/layers/quantization/__init__.py"
q = qinit.read_text(errors="replace")
i = q.find("QuantizationMethods = Literal[")
print("\\n=== QuantizationMethods 全文 ===")
print(q[i:i + 1400] if i > 0 else "not found")
print("compressed-tensors を含む:", "compressed-tensors" in q)

print("\\n=== METADATA の compressed-tensors 依存 ===")
meta_name = [n for n in __import__("zipfile").ZipFile(
    "/tmp/w/" + os.listdir("/tmp/w")[0]).namelist()
    if n.endswith(".dist-info/METADATA")][0]
z = __import__("zipfile").ZipFile("/tmp/w/" + os.listdir("/tmp/w")[0])
meta = z.read(meta_name).decode("utf-8", "replace")
for line in meta.splitlines():
    if "compressed" in line.lower() or "marlin" in line.lower():
        print("  ", line[:160])

print("\\n=== scheme クラス ===")
for f in (sorted(ct.rglob("*scheme*.py")) if ct.exists() else []):
    t = f.read_text(errors="replace")
    for i2, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if s.startswith("class ") or "def can_implement" in s or "def __init__" in s:
            print(f"  {f.name}:{i2}: {s[:140]}")

print("\\n=== symmetric / asymmetric / group_size の扱い ===")
for f in sorted(root.rglob("*.py")):
    p = str(f)
    if not any(k in p for k in ("compressed_tensors", "marlin", "awq")):
        continue
    t = f.read_text(errors="replace")
    for i2, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if ("symmetric" in s or "asymmetric" in s) and (
                "raise" in s or "assert" in s or "support" in s or "if " in s):
            print(f"  {f.relative_to(root)}:{i2}: {s[:150]}")

print("\\n=== 4bit / capability ガード（marlin・cutlass）===")
for f in sorted(root.rglob("*.py")):
    p = str(f)
    if not any(k in p for k in ("marlin", "cutlass", "quantization/utils")):
        continue
    t = f.read_text(errors="replace")
    for i2, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if "capability" in s and ("<" in s or ">=" in s or "==" in s):
            print(f"  {f.relative_to(root)}:{i2}: {s[:150]}")

print("\\n=== 実際の config と突き合わせ ===")
''' + ROOT_FINDER + '''
cfg = json.load(open(os.path.join(ROOT, "config.json")))
qc = cfg.get("quantization_config") or {}
for k in ("quant_method", "format", "version", "kv_cache_scheme"):
    print(f"  {k}: {qc.get(k)}")
grp = (qc.get("config_groups") or {}).get("group_0") or {}
print("  group_0 targets:", grp.get("targets"))
print("  weights:", json.dumps(grp.get("weights"), ensure_ascii=False))
print("  ignore count:", len(qc.get("ignore") or []))
'''


# --------------------------------------------------------------------------
# 4. wheel 全体の sm75 ガード一覧
# --------------------------------------------------------------------------
T4MATRIX = WHEEL_BOOT + '''
import collections
import re

DIRS = ("attention", "quantization", "model_executor/layers", "platforms",
        "v1/attention", "v1/spec_decode", "distributed", "compilation")
pat = re.compile("capability|is_sm|sm75|sm80|sm90")

rows = []
for f in sorted(root.rglob("*.py")):
    rel = str(f.relative_to(root))
    if not any(rel.startswith(d) for d in DIRS):
        continue
    try:
        t = f.read_text(errors="replace")
    except Exception:
        continue
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if not pat.search(s):
            continue
        if not any(k in s for k in ("<", ">", "==", "if ", "elif ", "raise", "assert")):
            continue
        rows.append((rel, i, s[:140]))

print("hit count:", len(rows))
by_file = collections.Counter(r[0] for r in rows)
print("\\n=== ファイル別件数（上位 20）===")
for name, n in by_file.most_common(20):
    print(f"  {n:4d}  {name}")

print("\\n=== < 80 / < 75 / == 75 の行（T4 で分岐する箇所）===")
for rel, i, s in rows:
    if re.search(r"[<>]=?\\s*(75|80|89|90)|==\\s*(75|80|90)", s):
        print(f"  {rel}:{i}: {s}")

print("\\n=== marlin / fp8 / flashinfer の可否 ===")
KEYS = ["marlin", "fp8", "flashinfer", "xformers", "triton"]
for k in KEYS:
    n = sum(1 for r in rows if k in r[2].lower())
    print(f"  {k}: {n} 行で capability 判定と同居")
'''


# --------------------------------------------------------------------------
# 5. template の tools / vision / reasoning_effort
# --------------------------------------------------------------------------
TPLPLUS = '''import json
import os
import subprocess
import sys

''' + ROOT_FINDER + '''
print("transformers を用意")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                "transformers==5.16.1"], capture_output=True, text=True)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(ROOT)
print("tokenizer:", type(tok).__name__)
print("chat_template 解決:", tok.chat_template is not None,
      len(tok.chat_template or ""))
print("eos:", tok.eos_token, tok.eos_token_id,
      "| bos:", tok.bos_token, tok.bos_token_id,
      "| pad:", tok.pad_token)

MSGS = [{"role": "system", "content": "You are concise."},
        {"role": "user", "content": "1+1 は？"}]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "天気を返す",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def render(label, **kw):
    try:
        s = tok.apply_chat_template(MSGS, tokenize=False,
                                    add_generation_prompt=True, **kw)
    except Exception as e:
        print(f"\\n--- {label}: ERROR {type(e).__name__}: {e}")
        return
    ids = tok.encode(s, add_special_tokens=False)
    print(f"\\n--- {label}: chars={len(s)} tokens={len(ids)}")
    print("   head:", repr(s[:200]))
    print("   tail:", repr(s[-160:]))


render("basic")
for eff in ("low", "medium", "high", "xhigh"):
    render(f"reasoning_effort={eff}", reasoning_effort=eff)
render("tools", tools=TOOLS)
render("tools+eff=low", tools=TOOLS, reasoning_effort="low")

print("\\n=== tool_call の XML 形式 ===")
s = tok.apply_chat_template(MSGS + [{"role": "assistant", "content":
                                    '<tool_call><function=get_weather>'
                                    '<parameter=city>Tokyo</parameter>'
                                    '</function></tool_call>'}],
                            tokenize=False)
print(repr(s[-260:]))
print("tokenize 往復:", tok.decode(tok.encode(s, add_special_tokens=False))[:80])

print("\\n=== vision content ===")
vis = [{"role": "user", "content": [
    {"type": "image", "image": "x.png"},  # ダミー。レンダラだけ確認
    {"type": "text", "text": "これは？"}]}]
try:
    s = tok.apply_chat_template(vis, tokenize=False, add_generation_prompt=True)
    print("   ok chars:", len(s))
    print("   ", repr(s[:200]))
except Exception as e:
    print("   ERROR:", type(e).__name__, e)

print("\\n=== 長文でのトークン効率（4k 文字）===")
long_msgs = [{"role": "user", "content": "あ" * 4000}]
s = tok.apply_chat_template(long_msgs, tokenize=False, add_generation_prompt=True)
print("   chars:", len(s), "tokens:", len(tok.encode(s, add_special_tokens=False)))

print("\\n=== generation_config / 特殊トークン ===")
for name in ("generation_config.json", "special_tokens_map.json"):
    p = os.path.join(ROOT, name)
    print(f"  {name}: {'あり' if os.path.isfile(p) else 'なし'}")
    if os.path.isfile(p):
        print("   ", open(p, encoding="utf-8").read()[:400])
'''


if __name__ == "__main__":
    _write("step2-vllm-qwen35src", "Step2 vLLM qwen35src",
           "# 🧠 qwen3_5 実装 × sm75\n\n"
           "線形注意(GDN)・flash_attn・バックエンド選択を wheel から確認。\n",
           QWEN35SRC, [])
    _write("step2-vllm-mtpload", "Step2 vLLM mtpload",
           "# 🧩 MTP 重みの実体 × ローダ期待キー\n\n"
           "`mtp.*` キーと vLLM の読み取り規約が一致するか。\n",
           MTPLOAD, [DATASET_ID])
    _write("step2-vllm-quantschema", "Step2 vLLM quantschema",
           "# ⚖️ pack-quantized × sm75\n\n"
           "asymmetric 4bit / group_size 128 / marlin 可否。\n",
           QUANTSCHEMA, [DATASET_ID])
    _write("step2-vllm-t4matrix", "Step2 vLLM t4matrix",
           "# 📊 sm75 ガード全一覧\n\n"
           "wheel 内の capability 分岐を抽出して T4 の可否を表にする。\n",
           T4MATRIX, [])
    _write("step2-vllm-tplplus", "Step2 vLLM tplplus",
           "# 🧵 template 深掘り\n\n"
           "tools / vision / reasoning_effort / 長文トークン効率。\n",
           TPLPLUS, [DATASET_ID])

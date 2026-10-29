#!/usr/bin/env python3
"""CPU 網羅テスト 第 9 弾 — 3 本。

  1. step2-vllm-specmat  … SpeculativeConfig(mtp) を target_parallel_config 付きで構築し
                           draft config の解決結果（型・n_predict・量子化継承）を確認
  2. step2-vllm-filemat  … config.json を全項目走査。層種別・KV ヘッド・rope 上限から
                           KV サイズと TP=2 分割可否を実測ベースで計算
  3. step2-vllm-loadflow … safetensors 探索（model-mtp を target も読むか）と
                           読み込み量・推定時間

    python3 kaggle/build_cpu_probe9.py
    bash kaggle/push_probe9.sh
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
SLUGS = ["specmat", "filemat", "loadflow"]

ROOT_FINDER = '''
def find_root():
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


# --------------------------------------------------------------------------
# 1. SpeculativeConfig(mtp) を正しく構築する
# --------------------------------------------------------------------------
SPECMAT = INSTALL_BOOT + ROOT_FINDER + r'''
ROOT = find_root()

CASE = r"""
import json
import sys

ROOT = sys.argv[1]
from vllm.config import ModelConfig, SpeculativeConfig
from vllm.config.parallel import ParallelConfig


def show(tag, fn):
    try:
        v = fn()
        print(f"[OK] {tag}")
        return v
    except Exception as e:
        print(f"[NG] {tag}: {type(e).__name__}: {str(e)[:500]}")
        return None


target = ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                     trust_remote_code=True)
print("target: quant=", target.quantization, "arch=", target.hf_config.architectures)
tc = target.hf_config.text_config
print("text_config: num_hidden_layers=", tc.num_hidden_layers,
      "mtp_num_hidden_layers=", getattr(tc, "mtp_num_hidden_layers", None),
      "num_kv_heads=", getattr(tc, "num_key_value_heads", None),
      "head_dim=", getattr(tc, "head_dim", None),
      "layer_types=", (getattr(tc, "layer_types", None) or [])[:6])

par = show("ParallelConfig(tp=2)", lambda: ParallelConfig(tensor_parallel_size=2))
if not par:
    raise SystemExit("ParallelConfig が作れない")

for kw in (dict(num_speculative_tokens=1),
           dict(num_speculative_tokens=2),
           dict(num_speculative_tokens=3),
           dict(num_speculative_tokens=4),
           dict(),
           dict(num_speculative_tokens=1, model=ROOT),
           dict(num_speculative_tokens=2, draft_tensor_parallel_size=2)):
    tag = json.dumps(kw)
    sp = show(f"Spec {tag}", lambda kw=kw: SpeculativeConfig(
        method="mtp", target_model_config=target,
        target_parallel_config=par, **kw))
    if not sp:
        continue
    dm = sp.draft_model_config
    print(f"    n={sp.num_speculative_tokens} method={sp.method} "
          f"draft_tp={getattr(sp, 'draft_tensor_parallel_size', None)}")
    print(f"    draft.model={dm.model!r}")
    print(f"    draft.arch={dm.hf_config.architectures} "
          f"model_type={dm.hf_config.model_type}")
    print(f"    draft.n_predict={getattr(dm.hf_config, 'n_predict', None)} "
          f"mtp_num_hidden_layers={getattr(dm.hf_config, 'mtp_num_hidden_layers', None)}")
    print(f"    draft.quant={dm.quantization!r} dtype={dm.dtype} "
          f"max_model_len={dm.max_model_len}")
    print(f"    draft_parallel.tp={getattr(sp.draft_parallel_config, 'tensor_parallel_size', None)}")

print("\\n=== draft が target と同じディレクトリを指すか（重要）===")
sp = SpeculativeConfig(method="mtp", num_speculative_tokens=1,
                       target_model_config=target, target_parallel_config=par)
print("  target.model == draft.model:", target.model == sp.draft_model_config.model)
print("  値:", target.model, "|", sp.draft_model_config.model)

print("\\n=== 量子化明示との組み合わせ ===")
tct = ModelConfig(model=ROOT, dtype="half", max_model_len=8192,
                  quantization="compressed-tensors", trust_remote_code=True)
sp2 = show("Spec + ct", lambda: SpeculativeConfig(
    method="mtp", num_speculative_tokens=1, target_model_config=tct,
    target_parallel_config=par))
if sp2:
    print("  draft.quant:", sp2.draft_model_config.quantization)

print("\\n=== method の正規化 ===")
for m in ("mtp", "qwen3_5_mtp", "qwen3_next_mtp", "MTP_model"):
    sp3 = show(f"method={m}", lambda m=m: SpeculativeConfig(
        method=m, num_speculative_tokens=1, target_model_config=target,
        target_parallel_config=par))
    if sp3:
        print("    -> method:", sp3.method)
"""

import os
import subprocess
import sys
import tempfile

case = os.path.join(tempfile.gettempdir(), "speccase.py")
open(case, "w").write(CASE)
r = subprocess.run([sys.executable, case, ROOT], capture_output=True, text=True,
                   timeout=3600)
print((r.stdout or "")[-14000:])
if r.returncode != 0:
    print("=== stderr ===")
    print((r.stderr or "")[-4000:])
'''


# --------------------------------------------------------------------------
# 2. config.json 全走査（層種別・KV・rope）
# --------------------------------------------------------------------------
FILEMAT = ROOT_FINDER + r'''
import json

ROOT = find_root()
cfg = json.load(open(ROOT + "/config.json"))
tc = cfg.get("text_config") or {}

print("=== トップレベル ===")
for k in sorted(cfg):
    v = cfg[k]
    if isinstance(v, (dict, list)):
        print(f"  {k}: {type(v).__name__}({len(v)})")
    else:
        print(f"  {k}: {v}")

print("\n=== text_config（数値/文字列のみ）===")
for k in sorted(tc):
    v = tc[k]
    if isinstance(v, (dict, list)):
        print(f"  {k}: {type(v).__name__}({len(v)})")
    else:
        print(f"  {k}: {v}")

print("\n=== text_config の dict/list の中身 ===")
for k in sorted(tc):
    v = tc[k]
    if isinstance(v, list):
        print(f"  {k} ({len(v)}): {v[:8]}")
    elif isinstance(v, dict):
        print(f"  {k}: {json.dumps(v)[:300]}")

layers = tc.get("layer_types") or []
n = tc.get("num_hidden_layers")
print("\n=== 層種別 ===")
print("  num_hidden_layers:", n, "| layer_types:", len(layers),
      "| 一致:", n == len(layers))
if layers:
    from collections import Counter
    print("  内訳:", Counter(layers))
    full = [i for i, t in enumerate(layers) if "full" in t]
    print("  full_attention 層:", full)

heads = tc.get("num_attention_heads")
kv = tc.get("num_key_value_heads")
hd = tc.get("head_dim") or (tc.get("hidden_size", 0) // max(heads or 1, 1))
hidden = tc.get("hidden_size")
inter = tc.get("intermediate_size")
print("\n=== 形状 ===")
print(f"  hidden={hidden} heads={heads} kv_heads={kv} head_dim={hd} "
      f"intermediate={inter}")
print(f"  q_dim={heads * hd if heads and hd else None} "
      f"kv_dim={kv * hd if kv and hd else None}")

print("\n=== 層種別ごとの重み形状（safetensors header から）===")
import struct


def header(path):
    with open(path, "rb") as f:
        ln = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(ln).decode("utf-8", "replace")), 8 + ln


h, base = header(ROOT + "/model.safetensors")
FMT = {"I32": ("<2i", 8), "I64": ("<2q", 16), "U32": ("<2I", 8), "U64": ("<2Q", 16)}
shapes = {}
with open(ROOT + "/model.safetensors", "rb") as f:
    for k, v in h.items():
        if not k.endswith("weight_shape"):
            continue
        a, b = v["data_offsets"]
        fmt, need = FMT.get(v.get("dtype"), ("<2i", 8))
        f.seek(base + a)
        shapes[k] = struct.unpack(fmt, f.read(need))
print("  weight_shape dtype:", {h[k].get("dtype") for k in shapes},
      "| 件数:", len(shapes))
for idx in (0, 1, 2, 3):
    print(f"  --- layer {idx} ---")
    for k in sorted(shapes):
        if f".layers.{idx}." in k and k.endswith((
                "in_proj_qkv.weight_shape", "in_proj_z.weight_shape",
                "out_proj.weight_shape", "gate_up_proj.weight_shape",
                "down_proj.weight_shape", "q_proj.weight_shape",
                "o_proj.weight_shape")):
            print(f"    {k.split('.layers.')[-1]}: {shapes[k]}")

print("\n=== KV サイズ計算（実測 config から）===")
kv_bytes_per_tok = 2 * len(full) * kv * hd * 2 if kv and hd else None
print(f"  full 層 {len(full)} × kv_heads {kv} × head_dim {hd} × 2(K,V) × 2B "
      f"= {kv_bytes_per_tok} B/token = {(kv_bytes_per_tok or 0) / 1024:.1f} KiB")
for mlen in (8192, 16384, 32768, 65536):
    if kv_bytes_per_tok:
        per = kv_bytes_per_tok * mlen / 2 / 2**30
        print(f"    max_len={mlen:6d}: {per:.2f} GiB/GPU (TP=2, KV only)")

print("\n=== 線形注意の状態サイズ ===")
for k in sorted(tc):
    if any(x in k for x in ("conv", "ssm", "linear", "state", "chunk", "expand")):
        print(f"  {k}: {tc[k]}")

print("\n=== rope / 長さ上限 ===")
for k in sorted(tc):
    if "rope" in k or "max_position" in k or "sliding" in k:
        print(f"  {k}: {tc[k]}")
print("  トップレベル rope/max:", {k: cfg.get(k) for k in
                              ("max_position_embeddings", "rope_scaling",
                               "rope_theta") if k in cfg})
'''


# --------------------------------------------------------------------------
# 3. safetensors 探索と読み込み量
# --------------------------------------------------------------------------
LOADFLOW = WHEEL_BOOT + ROOT_FINDER + r'''
import json
import os

ROOT = find_root()

print("=== Dataset 内の safetensors ===")
for fn in sorted(os.listdir(ROOT)):
    if fn.endswith((".safetensors", ".bin", ".pt")):
        p = os.path.join(ROOT, fn)
        print(f"  {fn}: {os.path.getsize(p) / 2**30:.2f} GiB")

print("\n=== default_loader の探索パターン ===")
t = (root / "model_executor/model_loader/default_loader.py").read_text(errors="replace")
for i, l in enumerate(t.splitlines(), 1):
    s = l.strip()
    if "allow_patterns" in s and ("=" in s or "glob" in s):
        print(f"  {i}: {s[:150]}")

print("\n=== index 無しのときの扱い ===")
w = (root / "model_executor/model_loader/weight_utils.py").read_text(errors="replace")
i = w.find("def filter_duplicate_safetensors_files")
print(w[i:i + 1400])

print("\n=== target と draft で同じファイルを読むか ===")
print("  - target: allow_patterns=['*.safetensors'] → model.safetensors と")
print("            model-mtp.safetensors の両方を glob する（index.json 無しのため絞れない）")
print("  - target 側 mapper は 'mtp.' -> None なので mtp の 15 テンソルは破棄")
print("  - draft: 同ディレクトリを再度読む → 実質 0.79 GiB を 2 回読む")

print("\n=== 読み込み量と時間（実測 82 MiB/s）===")
bw = 82 * 2**20
main = os.path.getsize(ROOT + "/model.safetensors")
mtp = os.path.getsize(ROOT + "/model-mtp.safetensors")
print(f"  main={main / 2**30:.2f} GiB mtp={mtp / 2**30:.2f} GiB")
for label, total in (("target(main+mtp)", main + mtp),
                     ("draft(mtp のみ?)", mtp),
                     ("合計(素直な見積り)", main + mtp * 2)):
    print(f"  {label:20s} {total / 2**30:5.2f} GiB → {total / bw:6.1f}s")
print(f"  TP=2 で各 GPU が半分読む場合の target: {(main + mtp) / 2 / bw:.1f}s")

print("\n=== mtp テンソルの shape（BF16 なので size からは分からない）===")
import struct


def header(path):
    with open(path, "rb") as f:
        ln = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(ln).decode("utf-8", "replace"))


h = header(ROOT + "/model-mtp.safetensors")
for k, v in sorted(h.items()):
    if k == "__metadata__":
        continue
    size = v["data_offsets"][1] - v["data_offsets"][0]
    print(f"  {k:44s} {v.get('dtype'):5s} {size / 2**20:8.2f} MiB")

print("\n=== config の ignore が mtp を含むか ===")
cfg = json.load(open(ROOT + "/config.json"))
ig = (cfg.get("quantization_config") or {}).get("ignore") or []
print("  ignore 件数:", len(ig), "| mtp を含む:", [x for x in ig if "mtp" in x][:3])
'''


if __name__ == "__main__":
    _write("step2-vllm-specmat", "Step2 vLLM specmat",
           "# 🧬 SpeculativeConfig(mtp) の実構築\n\n"
           "target_parallel_config 付きで draft config の解決を確認。\n",
           SPECMAT, [DATASET_ID])
    _write("step2-vllm-filemat", "Step2 vLLM filemat",
           "# 📋 config.json 全走査\n\n"
           "層種別・KV ヘッド・rope 上限・重み形状から KV を計算。\n",
           FILEMAT, [DATASET_ID])
    _write("step2-vllm-loadflow", "Step2 vLLM loadflow",
           "# 📥 safetensors 探索と読み込み量\n\n"
           "model-mtp を target も読むか、実 read 量の見積り。\n",
           LOADFLOW, [DATASET_ID])

    script = HERE / "push_probe9.sh"
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

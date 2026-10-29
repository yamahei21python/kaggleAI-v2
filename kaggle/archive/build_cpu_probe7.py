#!/usr/bin/env python3
"""CPU 網羅テスト 第 7 弾 — 5 本（GPU 不使用・スロットが空き次第 push）。

  1. step2-vllm-pypiwheels  … torch/torchaudio/torchvision/torchcodec の cp313 wheel 実在確認
  2. step2-vllm-specschema  … SpeculativeConfig の mtp 分岐・上限・必須項目
  3. step2-vllm-tplflag     … `--chat-template` の実体と .jinja 自動解決の有無
  4. step2-vllm-ioread      … /kaggle/input の読み速度・cloudflared 有無（起動時間の見積り）
  5. step2-vllm-vram2       … 重み内訳（量子化/ignore）から TP=2 の VRAM 予算表

CPU 上限 5 のため、push は空きが出るたびに 1 本ずつ行う:

    python3 kaggle/build_cpu_probe7.py
    bash kaggle/push_probe7.sh
"""
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
MODEL_ROOT = ("/kaggle/input/datasets/yamahei21python/"
              "qwen38-27b-uncensored-hauhaucs-awq-mtp")

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
SLUGS = ["pypiwheels", "specschema", "tplflag", "ioread", "vram2"]

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
# 1. 依存 wheel の実在（torchaudio 2.11.0 修復が成立するか）
# --------------------------------------------------------------------------
PYPIWHEELS = '''import json
import subprocess
import sys
import urllib.request

PKGS = [("torch", "2.13.0"), ("torchaudio", "2.11.0"),
        ("torchvision", "0.28.0"), ("torchcodec", "0.14.0"),
        ("vllm", "0.32.0"), ("compressed-tensors", None)]


def files(pkg, ver):
    url = f"https://pypi.org/pypi/{pkg}/json" if ver is None else \\
          f"https://pypi.org/pypi/{pkg}/{ver}/json"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
    except Exception as e:
        print(f"  {pkg} {ver}: 取得失敗 {type(e).__name__} {e}")
        return None
    if ver is None:
        ver = d["info"]["version"]
        print(f"  latest: {ver}")
    return d


print("=== PyPI wheel 実在チェック（linux x86_64 / cp313）===")
for pkg, ver in PKGS:
    print(f"\\n-- {pkg} {ver}")
    d = files(pkg, ver)
    if not d:
        continue
    names = [u["filename"] for u in d["urls"]]
    print("   files:", len(names))
    hits = [n for n in names
            if ("cp313" in n or "py3-none" in n)
            and ("x86_64" in n or "manylinux" in n or "linux" in n or "any" in n)]
    for n in hits[:8]:
        print("    ", n)
    if not hits:
        print("     cp313/linux の wheel なし。全候補:", names[:6])

print("\\n=== インストール済みバージョン（イメージ素の状態）===")
for mod in ("torch", "torchaudio", "torchvision"):
    r = subprocess.run([sys.executable, "-c",
                        f"import {mod}; print({mod}.__version__)"],
                       capture_output=True, text=True)
    print(f"  {mod}: {(r.stdout or r.stderr).strip()[:120]}")

print("\\n=== pip の解決だけ確認（--dry-run --no-deps）===")
r = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run",
                    "--no-input", "--no-deps", "torchaudio==2.11.0"],
                   capture_output=True, text=True)
out = (r.stdout or "") + (r.stderr or "")
print("  rc:", r.returncode)
print("  ", " / ".join([l.strip() for l in out.splitlines() if l.strip()][-4:])[:400])
'''


# --------------------------------------------------------------------------
# 2. SpeculativeConfig(method=mtp) の要求
# --------------------------------------------------------------------------
SPECSCHEMA = WHEEL_BOOT + '''
import re

sp = root / "config/speculative.py"
print("speculative.py exists:", sp.exists())
t = sp.read_text(errors="replace")

print("\\n=== class / フィールド定義 ===")
for i, l in enumerate(t.splitlines(), 1):
    s = l.strip()
    if s.startswith("class ") or s.startswith("def ") or "mtp" in s.lower():
        print(f"  {i}: {s[:150]}")

print("\\n=== 数値制約・必須項目 ===")
for i, l in enumerate(t.splitlines(), 1):
    s = l.strip()
    if ("gt=" in s or "ge=" in s or "le=" in s or "lt=" in s
            or "Field(" in s or "model:" in s or "num_speculative_tokens" in s):
        print(f"  {i}: {s[:150]}")

print("\\n=== mtp のバリデーション（前後 12 行）===")
lines = t.splitlines()
for i, l in enumerate(lines, 1):
    if "mtp" in l.lower() and ("if " in l or "elif " in l or "raise" in l
                               or "assert" in l or "=" in l):
        a, b = max(0, i - 12), min(len(lines), i + 12)
        print(f"  --- line {i} ---")
        for j in range(a, b):
            print(f"   {j + 1}: {lines[j][:150]}")
        break

print("\\n=== MTPModelTypes ===")
i = t.find("MTPModelTypes")
print(t[i:i + 900] if i > 0 else "not found")

print("\\n=== 既定値（--speculative-config 未指定時）===")
for i, l in enumerate(t.splitlines(), 1):
    if "num_speculative_tokens" in l and "=" in l:
        print(f"  {i}: {l.strip()[:150]}")
'''


# --------------------------------------------------------------------------
# 3. --chat-template の実体と .jinja 自動解決
# --------------------------------------------------------------------------
TPLFLAG = WHEEL_BOOT + '''
import re

print("=== chat_template の定義箇所 ===")
for f in sorted(root.rglob("*.py")):
    t = f.read_text(errors="replace")
    if "chat_template" not in t:
        continue
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if "chat_template" in s and ("add_argument" in s or "kwargs[" in s
                                     or "chat_template:" in s
                                     or "chat_template =" in s
                                     or "chat_template.jinja" in s):
            print(f"  {f.relative_to(root)}:{i}: {s[:160]}")

print("\\n=== ChatTemplateConfig（あれば）===")
for f in sorted(root.rglob("*chat_template*.py")):
    print("  file:", f.relative_to(root))
    t = f.read_text(errors="replace")
    for i, l in enumerate(t.splitlines(), 1):
        s = l.strip()
        if s.startswith("class ") or s.startswith("def "):
            print(f"    {i}: {s[:140]}")

print("\\n=== tokenizer 側の解決（tokenizer_config に無い場合）===")
for f in sorted(root.rglob("transformers_utils/*.py")) + sorted(root.rglob("entrypoints/**/*.py")):
    t = f.read_text(errors="replace")
    if "chat_template.jinja" in t or "chat_template.json" in t:
        for i, l in enumerate(t.splitlines(), 1):
            if "chat_template" in l:
                print(f"  {f.relative_to(root)}:{i}: {l.strip()[:160]}")

print("\\n=== get_chat_template / load_chat_template 実装 ===")
for f in sorted(root.rglob("*.py")):
    t = f.read_text(errors="replace")
    for m in re.finditer(r"def (get_chat_template|load_chat_template|"
                         r"resolve_chat_template)[^(]*\\\\(", t):
        i = m.start()
        print(f"  --- {f.relative_to(root)} ---")
        print(t[i:i + 1200])
        break

print("\\n=== API リクエスト時の chat_template_kwargs 対応 ===")
for f in sorted(root.rglob("*.py")):
    t = f.read_text(errors="replace")
    if "chat_template_kwargs" in t:
        n = t.count("chat_template_kwargs")
        print(f"  {f.relative_to(root)}: {n}")
'''


# --------------------------------------------------------------------------
# 4. /kaggle/input の読み速度と cloudflared
# --------------------------------------------------------------------------
IOREAD = '''import os
import shutil
import subprocess
import time

''' + ROOT_FINDER + '''

print("=== Dataset の配置 ===")
for name in sorted(os.listdir(ROOT))[:20]:
    p = os.path.join(ROOT, name)
    print(f"  {name:32s} {'DIR' if os.path.isdir(p) else os.path.getsize(p)}")

print("\\n=== 読み速度（512MB, mmap なし）===")
p = os.path.join(ROOT, "model.safetensors")
CHUNK = 8 << 20
NEED = 512 << 20
t0 = time.time()
got = 0
with open(p, "rb") as f:
    while got < NEED:
        b = f.read(CHUNK)
        if not b:
            break
        got += len(b)
dt = time.time() - t0
print(f"  {got / 2**20:.0f} MiB in {dt:.2f}s = {got / 2**20 / dt:.0f} MiB/s")
total = 19.5 * 2**30
print(f"  19.5GiB 全体の推定: {total / (got / dt):.0f}s"
      f"（TP=2 で各 GPU が半分読むなら約 {total / 2 / (got / dt):.0f}s）")

print("\\n=== cloudflared ===")
print("  which:", shutil.which("cloudflared"))
try:
    r = subprocess.run(["cloudflared", "--version"], capture_output=True,
                       text=True, timeout=60)
    print("  version rc:", r.returncode,
          ((r.stdout or "") + (r.stderr or "")).strip()[:120])
except FileNotFoundError:
    print("  cloudflared 未インストール（毎回ダウンロードが必要）")
for d in ("/usr/local/bin", "/usr/bin", "/opt"):
    if os.path.isdir(d):
        hits = [n for n in os.listdir(d) if "cloud" in n.lower()]
        if hits:
            print(f"  {d}: {hits}")

print("\\n=== ディスク空き ===")
s = shutil.disk_usage("/kaggle/working")
print(f"  working: free {s.free / 2**30:.1f} GiB / total {s.total / 2**30:.1f} GiB")

print("\\n=== /tmp と /kaggle/working の速度差（256MB write+read）===")
for base in ("/tmp", "/kaggle/working"):
    path = os.path.join(base, "iotest.bin")
    blob = b"0" * (32 << 20)
    t0 = time.time()
    with open(path, "wb") as f:
        for _ in range(8):
            f.write(blob)
    w = time.time() - t0
    t0 = time.time()
    with open(path, "rb") as f:
        while f.read(CHUNK):
            pass
    r2 = time.time() - t0
    os.remove(path)
    print(f"  {base}: write {256 / w:.0f} MiB/s read {256 / r2:.0f} MiB/s")
'''


# --------------------------------------------------------------------------
# 5. VRAM 予算表（量子化対象 / ignore を分けて集計）
# --------------------------------------------------------------------------
VRAM2 = '''import json
import os
import struct

''' + ROOT_FINDER + '''


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace"))


cfg = json.load(open(os.path.join(ROOT, "config.json")))
qc = cfg.get("quantization_config") or {}
ignore = set(qc.get("ignore") or [])

for name in ("model.safetensors", "model-mtp.safetensors"):
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        print("missing:", name)
        continue
    h = header(p)
    h.pop("__metadata__", None)
    q = 0
    nq = 0
    qn = 0
    for k, v in h.items():
        size = v["data_offsets"][1] - v["data_offsets"][0]
        base = k.rsplit(".", 1)[0]
        if base in ignore or any(base.startswith(ig) for ig in ignore):
            nq += size
            qn += 1
        elif k.endswith(("weight_packed", "weight_scale", "weight_shape",
                         "weight_zero_point", "weight_global_scale")):
            q += size
        else:
            nq += size
            qn += 1
    print(f"{name}: 量子化 {q / 2**30:.2f} GiB / 非量子化 {nq / 2**30:.2f} GiB "
          f"({qn} tensors)")

print("\\n=== TP=2 の見積り ===")
W_TOTAL = 19.5
TP = 2
for util, cap in ((0.90, 16.0), (0.92, 16.0), (0.95, 16.0)):
    budget = cap * util
    weights = W_TOTAL / TP
    kv_tok = 2 * 16 * 4 * 256 * 2  # bytes/token
    print(f"  util={util}: budget/GPU={budget:.2f} GiB 重み={weights:.2f} GiB "
          f"残={budget - weights:.2f} GiB")

kv_tok = 2 * 16 * 4 * 256 * 2
for mlen in (8192, 16384, 32768, 65536):
    per_gpu = kv_tok * mlen / TP / 2**30
    print(f"  max_len={mlen:6d}: KV {per_gpu:.2f} GiB/GPU (offset 込み +20% で "
          f"{per_gpu * 1.2:.2f})")

print("\\n=== 起動時ロード量 ===")
for name in ("model.safetensors", "model-mtp.safetensors"):
    p = os.path.join(ROOT, name)
    if os.path.isfile(p):
        print(f"  {name}: {os.path.getsize(p) / 2**30:.2f} GiB (disk)")

print("\\n=== vision tower の重み（ignore 側）===")
h = header(os.path.join(ROOT, "model.safetensors"))
h.pop("__metadata__", None)
vis = sum(v["data_offsets"][1] - v["data_offsets"][0]
          for k, v in h.items() if k.startswith("model.visual."))
print(f"  visual 合計: {vis / 2**30:.2f} GiB "
      f"（--language-model-only で削減できる上限）")
'''


if __name__ == "__main__":
    _write("step2-vllm-pypiwheels", "Step2 vLLM pypiwheels",
           "# 📦 依存 wheel の実在確認\n\n"
           "torch 2.13.0 / torchaudio 2.11.0 が cp313 linux で取れるか。\n",
           PYPIWHEELS, [])
    _write("step2-vllm-specschema", "Step2 vLLM specschema",
           "# 🚀 SpeculativeConfig(mtp) の要求\n\n"
           "必須項目・上限・model 指定の要否。\n",
           SPECSCHEMA, [])
    _write("step2-vllm-tplflag", "Step2 vLLM tplflag",
           "# 🏷 chat-template の実体\n\n"
           "`--chat-template` の定義と .jinja 自動解決の有無。\n",
           TPLFLAG, [])
    _write("step2-vllm-ioread", "Step2 vLLM ioread",
           "# 💽 読み速度と cloudflared\n\n"
           "Dataset 読み出し帯域から起動時間を見積る。\n",
           IOREAD, [DATASET_ID])
    _write("step2-vllm-vram2", "Step2 vLLM vram2",
           "# 📐 VRAM 予算表\n\n"
           "量子化/非量子化の内訳と TP=2 の余裕。\n",
           VRAM2, [DATASET_ID])

    script = HERE / "push_probe7.sh"
    script.write_text(
        "#!/bin/bash\n"
        "# CPU セッション上限 5。空きが出るたびに 1 本ずつ push する。\n"
        "cd \"$(dirname \"$0\")\"\n"
        "for k in " + " ".join(SLUGS) + "; do\n"
        "  for i in $(seq 1 40); do\n"
        "    if (cd \"step2-vllm-$k\" && kaggle kernels push -p . 2>&1 "
        "| tail -1 | grep -q 'successfully pushed'); then\n"
        "      echo \"$k: pushed\"; break\n"
        "    fi\n"
        "    echo \"$k: 待機 ($i)\"; sleep 60\n"
        "  done\n"
        "done\n", encoding="utf-8")
    script.chmod(0o755)
    print(f"wrote {script}")

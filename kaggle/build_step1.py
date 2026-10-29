#!/usr/bin/env python3
"""Step1 notebook builder: orcarouter/Qwen3.8-27B-Uncensored -> Kaggle Dataset"""
import json

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
DATASET_SLUG = "my-ollama-orcarouter-qwen38-27b"
OUT = "step1-orcarouter-qwen38-27b-dl.ipynb"

cells = []

cells.append(("markdown", f"""# 🚀 Step 1: モデルDL → Kaggle Dataset 保存

**初回のみ実行。** 2回目以降は Step 2 を使ってください。

---

### 前提条件（Kaggle Secrets）
Add-ons → Secrets で以下を設定:
- `KAGGLE_USERNAME` = あなたの Kaggle ユーザー名
- `KAGGLE_KEY` = API Key（kaggle.json の key）

### 設定
| 項目 | 値 |
|------|-----|
| モデル | `{MODEL}` |
| サイズ | 16.52 GiB（Q4_K_M / 27.3B / vision + thinking + tools） |
| 保存先 | `/kaggle/tmp/ollama_models` |
| Dataset | `{DATASET_SLUG}` |

> **このモデルについて**
> Abliterlitics の Qwen3.8-27B 12変体比較で総合1位（judge ASR 82.2%、capability 劣化なし）。
> 手元の `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` とは **sha256 が完全一致**
> （`3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`）。
"""))

cells.append(("markdown", "---\n### ⚙️ 設定\n"))

cells.append(("code", f'''# ============================================================
# Step 1: モデルDL → Kaggle Dataset 保存
# ============================================================
import os, subprocess, time

MODEL = "{MODEL}"
DATASET_SLUG = "{DATASET_SLUG}"
OLLAMA_MODELS_DIR = "/kaggle/tmp/ollama_models"

os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)
os.environ["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR

# Kaggle Secrets から認証情報を取得
from kaggle_secrets import UserSecretsClient
_sec = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = _sec.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = _sec.get_secret("KAGGLE_KEY")

print(f"📦 Model      : {{MODEL}}")
print(f"📁 Save to    : {{OLLAMA_MODELS_DIR}}")
print(f"🗂  Dataset    : {{DATASET_SLUG}}")
print(f"👤 Owner      : {{os.environ['KAGGLE_USERNAME']}}")
'''))

cells.append(("markdown", "---\n### 📦 Ollama インストール\n"))

cells.append(("code", '''# Ollama インストール（zstd を先に入れる: 最新版は .tar.zst）
import subprocess

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True, timeout=600)

# インストール結果の確認
for p in ["/usr/local/bin/ollama", "/usr/local/lib/ollama/llama-server"]:
    ok = os.path.isfile(p)
    size = os.path.getsize(p) if ok else 0
    print(f"{'✅' if ok else '❌'} {p}  ({size:,} B)")

print(subprocess.run(["/usr/local/bin/ollama", "--version"],
                     capture_output=True, text=True).stdout.strip())
'''))

cells.append(("markdown", "---\n### ⬇️ モデルダウンロード（16.5 GiB / 10〜30分）\n"))

cells.append(("code", f'''# Ollama サーバー起動 → モデル pull
import subprocess, os, time

OLLAMA_BIN = "/usr/local/bin/ollama"
env = {{**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS_DIR}}

with open("/tmp/ollama_serve.log", "w") as log:
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                            stdout=log, stderr=log)

# 起動待機（最大60秒）
for i in range(60):
    r = subprocess.run([OLLAMA_BIN, "list"], env=env,
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ Ollama サーバー起動（{{i+1}}s）")
        break
    time.sleep(1)
else:
    print(open("/tmp/ollama_serve.log").read()[-2000:])
    raise RuntimeError("Ollama サーバー起動失敗")

print(f"⏳ pull: {{MODEL}}  （16.5 GiB / 10〜30分）")
t0 = time.time()
subprocess.run([OLLAMA_BIN, "pull", MODEL], env=env, check=True, timeout=7200)
print(f"✅ ダウンロード完了（{{(time.time()-t0)/60:.1f}} 分）")

print(subprocess.run([OLLAMA_BIN, "list"], env=env,
                     capture_output=True, text=True).stdout)
'''))

cells.append(("markdown", "---\n### 💾 バイナリ + モデルを Dataset へ\n"))

cells.append(("code", '''# バイナリをキャッシュ（次回起動を高速化）
import os, shutil, subprocess

BIN_CACHE = os.path.join(OLLAMA_MODELS_DIR, "bin")
os.makedirs(BIN_CACHE, exist_ok=True)

# ollama 本体 / llama-server / ngrok を全部キャッシュする
# （llama-server を忘れると Step2 で毎回 install.sh を走らせることになる）
CANDIDATES = [
    "/usr/local/bin/ollama",
    "/usr/local/lib/ollama/llama-server",
    "/usr/local/bin/ngrok",
]
for src in CANDIDATES:
    if os.path.isfile(src):
        dst = os.path.join(BIN_CACHE, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"✅ cache: {os.path.basename(src)}  ({os.path.getsize(dst):,} B)")
    else:
        print(f"—  skip : {src}（未インストール）")

subprocess.run(f"du -sh {OLLAMA_MODELS_DIR}", shell=True)
subprocess.run("df -h /kaggle/tmp", shell=True)
'''))

cells.append(("code", f'''# Dataset としてアップロード
import json, os, subprocess

DATASET = "{DATASET_SLUG}"
OWNER = os.environ["KAGGLE_USERNAME"]

meta = {{
    "title": DATASET,
    "id": f"{{OWNER}}/{{DATASET}}",
    "licenses": [{{"name": "CC0-1.0"}}],
    "subtitle": "Ollama model: orcarouter/Qwen3.8-27B-Uncensored (Q4_K_M)",
    "description": (
        "orcarouter/Qwen3.8-27B-Uncensored:latest — Qwen3.8 27B abliterated, "
        "Q4_K_M, vision + thinking + tools, 262K context. "
        "Includes ollama/llama-server binary cache for fast Kaggle startup."
    ),
}}
with open(f"{{OLLAMA_MODELS_DIR}}/dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("📤 kaggle datasets create（--dir-mode zip）...")
r = subprocess.run(
    ["kaggle", "datasets", "create", "-p", OLLAMA_MODELS_DIR, "--dir-mode", "zip"],
    capture_output=True, text=True, timeout=7200,
)
out = r.stdout + r.stderr
clean = [l for l in out.split("\\n") if "%|" not in l and "[" not in l]
print("\\n".join(clean[-25:]))
print(f"exit: {{r.returncode}}")
if r.returncode != 0:
    raise RuntimeError("Dataset 作成失敗（既存の場合は datasets version を使ってください）")
print("✅ Dataset アップロード完了")
'''))

cells.append(("markdown", f"""---
### ✅ 次の手順

1. この Notebook を **Save Version → Save & Run All** で実行
2. 完了後、Dataset `{DATASET_SLUG}` が作成されているのを確認
3. **Step 2** の Notebook を開き、Add Data → この Dataset を追加
4. Accelerator を `GPU T4 x2`、Internet を `On` にして Step 2 を実行

> Dataset を更新したい場合は `datasets create` ではなく
> `kaggle datasets version -p <dir> -m "message" --dir-mode zip` を使います。
"""))

nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if t == "code" else {}),
        }
        for t, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}: {len(cells)} cells")

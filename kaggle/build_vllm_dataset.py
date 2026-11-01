#!/usr/bin/env python3
"""vLLM 用 AWQ+MTP モデルを Kaggle Dataset にするノートブックを生成する。

実行方法:
    python3 kaggle/build_vllm_dataset.py
    cd kaggle/step2-vllm
    kaggle kernels push

Kaggle 上でノートブックを実行すると、HuggingFace からモデルをダウンロードし、
Kaggle Dataset として公開する。
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kbuild import metadata, pack_nb  # noqa: E402
OUT_DIR = HERE / "step2-vllm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NB_PATH = OUT_DIR / "prepare-vllm-dataset.ipynb"
META_PATH = OUT_DIR / "kernel-metadata.json"

OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
DATASET_ID = f"{OWNER}/{DATASET_SLUG}"
MODEL_ID = "twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP"


def _cell(cell_type: str, source: str):
    d = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


cells = []

cells.append(_cell("markdown", f"""# vLLM 用 AWQ+MTP モデル Dataset 作成

HuggingFace から `{MODEL_ID}` をダウンロードし、Kaggle Dataset 化する。

**手順**

1. このノートブックを Kaggle で実行（GPU不要・Internet On）
2. 出力された Dataset URL を控える
3. `build_step2_vllm.py` でその Dataset ID を指定して Step2 notebook を生成

**事前準備**

- Kaggle Secrets に `HF_TOKEN` を登録（gated モデルの場合）
- 必要なら HuggingFace 側で対象モデルの利用条件に同意済みであること
"""))

cells.append(_cell("code", """# 必要なパッケージ
# `!pip` ではなく subprocess で呼ぶ（script 化しても同じコードで動く）
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub", "kaggle"], check=False)"""))

cells.append(_cell("code", f"""import os, json, subprocess, time
from huggingface_hub import snapshot_download

MODEL_ID = "{MODEL_ID}"
DATASET_ID = "{DATASET_ID}"
DATASET_SLUG = "{DATASET_SLUG}"

# 置き場所は /kaggle/tmp（作業用の大きいディスク）。
# `/kaggle/working` は **19.5 GB 上限**で、このモデルは 19.57 GB あるため
# 途中で disk full になり、`.cache/huggingface/download/*.lock` だけが残る
# （実測: 2026-10-30 の Dataset がまさにその状態だった）。
OUT_DIR = "/kaggle/tmp/{DATASET_SLUG}"

# Kaggle Secrets から HF_TOKEN を取得（このモデルは public なので無くても落ちる）
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("ℹ HF_TOKEN 未設定。public リポジトリなのでこのまま続行する")
    HF_TOKEN = None
else:
    print("✅ HF_TOKEN を取得しました")

print(f"ダウンロード先: {{OUT_DIR}}")
print(f"モデル: {{MODEL_ID}}")
_sp.run(["df", "-h", OUT_DIR if os.path.isdir(OUT_DIR) else "/kaggle"], check=False)
"""))

cells.append(_cell("code", """# モデルダウンロード
# 合計 19.57 GB（model.safetensors 18.70 + model-mtp 0.85）。数分かかる。
# `local_dir_use_symlinks` / `resume_download` は新しい huggingface_hub で
# 非推奨・削除済みなので渡さない（渡すと TypeError で落ちる版がある）。
start = time.time()
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=OUT_DIR,
    token=HF_TOKEN,
)
elapsed = time.time() - start
print(f"✅ ダウンロード完了: {elapsed/60:.1f}分")

# サイズ確認
_sp.run(["du", "-sh", OUT_DIR], check=False)
_sp.run(["ls", "-la", OUT_DIR], check=False)

# 重みの実在確認。ここが空のまま進むと「.cache だけの Dataset」ができる
# （2026-10-30 に実際に起きた）。無ければ後続を走らせずに止める。
weights = [f for f in os.listdir(OUT_DIR) if f.endswith(".safetensors")]
total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in weights)
print(f"重み: {len(weights)} 本 / {total/1e9:.2f} GB {weights}")
assert total > 10e9, f"重みが足りません（{total/1e9:.2f} GB）。disk full を疑う"
"""))

cells.append(_cell("code", f"""# Kaggle Dataset メタデータ作成
meta = {{
    "title": "Qwen3.8-27B-Uncensored-HauhauCS-AWQ-MTP",
    "id": "{DATASET_ID}",
    "licenses": [{{"name": "Apache 2.0"}}],
}}
with open(f"{{OUT_DIR}}/dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ dataset-metadata.json 作成: {{OUT_DIR}}/dataset-metadata.json")
"""))

cells.append(_cell("code", """# Kaggle Dataset 作成
# 既存 Dataset がある場合は version コマンドに切り替える
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

try:
    try:
        api.dataset_create_new(
            folder=OUT_DIR,
            convert_to_csv=False,
            dir_mode="zip",
        )
        print(f"✅ Dataset 作成完了: https://www.kaggle.com/datasets/{DATASET_ID}")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print("既存 Dataset 検出。新バージョン作成中...")
            api.dataset_create_version(
                folder=OUT_DIR,
                version_notes="update from build_vllm_dataset",
                convert_to_csv=False,
                dir_mode="zip",
            )
            print(f"✅ Dataset 新バージョン作成完了: https://www.kaggle.com/datasets/{DATASET_ID}")
        else:
            raise
except Exception as e:
    print(f"⚠ 作成エラー: {e}")
    print("Kaggle 認証/kaggle.json、Dataset ID、権限を確認してください。")
"""))

cells.append(_cell("markdown", f"""---

## 完了後

Dataset URL: `https://www.kaggle.com/datasets/{DATASET_ID}`

この URL を `build_step2_vllm.py` の `DATASET_ID` に設定して Step2 notebook を生成する。
"""))

if __name__ == "__main__":
    pack_nb.write_cells(NB_PATH, cells)
    kernel_meta = {
        "id": f"{OWNER}/prepare-vllm-dataset",
        "title": "Prepare vLLM Dataset",
        "code_file": "prepare-vllm-dataset.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }

    metadata.write(META_PATH, kernel_meta)
    print(f"wrote {META_PATH}")

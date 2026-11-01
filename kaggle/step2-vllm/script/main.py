#!/usr/bin/env python3
"""prepare vllm dataset script（自動生成。直接編集しない）。

生成: kaggle/build_batch_script.py ← kaggle/build_vllm_dataset.py のセル列
"""


# ── vLLM 用 AWQ+MTP モデル Dataset 作成 ───────────────


# 必要なパッケージ
# `!pip` ではなく subprocess で呼ぶ（script 化しても同じコードで動く）
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub", "kaggle"], check=False)


import os, json, subprocess, time
from huggingface_hub import snapshot_download

MODEL_ID = "twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP"
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"

# 置き場所は /kaggle/tmp（作業用の大きいディスク）。
# `/kaggle/working` は **19.5 GB 上限**で、このモデルは 19.57 GB あるため
# 途中で disk full になり、`.cache/huggingface/download/*.lock` だけが残る
# （実測: 2026-10-30 の Dataset がまさにその状態だった）。
OUT_DIR = "/kaggle/tmp/qwen38-27b-uncensored-hauhaucs-awq-mtp"

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

print(f"ダウンロード先: {OUT_DIR}")
print(f"モデル: {MODEL_ID}")
_sp.run(["df", "-h", OUT_DIR if os.path.isdir(OUT_DIR) else "/kaggle"], check=False)


# モデルダウンロード
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


# Kaggle Dataset メタデータ作成
meta = {
    "title": "Qwen3.8-27B-Uncensored-HauhauCS-AWQ-MTP",
    "id": "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
    "licenses": [{"name": "Apache 2.0"}],
}
with open(f"{OUT_DIR}/dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ dataset-metadata.json 作成: {OUT_DIR}/dataset-metadata.json")


# Kaggle Dataset 作成
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


# ── 完了後 ─────────────────────────────────────────

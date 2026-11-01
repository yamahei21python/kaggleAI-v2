# 接続情報（クライアント側の設定にコピペしてください）
# セル出力・ログは後から取れないので /kaggle/working に保存する（/output にも複製）。
import json

try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

# コンテキスト長はエンジンが決める（未定義ならエンジン既定と表示）
_ctx = f"{CONTEXT_LENGTH:,}" if "CONTEXT_LENGTH" in globals() else "エンジン既定"

opencode_cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "kaggle": {
            "npm": "@ai-sdk/openai-compatible",
            "name": f"Kaggle {ENGINE_DISPLAY}",
            "options": {"baseURL": f"{url}/v1"},
            "models": {MODEL_NAME: {"name": MODEL_DISPLAY}},
        }
    },
    "model": f"kaggle/{MODEL_NAME}",
}

buf = [
    "=" * 64,
    f"  🌍 Public URL : {url}",
    f"  💬 チャット UI : {url}/",
    f"  🤖 Model      : {MODEL_NAME}",
    f"  📏 Context    : {_ctx}",
    "  🔓 認証        : なし（URL を知れば誰でも使える）",
    "=" * 64,
    "",
    "📋 curl:",
    f"  curl {url}/v1/models",
    "",
    "📋 Python（OpenAI 互換クライアント）:",
    "  import httpx",
    "  from openai import OpenAI",
    "  client = OpenAI(",
    f"      base_url='{url}/v1',",
    "      api_key='dummy',",
    "  )",
    f"  client.chat.completions.create(model='{MODEL_NAME}', messages=[...])",
    "",
    "📋 opencode.json:",
    json.dumps(opencode_cfg, indent=2, ensure_ascii=False),
    "",
    "⚠ quick tunnel: 稼働保証なし。切れたらトンネルセルを再実行",
    "⚠ 公開URLは実行のたびに変わる。再実行後は接続情報を取り直す",
    "⚠ Kaggle: 9時間でカーネルが停止します",
]

report = "\n".join(buf)
print(report)
if not save_output("connection_info.txt", report + "\n"):
    # 接続情報は唯一の利用手掛かりなので、取れなければ成功にしない
    fail("connection_info.txt を永続領域に保存できませんでした")

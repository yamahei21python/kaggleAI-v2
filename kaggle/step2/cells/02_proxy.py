# リバースプロキシを起動（UI を / で配信し、API を同一オリジンにする）
# 認証は既定で無効（環境変数 API_PASS を設定したときだけ有効）
# 起動・待機・認証なし検査の実装は共通ランタイム（runtime/proxy.py）。
import os

# 認証なし（proxy は API_PASS が空なら素通しする）
proxy_env = {**os.environ, "API_USER": "", "API_PASS": ""}
proxy_proc = kc_start_proxy(
    PROXY_PORT, f"http://127.0.0.1:{PORT}", CHAT_PATH, PROXY_PATH,
    env=proxy_env, fail=fail,
)

# 認証は付けない（設計）。401 が返るなら設定ミス。
# 200 は Ollama 起動前だと 502 になるため、ここでは 401/403 でないことだけ見る。
# 完全判定（200 / streaming / OpenAI互換）は公開URL経由の verify セルへ一本化。
kc_assert_no_auth(PROXY_PORT, fail=fail)

# リバースプロキシを起動（UI を / で配信し、API を同一オリジンにする）
# 起動・待機・認証なし検査の実装は共通ランタイム（runtime/proxy.py）。
# 中継先・追加引数・認証は settings セルが定義する（PROXY_* / UPSTREAM_PORT）。
import os

# 認証は既定で無効。PROXY_AUTH に API_PASS が入っているときだけ有効になる。
proxy_env = {**os.environ, **PROXY_AUTH}
proxy_proc = kc_start_proxy(
    PROXY_PORT, f"http://127.0.0.1:{UPSTREAM_PORT}", CHAT_PATH, PROXY_PATH,
    extra_args=PROXY_EXTRA_ARGS, env=proxy_env, fail=fail,
)

# 推論サーバ起動前なので 5xx は正常。401/403 が返るなら認証の設定ミス。
# 完全判定（200 / streaming / OpenAI 互換）は公開URL経由の verify セルへ一本化。
kc_assert_no_auth(PROXY_PORT, path=PROXY_CHECK_PATH, fail=fail)

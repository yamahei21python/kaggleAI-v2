"""ポート等の固定値（真実源）。

カーネル内で単体実行されるファイル（`auth_proxy.py` / `chat_server.py`）は
import できないためリテラルを持つが、値の一致は `tests/test_constants.py` が
検査する。ビルダー側はこのモジュールから引く。
"""

# 推論エンジンの待受（公開しない）
CHAT_PORT = 9931          # exllamav3 chat_server / llama.cpp / ローカル uzu
OLLAMA_PORT = 11434
VLLM_PORT = 8000

# 認証プロキシ（公開側）
PROXY_PORT = 11435

# ローカル（macOS）
UI_PORT = 8899            # llama-chat.html の中継
RELAY_PORT = 19950        # run-kaggle-chat の中継（ローカル）

# テスト・検証用
KEEPALIVE_TEST_PORT = 8080
LOCAL_TEST_UPSTREAM = 18901
LOCAL_TEST_PROXY = 18902

# 時間予算（分）
EXL3_RUN_MINUTES = 60
EXL3_HARD_MAX_MINUTES = 180

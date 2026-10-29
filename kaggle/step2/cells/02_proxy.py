# リバースプロキシを起動（UI を / で配信し、API を同一オリジンにする）
# 認証は既定で無効（環境変数 API_PASS を設定したときだけ有効）
# import os は単体実行でも NameError にならないように（設定セル非依存に）
import os, subprocess, time, urllib.request, urllib.error

# 前回のプロキシを落とす
subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
time.sleep(1)

proxy_env = {
    **os.environ,
    # 認証なし（proxy は API_PASS が空なら素通しする）
    "API_USER": "",
    "API_PASS": "",
}
with open("/tmp/auth_proxy.log", "w") as log:
    proxy_proc = subprocess.Popen(
        ["python3", PROXY_PATH,
         "--port", str(PROXY_PORT),
         "--upstream", f"http://127.0.0.1:{PORT}",
         "--chat-html", CHAT_PATH],
        env=proxy_env, stdout=log, stderr=log,
    )
print(f"🔄 proxy PID: {proxy_proc.pid}")

# 起動待機（/healthz は上流を見ないので即応答する）
for i in range(30):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PROXY_PORT}/healthz", timeout=3
        ) as r:
            if r.status == 200:
                print(f"✅ プロキシ起動完了（{i+1}s）")
                break
    except Exception:
        pass
    time.sleep(1)
else:
    print(open("/tmp/auth_proxy.log").read()[-2000:])
    fail("認証プロキシの起動に失敗しました", "/tmp/auth_proxy.log")

# 認証は付けない（設計）。401 が返るなら設定ミス。
# 200 は Ollama 起動前だと 502 になるため、ここでは 401/403 でないことだけ見る。
# 完全判定（200 / streaming / OpenAI互換）は公開URL経由の verify セルへ一本化。
req = urllib.request.Request(f"http://127.0.0.1:{PROXY_PORT}/api/tags")
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"  ✅ 認証なし → HTTP {r.status}（401 でない）")
except urllib.error.HTTPError as e:
    if e.code in (401, 403):
        fail(f"認証なしで HTTP {e.code} — 認証が有効になっています")
    else:
        print(f"  ✅ 認証なし → HTTP {e.code}（上流未起動。401 でない）")
except Exception as e:
    fail(f"プロキシ検査に失敗: {type(e).__name__}: {e}")

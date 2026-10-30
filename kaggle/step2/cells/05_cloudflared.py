# cloudflared quick tunnel 起動 → 公開URL取得
# アカウント不要。authtoken も Secrets も要らない。
import os, subprocess, time

CF_BIN = "/usr/local/bin/cloudflared"
TUNNEL_NAME = "cloudflared"
TUNNEL_LOG = "/tmp/cloudflared.log"
RESTART_CMD = [CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]


# 前回のトンネルを確実に落とす（多重起動すると URL が混ざる）
subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as log:
    cf_proc = subprocess.Popen(RESTART_CMD, stdout=log, stderr=log)
print(f"🔄 cloudflared PID: {cf_proc.pid}")

# ---- 公開URL取得（ログに URL が出るまで待つ）----
public_url = kc_wait_url("cloudflared", TUNNEL_LOG)

if not public_url:
    # ここで止めないと後段が /tmp/public_url.txt なしでエラーになり、
    # どこで失敗したか分からなくなる。原因ログも永続領域に残す。
    print("⚠ 公開URLを取得できませんでした")
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("cloudflared の公開URL取得に失敗しました", TUNNEL_LOG)

# ---- トンネル確立まで待つ ----
# URL が出た直後はまだエッジに伝播しておらず、外から叩くと 000 になる。
# ここで待たないと次のセルの疎通確認が誤って失敗する。
print(f"🌍 URL: {public_url}")
print("⏳ トンネル確立待ち（最大 60s）...")
elapsed = kc_wait_http(f"{public_url}/healthz", attempts=20, interval=3)
if elapsed is not None:
    print(f"✅ トンネル確立（{elapsed}s）")
else:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
# 完了後に `kaggle kernels output` で取り出す用
save_output("public_url.txt", public_url + "\n")
print()
print("=" * 60)
print(f"  🌍 PUBLIC URL : {public_url}")
print("=" * 60)

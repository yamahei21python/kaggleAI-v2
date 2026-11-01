# ngrok 起動 → 公開URL取得（authtoken は Secrets の NGROK_TOKEN）
# 固定ドメインの既定値は settings セルが NGROK_DOMAIN_DEFAULT として持つ。
import json, os, subprocess, time, urllib.request

NGROK_BIN = "/usr/local/bin/ngrok"

# patch Dataset には入れていないので取得する（約 30 MB / 数秒）
if not os.path.isfile(NGROK_BIN):
    subprocess.run(
        ["bash", "-c",
         "curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
         " | tar xz -C /usr/local/bin"],
        check=True)

# ---- authtoken ----
_tok = os.environ.get("NGROK_TOKEN", "")
if not _tok:
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("NGROK_TOKEN") or ""
    except Exception:
        _tok = ""
if not _tok:
    fail("NGROK_TOKEN がありません（Kaggle Secrets に登録してください）")
subprocess.run([NGROK_BIN, "config", "add-authtoken", _tok],
               check=True, capture_output=True)

# ---- 固定ドメイン ----
# 優先順: 環境変数 NGROK_DOMAIN → Secrets NGROK_DOMAIN → 埋め込み既定値。
# 既定値を埋めているので、未設定でも必ず固定URLになる（Secret は上書き用）。
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()
if not NGROK_DOMAIN:
    try:
        from kaggle_secrets import UserSecretsClient
        NGROK_DOMAIN = (UserSecretsClient().get_secret("NGROK_DOMAIN") or "").strip()
    except Exception:
        NGROK_DOMAIN = ""
if not NGROK_DOMAIN:
    NGROK_DOMAIN = NGROK_DOMAIN_DEFAULT
print("📌 固定ドメイン: " + (NGROK_DOMAIN or "（未設定 → ランダムURL）"))

TUNNEL_NAME = "ngrok"
TUNNEL_LOG = "/tmp/ngrok.log"
# --pooling-enabled が無いと、同じ固定ドメインが前回セッションから残っている
# ときに ERR_NGROK_334 で即死する（実測）。
# プーリングを有効にすると複数エンドポイントを束ねられるので衝突しない。
RESTART_CMD = [NGROK_BIN, "http", str(PROXY_PORT),
               "--log=stdout", "--pooling-enabled"]
if NGROK_DOMAIN:
    # --domain は deprecated（実測で警告）。--url が現行。
    RESTART_CMD.append("--url=https://" + NGROK_DOMAIN)


def probe_url():
    """ngrok のローカル管理APIから公開URLを取る（共通ランタイム）。"""
    return kc_probe_url("ngrok")


# 前回のトンネルを確実に落とす（多重起動すると URL が混ざる）
subprocess.run(["pkill", "-x", "ngrok"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as _log:
    ng_proc = subprocess.Popen(RESTART_CMD, stdout=_log, stderr=_log)
print("🔄 ngrok PID: " + str(ng_proc.pid))

public_url = kc_wait_url("ngrok", TUNNEL_LOG)

if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("ngrok の公開URL取得に失敗しました", TUNNEL_LOG)

print("🌍 URL: " + public_url)
print("⏳ トンネル確立待ち（最大 60s）...")
elapsed = kc_wait_http(public_url + "/healthz", attempts=20, interval=3,
                       opener=kc_urlopen_v4)
if elapsed is not None:
    print("✅ トンネル確立（" + str(elapsed) + "s）")
else:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\n")
print()
print("=" * 60)
print("  🌍 PUBLIC URL : " + public_url)
print("=" * 60)

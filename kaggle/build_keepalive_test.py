#!/usr/bin/env python3
"""keepalive の ON/OFF + 外部受信確認用（GPU 不要・トンネル不要）"""
import base64
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = str(HERE / "step2" / "step2-keepalive-test.ipynb")
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)

# auth_proxy.py 全体を埋め込む
PROXY_SRC_B64 = base64.b64encode(
    (HERE / "step2" / "auth_proxy.py").read_bytes()
).decode()

# llama-chat.html を埋め込む
CHAT_SRC_B64 = base64.b64encode(
    (HERE.parent / "llama-chat.html").read_bytes()
).decode()


def _b64_chunks(b64_str, size=1000):
    """長い base64 を複数行の文字列リテラルに分ける。"""
    lines = [b64_str[i:i+size] for i in range(0, len(b64_str), size)]
    return "\n".join(f'    "{line}"' for line in lines)


KEEPALIVE_CELL = '''# 🔋 キープアライブ ON/OFF 自動検証（GPU 不要）
# /tmp/keepalive.conf を自動操作して、残り時間表示 + リセット + 停止を検証。
import subprocess, time, re, json, urllib.request, urllib.error

MAX_MINUTES = 480
KEEPALIVE_CONF = "/tmp/keepalive.conf"
KEEPALIVE_KEEPALIVE = 60
SLEEP_SEC = 3                     # GPU なしなので 3 秒ごと（実機では 60）
RESET_AT = 10                     # 10 秒後にチャット送信シミュレーション
STOP_AT = 25                      # 25 秒後に停止ボタンシミュレーション


def _read_keepalive_conf():
    """RUN=1/0 と MINUTES を読み取り（デフォルトは 1 と KEEPALIVE_KEEPALIVE）。"""
    try:
        with open(KEEPALIVE_CONF) as f:
            lines = [l.strip() for l in f if l.strip() and "=" in l]
        d = dict(l.split("=", 1) for l in lines)
        run = int(d.get("RUN", 1))
        minutes = int(d.get("MINUTES", KEEPALIVE_KEEPALIVE))
    except FileNotFoundError:
        run, minutes = 1, KEEPALIVE_KEEPALIVE
    except Exception:
        run, minutes = 1, KEEPALIVE_KEEPALIVE
    if run <= 0:
        return 0, KEEPALIVE_KEEPALIVE
    if minutes <= 0:
        print(f"⚠ KEEPALIVE_MINUTES={minutes} は無効（1〜{MAX_MINUTES} で指定）")
        return 0, KEEPALIVE_KEEPALIVE
    if minutes > MAX_MINUTES:
        print(f"⚠ KEEPALIVE_MINUTES={minutes} → {MAX_MINUTES} に制限")
        minutes = MAX_MINUTES
    return run, minutes


def _reset_keepalive():
    """チャット送信ごとに時間リセット（ON 維持）。"""
    try:
        with open(KEEPALIVE_CONF, "w") as f:
            f.write(f"RUN=1\\nMINUTES={KEEPALIVE_KEEPALIVE}\\n")
        print("🔄 keepalive リセット → 60 分")
        return True
    except Exception as e:
        print(f"⚠ リセット失敗: {e}")
        return False


def _stop_keepalive():
    """停止ボタン：RUN=0 にする。"""
    try:
        with open(KEEPALIVE_CONF, "w") as f:
            f.write("RUN=0\\n")
        print("🛑 keepalive 停止 → RUN=0")
        return True
    except Exception as e:
        print(f"⚠ 停止失敗: {e}")
        return False


# 初期化
_reset_keepalive()
start = time.time()
last_reset = start
RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()
print(f"初期: RUN={RUN} MINUTES={KEEPALIVE_MINUTES}")
print(f"🔋 ON。{KEEPALIVE_KEEPALIVE}分間維持")
print(f"⏱ {RESET_AT}秒後にリセット、{STOP_AT}秒後に停止")
print()

log = []
reset_done = stop_done = False
i = 0
while True:
    i += 1
    now = time.time()
    elapsed = now - start
    RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()

    if not RUN:
        print(f"RUN=0 → {i}周目（{elapsed:.0f}秒経過）に終了")
        log.append(f"[{elapsed:.0f}s] RUN=0 → 終了")
        break

    # 自動イベント
    if not reset_done and elapsed >= RESET_AT:
        if _reset_keepalive():
            last_reset = time.time()
            reset_done = True
    if not stop_done and elapsed >= STOP_AT:
        if _stop_keepalive():
            stop_done = True

    # 残り時間（最後のリセットから）
    remaining_sec = max(0, KEEPALIVE_MINUTES * 60 - (now - last_reset))
    remaining_min = int(remaining_sec // 60)
    remaining_sec_only = int(remaining_sec % 60)

    line = f"[{elapsed:5.1f}s] alive | 残り {remaining_min:3d}分{remaining_sec_only:02d}秒"
    print(line)
    log.append(line)
    time.sleep(SLEEP_SEC)

report = "\\n".join([
    f"result={'stopped' if not RUN else 'alive'}",
    f"RUN={RUN}",
    f"minutes={KEEPALIVE_MINUTES}",
    f"elapsed={elapsed:.0f}s",
    f"reset_done={reset_done}",
    f"stop_done={stop_done}",
    "",
] + log)

with open("keepalive_test_result.txt", "w") as f:
    f.write(report)
print()
print("== keepalive_test_result.txt 保存 ==")
print(report)
'''


EXTERNAL_CELL_TEMPLATE = '''# 🌐 外部受信テスト（GPU 不要・Internet On）
# auth_proxy.py + cloudflared quick tunnel で、インターネットからのアクセスを確認。
import base64, os, subprocess, time, re, urllib.request, json

PROXY_SRC_B64 = (
__PROXY_B64__
)
CHAT_SRC_B64 = (
__CHAT_B64__
)

# --- cloudflared 準備 ---
CF_BIN = "/usr/local/bin/cloudflared"
if not os.path.exists(CF_BIN):
    print("cloudflared をダウンロード中...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, CF_BIN)
    os.chmod(CF_BIN, 0o755)
    print(f"✅ {CF_BIN} ダウンロード完了")

# --- auth_proxy.py と chat.html をデコード ---
with open("/tmp/auth_proxy.py", "wb") as f:
    f.write(base64.b64decode(PROXY_SRC_B64))
with open("/tmp/llama-chat.html", "wb") as f:
    f.write(base64.b64decode(CHAT_SRC_B64))
print("✅ proxy / chat ui デコード完了")

# --- プロキシ起動 ---
proxy_proc = subprocess.Popen(
    ["python3", "/tmp/auth_proxy.py", "--port", "8080",
     "--upstream", "http://127.0.0.1:11434",
     "--chat-html", "/tmp/llama-chat.html", "--no-quota"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
print(f"🚀 proxy PID: {proxy_proc.pid}")
time.sleep(2)

# --- cloudflared 起動 ---
TUNNEL_LOG = "/tmp/cloudflared.log"
subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)
with open(TUNNEL_LOG, "w") as log:
    cf_proc = subprocess.Popen([CF_BIN, "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"],
                                stdout=log, stderr=log)
print(f"🚀 cloudflared PID: {cf_proc.pid}")

# --- 公開URL取得 ---
URL_RE = re.compile(r"https://[a-z0-9-]+\\.trycloudflare\\.com")
public_url = ""
for i in range(90):
    try:
        with open(TUNNEL_LOG) as f:
            m = URL_RE.search(f.read())
        if m:
            public_url = m.group(0)
            break
    except FileNotFoundError:
        pass
    time.sleep(1)
    if i % 10 == 9:
        print(f"⏳ URL 待ち ({i+1}/90)...")

if not public_url:
    print("cloudflared ログ（末尾 2000 文字）:")
    try:
        print(open(TUNNEL_LOG).read()[-2000:])
    except FileNotFoundError:
        print("ログファイルなし")
    raise RuntimeError("公開URL取得失敗")

print(f"🌍 Public URL: {public_url}")

# --- DNS 伝播待ち ---
print()
print("DNS 伝播待ち 10秒...")
time.sleep(10)

# --- 接続情報保存 ---
with open("public_url.txt", "w") as f:
    f.write(public_url + "\\n")
with open("connection_info.txt", "w") as f:
    f.write(f"URL: {public_url}\\n")
    f.write(f"proxy PID: {proxy_proc.pid}\\n")
    f.write(f"cloudflared PID: {cf_proc.pid}\\n")

# --- 外部疎通テスト（自分自身へアクセス） ---
print()
print("=" * 60)
print(f"  公開URL: {public_url}")
print("=" * 60)
print()

print("外部受信テスト：公開URLに自分自身からアクセス...")
for path in ["/healthz", "/quota"]:
    ok = False
    for attempt in range(1, 11):
        try:
            with urllib.request.urlopen(f"{public_url}{path}", timeout=15) as r:
                body = r.read().decode()[:200]
                print(f"GET {path}: {r.status} {body}")
                ok = True
                break
        except Exception as e:
            print(f"GET {path} attempt {attempt}/10: {type(e).__name__}")
            time.sleep(3)
    if not ok:
        print(f"GET {path}: 10 回リトライ後も失敗")

# --- 停止テスト ---
print()
print("POST /stop ...")
try:
    req = urllib.request.Request(f"{public_url}/stop", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read().decode()
        print(f"/stop: {r.status} {body[:200]}")
except Exception as e:
    print(f"/stop error: {type(e).__name__}: {e}")

# /stop 後に再アクセス → 落ちているはず
time.sleep(3)
print()
print("停止後疎通テスト...")
try:
    with urllib.request.urlopen(f"{public_url}/healthz", timeout=5) as r:
        print(f"停止後 /healthz: {r.status} （まだ生きている）")
except Exception as e:
    print(f"停止後 /healthz error: {type(e).__name__} ✅ 落ちた")

# 掃除
proxy_proc.terminate()
cf_proc.terminate()
print("プロセス終了")
'''

EXTERNAL_CELL = EXTERNAL_CELL_TEMPLATE \
    .replace("__PROXY_B64__", _b64_chunks(PROXY_SRC_B64)) \
    .replace("__CHAT_B64__", _b64_chunks(CHAT_SRC_B64))


cells = []
cells.append(("markdown", """# 🔋 キープアライブ ON/OFF + 外部受信検証（GPU 不要）

GPU を使わず、以下を確認するノート。

1. keepalive の **残り時間表示・リセット・停止**
2. auth_proxy.py + cloudflared quick tunnel での **外部からのアクセス受信**

---

## セル 1: keepalive 自動検証ループ

"""))
cells.append(("code", KEEPALIVE_CELL))
cells.append(("markdown", """---

## セル 2: 外部受信テスト

auth_proxy.py + cloudflared quick tunnel を立てて、公開URLに自分自身からアクセス。

**自動で確認すること**

- `/healthz` → 200
- `/quota` → 200（`--no-quota` なので available=false）
- `POST /stop` → 200
- 停止後の `/healthz` → 接続失敗（プロキシが落ちた）

"""))
cells.append(("code", EXTERNAL_CELL))
cells.append(("markdown", """---

## セル 3: 結果確認

"""))
cells.append(("code", """# 結果ファイルを表示
for fname in ["keepalive_test_result.txt", "public_url.txt", "connection_info.txt"]:
    print(f"=== {fname} ===")
    try:
        print(open(fname).read())
    except FileNotFoundError:
        print("まだありません")
    print()
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

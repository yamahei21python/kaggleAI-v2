"""auth_proxy の起動と認証なし検査（生成コードへ埋め込む）。"""
import subprocess
import time
import urllib.error
import urllib.request


def kc_start_proxy(port, upstream, chat_html, proxy_path,
                   extra_args=(), env=None, log_path="/tmp/auth_proxy.log",
                   wait_seconds=30, label="プロキシ", fail=None):
    """auth_proxy.py を起動し /healthz を待つ。失敗時は fail() か RuntimeError。"""
    subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
    time.sleep(1)
    cmd = ["python3", proxy_path, "--port", str(port),
           "--upstream", upstream, "--chat-html", chat_html, *extra_args]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    print(f"🔄 proxy PID: {proc.pid}")

    elapsed = None
    for i in range(wait_seconds):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
                if r.status == 200:
                    elapsed = i + 1
                    break
        except Exception:
            pass
        time.sleep(1)
    if elapsed is None:
        try:
            print(open(log_path).read()[-2000:])
        except OSError:
            pass
        msg = f"{label}の起動に失敗しました（{log_path}）"
        if fail:
            fail(msg, log_path)
        raise RuntimeError(msg)
    print(f"✅ {label}起動完了（{elapsed}s）")
    return proc


def kc_assert_no_auth(port, path="/api/tags", fail=None, timeout=10):
    """401/403 なら認証が有効 = 設定ミス。上流未起動の 5xx は正常。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            print(f"  ✅ 認証なし → HTTP {r.status}")
            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            msg = f"認証なしで HTTP {e.code} — 認証が有効になっています"
            if fail:
                fail(msg)
            raise RuntimeError(msg)
        print(f"  ✅ 認証なし → HTTP {e.code}（401 でない）")
        return True
    except Exception as e:
        msg = f"プロキシ検査に失敗: {type(e).__name__}: {e}"
        if fail:
            fail(msg)
        raise

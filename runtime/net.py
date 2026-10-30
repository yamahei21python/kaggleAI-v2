"""トンネル URL 取得と HTTP 疎通の共通処理（生成コードへ埋め込む）。"""
import json
import re
import time
import urllib.error
import urllib.request

# hosts は表駆動。URL の出所（ログ解析 / 管理API）と待ち時間だけが違う。
TUNNEL_HOSTS = {
    "cloudflared": {
        "kind": "log",
        "url_re": re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com"),
        "wait_seconds": 90,
    },
    "ngrok": {
        "kind": "api",
        "api_url": "http://127.0.0.1:4040/api/tunnels",
        "wait_seconds": 60,
    },
}


def kc_probe_url(host, log_path=None):
    """現在の公開URLを返す（無ければ ""）。host は cloudflared / ngrok。"""
    spec = TUNNEL_HOSTS[host]
    if spec["kind"] == "api":
        try:
            with urllib.request.urlopen(spec["api_url"], timeout=3) as r:
                d = json.loads(r.read().decode())
            for t in d.get("tunnels", []):
                u = t.get("public_url", "")
                if u.startswith("https://"):
                    return u
        except Exception:
            return ""
        return ""
    if not log_path:
        return ""
    try:
        with open(log_path) as f:
            m = spec["url_re"].search(f.read())
        return m.group(0) if m else ""
    except OSError:
        return ""


def kc_wait_url(host, log_path=None, seconds=None, interval=1, progress_every=10):
    """公開URLが出るまで待つ。seconds 未指定なら host ごとの既定値。"""
    if seconds is None:
        seconds = TUNNEL_HOSTS[host]["wait_seconds"]
    for i in range(seconds):
        url = kc_probe_url(host, log_path)
        if url:
            return url
        if progress_every and i % progress_every == progress_every - 1:
            print(f"⏳ URL 待ち ({i + 1}/{seconds})...")
        time.sleep(interval)
    return ""


def kc_wait_http(url, attempts=20, interval=3, timeout=10, opener=None):
    """200 が返るまで待つ。成功したら経過秒（(i+1)*interval）、失敗なら None。"""
    opener = opener or urllib.request.urlopen
    for i in range(attempts):
        try:
            with opener(url, timeout=timeout) as r:
                if r.status == 200:
                    return (i + 1) * interval
        except Exception:
            pass
        time.sleep(interval)
    return None


def kc_retry(fn, tries=6, delay=5, skip_http_errors=True):
    """一過性の通信エラーを吸収する。HTTP 応答は「答え」なので既定で再試行しない。"""
    last = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if skip_http_errors:
                raise
            last = e
        except Exception as e:
            last = e
        if i + 1 < tries:
            wait = delay * (i + 1)
            print(f"  ⚠ 一時的な通信エラー（{i + 1}/{tries}）: {last} "
                  f"→ {wait}s 待って再試行")
            time.sleep(wait)
    raise last


def kc_post_json(url, payload, timeout=60, headers=None, opener=None):
    """JSON POST のレスポンスを返す（ストリームは呼び出し側で読む）。"""
    h = {"Content-Type": "application/json",
         "ngrok-skip-browser-warning": "true"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=h, method="POST")
    return (opener or urllib.request.urlopen)(req, timeout=timeout)

#!/usr/bin/env python3
"""Kaggle 上の Qwen3.8-27B と対話するクライアント。

公開URLは cloudflared quick tunnel で毎回変わるため、固定の Gist
（url.txt）から現在の URL を取得する。

ノートの既定は **認証なし**。環境変数 API_PASS（または ~/.orcarouter_pass）が
あるときだけ Basic 認証を付ける。

使い方:
    python3 ask.py "こんにちは"          # 1回だけ
    python3 ask.py                      # 対話モード（履歴保持）
    python3 ask.py --url https://... "..."  # Gist を介さず直接
    python3 ask.py --model orcarouter/Qwen3.8-27B-Uncensored:latest "..."
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

GIST_BASE = ("https://gist.githubusercontent.com/yamahei21python/"
             "2b44ac8cb115012a31d3c46661af933f/raw/")
GIST_RAW = GIST_BASE + "url.txt"
DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
PASS_FILE = pathlib.Path.home() / ".orcarouter_pass"
# ノートに埋め込まれた既定パスワード（build_step2.py と一致）。
# 環境変数 API_PASS / ~/.orcarouter_pass があれば Basic 認証を付ける（任意）。


def resolve_url(explicit=None):
    """現在の公開URLを決める。Gist は CDN キャッシュ対策で毎回クエリを変える。"""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("LLM_URL", "").strip()
    if env:
        return env.rstrip("/")
    req = urllib.request.Request(
        GIST_RAW + "?t=" + str(int(time.time())),
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        url = r.read().decode().strip()
    if not url.startswith("https://"):
        raise SystemExit("Gist の内容が URL ではありません: " + repr(url))
    return url.rstrip("/")


def _gist_text(name):
    """Gist の 1 ファイルを取る（CDN キャッシュ回避のクエリ付き）。"""
    req = urllib.request.Request(
        GIST_BASE + name + "?t=" + str(int(time.time())),
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode().strip()


def resolve_auth():
    """認証情報を決める。無ければ (None, None) = 認証なし。

    優先順: 環境変数 → ~/.orcarouter_pass。
    Gist にはもう pass.txt を置かない（既定が認証なしのため）。
    """
    user = os.environ.get("API_USER", "").strip() or "ollama"
    pw = os.environ.get("API_PASS", "").strip()
    if not pw and PASS_FILE.is_file():
        pw = PASS_FILE.read_text().strip()
    if not pw:
        return None, None
    return user, pw


def post(url, user, pw, payload, timeout=600):
    import base64
    headers = {
        "Content-Type": "application/json",
        # ngrok 無料枠のブラウザ向け警告ページを踏まないためのヘッダ
        "ngrok-skip-browser-warning": "true",
    }
    if pw:
        token = base64.b64encode(("%s:%s" % (user, pw)).encode()).decode()
        headers["Authorization"] = "Basic " + token
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(url, user, pw, model, messages, temperature, max_tokens,
        allow_retry=True, explicit_url=None):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        data = post(url, user, pw, payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        # 530/502/504 = tunnel 失効（cloudflared 再起動で URL が変わった）。
        # Gist から取り直して1回だけ再試行する。
        if allow_retry and e.code in (502, 503, 504, 530) and not explicit_url:
            print("# tunnel 失効 (HTTP %s) → URL を取り直して再試行" % e.code,
                  file=sys.stderr)
            return ask(resolve_url(), user, pw, model, messages, temperature,
                       max_tokens, allow_retry=False, explicit_url=None)
        raise SystemExit("HTTP %s: %s" % (e.code, body))
    except urllib.error.URLError as e:
        if allow_retry and not explicit_url:
            print("# 接続失敗 → URL を取り直して再試行", file=sys.stderr)
            return ask(resolve_url(), user, pw, model, messages, temperature,
                       max_tokens, allow_retry=False, explicit_url=None)
        raise SystemExit("接続失敗（tunnel 切れ / カーネル停止の可能性）: %s" % e)

    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    print(msg.get("content", "").strip())
    print("\n--- %s tok (prompt %s / completion %s) ---" % (
        usage.get("total_tokens"), usage.get("prompt_tokens"),
        usage.get("completion_tokens")), file=sys.stderr)
    return msg, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="空なら対話モード")
    ap.add_argument("--url", help="公開URL（Gist を介さない）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--system", default="日本語で簡潔に答える。")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    url = resolve_url(a.url)
    user, pw = resolve_auth()
    print("# " + url, file=sys.stderr)

    messages = [{"role": "system", "content": a.system}]

    if a.prompt:
        messages.append({"role": "user", "content": " ".join(a.prompt)})
        ask(url, user, pw, a.model, messages, a.temperature, a.max_tokens,
            explicit_url=a.url)
        return

    print("# 対話モード（/reset 履歴クリア, /quit 終了）", file=sys.stderr)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/reset":
            messages = [{"role": "system", "content": a.system}]
            print("# 履歴をクリア", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": line})
        msg, url = ask(url, user, pw, a.model, messages, a.temperature,
                       a.max_tokens, explicit_url=a.url)
        messages.append({"role": "assistant", "content": msg.get("content", "")})


if __name__ == "__main__":
    main()

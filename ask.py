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
import base64
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import kcconfig

_CFG = kcconfig.load()
GIST_RAW = ("https://gist.githubusercontent.com/%s/%s/raw/url.txt"
            % (_CFG.get("GIST_OWNER", ""), _CFG.get("GIST_ID", "")))
DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
PASS_FILE = pathlib.Path.home() / ".orcarouter_pass"
# Gist は書き換えられうる前提で扱う。cloudflared の quick tunnel 以外は信用しない
# （API_PASS を設定していると、任意ホストに Basic 認証を送ってしまうため）。
GIST_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com/?")


class AskError(Exception):
    """1 回の問い合わせの失敗（対話モードでは表示して続行する）。"""


def fixed_url(explicit=None):
    """--url / LLM_URL で固定された URL。無ければ None（Gist を使う）。"""
    url = explicit or os.environ.get("LLM_URL", "").strip()
    return url.rstrip("/") if url else None


def resolve_url(explicit=None):
    """現在の公開URLを決める。Gist は CDN キャッシュ対策で毎回クエリを変える。"""
    fixed = fixed_url(explicit)
    if fixed:
        return fixed
    req = urllib.request.Request(
        GIST_RAW + "?t=" + str(int(time.time())),
        headers={"Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            url = r.read().decode().strip()
    except (urllib.error.URLError, TimeoutError) as e:
        raise AskError("Gist から URL を取得できません: %s" % e) from e
    if not GIST_URL_RE.fullmatch(url):
        raise AskError("Gist の内容が想定外の URL です: " + repr(url))
    return url.rstrip("/")


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


def reply_text(msg):
    """表示・履歴用の本文。thinking 系モデルは content が null で
    reasoning_content だけを返すことがあるので、その場合はそちらを使う。"""
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def ask(url, user, pw, model, messages, temperature, max_tokens,
        allow_retry=True, fixed=False):
    """1 回問い合わせて (message, 使った URL) を返す。失敗は AskError。"""
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
        if allow_retry and e.code in (502, 503, 504, 530) and not fixed:
            print("# tunnel 失効 (HTTP %s) → URL を取り直して再試行" % e.code,
                  file=sys.stderr)
            return ask(resolve_url(), user, pw, model, messages, temperature,
                       max_tokens, allow_retry=False)
        raise AskError("HTTP %s: %s" % (e.code, body)) from e
    except (urllib.error.URLError, TimeoutError) as e:
        if allow_retry and not fixed:
            print("# 接続失敗 → URL を取り直して再試行", file=sys.stderr)
            return ask(resolve_url(), user, pw, model, messages, temperature,
                       max_tokens, allow_retry=False)
        raise AskError("接続失敗（tunnel 切れ / カーネル停止の可能性）: %s" % e) from e

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise AskError("想定外の応答: %s" % str(data)[:300]) from e
    usage = data.get("usage") or {}
    print(reply_text(msg))
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

    fixed = fixed_url(a.url) is not None
    try:
        url = resolve_url(a.url)
    except AskError as e:
        raise SystemExit(str(e)) from e
    user, pw = resolve_auth()
    print("# " + url, file=sys.stderr)

    messages = [{"role": "system", "content": a.system}]

    if a.prompt:
        messages.append({"role": "user", "content": " ".join(a.prompt)})
        try:
            ask(url, user, pw, a.model, messages, a.temperature, a.max_tokens,
                fixed=fixed)
        except AskError as e:
            raise SystemExit(str(e)) from e
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
        try:
            msg, url = ask(url, user, pw, a.model, messages, a.temperature,
                           a.max_tokens, fixed=fixed)
        except AskError as e:
            # 失敗した発言は履歴から外し、会話は続ける
            messages.pop()
            print("# " + str(e), file=sys.stderr)
            continue
        messages.append({"role": "assistant", "content": reply_text(msg)})


if __name__ == "__main__":
    main()

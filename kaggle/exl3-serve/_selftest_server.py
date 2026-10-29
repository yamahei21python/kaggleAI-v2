#!/usr/bin/env python3
"""chat_server.py の自己テスト（exllamav3 不要・CPU のみ）。

MockEngine に対して HTTP の形を確認する:
  /health, /v1/models, /v1/chat/completions（非ストリーム / SSE ストリーム）,
  400（不正リクエスト）, 404（未知パス）。
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_server import MockEngine, make_server  # noqa: E402


def http(port, path, body=None, timeout=10):
    url = f"http://127.0.0.1:{port}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    srv = make_server(MockEngine(), host="127.0.0.1", port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    try:
        # health / models
        code, body = http(port, "/health")
        check(code == 200, f"/health status {code}")
        check(json.loads(body)["status"] == "ok", "/health body")

        code, body = http(port, "/v1/models")
        check(code == 200, f"/v1/models status {code}")
        check(json.loads(body)["data"][0]["id"] == MockEngine.alias,
              "/v1/models id")

        # 非ストリーム
        code, body = http(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "こんにちは"}],
            "max_tokens": 32,
        })
        check(code == 200, f"completion status {code}: {body[:200]}")
        obj = json.loads(body)
        content = obj["choices"][0]["message"]["content"]
        check(content == "echo: こんにちは", f"content {content!r}")
        check(obj["usage"]["total_tokens"] > 0, "usage")

        # ストリーム（SSE）
        code, body = http(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "テスト"}],
            "max_tokens": 32, "stream": True,
        })
        check(code == 200, f"stream status {code}")
        lines = [ln for ln in body.splitlines() if ln.startswith("data: ")]
        check(lines[-1] == "data: [DONE]", f"stream end {lines[-1]!r}")
        text = ""
        for ln in lines[:-1]:
            payload = json.loads(ln[len("data: "):])
            text += payload["choices"][0].get("delta", {}).get("content", "")
        check(text == "echo: テスト", f"stream text {text!r}")

        # 400 / 404
        code, _ = http(port, "/v1/chat/completions", {"messages": []})
        check(code == 400, f"empty messages -> {code}")
        code, _ = http(port, "/nope")
        check(code == 404, f"unknown path -> {code}")
    finally:
        srv.shutdown()
        srv.server_close()

    print("OK: chat_server selftest")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

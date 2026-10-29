#!/usr/bin/env python3
"""CPU 空き枠用の追加テスト（GPU 不要）。

  1. step2-vllm-chattpl    … Dataset の chat_template を transformers で実際に描画（install 不要・1 分）
  2. step2-vllm-proxymatrix … --enforce-sampling のサンプリング矯正を全パターン検証（モック上流）

    python3 kaggle/build_cpu_probe3.py
    cd kaggle/step2-vllm-chattpl && kaggle kernels push -p .
    cd kaggle/step2-vllm-proxymatrix && kaggle kernels push -p .
"""
import base64
import json
import pathlib

OWNER = "yamahei21python"
HERE = pathlib.Path(__file__).resolve().parent
DATASET_ID = "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp"

PROXY_B64 = base64.b64encode(
    (HERE.parent / "step2" / "auth_proxy.py").read_bytes()).decode()


def _nb(md, code):
    cells = []
    for i, (t, src) in enumerate((("markdown", md), ("code", code))):
        d = {"cell_type": t, "id": f"cell{i:02d}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if t == "code":
            d["execution_count"] = None
            d["outputs"] = []
        cells.append(d)
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.13"}},
            "nbformat": 4, "nbformat_minor": 5}


def _write(slug, title, nb, datasets):
    d = HERE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": title,
            "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "keywords": [],
            "dataset_sources": datasets, "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {d}")


CHATTPL = '''# chat_template を transformers で実際に描画する（vLLM 不要・1 分）
# 目的: GPU に上げる前に「テンプレートが壊れていない / 思考タグが正しい」を確認。
import json
import os

import transformers

ROOT = ("/kaggle/input/datasets/yamahei21python/"
        "qwen38-27b-uncensored-hauhaucs-awq-mtp")
print("transformers:", transformers.__version__)
print("files:", sorted(os.listdir(ROOT))[:20])

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=True)
print("tokenizer:", type(tok).__name__, "vocab:", tok.vocab_size)
print("chat_template 長:", len(tok.chat_template or ""))
print("chat_template 先頭:", (tok.chat_template or "")[:300].replace("\\n", " | "))

MSGS = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "1+1 は？"},
    {"role": "assistant", "content": "2 です。"},
    {"role": "user", "content": "では 2+2 は？"},
]

def ids_of(**kw):
    """tokenize=True は BatchEncoding を返す（len() は 2 になる）。input_ids を見る。"""
    out = tok.apply_chat_template(MSGS, tokenize=True, **kw)
    if isinstance(out, dict):
        return out["input_ids"]
    if hasattr(out, "input_ids"):
        return out.input_ids
    return out


for kw in ({}, {"add_generation_prompt": True}):
    try:
        text = tok.apply_chat_template(MSGS, tokenize=False, **kw)
        ids = ids_of(**kw)
        print(f"\\n--- kw={kw} ---")
        print("chars:", len(text), "tokens:", len(ids))
        print(repr(text[-400:]))
        for tag in ("<|im_start|>", "<|im_end|>", "think", "tool_call"):
            print(f"  contains {tag!r}:", tag in text)
    except Exception as e:
        print(f"apply_chat_template NG kw={kw}: {type(e).__name__}: {e}")

# 特殊トークン ID の確認（reasoning parser / stop 条件に関係）
for name in ("eos_token", "pad_token", "bos_token"):
    t = getattr(tok, name, None)
    print(name, "=", t, getattr(t, "replace", lambda *a: None) and (tok.convert_tokens_to_ids(t) if t else None))
print("added_tokens:", len(tok.get_added_vocab()))

# ツール宣言がテンプレートで通るか（opencode 連携で効く）
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "天気を返す",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
try:
    t2 = tok.apply_chat_template(MSGS, tools=TOOLS, tokenize=False,
                                 add_generation_prompt=True)
    print("\\n--- tools ---")
    print("chars:", len(t2), "contains tool_call:", "tool_call" in t2)
    print(repr(t2[:600]))
except Exception as e:
    print("tools NG:", type(e).__name__, e)

# reasoning_effort がテンプレート変数か（system に xhigh が注入されていた）
tpl = tok.chat_template or ""
for kw in ("reasoning_effort", "xhigh", "effort"):
    print(f"template contains {kw!r}:", kw in tpl)
try:
    t3 = tok.apply_chat_template(MSGS, tokenize=False,
                                 add_generation_prompt=True,
                                 reasoning_effort="low")
    print("reasoning_effort=low chars:", len(t3))
    print(repr(t3[:300]))
except Exception as e:
    print("reasoning_effort NG:", type(e).__name__, e)

# generation_config の推奨サンプリング（RP / temp）
gen = json.load(open(os.path.join(ROOT, "generation_config.json")))
print("generation_config:", gen)
'''


PROXYMATRIX = '''# --enforce-sampling の全パターン検証（モック上流・GPU 不要）
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

UP_PORT = 18000
PROXY_PORT = 18435

# 上流: 受け取った sampling パラメータをそのまま返す
SEEN = []


class Up(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append({k: body.get(k) for k in
                     ("temperature", "repetition_penalty", "top_p", "top_k")})
        out = json.dumps({"id": "x", "object": "chat.completion",
                          "choices": [{"index": 0, "message": {"role": "assistant",
                                     "content": "ok"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                    "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", UP_PORT), Up)
threading.Thread(target=srv.serve_forever, daemon=True).start()

os.makedirs("/tmp/px", exist_ok=True)
open("/tmp/px/auth_proxy.py", "wb").write(base64.b64decode("__PROXY_B64__"))

CASES = [
    {},
    {"temperature": 0},
    {"temperature": 0.0, "repetition_penalty": 1.0},
    {"temperature": 0.2, "repetition_penalty": 1.01},
    {"temperature": 1.5, "repetition_penalty": 1.05},
    {"temperature": 0.7, "repetition_penalty": 1.7},
    {"top_p": 0.9, "top_k": 20},
]


def run_proxy(extra):
    p = subprocess.Popen([sys.executable, "/tmp/px/auth_proxy.py",
                          "--upstream", f"http://127.0.0.1:{UP_PORT}",
                          "--port", str(PROXY_PORT)] + extra,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        time.sleep(0.2)
        try:
            socket.create_connection(("127.0.0.1", PROXY_PORT), 0.5).close()
            return p
        except OSError:
            continue
    return p


def post(payload):
    data = json.dumps({"model": "m", "messages": [{"role": "user",
                       "content": "hi"}], **payload}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions",
                                 data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        r.read()


for extra in ([], ["--enforce-sampling"]):
    SEEN.clear()
    p = run_proxy(extra)
    label = "enforce" if extra else "passthrough"
    print(f"\\n===== {label} =====")
    for c in CASES:
        try:
            post(c)
        except Exception as e:
            print("REQ NG", c, type(e).__name__, e)
            continue
        print(f"  in={c} -> up={SEEN[-1]}")
    p.terminate()
    p.wait(timeout=10)
    time.sleep(0.5)

srv.shutdown()
print("\\nDONE")
'''


def main():
    _write("step2-vllm-chattpl", "Step2 vLLM chattpl",
           _nb("# 🧾 chat_template の実描画チェック（install 不要）\n\n"
               "Dataset の tokenizer でテンプレートを描画し、思考タグ・特殊トークンを確認。\n",
               CHATTPL), [DATASET_ID])
    _write("step2-vllm-proxymatrix", "Step2 vLLM proxymatrix",
           _nb("# 🎛 --enforce-sampling の全パターン検証（モック上流）\n\n"
               "temp=0 / rp<1.05 / 欠落 / 正常値 を通して、上流に何が渡るかを見る。\n",
               PROXYMATRIX.replace("__PROXY_B64__", PROXY_B64)), [])


if __name__ == "__main__":
    main()

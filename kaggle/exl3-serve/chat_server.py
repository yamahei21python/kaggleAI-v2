#!/usr/bin/env python3
"""ExLlamaV3 の薄い OpenAI 互換サーバ。

exllamav3 の Generator を 1 本のロックで直列化し、llama-server と同じ形の
`/v1/chat/completions`（stream 可）・`/v1/models`・`/health` を出す。
手前の auth_proxy.py / llama-chat.html / ask.py はそのまま使える。

exllamav3 と torch は Exl3Engine の生成時にだけ import するので、
モック Engine を渡せば CPU でもサーバ部分を自己テストできる
（`_selftest_server.py`）。
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_ALIAS = "qwen38-27b-exl3"
DEFAULT_CACHE_TOKENS = 8192


def _now():
    return int(time.time())


def _completion_id():
    return "chatcmpl-" + uuid.uuid4().hex[:24]


class Exl3Engine:
    """exllamav3 の Generator を 1 リクエストずつ回すエンジン。"""

    def __init__(self, model_dir, cache_tokens=DEFAULT_CACHE_TOKENS,
                 alias=DEFAULT_ALIAS, enable_thinking=False,
                 progressbar=True):
        import torch
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.alias = alias
        self.enable_thinking = enable_thinking
        self.lock = threading.Lock()
        self._prompt_len = 0

        cfg = Config.from_directory(model_dir)
        self.tokenizer = Tokenizer.from_config(cfg)
        model = Model.from_config(cfg)
        cache = Cache(model, max_num_tokens=cache_tokens, max_batch_size=1)
        model.load(progressbar=progressbar)  # 空き VRAM でレイヤー分割
        # 初回リクエストがカーネルコンパイル/オートチューニングを払わないよう、
        # Generator を付ける前に温める（失敗しても配信は続ける）
        try:
            model.warmup(cache=cache, progressbar=False)
        except Exception as e:
            print(f"warmup failed: {type(e).__name__}: {e}", flush=True)
        self.generator = Generator(model=model, cache=cache,
                                   tokenizer=self.tokenizer)
        self.context_length = cache_tokens

    # ---- 入出力の変換 ----
    def _input_ids(self, messages):
        try:
            return self.tokenizer.hf_chat_template(
                messages, enable_thinking=self.enable_thinking)
        except Exception:
            return self.tokenizer.hf_chat_template(messages)

    def _stop_conditions(self, extra=None):
        stops = ["<|im_end|>"]
        if isinstance(extra, str):
            extra = [extra]
        stops.extend(s for s in (extra or []) if isinstance(s, str) and s)
        try:
            sid = self.tokenizer.single_id("<|im_end|>")
            if isinstance(sid, int) and sid >= 0:
                stops.append(sid)
        except Exception:
            pass
        return stops

    @staticmethod
    def _sampler(temperature, top_p):
        from exllamav3 import GreedySampler, TopPSampler
        if not temperature or temperature <= 0:
            return GreedySampler()
        return TopPSampler(temperature=temperature, top_p=top_p,
                           temperature_last=True)

    def _job(self, messages, max_tokens, temperature, top_p, stop, seed):
        from exllamav3 import Job
        ids = self._input_ids(messages)
        # exllamav3 の完了結果に prompt_tokens は無いので入力長を控えておく
        self._prompt_len = int(ids.shape[-1])
        return Job(
            input_ids=ids,
            max_new_tokens=max(1, int(max_tokens)),
            stop_conditions=self._stop_conditions(stop),
            sampler=self._sampler(temperature, top_p),
            seed=seed,
        )

    def _usage(self, final):
        if not final:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {"prompt_tokens": int(final.get("prompt_tokens")
                                     or self._prompt_len or 0),
                "completion_tokens": int(final.get("new_tokens") or 0)}

    # ---- 非ストリーム ----
    def generate(self, messages, max_tokens=512, temperature=0.7,
                 top_p=1.0, stop=None, seed=None):
        with self.lock:
            job = self._job(messages, max_tokens, temperature, top_p, stop, seed)
            self.generator.enqueue(job)
            text = ""
            final = None
            while self.generator.num_remaining_jobs():
                for r in self.generator.iterate():
                    if r.get("stage") == "error":
                        raise r["error"]
                    text += r.get("text", "")
                    if r.get("eos"):
                        final = r
        return text, self._usage(final)

    # ---- ストリーム ----
    def stream(self, messages, max_tokens=512, temperature=0.7,
               top_p=1.0, stop=None, seed=None):
        with self.lock:
            job = self._job(messages, max_tokens, temperature, top_p, stop, seed)
            self.generator.enqueue(job)
            try:
                final = None
                while self.generator.num_remaining_jobs():
                    for r in self.generator.iterate():
                        if r.get("stage") == "error":
                            raise r["error"]
                        chunk = r.get("text", "")
                        if chunk:
                            yield chunk
                        if r.get("eos"):
                            final = r
                self._final = final
            except GeneratorExit:
                # クライアント切断時は残りのジョブを捨てる
                self.generator.clear_queue()
                raise


class MockEngine:
    """自己テスト用。exllamav3 無しでサーバを通す。"""

    alias = "mock-exl3"
    context_length = 512

    def generate(self, messages, max_tokens=512, temperature=0.7,
                 top_p=1.0, stop=None, seed=None):
        content = messages[-1]["content"] if messages else ""
        text = "echo: " + content
        return text, {"prompt_tokens": len(content),
                      "completion_tokens": len(text)}

    def stream(self, messages, max_tokens=512, temperature=0.7,
               top_p=1.0, stop=None, seed=None):
        content = messages[-1]["content"] if messages else ""
        for piece in ("echo: ", content[:4], content[4:]):
            if piece:
                yield piece
        self._final = {"prompt_tokens": len(content),
                       "completion_tokens": len(content) + 6}


def _usage_of(engine):
    final = getattr(engine, "_final", None)
    return engine._usage(final) if final else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "exl3-server/1.0"

    def log_message(self, fmt, *args):
        pass

    # ---- helpers ----
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg, err_type="invalid_request_error"):
        self._send_json({"error": {"message": msg, "type": err_type}}, code)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return None
        try:
            return json.loads(self.rfile.read(n).decode())
        except ValueError:
            return None

    # ---- GET ----
    def do_GET(self):
        engine = self.server.engine
        if self.path.split("?")[0] == "/health":
            self._send_json({"status": "ok", "model": engine.alias})
        elif self.path.split("?")[0] == "/v1/models":
            self._send_json({"object": "list", "data": [{
                "id": engine.alias, "object": "model", "created": _now(),
                "owned_by": "local"}]})
        elif self.path.split("?")[0] == "/props":
            self._send_json({"model_path": engine.alias,
                             "n_ctx": engine.context_length})
        else:
            self._error(404, "not found")

    # ---- POST ----
    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._error(404, "not found")
            return
        body = self._read_json()
        if not body or not isinstance(body.get("messages"), list) \
                or not body["messages"]:
            self._error(400, "messages is required")
            return
        messages = body["messages"]
        if not all(isinstance(m, dict) and m.get("role") in
                   ("system", "user", "assistant") for m in messages):
            self._error(400, "each message needs role/content")
            return

        engine = self.server.engine
        kwargs = dict(
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 512,
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 1.0),
            stop=body.get("stop"),
            seed=body.get("seed"),
        )
        cid = _completion_id()

        if body.get("stream"):
            self._stream(engine, cid, messages, kwargs)
            return

        try:
            text, usage = engine.generate(messages, **kwargs)
        except Exception as e:  # 生成器内の失敗は 500 で返す
            self._error(500, f"{type(e).__name__}: {e}", "server_error")
            return
        usage = usage or {}
        usage["total_tokens"] = (usage.get("prompt_tokens", 0)
                                 + usage.get("completion_tokens", 0))
        self._send_json({
            "id": cid, "object": "chat.completion", "created": _now(),
            "model": engine.alias,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": usage,
        })

    def _stream(self, engine, cid, messages, kwargs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": _now(), "model": engine.alias,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        try:
            self.wfile.write(
                ("data: " + json.dumps(chunk({"role": "assistant",
                                              "content": ""}),
                                       ensure_ascii=False) + "\n\n").encode())
            for piece in engine.stream(messages, **kwargs):
                self.wfile.write(
                    ("data: " + json.dumps(chunk({"content": piece}),
                                           ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(
                ("data: " + json.dumps(chunk({}, "stop"),
                                       ensure_ascii=False) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                err = {"error": {"message": f"{type(e).__name__}: {e}",
                                 "type": "server_error"}}
                self.wfile.write(
                    ("data: " + json.dumps(err) + "\n\n").encode())
            except Exception:
                pass


def make_server(engine, host="0.0.0.0", port=9931):
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.engine = engine
    return srv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--mock", action="store_true",
                    help="exllamav3 を使わず MockEngine で起動（配管テスト用）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9931)
    ap.add_argument("--alias", default=DEFAULT_ALIAS)
    ap.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    if a.mock:
        engine = MockEngine()
        engine.alias = a.alias
    else:
        if not a.model_dir:
            ap.error("--model-dir is required unless --mock")
        engine = Exl3Engine(a.model_dir, cache_tokens=a.cache_tokens,
                            alias=a.alias, enable_thinking=a.enable_thinking,
                            progressbar=not a.no_progress)
    print(f"loaded in {time.time() - t0:.0f}s; serving {a.alias} on "
          f"{a.host}:{a.port} (ctx {a.cache_tokens})", flush=True)
    make_server(engine, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()

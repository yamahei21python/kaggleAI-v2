#!/usr/bin/env python3
"""Basic 認証つきリバースプロキシ（標準ライブラリのみ）。

cloudflared の quick tunnel には Basic 認証機能が無いため、
Ollama の手前にこれを置いて認証を担わせる。

    API_USER=ollama API_PASS=secret python3 auth_proxy.py \
        --port 11435 --upstream http://127.0.0.1:11434

ストリーミング（Ollama の token 逐次出力）を通すため、
上流の応答は chunked transfer-encoding でそのまま中継する。
"""

import argparse
import base64
import hmac
import http.server
import json
import os
import signal
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# 上流（Ollama）への待ち時間。長い生成に耐えるよう長めに取る。
UPSTREAM_TIMEOUT = 3600
# 中継のバッファ上限。改行（NDJSON 行）が来ない場合のフォールバックで、
# ここに達するまで待つとストリームが溜まるので、小さいほど逐次性は高い。
CHUNK = 4096
# Basic 認証を免除するパス。
# /healthz は疎通確認、/ と /chat はチャット UI（静的な HTML のみ。
# 認証情報もモデルも含まないので、公開しても漏れるものは無い）。
# API 本体（/api/*, /v1/*）は従来どおり Basic 認証が要る。
OPEN_PATHS = {"/healthz", "/", "/chat", "/index.html", "/favicon.ico"}
UI_PATHS = {"/", "/chat", "/index.html"}

# 管理エンドポイント（上流へ流さない）。
# /quota : Kaggle の GPU クォータ残量（UI のヘッダに出す）
# /stop  : この中継と上流のセッションを止める。
#          POST + ヘッダ X-Kc-Stop: 1 が必須（GET のスキャナや他サイトからの
#          CSRF で落とされないように）。API_PASS 設定時は Basic 認証も要る。
ADMIN_PATHS = {"/quota", "/stop"}
STOP_HEADER = "X-Kc-Stop"

# keepalive を延長するのは「生成」リクエストだけ。/api/ps や /v1/models の
# ポーリング（ランチャーの起動確認・スキャナ）で GPU セッションを延命しない。
GENERATION_PATHS = {
    "/api/chat",
    "/api/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}

# Kaggle のクォータ API。kaggle CLI には quota サブコマンドが無いが、
# Web API は Basic 認証で読める（実測 2026-10-28）。
KAGGLE_QUOTA_URL = "https://www.kaggle.com/api/v1/kernels/quota"
KAGGLE_CRED = os.path.expanduser("~/.kaggle/kaggle.json")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "expect",
}

ARGS = None
EXPECTED_AUTH = ""
# チャット UI の HTML。起動時に一度だけ読む（毎リクエストでディスクを叩かない）。
CHAT_BYTES = b""

# ---- サンプリング強制（--enforce-sampling のときだけ有効）----
# vLLM 版で必要。クライアント（opencode 等）が temperature=0 を送ると
# 貪欲法になり、27B 級では同じ文の繰り返しで崩壊しやすい。
# repetition_penalty も未指定だと 1.0（=無効）になるため下限を入れる。
# 既定のプロキシ（Ollama 版）では無効なので挙動は変わらない。
RP_FLOOR = 1.05     # repetition_penalty の下限
TEMP_FLOOR = 0.7    # temperature の下限（0 = 貪欲法を禁止）


# トンネル/チャット経由のアクティビティで keepalive 時間をリセット。
# /tmp/keepalive.conf の RUN=1, MINUTES=60 を再書き込み。
_KEEPALIVE_CONF = "/tmp/keepalive.conf"


def _reset_keepalive():
    """チャット送信ごとに時間リセット（ON 維持）。上流エラー時でも呼ぶ。"""
    try:
        with open(_KEEPALIVE_CONF, "w") as f:
            f.write("RUN=1\nMINUTES=60\n")
    except Exception:
        pass


def _is_generation(path_only):
    return path_only.rstrip("/") in GENERATION_PATHS


def _num(value):
    """数値化できなければ None（bool は数値扱いしない）。"""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enforce_sampling(body, path_only):
    """chat/completions のサンプリング下限を強制する（--enforce-sampling 時）。

    - repetition_penalty < RP_FLOOR（未指定含む）→ RP_FLOOR
    - temperature <= 0（未指定含む）→ TEMP_FLOOR

    ボディが JSON でない/壊れている場合は素通し（上流に判断を委ねる）。
    戻り値は差し替え後の bytes。Content-Length は呼び出し側で付け直す。
    """
    if not ARGS.enforce_sampling or not body:
        return body
    # クエリ文字列（?api-version=... 等）が付くと endswith が外れるため、
    # パスだけを取り出して比較する。取りこぼすと矯正が無効化される。
    _path = urllib.parse.urlsplit(path_only).path
    if not _path.rstrip("/").endswith("/chat/completions"):
        return body
    try:
        data = json.loads(body)
    except Exception:
        return body
    if not isinstance(data, dict):
        return body
    changed = False
    rp = _num(data.get("repetition_penalty"))
    if rp is None or rp < RP_FLOOR:
        data["repetition_penalty"] = RP_FLOOR
        changed = True
    temp = _num(data.get("temperature"))
    if temp is None or temp <= 0:
        data["temperature"] = TEMP_FLOOR
        changed = True
    if not changed:
        return body
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OllamaAuthProxy/1.0"
    # 接続ごとのソケットタイムアウト。クライアントが無通信のまま残すと
    # ハンドラスレッドが無限に居座るので、5分で回収する（ストリーミング中
    # の上流読み取りは urllib 側ソケットなので影響を受けない）。
    timeout = 300

    # ---- ログを 1 行に圧縮（ノートブックのログを汚さない）----
    def log_message(self, fmt, *args):
        sys.stderr.write("  [proxy] %s\n" % (fmt % args))

    def _read_body(self):
        """リクエストボディを読む。chunked でも Content-Length でも両対応。

        Content-Length しか見ないと、HTTP/1.1 の chunked POST（ストリーミング入力や
        プロキシ経由の変換で発生しうる）で**ボディごと欠落**して上流に届かない。
        """
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = bytearray()
            while True:
                size_line = self.rfile.readline().strip()
                if b";" in size_line:          # chunk-ext は捨てる
                    size_line = size_line.split(b";", 1)[0]
                try:
                    size = int(size_line, 16)
                except ValueError:
                    break
                if size == 0:                  # トレーラー読み捨て
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                out += self.rfile.read(size)
                self.rfile.read(2)             # チャンク末尾の CRLF
            return bytes(out) or None

        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def _emit(self, data):
        """chunked の1分片を送る。

        改行単位で呼ぶので send 回数はトークン数に同比例（1バイト毎には送らない）。
        """
        if not data:
            return
        self.wfile.write(b"%x\r\n" % len(data))
        self.wfile.write(bytes(data))
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _send_json(self, code, obj, close=False, extra_headers=()):
        """JSON を 1 発で返す（管理エンドポイント・エラー応答の共通処理）。"""
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        if close:
            self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra_headers:
            self.send_header(key, value)
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def _authorized(self):
        # API_PASS が空 = 認証なし（公開）。UI を開いただけで使えるようにする。
        if not EXPECTED_AUTH:
            return True
        if self.path.split("?")[0] in OPEN_PATHS:
            return True
        # 常数時間比較（タイミング攻撃の緩和）。str 版 compare_digest は
        # 非 ASCII で TypeError になるので bytes で比較する。
        # OPTIONS は _relay 冒頭でローカル即答済み（ここには届かない）。
        got = self.headers.get("Authorization", "").encode("latin-1")
        return hmac.compare_digest(got, EXPECTED_AUTH.encode("ascii"))

    def _deny(self):
        # 401 で接続を閉じる。Keep-Alive のまま残すと、次のリクエストの
        # ボディ読み取り位置がずれて HTTP リクエストスモガリングが起きうる。
        self._send_json(401, {"error": "unauthorized"}, close=True,
                        extra_headers=[("WWW-Authenticate",
                                        'Basic realm="Ollama API"')])

    def _serve_ui(self):
        """チャット UI を返す。認証不要（静的ファイルのみ）。

        UI は 1 リクエストで終わるので **接続を閉じる**。認証パスと違い
        ここではボディを読まないため、keep-alive を維持すると
        Content-Length 付き POST で次のリクエスト位置がずれる
        （リクエストスマグリング）。
        """
        self.close_connection = True
        try:
            self._read_body()          # 想定外のボディは捨てる（読み残し防止）
        except Exception:
            pass
        if not CHAT_BYTES:
            body = (b"llama-chat.html not found. "
                    b"start the proxy with --chat-html <path>.")
            self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(CHAT_BYTES)))
        # UI は毎回同じ内容。プロキシ更新時にブラウザが古い版を使わないよう no-store。
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(CHAT_BYTES)

    def _serve_quota(self):
        """Kaggle の GPU クォータ残量を返す。

        UI のヘッダに「残り 08:59」と出すために使う。
        kaggle.json が無い、または API が落ちている場合は
        available=false を返して UI 側で非表示にする。
        """
        payload = {"available": False}
        if ARGS.no_quota:
            # Kaggle 以外のローカル（uzu など）ではクォータの概念が無い。
            # available=false を返し、UI 側は残り時間を非表示にする。
            self._send_json(200, {"available": False, "reason": "no_quota"})
            return

        try:
            if os.path.isfile(KAGGLE_CRED):
                with open(KAGGLE_CRED) as f:
                    cfg = json.load(f)
                token = base64.b64encode(
                    f"{cfg['username']}:{cfg['key']}".encode()
                ).decode()
                req = urllib.request.Request(
                    KAGGLE_QUOTA_URL,
                    headers={"Authorization": "Basic " + token},
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    d = json.loads(r.read())
                g = d["gpuQuota"]
                used = g["timeUsed"]["seconds"]
                total = g["totalTimeAllowed"]["seconds"]
                reserved = g.get("timeReserved", {}).get("seconds", 0)
                payload = {
                    "available": True,
                    "usedSeconds": used,
                    "totalSeconds": total,
                    "reservedSeconds": reserved,
                    "leftSeconds": total - used - reserved,
                    "refreshTime": d.get("quotaRefreshTime", ""),
                }
        except Exception as e:
            payload = {"available": False, "error": type(e).__name__}

        # 残量は刻々変わる。_send_json は no-store を付ける。
        self._send_json(200, payload)

    def _serve_stop(self):
        """この中継と上流のセッションを止める。

        呼び出し条件（POST + X-Kc-Stop + 認証）は _stop_allowed() で確認済み。

        ブラウザの停止ボタンから叩く。中継が落ちるので応答を返してから
        プロセスを終了させる（先に落とすとブラウザ側が通信エラーになる）。

        Kaggle 上では keepalive ループも止めるため、/tmp/keepalive.conf に
        RUN=0 を書き込む。これにより Kaggle カーネルセッションが終了し、
        GPU クォータを開放できる。

        --stop-pid が指定されていれば、その PID（ランチャー）にも
        SIGTERM を送る。uzu は run-uzu-server.command が子プロセスを
        抱えているので、親を落とせば trap cleanup で全部片付く。
        """
        # Kaggle 版 keepalive ループにも停止を伝える
        try:
            with open("/tmp/keepalive.conf", "w") as f:
                f.write("RUN=0\n")
        except Exception:
            pass

        self._send_json(200, {"stopped": True, "scope": "relay"})

        def _die():
            time.sleep(0.5)
            # 先にランチャーを落とす（子プロセスと UI も一緒に消える）
            if ARGS.stop_pid:
                try:
                    os.kill(ARGS.stop_pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=_die, daemon=True).start()

    def _relay(self):
        # CORS プリフライトは認証不要・ローカル即答。上流へ流さない
        # （素通しすると未認証リクエストが上流に届き、ボディ待ちで
        #  ハンドラスレッドも占められる）。
        if self.command == "OPTIONS":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # /healthz はローカルで即答する（上流の有無に関係なく生存確認できる）
        if self.path.split("?")[0] == "/healthz":
            self._send_json(200, {"status": "ok"})
            return

        path_only = self.path.split("?")[0]
        if path_only in UI_PATHS:
            self._serve_ui()
            return

        # ---- 管理エンドポイント（ローカル専用）----
        if path_only == "/quota":
            self._serve_quota()
            return
        if path_only == "/stop":
            # 落とせるのは UI の停止ボタン（POST + X-Kc-Stop）だけ。
            # 以前は認証より前に処理していたため、API_PASS を設定していても
            # 誰でも GET /stop で Kaggle セッションを止められた。
            try:
                self._read_body()
            except Exception:
                pass
            if self.command != "POST" or self.headers.get(STOP_HEADER) != "1":
                self._send_json(405, {"error": "POST with %s: 1 required"
                                      % STOP_HEADER}, close=True)
                return
            if not self._authorized():
                self._deny()
                return
            self._serve_stop()
            return
        if path_only == "/favicon.ico":
            # ブラウザが勝手に取りに来る。404 を返すとログが埋まる。
            self.close_connection = True
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not self._authorized():
            # 401 を返す前にボディを食っておく。POST のボディを読まずに
            # 接続を切るとクライアント側が RST を受けて「401 より前に
            # 接続が切れた」に見える（curl は RST をエラー表示）。
            try:
                self._read_body()
            except Exception:
                pass
            self._deny()
            return

        try:
            body = self._read_body()
        except Exception:
            # 不正な Content-Length / 壊れた chunked は上流に流さない。
            # 未捕捉だとスタックダンプの500になる（認証成功側だけ非対称にならないよう）
            self.close_connection = True
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return

        if _is_generation(path_only):
            _reset_keepalive()  # 生成リクエスト＝アクティビティあり → 60分リセット
        # サンプリング下限の強制（--enforce-sampling 時のみ。既定は素通し）
        body = enforce_sampling(body, path_only)
        url = ARGS.upstream.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, method=self.command)
        for key, value in self.headers.items():
            # content-length は body 再構築後の実長で送り直すためここで捨てる
            # （クライアント値のまま転送すると chunked + CL 併用でデシンクする）
            if key.lower() in HOP_BY_HOP | {"host", "authorization",
                                            "content-length"}:
                continue
            req.add_header(key, value)
        # 上流が ngrok のとき、ブラウザからのリクエストには
        # ngrok-skip-browser-warning が付かない（HTML 取得時）。
        # 中継側で必ず付けて、警告ページを回避する。
        if ARGS.upstream_header:
            for item in ARGS.upstream_header:
                if "=" in item:
                    k, v = item.split("=", 1)
                    req.add_header(k.strip(), v.strip())
        if body is not None:
            req.add_header("Content-Length", str(len(body)))

        try:
            resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
        except urllib.error.HTTPError as exc:
            # 上流が 4xx/5xx を返した場合も本文をそのまま返す
            payload = exc.read()
            self.send_response(exc.code)
            ctype = exc.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
            return
        except Exception:  # 接続失敗など
            # 内部例外の詳細はクライアントに渡さない（情報露出防止）。
            # 原因はサーバ側 stderr にだけ書く。
            sys.stderr.write("  [proxy] upstream error: %s\n"
                             % (sys.exc_info()[1],))
            self._send_json(502, {"error": "upstream unreachable"})
            return

        with resp:
            self.send_response(resp.status)
            for key, value in resp.headers.items():
                if key.lower() in HOP_BY_HOP or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            # HEAD は本文なし。chunked を宣言するとクライアントが終端チャンクを
            # 待ってハングする（Content-Length も渡せないので宣言しない）。
            # 204/304 は本文を持てない。chunked を宣言すると厳格な
            # クライアントが protocol error にする。
            no_body = resp.status in (204, 304) or self.command == "HEAD"
            if not no_body:
                # 長さ未知のストリームとして中継する
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if no_body:
                return
            try:
                # resp.read(N) は HTTPResponse._readinto_chunked の性質で
                # **N バイト揃うまで読み続ける**ため、ストリームが全面に溜まる
                # （Ollama の NDJSON が全トークン分貯まってから一斉に届く）。
                # readline(CHUNK) は改行か CHUNK バイトで返るので、逐次性を
                # 保ったまま 1 バイトずつの read ループより CPU を食わない。
                while True:
                    line = resp.readline(CHUNK)
                    if not line:
                        break
                    self._emit(line)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # クライアントが切った（Ctrl-C など）。正常系。
                pass

    do_GET = _relay
    do_POST = _relay
    do_PUT = _relay
    do_PATCH = _relay
    do_DELETE = _relay
    do_HEAD = _relay
    do_OPTIONS = _relay


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global ARGS, EXPECTED_AUTH

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--chat-html", default=os.environ.get("CHAT_HTML", ""),
                        help="チャット UI の HTML（/ で配信する）")
    parser.add_argument("--upstream-header", action="append", default=[],
                        metavar="K=V",
                        help="上流へ必ず付けるヘッダ（複数指定可）。"
                             "ngrok の警告ページ回避に使う")
    parser.add_argument("--stop-pid", type=int, default=0,
                        help="/stop 時に SIGTERM を送る PID。"
                             "uzu のようにランチャーが子プロセスを抱える場合、"
                             "ランチャーの PID を渡すと後片付けごと止まる")
    parser.add_argument("--no-quota", action="store_true",
                        help="/quota を無効化する（Kaggle 以外のローカル用）")
    parser.add_argument("--enforce-sampling", action="store_true",
                        help="chat/completions の temperature<=0 を %.1f に、"
                             "repetition_penalty を %.2f 以上に矯正する"
                             % (TEMP_FLOOR, RP_FLOOR))
    ARGS = parser.parse_args()

    user = os.environ.get("API_USER") or "ollama"
    password = os.environ.get("API_PASS") or ""
    if password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        EXPECTED_AUTH = "Basic " + token
    else:
        # 既定は認証なし。URL を知っていれば誰でも使える点だけ注意。
        print("auth: OFF (API_PASS 未設定)", flush=True)

    global CHAT_BYTES
    if ARGS.chat_html:
        try:
            with open(ARGS.chat_html, "rb") as f:
                CHAT_BYTES = f.read()
        except OSError as exc:
            # UI 無しでも API は動く。落とさず警告だけ出す。
            sys.stderr.write("chat ui not loaded: %s\n" % (exc,))
        else:
            print("chat ui: %s (%d B)" % (ARGS.chat_html, len(CHAT_BYTES)),
                  flush=True)

    httpd = Server((ARGS.host, ARGS.port), Handler)
    print(f"auth proxy: {ARGS.host}:{ARGS.port} -> {ARGS.upstream} "
          f"(user={user})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

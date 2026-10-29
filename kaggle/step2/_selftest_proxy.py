#!/usr/bin/env python3
"""ノートに埋め込まれた auth_proxy.py の実挙動を検証する。

検証項目（本番と同じ「認証なし」で起動して検証する）:
  1. /healthz は即応答（トンネル疎通用）
  2. 認証なしで /api/tags → 200（既定は認証 OFF）
  3. Authorization を付けても 200 で、上流には漏れない
  4. ストリーミング → chunked で逐次受信できる
  5. chunked リクエストボディ → 上流に正確に届く
  6. HEAD → ハングせず即完了
  7. OPTIONS プリフライト → 204 をローカル即答（上流へ流さない）
  8. 非 ASCII Authorization でも 500 にならない
  9. API_PASS を設定したときだけ 401 に戻る（任意機能の退行検知）
  + ノート埋め込み == auth_proxy.py / llama-chat.html
  + UI が認証なしで 200・秘密を含まない
"""
import base64
import atexit
import os
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
NB = HERE / "step2-orcarouter-qwen38-27b-cloudflared.ipynb"
# vLLM 版（別ビルダー）にも同じ proxy を埋め込む。片方だけ再生成すると
# 古い proxy が配信されるので、両方を機械的に突き合わせる。
NB_VLLM = (HERE.parent / "step2-vllm-kernel" / "cloudflared"
           / "step2-vllm-qwen38-27b-cloudflared.ipynb")
SRC = HERE / "auth_proxy.py"
# 展開先はリポジトリ内に置かない（毎回生成されるゴミが残るため）。名前は
# pkill のパターンに使うので固定。
RESTORED = pathlib.Path(tempfile.gettempdir()) / "_restored_auth_proxy.py"
# ログも tempdir に置く。repo 直下に残すと git status が汚れ、
# 実行のたびに未追跡ファイルが増える（以前 HERE/_proxy.log に書いていた）。
# PID を付けて同時実行でも混線させない。
PROXY_LOG = (pathlib.Path(tempfile.gettempdir())
             / f"_auth_proxy_selftest_{os.getpid()}.log")


def read_log(n=300):
    """ログ末尾を返す。無い/非 UTF-8 でも例外にしない（診断で落とさない）。"""
    try:
        return PROXY_LOG.read_text(errors="replace")[-n:]
    except OSError:
        return ""

UP_PORT, PROXY_PORT = 18901, 18902
USER, PASS_ = "tester", "s3cret-pass"
TOKEN = base64.b64encode(f"{USER}:{PASS_}".encode()).decode()
AUTH = {"Authorization": f"Basic {TOKEN}"}

results = []


def _cleanup():
    """例外終了でも proxy / 上流サーバを確実に落とす（残留は誤判定の元）。"""
    p = globals().get("proc")
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(3)
        except Exception:
            p.kill()
    u = globals().get("up")
    if u is not None:
        u.shutdown()
    # 成功時はログ不要。失敗時だけ残して原因を見られるようにする。
    if globals().get("_failed") is False:
        try:
            PROXY_LOG.unlink()
        except OSError:
            pass


atexit.register(_cleanup)


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


# ── 1) ノートから埋め込み proxy を復元し、ファイル一致を確認 ──
import json as _json


def _embedded(nb_path, var):
    """notebook の設定セルから `<var> = "<base64>"` を取り出す。"""
    try:
        nb = _json.load(open(nb_path))
    except OSError:
        return None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            if line.startswith(var + " = "):
                return line.split("=", 1)[1].strip().strip('"')
    return None


b64 = _embedded(NB, "PROXY_SRC_B64")
if not b64:
    print("PROXY_SRC_B64 が設定セルに見つからない")
    sys.exit(1)

# 前回実行が例外終了したときの残留プロセスを掃除する。
# 残留プロセスが先にポートを掴むと、**古いコード**が応答して誤判定する。
subprocess.run(["pkill", "-f", "_restored_auth_proxy.py"], capture_output=True)
time.sleep(0.6)

import base64 as _b64
RESTORED.write_bytes(_b64.b64decode(b64))
same = RESTORED.read_bytes() == SRC.read_bytes()
check("notebook埋め込み == auth_proxy.py", same,
      f"{len(RESTORED.read_bytes())} B")

# チャット UI も同じく埋め込み。片方だけ更新すると UI が古いまま配信される
# （proxy は動くので気づきにくい）。
# ビルダーはリポジトリ直下の llama-chat.html を埋め込む（ローカル uzu 版と同一）。
chat_src = HERE.parent.parent / "llama-chat.html"
chat_b64 = _embedded(NB, "CHAT_SRC_B64")
chat_ok = bool(chat_b64) and _b64.b64decode(chat_b64) == chat_src.read_bytes()
check("notebook埋め込み == llama-chat.html", chat_ok,
      f"{chat_src.stat().st_size} B" if chat_ok else "不一致 or 未埋め込み")

# vLLM 版も同じ内容でなければならない（再生成漏れの検出）。
if NB_VLLM.is_file():
    _p2 = _embedded(NB_VLLM, "PROXY_SRC_B64")
    check("vLLM notebook 埋め込み == auth_proxy.py",
          bool(_p2) and _b64.b64decode(_p2) == SRC.read_bytes(),
          "embed 無し" if not _p2 else
          ("一致" if _b64.b64decode(_p2) == SRC.read_bytes()
           else f"不一致 {len(_b64.b64decode(_p2))} B → build_step2_vllm.py 再生成"))
    _c2 = _embedded(NB_VLLM, "CHAT_SRC_B64")
    check("vLLM notebook 埋め込み == llama-chat.html",
          bool(_c2) and _b64.b64decode(_c2) == chat_src.read_bytes(),
          "embed 無し" if not _c2 else "一致")


# ── 2) ダミー上流（ストリーミング / echo / chunked 受信確認） ──
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    got_body = None
    got_te = None

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/stream"):
            body = json.dumps({"ok": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(5):
                piece = (json.dumps({"n": i}) + "\n").encode()
                self.wfile.write(b"%x\r\n" % len(piece))
                self.wfile.write(piece)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.35)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._echo(b"")

    def do_POST(self):
        Upstream.got_te = self.headers.get("Transfer-Encoding")
        if "chunked" in (Upstream.got_te or "").lower():
            out = bytearray()
            while True:
                ln = self.rfile.readline().strip().split(b";")[0]
                n = int(ln, 16)
                if n == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    break
                out += self.rfile.read(n)
                self.rfile.read(2)
            body = bytes(out)
        else:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Upstream.got_body = body
        self._echo(body)

    def _echo(self, body):
        payload = json.dumps({
            "echo": body.decode(errors="replace"),
            "path": self.path,
            "auth": self.headers.get("Authorization") is not None,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


up = ThreadingHTTPServer(("127.0.0.1", UP_PORT), Upstream)
threading.Thread(target=up.serve_forever, daemon=True).start()


# ── 3) 埋め込み proxy 起動 ──
# 本番と同じ「認証なし」で起動する（API_PASS 空 = 素通し）
env = {**__import__("os").environ, "API_USER": "", "API_PASS": ""}
plog = open(PROXY_LOG, "w")
proc = subprocess.Popen(
    [sys.executable, str(RESTORED), "--port", str(PROXY_PORT),
     "--upstream", f"http://127.0.0.1:{UP_PORT}",
     "--chat-html", str(HERE.parent.parent / "llama-chat.html")],
    env=env, stdout=plog, stderr=plog,
)
plog.close()   # 子は fd を dup 済み。親側は即解放（atexit まで開けっ放しにしない）
BASE = f"http://127.0.0.1:{PROXY_PORT}"


def wait_up():
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{BASE}/healthz", timeout=1).read()
            return True
        except Exception:
            time.sleep(0.25)
    return False


started = wait_up()
# 自プロセスが生きていること（= 自分がポートを掴んでいること）を確認
alive_self = proc.poll() is None
check("proxy 起動", started and alive_self,
      f"poll={proc.poll()} " + read_log())
if started and not alive_self:
    print("→ 古い残留プロセスが応答している可能性")
    started = False

if started:
    # 1) healthz 認証不要
    r = urllib.request.urlopen(f"{BASE}/healthz", timeout=5)
    check("/healthz 認証なしで200", r.status == 200, r.read().decode())

    # 1b) チャット UI は認証なしで 200（静的な HTML のみ）
    r = urllib.request.urlopen(f"{BASE}/", timeout=5)
    ui = r.read()
    # llama-chat.html は小文字の <!doctype html> で始まる（旧 chat.html は大文字）。
    # 大文字小文字を区別せずに判定する。
    _head = ui[:200].lower()
    check("UI / 認証なしで200", r.status == 200
          and r.headers.get("Content-Type", "").startswith("text/html")
          and b"<!doctype html" in _head,
          f"{len(ui)} B / {r.headers.get('Content-Type')}")

    # 1b2) HEAD / は本文なしで返る（ハングしない）
    req = urllib.request.Request(f"{BASE}/", method="HEAD")
    r = urllib.request.urlopen(req, timeout=5)
    check("HEAD / がハングしない", r.status == 200 and not r.read(),
          f"status={r.status}")

    # 1b3) POST / にボディを送っても次のリクエストが壊れない
    #      UI は認証前に応答するので、ボディを読み捨てないと
    #      keep-alive 上でリクエスト位置がずれる（スマグリング）。
    try:
        req = urllib.request.Request(f"{BASE}/", data=b"x" * 1024,
                                     method="POST")
        r = urllib.request.urlopen(req, timeout=5)
        r.read()
        # 同じ接続で続けて叩けることを確認（urllib は接続を再利用しないので
        # ここでは「サーバが落ちない/応答する」ことだけ見る）
        r2 = urllib.request.urlopen(f"{BASE}/api/tags", timeout=5)
        check("POST / 後も応答する", r.status == 200 and r2.status == 200,
              f"ui={r.status} tags={r2.status}")
    except urllib.error.HTTPError as e:
        check("POST / 後も応答する", False, f"HTTP {e.code}")
    except Exception as e:
        check("POST / 後も応答する", False, f"{type(e).__name__}: {e}")

    # 1c) UI に認証情報が埋まっていないこと（公開パスなので必須）
    check("UI に認証情報が無い", PASS_.encode() not in ui
          and b'id="pass"' not in ui,
          "認証情報が HTML に混入" if PASS_.encode() in ui else "ok")

    # 1b) UI に残り時間表示と停止ボタンの器があるか
    check("UI に残り時間の器", b'id="quotaLeft"' in ui, "quotaLeft")
    check("UI に停止ボタンの器", b'id="stopServer"' in ui, "stopServer")

    # 1c) /quota は管理エンドポイントとして応答する
    #     kaggle.json が無い環境では available=false でも 200 を返す仕様。
    try:
        r = urllib.request.urlopen(f"{BASE}/quota", timeout=20)
        qd = json.loads(r.read())
        check("/quota が応答", r.status == 200 and "available" in qd,
              f"available={qd.get('available')}")
    except Exception as e:
        check("/quota が応答", False, f"{type(e).__name__}: {e}")

    # 2) 認証なし → 200（既定は認証 OFF）
    try:
        r = urllib.request.urlopen(f"{BASE}/api/tags", timeout=5)
        check("認証なし→200（既定 OFF）", r.status == 200, f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        check("認証なし→200（既定 OFF）", False,
              f"HTTP {e.code} — 認証が有効になっている")

    # 3) Authorization を付けても 200
    #    認証ヘッダは proxy で消費し**上流には渡さない**（設計）。
    #    Ollama は認証を要求しないので、上流に auth=False なのが正しい。
    try:
        req = urllib.request.Request(f"{BASE}/api/tags", headers=AUTH)
        r = urllib.request.urlopen(req, timeout=5)
        d = json.loads(r.read())
        check("Authorization 付き→200（上流に漏れない）",
              r.status == 200 and d.get("path") == "/api/tags"
              and d.get("auth") is False,
              f"path={d.get('path')} upstream_auth={d.get('auth')}")
    except Exception as e:
        check("Authorization 付き→200", False, f"{type(e).__name__}: {e}")
        print("--- proxy log ---")
        print(read_log(2000))
        print("--- upstream は生存か ---")
        try:
            print("upstream:", urllib.request.urlopen(
                f"http://127.0.0.1:{UP_PORT}/api/tags", timeout=3).status)
        except Exception as e2:
            print("upstream NG:", type(e2).__name__, e2)

    # 4) ストリーミング（逐次受信）
    req = urllib.request.Request(f"{BASE}/stream", headers=AUTH)
    t0, times = time.time(), []
    with urllib.request.urlopen(req, timeout=30) as r:
        for raw in r:
            if raw.strip():
                times.append(round(time.time() - t0, 2))
    spread = (times[-1] - times[0]) if len(times) >= 2 else 0
    check("ストリーミング逐次受信", len(times) == 5 and spread > 0.8,
          f"chunks={len(times)} first={times[:1]} last={times[-1:]}")

    # 5) chunked リクエストボディ（改修点）
    #    urllib は常に Content-Length を付けるので、ここは http.client で
    #    Transfer-Encoding: chunked を原始的に組み立てて送る。
    import http.client

    payload = json.dumps({"big": "x" * 5000}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PROXY_PORT, timeout=10)
    conn.putrequest("POST", "/api/chat", skip_accept_encoding=True)
    conn.putheader("Transfer-Encoding", "chunked")
    conn.putheader("Content-Type", "application/json")
    for k, v in AUTH.items():
        conn.putheader(k, v)
    conn.endheaders()
    for i in range(0, len(payload), 1000):
        piece = payload[i:i + 1000]
        conn.send(b"%x\r\n" % len(piece) + piece + b"\r\n")
    conn.send(b"0\r\n\r\n")
    r = conn.getresponse()
    d = json.loads(r.read())
    conn.close()
    got = Upstream.got_body or b""
    # proxy は chunked を読み切ってから Content-Length 付きで上流へ再送する
    #（= ボディ欠落なし・チャンク境界を透過）。上流に正確に届けば正。
    check("chunked リクエストボディ到達",
          r.status == 200 and got == payload and d.get("echo") == payload.decode(),
          f"upstream TE={Upstream.got_te} "
          f"recv={len(got)}/{len(payload)}B match={got == payload}")

    # 6) HEAD（改修点）: ハングしないこと
    class _Timeout(Exception):
        pass

    def head_test():
        try:
            req = urllib.request.Request(f"{BASE}/api/tags", method="HEAD",
                                         headers=AUTH)
            with urllib.request.urlopen(req, timeout=8) as r:
                head_test.status = r.status
                r.read()
        except Exception as e:
            head_test.err = e

    head_test.status = head_test.err = None
    th = threading.Thread(target=head_test, daemon=True)
    th.start()
    th.join(8)
    check("HEAD がハングしない", not th.is_alive() and head_test.err is None,
          f"status={head_test.status} err={head_test.err}")

    # 7) OPTIONS はローカル即答 204（認証不要・上流に流さない）
    try:
        req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                     method="OPTIONS")
        r = urllib.request.urlopen(req, timeout=5)
        check("OPTIONS プリフライト→204（ローカル即答）",
              r.status == 204, f"HTTP {r.status}")
    except Exception as e:
        check("OPTIONS プリフライト→204（ローカル即答）", False,
              f"{type(e).__name__}: {e}")

    # 8) 非 ASCII の Authorization → 401（compare_digest が TypeError に
    #    ならないこと。例外なら 500 になりログが汚れる）
    import http.client as _hc
    c = _hc.HTTPConnection("127.0.0.1", PROXY_PORT, timeout=5)
    c.request("GET", "/api/tags", headers={"Authorization": "Basic \xff\xfe"})
    rr = c.getresponse()
    rr.read()
    c.close()
    check("非ASCII Authorization でも 500 にならない",
          rr.status in (200, 401), f"HTTP {rr.status}")

    # 9) API_PASS を設定したときだけ 401 に戻る（任意機能の退行検知）
    P2 = PROXY_PORT + 1
    env2 = {**__import__("os").environ, "API_USER": USER, "API_PASS": PASS_}
    p2 = subprocess.Popen(
        [sys.executable, str(RESTORED), "--port", str(P2),
         "--upstream", f"http://127.0.0.1:{UP_PORT}"],
        env=env2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{P2}/healthz",
                                       timeout=1).read()
                break
            except Exception:
                time.sleep(0.25)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{P2}/api/tags",
                                   timeout=5)
            check("API_PASS 設定時→401", False, "200 が返った")
        except urllib.error.HTTPError as e:
            check("API_PASS 設定時→401", e.code == 401, f"HTTP {e.code}")
        req = urllib.request.Request(f"http://127.0.0.1:{P2}/api/tags",
                                     headers=AUTH)
        r = urllib.request.urlopen(req, timeout=5)
        check("API_PASS 設定時→認証ありで200", r.status == 200,
              f"HTTP {r.status}")
    finally:
        p2.terminate()
        try:
            p2.wait(3)
        except Exception:
            p2.kill()

# ── 会話（マルチターン）が proxy で壊れない ──
# UI は messages に履歴を積んで送る。proxy はサンプリングだけ矯正して
# messages には触らないので、上流に完全一致で届くはず。
_conv = [
    {"role": "system", "content": "あなたは日本語で簡潔に答えます。"},
    {"role": "user", "content": "私の名前はコウヘイです。"},
    {"role": "assistant", "content": "コウヘイさん、覚えました。"},
    {"role": "user", "content": "私の名前は？"},
]
try:
    _body = _json.dumps({"model": "m", "messages": _conv,
                         "temperature": 0.2,
                         "repetition_penalty": 1.0}).encode()
    _req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=_body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(_req, timeout=30) as _r:
        _resp = _json.loads(_r.read())
    _got = _json.loads(Upstream.got_body or b"{}").get("messages")
    check("マルチターン履歴が上流へそのまま渡る", _got == _conv,
          f"n={len(_got or [])}")
    # 上流モックは choices ではなく echo を返す。欠落なく戻れば経路は通っている。
    check("会話応答がクライアントへ返る",
          bool((_resp.get("choices") or [{}])[0].get("message")
               or _resp.get("echo")),
          str(_resp)[:80])
except Exception as e:
    check("マルチターン履歴が上流へそのまま渡る", False,
          f"{type(e).__name__}: {e}")

# ── keepalive リセット（チャット送信で RUN=1 / MINUTES=60 を書き直す）──
# ここが動かないと「チャットしたのに 20 分で落ちる」。実ファイルを触るので
# 元の内容を退避し、最後に戻す。
KD = pathlib.Path("/tmp/keepalive.conf")
_ka_backup = KD.read_bytes() if KD.is_file() else None
try:
    def _post_chat():
        body = _json.dumps({"model": "x",
                            "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as _r:
            _r.read()

    KD.write_text("RUN=0\nMINUTES=1\n")      # 停止状態から
    try:
        _post_chat()
        txt = KD.read_text()
        check("チャット送信で RUN=1/MINUTES=60 にリセット",
              "RUN=1" in txt and "MINUTES=60" in txt, txt.replace("\n", " "))
    except Exception as e:
        check("チャット送信で RUN=1/MINUTES=60 にリセット", False,
              f"{type(e).__name__}: {e}")

    # 上流へ中継されるリクエストは GET でも延長される（実装は _relay の
    # 先頭で _reset_keepalive()）。UI のポーリングでも延びる仕様なので、
    # ここは「延びる」が正。延ばしたくないのは /quota（下で確認）。
    KD.write_text("RUN=0\nMINUTES=1\n")
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=10).read()
    except Exception:
        pass
    txt = KD.read_text()
    check("GET /v1/models（中継）でも延長される",
          "RUN=1" in txt and "MINUTES=60" in txt, txt.replace("\n", " "))

    # 管理系 /quota も書き換えない
    KD.write_text("RUN=0\nMINUTES=1\n")
    try:
        urllib.request.urlopen(f"{BASE}/quota", timeout=10).read()
    except Exception:
        pass
    txt = KD.read_text()
    check("/quota は keepalive を書き換えない", "MINUTES=1" in txt,
          txt.replace("\n", " "))
finally:
    if _ka_backup is None:
        try:
            KD.unlink()
        except OSError:
            pass
    else:
        KD.write_bytes(_ka_backup)

proc.terminate()
try:
    proc.wait(3)
except Exception:
    proc.kill()
up.shutdown()
atexit.unregister(_cleanup)

# ── サンプリング強制（--enforce-sampling）の単体確認 ──
# vLLM 版だけが使う任意機能。既定（Ollama 版）では素通しのままであること。
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_ap_unit", SRC)
_ap = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ap)
_ap.ARGS = type("A", (), {"enforce_sampling": False})()

_off = b'{"temperature": 0}'
check("enforce OFF は素通し",
      _ap.enforce_sampling(_off, "/v1/chat/completions") == _off)

_ap.ARGS = type("A", (), {"enforce_sampling": True})()
_d = _json.loads(_ap.enforce_sampling(
    b'{"temperature": 0, "repetition_penalty": 1.0}',
    "/v1/chat/completions").decode())
check("enforce ON で温度/RP を矯正",
      _d["temperature"] == 0.7 and _d["repetition_penalty"] == 1.05,
      f"temp={_d['temperature']} rp={_d['repetition_penalty']}")

_d2 = _json.loads(_ap.enforce_sampling(b'{"messages": []}',
                                      "/v1/chat/completions").decode())
check("enforce ON で未指定にも既定値を注入",
      _d2["temperature"] == 0.7 and _d2["repetition_penalty"] == 1.05)

_keep = b'{"temperature": 0.9, "repetition_penalty": 1.1}'
check("enforce ON でも正常値は変更しない",
      _ap.enforce_sampling(_keep, "/v1/chat/completions") == _keep)

_emb = b'{"input": "hello"}'
check("chat/completions 以外は触らない",
      _ap.enforce_sampling(_emb, "/v1/embeddings") == _emb)

# 境界値。矯正が「黙って無効化」される経路を潰す。
_q = _json.loads(_ap.enforce_sampling(
    b'{"temperature": 0}', "/v1/chat/completions?x=1").decode())
check("query 付きでも矯正される", _q["temperature"] == 0.7,
      f"temp={_q.get('temperature')}")

_q2 = _json.loads(_ap.enforce_sampling(
    b'{"temperature": 0}', "/chat/completions/?api-version=1").decode())
check("query + 末尾スラッシュでも矯正される", _q2["temperature"] == 0.7)

for _raw in (b"notjson", b"", b"[]", b'"str"', b"null"):
    check(f"不正ボディは素通し {_raw[:8]!r}",
          _ap.enforce_sampling(_raw, "/v1/chat/completions") == _raw)

for _bad in ('{"temperature": "0"}', '{"temperature": true}',
             '{"temperature": -1}', '{"temperature": null}',
             '{"temperature": 0.0}'):
    _d = _json.loads(_ap.enforce_sampling(
        _bad.encode(), "/v1/chat/completions").decode())
    check(f"温度 {_bad} -> 0.7", _d["temperature"] == 0.7,
          f"temp={_d.get('temperature')!r}")

for _bad in ('{"repetition_penalty": null}', '{"repetition_penalty": 0}',
             '{"repetition_penalty": 0.5}', '{"repetition_penalty": "1.0"}'):
    _d = _json.loads(_ap.enforce_sampling(
        _bad.encode(), "/v1/chat/completions").decode())
    check(f"RP {_bad} -> 1.05", _d["repetition_penalty"] == 1.05,
          f"rp={_d.get('repetition_penalty')!r}")

_body = _json.dumps({"model": "m", "unknown_x": 1,
                     "messages": [{"role": "user", "content": "日本語のテスト"}]},
                    ensure_ascii=False).encode()
_d = _json.loads(_ap.enforce_sampling(_body, "/v1/chat/completions").decode())
check("非ASCIIと未知フィールドを保持",
      _d["messages"][0]["content"] == "日本語のテスト" and _d["unknown_x"] == 1)

# ── 統合: --enforce-sampling 有効な proxy を別ポートで起動 ──
# 単体テストは関数を直接叩いているので、HTTP 経路（Content-Length 再計算・
# query 付きパス・非ASCII）まで通ることを 1 回だけ確認する。
# 既存 Upstream は他テストの状態を持つため、専用の最小上流を別ポートで立てる。
PORT2 = PROXY_PORT + 1
UP2_PORT = UP_PORT + 5
LOG2 = PROXY_LOG.with_name(PROXY_LOG.stem + "-enforce.log")


class Upstream2(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    got = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        Upstream2.got = self.rfile.read(n)
        payload = _json.dumps({"echo": Upstream2.got.decode(errors="replace")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


up2srv = ThreadingHTTPServer(("127.0.0.1", UP2_PORT), Upstream2)
threading.Thread(target=up2srv.serve_forever, daemon=True).start()
proxy2 = None
try:
    plog2 = open(LOG2, "w")
    proxy2 = subprocess.Popen(
        [sys.executable, str(RESTORED), "--port", str(PORT2),
         "--upstream", f"http://127.0.0.1:{UP2_PORT}",
         "--chat-html", str(HERE.parent.parent / "llama-chat.html"),
         "--enforce-sampling"],
        env=env, stdout=plog2, stderr=plog2)
    plog2.close()
    BASE2 = f"http://127.0.0.1:{PORT2}"
    up2 = False
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{BASE2}/healthz", timeout=1).read()
            up2 = True
            break
        except Exception:
            time.sleep(0.25)
    check("enforce proxy 起動", up2)
    if up2:
        _payload = json.dumps(
            {"model": "m", "temperature": 0, "repetition_penalty": 1.0,
             "messages": [{"role": "user", "content": "日本語のテスト"}]},
            ensure_ascii=False).encode()
        _req = urllib.request.Request(
            f"{BASE2}/v1/chat/completions?x=1", data=_payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(_req, timeout=30) as _r:
                _resp = _json.loads(_r.read().decode())
        except Exception as _exc:
            _log = ""
            try:
                _log = LOG2.read_text(errors="replace")[-600:]
            except OSError:
                pass
            check("HTTP 経路でも温度/RP を矯正", False,
                  f"{type(_exc).__name__}: {_exc} | log={_log} "
                  f"| upstream_got={(Upstream2.got or b'')[:120]!r}")
        else:
            _echoed = _json.loads(_resp.get("echo") or "{}")
            check("HTTP 経路でも温度/RP を矯正",
                  _echoed.get("temperature") == 0.7
                  and _echoed.get("repetition_penalty") == 1.05,
                  f"echo={str(_echoed)[:120]}")
            check("HTTP 経路でも日本語が壊れない",
                  (_echoed.get("messages") or [{}])[0].get("content") == "日本語のテスト")
finally:
    if proxy2 is not None:
        proxy2.terminate()
        try:
            proxy2.wait(3)
        except Exception:
            proxy2.kill()
    up2srv.shutdown()
    try:
        LOG2.unlink()
    except OSError:
        pass

nok = sum(1 for _, ok, _ in results if ok)
print(f"\n結果: {nok}/{len(results)} パス")
if nok == len(results):
    # 全通過ならログは不要。失敗時だけ残す（tempdir を溜めない）
    try:
        PROXY_LOG.unlink()
    except OSError:
        pass
sys.exit(0 if nok == len(results) else 1)

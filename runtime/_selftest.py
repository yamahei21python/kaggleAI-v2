#!/usr/bin/env python3
"""runtime/ の CPU 単体テスト（Kaggle セッション不要）。

    python3 runtime/_selftest.py
"""
import http.server
import json
import pathlib
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import runtime  # noqa: E402
from runtime import keepalive, net, proc, storage  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ✅ " if cond else "  ❌ ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
def test_bundle():
    print("\n=== source_bundle ===")
    src = runtime.source_bundle("net", "proc", "storage", "keepalive", "proxy", "ollama")
    for fn in ("def kc_probe_url", "def kc_proc_state", "def kc_save_output",
               "def kc_read_conf", "def kc_start_proxy", "def kc_unload_model",
               "def kc_detect_gpu", "def kc_wait_gpu_idle", "def kc_ollama_chat",
               "def kc_keepalive_deadline"):
        check(f"{fn} が含まれる", fn in src)
    try:
        compile(src, "<bundle>", "exec")
        check("連結セルがコンパイルできる", True)
    except SyntaxError as e:
        check("連結セルがコンパイルできる", False, str(e))
    check("未知のモジュール名は KeyError", _raises(lambda: runtime.source_bundle("nope"), KeyError))


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------
# net
# --------------------------------------------------------------------------
class _H(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def do_GET(self):
        if _H.mode == "ok":
            raw, code = b"{}", 200
        elif _H.mode == "401":
            raw, code = b"unauthorized", 401
        else:
            raw, code = b"busy", 503
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def test_net():
    print("\n=== net（トンネル URL / HTTP 待ち） ===")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="rt_net_"))
    log = tmp / "cloudflared.log"
    log.write_text("INF |  https://brave-tiger-xyz.trycloudflare.com  |\n")
    check("cloudflared ログから URL",
          net.kc_probe_url("cloudflared", str(log))
          == "https://brave-tiger-xyz.trycloudflare.com")
    log.write_text("https://not-atunnel.example.com\n")
    check("別ドメインは拾わない", net.kc_probe_url("cloudflared", str(log)) == "")
    check("ログ無しでも落ちない",
          net.kc_probe_url("cloudflared", str(tmp / "none.log")) == "")

    # ngrok は管理 API から取る（urlopen を差し替え）
    payload = json.dumps({"tunnels": [
        {"public_url": "http://127.0.0.1:4040"},
        {"public_url": "https://fixed.ngrok-free.dev"}]}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    orig = net.urllib.request.urlopen
    net.urllib.request.urlopen = lambda *a, **kw: _Resp()
    try:
        check("ngrok 管理 API から https URL",
              net.kc_probe_url("ngrok") == "https://fixed.ngrok-free.dev")
    finally:
        net.urllib.request.urlopen = orig

    # 待ちループは sleep を差し替えて即時化
    sleeps = []
    orig_sleep = net.time.sleep
    net.time.sleep = lambda s: sleeps.append(s)
    try:
        log.write_text("no url yet\n")
        check("URL 無し・時間切れ → 空",
              net.kc_wait_url("cloudflared", str(log), seconds=3) == "")
        check("既定の待ち回数ぶん sleep する", len(sleeps) == 3, str(sleeps))
        sleeps.clear()
        log.write_text("https://x-y-z.trycloudflare.com\n")
        check("URL が出たら即返す",
              net.kc_wait_url("cloudflared", str(log), seconds=5)
              == "https://x-y-z.trycloudflare.com")
        check("取得時は sleep しない", sleeps == [], str(sleeps))
    finally:
        net.time.sleep = orig_sleep
    shutil.rmtree(tmp, ignore_errors=True)

    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        _H.mode = "ok"
        check("kc_wait_http 200 → 経過秒", net.kc_wait_http(url, attempts=1, interval=2) == 2)
        _H.mode = "503"
        check("kc_wait_http 非200 → None", net.kc_wait_http(url, attempts=1, interval=1) is None)

        # retry
        n = {"c": 0}

        def flaky():
            n["c"] += 1
            if n["c"] < 3:
                raise OSError("boom")
            return "ok"

        check("kc_retry は一時エラーを吸収",
              net.kc_retry(flaky, tries=3, delay=0) == "ok" and n["c"] == 3)

        def http_err():
            raise urllib.error.HTTPError(url, 401, "no", {}, None)

        check("HTTPError は既定で即 raise",
              _raises(lambda: net.kc_retry(http_err, tries=3, delay=0),
                      urllib.error.HTTPError))
        n["c"] = 0

        def err_once():
            n["c"] += 1
            if n["c"] == 1:
                raise urllib.error.HTTPError(url, 503, "busy", {}, None)
            return "ok"

        check("skip_http_errors=False なら再試行",
              net.kc_retry(err_once, tries=2, delay=0, skip_http_errors=False) == "ok")
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------
# storage / keepalive
# --------------------------------------------------------------------------
def test_storage():
    print("\n=== storage ===")
    d1 = pathlib.Path(tempfile.mkdtemp(prefix="rt_o1_"))
    d2 = pathlib.Path(tempfile.mkdtemp(prefix="rt_o2_"))
    blocker = pathlib.Path(tempfile.mkdtemp(prefix="rt_blk_")) / "notadir"
    blocker.write_text("x")

    check("両方に書ける", storage.kc_save_output("a.txt", "hi", (str(d1), str(d2)))
          and (d1 / "a.txt").read_text() == "hi"
          and (d2 / "a.txt").read_text() == "hi")
    check("片方失敗でも True", storage.kc_save_output("b.txt", "x", (str(d1), str(blocker))))
    check("全滅なら False", storage.kc_save_output("c.txt", "x", (str(blocker),)) is False)

    tlog = d1 / "tail.log"
    tlog.write_text("line1\nline2\n")
    try:
        storage.kc_fail("boom", str(tlog), (str(d1),))
        check("kc_fail は RuntimeError", False)
    except RuntimeError as e:
        check("kc_fail は RuntimeError", "boom" in str(e))
    txt = (d1 / "FAILED.txt").read_text()
    check("FAILED.txt に理由と末尾", "boom" in txt and "line2" in txt)
    for d in (d1, d2, blocker.parent):
        shutil.rmtree(d, ignore_errors=True)


def test_keepalive():
    print("\n=== keepalive（RUN/MINUTES の読み取り） ===")
    d = pathlib.Path(tempfile.mkdtemp(prefix="rt_ka_"))
    conf = d / "keepalive.conf"

    def _fail(msg, *a, **kw):
        raise RuntimeError(msg)

    def rd(**kw):
        return keepalive.kc_read_conf(str(conf), **kw)

    check("ファイル無し → (1, 60)", rd()[:2] == (1, 60), str(rd()))
    check("ファイル無し mtime は None", rd()[2] is None, str(rd()))
    keepalive.kc_write_conf(conf, 1, 60)
    check("RUN=1/MINUTES=60", rd()[:2] == (1, 60))
    check("mtime が取れる", rd()[2] > 0)
    keepalive.kc_write_conf(conf, 0)
    check("RUN=0 → 停止（MINUTES は既定）", rd()[:2] == (0, 60), str(rd()))
    keepalive.kc_write_conf(conf, 1, 999)
    check("上限で頭打ち", rd(default_minutes=60, max_minutes=480)[:2] == (1, 480))
    keepalive.kc_write_conf(conf, 1, 0)
    check("MINUTES=0 は fail()", _raises(lambda: rd(fail=_fail), RuntimeError))
    check("MINUTES=0 / on_invalid=zero → 停止",
          rd(on_invalid="zero")[:2] == (0, 60), str(rd(on_invalid="zero")))
    conf.write_text("めちゃくちゃ\n")
    check("壊れた内容 → 既定", rd()[:2] == (1, 60), str(rd()))

    check("kc_conf_mtime は無いファイルで 0",
          keepalive.kc_conf_mtime(str(d / "none")) == 0.0)

    dl = keepalive.kc_keepalive_deadline
    check("締切: 起動 + minutes", dl(0, 5, None) == 300)
    check("締切: conf 更新で延長", dl(0, 5, 100) == 400)
    check("締切: hard_max で頭打ち", dl(0, 999, 500, hard_max_minutes=180) == 10800)
    check("締切: 古い mtime では縮まない", dl(100, 1, 50) == 160)
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# proc
# --------------------------------------------------------------------------
def test_proc():
    print("\n=== proc（死活 / 停止 / GPU） ===")
    # 一意な argv トークンを持つ子プロセスを作り、argv 一致で検証する。
    token = f"kcsleeper-{time.time_ns()}"
    p = subprocess.Popen([sys.executable, "-c",
                          "import time; time.sleep(30)", token])
    try:
        time.sleep(0.3)
        check("argv 一致で alive", proc.kc_proc_state(token, match="argv") == "alive")
        check("未知プロセスは DEAD", proc.kc_proc_state("no-such-proc-xyz") == "DEAD")
        check("kc_pids が PID を返す", str(p.pid) in proc.kc_pids(f"-f {token}"))
        left = proc.kc_stop_processes([(f"-f {token}", "child")],
                                      term_wait=0.2, final_wait=0.1)
        check("kc_stop_processes で停止", left == [], str(left))
        check("停止後は DEAD", p.wait(timeout=5) is not None
              and proc.kc_proc_state(token, match="argv") == "DEAD")
    finally:
        if p.poll() is None:
            p.kill()

    # GPU 検出は nvidia-smi を差し替えて CSV パースだけを見る
    class _R:
        stdout = "Tesla T4, 15360\nTesla T4, 15360\n"

    orig_run = proc.subprocess.run
    proc.subprocess.run = lambda *a, **kw: _R()
    try:
        names, mems = proc.kc_detect_gpu()
        check("kc_detect_gpu が名前/VRAM を返す",
              names == ["Tesla T4", "Tesla T4"] and mems == [15360, 15360],
              f"{names} {mems}")

        class _Idle:
            stdout = "0\n0\n"

        proc.subprocess.run = lambda *a, **kw: _Idle()
        check("kc_wait_gpu_idle は空きなら 0", proc.kc_wait_gpu_idle() == 0)

        class _Busy:
            stdout = "12000\n"

        proc.subprocess.run = lambda *a, **kw: _Busy()
        check("kc_wait_gpu_idle は埋まっていれば -1",
              proc.kc_wait_gpu_idle(timeout=1) == -1)
    finally:
        proc.subprocess.run = orig_run


def test_ollama():
    print("\n=== ollama（chat） ===")
    from runtime.engines import ollama
    seen = {}

    class _Chat(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen["payload"] = json.loads(raw)
            body = json.dumps({"message": {"content": "OK"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), _Chat)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = ollama.kc_ollama_chat(port, "m", "hi", num_predict=3, timeout=10)
        check("応答 dict を返す", d.get("message", {}).get("content") == "OK")
        p = seen["payload"]
        check("model/messages/options を送る",
              p["model"] == "m"
              and p["messages"] == [{"role": "user", "content": "hi"}]
              and p["options"] == {"num_predict": 3, "temperature": 0},
              str(p))
        check("think 未指定ならキーを入れない", "think" not in p)
        ollama.kc_ollama_chat(port, "m", "hi", think=False, timeout=10)
        check("think=False は送る", seen["payload"].get("think") is False)
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------
# proxy（認証なし検査のみ。起動は pkill を伴うためテストしない）
# --------------------------------------------------------------------------
def test_proxy():
    print("\n=== proxy（認証なし検査） ===")
    import runtime.proxy as rp
    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _H.mode = "ok"
        check("200 は OK", rp.kc_assert_no_auth(port) is True)
        _H.mode = "503"
        check("5xx は OK（上流未起動）", rp.kc_assert_no_auth(port) is True)
        _H.mode = "401"
        check("401 はエラー", _raises(lambda: rp.kc_assert_no_auth(port), RuntimeError))
        seen = []
        check("fail コールバックが呼ばれる",
              _raises(lambda: rp.kc_assert_no_auth(port, fail=lambda m: seen.append(m)),
                      RuntimeError) and bool(seen))
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    test_bundle()
    test_net()
    test_storage()
    test_keepalive()
    test_proc()
    test_ollama()
    test_proxy()
    print()
    if FAILS:
        print(f"結果: {len(FAILS)} 件失敗")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("結果: 全テスト パス")

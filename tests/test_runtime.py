"""runtime/ の CPU 単体テスト（Kaggle セッション不要）。"""
import base64
import http.server
import json
import os
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

import runtime
from runtime import keepalive, net, proc, quota, storage


def check(name, cond, detail=""):
    assert cond, name + (f" — {detail}" if detail else "")


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
def test_bundle():
    print("\n=== source_bundle ===")
    src = runtime.source_bundle("net", "proc", "storage", "keepalive", "proxy", "ollama")
    for fn in ("def kc_probe_url", "def kc_proc_state", "def kc_save_output",
               "def kc_read_conf", "def kc_start_proxy", "def kc_unload_model",
               "def kc_detect_gpu", "def kc_wait_gpu_idle", "def kc_ollama_chat",
               "def kc_keepalive_deadline", "def kc_keepalive_expired",
               "def kc_cloudflared_url", "def kc_find_dataset",
               "def kc_context_budget", "def kc_urlopen_v4"):
        check(f"{fn} が含まれる", fn in src)
    try:
        compile(src, "<bundle>", "exec")
        check("連結セルがコンパイルできる", True)
    except SyntaxError as e:
        check("連結セルがコンパイルできる", False, str(e))
    check("未知のモジュール名は KeyError", _raises(lambda: runtime.source_bundle("nope"), KeyError))

    root = pathlib.Path(runtime.__file__).resolve().parent.parent
    psrc = runtime.proxy_source(root / "kaggle" / "step2" / "auth_proxy.py")
    check("proxy_source に quota 実装が同梱",
          "def kc_fetch_quota" in psrc and "def main(" in psrc)
    try:
        compile(psrc, "<proxy>", "exec")
        check("同梱 proxy がコンパイルできる", True)
    except SyntaxError as e:
        check("同梱 proxy がコンパイルできる", False, str(e))


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
    check("cloudflared URL は版を固定",
          net.kc_cloudflared_url()
          == ("https://github.com/cloudflare/cloudflared/releases/download/"
              f"{net.KC_CLOUDFLARED_VERSION}/cloudflared-linux-amd64"))
    check("cloudflared URL は latest を使わない",
          "releases/latest" not in net.kc_cloudflared_url())
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
        _H.mode = "ok"
        with net.kc_urlopen_v4(url, timeout=5) as r:
            check("kc_urlopen_v4 通常経路", r.status == 200)

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
# quota
# --------------------------------------------------------------------------
def test_quota():
    print("\n=== quota（Kaggle API） ===")
    d = pathlib.Path(tempfile.mkdtemp(prefix="rt_q_"))
    cred = d / "kaggle.json"
    check("資格情報なし → (None, None)",
          quota.kc_kaggle_credentials(str(d / "none"), {}) == (None, None))
    cred.write_text(json.dumps({"username": "u", "key": "k"}))
    check("kaggle.json から読む",
          quota.kc_kaggle_credentials(str(cred), {}) == ("u", "k"))
    check("環境変数フォールバック",
          quota.kc_kaggle_credentials(
              str(d / "none"),
              {"KAGGLE_USERNAME": "eu", "KAGGLE_KEY": "ek"}) == ("eu", "ek"))
    cred.write_text("{}")
    check("空の kaggle.json は env へ",
          quota.kc_kaggle_credentials(
              str(cred),
              {"KAGGLE_USER": "gu", "KAGGLE_KEY": "gk"}) == ("gu", "gk"))

    seen = {}
    raw = json.dumps({
        "gpuQuota": {
            "timeUsed": {"seconds": 100},
            "totalTimeAllowed": {"seconds": 1800},
            "timeReserved": {"seconds": 50},
        },
        "quotaRefreshTime": "2026-11-01T00:00:00Z",
    }).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return raw

    def _open(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["timeout"] = timeout
        return _Resp()

    data = quota.kc_fetch_quota("u", "k", timeout=7, opener=_open)
    check("API URL / Basic 認証 / timeout",
          seen["url"] == quota.KC_QUOTA_URL
          and seen["auth"] == "Basic " + base64.b64encode(b"u:k").decode()
          and seen["timeout"] == 7, str(seen))
    s = quota.kc_quota_summary(data)
    check("要約（残り = 上限 - 使用 - 予約）",
          s["available"] and s["leftSeconds"] == 1650
          and s["refreshTime"] == "2026-11-01T00:00:00Z", str(s))
    check("gpuQuota 無しは available=False",
          quota.kc_quota_summary({})["available"] is False)
    shutil.rmtree(d, ignore_errors=True)


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


def test_find_dataset():
    print("\n=== storage.kc_find_dataset（マウント探索） ===")
    root = pathlib.Path(tempfile.mkdtemp(prefix="rt_ds_"))
    try:
        # 旧レイアウト: /kaggle/input/<slug>
        old = root / "old-slug"
        (old).mkdir()
        (old / "config.json").write_text("{}")
        check("旧レイアウトを見つける",
              storage.kc_find_dataset("old-slug", root=str(root)) == str(old))

        # 新レイアウト: /kaggle/input/datasets/<owner>/<slug>
        new = root / "datasets" / "owner" / "new-slug"
        new.mkdir(parents=True)
        (new / "config.json").write_text("{}")
        check("新レイアウト（datasets/<owner>/<slug>）も見つける",
              storage.kc_find_dataset("new-slug", root=str(root)) == str(new))

        # 空マウントは避け、中身のある方を選ぶ
        empty = root / "datasets" / "owner" / "both"
        empty.mkdir(parents=True)
        full = root / "both"
        full.mkdir()
        (full / "x").write_text("x")
        check("空でないディレクトリを優先",
              storage.kc_find_dataset("both", root=str(root)) == str(full))

        # 見つからないときは既定パスを返す（呼び出し側が isdir で判定する）
        check("見つからなければ既定パス",
              storage.kc_find_dataset("nope", root=str(root))
              == f"{root}/nope")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_context_budget():
    print("\n=== proc.kc_context_budget（VRAM 予算） ===")
    cands = [131072, 65536, 32768, 16384, 8192]
    # T4 x2 = 30720 MiB → 予算 26.4 GiB。モデル 16.52 GiB。
    ctx, budget = proc.kc_context_budget(30720, 16.52, 34.0, cands)
    check("予算は実効 VRAM の 88%", abs(budget - 30720 / 1024 * 0.88) < 1e-9,
          str(budget))
    check("収まる最大の候補を選ぶ", ctx == 131072, str(ctx))

    ctx2, _ = proc.kc_context_budget(15360, 16.52, 34.0, cands)
    check("VRAM が小さいほど候補が下がる", ctx2 < 131072, str(ctx2))

    # どれも収まらない → 最小候補（起動は試す）
    ctx3, _ = proc.kc_context_budget(15360, 30.0, 34.0, cands)
    check("全部収まらなければ最小候補", ctx3 == 8192, str(ctx3))

    # requested が収まるなら優先（direct の CTX_REQUESTED）
    ctx4, _ = proc.kc_context_budget(30720, 16.52, 34.0, cands,
                                     requested=32768)
    check("requested が収まるならそれを優先", ctx4 == 32768, str(ctx4))
    ctx5, _ = proc.kc_context_budget(30720, 16.52, 34.0, cands,
                                     requested=524288)
    check("requested が超過なら候補へ落とす", ctx5 == 131072, str(ctx5))

    check("kc_context_gib はモデル + KV",
          abs(proc.kc_context_gib(16.52, 34.0, 131072)
              - (16.52 + 34.0 * 131072 / 1024 / 1024)) < 1e-9)


def test_keepalive_module():
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

    # フェイククロック: now を明示して予算判定を検証する
    ex = keepalive.kc_keepalive_expired
    check("予算: 299 秒は未超過", not ex(299, 0, 5, None))
    check("予算: 300 秒で超過", ex(300, 0, 5, None))
    check("予算: conf 更新後は延長", not ex(350, 0, 5, 100) and ex(400, 0, 5, 100))
    check("予算: hard_max を超えない", ex(10800, 0, 999, 500))
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


def test_watch():
    print("\n=== proc（起動待ちの観測） ===")
    # /proc の形をしたディレクトリを作り、**変換だけ**を固定する。実プロセスの
    # RSS は環境で変わるので、そこを assert すると「たまたま通る」テストになる。
    root = pathlib.Path(tempfile.mkdtemp(prefix="kcproc-"))
    orig_vram = proc.kc_vram_used
    proc.kc_vram_used = lambda *a, **kw: []
    try:
        d = root / "4242"
        d.mkdir()
        (d / "status").write_text("Name:\tpython\nVmRSS:\t 1048576 kB\n")
        (d / "io").write_text("rchar: 3145728\nread_bytes: 2097152\n")
        s = proc.kc_sample(4242, root=str(root))
        check("VmRSS を MiB にする", s["rss_mib"] == 1024, str(s))
        check("read_bytes を MiB にする", s["read_mib"] == 2, str(s))
        check("rchar を MiB にする", s["char_mib"] == 3, str(s))
        check("GPU が無ければ vram は None", s["vram_mib"] is None, str(s))
        check("存在しない PID は None",
              proc.kc_sample(999999, root=str(root))["rss_mib"] is None)
        # io だけ読めない環境（権限が無い等）でも rss は取れて落ちない
        (d / "io").unlink()
        s2 = proc.kc_sample(4242, root=str(root))
        check("io が読めなくても rss は取れる",
              s2["rss_mib"] == 1024 and s2["read_mib"] is None, str(s2))
    finally:
        proc.kc_vram_used = orig_vram
        shutil.rmtree(root, ignore_errors=True)

    # 生きているプロセスに対しては溜まり、止めたら止まる。
    # `keep` を小さくして、上限で古いものが落ちることも同時に見る。
    # 溜まること・止まることは /proc が無くても確かめられる（os.kill だけで足りる）。
    p = subprocess.Popen([sys.executable, "-c",
                          "import time; time.sleep(30)"])
    samples = []
    try:
        samples, stop = proc.kc_watch(p.pid, every=0.1, keep=3)
        # 固定時間だけ待つと、`cli check` が他のテストと並んで走るときに
        # サンプルが 2 件しか来ず落ちる（実測）。`keep` に達するまで待つ。
        deadline = time.time() + 5
        while len(samples) < 3 and time.time() < deadline:
            time.sleep(0.02)
        check("kc_watch は keep 件まで溜める", len(samples) == 3, str(len(samples)))
        check("経過秒が入る", all("t" in x for x in samples), str(samples))
        if os.path.isdir("/proc"):          # macOS には無い
            check("rss が取れている", (samples[-1]["rss_mib"] or 0) > 0,
                  str(samples[-1]))
        else:
            print("  ℹ /proc が無いので rss の実測は飛ばす（Kaggle では効く）")
        # 相手が生きたままでも `stop()` で止まること。ここを見ないと、
        # 起動が通ったあと何時間も `[watch]` を吐き続ける実装に戻れる。
        stop()
        time.sleep(0.3)
        n = len(samples)
        time.sleep(0.4)
        check("stop() で観測を止める", len(samples) == n, f"{n} → {len(samples)}")
    finally:
        p.kill()
        p.wait()


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


# --------------------------------------------------------------------------
# gist（HTTP はローカルの偽サーバ。ネットワークに出ない）
# --------------------------------------------------------------------------
class _GistH(http.server.BaseHTTPRequestHandler):
    """GitHub Gist API の最小の偽物（PATCH / GET だけ）。"""

    def _send(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_PATCH(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.server.deny_patch:
            self._send(404, b'{"message":"Not Found"}')
            return
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            self._send(401, b'{"message":"Requires authentication"}')
            return
        self.server.patched = body
        self._send(200, b'{"ok":true}')

    def do_GET(self):
        if self.server.deny_get:
            self._send(500, b'{"message":"boom"}')
            return
        files = {}
        for name, v in (self.server.patched or {}).get("files", {}).items():
            content = v.get("content", "")
            files[name] = {"content": content, "size": len(content)}
        self._send(200, json.dumps({"files": files}).encode())

    def log_message(self, *a):
        pass


def test_gist():
    print("\n=== gist ===")
    import runtime.gist as g

    check("束に kc_gist_write が入る", "def kc_gist_write" in runtime.source_bundle("gist"))

    # ── 解決の経路。`""` だけ返すと「登録漏れ」と「紐付け漏れ」を分けられない ──
    saved = os.environ.pop("GITHUB_TOKEN", None)
    try:
        tok, why = g.kc_gh_token_detail()
        check("env が無ければ空", tok == "", tok)
        check("理由が付く", bool(why), why)
        os.environ["GITHUB_TOKEN"] = "dummy"
        check("env から読める", g.kc_gh_token_detail() == ("dummy", "env:GITHUB_TOKEN"))
    finally:
        os.environ.pop("GITHUB_TOKEN", None)
        if saved is not None:
            os.environ["GITHUB_TOKEN"] = saved

    # ── 失敗は必ず理由を持つ（元の不具合は「黙って return」だった） ──
    ok, why = g.kc_gist_write("", "f", "x", token="t")
    check("GIST_ID 空は失敗", ok is False and "GIST_ID" in why, why)
    ok, why = g.kc_gist_write("abc", "f", "x", token="")
    check("トークン空は失敗", ok is False and why, why)

    srv = socketserver.TCPServer(("127.0.0.1", 0), _GistH)
    srv.patched, srv.deny_patch, srv.deny_get = None, False, False
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = g._GIST_API
    g._GIST_API = f"http://127.0.0.1:{port}/gists/"
    try:
        r = g.kc_gist_selftest("abc", "vllm_status.txt", token="t")
        check("往復が通る", r["ok"] is True, json.dumps(r, ensure_ascii=False))
        check("読み返しで一致", r["matched"] is True)
        check("PATCH が届いている", srv.patched is not None)
        check("解決の経路も返す", r["token_detail"] == "引数で明示", r["token_detail"])

        # 404（ID 違い / 権限なし）は write_ok=False と**理由**で返る
        srv.deny_patch = True
        r = g.kc_gist_selftest("abc", "vllm_status.txt", token="t")
        check("404 は ok=False", r["ok"] is False)
        check("404 は理由が付く", "404" in r["write_detail"], r["write_detail"])

        # 書けたが読めない回は、write_ok と read_ok で分かれる
        srv.deny_patch = False
        srv.deny_get = True
        r = g.kc_gist_selftest("abc", "vllm_status.txt", token="t")
        check("読めない回は write_ok=True", r["write_ok"] is True)
        check("読めない回は read_ok=False", r["read_ok"] is False)
        check("読めない回も ok=False", r["ok"] is False)
        check("読めない回も理由が付く", bool(r["read_detail"]), r["read_detail"])
    finally:
        g._GIST_API = base
        srv.shutdown()
        srv.server_close()

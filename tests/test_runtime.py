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


def _raises_keyerror(fn, *args):
    """`fn(*args)` が KeyError になること（未知の host を弾く）。"""
    return _raises(lambda: fn(*args), KeyError)


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

    # ngrok は 2026-11-01 に廃止。表から消してあるので KeyError で気づける
    # （黙って "" を返すと「トンネルは張れたのに URL が取れない」に見える）。
    check("ngrok は表に無い（KeyError）", _raises_keyerror(net.kc_probe_url, "ngrok"))
    check("管理 API の分岐も消えている",
          '"api"' not in pathlib.Path(net.__file__).read_text())

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
# beacon（カーネルのコードを掲示板にする。ネットワークに出ない範囲だけ）
# --------------------------------------------------------------------------
def test_beacon():
    print("\n=== beacon ===")
    import runtime.beacon as b

    check("束に kc_beacon_publish が入る",
          "def kc_beacon_publish" in runtime.source_bundle("beacon"))

    # ── 掲示板のメタデータ。**実行しない**ので GPU もネットも要らない ──
    m = b.kc_beacon_meta("someone/serve-url-beacon")
    check("id がそのまま入る", m["id"] == "someone/serve-url-beacon")
    check("title は slug に一致", m["title"] == "serve-url-beacon")
    check("GPU を使わない", m["enable_gpu"] is False)
    check("ネットを使わない", m["enable_internet"] is False)
    check("非公開（URL が入口なので）", m["is_private"] is True)

    # ── 書いたものが読み戻せる（本文とパーサの契約） ──
    body = b.kc_beacon_body("https://x.trycloudflare.com", "serving", 1762051200.5)
    got = b.kc_beacon_parse(body)
    check("URL が戻る", got["url"] == "https://x.trycloudflare.com", got)
    check("STAGE が戻る", got["stage"] == "serving")
    check("UPDATED が float で戻る", got["updated"] == 1762051200.5)

    # 引用符が混ざっても壊れない（repr で埋めている）
    weird = b.kc_beacon_body('https://a/"b".example', "serving", 1.0)
    check("引用符入りでも戻る", b.kc_beacon_parse(weird)["url"] == 'https://a/"b".example')

    # ── 鮮度。古い値を掴む事故（2026-11-01）の防波堤 ──
    d = {"url": "u", "stage": "serving", "updated": 1000.0}
    check("push 前のスタンプなら本物", b.kc_beacon_is_fresh(d, 990) is True)
    check("1 時間後なら偽物", b.kc_beacon_is_fresh(d, 1000 + 3600) is False)
    check("UPDATED が無ければ偽物", b.kc_beacon_is_fresh({}, 0) is False)
    check("文字列の UPDATED は偽物",
          b.kc_beacon_is_fresh({"updated": "1000"}, 0) is False)

    # ── 失敗は必ず理由を持つ ──
    ok, why = b.kc_beacon_publish("u", slug="")
    check("slug 空は失敗", ok is False and "slug" in why, why)
    ok, why = b.kc_beacon_publish("u", slug="noowner")
    check("owner/name でない slug は失敗", ok is False and "owner/name" in why, why)

    saved = os.environ.pop("KC_BEACON_SLUG", None)
    try:
        os.environ["KC_BEACON_SLUG"] = "env/beacon"
        check("env から slug を読む", b.kc_beacon_slug() == "env/beacon")
        check("引数が env より優先", b.kc_beacon_slug("arg/x") == "arg/x")
    finally:
        os.environ.pop("KC_BEACON_SLUG", None)
        if saved is not None:
            os.environ["KC_BEACON_SLUG"] = saved


def test_beacon_cli_rejects_stale_and_probe(monkeypatch):
    """`cli beacon` は**古い値と検証用の値を返さない**。

    2026-11-01 に古いテスト URL を本物として掴む事故が出ている。ここが
    素通りすると、VM は存在しない URL に繋ぎに行って原因不明のタイムアウトになる。
    """
    print("\n=== cli beacon ===")
    import time

    from cli import beacon as cb
    from runtime import beacon as b

    now = time.time()
    cases = [
        ({"url": "https://ok", "stage": "serving", "updated": now}, 0, False),
        ({"url": "https://old", "stage": "serving", "updated": now - 99999}, 1, False),
        ({"url": "https://probe", "stage": "probe", "updated": now}, 1, False),
        ({"url": "https://probe", "stage": "probe", "updated": now}, 0, True),
        ({"url": "", "stage": "placeholder", "updated": 0.0}, 1, False),
    ]
    for data, want_rc, allow_probe in cases:
        monkeypatch.setattr(b, "kc_beacon_pull",
                            lambda *a, **k: (True, "pulled", dict(data)))
        argv = ["--json"] + (["--allow-probe"] if allow_probe else [])
        rc = cb.main(argv)
        check(f"{data['stage']}/{data['url'][:12]} → rc={want_rc}", rc == want_rc, rc)


def test_beacon_pull_falls_back_to_cli(tmp_path, monkeypatch):
    """Python パッケージが import できなくても CLI で引ける。

    手元の `python3` には `kaggle` が入っておらず（3.9 の user site にしか
    無い）、`cli beacon` が `No module named 'kaggle'` で死んでいた（実測）。
    逃げ道が無いと、掲示板を読む手段がカーネルの中だけになる。
    """
    print("\n=== beacon: CLI へのフォールバック ===")
    from runtime import beacon as b

    def boom():
        raise ModuleNotFoundError("No module named 'kaggle'")

    def fake_cli(slug, dest, exe="", timeout=180):
        with open(os.path.join(dest, "serve-url-beacon.py"), "w",
                  encoding="utf-8") as f:
            f.write(b.kc_beacon_body("https://cli.example", "serving", 1234.0))
        return True, "kaggle CLI"

    monkeypatch.setattr(b, "kc_kaggle_api", boom)
    monkeypatch.setattr(b, "kc_beacon_pull_cli", fake_cli)
    ok, detail, data = b.kc_beacon_pull("someone/board", dest=str(tmp_path))
    check("CLI 経由でも読める", ok is True, detail)
    check("URL が戻る", data["url"] == "https://cli.example", data)
    check("STAGE が戻る", data["stage"] == "serving", data)

    # 両方失敗したら、**両方の理由**を出す（片方だけだと原因を追えない）
    def fail_cli(slug, dest, exe="", timeout=180):
        return False, "kaggle CLI が 1 で終了"

    monkeypatch.setattr(b, "kc_beacon_pull_cli", fail_cli)
    ok, detail, data = b.kc_beacon_pull("someone/board", dest=str(tmp_path))
    check("両方失敗なら失敗", ok is False and data == {}, detail)
    check("API 側の理由も残る", "No module named" in detail, detail)
    check("CLI 側の理由も残る", "1 で終了" in detail, detail)


def test_beacon_kaggle_path_ignores_repo_dir(tmp_path):
    """リポジトリ直下の `kaggle/` がインストール済みパッケージを隠さない。

    `kaggle/`（カーネルのビルド元）は `__init__.py` を持たないので名前空間
    パッケージとして `import kaggle` を横取りし、`kaggle.api` が無いため
    `No module named 'kaggle.api'` になる（実測）。
    """
    print("\n=== beacon: kaggle パッケージの影を避ける ===")
    from runtime import beacon as b

    shadow = tmp_path / "repo"
    (shadow / "kaggle" / "cells").mkdir(parents=True)      # api が無い＝影
    real = tmp_path / "site"
    (real / "kaggle" / "api").mkdir(parents=True)          # 本物
    other = tmp_path / "stdlib"
    other.mkdir()

    keep = b.kc_kaggle_path([str(shadow), str(other), str(real)])
    check("影のパスは落ちる", str(shadow) not in keep, keep)
    check("本物は残る", str(real) in keep, keep)
    check("無関係なパスは残る", str(other) in keep, keep)

    entries = [str(shadow)]
    b.kc_kaggle_path(entries)
    check("引数のリストを書き換えない", entries == [str(shadow)], entries)


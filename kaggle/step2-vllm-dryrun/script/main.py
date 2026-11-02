#!/usr/bin/env python3
"""step2 vllm dryrun script（自動生成。直接編集しない）。

生成: kaggle/build_batch_script.py ← kaggle/build_vllm_dryrun.py のセル列
"""


# ── 🧪 vLLM Step2 ドライラン（GPU 不要） ──────────────────


# ── 🧩 共通ランタイム（proxy / tunnel / keepalive / verify） 


# ============================================================
# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）
# proxy / tunnel / keepalive / verify の配管をここに集約する。
# ============================================================
# ── runtime/net.py ────────────────────────────────────────
"""トンネル URL 取得と HTTP 疎通の共通処理（生成コードへ埋め込む）。"""
import json
import re
import socket
import time
import urllib.error
import urllib.request

# hosts は表駆動。今はログ解析の 1 種類しか無いが、待ち時間を URL の出所と
# 一緒に置いておくと `kc_wait_url` が既定値を知らずに済む。表に無い名前は
# KeyError になる（黙って空 URL を返さない）。
TUNNEL_HOSTS = {
    "cloudflared": {
        "url_re": re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com"),
        "wait_seconds": 90,
    },
}


# Dataset 同梱バイナリと同じ版に固定する（latest は再現性がない）。
KC_CLOUDFLARED_VERSION = "2026.10.4"


def kc_cloudflared_url(version=KC_CLOUDFLARED_VERSION):
    """cloudflared (linux amd64) のダウンロード URL。"""
    return ("https://github.com/cloudflare/cloudflared/releases/download/"
            f"{version}/cloudflared-linux-amd64")


def kc_probe_url(host, log_path=None):
    """現在の公開URLを返す（無ければ ""）。host は cloudflared のみ。"""
    spec = TUNNEL_HOSTS[host]
    if not log_path:
        return ""
    try:
        with open(log_path) as f:
            m = spec["url_re"].search(f.read())
        return m.group(0) if m else ""
    except OSError:
        return ""


def kc_wait_url(host, log_path=None, seconds=None, interval=1, progress_every=10):
    """公開URLが出るまで待つ。seconds 未指定なら host ごとの既定値。"""
    if seconds is None:
        seconds = TUNNEL_HOSTS[host]["wait_seconds"]
    for i in range(seconds):
        url = kc_probe_url(host, log_path)
        if url:
            return url
        if progress_every and i % progress_every == progress_every - 1:
            print(f"⏳ URL 待ち ({i + 1}/{seconds})...")
        time.sleep(interval)
    return ""


def kc_urlopen_v4(url, timeout=15):
    """公開 URL を開く。失敗したら IPv4 優先で名前解決して 1 回だけ再試行。

    Kaggle の VM は AAAA（IPv6）を返すのに IPv6 経路が無いことがあり、
    `urlopen` が "Network is unreachable" で落ちる（dryrun で実測）。
    自分の URL に届かないだけでトンネルは生きているので、ここで諦めると
    keepalive が**生きているトンネルを再起動して URL が変わってしまう**。
    """
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except Exception as first:
        err = first
    orig = socket.getaddrinfo

    def _v4(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _v4
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except Exception:
        raise err
    finally:
        socket.getaddrinfo = orig


def kc_wait_http(url, attempts=20, interval=3, timeout=10, opener=None):
    """200 が返るまで待つ。成功したら経過秒（(i+1)*interval）、失敗なら None。"""
    opener = opener or urllib.request.urlopen
    for i in range(attempts):
        try:
            with opener(url, timeout=timeout) as r:
                if r.status == 200:
                    return (i + 1) * interval
        except Exception:
            pass
        time.sleep(interval)
    return None


def kc_retry(fn, tries=6, delay=5, skip_http_errors=True):
    """一過性の通信エラーを吸収する。HTTP 応答は「答え」なので既定で再試行しない。"""
    last = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if skip_http_errors:
                raise
            last = e
        except Exception as e:
            last = e
        if i + 1 < tries:
            wait = delay * (i + 1)
            print(f"  ⚠ 一時的な通信エラー（{i + 1}/{tries}）: {last} "
                  f"→ {wait}s 待って再試行")
            time.sleep(wait)
    raise last


def kc_post_json(url, payload, timeout=60, headers=None, opener=None):
    """JSON POST のレスポンスを返す（ストリームは呼び出し側で読む）。"""
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=h, method="POST")
    return (opener or urllib.request.urlopen)(req, timeout=timeout)

# ── runtime/proc.py ───────────────────────────────────────
"""プロセス・GPU 状態の確認と後片付け（生成コードへ埋め込む）。"""
import re
import subprocess
import time


def _kc_live(pids, timeout=5):
    """zombie を除いた PID を返す。kill 直後は未 reap の Z が pgrep に残る。"""
    if not pids:
        return []
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,stat=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except Exception:
        return pids
    live = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[1].startswith("Z"):
            live.append(parts[0])
    return live


def kc_proc_state(name, match="exact", timeout=5):
    """alive / DEAD / unknown。match=exact は pgrep -x、argv は pgrep -f。"""
    try:
        flag = "-x" if match == "exact" else "-f"
        out = subprocess.run(["pgrep", flag, name], capture_output=True,
                             text=True, timeout=timeout).stdout
        pids = [p for p in out.split() if p.strip()]
        return "alive" if _kc_live(pids, timeout) else "DEAD"
    except Exception:
        return "unknown"


def kc_gpu_usage(timeout=5):
    """nvidia-smi の使用率/VRAM を 1 行で返す（CPU 環境では N/A）。"""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip().replace("\n", " | ")
    except Exception:
        return "N/A"


def kc_vram_used(timeout=5):
    """GPU ごとの使用 VRAM (MiB)。取れなければ []。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def kc_sample(pid, root="/proc"):
    """1 プロセスの様子を 1 つの dict にまとめる。取れない項目は None。

    `read_mib` は **実際にブロックデバイスまで行った読み込み量**（/proc の
    read_bytes）。ページキャッシュに当たった分は含まれないので、
    「重みを読み終えたか」はこちらで見る。`char_mib` は read(2) の総量で、
    キャッシュ命中も含む。両方出すのは、この 2 つが離れたときに
    「キャッシュから読んでいる」と言い切れるようにするため。

    `root` はテスト用の口。実プロセスの RSS は環境で変わるので、そこを
    assert すると「たまたま通る」テストになる。`/proc` の形をした
    ディレクトリを渡せば、変換（kB → MiB など）だけを固定できる。
    """
    s = {"rss_mib": None, "read_mib": None, "char_mib": None, "vram_mib": None}
    try:
        with open(f"{root}/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    s["rss_mib"] = round(int(line.split()[1]) / 1024)
                    break
    except OSError:
        pass
    try:
        with open(f"{root}/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    s["read_mib"] = int(line.split()[1]) // 2**20
                elif line.startswith("rchar:"):
                    s["char_mib"] = int(line.split()[1]) // 2**20
    except OSError:
        pass
    vram = kc_vram_used()
    if vram:
        s["vram_mib"] = sum(vram)
    return s


def kc_watch(pid, every=30, keep=80):
    """待機中のプロセスを `every` 秒ごとに観測する。戻り値は `(samples, stop)`。

    `stop()` を呼ぶと観測を止める。**プロセスの死だけでは止まらない**設計に
    してあるのは、待っている相手がサーバー本体だから: 起動が通れば同じ
    PID が何時間も生き続けるので、死を待つと `[watch]` が 30 秒ごとに
    何百行も出てログを埋める。呼び出し側は待機が終わった時点で止めること。

    無言の待機を「遅い」と「固まった」に分けるための計器。実測 2026-11-01
    exl3: 1800 秒の待機中に chat_server.log が **0 バイト**で、読み込みが
    進んでいたのか止まっていたのかまったく分からなかった（エンジン側の
    進捗表示は `--no-progress` で切ってあり、完走時の 1 行しか出ない）。
    そうなると次の手は「もう 1 回回す」しかなく、1 往復 30 分を捨てる。

    観測は read_bytes の**増分**が主役。1800 秒で 3 GiB しか進んでいなければ
    1.7 MiB/s ということなので、遅いのか止まったのかは即座に判る。

    戻り値の samples はそのまま result.json に入れる。失敗時にも
    「どこまで進んでいたか」が残るのが目的なので、成功・失敗を問わず入れる。
    """
    import os
    import threading

    samples = []
    stop = threading.Event()

    def loop():
        t0 = time.time()
        while not stop.wait(every):        # 止める指示が来たら即座に抜ける
            s = kc_sample(pid)
            s["t"] = round(time.time() - t0)
            samples.append(s)
            del samples[:-keep]
            print(f"  [watch] {s['t']}s rss={s['rss_mib']}MiB "
                  f"read={s['read_mib']}MiB vram={s['vram_mib']}MiB",
                  flush=True)
            try:
                os.kill(pid, 0)      # 死んでいれば ProcessLookupError
            except OSError:
                return

    threading.Thread(target=loop, daemon=True).start()
    return samples, stop.set


def kc_context_budget(vram_total_mib, model_gib, kv_kib_per_token, candidates,
                      requested=None, budget_ratio=0.88):
    """VRAM から (コンテキスト長, 予算 GiB) を決める。

    予算は実効 VRAM の `budget_ratio` 倍（T4 x2 の 30 GiB をそのまま
    使うと CUDA graph / activation で溢れるため 0.88 で見積もる）。
    `requested` が収まるならそれを優先し、駄目なら `candidates` を
    大きい順に試す。どれも収まらなければ最小候補を返す（起動は試す）。
    """
    budget_gib = vram_total_mib / 1024 * budget_ratio

    def fits(ctx):
        return model_gib + kv_kib_per_token * ctx / 1024 / 1024 <= budget_gib

    if requested and fits(requested):
        return requested, budget_gib
    for ctx in candidates:
        if fits(ctx):
            return ctx, budget_gib
    return candidates[-1], budget_gib


def kc_context_gib(model_gib, kv_kib_per_token, ctx):
    """コンテキスト 1 本ぶんの見積り（モデル + KV）を GiB で返す。"""
    return model_gib + kv_kib_per_token * ctx / 1024 / 1024


def kc_detect_gpu(timeout=15):
    """全 GPU の (名前, VRAM MiB) を返す。検出失敗は ([], [])。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip()
        names, mems = [], []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            names.append(parts[0].strip())
            mems.append(int(re.sub(r"[^0-9]", "", parts[1])))
        return names, mems
    except Exception as e:
        print(f"GPU 検出失敗: {e}")
        return [], []


def kc_wait_gpu_idle(limit_mib=500, timeout=180):
    """全 GPU の使用量が limit 未満になるまで待つ。経過秒、失敗は -1。

    名前は ollama の kc_wait_vram_free（/api/ps を見る）と衝突させない。
    """
    for i in range(timeout):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        used = [int(x) for x in out.splitlines() if x.strip().isdigit()]
        if used and max(used) < limit_mib:
            return i
        time.sleep(1)
    return -1


def _kc_argv(pattern):
    """"-x cloudflared" → ["-x", "cloudflared"]（shell を介さない）。"""
    return pattern.split()


def kc_pids(pattern, timeout=15):
    """pgrep の引数文字列（例: "-x cloudflared"）に一致する PID を返す。"""
    try:
        r = subprocess.run(["pgrep", *_kc_argv(pattern)],
                           capture_output=True, text=True, timeout=timeout)
        return _kc_live([p for p in r.stdout.split() if p.strip()], timeout)
    except Exception:
        return []


def kc_stop_processes(targets, term_wait=3, final_wait=2):
    """(pgrep_pattern, label) を TERM → 待機 → KILL で止め、残存ラベルを返す。"""
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-TERM", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  TERM → {label}")
    time.sleep(term_wait)
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-KILL", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  KILL → {label}（残存していた）")
    time.sleep(final_wait)
    return [label for pattern, label in targets if kc_pids(pattern)]

# ── runtime/storage.py ────────────────────────────────────
"""検証成果物の永続化と失敗記録（生成コードへ埋め込む）。"""
import glob
import os

# `kaggle kernels output` が zip 化するのは /kaggle/working だけ。
# /output は公式の永続先ではないが、書ければ保険になるので両方へ書く。
KC_OUT_DIRS = ("/kaggle/working", "/output")


def kc_find_dataset(slug, root="/kaggle/input"):
    """Dataset のマウント先を返す（見つからなければ既定パス）。

    Kaggle は `/kaggle/input/<slug>`（旧）と
    `/kaggle/input/datasets/<owner>/<slug>`（新）が同一日内に混在する。
    深さ 2 まで探し、**空でない**ディレクトリを優先する（空マウントを
    掴むと「Dataset はあるのに中身が無い」で落ちるため）。
    呼び出し側は戻り値をそのまま使ってよい（従来の `_find_input` と同じ）。
    """
    cands = []
    for pat in (f"{root}/{slug}", f"{root}/*/{slug}", f"{root}/*/*/{slug}"):
        cands += [p for p in glob.glob(pat) if os.path.isdir(p)]
    for p in sorted(cands):
        try:
            if os.listdir(p):
                return p
        except OSError:
            continue
    return cands[0] if cands else f"{root}/{slug}"


def kc_save_output(name, text, dirs=KC_OUT_DIRS):
    """1 つでも書ければ True（成功条件に使う）。"""
    saved = []
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
            saved.append(d)
        except Exception as e:
            print(f"  ⚠ {d} 書き出し失敗 ({e})")
    if saved:
        print(f"  📄 {name} 保存済: {', '.join(saved)}")
        return True
    print(f"  ⚠ 成果物を1つも保存できませんでした: {name}")
    return False


def kc_fail(msg, tail_path=None, dirs=KC_OUT_DIRS):
    """失敗原因を永続領域に残してから raise（ログ API は当てにできない）。"""
    tail = ""
    try:
        if tail_path and os.path.isfile(tail_path):
            tail = open(tail_path, errors="replace").read()[-3000:]
    except Exception:
        pass
    kc_save_output("FAILED.txt", f"{msg}\n\n{tail}", dirs)
    raise RuntimeError(msg)

# ── runtime/keepalive.py ──────────────────────────────────
"""キープアライブ設定ファイルと /stop の共通処理（生成コードへ埋め込む）。"""
import json
import os
import urllib.request


def kc_read_conf(path, default_minutes=60, max_minutes=480,
                 on_invalid="fail", fail=None, quiet_clamp=False):
    """(run, minutes, mtime) を返す。RUN=0 は (0, default_minutes, mtime)。

    on_invalid: "fail" は fail(msg)（未指定なら ValueError）、
                "zero" は停止扱い（RUN=0 と同じ）。
    quiet_clamp: proxy が書く既定値を上限で頭打ちにするとき黙る。
    """
    try:
        with open(path) as f:
            d = dict(l.strip().split("=", 1) for l in f
                     if l.strip() and "=" in l)
        run = int(d.get("RUN", 1))
        minutes = int(d.get("MINUTES", default_minutes))
    except OSError:
        run, minutes = 1, default_minutes
    except ValueError:
        run, minutes = 1, default_minutes
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if run <= 0:
        return 0, default_minutes, mtime
    if minutes <= 0:
        msg = f"KEEPALIVE_MINUTES={minutes} は無効（1〜{max_minutes} で指定）"
        if on_invalid == "zero":
            return 0, default_minutes, mtime
        if fail:
            fail(msg)
        raise ValueError(msg)
    if minutes > max_minutes:
        if not (quiet_clamp and minutes == default_minutes):
            print(f"⚠ KEEPALIVE_MINUTES={minutes} → {max_minutes} に制限")
        minutes = max_minutes
    return run, minutes, mtime


def kc_write_conf(path, run, minutes=None):
    with open(path, "w") as f:
        f.write(f"RUN={run}\n")
        if minutes is not None:
            f.write(f"MINUTES={minutes}\n")


def kc_conf_mtime(path):
    """keepalive.conf の更新時刻。proxy がチャット送信ごとに書き直す。"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def kc_keepalive_deadline(started, minutes, mtime, hard_max_minutes=180):
    """停止時刻。最後の活動（起動 or conf 更新）から minutes 分、全体は hard_max で頭打ち。"""
    last = max(started, mtime or started)
    return min(last + minutes * 60, started + hard_max_minutes * 60)


def kc_keepalive_expired(now, started, minutes, mtime, hard_max_minutes=180):
    """now 時点で時間予算を超えたか（now を渡すのでフェイククロックで検証できる）。"""
    return now >= kc_keepalive_deadline(started, minutes, mtime,
                                        hard_max_minutes)


def kc_stop_via_proxy(port, auth=None, timeout=30):
    headers = {"X-Kc-Stop": "1"}
    if auth:
        headers["Authorization"] = auth
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/stop", data=b"", method="POST",
        headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())

# ── runtime/proxy.py ──────────────────────────────────────
"""auth_proxy の起動と認証なし検査（生成コードへ埋め込む）。"""
import subprocess
import time
import urllib.error
import urllib.request


def kc_start_proxy(port, upstream, chat_html, proxy_path,
                   extra_args=(), env=None, log_path="/tmp/auth_proxy.log",
                   wait_seconds=30, label="プロキシ", fail=None,
                   python="python3"):
    """auth_proxy.py を起動し /healthz を待つ。失敗時は fail() か RuntimeError。"""
    subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
    time.sleep(1)
    cmd = [python, proxy_path, "--port", str(port),
           "--upstream", upstream, "--chat-html", chat_html, *extra_args]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    print(f"🔄 proxy PID: {proc.pid}")

    elapsed = None
    for i in range(wait_seconds):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
                if r.status == 200:
                    elapsed = i + 1
                    break
        except Exception:
            pass
        time.sleep(1)
    if elapsed is None:
        try:
            print(open(log_path).read()[-2000:])
        except OSError:
            pass
        msg = f"{label}の起動に失敗しました（{log_path}）"
        if fail:
            fail(msg, log_path)
        raise RuntimeError(msg)
    print(f"✅ {label}起動完了（{elapsed}s）")
    return proc


def kc_assert_no_auth(port, path="/api/tags", fail=None, timeout=10):
    """401/403 なら認証が有効 = 設定ミス。上流未起動の 5xx は正常。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            print(f"  ✅ 認証なし → HTTP {r.status}")
            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            msg = f"認証なしで HTTP {e.code} — 認証が有効になっています"
            if fail:
                fail(msg)
            raise RuntimeError(msg)
        print(f"  ✅ 認証なし → HTTP {e.code}（401 でない）")
        return True
    except Exception as e:
        msg = f"プロキシ検査に失敗: {type(e).__name__}: {e}"
        if fail:
            fail(msg)
        raise


# 設定（GPU 不要）
import json
import os
import re
import subprocess
import sys
import time

SERVED_NAME = "qwen3.8-27b-awq"
VLLM_PORT = 8000
PROXY_PORT = 11435
MTP_ENABLED = True
MTP_METHOD = "mtp"
MTP_TOKENS = 2
MAXLEN_CANDIDATES = [32768, 16384, 8192]
TP_SIZE = 2

PROXY_SRC_B64 = "IyDilIDilIAgcnVudGltZS9xdW90YS5wee+8iGF1dGhfcHJveHkucHkg44Gr5ZCM5qKx77yJCiIiIkthZ2dsZSDjga7jgq/jgqnjg7zjgr8gQVBJ77yIR1BVL1RQVSDmrovph4/vvInjgpLoqq3jgoDlhbHpgJrlh6bnkIbvvIjnlJ/miJDjgrPjg7zjg4njgbjln4vjgoHovrzjgoDvvInjgIIKCmthZ2dsZSBDTEkg44Gr44GvIHF1b3RhIOOCteODluOCs+ODnuODs+ODieOBjOeEoeOBhOOBjOOAgVdlYiBBUEkg44GvIEJhc2ljIOiqjeiovOOBp+iqreOCgeOCi+OAggrlkbzjgbPlh7rjgZflgbTjga8gYXV0aF9wcm94eS5wee+8iC9xdW90Ye+8ieOBqCBrYWdnbGUtcXVvdGEucHnvvIhDTEkg6KGo56S677yJ44CCCiIiIgppbXBvcnQgYmFzZTY0CmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdXJsbGliLnJlcXVlc3QKCktDX1FVT1RBX1VSTCA9ICJodHRwczovL3d3dy5rYWdnbGUuY29tL2FwaS92MS9rZXJuZWxzL3F1b3RhIgpLQ19DUkVEX1BBVEggPSBvcy5wYXRoLmV4cGFuZHVzZXIoIn4vLmthZ2dsZS9rYWdnbGUuanNvbiIpCgoKZGVmIGtjX2thZ2dsZV9jcmVkZW50aWFscyhjcmVkX3BhdGg9Tm9uZSwgZW52aXJvbj1Ob25lKToKICAgICIiIih1c2VybmFtZSwga2V5KSDjgpLov5TjgZnjgIJrYWdnbGUuanNvbiDihpIg55Kw5aKD5aSJ5pWw44Gu6aCG44CC54Sh44GR44KM44GwIChOb25lLCBOb25lKeOAggoKICAgIOOCq+ODvOODjeODq+OBq+OBryBrYWdnbGUuanNvbiDjgYznhKHjgYTjgYzjgIFLQUdHTEVfVVNFUk5BTUUgLyBLQUdHTEVfS0VZIOOBjAogICAg55Kw5aKD5aSJ5pWw44Gn5YWl44Gj44Gm44GE44KL44CCCiAgICAiIiIKICAgIGNyZWRfcGF0aCA9IEtDX0NSRURfUEFUSCBpZiBjcmVkX3BhdGggaXMgTm9uZSBlbHNlIGNyZWRfcGF0aAogICAgZW52aXJvbiA9IG9zLmVudmlyb24gaWYgZW52aXJvbiBpcyBOb25lIGVsc2UgZW52aXJvbgogICAgaWYgb3MucGF0aC5pc2ZpbGUoY3JlZF9wYXRoKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHdpdGggb3BlbihjcmVkX3BhdGgpIGFzIGY6CiAgICAgICAgICAgICAgICBjZmcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgdXNlciwga2V5ID0gY2ZnLmdldCgidXNlcm5hbWUiKSwgY2ZnLmdldCgia2V5IikKICAgICAgICAgICAgaWYgdXNlciBhbmQga2V5OgogICAgICAgICAgICAgICAgcmV0dXJuIHVzZXIsIGtleQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgIHVzZXIgPSBlbnZpcm9uLmdldCgiS0FHR0xFX1VTRVJOQU1FIikgb3IgZW52aXJvbi5nZXQoIktBR0dMRV9VU0VSIikKICAgIGtleSA9IGVudmlyb24uZ2V0KCJLQUdHTEVfS0VZIikKICAgIHJldHVybiAodXNlciwga2V5KSBpZiB1c2VyIGFuZCBrZXkgZWxzZSAoTm9uZSwgTm9uZSkKCgpkZWYga2NfZmV0Y2hfcXVvdGEodXNlciwga2V5LCB0aW1lb3V0PTQ1LCB1cmw9S0NfUVVPVEFfVVJMLCBvcGVuZXI9Tm9uZSk6CiAgICAiIiLjgq/jgqnjg7zjgr/jga7nlJ8gSlNPTiDjgpLov5TjgZnjgIJIVFRQIC8g5o6l57aa44Ko44Op44O844Gv5L6L5aSW44Gu44G+44G+5oqV44GS44KL44CCIiIiCiAgICB0b2tlbiA9IGJhc2U2NC5iNjRlbmNvZGUoZiJ7dXNlcn06e2tleX0iLmVuY29kZSgpKS5kZWNvZGUoKQogICAgcmVxID0gdXJsbGliLnJlcXVlc3QuUmVxdWVzdCgKICAgICAgICB1cmwsIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJhc2ljICIgKyB0b2tlbn0pCiAgICB3aXRoIChvcGVuZXIgb3IgdXJsbGliLnJlcXVlc3QudXJsb3BlbikocmVxLCB0aW1lb3V0PXRpbWVvdXQpIGFzIHI6CiAgICAgICAgcmV0dXJuIGpzb24ubG9hZHMoci5yZWFkKCkpCgoKZGVmIGtjX3F1b3RhX3N1bW1hcnkoZGF0YSk6CiAgICAiIiJVSSDnlKjjga4gR1BVIOOCr+OCqeODvOOCv+imgee0hOOAgmdwdVF1b3RhIOOBjOeEoeOBhOW/nOetlOOBryBhdmFpbGFibGU9RmFsc2XjgIIiIiIKICAgIGcgPSBkYXRhLmdldCgiZ3B1UXVvdGEiKSBvciB7fQogICAgdXNlZCA9IChnLmdldCgidGltZVVzZWQiKSBvciB7fSkuZ2V0KCJzZWNvbmRzIikKICAgIHRvdGFsID0gKGcuZ2V0KCJ0b3RhbFRpbWVBbGxvd2VkIikgb3Ige30pLmdldCgic2Vjb25kcyIpCiAgICBpZiB1c2VkIGlzIE5vbmUgb3IgdG90YWwgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJhdmFpbGFibGUiOiBGYWxzZSwgInJlYXNvbiI6ICJub19ncHVfcXVvdGEifQogICAgcmVzZXJ2ZWQgPSAoZy5nZXQoInRpbWVSZXNlcnZlZCIpIG9yIHt9KS5nZXQoInNlY29uZHMiLCAwKQogICAgcmV0dXJuIHsKICAgICAgICAiYXZhaWxhYmxlIjogVHJ1ZSwKICAgICAgICAidXNlZFNlY29uZHMiOiB1c2VkLAogICAgICAgICJ0b3RhbFNlY29uZHMiOiB0b3RhbCwKICAgICAgICAicmVzZXJ2ZWRTZWNvbmRzIjogcmVzZXJ2ZWQsCiAgICAgICAgImxlZnRTZWNvbmRzIjogdG90YWwgLSB1c2VkIC0gcmVzZXJ2ZWQsCiAgICAgICAgInJlZnJlc2hUaW1lIjogZGF0YS5nZXQoInF1b3RhUmVmcmVzaFRpbWUiLCAiIiksCiAgICB9CgojIS91c3IvYmluL2VudiBweXRob24zCiIiIkJhc2ljIOiqjeiovOOBpOOBjeODquODkOODvOOCueODl+ODreOCreOCt++8iOaomea6luODqeOCpOODluODqeODquOBruOBv++8ieOAggoKY2xvdWRmbGFyZWQg44GuIHF1aWNrIHR1bm5lbCDjgavjga8gQmFzaWMg6KqN6Ki85qmf6IO944GM54Sh44GE44Gf44KB44CBCk9sbGFtYSDjga7miYvliY3jgavjgZPjgozjgpLnva7jgYTjgaboqo3oqLzjgpLmi4Xjgo/jgZvjgovjgIIKCiAgICBBUElfVVNFUj1vbGxhbWEgQVBJX1BBU1M9c2VjcmV0IHB5dGhvbjMgYXV0aF9wcm94eS5weSBcCiAgICAgICAgLS1wb3J0IDExNDM1IC0tdXBzdHJlYW0gaHR0cDovLzEyNy4wLjAuMToxMTQzNAoK44K544OI44Oq44O844Of44Oz44Kw77yIT2xsYW1hIOOBriB0b2tlbiDpgJDmrKHlh7rlipvvvInjgpLpgJrjgZnjgZ/jgoHjgIEK5LiK5rWB44Gu5b+c562U44GvIGNodW5rZWQgdHJhbnNmZXItZW5jb2Rpbmcg44Gn44Gd44Gu44G+44G+5Lit57aZ44GZ44KL44CCCiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBiYXNlNjQKaW1wb3J0IGhtYWMKaW1wb3J0IGh0dHAuc2VydmVyCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgcGF0aGxpYgppbXBvcnQgc2lnbmFsCmltcG9ydCBzb2NrZXRzZXJ2ZXIKaW1wb3J0IHN5cwppbXBvcnQgdGhyZWFkaW5nCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5wYXJzZQppbXBvcnQgdXJsbGliLnJlcXVlc3QKCiMg5LiK5rWB77yIT2xsYW1h77yJ44G444Gu5b6F44Gh5pmC6ZaT44CC6ZW344GE55Sf5oiQ44Gr6ICQ44GI44KL44KI44GG6ZW344KB44Gr5Y+W44KL44CCClVQU1RSRUFNX1RJTUVPVVQgPSAzNjAwCiMg5Lit57aZ44Gu44OQ44OD44OV44Kh5LiK6ZmQ44CC5pS56KGM77yITkRKU09OIOihjO+8ieOBjOadpeOBquOBhOWgtOWQiOOBruODleOCqeODvOODq+ODkOODg+OCr+OBp+OAgQojIOOBk+OBk+OBq+mBlOOBmeOCi+OBvuOBp+W+heOBpOOBqOOCueODiOODquODvOODoOOBjOa6nOOBvuOCi+OBruOBp+OAgeWwj+OBleOBhOOBu+OBqemAkOasoeaAp+OBr+mrmOOBhOOAggpDSFVOSyA9IDQwOTYKIyBCYXNpYyDoqo3oqLzjgpLlhY3pmaTjgZnjgovjg5HjgrnjgIIKIyAvaGVhbHRoeiDjga/nlo7pgJrnorroqo3jgIEvIOOBqCAvY2hhdCDjga/jg4Hjg6Pjg4Pjg4ggVUnvvIjpnZnnmoTjgaogSFRNTCDjga7jgb/jgIIKIyDoqo3oqLzmg4XloLHjgoLjg6Ljg4fjg6vjgoLlkKvjgb7jgarjgYTjga7jgafjgIHlhazplovjgZfjgabjgoLmvI/jgozjgovjgoLjga7jga/nhKHjgYTvvInjgIIKIyBBUEkg5pys5L2T77yIL2FwaS8qLCAvdjEvKu+8ieOBr+W+k+adpeOBqeOBiuOCiiBCYXNpYyDoqo3oqLzjgYzopoHjgovjgIIKT1BFTl9QQVRIUyA9IHsiL2hlYWx0aHoiLCAiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCIsICIvZmF2aWNvbi5pY28ifQpVSV9QQVRIUyA9IHsiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCJ9CgojIOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiO+8iOS4iua1geOBuOa1geOBleOBquOBhO+8ieOAggojIC9xdW90YSA6IEthZ2dsZSDjga4gR1BVIOOCr+OCqeODvOOCv+aui+mHj++8iFVJIOOBruODmOODg+ODgOOBq+WHuuOBme+8iQojIC9zdG9wICA6IOOBk+OBruS4ree2meOBqOS4iua1geOBruOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCi+OAggojICAgICAgICAgIFBPU1QgKyDjg5jjg4Pjg4AgWC1LYy1TdG9wOiAxIOOBjOW/hemgiO+8iEdFVCDjga7jgrnjgq3jg6Pjg4rjgoTku5bjgrXjgqTjg4jjgYvjgonjga4KIyAgICAgICAgICBDU1JGIOOBp+iQveOBqOOBleOCjOOBquOBhOOCiOOBhuOBq++8ieOAgkFQSV9QQVNTIOioreWumuaZguOBryBCYXNpYyDoqo3oqLzjgoLopoHjgovjgIIKQURNSU5fUEFUSFMgPSB7Ii9xdW90YSIsICIvc3RvcCJ9ClNUT1BfSEVBREVSID0gIlgtS2MtU3RvcCIKCiMga2VlcGFsaXZlIOOCkuW7tumVt+OBmeOCi+OBruOBr+OAjOeUn+aIkOOAjeODquOCr+OCqOOCueODiOOBoOOBkeOAgi9hcGkvcHMg44KEIC92MS9tb2RlbHMg44GuCiMg44Od44O844Oq44Oz44Kw77yI44Op44Oz44OB44Oj44O844Gu6LW35YuV56K66KqN44O744K544Kt44Oj44OK77yJ44GnIEdQVSDjgrvjg4Pjgrfjg6fjg7PjgpLlu7blkb3jgZfjgarjgYTjgIIKR0VORVJBVElPTl9QQVRIUyA9IHsKICAgICIvYXBpL2NoYXQiLAogICAgIi9hcGkvZ2VuZXJhdGUiLAogICAgIi92MS9jaGF0L2NvbXBsZXRpb25zIiwKICAgICIvdjEvY29tcGxldGlvbnMiLAogICAgIi92MS9yZXNwb25zZXMiLAp9CgojIOOCr+OCqeODvOOCv+Wun+ijheOBryBydW50aW1lL3F1b3RhLnB5IOOBjOecn+Wun+a6kOOAguOCq+ODvOODjeODq+OBp+OBr+ODk+ODq+ODgOODvOOBjOWQjOOCveODvOOCueOCkgojIOWFiOmgreOBq+mAo+e1kOOBl+OBpuWfi+OCgei+vOOCgOOBn+OCgeOAgWtjXyog44GM5pyq5a6a576p44Gu44Ot44O844Kr44Or5a6f6KGM44Gg44GRIGltcG9ydCDjgZnjgovjgIIKaWYgImtjX2ZldGNoX3F1b3RhIiBub3QgaW4gZ2xvYmFscygpOgogICAgc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihwYXRobGliLlBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnRzWzJdKSkKICAgIGZyb20gcnVudGltZSBpbXBvcnQgcXVvdGEgYXMgX2tjX3F1b3RhCiAgICBrY19mZXRjaF9xdW90YSA9IF9rY19xdW90YS5rY19mZXRjaF9xdW90YQogICAga2NfcXVvdGFfc3VtbWFyeSA9IF9rY19xdW90YS5rY19xdW90YV9zdW1tYXJ5CiAgICBrY19rYWdnbGVfY3JlZGVudGlhbHMgPSBfa2NfcXVvdGEua2Nfa2FnZ2xlX2NyZWRlbnRpYWxzCgpIT1BfQllfSE9QID0gewogICAgImNvbm5lY3Rpb24iLAogICAgImtlZXAtYWxpdmUiLAogICAgInByb3h5LWF1dGhlbnRpY2F0ZSIsCiAgICAicHJveHktYXV0aG9yaXphdGlvbiIsCiAgICAidGUiLAogICAgInRyYWlsZXJzIiwKICAgICJ0cmFuc2Zlci1lbmNvZGluZyIsCiAgICAidXBncmFkZSIsCiAgICAiZXhwZWN0IiwKfQoKQVJHUyA9IE5vbmUKRVhQRUNURURfQVVUSCA9ICIiCiMg44OB44Oj44OD44OIIFVJIOOBriBIVE1M44CC6LW35YuV5pmC44Gr5LiA5bqm44Gg44GR6Kqt44KA77yI5q+O44Oq44Kv44Ko44K544OI44Gn44OH44Kj44K544Kv44KS5Y+p44GL44Gq44GE77yJ44CCCkNIQVRfQllURVMgPSBiIiIKCiMgLS0tLSDjgrXjg7Pjg5fjg6rjg7PjgrDlvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg44Gu44Go44GN44Gg44GR5pyJ5Yq577yJLS0tLQojIHZMTE0g54mI44Gn5b+F6KaB44CC44Kv44Op44Kk44Ki44Oz44OI77yIb3BlbmNvZGUg562J77yJ44GMIHRlbXBlcmF0dXJlPTAg44KS6YCB44KL44GoCiMg6LKq5qyy5rOV44Gr44Gq44KK44CBMjdCIOe0muOBp+OBr+WQjOOBmOaWh+OBrue5sOOCiui/lOOBl+OBp+W0qeWjiuOBl+OChOOBmeOBhOOAggojIHJlcGV0aXRpb25fcGVuYWx0eSDjgoLmnKrmjIflrprjgaDjgaggMS4w77yIPeeEoeWKue+8ieOBq+OBquOCi+OBn+OCgeS4i+mZkOOCkuWFpeOCjOOCi+OAggojIOaXouWumuOBruODl+ODreOCreOCt++8iE9sbGFtYSDniYjvvInjgafjga/nhKHlirnjgarjga7jgafmjJnli5Xjga/lpInjgo/jgonjgarjgYTjgIIKUlBfRkxPT1IgPSAxLjA1ICAgICAjIHJlcGV0aXRpb25fcGVuYWx0eSDjga7kuIvpmZAKVEVNUF9GTE9PUiA9IDAuNyAgICAjIHRlbXBlcmF0dXJlIOOBruS4i+mZkO+8iDAgPSDosqrmrLLms5XjgpLnpoHmraLvvIkKCgojIOODiOODs+ODjeODqy/jg4Hjg6Pjg4Pjg4jntYznlLHjga7jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgacga2VlcGFsaXZlIOaZgumWk+OCkuODquOCu+ODg+ODiOOAggojIC90bXAva2VlcGFsaXZlLmNvbmYg44GuIFJVTj0xLCBNSU5VVEVTPTYwIOOCkuWGjeabuOOBjei+vOOBv+OAggpfS0VFUEFMSVZFX0NPTkYgPSAiL3RtcC9rZWVwYWxpdmUuY29uZiIKCgpkZWYgX3Jlc2V0X2tlZXBhbGl2ZSgpOgogICAgIiIi44OB44Oj44OD44OI6YCB5L+h44GU44Go44Gr5pmC6ZaT44Oq44K744OD44OI77yIT04g57at5oyB77yJ44CC5LiK5rWB44Ko44Op44O85pmC44Gn44KC5ZG844G244CCIiIiCiAgICB0cnk6CiAgICAgICAgd2l0aCBvcGVuKF9LRUVQQUxJVkVfQ09ORiwgInciKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKCJSVU49MVxuTUlOVVRFUz02MFxuIikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKCmRlZiBfaXNfZ2VuZXJhdGlvbihwYXRoX29ubHkpOgogICAgcmV0dXJuIHBhdGhfb25seS5yc3RyaXAoIi8iKSBpbiBHRU5FUkFUSU9OX1BBVEhTCgoKZGVmIF9udW0odmFsdWUpOgogICAgIiIi5pWw5YCk5YyW44Gn44GN44Gq44GR44KM44GwIE5vbmXvvIhib29sIOOBr+aVsOWApOaJseOBhOOBl+OBquOBhO+8ieOAgiIiIgogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgYm9vbCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHRyeToKICAgICAgICByZXR1cm4gZmxvYXQodmFsdWUpCiAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcmV0dXJuIE5vbmUKCgpkZWYgZW5mb3JjZV9zYW1wbGluZyhib2R5LCBwYXRoX29ubHkpOgogICAgIiIiY2hhdC9jb21wbGV0aW9ucyDjga7jgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjgpLlvLfliLbjgZnjgovvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC77yJ44CCCgogICAgLSByZXBldGl0aW9uX3BlbmFsdHkgPCBSUF9GTE9PUu+8iOacquaMh+WumuWQq+OCgO+8ieKGkiBSUF9GTE9PUgogICAgLSB0ZW1wZXJhdHVyZSA8PSAw77yI5pyq5oyH5a6a5ZCr44KA77yJ4oaSIFRFTVBfRkxPT1IKCiAgICDjg5zjg4fjgqPjgYwgSlNPTiDjgafjgarjgYQv5aOK44KM44Gm44GE44KL5aC05ZCI44Gv57Sg6YCa44GX77yI5LiK5rWB44Gr5Yik5pat44KS5aeU44Gt44KL77yJ44CCCiAgICDmiLvjgorlgKTjga/lt67jgZfmm7/jgYjlvozjga4gYnl0ZXPjgIJDb250ZW50LUxlbmd0aCDjga/lkbzjgbPlh7rjgZflgbTjgafku5jjgZHnm7TjgZnjgIIKICAgICIiIgogICAgaWYgbm90IEFSR1MuZW5mb3JjZV9zYW1wbGluZyBvciBub3QgYm9keToKICAgICAgICByZXR1cm4gYm9keQogICAgIyDjgq/jgqjjg6rmloflrZfliJfvvIg/YXBpLXZlcnNpb249Li4uIOetie+8ieOBjOS7mOOBj+OBqCBlbmRzd2l0aCDjgYzlpJbjgozjgovjgZ/jgoHjgIEKICAgICMg44OR44K544Gg44GR44KS5Y+W44KK5Ye644GX44Gm5q+U6LyD44GZ44KL44CC5Y+W44KK44GT44G844GZ44Go55+v5q2j44GM54Sh5Yq55YyW44GV44KM44KL44CCCiAgICBfcGF0aCA9IHVybGxpYi5wYXJzZS51cmxzcGxpdChwYXRoX29ubHkpLnBhdGgKICAgIGlmIG5vdCBfcGF0aC5yc3RyaXAoIi8iKS5lbmRzd2l0aCgiL2NoYXQvY29tcGxldGlvbnMiKToKICAgICAgICByZXR1cm4gYm9keQogICAgdHJ5OgogICAgICAgIGRhdGEgPSBqc29uLmxvYWRzKGJvZHkpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBib2R5CiAgICBpZiBub3QgaXNpbnN0YW5jZShkYXRhLCBkaWN0KToKICAgICAgICByZXR1cm4gYm9keQogICAgY2hhbmdlZCA9IEZhbHNlCiAgICBycCA9IF9udW0oZGF0YS5nZXQoInJlcGV0aXRpb25fcGVuYWx0eSIpKQogICAgaWYgcnAgaXMgTm9uZSBvciBycCA8IFJQX0ZMT09SOgogICAgICAgIGRhdGFbInJlcGV0aXRpb25fcGVuYWx0eSJdID0gUlBfRkxPT1IKICAgICAgICBjaGFuZ2VkID0gVHJ1ZQogICAgdGVtcCA9IF9udW0oZGF0YS5nZXQoInRlbXBlcmF0dXJlIikpCiAgICBpZiB0ZW1wIGlzIE5vbmUgb3IgdGVtcCA8PSAwOgogICAgICAgIGRhdGFbInRlbXBlcmF0dXJlIl0gPSBURU1QX0ZMT09SCiAgICAgICAgY2hhbmdlZCA9IFRydWUKICAgIGlmIG5vdCBjaGFuZ2VkOgogICAgICAgIHJldHVybiBib2R5CiAgICByZXR1cm4ganNvbi5kdW1wcyhkYXRhLCBlbnN1cmVfYXNjaWk9RmFsc2UpLmVuY29kZSgidXRmLTgiKQoKCmNsYXNzIEhhbmRsZXIoaHR0cC5zZXJ2ZXIuQmFzZUhUVFBSZXF1ZXN0SGFuZGxlcik6CiAgICBwcm90b2NvbF92ZXJzaW9uID0gIkhUVFAvMS4xIgogICAgc2VydmVyX3ZlcnNpb24gPSAiT2xsYW1hQXV0aFByb3h5LzEuMCIKICAgICMg5o6l57aa44GU44Go44Gu44K944Kx44OD44OI44K/44Kk44Og44Ki44Km44OI44CC44Kv44Op44Kk44Ki44Oz44OI44GM54Sh6YCa5L+h44Gu44G+44G+5q6L44GZ44GoCiAgICAjIOODj+ODs+ODieODqeOCueODrOODg+ODieOBjOeEoemZkOOBq+WxheW6p+OCi+OBruOBp+OAgTXliIbjgaflm57lj47jgZnjgovvvIjjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK0KICAgICMg44Gu5LiK5rWB6Kqt44G/5Y+W44KK44GvIHVybGxpYiDlgbTjgr3jgrHjg4Pjg4jjgarjga7jgaflvbHpn7/jgpLlj5fjgZHjgarjgYTvvInjgIIKICAgIHRpbWVvdXQgPSAzMDAKCiAgICAjIC0tLS0g44Ot44Kw44KSIDEg6KGM44Gr5Zyn57iu77yI44OO44O844OI44OW44OD44Kv44Gu44Ot44Kw44KS5rGa44GV44Gq44GE77yJLS0tLQogICAgZGVmIGxvZ19tZXNzYWdlKHNlbGYsIGZtdCwgKmFyZ3MpOgogICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSAlc1xuIiAlIChmbXQgJSBhcmdzKSkKCiAgICBkZWYgX3JlYWRfYm9keShzZWxmKToKICAgICAgICAiIiLjg6rjgq/jgqjjgrnjg4jjg5zjg4fjgqPjgpLoqq3jgoDjgIJjaHVua2VkIOOBp+OCgiBDb250ZW50LUxlbmd0aCDjgafjgoLkuKHlr77lv5zjgIIKCiAgICAgICAgQ29udGVudC1MZW5ndGgg44GX44GL6KaL44Gq44GE44Go44CBSFRUUC8xLjEg44GuIGNodW5rZWQgUE9TVO+8iOOCueODiOODquODvOODn+ODs+OCsOWFpeWKm+OChAogICAgICAgIOODl+ODreOCreOCt+e1jOeUseOBruWkieaPm+OBp+eZuueUn+OBl+OBhuOCi++8ieOBpyoq44Oc44OH44Kj44GU44Go5qyg6JC9KirjgZfjgabkuIrmtYHjgavlsYrjgYvjgarjgYTjgIIKICAgICAgICAiIiIKICAgICAgICB0ZSA9IChzZWxmLmhlYWRlcnMuZ2V0KCJUcmFuc2Zlci1FbmNvZGluZyIpIG9yICIiKS5sb3dlcigpCiAgICAgICAgaWYgImNodW5rZWQiIGluIHRlOgogICAgICAgICAgICBvdXQgPSBieXRlYXJyYXkoKQogICAgICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICAgICAgc2l6ZV9saW5lID0gc2VsZi5yZmlsZS5yZWFkbGluZSgpLnN0cmlwKCkKICAgICAgICAgICAgICAgIGlmIGIiOyIgaW4gc2l6ZV9saW5lOiAgICAgICAgICAjIGNodW5rLWV4dCDjga/mjajjgabjgosKICAgICAgICAgICAgICAgICAgICBzaXplX2xpbmUgPSBzaXplX2xpbmUuc3BsaXQoYiI7IiwgMSlbMF0KICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBzaXplID0gaW50KHNpemVfbGluZSwgMTYpCiAgICAgICAgICAgICAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICAgICAgaWYgc2l6ZSA9PSAwOiAgICAgICAgICAgICAgICAgICMg44OI44Os44O844Op44O86Kqt44G/5o2o44GmCiAgICAgICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICAgICAgdHJhaWxlciA9IHNlbGYucmZpbGUucmVhZGxpbmUoKQogICAgICAgICAgICAgICAgICAgICAgICBpZiB0cmFpbGVyIGluIChiIlxyXG4iLCBiIlxuIiwgYiIiKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgIG91dCArPSBzZWxmLnJmaWxlLnJlYWQoc2l6ZSkKICAgICAgICAgICAgICAgIHNlbGYucmZpbGUucmVhZCgyKSAgICAgICAgICAgICAjIOODgeODo+ODs+OCr+acq+WwvuOBriBDUkxGCiAgICAgICAgICAgIHJldHVybiBieXRlcyhvdXQpIG9yIE5vbmUKCiAgICAgICAgbGVuZ3RoID0gaW50KHNlbGYuaGVhZGVycy5nZXQoIkNvbnRlbnQtTGVuZ3RoIikgb3IgMCkKICAgICAgICByZXR1cm4gc2VsZi5yZmlsZS5yZWFkKGxlbmd0aCkgaWYgbGVuZ3RoIGVsc2UgTm9uZQoKICAgIGRlZiBfZW1pdChzZWxmLCBkYXRhKToKICAgICAgICAiIiJjaHVua2VkIOOBrjHliIbniYfjgpLpgIHjgovjgIIKCiAgICAgICAg5pS56KGM5Y2Y5L2N44Gn5ZG844G244Gu44GnIHNlbmQg5Zue5pWw44Gv44OI44O844Kv44Oz5pWw44Gr5ZCM5q+U5L6L77yIMeODkOOCpOODiOavjuOBq+OBr+mAgeOCieOBquOBhO+8ieOAggogICAgICAgICIiIgogICAgICAgIGlmIG5vdCBkYXRhOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiJXhcclxuIiAlIGxlbihkYXRhKSkKICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJ5dGVzKGRhdGEpKQogICAgICAgIHNlbGYud2ZpbGUud3JpdGUoYiJcclxuIikKICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKCiAgICBkZWYgX3NlbmRfanNvbihzZWxmLCBjb2RlLCBvYmosIGNsb3NlPUZhbHNlLCBleHRyYV9oZWFkZXJzPSgpKToKICAgICAgICAiIiJKU09OIOOCkiAxIOeZuuOBp+i/lOOBme+8iOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiOODu+OCqOODqeODvOW/nOetlOOBruWFsemAmuWHpueQhu+8ieOAgiIiIgogICAgICAgIGJvZHkgPSBvYmogaWYgaXNpbnN0YW5jZShvYmosIGJ5dGVzKSBlbHNlIGpzb24uZHVtcHMob2JqKS5lbmNvZGUoKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGNvZGUpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ2FjaGUtQ29udHJvbCIsICJuby1zdG9yZSIpCiAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gZXh0cmFfaGVhZGVyczoKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb25uZWN0aW9uIiwgImNsb3NlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJvZHkpCiAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQoKICAgIGRlZiBfYXV0aG9yaXplZChzZWxmKToKICAgICAgICAjIEFQSV9QQVNTIOOBjOepuiA9IOiqjeiovOOBquOBl++8iOWFrOmWi++8ieOAglVJIOOCkumWi+OBhOOBn+OBoOOBkeOBp+S9v+OBiOOCi+OCiOOBhuOBq+OBmeOCi+OAggogICAgICAgIGlmIG5vdCBFWFBFQ1RFRF9BVVRIOgogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGlmIHNlbGYucGF0aC5zcGxpdCgiPyIpWzBdIGluIE9QRU5fUEFUSFM6CiAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgIyDluLjmlbDmmYLplpPmr5TovIPvvIjjgr/jgqTjg5/jg7PjgrDmlLvmkoPjga7nt6nlkozvvInjgIJzdHIg54mIIGNvbXBhcmVfZGlnZXN0IOOBrwogICAgICAgICMg6Z2eIEFTQ0lJIOOBpyBUeXBlRXJyb3Ig44Gr44Gq44KL44Gu44GnIGJ5dGVzIOOBp+avlOi8g+OBmeOCi+OAggogICAgICAgICMgT1BUSU9OUyDjga8gX3JlbGF5IOWGkumgreOBp+ODreODvOOCq+ODq+WNs+etlOa4iOOBv++8iOOBk+OBk+OBq+OBr+WxiuOBi+OBquOBhO+8ieOAggogICAgICAgIGdvdCA9IHNlbGYuaGVhZGVycy5nZXQoIkF1dGhvcml6YXRpb24iLCAiIikuZW5jb2RlKCJsYXRpbi0xIikKICAgICAgICByZXR1cm4gaG1hYy5jb21wYXJlX2RpZ2VzdChnb3QsIEVYUEVDVEVEX0FVVEguZW5jb2RlKCJhc2NpaSIpKQoKICAgIGRlZiBfZGVueShzZWxmKToKICAgICAgICAjIDQwMSDjgafmjqXntprjgpLplonjgZjjgovjgIJLZWVwLUFsaXZlIOOBruOBvuOBvuaui+OBmeOBqOOAgeasoeOBruODquOCr+OCqOOCueODiOOBrgogICAgICAgICMg44Oc44OH44Kj6Kqt44G/5Y+W44KK5L2N572u44GM44Ga44KM44GmIEhUVFAg44Oq44Kv44Ko44K544OI44K544Oi44Ks44Oq44Oz44Kw44GM6LW344GN44GG44KL44CCCiAgICAgICAgc2VsZi5fc2VuZF9qc29uKDQwMSwgeyJlcnJvciI6ICJ1bmF1dGhvcml6ZWQifSwgY2xvc2U9VHJ1ZSwKICAgICAgICAgICAgICAgICAgICAgICAgZXh0cmFfaGVhZGVycz1bKCJXV1ctQXV0aGVudGljYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICdCYXNpYyByZWFsbT0iT2xsYW1hIEFQSSInKV0pCgogICAgZGVmIF9zZXJ2ZV91aShzZWxmKToKICAgICAgICAiIiLjg4Hjg6Pjg4Pjg4ggVUkg44KS6L+U44GZ44CC6KqN6Ki85LiN6KaB77yI6Z2Z55qE44OV44Kh44Kk44Or44Gu44G/77yJ44CCCgogICAgICAgIFVJIOOBryAxIOODquOCr+OCqOOCueODiOOBp+e1guOCj+OCi+OBruOBpyAqKuaOpee2muOCkumWieOBmOOCiyoq44CC6KqN6Ki844OR44K544Go6YGV44GECiAgICAgICAg44GT44GT44Gn44Gv44Oc44OH44Kj44KS6Kqt44G+44Gq44GE44Gf44KB44CBa2VlcC1hbGl2ZSDjgpLntq3mjIHjgZnjgovjgagKICAgICAgICBDb250ZW50LUxlbmd0aCDku5jjgY0gUE9TVCDjgafmrKHjga7jg6rjgq/jgqjjgrnjg4jkvY3nva7jgYzjgZrjgozjgosKICAgICAgICDvvIjjg6rjgq/jgqjjgrnjg4jjgrnjg57jgrDjg6rjg7PjgrDvvInjgIIKICAgICAgICAiIiIKICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKSAgICAgICAgICAjIOaDs+WumuWkluOBruODnOODh+OCo+OBr+aNqOOBpuOCi++8iOiqreOBv+aui+OBl+mYsuatou+8iQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICBpZiBub3QgQ0hBVF9CWVRFUzoKICAgICAgICAgICAgYm9keSA9IChiImxsYW1hLWNoYXQuaHRtbCBub3QgZm91bmQuICIKICAgICAgICAgICAgICAgICAgICBiInN0YXJ0IHRoZSBwcm94eSB3aXRoIC0tY2hhdC1odG1sIDxwYXRoPi4iKQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoNTAzKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCAidGV4dC9wbGFpbjsgY2hhcnNldD11dGYtOCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShib2R5KQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjAwKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtVHlwZSIsICJ0ZXh0L2h0bWw7IGNoYXJzZXQ9dXRmLTgiKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihDSEFUX0JZVEVTKSkpCiAgICAgICAgIyBVSSDjga/mr47lm57lkIzjgZjlhoXlrrnjgILjg5fjg63jgq3jgrfmm7TmlrDmmYLjgavjg5bjg6njgqbjgrbjgYzlj6TjgYTniYjjgpLkvb/jgo/jgarjgYTjgojjgYYgbm8tc3RvcmXjgIIKICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDYWNoZS1Db250cm9sIiwgIm5vLXN0b3JlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKENIQVRfQllURVMpCgogICAgZGVmIF9zZXJ2ZV9xdW90YShzZWxmKToKICAgICAgICAiIiJLYWdnbGUg44GuIEdQVSDjgq/jgqnjg7zjgr/mrovph4/jgpLov5TjgZnjgIIKCiAgICAgICAgVUkg44Gu44OY44OD44OA44Gr44CM5q6L44KKIDA4OjU544CN44Go5Ye644GZ44Gf44KB44Gr5L2/44GG44CCCiAgICAgICAg6LOH5qC85oOF5aCx77yIa2FnZ2xlLmpzb24g44G+44Gf44Gv55Kw5aKD5aSJ5pWw77yJ44GM54Sh44GE44CB44G+44Gf44GvIEFQSSDjgYwKICAgICAgICDokL3jgaHjgabjgYTjgovloLTlkIjjga8gYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OBpiBVSSDlgbTjgafpnZ7ooajnpLrjgavjgZnjgovjgIIKICAgICAgICAiIiIKICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZX0KICAgICAgICBpZiBBUkdTLm5vX3F1b3RhOgogICAgICAgICAgICAjIEthZ2dsZSDku6XlpJbjga7jg63jg7zjgqvjg6vvvIh1enUg44Gq44Gp77yJ44Gn44Gv44Kv44Kp44O844K/44Gu5qaC5b+144GM54Sh44GE44CCCiAgICAgICAgICAgICMgYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OAgVVJIOWBtOOBr+aui+OCiuaZgumWk+OCkumdnuihqOekuuOBq+OBmeOCi+OAggogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7ImF2YWlsYWJsZSI6IEZhbHNlLCAicmVhc29uIjogIm5vX3F1b3RhIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHVzZXIsIGtleSA9IGtjX2thZ2dsZV9jcmVkZW50aWFscygpCiAgICAgICAgICAgIGlmIHVzZXIgYW5kIGtleToKICAgICAgICAgICAgICAgIHBheWxvYWQgPSBrY19xdW90YV9zdW1tYXJ5KAogICAgICAgICAgICAgICAgICAgIGtjX2ZldGNoX3F1b3RhKHVzZXIsIGtleSwgdGltZW91dD0xNSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZSwgImVycm9yIjogdHlwZShlKS5fX25hbWVfX30KCiAgICAgICAgIyDmrovph4/jga/liLvjgIXlpInjgo/jgovjgIJfc2VuZF9qc29uIOOBryBuby1zdG9yZSDjgpLku5jjgZHjgovjgIIKICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCBwYXlsb2FkKQoKICAgIGRlZiBfc2VydmVfc3RvcChzZWxmKToKICAgICAgICAiIiLjgZPjga7kuK3ntpnjgajkuIrmtYHjga7jgrvjg4Pjgrfjg6fjg7PjgpLmraLjgoHjgovjgIIKCiAgICAgICAg5ZG844Gz5Ye644GX5p2h5Lu277yIUE9TVCArIFgtS2MtU3RvcCArIOiqjeiovO+8ieOBryBfc3RvcF9hbGxvd2VkKCkg44Gn56K66KqN5riI44G/44CCCgogICAgICAgIOODluODqeOCpuOCtuOBruWBnOatouODnOOCv+ODs+OBi+OCieWPqeOBj+OAguS4ree2meOBjOiQveOBoeOCi+OBruOBp+W/nOetlOOCkui/lOOBl+OBpuOBi+OCiQogICAgICAgIOODl+ODreOCu+OCueOCkue1guS6huOBleOBm+OCi++8iOWFiOOBq+iQveOBqOOBmeOBqOODluODqeOCpuOCtuWBtOOBjOmAmuS/oeOCqOODqeODvOOBq+OBquOCi++8ieOAggoKICAgICAgICBLYWdnbGUg5LiK44Gn44GvIGtlZXBhbGl2ZSDjg6vjg7zjg5fjgoLmraLjgoHjgovjgZ/jgoHjgIEvdG1wL2tlZXBhbGl2ZS5jb25mIOOBqwogICAgICAgIFJVTj0wIOOCkuabuOOBjei+vOOCgOOAguOBk+OCjOOBq+OCiOOCiiBLYWdnbGUg44Kr44O844ON44Or44K744OD44K344On44Oz44GM57WC5LqG44GX44CBCiAgICAgICAgR1BVIOOCr+OCqeODvOOCv+OCkumWi+aUvuOBp+OBjeOCi+OAggoKICAgICAgICAtLXN0b3AtcGlkIOOBjOaMh+WumuOBleOCjOOBpuOBhOOCjOOBsOOAgeOBneOBriBQSUTvvIjjg6njg7Pjg4Hjg6Pjg7zvvInjgavjgoIKICAgICAgICBTSUdURVJNIOOCkumAgeOCi+OAgnV6dSDjga8gcnVuLXV6dS1zZXJ2ZXIuY29tbWFuZCDjgYzlrZDjg5fjg63jgrvjgrnjgpIKICAgICAgICDmirHjgYjjgabjgYTjgovjga7jgafjgIHopqrjgpLokL3jgajjgZvjgbAgdHJhcCBjbGVhbnVwIOOBp+WFqOmDqOeJh+S7mOOBj+OAggogICAgICAgICIiIgogICAgICAgICMgS2FnZ2xlIOeJiCBrZWVwYWxpdmUg44Or44O844OX44Gr44KC5YGc5q2i44KS5Lyd44GI44KLCiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oIi90bXAva2VlcGFsaXZlLmNvbmYiLCAidyIpIGFzIGY6CiAgICAgICAgICAgICAgICBmLndyaXRlKCJSVU49MFxuIikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgICAgIHNlbGYuX3NlbmRfanNvbigyMDAsIHsic3RvcHBlZCI6IFRydWUsICJzY29wZSI6ICJyZWxheSJ9KQoKICAgICAgICBkZWYgX2RpZSgpOgogICAgICAgICAgICB0aW1lLnNsZWVwKDAuNSkKICAgICAgICAgICAgIyDlhYjjgavjg6njg7Pjg4Hjg6Pjg7zjgpLokL3jgajjgZnvvIjlrZDjg5fjg63jgrvjgrnjgaggVUkg44KC5LiA57eS44Gr5raI44GI44KL77yJCiAgICAgICAgICAgIGlmIEFSR1Muc3RvcF9waWQ6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgb3Mua2lsbChBUkdTLnN0b3BfcGlkLCBzaWduYWwuU0lHVEVSTSkKICAgICAgICAgICAgICAgIGV4Y2VwdCAoUHJvY2Vzc0xvb2t1cEVycm9yLCBQZXJtaXNzaW9uRXJyb3IpOgogICAgICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgICAgIHRpbWUuc2xlZXAoMC41KQogICAgICAgICAgICBvcy5fZXhpdCgwKQoKICAgICAgICB0aHJlYWRpbmcuVGhyZWFkKHRhcmdldD1fZGllLCBkYWVtb249VHJ1ZSkuc3RhcnQoKQoKICAgIGRlZiBfcmVsYXkoc2VsZik6CiAgICAgICAgIyBDT1JTIOODl+ODquODleODqeOCpOODiOOBr+iqjeiovOS4jeimgeODu+ODreODvOOCq+ODq+WNs+etlOOAguS4iua1geOBuOa1geOBleOBquOBhAogICAgICAgICMg77yI57Sg6YCa44GX44GZ44KL44Go5pyq6KqN6Ki844Oq44Kv44Ko44K544OI44GM5LiK5rWB44Gr5bGK44GN44CB44Oc44OH44Kj5b6F44Gh44GnCiAgICAgICAgIyAg44OP44Oz44OJ44Op44K544Os44OD44OJ44KC5Y2g44KB44KJ44KM44KL77yJ44CCCiAgICAgICAgaWYgc2VsZi5jb21tYW5kID09ICJPUFRJT05TIjoKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDIwNCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgIyAvaGVhbHRoeiDjga/jg63jg7zjgqvjg6vjgafljbPnrZTjgZnjgovvvIjkuIrmtYHjga7mnInnhKHjgavplqLkv4LjgarjgY/nlJ/lrZjnorroqo3jgafjgY3jgovvvIkKICAgICAgICBpZiBzZWxmLnBhdGguc3BsaXQoIj8iKVswXSA9PSAiL2hlYWx0aHoiOgogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7InN0YXR1cyI6ICJvayJ9KQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgcGF0aF9vbmx5ID0gc2VsZi5wYXRoLnNwbGl0KCI/IilbMF0KICAgICAgICBpZiBwYXRoX29ubHkgaW4gVUlfUEFUSFM6CiAgICAgICAgICAgIHNlbGYuX3NlcnZlX3VpKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgICMgLS0tLSDnrqHnkIbjgqjjg7Pjg4njg53jgqTjg7Pjg4jvvIjjg63jg7zjgqvjg6vlsILnlKjvvIktLS0tCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvcXVvdGEiOgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9xdW90YSgpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGlmIHBhdGhfb25seSA9PSAiL3N0b3AiOgogICAgICAgICAgICAjIOiQveOBqOOBm+OCi+OBruOBryBVSSDjga7lgZzmraLjg5zjgr/jg7PvvIhQT1NUICsgWC1LYy1TdG9w77yJ44Gg44GR44CCCiAgICAgICAgICAgICMg5Lul5YmN44Gv6KqN6Ki844KI44KK5YmN44Gr5Yem55CG44GX44Gm44GE44Gf44Gf44KB44CBQVBJX1BBU1Mg44KS6Kit5a6a44GX44Gm44GE44Gm44KCCiAgICAgICAgICAgICMg6Kqw44Gn44KCIEdFVCAvc3RvcCDjgacgS2FnZ2xlIOOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCieOCjOOBn+OAggogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIlBPU1QiIG9yIHNlbGYuaGVhZGVycy5nZXQoU1RPUF9IRUFERVIpICE9ICIxIjoKICAgICAgICAgICAgICAgIHNlbGYuX3NlbmRfanNvbig0MDUsIHsiZXJyb3IiOiAiUE9TVCB3aXRoICVzOiAxIHJlcXVpcmVkIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICUgU1RPUF9IRUFERVJ9LCBjbG9zZT1UcnVlKQogICAgICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgICAgIGlmIG5vdCBzZWxmLl9hdXRob3JpemVkKCk6CiAgICAgICAgICAgICAgICBzZWxmLl9kZW55KCkKICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9zdG9wKCkKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvZmF2aWNvbi5pY28iOgogICAgICAgICAgICAjIOODluODqeOCpuOCtuOBjOWLneaJi+OBq+WPluOCiuOBq+adpeOCi+OAgjQwNCDjgpLov5TjgZnjgajjg63jgrDjgYzln4vjgb7jgovjgIIKICAgICAgICAgICAgc2VsZi5jbG9zZV9jb25uZWN0aW9uID0gVHJ1ZQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjA0KQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsICIwIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICBpZiBub3Qgc2VsZi5fYXV0aG9yaXplZCgpOgogICAgICAgICAgICAjIDQwMSDjgpLov5TjgZnliY3jgavjg5zjg4fjgqPjgpLpo5/jgaPjgabjgYrjgY/jgIJQT1NUIOOBruODnOODh+OCo+OCkuiqreOBvuOBmuOBqwogICAgICAgICAgICAjIOaOpee2muOCkuWIh+OCi+OBqOOCr+ODqeOCpOOCouODs+ODiOWBtOOBjCBSU1Qg44KS5Y+X44GR44Gm44CMNDAxIOOCiOOCiuWJjeOBqwogICAgICAgICAgICAjIOaOpee2muOBjOWIh+OCjOOBn+OAjeOBq+imi+OBiOOCi++8iGN1cmwg44GvIFJTVCDjgpLjgqjjg6njg7zooajnpLrvvInjgIIKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5fcmVhZF9ib2R5KCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgc2VsZi5fZGVueSgpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGJvZHkgPSBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICMg5LiN5q2j44GqIENvbnRlbnQtTGVuZ3RoIC8g5aOK44KM44GfIGNodW5rZWQg44Gv5LiK5rWB44Gr5rWB44GV44Gq44GE44CCCiAgICAgICAgICAgICMg5pyq5o2V5o2J44Gg44Go44K544K/44OD44Kv44OA44Oz44OX44GuNTAw44Gr44Gq44KL77yI6KqN6Ki85oiQ5Yqf5YG044Gg44GR6Z2e5a++56ew44Gr44Gq44KJ44Gq44GE44KI44GG77yJCiAgICAgICAgICAgIHNlbGYuY2xvc2VfY29ubmVjdGlvbiA9IFRydWUKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDQwMCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbm5lY3Rpb24iLCAiY2xvc2UiKQogICAgICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlmIF9pc19nZW5lcmF0aW9uKHBhdGhfb25seSk6CiAgICAgICAgICAgIF9yZXNldF9rZWVwYWxpdmUoKSAgIyDnlJ/miJDjg6rjgq/jgqjjgrnjg4jvvJ3jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgYLjgoog4oaSIDYw5YiG44Oq44K744OD44OICiAgICAgICAgIyDjgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjga7lvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC44Gu44G/44CC5pei5a6a44Gv57Sg6YCa44GX77yJCiAgICAgICAgYm9keSA9IGVuZm9yY2Vfc2FtcGxpbmcoYm9keSwgcGF0aF9vbmx5KQogICAgICAgIHVybCA9IEFSR1MudXBzdHJlYW0ucnN0cmlwKCIvIikgKyBzZWxmLnBhdGgKICAgICAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBtZXRob2Q9c2VsZi5jb21tYW5kKQogICAgICAgIGZvciBrZXksIHZhbHVlIGluIHNlbGYuaGVhZGVycy5pdGVtcygpOgogICAgICAgICAgICAjIGNvbnRlbnQtbGVuZ3RoIOOBryBib2R5IOWGjeani+evieW+jOOBruWun+mVt+OBp+mAgeOCiuebtOOBmeOBn+OCgeOBk+OBk+OBp+aNqOOBpuOCiwogICAgICAgICAgICAjIO+8iOOCr+ODqeOCpOOCouODs+ODiOWApOOBruOBvuOBvui7oumAgeOBmeOCi+OBqCBjaHVua2VkICsgQ0wg5L2155So44Gn44OH44K344Oz44Kv44GZ44KL77yJCiAgICAgICAgICAgIGlmIGtleS5sb3dlcigpIGluIEhPUF9CWV9IT1AgfCB7Imhvc3QiLCAiYXV0aG9yaXphdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImNvbnRlbnQtbGVuZ3RoIn06CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZXEuYWRkX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGJvZHkgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJlcS5hZGRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsIHN0cihsZW4oYm9keSkpKQoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3AgPSB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJlcSwgdGltZW91dD1VUFNUUkVBTV9USU1FT1VUKQogICAgICAgIGV4Y2VwdCB1cmxsaWIuZXJyb3IuSFRUUEVycm9yIGFzIGV4YzoKICAgICAgICAgICAgIyDkuIrmtYHjgYwgNHh4LzV4eCDjgpLov5TjgZfjgZ/loLTlkIjjgoLmnKzmlofjgpLjgZ3jga7jgb7jgb7ov5TjgZkKICAgICAgICAgICAgcGF5bG9hZCA9IGV4Yy5yZWFkKCkKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGV4Yy5jb2RlKQogICAgICAgICAgICBjdHlwZSA9IGV4Yy5oZWFkZXJzLmdldCgiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCBjdHlwZSkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCBzdHIobGVuKHBheWxvYWQpKSkKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIHNlbGYuY29tbWFuZCAhPSAiSEVBRCI6CiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKHBheWxvYWQpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246ICAjIOaOpee2muWkseaVl+OBquOBqQogICAgICAgICAgICAjIOWGhemDqOS+i+WkluOBruips+e0sOOBr+OCr+ODqeOCpOOCouODs+ODiOOBq+a4oeOBleOBquOBhO+8iOaDheWgsemcsuWHuumYsuatou+8ieOAggogICAgICAgICAgICAjIOWOn+WboOOBr+OCteODvOODkOWBtCBzdGRlcnIg44Gr44Gg44GR5pu444GP44CCCiAgICAgICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSB1cHN0cmVhbSBlcnJvcjogJXNcbiIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAlIChzeXMuZXhjX2luZm8oKVsxXSwpKQogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oNTAyLCB7ImVycm9yIjogInVwc3RyZWFtIHVucmVhY2hhYmxlIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB3aXRoIHJlc3A6CiAgICAgICAgICAgIHNlbGYuc2VuZF9yZXNwb25zZShyZXNwLnN0YXR1cykKICAgICAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gcmVzcC5oZWFkZXJzLml0ZW1zKCk6CiAgICAgICAgICAgICAgICBpZiBrZXkubG93ZXIoKSBpbiBIT1BfQllfSE9QIG9yIGtleS5sb3dlcigpID09ICJjb250ZW50LWxlbmd0aCI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoa2V5LCB2YWx1ZSkKICAgICAgICAgICAgIyBIRUFEIOOBr+acrOaWh+OBquOBl+OAgmNodW5rZWQg44KS5a6j6KiA44GZ44KL44Go44Kv44Op44Kk44Ki44Oz44OI44GM57WC56uv44OB44Oj44Oz44Kv44KSCiAgICAgICAgICAgICMg5b6F44Gj44Gm44OP44Oz44Kw44GZ44KL77yIQ29udGVudC1MZW5ndGgg44KC5rih44Gb44Gq44GE44Gu44Gn5a6j6KiA44GX44Gq44GE77yJ44CCCiAgICAgICAgICAgICMgMjA0LzMwNCDjga/mnKzmlofjgpLmjIHjgabjgarjgYTjgIJjaHVua2VkIOOCkuWuo+iogOOBmeOCi+OBqOWOs+agvOOBqgogICAgICAgICAgICAjIOOCr+ODqeOCpOOCouODs+ODiOOBjCBwcm90b2NvbCBlcnJvciDjgavjgZnjgovjgIIKICAgICAgICAgICAgbm9fYm9keSA9IHJlc3Auc3RhdHVzIGluICgyMDQsIDMwNCkgb3Igc2VsZi5jb21tYW5kID09ICJIRUFEIgogICAgICAgICAgICBpZiBub3Qgbm9fYm9keToKICAgICAgICAgICAgICAgICMg6ZW344GV5pyq55+l44Gu44K544OI44Oq44O844Og44Go44GX44Gm5Lit57aZ44GZ44KLCiAgICAgICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJUcmFuc2Zlci1FbmNvZGluZyIsICJjaHVua2VkIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIG5vX2JvZHk6CiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgIyByZXNwLnJlYWQoTikg44GvIEhUVFBSZXNwb25zZS5fcmVhZGludG9fY2h1bmtlZCDjga7mgKfos6rjgacKICAgICAgICAgICAgICAgICMgKipOIOODkOOCpOODiOaPg+OBhuOBvuOBp+iqreOBv+e2muOBkeOCiyoq44Gf44KB44CB44K544OI44Oq44O844Og44GM5YWo6Z2i44Gr5rqc44G+44KLCiAgICAgICAgICAgICAgICAjIO+8iE9sbGFtYSDjga4gTkRKU09OIOOBjOWFqOODiOODvOOCr+ODs+WIhuiyr+OBvuOBo+OBpuOBi+OCieS4gOaWieOBq+WxiuOBj++8ieOAggogICAgICAgICAgICAgICAgIyByZWFkbGluZShDSFVOSykg44Gv5pS56KGM44GLIENIVU5LIOODkOOCpOODiOOBp+i/lOOCi+OBruOBp+OAgemAkOasoeaAp+OCkgogICAgICAgICAgICAgICAgIyDkv53jgaPjgZ/jgb7jgb4gMSDjg5DjgqTjg4jjgZrjgaTjga4gcmVhZCDjg6vjg7zjg5fjgojjgoogQ1BVIOOCkumjn+OCj+OBquOBhOOAggogICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICBsaW5lID0gcmVzcC5yZWFkbGluZShDSFVOSykKICAgICAgICAgICAgICAgICAgICBpZiBub3QgbGluZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgICAgICBzZWxmLl9lbWl0KGxpbmUpCiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiMFxyXG5cclxuIikKICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICBleGNlcHQgKEJyb2tlblBpcGVFcnJvciwgQ29ubmVjdGlvblJlc2V0RXJyb3IpOgogICAgICAgICAgICAgICAgIyDjgq/jg6njgqTjgqLjg7Pjg4jjgYzliIfjgaPjgZ/vvIhDdHJsLUMg44Gq44Gp77yJ44CC5q2j5bi457O744CCCiAgICAgICAgICAgICAgICBwYXNzCgogICAgZG9fR0VUID0gX3JlbGF5CiAgICBkb19QT1NUID0gX3JlbGF5CiAgICBkb19QVVQgPSBfcmVsYXkKICAgIGRvX1BBVENIID0gX3JlbGF5CiAgICBkb19ERUxFVEUgPSBfcmVsYXkKICAgIGRvX0hFQUQgPSBfcmVsYXkKICAgIGRvX09QVElPTlMgPSBfcmVsYXkKCgpjbGFzcyBTZXJ2ZXIoc29ja2V0c2VydmVyLlRocmVhZGluZ01peEluLCBodHRwLnNlcnZlci5IVFRQU2VydmVyKToKICAgIGRhZW1vbl90aHJlYWRzID0gVHJ1ZQogICAgYWxsb3dfcmV1c2VfYWRkcmVzcyA9IFRydWUKCgpkZWYgbWFpbigpOgogICAgZ2xvYmFsIEFSR1MsIEVYUEVDVEVEX0FVVEgKCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXBvcnQiLCB0eXBlPWludCwgZGVmYXVsdD0xMTQzNSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0taG9zdCIsIGRlZmF1bHQ9IjEyNy4wLjAuMSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXVwc3RyZWFtIiwgZGVmYXVsdD0iaHR0cDovLzEyNy4wLjAuMToxMTQzNCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoYXQtaHRtbCIsIGRlZmF1bHQ9b3MuZW52aXJvbi5nZXQoIkNIQVRfSFRNTCIsICIiKSwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0i44OB44Oj44OD44OIIFVJIOOBriBIVE1M77yILyDjgafphY3kv6HjgZnjgovvvIkiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zdG9wLXBpZCIsIHR5cGU9aW50LCBkZWZhdWx0PTAsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9Ii9zdG9wIOaZguOBqyBTSUdURVJNIOOCkumAgeOCiyBQSUTjgIIiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInV6dSDjga7jgojjgYbjgavjg6njg7Pjg4Hjg6Pjg7zjgYzlrZDjg5fjg63jgrvjgrnjgpLmirHjgYjjgovloLTlkIjjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIuODqeODs+ODgeODo+ODvOOBriBQSUQg44KS5rih44GZ44Go5b6M54mH5LuY44GR44GU44Go5q2i44G+44KLIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm8tcXVvdGEiLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSIvcXVvdGEg44KS54Sh5Yq55YyW44GZ44KL77yIS2FnZ2xlIOS7peWkluOBruODreODvOOCq+ODq+eUqO+8iSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVuZm9yY2Utc2FtcGxpbmciLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJjaGF0L2NvbXBsZXRpb25zIOOBriB0ZW1wZXJhdHVyZTw9MCDjgpIgJS4xZiDjgavjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInJlcGV0aXRpb25fcGVuYWx0eSDjgpIgJS4yZiDku6XkuIrjgavnn6/mraPjgZnjgosiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJSAoVEVNUF9GTE9PUiwgUlBfRkxPT1IpKQogICAgQVJHUyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKCiAgICB1c2VyID0gb3MuZW52aXJvbi5nZXQoIkFQSV9VU0VSIikgb3IgIm9sbGFtYSIKICAgIHBhc3N3b3JkID0gb3MuZW52aXJvbi5nZXQoIkFQSV9QQVNTIikgb3IgIiIKICAgIGlmIHBhc3N3b3JkOgogICAgICAgIHRva2VuID0gYmFzZTY0LmI2NGVuY29kZShmInt1c2VyfTp7cGFzc3dvcmR9Ii5lbmNvZGUoKSkuZGVjb2RlKCkKICAgICAgICBFWFBFQ1RFRF9BVVRIID0gIkJhc2ljICIgKyB0b2tlbgogICAgZWxzZToKICAgICAgICAjIOaXouWumuOBr+iqjeiovOOBquOBl+OAglVSTCDjgpLnn6XjgaPjgabjgYTjgozjgbDoqrDjgafjgoLkvb/jgYjjgovngrnjgaDjgZHms6jmhI/jgIIKICAgICAgICBwcmludCgiYXV0aDogT0ZGIChBUElfUEFTUyDmnKroqK3lrpopIiwgZmx1c2g9VHJ1ZSkKCiAgICBnbG9iYWwgQ0hBVF9CWVRFUwogICAgaWYgQVJHUy5jaGF0X2h0bWw6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oQVJHUy5jaGF0X2h0bWwsICJyYiIpIGFzIGY6CiAgICAgICAgICAgICAgICBDSEFUX0JZVEVTID0gZi5yZWFkKCkKICAgICAgICBleGNlcHQgT1NFcnJvciBhcyBleGM6CiAgICAgICAgICAgICMgVUkg54Sh44GX44Gn44KCIEFQSSDjga/li5XjgY/jgILokL3jgajjgZXjgZrorablkYrjgaDjgZHlh7rjgZnjgIIKICAgICAgICAgICAgc3lzLnN0ZGVyci53cml0ZSgiY2hhdCB1aSBub3QgbG9hZGVkOiAlc1xuIiAlIChleGMsKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiY2hhdCB1aTogJXMgKCVkIEIpIiAlIChBUkdTLmNoYXRfaHRtbCwgbGVuKENIQVRfQllURVMpKSwKICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKCiAgICBodHRwZCA9IFNlcnZlcigoQVJHUy5ob3N0LCBBUkdTLnBvcnQpLCBIYW5kbGVyKQogICAgcHJpbnQoZiJhdXRoIHByb3h5OiB7QVJHUy5ob3N0fTp7QVJHUy5wb3J0fSAtPiB7QVJHUy51cHN0cmVhbX0gIgogICAgICAgICAgZiIodXNlcj17dXNlcn0pIiwgZmx1c2g9VHJ1ZSkKICAgIHRyeToKICAgICAgICBodHRwZC5zZXJ2ZV9mb3JldmVyKCkKICAgIGV4Y2VwdCBLZXlib2FyZEludGVycnVwdDoKICAgICAgICBwYXNzCiAgICBmaW5hbGx5OgogICAgICAgIGh0dHBkLnNlcnZlcl9jbG9zZSgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo="
CHAT_SRC_B64 = "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImphIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0idXRmLTgiIC8+CiAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIiAvPgogIDx0aXRsZT7jg4Hjg6Pjg4Pjg4hVSTwvdGl0bGU+CiAgPHN0eWxlPgogICAgOnJvb3QgeyAtLWJnOiMwYjBjMTA7IC0tcGFuZWw6IzExMTMxYTsgLS1pbms6I2U5ZWVmMjsgLS1tdXRlZDojOWZiMGJmOyAtLWFjYzojNmFhMWZmOyAtLWxpbmU6IzFlMjIzMDsgfQogICAgKntib3gtc2l6aW5nOmJvcmRlci1ib3h9CiAgICBodG1sLGJvZHl7aGVpZ2h0OjEwMCV9CiAgICBib2R5e21hcmdpbjowOyBmb250OjE0cHgvMS40NSBzeXN0ZW0tdWksIC1hcHBsZS1zeXN0ZW0sIFNlZ29lIFVJLCBSb2JvdG8sIFVidW50dSwgQ2FudGFyZWxsLCBOb3RvIFNhbnMsIEhlbHZldGljYSwgQXJpYWw7IGJhY2tncm91bmQ6dmFyKC0tYmcpOyBjb2xvcjp2YXIoLS1pbmspfQogICAgLndyYXB7ZGlzcGxheTpncmlkOyBncmlkLXRlbXBsYXRlLXJvd3M6YXV0byAxZnIgYXV0bzsgaGVpZ2h0OjEwMCV9CiAgICBoZWFkZXIsIGZvb3RlcntiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTt9CiAgICBoZWFkZXJ7ZGlzcGxheTpmbGV4OyBnYXA6Ljc1cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXA7IHBhZGRpbmc6LjZyZW0gLjhyZW07IHBvc2l0aW9uOnN0aWNreTsgdG9wOjA7IHotaW5kZXg6MTA7IGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWxpbmUpfQogICAgaGVhZGVyIC50aXRsZXtmb250LXdlaWdodDo2MDA7IGxldHRlci1zcGFjaW5nOi4ycHh9CgogICAgaGVhZGVyIC5icmFuZHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4xNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucHJvbW97Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLWFjYyk7IHRleHQtZGVjb3JhdGlvbjpub25lOyBvcGFjaXR5Oi45fQogICAgaGVhZGVyIC5wcm9tbzpob3ZlcntvcGFjaXR5OjE7IHRleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmV9CgogICAgLyogSW5mbyBkaWFsb2cgKi8KICAgIGRpYWxvZy5pbmZvLWRpYWxvZ3sKICAgICAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYm9yZGVyLXJhZGl1czoxNHB4OwogICAgICBwYWRkaW5nOjA7CiAgICAgIGJhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6bWluKDU2MHB4LCBjYWxjKDEwMHZ3IC0gMjRweCkpOwogICAgICB3aWR0aDogNTYwcHg7CiAgICB9CiAgICBkaWFsb2cuaW5mby1kaWFsb2c6OmJhY2tkcm9we2JhY2tncm91bmQ6cmdiYSgwLDAsMCwuNTUpfQogICAgLmluZm8tY2FyZHttYXJnaW46MDsgcGFkZGluZzouOXJlbSAxcmVtOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi43NXJlbX0KICAgIC5pbmZvLWhlYWR7ZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGp1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOyBnYXA6Ljc1cmVtOyBib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZy1ib3R0b206LjZyZW19CiAgICAuaW5mby10aXRsZXtmb250LXdlaWdodDo2MDB9CiAgICAuaW5mby1ib2R5IGF7Y29sb3I6dmFyKC0tYWNjKTsgdGV4dC1kZWNvcmF0aW9uOm5vbmV9CiAgICAuaW5mby1ib2R5IGE6aG92ZXJ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZX0KICAgIC5pbmZvLWFjdGlvbnN7ZGlzcGxheTpmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7IGdhcDouNXJlbX0KICAgIGhlYWRlciAuc3BhY2Vye2ZsZXg6MX0KICAgIGhlYWRlciBpbnB1dHtiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjQ1cmVtIC41cmVtOyBib3JkZXItcmFkaXVzOi41cmVtOyBvdXRsaW5lOm5vbmU7IHdpZHRoOjIyY2g7IGhlaWdodDozMnB4fQogICAgaGVhZGVyIGlucHV0OmZvY3Vze2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIHNlbGVjdHsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgICBwYWRkaW5nOi40NXJlbSAuNXJlbTsKICAgICAgYm9yZGVyLXJhZGl1czouNXJlbTsKICAgICAgb3V0bGluZTpub25lOwogICAgICB3aWR0aDoyMmNoOwogICAgICBoZWlnaHQ6MzJweDsKICAgICAgYXBwZWFyYW5jZTpub25lOwogICAgfQogICAgaGVhZGVyIHNlbGVjdDpmb2N1c3tib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIGhlYWRlciAuc21hbGx7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKX0KICAgIGhlYWRlciBsYWJlbC5zbWFsbHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4yNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucm93e2Rpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXB9CgogICAgLyogU2Nyb2xsLVZpZXdwb3J0IChzbW9vdGggbnVyIGbDvHIgVXNlci1TY3JvbGwsIHByb2dyYW1tYXRpc2NoIGVyendpbmdlbiB3aXIgJ2F1dG8nKSAqLwogICAgbWFpbiNzY3JvbGxlcntvdmVyZmxvdzphdXRvOyBwYWRkaW5nOjFyZW0gMXJlbSAxLjVyZW07IHNjcm9sbC1iZWhhdmlvcjpzbW9vdGh9CiAgICAubG9ne21heC13aWR0aDo5MDBweDsgbWFyZ2luOjFyZW0gYXV0byAwOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOjFyZW19CiAgICAubXNne2Rpc3BsYXk6ZmxleDsgZ2FwOi43NXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIC5hdmF0YXJ7d2lkdGg6MjhweDsgaGVpZ2h0OjI4cHg7IGRpc3BsYXk6Z3JpZDsgcGxhY2UtaXRlbXM6Y2VudGVyOyBib3JkZXItcmFkaXVzOjhweDsgZmxleDowIDAgMjhweH0KICAgIC51c2VyIC5hdmF0YXJ7YmFja2dyb3VuZDojMjAzMTRhfQogICAgLmJvdCAuYXZhdGFye2JhY2tncm91bmQ6IzFiMmQyMH0KICAgIC5zeXMgLmF2YXRhcntiYWNrZ3JvdW5kOiMyYTIyM2J9CiAgICAKICAgIC8qIFdyYXBwZXIgZsO8ciBkZW4gTmFjaHJpY2h0ZW5pbmhhbHQgKi8KICAgIC5tZXNzYWdlLWNvbnRlbnQtd3JhcHBlciB7CiAgICAgICAgZGlzcGxheTpmbGV4OwogICAgICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICAgICAgICBnYXA6IDAuNXJlbTsgLyogQWJzdGFuZCB6d2lzY2hlbiBEZW5rcHJvemVzcyB1bmQgQW50d29ydCAqLwogICAgICAgIHdpZHRoOiAxMDAlOwogICAgfQoKICAgIC5idWJibGV7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6Ljc1cmVtIC45cmVtOyBib3JkZXItcmFkaXVzOjEycHg7IG1heC13aWR0aDoxMDAlOyBvdmVyZmxvdy13cmFwOmFueXdoZXJlfQogICAgLm1ldGF7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKTsgbWFyZ2luLWJvdHRvbTouMjVyZW19CgogICAgZm9vdGVye2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi42cmVtIC44cmVtfQogICAgLmJhcnttYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW19CiAgICB0ZXh0YXJlYXt3aWR0aDoxMDAlOyByZXNpemU6dmVydGljYWw7IG1pbi1oZWlnaHQ6NjBweDsgbWF4LWhlaWdodDo0MHZoOyBiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjdyZW0gLjc1cmVtOyBib3JkZXItcmFkaXVzOi43NXJlbTsgb3V0bGluZTpub25lfQogICAgdGV4dGFyZWE6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CiAgICBidXR0b257YmFja2dyb3VuZDojMTAxODI3OyBjb2xvcjp2YXIoLS1pbmspOyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi41NXJlbSAuODVyZW07IGJvcmRlci1yYWRpdXM6LjZyZW07IGN1cnNvcjpwb2ludGVyfQogICAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIGJ1dHRvbntoZWlnaHQ6MzJweDsgZGlzcGxheTppbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6Y2VudGVyfQogICAgLnNlY29uZGFyeXtiYWNrZ3JvdW5kOiMwYzBmMTV9CiAgICAubXV0ZWR7Y29sb3I6dmFyKC0tbXV0ZWQpfQogICAgLmhpZGRlbntkaXNwbGF5Om5vbmV9CiAgICAuaGludHtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXJnaW4tbGVmdDouMjVyZW19CiAgICAuZHJvcHtkaXNwbGF5Om5vbmU7IGJvcmRlcjoycHggZGFzaGVkICMyYTMzNDY7IGJvcmRlci1yYWRpdXM6MTJweDsgcGFkZGluZzoxMnB4OyB0ZXh0LWFsaWduOmNlbnRlcjsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjoxcmVtIGF1dG8gLjc1cmVtO30KICAgIC5kcm9wLmRyYWd7YmFja2dyb3VuZDojMGUxMzIwOyBjb2xvcjp2YXIoLS1pbmspfQogICAgY29kZSwgcHJlLCBrYmR7Zm9udC1mYW1pbHk6IHVpLW1vbm9zcGFjZSwgU0ZNb25vLVJlZ3VsYXIsIE1lbmxvLCBDb25zb2xhcywgbW9ub3NwYWNlfQoKICAgIC8qIE1hcmtkb3duIHJlbmRlcmluZyAqLwogICAgLm1ke2xpbmUtaGVpZ2h0OjEuNn0KICAgIC5tZCBoMSwubWQgaDIsLm1kIGgze2xpbmUtaGVpZ2h0OjEuMjU7IG1hcmdpbjouMnJlbSAwIC42cmVtfQogICAgLm1kIGgxe2ZvbnQtc2l6ZToxLjM1cmVtfQogICAgLm1kIGgye2ZvbnQtc2l6ZToxLjJyZW19CiAgICAubWQgaDN7Zm9udC1zaXplOjEuMDVyZW19CiAgICAubWQgcHttYXJnaW46LjVyZW0gMH0KICAgIC5tZCB1bCwubWQgb2x7bWFyZ2luOi41cmVtIDAgLjVyZW0gMS4xcmVtfQogICAgLm1kIGxpe21hcmdpbjouMnJlbSAwfQogICAgLm1kIGJsb2NrcXVvdGV7bWFyZ2luOi42cmVtIDA7IHBhZGRpbmc6LjI1cmVtIC44cmVtOyBib3JkZXItbGVmdDozcHggc29saWQgIzMwNDQ1ZjsgYmFja2dyb3VuZDojMGMxMTFiOyBjb2xvcjojZDZlMmYwOyBib3JkZXItcmFkaXVzOjZweH0KICAgIC5tZCBocntib3JkZXI6bm9uZTsgYm9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7IG1hcmdpbjouOHJlbSAwfQogICAgLm1kIGNvZGV7YmFja2dyb3VuZDojMGMwZjE1OyBwYWRkaW5nOi4xMnJlbSAuMzJyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6LjM1cmVtfQogICAgLm1kIHByZXtiYWNrZ3JvdW5kOiMwYzBmMTU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgcGFkZGluZzouOHJlbTsgb3ZlcmZsb3c6YXV0bzsgcG9zaXRpb246cmVsYXRpdmV9CiAgICAubWQgcHJlPmNvZGV7ZGlzcGxheTpibG9jazsgd2hpdGUtc3BhY2U6cHJlfQogICAgLm1kIHRhYmxle2JvcmRlci1jb2xsYXBzZTpjb2xsYXBzZTsgbWFyZ2luOi44cmVtIDA7IHdpZHRoOjEwMCU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAubWQgdGgsLm1kIHRke3BhZGRpbmc6LjVyZW0gLjdyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHRleHQtYWxpZ246bGVmdH0KICAgIC5tZCB0aHtiYWNrZ3JvdW5kOiMwZjE3MjU7IGZvbnQtd2VpZ2h0OjYwMH0KICAgIC5jb2Rld3JhcHtwb3NpdGlvbjpyZWxhdGl2ZX0KICAgIC5jb3B5YnRue3Bvc2l0aW9uOmFic29sdXRlOyB0b3A6LjQ1cmVtOyByaWdodDouNDVyZW07IGZvbnQtc2l6ZToxMnB4OyBwYWRkaW5nOi4yNXJlbSAuNDVyZW07IGJvcmRlci1yYWRpdXM6LjRyZW07IGJhY2tncm91bmQ6IzBmMTcyNTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgY3Vyc29yOnBvaW50ZXJ9CiAgICAuY29weWJ0bjpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIC50YXNrLWRvbmV7b3BhY2l0eTouOX0KCiAgICAvKiDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4ggc3R5bGluZyAqLwogICAgLnN5c2lucHV0e3dpZHRoOjEwMCU7IHJlc2l6ZTp2ZXJ0aWNhbDsgbWluLWhlaWdodDo2MHB4OyBtYXgtaGVpZ2h0OjMwdmg7IGJhY2tncm91bmQ6IzBjMGYxNTsgY29sb3I6dmFyKC0taW5rKTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZzouN3JlbSAuNzVyZW07IGJvcmRlci1yYWRpdXM6Ljc1cmVtOyBvdXRsaW5lOm5vbmV9CiAgICAuc3lzaW5wdXQ6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CgogICAgLyogSlNPTiBFZGl0b3IgKi8KICAgIGRldGFpbHMucmF3e2Rpc3BsYXk6bm9uZTsgbWF4LXdpZHRoOjkwMHB4OyBtYXJnaW46MXJlbSBhdXRvIDA7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgYmFja2dyb3VuZDojMGMwZjE1fQogICAgZGV0YWlscy5yYXc+c3VtbWFyeXtjdXJzb3I6cG9pbnRlcjsgcGFkZGluZzouNnJlbSAuOHJlbTsgbGlzdC1zdHlsZTpub25lOyB1c2VyLXNlbGVjdDpub25lfQogICAgZGV0YWlscy5yYXc+c3VtbWFyeTo6bWFya2Vye2Rpc3BsYXk6bm9uZX0KICAgIGRldGFpbHMucmF3W29wZW5dPnN1bW1hcnl7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAucmF3IC5pbm5lcntwYWRkaW5nOi44cmVtfQogICAgLnJhdyB0ZXh0YXJlYXtmb250LWZhbWlseTogdWktbW9ub3NwYWNlLCBTRk1vbm8tUmVndWxhciwgTWVubG8sIENvbnNvbGFzLCBtb25vc3BhY2U7IG1pbi1oZWlnaHQ6MjIwcHg7IHdoaXRlLXNwYWNlOnByZTsgdGFiLXNpemU6Mn0KICAgIC5yYXcgLnJvd3tkaXNwbGF5OmZsZXg7IGdhcDouNXJlbTsgYWxpZ24taXRlbXM6Y2VudGVyOyBtYXJnaW4tdG9wOi41cmVtOyBmbGV4LXdyYXA6d3JhcH0KICAgIC5iYWRnZXtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpfQoKICAgIC8qIFNjcm9sbC10by1ib3R0b20gYnV0dG9uICovCiAgICAudG8tYm90dG9te3Bvc2l0aW9uOmZpeGVkOyByaWdodDoxNnB4OyBib3R0b206MTZweDsgcGFkZGluZzouNXJlbSAuN3JlbTsgYm9yZGVyLXJhZGl1czouNnJlbTsgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBjb2xvcjp2YXIoLS1pbmspOyBkaXNwbGF5Om5vbmU7IG9wYWNpdHk6LjkyfQogICAgLnRvLWJvdHRvbS5zaG93e2Rpc3BsYXk6YmxvY2t9CiAgICAKICAgIC8qIFNUWUxFUyBGw5xSIFRISU5LSU5HLUJPWCAqLwogICAgLnRoaW5rLWJ1YmJsZSB7CiAgICAgICAgd2lkdGg6IDEwMCU7CiAgICB9CiAgICAudGhpbmstYnViYmxlIGRldGFpbHMgewogICAgICAgIGJhY2tncm91bmQ6ICMxYTE2MjU7CiAgICAgICAgYm9yZGVyOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBib3JkZXItcmFkaXVzOiAxMHB4OwogICAgICAgIGZvbnQtc2l6ZTogMTNweDsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeSB7CiAgICAgICAgY3Vyc29yOiBwb2ludGVyOwogICAgICAgIHBhZGRpbmc6IC41cmVtIC44cmVtOwogICAgICAgIGZvbnQtd2VpZ2h0OiA1MDA7CiAgICAgICAgY29sb3I6IHZhcigtLW11dGVkKTsKICAgICAgICB1c2VyLXNlbGVjdDogbm9uZTsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeTpob3ZlciB7CiAgICAgICAgY29sb3I6IHZhcigtLWluayk7CiAgICB9CiAgICAudGhpbmstYnViYmxlIC50aGluay1jb250ZW50IHsKICAgICAgICBwYWRkaW5nOiAuMXJlbSAuOHJlbSAuN3JlbTsKICAgICAgICBib3JkZXItdG9wOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBjb2xvcjogI2MwYjhkMTsKICAgIH0KCS8qIOKAlCBQb2xpc2hpbmc6IFNhbXBsaW5nLUlucHV0cyBpbSBFaW5zdGVsbHVuZ3MtUGFuZWwg4oCUICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCB7ICAgICAgICAgICAgICAgLyogTGFiZWwgKyBGZWxkIHZlcnRpa2FsLCBrb21wYWt0ICovCiAgZGlzcGxheTogZmxleDsKICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwogIGdhcDogLjI1cmVtOwogIGFsaWduLWl0ZW1zOiBmbGV4LXN0YXJ0Owp9CgovKiBFaW5oZWl0bGljaGUsIHNjaG1hbGUgTnVtYmVyLUlucHV0cyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIHdpZHRoOiAxMGNoOyAgICAgICAgICAgICAgICAgICAgICAgIC8qIGZpeGUgQnJlaXRlIC0+IGdsZWljaGUgS2FudGUgKi8KICBoZWlnaHQ6IDMycHg7CiAgcGFkZGluZzogLjM1cmVtIC41cmVtOwogIGJhY2tncm91bmQ6ICMwYzBmMTU7CiAgY29sb3I6IHZhcigtLWluayk7CiAgYm9yZGVyOiAxcHggc29saWQgdmFyKC0tbGluZSk7CiAgYm9yZGVyLXJhZGl1czogLjVyZW07CiAgdGV4dC1hbGlnbjogcmlnaHQ7CiAgZm9udC12YXJpYW50LW51bWVyaWM6IHRhYnVsYXItbnVtczsgLyogc2F1YmVyZSwgYXVzZ2VyaWNodGV0ZSBaaWZmZXJuICovCiAgb3V0bGluZTogbm9uZTsKfQoKLyogRm9rdXMtRmVlZGJhY2sgcGFzc2VuZCB6dSBkZW4gcmVzdGxpY2hlbiBDb250cm9scyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Zm9jdXMgewogIGJvcmRlci1jb2xvcjogdmFyKC0tYWNjKTsKICBib3gtc2hhZG93OiAwIDAgMCAycHggcmdiYSgxMDYsMTYxLDI1NSwuMTUpOwp9CgovKiBEZXplbnRlIFBsYWNlaG9sZGVyLUZhcmJlICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6cGxhY2Vob2xkZXIgewogIGNvbG9yOiB2YXIoLS1tdXRlZCk7CiAgb3BhY2l0eTogLjk7Cn0KCi8qIFNwaW5uZXIgYmVpIE51bWJlci1JbnB1dHMgYXVzYmxlbmRlbiAobnVyIGltIFNldHRpbmdzLVBhbmVsKSAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Oi13ZWJraXQtb3V0ZXItc3Bpbi1idXR0b24sCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6LXdlYmtpdC1pbm5lci1zcGluLWJ1dHRvbiB7CiAgLXdlYmtpdC1hcHBlYXJhbmNlOiBub25lOwogIG1hcmdpbjogMDsKfQojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIC1tb3otYXBwZWFyYW5jZTogdGV4dGZpZWxkOwp9CgovKiBLbGVpbmVzIFR1bmluZyBmw7xyIGRpZSBHcnVwcGU6IGRpY2h0ZXIgenVzYW1tZW4sIGJyaWNodCBzYXViZXIgdW0gKi8KI3NldHRpbmdzUGFuZWwgPiAuYmFyID4gZGl2Omxhc3QtY2hpbGQsCiNzZXR0aW5nc1BhbmVsIC5iYXIgPiBkaXY6bGFzdC1jaGlsZCB7CiAgZ2FwOiAuNXJlbSAuNzVyZW07ICAgICAgICAgICAgICAgICAgLyogWmVpbGVuL1NwYWx0ZW4tR2FwICovCiAgZmxleC13cmFwOiB3cmFwOwp9CgogIAogICAgLyogLS0tLS0tLS0tLSBSZXNwb25zaXZlIC0tLS0tLS0tLS0gKi8KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA5ODBweCl7CiAgICAgIGhlYWRlcntnYXA6LjZyZW19CiAgICAgIGhlYWRlciAuc3BhY2Vye2Rpc3BsYXk6bm9uZX0KICAgICAgaGVhZGVyIGxhYmVsLnNtYWxse21pbi13aWR0aDoyMDBweH0KICAgICAgaGVhZGVyIC5yb3d7d2lkdGg6MTAwJX0KICAgICAgaGVhZGVyICN0b2tlblJhdGV7d2lkdGg6YXV0bzsgdGV4dC1hbGlnbjpsZWZ0ICFpbXBvcnRhbnR9CiAgICB9CiAgICBAbWVkaWEgKG1heC13aWR0aDogNzIwcHgpewogICAgICBoZWFkZXIgbGFiZWwuc21hbGx7ZmxleDoxIDEgMjIwcHh9CiAgICAgIGhlYWRlciBpbnB1dCwgaGVhZGVyIHNlbGVjdHt3aWR0aDoxMDAlfQogICAgICBtYWluI3Njcm9sbGVye3BhZGRpbmc6Ljg1cmVtIC43NXJlbSAxLjI1cmVtfQogICAgICAuYmFye2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgICAgIGZvb3RlciAucm93e2ZsZXgtd3JhcDp3cmFwfQogICAgICB0ZXh0YXJlYXttaW4taGVpZ2h0OjkwcHh9CiAgICAgIC8qIFNldHRpbmdzIHBhbmVsIHN0YWNrICovCiAgICAgICNzZXR0aW5nc1BhbmVse2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgIWltcG9ydGFudH0KICAgICAgI3NldHRpbmdzUGFuZWwgPiBkaXZ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAhaW1wb3J0YW50fQogICAgICAjc2V0dGluZ3NQYW5lbCA+IGRpdiA+IGRpdjpsYXN0LWNoaWxke21pbi13aWR0aDowICFpbXBvcnRhbnQ7IGFsaWduLXNlbGY6c3RhcnQgIWltcG9ydGFudH0KICAgIH0KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA0ODBweCl7CiAgICAgIGhlYWRlciBsYWJlbC5zbWFsbHtmbGV4OjEgMSAxMDAlfQogICAgICBoZWFkZXIgLnJvd3tnYXA6LjRyZW19CiAgICAgIGhlYWRlciBidXR0b24sIGZvb3RlciBidXR0b257cGFkZGluZzouNXJlbSAuNzVyZW19CiAgICB9CgogIAogICAgLyogQXR0YWNobWVudHMgKi8KICAgIC5jaGlwewogICAgICBkaXNwbGF5OmlubGluZS1mbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNDVyZW07CiAgICAgIHBhZGRpbmc6LjM1cmVtIC41NXJlbTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OyBib3JkZXItcmFkaXVzOjk5OXB4OyBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6MTAwJTsKICAgIH0KICAgIC5jaGlwIHNtYWxse2NvbG9yOnZhcigtLW11dGVkKX0KICAgIC5jaGlwIGJ1dHRvbnsKICAgICAgcGFkZGluZzouMTVyZW0gLjM1cmVtOyBib3JkZXItcmFkaXVzOjhweDsKICAgICAgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgfQogICAgLmNoaXAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQoKICAgIC8qIE1hcmtkb3duIGltYWdlcyAqLwogICAgLm1kIGltZ3ttYXgtd2lkdGg6MTAwJTsgaGVpZ2h0OmF1dG87IGJvcmRlci1yYWRpdXM6MTBweDsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KCiAgPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KPGRpdiBjbGFzcz0id3JhcCI+CiAgPGhlYWRlcj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9InRpdGxlIj7wn5KsIOODgeODo+ODg+ODiFVJPC9kaXY+CiAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzcGFjZXIiPjwvZGl2PgogICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+PHNwYW4+44Oi44OH44OrPC9zcGFuPgogICAgICA8c2VsZWN0IGlkPSJtb2RlbCI+CiAgICAgICAgPG9wdGlvbiB2YWx1ZT0iIiBzZWxlY3RlZCBkaXNhYmxlZD7igJQg44Oi44OH44Or44KS6Kqt44G/6L6844G/IOKAlDwvb3B0aW9uPgogICAgICA8L3NlbGVjdD4KICAgIDwvbGFiZWw+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICA8YnV0dG9uIGlkPSJhdXRoQnRuIiBjbGFzcz0ic2Vjb25kYXJ5IiB0aXRsZT0iQVBJIOiqjeiovO+8iEJhc2lj77yJIj7wn5SRPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImluZm9CdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSLmg4XloLEiPuaDheWgsTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICAgIDxkaXYgaWQ9InF1b3RhTGVmdCIgY2xhc3M9InNtYWxsIiBzdHlsZT0id2lkdGg6MTFjaDsgdGV4dC1hbGlnbjpyaWdodDsiIHRpdGxlPSJLYWdnbGUgR1BVIOaui+OCiuaZgumWkyI+PC9kaXY+CiAgICAgICAgPGRpdiBpZD0idG9rZW5SYXRlIiBjbGFzcz0ic21hbGwiIHN0eWxlPSJ3aWR0aDoxMmNoOyB0ZXh0LWFsaWduOnJpZ2h0OyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJvdyIgaWQ9InN0b3BSb3ciIHN0eWxlPSJkaXNwbGF5Om5vbmU7Ij4KICAgICAgPGJ1dHRvbiBpZD0ic3RvcFNlcnZlciIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IuOCteODvOODkOODvOOCkuWBnOatoiI+4pagIOWBnOatojwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9oZWFkZXI+CgogIDxtYWluIGlkPSJzY3JvbGxlciI+CiAgICA8ZGl2IGlkPSJkcm9wIiBjbGFzcz0iZHJvcCI+44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GZ44KL44Gr44Gv44OV44Kh44Kk44Or44KS44OJ44Ot44OD44OXIChKU09OIGluIHRoZSB1c3VhbCA8Y29kZT5be3JvbGUsIGNvbnRlbnR9XTwvY29kZT4gZm9ybWF0KS48YnI+PHNwYW4gY2xhc3M9ImhpbnQiPuOBvuOBn+OBryA8Yj7jgqTjg7Pjg53jg7zjg4g8L2I+IGFib3ZlLjwvc3Bhbj48L2Rpdj4KCiAgICA8ZGV0YWlscyBpZD0icmF3Qm94IiBjbGFzcz0icmF3Ij4KICAgICAgPHN1bW1hcnk+4pyOIEpTT07jgajjgZfjgabjg4Hjg6Pjg4Pjg4jjgpLnt6jpm4YgPHNwYW4gY2xhc3M9ImJhZGdlIj4o5YiH5pu/KTwvc3Bhbj48L3N1bW1hcnk+CiAgICAgIDxkaXYgY2xhc3M9ImlubmVyIj4KICAgICAgICA8dGV4dGFyZWEgaWQ9Impzb25FZGl0b3IiIHNwZWxsY2hlY2s9ImZhbHNlIiBwbGFjZWhvbGRlcj0nWyB7ICJyb2xlIjogInN5c3RlbSIsICJjb250ZW50IjogIi4uLiIgfSwgLi4uIF0nPjwvdGV4dGFyZWE+CiAgICAgICAgPGRpdiBjbGFzcz0icm93Ij4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25BcHBseSI+6YGp55SoICjnva7mj5spPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uQXBwZW5kIiBjbGFzcz0ic2Vjb25kYXJ5Ij7ov73oqJg8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25SZWZyZXNoIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgYvjgonmm7TmlrA8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25QcmV0dHkiIGNsYXNzPSJzZWNvbmRhcnkiPuaVtOW9ojwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBpZD0ianNvbk1pbmlmeSIgY2xhc3M9InNlY29uZGFyeSI+5Zyn57iuPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uVmFsaWRhdGUiIGNsYXNzPSJzZWNvbmRhcnkiPuaknOiovDwvYnV0dG9uPgogICAgICAgICAgPHNwYW4gaWQ9Impzb25TdGF0dXMiIGNsYXNzPSJiYWRnZSI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGV0YWlscz4KCiAgICA8ZGl2IGlkPSJsb2ciIGNsYXNzPSJsb2ciPjwvZGl2PgogIDwvbWFpbj4KCiAgPGZvb3Rlcj4KICAgIDxkaXYgY2xhc3M9ImJhciI+CiAgICAgIDx0ZXh0YXJlYSBpZD0iaW5wdXQiIHBsYWNlaG9sZGVyPSLjg6Hjg4Pjgrvjg7zjgrjjgpLlhaXlipvigKYgKOmAgeS/oTogRW50ZXIsIOaUueihjDogU2hpZnQrRW50ZXIpIj48L3RleHRhcmVhPgogICAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJkaXNwbGF5OmZsZXg7IGdhcDouNnJlbTsgYWxpZ24taXRlbXM6Y2VudGVyIj4KICAgICAgICA8YnV0dG9uIGlkPSJhdHRhY2hCdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSJBdHRhY2ggZmlsZXMiPvCfk448L2J1dHRvbj4KICAgICAgICA8aW5wdXQgaWQ9ImF0dGFjaElucHV0IiB0eXBlPSJmaWxlIiBjbGFzcz0iaGlkZGVuIiBtdWx0aXBsZSAvPgogICAgICAgIDxidXR0b24gaWQ9InNlbmQiPumAgeS/oTwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9InN0b3AiIGNsYXNzPSJzZWNvbmRhcnkiIGRpc2FibGVkPuWBnOatojwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9ImNsZWFyIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgq/jg6rjgqI8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGlkPSJzZXR0aW5nc1RvZ2dsZSIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IlRvZ2dsZSBzZXR0aW5ncyI+4pqZ77iPPC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0JhciIgY2xhc3M9ImJhciBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOi42cmVtOyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0xpc3QiIHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBmbGV4LXdyYXA6d3JhcCI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgaWQ9InNldHRpbmdzUGFuZWwiIGNsYXNzPSJiYXIgaGlkZGVuIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW07IGdyaWQtY29sdW1uOjEvLTE7IHdpZHRoOjEwMCUiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuOCt+OCueODhuODoOODl+ODreODs+ODl+ODiDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFyZWEgaWQ9InN5c3RlbSIgY2xhc3M9InN5c2lucHV0IiByb3dzPSIyIiBwbGFjZWhvbGRlcj0i5Lu75oSPIOKAlCBlLmcuIFlvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC4iPjwvdGV4dGFyZWE+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0ibWluLXdpZHRoOjI2MHB4OyBhbGlnbi1zZWxmOmVuZDsgZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNnJlbTsgZmxleC13cmFwOndyYXAiPgogICAgICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+VGVtcGVyYXR1cmUgPHNwYW4gaWQ9InRlbXBWYWwiIGNsYXNzPSJoaW50Ij4wLjc8L3NwYW4+PC9sYWJlbD4KICAgICAgICAgIDxpbnB1dCBpZD0idGVtcCIgdHlwZT0icmFuZ2UiIG1pbj0iMCIgbWF4PSIyIiBzdGVwPSIwLjEiIHZhbHVlPSIwLjciIHN0eWxlPSJ3aWR0aDoxNjBweCI+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3BfcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj5taW5fcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJtaW5fcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3Bfazxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfayIgdHlwZT0ibnVtYmVyIiBzdGVwPSIxIiBtaW49IjAiIHBsYWNlaG9sZGVyPSIoZGVmYXVsdCkiIHN0eWxlPSJ3aWR0aDo5Y2g7IHRleHQtYWxpZ246cmlnaHQiPgogICAgICAgICAgPC9sYWJlbD4KICAgICAgICAgIDxsYWJlbCBjbGFzcz0ic21hbGwiPnRoaW5raW5nPGJyPgogICAgICAgICAgICA8c2VsZWN0IGlkPSJ0aGlua2luZyIgc3R5bGU9IndpZHRoOjExY2giPgogICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+KGRlZmF1bHQpPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib24iPm9uPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib2ZmIj5vZmY8L29wdGlvbj4KICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICA8L2xhYmVsPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZm9vdGVyPgo8L2Rpdj4KCjxidXR0b24gaWQ9InRvQm90dG9tIiBjbGFzcz0idG8tYm90dG9tIiB0aXRsZT0i5LiA55Wq5LiL44G4Ij7ilrwg5LiA55Wq5LiLPC9idXR0b24+Cgo8ZGlhbG9nIGlkPSJpbmZvRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCfkqwg44OB44Oj44OD44OIVUk8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ic2Vjb25kYXJ5IiB2YWx1ZT0iY2xvc2UiIGFyaWEtbGFiZWw9IkNsb3NlIj7inJU8L2J1dHRvbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1ib2R5Ij4KICAgICAgPGRpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPuODkOODvOOCuOODp+ODszo8L3NwYW4+IDxjb2RlIGlkPSJpbmZvVmVyc2lvbiI+MC4wLjA8L2NvZGU+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZm9udC1zaXplOjEycHgiPuOCueODhuODvOODiOODrOOCuVVJIOKAlCBubyBzZXJ2ZXItc2lkZSBjaGF0IHN0b3JhZ2UuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImluZm8tYWN0aW9ucyI+CiAgICAgIDxidXR0b24gdmFsdWU9ImNsb3NlIj7plonjgZjjgos8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZm9ybT4KPC9kaWFsb2c+Cgo8ZGlhbG9nIGlkPSJhdXRoRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCflJEgQVBJIOiqjeiovDwvZGl2PgogICAgICA8YnV0dG9uIGNsYXNzPSJzZWNvbmRhcnkiIHZhbHVlPSJjbG9zZSIgYXJpYS1sYWJlbD0iQ2xvc2UiPuKclTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJpbmZvLWJvZHkiPgogICAgICA8ZGl2IGNsYXNzPSJtdXRlZCBzbWFsbCIgc3R5bGU9Im1hcmdpbi1ib3R0b206LjZyZW0iPuOCteODvOODkOODvOOBjCBCYXNpYyDoqo3oqLzjgpLopoHmsYLjgZnjgovloLTlkIjjgavlhaXlipvjgZfjgb7jgZnvvIhzZXNzaW9uU3RvcmFnZSDjgavkv53lrZjjg7vjgr/jg5bjgpLplonjgZjjgovjgajmtojjgYjjgb7jgZnvvInjgII8L2Rpdj4KICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbi1ib3R0b206LjM1cmVtIj7jg6bjg7zjgrbjg7w8L2xhYmVsPgogICAgICA8aW5wdXQgaWQ9ImF1dGhVc2VyIiB0eXBlPSJ0ZXh0IiBhdXRvY29tcGxldGU9InVzZXJuYW1lIiB2YWx1ZT0iZXhsMyIgc3R5bGU9IndpZHRoOjEwMCU7IG1hcmdpbi1ib3R0b206LjZyZW0iPgogICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuODkeOCueODr+ODvOODiTwvbGFiZWw+CiAgICAgIDxpbnB1dCBpZD0iYXV0aFBhc3MiIHR5cGU9InBhc3N3b3JkIiBhdXRvY29tcGxldGU9ImN1cnJlbnQtcGFzc3dvcmQiIHN0eWxlPSJ3aWR0aDoxMDAlOyBtYXJnaW4tYm90dG9tOi42cmVtIj4KICAgICAgPGRpdiBpZD0iYXV0aFN0YXR1cyIgY2xhc3M9InNtYWxsIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1hY3Rpb25zIj4KICAgICAgPGJ1dHRvbiBpZD0iYXV0aFNhdmUiIHR5cGU9ImJ1dHRvbiI+5L+d5a2YPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImF1dGhDbGVhciIgdHlwZT0iYnV0dG9uIiBjbGFzcz0ic2Vjb25kYXJ5Ij7op6PpmaQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiB2YWx1ZT0iY2xvc2UiIGNsYXNzPSJzZWNvbmRhcnkiPumWieOBmOOCizwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9mb3JtPgo8L2RpYWxvZz4KCgo8c2NyaXB0PgooKCkgPT4gewogIC8vIEFQSSDjga7mjqXntprlhYjjgpLoh6rli5XliKTlrprjgZnjgovjgILlkIzjgZggSFRNTCDjgpLjg63jg7zjgqvjg6vjgafjgoIgS2FnZ2xlIOOBp+OCguS9v+OBhuOAggogIC8vCiAgLy8gMSkgP2FwaT08dXJsPiDjgYzmmI7npLrjgZXjgozjgabjgYTjgozjgbDjgZ3jgozvvIjmnIDlhKrlhYjvvIkKICAvLyAyKSA/cG9ydD08bj4g44GM44GC44KM44GwIGh0dHA6Ly8xMjcuMC4wLjE6PG4+L3Yx77yI44Ot44O844Kr44OrIHV6dSAvIGxsYW1hLXNlcnZlcu+8iQogIC8vIDMpIGh0dHAocykg44Gn6YWN5L+h44GV44KM44Gm44GE44KL44Gq44KJ5ZCM5LiA44Kq44Oq44K444Oz77yIS2FnZ2xlIOS4ree2me+8iQogIC8vIDQpIGZpbGU6Ly8g44Gn6ZaL44GE44Gf5aC05ZCI44Gv5pei5a6a44Od44O844OICiAgY29uc3QgcGFyYW1zID0gbmV3IFVSTFNlYXJjaFBhcmFtcyh3aW5kb3cubG9jYXRpb24uc2VhcmNoKTsKICBjb25zdCBwb3J0UGFyYW0gPSBwYXJhbXMuZ2V0KCdwb3J0Jyk7CiAgY29uc3QgYXBpUGFyYW0gPSBwYXJhbXMuZ2V0KCdhcGknKTsKCiAgZnVuY3Rpb24gcmVzb2x2ZUFwaUJhc2UoKSB7CiAgICBpZiAoYXBpUGFyYW0pIHJldHVybiBhcGlQYXJhbS5yZXBsYWNlKC9cLyQvLCAnJyk7CiAgICBpZiAocG9ydFBhcmFtKSByZXR1cm4gYGh0dHA6Ly8xMjcuMC4wLjE6JHtwb3J0UGFyYW19L3YxYDsKICAgIC8vIOS4ree2meOChOODiOODs+ODjeODq+e1jOeUseOBp+mFjeS/oeOBleOCjOOBpuOBhOOCi+WgtOWQiOOBr+WQjOS4gOOCquODquOCuOODs+OAggogICAgLy8g44GT44GT44KSIDEyNy4wLjAuMSDlm7rlrprjgavjgZnjgovjgajjgIHjg6rjg6Ljg7zjg4jjgYvjgonplovjgYTjgZ/jgajjgY3jgasKICAgIC8vIOaJi+WFg+OBriBQQyDjgpLopovjgavooYzjgaPjgabjgZfjgb7jgYbjgIIKICAgIGlmIChsb2NhdGlvbi5wcm90b2NvbCA9PT0gJ2h0dHA6JyB8fCBsb2NhdGlvbi5wcm90b2NvbCA9PT0gJ2h0dHBzOicpIHsKICAgICAgcmV0dXJuIGxvY2F0aW9uLm9yaWdpbiArICcvdjEnOwogICAgfQogICAgcmV0dXJuICdodHRwOi8vMTI3LjAuMC4xOjk5MzEvdjEnOwogIH0KCiAgY29uc3QgQVBJX0JBU0UgPSByZXNvbHZlQXBpQmFzZSgpOwoKICBjb25zdCBCQVNFX0hFQURFUlMgPSB7fTsKICBjb25zdCBlbHMgPSB7CiAgICBzY3JvbGxlcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Njcm9sbGVyJyksCiAgICBsb2c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdsb2cnKSwKICAgIGRyb3A6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkcm9wJyksCiAgICBpbnB1dDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2lucHV0JyksCiAgICBzZW5kQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2VuZCcpLAogICAgc3RvcEJ0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0b3AnKSwKICAgIGNsZWFyQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2xlYXInKSwKICAgIGF0dGFjaEJ0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaEJ0bicpLAogICAgYXR0YWNoSW5wdXQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdHRhY2hJbnB1dCcpLAogICAgYXR0YWNobWVudHNCYXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdHRhY2htZW50c0JhcicpLAogICAgYXR0YWNobWVudHNMaXN0OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXR0YWNobWVudHNMaXN0JyksCiAgICBtb2RlbDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21vZGVsJyksCiAgICBzeXN0ZW06IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzeXN0ZW0nKSwKICAgIHRlbXA6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0ZW1wJyksCiAgICB0ZW1wVmFsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndGVtcFZhbCcpLAogICAgaW1wb3J0QnRuOiBudWxsLAogICAgaW1wb3J0RmlsZTogbnVsbCwKICAgIGV4cG9ydEJ0bjogbnVsbCwKICAgIGxvYWRNb2RlbHNCdG46IG51bGwsCiAgICBtb2RlbFN0YXR1czogbnVsbCwKICAgIHJhd0JveDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Jhd0JveCcpLAogICAganNvbkVkaXRvcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25FZGl0b3InKSwKICAgIGpzb25BcHBseTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25BcHBseScpLAogICAganNvbkFwcGVuZDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25BcHBlbmQnKSwKICAgIGpzb25SZWZyZXNoOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvblJlZnJlc2gnKSwKICAgIGpzb25QcmV0dHk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uUHJldHR5JyksCiAgICBqc29uTWluaWZ5OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbk1pbmlmeScpLAogICAganNvblZhbGlkYXRlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvblZhbGlkYXRlJyksCiAgICBqc29uU3RhdHVzOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvblN0YXR1cycpLAogICAgdG9Cb3R0b206IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0b0JvdHRvbScpLAogICAgc2V0dGluZ3NUb2dnbGU6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzZXR0aW5nc1RvZ2dsZScpLAogICAgc2V0dGluZ3NQYW5lbDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NldHRpbmdzUGFuZWwnKSwKICAgIHRva2VuUmF0ZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3Rva2VuUmF0ZScpLAogICAgcXVvdGFMZWZ0OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncXVvdGFMZWZ0JyksCiAgICBzdG9wUm93OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcFJvdycpLAogICAgc3RvcFNlcnZlcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0b3BTZXJ2ZXInKSwKICAgIGluZm9CdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvQnRuJyksCiAgICBpbmZvRGlhbG9nOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnaW5mb0RpYWxvZycpLAogICAgaW5mb1ZlcnNpb246IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvVmVyc2lvbicpLAogICAgaW5mb1Byb2plY3RVcmw6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvUHJvamVjdFVybCcpLAogICAgaW5mb0ltcHJpbnQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvSW1wcmludCcpLAogICAgYXV0aEJ0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhCdG4nKSwKICAgIGF1dGhEaWFsb2c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoRGlhbG9nJyksCiAgICBhdXRoVXNlcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhVc2VyJyksCiAgICBhdXRoUGFzczogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhQYXNzJyksCiAgICBhdXRoU3RhdHVzOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFN0YXR1cycpLAogICAgYXV0aFNhdmU6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoU2F2ZScpLAogICAgYXV0aENsZWFyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aENsZWFyJyksCiAgICAvLyBORVU6CiAgICB0b3BfcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RvcF9wJyksCiAgICBtaW5fcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21pbl9wJyksCiAgICB0b3BfazogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RvcF9rJyksCiAgICB0aGlua2luZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RoaW5raW5nJyksCiAgfTsKCiAgbGV0IG1lc3NhZ2VzID0gW107CiAgbGV0IGF0dGFjaG1lbnRzID0gW107IC8vIHtpZCxuYW1lLHR5cGUsc2l6ZSwga2luZDondGV4dCd8J2ltYWdlJ3wnb3RoZXInLCB0ZXh0PywgZGF0YVVybD8sIHRydW5jYXRlZD99CiAgbGV0IG5leHRBdHRhY2hJZCA9IDE7CgogIC8qIC0tLS0tLS0tLS0gQVBJIOiqjeiovO+8iEJhc2lj77yJIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjgYwgQVBJX1BBU1Mg44KS6Kit5a6a44GX44Gm44GE44KL44Go44GN44GvIC92MS8qIOOCgiAvc3RvcCDjgoIgNDAxIOOBq+OBquOCi+OAggogIC8vIOizh+agvOaDheWgseOBryBzZXNzaW9uU3RvcmFnZSDjgavjgaDjgZHnva7jgY/vvIhsb2NhbFN0b3JhZ2Ug44Gr44Gv5q6L44GV44Gq44GE77yJ44CCCiAgY29uc3QgQVVUSF9LRVkgPSAna2NfYmFzaWNfYXV0aCc7CiAgY29uc3QgQVVUSF9VU0VSX0tFWSA9ICdrY19iYXNpY191c2VyJzsKCiAgZnVuY3Rpb24gYjY0dXRmOChzKXsKICAgIC8vIFB5dGhvbiDlgbTjga8gVVRGLTgg44GnIGJhc2U2NCDljJbjgZfjgabjgYTjgovjga7jgafjgIFidG9hIOebtOWRvOOBs++8iExhdGluLTHvvInjgafjga8KICAgIC8vIOmdniBBU0NJSSDjga7jg5Hjgrnjg6/jg7zjg4njgafkuIDoh7TjgZfjgarjgYTjgIIKICAgIGNvbnN0IGJ5dGVzID0gbmV3IFRleHRFbmNvZGVyKCkuZW5jb2RlKHMpOwogICAgbGV0IGJpbiA9ICcnOwogICAgZm9yIChjb25zdCBiIG9mIGJ5dGVzKSBiaW4gKz0gU3RyaW5nLmZyb21DaGFyQ29kZShiKTsKICAgIHJldHVybiBidG9hKGJpbik7CiAgfQogIGZ1bmN0aW9uIGdldEF1dGhIZWFkZXIoKXsKICAgIGNvbnN0IHYgPSBzZXNzaW9uU3RvcmFnZS5nZXRJdGVtKEFVVEhfS0VZKTsKICAgIHJldHVybiB2ID8geyAnQXV0aG9yaXphdGlvbic6IGBCYXNpYyAke3Z9YCB9IDoge307CiAgfQogIGZ1bmN0aW9uIGFwaUhlYWRlcnMoZXh0cmEpewogICAgcmV0dXJuIHsgLi4uQkFTRV9IRUFERVJTLCAuLi5nZXRBdXRoSGVhZGVyKCksIC4uLihleHRyYSB8fCB7fSkgfTsKICB9CiAgZnVuY3Rpb24gc2V0QXV0aENyZWRlbnRpYWxzKHVzZXIsIHBhc3MpewogICAgc2Vzc2lvblN0b3JhZ2Uuc2V0SXRlbShBVVRIX0tFWSwgYjY0dXRmOChgJHt1c2VyfToke3Bhc3N9YCkpOwogICAgc2Vzc2lvblN0b3JhZ2Uuc2V0SXRlbShBVVRIX1VTRVJfS0VZLCB1c2VyKTsKICB9CiAgZnVuY3Rpb24gY2xlYXJBdXRoQ3JlZGVudGlhbHMoKXsKICAgIHNlc3Npb25TdG9yYWdlLnJlbW92ZUl0ZW0oQVVUSF9LRVkpOwogICAgc2Vzc2lvblN0b3JhZ2UucmVtb3ZlSXRlbShBVVRIX1VTRVJfS0VZKTsKICB9CiAgZnVuY3Rpb24gb3BlbkF1dGhEaWFsb2cobXNnKXsKICAgIGlmICghZWxzLmF1dGhEaWFsb2cpIHJldHVybjsKICAgIGlmIChlbHMuYXV0aFVzZXIpIHsKICAgICAgZWxzLmF1dGhVc2VyLnZhbHVlID0gc2Vzc2lvblN0b3JhZ2UuZ2V0SXRlbShBVVRIX1VTRVJfS0VZKSB8fCAnZXhsMyc7CiAgICB9CiAgICBpZiAoZWxzLmF1dGhQYXNzKSBlbHMuYXV0aFBhc3MudmFsdWUgPSAnJzsKICAgIGlmIChlbHMuYXV0aFN0YXR1cykgZWxzLmF1dGhTdGF0dXMudGV4dENvbnRlbnQgPSBtc2cgfHwgJyc7CiAgICBpZiAodHlwZW9mIGVscy5hdXRoRGlhbG9nLnNob3dNb2RhbCA9PT0gJ2Z1bmN0aW9uJykgZWxzLmF1dGhEaWFsb2cuc2hvd01vZGFsKCk7CiAgICBlbHNlIGVscy5hdXRoRGlhbG9nLnNldEF0dHJpYnV0ZSgnb3BlbicsICcnKTsKICAgIHNldFRpbWVvdXQoKCkgPT4gZWxzLmF1dGhQYXNzPy5mb2N1cygpLCA1MCk7CiAgfQogIGZ1bmN0aW9uIGNsb3NlQXV0aERpYWxvZygpewogICAgaWYgKCFlbHMuYXV0aERpYWxvZykgcmV0dXJuOwogICAgaWYgKHR5cGVvZiBlbHMuYXV0aERpYWxvZy5jbG9zZSA9PT0gJ2Z1bmN0aW9uJykgZWxzLmF1dGhEaWFsb2cuY2xvc2UoKTsKICAgIGVsc2UgZWxzLmF1dGhEaWFsb2cucmVtb3ZlQXR0cmlidXRlKCdvcGVuJyk7CiAgfQoKICBmdW5jdGlvbiBmbXRCeXRlcyhuKXsKICAgIGlmICghTnVtYmVyLmlzRmluaXRlKG4pKSByZXR1cm4gJyc7CiAgICBjb25zdCB1ID0gWydCJywnS0InLCdNQicsJ0dCJ107IGxldCBpPTA7IGxldCB4PW47CiAgICB3aGlsZSAoeD49MTAyNCAmJiBpPHUubGVuZ3RoLTEpeyB4Lz0xMDI0OyBpKys7IH0KICAgIHJldHVybiBgJHt4LnRvRml4ZWQoaT8xOjApfSAke3VbaV19YDsKICB9CgogIGZ1bmN0aW9uIHJlbmRlckF0dGFjaG1lbnRzKCl7CiAgICBjb25zdCBsaXN0ID0gZWxzLmF0dGFjaG1lbnRzTGlzdDsKICAgIGxpc3QuaW5uZXJIVE1MID0gJyc7CiAgICBpZiAoIWF0dGFjaG1lbnRzLmxlbmd0aCl7CiAgICAgIGVscy5hdHRhY2htZW50c0Jhci5jbGFzc0xpc3QuYWRkKCdoaWRkZW4nKTsKICAgICAgcmV0dXJuOwogICAgfQogICAgZWxzLmF0dGFjaG1lbnRzQmFyLmNsYXNzTGlzdC5yZW1vdmUoJ2hpZGRlbicpOwoKICAgIGZvciAoY29uc3QgYSBvZiBhdHRhY2htZW50cyl7CiAgICAgIGNvbnN0IGNoaXAgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTsKICAgICAgY2hpcC5jbGFzc05hbWUgPSAnY2hpcCc7CgogICAgICBjb25zdCBpY29uID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogICAgICBpY29uLnRleHRDb250ZW50ID0gKGEua2luZCA9PT0gJ2ltYWdlJykgPyAn8J+WvO+4jycgOiAoYS5raW5kID09PSAndGV4dCcpID8gJ/Cfk4QnIDogJ/Cfk6YnOwoKICAgICAgY29uc3QgbmFtZSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ3NwYW4nKTsKICAgICAgbmFtZS5zdHlsZS5vdmVyZmxvdyA9ICdoaWRkZW4nOwogICAgICBuYW1lLnN0eWxlLnRleHRPdmVyZmxvdyA9ICdlbGxpcHNpcyc7CiAgICAgIG5hbWUuc3R5bGUud2hpdGVTcGFjZSA9ICdub3dyYXAnOwogICAgICBuYW1lLnN0eWxlLm1heFdpZHRoID0gJzU1Y2gnOwogICAgICBuYW1lLnRleHRDb250ZW50ID0gYS5uYW1lOwoKICAgICAgY29uc3Qgc2l6ZSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ3NtYWxsJyk7CiAgICAgIHNpemUudGV4dENvbnRlbnQgPSBmbXRCeXRlcyhhLnNpemUpOwoKICAgICAgY29uc3QgcmVtb3ZlID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgICAgIHJlbW92ZS50aXRsZSA9ICdSZW1vdmUnOwogICAgICByZW1vdmUudGV4dENvbnRlbnQgPSAn4pyVJzsKICAgICAgcmVtb3ZlLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCk9PnsKICAgICAgICBhdHRhY2htZW50cyA9IGF0dGFjaG1lbnRzLmZpbHRlcih4ID0+IHguaWQgIT09IGEuaWQpOwogICAgICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgICAgIH0pOwoKICAgICAgY2hpcC5hcHBlbmQoaWNvbiwgbmFtZSwgc2l6ZSwgcmVtb3ZlKTsKICAgICAgbGlzdC5hcHBlbmRDaGlsZChjaGlwKTsKICAgIH0KICB9CgogIGNvbnN0IE1BWF9URVhUX0NIQVJTX1BFUl9GSUxFID0gMjAwMDA7CiAgY29uc3QgTUFYX1RPVEFMX0FUVEFDSE1FTlRTID0gMTI7CgogIGZ1bmN0aW9uIHJlYWRGaWxlQXNUZXh0KGZpbGUpewogICAgcmV0dXJuIG5ldyBQcm9taXNlKChyZXNvbHZlLHJlamVjdCk9PnsKICAgICAgY29uc3QgciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHIub25sb2FkID0gKCk9PiByZXNvbHZlKFN0cmluZyhyLnJlc3VsdHx8JycpKTsKICAgICAgci5vbmVycm9yID0gKCk9PiByZWplY3QobmV3IEVycm9yKCdyZWFkIGZhaWxlZCcpKTsKICAgICAgci5yZWFkQXNUZXh0KGZpbGUpOwogICAgfSk7CiAgfQogIGZ1bmN0aW9uIHJlYWRGaWxlQXNEYXRhVVJMKGZpbGUpewogICAgcmV0dXJuIG5ldyBQcm9taXNlKChyZXNvbHZlLHJlamVjdCk9PnsKICAgICAgY29uc3QgciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHIub25sb2FkID0gKCk9PiByZXNvbHZlKFN0cmluZyhyLnJlc3VsdHx8JycpKTsKICAgICAgci5vbmVycm9yID0gKCk9PiByZWplY3QobmV3IEVycm9yKCdyZWFkIGZhaWxlZCcpKTsKICAgICAgci5yZWFkQXNEYXRhVVJMKGZpbGUpOwogICAgfSk7CiAgfQoKICBhc3luYyBmdW5jdGlvbiBhZGRBdHRhY2htZW50c0Zyb21GaWxlTGlzdChmaWxlTGlzdCl7CiAgICBjb25zdCBmaWxlcyA9IEFycmF5LmZyb20oZmlsZUxpc3QgfHwgW10pOwogICAgaWYgKCFmaWxlcy5sZW5ndGgpIHJldHVybjsKCiAgICBmb3IgKGNvbnN0IGZpbGUgb2YgZmlsZXMuc2xpY2UoMCwgTWF0aC5tYXgoMCwgTUFYX1RPVEFMX0FUVEFDSE1FTlRTIC0gYXR0YWNobWVudHMubGVuZ3RoKSkpewogICAgICBjb25zdCBpZCA9IG5leHRBdHRhY2hJZCsrOwogICAgICBjb25zdCB0eXBlID0gZmlsZS50eXBlIHx8ICcnOwogICAgICBjb25zdCBuYW1lID0gZmlsZS5uYW1lIHx8IGBmaWxlXyR7aWR9YDsKICAgICAgY29uc3Qgc2l6ZSA9IGZpbGUuc2l6ZSB8fCAwOwoKICAgICAgaWYgKHR5cGUuc3RhcnRzV2l0aCgnaW1hZ2UvJykpewogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5re75LuYXSDnlLvlg4/jga/jgrXjg53jg7zjg4jjgZXjgozjgabjgYTjgb7jgZvjgpPvvIgtLW1tcHJvaiDjgarjgZfvvInjgILjg4bjgq3jgrnjg4jjg5XjgqHjgqTjg6vjga7jgb/mt7vku5jlj6/og73jgafjgZnjgIJgKTsKICAgICAgICBjb250aW51ZTsKICAgICAgfQoKICAgICAgY29uc3QgbG9va3NUZXh0ID0gdHlwZS5zdGFydHNXaXRoKCd0ZXh0LycpCiAgICAgICAgfHwgL1wuKHR4dHxtZHxqc29ufGpzb25sfGNzdnx0c3Z8bG9nfHlhbWx8eW1sfHB5fGpzfHRzfGh0bWx8Y3NzfHhtbHxpbml8dG9tbCkkL2kudGVzdChuYW1lKTsKCiAgICAgIGlmIChsb29rc1RleHQpewogICAgICAgIHRyeXsKICAgICAgICAgIGxldCB0ZXh0ID0gYXdhaXQgcmVhZEZpbGVBc1RleHQoZmlsZSk7CiAgICAgICAgICBsZXQgdHJ1bmNhdGVkID0gZmFsc2U7CiAgICAgICAgICBpZiAodGV4dC5sZW5ndGggPiBNQVhfVEVYVF9DSEFSU19QRVJfRklMRSl7CiAgICAgICAgICAgIHRleHQgPSB0ZXh0LnNsaWNlKDAsIE1BWF9URVhUX0NIQVJTX1BFUl9GSUxFKTsKICAgICAgICAgICAgdHJ1bmNhdGVkID0gdHJ1ZTsKICAgICAgICAgIH0KICAgICAgICAgIGF0dGFjaG1lbnRzLnB1c2goeyBpZCwgbmFtZSwgdHlwZSwgc2l6ZSwga2luZDondGV4dCcsIHRleHQsIHRydW5jYXRlZCB9KTsKICAgICAgICB9IGNhdGNoIHsKICAgICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5re75LuYXSDjg4bjgq3jgrnjg4jjg5XjgqHjgqTjg6vjgpLoqq3jgb/ovrzjgoHjgb7jgZvjgpPjgafjgZfjgZ86ICR7bmFtZX1gKTsKICAgICAgICB9CiAgICAgICAgY29udGludWU7CiAgICAgIH0KCiAgICAgIGF0dGFjaG1lbnRzLnB1c2goeyBpZCwgbmFtZSwgdHlwZSwgc2l6ZSwga2luZDonb3RoZXInIH0pOwogICAgfQoKICAgIGlmIChhdHRhY2htZW50cy5sZW5ndGggPj0gTUFYX1RPVEFMX0FUVEFDSE1FTlRTKSB7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5re75LuYXSDkuIrpmZDjgavpgZTjgZfjgb7jgZfjgZ8gKCR7TUFYX1RPVEFMX0FUVEFDSE1FTlRTfSBmaWxlcykuYCk7CiAgICB9CiAgICByZW5kZXJBdHRhY2htZW50cygpOwogIH0KCiAgbGV0IGFib3J0Q29udHJvbGxlciA9IG51bGw7CiAgY29uc3QgTkwgPSAnXG4nOwoKICBjb25zdCBBUFBfVkVSU0lPTiA9ICcxLjAxJzsKICBjb25zdCBQUk9KRUNUX1VSTCA9ICdodHRwczovL3d3dy5sb2NhbGxpZ2h0YWkuY29tL3NjdSc7CiAgY29uc3QgSU1QUklOVF9VUkwgPSAnaHR0cHM6Ly93d3cuc3J3YXJlLm5ldC9pbXByaW50JzsKICAKICBsZXQgc3RyZWFtU3RhcnRUaW1lID0gMDsKICBsZXQgbGFzdFVwZGF0ZVRpbWUgPSAwOwogIGxldCBmaW5hbENvbXBsZXRpb25Ub2tlbnMgPSAwOwogIGNvbnN0IFVQREFURV9JTlRFUlZBTCA9IDIwMDsgLy8gbXMKCiAgLyogLS0tLS0tLS0tLSBBdXRvLVNjcm9sbCAoZ2VmbGlja3QpIC0tLS0tLS0tLS0gKi8KICBsZXQgYXV0b3Njcm9sbCA9IHRydWU7CiAgbGV0IGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgY29uc3QgQk9UVE9NX0VQUyA9IDgwOwoKICBmdW5jdGlvbiBhdEJvdHRvbSgpewogICAgY29uc3QgcyA9IGVscy5zY3JvbGxlcjsKICAgIHJldHVybiAocy5zY3JvbGxIZWlnaHQgLSBzLnNjcm9sbFRvcCAtIHMuY2xpZW50SGVpZ2h0KSA8PSBCT1RUT01fRVBTOwogIH0KICBmdW5jdGlvbiBmb3JjZVNjcm9sbFRvQm90dG9tKCl7CiAgICBjb25zdCBzID0gZWxzLnNjcm9sbGVyOwogICAgY29uc3QgcHJldiA9IHMuc3R5bGUuc2Nyb2xsQmVoYXZpb3I7CiAgICBzLnN0eWxlLnNjcm9sbEJlaGF2aW9yID0gJ2F1dG8nOwogICAgcy5zY3JvbGxUb3AgPSBzLnNjcm9sbEhlaWdodDsKICAgIHMuc3R5bGUuc2Nyb2xsQmVoYXZpb3IgPSBwcmV2IHx8ICcnOwogIH0KICBmdW5jdGlvbiBtYXliZVNjcm9sbChmb3JjZT1mYWxzZSl7CiAgICBpZiAoZm9yY2UgfHwgYXV0b3Njcm9sbCB8fCBhdEJvdHRvbSgpKSBmb3JjZVNjcm9sbFRvQm90dG9tKCk7CiAgICBjb25zdCBzaG93ID0gIWF0Qm90dG9tKCk7CiAgICBlbHMudG9Cb3R0b20uY2xhc3NMaXN0LnRvZ2dsZSgnc2hvdycsIHNob3cpOwogICAgaWYgKCFzaG93KSBhdXRvc2Nyb2xsID0gdHJ1ZTsKICB9CiAgZWxzLnNjcm9sbGVyLmFkZEV2ZW50TGlzdGVuZXIoJ3Njcm9sbCcsICgpID0+IHsKICAgIGlmIChpc1JlbmRlcmluZykgcmV0dXJuOwogICAgYXV0b3Njcm9sbCA9IGF0Qm90dG9tKCk7CiAgICBlbHMudG9Cb3R0b20uY2xhc3NMaXN0LnRvZ2dsZSgnc2hvdycsICFhdXRvc2Nyb2xsKTsKICB9KTsKICBlbHMudG9Cb3R0b20uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7IGF1dG9zY3JvbGwgPSB0cnVlOyBtYXliZVNjcm9sbCh0cnVlKTsgfSk7CgogIC8qIC0tLS0tLS0tLS0gTWFya2Rvd24gLS0tLS0tLS0tLSAqLwogIGNvbnN0IGVzYyA9IHMgPT4gU3RyaW5nKHMpLnJlcGxhY2UoL1smPD4iJ10vZywgYyA9PiAoeycmJzonJmFtcDsnLCc8JzonJmx0OycsJz4nOicmZ3Q7JywnIic6JyZxdW90OycsIiciOicmIzM5Oyd9W2NdKSk7CiAgZnVuY3Rpb24gaW5saW5lTWQocyl7CiAgICAvLyBQcm90ZWN0IGlubGluZSBjb2RlIHdpdGggcGxhY2Vob2xkZXJzIGZpcnN0CiAgICBjb25zdCBjb2RlUGxhY2Vob2xkZXJzID0gW107CiAgICBzID0gcy5yZXBsYWNlKC9gKFteYF0rKWAvZywgKF8scCkgPT4geyBjb25zdCBpZHggPSBjb2RlUGxhY2Vob2xkZXJzLmxlbmd0aDsgY29kZVBsYWNlaG9sZGVycy5wdXNoKHApOyByZXR1cm4gYFx4MDBDT0RFJHtpZHh9XHgwMGA7IH0pOwogICAgLy8gSW1hZ2VzCiAgICAvLyDlpJbpg6jnlLvlg4/jga/oh6rli5Xoqq3jgb/ovrzjgb/jgZfjgarjgYTvvIjplrLopqfjga7ov73ot6Hjgavkvb/jgYjjgovvvInjgIJodHRwcy9odHRwIOOBruOBv+ODquODs+OCr+OBqOOBl+OBpuWHuuOBmeOAggogICAgcyA9IHMucmVwbGFjZSgvIVxbKFteXF1dKilcXVwoKGh0dHBzPzpbXilcc10rKVwpL2csIChfLGFsdCx1cmwpPT4nPGEgaHJlZj0iJyt1cmwrJyIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIgbm9yZWZlcnJlciI+W+eUu+WDjzogJysoYWx0fHx1cmwpKyddPC9hPicpOwogICAgLy8gQm9sZAogICAgcyA9IHMucmVwbGFjZSgvXCpcKihbXipdKylcKlwqL2csIChfLHApPT4nPHN0cm9uZz4nK3ArJzwvc3Ryb25nPicpOwogICAgLy8gSXRhbGljIChvbmx5IG91dHNpZGUgY29kZSkKICAgIHMgPSBzLnJlcGxhY2UoL1wqKFteKl0rKVwqL2csIChfLHApPT4nPGVtPicrcCsnPC9lbT4nKTsKICAgIC8vIExpbmtzCiAgICBzID0gcy5yZXBsYWNlKC9cWyhbXlxdXSspXF1cKChodHRwcz86W15cKVxzXSspW15cKV0qXCkvZywgKF8sdHh0LHVybCk9Pic8YSBocmVmPSInK3VybCsnIiB0YXJnZXQ9Il9ibGFuayIgcmVsPSJub29wZW5lciI+Jyt0eHQrJzwvYT4nKTsKICAgIC8vIEF1dG8tbGluayBiYXJlIFVSTHMgKHNraXAgYWxyZWFkeS1nZW5lcmF0ZWQgSFRNTCB0YWdzKQogICAgcyA9IHMucmVwbGFjZSgvKDxbXj5dKj4pfChcYmh0dHBzPzpcL1wvW15ccynvvInjgI3jgI/jgILjgIFdKykvZywgKG1hdGNoLCB0YWcsIHVybCkgPT4gewogICAgICBpZiAodGFnKSByZXR1cm4gdGFnOwogICAgICByZXR1cm4gJzxhIGhyZWY9IicrdXJsKyciIHRhcmdldD0iX2JsYW5rIiByZWw9Im5vb3BlbmVyIj4nK3VybCsnPC9hPic7CiAgICB9KTsKICAgIC8vIFJlc3RvcmUgY29kZQogICAgcyA9IHMucmVwbGFjZSgvXHgwMENPREUoXGQrKVx4MDAvZywgKF8saWR4KSA9PiAnPGNvZGU+Jytjb2RlUGxhY2Vob2xkZXJzW3BhcnNlSW50KGlkeCldKyc8L2NvZGU+Jyk7CiAgICByZXR1cm4gczsKICB9CiAgZnVuY3Rpb24gbWRUb0h0bWwobWQpewogICAgbWQgPSBTdHJpbmcobWQgPz8gJycpOwogICAgbWQgPSBtZC5yZXBsYWNlKC9cci9nLCcnKTsKICAgIGNvbnN0IGxpbmVzID0gbWQuc3BsaXQoJ1xuJyk7CiAgICBsZXQgaHRtbCA9ICcnLCBpbkNvZGUgPSBmYWxzZSwgY29kZUxhbmcgPSAnJywgbGlzdFR5cGUgPSBudWxsOwogICAgY29uc3QgY2xvc2VMaXN0ID0gKCkgPT4geyBpZiAobGlzdFR5cGUpIHsgaHRtbCArPSBgPC8ke2xpc3RUeXBlfT5gOyBsaXN0VHlwZT1udWxsOyB9IH07CiAgICBmb3IgKGxldCBpPTA7aTxsaW5lcy5sZW5ndGg7aSsrKXsKICAgICAgbGV0IGxpbmUgPSBsaW5lc1tpXTsKICAgICAgaWYgKCFpbkNvZGUgJiYgL15ccypgYGAvLnRlc3QobGluZSkpewogICAgICAgIGNsb3NlTGlzdCgpOwogICAgICAgIGNvbnN0IG0gPSBsaW5lLm1hdGNoKC9eXHMqYGBgXHMqKFtcdystXSopLyk7CiAgICAgICAgY29kZUxhbmcgPSBtICYmIG1bMV0gPyBtWzFdIDogJyc7CiAgICAgICAgaHRtbCArPSBgPGRpdiBjbGFzcz0iY29kZXdyYXAiPjxwcmU+PGNvZGUgZGF0YS1sYW5nPSIke2VzYyhjb2RlTGFuZyl9Ij5gOwogICAgICAgIGluQ29kZSA9IHRydWU7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmIChpbkNvZGUpewogICAgICAgIGlmICgvXlxzKmBgYFxzKiQvLnRlc3QobGluZSkpeyBodG1sICs9IGA8L2NvZGU+PC9wcmU+PGJ1dHRvbiBjbGFzcz0iY29weWJ0biI+Q29weTwvYnV0dG9uPjwvZGl2PmA7IGluQ29kZT1mYWxzZTsgY29udGludWU7IH0KICAgICAgICBodG1sICs9IGVzYyhsaW5lKSArICdcbic7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGNvbnN0IGggPSBsaW5lLm1hdGNoKC9eXHMqKCN7MSw2fSlccysoLiopJC8pOwogICAgICBpZiAoaCl7IGNsb3NlTGlzdCgpOyBjb25zdCBsdmw9aFsxXS5sZW5ndGg7IGh0bWwgKz0gYDxoJHtsdmx9PiR7aW5saW5lTWQoZXNjKGhbMl0pKX08L2gke2x2bH0+YDsgY29udGludWU7IH0KICAgICAgaWYgKC9eXHMqKD86LS0tfFwqXCpcKnxfX18pXHMqJC8udGVzdChsaW5lKSl7IGNsb3NlTGlzdCgpOyBodG1sICs9ICc8aHIvPic7IGNvbnRpbnVlOyB9CiAgICAgIGlmICgvXlxzKj5ccz8vLnRlc3QobGluZSkpewogICAgICAgIGNsb3NlTGlzdCgpOwogICAgICAgIGNvbnN0IHF1b3RlTGluZXMgPSBbXTsKICAgICAgICB3aGlsZSAoaSA8IGxpbmVzLmxlbmd0aCAmJiAvXlxzKj5ccz8vLnRlc3QobGluZXNbaV0pKSB7CiAgICAgICAgICBxdW90ZUxpbmVzLnB1c2gobGluZXNbaV0ucmVwbGFjZSgvXlxzKj5ccz8vLCAnJykpOwogICAgICAgICAgaSsrOwogICAgICAgIH0KICAgICAgICBpLS07CiAgICAgICAgY29uc3QgcXVvdGVUZXh0ID0gaW5saW5lTWQoZXNjKHF1b3RlTGluZXMuam9pbignXG4nKSkpOwogICAgICAgIGh0bWwgKz0gYDxibG9ja3F1b3RlPjxwPiR7cXVvdGVUZXh0LnJlcGxhY2UoL1xuL2csICc8YnIgLz4nKX08L3A+PC9ibG9ja3F1b3RlPmA7CiAgICAgICAgY29udGludWU7CiAgICAgIH0KICAgICAgCiAgICAgIC8vIC0tLSBTVEFSVDogVGFibGUgUGFyc2luZyAtLS0KICAgICAgY29uc3QgaXNUYWJsZUxpbmUgPSAobCkgPT4gbC5pbmNsdWRlcygnfCcpOwogICAgICBjb25zdCBpc1NlcGFyYXRvckxpbmUgPSAobCkgPT4gL15ccypcfD8oWy06XStcfCkrWy06XStcfD9ccyokLy50ZXN0KGwudHJpbSgpKTsKICAgICAgCiAgICAgIGlmIChpc1RhYmxlTGluZShsaW5lKSAmJiBpICsgMSA8IGxpbmVzLmxlbmd0aCAmJiBpc1NlcGFyYXRvckxpbmUobGluZXNbaSArIDFdKSkgewogICAgICAgIGNsb3NlTGlzdCgpOwogICAgICAgIGh0bWwgKz0gJzx0YWJsZT4nOwogICAgICAgIAogICAgICAgIC8vIEhlYWRlcgogICAgICAgIGNvbnN0IGhlYWRlcnMgPSBsaW5lLnNwbGl0KCd8JykubWFwKGggPT4gaC50cmltKCkpLnNsaWNlKDEsIC0xKTsKICAgICAgICBodG1sICs9ICc8dGhlYWQ+PHRyPic7CiAgICAgICAgaGVhZGVycy5mb3JFYWNoKGggPT4geyBodG1sICs9IGA8dGg+JHtpbmxpbmVNZChlc2MoaCkpfTwvdGg+YDsgfSk7CiAgICAgICAgaHRtbCArPSAnPC90cj48L3RoZWFkPic7CiAgICAgICAgCiAgICAgICAgaSArPSAyOyAvLyBTa2lwIGhlYWRlciB1bmQgc2VwYXJhdG9yIGxpbmUKICAgICAgICAKICAgICAgICAvLyBCb2R5CiAgICAgICAgaHRtbCArPSAnPHRib2R5Pic7CiAgICAgICAgd2hpbGUgKGkgPCBsaW5lcy5sZW5ndGggJiYgaXNUYWJsZUxpbmUobGluZXNbaV0pKSB7CiAgICAgICAgICAgIGNvbnN0IGNlbGxzID0gbGluZXNbaV0uc3BsaXQoJ3wnKS5tYXAoYyA9PiBjLnRyaW0oKSkuc2xpY2UoMSwgLTEpOwogICAgICAgICAgICBpZiAoY2VsbHMubGVuZ3RoID4gMCkgewogICAgICAgICAgICAgICAgaHRtbCArPSAnPHRyPic7CiAgICAgICAgICAgICAgICBmb3IgKGxldCBqID0gMDsgaiA8IGhlYWRlcnMubGVuZ3RoOyBqKyspIHsKICAgICAgICAgICAgICAgICAgICBjb25zdCBjZWxsQ29udGVudCA9IGNlbGxzW2pdIHx8ICcnOwogICAgICAgICAgICAgICAgICAgIGh0bWwgKz0gYDx0ZD4ke2lubGluZU1kKGVzYyhjZWxsQ29udGVudCkpfTwvdGQ+YDsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGh0bWwgKz0gJzwvdHI+JzsKICAgICAgICAgICAgfQogICAgICAgICAgICBpKys7CiAgICAgICAgfQogICAgICAgIGktLTsKICAgICAgICAKICAgICAgICBodG1sICs9ICc8L3Rib2R5PjwvdGFibGU+JzsKICAgICAgICBjb250aW51ZTsKICAgICAgfQogICAgICAvLyAtLS0gRU5EOiBUYWJsZSBQYXJzaW5nIC0tLQoKICAgICAgbGV0IG07CiAgICAgIGlmIChtID0gbGluZS5tYXRjaCgvXlxzKlstKitdXHMrXFsoIHx4fFgpXF1ccysoLiopJC8pKXsKICAgICAgICBjb25zdCBjaGVja2VkID0gL3gvaS50ZXN0KG1bMV0pOwogICAgICAgIGlmIChsaXN0VHlwZSE9PSd1bCcpeyBjbG9zZUxpc3QoKTsgbGlzdFR5cGU9J3VsJzsgaHRtbCArPSAnPHVsPic7IH0KICAgICAgICBodG1sICs9IGA8bGkgY2xhc3M9IiR7Y2hlY2tlZD8ndGFzay1kb25lJzonJ30iPmArKGNoZWNrZWQ/J+KchSAnOifirJwgJykraW5saW5lTWQoZXNjKG1bMl0pKSsnPC9saT4nOyBjb250aW51ZTsKICAgICAgfQogICAgICBpZiAoL15ccypbLSorXVxzKy8udGVzdChsaW5lKSl7CiAgICAgICAgaWYgKGxpc3RUeXBlIT09J3VsJyl7IGNsb3NlTGlzdCgpOyBsaXN0VHlwZT0ndWwnOyBodG1sICs9ICc8dWw+JzsgfQogICAgICAgIGh0bWwgKz0gYDxsaT4ke2lubGluZU1kKGVzYyhsaW5lLnJlcGxhY2UoL15ccypbLSorXVxzKy8sJycpKSl9PC9saT5gOyBjb250aW51ZTsKICAgICAgfQogICAgICBpZiAobSA9IGxpbmUubWF0Y2goL15ccyooXGQrKVwuXHMrKC4qKSQvKSl7CiAgICAgICAgaWYgKGxpc3RUeXBlIT09J29sJyl7IGNsb3NlTGlzdCgpOyBsaXN0VHlwZT0nb2wnOyBodG1sICs9ICc8b2w+JzsgfQogICAgICAgIGh0bWwgKz0gYDxsaT4ke2lubGluZU1kKGVzYyhtWzJdKSl9PC9saT5gOyBjb250aW51ZTsKICAgICAgfQogICAgICBpZiAoL15ccyokLy50ZXN0KGxpbmUudHJpbSgpKSl7IGNsb3NlTGlzdCgpOyB9CiAgICAgIGVsc2UgeyBjbG9zZUxpc3QoKTsgaHRtbCArPSBgPHA+JHtpbmxpbmVNZChlc2MobGluZSkpfTwvcD5gOyB9CiAgICB9CiAgICBjbG9zZUxpc3QoKTsgcmV0dXJuIGh0bWw7CiAgfQogIGZ1bmN0aW9uIGVuaGFuY2VDb2RlQ29weShjb250YWluZXIpewogICAgY29udGFpbmVyLnF1ZXJ5U2VsZWN0b3JBbGwoJy5jb2Rld3JhcCAuY29weWJ0bjpub3QoW2RhdGEtaW5pdF0pJykuZm9yRWFjaChidG49PnsKICAgICAgYnRuLmRhdGFzZXQuaW5pdCA9ICcxJzsKICAgICAgYnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCk9PnsKICAgICAgICBjb25zdCBjb2RlID0gYnRuLnBhcmVudEVsZW1lbnQucXVlcnlTZWxlY3RvcignY29kZScpOwogICAgICAgIGNvbnN0IHRleHQgPSBjb2RlID8gY29kZS50ZXh0Q29udGVudCA6ICcnOwogICAgICAgIG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KHRleHQpLnRoZW4oKCk9PnsgYnRuLnRleHRDb250ZW50PSdDb3BpZWQnOyBzZXRUaW1lb3V0KCgpPT5idG4udGV4dENvbnRlbnQ9J0NvcHknLDEwMDApOyB9KTsKICAgICAgfSk7CiAgICB9KTsKICB9CgogIC8qIC0tLS0tLS0tLS0gVUkgaGVscGVycyAtLS0tLS0tLS0tICovCiAgZnVuY3Rpb24gcmVuZGVyQnViYmxlQ29udGVudChlbCwgcm9sZSwgY29udGVudCl7CiAgICBlbC5jbGFzc0xpc3QuYWRkKCdtZCcpOyBlbC5pbm5lckhUTUwgPSBtZFRvSHRtbChub3JtYWxpemVDb250ZW50KGNvbnRlbnQpKTsgZW5oYW5jZUNvZGVDb3B5KGVsKTsKICB9CiAgCiAgZnVuY3Rpb24gYWRkTWVzc2FnZShyb2xlLCBjb250ZW50KSB7CiAgICBjb25zdCBjbHMgPSByb2xlID09PSAndXNlcicgPyAndXNlcicgOiByb2xlID09PSAnc3lzdGVtJyA/ICdzeXMnIDogJ2JvdCc7CiAgICBjb25zdCBhdmF0YXIgPSByb2xlID09PSAndXNlcicgPyAn8J+nkScgOiByb2xlID09PSAnc3lzdGVtJyA/ICfimpnvuI8nIDogJ/CfpJYnOwogICAgY29uc3QgbXNnRWwgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTsKICAgIG1zZ0VsLmNsYXNzTmFtZSA9IGBtc2cgJHtjbHN9YDsKICAgIG1zZ0VsLmlubmVySFRNTCA9IGAKICAgICAgPGRpdiBjbGFzcz0iYXZhdGFyIj4ke2F2YXRhcn08L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibWVzc2FnZS1jb250ZW50LXdyYXBwZXIiPgogICAgICAgIDxkaXYgY2xhc3M9ImJ1YmJsZSI+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJtZXRhIj4ke3JvbGV9PC9kaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJjb250ZW50IG1kIj48L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+YDsKICAgIGVscy5sb2cuYXBwZW5kQ2hpbGQobXNnRWwpOwogICAgY29uc3QgY29udGVudEVsID0gbXNnRWwucXVlcnlTZWxlY3RvcignLmNvbnRlbnQnKTsKICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQoY29udGVudEVsLCByb2xlLCBjb250ZW50KTsKICAgIGlmICghaXNSZW5kZXJpbmcpIG1heWJlU2Nyb2xsKCk7CiAgICByZXR1cm4gY29udGVudEVsOwogIH0KICAKICBmdW5jdGlvbiBhZGRUaGlua0J1YmJsZShhc3Npc3RhbnRDb250ZW50RWwpIHsKICAgIGNvbnN0IHdyYXBwZXIgPSBhc3Npc3RhbnRDb250ZW50RWwuY2xvc2VzdCgnLm1lc3NhZ2UtY29udGVudC13cmFwcGVyJyk7CiAgICBpZiAoIXdyYXBwZXIpIHJldHVybiBudWxsOwogICAgY29uc3QgYnViYmxlID0gd3JhcHBlci5xdWVyeVNlbGVjdG9yKCcuYnViYmxlJyk7CiAgICBpZiAoIWJ1YmJsZSkgcmV0dXJuIG51bGw7CgogICAgY29uc3QgdGhpbmtFbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgdGhpbmtFbC5jbGFzc05hbWUgPSAndGhpbmstYnViYmxlJzsKICAgIHRoaW5rRWwuaW5uZXJIVE1MID0gYAogICAgICA8ZGV0YWlscz4KICAgICAgICA8c3VtbWFyeT7wn6SUIFNob3cgdGhpbmtpbmc8L3N1bW1hcnk+CiAgICAgICAgPGRpdiBjbGFzcz0idGhpbmstY29udGVudCBtZCI+PC9kaXY+CiAgICAgIDwvZGV0YWlscz4KICAgIGA7CiAgICB3cmFwcGVyLmluc2VydEJlZm9yZSh0aGlua0VsLCBidWJibGUpOwogICAgcmV0dXJuIHRoaW5rRWwucXVlcnlTZWxlY3RvcignLnRoaW5rLWNvbnRlbnQnKTsKICB9CgogIGZ1bmN0aW9uIHNldFN0cmVhbWluZ1N0YXRlKGFjdGl2ZSkgeyBlbHMuc2VuZEJ0bi5kaXNhYmxlZCA9IGFjdGl2ZTsgZWxzLnN0b3BCdG4uZGlzYWJsZWQgPSAhYWN0aXZlOyB9CgogIGZ1bmN0aW9uIHJlbmRlckFsbCgpewogICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgZWxzLmxvZy5pbm5lckhUTUwgPSAnJzsKICAgIGZvciAoY29uc3QgbSBvZiBtZXNzYWdlcykgYWRkTWVzc2FnZShtLnJvbGUsIG0uY29udGVudCk7CiAgICBmb3JjZVNjcm9sbFRvQm90dG9tKCk7CiAgICBpc1JlbmRlcmluZyA9IGZhbHNlOwogICAgYXV0b3Njcm9sbCA9IHRydWU7CiAgICBlbHMudG9Cb3R0b20uY2xhc3NMaXN0LnJlbW92ZSgnc2hvdycpOwogIH0KCiAgLyogLS0tLS0tLS0tLSBOb3JtYWxpemVyIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBub3JtYWxpemVDb250ZW50KGMpewogICAgaWYgKHR5cGVvZiBjID09PSAnc3RyaW5nJykgcmV0dXJuIGM7CiAgICBpZiAoQXJyYXkuaXNBcnJheShjKSkgewogICAgICByZXR1cm4gYy5tYXAocCA9PiB7CiAgICAgICAgaWYgKHR5cGVvZiBwID09PSAnc3RyaW5nJykgcmV0dXJuIHA7CiAgICAgICAgaWYgKHA/LnR5cGUgPT09ICd0ZXh0JykgcmV0dXJuIHAudGV4dCA/PyAnJzsKICAgICAgICBpZiAocD8udHlwZSA9PT0gJ2ltYWdlX3VybCcpIHsKICAgICAgICAgIGNvbnN0IHUgPSBwPy5pbWFnZV91cmw/LnVybCB8fCAnJzsKICAgICAgICAgIHJldHVybiB1ID8gYCFbaW1hZ2VdKCR7dX0pYCA6ICcnOwogICAgICAgIH0KICAgICAgICByZXR1cm4gcD8udGV4dCA/PyAnJzsKICAgICAgfSkuam9pbihTdHJpbmcuZnJvbUNoYXJDb2RlKDEwKSk7CiAgICB9CiAgICBpZiAoYyAmJiB0eXBlb2YgYyA9PT0gJ29iamVjdCcgJiYgJ3RleHQnIGluIGMpIHJldHVybiBTdHJpbmcoYy50ZXh0ID8/ICcnKTsKICAgIHRyeSB7IHJldHVybiBKU09OLnN0cmluZ2lmeShjKTsgfSBjYXRjaCB7IHJldHVybiBTdHJpbmcoYyk7IH0KICB9CiAgZnVuY3Rpb24gbm9ybWFsaXplUm9sZShyKXsgY29uc3Qgcz1TdHJpbmcocnx8JycpLnRvTG93ZXJDYXNlKCk7IGlmIChzLnN0YXJ0c1dpdGgoJ2Fzc2lzdCcpKSByZXR1cm4gJ2Fzc2lzdGFudCc7IGlmIChzLnN0YXJ0c1dpdGgoJ3N5cycpKSByZXR1cm4gJ3N5c3RlbSc7IHJldHVybiAndXNlcic7IH0KCiAgLyogLS0tLS0tLS0tLSBFeHBvcnQgaW5rbC4gc3lzdGVtIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBidWlsZEV4cG9ydEFycmF5KCl7CiAgICBjb25zdCBvdXQgPSBtZXNzYWdlcy5tYXAobSA9PiAoeyByb2xlOiBtLnJvbGUsIGNvbnRlbnQ6IG0uY29udGVudCB9KSk7CiAgICBjb25zdCBzeXNUZXh0ID0gKGVscy5zeXN0ZW0udmFsdWUgfHwgJycpLnRyaW0oKTsKICAgIGNvbnN0IGZpcnN0SXNTeXN0ZW0gPSBvdXRbMF0/LnJvbGUgPT09ICdzeXN0ZW0nOwogICAgaWYgKHN5c1RleHQpIHsKICAgICAgaWYgKCFmaXJzdElzU3lzdGVtKSBvdXQudW5zaGlmdCh7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNUZXh0IH0pOwogICAgICBlbHNlIGlmIChvdXRbMF0uY29udGVudCAhPT0gc3lzVGV4dCkgb3V0WzBdID0geyByb2xlOiAnc3lzdGVtJywgY29udGVudDogc3lzVGV4dCB9OwogICAgfQogICAgcmV0dXJuIG91dDsKICB9CiAgICAKICBmdW5jdGlvbiB1cGRhdGVUb2tlblJhdGUoYnVmZmVyLCBmb3JjZUZpbmFsID0gZmFsc2UpIHsKICAgIGNvbnN0IG5vdyA9IHBlcmZvcm1hbmNlLm5vdygpOwogICAgaWYgKCFmb3JjZUZpbmFsICYmIG5vdyAtIGxhc3RVcGRhdGVUaW1lIDwgVVBEQVRFX0lOVEVSVkFMKSByZXR1cm47CiAgICBsYXN0VXBkYXRlVGltZSA9IG5vdzsKCiAgICBjb25zdCBlbGFwc2VkU2Vjb25kcyA9IChub3cgLSBzdHJlYW1TdGFydFRpbWUpIC8gMTAwMDsKICAgIGlmIChlbGFwc2VkU2Vjb25kcyA8IDAuMSkgcmV0dXJuOwoKICAgIGxldCB0b2tlbnMsIHJhdGUsIGxhYmVsOwogICAgaWYgKGZpbmFsQ29tcGxldGlvblRva2VucyA+IDApIHsKICAgICAgICB0b2tlbnMgPSBmaW5hbENvbXBsZXRpb25Ub2tlbnM7CiAgICAgICAgcmF0ZSA9IHRva2VucyAvIGVsYXBzZWRTZWNvbmRzOwogICAgICAgIGxhYmVsID0gYMOYICR7cmF0ZS50b0ZpeGVkKDEpfSB0L3NgOwogICAgfSBlbHNlIHsKICAgICAgICAvLyBTY2jDpHR6dW5nOiBjYS4gNCBaZWljaGVuIHBybyBUb2tlbgogICAgICAgIHRva2VucyA9IE1hdGgucm91bmQoYnVmZmVyLmxlbmd0aCAvIDQpOwogICAgICAgIHJhdGUgPSB0b2tlbnMgLyBlbGFwc2VkU2Vjb25kczsKICAgICAgICBsYWJlbCA9IGB+JHtyYXRlLnRvRml4ZWQoMSl9IHQvc2A7CiAgICB9CiAgICBlbHMudG9rZW5SYXRlLnRleHRDb250ZW50ID0gbGFiZWw7CiAgfQoKCiAgLyogLS0tLS0tLS0tLSBLYWdnbGUgR1BVIOaui+OCiuaZgumWkyAtLS0tLS0tLS0tICovCiAgLy8g5Lit57aZ44GuIC9xdW90YSDjgpLlj6njgY/jgILjg63jg7zjgqvjg6sgdXp1IOOBq+OBryAvcXVvdGEg44GM54Sh44GE44Gu44Gn44CBCiAgLy8g44Gd44Gu5aC05ZCI44GvIDQwNCDjgYzov5TjgorjgIHooajnpLrjga/lh7rjgarjgYTvvIjjgqjjg6njg7zmibHjgYTjgavjgZfjgarjgYTvvInjgIIKICBsZXQgcXVvdGFUaW1lciA9IG51bGw7CgogIGZ1bmN0aW9uIGZtdExlZnQoc2VjKSB7CiAgICBpZiAoc2VjID09IG51bGwgfHwgc2VjIDwgMCkgcmV0dXJuICcnOwogICAgY29uc3QgaCA9IE1hdGguZmxvb3Ioc2VjIC8gMzYwMCk7CiAgICBjb25zdCBtID0gTWF0aC5mbG9vcigoc2VjICUgMzYwMCkgLyA2MCk7CiAgICByZXR1cm4gYOaui+OCiiAke1N0cmluZyhoKS5wYWRTdGFydCgyLCAnMCcpfToke1N0cmluZyhtKS5wYWRTdGFydCgyLCAnMCcpfWA7CiAgfQoKICBhc3luYyBmdW5jdGlvbiByZWZyZXNoUXVvdGEoKSB7CiAgICBpZiAoIWVscy5xdW90YUxlZnQpIHJldHVybjsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKGAke2xvY2F0aW9uLm9yaWdpbn0vcXVvdGFgLCB7CiAgICAgICAgbWV0aG9kOiAnR0VUJywKICAgICAgICBoZWFkZXJzOiBhcGlIZWFkZXJzKHsgJ0FjY2VwdCc6ICdhcHBsaWNhdGlvbi9qc29uJyB9KSwKICAgICAgICBjYWNoZTogJ25vLXN0b3JlJywKICAgICAgfSk7CiAgICAgIGlmICghcmVzLm9rKSB7IGVscy5xdW90YUxlZnQudGV4dENvbnRlbnQgPSAnJzsgcmV0dXJuOyB9CiAgICAgIGNvbnN0IGQgPSBhd2FpdCByZXMuanNvbigpOwogICAgICBpZiAoIWQuYXZhaWxhYmxlKSB7IGVscy5xdW90YUxlZnQudGV4dENvbnRlbnQgPSAnJzsgcmV0dXJuOyB9CgogICAgICBjb25zdCBsZWZ0ID0gZC5sZWZ0U2Vjb25kczsKICAgICAgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9IGZtdExlZnQobGVmdCk7CiAgICAgIC8vIOaui+OCiuOBjOWwkeOBquOBhOOBqOiJsuOCkuWkieOBiOOCiwogICAgICBlbHMucXVvdGFMZWZ0LnN0eWxlLmNvbG9yID0gbGVmdCA8IDE4MDAgPyAnI2ZmNmI2YicKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA6IGxlZnQgPCA3MjAwID8gJyNlMGEwMzAnCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgOiAnJzsKICAgICAgZWxzLnF1b3RhTGVmdC50aXRsZSA9IGBLYWdnbGUgR1BVOiDmrovjgoogJHtmbXRMZWZ0KGxlZnQpLnJlcGxhY2UoJ+aui+OCiiAnLCAnJyl9XG5gCiAgICAgICAgICAgICAgICAgICAgICAgICAgKyBg5L2/55SoICR7TWF0aC5mbG9vcihkLnVzZWRTZWNvbmRzIC8gMzYwMCl9aCAvIOS4iumZkCAke01hdGguZmxvb3IoZC50b3RhbFNlY29uZHMgLyAzNjAwKX1oYAogICAgICAgICAgICAgICAgICAgICAgICAgICsgKGQucmVmcmVzaFRpbWUgPyBgXG7jg6rjgrvjg4Pjg4ggJHtkLnJlZnJlc2hUaW1lfWAgOiAnJyk7CiAgICB9IGNhdGNoIChfKSB7CiAgICAgIGVscy5xdW90YUxlZnQudGV4dENvbnRlbnQgPSAnJzsKICAgIH0KICB9CgogIGZ1bmN0aW9uIHN0YXJ0UXVvdGFQb2xsaW5nKCkgewogICAgcmVmcmVzaFF1b3RhKCk7CiAgICAvLyAxIOWIhuOBlOOBqOOAguaui+mHj+OBr+enkuWNmOS9jeOBp+a4m+OCi+OBjOOAgeihqOekuuOBr+WIhuWNmOS9jeOBquOBruOBp+WNgeWIhuOAggogICAgaWYgKHF1b3RhVGltZXIpIGNsZWFySW50ZXJ2YWwocXVvdGFUaW1lcik7CiAgICBxdW90YVRpbWVyID0gc2V0SW50ZXJ2YWwocmVmcmVzaFF1b3RhLCA2MDAwMCk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIOWBnOatouODnOOCv+ODs++8iOODreODvOOCq+ODq+eJiOOBruOBv++8iSAtLS0tLS0tLS0tICovCiAgLy8g5Lit57aZ44GuIC9zdG9wIOOCkuWPqeOBj+OAguS4ree2meOBjOiQveOBoeOCi+OBruOBp+OAgeOBk+OBriBVSSDjgoLkvb/jgYjjgarjgY/jgarjgovjgIIKICAvLyDjg63jg7zjgqvjg6sgdXp1IOOBq+OBryAvc3RvcCDjgYznhKHjgYTjga7jgafjgIE0MDQg44Gq44KJ44Oc44K/44Oz44KS6Zqg44GZ44CCCiAgYXN5bmMgZnVuY3Rpb24gZGV0ZWN0U3RvcFN1cHBvcnQoKSB7CiAgICBpZiAoIWVscy5zdG9wUm93KSByZXR1cm47CiAgICB0cnkgewogICAgICBjb25zdCByZXMgPSBhd2FpdCBmZXRjaChgJHtsb2NhdGlvbi5vcmlnaW59L3F1b3RhYCwgewogICAgICAgIG1ldGhvZDogJ0dFVCcsIGhlYWRlcnM6IGFwaUhlYWRlcnMoKSwgY2FjaGU6ICduby1zdG9yZScsCiAgICAgIH0pOwogICAgICAvLyAvcXVvdGEg44GMIEpTT04g44KS6L+U44GZID0g5Lit57aZ44GM56uL44Gj44Gm44GE44KLID0g5YGc5q2i44Gn44GN44KL44CCCiAgICAgIC8vIGF2YWlsYWJsZT1mYWxzZe+8iHV6dSDjgarjganjgq/jgqnjg7zjgr/nhKHjgZfvvInjgafjgoLkuK3ntpnoh6rkvZPjga/nlJ/jgY3jgabjgYTjgovjga7jgacKICAgICAgLy8g5YGc5q2i44Oc44K/44Oz44Gv5Ye644GZ44CCCiAgICAgIGlmIChyZXMub2spIHsKICAgICAgICBjb25zdCBkID0gYXdhaXQgcmVzLmpzb24oKTsKICAgICAgICBpZiAoZCAmJiB0eXBlb2YgZC5hdmFpbGFibGUgPT09ICdib29sZWFuJykgewogICAgICAgICAgZWxzLnN0b3BSb3cuc3R5bGUuZGlzcGxheSA9ICcnOwogICAgICAgIH0KICAgICAgfQogICAgfSBjYXRjaCAoXykgeyAvKiDkuK3ntpnjgafjgarjgYTjgarjgonlh7rjgZXjgarjgYQgKi8gfQogIH0KCiAgYXN5bmMgZnVuY3Rpb24gc3RvcFNlcnZlcigpIHsKICAgIGlmICghY29uZmlybSgn44K144O844OQ44O844KS5YGc5q2i44GX44G+44GZ44CC44KI44KN44GX44GE44Gn44GZ44GL77yfXG7vvIjjgZPjga7nlLvpnaLjga/kvb/jgYjjgarjgY/jgarjgorjgb7jgZnvvIknKSkgcmV0dXJuOwogICAgaWYgKGVscy5zdG9wU2VydmVyKSB7CiAgICAgIGVscy5zdG9wU2VydmVyLmRpc2FibGVkID0gdHJ1ZTsKICAgICAgZWxzLnN0b3BTZXJ2ZXIudGV4dENvbnRlbnQgPSAn5YGc5q2i5Lit4oCmJzsKICAgIH0KICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKGAke2xvY2F0aW9uLm9yaWdpbn0vc3RvcGAsIHsKICAgICAgICBtZXRob2Q6ICdQT1NUJywgaGVhZGVyczogYXBpSGVhZGVycyh7ICdYLUtjLVN0b3AnOiAnMScgfSksIGNhY2hlOiAnbm8tc3RvcmUnLAogICAgICB9KTsKICAgICAgaWYgKHJlcy5zdGF0dXMgPT09IDQwMSkgewogICAgICAgIG9wZW5BdXRoRGlhbG9nKCflgZzmraLjgavjga/oqo3oqLzjgYzlv4XopoHjgafjgZnvvIg0MDHvvIknKTsKICAgICAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSBmYWxzZTsgZWxzLnN0b3BTZXJ2ZXIudGV4dENvbnRlbnQgPSAn4pagIOWBnOatoic7IH0KICAgICAgICByZXR1cm47CiAgICAgIH0KICAgICAgaWYgKCFyZXMub2spIHsKICAgICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW+WBnOatol0g5aSx5pWX44GX44G+44GX44GfIChIVFRQICR7cmVzLnN0YXR1c30pYCk7CiAgICAgICAgaWYgKGVscy5zdG9wU2VydmVyKSB7IGVscy5zdG9wU2VydmVyLmRpc2FibGVkID0gZmFsc2U7IGVscy5zdG9wU2VydmVyLnRleHRDb250ZW50ID0gJ+KWoCDlgZzmraInOyB9CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICB9IGNhdGNoIChfKSB7IC8qIOS4ree2meOBjOiQveOBoeOCi+OBruOBp+mAmuS/oeOCqOODqeODvOOBr+aDs+WumuWGhSAqLyB9CiAgICBzZXRTdGF0dXMoJ+WBnOatoicpOwogICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgJ1vlgZzmraJdIOOCteODvOODkOODvOOCkuWBnOatouOBl+OBvuOBl+OBn+OAguOBk+OBrueUu+mdouOCkumWieOBmOOBpuOBj+OBoOOBleOBhOOAgicpOwogICAgaWYgKGVscy5zdG9wU2VydmVyKSBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICflgZzmraLmuIjjgb8nOwogIH0KCiAgLyogLS0tLS0tLS0tLSBBUEkgQmFzZSB2YWxpZGF0aW9uIChNaXhlZCBjb250ZW50IGd1YXJkKSAtLS0tLS0tLS0tICovCiAgZnVuY3Rpb24gaXNNaXhlZENvbnRlbnRBcGlCYXNlKCl7CiAgICBjb25zdCBwYWdlUHJvdG8gPSB3aW5kb3cubG9jYXRpb24ucHJvdG9jb2w7CiAgICBjb25zdCBiYXNlID0gKGVscy5hcGlCYXNlPy52YWx1ZSB8fCAnJykudHJpbSgpOwogICAgcmV0dXJuIHBhZ2VQcm90byA9PT0gJ2h0dHBzOicgJiYgL15odHRwOlwvXC8vaS50ZXN0KGJhc2UpOwogIH0KICBmdW5jdGlvbiBlbnN1cmVBcGlCYXNlQWxsb3dlZCgpewogICAgaWYgKGlzTWl4ZWRDb250ZW50QXBpQmFzZSgpKXsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywKICAgICAgICBgW0NvbmZpZ10gTWl4ZWQgY29udGVudCBibG9ja2VkOiB0aGlzIFVJIGlzIGxvYWRlZCB2aWEgSFRUUFMsIGJ1dCBBUEkgQmFzZSBpcyBIVFRQLlxuYCArCiAgICAgICAgYEJyb3dzZXJzIHdpbGwgdXN1YWxseSBibG9jayB0aGlzIHJlcXVlc3QuIFVzZSBhbiBodHRwczovLyBBUEkgQmFzZSwgb3Igb3BlbiB0aGUgVUkgdmlhIGh0dHA6Ly8gb3IgZmlsZTovLy5gCiAgICAgICk7CiAgICAgIHRyeXsgZWxzLmFwaUJhc2Uuc3R5bGUuYm9yZGVyQ29sb3IgPSAnI2I0MDAwMCc7IH1jYXRjaHt9CiAgICAgIHJldHVybiBmYWxzZTsKICAgIH0KICAgIHRyeXsgZWxzLmFwaUJhc2Uuc3R5bGUuYm9yZGVyQ29sb3IgPSAnJzsgfWNhdGNoe30KICAgIHJldHVybiB0cnVlOwogIH0KICAvLyBDbGVhciB0aGUgcmVkIGJvcmRlciBhcyBzb29uIGFzIHRoZSB1c2VyIGVkaXRzIHRoZSBmaWVsZAogIGlmIChlbHMuYXBpQmFzZSkgewogICAgZWxzLmFwaUJhc2UuYWRkRXZlbnRMaXN0ZW5lcignaW5wdXQnLCAoKSA9PiB7IHRyeXsgZWxzLmFwaUJhc2Uuc3R5bGUuYm9yZGVyQ29sb3IgPSAnJzsgfWNhdGNoe30gfSk7CiAgfQoKCiAgLyogLS0tLS0tLS0tLSBNb2RlbHMgKEdFVCAvdjEvbW9kZWxzKSAtLS0tLS0tLS0tICovCiAgYXN5bmMgZnVuY3Rpb24gbG9hZE1vZGVscygpIHsKICAgIGlmICghZW5zdXJlQXBpQmFzZUFsbG93ZWQoKSkgcmV0dXJuOwogICAgY29uc3QgYmFzZSA9IEFQSV9CQVNFLnJlcGxhY2UoL1wvJC8sICcnKTsKICAgIGNvbnN0IHVybCA9IGAke2Jhc2V9L21vZGVsc2A7CgogICAgY29uc3QgaGVhZGVycyA9IGFwaUhlYWRlcnMoeyAnQWNjZXB0JzogJ2FwcGxpY2F0aW9uL2pzb24nIH0pOwoKICAgIGlmIChlbHMubW9kZWxTdGF0dXMpIGVscy5tb2RlbFN0YXR1cy50ZXh0Q29udGVudCA9ICdMb2FkaW5n4oCmJzsKICAgIGlmIChlbHMubG9hZE1vZGVsc0J0bikgZWxzLmxvYWRNb2RlbHNCdG4uZGlzYWJsZWQgPSB0cnVlOwoKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKHVybCwgeyBtZXRob2Q6ICdHRVQnLCBoZWFkZXJzLCBtb2RlOiAnY29ycycgfSk7CiAgICAgIGlmIChyZXMuc3RhdHVzID09PSA0MDEpIG9wZW5BdXRoRGlhbG9nKCdBUEkg6KqN6Ki844GM5b+F6KaB44Gn44GZ77yINDAx77yJJyk7CiAgICAgIGlmICghcmVzLm9rKSB7CiAgICAgICAgY29uc3QgdCA9IGF3YWl0IHJlcy50ZXh0KCkuY2F0Y2goKCk9PiAnJyk7CiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBIVFRQICR7cmVzLnN0YXR1c306ICR7dCB8fCAnRmFpbGVkIHRvIGxvYWQgbW9kZWxzJ31gKTsKICAgICAgfQogICAgICBjb25zdCBkYXRhID0gYXdhaXQgcmVzLmpzb24oKTsKCiAgICAgIC8vIE9wZW5BSS1TdHlsZTogeyBkYXRhOiBbe2lkOiAiLi4uIn0sIC4uLl0gfQogICAgICAvLyBGYWxsYmFja3MgZsO8ciBhbmRlcmUgU2VydmVyOgogICAgICBjb25zdCBsaXN0ID0KICAgICAgICBBcnJheS5pc0FycmF5KGRhdGE/LmRhdGEpID8gZGF0YS5kYXRhIDoKICAgICAgICBBcnJheS5pc0FycmF5KGRhdGE/Lm1vZGVscykgPyBkYXRhLm1vZGVscyA6CiAgICAgICAgQXJyYXkuaXNBcnJheShkYXRhKSA/IGRhdGEgOgogICAgICAgIFtdOwoKICAgICAgY29uc3QgaWRzID0gbGlzdAogICAgICAgIC5tYXAobSA9PiAodHlwZW9mIG0gPT09ICdzdHJpbmcnID8gbSA6IG0/LmlkKSkKICAgICAgICAuZmlsdGVyKEJvb2xlYW4pCiAgICAgICAgLm1hcChTdHJpbmcpOwoKICAgICAgaWYgKCFpZHMubGVuZ3RoKSB0aHJvdyBuZXcgRXJyb3IoJ05vIG1vZGVscyBmb3VuZCBpbiAvdjEvbW9kZWxzLicpOwoKICAgICAgaWRzLnNvcnQoKGEsYik9PmEubG9jYWxlQ29tcGFyZShiKSk7CgogICAgICBjb25zdCBwcmV2ID0gZWxzLm1vZGVsLnZhbHVlOwogICAgICBlbHMubW9kZWwuaW5uZXJIVE1MID0gJyc7CiAgICAgIGZvciAoY29uc3QgaWQgb2YgaWRzKSB7CiAgICAgICAgY29uc3Qgb3B0ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnb3B0aW9uJyk7CiAgICAgICAgb3B0LnZhbHVlID0gaWQ7CiAgICAgICAgb3B0LnRleHRDb250ZW50ID0gaWQ7CiAgICAgICAgZWxzLm1vZGVsLmFwcGVuZENoaWxkKG9wdCk7CiAgICAgIH0KCiAgICAgIC8vIEF1c3dhaGw6IHZvcmhlcmlnZSBiZWhhbHRlbiB3ZW5uIG3DtmdsaWNoLCBzb25zdCBlcnN0ZXMKICAgICAgZWxzLm1vZGVsLnZhbHVlID0gaWRzLmluY2x1ZGVzKHByZXYpID8gcHJldiA6IGlkc1swXTsKCiAgICAgIGlmIChlbHMubW9kZWxTdGF0dXMpIGVscy5tb2RlbFN0YXR1cy50ZXh0Q29udGVudCA9IGAke2lkcy5sZW5ndGh9IG1vZGVsc2A7CiAgICB9IGNhdGNoIChlKSB7CiAgICAgIGlmIChlbHMubW9kZWxTdGF0dXMpIGVscy5tb2RlbFN0YXR1cy50ZXh0Q29udGVudCA9ICdFcnJvcic7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb44Oi44OH44Or6Kqt44G/6L6844G/XSAke2UubWVzc2FnZX1gKTsKICAgIH0gZmluYWxseSB7CiAgICAgIGlmIChlbHMubG9hZE1vZGVsc0J0bikgZWxzLmxvYWRNb2RlbHNCdG4uZGlzYWJsZWQgPSBmYWxzZTsKICAgIH0KICB9CgogIC8qIC0tLS0tLS0tLS0gU2VuZCAoVW50ZXJzdMO8dHp0IDx0aGluaz4gdW5kICJyZWFzb25pbmciLCBtaXQgUGVyZm9ybWFuY2UtRml4KSAtLS0tLS0tLS0tICovCiAgYXN5bmMgZnVuY3Rpb24gc2VuZE1lc3NhZ2UoKSB7CiAgICBsZXQgb3V0RWwgPSBudWxsOwogICAgbGV0IGFzc2lzdGFudEJ1ZmZlciA9ICcnOwogICAgbGV0IHRoaW5rQnVmZmVyID0gJyc7CiAgICBsZXQgc2F2ZWRUZXh0ID0gJyc7CiAgICBsZXQgc2F2ZWRBdHRhY2htZW50cyA9IFtdOwogICAgdHJ5IHsKICAgICAgY29uc3QgdGV4dCA9IGVscy5pbnB1dC52YWx1ZS50cmltKCk7CiAgICAgIHNhdmVkVGV4dCA9IHRleHQ7CiAgICAgIHNhdmVkQXR0YWNobWVudHMgPSBbLi4uYXR0YWNobWVudHNdOwogICAgICBpZiAoIXRleHQgJiYgYXR0YWNobWVudHMubGVuZ3RoID09PSAwKSByZXR1cm47CgogICAgICAvLyDjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK3jga/pgIHkv6HjgZfjgarjgYTvvIjlhaXlipvjga/lj6/og73vvIkKICAgICAgaWYgKCFlbHMuc3RvcEJ0bi5kaXNhYmxlZCkgcmV0dXJuOwoKICAgICAgaWYgKCFlbnN1cmVBcGlCYXNlQWxsb3dlZCgpKSByZXR1cm47CgogICAgICBpZiAoIWVscy5tb2RlbC52YWx1ZSkgewogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsICdb44OS44Oz44OIXSDjg6Ljg4fjg6vmnKrpgbjmip4g4oCUIOODouODh+ODq+OBjOiqreOBv+i+vOOBvuOCjOOBpuOBhOOBvuOBm+OCk+OAguODmuODvOOCuOOCkuODquODreODvOODieOBl+OBpuOBj+OBoOOBleOBhOOAgicpOwogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICAKICAgICAgZWxzLnRva2VuUmF0ZS50ZXh0Q29udGVudCA9ICcnOwogICAgICBmaW5hbENvbXBsZXRpb25Ub2tlbnMgPSAwOwogICAgICBzdHJlYW1TdGFydFRpbWUgPSAwOwoKICAgICAgLy8g44K344K544OG44Og44OX44Ot44Oz44OX44OI44KS5q+O5Zue5Y+N5pigCiAgICAgIGNvbnN0IHN5c1ZhbCA9IGVscy5zeXN0ZW0udmFsdWUudHJpbSgpOwogICAgICBpZiAoc3lzVmFsKSB7CiAgICAgICAgY29uc3QgZXhpc3RpbmdTeXNJZHggPSBtZXNzYWdlcy5maW5kSW5kZXgobSA9PiBtLnJvbGUgPT09ICdzeXN0ZW0nKTsKICAgICAgICBpZiAoZXhpc3RpbmdTeXNJZHggPj0gMCkgewogICAgICAgICAgbWVzc2FnZXNbZXhpc3RpbmdTeXNJZHhdLmNvbnRlbnQgPSBzeXNWYWw7CiAgICAgICAgfSBlbHNlIGlmIChtZXNzYWdlcy5sZW5ndGggPT09IDApIHsKICAgICAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnc3lzdGVtJywgY29udGVudDogc3lzVmFsIH0pOwogICAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgc3lzVmFsKTsKICAgICAgICB9CiAgICAgIH0KICAgICAgY29uc3QgaGFzQXR0ID0gYXR0YWNobWVudHMubGVuZ3RoID4gMDsKICAgICAgaWYgKHRleHQgfHwgaGFzQXR0KSB7CiAgICAgICAgY29uc3QgcGFydHMgPSBbXTsKICAgICAgICBpZiAodGV4dCkgcGFydHMucHVzaCh7IHR5cGU6J3RleHQnLCB0ZXh0IH0pOwogICAgICAgIGVsc2UgcGFydHMucHVzaCh7IHR5cGU6J3RleHQnLCB0ZXh0OiAnQXR0YWNoZWQgZmlsZXM6JyB9KTsKCiAgICAgICAgY29uc3QgdGV4dEZpbGVzID0gYXR0YWNobWVudHMuZmlsdGVyKGEgPT4gYS5raW5kID09PSAndGV4dCcpOwogICAgICAgIGlmICh0ZXh0RmlsZXMubGVuZ3RoKXsKICAgICAgICAgIGxldCBibG9iID0gJ1xuXG4tLS1cbkF0dGFjaGVkIGZpbGVzICh2ZXJiYXRpbSk6XG4nOwogICAgICAgICAgZm9yIChjb25zdCBhIG9mIHRleHRGaWxlcyl7CiAgICAgICAgICAgIGJsb2IgKz0gYFxuW0ZpbGU6ICR7YS5uYW1lfSR7YS50cnVuY2F0ZWQ/JyAoVFJVTkNBVEVEKSc6Jyd9XVxuXGBcYFxgXG4ke2EudGV4dH1cblxgXGBcYFxuYDsKICAgICAgICAgIH0KICAgICAgICAgIHBhcnRzLnB1c2goeyB0eXBlOid0ZXh0JywgdGV4dDogYmxvYiB9KTsKICAgICAgICB9CgogICAgICAgIGNvbnN0IGltZ3MgPSBhdHRhY2htZW50cy5maWx0ZXIoYSA9PiBhLmtpbmQgPT09ICdpbWFnZScpOwogICAgICAgIGZvciAoY29uc3QgYSBvZiBpbWdzKXsKICAgICAgICAgIGlmIChhLmRhdGFVcmwpewogICAgICAgICAgICBwYXJ0cy5wdXNoKHsgdHlwZTonaW1hZ2VfdXJsJywgaW1hZ2VfdXJsOnsgdXJsOiBhLmRhdGFVcmwgfX0pOwogICAgICAgICAgfQogICAgICAgIH0KCiAgICAgICAgY29uc3QgY29udGVudCA9IChwYXJ0cy5sZW5ndGggPT09IDEpID8gKHBhcnRzWzBdLnRleHQgPz8gJycpIDogcGFydHM7CgogICAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAndXNlcicsIGNvbnRlbnQgfSk7CiAgICAgICAgYWRkTWVzc2FnZSgndXNlcicsIGNvbnRlbnQpOwogICAgICAgIGVscy5pbnB1dC52YWx1ZSA9ICcnOwoKICAgICAgICBhdHRhY2htZW50cyA9IFtdOwogICAgICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgICAgIH0KCgogICAgICBvdXRFbCA9IGFkZE1lc3NhZ2UoJ2Fzc2lzdGFudCcsICcnKTsKCiAgICAgIC8vIE5FVTogb3B0aW9uYWxlIFNhbXBsaW5nLVBhcmFtZXRlciBudXIgbWl0c2VuZGVuLCB3ZW5uIGdlc2V0enQKICAgICAgZnVuY3Rpb24gcGFyc2VPcHRpb25hbEZsb2F0KGVsLCBtaW4sIG1heCkgewogICAgICAgIGNvbnN0IHYgPSAoZWw/LnZhbHVlID8/ICcnKS50cmltKCk7CiAgICAgICAgaWYgKHYgPT09ICcnKSByZXR1cm4gbnVsbDsKICAgICAgICBjb25zdCBudW0gPSBwYXJzZUZsb2F0KHYpOwogICAgICAgIGlmIChOdW1iZXIuaXNOYU4obnVtKSkgcmV0dXJuIG51bGw7CiAgICAgICAgaWYgKHR5cGVvZiBtaW4gPT09ICdudW1iZXInICYmIG51bSA8IG1pbikgcmV0dXJuIG51bGw7CiAgICAgICAgaWYgKHR5cGVvZiBtYXggPT09ICdudW1iZXInICYmIG51bSA+IG1heCkgcmV0dXJuIG51bGw7CiAgICAgICAgcmV0dXJuIG51bTsKICAgICAgfQogICAgICBmdW5jdGlvbiBwYXJzZU9wdGlvbmFsSW50KGVsLCBtaW4pIHsKICAgICAgICBjb25zdCB2ID0gKGVsPy52YWx1ZSA/PyAnJykudHJpbSgpOwogICAgICAgIGlmICh2ID09PSAnJykgcmV0dXJuIG51bGw7CiAgICAgICAgY29uc3QgbnVtID0gcGFyc2VJbnQodiwgMTApOwogICAgICAgIGlmIChOdW1iZXIuaXNOYU4obnVtKSkgcmV0dXJuIG51bGw7CiAgICAgICAgaWYgKHR5cGVvZiBtaW4gPT09ICdudW1iZXInICYmIG51bSA8IG1pbikgcmV0dXJuIG51bGw7CiAgICAgICAgcmV0dXJuIG51bTsKICAgICAgfQogICAgICBjb25zdCBleHRyYXMgPSB7fTsKICAgICAgY29uc3QgX3RvcF9wID0gcGFyc2VPcHRpb25hbEZsb2F0KGVscy50b3BfcCwgMCwgMSk7CiAgICAgIGNvbnN0IF9taW5fcCA9IHBhcnNlT3B0aW9uYWxGbG9hdChlbHMubWluX3AsIDAsIDEpOwogICAgICBjb25zdCBfdG9wX2sgPSBwYXJzZU9wdGlvbmFsSW50KGVscy50b3BfaywgMCk7CiAgICAgIGlmIChfdG9wX3AgIT09IG51bGwpIGV4dHJhcy50b3BfcCA9IF90b3BfcDsKICAgICAgaWYgKF9taW5fcCAhPT0gbnVsbCkgZXh0cmFzLm1pbl9wID0gX21pbl9wOwogICAgICBpZiAoX3RvcF9rICE9PSBudWxsKSBleHRyYXMudG9wX2sgPSBfdG9wX2s7CgogICAgICAvLyB0aGlua2luZzogdXp1IHJlYWRzIHJlYXNvbmluZ19lZmZvcnQgcGVyIHJlcXVlc3QuIE90aGVyIGJhY2tlbmRzIGlnbm9yZSBpdC4KICAgICAgY29uc3QgX3RoaW5raW5nID0gKGVscy50aGlua2luZz8udmFsdWUgPz8gJycpLnRyaW0oKTsKICAgICAgaWYgKF90aGlua2luZyA9PT0gJ29mZicpIGV4dHJhcy5yZWFzb25pbmdfZWZmb3J0ID0gJ2Rpc2FibGVkJzsKICAgICAgZWxzZSBpZiAoX3RoaW5raW5nID09PSAnb24nKSBleHRyYXMucmVhc29uaW5nX2VmZm9ydCA9ICdkZWZhdWx0JzsKCiAgICAgIGNvbnN0IGJvZHkgPSB7IG1vZGVsOiBlbHMubW9kZWwudmFsdWUudHJpbSgpLCBtZXNzYWdlcywgc3RyZWFtOiB0cnVlLCB0ZW1wZXJhdHVyZTogcGFyc2VGbG9hdChlbHMudGVtcC52YWx1ZSksIC4uLmV4dHJhcyB9OwogICAgICBjb25zdCBoZWFkZXJzID0gYXBpSGVhZGVycyh7ICdDb250ZW50LVR5cGUnOiAnYXBwbGljYXRpb24vanNvbicsICdBY2NlcHQnOiAndGV4dC9ldmVudC1zdHJlYW0nIH0pOwogICAgICBjb25zdCB1cmwgPSBgJHtBUElfQkFTRS5yZXBsYWNlKC9cLyQvLCAnJyl9L2NoYXQvY29tcGxldGlvbnNgOwoKICAgICAgYWJvcnRDb250cm9sbGVyID0gbmV3IEFib3J0Q29udHJvbGxlcigpOwogICAgICBzZXRTdHJlYW1pbmdTdGF0ZSh0cnVlKTsKCiAgICAgIGxldCBpblRoaW5rQmxvY2sgPSBmYWxzZTsKICAgICAgbGV0IHRoaW5rRWwgPSBudWxsOwoKICAgICAgLy8gLS0tIFNUQVJUIMOETkRFUlVORzogQ09SUyBoaW56dWdlZsO8Z3QgLS0tCiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKHVybCwgeyBtZXRob2Q6ICdQT1NUJywgaGVhZGVycywgYm9keTogSlNPTi5zdHJpbmdpZnkoYm9keSksIHNpZ25hbDogYWJvcnRDb250cm9sbGVyLnNpZ25hbCwgbW9kZTogJ2NvcnMnIH0pOwogICAgICAvLyAtLS0gRU5ERSDDhE5ERVJVTkcgLS0tCgogICAgICBpZiAocmVzLnN0YXR1cyA9PT0gNDAxKSBvcGVuQXV0aERpYWxvZygnQVBJIOiqjeiovOOBjOW/heimgeOBp+OBme+8iDQwMe+8iScpOwogICAgICBpZiAoIXJlcy5vaykgeyBjb25zdCB0ID0gYXdhaXQgcmVzLnRleHQoKS5jYXRjaCgoKT0+ICcnKTsgdGhyb3cgbmV3IEVycm9yKGBIVFRQICR7cmVzLnN0YXR1c306ICR7dCB8fCAnU2VydmVyIGVycm9yJ31gKTsgfQogICAgICAKICAgICAgY29uc3QgY3R5cGUgPSAocmVzLmhlYWRlcnMuZ2V0KCdjb250ZW50LXR5cGUnKSB8fCAnJykudG9Mb3dlckNhc2UoKTsKICAgICAgaWYgKCFjdHlwZS5pbmNsdWRlcygndGV4dC9ldmVudC1zdHJlYW0nKSkgewogICAgICAgIGNvbnN0IHJlcXVlc3RTdGFydFRpbWUgPSBwZXJmb3JtYW5jZS5ub3coKTsKICAgICAgICBjb25zdCByYXcgPSBhd2FpdCByZXMudGV4dCgpOyAKICAgICAgICBpZiAoIXJhdykgdGhyb3cgbmV3IEVycm9yKCdFbXB0eSByZXNwb25zZSAoc3RhdHVzIE9LLCBidXQgbm8gY29udGVudCkuJyk7CiAgICAgICAgCiAgICAgICAgc3RyZWFtU3RhcnRUaW1lID0gcmVxdWVzdFN0YXJ0VGltZTsKICAgICAgICBsZXQgZGF0YTsgdHJ5eyBkYXRhID0gSlNPTi5wYXJzZShyYXcpO31jYXRjaHsgdGhyb3cgbmV3IEVycm9yKGBVbmVyd2FydGV0ZXMgRm9ybWF0OiAke3Jhdy5zbGljZSgwLDIwMCl9Li4uYCk7IH0KICAgICAgICAKICAgICAgICBsZXQgZnVsbENvbnRlbnQgPSBkYXRhPy5jaG9pY2VzPy5bMF0/Lm1lc3NhZ2U/LmNvbnRlbnQgPz8gJyc7CiAgICAgICAgY29uc3QgcmVhc29uaW5nQ29udGVudCA9IGRhdGE/LmNob2ljZXM/LlswXT8ubWVzc2FnZT8ucmVhc29uaW5nX2NvbnRlbnQ7CgogICAgICAgIGlmIChyZWFzb25pbmdDb250ZW50KSB7CiAgICAgICAgICAgIHRoaW5rRWwgPSBhZGRUaGlua0J1YmJsZShvdXRFbCk7CiAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQodGhpbmtFbCwgJ3N5c3RlbScsIHJlYXNvbmluZ0NvbnRlbnQpOwogICAgICAgICAgICB0aGlua0J1ZmZlciA9IHJlYXNvbmluZ0NvbnRlbnQ7CiAgICAgICAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGZ1bGxDb250ZW50LnRyaW0oKTsKICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAvLyBGYWxsYmFjayBhdWYgPHRoaW5rPi1UYWctVmVyYXJiZWl0dW5nCiAgICAgICAgICAgIGNvbnN0IHRoaW5rTWF0Y2ggPSBmdWxsQ29udGVudC5tYXRjaCgvPHRoaW5rPihbXHNcU10qPyk8XC90aGluaz4vKTsKICAgICAgICAgICAgaWYgKHRoaW5rTWF0Y2gpIHsKICAgICAgICAgICAgICAgIGNvbnN0IHRoaW5rVGV4dCA9IHRoaW5rTWF0Y2hbMV0udHJpbSgpOwogICAgICAgICAgICAgICAgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQodGhpbmtFbCwgJ3N5c3RlbScsIHRoaW5rVGV4dCk7CiAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciA9IHRoaW5rVGV4dDsKICAgICAgICAgICAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGZ1bGxDb250ZW50LnJlcGxhY2UodGhpbmtNYXRjaFswXSwgJycpLnRyaW0oKTsKICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGZ1bGxDb250ZW50OwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgICAgIAogICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQob3V0RWwsICdhc3Npc3RhbnQnLCBhc3Npc3RhbnRCdWZmZXIpOwogICAgICAgIAogICAgICAgIGZpbmFsQ29tcGxldGlvblRva2VucyA9IGRhdGE/LnVzYWdlPy5jb21wbGV0aW9uX3Rva2VucyB8fCAwOwogICAgICAgIHVwZGF0ZVRva2VuUmF0ZShhc3Npc3RhbnRCdWZmZXIgKyB0aGlua0J1ZmZlciwgdHJ1ZSk7CiAgICAgICAgCiAgICAgICAgbWF5YmVTY3JvbGwoKTsKICAgICAgfSBlbHNlIHsKICAgICAgICAvLyAtLS0gT1BUSU1JRVJVTkcgU1RBUlQgLS0tCiAgICAgICAgbGV0IHJlbmRlclNjaGVkdWxlZCA9IGZhbHNlOwogICAgICAgIGxldCBhc3Npc3RhbnROZWVkc1JlbmRlciA9IGZhbHNlOwogICAgICAgIGxldCB0aGlua05lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgY29uc3QgUkVOREVSX0lOVEVSVkFMID0gMTUwOyAKCiAgICAgICAgY29uc3Qgc2NoZWR1bGVSZW5kZXIgPSAoKSA9PiB7CiAgICAgICAgICAgIGlmIChyZW5kZXJTY2hlZHVsZWQpIHJldHVybjsKICAgICAgICAgICAgcmVuZGVyU2NoZWR1bGVkID0gdHJ1ZTsKICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgICAgICAgICAgICBpZiAoYXNzaXN0YW50TmVlZHNSZW5kZXIpIHsKICAgICAgICAgICAgICAgICAgICByZW5kZXJCdWJibGVDb250ZW50KG91dEVsLCAnYXNzaXN0YW50JywgYXNzaXN0YW50QnVmZmVyKTsKICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnROZWVkc1JlbmRlciA9IGZhbHNlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaWYgKHRoaW5rTmVlZHNSZW5kZXIgJiYgdGhpbmtFbCkgewogICAgICAgICAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQodGhpbmtFbCwgJ3N5c3RlbScsIHRoaW5rQnVmZmVyKTsKICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICBtYXliZVNjcm9sbCgpOwogICAgICAgICAgICAgICAgcmVuZGVyU2NoZWR1bGVkID0gZmFsc2U7CiAgICAgICAgICAgIH0sIFJFTkRFUl9JTlRFUlZBTCk7CiAgICAgICAgfTsKICAgICAgICAvLyAtLS0gT1BUSU1JRVJVTkcgRU5ERSAtLS0KCiAgICAgICAgY29uc3QgcmVhZGVyID0gcmVzLmJvZHkuZ2V0UmVhZGVyKCk7IGNvbnN0IGRlY29kZXIgPSBuZXcgVGV4dERlY29kZXIoJ3V0Zi04Jyk7IGxldCBidWZmZXI9Jyc7IGxldCBkb25lPWZhbHNlOwogICAgICAgIHdoaWxlICghZG9uZSkgewogICAgICAgICAgY29uc3QgeyB2YWx1ZSwgZG9uZTogcmREb25lIH0gPSBhd2FpdCByZWFkZXIucmVhZCgpOyBkb25lID0gcmREb25lOwoKICAgICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPT09IDAgJiYgdmFsdWUpIHsKICAgICAgICAgICAgc3RyZWFtU3RhcnRUaW1lID0gcGVyZm9ybWFuY2Uubm93KCk7CiAgICAgICAgICB9CgogICAgICAgICAgYnVmZmVyICs9IGRlY29kZXIuZGVjb2RlKHZhbHVlIHx8IG5ldyBVaW50OEFycmF5KCksIHsgc3RyZWFtOiAhZG9uZSB9KTsKICAgICAgICAgIGxldCBldmVudHMgPSBidWZmZXIuc3BsaXQoJ1xuXG4nKTsgYnVmZmVyID0gZXZlbnRzLnBvcCgpIHx8ICcnOwogICAgICAgICAgZm9yIChjb25zdCBldiBvZiBldmVudHMpIHsKICAgICAgICAgICAgY29uc3QgbGluZXMgPSBldi5zcGxpdCgnXG4nKS5maWx0ZXIoQm9vbGVhbik7CiAgICAgICAgICAgIGZvciAoY29uc3QgbGluZSBvZiBsaW5lcykgewogICAgICAgICAgICAgIGlmICghbGluZS5zdGFydHNXaXRoKCdkYXRhOicpKSBjb250aW51ZTsKICAgICAgICAgICAgICBjb25zdCBkYXRhU3RyID0gbGluZS5zbGljZSg1KS50cmltKCk7IGlmIChkYXRhU3RyID09PSAnW0RPTkVdJykgeyBkb25lID0gdHJ1ZTsgYnJlYWs7IH0KICAgICAgICAgICAgICB0cnkgewogICAgICAgICAgICAgICAgY29uc3Qgb2JqID0gSlNPTi5wYXJzZShkYXRhU3RyKTsKICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgaWYgKG9iaj8udXNhZ2U/LmNvbXBsZXRpb25fdG9rZW5zKSB7CiAgICAgICAgICAgICAgICAgICAgZmluYWxDb21wbGV0aW9uVG9rZW5zID0gb2JqLnVzYWdlLmNvbXBsZXRpb25fdG9rZW5zOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBjb25zdCBkZWx0YSA9IG9iaj8uY2hvaWNlcz8uWzBdPy5kZWx0YTsKICAgICAgICAgICAgICAgIGlmICghZGVsdGEpIGNvbnRpbnVlOwoKICAgICAgICAgICAgICAgIGxldCBoYXNVcGRhdGVkID0gZmFsc2U7CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IHJlYXNvbmluZ0NodW5rID0gZGVsdGEucmVhc29uaW5nIHx8IGRlbHRhLnJlYXNvbmluZ19jb250ZW50IHx8ICcnOwogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBpZiAocmVhc29uaW5nQ2h1bmspIHsKICAgICAgICAgICAgICAgICAgICBpZiAoIXRoaW5rRWwpIHRoaW5rRWwgPSBhZGRUaGlua0J1YmJsZShvdXRFbCk7CiAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gcmVhc29uaW5nQ2h1bms7CiAgICAgICAgICAgICAgICAgICAgdGhpbmtOZWVkc1JlbmRlciA9IHRydWU7IAogICAgICAgICAgICAgICAgICAgIGhhc1VwZGF0ZWQgPSB0cnVlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBjb25zdCBjb250ZW50Q2h1bmsgPSBkZWx0YS5jb250ZW50IHx8ICcnOwogICAgICAgICAgICAgICAgaWYgKGNvbnRlbnRDaHVuaykgewogICAgICAgICAgICAgICAgICAgIGxldCB0ZW1wQ2h1bmsgPSBjb250ZW50Q2h1bms7CiAgICAgICAgICAgICAgICAgICAgd2hpbGUgKHRlbXBDaHVuay5sZW5ndGggPiAwKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChpblRoaW5rQmxvY2spIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IGVuZFBvcyA9IHRlbXBDaHVuay5pbmRleE9mKCc8L3RoaW5rPicpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGVuZFBvcyAhPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKDAsIGVuZFBvcyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gdGVtcENodW5rLnN1YnN0cmluZyhlbmRQb3MgKyA4KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpblRoaW5rQmxvY2sgPSBmYWxzZTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyICs9IHRlbXBDaHVuazsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZW1wQ2h1bmsgPSAnJzsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb25zdCBzdGFydFBvcyA9IHRlbXBDaHVuay5pbmRleE9mKCc8dGhpbms+Jyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoc3RhcnRQb3MgIT09IC0xKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyICs9IHRlbXBDaHVuay5zdWJzdHJpbmcoMCwgc3RhcnRQb3MpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCF0aGlua0VsKSB0aGlua0VsID0gYWRkVGhpbmtCdWJibGUob3V0RWwpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IHJlc3RPZkNodW5rID0gdGVtcENodW5rLnN1YnN0cmluZyhzdGFydFBvcyArIDcpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IGVuZFBvc0luUmVzdCA9IHJlc3RPZkNodW5rLmluZGV4T2YoJzwvdGhpbms+Jyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGVuZFBvc0luUmVzdCAhPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gcmVzdE9mQ2h1bmsuc3Vic3RyaW5nKDAsIGVuZFBvc0luUmVzdCk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9IHJlc3RPZkNodW5rLnN1YnN0cmluZyhlbmRQb3NJblJlc3QgKyA4KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtOZWVkc1JlbmRlciA9IHRydWU7IAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyICs9IHJlc3RPZkNodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZW1wQ2h1bmsgPSAnJzsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5UaGlua0Jsb2NrID0gdHJ1ZTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtOZWVkc1JlbmRlciA9IHRydWU7IAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyICs9IHRlbXBDaHVuazsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnROZWVkc1JlbmRlciA9IHRydWU7IAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgIGhhc1VwZGF0ZWQgPSB0cnVlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBpZiAoaGFzVXBkYXRlZCkgewogICAgICAgICAgICAgICAgICBzY2hlZHVsZVJlbmRlcigpOwogICAgICAgICAgICAgICAgICBpZiAoc3RyZWFtU3RhcnRUaW1lID4gMCkgdXBkYXRlVG9rZW5SYXRlKGFzc2lzdGFudEJ1ZmZlciArIHRoaW5rQnVmZmVyKTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICB9IGNhdGNoIHt9CiAgICAgICAgICAgIH0KICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgLy8gRmluYWxlcyBSZW5kZXJpbmcgYW0gRW5kZSBkZXMgU3RyZWFtcywgdW0gc2ljaGVyenVzdGVsbGVuLCBkYXNzIGFsbGVzIGFuZ2V6ZWlndCB3aXJkCiAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudChvdXRFbCwgJ2Fzc2lzdGFudCcsIGFzc2lzdGFudEJ1ZmZlcik7CiAgICAgICAgaWYgKHRoaW5rRWwpIHJlbmRlckJ1YmJsZUNvbnRlbnQodGhpbmtFbCwgJ3N5c3RlbScsIHRoaW5rQnVmZmVyKTsKICAgICAgICAKICAgICAgICBpZiAoc3RyZWFtU3RhcnRUaW1lID4gMCkgdXBkYXRlVG9rZW5SYXRlKGFzc2lzdGFudEJ1ZmZlciArIHRoaW5rQnVmZmVyLCB0cnVlKTsKICAgICAgICAKICAgICAgICBtYXliZVNjcm9sbCgpOwogICAgICB9CiAgICAgIAogICAgICBhc3Npc3RhbnRCdWZmZXIgPSBhc3Npc3RhbnRCdWZmZXIudHJpbSgpOwogICAgICBtZXNzYWdlcy5wdXNoKHsgcm9sZTogJ2Fzc2lzdGFudCcsIGNvbnRlbnQ6IGFzc2lzdGFudEJ1ZmZlciB8fCAnKGVtcHR5IHJlc3BvbnNlKScgfSk7CiAgICAgIGlmIChlbHMucmF3Qm94Lm9wZW4pIGVkaXRvckR1bXBGcm9tQ2hhdCgpOwogICAgfSBjYXRjaCAoZXJyKSB7CiAgICAgIGlmIChlcnI/Lm5hbWUgPT09ICdBYm9ydEVycm9yJykgewogICAgICAgIC8vIFN0b3Djg5zjgr/jg7M6IOmAlOS4reOBvuOBp+OBruWbnuetlOOCkuWxpeattOOBq+aui+OBmQogICAgICAgIGlmIChhc3Npc3RhbnRCdWZmZXIudHJpbSgpKSB7CiAgICAgICAgICBtZXNzYWdlcy5wdXNoKHsgcm9sZTogJ2Fzc2lzdGFudCcsIGNvbnRlbnQ6IGFzc2lzdGFudEJ1ZmZlci50cmltKCkgfSk7CiAgICAgICAgfQogICAgICB9IGVsc2UgewogICAgICAgIC8vIOmAmuS/oeOCqOODqeODvDog55u05YmN44GudXNlcuODoeODg+OCu+ODvOOCuOOCkuWxpeattOOBi+OCieaIu+OBmQogICAgICAgIGlmIChvdXRFbCAmJiAhb3V0RWwudGV4dENvbnRlbnQudHJpbSgpKSB7IG91dEVsLmNsb3Nlc3QoJy5tc2cnKT8ucmVtb3ZlKCk7IH0KICAgICAgICBpZiAobWVzc2FnZXMubGVuZ3RoID4gMCAmJiBtZXNzYWdlc1ttZXNzYWdlcy5sZW5ndGgtMV0ucm9sZSA9PT0gJ3VzZXInKSB7CiAgICAgICAgICBtZXNzYWdlcy5wb3AoKTsKICAgICAgICB9CiAgICAgICAgLy8g5YWl5Yqb44OG44Kt44K544OI44KS5b6p5YWD77yI5re75LuY5pmC44GvY29udGVudOOBjOmFjeWIl+OBq+OBquOCi+OBn+OCgXNhdmVkVGV4dOOCkuS9v+OBhu+8iQogICAgICAgIGVscy5pbnB1dC52YWx1ZSA9IHNhdmVkVGV4dDsKICAgICAgICAvLyDmt7vku5jjg5XjgqHjgqTjg6vjgoLlvqnlhYMKICAgICAgICBpZiAoc2F2ZWRBdHRhY2htZW50cy5sZW5ndGggPiAwKSB7CiAgICAgICAgICBhdHRhY2htZW50cyA9IHNhdmVkQXR0YWNobWVudHM7CiAgICAgICAgICByZW5kZXJBdHRhY2htZW50cygpOwogICAgICAgIH0KICAgICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW+OCqOODqeODvF0gJHtlcnIubWVzc2FnZX1gKTsgY29uc29sZS5lcnJvcihlcnIpOwogICAgICB9CiAgICB9IGZpbmFsbHkgewogICAgICBzZXRTdHJlYW1pbmdTdGF0ZShmYWxzZSk7IGFib3J0Q29udHJvbGxlciA9IG51bGw7CiAgICB9CiAgfQoKICBmdW5jdGlvbiBzdG9wU3RyZWFtKCl7IGlmIChhYm9ydENvbnRyb2xsZXIpIGFib3J0Q29udHJvbGxlci5hYm9ydCgpOyB9CiAgZnVuY3Rpb24gY2xlYXJDaGF0KCl7IG1lc3NhZ2VzID0gW107IGVscy5sb2cuaW5uZXJIVE1MID0gJyc7IGVscy50b2tlblJhdGUudGV4dENvbnRlbnQgPSAnJzsgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7IG1heWJlU2Nyb2xsKHRydWUpOyB9CgogIC8qIC0tLS0tLS0tLS0gSW1wb3J0IC8gRXhwb3J0IC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBleHBvcnRDaGF0KCl7CiAgICBjb25zdCBvdXQgPSBidWlsZEV4cG9ydEFycmF5KCk7CiAgICBjb25zdCBibG9iID0gbmV3IEJsb2IoW0pTT04uc3RyaW5naWZ5KG91dCwgbnVsbCwgMildLCB7IHR5cGU6ICdhcHBsaWNhdGlvbi9qc29uJyB9KTsKICAgIGNvbnN0IHVybCA9IFVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7CiAgICBjb25zdCBhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYScpOyBjb25zdCB0cyA9IG5ldyBEYXRlKCk7IGNvbnN0IHBhZCA9IG49PlN0cmluZyhuKS5wYWRTdGFydCgyLCcwJyk7CiAgICBhLmhyZWYgPSB1cmw7IGEuZG93bmxvYWQgPSBgY2hhdF8ke3RzLmdldEZ1bGxZZWFyKCl9LSR7cGFkKHRzLmdldE1vbnRoKCkrMSl9LSR7cGFkKHRzLmdldERhdGUoKSl9XyR7cGFkKHRzLmdldEhvdXJzKCkpfS0ke3BhZCh0cy5nZXRNaW51dGVzKCkpfS0ke3BhZCh0cy5nZXRTZWNvbmRzKCkpfS5qc29uYDsKICAgIGEuY2xpY2soKTsgc2V0VGltZW91dCgoKT0+VVJMLnJldm9rZU9iamVjdFVSTCh1cmwpLCAxMDAwKTsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIHBhcnNlSW1wb3J0KHRleHQpewogICAgY29uc3QgY2xlYW5lZCA9IHRleHQucmVwbGFjZSgvXHIvZywnJyk7CiAgICBsZXQgcmF3OyB0cnkgeyByYXcgPSBKU09OLnBhcnNlKGNsZWFuZWQpOyB9CiAgICBjYXRjaCB7IGNvbnN0IGxpbmVzID0gY2xlYW5lZC5zcGxpdCgnXG4nKS5tYXAocz0+cy50cmltKCkpLmZpbHRlcihCb29sZWFuKTsgY29uc3QgYXJyPVtdOyBmb3IgKGNvbnN0IGxpbmUgb2YgbGluZXMpeyB0cnl7IGFyci5wdXNoKEpTT04ucGFyc2UobGluZSkpOyB9Y2F0Y2h7fSB9IHJhdyA9IGFyci5sZW5ndGggPyBhcnIgOiBudWxsOyB9CiAgICBpZiAoIXJhdykgdGhyb3cgbmV3IEVycm9yKCdGaWxlIG5vdCByZWFkYWJsZSAobmVpdGhlciBKU09OIG5vciBKU09OTCkuJyk7CgogICAgbGV0IGFyciA9IFtdLCB0b3BTeXN0ZW0gPSAnJzsKICAgIGlmIChBcnJheS5pc0FycmF5KHJhdykpIGFyciA9IHJhdzsKICAgIGVsc2UgaWYgKEFycmF5LmlzQXJyYXkocmF3Lm1lc3NhZ2VzKSkgeyBhcnIgPSByYXcubWVzc2FnZXM7IHRvcFN5c3RlbSA9IFN0cmluZyhyYXcuc3lzdGVtIHx8ICcnKS50cmltKCk7IH0KICAgIGVsc2UgaWYgKEFycmF5LmlzQXJyYXkocmF3LmNvbnZlcnNhdGlvbikpIHsgYXJyID0gcmF3LmNvbnZlcnNhdGlvbjsgdG9wU3lzdGVtID0gU3RyaW5nKHJhdy5zeXN0ZW0gfHwgJycpLnRyaW0oKTsgfQogICAgZWxzZSB0aHJvdyBuZXcgRXJyb3IoJ05vIG1lc3NhZ2UgYXJyYXkgZm91bmQgKGV4cGVjdGVkOiBbIHtyb2xlLCBjb250ZW50fSwgLi4uIF0pLicpOwoKICAgIGNvbnN0IG5vcm1hbGl6ZWQgPSBhcnIubWFwKG0gPT4gKHsgcm9sZTogbm9ybWFsaXplUm9sZShtLnJvbGUpLCBjb250ZW50OiBub3JtYWxpemVDb250ZW50KG0uY29udGVudCkgfSkpCiAgICAgICAgICAgICAgICAgICAgICAgICAgLmZpbHRlcihtID0+IFsnc3lzdGVtJywndXNlcicsJ2Fzc2lzdGFudCddLmluY2x1ZGVzKG0ucm9sZSkgJiYgdHlwZW9mIG0uY29udGVudCA9PT0gJ3N0cmluZycpOwoKICAgIGNvbnN0IGhhc1N5c3RlbSA9IG5vcm1hbGl6ZWQuc29tZShtID0+IG0ucm9sZSA9PT0gJ3N5c3RlbScpOwogICAgaWYgKCFoYXNTeXN0ZW0gJiYgdG9wU3lzdGVtKSBub3JtYWxpemVkLnVuc2hpZnQoeyByb2xlOidzeXN0ZW0nLCBjb250ZW50OiB0b3BTeXN0ZW0gfSk7CiAgICBpZiAoIW5vcm1hbGl6ZWQubGVuZ3RoKSB0aHJvdyBuZXcgRXJyb3IoJ05vIHZhbGlkIG1lc3NhZ2VzIGZvdW5kIGluIGltcG9ydC4nKTsKICAgIHJldHVybiBub3JtYWxpemVkOwogIH0KCiAgZnVuY3Rpb24gaGFuZGxlRmlsZShmaWxlKXsKICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICByZWFkZXIub25sb2FkID0gKCkgPT4gewogICAgICBwYXJzZUltcG9ydChTdHJpbmcocmVhZGVyLnJlc3VsdCB8fCAnJykpCiAgICAgICAgLnRoZW4oYXJyID0+IHsgbWVzc2FnZXMgPSBhcnI7IGNvbnN0IHN5cyA9IG1lc3NhZ2VzLmZpbmQobT0+bS5yb2xlPT09J3N5c3RlbScpOyBlbHMuc3lzdGVtLnZhbHVlID0gc3lzP3N5cy5jb250ZW50OicnOyByZW5kZXJBbGwoKTsgYWRkTWVzc2FnZSgnc3lzdGVtJywgYOODgeODo+ODg+ODiOOCkuOCpOODs+ODneODvOODiOOBl+OBvuOBl+OBnyAoJHttZXNzYWdlcy5sZW5ndGh9IG1lc3NhZ2VzKS5gKTsgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7IH0pCiAgICAgICAgLmNhdGNoKGUgPT4gYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjgqTjg7Pjg53jg7zjg4jjgqjjg6njg7xdICR7ZS5tZXNzYWdlfWApKTsKICAgIH07CiAgICByZWFkZXIub25lcnJvciA9ICgpID0+IGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb44Kk44Oz44Od44O844OI44Ko44Op44O8XSBDb3VsZCBub3QgcmVhZCBmaWxlLmApOwogICAgcmVhZGVyLnJlYWRBc1RleHQoZmlsZSk7CiAgfQogIGlmIChlbHMuZHJvcCkgewogICAgWydkcmFnZW50ZXInLCdkcmFnb3ZlciddLmZvckVhY2goZXYgPT4gZWxzLmRyb3AuYWRkRXZlbnRMaXN0ZW5lcihldiwgZSA9PiB7IGUucHJldmVudERlZmF1bHQoKTsgZWxzLmRyb3AuY2xhc3NMaXN0LmFkZCgnZHJhZycpOyB9KSk7CiAgICBbJ2RyYWdsZWF2ZScsJ2Ryb3AnXS5mb3JFYWNoKGV2ID0+IGVscy5kcm9wLmFkZEV2ZW50TGlzdGVuZXIoZXYsIGUgPT4geyBlLnByZXZlbnREZWZhdWx0KCk7IGVscy5kcm9wLmNsYXNzTGlzdC5yZW1vdmUoJ2RyYWcnKTsgfSkpOwogICAgZWxzLmRyb3AuYWRkRXZlbnRMaXN0ZW5lcignZHJvcCcsIChlKSA9PiB7IGNvbnN0IGYgPSBlLmRhdGFUcmFuc2Zlcj8uZmlsZXM/LlswXTsgaWYgKGYpIGhhbmRsZUZpbGUoZik7IH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSBSQVcgSlNPTiBFZGl0b3IgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGVkaXRvclNldFN0YXR1cyhtc2csIG9rPXRydWUpeyBlbHMuanNvblN0YXR1cy50ZXh0Q29udGVudCA9IG1zZyB8fCAnJzsgZWxzLmpzb25FZGl0b3Iuc3R5bGUuYm9yZGVyQ29sb3IgPSBvayA/ICd2YXIoLS1saW5lKScgOiAnI2I0MDAwMCc7IH0KICBmdW5jdGlvbiBlZGl0b3JEdW1wRnJvbUNoYXQoKXsgZWxzLmpzb25FZGl0b3IudmFsdWUgPSBKU09OLnN0cmluZ2lmeShidWlsZEV4cG9ydEFycmF5KCksIG51bGwsIDIpOyBlZGl0b3JTZXRTdGF0dXMoJ2xvYWRlZCBmcm9tIGNoYXQnKTsgfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvclBhcnNlQ3VycmVudCgpewogICAgY29uc3QgdGV4dCA9IGVscy5qc29uRWRpdG9yLnZhbHVlLnRyaW0oKSB8fCAnW10nOwogICAgY29uc3QgYXJyID0gYXdhaXQgcGFyc2VJbXBvcnQodGV4dCk7CiAgICBjb25zdCBmaXJzdFN5cyA9IGFyci5maW5kKG09Pm0ucm9sZT09PSdzeXN0ZW0nKTsgaWYgKGZpcnN0U3lzKSBlbHMuc3lzdGVtLnZhbHVlID0gZmlyc3RTeXMuY29udGVudDsKICAgIHJldHVybiBhcnI7CiAgfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvclJlcGxhY2UoKXsKICAgIHRyeXsKICAgICAgY29uc3QgYXJyID0gYXdhaXQgZWRpdG9yUGFyc2VDdXJyZW50KCk7CiAgICAgIGlzUmVuZGVyaW5nID0gdHJ1ZTsKICAgICAgbWVzc2FnZXMgPSBhcnI7CiAgICAgIHJlbmRlckFsbCgpOwogICAgICBpc1JlbmRlcmluZyA9IGZhbHNlOwogICAgICBhdXRvc2Nyb2xsID0gdHJ1ZTsKICAgICAgZWRpdG9yU2V0U3RhdHVzKGBhcHBsaWVkIChyZXBsYWNlKSDigJQgJHttZXNzYWdlcy5sZW5ndGh9IG1lc3NhZ2VzYCk7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBKU09O44GL44KJ44OB44Oj44OD44OI44KS6Kit5a6a44GX44G+44GX44GfICgke21lc3NhZ2VzLmxlbmd0aH0pYCk7CiAgICB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLGZhbHNlKTsgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFtKU09O44Ko44Op44O8XSAke2UubWVzc2FnZX1gKTsgfQogIH0KICBhc3luYyBmdW5jdGlvbiBlZGl0b3JBcHBlbmQoKXsKICAgIHRyeXsKICAgICAgY29uc3QgYXJyID0gYXdhaXQgZWRpdG9yUGFyc2VDdXJyZW50KCk7CiAgICAgIGlzUmVuZGVyaW5nID0gdHJ1ZTsKICAgICAgbWVzc2FnZXMgPSBtZXNzYWdlcy5jb25jYXQoYXJyKTsKICAgICAgcmVuZGVyQWxsKCk7CiAgICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICAgIGF1dG9zY3JvbGwgPSB0cnVlOwogICAgICBlZGl0b3JTZXRTdGF0dXMoYGFwcGxpZWQgKGFwcGVuZCkg4oCUIG5vdyAke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXNgKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYEpTT07jgYvjgonjg4Hjg6Pjg4Pjg4jjgpLov73oqJjjgZfjgb7jgZfjgZ8gKCske2Fyci5sZW5ndGh9KWApOwogICAgfSBjYXRjaChlKXsgZWRpdG9yU2V0U3RhdHVzKGUubWVzc2FnZSxmYWxzZSk7IGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBbSlNPTuOCqOODqeODvF0gJHtlLm1lc3NhZ2V9YCk7IH0KICB9CiAgZnVuY3Rpb24gZWRpdG9yQmVhdXRpZnkoKXsgdHJ5IHsgY29uc3Qgb2JqID0gSlNPTi5wYXJzZShlbHMuanNvbkVkaXRvci52YWx1ZSB8fCAnW10nKTsgZWxzLmpzb25FZGl0b3IudmFsdWUgPSBKU09OLnN0cmluZ2lmeShvYmosIG51bGwsIDIpOyBlZGl0b3JTZXRTdGF0dXMoJ2JlYXV0aWZpZWQnKTsgfSBjYXRjaChlKXsgZWRpdG9yU2V0U3RhdHVzKCdpbnZhbGlkIEpTT04nLCBmYWxzZSk7fSB9CiAgZnVuY3Rpb24gZWRpdG9yTWluaWZ5KCl7IHRyeSB7IGNvbnN0IG9iaiA9IEpTT04ucGFyc2UoZWxzLmpzb25FZGl0b3IudmFsdWUgfHwgJ1tdJyk7IGVscy5qc29uRWRpdG9yLnZhbHVlID0gSlNPTi5zdHJpbmdpZnkob2JqKTsgZWRpdG9yU2V0U3RhdHVzKCdtaW5pZmllZCcpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoJ2ludmFsaWQgSlNPTicsIGZhbHNlKTt9IH0KICBhc3luYyBmdW5jdGlvbiBlZGl0b3JWYWxpZGF0ZSgpeyB0cnkgeyBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsgZWRpdG9yU2V0U3RhdHVzKCd2YWxpZCDinJMnKTsgfSBjYXRjaChlKXsgZWRpdG9yU2V0U3RhdHVzKGUubWVzc2FnZSwgZmFsc2UpOyB9IH0KCiAgZWxzLnJhd0JveC5hZGRFdmVudExpc3RlbmVyKCd0b2dnbGUnLCAoKSA9PiB7IGlmIChlbHMucmF3Qm94Lm9wZW4pIGVkaXRvckR1bXBGcm9tQ2hhdCgpOyB9KTsKICBlbHMuanNvblJlZnJlc2guYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlZGl0b3JEdW1wRnJvbUNoYXQpOwogIGVscy5qc29uQXBwbHkuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlZGl0b3JSZXBsYWNlKTsKICBlbHMuanNvbkFwcGVuZC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckFwcGVuZCk7CiAgZWxzLmpzb25QcmV0dHkuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlZGl0b3JCZWF1dGlmeSk7CiAgZWxzLmpzb25NaW5pZnkuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlZGl0b3JNaW5pZnkpOwogIGVscy5qc29uVmFsaWRhdGUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlZGl0b3JWYWxpZGF0ZSk7CgogIC8qIC0tLS0tLS0tLS0gV2lyZSBVSSAtLS0tLS0tLS0tICovCiAgZWxzLnNlbmRCdG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBzZW5kTWVzc2FnZSk7CiAgZWxzLnN0b3BCdG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBzdG9wU3RyZWFtKTsKICBlbHMuY2xlYXJCdG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBjbGVhckNoYXQpOwogIGlmIChlbHMuYXR0YWNoQnRuICYmIGVscy5hdHRhY2hJbnB1dCkgewogICAgZWxzLmF0dGFjaEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpPT4gZWxzLmF0dGFjaElucHV0LmNsaWNrKCkpOwogICAgZWxzLmF0dGFjaElucHV0LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsICgpPT57CiAgICAgIGFkZEF0dGFjaG1lbnRzRnJvbUZpbGVMaXN0KGVscy5hdHRhY2hJbnB1dC5maWxlcyk7CiAgICAgIGVscy5hdHRhY2hJbnB1dC52YWx1ZSA9ICcnOwogICAgfSk7CiAgfQogIGlmIChlbHMubG9hZE1vZGVsc0J0bikgZWxzLmxvYWRNb2RlbHNCdG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBsb2FkTW9kZWxzKTsKICBlbHMuaW5wdXQuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIChlKSA9PiB7CiAgICBpZiAoZS5rZXkgPT09ICdFbnRlcicgJiYgIWUuc2hpZnRLZXkgJiYgIWUuaXNDb21wb3NpbmcgJiYgZS5rZXlDb2RlICE9PSAyMjkpIHsKICAgICAgLy8g44K544OI44Oq44O844Of44Oz44Kw5Lit44Gv6YCB5L+h44Gb44Ga44CB5pS56KGM44KS6Kix5Y+vCiAgICAgIGlmICghZWxzLnN0b3BCdG4uZGlzYWJsZWQpIHJldHVybjsKICAgICAgZS5wcmV2ZW50RGVmYXVsdCgpOwogICAgICBzZW5kTWVzc2FnZSgpOwogICAgfQogIH0pOwogIGVscy50ZW1wLmFkZEV2ZW50TGlzdGVuZXIoJ2lucHV0JywgKCkgPT4geyBlbHMudGVtcFZhbC50ZXh0Q29udGVudCA9IHBhcnNlRmxvYXQoZWxzLnRlbXAudmFsdWUpLnRvRml4ZWQoMSk7IH0pOwogIGVscy5zZXR0aW5nc1RvZ2dsZS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsKICAgIGVscy5zZXR0aW5nc1BhbmVsLmNsYXNzTGlzdC50b2dnbGUoJ2hpZGRlbicpOwogIH0pOwoKICAvKiAtLS0tLS0tLS0tIEluZm8gZGlhbG9nIC0tLS0tLS0tLS0gKi8KICBpZiAoZWxzLmluZm9WZXJzaW9uKSBlbHMuaW5mb1ZlcnNpb24udGV4dENvbnRlbnQgPSBBUFBfVkVSU0lPTjsKICBpZiAoZWxzLmluZm9Qcm9qZWN0VXJsKSB7CiAgICBlbHMuaW5mb1Byb2plY3RVcmwudGV4dENvbnRlbnQgPSBQUk9KRUNUX1VSTDsKICAgIGVscy5pbmZvUHJvamVjdFVybC5ocmVmID0gUFJPSkVDVF9VUkw7CiAgfQogIGlmIChlbHMuaW5mb0ltcHJpbnQpIHsKICAgIGVscy5pbmZvSW1wcmludC50ZXh0Q29udGVudCA9IElNUFJJTlRfVVJMOwogICAgZWxzLmluZm9JbXByaW50LmhyZWYgPSBJTVBSSU5UX1VSTDsKICB9CiAgaWYgKGVscy5pbmZvQnRuICYmIGVscy5pbmZvRGlhbG9nKSB7CiAgICBlbHMuaW5mb0J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsKICAgICAgaWYgKHR5cGVvZiBlbHMuaW5mb0RpYWxvZy5zaG93TW9kYWwgPT09ICdmdW5jdGlvbicpIGVscy5pbmZvRGlhbG9nLnNob3dNb2RhbCgpOwogICAgICBlbHNlIGVscy5pbmZvRGlhbG9nLnNldEF0dHJpYnV0ZSgnb3BlbicsICcnKTsKICAgIH0pOwogICAgLy8gQ2xpY2sgb3V0c2lkZSB0aGUgY2FyZCBjbG9zZXMgdGhlIGRpYWxvZwogICAgZWxzLmluZm9EaWFsb2cuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4gewogICAgICAvLyBJZiBjbGljayBsYW5kcyBvbiB0aGUgYmFja2Ryb3AgKG5vdCBvbiB0aGUgY2FyZCksIGNsb3NlLgogICAgICBpZiAoZS50YXJnZXQgPT09IGVscy5pbmZvRGlhbG9nKSBlbHMuaW5mb0RpYWxvZy5jbG9zZSgpOwogICAgfSk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIEFQSSDoqo3oqLzjg4DjgqTjgqLjg63jgrAgLS0tLS0tLS0tLSAqLwogIGlmIChlbHMuYXV0aEJ0biAmJiBlbHMuYXV0aERpYWxvZykgewogICAgZWxzLmF1dGhCdG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiBvcGVuQXV0aERpYWxvZygpKTsKICB9CiAgaWYgKGVscy5hdXRoU2F2ZSkgewogICAgZWxzLmF1dGhTYXZlLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgICBjb25zdCB1c2VyID0gKGVscy5hdXRoVXNlcj8udmFsdWUgfHwgJycpLnRyaW0oKSB8fCAnZXhsMyc7CiAgICAgIGNvbnN0IHBhc3MgPSBlbHMuYXV0aFBhc3M/LnZhbHVlIHx8ICcnOwogICAgICBpZiAoIXBhc3MpIHsKICAgICAgICBpZiAoZWxzLmF1dGhTdGF0dXMpIGVscy5hdXRoU3RhdHVzLnRleHRDb250ZW50ID0gJ+ODkeOCueODr+ODvOODieOCkuWFpeWKm+OBl+OBpuOBj+OBoOOBleOBhCc7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIHNldEF1dGhDcmVkZW50aWFscyh1c2VyLCBwYXNzKTsKICAgICAgY2xvc2VBdXRoRGlhbG9nKCk7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb6KqN6Ki8XSAke3VzZXJ9IOOBp+S/neWtmOOBl+OBvuOBl+OBn++8iOOBk+OBruOCv+ODluOBruOBv++8iWApOwogICAgICBsb2FkTW9kZWxzKCk7CiAgICAgIHN0YXJ0UXVvdGFQb2xsaW5nKCk7CiAgICAgIGRldGVjdFN0b3BTdXBwb3J0KCk7CiAgICB9KTsKICB9CiAgaWYgKGVscy5hdXRoQ2xlYXIpIHsKICAgIGVscy5hdXRoQ2xlYXIuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7CiAgICAgIGNsZWFyQXV0aENyZWRlbnRpYWxzKCk7CiAgICAgIGNsb3NlQXV0aERpYWxvZygpOwogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCAnW+iqjeiovF0g6LOH5qC85oOF5aCx44KS6Kej6Zmk44GX44G+44GX44GfJyk7CiAgICB9KTsKICB9CiAgaWYgKGVscy5hdXRoUGFzcykgewogICAgZWxzLmF1dGhQYXNzLmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCAoZSkgPT4gewogICAgICBpZiAoZS5rZXkgPT09ICdFbnRlcicpIHsgZS5wcmV2ZW50RGVmYXVsdCgpOyBlbHMuYXV0aFNhdmU/LmNsaWNrKCk7IH0KICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhEaWFsb2cpIHsKICAgIGVscy5hdXRoRGlhbG9nLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKGUpID0+IHsKICAgICAgaWYgKGUudGFyZ2V0ID09PSBlbHMuYXV0aERpYWxvZykgY2xvc2VBdXRoRGlhbG9nKCk7CiAgICB9KTsKICB9CgogIC8qIC0tLS0tLS0tLS0g5q6L44KK5pmC6ZaTIC8g5YGc5q2i44Oc44K/44OzIC0tLS0tLS0tLS0gKi8KICBzdGFydFF1b3RhUG9sbGluZygpOwogIGRldGVjdFN0b3BTdXBwb3J0KCk7CiAgaWYgKGVscy5zdG9wU2VydmVyKSB7CiAgICBlbHMuc3RvcFNlcnZlci5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHN0b3BTZXJ2ZXIpOwogIH0KCiAgLyogLS0tLS0tLS0tLSBXZWxjb21lIC0tLS0tLS0tLS0gKi8KICBzZXRUaW1lb3V0KCgpID0+IHsgbG9hZE1vZGVscygpOyB9LCA1MDApOwp9KSgpOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

OUT_DIRS = ("/kaggle/working", "/output")

# Dataset のマウント先は共通ランタイム（runtime/storage.py）に一本化。
MODEL_SRC = kc_find_dataset("qwen38-27b-uncensored-hauhaucs-awq-mtp")


save_output = kc_save_output
fail = kc_fail


_files = sorted(os.listdir(MODEL_SRC)) if os.path.isdir(MODEL_SRC) else []
print("モデル:", MODEL_SRC)
print("直下ファイル:", len(_files))
for _f in _files:
    _p = os.path.join(MODEL_SRC, _f)
    if os.path.isfile(_p):
        print(f"  {_f:34s} {os.path.getsize(_p):>15,} B")
if "config.json" not in _files:
    fail("config.json がありません。Add Data を確認してください。")
if "model.safetensors" not in _files:
    fail("model.safetensors がありません。Dataset を確認してください。")


# ── 1️⃣ vLLM install とフラグ判定（所要時間を実測） ────────────


# vLLM を入れて、起動ラダーが前提にしているフラグの有無を確認する
import subprocess
import sys
import time

t0 = time.time()
_r = subprocess.run([sys.executable, "-c", "import vllm"],
                    capture_output=True, text=True)
install_sec = 0.0
if _r.returncode == 0:
    print("ℹ vLLM は既に入っている（install 時間 0）")
else:
    print("⬇️  pip install vllm 開始")
    _p = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-input", "vllm"],
        capture_output=True, text=True, timeout=3600)
    install_sec = time.time() - t0
    print((_p.stderr or "")[-1500:])
    if _p.returncode != 0:
        fail("vLLM install 失敗（上のログ）")

# torchaudio は transformers が import する。vLLM の pip install で torch だけが
# 差し替わると CUDA 版数が不一致になり、vllm CLI が即死する
# （実測: vllm 0.32.0 / torch 2.13.0+cu130 に対し torchaudio 2.8.3+cu128）。
#
# 検出は「実際に落ちる経路」で行う。`import vllm.entrypoints.cli.main` だけでは
# 不十分だった（v2 実測）: benchmark.main 等は main() の中で遅延 import される
# ため、深い依存（transformers → torchaudio）の不整合を見逃す。
# torchaudio の整合は serve より前に単独で確認する（v2 の真因）。
# PyPI の torchaudio 2.11.0 は +cu130 ビルド（wheel の version.py で実測）なので、
# torch 2.13.0+cu130 とそのまま一致する。
def _ta_check():
    r = subprocess.run([sys.executable, "-c", "import torchaudio"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-900:]


_ta_good, _ta_err = _ta_check()
if not _ta_good:
    print("⚠ torchaudio 不整合 → 2.11.0(+cu130) を --no-deps で入れ直す")
    print(_ta_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
                   capture_output=True, text=True, timeout=1800)
    _ta_good, _ta_err = _ta_check()
if not _ta_good:
    # vLLM が torchaudio を使うのは音声モデルだけ（midashenglm/audio.py 等）。
    # このモデルは音声不要なので、外すのが最終手段として妥当。
    print("⚠ まだ不整合 → torchaudio を削除")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ta_good, _ta_err = _ta_check()
print(f"torchaudio import: {'OK' if _ta_good else 'NG'}")

_PREFLIGHT_CMD = ["vllm", "serve", "--help"]


def _cli_ok():
    r = subprocess.run(_PREFLIGHT_CMD, capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    bad = ("Traceback" in out) or ("RuntimeError" in out) or len(out) < 500
    return (r.returncode == 0 and not bad), out[-1500:]


_ok, _err = _cli_ok()
if not _ok:
    # transformers 側は ImportError を握るので、外すのが最短（通信不要）。
    # vLLM 0.32 は torchaudio==2.11.0 を要求する（METADATA 実測）。
    # 版を揃えて入れ直すのが本筋。駄目なら外す（transformers は ImportError を握る）。
    print("⚠ vLLM CLI が起動しない → torchaudio を 2.11.0 で入れ直す")
    print(_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
                   capture_output=True, text=True, timeout=1800)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を外して再試行")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を torch に合わせて入れ直す")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "-U", "torchaudio"], capture_output=True, text=True,
                   timeout=3600)
    _ok, _err = _cli_ok()
if not _ok:
    # help が壊れていても serve が壊れているとは限らない（実測: CPU では
    # transformers ではなく arg_utils 側で別の失敗をする）。ここで止めず、
    # 起動を試す。壊れていれば最初の試行が即座に落ちるだけで済む。
    print("⚠ vLLM CLI の確認に失敗 → そのまま起動を試す")
    print(_err)

import torch
import vllm
print(f"✅ vLLM {vllm.__version__} / torch {torch.__version__} "
      f"/ CUDA {torch.version.cuda}")
print(f"⏱  install 所要: {install_sec/60:.1f} 分")

_h = subprocess.run(["vllm", "serve", "--help"],
                    capture_output=True, text=True, timeout=180)
HELP = (_h.stdout or "") + (_h.stderr or "")
# 判定が外れたときの唯一の手掛かり。ログと成果物の両方に残す。
print(f"vllm serve --help: exit={_h.returncode} {len(HELP)} 文字")
print(HELP[:1500])
save_output("vllm_help.txt", HELP[:20000])
HELP_OK = (_h.returncode == 0 and len(HELP) > 500
           and "Traceback (most recent call last)" not in HELP)

# ここに列挙した flag だけを avail に入れる。avail[flag] を直接引く箇所が
# あるため、後から使う flag を足し忘れると HELP 失敗時（= CPU では常に）に
# KeyError で即死する（v6 で実測）。
NEEDED = ["--speculative-config", "--trust-remote-code", "--reasoning-parser",
          "--enable-auto-tool-choice", "--tool-call-parser",
          "--disable-log-requests", "--kv-cache-dtype",
          "--gpu-memory-utilization", "--max-num-seqs",
          "--served-model-name", "--tensor-parallel-size",
          "--language-model-only"]
avail = {f: (f in HELP) for f in NEEDED}
if not HELP_OK:
    # 悲観側に倒す。0.32 で廃止されたフラグを付けると未知引数で即死する。
    avail = {f: False for f in NEEDED}
for f, ok in avail.items():
    print(("✅" if ok else "❌"), f)

missing = [f for f in ("--tensor-parallel-size", "--served-model-name",
                       "--max-num-seqs", "--gpu-memory-utilization")
           if not avail[f]]
if not HELP_OK:
    print("⚠ ヘルプ本文が取れない → フラグ判定は不能。"
          "本番は悲観側（任意フラグを落とす）で動く")
elif missing:
    # ここで止めると「help の書式が変わっただけ」でも全損になる。
    # 記録して先へ進む（本番ビルダーは同じ状況でも起動を試す）。
    print(f"⚠ 必須フラグが見つかりません（書式変更の可能性）: {missing}")

# 起動ラダーを組み立てる（実起動はしない）
# 量子化は指定しない（config の quantization_config は pack-quantized。
# awq 固定は誤り。詳細 notes/vllm-cpu-findings.md）
CORE = ["vllm", "serve", "<stage>", "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(TP_SIZE), "--dtype", "half",
        "--max-num-seqs", "2",
        "--gpu-memory-utilization", "0.90", "--kv-cache-dtype", "auto"]
# 本番と同じくテキスト専用で起動（視覚塔は落とす）。
if avail["--language-model-only"]:
    CORE.append("--language-model-only")
OPT = []
if avail["--trust-remote-code"]:
    OPT.append("--trust-remote-code")
if avail["--disable-log-requests"]:   # 0.32 では廃止（--enable-log-requests）
    OPT.append("--disable-log-requests")
if avail["--reasoning-parser"]:
    OPT += ["--reasoning-parser", "qwen3"]
if avail["--enable-auto-tool-choice"] and avail["--tool-call-parser"]:
    OPT += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]

SPEC = None
if MTP_ENABLED and avail["--speculative-config"]:
    SPEC = {"method": MTP_METHOD, "num_speculative_tokens": MTP_TOKENS}

PLAN = []
if SPEC:
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[0], True))
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[1], True))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[0], False))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[1], False))
PLAN.append(("plain", None, 8192, False))
PLAN.append(("triton", None, 8192, False))   # 0.32 に V0 は無い（明示 backend）

print()
print("起動ラダー（GPU 実行時にこの順で試す）:")
for i, (lab, sp, ml, use_opt) in enumerate(PLAN, 1):
    print(f"  {i}. spec={lab} max_len={ml} opt_flags={'on' if use_opt else 'off'}")
save_output("dryrun_plan.txt", json.dumps(
    {"vllm": vllm.__version__, "torch": torch.__version__,
     "install_sec": install_sec, "flags": avail,
     "plan": [[l, s, m, o] for l, s, m, o in PLAN]},
    indent=2, ensure_ascii=False) + "\n")


# ── 2️⃣ Dataset のマウントと symlink ステージング ───────────


# symlink ステージング（実体コピーなし = disk を食わない）
import os
import shutil

STAGE = "/kaggle/tmp/vllm-model"
MTP_FILE = "model-mtp.safetensors"


def _stage_model(with_mtp):
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)          # symlink のみなので実体は消えない
    os.makedirs(STAGE, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(MODEL_SRC)):
        if fn.startswith("."):
            continue
        if fn == MTP_FILE and not with_mtp:
            continue
        src = os.path.join(MODEL_SRC, fn)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(STAGE, fn))
        n += 1
    return n


n_on = _stage_model(True)
on_files = sorted(os.listdir(STAGE))
n_off = _stage_model(False)
off_files = sorted(os.listdir(STAGE))
print(f"MTP ON : {n_on} files  mtp={'model-mtp.safetensors' in on_files}")
print(f"MTP OFF: {n_off} files  mtp={'model-mtp.safetensors' in off_files}")
print(f"ステージ合計サイズ（実体参照）: "
      f"{sum(os.path.getsize(os.path.join(STAGE, f)) for f in off_files)/1e9:.1f} GB")
print(f"Dataset 実体は無傷: {os.path.isfile(os.path.join(MODEL_SRC, 'model.safetensors'))}")

if "model-mtp.safetensors" in on_files and \
        "model-mtp.safetensors" not in off_files and n_off >= 3:
    print("✅ ステージング OK")
else:
    fail(f"ステージング異常: on={n_on} off={n_off}")


# ── 3️⃣ モック vLLM 上流（8000番） ──────────────────────


# モック上流: vLLM の OpenAI 互換 API を最小再現する
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAST = {}


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": SERVED_NAME, "object": "model",
                 "max_model_len": 32768, "owned_by": "mock"}]})
        elif p == "/__last":
            self._json(LAST)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        global LAST
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        LAST = {"path": self.path, "body": body}

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(12):
                chunk = {"id": "mock", "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta":
                                      {"content": f"tok{i} "}}]}
                payload = ("data: " + json.dumps(chunk) + "\n\n").encode()
                self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
                self.wfile.flush()
                time.sleep(0.15)
            done = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        self._json({
            "id": "mock", "object": "chat.completion",
            "model": body.get("model", SERVED_NAME),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "OK. (mock)"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4,
                      "total_tokens": 12},
        })


subprocess.run(["pkill", "-f", "mock_vllm"], capture_output=True)
_server = ThreadingHTTPServer(("127.0.0.1", VLLM_PORT), Mock)
threading.Thread(target=_server.serve_forever, daemon=True).start()
time.sleep(0.5)
import urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models",
                            timeout=5) as r:
    _d = json.loads(r.read())
print(f"✅ モック起動: {VLLM_PORT} models={[m['id'] for m in _d['data']]}")


# ── 4️⃣ プロキシ起動（`--enforce-sampling`） ────────────


# プロキシ + UI を配置して起動（本番と同じ引数）
import base64
import os

with open(PROXY_PATH, "wb") as f:
    f.write(base64.b64decode(PROXY_SRC_B64))
with open(CHAT_PATH, "wb") as f:
    f.write(base64.b64decode(CHAT_SRC_B64))
print(f"auth_proxy.py {os.path.getsize(PROXY_PATH):,} B / "
      f"llama-chat.html {os.path.getsize(CHAT_PATH):,} B")

proxy_proc = kc_start_proxy(
    PROXY_PORT, f"http://127.0.0.1:{VLLM_PORT}", CHAT_PATH, PROXY_PATH,
    extra_args=("--enforce-sampling",), fail=fail)


# ── 5️⃣ cloudflared quick tunnel ────────────────


# cloudflared 起動（Dataset 優先 / 無ければ DL）
import os
import shutil
import subprocess
import time
import urllib.request

CF_BIN = "/usr/local/bin/cloudflared"

# 公開URLを開く（IPv4 優先で 1 回だけ再試行）。実体は runtime/net.py。
_open_public = kc_urlopen_v4

NONE = kc_find_dataset("my-cloudflared-bin")
src = None
for cand in (os.path.join(NONE, "cloudflared"),
             os.path.join(NONE, "bin", "cloudflared")):
    if os.path.isfile(cand) and os.path.getsize(cand) > 1024 * 1024:
        src = cand
        break
if src:
    shutil.copy2(src, CF_BIN)
    subprocess.run(["chmod", "+x", CF_BIN])
    print(f"✅ cloudflared 復元: {src}")
elif not os.path.isfile(CF_BIN):
    subprocess.run(
        "curl -fsSL -o /usr/local/bin/cloudflared "
        f"{kc_cloudflared_url()}"
        " && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True, timeout=300)

TUNNEL_NAME = "cloudflared"
TUNNEL_LOG = "/tmp/cloudflared.log"
RESTART_CMD = [CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]

subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)
with open(TUNNEL_LOG, "w") as log:
    subprocess.Popen(RESTART_CMD, stdout=log, stderr=log)

public_url = kc_wait_url("cloudflared", TUNNEL_LOG)
if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("公開URL取得失敗", TUNNEL_LOG)

print(f"🌍 {public_url}")
with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\n")

elapsed = kc_wait_http(f"{public_url}/healthz", attempts=20, interval=3,
                       opener=_open_public)
if elapsed is None:
    print("⚠ 公開URLへの疎通が 60s 以内に取れませんでした（次の検証で再試行）")
else:
    print(f"✅ トンネル確立（{elapsed}s）")


# ── 6️⃣ 公開URL経由の検証（モック上流） ───────────────────────


# 5点検証（実推論以外は本番と同一経路）
import json
import time as _time
import urllib.error
import urllib.request

with open("/tmp/public_url.txt") as f:
    public_url = f.read().strip()

lines = [f"PUBLIC_URL={public_url}", f"MODEL={SERVED_NAME}", ""]
failures = []


def _judge(idx, label, ok, detail):
    print(("✅" if ok else "❌"), f"{label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _post(url, payload, timeout=120, stream=False):
    return kc_post_json(url, payload, timeout=timeout)


def _with_retry(fn, tries=3, delay=3):
    """公開 URL は cloudflared 側の一時障害で `Network is unreachable` になる。
    実測: v5 で UI と /v1/models だけ落ち、localhost 系は全部通った。
    一過性なので数回だけ粘る（本番で 20 分消費した後に誤 fail しないため）。
    """
    return kc_retry(fn, tries=tries, delay=delay, skip_http_errors=False)


CHAT = {"model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 32}

# [1] UI
try:
    with _with_retry(lambda: _open_public(
            f"{public_url}/", timeout=60)) as r:
        html = r.read()
        ok_ui = (r.status == 200 and len(html) > 10000
                 and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
        _judge(1, "ui", ok_ui,
               f"HTTP {r.status} {len(html)}B")
except Exception as e:
    _judge(1, "ui", False, f"{type(e).__name__}: {e}")

# [2] /v1/models
try:
    with _with_retry(lambda: _open_public(
            f"{public_url}/v1/models", timeout=60)) as r:
        md = json.loads(r.read())
    ids = [m.get("id") for m in md.get("data", [])]
    _judge(2, "models", r.status == 200 and SERVED_NAME in ids,
           f"ids={ids}")
except Exception as e:
    _judge(2, "models", False, f"{type(e).__name__}: {e}")

# [3] streaming
try:
    resp = _post(f"{public_url}/v1/chat/completions",
                 dict(CHAT, stream=True), timeout=120)
    times = []
    t0 = _time.time()
    with resp as r:
        for raw in r:
            if not raw.strip():
                continue
            times.append(_time.time() - t0)
    total = _time.time() - t0
    if len(times) >= 3:
        first, spread = times[0], times[-1] - times[0]
        ok = spread > 0.2 and first < total * 0.8
        detail = (f"chunks={len(times)} first={first:.2f}s "
                  f"total={total:.2f}s spread={spread:.2f}s")
    else:
        ok, detail = False, f"chunks={len(times)}"
    _judge(3, "streaming", ok, detail)
except Exception as e:
    _judge(3, "streaming", False, f"{type(e).__name__}: {e}")

# [4] OpenAI 互換（temperature=0 でも 200 + 生成）
try:
    with _post(f"{public_url}/v1/chat/completions",
               dict(CHAT, temperature=0, repetition_penalty=1.0)) as r:
        st = r.status
        cj = json.loads(r.read())
    text = "".join((c.get("message") or {}).get("content") or ""
                   for c in cj.get("choices", []))
    usage = cj.get("usage") or {}
    _judge(4, "openai-compat", st == 200 and bool(text or usage),
           f"HTTP {st} reply={text[:30]!r} usage={usage}")
except Exception as e:
    _judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

# [5] enforce-sampling が上流まで届いているか（モックに記録されたボディ）
try:
    with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/__last", timeout=10) as r:
        last = json.loads(r.read())
    b = last.get("body") or {}
    temp = b.get("temperature")
    rp = b.get("repetition_penalty")
    ok = temp is not None and float(temp) >= 0.7 and \
        rp is not None and float(rp) >= 1.05
    _judge(5, "enforce-sampling", ok, f"upstream temp={temp} rp={rp}")
except Exception as e:
    _judge(5, "enforce-sampling", False, f"{type(e).__name__}: {e}")

# [6] 会話（履歴が上流までそのまま渡る）
# 実推論は GPU でしか試せないが、proxy が messages を欠落・改変しないことは
# ここで確認できる（role 順・日本語・system の保持）。
try:
    conv = [
        {"role": "system", "content": "あなたは日本語で簡潔に答えます。"},
        {"role": "user", "content": "私の名前はコウヘイです。"},
        {"role": "assistant", "content": "コウヘイさん、覚えました。"},
        {"role": "user", "content": "私の名前は？"},
    ]
    resp = _post(f"{public_url}/v1/chat/completions",
                 {"model": SERVED_NAME, "messages": conv, "max_tokens": 32},
                 timeout=120)
    with resp as r:
        r.read()
    with _open_public(f"http://127.0.0.1:{VLLM_PORT}/__last", timeout=10) as r:
        sent = ((json.loads(r.read()).get("body") or {}).get("messages") or [])
    ok6 = sent == conv
    _judge(6, "conversation-history", ok6,
           f"messages={len(sent)} equal={ok6}")
except Exception as e:
    _judge(6, "conversation-history", False, f"{type(e).__name__}: {e}")

report = "\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\n"):
    failures.append("verify_result.txt 保存失敗")
if failures:
    fail("検証失敗: " + "; ".join(failures))
print("✅ 6/6 パス（モック上流）— 推論以外は本番同一")


# ── 7️⃣ キープアライブ（2分・CPU） ─────────────────────────


# keepalive 2分（本番は 480 分）。プロキシ/トンネルの死活だけ見る
import time
import urllib.request

KEEPALIVE_MINUTES = 8
KEEPALIVE_CONF = "/tmp/keepalive.conf"


def _proc(name):
    return kc_proc_state(name, "exact" if name == "cloudflared" else "argv")


def _read_conf():
    run, minutes, _mtime = kc_read_conf(KEEPALIVE_CONF, KEEPALIVE_MINUTES,
                                        KEEPALIVE_MINUTES, on_invalid="zero")
    return run, minutes


print(f"Public URL: {public_url}")
log = []
for i in range(1, KEEPALIVE_MINUTES + 1):
    run, minutes = _read_conf()
    if run <= 0:
        print(f"RUN=0 で {i}周目に終了")
        break
    time.sleep(60)
    p_st, t_st = _proc("auth_proxy.py"), _proc("cloudflared")
    line = (f"[{time.strftime('%H:%M:%S')}] proxy={p_st} tunnel={t_st}")
    if i % 1 == 0:
        try:
            with urllib.request.urlopen(f"{public_url}/healthz", timeout=15) as r:
                line += " | tunnel=OK" if r.status == 200 else f" | HTTP{r.status}"
        except Exception as e:
            line += f" | tunnel=ERR({type(e).__name__})"
    log.append(line)
    print(line)

save_output("keepalive.txt", "\n".join(log) + "\n")

# 後片付け
left = kc_stop_processes([("-x cloudflared", "cloudflared"),
                          ("-f auth_proxy.py", "auth_proxy")])
if "_server" in globals():
    _server.shutdown()
print("🧹 停止済（proxy / cloudflared / mock）")
if left:
    print(f"⚠ 残存: {left}")

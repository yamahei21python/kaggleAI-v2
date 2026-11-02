#!/usr/bin/env python3
"""vllm serve（自動生成。直接編集しない）。

生成: kaggle/build_step2_vllm_script.py
"""
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

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
# 一緒に置いておくと `kc_wait_url` が既定値を知らずに済む。
# ngrok は 2026-11-01 に廃止した（`notes/url-publishing.md`）。表に残すと
# `kc_probe_url("ngrok")` が黙って空を返し、「トンネルは張れたのに URL が
# 取れない」に見えるので、項目ごと落とす（KeyError で即座に気づける）。
# 管理 API から URL を取る分岐（旧 ngrok 用）も一緒に消した。表に `api` の
# 項目が無い以上、通らない枝になる。
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

# ── runtime/gist.py ───────────────────────────────────────
"""Gist への進捗書き込み（生成コードへ埋め込む）。

Kaggle の UI ログは長大で貼りにくく、`kaggle kernels output` は**完了まで
何も返さない**。起動に失敗した回は「どこで止まったか」を外から見る手段が
要るので、待機中は 30 秒ごとに Gist へ流している。

ところが 2026-11-01 の `vllm-playground` は Gist を 10/28 から一度も
更新していなかった。理由（トークン無し / 401 / 404）はどこにも残らず、
外からは「何も起きていない」と区別がつかなかった。**黙って落ちるのが
一番高い**ので、この層は必ず理由を返す。

**この経路は専用アカウント運用とは両立しない。** Kaggle の Secret は
アカウントに登録するだけでは読めず、**カーネルごとに紐付ける**必要があり、
未紐付けの `get_secret()` は `ConnectionError` を返す（通信障害に見える）。
専用アカウント側では公開 URL を**ビーコンカーネル**（カーネルのコードを
掲示板にする方式）へ移した。経緯と実測は `notes/url-publishing.md`。
ここが生きているのは、進捗（起動待ちの状況）を運ぶ経路がまだ他に無いため。

`kc_` 接頭辞は生成コードの名前空間で衝突しないため（runtime/__init__.py）。
"""
import json
import os
import time
import urllib.error
import urllib.request

_GIST_API = "https://api.github.com/gists/"


def kc_gh_token_detail():
    """`(token, detail)`。detail は**どこで解決したか / どこで転んだか**。

    `detail` を返すのは、`""` の 1 語では対処が決まらないため。Kaggle の
    Secret は**アカウントに登録するだけでは足りず、カーネルごとに紐付ける**
    必要がある。2026-11-01 に新しく push した `gist-probe` は
    `token_present=false` で、登録漏れなのか紐付け漏れなのかを
    `""` からは読み取れなかった（Kaggle は「この notebook には無い」を
    例外で返すので、そこまで拾わないと区別できない）。
    """
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok, "env:GITHUB_TOKEN"
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as e:
        return "", f"kaggle_secrets を import できない（{type(e).__name__}: {e}）"
    try:
        tok = (UserSecretsClient().get_secret("GITHUB_TOKEN") or "").strip()
    except Exception as e:
        # ここに来る典型は「このカーネルに Secret が紐付いていない」。
        return "", (f"Secret を読めない（{type(e).__name__}: {e}）"
                    " — アカウントに登録済みでも、カーネルごとの紐付けが"
                    "無いと読めない")
    if not tok:
        return "", "Secret は読めたが中身が空"
    return tok, "secrets:GITHUB_TOKEN"


def kc_gist_write(gist_id, filename, text, token=None, timeout=10):
    """Gist の 1 ファイルを更新する。戻り値は `(ok, detail)`。

    `detail` は**失敗の理由**（成功なら空文字）。呼び出し側が 1 回だけ声に
    出すために使う。HTTP の本文も拾うのは、401（トークン不正）と
    404（Gist ID 違い / 権限なし）を取り違えないため。

    `token` を明示できるのはテストの口。省略時は env → Secrets の順で解決する。
    """
    if not gist_id:
        return False, "GIST_ID が空（settings の GIST_ID_DEFAULT も未設定）"
    if token is None:
        tok, why = kc_gh_token_detail()
        if not tok:
            return False, why
    else:
        tok = token
        if not tok:
            return False, "トークンが空（引数で明示された）"
    body = json.dumps({"files": {filename: {"content": text}}}).encode()
    req = urllib.request.Request(
        _GIST_API + gist_id, data=body, method="PATCH",
        headers={"Authorization": "Bearer " + tok,
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True, ""
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            msg = ""
        return False, f"HTTP {e.code}: {msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def kc_gist_selftest(gist_id, filename="vllm_status.txt", token=None,
                     timeout=10):
    """PATCH して GET で**読み返す**。往復が通ったときだけ ok=True。

    書けたつもりで読めない（別 Gist を指している / 権限が片方だけ）を
    分けるため、書き込みだけで判定しない。CPU だけで完結するので、
    GPU を使う前に Kaggle のコンテナから 1 回走らせておく。
    """
    stamp = "selftest " + time.strftime("%Y-%m-%d %H:%M:%S")
    if token is None:
        tok, token_why = kc_gh_token_detail()
    else:
        tok, token_why = token, "引数で明示"
    out = {"ok": False, "gist_id": gist_id, "filename": filename,
           "token_present": bool(tok), "token_detail": token_why, "stamp": stamp,
           "write_ok": False, "write_detail": "", "read_ok": False,
           "read_detail": "", "matched": False, "files": []}
    if not tok:
        # 解決に失敗した**理由**をそのまま write_detail に置く。ここで
        # kc_gist_write に空トークンを渡すと「引数で明示された」という
        # 無意味な理由に潰れて、紐付け漏れが分からなくなる。
        out["write_detail"] = token_why
        return out
    ok, detail = kc_gist_write(gist_id, filename, stamp + "\n",
                              token=tok, timeout=timeout)
    out["write_ok"], out["write_detail"] = ok, detail
    if not ok:
        return out
    req = urllib.request.Request(
        _GIST_API + gist_id,
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read() or b"{}")
        out["read_ok"] = True
        files = d.get("files") or {}
        out["files"] = sorted(files)
        content = (files.get(filename) or {}).get("content") or ""
        out["matched"] = stamp in content
        if not out["matched"]:
            out["read_detail"] = (f"{filename} が読めない、または中身が違う"
                                  f"（Gist 内: {sorted(files)}）")
    except urllib.error.HTTPError as e:
        out["read_detail"] = f"HTTP {e.code}"
    except Exception as e:
        out["read_detail"] = f"{type(e).__name__}: {e}"
    out["ok"] = out["read_ok"] and out["matched"]
    return out

# ── runtime/beacon.py ─────────────────────────────────────
"""ビーコンカーネル（Kaggle のカーネルを掲示板として使う）。

Gist 方式の代わりに、**カーネルのコードそのもの**を公開 URL の置き場にする。

- 書き手: 配信カーネルの中から `kernels_push` で小さなカーネルを更新する。
  認証は Kaggle がカーネルへ注入するもの（Secrets 不要）。
- 読み手: `kaggle kernels pull` でコードを引いて `URL` を読む。

なぜ Gist をやめたか: Kaggle の Secret は**カーネルごとに紐付ける**必要があり、
未紐付けの `get_secret()` は `ConnectionError` を返す（通信障害に見える）。
専用アカウント運用では、カーネルを増やすたびに UI 作業が要り、トークンも
個人アカウントのものが残る。ビーコンは**アカウント・Secrets・外部サービスを
増やさずに**同じことができる。経緯は `notes/url-publishing.md`。

この層の方針は Gist と同じ: **失敗しても配信を止めない。ただし理由は必ず返す。**
2026-11-01 に「Gist が 10/28 から一度も更新されていないことに誰も気づかない」
という失敗をしているので、`(ok, detail)` を返さない関数はここに置かない。

`kc_` 接頭辞は生成コードの名前空間で衝突しないため（runtime/__init__.py）。
"""
import json
import os
import shutil
import sys
import tempfile
import time

# 掲示板の中身。**この文字列がそのまま beacon カーネルの main.py になる。**
# 値は repr で埋めるので、URL に引用符が混ざっても壊れない。
_BEACON_BODY = '''"""公開 URL のビーコン（掲示板）。

このカーネルは**実行しない**。コードそのものが受け渡しの場所で、
配信カーネルが `kernels_push` でこのファイルを書き換える。
読む側は `kaggle kernels pull` して `URL` と `UPDATED` を読む。

`STAGE` は `serving` / `probe` など。**`probe` は本物ではない**ので、
読む側は `probe` を捨てること（テスト URL を掴む事故を防ぐ）。
`UPDATED` は発行時刻の epoch 秒。古い値を掴んでいないかの判定に使う。
"""

URL = {url!r}
STAGE = {stage!r}
UPDATED = {updated!r}
'''


def kc_beacon_slug(slug=""):
    """`owner/name` を解決する（引数 → 環境変数）。無ければ空文字。"""
    s = (slug or "").strip()
    if s:
        return s
    return (os.environ.get("KC_BEACON_SLUG", "") or "").strip()


def kc_kaggle_path(entries=None):
    """`import kaggle` を壊すパスを取り除いた `sys.path` を返す。

    リポジトリ直下には `kaggle/`（カーネルのビルド元）がある。これが
    **インストール済みの `kaggle` パッケージを隠す**ので、ローカルで
    `cli beacon` を叩くと `No module named 'kaggle.api'` で死ぬ（実測）。
    カーネル内には `kaggle/` ディレクトリが無いので、ここは素通りする。

    判定は「`kaggle/` があり、かつ `kaggle/api` が無い」パスだけを外す。
    `kaggle/api` を持つ本物の候補は残すので、site-packages は消えない。
    """
    src = list(sys.path if entries is None else entries)
    keep = []
    for p in src:
        base = p or "."
        try:
            shadow = (os.path.isdir(os.path.join(base, "kaggle"))
                      and not os.path.isdir(os.path.join(base, "kaggle", "api")))
        except OSError:                       # 権限の無いパスは触らない
            shadow = False
        if not shadow:
            keep.append(p)
    return keep


def kc_kaggle_api():
    """`KaggleApi` クラスを返す。無ければ例外を投げる（呼び出し側が拾う）。

    普通に import して駄目だったときだけ `kc_kaggle_path` で探し直す。
    一度失敗した `kaggle` は名前空間パッケージとして `sys.modules` に
    残っているので、それも捨てないと再試行が同じ場所へ行く。
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi
    except Exception:
        saved = list(sys.path)
        try:
            sys.path[:] = kc_kaggle_path(saved)
            sys.modules.pop("kaggle", None)
            sys.modules.pop("kaggle.api", None)
            from kaggle.api.kaggle_api_extended import KaggleApi
            return KaggleApi
        finally:
            sys.path[:] = saved


def kc_beacon_meta(slug):
    """beacon カーネルの kernel-metadata.json を組み立てる。

    **実行しない掲示板**なので GPU もインターネットも要らない。非公開にするのは、
    URL が「知っていれば誰でも入れる入口」だから（Gist と同じ扱い）。
    """
    name = slug.rsplit("/", 1)[-1]
    return {
        "id": slug,
        "title": name,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def kc_beacon_body(url, stage="", updated=None):
    """掲示板の main.py の中身。"""
    return _BEACON_BODY.format(
        url=url, stage=stage, updated=time.time() if updated is None else updated)


def kc_beacon_publish(url, stage="", slug="", timeout=180):
    """掲示板を更新する。戻り値は `(ok, detail)`。

    `detail` は成功なら `"pushed v3 (platform)"` のような内訳、失敗なら理由。
    **例外は投げない**（配信を止めない）。呼び出し側は成否を `result` に残す。
    """
    s = kc_beacon_slug(slug)
    if not s:
        return False, "beacon の slug が無い（KC_BEACON_SLUG か引数で渡す）"
    if "/" not in s:
        return False, f"slug は owner/name の形で渡す: {s!r}"
    try:
        KaggleApi = kc_kaggle_api()
    except Exception as e:
        return False, f"kaggle パッケージが無い（{type(e).__name__}: {e}）"

    d = tempfile.mkdtemp(prefix="beacon-")
    try:
        with open(os.path.join(d, "main.py"), "w", encoding="utf-8") as f:
            f.write(kc_beacon_body(url, stage))
        with open(os.path.join(d, "kernel-metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(kc_beacon_meta(s), f, indent=2, ensure_ascii=False)

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            return False, (f"Kaggle 認証が無い（{type(e).__name__}: {e}）"
                           " — カーネル外からは ~/.kaggle/kaggle.json が要る")
        try:
            resp = api.kernels_push(d)
        except SystemExit as e:                      # kaggle は終了コードで返す
            return False, f"kernels_push が終了した（SystemExit {e.code}）"
        except Exception as e:
            return False, f"kernels_push 失敗（{type(e).__name__}: {e}）"
        ver = getattr(resp, "version_number", None)
        return True, f"pushed v{ver} ({s})"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def kc_beacon_parse(text):
    """掲示板のソースから `url` / `stage` / `updated` を取り出す。

    `exec` しない。中身は自分で書いたものだが、読む側（VM）にとっては
    ネットワーク越しの入力なので、評価せずに読む。
    """
    out = {"url": "", "stage": "", "updated": None}
    keys = {"URL": "url", "STAGE": "stage", "UPDATED": "updated"}
    for line in (text or "").splitlines():
        s = line.strip()
        for key, field in keys.items():
            if not s.startswith(key + " ="):
                continue
            raw = s.split("=", 1)[1].strip()
            if field == "updated":
                try:
                    out[field] = float(raw)
                except ValueError:
                    out[field] = None
            else:
                out[field] = raw.strip("\"'")
    return out


def kc_beacon_is_fresh(data, since, slack=5.0):
    """`since` より後に発行されたものだけを本物とみなす。

    2026-11-01 に**古いテスト URL を掴む**事故が出ている。push の前後で
    スタンプが変わったことを確かめないと、「まだ前の値」を新しい値として
    読んでしまう。`slack` は VM と Kaggle の時計のずれぶん。
    """
    u = (data or {}).get("updated")
    if not isinstance(u, (int, float)):
        return False
    return u >= (float(since) - float(slack))


def kc_beacon_pull_cli(slug, dest, exe="", timeout=180):
    """`kaggle kernels pull` を**CLI で**実行する。戻り値は `(ok, detail)`。

    Python パッケージの `kaggle` が import できない環境のための逃げ道。
    手元の python3（3.13）には入っておらず、`kaggle` CLI だけが 3.9 の
    user site に入っている、という状態で実際に踏んだ（`cli beacon` が
    `No module named 'kaggle'` で死んでいた）。カーネル内では使われない。
    """
    import subprocess                      # 束ねたときの F811 を避けて局所 import
    cmd = [exe or shutil.which("kaggle") or "", "kernels", "pull", slug, "-p", dest]
    if not cmd[0]:
        return False, "kaggle パッケージも kaggle CLI も見つからない"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return False, f"kaggle CLI の起動に失敗（{type(e).__name__}: {e}）"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or [""]
        return False, f"kaggle CLI が {p.returncode} で終了（{tail[0][:200]}）"
    return True, "kaggle CLI"


def kc_beacon_pull(slug="", dest=None, timeout=180, exe=""):
    """掲示板を引いて読む。戻り値は `(ok, detail, data)`。

    `data` は `kc_beacon_parse` の結果（失敗なら空）。
    `exe` は CLI 経由に落ちたときに使う `kaggle` のパス（`cli push` と同じ
    解決順を呼び出し側が渡す）。空なら PATH から探す。
    """
    s = kc_beacon_slug(slug)
    if not s:
        return False, "beacon の slug が無い", {}
    d = dest or tempfile.mkdtemp(prefix="beacon-pull-")
    try:
        # まず Python パッケージ。駄目なら CLI に落ちる（理由は捨てない）。
        why_api = ""
        try:
            KaggleApi = kc_kaggle_api()
        except Exception as e:
            why_api = f"{type(e).__name__}: {e}"
        else:
            try:
                api = KaggleApi()
                api.authenticate()
                api.kernels_pull(s, path=d)
                why_api = ""
            except SystemExit as e:          # kaggle は終了コードで返す
                why_api = f"kernels_pull が終了した（SystemExit {e.code}）"
            except Exception as e:
                why_api = f"kernels_pull 失敗（{type(e).__name__}: {e}）"
        if why_api:
            ok, detail = kc_beacon_pull_cli(s, d, exe=exe, timeout=timeout)
            if not ok:
                return False, f"API でも CLI でも引けなかった（API: {why_api} / CLI: {detail}）", {}
        # 引いたファイル名は版によって変わる（main.py / <name>.py）ので、
        # 中身で選ぶ。UPDATED を含む .py が掲示板。
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                with open(os.path.join(root, fn), encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
                if "UPDATED" in text:
                    data = kc_beacon_parse(text)
                    return True, f"pulled {fn}", data
        return False, "pull した中に掲示板（UPDATED を含む .py）が無い", {}
    finally:
        if dest is None:
            shutil.rmtree(d, ignore_errors=True)



# ── エンジン固有の定数（旧 settings セル）──────────────────────
# 骨格が読む名前（PORT / ALIAS / CONTEXT_LENGTH など）はここで上書きできる。
# 差は「データ」だけにして、骨格には手を入れない。
import os
os.environ.setdefault('VLLM_SAFETENSORS_STRATEGY', 'prefetch')
os.environ.setdefault('VLLM_SAFETENSORS_PREFETCH_THREADS', '1')
os.environ.setdefault('VLLM_STARTUP_BUDGET', '7000')
os.environ.setdefault('VLLM_ATTEMPT_TIMEOUT', '7000')

# ============================================================
# Step 2 (vLLM): 配信カーネルの設定
# ============================================================
import json
import os
import subprocess

TIMINGS_PATH = "/kaggle/working/timings.json"


def kc_save_timings(d):
    """計測値を timings.json に足す（result.json へはテンプレートの save() が合流）。

    セルから `data` を直接触らないのは、**notebook 版には `data` が無い**ため
    （セル順の「定義前使用」で落ちる。2026-11-01 実測）。ファイル経由なら
    notebook / script の両方で同じコードが動く。

    計測は「高度化の基礎」なので、失敗しても本処理は止めない。
    """
    try:
        with open(TIMINGS_PATH, encoding="utf-8") as f:
            cur = json.load(f)
    except (OSError, ValueError):
        cur = {}
    cur.update(d)
    try:
        with open(TIMINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except OSError as _e:
        print(f"⚠ timings.json を書けません: {_e}")

# Dataset のマウント先は共通ランタイム（runtime/storage.py）に一本化。
# 旧 /kaggle/input/<slug> と新 /kaggle/input/datasets/<owner>/<slug> の
# 両対応・空マウント回避は kc_find_dataset が持つ。
MODEL_SRC = kc_find_dataset("qwen38-27b-uncensored-hauhaucs-awq-mtp")
# 共有セルは MODEL_NAME を読む。vLLM の文脈では served-model-name と同義。
SERVED_NAME = MODEL_NAME = "qwen3.8-27b-awq"
MODEL_DISPLAY = "Qwen3.8-27B-Uncensored (AWQ)"
ENGINE_DISPLAY = "vLLM"
ENGINE_LABEL = "vllm"

# ---- ポート ----
# vLLM 本体は 8000、公開側はプロキシが 11435 で受ける。
PORT = UPSTREAM_PORT = VLLM_PORT = 8000
PROXY_PORT = 11435
PROXY_CHECK_PATH = "/v1/models"
# クライアントが temperature=0 / repetition_penalty 未指定で送っても
# 崩壊しないよう、サーバ側で下限を矯正する（RP>=1.05 / temp>0）。
PROXY_EXTRA_ARGS = ("--enforce-sampling",)
TUNNEL_KIND = "cloudflared"

# Basic 認証プロキシ / UI のソース（ビルダーが埋め込む）
PROXY_SRC_B64 = "IyDilIDilIAgcnVudGltZS9xdW90YS5wee+8iGF1dGhfcHJveHkucHkg44Gr5ZCM5qKx77yJCiIiIkthZ2dsZSDjga7jgq/jgqnjg7zjgr8gQVBJ77yIR1BVL1RQVSDmrovph4/vvInjgpLoqq3jgoDlhbHpgJrlh6bnkIbvvIjnlJ/miJDjgrPjg7zjg4njgbjln4vjgoHovrzjgoDvvInjgIIKCmthZ2dsZSBDTEkg44Gr44GvIHF1b3RhIOOCteODluOCs+ODnuODs+ODieOBjOeEoeOBhOOBjOOAgVdlYiBBUEkg44GvIEJhc2ljIOiqjeiovOOBp+iqreOCgeOCi+OAggrlkbzjgbPlh7rjgZflgbTjga8gYXV0aF9wcm94eS5wee+8iC9xdW90Ye+8ieOBqCBrYWdnbGUtcXVvdGEucHnvvIhDTEkg6KGo56S677yJ44CCCiIiIgppbXBvcnQgYmFzZTY0CmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdXJsbGliLnJlcXVlc3QKCktDX1FVT1RBX1VSTCA9ICJodHRwczovL3d3dy5rYWdnbGUuY29tL2FwaS92MS9rZXJuZWxzL3F1b3RhIgpLQ19DUkVEX1BBVEggPSBvcy5wYXRoLmV4cGFuZHVzZXIoIn4vLmthZ2dsZS9rYWdnbGUuanNvbiIpCgoKZGVmIGtjX2thZ2dsZV9jcmVkZW50aWFscyhjcmVkX3BhdGg9Tm9uZSwgZW52aXJvbj1Ob25lKToKICAgICIiIih1c2VybmFtZSwga2V5KSDjgpLov5TjgZnjgIJrYWdnbGUuanNvbiDihpIg55Kw5aKD5aSJ5pWw44Gu6aCG44CC54Sh44GR44KM44GwIChOb25lLCBOb25lKeOAggoKICAgIOOCq+ODvOODjeODq+OBq+OBryBrYWdnbGUuanNvbiDjgYznhKHjgYTjgYzjgIFLQUdHTEVfVVNFUk5BTUUgLyBLQUdHTEVfS0VZIOOBjAogICAg55Kw5aKD5aSJ5pWw44Gn5YWl44Gj44Gm44GE44KL44CCCiAgICAiIiIKICAgIGNyZWRfcGF0aCA9IEtDX0NSRURfUEFUSCBpZiBjcmVkX3BhdGggaXMgTm9uZSBlbHNlIGNyZWRfcGF0aAogICAgZW52aXJvbiA9IG9zLmVudmlyb24gaWYgZW52aXJvbiBpcyBOb25lIGVsc2UgZW52aXJvbgogICAgaWYgb3MucGF0aC5pc2ZpbGUoY3JlZF9wYXRoKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHdpdGggb3BlbihjcmVkX3BhdGgpIGFzIGY6CiAgICAgICAgICAgICAgICBjZmcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgdXNlciwga2V5ID0gY2ZnLmdldCgidXNlcm5hbWUiKSwgY2ZnLmdldCgia2V5IikKICAgICAgICAgICAgaWYgdXNlciBhbmQga2V5OgogICAgICAgICAgICAgICAgcmV0dXJuIHVzZXIsIGtleQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgIHVzZXIgPSBlbnZpcm9uLmdldCgiS0FHR0xFX1VTRVJOQU1FIikgb3IgZW52aXJvbi5nZXQoIktBR0dMRV9VU0VSIikKICAgIGtleSA9IGVudmlyb24uZ2V0KCJLQUdHTEVfS0VZIikKICAgIHJldHVybiAodXNlciwga2V5KSBpZiB1c2VyIGFuZCBrZXkgZWxzZSAoTm9uZSwgTm9uZSkKCgpkZWYga2NfZmV0Y2hfcXVvdGEodXNlciwga2V5LCB0aW1lb3V0PTQ1LCB1cmw9S0NfUVVPVEFfVVJMLCBvcGVuZXI9Tm9uZSk6CiAgICAiIiLjgq/jgqnjg7zjgr/jga7nlJ8gSlNPTiDjgpLov5TjgZnjgIJIVFRQIC8g5o6l57aa44Ko44Op44O844Gv5L6L5aSW44Gu44G+44G+5oqV44GS44KL44CCIiIiCiAgICB0b2tlbiA9IGJhc2U2NC5iNjRlbmNvZGUoZiJ7dXNlcn06e2tleX0iLmVuY29kZSgpKS5kZWNvZGUoKQogICAgcmVxID0gdXJsbGliLnJlcXVlc3QuUmVxdWVzdCgKICAgICAgICB1cmwsIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJhc2ljICIgKyB0b2tlbn0pCiAgICB3aXRoIChvcGVuZXIgb3IgdXJsbGliLnJlcXVlc3QudXJsb3BlbikocmVxLCB0aW1lb3V0PXRpbWVvdXQpIGFzIHI6CiAgICAgICAgcmV0dXJuIGpzb24ubG9hZHMoci5yZWFkKCkpCgoKZGVmIGtjX3F1b3RhX3N1bW1hcnkoZGF0YSk6CiAgICAiIiJVSSDnlKjjga4gR1BVIOOCr+OCqeODvOOCv+imgee0hOOAgmdwdVF1b3RhIOOBjOeEoeOBhOW/nOetlOOBryBhdmFpbGFibGU9RmFsc2XjgIIiIiIKICAgIGcgPSBkYXRhLmdldCgiZ3B1UXVvdGEiKSBvciB7fQogICAgdXNlZCA9IChnLmdldCgidGltZVVzZWQiKSBvciB7fSkuZ2V0KCJzZWNvbmRzIikKICAgIHRvdGFsID0gKGcuZ2V0KCJ0b3RhbFRpbWVBbGxvd2VkIikgb3Ige30pLmdldCgic2Vjb25kcyIpCiAgICBpZiB1c2VkIGlzIE5vbmUgb3IgdG90YWwgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJhdmFpbGFibGUiOiBGYWxzZSwgInJlYXNvbiI6ICJub19ncHVfcXVvdGEifQogICAgcmVzZXJ2ZWQgPSAoZy5nZXQoInRpbWVSZXNlcnZlZCIpIG9yIHt9KS5nZXQoInNlY29uZHMiLCAwKQogICAgcmV0dXJuIHsKICAgICAgICAiYXZhaWxhYmxlIjogVHJ1ZSwKICAgICAgICAidXNlZFNlY29uZHMiOiB1c2VkLAogICAgICAgICJ0b3RhbFNlY29uZHMiOiB0b3RhbCwKICAgICAgICAicmVzZXJ2ZWRTZWNvbmRzIjogcmVzZXJ2ZWQsCiAgICAgICAgImxlZnRTZWNvbmRzIjogdG90YWwgLSB1c2VkIC0gcmVzZXJ2ZWQsCiAgICAgICAgInJlZnJlc2hUaW1lIjogZGF0YS5nZXQoInF1b3RhUmVmcmVzaFRpbWUiLCAiIiksCiAgICB9CgojIS91c3IvYmluL2VudiBweXRob24zCiIiIkJhc2ljIOiqjeiovOOBpOOBjeODquODkOODvOOCueODl+ODreOCreOCt++8iOaomea6luODqeOCpOODluODqeODquOBruOBv++8ieOAggoKY2xvdWRmbGFyZWQg44GuIHF1aWNrIHR1bm5lbCDjgavjga8gQmFzaWMg6KqN6Ki85qmf6IO944GM54Sh44GE44Gf44KB44CBCk9sbGFtYSDjga7miYvliY3jgavjgZPjgozjgpLnva7jgYTjgaboqo3oqLzjgpLmi4Xjgo/jgZvjgovjgIIKCiAgICBBUElfVVNFUj1vbGxhbWEgQVBJX1BBU1M9c2VjcmV0IHB5dGhvbjMgYXV0aF9wcm94eS5weSBcCiAgICAgICAgLS1wb3J0IDExNDM1IC0tdXBzdHJlYW0gaHR0cDovLzEyNy4wLjAuMToxMTQzNAoK44K544OI44Oq44O844Of44Oz44Kw77yIT2xsYW1hIOOBriB0b2tlbiDpgJDmrKHlh7rlipvvvInjgpLpgJrjgZnjgZ/jgoHjgIEK5LiK5rWB44Gu5b+c562U44GvIGNodW5rZWQgdHJhbnNmZXItZW5jb2Rpbmcg44Gn44Gd44Gu44G+44G+5Lit57aZ44GZ44KL44CCCiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBiYXNlNjQKaW1wb3J0IGhtYWMKaW1wb3J0IGh0dHAuc2VydmVyCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgcGF0aGxpYgppbXBvcnQgc2lnbmFsCmltcG9ydCBzb2NrZXRzZXJ2ZXIKaW1wb3J0IHN5cwppbXBvcnQgdGhyZWFkaW5nCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5wYXJzZQppbXBvcnQgdXJsbGliLnJlcXVlc3QKCiMg5LiK5rWB77yIT2xsYW1h77yJ44G444Gu5b6F44Gh5pmC6ZaT44CC6ZW344GE55Sf5oiQ44Gr6ICQ44GI44KL44KI44GG6ZW344KB44Gr5Y+W44KL44CCClVQU1RSRUFNX1RJTUVPVVQgPSAzNjAwCiMg5Lit57aZ44Gu44OQ44OD44OV44Kh5LiK6ZmQ44CC5pS56KGM77yITkRKU09OIOihjO+8ieOBjOadpeOBquOBhOWgtOWQiOOBruODleOCqeODvOODq+ODkOODg+OCr+OBp+OAgQojIOOBk+OBk+OBq+mBlOOBmeOCi+OBvuOBp+W+heOBpOOBqOOCueODiOODquODvOODoOOBjOa6nOOBvuOCi+OBruOBp+OAgeWwj+OBleOBhOOBu+OBqemAkOasoeaAp+OBr+mrmOOBhOOAggpDSFVOSyA9IDQwOTYKIyBCYXNpYyDoqo3oqLzjgpLlhY3pmaTjgZnjgovjg5HjgrnjgIIKIyAvaGVhbHRoeiDjga/nlo7pgJrnorroqo3jgIEvIOOBqCAvY2hhdCDjga/jg4Hjg6Pjg4Pjg4ggVUnvvIjpnZnnmoTjgaogSFRNTCDjga7jgb/jgIIKIyDoqo3oqLzmg4XloLHjgoLjg6Ljg4fjg6vjgoLlkKvjgb7jgarjgYTjga7jgafjgIHlhazplovjgZfjgabjgoLmvI/jgozjgovjgoLjga7jga/nhKHjgYTvvInjgIIKIyBBUEkg5pys5L2T77yIL2FwaS8qLCAvdjEvKu+8ieOBr+W+k+adpeOBqeOBiuOCiiBCYXNpYyDoqo3oqLzjgYzopoHjgovjgIIKT1BFTl9QQVRIUyA9IHsiL2hlYWx0aHoiLCAiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCIsICIvZmF2aWNvbi5pY28ifQpVSV9QQVRIUyA9IHsiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCJ9CgojIOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiO+8iOS4iua1geOBuOa1geOBleOBquOBhO+8ieOAggojIC9xdW90YSA6IEthZ2dsZSDjga4gR1BVIOOCr+OCqeODvOOCv+aui+mHj++8iFVJIOOBruODmOODg+ODgOOBq+WHuuOBme+8iQojIC9zdG9wICA6IOOBk+OBruS4ree2meOBqOS4iua1geOBruOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCi+OAggojICAgICAgICAgIFBPU1QgKyDjg5jjg4Pjg4AgWC1LYy1TdG9wOiAxIOOBjOW/hemgiO+8iEdFVCDjga7jgrnjgq3jg6Pjg4rjgoTku5bjgrXjgqTjg4jjgYvjgonjga4KIyAgICAgICAgICBDU1JGIOOBp+iQveOBqOOBleOCjOOBquOBhOOCiOOBhuOBq++8ieOAgkFQSV9QQVNTIOioreWumuaZguOBryBCYXNpYyDoqo3oqLzjgoLopoHjgovjgIIKQURNSU5fUEFUSFMgPSB7Ii9xdW90YSIsICIvc3RvcCJ9ClNUT1BfSEVBREVSID0gIlgtS2MtU3RvcCIKCiMga2VlcGFsaXZlIOOCkuW7tumVt+OBmeOCi+OBruOBr+OAjOeUn+aIkOOAjeODquOCr+OCqOOCueODiOOBoOOBkeOAgi9hcGkvcHMg44KEIC92MS9tb2RlbHMg44GuCiMg44Od44O844Oq44Oz44Kw77yI44Op44Oz44OB44Oj44O844Gu6LW35YuV56K66KqN44O744K544Kt44Oj44OK77yJ44GnIEdQVSDjgrvjg4Pjgrfjg6fjg7PjgpLlu7blkb3jgZfjgarjgYTjgIIKR0VORVJBVElPTl9QQVRIUyA9IHsKICAgICIvYXBpL2NoYXQiLAogICAgIi9hcGkvZ2VuZXJhdGUiLAogICAgIi92MS9jaGF0L2NvbXBsZXRpb25zIiwKICAgICIvdjEvY29tcGxldGlvbnMiLAogICAgIi92MS9yZXNwb25zZXMiLAp9CgojIOOCr+OCqeODvOOCv+Wun+ijheOBryBydW50aW1lL3F1b3RhLnB5IOOBjOecn+Wun+a6kOOAguOCq+ODvOODjeODq+OBp+OBr+ODk+ODq+ODgOODvOOBjOWQjOOCveODvOOCueOCkgojIOWFiOmgreOBq+mAo+e1kOOBl+OBpuWfi+OCgei+vOOCgOOBn+OCgeOAgWtjXyog44GM5pyq5a6a576p44Gu44Ot44O844Kr44Or5a6f6KGM44Gg44GRIGltcG9ydCDjgZnjgovjgIIKaWYgImtjX2ZldGNoX3F1b3RhIiBub3QgaW4gZ2xvYmFscygpOgogICAgc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihwYXRobGliLlBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnRzWzJdKSkKICAgIGZyb20gcnVudGltZSBpbXBvcnQgcXVvdGEgYXMgX2tjX3F1b3RhCiAgICBrY19mZXRjaF9xdW90YSA9IF9rY19xdW90YS5rY19mZXRjaF9xdW90YQogICAga2NfcXVvdGFfc3VtbWFyeSA9IF9rY19xdW90YS5rY19xdW90YV9zdW1tYXJ5CiAgICBrY19rYWdnbGVfY3JlZGVudGlhbHMgPSBfa2NfcXVvdGEua2Nfa2FnZ2xlX2NyZWRlbnRpYWxzCgpIT1BfQllfSE9QID0gewogICAgImNvbm5lY3Rpb24iLAogICAgImtlZXAtYWxpdmUiLAogICAgInByb3h5LWF1dGhlbnRpY2F0ZSIsCiAgICAicHJveHktYXV0aG9yaXphdGlvbiIsCiAgICAidGUiLAogICAgInRyYWlsZXJzIiwKICAgICJ0cmFuc2Zlci1lbmNvZGluZyIsCiAgICAidXBncmFkZSIsCiAgICAiZXhwZWN0IiwKfQoKQVJHUyA9IE5vbmUKRVhQRUNURURfQVVUSCA9ICIiCiMg44OB44Oj44OD44OIIFVJIOOBriBIVE1M44CC6LW35YuV5pmC44Gr5LiA5bqm44Gg44GR6Kqt44KA77yI5q+O44Oq44Kv44Ko44K544OI44Gn44OH44Kj44K544Kv44KS5Y+p44GL44Gq44GE77yJ44CCCkNIQVRfQllURVMgPSBiIiIKCiMgLS0tLSDjgrXjg7Pjg5fjg6rjg7PjgrDlvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg44Gu44Go44GN44Gg44GR5pyJ5Yq577yJLS0tLQojIHZMTE0g54mI44Gn5b+F6KaB44CC44Kv44Op44Kk44Ki44Oz44OI77yIb3BlbmNvZGUg562J77yJ44GMIHRlbXBlcmF0dXJlPTAg44KS6YCB44KL44GoCiMg6LKq5qyy5rOV44Gr44Gq44KK44CBMjdCIOe0muOBp+OBr+WQjOOBmOaWh+OBrue5sOOCiui/lOOBl+OBp+W0qeWjiuOBl+OChOOBmeOBhOOAggojIHJlcGV0aXRpb25fcGVuYWx0eSDjgoLmnKrmjIflrprjgaDjgaggMS4w77yIPeeEoeWKue+8ieOBq+OBquOCi+OBn+OCgeS4i+mZkOOCkuWFpeOCjOOCi+OAggojIOaXouWumuOBruODl+ODreOCreOCt++8iE9sbGFtYSDniYjvvInjgafjga/nhKHlirnjgarjga7jgafmjJnli5Xjga/lpInjgo/jgonjgarjgYTjgIIKUlBfRkxPT1IgPSAxLjA1ICAgICAjIHJlcGV0aXRpb25fcGVuYWx0eSDjga7kuIvpmZAKVEVNUF9GTE9PUiA9IDAuNyAgICAjIHRlbXBlcmF0dXJlIOOBruS4i+mZkO+8iDAgPSDosqrmrLLms5XjgpLnpoHmraLvvIkKCgojIOODiOODs+ODjeODqy/jg4Hjg6Pjg4Pjg4jntYznlLHjga7jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgacga2VlcGFsaXZlIOaZgumWk+OCkuODquOCu+ODg+ODiOOAggojIC90bXAva2VlcGFsaXZlLmNvbmYg44GuIFJVTj0xLCBNSU5VVEVTPTYwIOOCkuWGjeabuOOBjei+vOOBv+OAggpfS0VFUEFMSVZFX0NPTkYgPSAiL3RtcC9rZWVwYWxpdmUuY29uZiIKCgpkZWYgX3Jlc2V0X2tlZXBhbGl2ZSgpOgogICAgIiIi44OB44Oj44OD44OI6YCB5L+h44GU44Go44Gr5pmC6ZaT44Oq44K744OD44OI77yIT04g57at5oyB77yJ44CC5LiK5rWB44Ko44Op44O85pmC44Gn44KC5ZG844G244CCIiIiCiAgICB0cnk6CiAgICAgICAgd2l0aCBvcGVuKF9LRUVQQUxJVkVfQ09ORiwgInciKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKCJSVU49MVxuTUlOVVRFUz02MFxuIikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKCmRlZiBfaXNfZ2VuZXJhdGlvbihwYXRoX29ubHkpOgogICAgcmV0dXJuIHBhdGhfb25seS5yc3RyaXAoIi8iKSBpbiBHRU5FUkFUSU9OX1BBVEhTCgoKZGVmIF9udW0odmFsdWUpOgogICAgIiIi5pWw5YCk5YyW44Gn44GN44Gq44GR44KM44GwIE5vbmXvvIhib29sIOOBr+aVsOWApOaJseOBhOOBl+OBquOBhO+8ieOAgiIiIgogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgYm9vbCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHRyeToKICAgICAgICByZXR1cm4gZmxvYXQodmFsdWUpCiAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcmV0dXJuIE5vbmUKCgpkZWYgZW5mb3JjZV9zYW1wbGluZyhib2R5LCBwYXRoX29ubHkpOgogICAgIiIiY2hhdC9jb21wbGV0aW9ucyDjga7jgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjgpLlvLfliLbjgZnjgovvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC77yJ44CCCgogICAgLSByZXBldGl0aW9uX3BlbmFsdHkgPCBSUF9GTE9PUu+8iOacquaMh+WumuWQq+OCgO+8ieKGkiBSUF9GTE9PUgogICAgLSB0ZW1wZXJhdHVyZSA8PSAw77yI5pyq5oyH5a6a5ZCr44KA77yJ4oaSIFRFTVBfRkxPT1IKCiAgICDjg5zjg4fjgqPjgYwgSlNPTiDjgafjgarjgYQv5aOK44KM44Gm44GE44KL5aC05ZCI44Gv57Sg6YCa44GX77yI5LiK5rWB44Gr5Yik5pat44KS5aeU44Gt44KL77yJ44CCCiAgICDmiLvjgorlgKTjga/lt67jgZfmm7/jgYjlvozjga4gYnl0ZXPjgIJDb250ZW50LUxlbmd0aCDjga/lkbzjgbPlh7rjgZflgbTjgafku5jjgZHnm7TjgZnjgIIKICAgICIiIgogICAgaWYgbm90IEFSR1MuZW5mb3JjZV9zYW1wbGluZyBvciBub3QgYm9keToKICAgICAgICByZXR1cm4gYm9keQogICAgIyDjgq/jgqjjg6rmloflrZfliJfvvIg/YXBpLXZlcnNpb249Li4uIOetie+8ieOBjOS7mOOBj+OBqCBlbmRzd2l0aCDjgYzlpJbjgozjgovjgZ/jgoHjgIEKICAgICMg44OR44K544Gg44GR44KS5Y+W44KK5Ye644GX44Gm5q+U6LyD44GZ44KL44CC5Y+W44KK44GT44G844GZ44Go55+v5q2j44GM54Sh5Yq55YyW44GV44KM44KL44CCCiAgICBfcGF0aCA9IHVybGxpYi5wYXJzZS51cmxzcGxpdChwYXRoX29ubHkpLnBhdGgKICAgIGlmIG5vdCBfcGF0aC5yc3RyaXAoIi8iKS5lbmRzd2l0aCgiL2NoYXQvY29tcGxldGlvbnMiKToKICAgICAgICByZXR1cm4gYm9keQogICAgdHJ5OgogICAgICAgIGRhdGEgPSBqc29uLmxvYWRzKGJvZHkpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBib2R5CiAgICBpZiBub3QgaXNpbnN0YW5jZShkYXRhLCBkaWN0KToKICAgICAgICByZXR1cm4gYm9keQogICAgY2hhbmdlZCA9IEZhbHNlCiAgICBycCA9IF9udW0oZGF0YS5nZXQoInJlcGV0aXRpb25fcGVuYWx0eSIpKQogICAgaWYgcnAgaXMgTm9uZSBvciBycCA8IFJQX0ZMT09SOgogICAgICAgIGRhdGFbInJlcGV0aXRpb25fcGVuYWx0eSJdID0gUlBfRkxPT1IKICAgICAgICBjaGFuZ2VkID0gVHJ1ZQogICAgdGVtcCA9IF9udW0oZGF0YS5nZXQoInRlbXBlcmF0dXJlIikpCiAgICBpZiB0ZW1wIGlzIE5vbmUgb3IgdGVtcCA8PSAwOgogICAgICAgIGRhdGFbInRlbXBlcmF0dXJlIl0gPSBURU1QX0ZMT09SCiAgICAgICAgY2hhbmdlZCA9IFRydWUKICAgIGlmIG5vdCBjaGFuZ2VkOgogICAgICAgIHJldHVybiBib2R5CiAgICByZXR1cm4ganNvbi5kdW1wcyhkYXRhLCBlbnN1cmVfYXNjaWk9RmFsc2UpLmVuY29kZSgidXRmLTgiKQoKCmNsYXNzIEhhbmRsZXIoaHR0cC5zZXJ2ZXIuQmFzZUhUVFBSZXF1ZXN0SGFuZGxlcik6CiAgICBwcm90b2NvbF92ZXJzaW9uID0gIkhUVFAvMS4xIgogICAgc2VydmVyX3ZlcnNpb24gPSAiT2xsYW1hQXV0aFByb3h5LzEuMCIKICAgICMg5o6l57aa44GU44Go44Gu44K944Kx44OD44OI44K/44Kk44Og44Ki44Km44OI44CC44Kv44Op44Kk44Ki44Oz44OI44GM54Sh6YCa5L+h44Gu44G+44G+5q6L44GZ44GoCiAgICAjIOODj+ODs+ODieODqeOCueODrOODg+ODieOBjOeEoemZkOOBq+WxheW6p+OCi+OBruOBp+OAgTXliIbjgaflm57lj47jgZnjgovvvIjjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK0KICAgICMg44Gu5LiK5rWB6Kqt44G/5Y+W44KK44GvIHVybGxpYiDlgbTjgr3jgrHjg4Pjg4jjgarjga7jgaflvbHpn7/jgpLlj5fjgZHjgarjgYTvvInjgIIKICAgIHRpbWVvdXQgPSAzMDAKCiAgICAjIC0tLS0g44Ot44Kw44KSIDEg6KGM44Gr5Zyn57iu77yI44OO44O844OI44OW44OD44Kv44Gu44Ot44Kw44KS5rGa44GV44Gq44GE77yJLS0tLQogICAgZGVmIGxvZ19tZXNzYWdlKHNlbGYsIGZtdCwgKmFyZ3MpOgogICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSAlc1xuIiAlIChmbXQgJSBhcmdzKSkKCiAgICBkZWYgX3JlYWRfYm9keShzZWxmKToKICAgICAgICAiIiLjg6rjgq/jgqjjgrnjg4jjg5zjg4fjgqPjgpLoqq3jgoDjgIJjaHVua2VkIOOBp+OCgiBDb250ZW50LUxlbmd0aCDjgafjgoLkuKHlr77lv5zjgIIKCiAgICAgICAgQ29udGVudC1MZW5ndGgg44GX44GL6KaL44Gq44GE44Go44CBSFRUUC8xLjEg44GuIGNodW5rZWQgUE9TVO+8iOOCueODiOODquODvOODn+ODs+OCsOWFpeWKm+OChAogICAgICAgIOODl+ODreOCreOCt+e1jOeUseOBruWkieaPm+OBp+eZuueUn+OBl+OBhuOCi++8ieOBpyoq44Oc44OH44Kj44GU44Go5qyg6JC9KirjgZfjgabkuIrmtYHjgavlsYrjgYvjgarjgYTjgIIKICAgICAgICAiIiIKICAgICAgICB0ZSA9IChzZWxmLmhlYWRlcnMuZ2V0KCJUcmFuc2Zlci1FbmNvZGluZyIpIG9yICIiKS5sb3dlcigpCiAgICAgICAgaWYgImNodW5rZWQiIGluIHRlOgogICAgICAgICAgICBvdXQgPSBieXRlYXJyYXkoKQogICAgICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICAgICAgc2l6ZV9saW5lID0gc2VsZi5yZmlsZS5yZWFkbGluZSgpLnN0cmlwKCkKICAgICAgICAgICAgICAgIGlmIGIiOyIgaW4gc2l6ZV9saW5lOiAgICAgICAgICAjIGNodW5rLWV4dCDjga/mjajjgabjgosKICAgICAgICAgICAgICAgICAgICBzaXplX2xpbmUgPSBzaXplX2xpbmUuc3BsaXQoYiI7IiwgMSlbMF0KICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBzaXplID0gaW50KHNpemVfbGluZSwgMTYpCiAgICAgICAgICAgICAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICAgICAgaWYgc2l6ZSA9PSAwOiAgICAgICAgICAgICAgICAgICMg44OI44Os44O844Op44O86Kqt44G/5o2o44GmCiAgICAgICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICAgICAgdHJhaWxlciA9IHNlbGYucmZpbGUucmVhZGxpbmUoKQogICAgICAgICAgICAgICAgICAgICAgICBpZiB0cmFpbGVyIGluIChiIlxyXG4iLCBiIlxuIiwgYiIiKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgIG91dCArPSBzZWxmLnJmaWxlLnJlYWQoc2l6ZSkKICAgICAgICAgICAgICAgIHNlbGYucmZpbGUucmVhZCgyKSAgICAgICAgICAgICAjIOODgeODo+ODs+OCr+acq+WwvuOBriBDUkxGCiAgICAgICAgICAgIHJldHVybiBieXRlcyhvdXQpIG9yIE5vbmUKCiAgICAgICAgbGVuZ3RoID0gaW50KHNlbGYuaGVhZGVycy5nZXQoIkNvbnRlbnQtTGVuZ3RoIikgb3IgMCkKICAgICAgICByZXR1cm4gc2VsZi5yZmlsZS5yZWFkKGxlbmd0aCkgaWYgbGVuZ3RoIGVsc2UgTm9uZQoKICAgIGRlZiBfZW1pdChzZWxmLCBkYXRhKToKICAgICAgICAiIiJjaHVua2VkIOOBrjHliIbniYfjgpLpgIHjgovjgIIKCiAgICAgICAg5pS56KGM5Y2Y5L2N44Gn5ZG844G244Gu44GnIHNlbmQg5Zue5pWw44Gv44OI44O844Kv44Oz5pWw44Gr5ZCM5q+U5L6L77yIMeODkOOCpOODiOavjuOBq+OBr+mAgeOCieOBquOBhO+8ieOAggogICAgICAgICIiIgogICAgICAgIGlmIG5vdCBkYXRhOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiJXhcclxuIiAlIGxlbihkYXRhKSkKICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJ5dGVzKGRhdGEpKQogICAgICAgIHNlbGYud2ZpbGUud3JpdGUoYiJcclxuIikKICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKCiAgICBkZWYgX3NlbmRfanNvbihzZWxmLCBjb2RlLCBvYmosIGNsb3NlPUZhbHNlLCBleHRyYV9oZWFkZXJzPSgpKToKICAgICAgICAiIiJKU09OIOOCkiAxIOeZuuOBp+i/lOOBme+8iOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiOODu+OCqOODqeODvOW/nOetlOOBruWFsemAmuWHpueQhu+8ieOAgiIiIgogICAgICAgIGJvZHkgPSBvYmogaWYgaXNpbnN0YW5jZShvYmosIGJ5dGVzKSBlbHNlIGpzb24uZHVtcHMob2JqKS5lbmNvZGUoKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGNvZGUpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ2FjaGUtQ29udHJvbCIsICJuby1zdG9yZSIpCiAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gZXh0cmFfaGVhZGVyczoKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb25uZWN0aW9uIiwgImNsb3NlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJvZHkpCiAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQoKICAgIGRlZiBfYXV0aG9yaXplZChzZWxmKToKICAgICAgICAjIEFQSV9QQVNTIOOBjOepuiA9IOiqjeiovOOBquOBl++8iOWFrOmWi++8ieOAglVJIOOCkumWi+OBhOOBn+OBoOOBkeOBp+S9v+OBiOOCi+OCiOOBhuOBq+OBmeOCi+OAggogICAgICAgIGlmIG5vdCBFWFBFQ1RFRF9BVVRIOgogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGlmIHNlbGYucGF0aC5zcGxpdCgiPyIpWzBdIGluIE9QRU5fUEFUSFM6CiAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgIyDluLjmlbDmmYLplpPmr5TovIPvvIjjgr/jgqTjg5/jg7PjgrDmlLvmkoPjga7nt6nlkozvvInjgIJzdHIg54mIIGNvbXBhcmVfZGlnZXN0IOOBrwogICAgICAgICMg6Z2eIEFTQ0lJIOOBpyBUeXBlRXJyb3Ig44Gr44Gq44KL44Gu44GnIGJ5dGVzIOOBp+avlOi8g+OBmeOCi+OAggogICAgICAgICMgT1BUSU9OUyDjga8gX3JlbGF5IOWGkumgreOBp+ODreODvOOCq+ODq+WNs+etlOa4iOOBv++8iOOBk+OBk+OBq+OBr+WxiuOBi+OBquOBhO+8ieOAggogICAgICAgIGdvdCA9IHNlbGYuaGVhZGVycy5nZXQoIkF1dGhvcml6YXRpb24iLCAiIikuZW5jb2RlKCJsYXRpbi0xIikKICAgICAgICByZXR1cm4gaG1hYy5jb21wYXJlX2RpZ2VzdChnb3QsIEVYUEVDVEVEX0FVVEguZW5jb2RlKCJhc2NpaSIpKQoKICAgIGRlZiBfZGVueShzZWxmKToKICAgICAgICAjIDQwMSDjgafmjqXntprjgpLplonjgZjjgovjgIJLZWVwLUFsaXZlIOOBruOBvuOBvuaui+OBmeOBqOOAgeasoeOBruODquOCr+OCqOOCueODiOOBrgogICAgICAgICMg44Oc44OH44Kj6Kqt44G/5Y+W44KK5L2N572u44GM44Ga44KM44GmIEhUVFAg44Oq44Kv44Ko44K544OI44K544Oi44Ks44Oq44Oz44Kw44GM6LW344GN44GG44KL44CCCiAgICAgICAgc2VsZi5fc2VuZF9qc29uKDQwMSwgeyJlcnJvciI6ICJ1bmF1dGhvcml6ZWQifSwgY2xvc2U9VHJ1ZSwKICAgICAgICAgICAgICAgICAgICAgICAgZXh0cmFfaGVhZGVycz1bKCJXV1ctQXV0aGVudGljYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICdCYXNpYyByZWFsbT0iT2xsYW1hIEFQSSInKV0pCgogICAgZGVmIF9zZXJ2ZV91aShzZWxmKToKICAgICAgICAiIiLjg4Hjg6Pjg4Pjg4ggVUkg44KS6L+U44GZ44CC6KqN6Ki85LiN6KaB77yI6Z2Z55qE44OV44Kh44Kk44Or44Gu44G/77yJ44CCCgogICAgICAgIFVJIOOBryAxIOODquOCr+OCqOOCueODiOOBp+e1guOCj+OCi+OBruOBpyAqKuaOpee2muOCkumWieOBmOOCiyoq44CC6KqN6Ki844OR44K544Go6YGV44GECiAgICAgICAg44GT44GT44Gn44Gv44Oc44OH44Kj44KS6Kqt44G+44Gq44GE44Gf44KB44CBa2VlcC1hbGl2ZSDjgpLntq3mjIHjgZnjgovjgagKICAgICAgICBDb250ZW50LUxlbmd0aCDku5jjgY0gUE9TVCDjgafmrKHjga7jg6rjgq/jgqjjgrnjg4jkvY3nva7jgYzjgZrjgozjgosKICAgICAgICDvvIjjg6rjgq/jgqjjgrnjg4jjgrnjg57jgrDjg6rjg7PjgrDvvInjgIIKICAgICAgICAiIiIKICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKSAgICAgICAgICAjIOaDs+WumuWkluOBruODnOODh+OCo+OBr+aNqOOBpuOCi++8iOiqreOBv+aui+OBl+mYsuatou+8iQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICBpZiBub3QgQ0hBVF9CWVRFUzoKICAgICAgICAgICAgYm9keSA9IChiImxsYW1hLWNoYXQuaHRtbCBub3QgZm91bmQuICIKICAgICAgICAgICAgICAgICAgICBiInN0YXJ0IHRoZSBwcm94eSB3aXRoIC0tY2hhdC1odG1sIDxwYXRoPi4iKQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoNTAzKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCAidGV4dC9wbGFpbjsgY2hhcnNldD11dGYtOCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShib2R5KQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjAwKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtVHlwZSIsICJ0ZXh0L2h0bWw7IGNoYXJzZXQ9dXRmLTgiKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihDSEFUX0JZVEVTKSkpCiAgICAgICAgIyBVSSDjga/mr47lm57lkIzjgZjlhoXlrrnjgILjg5fjg63jgq3jgrfmm7TmlrDmmYLjgavjg5bjg6njgqbjgrbjgYzlj6TjgYTniYjjgpLkvb/jgo/jgarjgYTjgojjgYYgbm8tc3RvcmXjgIIKICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDYWNoZS1Db250cm9sIiwgIm5vLXN0b3JlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKENIQVRfQllURVMpCgogICAgZGVmIF9zZXJ2ZV9xdW90YShzZWxmKToKICAgICAgICAiIiJLYWdnbGUg44GuIEdQVSDjgq/jgqnjg7zjgr/mrovph4/jgpLov5TjgZnjgIIKCiAgICAgICAgVUkg44Gu44OY44OD44OA44Gr44CM5q6L44KKIDA4OjU544CN44Go5Ye644GZ44Gf44KB44Gr5L2/44GG44CCCiAgICAgICAg6LOH5qC85oOF5aCx77yIa2FnZ2xlLmpzb24g44G+44Gf44Gv55Kw5aKD5aSJ5pWw77yJ44GM54Sh44GE44CB44G+44Gf44GvIEFQSSDjgYwKICAgICAgICDokL3jgaHjgabjgYTjgovloLTlkIjjga8gYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OBpiBVSSDlgbTjgafpnZ7ooajnpLrjgavjgZnjgovjgIIKICAgICAgICAiIiIKICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZX0KICAgICAgICBpZiBBUkdTLm5vX3F1b3RhOgogICAgICAgICAgICAjIEthZ2dsZSDku6XlpJbjga7jg63jg7zjgqvjg6vvvIh1enUg44Gq44Gp77yJ44Gn44Gv44Kv44Kp44O844K/44Gu5qaC5b+144GM54Sh44GE44CCCiAgICAgICAgICAgICMgYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OAgVVJIOWBtOOBr+aui+OCiuaZgumWk+OCkumdnuihqOekuuOBq+OBmeOCi+OAggogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7ImF2YWlsYWJsZSI6IEZhbHNlLCAicmVhc29uIjogIm5vX3F1b3RhIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHVzZXIsIGtleSA9IGtjX2thZ2dsZV9jcmVkZW50aWFscygpCiAgICAgICAgICAgIGlmIHVzZXIgYW5kIGtleToKICAgICAgICAgICAgICAgIHBheWxvYWQgPSBrY19xdW90YV9zdW1tYXJ5KAogICAgICAgICAgICAgICAgICAgIGtjX2ZldGNoX3F1b3RhKHVzZXIsIGtleSwgdGltZW91dD0xNSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZSwgImVycm9yIjogdHlwZShlKS5fX25hbWVfX30KCiAgICAgICAgIyDmrovph4/jga/liLvjgIXlpInjgo/jgovjgIJfc2VuZF9qc29uIOOBryBuby1zdG9yZSDjgpLku5jjgZHjgovjgIIKICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCBwYXlsb2FkKQoKICAgIGRlZiBfc2VydmVfc3RvcChzZWxmKToKICAgICAgICAiIiLjgZPjga7kuK3ntpnjgajkuIrmtYHjga7jgrvjg4Pjgrfjg6fjg7PjgpLmraLjgoHjgovjgIIKCiAgICAgICAg5ZG844Gz5Ye644GX5p2h5Lu277yIUE9TVCArIFgtS2MtU3RvcCArIOiqjeiovO+8ieOBryBfc3RvcF9hbGxvd2VkKCkg44Gn56K66KqN5riI44G/44CCCgogICAgICAgIOODluODqeOCpuOCtuOBruWBnOatouODnOOCv+ODs+OBi+OCieWPqeOBj+OAguS4ree2meOBjOiQveOBoeOCi+OBruOBp+W/nOetlOOCkui/lOOBl+OBpuOBi+OCiQogICAgICAgIOODl+ODreOCu+OCueOCkue1guS6huOBleOBm+OCi++8iOWFiOOBq+iQveOBqOOBmeOBqOODluODqeOCpuOCtuWBtOOBjOmAmuS/oeOCqOODqeODvOOBq+OBquOCi++8ieOAggoKICAgICAgICBLYWdnbGUg5LiK44Gn44GvIGtlZXBhbGl2ZSDjg6vjg7zjg5fjgoLmraLjgoHjgovjgZ/jgoHjgIEvdG1wL2tlZXBhbGl2ZS5jb25mIOOBqwogICAgICAgIFJVTj0wIOOCkuabuOOBjei+vOOCgOOAguOBk+OCjOOBq+OCiOOCiiBLYWdnbGUg44Kr44O844ON44Or44K744OD44K344On44Oz44GM57WC5LqG44GX44CBCiAgICAgICAgR1BVIOOCr+OCqeODvOOCv+OCkumWi+aUvuOBp+OBjeOCi+OAggoKICAgICAgICAtLXN0b3AtcGlkIOOBjOaMh+WumuOBleOCjOOBpuOBhOOCjOOBsOOAgeOBneOBriBQSUTvvIjjg6njg7Pjg4Hjg6Pjg7zvvInjgavjgoIKICAgICAgICBTSUdURVJNIOOCkumAgeOCi+OAgnV6dSDjga8gcnVuLXV6dS1zZXJ2ZXIuY29tbWFuZCDjgYzlrZDjg5fjg63jgrvjgrnjgpIKICAgICAgICDmirHjgYjjgabjgYTjgovjga7jgafjgIHopqrjgpLokL3jgajjgZvjgbAgdHJhcCBjbGVhbnVwIOOBp+WFqOmDqOeJh+S7mOOBj+OAggogICAgICAgICIiIgogICAgICAgICMgS2FnZ2xlIOeJiCBrZWVwYWxpdmUg44Or44O844OX44Gr44KC5YGc5q2i44KS5Lyd44GI44KLCiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oIi90bXAva2VlcGFsaXZlLmNvbmYiLCAidyIpIGFzIGY6CiAgICAgICAgICAgICAgICBmLndyaXRlKCJSVU49MFxuIikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgICAgIHNlbGYuX3NlbmRfanNvbigyMDAsIHsic3RvcHBlZCI6IFRydWUsICJzY29wZSI6ICJyZWxheSJ9KQoKICAgICAgICBkZWYgX2RpZSgpOgogICAgICAgICAgICB0aW1lLnNsZWVwKDAuNSkKICAgICAgICAgICAgIyDlhYjjgavjg6njg7Pjg4Hjg6Pjg7zjgpLokL3jgajjgZnvvIjlrZDjg5fjg63jgrvjgrnjgaggVUkg44KC5LiA57eS44Gr5raI44GI44KL77yJCiAgICAgICAgICAgIGlmIEFSR1Muc3RvcF9waWQ6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgb3Mua2lsbChBUkdTLnN0b3BfcGlkLCBzaWduYWwuU0lHVEVSTSkKICAgICAgICAgICAgICAgIGV4Y2VwdCAoUHJvY2Vzc0xvb2t1cEVycm9yLCBQZXJtaXNzaW9uRXJyb3IpOgogICAgICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgICAgIHRpbWUuc2xlZXAoMC41KQogICAgICAgICAgICBvcy5fZXhpdCgwKQoKICAgICAgICB0aHJlYWRpbmcuVGhyZWFkKHRhcmdldD1fZGllLCBkYWVtb249VHJ1ZSkuc3RhcnQoKQoKICAgIGRlZiBfcmVsYXkoc2VsZik6CiAgICAgICAgIyBDT1JTIOODl+ODquODleODqeOCpOODiOOBr+iqjeiovOS4jeimgeODu+ODreODvOOCq+ODq+WNs+etlOOAguS4iua1geOBuOa1geOBleOBquOBhAogICAgICAgICMg77yI57Sg6YCa44GX44GZ44KL44Go5pyq6KqN6Ki844Oq44Kv44Ko44K544OI44GM5LiK5rWB44Gr5bGK44GN44CB44Oc44OH44Kj5b6F44Gh44GnCiAgICAgICAgIyAg44OP44Oz44OJ44Op44K544Os44OD44OJ44KC5Y2g44KB44KJ44KM44KL77yJ44CCCiAgICAgICAgaWYgc2VsZi5jb21tYW5kID09ICJPUFRJT05TIjoKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDIwNCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgIyAvaGVhbHRoeiDjga/jg63jg7zjgqvjg6vjgafljbPnrZTjgZnjgovvvIjkuIrmtYHjga7mnInnhKHjgavplqLkv4LjgarjgY/nlJ/lrZjnorroqo3jgafjgY3jgovvvIkKICAgICAgICBpZiBzZWxmLnBhdGguc3BsaXQoIj8iKVswXSA9PSAiL2hlYWx0aHoiOgogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7InN0YXR1cyI6ICJvayJ9KQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgcGF0aF9vbmx5ID0gc2VsZi5wYXRoLnNwbGl0KCI/IilbMF0KICAgICAgICBpZiBwYXRoX29ubHkgaW4gVUlfUEFUSFM6CiAgICAgICAgICAgIHNlbGYuX3NlcnZlX3VpKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgICMgLS0tLSDnrqHnkIbjgqjjg7Pjg4njg53jgqTjg7Pjg4jvvIjjg63jg7zjgqvjg6vlsILnlKjvvIktLS0tCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvcXVvdGEiOgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9xdW90YSgpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGlmIHBhdGhfb25seSA9PSAiL3N0b3AiOgogICAgICAgICAgICAjIOiQveOBqOOBm+OCi+OBruOBryBVSSDjga7lgZzmraLjg5zjgr/jg7PvvIhQT1NUICsgWC1LYy1TdG9w77yJ44Gg44GR44CCCiAgICAgICAgICAgICMg5Lul5YmN44Gv6KqN6Ki844KI44KK5YmN44Gr5Yem55CG44GX44Gm44GE44Gf44Gf44KB44CBQVBJX1BBU1Mg44KS6Kit5a6a44GX44Gm44GE44Gm44KCCiAgICAgICAgICAgICMg6Kqw44Gn44KCIEdFVCAvc3RvcCDjgacgS2FnZ2xlIOOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCieOCjOOBn+OAggogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIlBPU1QiIG9yIHNlbGYuaGVhZGVycy5nZXQoU1RPUF9IRUFERVIpICE9ICIxIjoKICAgICAgICAgICAgICAgIHNlbGYuX3NlbmRfanNvbig0MDUsIHsiZXJyb3IiOiAiUE9TVCB3aXRoICVzOiAxIHJlcXVpcmVkIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICUgU1RPUF9IRUFERVJ9LCBjbG9zZT1UcnVlKQogICAgICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgICAgIGlmIG5vdCBzZWxmLl9hdXRob3JpemVkKCk6CiAgICAgICAgICAgICAgICBzZWxmLl9kZW55KCkKICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9zdG9wKCkKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvZmF2aWNvbi5pY28iOgogICAgICAgICAgICAjIOODluODqeOCpuOCtuOBjOWLneaJi+OBq+WPluOCiuOBq+adpeOCi+OAgjQwNCDjgpLov5TjgZnjgajjg63jgrDjgYzln4vjgb7jgovjgIIKICAgICAgICAgICAgc2VsZi5jbG9zZV9jb25uZWN0aW9uID0gVHJ1ZQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjA0KQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsICIwIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICBpZiBub3Qgc2VsZi5fYXV0aG9yaXplZCgpOgogICAgICAgICAgICAjIDQwMSDjgpLov5TjgZnliY3jgavjg5zjg4fjgqPjgpLpo5/jgaPjgabjgYrjgY/jgIJQT1NUIOOBruODnOODh+OCo+OCkuiqreOBvuOBmuOBqwogICAgICAgICAgICAjIOaOpee2muOCkuWIh+OCi+OBqOOCr+ODqeOCpOOCouODs+ODiOWBtOOBjCBSU1Qg44KS5Y+X44GR44Gm44CMNDAxIOOCiOOCiuWJjeOBqwogICAgICAgICAgICAjIOaOpee2muOBjOWIh+OCjOOBn+OAjeOBq+imi+OBiOOCi++8iGN1cmwg44GvIFJTVCDjgpLjgqjjg6njg7zooajnpLrvvInjgIIKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5fcmVhZF9ib2R5KCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgc2VsZi5fZGVueSgpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGJvZHkgPSBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICMg5LiN5q2j44GqIENvbnRlbnQtTGVuZ3RoIC8g5aOK44KM44GfIGNodW5rZWQg44Gv5LiK5rWB44Gr5rWB44GV44Gq44GE44CCCiAgICAgICAgICAgICMg5pyq5o2V5o2J44Gg44Go44K544K/44OD44Kv44OA44Oz44OX44GuNTAw44Gr44Gq44KL77yI6KqN6Ki85oiQ5Yqf5YG044Gg44GR6Z2e5a++56ew44Gr44Gq44KJ44Gq44GE44KI44GG77yJCiAgICAgICAgICAgIHNlbGYuY2xvc2VfY29ubmVjdGlvbiA9IFRydWUKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDQwMCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbm5lY3Rpb24iLCAiY2xvc2UiKQogICAgICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlmIF9pc19nZW5lcmF0aW9uKHBhdGhfb25seSk6CiAgICAgICAgICAgIF9yZXNldF9rZWVwYWxpdmUoKSAgIyDnlJ/miJDjg6rjgq/jgqjjgrnjg4jvvJ3jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgYLjgoog4oaSIDYw5YiG44Oq44K744OD44OICiAgICAgICAgIyDjgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjga7lvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC44Gu44G/44CC5pei5a6a44Gv57Sg6YCa44GX77yJCiAgICAgICAgYm9keSA9IGVuZm9yY2Vfc2FtcGxpbmcoYm9keSwgcGF0aF9vbmx5KQogICAgICAgIHVybCA9IEFSR1MudXBzdHJlYW0ucnN0cmlwKCIvIikgKyBzZWxmLnBhdGgKICAgICAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBtZXRob2Q9c2VsZi5jb21tYW5kKQogICAgICAgIGZvciBrZXksIHZhbHVlIGluIHNlbGYuaGVhZGVycy5pdGVtcygpOgogICAgICAgICAgICAjIGNvbnRlbnQtbGVuZ3RoIOOBryBib2R5IOWGjeani+evieW+jOOBruWun+mVt+OBp+mAgeOCiuebtOOBmeOBn+OCgeOBk+OBk+OBp+aNqOOBpuOCiwogICAgICAgICAgICAjIO+8iOOCr+ODqeOCpOOCouODs+ODiOWApOOBruOBvuOBvui7oumAgeOBmeOCi+OBqCBjaHVua2VkICsgQ0wg5L2155So44Gn44OH44K344Oz44Kv44GZ44KL77yJCiAgICAgICAgICAgIGlmIGtleS5sb3dlcigpIGluIEhPUF9CWV9IT1AgfCB7Imhvc3QiLCAiYXV0aG9yaXphdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImNvbnRlbnQtbGVuZ3RoIn06CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZXEuYWRkX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGJvZHkgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJlcS5hZGRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsIHN0cihsZW4oYm9keSkpKQoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3AgPSB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJlcSwgdGltZW91dD1VUFNUUkVBTV9USU1FT1VUKQogICAgICAgIGV4Y2VwdCB1cmxsaWIuZXJyb3IuSFRUUEVycm9yIGFzIGV4YzoKICAgICAgICAgICAgIyDkuIrmtYHjgYwgNHh4LzV4eCDjgpLov5TjgZfjgZ/loLTlkIjjgoLmnKzmlofjgpLjgZ3jga7jgb7jgb7ov5TjgZkKICAgICAgICAgICAgcGF5bG9hZCA9IGV4Yy5yZWFkKCkKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGV4Yy5jb2RlKQogICAgICAgICAgICBjdHlwZSA9IGV4Yy5oZWFkZXJzLmdldCgiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCBjdHlwZSkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCBzdHIobGVuKHBheWxvYWQpKSkKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIHNlbGYuY29tbWFuZCAhPSAiSEVBRCI6CiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKHBheWxvYWQpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246ICAjIOaOpee2muWkseaVl+OBquOBqQogICAgICAgICAgICAjIOWGhemDqOS+i+WkluOBruips+e0sOOBr+OCr+ODqeOCpOOCouODs+ODiOOBq+a4oeOBleOBquOBhO+8iOaDheWgsemcsuWHuumYsuatou+8ieOAggogICAgICAgICAgICAjIOWOn+WboOOBr+OCteODvOODkOWBtCBzdGRlcnIg44Gr44Gg44GR5pu444GP44CCCiAgICAgICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSB1cHN0cmVhbSBlcnJvcjogJXNcbiIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAlIChzeXMuZXhjX2luZm8oKVsxXSwpKQogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oNTAyLCB7ImVycm9yIjogInVwc3RyZWFtIHVucmVhY2hhYmxlIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB3aXRoIHJlc3A6CiAgICAgICAgICAgIHNlbGYuc2VuZF9yZXNwb25zZShyZXNwLnN0YXR1cykKICAgICAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gcmVzcC5oZWFkZXJzLml0ZW1zKCk6CiAgICAgICAgICAgICAgICBpZiBrZXkubG93ZXIoKSBpbiBIT1BfQllfSE9QIG9yIGtleS5sb3dlcigpID09ICJjb250ZW50LWxlbmd0aCI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoa2V5LCB2YWx1ZSkKICAgICAgICAgICAgIyBIRUFEIOOBr+acrOaWh+OBquOBl+OAgmNodW5rZWQg44KS5a6j6KiA44GZ44KL44Go44Kv44Op44Kk44Ki44Oz44OI44GM57WC56uv44OB44Oj44Oz44Kv44KSCiAgICAgICAgICAgICMg5b6F44Gj44Gm44OP44Oz44Kw44GZ44KL77yIQ29udGVudC1MZW5ndGgg44KC5rih44Gb44Gq44GE44Gu44Gn5a6j6KiA44GX44Gq44GE77yJ44CCCiAgICAgICAgICAgICMgMjA0LzMwNCDjga/mnKzmlofjgpLmjIHjgabjgarjgYTjgIJjaHVua2VkIOOCkuWuo+iogOOBmeOCi+OBqOWOs+agvOOBqgogICAgICAgICAgICAjIOOCr+ODqeOCpOOCouODs+ODiOOBjCBwcm90b2NvbCBlcnJvciDjgavjgZnjgovjgIIKICAgICAgICAgICAgbm9fYm9keSA9IHJlc3Auc3RhdHVzIGluICgyMDQsIDMwNCkgb3Igc2VsZi5jb21tYW5kID09ICJIRUFEIgogICAgICAgICAgICBpZiBub3Qgbm9fYm9keToKICAgICAgICAgICAgICAgICMg6ZW344GV5pyq55+l44Gu44K544OI44Oq44O844Og44Go44GX44Gm5Lit57aZ44GZ44KLCiAgICAgICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJUcmFuc2Zlci1FbmNvZGluZyIsICJjaHVua2VkIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIG5vX2JvZHk6CiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgIyByZXNwLnJlYWQoTikg44GvIEhUVFBSZXNwb25zZS5fcmVhZGludG9fY2h1bmtlZCDjga7mgKfos6rjgacKICAgICAgICAgICAgICAgICMgKipOIOODkOOCpOODiOaPg+OBhuOBvuOBp+iqreOBv+e2muOBkeOCiyoq44Gf44KB44CB44K544OI44Oq44O844Og44GM5YWo6Z2i44Gr5rqc44G+44KLCiAgICAgICAgICAgICAgICAjIO+8iE9sbGFtYSDjga4gTkRKU09OIOOBjOWFqOODiOODvOOCr+ODs+WIhuiyr+OBvuOBo+OBpuOBi+OCieS4gOaWieOBq+WxiuOBj++8ieOAggogICAgICAgICAgICAgICAgIyByZWFkbGluZShDSFVOSykg44Gv5pS56KGM44GLIENIVU5LIOODkOOCpOODiOOBp+i/lOOCi+OBruOBp+OAgemAkOasoeaAp+OCkgogICAgICAgICAgICAgICAgIyDkv53jgaPjgZ/jgb7jgb4gMSDjg5DjgqTjg4jjgZrjgaTjga4gcmVhZCDjg6vjg7zjg5fjgojjgoogQ1BVIOOCkumjn+OCj+OBquOBhOOAggogICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICBsaW5lID0gcmVzcC5yZWFkbGluZShDSFVOSykKICAgICAgICAgICAgICAgICAgICBpZiBub3QgbGluZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgICAgICBzZWxmLl9lbWl0KGxpbmUpCiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiMFxyXG5cclxuIikKICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICBleGNlcHQgKEJyb2tlblBpcGVFcnJvciwgQ29ubmVjdGlvblJlc2V0RXJyb3IpOgogICAgICAgICAgICAgICAgIyDjgq/jg6njgqTjgqLjg7Pjg4jjgYzliIfjgaPjgZ/vvIhDdHJsLUMg44Gq44Gp77yJ44CC5q2j5bi457O744CCCiAgICAgICAgICAgICAgICBwYXNzCgogICAgZG9fR0VUID0gX3JlbGF5CiAgICBkb19QT1NUID0gX3JlbGF5CiAgICBkb19QVVQgPSBfcmVsYXkKICAgIGRvX1BBVENIID0gX3JlbGF5CiAgICBkb19ERUxFVEUgPSBfcmVsYXkKICAgIGRvX0hFQUQgPSBfcmVsYXkKICAgIGRvX09QVElPTlMgPSBfcmVsYXkKCgpjbGFzcyBTZXJ2ZXIoc29ja2V0c2VydmVyLlRocmVhZGluZ01peEluLCBodHRwLnNlcnZlci5IVFRQU2VydmVyKToKICAgIGRhZW1vbl90aHJlYWRzID0gVHJ1ZQogICAgYWxsb3dfcmV1c2VfYWRkcmVzcyA9IFRydWUKCgpkZWYgbWFpbigpOgogICAgZ2xvYmFsIEFSR1MsIEVYUEVDVEVEX0FVVEgKCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXBvcnQiLCB0eXBlPWludCwgZGVmYXVsdD0xMTQzNSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0taG9zdCIsIGRlZmF1bHQ9IjEyNy4wLjAuMSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXVwc3RyZWFtIiwgZGVmYXVsdD0iaHR0cDovLzEyNy4wLjAuMToxMTQzNCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoYXQtaHRtbCIsIGRlZmF1bHQ9b3MuZW52aXJvbi5nZXQoIkNIQVRfSFRNTCIsICIiKSwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0i44OB44Oj44OD44OIIFVJIOOBriBIVE1M77yILyDjgafphY3kv6HjgZnjgovvvIkiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zdG9wLXBpZCIsIHR5cGU9aW50LCBkZWZhdWx0PTAsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9Ii9zdG9wIOaZguOBqyBTSUdURVJNIOOCkumAgeOCiyBQSUTjgIIiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInV6dSDjga7jgojjgYbjgavjg6njg7Pjg4Hjg6Pjg7zjgYzlrZDjg5fjg63jgrvjgrnjgpLmirHjgYjjgovloLTlkIjjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIuODqeODs+ODgeODo+ODvOOBriBQSUQg44KS5rih44GZ44Go5b6M54mH5LuY44GR44GU44Go5q2i44G+44KLIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm8tcXVvdGEiLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSIvcXVvdGEg44KS54Sh5Yq55YyW44GZ44KL77yIS2FnZ2xlIOS7peWkluOBruODreODvOOCq+ODq+eUqO+8iSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVuZm9yY2Utc2FtcGxpbmciLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJjaGF0L2NvbXBsZXRpb25zIOOBriB0ZW1wZXJhdHVyZTw9MCDjgpIgJS4xZiDjgavjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInJlcGV0aXRpb25fcGVuYWx0eSDjgpIgJS4yZiDku6XkuIrjgavnn6/mraPjgZnjgosiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJSAoVEVNUF9GTE9PUiwgUlBfRkxPT1IpKQogICAgQVJHUyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKCiAgICB1c2VyID0gb3MuZW52aXJvbi5nZXQoIkFQSV9VU0VSIikgb3IgIm9sbGFtYSIKICAgIHBhc3N3b3JkID0gb3MuZW52aXJvbi5nZXQoIkFQSV9QQVNTIikgb3IgIiIKICAgIGlmIHBhc3N3b3JkOgogICAgICAgIHRva2VuID0gYmFzZTY0LmI2NGVuY29kZShmInt1c2VyfTp7cGFzc3dvcmR9Ii5lbmNvZGUoKSkuZGVjb2RlKCkKICAgICAgICBFWFBFQ1RFRF9BVVRIID0gIkJhc2ljICIgKyB0b2tlbgogICAgZWxzZToKICAgICAgICAjIOaXouWumuOBr+iqjeiovOOBquOBl+OAglVSTCDjgpLnn6XjgaPjgabjgYTjgozjgbDoqrDjgafjgoLkvb/jgYjjgovngrnjgaDjgZHms6jmhI/jgIIKICAgICAgICBwcmludCgiYXV0aDogT0ZGIChBUElfUEFTUyDmnKroqK3lrpopIiwgZmx1c2g9VHJ1ZSkKCiAgICBnbG9iYWwgQ0hBVF9CWVRFUwogICAgaWYgQVJHUy5jaGF0X2h0bWw6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oQVJHUy5jaGF0X2h0bWwsICJyYiIpIGFzIGY6CiAgICAgICAgICAgICAgICBDSEFUX0JZVEVTID0gZi5yZWFkKCkKICAgICAgICBleGNlcHQgT1NFcnJvciBhcyBleGM6CiAgICAgICAgICAgICMgVUkg54Sh44GX44Gn44KCIEFQSSDjga/li5XjgY/jgILokL3jgajjgZXjgZrorablkYrjgaDjgZHlh7rjgZnjgIIKICAgICAgICAgICAgc3lzLnN0ZGVyci53cml0ZSgiY2hhdCB1aSBub3QgbG9hZGVkOiAlc1xuIiAlIChleGMsKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiY2hhdCB1aTogJXMgKCVkIEIpIiAlIChBUkdTLmNoYXRfaHRtbCwgbGVuKENIQVRfQllURVMpKSwKICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKCiAgICBodHRwZCA9IFNlcnZlcigoQVJHUy5ob3N0LCBBUkdTLnBvcnQpLCBIYW5kbGVyKQogICAgcHJpbnQoZiJhdXRoIHByb3h5OiB7QVJHUy5ob3N0fTp7QVJHUy5wb3J0fSAtPiB7QVJHUy51cHN0cmVhbX0gIgogICAgICAgICAgZiIodXNlcj17dXNlcn0pIiwgZmx1c2g9VHJ1ZSkKICAgIHRyeToKICAgICAgICBodHRwZC5zZXJ2ZV9mb3JldmVyKCkKICAgIGV4Y2VwdCBLZXlib2FyZEludGVycnVwdDoKICAgICAgICBwYXNzCiAgICBmaW5hbGx5OgogICAgICAgIGh0dHBkLnNlcnZlcl9jbG9zZSgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo="
CHAT_SRC_B64 = "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImphIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0idXRmLTgiIC8+CiAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIiAvPgogIDx0aXRsZT7jg4Hjg6Pjg4Pjg4hVSTwvdGl0bGU+CiAgPHN0eWxlPgogICAgOnJvb3QgeyAtLWJnOiMwYjBjMTA7IC0tcGFuZWw6IzExMTMxYTsgLS1pbms6I2U5ZWVmMjsgLS1tdXRlZDojOWZiMGJmOyAtLWFjYzojNmFhMWZmOyAtLWxpbmU6IzFlMjIzMDsgfQogICAgKntib3gtc2l6aW5nOmJvcmRlci1ib3h9CiAgICBodG1sLGJvZHl7aGVpZ2h0OjEwMCV9CiAgICBib2R5e21hcmdpbjowOyBmb250OjE0cHgvMS40NSBzeXN0ZW0tdWksIC1hcHBsZS1zeXN0ZW0sIFNlZ29lIFVJLCBSb2JvdG8sIFVidW50dSwgQ2FudGFyZWxsLCBOb3RvIFNhbnMsIEhlbHZldGljYSwgQXJpYWw7IGJhY2tncm91bmQ6dmFyKC0tYmcpOyBjb2xvcjp2YXIoLS1pbmspfQogICAgLndyYXB7ZGlzcGxheTpncmlkOyBncmlkLXRlbXBsYXRlLXJvd3M6YXV0byAxZnIgYXV0bzsgaGVpZ2h0OjEwMCV9CiAgICBoZWFkZXIsIGZvb3RlcntiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTt9CiAgICBoZWFkZXJ7ZGlzcGxheTpmbGV4OyBnYXA6Ljc1cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXA7IHBhZGRpbmc6LjZyZW0gLjhyZW07IHBvc2l0aW9uOnN0aWNreTsgdG9wOjA7IHotaW5kZXg6MTA7IGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWxpbmUpfQogICAgaGVhZGVyIC50aXRsZXtmb250LXdlaWdodDo2MDA7IGxldHRlci1zcGFjaW5nOi4ycHh9CgogICAgaGVhZGVyIC5icmFuZHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4xNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucHJvbW97Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLWFjYyk7IHRleHQtZGVjb3JhdGlvbjpub25lOyBvcGFjaXR5Oi45fQogICAgaGVhZGVyIC5wcm9tbzpob3ZlcntvcGFjaXR5OjE7IHRleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmV9CgogICAgLyogSW5mbyBkaWFsb2cgKi8KICAgIGRpYWxvZy5pbmZvLWRpYWxvZ3sKICAgICAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYm9yZGVyLXJhZGl1czoxNHB4OwogICAgICBwYWRkaW5nOjA7CiAgICAgIGJhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6bWluKDU2MHB4LCBjYWxjKDEwMHZ3IC0gMjRweCkpOwogICAgICB3aWR0aDogNTYwcHg7CiAgICB9CiAgICBkaWFsb2cuaW5mby1kaWFsb2c6OmJhY2tkcm9we2JhY2tncm91bmQ6cmdiYSgwLDAsMCwuNTUpfQogICAgLmluZm8tY2FyZHttYXJnaW46MDsgcGFkZGluZzouOXJlbSAxcmVtOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi43NXJlbX0KICAgIC5pbmZvLWhlYWR7ZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGp1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOyBnYXA6Ljc1cmVtOyBib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZy1ib3R0b206LjZyZW19CiAgICAuaW5mby10aXRsZXtmb250LXdlaWdodDo2MDB9CiAgICAuaW5mby1ib2R5IGF7Y29sb3I6dmFyKC0tYWNjKTsgdGV4dC1kZWNvcmF0aW9uOm5vbmV9CiAgICAuaW5mby1ib2R5IGE6aG92ZXJ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZX0KICAgIC5pbmZvLWFjdGlvbnN7ZGlzcGxheTpmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7IGdhcDouNXJlbX0KICAgIGhlYWRlciAuc3BhY2Vye2ZsZXg6MX0KICAgIGhlYWRlciBpbnB1dHtiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjQ1cmVtIC41cmVtOyBib3JkZXItcmFkaXVzOi41cmVtOyBvdXRsaW5lOm5vbmU7IHdpZHRoOjIyY2g7IGhlaWdodDozMnB4fQogICAgaGVhZGVyIGlucHV0OmZvY3Vze2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIHNlbGVjdHsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgICBwYWRkaW5nOi40NXJlbSAuNXJlbTsKICAgICAgYm9yZGVyLXJhZGl1czouNXJlbTsKICAgICAgb3V0bGluZTpub25lOwogICAgICB3aWR0aDoyMmNoOwogICAgICBoZWlnaHQ6MzJweDsKICAgICAgYXBwZWFyYW5jZTpub25lOwogICAgfQogICAgaGVhZGVyIHNlbGVjdDpmb2N1c3tib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIGhlYWRlciAuc21hbGx7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKX0KICAgIGhlYWRlciBsYWJlbC5zbWFsbHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4yNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucm93e2Rpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXB9CgogICAgLyogU2Nyb2xsLVZpZXdwb3J0IChzbW9vdGggbnVyIGbDvHIgVXNlci1TY3JvbGwsIHByb2dyYW1tYXRpc2NoIGVyendpbmdlbiB3aXIgJ2F1dG8nKSAqLwogICAgbWFpbiNzY3JvbGxlcntvdmVyZmxvdzphdXRvOyBwYWRkaW5nOjFyZW0gMXJlbSAxLjVyZW07IHNjcm9sbC1iZWhhdmlvcjpzbW9vdGh9CiAgICAubG9ne21heC13aWR0aDo5MDBweDsgbWFyZ2luOjFyZW0gYXV0byAwOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOjFyZW19CiAgICAubXNne2Rpc3BsYXk6ZmxleDsgZ2FwOi43NXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIC5hdmF0YXJ7d2lkdGg6MjhweDsgaGVpZ2h0OjI4cHg7IGRpc3BsYXk6Z3JpZDsgcGxhY2UtaXRlbXM6Y2VudGVyOyBib3JkZXItcmFkaXVzOjhweDsgZmxleDowIDAgMjhweH0KICAgIC51c2VyIC5hdmF0YXJ7YmFja2dyb3VuZDojMjAzMTRhfQogICAgLmJvdCAuYXZhdGFye2JhY2tncm91bmQ6IzFiMmQyMH0KICAgIC5zeXMgLmF2YXRhcntiYWNrZ3JvdW5kOiMyYTIyM2J9CiAgICAKICAgIC8qIFdyYXBwZXIgZsO8ciBkZW4gTmFjaHJpY2h0ZW5pbmhhbHQgKi8KICAgIC5tZXNzYWdlLWNvbnRlbnQtd3JhcHBlciB7CiAgICAgICAgZGlzcGxheTpmbGV4OwogICAgICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICAgICAgICBnYXA6IDAuNXJlbTsgLyogQWJzdGFuZCB6d2lzY2hlbiBEZW5rcHJvemVzcyB1bmQgQW50d29ydCAqLwogICAgICAgIHdpZHRoOiAxMDAlOwogICAgfQoKICAgIC5idWJibGV7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6Ljc1cmVtIC45cmVtOyBib3JkZXItcmFkaXVzOjEycHg7IG1heC13aWR0aDoxMDAlOyBvdmVyZmxvdy13cmFwOmFueXdoZXJlfQogICAgLm1ldGF7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKTsgbWFyZ2luLWJvdHRvbTouMjVyZW19CgogICAgZm9vdGVye2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi42cmVtIC44cmVtfQogICAgLmJhcnttYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW19CiAgICB0ZXh0YXJlYXt3aWR0aDoxMDAlOyByZXNpemU6dmVydGljYWw7IG1pbi1oZWlnaHQ6NjBweDsgbWF4LWhlaWdodDo0MHZoOyBiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjdyZW0gLjc1cmVtOyBib3JkZXItcmFkaXVzOi43NXJlbTsgb3V0bGluZTpub25lfQogICAgdGV4dGFyZWE6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CiAgICBidXR0b257YmFja2dyb3VuZDojMTAxODI3OyBjb2xvcjp2YXIoLS1pbmspOyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi41NXJlbSAuODVyZW07IGJvcmRlci1yYWRpdXM6LjZyZW07IGN1cnNvcjpwb2ludGVyfQogICAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIGJ1dHRvbntoZWlnaHQ6MzJweDsgZGlzcGxheTppbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6Y2VudGVyfQogICAgLnNlY29uZGFyeXtiYWNrZ3JvdW5kOiMwYzBmMTV9CiAgICAubXV0ZWR7Y29sb3I6dmFyKC0tbXV0ZWQpfQogICAgLmhpZGRlbntkaXNwbGF5Om5vbmV9CiAgICAuaGludHtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXJnaW4tbGVmdDouMjVyZW19CiAgICAuZHJvcHtkaXNwbGF5Om5vbmU7IGJvcmRlcjoycHggZGFzaGVkICMyYTMzNDY7IGJvcmRlci1yYWRpdXM6MTJweDsgcGFkZGluZzoxMnB4OyB0ZXh0LWFsaWduOmNlbnRlcjsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjoxcmVtIGF1dG8gLjc1cmVtO30KICAgIC5kcm9wLmRyYWd7YmFja2dyb3VuZDojMGUxMzIwOyBjb2xvcjp2YXIoLS1pbmspfQogICAgY29kZSwgcHJlLCBrYmR7Zm9udC1mYW1pbHk6IHVpLW1vbm9zcGFjZSwgU0ZNb25vLVJlZ3VsYXIsIE1lbmxvLCBDb25zb2xhcywgbW9ub3NwYWNlfQoKICAgIC8qIE1hcmtkb3duIHJlbmRlcmluZyAqLwogICAgLm1ke2xpbmUtaGVpZ2h0OjEuNn0KICAgIC5tZCBoMSwubWQgaDIsLm1kIGgze2xpbmUtaGVpZ2h0OjEuMjU7IG1hcmdpbjouMnJlbSAwIC42cmVtfQogICAgLm1kIGgxe2ZvbnQtc2l6ZToxLjM1cmVtfQogICAgLm1kIGgye2ZvbnQtc2l6ZToxLjJyZW19CiAgICAubWQgaDN7Zm9udC1zaXplOjEuMDVyZW19CiAgICAubWQgcHttYXJnaW46LjVyZW0gMH0KICAgIC5tZCB1bCwubWQgb2x7bWFyZ2luOi41cmVtIDAgLjVyZW0gMS4xcmVtfQogICAgLm1kIGxpe21hcmdpbjouMnJlbSAwfQogICAgLm1kIGJsb2NrcXVvdGV7bWFyZ2luOi42cmVtIDA7IHBhZGRpbmc6LjI1cmVtIC44cmVtOyBib3JkZXItbGVmdDozcHggc29saWQgIzMwNDQ1ZjsgYmFja2dyb3VuZDojMGMxMTFiOyBjb2xvcjojZDZlMmYwOyBib3JkZXItcmFkaXVzOjZweH0KICAgIC5tZCBocntib3JkZXI6bm9uZTsgYm9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7IG1hcmdpbjouOHJlbSAwfQogICAgLm1kIGNvZGV7YmFja2dyb3VuZDojMGMwZjE1OyBwYWRkaW5nOi4xMnJlbSAuMzJyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6LjM1cmVtfQogICAgLm1kIHByZXtiYWNrZ3JvdW5kOiMwYzBmMTU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgcGFkZGluZzouOHJlbTsgb3ZlcmZsb3c6YXV0bzsgcG9zaXRpb246cmVsYXRpdmV9CiAgICAubWQgcHJlPmNvZGV7ZGlzcGxheTpibG9jazsgd2hpdGUtc3BhY2U6cHJlfQogICAgLm1kIHRhYmxle2JvcmRlci1jb2xsYXBzZTpjb2xsYXBzZTsgbWFyZ2luOi44cmVtIDA7IHdpZHRoOjEwMCU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAubWQgdGgsLm1kIHRke3BhZGRpbmc6LjVyZW0gLjdyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHRleHQtYWxpZ246bGVmdH0KICAgIC5tZCB0aHtiYWNrZ3JvdW5kOiMwZjE3MjU7IGZvbnQtd2VpZ2h0OjYwMH0KICAgIC5jb2Rld3JhcHtwb3NpdGlvbjpyZWxhdGl2ZX0KICAgIC5jb3B5YnRue3Bvc2l0aW9uOmFic29sdXRlOyB0b3A6LjQ1cmVtOyByaWdodDouNDVyZW07IGZvbnQtc2l6ZToxMnB4OyBwYWRkaW5nOi4yNXJlbSAuNDVyZW07IGJvcmRlci1yYWRpdXM6LjRyZW07IGJhY2tncm91bmQ6IzBmMTcyNTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgY3Vyc29yOnBvaW50ZXJ9CiAgICAuY29weWJ0bjpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIC50YXNrLWRvbmV7b3BhY2l0eTouOX0KCiAgICAvKiDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4ggc3R5bGluZyAqLwogICAgLnN5c2lucHV0e3dpZHRoOjEwMCU7IHJlc2l6ZTp2ZXJ0aWNhbDsgbWluLWhlaWdodDo2MHB4OyBtYXgtaGVpZ2h0OjMwdmg7IGJhY2tncm91bmQ6IzBjMGYxNTsgY29sb3I6dmFyKC0taW5rKTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZzouN3JlbSAuNzVyZW07IGJvcmRlci1yYWRpdXM6Ljc1cmVtOyBvdXRsaW5lOm5vbmV9CiAgICAuc3lzaW5wdXQ6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CgogICAgLyogSlNPTiBFZGl0b3IgKi8KICAgIGRldGFpbHMucmF3e2Rpc3BsYXk6bm9uZTsgbWF4LXdpZHRoOjkwMHB4OyBtYXJnaW46MXJlbSBhdXRvIDA7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgYmFja2dyb3VuZDojMGMwZjE1fQogICAgZGV0YWlscy5yYXc+c3VtbWFyeXtjdXJzb3I6cG9pbnRlcjsgcGFkZGluZzouNnJlbSAuOHJlbTsgbGlzdC1zdHlsZTpub25lOyB1c2VyLXNlbGVjdDpub25lfQogICAgZGV0YWlscy5yYXc+c3VtbWFyeTo6bWFya2Vye2Rpc3BsYXk6bm9uZX0KICAgIGRldGFpbHMucmF3W29wZW5dPnN1bW1hcnl7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAucmF3IC5pbm5lcntwYWRkaW5nOi44cmVtfQogICAgLnJhdyB0ZXh0YXJlYXtmb250LWZhbWlseTogdWktbW9ub3NwYWNlLCBTRk1vbm8tUmVndWxhciwgTWVubG8sIENvbnNvbGFzLCBtb25vc3BhY2U7IG1pbi1oZWlnaHQ6MjIwcHg7IHdoaXRlLXNwYWNlOnByZTsgdGFiLXNpemU6Mn0KICAgIC5yYXcgLnJvd3tkaXNwbGF5OmZsZXg7IGdhcDouNXJlbTsgYWxpZ24taXRlbXM6Y2VudGVyOyBtYXJnaW4tdG9wOi41cmVtOyBmbGV4LXdyYXA6d3JhcH0KICAgIC5iYWRnZXtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpfQoKICAgIC8qIFNjcm9sbC10by1ib3R0b20gYnV0dG9uICovCiAgICAudG8tYm90dG9te3Bvc2l0aW9uOmZpeGVkOyByaWdodDoxNnB4OyBib3R0b206MTZweDsgcGFkZGluZzouNXJlbSAuN3JlbTsgYm9yZGVyLXJhZGl1czouNnJlbTsgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBjb2xvcjp2YXIoLS1pbmspOyBkaXNwbGF5Om5vbmU7IG9wYWNpdHk6LjkyfQogICAgLnRvLWJvdHRvbS5zaG93e2Rpc3BsYXk6YmxvY2t9CiAgICAKICAgIC8qIFNUWUxFUyBGw5xSIFRISU5LSU5HLUJPWCAqLwogICAgLnRoaW5rLWJ1YmJsZSB7CiAgICAgICAgd2lkdGg6IDEwMCU7CiAgICB9CiAgICAudGhpbmstYnViYmxlIGRldGFpbHMgewogICAgICAgIGJhY2tncm91bmQ6ICMxYTE2MjU7CiAgICAgICAgYm9yZGVyOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBib3JkZXItcmFkaXVzOiAxMHB4OwogICAgICAgIGZvbnQtc2l6ZTogMTNweDsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeSB7CiAgICAgICAgY3Vyc29yOiBwb2ludGVyOwogICAgICAgIHBhZGRpbmc6IC41cmVtIC44cmVtOwogICAgICAgIGZvbnQtd2VpZ2h0OiA1MDA7CiAgICAgICAgY29sb3I6IHZhcigtLW11dGVkKTsKICAgICAgICB1c2VyLXNlbGVjdDogbm9uZTsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeTpob3ZlciB7CiAgICAgICAgY29sb3I6IHZhcigtLWluayk7CiAgICB9CiAgICAudGhpbmstYnViYmxlIC50aGluay1jb250ZW50IHsKICAgICAgICBwYWRkaW5nOiAuMXJlbSAuOHJlbSAuN3JlbTsKICAgICAgICBib3JkZXItdG9wOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBjb2xvcjogI2MwYjhkMTsKICAgIH0KCS8qIOKAlCBQb2xpc2hpbmc6IFNhbXBsaW5nLUlucHV0cyBpbSBFaW5zdGVsbHVuZ3MtUGFuZWwg4oCUICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCB7ICAgICAgICAgICAgICAgLyogTGFiZWwgKyBGZWxkIHZlcnRpa2FsLCBrb21wYWt0ICovCiAgZGlzcGxheTogZmxleDsKICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwogIGdhcDogLjI1cmVtOwogIGFsaWduLWl0ZW1zOiBmbGV4LXN0YXJ0Owp9CgovKiBFaW5oZWl0bGljaGUsIHNjaG1hbGUgTnVtYmVyLUlucHV0cyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIHdpZHRoOiAxMGNoOyAgICAgICAgICAgICAgICAgICAgICAgIC8qIGZpeGUgQnJlaXRlIC0+IGdsZWljaGUgS2FudGUgKi8KICBoZWlnaHQ6IDMycHg7CiAgcGFkZGluZzogLjM1cmVtIC41cmVtOwogIGJhY2tncm91bmQ6ICMwYzBmMTU7CiAgY29sb3I6IHZhcigtLWluayk7CiAgYm9yZGVyOiAxcHggc29saWQgdmFyKC0tbGluZSk7CiAgYm9yZGVyLXJhZGl1czogLjVyZW07CiAgdGV4dC1hbGlnbjogcmlnaHQ7CiAgZm9udC12YXJpYW50LW51bWVyaWM6IHRhYnVsYXItbnVtczsgLyogc2F1YmVyZSwgYXVzZ2VyaWNodGV0ZSBaaWZmZXJuICovCiAgb3V0bGluZTogbm9uZTsKfQoKLyogRm9rdXMtRmVlZGJhY2sgcGFzc2VuZCB6dSBkZW4gcmVzdGxpY2hlbiBDb250cm9scyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Zm9jdXMgewogIGJvcmRlci1jb2xvcjogdmFyKC0tYWNjKTsKICBib3gtc2hhZG93OiAwIDAgMCAycHggcmdiYSgxMDYsMTYxLDI1NSwuMTUpOwp9CgovKiBEZXplbnRlIFBsYWNlaG9sZGVyLUZhcmJlICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6cGxhY2Vob2xkZXIgewogIGNvbG9yOiB2YXIoLS1tdXRlZCk7CiAgb3BhY2l0eTogLjk7Cn0KCi8qIFNwaW5uZXIgYmVpIE51bWJlci1JbnB1dHMgYXVzYmxlbmRlbiAobnVyIGltIFNldHRpbmdzLVBhbmVsKSAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Oi13ZWJraXQtb3V0ZXItc3Bpbi1idXR0b24sCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6LXdlYmtpdC1pbm5lci1zcGluLWJ1dHRvbiB7CiAgLXdlYmtpdC1hcHBlYXJhbmNlOiBub25lOwogIG1hcmdpbjogMDsKfQojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIC1tb3otYXBwZWFyYW5jZTogdGV4dGZpZWxkOwp9CgovKiBLbGVpbmVzIFR1bmluZyBmw7xyIGRpZSBHcnVwcGU6IGRpY2h0ZXIgenVzYW1tZW4sIGJyaWNodCBzYXViZXIgdW0gKi8KI3NldHRpbmdzUGFuZWwgPiAuYmFyID4gZGl2Omxhc3QtY2hpbGQsCiNzZXR0aW5nc1BhbmVsIC5iYXIgPiBkaXY6bGFzdC1jaGlsZCB7CiAgZ2FwOiAuNXJlbSAuNzVyZW07ICAgICAgICAgICAgICAgICAgLyogWmVpbGVuL1NwYWx0ZW4tR2FwICovCiAgZmxleC13cmFwOiB3cmFwOwp9CgogIAogICAgLyogLS0tLS0tLS0tLSBSZXNwb25zaXZlIC0tLS0tLS0tLS0gKi8KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA5ODBweCl7CiAgICAgIGhlYWRlcntnYXA6LjZyZW19CiAgICAgIGhlYWRlciAuc3BhY2Vye2Rpc3BsYXk6bm9uZX0KICAgICAgaGVhZGVyIGxhYmVsLnNtYWxse21pbi13aWR0aDoyMDBweH0KICAgICAgaGVhZGVyIC5yb3d7d2lkdGg6MTAwJX0KICAgICAgaGVhZGVyICN0b2tlblJhdGV7d2lkdGg6YXV0bzsgdGV4dC1hbGlnbjpsZWZ0ICFpbXBvcnRhbnR9CiAgICB9CiAgICBAbWVkaWEgKG1heC13aWR0aDogNzIwcHgpewogICAgICBoZWFkZXIgbGFiZWwuc21hbGx7ZmxleDoxIDEgMjIwcHh9CiAgICAgIGhlYWRlciBpbnB1dCwgaGVhZGVyIHNlbGVjdHt3aWR0aDoxMDAlfQogICAgICBtYWluI3Njcm9sbGVye3BhZGRpbmc6Ljg1cmVtIC43NXJlbSAxLjI1cmVtfQogICAgICAuYmFye2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgICAgIGZvb3RlciAucm93e2ZsZXgtd3JhcDp3cmFwfQogICAgICB0ZXh0YXJlYXttaW4taGVpZ2h0OjkwcHh9CiAgICAgIC8qIFNldHRpbmdzIHBhbmVsIHN0YWNrICovCiAgICAgICNzZXR0aW5nc1BhbmVse2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgIWltcG9ydGFudH0KICAgICAgI3NldHRpbmdzUGFuZWwgPiBkaXZ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAhaW1wb3J0YW50fQogICAgICAjc2V0dGluZ3NQYW5lbCA+IGRpdiA+IGRpdjpsYXN0LWNoaWxke21pbi13aWR0aDowICFpbXBvcnRhbnQ7IGFsaWduLXNlbGY6c3RhcnQgIWltcG9ydGFudH0KICAgIH0KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA0ODBweCl7CiAgICAgIGhlYWRlciBsYWJlbC5zbWFsbHtmbGV4OjEgMSAxMDAlfQogICAgICBoZWFkZXIgLnJvd3tnYXA6LjRyZW19CiAgICAgIGhlYWRlciBidXR0b24sIGZvb3RlciBidXR0b257cGFkZGluZzouNXJlbSAuNzVyZW19CiAgICB9CgogIAogICAgLyogQXR0YWNobWVudHMgKi8KICAgIC5jaGlwewogICAgICBkaXNwbGF5OmlubGluZS1mbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNDVyZW07CiAgICAgIHBhZGRpbmc6LjM1cmVtIC41NXJlbTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OyBib3JkZXItcmFkaXVzOjk5OXB4OyBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6MTAwJTsKICAgIH0KICAgIC5jaGlwIHNtYWxse2NvbG9yOnZhcigtLW11dGVkKX0KICAgIC5jaGlwIGJ1dHRvbnsKICAgICAgcGFkZGluZzouMTVyZW0gLjM1cmVtOyBib3JkZXItcmFkaXVzOjhweDsKICAgICAgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgfQogICAgLmNoaXAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQoKICAgIC8qIE1hcmtkb3duIGltYWdlcyAqLwogICAgLm1kIGltZ3ttYXgtd2lkdGg6MTAwJTsgaGVpZ2h0OmF1dG87IGJvcmRlci1yYWRpdXM6MTBweDsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KCiAgPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KPGRpdiBjbGFzcz0id3JhcCI+CiAgPGhlYWRlcj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9InRpdGxlIj7wn5KsIOODgeODo+ODg+ODiFVJPC9kaXY+CiAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzcGFjZXIiPjwvZGl2PgogICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+PHNwYW4+44Oi44OH44OrPC9zcGFuPgogICAgICA8c2VsZWN0IGlkPSJtb2RlbCI+CiAgICAgICAgPG9wdGlvbiB2YWx1ZT0iIiBzZWxlY3RlZCBkaXNhYmxlZD7igJQg44Oi44OH44Or44KS6Kqt44G/6L6844G/IOKAlDwvb3B0aW9uPgogICAgICA8L3NlbGVjdD4KICAgIDwvbGFiZWw+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICA8YnV0dG9uIGlkPSJhdXRoQnRuIiBjbGFzcz0ic2Vjb25kYXJ5IiB0aXRsZT0iQVBJIOiqjeiovO+8iEJhc2lj77yJIj7wn5SRPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImluZm9CdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSLmg4XloLEiPuaDheWgsTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICAgIDxkaXYgaWQ9InF1b3RhTGVmdCIgY2xhc3M9InNtYWxsIiBzdHlsZT0id2lkdGg6MTFjaDsgdGV4dC1hbGlnbjpyaWdodDsiIHRpdGxlPSJLYWdnbGUgR1BVIOaui+OCiuaZgumWkyI+PC9kaXY+CiAgICAgICAgPGRpdiBpZD0idG9rZW5SYXRlIiBjbGFzcz0ic21hbGwiIHN0eWxlPSJ3aWR0aDoxMmNoOyB0ZXh0LWFsaWduOnJpZ2h0OyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJvdyIgaWQ9InN0b3BSb3ciIHN0eWxlPSJkaXNwbGF5Om5vbmU7Ij4KICAgICAgPGJ1dHRvbiBpZD0ic3RvcFNlcnZlciIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IuOCteODvOODkOODvOOCkuWBnOatoiI+4pagIOWBnOatojwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9oZWFkZXI+CgogIDxtYWluIGlkPSJzY3JvbGxlciI+CiAgICA8ZGl2IGlkPSJkcm9wIiBjbGFzcz0iZHJvcCI+44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GZ44KL44Gr44Gv44OV44Kh44Kk44Or44KS44OJ44Ot44OD44OXIChKU09OIGluIHRoZSB1c3VhbCA8Y29kZT5be3JvbGUsIGNvbnRlbnR9XTwvY29kZT4gZm9ybWF0KS48YnI+PHNwYW4gY2xhc3M9ImhpbnQiPuOBvuOBn+OBryA8Yj7jgqTjg7Pjg53jg7zjg4g8L2I+IGFib3ZlLjwvc3Bhbj48L2Rpdj4KCiAgICA8ZGV0YWlscyBpZD0icmF3Qm94IiBjbGFzcz0icmF3Ij4KICAgICAgPHN1bW1hcnk+4pyOIEpTT07jgajjgZfjgabjg4Hjg6Pjg4Pjg4jjgpLnt6jpm4YgPHNwYW4gY2xhc3M9ImJhZGdlIj4o5YiH5pu/KTwvc3Bhbj48L3N1bW1hcnk+CiAgICAgIDxkaXYgY2xhc3M9ImlubmVyIj4KICAgICAgICA8dGV4dGFyZWEgaWQ9Impzb25FZGl0b3IiIHNwZWxsY2hlY2s9ImZhbHNlIiBwbGFjZWhvbGRlcj0nWyB7ICJyb2xlIjogInN5c3RlbSIsICJjb250ZW50IjogIi4uLiIgfSwgLi4uIF0nPjwvdGV4dGFyZWE+CiAgICAgICAgPGRpdiBjbGFzcz0icm93Ij4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25BcHBseSI+6YGp55SoICjnva7mj5spPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uQXBwZW5kIiBjbGFzcz0ic2Vjb25kYXJ5Ij7ov73oqJg8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25SZWZyZXNoIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgYvjgonmm7TmlrA8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25QcmV0dHkiIGNsYXNzPSJzZWNvbmRhcnkiPuaVtOW9ojwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBpZD0ianNvbk1pbmlmeSIgY2xhc3M9InNlY29uZGFyeSI+5Zyn57iuPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uVmFsaWRhdGUiIGNsYXNzPSJzZWNvbmRhcnkiPuaknOiovDwvYnV0dG9uPgogICAgICAgICAgPHNwYW4gaWQ9Impzb25TdGF0dXMiIGNsYXNzPSJiYWRnZSI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGV0YWlscz4KCiAgICA8ZGl2IGlkPSJsb2ciIGNsYXNzPSJsb2ciPjwvZGl2PgogIDwvbWFpbj4KCiAgPGZvb3Rlcj4KICAgIDxkaXYgY2xhc3M9ImJhciI+CiAgICAgIDx0ZXh0YXJlYSBpZD0iaW5wdXQiIHBsYWNlaG9sZGVyPSLjg6Hjg4Pjgrvjg7zjgrjjgpLlhaXlipvigKYgKOmAgeS/oTogRW50ZXIsIOaUueihjDogU2hpZnQrRW50ZXIpIj48L3RleHRhcmVhPgogICAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJkaXNwbGF5OmZsZXg7IGdhcDouNnJlbTsgYWxpZ24taXRlbXM6Y2VudGVyIj4KICAgICAgICA8YnV0dG9uIGlkPSJhdHRhY2hCdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSJBdHRhY2ggZmlsZXMiPvCfk448L2J1dHRvbj4KICAgICAgICA8aW5wdXQgaWQ9ImF0dGFjaElucHV0IiB0eXBlPSJmaWxlIiBjbGFzcz0iaGlkZGVuIiBtdWx0aXBsZSAvPgogICAgICAgIDxidXR0b24gaWQ9InNlbmQiPumAgeS/oTwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9InN0b3AiIGNsYXNzPSJzZWNvbmRhcnkiIGRpc2FibGVkPuWBnOatojwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9ImNsZWFyIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgq/jg6rjgqI8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGlkPSJzZXR0aW5nc1RvZ2dsZSIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IlRvZ2dsZSBzZXR0aW5ncyI+4pqZ77iPPC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0JhciIgY2xhc3M9ImJhciBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOi42cmVtOyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0xpc3QiIHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBmbGV4LXdyYXA6d3JhcCI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgaWQ9InNldHRpbmdzUGFuZWwiIGNsYXNzPSJiYXIgaGlkZGVuIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW07IGdyaWQtY29sdW1uOjEvLTE7IHdpZHRoOjEwMCUiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuOCt+OCueODhuODoOODl+ODreODs+ODl+ODiDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFyZWEgaWQ9InN5c3RlbSIgY2xhc3M9InN5c2lucHV0IiByb3dzPSIyIiBwbGFjZWhvbGRlcj0i5Lu75oSPIOKAlCBlLmcuIFlvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC4iPjwvdGV4dGFyZWE+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0ibWluLXdpZHRoOjI2MHB4OyBhbGlnbi1zZWxmOmVuZDsgZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNnJlbTsgZmxleC13cmFwOndyYXAiPgogICAgICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+VGVtcGVyYXR1cmUgPHNwYW4gaWQ9InRlbXBWYWwiIGNsYXNzPSJoaW50Ij4wLjc8L3NwYW4+PC9sYWJlbD4KICAgICAgICAgIDxpbnB1dCBpZD0idGVtcCIgdHlwZT0icmFuZ2UiIG1pbj0iMCIgbWF4PSIyIiBzdGVwPSIwLjEiIHZhbHVlPSIwLjciIHN0eWxlPSJ3aWR0aDoxNjBweCI+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3BfcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj5taW5fcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJtaW5fcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3Bfazxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfayIgdHlwZT0ibnVtYmVyIiBzdGVwPSIxIiBtaW49IjAiIHBsYWNlaG9sZGVyPSIoZGVmYXVsdCkiIHN0eWxlPSJ3aWR0aDo5Y2g7IHRleHQtYWxpZ246cmlnaHQiPgogICAgICAgICAgPC9sYWJlbD4KICAgICAgICAgIDxsYWJlbCBjbGFzcz0ic21hbGwiPnRoaW5raW5nPGJyPgogICAgICAgICAgICA8c2VsZWN0IGlkPSJ0aGlua2luZyIgc3R5bGU9IndpZHRoOjExY2giPgogICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+KGRlZmF1bHQpPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib24iPm9uPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib2ZmIj5vZmY8L29wdGlvbj4KICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICA8L2xhYmVsPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZm9vdGVyPgo8L2Rpdj4KCjxidXR0b24gaWQ9InRvQm90dG9tIiBjbGFzcz0idG8tYm90dG9tIiB0aXRsZT0i5LiA55Wq5LiL44G4Ij7ilrwg5LiA55Wq5LiLPC9idXR0b24+Cgo8ZGlhbG9nIGlkPSJpbmZvRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCfkqwg44OB44Oj44OD44OIVUk8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ic2Vjb25kYXJ5IiB2YWx1ZT0iY2xvc2UiIGFyaWEtbGFiZWw9IkNsb3NlIj7inJU8L2J1dHRvbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1ib2R5Ij4KICAgICAgPGRpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPuODkOODvOOCuOODp+ODszo8L3NwYW4+IDxjb2RlIGlkPSJpbmZvVmVyc2lvbiI+MC4wLjA8L2NvZGU+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZm9udC1zaXplOjEycHgiPuOCueODhuODvOODiOODrOOCuVVJIOKAlCBubyBzZXJ2ZXItc2lkZSBjaGF0IHN0b3JhZ2UuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImluZm8tYWN0aW9ucyI+CiAgICAgIDxidXR0b24gdmFsdWU9ImNsb3NlIj7plonjgZjjgos8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZm9ybT4KPC9kaWFsb2c+Cgo8ZGlhbG9nIGlkPSJhdXRoRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCflJEgQVBJIOiqjeiovDwvZGl2PgogICAgICA8YnV0dG9uIGNsYXNzPSJzZWNvbmRhcnkiIHZhbHVlPSJjbG9zZSIgYXJpYS1sYWJlbD0iQ2xvc2UiPuKclTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJpbmZvLWJvZHkiPgogICAgICA8ZGl2IGNsYXNzPSJtdXRlZCBzbWFsbCIgc3R5bGU9Im1hcmdpbi1ib3R0b206LjZyZW0iPuOCteODvOODkOODvOOBjCBCYXNpYyDoqo3oqLzjgpLopoHmsYLjgZnjgovloLTlkIjjgavlhaXlipvjgZfjgb7jgZnvvIhzZXNzaW9uU3RvcmFnZSDjgavkv53lrZjjg7vjgr/jg5bjgpLplonjgZjjgovjgajmtojjgYjjgb7jgZnvvInjgII8L2Rpdj4KICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbi1ib3R0b206LjM1cmVtIj7jg6bjg7zjgrbjg7w8L2xhYmVsPgogICAgICA8aW5wdXQgaWQ9ImF1dGhVc2VyIiB0eXBlPSJ0ZXh0IiBhdXRvY29tcGxldGU9InVzZXJuYW1lIiB2YWx1ZT0iZXhsMyIgc3R5bGU9IndpZHRoOjEwMCU7IG1hcmdpbi1ib3R0b206LjZyZW0iPgogICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuODkeOCueODr+ODvOODiTwvbGFiZWw+CiAgICAgIDxpbnB1dCBpZD0iYXV0aFBhc3MiIHR5cGU9InBhc3N3b3JkIiBhdXRvY29tcGxldGU9ImN1cnJlbnQtcGFzc3dvcmQiIHN0eWxlPSJ3aWR0aDoxMDAlOyBtYXJnaW4tYm90dG9tOi42cmVtIj4KICAgICAgPGRpdiBpZD0iYXV0aFN0YXR1cyIgY2xhc3M9InNtYWxsIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1hY3Rpb25zIj4KICAgICAgPGJ1dHRvbiBpZD0iYXV0aFNhdmUiIHR5cGU9ImJ1dHRvbiI+5L+d5a2YPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImF1dGhDbGVhciIgdHlwZT0iYnV0dG9uIiBjbGFzcz0ic2Vjb25kYXJ5Ij7op6PpmaQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiB2YWx1ZT0iY2xvc2UiIGNsYXNzPSJzZWNvbmRhcnkiPumWieOBmOOCizwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9mb3JtPgo8L2RpYWxvZz4KCgo8c2NyaXB0PgooKCkgPT4gewogIC8vIEFQSSDjga7mjqXntprlhYjjgpLoh6rli5XliKTlrprjgZnjgovjgILlkIzjgZggSFRNTCDjgpLjg63jg7zjgqvjg6vjgafjgoIgS2FnZ2xlIOOBp+OCguS9v+OBhuOAggogIC8vCiAgLy8gMSkgP2FwaT08dXJsPiDjgYzmmI7npLrjgZXjgozjgabjgYTjgozjgbDjgZ3jgozvvIjmnIDlhKrlhYjvvIkKICAvLyAyKSA/cG9ydD08bj4g44GM44GC44KM44GwIGh0dHA6Ly8xMjcuMC4wLjE6PG4+L3Yx77yI44Ot44O844Kr44OrIHV6dSAvIGxsYW1hLXNlcnZlcu+8iQogIC8vIDMpIGh0dHAocykg44Gn6YWN5L+h44GV44KM44Gm44GE44KL44Gq44KJ5ZCM5LiA44Kq44Oq44K444Oz77yIS2FnZ2xlIOS4ree2me+8iQogIC8vIDQpIGZpbGU6Ly8g44Gn6ZaL44GE44Gf5aC05ZCI44Gv5pei5a6a44Od44O844OICiAgY29uc3QgcGFyYW1zID0gbmV3IFVSTFNlYXJjaFBhcmFtcyh3aW5kb3cubG9jYXRpb24uc2VhcmNoKTsKICBjb25zdCBwb3J0UGFyYW0gPSBwYXJhbXMuZ2V0KCdwb3J0Jyk7CiAgY29uc3QgYXBpUGFyYW0gPSBwYXJhbXMuZ2V0KCdhcGknKTsKCiAgZnVuY3Rpb24gcmVzb2x2ZUFwaUJhc2UoKSB7CiAgICBpZiAoYXBpUGFyYW0pIHJldHVybiBhcGlQYXJhbS5yZXBsYWNlKC9cLyQvLCAnJyk7CiAgICBpZiAocG9ydFBhcmFtKSByZXR1cm4gYGh0dHA6Ly8xMjcuMC4wLjE6JHtwb3J0UGFyYW19L3YxYDsKICAgIC8vIOS4ree2meOChCBuZ3JvayDntYznlLHjgafphY3kv6HjgZXjgozjgabjgYTjgovloLTlkIjjga/lkIzkuIDjgqrjg6rjgrjjg7PjgIIKICAgIC8vIOOBk+OBk+OCkiAxMjcuMC4wLjEg5Zu65a6a44Gr44GZ44KL44Go44CB44Oq44Oi44O844OI44GL44KJ6ZaL44GE44Gf44Go44GN44GrCiAgICAvLyDmiYvlhYPjga4gUEMg44KS6KaL44Gr6KGM44Gj44Gm44GX44G+44GG44CCCiAgICBpZiAobG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwOicgfHwgbG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwczonKSB7CiAgICAgIHJldHVybiBsb2NhdGlvbi5vcmlnaW4gKyAnL3YxJzsKICAgIH0KICAgIHJldHVybiAnaHR0cDovLzEyNy4wLjAuMTo5OTMxL3YxJzsKICB9CgogIGNvbnN0IEFQSV9CQVNFID0gcmVzb2x2ZUFwaUJhc2UoKTsKCiAgLy8gbmdyb2sg54Sh5paZ5p6g44Gv44OW44Op44Km44K244GL44KJ44GuIEFQSSDlj5blvpfjgavjgoLorablkYrjg5rjg7zjgrjjgpLmjJ/jgoDjgZPjgajjgYzjgYLjgovjgIIKICAvLyDjgZPjga7jg5jjg4Pjg4Djga/jg63jg7zjgqvjg6vjgrXjg7zjg5Djg7zjgafjga/nhKHoppbjgZXjgozjgovjga7jgafjgIHluLjjgavku5jjgZHjgabjgoLlrrPjga/jgarjgYTjgIIKICBjb25zdCBCQVNFX0hFQURFUlMgPSB7ICduZ3Jvay1za2lwLWJyb3dzZXItd2FybmluZyc6ICd0cnVlJyB9OwogIGNvbnN0IGVscyA9IHsKICAgIHNjcm9sbGVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Nyb2xsZXInKSwKICAgIGxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xvZycpLAogICAgZHJvcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Ryb3AnKSwKICAgIGlucHV0OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnaW5wdXQnKSwKICAgIHNlbmRCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzZW5kJyksCiAgICBzdG9wQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcCcpLAogICAgY2xlYXJCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbGVhcicpLAogICAgYXR0YWNoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXR0YWNoQnRuJyksCiAgICBhdHRhY2hJbnB1dDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaElucHV0JyksCiAgICBhdHRhY2htZW50c0JhcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaG1lbnRzQmFyJyksCiAgICBhdHRhY2htZW50c0xpc3Q6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdHRhY2htZW50c0xpc3QnKSwKICAgIG1vZGVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbW9kZWwnKSwKICAgIHN5c3RlbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N5c3RlbScpLAogICAgdGVtcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RlbXAnKSwKICAgIHRlbXBWYWw6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0ZW1wVmFsJyksCiAgICBpbXBvcnRCdG46IG51bGwsCiAgICBpbXBvcnRGaWxlOiBudWxsLAogICAgZXhwb3J0QnRuOiBudWxsLAogICAgbG9hZE1vZGVsc0J0bjogbnVsbCwKICAgIG1vZGVsU3RhdHVzOiBudWxsLAogICAgcmF3Qm94OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmF3Qm94JyksCiAgICBqc29uRWRpdG9yOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkVkaXRvcicpLAogICAganNvbkFwcGx5OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGx5JyksCiAgICBqc29uQXBwZW5kOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGVuZCcpLAogICAganNvblJlZnJlc2g6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uUmVmcmVzaCcpLAogICAganNvblByZXR0eTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25QcmV0dHknKSwKICAgIGpzb25NaW5pZnk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uTWluaWZ5JyksCiAgICBqc29uVmFsaWRhdGU6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uVmFsaWRhdGUnKSwKICAgIGpzb25TdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uU3RhdHVzJyksCiAgICB0b0JvdHRvbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RvQm90dG9tJyksCiAgICBzZXR0aW5nc1RvZ2dsZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NldHRpbmdzVG9nZ2xlJyksCiAgICBzZXR0aW5nc1BhbmVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2V0dGluZ3NQYW5lbCcpLAogICAgdG9rZW5SYXRlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9rZW5SYXRlJyksCiAgICBxdW90YUxlZnQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdxdW90YUxlZnQnKSwKICAgIHN0b3BSb3c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdG9wUm93JyksCiAgICBzdG9wU2VydmVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcFNlcnZlcicpLAogICAgaW5mb0J0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9CdG4nKSwKICAgIGluZm9EaWFsb2c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvRGlhbG9nJyksCiAgICBpbmZvVmVyc2lvbjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9WZXJzaW9uJyksCiAgICBpbmZvUHJvamVjdFVybDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9Qcm9qZWN0VXJsJyksCiAgICBpbmZvSW1wcmludDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9JbXByaW50JyksCiAgICBhdXRoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aEJ0bicpLAogICAgYXV0aERpYWxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhEaWFsb2cnKSwKICAgIGF1dGhVc2VyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFVzZXInKSwKICAgIGF1dGhQYXNzOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFBhc3MnKSwKICAgIGF1dGhTdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoU3RhdHVzJyksCiAgICBhdXRoU2F2ZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhTYXZlJyksCiAgICBhdXRoQ2xlYXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoQ2xlYXInKSwKICAgIC8vIE5FVToKICAgIHRvcF9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX3AnKSwKICAgIG1pbl9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWluX3AnKSwKICAgIHRvcF9rOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX2snKSwKICAgIHRoaW5raW5nOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndGhpbmtpbmcnKSwKICB9OwoKICBsZXQgbWVzc2FnZXMgPSBbXTsKICBsZXQgYXR0YWNobWVudHMgPSBbXTsgLy8ge2lkLG5hbWUsdHlwZSxzaXplLCBraW5kOid0ZXh0J3wnaW1hZ2UnfCdvdGhlcicsIHRleHQ/LCBkYXRhVXJsPywgdHJ1bmNhdGVkP30KICBsZXQgbmV4dEF0dGFjaElkID0gMTsKCiAgLyogLS0tLS0tLS0tLSBBUEkg6KqN6Ki877yIQmFzaWPvvIkgLS0tLS0tLS0tLSAqLwogIC8vIOS4ree2meOBjCBBUElfUEFTUyDjgpLoqK3lrprjgZfjgabjgYTjgovjgajjgY3jga8gL3YxLyog44KCIC9zdG9wIOOCgiA0MDEg44Gr44Gq44KL44CCCiAgLy8g6LOH5qC85oOF5aCx44GvIHNlc3Npb25TdG9yYWdlIOOBq+OBoOOBkee9ruOBj++8iGxvY2FsU3RvcmFnZSDjgavjga/mrovjgZXjgarjgYTvvInjgIIKICBjb25zdCBBVVRIX0tFWSA9ICdrY19iYXNpY19hdXRoJzsKICBjb25zdCBBVVRIX1VTRVJfS0VZID0gJ2tjX2Jhc2ljX3VzZXInOwoKICBmdW5jdGlvbiBiNjR1dGY4KHMpewogICAgLy8gUHl0aG9uIOWBtOOBryBVVEYtOCDjgacgYmFzZTY0IOWMluOBl+OBpuOBhOOCi+OBruOBp+OAgWJ0b2Eg55u05ZG844Gz77yITGF0aW4tMe+8ieOBp+OBrwogICAgLy8g6Z2eIEFTQ0lJIOOBruODkeOCueODr+ODvOODieOBp+S4gOiHtOOBl+OBquOBhOOAggogICAgY29uc3QgYnl0ZXMgPSBuZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUocyk7CiAgICBsZXQgYmluID0gJyc7CiAgICBmb3IgKGNvbnN0IGIgb2YgYnl0ZXMpIGJpbiArPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGIpOwogICAgcmV0dXJuIGJ0b2EoYmluKTsKICB9CiAgZnVuY3Rpb24gZ2V0QXV0aEhlYWRlcigpewogICAgY29uc3QgdiA9IHNlc3Npb25TdG9yYWdlLmdldEl0ZW0oQVVUSF9LRVkpOwogICAgcmV0dXJuIHYgPyB7ICdBdXRob3JpemF0aW9uJzogYEJhc2ljICR7dn1gIH0gOiB7fTsKICB9CiAgZnVuY3Rpb24gYXBpSGVhZGVycyhleHRyYSl7CiAgICByZXR1cm4geyAuLi5CQVNFX0hFQURFUlMsIC4uLmdldEF1dGhIZWFkZXIoKSwgLi4uKGV4dHJhIHx8IHt9KSB9OwogIH0KICBmdW5jdGlvbiBzZXRBdXRoQ3JlZGVudGlhbHModXNlciwgcGFzcyl7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfS0VZLCBiNjR1dGY4KGAke3VzZXJ9OiR7cGFzc31gKSk7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfVVNFUl9LRVksIHVzZXIpOwogIH0KICBmdW5jdGlvbiBjbGVhckF1dGhDcmVkZW50aWFscygpewogICAgc2Vzc2lvblN0b3JhZ2UucmVtb3ZlSXRlbShBVVRIX0tFWSk7CiAgICBzZXNzaW9uU3RvcmFnZS5yZW1vdmVJdGVtKEFVVEhfVVNFUl9LRVkpOwogIH0KICBmdW5jdGlvbiBvcGVuQXV0aERpYWxvZyhtc2cpewogICAgaWYgKCFlbHMuYXV0aERpYWxvZykgcmV0dXJuOwogICAgaWYgKGVscy5hdXRoVXNlcikgewogICAgICBlbHMuYXV0aFVzZXIudmFsdWUgPSBzZXNzaW9uU3RvcmFnZS5nZXRJdGVtKEFVVEhfVVNFUl9LRVkpIHx8ICdleGwzJzsKICAgIH0KICAgIGlmIChlbHMuYXV0aFBhc3MpIGVscy5hdXRoUGFzcy52YWx1ZSA9ICcnOwogICAgaWYgKGVscy5hdXRoU3RhdHVzKSBlbHMuYXV0aFN0YXR1cy50ZXh0Q29udGVudCA9IG1zZyB8fCAnJzsKICAgIGlmICh0eXBlb2YgZWxzLmF1dGhEaWFsb2cuc2hvd01vZGFsID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5zaG93TW9kYWwoKTsKICAgIGVsc2UgZWxzLmF1dGhEaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgc2V0VGltZW91dCgoKSA9PiBlbHMuYXV0aFBhc3M/LmZvY3VzKCksIDUwKTsKICB9CiAgZnVuY3Rpb24gY2xvc2VBdXRoRGlhbG9nKCl7CiAgICBpZiAoIWVscy5hdXRoRGlhbG9nKSByZXR1cm47CiAgICBpZiAodHlwZW9mIGVscy5hdXRoRGlhbG9nLmNsb3NlID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5jbG9zZSgpOwogICAgZWxzZSBlbHMuYXV0aERpYWxvZy5yZW1vdmVBdHRyaWJ1dGUoJ29wZW4nKTsKICB9CgogIGZ1bmN0aW9uIGZtdEJ5dGVzKG4pewogICAgaWYgKCFOdW1iZXIuaXNGaW5pdGUobikpIHJldHVybiAnJzsKICAgIGNvbnN0IHUgPSBbJ0InLCdLQicsJ01CJywnR0InXTsgbGV0IGk9MDsgbGV0IHg9bjsKICAgIHdoaWxlICh4Pj0xMDI0ICYmIGk8dS5sZW5ndGgtMSl7IHgvPTEwMjQ7IGkrKzsgfQogICAgcmV0dXJuIGAke3gudG9GaXhlZChpPzE6MCl9ICR7dVtpXX1gOwogIH0KCiAgZnVuY3Rpb24gcmVuZGVyQXR0YWNobWVudHMoKXsKICAgIGNvbnN0IGxpc3QgPSBlbHMuYXR0YWNobWVudHNMaXN0OwogICAgbGlzdC5pbm5lckhUTUwgPSAnJzsKICAgIGlmICghYXR0YWNobWVudHMubGVuZ3RoKXsKICAgICAgZWxzLmF0dGFjaG1lbnRzQmFyLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpOwogICAgICByZXR1cm47CiAgICB9CiAgICBlbHMuYXR0YWNobWVudHNCYXIuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7CgogICAgZm9yIChjb25zdCBhIG9mIGF0dGFjaG1lbnRzKXsKICAgICAgY29uc3QgY2hpcCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgICBjaGlwLmNsYXNzTmFtZSA9ICdjaGlwJzsKCiAgICAgIGNvbnN0IGljb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdzcGFuJyk7CiAgICAgIGljb24udGV4dENvbnRlbnQgPSAoYS5raW5kID09PSAnaW1hZ2UnKSA/ICfwn5a877iPJyA6IChhLmtpbmQgPT09ICd0ZXh0JykgPyAn8J+ThCcgOiAn8J+Tpic7CgogICAgICBjb25zdCBuYW1lID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogICAgICBuYW1lLnN0eWxlLm92ZXJmbG93ID0gJ2hpZGRlbic7CiAgICAgIG5hbWUuc3R5bGUudGV4dE92ZXJmbG93ID0gJ2VsbGlwc2lzJzsKICAgICAgbmFtZS5zdHlsZS53aGl0ZVNwYWNlID0gJ25vd3JhcCc7CiAgICAgIG5hbWUuc3R5bGUubWF4V2lkdGggPSAnNTVjaCc7CiAgICAgIG5hbWUudGV4dENvbnRlbnQgPSBhLm5hbWU7CgogICAgICBjb25zdCBzaXplID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc21hbGwnKTsKICAgICAgc2l6ZS50ZXh0Q29udGVudCA9IGZtdEJ5dGVzKGEuc2l6ZSk7CgogICAgICBjb25zdCByZW1vdmUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdidXR0b24nKTsKICAgICAgcmVtb3ZlLnRpdGxlID0gJ1JlbW92ZSc7CiAgICAgIHJlbW92ZS50ZXh0Q29udGVudCA9ICfinJUnOwogICAgICByZW1vdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGF0dGFjaG1lbnRzID0gYXR0YWNobWVudHMuZmlsdGVyKHggPT4geC5pZCAhPT0gYS5pZCk7CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfSk7CgogICAgICBjaGlwLmFwcGVuZChpY29uLCBuYW1lLCBzaXplLCByZW1vdmUpOwogICAgICBsaXN0LmFwcGVuZENoaWxkKGNoaXApOwogICAgfQogIH0KCiAgY29uc3QgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUgPSAyMDAwMDsKICBjb25zdCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgPSAxMjsKCiAgZnVuY3Rpb24gcmVhZEZpbGVBc1RleHQoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc1RleHQoZmlsZSk7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gcmVhZEZpbGVBc0RhdGFVUkwoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc0RhdGFVUkwoZmlsZSk7CiAgICB9KTsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIGFkZEF0dGFjaG1lbnRzRnJvbUZpbGVMaXN0KGZpbGVMaXN0KXsKICAgIGNvbnN0IGZpbGVzID0gQXJyYXkuZnJvbShmaWxlTGlzdCB8fCBbXSk7CiAgICBpZiAoIWZpbGVzLmxlbmd0aCkgcmV0dXJuOwoKICAgIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcy5zbGljZSgwLCBNYXRoLm1heCgwLCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgLSBhdHRhY2htZW50cy5sZW5ndGgpKSl7CiAgICAgIGNvbnN0IGlkID0gbmV4dEF0dGFjaElkKys7CiAgICAgIGNvbnN0IHR5cGUgPSBmaWxlLnR5cGUgfHwgJyc7CiAgICAgIGNvbnN0IG5hbWUgPSBmaWxlLm5hbWUgfHwgYGZpbGVfJHtpZH1gOwogICAgICBjb25zdCBzaXplID0gZmlsZS5zaXplIHx8IDA7CgogICAgICBpZiAodHlwZS5zdGFydHNXaXRoKCdpbWFnZS8nKSl7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOeUu+WDj+OBr+OCteODneODvOODiOOBleOCjOOBpuOBhOOBvuOBm+OCk++8iC0tbW1wcm9qIOOBquOBl++8ieOAguODhuOCreOCueODiOODleOCoeOCpOODq+OBruOBv+a3u+S7mOWPr+iDveOBp+OBmeOAgmApOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CgogICAgICBjb25zdCBsb29rc1RleHQgPSB0eXBlLnN0YXJ0c1dpdGgoJ3RleHQvJykKICAgICAgICB8fCAvXC4odHh0fG1kfGpzb258anNvbmx8Y3N2fHRzdnxsb2d8eWFtbHx5bWx8cHl8anN8dHN8aHRtbHxjc3N8eG1sfGluaXx0b21sKSQvaS50ZXN0KG5hbWUpOwoKICAgICAgaWYgKGxvb2tzVGV4dCl7CiAgICAgICAgdHJ5ewogICAgICAgICAgbGV0IHRleHQgPSBhd2FpdCByZWFkRmlsZUFzVGV4dChmaWxlKTsKICAgICAgICAgIGxldCB0cnVuY2F0ZWQgPSBmYWxzZTsKICAgICAgICAgIGlmICh0ZXh0Lmxlbmd0aCA+IE1BWF9URVhUX0NIQVJTX1BFUl9GSUxFKXsKICAgICAgICAgICAgdGV4dCA9IHRleHQuc2xpY2UoMCwgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUpOwogICAgICAgICAgICB0cnVuY2F0ZWQgPSB0cnVlOwogICAgICAgICAgfQogICAgICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOid0ZXh0JywgdGV4dCwgdHJ1bmNhdGVkIH0pOwogICAgICAgIH0gY2F0Y2ggewogICAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOODhuOCreOCueODiOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnzogJHtuYW1lfWApOwogICAgICAgIH0KICAgICAgICBjb250aW51ZTsKICAgICAgfQoKICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOidvdGhlcicgfSk7CiAgICB9CgogICAgaWYgKGF0dGFjaG1lbnRzLmxlbmd0aCA+PSBNQVhfVE9UQUxfQVRUQUNITUVOVFMpIHsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOS4iumZkOOBq+mBlOOBl+OBvuOBl+OBnyAoJHtNQVhfVE9UQUxfQVRUQUNITUVOVFN9IGZpbGVzKS5gKTsKICAgIH0KICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgfQoKICBsZXQgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICBjb25zdCBOTCA9ICdcbic7CgogIGNvbnN0IEFQUF9WRVJTSU9OID0gJzEuMDEnOwogIGNvbnN0IFBST0pFQ1RfVVJMID0gJ2h0dHBzOi8vd3d3LmxvY2FsbGlnaHRhaS5jb20vc2N1JzsKICBjb25zdCBJTVBSSU5UX1VSTCA9ICdodHRwczovL3d3dy5zcndhcmUubmV0L2ltcHJpbnQnOwogIAogIGxldCBzdHJlYW1TdGFydFRpbWUgPSAwOwogIGxldCBsYXN0VXBkYXRlVGltZSA9IDA7CiAgbGV0IGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgY29uc3QgVVBEQVRFX0lOVEVSVkFMID0gMjAwOyAvLyBtcwoKICAvKiAtLS0tLS0tLS0tIEF1dG8tU2Nyb2xsIChnZWZsaWNrdCkgLS0tLS0tLS0tLSAqLwogIGxldCBhdXRvc2Nyb2xsID0gdHJ1ZTsKICBsZXQgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICBjb25zdCBCT1RUT01fRVBTID0gODA7CgogIGZ1bmN0aW9uIGF0Qm90dG9tKCl7CiAgICBjb25zdCBzID0gZWxzLnNjcm9sbGVyOwogICAgcmV0dXJuIChzLnNjcm9sbEhlaWdodCAtIHMuc2Nyb2xsVG9wIC0gcy5jbGllbnRIZWlnaHQpIDw9IEJPVFRPTV9FUFM7CiAgfQogIGZ1bmN0aW9uIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKXsKICAgIGNvbnN0IHMgPSBlbHMuc2Nyb2xsZXI7CiAgICBjb25zdCBwcmV2ID0gcy5zdHlsZS5zY3JvbGxCZWhhdmlvcjsKICAgIHMuc3R5bGUuc2Nyb2xsQmVoYXZpb3IgPSAnYXV0byc7CiAgICBzLnNjcm9sbFRvcCA9IHMuc2Nyb2xsSGVpZ2h0OwogICAgcy5zdHlsZS5zY3JvbGxCZWhhdmlvciA9IHByZXYgfHwgJyc7CiAgfQogIGZ1bmN0aW9uIG1heWJlU2Nyb2xsKGZvcmNlPWZhbHNlKXsKICAgIGlmIChmb3JjZSB8fCBhdXRvc2Nyb2xsIHx8IGF0Qm90dG9tKCkpIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGNvbnN0IHNob3cgPSAhYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93Jywgc2hvdyk7CiAgICBpZiAoIXNob3cpIGF1dG9zY3JvbGwgPSB0cnVlOwogIH0KICBlbHMuc2Nyb2xsZXIuYWRkRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgKCkgPT4gewogICAgaWYgKGlzUmVuZGVyaW5nKSByZXR1cm47CiAgICBhdXRvc2Nyb2xsID0gYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93JywgIWF1dG9zY3JvbGwpOwogIH0pOwogIGVscy50b0JvdHRvbS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsgYXV0b3Njcm9sbCA9IHRydWU7IG1heWJlU2Nyb2xsKHRydWUpOyB9KTsKCiAgLyogLS0tLS0tLS0tLSBNYXJrZG93biAtLS0tLS0tLS0tICovCiAgY29uc3QgZXNjID0gcyA9PiBTdHJpbmcocykucmVwbGFjZSgvWyY8PiInXS9nLCBjID0+ICh7JyYnOicmYW1wOycsJzwnOicmbHQ7JywnPic6JyZndDsnLCciJzonJnF1b3Q7JywiJyI6JyYjMzk7J31bY10pKTsKICBmdW5jdGlvbiBpbmxpbmVNZChzKXsKICAgIC8vIFByb3RlY3QgaW5saW5lIGNvZGUgd2l0aCBwbGFjZWhvbGRlcnMgZmlyc3QKICAgIGNvbnN0IGNvZGVQbGFjZWhvbGRlcnMgPSBbXTsKICAgIHMgPSBzLnJlcGxhY2UoL2AoW15gXSspYC9nLCAoXyxwKSA9PiB7IGNvbnN0IGlkeCA9IGNvZGVQbGFjZWhvbGRlcnMubGVuZ3RoOyBjb2RlUGxhY2Vob2xkZXJzLnB1c2gocCk7IHJldHVybiBgXHgwMENPREUke2lkeH1ceDAwYDsgfSk7CiAgICAvLyBJbWFnZXMKICAgIC8vIOWklumDqOeUu+WDj+OBr+iHquWLleiqreOBv+i+vOOBv+OBl+OBquOBhO+8iOmWsuimp+OBrui/vei3oeOBq+S9v+OBiOOCi++8ieOAgmh0dHBzL2h0dHAg44Gu44G/44Oq44Oz44Kv44Go44GX44Gm5Ye644GZ44CCCiAgICBzID0gcy5yZXBsYWNlKC8hXFsoW15cXV0qKVxdXCgoaHR0cHM/OlteKVxzXSspXCkvZywgKF8sYWx0LHVybCk9Pic8YSBocmVmPSInK3VybCsnIiB0YXJnZXQ9Il9ibGFuayIgcmVsPSJub29wZW5lciBub3JlZmVycmVyIj5b55S75YOPOiAnKyhhbHR8fHVybCkrJ108L2E+Jyk7CiAgICAvLyBCb2xkCiAgICBzID0gcy5yZXBsYWNlKC9cKlwqKFteKl0rKVwqXCovZywgKF8scCk9Pic8c3Ryb25nPicrcCsnPC9zdHJvbmc+Jyk7CiAgICAvLyBJdGFsaWMgKG9ubHkgb3V0c2lkZSBjb2RlKQogICAgcyA9IHMucmVwbGFjZSgvXCooW14qXSspXCovZywgKF8scCk9Pic8ZW0+JytwKyc8L2VtPicpOwogICAgLy8gTGlua3MKICAgIHMgPSBzLnJlcGxhY2UoL1xbKFteXF1dKylcXVwoKGh0dHBzPzpbXlwpXHNdKylbXlwpXSpcKS9nLCAoXyx0eHQsdXJsKT0+JzxhIGhyZWY9IicrdXJsKyciIHRhcmdldD0iX2JsYW5rIiByZWw9Im5vb3BlbmVyIj4nK3R4dCsnPC9hPicpOwogICAgLy8gQXV0by1saW5rIGJhcmUgVVJMcyAoc2tpcCBhbHJlYWR5LWdlbmVyYXRlZCBIVE1MIHRhZ3MpCiAgICBzID0gcy5yZXBsYWNlKC8oPFtePl0qPil8KFxiaHR0cHM/OlwvXC9bXlxzKe+8ieOAjeOAj+OAguOAgV0rKS9nLCAobWF0Y2gsIHRhZywgdXJsKSA9PiB7CiAgICAgIGlmICh0YWcpIHJldHVybiB0YWc7CiAgICAgIHJldHVybiAnPGEgaHJlZj0iJyt1cmwrJyIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPicrdXJsKyc8L2E+JzsKICAgIH0pOwogICAgLy8gUmVzdG9yZSBjb2RlCiAgICBzID0gcy5yZXBsYWNlKC9ceDAwQ09ERShcZCspXHgwMC9nLCAoXyxpZHgpID0+ICc8Y29kZT4nK2NvZGVQbGFjZWhvbGRlcnNbcGFyc2VJbnQoaWR4KV0rJzwvY29kZT4nKTsKICAgIHJldHVybiBzOwogIH0KICBmdW5jdGlvbiBtZFRvSHRtbChtZCl7CiAgICBtZCA9IFN0cmluZyhtZCA/PyAnJyk7CiAgICBtZCA9IG1kLnJlcGxhY2UoL1xyL2csJycpOwogICAgY29uc3QgbGluZXMgPSBtZC5zcGxpdCgnXG4nKTsKICAgIGxldCBodG1sID0gJycsIGluQ29kZSA9IGZhbHNlLCBjb2RlTGFuZyA9ICcnLCBsaXN0VHlwZSA9IG51bGw7CiAgICBjb25zdCBjbG9zZUxpc3QgPSAoKSA9PiB7IGlmIChsaXN0VHlwZSkgeyBodG1sICs9IGA8LyR7bGlzdFR5cGV9PmA7IGxpc3RUeXBlPW51bGw7IH0gfTsKICAgIGZvciAobGV0IGk9MDtpPGxpbmVzLmxlbmd0aDtpKyspewogICAgICBsZXQgbGluZSA9IGxpbmVzW2ldOwogICAgICBpZiAoIWluQ29kZSAmJiAvXlxzKmBgYC8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgbSA9IGxpbmUubWF0Y2goL15ccypgYGBccyooW1x3Ky1dKikvKTsKICAgICAgICBjb2RlTGFuZyA9IG0gJiYgbVsxXSA/IG1bMV0gOiAnJzsKICAgICAgICBodG1sICs9IGA8ZGl2IGNsYXNzPSJjb2Rld3JhcCI+PHByZT48Y29kZSBkYXRhLWxhbmc9IiR7ZXNjKGNvZGVMYW5nKX0iPmA7CiAgICAgICAgaW5Db2RlID0gdHJ1ZTsgY29udGludWU7CiAgICAgIH0KICAgICAgaWYgKGluQ29kZSl7CiAgICAgICAgaWYgKC9eXHMqYGBgXHMqJC8udGVzdChsaW5lKSl7IGh0bWwgKz0gYDwvY29kZT48L3ByZT48YnV0dG9uIGNsYXNzPSJjb3B5YnRuIj5Db3B5PC9idXR0b24+PC9kaXY+YDsgaW5Db2RlPWZhbHNlOyBjb250aW51ZTsgfQogICAgICAgIGh0bWwgKz0gZXNjKGxpbmUpICsgJ1xuJzsgY29udGludWU7CiAgICAgIH0KICAgICAgY29uc3QgaCA9IGxpbmUubWF0Y2goL15ccyooI3sxLDZ9KVxzKyguKikkLyk7CiAgICAgIGlmIChoKXsgY2xvc2VMaXN0KCk7IGNvbnN0IGx2bD1oWzFdLmxlbmd0aDsgaHRtbCArPSBgPGgke2x2bH0+JHtpbmxpbmVNZChlc2MoaFsyXSkpfTwvaCR7bHZsfT5gOyBjb250aW51ZTsgfQogICAgICBpZiAoL15ccyooPzotLS18XCpcKlwqfF9fXylccyokLy50ZXN0KGxpbmUpKXsgY2xvc2VMaXN0KCk7IGh0bWwgKz0gJzxoci8+JzsgY29udGludWU7IH0KICAgICAgaWYgKC9eXHMqPlxzPy8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgcXVvdGVMaW5lcyA9IFtdOwogICAgICAgIHdoaWxlIChpIDwgbGluZXMubGVuZ3RoICYmIC9eXHMqPlxzPy8udGVzdChsaW5lc1tpXSkpIHsKICAgICAgICAgIHF1b3RlTGluZXMucHVzaChsaW5lc1tpXS5yZXBsYWNlKC9eXHMqPlxzPy8sICcnKSk7CiAgICAgICAgICBpKys7CiAgICAgICAgfQogICAgICAgIGktLTsKICAgICAgICBjb25zdCBxdW90ZVRleHQgPSBpbmxpbmVNZChlc2MocXVvdGVMaW5lcy5qb2luKCdcbicpKSk7CiAgICAgICAgaHRtbCArPSBgPGJsb2NrcXVvdGU+PHA+JHtxdW90ZVRleHQucmVwbGFjZSgvXG4vZywgJzxiciAvPicpfTwvcD48L2Jsb2NrcXVvdGU+YDsKICAgICAgICBjb250aW51ZTsKICAgICAgfQogICAgICAKICAgICAgLy8gLS0tIFNUQVJUOiBUYWJsZSBQYXJzaW5nIC0tLQogICAgICBjb25zdCBpc1RhYmxlTGluZSA9IChsKSA9PiBsLmluY2x1ZGVzKCd8Jyk7CiAgICAgIGNvbnN0IGlzU2VwYXJhdG9yTGluZSA9IChsKSA9PiAvXlxzKlx8PyhbLTpdK1x8KStbLTpdK1x8P1xzKiQvLnRlc3QobC50cmltKCkpOwogICAgICAKICAgICAgaWYgKGlzVGFibGVMaW5lKGxpbmUpICYmIGkgKyAxIDwgbGluZXMubGVuZ3RoICYmIGlzU2VwYXJhdG9yTGluZShsaW5lc1tpICsgMV0pKSB7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgaHRtbCArPSAnPHRhYmxlPic7CiAgICAgICAgCiAgICAgICAgLy8gSGVhZGVyCiAgICAgICAgY29uc3QgaGVhZGVycyA9IGxpbmUuc3BsaXQoJ3wnKS5tYXAoaCA9PiBoLnRyaW0oKSkuc2xpY2UoMSwgLTEpOwogICAgICAgIGh0bWwgKz0gJzx0aGVhZD48dHI+JzsKICAgICAgICBoZWFkZXJzLmZvckVhY2goaCA9PiB7IGh0bWwgKz0gYDx0aD4ke2lubGluZU1kKGVzYyhoKSl9PC90aD5gOyB9KTsKICAgICAgICBodG1sICs9ICc8L3RyPjwvdGhlYWQ+JzsKICAgICAgICAKICAgICAgICBpICs9IDI7IC8vIFNraXAgaGVhZGVyIHVuZCBzZXBhcmF0b3IgbGluZQogICAgICAgIAogICAgICAgIC8vIEJvZHkKICAgICAgICBodG1sICs9ICc8dGJvZHk+JzsKICAgICAgICB3aGlsZSAoaSA8IGxpbmVzLmxlbmd0aCAmJiBpc1RhYmxlTGluZShsaW5lc1tpXSkpIHsKICAgICAgICAgICAgY29uc3QgY2VsbHMgPSBsaW5lc1tpXS5zcGxpdCgnfCcpLm1hcChjID0+IGMudHJpbSgpKS5zbGljZSgxLCAtMSk7CiAgICAgICAgICAgIGlmIChjZWxscy5sZW5ndGggPiAwKSB7CiAgICAgICAgICAgICAgICBodG1sICs9ICc8dHI+JzsKICAgICAgICAgICAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgaGVhZGVycy5sZW5ndGg7IGorKykgewogICAgICAgICAgICAgICAgICAgIGNvbnN0IGNlbGxDb250ZW50ID0gY2VsbHNbal0gfHwgJyc7CiAgICAgICAgICAgICAgICAgICAgaHRtbCArPSBgPHRkPiR7aW5saW5lTWQoZXNjKGNlbGxDb250ZW50KSl9PC90ZD5gOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaHRtbCArPSAnPC90cj4nOwogICAgICAgICAgICB9CiAgICAgICAgICAgIGkrKzsKICAgICAgICB9CiAgICAgICAgaS0tOwogICAgICAgIAogICAgICAgIGh0bWwgKz0gJzwvdGJvZHk+PC90YWJsZT4nOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CiAgICAgIC8vIC0tLSBFTkQ6IFRhYmxlIFBhcnNpbmcgLS0tCgogICAgICBsZXQgbTsKICAgICAgaWYgKG0gPSBsaW5lLm1hdGNoKC9eXHMqWy0qK11ccytcWyggfHh8WClcXVxzKyguKikkLykpewogICAgICAgIGNvbnN0IGNoZWNrZWQgPSAveC9pLnRlc3QobVsxXSk7CiAgICAgICAgaWYgKGxpc3RUeXBlIT09J3VsJyl7IGNsb3NlTGlzdCgpOyBsaXN0VHlwZT0ndWwnOyBodG1sICs9ICc8dWw+JzsgfQogICAgICAgIGh0bWwgKz0gYDxsaSBjbGFzcz0iJHtjaGVja2VkPyd0YXNrLWRvbmUnOicnfSI+YCsoY2hlY2tlZD8n4pyFICc6J+KsnCAnKStpbmxpbmVNZChlc2MobVsyXSkpKyc8L2xpPic7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKlstKitdXHMrLy50ZXN0KGxpbmUpKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0ndWwnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSd1bCc7IGh0bWwgKz0gJzx1bD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKGxpbmUucmVwbGFjZSgvXlxzKlstKitdXHMrLywnJykpKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmIChtID0gbGluZS5tYXRjaCgvXlxzKihcZCspXC5ccysoLiopJC8pKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0nb2wnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSdvbCc7IGh0bWwgKz0gJzxvbD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKG1bMl0pKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKiQvLnRlc3QobGluZS50cmltKCkpKXsgY2xvc2VMaXN0KCk7IH0KICAgICAgZWxzZSB7IGNsb3NlTGlzdCgpOyBodG1sICs9IGA8cD4ke2lubGluZU1kKGVzYyhsaW5lKSl9PC9wPmA7IH0KICAgIH0KICAgIGNsb3NlTGlzdCgpOyByZXR1cm4gaHRtbDsKICB9CiAgZnVuY3Rpb24gZW5oYW5jZUNvZGVDb3B5KGNvbnRhaW5lcil7CiAgICBjb250YWluZXIucXVlcnlTZWxlY3RvckFsbCgnLmNvZGV3cmFwIC5jb3B5YnRuOm5vdChbZGF0YS1pbml0XSknKS5mb3JFYWNoKGJ0bj0+ewogICAgICBidG4uZGF0YXNldC5pbml0ID0gJzEnOwogICAgICBidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGNvbnN0IGNvZGUgPSBidG4ucGFyZW50RWxlbWVudC5xdWVyeVNlbGVjdG9yKCdjb2RlJyk7CiAgICAgICAgY29uc3QgdGV4dCA9IGNvZGUgPyBjb2RlLnRleHRDb250ZW50IDogJyc7CiAgICAgICAgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQodGV4dCkudGhlbigoKT0+eyBidG4udGV4dENvbnRlbnQ9J0NvcGllZCc7IHNldFRpbWVvdXQoKCk9PmJ0bi50ZXh0Q29udGVudD0nQ29weScsMTAwMCk7IH0pOwogICAgICB9KTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSBVSSBoZWxwZXJzIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiByZW5kZXJCdWJibGVDb250ZW50KGVsLCByb2xlLCBjb250ZW50KXsKICAgIGVsLmNsYXNzTGlzdC5hZGQoJ21kJyk7IGVsLmlubmVySFRNTCA9IG1kVG9IdG1sKG5vcm1hbGl6ZUNvbnRlbnQoY29udGVudCkpOyBlbmhhbmNlQ29kZUNvcHkoZWwpOwogIH0KICAKICBmdW5jdGlvbiBhZGRNZXNzYWdlKHJvbGUsIGNvbnRlbnQpIHsKICAgIGNvbnN0IGNscyA9IHJvbGUgPT09ICd1c2VyJyA/ICd1c2VyJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ3N5cycgOiAnYm90JzsKICAgIGNvbnN0IGF2YXRhciA9IHJvbGUgPT09ICd1c2VyJyA/ICfwn6eRJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ+Kame+4jycgOiAn8J+klic7CiAgICBjb25zdCBtc2dFbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgbXNnRWwuY2xhc3NOYW1lID0gYG1zZyAke2Nsc31gOwogICAgbXNnRWwuaW5uZXJIVE1MID0gYAogICAgICA8ZGl2IGNsYXNzPSJhdmF0YXIiPiR7YXZhdGFyfTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXNzYWdlLWNvbnRlbnQtd3JhcHBlciI+CiAgICAgICAgPGRpdiBjbGFzcz0iYnViYmxlIj4KICAgICAgICAgIDxkaXYgY2xhc3M9Im1ldGEiPiR7cm9sZX08L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNvbnRlbnQgbWQiPjwvZGl2PgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj5gOwogICAgZWxzLmxvZy5hcHBlbmRDaGlsZChtc2dFbCk7CiAgICBjb25zdCBjb250ZW50RWwgPSBtc2dFbC5xdWVyeVNlbGVjdG9yKCcuY29udGVudCcpOwogICAgcmVuZGVyQnViYmxlQ29udGVudChjb250ZW50RWwsIHJvbGUsIGNvbnRlbnQpOwogICAgaWYgKCFpc1JlbmRlcmluZykgbWF5YmVTY3JvbGwoKTsKICAgIHJldHVybiBjb250ZW50RWw7CiAgfQogIAogIGZ1bmN0aW9uIGFkZFRoaW5rQnViYmxlKGFzc2lzdGFudENvbnRlbnRFbCkgewogICAgY29uc3Qgd3JhcHBlciA9IGFzc2lzdGFudENvbnRlbnRFbC5jbG9zZXN0KCcubWVzc2FnZS1jb250ZW50LXdyYXBwZXInKTsKICAgIGlmICghd3JhcHBlcikgcmV0dXJuIG51bGw7CiAgICBjb25zdCBidWJibGUgPSB3cmFwcGVyLnF1ZXJ5U2VsZWN0b3IoJy5idWJibGUnKTsKICAgIGlmICghYnViYmxlKSByZXR1cm4gbnVsbDsKCiAgICBjb25zdCB0aGlua0VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7CiAgICB0aGlua0VsLmNsYXNzTmFtZSA9ICd0aGluay1idWJibGUnOwogICAgdGhpbmtFbC5pbm5lckhUTUwgPSBgCiAgICAgIDxkZXRhaWxzPgogICAgICAgIDxzdW1tYXJ5PvCfpJQgU2hvdyB0aGlua2luZzwvc3VtbWFyeT4KICAgICAgICA8ZGl2IGNsYXNzPSJ0aGluay1jb250ZW50IG1kIj48L2Rpdj4KICAgICAgPC9kZXRhaWxzPgogICAgYDsKICAgIHdyYXBwZXIuaW5zZXJ0QmVmb3JlKHRoaW5rRWwsIGJ1YmJsZSk7CiAgICByZXR1cm4gdGhpbmtFbC5xdWVyeVNlbGVjdG9yKCcudGhpbmstY29udGVudCcpOwogIH0KCiAgZnVuY3Rpb24gc2V0U3RyZWFtaW5nU3RhdGUoYWN0aXZlKSB7IGVscy5zZW5kQnRuLmRpc2FibGVkID0gYWN0aXZlOyBlbHMuc3RvcEJ0bi5kaXNhYmxlZCA9ICFhY3RpdmU7IH0KCiAgZnVuY3Rpb24gcmVuZGVyQWxsKCl7CiAgICBpc1JlbmRlcmluZyA9IHRydWU7CiAgICBlbHMubG9nLmlubmVySFRNTCA9ICcnOwogICAgZm9yIChjb25zdCBtIG9mIG1lc3NhZ2VzKSBhZGRNZXNzYWdlKG0ucm9sZSwgbS5jb250ZW50KTsKICAgIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICBhdXRvc2Nyb2xsID0gdHJ1ZTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKCdzaG93Jyk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIE5vcm1hbGl6ZXIgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIG5vcm1hbGl6ZUNvbnRlbnQoYyl7CiAgICBpZiAodHlwZW9mIGMgPT09ICdzdHJpbmcnKSByZXR1cm4gYzsKICAgIGlmIChBcnJheS5pc0FycmF5KGMpKSB7CiAgICAgIHJldHVybiBjLm1hcChwID0+IHsKICAgICAgICBpZiAodHlwZW9mIHAgPT09ICdzdHJpbmcnKSByZXR1cm4gcDsKICAgICAgICBpZiAocD8udHlwZSA9PT0gJ3RleHQnKSByZXR1cm4gcC50ZXh0ID8/ICcnOwogICAgICAgIGlmIChwPy50eXBlID09PSAnaW1hZ2VfdXJsJykgewogICAgICAgICAgY29uc3QgdSA9IHA/LmltYWdlX3VybD8udXJsIHx8ICcnOwogICAgICAgICAgcmV0dXJuIHUgPyBgIVtpbWFnZV0oJHt1fSlgIDogJyc7CiAgICAgICAgfQogICAgICAgIHJldHVybiBwPy50ZXh0ID8/ICcnOwogICAgICB9KS5qb2luKFN0cmluZy5mcm9tQ2hhckNvZGUoMTApKTsKICAgIH0KICAgIGlmIChjICYmIHR5cGVvZiBjID09PSAnb2JqZWN0JyAmJiAndGV4dCcgaW4gYykgcmV0dXJuIFN0cmluZyhjLnRleHQgPz8gJycpOwogICAgdHJ5IHsgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGMpOyB9IGNhdGNoIHsgcmV0dXJuIFN0cmluZyhjKTsgfQogIH0KICBmdW5jdGlvbiBub3JtYWxpemVSb2xlKHIpeyBjb25zdCBzPVN0cmluZyhyfHwnJykudG9Mb3dlckNhc2UoKTsgaWYgKHMuc3RhcnRzV2l0aCgnYXNzaXN0JykpIHJldHVybiAnYXNzaXN0YW50JzsgaWYgKHMuc3RhcnRzV2l0aCgnc3lzJykpIHJldHVybiAnc3lzdGVtJzsgcmV0dXJuICd1c2VyJzsgfQoKICAvKiAtLS0tLS0tLS0tIEV4cG9ydCBpbmtsLiBzeXN0ZW0gLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGJ1aWxkRXhwb3J0QXJyYXkoKXsKICAgIGNvbnN0IG91dCA9IG1lc3NhZ2VzLm1hcChtID0+ICh7IHJvbGU6IG0ucm9sZSwgY29udGVudDogbS5jb250ZW50IH0pKTsKICAgIGNvbnN0IHN5c1RleHQgPSAoZWxzLnN5c3RlbS52YWx1ZSB8fCAnJykudHJpbSgpOwogICAgY29uc3QgZmlyc3RJc1N5c3RlbSA9IG91dFswXT8ucm9sZSA9PT0gJ3N5c3RlbSc7CiAgICBpZiAoc3lzVGV4dCkgewogICAgICBpZiAoIWZpcnN0SXNTeXN0ZW0pIG91dC51bnNoaWZ0KHsgcm9sZTogJ3N5c3RlbScsIGNvbnRlbnQ6IHN5c1RleHQgfSk7CiAgICAgIGVsc2UgaWYgKG91dFswXS5jb250ZW50ICE9PSBzeXNUZXh0KSBvdXRbMF0gPSB7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNUZXh0IH07CiAgICB9CiAgICByZXR1cm4gb3V0OwogIH0KICAgIAogIGZ1bmN0aW9uIHVwZGF0ZVRva2VuUmF0ZShidWZmZXIsIGZvcmNlRmluYWwgPSBmYWxzZSkgewogICAgY29uc3Qgbm93ID0gcGVyZm9ybWFuY2Uubm93KCk7CiAgICBpZiAoIWZvcmNlRmluYWwgJiYgbm93IC0gbGFzdFVwZGF0ZVRpbWUgPCBVUERBVEVfSU5URVJWQUwpIHJldHVybjsKICAgIGxhc3RVcGRhdGVUaW1lID0gbm93OwoKICAgIGNvbnN0IGVsYXBzZWRTZWNvbmRzID0gKG5vdyAtIHN0cmVhbVN0YXJ0VGltZSkgLyAxMDAwOwogICAgaWYgKGVsYXBzZWRTZWNvbmRzIDwgMC4xKSByZXR1cm47CgogICAgbGV0IHRva2VucywgcmF0ZSwgbGFiZWw7CiAgICBpZiAoZmluYWxDb21wbGV0aW9uVG9rZW5zID4gMCkgewogICAgICAgIHRva2VucyA9IGZpbmFsQ29tcGxldGlvblRva2VuczsKICAgICAgICByYXRlID0gdG9rZW5zIC8gZWxhcHNlZFNlY29uZHM7CiAgICAgICAgbGFiZWwgPSBgw5ggJHtyYXRlLnRvRml4ZWQoMSl9IHQvc2A7CiAgICB9IGVsc2UgewogICAgICAgIC8vIFNjaMOkdHp1bmc6IGNhLiA0IFplaWNoZW4gcHJvIFRva2VuCiAgICAgICAgdG9rZW5zID0gTWF0aC5yb3VuZChidWZmZXIubGVuZ3RoIC8gNCk7CiAgICAgICAgcmF0ZSA9IHRva2VucyAvIGVsYXBzZWRTZWNvbmRzOwogICAgICAgIGxhYmVsID0gYH4ke3JhdGUudG9GaXhlZCgxKX0gdC9zYDsKICAgIH0KICAgIGVscy50b2tlblJhdGUudGV4dENvbnRlbnQgPSBsYWJlbDsKICB9CgoKICAvKiAtLS0tLS0tLS0tIEthZ2dsZSBHUFUg5q6L44KK5pmC6ZaTIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3F1b3RhIOOCkuWPqeOBj+OAguODreODvOOCq+ODqyB1enUg44Gr44GvIC9xdW90YSDjgYznhKHjgYTjga7jgafjgIEKICAvLyDjgZ3jga7loLTlkIjjga8gNDA0IOOBjOi/lOOCiuOAgeihqOekuuOBr+WHuuOBquOBhO+8iOOCqOODqeODvOaJseOBhOOBq+OBl+OBquOBhO+8ieOAggogIGxldCBxdW90YVRpbWVyID0gbnVsbDsKCiAgZnVuY3Rpb24gZm10TGVmdChzZWMpIHsKICAgIGlmIChzZWMgPT0gbnVsbCB8fCBzZWMgPCAwKSByZXR1cm4gJyc7CiAgICBjb25zdCBoID0gTWF0aC5mbG9vcihzZWMgLyAzNjAwKTsKICAgIGNvbnN0IG0gPSBNYXRoLmZsb29yKChzZWMgJSAzNjAwKSAvIDYwKTsKICAgIHJldHVybiBg5q6L44KKICR7U3RyaW5nKGgpLnBhZFN0YXJ0KDIsICcwJyl9OiR7U3RyaW5nKG0pLnBhZFN0YXJ0KDIsICcwJyl9YDsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIHJlZnJlc2hRdW90YSgpIHsKICAgIGlmICghZWxzLnF1b3RhTGVmdCkgcmV0dXJuOwogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9xdW90YWAsIHsKICAgICAgICBtZXRob2Q6ICdHRVQnLAogICAgICAgIGhlYWRlcnM6IGFwaUhlYWRlcnMoeyAnQWNjZXB0JzogJ2FwcGxpY2F0aW9uL2pzb24nIH0pLAogICAgICAgIGNhY2hlOiAnbm8tc3RvcmUnLAogICAgICB9KTsKICAgICAgaWYgKCFyZXMub2spIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KICAgICAgY29uc3QgZCA9IGF3YWl0IHJlcy5qc29uKCk7CiAgICAgIGlmICghZC5hdmFpbGFibGUpIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KCiAgICAgIGNvbnN0IGxlZnQgPSBkLmxlZnRTZWNvbmRzOwogICAgICBlbHMucXVvdGFMZWZ0LnRleHRDb250ZW50ID0gZm10TGVmdChsZWZ0KTsKICAgICAgLy8g5q6L44KK44GM5bCR44Gq44GE44Go6Imy44KS5aSJ44GI44KLCiAgICAgIGVscy5xdW90YUxlZnQuc3R5bGUuY29sb3IgPSBsZWZ0IDwgMTgwMCA/ICcjZmY2YjZiJwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDogbGVmdCA8IDcyMDAgPyAnI2UwYTAzMCcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA6ICcnOwogICAgICBlbHMucXVvdGFMZWZ0LnRpdGxlID0gYEthZ2dsZSBHUFU6IOaui+OCiiAke2ZtdExlZnQobGVmdCkucmVwbGFjZSgn5q6L44KKICcsICcnKX1cbmAKICAgICAgICAgICAgICAgICAgICAgICAgICArIGDkvb/nlKggJHtNYXRoLmZsb29yKGQudXNlZFNlY29uZHMgLyAzNjAwKX1oIC8g5LiK6ZmQICR7TWF0aC5mbG9vcihkLnRvdGFsU2Vjb25kcyAvIDM2MDApfWhgCiAgICAgICAgICAgICAgICAgICAgICAgICAgKyAoZC5yZWZyZXNoVGltZSA/IGBcbuODquOCu+ODg+ODiCAke2QucmVmcmVzaFRpbWV9YCA6ICcnKTsKICAgIH0gY2F0Y2ggKF8pIHsKICAgICAgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOwogICAgfQogIH0KCiAgZnVuY3Rpb24gc3RhcnRRdW90YVBvbGxpbmcoKSB7CiAgICByZWZyZXNoUXVvdGEoKTsKICAgIC8vIDEg5YiG44GU44Go44CC5q6L6YeP44Gv56eS5Y2Y5L2N44Gn5rib44KL44GM44CB6KGo56S644Gv5YiG5Y2Y5L2N44Gq44Gu44Gn5Y2B5YiG44CCCiAgICBpZiAocXVvdGFUaW1lcikgY2xlYXJJbnRlcnZhbChxdW90YVRpbWVyKTsKICAgIHF1b3RhVGltZXIgPSBzZXRJbnRlcnZhbChyZWZyZXNoUXVvdGEsIDYwMDAwKTsKICB9CgogIC8qIC0tLS0tLS0tLS0g5YGc5q2i44Oc44K/44Oz77yI44Ot44O844Kr44Or54mI44Gu44G/77yJIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3N0b3Ag44KS5Y+p44GP44CC5Lit57aZ44GM6JC944Gh44KL44Gu44Gn44CB44GT44GuIFVJIOOCguS9v+OBiOOBquOBj+OBquOCi+OAggogIC8vIOODreODvOOCq+ODqyB1enUg44Gr44GvIC9zdG9wIOOBjOeEoeOBhOOBruOBp+OAgTQwNCDjgarjgonjg5zjgr/jg7PjgpLpmqDjgZnjgIIKICBhc3luYyBmdW5jdGlvbiBkZXRlY3RTdG9wU3VwcG9ydCgpIHsKICAgIGlmICghZWxzLnN0b3BSb3cpIHJldHVybjsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKGAke2xvY2F0aW9uLm9yaWdpbn0vcXVvdGFgLCB7CiAgICAgICAgbWV0aG9kOiAnR0VUJywgaGVhZGVyczogYXBpSGVhZGVycygpLCBjYWNoZTogJ25vLXN0b3JlJywKICAgICAgfSk7CiAgICAgIC8vIC9xdW90YSDjgYwgSlNPTiDjgpLov5TjgZkgPSDkuK3ntpnjgYznq4vjgaPjgabjgYTjgosgPSDlgZzmraLjgafjgY3jgovjgIIKICAgICAgLy8gYXZhaWxhYmxlPWZhbHNl77yIdXp1IOOBquOBqeOCr+OCqeODvOOCv+eEoeOBl++8ieOBp+OCguS4ree2meiHquS9k+OBr+eUn+OBjeOBpuOBhOOCi+OBruOBpwogICAgICAvLyDlgZzmraLjg5zjgr/jg7Pjga/lh7rjgZnjgIIKICAgICAgaWYgKHJlcy5vaykgewogICAgICAgIGNvbnN0IGQgPSBhd2FpdCByZXMuanNvbigpOwogICAgICAgIGlmIChkICYmIHR5cGVvZiBkLmF2YWlsYWJsZSA9PT0gJ2Jvb2xlYW4nKSB7CiAgICAgICAgICBlbHMuc3RvcFJvdy5zdHlsZS5kaXNwbGF5ID0gJyc7CiAgICAgICAgfQogICAgICB9CiAgICB9IGNhdGNoIChfKSB7IC8qIOS4ree2meOBp+OBquOBhOOBquOCieWHuuOBleOBquOBhCAqLyB9CiAgfQoKICBhc3luYyBmdW5jdGlvbiBzdG9wU2VydmVyKCkgewogICAgaWYgKCFjb25maXJtKCfjgrXjg7zjg5Djg7zjgpLlgZzmraLjgZfjgb7jgZnjgILjgojjgo3jgZfjgYTjgafjgZnjgYvvvJ9cbu+8iOOBk+OBrueUu+mdouOBr+S9v+OBiOOBquOBj+OBquOCiuOBvuOBme+8iScpKSByZXR1cm47CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgICAgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSB0cnVlOwogICAgICBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICflgZzmraLkuK3igKYnOwogICAgfQogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9zdG9wYCwgewogICAgICAgIG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzOiBhcGlIZWFkZXJzKHsgJ1gtS2MtU3RvcCc6ICcxJyB9KSwgY2FjaGU6ICduby1zdG9yZScsCiAgICAgIH0pOwogICAgICBpZiAocmVzLnN0YXR1cyA9PT0gNDAxKSB7CiAgICAgICAgb3BlbkF1dGhEaWFsb2coJ+WBnOatouOBq+OBr+iqjeiovOOBjOW/heimgeOBp+OBme+8iDQwMe+8iScpOwogICAgICAgIGlmIChlbHMuc3RvcFNlcnZlcikgeyBlbHMuc3RvcFNlcnZlci5kaXNhYmxlZCA9IGZhbHNlOyBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICfilqAg5YGc5q2iJzsgfQogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICBpZiAoIXJlcy5vaykgewogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5YGc5q2iXSDlpLHmlZfjgZfjgb7jgZfjgZ8gKEhUVFAgJHtyZXMuc3RhdHVzfSlgKTsKICAgICAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSBmYWxzZTsgZWxzLnN0b3BTZXJ2ZXIudGV4dENvbnRlbnQgPSAn4pagIOWBnOatoic7IH0KICAgICAgICByZXR1cm47CiAgICAgIH0KICAgIH0gY2F0Y2ggKF8pIHsgLyog5Lit57aZ44GM6JC944Gh44KL44Gu44Gn6YCa5L+h44Ko44Op44O844Gv5oOz5a6a5YaFICovIH0KICAgIHNldFN0YXR1cygn5YGc5q2iJyk7CiAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCAnW+WBnOatol0g44K144O844OQ44O844KS5YGc5q2i44GX44G+44GX44Gf44CC44GT44Gu55S76Z2i44KS6ZaJ44GY44Gm44GP44Gg44GV44GE44CCJyk7CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIGVscy5zdG9wU2VydmVyLnRleHRDb250ZW50ID0gJ+WBnOatoua4iOOBvyc7CiAgfQoKICAvKiAtLS0tLS0tLS0tIEFQSSBCYXNlIHZhbGlkYXRpb24gKE1peGVkIGNvbnRlbnQgZ3VhcmQpIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBpc01peGVkQ29udGVudEFwaUJhc2UoKXsKICAgIGNvbnN0IHBhZ2VQcm90byA9IHdpbmRvdy5sb2NhdGlvbi5wcm90b2NvbDsKICAgIGNvbnN0IGJhc2UgPSAoZWxzLmFwaUJhc2U/LnZhbHVlIHx8ICcnKS50cmltKCk7CiAgICByZXR1cm4gcGFnZVByb3RvID09PSAnaHR0cHM6JyAmJiAvXmh0dHA6XC9cLy9pLnRlc3QoYmFzZSk7CiAgfQogIGZ1bmN0aW9uIGVuc3VyZUFwaUJhc2VBbGxvd2VkKCl7CiAgICBpZiAoaXNNaXhlZENvbnRlbnRBcGlCYXNlKCkpewogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLAogICAgICAgIGBbQ29uZmlnXSBNaXhlZCBjb250ZW50IGJsb2NrZWQ6IHRoaXMgVUkgaXMgbG9hZGVkIHZpYSBIVFRQUywgYnV0IEFQSSBCYXNlIGlzIEhUVFAuXG5gICsKICAgICAgICBgQnJvd3NlcnMgd2lsbCB1c3VhbGx5IGJsb2NrIHRoaXMgcmVxdWVzdC4gVXNlIGFuIGh0dHBzOi8vIEFQSSBCYXNlLCBvciBvcGVuIHRoZSBVSSB2aWEgaHR0cDovLyBvciBmaWxlOi8vLmAKICAgICAgKTsKICAgICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcjYjQwMDAwJzsgfWNhdGNoe30KICAgICAgcmV0dXJuIGZhbHNlOwogICAgfQogICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fQogICAgcmV0dXJuIHRydWU7CiAgfQogIC8vIENsZWFyIHRoZSByZWQgYm9yZGVyIGFzIHNvb24gYXMgdGhlIHVzZXIgZWRpdHMgdGhlIGZpZWxkCiAgaWYgKGVscy5hcGlCYXNlKSB7CiAgICBlbHMuYXBpQmFzZS5hZGRFdmVudExpc3RlbmVyKCdpbnB1dCcsICgpID0+IHsgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fSB9KTsKICB9CgoKICAvKiAtLS0tLS0tLS0tIE1vZGVscyAoR0VUIC92MS9tb2RlbHMpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBsb2FkTW9kZWxzKCkgewogICAgaWYgKCFlbnN1cmVBcGlCYXNlQWxsb3dlZCgpKSByZXR1cm47CiAgICBjb25zdCBiYXNlID0gQVBJX0JBU0UucmVwbGFjZSgvXC8kLywgJycpOwogICAgY29uc3QgdXJsID0gYCR7YmFzZX0vbW9kZWxzYDsKCiAgICBjb25zdCBoZWFkZXJzID0gYXBpSGVhZGVycyh7ICdBY2NlcHQnOiAnYXBwbGljYXRpb24vanNvbicgfSk7CgogICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0xvYWRpbmfigKYnOwogICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IHRydWU7CgogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ0dFVCcsIGhlYWRlcnMsIG1vZGU6ICdjb3JzJyB9KTsKICAgICAgaWYgKHJlcy5zdGF0dXMgPT09IDQwMSkgb3BlbkF1dGhEaWFsb2coJ0FQSSDoqo3oqLzjgYzlv4XopoHjgafjgZnvvIg0MDHvvIknKTsKICAgICAgaWYgKCFyZXMub2spIHsKICAgICAgICBjb25zdCB0ID0gYXdhaXQgcmVzLnRleHQoKS5jYXRjaCgoKT0+ICcnKTsKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdGYWlsZWQgdG8gbG9hZCBtb2RlbHMnfWApOwogICAgICB9CiAgICAgIGNvbnN0IGRhdGEgPSBhd2FpdCByZXMuanNvbigpOwoKICAgICAgLy8gT3BlbkFJLVN0eWxlOiB7IGRhdGE6IFt7aWQ6ICIuLi4ifSwgLi4uXSB9CiAgICAgIC8vIEZhbGxiYWNrcyBmw7xyIGFuZGVyZSBTZXJ2ZXI6CiAgICAgIGNvbnN0IGxpc3QgPQogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8uZGF0YSkgPyBkYXRhLmRhdGEgOgogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8ubW9kZWxzKSA/IGRhdGEubW9kZWxzIDoKICAgICAgICBBcnJheS5pc0FycmF5KGRhdGEpID8gZGF0YSA6CiAgICAgICAgW107CgogICAgICBjb25zdCBpZHMgPSBsaXN0CiAgICAgICAgLm1hcChtID0+ICh0eXBlb2YgbSA9PT0gJ3N0cmluZycgPyBtIDogbT8uaWQpKQogICAgICAgIC5maWx0ZXIoQm9vbGVhbikKICAgICAgICAubWFwKFN0cmluZyk7CgogICAgICBpZiAoIWlkcy5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gbW9kZWxzIGZvdW5kIGluIC92MS9tb2RlbHMuJyk7CgogICAgICBpZHMuc29ydCgoYSxiKT0+YS5sb2NhbGVDb21wYXJlKGIpKTsKCiAgICAgIGNvbnN0IHByZXYgPSBlbHMubW9kZWwudmFsdWU7CiAgICAgIGVscy5tb2RlbC5pbm5lckhUTUwgPSAnJzsKICAgICAgZm9yIChjb25zdCBpZCBvZiBpZHMpIHsKICAgICAgICBjb25zdCBvcHQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdvcHRpb24nKTsKICAgICAgICBvcHQudmFsdWUgPSBpZDsKICAgICAgICBvcHQudGV4dENvbnRlbnQgPSBpZDsKICAgICAgICBlbHMubW9kZWwuYXBwZW5kQ2hpbGQob3B0KTsKICAgICAgfQoKICAgICAgLy8gQXVzd2FobDogdm9yaGVyaWdlIGJlaGFsdGVuIHdlbm4gbcO2Z2xpY2gsIHNvbnN0IGVyc3RlcwogICAgICBlbHMubW9kZWwudmFsdWUgPSBpZHMuaW5jbHVkZXMocHJldikgPyBwcmV2IDogaWRzWzBdOwoKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gYCR7aWRzLmxlbmd0aH0gbW9kZWxzYDsKICAgIH0gY2F0Y2ggKGUpIHsKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0Vycm9yJzsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjg6Ljg4fjg6voqq3jgb/ovrzjgb9dICR7ZS5tZXNzYWdlfWApOwogICAgfSBmaW5hbGx5IHsKICAgICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IGZhbHNlOwogICAgfQogIH0KCiAgLyogLS0tLS0tLS0tLSBTZW5kIChVbnRlcnN0w7x0enQgPHRoaW5rPiB1bmQgInJlYXNvbmluZyIsIG1pdCBQZXJmb3JtYW5jZS1GaXgpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBzZW5kTWVzc2FnZSgpIHsKICAgIGxldCBvdXRFbCA9IG51bGw7CiAgICBsZXQgYXNzaXN0YW50QnVmZmVyID0gJyc7CiAgICBsZXQgdGhpbmtCdWZmZXIgPSAnJzsKICAgIGxldCBzYXZlZFRleHQgPSAnJzsKICAgIGxldCBzYXZlZEF0dGFjaG1lbnRzID0gW107CiAgICB0cnkgewogICAgICBjb25zdCB0ZXh0ID0gZWxzLmlucHV0LnZhbHVlLnRyaW0oKTsKICAgICAgc2F2ZWRUZXh0ID0gdGV4dDsKICAgICAgc2F2ZWRBdHRhY2htZW50cyA9IFsuLi5hdHRhY2htZW50c107CiAgICAgIGlmICghdGV4dCAmJiBhdHRhY2htZW50cy5sZW5ndGggPT09IDApIHJldHVybjsKCiAgICAgIC8vIOOCueODiOODquODvOODn+ODs+OCsOS4reOBr+mAgeS/oeOBl+OBquOBhO+8iOWFpeWKm+OBr+WPr+iDve+8iQogICAgICBpZiAoIWVscy5zdG9wQnRuLmRpc2FibGVkKSByZXR1cm47CgogICAgICBpZiAoIWVuc3VyZUFwaUJhc2VBbGxvd2VkKCkpIHJldHVybjsKCiAgICAgIGlmICghZWxzLm1vZGVsLnZhbHVlKSB7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgJ1vjg5Ljg7Pjg4hdIOODouODh+ODq+acqumBuOaKniDigJQg44Oi44OH44Or44GM6Kqt44G/6L6844G+44KM44Gm44GE44G+44Gb44KT44CC44Oa44O844K444KS44Oq44Ot44O844OJ44GX44Gm44GP44Gg44GV44GE44CCJyk7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIAogICAgICBlbHMudG9rZW5SYXRlLnRleHRDb250ZW50ID0gJyc7CiAgICAgIGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgICAgIHN0cmVhbVN0YXJ0VGltZSA9IDA7CgogICAgICAvLyDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4jjgpLmr47lm57lj43mmKAKICAgICAgY29uc3Qgc3lzVmFsID0gZWxzLnN5c3RlbS52YWx1ZS50cmltKCk7CiAgICAgIGlmIChzeXNWYWwpIHsKICAgICAgICBjb25zdCBleGlzdGluZ1N5c0lkeCA9IG1lc3NhZ2VzLmZpbmRJbmRleChtID0+IG0ucm9sZSA9PT0gJ3N5c3RlbScpOwogICAgICAgIGlmIChleGlzdGluZ1N5c0lkeCA+PSAwKSB7CiAgICAgICAgICBtZXNzYWdlc1tleGlzdGluZ1N5c0lkeF0uY29udGVudCA9IHN5c1ZhbDsKICAgICAgICB9IGVsc2UgaWYgKG1lc3NhZ2VzLmxlbmd0aCA9PT0gMCkgewogICAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNWYWwgfSk7CiAgICAgICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBzeXNWYWwpOwogICAgICAgIH0KICAgICAgfQogICAgICBjb25zdCBoYXNBdHQgPSBhdHRhY2htZW50cy5sZW5ndGggPiAwOwogICAgICBpZiAodGV4dCB8fCBoYXNBdHQpIHsKICAgICAgICBjb25zdCBwYXJ0cyA9IFtdOwogICAgICAgIGlmICh0ZXh0KSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQgfSk7CiAgICAgICAgZWxzZSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQ6ICdBdHRhY2hlZCBmaWxlczonIH0pOwoKICAgICAgICBjb25zdCB0ZXh0RmlsZXMgPSBhdHRhY2htZW50cy5maWx0ZXIoYSA9PiBhLmtpbmQgPT09ICd0ZXh0Jyk7CiAgICAgICAgaWYgKHRleHRGaWxlcy5sZW5ndGgpewogICAgICAgICAgbGV0IGJsb2IgPSAnXG5cbi0tLVxuQXR0YWNoZWQgZmlsZXMgKHZlcmJhdGltKTpcbic7CiAgICAgICAgICBmb3IgKGNvbnN0IGEgb2YgdGV4dEZpbGVzKXsKICAgICAgICAgICAgYmxvYiArPSBgXG5bRmlsZTogJHthLm5hbWV9JHthLnRydW5jYXRlZD8nIChUUlVOQ0FURUQpJzonJ31dXG5cYFxgXGBcbiR7YS50ZXh0fVxuXGBcYFxgXG5gOwogICAgICAgICAgfQogICAgICAgICAgcGFydHMucHVzaCh7IHR5cGU6J3RleHQnLCB0ZXh0OiBibG9iIH0pOwogICAgICAgIH0KCiAgICAgICAgY29uc3QgaW1ncyA9IGF0dGFjaG1lbnRzLmZpbHRlcihhID0+IGEua2luZCA9PT0gJ2ltYWdlJyk7CiAgICAgICAgZm9yIChjb25zdCBhIG9mIGltZ3MpewogICAgICAgICAgaWYgKGEuZGF0YVVybCl7CiAgICAgICAgICAgIHBhcnRzLnB1c2goeyB0eXBlOidpbWFnZV91cmwnLCBpbWFnZV91cmw6eyB1cmw6IGEuZGF0YVVybCB9fSk7CiAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBjb25zdCBjb250ZW50ID0gKHBhcnRzLmxlbmd0aCA9PT0gMSkgPyAocGFydHNbMF0udGV4dCA/PyAnJykgOiBwYXJ0czsKCiAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICd1c2VyJywgY29udGVudCB9KTsKICAgICAgICBhZGRNZXNzYWdlKCd1c2VyJywgY29udGVudCk7CiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gJyc7CgogICAgICAgIGF0dGFjaG1lbnRzID0gW107CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfQoKCiAgICAgIG91dEVsID0gYWRkTWVzc2FnZSgnYXNzaXN0YW50JywgJycpOwoKICAgICAgLy8gTkVVOiBvcHRpb25hbGUgU2FtcGxpbmctUGFyYW1ldGVyIG51ciBtaXRzZW5kZW4sIHdlbm4gZ2VzZXR6dAogICAgICBmdW5jdGlvbiBwYXJzZU9wdGlvbmFsRmxvYXQoZWwsIG1pbiwgbWF4KSB7CiAgICAgICAgY29uc3QgdiA9IChlbD8udmFsdWUgPz8gJycpLnRyaW0oKTsKICAgICAgICBpZiAodiA9PT0gJycpIHJldHVybiBudWxsOwogICAgICAgIGNvbnN0IG51bSA9IHBhcnNlRmxvYXQodik7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1heCA9PT0gJ251bWJlcicgJiYgbnVtID4gbWF4KSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGZ1bmN0aW9uIHBhcnNlT3B0aW9uYWxJbnQoZWwsIG1pbikgewogICAgICAgIGNvbnN0IHYgPSAoZWw/LnZhbHVlID8/ICcnKS50cmltKCk7CiAgICAgICAgaWYgKHYgPT09ICcnKSByZXR1cm4gbnVsbDsKICAgICAgICBjb25zdCBudW0gPSBwYXJzZUludCh2LCAxMCk7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGNvbnN0IGV4dHJhcyA9IHt9OwogICAgICBjb25zdCBfdG9wX3AgPSBwYXJzZU9wdGlvbmFsRmxvYXQoZWxzLnRvcF9wLCAwLCAxKTsKICAgICAgY29uc3QgX21pbl9wID0gcGFyc2VPcHRpb25hbEZsb2F0KGVscy5taW5fcCwgMCwgMSk7CiAgICAgIGNvbnN0IF90b3BfayA9IHBhcnNlT3B0aW9uYWxJbnQoZWxzLnRvcF9rLCAwKTsKICAgICAgaWYgKF90b3BfcCAhPT0gbnVsbCkgZXh0cmFzLnRvcF9wID0gX3RvcF9wOwogICAgICBpZiAoX21pbl9wICE9PSBudWxsKSBleHRyYXMubWluX3AgPSBfbWluX3A7CiAgICAgIGlmIChfdG9wX2sgIT09IG51bGwpIGV4dHJhcy50b3BfayA9IF90b3BfazsKCiAgICAgIC8vIHRoaW5raW5nOiB1enUgcmVhZHMgcmVhc29uaW5nX2VmZm9ydCBwZXIgcmVxdWVzdC4gT3RoZXIgYmFja2VuZHMgaWdub3JlIGl0LgogICAgICBjb25zdCBfdGhpbmtpbmcgPSAoZWxzLnRoaW5raW5nPy52YWx1ZSA/PyAnJykudHJpbSgpOwogICAgICBpZiAoX3RoaW5raW5nID09PSAnb2ZmJykgZXh0cmFzLnJlYXNvbmluZ19lZmZvcnQgPSAnZGlzYWJsZWQnOwogICAgICBlbHNlIGlmIChfdGhpbmtpbmcgPT09ICdvbicpIGV4dHJhcy5yZWFzb25pbmdfZWZmb3J0ID0gJ2RlZmF1bHQnOwoKICAgICAgY29uc3QgYm9keSA9IHsgbW9kZWw6IGVscy5tb2RlbC52YWx1ZS50cmltKCksIG1lc3NhZ2VzLCBzdHJlYW06IHRydWUsIHRlbXBlcmF0dXJlOiBwYXJzZUZsb2F0KGVscy50ZW1wLnZhbHVlKSwgLi4uZXh0cmFzIH07CiAgICAgIGNvbnN0IGhlYWRlcnMgPSBhcGlIZWFkZXJzKHsgJ0NvbnRlbnQtVHlwZSc6ICdhcHBsaWNhdGlvbi9qc29uJywgJ0FjY2VwdCc6ICd0ZXh0L2V2ZW50LXN0cmVhbScgfSk7CiAgICAgIGNvbnN0IHVybCA9IGAke0FQSV9CQVNFLnJlcGxhY2UoL1wvJC8sICcnKX0vY2hhdC9jb21wbGV0aW9uc2A7CgogICAgICBhYm9ydENvbnRyb2xsZXIgPSBuZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKHRydWUpOwoKICAgICAgbGV0IGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICBsZXQgdGhpbmtFbCA9IG51bGw7CgogICAgICAvLyAtLS0gU1RBUlQgw4ROREVSVU5HOiBDT1JTIGhpbnp1Z2Vmw7xndCAtLS0KICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzLCBib2R5OiBKU09OLnN0cmluZ2lmeShib2R5KSwgc2lnbmFsOiBhYm9ydENvbnRyb2xsZXIuc2lnbmFsLCBtb2RlOiAnY29ycycgfSk7CiAgICAgIC8vIC0tLSBFTkRFIMOETkRFUlVORyAtLS0KCiAgICAgIGlmIChyZXMuc3RhdHVzID09PSA0MDEpIG9wZW5BdXRoRGlhbG9nKCdBUEkg6KqN6Ki844GM5b+F6KaB44Gn44GZ77yINDAx77yJJyk7CiAgICAgIGlmICghcmVzLm9rKSB7IGNvbnN0IHQgPSBhd2FpdCByZXMudGV4dCgpLmNhdGNoKCgpPT4gJycpOyB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdTZXJ2ZXIgZXJyb3InfWApOyB9CiAgICAgIAogICAgICBjb25zdCBjdHlwZSA9IChyZXMuaGVhZGVycy5nZXQoJ2NvbnRlbnQtdHlwZScpIHx8ICcnKS50b0xvd2VyQ2FzZSgpOwogICAgICBpZiAoIWN0eXBlLmluY2x1ZGVzKCd0ZXh0L2V2ZW50LXN0cmVhbScpKSB7CiAgICAgICAgY29uc3QgcmVxdWVzdFN0YXJ0VGltZSA9IHBlcmZvcm1hbmNlLm5vdygpOwogICAgICAgIGNvbnN0IHJhdyA9IGF3YWl0IHJlcy50ZXh0KCk7IAogICAgICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0VtcHR5IHJlc3BvbnNlIChzdGF0dXMgT0ssIGJ1dCBubyBjb250ZW50KS4nKTsKICAgICAgICAKICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSByZXF1ZXN0U3RhcnRUaW1lOwogICAgICAgIGxldCBkYXRhOyB0cnl7IGRhdGEgPSBKU09OLnBhcnNlKHJhdyk7fWNhdGNoeyB0aHJvdyBuZXcgRXJyb3IoYFVuZXJ3YXJ0ZXRlcyBGb3JtYXQ6ICR7cmF3LnNsaWNlKDAsMjAwKX0uLi5gKTsgfQogICAgICAgIAogICAgICAgIGxldCBmdWxsQ29udGVudCA9IGRhdGE/LmNob2ljZXM/LlswXT8ubWVzc2FnZT8uY29udGVudCA/PyAnJzsKICAgICAgICBjb25zdCByZWFzb25pbmdDb250ZW50ID0gZGF0YT8uY2hvaWNlcz8uWzBdPy5tZXNzYWdlPy5yZWFzb25pbmdfY29udGVudDsKCiAgICAgICAgaWYgKHJlYXNvbmluZ0NvbnRlbnQpIHsKICAgICAgICAgICAgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgcmVhc29uaW5nQ29udGVudCk7CiAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gcmVhc29uaW5nQ29udGVudDsKICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQudHJpbSgpOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIC8vIEZhbGxiYWNrIGF1ZiA8dGhpbms+LVRhZy1WZXJhcmJlaXR1bmcKICAgICAgICAgICAgY29uc3QgdGhpbmtNYXRjaCA9IGZ1bGxDb250ZW50Lm1hdGNoKC88dGhpbms+KFtcc1xTXSo/KTxcL3RoaW5rPi8pOwogICAgICAgICAgICBpZiAodGhpbmtNYXRjaCkgewogICAgICAgICAgICAgICAgY29uc3QgdGhpbmtUZXh0ID0gdGhpbmtNYXRjaFsxXS50cmltKCk7CiAgICAgICAgICAgICAgICB0aGlua0VsID0gYWRkVGhpbmtCdWJibGUob3V0RWwpOwogICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtUZXh0KTsKICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gdGhpbmtUZXh0OwogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQucmVwbGFjZSh0aGlua01hdGNoWzBdLCAnJykudHJpbSgpOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQ7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgCiAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudChvdXRFbCwgJ2Fzc2lzdGFudCcsIGFzc2lzdGFudEJ1ZmZlcik7CiAgICAgICAgCiAgICAgICAgZmluYWxDb21wbGV0aW9uVG9rZW5zID0gZGF0YT8udXNhZ2U/LmNvbXBsZXRpb25fdG9rZW5zIHx8IDA7CiAgICAgICAgdXBkYXRlVG9rZW5SYXRlKGFzc2lzdGFudEJ1ZmZlciArIHRoaW5rQnVmZmVyLCB0cnVlKTsKICAgICAgICAKICAgICAgICBtYXliZVNjcm9sbCgpOwogICAgICB9IGVsc2UgewogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBTVEFSVCAtLS0KICAgICAgICBsZXQgcmVuZGVyU2NoZWR1bGVkID0gZmFsc2U7CiAgICAgICAgbGV0IGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgbGV0IHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICBjb25zdCBSRU5ERVJfSU5URVJWQUwgPSAxNTA7IAoKICAgICAgICBjb25zdCBzY2hlZHVsZVJlbmRlciA9ICgpID0+IHsKICAgICAgICAgICAgaWYgKHJlbmRlclNjaGVkdWxlZCkgcmV0dXJuOwogICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSB0cnVlOwogICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgICAgICAgICAgIGlmIChhc3Npc3RhbnROZWVkc1JlbmRlcikgewogICAgICAgICAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQob3V0RWwsICdhc3Npc3RhbnQnLCBhc3Npc3RhbnRCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICBpZiAodGhpbmtOZWVkc1JlbmRlciAmJiB0aGlua0VsKSB7CiAgICAgICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSBmYWxzZTsKICAgICAgICAgICAgfSwgUkVOREVSX0lOVEVSVkFMKTsKICAgICAgICB9OwogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBFTkRFIC0tLQoKICAgICAgICBjb25zdCByZWFkZXIgPSByZXMuYm9keS5nZXRSZWFkZXIoKTsgY29uc3QgZGVjb2RlciA9IG5ldyBUZXh0RGVjb2RlcigndXRmLTgnKTsgbGV0IGJ1ZmZlcj0nJzsgbGV0IGRvbmU9ZmFsc2U7CiAgICAgICAgd2hpbGUgKCFkb25lKSB7CiAgICAgICAgICBjb25zdCB7IHZhbHVlLCBkb25lOiByZERvbmUgfSA9IGF3YWl0IHJlYWRlci5yZWFkKCk7IGRvbmUgPSByZERvbmU7CgogICAgICAgICAgaWYgKHN0cmVhbVN0YXJ0VGltZSA9PT0gMCAmJiB2YWx1ZSkgewogICAgICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSBwZXJmb3JtYW5jZS5ub3coKTsKICAgICAgICAgIH0KCiAgICAgICAgICBidWZmZXIgKz0gZGVjb2Rlci5kZWNvZGUodmFsdWUgfHwgbmV3IFVpbnQ4QXJyYXkoKSwgeyBzdHJlYW06ICFkb25lIH0pOwogICAgICAgICAgbGV0IGV2ZW50cyA9IGJ1ZmZlci5zcGxpdCgnXG5cbicpOyBidWZmZXIgPSBldmVudHMucG9wKCkgfHwgJyc7CiAgICAgICAgICBmb3IgKGNvbnN0IGV2IG9mIGV2ZW50cykgewogICAgICAgICAgICBjb25zdCBsaW5lcyA9IGV2LnNwbGl0KCdcbicpLmZpbHRlcihCb29sZWFuKTsKICAgICAgICAgICAgZm9yIChjb25zdCBsaW5lIG9mIGxpbmVzKSB7CiAgICAgICAgICAgICAgaWYgKCFsaW5lLnN0YXJ0c1dpdGgoJ2RhdGE6JykpIGNvbnRpbnVlOwogICAgICAgICAgICAgIGNvbnN0IGRhdGFTdHIgPSBsaW5lLnNsaWNlKDUpLnRyaW0oKTsgaWYgKGRhdGFTdHIgPT09ICdbRE9ORV0nKSB7IGRvbmUgPSB0cnVlOyBicmVhazsgfQogICAgICAgICAgICAgIHRyeSB7CiAgICAgICAgICAgICAgICBjb25zdCBvYmogPSBKU09OLnBhcnNlKGRhdGFTdHIpOwogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBpZiAob2JqPy51c2FnZT8uY29tcGxldGlvbl90b2tlbnMpIHsKICAgICAgICAgICAgICAgICAgICBmaW5hbENvbXBsZXRpb25Ub2tlbnMgPSBvYmoudXNhZ2UuY29tcGxldGlvbl90b2tlbnM7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGRlbHRhID0gb2JqPy5jaG9pY2VzPy5bMF0/LmRlbHRhOwogICAgICAgICAgICAgICAgaWYgKCFkZWx0YSkgY29udGludWU7CgogICAgICAgICAgICAgICAgbGV0IGhhc1VwZGF0ZWQgPSBmYWxzZTsKICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgY29uc3QgcmVhc29uaW5nQ2h1bmsgPSBkZWx0YS5yZWFzb25pbmcgfHwgZGVsdGEucmVhc29uaW5nX2NvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChyZWFzb25pbmdDaHVuaykgewogICAgICAgICAgICAgICAgICAgIGlmICghdGhpbmtFbCkgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZWFzb25pbmdDaHVuazsKICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGNvbnRlbnRDaHVuayA9IGRlbHRhLmNvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICBpZiAoY29udGVudENodW5rKSB7CiAgICAgICAgICAgICAgICAgICAgbGV0IHRlbXBDaHVuayA9IGNvbnRlbnRDaHVuazsKICAgICAgICAgICAgICAgICAgICB3aGlsZSAodGVtcENodW5rLmxlbmd0aCA+IDApIHsKICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluVGhpbmtCbG9jaykgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zID0gdGVtcENodW5rLmluZGV4T2YoJzwvdGhpbms+Jyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyICs9IHRlbXBDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZW1wQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKGVuZFBvcyArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IHN0YXJ0UG9zID0gdGVtcENodW5rLmluZGV4T2YoJzx0aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChzdGFydFBvcyAhPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rLnN1YnN0cmluZygwLCBzdGFydFBvcyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXNzaXN0YW50TmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoIXRoaW5rRWwpIHRoaW5rRWwgPSBhZGRUaGlua0J1YmJsZShvdXRFbCk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgcmVzdE9mQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKHN0YXJ0UG9zICsgNyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zSW5SZXN0ID0gcmVzdE9mQ2h1bmsuaW5kZXhPZignPC90aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zSW5SZXN0ICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZXN0T2ZDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zSW5SZXN0KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gcmVzdE9mQ2h1bmsuc3Vic3RyaW5nKGVuZFBvc0luUmVzdCArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gcmVzdE9mQ2h1bms7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpblRoaW5rQmxvY2sgPSB0cnVlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gJyc7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChoYXNVcGRhdGVkKSB7CiAgICAgICAgICAgICAgICAgIHNjaGVkdWxlUmVuZGVyKCk7CiAgICAgICAgICAgICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgIH0gY2F0Y2gge30KICAgICAgICAgICAgfQogICAgICAgICAgfQogICAgICAgIH0KICAgICAgICAvLyBGaW5hbGVzIFJlbmRlcmluZyBhbSBFbmRlIGRlcyBTdHJlYW1zLCB1bSBzaWNoZXJ6dXN0ZWxsZW4sIGRhc3MgYWxsZXMgYW5nZXplaWd0IHdpcmQKICAgICAgICByZW5kZXJCdWJibGVDb250ZW50KG91dEVsLCAnYXNzaXN0YW50JywgYXNzaXN0YW50QnVmZmVyKTsKICAgICAgICBpZiAodGhpbmtFbCkgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgIAogICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIsIHRydWUpOwogICAgICAgIAogICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgIH0KICAgICAgCiAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGFzc2lzdGFudEJ1ZmZlci50cmltKCk7CiAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyIHx8ICcoZW1wdHkgcmVzcG9uc2UpJyB9KTsKICAgICAgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgaWYgKGVycj8ubmFtZSA9PT0gJ0Fib3J0RXJyb3InKSB7CiAgICAgICAgLy8gU3RvcOODnOOCv+ODszog6YCU5Lit44G+44Gn44Gu5Zue562U44KS5bGl5q2044Gr5q6L44GZCiAgICAgICAgaWYgKGFzc2lzdGFudEJ1ZmZlci50cmltKCkpIHsKICAgICAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyLnRyaW0oKSB9KTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgLy8g6YCa5L+h44Ko44Op44O8OiDnm7TliY3jga51c2Vy44Oh44OD44K744O844K444KS5bGl5q2044GL44KJ5oi744GZCiAgICAgICAgaWYgKG91dEVsICYmICFvdXRFbC50ZXh0Q29udGVudC50cmltKCkpIHsgb3V0RWwuY2xvc2VzdCgnLm1zZycpPy5yZW1vdmUoKTsgfQogICAgICAgIGlmIChtZXNzYWdlcy5sZW5ndGggPiAwICYmIG1lc3NhZ2VzW21lc3NhZ2VzLmxlbmd0aC0xXS5yb2xlID09PSAndXNlcicpIHsKICAgICAgICAgIG1lc3NhZ2VzLnBvcCgpOwogICAgICAgIH0KICAgICAgICAvLyDlhaXlipvjg4bjgq3jgrnjg4jjgpLlvqnlhYPvvIjmt7vku5jmmYLjga9jb250ZW5044GM6YWN5YiX44Gr44Gq44KL44Gf44KBc2F2ZWRUZXh044KS5L2/44GG77yJCiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gc2F2ZWRUZXh0OwogICAgICAgIC8vIOa3u+S7mOODleOCoeOCpOODq+OCguW+qeWFgwogICAgICAgIGlmIChzYXZlZEF0dGFjaG1lbnRzLmxlbmd0aCA+IDApIHsKICAgICAgICAgIGF0dGFjaG1lbnRzID0gc2F2ZWRBdHRhY2htZW50czsKICAgICAgICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgICAgICAgfQogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb44Ko44Op44O8XSAke2Vyci5tZXNzYWdlfWApOyBjb25zb2xlLmVycm9yKGVycik7CiAgICAgIH0KICAgIH0gZmluYWxseSB7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKGZhbHNlKTsgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICAgIH0KICB9CgogIGZ1bmN0aW9uIHN0b3BTdHJlYW0oKXsgaWYgKGFib3J0Q29udHJvbGxlcikgYWJvcnRDb250cm9sbGVyLmFib3J0KCk7IH0KICBmdW5jdGlvbiBjbGVhckNoYXQoKXsgbWVzc2FnZXMgPSBbXTsgZWxzLmxvZy5pbm5lckhUTUwgPSAnJzsgZWxzLnRva2VuUmF0ZS50ZXh0Q29udGVudCA9ICcnOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgbWF5YmVTY3JvbGwodHJ1ZSk7IH0KCiAgLyogLS0tLS0tLS0tLSBJbXBvcnQgLyBFeHBvcnQgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGV4cG9ydENoYXQoKXsKICAgIGNvbnN0IG91dCA9IGJ1aWxkRXhwb3J0QXJyYXkoKTsKICAgIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbSlNPTi5zdHJpbmdpZnkob3V0LCBudWxsLCAyKV0sIHsgdHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nIH0pOwogICAgY29uc3QgdXJsID0gVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTsKICAgIGNvbnN0IGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdhJyk7IGNvbnN0IHRzID0gbmV3IERhdGUoKTsgY29uc3QgcGFkID0gbj0+U3RyaW5nKG4pLnBhZFN0YXJ0KDIsJzAnKTsKICAgIGEuaHJlZiA9IHVybDsgYS5kb3dubG9hZCA9IGBjaGF0XyR7dHMuZ2V0RnVsbFllYXIoKX0tJHtwYWQodHMuZ2V0TW9udGgoKSsxKX0tJHtwYWQodHMuZ2V0RGF0ZSgpKX1fJHtwYWQodHMuZ2V0SG91cnMoKSl9LSR7cGFkKHRzLmdldE1pbnV0ZXMoKSl9LSR7cGFkKHRzLmdldFNlY29uZHMoKSl9Lmpzb25gOwogICAgYS5jbGljaygpOyBzZXRUaW1lb3V0KCgpPT5VUkwucmV2b2tlT2JqZWN0VVJMKHVybCksIDEwMDApOwogIH0KCiAgYXN5bmMgZnVuY3Rpb24gcGFyc2VJbXBvcnQodGV4dCl7CiAgICBjb25zdCBjbGVhbmVkID0gdGV4dC5yZXBsYWNlKC9cci9nLCcnKTsKICAgIGxldCByYXc7IHRyeSB7IHJhdyA9IEpTT04ucGFyc2UoY2xlYW5lZCk7IH0KICAgIGNhdGNoIHsgY29uc3QgbGluZXMgPSBjbGVhbmVkLnNwbGl0KCdcbicpLm1hcChzPT5zLnRyaW0oKSkuZmlsdGVyKEJvb2xlYW4pOyBjb25zdCBhcnI9W107IGZvciAoY29uc3QgbGluZSBvZiBsaW5lcyl7IHRyeXsgYXJyLnB1c2goSlNPTi5wYXJzZShsaW5lKSk7IH1jYXRjaHt9IH0gcmF3ID0gYXJyLmxlbmd0aCA/IGFyciA6IG51bGw7IH0KICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0ZpbGUgbm90IHJlYWRhYmxlIChuZWl0aGVyIEpTT04gbm9yIEpTT05MKS4nKTsKCiAgICBsZXQgYXJyID0gW10sIHRvcFN5c3RlbSA9ICcnOwogICAgaWYgKEFycmF5LmlzQXJyYXkocmF3KSkgYXJyID0gcmF3OwogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcubWVzc2FnZXMpKSB7IGFyciA9IHJhdy5tZXNzYWdlczsgdG9wU3lzdGVtID0gU3RyaW5nKHJhdy5zeXN0ZW0gfHwgJycpLnRyaW0oKTsgfQogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcuY29udmVyc2F0aW9uKSkgeyBhcnIgPSByYXcuY29udmVyc2F0aW9uOyB0b3BTeXN0ZW0gPSBTdHJpbmcocmF3LnN5c3RlbSB8fCAnJykudHJpbSgpOyB9CiAgICBlbHNlIHRocm93IG5ldyBFcnJvcignTm8gbWVzc2FnZSBhcnJheSBmb3VuZCAoZXhwZWN0ZWQ6IFsge3JvbGUsIGNvbnRlbnR9LCAuLi4gXSkuJyk7CgogICAgY29uc3Qgbm9ybWFsaXplZCA9IGFyci5tYXAobSA9PiAoeyByb2xlOiBub3JtYWxpemVSb2xlKG0ucm9sZSksIGNvbnRlbnQ6IG5vcm1hbGl6ZUNvbnRlbnQobS5jb250ZW50KSB9KSkKICAgICAgICAgICAgICAgICAgICAgICAgICAuZmlsdGVyKG0gPT4gWydzeXN0ZW0nLCd1c2VyJywnYXNzaXN0YW50J10uaW5jbHVkZXMobS5yb2xlKSAmJiB0eXBlb2YgbS5jb250ZW50ID09PSAnc3RyaW5nJyk7CgogICAgY29uc3QgaGFzU3lzdGVtID0gbm9ybWFsaXplZC5zb21lKG0gPT4gbS5yb2xlID09PSAnc3lzdGVtJyk7CiAgICBpZiAoIWhhc1N5c3RlbSAmJiB0b3BTeXN0ZW0pIG5vcm1hbGl6ZWQudW5zaGlmdCh7IHJvbGU6J3N5c3RlbScsIGNvbnRlbnQ6IHRvcFN5c3RlbSB9KTsKICAgIGlmICghbm9ybWFsaXplZC5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gdmFsaWQgbWVzc2FnZXMgZm91bmQgaW4gaW1wb3J0LicpOwogICAgcmV0dXJuIG5vcm1hbGl6ZWQ7CiAgfQoKICBmdW5jdGlvbiBoYW5kbGVGaWxlKGZpbGUpewogICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgIHJlYWRlci5vbmxvYWQgPSAoKSA9PiB7CiAgICAgIHBhcnNlSW1wb3J0KFN0cmluZyhyZWFkZXIucmVzdWx0IHx8ICcnKSkKICAgICAgICAudGhlbihhcnIgPT4geyBtZXNzYWdlcyA9IGFycjsgY29uc3Qgc3lzID0gbWVzc2FnZXMuZmluZChtPT5tLnJvbGU9PT0nc3lzdGVtJyk7IGVscy5zeXN0ZW0udmFsdWUgPSBzeXM/c3lzLmNvbnRlbnQ6Jyc7IHJlbmRlckFsbCgpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBg44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GX44G+44GX44GfICgke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXMpLmApOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgfSkKICAgICAgICAuY2F0Y2goZSA9PiBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW+OCpOODs+ODneODvOODiOOCqOODqeODvF0gJHtlLm1lc3NhZ2V9YCkpOwogICAgfTsKICAgIHJlYWRlci5vbmVycm9yID0gKCkgPT4gYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjgqTjg7Pjg53jg7zjg4jjgqjjg6njg7xdIENvdWxkIG5vdCByZWFkIGZpbGUuYCk7CiAgICByZWFkZXIucmVhZEFzVGV4dChmaWxlKTsKICB9CiAgaWYgKGVscy5kcm9wKSB7CiAgICBbJ2RyYWdlbnRlcicsJ2RyYWdvdmVyJ10uZm9yRWFjaChldiA9PiBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKGV2LCBlID0+IHsgZS5wcmV2ZW50RGVmYXVsdCgpOyBlbHMuZHJvcC5jbGFzc0xpc3QuYWRkKCdkcmFnJyk7IH0pKTsKICAgIFsnZHJhZ2xlYXZlJywnZHJvcCddLmZvckVhY2goZXYgPT4gZWxzLmRyb3AuYWRkRXZlbnRMaXN0ZW5lcihldiwgZSA9PiB7IGUucHJldmVudERlZmF1bHQoKTsgZWxzLmRyb3AuY2xhc3NMaXN0LnJlbW92ZSgnZHJhZycpOyB9KSk7CiAgICBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKCdkcm9wJywgKGUpID0+IHsgY29uc3QgZiA9IGUuZGF0YVRyYW5zZmVyPy5maWxlcz8uWzBdOyBpZiAoZikgaGFuZGxlRmlsZShmKTsgfSk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFJBVyBKU09OIEVkaXRvciAtLS0tLS0tLS0tICovCiAgZnVuY3Rpb24gZWRpdG9yU2V0U3RhdHVzKG1zZywgb2s9dHJ1ZSl7IGVscy5qc29uU3RhdHVzLnRleHRDb250ZW50ID0gbXNnIHx8ICcnOyBlbHMuanNvbkVkaXRvci5zdHlsZS5ib3JkZXJDb2xvciA9IG9rID8gJ3ZhcigtLWxpbmUpJyA6ICcjYjQwMDAwJzsgfQogIGZ1bmN0aW9uIGVkaXRvckR1bXBGcm9tQ2hhdCgpeyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KGJ1aWxkRXhwb3J0QXJyYXkoKSwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnbG9hZGVkIGZyb20gY2hhdCcpOyB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUGFyc2VDdXJyZW50KCl7CiAgICBjb25zdCB0ZXh0ID0gZWxzLmpzb25FZGl0b3IudmFsdWUudHJpbSgpIHx8ICdbXSc7CiAgICBjb25zdCBhcnIgPSBhd2FpdCBwYXJzZUltcG9ydCh0ZXh0KTsKICAgIGNvbnN0IGZpcnN0U3lzID0gYXJyLmZpbmQobT0+bS5yb2xlPT09J3N5c3RlbScpOyBpZiAoZmlyc3RTeXMpIGVscy5zeXN0ZW0udmFsdWUgPSBmaXJzdFN5cy5jb250ZW50OwogICAgcmV0dXJuIGFycjsKICB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUmVwbGFjZSgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IGFycjsKICAgICAgcmVuZGVyQWxsKCk7CiAgICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICAgIGF1dG9zY3JvbGwgPSB0cnVlOwogICAgICBlZGl0b3JTZXRTdGF0dXMoYGFwcGxpZWQgKHJlcGxhY2UpIOKAlCAke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXNgKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYEpTT07jgYvjgonjg4Hjg6Pjg4Pjg4jjgpLoqK3lrprjgZfjgb7jgZfjgZ8gKCR7bWVzc2FnZXMubGVuZ3RofSlgKTsKICAgIH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cyhlLm1lc3NhZ2UsZmFsc2UpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW0pTT07jgqjjg6njg7xdICR7ZS5tZXNzYWdlfWApOyB9CiAgfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvckFwcGVuZCgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IG1lc3NhZ2VzLmNvbmNhdChhcnIpOwogICAgICByZW5kZXJBbGwoKTsKICAgICAgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICAgICAgYXV0b3Njcm9sbCA9IHRydWU7CiAgICAgIGVkaXRvclNldFN0YXR1cyhgYXBwbGllZCAoYXBwZW5kKSDigJQgbm93ICR7bWVzc2FnZXMubGVuZ3RofSBtZXNzYWdlc2ApOwogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgSlNPTuOBi+OCieODgeODo+ODg+ODiOOCkui/veiomOOBl+OBvuOBl+OBnyAoKyR7YXJyLmxlbmd0aH0pYCk7CiAgICB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLGZhbHNlKTsgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFtKU09O44Ko44Op44O8XSAke2UubWVzc2FnZX1gKTsgfQogIH0KICBmdW5jdGlvbiBlZGl0b3JCZWF1dGlmeSgpeyB0cnkgeyBjb25zdCBvYmogPSBKU09OLnBhcnNlKGVscy5qc29uRWRpdG9yLnZhbHVlIHx8ICdbXScpOyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KG9iaiwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnYmVhdXRpZmllZCcpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoJ2ludmFsaWQgSlNPTicsIGZhbHNlKTt9IH0KICBmdW5jdGlvbiBlZGl0b3JNaW5pZnkoKXsgdHJ5IHsgY29uc3Qgb2JqID0gSlNPTi5wYXJzZShlbHMuanNvbkVkaXRvci52YWx1ZSB8fCAnW10nKTsgZWxzLmpzb25FZGl0b3IudmFsdWUgPSBKU09OLnN0cmluZ2lmeShvYmopOyBlZGl0b3JTZXRTdGF0dXMoJ21pbmlmaWVkJyk7IH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cygnaW52YWxpZCBKU09OJywgZmFsc2UpO30gfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvclZhbGlkYXRlKCl7IHRyeSB7IGF3YWl0IGVkaXRvclBhcnNlQ3VycmVudCgpOyBlZGl0b3JTZXRTdGF0dXMoJ3ZhbGlkIOKckycpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLCBmYWxzZSk7IH0gfQoKICBlbHMucmF3Qm94LmFkZEV2ZW50TGlzdGVuZXIoJ3RvZ2dsZScsICgpID0+IHsgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7IH0pOwogIGVscy5qc29uUmVmcmVzaC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckR1bXBGcm9tQ2hhdCk7CiAgZWxzLmpzb25BcHBseS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclJlcGxhY2UpOwogIGVscy5qc29uQXBwZW5kLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgZWRpdG9yQXBwZW5kKTsKICBlbHMuanNvblByZXR0eS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckJlYXV0aWZ5KTsKICBlbHMuanNvbk1pbmlmeS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvck1pbmlmeSk7CiAgZWxzLmpzb25WYWxpZGF0ZS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclZhbGlkYXRlKTsKCiAgLyogLS0tLS0tLS0tLSBXaXJlIFVJIC0tLS0tLS0tLS0gKi8KICBlbHMuc2VuZEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHNlbmRNZXNzYWdlKTsKICBlbHMuc3RvcEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHN0b3BTdHJlYW0pOwogIGVscy5jbGVhckJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGNsZWFyQ2hhdCk7CiAgaWYgKGVscy5hdHRhY2hCdG4gJiYgZWxzLmF0dGFjaElucHV0KSB7CiAgICBlbHMuYXR0YWNoQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCk9PiBlbHMuYXR0YWNoSW5wdXQuY2xpY2soKSk7CiAgICBlbHMuYXR0YWNoSW5wdXQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKCk9PnsKICAgICAgYWRkQXR0YWNobWVudHNGcm9tRmlsZUxpc3QoZWxzLmF0dGFjaElucHV0LmZpbGVzKTsKICAgICAgZWxzLmF0dGFjaElucHV0LnZhbHVlID0gJyc7CiAgICB9KTsKICB9CiAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGxvYWRNb2RlbHMpOwogIGVscy5pbnB1dC5hZGRFdmVudExpc3RlbmVyKCdrZXlkb3duJywgKGUpID0+IHsKICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJyAmJiAhZS5zaGlmdEtleSAmJiAhZS5pc0NvbXBvc2luZyAmJiBlLmtleUNvZGUgIT09IDIyOSkgewogICAgICAvLyDjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK3jga/pgIHkv6HjgZvjgZrjgIHmlLnooYzjgpLoqLHlj68KICAgICAgaWYgKCFlbHMuc3RvcEJ0bi5kaXNhYmxlZCkgcmV0dXJuOwogICAgICBlLnByZXZlbnREZWZhdWx0KCk7CiAgICAgIHNlbmRNZXNzYWdlKCk7CiAgICB9CiAgfSk7CiAgZWxzLnRlbXAuYWRkRXZlbnRMaXN0ZW5lcignaW5wdXQnLCAoKSA9PiB7IGVscy50ZW1wVmFsLnRleHRDb250ZW50ID0gcGFyc2VGbG9hdChlbHMudGVtcC52YWx1ZSkudG9GaXhlZCgxKTsgfSk7CiAgZWxzLnNldHRpbmdzVG9nZ2xlLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgZWxzLnNldHRpbmdzUGFuZWwuY2xhc3NMaXN0LnRvZ2dsZSgnaGlkZGVuJyk7CiAgfSk7CgogIC8qIC0tLS0tLS0tLS0gSW5mbyBkaWFsb2cgLS0tLS0tLS0tLSAqLwogIGlmIChlbHMuaW5mb1ZlcnNpb24pIGVscy5pbmZvVmVyc2lvbi50ZXh0Q29udGVudCA9IEFQUF9WRVJTSU9OOwogIGlmIChlbHMuaW5mb1Byb2plY3RVcmwpIHsKICAgIGVscy5pbmZvUHJvamVjdFVybC50ZXh0Q29udGVudCA9IFBST0pFQ1RfVVJMOwogICAgZWxzLmluZm9Qcm9qZWN0VXJsLmhyZWYgPSBQUk9KRUNUX1VSTDsKICB9CiAgaWYgKGVscy5pbmZvSW1wcmludCkgewogICAgZWxzLmluZm9JbXByaW50LnRleHRDb250ZW50ID0gSU1QUklOVF9VUkw7CiAgICBlbHMuaW5mb0ltcHJpbnQuaHJlZiA9IElNUFJJTlRfVVJMOwogIH0KICBpZiAoZWxzLmluZm9CdG4gJiYgZWxzLmluZm9EaWFsb2cpIHsKICAgIGVscy5pbmZvQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgICBpZiAodHlwZW9mIGVscy5pbmZvRGlhbG9nLnNob3dNb2RhbCA9PT0gJ2Z1bmN0aW9uJykgZWxzLmluZm9EaWFsb2cuc2hvd01vZGFsKCk7CiAgICAgIGVsc2UgZWxzLmluZm9EaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgfSk7CiAgICAvLyBDbGljayBvdXRzaWRlIHRoZSBjYXJkIGNsb3NlcyB0aGUgZGlhbG9nCiAgICBlbHMuaW5mb0RpYWxvZy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIChlKSA9PiB7CiAgICAgIC8vIElmIGNsaWNrIGxhbmRzIG9uIHRoZSBiYWNrZHJvcCAobm90IG9uIHRoZSBjYXJkKSwgY2xvc2UuCiAgICAgIGlmIChlLnRhcmdldCA9PT0gZWxzLmluZm9EaWFsb2cpIGVscy5pbmZvRGlhbG9nLmNsb3NlKCk7CiAgICB9KTsKICB9CgogIC8qIC0tLS0tLS0tLS0gQVBJIOiqjeiovOODgOOCpOOCouODreOCsCAtLS0tLS0tLS0tICovCiAgaWYgKGVscy5hdXRoQnRuICYmIGVscy5hdXRoRGlhbG9nKSB7CiAgICBlbHMuYXV0aEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IG9wZW5BdXRoRGlhbG9nKCkpOwogIH0KICBpZiAoZWxzLmF1dGhTYXZlKSB7CiAgICBlbHMuYXV0aFNhdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7CiAgICAgIGNvbnN0IHVzZXIgPSAoZWxzLmF1dGhVc2VyPy52YWx1ZSB8fCAnJykudHJpbSgpIHx8ICdleGwzJzsKICAgICAgY29uc3QgcGFzcyA9IGVscy5hdXRoUGFzcz8udmFsdWUgfHwgJyc7CiAgICAgIGlmICghcGFzcykgewogICAgICAgIGlmIChlbHMuYXV0aFN0YXR1cykgZWxzLmF1dGhTdGF0dXMudGV4dENvbnRlbnQgPSAn44OR44K544Ov44O844OJ44KS5YWl5Yqb44GX44Gm44GP44Gg44GV44GEJzsKICAgICAgICByZXR1cm47CiAgICAgIH0KICAgICAgc2V0QXV0aENyZWRlbnRpYWxzKHVzZXIsIHBhc3MpOwogICAgICBjbG9zZUF1dGhEaWFsb2coKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvoqo3oqLxdICR7dXNlcn0g44Gn5L+d5a2Y44GX44G+44GX44Gf77yI44GT44Gu44K/44OW44Gu44G/77yJYCk7CiAgICAgIGxvYWRNb2RlbHMoKTsKICAgICAgc3RhcnRRdW90YVBvbGxpbmcoKTsKICAgICAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhDbGVhcikgewogICAgZWxzLmF1dGhDbGVhci5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsKICAgICAgY2xlYXJBdXRoQ3JlZGVudGlhbHMoKTsKICAgICAgY2xvc2VBdXRoRGlhbG9nKCk7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsICdb6KqN6Ki8XSDos4fmoLzmg4XloLHjgpLop6PpmaTjgZfjgb7jgZfjgZ8nKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhQYXNzKSB7CiAgICBlbHMuYXV0aFBhc3MuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIChlKSA9PiB7CiAgICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJykgeyBlLnByZXZlbnREZWZhdWx0KCk7IGVscy5hdXRoU2F2ZT8uY2xpY2soKTsgfQogICAgfSk7CiAgfQogIGlmIChlbHMuYXV0aERpYWxvZykgewogICAgZWxzLmF1dGhEaWFsb2cuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4gewogICAgICBpZiAoZS50YXJnZXQgPT09IGVscy5hdXRoRGlhbG9nKSBjbG9zZUF1dGhEaWFsb2coKTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSDmrovjgormmYLplpMgLyDlgZzmraLjg5zjgr/jg7MgLS0tLS0tLS0tLSAqLwogIHN0YXJ0UXVvdGFQb2xsaW5nKCk7CiAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgIGVscy5zdG9wU2VydmVyLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgc3RvcFNlcnZlcik7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFdlbGNvbWUgLS0tLS0tLS0tLSAqLwogIHNldFRpbWVvdXQoKCkgPT4geyBsb2FkTW9kZWxzKCk7IH0sIDUwMCk7Cn0pKCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

# pip に渡す vLLM 指定（切り戻したいときは "vllm==0.10.2" 等に変える）
VLLM_PIP_SPEC = os.environ.get("VLLM_PIP_SPEC", "vllm==0.32.0")


_gpu_names, _vram_each = kc_detect_gpu()
VRAM_TOTAL = sum(_vram_each)
print(f"GPU      : {_gpu_names}")
print(f"VRAM     : {_vram_each} MiB each  →  合計 {VRAM_TOTAL} MiB")

# /dev/shm は TP>1 の必須資源。vLLM のリングバッファは既定で
# (24 MiB + 数 B) x 10 ブロック ≈ 240 MiB を 1 本で確保する。
# 足りなければ即 RuntimeError になるが、**境界値だと起動後に詰まる**（実測:
# 32K ctx で "No available shared memory broadcast block found" が 60s 間隔で連発）。
def _shm_required_mib(tp):
    """/dev/shm の必要量（MiB）の目安。TP 本ぶん + 2 倍の余裕。

    1 本あたり 24 MiB x 10 ブロック ≒ 240 MiB。TP=2 なら 480 MiB、
    余裕を 2 倍見て 960 MiB。**空き**で判定する（total では足りない。
    他プロセスの残骸が掴んでいる場合がある）。
    """
    return 240 * max(1, int(tp)) * 2


SHM_FREE_MIB = SHM_TOTAL_MIB = 0
try:
    _shm = os.statvfs("/dev/shm")
    SHM_FREE_MIB = _shm.f_bavail * _shm.f_frsize // (1 << 20)
    SHM_TOTAL_MIB = _shm.f_blocks * _shm.f_frsize // (1 << 20)
    print(f"/dev/shm : {SHM_TOTAL_MIB} MiB（空き {SHM_FREE_MIB} MiB）"
          f"  cpu={os.cpu_count()}")
except Exception as _e:
    print(f"/dev/shm : 取得失敗 {_e!r}")

if VRAM_TOTAL == 0:
    raise RuntimeError(
        "GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")
if len(_vram_each) < 2:
    print("⚠ GPU が 1 枚です。Tensor Parallel は 1 に落とします"
          "（27B AWQ は 1 枚に収まらない可能性が高い）")
TP_SIZE = 2 if len(_vram_each) >= 2 else 1

# 必要量は TP が決まってから判定する（TP に比例するため）。
SHM_REQUIRED_MIB = _shm_required_mib(TP_SIZE)
if SHM_TOTAL_MIB and SHM_FREE_MIB < SHM_REQUIRED_MIB:
    print(f"⚠ /dev/shm の空き {SHM_FREE_MIB} MiB < 必要 {SHM_REQUIRED_MIB} MiB"
          f"（TP={TP_SIZE}）→ 拡張を試みる")
    try:
        _rm = subprocess.run(
            ["mount", "-o", "remount,size=8G", "/dev/shm"],
            capture_output=True, text=True, timeout=60)
        print(f"   remount rc={_rm.returncode} {( _rm.stderr or '').strip()[:120]}")
    except Exception as _e:
        print(f"   remount 失敗: {_e!r}")
    try:
        _shm = os.statvfs("/dev/shm")
        SHM_FREE_MIB = _shm.f_bavail * _shm.f_frsize // (1 << 20)
        SHM_TOTAL_MIB = _shm.f_blocks * _shm.f_frsize // (1 << 20)
        print(f"/dev/shm : 再測定 {SHM_TOTAL_MIB} MiB（空き {SHM_FREE_MIB} MiB）")
    except Exception as _e:
        print(f"/dev/shm : 再測定失敗 {_e!r}")
    if SHM_FREE_MIB < SHM_REQUIRED_MIB:
        # ここで落とす。放置すると起動はするが forward で詰まり、
        # 待機側が 40 分待つ（1 往復を丸損する）。
        raise RuntimeError(
            f"/dev/shm が不足しています（空き {SHM_FREE_MIB} MiB / 必要 "
            f"{SHM_REQUIRED_MIB} MiB, TP={TP_SIZE}）。TP=1 に落とすか、"
            "max-model-len を下げてください。")

# ---- 起動条件 ----
# 27B AWQ は約 19 GB。T4 x2 = 30 GB のうち 0.90 = 27 GB を使う想定で、
# 残り ~8 GB が KV cache になる。実際に収まるかは config 次第なので、
# 起動失敗時に max-model-len を落として再試行する（serve セル）。
# 2026-10-29 机上計算: 重み 18.20 GiB + KV f16 68 KiB/token。
# 64K=22.45 GiB（予算 27.0 で余裕 4.5）、96K=24.58（余裕 2.4）、
# 128K=26.70（余裕 0.3 = graph/activation で確実に溢れる）。
# T4 は fp8 KV 不可なので Ollama の q8_0（34 KiB/token）のようには詰められない。
# → 上限狙いは 64K。実測で 1〜2 GiB ずれるので 32K/16K も残す。
MAXLEN_CANDIDATES = [65536, 32768, 16384, 8192]
MAX_MODEL_LEN = MAXLEN_CANDIDATES[0]
# 共有セル（接続情報）が読む。起動成功後は実際の max_model_len で上書きする。
CONTEXT_LENGTH = MAX_MODEL_LEN
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS = 2
KV_CACHE_DTYPE = "auto"        # T4 は FP8 非対応。fp8 を指定してはいけない

# MTP（multi-token prediction）は任意。失敗したら自動で無効にして再起動する。
MTP_ENABLED = os.environ.get("VLLM_MTP", "1") != "0"
MTP_METHOD = os.environ.get("VLLM_SPEC_METHOD", "mtp")
MTP_TOKENS = int(os.environ.get("VLLM_SPEC_TOKENS", "2"))

# ---- 認証 ----
# 既定で無効。API_PASS が入っているときだけ proxy が有効化する。
PROXY_AUTH = {
    "API_USER": os.environ.get("API_USER", "") or "ollama",
    "API_PASS": os.environ.get("API_PASS", ""),
}

# ---- Secrets の既定値（config.env 由来。Secret があれば上書きされる）----
GIST_ID_DEFAULT = "2b44ac8cb115012a31d3c46661af933f"
GIST_OWNER = "yamahei21python"

# ---- キープアライブ ----
# 上限。チャット 1 回で proxy が MINUTES=60 を書くため、上限が無いと
# 「1 回送信 → 60 分に延長」で残クォータを超えて強制終了される
# （強制終了は成果物の zip が保証されない）。検証中は 30 分で頭打ちにする。
MAX_MINUTES = 30
KEEPALIVE_RESET_MINUTES = 60      # チャット送信ごとにリセットする時間（分）
TUNNEL_CHECK_EVERY = 10           # 公開URL疎通は 10 分ごと

# ---- 死活監視 / 後片付けの能力表（共有セルが読む）----
ENGINE_LOG = "/tmp/vllm_1.log"
ENGINE_PROCS = [("vllm serve", "argv"), ("auth_proxy.py", "argv")]
CLEANUP_TARGETS = [
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f vllm serve", "vllm"),
    ("-f VLLM::EngineCore", "engine-core"),
]


def engine_cleanup():
    """次回起動を短くするため compile / Triton キャッシュを退避する。

    VRAM の解放はプロセス停止（CLEANUP_TARGETS が EngineCore まで落とす）で行う。
    3GB を超える場合は退避しない（転送と容量の無駄）。
    """
    try:
        import tarfile as _tf
        keep = []
        for _d in ("/root/.cache/vllm", "/root/.triton", "/root/.cache/triton"):
            if os.path.isdir(_d) and not any(_d.startswith(k + "/") for k in keep):
                keep.append(_d)
        size = 0
        for _d in keep:
            for _root, _dirs, _files in os.walk(_d):
                for _f in _files:
                    try:
                        size += os.path.getsize(os.path.join(_root, _f))
                    except OSError:
                        pass
        print(f"🧠 キャッシュ {size/1e6:.0f} MB {keep}")
        if keep and size and size < 3e9:
            out = os.path.join(OUT_DIRS[0], "warm-cache.tar")
            with _tf.open(out, "w") as _t:
                for _d in keep:
                    _t.add(_d, arcname=os.path.relpath(_d, "/"))
            print(f"   退避: {out} {os.path.getsize(out)/1e6:.0f} MB")
            return f"cache_tar={os.path.getsize(out)}"
        return "cache_skipped" if size else "cache_none"
    except Exception as _e:
        print(f"⚠ キャッシュ退避をスキップ: {_e}")
        return f"cache_error={type(_e).__name__}"


# ---- 検証成果物の永続化 ----
# `kaggle kernels output` が zip 化するのは /kaggle/working のみ。
OUT_DIRS = KC_OUT_DIRS
save_output = kc_save_output
fail = kc_fail


print()
print("=" * 56)
print(f"  モデル      : {MODEL_SRC}")
print(f"  配信名      : {SERVED_NAME}")
print(f"  TP          : {TP_SIZE}")
print(f"  MTP         : {'ON' if MTP_ENABLED else 'OFF'}")
print(f"  公開ポート  : {PROXY_PORT}（vLLM 本体は {VLLM_PORT}）")
print(f"  認証        : {'ON' if PROXY_AUTH['API_PASS'] else 'OFF（URL を知れば誰でも利用可）'}")
print("=" * 56)

_files = sorted(os.listdir(MODEL_SRC)) if os.path.isdir(MODEL_SRC) else []
print(f"  Dataset 直下: {len(_files)} 件")
for _f in _files[:12]:
    print(f"    - {_f}")
if "config.json" not in _files:
    fail("config.json が Dataset 直下にありません（Add Data を確認）")
if "model.safetensors" not in _files:
    fail("model.safetensors が Dataset 直下にありません（Add Data を確認）")


MODE = "serve"                  # 配信モード（エンジン側の意味付け）
ENGINE = "vllm"
VERSION = "0.32.0"
ALIAS = "qwen3.8-27b-awq"
PORT = 8000                    # 推論サーバ（非公開）
PROXY_PORT = 11435        # 認証プロキシ（公開）
TP = 2
RUN_MINUTES = 60
HARD_MAX_MINUTES = 180
# smoke: トンネル / Gist / keepalive を省略し、疎通だけ見て終了する（GPU 時間の節約）
SMOKE = False
ENGINE_PATTERN = "-f vllm"
TUNNEL = "cloudflared"              # 実際に張るトンネル（cloudflared のみ）
KNOBS = {}
RESULT = "/kaggle/working/result.json"
# セル側が書く計測の置き場。save() が result.json の `timings` に合流させる。
TIMINGS = "/kaggle/working/timings.json"
URL_FILE = "/kaggle/working/url.txt"
WORK = "/tmp/vllm-serve"
CONF = "/tmp/keepalive.conf"
# 配信の証跡。notebook 版（`kaggle/serve-cells/keepalive.py`）と同じ名前で出す。
# これが無いと、外から繋がらなかった回に「いつ URL が死んだか」を
# 成果物から読めない（`notes/serve-parity.md`）。
KEEPALIVE_LOG = "/kaggle/working/keepalive.txt"
CLEANUP_LOG = "/kaggle/working/cleanup.txt"
LOOP_SLEEP = 20                 # ループ周期（秒）
# 公開 URL の疎通は 10 分ごと。notebook 版は 60 秒ループ × 10 周
# （`TUNNEL_CHECK_EVERY = 10`）なので、周期が違うぶんを周回数に直す。
TUNNEL_CHECK_EVERY = 600 // LOOP_SLEEP
GIST_ID_DEFAULT = "2b44ac8cb115012a31d3c46661af933f"
GIST_OWNER = "yamahei21python"
# 公開 URL を置く掲示板カーネルの名前（owner は GIST_OWNER と同じ）。
# カーネルから push できるのは自分のカーネルだけなので、配信カーネルと
# 同じアカウントに置く。中身はビルダーが生成する（kaggle/serve-url-beacon/）。
BEACON_NAME = "serve-url-beacon"
PROXY_B64 = "IyDilIDilIAgcnVudGltZS9xdW90YS5wee+8iGF1dGhfcHJveHkucHkg44Gr5ZCM5qKx77yJCiIiIkthZ2dsZSDjga7jgq/jgqnjg7zjgr8gQVBJ77yIR1BVL1RQVSDmrovph4/vvInjgpLoqq3jgoDlhbHpgJrlh6bnkIbvvIjnlJ/miJDjgrPjg7zjg4njgbjln4vjgoHovrzjgoDvvInjgIIKCmthZ2dsZSBDTEkg44Gr44GvIHF1b3RhIOOCteODluOCs+ODnuODs+ODieOBjOeEoeOBhOOBjOOAgVdlYiBBUEkg44GvIEJhc2ljIOiqjeiovOOBp+iqreOCgeOCi+OAggrlkbzjgbPlh7rjgZflgbTjga8gYXV0aF9wcm94eS5wee+8iC9xdW90Ye+8ieOBqCBrYWdnbGUtcXVvdGEucHnvvIhDTEkg6KGo56S677yJ44CCCiIiIgppbXBvcnQgYmFzZTY0CmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdXJsbGliLnJlcXVlc3QKCktDX1FVT1RBX1VSTCA9ICJodHRwczovL3d3dy5rYWdnbGUuY29tL2FwaS92MS9rZXJuZWxzL3F1b3RhIgpLQ19DUkVEX1BBVEggPSBvcy5wYXRoLmV4cGFuZHVzZXIoIn4vLmthZ2dsZS9rYWdnbGUuanNvbiIpCgoKZGVmIGtjX2thZ2dsZV9jcmVkZW50aWFscyhjcmVkX3BhdGg9Tm9uZSwgZW52aXJvbj1Ob25lKToKICAgICIiIih1c2VybmFtZSwga2V5KSDjgpLov5TjgZnjgIJrYWdnbGUuanNvbiDihpIg55Kw5aKD5aSJ5pWw44Gu6aCG44CC54Sh44GR44KM44GwIChOb25lLCBOb25lKeOAggoKICAgIOOCq+ODvOODjeODq+OBq+OBryBrYWdnbGUuanNvbiDjgYznhKHjgYTjgYzjgIFLQUdHTEVfVVNFUk5BTUUgLyBLQUdHTEVfS0VZIOOBjAogICAg55Kw5aKD5aSJ5pWw44Gn5YWl44Gj44Gm44GE44KL44CCCiAgICAiIiIKICAgIGNyZWRfcGF0aCA9IEtDX0NSRURfUEFUSCBpZiBjcmVkX3BhdGggaXMgTm9uZSBlbHNlIGNyZWRfcGF0aAogICAgZW52aXJvbiA9IG9zLmVudmlyb24gaWYgZW52aXJvbiBpcyBOb25lIGVsc2UgZW52aXJvbgogICAgaWYgb3MucGF0aC5pc2ZpbGUoY3JlZF9wYXRoKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHdpdGggb3BlbihjcmVkX3BhdGgpIGFzIGY6CiAgICAgICAgICAgICAgICBjZmcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgdXNlciwga2V5ID0gY2ZnLmdldCgidXNlcm5hbWUiKSwgY2ZnLmdldCgia2V5IikKICAgICAgICAgICAgaWYgdXNlciBhbmQga2V5OgogICAgICAgICAgICAgICAgcmV0dXJuIHVzZXIsIGtleQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgIHVzZXIgPSBlbnZpcm9uLmdldCgiS0FHR0xFX1VTRVJOQU1FIikgb3IgZW52aXJvbi5nZXQoIktBR0dMRV9VU0VSIikKICAgIGtleSA9IGVudmlyb24uZ2V0KCJLQUdHTEVfS0VZIikKICAgIHJldHVybiAodXNlciwga2V5KSBpZiB1c2VyIGFuZCBrZXkgZWxzZSAoTm9uZSwgTm9uZSkKCgpkZWYga2NfZmV0Y2hfcXVvdGEodXNlciwga2V5LCB0aW1lb3V0PTQ1LCB1cmw9S0NfUVVPVEFfVVJMLCBvcGVuZXI9Tm9uZSk6CiAgICAiIiLjgq/jgqnjg7zjgr/jga7nlJ8gSlNPTiDjgpLov5TjgZnjgIJIVFRQIC8g5o6l57aa44Ko44Op44O844Gv5L6L5aSW44Gu44G+44G+5oqV44GS44KL44CCIiIiCiAgICB0b2tlbiA9IGJhc2U2NC5iNjRlbmNvZGUoZiJ7dXNlcn06e2tleX0iLmVuY29kZSgpKS5kZWNvZGUoKQogICAgcmVxID0gdXJsbGliLnJlcXVlc3QuUmVxdWVzdCgKICAgICAgICB1cmwsIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJhc2ljICIgKyB0b2tlbn0pCiAgICB3aXRoIChvcGVuZXIgb3IgdXJsbGliLnJlcXVlc3QudXJsb3BlbikocmVxLCB0aW1lb3V0PXRpbWVvdXQpIGFzIHI6CiAgICAgICAgcmV0dXJuIGpzb24ubG9hZHMoci5yZWFkKCkpCgoKZGVmIGtjX3F1b3RhX3N1bW1hcnkoZGF0YSk6CiAgICAiIiJVSSDnlKjjga4gR1BVIOOCr+OCqeODvOOCv+imgee0hOOAgmdwdVF1b3RhIOOBjOeEoeOBhOW/nOetlOOBryBhdmFpbGFibGU9RmFsc2XjgIIiIiIKICAgIGcgPSBkYXRhLmdldCgiZ3B1UXVvdGEiKSBvciB7fQogICAgdXNlZCA9IChnLmdldCgidGltZVVzZWQiKSBvciB7fSkuZ2V0KCJzZWNvbmRzIikKICAgIHRvdGFsID0gKGcuZ2V0KCJ0b3RhbFRpbWVBbGxvd2VkIikgb3Ige30pLmdldCgic2Vjb25kcyIpCiAgICBpZiB1c2VkIGlzIE5vbmUgb3IgdG90YWwgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJhdmFpbGFibGUiOiBGYWxzZSwgInJlYXNvbiI6ICJub19ncHVfcXVvdGEifQogICAgcmVzZXJ2ZWQgPSAoZy5nZXQoInRpbWVSZXNlcnZlZCIpIG9yIHt9KS5nZXQoInNlY29uZHMiLCAwKQogICAgcmV0dXJuIHsKICAgICAgICAiYXZhaWxhYmxlIjogVHJ1ZSwKICAgICAgICAidXNlZFNlY29uZHMiOiB1c2VkLAogICAgICAgICJ0b3RhbFNlY29uZHMiOiB0b3RhbCwKICAgICAgICAicmVzZXJ2ZWRTZWNvbmRzIjogcmVzZXJ2ZWQsCiAgICAgICAgImxlZnRTZWNvbmRzIjogdG90YWwgLSB1c2VkIC0gcmVzZXJ2ZWQsCiAgICAgICAgInJlZnJlc2hUaW1lIjogZGF0YS5nZXQoInF1b3RhUmVmcmVzaFRpbWUiLCAiIiksCiAgICB9CgojIS91c3IvYmluL2VudiBweXRob24zCiIiIkJhc2ljIOiqjeiovOOBpOOBjeODquODkOODvOOCueODl+ODreOCreOCt++8iOaomea6luODqeOCpOODluODqeODquOBruOBv++8ieOAggoKY2xvdWRmbGFyZWQg44GuIHF1aWNrIHR1bm5lbCDjgavjga8gQmFzaWMg6KqN6Ki85qmf6IO944GM54Sh44GE44Gf44KB44CBCk9sbGFtYSDjga7miYvliY3jgavjgZPjgozjgpLnva7jgYTjgaboqo3oqLzjgpLmi4Xjgo/jgZvjgovjgIIKCiAgICBBUElfVVNFUj1vbGxhbWEgQVBJX1BBU1M9c2VjcmV0IHB5dGhvbjMgYXV0aF9wcm94eS5weSBcCiAgICAgICAgLS1wb3J0IDExNDM1IC0tdXBzdHJlYW0gaHR0cDovLzEyNy4wLjAuMToxMTQzNAoK44K544OI44Oq44O844Of44Oz44Kw77yIT2xsYW1hIOOBriB0b2tlbiDpgJDmrKHlh7rlipvvvInjgpLpgJrjgZnjgZ/jgoHjgIEK5LiK5rWB44Gu5b+c562U44GvIGNodW5rZWQgdHJhbnNmZXItZW5jb2Rpbmcg44Gn44Gd44Gu44G+44G+5Lit57aZ44GZ44KL44CCCiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBiYXNlNjQKaW1wb3J0IGhtYWMKaW1wb3J0IGh0dHAuc2VydmVyCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgcGF0aGxpYgppbXBvcnQgc2lnbmFsCmltcG9ydCBzb2NrZXRzZXJ2ZXIKaW1wb3J0IHN5cwppbXBvcnQgdGhyZWFkaW5nCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5wYXJzZQppbXBvcnQgdXJsbGliLnJlcXVlc3QKCiMg5LiK5rWB77yIT2xsYW1h77yJ44G444Gu5b6F44Gh5pmC6ZaT44CC6ZW344GE55Sf5oiQ44Gr6ICQ44GI44KL44KI44GG6ZW344KB44Gr5Y+W44KL44CCClVQU1RSRUFNX1RJTUVPVVQgPSAzNjAwCiMg5Lit57aZ44Gu44OQ44OD44OV44Kh5LiK6ZmQ44CC5pS56KGM77yITkRKU09OIOihjO+8ieOBjOadpeOBquOBhOWgtOWQiOOBruODleOCqeODvOODq+ODkOODg+OCr+OBp+OAgQojIOOBk+OBk+OBq+mBlOOBmeOCi+OBvuOBp+W+heOBpOOBqOOCueODiOODquODvOODoOOBjOa6nOOBvuOCi+OBruOBp+OAgeWwj+OBleOBhOOBu+OBqemAkOasoeaAp+OBr+mrmOOBhOOAggpDSFVOSyA9IDQwOTYKIyBCYXNpYyDoqo3oqLzjgpLlhY3pmaTjgZnjgovjg5HjgrnjgIIKIyAvaGVhbHRoeiDjga/nlo7pgJrnorroqo3jgIEvIOOBqCAvY2hhdCDjga/jg4Hjg6Pjg4Pjg4ggVUnvvIjpnZnnmoTjgaogSFRNTCDjga7jgb/jgIIKIyDoqo3oqLzmg4XloLHjgoLjg6Ljg4fjg6vjgoLlkKvjgb7jgarjgYTjga7jgafjgIHlhazplovjgZfjgabjgoLmvI/jgozjgovjgoLjga7jga/nhKHjgYTvvInjgIIKIyBBUEkg5pys5L2T77yIL2FwaS8qLCAvdjEvKu+8ieOBr+W+k+adpeOBqeOBiuOCiiBCYXNpYyDoqo3oqLzjgYzopoHjgovjgIIKT1BFTl9QQVRIUyA9IHsiL2hlYWx0aHoiLCAiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCIsICIvZmF2aWNvbi5pY28ifQpVSV9QQVRIUyA9IHsiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCJ9CgojIOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiO+8iOS4iua1geOBuOa1geOBleOBquOBhO+8ieOAggojIC9xdW90YSA6IEthZ2dsZSDjga4gR1BVIOOCr+OCqeODvOOCv+aui+mHj++8iFVJIOOBruODmOODg+ODgOOBq+WHuuOBme+8iQojIC9zdG9wICA6IOOBk+OBruS4ree2meOBqOS4iua1geOBruOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCi+OAggojICAgICAgICAgIFBPU1QgKyDjg5jjg4Pjg4AgWC1LYy1TdG9wOiAxIOOBjOW/hemgiO+8iEdFVCDjga7jgrnjgq3jg6Pjg4rjgoTku5bjgrXjgqTjg4jjgYvjgonjga4KIyAgICAgICAgICBDU1JGIOOBp+iQveOBqOOBleOCjOOBquOBhOOCiOOBhuOBq++8ieOAgkFQSV9QQVNTIOioreWumuaZguOBryBCYXNpYyDoqo3oqLzjgoLopoHjgovjgIIKQURNSU5fUEFUSFMgPSB7Ii9xdW90YSIsICIvc3RvcCJ9ClNUT1BfSEVBREVSID0gIlgtS2MtU3RvcCIKCiMga2VlcGFsaXZlIOOCkuW7tumVt+OBmeOCi+OBruOBr+OAjOeUn+aIkOOAjeODquOCr+OCqOOCueODiOOBoOOBkeOAgi9hcGkvcHMg44KEIC92MS9tb2RlbHMg44GuCiMg44Od44O844Oq44Oz44Kw77yI44Op44Oz44OB44Oj44O844Gu6LW35YuV56K66KqN44O744K544Kt44Oj44OK77yJ44GnIEdQVSDjgrvjg4Pjgrfjg6fjg7PjgpLlu7blkb3jgZfjgarjgYTjgIIKR0VORVJBVElPTl9QQVRIUyA9IHsKICAgICIvYXBpL2NoYXQiLAogICAgIi9hcGkvZ2VuZXJhdGUiLAogICAgIi92MS9jaGF0L2NvbXBsZXRpb25zIiwKICAgICIvdjEvY29tcGxldGlvbnMiLAogICAgIi92MS9yZXNwb25zZXMiLAp9CgojIOOCr+OCqeODvOOCv+Wun+ijheOBryBydW50aW1lL3F1b3RhLnB5IOOBjOecn+Wun+a6kOOAguOCq+ODvOODjeODq+OBp+OBr+ODk+ODq+ODgOODvOOBjOWQjOOCveODvOOCueOCkgojIOWFiOmgreOBq+mAo+e1kOOBl+OBpuWfi+OCgei+vOOCgOOBn+OCgeOAgWtjXyog44GM5pyq5a6a576p44Gu44Ot44O844Kr44Or5a6f6KGM44Gg44GRIGltcG9ydCDjgZnjgovjgIIKaWYgImtjX2ZldGNoX3F1b3RhIiBub3QgaW4gZ2xvYmFscygpOgogICAgc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihwYXRobGliLlBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnRzWzJdKSkKICAgIGZyb20gcnVudGltZSBpbXBvcnQgcXVvdGEgYXMgX2tjX3F1b3RhCiAgICBrY19mZXRjaF9xdW90YSA9IF9rY19xdW90YS5rY19mZXRjaF9xdW90YQogICAga2NfcXVvdGFfc3VtbWFyeSA9IF9rY19xdW90YS5rY19xdW90YV9zdW1tYXJ5CiAgICBrY19rYWdnbGVfY3JlZGVudGlhbHMgPSBfa2NfcXVvdGEua2Nfa2FnZ2xlX2NyZWRlbnRpYWxzCgpIT1BfQllfSE9QID0gewogICAgImNvbm5lY3Rpb24iLAogICAgImtlZXAtYWxpdmUiLAogICAgInByb3h5LWF1dGhlbnRpY2F0ZSIsCiAgICAicHJveHktYXV0aG9yaXphdGlvbiIsCiAgICAidGUiLAogICAgInRyYWlsZXJzIiwKICAgICJ0cmFuc2Zlci1lbmNvZGluZyIsCiAgICAidXBncmFkZSIsCiAgICAiZXhwZWN0IiwKfQoKQVJHUyA9IE5vbmUKRVhQRUNURURfQVVUSCA9ICIiCiMg44OB44Oj44OD44OIIFVJIOOBriBIVE1M44CC6LW35YuV5pmC44Gr5LiA5bqm44Gg44GR6Kqt44KA77yI5q+O44Oq44Kv44Ko44K544OI44Gn44OH44Kj44K544Kv44KS5Y+p44GL44Gq44GE77yJ44CCCkNIQVRfQllURVMgPSBiIiIKCiMgLS0tLSDjgrXjg7Pjg5fjg6rjg7PjgrDlvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg44Gu44Go44GN44Gg44GR5pyJ5Yq577yJLS0tLQojIHZMTE0g54mI44Gn5b+F6KaB44CC44Kv44Op44Kk44Ki44Oz44OI77yIb3BlbmNvZGUg562J77yJ44GMIHRlbXBlcmF0dXJlPTAg44KS6YCB44KL44GoCiMg6LKq5qyy5rOV44Gr44Gq44KK44CBMjdCIOe0muOBp+OBr+WQjOOBmOaWh+OBrue5sOOCiui/lOOBl+OBp+W0qeWjiuOBl+OChOOBmeOBhOOAggojIHJlcGV0aXRpb25fcGVuYWx0eSDjgoLmnKrmjIflrprjgaDjgaggMS4w77yIPeeEoeWKue+8ieOBq+OBquOCi+OBn+OCgeS4i+mZkOOCkuWFpeOCjOOCi+OAggojIOaXouWumuOBruODl+ODreOCreOCt++8iE9sbGFtYSDniYjvvInjgafjga/nhKHlirnjgarjga7jgafmjJnli5Xjga/lpInjgo/jgonjgarjgYTjgIIKUlBfRkxPT1IgPSAxLjA1ICAgICAjIHJlcGV0aXRpb25fcGVuYWx0eSDjga7kuIvpmZAKVEVNUF9GTE9PUiA9IDAuNyAgICAjIHRlbXBlcmF0dXJlIOOBruS4i+mZkO+8iDAgPSDosqrmrLLms5XjgpLnpoHmraLvvIkKCgojIOODiOODs+ODjeODqy/jg4Hjg6Pjg4Pjg4jntYznlLHjga7jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgacga2VlcGFsaXZlIOaZgumWk+OCkuODquOCu+ODg+ODiOOAggojIC90bXAva2VlcGFsaXZlLmNvbmYg44GuIFJVTj0xLCBNSU5VVEVTPTYwIOOCkuWGjeabuOOBjei+vOOBv+OAggpfS0VFUEFMSVZFX0NPTkYgPSAiL3RtcC9rZWVwYWxpdmUuY29uZiIKCgpkZWYgX3Jlc2V0X2tlZXBhbGl2ZSgpOgogICAgIiIi44OB44Oj44OD44OI6YCB5L+h44GU44Go44Gr5pmC6ZaT44Oq44K744OD44OI77yIT04g57at5oyB77yJ44CC5LiK5rWB44Ko44Op44O85pmC44Gn44KC5ZG844G244CCIiIiCiAgICB0cnk6CiAgICAgICAgd2l0aCBvcGVuKF9LRUVQQUxJVkVfQ09ORiwgInciKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKCJSVU49MVxuTUlOVVRFUz02MFxuIikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKCmRlZiBfaXNfZ2VuZXJhdGlvbihwYXRoX29ubHkpOgogICAgcmV0dXJuIHBhdGhfb25seS5yc3RyaXAoIi8iKSBpbiBHRU5FUkFUSU9OX1BBVEhTCgoKZGVmIF9udW0odmFsdWUpOgogICAgIiIi5pWw5YCk5YyW44Gn44GN44Gq44GR44KM44GwIE5vbmXvvIhib29sIOOBr+aVsOWApOaJseOBhOOBl+OBquOBhO+8ieOAgiIiIgogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgYm9vbCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHRyeToKICAgICAgICByZXR1cm4gZmxvYXQodmFsdWUpCiAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcmV0dXJuIE5vbmUKCgpkZWYgZW5mb3JjZV9zYW1wbGluZyhib2R5LCBwYXRoX29ubHkpOgogICAgIiIiY2hhdC9jb21wbGV0aW9ucyDjga7jgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjgpLlvLfliLbjgZnjgovvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC77yJ44CCCgogICAgLSByZXBldGl0aW9uX3BlbmFsdHkgPCBSUF9GTE9PUu+8iOacquaMh+WumuWQq+OCgO+8ieKGkiBSUF9GTE9PUgogICAgLSB0ZW1wZXJhdHVyZSA8PSAw77yI5pyq5oyH5a6a5ZCr44KA77yJ4oaSIFRFTVBfRkxPT1IKCiAgICDjg5zjg4fjgqPjgYwgSlNPTiDjgafjgarjgYQv5aOK44KM44Gm44GE44KL5aC05ZCI44Gv57Sg6YCa44GX77yI5LiK5rWB44Gr5Yik5pat44KS5aeU44Gt44KL77yJ44CCCiAgICDmiLvjgorlgKTjga/lt67jgZfmm7/jgYjlvozjga4gYnl0ZXPjgIJDb250ZW50LUxlbmd0aCDjga/lkbzjgbPlh7rjgZflgbTjgafku5jjgZHnm7TjgZnjgIIKICAgICIiIgogICAgaWYgbm90IEFSR1MuZW5mb3JjZV9zYW1wbGluZyBvciBub3QgYm9keToKICAgICAgICByZXR1cm4gYm9keQogICAgIyDjgq/jgqjjg6rmloflrZfliJfvvIg/YXBpLXZlcnNpb249Li4uIOetie+8ieOBjOS7mOOBj+OBqCBlbmRzd2l0aCDjgYzlpJbjgozjgovjgZ/jgoHjgIEKICAgICMg44OR44K544Gg44GR44KS5Y+W44KK5Ye644GX44Gm5q+U6LyD44GZ44KL44CC5Y+W44KK44GT44G844GZ44Go55+v5q2j44GM54Sh5Yq55YyW44GV44KM44KL44CCCiAgICBfcGF0aCA9IHVybGxpYi5wYXJzZS51cmxzcGxpdChwYXRoX29ubHkpLnBhdGgKICAgIGlmIG5vdCBfcGF0aC5yc3RyaXAoIi8iKS5lbmRzd2l0aCgiL2NoYXQvY29tcGxldGlvbnMiKToKICAgICAgICByZXR1cm4gYm9keQogICAgdHJ5OgogICAgICAgIGRhdGEgPSBqc29uLmxvYWRzKGJvZHkpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBib2R5CiAgICBpZiBub3QgaXNpbnN0YW5jZShkYXRhLCBkaWN0KToKICAgICAgICByZXR1cm4gYm9keQogICAgY2hhbmdlZCA9IEZhbHNlCiAgICBycCA9IF9udW0oZGF0YS5nZXQoInJlcGV0aXRpb25fcGVuYWx0eSIpKQogICAgaWYgcnAgaXMgTm9uZSBvciBycCA8IFJQX0ZMT09SOgogICAgICAgIGRhdGFbInJlcGV0aXRpb25fcGVuYWx0eSJdID0gUlBfRkxPT1IKICAgICAgICBjaGFuZ2VkID0gVHJ1ZQogICAgdGVtcCA9IF9udW0oZGF0YS5nZXQoInRlbXBlcmF0dXJlIikpCiAgICBpZiB0ZW1wIGlzIE5vbmUgb3IgdGVtcCA8PSAwOgogICAgICAgIGRhdGFbInRlbXBlcmF0dXJlIl0gPSBURU1QX0ZMT09SCiAgICAgICAgY2hhbmdlZCA9IFRydWUKICAgIGlmIG5vdCBjaGFuZ2VkOgogICAgICAgIHJldHVybiBib2R5CiAgICByZXR1cm4ganNvbi5kdW1wcyhkYXRhLCBlbnN1cmVfYXNjaWk9RmFsc2UpLmVuY29kZSgidXRmLTgiKQoKCmNsYXNzIEhhbmRsZXIoaHR0cC5zZXJ2ZXIuQmFzZUhUVFBSZXF1ZXN0SGFuZGxlcik6CiAgICBwcm90b2NvbF92ZXJzaW9uID0gIkhUVFAvMS4xIgogICAgc2VydmVyX3ZlcnNpb24gPSAiT2xsYW1hQXV0aFByb3h5LzEuMCIKICAgICMg5o6l57aa44GU44Go44Gu44K944Kx44OD44OI44K/44Kk44Og44Ki44Km44OI44CC44Kv44Op44Kk44Ki44Oz44OI44GM54Sh6YCa5L+h44Gu44G+44G+5q6L44GZ44GoCiAgICAjIOODj+ODs+ODieODqeOCueODrOODg+ODieOBjOeEoemZkOOBq+WxheW6p+OCi+OBruOBp+OAgTXliIbjgaflm57lj47jgZnjgovvvIjjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK0KICAgICMg44Gu5LiK5rWB6Kqt44G/5Y+W44KK44GvIHVybGxpYiDlgbTjgr3jgrHjg4Pjg4jjgarjga7jgaflvbHpn7/jgpLlj5fjgZHjgarjgYTvvInjgIIKICAgIHRpbWVvdXQgPSAzMDAKCiAgICAjIC0tLS0g44Ot44Kw44KSIDEg6KGM44Gr5Zyn57iu77yI44OO44O844OI44OW44OD44Kv44Gu44Ot44Kw44KS5rGa44GV44Gq44GE77yJLS0tLQogICAgZGVmIGxvZ19tZXNzYWdlKHNlbGYsIGZtdCwgKmFyZ3MpOgogICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSAlc1xuIiAlIChmbXQgJSBhcmdzKSkKCiAgICBkZWYgX3JlYWRfYm9keShzZWxmKToKICAgICAgICAiIiLjg6rjgq/jgqjjgrnjg4jjg5zjg4fjgqPjgpLoqq3jgoDjgIJjaHVua2VkIOOBp+OCgiBDb250ZW50LUxlbmd0aCDjgafjgoLkuKHlr77lv5zjgIIKCiAgICAgICAgQ29udGVudC1MZW5ndGgg44GX44GL6KaL44Gq44GE44Go44CBSFRUUC8xLjEg44GuIGNodW5rZWQgUE9TVO+8iOOCueODiOODquODvOODn+ODs+OCsOWFpeWKm+OChAogICAgICAgIOODl+ODreOCreOCt+e1jOeUseOBruWkieaPm+OBp+eZuueUn+OBl+OBhuOCi++8ieOBpyoq44Oc44OH44Kj44GU44Go5qyg6JC9KirjgZfjgabkuIrmtYHjgavlsYrjgYvjgarjgYTjgIIKICAgICAgICAiIiIKICAgICAgICB0ZSA9IChzZWxmLmhlYWRlcnMuZ2V0KCJUcmFuc2Zlci1FbmNvZGluZyIpIG9yICIiKS5sb3dlcigpCiAgICAgICAgaWYgImNodW5rZWQiIGluIHRlOgogICAgICAgICAgICBvdXQgPSBieXRlYXJyYXkoKQogICAgICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICAgICAgc2l6ZV9saW5lID0gc2VsZi5yZmlsZS5yZWFkbGluZSgpLnN0cmlwKCkKICAgICAgICAgICAgICAgIGlmIGIiOyIgaW4gc2l6ZV9saW5lOiAgICAgICAgICAjIGNodW5rLWV4dCDjga/mjajjgabjgosKICAgICAgICAgICAgICAgICAgICBzaXplX2xpbmUgPSBzaXplX2xpbmUuc3BsaXQoYiI7IiwgMSlbMF0KICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBzaXplID0gaW50KHNpemVfbGluZSwgMTYpCiAgICAgICAgICAgICAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICAgICAgaWYgc2l6ZSA9PSAwOiAgICAgICAgICAgICAgICAgICMg44OI44Os44O844Op44O86Kqt44G/5o2o44GmCiAgICAgICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICAgICAgdHJhaWxlciA9IHNlbGYucmZpbGUucmVhZGxpbmUoKQogICAgICAgICAgICAgICAgICAgICAgICBpZiB0cmFpbGVyIGluIChiIlxyXG4iLCBiIlxuIiwgYiIiKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgIG91dCArPSBzZWxmLnJmaWxlLnJlYWQoc2l6ZSkKICAgICAgICAgICAgICAgIHNlbGYucmZpbGUucmVhZCgyKSAgICAgICAgICAgICAjIOODgeODo+ODs+OCr+acq+WwvuOBriBDUkxGCiAgICAgICAgICAgIHJldHVybiBieXRlcyhvdXQpIG9yIE5vbmUKCiAgICAgICAgbGVuZ3RoID0gaW50KHNlbGYuaGVhZGVycy5nZXQoIkNvbnRlbnQtTGVuZ3RoIikgb3IgMCkKICAgICAgICByZXR1cm4gc2VsZi5yZmlsZS5yZWFkKGxlbmd0aCkgaWYgbGVuZ3RoIGVsc2UgTm9uZQoKICAgIGRlZiBfZW1pdChzZWxmLCBkYXRhKToKICAgICAgICAiIiJjaHVua2VkIOOBrjHliIbniYfjgpLpgIHjgovjgIIKCiAgICAgICAg5pS56KGM5Y2Y5L2N44Gn5ZG844G244Gu44GnIHNlbmQg5Zue5pWw44Gv44OI44O844Kv44Oz5pWw44Gr5ZCM5q+U5L6L77yIMeODkOOCpOODiOavjuOBq+OBr+mAgeOCieOBquOBhO+8ieOAggogICAgICAgICIiIgogICAgICAgIGlmIG5vdCBkYXRhOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiJXhcclxuIiAlIGxlbihkYXRhKSkKICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJ5dGVzKGRhdGEpKQogICAgICAgIHNlbGYud2ZpbGUud3JpdGUoYiJcclxuIikKICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKCiAgICBkZWYgX3NlbmRfanNvbihzZWxmLCBjb2RlLCBvYmosIGNsb3NlPUZhbHNlLCBleHRyYV9oZWFkZXJzPSgpKToKICAgICAgICAiIiJKU09OIOOCkiAxIOeZuuOBp+i/lOOBme+8iOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiOODu+OCqOODqeODvOW/nOetlOOBruWFsemAmuWHpueQhu+8ieOAgiIiIgogICAgICAgIGJvZHkgPSBvYmogaWYgaXNpbnN0YW5jZShvYmosIGJ5dGVzKSBlbHNlIGpzb24uZHVtcHMob2JqKS5lbmNvZGUoKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGNvZGUpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ2FjaGUtQ29udHJvbCIsICJuby1zdG9yZSIpCiAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gZXh0cmFfaGVhZGVyczoKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb25uZWN0aW9uIiwgImNsb3NlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJvZHkpCiAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQoKICAgIGRlZiBfYXV0aG9yaXplZChzZWxmKToKICAgICAgICAjIEFQSV9QQVNTIOOBjOepuiA9IOiqjeiovOOBquOBl++8iOWFrOmWi++8ieOAglVJIOOCkumWi+OBhOOBn+OBoOOBkeOBp+S9v+OBiOOCi+OCiOOBhuOBq+OBmeOCi+OAggogICAgICAgIGlmIG5vdCBFWFBFQ1RFRF9BVVRIOgogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGlmIHNlbGYucGF0aC5zcGxpdCgiPyIpWzBdIGluIE9QRU5fUEFUSFM6CiAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgIyDluLjmlbDmmYLplpPmr5TovIPvvIjjgr/jgqTjg5/jg7PjgrDmlLvmkoPjga7nt6nlkozvvInjgIJzdHIg54mIIGNvbXBhcmVfZGlnZXN0IOOBrwogICAgICAgICMg6Z2eIEFTQ0lJIOOBpyBUeXBlRXJyb3Ig44Gr44Gq44KL44Gu44GnIGJ5dGVzIOOBp+avlOi8g+OBmeOCi+OAggogICAgICAgICMgT1BUSU9OUyDjga8gX3JlbGF5IOWGkumgreOBp+ODreODvOOCq+ODq+WNs+etlOa4iOOBv++8iOOBk+OBk+OBq+OBr+WxiuOBi+OBquOBhO+8ieOAggogICAgICAgIGdvdCA9IHNlbGYuaGVhZGVycy5nZXQoIkF1dGhvcml6YXRpb24iLCAiIikuZW5jb2RlKCJsYXRpbi0xIikKICAgICAgICByZXR1cm4gaG1hYy5jb21wYXJlX2RpZ2VzdChnb3QsIEVYUEVDVEVEX0FVVEguZW5jb2RlKCJhc2NpaSIpKQoKICAgIGRlZiBfZGVueShzZWxmKToKICAgICAgICAjIDQwMSDjgafmjqXntprjgpLplonjgZjjgovjgIJLZWVwLUFsaXZlIOOBruOBvuOBvuaui+OBmeOBqOOAgeasoeOBruODquOCr+OCqOOCueODiOOBrgogICAgICAgICMg44Oc44OH44Kj6Kqt44G/5Y+W44KK5L2N572u44GM44Ga44KM44GmIEhUVFAg44Oq44Kv44Ko44K544OI44K544Oi44Ks44Oq44Oz44Kw44GM6LW344GN44GG44KL44CCCiAgICAgICAgc2VsZi5fc2VuZF9qc29uKDQwMSwgeyJlcnJvciI6ICJ1bmF1dGhvcml6ZWQifSwgY2xvc2U9VHJ1ZSwKICAgICAgICAgICAgICAgICAgICAgICAgZXh0cmFfaGVhZGVycz1bKCJXV1ctQXV0aGVudGljYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICdCYXNpYyByZWFsbT0iT2xsYW1hIEFQSSInKV0pCgogICAgZGVmIF9zZXJ2ZV91aShzZWxmKToKICAgICAgICAiIiLjg4Hjg6Pjg4Pjg4ggVUkg44KS6L+U44GZ44CC6KqN6Ki85LiN6KaB77yI6Z2Z55qE44OV44Kh44Kk44Or44Gu44G/77yJ44CCCgogICAgICAgIFVJIOOBryAxIOODquOCr+OCqOOCueODiOOBp+e1guOCj+OCi+OBruOBpyAqKuaOpee2muOCkumWieOBmOOCiyoq44CC6KqN6Ki844OR44K544Go6YGV44GECiAgICAgICAg44GT44GT44Gn44Gv44Oc44OH44Kj44KS6Kqt44G+44Gq44GE44Gf44KB44CBa2VlcC1hbGl2ZSDjgpLntq3mjIHjgZnjgovjgagKICAgICAgICBDb250ZW50LUxlbmd0aCDku5jjgY0gUE9TVCDjgafmrKHjga7jg6rjgq/jgqjjgrnjg4jkvY3nva7jgYzjgZrjgozjgosKICAgICAgICDvvIjjg6rjgq/jgqjjgrnjg4jjgrnjg57jgrDjg6rjg7PjgrDvvInjgIIKICAgICAgICAiIiIKICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKSAgICAgICAgICAjIOaDs+WumuWkluOBruODnOODh+OCo+OBr+aNqOOBpuOCi++8iOiqreOBv+aui+OBl+mYsuatou+8iQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICBpZiBub3QgQ0hBVF9CWVRFUzoKICAgICAgICAgICAgYm9keSA9IChiImxsYW1hLWNoYXQuaHRtbCBub3QgZm91bmQuICIKICAgICAgICAgICAgICAgICAgICBiInN0YXJ0IHRoZSBwcm94eSB3aXRoIC0tY2hhdC1odG1sIDxwYXRoPi4iKQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoNTAzKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCAidGV4dC9wbGFpbjsgY2hhcnNldD11dGYtOCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShib2R5KQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjAwKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtVHlwZSIsICJ0ZXh0L2h0bWw7IGNoYXJzZXQ9dXRmLTgiKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihDSEFUX0JZVEVTKSkpCiAgICAgICAgIyBVSSDjga/mr47lm57lkIzjgZjlhoXlrrnjgILjg5fjg63jgq3jgrfmm7TmlrDmmYLjgavjg5bjg6njgqbjgrbjgYzlj6TjgYTniYjjgpLkvb/jgo/jgarjgYTjgojjgYYgbm8tc3RvcmXjgIIKICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDYWNoZS1Db250cm9sIiwgIm5vLXN0b3JlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKENIQVRfQllURVMpCgogICAgZGVmIF9zZXJ2ZV9xdW90YShzZWxmKToKICAgICAgICAiIiJLYWdnbGUg44GuIEdQVSDjgq/jgqnjg7zjgr/mrovph4/jgpLov5TjgZnjgIIKCiAgICAgICAgVUkg44Gu44OY44OD44OA44Gr44CM5q6L44KKIDA4OjU544CN44Go5Ye644GZ44Gf44KB44Gr5L2/44GG44CCCiAgICAgICAg6LOH5qC85oOF5aCx77yIa2FnZ2xlLmpzb24g44G+44Gf44Gv55Kw5aKD5aSJ5pWw77yJ44GM54Sh44GE44CB44G+44Gf44GvIEFQSSDjgYwKICAgICAgICDokL3jgaHjgabjgYTjgovloLTlkIjjga8gYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OBpiBVSSDlgbTjgafpnZ7ooajnpLrjgavjgZnjgovjgIIKICAgICAgICAiIiIKICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZX0KICAgICAgICBpZiBBUkdTLm5vX3F1b3RhOgogICAgICAgICAgICAjIEthZ2dsZSDku6XlpJbjga7jg63jg7zjgqvjg6vvvIh1enUg44Gq44Gp77yJ44Gn44Gv44Kv44Kp44O844K/44Gu5qaC5b+144GM54Sh44GE44CCCiAgICAgICAgICAgICMgYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OAgVVJIOWBtOOBr+aui+OCiuaZgumWk+OCkumdnuihqOekuuOBq+OBmeOCi+OAggogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7ImF2YWlsYWJsZSI6IEZhbHNlLCAicmVhc29uIjogIm5vX3F1b3RhIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHVzZXIsIGtleSA9IGtjX2thZ2dsZV9jcmVkZW50aWFscygpCiAgICAgICAgICAgIGlmIHVzZXIgYW5kIGtleToKICAgICAgICAgICAgICAgIHBheWxvYWQgPSBrY19xdW90YV9zdW1tYXJ5KAogICAgICAgICAgICAgICAgICAgIGtjX2ZldGNoX3F1b3RhKHVzZXIsIGtleSwgdGltZW91dD0xNSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZSwgImVycm9yIjogdHlwZShlKS5fX25hbWVfX30KCiAgICAgICAgIyDmrovph4/jga/liLvjgIXlpInjgo/jgovjgIJfc2VuZF9qc29uIOOBryBuby1zdG9yZSDjgpLku5jjgZHjgovjgIIKICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCBwYXlsb2FkKQoKICAgIGRlZiBfc2VydmVfc3RvcChzZWxmKToKICAgICAgICAiIiLjgZPjga7kuK3ntpnjgajkuIrmtYHjga7jgrvjg4Pjgrfjg6fjg7PjgpLmraLjgoHjgovjgIIKCiAgICAgICAg5ZG844Gz5Ye644GX5p2h5Lu277yIUE9TVCArIFgtS2MtU3RvcCArIOiqjeiovO+8ieOBryBfc3RvcF9hbGxvd2VkKCkg44Gn56K66KqN5riI44G/44CCCgogICAgICAgIOODluODqeOCpuOCtuOBruWBnOatouODnOOCv+ODs+OBi+OCieWPqeOBj+OAguS4ree2meOBjOiQveOBoeOCi+OBruOBp+W/nOetlOOCkui/lOOBl+OBpuOBi+OCiQogICAgICAgIOODl+ODreOCu+OCueOCkue1guS6huOBleOBm+OCi++8iOWFiOOBq+iQveOBqOOBmeOBqOODluODqeOCpuOCtuWBtOOBjOmAmuS/oeOCqOODqeODvOOBq+OBquOCi++8ieOAggoKICAgICAgICBLYWdnbGUg5LiK44Gn44GvIGtlZXBhbGl2ZSDjg6vjg7zjg5fjgoLmraLjgoHjgovjgZ/jgoHjgIEvdG1wL2tlZXBhbGl2ZS5jb25mIOOBqwogICAgICAgIFJVTj0wIOOCkuabuOOBjei+vOOCgOOAguOBk+OCjOOBq+OCiOOCiiBLYWdnbGUg44Kr44O844ON44Or44K744OD44K344On44Oz44GM57WC5LqG44GX44CBCiAgICAgICAgR1BVIOOCr+OCqeODvOOCv+OCkumWi+aUvuOBp+OBjeOCi+OAggoKICAgICAgICAtLXN0b3AtcGlkIOOBjOaMh+WumuOBleOCjOOBpuOBhOOCjOOBsOOAgeOBneOBriBQSUTvvIjjg6njg7Pjg4Hjg6Pjg7zvvInjgavjgoIKICAgICAgICBTSUdURVJNIOOCkumAgeOCi+OAgnV6dSDjga8gcnVuLXV6dS1zZXJ2ZXIuY29tbWFuZCDjgYzlrZDjg5fjg63jgrvjgrnjgpIKICAgICAgICDmirHjgYjjgabjgYTjgovjga7jgafjgIHopqrjgpLokL3jgajjgZvjgbAgdHJhcCBjbGVhbnVwIOOBp+WFqOmDqOeJh+S7mOOBj+OAggogICAgICAgICIiIgogICAgICAgICMgS2FnZ2xlIOeJiCBrZWVwYWxpdmUg44Or44O844OX44Gr44KC5YGc5q2i44KS5Lyd44GI44KLCiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oIi90bXAva2VlcGFsaXZlLmNvbmYiLCAidyIpIGFzIGY6CiAgICAgICAgICAgICAgICBmLndyaXRlKCJSVU49MFxuIikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgICAgIHNlbGYuX3NlbmRfanNvbigyMDAsIHsic3RvcHBlZCI6IFRydWUsICJzY29wZSI6ICJyZWxheSJ9KQoKICAgICAgICBkZWYgX2RpZSgpOgogICAgICAgICAgICB0aW1lLnNsZWVwKDAuNSkKICAgICAgICAgICAgIyDlhYjjgavjg6njg7Pjg4Hjg6Pjg7zjgpLokL3jgajjgZnvvIjlrZDjg5fjg63jgrvjgrnjgaggVUkg44KC5LiA57eS44Gr5raI44GI44KL77yJCiAgICAgICAgICAgIGlmIEFSR1Muc3RvcF9waWQ6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgb3Mua2lsbChBUkdTLnN0b3BfcGlkLCBzaWduYWwuU0lHVEVSTSkKICAgICAgICAgICAgICAgIGV4Y2VwdCAoUHJvY2Vzc0xvb2t1cEVycm9yLCBQZXJtaXNzaW9uRXJyb3IpOgogICAgICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgICAgIHRpbWUuc2xlZXAoMC41KQogICAgICAgICAgICBvcy5fZXhpdCgwKQoKICAgICAgICB0aHJlYWRpbmcuVGhyZWFkKHRhcmdldD1fZGllLCBkYWVtb249VHJ1ZSkuc3RhcnQoKQoKICAgIGRlZiBfcmVsYXkoc2VsZik6CiAgICAgICAgIyBDT1JTIOODl+ODquODleODqeOCpOODiOOBr+iqjeiovOS4jeimgeODu+ODreODvOOCq+ODq+WNs+etlOOAguS4iua1geOBuOa1geOBleOBquOBhAogICAgICAgICMg77yI57Sg6YCa44GX44GZ44KL44Go5pyq6KqN6Ki844Oq44Kv44Ko44K544OI44GM5LiK5rWB44Gr5bGK44GN44CB44Oc44OH44Kj5b6F44Gh44GnCiAgICAgICAgIyAg44OP44Oz44OJ44Op44K544Os44OD44OJ44KC5Y2g44KB44KJ44KM44KL77yJ44CCCiAgICAgICAgaWYgc2VsZi5jb21tYW5kID09ICJPUFRJT05TIjoKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDIwNCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgIyAvaGVhbHRoeiDjga/jg63jg7zjgqvjg6vjgafljbPnrZTjgZnjgovvvIjkuIrmtYHjga7mnInnhKHjgavplqLkv4LjgarjgY/nlJ/lrZjnorroqo3jgafjgY3jgovvvIkKICAgICAgICBpZiBzZWxmLnBhdGguc3BsaXQoIj8iKVswXSA9PSAiL2hlYWx0aHoiOgogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7InN0YXR1cyI6ICJvayJ9KQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgcGF0aF9vbmx5ID0gc2VsZi5wYXRoLnNwbGl0KCI/IilbMF0KICAgICAgICBpZiBwYXRoX29ubHkgaW4gVUlfUEFUSFM6CiAgICAgICAgICAgIHNlbGYuX3NlcnZlX3VpKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgICMgLS0tLSDnrqHnkIbjgqjjg7Pjg4njg53jgqTjg7Pjg4jvvIjjg63jg7zjgqvjg6vlsILnlKjvvIktLS0tCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvcXVvdGEiOgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9xdW90YSgpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGlmIHBhdGhfb25seSA9PSAiL3N0b3AiOgogICAgICAgICAgICAjIOiQveOBqOOBm+OCi+OBruOBryBVSSDjga7lgZzmraLjg5zjgr/jg7PvvIhQT1NUICsgWC1LYy1TdG9w77yJ44Gg44GR44CCCiAgICAgICAgICAgICMg5Lul5YmN44Gv6KqN6Ki844KI44KK5YmN44Gr5Yem55CG44GX44Gm44GE44Gf44Gf44KB44CBQVBJX1BBU1Mg44KS6Kit5a6a44GX44Gm44GE44Gm44KCCiAgICAgICAgICAgICMg6Kqw44Gn44KCIEdFVCAvc3RvcCDjgacgS2FnZ2xlIOOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCieOCjOOBn+OAggogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIlBPU1QiIG9yIHNlbGYuaGVhZGVycy5nZXQoU1RPUF9IRUFERVIpICE9ICIxIjoKICAgICAgICAgICAgICAgIHNlbGYuX3NlbmRfanNvbig0MDUsIHsiZXJyb3IiOiAiUE9TVCB3aXRoICVzOiAxIHJlcXVpcmVkIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICUgU1RPUF9IRUFERVJ9LCBjbG9zZT1UcnVlKQogICAgICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgICAgIGlmIG5vdCBzZWxmLl9hdXRob3JpemVkKCk6CiAgICAgICAgICAgICAgICBzZWxmLl9kZW55KCkKICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9zdG9wKCkKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvZmF2aWNvbi5pY28iOgogICAgICAgICAgICAjIOODluODqeOCpuOCtuOBjOWLneaJi+OBq+WPluOCiuOBq+adpeOCi+OAgjQwNCDjgpLov5TjgZnjgajjg63jgrDjgYzln4vjgb7jgovjgIIKICAgICAgICAgICAgc2VsZi5jbG9zZV9jb25uZWN0aW9uID0gVHJ1ZQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjA0KQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsICIwIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICBpZiBub3Qgc2VsZi5fYXV0aG9yaXplZCgpOgogICAgICAgICAgICAjIDQwMSDjgpLov5TjgZnliY3jgavjg5zjg4fjgqPjgpLpo5/jgaPjgabjgYrjgY/jgIJQT1NUIOOBruODnOODh+OCo+OCkuiqreOBvuOBmuOBqwogICAgICAgICAgICAjIOaOpee2muOCkuWIh+OCi+OBqOOCr+ODqeOCpOOCouODs+ODiOWBtOOBjCBSU1Qg44KS5Y+X44GR44Gm44CMNDAxIOOCiOOCiuWJjeOBqwogICAgICAgICAgICAjIOaOpee2muOBjOWIh+OCjOOBn+OAjeOBq+imi+OBiOOCi++8iGN1cmwg44GvIFJTVCDjgpLjgqjjg6njg7zooajnpLrvvInjgIIKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5fcmVhZF9ib2R5KCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgc2VsZi5fZGVueSgpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGJvZHkgPSBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICMg5LiN5q2j44GqIENvbnRlbnQtTGVuZ3RoIC8g5aOK44KM44GfIGNodW5rZWQg44Gv5LiK5rWB44Gr5rWB44GV44Gq44GE44CCCiAgICAgICAgICAgICMg5pyq5o2V5o2J44Gg44Go44K544K/44OD44Kv44OA44Oz44OX44GuNTAw44Gr44Gq44KL77yI6KqN6Ki85oiQ5Yqf5YG044Gg44GR6Z2e5a++56ew44Gr44Gq44KJ44Gq44GE44KI44GG77yJCiAgICAgICAgICAgIHNlbGYuY2xvc2VfY29ubmVjdGlvbiA9IFRydWUKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDQwMCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbm5lY3Rpb24iLCAiY2xvc2UiKQogICAgICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlmIF9pc19nZW5lcmF0aW9uKHBhdGhfb25seSk6CiAgICAgICAgICAgIF9yZXNldF9rZWVwYWxpdmUoKSAgIyDnlJ/miJDjg6rjgq/jgqjjgrnjg4jvvJ3jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgYLjgoog4oaSIDYw5YiG44Oq44K744OD44OICiAgICAgICAgIyDjgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjga7lvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC44Gu44G/44CC5pei5a6a44Gv57Sg6YCa44GX77yJCiAgICAgICAgYm9keSA9IGVuZm9yY2Vfc2FtcGxpbmcoYm9keSwgcGF0aF9vbmx5KQogICAgICAgIHVybCA9IEFSR1MudXBzdHJlYW0ucnN0cmlwKCIvIikgKyBzZWxmLnBhdGgKICAgICAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBtZXRob2Q9c2VsZi5jb21tYW5kKQogICAgICAgIGZvciBrZXksIHZhbHVlIGluIHNlbGYuaGVhZGVycy5pdGVtcygpOgogICAgICAgICAgICAjIGNvbnRlbnQtbGVuZ3RoIOOBryBib2R5IOWGjeani+evieW+jOOBruWun+mVt+OBp+mAgeOCiuebtOOBmeOBn+OCgeOBk+OBk+OBp+aNqOOBpuOCiwogICAgICAgICAgICAjIO+8iOOCr+ODqeOCpOOCouODs+ODiOWApOOBruOBvuOBvui7oumAgeOBmeOCi+OBqCBjaHVua2VkICsgQ0wg5L2155So44Gn44OH44K344Oz44Kv44GZ44KL77yJCiAgICAgICAgICAgIGlmIGtleS5sb3dlcigpIGluIEhPUF9CWV9IT1AgfCB7Imhvc3QiLCAiYXV0aG9yaXphdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImNvbnRlbnQtbGVuZ3RoIn06CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZXEuYWRkX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGJvZHkgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJlcS5hZGRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsIHN0cihsZW4oYm9keSkpKQoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3AgPSB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJlcSwgdGltZW91dD1VUFNUUkVBTV9USU1FT1VUKQogICAgICAgIGV4Y2VwdCB1cmxsaWIuZXJyb3IuSFRUUEVycm9yIGFzIGV4YzoKICAgICAgICAgICAgIyDkuIrmtYHjgYwgNHh4LzV4eCDjgpLov5TjgZfjgZ/loLTlkIjjgoLmnKzmlofjgpLjgZ3jga7jgb7jgb7ov5TjgZkKICAgICAgICAgICAgcGF5bG9hZCA9IGV4Yy5yZWFkKCkKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGV4Yy5jb2RlKQogICAgICAgICAgICBjdHlwZSA9IGV4Yy5oZWFkZXJzLmdldCgiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCBjdHlwZSkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCBzdHIobGVuKHBheWxvYWQpKSkKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIHNlbGYuY29tbWFuZCAhPSAiSEVBRCI6CiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKHBheWxvYWQpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246ICAjIOaOpee2muWkseaVl+OBquOBqQogICAgICAgICAgICAjIOWGhemDqOS+i+WkluOBruips+e0sOOBr+OCr+ODqeOCpOOCouODs+ODiOOBq+a4oeOBleOBquOBhO+8iOaDheWgsemcsuWHuumYsuatou+8ieOAggogICAgICAgICAgICAjIOWOn+WboOOBr+OCteODvOODkOWBtCBzdGRlcnIg44Gr44Gg44GR5pu444GP44CCCiAgICAgICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSB1cHN0cmVhbSBlcnJvcjogJXNcbiIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAlIChzeXMuZXhjX2luZm8oKVsxXSwpKQogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oNTAyLCB7ImVycm9yIjogInVwc3RyZWFtIHVucmVhY2hhYmxlIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB3aXRoIHJlc3A6CiAgICAgICAgICAgIHNlbGYuc2VuZF9yZXNwb25zZShyZXNwLnN0YXR1cykKICAgICAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gcmVzcC5oZWFkZXJzLml0ZW1zKCk6CiAgICAgICAgICAgICAgICBpZiBrZXkubG93ZXIoKSBpbiBIT1BfQllfSE9QIG9yIGtleS5sb3dlcigpID09ICJjb250ZW50LWxlbmd0aCI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoa2V5LCB2YWx1ZSkKICAgICAgICAgICAgIyBIRUFEIOOBr+acrOaWh+OBquOBl+OAgmNodW5rZWQg44KS5a6j6KiA44GZ44KL44Go44Kv44Op44Kk44Ki44Oz44OI44GM57WC56uv44OB44Oj44Oz44Kv44KSCiAgICAgICAgICAgICMg5b6F44Gj44Gm44OP44Oz44Kw44GZ44KL77yIQ29udGVudC1MZW5ndGgg44KC5rih44Gb44Gq44GE44Gu44Gn5a6j6KiA44GX44Gq44GE77yJ44CCCiAgICAgICAgICAgICMgMjA0LzMwNCDjga/mnKzmlofjgpLmjIHjgabjgarjgYTjgIJjaHVua2VkIOOCkuWuo+iogOOBmeOCi+OBqOWOs+agvOOBqgogICAgICAgICAgICAjIOOCr+ODqeOCpOOCouODs+ODiOOBjCBwcm90b2NvbCBlcnJvciDjgavjgZnjgovjgIIKICAgICAgICAgICAgbm9fYm9keSA9IHJlc3Auc3RhdHVzIGluICgyMDQsIDMwNCkgb3Igc2VsZi5jb21tYW5kID09ICJIRUFEIgogICAgICAgICAgICBpZiBub3Qgbm9fYm9keToKICAgICAgICAgICAgICAgICMg6ZW344GV5pyq55+l44Gu44K544OI44Oq44O844Og44Go44GX44Gm5Lit57aZ44GZ44KLCiAgICAgICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJUcmFuc2Zlci1FbmNvZGluZyIsICJjaHVua2VkIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIG5vX2JvZHk6CiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgIyByZXNwLnJlYWQoTikg44GvIEhUVFBSZXNwb25zZS5fcmVhZGludG9fY2h1bmtlZCDjga7mgKfos6rjgacKICAgICAgICAgICAgICAgICMgKipOIOODkOOCpOODiOaPg+OBhuOBvuOBp+iqreOBv+e2muOBkeOCiyoq44Gf44KB44CB44K544OI44Oq44O844Og44GM5YWo6Z2i44Gr5rqc44G+44KLCiAgICAgICAgICAgICAgICAjIO+8iE9sbGFtYSDjga4gTkRKU09OIOOBjOWFqOODiOODvOOCr+ODs+WIhuiyr+OBvuOBo+OBpuOBi+OCieS4gOaWieOBq+WxiuOBj++8ieOAggogICAgICAgICAgICAgICAgIyByZWFkbGluZShDSFVOSykg44Gv5pS56KGM44GLIENIVU5LIOODkOOCpOODiOOBp+i/lOOCi+OBruOBp+OAgemAkOasoeaAp+OCkgogICAgICAgICAgICAgICAgIyDkv53jgaPjgZ/jgb7jgb4gMSDjg5DjgqTjg4jjgZrjgaTjga4gcmVhZCDjg6vjg7zjg5fjgojjgoogQ1BVIOOCkumjn+OCj+OBquOBhOOAggogICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICBsaW5lID0gcmVzcC5yZWFkbGluZShDSFVOSykKICAgICAgICAgICAgICAgICAgICBpZiBub3QgbGluZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgICAgICBzZWxmLl9lbWl0KGxpbmUpCiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiMFxyXG5cclxuIikKICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICBleGNlcHQgKEJyb2tlblBpcGVFcnJvciwgQ29ubmVjdGlvblJlc2V0RXJyb3IpOgogICAgICAgICAgICAgICAgIyDjgq/jg6njgqTjgqLjg7Pjg4jjgYzliIfjgaPjgZ/vvIhDdHJsLUMg44Gq44Gp77yJ44CC5q2j5bi457O744CCCiAgICAgICAgICAgICAgICBwYXNzCgogICAgZG9fR0VUID0gX3JlbGF5CiAgICBkb19QT1NUID0gX3JlbGF5CiAgICBkb19QVVQgPSBfcmVsYXkKICAgIGRvX1BBVENIID0gX3JlbGF5CiAgICBkb19ERUxFVEUgPSBfcmVsYXkKICAgIGRvX0hFQUQgPSBfcmVsYXkKICAgIGRvX09QVElPTlMgPSBfcmVsYXkKCgpjbGFzcyBTZXJ2ZXIoc29ja2V0c2VydmVyLlRocmVhZGluZ01peEluLCBodHRwLnNlcnZlci5IVFRQU2VydmVyKToKICAgIGRhZW1vbl90aHJlYWRzID0gVHJ1ZQogICAgYWxsb3dfcmV1c2VfYWRkcmVzcyA9IFRydWUKCgpkZWYgbWFpbigpOgogICAgZ2xvYmFsIEFSR1MsIEVYUEVDVEVEX0FVVEgKCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXBvcnQiLCB0eXBlPWludCwgZGVmYXVsdD0xMTQzNSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0taG9zdCIsIGRlZmF1bHQ9IjEyNy4wLjAuMSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXVwc3RyZWFtIiwgZGVmYXVsdD0iaHR0cDovLzEyNy4wLjAuMToxMTQzNCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoYXQtaHRtbCIsIGRlZmF1bHQ9b3MuZW52aXJvbi5nZXQoIkNIQVRfSFRNTCIsICIiKSwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0i44OB44Oj44OD44OIIFVJIOOBriBIVE1M77yILyDjgafphY3kv6HjgZnjgovvvIkiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zdG9wLXBpZCIsIHR5cGU9aW50LCBkZWZhdWx0PTAsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9Ii9zdG9wIOaZguOBqyBTSUdURVJNIOOCkumAgeOCiyBQSUTjgIIiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInV6dSDjga7jgojjgYbjgavjg6njg7Pjg4Hjg6Pjg7zjgYzlrZDjg5fjg63jgrvjgrnjgpLmirHjgYjjgovloLTlkIjjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIuODqeODs+ODgeODo+ODvOOBriBQSUQg44KS5rih44GZ44Go5b6M54mH5LuY44GR44GU44Go5q2i44G+44KLIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm8tcXVvdGEiLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSIvcXVvdGEg44KS54Sh5Yq55YyW44GZ44KL77yIS2FnZ2xlIOS7peWkluOBruODreODvOOCq+ODq+eUqO+8iSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVuZm9yY2Utc2FtcGxpbmciLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJjaGF0L2NvbXBsZXRpb25zIOOBriB0ZW1wZXJhdHVyZTw9MCDjgpIgJS4xZiDjgavjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInJlcGV0aXRpb25fcGVuYWx0eSDjgpIgJS4yZiDku6XkuIrjgavnn6/mraPjgZnjgosiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJSAoVEVNUF9GTE9PUiwgUlBfRkxPT1IpKQogICAgQVJHUyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKCiAgICB1c2VyID0gb3MuZW52aXJvbi5nZXQoIkFQSV9VU0VSIikgb3IgIm9sbGFtYSIKICAgIHBhc3N3b3JkID0gb3MuZW52aXJvbi5nZXQoIkFQSV9QQVNTIikgb3IgIiIKICAgIGlmIHBhc3N3b3JkOgogICAgICAgIHRva2VuID0gYmFzZTY0LmI2NGVuY29kZShmInt1c2VyfTp7cGFzc3dvcmR9Ii5lbmNvZGUoKSkuZGVjb2RlKCkKICAgICAgICBFWFBFQ1RFRF9BVVRIID0gIkJhc2ljICIgKyB0b2tlbgogICAgZWxzZToKICAgICAgICAjIOaXouWumuOBr+iqjeiovOOBquOBl+OAglVSTCDjgpLnn6XjgaPjgabjgYTjgozjgbDoqrDjgafjgoLkvb/jgYjjgovngrnjgaDjgZHms6jmhI/jgIIKICAgICAgICBwcmludCgiYXV0aDogT0ZGIChBUElfUEFTUyDmnKroqK3lrpopIiwgZmx1c2g9VHJ1ZSkKCiAgICBnbG9iYWwgQ0hBVF9CWVRFUwogICAgaWYgQVJHUy5jaGF0X2h0bWw6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oQVJHUy5jaGF0X2h0bWwsICJyYiIpIGFzIGY6CiAgICAgICAgICAgICAgICBDSEFUX0JZVEVTID0gZi5yZWFkKCkKICAgICAgICBleGNlcHQgT1NFcnJvciBhcyBleGM6CiAgICAgICAgICAgICMgVUkg54Sh44GX44Gn44KCIEFQSSDjga/li5XjgY/jgILokL3jgajjgZXjgZrorablkYrjgaDjgZHlh7rjgZnjgIIKICAgICAgICAgICAgc3lzLnN0ZGVyci53cml0ZSgiY2hhdCB1aSBub3QgbG9hZGVkOiAlc1xuIiAlIChleGMsKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiY2hhdCB1aTogJXMgKCVkIEIpIiAlIChBUkdTLmNoYXRfaHRtbCwgbGVuKENIQVRfQllURVMpKSwKICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKCiAgICBodHRwZCA9IFNlcnZlcigoQVJHUy5ob3N0LCBBUkdTLnBvcnQpLCBIYW5kbGVyKQogICAgcHJpbnQoZiJhdXRoIHByb3h5OiB7QVJHUy5ob3N0fTp7QVJHUy5wb3J0fSAtPiB7QVJHUy51cHN0cmVhbX0gIgogICAgICAgICAgZiIodXNlcj17dXNlcn0pIiwgZmx1c2g9VHJ1ZSkKICAgIHRyeToKICAgICAgICBodHRwZC5zZXJ2ZV9mb3JldmVyKCkKICAgIGV4Y2VwdCBLZXlib2FyZEludGVycnVwdDoKICAgICAgICBwYXNzCiAgICBmaW5hbGx5OgogICAgICAgIGh0dHBkLnNlcnZlcl9jbG9zZSgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo="
HTML_B64 = "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImphIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0idXRmLTgiIC8+CiAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIiAvPgogIDx0aXRsZT7jg4Hjg6Pjg4Pjg4hVSTwvdGl0bGU+CiAgPHN0eWxlPgogICAgOnJvb3QgeyAtLWJnOiMwYjBjMTA7IC0tcGFuZWw6IzExMTMxYTsgLS1pbms6I2U5ZWVmMjsgLS1tdXRlZDojOWZiMGJmOyAtLWFjYzojNmFhMWZmOyAtLWxpbmU6IzFlMjIzMDsgfQogICAgKntib3gtc2l6aW5nOmJvcmRlci1ib3h9CiAgICBodG1sLGJvZHl7aGVpZ2h0OjEwMCV9CiAgICBib2R5e21hcmdpbjowOyBmb250OjE0cHgvMS40NSBzeXN0ZW0tdWksIC1hcHBsZS1zeXN0ZW0sIFNlZ29lIFVJLCBSb2JvdG8sIFVidW50dSwgQ2FudGFyZWxsLCBOb3RvIFNhbnMsIEhlbHZldGljYSwgQXJpYWw7IGJhY2tncm91bmQ6dmFyKC0tYmcpOyBjb2xvcjp2YXIoLS1pbmspfQogICAgLndyYXB7ZGlzcGxheTpncmlkOyBncmlkLXRlbXBsYXRlLXJvd3M6YXV0byAxZnIgYXV0bzsgaGVpZ2h0OjEwMCV9CiAgICBoZWFkZXIsIGZvb3RlcntiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTt9CiAgICBoZWFkZXJ7ZGlzcGxheTpmbGV4OyBnYXA6Ljc1cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXA7IHBhZGRpbmc6LjZyZW0gLjhyZW07IHBvc2l0aW9uOnN0aWNreTsgdG9wOjA7IHotaW5kZXg6MTA7IGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWxpbmUpfQogICAgaGVhZGVyIC50aXRsZXtmb250LXdlaWdodDo2MDA7IGxldHRlci1zcGFjaW5nOi4ycHh9CgogICAgaGVhZGVyIC5icmFuZHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4xNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucHJvbW97Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLWFjYyk7IHRleHQtZGVjb3JhdGlvbjpub25lOyBvcGFjaXR5Oi45fQogICAgaGVhZGVyIC5wcm9tbzpob3ZlcntvcGFjaXR5OjE7IHRleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmV9CgogICAgLyogSW5mbyBkaWFsb2cgKi8KICAgIGRpYWxvZy5pbmZvLWRpYWxvZ3sKICAgICAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYm9yZGVyLXJhZGl1czoxNHB4OwogICAgICBwYWRkaW5nOjA7CiAgICAgIGJhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6bWluKDU2MHB4LCBjYWxjKDEwMHZ3IC0gMjRweCkpOwogICAgICB3aWR0aDogNTYwcHg7CiAgICB9CiAgICBkaWFsb2cuaW5mby1kaWFsb2c6OmJhY2tkcm9we2JhY2tncm91bmQ6cmdiYSgwLDAsMCwuNTUpfQogICAgLmluZm8tY2FyZHttYXJnaW46MDsgcGFkZGluZzouOXJlbSAxcmVtOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi43NXJlbX0KICAgIC5pbmZvLWhlYWR7ZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGp1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOyBnYXA6Ljc1cmVtOyBib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZy1ib3R0b206LjZyZW19CiAgICAuaW5mby10aXRsZXtmb250LXdlaWdodDo2MDB9CiAgICAuaW5mby1ib2R5IGF7Y29sb3I6dmFyKC0tYWNjKTsgdGV4dC1kZWNvcmF0aW9uOm5vbmV9CiAgICAuaW5mby1ib2R5IGE6aG92ZXJ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZX0KICAgIC5pbmZvLWFjdGlvbnN7ZGlzcGxheTpmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7IGdhcDouNXJlbX0KICAgIGhlYWRlciAuc3BhY2Vye2ZsZXg6MX0KICAgIGhlYWRlciBpbnB1dHtiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjQ1cmVtIC41cmVtOyBib3JkZXItcmFkaXVzOi41cmVtOyBvdXRsaW5lOm5vbmU7IHdpZHRoOjIyY2g7IGhlaWdodDozMnB4fQogICAgaGVhZGVyIGlucHV0OmZvY3Vze2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIHNlbGVjdHsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgICBwYWRkaW5nOi40NXJlbSAuNXJlbTsKICAgICAgYm9yZGVyLXJhZGl1czouNXJlbTsKICAgICAgb3V0bGluZTpub25lOwogICAgICB3aWR0aDoyMmNoOwogICAgICBoZWlnaHQ6MzJweDsKICAgICAgYXBwZWFyYW5jZTpub25lOwogICAgfQogICAgaGVhZGVyIHNlbGVjdDpmb2N1c3tib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIGhlYWRlciAuc21hbGx7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKX0KICAgIGhlYWRlciBsYWJlbC5zbWFsbHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4yNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucm93e2Rpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXB9CgogICAgLyogU2Nyb2xsLVZpZXdwb3J0IChzbW9vdGggbnVyIGbDvHIgVXNlci1TY3JvbGwsIHByb2dyYW1tYXRpc2NoIGVyendpbmdlbiB3aXIgJ2F1dG8nKSAqLwogICAgbWFpbiNzY3JvbGxlcntvdmVyZmxvdzphdXRvOyBwYWRkaW5nOjFyZW0gMXJlbSAxLjVyZW07IHNjcm9sbC1iZWhhdmlvcjpzbW9vdGh9CiAgICAubG9ne21heC13aWR0aDo5MDBweDsgbWFyZ2luOjFyZW0gYXV0byAwOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOjFyZW19CiAgICAubXNne2Rpc3BsYXk6ZmxleDsgZ2FwOi43NXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIC5hdmF0YXJ7d2lkdGg6MjhweDsgaGVpZ2h0OjI4cHg7IGRpc3BsYXk6Z3JpZDsgcGxhY2UtaXRlbXM6Y2VudGVyOyBib3JkZXItcmFkaXVzOjhweDsgZmxleDowIDAgMjhweH0KICAgIC51c2VyIC5hdmF0YXJ7YmFja2dyb3VuZDojMjAzMTRhfQogICAgLmJvdCAuYXZhdGFye2JhY2tncm91bmQ6IzFiMmQyMH0KICAgIC5zeXMgLmF2YXRhcntiYWNrZ3JvdW5kOiMyYTIyM2J9CiAgICAKICAgIC8qIFdyYXBwZXIgZsO8ciBkZW4gTmFjaHJpY2h0ZW5pbmhhbHQgKi8KICAgIC5tZXNzYWdlLWNvbnRlbnQtd3JhcHBlciB7CiAgICAgICAgZGlzcGxheTpmbGV4OwogICAgICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICAgICAgICBnYXA6IDAuNXJlbTsgLyogQWJzdGFuZCB6d2lzY2hlbiBEZW5rcHJvemVzcyB1bmQgQW50d29ydCAqLwogICAgICAgIHdpZHRoOiAxMDAlOwogICAgfQoKICAgIC5idWJibGV7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6Ljc1cmVtIC45cmVtOyBib3JkZXItcmFkaXVzOjEycHg7IG1heC13aWR0aDoxMDAlOyBvdmVyZmxvdy13cmFwOmFueXdoZXJlfQogICAgLm1ldGF7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKTsgbWFyZ2luLWJvdHRvbTouMjVyZW19CgogICAgZm9vdGVye2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi42cmVtIC44cmVtfQogICAgLmJhcnttYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW19CiAgICB0ZXh0YXJlYXt3aWR0aDoxMDAlOyByZXNpemU6dmVydGljYWw7IG1pbi1oZWlnaHQ6NjBweDsgbWF4LWhlaWdodDo0MHZoOyBiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjdyZW0gLjc1cmVtOyBib3JkZXItcmFkaXVzOi43NXJlbTsgb3V0bGluZTpub25lfQogICAgdGV4dGFyZWE6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CiAgICBidXR0b257YmFja2dyb3VuZDojMTAxODI3OyBjb2xvcjp2YXIoLS1pbmspOyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi41NXJlbSAuODVyZW07IGJvcmRlci1yYWRpdXM6LjZyZW07IGN1cnNvcjpwb2ludGVyfQogICAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIGJ1dHRvbntoZWlnaHQ6MzJweDsgZGlzcGxheTppbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6Y2VudGVyfQogICAgLnNlY29uZGFyeXtiYWNrZ3JvdW5kOiMwYzBmMTV9CiAgICAubXV0ZWR7Y29sb3I6dmFyKC0tbXV0ZWQpfQogICAgLmhpZGRlbntkaXNwbGF5Om5vbmV9CiAgICAuaGludHtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXJnaW4tbGVmdDouMjVyZW19CiAgICAuZHJvcHtkaXNwbGF5Om5vbmU7IGJvcmRlcjoycHggZGFzaGVkICMyYTMzNDY7IGJvcmRlci1yYWRpdXM6MTJweDsgcGFkZGluZzoxMnB4OyB0ZXh0LWFsaWduOmNlbnRlcjsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjoxcmVtIGF1dG8gLjc1cmVtO30KICAgIC5kcm9wLmRyYWd7YmFja2dyb3VuZDojMGUxMzIwOyBjb2xvcjp2YXIoLS1pbmspfQogICAgY29kZSwgcHJlLCBrYmR7Zm9udC1mYW1pbHk6IHVpLW1vbm9zcGFjZSwgU0ZNb25vLVJlZ3VsYXIsIE1lbmxvLCBDb25zb2xhcywgbW9ub3NwYWNlfQoKICAgIC8qIE1hcmtkb3duIHJlbmRlcmluZyAqLwogICAgLm1ke2xpbmUtaGVpZ2h0OjEuNn0KICAgIC5tZCBoMSwubWQgaDIsLm1kIGgze2xpbmUtaGVpZ2h0OjEuMjU7IG1hcmdpbjouMnJlbSAwIC42cmVtfQogICAgLm1kIGgxe2ZvbnQtc2l6ZToxLjM1cmVtfQogICAgLm1kIGgye2ZvbnQtc2l6ZToxLjJyZW19CiAgICAubWQgaDN7Zm9udC1zaXplOjEuMDVyZW19CiAgICAubWQgcHttYXJnaW46LjVyZW0gMH0KICAgIC5tZCB1bCwubWQgb2x7bWFyZ2luOi41cmVtIDAgLjVyZW0gMS4xcmVtfQogICAgLm1kIGxpe21hcmdpbjouMnJlbSAwfQogICAgLm1kIGJsb2NrcXVvdGV7bWFyZ2luOi42cmVtIDA7IHBhZGRpbmc6LjI1cmVtIC44cmVtOyBib3JkZXItbGVmdDozcHggc29saWQgIzMwNDQ1ZjsgYmFja2dyb3VuZDojMGMxMTFiOyBjb2xvcjojZDZlMmYwOyBib3JkZXItcmFkaXVzOjZweH0KICAgIC5tZCBocntib3JkZXI6bm9uZTsgYm9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7IG1hcmdpbjouOHJlbSAwfQogICAgLm1kIGNvZGV7YmFja2dyb3VuZDojMGMwZjE1OyBwYWRkaW5nOi4xMnJlbSAuMzJyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6LjM1cmVtfQogICAgLm1kIHByZXtiYWNrZ3JvdW5kOiMwYzBmMTU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgcGFkZGluZzouOHJlbTsgb3ZlcmZsb3c6YXV0bzsgcG9zaXRpb246cmVsYXRpdmV9CiAgICAubWQgcHJlPmNvZGV7ZGlzcGxheTpibG9jazsgd2hpdGUtc3BhY2U6cHJlfQogICAgLm1kIHRhYmxle2JvcmRlci1jb2xsYXBzZTpjb2xsYXBzZTsgbWFyZ2luOi44cmVtIDA7IHdpZHRoOjEwMCU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAubWQgdGgsLm1kIHRke3BhZGRpbmc6LjVyZW0gLjdyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHRleHQtYWxpZ246bGVmdH0KICAgIC5tZCB0aHtiYWNrZ3JvdW5kOiMwZjE3MjU7IGZvbnQtd2VpZ2h0OjYwMH0KICAgIC5jb2Rld3JhcHtwb3NpdGlvbjpyZWxhdGl2ZX0KICAgIC5jb3B5YnRue3Bvc2l0aW9uOmFic29sdXRlOyB0b3A6LjQ1cmVtOyByaWdodDouNDVyZW07IGZvbnQtc2l6ZToxMnB4OyBwYWRkaW5nOi4yNXJlbSAuNDVyZW07IGJvcmRlci1yYWRpdXM6LjRyZW07IGJhY2tncm91bmQ6IzBmMTcyNTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgY3Vyc29yOnBvaW50ZXJ9CiAgICAuY29weWJ0bjpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIC50YXNrLWRvbmV7b3BhY2l0eTouOX0KCiAgICAvKiDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4ggc3R5bGluZyAqLwogICAgLnN5c2lucHV0e3dpZHRoOjEwMCU7IHJlc2l6ZTp2ZXJ0aWNhbDsgbWluLWhlaWdodDo2MHB4OyBtYXgtaGVpZ2h0OjMwdmg7IGJhY2tncm91bmQ6IzBjMGYxNTsgY29sb3I6dmFyKC0taW5rKTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZzouN3JlbSAuNzVyZW07IGJvcmRlci1yYWRpdXM6Ljc1cmVtOyBvdXRsaW5lOm5vbmV9CiAgICAuc3lzaW5wdXQ6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CgogICAgLyogSlNPTiBFZGl0b3IgKi8KICAgIGRldGFpbHMucmF3e2Rpc3BsYXk6bm9uZTsgbWF4LXdpZHRoOjkwMHB4OyBtYXJnaW46MXJlbSBhdXRvIDA7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgYmFja2dyb3VuZDojMGMwZjE1fQogICAgZGV0YWlscy5yYXc+c3VtbWFyeXtjdXJzb3I6cG9pbnRlcjsgcGFkZGluZzouNnJlbSAuOHJlbTsgbGlzdC1zdHlsZTpub25lOyB1c2VyLXNlbGVjdDpub25lfQogICAgZGV0YWlscy5yYXc+c3VtbWFyeTo6bWFya2Vye2Rpc3BsYXk6bm9uZX0KICAgIGRldGFpbHMucmF3W29wZW5dPnN1bW1hcnl7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAucmF3IC5pbm5lcntwYWRkaW5nOi44cmVtfQogICAgLnJhdyB0ZXh0YXJlYXtmb250LWZhbWlseTogdWktbW9ub3NwYWNlLCBTRk1vbm8tUmVndWxhciwgTWVubG8sIENvbnNvbGFzLCBtb25vc3BhY2U7IG1pbi1oZWlnaHQ6MjIwcHg7IHdoaXRlLXNwYWNlOnByZTsgdGFiLXNpemU6Mn0KICAgIC5yYXcgLnJvd3tkaXNwbGF5OmZsZXg7IGdhcDouNXJlbTsgYWxpZ24taXRlbXM6Y2VudGVyOyBtYXJnaW4tdG9wOi41cmVtOyBmbGV4LXdyYXA6d3JhcH0KICAgIC5iYWRnZXtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpfQoKICAgIC8qIFNjcm9sbC10by1ib3R0b20gYnV0dG9uICovCiAgICAudG8tYm90dG9te3Bvc2l0aW9uOmZpeGVkOyByaWdodDoxNnB4OyBib3R0b206MTZweDsgcGFkZGluZzouNXJlbSAuN3JlbTsgYm9yZGVyLXJhZGl1czouNnJlbTsgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBjb2xvcjp2YXIoLS1pbmspOyBkaXNwbGF5Om5vbmU7IG9wYWNpdHk6LjkyfQogICAgLnRvLWJvdHRvbS5zaG93e2Rpc3BsYXk6YmxvY2t9CiAgICAKICAgIC8qIFNUWUxFUyBGw5xSIFRISU5LSU5HLUJPWCAqLwogICAgLnRoaW5rLWJ1YmJsZSB7CiAgICAgICAgd2lkdGg6IDEwMCU7CiAgICB9CiAgICAudGhpbmstYnViYmxlIGRldGFpbHMgewogICAgICAgIGJhY2tncm91bmQ6ICMxYTE2MjU7CiAgICAgICAgYm9yZGVyOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBib3JkZXItcmFkaXVzOiAxMHB4OwogICAgICAgIGZvbnQtc2l6ZTogMTNweDsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeSB7CiAgICAgICAgY3Vyc29yOiBwb2ludGVyOwogICAgICAgIHBhZGRpbmc6IC41cmVtIC44cmVtOwogICAgICAgIGZvbnQtd2VpZ2h0OiA1MDA7CiAgICAgICAgY29sb3I6IHZhcigtLW11dGVkKTsKICAgICAgICB1c2VyLXNlbGVjdDogbm9uZTsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeTpob3ZlciB7CiAgICAgICAgY29sb3I6IHZhcigtLWluayk7CiAgICB9CiAgICAudGhpbmstYnViYmxlIC50aGluay1jb250ZW50IHsKICAgICAgICBwYWRkaW5nOiAuMXJlbSAuOHJlbSAuN3JlbTsKICAgICAgICBib3JkZXItdG9wOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBjb2xvcjogI2MwYjhkMTsKICAgIH0KCS8qIOKAlCBQb2xpc2hpbmc6IFNhbXBsaW5nLUlucHV0cyBpbSBFaW5zdGVsbHVuZ3MtUGFuZWwg4oCUICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCB7ICAgICAgICAgICAgICAgLyogTGFiZWwgKyBGZWxkIHZlcnRpa2FsLCBrb21wYWt0ICovCiAgZGlzcGxheTogZmxleDsKICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwogIGdhcDogLjI1cmVtOwogIGFsaWduLWl0ZW1zOiBmbGV4LXN0YXJ0Owp9CgovKiBFaW5oZWl0bGljaGUsIHNjaG1hbGUgTnVtYmVyLUlucHV0cyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIHdpZHRoOiAxMGNoOyAgICAgICAgICAgICAgICAgICAgICAgIC8qIGZpeGUgQnJlaXRlIC0+IGdsZWljaGUgS2FudGUgKi8KICBoZWlnaHQ6IDMycHg7CiAgcGFkZGluZzogLjM1cmVtIC41cmVtOwogIGJhY2tncm91bmQ6ICMwYzBmMTU7CiAgY29sb3I6IHZhcigtLWluayk7CiAgYm9yZGVyOiAxcHggc29saWQgdmFyKC0tbGluZSk7CiAgYm9yZGVyLXJhZGl1czogLjVyZW07CiAgdGV4dC1hbGlnbjogcmlnaHQ7CiAgZm9udC12YXJpYW50LW51bWVyaWM6IHRhYnVsYXItbnVtczsgLyogc2F1YmVyZSwgYXVzZ2VyaWNodGV0ZSBaaWZmZXJuICovCiAgb3V0bGluZTogbm9uZTsKfQoKLyogRm9rdXMtRmVlZGJhY2sgcGFzc2VuZCB6dSBkZW4gcmVzdGxpY2hlbiBDb250cm9scyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Zm9jdXMgewogIGJvcmRlci1jb2xvcjogdmFyKC0tYWNjKTsKICBib3gtc2hhZG93OiAwIDAgMCAycHggcmdiYSgxMDYsMTYxLDI1NSwuMTUpOwp9CgovKiBEZXplbnRlIFBsYWNlaG9sZGVyLUZhcmJlICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6cGxhY2Vob2xkZXIgewogIGNvbG9yOiB2YXIoLS1tdXRlZCk7CiAgb3BhY2l0eTogLjk7Cn0KCi8qIFNwaW5uZXIgYmVpIE51bWJlci1JbnB1dHMgYXVzYmxlbmRlbiAobnVyIGltIFNldHRpbmdzLVBhbmVsKSAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Oi13ZWJraXQtb3V0ZXItc3Bpbi1idXR0b24sCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6LXdlYmtpdC1pbm5lci1zcGluLWJ1dHRvbiB7CiAgLXdlYmtpdC1hcHBlYXJhbmNlOiBub25lOwogIG1hcmdpbjogMDsKfQojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIC1tb3otYXBwZWFyYW5jZTogdGV4dGZpZWxkOwp9CgovKiBLbGVpbmVzIFR1bmluZyBmw7xyIGRpZSBHcnVwcGU6IGRpY2h0ZXIgenVzYW1tZW4sIGJyaWNodCBzYXViZXIgdW0gKi8KI3NldHRpbmdzUGFuZWwgPiAuYmFyID4gZGl2Omxhc3QtY2hpbGQsCiNzZXR0aW5nc1BhbmVsIC5iYXIgPiBkaXY6bGFzdC1jaGlsZCB7CiAgZ2FwOiAuNXJlbSAuNzVyZW07ICAgICAgICAgICAgICAgICAgLyogWmVpbGVuL1NwYWx0ZW4tR2FwICovCiAgZmxleC13cmFwOiB3cmFwOwp9CgogIAogICAgLyogLS0tLS0tLS0tLSBSZXNwb25zaXZlIC0tLS0tLS0tLS0gKi8KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA5ODBweCl7CiAgICAgIGhlYWRlcntnYXA6LjZyZW19CiAgICAgIGhlYWRlciAuc3BhY2Vye2Rpc3BsYXk6bm9uZX0KICAgICAgaGVhZGVyIGxhYmVsLnNtYWxse21pbi13aWR0aDoyMDBweH0KICAgICAgaGVhZGVyIC5yb3d7d2lkdGg6MTAwJX0KICAgICAgaGVhZGVyICN0b2tlblJhdGV7d2lkdGg6YXV0bzsgdGV4dC1hbGlnbjpsZWZ0ICFpbXBvcnRhbnR9CiAgICB9CiAgICBAbWVkaWEgKG1heC13aWR0aDogNzIwcHgpewogICAgICBoZWFkZXIgbGFiZWwuc21hbGx7ZmxleDoxIDEgMjIwcHh9CiAgICAgIGhlYWRlciBpbnB1dCwgaGVhZGVyIHNlbGVjdHt3aWR0aDoxMDAlfQogICAgICBtYWluI3Njcm9sbGVye3BhZGRpbmc6Ljg1cmVtIC43NXJlbSAxLjI1cmVtfQogICAgICAuYmFye2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgICAgIGZvb3RlciAucm93e2ZsZXgtd3JhcDp3cmFwfQogICAgICB0ZXh0YXJlYXttaW4taGVpZ2h0OjkwcHh9CiAgICAgIC8qIFNldHRpbmdzIHBhbmVsIHN0YWNrICovCiAgICAgICNzZXR0aW5nc1BhbmVse2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgIWltcG9ydGFudH0KICAgICAgI3NldHRpbmdzUGFuZWwgPiBkaXZ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAhaW1wb3J0YW50fQogICAgICAjc2V0dGluZ3NQYW5lbCA+IGRpdiA+IGRpdjpsYXN0LWNoaWxke21pbi13aWR0aDowICFpbXBvcnRhbnQ7IGFsaWduLXNlbGY6c3RhcnQgIWltcG9ydGFudH0KICAgIH0KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA0ODBweCl7CiAgICAgIGhlYWRlciBsYWJlbC5zbWFsbHtmbGV4OjEgMSAxMDAlfQogICAgICBoZWFkZXIgLnJvd3tnYXA6LjRyZW19CiAgICAgIGhlYWRlciBidXR0b24sIGZvb3RlciBidXR0b257cGFkZGluZzouNXJlbSAuNzVyZW19CiAgICB9CgogIAogICAgLyogQXR0YWNobWVudHMgKi8KICAgIC5jaGlwewogICAgICBkaXNwbGF5OmlubGluZS1mbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNDVyZW07CiAgICAgIHBhZGRpbmc6LjM1cmVtIC41NXJlbTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OyBib3JkZXItcmFkaXVzOjk5OXB4OyBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6MTAwJTsKICAgIH0KICAgIC5jaGlwIHNtYWxse2NvbG9yOnZhcigtLW11dGVkKX0KICAgIC5jaGlwIGJ1dHRvbnsKICAgICAgcGFkZGluZzouMTVyZW0gLjM1cmVtOyBib3JkZXItcmFkaXVzOjhweDsKICAgICAgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgfQogICAgLmNoaXAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQoKICAgIC8qIE1hcmtkb3duIGltYWdlcyAqLwogICAgLm1kIGltZ3ttYXgtd2lkdGg6MTAwJTsgaGVpZ2h0OmF1dG87IGJvcmRlci1yYWRpdXM6MTBweDsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KCiAgPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KPGRpdiBjbGFzcz0id3JhcCI+CiAgPGhlYWRlcj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9InRpdGxlIj7wn5KsIOODgeODo+ODg+ODiFVJPC9kaXY+CiAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzcGFjZXIiPjwvZGl2PgogICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+PHNwYW4+44Oi44OH44OrPC9zcGFuPgogICAgICA8c2VsZWN0IGlkPSJtb2RlbCI+CiAgICAgICAgPG9wdGlvbiB2YWx1ZT0iIiBzZWxlY3RlZCBkaXNhYmxlZD7igJQg44Oi44OH44Or44KS6Kqt44G/6L6844G/IOKAlDwvb3B0aW9uPgogICAgICA8L3NlbGVjdD4KICAgIDwvbGFiZWw+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICA8YnV0dG9uIGlkPSJhdXRoQnRuIiBjbGFzcz0ic2Vjb25kYXJ5IiB0aXRsZT0iQVBJIOiqjeiovO+8iEJhc2lj77yJIj7wn5SRPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImluZm9CdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSLmg4XloLEiPuaDheWgsTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICAgIDxkaXYgaWQ9InF1b3RhTGVmdCIgY2xhc3M9InNtYWxsIiBzdHlsZT0id2lkdGg6MTFjaDsgdGV4dC1hbGlnbjpyaWdodDsiIHRpdGxlPSJLYWdnbGUgR1BVIOaui+OCiuaZgumWkyI+PC9kaXY+CiAgICAgICAgPGRpdiBpZD0idG9rZW5SYXRlIiBjbGFzcz0ic21hbGwiIHN0eWxlPSJ3aWR0aDoxMmNoOyB0ZXh0LWFsaWduOnJpZ2h0OyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJvdyIgaWQ9InN0b3BSb3ciIHN0eWxlPSJkaXNwbGF5Om5vbmU7Ij4KICAgICAgPGJ1dHRvbiBpZD0ic3RvcFNlcnZlciIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IuOCteODvOODkOODvOOCkuWBnOatoiI+4pagIOWBnOatojwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9oZWFkZXI+CgogIDxtYWluIGlkPSJzY3JvbGxlciI+CiAgICA8ZGl2IGlkPSJkcm9wIiBjbGFzcz0iZHJvcCI+44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GZ44KL44Gr44Gv44OV44Kh44Kk44Or44KS44OJ44Ot44OD44OXIChKU09OIGluIHRoZSB1c3VhbCA8Y29kZT5be3JvbGUsIGNvbnRlbnR9XTwvY29kZT4gZm9ybWF0KS48YnI+PHNwYW4gY2xhc3M9ImhpbnQiPuOBvuOBn+OBryA8Yj7jgqTjg7Pjg53jg7zjg4g8L2I+IGFib3ZlLjwvc3Bhbj48L2Rpdj4KCiAgICA8ZGV0YWlscyBpZD0icmF3Qm94IiBjbGFzcz0icmF3Ij4KICAgICAgPHN1bW1hcnk+4pyOIEpTT07jgajjgZfjgabjg4Hjg6Pjg4Pjg4jjgpLnt6jpm4YgPHNwYW4gY2xhc3M9ImJhZGdlIj4o5YiH5pu/KTwvc3Bhbj48L3N1bW1hcnk+CiAgICAgIDxkaXYgY2xhc3M9ImlubmVyIj4KICAgICAgICA8dGV4dGFyZWEgaWQ9Impzb25FZGl0b3IiIHNwZWxsY2hlY2s9ImZhbHNlIiBwbGFjZWhvbGRlcj0nWyB7ICJyb2xlIjogInN5c3RlbSIsICJjb250ZW50IjogIi4uLiIgfSwgLi4uIF0nPjwvdGV4dGFyZWE+CiAgICAgICAgPGRpdiBjbGFzcz0icm93Ij4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25BcHBseSI+6YGp55SoICjnva7mj5spPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uQXBwZW5kIiBjbGFzcz0ic2Vjb25kYXJ5Ij7ov73oqJg8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25SZWZyZXNoIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgYvjgonmm7TmlrA8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25QcmV0dHkiIGNsYXNzPSJzZWNvbmRhcnkiPuaVtOW9ojwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBpZD0ianNvbk1pbmlmeSIgY2xhc3M9InNlY29uZGFyeSI+5Zyn57iuPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uVmFsaWRhdGUiIGNsYXNzPSJzZWNvbmRhcnkiPuaknOiovDwvYnV0dG9uPgogICAgICAgICAgPHNwYW4gaWQ9Impzb25TdGF0dXMiIGNsYXNzPSJiYWRnZSI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGV0YWlscz4KCiAgICA8ZGl2IGlkPSJsb2ciIGNsYXNzPSJsb2ciPjwvZGl2PgogIDwvbWFpbj4KCiAgPGZvb3Rlcj4KICAgIDxkaXYgY2xhc3M9ImJhciI+CiAgICAgIDx0ZXh0YXJlYSBpZD0iaW5wdXQiIHBsYWNlaG9sZGVyPSLjg6Hjg4Pjgrvjg7zjgrjjgpLlhaXlipvigKYgKOmAgeS/oTogRW50ZXIsIOaUueihjDogU2hpZnQrRW50ZXIpIj48L3RleHRhcmVhPgogICAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJkaXNwbGF5OmZsZXg7IGdhcDouNnJlbTsgYWxpZ24taXRlbXM6Y2VudGVyIj4KICAgICAgICA8YnV0dG9uIGlkPSJhdHRhY2hCdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSJBdHRhY2ggZmlsZXMiPvCfk448L2J1dHRvbj4KICAgICAgICA8aW5wdXQgaWQ9ImF0dGFjaElucHV0IiB0eXBlPSJmaWxlIiBjbGFzcz0iaGlkZGVuIiBtdWx0aXBsZSAvPgogICAgICAgIDxidXR0b24gaWQ9InNlbmQiPumAgeS/oTwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9InN0b3AiIGNsYXNzPSJzZWNvbmRhcnkiIGRpc2FibGVkPuWBnOatojwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9ImNsZWFyIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgq/jg6rjgqI8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGlkPSJzZXR0aW5nc1RvZ2dsZSIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IlRvZ2dsZSBzZXR0aW5ncyI+4pqZ77iPPC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0JhciIgY2xhc3M9ImJhciBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOi42cmVtOyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0xpc3QiIHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBmbGV4LXdyYXA6d3JhcCI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgaWQ9InNldHRpbmdzUGFuZWwiIGNsYXNzPSJiYXIgaGlkZGVuIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW07IGdyaWQtY29sdW1uOjEvLTE7IHdpZHRoOjEwMCUiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuOCt+OCueODhuODoOODl+ODreODs+ODl+ODiDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFyZWEgaWQ9InN5c3RlbSIgY2xhc3M9InN5c2lucHV0IiByb3dzPSIyIiBwbGFjZWhvbGRlcj0i5Lu75oSPIOKAlCBlLmcuIFlvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC4iPjwvdGV4dGFyZWE+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0ibWluLXdpZHRoOjI2MHB4OyBhbGlnbi1zZWxmOmVuZDsgZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNnJlbTsgZmxleC13cmFwOndyYXAiPgogICAgICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+VGVtcGVyYXR1cmUgPHNwYW4gaWQ9InRlbXBWYWwiIGNsYXNzPSJoaW50Ij4wLjc8L3NwYW4+PC9sYWJlbD4KICAgICAgICAgIDxpbnB1dCBpZD0idGVtcCIgdHlwZT0icmFuZ2UiIG1pbj0iMCIgbWF4PSIyIiBzdGVwPSIwLjEiIHZhbHVlPSIwLjciIHN0eWxlPSJ3aWR0aDoxNjBweCI+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3BfcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj5taW5fcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJtaW5fcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3Bfazxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfayIgdHlwZT0ibnVtYmVyIiBzdGVwPSIxIiBtaW49IjAiIHBsYWNlaG9sZGVyPSIoZGVmYXVsdCkiIHN0eWxlPSJ3aWR0aDo5Y2g7IHRleHQtYWxpZ246cmlnaHQiPgogICAgICAgICAgPC9sYWJlbD4KICAgICAgICAgIDxsYWJlbCBjbGFzcz0ic21hbGwiPnRoaW5raW5nPGJyPgogICAgICAgICAgICA8c2VsZWN0IGlkPSJ0aGlua2luZyIgc3R5bGU9IndpZHRoOjExY2giPgogICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+KGRlZmF1bHQpPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib24iPm9uPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib2ZmIj5vZmY8L29wdGlvbj4KICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICA8L2xhYmVsPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZm9vdGVyPgo8L2Rpdj4KCjxidXR0b24gaWQ9InRvQm90dG9tIiBjbGFzcz0idG8tYm90dG9tIiB0aXRsZT0i5LiA55Wq5LiL44G4Ij7ilrwg5LiA55Wq5LiLPC9idXR0b24+Cgo8ZGlhbG9nIGlkPSJpbmZvRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCfkqwg44OB44Oj44OD44OIVUk8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ic2Vjb25kYXJ5IiB2YWx1ZT0iY2xvc2UiIGFyaWEtbGFiZWw9IkNsb3NlIj7inJU8L2J1dHRvbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1ib2R5Ij4KICAgICAgPGRpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPuODkOODvOOCuOODp+ODszo8L3NwYW4+IDxjb2RlIGlkPSJpbmZvVmVyc2lvbiI+MC4wLjA8L2NvZGU+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZm9udC1zaXplOjEycHgiPuOCueODhuODvOODiOODrOOCuVVJIOKAlCBubyBzZXJ2ZXItc2lkZSBjaGF0IHN0b3JhZ2UuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImluZm8tYWN0aW9ucyI+CiAgICAgIDxidXR0b24gdmFsdWU9ImNsb3NlIj7plonjgZjjgos8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZm9ybT4KPC9kaWFsb2c+Cgo8ZGlhbG9nIGlkPSJhdXRoRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCflJEgQVBJIOiqjeiovDwvZGl2PgogICAgICA8YnV0dG9uIGNsYXNzPSJzZWNvbmRhcnkiIHZhbHVlPSJjbG9zZSIgYXJpYS1sYWJlbD0iQ2xvc2UiPuKclTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJpbmZvLWJvZHkiPgogICAgICA8ZGl2IGNsYXNzPSJtdXRlZCBzbWFsbCIgc3R5bGU9Im1hcmdpbi1ib3R0b206LjZyZW0iPuOCteODvOODkOODvOOBjCBCYXNpYyDoqo3oqLzjgpLopoHmsYLjgZnjgovloLTlkIjjgavlhaXlipvjgZfjgb7jgZnvvIhzZXNzaW9uU3RvcmFnZSDjgavkv53lrZjjg7vjgr/jg5bjgpLplonjgZjjgovjgajmtojjgYjjgb7jgZnvvInjgII8L2Rpdj4KICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbi1ib3R0b206LjM1cmVtIj7jg6bjg7zjgrbjg7w8L2xhYmVsPgogICAgICA8aW5wdXQgaWQ9ImF1dGhVc2VyIiB0eXBlPSJ0ZXh0IiBhdXRvY29tcGxldGU9InVzZXJuYW1lIiB2YWx1ZT0iZXhsMyIgc3R5bGU9IndpZHRoOjEwMCU7IG1hcmdpbi1ib3R0b206LjZyZW0iPgogICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuODkeOCueODr+ODvOODiTwvbGFiZWw+CiAgICAgIDxpbnB1dCBpZD0iYXV0aFBhc3MiIHR5cGU9InBhc3N3b3JkIiBhdXRvY29tcGxldGU9ImN1cnJlbnQtcGFzc3dvcmQiIHN0eWxlPSJ3aWR0aDoxMDAlOyBtYXJnaW4tYm90dG9tOi42cmVtIj4KICAgICAgPGRpdiBpZD0iYXV0aFN0YXR1cyIgY2xhc3M9InNtYWxsIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1hY3Rpb25zIj4KICAgICAgPGJ1dHRvbiBpZD0iYXV0aFNhdmUiIHR5cGU9ImJ1dHRvbiI+5L+d5a2YPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImF1dGhDbGVhciIgdHlwZT0iYnV0dG9uIiBjbGFzcz0ic2Vjb25kYXJ5Ij7op6PpmaQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiB2YWx1ZT0iY2xvc2UiIGNsYXNzPSJzZWNvbmRhcnkiPumWieOBmOOCizwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9mb3JtPgo8L2RpYWxvZz4KCgo8c2NyaXB0PgooKCkgPT4gewogIC8vIEFQSSDjga7mjqXntprlhYjjgpLoh6rli5XliKTlrprjgZnjgovjgILlkIzjgZggSFRNTCDjgpLjg63jg7zjgqvjg6vjgafjgoIgS2FnZ2xlIOOBp+OCguS9v+OBhuOAggogIC8vCiAgLy8gMSkgP2FwaT08dXJsPiDjgYzmmI7npLrjgZXjgozjgabjgYTjgozjgbDjgZ3jgozvvIjmnIDlhKrlhYjvvIkKICAvLyAyKSA/cG9ydD08bj4g44GM44GC44KM44GwIGh0dHA6Ly8xMjcuMC4wLjE6PG4+L3Yx77yI44Ot44O844Kr44OrIHV6dSAvIGxsYW1hLXNlcnZlcu+8iQogIC8vIDMpIGh0dHAocykg44Gn6YWN5L+h44GV44KM44Gm44GE44KL44Gq44KJ5ZCM5LiA44Kq44Oq44K444Oz77yIS2FnZ2xlIOS4ree2me+8iQogIC8vIDQpIGZpbGU6Ly8g44Gn6ZaL44GE44Gf5aC05ZCI44Gv5pei5a6a44Od44O844OICiAgY29uc3QgcGFyYW1zID0gbmV3IFVSTFNlYXJjaFBhcmFtcyh3aW5kb3cubG9jYXRpb24uc2VhcmNoKTsKICBjb25zdCBwb3J0UGFyYW0gPSBwYXJhbXMuZ2V0KCdwb3J0Jyk7CiAgY29uc3QgYXBpUGFyYW0gPSBwYXJhbXMuZ2V0KCdhcGknKTsKCiAgZnVuY3Rpb24gcmVzb2x2ZUFwaUJhc2UoKSB7CiAgICBpZiAoYXBpUGFyYW0pIHJldHVybiBhcGlQYXJhbS5yZXBsYWNlKC9cLyQvLCAnJyk7CiAgICBpZiAocG9ydFBhcmFtKSByZXR1cm4gYGh0dHA6Ly8xMjcuMC4wLjE6JHtwb3J0UGFyYW19L3YxYDsKICAgIC8vIOS4ree2meOChCBuZ3JvayDntYznlLHjgafphY3kv6HjgZXjgozjgabjgYTjgovloLTlkIjjga/lkIzkuIDjgqrjg6rjgrjjg7PjgIIKICAgIC8vIOOBk+OBk+OCkiAxMjcuMC4wLjEg5Zu65a6a44Gr44GZ44KL44Go44CB44Oq44Oi44O844OI44GL44KJ6ZaL44GE44Gf44Go44GN44GrCiAgICAvLyDmiYvlhYPjga4gUEMg44KS6KaL44Gr6KGM44Gj44Gm44GX44G+44GG44CCCiAgICBpZiAobG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwOicgfHwgbG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwczonKSB7CiAgICAgIHJldHVybiBsb2NhdGlvbi5vcmlnaW4gKyAnL3YxJzsKICAgIH0KICAgIHJldHVybiAnaHR0cDovLzEyNy4wLjAuMTo5OTMxL3YxJzsKICB9CgogIGNvbnN0IEFQSV9CQVNFID0gcmVzb2x2ZUFwaUJhc2UoKTsKCiAgLy8gbmdyb2sg54Sh5paZ5p6g44Gv44OW44Op44Km44K244GL44KJ44GuIEFQSSDlj5blvpfjgavjgoLorablkYrjg5rjg7zjgrjjgpLmjJ/jgoDjgZPjgajjgYzjgYLjgovjgIIKICAvLyDjgZPjga7jg5jjg4Pjg4Djga/jg63jg7zjgqvjg6vjgrXjg7zjg5Djg7zjgafjga/nhKHoppbjgZXjgozjgovjga7jgafjgIHluLjjgavku5jjgZHjgabjgoLlrrPjga/jgarjgYTjgIIKICBjb25zdCBCQVNFX0hFQURFUlMgPSB7ICduZ3Jvay1za2lwLWJyb3dzZXItd2FybmluZyc6ICd0cnVlJyB9OwogIGNvbnN0IGVscyA9IHsKICAgIHNjcm9sbGVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Nyb2xsZXInKSwKICAgIGxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xvZycpLAogICAgZHJvcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Ryb3AnKSwKICAgIGlucHV0OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnaW5wdXQnKSwKICAgIHNlbmRCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzZW5kJyksCiAgICBzdG9wQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcCcpLAogICAgY2xlYXJCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbGVhcicpLAogICAgYXR0YWNoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXR0YWNoQnRuJyksCiAgICBhdHRhY2hJbnB1dDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaElucHV0JyksCiAgICBhdHRhY2htZW50c0JhcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaG1lbnRzQmFyJyksCiAgICBhdHRhY2htZW50c0xpc3Q6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdHRhY2htZW50c0xpc3QnKSwKICAgIG1vZGVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbW9kZWwnKSwKICAgIHN5c3RlbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N5c3RlbScpLAogICAgdGVtcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RlbXAnKSwKICAgIHRlbXBWYWw6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0ZW1wVmFsJyksCiAgICBpbXBvcnRCdG46IG51bGwsCiAgICBpbXBvcnRGaWxlOiBudWxsLAogICAgZXhwb3J0QnRuOiBudWxsLAogICAgbG9hZE1vZGVsc0J0bjogbnVsbCwKICAgIG1vZGVsU3RhdHVzOiBudWxsLAogICAgcmF3Qm94OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmF3Qm94JyksCiAgICBqc29uRWRpdG9yOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkVkaXRvcicpLAogICAganNvbkFwcGx5OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGx5JyksCiAgICBqc29uQXBwZW5kOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGVuZCcpLAogICAganNvblJlZnJlc2g6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uUmVmcmVzaCcpLAogICAganNvblByZXR0eTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25QcmV0dHknKSwKICAgIGpzb25NaW5pZnk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uTWluaWZ5JyksCiAgICBqc29uVmFsaWRhdGU6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uVmFsaWRhdGUnKSwKICAgIGpzb25TdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uU3RhdHVzJyksCiAgICB0b0JvdHRvbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RvQm90dG9tJyksCiAgICBzZXR0aW5nc1RvZ2dsZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NldHRpbmdzVG9nZ2xlJyksCiAgICBzZXR0aW5nc1BhbmVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2V0dGluZ3NQYW5lbCcpLAogICAgdG9rZW5SYXRlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9rZW5SYXRlJyksCiAgICBxdW90YUxlZnQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdxdW90YUxlZnQnKSwKICAgIHN0b3BSb3c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdG9wUm93JyksCiAgICBzdG9wU2VydmVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcFNlcnZlcicpLAogICAgaW5mb0J0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9CdG4nKSwKICAgIGluZm9EaWFsb2c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvRGlhbG9nJyksCiAgICBpbmZvVmVyc2lvbjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9WZXJzaW9uJyksCiAgICBpbmZvUHJvamVjdFVybDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9Qcm9qZWN0VXJsJyksCiAgICBpbmZvSW1wcmludDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9JbXByaW50JyksCiAgICBhdXRoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aEJ0bicpLAogICAgYXV0aERpYWxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhEaWFsb2cnKSwKICAgIGF1dGhVc2VyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFVzZXInKSwKICAgIGF1dGhQYXNzOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFBhc3MnKSwKICAgIGF1dGhTdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoU3RhdHVzJyksCiAgICBhdXRoU2F2ZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhTYXZlJyksCiAgICBhdXRoQ2xlYXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoQ2xlYXInKSwKICAgIC8vIE5FVToKICAgIHRvcF9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX3AnKSwKICAgIG1pbl9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWluX3AnKSwKICAgIHRvcF9rOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX2snKSwKICAgIHRoaW5raW5nOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndGhpbmtpbmcnKSwKICB9OwoKICBsZXQgbWVzc2FnZXMgPSBbXTsKICBsZXQgYXR0YWNobWVudHMgPSBbXTsgLy8ge2lkLG5hbWUsdHlwZSxzaXplLCBraW5kOid0ZXh0J3wnaW1hZ2UnfCdvdGhlcicsIHRleHQ/LCBkYXRhVXJsPywgdHJ1bmNhdGVkP30KICBsZXQgbmV4dEF0dGFjaElkID0gMTsKCiAgLyogLS0tLS0tLS0tLSBBUEkg6KqN6Ki877yIQmFzaWPvvIkgLS0tLS0tLS0tLSAqLwogIC8vIOS4ree2meOBjCBBUElfUEFTUyDjgpLoqK3lrprjgZfjgabjgYTjgovjgajjgY3jga8gL3YxLyog44KCIC9zdG9wIOOCgiA0MDEg44Gr44Gq44KL44CCCiAgLy8g6LOH5qC85oOF5aCx44GvIHNlc3Npb25TdG9yYWdlIOOBq+OBoOOBkee9ruOBj++8iGxvY2FsU3RvcmFnZSDjgavjga/mrovjgZXjgarjgYTvvInjgIIKICBjb25zdCBBVVRIX0tFWSA9ICdrY19iYXNpY19hdXRoJzsKICBjb25zdCBBVVRIX1VTRVJfS0VZID0gJ2tjX2Jhc2ljX3VzZXInOwoKICBmdW5jdGlvbiBiNjR1dGY4KHMpewogICAgLy8gUHl0aG9uIOWBtOOBryBVVEYtOCDjgacgYmFzZTY0IOWMluOBl+OBpuOBhOOCi+OBruOBp+OAgWJ0b2Eg55u05ZG844Gz77yITGF0aW4tMe+8ieOBp+OBrwogICAgLy8g6Z2eIEFTQ0lJIOOBruODkeOCueODr+ODvOODieOBp+S4gOiHtOOBl+OBquOBhOOAggogICAgY29uc3QgYnl0ZXMgPSBuZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUocyk7CiAgICBsZXQgYmluID0gJyc7CiAgICBmb3IgKGNvbnN0IGIgb2YgYnl0ZXMpIGJpbiArPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGIpOwogICAgcmV0dXJuIGJ0b2EoYmluKTsKICB9CiAgZnVuY3Rpb24gZ2V0QXV0aEhlYWRlcigpewogICAgY29uc3QgdiA9IHNlc3Npb25TdG9yYWdlLmdldEl0ZW0oQVVUSF9LRVkpOwogICAgcmV0dXJuIHYgPyB7ICdBdXRob3JpemF0aW9uJzogYEJhc2ljICR7dn1gIH0gOiB7fTsKICB9CiAgZnVuY3Rpb24gYXBpSGVhZGVycyhleHRyYSl7CiAgICByZXR1cm4geyAuLi5CQVNFX0hFQURFUlMsIC4uLmdldEF1dGhIZWFkZXIoKSwgLi4uKGV4dHJhIHx8IHt9KSB9OwogIH0KICBmdW5jdGlvbiBzZXRBdXRoQ3JlZGVudGlhbHModXNlciwgcGFzcyl7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfS0VZLCBiNjR1dGY4KGAke3VzZXJ9OiR7cGFzc31gKSk7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfVVNFUl9LRVksIHVzZXIpOwogIH0KICBmdW5jdGlvbiBjbGVhckF1dGhDcmVkZW50aWFscygpewogICAgc2Vzc2lvblN0b3JhZ2UucmVtb3ZlSXRlbShBVVRIX0tFWSk7CiAgICBzZXNzaW9uU3RvcmFnZS5yZW1vdmVJdGVtKEFVVEhfVVNFUl9LRVkpOwogIH0KICBmdW5jdGlvbiBvcGVuQXV0aERpYWxvZyhtc2cpewogICAgaWYgKCFlbHMuYXV0aERpYWxvZykgcmV0dXJuOwogICAgaWYgKGVscy5hdXRoVXNlcikgewogICAgICBlbHMuYXV0aFVzZXIudmFsdWUgPSBzZXNzaW9uU3RvcmFnZS5nZXRJdGVtKEFVVEhfVVNFUl9LRVkpIHx8ICdleGwzJzsKICAgIH0KICAgIGlmIChlbHMuYXV0aFBhc3MpIGVscy5hdXRoUGFzcy52YWx1ZSA9ICcnOwogICAgaWYgKGVscy5hdXRoU3RhdHVzKSBlbHMuYXV0aFN0YXR1cy50ZXh0Q29udGVudCA9IG1zZyB8fCAnJzsKICAgIGlmICh0eXBlb2YgZWxzLmF1dGhEaWFsb2cuc2hvd01vZGFsID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5zaG93TW9kYWwoKTsKICAgIGVsc2UgZWxzLmF1dGhEaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgc2V0VGltZW91dCgoKSA9PiBlbHMuYXV0aFBhc3M/LmZvY3VzKCksIDUwKTsKICB9CiAgZnVuY3Rpb24gY2xvc2VBdXRoRGlhbG9nKCl7CiAgICBpZiAoIWVscy5hdXRoRGlhbG9nKSByZXR1cm47CiAgICBpZiAodHlwZW9mIGVscy5hdXRoRGlhbG9nLmNsb3NlID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5jbG9zZSgpOwogICAgZWxzZSBlbHMuYXV0aERpYWxvZy5yZW1vdmVBdHRyaWJ1dGUoJ29wZW4nKTsKICB9CgogIGZ1bmN0aW9uIGZtdEJ5dGVzKG4pewogICAgaWYgKCFOdW1iZXIuaXNGaW5pdGUobikpIHJldHVybiAnJzsKICAgIGNvbnN0IHUgPSBbJ0InLCdLQicsJ01CJywnR0InXTsgbGV0IGk9MDsgbGV0IHg9bjsKICAgIHdoaWxlICh4Pj0xMDI0ICYmIGk8dS5sZW5ndGgtMSl7IHgvPTEwMjQ7IGkrKzsgfQogICAgcmV0dXJuIGAke3gudG9GaXhlZChpPzE6MCl9ICR7dVtpXX1gOwogIH0KCiAgZnVuY3Rpb24gcmVuZGVyQXR0YWNobWVudHMoKXsKICAgIGNvbnN0IGxpc3QgPSBlbHMuYXR0YWNobWVudHNMaXN0OwogICAgbGlzdC5pbm5lckhUTUwgPSAnJzsKICAgIGlmICghYXR0YWNobWVudHMubGVuZ3RoKXsKICAgICAgZWxzLmF0dGFjaG1lbnRzQmFyLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpOwogICAgICByZXR1cm47CiAgICB9CiAgICBlbHMuYXR0YWNobWVudHNCYXIuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7CgogICAgZm9yIChjb25zdCBhIG9mIGF0dGFjaG1lbnRzKXsKICAgICAgY29uc3QgY2hpcCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgICBjaGlwLmNsYXNzTmFtZSA9ICdjaGlwJzsKCiAgICAgIGNvbnN0IGljb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdzcGFuJyk7CiAgICAgIGljb24udGV4dENvbnRlbnQgPSAoYS5raW5kID09PSAnaW1hZ2UnKSA/ICfwn5a877iPJyA6IChhLmtpbmQgPT09ICd0ZXh0JykgPyAn8J+ThCcgOiAn8J+Tpic7CgogICAgICBjb25zdCBuYW1lID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogICAgICBuYW1lLnN0eWxlLm92ZXJmbG93ID0gJ2hpZGRlbic7CiAgICAgIG5hbWUuc3R5bGUudGV4dE92ZXJmbG93ID0gJ2VsbGlwc2lzJzsKICAgICAgbmFtZS5zdHlsZS53aGl0ZVNwYWNlID0gJ25vd3JhcCc7CiAgICAgIG5hbWUuc3R5bGUubWF4V2lkdGggPSAnNTVjaCc7CiAgICAgIG5hbWUudGV4dENvbnRlbnQgPSBhLm5hbWU7CgogICAgICBjb25zdCBzaXplID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc21hbGwnKTsKICAgICAgc2l6ZS50ZXh0Q29udGVudCA9IGZtdEJ5dGVzKGEuc2l6ZSk7CgogICAgICBjb25zdCByZW1vdmUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdidXR0b24nKTsKICAgICAgcmVtb3ZlLnRpdGxlID0gJ1JlbW92ZSc7CiAgICAgIHJlbW92ZS50ZXh0Q29udGVudCA9ICfinJUnOwogICAgICByZW1vdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGF0dGFjaG1lbnRzID0gYXR0YWNobWVudHMuZmlsdGVyKHggPT4geC5pZCAhPT0gYS5pZCk7CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfSk7CgogICAgICBjaGlwLmFwcGVuZChpY29uLCBuYW1lLCBzaXplLCByZW1vdmUpOwogICAgICBsaXN0LmFwcGVuZENoaWxkKGNoaXApOwogICAgfQogIH0KCiAgY29uc3QgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUgPSAyMDAwMDsKICBjb25zdCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgPSAxMjsKCiAgZnVuY3Rpb24gcmVhZEZpbGVBc1RleHQoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc1RleHQoZmlsZSk7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gcmVhZEZpbGVBc0RhdGFVUkwoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc0RhdGFVUkwoZmlsZSk7CiAgICB9KTsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIGFkZEF0dGFjaG1lbnRzRnJvbUZpbGVMaXN0KGZpbGVMaXN0KXsKICAgIGNvbnN0IGZpbGVzID0gQXJyYXkuZnJvbShmaWxlTGlzdCB8fCBbXSk7CiAgICBpZiAoIWZpbGVzLmxlbmd0aCkgcmV0dXJuOwoKICAgIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcy5zbGljZSgwLCBNYXRoLm1heCgwLCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgLSBhdHRhY2htZW50cy5sZW5ndGgpKSl7CiAgICAgIGNvbnN0IGlkID0gbmV4dEF0dGFjaElkKys7CiAgICAgIGNvbnN0IHR5cGUgPSBmaWxlLnR5cGUgfHwgJyc7CiAgICAgIGNvbnN0IG5hbWUgPSBmaWxlLm5hbWUgfHwgYGZpbGVfJHtpZH1gOwogICAgICBjb25zdCBzaXplID0gZmlsZS5zaXplIHx8IDA7CgogICAgICBpZiAodHlwZS5zdGFydHNXaXRoKCdpbWFnZS8nKSl7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOeUu+WDj+OBr+OCteODneODvOODiOOBleOCjOOBpuOBhOOBvuOBm+OCk++8iC0tbW1wcm9qIOOBquOBl++8ieOAguODhuOCreOCueODiOODleOCoeOCpOODq+OBruOBv+a3u+S7mOWPr+iDveOBp+OBmeOAgmApOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CgogICAgICBjb25zdCBsb29rc1RleHQgPSB0eXBlLnN0YXJ0c1dpdGgoJ3RleHQvJykKICAgICAgICB8fCAvXC4odHh0fG1kfGpzb258anNvbmx8Y3N2fHRzdnxsb2d8eWFtbHx5bWx8cHl8anN8dHN8aHRtbHxjc3N8eG1sfGluaXx0b21sKSQvaS50ZXN0KG5hbWUpOwoKICAgICAgaWYgKGxvb2tzVGV4dCl7CiAgICAgICAgdHJ5ewogICAgICAgICAgbGV0IHRleHQgPSBhd2FpdCByZWFkRmlsZUFzVGV4dChmaWxlKTsKICAgICAgICAgIGxldCB0cnVuY2F0ZWQgPSBmYWxzZTsKICAgICAgICAgIGlmICh0ZXh0Lmxlbmd0aCA+IE1BWF9URVhUX0NIQVJTX1BFUl9GSUxFKXsKICAgICAgICAgICAgdGV4dCA9IHRleHQuc2xpY2UoMCwgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUpOwogICAgICAgICAgICB0cnVuY2F0ZWQgPSB0cnVlOwogICAgICAgICAgfQogICAgICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOid0ZXh0JywgdGV4dCwgdHJ1bmNhdGVkIH0pOwogICAgICAgIH0gY2F0Y2ggewogICAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOODhuOCreOCueODiOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnzogJHtuYW1lfWApOwogICAgICAgIH0KICAgICAgICBjb250aW51ZTsKICAgICAgfQoKICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOidvdGhlcicgfSk7CiAgICB9CgogICAgaWYgKGF0dGFjaG1lbnRzLmxlbmd0aCA+PSBNQVhfVE9UQUxfQVRUQUNITUVOVFMpIHsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOS4iumZkOOBq+mBlOOBl+OBvuOBl+OBnyAoJHtNQVhfVE9UQUxfQVRUQUNITUVOVFN9IGZpbGVzKS5gKTsKICAgIH0KICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgfQoKICBsZXQgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICBjb25zdCBOTCA9ICdcbic7CgogIGNvbnN0IEFQUF9WRVJTSU9OID0gJzEuMDEnOwogIGNvbnN0IFBST0pFQ1RfVVJMID0gJ2h0dHBzOi8vd3d3LmxvY2FsbGlnaHRhaS5jb20vc2N1JzsKICBjb25zdCBJTVBSSU5UX1VSTCA9ICdodHRwczovL3d3dy5zcndhcmUubmV0L2ltcHJpbnQnOwogIAogIGxldCBzdHJlYW1TdGFydFRpbWUgPSAwOwogIGxldCBsYXN0VXBkYXRlVGltZSA9IDA7CiAgbGV0IGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgY29uc3QgVVBEQVRFX0lOVEVSVkFMID0gMjAwOyAvLyBtcwoKICAvKiAtLS0tLS0tLS0tIEF1dG8tU2Nyb2xsIChnZWZsaWNrdCkgLS0tLS0tLS0tLSAqLwogIGxldCBhdXRvc2Nyb2xsID0gdHJ1ZTsKICBsZXQgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICBjb25zdCBCT1RUT01fRVBTID0gODA7CgogIGZ1bmN0aW9uIGF0Qm90dG9tKCl7CiAgICBjb25zdCBzID0gZWxzLnNjcm9sbGVyOwogICAgcmV0dXJuIChzLnNjcm9sbEhlaWdodCAtIHMuc2Nyb2xsVG9wIC0gcy5jbGllbnRIZWlnaHQpIDw9IEJPVFRPTV9FUFM7CiAgfQogIGZ1bmN0aW9uIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKXsKICAgIGNvbnN0IHMgPSBlbHMuc2Nyb2xsZXI7CiAgICBjb25zdCBwcmV2ID0gcy5zdHlsZS5zY3JvbGxCZWhhdmlvcjsKICAgIHMuc3R5bGUuc2Nyb2xsQmVoYXZpb3IgPSAnYXV0byc7CiAgICBzLnNjcm9sbFRvcCA9IHMuc2Nyb2xsSGVpZ2h0OwogICAgcy5zdHlsZS5zY3JvbGxCZWhhdmlvciA9IHByZXYgfHwgJyc7CiAgfQogIGZ1bmN0aW9uIG1heWJlU2Nyb2xsKGZvcmNlPWZhbHNlKXsKICAgIGlmIChmb3JjZSB8fCBhdXRvc2Nyb2xsIHx8IGF0Qm90dG9tKCkpIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGNvbnN0IHNob3cgPSAhYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93Jywgc2hvdyk7CiAgICBpZiAoIXNob3cpIGF1dG9zY3JvbGwgPSB0cnVlOwogIH0KICBlbHMuc2Nyb2xsZXIuYWRkRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgKCkgPT4gewogICAgaWYgKGlzUmVuZGVyaW5nKSByZXR1cm47CiAgICBhdXRvc2Nyb2xsID0gYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93JywgIWF1dG9zY3JvbGwpOwogIH0pOwogIGVscy50b0JvdHRvbS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsgYXV0b3Njcm9sbCA9IHRydWU7IG1heWJlU2Nyb2xsKHRydWUpOyB9KTsKCiAgLyogLS0tLS0tLS0tLSBNYXJrZG93biAtLS0tLS0tLS0tICovCiAgY29uc3QgZXNjID0gcyA9PiBTdHJpbmcocykucmVwbGFjZSgvWyY8PiInXS9nLCBjID0+ICh7JyYnOicmYW1wOycsJzwnOicmbHQ7JywnPic6JyZndDsnLCciJzonJnF1b3Q7JywiJyI6JyYjMzk7J31bY10pKTsKICBmdW5jdGlvbiBpbmxpbmVNZChzKXsKICAgIC8vIFByb3RlY3QgaW5saW5lIGNvZGUgd2l0aCBwbGFjZWhvbGRlcnMgZmlyc3QKICAgIGNvbnN0IGNvZGVQbGFjZWhvbGRlcnMgPSBbXTsKICAgIHMgPSBzLnJlcGxhY2UoL2AoW15gXSspYC9nLCAoXyxwKSA9PiB7IGNvbnN0IGlkeCA9IGNvZGVQbGFjZWhvbGRlcnMubGVuZ3RoOyBjb2RlUGxhY2Vob2xkZXJzLnB1c2gocCk7IHJldHVybiBgXHgwMENPREUke2lkeH1ceDAwYDsgfSk7CiAgICAvLyBJbWFnZXMKICAgIC8vIOWklumDqOeUu+WDj+OBr+iHquWLleiqreOBv+i+vOOBv+OBl+OBquOBhO+8iOmWsuimp+OBrui/vei3oeOBq+S9v+OBiOOCi++8ieOAgmh0dHBzL2h0dHAg44Gu44G/44Oq44Oz44Kv44Go44GX44Gm5Ye644GZ44CCCiAgICBzID0gcy5yZXBsYWNlKC8hXFsoW15cXV0qKVxdXCgoaHR0cHM/OlteKVxzXSspXCkvZywgKF8sYWx0LHVybCk9Pic8YSBocmVmPSInK3VybCsnIiB0YXJnZXQ9Il9ibGFuayIgcmVsPSJub29wZW5lciBub3JlZmVycmVyIj5b55S75YOPOiAnKyhhbHR8fHVybCkrJ108L2E+Jyk7CiAgICAvLyBCb2xkCiAgICBzID0gcy5yZXBsYWNlKC9cKlwqKFteKl0rKVwqXCovZywgKF8scCk9Pic8c3Ryb25nPicrcCsnPC9zdHJvbmc+Jyk7CiAgICAvLyBJdGFsaWMgKG9ubHkgb3V0c2lkZSBjb2RlKQogICAgcyA9IHMucmVwbGFjZSgvXCooW14qXSspXCovZywgKF8scCk9Pic8ZW0+JytwKyc8L2VtPicpOwogICAgLy8gTGlua3MKICAgIHMgPSBzLnJlcGxhY2UoL1xbKFteXF1dKylcXVwoKGh0dHBzPzpbXlwpXHNdKylbXlwpXSpcKS9nLCAoXyx0eHQsdXJsKT0+JzxhIGhyZWY9IicrdXJsKyciIHRhcmdldD0iX2JsYW5rIiByZWw9Im5vb3BlbmVyIj4nK3R4dCsnPC9hPicpOwogICAgLy8gQXV0by1saW5rIGJhcmUgVVJMcyAoc2tpcCBhbHJlYWR5LWdlbmVyYXRlZCBIVE1MIHRhZ3MpCiAgICBzID0gcy5yZXBsYWNlKC8oPFtePl0qPil8KFxiaHR0cHM/OlwvXC9bXlxzKe+8ieOAjeOAj+OAguOAgV0rKS9nLCAobWF0Y2gsIHRhZywgdXJsKSA9PiB7CiAgICAgIGlmICh0YWcpIHJldHVybiB0YWc7CiAgICAgIHJldHVybiAnPGEgaHJlZj0iJyt1cmwrJyIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPicrdXJsKyc8L2E+JzsKICAgIH0pOwogICAgLy8gUmVzdG9yZSBjb2RlCiAgICBzID0gcy5yZXBsYWNlKC9ceDAwQ09ERShcZCspXHgwMC9nLCAoXyxpZHgpID0+ICc8Y29kZT4nK2NvZGVQbGFjZWhvbGRlcnNbcGFyc2VJbnQoaWR4KV0rJzwvY29kZT4nKTsKICAgIHJldHVybiBzOwogIH0KICBmdW5jdGlvbiBtZFRvSHRtbChtZCl7CiAgICBtZCA9IFN0cmluZyhtZCA/PyAnJyk7CiAgICBtZCA9IG1kLnJlcGxhY2UoL1xyL2csJycpOwogICAgY29uc3QgbGluZXMgPSBtZC5zcGxpdCgnXG4nKTsKICAgIGxldCBodG1sID0gJycsIGluQ29kZSA9IGZhbHNlLCBjb2RlTGFuZyA9ICcnLCBsaXN0VHlwZSA9IG51bGw7CiAgICBjb25zdCBjbG9zZUxpc3QgPSAoKSA9PiB7IGlmIChsaXN0VHlwZSkgeyBodG1sICs9IGA8LyR7bGlzdFR5cGV9PmA7IGxpc3RUeXBlPW51bGw7IH0gfTsKICAgIGZvciAobGV0IGk9MDtpPGxpbmVzLmxlbmd0aDtpKyspewogICAgICBsZXQgbGluZSA9IGxpbmVzW2ldOwogICAgICBpZiAoIWluQ29kZSAmJiAvXlxzKmBgYC8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgbSA9IGxpbmUubWF0Y2goL15ccypgYGBccyooW1x3Ky1dKikvKTsKICAgICAgICBjb2RlTGFuZyA9IG0gJiYgbVsxXSA/IG1bMV0gOiAnJzsKICAgICAgICBodG1sICs9IGA8ZGl2IGNsYXNzPSJjb2Rld3JhcCI+PHByZT48Y29kZSBkYXRhLWxhbmc9IiR7ZXNjKGNvZGVMYW5nKX0iPmA7CiAgICAgICAgaW5Db2RlID0gdHJ1ZTsgY29udGludWU7CiAgICAgIH0KICAgICAgaWYgKGluQ29kZSl7CiAgICAgICAgaWYgKC9eXHMqYGBgXHMqJC8udGVzdChsaW5lKSl7IGh0bWwgKz0gYDwvY29kZT48L3ByZT48YnV0dG9uIGNsYXNzPSJjb3B5YnRuIj5Db3B5PC9idXR0b24+PC9kaXY+YDsgaW5Db2RlPWZhbHNlOyBjb250aW51ZTsgfQogICAgICAgIGh0bWwgKz0gZXNjKGxpbmUpICsgJ1xuJzsgY29udGludWU7CiAgICAgIH0KICAgICAgY29uc3QgaCA9IGxpbmUubWF0Y2goL15ccyooI3sxLDZ9KVxzKyguKikkLyk7CiAgICAgIGlmIChoKXsgY2xvc2VMaXN0KCk7IGNvbnN0IGx2bD1oWzFdLmxlbmd0aDsgaHRtbCArPSBgPGgke2x2bH0+JHtpbmxpbmVNZChlc2MoaFsyXSkpfTwvaCR7bHZsfT5gOyBjb250aW51ZTsgfQogICAgICBpZiAoL15ccyooPzotLS18XCpcKlwqfF9fXylccyokLy50ZXN0KGxpbmUpKXsgY2xvc2VMaXN0KCk7IGh0bWwgKz0gJzxoci8+JzsgY29udGludWU7IH0KICAgICAgaWYgKC9eXHMqPlxzPy8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgcXVvdGVMaW5lcyA9IFtdOwogICAgICAgIHdoaWxlIChpIDwgbGluZXMubGVuZ3RoICYmIC9eXHMqPlxzPy8udGVzdChsaW5lc1tpXSkpIHsKICAgICAgICAgIHF1b3RlTGluZXMucHVzaChsaW5lc1tpXS5yZXBsYWNlKC9eXHMqPlxzPy8sICcnKSk7CiAgICAgICAgICBpKys7CiAgICAgICAgfQogICAgICAgIGktLTsKICAgICAgICBjb25zdCBxdW90ZVRleHQgPSBpbmxpbmVNZChlc2MocXVvdGVMaW5lcy5qb2luKCdcbicpKSk7CiAgICAgICAgaHRtbCArPSBgPGJsb2NrcXVvdGU+PHA+JHtxdW90ZVRleHQucmVwbGFjZSgvXG4vZywgJzxiciAvPicpfTwvcD48L2Jsb2NrcXVvdGU+YDsKICAgICAgICBjb250aW51ZTsKICAgICAgfQogICAgICAKICAgICAgLy8gLS0tIFNUQVJUOiBUYWJsZSBQYXJzaW5nIC0tLQogICAgICBjb25zdCBpc1RhYmxlTGluZSA9IChsKSA9PiBsLmluY2x1ZGVzKCd8Jyk7CiAgICAgIGNvbnN0IGlzU2VwYXJhdG9yTGluZSA9IChsKSA9PiAvXlxzKlx8PyhbLTpdK1x8KStbLTpdK1x8P1xzKiQvLnRlc3QobC50cmltKCkpOwogICAgICAKICAgICAgaWYgKGlzVGFibGVMaW5lKGxpbmUpICYmIGkgKyAxIDwgbGluZXMubGVuZ3RoICYmIGlzU2VwYXJhdG9yTGluZShsaW5lc1tpICsgMV0pKSB7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgaHRtbCArPSAnPHRhYmxlPic7CiAgICAgICAgCiAgICAgICAgLy8gSGVhZGVyCiAgICAgICAgY29uc3QgaGVhZGVycyA9IGxpbmUuc3BsaXQoJ3wnKS5tYXAoaCA9PiBoLnRyaW0oKSkuc2xpY2UoMSwgLTEpOwogICAgICAgIGh0bWwgKz0gJzx0aGVhZD48dHI+JzsKICAgICAgICBoZWFkZXJzLmZvckVhY2goaCA9PiB7IGh0bWwgKz0gYDx0aD4ke2lubGluZU1kKGVzYyhoKSl9PC90aD5gOyB9KTsKICAgICAgICBodG1sICs9ICc8L3RyPjwvdGhlYWQ+JzsKICAgICAgICAKICAgICAgICBpICs9IDI7IC8vIFNraXAgaGVhZGVyIHVuZCBzZXBhcmF0b3IgbGluZQogICAgICAgIAogICAgICAgIC8vIEJvZHkKICAgICAgICBodG1sICs9ICc8dGJvZHk+JzsKICAgICAgICB3aGlsZSAoaSA8IGxpbmVzLmxlbmd0aCAmJiBpc1RhYmxlTGluZShsaW5lc1tpXSkpIHsKICAgICAgICAgICAgY29uc3QgY2VsbHMgPSBsaW5lc1tpXS5zcGxpdCgnfCcpLm1hcChjID0+IGMudHJpbSgpKS5zbGljZSgxLCAtMSk7CiAgICAgICAgICAgIGlmIChjZWxscy5sZW5ndGggPiAwKSB7CiAgICAgICAgICAgICAgICBodG1sICs9ICc8dHI+JzsKICAgICAgICAgICAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgaGVhZGVycy5sZW5ndGg7IGorKykgewogICAgICAgICAgICAgICAgICAgIGNvbnN0IGNlbGxDb250ZW50ID0gY2VsbHNbal0gfHwgJyc7CiAgICAgICAgICAgICAgICAgICAgaHRtbCArPSBgPHRkPiR7aW5saW5lTWQoZXNjKGNlbGxDb250ZW50KSl9PC90ZD5gOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaHRtbCArPSAnPC90cj4nOwogICAgICAgICAgICB9CiAgICAgICAgICAgIGkrKzsKICAgICAgICB9CiAgICAgICAgaS0tOwogICAgICAgIAogICAgICAgIGh0bWwgKz0gJzwvdGJvZHk+PC90YWJsZT4nOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CiAgICAgIC8vIC0tLSBFTkQ6IFRhYmxlIFBhcnNpbmcgLS0tCgogICAgICBsZXQgbTsKICAgICAgaWYgKG0gPSBsaW5lLm1hdGNoKC9eXHMqWy0qK11ccytcWyggfHh8WClcXVxzKyguKikkLykpewogICAgICAgIGNvbnN0IGNoZWNrZWQgPSAveC9pLnRlc3QobVsxXSk7CiAgICAgICAgaWYgKGxpc3RUeXBlIT09J3VsJyl7IGNsb3NlTGlzdCgpOyBsaXN0VHlwZT0ndWwnOyBodG1sICs9ICc8dWw+JzsgfQogICAgICAgIGh0bWwgKz0gYDxsaSBjbGFzcz0iJHtjaGVja2VkPyd0YXNrLWRvbmUnOicnfSI+YCsoY2hlY2tlZD8n4pyFICc6J+KsnCAnKStpbmxpbmVNZChlc2MobVsyXSkpKyc8L2xpPic7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKlstKitdXHMrLy50ZXN0KGxpbmUpKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0ndWwnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSd1bCc7IGh0bWwgKz0gJzx1bD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKGxpbmUucmVwbGFjZSgvXlxzKlstKitdXHMrLywnJykpKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmIChtID0gbGluZS5tYXRjaCgvXlxzKihcZCspXC5ccysoLiopJC8pKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0nb2wnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSdvbCc7IGh0bWwgKz0gJzxvbD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKG1bMl0pKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKiQvLnRlc3QobGluZS50cmltKCkpKXsgY2xvc2VMaXN0KCk7IH0KICAgICAgZWxzZSB7IGNsb3NlTGlzdCgpOyBodG1sICs9IGA8cD4ke2lubGluZU1kKGVzYyhsaW5lKSl9PC9wPmA7IH0KICAgIH0KICAgIGNsb3NlTGlzdCgpOyByZXR1cm4gaHRtbDsKICB9CiAgZnVuY3Rpb24gZW5oYW5jZUNvZGVDb3B5KGNvbnRhaW5lcil7CiAgICBjb250YWluZXIucXVlcnlTZWxlY3RvckFsbCgnLmNvZGV3cmFwIC5jb3B5YnRuOm5vdChbZGF0YS1pbml0XSknKS5mb3JFYWNoKGJ0bj0+ewogICAgICBidG4uZGF0YXNldC5pbml0ID0gJzEnOwogICAgICBidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGNvbnN0IGNvZGUgPSBidG4ucGFyZW50RWxlbWVudC5xdWVyeVNlbGVjdG9yKCdjb2RlJyk7CiAgICAgICAgY29uc3QgdGV4dCA9IGNvZGUgPyBjb2RlLnRleHRDb250ZW50IDogJyc7CiAgICAgICAgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQodGV4dCkudGhlbigoKT0+eyBidG4udGV4dENvbnRlbnQ9J0NvcGllZCc7IHNldFRpbWVvdXQoKCk9PmJ0bi50ZXh0Q29udGVudD0nQ29weScsMTAwMCk7IH0pOwogICAgICB9KTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSBVSSBoZWxwZXJzIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiByZW5kZXJCdWJibGVDb250ZW50KGVsLCByb2xlLCBjb250ZW50KXsKICAgIGVsLmNsYXNzTGlzdC5hZGQoJ21kJyk7IGVsLmlubmVySFRNTCA9IG1kVG9IdG1sKG5vcm1hbGl6ZUNvbnRlbnQoY29udGVudCkpOyBlbmhhbmNlQ29kZUNvcHkoZWwpOwogIH0KICAKICBmdW5jdGlvbiBhZGRNZXNzYWdlKHJvbGUsIGNvbnRlbnQpIHsKICAgIGNvbnN0IGNscyA9IHJvbGUgPT09ICd1c2VyJyA/ICd1c2VyJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ3N5cycgOiAnYm90JzsKICAgIGNvbnN0IGF2YXRhciA9IHJvbGUgPT09ICd1c2VyJyA/ICfwn6eRJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ+Kame+4jycgOiAn8J+klic7CiAgICBjb25zdCBtc2dFbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgbXNnRWwuY2xhc3NOYW1lID0gYG1zZyAke2Nsc31gOwogICAgbXNnRWwuaW5uZXJIVE1MID0gYAogICAgICA8ZGl2IGNsYXNzPSJhdmF0YXIiPiR7YXZhdGFyfTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXNzYWdlLWNvbnRlbnQtd3JhcHBlciI+CiAgICAgICAgPGRpdiBjbGFzcz0iYnViYmxlIj4KICAgICAgICAgIDxkaXYgY2xhc3M9Im1ldGEiPiR7cm9sZX08L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNvbnRlbnQgbWQiPjwvZGl2PgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj5gOwogICAgZWxzLmxvZy5hcHBlbmRDaGlsZChtc2dFbCk7CiAgICBjb25zdCBjb250ZW50RWwgPSBtc2dFbC5xdWVyeVNlbGVjdG9yKCcuY29udGVudCcpOwogICAgcmVuZGVyQnViYmxlQ29udGVudChjb250ZW50RWwsIHJvbGUsIGNvbnRlbnQpOwogICAgaWYgKCFpc1JlbmRlcmluZykgbWF5YmVTY3JvbGwoKTsKICAgIHJldHVybiBjb250ZW50RWw7CiAgfQogIAogIGZ1bmN0aW9uIGFkZFRoaW5rQnViYmxlKGFzc2lzdGFudENvbnRlbnRFbCkgewogICAgY29uc3Qgd3JhcHBlciA9IGFzc2lzdGFudENvbnRlbnRFbC5jbG9zZXN0KCcubWVzc2FnZS1jb250ZW50LXdyYXBwZXInKTsKICAgIGlmICghd3JhcHBlcikgcmV0dXJuIG51bGw7CiAgICBjb25zdCBidWJibGUgPSB3cmFwcGVyLnF1ZXJ5U2VsZWN0b3IoJy5idWJibGUnKTsKICAgIGlmICghYnViYmxlKSByZXR1cm4gbnVsbDsKCiAgICBjb25zdCB0aGlua0VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7CiAgICB0aGlua0VsLmNsYXNzTmFtZSA9ICd0aGluay1idWJibGUnOwogICAgdGhpbmtFbC5pbm5lckhUTUwgPSBgCiAgICAgIDxkZXRhaWxzPgogICAgICAgIDxzdW1tYXJ5PvCfpJQgU2hvdyB0aGlua2luZzwvc3VtbWFyeT4KICAgICAgICA8ZGl2IGNsYXNzPSJ0aGluay1jb250ZW50IG1kIj48L2Rpdj4KICAgICAgPC9kZXRhaWxzPgogICAgYDsKICAgIHdyYXBwZXIuaW5zZXJ0QmVmb3JlKHRoaW5rRWwsIGJ1YmJsZSk7CiAgICByZXR1cm4gdGhpbmtFbC5xdWVyeVNlbGVjdG9yKCcudGhpbmstY29udGVudCcpOwogIH0KCiAgZnVuY3Rpb24gc2V0U3RyZWFtaW5nU3RhdGUoYWN0aXZlKSB7IGVscy5zZW5kQnRuLmRpc2FibGVkID0gYWN0aXZlOyBlbHMuc3RvcEJ0bi5kaXNhYmxlZCA9ICFhY3RpdmU7IH0KCiAgZnVuY3Rpb24gcmVuZGVyQWxsKCl7CiAgICBpc1JlbmRlcmluZyA9IHRydWU7CiAgICBlbHMubG9nLmlubmVySFRNTCA9ICcnOwogICAgZm9yIChjb25zdCBtIG9mIG1lc3NhZ2VzKSBhZGRNZXNzYWdlKG0ucm9sZSwgbS5jb250ZW50KTsKICAgIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICBhdXRvc2Nyb2xsID0gdHJ1ZTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKCdzaG93Jyk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIE5vcm1hbGl6ZXIgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIG5vcm1hbGl6ZUNvbnRlbnQoYyl7CiAgICBpZiAodHlwZW9mIGMgPT09ICdzdHJpbmcnKSByZXR1cm4gYzsKICAgIGlmIChBcnJheS5pc0FycmF5KGMpKSB7CiAgICAgIHJldHVybiBjLm1hcChwID0+IHsKICAgICAgICBpZiAodHlwZW9mIHAgPT09ICdzdHJpbmcnKSByZXR1cm4gcDsKICAgICAgICBpZiAocD8udHlwZSA9PT0gJ3RleHQnKSByZXR1cm4gcC50ZXh0ID8/ICcnOwogICAgICAgIGlmIChwPy50eXBlID09PSAnaW1hZ2VfdXJsJykgewogICAgICAgICAgY29uc3QgdSA9IHA/LmltYWdlX3VybD8udXJsIHx8ICcnOwogICAgICAgICAgcmV0dXJuIHUgPyBgIVtpbWFnZV0oJHt1fSlgIDogJyc7CiAgICAgICAgfQogICAgICAgIHJldHVybiBwPy50ZXh0ID8/ICcnOwogICAgICB9KS5qb2luKFN0cmluZy5mcm9tQ2hhckNvZGUoMTApKTsKICAgIH0KICAgIGlmIChjICYmIHR5cGVvZiBjID09PSAnb2JqZWN0JyAmJiAndGV4dCcgaW4gYykgcmV0dXJuIFN0cmluZyhjLnRleHQgPz8gJycpOwogICAgdHJ5IHsgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGMpOyB9IGNhdGNoIHsgcmV0dXJuIFN0cmluZyhjKTsgfQogIH0KICBmdW5jdGlvbiBub3JtYWxpemVSb2xlKHIpeyBjb25zdCBzPVN0cmluZyhyfHwnJykudG9Mb3dlckNhc2UoKTsgaWYgKHMuc3RhcnRzV2l0aCgnYXNzaXN0JykpIHJldHVybiAnYXNzaXN0YW50JzsgaWYgKHMuc3RhcnRzV2l0aCgnc3lzJykpIHJldHVybiAnc3lzdGVtJzsgcmV0dXJuICd1c2VyJzsgfQoKICAvKiAtLS0tLS0tLS0tIEV4cG9ydCBpbmtsLiBzeXN0ZW0gLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGJ1aWxkRXhwb3J0QXJyYXkoKXsKICAgIGNvbnN0IG91dCA9IG1lc3NhZ2VzLm1hcChtID0+ICh7IHJvbGU6IG0ucm9sZSwgY29udGVudDogbS5jb250ZW50IH0pKTsKICAgIGNvbnN0IHN5c1RleHQgPSAoZWxzLnN5c3RlbS52YWx1ZSB8fCAnJykudHJpbSgpOwogICAgY29uc3QgZmlyc3RJc1N5c3RlbSA9IG91dFswXT8ucm9sZSA9PT0gJ3N5c3RlbSc7CiAgICBpZiAoc3lzVGV4dCkgewogICAgICBpZiAoIWZpcnN0SXNTeXN0ZW0pIG91dC51bnNoaWZ0KHsgcm9sZTogJ3N5c3RlbScsIGNvbnRlbnQ6IHN5c1RleHQgfSk7CiAgICAgIGVsc2UgaWYgKG91dFswXS5jb250ZW50ICE9PSBzeXNUZXh0KSBvdXRbMF0gPSB7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNUZXh0IH07CiAgICB9CiAgICByZXR1cm4gb3V0OwogIH0KICAgIAogIGZ1bmN0aW9uIHVwZGF0ZVRva2VuUmF0ZShidWZmZXIsIGZvcmNlRmluYWwgPSBmYWxzZSkgewogICAgY29uc3Qgbm93ID0gcGVyZm9ybWFuY2Uubm93KCk7CiAgICBpZiAoIWZvcmNlRmluYWwgJiYgbm93IC0gbGFzdFVwZGF0ZVRpbWUgPCBVUERBVEVfSU5URVJWQUwpIHJldHVybjsKICAgIGxhc3RVcGRhdGVUaW1lID0gbm93OwoKICAgIGNvbnN0IGVsYXBzZWRTZWNvbmRzID0gKG5vdyAtIHN0cmVhbVN0YXJ0VGltZSkgLyAxMDAwOwogICAgaWYgKGVsYXBzZWRTZWNvbmRzIDwgMC4xKSByZXR1cm47CgogICAgbGV0IHRva2VucywgcmF0ZSwgbGFiZWw7CiAgICBpZiAoZmluYWxDb21wbGV0aW9uVG9rZW5zID4gMCkgewogICAgICAgIHRva2VucyA9IGZpbmFsQ29tcGxldGlvblRva2VuczsKICAgICAgICByYXRlID0gdG9rZW5zIC8gZWxhcHNlZFNlY29uZHM7CiAgICAgICAgbGFiZWwgPSBgw5ggJHtyYXRlLnRvRml4ZWQoMSl9IHQvc2A7CiAgICB9IGVsc2UgewogICAgICAgIC8vIFNjaMOkdHp1bmc6IGNhLiA0IFplaWNoZW4gcHJvIFRva2VuCiAgICAgICAgdG9rZW5zID0gTWF0aC5yb3VuZChidWZmZXIubGVuZ3RoIC8gNCk7CiAgICAgICAgcmF0ZSA9IHRva2VucyAvIGVsYXBzZWRTZWNvbmRzOwogICAgICAgIGxhYmVsID0gYH4ke3JhdGUudG9GaXhlZCgxKX0gdC9zYDsKICAgIH0KICAgIGVscy50b2tlblJhdGUudGV4dENvbnRlbnQgPSBsYWJlbDsKICB9CgoKICAvKiAtLS0tLS0tLS0tIEthZ2dsZSBHUFUg5q6L44KK5pmC6ZaTIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3F1b3RhIOOCkuWPqeOBj+OAguODreODvOOCq+ODqyB1enUg44Gr44GvIC9xdW90YSDjgYznhKHjgYTjga7jgafjgIEKICAvLyDjgZ3jga7loLTlkIjjga8gNDA0IOOBjOi/lOOCiuOAgeihqOekuuOBr+WHuuOBquOBhO+8iOOCqOODqeODvOaJseOBhOOBq+OBl+OBquOBhO+8ieOAggogIGxldCBxdW90YVRpbWVyID0gbnVsbDsKCiAgZnVuY3Rpb24gZm10TGVmdChzZWMpIHsKICAgIGlmIChzZWMgPT0gbnVsbCB8fCBzZWMgPCAwKSByZXR1cm4gJyc7CiAgICBjb25zdCBoID0gTWF0aC5mbG9vcihzZWMgLyAzNjAwKTsKICAgIGNvbnN0IG0gPSBNYXRoLmZsb29yKChzZWMgJSAzNjAwKSAvIDYwKTsKICAgIHJldHVybiBg5q6L44KKICR7U3RyaW5nKGgpLnBhZFN0YXJ0KDIsICcwJyl9OiR7U3RyaW5nKG0pLnBhZFN0YXJ0KDIsICcwJyl9YDsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIHJlZnJlc2hRdW90YSgpIHsKICAgIGlmICghZWxzLnF1b3RhTGVmdCkgcmV0dXJuOwogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9xdW90YWAsIHsKICAgICAgICBtZXRob2Q6ICdHRVQnLAogICAgICAgIGhlYWRlcnM6IGFwaUhlYWRlcnMoeyAnQWNjZXB0JzogJ2FwcGxpY2F0aW9uL2pzb24nIH0pLAogICAgICAgIGNhY2hlOiAnbm8tc3RvcmUnLAogICAgICB9KTsKICAgICAgaWYgKCFyZXMub2spIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KICAgICAgY29uc3QgZCA9IGF3YWl0IHJlcy5qc29uKCk7CiAgICAgIGlmICghZC5hdmFpbGFibGUpIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KCiAgICAgIGNvbnN0IGxlZnQgPSBkLmxlZnRTZWNvbmRzOwogICAgICBlbHMucXVvdGFMZWZ0LnRleHRDb250ZW50ID0gZm10TGVmdChsZWZ0KTsKICAgICAgLy8g5q6L44KK44GM5bCR44Gq44GE44Go6Imy44KS5aSJ44GI44KLCiAgICAgIGVscy5xdW90YUxlZnQuc3R5bGUuY29sb3IgPSBsZWZ0IDwgMTgwMCA/ICcjZmY2YjZiJwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDogbGVmdCA8IDcyMDAgPyAnI2UwYTAzMCcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA6ICcnOwogICAgICBlbHMucXVvdGFMZWZ0LnRpdGxlID0gYEthZ2dsZSBHUFU6IOaui+OCiiAke2ZtdExlZnQobGVmdCkucmVwbGFjZSgn5q6L44KKICcsICcnKX1cbmAKICAgICAgICAgICAgICAgICAgICAgICAgICArIGDkvb/nlKggJHtNYXRoLmZsb29yKGQudXNlZFNlY29uZHMgLyAzNjAwKX1oIC8g5LiK6ZmQICR7TWF0aC5mbG9vcihkLnRvdGFsU2Vjb25kcyAvIDM2MDApfWhgCiAgICAgICAgICAgICAgICAgICAgICAgICAgKyAoZC5yZWZyZXNoVGltZSA/IGBcbuODquOCu+ODg+ODiCAke2QucmVmcmVzaFRpbWV9YCA6ICcnKTsKICAgIH0gY2F0Y2ggKF8pIHsKICAgICAgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOwogICAgfQogIH0KCiAgZnVuY3Rpb24gc3RhcnRRdW90YVBvbGxpbmcoKSB7CiAgICByZWZyZXNoUXVvdGEoKTsKICAgIC8vIDEg5YiG44GU44Go44CC5q6L6YeP44Gv56eS5Y2Y5L2N44Gn5rib44KL44GM44CB6KGo56S644Gv5YiG5Y2Y5L2N44Gq44Gu44Gn5Y2B5YiG44CCCiAgICBpZiAocXVvdGFUaW1lcikgY2xlYXJJbnRlcnZhbChxdW90YVRpbWVyKTsKICAgIHF1b3RhVGltZXIgPSBzZXRJbnRlcnZhbChyZWZyZXNoUXVvdGEsIDYwMDAwKTsKICB9CgogIC8qIC0tLS0tLS0tLS0g5YGc5q2i44Oc44K/44Oz77yI44Ot44O844Kr44Or54mI44Gu44G/77yJIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3N0b3Ag44KS5Y+p44GP44CC5Lit57aZ44GM6JC944Gh44KL44Gu44Gn44CB44GT44GuIFVJIOOCguS9v+OBiOOBquOBj+OBquOCi+OAggogIC8vIOODreODvOOCq+ODqyB1enUg44Gr44GvIC9zdG9wIOOBjOeEoeOBhOOBruOBp+OAgTQwNCDjgarjgonjg5zjgr/jg7PjgpLpmqDjgZnjgIIKICBhc3luYyBmdW5jdGlvbiBkZXRlY3RTdG9wU3VwcG9ydCgpIHsKICAgIGlmICghZWxzLnN0b3BSb3cpIHJldHVybjsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKGAke2xvY2F0aW9uLm9yaWdpbn0vcXVvdGFgLCB7CiAgICAgICAgbWV0aG9kOiAnR0VUJywgaGVhZGVyczogYXBpSGVhZGVycygpLCBjYWNoZTogJ25vLXN0b3JlJywKICAgICAgfSk7CiAgICAgIC8vIC9xdW90YSDjgYwgSlNPTiDjgpLov5TjgZkgPSDkuK3ntpnjgYznq4vjgaPjgabjgYTjgosgPSDlgZzmraLjgafjgY3jgovjgIIKICAgICAgLy8gYXZhaWxhYmxlPWZhbHNl77yIdXp1IOOBquOBqeOCr+OCqeODvOOCv+eEoeOBl++8ieOBp+OCguS4ree2meiHquS9k+OBr+eUn+OBjeOBpuOBhOOCi+OBruOBpwogICAgICAvLyDlgZzmraLjg5zjgr/jg7Pjga/lh7rjgZnjgIIKICAgICAgaWYgKHJlcy5vaykgewogICAgICAgIGNvbnN0IGQgPSBhd2FpdCByZXMuanNvbigpOwogICAgICAgIGlmIChkICYmIHR5cGVvZiBkLmF2YWlsYWJsZSA9PT0gJ2Jvb2xlYW4nKSB7CiAgICAgICAgICBlbHMuc3RvcFJvdy5zdHlsZS5kaXNwbGF5ID0gJyc7CiAgICAgICAgfQogICAgICB9CiAgICB9IGNhdGNoIChfKSB7IC8qIOS4ree2meOBp+OBquOBhOOBquOCieWHuuOBleOBquOBhCAqLyB9CiAgfQoKICBhc3luYyBmdW5jdGlvbiBzdG9wU2VydmVyKCkgewogICAgaWYgKCFjb25maXJtKCfjgrXjg7zjg5Djg7zjgpLlgZzmraLjgZfjgb7jgZnjgILjgojjgo3jgZfjgYTjgafjgZnjgYvvvJ9cbu+8iOOBk+OBrueUu+mdouOBr+S9v+OBiOOBquOBj+OBquOCiuOBvuOBme+8iScpKSByZXR1cm47CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgICAgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSB0cnVlOwogICAgICBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICflgZzmraLkuK3igKYnOwogICAgfQogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9zdG9wYCwgewogICAgICAgIG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzOiBhcGlIZWFkZXJzKHsgJ1gtS2MtU3RvcCc6ICcxJyB9KSwgY2FjaGU6ICduby1zdG9yZScsCiAgICAgIH0pOwogICAgICBpZiAocmVzLnN0YXR1cyA9PT0gNDAxKSB7CiAgICAgICAgb3BlbkF1dGhEaWFsb2coJ+WBnOatouOBq+OBr+iqjeiovOOBjOW/heimgeOBp+OBme+8iDQwMe+8iScpOwogICAgICAgIGlmIChlbHMuc3RvcFNlcnZlcikgeyBlbHMuc3RvcFNlcnZlci5kaXNhYmxlZCA9IGZhbHNlOyBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICfilqAg5YGc5q2iJzsgfQogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICBpZiAoIXJlcy5vaykgewogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5YGc5q2iXSDlpLHmlZfjgZfjgb7jgZfjgZ8gKEhUVFAgJHtyZXMuc3RhdHVzfSlgKTsKICAgICAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSBmYWxzZTsgZWxzLnN0b3BTZXJ2ZXIudGV4dENvbnRlbnQgPSAn4pagIOWBnOatoic7IH0KICAgICAgICByZXR1cm47CiAgICAgIH0KICAgIH0gY2F0Y2ggKF8pIHsgLyog5Lit57aZ44GM6JC944Gh44KL44Gu44Gn6YCa5L+h44Ko44Op44O844Gv5oOz5a6a5YaFICovIH0KICAgIHNldFN0YXR1cygn5YGc5q2iJyk7CiAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCAnW+WBnOatol0g44K144O844OQ44O844KS5YGc5q2i44GX44G+44GX44Gf44CC44GT44Gu55S76Z2i44KS6ZaJ44GY44Gm44GP44Gg44GV44GE44CCJyk7CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIGVscy5zdG9wU2VydmVyLnRleHRDb250ZW50ID0gJ+WBnOatoua4iOOBvyc7CiAgfQoKICAvKiAtLS0tLS0tLS0tIEFQSSBCYXNlIHZhbGlkYXRpb24gKE1peGVkIGNvbnRlbnQgZ3VhcmQpIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBpc01peGVkQ29udGVudEFwaUJhc2UoKXsKICAgIGNvbnN0IHBhZ2VQcm90byA9IHdpbmRvdy5sb2NhdGlvbi5wcm90b2NvbDsKICAgIGNvbnN0IGJhc2UgPSAoZWxzLmFwaUJhc2U/LnZhbHVlIHx8ICcnKS50cmltKCk7CiAgICByZXR1cm4gcGFnZVByb3RvID09PSAnaHR0cHM6JyAmJiAvXmh0dHA6XC9cLy9pLnRlc3QoYmFzZSk7CiAgfQogIGZ1bmN0aW9uIGVuc3VyZUFwaUJhc2VBbGxvd2VkKCl7CiAgICBpZiAoaXNNaXhlZENvbnRlbnRBcGlCYXNlKCkpewogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLAogICAgICAgIGBbQ29uZmlnXSBNaXhlZCBjb250ZW50IGJsb2NrZWQ6IHRoaXMgVUkgaXMgbG9hZGVkIHZpYSBIVFRQUywgYnV0IEFQSSBCYXNlIGlzIEhUVFAuXG5gICsKICAgICAgICBgQnJvd3NlcnMgd2lsbCB1c3VhbGx5IGJsb2NrIHRoaXMgcmVxdWVzdC4gVXNlIGFuIGh0dHBzOi8vIEFQSSBCYXNlLCBvciBvcGVuIHRoZSBVSSB2aWEgaHR0cDovLyBvciBmaWxlOi8vLmAKICAgICAgKTsKICAgICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcjYjQwMDAwJzsgfWNhdGNoe30KICAgICAgcmV0dXJuIGZhbHNlOwogICAgfQogICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fQogICAgcmV0dXJuIHRydWU7CiAgfQogIC8vIENsZWFyIHRoZSByZWQgYm9yZGVyIGFzIHNvb24gYXMgdGhlIHVzZXIgZWRpdHMgdGhlIGZpZWxkCiAgaWYgKGVscy5hcGlCYXNlKSB7CiAgICBlbHMuYXBpQmFzZS5hZGRFdmVudExpc3RlbmVyKCdpbnB1dCcsICgpID0+IHsgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fSB9KTsKICB9CgoKICAvKiAtLS0tLS0tLS0tIE1vZGVscyAoR0VUIC92MS9tb2RlbHMpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBsb2FkTW9kZWxzKCkgewogICAgaWYgKCFlbnN1cmVBcGlCYXNlQWxsb3dlZCgpKSByZXR1cm47CiAgICBjb25zdCBiYXNlID0gQVBJX0JBU0UucmVwbGFjZSgvXC8kLywgJycpOwogICAgY29uc3QgdXJsID0gYCR7YmFzZX0vbW9kZWxzYDsKCiAgICBjb25zdCBoZWFkZXJzID0gYXBpSGVhZGVycyh7ICdBY2NlcHQnOiAnYXBwbGljYXRpb24vanNvbicgfSk7CgogICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0xvYWRpbmfigKYnOwogICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IHRydWU7CgogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ0dFVCcsIGhlYWRlcnMsIG1vZGU6ICdjb3JzJyB9KTsKICAgICAgaWYgKHJlcy5zdGF0dXMgPT09IDQwMSkgb3BlbkF1dGhEaWFsb2coJ0FQSSDoqo3oqLzjgYzlv4XopoHjgafjgZnvvIg0MDHvvIknKTsKICAgICAgaWYgKCFyZXMub2spIHsKICAgICAgICBjb25zdCB0ID0gYXdhaXQgcmVzLnRleHQoKS5jYXRjaCgoKT0+ICcnKTsKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdGYWlsZWQgdG8gbG9hZCBtb2RlbHMnfWApOwogICAgICB9CiAgICAgIGNvbnN0IGRhdGEgPSBhd2FpdCByZXMuanNvbigpOwoKICAgICAgLy8gT3BlbkFJLVN0eWxlOiB7IGRhdGE6IFt7aWQ6ICIuLi4ifSwgLi4uXSB9CiAgICAgIC8vIEZhbGxiYWNrcyBmw7xyIGFuZGVyZSBTZXJ2ZXI6CiAgICAgIGNvbnN0IGxpc3QgPQogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8uZGF0YSkgPyBkYXRhLmRhdGEgOgogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8ubW9kZWxzKSA/IGRhdGEubW9kZWxzIDoKICAgICAgICBBcnJheS5pc0FycmF5KGRhdGEpID8gZGF0YSA6CiAgICAgICAgW107CgogICAgICBjb25zdCBpZHMgPSBsaXN0CiAgICAgICAgLm1hcChtID0+ICh0eXBlb2YgbSA9PT0gJ3N0cmluZycgPyBtIDogbT8uaWQpKQogICAgICAgIC5maWx0ZXIoQm9vbGVhbikKICAgICAgICAubWFwKFN0cmluZyk7CgogICAgICBpZiAoIWlkcy5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gbW9kZWxzIGZvdW5kIGluIC92MS9tb2RlbHMuJyk7CgogICAgICBpZHMuc29ydCgoYSxiKT0+YS5sb2NhbGVDb21wYXJlKGIpKTsKCiAgICAgIGNvbnN0IHByZXYgPSBlbHMubW9kZWwudmFsdWU7CiAgICAgIGVscy5tb2RlbC5pbm5lckhUTUwgPSAnJzsKICAgICAgZm9yIChjb25zdCBpZCBvZiBpZHMpIHsKICAgICAgICBjb25zdCBvcHQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdvcHRpb24nKTsKICAgICAgICBvcHQudmFsdWUgPSBpZDsKICAgICAgICBvcHQudGV4dENvbnRlbnQgPSBpZDsKICAgICAgICBlbHMubW9kZWwuYXBwZW5kQ2hpbGQob3B0KTsKICAgICAgfQoKICAgICAgLy8gQXVzd2FobDogdm9yaGVyaWdlIGJlaGFsdGVuIHdlbm4gbcO2Z2xpY2gsIHNvbnN0IGVyc3RlcwogICAgICBlbHMubW9kZWwudmFsdWUgPSBpZHMuaW5jbHVkZXMocHJldikgPyBwcmV2IDogaWRzWzBdOwoKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gYCR7aWRzLmxlbmd0aH0gbW9kZWxzYDsKICAgIH0gY2F0Y2ggKGUpIHsKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0Vycm9yJzsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjg6Ljg4fjg6voqq3jgb/ovrzjgb9dICR7ZS5tZXNzYWdlfWApOwogICAgfSBmaW5hbGx5IHsKICAgICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IGZhbHNlOwogICAgfQogIH0KCiAgLyogLS0tLS0tLS0tLSBTZW5kIChVbnRlcnN0w7x0enQgPHRoaW5rPiB1bmQgInJlYXNvbmluZyIsIG1pdCBQZXJmb3JtYW5jZS1GaXgpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBzZW5kTWVzc2FnZSgpIHsKICAgIGxldCBvdXRFbCA9IG51bGw7CiAgICBsZXQgYXNzaXN0YW50QnVmZmVyID0gJyc7CiAgICBsZXQgdGhpbmtCdWZmZXIgPSAnJzsKICAgIGxldCBzYXZlZFRleHQgPSAnJzsKICAgIGxldCBzYXZlZEF0dGFjaG1lbnRzID0gW107CiAgICB0cnkgewogICAgICBjb25zdCB0ZXh0ID0gZWxzLmlucHV0LnZhbHVlLnRyaW0oKTsKICAgICAgc2F2ZWRUZXh0ID0gdGV4dDsKICAgICAgc2F2ZWRBdHRhY2htZW50cyA9IFsuLi5hdHRhY2htZW50c107CiAgICAgIGlmICghdGV4dCAmJiBhdHRhY2htZW50cy5sZW5ndGggPT09IDApIHJldHVybjsKCiAgICAgIC8vIOOCueODiOODquODvOODn+ODs+OCsOS4reOBr+mAgeS/oeOBl+OBquOBhO+8iOWFpeWKm+OBr+WPr+iDve+8iQogICAgICBpZiAoIWVscy5zdG9wQnRuLmRpc2FibGVkKSByZXR1cm47CgogICAgICBpZiAoIWVuc3VyZUFwaUJhc2VBbGxvd2VkKCkpIHJldHVybjsKCiAgICAgIGlmICghZWxzLm1vZGVsLnZhbHVlKSB7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgJ1vjg5Ljg7Pjg4hdIOODouODh+ODq+acqumBuOaKniDigJQg44Oi44OH44Or44GM6Kqt44G/6L6844G+44KM44Gm44GE44G+44Gb44KT44CC44Oa44O844K444KS44Oq44Ot44O844OJ44GX44Gm44GP44Gg44GV44GE44CCJyk7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIAogICAgICBlbHMudG9rZW5SYXRlLnRleHRDb250ZW50ID0gJyc7CiAgICAgIGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgICAgIHN0cmVhbVN0YXJ0VGltZSA9IDA7CgogICAgICAvLyDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4jjgpLmr47lm57lj43mmKAKICAgICAgY29uc3Qgc3lzVmFsID0gZWxzLnN5c3RlbS52YWx1ZS50cmltKCk7CiAgICAgIGlmIChzeXNWYWwpIHsKICAgICAgICBjb25zdCBleGlzdGluZ1N5c0lkeCA9IG1lc3NhZ2VzLmZpbmRJbmRleChtID0+IG0ucm9sZSA9PT0gJ3N5c3RlbScpOwogICAgICAgIGlmIChleGlzdGluZ1N5c0lkeCA+PSAwKSB7CiAgICAgICAgICBtZXNzYWdlc1tleGlzdGluZ1N5c0lkeF0uY29udGVudCA9IHN5c1ZhbDsKICAgICAgICB9IGVsc2UgaWYgKG1lc3NhZ2VzLmxlbmd0aCA9PT0gMCkgewogICAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNWYWwgfSk7CiAgICAgICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBzeXNWYWwpOwogICAgICAgIH0KICAgICAgfQogICAgICBjb25zdCBoYXNBdHQgPSBhdHRhY2htZW50cy5sZW5ndGggPiAwOwogICAgICBpZiAodGV4dCB8fCBoYXNBdHQpIHsKICAgICAgICBjb25zdCBwYXJ0cyA9IFtdOwogICAgICAgIGlmICh0ZXh0KSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQgfSk7CiAgICAgICAgZWxzZSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQ6ICdBdHRhY2hlZCBmaWxlczonIH0pOwoKICAgICAgICBjb25zdCB0ZXh0RmlsZXMgPSBhdHRhY2htZW50cy5maWx0ZXIoYSA9PiBhLmtpbmQgPT09ICd0ZXh0Jyk7CiAgICAgICAgaWYgKHRleHRGaWxlcy5sZW5ndGgpewogICAgICAgICAgbGV0IGJsb2IgPSAnXG5cbi0tLVxuQXR0YWNoZWQgZmlsZXMgKHZlcmJhdGltKTpcbic7CiAgICAgICAgICBmb3IgKGNvbnN0IGEgb2YgdGV4dEZpbGVzKXsKICAgICAgICAgICAgYmxvYiArPSBgXG5bRmlsZTogJHthLm5hbWV9JHthLnRydW5jYXRlZD8nIChUUlVOQ0FURUQpJzonJ31dXG5cYFxgXGBcbiR7YS50ZXh0fVxuXGBcYFxgXG5gOwogICAgICAgICAgfQogICAgICAgICAgcGFydHMucHVzaCh7IHR5cGU6J3RleHQnLCB0ZXh0OiBibG9iIH0pOwogICAgICAgIH0KCiAgICAgICAgY29uc3QgaW1ncyA9IGF0dGFjaG1lbnRzLmZpbHRlcihhID0+IGEua2luZCA9PT0gJ2ltYWdlJyk7CiAgICAgICAgZm9yIChjb25zdCBhIG9mIGltZ3MpewogICAgICAgICAgaWYgKGEuZGF0YVVybCl7CiAgICAgICAgICAgIHBhcnRzLnB1c2goeyB0eXBlOidpbWFnZV91cmwnLCBpbWFnZV91cmw6eyB1cmw6IGEuZGF0YVVybCB9fSk7CiAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBjb25zdCBjb250ZW50ID0gKHBhcnRzLmxlbmd0aCA9PT0gMSkgPyAocGFydHNbMF0udGV4dCA/PyAnJykgOiBwYXJ0czsKCiAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICd1c2VyJywgY29udGVudCB9KTsKICAgICAgICBhZGRNZXNzYWdlKCd1c2VyJywgY29udGVudCk7CiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gJyc7CgogICAgICAgIGF0dGFjaG1lbnRzID0gW107CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfQoKCiAgICAgIG91dEVsID0gYWRkTWVzc2FnZSgnYXNzaXN0YW50JywgJycpOwoKICAgICAgLy8gTkVVOiBvcHRpb25hbGUgU2FtcGxpbmctUGFyYW1ldGVyIG51ciBtaXRzZW5kZW4sIHdlbm4gZ2VzZXR6dAogICAgICBmdW5jdGlvbiBwYXJzZU9wdGlvbmFsRmxvYXQoZWwsIG1pbiwgbWF4KSB7CiAgICAgICAgY29uc3QgdiA9IChlbD8udmFsdWUgPz8gJycpLnRyaW0oKTsKICAgICAgICBpZiAodiA9PT0gJycpIHJldHVybiBudWxsOwogICAgICAgIGNvbnN0IG51bSA9IHBhcnNlRmxvYXQodik7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1heCA9PT0gJ251bWJlcicgJiYgbnVtID4gbWF4KSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGZ1bmN0aW9uIHBhcnNlT3B0aW9uYWxJbnQoZWwsIG1pbikgewogICAgICAgIGNvbnN0IHYgPSAoZWw/LnZhbHVlID8/ICcnKS50cmltKCk7CiAgICAgICAgaWYgKHYgPT09ICcnKSByZXR1cm4gbnVsbDsKICAgICAgICBjb25zdCBudW0gPSBwYXJzZUludCh2LCAxMCk7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGNvbnN0IGV4dHJhcyA9IHt9OwogICAgICBjb25zdCBfdG9wX3AgPSBwYXJzZU9wdGlvbmFsRmxvYXQoZWxzLnRvcF9wLCAwLCAxKTsKICAgICAgY29uc3QgX21pbl9wID0gcGFyc2VPcHRpb25hbEZsb2F0KGVscy5taW5fcCwgMCwgMSk7CiAgICAgIGNvbnN0IF90b3BfayA9IHBhcnNlT3B0aW9uYWxJbnQoZWxzLnRvcF9rLCAwKTsKICAgICAgaWYgKF90b3BfcCAhPT0gbnVsbCkgZXh0cmFzLnRvcF9wID0gX3RvcF9wOwogICAgICBpZiAoX21pbl9wICE9PSBudWxsKSBleHRyYXMubWluX3AgPSBfbWluX3A7CiAgICAgIGlmIChfdG9wX2sgIT09IG51bGwpIGV4dHJhcy50b3BfayA9IF90b3BfazsKCiAgICAgIC8vIHRoaW5raW5nOiB1enUgcmVhZHMgcmVhc29uaW5nX2VmZm9ydCBwZXIgcmVxdWVzdC4gT3RoZXIgYmFja2VuZHMgaWdub3JlIGl0LgogICAgICBjb25zdCBfdGhpbmtpbmcgPSAoZWxzLnRoaW5raW5nPy52YWx1ZSA/PyAnJykudHJpbSgpOwogICAgICBpZiAoX3RoaW5raW5nID09PSAnb2ZmJykgZXh0cmFzLnJlYXNvbmluZ19lZmZvcnQgPSAnZGlzYWJsZWQnOwogICAgICBlbHNlIGlmIChfdGhpbmtpbmcgPT09ICdvbicpIGV4dHJhcy5yZWFzb25pbmdfZWZmb3J0ID0gJ2RlZmF1bHQnOwoKICAgICAgY29uc3QgYm9keSA9IHsgbW9kZWw6IGVscy5tb2RlbC52YWx1ZS50cmltKCksIG1lc3NhZ2VzLCBzdHJlYW06IHRydWUsIHRlbXBlcmF0dXJlOiBwYXJzZUZsb2F0KGVscy50ZW1wLnZhbHVlKSwgLi4uZXh0cmFzIH07CiAgICAgIGNvbnN0IGhlYWRlcnMgPSBhcGlIZWFkZXJzKHsgJ0NvbnRlbnQtVHlwZSc6ICdhcHBsaWNhdGlvbi9qc29uJywgJ0FjY2VwdCc6ICd0ZXh0L2V2ZW50LXN0cmVhbScgfSk7CiAgICAgIGNvbnN0IHVybCA9IGAke0FQSV9CQVNFLnJlcGxhY2UoL1wvJC8sICcnKX0vY2hhdC9jb21wbGV0aW9uc2A7CgogICAgICBhYm9ydENvbnRyb2xsZXIgPSBuZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKHRydWUpOwoKICAgICAgbGV0IGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICBsZXQgdGhpbmtFbCA9IG51bGw7CgogICAgICAvLyAtLS0gU1RBUlQgw4ROREVSVU5HOiBDT1JTIGhpbnp1Z2Vmw7xndCAtLS0KICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzLCBib2R5OiBKU09OLnN0cmluZ2lmeShib2R5KSwgc2lnbmFsOiBhYm9ydENvbnRyb2xsZXIuc2lnbmFsLCBtb2RlOiAnY29ycycgfSk7CiAgICAgIC8vIC0tLSBFTkRFIMOETkRFUlVORyAtLS0KCiAgICAgIGlmIChyZXMuc3RhdHVzID09PSA0MDEpIG9wZW5BdXRoRGlhbG9nKCdBUEkg6KqN6Ki844GM5b+F6KaB44Gn44GZ77yINDAx77yJJyk7CiAgICAgIGlmICghcmVzLm9rKSB7IGNvbnN0IHQgPSBhd2FpdCByZXMudGV4dCgpLmNhdGNoKCgpPT4gJycpOyB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdTZXJ2ZXIgZXJyb3InfWApOyB9CiAgICAgIAogICAgICBjb25zdCBjdHlwZSA9IChyZXMuaGVhZGVycy5nZXQoJ2NvbnRlbnQtdHlwZScpIHx8ICcnKS50b0xvd2VyQ2FzZSgpOwogICAgICBpZiAoIWN0eXBlLmluY2x1ZGVzKCd0ZXh0L2V2ZW50LXN0cmVhbScpKSB7CiAgICAgICAgY29uc3QgcmVxdWVzdFN0YXJ0VGltZSA9IHBlcmZvcm1hbmNlLm5vdygpOwogICAgICAgIGNvbnN0IHJhdyA9IGF3YWl0IHJlcy50ZXh0KCk7IAogICAgICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0VtcHR5IHJlc3BvbnNlIChzdGF0dXMgT0ssIGJ1dCBubyBjb250ZW50KS4nKTsKICAgICAgICAKICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSByZXF1ZXN0U3RhcnRUaW1lOwogICAgICAgIGxldCBkYXRhOyB0cnl7IGRhdGEgPSBKU09OLnBhcnNlKHJhdyk7fWNhdGNoeyB0aHJvdyBuZXcgRXJyb3IoYFVuZXJ3YXJ0ZXRlcyBGb3JtYXQ6ICR7cmF3LnNsaWNlKDAsMjAwKX0uLi5gKTsgfQogICAgICAgIAogICAgICAgIGxldCBmdWxsQ29udGVudCA9IGRhdGE/LmNob2ljZXM/LlswXT8ubWVzc2FnZT8uY29udGVudCA/PyAnJzsKICAgICAgICBjb25zdCByZWFzb25pbmdDb250ZW50ID0gZGF0YT8uY2hvaWNlcz8uWzBdPy5tZXNzYWdlPy5yZWFzb25pbmdfY29udGVudDsKCiAgICAgICAgaWYgKHJlYXNvbmluZ0NvbnRlbnQpIHsKICAgICAgICAgICAgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgcmVhc29uaW5nQ29udGVudCk7CiAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gcmVhc29uaW5nQ29udGVudDsKICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQudHJpbSgpOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIC8vIEZhbGxiYWNrIGF1ZiA8dGhpbms+LVRhZy1WZXJhcmJlaXR1bmcKICAgICAgICAgICAgY29uc3QgdGhpbmtNYXRjaCA9IGZ1bGxDb250ZW50Lm1hdGNoKC88dGhpbms+KFtcc1xTXSo/KTxcL3RoaW5rPi8pOwogICAgICAgICAgICBpZiAodGhpbmtNYXRjaCkgewogICAgICAgICAgICAgICAgY29uc3QgdGhpbmtUZXh0ID0gdGhpbmtNYXRjaFsxXS50cmltKCk7CiAgICAgICAgICAgICAgICB0aGlua0VsID0gYWRkVGhpbmtCdWJibGUob3V0RWwpOwogICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtUZXh0KTsKICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gdGhpbmtUZXh0OwogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQucmVwbGFjZSh0aGlua01hdGNoWzBdLCAnJykudHJpbSgpOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQ7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgCiAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudChvdXRFbCwgJ2Fzc2lzdGFudCcsIGFzc2lzdGFudEJ1ZmZlcik7CiAgICAgICAgCiAgICAgICAgZmluYWxDb21wbGV0aW9uVG9rZW5zID0gZGF0YT8udXNhZ2U/LmNvbXBsZXRpb25fdG9rZW5zIHx8IDA7CiAgICAgICAgdXBkYXRlVG9rZW5SYXRlKGFzc2lzdGFudEJ1ZmZlciArIHRoaW5rQnVmZmVyLCB0cnVlKTsKICAgICAgICAKICAgICAgICBtYXliZVNjcm9sbCgpOwogICAgICB9IGVsc2UgewogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBTVEFSVCAtLS0KICAgICAgICBsZXQgcmVuZGVyU2NoZWR1bGVkID0gZmFsc2U7CiAgICAgICAgbGV0IGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgbGV0IHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICBjb25zdCBSRU5ERVJfSU5URVJWQUwgPSAxNTA7IAoKICAgICAgICBjb25zdCBzY2hlZHVsZVJlbmRlciA9ICgpID0+IHsKICAgICAgICAgICAgaWYgKHJlbmRlclNjaGVkdWxlZCkgcmV0dXJuOwogICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSB0cnVlOwogICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgICAgICAgICAgIGlmIChhc3Npc3RhbnROZWVkc1JlbmRlcikgewogICAgICAgICAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQob3V0RWwsICdhc3Npc3RhbnQnLCBhc3Npc3RhbnRCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICBpZiAodGhpbmtOZWVkc1JlbmRlciAmJiB0aGlua0VsKSB7CiAgICAgICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSBmYWxzZTsKICAgICAgICAgICAgfSwgUkVOREVSX0lOVEVSVkFMKTsKICAgICAgICB9OwogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBFTkRFIC0tLQoKICAgICAgICBjb25zdCByZWFkZXIgPSByZXMuYm9keS5nZXRSZWFkZXIoKTsgY29uc3QgZGVjb2RlciA9IG5ldyBUZXh0RGVjb2RlcigndXRmLTgnKTsgbGV0IGJ1ZmZlcj0nJzsgbGV0IGRvbmU9ZmFsc2U7CiAgICAgICAgd2hpbGUgKCFkb25lKSB7CiAgICAgICAgICBjb25zdCB7IHZhbHVlLCBkb25lOiByZERvbmUgfSA9IGF3YWl0IHJlYWRlci5yZWFkKCk7IGRvbmUgPSByZERvbmU7CgogICAgICAgICAgaWYgKHN0cmVhbVN0YXJ0VGltZSA9PT0gMCAmJiB2YWx1ZSkgewogICAgICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSBwZXJmb3JtYW5jZS5ub3coKTsKICAgICAgICAgIH0KCiAgICAgICAgICBidWZmZXIgKz0gZGVjb2Rlci5kZWNvZGUodmFsdWUgfHwgbmV3IFVpbnQ4QXJyYXkoKSwgeyBzdHJlYW06ICFkb25lIH0pOwogICAgICAgICAgbGV0IGV2ZW50cyA9IGJ1ZmZlci5zcGxpdCgnXG5cbicpOyBidWZmZXIgPSBldmVudHMucG9wKCkgfHwgJyc7CiAgICAgICAgICBmb3IgKGNvbnN0IGV2IG9mIGV2ZW50cykgewogICAgICAgICAgICBjb25zdCBsaW5lcyA9IGV2LnNwbGl0KCdcbicpLmZpbHRlcihCb29sZWFuKTsKICAgICAgICAgICAgZm9yIChjb25zdCBsaW5lIG9mIGxpbmVzKSB7CiAgICAgICAgICAgICAgaWYgKCFsaW5lLnN0YXJ0c1dpdGgoJ2RhdGE6JykpIGNvbnRpbnVlOwogICAgICAgICAgICAgIGNvbnN0IGRhdGFTdHIgPSBsaW5lLnNsaWNlKDUpLnRyaW0oKTsgaWYgKGRhdGFTdHIgPT09ICdbRE9ORV0nKSB7IGRvbmUgPSB0cnVlOyBicmVhazsgfQogICAgICAgICAgICAgIHRyeSB7CiAgICAgICAgICAgICAgICBjb25zdCBvYmogPSBKU09OLnBhcnNlKGRhdGFTdHIpOwogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBpZiAob2JqPy51c2FnZT8uY29tcGxldGlvbl90b2tlbnMpIHsKICAgICAgICAgICAgICAgICAgICBmaW5hbENvbXBsZXRpb25Ub2tlbnMgPSBvYmoudXNhZ2UuY29tcGxldGlvbl90b2tlbnM7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGRlbHRhID0gb2JqPy5jaG9pY2VzPy5bMF0/LmRlbHRhOwogICAgICAgICAgICAgICAgaWYgKCFkZWx0YSkgY29udGludWU7CgogICAgICAgICAgICAgICAgbGV0IGhhc1VwZGF0ZWQgPSBmYWxzZTsKICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgY29uc3QgcmVhc29uaW5nQ2h1bmsgPSBkZWx0YS5yZWFzb25pbmcgfHwgZGVsdGEucmVhc29uaW5nX2NvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChyZWFzb25pbmdDaHVuaykgewogICAgICAgICAgICAgICAgICAgIGlmICghdGhpbmtFbCkgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZWFzb25pbmdDaHVuazsKICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGNvbnRlbnRDaHVuayA9IGRlbHRhLmNvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICBpZiAoY29udGVudENodW5rKSB7CiAgICAgICAgICAgICAgICAgICAgbGV0IHRlbXBDaHVuayA9IGNvbnRlbnRDaHVuazsKICAgICAgICAgICAgICAgICAgICB3aGlsZSAodGVtcENodW5rLmxlbmd0aCA+IDApIHsKICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluVGhpbmtCbG9jaykgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zID0gdGVtcENodW5rLmluZGV4T2YoJzwvdGhpbms+Jyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyICs9IHRlbXBDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZW1wQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKGVuZFBvcyArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IHN0YXJ0UG9zID0gdGVtcENodW5rLmluZGV4T2YoJzx0aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChzdGFydFBvcyAhPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rLnN1YnN0cmluZygwLCBzdGFydFBvcyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXNzaXN0YW50TmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoIXRoaW5rRWwpIHRoaW5rRWwgPSBhZGRUaGlua0J1YmJsZShvdXRFbCk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgcmVzdE9mQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKHN0YXJ0UG9zICsgNyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zSW5SZXN0ID0gcmVzdE9mQ2h1bmsuaW5kZXhPZignPC90aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zSW5SZXN0ICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZXN0T2ZDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zSW5SZXN0KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gcmVzdE9mQ2h1bmsuc3Vic3RyaW5nKGVuZFBvc0luUmVzdCArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gcmVzdE9mQ2h1bms7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpblRoaW5rQmxvY2sgPSB0cnVlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gJyc7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChoYXNVcGRhdGVkKSB7CiAgICAgICAgICAgICAgICAgIHNjaGVkdWxlUmVuZGVyKCk7CiAgICAgICAgICAgICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgIH0gY2F0Y2gge30KICAgICAgICAgICAgfQogICAgICAgICAgfQogICAgICAgIH0KICAgICAgICAvLyBGaW5hbGVzIFJlbmRlcmluZyBhbSBFbmRlIGRlcyBTdHJlYW1zLCB1bSBzaWNoZXJ6dXN0ZWxsZW4sIGRhc3MgYWxsZXMgYW5nZXplaWd0IHdpcmQKICAgICAgICByZW5kZXJCdWJibGVDb250ZW50KG91dEVsLCAnYXNzaXN0YW50JywgYXNzaXN0YW50QnVmZmVyKTsKICAgICAgICBpZiAodGhpbmtFbCkgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgIAogICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIsIHRydWUpOwogICAgICAgIAogICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgIH0KICAgICAgCiAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGFzc2lzdGFudEJ1ZmZlci50cmltKCk7CiAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyIHx8ICcoZW1wdHkgcmVzcG9uc2UpJyB9KTsKICAgICAgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgaWYgKGVycj8ubmFtZSA9PT0gJ0Fib3J0RXJyb3InKSB7CiAgICAgICAgLy8gU3RvcOODnOOCv+ODszog6YCU5Lit44G+44Gn44Gu5Zue562U44KS5bGl5q2044Gr5q6L44GZCiAgICAgICAgaWYgKGFzc2lzdGFudEJ1ZmZlci50cmltKCkpIHsKICAgICAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyLnRyaW0oKSB9KTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgLy8g6YCa5L+h44Ko44Op44O8OiDnm7TliY3jga51c2Vy44Oh44OD44K744O844K444KS5bGl5q2044GL44KJ5oi744GZCiAgICAgICAgaWYgKG91dEVsICYmICFvdXRFbC50ZXh0Q29udGVudC50cmltKCkpIHsgb3V0RWwuY2xvc2VzdCgnLm1zZycpPy5yZW1vdmUoKTsgfQogICAgICAgIGlmIChtZXNzYWdlcy5sZW5ndGggPiAwICYmIG1lc3NhZ2VzW21lc3NhZ2VzLmxlbmd0aC0xXS5yb2xlID09PSAndXNlcicpIHsKICAgICAgICAgIG1lc3NhZ2VzLnBvcCgpOwogICAgICAgIH0KICAgICAgICAvLyDlhaXlipvjg4bjgq3jgrnjg4jjgpLlvqnlhYPvvIjmt7vku5jmmYLjga9jb250ZW5044GM6YWN5YiX44Gr44Gq44KL44Gf44KBc2F2ZWRUZXh044KS5L2/44GG77yJCiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gc2F2ZWRUZXh0OwogICAgICAgIC8vIOa3u+S7mOODleOCoeOCpOODq+OCguW+qeWFgwogICAgICAgIGlmIChzYXZlZEF0dGFjaG1lbnRzLmxlbmd0aCA+IDApIHsKICAgICAgICAgIGF0dGFjaG1lbnRzID0gc2F2ZWRBdHRhY2htZW50czsKICAgICAgICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgICAgICAgfQogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb44Ko44Op44O8XSAke2Vyci5tZXNzYWdlfWApOyBjb25zb2xlLmVycm9yKGVycik7CiAgICAgIH0KICAgIH0gZmluYWxseSB7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKGZhbHNlKTsgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICAgIH0KICB9CgogIGZ1bmN0aW9uIHN0b3BTdHJlYW0oKXsgaWYgKGFib3J0Q29udHJvbGxlcikgYWJvcnRDb250cm9sbGVyLmFib3J0KCk7IH0KICBmdW5jdGlvbiBjbGVhckNoYXQoKXsgbWVzc2FnZXMgPSBbXTsgZWxzLmxvZy5pbm5lckhUTUwgPSAnJzsgZWxzLnRva2VuUmF0ZS50ZXh0Q29udGVudCA9ICcnOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgbWF5YmVTY3JvbGwodHJ1ZSk7IH0KCiAgLyogLS0tLS0tLS0tLSBJbXBvcnQgLyBFeHBvcnQgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGV4cG9ydENoYXQoKXsKICAgIGNvbnN0IG91dCA9IGJ1aWxkRXhwb3J0QXJyYXkoKTsKICAgIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbSlNPTi5zdHJpbmdpZnkob3V0LCBudWxsLCAyKV0sIHsgdHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nIH0pOwogICAgY29uc3QgdXJsID0gVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTsKICAgIGNvbnN0IGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdhJyk7IGNvbnN0IHRzID0gbmV3IERhdGUoKTsgY29uc3QgcGFkID0gbj0+U3RyaW5nKG4pLnBhZFN0YXJ0KDIsJzAnKTsKICAgIGEuaHJlZiA9IHVybDsgYS5kb3dubG9hZCA9IGBjaGF0XyR7dHMuZ2V0RnVsbFllYXIoKX0tJHtwYWQodHMuZ2V0TW9udGgoKSsxKX0tJHtwYWQodHMuZ2V0RGF0ZSgpKX1fJHtwYWQodHMuZ2V0SG91cnMoKSl9LSR7cGFkKHRzLmdldE1pbnV0ZXMoKSl9LSR7cGFkKHRzLmdldFNlY29uZHMoKSl9Lmpzb25gOwogICAgYS5jbGljaygpOyBzZXRUaW1lb3V0KCgpPT5VUkwucmV2b2tlT2JqZWN0VVJMKHVybCksIDEwMDApOwogIH0KCiAgYXN5bmMgZnVuY3Rpb24gcGFyc2VJbXBvcnQodGV4dCl7CiAgICBjb25zdCBjbGVhbmVkID0gdGV4dC5yZXBsYWNlKC9cci9nLCcnKTsKICAgIGxldCByYXc7IHRyeSB7IHJhdyA9IEpTT04ucGFyc2UoY2xlYW5lZCk7IH0KICAgIGNhdGNoIHsgY29uc3QgbGluZXMgPSBjbGVhbmVkLnNwbGl0KCdcbicpLm1hcChzPT5zLnRyaW0oKSkuZmlsdGVyKEJvb2xlYW4pOyBjb25zdCBhcnI9W107IGZvciAoY29uc3QgbGluZSBvZiBsaW5lcyl7IHRyeXsgYXJyLnB1c2goSlNPTi5wYXJzZShsaW5lKSk7IH1jYXRjaHt9IH0gcmF3ID0gYXJyLmxlbmd0aCA/IGFyciA6IG51bGw7IH0KICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0ZpbGUgbm90IHJlYWRhYmxlIChuZWl0aGVyIEpTT04gbm9yIEpTT05MKS4nKTsKCiAgICBsZXQgYXJyID0gW10sIHRvcFN5c3RlbSA9ICcnOwogICAgaWYgKEFycmF5LmlzQXJyYXkocmF3KSkgYXJyID0gcmF3OwogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcubWVzc2FnZXMpKSB7IGFyciA9IHJhdy5tZXNzYWdlczsgdG9wU3lzdGVtID0gU3RyaW5nKHJhdy5zeXN0ZW0gfHwgJycpLnRyaW0oKTsgfQogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcuY29udmVyc2F0aW9uKSkgeyBhcnIgPSByYXcuY29udmVyc2F0aW9uOyB0b3BTeXN0ZW0gPSBTdHJpbmcocmF3LnN5c3RlbSB8fCAnJykudHJpbSgpOyB9CiAgICBlbHNlIHRocm93IG5ldyBFcnJvcignTm8gbWVzc2FnZSBhcnJheSBmb3VuZCAoZXhwZWN0ZWQ6IFsge3JvbGUsIGNvbnRlbnR9LCAuLi4gXSkuJyk7CgogICAgY29uc3Qgbm9ybWFsaXplZCA9IGFyci5tYXAobSA9PiAoeyByb2xlOiBub3JtYWxpemVSb2xlKG0ucm9sZSksIGNvbnRlbnQ6IG5vcm1hbGl6ZUNvbnRlbnQobS5jb250ZW50KSB9KSkKICAgICAgICAgICAgICAgICAgICAgICAgICAuZmlsdGVyKG0gPT4gWydzeXN0ZW0nLCd1c2VyJywnYXNzaXN0YW50J10uaW5jbHVkZXMobS5yb2xlKSAmJiB0eXBlb2YgbS5jb250ZW50ID09PSAnc3RyaW5nJyk7CgogICAgY29uc3QgaGFzU3lzdGVtID0gbm9ybWFsaXplZC5zb21lKG0gPT4gbS5yb2xlID09PSAnc3lzdGVtJyk7CiAgICBpZiAoIWhhc1N5c3RlbSAmJiB0b3BTeXN0ZW0pIG5vcm1hbGl6ZWQudW5zaGlmdCh7IHJvbGU6J3N5c3RlbScsIGNvbnRlbnQ6IHRvcFN5c3RlbSB9KTsKICAgIGlmICghbm9ybWFsaXplZC5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gdmFsaWQgbWVzc2FnZXMgZm91bmQgaW4gaW1wb3J0LicpOwogICAgcmV0dXJuIG5vcm1hbGl6ZWQ7CiAgfQoKICBmdW5jdGlvbiBoYW5kbGVGaWxlKGZpbGUpewogICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgIHJlYWRlci5vbmxvYWQgPSAoKSA9PiB7CiAgICAgIHBhcnNlSW1wb3J0KFN0cmluZyhyZWFkZXIucmVzdWx0IHx8ICcnKSkKICAgICAgICAudGhlbihhcnIgPT4geyBtZXNzYWdlcyA9IGFycjsgY29uc3Qgc3lzID0gbWVzc2FnZXMuZmluZChtPT5tLnJvbGU9PT0nc3lzdGVtJyk7IGVscy5zeXN0ZW0udmFsdWUgPSBzeXM/c3lzLmNvbnRlbnQ6Jyc7IHJlbmRlckFsbCgpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBg44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GX44G+44GX44GfICgke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXMpLmApOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgfSkKICAgICAgICAuY2F0Y2goZSA9PiBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW+OCpOODs+ODneODvOODiOOCqOODqeODvF0gJHtlLm1lc3NhZ2V9YCkpOwogICAgfTsKICAgIHJlYWRlci5vbmVycm9yID0gKCkgPT4gYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjgqTjg7Pjg53jg7zjg4jjgqjjg6njg7xdIENvdWxkIG5vdCByZWFkIGZpbGUuYCk7CiAgICByZWFkZXIucmVhZEFzVGV4dChmaWxlKTsKICB9CiAgaWYgKGVscy5kcm9wKSB7CiAgICBbJ2RyYWdlbnRlcicsJ2RyYWdvdmVyJ10uZm9yRWFjaChldiA9PiBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKGV2LCBlID0+IHsgZS5wcmV2ZW50RGVmYXVsdCgpOyBlbHMuZHJvcC5jbGFzc0xpc3QuYWRkKCdkcmFnJyk7IH0pKTsKICAgIFsnZHJhZ2xlYXZlJywnZHJvcCddLmZvckVhY2goZXYgPT4gZWxzLmRyb3AuYWRkRXZlbnRMaXN0ZW5lcihldiwgZSA9PiB7IGUucHJldmVudERlZmF1bHQoKTsgZWxzLmRyb3AuY2xhc3NMaXN0LnJlbW92ZSgnZHJhZycpOyB9KSk7CiAgICBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKCdkcm9wJywgKGUpID0+IHsgY29uc3QgZiA9IGUuZGF0YVRyYW5zZmVyPy5maWxlcz8uWzBdOyBpZiAoZikgaGFuZGxlRmlsZShmKTsgfSk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFJBVyBKU09OIEVkaXRvciAtLS0tLS0tLS0tICovCiAgZnVuY3Rpb24gZWRpdG9yU2V0U3RhdHVzKG1zZywgb2s9dHJ1ZSl7IGVscy5qc29uU3RhdHVzLnRleHRDb250ZW50ID0gbXNnIHx8ICcnOyBlbHMuanNvbkVkaXRvci5zdHlsZS5ib3JkZXJDb2xvciA9IG9rID8gJ3ZhcigtLWxpbmUpJyA6ICcjYjQwMDAwJzsgfQogIGZ1bmN0aW9uIGVkaXRvckR1bXBGcm9tQ2hhdCgpeyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KGJ1aWxkRXhwb3J0QXJyYXkoKSwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnbG9hZGVkIGZyb20gY2hhdCcpOyB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUGFyc2VDdXJyZW50KCl7CiAgICBjb25zdCB0ZXh0ID0gZWxzLmpzb25FZGl0b3IudmFsdWUudHJpbSgpIHx8ICdbXSc7CiAgICBjb25zdCBhcnIgPSBhd2FpdCBwYXJzZUltcG9ydCh0ZXh0KTsKICAgIGNvbnN0IGZpcnN0U3lzID0gYXJyLmZpbmQobT0+bS5yb2xlPT09J3N5c3RlbScpOyBpZiAoZmlyc3RTeXMpIGVscy5zeXN0ZW0udmFsdWUgPSBmaXJzdFN5cy5jb250ZW50OwogICAgcmV0dXJuIGFycjsKICB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUmVwbGFjZSgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IGFycjsKICAgICAgcmVuZGVyQWxsKCk7CiAgICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICAgIGF1dG9zY3JvbGwgPSB0cnVlOwogICAgICBlZGl0b3JTZXRTdGF0dXMoYGFwcGxpZWQgKHJlcGxhY2UpIOKAlCAke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXNgKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYEpTT07jgYvjgonjg4Hjg6Pjg4Pjg4jjgpLoqK3lrprjgZfjgb7jgZfjgZ8gKCR7bWVzc2FnZXMubGVuZ3RofSlgKTsKICAgIH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cyhlLm1lc3NhZ2UsZmFsc2UpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW0pTT07jgqjjg6njg7xdICR7ZS5tZXNzYWdlfWApOyB9CiAgfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvckFwcGVuZCgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IG1lc3NhZ2VzLmNvbmNhdChhcnIpOwogICAgICByZW5kZXJBbGwoKTsKICAgICAgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICAgICAgYXV0b3Njcm9sbCA9IHRydWU7CiAgICAgIGVkaXRvclNldFN0YXR1cyhgYXBwbGllZCAoYXBwZW5kKSDigJQgbm93ICR7bWVzc2FnZXMubGVuZ3RofSBtZXNzYWdlc2ApOwogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgSlNPTuOBi+OCieODgeODo+ODg+ODiOOCkui/veiomOOBl+OBvuOBl+OBnyAoKyR7YXJyLmxlbmd0aH0pYCk7CiAgICB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLGZhbHNlKTsgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFtKU09O44Ko44Op44O8XSAke2UubWVzc2FnZX1gKTsgfQogIH0KICBmdW5jdGlvbiBlZGl0b3JCZWF1dGlmeSgpeyB0cnkgeyBjb25zdCBvYmogPSBKU09OLnBhcnNlKGVscy5qc29uRWRpdG9yLnZhbHVlIHx8ICdbXScpOyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KG9iaiwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnYmVhdXRpZmllZCcpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoJ2ludmFsaWQgSlNPTicsIGZhbHNlKTt9IH0KICBmdW5jdGlvbiBlZGl0b3JNaW5pZnkoKXsgdHJ5IHsgY29uc3Qgb2JqID0gSlNPTi5wYXJzZShlbHMuanNvbkVkaXRvci52YWx1ZSB8fCAnW10nKTsgZWxzLmpzb25FZGl0b3IudmFsdWUgPSBKU09OLnN0cmluZ2lmeShvYmopOyBlZGl0b3JTZXRTdGF0dXMoJ21pbmlmaWVkJyk7IH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cygnaW52YWxpZCBKU09OJywgZmFsc2UpO30gfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvclZhbGlkYXRlKCl7IHRyeSB7IGF3YWl0IGVkaXRvclBhcnNlQ3VycmVudCgpOyBlZGl0b3JTZXRTdGF0dXMoJ3ZhbGlkIOKckycpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLCBmYWxzZSk7IH0gfQoKICBlbHMucmF3Qm94LmFkZEV2ZW50TGlzdGVuZXIoJ3RvZ2dsZScsICgpID0+IHsgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7IH0pOwogIGVscy5qc29uUmVmcmVzaC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckR1bXBGcm9tQ2hhdCk7CiAgZWxzLmpzb25BcHBseS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclJlcGxhY2UpOwogIGVscy5qc29uQXBwZW5kLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgZWRpdG9yQXBwZW5kKTsKICBlbHMuanNvblByZXR0eS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckJlYXV0aWZ5KTsKICBlbHMuanNvbk1pbmlmeS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvck1pbmlmeSk7CiAgZWxzLmpzb25WYWxpZGF0ZS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclZhbGlkYXRlKTsKCiAgLyogLS0tLS0tLS0tLSBXaXJlIFVJIC0tLS0tLS0tLS0gKi8KICBlbHMuc2VuZEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHNlbmRNZXNzYWdlKTsKICBlbHMuc3RvcEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHN0b3BTdHJlYW0pOwogIGVscy5jbGVhckJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGNsZWFyQ2hhdCk7CiAgaWYgKGVscy5hdHRhY2hCdG4gJiYgZWxzLmF0dGFjaElucHV0KSB7CiAgICBlbHMuYXR0YWNoQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCk9PiBlbHMuYXR0YWNoSW5wdXQuY2xpY2soKSk7CiAgICBlbHMuYXR0YWNoSW5wdXQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKCk9PnsKICAgICAgYWRkQXR0YWNobWVudHNGcm9tRmlsZUxpc3QoZWxzLmF0dGFjaElucHV0LmZpbGVzKTsKICAgICAgZWxzLmF0dGFjaElucHV0LnZhbHVlID0gJyc7CiAgICB9KTsKICB9CiAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGxvYWRNb2RlbHMpOwogIGVscy5pbnB1dC5hZGRFdmVudExpc3RlbmVyKCdrZXlkb3duJywgKGUpID0+IHsKICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJyAmJiAhZS5zaGlmdEtleSAmJiAhZS5pc0NvbXBvc2luZyAmJiBlLmtleUNvZGUgIT09IDIyOSkgewogICAgICAvLyDjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK3jga/pgIHkv6HjgZvjgZrjgIHmlLnooYzjgpLoqLHlj68KICAgICAgaWYgKCFlbHMuc3RvcEJ0bi5kaXNhYmxlZCkgcmV0dXJuOwogICAgICBlLnByZXZlbnREZWZhdWx0KCk7CiAgICAgIHNlbmRNZXNzYWdlKCk7CiAgICB9CiAgfSk7CiAgZWxzLnRlbXAuYWRkRXZlbnRMaXN0ZW5lcignaW5wdXQnLCAoKSA9PiB7IGVscy50ZW1wVmFsLnRleHRDb250ZW50ID0gcGFyc2VGbG9hdChlbHMudGVtcC52YWx1ZSkudG9GaXhlZCgxKTsgfSk7CiAgZWxzLnNldHRpbmdzVG9nZ2xlLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgZWxzLnNldHRpbmdzUGFuZWwuY2xhc3NMaXN0LnRvZ2dsZSgnaGlkZGVuJyk7CiAgfSk7CgogIC8qIC0tLS0tLS0tLS0gSW5mbyBkaWFsb2cgLS0tLS0tLS0tLSAqLwogIGlmIChlbHMuaW5mb1ZlcnNpb24pIGVscy5pbmZvVmVyc2lvbi50ZXh0Q29udGVudCA9IEFQUF9WRVJTSU9OOwogIGlmIChlbHMuaW5mb1Byb2plY3RVcmwpIHsKICAgIGVscy5pbmZvUHJvamVjdFVybC50ZXh0Q29udGVudCA9IFBST0pFQ1RfVVJMOwogICAgZWxzLmluZm9Qcm9qZWN0VXJsLmhyZWYgPSBQUk9KRUNUX1VSTDsKICB9CiAgaWYgKGVscy5pbmZvSW1wcmludCkgewogICAgZWxzLmluZm9JbXByaW50LnRleHRDb250ZW50ID0gSU1QUklOVF9VUkw7CiAgICBlbHMuaW5mb0ltcHJpbnQuaHJlZiA9IElNUFJJTlRfVVJMOwogIH0KICBpZiAoZWxzLmluZm9CdG4gJiYgZWxzLmluZm9EaWFsb2cpIHsKICAgIGVscy5pbmZvQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgICBpZiAodHlwZW9mIGVscy5pbmZvRGlhbG9nLnNob3dNb2RhbCA9PT0gJ2Z1bmN0aW9uJykgZWxzLmluZm9EaWFsb2cuc2hvd01vZGFsKCk7CiAgICAgIGVsc2UgZWxzLmluZm9EaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgfSk7CiAgICAvLyBDbGljayBvdXRzaWRlIHRoZSBjYXJkIGNsb3NlcyB0aGUgZGlhbG9nCiAgICBlbHMuaW5mb0RpYWxvZy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIChlKSA9PiB7CiAgICAgIC8vIElmIGNsaWNrIGxhbmRzIG9uIHRoZSBiYWNrZHJvcCAobm90IG9uIHRoZSBjYXJkKSwgY2xvc2UuCiAgICAgIGlmIChlLnRhcmdldCA9PT0gZWxzLmluZm9EaWFsb2cpIGVscy5pbmZvRGlhbG9nLmNsb3NlKCk7CiAgICB9KTsKICB9CgogIC8qIC0tLS0tLS0tLS0gQVBJIOiqjeiovOODgOOCpOOCouODreOCsCAtLS0tLS0tLS0tICovCiAgaWYgKGVscy5hdXRoQnRuICYmIGVscy5hdXRoRGlhbG9nKSB7CiAgICBlbHMuYXV0aEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IG9wZW5BdXRoRGlhbG9nKCkpOwogIH0KICBpZiAoZWxzLmF1dGhTYXZlKSB7CiAgICBlbHMuYXV0aFNhdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7CiAgICAgIGNvbnN0IHVzZXIgPSAoZWxzLmF1dGhVc2VyPy52YWx1ZSB8fCAnJykudHJpbSgpIHx8ICdleGwzJzsKICAgICAgY29uc3QgcGFzcyA9IGVscy5hdXRoUGFzcz8udmFsdWUgfHwgJyc7CiAgICAgIGlmICghcGFzcykgewogICAgICAgIGlmIChlbHMuYXV0aFN0YXR1cykgZWxzLmF1dGhTdGF0dXMudGV4dENvbnRlbnQgPSAn44OR44K544Ov44O844OJ44KS5YWl5Yqb44GX44Gm44GP44Gg44GV44GEJzsKICAgICAgICByZXR1cm47CiAgICAgIH0KICAgICAgc2V0QXV0aENyZWRlbnRpYWxzKHVzZXIsIHBhc3MpOwogICAgICBjbG9zZUF1dGhEaWFsb2coKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvoqo3oqLxdICR7dXNlcn0g44Gn5L+d5a2Y44GX44G+44GX44Gf77yI44GT44Gu44K/44OW44Gu44G/77yJYCk7CiAgICAgIGxvYWRNb2RlbHMoKTsKICAgICAgc3RhcnRRdW90YVBvbGxpbmcoKTsKICAgICAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhDbGVhcikgewogICAgZWxzLmF1dGhDbGVhci5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsKICAgICAgY2xlYXJBdXRoQ3JlZGVudGlhbHMoKTsKICAgICAgY2xvc2VBdXRoRGlhbG9nKCk7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsICdb6KqN6Ki8XSDos4fmoLzmg4XloLHjgpLop6PpmaTjgZfjgb7jgZfjgZ8nKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhQYXNzKSB7CiAgICBlbHMuYXV0aFBhc3MuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIChlKSA9PiB7CiAgICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJykgeyBlLnByZXZlbnREZWZhdWx0KCk7IGVscy5hdXRoU2F2ZT8uY2xpY2soKTsgfQogICAgfSk7CiAgfQogIGlmIChlbHMuYXV0aERpYWxvZykgewogICAgZWxzLmF1dGhEaWFsb2cuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4gewogICAgICBpZiAoZS50YXJnZXQgPT09IGVscy5hdXRoRGlhbG9nKSBjbG9zZUF1dGhEaWFsb2coKTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSDmrovjgormmYLplpMgLyDlgZzmraLjg5zjgr/jg7MgLS0tLS0tLS0tLSAqLwogIHN0YXJ0UXVvdGFQb2xsaW5nKCk7CiAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgIGVscy5zdG9wU2VydmVyLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgc3RvcFNlcnZlcik7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFdlbGNvbWUgLS0tLS0tLS0tLSAqLwogIHNldFRpbWVvdXQoKCkgPT4geyBsb2FkTW9kZWxzKCk7IH0sIDUwMCk7Cn0pKCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K"
# (ファイル名, base64) を WORK へ展開する。proxy / UI / 同梱サーバの実体。
AUX_FILES = (('auth_proxy.py', 'IyDilIDilIAgcnVudGltZS9xdW90YS5wee+8iGF1dGhfcHJveHkucHkg44Gr5ZCM5qKx77yJCiIiIkthZ2dsZSDjga7jgq/jgqnjg7zjgr8gQVBJ77yIR1BVL1RQVSDmrovph4/vvInjgpLoqq3jgoDlhbHpgJrlh6bnkIbvvIjnlJ/miJDjgrPjg7zjg4njgbjln4vjgoHovrzjgoDvvInjgIIKCmthZ2dsZSBDTEkg44Gr44GvIHF1b3RhIOOCteODluOCs+ODnuODs+ODieOBjOeEoeOBhOOBjOOAgVdlYiBBUEkg44GvIEJhc2ljIOiqjeiovOOBp+iqreOCgeOCi+OAggrlkbzjgbPlh7rjgZflgbTjga8gYXV0aF9wcm94eS5wee+8iC9xdW90Ye+8ieOBqCBrYWdnbGUtcXVvdGEucHnvvIhDTEkg6KGo56S677yJ44CCCiIiIgppbXBvcnQgYmFzZTY0CmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdXJsbGliLnJlcXVlc3QKCktDX1FVT1RBX1VSTCA9ICJodHRwczovL3d3dy5rYWdnbGUuY29tL2FwaS92MS9rZXJuZWxzL3F1b3RhIgpLQ19DUkVEX1BBVEggPSBvcy5wYXRoLmV4cGFuZHVzZXIoIn4vLmthZ2dsZS9rYWdnbGUuanNvbiIpCgoKZGVmIGtjX2thZ2dsZV9jcmVkZW50aWFscyhjcmVkX3BhdGg9Tm9uZSwgZW52aXJvbj1Ob25lKToKICAgICIiIih1c2VybmFtZSwga2V5KSDjgpLov5TjgZnjgIJrYWdnbGUuanNvbiDihpIg55Kw5aKD5aSJ5pWw44Gu6aCG44CC54Sh44GR44KM44GwIChOb25lLCBOb25lKeOAggoKICAgIOOCq+ODvOODjeODq+OBq+OBryBrYWdnbGUuanNvbiDjgYznhKHjgYTjgYzjgIFLQUdHTEVfVVNFUk5BTUUgLyBLQUdHTEVfS0VZIOOBjAogICAg55Kw5aKD5aSJ5pWw44Gn5YWl44Gj44Gm44GE44KL44CCCiAgICAiIiIKICAgIGNyZWRfcGF0aCA9IEtDX0NSRURfUEFUSCBpZiBjcmVkX3BhdGggaXMgTm9uZSBlbHNlIGNyZWRfcGF0aAogICAgZW52aXJvbiA9IG9zLmVudmlyb24gaWYgZW52aXJvbiBpcyBOb25lIGVsc2UgZW52aXJvbgogICAgaWYgb3MucGF0aC5pc2ZpbGUoY3JlZF9wYXRoKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHdpdGggb3BlbihjcmVkX3BhdGgpIGFzIGY6CiAgICAgICAgICAgICAgICBjZmcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgdXNlciwga2V5ID0gY2ZnLmdldCgidXNlcm5hbWUiKSwgY2ZnLmdldCgia2V5IikKICAgICAgICAgICAgaWYgdXNlciBhbmQga2V5OgogICAgICAgICAgICAgICAgcmV0dXJuIHVzZXIsIGtleQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgIHVzZXIgPSBlbnZpcm9uLmdldCgiS0FHR0xFX1VTRVJOQU1FIikgb3IgZW52aXJvbi5nZXQoIktBR0dMRV9VU0VSIikKICAgIGtleSA9IGVudmlyb24uZ2V0KCJLQUdHTEVfS0VZIikKICAgIHJldHVybiAodXNlciwga2V5KSBpZiB1c2VyIGFuZCBrZXkgZWxzZSAoTm9uZSwgTm9uZSkKCgpkZWYga2NfZmV0Y2hfcXVvdGEodXNlciwga2V5LCB0aW1lb3V0PTQ1LCB1cmw9S0NfUVVPVEFfVVJMLCBvcGVuZXI9Tm9uZSk6CiAgICAiIiLjgq/jgqnjg7zjgr/jga7nlJ8gSlNPTiDjgpLov5TjgZnjgIJIVFRQIC8g5o6l57aa44Ko44Op44O844Gv5L6L5aSW44Gu44G+44G+5oqV44GS44KL44CCIiIiCiAgICB0b2tlbiA9IGJhc2U2NC5iNjRlbmNvZGUoZiJ7dXNlcn06e2tleX0iLmVuY29kZSgpKS5kZWNvZGUoKQogICAgcmVxID0gdXJsbGliLnJlcXVlc3QuUmVxdWVzdCgKICAgICAgICB1cmwsIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJhc2ljICIgKyB0b2tlbn0pCiAgICB3aXRoIChvcGVuZXIgb3IgdXJsbGliLnJlcXVlc3QudXJsb3BlbikocmVxLCB0aW1lb3V0PXRpbWVvdXQpIGFzIHI6CiAgICAgICAgcmV0dXJuIGpzb24ubG9hZHMoci5yZWFkKCkpCgoKZGVmIGtjX3F1b3RhX3N1bW1hcnkoZGF0YSk6CiAgICAiIiJVSSDnlKjjga4gR1BVIOOCr+OCqeODvOOCv+imgee0hOOAgmdwdVF1b3RhIOOBjOeEoeOBhOW/nOetlOOBryBhdmFpbGFibGU9RmFsc2XjgIIiIiIKICAgIGcgPSBkYXRhLmdldCgiZ3B1UXVvdGEiKSBvciB7fQogICAgdXNlZCA9IChnLmdldCgidGltZVVzZWQiKSBvciB7fSkuZ2V0KCJzZWNvbmRzIikKICAgIHRvdGFsID0gKGcuZ2V0KCJ0b3RhbFRpbWVBbGxvd2VkIikgb3Ige30pLmdldCgic2Vjb25kcyIpCiAgICBpZiB1c2VkIGlzIE5vbmUgb3IgdG90YWwgaXMgTm9uZToKICAgICAgICByZXR1cm4geyJhdmFpbGFibGUiOiBGYWxzZSwgInJlYXNvbiI6ICJub19ncHVfcXVvdGEifQogICAgcmVzZXJ2ZWQgPSAoZy5nZXQoInRpbWVSZXNlcnZlZCIpIG9yIHt9KS5nZXQoInNlY29uZHMiLCAwKQogICAgcmV0dXJuIHsKICAgICAgICAiYXZhaWxhYmxlIjogVHJ1ZSwKICAgICAgICAidXNlZFNlY29uZHMiOiB1c2VkLAogICAgICAgICJ0b3RhbFNlY29uZHMiOiB0b3RhbCwKICAgICAgICAicmVzZXJ2ZWRTZWNvbmRzIjogcmVzZXJ2ZWQsCiAgICAgICAgImxlZnRTZWNvbmRzIjogdG90YWwgLSB1c2VkIC0gcmVzZXJ2ZWQsCiAgICAgICAgInJlZnJlc2hUaW1lIjogZGF0YS5nZXQoInF1b3RhUmVmcmVzaFRpbWUiLCAiIiksCiAgICB9CgojIS91c3IvYmluL2VudiBweXRob24zCiIiIkJhc2ljIOiqjeiovOOBpOOBjeODquODkOODvOOCueODl+ODreOCreOCt++8iOaomea6luODqeOCpOODluODqeODquOBruOBv++8ieOAggoKY2xvdWRmbGFyZWQg44GuIHF1aWNrIHR1bm5lbCDjgavjga8gQmFzaWMg6KqN6Ki85qmf6IO944GM54Sh44GE44Gf44KB44CBCk9sbGFtYSDjga7miYvliY3jgavjgZPjgozjgpLnva7jgYTjgaboqo3oqLzjgpLmi4Xjgo/jgZvjgovjgIIKCiAgICBBUElfVVNFUj1vbGxhbWEgQVBJX1BBU1M9c2VjcmV0IHB5dGhvbjMgYXV0aF9wcm94eS5weSBcCiAgICAgICAgLS1wb3J0IDExNDM1IC0tdXBzdHJlYW0gaHR0cDovLzEyNy4wLjAuMToxMTQzNAoK44K544OI44Oq44O844Of44Oz44Kw77yIT2xsYW1hIOOBriB0b2tlbiDpgJDmrKHlh7rlipvvvInjgpLpgJrjgZnjgZ/jgoHjgIEK5LiK5rWB44Gu5b+c562U44GvIGNodW5rZWQgdHJhbnNmZXItZW5jb2Rpbmcg44Gn44Gd44Gu44G+44G+5Lit57aZ44GZ44KL44CCCiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBiYXNlNjQKaW1wb3J0IGhtYWMKaW1wb3J0IGh0dHAuc2VydmVyCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgcGF0aGxpYgppbXBvcnQgc2lnbmFsCmltcG9ydCBzb2NrZXRzZXJ2ZXIKaW1wb3J0IHN5cwppbXBvcnQgdGhyZWFkaW5nCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5wYXJzZQppbXBvcnQgdXJsbGliLnJlcXVlc3QKCiMg5LiK5rWB77yIT2xsYW1h77yJ44G444Gu5b6F44Gh5pmC6ZaT44CC6ZW344GE55Sf5oiQ44Gr6ICQ44GI44KL44KI44GG6ZW344KB44Gr5Y+W44KL44CCClVQU1RSRUFNX1RJTUVPVVQgPSAzNjAwCiMg5Lit57aZ44Gu44OQ44OD44OV44Kh5LiK6ZmQ44CC5pS56KGM77yITkRKU09OIOihjO+8ieOBjOadpeOBquOBhOWgtOWQiOOBruODleOCqeODvOODq+ODkOODg+OCr+OBp+OAgQojIOOBk+OBk+OBq+mBlOOBmeOCi+OBvuOBp+W+heOBpOOBqOOCueODiOODquODvOODoOOBjOa6nOOBvuOCi+OBruOBp+OAgeWwj+OBleOBhOOBu+OBqemAkOasoeaAp+OBr+mrmOOBhOOAggpDSFVOSyA9IDQwOTYKIyBCYXNpYyDoqo3oqLzjgpLlhY3pmaTjgZnjgovjg5HjgrnjgIIKIyAvaGVhbHRoeiDjga/nlo7pgJrnorroqo3jgIEvIOOBqCAvY2hhdCDjga/jg4Hjg6Pjg4Pjg4ggVUnvvIjpnZnnmoTjgaogSFRNTCDjga7jgb/jgIIKIyDoqo3oqLzmg4XloLHjgoLjg6Ljg4fjg6vjgoLlkKvjgb7jgarjgYTjga7jgafjgIHlhazplovjgZfjgabjgoLmvI/jgozjgovjgoLjga7jga/nhKHjgYTvvInjgIIKIyBBUEkg5pys5L2T77yIL2FwaS8qLCAvdjEvKu+8ieOBr+W+k+adpeOBqeOBiuOCiiBCYXNpYyDoqo3oqLzjgYzopoHjgovjgIIKT1BFTl9QQVRIUyA9IHsiL2hlYWx0aHoiLCAiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCIsICIvZmF2aWNvbi5pY28ifQpVSV9QQVRIUyA9IHsiLyIsICIvY2hhdCIsICIvaW5kZXguaHRtbCJ9CgojIOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiO+8iOS4iua1geOBuOa1geOBleOBquOBhO+8ieOAggojIC9xdW90YSA6IEthZ2dsZSDjga4gR1BVIOOCr+OCqeODvOOCv+aui+mHj++8iFVJIOOBruODmOODg+ODgOOBq+WHuuOBme+8iQojIC9zdG9wICA6IOOBk+OBruS4ree2meOBqOS4iua1geOBruOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCi+OAggojICAgICAgICAgIFBPU1QgKyDjg5jjg4Pjg4AgWC1LYy1TdG9wOiAxIOOBjOW/hemgiO+8iEdFVCDjga7jgrnjgq3jg6Pjg4rjgoTku5bjgrXjgqTjg4jjgYvjgonjga4KIyAgICAgICAgICBDU1JGIOOBp+iQveOBqOOBleOCjOOBquOBhOOCiOOBhuOBq++8ieOAgkFQSV9QQVNTIOioreWumuaZguOBryBCYXNpYyDoqo3oqLzjgoLopoHjgovjgIIKQURNSU5fUEFUSFMgPSB7Ii9xdW90YSIsICIvc3RvcCJ9ClNUT1BfSEVBREVSID0gIlgtS2MtU3RvcCIKCiMga2VlcGFsaXZlIOOCkuW7tumVt+OBmeOCi+OBruOBr+OAjOeUn+aIkOOAjeODquOCr+OCqOOCueODiOOBoOOBkeOAgi9hcGkvcHMg44KEIC92MS9tb2RlbHMg44GuCiMg44Od44O844Oq44Oz44Kw77yI44Op44Oz44OB44Oj44O844Gu6LW35YuV56K66KqN44O744K544Kt44Oj44OK77yJ44GnIEdQVSDjgrvjg4Pjgrfjg6fjg7PjgpLlu7blkb3jgZfjgarjgYTjgIIKR0VORVJBVElPTl9QQVRIUyA9IHsKICAgICIvYXBpL2NoYXQiLAogICAgIi9hcGkvZ2VuZXJhdGUiLAogICAgIi92MS9jaGF0L2NvbXBsZXRpb25zIiwKICAgICIvdjEvY29tcGxldGlvbnMiLAogICAgIi92MS9yZXNwb25zZXMiLAp9CgojIOOCr+OCqeODvOOCv+Wun+ijheOBryBydW50aW1lL3F1b3RhLnB5IOOBjOecn+Wun+a6kOOAguOCq+ODvOODjeODq+OBp+OBr+ODk+ODq+ODgOODvOOBjOWQjOOCveODvOOCueOCkgojIOWFiOmgreOBq+mAo+e1kOOBl+OBpuWfi+OCgei+vOOCgOOBn+OCgeOAgWtjXyog44GM5pyq5a6a576p44Gu44Ot44O844Kr44Or5a6f6KGM44Gg44GRIGltcG9ydCDjgZnjgovjgIIKaWYgImtjX2ZldGNoX3F1b3RhIiBub3QgaW4gZ2xvYmFscygpOgogICAgc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihwYXRobGliLlBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnRzWzJdKSkKICAgIGZyb20gcnVudGltZSBpbXBvcnQgcXVvdGEgYXMgX2tjX3F1b3RhCiAgICBrY19mZXRjaF9xdW90YSA9IF9rY19xdW90YS5rY19mZXRjaF9xdW90YQogICAga2NfcXVvdGFfc3VtbWFyeSA9IF9rY19xdW90YS5rY19xdW90YV9zdW1tYXJ5CiAgICBrY19rYWdnbGVfY3JlZGVudGlhbHMgPSBfa2NfcXVvdGEua2Nfa2FnZ2xlX2NyZWRlbnRpYWxzCgpIT1BfQllfSE9QID0gewogICAgImNvbm5lY3Rpb24iLAogICAgImtlZXAtYWxpdmUiLAogICAgInByb3h5LWF1dGhlbnRpY2F0ZSIsCiAgICAicHJveHktYXV0aG9yaXphdGlvbiIsCiAgICAidGUiLAogICAgInRyYWlsZXJzIiwKICAgICJ0cmFuc2Zlci1lbmNvZGluZyIsCiAgICAidXBncmFkZSIsCiAgICAiZXhwZWN0IiwKfQoKQVJHUyA9IE5vbmUKRVhQRUNURURfQVVUSCA9ICIiCiMg44OB44Oj44OD44OIIFVJIOOBriBIVE1M44CC6LW35YuV5pmC44Gr5LiA5bqm44Gg44GR6Kqt44KA77yI5q+O44Oq44Kv44Ko44K544OI44Gn44OH44Kj44K544Kv44KS5Y+p44GL44Gq44GE77yJ44CCCkNIQVRfQllURVMgPSBiIiIKCiMgLS0tLSDjgrXjg7Pjg5fjg6rjg7PjgrDlvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg44Gu44Go44GN44Gg44GR5pyJ5Yq577yJLS0tLQojIHZMTE0g54mI44Gn5b+F6KaB44CC44Kv44Op44Kk44Ki44Oz44OI77yIb3BlbmNvZGUg562J77yJ44GMIHRlbXBlcmF0dXJlPTAg44KS6YCB44KL44GoCiMg6LKq5qyy5rOV44Gr44Gq44KK44CBMjdCIOe0muOBp+OBr+WQjOOBmOaWh+OBrue5sOOCiui/lOOBl+OBp+W0qeWjiuOBl+OChOOBmeOBhOOAggojIHJlcGV0aXRpb25fcGVuYWx0eSDjgoLmnKrmjIflrprjgaDjgaggMS4w77yIPeeEoeWKue+8ieOBq+OBquOCi+OBn+OCgeS4i+mZkOOCkuWFpeOCjOOCi+OAggojIOaXouWumuOBruODl+ODreOCreOCt++8iE9sbGFtYSDniYjvvInjgafjga/nhKHlirnjgarjga7jgafmjJnli5Xjga/lpInjgo/jgonjgarjgYTjgIIKUlBfRkxPT1IgPSAxLjA1ICAgICAjIHJlcGV0aXRpb25fcGVuYWx0eSDjga7kuIvpmZAKVEVNUF9GTE9PUiA9IDAuNyAgICAjIHRlbXBlcmF0dXJlIOOBruS4i+mZkO+8iDAgPSDosqrmrLLms5XjgpLnpoHmraLvvIkKCgojIOODiOODs+ODjeODqy/jg4Hjg6Pjg4Pjg4jntYznlLHjga7jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgacga2VlcGFsaXZlIOaZgumWk+OCkuODquOCu+ODg+ODiOOAggojIC90bXAva2VlcGFsaXZlLmNvbmYg44GuIFJVTj0xLCBNSU5VVEVTPTYwIOOCkuWGjeabuOOBjei+vOOBv+OAggpfS0VFUEFMSVZFX0NPTkYgPSAiL3RtcC9rZWVwYWxpdmUuY29uZiIKCgpkZWYgX3Jlc2V0X2tlZXBhbGl2ZSgpOgogICAgIiIi44OB44Oj44OD44OI6YCB5L+h44GU44Go44Gr5pmC6ZaT44Oq44K744OD44OI77yIT04g57at5oyB77yJ44CC5LiK5rWB44Ko44Op44O85pmC44Gn44KC5ZG844G244CCIiIiCiAgICB0cnk6CiAgICAgICAgd2l0aCBvcGVuKF9LRUVQQUxJVkVfQ09ORiwgInciKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKCJSVU49MVxuTUlOVVRFUz02MFxuIikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcGFzcwoKCmRlZiBfaXNfZ2VuZXJhdGlvbihwYXRoX29ubHkpOgogICAgcmV0dXJuIHBhdGhfb25seS5yc3RyaXAoIi8iKSBpbiBHRU5FUkFUSU9OX1BBVEhTCgoKZGVmIF9udW0odmFsdWUpOgogICAgIiIi5pWw5YCk5YyW44Gn44GN44Gq44GR44KM44GwIE5vbmXvvIhib29sIOOBr+aVsOWApOaJseOBhOOBl+OBquOBhO+8ieOAgiIiIgogICAgaWYgaXNpbnN0YW5jZSh2YWx1ZSwgYm9vbCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHRyeToKICAgICAgICByZXR1cm4gZmxvYXQodmFsdWUpCiAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcmV0dXJuIE5vbmUKCgpkZWYgZW5mb3JjZV9zYW1wbGluZyhib2R5LCBwYXRoX29ubHkpOgogICAgIiIiY2hhdC9jb21wbGV0aW9ucyDjga7jgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjgpLlvLfliLbjgZnjgovvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC77yJ44CCCgogICAgLSByZXBldGl0aW9uX3BlbmFsdHkgPCBSUF9GTE9PUu+8iOacquaMh+WumuWQq+OCgO+8ieKGkiBSUF9GTE9PUgogICAgLSB0ZW1wZXJhdHVyZSA8PSAw77yI5pyq5oyH5a6a5ZCr44KA77yJ4oaSIFRFTVBfRkxPT1IKCiAgICDjg5zjg4fjgqPjgYwgSlNPTiDjgafjgarjgYQv5aOK44KM44Gm44GE44KL5aC05ZCI44Gv57Sg6YCa44GX77yI5LiK5rWB44Gr5Yik5pat44KS5aeU44Gt44KL77yJ44CCCiAgICDmiLvjgorlgKTjga/lt67jgZfmm7/jgYjlvozjga4gYnl0ZXPjgIJDb250ZW50LUxlbmd0aCDjga/lkbzjgbPlh7rjgZflgbTjgafku5jjgZHnm7TjgZnjgIIKICAgICIiIgogICAgaWYgbm90IEFSR1MuZW5mb3JjZV9zYW1wbGluZyBvciBub3QgYm9keToKICAgICAgICByZXR1cm4gYm9keQogICAgIyDjgq/jgqjjg6rmloflrZfliJfvvIg/YXBpLXZlcnNpb249Li4uIOetie+8ieOBjOS7mOOBj+OBqCBlbmRzd2l0aCDjgYzlpJbjgozjgovjgZ/jgoHjgIEKICAgICMg44OR44K544Gg44GR44KS5Y+W44KK5Ye644GX44Gm5q+U6LyD44GZ44KL44CC5Y+W44KK44GT44G844GZ44Go55+v5q2j44GM54Sh5Yq55YyW44GV44KM44KL44CCCiAgICBfcGF0aCA9IHVybGxpYi5wYXJzZS51cmxzcGxpdChwYXRoX29ubHkpLnBhdGgKICAgIGlmIG5vdCBfcGF0aC5yc3RyaXAoIi8iKS5lbmRzd2l0aCgiL2NoYXQvY29tcGxldGlvbnMiKToKICAgICAgICByZXR1cm4gYm9keQogICAgdHJ5OgogICAgICAgIGRhdGEgPSBqc29uLmxvYWRzKGJvZHkpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBib2R5CiAgICBpZiBub3QgaXNpbnN0YW5jZShkYXRhLCBkaWN0KToKICAgICAgICByZXR1cm4gYm9keQogICAgY2hhbmdlZCA9IEZhbHNlCiAgICBycCA9IF9udW0oZGF0YS5nZXQoInJlcGV0aXRpb25fcGVuYWx0eSIpKQogICAgaWYgcnAgaXMgTm9uZSBvciBycCA8IFJQX0ZMT09SOgogICAgICAgIGRhdGFbInJlcGV0aXRpb25fcGVuYWx0eSJdID0gUlBfRkxPT1IKICAgICAgICBjaGFuZ2VkID0gVHJ1ZQogICAgdGVtcCA9IF9udW0oZGF0YS5nZXQoInRlbXBlcmF0dXJlIikpCiAgICBpZiB0ZW1wIGlzIE5vbmUgb3IgdGVtcCA8PSAwOgogICAgICAgIGRhdGFbInRlbXBlcmF0dXJlIl0gPSBURU1QX0ZMT09SCiAgICAgICAgY2hhbmdlZCA9IFRydWUKICAgIGlmIG5vdCBjaGFuZ2VkOgogICAgICAgIHJldHVybiBib2R5CiAgICByZXR1cm4ganNvbi5kdW1wcyhkYXRhLCBlbnN1cmVfYXNjaWk9RmFsc2UpLmVuY29kZSgidXRmLTgiKQoKCmNsYXNzIEhhbmRsZXIoaHR0cC5zZXJ2ZXIuQmFzZUhUVFBSZXF1ZXN0SGFuZGxlcik6CiAgICBwcm90b2NvbF92ZXJzaW9uID0gIkhUVFAvMS4xIgogICAgc2VydmVyX3ZlcnNpb24gPSAiT2xsYW1hQXV0aFByb3h5LzEuMCIKICAgICMg5o6l57aa44GU44Go44Gu44K944Kx44OD44OI44K/44Kk44Og44Ki44Km44OI44CC44Kv44Op44Kk44Ki44Oz44OI44GM54Sh6YCa5L+h44Gu44G+44G+5q6L44GZ44GoCiAgICAjIOODj+ODs+ODieODqeOCueODrOODg+ODieOBjOeEoemZkOOBq+WxheW6p+OCi+OBruOBp+OAgTXliIbjgaflm57lj47jgZnjgovvvIjjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK0KICAgICMg44Gu5LiK5rWB6Kqt44G/5Y+W44KK44GvIHVybGxpYiDlgbTjgr3jgrHjg4Pjg4jjgarjga7jgaflvbHpn7/jgpLlj5fjgZHjgarjgYTvvInjgIIKICAgIHRpbWVvdXQgPSAzMDAKCiAgICAjIC0tLS0g44Ot44Kw44KSIDEg6KGM44Gr5Zyn57iu77yI44OO44O844OI44OW44OD44Kv44Gu44Ot44Kw44KS5rGa44GV44Gq44GE77yJLS0tLQogICAgZGVmIGxvZ19tZXNzYWdlKHNlbGYsIGZtdCwgKmFyZ3MpOgogICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSAlc1xuIiAlIChmbXQgJSBhcmdzKSkKCiAgICBkZWYgX3JlYWRfYm9keShzZWxmKToKICAgICAgICAiIiLjg6rjgq/jgqjjgrnjg4jjg5zjg4fjgqPjgpLoqq3jgoDjgIJjaHVua2VkIOOBp+OCgiBDb250ZW50LUxlbmd0aCDjgafjgoLkuKHlr77lv5zjgIIKCiAgICAgICAgQ29udGVudC1MZW5ndGgg44GX44GL6KaL44Gq44GE44Go44CBSFRUUC8xLjEg44GuIGNodW5rZWQgUE9TVO+8iOOCueODiOODquODvOODn+ODs+OCsOWFpeWKm+OChAogICAgICAgIOODl+ODreOCreOCt+e1jOeUseOBruWkieaPm+OBp+eZuueUn+OBl+OBhuOCi++8ieOBpyoq44Oc44OH44Kj44GU44Go5qyg6JC9KirjgZfjgabkuIrmtYHjgavlsYrjgYvjgarjgYTjgIIKICAgICAgICAiIiIKICAgICAgICB0ZSA9IChzZWxmLmhlYWRlcnMuZ2V0KCJUcmFuc2Zlci1FbmNvZGluZyIpIG9yICIiKS5sb3dlcigpCiAgICAgICAgaWYgImNodW5rZWQiIGluIHRlOgogICAgICAgICAgICBvdXQgPSBieXRlYXJyYXkoKQogICAgICAgICAgICB3aGlsZSBUcnVlOgogICAgICAgICAgICAgICAgc2l6ZV9saW5lID0gc2VsZi5yZmlsZS5yZWFkbGluZSgpLnN0cmlwKCkKICAgICAgICAgICAgICAgIGlmIGIiOyIgaW4gc2l6ZV9saW5lOiAgICAgICAgICAjIGNodW5rLWV4dCDjga/mjajjgabjgosKICAgICAgICAgICAgICAgICAgICBzaXplX2xpbmUgPSBzaXplX2xpbmUuc3BsaXQoYiI7IiwgMSlbMF0KICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBzaXplID0gaW50KHNpemVfbGluZSwgMTYpCiAgICAgICAgICAgICAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICAgICAgaWYgc2l6ZSA9PSAwOiAgICAgICAgICAgICAgICAgICMg44OI44Os44O844Op44O86Kqt44G/5o2o44GmCiAgICAgICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICAgICAgdHJhaWxlciA9IHNlbGYucmZpbGUucmVhZGxpbmUoKQogICAgICAgICAgICAgICAgICAgICAgICBpZiB0cmFpbGVyIGluIChiIlxyXG4iLCBiIlxuIiwgYiIiKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgIG91dCArPSBzZWxmLnJmaWxlLnJlYWQoc2l6ZSkKICAgICAgICAgICAgICAgIHNlbGYucmZpbGUucmVhZCgyKSAgICAgICAgICAgICAjIOODgeODo+ODs+OCr+acq+WwvuOBriBDUkxGCiAgICAgICAgICAgIHJldHVybiBieXRlcyhvdXQpIG9yIE5vbmUKCiAgICAgICAgbGVuZ3RoID0gaW50KHNlbGYuaGVhZGVycy5nZXQoIkNvbnRlbnQtTGVuZ3RoIikgb3IgMCkKICAgICAgICByZXR1cm4gc2VsZi5yZmlsZS5yZWFkKGxlbmd0aCkgaWYgbGVuZ3RoIGVsc2UgTm9uZQoKICAgIGRlZiBfZW1pdChzZWxmLCBkYXRhKToKICAgICAgICAiIiJjaHVua2VkIOOBrjHliIbniYfjgpLpgIHjgovjgIIKCiAgICAgICAg5pS56KGM5Y2Y5L2N44Gn5ZG844G244Gu44GnIHNlbmQg5Zue5pWw44Gv44OI44O844Kv44Oz5pWw44Gr5ZCM5q+U5L6L77yIMeODkOOCpOODiOavjuOBq+OBr+mAgeOCieOBquOBhO+8ieOAggogICAgICAgICIiIgogICAgICAgIGlmIG5vdCBkYXRhOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiJXhcclxuIiAlIGxlbihkYXRhKSkKICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJ5dGVzKGRhdGEpKQogICAgICAgIHNlbGYud2ZpbGUud3JpdGUoYiJcclxuIikKICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKCiAgICBkZWYgX3NlbmRfanNvbihzZWxmLCBjb2RlLCBvYmosIGNsb3NlPUZhbHNlLCBleHRyYV9oZWFkZXJzPSgpKToKICAgICAgICAiIiJKU09OIOOCkiAxIOeZuuOBp+i/lOOBme+8iOeuoeeQhuOCqOODs+ODieODneOCpOODs+ODiOODu+OCqOODqeODvOW/nOetlOOBruWFsemAmuWHpueQhu+8ieOAgiIiIgogICAgICAgIGJvZHkgPSBvYmogaWYgaXNpbnN0YW5jZShvYmosIGJ5dGVzKSBlbHNlIGpzb24uZHVtcHMob2JqKS5lbmNvZGUoKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGNvZGUpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ2FjaGUtQ29udHJvbCIsICJuby1zdG9yZSIpCiAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gZXh0cmFfaGVhZGVyczoKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGNsb3NlOgogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb25uZWN0aW9uIiwgImNsb3NlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGJvZHkpCiAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQoKICAgIGRlZiBfYXV0aG9yaXplZChzZWxmKToKICAgICAgICAjIEFQSV9QQVNTIOOBjOepuiA9IOiqjeiovOOBquOBl++8iOWFrOmWi++8ieOAglVJIOOCkumWi+OBhOOBn+OBoOOBkeOBp+S9v+OBiOOCi+OCiOOBhuOBq+OBmeOCi+OAggogICAgICAgIGlmIG5vdCBFWFBFQ1RFRF9BVVRIOgogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGlmIHNlbGYucGF0aC5zcGxpdCgiPyIpWzBdIGluIE9QRU5fUEFUSFM6CiAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgIyDluLjmlbDmmYLplpPmr5TovIPvvIjjgr/jgqTjg5/jg7PjgrDmlLvmkoPjga7nt6nlkozvvInjgIJzdHIg54mIIGNvbXBhcmVfZGlnZXN0IOOBrwogICAgICAgICMg6Z2eIEFTQ0lJIOOBpyBUeXBlRXJyb3Ig44Gr44Gq44KL44Gu44GnIGJ5dGVzIOOBp+avlOi8g+OBmeOCi+OAggogICAgICAgICMgT1BUSU9OUyDjga8gX3JlbGF5IOWGkumgreOBp+ODreODvOOCq+ODq+WNs+etlOa4iOOBv++8iOOBk+OBk+OBq+OBr+WxiuOBi+OBquOBhO+8ieOAggogICAgICAgIGdvdCA9IHNlbGYuaGVhZGVycy5nZXQoIkF1dGhvcml6YXRpb24iLCAiIikuZW5jb2RlKCJsYXRpbi0xIikKICAgICAgICByZXR1cm4gaG1hYy5jb21wYXJlX2RpZ2VzdChnb3QsIEVYUEVDVEVEX0FVVEguZW5jb2RlKCJhc2NpaSIpKQoKICAgIGRlZiBfZGVueShzZWxmKToKICAgICAgICAjIDQwMSDjgafmjqXntprjgpLplonjgZjjgovjgIJLZWVwLUFsaXZlIOOBruOBvuOBvuaui+OBmeOBqOOAgeasoeOBruODquOCr+OCqOOCueODiOOBrgogICAgICAgICMg44Oc44OH44Kj6Kqt44G/5Y+W44KK5L2N572u44GM44Ga44KM44GmIEhUVFAg44Oq44Kv44Ko44K544OI44K544Oi44Ks44Oq44Oz44Kw44GM6LW344GN44GG44KL44CCCiAgICAgICAgc2VsZi5fc2VuZF9qc29uKDQwMSwgeyJlcnJvciI6ICJ1bmF1dGhvcml6ZWQifSwgY2xvc2U9VHJ1ZSwKICAgICAgICAgICAgICAgICAgICAgICAgZXh0cmFfaGVhZGVycz1bKCJXV1ctQXV0aGVudGljYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICdCYXNpYyByZWFsbT0iT2xsYW1hIEFQSSInKV0pCgogICAgZGVmIF9zZXJ2ZV91aShzZWxmKToKICAgICAgICAiIiLjg4Hjg6Pjg4Pjg4ggVUkg44KS6L+U44GZ44CC6KqN6Ki85LiN6KaB77yI6Z2Z55qE44OV44Kh44Kk44Or44Gu44G/77yJ44CCCgogICAgICAgIFVJIOOBryAxIOODquOCr+OCqOOCueODiOOBp+e1guOCj+OCi+OBruOBpyAqKuaOpee2muOCkumWieOBmOOCiyoq44CC6KqN6Ki844OR44K544Go6YGV44GECiAgICAgICAg44GT44GT44Gn44Gv44Oc44OH44Kj44KS6Kqt44G+44Gq44GE44Gf44KB44CBa2VlcC1hbGl2ZSDjgpLntq3mjIHjgZnjgovjgagKICAgICAgICBDb250ZW50LUxlbmd0aCDku5jjgY0gUE9TVCDjgafmrKHjga7jg6rjgq/jgqjjgrnjg4jkvY3nva7jgYzjgZrjgozjgosKICAgICAgICDvvIjjg6rjgq/jgqjjgrnjg4jjgrnjg57jgrDjg6rjg7PjgrDvvInjgIIKICAgICAgICAiIiIKICAgICAgICBzZWxmLmNsb3NlX2Nvbm5lY3Rpb24gPSBUcnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKSAgICAgICAgICAjIOaDs+WumuWkluOBruODnOODh+OCo+OBr+aNqOOBpuOCi++8iOiqreOBv+aui+OBl+mYsuatou+8iQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICBpZiBub3QgQ0hBVF9CWVRFUzoKICAgICAgICAgICAgYm9keSA9IChiImxsYW1hLWNoYXQuaHRtbCBub3QgZm91bmQuICIKICAgICAgICAgICAgICAgICAgICBiInN0YXJ0IHRoZSBwcm94eSB3aXRoIC0tY2hhdC1odG1sIDxwYXRoPi4iKQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoNTAzKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCAidGV4dC9wbGFpbjsgY2hhcnNldD11dGYtOCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihib2R5KSkpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShib2R5KQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjAwKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtVHlwZSIsICJ0ZXh0L2h0bWw7IGNoYXJzZXQ9dXRmLTgiKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbnRlbnQtTGVuZ3RoIiwgc3RyKGxlbihDSEFUX0JZVEVTKSkpCiAgICAgICAgIyBVSSDjga/mr47lm57lkIzjgZjlhoXlrrnjgILjg5fjg63jgq3jgrfmm7TmlrDmmYLjgavjg5bjg6njgqbjgrbjgYzlj6TjgYTniYjjgpLkvb/jgo/jgarjgYTjgojjgYYgbm8tc3RvcmXjgIIKICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDYWNoZS1Db250cm9sIiwgIm5vLXN0b3JlIikKICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIkhFQUQiOgogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKENIQVRfQllURVMpCgogICAgZGVmIF9zZXJ2ZV9xdW90YShzZWxmKToKICAgICAgICAiIiJLYWdnbGUg44GuIEdQVSDjgq/jgqnjg7zjgr/mrovph4/jgpLov5TjgZnjgIIKCiAgICAgICAgVUkg44Gu44OY44OD44OA44Gr44CM5q6L44KKIDA4OjU544CN44Go5Ye644GZ44Gf44KB44Gr5L2/44GG44CCCiAgICAgICAg6LOH5qC85oOF5aCx77yIa2FnZ2xlLmpzb24g44G+44Gf44Gv55Kw5aKD5aSJ5pWw77yJ44GM54Sh44GE44CB44G+44Gf44GvIEFQSSDjgYwKICAgICAgICDokL3jgaHjgabjgYTjgovloLTlkIjjga8gYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OBpiBVSSDlgbTjgafpnZ7ooajnpLrjgavjgZnjgovjgIIKICAgICAgICAiIiIKICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZX0KICAgICAgICBpZiBBUkdTLm5vX3F1b3RhOgogICAgICAgICAgICAjIEthZ2dsZSDku6XlpJbjga7jg63jg7zjgqvjg6vvvIh1enUg44Gq44Gp77yJ44Gn44Gv44Kv44Kp44O844K/44Gu5qaC5b+144GM54Sh44GE44CCCiAgICAgICAgICAgICMgYXZhaWxhYmxlPWZhbHNlIOOCkui/lOOBl+OAgVVJIOWBtOOBr+aui+OCiuaZgumWk+OCkumdnuihqOekuuOBq+OBmeOCi+OAggogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7ImF2YWlsYWJsZSI6IEZhbHNlLCAicmVhc29uIjogIm5vX3F1b3RhIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHVzZXIsIGtleSA9IGtjX2thZ2dsZV9jcmVkZW50aWFscygpCiAgICAgICAgICAgIGlmIHVzZXIgYW5kIGtleToKICAgICAgICAgICAgICAgIHBheWxvYWQgPSBrY19xdW90YV9zdW1tYXJ5KAogICAgICAgICAgICAgICAgICAgIGtjX2ZldGNoX3F1b3RhKHVzZXIsIGtleSwgdGltZW91dD0xNSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXlsb2FkID0geyJhdmFpbGFibGUiOiBGYWxzZSwgImVycm9yIjogdHlwZShlKS5fX25hbWVfX30KCiAgICAgICAgIyDmrovph4/jga/liLvjgIXlpInjgo/jgovjgIJfc2VuZF9qc29uIOOBryBuby1zdG9yZSDjgpLku5jjgZHjgovjgIIKICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCBwYXlsb2FkKQoKICAgIGRlZiBfc2VydmVfc3RvcChzZWxmKToKICAgICAgICAiIiLjgZPjga7kuK3ntpnjgajkuIrmtYHjga7jgrvjg4Pjgrfjg6fjg7PjgpLmraLjgoHjgovjgIIKCiAgICAgICAg5ZG844Gz5Ye644GX5p2h5Lu277yIUE9TVCArIFgtS2MtU3RvcCArIOiqjeiovO+8ieOBryBfc3RvcF9hbGxvd2VkKCkg44Gn56K66KqN5riI44G/44CCCgogICAgICAgIOODluODqeOCpuOCtuOBruWBnOatouODnOOCv+ODs+OBi+OCieWPqeOBj+OAguS4ree2meOBjOiQveOBoeOCi+OBruOBp+W/nOetlOOCkui/lOOBl+OBpuOBi+OCiQogICAgICAgIOODl+ODreOCu+OCueOCkue1guS6huOBleOBm+OCi++8iOWFiOOBq+iQveOBqOOBmeOBqOODluODqeOCpuOCtuWBtOOBjOmAmuS/oeOCqOODqeODvOOBq+OBquOCi++8ieOAggoKICAgICAgICBLYWdnbGUg5LiK44Gn44GvIGtlZXBhbGl2ZSDjg6vjg7zjg5fjgoLmraLjgoHjgovjgZ/jgoHjgIEvdG1wL2tlZXBhbGl2ZS5jb25mIOOBqwogICAgICAgIFJVTj0wIOOCkuabuOOBjei+vOOCgOOAguOBk+OCjOOBq+OCiOOCiiBLYWdnbGUg44Kr44O844ON44Or44K744OD44K344On44Oz44GM57WC5LqG44GX44CBCiAgICAgICAgR1BVIOOCr+OCqeODvOOCv+OCkumWi+aUvuOBp+OBjeOCi+OAggoKICAgICAgICAtLXN0b3AtcGlkIOOBjOaMh+WumuOBleOCjOOBpuOBhOOCjOOBsOOAgeOBneOBriBQSUTvvIjjg6njg7Pjg4Hjg6Pjg7zvvInjgavjgoIKICAgICAgICBTSUdURVJNIOOCkumAgeOCi+OAgnV6dSDjga8gcnVuLXV6dS1zZXJ2ZXIuY29tbWFuZCDjgYzlrZDjg5fjg63jgrvjgrnjgpIKICAgICAgICDmirHjgYjjgabjgYTjgovjga7jgafjgIHopqrjgpLokL3jgajjgZvjgbAgdHJhcCBjbGVhbnVwIOOBp+WFqOmDqOeJh+S7mOOBj+OAggogICAgICAgICIiIgogICAgICAgICMgS2FnZ2xlIOeJiCBrZWVwYWxpdmUg44Or44O844OX44Gr44KC5YGc5q2i44KS5Lyd44GI44KLCiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oIi90bXAva2VlcGFsaXZlLmNvbmYiLCAidyIpIGFzIGY6CiAgICAgICAgICAgICAgICBmLndyaXRlKCJSVU49MFxuIikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgICAgIHNlbGYuX3NlbmRfanNvbigyMDAsIHsic3RvcHBlZCI6IFRydWUsICJzY29wZSI6ICJyZWxheSJ9KQoKICAgICAgICBkZWYgX2RpZSgpOgogICAgICAgICAgICB0aW1lLnNsZWVwKDAuNSkKICAgICAgICAgICAgIyDlhYjjgavjg6njg7Pjg4Hjg6Pjg7zjgpLokL3jgajjgZnvvIjlrZDjg5fjg63jgrvjgrnjgaggVUkg44KC5LiA57eS44Gr5raI44GI44KL77yJCiAgICAgICAgICAgIGlmIEFSR1Muc3RvcF9waWQ6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgb3Mua2lsbChBUkdTLnN0b3BfcGlkLCBzaWduYWwuU0lHVEVSTSkKICAgICAgICAgICAgICAgIGV4Y2VwdCAoUHJvY2Vzc0xvb2t1cEVycm9yLCBQZXJtaXNzaW9uRXJyb3IpOgogICAgICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgICAgIHRpbWUuc2xlZXAoMC41KQogICAgICAgICAgICBvcy5fZXhpdCgwKQoKICAgICAgICB0aHJlYWRpbmcuVGhyZWFkKHRhcmdldD1fZGllLCBkYWVtb249VHJ1ZSkuc3RhcnQoKQoKICAgIGRlZiBfcmVsYXkoc2VsZik6CiAgICAgICAgIyBDT1JTIOODl+ODquODleODqeOCpOODiOOBr+iqjeiovOS4jeimgeODu+ODreODvOOCq+ODq+WNs+etlOOAguS4iua1geOBuOa1geOBleOBquOBhAogICAgICAgICMg77yI57Sg6YCa44GX44GZ44KL44Go5pyq6KqN6Ki844Oq44Kv44Ko44K544OI44GM5LiK5rWB44Gr5bGK44GN44CB44Oc44OH44Kj5b6F44Gh44GnCiAgICAgICAgIyAg44OP44Oz44OJ44Op44K544Os44OD44OJ44KC5Y2g44KB44KJ44KM44KL77yJ44CCCiAgICAgICAgaWYgc2VsZi5jb21tYW5kID09ICJPUFRJT05TIjoKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDIwNCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgIyAvaGVhbHRoeiDjga/jg63jg7zjgqvjg6vjgafljbPnrZTjgZnjgovvvIjkuIrmtYHjga7mnInnhKHjgavplqLkv4LjgarjgY/nlJ/lrZjnorroqo3jgafjgY3jgovvvIkKICAgICAgICBpZiBzZWxmLnBhdGguc3BsaXQoIj8iKVswXSA9PSAiL2hlYWx0aHoiOgogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oMjAwLCB7InN0YXR1cyI6ICJvayJ9KQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgcGF0aF9vbmx5ID0gc2VsZi5wYXRoLnNwbGl0KCI/IilbMF0KICAgICAgICBpZiBwYXRoX29ubHkgaW4gVUlfUEFUSFM6CiAgICAgICAgICAgIHNlbGYuX3NlcnZlX3VpKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgICMgLS0tLSDnrqHnkIbjgqjjg7Pjg4njg53jgqTjg7Pjg4jvvIjjg63jg7zjgqvjg6vlsILnlKjvvIktLS0tCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvcXVvdGEiOgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9xdW90YSgpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGlmIHBhdGhfb25seSA9PSAiL3N0b3AiOgogICAgICAgICAgICAjIOiQveOBqOOBm+OCi+OBruOBryBVSSDjga7lgZzmraLjg5zjgr/jg7PvvIhQT1NUICsgWC1LYy1TdG9w77yJ44Gg44GR44CCCiAgICAgICAgICAgICMg5Lul5YmN44Gv6KqN6Ki844KI44KK5YmN44Gr5Yem55CG44GX44Gm44GE44Gf44Gf44KB44CBQVBJX1BBU1Mg44KS6Kit5a6a44GX44Gm44GE44Gm44KCCiAgICAgICAgICAgICMg6Kqw44Gn44KCIEdFVCAvc3RvcCDjgacgS2FnZ2xlIOOCu+ODg+OCt+ODp+ODs+OCkuatouOCgeOCieOCjOOBn+OAggogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgICAgICBpZiBzZWxmLmNvbW1hbmQgIT0gIlBPU1QiIG9yIHNlbGYuaGVhZGVycy5nZXQoU1RPUF9IRUFERVIpICE9ICIxIjoKICAgICAgICAgICAgICAgIHNlbGYuX3NlbmRfanNvbig0MDUsIHsiZXJyb3IiOiAiUE9TVCB3aXRoICVzOiAxIHJlcXVpcmVkIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICUgU1RPUF9IRUFERVJ9LCBjbG9zZT1UcnVlKQogICAgICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgICAgIGlmIG5vdCBzZWxmLl9hdXRob3JpemVkKCk6CiAgICAgICAgICAgICAgICBzZWxmLl9kZW55KCkKICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICBzZWxmLl9zZXJ2ZV9zdG9wKCkKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgaWYgcGF0aF9vbmx5ID09ICIvZmF2aWNvbi5pY28iOgogICAgICAgICAgICAjIOODluODqeOCpuOCtuOBjOWLneaJi+OBq+WPluOCiuOBq+adpeOCi+OAgjQwNCDjgpLov5TjgZnjgajjg63jgrDjgYzln4vjgb7jgovjgIIKICAgICAgICAgICAgc2VsZi5jbG9zZV9jb25uZWN0aW9uID0gVHJ1ZQogICAgICAgICAgICBzZWxmLnNlbmRfcmVzcG9uc2UoMjA0KQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsICIwIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICBpZiBub3Qgc2VsZi5fYXV0aG9yaXplZCgpOgogICAgICAgICAgICAjIDQwMSDjgpLov5TjgZnliY3jgavjg5zjg4fjgqPjgpLpo5/jgaPjgabjgYrjgY/jgIJQT1NUIOOBruODnOODh+OCo+OCkuiqreOBvuOBmuOBqwogICAgICAgICAgICAjIOaOpee2muOCkuWIh+OCi+OBqOOCr+ODqeOCpOOCouODs+ODiOWBtOOBjCBSU1Qg44KS5Y+X44GR44Gm44CMNDAxIOOCiOOCiuWJjeOBqwogICAgICAgICAgICAjIOaOpee2muOBjOWIh+OCjOOBn+OAjeOBq+imi+OBiOOCi++8iGN1cmwg44GvIFJTVCDjgpLjgqjjg6njg7zooajnpLrvvInjgIIKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5fcmVhZF9ib2R5KCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICAgICAgc2VsZi5fZGVueSgpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGJvZHkgPSBzZWxmLl9yZWFkX2JvZHkoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICMg5LiN5q2j44GqIENvbnRlbnQtTGVuZ3RoIC8g5aOK44KM44GfIGNodW5rZWQg44Gv5LiK5rWB44Gr5rWB44GV44Gq44GE44CCCiAgICAgICAgICAgICMg5pyq5o2V5o2J44Gg44Go44K544K/44OD44Kv44OA44Oz44OX44GuNTAw44Gr44Gq44KL77yI6KqN6Ki85oiQ5Yqf5YG044Gg44GR6Z2e5a++56ew44Gr44Gq44KJ44Gq44GE44KI44GG77yJCiAgICAgICAgICAgIHNlbGYuY2xvc2VfY29ubmVjdGlvbiA9IFRydWUKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDQwMCkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCAiMCIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbm5lY3Rpb24iLCAiY2xvc2UiKQogICAgICAgICAgICBzZWxmLmVuZF9oZWFkZXJzKCkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlmIF9pc19nZW5lcmF0aW9uKHBhdGhfb25seSk6CiAgICAgICAgICAgIF9yZXNldF9rZWVwYWxpdmUoKSAgIyDnlJ/miJDjg6rjgq/jgqjjgrnjg4jvvJ3jgqLjgq/jg4bjgqPjg5Pjg4bjgqPjgYLjgoog4oaSIDYw5YiG44Oq44K744OD44OICiAgICAgICAgIyDjgrXjg7Pjg5fjg6rjg7PjgrDkuIvpmZDjga7lvLfliLbvvIgtLWVuZm9yY2Utc2FtcGxpbmcg5pmC44Gu44G/44CC5pei5a6a44Gv57Sg6YCa44GX77yJCiAgICAgICAgYm9keSA9IGVuZm9yY2Vfc2FtcGxpbmcoYm9keSwgcGF0aF9vbmx5KQogICAgICAgIHVybCA9IEFSR1MudXBzdHJlYW0ucnN0cmlwKCIvIikgKyBzZWxmLnBhdGgKICAgICAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBtZXRob2Q9c2VsZi5jb21tYW5kKQogICAgICAgIGZvciBrZXksIHZhbHVlIGluIHNlbGYuaGVhZGVycy5pdGVtcygpOgogICAgICAgICAgICAjIGNvbnRlbnQtbGVuZ3RoIOOBryBib2R5IOWGjeani+evieW+jOOBruWun+mVt+OBp+mAgeOCiuebtOOBmeOBn+OCgeOBk+OBk+OBp+aNqOOBpuOCiwogICAgICAgICAgICAjIO+8iOOCr+ODqeOCpOOCouODs+ODiOWApOOBruOBvuOBvui7oumAgeOBmeOCi+OBqCBjaHVua2VkICsgQ0wg5L2155So44Gn44OH44K344Oz44Kv44GZ44KL77yJCiAgICAgICAgICAgIGlmIGtleS5sb3dlcigpIGluIEhPUF9CWV9IT1AgfCB7Imhvc3QiLCAiYXV0aG9yaXphdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImNvbnRlbnQtbGVuZ3RoIn06CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZXEuYWRkX2hlYWRlcihrZXksIHZhbHVlKQogICAgICAgIGlmIGJvZHkgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJlcS5hZGRfaGVhZGVyKCJDb250ZW50LUxlbmd0aCIsIHN0cihsZW4oYm9keSkpKQoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3AgPSB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJlcSwgdGltZW91dD1VUFNUUkVBTV9USU1FT1VUKQogICAgICAgIGV4Y2VwdCB1cmxsaWIuZXJyb3IuSFRUUEVycm9yIGFzIGV4YzoKICAgICAgICAgICAgIyDkuIrmtYHjgYwgNHh4LzV4eCDjgpLov5TjgZfjgZ/loLTlkIjjgoLmnKzmlofjgpLjgZ3jga7jgb7jgb7ov5TjgZkKICAgICAgICAgICAgcGF5bG9hZCA9IGV4Yy5yZWFkKCkKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKGV4Yy5jb2RlKQogICAgICAgICAgICBjdHlwZSA9IGV4Yy5oZWFkZXJzLmdldCgiQ29udGVudC1UeXBlIiwgImFwcGxpY2F0aW9uL2pzb24iKQogICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCBjdHlwZSkKICAgICAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiQ29udGVudC1MZW5ndGgiLCBzdHIobGVuKHBheWxvYWQpKSkKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIHNlbGYuY29tbWFuZCAhPSAiSEVBRCI6CiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKHBheWxvYWQpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246ICAjIOaOpee2muWkseaVl+OBquOBqQogICAgICAgICAgICAjIOWGhemDqOS+i+WkluOBruips+e0sOOBr+OCr+ODqeOCpOOCouODs+ODiOOBq+a4oeOBleOBquOBhO+8iOaDheWgsemcsuWHuumYsuatou+8ieOAggogICAgICAgICAgICAjIOWOn+WboOOBr+OCteODvOODkOWBtCBzdGRlcnIg44Gr44Gg44GR5pu444GP44CCCiAgICAgICAgICAgIHN5cy5zdGRlcnIud3JpdGUoIiAgW3Byb3h5XSB1cHN0cmVhbSBlcnJvcjogJXNcbiIKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAlIChzeXMuZXhjX2luZm8oKVsxXSwpKQogICAgICAgICAgICBzZWxmLl9zZW5kX2pzb24oNTAyLCB7ImVycm9yIjogInVwc3RyZWFtIHVucmVhY2hhYmxlIn0pCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB3aXRoIHJlc3A6CiAgICAgICAgICAgIHNlbGYuc2VuZF9yZXNwb25zZShyZXNwLnN0YXR1cykKICAgICAgICAgICAgZm9yIGtleSwgdmFsdWUgaW4gcmVzcC5oZWFkZXJzLml0ZW1zKCk6CiAgICAgICAgICAgICAgICBpZiBrZXkubG93ZXIoKSBpbiBIT1BfQllfSE9QIG9yIGtleS5sb3dlcigpID09ICJjb250ZW50LWxlbmd0aCI6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoa2V5LCB2YWx1ZSkKICAgICAgICAgICAgIyBIRUFEIOOBr+acrOaWh+OBquOBl+OAgmNodW5rZWQg44KS5a6j6KiA44GZ44KL44Go44Kv44Op44Kk44Ki44Oz44OI44GM57WC56uv44OB44Oj44Oz44Kv44KSCiAgICAgICAgICAgICMg5b6F44Gj44Gm44OP44Oz44Kw44GZ44KL77yIQ29udGVudC1MZW5ndGgg44KC5rih44Gb44Gq44GE44Gu44Gn5a6j6KiA44GX44Gq44GE77yJ44CCCiAgICAgICAgICAgICMgMjA0LzMwNCDjga/mnKzmlofjgpLmjIHjgabjgarjgYTjgIJjaHVua2VkIOOCkuWuo+iogOOBmeOCi+OBqOWOs+agvOOBqgogICAgICAgICAgICAjIOOCr+ODqeOCpOOCouODs+ODiOOBjCBwcm90b2NvbCBlcnJvciDjgavjgZnjgovjgIIKICAgICAgICAgICAgbm9fYm9keSA9IHJlc3Auc3RhdHVzIGluICgyMDQsIDMwNCkgb3Igc2VsZi5jb21tYW5kID09ICJIRUFEIgogICAgICAgICAgICBpZiBub3Qgbm9fYm9keToKICAgICAgICAgICAgICAgICMg6ZW344GV5pyq55+l44Gu44K544OI44Oq44O844Og44Go44GX44Gm5Lit57aZ44GZ44KLCiAgICAgICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJUcmFuc2Zlci1FbmNvZGluZyIsICJjaHVua2VkIikKICAgICAgICAgICAgc2VsZi5lbmRfaGVhZGVycygpCiAgICAgICAgICAgIGlmIG5vX2JvZHk6CiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgIyByZXNwLnJlYWQoTikg44GvIEhUVFBSZXNwb25zZS5fcmVhZGludG9fY2h1bmtlZCDjga7mgKfos6rjgacKICAgICAgICAgICAgICAgICMgKipOIOODkOOCpOODiOaPg+OBhuOBvuOBp+iqreOBv+e2muOBkeOCiyoq44Gf44KB44CB44K544OI44Oq44O844Og44GM5YWo6Z2i44Gr5rqc44G+44KLCiAgICAgICAgICAgICAgICAjIO+8iE9sbGFtYSDjga4gTkRKU09OIOOBjOWFqOODiOODvOOCr+ODs+WIhuiyr+OBvuOBo+OBpuOBi+OCieS4gOaWieOBq+WxiuOBj++8ieOAggogICAgICAgICAgICAgICAgIyByZWFkbGluZShDSFVOSykg44Gv5pS56KGM44GLIENIVU5LIOODkOOCpOODiOOBp+i/lOOCi+OBruOBp+OAgemAkOasoeaAp+OCkgogICAgICAgICAgICAgICAgIyDkv53jgaPjgZ/jgb7jgb4gMSDjg5DjgqTjg4jjgZrjgaTjga4gcmVhZCDjg6vjg7zjg5fjgojjgoogQ1BVIOOCkumjn+OCj+OBquOBhOOAggogICAgICAgICAgICAgICAgd2hpbGUgVHJ1ZToKICAgICAgICAgICAgICAgICAgICBsaW5lID0gcmVzcC5yZWFkbGluZShDSFVOSykKICAgICAgICAgICAgICAgICAgICBpZiBub3QgbGluZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgICAgICBzZWxmLl9lbWl0KGxpbmUpCiAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiMFxyXG5cclxuIikKICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICBleGNlcHQgKEJyb2tlblBpcGVFcnJvciwgQ29ubmVjdGlvblJlc2V0RXJyb3IpOgogICAgICAgICAgICAgICAgIyDjgq/jg6njgqTjgqLjg7Pjg4jjgYzliIfjgaPjgZ/vvIhDdHJsLUMg44Gq44Gp77yJ44CC5q2j5bi457O744CCCiAgICAgICAgICAgICAgICBwYXNzCgogICAgZG9fR0VUID0gX3JlbGF5CiAgICBkb19QT1NUID0gX3JlbGF5CiAgICBkb19QVVQgPSBfcmVsYXkKICAgIGRvX1BBVENIID0gX3JlbGF5CiAgICBkb19ERUxFVEUgPSBfcmVsYXkKICAgIGRvX0hFQUQgPSBfcmVsYXkKICAgIGRvX09QVElPTlMgPSBfcmVsYXkKCgpjbGFzcyBTZXJ2ZXIoc29ja2V0c2VydmVyLlRocmVhZGluZ01peEluLCBodHRwLnNlcnZlci5IVFRQU2VydmVyKToKICAgIGRhZW1vbl90aHJlYWRzID0gVHJ1ZQogICAgYWxsb3dfcmV1c2VfYWRkcmVzcyA9IFRydWUKCgpkZWYgbWFpbigpOgogICAgZ2xvYmFsIEFSR1MsIEVYUEVDVEVEX0FVVEgKCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXBvcnQiLCB0eXBlPWludCwgZGVmYXVsdD0xMTQzNSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0taG9zdCIsIGRlZmF1bHQ9IjEyNy4wLjAuMSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXVwc3RyZWFtIiwgZGVmYXVsdD0iaHR0cDovLzEyNy4wLjAuMToxMTQzNCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoYXQtaHRtbCIsIGRlZmF1bHQ9b3MuZW52aXJvbi5nZXQoIkNIQVRfSFRNTCIsICIiKSwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0i44OB44Oj44OD44OIIFVJIOOBriBIVE1M77yILyDjgafphY3kv6HjgZnjgovvvIkiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zdG9wLXBpZCIsIHR5cGU9aW50LCBkZWZhdWx0PTAsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9Ii9zdG9wIOaZguOBqyBTSUdURVJNIOOCkumAgeOCiyBQSUTjgIIiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInV6dSDjga7jgojjgYbjgavjg6njg7Pjg4Hjg6Pjg7zjgYzlrZDjg5fjg63jgrvjgrnjgpLmirHjgYjjgovloLTlkIjjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIuODqeODs+ODgeODo+ODvOOBriBQSUQg44KS5rih44GZ44Go5b6M54mH5LuY44GR44GU44Go5q2i44G+44KLIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm8tcXVvdGEiLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSIvcXVvdGEg44KS54Sh5Yq55YyW44GZ44KL77yIS2FnZ2xlIOS7peWkluOBruODreODvOOCq+ODq+eUqO+8iSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVuZm9yY2Utc2FtcGxpbmciLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJjaGF0L2NvbXBsZXRpb25zIOOBriB0ZW1wZXJhdHVyZTw9MCDjgpIgJS4xZiDjgavjgIEiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInJlcGV0aXRpb25fcGVuYWx0eSDjgpIgJS4yZiDku6XkuIrjgavnn6/mraPjgZnjgosiCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJSAoVEVNUF9GTE9PUiwgUlBfRkxPT1IpKQogICAgQVJHUyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKCiAgICB1c2VyID0gb3MuZW52aXJvbi5nZXQoIkFQSV9VU0VSIikgb3IgIm9sbGFtYSIKICAgIHBhc3N3b3JkID0gb3MuZW52aXJvbi5nZXQoIkFQSV9QQVNTIikgb3IgIiIKICAgIGlmIHBhc3N3b3JkOgogICAgICAgIHRva2VuID0gYmFzZTY0LmI2NGVuY29kZShmInt1c2VyfTp7cGFzc3dvcmR9Ii5lbmNvZGUoKSkuZGVjb2RlKCkKICAgICAgICBFWFBFQ1RFRF9BVVRIID0gIkJhc2ljICIgKyB0b2tlbgogICAgZWxzZToKICAgICAgICAjIOaXouWumuOBr+iqjeiovOOBquOBl+OAglVSTCDjgpLnn6XjgaPjgabjgYTjgozjgbDoqrDjgafjgoLkvb/jgYjjgovngrnjgaDjgZHms6jmhI/jgIIKICAgICAgICBwcmludCgiYXV0aDogT0ZGIChBUElfUEFTUyDmnKroqK3lrpopIiwgZmx1c2g9VHJ1ZSkKCiAgICBnbG9iYWwgQ0hBVF9CWVRFUwogICAgaWYgQVJHUy5jaGF0X2h0bWw6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4oQVJHUy5jaGF0X2h0bWwsICJyYiIpIGFzIGY6CiAgICAgICAgICAgICAgICBDSEFUX0JZVEVTID0gZi5yZWFkKCkKICAgICAgICBleGNlcHQgT1NFcnJvciBhcyBleGM6CiAgICAgICAgICAgICMgVUkg54Sh44GX44Gn44KCIEFQSSDjga/li5XjgY/jgILokL3jgajjgZXjgZrorablkYrjgaDjgZHlh7rjgZnjgIIKICAgICAgICAgICAgc3lzLnN0ZGVyci53cml0ZSgiY2hhdCB1aSBub3QgbG9hZGVkOiAlc1xuIiAlIChleGMsKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiY2hhdCB1aTogJXMgKCVkIEIpIiAlIChBUkdTLmNoYXRfaHRtbCwgbGVuKENIQVRfQllURVMpKSwKICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKCiAgICBodHRwZCA9IFNlcnZlcigoQVJHUy5ob3N0LCBBUkdTLnBvcnQpLCBIYW5kbGVyKQogICAgcHJpbnQoZiJhdXRoIHByb3h5OiB7QVJHUy5ob3N0fTp7QVJHUy5wb3J0fSAtPiB7QVJHUy51cHN0cmVhbX0gIgogICAgICAgICAgZiIodXNlcj17dXNlcn0pIiwgZmx1c2g9VHJ1ZSkKICAgIHRyeToKICAgICAgICBodHRwZC5zZXJ2ZV9mb3JldmVyKCkKICAgIGV4Y2VwdCBLZXlib2FyZEludGVycnVwdDoKICAgICAgICBwYXNzCiAgICBmaW5hbGx5OgogICAgICAgIGh0dHBkLnNlcnZlcl9jbG9zZSgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo='), ('llama-chat.html', 'PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImphIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0idXRmLTgiIC8+CiAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIiAvPgogIDx0aXRsZT7jg4Hjg6Pjg4Pjg4hVSTwvdGl0bGU+CiAgPHN0eWxlPgogICAgOnJvb3QgeyAtLWJnOiMwYjBjMTA7IC0tcGFuZWw6IzExMTMxYTsgLS1pbms6I2U5ZWVmMjsgLS1tdXRlZDojOWZiMGJmOyAtLWFjYzojNmFhMWZmOyAtLWxpbmU6IzFlMjIzMDsgfQogICAgKntib3gtc2l6aW5nOmJvcmRlci1ib3h9CiAgICBodG1sLGJvZHl7aGVpZ2h0OjEwMCV9CiAgICBib2R5e21hcmdpbjowOyBmb250OjE0cHgvMS40NSBzeXN0ZW0tdWksIC1hcHBsZS1zeXN0ZW0sIFNlZ29lIFVJLCBSb2JvdG8sIFVidW50dSwgQ2FudGFyZWxsLCBOb3RvIFNhbnMsIEhlbHZldGljYSwgQXJpYWw7IGJhY2tncm91bmQ6dmFyKC0tYmcpOyBjb2xvcjp2YXIoLS1pbmspfQogICAgLndyYXB7ZGlzcGxheTpncmlkOyBncmlkLXRlbXBsYXRlLXJvd3M6YXV0byAxZnIgYXV0bzsgaGVpZ2h0OjEwMCV9CiAgICBoZWFkZXIsIGZvb3RlcntiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTt9CiAgICBoZWFkZXJ7ZGlzcGxheTpmbGV4OyBnYXA6Ljc1cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXA7IHBhZGRpbmc6LjZyZW0gLjhyZW07IHBvc2l0aW9uOnN0aWNreTsgdG9wOjA7IHotaW5kZXg6MTA7IGJvcmRlci1ib3R0b206MXB4IHNvbGlkIHZhcigtLWxpbmUpfQogICAgaGVhZGVyIC50aXRsZXtmb250LXdlaWdodDo2MDA7IGxldHRlci1zcGFjaW5nOi4ycHh9CgogICAgaGVhZGVyIC5icmFuZHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4xNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucHJvbW97Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLWFjYyk7IHRleHQtZGVjb3JhdGlvbjpub25lOyBvcGFjaXR5Oi45fQogICAgaGVhZGVyIC5wcm9tbzpob3ZlcntvcGFjaXR5OjE7IHRleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmV9CgogICAgLyogSW5mbyBkaWFsb2cgKi8KICAgIGRpYWxvZy5pbmZvLWRpYWxvZ3sKICAgICAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYm9yZGVyLXJhZGl1czoxNHB4OwogICAgICBwYWRkaW5nOjA7CiAgICAgIGJhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6bWluKDU2MHB4LCBjYWxjKDEwMHZ3IC0gMjRweCkpOwogICAgICB3aWR0aDogNTYwcHg7CiAgICB9CiAgICBkaWFsb2cuaW5mby1kaWFsb2c6OmJhY2tkcm9we2JhY2tncm91bmQ6cmdiYSgwLDAsMCwuNTUpfQogICAgLmluZm8tY2FyZHttYXJnaW46MDsgcGFkZGluZzouOXJlbSAxcmVtOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi43NXJlbX0KICAgIC5pbmZvLWhlYWR7ZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGp1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVuOyBnYXA6Ljc1cmVtOyBib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZy1ib3R0b206LjZyZW19CiAgICAuaW5mby10aXRsZXtmb250LXdlaWdodDo2MDB9CiAgICAuaW5mby1ib2R5IGF7Y29sb3I6dmFyKC0tYWNjKTsgdGV4dC1kZWNvcmF0aW9uOm5vbmV9CiAgICAuaW5mby1ib2R5IGE6aG92ZXJ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZX0KICAgIC5pbmZvLWFjdGlvbnN7ZGlzcGxheTpmbGV4OyBqdXN0aWZ5LWNvbnRlbnQ6ZmxleC1lbmQ7IGdhcDouNXJlbX0KICAgIGhlYWRlciAuc3BhY2Vye2ZsZXg6MX0KICAgIGhlYWRlciBpbnB1dHtiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjQ1cmVtIC41cmVtOyBib3JkZXItcmFkaXVzOi41cmVtOyBvdXRsaW5lOm5vbmU7IHdpZHRoOjIyY2g7IGhlaWdodDozMnB4fQogICAgaGVhZGVyIGlucHV0OmZvY3Vze2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIHNlbGVjdHsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OwogICAgICBjb2xvcjp2YXIoLS1pbmspOwogICAgICBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgICBwYWRkaW5nOi40NXJlbSAuNXJlbTsKICAgICAgYm9yZGVyLXJhZGl1czouNXJlbTsKICAgICAgb3V0bGluZTpub25lOwogICAgICB3aWR0aDoyMmNoOwogICAgICBoZWlnaHQ6MzJweDsKICAgICAgYXBwZWFyYW5jZTpub25lOwogICAgfQogICAgaGVhZGVyIHNlbGVjdDpmb2N1c3tib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIGhlYWRlciAuc21hbGx7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKX0KICAgIGhlYWRlciBsYWJlbC5zbWFsbHtkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOi4yNXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIGhlYWRlciAucm93e2Rpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBhbGlnbi1pdGVtczpmbGV4LWVuZDsgZmxleC13cmFwOndyYXB9CgogICAgLyogU2Nyb2xsLVZpZXdwb3J0IChzbW9vdGggbnVyIGbDvHIgVXNlci1TY3JvbGwsIHByb2dyYW1tYXRpc2NoIGVyendpbmdlbiB3aXIgJ2F1dG8nKSAqLwogICAgbWFpbiNzY3JvbGxlcntvdmVyZmxvdzphdXRvOyBwYWRkaW5nOjFyZW0gMXJlbSAxLjVyZW07IHNjcm9sbC1iZWhhdmlvcjpzbW9vdGh9CiAgICAubG9ne21heC13aWR0aDo5MDBweDsgbWFyZ2luOjFyZW0gYXV0byAwOyBkaXNwbGF5OmZsZXg7IGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsgZ2FwOjFyZW19CiAgICAubXNne2Rpc3BsYXk6ZmxleDsgZ2FwOi43NXJlbTsgYWxpZ24taXRlbXM6ZmxleC1zdGFydH0KICAgIC5hdmF0YXJ7d2lkdGg6MjhweDsgaGVpZ2h0OjI4cHg7IGRpc3BsYXk6Z3JpZDsgcGxhY2UtaXRlbXM6Y2VudGVyOyBib3JkZXItcmFkaXVzOjhweDsgZmxleDowIDAgMjhweH0KICAgIC51c2VyIC5hdmF0YXJ7YmFja2dyb3VuZDojMjAzMTRhfQogICAgLmJvdCAuYXZhdGFye2JhY2tncm91bmQ6IzFiMmQyMH0KICAgIC5zeXMgLmF2YXRhcntiYWNrZ3JvdW5kOiMyYTIyM2J9CiAgICAKICAgIC8qIFdyYXBwZXIgZsO8ciBkZW4gTmFjaHJpY2h0ZW5pbmhhbHQgKi8KICAgIC5tZXNzYWdlLWNvbnRlbnQtd3JhcHBlciB7CiAgICAgICAgZGlzcGxheTpmbGV4OwogICAgICAgIGZsZXgtZGlyZWN0aW9uOmNvbHVtbjsKICAgICAgICBnYXA6IDAuNXJlbTsgLyogQWJzdGFuZCB6d2lzY2hlbiBEZW5rcHJvemVzcyB1bmQgQW50d29ydCAqLwogICAgICAgIHdpZHRoOiAxMDAlOwogICAgfQoKICAgIC5idWJibGV7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6Ljc1cmVtIC45cmVtOyBib3JkZXItcmFkaXVzOjEycHg7IG1heC13aWR0aDoxMDAlOyBvdmVyZmxvdy13cmFwOmFueXdoZXJlfQogICAgLm1ldGF7Zm9udC1zaXplOjEycHg7IGNvbG9yOnZhcigtLW11dGVkKTsgbWFyZ2luLWJvdHRvbTouMjVyZW19CgogICAgZm9vdGVye2JvcmRlci10b3A6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi42cmVtIC44cmVtfQogICAgLmJhcnttYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW19CiAgICB0ZXh0YXJlYXt3aWR0aDoxMDAlOyByZXNpemU6dmVydGljYWw7IG1pbi1oZWlnaHQ6NjBweDsgbWF4LWhlaWdodDo0MHZoOyBiYWNrZ3JvdW5kOiMwYzBmMTU7IGNvbG9yOnZhcigtLWluayk7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHBhZGRpbmc6LjdyZW0gLjc1cmVtOyBib3JkZXItcmFkaXVzOi43NXJlbTsgb3V0bGluZTpub25lfQogICAgdGV4dGFyZWE6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CiAgICBidXR0b257YmFja2dyb3VuZDojMTAxODI3OyBjb2xvcjp2YXIoLS1pbmspOyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBwYWRkaW5nOi41NXJlbSAuODVyZW07IGJvcmRlci1yYWRpdXM6LjZyZW07IGN1cnNvcjpwb2ludGVyfQogICAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQogICAgaGVhZGVyIGJ1dHRvbntoZWlnaHQ6MzJweDsgZGlzcGxheTppbmxpbmUtZmxleDsgYWxpZ24taXRlbXM6Y2VudGVyfQogICAgLnNlY29uZGFyeXtiYWNrZ3JvdW5kOiMwYzBmMTV9CiAgICAubXV0ZWR7Y29sb3I6dmFyKC0tbXV0ZWQpfQogICAgLmhpZGRlbntkaXNwbGF5Om5vbmV9CiAgICAuaGludHtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXJnaW4tbGVmdDouMjVyZW19CiAgICAuZHJvcHtkaXNwbGF5Om5vbmU7IGJvcmRlcjoycHggZGFzaGVkICMyYTMzNDY7IGJvcmRlci1yYWRpdXM6MTJweDsgcGFkZGluZzoxMnB4OyB0ZXh0LWFsaWduOmNlbnRlcjsgY29sb3I6dmFyKC0tbXV0ZWQpOyBtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjoxcmVtIGF1dG8gLjc1cmVtO30KICAgIC5kcm9wLmRyYWd7YmFja2dyb3VuZDojMGUxMzIwOyBjb2xvcjp2YXIoLS1pbmspfQogICAgY29kZSwgcHJlLCBrYmR7Zm9udC1mYW1pbHk6IHVpLW1vbm9zcGFjZSwgU0ZNb25vLVJlZ3VsYXIsIE1lbmxvLCBDb25zb2xhcywgbW9ub3NwYWNlfQoKICAgIC8qIE1hcmtkb3duIHJlbmRlcmluZyAqLwogICAgLm1ke2xpbmUtaGVpZ2h0OjEuNn0KICAgIC5tZCBoMSwubWQgaDIsLm1kIGgze2xpbmUtaGVpZ2h0OjEuMjU7IG1hcmdpbjouMnJlbSAwIC42cmVtfQogICAgLm1kIGgxe2ZvbnQtc2l6ZToxLjM1cmVtfQogICAgLm1kIGgye2ZvbnQtc2l6ZToxLjJyZW19CiAgICAubWQgaDN7Zm9udC1zaXplOjEuMDVyZW19CiAgICAubWQgcHttYXJnaW46LjVyZW0gMH0KICAgIC5tZCB1bCwubWQgb2x7bWFyZ2luOi41cmVtIDAgLjVyZW0gMS4xcmVtfQogICAgLm1kIGxpe21hcmdpbjouMnJlbSAwfQogICAgLm1kIGJsb2NrcXVvdGV7bWFyZ2luOi42cmVtIDA7IHBhZGRpbmc6LjI1cmVtIC44cmVtOyBib3JkZXItbGVmdDozcHggc29saWQgIzMwNDQ1ZjsgYmFja2dyb3VuZDojMGMxMTFiOyBjb2xvcjojZDZlMmYwOyBib3JkZXItcmFkaXVzOjZweH0KICAgIC5tZCBocntib3JkZXI6bm9uZTsgYm9yZGVyLXRvcDoxcHggc29saWQgdmFyKC0tbGluZSk7IG1hcmdpbjouOHJlbSAwfQogICAgLm1kIGNvZGV7YmFja2dyb3VuZDojMGMwZjE1OyBwYWRkaW5nOi4xMnJlbSAuMzJyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6LjM1cmVtfQogICAgLm1kIHByZXtiYWNrZ3JvdW5kOiMwYzBmMTU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgcGFkZGluZzouOHJlbTsgb3ZlcmZsb3c6YXV0bzsgcG9zaXRpb246cmVsYXRpdmV9CiAgICAubWQgcHJlPmNvZGV7ZGlzcGxheTpibG9jazsgd2hpdGUtc3BhY2U6cHJlfQogICAgLm1kIHRhYmxle2JvcmRlci1jb2xsYXBzZTpjb2xsYXBzZTsgbWFyZ2luOi44cmVtIDA7IHdpZHRoOjEwMCU7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAubWQgdGgsLm1kIHRke3BhZGRpbmc6LjVyZW0gLjdyZW07IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IHRleHQtYWxpZ246bGVmdH0KICAgIC5tZCB0aHtiYWNrZ3JvdW5kOiMwZjE3MjU7IGZvbnQtd2VpZ2h0OjYwMH0KICAgIC5jb2Rld3JhcHtwb3NpdGlvbjpyZWxhdGl2ZX0KICAgIC5jb3B5YnRue3Bvc2l0aW9uOmFic29sdXRlOyB0b3A6LjQ1cmVtOyByaWdodDouNDVyZW07IGZvbnQtc2l6ZToxMnB4OyBwYWRkaW5nOi4yNXJlbSAuNDVyZW07IGJvcmRlci1yYWRpdXM6LjRyZW07IGJhY2tncm91bmQ6IzBmMTcyNTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgY3Vyc29yOnBvaW50ZXJ9CiAgICAuY29weWJ0bjpob3Zlcntib3JkZXItY29sb3I6dmFyKC0tYWNjKX0KICAgIC50YXNrLWRvbmV7b3BhY2l0eTouOX0KCiAgICAvKiDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4ggc3R5bGluZyAqLwogICAgLnN5c2lucHV0e3dpZHRoOjEwMCU7IHJlc2l6ZTp2ZXJ0aWNhbDsgbWluLWhlaWdodDo2MHB4OyBtYXgtaGVpZ2h0OjMwdmg7IGJhY2tncm91bmQ6IzBjMGYxNTsgY29sb3I6dmFyKC0taW5rKTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsgcGFkZGluZzouN3JlbSAuNzVyZW07IGJvcmRlci1yYWRpdXM6Ljc1cmVtOyBvdXRsaW5lOm5vbmV9CiAgICAuc3lzaW5wdXQ6Zm9jdXN7Ym9yZGVyLWNvbG9yOnZhcigtLWFjYyl9CgogICAgLyogSlNPTiBFZGl0b3IgKi8KICAgIGRldGFpbHMucmF3e2Rpc3BsYXk6bm9uZTsgbWF4LXdpZHRoOjkwMHB4OyBtYXJnaW46MXJlbSBhdXRvIDA7IGJvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7IGJvcmRlci1yYWRpdXM6MTBweDsgYmFja2dyb3VuZDojMGMwZjE1fQogICAgZGV0YWlscy5yYXc+c3VtbWFyeXtjdXJzb3I6cG9pbnRlcjsgcGFkZGluZzouNnJlbSAuOHJlbTsgbGlzdC1zdHlsZTpub25lOyB1c2VyLXNlbGVjdDpub25lfQogICAgZGV0YWlscy5yYXc+c3VtbWFyeTo6bWFya2Vye2Rpc3BsYXk6bm9uZX0KICAgIGRldGFpbHMucmF3W29wZW5dPnN1bW1hcnl7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgdmFyKC0tbGluZSl9CiAgICAucmF3IC5pbm5lcntwYWRkaW5nOi44cmVtfQogICAgLnJhdyB0ZXh0YXJlYXtmb250LWZhbWlseTogdWktbW9ub3NwYWNlLCBTRk1vbm8tUmVndWxhciwgTWVubG8sIENvbnNvbGFzLCBtb25vc3BhY2U7IG1pbi1oZWlnaHQ6MjIwcHg7IHdoaXRlLXNwYWNlOnByZTsgdGFiLXNpemU6Mn0KICAgIC5yYXcgLnJvd3tkaXNwbGF5OmZsZXg7IGdhcDouNXJlbTsgYWxpZ24taXRlbXM6Y2VudGVyOyBtYXJnaW4tdG9wOi41cmVtOyBmbGV4LXdyYXA6d3JhcH0KICAgIC5iYWRnZXtmb250LXNpemU6MTJweDsgY29sb3I6dmFyKC0tbXV0ZWQpfQoKICAgIC8qIFNjcm9sbC10by1ib3R0b20gYnV0dG9uICovCiAgICAudG8tYm90dG9te3Bvc2l0aW9uOmZpeGVkOyByaWdodDoxNnB4OyBib3R0b206MTZweDsgcGFkZGluZzouNXJlbSAuN3JlbTsgYm9yZGVyLXJhZGl1czouNnJlbTsgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOyBjb2xvcjp2YXIoLS1pbmspOyBkaXNwbGF5Om5vbmU7IG9wYWNpdHk6LjkyfQogICAgLnRvLWJvdHRvbS5zaG93e2Rpc3BsYXk6YmxvY2t9CiAgICAKICAgIC8qIFNUWUxFUyBGw5xSIFRISU5LSU5HLUJPWCAqLwogICAgLnRoaW5rLWJ1YmJsZSB7CiAgICAgICAgd2lkdGg6IDEwMCU7CiAgICB9CiAgICAudGhpbmstYnViYmxlIGRldGFpbHMgewogICAgICAgIGJhY2tncm91bmQ6ICMxYTE2MjU7CiAgICAgICAgYm9yZGVyOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBib3JkZXItcmFkaXVzOiAxMHB4OwogICAgICAgIGZvbnQtc2l6ZTogMTNweDsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeSB7CiAgICAgICAgY3Vyc29yOiBwb2ludGVyOwogICAgICAgIHBhZGRpbmc6IC41cmVtIC44cmVtOwogICAgICAgIGZvbnQtd2VpZ2h0OiA1MDA7CiAgICAgICAgY29sb3I6IHZhcigtLW11dGVkKTsKICAgICAgICB1c2VyLXNlbGVjdDogbm9uZTsKICAgIH0KICAgIC50aGluay1idWJibGUgc3VtbWFyeTpob3ZlciB7CiAgICAgICAgY29sb3I6IHZhcigtLWluayk7CiAgICB9CiAgICAudGhpbmstYnViYmxlIC50aGluay1jb250ZW50IHsKICAgICAgICBwYWRkaW5nOiAuMXJlbSAuOHJlbSAuN3JlbTsKICAgICAgICBib3JkZXItdG9wOiAxcHggc29saWQgIzNjMzI0ZjsKICAgICAgICBjb2xvcjogI2MwYjhkMTsKICAgIH0KCS8qIOKAlCBQb2xpc2hpbmc6IFNhbXBsaW5nLUlucHV0cyBpbSBFaW5zdGVsbHVuZ3MtUGFuZWwg4oCUICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCB7ICAgICAgICAgICAgICAgLyogTGFiZWwgKyBGZWxkIHZlcnRpa2FsLCBrb21wYWt0ICovCiAgZGlzcGxheTogZmxleDsKICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwogIGdhcDogLjI1cmVtOwogIGFsaWduLWl0ZW1zOiBmbGV4LXN0YXJ0Owp9CgovKiBFaW5oZWl0bGljaGUsIHNjaG1hbGUgTnVtYmVyLUlucHV0cyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIHdpZHRoOiAxMGNoOyAgICAgICAgICAgICAgICAgICAgICAgIC8qIGZpeGUgQnJlaXRlIC0+IGdsZWljaGUgS2FudGUgKi8KICBoZWlnaHQ6IDMycHg7CiAgcGFkZGluZzogLjM1cmVtIC41cmVtOwogIGJhY2tncm91bmQ6ICMwYzBmMTU7CiAgY29sb3I6IHZhcigtLWluayk7CiAgYm9yZGVyOiAxcHggc29saWQgdmFyKC0tbGluZSk7CiAgYm9yZGVyLXJhZGl1czogLjVyZW07CiAgdGV4dC1hbGlnbjogcmlnaHQ7CiAgZm9udC12YXJpYW50LW51bWVyaWM6IHRhYnVsYXItbnVtczsgLyogc2F1YmVyZSwgYXVzZ2VyaWNodGV0ZSBaaWZmZXJuICovCiAgb3V0bGluZTogbm9uZTsKfQoKLyogRm9rdXMtRmVlZGJhY2sgcGFzc2VuZCB6dSBkZW4gcmVzdGxpY2hlbiBDb250cm9scyAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Zm9jdXMgewogIGJvcmRlci1jb2xvcjogdmFyKC0tYWNjKTsKICBib3gtc2hhZG93OiAwIDAgMCAycHggcmdiYSgxMDYsMTYxLDI1NSwuMTUpOwp9CgovKiBEZXplbnRlIFBsYWNlaG9sZGVyLUZhcmJlICovCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6cGxhY2Vob2xkZXIgewogIGNvbG9yOiB2YXIoLS1tdXRlZCk7CiAgb3BhY2l0eTogLjk7Cn0KCi8qIFNwaW5uZXIgYmVpIE51bWJlci1JbnB1dHMgYXVzYmxlbmRlbiAobnVyIGltIFNldHRpbmdzLVBhbmVsKSAqLwojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl06Oi13ZWJraXQtb3V0ZXItc3Bpbi1idXR0b24sCiNzZXR0aW5nc1BhbmVsIC5zbWFsbCBpbnB1dFt0eXBlPSJudW1iZXIiXTo6LXdlYmtpdC1pbm5lci1zcGluLWJ1dHRvbiB7CiAgLXdlYmtpdC1hcHBlYXJhbmNlOiBub25lOwogIG1hcmdpbjogMDsKfQojc2V0dGluZ3NQYW5lbCAuc21hbGwgaW5wdXRbdHlwZT0ibnVtYmVyIl0gewogIC1tb3otYXBwZWFyYW5jZTogdGV4dGZpZWxkOwp9CgovKiBLbGVpbmVzIFR1bmluZyBmw7xyIGRpZSBHcnVwcGU6IGRpY2h0ZXIgenVzYW1tZW4sIGJyaWNodCBzYXViZXIgdW0gKi8KI3NldHRpbmdzUGFuZWwgPiAuYmFyID4gZGl2Omxhc3QtY2hpbGQsCiNzZXR0aW5nc1BhbmVsIC5iYXIgPiBkaXY6bGFzdC1jaGlsZCB7CiAgZ2FwOiAuNXJlbSAuNzVyZW07ICAgICAgICAgICAgICAgICAgLyogWmVpbGVuL1NwYWx0ZW4tR2FwICovCiAgZmxleC13cmFwOiB3cmFwOwp9CgogIAogICAgLyogLS0tLS0tLS0tLSBSZXNwb25zaXZlIC0tLS0tLS0tLS0gKi8KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA5ODBweCl7CiAgICAgIGhlYWRlcntnYXA6LjZyZW19CiAgICAgIGhlYWRlciAuc3BhY2Vye2Rpc3BsYXk6bm9uZX0KICAgICAgaGVhZGVyIGxhYmVsLnNtYWxse21pbi13aWR0aDoyMDBweH0KICAgICAgaGVhZGVyIC5yb3d7d2lkdGg6MTAwJX0KICAgICAgaGVhZGVyICN0b2tlblJhdGV7d2lkdGg6YXV0bzsgdGV4dC1hbGlnbjpsZWZ0ICFpbXBvcnRhbnR9CiAgICB9CiAgICBAbWVkaWEgKG1heC13aWR0aDogNzIwcHgpewogICAgICBoZWFkZXIgbGFiZWwuc21hbGx7ZmxleDoxIDEgMjIwcHh9CiAgICAgIGhlYWRlciBpbnB1dCwgaGVhZGVyIHNlbGVjdHt3aWR0aDoxMDAlfQogICAgICBtYWluI3Njcm9sbGVye3BhZGRpbmc6Ljg1cmVtIC43NXJlbSAxLjI1cmVtfQogICAgICAuYmFye2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnJ9CiAgICAgIGZvb3RlciAucm93e2ZsZXgtd3JhcDp3cmFwfQogICAgICB0ZXh0YXJlYXttaW4taGVpZ2h0OjkwcHh9CiAgICAgIC8qIFNldHRpbmdzIHBhbmVsIHN0YWNrICovCiAgICAgICNzZXR0aW5nc1BhbmVse2dyaWQtdGVtcGxhdGUtY29sdW1uczoxZnIgIWltcG9ydGFudH0KICAgICAgI3NldHRpbmdzUGFuZWwgPiBkaXZ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAhaW1wb3J0YW50fQogICAgICAjc2V0dGluZ3NQYW5lbCA+IGRpdiA+IGRpdjpsYXN0LWNoaWxke21pbi13aWR0aDowICFpbXBvcnRhbnQ7IGFsaWduLXNlbGY6c3RhcnQgIWltcG9ydGFudH0KICAgIH0KICAgIEBtZWRpYSAobWF4LXdpZHRoOiA0ODBweCl7CiAgICAgIGhlYWRlciBsYWJlbC5zbWFsbHtmbGV4OjEgMSAxMDAlfQogICAgICBoZWFkZXIgLnJvd3tnYXA6LjRyZW19CiAgICAgIGhlYWRlciBidXR0b24sIGZvb3RlciBidXR0b257cGFkZGluZzouNXJlbSAuNzVyZW19CiAgICB9CgogIAogICAgLyogQXR0YWNobWVudHMgKi8KICAgIC5jaGlwewogICAgICBkaXNwbGF5OmlubGluZS1mbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNDVyZW07CiAgICAgIHBhZGRpbmc6LjM1cmVtIC41NXJlbTsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTsKICAgICAgYmFja2dyb3VuZDojMGMwZjE1OyBib3JkZXItcmFkaXVzOjk5OXB4OyBjb2xvcjp2YXIoLS1pbmspOwogICAgICBtYXgtd2lkdGg6MTAwJTsKICAgIH0KICAgIC5jaGlwIHNtYWxse2NvbG9yOnZhcigtLW11dGVkKX0KICAgIC5jaGlwIGJ1dHRvbnsKICAgICAgcGFkZGluZzouMTVyZW0gLjM1cmVtOyBib3JkZXItcmFkaXVzOjhweDsKICAgICAgYmFja2dyb3VuZDojMTAxODI3OyBib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpOwogICAgfQogICAgLmNoaXAgYnV0dG9uOmhvdmVye2JvcmRlci1jb2xvcjp2YXIoLS1hY2MpfQoKICAgIC8qIE1hcmtkb3duIGltYWdlcyAqLwogICAgLm1kIGltZ3ttYXgtd2lkdGg6MTAwJTsgaGVpZ2h0OmF1dG87IGJvcmRlci1yYWRpdXM6MTBweDsgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KCiAgPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KPGRpdiBjbGFzcz0id3JhcCI+CiAgPGhlYWRlcj4KICAgIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgIDxkaXYgY2xhc3M9InRpdGxlIj7wn5KsIOODgeODo+ODg+ODiFVJPC9kaXY+CiAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJzcGFjZXIiPjwvZGl2PgogICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+PHNwYW4+44Oi44OH44OrPC9zcGFuPgogICAgICA8c2VsZWN0IGlkPSJtb2RlbCI+CiAgICAgICAgPG9wdGlvbiB2YWx1ZT0iIiBzZWxlY3RlZCBkaXNhYmxlZD7igJQg44Oi44OH44Or44KS6Kqt44G/6L6844G/IOKAlDwvb3B0aW9uPgogICAgICA8L3NlbGVjdD4KICAgIDwvbGFiZWw+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICA8YnV0dG9uIGlkPSJhdXRoQnRuIiBjbGFzcz0ic2Vjb25kYXJ5IiB0aXRsZT0iQVBJIOiqjeiovO+8iEJhc2lj77yJIj7wn5SRPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImluZm9CdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSLmg4XloLEiPuaDheWgsTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJyb3ciPgogICAgICAgIDxkaXYgaWQ9InF1b3RhTGVmdCIgY2xhc3M9InNtYWxsIiBzdHlsZT0id2lkdGg6MTFjaDsgdGV4dC1hbGlnbjpyaWdodDsiIHRpdGxlPSJLYWdnbGUgR1BVIOaui+OCiuaZgumWkyI+PC9kaXY+CiAgICAgICAgPGRpdiBpZD0idG9rZW5SYXRlIiBjbGFzcz0ic21hbGwiIHN0eWxlPSJ3aWR0aDoxMmNoOyB0ZXh0LWFsaWduOnJpZ2h0OyI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InJvdyIgaWQ9InN0b3BSb3ciIHN0eWxlPSJkaXNwbGF5Om5vbmU7Ij4KICAgICAgPGJ1dHRvbiBpZD0ic3RvcFNlcnZlciIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IuOCteODvOODkOODvOOCkuWBnOatoiI+4pagIOWBnOatojwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9oZWFkZXI+CgogIDxtYWluIGlkPSJzY3JvbGxlciI+CiAgICA8ZGl2IGlkPSJkcm9wIiBjbGFzcz0iZHJvcCI+44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GZ44KL44Gr44Gv44OV44Kh44Kk44Or44KS44OJ44Ot44OD44OXIChKU09OIGluIHRoZSB1c3VhbCA8Y29kZT5be3JvbGUsIGNvbnRlbnR9XTwvY29kZT4gZm9ybWF0KS48YnI+PHNwYW4gY2xhc3M9ImhpbnQiPuOBvuOBn+OBryA8Yj7jgqTjg7Pjg53jg7zjg4g8L2I+IGFib3ZlLjwvc3Bhbj48L2Rpdj4KCiAgICA8ZGV0YWlscyBpZD0icmF3Qm94IiBjbGFzcz0icmF3Ij4KICAgICAgPHN1bW1hcnk+4pyOIEpTT07jgajjgZfjgabjg4Hjg6Pjg4Pjg4jjgpLnt6jpm4YgPHNwYW4gY2xhc3M9ImJhZGdlIj4o5YiH5pu/KTwvc3Bhbj48L3N1bW1hcnk+CiAgICAgIDxkaXYgY2xhc3M9ImlubmVyIj4KICAgICAgICA8dGV4dGFyZWEgaWQ9Impzb25FZGl0b3IiIHNwZWxsY2hlY2s9ImZhbHNlIiBwbGFjZWhvbGRlcj0nWyB7ICJyb2xlIjogInN5c3RlbSIsICJjb250ZW50IjogIi4uLiIgfSwgLi4uIF0nPjwvdGV4dGFyZWE+CiAgICAgICAgPGRpdiBjbGFzcz0icm93Ij4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25BcHBseSI+6YGp55SoICjnva7mj5spPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uQXBwZW5kIiBjbGFzcz0ic2Vjb25kYXJ5Ij7ov73oqJg8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25SZWZyZXNoIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgYvjgonmm7TmlrA8L2J1dHRvbj4KICAgICAgICAgIDxidXR0b24gaWQ9Impzb25QcmV0dHkiIGNsYXNzPSJzZWNvbmRhcnkiPuaVtOW9ojwvYnV0dG9uPgogICAgICAgICAgPGJ1dHRvbiBpZD0ianNvbk1pbmlmeSIgY2xhc3M9InNlY29uZGFyeSI+5Zyn57iuPC9idXR0b24+CiAgICAgICAgICA8YnV0dG9uIGlkPSJqc29uVmFsaWRhdGUiIGNsYXNzPSJzZWNvbmRhcnkiPuaknOiovDwvYnV0dG9uPgogICAgICAgICAgPHNwYW4gaWQ9Impzb25TdGF0dXMiIGNsYXNzPSJiYWRnZSI+PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGV0YWlscz4KCiAgICA8ZGl2IGlkPSJsb2ciIGNsYXNzPSJsb2ciPjwvZGl2PgogIDwvbWFpbj4KCiAgPGZvb3Rlcj4KICAgIDxkaXYgY2xhc3M9ImJhciI+CiAgICAgIDx0ZXh0YXJlYSBpZD0iaW5wdXQiIHBsYWNlaG9sZGVyPSLjg6Hjg4Pjgrvjg7zjgrjjgpLlhaXlipvigKYgKOmAgeS/oTogRW50ZXIsIOaUueihjDogU2hpZnQrRW50ZXIpIj48L3RleHRhcmVhPgogICAgICA8ZGl2IGNsYXNzPSJyb3ciIHN0eWxlPSJkaXNwbGF5OmZsZXg7IGdhcDouNnJlbTsgYWxpZ24taXRlbXM6Y2VudGVyIj4KICAgICAgICA8YnV0dG9uIGlkPSJhdHRhY2hCdG4iIGNsYXNzPSJzZWNvbmRhcnkiIHRpdGxlPSJBdHRhY2ggZmlsZXMiPvCfk448L2J1dHRvbj4KICAgICAgICA8aW5wdXQgaWQ9ImF0dGFjaElucHV0IiB0eXBlPSJmaWxlIiBjbGFzcz0iaGlkZGVuIiBtdWx0aXBsZSAvPgogICAgICAgIDxidXR0b24gaWQ9InNlbmQiPumAgeS/oTwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9InN0b3AiIGNsYXNzPSJzZWNvbmRhcnkiIGRpc2FibGVkPuWBnOatojwvYnV0dG9uPgogICAgICAgIDxidXR0b24gaWQ9ImNsZWFyIiBjbGFzcz0ic2Vjb25kYXJ5Ij7jg4Hjg6Pjg4Pjg4jjgq/jg6rjgqI8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGlkPSJzZXR0aW5nc1RvZ2dsZSIgY2xhc3M9InNlY29uZGFyeSIgdGl0bGU9IlRvZ2dsZSBzZXR0aW5ncyI+4pqZ77iPPC9idXR0b24+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0JhciIgY2xhc3M9ImJhciBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOi42cmVtOyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6MWZyOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IGlkPSJhdHRhY2htZW50c0xpc3QiIHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6ZmxleDsgZ2FwOi41cmVtOyBmbGV4LXdyYXA6d3JhcCI+PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgaWQ9InNldHRpbmdzUGFuZWwiIGNsYXNzPSJiYXIgaGlkZGVuIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW0iPgogICAgICA8ZGl2IHN0eWxlPSJtYXgtd2lkdGg6OTAwcHg7IG1hcmdpbjowIGF1dG87IGRpc3BsYXk6Z3JpZDsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciBhdXRvOyBnYXA6LjZyZW07IGdyaWQtY29sdW1uOjEvLTE7IHdpZHRoOjEwMCUiPgogICAgICAgIDxkaXY+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuOCt+OCueODhuODoOODl+ODreODs+ODl+ODiDwvbGFiZWw+CiAgICAgICAgICA8dGV4dGFyZWEgaWQ9InN5c3RlbSIgY2xhc3M9InN5c2lucHV0IiByb3dzPSIyIiBwbGFjZWhvbGRlcj0i5Lu75oSPIOKAlCBlLmcuIFlvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC4iPjwvdGV4dGFyZWE+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBzdHlsZT0ibWluLXdpZHRoOjI2MHB4OyBhbGlnbi1zZWxmOmVuZDsgZGlzcGxheTpmbGV4OyBhbGlnbi1pdGVtczpjZW50ZXI7IGdhcDouNnJlbTsgZmxleC13cmFwOndyYXAiPgogICAgICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCI+VGVtcGVyYXR1cmUgPHNwYW4gaWQ9InRlbXBWYWwiIGNsYXNzPSJoaW50Ij4wLjc8L3NwYW4+PC9sYWJlbD4KICAgICAgICAgIDxpbnB1dCBpZD0idGVtcCIgdHlwZT0icmFuZ2UiIG1pbj0iMCIgbWF4PSIyIiBzdGVwPSIwLjEiIHZhbHVlPSIwLjciIHN0eWxlPSJ3aWR0aDoxNjBweCI+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3BfcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj5taW5fcDxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJtaW5fcCIgdHlwZT0ibnVtYmVyIiBpbnB1dG1vZGU9ImRlY2ltYWwiIHN0ZXA9IjAuMDEiIG1pbj0iMCIgbWF4PSIxIiBwbGFjZWhvbGRlcj0iKGRlZmF1bHQpIiBzdHlsZT0id2lkdGg6OWNoOyB0ZXh0LWFsaWduOnJpZ2h0Ij4KICAgICAgICAgIDwvbGFiZWw+CiAgICAgICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIj50b3Bfazxicj4KICAgICAgICAgICAgPGlucHV0IGlkPSJ0b3BfayIgdHlwZT0ibnVtYmVyIiBzdGVwPSIxIiBtaW49IjAiIHBsYWNlaG9sZGVyPSIoZGVmYXVsdCkiIHN0eWxlPSJ3aWR0aDo5Y2g7IHRleHQtYWxpZ246cmlnaHQiPgogICAgICAgICAgPC9sYWJlbD4KICAgICAgICAgIDxsYWJlbCBjbGFzcz0ic21hbGwiPnRoaW5raW5nPGJyPgogICAgICAgICAgICA8c2VsZWN0IGlkPSJ0aGlua2luZyIgc3R5bGU9IndpZHRoOjExY2giPgogICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+KGRlZmF1bHQpPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib24iPm9uPC9vcHRpb24+CiAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0ib2ZmIj5vZmY8L29wdGlvbj4KICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICA8L2xhYmVsPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZm9vdGVyPgo8L2Rpdj4KCjxidXR0b24gaWQ9InRvQm90dG9tIiBjbGFzcz0idG8tYm90dG9tIiB0aXRsZT0i5LiA55Wq5LiL44G4Ij7ilrwg5LiA55Wq5LiLPC9idXR0b24+Cgo8ZGlhbG9nIGlkPSJpbmZvRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCfkqwg44OB44Oj44OD44OIVUk8L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0ic2Vjb25kYXJ5IiB2YWx1ZT0iY2xvc2UiIGFyaWEtbGFiZWw9IkNsb3NlIj7inJU8L2J1dHRvbj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1ib2R5Ij4KICAgICAgPGRpdj48c3BhbiBjbGFzcz0ibXV0ZWQiPuODkOODvOOCuOODp+ODszo8L3NwYW4+IDxjb2RlIGlkPSJpbmZvVmVyc2lvbiI+MC4wLjA8L2NvZGU+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9Im11dGVkIiBzdHlsZT0ibWFyZ2luLXRvcDouNnJlbTsgZm9udC1zaXplOjEycHgiPuOCueODhuODvOODiOODrOOCuVVJIOKAlCBubyBzZXJ2ZXItc2lkZSBjaGF0IHN0b3JhZ2UuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImluZm8tYWN0aW9ucyI+CiAgICAgIDxidXR0b24gdmFsdWU9ImNsb3NlIj7plonjgZjjgos8L2J1dHRvbj4KICAgIDwvZGl2PgogIDwvZm9ybT4KPC9kaWFsb2c+Cgo8ZGlhbG9nIGlkPSJhdXRoRGlhbG9nIiBjbGFzcz0iaW5mby1kaWFsb2ciPgogIDxmb3JtIG1ldGhvZD0iZGlhbG9nIiBjbGFzcz0iaW5mby1jYXJkIj4KICAgIDxkaXYgY2xhc3M9ImluZm8taGVhZCI+CiAgICAgIDxkaXYgY2xhc3M9ImluZm8tdGl0bGUiPvCflJEgQVBJIOiqjeiovDwvZGl2PgogICAgICA8YnV0dG9uIGNsYXNzPSJzZWNvbmRhcnkiIHZhbHVlPSJjbG9zZSIgYXJpYS1sYWJlbD0iQ2xvc2UiPuKclTwvYnV0dG9uPgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJpbmZvLWJvZHkiPgogICAgICA8ZGl2IGNsYXNzPSJtdXRlZCBzbWFsbCIgc3R5bGU9Im1hcmdpbi1ib3R0b206LjZyZW0iPuOCteODvOODkOODvOOBjCBCYXNpYyDoqo3oqLzjgpLopoHmsYLjgZnjgovloLTlkIjjgavlhaXlipvjgZfjgb7jgZnvvIhzZXNzaW9uU3RvcmFnZSDjgavkv53lrZjjg7vjgr/jg5bjgpLplonjgZjjgovjgajmtojjgYjjgb7jgZnvvInjgII8L2Rpdj4KICAgICAgPGxhYmVsIGNsYXNzPSJzbWFsbCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbi1ib3R0b206LjM1cmVtIj7jg6bjg7zjgrbjg7w8L2xhYmVsPgogICAgICA8aW5wdXQgaWQ9ImF1dGhVc2VyIiB0eXBlPSJ0ZXh0IiBhdXRvY29tcGxldGU9InVzZXJuYW1lIiB2YWx1ZT0iZXhsMyIgc3R5bGU9IndpZHRoOjEwMCU7IG1hcmdpbi1ib3R0b206LjZyZW0iPgogICAgICA8bGFiZWwgY2xhc3M9InNtYWxsIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luLWJvdHRvbTouMzVyZW0iPuODkeOCueODr+ODvOODiTwvbGFiZWw+CiAgICAgIDxpbnB1dCBpZD0iYXV0aFBhc3MiIHR5cGU9InBhc3N3b3JkIiBhdXRvY29tcGxldGU9ImN1cnJlbnQtcGFzc3dvcmQiIHN0eWxlPSJ3aWR0aDoxMDAlOyBtYXJnaW4tYm90dG9tOi42cmVtIj4KICAgICAgPGRpdiBpZD0iYXV0aFN0YXR1cyIgY2xhc3M9InNtYWxsIj48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iaW5mby1hY3Rpb25zIj4KICAgICAgPGJ1dHRvbiBpZD0iYXV0aFNhdmUiIHR5cGU9ImJ1dHRvbiI+5L+d5a2YPC9idXR0b24+CiAgICAgIDxidXR0b24gaWQ9ImF1dGhDbGVhciIgdHlwZT0iYnV0dG9uIiBjbGFzcz0ic2Vjb25kYXJ5Ij7op6PpmaQ8L2J1dHRvbj4KICAgICAgPGJ1dHRvbiB2YWx1ZT0iY2xvc2UiIGNsYXNzPSJzZWNvbmRhcnkiPumWieOBmOOCizwvYnV0dG9uPgogICAgPC9kaXY+CiAgPC9mb3JtPgo8L2RpYWxvZz4KCgo8c2NyaXB0PgooKCkgPT4gewogIC8vIEFQSSDjga7mjqXntprlhYjjgpLoh6rli5XliKTlrprjgZnjgovjgILlkIzjgZggSFRNTCDjgpLjg63jg7zjgqvjg6vjgafjgoIgS2FnZ2xlIOOBp+OCguS9v+OBhuOAggogIC8vCiAgLy8gMSkgP2FwaT08dXJsPiDjgYzmmI7npLrjgZXjgozjgabjgYTjgozjgbDjgZ3jgozvvIjmnIDlhKrlhYjvvIkKICAvLyAyKSA/cG9ydD08bj4g44GM44GC44KM44GwIGh0dHA6Ly8xMjcuMC4wLjE6PG4+L3Yx77yI44Ot44O844Kr44OrIHV6dSAvIGxsYW1hLXNlcnZlcu+8iQogIC8vIDMpIGh0dHAocykg44Gn6YWN5L+h44GV44KM44Gm44GE44KL44Gq44KJ5ZCM5LiA44Kq44Oq44K444Oz77yIS2FnZ2xlIOS4ree2me+8iQogIC8vIDQpIGZpbGU6Ly8g44Gn6ZaL44GE44Gf5aC05ZCI44Gv5pei5a6a44Od44O844OICiAgY29uc3QgcGFyYW1zID0gbmV3IFVSTFNlYXJjaFBhcmFtcyh3aW5kb3cubG9jYXRpb24uc2VhcmNoKTsKICBjb25zdCBwb3J0UGFyYW0gPSBwYXJhbXMuZ2V0KCdwb3J0Jyk7CiAgY29uc3QgYXBpUGFyYW0gPSBwYXJhbXMuZ2V0KCdhcGknKTsKCiAgZnVuY3Rpb24gcmVzb2x2ZUFwaUJhc2UoKSB7CiAgICBpZiAoYXBpUGFyYW0pIHJldHVybiBhcGlQYXJhbS5yZXBsYWNlKC9cLyQvLCAnJyk7CiAgICBpZiAocG9ydFBhcmFtKSByZXR1cm4gYGh0dHA6Ly8xMjcuMC4wLjE6JHtwb3J0UGFyYW19L3YxYDsKICAgIC8vIOS4ree2meOChCBuZ3JvayDntYznlLHjgafphY3kv6HjgZXjgozjgabjgYTjgovloLTlkIjjga/lkIzkuIDjgqrjg6rjgrjjg7PjgIIKICAgIC8vIOOBk+OBk+OCkiAxMjcuMC4wLjEg5Zu65a6a44Gr44GZ44KL44Go44CB44Oq44Oi44O844OI44GL44KJ6ZaL44GE44Gf44Go44GN44GrCiAgICAvLyDmiYvlhYPjga4gUEMg44KS6KaL44Gr6KGM44Gj44Gm44GX44G+44GG44CCCiAgICBpZiAobG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwOicgfHwgbG9jYXRpb24ucHJvdG9jb2wgPT09ICdodHRwczonKSB7CiAgICAgIHJldHVybiBsb2NhdGlvbi5vcmlnaW4gKyAnL3YxJzsKICAgIH0KICAgIHJldHVybiAnaHR0cDovLzEyNy4wLjAuMTo5OTMxL3YxJzsKICB9CgogIGNvbnN0IEFQSV9CQVNFID0gcmVzb2x2ZUFwaUJhc2UoKTsKCiAgLy8gbmdyb2sg54Sh5paZ5p6g44Gv44OW44Op44Km44K244GL44KJ44GuIEFQSSDlj5blvpfjgavjgoLorablkYrjg5rjg7zjgrjjgpLmjJ/jgoDjgZPjgajjgYzjgYLjgovjgIIKICAvLyDjgZPjga7jg5jjg4Pjg4Djga/jg63jg7zjgqvjg6vjgrXjg7zjg5Djg7zjgafjga/nhKHoppbjgZXjgozjgovjga7jgafjgIHluLjjgavku5jjgZHjgabjgoLlrrPjga/jgarjgYTjgIIKICBjb25zdCBCQVNFX0hFQURFUlMgPSB7ICduZ3Jvay1za2lwLWJyb3dzZXItd2FybmluZyc6ICd0cnVlJyB9OwogIGNvbnN0IGVscyA9IHsKICAgIHNjcm9sbGVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2Nyb2xsZXInKSwKICAgIGxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xvZycpLAogICAgZHJvcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2Ryb3AnKSwKICAgIGlucHV0OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnaW5wdXQnKSwKICAgIHNlbmRCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzZW5kJyksCiAgICBzdG9wQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcCcpLAogICAgY2xlYXJCdG46IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjbGVhcicpLAogICAgYXR0YWNoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXR0YWNoQnRuJyksCiAgICBhdHRhY2hJbnB1dDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaElucHV0JyksCiAgICBhdHRhY2htZW50c0JhcjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F0dGFjaG1lbnRzQmFyJyksCiAgICBhdHRhY2htZW50c0xpc3Q6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdHRhY2htZW50c0xpc3QnKSwKICAgIG1vZGVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbW9kZWwnKSwKICAgIHN5c3RlbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N5c3RlbScpLAogICAgdGVtcDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RlbXAnKSwKICAgIHRlbXBWYWw6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0ZW1wVmFsJyksCiAgICBpbXBvcnRCdG46IG51bGwsCiAgICBpbXBvcnRGaWxlOiBudWxsLAogICAgZXhwb3J0QnRuOiBudWxsLAogICAgbG9hZE1vZGVsc0J0bjogbnVsbCwKICAgIG1vZGVsU3RhdHVzOiBudWxsLAogICAgcmF3Qm94OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmF3Qm94JyksCiAgICBqc29uRWRpdG9yOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkVkaXRvcicpLAogICAganNvbkFwcGx5OiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGx5JyksCiAgICBqc29uQXBwZW5kOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnanNvbkFwcGVuZCcpLAogICAganNvblJlZnJlc2g6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uUmVmcmVzaCcpLAogICAganNvblByZXR0eTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb25QcmV0dHknKSwKICAgIGpzb25NaW5pZnk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uTWluaWZ5JyksCiAgICBqc29uVmFsaWRhdGU6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uVmFsaWRhdGUnKSwKICAgIGpzb25TdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdqc29uU3RhdHVzJyksCiAgICB0b0JvdHRvbTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3RvQm90dG9tJyksCiAgICBzZXR0aW5nc1RvZ2dsZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NldHRpbmdzVG9nZ2xlJyksCiAgICBzZXR0aW5nc1BhbmVsOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2V0dGluZ3NQYW5lbCcpLAogICAgdG9rZW5SYXRlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9rZW5SYXRlJyksCiAgICBxdW90YUxlZnQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdxdW90YUxlZnQnKSwKICAgIHN0b3BSb3c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdG9wUm93JyksCiAgICBzdG9wU2VydmVyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc3RvcFNlcnZlcicpLAogICAgaW5mb0J0bjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9CdG4nKSwKICAgIGluZm9EaWFsb2c6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdpbmZvRGlhbG9nJyksCiAgICBpbmZvVmVyc2lvbjogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9WZXJzaW9uJyksCiAgICBpbmZvUHJvamVjdFVybDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9Qcm9qZWN0VXJsJyksCiAgICBpbmZvSW1wcmludDogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2luZm9JbXByaW50JyksCiAgICBhdXRoQnRuOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aEJ0bicpLAogICAgYXV0aERpYWxvZzogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhEaWFsb2cnKSwKICAgIGF1dGhVc2VyOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFVzZXInKSwKICAgIGF1dGhQYXNzOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXV0aFBhc3MnKSwKICAgIGF1dGhTdGF0dXM6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoU3RhdHVzJyksCiAgICBhdXRoU2F2ZTogZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1dGhTYXZlJyksCiAgICBhdXRoQ2xlYXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhdXRoQ2xlYXInKSwKICAgIC8vIE5FVToKICAgIHRvcF9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX3AnKSwKICAgIG1pbl9wOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWluX3AnKSwKICAgIHRvcF9rOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndG9wX2snKSwKICAgIHRoaW5raW5nOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndGhpbmtpbmcnKSwKICB9OwoKICBsZXQgbWVzc2FnZXMgPSBbXTsKICBsZXQgYXR0YWNobWVudHMgPSBbXTsgLy8ge2lkLG5hbWUsdHlwZSxzaXplLCBraW5kOid0ZXh0J3wnaW1hZ2UnfCdvdGhlcicsIHRleHQ/LCBkYXRhVXJsPywgdHJ1bmNhdGVkP30KICBsZXQgbmV4dEF0dGFjaElkID0gMTsKCiAgLyogLS0tLS0tLS0tLSBBUEkg6KqN6Ki877yIQmFzaWPvvIkgLS0tLS0tLS0tLSAqLwogIC8vIOS4ree2meOBjCBBUElfUEFTUyDjgpLoqK3lrprjgZfjgabjgYTjgovjgajjgY3jga8gL3YxLyog44KCIC9zdG9wIOOCgiA0MDEg44Gr44Gq44KL44CCCiAgLy8g6LOH5qC85oOF5aCx44GvIHNlc3Npb25TdG9yYWdlIOOBq+OBoOOBkee9ruOBj++8iGxvY2FsU3RvcmFnZSDjgavjga/mrovjgZXjgarjgYTvvInjgIIKICBjb25zdCBBVVRIX0tFWSA9ICdrY19iYXNpY19hdXRoJzsKICBjb25zdCBBVVRIX1VTRVJfS0VZID0gJ2tjX2Jhc2ljX3VzZXInOwoKICBmdW5jdGlvbiBiNjR1dGY4KHMpewogICAgLy8gUHl0aG9uIOWBtOOBryBVVEYtOCDjgacgYmFzZTY0IOWMluOBl+OBpuOBhOOCi+OBruOBp+OAgWJ0b2Eg55u05ZG844Gz77yITGF0aW4tMe+8ieOBp+OBrwogICAgLy8g6Z2eIEFTQ0lJIOOBruODkeOCueODr+ODvOODieOBp+S4gOiHtOOBl+OBquOBhOOAggogICAgY29uc3QgYnl0ZXMgPSBuZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUocyk7CiAgICBsZXQgYmluID0gJyc7CiAgICBmb3IgKGNvbnN0IGIgb2YgYnl0ZXMpIGJpbiArPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGIpOwogICAgcmV0dXJuIGJ0b2EoYmluKTsKICB9CiAgZnVuY3Rpb24gZ2V0QXV0aEhlYWRlcigpewogICAgY29uc3QgdiA9IHNlc3Npb25TdG9yYWdlLmdldEl0ZW0oQVVUSF9LRVkpOwogICAgcmV0dXJuIHYgPyB7ICdBdXRob3JpemF0aW9uJzogYEJhc2ljICR7dn1gIH0gOiB7fTsKICB9CiAgZnVuY3Rpb24gYXBpSGVhZGVycyhleHRyYSl7CiAgICByZXR1cm4geyAuLi5CQVNFX0hFQURFUlMsIC4uLmdldEF1dGhIZWFkZXIoKSwgLi4uKGV4dHJhIHx8IHt9KSB9OwogIH0KICBmdW5jdGlvbiBzZXRBdXRoQ3JlZGVudGlhbHModXNlciwgcGFzcyl7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfS0VZLCBiNjR1dGY4KGAke3VzZXJ9OiR7cGFzc31gKSk7CiAgICBzZXNzaW9uU3RvcmFnZS5zZXRJdGVtKEFVVEhfVVNFUl9LRVksIHVzZXIpOwogIH0KICBmdW5jdGlvbiBjbGVhckF1dGhDcmVkZW50aWFscygpewogICAgc2Vzc2lvblN0b3JhZ2UucmVtb3ZlSXRlbShBVVRIX0tFWSk7CiAgICBzZXNzaW9uU3RvcmFnZS5yZW1vdmVJdGVtKEFVVEhfVVNFUl9LRVkpOwogIH0KICBmdW5jdGlvbiBvcGVuQXV0aERpYWxvZyhtc2cpewogICAgaWYgKCFlbHMuYXV0aERpYWxvZykgcmV0dXJuOwogICAgaWYgKGVscy5hdXRoVXNlcikgewogICAgICBlbHMuYXV0aFVzZXIudmFsdWUgPSBzZXNzaW9uU3RvcmFnZS5nZXRJdGVtKEFVVEhfVVNFUl9LRVkpIHx8ICdleGwzJzsKICAgIH0KICAgIGlmIChlbHMuYXV0aFBhc3MpIGVscy5hdXRoUGFzcy52YWx1ZSA9ICcnOwogICAgaWYgKGVscy5hdXRoU3RhdHVzKSBlbHMuYXV0aFN0YXR1cy50ZXh0Q29udGVudCA9IG1zZyB8fCAnJzsKICAgIGlmICh0eXBlb2YgZWxzLmF1dGhEaWFsb2cuc2hvd01vZGFsID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5zaG93TW9kYWwoKTsKICAgIGVsc2UgZWxzLmF1dGhEaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgc2V0VGltZW91dCgoKSA9PiBlbHMuYXV0aFBhc3M/LmZvY3VzKCksIDUwKTsKICB9CiAgZnVuY3Rpb24gY2xvc2VBdXRoRGlhbG9nKCl7CiAgICBpZiAoIWVscy5hdXRoRGlhbG9nKSByZXR1cm47CiAgICBpZiAodHlwZW9mIGVscy5hdXRoRGlhbG9nLmNsb3NlID09PSAnZnVuY3Rpb24nKSBlbHMuYXV0aERpYWxvZy5jbG9zZSgpOwogICAgZWxzZSBlbHMuYXV0aERpYWxvZy5yZW1vdmVBdHRyaWJ1dGUoJ29wZW4nKTsKICB9CgogIGZ1bmN0aW9uIGZtdEJ5dGVzKG4pewogICAgaWYgKCFOdW1iZXIuaXNGaW5pdGUobikpIHJldHVybiAnJzsKICAgIGNvbnN0IHUgPSBbJ0InLCdLQicsJ01CJywnR0InXTsgbGV0IGk9MDsgbGV0IHg9bjsKICAgIHdoaWxlICh4Pj0xMDI0ICYmIGk8dS5sZW5ndGgtMSl7IHgvPTEwMjQ7IGkrKzsgfQogICAgcmV0dXJuIGAke3gudG9GaXhlZChpPzE6MCl9ICR7dVtpXX1gOwogIH0KCiAgZnVuY3Rpb24gcmVuZGVyQXR0YWNobWVudHMoKXsKICAgIGNvbnN0IGxpc3QgPSBlbHMuYXR0YWNobWVudHNMaXN0OwogICAgbGlzdC5pbm5lckhUTUwgPSAnJzsKICAgIGlmICghYXR0YWNobWVudHMubGVuZ3RoKXsKICAgICAgZWxzLmF0dGFjaG1lbnRzQmFyLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpOwogICAgICByZXR1cm47CiAgICB9CiAgICBlbHMuYXR0YWNobWVudHNCYXIuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7CgogICAgZm9yIChjb25zdCBhIG9mIGF0dGFjaG1lbnRzKXsKICAgICAgY29uc3QgY2hpcCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgICBjaGlwLmNsYXNzTmFtZSA9ICdjaGlwJzsKCiAgICAgIGNvbnN0IGljb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdzcGFuJyk7CiAgICAgIGljb24udGV4dENvbnRlbnQgPSAoYS5raW5kID09PSAnaW1hZ2UnKSA/ICfwn5a877iPJyA6IChhLmtpbmQgPT09ICd0ZXh0JykgPyAn8J+ThCcgOiAn8J+Tpic7CgogICAgICBjb25zdCBuYW1lID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogICAgICBuYW1lLnN0eWxlLm92ZXJmbG93ID0gJ2hpZGRlbic7CiAgICAgIG5hbWUuc3R5bGUudGV4dE92ZXJmbG93ID0gJ2VsbGlwc2lzJzsKICAgICAgbmFtZS5zdHlsZS53aGl0ZVNwYWNlID0gJ25vd3JhcCc7CiAgICAgIG5hbWUuc3R5bGUubWF4V2lkdGggPSAnNTVjaCc7CiAgICAgIG5hbWUudGV4dENvbnRlbnQgPSBhLm5hbWU7CgogICAgICBjb25zdCBzaXplID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc21hbGwnKTsKICAgICAgc2l6ZS50ZXh0Q29udGVudCA9IGZtdEJ5dGVzKGEuc2l6ZSk7CgogICAgICBjb25zdCByZW1vdmUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdidXR0b24nKTsKICAgICAgcmVtb3ZlLnRpdGxlID0gJ1JlbW92ZSc7CiAgICAgIHJlbW92ZS50ZXh0Q29udGVudCA9ICfinJUnOwogICAgICByZW1vdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGF0dGFjaG1lbnRzID0gYXR0YWNobWVudHMuZmlsdGVyKHggPT4geC5pZCAhPT0gYS5pZCk7CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfSk7CgogICAgICBjaGlwLmFwcGVuZChpY29uLCBuYW1lLCBzaXplLCByZW1vdmUpOwogICAgICBsaXN0LmFwcGVuZENoaWxkKGNoaXApOwogICAgfQogIH0KCiAgY29uc3QgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUgPSAyMDAwMDsKICBjb25zdCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgPSAxMjsKCiAgZnVuY3Rpb24gcmVhZEZpbGVBc1RleHQoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc1RleHQoZmlsZSk7CiAgICB9KTsKICB9CiAgZnVuY3Rpb24gcmVhZEZpbGVBc0RhdGFVUkwoZmlsZSl7CiAgICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUscmVqZWN0KT0+ewogICAgICBjb25zdCByID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgci5vbmxvYWQgPSAoKT0+IHJlc29sdmUoU3RyaW5nKHIucmVzdWx0fHwnJykpOwogICAgICByLm9uZXJyb3IgPSAoKT0+IHJlamVjdChuZXcgRXJyb3IoJ3JlYWQgZmFpbGVkJykpOwogICAgICByLnJlYWRBc0RhdGFVUkwoZmlsZSk7CiAgICB9KTsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIGFkZEF0dGFjaG1lbnRzRnJvbUZpbGVMaXN0KGZpbGVMaXN0KXsKICAgIGNvbnN0IGZpbGVzID0gQXJyYXkuZnJvbShmaWxlTGlzdCB8fCBbXSk7CiAgICBpZiAoIWZpbGVzLmxlbmd0aCkgcmV0dXJuOwoKICAgIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcy5zbGljZSgwLCBNYXRoLm1heCgwLCBNQVhfVE9UQUxfQVRUQUNITUVOVFMgLSBhdHRhY2htZW50cy5sZW5ndGgpKSl7CiAgICAgIGNvbnN0IGlkID0gbmV4dEF0dGFjaElkKys7CiAgICAgIGNvbnN0IHR5cGUgPSBmaWxlLnR5cGUgfHwgJyc7CiAgICAgIGNvbnN0IG5hbWUgPSBmaWxlLm5hbWUgfHwgYGZpbGVfJHtpZH1gOwogICAgICBjb25zdCBzaXplID0gZmlsZS5zaXplIHx8IDA7CgogICAgICBpZiAodHlwZS5zdGFydHNXaXRoKCdpbWFnZS8nKSl7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOeUu+WDj+OBr+OCteODneODvOODiOOBleOCjOOBpuOBhOOBvuOBm+OCk++8iC0tbW1wcm9qIOOBquOBl++8ieOAguODhuOCreOCueODiOODleOCoeOCpOODq+OBruOBv+a3u+S7mOWPr+iDveOBp+OBmeOAgmApOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CgogICAgICBjb25zdCBsb29rc1RleHQgPSB0eXBlLnN0YXJ0c1dpdGgoJ3RleHQvJykKICAgICAgICB8fCAvXC4odHh0fG1kfGpzb258anNvbmx8Y3N2fHRzdnxsb2d8eWFtbHx5bWx8cHl8anN8dHN8aHRtbHxjc3N8eG1sfGluaXx0b21sKSQvaS50ZXN0KG5hbWUpOwoKICAgICAgaWYgKGxvb2tzVGV4dCl7CiAgICAgICAgdHJ5ewogICAgICAgICAgbGV0IHRleHQgPSBhd2FpdCByZWFkRmlsZUFzVGV4dChmaWxlKTsKICAgICAgICAgIGxldCB0cnVuY2F0ZWQgPSBmYWxzZTsKICAgICAgICAgIGlmICh0ZXh0Lmxlbmd0aCA+IE1BWF9URVhUX0NIQVJTX1BFUl9GSUxFKXsKICAgICAgICAgICAgdGV4dCA9IHRleHQuc2xpY2UoMCwgTUFYX1RFWFRfQ0hBUlNfUEVSX0ZJTEUpOwogICAgICAgICAgICB0cnVuY2F0ZWQgPSB0cnVlOwogICAgICAgICAgfQogICAgICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOid0ZXh0JywgdGV4dCwgdHJ1bmNhdGVkIH0pOwogICAgICAgIH0gY2F0Y2ggewogICAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOODhuOCreOCueODiOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnzogJHtuYW1lfWApOwogICAgICAgIH0KICAgICAgICBjb250aW51ZTsKICAgICAgfQoKICAgICAgYXR0YWNobWVudHMucHVzaCh7IGlkLCBuYW1lLCB0eXBlLCBzaXplLCBraW5kOidvdGhlcicgfSk7CiAgICB9CgogICAgaWYgKGF0dGFjaG1lbnRzLmxlbmd0aCA+PSBNQVhfVE9UQUxfQVRUQUNITUVOVFMpIHsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvmt7vku5hdIOS4iumZkOOBq+mBlOOBl+OBvuOBl+OBnyAoJHtNQVhfVE9UQUxfQVRUQUNITUVOVFN9IGZpbGVzKS5gKTsKICAgIH0KICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgfQoKICBsZXQgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICBjb25zdCBOTCA9ICdcbic7CgogIGNvbnN0IEFQUF9WRVJTSU9OID0gJzEuMDEnOwogIGNvbnN0IFBST0pFQ1RfVVJMID0gJ2h0dHBzOi8vd3d3LmxvY2FsbGlnaHRhaS5jb20vc2N1JzsKICBjb25zdCBJTVBSSU5UX1VSTCA9ICdodHRwczovL3d3dy5zcndhcmUubmV0L2ltcHJpbnQnOwogIAogIGxldCBzdHJlYW1TdGFydFRpbWUgPSAwOwogIGxldCBsYXN0VXBkYXRlVGltZSA9IDA7CiAgbGV0IGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgY29uc3QgVVBEQVRFX0lOVEVSVkFMID0gMjAwOyAvLyBtcwoKICAvKiAtLS0tLS0tLS0tIEF1dG8tU2Nyb2xsIChnZWZsaWNrdCkgLS0tLS0tLS0tLSAqLwogIGxldCBhdXRvc2Nyb2xsID0gdHJ1ZTsKICBsZXQgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICBjb25zdCBCT1RUT01fRVBTID0gODA7CgogIGZ1bmN0aW9uIGF0Qm90dG9tKCl7CiAgICBjb25zdCBzID0gZWxzLnNjcm9sbGVyOwogICAgcmV0dXJuIChzLnNjcm9sbEhlaWdodCAtIHMuc2Nyb2xsVG9wIC0gcy5jbGllbnRIZWlnaHQpIDw9IEJPVFRPTV9FUFM7CiAgfQogIGZ1bmN0aW9uIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKXsKICAgIGNvbnN0IHMgPSBlbHMuc2Nyb2xsZXI7CiAgICBjb25zdCBwcmV2ID0gcy5zdHlsZS5zY3JvbGxCZWhhdmlvcjsKICAgIHMuc3R5bGUuc2Nyb2xsQmVoYXZpb3IgPSAnYXV0byc7CiAgICBzLnNjcm9sbFRvcCA9IHMuc2Nyb2xsSGVpZ2h0OwogICAgcy5zdHlsZS5zY3JvbGxCZWhhdmlvciA9IHByZXYgfHwgJyc7CiAgfQogIGZ1bmN0aW9uIG1heWJlU2Nyb2xsKGZvcmNlPWZhbHNlKXsKICAgIGlmIChmb3JjZSB8fCBhdXRvc2Nyb2xsIHx8IGF0Qm90dG9tKCkpIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGNvbnN0IHNob3cgPSAhYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93Jywgc2hvdyk7CiAgICBpZiAoIXNob3cpIGF1dG9zY3JvbGwgPSB0cnVlOwogIH0KICBlbHMuc2Nyb2xsZXIuYWRkRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgKCkgPT4gewogICAgaWYgKGlzUmVuZGVyaW5nKSByZXR1cm47CiAgICBhdXRvc2Nyb2xsID0gYXRCb3R0b20oKTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QudG9nZ2xlKCdzaG93JywgIWF1dG9zY3JvbGwpOwogIH0pOwogIGVscy50b0JvdHRvbS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsgYXV0b3Njcm9sbCA9IHRydWU7IG1heWJlU2Nyb2xsKHRydWUpOyB9KTsKCiAgLyogLS0tLS0tLS0tLSBNYXJrZG93biAtLS0tLS0tLS0tICovCiAgY29uc3QgZXNjID0gcyA9PiBTdHJpbmcocykucmVwbGFjZSgvWyY8PiInXS9nLCBjID0+ICh7JyYnOicmYW1wOycsJzwnOicmbHQ7JywnPic6JyZndDsnLCciJzonJnF1b3Q7JywiJyI6JyYjMzk7J31bY10pKTsKICBmdW5jdGlvbiBpbmxpbmVNZChzKXsKICAgIC8vIFByb3RlY3QgaW5saW5lIGNvZGUgd2l0aCBwbGFjZWhvbGRlcnMgZmlyc3QKICAgIGNvbnN0IGNvZGVQbGFjZWhvbGRlcnMgPSBbXTsKICAgIHMgPSBzLnJlcGxhY2UoL2AoW15gXSspYC9nLCAoXyxwKSA9PiB7IGNvbnN0IGlkeCA9IGNvZGVQbGFjZWhvbGRlcnMubGVuZ3RoOyBjb2RlUGxhY2Vob2xkZXJzLnB1c2gocCk7IHJldHVybiBgXHgwMENPREUke2lkeH1ceDAwYDsgfSk7CiAgICAvLyBJbWFnZXMKICAgIC8vIOWklumDqOeUu+WDj+OBr+iHquWLleiqreOBv+i+vOOBv+OBl+OBquOBhO+8iOmWsuimp+OBrui/vei3oeOBq+S9v+OBiOOCi++8ieOAgmh0dHBzL2h0dHAg44Gu44G/44Oq44Oz44Kv44Go44GX44Gm5Ye644GZ44CCCiAgICBzID0gcy5yZXBsYWNlKC8hXFsoW15cXV0qKVxdXCgoaHR0cHM/OlteKVxzXSspXCkvZywgKF8sYWx0LHVybCk9Pic8YSBocmVmPSInK3VybCsnIiB0YXJnZXQ9Il9ibGFuayIgcmVsPSJub29wZW5lciBub3JlZmVycmVyIj5b55S75YOPOiAnKyhhbHR8fHVybCkrJ108L2E+Jyk7CiAgICAvLyBCb2xkCiAgICBzID0gcy5yZXBsYWNlKC9cKlwqKFteKl0rKVwqXCovZywgKF8scCk9Pic8c3Ryb25nPicrcCsnPC9zdHJvbmc+Jyk7CiAgICAvLyBJdGFsaWMgKG9ubHkgb3V0c2lkZSBjb2RlKQogICAgcyA9IHMucmVwbGFjZSgvXCooW14qXSspXCovZywgKF8scCk9Pic8ZW0+JytwKyc8L2VtPicpOwogICAgLy8gTGlua3MKICAgIHMgPSBzLnJlcGxhY2UoL1xbKFteXF1dKylcXVwoKGh0dHBzPzpbXlwpXHNdKylbXlwpXSpcKS9nLCAoXyx0eHQsdXJsKT0+JzxhIGhyZWY9IicrdXJsKyciIHRhcmdldD0iX2JsYW5rIiByZWw9Im5vb3BlbmVyIj4nK3R4dCsnPC9hPicpOwogICAgLy8gQXV0by1saW5rIGJhcmUgVVJMcyAoc2tpcCBhbHJlYWR5LWdlbmVyYXRlZCBIVE1MIHRhZ3MpCiAgICBzID0gcy5yZXBsYWNlKC8oPFtePl0qPil8KFxiaHR0cHM/OlwvXC9bXlxzKe+8ieOAjeOAj+OAguOAgV0rKS9nLCAobWF0Y2gsIHRhZywgdXJsKSA9PiB7CiAgICAgIGlmICh0YWcpIHJldHVybiB0YWc7CiAgICAgIHJldHVybiAnPGEgaHJlZj0iJyt1cmwrJyIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPicrdXJsKyc8L2E+JzsKICAgIH0pOwogICAgLy8gUmVzdG9yZSBjb2RlCiAgICBzID0gcy5yZXBsYWNlKC9ceDAwQ09ERShcZCspXHgwMC9nLCAoXyxpZHgpID0+ICc8Y29kZT4nK2NvZGVQbGFjZWhvbGRlcnNbcGFyc2VJbnQoaWR4KV0rJzwvY29kZT4nKTsKICAgIHJldHVybiBzOwogIH0KICBmdW5jdGlvbiBtZFRvSHRtbChtZCl7CiAgICBtZCA9IFN0cmluZyhtZCA/PyAnJyk7CiAgICBtZCA9IG1kLnJlcGxhY2UoL1xyL2csJycpOwogICAgY29uc3QgbGluZXMgPSBtZC5zcGxpdCgnXG4nKTsKICAgIGxldCBodG1sID0gJycsIGluQ29kZSA9IGZhbHNlLCBjb2RlTGFuZyA9ICcnLCBsaXN0VHlwZSA9IG51bGw7CiAgICBjb25zdCBjbG9zZUxpc3QgPSAoKSA9PiB7IGlmIChsaXN0VHlwZSkgeyBodG1sICs9IGA8LyR7bGlzdFR5cGV9PmA7IGxpc3RUeXBlPW51bGw7IH0gfTsKICAgIGZvciAobGV0IGk9MDtpPGxpbmVzLmxlbmd0aDtpKyspewogICAgICBsZXQgbGluZSA9IGxpbmVzW2ldOwogICAgICBpZiAoIWluQ29kZSAmJiAvXlxzKmBgYC8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgbSA9IGxpbmUubWF0Y2goL15ccypgYGBccyooW1x3Ky1dKikvKTsKICAgICAgICBjb2RlTGFuZyA9IG0gJiYgbVsxXSA/IG1bMV0gOiAnJzsKICAgICAgICBodG1sICs9IGA8ZGl2IGNsYXNzPSJjb2Rld3JhcCI+PHByZT48Y29kZSBkYXRhLWxhbmc9IiR7ZXNjKGNvZGVMYW5nKX0iPmA7CiAgICAgICAgaW5Db2RlID0gdHJ1ZTsgY29udGludWU7CiAgICAgIH0KICAgICAgaWYgKGluQ29kZSl7CiAgICAgICAgaWYgKC9eXHMqYGBgXHMqJC8udGVzdChsaW5lKSl7IGh0bWwgKz0gYDwvY29kZT48L3ByZT48YnV0dG9uIGNsYXNzPSJjb3B5YnRuIj5Db3B5PC9idXR0b24+PC9kaXY+YDsgaW5Db2RlPWZhbHNlOyBjb250aW51ZTsgfQogICAgICAgIGh0bWwgKz0gZXNjKGxpbmUpICsgJ1xuJzsgY29udGludWU7CiAgICAgIH0KICAgICAgY29uc3QgaCA9IGxpbmUubWF0Y2goL15ccyooI3sxLDZ9KVxzKyguKikkLyk7CiAgICAgIGlmIChoKXsgY2xvc2VMaXN0KCk7IGNvbnN0IGx2bD1oWzFdLmxlbmd0aDsgaHRtbCArPSBgPGgke2x2bH0+JHtpbmxpbmVNZChlc2MoaFsyXSkpfTwvaCR7bHZsfT5gOyBjb250aW51ZTsgfQogICAgICBpZiAoL15ccyooPzotLS18XCpcKlwqfF9fXylccyokLy50ZXN0KGxpbmUpKXsgY2xvc2VMaXN0KCk7IGh0bWwgKz0gJzxoci8+JzsgY29udGludWU7IH0KICAgICAgaWYgKC9eXHMqPlxzPy8udGVzdChsaW5lKSl7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgY29uc3QgcXVvdGVMaW5lcyA9IFtdOwogICAgICAgIHdoaWxlIChpIDwgbGluZXMubGVuZ3RoICYmIC9eXHMqPlxzPy8udGVzdChsaW5lc1tpXSkpIHsKICAgICAgICAgIHF1b3RlTGluZXMucHVzaChsaW5lc1tpXS5yZXBsYWNlKC9eXHMqPlxzPy8sICcnKSk7CiAgICAgICAgICBpKys7CiAgICAgICAgfQogICAgICAgIGktLTsKICAgICAgICBjb25zdCBxdW90ZVRleHQgPSBpbmxpbmVNZChlc2MocXVvdGVMaW5lcy5qb2luKCdcbicpKSk7CiAgICAgICAgaHRtbCArPSBgPGJsb2NrcXVvdGU+PHA+JHtxdW90ZVRleHQucmVwbGFjZSgvXG4vZywgJzxiciAvPicpfTwvcD48L2Jsb2NrcXVvdGU+YDsKICAgICAgICBjb250aW51ZTsKICAgICAgfQogICAgICAKICAgICAgLy8gLS0tIFNUQVJUOiBUYWJsZSBQYXJzaW5nIC0tLQogICAgICBjb25zdCBpc1RhYmxlTGluZSA9IChsKSA9PiBsLmluY2x1ZGVzKCd8Jyk7CiAgICAgIGNvbnN0IGlzU2VwYXJhdG9yTGluZSA9IChsKSA9PiAvXlxzKlx8PyhbLTpdK1x8KStbLTpdK1x8P1xzKiQvLnRlc3QobC50cmltKCkpOwogICAgICAKICAgICAgaWYgKGlzVGFibGVMaW5lKGxpbmUpICYmIGkgKyAxIDwgbGluZXMubGVuZ3RoICYmIGlzU2VwYXJhdG9yTGluZShsaW5lc1tpICsgMV0pKSB7CiAgICAgICAgY2xvc2VMaXN0KCk7CiAgICAgICAgaHRtbCArPSAnPHRhYmxlPic7CiAgICAgICAgCiAgICAgICAgLy8gSGVhZGVyCiAgICAgICAgY29uc3QgaGVhZGVycyA9IGxpbmUuc3BsaXQoJ3wnKS5tYXAoaCA9PiBoLnRyaW0oKSkuc2xpY2UoMSwgLTEpOwogICAgICAgIGh0bWwgKz0gJzx0aGVhZD48dHI+JzsKICAgICAgICBoZWFkZXJzLmZvckVhY2goaCA9PiB7IGh0bWwgKz0gYDx0aD4ke2lubGluZU1kKGVzYyhoKSl9PC90aD5gOyB9KTsKICAgICAgICBodG1sICs9ICc8L3RyPjwvdGhlYWQ+JzsKICAgICAgICAKICAgICAgICBpICs9IDI7IC8vIFNraXAgaGVhZGVyIHVuZCBzZXBhcmF0b3IgbGluZQogICAgICAgIAogICAgICAgIC8vIEJvZHkKICAgICAgICBodG1sICs9ICc8dGJvZHk+JzsKICAgICAgICB3aGlsZSAoaSA8IGxpbmVzLmxlbmd0aCAmJiBpc1RhYmxlTGluZShsaW5lc1tpXSkpIHsKICAgICAgICAgICAgY29uc3QgY2VsbHMgPSBsaW5lc1tpXS5zcGxpdCgnfCcpLm1hcChjID0+IGMudHJpbSgpKS5zbGljZSgxLCAtMSk7CiAgICAgICAgICAgIGlmIChjZWxscy5sZW5ndGggPiAwKSB7CiAgICAgICAgICAgICAgICBodG1sICs9ICc8dHI+JzsKICAgICAgICAgICAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgaGVhZGVycy5sZW5ndGg7IGorKykgewogICAgICAgICAgICAgICAgICAgIGNvbnN0IGNlbGxDb250ZW50ID0gY2VsbHNbal0gfHwgJyc7CiAgICAgICAgICAgICAgICAgICAgaHRtbCArPSBgPHRkPiR7aW5saW5lTWQoZXNjKGNlbGxDb250ZW50KSl9PC90ZD5gOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaHRtbCArPSAnPC90cj4nOwogICAgICAgICAgICB9CiAgICAgICAgICAgIGkrKzsKICAgICAgICB9CiAgICAgICAgaS0tOwogICAgICAgIAogICAgICAgIGh0bWwgKz0gJzwvdGJvZHk+PC90YWJsZT4nOwogICAgICAgIGNvbnRpbnVlOwogICAgICB9CiAgICAgIC8vIC0tLSBFTkQ6IFRhYmxlIFBhcnNpbmcgLS0tCgogICAgICBsZXQgbTsKICAgICAgaWYgKG0gPSBsaW5lLm1hdGNoKC9eXHMqWy0qK11ccytcWyggfHh8WClcXVxzKyguKikkLykpewogICAgICAgIGNvbnN0IGNoZWNrZWQgPSAveC9pLnRlc3QobVsxXSk7CiAgICAgICAgaWYgKGxpc3RUeXBlIT09J3VsJyl7IGNsb3NlTGlzdCgpOyBsaXN0VHlwZT0ndWwnOyBodG1sICs9ICc8dWw+JzsgfQogICAgICAgIGh0bWwgKz0gYDxsaSBjbGFzcz0iJHtjaGVja2VkPyd0YXNrLWRvbmUnOicnfSI+YCsoY2hlY2tlZD8n4pyFICc6J+KsnCAnKStpbmxpbmVNZChlc2MobVsyXSkpKyc8L2xpPic7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKlstKitdXHMrLy50ZXN0KGxpbmUpKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0ndWwnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSd1bCc7IGh0bWwgKz0gJzx1bD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKGxpbmUucmVwbGFjZSgvXlxzKlstKitdXHMrLywnJykpKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmIChtID0gbGluZS5tYXRjaCgvXlxzKihcZCspXC5ccysoLiopJC8pKXsKICAgICAgICBpZiAobGlzdFR5cGUhPT0nb2wnKXsgY2xvc2VMaXN0KCk7IGxpc3RUeXBlPSdvbCc7IGh0bWwgKz0gJzxvbD4nOyB9CiAgICAgICAgaHRtbCArPSBgPGxpPiR7aW5saW5lTWQoZXNjKG1bMl0pKX08L2xpPmA7IGNvbnRpbnVlOwogICAgICB9CiAgICAgIGlmICgvXlxzKiQvLnRlc3QobGluZS50cmltKCkpKXsgY2xvc2VMaXN0KCk7IH0KICAgICAgZWxzZSB7IGNsb3NlTGlzdCgpOyBodG1sICs9IGA8cD4ke2lubGluZU1kKGVzYyhsaW5lKSl9PC9wPmA7IH0KICAgIH0KICAgIGNsb3NlTGlzdCgpOyByZXR1cm4gaHRtbDsKICB9CiAgZnVuY3Rpb24gZW5oYW5jZUNvZGVDb3B5KGNvbnRhaW5lcil7CiAgICBjb250YWluZXIucXVlcnlTZWxlY3RvckFsbCgnLmNvZGV3cmFwIC5jb3B5YnRuOm5vdChbZGF0YS1pbml0XSknKS5mb3JFYWNoKGJ0bj0+ewogICAgICBidG4uZGF0YXNldC5pbml0ID0gJzEnOwogICAgICBidG4uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKT0+ewogICAgICAgIGNvbnN0IGNvZGUgPSBidG4ucGFyZW50RWxlbWVudC5xdWVyeVNlbGVjdG9yKCdjb2RlJyk7CiAgICAgICAgY29uc3QgdGV4dCA9IGNvZGUgPyBjb2RlLnRleHRDb250ZW50IDogJyc7CiAgICAgICAgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQodGV4dCkudGhlbigoKT0+eyBidG4udGV4dENvbnRlbnQ9J0NvcGllZCc7IHNldFRpbWVvdXQoKCk9PmJ0bi50ZXh0Q29udGVudD0nQ29weScsMTAwMCk7IH0pOwogICAgICB9KTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSBVSSBoZWxwZXJzIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiByZW5kZXJCdWJibGVDb250ZW50KGVsLCByb2xlLCBjb250ZW50KXsKICAgIGVsLmNsYXNzTGlzdC5hZGQoJ21kJyk7IGVsLmlubmVySFRNTCA9IG1kVG9IdG1sKG5vcm1hbGl6ZUNvbnRlbnQoY29udGVudCkpOyBlbmhhbmNlQ29kZUNvcHkoZWwpOwogIH0KICAKICBmdW5jdGlvbiBhZGRNZXNzYWdlKHJvbGUsIGNvbnRlbnQpIHsKICAgIGNvbnN0IGNscyA9IHJvbGUgPT09ICd1c2VyJyA/ICd1c2VyJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ3N5cycgOiAnYm90JzsKICAgIGNvbnN0IGF2YXRhciA9IHJvbGUgPT09ICd1c2VyJyA/ICfwn6eRJyA6IHJvbGUgPT09ICdzeXN0ZW0nID8gJ+Kame+4jycgOiAn8J+klic7CiAgICBjb25zdCBtc2dFbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogICAgbXNnRWwuY2xhc3NOYW1lID0gYG1zZyAke2Nsc31gOwogICAgbXNnRWwuaW5uZXJIVE1MID0gYAogICAgICA8ZGl2IGNsYXNzPSJhdmF0YXIiPiR7YXZhdGFyfTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJtZXNzYWdlLWNvbnRlbnQtd3JhcHBlciI+CiAgICAgICAgPGRpdiBjbGFzcz0iYnViYmxlIj4KICAgICAgICAgIDxkaXYgY2xhc3M9Im1ldGEiPiR7cm9sZX08L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNvbnRlbnQgbWQiPjwvZGl2PgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj5gOwogICAgZWxzLmxvZy5hcHBlbmRDaGlsZChtc2dFbCk7CiAgICBjb25zdCBjb250ZW50RWwgPSBtc2dFbC5xdWVyeVNlbGVjdG9yKCcuY29udGVudCcpOwogICAgcmVuZGVyQnViYmxlQ29udGVudChjb250ZW50RWwsIHJvbGUsIGNvbnRlbnQpOwogICAgaWYgKCFpc1JlbmRlcmluZykgbWF5YmVTY3JvbGwoKTsKICAgIHJldHVybiBjb250ZW50RWw7CiAgfQogIAogIGZ1bmN0aW9uIGFkZFRoaW5rQnViYmxlKGFzc2lzdGFudENvbnRlbnRFbCkgewogICAgY29uc3Qgd3JhcHBlciA9IGFzc2lzdGFudENvbnRlbnRFbC5jbG9zZXN0KCcubWVzc2FnZS1jb250ZW50LXdyYXBwZXInKTsKICAgIGlmICghd3JhcHBlcikgcmV0dXJuIG51bGw7CiAgICBjb25zdCBidWJibGUgPSB3cmFwcGVyLnF1ZXJ5U2VsZWN0b3IoJy5idWJibGUnKTsKICAgIGlmICghYnViYmxlKSByZXR1cm4gbnVsbDsKCiAgICBjb25zdCB0aGlua0VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7CiAgICB0aGlua0VsLmNsYXNzTmFtZSA9ICd0aGluay1idWJibGUnOwogICAgdGhpbmtFbC5pbm5lckhUTUwgPSBgCiAgICAgIDxkZXRhaWxzPgogICAgICAgIDxzdW1tYXJ5PvCfpJQgU2hvdyB0aGlua2luZzwvc3VtbWFyeT4KICAgICAgICA8ZGl2IGNsYXNzPSJ0aGluay1jb250ZW50IG1kIj48L2Rpdj4KICAgICAgPC9kZXRhaWxzPgogICAgYDsKICAgIHdyYXBwZXIuaW5zZXJ0QmVmb3JlKHRoaW5rRWwsIGJ1YmJsZSk7CiAgICByZXR1cm4gdGhpbmtFbC5xdWVyeVNlbGVjdG9yKCcudGhpbmstY29udGVudCcpOwogIH0KCiAgZnVuY3Rpb24gc2V0U3RyZWFtaW5nU3RhdGUoYWN0aXZlKSB7IGVscy5zZW5kQnRuLmRpc2FibGVkID0gYWN0aXZlOyBlbHMuc3RvcEJ0bi5kaXNhYmxlZCA9ICFhY3RpdmU7IH0KCiAgZnVuY3Rpb24gcmVuZGVyQWxsKCl7CiAgICBpc1JlbmRlcmluZyA9IHRydWU7CiAgICBlbHMubG9nLmlubmVySFRNTCA9ICcnOwogICAgZm9yIChjb25zdCBtIG9mIG1lc3NhZ2VzKSBhZGRNZXNzYWdlKG0ucm9sZSwgbS5jb250ZW50KTsKICAgIGZvcmNlU2Nyb2xsVG9Cb3R0b20oKTsKICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICBhdXRvc2Nyb2xsID0gdHJ1ZTsKICAgIGVscy50b0JvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKCdzaG93Jyk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIE5vcm1hbGl6ZXIgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIG5vcm1hbGl6ZUNvbnRlbnQoYyl7CiAgICBpZiAodHlwZW9mIGMgPT09ICdzdHJpbmcnKSByZXR1cm4gYzsKICAgIGlmIChBcnJheS5pc0FycmF5KGMpKSB7CiAgICAgIHJldHVybiBjLm1hcChwID0+IHsKICAgICAgICBpZiAodHlwZW9mIHAgPT09ICdzdHJpbmcnKSByZXR1cm4gcDsKICAgICAgICBpZiAocD8udHlwZSA9PT0gJ3RleHQnKSByZXR1cm4gcC50ZXh0ID8/ICcnOwogICAgICAgIGlmIChwPy50eXBlID09PSAnaW1hZ2VfdXJsJykgewogICAgICAgICAgY29uc3QgdSA9IHA/LmltYWdlX3VybD8udXJsIHx8ICcnOwogICAgICAgICAgcmV0dXJuIHUgPyBgIVtpbWFnZV0oJHt1fSlgIDogJyc7CiAgICAgICAgfQogICAgICAgIHJldHVybiBwPy50ZXh0ID8/ICcnOwogICAgICB9KS5qb2luKFN0cmluZy5mcm9tQ2hhckNvZGUoMTApKTsKICAgIH0KICAgIGlmIChjICYmIHR5cGVvZiBjID09PSAnb2JqZWN0JyAmJiAndGV4dCcgaW4gYykgcmV0dXJuIFN0cmluZyhjLnRleHQgPz8gJycpOwogICAgdHJ5IHsgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGMpOyB9IGNhdGNoIHsgcmV0dXJuIFN0cmluZyhjKTsgfQogIH0KICBmdW5jdGlvbiBub3JtYWxpemVSb2xlKHIpeyBjb25zdCBzPVN0cmluZyhyfHwnJykudG9Mb3dlckNhc2UoKTsgaWYgKHMuc3RhcnRzV2l0aCgnYXNzaXN0JykpIHJldHVybiAnYXNzaXN0YW50JzsgaWYgKHMuc3RhcnRzV2l0aCgnc3lzJykpIHJldHVybiAnc3lzdGVtJzsgcmV0dXJuICd1c2VyJzsgfQoKICAvKiAtLS0tLS0tLS0tIEV4cG9ydCBpbmtsLiBzeXN0ZW0gLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGJ1aWxkRXhwb3J0QXJyYXkoKXsKICAgIGNvbnN0IG91dCA9IG1lc3NhZ2VzLm1hcChtID0+ICh7IHJvbGU6IG0ucm9sZSwgY29udGVudDogbS5jb250ZW50IH0pKTsKICAgIGNvbnN0IHN5c1RleHQgPSAoZWxzLnN5c3RlbS52YWx1ZSB8fCAnJykudHJpbSgpOwogICAgY29uc3QgZmlyc3RJc1N5c3RlbSA9IG91dFswXT8ucm9sZSA9PT0gJ3N5c3RlbSc7CiAgICBpZiAoc3lzVGV4dCkgewogICAgICBpZiAoIWZpcnN0SXNTeXN0ZW0pIG91dC51bnNoaWZ0KHsgcm9sZTogJ3N5c3RlbScsIGNvbnRlbnQ6IHN5c1RleHQgfSk7CiAgICAgIGVsc2UgaWYgKG91dFswXS5jb250ZW50ICE9PSBzeXNUZXh0KSBvdXRbMF0gPSB7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNUZXh0IH07CiAgICB9CiAgICByZXR1cm4gb3V0OwogIH0KICAgIAogIGZ1bmN0aW9uIHVwZGF0ZVRva2VuUmF0ZShidWZmZXIsIGZvcmNlRmluYWwgPSBmYWxzZSkgewogICAgY29uc3Qgbm93ID0gcGVyZm9ybWFuY2Uubm93KCk7CiAgICBpZiAoIWZvcmNlRmluYWwgJiYgbm93IC0gbGFzdFVwZGF0ZVRpbWUgPCBVUERBVEVfSU5URVJWQUwpIHJldHVybjsKICAgIGxhc3RVcGRhdGVUaW1lID0gbm93OwoKICAgIGNvbnN0IGVsYXBzZWRTZWNvbmRzID0gKG5vdyAtIHN0cmVhbVN0YXJ0VGltZSkgLyAxMDAwOwogICAgaWYgKGVsYXBzZWRTZWNvbmRzIDwgMC4xKSByZXR1cm47CgogICAgbGV0IHRva2VucywgcmF0ZSwgbGFiZWw7CiAgICBpZiAoZmluYWxDb21wbGV0aW9uVG9rZW5zID4gMCkgewogICAgICAgIHRva2VucyA9IGZpbmFsQ29tcGxldGlvblRva2VuczsKICAgICAgICByYXRlID0gdG9rZW5zIC8gZWxhcHNlZFNlY29uZHM7CiAgICAgICAgbGFiZWwgPSBgw5ggJHtyYXRlLnRvRml4ZWQoMSl9IHQvc2A7CiAgICB9IGVsc2UgewogICAgICAgIC8vIFNjaMOkdHp1bmc6IGNhLiA0IFplaWNoZW4gcHJvIFRva2VuCiAgICAgICAgdG9rZW5zID0gTWF0aC5yb3VuZChidWZmZXIubGVuZ3RoIC8gNCk7CiAgICAgICAgcmF0ZSA9IHRva2VucyAvIGVsYXBzZWRTZWNvbmRzOwogICAgICAgIGxhYmVsID0gYH4ke3JhdGUudG9GaXhlZCgxKX0gdC9zYDsKICAgIH0KICAgIGVscy50b2tlblJhdGUudGV4dENvbnRlbnQgPSBsYWJlbDsKICB9CgoKICAvKiAtLS0tLS0tLS0tIEthZ2dsZSBHUFUg5q6L44KK5pmC6ZaTIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3F1b3RhIOOCkuWPqeOBj+OAguODreODvOOCq+ODqyB1enUg44Gr44GvIC9xdW90YSDjgYznhKHjgYTjga7jgafjgIEKICAvLyDjgZ3jga7loLTlkIjjga8gNDA0IOOBjOi/lOOCiuOAgeihqOekuuOBr+WHuuOBquOBhO+8iOOCqOODqeODvOaJseOBhOOBq+OBl+OBquOBhO+8ieOAggogIGxldCBxdW90YVRpbWVyID0gbnVsbDsKCiAgZnVuY3Rpb24gZm10TGVmdChzZWMpIHsKICAgIGlmIChzZWMgPT0gbnVsbCB8fCBzZWMgPCAwKSByZXR1cm4gJyc7CiAgICBjb25zdCBoID0gTWF0aC5mbG9vcihzZWMgLyAzNjAwKTsKICAgIGNvbnN0IG0gPSBNYXRoLmZsb29yKChzZWMgJSAzNjAwKSAvIDYwKTsKICAgIHJldHVybiBg5q6L44KKICR7U3RyaW5nKGgpLnBhZFN0YXJ0KDIsICcwJyl9OiR7U3RyaW5nKG0pLnBhZFN0YXJ0KDIsICcwJyl9YDsKICB9CgogIGFzeW5jIGZ1bmN0aW9uIHJlZnJlc2hRdW90YSgpIHsKICAgIGlmICghZWxzLnF1b3RhTGVmdCkgcmV0dXJuOwogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9xdW90YWAsIHsKICAgICAgICBtZXRob2Q6ICdHRVQnLAogICAgICAgIGhlYWRlcnM6IGFwaUhlYWRlcnMoeyAnQWNjZXB0JzogJ2FwcGxpY2F0aW9uL2pzb24nIH0pLAogICAgICAgIGNhY2hlOiAnbm8tc3RvcmUnLAogICAgICB9KTsKICAgICAgaWYgKCFyZXMub2spIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KICAgICAgY29uc3QgZCA9IGF3YWl0IHJlcy5qc29uKCk7CiAgICAgIGlmICghZC5hdmFpbGFibGUpIHsgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOyByZXR1cm47IH0KCiAgICAgIGNvbnN0IGxlZnQgPSBkLmxlZnRTZWNvbmRzOwogICAgICBlbHMucXVvdGFMZWZ0LnRleHRDb250ZW50ID0gZm10TGVmdChsZWZ0KTsKICAgICAgLy8g5q6L44KK44GM5bCR44Gq44GE44Go6Imy44KS5aSJ44GI44KLCiAgICAgIGVscy5xdW90YUxlZnQuc3R5bGUuY29sb3IgPSBsZWZ0IDwgMTgwMCA/ICcjZmY2YjZiJwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDogbGVmdCA8IDcyMDAgPyAnI2UwYTAzMCcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA6ICcnOwogICAgICBlbHMucXVvdGFMZWZ0LnRpdGxlID0gYEthZ2dsZSBHUFU6IOaui+OCiiAke2ZtdExlZnQobGVmdCkucmVwbGFjZSgn5q6L44KKICcsICcnKX1cbmAKICAgICAgICAgICAgICAgICAgICAgICAgICArIGDkvb/nlKggJHtNYXRoLmZsb29yKGQudXNlZFNlY29uZHMgLyAzNjAwKX1oIC8g5LiK6ZmQICR7TWF0aC5mbG9vcihkLnRvdGFsU2Vjb25kcyAvIDM2MDApfWhgCiAgICAgICAgICAgICAgICAgICAgICAgICAgKyAoZC5yZWZyZXNoVGltZSA/IGBcbuODquOCu+ODg+ODiCAke2QucmVmcmVzaFRpbWV9YCA6ICcnKTsKICAgIH0gY2F0Y2ggKF8pIHsKICAgICAgZWxzLnF1b3RhTGVmdC50ZXh0Q29udGVudCA9ICcnOwogICAgfQogIH0KCiAgZnVuY3Rpb24gc3RhcnRRdW90YVBvbGxpbmcoKSB7CiAgICByZWZyZXNoUXVvdGEoKTsKICAgIC8vIDEg5YiG44GU44Go44CC5q6L6YeP44Gv56eS5Y2Y5L2N44Gn5rib44KL44GM44CB6KGo56S644Gv5YiG5Y2Y5L2N44Gq44Gu44Gn5Y2B5YiG44CCCiAgICBpZiAocXVvdGFUaW1lcikgY2xlYXJJbnRlcnZhbChxdW90YVRpbWVyKTsKICAgIHF1b3RhVGltZXIgPSBzZXRJbnRlcnZhbChyZWZyZXNoUXVvdGEsIDYwMDAwKTsKICB9CgogIC8qIC0tLS0tLS0tLS0g5YGc5q2i44Oc44K/44Oz77yI44Ot44O844Kr44Or54mI44Gu44G/77yJIC0tLS0tLS0tLS0gKi8KICAvLyDkuK3ntpnjga4gL3N0b3Ag44KS5Y+p44GP44CC5Lit57aZ44GM6JC944Gh44KL44Gu44Gn44CB44GT44GuIFVJIOOCguS9v+OBiOOBquOBj+OBquOCi+OAggogIC8vIOODreODvOOCq+ODqyB1enUg44Gr44GvIC9zdG9wIOOBjOeEoeOBhOOBruOBp+OAgTQwNCDjgarjgonjg5zjgr/jg7PjgpLpmqDjgZnjgIIKICBhc3luYyBmdW5jdGlvbiBkZXRlY3RTdG9wU3VwcG9ydCgpIHsKICAgIGlmICghZWxzLnN0b3BSb3cpIHJldHVybjsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlcyA9IGF3YWl0IGZldGNoKGAke2xvY2F0aW9uLm9yaWdpbn0vcXVvdGFgLCB7CiAgICAgICAgbWV0aG9kOiAnR0VUJywgaGVhZGVyczogYXBpSGVhZGVycygpLCBjYWNoZTogJ25vLXN0b3JlJywKICAgICAgfSk7CiAgICAgIC8vIC9xdW90YSDjgYwgSlNPTiDjgpLov5TjgZkgPSDkuK3ntpnjgYznq4vjgaPjgabjgYTjgosgPSDlgZzmraLjgafjgY3jgovjgIIKICAgICAgLy8gYXZhaWxhYmxlPWZhbHNl77yIdXp1IOOBquOBqeOCr+OCqeODvOOCv+eEoeOBl++8ieOBp+OCguS4ree2meiHquS9k+OBr+eUn+OBjeOBpuOBhOOCi+OBruOBpwogICAgICAvLyDlgZzmraLjg5zjgr/jg7Pjga/lh7rjgZnjgIIKICAgICAgaWYgKHJlcy5vaykgewogICAgICAgIGNvbnN0IGQgPSBhd2FpdCByZXMuanNvbigpOwogICAgICAgIGlmIChkICYmIHR5cGVvZiBkLmF2YWlsYWJsZSA9PT0gJ2Jvb2xlYW4nKSB7CiAgICAgICAgICBlbHMuc3RvcFJvdy5zdHlsZS5kaXNwbGF5ID0gJyc7CiAgICAgICAgfQogICAgICB9CiAgICB9IGNhdGNoIChfKSB7IC8qIOS4ree2meOBp+OBquOBhOOBquOCieWHuuOBleOBquOBhCAqLyB9CiAgfQoKICBhc3luYyBmdW5jdGlvbiBzdG9wU2VydmVyKCkgewogICAgaWYgKCFjb25maXJtKCfjgrXjg7zjg5Djg7zjgpLlgZzmraLjgZfjgb7jgZnjgILjgojjgo3jgZfjgYTjgafjgZnjgYvvvJ9cbu+8iOOBk+OBrueUu+mdouOBr+S9v+OBiOOBquOBj+OBquOCiuOBvuOBme+8iScpKSByZXR1cm47CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgICAgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSB0cnVlOwogICAgICBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICflgZzmraLkuK3igKYnOwogICAgfQogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goYCR7bG9jYXRpb24ub3JpZ2lufS9zdG9wYCwgewogICAgICAgIG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzOiBhcGlIZWFkZXJzKHsgJ1gtS2MtU3RvcCc6ICcxJyB9KSwgY2FjaGU6ICduby1zdG9yZScsCiAgICAgIH0pOwogICAgICBpZiAocmVzLnN0YXR1cyA9PT0gNDAxKSB7CiAgICAgICAgb3BlbkF1dGhEaWFsb2coJ+WBnOatouOBq+OBr+iqjeiovOOBjOW/heimgeOBp+OBme+8iDQwMe+8iScpOwogICAgICAgIGlmIChlbHMuc3RvcFNlcnZlcikgeyBlbHMuc3RvcFNlcnZlci5kaXNhYmxlZCA9IGZhbHNlOyBlbHMuc3RvcFNlcnZlci50ZXh0Q29udGVudCA9ICfilqAg5YGc5q2iJzsgfQogICAgICAgIHJldHVybjsKICAgICAgfQogICAgICBpZiAoIXJlcy5vaykgewogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb5YGc5q2iXSDlpLHmlZfjgZfjgb7jgZfjgZ8gKEhUVFAgJHtyZXMuc3RhdHVzfSlgKTsKICAgICAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsgZWxzLnN0b3BTZXJ2ZXIuZGlzYWJsZWQgPSBmYWxzZTsgZWxzLnN0b3BTZXJ2ZXIudGV4dENvbnRlbnQgPSAn4pagIOWBnOatoic7IH0KICAgICAgICByZXR1cm47CiAgICAgIH0KICAgIH0gY2F0Y2ggKF8pIHsgLyog5Lit57aZ44GM6JC944Gh44KL44Gu44Gn6YCa5L+h44Ko44Op44O844Gv5oOz5a6a5YaFICovIH0KICAgIHNldFN0YXR1cygn5YGc5q2iJyk7CiAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCAnW+WBnOatol0g44K144O844OQ44O844KS5YGc5q2i44GX44G+44GX44Gf44CC44GT44Gu55S76Z2i44KS6ZaJ44GY44Gm44GP44Gg44GV44GE44CCJyk7CiAgICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIGVscy5zdG9wU2VydmVyLnRleHRDb250ZW50ID0gJ+WBnOatoua4iOOBvyc7CiAgfQoKICAvKiAtLS0tLS0tLS0tIEFQSSBCYXNlIHZhbGlkYXRpb24gKE1peGVkIGNvbnRlbnQgZ3VhcmQpIC0tLS0tLS0tLS0gKi8KICBmdW5jdGlvbiBpc01peGVkQ29udGVudEFwaUJhc2UoKXsKICAgIGNvbnN0IHBhZ2VQcm90byA9IHdpbmRvdy5sb2NhdGlvbi5wcm90b2NvbDsKICAgIGNvbnN0IGJhc2UgPSAoZWxzLmFwaUJhc2U/LnZhbHVlIHx8ICcnKS50cmltKCk7CiAgICByZXR1cm4gcGFnZVByb3RvID09PSAnaHR0cHM6JyAmJiAvXmh0dHA6XC9cLy9pLnRlc3QoYmFzZSk7CiAgfQogIGZ1bmN0aW9uIGVuc3VyZUFwaUJhc2VBbGxvd2VkKCl7CiAgICBpZiAoaXNNaXhlZENvbnRlbnRBcGlCYXNlKCkpewogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLAogICAgICAgIGBbQ29uZmlnXSBNaXhlZCBjb250ZW50IGJsb2NrZWQ6IHRoaXMgVUkgaXMgbG9hZGVkIHZpYSBIVFRQUywgYnV0IEFQSSBCYXNlIGlzIEhUVFAuXG5gICsKICAgICAgICBgQnJvd3NlcnMgd2lsbCB1c3VhbGx5IGJsb2NrIHRoaXMgcmVxdWVzdC4gVXNlIGFuIGh0dHBzOi8vIEFQSSBCYXNlLCBvciBvcGVuIHRoZSBVSSB2aWEgaHR0cDovLyBvciBmaWxlOi8vLmAKICAgICAgKTsKICAgICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcjYjQwMDAwJzsgfWNhdGNoe30KICAgICAgcmV0dXJuIGZhbHNlOwogICAgfQogICAgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fQogICAgcmV0dXJuIHRydWU7CiAgfQogIC8vIENsZWFyIHRoZSByZWQgYm9yZGVyIGFzIHNvb24gYXMgdGhlIHVzZXIgZWRpdHMgdGhlIGZpZWxkCiAgaWYgKGVscy5hcGlCYXNlKSB7CiAgICBlbHMuYXBpQmFzZS5hZGRFdmVudExpc3RlbmVyKCdpbnB1dCcsICgpID0+IHsgdHJ5eyBlbHMuYXBpQmFzZS5zdHlsZS5ib3JkZXJDb2xvciA9ICcnOyB9Y2F0Y2h7fSB9KTsKICB9CgoKICAvKiAtLS0tLS0tLS0tIE1vZGVscyAoR0VUIC92MS9tb2RlbHMpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBsb2FkTW9kZWxzKCkgewogICAgaWYgKCFlbnN1cmVBcGlCYXNlQWxsb3dlZCgpKSByZXR1cm47CiAgICBjb25zdCBiYXNlID0gQVBJX0JBU0UucmVwbGFjZSgvXC8kLywgJycpOwogICAgY29uc3QgdXJsID0gYCR7YmFzZX0vbW9kZWxzYDsKCiAgICBjb25zdCBoZWFkZXJzID0gYXBpSGVhZGVycyh7ICdBY2NlcHQnOiAnYXBwbGljYXRpb24vanNvbicgfSk7CgogICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0xvYWRpbmfigKYnOwogICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IHRydWU7CgogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ0dFVCcsIGhlYWRlcnMsIG1vZGU6ICdjb3JzJyB9KTsKICAgICAgaWYgKHJlcy5zdGF0dXMgPT09IDQwMSkgb3BlbkF1dGhEaWFsb2coJ0FQSSDoqo3oqLzjgYzlv4XopoHjgafjgZnvvIg0MDHvvIknKTsKICAgICAgaWYgKCFyZXMub2spIHsKICAgICAgICBjb25zdCB0ID0gYXdhaXQgcmVzLnRleHQoKS5jYXRjaCgoKT0+ICcnKTsKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdGYWlsZWQgdG8gbG9hZCBtb2RlbHMnfWApOwogICAgICB9CiAgICAgIGNvbnN0IGRhdGEgPSBhd2FpdCByZXMuanNvbigpOwoKICAgICAgLy8gT3BlbkFJLVN0eWxlOiB7IGRhdGE6IFt7aWQ6ICIuLi4ifSwgLi4uXSB9CiAgICAgIC8vIEZhbGxiYWNrcyBmw7xyIGFuZGVyZSBTZXJ2ZXI6CiAgICAgIGNvbnN0IGxpc3QgPQogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8uZGF0YSkgPyBkYXRhLmRhdGEgOgogICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YT8ubW9kZWxzKSA/IGRhdGEubW9kZWxzIDoKICAgICAgICBBcnJheS5pc0FycmF5KGRhdGEpID8gZGF0YSA6CiAgICAgICAgW107CgogICAgICBjb25zdCBpZHMgPSBsaXN0CiAgICAgICAgLm1hcChtID0+ICh0eXBlb2YgbSA9PT0gJ3N0cmluZycgPyBtIDogbT8uaWQpKQogICAgICAgIC5maWx0ZXIoQm9vbGVhbikKICAgICAgICAubWFwKFN0cmluZyk7CgogICAgICBpZiAoIWlkcy5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gbW9kZWxzIGZvdW5kIGluIC92MS9tb2RlbHMuJyk7CgogICAgICBpZHMuc29ydCgoYSxiKT0+YS5sb2NhbGVDb21wYXJlKGIpKTsKCiAgICAgIGNvbnN0IHByZXYgPSBlbHMubW9kZWwudmFsdWU7CiAgICAgIGVscy5tb2RlbC5pbm5lckhUTUwgPSAnJzsKICAgICAgZm9yIChjb25zdCBpZCBvZiBpZHMpIHsKICAgICAgICBjb25zdCBvcHQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdvcHRpb24nKTsKICAgICAgICBvcHQudmFsdWUgPSBpZDsKICAgICAgICBvcHQudGV4dENvbnRlbnQgPSBpZDsKICAgICAgICBlbHMubW9kZWwuYXBwZW5kQ2hpbGQob3B0KTsKICAgICAgfQoKICAgICAgLy8gQXVzd2FobDogdm9yaGVyaWdlIGJlaGFsdGVuIHdlbm4gbcO2Z2xpY2gsIHNvbnN0IGVyc3RlcwogICAgICBlbHMubW9kZWwudmFsdWUgPSBpZHMuaW5jbHVkZXMocHJldikgPyBwcmV2IDogaWRzWzBdOwoKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gYCR7aWRzLmxlbmd0aH0gbW9kZWxzYDsKICAgIH0gY2F0Y2ggKGUpIHsKICAgICAgaWYgKGVscy5tb2RlbFN0YXR1cykgZWxzLm1vZGVsU3RhdHVzLnRleHRDb250ZW50ID0gJ0Vycm9yJzsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjg6Ljg4fjg6voqq3jgb/ovrzjgb9dICR7ZS5tZXNzYWdlfWApOwogICAgfSBmaW5hbGx5IHsKICAgICAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5kaXNhYmxlZCA9IGZhbHNlOwogICAgfQogIH0KCiAgLyogLS0tLS0tLS0tLSBTZW5kIChVbnRlcnN0w7x0enQgPHRoaW5rPiB1bmQgInJlYXNvbmluZyIsIG1pdCBQZXJmb3JtYW5jZS1GaXgpIC0tLS0tLS0tLS0gKi8KICBhc3luYyBmdW5jdGlvbiBzZW5kTWVzc2FnZSgpIHsKICAgIGxldCBvdXRFbCA9IG51bGw7CiAgICBsZXQgYXNzaXN0YW50QnVmZmVyID0gJyc7CiAgICBsZXQgdGhpbmtCdWZmZXIgPSAnJzsKICAgIGxldCBzYXZlZFRleHQgPSAnJzsKICAgIGxldCBzYXZlZEF0dGFjaG1lbnRzID0gW107CiAgICB0cnkgewogICAgICBjb25zdCB0ZXh0ID0gZWxzLmlucHV0LnZhbHVlLnRyaW0oKTsKICAgICAgc2F2ZWRUZXh0ID0gdGV4dDsKICAgICAgc2F2ZWRBdHRhY2htZW50cyA9IFsuLi5hdHRhY2htZW50c107CiAgICAgIGlmICghdGV4dCAmJiBhdHRhY2htZW50cy5sZW5ndGggPT09IDApIHJldHVybjsKCiAgICAgIC8vIOOCueODiOODquODvOODn+ODs+OCsOS4reOBr+mAgeS/oeOBl+OBquOBhO+8iOWFpeWKm+OBr+WPr+iDve+8iQogICAgICBpZiAoIWVscy5zdG9wQnRuLmRpc2FibGVkKSByZXR1cm47CgogICAgICBpZiAoIWVuc3VyZUFwaUJhc2VBbGxvd2VkKCkpIHJldHVybjsKCiAgICAgIGlmICghZWxzLm1vZGVsLnZhbHVlKSB7CiAgICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgJ1vjg5Ljg7Pjg4hdIOODouODh+ODq+acqumBuOaKniDigJQg44Oi44OH44Or44GM6Kqt44G/6L6844G+44KM44Gm44GE44G+44Gb44KT44CC44Oa44O844K444KS44Oq44Ot44O844OJ44GX44Gm44GP44Gg44GV44GE44CCJyk7CiAgICAgICAgcmV0dXJuOwogICAgICB9CiAgICAgIAogICAgICBlbHMudG9rZW5SYXRlLnRleHRDb250ZW50ID0gJyc7CiAgICAgIGZpbmFsQ29tcGxldGlvblRva2VucyA9IDA7CiAgICAgIHN0cmVhbVN0YXJ0VGltZSA9IDA7CgogICAgICAvLyDjgrfjgrnjg4bjg6Djg5fjg63jg7Pjg5fjg4jjgpLmr47lm57lj43mmKAKICAgICAgY29uc3Qgc3lzVmFsID0gZWxzLnN5c3RlbS52YWx1ZS50cmltKCk7CiAgICAgIGlmIChzeXNWYWwpIHsKICAgICAgICBjb25zdCBleGlzdGluZ1N5c0lkeCA9IG1lc3NhZ2VzLmZpbmRJbmRleChtID0+IG0ucm9sZSA9PT0gJ3N5c3RlbScpOwogICAgICAgIGlmIChleGlzdGluZ1N5c0lkeCA+PSAwKSB7CiAgICAgICAgICBtZXNzYWdlc1tleGlzdGluZ1N5c0lkeF0uY29udGVudCA9IHN5c1ZhbDsKICAgICAgICB9IGVsc2UgaWYgKG1lc3NhZ2VzLmxlbmd0aCA9PT0gMCkgewogICAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICdzeXN0ZW0nLCBjb250ZW50OiBzeXNWYWwgfSk7CiAgICAgICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBzeXNWYWwpOwogICAgICAgIH0KICAgICAgfQogICAgICBjb25zdCBoYXNBdHQgPSBhdHRhY2htZW50cy5sZW5ndGggPiAwOwogICAgICBpZiAodGV4dCB8fCBoYXNBdHQpIHsKICAgICAgICBjb25zdCBwYXJ0cyA9IFtdOwogICAgICAgIGlmICh0ZXh0KSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQgfSk7CiAgICAgICAgZWxzZSBwYXJ0cy5wdXNoKHsgdHlwZTondGV4dCcsIHRleHQ6ICdBdHRhY2hlZCBmaWxlczonIH0pOwoKICAgICAgICBjb25zdCB0ZXh0RmlsZXMgPSBhdHRhY2htZW50cy5maWx0ZXIoYSA9PiBhLmtpbmQgPT09ICd0ZXh0Jyk7CiAgICAgICAgaWYgKHRleHRGaWxlcy5sZW5ndGgpewogICAgICAgICAgbGV0IGJsb2IgPSAnXG5cbi0tLVxuQXR0YWNoZWQgZmlsZXMgKHZlcmJhdGltKTpcbic7CiAgICAgICAgICBmb3IgKGNvbnN0IGEgb2YgdGV4dEZpbGVzKXsKICAgICAgICAgICAgYmxvYiArPSBgXG5bRmlsZTogJHthLm5hbWV9JHthLnRydW5jYXRlZD8nIChUUlVOQ0FURUQpJzonJ31dXG5cYFxgXGBcbiR7YS50ZXh0fVxuXGBcYFxgXG5gOwogICAgICAgICAgfQogICAgICAgICAgcGFydHMucHVzaCh7IHR5cGU6J3RleHQnLCB0ZXh0OiBibG9iIH0pOwogICAgICAgIH0KCiAgICAgICAgY29uc3QgaW1ncyA9IGF0dGFjaG1lbnRzLmZpbHRlcihhID0+IGEua2luZCA9PT0gJ2ltYWdlJyk7CiAgICAgICAgZm9yIChjb25zdCBhIG9mIGltZ3MpewogICAgICAgICAgaWYgKGEuZGF0YVVybCl7CiAgICAgICAgICAgIHBhcnRzLnB1c2goeyB0eXBlOidpbWFnZV91cmwnLCBpbWFnZV91cmw6eyB1cmw6IGEuZGF0YVVybCB9fSk7CiAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBjb25zdCBjb250ZW50ID0gKHBhcnRzLmxlbmd0aCA9PT0gMSkgPyAocGFydHNbMF0udGV4dCA/PyAnJykgOiBwYXJ0czsKCiAgICAgICAgbWVzc2FnZXMucHVzaCh7IHJvbGU6ICd1c2VyJywgY29udGVudCB9KTsKICAgICAgICBhZGRNZXNzYWdlKCd1c2VyJywgY29udGVudCk7CiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gJyc7CgogICAgICAgIGF0dGFjaG1lbnRzID0gW107CiAgICAgICAgcmVuZGVyQXR0YWNobWVudHMoKTsKICAgICAgfQoKCiAgICAgIG91dEVsID0gYWRkTWVzc2FnZSgnYXNzaXN0YW50JywgJycpOwoKICAgICAgLy8gTkVVOiBvcHRpb25hbGUgU2FtcGxpbmctUGFyYW1ldGVyIG51ciBtaXRzZW5kZW4sIHdlbm4gZ2VzZXR6dAogICAgICBmdW5jdGlvbiBwYXJzZU9wdGlvbmFsRmxvYXQoZWwsIG1pbiwgbWF4KSB7CiAgICAgICAgY29uc3QgdiA9IChlbD8udmFsdWUgPz8gJycpLnRyaW0oKTsKICAgICAgICBpZiAodiA9PT0gJycpIHJldHVybiBudWxsOwogICAgICAgIGNvbnN0IG51bSA9IHBhcnNlRmxvYXQodik7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1heCA9PT0gJ251bWJlcicgJiYgbnVtID4gbWF4KSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGZ1bmN0aW9uIHBhcnNlT3B0aW9uYWxJbnQoZWwsIG1pbikgewogICAgICAgIGNvbnN0IHYgPSAoZWw/LnZhbHVlID8/ICcnKS50cmltKCk7CiAgICAgICAgaWYgKHYgPT09ICcnKSByZXR1cm4gbnVsbDsKICAgICAgICBjb25zdCBudW0gPSBwYXJzZUludCh2LCAxMCk7CiAgICAgICAgaWYgKE51bWJlci5pc05hTihudW0pKSByZXR1cm4gbnVsbDsKICAgICAgICBpZiAodHlwZW9mIG1pbiA9PT0gJ251bWJlcicgJiYgbnVtIDwgbWluKSByZXR1cm4gbnVsbDsKICAgICAgICByZXR1cm4gbnVtOwogICAgICB9CiAgICAgIGNvbnN0IGV4dHJhcyA9IHt9OwogICAgICBjb25zdCBfdG9wX3AgPSBwYXJzZU9wdGlvbmFsRmxvYXQoZWxzLnRvcF9wLCAwLCAxKTsKICAgICAgY29uc3QgX21pbl9wID0gcGFyc2VPcHRpb25hbEZsb2F0KGVscy5taW5fcCwgMCwgMSk7CiAgICAgIGNvbnN0IF90b3BfayA9IHBhcnNlT3B0aW9uYWxJbnQoZWxzLnRvcF9rLCAwKTsKICAgICAgaWYgKF90b3BfcCAhPT0gbnVsbCkgZXh0cmFzLnRvcF9wID0gX3RvcF9wOwogICAgICBpZiAoX21pbl9wICE9PSBudWxsKSBleHRyYXMubWluX3AgPSBfbWluX3A7CiAgICAgIGlmIChfdG9wX2sgIT09IG51bGwpIGV4dHJhcy50b3BfayA9IF90b3BfazsKCiAgICAgIC8vIHRoaW5raW5nOiB1enUgcmVhZHMgcmVhc29uaW5nX2VmZm9ydCBwZXIgcmVxdWVzdC4gT3RoZXIgYmFja2VuZHMgaWdub3JlIGl0LgogICAgICBjb25zdCBfdGhpbmtpbmcgPSAoZWxzLnRoaW5raW5nPy52YWx1ZSA/PyAnJykudHJpbSgpOwogICAgICBpZiAoX3RoaW5raW5nID09PSAnb2ZmJykgZXh0cmFzLnJlYXNvbmluZ19lZmZvcnQgPSAnZGlzYWJsZWQnOwogICAgICBlbHNlIGlmIChfdGhpbmtpbmcgPT09ICdvbicpIGV4dHJhcy5yZWFzb25pbmdfZWZmb3J0ID0gJ2RlZmF1bHQnOwoKICAgICAgY29uc3QgYm9keSA9IHsgbW9kZWw6IGVscy5tb2RlbC52YWx1ZS50cmltKCksIG1lc3NhZ2VzLCBzdHJlYW06IHRydWUsIHRlbXBlcmF0dXJlOiBwYXJzZUZsb2F0KGVscy50ZW1wLnZhbHVlKSwgLi4uZXh0cmFzIH07CiAgICAgIGNvbnN0IGhlYWRlcnMgPSBhcGlIZWFkZXJzKHsgJ0NvbnRlbnQtVHlwZSc6ICdhcHBsaWNhdGlvbi9qc29uJywgJ0FjY2VwdCc6ICd0ZXh0L2V2ZW50LXN0cmVhbScgfSk7CiAgICAgIGNvbnN0IHVybCA9IGAke0FQSV9CQVNFLnJlcGxhY2UoL1wvJC8sICcnKX0vY2hhdC9jb21wbGV0aW9uc2A7CgogICAgICBhYm9ydENvbnRyb2xsZXIgPSBuZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKHRydWUpOwoKICAgICAgbGV0IGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICBsZXQgdGhpbmtFbCA9IG51bGw7CgogICAgICAvLyAtLS0gU1RBUlQgw4ROREVSVU5HOiBDT1JTIGhpbnp1Z2Vmw7xndCAtLS0KICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2godXJsLCB7IG1ldGhvZDogJ1BPU1QnLCBoZWFkZXJzLCBib2R5OiBKU09OLnN0cmluZ2lmeShib2R5KSwgc2lnbmFsOiBhYm9ydENvbnRyb2xsZXIuc2lnbmFsLCBtb2RlOiAnY29ycycgfSk7CiAgICAgIC8vIC0tLSBFTkRFIMOETkRFUlVORyAtLS0KCiAgICAgIGlmIChyZXMuc3RhdHVzID09PSA0MDEpIG9wZW5BdXRoRGlhbG9nKCdBUEkg6KqN6Ki844GM5b+F6KaB44Gn44GZ77yINDAx77yJJyk7CiAgICAgIGlmICghcmVzLm9rKSB7IGNvbnN0IHQgPSBhd2FpdCByZXMudGV4dCgpLmNhdGNoKCgpPT4gJycpOyB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXMuc3RhdHVzfTogJHt0IHx8ICdTZXJ2ZXIgZXJyb3InfWApOyB9CiAgICAgIAogICAgICBjb25zdCBjdHlwZSA9IChyZXMuaGVhZGVycy5nZXQoJ2NvbnRlbnQtdHlwZScpIHx8ICcnKS50b0xvd2VyQ2FzZSgpOwogICAgICBpZiAoIWN0eXBlLmluY2x1ZGVzKCd0ZXh0L2V2ZW50LXN0cmVhbScpKSB7CiAgICAgICAgY29uc3QgcmVxdWVzdFN0YXJ0VGltZSA9IHBlcmZvcm1hbmNlLm5vdygpOwogICAgICAgIGNvbnN0IHJhdyA9IGF3YWl0IHJlcy50ZXh0KCk7IAogICAgICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0VtcHR5IHJlc3BvbnNlIChzdGF0dXMgT0ssIGJ1dCBubyBjb250ZW50KS4nKTsKICAgICAgICAKICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSByZXF1ZXN0U3RhcnRUaW1lOwogICAgICAgIGxldCBkYXRhOyB0cnl7IGRhdGEgPSBKU09OLnBhcnNlKHJhdyk7fWNhdGNoeyB0aHJvdyBuZXcgRXJyb3IoYFVuZXJ3YXJ0ZXRlcyBGb3JtYXQ6ICR7cmF3LnNsaWNlKDAsMjAwKX0uLi5gKTsgfQogICAgICAgIAogICAgICAgIGxldCBmdWxsQ29udGVudCA9IGRhdGE/LmNob2ljZXM/LlswXT8ubWVzc2FnZT8uY29udGVudCA/PyAnJzsKICAgICAgICBjb25zdCByZWFzb25pbmdDb250ZW50ID0gZGF0YT8uY2hvaWNlcz8uWzBdPy5tZXNzYWdlPy5yZWFzb25pbmdfY29udGVudDsKCiAgICAgICAgaWYgKHJlYXNvbmluZ0NvbnRlbnQpIHsKICAgICAgICAgICAgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgcmVhc29uaW5nQ29udGVudCk7CiAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gcmVhc29uaW5nQ29udGVudDsKICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQudHJpbSgpOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIC8vIEZhbGxiYWNrIGF1ZiA8dGhpbms+LVRhZy1WZXJhcmJlaXR1bmcKICAgICAgICAgICAgY29uc3QgdGhpbmtNYXRjaCA9IGZ1bGxDb250ZW50Lm1hdGNoKC88dGhpbms+KFtcc1xTXSo/KTxcL3RoaW5rPi8pOwogICAgICAgICAgICBpZiAodGhpbmtNYXRjaCkgewogICAgICAgICAgICAgICAgY29uc3QgdGhpbmtUZXh0ID0gdGhpbmtNYXRjaFsxXS50cmltKCk7CiAgICAgICAgICAgICAgICB0aGlua0VsID0gYWRkVGhpbmtCdWJibGUob3V0RWwpOwogICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtUZXh0KTsKICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyID0gdGhpbmtUZXh0OwogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQucmVwbGFjZSh0aGlua01hdGNoWzBdLCAnJykudHJpbSgpOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgYXNzaXN0YW50QnVmZmVyID0gZnVsbENvbnRlbnQ7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgCiAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudChvdXRFbCwgJ2Fzc2lzdGFudCcsIGFzc2lzdGFudEJ1ZmZlcik7CiAgICAgICAgCiAgICAgICAgZmluYWxDb21wbGV0aW9uVG9rZW5zID0gZGF0YT8udXNhZ2U/LmNvbXBsZXRpb25fdG9rZW5zIHx8IDA7CiAgICAgICAgdXBkYXRlVG9rZW5SYXRlKGFzc2lzdGFudEJ1ZmZlciArIHRoaW5rQnVmZmVyLCB0cnVlKTsKICAgICAgICAKICAgICAgICBtYXliZVNjcm9sbCgpOwogICAgICB9IGVsc2UgewogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBTVEFSVCAtLS0KICAgICAgICBsZXQgcmVuZGVyU2NoZWR1bGVkID0gZmFsc2U7CiAgICAgICAgbGV0IGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgbGV0IHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICBjb25zdCBSRU5ERVJfSU5URVJWQUwgPSAxNTA7IAoKICAgICAgICBjb25zdCBzY2hlZHVsZVJlbmRlciA9ICgpID0+IHsKICAgICAgICAgICAgaWYgKHJlbmRlclNjaGVkdWxlZCkgcmV0dXJuOwogICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSB0cnVlOwogICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgICAgICAgICAgIGlmIChhc3Npc3RhbnROZWVkc1JlbmRlcikgewogICAgICAgICAgICAgICAgICAgIHJlbmRlckJ1YmJsZUNvbnRlbnQob3V0RWwsICdhc3Npc3RhbnQnLCBhc3Npc3RhbnRCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gZmFsc2U7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICBpZiAodGhpbmtOZWVkc1JlbmRlciAmJiB0aGlua0VsKSB7CiAgICAgICAgICAgICAgICAgICAgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSBmYWxzZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgICAgICAgICAgICByZW5kZXJTY2hlZHVsZWQgPSBmYWxzZTsKICAgICAgICAgICAgfSwgUkVOREVSX0lOVEVSVkFMKTsKICAgICAgICB9OwogICAgICAgIC8vIC0tLSBPUFRJTUlFUlVORyBFTkRFIC0tLQoKICAgICAgICBjb25zdCByZWFkZXIgPSByZXMuYm9keS5nZXRSZWFkZXIoKTsgY29uc3QgZGVjb2RlciA9IG5ldyBUZXh0RGVjb2RlcigndXRmLTgnKTsgbGV0IGJ1ZmZlcj0nJzsgbGV0IGRvbmU9ZmFsc2U7CiAgICAgICAgd2hpbGUgKCFkb25lKSB7CiAgICAgICAgICBjb25zdCB7IHZhbHVlLCBkb25lOiByZERvbmUgfSA9IGF3YWl0IHJlYWRlci5yZWFkKCk7IGRvbmUgPSByZERvbmU7CgogICAgICAgICAgaWYgKHN0cmVhbVN0YXJ0VGltZSA9PT0gMCAmJiB2YWx1ZSkgewogICAgICAgICAgICBzdHJlYW1TdGFydFRpbWUgPSBwZXJmb3JtYW5jZS5ub3coKTsKICAgICAgICAgIH0KCiAgICAgICAgICBidWZmZXIgKz0gZGVjb2Rlci5kZWNvZGUodmFsdWUgfHwgbmV3IFVpbnQ4QXJyYXkoKSwgeyBzdHJlYW06ICFkb25lIH0pOwogICAgICAgICAgbGV0IGV2ZW50cyA9IGJ1ZmZlci5zcGxpdCgnXG5cbicpOyBidWZmZXIgPSBldmVudHMucG9wKCkgfHwgJyc7CiAgICAgICAgICBmb3IgKGNvbnN0IGV2IG9mIGV2ZW50cykgewogICAgICAgICAgICBjb25zdCBsaW5lcyA9IGV2LnNwbGl0KCdcbicpLmZpbHRlcihCb29sZWFuKTsKICAgICAgICAgICAgZm9yIChjb25zdCBsaW5lIG9mIGxpbmVzKSB7CiAgICAgICAgICAgICAgaWYgKCFsaW5lLnN0YXJ0c1dpdGgoJ2RhdGE6JykpIGNvbnRpbnVlOwogICAgICAgICAgICAgIGNvbnN0IGRhdGFTdHIgPSBsaW5lLnNsaWNlKDUpLnRyaW0oKTsgaWYgKGRhdGFTdHIgPT09ICdbRE9ORV0nKSB7IGRvbmUgPSB0cnVlOyBicmVhazsgfQogICAgICAgICAgICAgIHRyeSB7CiAgICAgICAgICAgICAgICBjb25zdCBvYmogPSBKU09OLnBhcnNlKGRhdGFTdHIpOwogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBpZiAob2JqPy51c2FnZT8uY29tcGxldGlvbl90b2tlbnMpIHsKICAgICAgICAgICAgICAgICAgICBmaW5hbENvbXBsZXRpb25Ub2tlbnMgPSBvYmoudXNhZ2UuY29tcGxldGlvbl90b2tlbnM7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGRlbHRhID0gb2JqPy5jaG9pY2VzPy5bMF0/LmRlbHRhOwogICAgICAgICAgICAgICAgaWYgKCFkZWx0YSkgY29udGludWU7CgogICAgICAgICAgICAgICAgbGV0IGhhc1VwZGF0ZWQgPSBmYWxzZTsKICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgY29uc3QgcmVhc29uaW5nQ2h1bmsgPSBkZWx0YS5yZWFzb25pbmcgfHwgZGVsdGEucmVhc29uaW5nX2NvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChyZWFzb25pbmdDaHVuaykgewogICAgICAgICAgICAgICAgICAgIGlmICghdGhpbmtFbCkgdGhpbmtFbCA9IGFkZFRoaW5rQnViYmxlKG91dEVsKTsKICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZWFzb25pbmdDaHVuazsKICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGNvbnN0IGNvbnRlbnRDaHVuayA9IGRlbHRhLmNvbnRlbnQgfHwgJyc7CiAgICAgICAgICAgICAgICBpZiAoY29udGVudENodW5rKSB7CiAgICAgICAgICAgICAgICAgICAgbGV0IHRlbXBDaHVuayA9IGNvbnRlbnRDaHVuazsKICAgICAgICAgICAgICAgICAgICB3aGlsZSAodGVtcENodW5rLmxlbmd0aCA+IDApIHsKICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluVGhpbmtCbG9jaykgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zID0gdGVtcENodW5rLmluZGV4T2YoJzwvdGhpbms+Jyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rQnVmZmVyICs9IHRlbXBDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZW1wQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKGVuZFBvcyArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluVGhpbmtCbG9jayA9IGZhbHNlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaW5rTmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IHN0YXJ0UG9zID0gdGVtcENodW5rLmluZGV4T2YoJzx0aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChzdGFydFBvcyAhPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rLnN1YnN0cmluZygwLCBzdGFydFBvcyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYXNzaXN0YW50TmVlZHNSZW5kZXIgPSB0cnVlOyAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoIXRoaW5rRWwpIHRoaW5rRWwgPSBhZGRUaGlua0J1YmJsZShvdXRFbCk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgcmVzdE9mQ2h1bmsgPSB0ZW1wQ2h1bmsuc3Vic3RyaW5nKHN0YXJ0UG9zICsgNyk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZW5kUG9zSW5SZXN0ID0gcmVzdE9mQ2h1bmsuaW5kZXhPZignPC90aGluaz4nKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoZW5kUG9zSW5SZXN0ICE9PSAtMSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua0J1ZmZlciArPSByZXN0T2ZDaHVuay5zdWJzdHJpbmcoMCwgZW5kUG9zSW5SZXN0KTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gcmVzdE9mQ2h1bmsuc3Vic3RyaW5nKGVuZFBvc0luUmVzdCArIDgpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpbmtCdWZmZXIgKz0gcmVzdE9mQ2h1bms7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBDaHVuayA9ICcnOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpblRoaW5rQmxvY2sgPSB0cnVlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlua05lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhc3Npc3RhbnRCdWZmZXIgKz0gdGVtcENodW5rOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFzc2lzdGFudE5lZWRzUmVuZGVyID0gdHJ1ZTsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGVtcENodW5rID0gJyc7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgaGFzVXBkYXRlZCA9IHRydWU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGlmIChoYXNVcGRhdGVkKSB7CiAgICAgICAgICAgICAgICAgIHNjaGVkdWxlUmVuZGVyKCk7CiAgICAgICAgICAgICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIpOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgIH0gY2F0Y2gge30KICAgICAgICAgICAgfQogICAgICAgICAgfQogICAgICAgIH0KICAgICAgICAvLyBGaW5hbGVzIFJlbmRlcmluZyBhbSBFbmRlIGRlcyBTdHJlYW1zLCB1bSBzaWNoZXJ6dXN0ZWxsZW4sIGRhc3MgYWxsZXMgYW5nZXplaWd0IHdpcmQKICAgICAgICByZW5kZXJCdWJibGVDb250ZW50KG91dEVsLCAnYXNzaXN0YW50JywgYXNzaXN0YW50QnVmZmVyKTsKICAgICAgICBpZiAodGhpbmtFbCkgcmVuZGVyQnViYmxlQ29udGVudCh0aGlua0VsLCAnc3lzdGVtJywgdGhpbmtCdWZmZXIpOwogICAgICAgIAogICAgICAgIGlmIChzdHJlYW1TdGFydFRpbWUgPiAwKSB1cGRhdGVUb2tlblJhdGUoYXNzaXN0YW50QnVmZmVyICsgdGhpbmtCdWZmZXIsIHRydWUpOwogICAgICAgIAogICAgICAgIG1heWJlU2Nyb2xsKCk7CiAgICAgIH0KICAgICAgCiAgICAgIGFzc2lzdGFudEJ1ZmZlciA9IGFzc2lzdGFudEJ1ZmZlci50cmltKCk7CiAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyIHx8ICcoZW1wdHkgcmVzcG9uc2UpJyB9KTsKICAgICAgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgaWYgKGVycj8ubmFtZSA9PT0gJ0Fib3J0RXJyb3InKSB7CiAgICAgICAgLy8gU3RvcOODnOOCv+ODszog6YCU5Lit44G+44Gn44Gu5Zue562U44KS5bGl5q2044Gr5q6L44GZCiAgICAgICAgaWYgKGFzc2lzdGFudEJ1ZmZlci50cmltKCkpIHsKICAgICAgICAgIG1lc3NhZ2VzLnB1c2goeyByb2xlOiAnYXNzaXN0YW50JywgY29udGVudDogYXNzaXN0YW50QnVmZmVyLnRyaW0oKSB9KTsKICAgICAgICB9CiAgICAgIH0gZWxzZSB7CiAgICAgICAgLy8g6YCa5L+h44Ko44Op44O8OiDnm7TliY3jga51c2Vy44Oh44OD44K744O844K444KS5bGl5q2044GL44KJ5oi744GZCiAgICAgICAgaWYgKG91dEVsICYmICFvdXRFbC50ZXh0Q29udGVudC50cmltKCkpIHsgb3V0RWwuY2xvc2VzdCgnLm1zZycpPy5yZW1vdmUoKTsgfQogICAgICAgIGlmIChtZXNzYWdlcy5sZW5ndGggPiAwICYmIG1lc3NhZ2VzW21lc3NhZ2VzLmxlbmd0aC0xXS5yb2xlID09PSAndXNlcicpIHsKICAgICAgICAgIG1lc3NhZ2VzLnBvcCgpOwogICAgICAgIH0KICAgICAgICAvLyDlhaXlipvjg4bjgq3jgrnjg4jjgpLlvqnlhYPvvIjmt7vku5jmmYLjga9jb250ZW5044GM6YWN5YiX44Gr44Gq44KL44Gf44KBc2F2ZWRUZXh044KS5L2/44GG77yJCiAgICAgICAgZWxzLmlucHV0LnZhbHVlID0gc2F2ZWRUZXh0OwogICAgICAgIC8vIOa3u+S7mOODleOCoeOCpOODq+OCguW+qeWFgwogICAgICAgIGlmIChzYXZlZEF0dGFjaG1lbnRzLmxlbmd0aCA+IDApIHsKICAgICAgICAgIGF0dGFjaG1lbnRzID0gc2F2ZWRBdHRhY2htZW50czsKICAgICAgICAgIHJlbmRlckF0dGFjaG1lbnRzKCk7CiAgICAgICAgfQogICAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsIGBb44Ko44Op44O8XSAke2Vyci5tZXNzYWdlfWApOyBjb25zb2xlLmVycm9yKGVycik7CiAgICAgIH0KICAgIH0gZmluYWxseSB7CiAgICAgIHNldFN0cmVhbWluZ1N0YXRlKGZhbHNlKTsgYWJvcnRDb250cm9sbGVyID0gbnVsbDsKICAgIH0KICB9CgogIGZ1bmN0aW9uIHN0b3BTdHJlYW0oKXsgaWYgKGFib3J0Q29udHJvbGxlcikgYWJvcnRDb250cm9sbGVyLmFib3J0KCk7IH0KICBmdW5jdGlvbiBjbGVhckNoYXQoKXsgbWVzc2FnZXMgPSBbXTsgZWxzLmxvZy5pbm5lckhUTUwgPSAnJzsgZWxzLnRva2VuUmF0ZS50ZXh0Q29udGVudCA9ICcnOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgbWF5YmVTY3JvbGwodHJ1ZSk7IH0KCiAgLyogLS0tLS0tLS0tLSBJbXBvcnQgLyBFeHBvcnQgLS0tLS0tLS0tLSAqLwogIGZ1bmN0aW9uIGV4cG9ydENoYXQoKXsKICAgIGNvbnN0IG91dCA9IGJ1aWxkRXhwb3J0QXJyYXkoKTsKICAgIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbSlNPTi5zdHJpbmdpZnkob3V0LCBudWxsLCAyKV0sIHsgdHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nIH0pOwogICAgY29uc3QgdXJsID0gVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTsKICAgIGNvbnN0IGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdhJyk7IGNvbnN0IHRzID0gbmV3IERhdGUoKTsgY29uc3QgcGFkID0gbj0+U3RyaW5nKG4pLnBhZFN0YXJ0KDIsJzAnKTsKICAgIGEuaHJlZiA9IHVybDsgYS5kb3dubG9hZCA9IGBjaGF0XyR7dHMuZ2V0RnVsbFllYXIoKX0tJHtwYWQodHMuZ2V0TW9udGgoKSsxKX0tJHtwYWQodHMuZ2V0RGF0ZSgpKX1fJHtwYWQodHMuZ2V0SG91cnMoKSl9LSR7cGFkKHRzLmdldE1pbnV0ZXMoKSl9LSR7cGFkKHRzLmdldFNlY29uZHMoKSl9Lmpzb25gOwogICAgYS5jbGljaygpOyBzZXRUaW1lb3V0KCgpPT5VUkwucmV2b2tlT2JqZWN0VVJMKHVybCksIDEwMDApOwogIH0KCiAgYXN5bmMgZnVuY3Rpb24gcGFyc2VJbXBvcnQodGV4dCl7CiAgICBjb25zdCBjbGVhbmVkID0gdGV4dC5yZXBsYWNlKC9cci9nLCcnKTsKICAgIGxldCByYXc7IHRyeSB7IHJhdyA9IEpTT04ucGFyc2UoY2xlYW5lZCk7IH0KICAgIGNhdGNoIHsgY29uc3QgbGluZXMgPSBjbGVhbmVkLnNwbGl0KCdcbicpLm1hcChzPT5zLnRyaW0oKSkuZmlsdGVyKEJvb2xlYW4pOyBjb25zdCBhcnI9W107IGZvciAoY29uc3QgbGluZSBvZiBsaW5lcyl7IHRyeXsgYXJyLnB1c2goSlNPTi5wYXJzZShsaW5lKSk7IH1jYXRjaHt9IH0gcmF3ID0gYXJyLmxlbmd0aCA/IGFyciA6IG51bGw7IH0KICAgIGlmICghcmF3KSB0aHJvdyBuZXcgRXJyb3IoJ0ZpbGUgbm90IHJlYWRhYmxlIChuZWl0aGVyIEpTT04gbm9yIEpTT05MKS4nKTsKCiAgICBsZXQgYXJyID0gW10sIHRvcFN5c3RlbSA9ICcnOwogICAgaWYgKEFycmF5LmlzQXJyYXkocmF3KSkgYXJyID0gcmF3OwogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcubWVzc2FnZXMpKSB7IGFyciA9IHJhdy5tZXNzYWdlczsgdG9wU3lzdGVtID0gU3RyaW5nKHJhdy5zeXN0ZW0gfHwgJycpLnRyaW0oKTsgfQogICAgZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyYXcuY29udmVyc2F0aW9uKSkgeyBhcnIgPSByYXcuY29udmVyc2F0aW9uOyB0b3BTeXN0ZW0gPSBTdHJpbmcocmF3LnN5c3RlbSB8fCAnJykudHJpbSgpOyB9CiAgICBlbHNlIHRocm93IG5ldyBFcnJvcignTm8gbWVzc2FnZSBhcnJheSBmb3VuZCAoZXhwZWN0ZWQ6IFsge3JvbGUsIGNvbnRlbnR9LCAuLi4gXSkuJyk7CgogICAgY29uc3Qgbm9ybWFsaXplZCA9IGFyci5tYXAobSA9PiAoeyByb2xlOiBub3JtYWxpemVSb2xlKG0ucm9sZSksIGNvbnRlbnQ6IG5vcm1hbGl6ZUNvbnRlbnQobS5jb250ZW50KSB9KSkKICAgICAgICAgICAgICAgICAgICAgICAgICAuZmlsdGVyKG0gPT4gWydzeXN0ZW0nLCd1c2VyJywnYXNzaXN0YW50J10uaW5jbHVkZXMobS5yb2xlKSAmJiB0eXBlb2YgbS5jb250ZW50ID09PSAnc3RyaW5nJyk7CgogICAgY29uc3QgaGFzU3lzdGVtID0gbm9ybWFsaXplZC5zb21lKG0gPT4gbS5yb2xlID09PSAnc3lzdGVtJyk7CiAgICBpZiAoIWhhc1N5c3RlbSAmJiB0b3BTeXN0ZW0pIG5vcm1hbGl6ZWQudW5zaGlmdCh7IHJvbGU6J3N5c3RlbScsIGNvbnRlbnQ6IHRvcFN5c3RlbSB9KTsKICAgIGlmICghbm9ybWFsaXplZC5sZW5ndGgpIHRocm93IG5ldyBFcnJvcignTm8gdmFsaWQgbWVzc2FnZXMgZm91bmQgaW4gaW1wb3J0LicpOwogICAgcmV0dXJuIG5vcm1hbGl6ZWQ7CiAgfQoKICBmdW5jdGlvbiBoYW5kbGVGaWxlKGZpbGUpewogICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgIHJlYWRlci5vbmxvYWQgPSAoKSA9PiB7CiAgICAgIHBhcnNlSW1wb3J0KFN0cmluZyhyZWFkZXIucmVzdWx0IHx8ICcnKSkKICAgICAgICAudGhlbihhcnIgPT4geyBtZXNzYWdlcyA9IGFycjsgY29uc3Qgc3lzID0gbWVzc2FnZXMuZmluZChtPT5tLnJvbGU9PT0nc3lzdGVtJyk7IGVscy5zeXN0ZW0udmFsdWUgPSBzeXM/c3lzLmNvbnRlbnQ6Jyc7IHJlbmRlckFsbCgpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBg44OB44Oj44OD44OI44KS44Kk44Oz44Od44O844OI44GX44G+44GX44GfICgke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXMpLmApOyBpZiAoZWxzLnJhd0JveC5vcGVuKSBlZGl0b3JEdW1wRnJvbUNoYXQoKTsgfSkKICAgICAgICAuY2F0Y2goZSA9PiBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW+OCpOODs+ODneODvOODiOOCqOODqeODvF0gJHtlLm1lc3NhZ2V9YCkpOwogICAgfTsKICAgIHJlYWRlci5vbmVycm9yID0gKCkgPT4gYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvjgqTjg7Pjg53jg7zjg4jjgqjjg6njg7xdIENvdWxkIG5vdCByZWFkIGZpbGUuYCk7CiAgICByZWFkZXIucmVhZEFzVGV4dChmaWxlKTsKICB9CiAgaWYgKGVscy5kcm9wKSB7CiAgICBbJ2RyYWdlbnRlcicsJ2RyYWdvdmVyJ10uZm9yRWFjaChldiA9PiBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKGV2LCBlID0+IHsgZS5wcmV2ZW50RGVmYXVsdCgpOyBlbHMuZHJvcC5jbGFzc0xpc3QuYWRkKCdkcmFnJyk7IH0pKTsKICAgIFsnZHJhZ2xlYXZlJywnZHJvcCddLmZvckVhY2goZXYgPT4gZWxzLmRyb3AuYWRkRXZlbnRMaXN0ZW5lcihldiwgZSA9PiB7IGUucHJldmVudERlZmF1bHQoKTsgZWxzLmRyb3AuY2xhc3NMaXN0LnJlbW92ZSgnZHJhZycpOyB9KSk7CiAgICBlbHMuZHJvcC5hZGRFdmVudExpc3RlbmVyKCdkcm9wJywgKGUpID0+IHsgY29uc3QgZiA9IGUuZGF0YVRyYW5zZmVyPy5maWxlcz8uWzBdOyBpZiAoZikgaGFuZGxlRmlsZShmKTsgfSk7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFJBVyBKU09OIEVkaXRvciAtLS0tLS0tLS0tICovCiAgZnVuY3Rpb24gZWRpdG9yU2V0U3RhdHVzKG1zZywgb2s9dHJ1ZSl7IGVscy5qc29uU3RhdHVzLnRleHRDb250ZW50ID0gbXNnIHx8ICcnOyBlbHMuanNvbkVkaXRvci5zdHlsZS5ib3JkZXJDb2xvciA9IG9rID8gJ3ZhcigtLWxpbmUpJyA6ICcjYjQwMDAwJzsgfQogIGZ1bmN0aW9uIGVkaXRvckR1bXBGcm9tQ2hhdCgpeyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KGJ1aWxkRXhwb3J0QXJyYXkoKSwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnbG9hZGVkIGZyb20gY2hhdCcpOyB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUGFyc2VDdXJyZW50KCl7CiAgICBjb25zdCB0ZXh0ID0gZWxzLmpzb25FZGl0b3IudmFsdWUudHJpbSgpIHx8ICdbXSc7CiAgICBjb25zdCBhcnIgPSBhd2FpdCBwYXJzZUltcG9ydCh0ZXh0KTsKICAgIGNvbnN0IGZpcnN0U3lzID0gYXJyLmZpbmQobT0+bS5yb2xlPT09J3N5c3RlbScpOyBpZiAoZmlyc3RTeXMpIGVscy5zeXN0ZW0udmFsdWUgPSBmaXJzdFN5cy5jb250ZW50OwogICAgcmV0dXJuIGFycjsKICB9CiAgYXN5bmMgZnVuY3Rpb24gZWRpdG9yUmVwbGFjZSgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IGFycjsKICAgICAgcmVuZGVyQWxsKCk7CiAgICAgIGlzUmVuZGVyaW5nID0gZmFsc2U7CiAgICAgIGF1dG9zY3JvbGwgPSB0cnVlOwogICAgICBlZGl0b3JTZXRTdGF0dXMoYGFwcGxpZWQgKHJlcGxhY2UpIOKAlCAke21lc3NhZ2VzLmxlbmd0aH0gbWVzc2FnZXNgKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYEpTT07jgYvjgonjg4Hjg6Pjg4Pjg4jjgpLoqK3lrprjgZfjgb7jgZfjgZ8gKCR7bWVzc2FnZXMubGVuZ3RofSlgKTsKICAgIH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cyhlLm1lc3NhZ2UsZmFsc2UpOyBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgW0pTT07jgqjjg6njg7xdICR7ZS5tZXNzYWdlfWApOyB9CiAgfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvckFwcGVuZCgpewogICAgdHJ5ewogICAgICBjb25zdCBhcnIgPSBhd2FpdCBlZGl0b3JQYXJzZUN1cnJlbnQoKTsKICAgICAgaXNSZW5kZXJpbmcgPSB0cnVlOwogICAgICBtZXNzYWdlcyA9IG1lc3NhZ2VzLmNvbmNhdChhcnIpOwogICAgICByZW5kZXJBbGwoKTsKICAgICAgaXNSZW5kZXJpbmcgPSBmYWxzZTsKICAgICAgYXV0b3Njcm9sbCA9IHRydWU7CiAgICAgIGVkaXRvclNldFN0YXR1cyhgYXBwbGllZCAoYXBwZW5kKSDigJQgbm93ICR7bWVzc2FnZXMubGVuZ3RofSBtZXNzYWdlc2ApOwogICAgICBhZGRNZXNzYWdlKCdzeXN0ZW0nLCBgSlNPTuOBi+OCieODgeODo+ODg+ODiOOCkui/veiomOOBl+OBvuOBl+OBnyAoKyR7YXJyLmxlbmd0aH0pYCk7CiAgICB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLGZhbHNlKTsgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFtKU09O44Ko44Op44O8XSAke2UubWVzc2FnZX1gKTsgfQogIH0KICBmdW5jdGlvbiBlZGl0b3JCZWF1dGlmeSgpeyB0cnkgeyBjb25zdCBvYmogPSBKU09OLnBhcnNlKGVscy5qc29uRWRpdG9yLnZhbHVlIHx8ICdbXScpOyBlbHMuanNvbkVkaXRvci52YWx1ZSA9IEpTT04uc3RyaW5naWZ5KG9iaiwgbnVsbCwgMik7IGVkaXRvclNldFN0YXR1cygnYmVhdXRpZmllZCcpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoJ2ludmFsaWQgSlNPTicsIGZhbHNlKTt9IH0KICBmdW5jdGlvbiBlZGl0b3JNaW5pZnkoKXsgdHJ5IHsgY29uc3Qgb2JqID0gSlNPTi5wYXJzZShlbHMuanNvbkVkaXRvci52YWx1ZSB8fCAnW10nKTsgZWxzLmpzb25FZGl0b3IudmFsdWUgPSBKU09OLnN0cmluZ2lmeShvYmopOyBlZGl0b3JTZXRTdGF0dXMoJ21pbmlmaWVkJyk7IH0gY2F0Y2goZSl7IGVkaXRvclNldFN0YXR1cygnaW52YWxpZCBKU09OJywgZmFsc2UpO30gfQogIGFzeW5jIGZ1bmN0aW9uIGVkaXRvclZhbGlkYXRlKCl7IHRyeSB7IGF3YWl0IGVkaXRvclBhcnNlQ3VycmVudCgpOyBlZGl0b3JTZXRTdGF0dXMoJ3ZhbGlkIOKckycpOyB9IGNhdGNoKGUpeyBlZGl0b3JTZXRTdGF0dXMoZS5tZXNzYWdlLCBmYWxzZSk7IH0gfQoKICBlbHMucmF3Qm94LmFkZEV2ZW50TGlzdGVuZXIoJ3RvZ2dsZScsICgpID0+IHsgaWYgKGVscy5yYXdCb3gub3BlbikgZWRpdG9yRHVtcEZyb21DaGF0KCk7IH0pOwogIGVscy5qc29uUmVmcmVzaC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckR1bXBGcm9tQ2hhdCk7CiAgZWxzLmpzb25BcHBseS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclJlcGxhY2UpOwogIGVscy5qc29uQXBwZW5kLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgZWRpdG9yQXBwZW5kKTsKICBlbHMuanNvblByZXR0eS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvckJlYXV0aWZ5KTsKICBlbHMuanNvbk1pbmlmeS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvck1pbmlmeSk7CiAgZWxzLmpzb25WYWxpZGF0ZS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGVkaXRvclZhbGlkYXRlKTsKCiAgLyogLS0tLS0tLS0tLSBXaXJlIFVJIC0tLS0tLS0tLS0gKi8KICBlbHMuc2VuZEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHNlbmRNZXNzYWdlKTsKICBlbHMuc3RvcEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIHN0b3BTdHJlYW0pOwogIGVscy5jbGVhckJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGNsZWFyQ2hhdCk7CiAgaWYgKGVscy5hdHRhY2hCdG4gJiYgZWxzLmF0dGFjaElucHV0KSB7CiAgICBlbHMuYXR0YWNoQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCk9PiBlbHMuYXR0YWNoSW5wdXQuY2xpY2soKSk7CiAgICBlbHMuYXR0YWNoSW5wdXQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKCk9PnsKICAgICAgYWRkQXR0YWNobWVudHNGcm9tRmlsZUxpc3QoZWxzLmF0dGFjaElucHV0LmZpbGVzKTsKICAgICAgZWxzLmF0dGFjaElucHV0LnZhbHVlID0gJyc7CiAgICB9KTsKICB9CiAgaWYgKGVscy5sb2FkTW9kZWxzQnRuKSBlbHMubG9hZE1vZGVsc0J0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGxvYWRNb2RlbHMpOwogIGVscy5pbnB1dC5hZGRFdmVudExpc3RlbmVyKCdrZXlkb3duJywgKGUpID0+IHsKICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJyAmJiAhZS5zaGlmdEtleSAmJiAhZS5pc0NvbXBvc2luZyAmJiBlLmtleUNvZGUgIT09IDIyOSkgewogICAgICAvLyDjgrnjg4jjg6rjg7zjg5/jg7PjgrDkuK3jga/pgIHkv6HjgZvjgZrjgIHmlLnooYzjgpLoqLHlj68KICAgICAgaWYgKCFlbHMuc3RvcEJ0bi5kaXNhYmxlZCkgcmV0dXJuOwogICAgICBlLnByZXZlbnREZWZhdWx0KCk7CiAgICAgIHNlbmRNZXNzYWdlKCk7CiAgICB9CiAgfSk7CiAgZWxzLnRlbXAuYWRkRXZlbnRMaXN0ZW5lcignaW5wdXQnLCAoKSA9PiB7IGVscy50ZW1wVmFsLnRleHRDb250ZW50ID0gcGFyc2VGbG9hdChlbHMudGVtcC52YWx1ZSkudG9GaXhlZCgxKTsgfSk7CiAgZWxzLnNldHRpbmdzVG9nZ2xlLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgZWxzLnNldHRpbmdzUGFuZWwuY2xhc3NMaXN0LnRvZ2dsZSgnaGlkZGVuJyk7CiAgfSk7CgogIC8qIC0tLS0tLS0tLS0gSW5mbyBkaWFsb2cgLS0tLS0tLS0tLSAqLwogIGlmIChlbHMuaW5mb1ZlcnNpb24pIGVscy5pbmZvVmVyc2lvbi50ZXh0Q29udGVudCA9IEFQUF9WRVJTSU9OOwogIGlmIChlbHMuaW5mb1Byb2plY3RVcmwpIHsKICAgIGVscy5pbmZvUHJvamVjdFVybC50ZXh0Q29udGVudCA9IFBST0pFQ1RfVVJMOwogICAgZWxzLmluZm9Qcm9qZWN0VXJsLmhyZWYgPSBQUk9KRUNUX1VSTDsKICB9CiAgaWYgKGVscy5pbmZvSW1wcmludCkgewogICAgZWxzLmluZm9JbXByaW50LnRleHRDb250ZW50ID0gSU1QUklOVF9VUkw7CiAgICBlbHMuaW5mb0ltcHJpbnQuaHJlZiA9IElNUFJJTlRfVVJMOwogIH0KICBpZiAoZWxzLmluZm9CdG4gJiYgZWxzLmluZm9EaWFsb2cpIHsKICAgIGVscy5pbmZvQnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgICBpZiAodHlwZW9mIGVscy5pbmZvRGlhbG9nLnNob3dNb2RhbCA9PT0gJ2Z1bmN0aW9uJykgZWxzLmluZm9EaWFsb2cuc2hvd01vZGFsKCk7CiAgICAgIGVsc2UgZWxzLmluZm9EaWFsb2cuc2V0QXR0cmlidXRlKCdvcGVuJywgJycpOwogICAgfSk7CiAgICAvLyBDbGljayBvdXRzaWRlIHRoZSBjYXJkIGNsb3NlcyB0aGUgZGlhbG9nCiAgICBlbHMuaW5mb0RpYWxvZy5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIChlKSA9PiB7CiAgICAgIC8vIElmIGNsaWNrIGxhbmRzIG9uIHRoZSBiYWNrZHJvcCAobm90IG9uIHRoZSBjYXJkKSwgY2xvc2UuCiAgICAgIGlmIChlLnRhcmdldCA9PT0gZWxzLmluZm9EaWFsb2cpIGVscy5pbmZvRGlhbG9nLmNsb3NlKCk7CiAgICB9KTsKICB9CgogIC8qIC0tLS0tLS0tLS0gQVBJIOiqjeiovOODgOOCpOOCouODreOCsCAtLS0tLS0tLS0tICovCiAgaWYgKGVscy5hdXRoQnRuICYmIGVscy5hdXRoRGlhbG9nKSB7CiAgICBlbHMuYXV0aEJ0bi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IG9wZW5BdXRoRGlhbG9nKCkpOwogIH0KICBpZiAoZWxzLmF1dGhTYXZlKSB7CiAgICBlbHMuYXV0aFNhdmUuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7CiAgICAgIGNvbnN0IHVzZXIgPSAoZWxzLmF1dGhVc2VyPy52YWx1ZSB8fCAnJykudHJpbSgpIHx8ICdleGwzJzsKICAgICAgY29uc3QgcGFzcyA9IGVscy5hdXRoUGFzcz8udmFsdWUgfHwgJyc7CiAgICAgIGlmICghcGFzcykgewogICAgICAgIGlmIChlbHMuYXV0aFN0YXR1cykgZWxzLmF1dGhTdGF0dXMudGV4dENvbnRlbnQgPSAn44OR44K544Ov44O844OJ44KS5YWl5Yqb44GX44Gm44GP44Gg44GV44GEJzsKICAgICAgICByZXR1cm47CiAgICAgIH0KICAgICAgc2V0QXV0aENyZWRlbnRpYWxzKHVzZXIsIHBhc3MpOwogICAgICBjbG9zZUF1dGhEaWFsb2coKTsKICAgICAgYWRkTWVzc2FnZSgnc3lzdGVtJywgYFvoqo3oqLxdICR7dXNlcn0g44Gn5L+d5a2Y44GX44G+44GX44Gf77yI44GT44Gu44K/44OW44Gu44G/77yJYCk7CiAgICAgIGxvYWRNb2RlbHMoKTsKICAgICAgc3RhcnRRdW90YVBvbGxpbmcoKTsKICAgICAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhDbGVhcikgewogICAgZWxzLmF1dGhDbGVhci5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHsKICAgICAgY2xlYXJBdXRoQ3JlZGVudGlhbHMoKTsKICAgICAgY2xvc2VBdXRoRGlhbG9nKCk7CiAgICAgIGFkZE1lc3NhZ2UoJ3N5c3RlbScsICdb6KqN6Ki8XSDos4fmoLzmg4XloLHjgpLop6PpmaTjgZfjgb7jgZfjgZ8nKTsKICAgIH0pOwogIH0KICBpZiAoZWxzLmF1dGhQYXNzKSB7CiAgICBlbHMuYXV0aFBhc3MuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIChlKSA9PiB7CiAgICAgIGlmIChlLmtleSA9PT0gJ0VudGVyJykgeyBlLnByZXZlbnREZWZhdWx0KCk7IGVscy5hdXRoU2F2ZT8uY2xpY2soKTsgfQogICAgfSk7CiAgfQogIGlmIChlbHMuYXV0aERpYWxvZykgewogICAgZWxzLmF1dGhEaWFsb2cuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4gewogICAgICBpZiAoZS50YXJnZXQgPT09IGVscy5hdXRoRGlhbG9nKSBjbG9zZUF1dGhEaWFsb2coKTsKICAgIH0pOwogIH0KCiAgLyogLS0tLS0tLS0tLSDmrovjgormmYLplpMgLyDlgZzmraLjg5zjgr/jg7MgLS0tLS0tLS0tLSAqLwogIHN0YXJ0UXVvdGFQb2xsaW5nKCk7CiAgZGV0ZWN0U3RvcFN1cHBvcnQoKTsKICBpZiAoZWxzLnN0b3BTZXJ2ZXIpIHsKICAgIGVscy5zdG9wU2VydmVyLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgc3RvcFNlcnZlcik7CiAgfQoKICAvKiAtLS0tLS0tLS0tIFdlbGNvbWUgLS0tLS0tLS0tLSAqLwogIHNldFRpbWVvdXQoKCkgPT4geyBsb2FkTW9kZWxzKCk7IH0sIDUwMCk7Cn0pKCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K'))
SECRET_USER = "API_USER"
SECRET_PASS = "API_PASS"

# ── 段の有無（エンジン差はここだけ）──────────────────────────
INSTALL = True              # stage_install() を呼ぶか
NEEDS_MODEL = False      # find_model() を要求するか
TP_REQUIRED = True      # GPU が TP 枚必要か
KEEPALIVE_SCENARIO = False   # keepalive 実測シナリオを回すか
PUBLISH = True              # 公開 URL を Gist へ書くか
VERIFY_PUBLIC = True  # 公開 URL 経由の疎通も見るか
CONF_DRIVEN = True      # 停止判断を keepalive.conf に従うか
SERVE = True                  # 配信カーネルか（False = 単発のバッチ処理）
WARMUP = True                # warmup() を呼ぶか
CLEANUP = False              # cleanup() を停止時に呼ぶか

data = {"engine": ENGINE, "stage": "init", "tunnel": TUNNEL_KIND, "model": MODEL_NAME, "context_length": CONTEXT_LENGTH, "smoke": SMOKE}
T0 = time.time()
# キープアライブの毎周期の控え。`save_evidence()` が keepalive.txt に落とす。
# result.json に入れないのは、数百行になって成果物が読みにくくなるため。
KEEPALIVE_LINES = []
# 設定セルが TUNNEL_KIND を名乗っていても、実装が選んだ方を正とする
# （設定と実装がずれると result.json の tunnel が嘘になる）。
data["tunnel"] = TUNNEL


def save_evidence():
    """配信の証跡を書き出す（notebook 版の keepalive.txt / cleanup.txt 相当）。

    **毎回書き直す**（追記にしない）。`save()` は段の変わり目に何度も呼ばれる
    ので、追記だと同じ行が何度も入る。最後に呼ばれた時点の内容が正になる。
    """
    if KEEPALIVE_LINES:
        dead = data.get("tunnel_dead", 0)
        head = [
            f"result={'DEAD' if dead >= 2 else data.get('stopped_by', '')}",
            f"stopped_by={data.get('stopped_by')}",
            f"cycles={len(KEEPALIVE_LINES)}",
            f"tunnel={TUNNEL}",
            f"tunnel_dead={dead}",
            f"tunnel_restarted={data.get('tunnel_restarted', False)}",
            f"url={data.get('url', '')}",
            "",
        ]
        try:
            with open(KEEPALIVE_LOG, "w") as f:
                f.write("\n".join(head + KEEPALIVE_LINES) + "\n")
        except OSError as e:
            print(f"⚠ keepalive.txt を書けない: {type(e).__name__}: {e}")
    if SERVE:
        try:
            with open(CLEANUP_LOG, "w") as f:
                f.write(f"engine_cleanup={data.get('engine_cleanup', 'skipped')}\n"
                        f"leftover={','.join(data.get('stop_remaining') or []) or 'none'}\n")
        except OSError as e:
            print(f"⚠ cleanup.txt を書けない: {type(e).__name__}: {e}")


def save():
    data["elapsed_seconds"] = round(time.time() - T0, 1)
    # セル側（vLLM の install / serve など）が書いた計測を合流させる。
    # セルから `data` を直接触らないのは、**notebook 版には `data` が無い**ため
    # （セル順の「定義前使用」で落ちる。2026-11-01 実測）。ファイル経由なら
    # notebook / script の両方で同じセルが動く。
    try:
        with open(TIMINGS, encoding="utf-8") as _f:
            _t = json.load(_f)
        if _t:
            data["timings"] = {**(data.get("timings") or {}), **_t}
    except (OSError, ValueError):
        pass
    with open(RESULT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    save_evidence()


def sh(cmd, timeout=1800, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired as e:
        class _R:
            returncode = -1
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + "\n[timeout]"
        return _R(), round(time.time() - t0, 1)


def tail(s, n=30):
    return "\n".join((s or "").strip().splitlines()[-n:])


def http_json(url, payload=None, auth=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def wait_http(url, timeout, auth=None, interval=5, grace=60):
    """ready になるまで待つ。

    エンジンが**死んだら待ち続けない**。実測（2026-10-31 lmdeploy）: turbomind が
    重み 67% で CUDA OOM して 6 分で終了したのに、待機側は 40 分待ち続け、
    1 往復で 46 分を捨てた。プロセスの消滅で諦めれば 10 分で次の手を打てる。
    起動直後は exec 前で pgrep が空になるので、`grace` 秒までは判定しない。
    """
    t0 = time.time()
    last = None
    gone = 0
    while time.time() - t0 < timeout:
        try:
            code, _ = http_json(url, auth=auth, timeout=10)
            if code == 200:
                return round(time.time() - t0, 1)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if time.time() - t0 > grace:
            try:
                gone = 0 if kc_pids(ENGINE_PATTERN, timeout=5) else gone + 1
            except Exception:
                gone = 0          # 判定できないときは待つ側に倒す
            if gone >= 3:         # 3 回連続（≒15 秒）で「死んだ」と判断する
                raise RuntimeError(
                    f"エンジンが終了した（{ENGINE_PATTERN} に一致するプロセスが無い）"
                    f" {url}（{last}）")
        time.sleep(interval)
    raise RuntimeError(f"待機タイムアウト {url}（{last}）")


def secret(name):
    """Secrets を読む（env 優先。Kaggle では kaggle_secrets からも取れる）。"""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


# ── 1. 環境 ─────────────────────────────────────────────────
def stage_env():
    print("=" * 60)
    print(f"{ENGINE} serve ({MODE})")
    print("=" * 60)
    data["python"] = sys.version.split()[0]
    print(f"python : {sys.version.split()[0]}")
    names, mems = kc_detect_gpu(timeout=15)
    data["gpu"] = {"names": names, "memory_mib": mems}
    if names:
        text = "\n".join(f"{n}, {m} MiB" for n, m in zip(names, mems))
        print(text)
        data["nvidia_smi"] = text
    else:
        # CPU カーネル（乾燥走行）には nvidia-smi が無い
        print("nvidia-smi: なし（CPU ホスト）")
        data["nvidia_smi"] = None
    # torch は**子プロセス**で調べる。親で import してはいけない。
    # vLLM はインストール段で torch を差し替える（イメージ 2.11 → 2.13）ので、
    # 親が先に import していると「メモリ上の旧 torch」と「ディスク上の新 torch」が
    # 混ざる。すると torch 自身のファイル同士が食い違い、
    #   TypeError: Config() got an unexpected keyword argument 'deprecated'
    # という無関係に見える失敗になる（実測 2026-10-31。子プロセスでは成功するのに
    # 親だけ落ちる、という形で 8 往復溶かした）。CUDA 文脈の初期化も避けたい。
    try:
        _r = subprocess.run(
            [sys.executable, "-c",
             "import json, torch; print(json.dumps({'v': torch.__version__,"
             " 'c': torch.version.cuda, 'a': torch.cuda.is_available(),"
             " 'n': torch.cuda.device_count()}))"],
            capture_output=True, text=True, timeout=180)
        if _r.returncode != 0:
            _err = (_r.stderr or "").strip().splitlines()
            raise RuntimeError(_err[-1] if _err else "torch の確認に失敗")
        _t = json.loads((_r.stdout or "").strip().splitlines()[-1])
        data.update(torch=_t["v"], torch_cuda=_t["c"],
                    cuda_available=_t["a"], device_count=_t["n"])
        print(f"torch  : {_t['v']} / cuda {_t['c']}")
    except Exception as e:
        data["torch_error"] = f"{type(e).__name__}: {e}"
    data["stage"] = "env"
    save()
    return names


# ── 2. モデル ───────────────────────────────────────────────
def find_model():
    """量子化モデルのディレクトリを返す（見つからなければ None）。

    量子化の持ち方は 2 通りある。**`quantization_config.json` を別ファイルで置く形**と、
    **`config.json` の中に `quantization_config` を埋め込む形**。以前は前者しか
    受け付けていなかったため、後者だけの AWQ Dataset を取り逃していた
    （実測 2026-10-31: lmdeploy / sglang が `モデル Dataset が見つからない` で即死。
    Dataset には config.json と model.safetensors があるのに、判定だけが落ちていた）。
    形式を推測するより**重みの実在**で判定する方が確実。
    """
    for pat in ("/kaggle/input/datasets/*/*/config.json",
                "/kaggle/input/*/config.json"):
        for cfg in sorted(glob.glob(pat)):
            d = os.path.dirname(cfg)
            if glob.glob(os.path.join(d, "*.safetensors")):
                return d
            if os.path.isfile(os.path.join(d, "quantization_config.json")):
                return d
    return None


# ── 3. 補助ファイル展開 ─────────────────────────────────────
def write_aux():
    os.makedirs(WORK, exist_ok=True)
    for name, b64 in AUX_FILES:
        with open(os.path.join(WORK, name), "wb") as f:
            f.write(base64.b64decode(b64))


# ── 4. 認証プロキシ ─────────────────────────────────────────
def start_proxy(engine_pid):
    user = secret(SECRET_USER) or ENGINE
    pass_secret = secret(SECRET_PASS)
    # パスワードは result.json / ログに出さない（kernels output で誰でも
    # 取得できてしまうため）。未設定時は乱数で起動するが、値はどこにも残さない。
    pass_ = pass_secret or os.urandom(8).hex()
    auth = "Basic " + base64.b64encode(f"{user}:{pass_}".encode()).decode()
    env = {**os.environ, "API_USER": user, "API_PASS": pass_}
    p = kc_start_proxy(
        PROXY_PORT, f"http://127.0.0.1:{PORT}",
        f"{WORK}/llama-chat.html", f"{WORK}/auth_proxy.py",
        extra_args=("--stop-pid", str(engine_pid)) if engine_pid else (),
        env=env, log_path="/kaggle/working/auth_proxy.log",
        wait_seconds=60, label="auth_proxy", python=sys.executable)
    data["auth"] = "on"
    data["auth_user"] = user
    if not pass_secret:
        data["auth_warning"] = f"{SECRET_PASS} 未設定（乱数。値は非公開）"
        print(f"{SECRET_PASS} 未設定: 乱数パスワードで起動（値は残さない）")
    print(f"auth_proxy 起動: 127.0.0.1:{PROXY_PORT}")
    return p, auth


def engine_checks(judge, retry, base, auth=None, direct=False):
    """既定の疎通確認。エンジン固有の項目は CHECKS_CODE スロットが上書きする。

    本文が空なら失敗として扱う（無言の空応答を成功にしない）。

    `direct` は `verify()` が**必ず位置引数で渡す**ので、既定でも受け取る必要がある。
    ここを `auth=None` で止めていたため、`checks_code` を持たないエンジン
    （exl3）が `engine_checks() takes from 3 to 4 positional arguments but 5 were
    given` で落ちた（2026-11-01）。既定の 2 項目は proxy を挟んでも直叩きでも
    同じように成立するので、`direct` は受け取るだけで分岐には使わない。
    """
    code, body = http_json(f"{base}/v1/models", auth=auth)
    data["models"] = body
    judge(0, "models", code == 200, f"HTTP {code}")

    payload = {"model": ALIAS,
               "messages": [{"role": "user", "content": "1+1は？"}],
               "max_tokens": 16, "temperature": 0}
    code, body = http_json(f"{base}/v1/chat/completions", payload,
                           auth=auth, timeout=900)
    text = (body.get("choices", [{}])[0].get("message", {})
            .get("content") or "")
    data["chat"] = {"status": code, "text": text[:200],
                    "usage": body.get("usage") or {}}
    print(f"疎通: {code} / {text[:80]!r}")
    judge(1, "chat", code == 200 and bool(text.strip()),
          f"HTTP {code} {text[:60]!r}")


def verify(auth, base, key="verify", direct=False):
    """疎通確認の共通の足場。判定項目は engine_checks() が持つ。

    1 項目でも落ちたら raise する（検証不能のまま配信を続けない）。

    `direct=True` は「proxy を挟まずエンジンを直叩きしている」の意味。
    smoke はトンネルも proxy も立てないので、proxy が返すもの（チャット UI）を
    検査する項目は成立しない。engine_checks() 側がその項目を SKIP する。
    """
    lines, failures = [], []

    def judge(idx, label, ok, detail):
        # ok=None は「この条件では判定できない」= SKIP。失敗には数えないが、
        # PASS とも書かない（後から見て「通った」と誤読しないようにする）。
        if ok is None:
            print(f"⏭ {label}: {detail}")
            lines.append(f"[{idx}] {label} -> SKIP: {detail}")
            return
        print(f"{'✅' if ok else '❌'} {label}: {detail}")
        lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
        if not ok:
            failures.append(f"[{idx}] {label}: {detail}")

    def retry(fn, tries=6, delay=5):
        # 瞬間的な DNS / 経路断を吸収する。HTTP 応答は「答え」なので再試行しない。
        return kc_retry(fn, tries=tries, delay=delay)

    t0 = time.time()
    engine_checks(judge, retry, base, auth, direct)
    data[key] = {"ok": not failures, "checks": lines,
                 "wall_seconds": round(time.time() - t0, 1)}
    if failures:
        raise RuntimeError(f"{key}: " + "; ".join(failures))


def verify_public(auth, url, attempts=3):
    """公開 URL 経由の疎通。失敗しても配信は続ける。

    Kaggle コンテナ → trycloudflare エッジの経路が時々 ENETUNREACH に
    なる（同じ URL の GET が直前に成功していても）。クライアントは外側の
    ブラウザなので、ここでの失敗はセッションを止める理由にならない。
    """
    last = None
    for i in range(attempts):
        try:
            verify(auth, url, key="public_verify")
            data["public_verify_ok"] = True
            return
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            print(f"公開疎通 失敗 ({i + 1}/{attempts}): {last}")
            time.sleep(5)
    data["public_verify_ok"] = False
    data["public_verify_error"] = last


# ── 5. トンネル / 公開 URL ────────────────────────────────
def find_cloudflared():
    for pat in ("/kaggle/input/*/cloudflared",
                "/kaggle/input/*/*/cloudflared",
                "/kaggle/input/datasets/*/*/cloudflared"):
        for p in sorted(glob.glob(pat)):
            if os.path.isfile(p):
                return p
    return None


def _wait_tunnel(proc, host, log_path, seconds, key="tunnel_seconds"):
    """公開 URL が出るまで待つ。落ちていたら理由を添えて失敗させる。"""
    t0 = time.time()
    url = kc_wait_url(host, log_path, seconds=seconds)
    if not url:
        if proc.poll() is not None:
            raise RuntimeError(f"{host} 終了 ({proc.returncode})")
        raise RuntimeError(f"{host} の URL を取得できなかった")
    data[key] = round(time.time() - t0, 1)
    print(f"公開 URL ({host}): {url}")
    return proc, url


def start_cloudflared(key="tunnel_seconds"):
    cf = find_cloudflared()
    dst = f"{WORK}/cloudflared"
    if cf:
        # /kaggle/input は読み取り専用で実行ビットも付けられないため、
        # 書き込み可能な WORK へコピーしてから chmod する。
        shutil.copy2(cf, dst)
        os.chmod(dst, 0o755)
        data["cloudflared_source"] = f"{cf} (copied)"
    else:
        urllib.request.urlretrieve(kc_cloudflared_url(), dst)
        os.chmod(dst, 0o755)
        data["cloudflared_source"] = "download"

    log_path = "/kaggle/working/cloudflared.log"
    with open(log_path, "w") as log:
        p = subprocess.Popen([dst, "tunnel", "--url",
                              f"http://127.0.0.1:{PROXY_PORT}", "--no-autoupdate"],
                             stdout=log, stderr=subprocess.STDOUT)
    return _wait_tunnel(p, "cloudflared", log_path, 120, key=key)


def start_tunnel():
    """TUNNEL に従ってトンネルを張り、`(proc, url)` を返す。

    選ぶ先は今 cloudflared しか無い。ngrok は 2026-11-01 に廃止した。
    理由は 2 つで、どちらも実測:
      - `NGROK_TOKEN` を Secrets から読むが、専用アカウントでは未紐付けの
        `get_secret()` が `ConnectionError` を返す（`notes/url-publishing.md`）。
        つまりビーコンで URL を配っても、その手前で起動できない
      - 固定ドメインを維持する理由がビーコンで消えた（URL が毎回変わっても
        `python3 -m cli beacon` で読める）
    cloudflared の quick tunnel はアカウントも authtoken も要らない。

    分岐を残しているのは、`TUNNEL` が骨格（builder）の環境変数から来る値で、
    ここだけがその値を見る場所だから。値を増やすときは `_wait_tunnel` の
    host 名と `data["tunnel"]` の整合も一緒に直すこと。
    """
    if TUNNEL == "cloudflared":
        return start_cloudflared()
    raise RuntimeError(f"TUNNEL={TUNNEL!r} は未対応（cloudflared のみ）")


def publish_url(url):
    """公開 URL を url.txt / Gist / ビーコンに書く。**失敗しても配信は続ける。**

    3 経路あるのは移行途中だから:
      - `url.txt` … 同じカーネルの中から読む用（従来どおり）
      - Gist      … 個人アカウント時代の読み手。Secrets が無ければスキップ
      - ビーコン  … カーネルのコードを掲示板にする方式。**Secrets 不要**なので
                    専用アカウント運用でも使える（`notes/url-publishing.md`）

    Gist の分岐で早期 return しないこと。以前ここが `return` だったので、
    Secrets が無い環境ではビーコンまで届かなかった。
    """
    with open(URL_FILE, "w") as f:
        f.write(url + "\n")
    _publish_gist(url)
    _publish_beacon(url)


def _publish_gist(url):
    gist_id = (secret("GIST_ID") or GIST_ID_DEFAULT or "").strip()
    token = (secret("GITHUB_TOKEN") or "").strip()
    if not gist_id or not token:
        data["publish"] = "skipped (GIST_ID / GITHUB_TOKEN なし)"
        print("Gist 公開: スキップ（GIST_ID / GITHUB_TOKEN なし）")
        return
    body = json.dumps({"files": {"url.txt": {"content": url + "\n"}}}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}", data=body, method="PATCH",
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": f"{ENGINE}-serve-kernel"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data["publish"] = f"gist ok ({r.status})"
        print("Gist 公開: " + "https://gist.githubusercontent.com/"
              f"{GIST_OWNER}/{gist_id}/raw/url.txt")
    except Exception as e:
        data["publish"] = f"gist failed: {type(e).__name__}: {e}"
        print(f"Gist 公開 失敗（続行）: {e}")


def _publish_beacon(url):
    """ビーコン（掲示板カーネル）を更新する。成否は `data["beacon"]` に残す。"""
    slug = f"{GIST_OWNER}/{BEACON_NAME}" if GIST_OWNER else ""
    ok, why = kc_beacon_publish(url, stage="serving", slug=slug)
    data["beacon"] = why
    print(("ビーコン公開: " + why) if ok else ("ビーコン公開 失敗（続行）: " + why))


def wait_public(url):
    try:
        dt = kc_wait_http(url + "/healthz", attempts=24, interval=5)
        if dt is None:
            raise RuntimeError("公開 /healthz タイムアウト")
        data["public_health_seconds"] = dt
        print(f"公開 /healthz OK: {dt}s")
    except Exception as e:
        data["public_health_error"] = f"{type(e).__name__}: {e}"
        print(f"公開 /healthz 失敗（続行）: {e}")


# ── 6. キープアライブ / 時間予算 ────────────────────────────
def read_conf():
    # 上限は HARD_MAX_MINUTES 側で掛けるので、ここでは conf をクランプしない。
    return kc_read_conf(CONF, RUN_MINUTES, 10 ** 9,
                        on_invalid="zero", quiet_clamp=True)


def _public_health(url):
    """公開 URL の `/healthz` を 1 回叩く。**例外を投げない**（戻り値で返す）。"""
    try:
        with kc_urlopen_v4(url + "/healthz", timeout=15) as r:
            return r.status == 200, f"HTTP{r.status}"
    except Exception as e:
        return False, type(e).__name__


def restart_tunnel(procs):
    """落ちたトンネルを **1 回だけ** 張り直し、新しい URL を返す（駄目なら ""）。

    notebook 版（`kaggle/serve-cells/keepalive.py`）と同じ制御。URL が変わるので
    `publish_url` まで呼ぶ。呼ばないと url.txt / Gist / ビーコンが失効 URL を
    指したままになり、**外からは「動いているのに繋がらない」**状態が続く
    （`notes/serve-parity.md`）。

    再起動は 1 回だけ。2 回目以降は呼び出し側が `_exited` として終わらせる
    （同じ原因で落ち続けるなら、起動を繰り返しても URL は生き残らない）。
    """
    try:
        subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
        time.sleep(1)
        proc, new_url = start_cloudflared(key="tunnel_restart_seconds")
    except Exception as e:
        data["tunnel_restart_error"] = f"{type(e).__name__}: {e}"
        print(f"⚠ {TUNNEL} 再起動失敗: {type(e).__name__}: {e}")
        return ""
    # ここから先は失敗しても「トンネルは生きている」。公開の失敗で
    # 張り直しを無かったことにしない（URL は生きているのに捨てることになる）。
    procs[TUNNEL] = proc
    data["tunnel_restarted"] = True
    try:
        publish_url(new_url)
    except Exception as e:
        data["publish_error"] = f"{type(e).__name__}: {e}"
        print(f"⚠ 新 URL の公開に失敗（配信は続く）: {type(e).__name__}: {e}")
    print(f"♻ {TUNNEL} 再起動 → 新URL: {new_url}")
    return new_url


def loop(procs, url=""):
    """キープアライブ。**notebook 版と同じ監視を持つ**。

    以前はエンジンの `poll()` しか見ておらず、quick tunnel の URL が途中で
    失効しても `result.json` は `ok: true` のまま外から繋がらない状態で
    走り続けた。いまは notebook 版（`serve-cells/keepalive.py`）と同じく

      - トンネルのプロセスが死んだら **1 回だけ**張り直して URL を再公開
      - 10 分ごとに `/healthz` を叩き、失敗を数える（2 回で失敗扱い）
      - 毎周期の状態を `KEEPALIVE_LINES` に残す（`save_evidence()` が保存）

    を回す。`url` が空（トンネルを張らない経路）なら疎通確認はしない。
    """
    started = time.time()
    restarted = 0
    dead = 0
    restarted_now = False
    n = 0
    print(f"🔋 キープアライブ開始（{LOOP_SLEEP}s 周期 / 公開URL確認は "
          f"{TUNNEL_CHECK_EVERY * LOOP_SLEEP // 60} 分ごと）")
    while True:
        n += 1
        if CONF_DRIVEN:
            run_flag, minutes, mtime = read_conf()
            if run_flag == 0:
                return "stop"
            if kc_keepalive_expired(time.time(), started, minutes, mtime,
                                    HARD_MAX_MINUTES):
                return "budget"
        elif time.time() - started >= RUN_MINUTES * 60:
            # 乾燥走行は配管テストなので keepalive を見ない（生成リクエストが
            # conf を書き換えて予算が延びるため）。
            return "budget"

        # トンネルは先に見る。プロセスが生きていても URL は失効しうるが、
        # 落ちている場合は張り直せる（他は張り直せないので終了）。
        t = procs.get(TUNNEL)
        if t is not None and t.poll() is not None:
            if restarted:
                return f"{TUNNEL}_exited({t.returncode})"
            restarted = 1
            new_url = restart_tunnel(procs)
            if not new_url:
                return f"{TUNNEL}_exited({t.returncode})"
            url = new_url
            # セッションの現在 URL は loop が持つ。`save_evidence()` の
            # keepalive.txt も `data["url"]` を読むので、ここで必ず更新する
            # （失効 URL が成果物に残ると、次に読む人がそれを叩いて混乱する）。
            data["url"] = url
            dead = 0                 # 別勘定。旧カウントを引き継がない
            restarted_now = True     # 新 URL のエッジ伝播待ち中は判定しない

        for name, p in procs.items():
            if name == TUNNEL:
                continue             # 上で見た
            if p.poll() is not None:
                return f"{name}_exited({p.returncode})"

        line = (f"[{time.strftime('%H:%M:%S')}] "
                f"tunnel={TUNNEL}:"
                f"{kc_proc_state(TUNNEL, 'exact')} | GPU: {kc_gpu_usage()}")
        # pid だけでは tunnel の死は分からない（quick tunnel は URL 失効がありうる）。
        # 張り直した直後は新 URL のエッジ伝播待ちなので判定しない。
        if url and n % TUNNEL_CHECK_EVERY == 0 and not restarted_now:
            ok, detail = _public_health(url)
            if ok:
                line += " | tunnel=OK"
            else:
                dead += 1
                line += f" | tunnel=DEAD({detail}) x{dead}"
                if not restarted:
                    line += " ← プロセスは生存。URL だけ失効の可能性"
                print(f"⚠ 公開 /healthz 失敗 {dead} 回目: {detail}")
        restarted_now = False
        KEEPALIVE_LINES.append(line)
        data["tunnel_dead"] = dead
        print(line)
        time.sleep(LOOP_SLEEP)


def run_keepalive_scenario(auth, procs):
    """keepalive の実挙動を実測する（KEEPALIVE_SCENARIO のときだけ呼ぶ）。

    1. 予算 2 分を書き込む
    2. 75 秒後に生成リクエスト → auth_proxy が MINUTES=60 で mtime 更新
    3. 元の締切（120 秒）を過ぎても止まらない = 延長が効いた
    4. 180 秒に /stop → loop が "stop" で抜ける
    """
    ka = {}
    kc_write_conf(CONF, 1, 2)
    t0 = time.time()

    def activity():
        time.sleep(75)
        try:
            verify(auth, f"http://127.0.0.1:{PROXY_PORT}",
                   key="keepalive_activity")
            run_flag, minutes, mtime = read_conf()
            ka["after_activity"] = {
                "run": run_flag, "minutes": minutes,
                "mtime_age": round(time.time() - (mtime or t0), 1),
            }
            print(f"keepalive 更新: RUN={run_flag} MINUTES={minutes}")
        except Exception as e:
            ka["activity_error"] = f"{type(e).__name__}: {e}"

    def stopper():
        time.sleep(180)
        try:
            code, body = kc_stop_via_proxy(PROXY_PORT, auth=auth)
            ka["stop_response"] = {"status": code, "body": body}
            print(f"/stop: {code} {body}")
        except Exception as e:
            ka["stop_error"] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=activity),
               threading.Thread(target=stopper)]
    for t in threads:
        t.start()
    reason = loop(procs)
    elapsed = round(time.time() - t0, 1)
    for t in threads:
        t.join()

    ka.update(reason=reason, loop_seconds=elapsed,
              # 120 秒の締切で止まっていないこと（延長が効いた）
              extended_past_short_budget=elapsed > 130,
              passed=(reason == "stop" and elapsed > 130))
    print(f"keepalive 判定: {ka['passed']} (reason={reason} {elapsed}s)")
    return ka


def stop_all(procs):
    """TERM → 待機 → KILL。エンジンのワーカー（子孫）も回収する。"""
    if CLEANUP:
        # VRAM を返してから落とす（先に落とすと解放されない実装がある）。
        try:
            data["engine_cleanup"] = cleanup()
        except Exception as e:
            data["engine_cleanup"] = f"failed: {type(e).__name__}: {e}"
            print(f"cleanup 失敗: {type(e).__name__}: {e}")
    for name, p in procs.items():
        if p is None or p.poll() is not None:
            continue
        try:
            p.terminate()
            print(f"  TERM → {name}")
        except Exception as e:
            print(f"  TERM 失敗 {name}: {type(e).__name__}")
    for name, p in procs.items():
        if p is None:
            continue
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    try:
        remaining = kc_stop_processes([
            ("-f auth_proxy.py", "auth_proxy"),
            ("-x cloudflared", "cloudflared"),
            (ENGINE_PATTERN, ENGINE),
        ])
    except Exception as e:          # 停止処理で元の例外をマスクしない
        remaining = [f"停止処理で例外: {type(e).__name__}: {e}"]
    data["stop_remaining"] = remaining
    print(f"残存プロセス: {remaining or 'なし'}")


# ── 6b. 省略可能な段（既定は何もしない）─────────────────────
def warmup():
    """ウォームアップ。WARMUP=True のときだけ呼ばれる（既定は何もしない）。"""


def cleanup():
    """停止前の後片付け。CLEANUP=True のときだけ呼ばれる（既定は何もしない）。"""


# ── 7. エンジン固有（確認項目 / install / launch / warmup / cleanup）──
# 骨格の engine_checks() をエンジン固有の項目で置き換える（空なら既定のまま）。
# vLLM 固有の確認項目（足場は kaggle/serve-cells/verify.py が持つ）
# 呼び出し側: engine_checks(judge, retry, base)
import json, subprocess as _sp, time as _time
import urllib.request as _urlreq

# 骨組みの verify() が auth を渡す。notebook 版は認証 OFF なので None のまま。
AUTH = None


def _req(url, timeout):
    """AUTH があれば Authorization を付けた Request を返す。"""
    if not AUTH:
        return url
    return _urlreq.Request(url, headers={"Authorization": AUTH})


def _post(url, payload, timeout=600):
    # IPv4 フォールバックも効かせる（Kaggle VM は AAAA だけ返すことがある）。
    return kc_post_json(url, payload, timeout=timeout, opener=kc_urlopen_v4,
                        headers={"Authorization": AUTH} if AUTH else None)


CHAT_BODY = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
}


def _reply_parts(j):
    """content と reasoning_content を分けて返す（thinking モデル対策）。"""
    m = (j.get("choices") or [{}])[0].get("message") or {}
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip()


def engine_checks(judge, retry, base, auth=None, direct=False):
    global AUTH
    AUTH = auth
    # [1] 認証なしでチャット UI が取れる（/ は 200 で HTML）
    #     UI は proxy が返すもの。smoke（direct=True）は proxy を立てないので
    #     `/` はエンジン自身の応答（vLLM は 404）になる。誤判定を避けて SKIP する。
    print("[1] 認証なしで UI")
    if direct:
        judge(1, "ui", None, "skip: エンジン直（proxy 無し。smoke）")
    else:
        try:
            with retry(lambda: kc_urlopen_v4(f"{base}/", timeout=60)) as r:
                html = r.read()
                # UI 判定は HTML であること + 十分なサイズで行う（文言に依存させない）
                ok_ui = (r.status == 200 and len(html) > 10000
                         and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
                judge(1, "ui", ok_ui, f"HTTP {r.status} {len(html)}B")
        except Exception as e:
            judge(1, "ui", False, f"{type(e).__name__}: {e}")

    # [2] /v1/models（vLLM が応答している証明）
    print("[2] /v1/models")
    try:
        with retry(lambda: kc_urlopen_v4(_req(f"{base}/v1/models", 60), timeout=60)) as r:
            md = json.loads(r.read())
        ids = [m.get("id") for m in md.get("data", [])]
        judge(2, "models", r.status == 200 and MODEL_NAME in ids,
              f"HTTP {r.status} ids={ids}")
    except Exception as e:
        judge(2, "models", False, f"{type(e).__name__}: {e}")

    # [3] ストリーミング（SSE がトンネルを抜けるか）
    print("[3] ストリーミング")
    try:
        resp = retry(lambda: _post(f"{base}/v1/chat/completions",
                                   dict(CHAT_BODY, stream=True)))
        times = []
        t0 = _time.time()
        with resp as r:
            for raw in r:            # 行到着 = 逐次受信の証拠
                if not raw.strip():
                    continue
                times.append(_time.time() - t0)
        total = _time.time() - t0
        if len(times) >= 3:
            first = times[0]
            spread = times[-1] - first
            ok = spread > 0.2 and first < total * 0.8
            detail = (f"chunks={len(times)} first={first:.2f}s "
                      f"total={total:.2f}s spread={spread:.2f}s")
        else:
            ok, detail = False, f"chunks={len(times)} first=N/A"
        judge(3, "streaming", ok, detail)
    except Exception as e:
        judge(3, "streaming", False, f"{type(e).__name__}: {e}")

    # [4] OpenAI 互換（temperature=0 を送っても 200 で生成される）
    #     プロキシの --enforce-sampling が 0.7 に矯正する経路の確認でもある。
    print("[4] /v1/chat/completions (temperature=0)")
    try:
        body4 = dict(CHAT_BODY, temperature=0, repetition_penalty=1.0)
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 body4, timeout=300)) as r:
            st4 = r.status
            cj = json.loads(r.read())
        ctext = "".join((c.get("message") or {}).get("content") or ""
                        for c in cj.get("choices", []))
        usage = cj.get("usage") or {}
        ok4 = st4 == 200 and bool(ctext or usage.get("completion_tokens"))
        judge(4, "openai-compat", ok4,
              f"HTTP {st4} reply={ctext[:40]!r} usage={usage}")
    except Exception as e:
        judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

    # [5] GPU 常駐（CPU フォールバック検出）
    print("[5] GPU 常駐")
    try:
        mem = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used = [int(x.split(",")[0]) for x in mem.splitlines() if "," in x]
        alive = _sp.run(["pgrep", "-f", "vllm serve"],
                        capture_output=True, text=True).stdout.strip() != ""
        judge(5, "gpu-resident", alive and bool(used) and sum(used) > 4000,
              f"proc={'alive' if alive else 'DEAD'} vram={used}")
    except Exception as e:
        judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

    # [6] 会話（マルチターン: 履歴を渡して文脈が保たれるか）
    print("[6] 会話（マルチターン）")
    try:
        conv = [{"role": "user",
                 "content": "私の名前はコウヘイです。覚えてください。短く返事して。"}]
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 dict(CHAT_BODY, messages=conv,
                                      max_tokens=96), timeout=300)) as r1:
            j1 = json.loads(r1.read())
        c1, t1 = _reply_parts(j1)
        # thinking だけで content が空のことがある。履歴には本文を優先し、
        # 無ければ思考の先頭を使う（空の assistant を積むと文脈が切れる）。
        conv.append({"role": "assistant", "content": c1 or t1[:600] or "(empty)"})
        conv.append({"role": "user", "content": "私の名前は何？名前だけ答えて。"})
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 dict(CHAT_BODY, messages=conv,
                                      max_tokens=96), timeout=300)) as r2:
            j2 = json.loads(r2.read())
        c2, t2 = _reply_parts(j2)
        a2 = (c2 + "\n" + t2).strip()
        name_ok = any(x in a2.lower() for x in ("コウヘイ", "コーヘイ", "kohei"))
        judge(6, "conversation", bool(c1 or t1) and name_ok,
              f"turn1={(c1 or t1)[:50]!r} turn2={a2[:70]!r}")
    except Exception as e:
        judge(6, "conversation", False, f"{type(e).__name__}: {e}")


def stage_install():
    # vLLM をインストールし、プロキシと UI を書き出す
    import base64
    import glob as _glob
    import os
    import shutil
    import subprocess
    import sys
    import time

    BIN_DIR = "/usr/local/bin"
    MIN_SIZE = 1024 * 1024

    # ---- 計測（高度化の基礎）----
    # result.json の `timings.install_seconds` は **None のまま**だった。vLLM は
    # install をテンプレートの `stage_install()` ではなくこのセルで行うため、
    # 誰も記録していなかった。比較できないと「速くなった/遅くなった」を言えない。
    # 区間の**絶対オフセット**（セル開始からの秒）を残す。差分は後から計算できる。
    _T_INSTALL = time.time()
    _MARKS = {}


    def _mark(name):
        """区間の到達時刻を記録する（セル開始からの秒）。"""
        _MARKS[name] = round(time.time() - _T_INSTALL, 1)
        return _MARKS[name]

    # ---- cloudflared（Dataset キャッシュ優先）----
    # ngrok は 2026-11-01 に廃止したので、トンネルの種類で分岐しない
    # （`notes/url-publishing.md`）。cloudflared は常に用意する。
    CF_PATH = kc_find_dataset("my-cloudflared-bin")
    dst = os.path.join(BIN_DIR, "cloudflared")
    src = None
    for cand in (os.path.join(CF_PATH, "cloudflared"),
                 os.path.join(CF_PATH, "bin", "cloudflared")):
        if os.path.isfile(cand) and os.path.getsize(cand) >= MIN_SIZE:
            src = cand
            break
    if src:
        shutil.copy2(src, dst)
        subprocess.run(["chmod", "+x", dst])
        print(f"✅ cloudflared 復元: {src}")
    elif not (os.path.isfile(dst) and os.path.getsize(dst) >= MIN_SIZE):
        print("⬇️  cloudflared をダウンロード中...")
        subprocess.run(
            "curl -fsSL --connect-timeout 15 -o /usr/local/bin/cloudflared "
            f"{kc_cloudflared_url()}"
            " && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=False, timeout=300,
        )
    if not os.path.isfile(dst):
        fail("cloudflared がありません（Dataset 未マウント / 取得失敗）")
    r = subprocess.run([dst, "--version"], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        fail("cloudflared の実行に失敗しました（実行ビット/Dataset を確認）")

    # ---- vLLM ----
    # wheelhouse Dataset（vllm 0.32.0 + 依存 wheel / 3.25 GB）を 2 つに分けて持つ:
    #   vllm-wheelhouse        … 190 wheel（torch 2.13.0 + cu13 一式）
    #   vllm-wheelhouse-patch  … nvidia-cublas 13.1.1.3 のみ
    # `cuda-toolkit 13.0.3` が cublas を `==13.1.1.3.*` で固定するため、本体側の
    # 13.8.2.12 だけでは `--no-index` が解決できない（CPU 実測 2026-10-31）。
    # 解決に失敗したら PyPI にフォールバックする（起動できない方が損失が大きい）。
    _WH_SLUGS = ("vllm-wheelhouse", "vllm-wheelhouse-patch")
    _pip_extra = []


    def _whl_dir(root):
        if _glob.glob(os.path.join(root, "*.whl")):
            return root
        for _s in sorted(_glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(_s) and _glob.glob(os.path.join(_s, "*.whl")):
                return _s
        return None


    def _unzip_wheels(root, tag):
        """--dir-mode zip の Dataset を 1 回だけ展開する。"""
        _zips = (_glob.glob(os.path.join(root, "*.zip"))
                 + _glob.glob(os.path.join(root, "*", "*.zip")))
        if not _zips:
            return None
        import time as _t
        import zipfile as _zf
        _dst = os.path.join("/kaggle/tmp/whl", tag)
        os.makedirs(_dst, exist_ok=True)
        _t0 = _t.time()
        print(f"📦 {tag} の zip を展開: {[os.path.basename(z) for z in _zips]}")
        for _z in _zips:
            with _zf.ZipFile(_z) as _f:
                _f.extractall(_dst)
        print(f"   展開 {_t.time() - _t0:.1f}s")
        return _whl_dir(_dst) or _dst


    _wh_dirs = []
    for _slug in _WH_SLUGS:
        _root = kc_find_dataset(_slug)
        if not os.path.isdir(_root) or not os.listdir(_root):
            print(f"ℹ {_slug} なし")
            continue
        _d = _whl_dir(_root) or _unzip_wheels(_root, _slug)
        if not _d:
            print(f"⚠ {_slug} に wheel が無い")
            continue
        _n = len(_glob.glob(os.path.join(_d, "*.whl")))
        print(f"📦 ローカル wheelhouse: {_slug}（{_n} wheel）{_d}")
        _wh_dirs.append(_d)

    if _wh_dirs:
        _pip_extra = ["--no-index"]
        for _d in _wh_dirs:
            _pip_extra += ["--find-links", _d]
    else:
        print("ℹ wheelhouse Dataset なし → PyPI から取得（5〜10分）")
        # 「Dataset を付け忘れた」と「マウント先が想定と違う」を切り分ける。
        # これが無いと、metadata を確認するために 1 往復（15 分）を使うことになる。
        _seen = []
        for _pat in ("/kaggle/input/*", "/kaggle/input/datasets/*/*"):
            _seen += sorted(_p for _p in _glob.glob(_pat) if os.path.isdir(_p))
        for _p in _seen[:12]:
            print(f"    {_p}")

    # ---- warm cache（任意 Dataset）----
    # torch.compile の AOT 成果物と Triton JIT を Dataset から復元する。
    # コンテナは毎回まっさらなので、これがあると compile 30〜60s を省ける。
    # キャッシュキーが違えば単にミスするだけ（壊れない）。
    #
    # 復元の ON/OFF を外から切り替えられるようにしてある（`KC_WARM_CACHE=0`）。
    # Dataset を付けたまま「復元しない」比較を playground で取りたいため。
    # これが無いと、Dataset を付けた瞬間に A/B の対照群が消える。
    #
    # 3 値にしてある:
    #   "0"       … 復元しない（対照群）
    #   "1"       … あれば復元。無ければ「なし」と出して続行（本番の既定）
    #   "require" … **無ければ失敗**。実験用
    #
    # `require` が要る理由: `1` のままだと Dataset を付け忘れた実験が
    # **baseline とまったく同じ結果を出し**、「キャッシュは効かない」という
    # 偽陰性になる。1 往復（約 40 分）を捨てるより、その場で落ちたほうが安い。
    # 実測 2026-11-01: `vllm-warm-cache` はまだ Kaggle に存在せず、playground の
    # push が `not valid dataset sources` で落としていた（＝気づかないまま走る）。
    _KC_WC_MODE = os.environ.get("KC_WARM_CACHE", "1").strip().lower()
    _USE_WARM_CACHE = _KC_WC_MODE not in ("0", "off", "no", "false")
    _WC_REQUIRED = _KC_WC_MODE in ("require", "required")
    # 復元の結果は `timings` に残す。**print だけだと 1 回の実行から読み取れない**
    # （ログは流れるが、`cli timings` で横並びにできない）。
    _WC_INFO = {"mode": _KC_WC_MODE, "restored": False, "source": "", "tar": "",
                "mib": 0.0, "entries": 0, "seconds": 0.0, "missing": []}
    # 「完全な tar か」の判定に**件数を使わない**。件数は Triton の版で変わるし、
    # しきい値を置くと「正当な tar を弾く」ほうに倒れる。代わりに**名前で見る**。
    #
    # 実測 2026-11-01（`vllm-playground` の 2 本の tar を数えた）:
    #   不完全: 30 エントリ / 70 KB   … profile run に到達していない
    #   完全  : 2,077 エントリ / 195 MB / カーネル名 140 種
    # 差は GDN（gated delta net）の chunked カーネル群が入っているか。
    # 373 秒の主役はこの 2 つ（名前は vLLM の fla 実装にハードコードされている）:
    #   chunk_fwd_kernel_o                                 x252
    #   chunk_gated_delta_rule_fwd_kernel_h_blockdim64      x84
    # **`chunk_o` / `chunk_delta_h` で探してはいけない**。実際のファイル名は
    # 上のように長く、部分一致しないので「完全な tar を不完全と誤判定」する
    # （最初にこの 2 つで書いて、v4 の tar を弾きかけた）。
    # 末尾の `_blockdim64` は head dim を名前に含めたもので、版が変わると
    # 変わりうる。だから**前側だけ**を条件にする。
    _WC_REQUIRED_KERNELS = ("chunk_fwd_kernel_o", "chunk_gated_delta_rule_fwd_kernel_h")
    _wc = kc_find_dataset("vllm-warm-cache")
    # `kc_find_dataset` は**見つからなくても既定パスを返す**（`/kaggle/input/<slug>`）。
    # そのため `if _wc:` だけでは常に真になり、tar が 1 本も無いときに**何も
    # 表示しないまま素通り**していた（実測 2026-11-01: ログに warm cache の行が
    # 1 行も出ず、効いているのか判別できなかった）。
    #
    # **Kaggle は Dataset に上げた `.tar` を展開して保持する。** 実測 2026-11-01:
    # 195 MB の tar を `datasets create` したら、マウント側は
    # `root/.triton/cache/<hash>/...` の木になっていて `*.tar` は 1 本も無かった。
    # 「tar が無い＝キャッシュが無い」と読むと**正当な Dataset を落とす**ので、
    # tar と展開済みの木の両方を受ける。
    def _kc_wc_tree(root):
        """Dataset の中の Triton キャッシュのディレクトリ（無ければ ""）。"""
        if not root or not os.path.isdir(root):
            return ""
        base = root.rstrip(os.sep).count(os.sep)
        for _dp, _dn, _fs in os.walk(root):
            if (os.path.basename(_dp) == "cache"
                    and os.path.basename(os.path.dirname(_dp)) == ".triton"):
                return _dp
            if _dp.count(os.sep) - base >= 4:      # これ以上は潜らない
                _dn[:] = []
        return ""


    _tars = [] if not _USE_WARM_CACHE else (
        _glob.glob(os.path.join(_wc, "*.tar"))
        + _glob.glob(os.path.join(_wc, "*", "*.tar")))
    _tree = "" if not _USE_WARM_CACHE else _kc_wc_tree(_wc)
    if not _USE_WARM_CACHE:
        print("ℹ warm cache: KC_WARM_CACHE=0 なので復元しない（A/B の対照）")
    elif _tars or _tree:
        import shutil as _sh2
        import tarfile as _tf2
        # 復元先を**数える**。`extractall("/")` は tar の arcname 次第で
        # どこへでも落ちるので、サイズだけ出していると「復元したのに 1 件も
        # 当たらない」を 1 往復使って調べることになる。
        # Triton は 1 カーネル 1 ディレクトリなので直下の数で足りる。
        _tc = os.path.join(os.path.expanduser("~"), ".triton", "cache")
        _t0 = __import__("time").time()
        # tar があれば tar（展開済みの木は見ない＝二重にコピーしない）。
        _srcs = list(_tars) if _tars else [_tree]
        for _s in _srcs:
            try:
                if os.path.isdir(_s):
                    _mib = 0.0
                    for _r, _d, _fs in os.walk(_s):
                        for _fn in _fs:
                            try:
                                _mib += os.path.getsize(os.path.join(_r, _fn))
                            except OSError:
                                pass
                    _mib = round(_mib / 1e6, 1)
                    os.makedirs(_tc, exist_ok=True)
                    _sh2.copytree(_s, _tc, dirs_exist_ok=True)
                    _src = _s
                else:
                    _mib = round(os.path.getsize(_s) / 1e6, 1)
                    with _tf2.open(_s) as _f:
                        _f.extractall("/")
                    _src = os.path.basename(_s)
                _n = len(os.listdir(_tc)) if os.path.isdir(_tc) else 0
                # 入っているカーネル名を集める。Triton は 1 カーネル 1 ディレクトリで、
                # 中に `<カーネル名>.json` / `.ptx` などを置く。名前で見るのは
                # 件数より意味がはっきりしているため（件数は Triton の版で変わる）。
                _names = []
                for _r, _d, _fs in os.walk(_tc):
                    for _fn in _fs:
                        _names.append(_fn.lower())
                _missing = [k for k in _WC_REQUIRED_KERNELS
                            if not any(k in s for s in _names)]
                _WC_INFO.update(restored=True, source=_src, tar=_src, mib=_mib,
                                entries=_n, missing=_missing,
                                seconds=round(__import__("time").time() - _t0, 1))
                print(f"🧠 キャッシュ復元: {_src} {_mib:.0f} MB "
                      f"({_WC_INFO['seconds']}s) / Triton エントリ {_n}（{_tc}）"
                      + (f" / ⚠ 欠け: {', '.join(_missing)}" if _missing else ""))
            except Exception as _e:
                print(f"⚠ キャッシュ復元をスキップ: {_e}")
        # 復元はできたが**中身が薄い**場合。不完全な tar を復元すると一部だけ当たって
        # 「少し効いた」ように見え、373 秒が戻る理由を「キャッシュは効かない」と
        # 誤読する。`require` はその誤読を防ぐためのモードなので、ここも落とす。
        if _WC_REQUIRED and not _WC_INFO["restored"]:
            raise RuntimeError(
                f"KC_WARM_CACHE=require なのに復元できなかった: {_srcs}。"
                "抽出／コピーで例外が出ている（上の ⚠ を参照）。")
        if _WC_REQUIRED and _WC_INFO["missing"]:
            raise RuntimeError(
                f"KC_WARM_CACHE=require なのに "
                f"{', '.join(_WC_INFO['missing'])} が入っていない: "
                f"{_WC_INFO['source']} {_WC_INFO['mib']} MB / "
                f"エントリ {_WC_INFO['entries']}。"
                "この 2 つは 373 秒の主役（profile run の GDN Triton JIT）で、"
                "無いということは**起動に成功していない回**のキャッシュということ。"
                "不完全なまま測ると『キャッシュは効かない』という偽陰性になる。"
                "`KC_EXPERIMENT=baseline` の成果物から採り直すこと"
                "（判断できる場合は KC_WARM_CACHE=1 でこの検査を外せる）。")
    elif _WC_REQUIRED:
        # ここで落とす。続行すると baseline と同じ数字が出て、効かなかったのか
        # Dataset が無かったのか区別できないログだけが残る。
        raise RuntimeError(
            f"KC_WARM_CACHE=require なのに tar も展開済みの木も無い: {_wc}"
            f"（exists={os.path.isdir(_wc)}）。"
            "Dataset `vllm-warm-cache` が未作成か、空のまま push している。"
            "このまま走らせると baseline と同じ結果になり、"
            "「キャッシュは効かない」という偽陰性になるので中断する。")
    else:
        # 「Dataset を付け忘れ」と「中身が入っていない」を切り分ける。
        print(f"ℹ warm cache なし（compile をやり直す）: {_wc}"
              f" exists={os.path.isdir(_wc)}")

    # ---- torch を「消してから入れる」ための道具 ----
    # `pip uninstall torch` では足りない（実測 v7）。uninstall は RECORD に載っている
    # ファイルしか消さないので、**新しい版で削除されたファイル**は残る。結果、
    # 2.11 の `torch/utils/_config_module.py`（`deprecated=` を知らない Config）と
    # 2.13 の `torch/_dynamo/config.py`（`Config(deprecated=True)` を呼ぶ）が同居し、
    #   TypeError: Config() got an unexpected keyword argument 'deprecated'
    # になる。torch 自身のファイル同士が食い違っているので、版を揃え直す以外に直せない。
    def _site_dirs():
        r = subprocess.run(
            [sys.executable, "-c",
             "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, timeout=120)
        d = (r.stdout or "").strip()
        return [d] if d and os.path.isdir(d) else []


    def _purge_torch():
        """site-packages の torch / torchvision / torchaudio 系を物理削除する。

        `torch-*` はハイフン付きなので torchmetrics / torchcodec を巻き込まない
        （巻き込むと無関係なパッケージまで壊す）。
        """
        n = 0
        for _d in _site_dirs():
            for _pat in ("torch", "torch-*", "torchvision", "torchvision-*",
                         "torchaudio", "torchaudio-*", "torchgen", "torchgen-*",
                         "functorch", "functorch-*"):
                for _p in _glob.glob(os.path.join(_d, _pat)):
                    try:
                        if os.path.isdir(_p):
                            shutil.rmtree(_p, ignore_errors=True)
                        else:
                            os.remove(_p)
                        n += 1
                    except OSError as _e:
                        print(f"⚠ 削除できず: {_p} ({_e})")
        return n


    def _torch_diag():
        """torch が混ざっているかを 1 画面で分かる形で出す。

        これが無いと「import に失敗」だけ見て 6 分を無駄にし、原因を推測する羽目になる。
        `_config_module.py` に `deprecated` が無ければ、そちらが旧版の残骸。
        """
        out = []
        for _d in _site_dirs():
            for _p in sorted(_glob.glob(os.path.join(_d, "torch*"))):
                out.append(f"  {os.path.basename(_p)}")
            _cm = os.path.join(_d, "torch", "utils", "_config_module.py")
            if os.path.isfile(_cm):
                _src = open(_cm, encoding="utf-8", errors="replace").read()
                out.append(f"  _config_module.py {len(_src)}B "
                           f"deprecated={'deprecated' in _src}")
            else:
                out.append("  _config_module.py なし")
        return "\n".join(out) or "  (site-packages が見つからない)"


    def _probe(code, timeout=900):
        """別プロセスで import を試す。親で import すると半端な状態が尾を引く。"""
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-1200:]


    _r = subprocess.run([sys.executable, "-c", "import vllm"],
                        capture_output=True, text=True)
    if _r.returncode == 0:
        print("✅ vLLM は既にインストール済み")
    else:
        # ドライバと現行 torch を先に記録する。CUDA 13 系の wheel を入れると
        # `libcudart.so.13` 不在や「driver version is insufficient」で落ちるので、
        # 失敗したときに原因を 1 往復で特定できるようにしておく。
        _drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True)
        _tv = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.__version__, torch.version.cuda)"],
            capture_output=True, text=True)
        print(f"🧩 driver: {(_drv.stdout or '').strip().splitlines()[:1]} / "
              f"torch: {(_tv.stdout or '').strip()}")
        print(f"⬇️  vLLM をインストール中: {VLLM_PIP_SPEC}（5〜10分）")
        # 依存も一緒に入れる（`--no-deps` を付けない）。
        # 理由: vLLM 0.32.0 は CUDA 13 ビルドで、イメージの torch 2.11.0+cu128 のままでは
        # `ImportError: libcudart.so.13` で起動できない（実測 2026-10-31）。torch 2.13.0 と
        # nvidia cu13 一式まで入れて初めて噛み合う。ドライバは 580.178.04 で CUDA 13 対応済み。
        #
        # ただし**上書きインストールは不可**。2.11 のファイルが残って 2.13 と混ざり、
        # `TypeError: Config() got an unexpected keyword argument 'deprecated'`
        # （torch._dynamo）のような内部不整合になる（実測）。しかも pip uninstall では
        # 消し切れない（新版で削除されたファイルが残る）ので、物理削除する。
        print("🧹 site-packages の torch 系を物理削除してから入れる")
        print(f"   {_purge_torch()} 個のパスを削除")

        # ---- インストーラの選択（実測で決めた）----
        # CPU ベンチ `vllm-install-bench`（2026-11-01、`pip install --target` で同じ
        # 190 wheel を 4 通りに展開）:
        #
        #   pip（暖）        200.1s   81,844 files
        #   pip --no-compile 112.6s   56,416 files
        #   uv               35.7s   56,812 files
        #
        # 差の大半は **pyc の生成（25,428 個）と pip の直列展開**。転送ではない
        # （wheelhouse を付けても pip は 199.6 → 230.4s と遅くなった）。
        #
        # ⚠ **この比は本番に移らなかった**（実測 2026-11-01 / v12）。
        #   bench は `--target` の**まっさらなディレクトリ**へ入れる。本番は
        #   torch を物理削除した site-packages へ入れる。同じ wheel 同じホストでも
        #   pip 261.6s → **uv 1151.7s（4.4 倍遅い）**。理由は未同定（uv のキャッシュ
        #   経路が GCSFUSE 上の wheel を引き直している疑い）。**既定は pip**。
        #   uv を試すときは `KC_INSTALLER=uv` を明示した回だけにする（既定を壊さない）。
        _want_uv = os.environ.get("KC_INSTALLER", "").strip().lower() == "uv"
        _uv = None
        if _want_uv:
            _uv = shutil.which("uv")
            if not _uv:
                _ru = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                      "--no-input", "uv"],
                                     capture_output=True, text=True, timeout=600)
                _uv = shutil.which("uv")
                print(f"⬇️  uv を用意: {_uv}（rc={_ru.returncode}）")
            if not _uv:
                print("⚠ uv を用意できない → pip で入れる")
        _installer = "pip"
        _p = None
        if _uv:
            _uv_cmd = ([_uv, "pip", "install", "--python", sys.executable,
                        "--quiet"] + _pip_extra + [VLLM_PIP_SPEC])
            print(f"⬇️  vLLM を uv でインストール: {VLLM_PIP_SPEC}"
                  f"（wheelhouse {len(_wh_dirs)} 個）")
            try:
                _p = subprocess.run(_uv_cmd, capture_output=True, text=True,
                                    timeout=1800)
            except subprocess.TimeoutExpired:
                print("⚠ uv が 1800 秒で終わらない → pip に切り替え")
                _p = None
            if _p is not None:
                print(((_p.stdout or "") + (_p.stderr or ""))[-1500:])
                # uv は entry point を作るが、`--no-index` で解決が欠けたときに
                # **部分的に成功**して rc=0 を返すことがある。`vllm` CLI の実在まで
                # 確かめる（無いと serve が始まらない）。
                if _p.returncode == 0 and shutil.which("vllm"):
                    _installer = "uv"
                else:
                    print(f"⚠ uv が不完全（rc={_p.returncode}, "
                          f"vllm={shutil.which('vllm')}）→ pip で入れ直す")
        if _installer != "uv":
            _p = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                + _pip_extra + [VLLM_PIP_SPEC],
                capture_output=True, text=True, timeout=3600,
            )
            print((_p.stdout or "")[-2000:])
            print((_p.stderr or "")[-2000:])
            if _p.returncode != 0 and _pip_extra:
                # wheelhouse の版ずれ等で解決できないことがある（実測: cublas 13.1.1.3）。
                # ここで止めると 1 回分の GPU を捨てるので、ネットワークに切り替えて再試行。
                print("⚠ ローカル wheelhouse で解決失敗 → PyPI にフォールバック")
                _pip_extra = []
                _p = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                     VLLM_PIP_SPEC],
                    capture_output=True, text=True, timeout=3600,
                )
                print((_p.stdout or "")[-2000:])
                print((_p.stderr or "")[-2000:])
            if _p.returncode != 0:
                fail("vLLM のインストールに失敗しました（上のログ参照）")
        print(f"   install 手段: {_installer}")
        _mark("pip")

        # 入れた直後に別プロセスで確かめる。ここで落ちるなら torch はまだ混ざっている。
        # torchaudio の調整に進んでも直らない種類の失敗なので、構成を出して**即座に**止める
        # （先へ進むと 5 分以上無駄にする）。
        _ok_imp, _imp_out = _probe(
            "import torch, vllm;print(torch.__version__, torch.version.cuda, "
            "torch.__file__);print('vllm', vllm.__version__)")
        if not _ok_imp:
            print("⚠ インストール直後の import に失敗。torch の構成を出す:")
            print(_torch_diag())
            print(_imp_out)
            fail("vLLM の import に失敗しました（torch の構成は上記）")

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
    def _vllm_pin(name):
        """vLLM の METADATA が要求する版を取り出す（例: 'torchaudio==2.11.0'）。

        torchaudio の版は **torch の版ではない**。vllm 0.32.0 は torch==2.13.0 と
        同時に torchaudio==2.11.0 を要求する（METADATA 実測）。torch に合わせて
        2.13.0 を探すと存在せず、1 往復無駄にする。
        """
        r = subprocess.run(
            [sys.executable, "-c",
             "import importlib.metadata as m;print(chr(10).join(m.requires('vllm') or []))"],
            capture_output=True, text=True, timeout=120)
        for _ln in (r.stdout or "").splitlines():
            _ln = _ln.split(";")[0].strip()          # 環境マーカーは落とす
            if _ln.lower().startswith(name + "=="):
                return _ln
        return None


    # 版が同じでも**ビルド**が違うと落ちる。pip は `2.11.0+cu128` を `==2.11.0` の
    # 充足とみなす（PEP 440 ではローカル版を無視する）ので、イメージ残りの +cu128 が
    # そのまま残り「PyTorch has CUDA 13.0 whereas TorchAudio has 12.8」になる（実測）。
    # PyPI の torchaudio 2.11.0 は +cu130 ビルドなので、--force-reinstall で置き換える。
    _TA_SPEC = _vllm_pin("torchaudio") or "torchaudio==2.11.0"


    def _ta_check():
        r = subprocess.run([sys.executable, "-c", "import torchaudio"],
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-900:]


    _ta_good, _ta_err = _ta_check()
    if not _ta_good:
        print(f"⚠ torchaudio 不整合 → {_TA_SPEC} を --no-deps で入れ直す")
        print(_ta_err)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                       + _pip_extra
                       + ["--no-deps", "--force-reinstall", _TA_SPEC],
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
    _mark("torchaudio")

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
        print(f"⚠ vLLM CLI が起動しない → torchaudio を {_TA_SPEC} で入れ直す")
        print(_err)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                       + _pip_extra
                       + ["--no-deps", "--force-reinstall", _TA_SPEC],
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
    _mark("cli_help")

    try:
        import torch
        import vllm
    except Exception as _e:
        # import 失敗は「torch の入れ替わり」で起きやすい。原因を残して止める。
        fail(f"vllm/torch の import に失敗: {type(_e).__name__}: {_e}")
    print(f"✅ vLLM {vllm.__version__} / torch {torch.__version__} "
          f"/ CUDA {torch.version.cuda}")
    print(f"   GPU 認識: {torch.cuda.device_count()} 枚")
    _mark("import")

    # 比較可能な形で残す。`install_seconds` はテンプレートの timings が読む。
    # セルから data を直接触らない理由は kc_save_timings の説明を参照。
    #
    # `torch_after_install` を別名で残すのは、result.json の `torch` が
    # **install 前**（イメージ同梱版）の値だから。実測 2026-11-01 は
    # `torch: 2.11.0+cu128`（イメージ）→ `torch 2.13.0+cu130`（実際に動いた版）で、
    # 表だけ見ると「2.11 で動いた」と誤読する。
    kc_save_timings({"install_seconds": _MARKS["import"],
                     "install_marks": dict(_MARKS),
                     "installer": _installer,
                     "torch_after_install": torch.__version__,
                     "cuda_after_install": torch.version.cuda,
                     "wheelhouse": {"dirs": len(_wh_dirs),
                                    "no_index": bool(_pip_extra)},
                     # 復元の有無と**エントリ数**。`entries` が小さいまま起動が
                     # 速くならなければ「効いていない」ではなく「tar が不完全」。
                     "warm_cache_restore": dict(_WC_INFO),
                     "vllm_version": vllm.__version__})
    print(f"⏱ install 内訳: {_MARKS}")

    # ---- プロキシ + UI ----
    with open(PROXY_PATH, "wb") as f:
        f.write(base64.b64decode(PROXY_SRC_B64))
    with open(CHAT_PATH, "wb") as f:
        f.write(base64.b64decode(CHAT_SRC_B64))
    print(f"✅ auth_proxy.py: {os.path.getsize(PROXY_PATH):,} B")
    print(f"✅ llama-chat.html: {os.path.getsize(CHAT_PATH):,} B")

def launch(model_dir):
    global CONTEXT_LENGTH
    # vLLM 起動 → /v1/models が返るまで待機（失敗時は条件を落として再試行）
    import json
    import os
    import re
    import shutil
    import signal
    import subprocess
    import threading
    import time
    import urllib.error
    import urllib.request

    STAGE = "/kaggle/tmp/vllm-model"
    MTP_FILE = "model-mtp.safetensors"

    # 起動ログの「節目」。末尾 1 行だけを見ていると、長い工程に入った瞬間から
    # 何も分からなくなる。実測 2026-10-31: torch.compile 終了（514s）から CUDA graph
    # capture 開始（874s）まで **360 秒**、ログは `shm_broadcast ... 60 seconds` の
    # 繰り返しだけで、重みロードの 2 回目なのか autotune なのか特定できなかった。
    # 節目を別枠で拾えば「最後に越えた関門」が常に分かる。
    _MILESTONES = (
        "Starting to load model", "load_format", "Loading weights took",
        "Model loading took", "init engine", "Starting torch.compile",
        "torch.compile took", "available_kv_cache_memory", "Memory profiling",
        "Capturing CUDA graphs", "Graph capturing finished", "init distributed",
        "Starting vLLM API server", "route",
    )

    # ログ行頭の時刻（`10-31 13:22:08`）。日付は使わず時刻だけ取り出す。
    _TS_RE = re.compile(r"\b\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})\b")


    def _phase_marks(log_text, t0_wall):
        """起動ログから「各節目に到達した時刻（起動からの秒）」を取り出す。

        60 秒ポーリングでは graph capture（実測 41 秒）すら分解できない。
        vLLM のログ行には `10-31 13:22:08` 形式の時刻が入っているので、起動時刻と
        突き合わせれば **1 秒精度**になる。これが無いと「360 秒の無音区間」の
        正体を毎回推測することになる（実測 2026-10-31）。

        同じ節目が複数回出たら**最初**を採る。最後を採ると、再試行やサブプロセスの
        重複出力で区間が縮んで見える。取れなかった節目はキーごと欠落させる
        （0 で埋めると「一瞬で終わった」と誤読する）。
        """
        out = {}
        for ln in log_text.splitlines():
            m = _TS_RE.search(ln)
            if not m:
                continue
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if t + 43200 < t0_wall:      # 0 時をまたいだ
                t += 86400
            for name in _MILESTONES:
                if name in ln and name not in out:
                    out[name] = round(t - t0_wall, 1)
        return out


    _TS_ONLY = re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")


    def _long_gaps(log_text, t0_wall, min_seconds=30, keep=5):
        """時刻付き行の間隔から「無音区間」を抽出する（長い順）。

        vLLM は**長い工程に入ると何も出さない**。実測 2026-11-01: torch.compile
        終了（282s）→ CUDA graph capture 開始（661s）の **379 秒**は
        `No available shared memory broadcast block ... 60 seconds` だけが並ぶ。
        区間の長さは分かっても中身が分からないので、**無音の直前と直後の行**を
        対で残す。次に疑うべき対象はこの 2 行で決まる。

        同じ警告が 60 秒おきに繰り返す場合は 1 本に**連結**する。連結しないと
        60 秒 × 5 本が並び、「381 秒詰まった」ではなく「60 秒が 5 回」に見える。
        判定は「区切りの行が**埋め草**（同じ本文が何度も出る行）かどうか」で行う。
        埋め草は無音の**中身**であって終わりではないので、次の無音と繋げる。
        """
        stamps = []
        for ln in log_text.splitlines():
            m = _TS_RE.search(ln)
            if not m:
                continue
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if t + 43200 < t0_wall:      # 0 時をまたいだ
                t += 86400
            stamps.append((t - t0_wall, ln))

        def _body(s):
            """時刻を落とした本文（同じ行の繰り返しを判定するため）。"""
            return _TS_ONLY.sub("", s).strip()

        seen = {}
        for _, ln in stamps:
            b = _body(ln)
            seen[b] = seen.get(b, 0) + 1

        gaps = []
        for (ta, la), (tb, lb) in zip(stamps, stamps[1:]):
            if tb - ta < min_seconds:
                continue
            gaps.append({"from": round(ta, 1), "seconds": round(tb - ta, 1),
                         "before": la.strip()[:160], "after": lb.strip()[:160],
                         "_ab": _body(lb)})

        merged = []
        for g in gaps:
            prev = merged[-1] if merged else None
            # 直前の無音が**埋め草で終わっている**なら、無音はまだ続いている。
            if prev is not None and seen.get(prev["_ab"], 0) > 1:
                prev["seconds"] = round(g["from"] + g["seconds"] - prev["from"], 1)
                prev["after"] = g["after"]
                prev["_ab"] = g["_ab"]
            else:
                merged.append(g)
        for g in merged:
            g.pop("_ab", None)
        merged.sort(key=lambda g: -g["seconds"])
        return merged[:keep]


    def _phase_spans(marks):
        """主要区間の所要（秒）。片方の節目が無ければ None（0 と区別する）。"""
        pairs = {
            "weights": ("Starting to load model", "Starting torch.compile"),
            "compile": ("Starting torch.compile", "torch.compile took"),
            "after_compile": ("torch.compile took", "Capturing CUDA graphs"),
            "capture": ("Capturing CUDA graphs", "Graph capturing finished"),
        }
        out = {}
        for name, (a, b) in pairs.items():
            if a in marks and b in marks and marks[b] >= marks[a]:
                out[name] = round(marks[b] - marks[a], 1)
            else:
                out[name] = None
        return out



    def _stage_model(with_mtp):
        """モデルを symlink で 1 か所にまとめる（19 GB をコピーしない）。

        MTP の draft head (model-mtp.safetensors) は、speculative を切った状態で
        読ませると unexpected key になるため、その場合だけ除外する。
        """
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


    DRAFT_DIR = "/kaggle/tmp/vllm-draft"
    # draft に必要なのは mtp の重み + config 類だけ。残すと `*.safetensors` の glob が
    # 本体 17.41 GiB まで拾い、draft が同じだけ読み直す（実測 §27）。
    DRAFT_KEEP = ("config", "tokenizer", "chat_template", "generation_config",
                  "special_tokens_map", "vocab", "merges", "added_tokens")


    def _stage_draft():
        """draft (MTP) 専用の最小ディレクトリを作る。戻り値は symlink 本数。"""
        if not os.path.isfile(os.path.join(MODEL_SRC, MTP_FILE)):
            return 0
        if os.path.isdir(DRAFT_DIR):
            shutil.rmtree(DRAFT_DIR)
        os.makedirs(DRAFT_DIR, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(MODEL_SRC)):
            if fn.startswith("."):
                continue
            if fn != MTP_FILE and not fn.startswith(DRAFT_KEEP):
                continue
            src = os.path.join(MODEL_SRC, fn)
            if not os.path.isfile(src):
                continue
            os.symlink(src, os.path.join(DRAFT_DIR, fn))
            n += 1
        return n


    # ---- 利用可能なフラグを確認（バージョン差の吸収）----
    # vLLM 0.32 の `serve --help` は Config Group の一覧しか出さず、
    # 個々のフラグは `--help=all` にしか無い（GPU 実測 2026-10-29）。
    # 旧実装は 2,134 文字の要約を「完全な help」と誤認し、_has() が全フラグを
    # 偽判定 → MTP / LMO / chat-template が全部落ちた。既知フラグの有無で
    # 「完全な help か」を判定し、ダメなら VERIFIED_FLAGS に倒す。
    def _run_help(arg):
        _p = subprocess.run(["vllm", "serve", arg], capture_output=True,
                            text=True, timeout=180)
        return _p.returncode, (_p.stdout or "") + (_p.stderr or "")


    def _looks_complete(text):
        """個々のフラグまで含む help か。要約だけなら False。"""
        return (len(text) > 5000
                and "--tensor-parallel-size" in text
                and "--max-model-len" in text)


    _rc, HELP = _run_help("--help=all")
    if not _looks_complete(HELP):
        _rc2, _h2 = _run_help("--help")
        if len(_h2) > len(HELP):
            HELP, _rc = _h2, _rc2
    # ヘルプ本文を成果物に残す（フラグ判定が外れたときの唯一の手掛かり）
    save_output("vllm_help.txt", HELP[:20000])
    print(f"  vllm serve --help=all: exit={_rc} {len(HELP)} 文字 "
          f"complete={_looks_complete(HELP)}")
    print(HELP[:1200])
    HELP_OK = (_rc == 0 and _looks_complete(HELP)
               and "Traceback (most recent call last)" not in HELP)
    if not HELP_OK:
        print("⚠ 完全な help が取れない → VERIFIED_FLAGS で判定する")


    # vllm 0.32.0 wheel の arg_utils/cli_args を静的に走査して実在を確認したもの。
    # （詳細 notes/vllm-cpu-findings.md §22）
    VERIFIED_FLAGS = {
        "--trust-remote-code", "--reasoning-parser", "--language-model-only",
        "--limit-mm-per-prompt", "--enable-prefix-caching", "--enforce-eager",
        "--tool-call-parser", "--enable-auto-tool-choice", "--speculative-config",
        "--spec-method", "--kv-cache-dtype", "--enable-log-requests",
        "--chat-template", "--chat-template-content-format", "--default-chat-template-kwargs",
        "--attention-backend", "--max-num-batched-tokens", "--generation-config",
        "--override-generation-config", "--load-format", "--tokenizer",
    }


    def _has(flag):
        """ヘルプ本文の「行頭オプション」として存在するか。

        単純な部分文字列一致だと説明文中の言及で誤判定するため行頭で見る。
        ヘルプが取れない環境では楽観側（True）に倒さず、wheel を静的に走査して
        実在を確認した VERIFIED_FLAGS だけを許可する。楽観側だと 0.32 で消えた
        フラグを付けて未知引数で即死する（過去に実際に踏んだ）。
        """
        if not HELP_OK:
            # 事前に wheel を静的解析して「実在する」と確認済みのものだけ許可する。
            # ここを楽観（True）にすると、0.32 で消えた --chat-template のような
            # フラグを付けて即死する（実測で該当）。
            return flag in VERIFIED_FLAGS
        return re.search(r"^\s*" + re.escape(flag) + r"\b", HELP,
                         re.MULTILINE) is not None


    # 必須フラグ（ここが通らなければ何を試しても無駄）
    CORE_ARGS = [
        "vllm", "serve", STAGE,
        "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(TP_SIZE),
        "--dtype", "half",
        # 量子化の指定は試行ごとに変える（下の PLAN の 7 要素目）。
        # このモデルの quantization_config は format="pack-quantized"（= compressed-tensors）
        # であり、awq 固定は誤り。auto（指定なし）を第一候補にする。
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--kv-cache-dtype", KV_CACHE_DTYPE,
    ]

    # 任意フラグ。バージョンによって引数値が違うことがあるので、
    # 後半の試行では落として「フラグが原因の失敗」を切り分ける。
    OPT_ARGS = []
    if _has("--trust-remote-code"):
        OPT_ARGS.append("--trust-remote-code")
    # 0.32 では --disable-log-requests が廃止され --enable-log-requests に反転。
    # 省略しても害がないので、存在した場合だけ付ける（誤指定で試行を落とさない）。
    if _has("--disable-log-requests"):
        OPT_ARGS.append("--disable-log-requests")
    if _has("--reasoning-parser"):
        # Qwen3 系の思考タグを reasoning_content に分離する
        OPT_ARGS += ["--reasoning-parser", "qwen3"]
    if _has("--enable-auto-tool-choice") and _has("--tool-call-parser"):
        OPT_ARGS += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]

    # ---- 試行計画（上から順に）----
    # 各要素: (ラベル, speculative config, max-model-len, 追加env, 任意フラグを使うか,
    #          追加引数, 量子化引数)
    # 量子化は「指定なし（auto）」を第一候補。config.json の quantization_config は
    # format="pack-quantized" で、AWQ 形式ではない（cfgdeep 実測）。
    SPEC = None
    if MTP_ENABLED and _has("--speculative-config"):
        SPEC = {"method": MTP_METHOD, "num_speculative_tokens": MTP_TOKENS}
    else:
        print("ℹ --speculative-config 非対応 or MTP=0 → MTP なしで起動")

    # draft を最小ディレクトリに向ける（17.41 GiB の二重読みを回避）。
    # 失敗しても次の段が draft dir なしで再試行するので、ここは楽観で良い。
    SPEC_DIR = None
    if SPEC:
        _dn = _stage_draft()
        if _dn:
            SPEC_DIR = {**SPEC, "model": DRAFT_DIR}
            print(f"  draft dir : {DRAFT_DIR}（symlink {_dn} 個）")
        else:
            print("ℹ mtp ファイルが無い → draft は target と同じ dir を使う")

    # 注意: 0.32 に V0 エンジンは無く、VLLM_USE_V1 / VLLM_ATTENTION_BACKEND は
    # envs.py に定義が無い（= 未知 env として警告されるだけ。設定しても無効）。
    # T4(sm75) では FlashAttention backend が capability>=8.0 を要求して弾かれ、
    # 既定で TRITON_ATTN が選ばれる（triton_attn.supports_compute_capability=True）。
    # よってフォールバック段は「明示的に TRITON_ATTN を指定する」に置き換える。

    # 視覚塔を落としてテキスト専用で起動する（全段の既定）。
    #  - テキスト配信しかしないので不要。0.86 GiB と mm profiling の VRAM を節約。
    #  - 実装は StageMissingLayer で子モジュール登録されないため未使用キー警告も出ない
    #    （models/utils.py:756 "Don't register this as a child module in order to
    #      avoid missing keys when loading weights"）。
    # なお quantization_config.ignore の `model.visual.*` は「prefix 文字列」と
    # 完全一致で照合される（should_ignore_layer → is_equal_or_regex_match）。
    # 視覚塔の prefix は `model.visual...` なので一致する（keymap の key 比較では
    # 0 件だったが、判定に使うのは key ではなく prefix）。`re:.*mtp.*` も同様に
    # draft の prefix `mtp.*` に一致する。
    LMO = ["--language-model-only"] if _has("--language-model-only") else []
    if not LMO:
        print("ℹ --language-model-only 非対応 → 視覚塔ありで起動")

    AUTO_Q = []                                    # 指定なし（config から自動判定）
    CT_Q = ["--quantization", "compressed-tensors"]  # 明示（pack-quantized）
    # --quantization awq / auto は config の compressed-tensors と不一致で
    # ModelConfig の検証に落ちる（cfgresolve 実測）。試行段からは外した。

    STARTUP_BUDGET = int(os.environ.get("VLLM_STARTUP_BUDGET", "4200"))   # 70 分
    ATTEMPT_TIMEOUT = int(os.environ.get("VLLM_ATTEMPT_TIMEOUT", "4200"))  # 1 試行に総予算を渡す
    # 10/31 のリセット後（30h）前提の値。install は wheelhouse Dataset で
    # 60〜120s になる見込み（未実測）。短い予算で回すときは環境変数で下げる。
    # 注意: 残クォータが 30 分未満のときにこの設定で push しないこと。
    #
    # **なぜ 1800/900 をやめたか（2026-11-01 実測）**
    # `baseline` が 2 試行とも「切られた」だけで、壊れてはいなかった:
    #   attempt 1 … 900 秒（ATTEMPT_TIMEOUT）で終了、重みロードの途中
    #   attempt 2 … 806 秒（残予算でクランプ）で SIGTERM、
    #               08:07:53 開始 → 08:14:32 に embed_tokens まで到達、
    #               その後 `Loading weights took` が出ないまま打ち切り
    # 同じ回の vLLM ログは `Checkpoint size: 17.41 GiB / Available RAM: 5.85 GiB`
    # で prefetch を切り、GCSFUSE からの読みが 20 MiB/s 前後まで落ちていた。
    # 17.41 GiB ÷ 20 MiB/s ≒ 900 秒。**重みだけで 1 試行の枠を使い切る**ので、
    # そのあとの profile run（373 秒）に絶対に届かない。
    # 実測の幅は 103 MiB/s（out2g: 18.21 GiB / 181s）〜 20 MiB/s と 5 倍ある。
    # 予算は**速い回**に合わせると遅い回が必ず落ちるので、遅い側に合わせる。
    #
    # **なぜ 2700 でも足りなかったか（2026-11-01 の v3 実測）**
    # 2700 に上げた回は「切られる」のではなく、**最後の層まで届いてから**切られた:
    #   t=250s   attempt 1 開始（install は 222s で完了していた）
    #   t=2890s  layers.61 の最後のテンソル（= 全 62 層ぶんの読み込みが終わった）
    #   t=2722s  ← ここで予算超過を判定し、残り 8 段を打ち切り
    # つまり**重みだけで約 2,640 秒**。15.5 GiB を TP=2 の 2 ワーカーが
    # それぞれ読むので実 I/O は約 31 GB、実効 11.8 MiB/s。まだ profile run
    # （373 秒）が残るので、2,640 + 373 ≒ 3,013 秒が下限。2,700 では 300 秒足りない。
    # ホスト RAM は `avail=25,071 free=437 cache=23,455 MiB`、ワーカー RSS が
    # 12.1 GB × 2。空きが 500 MiB を割ったまま読み続けており、ページキャッシュが
    # 追い出されて GCSFUSE を読み直している疑いがある（→ notes §20）。
    # 4,200 は 3,013 秒に対して約 1,180 秒の余裕。ノードが 1.4 倍遅くても届く。
    #
    # `ATTEMPT_TIMEOUT` を総予算と同値にしてあるのは、「遅い」失敗と
    # 「非対応」失敗を区別するため。非対応は即座に落ちる（SGLang の sm80 など）ので
    # 予算をほとんど食わず、次の段に回せる。遅いだけの試行を 15 分で切ると
    # **必ず落ちる**うえ、次段も同じ理由で落ちる。

    # 重みロードの prefetch（`notes/vllm-startup-speedup.md` §22）。
    # vLLM 0.32 の `_prefetch_all_checkpoints` は `sorted_files[rank::world_size]`
    # で**rank ごとにファイルを分担**して page cache に載せる。TP=2 でも総読みが
    # チェックポイント 1 回分になる計算（いまは 2 ワーカーが各自 mmap して 2 回読む）。
    # ただし既定のスレッド数 8 は CPU 実測で一番遅い形（8 本 2.3 MiB/s / 1 本 110 MiB/s）。
    # **env が空なら 1 バイトも足さない**（既存の生成物と挙動を変えない）。
    _PF_STRATEGY = os.environ.get("VLLM_SAFETENSORS_STRATEGY", "").strip()
    _PF_THREADS = os.environ.get("VLLM_SAFETENSORS_PREFETCH_THREADS", "").strip()
    LOAD_ARGS = []
    if _PF_STRATEGY:
        if _has("--safetensors-load-strategy"):
            LOAD_ARGS += ["--safetensors-load-strategy", _PF_STRATEGY]
            # スレッド数は prefetch のときだけ意味がある（lazy/eager では無視される）
            if _PF_THREADS and _has("--safetensors-prefetch-num-threads"):
                LOAD_ARGS += ["--safetensors-prefetch-num-threads", _PF_THREADS]
        else:
            # 黙って落とすと「prefetch は効かなかった」という偽陰性になる
            print("⚠ --safetensors-load-strategy 非対応 → 重み prefetch は付けない")

    # vLLM 側が prefetch を**辞退した**ときの文言（`weight_utils.py` の warning）。
    # チェックポイント 18.21 GiB に対して RAM 32 GB は余裕が無く、同じコードでも
    # 「載せる」ノードと「辞退する」ノードがある（実測 2026-11-02: 片方 1,081 秒 /
    # 片方 47 秒per層）。辞退した回を同じ表に混ぜると施策の効果を読み違えるので、
    # 見つけたら記録して警告を出す。**辞退しても起動は続く**（遅い経路に落ちるだけ）。
    _PF_GUARD = "exceeds 90% of available RAM"
    _PF_GUARD_HIT = ""

    PLAN = []
    # 予算 1500s で 1 試行 600s。MTP を 2 段続けると失敗時に予算を食い潰すので、
    # 「MTP+draft dir」→ 失敗したら MTP を一旦あきらめて素の起動を優先する。
    if SPEC_DIR:
        PLAN.append(("mtp+dd", SPEC_DIR, MAXLEN_CANDIDATES[0], {}, True, LMO, AUTO_Q))
    # ここからは任意フラグを落として最小構成で試す
    PLAN.append(("plain", None, MAXLEN_CANDIDATES[0], {}, False, LMO, AUTO_Q))
    PLAN.append(("plain", None, MAXLEN_CANDIDATES[1], {}, False, LMO, AUTO_Q))
    PLAN.append(("plain", None, 16384, {}, False, LMO, AUTO_Q))
    # 2026-10-29 GPU 実測: 32k は torch.compile(61s) 完了後に
    # "No available shared memory broadcast block found" を 60s 間隔で出し続け、
    # 420s で打ち切った（= ハング or CUDA graph capture/ダミー実行が異常に遅い）。
    # eager は graph capture を丸ごと飛ばすので、仮説の切り分けを最優先で入れる。
    PLAN.insert(1, ("eager", None, MAXLEN_CANDIDATES[0], {},
                    False, LMO + ["--enforce-eager"], AUTO_Q))
    # 量子化の明示指定（auto が外した場合）
    PLAN.append(("ct", None, 16384, {}, False, LMO, CT_Q))
    # attention backend を明示（auto でも TRITON_ATTN になるが、選択を固定する）
    PLAN.append(("triton", None, 8192, {}, False,
                 LMO + ["--attention-backend", "TRITON_ATTN"], AUTO_Q))
    # （eager は上の insert で 2 番目に移した。重複させない）
    # MTP の再挑戦（draft dir 無し = 17 GiB を読み直すので最後に置く）
    if SPEC:
        PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[2], {}, True, LMO, AUTO_Q))
    # language-model-only が原因で落ちた場合の退避。視覚塔は積むが modality を 0 に
    # して実体化させない（get_limit_per_prompt=0 → 同じく StageMissingLayer 経路）。
    PLAN.append(("vl0", None, 8192, {}, False,
                 ["--limit-mm-per-prompt", '{"image":0,"video":0}'], AUTO_Q))
    assert all(len(p) == 7 for p in PLAN), "PLAN の要素数が 7 でない"

    # prefetch は「読み込み段」の話なので、全試行に同じフラグを足す。
    # 段ごとに変えると、効いたときにどの段のおかげか分からなくなる（v14 の轍）。
    if LOAD_ARGS:
        PLAN = [(_l, _s, _m, _e, _o, _f + LOAD_ARGS, _q)
                for (_l, _s, _m, _e, _o, _f, _q) in PLAN]
        print("ℹ 重み prefetch: " + " ".join(LOAD_ARGS))

    # 予算が短いとき（< 900s）は「最も軽い仮説検証」を先頭に持ってくる。
    # 実測: 27B の重みロードだけで約 250s、torch.compile が 61s。
    # 64K+MTP+グラフを試す余裕は無いので、eager（graph capture 無し）で
    # 「そもそも起動できるのか」を先に確定させる。
    if STARTUP_BUDGET < 900:
        PLAN.sort(key=lambda _p: 0 if _p[0] == "eager" else 1)
        print("ℹ 起動予算が短い → eager を先頭に並べ替え")


    def _gpu_used_mib():
        """全 GPU の使用 VRAM 合計（MiB）。取れなければ -1。"""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout
            return sum(int(x) for x in re.findall(r"[0-9]+", out))
        except Exception:
            return -1


    def _gpu_pids():
        """GPU を掴んでいる PID をドライバに聞く（`ps` より確実）。

        vLLM は `setproctitle` でプロセス名を `VLLM::Worker_TP0` のように書き換える。
        名前で殺しに行くと**綴りを外した瞬間に静かに失敗**する（実測 2026-11-01:
        `VLLM::WorkerTP` は `VLLM::Worker_TP0` に一致せず、18,194 MiB を掴んだまま
        5 試行を無駄にした）。ドライバに PID を聞けば名前は関係ない。
        """
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout
            return [int(x) for x in re.findall(r"[0-9]+", out)]
        except Exception:
            return []


    def _free_vram(limit_mib=2000, tries=18):
        """VRAM が実際に空くまで落とし続け、最後の使用量（MiB）を返す。

        「殺したつもり」で次の試行へ進むと、失敗理由が必ず
        `Free memory on device cuda:0 ... is less than desired GPU memory utilization`
        になり、原因（残骸）ではなく**症状**を追いかけることになる。
        """
        for i in range(tries):
            _u = _gpu_used_mib()
            if 0 <= _u < limit_mib:
                return _u
            if i in (0, 3, 8):        # 毎回ではなく段階的に掃除する
                for _pat in ("VLLM::Worker", "VLLM::EngineCore", "vllm serve",
                             "vllm.entrypoints"):
                    subprocess.run(["pkill", "-9", "-f", _pat],
                                   capture_output=True)
                for _pid in _gpu_pids():
                    if _pid in (os.getpid(), os.getppid()):
                        continue      # 自分自身は殺さない
                    try:
                        os.kill(_pid, signal.SIGKILL)
                        print(f"  🔫 GPU を掴んでいる PID {_pid} を停止")
                    except OSError:
                        pass
            time.sleep(5)
        return _gpu_used_mib()


    def _mem_info():
        """`/proc/meminfo` の主要値（MiB）。**空きだけ見ても原因は分からない。**

        実測 2026-11-01（v12）: 試行開始の 4 分前に `MemAvailable` が 30,315 MiB
        あったのに、vLLM が重みを読む時点では **5.68 GiB** まで落ちていた。
        その 24 GiB が何なのか（page cache か、anon か、/dev/shm か）を切り分けるには
        `Cached` / `Shmem` / `MemFree` を並べて見る必要がある。片方だけでは
        「誰かが食った」のか「キャッシュが張り付いた」のか決められない。
        """
        want = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem")
        out = {}
        try:
            for ln in open("/proc/meminfo", encoding="utf-8"):
                k = ln.split(":", 1)[0]
                if k in want:
                    out[k] = int(ln.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return out


    def _mem_available_mib():
        """空き RAM（MiB）。重みロードが固まるかの予兆になる。"""
        return _mem_info().get("MemAvailable", -1)


    def _mem_line():
        """1 行にまとめた RAM の内訳。無音区間の原因を後から追うための控え。"""
        m = _mem_info()
        if not m:
            return "RAM: 取得できません"
        return (f"RAM 空き {m.get('MemAvailable', -1)} / 空 {m.get('MemFree', -1)}"
                f" / cache {m.get('Cached', -1)} / shm {m.get('Shmem', -1)} MiB"
                f"（全 {m.get('MemTotal', -1)}）")


    def _top_rss(n=3):
        """RSS 上位 n 件（`名前:MiB`）。24 GiB を誰が持っているかを名前で見る。"""
        rows = []
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/statm", encoding="utf-8") as f:
                        rss = int(f.read().split()[1]) * 4096 // (1024 * 1024)
                    with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                        name = f.read().strip()
                except (OSError, ValueError, IndexError):
                    continue
                if rss:
                    rows.append((rss, f"{name}:{rss}"))
        except OSError:
            return ""
        rows.sort(reverse=True)
        return " ".join(t for _, t in rows[:n])


    # 署名から潰す「揺れる値」。同じ失敗なのに毎回違う文字列が入ると、
    # ラダーの打ち切り判定が効かない。
    _SIG_NOISE = (
        # 先頭に `\b` を付けない: `psm_3e24306e` の `3` の前は `_`（単語文字）なので
        # 単語境界にならず、付けると**ハッシュが消えない**（実測 2026-11-01）。
        (re.compile(r"[0-9a-f]{6,}"), "<hex>"),            # psm_3e24306e / dist_...
        (re.compile(r"pid=\d+"), "pid=<n>"),
        (re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b"), "<ts>"),
        (re.compile(r"\b\d+\.\d+\b"), "<f>"),
    )


    def _sig_of(tail, n=3):
        """失敗の署名。**揺れる値を潰してから**比べる。

        実測 2026-11-01: VRAM 未解放という**同一の理由**で 5 回失敗したのに、
        末尾に毎回違う `{'/psm_3e24306e'}` のハッシュが入るため署名が一致せず、
        ラダーが 5 回続行して約 900 秒を捨てた。原因ではなく**偶然**を比べていた。
        """
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()][-n:]
        s = " | ".join(lines)
        for _pat, _rep in _SIG_NOISE:
            s = _pat.sub(_rep, s)
        return s[:400]


    def _kill(proc):
        """vLLM のプロセス木ごと落とし、VRAM が実際に空くまで待つ。

        2026-10-29 実測: 試行 1 を terminate した直後に試行 2 を起動したところ、
        125 秒で "Engine core initialization failed. See root cause above."
        で即死した（EngineCore / Worker が残って VRAM を掴んでいた）。
        SIGTERM は API サーバーにしか届かず、孫（Worker_TP*）は残る。
        """
        try:
            _pgid = os.getpgid(proc.pid)
            os.killpg(_pgid, signal.SIGTERM)
            proc.wait(timeout=20)
            os.killpg(_pgid, signal.SIGKILL)   # 残った子を確実に落とす
        except Exception:
            pass
        # /dev/shm の残骸を掃除する。SharedMemory が unlink されずに残ると
        # 次の試行の check_shm_free_space で Engine core が即死する
        # （実測: attempt 2/3 が 125 秒で "Failed core proc(s): {}"、
        #   同時に resource_tracker が /psm_* のリークを警告）。
        _left = 0
        try:
            for _fn in os.listdir("/dev/shm"):
                if _fn.startswith(("vllm", "psm_")):
                    try:
                        os.unlink(os.path.join("/dev/shm", _fn))
                        _left += 1
                    except Exception:
                        pass
        except Exception:
            pass
        if _left:
            print(f"  🧽 /dev/shm の残骸 {_left} 件を削除")
        # 解放待ち。ここを飛ばすと次の試行の失敗理由が「前のプロセスの残骸」になり、
        # 切り分けが壊れる。**戻り値（実際に空いたか）を呼び出し側で判定する**。
        _u = _free_vram()
        print(f"  🧹 停止後の VRAM 使用量: {_u} MiB")
        return _u


    def _models(timeout=8):
        """起動済みなら /v1/models の JSON を返す（未起動なら None）。"""
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=timeout
            ) as r:
                return json.loads(r.read())
        except Exception:
            return None


    def _smoke(attempts=2, timeout=90):
        """起動後に 1 回だけ生成させ、初回 forward が通るか確かめる。

        /v1/models が返るだけでは不十分。量子化カーネル（Marlin/Exllama）や
        GDN の Triton カーネルは **推論して初めて** 落ちることがあり、その場合
        サーバーは 500 を返し続ける。ここで落とせば次の段に進める。
        """
        body = json.dumps({
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 16,
            "temperature": 0.7,
            "stream": False,
        }).encode()
        last = ""
        for i in range(attempts):
            req = urllib.request.Request(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                ch = (d.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                txt = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
                return True, (f"tok={(d.get('usage') or {}).get('completion_tokens')} "
                              f"fin={ch.get('finish_reason')} "
                              f"text={txt[:40]!r}")
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:300].decode("utf-8", "replace")
                last = f"HTTP {exc.code}: {detail}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            print(f"    smoke {i+1}/{attempts} NG: {last[:200]}")
            time.sleep(3)
        return False, last


    proc = None
    argv_used = None
    started = False
    # 起動中の進捗を Gist に流す。Kaggle の UI ログは長大で貼りにくく、
    # `kaggle kernels output` は完了まで何も返さないため、外から見える場所が要る。
    _GIST_ID = (os.environ.get("GIST_ID") or GIST_ID_DEFAULT)
    # 書けたかどうかを数える。**黙って落ちるのが一番高い**——2026-11-01 の
    # vllm-playground は Gist を 10/28 から一度も更新しておらず、理由
    # （トークン無し / 401 / 404）はログにも result.json にも残っていなかった。
    # 外からは「何も起きていない」と区別がつかず、起動待ち 30 分を
    # 目視で追うことになる。
    _GIST_STATE = {"gist_id": _GIST_ID, "attempts": 0, "ok": 0, "last_error": ""}
    _GIST_SAID = [False]


    def _gist_status(text):
        """起動状況を Gist（vllm_status.txt）に書く。失敗は**1 回だけ**声に出す。

        毎回出すと 30 秒ごとに同じ行が並んでログが埋まるので、初回だけ警告し、
        以降は `_GIST_STATE` に数える。呼び出し側は戻り値を見ない（進捗書き込みで
        起動を止めない）が、結果は `_gist_summary()` で 1 行にできる。
        """
        _GIST_STATE["attempts"] += 1
        ok, detail = kc_gist_write(_GIST_ID, "vllm_status.txt", text)
        if ok:
            _GIST_STATE["ok"] += 1
            return True
        _GIST_STATE["last_error"] = detail
        if not _GIST_SAID[0]:
            _GIST_SAID[0] = True
            print(f"⚠ Gist 進捗を書けません: {detail}")
            print("   以降は黙って続行し、最後にまとめて出します"
                  "（CPU で確かめるには gist-probe カーネル）")
        return False


    def _gist_summary():
        """書き込みの成否を 1 行にする。0 件なら**それ自体**が異常。"""
        s = _GIST_STATE
        if s["attempts"] == 0:
            return "ℹ Gist 進捗: 一度も呼ばれていない（待機ループに到達していない）"
        if s["ok"] == 0:
            return (f"⚠ Gist 進捗: 0/{s['attempts']} 成功 — 外から見えていない。"
                    f" 理由: {s['last_error']}")
        return f"ℹ Gist 進捗: {s['ok']}/{s['attempts']} 成功"


    # 待機ループへ入る**前に** 1 回だけ往復を試す。理由は 2 つ:
    #  - ログの先頭は残る。30 分待って落ちた回でも「トークンが無かった」が読める
    #  - 進捗ファイル（vllm_status.txt）とは**別名**にする。前回の最終状態を
    #    起動直後に消さないため（Gist 側に「最後に通った時刻」も残る）
    _GIST_SELFTEST = kc_gist_selftest(_GIST_ID, "vllm_selftest.txt")
    if _GIST_SELFTEST["ok"]:
        print(f"Gist 往復: OK（{_GIST_SELFTEST['token_detail']}）")
    else:
        _why = (_GIST_SELFTEST.get("write_detail")
                or _GIST_SELFTEST.get("read_detail") or "理由不明")
        print(f"Gist 往復: NG — {_why}")
        print(f"  トークン解決: {_GIST_SELFTEST.get('token_detail', '')}")
    # GPU クォータは有限。ラダーを全部試すと 1 時間超になり、途中で打ち切られると
    # 成果物が残らない。総予算を切って、必ずログを残して終わるようにする。
    # クォータ残 39 分（2,395s）で pip install に 5 分前後かかる前提の値。
    # 25 分使うと公開できたとしても数分で打ち切られるため短くしてある。
    # 2026-10-29 実測: 27B ハイブリッド（GDN/Mamba）を T4 x2 で起動すると、
    # torch.compile 61s の後に "No available shared memory broadcast block" が
    # 60s 間隔で出続け、420s では足りなかった（ハングか graph capture が異常に遅い）。
    # **クォータ 30h のリセット後は 1 試行 15 分で回す**。
    # 10/28 の残 39 分では 720/420 に落として使った（その値では足りなかった）。
    _attempts = []
    _prev_sig = None
    _t_start = time.time()
    tail = ""          # 予算切れで 1 度も走らなかった場合に備えて初期化
    last_log = ""
    smoke_fail = ""    # 起動はしたが生成で落ちた場合の HTTP 応答（ログに出ない）
    save_output("vllm_attempts.txt", "起動試行ログ（追記保存）\n")

    # 空き RAM は「重みロードが固まるか」の予兆になる。実測 2026-11-01:
    # `Available RAM: 6.12 GiB` / checkpoint 18.21 GiB のとき、GCSFUSE 上の
    # safetensors を mmap で読む工程が **840 秒間 1 行も進まなかった**
    # （ログ最終行は `Loading safetensors checkpoint shards: 50%` のまま）。
    # ここで気づけないと「起動が遅い」ではなく「原因不明のハング」に見える。
    _mem = _mem_available_mib()
    print(f"🧠 {_mem_line()}")
    print(f"   上位 RSS: {_top_rss(5)}")
    if 0 <= _mem < 8192:
        print(f"  ⚠ 空き RAM が少ない（{_mem} MiB < 8192）。重みロードで詰まる可能性。"
              f" 症状は「ログの最終行から無進捗のまま試行上限まで待つ」")


    def _weights_gib():
        """重み（*.safetensors）の合計 GiB。prefetch の可否を決める分母になる。"""
        total = 0
        try:
            for fn in os.listdir(MODEL_SRC):
                if fn.endswith(".safetensors"):
                    total += os.path.getsize(os.path.join(MODEL_SRC, fn))
        except OSError:
            pass
        return round(total / 2**30, 2)


    def _drop_caches():
        """clean page cache を捨てる。**prefetch を復活させるためだけ**にやる。

        vLLM は「checkpoint が空き RAM の 90% を超える」と auto-prefetch を切り、
        GCSFUSE 上の safetensors を頁単位で mmap する遅い経路に落ちる。v10 は
        同じ手順で 133 秒だったので、差は起動時点の `MemAvailable` だけ。
        `Cached` が空きを圧迫しているなら、ここで捨てれば条件を満たし直せる。

        **ただし Kaggle では書けない**（実測 2026-11-01 / `vllm-io-bench` v2）:
        `OSError: [Errno 30] Read-only file system: '/proc/sys/vm/drop_caches'`。
        それでも呼ぶのは、環境が変わったときに 1 行で気づけるようにするため。
        結果（OK/NG）は必ず表示する。
        """
        before = _mem_info()
        ok = False
        try:
            with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
                f.write("1\n")          # 1 = page cache のみ（dentry/inode は残す）
            ok = True
        except OSError as e:
            return False, f"{type(e).__name__}: {e}"
        after = _mem_info()
        return ok, (f"{before.get('MemAvailable', -1)} → "
                    f"{after.get('MemAvailable', -1)} MiB"
                    f"（cache {before.get('Cached', -1)} → {after.get('Cached', -1)}）")


    _WEIGHTS_GIB = _weights_gib()
    # vLLM のしきい値（90%）に少し余裕を足す。ここを割ると prefetch が切られる。
    _NEED_MIB = int(_WEIGHTS_GIB * 1024 / 0.9 * 1.05) if _WEIGHTS_GIB else 0
    print(f"🧮 重み {_WEIGHTS_GIB} GiB / prefetch に必要な空き {_NEED_MIB} MiB")


    def _read_range(path, off, nbytes, threads=1, chunk_mib=32):
        """`off` から `nbytes` を `threads` 本の `pread` で読む（先読みの実体）。

        同じ fd を複数スレッドで `pread` するのは安全（読み取り専用・位置指定）。
        戻り値は読めたバイト数（**足りない分は黙って捨てない**。呼び側が比べる）。
        """
        chunk = chunk_mib * 1024 * 1024
        total = 0
        fd = os.open(path, os.O_RDONLY)
        try:
            def _one(o):
                n = min(chunk, off + nbytes - o)
                if n <= 0:
                    return 0
                try:
                    return len(os.pread(fd, n, o))
                except OSError:
                    return 0

            with _pool(max(1, threads)) as ex:
                for got in ex.map(_one, range(off, off + nbytes, chunk)):
                    total += got
        finally:
            os.close(fd)
        return total


    def _prewarm_weights(threads=1, chunk_mib=32, budget_s=300, batch_mib=512):
        """重みを先に読んで page cache に載せる（**時間予算つき**）。

        実測 2026-11-01（`vllm-io-bench` v2・CPU・GPU 0 秒）:

        | 読み方 | 実測 |
        |---|---|
        | GCSFUSE 1 本（64 MiB バッファ） | **110〜130 MiB/s** |
        | GCSFUSE 2/4/8/16 本（32 MiB チャンク） | **2.2〜2.4 MiB/s** ← 13〜50 倍遅い |
        | page cache に載った後の mmap | **5,677 MiB/s** |

        **並列にすると壊滅的に遅い。** 1 GiB を 8 本で読むと 449 秒（1 本なら 33 秒）。
        v10 の重み 133 秒（140 MiB/s）を「並列の証拠」と読んだのは**誤り**で、
        あれは単に回線が速い回だっただけ（同じ 1 本読みでも 43.8 MiB/s の回がある）。

        → **既定は 1 本**。並列は `KC_PREWARM_THREADS` で明示したときだけ試し、
        必ず 1 本と実測で比べて速い方を採る（決め打ちにしない）。

        `budget_s` を超える見込みなら**始めない**（部分的な先読みは「残りが遅い」
        という症状を防げない）。走り出したあとで速度が崩れた場合も、予算で打ち切る。
        """
        files = []
        try:
            for fn in sorted(os.listdir(MODEL_SRC)):
                if fn.endswith(".safetensors"):
                    files.append(os.path.join(MODEL_SRC, fn))
        except OSError:
            return None
        if not files:
            return None

        t0 = time.time()
        big = max(files, key=os.path.getsize)
        big_size = os.path.getsize(big)
        probe_n = min(64 * 2**20, big_size)
        probe_off = max(0, big_size - probe_n)      # 末尾から（本読みと窓をずらす）
        cands = [1]
        if int(threads) > 1:
            cands.append(max(1, int(threads)))
        probe = {}
        for n in cands:
            _t = time.time()
            got = _read_range(big, probe_off, probe_n, threads=n, chunk_mib=chunk_mib)
            _dt = time.time() - _t
            probe[n] = round(got / 2**20 / _dt, 1) if _dt else None
        best = max(probe, key=lambda n: probe[n] or 0)
        rate = probe[best] or 0
        total_bytes = sum(os.path.getsize(p) for p in files)
        projected = (total_bytes / 2**20 / rate) if rate else None

        # 予算に収まらないと分かっているなら**始めない**。部分的な先読みは
        # 「残りが遅い」という症状を防げないので、時間だけ払って何も変わらない。
        # 1.5 倍の余裕は、probe が 64 MiB と短く、本読みの実効速度とずれるため。
        if projected is not None and projected > budget_s * 1.5:
            return {"skipped": "over budget", "probe": probe, "threads": best,
                    "mib_s": rate, "projected_s": round(projected, 1),
                    "seconds": round(time.time() - t0, 1)}

        total = 0
        aborted = False
        for path in files:
            size = os.path.getsize(path)
            off = 0
            while off < size:
                if time.time() - t0 > budget_s:
                    aborted = True
                    break
                n = min(batch_mib * 2**20, size - off)
                total += _read_range(path, off, n, threads=best, chunk_mib=chunk_mib)
                off += n
            if aborted:
                break
        dt = time.time() - t0
        return {"bytes": total, "seconds": round(dt, 1),
                "mib_s": round(total / 2**20 / dt, 1) if dt else None,
                "threads": best, "probe": probe, "aborted": aborted,
                "budget_s": budget_s, "projected_s": projected}


    def _pool(n):
        """スレッドプール。**I/O 待ちなので GIL は関係ない**（`os.pread` は解放する）。"""
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=n)


    def _ram_trace(stop, path, every=5.0):
        """起動中の RAM を 5 秒ごとに残す。**24 GiB が消える瞬間**を捕まえるため。

        実測 2026-11-01（v12）: 試行開始の 4 分前に `MemAvailable` 30,315 MiB、
        vLLM が重みを読む時点で 5.68 GiB。差分 24 GiB の正体が分からないまま
        対策を積むのは順序が逆。5 秒粒度なら「どのプロセスが、いつ、どれだけ」が
        そのまま読める。
        """
        t0 = time.time()
        try:
            with open(path, "a", encoding="utf-8") as f:
                while not stop.is_set():
                    m = _mem_info()
                    f.write(f"{time.time() - t0:7.1f}s avail={m.get('MemAvailable', -1)}"
                            f" free={m.get('MemFree', -1)} cache={m.get('Cached', -1)}"
                            f" shm={m.get('Shmem', -1)} top={_top_rss(3)}\n")
                    f.flush()
                    stop.wait(every)
        except OSError:
            pass


    # ---- Triton キャッシュ（起動の 54% を占める無音の正体）----
    # v14 実測（2026-11-01, T4, eager）の時系列:
    #   03:52:25  Using LBNHC KV cache layout            ← 82s 地点
    #   ……373 秒の無音（shm_broadcast が 60 秒ごとに 6 回）……
    #   03:58:38  Available KV cache memory: 3.45 GiB
    #   03:58:38  Warming up Qwen GDN Triton kernels     ← ここは 3 秒
    #   03:58:41  init engine (profile, create kv cache, warmup model) took 376.24s
    # つまり **名前の付いた warmup は 3 秒**で、コストはその手前
    # （`determine_available_memory` の profile run）にある。ram_trace を見ると
    # その間ワーカーの RSS が 6,987 → 8,361 MiB と 11 MiB/s で増え続けており、
    # ハングではなく**コンパイルしている**。成果物は `~/.triton/cache` に落ちる。
    #
    # したがって「前回の `~/.triton` を持ち込む」が筋。ただし **キーが違えば黙って
    # ミスする**（Triton のキーは Triton の版・対象 sm・ソースハッシュを含む）ので、
    # 判定は「ファイルがある」ではなく **「エントリが増えなかった」**にする。
    # 増えた数が残っていれば、373 秒が戻ったときに「ミスした」のか「別の原因」かを
    # 区別できる（この区別が無いまま対策を積むと、v14 の先読みと同じことになる）。
    #
    # パスは `TRITON_CACHE_DIR` で**明示する**。既定は Triton の版によって
    # `~/.triton/cache` と `~/.cache/triton` のどちらにもなり得て、復元側（01 の
    # tar）と食い違うと「復元したのに 1 件も当たらない」になる。
    _TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser(
        "~/.triton/cache")


    def _uniq_dirs(paths):
        """実体で重複を落とす。`~/.triton` が別名で 2 回入ると tar が二重になる。"""
        out, seen = [], set()
        for p in paths:
            try:
                real = os.path.realpath(p)
            except OSError:
                real = p
            if real not in seen:
                seen.add(real)
                out.append(p)
        return tuple(out)


    _TRITON_DIRS = _uniq_dirs((
        os.path.dirname(_TRITON_CACHE.rstrip("/")) or _TRITON_CACHE,
        os.path.expanduser("~/.triton"),
        "/root/.triton",
        "/root/.cache/triton",
    ))


    def _triton_snapshot():
        """Triton キャッシュの「いまの顔ぶれ」を返す（比較用）。

        名前の**集合**を返すのは、再利用されたかを判定するため。「ファイルがある」
        だけでは、前回と別のカーネルをコンパイルした（＝当たっていない）場合を
        区別できない。1 カーネル 1 ディレクトリなので直下の名前で足りる。
        """
        names, where = set(), ""
        cands = (_TRITON_CACHE,) + tuple(os.path.join(x, "cache")
                                         for x in _TRITON_DIRS)
        for d in cands:
            if not os.path.isdir(d):
                continue
            where = where or d
            try:
                names.update(e.name for e in os.scandir(d))
            except OSError:
                pass
        return {"dir": where, "names": names}


    def _triton_report(before, after):
        """比較結果を **JSON に書ける形**にする（集合は落とす。timings が壊れる）。"""
        n_before, n_after = len(before["names"]), len(after["names"])
        new = len(after["names"] - before["names"])
        return {"dir": after["dir"] or before["dir"],
                "entries_before": n_before,
                "entries_after": n_after,
                "new_entries": new,
                # 今回使ったエントリのうち、前回からあった割合＝当たった割合。
                "reused_pct": (round(100.0 * (n_after - new) / n_after, 1)
                               if n_after else None)}


    def _capture_warm_cache(limit_bytes=3e9):
        """`~/.triton` を tar にして成果物へ置く（次回の Dataset 用）。

        既存の `engine_cleanup()` も同じ tar を作るが、あれはキープアライブの
        **終了時**に走るため、v14 では一度も実行されなかった（ログにキャッシュの
        行が 1 本も無く、`warm-cache.tar` も成果物に無い）。起動が通った時点で
        キャッシュは揃っているので、ここで取る方が確実。

        3 GB を超えたら諦める（転送と容量の無駄）。**失敗しても本処理は止めない**
        — 止めると「キャッシュが取れなかっただけで起動できた回」を失う。
        """
        if os.environ.get("KC_CAPTURE_CACHE", "1") == "0":
            return {"skipped": "KC_CAPTURE_CACHE=0"}
        try:
            import tarfile
            dirs = [d for d in _TRITON_DIRS if os.path.isdir(d)]
            if not dirs:
                return {"skipped": "no triton dir"}
            size = 0
            for d in dirs:
                for root_, _dirs, files in os.walk(d):
                    for f in files:
                        try:
                            size += os.path.getsize(os.path.join(root_, f))
                        except OSError:
                            pass
                    if size > limit_bytes:
                        break
                if size > limit_bytes:
                    break
            if not size:
                return {"skipped": "empty", "dirs": dirs}
            if size > limit_bytes:
                return {"skipped": "too big", "bytes": size, "dirs": dirs}
            out = os.path.join(OUT_DIRS[0], "warm-cache.tar")
            os.makedirs(OUT_DIRS[0], exist_ok=True)
            t0 = time.time()
            with tarfile.open(out, "w") as tf:
                for d in dirs:
                    # 復元側（01 の `extractall("/")`）に合わせて `/` からの相対で入れる。
                    tf.add(d, arcname=os.path.relpath(d, "/"))
            return {"path": out, "bytes": os.path.getsize(out),
                    "source_bytes": size, "seconds": round(time.time() - t0, 1),
                    "dirs": dirs}
        except Exception as e:                      # noqa: BLE001
            print(f"⚠ キャッシュ退避をスキップ: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {e}"}


    # ---- 先読み（既定 OFF。2026-11-01 に既定を反転した）----
    # **先読みは「152.5 秒の純コスト」では済まない。後続の重みロードまで 4.7 倍遅くする。**
    #
    # 実測の並び（すべてログの時刻差）:
    #
    #   out2g（10-31, 先読みなし）: 重みロード 約 181s → 起動 642.4s → **31.7 t/s**
    #   v14 （11-01, 先読みあり）: 先読み 152.5s → 試行 1 の重みロード **705.00s**
    #                              → 921s で打ち切り → eager に落ちて 4.5 t/s
    #
    # 同じマシン・同じ重み（18.21 GiB）で 181s と 705s。**先読み自身が測った速度は
    # 122.2 MiB/s** なので、素直に読めば 18.21 GiB ÷ 122 MiB/s ≈ 150s。181s はこれと
    # 一致し、705s（26 MiB/s）は**機械自身の実力の 4.7 倍遅い**。つまり読み出し中に
    # 何かが競合している。
    #
    # 機構（推定・ただし数字は整合する）: 先読みが page cache を 25 GiB まで満たす →
    # ワーカーが 8.88 GiB × 2 を anonymous で確保 → カーネルが page cache を 17.8 GiB
    # 追い出す（＝先読みした重みが消える）→ 試行は cold read に戻り、しかも reclaim と
    # 競合する。v14 の `ram_trace.txt` は `Cached` が 20〜23 GiB のまま推移しており、
    # この押し出しと矛盾しない。
    #
    # したがって既定は **OFF**。`KC_PREWARM=1` で明示したときだけ試す（A/B 用の口は残す）。
    _PREWARM = None
    if _WEIGHTS_GIB and os.environ.get("KC_PREWARM", "0") != "0":
        # 既定は **1 本**。並列は GCSFUSE で 13〜50 倍遅い（実測 2026-11-01）。
        _pw_threads = int(os.environ.get("KC_PREWARM_THREADS", "1"))
        _pw_budget = int(os.environ.get("KC_PREWARM_BUDGET", "300"))
        print(f"🔥 重みを先に読む（{_WEIGHTS_GIB} GiB / {_pw_threads} 本 / "
              f"予算 {_pw_budget}s）…")
        _PREWARM = _prewarm_weights(threads=_pw_threads, budget_s=_pw_budget)
        if _PREWARM and _PREWARM.get("skipped"):
            print(f"   ⏭ 先読みしない（{_PREWARM['skipped']} / "
                  f"{_PREWARM.get('mib_s')} MiB/s / 見込み "
                  f"{_PREWARM.get('projected_s')}s / probe {_PREWARM['probe']}）")
        elif _PREWARM:
            print(f"   {_PREWARM['mib_s']} MiB/s / {_PREWARM['seconds']}s"
                  f" / {_PREWARM['threads']} 本 / probe {_PREWARM['probe']}"
                  f"{' / ⚠ 予算切れ（部分）' if _PREWARM['aborted'] else ''}")
        kc_save_timings({"prewarm": _PREWARM})

    # キャッシュの「起動前の顔ぶれ」を控える。試行後に差分を取れば
    # **当たったかどうか**が分かる（`new_entries` が多い＝作り直した＝ミス）。
    _TRITON_BEFORE = _triton_snapshot()
    print(f"🧠 Triton キャッシュ: {len(_TRITON_BEFORE['names'])} エントリ"
          f"（{_TRITON_BEFORE['dir'] or 'なし'} / 環境変数 {_TRITON_CACHE}）")
    kc_save_timings({"triton_cache_before": {
        "entries": len(_TRITON_BEFORE["names"]), "dir": _TRITON_BEFORE["dir"]}})

    # 先読みは「起動の準備」であって起動試行ではない。予算（STARTUP_BUDGET）は
    # 試行に使う。ここを混ぜると、先読みが 400 秒かかった回に試行の時間が
    # 残らず「先読みしたせいで起動できなかった」という最悪の形になる。
    _t_start = time.time()

    # 起動中の RAM を 5 秒ごとに記録する（別スレッド）。ポーリングは 60 秒なので、
    # そのままだと「30 GiB → 5.7 GiB」が 1 区間に潰れて誰も特定できない。
    RAM_TRACE = "/kaggle/working/ram_trace.txt"
    _RAM_STOP = threading.Event()
    _RAM_THREAD = threading.Thread(target=_ram_trace, args=(_RAM_STOP, RAM_TRACE),
                                   daemon=True)
    _RAM_THREAD.start()


    for idx, (label, spec, maxlen, env_extra, use_opt, extra_args,
              quant_args) in enumerate(PLAN, 1):
        _elapsed = time.time() - _t_start
        if _elapsed > STARTUP_BUDGET:
            print(f"⏱ 起動予算 {STARTUP_BUDGET}s 超過（{int(_elapsed)}s）"
                  f" → 残り {len(PLAN) - idx + 1} 試行を打ち切り")
            break
        n_staged = _stage_model(with_mtp=spec is not None)
        cmd = CORE_ARGS + (OPT_ARGS if use_opt else [])
        cmd += ["--max-model-len", str(maxlen)]
        if spec:
            cmd += ["--speculative-config", json.dumps(spec)]
        cmd += extra_args
        cmd += quant_args
        # tokenizer_config.json に chat_template が無いので明示指定する。
        # 0.32 でも --chat-template は FrontendArgs のフィールドとして実在
        # （entrypoints/launchers/cli_args.py:86。dataclass 経由で自動登録されるため
        #  add_argument の grep では見つからない）。存在しなければ付けない。
        _tpl = os.path.join(STAGE, "chat_template.jinja")
        if os.path.isfile(_tpl) and _has("--chat-template"):
            cmd += ["--chat-template", _tpl]
        log_path = f"/tmp/vllm_{idx}.log"
        last_log = log_path
        # torch.compile / Triton JIT は CPU バウンド。vLLM は既定で 1 スレッドに
        # 落とす（"Reducing Torch threads from 2 to 1 for serving"）ため明示する。
        # `TRITON_CACHE_DIR` も明示する: Triton の既定は版によって
        # `~/.triton/cache` と `~/.cache/triton` のどちらにもなり得て、復元した
        # ディレクトリと食い違うと**1 件も当たらないまま 373 秒が戻る**。
        env_run = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
                   "TRITON_CACHE_DIR": _TRITON_CACHE,
                   **env_extra}
        print()
        print(f"--- 試行 {idx}/{len(PLAN)}: spec={label} "
              f"max_len={maxlen} env={env_extra or 'default'} "
              f"quant={quant_args[1] if quant_args else 'auto'} "
              f"opt_flags={'on' if use_opt else 'off'} "
              f"staged={n_staged} files ---")
        # 起動の直前に空きを確保し直す。ここを飛ばすと「prefetch 無効の遅い経路」に
        # 入り、重みロードが 800 秒進まないまま試行上限を焼く（実測 2026-11-01）。
        _avail = _mem_available_mib()
        if _NEED_MIB and 0 <= _avail < _NEED_MIB:
            print(f"  🧠 空き {_avail} MiB < 必要 {_NEED_MIB} MiB → prefetch が切られる。"
                  f" page cache を捨てて再測定")
            _ok, _detail = _drop_caches()
            print(f"  🧹 drop_caches: {'OK' if _ok else 'NG'} {_detail}")
            _avail = _mem_available_mib()
            if 0 <= _avail < _NEED_MIB:
                print(f"  ⚠ それでも足りない（{_avail} MiB）。重みロードで詰まる可能性。"
                      f" 症状は「tqdm が 50% で止まったまま無進捗」")
        with open(log_path, "w") as log:
            _pre = _gpu_used_mib()
            if _pre > 3000:
                print(f"  ⚠ 起動前の VRAM 使用量 {_pre} MiB（残骸の疑い）→ 掃除")
                # 掃除した**結果**を見る。ここを見ないと、解放できていないまま
                # 5 試行を「Free memory ... is less than desired GPU memory
                # utilization」で焼き捨てる（実測 2026-11-01: 18,194 MiB が
                # 6 試行ぶん残り、約 900 秒を捨てた）。
                _pre = _free_vram()
                print(f"  🧹 掃除後: {_pre} MiB")
                if _pre > 3000:
                    fail(f"VRAM が解放されません（{_pre} MiB 使用中）。前の実行の"
                         f"残骸か他プロセスが掴んでいます。ここで止めます"
                         f"（起動しても 0.9 × VRAM を確保できず即死する）。",
                         last_log or log_path)
            # start_new_session で独立したプロセスグループにする（_kill で木ごと落とす）
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env_run,
                                    start_new_session=True)
        print(f"🔄 vLLM PID: {proc.pid}  log: {log_path}")

        t0 = time.time()
        # ログの時刻（時:分:秒）と突き合わせるための起動時刻（秒・日内）。
        _lt = time.localtime()
        _t0_wall = _lt.tm_hour * 3600 + _lt.tm_min * 60 + _lt.tm_sec
        # 1 試行の上限は残予算でクランプする（最後の 1 段が予算超過で
        # 中途半端に切られ、ログが残らないのを避ける）。
        _tmo = min(ATTEMPT_TIMEOUT,
                   max(45, int(STARTUP_BUDGET - (t0 - _t_start))))
        _t_print = 0.0
        _sz_prev = 0
        _poll_marks = {}
        _mark_last = "(節目なし)"
        while time.time() - t0 < _tmo:
            d = _models(timeout=3)
            if d and d.get("data"):
                started = True
                break
            if proc.poll() is not None:
                break
            _now = time.time() - t0
            if _now - _t_print >= 60:
                _t_print = _now
                # 「起動待ち Ns (0 models)」だけでは何で詰まっているか分からない。
                # ログ末尾をそのまま出す（GPU 実行を 1 回無駄にしないため）。
                try:
                    _sz = os.path.getsize(log_path)
                    _lns = [ln.strip() for ln in
                            open(log_path, errors="replace").read().splitlines()
                            if ln.strip()]
                except Exception:
                    _sz, _lns = 0, []
                _last_line = _lns[-1][:170] if _lns else "(ログ空)"
                # 末尾 1 行は警告で埋まりやすい。最後に越えた節目を別に出す。
                _hit, _mark = None, "(節目なし)"
                for _ln in reversed(_lns):
                    _h = next((m for m in _MILESTONES if m in _ln), None)
                    if _h:
                        _hit, _mark = _h, _ln[:150]
                        break
                if _hit and _hit not in _poll_marks:
                    # 60 秒分解能の控え。tqdm の行（graph capture）は**ログに時刻が
                    # 無い**ので、ログ時刻だけでは拾えない。ポーリング側で補完する。
                    _poll_marks[_hit] = round(_now, 1)
                if _hit:
                    _mark_last = _mark
                # prefetch を辞退されたかを見る。ここを見ないと「prefetch を付けたのに
                # 遅い」を延々と読み違える（実測 2026-11-02）。ログ全体を毎回走査して
                # いるので追加コストはほぼ無い。
                if LOAD_ARGS and not _PF_GUARD_HIT:
                    for _ln in _lns:
                        if _PF_GUARD in _ln:
                            _PF_GUARD_HIT = _ln.strip()[:300]
                            print("  ⚠ vLLM が prefetch を辞退した（RAM 判定）: "
                                  + _PF_GUARD_HIT)
                            break
                _grow = _sz - _sz_prev
                _sz_prev = _sz
                _gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True).stdout.strip().replace("\n", " ")
                print(f"  ⏳ 起動待ち {int(_now)}s"
                      f"（{len(d.get('data', [])) if d else 0} models）: "
                      f"{_last_line}")
                print(f"     log {_sz // 1024} KiB (+{_grow}) / 節目: {_mark}")
                print(f"     GPU util/mem: {_gpu}")
                # RAM の内訳も毎回残す。重みロードで固まる原因は「空きが少ない」
                # だけでは特定できない（cache なのか anon なのか /dev/shm なのか）。
                print(f"     {_mem_line()} / 上位: {_top_rss()}")
                _gist_status(
                    f"attempt {idx}/{len(PLAN)} label={label} maxlen={maxlen}\n"
                    f"elapsed={int(_now)}s / limit={int(_tmo)}s\n"
                    f"last_log: {_last_line}\n"
                    f"milestone: {_mark}\n"
                    f"{_mem_line()}\n")
            time.sleep(5)

        # **なぜ待つのをやめたか**を残す。これが無いと、予算で切った回も
        # 「起動に失敗しました」と報告され、設定の誤りを疑って何時間も溶かす
        # （実測 2026-11-02: 予算切れを起動失敗と読んで原因究明に 1 時間）。
        if started:
            _why = "started"
        elif proc.poll() is not None:
            _why = "died"
        else:
            _why = "budget"
        _PF_STATE = {"requested": bool(LOAD_ARGS),
                     "args": LOAD_ARGS,
                     "declined": bool(_PF_GUARD_HIT),
                     "declined_line": _PF_GUARD_HIT}
        if _PF_STATE["requested"]:
            print(("⚠ prefetch は辞退された（この回の重みロードは遅い経路）"
                   if _PF_STATE["declined"] else "ℹ prefetch は辞退されていない"))
            kc_save_timings({"prefetch": _PF_STATE})

        smoke_fail = ""
        print(_gist_summary())      # 外から見えていたか（0 件は異常）
        if started:
            # 起動できても初回 forward でカーネルが落ちる領域（sm75 + 量子化 +
            # GDN Triton）があるため、1 回生成させて通ったときだけ成功とする。
            # smoke の待ち時間も総予算から差し引く（予算超過で次段が走らない事故を防ぐ）。
            _remain = int(STARTUP_BUDGET - (time.time() - _t_start))
            ok, detail = _smoke(timeout=max(30, min(90, _remain)))
            if ok:
                argv_used = cmd
                print(f"✅ vLLM 起動+生成 OK（{int(time.time()-t0)}s, 試行 {idx}）"
                      f" {detail}")
                # 起動の内訳を残す（高度化の基礎）。これが無いと毎回「なんとなく
                # 遅い」で終わり、施策の効果を比べられない。
                _load_s = round(time.time() - t0, 1)
                _span = None
                _gaps = []
                _ok_log = ""
                try:
                    _ok_log = open(log_path, errors="replace").read()
                    _marks = _phase_marks(_ok_log, _t0_wall)
                    # ログ時刻（1 秒精度）を優先し、無い節目はポーリング（60 秒）で補う。
                    # graph capture は tqdm 行なので時刻が無く、ポーリング側にしか出ない。
                    _marks = {**_poll_marks, **_marks}
                    _span = _phase_spans(_marks)
                    _gaps = _long_gaps(_ok_log, _t0_wall)
                    print(f"⏱ 起動の節目: {_marks}")
                    print(f"⏱ 区間: {_span}")
                    for _g in _gaps:
                        print(f"⏱ 無音 {_g['seconds']}s @{_g['from']}s: "
                              f"「{_g['before'][:90]}」→「{_g['after'][:90]}」")
                except Exception as _e:
                    _marks = {"error": f"{type(_e).__name__}: {_e}"}
                # **成功時こそログを残す。** 旧実装は失敗時しか保存しておらず、
                # 「compile 後 379 秒の無音区間」の中身を読めなかった（2026-11-01）。
                # 起動できた回のログは 200 KiB 程度なので、毎回残して損はない。
                if _ok_log:
                    save_output(f"vllm_{idx}.log", _ok_log[-3000000:])
                kc_save_timings({"load_seconds": _load_s,
                                 "attempt": {"index": idx, "label": label,
                                             "max_model_len": maxlen},
                                 "startup_marks": _marks,
                                 "startup_spans": _span,
                                 "startup_gaps": _gaps})
                # キャッシュが**当たったか**を判定する。「ファイルがある」ではなく
                # 「エントリが増えなかった」。373 秒の無音が戻ったときに、ミスなのか
                # 別の原因なのかをここで切り分けられるようにする。
                _tr = _triton_report(_TRITON_BEFORE, _triton_snapshot())
                print(f"🧠 Triton キャッシュ: {_tr['entries_before']} → "
                      f"{_tr['entries_after']} エントリ"
                      f"（新規 {_tr['new_entries']} / 再利用率 {_tr['reused_pct']}%）")
                if _tr["new_entries"] and not _tr["entries_before"]:
                    print("   ℹ 前回のキャッシュが無いので今回は全部作り直し"
                          "（この tar を Dataset 化すれば次回効く）")
                elif (_tr["entries_before"]
                      and _tr["new_entries"] > _tr["entries_before"] * 0.2):
                    print("   ⚠ 新規エントリが多い＝キャッシュが当たっていない"
                          "（Triton の版・sm・ソースが前回と違う）")
                kc_save_timings({"triton_cache": _tr})
                break
            print(f"⚠ 起動はしたが生成 NG → 次段へ: {detail[:300]}")
            smoke_fail = detail[:800]
            started = False

        tail = ""
        try:
            _full = open(log_path, errors="replace").read()
        except Exception:
            _full = ""
        tail = _full[-4000:]
        # 末尾だけでは「Engine core initialization failed. See root cause above.」
        # のように**原因が上にある**ケースを読めない（2026-10-29 実測）。
        # 失敗ログは丸ごと成果物に残し、原因らしき行も抜き出して表示する。
        if _full:
            save_output(f"vllm_{idx}.log", _full[-3000000:])
        _keys = ("Traceback", "Error", "error", "OutOfMemory", "OOM",
                 "RuntimeError", "assert", "CUDA", "not supported",
                 "No available shared memory")
        _sig_lines = [ln.strip() for ln in _full.splitlines()
                      if any(k in ln for k in _keys)]
        if _sig_lines:
            print("❌ 起動失敗。原因候補（ログ内の該当行, 先頭20）:")
            for _l in _sig_lines[:20]:
                print("   | " + _l[:200])
        print("❌ 起動失敗。ログ末尾:")
        print(tail)
        # 途中で GPU クォータが尽きても原因が残るよう、失敗ごとに追記保存する。
        _attempts.append(f"--- 試行 {idx} {label} max_len={maxlen} "
                         f"opt={'on' if use_opt else 'off'} ---\n"
                         + (f"[smoke NG] {smoke_fail}\n" if smoke_fail else "")
                         + tail)
        save_output("vllm_attempts.txt", "\n\n".join(_attempts) + "\n")
        _kill(proc)
        # 同一エラーで落ち続けるなら、フラグを変えても無駄。予算を残して止める。
        # 末尾 3 行を署名にする。最終行だけだと TypeError 等の一般論で一致して
        # 別原因の失敗まで打ち切ってしまう（残りクォータを捨てる損失が大きい）。
        _sig = _sig_of(tail)
        # smoke 失敗はサーバーログ末尾が正常行になるため、署名一致で打ち切ると
        # 別段の失敗まで巻き込む。smoke 失敗時は次段へ必ず進む。
        if _sig and _sig == _prev_sig and not smoke_fail:
            print(f"⛔ 同一エラーで再失敗 → ラダー打ち切り: {_sig}")
            break
        _prev_sig = _sig

    # 起動中の RAM 推移を残す（成功・失敗どちらでも）。24 GiB が消える瞬間を
    # 捕まえるための唯一の記録で、これが無いと毎回「謎のハング」に戻る。
    _RAM_STOP.set()
    _RAM_THREAD.join(timeout=10)
    try:
        _rt = open(RAM_TRACE, encoding="utf-8").read()
        if _rt.strip():
            save_output("ram_trace.txt", _rt)
            print(f"  📄 ram_trace.txt 保存済（{len(_rt)} B）")
    except OSError:
        pass

    # ---- 次回用に Triton キャッシュを退避 ----
    # 成功・失敗の**合流点**に置く。失敗した回でも、373 秒かけて作ったカーネルは
    # ディスクに残っているので、tar にして持ち出せば次回は当たる。既存の
    # `engine_cleanup()`（キープアライブ終了時）は v14 で一度も走らなかったため、
    # 起動シーケンスの中で確実に通るここで取る。
    _WARM_CACHE = _capture_warm_cache()
    if _WARM_CACHE.get("path"):
        print(f"  📦 warm-cache.tar {_WARM_CACHE['bytes']/1e6:.0f} MB"
              f"（{_WARM_CACHE['seconds']}s, 元 {_WARM_CACHE['source_bytes']/1e6:.0f} MB）"
              f" → Dataset 化すると次回の JIT を省ける")
    elif _WARM_CACHE.get("skipped"):
        print(f"  ⏭ キャッシュ退避なし: {_WARM_CACHE['skipped']}")
    kc_save_timings({"warm_cache": _WARM_CACHE})
    # Gist の往復結果も残す。ログを貼らずに `kaggle kernels output` だけで
    # 「外から見えていたか」を判定できるようにする（timings と二重化）。
    save_output("gist_selftest.json",
                json.dumps(_GIST_SELFTEST, ensure_ascii=False, indent=2))
    kc_save_timings({"gist_selftest": _GIST_SELFTEST})

    if not started:
        save_output("vllm_startup.log", tail)
        save_output("vllm_attempts.txt", "\n\n".join(_attempts) + "\n")
        if _why == "budget":
            # 予算切れは**設定の誤りではない**。ここを「起動失敗」と報告すると、
            # 次の回にフラグを疑って同じ 70 分を無駄にする。最後の節目を添えて、
            # 「どこまで進んで切られたか」だけを言う。
            fail(f"起動が予算 {STARTUP_BUDGET}s 内に終わらなかった（設定の誤りでは"
                 f"ない。遅いノードで重みロードが間に合わなかった）。"
                 f"最後に越えた節目: {_mark_last}。"
                 f"prefetch={'辞退された' if _PF_GUARD_HIT else '効いていた'}"
                 f"（vllm_startup.log / vllm_attempts.txt 参照）",
                 last_log or "/tmp/vllm_1.log")
        fail("vLLM の起動に失敗しました（vllm_startup.log / vllm_*.log 参照）",
             last_log or "/tmp/vllm_1.log")

    # ---- 起動結果を保存 ----
    info = _models() or {}
    ids = [m.get("id") for m in info.get("data", [])]
    max_len = (info.get("data") or [{}])[0].get("max_model_len")
    # 共有セル（接続情報）が読む値。設定セルの候補値ではなく実測値で上書きする。
    CONTEXT_LENGTH = max_len or CONTEXT_LENGTH
    with open("/tmp/vllm_cmd.json", "w") as f:
        json.dump({"argv": argv_used, "spec": bool(SPEC), "models": ids,
                   "max_model_len": max_len}, f, indent=2)
    print()
    print(f"  models        : {ids}")
    print(f"  max_model_len : {max_len}")
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip())
    save_output("vllm_cmd.json", json.dumps(
        {"argv": argv_used, "models": ids, "max_model_len": max_len},
        indent=2) + "\n")
    return proc


# warmup() / cleanup() は省略可（WARMUP / CLEANUP が False なら呼ばれない）。
def warmup():
    # ダミー生成で疎通確認 + 速度計測
    import json
    import subprocess
    import time
    import urllib.request

    payload = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 64,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print("⏳ ウォームアップ中...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
    except Exception as e:
        # 起動ログは試行番号で名前が変わる。読めなくても失敗自体は伝える。
        for _i in range(1, 7):
            _p = "/tmp/vllm_%d.log" % _i
            try:
                print(f"--- {_p} ---")
                print(open(_p, errors="replace").read()[-2500:])
            except OSError:
                continue
        fail(f"ウォームアップ失敗: {type(e).__name__}: {e}")

    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage") or {}
    ct = usage.get("completion_tokens") or 0
    print(f"✅ 応答あり（{dt:.1f}s）")
    print(f"   reply : {content[:120]!r}")
    print(f"   think : {reasoning[:80]!r}")
    print(f"   tokens: completion={ct} total={usage.get('total_tokens')}")
    if ct:
        print(f"   速度  : {ct/dt:.1f} tokens/s（ウォームアップ込み）")
    if not (content or reasoning or ct):
        fail("生成結果が空です（プロンプト/チャットテンプレートを確認）")
    # 速度も残す。ログにしか無いと「前回より速い/遅い」を比べられない。
    kc_save_timings({"speed_tps": round(ct / dt, 1) if ct and dt else None,
                     "warmup_seconds": round(dt, 1)})

    # GPU 常駐（CPU フォールバックでも HTTP 200 は返るので必ず見る）
    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    print(f"   VRAM  : {mem}")
    # nvidia-smi の値は "N/A" や空になり得る。int() で落とさず、数字だけ拾う。
    used = []
    for _ln in mem.splitlines():
        _a = _ln.split(",")[0].strip()
        if _a.isdigit():
            used.append(int(_a))
    if not used:
        # ここが旧実装の穴だった。`if used and ...` は **used が空だと素通り**するので、
        # nvidia-smi が何も返さない（= GPU 不在 / CPU 推論）ケースを検出できなかった。
        # 「エラーが出ずに黙って CPU 推論」は実測 0.9 t/s で、warning だけが手がかり。
        fail(f"nvidia-smi から VRAM を取得できません（mem={mem!r}）。CPU 実行の可能性。")
    if sum(used) < 4000:
        fail(f"VRAM 使用量が少なすぎます（{used} MiB）。CPU 実行の可能性。")
    kc_save_timings({"gpu_vram_mib": used})



def main():
    procs = {}
    try:
        reason = run(procs)
    finally:
        try:
            stop_all(procs)
        except Exception as e:      # 元の例外をマスクしない
            print(f"停止処理で例外: {type(e).__name__}: {e}")
        save()
    data["stage"] = "done"
    data["stopped_by"] = reason
    save()
    print(f"停止: {reason}")
    return reason


def run(procs):
    names = stage_env()
    write_aux()
    if TP_REQUIRED and len(names) < TP:
        raise RuntimeError(f"GPU が {TP} 枚見えない（TP={TP} で起動できない）")

    model_dir = None
    if NEEDS_MODEL:
        model_dir = find_model()
        if not model_dir:
            # 「未マウント」と「形式違い」を切り分けられるよう、見えている物を出す。
            # これが無いと 1 往復（10 分）を推測に使うことになる。
            print("⚠ /kaggle/input の中身:")
            for _pat in ("/kaggle/input/*", "/kaggle/input/datasets/*/*"):
                for _p in sorted(glob.glob(_pat))[:30]:
                    print(f"  {_p}")
            raise RuntimeError("モデル Dataset が見つからない")
        data["model_dir"] = model_dir
        print(f"モデル: {model_dir}")

    if INSTALL:
        stage_install()

    engine = launch(model_dir)
    if engine is not None:
        procs[ENGINE] = engine

    if WARMUP:
        warmup()

    if not SERVE:
        # 配信しないカーネル（データ準備・ベンチ等）。ここで終わり。
        data["stage"] = "batch-done"
        save()
        return "batch"

    if SMOKE:
        # GPU 時間の節約: install → launch → 会話 → 終了。
        # proxy / トンネル / Gist / keepalive は CPU 側で検証済みなので省略する。
        # direct=True: base はエンジン直。proxy が返すチャット UI は存在しない。
        verify(None, f"http://127.0.0.1:{PORT}", key="verify_direct", direct=True)
        data["timings"] = {"install_seconds": data.get("install_seconds"),
                           "load_seconds": data.get("load_seconds")}
        save()
        print("smoke 完了（トンネル / Gist / keepalive は省略）")
        return "smoke"

    proxy, auth = start_proxy(engine.pid)
    procs["proxy"] = proxy
    verify(auth, f"http://127.0.0.1:{PROXY_PORT}")

    if KEEPALIVE_SCENARIO:
        # トンネルは不要（公開経路は別途確認済み）。keepalive だけを実測する。
        data["keepalive"] = run_keepalive_scenario(auth, procs)
        data["stopped_by"] = data["keepalive"]["reason"]
        return data["keepalive"]["reason"]

    tunnel, url = start_tunnel()
    procs[TUNNEL] = tunnel
    data.update(url=url, stage="serving")
    save()
    if PUBLISH:
        publish_url(url)
    wait_public(url)
    if VERIFY_PUBLIC:
        verify_public(auth, url)

    reason = loop(procs, url)
    data["stopped_by"] = reason
    save()
    # notebook 版と同じ判定: `/healthz` の失敗が 2 回以上なら「配信できていない」。
    # ここを `ok: true` のまま終わらせると、**外から繋がらないのに成功に見える**
    # （失効した URL を配ったまま COMPLETE する。`notes/serve-parity.md`）。
    if data.get("tunnel_dead", 0) >= 2:
        raise RuntimeError(
            f"公開 URL が失効したままです（/healthz 連続失敗 "
            f"{data['tunnel_dead']} 回・再起動 "
            f"{'あり' if data.get('tunnel_restarted') else 'なし'}）。"
            f"keepalive.txt を見てください。")
    return reason


if __name__ == "__main__":
    ok = False
    try:
        main()
        data["ok"] = True
        ok = True
    except Exception as e:
        traceback.print_exc()
        data["ok"] = False
        data["error"] = f"{type(e).__name__}: {e}"
        data["traceback"] = tail(traceback.format_exc(), 20)
    save()
    print()
    print("RESULT: " + json.dumps(data, ensure_ascii=False))
    # 失敗は Kaggle 側でも失敗として扱う（result.json は残る）
    sys.exit(0 if ok else 1)

"""script カーネル（1 ファイル main.py）の共通テンプレート。

exl3 / LMDeploy / SGLang は**同じ骨格**で生成する。エンジンごとの差は 2 つだけ。

  engine_code … `stage_install()` と `launch(model_dir)` の実装（エンジン固有）
  flow        … 段の有無（install / model / tp / keepalive / publish /
                verify_public / conf）

proxy / トンネル / keepalive / 検証 / URL 配布は `runtime/` の kc_* と
このテンプレートの共通実装をそのまま使う。エンジン側に配管を書かない。
トンネルは cloudflared の quick tunnel だけ（`tunnel=` で選ぶ）。ngrok は
2026-11-01 に廃止した（`notes/url-publishing.md`）。

テンプレートを分けない理由: 以前は exl3 用と LMDeploy 用で 2 本に分かれており、
共通関数 21 本のうち完全一致が 2 本だけになっていた（`verify` が空応答を
成功扱いする / `secret` が env を読まない、という実害のある食い違いを含む）。
"""
from string import Template

from kbuild import pack_script
from runtime import module_source

# Secret 名の既定値（外部エンジンを起動するカーネル）。
# エンジン辞書の `secret_user` / `secret_pass` で上書きする。
# `tests/test_secrets.py` がこの辞書も台帳と突き合わせる。
DEFAULT_SECRETS = {
    "secret_user": "API_USER",
    "secret_pass": "API_PASS",
}

# 段の有無の既定値（外部エンジンを起動するカーネル）。
DEFAULT_FLOW = {
    "install": True,
    "model": True,
    "tp": True,
    "keepalive": False,
    "publish": True,
    "verify_public": False,
    "conf": True,
    "serve": True,          # False = 配信しない単発カーネル（データ準備・ベンチ）
    "warmup": False,        # engine["warmup"] があるときだけ True にする
    "cleanup": False,       # engine["cleanup"] があるときだけ True にする
}


def template_tokens(tpl: Template) -> set:
    """テンプレートが参照している `$TOKEN` の集合。

    `substitute()` は**足りない**トークンを KeyError で知らせるが、**余った**
    引数（テンプレートから消したのに `render()` に残った名前）は黙って捨てる。
    放っておくと「渡しているのに効かない設定」が残り、次に触る人が「なぜ変えて
    も挙動が変わらないのか」を追うことになる（2026-11-02 に関数の重複を数えた
    ときも、消し忘れの検出だけが手作業だった）。
    """
    found = set()
    for m in Template.pattern.finditer(tpl.template):
        name = m.group("named") or m.group("braced")
        if name:
            found.add(name)
        elif m.group("invalid") is not None:
            # `$` の直後が識別子でない（`$1` / `$ ` / `$-` など）。substitute() も
            # ValueError で落ちるが、どこかは教えてくれない。
            lo = max(0, m.start() - 40)
            raise ValueError(
                "テンプレートの $ の書き方が不正（{} 文字目付近: {!r}）。"
                "リテラルの $ は $$ と書く。".format(
                    m.start(), tpl.template[lo:m.start() + 40]))
    return found


def _check_tokens(tpl: Template, subs: dict, *, where: str) -> None:
    """テンプレートの `$TOKEN` と渡す引数が過不足なく一致することを確かめる。

    両方向を見る。足りない方は `substitute()` も落ちるが、余っている方は
    どこも文句を言わない。
    """
    want = template_tokens(tpl)
    got = set(subs)
    if want - got:
        raise ValueError(
            "{}: テンプレートが必要としている引数が渡されていない: {}".format(
                where, sorted(want - got)))
    if got - want:
        raise ValueError(
            "{}: テンプレートが使っていない引数が渡されている: {}"
            "（テンプレート側を消したら render() からも消す）".format(
                where, sorted(got - want)))


TEMPLATE = Template(r'''#!/usr/bin/env python3
"""$ENGINE serve（自動生成。直接編集しない）。

生成: $BUILDER
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

$RUNTIME

# ── エンジン固有の定数（旧 settings セル）──────────────────────
# 骨格が読む名前（PORT / ALIAS / CONTEXT_LENGTH など）はここで上書きできる。
# 差は「データ」だけにして、骨格には手を入れない。
$SETTINGS

MODE = "$MODE"                  # 配信モード（エンジン側の意味付け）
ENGINE = "$ENGINE"
VERSION = "$VERSION"
ALIAS = "$ALIAS"
PORT = $PORT                    # 推論サーバ（非公開）
PROXY_PORT = $PROXY_PORT        # 認証プロキシ（公開）
TP = $TP
RUN_MINUTES = $RUN_MINUTES
HARD_MAX_MINUTES = $HARD_MAX_MINUTES
# smoke: トンネル / URL 配布 / keepalive を省略し、疎通だけ見て終了する（GPU 時間の節約）
SMOKE = $SMOKE
ENGINE_PATTERN = "$ENGINE_PATTERN"
TUNNEL = "$TUNNEL"              # 実際に張るトンネル（cloudflared のみ）
KNOBS = $KNOBS
RESULT = "/kaggle/working/result.json"
# セル側が書く計測の置き場。save() が result.json の `timings` に合流させる。
TIMINGS = "/kaggle/working/timings.json"
URL_FILE = "/kaggle/working/url.txt"
WORK = "$WORK"
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
KAGGLE_OWNER = "$KAGGLE_OWNER"
# 公開 URL を置く掲示板カーネルの名前（owner は KAGGLE_OWNER と同じ）。
# カーネルから push できるのは自分のカーネルだけなので、配信カーネルと
# 同じアカウントに置く。中身はビルダーが生成する（kaggle/serve-url-beacon/）。
BEACON_NAME = "serve-url-beacon"
PROXY_B64 = "$PROXY_B64"
HTML_B64 = "$HTML_B64"
# (ファイル名, base64) を WORK へ展開する。proxy / UI / 同梱サーバの実体。
AUX_FILES = $AUX_FILES
SECRET_USER = "$SECRET_USER"
SECRET_PASS = "$SECRET_PASS"

# ── 段の有無（エンジン差はここだけ）──────────────────────────
INSTALL = $INSTALL              # stage_install() を呼ぶか
NEEDS_MODEL = $NEEDS_MODEL      # find_model() を要求するか
TP_REQUIRED = $TP_REQUIRED      # GPU が TP 枚必要か
KEEPALIVE_SCENARIO = $KEEPALIVE_SCENARIO   # keepalive 実測シナリオを回すか
PUBLISH = $PUBLISH              # 公開 URL をビーコンへ書くか
VERIFY_PUBLIC = $VERIFY_PUBLIC  # 公開 URL 経由の疎通も見るか
CONF_DRIVEN = $CONF_DRIVEN      # 停止判断を keepalive.conf に従うか
SERVE = $SERVE                  # 配信カーネルか（False = 単発のバッチ処理）
WARMUP = $WARMUP                # warmup() を呼ぶか
CLEANUP = $CLEANUP              # cleanup() を停止時に呼ぶか

data = $DATA_INIT
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
    """公開 URL を url.txt とビーコンに書く。**失敗しても配信は続ける。**

    2 経路あるのは役割が違うから:
      - `url.txt` … 同じカーネルの中から読む用（従来どおり）
      - ビーコン  … カーネルのコードを掲示板にする方式。**Secrets 不要**なので
                    専用アカウント運用でも使える（`notes/url-publishing.md`）

    Gist 版は 2026-11-02 に廃止した。`notes/url-publishing.md` のとおり、
    Kaggle の Secret は**カーネルごとに紐付ける**必要があり、未紐付けの
    `get_secret()` は `ConnectionError`（＝通信障害に見える）を返す。
    ビーコンはその作業ごと不要にする。
    """
    with open(URL_FILE, "w") as f:
        f.write(url + "\n")
    _publish_beacon(url)


def _publish_beacon(url):
    """ビーコン（掲示板カーネル）を更新する。成否は `data["beacon"]` に残す。"""
    slug = f"{KAGGLE_OWNER}/{BEACON_NAME}" if KAGGLE_OWNER else ""
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
    `publish_url` まで呼ぶ。呼ばないと url.txt / ビーコンが失効 URL を
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
$CHECKS_CODE

$ENGINE_CODE

# warmup() / cleanup() は省略可（WARMUP / CLEANUP が False なら呼ばれない）。
$WARMUP_CODE

$CLEANUP_CODE

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
        # proxy / トンネル / ビーコン / keepalive は CPU 側で検証済みなので省略する。
        # direct=True: base はエンジン直。proxy が返すチャット UI は存在しない。
        verify(None, f"http://127.0.0.1:{PORT}", key="verify_direct", direct=True)
        data["timings"] = {"install_seconds": data.get("install_seconds"),
                           "load_seconds": data.get("load_seconds")}
        save()
        print("smoke 完了（トンネル / ビーコン / keepalive は省略）")
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
''')

# 外部エンジン（LMDeploy / SGLang など）の install / launch。
# `pip 1 回 + 自前サーバ起動` が基本形。ただし **SGLang は 1 回で通らない**
# （実測 2026-11-01: `Failed building wheel for outlines_core`）ので、
# `PIP_STEPS` で**複数ステップ**を順に実行できるようにしてある。
# 1 ステップ目で落ちたときに「どのステップか」を result.json に残さないと、
# 次に読む人がまた同じ所で転ぶ。
EXTERNAL_ENGINE_CODE = Template('''PIP_STEPS = $PIP_STEPS
READY_URL = $READY_URL


def stage_install():
    data["install_steps"] = []
    for i, argv in enumerate(PIP_STEPS, 1):
        print(f"  ⬇ install step {i}/{len(PIP_STEPS)}: "
              f"{' '.join(a if len(a) < 60 else a[:57] + '...' for a in argv)}")
        r, sec = sh(argv, timeout=1800)
        rec = {"index": i, "of": len(PIP_STEPS), "seconds": round(sec, 1),
               "rc": r.returncode,
               "cmd": " ".join(a if len(a) < 60 else a[:57] + "..." for a in argv),
               # 成否にかかわらず末尾を残す。**成功したステップの出力を捨てると、
               # 後段で落ちたときに「どの版が入ったか」が分からなくなる**
               # （実測 2026-11-01: SGLang は install 成功 → 起動で torchaudio 不整合。
               #  ステップの出力が無く、どの版が入ったか追えなかった）。
               "tail": tail(r.stderr + r.stdout, 12)}
        for ln in rec["tail"].splitlines():
            print(f"     {ln}")
        if r.returncode != 0:
            data["install_steps"].append(rec)
            # 合計は**ステップの実測の和**にする（壁時計だと、テストで
            # `sh` を差し替えたときに 0 秒になり、記録の意味が消える）。
            data["install_seconds"] = round(
                sum(s["seconds"] for s in data["install_steps"]), 1)
            data["install_tail"] = rec["tail"]
            save()
            raise RuntimeError(
                f"install 失敗 ({r.returncode}) "
                f"step {i}/{len(PIP_STEPS)}: {rec['cmd']}")
        data["install_steps"].append(rec)
        save()
    data["install_seconds"] = round(
        sum(s["seconds"] for s in data["install_steps"]), 1)
    data["stage"] = "install"
    save()


def launch(model_dir):
    argv = $LAUNCH_ARGV
    data["argv"] = [a.replace(model_dir, "<model>") for a in argv]
    log_path = "/kaggle/working/engine.log"
    with open(log_path, "w") as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    data["stage"] = "load"
    save()
    t0 = time.time()
    # 起動待ちは無言になりやすい（実測 2026-11-01 vLLM: 重み読み込み後
    # 〜profile run まで約 360 秒、ログにも出ず原因不明のまま 4 往復した）。
    # 観測を残せば「遅い」と「固まった」を区別できる。
    watch, watch_stop = kc_watch(p.pid)
    try:
        wait_http(READY_URL, timeout=2400)
    except Exception:
        data["load_watch"] = watch
        try:
            with open(log_path) as f:
                data["engine_log_tail"] = tail(f.read(), 40)
        except OSError:
            pass
        raise
    finally:
        watch_stop()      # 起動できたら止める（相手は何時間も生きる）
    data["load_watch"] = watch
    data["load_seconds"] = round(time.time() - t0, 1)
    print(f"{ENGINE} 起動: {data['load_seconds']}s")
    return p
''')


def _data_init(engine: dict) -> str:
    """result.json の初期値。エンジンが何を名乗るかはここだけの差。"""
    if engine.get("data_init"):
        return engine["data_init"]
    return ('{"engine": ENGINE, "version": VERSION, "stage": "init", '
            '"tp": TP, "spec_decode": False, "smoke": SMOKE, "knobs": KNOBS}')


def render(engine: dict, *, run_minutes: int, runtime: str, proxy_b64: str,
           html_b64: str, kaggle_owner: str,
           port: int = 8000, proxy_port: int = 11435,
           hard_max_minutes: int = 180, smoke: bool = False,
           mode: str = "serve", tunnel: str = "cloudflared",
           builder: str = "kaggle/build_engine_serve.py") -> str:
    # `none` は「トンネルを張らない」実験用（playground）。公開経路を持たない
    # カーネルでだけ意味があるので、`smoke=True` と組のときだけ許す。
    # 本番（smoke=False）で `none` を渡すのは設定ミスなので、従来どおり弾く。
    # ngrok は廃止（`notes/url-publishing.md`）。ここで弾かないと、指定した
    # ビルダーが「ngrok のつもり」で cloudflared を張って静かに動いてしまう。
    if tunnel not in ("cloudflared",) and not (smoke and tunnel == "none"):
        raise ValueError(
            f"tunnel={tunnel!r} は未対応（cloudflared / smoke 時のみ none）")
    flow = {**DEFAULT_FLOW, **(engine.get("flow") or {})}
    # 段の名前を間違えると `{**DEFAULT_FLOW, ...}` が黙って受け入れ、その段は
    # 既定のまま動く（＝「False にしたのに止まらない」）。名前と型の両方を見る。
    # 型は bool に限る: `INSTALL = $INSTALL` は素の Python 式として埋め込むので、
    # 文字列 "false" を渡すとカーネル側で `INSTALL = false` になり NameError に
    # なる（ビルドは通り、GPU を掴んでから落ちる）。
    unknown = sorted(set(engine.get("flow") or {}) - set(DEFAULT_FLOW))
    if unknown:
        raise ValueError(
            "engine[{!r}]['flow'] に知らない段がある: {}（使えるのは {}）".format(
                engine["name"], unknown, sorted(DEFAULT_FLOW)))
    not_bool = {k: v for k, v in flow.items() if not isinstance(v, bool)}
    if not_bool:
        raise TypeError(
            "engine[{!r}]['flow'] の値は bool にする: {}".format(
                engine["name"], not_bool))

    # 実体があるスロットは自動で有効化する（flow で明示しなくても呼ばれる）。
    if engine.get("warmup"):
        flow["warmup"] = True
    if engine.get("cleanup"):
        flow["cleanup"] = True
    aux = engine.get("aux") or (("auth_proxy.py", proxy_b64),
                                ("llama-chat.html", html_b64))
    code = engine.get("engine_code")
    if not code:
        ext = {
            # `pip_argv`（1 回）しか持たないエンジンは 1 要素のリストに包む。
            # LMDeploy はこれまで通り 1 回で通るので、そのまま動く。
            "PIP_STEPS": engine.get("pip_steps") or f"[{engine['pip_argv']}]",
            "LAUNCH_ARGV": engine["launch_argv"],
            "READY_URL": engine["ready_url"],
        }
        _check_tokens(EXTERNAL_ENGINE_CODE, ext,
                      where=f"{engine['name']} の engine_code")
        code = EXTERNAL_ENGINE_CODE.substitute(**ext)
    # ビーコンは骨格側で必ず足す。各ビルダーに `"beacon"` を書かせると
    # 1 本書き忘れただけで「URL が更新されないのに配信は続く」状態になる
    # （2026-11-01 に URL 配布で同じ穴を踏んだ）。公開しない構成でも、束ねて
    # おけば `publish_url` が NameError にならない。
    runtime = runtime + module_source("beacon")
    subs = dict(
        BUILDER=builder,
        ENGINE=engine["name"],
        ENGINE_PATTERN=engine.get("pattern", f"-f {engine['name']}"),
        VERSION=engine.get("version", ""),
        ALIAS=engine["alias"],
        MODE=mode,
        PORT=port,
        PROXY_PORT=proxy_port,
        TP=engine.get("tp", 2),
        HARD_MAX_MINUTES=hard_max_minutes,
        SMOKE="True" if smoke else "False",
        TUNNEL=tunnel,
        KNOBS=engine.get("knobs", "{}"),
        RUN_MINUTES=run_minutes,
        RUNTIME=runtime,
        SETTINGS=engine.get("settings", ""),
        CHECKS_CODE=engine.get("checks_code", ""),
        WARMUP_CODE=engine.get("warmup", ""),
        CLEANUP_CODE=engine.get("cleanup", ""),
        WORK=engine.get("work") or f"/tmp/{engine['name']}-serve",
        AUX_FILES=aux,
        PROXY_B64=proxy_b64,
        HTML_B64=html_b64,
        SECRET_USER=engine.get("secret_user", DEFAULT_SECRETS["secret_user"]),
        SECRET_PASS=engine.get("secret_pass", DEFAULT_SECRETS["secret_pass"]),
        DATA_INIT=_data_init(engine),
        ENGINE_CODE=code,
        INSTALL=flow["install"],
        NEEDS_MODEL=flow["model"],
        TP_REQUIRED=flow["tp"],
        KEEPALIVE_SCENARIO=flow["keepalive"],
        PUBLISH=flow["publish"],
        VERIFY_PUBLIC=flow["verify_public"],
        CONF_DRIVEN=flow["conf"],
        SERVE=flow["serve"],
        WARMUP=flow["warmup"],
        CLEANUP=flow["cleanup"],
        KAGGLE_OWNER=kaggle_owner,
    )
    # 引数の過不足はここで落とす。`substitute()` は「足りない」しか見ないので、
    # テンプレートからトークンを消したときに `render()` 側の名前が残る。
    _check_tokens(TEMPLATE, subs, where=f"{engine['name']} の TEMPLATE")
    return TEMPLATE.substitute(**subs)


def write(out_dir, engine: dict, meta: dict, **kwargs) -> None:
    body = render(engine, **kwargs)
    pack_script.write(out_dir, body, meta)

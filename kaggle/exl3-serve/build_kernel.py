#!/usr/bin/env python3
"""EXL3 配信カーネル（script 1 ファイル）を生成する。

Kaggle の script カーネルは code_file を 1 つしか push できないため、
chat_server.py / auth_proxy.py / llama-chat.html を base64 で埋め込んだ
自己完結の main.py を作る。

    python3 kaggle/exl3-serve/build_kernel.py

生成物:
  kernel/main.py   … GPU 版（enable_gpu true）
  dryrun/main.py   … MockEngine 版（CPU・配管テスト用）
"""
import pathlib
import sys
from string import Template

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import embed, metadata, pack_script  # noqa: E402
from runtime import source_bundle  # noqa: E402

CFG = kcconfig.load(ROOT)
GIST_ID_DEFAULT = CFG.get("GIST_ID", "")
GIST_OWNER = CFG.get("GIST_OWNER", "")

ALIAS = "qwen38-exl3"
CACHE_TOKENS = 8192
RUN_MINUTES = 60

CHAT_B64 = embed.b64_file(HERE / "chat_server.py")
PROXY_B64 = embed.b64_file(ROOT / "kaggle" / "step2" / "auth_proxy.py")
HTML_B64 = embed.b64_file(ROOT / "llama-chat.html")
# proxy 起動・tunnel URL 待ち・keepalive conf は runtime/ の真実源を使う。
RUNTIME = source_bundle("net", "proc", "keepalive", "proxy")

TEMPLATE = Template(r'''#!/usr/bin/env python3
# 自動生成: kaggle/exl3-serve/build_kernel.py（直接編集しない）
"""ExLlamaV3 (EXL3) 配信カーネル。

モデルロード → OpenAI 互換サーバ → Basic 認証プロキシ → cloudflared 公開。
/stop（プロキシの管理エンドポイント）か時間予算で自動停止する。
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

MODE = "$MODE"                 # "exl3" | "mock"
ALIAS = "$ALIAS"
CHAT_PORT = 9931
PROXY_PORT = 11435
CACHE_TOKENS = $CACHE_TOKENS
RUN_MINUTES = $RUN_MINUTES
HARD_MAX_MINUTES = 180
RESULT = "/kaggle/working/result.json"
URL_FILE = "/kaggle/working/url.txt"
WORK = "/tmp/exl3-serve"
CONF = "/tmp/keepalive.conf"
GIST_ID_DEFAULT = "$GIST_ID"
GIST_OWNER = "$GIST_OWNER"

CHAT_B64 = "$CHAT_B64"
PROXY_B64 = "$PROXY_B64"
HTML_B64 = "$HTML_B64"

data = {"stage": "init", "mode": MODE}
T0 = time.time()


def save():
    data["elapsed_seconds"] = round(time.time() - T0, 1)
    with open(RESULT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=600, env=None):
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


def http_json(url, payload=None, auth=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def wait_http(url, timeout, auth=None):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            code, _ = http_json(url, auth=auth, timeout=10)
            if code == 200:
                return round(time.time() - t0, 1)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(5)
    raise RuntimeError(f"待機タイムアウト {url}（{last}）")


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


# ── 1. 環境 ─────────────────────────────────────────────────
def stage_env():
    print("=" * 60)
    print(f"EXL3 serve ({MODE})")
    print("=" * 60)
    print(f"python : {sys.version.split()[0]}")
    names, mems = kc_detect_gpu()
    if names:
        text = "\n".join(f"{n}, {m} MiB" for n, m in zip(names, mems))
        print(text)
        data["nvidia_smi"] = text
    else:
        # CPU カーネル（乾燥走行）には nvidia-smi が無い
        print("nvidia-smi: なし（CPU ホスト）")
        data["nvidia_smi"] = None
    try:
        import torch
        data.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(),
                    device_count=torch.cuda.device_count())
        print(f"torch  : {torch.__version__} / cuda {torch.version.cuda}")
    except Exception as e:
        data["torch_error"] = f"{type(e).__name__}: {e}"
    data["stage"] = "env"
    save()


# ── 2. exllamav3 導入（GPU 版のみ）───────────────────────────
EXL3_VERSION = "1.6.5"


def wheel_candidates():
    import torch
    tv = torch.__version__.split("+")[0]
    cu = (torch.version.cuda or "").replace(".", "")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cus = ([f"cu{cu}"] if cu else []) + ["cu132", "cu130", "cu128"]
    seen = []
    for c in cus:
        if c not in seen:
            seen.append(c)
    return [f"exllamav3-{EXL3_VERSION}+{c}.torch{tv}-{py}-{py}-linux_x86_64.whl"
            for c in seen]


def pick_wheel_url():
    import urllib.parse
    for name in wheel_candidates():
        url = ("https://github.com/turboderp-org/exllamav3/releases/download/"
               f"v{EXL3_VERSION}/{urllib.parse.quote(name, safe='')}")
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30):
                return url, name
        except Exception:
            continue
    return None, None


def pip_install(target):
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
           "MAX_JOBS": str(min(4, os.cpu_count() or 4))}
    r, dt = sh([sys.executable, "-m", "pip", "install", "-U",
                "--no-cache-dir", target], timeout=1800, env=env)
    print(f"pip {target}: rc={r.returncode} {dt}s")
    print(tail(r.stdout + "\n" + r.stderr, 20))
    return r, dt


def stage_install():
    print()
    print("--- exllamav3 ---")
    url, name = pick_wheel_url()
    if url:
        data.update(install_method="wheel", wheel_name=name)
        r, dt = pip_install(url)
        target = "exllamav3"
    else:
        data["install_method"] = "pypi_jit"
        r, dt = pip_install("exllamav3")
        target = "exllamav3"
    data.update(pip_rc=r.returncode, pip_seconds=dt)
    if r.returncode != 0 and url:
        data["install_method"] = "pypi_jit_fallback"
        r, dt = pip_install(target)
        data.update(pip_rc=r.returncode, pip_seconds=dt)
    if r.returncode != 0:
        raise RuntimeError("pip install 失敗")
    data["stage"] = "install"
    save()


def find_model():
    for pat in ("/kaggle/input/datasets/*/*/config.json",
                "/kaggle/input/*/config.json"):
        for cfg in sorted(glob.glob(pat)):
            d = os.path.dirname(cfg)
            if os.path.isfile(os.path.join(d, "quantization_config.json")):
                return d
    return None


# ── 3. 補助ファイル展開 ─────────────────────────────────────
def write_aux():
    os.makedirs(WORK, exist_ok=True)
    for name, b64 in (("chat_server.py", CHAT_B64),
                      ("auth_proxy.py", PROXY_B64),
                      ("llama-chat.html", HTML_B64)):
        with open(os.path.join(WORK, name), "wb") as f:
            f.write(base64.b64decode(b64))


# ── 4. サーバ起動 ───────────────────────────────────────────
def start_chat():
    cmd = [sys.executable, f"{WORK}/chat_server.py",
           "--host", "127.0.0.1", "--port", str(CHAT_PORT),
           "--alias", ALIAS, "--cache-tokens", str(CACHE_TOKENS),
           "--no-progress"]
    if MODE != "exl3":
        cmd.append("--mock")
    else:
        cmd += ["--model-dir", data["model_dir"]]
    log_path = "/kaggle/working/chat_server.log"
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    data["stage"] = "load"
    save()
    t0 = time.time()
    timeout = 60 if MODE == "mock" else 1800
    try:
        wait_http(f"http://127.0.0.1:{CHAT_PORT}/health", timeout)
    except Exception:
        try:
            with open(log_path) as f:
                data["chat_log_tail"] = tail(f.read(), 40)
        except OSError:
            pass
        raise
    data["load_seconds"] = round(time.time() - t0, 1)
    print(f"chat_server 起動: {data['load_seconds']}s")
    return p


def start_proxy():
    user = secret("EXL3_API_USER") or "exl3"
    pass_secret = secret("EXL3_API_PASS")
    # パスワードは result.json / ログに出さない（kernels output で誰でも
    # 取得できてしまうため）。未設定時は乱数で起動するが、値はどこにも残さない。
    pass_ = pass_secret or os.urandom(8).hex()
    auth = "Basic " + base64.b64encode(
        f"{user}:{pass_}".encode()).decode()
    env = {**os.environ, "API_USER": user, "API_PASS": pass_}
    p = kc_start_proxy(
        PROXY_PORT, f"http://127.0.0.1:{CHAT_PORT}",
        f"{WORK}/llama-chat.html", f"{WORK}/auth_proxy.py",
        env=env, log_path="/kaggle/working/auth_proxy.log",
        wait_seconds=60, label="auth_proxy", python=sys.executable)
    data["auth"] = "on"
    data["auth_user"] = user
    if not pass_secret:
        data["auth_warning"] = "EXL3_API_PASS 未設定（乱数。値は非公開）"
        print("EXL3_API_PASS 未設定: 乱数パスワードで起動（値は残さない）")
    print(f"auth_proxy 起動: 127.0.0.1:{PROXY_PORT}")
    return p, auth


def verify(auth, base, key="verify"):
    code, body = http_json(f"{base}/v1/models", auth=auth)
    data["models"] = body
    payload = {"model": ALIAS,
               "messages": [{"role": "user", "content": "1+1は？"}],
               "max_tokens": 16, "temperature": 0}
    t0 = time.time()
    code, body = http_json(f"{base}/v1/chat/completions", payload,
                           auth=auth, timeout=900)
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    data[key] = {"status": code, "text": text[:200],
                 "usage": body.get("usage") or {},
                 "wall_seconds": round(time.time() - t0, 1)}
    print(f"疎通: {code} / {text[:80]!r}")


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


# ── 5. トンネル ─────────────────────────────────────────────
def find_cloudflared():
    for pat in ("/kaggle/input/*/cloudflared",
                "/kaggle/input/*/*/cloudflared",
                "/kaggle/input/datasets/*/*/cloudflared"):
        for p in sorted(glob.glob(pat)):
            if os.path.isfile(p):
                return p
    return None


def start_tunnel():
    cf = find_cloudflared()
    if cf:
        # /kaggle/input は読み取り専用で実行ビットも付けられないため、
        # 書き込み可能な WORK へコピーしてから chmod する。
        src = cf
        cf = f"{WORK}/cloudflared"
        shutil.copy2(src, cf)
        os.chmod(cf, 0o755)
        data["cloudflared_source"] = f"{src} (copied)"
    else:
        cf = f"{WORK}/cloudflared"
        url = ("https://github.com/cloudflare/cloudflared/releases/latest/"
               "download/cloudflared-linux-amd64")
        urllib.request.urlretrieve(url, cf)
        os.chmod(cf, 0o755)
        data["cloudflared_source"] = "download"

    log_path = "/kaggle/working/cloudflared.log"
    log = open(log_path, "w")
    p = subprocess.Popen([cf, "tunnel", "--url",
                          f"http://127.0.0.1:{PROXY_PORT}", "--no-autoupdate"],
                         stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    url = kc_wait_url("cloudflared", log_path, seconds=120)
    if not url:
        if p.poll() is not None:
            raise RuntimeError(f"cloudflared 終了 ({p.returncode})")
        raise RuntimeError("cloudflared の URL を取得できなかった")
    data["tunnel_seconds"] = round(time.time() - t0, 1)
    print(f"公開 URL: {url}")
    return p, url


def publish_url(url):
    """公開 URL を url.txt と Gist に書く（ask.py が Gist を読む）。

    GIST_ID / GITHUB_TOKEN が無ければスキップ。失敗しても配信は続ける。
    """
    with open(URL_FILE, "w") as f:
        f.write(url + "\n")
    gist_id = (os.environ.get("GIST_ID") or secret("GIST_ID")
               or GIST_ID_DEFAULT or "").strip()
    token = (os.environ.get("GITHUB_TOKEN")
             or secret("GITHUB_TOKEN") or "").strip()
    if not gist_id or not token:
        data["publish"] = "skipped (GIST_ID / GITHUB_TOKEN なし)"
        print("Gist 公開: スキップ（GIST_ID / GITHUB_TOKEN なし）")
        return
    body = json.dumps({"files": {"url.txt": {"content": url + "\n"}}}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}", data=body, method="PATCH",
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "exl3-serve-kernel"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data["publish"] = f"gist ok ({r.status})"
        print("Gist 公開: " + "https://gist.githubusercontent.com/"
              f"{GIST_OWNER}/{gist_id}/raw/url.txt")
    except Exception as e:
        data["publish"] = f"gist failed: {type(e).__name__}: {e}"
        print(f"Gist 公開 失敗（続行）: {e}")


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


def loop(procs):
    started = time.time()
    while True:
        if MODE == "mock":
            # 乾燥走行は配管テストなので keepalive を見ない。
            # （ローカル疎通の生成リクエストが MINUTES=60 を書き込み、
            #   5 分の予算が 60 分に延びるため）
            if time.time() - started >= RUN_MINUTES * 60:
                return "budget"
        else:
            run, minutes, mtime = read_conf()
            if run == 0:
                return "stop"
            deadline = kc_keepalive_deadline(started, minutes, mtime,
                                             HARD_MAX_MINUTES)
            if time.time() >= deadline:
                return "budget"
        for name, p in procs.items():
            if p.poll() is not None:
                return f"{name}_exited({p.returncode})"
        time.sleep(20)


def run_keepalive_scenario(auth, procs):
    """keepalive の実挙動を実測する（MODE=keepalive）。

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
            run, minutes, mtime = read_conf()
            ka["after_activity"] = {
                "run": run, "minutes": minutes,
                "mtime_age": round(time.time() - (mtime or t0), 1),
            }
            print(f"keepalive 更新: RUN={run} MINUTES={minutes}")
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


def main():
    stage_env()
    write_aux()
    if MODE == "exl3":
        stage_install()
        model_dir = find_model()
        if not model_dir:
            raise RuntimeError("モデル Dataset が見つからない")
        data["model_dir"] = model_dir
        print(f"モデル: {model_dir}")

    chat = start_chat()
    proxy, auth = start_proxy()
    verify(auth, f"http://127.0.0.1:{PROXY_PORT}")

    if MODE == "keepalive":
        # トンネルは不要（v4 で公開経路は確認済み）。keepalive だけを実測する。
        data["keepalive"] = run_keepalive_scenario(
            auth, {"chat": chat, "proxy": proxy})
        data["stopped_by"] = data["keepalive"]["reason"]
        for p in (proxy, chat):
            try:
                p.terminate()
            except Exception:
                pass
        data["stage"] = "done"
        return

    tunnel, url = start_tunnel()
    data.update(url=url, stage="serving")
    save()
    if MODE == "exl3":
        publish_url(url)   # mock の URL で Gist を上書きしない
    wait_public(url)
    verify_public(auth, url)

    reason = loop({"chat": chat, "proxy": proxy, "cloudflared": tunnel})
    data["stopped_by"] = reason
    print(f"停止: {reason}")
    for p in (tunnel, proxy, chat):
        try:
            p.terminate()
        except Exception:
            pass
    data["stage"] = "done"
    save()


if __name__ == "__main__":
    try:
        main()
        data["ok"] = True
    except Exception as e:
        traceback.print_exc()
        data["ok"] = False
        data["error"] = f"{type(e).__name__}: {e}"
    save()
    print()
    print("RESULT: " + json.dumps(data, ensure_ascii=False))
    sys.exit(0)
''')


OWNER = "yamahei21python"
EXL3_DATASET = f"{OWNER}/qwen38-27b-uncensored-exl3-4bpw"
CLOUDFLARED_DATASET = f"{OWNER}/my-cloudflared-bin"

METADATA_REAL = metadata.base(
    id=f"{OWNER}/exl3-serve",
    title="EXL3 serve",
    code_file="main.py",
    kernel_type="script",
    gpu=True,
    datasets=[EXL3_DATASET, CLOUDFLARED_DATASET],
)

METADATA_DRYRUN = metadata.base(
    id=f"{OWNER}/exl3-serve-dryrun",
    title="EXL3 serve dryrun",
    code_file="main.py",
    kernel_type="script",
    gpu=False,
    datasets=[CLOUDFLARED_DATASET],
)

METADATA_KEEPALIVE = metadata.base(
    id=f"{OWNER}/exl3-serve-keepalive",
    title="EXL3 serve keepalive",
    code_file="main.py",
    kernel_type="script",
    gpu=False,
    datasets=[],
)


def write(mode, out_dir, meta, run_minutes):
    body = TEMPLATE.substitute(
        MODE=mode, ALIAS=ALIAS, CACHE_TOKENS=CACHE_TOKENS,
        RUN_MINUTES=run_minutes, CHAT_B64=CHAT_B64,
        PROXY_B64=PROXY_B64, HTML_B64=HTML_B64,
        GIST_ID=GIST_ID_DEFAULT, GIST_OWNER=GIST_OWNER,
        RUNTIME=RUNTIME,
    )
    pack_script.write(out_dir, body, meta)


if __name__ == "__main__":
    # 本番 60 分、乾燥走行は疎通確認だけで 5 分、
    # keepalive は 2 分予算 + 延長 + /stop の決め打ちシナリオ
    write("exl3", HERE / "kernel", METADATA_REAL, RUN_MINUTES)
    write("mock", HERE / "dryrun", METADATA_DRYRUN, 5)
    write("keepalive", HERE / "keepalive", METADATA_KEEPALIVE, 2)

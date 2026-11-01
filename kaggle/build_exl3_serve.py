#!/usr/bin/env python3
"""EXL3 配信カーネル（script 1 ファイル）を生成する。

Kaggle の script カーネルは code_file を 1 つしか push できないため、
chat_server.py / auth_proxy.py / llama-chat.html を base64 で埋め込んだ
自己完結の main.py を作る。骨格は LMDeploy / SGLang と共通
（`kbuild/serve_script.py`）で、ここが持つのは**エンジン固有の 2 つだけ**:
`stage_install()`（wheel 選択）と `launch()`（chat_server 起動）。

    python3 kaggle/build_exl3_serve.py

生成物:
  kaggle/exl3-serve-kernel/serve/main.py     … GPU 版（enable_gpu true）
  kaggle/exl3-serve-kernel/dryrun/main.py    … MockEngine 版（CPU・配管テスト用）
  kaggle/exl3-serve-kernel/keepalive/main.py … keepalive 実測（CPU）
"""
import pathlib
import sys
from string import Template

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent                                      # リポジトリ直下
OUT = HERE / "exl3-serve-kernel"
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import embed, metadata, serve_script  # noqa: E402
from runtime import proxy_source, source_bundle  # noqa: E402
from runtime.constants import (  # noqa: E402
    CHAT_PORT,
    EXL3_HARD_MAX_MINUTES,
    EXL3_RUN_MINUTES,
    PROXY_PORT,
)

CFG = kcconfig.load(ROOT)
GIST_ID_DEFAULT = CFG.get("GIST_ID", "")
GIST_OWNER = CFG.get("GIST_OWNER", "")

ALIAS = "qwen38-exl3"
CACHE_TOKENS = 8192
RUN_MINUTES = EXL3_RUN_MINUTES
EXL3_VERSION = "1.6.5"

CHAT_B64 = embed.b64_file(HERE / "exl3-serve" / "chat_server.py")
PROXY_B64 = embed.b64_text(
    proxy_source(ROOT / "kaggle" / "step2" / "auth_proxy.py"))
HTML_B64 = embed.b64_file(ROOT / "llama-chat.html")
# proxy 起動・tunnel URL 待ち・keepalive conf は runtime/ の真実源を使う。
RUNTIME = source_bundle("net", "proc", "keepalive", "proxy")

# エンジン固有ブロック。共通テンプレートの `$ENGINE_CODE` に入る。
# ここだけが exl3 の「差」— 配管（proxy / tunnel / keepalive / 検証）は書かない。
ENGINE_CODE = Template('''EXL3_VERSION = "$EXL3_VERSION"
CACHE_TOKENS = $CACHE_TOKENS


def wheel_candidates():
    """exllamav3 の wheel 名候補（torch / CUDA / python に合わせる）。"""
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
    print(tail(r.stdout + "\\n" + r.stderr, 20))
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


def launch(model_dir):
    """同梱の chat_server.py を起動する（外部エンジンではない）。"""
    cmd = [sys.executable, f"{WORK}/chat_server.py",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--alias", ALIAS, "--cache-tokens", str(CACHE_TOKENS),
           "--no-progress"]
    if MODE != "exl3":
        cmd.append("--mock")
    else:
        cmd += ["--model-dir", model_dir]
    log_path = "/kaggle/working/chat_server.log"
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    data["stage"] = "load"
    save()
    t0 = time.time()
    timeout = 60 if MODE == "mock" else 1800
    try:
        wait_http(f"http://127.0.0.1:{PORT}/health", timeout)
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
''').substitute(EXL3_VERSION=EXL3_VERSION, CACHE_TOKENS=CACHE_TOKENS)


def _engine(mode: str) -> dict:
    """モードごとの「段の有無」。骨格は serve_script 側で共通。"""
    real = mode == "exl3"
    return {
        "name": "exl3",
        "version": EXL3_VERSION,
        "alias": ALIAS,
        "pattern": "-f chat_server.py",
        "work": "/tmp/exl3-serve",
        "secret_user": "EXL3_API_USER",
        "secret_pass": "EXL3_API_PASS",
        "data_init": '{"stage": "init", "mode": MODE, "smoke": SMOKE}',
        "aux": (("chat_server.py", CHAT_B64),
                ("auth_proxy.py", PROXY_B64),
                ("llama-chat.html", HTML_B64)),
        "engine_code": ENGINE_CODE,
        "flow": {
            "install": real,          # mock / keepalive は導入済み前提
            "model": real,            # mock はモデル不要
            "tp": False,              # EXL3 は TP 非対応（1 枚で動かす）
            "keepalive": mode == "keepalive",
            "publish": real,          # mock の URL で Gist を上書きしない
            "verify_public": True,
            "conf": mode != "mock",   # mock は keepalive を見ない
        },
    }


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


def write(mode, out_dir, meta, run_minutes, smoke=False):
    serve_script.write(
        out_dir, _engine(mode), meta,
        run_minutes=run_minutes,
        runtime=RUNTIME,
        proxy_b64=PROXY_B64,
        html_b64=HTML_B64,
        gist_id=GIST_ID_DEFAULT,
        gist_owner=GIST_OWNER,
        port=CHAT_PORT,
        proxy_port=PROXY_PORT,
        hard_max_minutes=EXL3_HARD_MAX_MINUTES,
        smoke=smoke,
        mode=mode,
        builder="kaggle/build_exl3_serve.py",
    )


if __name__ == "__main__":
    # 本番 60 分、乾燥走行は疎通確認だけで 5 分、
    # keepalive は 2 分予算 + 延長 + /stop の決め打ちシナリオ
    # --smoke は exl3-serve だけを「疎通 → 終了」変種で上書きする（push 直前に再生成）。
    smoke = "--smoke" in sys.argv
    if smoke:
        write("exl3", OUT / "serve", METADATA_REAL, 5, smoke=True)
        sys.exit(0)
    write("exl3", OUT / "serve", METADATA_REAL, RUN_MINUTES)
    write("mock", OUT / "dryrun", METADATA_DRYRUN, 5)
    write("keepalive", OUT / "keepalive", METADATA_KEEPALIVE, 2)

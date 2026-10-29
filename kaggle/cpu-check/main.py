#!/usr/bin/env python3
"""Kaggle CPU（GPUなし）で llama-server 直起動が成立するかの検証。

流用元:
  kaggle/build_step2_direct.py
    - _find_input()            Dataset 2系統のマウント先探索
    - バイナリ復元              llama-server + lib/ollama/*.so のサイズ検証付きコピー
    - manifest からの blob 解決
    - 起動待ち                 /health ポーリング
  ask.py
    - post() / reply_text()    /v1/chat/completions 呼び出し

差分（CPU検証のため）:
  - -ngl 0 / -t 4 / -c 2048（VRAM の代わりに RAM）
  - KV 量子化なし（f16 既定。q8_0 KV は -fa 前提のため CPU では使わない）
  - CUDA ライブラリを LD_LIBRARY_PATH に入れない（cuda_v12 もコピーしない）
  - MTP なし（まず素の直起動が目的）
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

MODEL_NAME = "orcarouter/Qwen3.8-27B-Uncensored:latest"
DATASET_SLUG = "my-ollama-orcarouter-qwen38-27b"
PATCH_SLUG = "my-ollama-orcarouter-qwen38-27b-patch"

PORT = 11434
HOST = "127.0.0.1"
CTX = 2048
THREADS = min(4, os.cpu_count() or 1)

LIB_DIR = "/usr/local/lib/ollama"
LLAMA_SERVER = os.path.join(LIB_DIR, "llama-server")

WORK = "/kaggle/working"
LOG = os.path.join(WORK, "llama-server.log")
RESULT = os.path.join(WORK, "result.json")

RESULT_DATA = {"stage": "init", "cpu_start_ok": False}


def _save():
    with open(RESULT, "w") as f:
        json.dump(RESULT_DATA, f, ensure_ascii=False, indent=2)


def _sh(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _find_input(slug):
    """Dataset のマウント先を探す（build_step2_direct.py から流用）。

    Kaggle は /kaggle/input/<slug>（旧）と
    /kaggle/input/datasets/<owner>/<slug>（新）が混在する。
    """
    cands = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/yamahei21python/{slug}",
    ]
    for base in ("/kaggle/input", "/kaggle/input/datasets"):
        try:
            for owner in os.listdir(base):
                cands.append(os.path.join(base, owner, slug))
        except OSError:
            pass
    for c in cands:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return f"/kaggle/input/{slug}"


def _meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.strip().split()[0]) // 1024  # MiB
    return d


def stage_env():
    print("=" * 60)
    print("Kaggle CPU で llama-server 直起動の検証")
    print("=" * 60)
    mem = _meminfo()
    print(f"nproc        : {os.cpu_count()}")
    print(f"MemTotal     : {mem.get('MemTotal', 0)} MiB "
          f"({mem.get('MemTotal', 0) / 1024:.1f} GiB)")
    print(f"MemAvailable : {mem.get('MemAvailable', 0)} MiB")
    print(f"nvidia-smi   : {'present' if shutil.which('nvidia-smi') else 'ABSENT（CPUのみ）'}")

    cpu = ""
    flags = ""
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name") and not cpu:
                cpu = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line.split(":", 1)[1].strip()
            if cpu and flags:
                break
    print(f"CPU          : {cpu}")
    print("ISA          : " + ", ".join(
        k for k in ("avx2", "avx512f", "fma") if k in flags.split()))

    df = _sh(["df", "-h", "/kaggle/working"]).stdout.strip().splitlines()
    if len(df) > 1:
        print("disk         : " + df[-1])

    RESULT_DATA.update(
        cpu=cpu, nproc=os.cpu_count(),
        mem_total_mib=mem.get("MemTotal", 0),
        gpu_present=shutil.which("nvidia-smi") is not None,
    )
    RESULT_DATA["stage"] = "env"
    _save()


def stage_restore(patch_path):
    """llama-server + 依存 .so を patch Dataset から復元する（流用）。

    CUDA は使わないので cuda_v12/ はコピーしない（約 440 MB 節約）。
    """
    MIN_SIZE_LLAMA = 4096
    MIN_SIZE_SO = 64 * 1024
    LIB_CACHE = os.path.join(patch_path, "lib", "ollama")

    restored, skipped = [], []
    os.makedirs(LIB_DIR, exist_ok=True)

    def _copy(src, dst, min_size):
        if not (os.path.isfile(src) and os.path.getsize(src) >= min_size):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True

    for cand in (os.path.join(LIB_CACHE, "llama-server"),
                 os.path.join(patch_path, "bin", "llama-server")):
        if _copy(cand, LLAMA_SERVER, MIN_SIZE_LLAMA):
            restored.append("llama-server")
            break
    else:
        raise RuntimeError("llama-server が patch Dataset にありません")

    n = 0
    for root, dirs, files in os.walk(LIB_CACHE):
        rel = os.path.relpath(root, LIB_CACHE)
        if rel.split(os.sep)[0] == "cuda_v12":
            skipped.append("cuda_v12")
            dirs[:] = []
            continue
        dst_dir = LIB_DIR if rel == "." else os.path.join(LIB_DIR, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            dst = os.path.join(dst_dir, fn)
            if os.path.abspath(dst) == os.path.abspath(LLAMA_SERVER):
                continue
            shutil.copy2(os.path.join(root, fn), dst)
            n += 1
    restored.append(f"lib/ollama ({n} files)")

    print(f"✅ 復元: {', '.join(restored)}")
    if skipped:
        print(f"⏭ スキップ: {', '.join(sorted(set(skipped)))}（CPU検証では不要）")

    exe_mode = os.stat(LLAMA_SERVER).st_mode
    if (exe_mode & 0o111) != 0o111:
        os.chmod(LLAMA_SERVER, 0o755)
        print(f"🔧 実行ビット付与: {LLAMA_SERVER}")

    for path, min_sz, label in [
        (LLAMA_SERVER, MIN_SIZE_LLAMA, "llama-server"),
        (os.path.join(LIB_DIR, "libllama-server-impl.so"), MIN_SIZE_SO,
         "libllama-server-impl.so"),
        (os.path.join(LIB_DIR, "libllama.so.0"), MIN_SIZE_SO, "libllama.so.0"),
        (os.path.join(LIB_DIR, "libggml-base.so.0"), MIN_SIZE_SO,
         "libggml-base.so.0"),
        (os.path.join(LIB_DIR, "libggml.so.0"), MIN_SIZE_SO, "libggml.so.0"),
        (os.path.join(LIB_DIR, "libllama-common.so.0"), MIN_SIZE_SO,
         "libllama-common.so.0"),
    ]:
        ok = os.path.isfile(path) and os.path.getsize(path) >= min_sz
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"{'✅' if ok else '❌'} {label}  ({size:,} B)")
        if not ok:
            raise RuntimeError(f"{label} が不足しています（patch Dataset を確認）")

    env = {**os.environ, "LD_LIBRARY_PATH": LIB_DIR}
    r = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True,
                       text=True, env=env, timeout=60)
    ver = ((r.stdout or "") + (r.stderr or "")).strip()
    print()
    print(ver[:600])
    if r.returncode != 0:
        raise RuntimeError(f"llama-server --version 失敗 (rc={r.returncode})")
    RESULT_DATA["version"] = ver.splitlines()[0] if ver else ""
    RESULT_DATA["stage"] = "restore"
    _save()


def stage_resolve(dataset_path):
    """manifest から model blob を解決する（流用）。"""
    manifest = os.path.join(
        dataset_path, "manifests", "registry.ollama.ai",
        "orcarouter", "Qwen3.8-27B-Uncensored", "latest",
    )
    blobs = os.path.join(dataset_path, "blobs")
    print(f"manifest: {manifest}")
    if not os.path.isfile(manifest):
        raise RuntimeError(f"manifest が見つかりません: {manifest}")

    with open(manifest) as f:
        mf = json.load(f)

    model_blob = None
    for layer in mf.get("layers", []):
        if layer.get("mediaType") != "application/vnd.ollama.image.model":
            continue
        digest = layer.get("digest", "")
        if digest.startswith("sha256:"):
            model_blob = os.path.join(blobs, "sha256-" + digest.split(":", 1)[1])

    if not model_blob or not os.path.isfile(model_blob):
        raise RuntimeError(f"model blob が見つかりません: {model_blob}")
    print(f"✅ model: {os.path.basename(model_blob)} "
          f"({os.path.getsize(model_blob) / 1024**3:.2f} GiB)")
    RESULT_DATA["model_blob"] = os.path.basename(model_blob)
    RESULT_DATA["stage"] = "resolve"
    _save()
    return model_blob


def stage_launch(model_blob):
    """llama-server を CPU 専用で起動し、/health が通るまで待つ（起動待ちは流用）。"""
    subprocess.run(["pkill", "-TERM", "-f", "llama-server"], capture_output=True)
    time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)

    args = [
        LLAMA_SERVER,
        "-m", model_blob,
        "-c", str(CTX),
        "-ngl", "0",            # GPU を使わない
        "-t", str(THREADS),
        "-np", "1",
        "--host", HOST,
        "--port", str(PORT),
        "--jinja",
        "--temp", "0",
        "--alias", MODEL_NAME,
        "--reasoning", "off",
        "--reasoning-budget", "0",
    ]
    env = {**os.environ, "LD_LIBRARY_PATH": LIB_DIR}
    print()
    print("起動コマンド:")
    print("  " + " ".join(args))
    print()

    with open(LOG, "w") as lf:
        proc = subprocess.Popen(args, env=env, stdout=lf, stderr=lf)
    print(f"🔄 llama-server PID: {proc.pid}")

    print("⏳ ロード待ち（最大 1800s、CPU なので長め）...")
    ready = False
    t0 = time.time()
    for i in range(360):
        if proc.poll() is not None:
            print(open(LOG).read()[-5000:])
            raise RuntimeError(f"llama-server が即終了しました (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(
                    f"http://{HOST}:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        if i % 12 == 11:
            print(f"   ... {int(time.time() - t0)}s")
        time.sleep(5)

    if not ready:
        print(open(LOG).read()[-6000:])
        proc.kill()
        raise RuntimeError("llama-server が 1800s 以内に応答しませんでした")

    load_s = int(time.time() - t0)
    print(f"✅ 起動完了（{load_s}s）")
    RESULT_DATA["load_seconds"] = load_s
    RESULT_DATA["stage"] = "ready"
    _save()
    return proc


def stage_verify():
    """HTTP でモデル一覧と 1 回の生成を確認する（ask.py の post を流用）。"""
    def get(path):
        with urllib.request.urlopen(f"http://{HOST}:{PORT}{path}", timeout=30) as r:
            return json.loads(r.read().decode())

    def post(payload, timeout=900):
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    models = get("/v1/models")
    ids = [m.get("id") for m in models.get("data", [])]
    print(f"✅ /v1/models: {ids}")
    RESULT_DATA["models"] = ids

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "1から5まで数えて。数字だけ。"}],
        "temperature": 0,
        "max_tokens": 32,
    }
    t0 = time.time()
    data = post(payload)
    dt = time.time() - t0

    msg = data["choices"][0]["message"]
    # thinking 系は content が null で reasoning_content だけ返ることがある（ask.py 流用）
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    usage = data.get("usage") or {}
    gen = usage.get("completion_tokens") or 0
    tps = gen / dt if dt > 0 else 0.0

    print()
    print("--- 生成結果 ---")
    print(repr(text[:200]))
    print(f"prompt {usage.get('prompt_tokens')} tok / completion {gen} tok "
          f"/ {dt:.1f}s → {tps:.2f} tok/s")

    RESULT_DATA.update(
        reply=text[:200], completion_tokens=gen,
        elapsed_seconds=round(dt, 1), tps=round(tps, 2),
    )


def stage_log_summary():
    log_txt = open(LOG).read()
    print()
    print("--- 起動ログの要点 ---")
    keys = ["load time", "using device", "ggml-cpu", "CPU_Mapped", "n_ctx",
            "model size", "model params", "offloaded", "CUDA",
            "prompt eval time", "eval time", "tokens per second"]
    for line in log_txt.splitlines():
        if any(k in line for k in keys):
            print("  " + line.strip()[:140])

    if "CUDA" in log_txt and "no CUDA" not in log_txt:
        RESULT_DATA["cuda_line_seen"] = True
    RESULT_DATA["cpu_backend_seen"] = "ggml-cpu" in log_txt or "CPU" in log_txt


def main():
    dataset_path = _find_input(DATASET_SLUG)
    patch_path = _find_input(PATCH_SLUG)
    print(f"DATASET : {dataset_path}")
    print(f"PATCH   : {patch_path}")
    RESULT_DATA.update(dataset=dataset_path, patch=patch_path)

    stage_env()
    stage_restore(patch_path)
    model_blob = stage_resolve(dataset_path)
    proc = stage_launch(model_blob)
    try:
        stage_verify()
        stage_log_summary()
        RESULT_DATA["cpu_start_ok"] = True
        RESULT_DATA["stage"] = "done"
    finally:
        print()
        print("🛑 停止中...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✅ 停止しました")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        RESULT_DATA["stage"] = "failed"
        RESULT_DATA["error"] = f"{type(e).__name__}: {e}"
    _save()
    print()
    print("RESULT: " + json.dumps(RESULT_DATA, ensure_ascii=False))
    sys.exit(0)

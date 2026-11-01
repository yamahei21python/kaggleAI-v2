#!/usr/bin/env python3
"""tiny-llama: llama-server の高度化フラグを CPU で比較する（GPU 0 秒）。

本番と同じ Ollama 同梱 llama-server を tiny GGUF（0.5B）で起動し、ケースごとに
「起動できるか / 何と言って落ちるか / 生成 t/s」を記録する。

ケース:
  base       … 既定
  kv_q8      … --cache-type-k/v q8_0 + -fa on（本番の KV 量子化）
  spec_mtp   … --spec-type draft-mtp（MTP。CPU では draft 無しで失敗する想定）
  parallel2  … -np 2（並列 2）
  threads4   … -t 4（CPU スレッド数の効果）

CPU の t/s は絶対値に意味は無いが、**同一条件内の相対差**は GPU に持ち込める。
失敗ケースも log_tail を残す（フラグ実在とエラー文の確認が目的）。

出力: /kaggle/working/result.json
"""
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.request

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
GGUF = os.path.join(WORK, "tiny.gguf")
GGUF_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
            "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")
LIB = "/usr/local/lib/ollama"
SERVER = os.path.join(LIB, "llama-server")
PORT = 8089
BASE_URL = f"http://127.0.0.1:{PORT}"

DATA = {"stage": "init", "ok": False, "cases": {}}


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=900, env=None):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def install_ollama():
    """llama-server バイナリを得る（serve はしない）。

    install.sh は `.tar.zst` を優先し、zstd が無いと即 exit 1 する（実測）。
    Kaggle CPU イメージには zstd が入っていないので先に入れる。それでも
    駄目なら tgz を直接展開する。
    """
    if os.path.isfile(SERVER):
        return "既存"
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True,
                   timeout=600)
    subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"],
                   capture_output=True, timeout=600)
    r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, capture_output=True, text=True, timeout=1800)
    log = (r.stdout or "") + (r.stderr or "")
    DATA["install_log"] = log[-1500:]
    if os.path.isfile(SERVER):
        return "install.sh"

    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    tgz = "/tmp/ollama-linux.tgz"
    print(f"⚠ install.sh 失敗 → tgz 直接展開（{arch}）")
    d = sh(["curl", "-fL", "--retry", "3", "-o", tgz,
            f"https://ollama.com/download/ollama-linux-{arch}.tgz"], timeout=1800)
    if d.returncode != 0:
        raise RuntimeError(f"tgz 取得失敗: {d.stderr[-600:]}")
    x = sh(["tar", "-xzf", tgz, "-C", "/usr/local"], timeout=900)
    if not os.path.isfile(SERVER):
        raise RuntimeError(f"展開後も llama-server が無い: {x.stderr[-600:]}")
    return "tgz"


def fetch_gguf():
    if os.path.isfile(GGUF) and os.path.getsize(GGUF) > 100 << 20:
        return
    print(f"⬇️  tiny GGUF: {GGUF_URL}")
    r = sh(["curl", "-fL", "--retry", "3", "-o", GGUF, GGUF_URL], timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"GGUF 取得失敗: {r.stderr[-800:]}")
    DATA["gguf_bytes"] = os.path.getsize(GGUF)


def help_flags():
    r = sh([SERVER, "--help"], timeout=120)
    h = (r.stdout or "") + (r.stderr or "")
    DATA["help_len"] = len(h)
    out = {f: f in h for f in ("--spec-type", "draft-mtp", "--cache-type-k",
                               "--cache-type-v", "-fa", "--jinja",
                               "--reasoning-budget", "-np", "--parallel", "-t")}
    print("フラグ実在:", out)
    return out


def api(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bench(n_predict=64):
    out = api("/completion", {"prompt": "Q: 1+1=? A:", "n_predict": n_predict,
                              "temperature": 0, "cache_prompt": False},
              timeout=600)
    t = out.get("timings") or {}
    return {"gen_tps": round(t.get("predicted_per_second") or 0, 2),
            "prompt_tps": round(t.get("prompt_per_second") or 0, 2),
            "tokens": t.get("predicted_n")}


def run_case(name, extra, ctx=4096, np_=1):
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(1)
    args = [SERVER, "-m", GGUF, "-c", str(ctx), "-ngl", "0",
            "-np", str(np_), "--host", "127.0.0.1", "--port", str(PORT),
            "--jinja", "--temp", "0", "--alias", "tiny",
            "--reasoning", "off", "--reasoning-budget", "0",
            "--cors-origins", "*"] + list(extra)
    log = os.path.join(WORK, f"{name}.log")
    rec = {"args": args[1:]}
    print(f"\n=== {name} ===")
    with open(log, "wb") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=lf, cwd=LIB,
                                env={**os.environ, "LD_LIBRARY_PATH": LIB},
                                start_new_session=True)
    t0 = time.time()
    try:
        ready = False
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(BASE_URL + "/health", timeout=3) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(2)
        rec["startup_s"] = round(time.time() - t0, 1)
        if not ready:
            rec.update(ok=False, fail="起動せず",
                       log_tail=open(log, errors="replace").read()[-1800:])
            return rec
        rec["models"] = [m.get("id") for m in
                         api("/v1/models", timeout=30).get("data", [])]
        rec["bench"] = bench()
        rec["ok"] = bool(rec["bench"]["tokens"])
        print(f"  ok={rec['ok']} startup={rec['startup_s']}s "
              f"tps={rec['bench']['gen_tps']}")
        return rec
    except Exception as e:
        rec.update(ok=False, fail=f"{type(e).__name__}: {e}")
        try:
            rec["log_tail"] = open(log, errors="replace").read()[-1500:]
        except OSError:
            pass
        return rec
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        subprocess.run(["pkill", "-9", "-f", "llama-server"],
                       capture_output=True)


CASES = [
    ("base", [], 4096, 1),
    ("kv_q8", ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "-fa", "on"], 4096, 1),
    ("spec_mtp", ["--spec-type", "draft-mtp"], 4096, 1),
    ("parallel2", [], 8192, 2),
    ("threads4", ["-t", "4"], 4096, 1),
]


def main():
    DATA["stage"] = "install"
    save()
    DATA["install"] = install_ollama()
    fetch_gguf()
    DATA["stage"] = "help"
    DATA["flags"] = help_flags()
    DATA["cpu"] = platform.processor() or platform.machine()
    DATA["cpu_count"] = os.cpu_count()
    save()

    for name, extra, ctx, np_ in CASES:
        try:
            DATA["cases"][name] = run_case(name, extra, ctx=ctx, np_=np_)
        except Exception as e:
            DATA["cases"][name] = {"ok": False,
                                   "fail": f"{type(e).__name__}: {e}"}
        save()

    DATA["stage"] = "done"
    # 「探ること」が目的なので、base が通れば合格。他の失敗は記録として残す。
    DATA["ok"] = bool((DATA["cases"].get("base") or {}).get("ok"))
    DATA["gpu_unknown"] = ["VRAM 収容", "GPU flash attention 実効",
                           "MTP accept 率（CPU では draft が読めない）",
                           "CUDA backend 解決（GGML_BACKEND_PATH）"]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        DATA["stage"] = "failed"
        DATA["error"] = f"{type(e).__name__}: {e}"
    save()
    print("\nRESULT: " + json.dumps(DATA, ensure_ascii=False)[:3000])
    sys.exit(0 if DATA.get("stage") == "done" else 1)

#!/usr/bin/env python3
"""tiny-ollama: Ollama の ctx / KV 量子化を CPU で比較する（GPU 0 秒）。

本番と同じ env（OLLAMA_CONTEXT_LENGTH / OLLAMA_KV_CACHE_TYPE /
OLLAMA_FLASH_ATTENTION / OLLAMA_NUM_PARALLEL）で `ollama serve` を起動し、
tiny モデルでケースごとに以下を記録する。

ケース:
  base      … ctx 4096 / KV f16（基準）
  kv_q8     … ctx 4096 / KV q8_0
  ctx32k    … ctx 32768 / KV q8_0
  ctx128k   … ctx 131072 / KV q8_0（本番設定）

見るもの:
  - /api/ps の context_length が env と一致するか（無言の無視を検出）
  - ロード込みの所要秒と生成 t/s（CPU 相対比較）
  - size_vram（CPU なので常に 0。GPU 時の判定式の予行）

出力: /kaggle/working/result.json
"""
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
MODELS = os.path.join(WORK, "models")
HOST = "127.0.0.1"
PORT = 11434
BASE_URL = f"http://{HOST}:{PORT}"
MODEL = "qwen2.5:0.5b"
GGUF_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
            "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")

DATA = {"stage": "init", "ok": False, "cases": {}}


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=900, env=None):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def api(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def install():
    """install.sh は zstd 必須（Kaggle CPU には無い）→ 先に入れる。"""
    exe = "/usr/local/bin/ollama"
    if os.path.isfile(exe):
        return exe, "既存"
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True,
                   timeout=600)
    subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"],
                   capture_output=True, timeout=600)
    r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, capture_output=True, text=True, timeout=1800)
    DATA["install_log"] = ((r.stdout or "") + (r.stderr or ""))[-1500:]
    if os.path.isfile(exe):
        return exe, "install.sh"
    tgz = "/tmp/ollama-linux.tgz"
    print("⚠ install.sh 失敗 → tgz 直接展開")
    d = sh(["curl", "-fL", "--retry", "3", "-o", tgz,
            "https://ollama.com/download/ollama-linux-amd64.tgz"], timeout=1800)
    if d.returncode != 0:
        raise RuntimeError(f"tgz 取得失敗: {d.stderr[-600:]}")
    sh(["tar", "-xzf", tgz, "-C", "/usr/local"], timeout=900)
    if not os.path.isfile(exe):
        raise RuntimeError("展開後も ollama が無い")
    return exe, "tgz"


def ensure_model(exe, env):
    r = sh([exe, "pull", MODEL], env=env, timeout=1800)
    DATA["pull"] = {"rc": r.returncode,
                    "tail": ((r.stdout or "") + (r.stderr or ""))[-500:]}
    if r.returncode == 0:
        return MODEL
    print("⚠ pull 失敗 → GGUF から create")
    gguf = os.path.join(WORK, "tiny.gguf")
    d = sh(["curl", "-fL", "--retry", "3", "-o", gguf, GGUF_URL], timeout=1800)
    DATA["gguf"] = {"rc": d.returncode,
                    "bytes": os.path.getsize(gguf) if os.path.isfile(gguf) else 0}
    if d.returncode != 0:
        raise RuntimeError(f"GGUF 取得失敗: {d.stderr[-600:]}")
    mf = os.path.join(WORK, "Modelfile")
    with open(mf, "w") as f:
        f.write(f"FROM {gguf}\nPARAMETER temperature 0\n")
    r = sh([exe, "create", "tiny", "-f", mf], env=env, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("ollama create 失敗: " + (r.stderr or "")[-600:])
    return "tiny"


def env_for(ctx, kv, np=1):
    return {
        **os.environ,
        "OLLAMA_MODELS": MODELS,
        "OLLAMA_HOST": f"{HOST}:{PORT}",
        "OLLAMA_ORIGINS": "*",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_CONTEXT_LENGTH": str(ctx),
        "OLLAMA_KV_CACHE_TYPE": kv,
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_LOAD_TIMEOUT": "600",
        "OLLAMA_NUM_PARALLEL": str(np),
    }


def concurrent(n, n_predict=48):
    """同時リクエスト n 本。総スループット（tokens/壁時計）を見る。"""
    import threading
    reqs, lock = [], threading.Lock()

    def worker(i):
        t0 = time.time()
        try:
            g = api("/api/generate", {"model": MODEL, "prompt": "Q: 2+2=? A:",
                                      "stream": False,
                                      "options": {"temperature": 0,
                                                  "num_predict": n_predict}})
            tok = g.get("eval_count") or 0
            dur = (g.get("eval_duration") or 1) / 1e9
            with lock:
                reqs.append({"i": i, "seconds": round(time.time() - t0, 2),
                             "tokens": tok, "tps": round(tok / dur, 2)})
        except Exception as e:
            with lock:
                reqs.append({"i": i, "err": f"{type(e).__name__}: {e}"[:200]})

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    total = sum(r.get("tokens") or 0 for r in reqs)
    return {"n": n, "wall_s": round(wall, 2), "total_tokens": total,
            "aggregate_tps": round(total / wall, 2) if wall else 0,
            "reqs": sorted(reqs, key=lambda r: r["i"])}


def serve(exe, env, log):
    with open(log, "wb") as lf:
        proc = subprocess.Popen([exe, "serve"], env=env, stdout=lf, stderr=lf,
                                start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError("serve 即終了: "
                               + open(log, errors="replace").read()[-1200:])
        try:
            if api("/api/version", timeout=3).get("version"):
                return proc, round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("serve が 180s で応答しない")


def run_case(exe, name, ctx, kv, np_=1, conc=0):
    env = env_for(ctx, kv, np_)
    log = os.path.join(WORK, f"{name}.log")
    rec = {"ctx_env": ctx, "kv": kv}
    print(f"\n=== {name} (ctx={ctx} kv={kv}) ===")
    proc, up_s = serve(exe, env, log)
    rec["serve_up_s"] = up_s
    t0 = time.time()
    try:
        g = api("/api/generate", {"model": MODEL, "prompt": "Q: 1+1=? A:",
                                  "stream": False,
                                  "options": {"temperature": 0,
                                              "num_predict": 64}})
        rec["load_plus_gen_s"] = round(time.time() - t0, 1)
        txt = (g.get("response") or "").strip()
        n = g.get("eval_count") or 0
        dur = (g.get("eval_duration") or 1) / 1e9
        rec["gen_tps"] = round(n / dur, 2)
        rec["tokens"] = n
        rec["load_ms"] = round((g.get("load_duration") or 0) / 1e6, 1)
        rec["prompt_tps"] = round((g.get("prompt_eval_count") or 0)
                                  / ((g.get("prompt_eval_duration") or 1) / 1e9),
                                  2)
        rec["text"] = txt[:40]
        rec["ok"] = bool(txt) and n > 0
    except Exception as e:
        rec.update(ok=False, fail=f"{type(e).__name__}: {e}",
                   log_tail=open(log, errors="replace").read()[-2000:])

    try:
        loaded = (api("/api/ps", timeout=30).get("models") or [{}])[0]
    except Exception:
        loaded = {}
    rec["loaded"] = {k: loaded.get(k) for k in
                     ("size", "size_vram", "context_length")}
    rec["ctx_matches_env"] = loaded.get("context_length") == ctx
    rec["log_warnings"] = [ln for ln in
                           open(log, errors="replace").read().splitlines()
                           if any(k in ln.lower() for k in
                                  ("kv cache", "flash", "not supported"))][-8:]
    if conc > 1:
        rec["concurrent"] = concurrent(conc)
    print(f"  ok={rec['ok']} tps={rec.get('gen_tps')} "
          f"ctx={rec['loaded'].get('context_length')} "
          f"np={np_} load={rec.get('load_ms')}ms"
          + (f" agg_tps={rec['concurrent']['aggregate_tps']}"
             if conc > 1 else ""))
    try:
        api("/api/generate", {"model": MODEL, "keep_alive": 0}, timeout=120)
    except Exception:
        pass
    return proc, rec


# (name, ctx, kv, num_parallel, 同時本数)
CASES = [
    ("base", 4096, "f16", 1, 0),
    ("kv_q8", 4096, "q8_0", 1, 0),
    ("ctx32k", 32768, "q8_0", 1, 0),
    ("ctx128k", 131072, "q8_0", 1, 0),
    # 並列 2。同時 2 本を投げて総スループットを測る（ctx はスロット分割される）
    ("np2", 8192, "q8_0", 2, 2),
]


def main():
    DATA["stage"] = "install"
    save()
    exe, how = install()
    DATA["install"] = how
    os.makedirs(MODELS, exist_ok=True)
    DATA["stage"] = "model"
    # pull / create は API 経由なので、先に serve を立てる（実測: 未起動だと
    # "could not connect to ollama server" で落ちる）。
    boot, _up = serve(exe, env_for(4096, "q8_0"),
                      os.path.join(WORK, "boot.log"))
    try:
        DATA["model"] = ensure_model(exe, env_for(4096, "q8_0"))
    finally:
        boot.terminate()
        try:
            boot.wait(timeout=30)
        except subprocess.TimeoutExpired:
            boot.kill()
        subprocess.run(["pkill", "-9", "-f", "ollama serve"],
                       capture_output=True)
        time.sleep(2)
    save()

    DATA["stage"] = "cases"
    for name, ctx, kv, np_, conc in CASES:
        proc = None
        try:
            proc, rec = run_case(exe, name, ctx, kv, np_, conc)
            DATA["cases"][name] = rec
        except Exception as e:
            DATA["cases"][name] = {"ok": False,
                                   "fail": f"{type(e).__name__}: {e}"}
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            subprocess.run(["pkill", "-9", "-f", "ollama serve"],
                           capture_output=True)
            time.sleep(2)
        save()

    DATA["stage"] = "done"
    DATA["ok"] = bool((DATA["cases"].get("base") or {}).get("ok"))
    DATA["gpu_unknown"] = ["q8_0 KV の実効（CPU では f16 と差が出ない）",
                           "flash attention の実効",
                           "size_vram（GPU 割当）"]


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

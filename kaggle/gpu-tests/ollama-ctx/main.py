#!/usr/bin/env python3
"""gpu-ollama-ctx: Ollama のコンテキスト長 / KV 量子化を 1 セッションで比較する。

KC_EXP（カンマ区切り可。既定 all）:
  32k_q8    … ctx 32768 / KV q8_0（VRAM 余裕の基準点）
  128k_q8   … ctx 131072 / KV q8_0（本番設定）
  128k_f16  … ctx 131072 / KV f16（量子化なし。VRAM と速度の比較）

各ケースは ollama serve を起動し直し（env は起動時に読まれる）、モデルを
ロードして以下を残す:
  ロード秒 / /api/ps の context_length / size_vram（CPU 落ちの検出）/ 生成 t/s
`size_vram` が総 VRAM を超える・0 の場合は CPU 実行として明示的に失敗扱い。

出力: /kaggle/working/result.json
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
DATASET = "my-ollama-orcarouter-qwen38-27b"
PATCH = "my-ollama-orcarouter-qwen38-27b-patch"
MODEL_NAME = "orcarouter/Qwen3.8-27B-Uncensored:latest"
MODELS = "/kaggle/tmp/ollama_models"
HOST = "127.0.0.1"
PORT = 11434
BASE = f"http://{HOST}:{PORT}"
BUDGET = int(os.environ.get("KC_BUDGET", "3300"))
CASE_TIMEOUT = int(os.environ.get("KC_CASE_TIMEOUT", "900"))

DATA = {"stage": "init", "ok": False, "cases": {}}


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=900, env=None):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def find_dir(slug):
    cands = []
    for pat in (f"/kaggle/input/{slug}", f"/kaggle/input/*/{slug}",
                f"/kaggle/input/*/*/{slug}"):
        cands += [p for p in glob.glob(pat) if os.path.isdir(p)]
    for p in sorted(cands):
        try:
            if os.listdir(p):
                return p
        except OSError:
            continue
    return cands[0] if cands else None


def vram():
    """(合計 MiB, 使用 MiB)。取れなければ (-1, -1)。"""
    try:
        out = sh(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                  "--format=csv,noheader,nounits"], timeout=20).stdout
        tot = used = 0
        for ln in out.strip().splitlines():
            a, b = [int(x) for x in ln.split(",")[:2]]
            tot += a
            used += b
        return tot, used
    except Exception:
        return -1, -1


def install():
    exe = "/usr/local/bin/ollama"
    if os.path.isfile(exe):
        return exe, "既存"
    r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, capture_output=True, text=True, timeout=1200)
    if not os.path.isfile(exe):
        raise RuntimeError(f"ollama が入らない: {r.stderr[-800:]}")
    return exe, "install.sh"


def stage(name, roots):
    """blobs / manifests を実 dir + file symlink で配置（実証済み手順）。"""
    dst = os.path.join(MODELS, name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root in roots:
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        for p in sorted(glob.glob(os.path.join(base, "**"), recursive=True)):
            if not os.path.isfile(p):
                continue
            d = os.path.join(dst, os.path.relpath(p, base))
            if name == "manifests":
                if os.path.lexists(d) and not os.path.islink(d) and os.path.isfile(d):
                    continue
                if os.path.lexists(d):
                    os.remove(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(p, d)
                n += 1
                continue
            if os.path.lexists(d) and os.path.isfile(d):
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                os.symlink(p, d)
            except OSError:
                if os.path.getsize(p) > (1 << 30):
                    raise
                shutil.copy2(p, d)
            n += 1
    return n


def api(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def serve(exe, env, log):
    with open(log, "wb") as lf:
        proc = subprocess.Popen([exe, "serve"], env=env, stdout=lf, stderr=lf,
                                start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError("ollama serve が即終了: "
                               + open(log, errors="replace").read()[-1500:])
        try:
            if api("/api/version", timeout=3).get("version"):
                return proc, round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("ollama serve が 180s で応答しない")


def unload(name):
    try:
        api("/api/generate", {"model": name, "keep_alive": 0}, timeout=120)
    except Exception:
        pass


def run_case(exe, name, ctx, kv):
    env = {
        **os.environ,
        "OLLAMA_MODELS": MODELS,
        "OLLAMA_HOST": f"{HOST}:{PORT}",
        "OLLAMA_ORIGINS": "*",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_CONTEXT_LENGTH": str(ctx),
        "OLLAMA_KV_CACHE_TYPE": kv,
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_LOAD_TIMEOUT": "900",
        "OLLAMA_NUM_PARALLEL": "1",
    }
    log = os.path.join(WORK, f"{name}.log")
    rec = {"ctx_env": ctx, "kv": kv}
    tot, used0 = vram()
    rec["vram_before_mib"] = used0
    print(f"\n=== {name} (ctx={ctx} kv={kv}) ===")
    proc, up_s = serve(exe, env, log)
    rec["serve_up_s"] = up_s
    t0 = time.time()
    try:
        g = api("/api/generate", {"model": MODEL_NAME, "prompt": "Q: 1+1=? A:",
                                  "stream": False, "options": {
                                      "temperature": 0, "num_predict": 128}},
                timeout=CASE_TIMEOUT)
        rec["load_plus_gen_s"] = round(time.time() - t0, 1)
        txt = (g.get("response") or "").strip()
        rec["tps"] = round((g.get("eval_count") or 0)
                           / ((g.get("eval_duration") or 1) / 1e9), 2)
        rec["tokens"] = g.get("eval_count")
    except Exception as e:
        rec.update(ok=False, fail=f"{type(e).__name__}: {e}")
        rec["log_tail"] = open(log, errors="replace").read()[-2500:]
        unload(MODEL_NAME)
        return proc, rec

    ps = api("/api/ps", timeout=30)
    loaded = (ps.get("models") or [{}])[0]
    tot, used = vram()
    rec["loaded"] = {k: loaded.get(k) for k in
                     ("name", "size", "size_vram", "context_length")}
    rec["vram_loaded_mib"] = used
    rec["gpu_share"] = (round(loaded.get("size_vram", 0)
                              / max(loaded.get("size", 1), 1), 3)
                        if loaded else None)
    rec["ok"] = bool(txt) and (loaded.get("size_vram") or 0) > 0
    if not rec["ok"]:
        rec["fail"] = ("空応答" if not txt else "size_vram=0（CPU 実行）")
    print(f"  ok={rec['ok']} tps={rec.get('tps')} "
          f"ctx={loaded.get('context_length')} "
          f"vram={(loaded.get('size_vram') or 0) / 1024**3:.1f} GiB")
    unload(MODEL_NAME)
    return proc, rec


CASES = {
    "32k_q8": {"ctx": 32768, "kv": "q8_0"},
    "128k_q8": {"ctx": 131072, "kv": "q8_0"},
    "128k_f16": {"ctx": 131072, "kv": "f16"},
}
ORDER = ["32k_q8", "128k_q8", "128k_f16"]


def select_exps():
    raw = (os.environ.get("KC_EXP") or "all").strip()
    if raw in ("", "all"):
        return list(ORDER)
    want = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in want if x not in CASES]
    if unknown:
        raise RuntimeError(f"未知の KC_EXP: {unknown}（有効: {ORDER}）")
    return want


def main():
    tot, used = vram()
    DATA.update(vram_total_mib=tot, vram_used_mib=used, exps=select_exps())
    save()
    if tot < 0:
        raise RuntimeError("GPU が無い。Accelerator を GPU T4 x2 にして push する。")

    DATA["stage"] = "install"
    t0 = time.time()
    exe, how = install()
    DATA["install"] = {"seconds": round(time.time() - t0, 1), "how": how}
    save()

    DATA["stage"] = "stage"
    ds, patch = find_dir(DATASET), find_dir(PATCH)
    if not ds:
        raise RuntimeError(f"Dataset が無い: {DATASET}")
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    DATA["staging"] = {"blobs": stage("blobs", (ds, patch)),
                       "manifests": stage("manifests", (ds, patch)),
                       "seconds": round(time.time() - t0, 1)}
    save()

    DATA["stage"] = "cases"
    started = time.time()
    for name in DATA["exps"]:
        if time.time() - started > BUDGET:
            DATA["cases"][name] = {"ok": False, "skipped": True}
            save()
            continue
        proc = None
        try:
            proc, rec = run_case(exe, name, CASES[name]["ctx"],
                                 CASES[name]["kv"])
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
            time.sleep(3)
        save()

    DATA["stage"] = "done"
    DATA["total_s"] = round(time.time() - started, 1)
    DATA["ok"] = bool(DATA["cases"]) and all(
        c.get("ok") for c in DATA["cases"].values())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        DATA["stage"] = "failed"
        DATA["error"] = f"{type(e).__name__}: {e}"
    save()
    print("\nRESULT: " + json.dumps(DATA, ensure_ascii=False)[:3000])
    sys.exit(0 if DATA.get("ok") else 1)

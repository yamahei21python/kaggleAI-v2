#!/usr/bin/env python3
"""gpu-llama-mtp: llama-server 直起動の高度化を 1 セッションで比較する GPU テスト。

KC_EXP（カンマ区切り可。既定 all）:
  none       … 既定（MTP なし・KV q8_0・-np 1）
  draft_mtp  … --spec-type draft-mtp（MTP 有効）
  parallel2  … -np 2（並列 2。ctx は 32768 に落として VRAM を確保）
  kv_f16     … KV を f16（量子化なし。VRAM と速度の比較用）

各ケースは llama-server を起動し直し、以下を結果に残す:
  ロード秒 / health 到達秒 / 生成 t/s / MTP の draft 受入率（timings）/ VRAM
失敗・ハングはケース単位で隔離（1 本落ちても次へ進む）。

出力: /kaggle/working/result.json
"""
import glob
import json
import os
import shutil
import signal
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
LIB = "/usr/local/lib/ollama"
PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"
BUDGET = int(os.environ.get("KC_BUDGET", "3300"))
CASE_TIMEOUT = int(os.environ.get("KC_CASE_TIMEOUT", "900"))

DATA = {"stage": "init", "ok": False, "cases": {},
        "budget_s": BUDGET, "case_timeout_s": CASE_TIMEOUT}


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


def nvidia():
    try:
        out = sh(["nvidia-smi", "--query-gpu=memory.used",
                  "--format=csv,noheader,nounits"], timeout=20).stdout
        return sum(int(x) for x in out.split())
    except Exception:
        return -1


def install_ollama():
    """llama-server バイナリだけ得たい（serve はしない）。"""
    server = os.path.join(LIB, "llama-server")
    if os.path.isfile(server):
        return server, "既存"
    r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, capture_output=True, text=True, timeout=1200)
    if not os.path.isfile(server):
        raise RuntimeError(f"llama-server が入らない: {r.stderr[-800:]}")
    return server, "install.sh"


def stage(name, roots):
    """Dataset の blobs / manifests を実 dir + ファイル symlink で配置。

    dir ごと symlink にすると読み取り専用マウント上で ollama が欠損補完に
    失敗する（実測）。manifest は ollama が symlink を拒否するので実体コピー。
    """
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


def model_blob():
    repo, tag = MODEL_NAME.rsplit(":", 1)
    man = os.path.join(MODELS, "manifests", "registry.ollama.ai",
                       *repo.split("/"), tag)
    if not os.path.isfile(man):
        raise RuntimeError(f"manifest が無い: {man}")
    with open(man) as f:
        mf = json.load(f)
    need = [mf.get("config", {})] + mf.get("layers", [])
    miss = [e["digest"] for e in need
            if e.get("digest") and not os.path.lexists(os.path.join(
                MODELS, "blobs", "sha256-" + e["digest"].split(":", 1)[1]))]
    if miss:
        raise RuntimeError(f"blob 欠落 {len(miss)} 件: {miss[:5]}")
    model = mmproj = None
    for layer in mf.get("layers", []):
        dg = layer.get("digest", "")
        if not dg.startswith("sha256:"):
            continue
        p = os.path.join(MODELS, "blobs", "sha256-" + dg.split(":", 1)[1])
        mt = layer.get("mediaType", "")
        if mt == "application/vnd.ollama.image.model":
            model = p
        elif mt == "application/vnd.ollama.image.projector":
            mmproj = p
    if not model or not os.path.isfile(model):
        raise RuntimeError(f"model blob が無い: {model}")
    return model, (mmproj if mmproj and os.path.isfile(mmproj) else None)


def lib_env():
    """LD_LIBRARY_PATH + GGML_BACKEND_PATH + libcuda.so.1 を揃える。

    欠けると**エラーを出さず CPU 推論**に落ちる（実測 0.8 t/s）。
    """
    cuda_so = os.path.join(LIB, "cuda_v12", "libggml-cuda.so")
    driver = None
    for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
              "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
              "/usr/local/cuda/lib64", "/usr/lib"):
        if os.path.exists(os.path.join(d, "libcuda.so.1")):
            driver = d
            break
    if driver is None:
        found = sh(["find", "/usr", "/opt", "-maxdepth", "6", "-name",
                    "libcuda.so.1"]).stdout.split()
        if found:
            driver = os.path.dirname(found[0])
    dirs = [LIB, os.path.join(LIB, "cuda_v12")]
    if driver:
        dirs.append(driver)
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(dirs)}
    if os.path.isfile(cuda_so):
        env["GGML_BACKEND_PATH"] = cuda_so
    return env, driver, cuda_so


def help_flags(server):
    p = sh([server, "--help"], timeout=180)
    txt = (p.stdout or "") + (p.stderr or "")
    return {f: f in txt for f in ("--spec-type", "draft-mtp", "--draft-max",
                                  "--cache-type-k", "-fa", "--jinja",
                                  "--parallel", "-np", "--reasoning-budget",
                                  "--tensor-split", "--split-mode", "-b", "-ub")}


def api(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bench(n_predict=128, prompt="Q: 1+1=? A:", cache_prompt=False):
    """/completion で timings（draft 受入率込み）を取る。"""
    out = api("/completion", {"prompt": prompt, "n_predict": n_predict,
                              "temperature": 0, "cache_prompt": cache_prompt},
              timeout=600)
    t = out.get("timings") or {}
    return {"tps": round(t.get("predicted_per_second") or 0, 2),
            "tokens": t.get("predicted_n"),
            "draft_n": t.get("draft_n"),
            "draft_accepted": t.get("draft_n_accepted"),
            "accept_rate": (round((t.get("draft_n_accepted") or 0)
                                  / t["draft_n"], 3)
                            if t.get("draft_n") else None)}


def wait_health(proc, log, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(3)
    return None


def kill_all():
    for pat in ("llama-server",):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(2)
    for _ in range(24):
        used = nvidia()
        if 0 <= used < 2000:
            return used
        time.sleep(5)
    return nvidia()


def run_case(name, case, server, blob, mmproj, env):
    missing = [f for f in case.get("requires", [])
               if not DATA.get("flags", {}).get(f, False)]
    if missing:
        return {"ok": False, "skipped": True,
                "reason": "unsupported flags", "missing_flags": missing,
                "case": dict(case)}
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2)
    args = [server, "-m", blob, "-c", str(case["ctx"]),
            "-b", str(case.get("batch", 512)),
            "-ub", str(case.get("ubatch", 512)),
            "--cache-type-k", case.get("kv", "q8_0"),
            "--cache-type-v", case.get("kv", "q8_0"),
            "-ngl", "all", "-fa", "on", "-np", str(case.get("np", 1)),
            "--host", "127.0.0.1", "--port", str(PORT),
            "--jinja", "--temp", "0", "--alias", MODEL_NAME,
            "--reasoning", "off", "--reasoning-budget", "0",
            "--cors-origins", "*"] + case.get("extra", [])
    if mmproj:
        args += ["--mmproj", mmproj]
    log = os.path.join(WORK, f"{name}.log")
    rec = {"args": args[1:], "vram_before_mib": nvidia(),
           "case": dict(case)}
    print(f"\n=== {name} ===")
    print("起動:", " ".join(args))
    t0 = time.time()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(args, env=env, stdout=lf, stderr=lf,
                                start_new_session=True)
    try:
        ready = wait_health(proc, log, CASE_TIMEOUT)
        rec["startup_s"] = round(time.time() - t0, 1)
        if ready is None:
            rec["ok"] = False
            rec["fail"] = ("即終了" if proc.poll() is not None
                           else f"{CASE_TIMEOUT}s で health 未達")
            rec["log_tail"] = open(log, errors="replace").read()[-2500:]
            return rec
        rec["vram_loaded_mib"] = nvidia()
        rec["models"] = [m.get("id") for m in
                         api("/v1/models", timeout=30).get("data", [])]
        rec["chat"] = ((api("/v1/chat/completions", {
            "model": MODEL_NAME, "temperature": 0, "max_tokens": 24,
            "messages": [{"role": "user", "content": "1+1=? only digits"}]},
            timeout=600).get("choices") or [{}])[0].get("message") or {}
            ).get("content", "")[:40]
        rec["bench"] = bench()
        if case.get("repeat_prompt"):
            rec["prefix_repeat"] = {
                "first": bench(prompt=case["repeat_prompt"], cache_prompt=True),
                "second": bench(prompt=case["repeat_prompt"], cache_prompt=True),
            }
        rec["ok"] = bool(rec["bench"]["tokens"])
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
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        subprocess.run(["pkill", "-9", "-f", "llama-server"],
                       capture_output=True)
        time.sleep(2)


CASES = {
    "none": {"ctx": 65536, "requires": ["--cache-type-k", "-fa", "-np", "-b", "-ub"]},
    "draft_mtp": {"ctx": 65536, "extra": ["--spec-type", "draft-mtp"],
                   "requires": ["--spec-type"]},
    "mtp_tok1": {"ctx": 65536, "extra": ["--spec-type", "draft-mtp",
                 "--draft-max", "1"], "requires": ["--draft-max"]},
    "mtp_tok4": {"ctx": 65536, "extra": ["--spec-type", "draft-mtp",
                 "--draft-max", "4"], "requires": ["--draft-max"]},
    "parallel2": {"ctx": 32768, "np": 2, "requires": ["-np"]},
    "parallel4": {"ctx": 16384, "np": 4, "requires": ["-np"]},
    "kv_f16": {"ctx": 32768, "kv": "f16", "requires": ["--cache-type-k"]},
    "kv_q4": {"ctx": 32768, "kv": "q4_0", "requires": ["--cache-type-k"]},
    "batch128": {"ctx": 32768, "batch": 128, "ubatch": 128, "requires": ["-b", "-ub"]},
    "batch256": {"ctx": 32768, "batch": 256, "ubatch": 256, "requires": ["-b", "-ub"]},
    "batch1024": {"ctx": 32768, "batch": 1024, "ubatch": 1024, "requires": ["-b", "-ub"]},
    "split4852": {"ctx": 32768, "extra": ["--tensor-split", "48,52"]},
    "split4555": {"ctx": 32768, "extra": ["--tensor-split", "45,55"]},
    "split_row": {"ctx": 32768, "extra": ["--split-mode", "row",
                 "--tensor-split", "50,50"],
                  "requires": ["--split-mode", "--tensor-split"]},
    "prefix_repeat": {"ctx": 32768, "repeat_prompt":
                       "固定 system prompt: あなたは厳密なコードレビュアー。"},
}
# 既定は短い基準。重い sweep は KC_EXP で明示実行。
ORDER = ["none", "draft_mtp", "parallel2", "kv_f16"]


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
    DATA["stage"] = "gpu"
    DATA["vram_total_mib"] = nvidia()
    DATA["exps"] = select_exps()
    save()
    if DATA["vram_total_mib"] < 0:
        raise RuntimeError("GPU が無い。Accelerator を GPU T4 x2 にして push する。")

    DATA["stage"] = "install"
    t0 = time.time()
    server, how = install_ollama()
    DATA["install"] = {"seconds": round(time.time() - t0, 1), "how": how}
    DATA["flags"] = help_flags(server)
    print("フラグ:", DATA["flags"])
    save()

    DATA["stage"] = "stage"
    ds, patch = find_dir(DATASET), find_dir(PATCH)
    if not ds:
        raise RuntimeError(f"Dataset が無い: {DATASET}")
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    nb = stage("blobs", (ds, patch))
    nm = stage("manifests", (ds, patch))
    DATA["staging"] = {"blobs": nb, "manifests": nm,
                       "seconds": round(time.time() - t0, 1)}
    blob, mmproj = model_blob()
    DATA["model_gib"] = round(os.path.getsize(blob) / 1024**3, 2)
    env, driver, cuda_so = lib_env()
    DATA["lib"] = {"driver_dir": driver,
                   "cuda_backend": os.path.basename(cuda_so)
                   if os.path.isfile(cuda_so) else None}
    print(f"model {DATA['model_gib']} GiB / cuda={DATA['lib']['cuda_backend']}")
    save()

    DATA["stage"] = "cases"
    started = time.time()
    for name in DATA["exps"]:
        if time.time() - started > BUDGET:
            DATA["cases"][name] = {"ok": False, "skipped": True}
            save()
            continue
        try:
            DATA["cases"][name] = run_case(name, CASES[name], server, blob,
                                           mmproj, env)
        except Exception as e:
            DATA["cases"][name] = {"ok": False,
                                   "fail": f"{type(e).__name__}: {e}"}
        save()

    DATA["stage"] = "done"
    DATA["total_s"] = round(time.time() - started, 1)
    DATA["ok"] = bool(DATA["cases"]) and all(
        c.get("ok") or c.get("skipped") for c in DATA["cases"].values())


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

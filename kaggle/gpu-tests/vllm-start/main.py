#!/usr/bin/env python3
"""gpu-vllm-start: vLLM の起動時間・起動可否を 1 セッションで比較する GPU テスト。

KC_EXP（カンマ区切り可。既定 all）:
  baseline   … max-model-len 65536 / 既定（compile + graph capture あり）
  eager      … 同 + --enforce-eager（graph capture を飛ばす）
  maxlen8192 … max-model-len 8192（KV を減らして起動を軽くする）
  mtp_draft  … maxlen 8192 + MTP + draft dir（MTP head だけの最小 dir）

1 ケース = 1 回の serve 起動。ハングは「ログが KC_STALL 秒伸びない」で検知して
打ち切り、次のケースへ進む（1 本のハングでセッション全体を失わない）。

出力: /kaggle/working/result.json（ケース別 phase 実秒 / tps / log_tail）
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
MODEL_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
WH_SLUGS = ("vllm-wheelhouse", "vllm-wheelhouse-patch")
STAGE = "/kaggle/tmp/vllm-model"
DRAFT = "/kaggle/tmp/vllm-draft"
MTP_FILE = "model-mtp.safetensors"
DRAFT_KEEP = ("config", "tokenizer", "chat_template", "generation_config",
              "special_tokens_map", "vocab", "merges", "added_tokens")
PORT = 8000
SERVED = "qwen38"
VLLM_SPEC = os.environ.get("KC_VLLM_SPEC", "vllm==0.32.0")

BUDGET = int(os.environ.get("KC_BUDGET", "3300"))       # 全体予算（秒）
CASE_TIMEOUT = int(os.environ.get("KC_CASE_TIMEOUT", "900"))
STALL = int(os.environ.get("KC_STALL", "240"))          # ログ無進展の打ち切り

DATA = {"stage": "init", "ok": False, "cases": {},
        "budget_s": BUDGET, "case_timeout_s": CASE_TIMEOUT}

# 起動ログの節目。初出の経過秒を phase に残す。
MARKERS = [
    ("load_weights", re.compile(r"Loading (model weights|safetensors)|"
                                r"Start to load model weights", re.I)),
    ("weights_done", re.compile(r"Model loading took|"
                                r"Loading model weights took", re.I)),
    ("compile", re.compile(r"torch\.compile|Compil(e|ing)|compilation", re.I)),
    ("capture", re.compile(r"(Capturing|Capture).{0,20}(CUDA )?graph", re.I)),
    ("engine", re.compile(r"init engine|Starting engine|EngineCore", re.I)),
    ("server", re.compile(r"Starting vLLM server|Uvicorn running|"
                          r"Application startup complete", re.I)),
    # 2026-10-29 実測のハング署名（60 秒間隔で出続ける）
    ("hang_sig", re.compile(r"No available shared memory broadcast block", re.I)),
]


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=3600, env=None, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=env, check=check)


def find_dir(slug):
    """Dataset のマウント先（旧 / 新レイアウト両対応）。"""
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


def _wheels(d):
    return glob.glob(os.path.join(d, "*.whl"))


def wheel_dir(root, tag):
    """zip 置き場 / 展開済み / サブディレクトリを吸収して wheel 置き場を返す。"""
    if not root:
        return None
    if _wheels(root):
        return root
    zips = (glob.glob(os.path.join(root, "*.zip"))
            + glob.glob(os.path.join(root, "*", "*.zip")))
    if zips:
        dst = os.path.join("/kaggle/tmp/whl", tag)
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        import zipfile
        for z in zips:
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)
        print(f"  zip 展開 {tag}: {time.time() - t0:.1f}s")
        root = dst
    if _wheels(root):
        return root
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(sub) and _wheels(sub):
            return sub
    return None


def install_vllm():
    """wheelhouse（本体 + patch）→ 失敗なら PyPI。install 秒数を返す。"""
    find, dirs = [], []
    for slug in WH_SLUGS:
        d = wheel_dir(find_dir(slug), slug)
        if not d:
            print(f"ℹ {slug}: wheel 無し")
            continue
        print(f"📦 {slug}: {len(_wheels(d))} wheel / {d}")
        find += ["--find-links", d]
        dirs.append(d)
    DATA["wheelhouse"] = [os.path.basename(d) for d in dirs]

    t0 = time.time()
    base = [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
    p = sh(base + (["--no-index"] + find if find else []) + [VLLM_SPEC],
           timeout=3600)
    print((p.stdout or "")[-1500:])
    print((p.stderr or "")[-1500:])
    used = "wheelhouse" if find else "pypi"
    if p.returncode != 0 and find:
        print("⚠ wheelhouse で解決不能 → PyPI にフォールバック")
        used = "pypi-fallback"
        p = sh(base + [VLLM_SPEC], timeout=3600)
        print((p.stdout or "")[-1500:])
        print((p.stderr or "")[-1500:])
    DATA["install"] = {"source": used, "rc": p.returncode,
                       "seconds": round(time.time() - t0, 1)}
    if p.returncode != 0:
        raise RuntimeError("vLLM の install に失敗（上のログ参照）")
    return DATA["install"]["seconds"]


def nvidia():
    try:
        out = sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                  "--format=csv,noheader,nounits"], timeout=20).stdout
    except Exception:
        return {"names": [], "total_mib": 0, "used_mib": 0}
    names, total, used = [], 0, 0
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 3:
            names.append(parts[0])
            total += int(parts[1])
            used += int(parts[2])
    return {"names": names, "total_mib": total, "used_mib": used}


def model_src():
    root = find_dir(MODEL_SLUG)
    if not root:
        raise RuntimeError(f"モデル Dataset が無い: {MODEL_SLUG}")
    if os.path.isfile(os.path.join(root, "config.json")):
        return root
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isfile(os.path.join(sub, "config.json")):
            return sub
    raise RuntimeError(f"config.json が無い: {root}")


def stage_model(src, with_mtp):
    """symlink で 1 か所にまとめる（19 GB をコピーしない）。"""
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)      # symlink のみなので実体は消えない
    os.makedirs(STAGE, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(src)):
        if fn.startswith(".") or (fn == MTP_FILE and not with_mtp):
            continue
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            os.symlink(p, os.path.join(STAGE, fn))
            n += 1
    return n


def stage_draft(src):
    """MTP head + config 類だけの最小 dir（本体 17 GiB の二重読みを防ぐ）。"""
    if not os.path.isfile(os.path.join(src, MTP_FILE)):
        return 0
    if os.path.isdir(DRAFT):
        shutil.rmtree(DRAFT)
    os.makedirs(DRAFT, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(src)):
        if fn.startswith(".") or (fn != MTP_FILE and
                                  not fn.startswith(DRAFT_KEEP)):
            continue
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            os.symlink(p, os.path.join(DRAFT, fn))
            n += 1
    return n


def vllm_flags():
    """`vllm serve --help` からフラグ実在を取る（1 回だけ）。"""
    exe = shutil.which("vllm")
    cmd = ([exe, "serve", "--help"] if exe
           else [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                 "--help"])
    try:
        p = sh(cmd, timeout=300)
    except Exception as e:
        return {"help_error": f"{type(e).__name__}: {e}"}
    txt = (p.stdout or "") + (p.stderr or "")
    flags = ["--trust-remote-code", "--chat-template", "--speculative-config",
             "--enforce-eager", "--language-model-only", "--reasoning-parser",
             "--max-model-len", "--kv-cache-dtype", "--enable-prefix-caching",
             "--max-num-seqs"]
    return {f: f in txt for f in flags}


def server_cmd(maxlen, extra, spec):
    exe = shutil.which("vllm")
    if exe:
        cmd = [exe, "serve", STAGE]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", STAGE]
    cmd += ["--served-model-name", SERVED, "--host", "127.0.0.1",
            "--port", str(PORT),
            "--tensor-parallel-size", str(DATA.get("tp", 2)),
            "--dtype", "half",
            "--gpu-memory-utilization", "0.90",
            "--kv-cache-dtype", "auto",
            "--max-model-len", str(maxlen)]
    if DATA.get("flags", {}).get("--trust-remote-code"):
        cmd.append("--trust-remote-code")
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    tpl = os.path.join(STAGE, "chat_template.jinja")
    if os.path.isfile(tpl) and DATA.get("flags", {}).get("--chat-template"):
        cmd += ["--chat-template", tpl]
    return cmd + list(extra)


def alive_models(timeout=5):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/models", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def generate(max_tokens=64):
    """1 往復生成して t/s を測る（初回 forward の実効確認を兼ねる）。"""
    body = json.dumps({
        "model": SERVED,
        "messages": [{"role": "user", "content": "1+1=? answer only"}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    usage = d.get("usage") or {}
    n = usage.get("completion_tokens") or 0
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    return {"seconds": round(dt, 2), "tokens": n,
            "tps": round(n / dt, 2) if dt and n else 0,
            "text": ((msg.get("content") or msg.get("reasoning_content") or "")
                     .strip()[:60])}


def kill_all():
    """プロセス木ごと落とし、VRAM と /dev/shm を空ける。

    SIGTERM は API サーバーにしか届かず Worker_TP* が残る（2026-10-29 実測）。
    残骸があると次のケースが "Engine core initialization failed" で即死する。
    """
    for pat in ("vllm serve", "VLLM::EngineCore", "VLLM::WorkerTP"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    try:
        for fn in os.listdir("/dev/shm"):
            if fn.startswith(("vllm", "psm_")):
                try:
                    os.unlink(os.path.join("/dev/shm", fn))
                except OSError:
                    pass
    except OSError:
        pass
    for _ in range(24):
        used = nvidia()["used_mib"]
        if used < 2000:
            return used
        time.sleep(5)
    return nvidia()["used_mib"]


def run_case(name, spec_conf):
    missing = [f for f in spec_conf.get("requires", [])
               if not DATA.get("flags", {}).get(f, False)]
    if missing:
        return {"ok": False, "skipped": True, "missing_flags": missing}
    maxlen = spec_conf["maxlen"]
    extra = spec_conf.get("extra", [])
    spec = None
    draft_n = 0
    if spec_conf.get("spec") and name == "mtp_draft":
        draft_n = stage_draft(DATA["model_src"])
        if draft_n:
            spec = {**spec_conf["spec"], "model": DRAFT}
        elif "mtp_draft" in name:
            # draft 専用 dir 作成不可（MTP 層なし等）→ 本体 dir に戻す
            spec = {k: v for k, v in spec_conf["spec"].items() if k != "model"}
    stage_model(DATA["model_src"], with_mtp=spec is not None)
    cmd = server_cmd(maxlen, extra, spec)

    rec = {"maxlen": maxlen, "extra": extra, "spec": spec,
           "draft_symlinks": draft_n, "phase": {}}
    print(f"\n=== {name} (spec={'MTP' if spec else 'none'}) ===")
    print("起動:", " ".join(cmd))
    log = os.path.join(WORK, f"{name}.log")
    used0 = kill_all()
    rec["vram_before_mib"] = used0
    t0 = time.time()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                start_new_session=True)
    ready = False
    stall = False
    with open(log, "rb") as lf:
        pending, last_growth = "", time.time()
        while time.time() - t0 < CASE_TIMEOUT:
            if proc.poll() is not None:
                rec["exit_code"] = proc.returncode
                break
            chunk = lf.read()
            if chunk:
                pending += chunk.decode("utf-8", "replace")
                lines = pending.split("\n")
                pending = lines[-1]
                last_growth = time.time()
                for ln in lines:
                    for key, rx in MARKERS:
                        if key not in rec["phase"] and rx.search(ln):
                            rec["phase"][key] = round(time.time() - t0, 1)
            if alive_models():
                ready = True
                rec["ready_s"] = round(time.time() - t0, 1)
                break
            if time.time() - last_growth > STALL:
                stall = True
                break
            time.sleep(2)

    rec["startup_s"] = round(time.time() - t0, 1)
    if ready:
        rec["gen"] = generate()
        rec["vram_after_mib"] = nvidia()["used_mib"]
        rec["ok"] = rec["gen"]["tokens"] > 0
    else:
        rec["ok"] = False
        rec["fail"] = "stalled" if stall else "not ready"
        rec["log_tail"] = open(log, errors="replace").read()[-2500:]
    try:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        kill_all()
    print(f"  ok={rec['ok']} startup={rec['startup_s']}s "
          f"phase={rec.get('phase')} fail={rec.get('fail', '')}")
    return rec


# (name, maxlen, extra, spec)
# baseline/eager/maxlen8192 は現行のプラン。
# mtp_full / mtp_draft は P0-1（MTP 重み二重読込解消）の比較:
#   mtp_full  = speculative config + 本体 dir 全体（draft も含めて 17 GiB 再読込）
#   mtp_draft = draft 専用最小 dir（model-mtp.safetensors 0.79 GiB のみ）
#   → "load_weights" 節の秒差で二重読込の有無を確認
# prefixcache は P1-6。--enable-prefix-caching 受理と、
# --enforce-eager 起動でカーネルキャッシュが再利用されるか（compile 秒）を見る。
CASES = {
    "baseline": {"maxlen": 65536, "extra": [], "spec": None},
    "eager": {"maxlen": 65536, "extra": ["--enforce-eager"], "spec": None,
              "requires": ["--enforce-eager"]},
    "maxlen8192": {"maxlen": 8192, "extra": [], "spec": None},
    "mtp_full": {"maxlen": 65536, "extra": [],
                 "spec": {"method": "mtp",
                          "num_speculative_tokens": 2}},
    "mtp_draft": {"maxlen": 8192, "extra": [],
                  "spec": {"method": "mtp",
                           "num_speculative_tokens": 2,
                           "model": DRAFT},
                  "requires": ["--speculative-config"]},
    "prefixcache": {"maxlen": 8192, "extra": ["--enable-prefix-caching"],
                    "spec": None, "requires": ["--enable-prefix-caching"]},
    "maxseq1": {"maxlen": 32768, "extra": ["--max-num-seqs", "1"],
                 "spec": None, "requires": ["--max-num-seqs"]},
    "maxseq4": {"maxlen": 16384, "extra": ["--max-num-seqs", "4"],
                 "spec": None, "requires": ["--max-num-seqs"]},
    "profile_fast": {"maxlen": 8192, "extra": ["--max-num-seqs", "1"],
                      "spec": None},
    "profile_normal": {"maxlen": 32768, "extra": ["--max-num-seqs", "2"],
                        "spec": None},
    "profile_long": {"maxlen": 131072, "extra": ["--max-num-seqs", "1"],
                      "spec": None},
    "profile_batch": {"maxlen": 16384, "extra": ["--max-num-seqs", "4"],
                       "spec": None},
}
ORDER = ["baseline", "eager", "maxlen8192", "mtp_full", "mtp_draft",
         "prefixcache"]


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
    DATA["gpu"] = nvidia()
    DATA["exps"] = select_exps()
    save()
    if not DATA["gpu"]["names"]:
        raise RuntimeError("GPU が無い。Accelerator を GPU T4 x2 にして push する。")
    DATA["tp"] = 2 if len(DATA["gpu"]["names"]) >= 2 else 1

    DATA["stage"] = "install"
    install_vllm()
    save()

    DATA["stage"] = "flags"
    DATA["flags"] = vllm_flags()
    print("フラグ:", DATA["flags"])
    DATA["model_src"] = model_src()
    save()

    DATA["stage"] = "cases"
    started = time.time()
    for name in DATA["exps"]:
        spent = time.time() - started
        if spent > BUDGET:
            DATA["cases"][name] = {"ok": False, "skipped": True,
                                   "reason": f"予算超過 {int(spent)}s"}
            save()
            continue
        try:
            DATA["cases"][name] = run_case(name, CASES[name])
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

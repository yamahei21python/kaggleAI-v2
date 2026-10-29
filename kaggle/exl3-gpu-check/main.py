#!/usr/bin/env python3
"""ExLlamaV3 (EXL3) の Kaggle GPU 起動検証。

1 回の GPU セッションで最小限を通す:

  1. 環境（GPU / torch / CUDA / nvcc / disk）
  2. exllamav3 の導入（torch/CUDA 一致のプリビルドwheel → 無ければ JIT）
  3. import 検証（JIT の場合はここでコンパイルが走る）
  4. モデル取得（マウント済み Dataset → 無ければ HF から DL）
  5. 2x T4 にレイヤー分割でロード
  6. 1 回だけ生成して tok/s を測る
  7. MTP ヘッドを載せて投機デコードを実測（失敗しても継続）

CPU イメージでは import すら不可（CUDA_HOME 無し）と確定済みなので、
GPU クォータを使うのはこのカーネルだけに絞る。サーバは立てない。
"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

RESULT = "/kaggle/working/result.json"
MODEL_REPO = "Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw"
EXL3_VERSION = "1.6.5"

data = {"stage": "init"}
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
            stdout = (e.stdout or "")
            stderr = (e.stderr or "") + "\n[timeout]"
        return _R(), round(time.time() - t0, 1)


def tail(s, n=40):
    return "\n".join((s or "").strip().splitlines()[-n:])


def free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


# ── 1. 環境 ─────────────────────────────────────────────────
def stage_env():
    print("=" * 60)
    print("ExLlamaV3 GPU check")
    print("=" * 60)
    print(f"python : {sys.version.split()[0]}")

    r, _ = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader"])
    print("nvidia-smi:")
    print(r.stdout.strip() or r.stderr.strip())
    data["nvidia_smi"] = r.stdout.strip()

    import torch
    print(f"torch  : {torch.__version__}")
    print(f"torch cuda build : {torch.version.cuda}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    print(f"device count     : {torch.cuda.device_count()}")
    data.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                device_count=torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が使えません（GPU セッションになっていない）")

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name}  {p.total_memory / 1024**3:.1f} GiB  sm{p.major}{p.minor}")

    nvcc = next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))), None)
    if nvcc:
        rr, _ = sh([nvcc, "--version"])
        line = next((ln for ln in rr.stdout.splitlines() if "release" in ln), "")
        print(f"nvcc   : {nvcc}  {line.strip()}")
        data.update(nvcc=nvcc, nvcc_release=line.strip())
    else:
        print("nvcc   : 見つからない（JIT ビルド不可・wheel 必須）")
        data["nvcc"] = None

    for p in ("/kaggle/working", "/kaggle/temp", "/tmp"):
        if os.path.isdir(p):
            print(f"disk   : {p} {free_gib(p):.1f} GiB free")

    data["stage"] = "env"
    save()


# ── 2. インストール ─────────────────────────────────────────
def pick_wheel_url():
    """torch/CUDA/python が一致するプリビルド wheel の URL。無ければ None。"""
    import torch
    tv = torch.__version__.split("+")[0]
    cu = (torch.version.cuda or "").replace(".", "")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if cu != "128" or tv not in ("2.8.0", "2.10.0", "2.11.0"):
        return None
    name = (f"exllamav3-{EXL3_VERSION}+cu{cu}.torch{tv}-"
            f"{py}-{py}-linux_x86_64.whl")
    url = ("https://github.com/turboderp-org/exllamav3/releases/download/"
           f"v{EXL3_VERSION}/{name}")
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30):
            return url
    except Exception:
        return None


def stage_install():
    print()
    print("--- pip install exllamav3 ---")
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
           "MAX_JOBS": str(min(4, os.cpu_count() or 4))}

    url = pick_wheel_url()
    if url:
        print(f"プリビルド wheel: {os.path.basename(url)}")
        data["install_method"] = "wheel"
        data["wheel_url"] = url
        target = url
    else:
        print("一致する wheel なし → PyPI 版（初回 import で JIT コンパイル）")
        data["install_method"] = "pypi_jit"
        target = "exllamav3"

    r, dt = sh([sys.executable, "-m", "pip", "install", "-U",
                "--no-cache-dir", target], timeout=1800, env=env)
    print(f"rc={r.returncode}  {dt}s")
    print(tail(r.stdout + "\n" + r.stderr, 30))
    data.update(pip_rc=r.returncode, pip_seconds=dt)
    if r.returncode != 0:
        data["pip_error_tail"] = tail(r.stderr, 60)
        raise RuntimeError("pip install 失敗")

    # 任意: DL 高速化
    sh([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
       timeout=300, env=env)

    data["stage"] = "install"
    save()


# ── 3. import（JIT ならここでコンパイル）─────────────────────
def stage_import():
    print()
    print("--- import exllamav3 ---")
    code = (
        "import json, exllamav3\n"
        "print(json.dumps({'version': getattr(exllamav3, '__version__', '?'),\n"
        "                  'dir': exllamav3.__file__}))\n"
    )
    r, dt = sh([sys.executable, "-c", code], timeout=5400)
    print(f"rc={r.returncode}  {dt}s")
    print(tail(r.stdout + "\n" + r.stderr, 30))
    data.update(import_ok=r.returncode == 0, import_seconds=dt)
    if r.returncode != 0:
        data["import_error_tail"] = tail(r.stderr, 60)
        raise RuntimeError("import exllamav3 失敗")
    try:
        data["exllamav3_version"] = json.loads(
            r.stdout.strip().splitlines()[-1])["version"]
    except Exception:
        pass
    data["stage"] = "import"
    save()


# ── 4. モデル DL ────────────────────────────────────────────
def pick_model_dir(need_gib=19.0):
    best = None
    for c in ("/kaggle/temp", "/tmp", "/kaggle/working"):
        if not os.path.isdir(c):
            continue
        g = free_gib(c)
        print(f"  free {c}: {g:.1f} GiB")
        if g >= need_gib and (best is None or g > best[1]):
            best = (c, g)
    if best is None:
        raise RuntimeError(f"{need_gib:.0f} GiB 以上の空きがあるディレクトリが無い")
    return os.path.join(best[0], "exl3-model")


def find_mounted_model():
    """入力 Dataset として既にマウントされていればそのパス。無ければ None。"""
    pats = (
        "/kaggle/input/datasets/*/*/config.json",
        "/kaggle/input/*/config.json",
    )
    for pat in pats:
        for cfg in sorted(glob.glob(pat)):
            d = os.path.dirname(cfg)
            if os.path.isfile(os.path.join(d, "quantization_config.json")):
                return d
    return None


def stage_download():
    print()
    print("--- モデル取得 ---")
    mounted = find_mounted_model()
    if mounted:
        total = sum(os.path.getsize(f) for f in glob.glob(
            os.path.join(mounted, "*.safetensors")))
        print(f" Dataset を使用: {mounted}（{total / 1024**3:.2f} GiB）")
        data.update(model_dir=mounted, model_source="dataset",
                    model_gib=round(total / 1024**3, 2))
        if total < 15 * 1024**3:
            raise RuntimeError("Dataset の safetensors が想定より小さい")
        data["stage"] = "download"
        save()
        return

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    dest = pick_model_dir()
    print(f" Dataset なし → HF から DL: {dest}")
    data.update(model_dir=dest, model_source="huggingface")

    t0 = time.time()
    snapshot_download(repo_id=MODEL_REPO, local_dir=dest)
    dt = round(time.time() - t0, 1)

    sizes = {}
    total = 0
    for f in glob.glob(os.path.join(dest, "*.safetensors")):
        s = os.path.getsize(f)
        sizes[os.path.basename(f)] = s
        total += s
    print(f"DL 完了: {total / 1024**3:.2f} GiB / {dt}s"
          f"（{total / 1024**2 / max(dt, 1):.1f} MiB/s）")
    data.update(download_seconds=dt, model_gib=round(total / 1024**3, 2),
                shards=sizes)
    if len(sizes) < 3:
        raise RuntimeError(f"safetensors が 3 個そろっていない: {list(sizes)}")
    data["stage"] = "download"
    save()


# ── 5+6. ロードと生成 ───────────────────────────────────────
def run_job(generator, ids, stops, max_new_tokens=96):
    """1 ジョブ流して (text, final_result, wall_seconds) を返す。"""
    from exllamav3 import Job

    job = Job(input_ids=ids, max_new_tokens=max_new_tokens, stop_conditions=stops)
    generator.enqueue(job)

    text = ""
    final = None
    t1 = time.time()
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("stage") == "error":
                raise res["error"]
            text += res.get("text", "")
            if res.get("eos"):
                final = res
    return text, final, round(time.time() - t1, 1)


def stage_load_generate():
    print()
    print("--- ロード → 生成 ---")
    import torch
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer

    t0 = time.time()
    config = Config.from_directory(data["model_dir"])
    config.max_seq_len = 8192
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
    model.load(progressbar=False)  # 空き VRAM を見て 2 枚にレイヤー分割
    load_dt = round(time.time() - t0, 1)
    print(f"ロード完了: {load_dt}s")

    per_gpu = {}
    for i in range(torch.cuda.device_count()):
        per_gpu[f"cuda:{i}"] = round(torch.cuda.memory_allocated(i) / 1024**3, 2)
    print(f"VRAM 使用: {per_gpu}")
    data.update(load_seconds=load_dt, vram_gib=per_gpu,
                active_devices=[str(d) for d in model.active_devices])

    messages = [{"role": "user", "content": "1+1は？日本語で短く答えて"}]
    try:
        ids = tokenizer.hf_chat_template(messages, enable_thinking=False)
    except Exception:
        ids = tokenizer.hf_chat_template(messages)

    stops = ["<|im_end|>"]
    for src in (getattr(config, "eos_token_id", None),
                getattr(tokenizer, "eos_token_id", None)):
        if isinstance(src, int):
            stops.append(src)
    try:
        sid = tokenizer.single_id("<|im_end|>")
        if isinstance(sid, int) and sid >= 0:
            stops.append(sid)
    except Exception:
        pass

    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    text, final, wall = run_job(generator, ids, stops)

    tokens = final.get("new_tokens") if final else None
    gen_dt = final.get("time_generate") if final else wall
    tps = (tokens / gen_dt) if (tokens and gen_dt) else 0

    print(f"生成: {tokens} tok / {gen_dt:.2f}s = {tps:.2f} tok/s"
          f"（wall {wall}s）")
    print("応答:", text.strip()[:400])
    data.update(prompt_tokens=(final or {}).get("prompt_tokens")
                or (int(ids.shape[-1]) if hasattr(ids, "shape") else None),
                generated_tokens=tokens,
                prefill_seconds=round((final or {}).get("time_prefill", 0), 2),
                generate_seconds=round(gen_dt, 2),
                wall_seconds=wall,
                tokens_per_second=round(tps, 2),
                response=text.strip()[:800])
    data["stage"] = "generate"
    save()

    if os.environ.get("MEASURE_MTP", "1") == "1":
        stage_mtp(model, config, cache, tokenizer, ids, stops, tps)


# ── 7. MTP 投機デコード（任意・失敗しても継続）───────────────
def stage_mtp(model, config, cache, tokenizer, ids, stops, base_tps):
    print()
    print("--- MTP 投機デコード ---")
    try:
        if "mtp" not in getattr(config, "model_classes", {}):
            print(" MTP コンポーネントが無い → skip")
            data["mtp"] = {"ok": False, "reason": "no mtp component"}
            data["stage"] = "done"
            save()
            return
        from exllamav3 import Cache, Generator, Model
        from exllamav3.cache import CacheLayer_fp16

        t0 = time.time()
        draft_model = Model.from_config(config, component="mtp")
        max_history = max(draft_model.caps.get("default_draft_size", 4), 4)
        draft_cache = Cache(draft_model, max_num_tokens=4096,
                            layer_type=CacheLayer_fp16,
                            max_history=max_history, max_batch_size=1)
        draft_model.load(progressbar=False)
        load_dt = round(time.time() - t0, 1)
        print(f"MTP ロード: {load_dt}s（max_history={max_history}）")

        generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                              draft_model=draft_model,
                              draft_cache=draft_cache)
        text, final, wall = run_job(generator, ids, stops)

        tokens = (final or {}).get("new_tokens")
        gen_dt = (final or {}).get("time_generate") or wall
        tps = (tokens / gen_dt) if (tokens and gen_dt) else 0
        stats = {k: v for k, v in (final or {}).items()
                 if isinstance(v, (int, float))
                 and ("draft" in k.lower() or "accept" in k.lower())}

        print(f"MTP 生成: {tokens} tok / {gen_dt:.2f}s = {tps:.2f} tok/s"
              f"（baseline {base_tps:.2f} → {tps / base_tps:.2f}x）")
        print("応答:", text.strip()[:400])
        data["mtp"] = {
            "ok": True,
            "load_seconds": load_dt,
            "max_history": max_history,
            "generated_tokens": tokens,
            "generate_seconds": round(gen_dt, 2),
            "wall_seconds": wall,
            "tokens_per_second": round(tps, 2),
            "speedup": round(tps / base_tps, 2) if base_tps else None,
            "draft_stats": stats,
            "result_keys": sorted((final or {}).keys()),
            "response": text.strip()[:800],
        }
    except Exception as e:
        traceback.print_exc()
        data["mtp"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    data["stage"] = "done"
    save()


def main():
    stage_env()
    stage_install()
    stage_import()
    stage_download()
    stage_load_generate()


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

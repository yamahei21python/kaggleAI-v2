#!/usr/bin/env python3
"""ExLlamaV2 の Kaggle CPU プリフライト。

ExLlamaV2 は CUDA 拡張（exllamav2_ext）必須で CPU 推論はできない。
GPU クォータを浪費しないため、GPU を要さない範囲だけを CPU で先に検証する:

  1. 環境（python / torch / CUDA toolkit / nvcc / GPU 有無）
  2. pip install exllamav2（ソースビルド可否・所要時間）
  3. import と __version__
  4. exllamav2_ext*.so の生成有無
  5. ExLlamaV2Config（純 Python・GPU 不要）
  6. CLI（python -m exllamav2.server --help）

CPU で「起動成功」と言えるのは 5 まで。モデルロードは GPU 必須。
"""
import json
import glob
import os
import shutil
import subprocess
import sys
import time
import traceback

RESULT = "/kaggle/working/result.json"
data = {"stage": "init"}


def save():
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
    lines = (s or "").strip().splitlines()
    return "\n".join(lines[-n:])


def stage_env():
    print("=" * 60)
    print("ExLlamaV2 CPU プリフライト")
    print("=" * 60)
    print(f"python : {sys.version.split()[0]}")
    print(f"nproc  : {os.cpu_count()}")

    try:
        import torch
        print(f"torch  : {torch.__version__}")
        print(f"torch cuda build : {torch.version.cuda}")
        print(f"cuda available   : {torch.cuda.is_available()}")
        data.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available())
    except Exception as e:
        print(f"torch  : import 失敗 {e}")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        cands = sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
        nvcc = cands[-1] if cands else "/usr/local/cuda/bin/nvcc"
    if os.path.isfile(nvcc):
        r, _ = sh([nvcc, "--version"])
        line = next((ln for ln in r.stdout.splitlines() if "release" in ln), "")
        print(f"nvcc   : {nvcc}  {line.strip()}")
        data.update(nvcc=nvcc, nvcc_release=line.strip())
    else:
        print("nvcc   : 見つからない（ソースビルド不可の可能性）")
        data.update(nvcc=None)

    print(f"nvidia-smi : {'present' if shutil.which('nvidia-smi') else 'ABSENT（CPUのみ）'}")
    df = (sh(["df", "-h", "/kaggle/working"])[0].stdout or "").strip().splitlines()
    if len(df) > 1:
        print("disk   : " + df[-1])

    data["stage"] = "env"
    save()


def stage_install():
    print()
    print("--- pip install exllamav2 ---")
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    r, dt = sh([sys.executable, "-m", "pip", "install", "-U",
                "--no-cache-dir", "exllamav2"], timeout=3000, env=env)
    ok = r.returncode == 0
    print(f"rc={r.returncode}  {dt}s")
    print(tail(r.stdout + "\n" + r.stderr, 40))
    data.update(pip_install_ok=ok, pip_install_seconds=dt,
                pip_rc=r.returncode)
    if not ok:
        data["pip_error_tail"] = tail(r.stderr, 60)
    data["stage"] = "install"
    save()
    return ok


def stage_import():
    print()
    print("--- import 検証 ---")
    code = (
        "import json, os, exllamav2\n"
        "d = os.path.dirname(exllamav2.__file__)\n"
        "ext = sorted(f for f in os.listdir(d) if f.startswith('exllamav2_ext'))\n"
        "print(json.dumps({'version': getattr(exllamav2, '__version__', '?'),\n"
        "                  'dir': d, 'ext': ext}))\n"
    )
    r, dt = sh([sys.executable, "-c", code], timeout=300)
    print(f"rc={r.returncode}")
    print(tail(r.stdout + "\n" + r.stderr, 30))
    info = {}
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    data.update(import_ok=r.returncode == 0,
                exllamav2_version=info.get("version"),
                ext_files=info.get("ext"),
                import_seconds=dt)
    if r.returncode != 0:
        data["import_error_tail"] = tail(r.stderr, 40)

    # GPU 不要の config クラス（純 Python）
    code2 = (
        "import json\n"
        "from exllamav2 import ExLlamaV2Config\n"
        "c = ExLlamaV2Config()\n"
        "c.model_dir = '/tmp/nonexistent_exl2'\n"
        "print(json.dumps({'config_ok': True,\n"
        "                  'max_seq_len_default': getattr(c, 'max_seq_len', None),\n"
        "                  'attrs': len(vars(c))}))\n"
    )
    r2, _ = sh([sys.executable, "-c", code2], timeout=120)
    print(f"config rc={r2.returncode}")
    print(tail(r2.stdout + "\n" + r2.stderr, 20))
    data.update(config_ok=r2.returncode == 0)
    if r2.returncode != 0:
        data["config_error_tail"] = tail(r2.stderr, 30)

    data["stage"] = "import"
    save()


def stage_cli():
    print()
    print("--- CLI: python -m exllamav2.server --help ---")
    r, dt = sh([sys.executable, "-m", "exllamav2.server", "--help"],
               timeout=180)
    print(f"rc={r.returncode}")
    out = (r.stdout or "") + (r.stderr or "")
    print(tail(out, 35))
    data.update(server_help_ok=r.returncode == 0,
                server_help_seconds=dt)
    if r.returncode != 0:
        data["server_help_error_tail"] = tail(out, 40)
    data["stage"] = "cli"
    save()


def main():
    stage_env()
    stage_install()
    stage_import()
    stage_cli()

    data["cpu_feasible_upto"] = (
        "install/import/config/CLI まで" if data.get("config_ok")
        else "install まで" if data.get("pip_install_ok")
        else "何もできない（ビルド不可）"
    )
    data["gpu_required_for"] = "モデルロード以降すべて（exllamav2_ext が CUDA を呼ぶ）"
    data["stage"] = "done"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        data["stage"] = "failed"
        data["error"] = f"{type(e).__name__}: {e}"
    save()
    print()
    print("RESULT: " + json.dumps(data, ensure_ascii=False))
    sys.exit(0)

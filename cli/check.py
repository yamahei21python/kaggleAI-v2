"""CPU で回せる検証をまとめて実行する（GPU 不要）。

    python3 -m cli check

順番: 生成 → ruff → pytest → ノート一致 → script 一致 → 生成物の compileall。
どこかで落ちたら止める。
"""
import shutil
import subprocess
import sys

from kbuild import spec

ROOT = spec.ROOT


def _module_available(module):
    return subprocess.run([sys.executable, "-c", f"import {module}"],
                          capture_output=True).returncode == 0


def _tool(module, exe, *args):
    """`-m <module>` を優先し、無ければ PATH の実行ファイルを使う。

    ruff / pytest を単体で入れている環境（venv に入れていない）でも同じ手順で回す。
    """
    if _module_available(module):
        return [sys.executable, "-m", module, *args]
    found = shutil.which(exe)
    if not found:
        raise SystemExit(f"{exe} が見つからない（pip install {exe} か PATH を確認）")
    return [found, *args]


def steps():
    return [
        ("build", [sys.executable, "-m", "cli", "build"]),
        ("ruff", _tool("ruff", "ruff", "check", ".")),
        ("pytest", _tool("pytest", "pytest", "tests", "-q")),
        ("notebooks", [sys.executable, "kaggle/check_notebooks.py"]),
        # script 側の突き合わせ。notebook と同じく「commit 済み = 再生成結果」を
        # バイト比較する（生成し忘れた main.py が push されるのを防ぐ）。
        ("scripts", [sys.executable, "kaggle/check_scripts.py"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q",
                        "kaggle/exl3-serve-kernel", "kaggle/lmdeploy-serve",
                        "kaggle/sglang-serve", "kaggle/ci-cpu",
                        "kaggle/step2-script", "kaggle/step2-vllm-script",
                        "kaggle/llama-server-script",
                        # 配信しないカーネルの script 版（セル連結）
                        "kaggle/step1/script", "kaggle/step2/script",
                        "kaggle/step2-vllm/script", "kaggle/step2-vllm-dryrun/script",
                        "kaggle/smoke/script", "kaggle/bench/script",
                        "kaggle/bench_mtp/script"]),
    ]


def run():
    for name, cmd in steps():
        p = subprocess.run(cmd, cwd=ROOT)
        print(f"{'OK ' if p.returncode == 0 else 'NG '} {name}")
        if p.returncode != 0:
            return 1
    return 0

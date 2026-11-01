"""SGLang 用の起動引数。純関数だけ。

公式は sm75 以上（T4 を含む）を対応に挙げているが、同じ文書に sm75+ の
FlashInfer 問題と回避策（`--attention-backend triton --sampling-backend pytorch`）
が載っている。T4 では CUDA graph も落ちる報告があるため既定で無効にする。
"""
import os

NAME = "sglang"
VERSION = "0.5.22"
READY_PATH = "/v1/models"
ALIAS = "qwen3.8-27b-awq"

# `sglang==0.5.22` は `outlines==0.1.11` を固定し、それが
# `outlines_core==0.1.26` を**完全固定**する。0.1.26 には cp313 の wheel が無く
# （Rust 実装）、Kaggle の Python 3.13.15 では sdist ビルドになって落ちる。
# 実測 2026-11-01（`sglang-install-probe`・GPU 0 秒）:
#
# | 経路 | 実測 | 結果 |
# |---|---|---|
# | 素の `pip install sglang==0.5.22` | 297s | ❌ `Failed building wheel for outlines_core` |
# | **Rust を入れてから素の pip**（筋 A） | rustup **9.4s** + pip **389.8s** | ✅ `import sglang` / `--help` とも通る |
# | 固定回避（`outlines_core==0.1.17`） | 未検証 | — |
#
# 直接依存 26 本を cp313 wheel の有無で調べたところ、wheel が無いのは
# `outlines_core==0.1.26` と `cuda-tile==1.6.0rc5` の 2 本だけ。cuda-tile は
# sdist からビルドが通る（失敗一覧に出ていない）。つまり**Rust があれば通る**。
# 既定は実測で通った筋 A。筋 B は「Rust が入らない環境」用に残す。
RUSTUP_CMD = ("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
              "| sh -s -- -y --profile minimal --default-toolchain stable")
CARGO_ENV = '. "$HOME/.cargo/env"'

# 筋 B で差し替える版（0.1.x で cp313 wheel があるのは 0.1.17 だけ）
OUTLINES_CORE_FALLBACK = "0.1.17"

# 筋 B の「残り」。`--no-deps` で sglang を入れたあとに手で並べる。
# 上から順に入れる必要はないが、`outlines` 系より**後**に置くこと。
_PIN_REST = (
    "torch==2.14.1", "torchaudio==2.11.0", "torchvision",
    "transformers==5.19.0", "tokenizers==0.23.2", "xgrammar==0.2.7",
    "flashinfer_python[cu13]==0.7.1", "sglang-kernel==0.4.9",
    "compressed-tensors==0.18.0", "cuda-python>=13.0",
    "apache-tvm-ffi==0.1.11", "tilelang==0.1.12", "numba==0.65.1",
    "nvidia-cutlass-dsl[cu13]==4.8.0", "nvidia-mathdx==25.6.0",
    "humming-kernels[cu13]==0.1.12", "quack-kernels==0.6.5",
    "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2",
    "tokenspeed_mla==0.1.8", "smg-grpc-servicer>=0.9.0",
    "openai-harmony==0.0.4", "openai==2.6.1", "sentencepiece==0.2.1",
    "timm==1.0.16", "soundfile==0.13.1", "blobfile==3.0.0",
    "fastapi", "uvicorn", "uvloop", "watchfiles", "msgspec", "orjson",
    "pydantic", "psutil", "pyzmq>=25.1.2", "pybase64", "einops",
    "easydict", "interegular", "partial_json_parser", "regex", "scipy",
    "setproctitle", "tiktoken", "tqdm", "xxhash", "zstandard",
    "prometheus-client>=0.20.0", "python-multipart", "aiohttp", "requests",
    "distro", "packaging", "pillow", "numpy", "gguf", "py-spy", "ninja",
    "build", "IPython", "datasets", "modelscope", "mistral_common>=1.11.5",
    "llguidance>=1.7.6,<2.0.0", "kernels>=0.14.1,<0.15",
    "anthropic>=0.20.0", "nvshmem4py-cu13", "torchcodec==0.17.0",
    "torch_memory_saver>=0.0.9.post1", "cuda-tile==1.6.0rc5",
)


# install が通っても**起動で落ちる**（実測 2026-11-01 / v3）。
#
#   RuntimeError: Detected that PyTorch and TorchAudio were compiled with
#   different CUDA versions. PyTorch has CUDA version 13.0 whereas
#   TorchAudio has CUDA version 12.8.
#
# vLLM と同じ落とし穴（`notes/pitfall-coverage.md` #2）。`sglang` は
# `torch==2.14.1` を入れるが、イメージ残りの `torchaudio 2.8.3+cu128` が
# そのまま残ると transformers → torchaudio の遅延 import で落ちる。
# PyPI の `torchaudio==2.11.0` は **+cu130 ビルド**（wheel の `version.py` を
# 実際に開いて確認: `__version__ = '2.11.0+cu130'`）なので置き換えれば直る。
TORCHAUDIO_SPEC = "torchaudio==2.11.0"

# **置き換えたあとに、もう一度 import して確かめる。** ここで黙って通すと、
# 15 分の起動待ちのあとに同じ traceback を読むことになる。
_TORCHAUDIO_FIX = (
    "import subprocess, sys\n"
    "\n"
    "SPEC = " + repr(TORCHAUDIO_SPEC) + "\n"
    "\n"
    "\n"
    "def _check():\n"
    "    r = subprocess.run(\n"
    "        [sys.executable, '-c',\n"
    "         'import torch, torchaudio;'\n"
    "         'print(\"torch\", torch.__version__, torch.version.cuda);'\n"
    "         'print(\"torchaudio\", torchaudio.__version__);'\n"
    "         'print(\"path\", torchaudio.__file__)'],\n"
    "        capture_output=True, text=True, timeout=600)\n"
    "    return r.returncode == 0, ((r.stdout or '') + (r.stderr or ''))[-900:]\n"
    "\n"
    "\n"
    "ok, out = _check()\n"
    "print(out.strip())\n"
    "if not ok:\n"
    "    print('torchaudio 不整合 -> %s を --no-deps --force-reinstall' % SPEC)\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                    '--no-deps', '--force-reinstall', SPEC],\n"
    "                   capture_output=True, text=True, timeout=1800)\n"
    "    ok, out = _check()\n"
    "    print(out.strip())\n"
    "print('torchaudio import:', 'OK' if ok else 'NG')\n"
    "sys.exit(0 if ok else 1)\n"
)


def kc_sglang_pip_steps(python, version=VERSION, route=None):
    """install 手順を**ステップのリスト**で返す（1 回では通らない）。

    `route` は `"rust"`（既定・実測で通った）か `"pin"`（固定回避・未検証）。
    `KC_SGLANG_ROUTE` で実行時に切り替えられる（生成コードに式として埋まる）。
    """
    route = (route or os.environ.get("KC_SGLANG_ROUTE", "rust")).strip().lower()
    if route == "pin":
        # 順序が本質: `outlines` を先に入れると 0.1.26 を取りに行って同じ所で死ぬ。
        return [
            [python, "-m", "pip", "install", "--no-deps", f"sglang=={version}"],
            [python, "-m", "pip", "install",
             f"outlines-core=={OUTLINES_CORE_FALLBACK}"],
            [python, "-m", "pip", "install", "--no-deps", "outlines==0.1.11"],
            [python, "-m", "pip", "install", *_PIN_REST],
            [python, "-c", _TORCHAUDIO_FIX],
        ]
    # 筋 A。`pip` 側も `bash -lc` で包むのは、rustup が `~/.cargo/bin` に
    # 入れるだけで PATH を書き換えないため（`sh()` は毎回新しい env で起動する）。
    return [
        ["bash", "-lc", f"{RUSTUP_CMD} && {CARGO_ENV} && cargo --version"],
        ["bash", "-lc",
         f'{CARGO_ENV} && exec "{python}" -m pip install "sglang=={version}"'],
        # 3 番目は pip ではなく**確認と修理**。ここを飛ばすと起動で落ちる。
        [python, "-c", _TORCHAUDIO_FIX],
    ]


def kc_sglang_argv(
    python,
    model_dir,
    *,
    tp=2,
    port=8000,
    attention_backend="triton",
    sampling_backend="pytorch",
    disable_cuda_graph=True,
    mem_fraction_static=0.75,
    served_name=ALIAS,
):
    argv = [
        python, "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--served-model-name", served_name,
        "--tp", str(tp),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--attention-backend", attention_backend,
        "--mem-fraction-static", str(mem_fraction_static),
        "--trust-remote-code",
    ]
    if sampling_backend:
        argv += ["--sampling-backend", sampling_backend]
    if disable_cuda_graph:
        argv.append("--disable-cuda-graph")
    return argv


def kc_sglang_ready_url(port=8000):
    return f"http://127.0.0.1:{port}{READY_PATH}"

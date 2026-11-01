#!/usr/bin/env python3
"""LMDeploy / SGLang の配信カーネル（script 1 ファイル）を生成する。

    python3 kaggle/build_engine_serve.py            # 通常（配信）
    python3 kaggle/build_engine_serve.py --smoke    # 疎通だけ（GPU 時間の節約）

生成物:
  kaggle/lmdeploy-serve/main.py … LMDeploy pytorch backend + AWQ（T4 x2）
  kaggle/sglang-serve/main.py   … SGLang（triton backend / CUDA graph 無効）

狙いは vLLM との比較。既存の AWQ Dataset をそのまま使い、量子化資産を増やさない。
MTP は使わない（`spec_decode: false` を result.json に明示）。
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent                                      # リポジトリ直下
OUT = HERE
sys.path.insert(0, str(ROOT))
from cli import config  # noqa: E402
from kbuild import embed, metadata, spec  # noqa: E402
from kbuild import serve_script  # noqa: E402
from runtime import proxy_source, source_bundle  # noqa: E402
from runtime.constants import (  # noqa: E402
    EXL3_HARD_MAX_MINUTES,
    EXL3_RUN_MINUTES,
    PROXY_PORT,
    VLLM_PORT,
)

CFG = config.load(ROOT)
OWNER = "yamahei21python"
AWQ = f"{OWNER}/qwen38-27b-uncensored-hauhaucs-awq-mtp"
CLOUDFLARED = f"{OWNER}/my-cloudflared-bin"

PROXY_B64 = embed.b64_text(proxy_source(ROOT / "kaggle" / "step2" / "auth_proxy.py"))
HTML_B64 = embed.b64_file(ROOT / "llama-chat.html")

ENGINES = {
    "lmdeploy": {
        "name": "lmdeploy",
        "version": "0.18.0",
        "alias": "qwen3.8-27b-awq",
        "pattern": "-f lmdeploy",
        "pip_argv": "kc_lmdeploy_pip_argv(sys.executable)",
        # turbomind は T4（sm75）でこの AWQ を載せられなかった（実測 2026-10-31）:
        # 重み 43/64（67%）で `[TM][FATAL] allocator.cc:49 CUDA error: out of memory`。
        # 19.57 GB を 2 枚に割るはずが 1 枚で ~14 GB に達しており、TP=2 が効かず
        # fp16 に展開されている挙動に見える。pytorch バックエンドは TP と AWQ
        # （4bit のまま）を素直に扱うので、そちらで載せ直す。
        "launch_argv": (
            "kc_lmdeploy_argv(sys.executable, model_dir, tp=TP, port=PORT, "
            "session_len=8192, cache_max_entry_count=0.5, "
            'model_name=ALIAS, backend="pytorch")'),
        "ready_url": "kc_lmdeploy_ready_url(PORT)",
        "knobs": ('{"backend": "pytorch", "session_len": 8192, '
                  '"cache_max_entry_count": 0.5}'),
    },
    "sglang": {
        "name": "sglang",
        "version": "0.5.22",
        "alias": "qwen3.8-27b-awq",
        "pattern": "-f sglang.launch_server",
        # SGLang は 1 回の pip では通らない（outlines_core に cp313 wheel が無い）。
        # 既定は「Rust を入れてから素の pip」= 実測で通った筋（9.4s + 389.8s）。
        "pip_steps": "kc_sglang_pip_steps(sys.executable)",
        "launch_argv": (
            "kc_sglang_argv(sys.executable, model_dir, tp=TP, port=PORT, "
            'attention_backend="triton", sampling_backend="pytorch", '
            "disable_cuda_graph=True, "
            "mem_fraction_static=0.75, served_name=ALIAS)"),
        "ready_url": "kc_sglang_ready_url(PORT)",
        # KNOBS は生成コードの Python dict リテラルとして埋め込まれる。
        # JSON の true/false ではなく Python の True/False で書く
        # （result.json へは json.dump が true/false に直す）。
        "knobs": ('{"attention_backend": "triton", '
                  '"sampling_backend": "pytorch", '
                  '"disable_cuda_graph": True, '
                  '"mem_fraction_static": 0.75, "trust_remote_code": True}'),
    },
}

# metadata は台帳から引く（手書きすると台帳とずれても気づけない）。
# キーは ENGINE 名、引くのは台帳の slug。
METADATA = {
    name: metadata.for_kernel(
        spec.find(f"step2-{name}-qwen38-27b-cloudflared"))
    for name in ("lmdeploy", "sglang")
}

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    for name, engine in ENGINES.items():
        serve_script.write(
            OUT / f"{name}-serve",
            engine,
            METADATA[name],
            run_minutes=EXL3_RUN_MINUTES,
            runtime=source_bundle("net", "proc", "keepalive", "proxy", name),
            proxy_b64=PROXY_B64,
            html_b64=HTML_B64,
            gist_id=CFG.get("GIST_ID", ""),
            gist_owner=CFG.get("GIST_OWNER", ""),
            port=VLLM_PORT,
            proxy_port=PROXY_PORT,
            hard_max_minutes=EXL3_HARD_MAX_MINUTES,
            smoke=smoke,
        )

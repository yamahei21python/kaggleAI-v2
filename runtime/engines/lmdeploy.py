"""LMDeploy（Turbomind / AWQ）用の起動引数。純関数だけ。

カーネルに埋め込まれて実行時に呼ばれるため、外部 import を持たない。
T4（sm75）×2 を前提に、TP=2・session-len は控えめから始める。
"""

NAME = "lmdeploy"
VERSION = "0.18.0"
READY_PATH = "/v1/models"
ALIAS = "qwen3.8-27b-awq"


def kc_lmdeploy_pip_argv(python, version=VERSION):
    return [python, "-m", "pip", "install", "-q", f"lmdeploy=={version}"]


def kc_lmdeploy_argv(
    python,
    model_dir,
    *,
    tp=2,
    port=8000,
    session_len=8192,
    cache_max_entry_count=0.8,
    model_name=ALIAS,
    backend="pytorch",
):
    """api_server の起動 argv。

    MTP ヘッド（model-mtp.safetensors）は使わない。
    """
    return [
        python, "-m", "lmdeploy", "serve", "api_server", model_dir,
        "--backend", backend,
        "--model-name", model_name,
        "--tp", str(tp),
        "--server-name", "0.0.0.0",
        "--server-port", str(port),
        "--session-len", str(session_len),
        "--cache-max-entry-count", str(cache_max_entry_count),
    ]


def kc_lmdeploy_ready_url(port=8000):
    return f"http://127.0.0.1:{port}{READY_PATH}"

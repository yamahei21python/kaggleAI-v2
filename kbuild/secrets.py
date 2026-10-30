"""カーネル（ビルダー）が要求する Kaggle Secrets の台帳。

コード中で Secret を参照したら、ここにも追加する。CI の
tests/test_secrets.py が「台帳」と「get_secret("...") / secret("...") の
リテラル」を突き合わせる。Secrets の登録自体は Kaggle の UI で行う。
"""

REQUIRED = {
    "kaggle/build_step1.py": frozenset({
        "KAGGLE_USERNAME", "KAGGLE_KEY",
    }),
    "kaggle/build_step2.py": frozenset({
        "GIST_ID", "GITHUB_TOKEN", "NGROK_TOKEN", "NGROK_DOMAIN",
    }),
    "kaggle/build_step2_direct.py": frozenset({
        "NGROK_TOKEN", "NGROK_DOMAIN",
    }),
    "kaggle/build_step2_vllm.py": frozenset({
        "GIST_ID", "GITHUB_TOKEN", "NGROK_TOKEN", "NGROK_DOMAIN",
    }),
    "kaggle/build_vllm_dataset.py": frozenset({
        "HF_TOKEN",
    }),
    "kaggle/exl3-dataset-upload/main.py": frozenset({
        "KAGGLE_USERNAME", "KAGGLE_KEY",
    }),
    "kaggle/exl3-serve/build_kernel.py": frozenset({
        "EXL3_API_USER", "EXL3_API_PASS", "GIST_ID", "GITHUB_TOKEN",
    }),
}

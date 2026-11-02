"""カーネル（ビルダー）が要求する Kaggle Secrets の台帳。

コード中で Secret を参照したら、ここにも追加する。CI の
tests/test_secrets.py が「台帳」と「get_secret("...") / secret("...") の
リテラル」を突き合わせる。Secrets の登録自体は Kaggle の UI で行う。

notebook 配信カーネル（Ollama / llama-server / vLLM）の Secret 参照は
**共有セル（kaggle/serve-cells/）に集約**されている。ビルダー側は
`__KAGGLE_OWNER__` のような既定値テンプレートしか持たないので台帳に載らない。

公開 URL の配布はビーコンカーネル（`runtime/beacon.py`）が担うので、
外部サービス用の Secret は持たない。
"""

REQUIRED = {
    "kaggle/build_step1.py": frozenset({
        "KAGGLE_USERNAME", "KAGGLE_KEY",
    }),
    "kaggle/build_vllm_dataset.py": frozenset({
        "HF_TOKEN",
    }),
    "kaggle/exl3-dataset-upload/main.py": frozenset({
        "KAGGLE_USERNAME", "KAGGLE_KEY",
    }),
    "kaggle/build_exl3_serve.py": frozenset({
        "EXL3_API_USER", "EXL3_API_PASS",
    }),
    # ── script カーネル（main.py を base64 で埋め込む系統）──
    # トンネルは cloudflared だけ（quick tunnel なので Secret 不要）。
    # ビーコンは**カーネルに注入された Kaggle 認証**を使うので Secret は不要。
    "kbuild/serve_script.py": frozenset({
        "API_USER", "API_PASS",
    }),
}

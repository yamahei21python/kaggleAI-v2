# archive

調査・切り分けに使った使い捨てのノートとビルダー（cpu/tpu/whl probe、vLLM の各種検証、diag、push/cycle スクリプト）。
本番経路（`kaggle/step1`, `step2`, `step2-ngrok`, `step2-vllm-kernel`, `bench*`, `smoke`）からは参照していない。
結果は `notes/` と README にまとめてある。

- 再実行は想定していない。ビルダーはこのディレクトリ基準で出力する（`build_cpu_probe3.py` は `../step2/auth_proxy.py` を参照）。
- `_v8_cycle.sh` / `_v10_cycle.sh` は作業時の絶対パス（`/Users/...`）を含む。
- ruff / CI の対象外。

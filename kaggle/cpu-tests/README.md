# CPU テスト専用（GPU クォータ 0 秒）

GPU を使わずに「高度化の前提」だけを潰す置き場。1 ディレクトリ = 1 カーネル
（台帳は `kbuild/spec.py`、`gpu=False`）。

| ディレクトリ | 調べるもの | 実行 |
|---|---|---|
| `tiny-llama/` | llama-server の高度化フラグ（KV q8_0 / `-fa` / `--spec-type draft-mtp` / `-np 2` / `-t 4`）。起動可否・エラー文・CPU 生成 t/s をケース別に | `python3 -m cli push tiny-llama-probe --yes` |
| `tiny-ollama/` | 本番 `OLLAMA_ENV` の ctx / KV 量子化（4096・32k・128k、f16・q8_0）。`/api/ps` の実値で env の無言無視を検出 + CPU t/s | `python3 -m cli push tiny-ollama-probe --yes` |
| `vllm-flags/` | vLLM 0.32.0 のフラグ実在・選択肢・argv 受理（`--speculative-config` / `--kv-cache-dtype` / `--enforce-eager` / `--attention-backend`）。CUDA 前段で判定 | `python3 -m cli push vllm-flag-probe --yes` |
| `whl-probe/` | （`kaggle/whl-probe`）vLLM wheelhouse の `--no-index` 解決可否 | `python3 -m cli push vllm-whl-probe --yes` |

どれも tiny モデル（0.4〜0.5B）を実行時に落とす。GPU カーネルは 1 本も回さない。

`gist/`（`gist-probe`）は 2026-11-02 に廃止した。公開 URL の配布をビーコン
カーネルへ移し、Gist への進捗書き込みもしなくなったため（`notes/url-publishing.md`）。
ビーコンの往復を見るのは `beacon/`（`beacon-probe`）。

## CPU で確定できること / できないこと

確定できる:
- フラグ・環境変数の**実在**（未知フラグ即死、typo の無言無視を検出）
- 生成 argv の形（本番と同じ組み立てを tiny で回す）
- `/v1` の応答形（stream / alias / chat template）
- 起動失敗時の**ログ文**（切り分け表を実測で更新できる）
- KV 量子化・flash attention を CPU に落としたときの**劣化の仕方**（落ちるか警告か）

確定できない（GPU 必須）:
- VRAM に載るか、`-ngl all` の実際のオフロード
- t/s、MTP の accept 率
- `GGML_BACKEND_PATH` / `libcuda.so.1` の解決（CUDA が無いので必ず失敗する）

## 使い方

```bash
python3 -m cli push tiny-llama-probe --yes
kaggle kernels output yamahei21python/tiny-llama-probe -p /tmp/tl/   # result.json
```

GPU クォータは表示されるが消費は 0（`enable_gpu: false`）。

## 設計メモ

- **単一ファイル push** なので、3 本とも Dataset 探索 / zip 展開 / `result.json`
  保存をそれぞれ持つ。`_shared.py` を import する形は Kaggle で壊れるため、
  重複は意図的（共有化するなら `kbuild` が生成時に埋め込む）。
- 失敗時は **exit 1**（`result.json` の `stage="failed"` と二重化）。GPU と違い
  無料なので、状態を COMPLETE にせず気付けるようにする。
- 出力には `gpu_unknown` を必ず入れる（CPU で確定できない項目の明示）。

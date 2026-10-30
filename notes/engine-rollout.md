# 追加エンジン導入メモ（LMDeploy / SGLang）

2026-10-30 時点。既存 4 エンジン（Ollama / llama-server / vLLM / ExLlamaV3）に続く
比較候補の導入方法。**TabbyAPI は同日に見送り**（理由は「見送り」節）。**カーネル id は既存のものを流用せず、新規に切る**
（Secrets・dataset_sources・実行履歴が id に紐づくため）。

対象は `notes/refactor-plan.md` の「共通化の範囲（2 層）」で残す層。
proxy / tunnel / keepalive / verify の共通層はそのまま使い、
`runtime/engines/<name>.py` だけを足す。

## 前提

| 項目 | 値 |
|---|---|
| GPU | T4 x2（sm_75、16 GB x2、PCIe。NVLink なし） |
| Python | 3.13（Kaggle GPU 画像） |
| torch / CUDA | torch 2.13.0+cu130 / CUDA 13.0（`notes/vllm-cpu-findings.md`） |
| 制約 | Docker 不可、pip のみ、GPU セッションは 12h / クォータ制 |
| 再利用するモデル資産 | AWQ `qwen38-27b-uncensored-hauhaucs-awq-mtp`（#1）/ EXL3 `qwen38-27b-uncensored-exl3-4bpw`（#6） |
| 揃える API | OpenAI `/v1`（`/v1/models`、`/v1/chat/completions`、SSE） |

**MTP（投機デコード）は vLLM 固有。** 2 エンジンとも MTP ヘッドは使わない。
速度比較は「MTP あり vLLM vs MTP なし 他」になるため、結果表に必ず明記する。

## 比較

| | LMDeploy | SGLang |
|---|---|---|
| 再利用する量子化 | AWQ / GPTQ（既存 #1） | AWQ（既存 #1） |
| T4 対応 | 公式（Turing sm75 明記） | sm75 は対応範囲。ただし要検証 |
| TP=2 | `--tp 2` | `--tp 2` |
| 認証 | 版により差。共通 proxy で Basic を付ける | `--api-key` あり。共通 proxy 推奨 |
| 導入 | PyPI wheel（cp313 あり） | PyPI wheel（cp313 あり） |
| PyPI 版（確認日 2026-10-30） | 0.18.0 | 0.5.22 |
| 主なリスク | torch の版ずれ | FlashInfer / CUDA graph が sm75 で落ちる可能性 |

## 共通の導入手順

1. **wheel の存在確認（CPU 可）**
   ```bash
   # Kaggle と同じ Python 3.13 / x86_64 / manylinux の wheel があるか
   pip download --only-binary=:all: --no-deps \
     --python-version 3.13 --implementation cp --abi cp313 \
     --platform manylinux_2_28_x86_64 \
     lmdeploy==0.18.0 -d /tmp/wh-lmdeploy

   pip download --only-binary=:all: --no-deps \
     --python-version 3.13 --implementation cp --abi cp313 \
     --platform manylinux_2_34_x86_64 \
     sglang==0.5.22 -d /tmp/wh-sglang
   ```
   SGLang は `manylinux_2_34`（Ubuntu 22.04 の glibc 2.35 で OK）。

2. **依存の衝突確認** — Kaggle 画像の torch は 2.13.0+cu130。LMDeploy / SGLang は
   それぞれ torch を要求するため、`pip install --dry-run` でダウングレード有無を確認。
   毎回イメージから作り直す前提なら許容。固定したい場合は vLLM と同じく
   wheelhouse Dataset を作る（`vllm-wheelhouse` と同方式、3〜4 GB 級）。

3. **起動は必ず `wait_http(/healthz または /v1/models)` の後**に proxy を立てる。
   既存の `runtime/net.py` / `runtime/proxy.py` の順序を踏襲。

4. **検証は既存と同じ 5 点** — `/v1/models`、日本語の非ストリーム、SSE、
   401（proxy 経由）、`/stop` と keepalive。公開 URL は既存の Gist 配布。

5. **GPU push は 1 エンジン 1 回に集約**。CPU でできる確認（wheel、`--help`、
   引数組み立ての純関数テスト）を先に通し、GPU は最終確認だけにする。

## LMDeploy（第一候補）

AWQ をそのまま使えるため、追加の量子化資産が不要で導入コストが最小。

```bash
pip install -U lmdeploy==0.18.0
lmdeploy serve api_server <AWQ_MOUNT> \
  --backend turbomind \
  --model-name qwen3.8-27b-awq \
  --tp 2 \
  --server-name 0.0.0.0 --server-port 8000 \
  --session-len 32768 \
  --cache-max-entry-count 0.8
```

- `<AWQ_MOUNT>` は `/kaggle/input/qwen38-27b-uncensored-hauhaucs-awq-mtp/` 配下の
  モデルディレクトリ（`runtime/storage.py` の探索を使う）。
- `model-mtp.safetensors` は無視される。MTP は使わない。
- 16 GB x2 で 27B AWQ は TP=2 前提。まず `--session-len 8192` で疎通し、
  VRAM を見て 16k / 32k へ上げる。
- 認証は LMDeploy 側に依存せず、共通 proxy の Basic 認証で統一する。
- 固有ノブ: `backend` / `tp` / `session_len` / `cache_max_entry_count`。

**検証ポイント**: AWQ カーネルが sm75 で起動するか、TP=2 の初期化時間、
`/v1/models` のモデル名が `--model-name` と一致するか。

## SGLang（第二候補）

AWQ を再利用でき、RadixAttention（プレフィックスキャッシュ）の効果を測れる。
ただし T4 では FlashInfer / CUDA graph を無効化して当たる。

```bash
pip install -U "sglang==0.5.22"
python -m sglang.launch_server \
  --model-path <AWQ_MOUNT> \
  --served-model-name qwen3.8-27b-awq \
  --tp 2 \
  --host 0.0.0.0 --port 8000 \
  --attention-backend triton \
  --disable-cuda-graph \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

- SGLang 0.5.22 は Python 3.11+ / CUDA 13 前提で、Kaggle 画像と一致する。
- **FlashInfer は sm80+ 前提の機能がある**ため `--attention-backend triton` を明示。
- CUDA graph capture が T4 で失敗する報告があるため `--disable-cuda-graph` から始める。
  起動できたら graph 有効化を試し、VRAM と起動時間を比較。
- 認証は `--api-key` が使えるが、`/quota`・`/stop` は共通 proxy が持つので
  proxy を外さない。
- 固有ノブ: `attention_backend` / `disable_cuda_graph` / `mem_fraction_static` /
  `tp` / `api_key`。

**検証ポイント**: `sgl_kernel` の import、Triton JIT の初回コンパイル時間、
TP=2 の NCCL 初期化、16 GB x2 での `mem_fraction_static` 調整。

## 見送り: TabbyAPI（2026-10-30 決定）

導入しない。理由:

- 中身は ExLlamaV3 なので、速度は既存 `exl3-serve` とほぼ同じ。新しい比較軸にならない。
- EXL3 4bpw（約 15.9 GiB）は T4 16 GB 1 枚で余裕がなく、TP は exllamav3 側が
  実験的・TabbyAPI 側の対応も限定的で、T4×2 の旨みが薄い。
- 認証は内蔵しているが、`/quota`・`/stop`・keepalive・UI は既存の共通層が必要で、
  置き換えても配管は減らない。
- EXL3 の OpenAI 化は既存 `chat_server.py` で足りている。必要になった時点で再検討する。

## リポジトリへの統合

`runtime/engines/<name>.py` の共通インターフェース（案）:

```python
NAME = "lmdeploy"
CAPS = {"quant": {"awq", "gptq"}, "tp": True, "spec_decode": False}

def build_launch(spec) -> list[str]: ...
def wait_ready(base_url: str, timeout: int) -> None: ...
def verify_extra(base_url: str, auth: str | None) -> dict: ...
```

- spec に `engine="lmdeploy" | "sglang"` と `knobs` を追加。
- 生成カーネルは新 id（例: `step2-lmdeploy-qwen38-27b-cloudflared`）。
  既存 id の `dataset_sources` は変えない。
- `result.json` に `engine` / `version` / `quant` / `tp` / `spec_decode=false` を残す。
- `kbuild` 側は変更不要（梱包はエンジン非依存）。

## 実施順と合格条件

1. **LMDeploy**
   - CPU: wheel 解決、`lmdeploy serve api_server -h` の引数確認、起動引数の純関数テスト
   - GPU: TP=2 起動、`/v1` 日本語 SSE、tok/s、VRAM ピーク
2. **SGLang**
   - CPU: wheel 解決、`--help` のフラグ確認（`--attention-backend` / `--disable-cuda-graph`）
   - GPU: triton backend で起動、RadixAttention のヒット率、tok/s
各段の GPU 実測は 1 回にまとめ、失敗時はログ末尾を `result.json` に残す。
比較表には「MTP なし」を明記し、vLLM の数値と直接比較しない。

## エンジンランキング（2026-10-30・推定）

実測ではなく、公開ベンチ・コミュニティ評判・T4 の制約からの推定。
評価軸: T4 適合 25% / 性能 20% / 導入・運用 20% / 資産再利用 20% / 保守 15%。
「区分」は S=今すぐ試す、A=条件付き有力、B=用途次第、C=代替・学習用、D=今は不要。

| 順位 | エンジン | 区分 | スコア | 一言 | 判定 |
|---|---|---|---|---|---|
| 1 | vLLM | S | 9.0 | AWQ + MTP の実績。並列に強い | 現行の基準 |
| 2 | LMDeploy | S | 8.5 | T4 の AWQ は公式対応。単発〜中並列で最有力 | 次に導入 |
| 3 | ExLlamaV3 | A | 8.0 | 単発最速。15.9 GiB が 16 GB にギリギリ | 既存を維持 |
| 4 | llama.cpp（llama-server） | A | 7.5 | 何でも動く安定枠。並列は弱い | 既存を維持 |
| 5 | SGLang | B | 6.5 | 長文・prefix 再利用で化ける。T4 は博打 | 3 番目に検証 |
| 6 | Aphrodite | B | 6.0 | vLLM 互換で量子化の幅が広い。vLLM との差は小さい | 保留 |
| 7 | TGI (HF) | C | 5.0 | T4 実績はあるが Docker 前提気味 | 保留 |
| 8 | Ollama | C | 4.5 | 導入は最速。速度・context 制御は弱い | 開発用 |
| 9 | KoboldCpp / llama-cpp-python | C | 4.0 | llama-server と同系統。新しい軸にならない | 不要 |
| 10 | MLC-LLM | C | 3.5 | 汎用だがモデルコンパイルが必要 | 不要 |
| 11 | LocalAI / XInference | D | 3.0 | エンジンではなくオーケストレータ | 不要 |
| 12 | ONNX Runtime GenAI | D | 2.5 | INT4 変換が必要。変換コストに見合わない | 不要 |
| 13 | CTranslate2 | D | 2.0 | 軽量だが OpenAI サーバを自作する必要 | 不要 |
| 14 | TabbyAPI | D | 2.0 | EXL3 のラッパー。既存と重複（見送り済み） | 不要 |
| 15 | vLLM TPU / JAX MaxText / JetStream | D | 1.5 | TPU 別トラック。bf16 変換が必要 | 別件 |

軸別（上位のみ）:

- **T4 での素の速度（単発）**: LMDeploy ≳ ExLlamaV3 > vLLM(MTP) > llama.cpp > SGLang > Ollama
- **並列スループット**: vLLM ≒ SGLang > LMDeploy > Aphrodite > ExLlamaV3 > llama.cpp > Ollama
- **導入の楽さ**: Ollama > KoboldCpp > llama.cpp > LMDeploy > vLLM > ExLlamaV3 > Aphrodite > SGLang > TGI > MLC-LLM > ONNX / CTranslate2 > LocalAI / XInference > TPU
- **長文・エージェント**: SGLang > vLLM > LMDeploy > 以下同列

前提が変わると順位も変わる: マルチユーザー重視なら SGLang/vLLM、
単発レイテンシ重視なら LMDeploy/ExLlamaV3、TPU が使えるなら TPU 枠が浮上する。

## 参考

- LMDeploy W4A16（Turing sm75 / T4）: https://lmdeploy.readthedocs.io/en/v0.4.1/quantization/w4a16.html
- LMDeploy api_server: https://github.com/InternLM/lmdeploy/blob/main/docs/en/llm/api_server.md
- SGLang install: https://docs.sglang.ai/get_started/install.html
- SGLang server args: https://sgl-project.github.io/advanced_features/server_arguments.html

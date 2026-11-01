# 追加エンジン導入メモ（LMDeploy / SGLang）

2026-10-30 時点。既存 4 エンジン（Ollama / llama-server / vLLM / ExLlamaV3）に続く
比較候補の導入方法。**TabbyAPI は同日に見送り**（理由は「見送り」節）。**カーネル id は既存のものを流用せず、新規に切る**
（Secrets・dataset_sources・実行履歴が id に紐づくため）。

対象は `notes/refactor-plan.md` の「共通化の範囲（2 層）」で残す層。
proxy / tunnel / keepalive / verify の共通層はそのまま使い、
`runtime/engines/<name>.py` だけを足す。

## 実装状況（2026-10-30・CPU 検証まで）

2 本とも**カーネル生成と CPU テストまで完了**。GPU 実測は未実施（クォータ待ち）。

| 項目 | LMDeploy | SGLang |
|---|---|---|
| ビルダー | `kaggle/build_engine_serve.py`（共通。エンジン式を差し替え） | 同左 |
| テンプレート | `kbuild/serve_script.py` | 同左 |
| 起動引数 | `runtime/engines/lmdeploy.py` | `runtime/engines/sglang.py` |
| 生成物 | `kaggle/lmdeploy-serve/`（gitignore） | `kaggle/sglang-serve/`（gitignore） |
| カーネル id | `step2-lmdeploy-qwen38-27b-cloudflared` | `step2-sglang-qwen38-27b-cloudflared` |
| 状態 | semi（比較用） | semi（比較用） |
| CPU テスト | `tests/test_engines.py`（argv / pip 固定 / 生成決定性 / AST / Secret 非混入） | 同左 |
| smoke | `--smoke` で install → launch → 会話 → 停止 | 同左 |

```bash
python3 kaggle/build_engine_serve.py            # 2 本を生成（配信）
python3 kaggle/build_engine_serve.py --smoke    # 疎通だけの変種（GPU 時間の節約）
python3 -m pytest tests/test_engines.py -q
python3 -m cli list | grep -E "lmdeploy|sglang"
python3 -m cli push step2-lmdeploy-qwen38-27b-cloudflared --smoke --yes   # GPU 実測（1 回）
```

生成される `main.py` の流れは exl3 と同じ:
`stage_env → find_model → pip install → launch → proxy → verify → tunnel →
Gist 公開 → keepalive/予算ループ → 停止`。差分は推論サーバの起動 argv だけ。
`result.json` に `engine` / `version` / `tp` / `knobs` / `spec_decode=false` /
`smoke` / `stop_remaining` を残す。

smoke は auth_proxy / cloudflared / Gist / keepalive を**省略**し、エンジンへ直接
`/v1/chat/completions` を投げて終了する。CPU 側で検証済みの経路を GPU で
なぞらないための仕組み（詳細: `notes/gpu-smoke-plan.md`）。

実行時バグの修正（2026-10-30 レビュー反映）:
- `secret()` が env のみ → `kaggle_secrets.UserSecretsClient` フォールバック追加
- 例外時にプロセス残留 → `finally` で `stop_all()`（TERM → wait → KILL → `kc_stop_processes`）
- `terminate()` だけ → 段階停止（残存を `result.json` の `stop_remaining` に記録）
- `/stop` がエンジンを落とさない → `kc_start_proxy(..., extra_args=("--stop-pid", pid))`

## 骨格の統一（2026-10-30 追記・第 4 次計画）

「全部 1 本の骨組みにしたい」方針で、**配信カーネルは notebook / script とも
1 本の骨格に集約**した。エンジンを足すときに増やすのは「差のデータ」だけになる。

| | notebook 版 | script 版 |
|---|---|---|
| 骨格 | `kbuild/serve_notebook.py` の `Flow` | `kbuild/serve_script.py` |
| エンジン差 | `kaggle/<kernel>/cells/*.py` | 同じセルを `textwrap.indent` で関数化 |
| 設定 | `kaggle/step2/cells/00_settings.py`（両者が同じファイルを読む） | 同左 |
| 疎通確認 | `engine_checks(judge, retry, base, auth=None)` | 同左（同じ体裁） |

骨格に足した差の表現:

| 差 | 表現 |
|---|---|
| 配信するか | `SERVE=False` で単発モード（install → launch → 終了） |
| ウォームアップ / 後片付け | `WARMUP` / `CLEANUP` フラグ + `warmup()` / `cleanup()`（既定は no-op） |
| エンジン固有の定数 | `$SETTINGS` スロット（`PORT` / `ALIAS` / `CONTEXT_LENGTH` を上書き） |
| 判定項目 | `engine_checks()` のみ。`verify()` の足場は共通 |

これで 6 エンジン（Ollama / llama-server / vLLM / ExLlamaV3 / LMDeploy / SGLang）が
**同じ `result.json` の形**を返す。

**トンネルは cloudflared だけ**（2026-11-01。それまでは ngrok と両対応だった）。
`kbuild/serve_script.py` の `start_tunnel()` が唯一の分岐点で、今は
`start_cloudflared()` を呼ぶだけ。`data["tunnel"] = TUNNEL` により
`result.json` の `tunnel` は**実際に張った方**になる（設定セルの `TUNNEL_KIND` より
骨格の実装が優先）。ngrok を落とした理由は `notes/url-publishing.md`:
`NGROK_TOKEN` を Secrets から読めない（専用アカウント）うえ、固定ドメインの
必要性自体がビーコンで消えた。

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
| T4 対応 | 公式（Turing sm75 明記） | 公式に sm75 以上（T4 を例示）。ただし FlashInfer 問題あり |
| T4 の必須オプション | `--backend turbomind`（失敗時 `pytorch`） | `--attention-backend triton --sampling-backend pytorch` |
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
公式は sm75 以上（T4 を例に挙げている）を対応とするが、同じ文書に sm75+ の
FlashInfer 問題と回避策（`--attention-backend triton --sampling-backend pytorch`）が
載っている。CUDA graph も sm75 で落ちる報告があるため無効から始める。

```bash
# ⚠ この 1 行では入らない（実測 2026-11-01）。下の「install の実測」を見る。
pip install -U "sglang==0.5.22"
python -m sglang.launch_server \
  --model-path <AWQ_MOUNT> \
  --served-model-name qwen3.8-27b-awq \
  --tp 2 \
  --host 0.0.0.0 --port 8000 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

### install の実測（2026-11-01 / `sglang-install-probe`・GPU 0 秒）

素の `pip install sglang==0.5.22` は **297 秒で `pip install 失敗 (1)`**。
`install_tail` は `Failed building wheel for outlines_core` だけで、
**エンジンは一度も起動していなかった**（「T4 で動かない」は未検証だった）。

| 対象 | cp313 wheel | 備考 |
|---|---|---|
| `outlines-core==0.1.26` | **無い** | `sglang → outlines==0.1.11` が完全固定。Rust 実装（maturin） |
| `outlines-core==0.1.17` | ある | 0.1.x で cp313 があるのはここだけ |
| `cuda-tile==1.6.0rc5` | 無い（sdist） | ただし**ビルドは通る**（失敗一覧に出ない） |

直接依存 26 本を調べた結果、**cp313 wheel が無いのはこの 2 本だけ**。
環境に `cargo` / `rustc` が無いことが原因なので、**Rust を入れれば通る**:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable   # 9.4s
. "$HOME/.cargo/env"
pip install "sglang==0.5.22"                                    # 389.8s
```

結果: `import sglang` 26.6s / `launch_server --help` 12.5s とも成功。
torch はイメージの 2.13.0+cu130 → **2.14.1+cu130** に差し替わる。
→ カーネル側は `kc_sglang_pip_steps()`（2 ステップ）で実装済み。

### 判定: **T4 では不可**（2026-11-01 / v4 実測）

install を通したあと、モデルの量子化を解釈する段で止まる:

```
RuntimeError: ('Quantization scheme is not supported for ', 'the current GPU.
Min capability: 80. ', 'Current capability: 75.')
```

上流のソースで裏取り（GPU 0 秒）:

- `CompressedTensorsWNA16.get_min_capability()` は **80**
  （`schemes/compressed_tensors_wNa16.py:94`）。T4 は **75**。
- 同ファイルのコメント: symmetric / asymmetric とも **Marlin カーネル経路**。
  Marlin は sm80+ 専用。
- Triton 実装は `CompressedTensorsWNA16TritonMoE`（**MoE 専用**）しかなく、
  Linear 層に sm75 の逃げ道が無い。

**LMDeploy との違い**: LMDeploy は `group_size=128` を**読めなかった**（形式）。
SGLang は**読めるがカーネルが sm80 必須**（能力）。根は違うが、結論は同じ「不可」。

→ 比較表は vLLM と llama-server の 2 本で確定（`notes/gpu-run-sheet.md` §9）。

- SGLang 0.5.22 は Python 3.11+ / CUDA 13 前提で、Kaggle 画像と一致する。
- **FlashInfer は sm80+ 前提の機能がある**ため `--attention-backend triton` を明示。
  公式の回避策に従い `--sampling-backend pytorch` も付ける（既定値に設定済み）。
- CUDA graph capture が T4 で失敗する報告があるため `--disable-cuda-graph` から始める。
  起動できたら graph 有効化を試し、VRAM と起動時間を比較。
- 認証は `--api-key` が使えるが、`/quota`・`/stop` は共通 proxy が持つので
  proxy を外さない。
- 固有ノブ: `attention_backend` / `sampling_backend` / `disable_cuda_graph` /
  `mem_fraction_static` / `tp` / `api_key`。

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

### 推定のうち、実測で決着したもの（2026-11-01）

| 推定 | 実測 | 結論 |
|---|---|---|
| LMDeploy が「次に導入」で最有力 | `group_size=128` を読めず `ValueError`（125s） | **不可が確定**（推定は外れ） |
| SGLang は「T4 は博打」 | install は Rust で通った（513s）。しかし **WNA16 が sm80 必須**で起動時に停止 | **博打ではなく非対応が確定** |
| ExLlamaV3 は「既存を維持」 | **エンジンは起動した**（v2）。`loaded in 229s; serving qwen38-exl3 on 127.0.0.1:9931 (ctx 8192)`。落ちたのはこちらの `engine_checks()` の引数不一致。ただし **v3 は 1800 秒待っても起動しなかった** | **T4 で動くのは確実。安定性は未確定** |
| vLLM が「現行の基準」 | 31.7 t/s / 929s（v10 は 31.0 t/s / 978s） | **維持**（ただし起動に 373 秒の JIT を含む） |

→ **「ERROR」を「不可」と読まない。** 判定に到達したのは LMDeploy だけで、
SGLang と ExLlamaV3 はどちらも**エンジンが起動する前**に落ちている。
逆も同じで、**1 回の成功を「動く」と読まない**（ExLlamaV3 は v2 が成功、
v3 が起動失敗。同じ構成で 230 秒 → 1800 秒超）。

### ExLlamaV3 の実測（2026-11-01 / `exl3-serve` v2）

梱包ミス（`.py` を notebook として push）を直して再投入したら、**エンジンは普通に
起動した**。落ちたのは 3 段目ではなく、その後の疎通確認。

```
install_method : wheel
wheel_name     : exllamav3-1.6.5+cu128.torch2.11.0-cp313-cp313-linux_x86_64.whl
pip_seconds    : 12.8
load_seconds   : 230.4        ← 15.9 GiB を 1 枚の T4 に
chat_server.log: loaded in 229s; serving qwen38-exl3 on 127.0.0.1:9931 (ctx 8192)
elapsed_seconds: 263.6
error          : TypeError: engine_checks() takes from 3 to 4 positional
                 arguments but 5 were given
```

読み取れること:

1. **ExLlamaV3 は T4（sm75）で動く。** wheel が `+cu128` で 12.8 秒で入る。TP 非対応
   なので 1 枚だけ使い、2 枚目は空く。
2. **ロード 230 秒**は vLLM の 181 秒（page cache が温まっていない素の読み）と同程度。
   ただしこちらは事前の先読みをしていない。**vLLM の 373 秒 JIT に相当する山が無い**
   のが ExLlamaV3 の利点で、起動 263.6 秒は vLLM の 929 秒に対して約 3.5 分の 1。
3. 落ちた原因は**エンジンではなくこちらの骨格**。`verify()` は
   `engine_checks(judge, retry, base, auth, direct)` と 5 引数で呼ぶのに、骨格の
   **既定** `engine_checks` だけ `auth=None` で止まっていた。`checks_code` を持つ
   エンジン（ollama / vllm / llama-server）は自前の 5 引数版で上書きするので露見せず、
   **exl3 だけが既定を使う**ため初めて踏んだ。修正済み＋回帰テスト追加
   （`tests/test_smoke_checks.py`）。
4. 速度（t/s）は**まだ測れていない**。`stage: "load"` で止まっており、生成まで
   到達していない。単発レイテンシという別軸の比較は次回。

### ExLlamaV3 v3（2026-11-01）: 起動すらしなかった

`engine_checks` を直して再投入したら、**今度は起動しなかった**。

```
pip_seconds    : 11.6         （wheel は同じ。導入は問題ない）
chat_server.log: 0 バイト      ← 1800 秒間、1 文字も出ていない
elapsed_seconds: 1834.7
error          : RuntimeError: 待機タイムアウト
                 http://127.0.0.1:9931/health（Connection refused）
```

v2 と v3 で同じモデル・同じ wheel・同じ GPU（T4 x2）なのに、**230 秒で終わる
工程が 1800 秒に収まらなかった**。8 倍以上のばらつき。

**手がかりがゼロだったのが問題。** `chat_server.log` が空なのは異常ではなく、
そういう作りになっている:

- 起動時に `--no-progress` を渡しているので exllamav3 の読み込み進捗が出ない
- `chat_server.py` の `print` は「loaded in Ns」と「warmup failed」の 2 つだけ

つまり読み込みが 99% まで進んでいたのか、最初の 1 バイトで固まっていたのか
**区別できない**。次の手は「もう 1 回回す」しかなく、1 往復 30 分を捨てる。
vLLM の 360 秒無音区間の正体を掴むのに 4 往復使ったのと同じ構造なので、
計器を先に入れた（`runtime/proc.py` の `kc_sample` / `kc_watch`）。

次回の v4 で見るべきもの:

| 指標 | 読み方 |
|---|---|
| `read_mib` の増分 | 15.9 GiB に対して何 MiB/s か。v2 の 229 秒は約 70 MiB/s |
| `read_mib` が動かない | 固まっている（読み込みではなく別の工程で停止） |
| `char_mib` だけ増える | ページキャッシュから読んでいる（ディスクは触っていない） |
| `rss_mib` が増えない | mmap で読んでいる（anonymous に載っていない） |
| `vram_mib` が増えない | 重みがまだ GPU に載っていない |

**この 2 回で「ExLlamaV3 は T4 で動く」は確定、「安定して起動する」は未確定。**
v2 の成功を「再現性のある事実」として扱ってはいけない。

## 参考

- LMDeploy W4A16（Turing sm75 / T4）: https://lmdeploy.readthedocs.io/en/v0.4.1/quantization/w4a16.html
- LMDeploy api_server: https://github.com/InternLM/lmdeploy/blob/main/docs/en/llm/api_server.md
- SGLang install: https://docs.sglang.ai/get_started/install.html
- SGLang server args: https://sgl-project.github.io/advanced_features/server_arguments.html

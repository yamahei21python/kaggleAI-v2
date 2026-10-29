# ExLlamaV3 on Kaggle — 決定事項ログ（2026-10-30）

Kaggle での推論エンジン選定〜ExLlamaV3版の設計方針を記録する。

## 現在の構成（変更なし）

| 項目 | 内容 |
|---|---|
| モデル | `orcarouter/Qwen3.8-27B-Uncensored:latest`（Q4_K_M, 16.52 GiB, vision対応） |
| GPU構成 | T4 x2（16GB x2）、GPUクォータは限られている |
| 直起動 | `llama-server`（Ollama同梱バイナリ）を patch Dataset から復元して起動 |
| 公開 | auth_proxy.py（UI配信/認証/keepalive）→ ngrok / cloudflared |
| カーネル | `yamahei21python/step2-direct-llama-server-ngrok`（現在ERROR: 起動待ち900s切れ+fail()未定義バグ） |

## 決定

1. **エンジン: ExLlamaV3（EXL3）** を採用する。
   - ExLlamaV2 は選ばない（理由は下記「不可・却下」）。
2. **モデル: `Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw`（17.1 GB）** を第一候補とする。
   - 現行モデルと同じ Qwen3.8-27B Uncensored 系を維持できる唯一の量子化形式。
   - フォールバック: `darkbit1001/…Heretic-Uncensored…EXL3-4.00bpw-HB8`（16.9GB）、
     `Honkware/…Uncensored…exl3-4.0bpw`（16.3GB）。
3. **テキスト専用で割り切る（visionは不可）**。
   - exllamav3 は multimodal / tensor parallel が未対応。Qwen3.8のvision能力は使えない。
4. **CPU検証は「GPU不要部分まで」で打ち切る**。
   - ExLlama系は CPU では import すら不可と確定したため、以降の検証はGPUセッションで行う。
   - ただしCPUカーネル（GPUクォータ非消費）で可能なプリフライトは今後も先に回す。

## 不可・却下（調査結果）

| 選択肢 | 判定 | 根拠 |
|---|---|---|
| Qwen3.8 の EXL2 | **存在しない** | HF検索で 0 件。Qwen3.8世代はEXL3のみ |
| ExLlamaV2 で ExLlamaV3 (EXL3) を読む | 不可 | 形式非互換。EXL3は exllamav3 専用 |
| ExLlamaV2 を CPU 実行 | **不可** | 下記プリフライト結果 |
| Qwen3-32B-Uncensored EXL2 4bpw（17.8GB） | 却下 | V2で動くが一世代前。Qwen3.8維持を優先 |
| Gemma3-27B abliterated の EXL2 | 不可 | 該当リポジトリは measurement のみで重みが無い |
| Qwen3-VL の EXL2 | 不可 | 配布は EXL3 のみ（=V3必要）＋V3もVL未対応 |
| uncensored × vision × EXL2 の3条件 | 不可 | 3条件を満たす配布物が存在しない |
| Kaggle内で EXL2/EXL3 を新規量子化 | 不可 | 元重み 55.6GB。disk 20GB/VRAM 32GB に載らない |

## プリフライト結果（CPU, 2026-10-30）

カーネル: `yamahei21python/exllamav2-cpu-preflight`（ローカル: `kaggle/exllama-cpu-preflight/`）

| 項目 | 結果 |
|---|---|
| CPUイメージ torch | **2.11.0+cpu**（CUDAビルドでない） / `torch.version.cuda = None` |
| nvcc / CUDA toolkit | 無し |
| `pip install exllamav2` | **成功（5.9s）** |
| `import exllamav2` | **失敗** — `exllamav2_ext` JITコンパイルで `CUDA_HOME` 未設定 |
| `ExLlamaV2Config` / server CLI | 同上の理由で不可 |
| 結論 | CPUで可能なのは install まで。**ロード以降も含めGPUイメージ必須** |

## ExLlamaV3 の制約・リスク

- **インストール**: PyPI の `exllamav3`（1.6.5, pure-python wheel）は初回 import 時に
  JITコンパイル（数分）。GPUイメージの nvcc が必要。
  事前ビルドwheel（`cu128.torch2.8.0-cp313` 等）は GitHub Releases にあるが、
  Kaggle GPUイメージの torch/CUDA 版に一致するか未確認。
- **T4 (sm75)**: メンテナが issue #162 で「Turing is now supported（速度は控えめ）」と回答。
  ただし GEMM は Turing 向けに作り直された経緯があり、**動作は要実測**。
- **マルチGPU**: `--tensor_parallel`（native / nccl）が実装済み（master 確認）。
  ただし T4 x2 は PCIe 接続なので **レイヤー分割を本命**とし、TP は比較候補。
- **サーバ**: 公式推奨は **TabbyAPI**（OpenAI互換）。`examples/chat.py` は対話用。
  自前の auth_proxy 構成に載せるなら TabbyAPI か薄いラッパーのどちらかにする（未決）。
- **Kaggle GPUイメージ**: python 3.13（既存直起動カーネルのログより）。

## コード確認で確定した事実（2026-10-30 追記）

- モデルの実体: `architectures = ["Qwen3_5ForConditionalGeneration"]`、
  `model_type = qwen3_5`。**ハイブリッド線形注意（full attention は 4 層に 1 回）**。
- exllamav3 本体に `architecture/qwen3_5.py` と `qwen3_5_mtp.py` があり、
  **アーキテクチャは公式対応**。Turing 用カーネル `exllamav3_ext/turing/fa75.cu` も存在。
- quant 設定: `quant_method=exl3`, `version=1.4.6`, `bits=4.0`, `head_bits=6`,
  **`mtp_bits=8`（MTP ヘッド入り）** → 将来的に `--mtp` 投機デコード可（初回は使わない）。
- プリビルドwheel（v1.6.5）: `cu128 × torch2.8.0/2.10.0/2.11.0 × cp310–cp314` の
  linux_x86_64 が GitHub Releases にあり。GPUイメージの torch/CUDA と一致すれば
  **JIT コンパイル（nvcc・数分〜十数分）を回避できる**。cu が 12.8 以外なら wheel は使わず JIT。
- モデルリポジトリは public・gated 無し。3 シャード計 **15.9 GiB**。
- 生成 API（最小構成）:
  `Config.from_directory` → `Model.from_config` → `Cache(model, max_num_tokens=...)`
  → `model.load(progressbar=True)`（引数なし = 空きVRAMで自動レイヤー分割）
  → `Tokenizer.from_config` → `Generator(model, cache, tokenizer)`
  → `Job(input_ids=tokenizer.hf_chat_template(...), max_new_tokens=...)` → `enqueue`/`iterate`。

## GPU 検証カーネル（作成済み）

ローカル: `kaggle/exl3-gpu-check/`（`yamahei21python/exl3-qwen38-gpu-check`）

1 回の GPU セッションで「環境 → 導入 → import → モデル取得 → ロード → 生成 1 回 →
MTP 投機デコード実測」を通し、`/kaggle/working/result.json` に各段の所要秒・tok/s・
エラーを残す。サーバは立てない。どの段で失敗しても exit 0 で結果を回収できる。
モデルは **Dataset がマウントされていればそれを使い**、無ければ HF から DL
（`/kaggle/temp` → `/tmp` → `/kaggle/working` の順に空きの大きい所。working は 20GiB 上限）。

MTP ステージ（`MEASURE_MTP=0` で無効化可）の配線は model_init.py と同一:
`Model.from_config(config, component="mtp")` → `Cache(draft_model, max_num_tokens=4096,
layer_type=CacheLayer_fp16, max_history=caps["default_draft_size"], max_batch_size=1)` →
`draft_model.load()` → `Generator(..., draft_model=, draft_cache=)`。
MTP コンポーネントが無い/エラー時は `mtp.ok=false` を記録するだけで本体は成功扱い。

## モデル Dataset 化（GPU クォータ温存）

ローカル: `kaggle/exl3-dataset-upload/`（`yamahei21python/exl3-dataset-upload`）

CPU カーネルで HF → Kaggle Dataset へ**直接アップロード**する（ローカル PC を経由しない）。
Kaggle ノートには `KAGGLE_USERNAME` / `KAGGLE_KEY` が環境変数で入っているため、
`kaggle datasets create` をカーネル内から実行できる想定。

- Dataset: `yamahei21python/qwen38-27b-uncensored-exl3-4bpw`（private, 15.9GiB）
- 意図: GPU セッション中の 15.9GiB DL をなくす（時間短縮＋失敗リスク減）
- ワークアラウンド: DL は重複を避けるため作業ディレクトリ選択 → アップロード後に
  ローカルコピーを削除（`/kaggle/working` の出力を太らせない）
- v1 は Kaggle Secrets 未設定で失敗 → Secrets は**カーネル単位**に紐づくため
  UI で `KAGGLE_USERNAME` / `KAGGLE_KEY` を追加して再実行（v3）で成功。
- **結果（2026-10-30 完了）**: status `ready`、3 シャードを含む 15 ファイルを確認。
  DL 68.1s / アップロード 226.7s / 計 306.2s（CPU カーネル・GPU 消費ゼロ）。
  - v3 の result.json は `ok:false`（"アップロード後に safetensors が見えない"）だが
    これは CLI の一覧反映ラグによる**誤検知**。実体は `kaggle datasets files` で確認済み。
- GPU 検証カーネルの `dataset_sources` にこの Dataset を登録済み。
  `find_mounted_model()` がマウントを検出すると HF DL をスキップする。

### GPU クォータ（2026-10-30 時点）

- GPU: 29h49m / 30h 使用済み、**残り 10 分**。リセット 2026-10-31T00:00Z（**09:00 JST**）
- TPU: 20h 残（ExLlamaV3 は CUDA 専用のため使えない）

## 高速化のための事前準備（GPU 不要）

1. **Dataset 化**（上記）— **完了**。GPU セッションからモデル DL を削除
2. **薄い OpenAI 互換ラッパー**を先に作り、CPU でモック自己テスト — **完了**。
   `kaggle/exl3-serve/chat_server.py`（`/v1/chat/completions` non-stream + SSE、
   `/v1/models`、`/health`）と `_selftest_server.py`（MockEngine で全経路 green）。
   GPU では「ロード → 配信」だけにする。既存の auth_proxy.py / ngrok / ask.py / keepalive を再利用
3. **配信カーネルに時間予算（自動停止）** — 30/60 分で自分で exit（クォータ事故防止）※未実装
4. **MTP の実測を検証 1 回に含める** — **完了**（検証カーネルにステージ追加）。GGUF では 1.18x
5. **環境プローブを最初に** — **完了**。wheel が torch/CUDA に一致すれば JIT（5〜20 分）を回避
6. **cache サイズを絞る**（4096–8192）— **完了**（検証は 4096）

## llama-server の CPU 起動検証結果（2026-10-30 完了）

カーネル: `yamahei21python/llama-server-cpu-check`（ローカル: `kaggle/cpu-check/`）

| 項目 | 結果 |
|---|---|
| 起動 | **成功**（GPUなし・Xeon 4 vCPU / RAM 32 GiB） |
| モデル | 16.8 GB の model blob を 16.8 GiB に展開してロード |
| ロード時間 | **1375 秒（約 23 分）** |
| 生成 | `1\n2\n3\n4\n5` を正常返却 |
| 速度 | prompt eval 1.41 t/s / 生成 1.01 t/s（ctx 2048） |
| 備考 | `blk.64.nextn.*`（MTP層）は未使用警告。CPU では `--spec-type` は無意味 |
| 実用性 | **不可**（起動はするが 1 tok/s 前後）。CPU は「起動確認」まで |

## 未決事項 / 次アクション

1. **Dataset アップロード** → **完了**。`qwen38-27b-uncensored-exl3-4bpw` は `ready`
   （3 シャード）。`exl3-gpu-check` の `dataset_sources` に登録済み。
2. **GPU 検証の実行**はリセット後（2026-10-31 09:00 JST）に 1 回。
   「導入 → ロード → 生成」に加えて **MTP 投機デコードの tok/s 倍率**も同時に回収する。
   wheel 一致（=JIT 回避）もこの 1 回で確定させる。
3. **サーバ方式** → **薄い OpenAI 互換ラッパーを採用**（`kaggle/exl3-serve/chat_server.py`）。
   TabbyAPI は不採用: exllamav3 を直接叩けば足り、依存と設定面が増えるだけ。
   既存の `auth_proxy.py`（Basic 認証・keepalive リセット・/quota・/stop）と
   そのまま直列に繋げられる。
   残作業は配信カーネル（トンネル + 時間予算での自動停止）で、**T4 の速度確認後**に作る。
4. **llama.cpp のCPU起動検証** → **完了（起動可・約 1 tok/s で実用不可）**。
   上記「llama-server の CPU 起動検証結果」参照。

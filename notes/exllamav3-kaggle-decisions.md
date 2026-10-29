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
- **マルチGPU**: tensor parallel 未対応 → **レイヤー分割のみ**。T4 x2 に 17.1GB を分割。
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

## GPU 検証カーネル（作成済み・未push）

ローカル: `kaggle/exl3-gpu-check/`（`yamahei21python/exl3-qwen38-gpu-check`）

1 回の GPU セッションで「環境 → 導入 → import → DL → ロード → 生成 1 回」を通し、
`/kaggle/working/result.json` に各段の所要秒・tok/s・エラーを残す。サーバは立てない。
どの段で失敗しても exit 0 で結果を回収できる。モデルは `/kaggle/temp` → `/tmp` →
`/kaggle/working` の順に空きの大きい所へ DL（`/kaggle/working` は 20GiB 上限）。

## 未決事項 / 次アクション

1. **GPU実行の設計**（1回のGPUセッションで「install → モデルDL → ロード → 生成」を通す）
   - モデルは HF からランタイムDL（Kaggle Dataset 未作成）。17.1GB を `/kaggle/working`（20GB）に置く想定。
   - サーバは TabbyAPI か最小ラッパーかを決める。
2. **llama.cpp のCPU起動検証**（`yamahei21python/llama-server-cpu-check`）が実行中。
   - CPUはロードに15分以上かかる見込み。完了後に result.json を回収し、CPU起動可否を確定。
3. GPUクォータの残量確認 → 実行は1回で決める（失敗時は llama.cpp/GGUF にフォールバック）。

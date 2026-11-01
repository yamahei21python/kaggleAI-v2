# notes 索引

調査メモ・計画・ベンチ結果の置き場。現役の設計判断はここ、過去の生データは `archive/`。

## 設計・在庫

| ファイル | 内容 |
|---|---|
| `cleanup-plan.md` | **現役の整理案（2026-11-01 実測）**。散らかりの実測値・P0〜P3 の優先順位・決めることリスト。**P0 は実施済み**（下記「実施ログ」） |
| `refactor-plan.md` | リポジトリ再編の方針・段取り・CPU 受け入れ条件。**第 1〜4 次**（第 4 次 = notebook → script 移行の実施記録と残タスク）。「なぜそう決めたか」の履歴として読む |
| `kernel-inventory.md` | カーネル / Dataset / Secrets の台帳（`python3 -m cli list` と同源）。台帳は **47 本 / うち GPU 24 本** |
| `dataset-inventory.md` | Dataset の中身と整理案 |
| `engine-rollout.md` | 推論エンジン（Ollama / llama-server / vLLM / ExLlamaV3 / LMDeploy / SGLang）の展開状況 |
| `gpu-smoke-plan.md` | GPU 復活後の起動・会話テスト計画（smoke モードで時間を節約） |
| `gpu-run-sheet.md` | **当日そのまま打てる実行シート**。カーネル作成済みの状態・クォータ・Secrets・判定・切り分け。**v10〜v14 の実測ログ**を含む |
| `operations.md` | 手順の詳細（README から切り出し。パスは 2026-10-30 時点のスナップショット） |
| `runtime-notes.md` | 起動高速化・技術詳細・既知の問題・ローカル検証（同上） |
| `url-publishing.md` | **公開 URL の伝え方**。Gist 方式（このリポジトリの実装）と、専用アカウントで採った**ビーコンカーネル**方式の比較・実測・未検証点。起動の進捗はどちらもカバーしていない |
| `serve-parity.md` | **配信の 2 実装（notebook セル / script 骨格）の差分台帳**。機能ごとの対応と、いま効いている差（script 版は公開 URL の失効に気づかない） |

## 計測・落とし穴（2026-11-01 に新設）

| ファイル | 内容 |
|---|---|
| `measurement.md` | 何をどう測るか（`timings` の意味・節目の取り方） |
| `cpu-test-results.md` | **GPU 0 秒**で回した計器の結果（`vllm-install-bench` / `vllm-io-bench` / `sglang-install-probe`）と、そこから訂正した誤読 |
| `pitfall-coverage.md` | **症状 → 防御 → テスト**の対応表（31 項目）。実バグ 2 件の記録つき |

CPU プローブは `kaggle/cpu-tests/`（`vllm-flags` / `tiny-llama` / `tiny-ollama` /
`install-bench` / `io-bench` / `sglang-probe`）。**GPU クォータを 1 秒も使わない**ので、
「GPU で賭ける前に切り分ける」用途に使う。

## エンジン別メモ

| ファイル | 内容 |
|---|---|
| `exllamav3-kaggle-decisions.md` | EXL3 版の決定事項ログ |
| `vllm-cpu-findings.md` | vLLM を Kaggle で動かすまでの実測・失敗ログ |
| `vllm-startup-speedup.md` | vLLM 起動時間の短縮策。**§24 が最新**: 本番 v16 では prefetch が **vLLM 側の RAM 判定で辞退され**、重みロードが lazy と同じ 47 秒/層に戻り、予算 4,200 秒で切られた（§23 の playground 実測 1,186 秒は再現せず）。§23 は prefetch の機構と `bake_env` の入れ方。§18 の「先読みは退行」は `KC_PREWARM`（自前実装）の話で、vLLM 内蔵 prefetch は別物 |
| `tpu-llm-plan.md` | TPU で LLM を動かす検討 |

## ローカル uzu / Qwen3.5-9B

| ファイル | 内容 |
|---|---|
| `uzu-9b-local-setup.md` | ローカル環境のセットアップ |
| `uzu-opencode-server.md` | opencode 用サーバ設定 |
| `qwen35-9b-engine-comparison.md` | エンジン比較の結果 |
| `run_uzu_9b.sh` | 起動ラッパー（`uzu-backup/` から encoding 等を復元） |
| `Modelfile.qwen35-local` | ローカル Ollama 用 Modelfile |
| `uzu-backup/` | uzu の設定・CLI・パッチの退避（**現役**。ランチャーが参照） |

## ベンチ

| ファイル | 内容 |
|---|---|
| `prompt-length-benchmark-plan.md` | プロンプト長 vs 生成速度の夜間ベンチ計画 |
| `prompt_sweep.py` | ベンチ実行スクリプト（既定出力は `archive/`） |
| `archive/` | 過去の `llama-bench-*` / `prompt-sweep-*` の JSON・ログ、`uzu-9b-download.log` |

`archive/` は再実行しない生データ。新しい計測結果もここへ追加する。

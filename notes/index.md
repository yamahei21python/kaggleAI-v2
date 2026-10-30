# notes 索引

調査メモ・計画・ベンチ結果の置き場。現役の設計判断はここ、過去の生データは `archive/`。

## 設計・在庫

| ファイル | 内容 |
|---|---|
| `refactor-plan.md` | リポジトリ再編の方針・段取り・CPU 受け入れ条件 |
| `kernel-inventory.md` | カーネル / Dataset / Secrets の台帳 |
| `dataset-inventory.md` | Dataset の中身と整理案 |
| `engine-rollout.md` | 推論エンジン（Ollama / llama-server / vLLM / ExLlamaV3）の展開状況 |

## エンジン別メモ

| ファイル | 内容 |
|---|---|
| `exllamav3-kaggle-decisions.md` | EXL3 版の決定事項ログ |
| `vllm-cpu-findings.md` | vLLM を Kaggle で動かすまでの実測・失敗ログ |
| `vllm-startup-speedup.md` | vLLM 起動時間の短縮策 |
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

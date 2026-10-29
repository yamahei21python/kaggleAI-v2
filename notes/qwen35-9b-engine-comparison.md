# 3エンジン比較結果（同モデル Qwen3.5 9B、同プロンプト）

| エンジン | 量子化 | 生成速度 | t/f | メモリ |
|---|---|---|---|---|
| uzu (投機ON) | 独自 (BF16) | 61.96 t/s | 11.12 | 5.72 GB |
| uzu (投機OFF) | 独自 (BF16) | 15.07 t/s | 1.00 | 4.95 GB |
| llama.cpp | Q4_K_M | 13.3 t/s | 1.00 | ~5.5 GB |
| mlx-lm | OptiQ-4bit | 13.25 t/s | 1.00 | 7.30 GB |

## 結論

- uzuはllama.cppの約4.7倍速（長文時t/f=11.12）
- 短い応答ではオーバーヘッドで微減（12 t/s）するが、実用的な長文で真価発揮
- M3でのベンチマーク修正が必須：shapes.json M3追加 + rht_wrapper.rs activation_quantizationチェック削除
- MLXはvenv Python 3.12 + mlx-lm 0.32.2 で動作確認（投稿の25.1 t/s未達、ただし投稿はM5前提の可能性）

## 追加実験

- Prompt長 vs 生成速度の夜間ベンチ計画: `notes/prompt-length-benchmark-plan.md`
- 実行スクリプト: `notes/prompt_sweep.py`
- 現時点の llama.cpp isolated 結果: 147 tok → decode 14.63 t/s、8207 tok → 8.89 t/s
- Ollama local model は同GGUF import で decode 6.17 t/s（要再測定）
- uzu 9B-M ローカル実行を復旧・検証済み: 長文で **75.76 t/s / t/f 14.09**
  - セットアップ: `notes/uzu-9b-local-setup.md`
  - 実行: `notes/run_uzu_9b.sh "プロンプト"`

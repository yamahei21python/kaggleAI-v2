# Qwen3.5 9B Prompt長 vs 生成速度 ベンチ計画

## 目的

- 同モデル・同量子化で prompt length を変え、prefill / decode t/s を測る。
- 既存資産のみ。モデルDLなし。
- 夜間に実行。M3 の熱ダウンを避けるため、各測定点の間に 120 秒 cooldown を入れる。

## 既存資産

| 項目 | 値 |
|---|---|
| GGUF | `/Users/kohei/Myproject/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` |
| Ollama import 済 | `qwen35-9b-local-q4km` |
| ベンチスクリプト | `notes/prompt_sweep.py` |
| llama.cpp | `llama-server` build 11669 |
| 環境 | Apple M3 / macOS |

## 重要: M3 熱ダウン

- 連続実行すると decode が 15 t/s → 6 t/s まで落ちる。
- 30 秒 cooldown では不足。**120 秒 cooldown + 測定前に 180 秒 idle** が必要。
- 下の「現時点の結果」にある batch 結果は熱の影響を含む参考値。夜間は isolated で取り直す。

## 夜間実行コマンド

### 0. 事前 cooldown

```bash
cd /Users/kohei/Myproject/kaggleAI
sleep 180
```

### 1. llama.cpp full isolated sweep

```bash
caffeinate -ims python3 notes/prompt_sweep.py \
  --engine llama \
  --model /Users/kohei/Myproject/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --lengths 128,256,512,1024,2048,4096,8192 \
  --gen 256 \
  --cooldown 120 \
  --ctx 16384 \
  --port 18080 \
  --out notes/prompt-sweep-llama-full.json \
  --server-log notes/prompt-sweep-llama-full-server.log
```

### 2. Ollama full isolated sweep

```bash
sleep 180
caffeinate -ims python3 notes/prompt_sweep.py \
  --engine ollama \
  --model qwen35-9b-local-q4km \
  --lengths 128,256,512,1024,2048,4096,8192 \
  --gen 256 \
  --cooldown 120 \
  --ctx 16384 \
  --out notes/prompt-sweep-ollama-full.json
```

### 3. 任意: output length sweep

prompt 2048 固定、生成長だけ変更。llama.cpp のみ。

```bash
for n in 64 128 256 512; do
  sleep 180
  caffeinate -ims python3 notes/prompt_sweep.py \
    --engine llama \
    --model /Users/kohei/Myproject/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    --lengths 2048 \
    --gen "$n" \
    --cooldown 0 \
    --ctx 16384 \
    --port 18080 \
    --out "notes/prompt-sweep-llama-gen${n}.json" \
    --server-log "notes/prompt-sweep-llama-gen${n}-server.log"
done
```

### 4. 任意: 再現性確認

同じ条件で full sweep をもう一度実行し、中央値を見る。

```bash
sleep 180
caffeinate -ims python3 notes/prompt_sweep.py \
  --engine llama \
  --model /Users/kohei/Myproject/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --lengths 128,1024,4096,8192 \
  --gen 256 \
  --cooldown 120 \
  --ctx 16384 \
  --port 18080 \
  --out notes/prompt-sweep-llama-repeat.json \
  --server-log notes/prompt-sweep-llama-repeat-server.log
```

## 解析方法

- `prefill_tps` = `timings.prompt_per_second`
- `decode_tps` = `timings.predicted_per_second`
- `prompt_tokens` = 実測値。target ではなく実測値を使う。
- グラフ: x=`prompt_tokens`, y=`decode_tps` / `prefill_tps`
- llama.cpp と Ollama は同じ GGUF だが、Ollama runner 設定差で速度が変わる可能性あり。同一条件で比較する。

## 現時点の結果

### llama.cpp isolated（熱対策あり、採用候補）

| prompt_tokens | prefill t/s | decode t/s |
|---:|---:|---:|
| 147 | 147.46 | 14.63 |
| 1037 | 174.97 | 14.86 |
| 4117 | 143.59 | 11.14 |
| 8207 | 133.90 | 8.89 |

### llama.cpp batch（熱の影響あり、参考値）

| prompt_tokens | prefill t/s | decode t/s |
|---:|---:|---:|
| 147 | 152.08 | 15.23 |
| 277 | 162.23 | 15.32 |
| 527 | 155.14 | 14.82 |
| 1037 | 146.50 | 11.60 |
| 2067 | 124.28 | 8.76 |
| 4117 | 155.56 | 12.91 |
| 8207 | 74.24 | 5.93 |

### Ollama local model（途中、1点のみ）

| prompt_tokens | prefill t/s | decode t/s |
|---:|---:|---:|
| 147 | 145.89 | 6.17 |

- Ollama は `qwen35-9b-local-q4km`（同じ GGUF を import）を使用。
- 100% GPU だが llama-server より decode が遅い。runner 設定差の可能性。夜間 full sweep で再確認。

## 既知のブロッカー

- **uzu**: バイナリ未ビルド。`cargo run` が必要。M3 修正（`shapes.json` M3追加 + `rht_wrapper.rs` activation_quantization チェック削除）未適用。
- **MLX**: インストール済み `mlx-lm 0.29.1` は `qwen3_5` 非対応。`mlx-lm 0.32.2` + Python 3.12 の venv が必要。既存 venv なし。DL不可なら保留。
- **Ollama**: 同じ GGUF でも decode が遅い。`num_batch`、flash attention、KV cache、Ollama runner version を確認する価値あり。

## 総所要時間の目安

| 項目 | 時間 |
|---|---:|
| llama.cpp full | 約20分 |
| Ollama full | 約25分 |
| output length sweep | 約15分 |
| repeat | 約10分 |
| 合計 | 約70分 |

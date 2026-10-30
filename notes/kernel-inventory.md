# カーネル棚卸し（2026-10-30）

移行 1 の成果物。**カーネル id は変えない**（Secrets・dataset_sources・実行履歴が
id に紐づく）。ここでは「どれが現役で、どのビルダーが生成し、どこまで検証されて
いるか」だけを固定する。

- 現役: 常用する。リファクタ後も同じ id で push する。
- 準現役: 必要時に使う（測定・CPU 検証・代替経路）。壊さないが優先度は低い。
- 退役: 参照しない。ビルダーは残すが再生成・push の対象外。

## 一覧

| 状態 | カーネル id | ローカル | 生成元 | 形式 / GPU | datasets |
|---|---|---|---|---|---|
| 現役 | `step1-orcarouter-qwen38-27b-dl` | `kaggle/step1/` | `build_step1.py` | notebook / CPU | 0 |
| 現役 | `step2-orcarouter-qwen38-27b-cloudflared` | `kaggle/step2/` | `build_step2.py`（既定） | notebook / GPU | 3 |
| 現役 | `step2-orcarouter-qwen38-27b-ngrok-fixed` | `kaggle/step2-ngrok/` | `build_step2.py` `STEP2_TUNNEL=ngrok` | notebook / GPU | 2 |
| 現役 | `prepare-vllm-dataset` | `kaggle/step2-vllm/` | `build_vllm_dataset.py` | notebook / CPU | 0 |
| 現役 | `step2-vllm-qwen38-27b-cloudflared` | `kaggle/step2-vllm-kernel/cloudflared/` | `build_step2_vllm.py`（既定） | notebook / GPU | 3 |
| 現役 | `step2-vllm-qwen38-27b-ngrok` | `kaggle/step2-vllm-kernel/ngrok/` | `build_step2_vllm.py` `STEP2_TUNNEL=ngrok` | notebook / GPU | 2 |
| 現役 | `keepalive-test` | `kaggle/step2/`（`kernel-metadata-keepalive-test.json`） | `build_keepalive_test.py` | notebook / CPU | 0 |
| 現役 | `exl3-qwen38-gpu-check` | `kaggle/exl3-gpu-check/` | なし（`main.py` 直接編集） | script / GPU | 1 |
| 現役 | `exl3-dataset-upload` | `kaggle/exl3-dataset-upload/` | なし（直接編集） | script / CPU | 0 |
| 現役 | `exl3-serve` | `kaggle/exl3-serve/kernel/` | `exl3-serve/build_kernel.py` | script / GPU | 2 |
| 現役 | `exl3-serve-dryrun` | `kaggle/exl3-serve/dryrun/` | 同上（mock） | script / CPU | 1 |
| 現役 | `exl3-serve-keepalive` | `kaggle/exl3-serve/keepalive/` | 同上（keepalive） | script / CPU | 0 |
| 現役 | `llama-server-cpu-check` | `kaggle/cpu-check/` | なし（直接編集） | script / CPU | 2 |
| 準現役 | `step2-direct-llama-server-ngrok` | `kaggle/step2-direct-ngrok/` | `build_step2_direct.py`（既定 ngrok） | notebook / GPU | 2 |
| 準現役 | `step2-vllm-dryrun` | `kaggle/step2-vllm-dryrun/` | `build_vllm_dryrun.py` | notebook / CPU | 2 |
| 準現役 | `bench-ctx-vs-speed-orcarouter-qwen3-8-27b` | `kaggle/bench/` | `build_bench.py` | notebook / GPU | 0 |
| 準現役 | `bench-mtp-spec-decoding-orcarouter-qwen3-8-27b` | `kaggle/bench_mtp/` | `build_bench_mtp.py` | notebook / GPU | 0 |
| 準現役 | `smoke-orcarouter-qwen38-27b` | `kaggle/smoke/` | `build_smoke.py` | notebook / GPU | 0 |
| 準現役 | `exllamav2-cpu-preflight` | `kaggle/exllama-cpu-preflight/` | なし（直接編集） | script / CPU | 0 |
| 退役 | `capture-ollama-patch-v2` | `kaggle/capture_patch/` | `build_capture_patch.py` | notebook / CPU | 0 |
| 退役 | `keepalive-test-1` | `kaggle/step2-ka-test/` | なし（旧版が残存） | notebook / CPU | 0 |

`kaggle/archive/` の 46 スナップショット（json+ipynb）は上表に含めない。
削除せず、検索用に残す。

## 注意点

1. **`keepalive-test` と `keepalive-test-1` は別カーネル**。現役は
   `kaggle/step2/step2-keepalive-test.ipynb` + `kernel-metadata-keepalive-test.json`
   （id `keepalive-test`）。`kaggle/step2-ka-test/` は旧 `keepalive-test-1` で、
   どのビルダーも生成しない。混同しやすいので、移行時は `step2-ka-test/` を
   退役として扱い、`step2/` 側だけをビルダー対象に保つ。
2. **`step2-direct` は ngrok 版だけコミットされている**。`STEP2D_TUNNEL=cloudflared`
   を指定すると `kaggle/step2-direct/` が新規生成される（未追跡）。
3. **`exl3-serve/{kernel,dryrun,keepalive}/` は生成物**。`build_kernel.py` から
   再生成でき、現時点では未追跡。commit / gitignore の確定は F-3。
4. **ビルダーを持たない 4 本**（`exl3-gpu-check`, `exl3-dataset-upload`,
   `cpu-check`, `exllama-cpu-preflight`）は `main.py` が真実源。kbuild 移行の
   対象外だが、`runtime/` の probe / net を使えるようになったら寄せる。
5. **`build_step2_direct.py` は既定が ngrok**（`build_step2.py` / vLLM は既定
   cloudflared）。kbuild の spec ではトンネルを必須項目にして既定値の揺れを消す。

## 生成一致チェックの現状

`kaggle/check_notebooks.py` が再生成して比較しているのは 7 冊:

- `step1` / `step2`（cloudflared・ngrok）/ `smoke` / `bench` / `bench_mtp` /
  `step2-keepalive-test`

対象外（理由）:

- `step2-direct-ngrok` … TARGETS 未登録。追加は段 3 で行う。
- `step2-vllm-kernel/*`、`step2-vllm-dryrun` … 同じく未登録。段 4–5 で追加。
- `exl3-serve/*` … script 方式。`_selftest_build.py` がバイト一致を確認。
- ビルダー無しの script 4 本 … 生成物ではなくソースそのもの。

## 移行 2 への引き継ぎ

- 最初に移すのは **`exl3-serve`（単一 Template・バイト一致が取りやすい）** と
  **`build_step2.py`（cells 外部化済み）**。
- `build_step2_direct.py` / `build_vllm_dryrun.py` は準現役なので、移行は
  段 3–4 でよいが、probe / net の共有だけは段 2 で先に済ませる。
- bench / smoke / bench_mtp は `_gpus` / `_chat` の第 2 系統。段 6 まで放置すると
  共通化が半分で止まるため、probe / net は段 2 で `runtime/` に寄せる。

# CPU テスト結果（GPU 0 秒 / 2026-10-31 更新）

`kaggle/cpu-tests/` の 4 本＋`whl-probe` の実測。t/s は CPU なので**相対差**だけ。

## tiny-llama（llama-server / 0.5B q4_k_m / CPU 4vCPU）

| ケース | 起動 | 生成 t/s | 判定 |
|---|---|---|---|
| base | 2.1s | 27.3 | ✅ |
| kv_q8（q8_0 KV + `-fa on`） | 2.0s | 27.3 | ✅ |
| spec_mtp（`--spec-type draft-mtp`） | — | — | ❌ `model doesn't contain MTP layers` |
| parallel2（`-np 2`, ctx 8192） | 2.0s | 27.9 | ✅ |
| threads4（`-t 4`） | 2.0s | **31.4** | ✅ +15% |

- フラグ実在 9/9（`--spec-type`/`draft-mtp`/`--cache-type-k/v`/`-fa`/`--jinja`/`--reasoning-budget`/`-np`/`--parallel`/`-t`）
- `draft-mtp` は MTP 層がないと起動拒否 → 27B（GPU）で判定
- CPU では q8_0=f16（量子化効果は VRAM 側）

## tiny-ollama（0.5B / `/api/ps` 実値）

| ケース | env ctx | 実 ctx | t/s |
|---|---|---|---|
| base（f16） | 4096 | 4096 | 32.5 |
| kv_q8 | 4096 | 4096 | 32.0 |
| ctx32k | 32768 | 32768 | 30.1 |
| ctx128k | 131072 | **32768** | 28.4 |
| np2（`OLLAMA_NUM_PARALLEL=2`, ctx 8192） | 8192 | 8192 | 32.1 |
| → 同時 2 本 | — | — | **12.5 t/s 総 throughput（0.96s）** |

- env は無視されない（`context_length` 一致）
- **ctx は 2 段階**: `env で指定 → 学習上限でクランプ → num_parallel でスロット分割`
  （131072→32768 は学習上限、np2→8192 は 8192/2）
- size_vram=0（CPU）→ GPU 判定式 `size_vram/size` はそのまま使える

## vllm-flag-probe（vLLM 0.32.0 / CPU / `VLLM_TARGET_DEVICE=cpu`）

**v2 完走（2026-11-01 取得、対照実験 2 本込み）**。install 306.1s / `--help` 18s /
argv 8 ケース 361s / 合計 ~12 分。**GPU クォータ 0 秒**。

`EngineArgs` フィールド **11/11 実在**（`speculative_config` / `enforce_eager` /
`kv_cache_dtype` / `max_model_len` / `language_model_only` / `attention_backend` /
`compilation_config` / `quantization` / `max_num_seqs` / `gpu_memory_utilization` /
`enable_prefix_caching`）。

### argv 受理（`rejected_by_argparse` が False = 引数段階は通った）

| ケース | argv | 判定 | 落ち方 |
|---|---|---|---|
| `spec_mtp` | `--speculative-config {"method":"mtp",...}` | ✅ 受理 | 後段で `NotImplementedError`（下記） |
| `spec_mtp_draft` | 同上 + `model` | ✅ 受理 | 同上 |
| `kv_fp8` | `--kv-cache-dtype fp8` | ✅ 受理 | engine 起動で失敗（CPU なので当然） |
| `eager` | `--enforce-eager` | ✅ 受理 | 同上 |
| `maxlen8192` | `--max-model-len 8192` | ✅ 受理 | 同上 |
| `lmo` | `--language-model-only` | ✅ 受理 | 同上 |
| `triton` | `--attention-backend TRITON_ATTN` | ✅ 受理 | 同上 |
| **`kv_q8`（負）** | `--kv-cache-dtype q8_0` | ❌ **invalid choice** | argparse |
| **`bogus`（負）** | `--definitely-not-a-flag` | ❌ **unrecognized arguments** | argparse |

負のケースが 2 本とも argparse で弾かれた → `detector_works=true`。以後の
「受理」は信用してよい。

### ⚠ `method: "mtp"` は stub では解決できない（本番は成功している）

```
NotImplementedError: Unsupported speculative method: 'mtp'
  vllm/config/speculative.py:1389 __post_init__
```

これは**失敗ではなく stub モデルの限界**。根拠:

- `--spec-method` の choices に `mtp` は**実在する**（下記一覧）
- 本番 #2 は**同じ argv**（`{"method":"mtp","num_speculative_tokens":2,"model":"/kaggle/tmp/vllm-draft"}`）
  で起動に成功し、ログに
  `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings`
  が出ている（= speculative 有効）
- 差は target の `config.json`。実モデルなら architecture から具体的な MTP 実装に
  解決できるが、stub の偽 config では解決先が無い

→ **CPU プローブで MTP の可否は判定できない**（`gpu_unknown` 行き）。ここは
「引数が通る」までで止めるのが正しい。

### `--spec-method` choices（全 41 個、`mtp` を含む）

`bailing_hybrid_mtp` `bailing_hybrid_v3_mtp` `custom_class` `deepseek_mtp` `dflash`
`dots3_note_mtp` `draft_model` `dspark` `eagle` `eagle3` `ernie_mtp` `exaone4_5_mtp`
`exaone_moe_mtp` `extract_hidden_states` `gemma4_mtp` `glm4_moe_lite_mtp` `glm4_moe_mtp`
`glm5_next_mtp` `glm_ocr_mtp` `hy_v3_mtp` `hy_v4_mtp` `inkling_mtp` `kimi_k3_mtp`
`longcat_flash_mtp` `medusa` `mimo_mtp` `mimo_v2_mtp` `minimax_m3_mtp` `mlp_speculator`
**`mtp`** `nemotron_h_mtp` `ngram` `ngram_gpu` `pangu_ultra_moe_mtp` `qwen3_5_mtp`
`qwen3_next_mtp` `qwen4_exp_mtp` `step3p5_mtp` `suffix` `None`

### `--kv-cache-dtype` 選択肢（19 個）— **q8_0 は無い**

`auto` `bfloat16` `float16` `fp8` `fp8_ds_mla` `fp8_e4m3` `fp8_e5m2` `fp8_inc`
`fp8_per_token_head` `int4_per_token_head` `int8_per_token_head` `nvfp4` `nvfp4_4over6`
`nvfp4_ds_mla` `turboquant_3bit_nc` `turboquant_4bit_nc` `turboquant_k3v4_nc`
`turboquant_k8v4` `ultraquant_4bit`

→ llama.cpp の `q8_0` に相当するものは**無い**。vLLM で KV を軽くするなら `fp8`。

### `--attention-backend` 選択肢（47 個）

`FLASH_ATTN` `FLASH_ATTN_DIFFKV` `TRITON_ATTN` `TRITON_ATTN_DIFFKV`
`TRITON_FLASH_ATTN` `ROCM_*` `TORCH_SDPA` `FLASHINFER` `*_MLA*` `FLEX_ATTENTION`
`CPU_ATTN` `CPU_MLA` `AMX_MLA` `TURBOQUANT` `ULTRAQUANT` `CUSTOM` ...
T4（sm75）で実用的なのは **`TRITON_ATTN`（既定）** と `FLEX_ATTENTION` 程度。
`FLASH_ATTN` は capability>=8.0 要求で選ばれない。

### プローブ自身のバグを 1 件修正

v1 は **完走しているのに `ok=false`** を返していた。`kv_q8`（存在しない値）を
positive 側に数えていたため。`_NEGATIVE = ("bogus", "kv_q8")` を切って
`ok` と `detector_works` を分離した（修正後の実データ検算で `ok=true`）。
`tests/test_cpu_probes.py::test_vllm_flag_verdict_excludes_designed_negatives`
で固定。

## vllm-whl-probe（実行済）

- wheelhouse 190 wheel/3.25 GB + patch マウント OK
- `--no-index` は **まだ解決不能**（cuda-toolkit 13.0.3 vs humming-kernels[cu13] 衝突）
- 対策: install セルが wheelhouse 失敗→PyPI 自動フォールバック

**追記（2026-11-01 / v10 実測）**: `vllm-wheelhouse-patch`（cublas 13.1.1.3）を
足したことで **`--no-index` は解決できるようになった**（v10 のログに
`📦 ローカル wheelhouse: vllm-wheelhouse（190 wheel）` が出て、PyPI への
フォールバックは発動していない）。ただし**速くはない**:

| 経路 | pip の所要 |
|---|---|
| PyPI（v9） | 199.6s |
| wheelhouse `--no-index`（v10） | **230.4s** |

転送を消しても縮まない ＝ 支配的なのは**展開と小ファイル書き込み**。
「転送が支配（3.94 GB / 137s）」という見積りは外れていた。この床を動かせるのは
展開の仕方だけなので、`vllm-install-bench`（CPU・GPU 0 秒）で 4 通りを測る。

## vllm-install-bench（CPU・GPU 0 秒）

`pip install --target D` で毎回まっさらな D に入れ、所要・サイズ・ファイル数を
比べる。`--target` はシステムを汚さないので 4 通りを同じ条件で繰り返せる
（本番は purge + システムへ install なので絶対値はずれるが、**差の向き**は効く）。

| 戦略 | 何を変えるか |
|---|---|
| `pip` | 本番と同じ（解決 + 展開 + pyc 生成） |
| `pip-nocompile` | `--no-compile`。pyc を書かない（初回 import で遅延コンパイルに移る） |
| `uv` | 並列展開。pip の Python 実装より速いことが多い |
| `pip-paths-nodeps` | `--dry-run --report` で wheel パスを出し、`--no-deps` で直指定（依存グラフを解かない） |
| `pip-again` | 対照。2 回目が速ければ「キャッシュ暖まり」を測っているだけ |

### 結果（2026-11-01 実行・`vllm-install-bench` v1）

同じ 190 wheel（+ patch 11）を `--target` へ展開した所要。`uv 0.13.5` / pip 26.2.2。

| 戦略 | 所要 | サイズ | ファイル数 | 対 warm pip |
|---|---|---|---|---|
| `pip`（1 回目・冷） | 394.5s | 8,517 MiB | 81,844 | — |
| `pip-nocompile` | **112.6s** | 8,032 MiB | 56,416 | **-44%** |
| **`uv`** | **35.7s** | 8,033 MiB | 56,812 | **-82%** |
| `pip-paths-nodeps`（`--no-compile` 込み） | 101.5s | 7,608 MiB | 47,835 | -49% |
| `pip`（2 回目・暖） | 200.1s | 8,517 MiB | 81,844 | 基準 |

`cache_warm_bias = 0.51`（2 回目の pip が 1 回目の 2 倍速い）。
つまり 1 回目の 394.5s は**ディスクが冷えていた分**を含む。公平な比較は
2 回目以降（暖）との差で読む。

### 読み取れること

1. **支配的なのは pyc の生成**。`--no-compile` だけで 200.1 → 112.6s（-87.5s）、
   ファイル数が 81,844 → 56,416（**-25,428 個＝ pyc**）。サイズは 8,517 →
   8,032 MiB（-485 MiB）しか変わらないのに、時間は 44% 減る。
   ＝ **小ファイルの数**が効いている（1 ファイルあたりの書き込みが遅い）。
2. **uv はさらに 3 倍速い**（112.6 → 35.7s）。同じ 56,812 ファイルを書いているので、
   差は「書き方」（並列・バッファ）であって量ではない。
3. **依存解決を飛ばす効果は小さい**（`--no-deps` + パス直指定で 101.5s）。
   `--no-compile` 込みなので nocompile（112.6s）との差は **-11.1s**。
   190 wheel の解決は 10 秒程度。ここを削っても意味は薄い。
4. 入った版は 3 通りとも同じ（`torch 2.13.0` / `torchaudio 2.11.0` /
   `vllm 0.32.0` / `triton 3.7.1`）。uv に版ずれは無い。

### GPU への当てはめ — **外れた**（2026-11-01 実測）

推定 41s に対し、本番（torch を物理削除した site-packages へ install）は
**1151.7s**。同じ wheel・同じホストで **pip 261.6s の 4.4 倍遅い**。

| 経路 | install | 内訳 |
|---|---|---|
| v10 pip + wheelhouse | 271.3s | pip 230.4 |
| v11 pip + wheelhouse | 304.2s | pip 261.6 |
| v12 uv + wheelhouse | 1208.5s | **uv 1151.7** |

`--target`（まっさらな D）と site-packages では**比すら移らない**。
理由は未同定（uv のキャッシュ経路が GCSFUSE 上の wheel を引き直している疑い）。
→ **既定は pip に戻した。** uv は `KC_INSTALLER=uv` の明示時のみ。

**教訓**: `--target` のベンチで言えるのは「展開の仕方の差」まで。
install 先が違えば絶対値も比も本番に持ち込めない。既定を変える前に、
本番と同じ形（site-packages へ入れる）で 1 回測る。

## vllm-io-bench（CPU・GPU 0 秒）

重みロードが「なぜ固まるのか」を切り分ける計器。GCSFUSE の読み出し 3 通りと、
各フェーズの `meminfo` を残す。

### 結果（2026-11-01 実行・v1）

| 位相 | 実測 | 意味 |
|---|---|---|
| GCSFUSE 逐次読み（1 本・64 MiB バッファ） | **43.8 MiB/s** | 17.41 GiB で **407s** |
| GCSFUSE を mmap して頁を触る（cache 済） | **5,402 MiB/s** | cache に載れば事実上ゼロ |
| ローカルへ実コピー | **不可** | `/kaggle/tmp` が無い（CPU カーネル）／`/kaggle/working` は 20 GB で 18.7 GB は入らない |

`meminfo` の推移:

| 時点 | MemAvailable | Cached |
|---|---|---|
| 開始 | 31,164 MiB | 3,230 |
| 2 GiB 読んだ後 | 31,119 MiB | 5,282 |

**読み出しは `MemAvailable` を削らない**（`Cached` は増えるが回収可能）。
つまり v12 で消えた 24 GiB は **page cache ではない**。

### 読み取れること

1. **床は 43.8 MiB/s**。1 本読みだと 17.41 GiB に 407 秒。しかもこれは
   64 MiB バッファの**バルク読み**の値で、頁フォールト駆動の mmap はもっと遅い
   （＝ v12 の「50% で 800 秒動かない」と整合する）。
2. **v10 の 133 秒は 140 MiB/s** = 1 本読みの **3.2 倍**。並列に読めば同じ速度が
   出る、という意味では「同じ並列度で先に読めば同じ速度が出る」。
   ~~vLLM の auto-prefetch は並列読みなので~~
   → **2026-11-01 訂正**: GCSFUSE では auto-prefetch は**常に無効**（`is_net_fs`
   が False。`notes/vllm-startup-speedup.md` §19）。v10 の 140 MiB/s を
   「prefetch のおかげ」とは説明できない。明示フラグ
   `--safetensors-load-strategy=prefetch` なら有効化できるが、このリポジトリでは
   まだ一度も試していない。
3. **ローカル実コピーは元が取れない**。コピー自体が同じ 43.8 MiB/s の読み出しで
   407 秒かかり、その後のローカル読みが速くても合計は 407 秒以上。しかも
   18.7 GB を置く場所が無い（`/kaggle/working` は 20 GB）。
4. **消えた 24 GiB は page cache ではない**ので、`drop_caches` では戻らない
   可能性が高い。ただし「戻るなら 1 秒で戻る」ので、起動前に試す価値はある。

### v2 の追加測定（並列度・drop_caches）— 2026-11-01

| 読み方 | 実測 | 1 GiB の所要 |
|---|---|---|
| GCSFUSE 1 本（64 MiB バッファ） | **110〜130 MiB/s** | 8〜33s |
| GCSFUSE 2 本（32 MiB チャンク） | **2.3 MiB/s** | 450s |
| GCSFUSE 4 本 | **2.2 MiB/s** | 460s |
| GCSFUSE 8 本 | **2.2〜2.3 MiB/s** | 449〜461s |
| GCSFUSE 16 本 | **2.2〜2.4 MiB/s** | 434〜456s |
| page cache に載った後の mmap | **5,677 MiB/s** | 0.4s |

**並列にすると 13〜50 倍遅い。** 逆順でも同じなので順序効果ではない。

`drop_caches`:

```
OSError: [Errno 30] Read-only file system: '/proc/sys/vm/drop_caches'
```

**Kaggle では書けない。** 空きは 31,112 MiB のまま動かない（`delta_avail: 0`）。
`posix_fadvise(DONTNEED)` は成功する（自ファイルの cache だけなら捨てられる）。

### v1 の解釈を 2 つ訂正

1. **「v10 の 140 MiB/s は並列読みの証拠」は誤り。** 1 本読みでも回によって
   43.8 → 130 MiB/s と 3 倍動く。v10 は**回線が速い回だっただけ**。
   この誤読で v13 に 8 並列の先読みを入れてしまい、2.3 MiB/s × 18.21 GiB =
   **約 135 分**を溶かした（実測で発覚、v14 で 1 本に修正）。
2. **`drop_caches` は対策にならない。** 読み取り専用なので、起動前の
   空き確保は「捨てる」では実現できない。

**教訓**: 「速い回の数字」から機構を推定しない。同じ条件で 1 本と N 本を
**並べて測る**（今回はそれが v2 の位相 5 で、13〜50 倍という桁違いが出た）。

## 決定事項（GPU 本番に反映）

1. **vLLM の KV 量子化は q8_0 禁止** → `fp8` を使う（`q8_0` は選択肢にすら無い）
2. **Ollama ctx は「学習上限でクランプ＋np 分割」** → /api/ps 実値で判定（env を信じない）
3. **MTP は 27B でしか判定不能** → tiny CPU では「フラグ受理」まで
4. **vLLM --help は CPU で device 推論に落ちる** → `VLLM_TARGET_DEVICE=cpu` 必須
5. **`method:"mtp"` の解決は target の config 依存** → CPU プローブの
   `NotImplementedError` を「非対応」と読まない（本番は成功済み）


## SGLang install probe（2026-11-01 13:19・GPU 0 秒）

`kaggle/cpu-tests/sglang-probe`（`sglang-install-probe`）。**`ok=true`**。

背景: `step2-sglang-qwen38-27b-cloudflared` は 297 秒で `pip install 失敗 (1)`。
`install_tail` は `Failed building wheel for outlines_core` だけで、
**エンジンは一度も起動していなかった**（「T4 で動かない」は未検証だった）。

### 原因（pip に判定させた・推測ではない）

| 対象 | cp313 wheel | 実測 |
|---|---|---|
| `outlines-core==0.1.26`（`outlines==0.1.11` が完全固定） | **無い** | `pip download --only-binary=:all:` が失敗（1.0s） |
| `outlines-core==0.1.17` | **ある** | 成功（0.7s） |
| `cuda-tile==1.6.0rc5` | 無い（sdist） | ただしビルドは通る（失敗一覧に出ない） |

直接依存 26 本を全部調べた結果、**cp313 wheel が無いのはこの 2 本だけ**。
`outlines_core` は Rust 実装（maturin）なので、Rust が無いと sdist ビルドで落ちる。

### 2 つの筋の実測

| 筋 | 手順 | 実測 | 結果 |
|---|---|---|---|
| **A（採用）** | rustup → 素の `pip install sglang==0.5.22` | **9.4s + 389.8s** | ✅ `import sglang` 26.6s / `--help` 12.5s |
| B（保留） | `--no-deps` + `outlines-core==0.1.17` + 残りを手で | 未実行（A が通ったため） | — |

環境: python 3.13.15 / pip 26.2.2 / gcc あり / **cargo・rustc は無い**（だから落ちた）。
install 後は `torch 2.14.1+cu130` に差し替わる（イメージは 2.13.0+cu130）。

### 実装への反映

- `runtime/engines/sglang.py`: `kc_sglang_pip_argv` → **`kc_sglang_pip_steps`**（2 ステップ）。
  `KC_SGLANG_ROUTE=pin` で筋 B に切り替わる（未検証なので既定にしない）。
- `kbuild/serve_script.py`: `PIP_ARGV`（1 回）→ **`PIP_STEPS`（複数）**。
  失敗時に `step 2/3: <cmd>` を例外へ入れ、`install_steps[]` に全ステップを残す。
  LMDeploy は `[pip_argv]` の 1 要素に包まれて**従来通り**動く。
- `install_seconds` は**ステップ実測の和**にした（壁時計だと、`sh` を差し替えた
  テストで 0 秒になり記録の意味が消える）。

**教訓**: 「wheel が無い」を手で書かない。`--only-binary=:all:` を付けた
`pip download` に判定させれば、上流が wheel を足した日に自動で嘘になる。

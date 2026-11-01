# 落とし穴とテストの対応表（2026-11-01 20:55 更新）

GPU 実行で**実際に踏んだ**不具合を、1 件 1 テストで固定している。
「同じ穴に二度落ちない」ことが目的なので、テストは**構造**を見る
（防御が存在するか / 順序が正しいか / 空・0・例外で素通りしないか）。
性能値そのものは検証しない（それは実行ログ側の仕事）。

## テストの置き場所

| ファイル | 役割 | 本数 |
|---|---|---|
| `tests/test_gpu_pitfalls.py` | **症状 → 防御**の一覧（この文書の表） | 23 |
| `tests/test_vllm_cells.py` | vLLM の**セル内部**の契約（順序・版の出どころ・可観測性） | 27 |
| `tests/test_serve_script_guards.py` | **全エンジン共通**の足場（`wait_http` / `find_model` / `stage_env` / `ENGINE_PATTERN` / `PIP_STEPS`） | 17 |
| `tests/test_startup_timing.py` | **計測**（節目の解釈・timings の合流・`cli timings`） | 21 |
| `tests/test_cpu_probes.py` | CPU プローブ（GPU 0 秒）の契約 | 25 |
| `tests/test_spec.py` | 台帳（`kbuild/spec.py`）と生成物の一致 | 57 |
| `tests/test_engines.py` | `runtime/engines/*` の差し替え | 20 |
| `tests/test_cli.py` | `cli` のサブコマンド | 15 |
| `tests/test_vllm_playground.py` | 実験カーネル（`KC_EXPERIMENT` の効き方） | 9 |
| `tests/test_smoke_checks.py` | 疎通確認の判定（SKIP 契約・引数の整合） | 6 |
| その他（`test_serve_notebook` / `test_step2_builder` / `test_runtime` / `test_constants` / `test_exl3_*` / `test_pinned_wheel` / `test_kbuild` / `test_secrets` / `test_notebooks` / `test_step2_proxy`） | — | 41 |
| `tests/_cells.py` | セルから関数を切り出す共通ヘルパー（規則はここ 1 箇所） | — |

実行: `python3 -m pytest tests/ -q`（**261 本**）。
落とし穴の一覧だけなら `python3 -m pytest tests/test_gpu_pitfalls.py -q`。

**同じアサーションを 2 箇所に置かない**。整理のとき、`torchaudio の版` を
2 ファイルに重複させていたのを 1 箇所（`test_vllm_cells.py`）に寄せた。

| # | 症状（実測） | 防御の実装場所 | テスト |
|---|---|---|---|
| 1 | 重み 19.5 GB のロードが 244s（`/kaggle/input` は 82 MiB/s） | `_stage_model` が **symlink のみ**（実体コピー無し） | `test_gpu_pitfalls.py::test_model_staging_uses_symlink_not_copy` |
| 2 | torch 2.13.0+cu130 と torchaudio 2.8.3+cu128 の混在で serve が死ぬ。`import vllm` では検出できない（transformers → torchaudio は**遅延 import**） | install セルが `import torchaudio` を**子プロセスで単独実行** | `test_gpu_pitfalls.py::test_torchaudio_is_probed_by_import_before_serve` / `test_vllm_cells.py::test_torchaudio_version_comes_from_vllm_metadata` |
| 3 | `libggml-cuda.so` 欠落 → エラー無しで CPU 推論（0.9 t/s） | `01_restore_binaries.py` の検証対象に `CUDA_LIB` を入れる（欠ければ fail） | `test_cuda_backend_lib_is_a_hard_failure` |
| 4 | **vLLM も黙って CPU 推論**（13.7 t/s が 0.9 t/s に）。`/v1/models` は 200 を返す | warmup で `nvidia-smi memory.used` を見て 4 GiB 未満なら fail / 検証 `[5] gpu-resident` | `test_cpu_fallback_detected_when_nvidia_smi_returns_nothing` / `test_verify_checks_gpu_residency` |

| 5 | `--kv-cache-dtype fp8` は T4（sm75）非対応で起動失敗 | `KV_CACHE_DTYPE = "auto"` 固定（CPU プローブで選択肢 19 個を確認済み） | `test_kv_cache_dtype_is_auto_and_never_fp8` |
| 6 | 32K ctx で `No available shared memory broadcast block` が 60 秒間隔 → 打ち切り。**v14 で `eager` でも 373 秒出ると判明**（＝ compile ではない） | ラダーに `--enforce-eager` を置くが、**これで無音は消えない**（消えるのは compile の分だけ）。正体は memory profiling 中の GDN Triton JIT | `test_ladder_has_eager_step_near_the_front` |
| 7 | `terminate()` では Worker_TP が残り VRAM を掴む → 次試行が 125s で即死 | `os.killpg`（TERM→KILL）+ `/dev/shm` 掃除 + `start_new_session` + **ドライバの PID で kill** + 解放できなければ fail | `test_engine_stop_kills_the_process_group_and_sweeps_shm` / `test_kill_patterns_match_the_real_process_titles` / `test_vram_is_freed_by_driver_pid_not_by_name` / `test_startup_fails_fast_when_vram_cannot_be_freed` |
| 8 | コンテナは毎回真っ新 → cloudflared を毎回 DL | `_find_cached` が専用 Dataset（`CF_PATH`）を先に見る（flatten 対策で 2 段） | `test_cloudflared_prefers_the_dataset_cache` |
| 9 | `/v1/models` は 200 でも**初回 forward** で落ちる（Marlin / GDN Triton JIT / MTP draft） | 起動直後に `max_tokens: 16` の生成スモーク。失敗は次の段へ | `test_startup_requires_a_generation_smoke` |
| 10 | Ollama の qwen3.5 系は `sched.go` の blocklist で numParallel=1 固定 | `OLLAMA_NUM_PARALLEL="1"` + 理由をコメント | `test_ollama_parallel_is_pinned_to_one_with_reason` |
| 11 | llama-server は 15 KB の薄いラッパー（実体は `libllama-server-impl.so` を dlopen）。1 MB 判定だと**良品を弾く** | `MIN_SIZE_LLAMA = 4096` を別扱い | `test_llama_server_size_check_is_relaxed` |
| 12 | `/dev/shm` 不足で TP>1 が即死 / 境界値だと起動後に詰まる | `_shm_required_mib(TP)` で**空き**を判定 → remount 試行 → なお不足なら fail | `test_shm_requirement_scales_with_tp` / `test_shm_shortage_fails_fast_after_a_remount_attempt` |
| 13 | 16.52 GiB は T4 1 枚（15 GiB）に載らない → TP=1 不可 | `TP_SIZE = 2 if len(_vram_each) >= 2 else 1` | `test_tensor_parallel_follows_the_gpu_count` |
| 14 | **CPU ベンチの比が本番に移らない**。`--target` の uv 35.7s が、site-packages では **1151.7s**（pip 261.6s の 4.4 倍遅い） | 既定は pip。uv は `KC_INSTALLER=uv` の明示時のみ | `test_vllm_cells.py::test_installer_defaults_to_pip` |
| 15 | 空き RAM 5.68 GiB で checkpoint 18.21 GiB → prefetch が切られ、GCSFUSE の mmap が `50%` で 800 秒停止（v11/v12）。**ただし 2026-11-01 訂正: GCSFUSE では空き RAM に関係なく auto-prefetch は常に無効**（`is_net_fs` が False）。RAM の数字は文言を選ぶだけ | 原因を**計器で 2 つに分けた**: 冷えた読み = `timings.prewarm` / 無音 = `startup_gaps`。`_drop_caches()` は Kaggle では**書けない**（`Errno 30`）ので保険として残すだけ | `test_page_cache_is_dropped_before_startup_when_ram_is_short` / `test_prewarm_reads_every_shard_and_reports_throughput` |
| 16 | 30,315 MiB → 5.68 GiB の **24 GiB がどこへ消えたか分からない**（60 秒ポーリングでは 1 区間に潰れる） | `_ram_trace()`（5 秒ごと・別スレッド）+ `_mem_line()` / `_top_rss()`。ループの外で保存するので**失敗時も残る** | `test_ram_breakdown_is_recorded_while_startup_stalls` / `test_ram_trace_is_saved_on_both_paths` |
| 17 | 先読みが 400 秒かかった回に、試行の予算が残らない | `STARTUP_BUDGET` の起点を先読みの**後**に置く | `test_prewarm_is_skipped_when_ram_is_short`（空き不足時は読まない） |
| 18 | **先読みは 152.5 秒払っても効かない**（v14）。1 発目 `Loading weights took` **64.37s** / 2 発目 **6.39s**。ワーカーの anon 8.88 GiB × 2 が page cache を追い出す | `KC_PREWARM=0` で切れる。**既定を外すのが次の判断**（v14 の実測） | `test_prewarm_picks_the_faster_thread_count` |
| 19 | **並列先読みは 13〜50 倍遅い**（2/4/8/16 本で 2.2〜2.4 MiB/s、1 本は 110〜130 MiB/s） | `KC_PREWARM_THREADS` の既定は **1**。probe で 1 本と N 本を測って速い方を選ぶ | `test_prewarm_picks_the_faster_thread_count` |
| 20 | `--enforce-eager` に落ちると **31.0 → 4.5 t/s（7 倍遅い）**。v14 はこれで「起動 −111 秒」に見えた | eager を「時間を稼ぐ fallback」として**数えない**（起動時間だけで判定しない） | テストにしない（速度は GPU のみ。`notes/gpu-run-sheet.md` v14） |
| 21 | `pip install` **1 回では SGLang が入らない**（`outlines_core==0.1.26` に cp313 wheel が無い） | `PIP_STEPS`（複数ステップ）+ rustup。失敗時は `step 2/3: <cmd>` を例外と `install_steps[]` に残す | `test_serve_script_guards.py::test_install_stops_at_the_failing_step_and_says_which` ほか 4 本 |
| 22 | wheel の有無を**手で書く**と、上流が wheel を足した日に嘘になる | `pip download --only-binary=:all:` に判定させる（推測しない） | `test_cpu_probes.py::test_sglang_probe_asks_pip_instead_of_asserting_no_wheel` |
| 23 | **対応 GPU の下限は上流のソースに書いてある**のに、GPU で確かめた（SGLang: `WNA16.get_min_capability() = 80`、T4 は 75。2 往復 18 分を無駄にした） | 新しいエンジンは `get_min_capability()` / 量子化スキームを**先に読む**（`notes/engine-rollout.md`） | テストにしない（上流のコード。台帳のコメントに残す） |
| 24 | **名前の付いた warmup を「コストの場所」と読み違えた**。`Warming up Qwen GDN Triton kernels` は 3 秒で、373 秒はその手前の profile run（`determine_available_memory`）で起きていた | ログの**行番号と時刻**で区間を確定してから施策を決める（`init engine ... took 376.24s` の内訳を割る） | `test_vllm_cells.py::test_triton_snapshot_happens_before_the_budget_starts` ほか |
| 25 | キャッシュ退避を**後片付け（キープアライブ終了時）に置いた**ため、v14 で一度も走らなかった（成果物に tar が無い） | 退避は「成功・失敗の合流点」に置く。失敗した回のカーネルもディスクに残っている | `test_vllm_cells.py::test_warm_cache_is_captured_on_both_paths` |
| 26 | キャッシュを「入れた」で終わらせると、**キー不一致で黙ってミス**していても気づけない（373 秒が戻るだけ） | 「ファイルがある」ではなく**エントリの増分**（`new_entries` / `reused_pct`）で判定する | `test_vllm_cells.py::test_triton_measurement_counts_new_entries` |
| 27 | **存在しない Dataset を `dataset_sources` に入れても実行は止まらない**。`/kaggle/input/<slug>` が無いだけなので、`KC_WARM_CACHE=1` が「なし」と出して続行し、`cache` 実験が **baseline とまったく同じ数字**を出す（＝偽陰性）。実測 2026-11-01: `vllm-warm-cache` は未作成だった | `KC_WARM_CACHE` を 3 値にし、実験は `require`（無ければ `RuntimeError`）。`--no-run` で push すると `not valid dataset sources` の警告が出るので、Quick Save を警告チェックに使う | `test_vllm_cells.py::test_warm_cache_require_fails_instead_of_silently_degrading` / `test_vllm_playground.py::test_cache_experiment_requires_the_dataset` |
| 28 | **起動待ちが無言だと「遅い」と「固まった」を区別できない**。実測 2026-11-01 exl3 v3: 1800 秒待って `chat_server.log` が **0 バイト**（`--no-progress` で進捗を切り、`print` は完走時の 1 行だけ）。手がかりゼロなので次の手が「もう 1 回回す」になり、1 往復 30 分を捨てる | 待機中は 30 秒ごとに `kc_sample`（`read_bytes` が主役）で観測し、`load_watch` に残す。`read_mib` が止まっていれば停止、増えていれば遅いだけ | `test_runtime.py::test_watch` / `test_exl3_build.py`（呼び出しと `stop()` の 2 点） |
| 29 | **1 回の成功を「動く」と読まない**。ExLlamaV3 は同じモデル・同じ wheel・同じ T4 x2 で v2 が 229 秒、v3 が 1800 秒超（起動せず）。1 回の成功は再現性の証拠にならない | 台帳には「T4 で動くのは確実／安定性は未確定」と分けて書く。**同じ条件で 2 回通るまで既定に昇格しない** | テストにしない（実行の履歴。`notes/engine-rollout.md`） |
| 30 | **tar はあるが中身が薄い**ケースを「キャッシュは効かない」と誤読する。v15 の tar は 30 エントリで GDN の chunked カーネルが無い（profile run に未到達）。復元すると一部だけ当たって「少し効いた」ように見える。**条件を推測で書くと逆に正当な tar を弾く**: 最初 `chunk_o` / `chunk_delta_h` と書いたが、実測の名前は `chunk_fwd_kernel_o` / `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` で部分一致しない | `require` では**復元したファイル名**に実測名が無ければ落とす。件数で判定しない（Triton の版で変わり、しきい値を置くと正当な tar を弾く）。末尾 `_blockdim64` は版で変わりうるので前側だけを見る。内訳は `timings.warm_cache_restore` にも残す | `test_vllm_cells.py::test_warm_cache_require_rejects_a_tar_without_the_profile_run_kernels` / `..._accepts_a_complete_tar_and_records_the_count` / `..._guard_uses_the_measured_kernel_names` |
| 31 | **「遅い」失敗を「壊れた」失敗として扱う**。`baseline` は 2 試行とも重みロードの途中で切られた（900 秒 / 806 秒）。重み 17.41 GiB を 20 MiB/s で読むと 900 秒で、そのあとに profile run 373 秒が乗る。速い回（103 MiB/s）に合わせた予算は遅い回で必ず落ちる | 1 試行の上限を総予算と同値にし、遅い回に予算を全部渡す。「非対応」は即座に落ちるので予算を食わない | `test_vllm_cells.py::test_attempt_timeout_covers_the_whole_startup_budget` |
| 32 | **予算を「重みロード」だけで決めてしまう**。2700 秒に上げた v3 は全 62 層を読み終えた **t=2890s** のあと t=2972s で打ち切られた。重み 2,640 秒 + profile run 373 秒 = **3,013 秒が下限**で、2700 では 300 秒足りない | 予算は「重み + profile run + 余裕」で決める。下限を 3600 としてテストで固定（実測 3,013 秒の 1.2 倍） | `test_vllm_cells.py::test_attempt_timeout_covers_the_whole_startup_budget`（`b >= 3600`） |
| 33 | **Kaggle Secrets の「未紐付け」は `ConnectionError` として出る**。`get_secret()` が `Connection error trying to communicate with service.` を投げるので、ネットワーク障害を疑って時間を溶かす。実際は「このカーネルに Secret が 1 つも付いていない」で、アカウントへの登録だけでは読めない | 例外を握りつぶさず**理由ごと返す**（`kc_gh_token_detail`）。`""` の 1 語では「登録漏れ」と「紐付け漏れ」を分けられない | `test_runtime.py::test_gist`（理由が空にならないこと）+ `cpu-tests/gist`（実機で `token_detail` を確認） |
| 34 | **Dataset に上げた `.tar` は展開されて保持される**。195 MB の tar を `datasets create` したら、マウント側は `root/.triton/cache/<hash>/...` の木になり `*.tar` は 1 本も無かった。`*.tar` だけを探す復元は「Dataset が空」と誤判定し、`require` が**正当な Dataset を落とす** | tar と展開済みの木の両方を受ける（`_kc_wc_tree` が `.triton/cache` を探す）。`timings.warm_cache_restore.source` にどちらから復元したかを残す | `test_vllm_cells.py::test_warm_cache_restores_an_expanded_dataset` |


## 既に別ファイルで固定済みのもの

| 症状 | テスト |
|---|---|
| `pip uninstall torch` では残骸が消えない（`Config() got an unexpected keyword 'deprecated'`） | `test_gpu_pitfalls.py::test_torch_purge_covers_siblings_without_touching_torchmetrics`（範囲）+ `test_vllm_cells.py::test_install_purges_torch_before_pip`（順序） |
| 親プロセスで `import torch` すると新旧の torch が混ざる | `test_serve_script_guards.py::test_stage_env_does_not_import_torch_in_parent` |
| エンジンが死んでも 40 分待ち続ける（実測 46 分の空振り） | `test_serve_script_guards.py::test_engine_pattern_matches_its_launch_command`（6 エンジン） |
| 起動待ちログが末尾 1 行だけで 360 秒の正体が消える | `test_vllm_cells.py::test_startup_wait_reports_last_milestone` |
| CPU プローブの `ok` 判定に「弾かれて正解」を混ぜる | `test_cpu_probes.py::test_vllm_flag_verdict_excludes_designed_negatives` |

## この棚卸しで見つかった実バグ 2 件（修正済み）

### (a) `if used and sum(used) < 4000` は **空リストで素通り**する

`04_warmup.py` の CPU フォールバック検出。`nvidia-smi` が存在しても
デバイスが無い場合（ドライバ未接続）は **stdout が空**になり、`used == []`。
`if used and ...` は False になるので **fail せずに通過**していた。
＝ まさに「黙って CPU 推論」を検出できない穴。

- `if not used:` を追加し、取得できない時点で fail
- `int(x.split(",")[0])` は `N/A` で ValueError になるので `.isdigit()` で選別

### (b) `/dev/shm` は「記録するだけ」で判定していなかった

`SHM_TOTAL_MIB < 512` のとき警告を出すだけだった。必要量は **TP に比例**する
（1 本 240 MiB × 2 倍の余裕）。空きで判定し、足りなければ `remount` を試し、
なお足りなければ **起動前に落とす**（起動後に詰まると待機側が 40 分待つ）。

**追記（2026-11-01）**: `/dev/shm` の判定は入ったが、起動ログに出る
`shm_broadcast.py: No available shared memory broadcast block found in 60 seconds`
は**容量不足のサインではない**ことが分かった。v10 の実測では
`/dev/shm 14,336 MiB（空き 14,336 MiB）`＝**まったく足りている**状態で、
それでも 60 秒おきに 5 回出た。この警告は vLLM 自身が本文で言っている通り
「**どこかのプロセスが時間のかかる作業中**」の意味で、EngineCore が worker の
返事を待っているだけ（実測: 同時刻の GPU util は 0%、VRAM は漸増）。
→ **容量の判定（fail-fast）は残す**が、この警告を容量の証拠として読まない。

**確定（v14・2026-11-01）**: 381 秒（v10）/ 373 秒（v14）の正体は
**memory profiling（profile run）中の GDN Triton JIT**。v14 は `--enforce-eager` で
走った回で、compile が無いのに同じ 373 秒が出た。直後に

```
[gpu_worker.py:759] Available KV cache memory: 3.45 GiB
[qwen_triton_warmup.py:296] Warming up Qwen GDN Triton kernels for model_type=qwen3_5_text
```

が並ぶ。`/dev/shm` は 29 MiB しか使っておらず（`ram_trace.txt`）、容量の問題ではない。
起動時間の **54%** がここ。詳細は `notes/gpu-run-sheet.md` v14 節。

**ただし読み違えないこと（2026-11-01 追記）**: 上の 2 行が**隣り合っている**ため
「`Warming up ...` が 373 秒」と読んでしまいやすいが、時刻で割ると
`Available KV cache memory`（03:58:38）→ `Warming up`（03:58:38）→
`init engine ... took 376.24s`（03:58:41）で、**warmup は 3 秒**。373 秒は
`determine_available_memory` の profile run 側。`qwen_triton_warmup` は
「本番の生成で JIT しないための予熱」で、起動時間の短縮には寄与しない。
ハングでないことは `ram_trace.txt` のワーカー RSS が 125 秒で 1.37 GiB
（約 11 MiB/s）増え続けていることで裏が取れる。

## テストにしない（できない）もの

| 項目 | 理由 |
|---|---|
| 実効 t/s（31.7 / 14.2 / 13.7） | GPU 実行でしか出ない。ログ側で記録する |
| MTP の実効可否 | target の `config.json` 依存。CPU プローブの `NotImplementedError` は**非対応の証拠にならない**（`notes/cpu-test-results.md`） |
| Dataset のマウント有無 | Kaggle 側の状態。`kc_find_dataset` の分岐はテスト済み |
| 上流の並列 blocklist（Ollama PR 未マージ） | こちらのコード外。設定値と理由だけ固定 |

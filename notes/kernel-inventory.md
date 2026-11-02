# カーネル棚卸し（2026-10-30）

移行 1 の成果物。**カーネル id は原則として変えない**（Secrets・dataset_sources・
実行履歴が id に紐づく）。ここでは「どれが現役で、どのビルダーが生成し、どこまで
検証されて」いるかだけを固定する。

> **改名（2026-10-30）**: llama-server 直起動の一族だけ `step2-` を落とした。
> `step2-direct-llama-server-{tunnel}` → `llama-server-{tunnel}`、
> `…-script` → `llama-server-{tunnel}-script`。ディレクトリも
> `kaggle/step2-direct/` → `kaggle/llama-server/`（cells + cloudflared 出力）、
> `kaggle/step2-direct-ngrok/` → `kaggle/llama-server-ngrok/`、
> `kaggle/step2-direct-script/` → `kaggle/llama-server-script/`。
> ビルダーも `build_step2_direct{,_script}.py` → `build_llama_server{,_script}.py`、
> ノブは `STEP2D_*` → `LLAMA_*`。**Kaggle 側では新規カーネル**になる（旧 id は
> 残る。GPU 実測前だったので失うものは無い）。

機械可読な正は `kbuild/spec.py`（`python3 -m cli list` が同じ内容を出す）。
このファイルはその補足（注意点・Secrets・検証状況）を書く。

- 現役: 常用する。リファクタ後も同じ id で push する。
- 準現役: 必要時に使う（測定・CPU 検証・代替経路）。壊さないが優先度は低い。
- 退役: 参照しない。ビルダーは残すが再生成・push の対象外。

全体は **51 件**（notebook 14 / script 37）。うち現役 12・準現役 31・退役 8、
ビルダー持ち 30 件。`.ipynb` は 1 冊も削除していない（script 版は並走）。

> **数え方は台帳から引く**（2026-11-02 に 20/31 → 14/37 の食い違いを直した）。
> `python3 -m cli list` と同じ内容なので、件数を手で足さない。

> **ngrok 廃止（2026-11-01）**: トンネルを cloudflared の quick tunnel + ビーコンに
> 一本化したため、ngrok 系 6 本を retired にした（`notes/url-publishing.md`）。
> **生成物（`main.py` / `.ipynb`）は履歴として残してある。** 骨格が
> `STEP2_TUNNEL=ngrok` を弾くようになったので**再生成はできない**。
> 副作用として **llama-server は GPU カーネルを失った**（あの系統は ngrok 版
> しか無かった）。コードは残っているので台帳に 1 行足せば戻る。

## 一覧

| 状態 | カーネル id | ローカル | 生成元 | 形式 / GPU | datasets |
|---|---|---|---|---|---|
| 現役 | `step1-orcarouter-qwen38-27b-dl` | `kaggle/step1/` | `build_step1.py` | notebook / CPU | 0 |
| 現役 | `step2-orcarouter-qwen38-27b-cloudflared` | `kaggle/step2/` | `build_step2.py`（既定） | notebook / GPU | 3 |
| 退役 | `step2-orcarouter-qwen38-27b-ngrok-fixed` | `kaggle/step2-ngrok/` | 生成不可（ngrok 廃止） | notebook / GPU | 2 |
| 現役 | `prepare-vllm-dataset` | `kaggle/step2-vllm/` | `build_vllm_dataset.py` | notebook / CPU | 0 |
| 現役 | `step2-vllm-qwen38-27b-cloudflared` | `kaggle/step2-vllm-kernel/cloudflared/` | `build_step2_vllm.py`（既定） | notebook / GPU | 3 |
| 退役 | `step2-vllm-qwen38-27b-ngrok` | `kaggle/step2-vllm-kernel/ngrok/` | 生成不可（ngrok 廃止） | notebook / GPU | 2 |
| 現役 | `keepalive-test` | `kaggle/step2/`（`kernel-metadata-keepalive-test.json`） | `build_keepalive_test.py` | notebook / CPU | 0 |
| 現役 | `exl3-qwen38-gpu-check` | `kaggle/exl3-gpu-check/` | なし（`main.py` 直接編集） | script / GPU | 1 |
| 現役 | `exl3-dataset-upload` | `kaggle/exl3-dataset-upload/` | なし（直接編集） | script / CPU | 0 |
| 現役 | `exl3-serve` | `kaggle/exl3-serve-kernel/serve/` | `kaggle/build_exl3_serve.py` | script / GPU | 2 |
| 現役 | `exl3-serve-dryrun` | `kaggle/exl3-serve-kernel/dryrun/` | 同上（mock） | script / CPU | 1 |
| 現役 | `exl3-serve-keepalive` | `kaggle/exl3-serve-kernel/keepalive/` | 同上（keepalive） | script / CPU | 0 |
| 現役 | `llama-server-cpu-check` | `kaggle/cpu-check/` | なし（直接編集） | script / CPU | 2 |
| 準現役 | `step2-lmdeploy-qwen38-27b-cloudflared` | `kaggle/lmdeploy-serve/` | `build_engine_serve.py` | script / GPU | 2 |
| 準現役 | `step2-sglang-qwen38-27b-cloudflared` | `kaggle/sglang-serve/` | `build_engine_serve.py` | script / GPU | 2 |
| 退役 | `llama-server-ngrok` | `kaggle/llama-server-ngrok/` | 生成不可（ngrok 廃止） | notebook / GPU | 2 |
| 準現役 | `step2-vllm-dryrun` | `kaggle/step2-vllm-dryrun/` | `build_vllm_dryrun.py` | notebook / CPU | 2 |
| 準現役 | `bench-ctx-vs-speed-orcarouter-qwen3-8-27b` | `kaggle/bench/` | `build_bench.py` | notebook / GPU | 0 |
| 準現役 | `bench-mtp-spec-decoding-orcarouter-qwen3-8-27b` | `kaggle/bench_mtp/` | `build_bench_mtp.py` | notebook / GPU | 0 |
| 準現役 | `smoke-orcarouter-qwen38-27b` | `kaggle/smoke/` | `build_smoke.py` | notebook / GPU | 0 |
| 準現役 | `step2-vllm-qwen38-27b-cloudflared`（smoke・id は通常版と同じ） | `kaggle/step2-vllm-kernel/smoke-cloudflared/` | `build_step2_vllm.py --smoke`（`KC_SMOKE=1` でも可） | notebook / GPU | 3 |
| 準現役 | `exllamav2-cpu-preflight` | `kaggle/exllama-cpu-preflight/` | なし（直接編集） | script / CPU | 0 |
| 退役 | `capture-ollama-patch-v2` | `kaggle/capture_patch/` | `build_capture_patch.py` | notebook / CPU | 0 |
| 退役 | `keepalive-test-1` | `kaggle/step2-ka-test/` | なし（旧版が残存） | notebook / CPU | 0 |

`kaggle/archive/` の 46 スナップショット（json+ipynb）は上表に含めない。
削除せず、検索用に残す。

### CPU プローブ（`enable_gpu: false`・**GPU クォータ 0 秒**）

「GPU で賭ける前に切り分ける」ための計器。台帳では `gpu=False`、
`tests/test_cpu_probes.py` が `enable_gpu` の取り違えを検出する
（true にすると無料のつもりがクォータを食う）。

| 状態 | カーネル | ディレクトリ | 何を測るか |
|---|---|---|---|
| 現役 | `vllm-flag-probe` | `kaggle/cpu-tests/vllm-flags` | `vllm serve` のフラグ受理（対照実験つき） |
| 現役 | `tiny-llama-probe` | `kaggle/cpu-tests/tiny-llama` | llama-server の小モデル起動 |
| 現役 | `tiny-ollama-probe` | `kaggle/cpu-tests/tiny-ollama` | Ollama の小モデル起動 |
| 現役 | `vllm-install-bench` | `kaggle/cpu-tests/install-bench` | pip / uv / wheelhouse の install 比較 |
| 現役 | `vllm-io-bench` | `kaggle/cpu-tests/io-bench` | GCSFUSE の速度・並列スケーリング・`drop_caches` の可否 |
| 現役 | `sglang-install-probe` | `kaggle/cpu-tests/sglang-probe` | SGLang が cp313 で入るか（wheel の有無は pip に判定させる） |
| 現役 | `vllm-whl-probe` | `kaggle/whl-probe` | wheel の解決 |
| 現役 | `ci-cpu` | `kaggle/ci-cpu` | 生成物の一致検査 |

結果は `notes/cpu-test-results.md`。**CPU で分からないことは分からないと書く**
（`gpu_unknown` / `sm75_untested` を `result.json` に残す）。

### script 版（2026-10-30 追加・notebook と並走）

**notebook は消さず、別 id で script を追加**したもの。同じビルダーが両方を出すので
セル列の真実源は 1 つ。id が別なので Secrets / dataset_sources は独立（notebook 側は無傷）。

| 状態 | カーネル id | ローカル | 生成元 | GPU |
|---|---|---|---|---|
| 準現役 | `step2-orcarouter-qwen38-27b-cloudflared-script` | `kaggle/step2-script/cloudflared/` | `build_step2_script.py` `STEP2_TUNNEL=cloudflared` | ✓ |
| 退役 | `step2-orcarouter-qwen38-27b-ngrok-script` | `kaggle/step2-script/ngrok/` | 生成不可（ngrok 廃止） | ✓ |
| 準現役 | `step2-vllm-qwen38-27b-cloudflared-script` | `kaggle/step2-vllm-script/cloudflared/` | `build_step2_vllm_script.py` `STEP2_TUNNEL=cloudflared` | ✓ |
| 退役 | `step2-vllm-qwen38-27b-ngrok-script` | `kaggle/step2-vllm-script/ngrok/` | 生成不可（ngrok 廃止） | ✓ |
| 退役 | `llama-server-ngrok-script` | `kaggle/llama-server-script/ngrok/` | 生成不可（ngrok 廃止） | ✓ |
| 準現役 | `prepare-vllm-dataset-script` | `kaggle/step2-vllm/script/` | `build_batch_script.py` ← `build_vllm_dataset` | |
| 準現役 | `step1-orcarouter-qwen38-27b-dl-script` | `kaggle/step1/script/` | 同上 ← `build_step1` | |
| 準現役 | `keepalive-test-script` | `kaggle/step2/script/` | 同上 ← `build_keepalive_test` | |
| 準現役 | `step2-vllm-dryrun-script` | `kaggle/step2-vllm-dryrun/script/` | 同上 ← `build_vllm_dryrun` | |
| 準現役 | `smoke-orcarouter-qwen38-27b-script` | `kaggle/smoke/script/` | 同上 ← `build_smoke` | ✓ |
| 準現役 | `bench-ctx-vs-speed-orcarouter-qwen3-8-27b-script` | `kaggle/bench/script/` | 同上 ← `build_bench` | ✓ |
| 準現役 | `bench-mtp-orcarouter-qwen3-8-27b-script` | `kaggle/bench_mtp/script/` | 同上 ← `build_bench_mtp` | ✓ |

- 上 5 本は **骨格版**（`kbuild/serve_script.py`）。`stage_install()` と
  `launch(model_dir)` だけを渡し、proxy / tunnel / ビーコン / keepalive / 予算ループは骨格が持つ。
- 下 7 本は **セル連結版**。`build_batch_script.py` が元ビルダーの `cells` をそのまま
  1 ファイルへ連結する（`SERVE=False` の単発モード）。セルの中身には手を触れていない。
- `-script` の出力先は**親ディレクトリ配下の別名**（`kaggle/step2-script/`）にして、
  notebook の `.ipynb` と同じディレクトリに混ぜない。`/script/` は親の中の子ディレクトリ
  （`kaggle/step2/script/`）。どちらも `kernel-metadata.json` を持つので push は
  `-p <そのディレクトリ>` で完結する。
- 状態が `semi` なのは **GPU 実測がまだ 1 回も無い**ため。CPU 検証（生成・ruff・
  pytest・バイト一致・compileall）は通っている。

## 注意点

1. **`keepalive-test` と `keepalive-test-1` は別カーネル**。現役は
   `kaggle/step2/step2-keepalive-test.ipynb` + `kernel-metadata-keepalive-test.json`
   （id `keepalive-test`）。`kaggle/step2-ka-test/` は旧 `keepalive-test-1` で、
   どのビルダーも生成しない。混同しやすいので、移行時は `step2-ka-test/` を
   退役として扱い、`step2/` 側だけをビルダー対象に保つ。
2. **`llama-server` は GPU カーネルを持っていない**（2026-11-01）。あの系統は
   ngrok 版しか無く、ngrok 廃止で両方 retired にしたため。`LLAMA_TUNNEL=cloudflared`
   で `kaggle/llama-server/llama-server-cloudflared.ipynb` は生成できる
   （セル置き場 `kaggle/llama-server/cells/`）が、**台帳に id が無いので push
   できない**。戻すなら `kbuild/spec.py` に 1 行足す（新規 Kaggle カーネルになる）。
3. **`kaggle/exl3-serve-kernel/{serve,dryrun,keepalive}/` は生成物**。
   `kaggle/build_exl3_serve.py` から再生成でき、`.gitignore` 済み。
   `kaggle/exl3-serve/` に残るのは `chat_server.py`（真実源）だけ。
4. **ビルダーを持たない 4 本**（`exl3-gpu-check`, `exl3-dataset-upload`,
   `cpu-check`, `exllama-cpu-preflight`）は `main.py` が真実源。kbuild 移行の
   対象外だが、`runtime/` の probe / net を使えるようになったら寄せる。
5. **smoke 版は id を共有する別成果物**。vLLM の smoke は
   `kaggle/step2-vllm-kernel/smoke-<tunnel>/`（15 セル）に書かれ、通常版
   `cloudflared/`（24 セル）とはパスもセル数も別。`cli push --smoke` は
   smoke 側を staging するので、**手元の通常版は上書きされない**。
   ただし Kaggle 上のカーネルは同じ id なので、smoke を push すると中身が
   smoke 版に入れ替わる。実稼働前に `cli push <name> --yes`（smoke 無し）で
   戻すこと。smoke ディレクトリは再生成可能な一時成果物だが、
   台帳の `smoke_out` に入っているので消すと検証が落ちる
   （`check_generated.py` の対象は台帳から導出される）。
6. **トンネルの既定は 3 ビルダーとも cloudflared**（2026-11-01 に
   `build_llama_server.py` の既定 ngrok を直した）。`STEP2_TUNNEL` / `LLAMA_TUNNEL`
   に cloudflared 以外を渡すと、ビルダーは生成前に落ちる。
7. **script 版は notebook と同じディレクトリに置かない**。`-script` は
   `kaggle/step2-script/`、セル連結版は `kaggle/step2/script/` のように分ける。
   `kaggle/step2/` には `kernel-metadata.json` と `kernel-metadata-keepalive-test.json`
   が同居していて、`kaggle kernels push -p kaggle/step2` を直に叩くと
   **どちらを読むか決まらない**。`cli push` は一時ディレクトリで
   `kernel-metadata.json` に揃えてから push するので安全（`cli/push.py:_stage`）。
8. **新規作成では title が 2 つの制約を満たす必要がある**（既存 id は対象外）。
   - `slugify(title)` が id と一致しないと **409 Conflict**。
     `Smoke: orcarouter Qwen3.8-27B on T4x2` は `qwen3-8` になるため
     `smoke-orcarouter-qwen38-27b` と衝突した → `Smoke orcarouter Qwen38 27B` に修正。
   - title が **50 文字を超えると 400 Bad Request**（slug も同じ長さになるので
     id 自体が作れない）。`bench-mtp-spec-decoding-orcarouter-qwen3-8-27b-script`
     （53）→ `bench-mtp-orcarouter-qwen3-8-27b-script`（37）に短縮した。
   どちらも `tests/test_spec.py` が見張っている
   （`test_new_kernel_title_slugifies_to_id` / `test_title_fits_kaggle_limit`）。
   移行前からの `step1-orcarouter-qwen38-27b-dl` だけは例外として除外してある
   （既存カーネルの title を変えると slug が変わり push が通らなくなるため）。
8. **script の `main.py` は 150〜480 KB になる**（`llama-chat.html` /
   `auth_proxy.py` / `kcconfig.py` を base64 で埋め込むため。骨格版 3 種が
   440〜480 KB、セル連結版が 12〜350 KB）。差分レビューは骨格部分
   （`kbuild/serve_script.py`）とセル（`kaggle/<kernel>/cells/`）で行い、
   生成物は「一致しているか」だけを見る。

## 新しい kernel id を作るときのチェックリスト

1. **id / title / slug を一致させる**。`title` が id の slug に解決できないと
   push が 409 Conflict になる（`kbuild/metadata.py` が生成）。
2. **Secrets はカーネル単位**。上の台帳から必要なものだけ UI で登録する。
   追加したら `kbuild/secrets.py` と `tests/test_secrets.py` も更新する。
3. **`dataset_sources` を確認**。Dataset を版ごとに入れ替える場合は、
   参照している全カーネル（`kernels list -m --dataset`）を洗う。
4. **`kernel_type` と `code_file`**。script カーネルは 1 ファイルのみ。
   複数ファイルは base64 で埋め込む（`kbuild.embed`）。
5. **accel / internet**。GPU を使うなら `enable_gpu`、トンネルには
   `enable_internet` が必須（どちらも `metadata.base` の引数）。
6. **初回 push は GPU クォータを消費する**。残量を
   `python3 kaggle-quota.py` で確認し、依頼があるときだけ push する。
7. 生成物は `check_generated.py` / `tests/test_exl3_build.py` の対象に加える。
   **notebook も script も台帳から自動で拾う**ので、`kbuild/spec.py` に
   `builder` / `smoke_out` を書けば一覧更新は要らない（2026-11-02 に
   `TARGETS` の手書きを廃止した）。

## Secrets 台帳

Secrets は**カーネル単位**で UI から登録する（コードからは登録できない）。
コード中の参照は `kbuild/secrets.py` に台帳化し、`tests/test_secrets.py` が
`get_secret("...")` / `secret("...")` のリテラルと突き合わせる。

| カーネル | 必須 Secret |
|---|---|
| `step1-orcarouter-qwen38-27b-dl` | `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| `step2-…-cloudflared` / `step2-vllm-…-cloudflared` | なし（2026-11-02 に Gist を廃止） |
| `prepare-vllm-dataset` | `HF_TOKEN` |
| `exl3-serve` / `-dryrun` / `-keepalive` | `EXL3_API_USER`, `EXL3_API_PASS` |
| `exl3-dataset-upload` | `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| **`…-script`（骨格版 3 本）** | `API_USER`, `API_PASS`（台帳は `kbuild/serve_script.py`） |
| **`…/script`（セル連結版 7 本）** | 元の notebook 版と同じ（`step1` は `KAGGLE_*`、`prepare-vllm-dataset` は `HF_TOKEN`、他はなし） |
| `keepalive-test` / `ci-cpu` / その他の CPU 検証 | なし |

`EXL3_API_PASS` が未設定でも乱数で起動するが、`result.json` には値を残さない。

> script 骨格が実装するトンネルは **cloudflared だけ**（2026-11-01 に ngrok を
> 落とした）。`NGROK_TOKEN` / `NGROK_DOMAIN` は台帳からも消してある。
> **どの配信カーネルも Secrets 無しで起動できる**（URL はビーコンが運ぶ）。

## 生成一致チェックの現状

「commit 済みの生成物 = ビルダーの再出力」をバイト比較する。2 本立て:

### notebook — `kaggle/check_generated.py`（10 冊）

- `step1` / `step2`（cloudflared）/ `smoke` / `bench` / `bench_mtp` /
  `step2-keepalive-test`
- `prepare-vllm-dataset` / `step2-vllm-dryrun` / `step2-vllm-qwen38-27b-cloudflared`
- `step2-vllm-kernel/smoke-cloudflared/…`（smoke 版・15 セル。`KC_SMOKE=1`）

退役した ngrok 版は対象外（再生成できないので比較する相手が無い）。

notebook 側の対象表は**台帳から導出**する（`check_generated.py` の `targets()`）。
`kind == "notebook"` かつ `builder` 持ちを引き、`smoke_out` を持つカーネルは
`KC_SMOKE=1` でもう一度走らせて smoke 側も突き合わせる。2026-11-02 まで
`TARGETS` に手で 10 行書いていたが、台帳に `smoke` 欄を足して廃止した。

### script — `kaggle/check_generated.py`（44 ファイル）

`kbuild/spec.py` から `kind == "script"` かつ `builder` 持ちを引くので、
**script カーネルを増やしても一覧の更新は要らない**。同じビルダーが複数の出力を
持つ場合はビルダー単位で 1 回だけ走らせる。

`smoke_out` を持つカーネルは、同じビルダーを `KC_SMOKE=1` でもう一度走らせて
smoke 側（`smoke-<tunnel>/`）も突き合わせる。通常版を smoke で上書きしていないことと、
smoke 版の再生成漏れを同時に見る。

- 骨格版 5 本（`-script`）＋ その smoke 5 本 ＋ セル連結版 7 本
- `exl3-serve-kernel/{serve,dryrun,keepalive}` / `lmdeploy-serve` / `sglang-serve` /
  `ci-cpu`

### 共通部

どちらも「一時ディレクトリへ写す → ビルダーを走らせる → バイト比較」で、
その部分は `kaggle/_regen.py`（`stage` / `run` / `label` / `diff` / `report`）。
**写した生成物を先に消す**ので、ビルダーが書き損ねたときに古いファイルと
一致して通ることはない。

### 限界（正直に書いておく）

`check_generated.py` の比較相手は **ディスク上のファイル**（`ROOT/<rel>`）で、
git の index ではない。したがって:

- commit 対象の script（`-script` / `/script/`）… commit 済みと比較するのと同じ。
  生成し忘れた `main.py` が push されるのを防げる。
- `.gitignore` 済みの script（exl3 / lmdeploy / sglang）… 直近の `cli build` が
  書いたファイルとの比較になる。「ビルダーを変えたのに再生成していない」は
  検出できるが、「git に載っていない古い状態」は原理的に検出できない。
  こちらは `tests/test_exl3_build.py` の sha256 決定性で担保する。

対象外:

- ビルダー無しの script 4 本（`exl3-gpu-check` / `exl3-dataset-upload` /
  `cpu-check` / `exllama-cpu-preflight`）… 生成物ではなくソースそのもの。
- 退役 2 本 … 再生成の対象外。

## 次の段への引き継ぎ（2026-10-30 時点）

移行 1・2 と第 3・4 次計画は完了。残っているのは **GPU 実測**と**方針決定**:

1. `…-script` を `--smoke` で 1 回ずつ GPU 実測（Ollama / vLLM / llama-server）。
   smoke 版は `smoke-<tunnel>/` に別書きされ、`cli push --smoke` がそちらを押す。
   手順は `notes/gpu-smoke-plan.md`。
2. `kernel_type` を同一 id で変えられるかは未検証（Phase 0-a）。
   不可なら「notebook を消す」段は新 id 前提になる。
3. script 生成物 15 ディレクトリ（約 6.4 MB）が**未追跡**。commit するか
   `.gitignore` に寄せるか（後者なら `check_generated.py` の対象から外す）。

# Kaggle GPU で Qwen3.8-27B-Uncensored を動かす

T4 x2（16 GB × 2）の Kaggle カーネルで 27B 級モデルを動かし、
cloudflared の quick tunnel で OpenAI 互換 API + チャット UI として公開する。
URL は毎回変わるので、**ビーコン**（掲示板カーネルのコードに URL を書かせて
`python3 -m cli beacon` で読む）で受け渡す。ngrok は 2026-11-01 に廃止した
（`notes/url-publishing.md`）。
ローカル Mac（RAM 16 GiB）では実用速度が出ないため、推論はすべて Kaggle 側。

- 推論エンジン 6 系統: **Ollama（GGUF）/ llama-server 直起動 / vLLM（AWQ+MTP）/ ExLlamaV3（EXL3）/
  LMDeploy / SGLang**（後ろ 2 本は比較用・GPU 実測待ち）
- 生成物はすべて `kaggle/build_*.py` から再生成できる（手で notebook を編集しない）
- カーネル台帳は `kbuild/spec.py`、操作は `python3 -m cli`
- **notebook 型と script 型が並走**している（同じビルダーから両方出せる。移行状況は
  「notebook と script（二本立て）」）。script の配信カーネルは
  `kbuild/serve_script.py` **1 本の骨格**に集約済み

## 結論から

| 用途 | 選択 | 理由 |
|---|---|---|
| 常用（速度・安定） | **vLLM + AWQ 4bit + MTP** | T4 x2 で実測最速。TP=2 が効く |
| 手軽さ優先 | Ollama（GGUF Q4_K_M） | セットアップが最短。vision も使える |
| 配管だけ試す | `exl3-serve-dryrun` / `step2-vllm-dryrun`（CPU） | GPU クォータを消費しない |
| 量子化形式の比較 | ExLlamaV3（EXL3 4bpw） | vLLM と同程度。TP 非対応・vision 非対応 |
| 速度の比較（検証中） | LMDeploy（Turbomind + AWQ） | T4 の AWQ は公式対応。単発に強い |
| 並列・長文（検証中） | SGLang（triton backend） | prefix 再利用が強い。T4 は博打 |

GPU クォータは 30h/週。**push は即実行**なので、検証は CPU 側で済ませてから
GPU を使う（`python3 -m cli check`）。

## クイックスタート

```bash
# 0) 生成（GPU 不要・クォータ不要）。既定は build のみ
python3 -m cli build

# 1) 台帳を見る
python3 -m cli list

# 2) 残量を見る
python3 -m cli quota        # または python3 kaggle-quota.py

# 3) push（即実行・クォータ消費）。--yes が必須
python3 -m cli push step2-vllm-qwen38-27b-cloudflared --yes

# 3a) script 版を push する（同じエンジン・別 id）
python3 -m cli push step2-vllm-qwen38-27b-cloudflared-script --yes

# 3b) 起動と会話だけ試す（トンネル / keepalive / Gist を省略して GPU 時間を節約）
python3 -m cli push step2-lmdeploy-qwen38-27b-cloudflared --smoke --yes

# 4) 実行結果の内訳を比べる（GPU 不要）
#    未計測は `-`。0 で埋めないので「一瞬で終わった」と誤読しない。
python3 -m cli timings /tmp/out1/result.json /tmp/out2/result.json

# 3c) 実行せずカーネルだけ作る（Quick Save 相当・GPU を消費しない）
python3 -m cli push llama-server-cpu-check --no-run --yes

# 4) 出力（result.json / url.txt）を取得
kaggle kernels output <id> -p out/
```

push 前に残量が表示される。`--yes` を付けない限り push しない。
`--smoke` は install → 起動 → `/v1/chat/completions` → 停止だけを行う
（計画と合否基準: `notes/gpu-smoke-plan.md`）。script 版も `--smoke` に対応し、
通常版とは**別ディレクトリ**（`smoke-<tunnel>/`）へ書くので通常版を壊さない。

`--no-run` は**バージョンを保存するだけで 1 セルも実行しない**（Quick Save 相当）。
クォータが尽きていても id と設定（GPU / Dataset / Internet）だけ先に確保できる。
`kaggle` CLI **2.x のフラグ**なので、リポジトリ内の `.venv-kaggle-cli/` を使う
（無ければ作り方を表示して止まる。`KC_KAGGLE` で差し替え可）。当日の手順は
`notes/gpu-run-sheet.md`。

### ローカルから対話する

```bash
python3 -m cli ask "こんにちは"                 # Gist（url.txt）から URL を解決
python3 -m cli ask --url https://xxx.trycloudflare.com "..."
```

ダブルクリック起動は `run-ai.command` / `run-llama-server.command` /
`run-uzu-server.command` / `run-kaggle-chat.command`。
設定は `config.env`（個人値は `config.local.env`、gitignore）。

## 構成

```
# ---- リポジトリ直下 ----
README.md / ruff.toml / config.env   # 設定（個人値は config.local.env、gitignore）
run-{ai,kaggle-chat,llama-server,uzu-server}.command   # ダブルクリック起動
llama-chat.html             # カーネルに埋め込むチャット UI の原本（build_* が読む）
ask.py kcconfig.py kaggle-quota.py   # 後方互換シム（実体は cli/）
opencode.json / opencode-llama-local.json  # opencode 用の設定
models/                     # ローカル GGUF の置き場（gitignore・現在は空）
datasets/                   # Dataset のアップロード用ステージング（gitignore）
  gguf/ orcarouter/         #   dataset-metadata.json のみ（重みは 2026-10-30 に削除）
  orcarouter-patch/         #   ollama バイナリ + lib（assemble_patch_dataset.py が出力）
  cloudflared/              #   cloudflared-linux-amd64
archive/                    # 生成物・ログの退避先（gitignore。中身は下記）

# ---- kaggle/ ----
kaggle/
  build_*.py                  # ビルダー（18 本。notebook / script を生成）
  serve-cells/                # notebook 配信の共有セル（proxy / tunnel / gist / verify …）
  step1/ step2/               # 生成物（Ollama 系 notebook）
  step2-ngrok/                # 退役（2026-11-01）。履歴として残置、再生成不可
  llama-server/               # llama-server 直起動（cells/ + cloudflared 版の出力先）
  llama-server-ngrok/         # 退役（同上）
  step2-vllm/                 # Dataset 作成 notebook
  step2-vllm-kernel/{cloudflared,smoke-cloudflared}/  # vLLM 本体
  step2-vllm-kernel/ngrok/    # 退役（同上）
  step2-vllm-dryrun/          # vLLM 配管テスト（CPU）
  # ---- script 版（notebook と並走。配信は kbuild/serve_script.py の骨格）----
  step2-script/cloudflared/          # Ollama（smoke-cloudflared/ は --smoke 版）
  step2-vllm-script/cloudflared/     # vLLM（同上）
  step2-script/ngrok/                # 退役（同上。smoke-ngrok/ も）
  step2-vllm-script/ngrok/           # 退役（同上）
  llama-server-script/ngrok/         # 退役（同上）
  # ---- script 版（配信しないカーネル。セル列をそのまま 1 ファイルへ）----
  {step1,step2,step2-vllm,step2-vllm-dryrun,smoke,bench,bench_mtp}/script/
  exl3-serve/                 # chat_server.py（EXL3 用 OpenAI 互換サーバの真実源）
  exl3-serve-kernel/{serve,dryrun,keepalive}/  # EXL3 生成物（script・gitignore）
  lmdeploy-serve/ sglang-serve/                # LMDeploy / SGLang 生成物（script・gitignore）
  build_engine_serve.py       # 上記 2 本のビルダー（引数だけ差し替え）
  exl3-gpu-check/ exl3-dataset-upload/         # ビルダー無し（main.py が真実源）
  cpu-check/ exllama-cpu-preflight/            # CPU 検証
  ci-cpu/                     # CI を Kaggle の Python 3.13 で回す
  check_notebooks.py          # notebook の一致検査
  check_scripts.py            # script の一致検査（台帳から対象を引く）
  _regen.py                   # 上記 2 本の共通部（写す・走らせる・比べる）
  archive/                    # 旧カーネルのスナップショット（参照しない。ルートの archive/ とは別）

kbuild/      # 梱包の共通層（spec / embed / metadata / pack_nb / pack_script /
             #   serve_notebook / serve_script / settings_cell / secrets）
runtime/     # カーネルに埋め込む真実源（net / proc / storage / keepalive / proxy / quota / engines）
cli/         # CLI（build / push / check / list / ask / quota）
tests/       # pytest（114 本）
lib/         # ローカルランチャー共通（launcher-common.zsh）
notes/       # 設計メモ・台帳・実測ログ（索引は notes/index.md）
```

`kcconfig.py` / `ask.py` / `kaggle-quota.py` は後方互換シム（実体は `cli/`）。

`archive/` は**追跡しない退避先**（`.gitignore` 済み）。ルート直下が散らからないよう、
一時プローブの出力とログだけをここへ集めてある。中身は消しても再生成できる。

| 置き場 | 元 | 中身 |
|---|---|---|
| `archive/probes/` | `out/` と `out.*/` | ビルダー検証時の一時出力 64 個（`out.dryrun7` など） |
| `archive/logs/` | ルートの `*.log` + `diagout/` | 実測ログ（`bench-*` / `cpu-probe10-*` / `offline-restart` など 9 本） |

生の計測データを残したいときは `notes/archive/`（追跡する）へ、
一度きりの捨て出力は `archive/` へ、と使い分ける。

`datasets/` は `kaggle datasets create/version -p datasets/<name>` で**アップロードする
ためのステージング**（実体は gitignore）。repo の外へは出していない — 同一ボリューム
なので移動しても空き容量は増えず、`-p` を相対パスで叩ける今の形の方が事故らない。

**2026-10-30 にローカルのモデル重みを削除した**（Kaggle は `dataset_sources` で
Dataset をマウントするだけで、ローカルのファイルは一切見ないため）。残したのは
再アップロード用の足場だけ:

| 残っているもの | 用途 |
|---|---|
| `datasets/gguf/dataset-metadata.json` | `qwen3-27b-uncensored-gguf` を作り直すとき |
| `datasets/orcarouter/dataset-metadata.json` | `my-ollama-orcarouter-qwen38-27b` を作り直すとき |
| `datasets/orcarouter-patch/`（1.4 GB） | patch Dataset の再アップロード（作るのに GPU が要るので残す） |
| `datasets/cloudflared/`（38 MB） | cloudflared Dataset の再アップロード |

重みは `ollama pull orcarouter/Qwen3.8-27B-Uncensored:latest`（2.6 分）で戻る。
**`run-llama-server.command` はローカルのモデルを要求するので、この状態では
「先に `ollama pull` してください」で止まる**（設計どおりの挙動）。

## カーネル台帳

id は**原則**変えない（Secrets / `dataset_sources` / 実行履歴が id に紐づく）。
例外は 2026-10-30 の `step2-direct-llama-server-*` → `llama-server-*` の改名だけで、
これは **Kaggle 上では新規カーネル**になる（旧 id はそのまま残るので、
不要になったら UI から消す。GPU 実測前だったので失うものは無い）。
全 **51 件**（現役 12 / 準現役 31 / 退役 8）、うちビルダー持ち 30 件。
全件は `python3 -m cli list`。主なもの:

| 状態 | カーネル | 種別 | 役割 |
|---|---|---|---|
| 現役 | `step1-orcarouter-qwen38-27b-dl` | notebook / CPU | GGUF を Dataset 化（初回のみ） |
| 現役 | `step2-orcarouter-qwen38-27b-cloudflared` | notebook / GPU | Ollama + cloudflared |
| 現役 | `step2-vllm-qwen38-27b-cloudflared` | notebook / GPU | vLLM + AWQ + MTP |
| 現役 | `prepare-vllm-dataset` | notebook / CPU | wheelhouse / AWQ の Dataset 化 |
| 現役 | `exl3-serve` | script / GPU | EXL3 配信（60 分で自動停止） |
| 現役 | `exl3-serve-dryrun` / `-keepalive` | script / CPU | 配管・keepalive の実測 |
| 現役 | `keepalive-test` | notebook / CPU | keepalive の延長と `/stop` |
| 現役 | `ci-cpu` | script / CPU | CI を Kaggle 3.13 で実行 |
| 準現役 | `step2-lmdeploy-…` / `step2-sglang-…` | script / GPU | vLLM との比較（GPU 実測待ち） |
| 準現役 | `…-script`（Ollama / vLLM / llama-server） | script / GPU | notebook 版と並走。骨格 `serve_script.py` |
| 準現役 | `step1` / `step2` / `dryrun` / `smoke` / `bench` ×2 の `-script` | script / CPU・GPU | 配信しないカーネルの script 版（セル連結） |
| 準現役 | `smoke-…`, `bench-…` ×2, `step2-vllm-dryrun`, `llama-server-…` | notebook | 測定・代替経路 |
| 退役 | `keepalive-test-1`, `capture-ollama-patch-v2` | — | 参照しない |
| 退役 | `…-ngrok` 系 6 本（2026-11-01） | notebook / script | ngrok 廃止に伴い停止。`main.py` / `.ipynb` は履歴として残置 |

Secrets は**カーネル id 単位**で UI から登録する。台帳は `kbuild/secrets.py`、
コード中の参照との一致は `tests/test_secrets.py` が検査する。

## notebook と script（二本立て）

Kaggle は `kernel_type` が `notebook`（`.ipynb`）と `script`（`main.py` 単体）の
2 種で、**`kernels push` はディレクトリ直下の `kernel-metadata.json` を読む**だけ。
つまり「どちらを push するか」は出力ディレクトリで決まる。

現状は **notebook を消さずに script を並走**させている（`.ipynb` は 1 冊も削除して
いない）。同じビルダーが両方を出すので、セル列の真実源は 1 つ。

| | notebook 版 | script 版 |
|---|---|---|
| 実体 | `.ipynb`（セル JSON） | `main.py` 1 ファイル（補助ファイルは base64） |
| 配信の骨格 | `kbuild/serve_notebook.py` の `Flow` | `kbuild/serve_script.py` 1 本 |
| エンジン差の置き場 | `kaggle/<kernel>/cells/*.py` | 同じセルを `textwrap.indent` で関数化 |
| トンネル | `kaggle/serve-cells/tunnel_cloudflared.py` | 骨格が持つ（`start_tunnel()` → cloudflared のみ） |
| レビュー | GitHub で差分が見える | 骨格版は読める／セル連結版はほぼ読める |
| 一致検査 | `kaggle/check_notebooks.py`（13 冊） | `kaggle/check_scripts.py`（18 カーネル / 46 ファイル） |
| 状態 | 常用（実績あり） | 生成・CPU 検証まで完了。**GPU 実測は未実施** |

script 側の作り方は 3 通り:

1. **骨格版（配信）** — `build_step2_script.py` / `build_step2_vllm_script.py` /
   `build_llama_server_script.py`。`stage_install()` と `launch(model_dir)` を
   渡すだけで、proxy / tunnel（cloudflared）/ gist / keepalive / 予算ループは
   骨格が持つ。`--smoke` は `smoke-cloudflared/` へ別書きする。
2. **セル連結版（配信しない）** — `build_batch_script.py` が `KC_BATCH`（元ビルダー）と
   `KC_BATCH_OUT`（出力先）を受け、`cells` をそのまま 1 ファイルへ連結する。
   `SERVE=False` の単発モードなので骨格は使わない（意味を変えないための割り切り）。
3. **旧 script（exl3 / lmdeploy / sglang）** — 骨格化前の実装。`.gitignore` 済みで、
   決定性は `tests/test_exl3_build.py` が sha256 で担保する。

移行の判断材料と残タスクは `notes/refactor-plan.md` の第 4 次計画。

## エンジン別の要点

### Ollama（GGUF Q4_K_M）
- `orcarouter/Qwen3.8-27B-Uncensored:latest`（16.52 GiB、vision 対応）
- `OLLAMA_LOAD_TIMEOUT` を延ばさないと 5 分で落ちる（既定 5m）
- Dataset 経由のバイナリは実行ビットが落ちる → `chmod +x` 必須

### llama-server 直起動
- Ollama 同梱バイナリ + patch Dataset から復元
- `--spec-type`（MTP）と `-c` を明示。KV は f16 固定に注意

### vLLM（AWQ + MTP）
- TP=2、`--enable-auto-tool-choice`、MTP は `--speculative-config`
- wheelhouse は Dataset から復元（JIT 回避）。`--help` の差分は起動ラダーで吸収
- 起動 10 分超は失敗とみなして段を落とす（`VLLM_ATTEMPT_TIMEOUT`）

### ExLlamaV3（EXL3 4bpw）
- `Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw`（15.9 GiB / 3 シャード）
- **TP 非対応・vision 非対応**。T4 1 枚で動かす
- CPU では `import` すら不可（JIT に nvcc が要る）。検証は GPU のみ
- 配信は `chat_server.py`（OpenAI 互換）+ `auth_proxy.py`（認証 / UI / keepalive）

### LMDeploy / SGLang（比較用・GPU 実測待ち）
- 既存の AWQ Dataset をそのまま使う（量子化資産を増やさない）。**MTP は使わない**。
- 生成: `python3 kaggle/build_engine_serve.py` → `kaggle/{lmdeploy,sglang}-serve/`。
- 起動引数は `runtime/engines/{lmdeploy,sglang}.py` の純関数。TP=2 固定。
- LMDeploy: `turbomind` / `--session-len 8192` から開始。T4（sm75）は公式対応。
- SGLang: `--attention-backend triton` / `--sampling-backend pytorch` /
  `--disable-cuda-graph` / `mem-fraction-static 0.75`。
  公式は sm75 以上（T4 含む）対応だが、sm75+ の FlashInfer 問題の回避策が
  この 2 つ。CUDA graph も落ちる報告があるため既定で無効。
- 配管は exl3 と同一（`kbuild/serve_script.py`）: pip → launch → proxy → tunnel → Gist → 予算ループ。
- `result.json` に `engine` / `version` / `tp` / `knobs` / `spec_decode=false` を残す。
- 詳細と実測手順: `notes/engine-rollout.md`。

## 生成物ポリシー（F-3）

- **notebook は commit する**（GitHub で差分レビューでき、clone 直後に push できる）。
  手で編集せず、必ずビルダーで再生成する。`kaggle/check_notebooks.py` が一致を検査。
- **script の `main.py` は種類で扱いが違う**:

  | 種類 | 出力先 | git（方針） | 一致検査 |
  |---|---|---|---|
  | 骨格版（配信） | `kaggle/step2-script/` `step2-vllm-script/` `llama-server-script/` | commit | `check_scripts.py` |
  | 骨格版の smoke | 同 `smoke-<tunnel>/`（`cli build --smoke` で生成） | commit | `check_scripts.py`（`KC_SMOKE=1` でもう一度走らせる） |
  | セル連結版（配信しない） | `kaggle/{step1,step2,step2-vllm,…}/script/` | commit | `check_scripts.py` |
  | 旧 script（exl3 / lmdeploy / sglang） | `kaggle/exl3-serve-kernel/` など | gitignore | `tests/test_exl3_build.py`（sha256） |

  旧 script だけ gitignore なのは、base64 埋め込みが巨大でレビューが効かず、
  代わりに決定性テストで担保しているため。新しい script は**骨格のおかげで
  読める部分が多く**、バイト比較できる価値があるので commit 対象にした。

  > 骨格版・セル連結版の 10 ディレクトリ（約 4.2 MB）と smoke 5 つ
  > （約 2.2 MB）は**まだ未追跡**。commit するか gitignore に寄せるかは
  > 未決定（下の残タスク #7）。`check_scripts.py` は git ではなくディスクを
  > 見るので、どちらでも動く。
- `build/` ディレクトリは作らない（生成物の二重パスを避ける）。
- **commit 対象は notebook と script の `main.py`、ビルダー無しカーネルの `main.py`**。
  gitignore された旧 script 生成物だけは commit しない（`cli push` が直前で再生成する）。

  > **旧 script 3 系統（exl3 / lmdeploy / sglang）だけ smoke の扱いが違う。**
  > これらは `smoke_out` を持たず、`--smoke` は同じディレクトリへ書き込む
  > （`cli push --smoke` が直前に再生成して押すので実害は出ないが、手元の
  > 通常版は smoke 変種で上書きされる）。`check_scripts.py` が直後に
  > 不一致として検出するので、`cli build` で戻す。gitignore 済みの生成物に
  > smoke 専用ディレクトリを増やすと、クリーン clone で `test_spec` が
  > 落ちるため、あえて揃えていない。

## 開発時のチェック

```bash
python3 -m cli check        # build → ruff → pytest → notebook 一致 → script 一致 → compileall
```

個別に回す場合:

```bash
python3 -m cli build                        # 全ビルダー
python3 -m cli build --smoke                # smoke 変種も生成（別ディレクトリ）
python3 kaggle/build_engine_serve.py        # LMDeploy / SGLang（cli build に含まれる）
python3 -m ruff check .
python3 -m pytest tests -q                  # 109 本
python3 kaggle/check_notebooks.py           # 13 冊の一致
python3 kaggle/check_scripts.py             # 18 カーネル / 46 ファイルの一致
python3 -m compileall -q kaggle/exl3-serve-kernel kaggle/ci-cpu kaggle/step2-script
```

`check_notebooks.py` と `check_scripts.py` は「一時ディレクトリへ写す → ビルダーを
走らせる → commit 済みとバイト比較」で、その共通部は `kaggle/_regen.py` にある。
**写した生成物を先に消す**ので、ビルダーが黙って書き損ねたときに古いファイルと
一致して通ることはない。

`check_scripts.py` は台帳の `smoke_out` を見て、smoke 変種も `KC_SMOKE=1` で
もう一度走らせて突き合わせる（通常版を smoke で上書きしていないことも同時に見る）。

CI は GitHub Actions（py3.10）と `ci-cpu` カーネル（py3.13）の二系統。
どちらも生成物を ruff / compileall に掛ける。

### テストの見どころ

| ファイル | 内容 |
|---|---|
| `test_spec.py` | 台帳と生成メタデータの一致・id 重複・ビルダー存在 |
| `test_constants.py` | ポート番号の一元化（`runtime/constants.py` との突合） |
| `test_exl3_build.py` | 3 モードの生成・sha256 決定性・Secret 非混入・AST |
| `test_exl3_server.py` | MockEngine で `/v1` の形（SSE / 400 / 404） |
| `test_engines.py` | LMDeploy / SGLang の argv・pip 固定・生成決定性・トンネル分岐・`from_cells` の連結 |
| `test_serve_notebook.py` | notebook の 1 本化が崩れていないか（共有セルのバイト一致・順序・エンジン名の混入） |
| `test_step2_builder.py` | Ollama / llama-server / vLLM のセル順と未定義名（`test_cell_name_order`） |
| `test_step2_proxy.py` | Basic 認証 ON/OFF・keepalive・`/stop`・`/quota` |
| `test_notebooks.py` | 13 冊がビルダー出力と一致 |
| `test_secrets.py` | Secret 参照と台帳の一致（生成物は除外） |
| `test_kbuild.py` / `test_runtime.py` / `test_cli.py` | 梱包・埋め込み・runtime の純関数・CLI の引数 |

## 既知の罠（要約）

| 症状 | 原因 / 対処 |
|---|---|
| モデルロードが 5 分で落ちる | `OLLAMA_LOAD_TIMEOUT` 未設定 |
| 黙って CPU 推論になる | CUDA バックエンド欠落。`nvidia-smi` と torch の版を確認 |
| `llama-server` が起動しない | Dataset 経由で実行ビットが落ちる → `chmod +x` |
| 15 KB のバイナリを弾く | サイズ判定が 1 MB 閾値 → 判定を緩める |
| verify が稀に落ちる | ネットワーク瞬断。リトライで吸収 |
| MTP が効かない | f16 KV + 長いコンテキストが先に効く。KV 型を明示 |
| UI から 401 | `EXL3_API_*` / `API_PASS` と UI の認証入力が不一致 |

詳細と実測値: `notes/runtime-notes.md`。手順の詳細: `notes/operations.md`。
設計判断: `notes/exllamav3-kaggle-decisions.md` / `notes/engine-rollout.md`。

## 残タスク

CPU でできることは一通り終わっている（2026-10-30 に「先に潰す穴」2 件を解消）。
残りは **GPU が要るもの**と**方針決定が要るもの**。詳細は
`notes/refactor-plan.md` の第 4 次計画。

### 済んだこと（2026-10-30）

- **script 骨格が ngrok に対応**した（→ 2026-11-01 に撤回。骨格は cloudflared
  だけを持つ。`start_tunnel()` が `start_cloudflared()` へ渡すだけになった）。
  `result.json` の `tunnel` は**実際に張った方**を書く。
- **script 版が `--smoke` に対応**した。`smoke_out`（`smoke-<tunnel>/`）へ
  別書きするので通常版を壊さない。`cli push --smoke` も通る。

### GPU が要る（クォータ承認待ち）— **3 種 × 1 回・全種類**

> **カーネルは作成済み**（2026-10-30・`--no-run` で version 1 を保存。GPU 消費ゼロ）。
> 台帳の **GPU カーネル 20 本すべて**（6 エンジン + 測定系）が Kaggle 上に存在する。
> クォータのリセットは **2026-10-31 09:00 JST**。当日の手順・Secrets・判定は
> `notes/gpu-run-sheet.md`。

骨格版 script（`kbuild/serve_script.py`）を **3 種 × 1 回**だけ `--smoke` で回す。
1 種につき 1 回なので、**全種類を試す**（1 本通ったから他を省く、はしない）。

| # | 種 | カーネル | 回数 | 1 回 |
|---|---|---|---|---|
| 1 | Ollama（GGUF） | `step2-orcarouter-qwen38-27b-cloudflared-script` | 1 | 8〜12 分 |
| 2 | vLLM（AWQ + MTP） | `step2-vllm-qwen38-27b-cloudflared-script` | 1 | 8〜25 分 |
| 3 | llama-server（直起動） | **配信カーネル無し**（下記） | — | — |

> ⚠ **llama-server は 2026-11-01 から GPU カーネルを持っていない。**
> この系統は `llama-server-ngrok` / `-ngrok-script` の 2 本しか無く、ngrok 廃止で
> 両方 retired にしたため。コード（`kaggle/llama-server/cells` と
> `kaggle/build_llama_server.py`・既定 cloudflared）は残っているので、台帳に
> 1 行足せば復活する。**新しい Kaggle カーネルを作るので、やるなら別途判断**。

合計 **3 回・約 1 時間**。合否は `result.json` の `stage=done` /
`verify_direct.checks` が全 true / `stop_remaining=[]`。

この 3 回で**骨格が Kaggle で起動して会話できること**までが確定する。smoke は
proxy / トンネル / Gist / keepalive を省略するので、そこは次の別枠:

| # | 内容 | 回数 | 備考 |
|---|---|---|---|
| 4 | ~~ngrok の**実配線**~~ | — | **不要**。2026-11-01 に ngrok を廃止し、ビーコン + cloudflared に一本化 |
| 5 | 配信カーネル 1 本で keepalive / `/stop` / UI | 1 | 3 種のどれか 1 本が通ってから |
| 6 | LMDeploy / SGLang の比較実測 | 2 | 従来からの宿題（semi・比較用） |

手順と合否基準は `notes/gpu-smoke-plan.md`。

### 方針決定が要る（無料）

| # | 内容 | 選択肢 |
|---|---|---|
| 7 | script 生成物 15 ディレクトリ（約 6.4 MB・現在未追跡）を commit するか | commit（`check_scripts.py` がそのまま効く）／gitignore（exl3 方式の sha256 決定性テストへ） |
| 8 | `kernel_type` を**同一 id のまま**変更できるか | `kaggle kernels push --no-run`（QUICK_SAVE）で実験。**不可なら移行は id 増**（Secrets を UI で再登録） |
| 9 | notebook を `kaggle/archive/` へ移して削除してよいか | GPU の #1〜#3（3 種）と #5 が通ってから。1 カーネルずつ（一括 `git rm` をしない） |

### 据え置き（やらないと決めたこと）

- `kaggle/build_vllm_dryrun.py` / `build_keepalive_test.py` は `Flow` を使わない
  （配信カーネルではないので共有対象外）。keepalive セルの重複だけ残っている。
- `runtime/engines/{vllm,llama_server}.py` は作らない（起動ラダーは共有しない）。
- bench / smoke / bench_mtp の notebook 版は消さない（semi で便益が小さい）。
- 旧 script 3 系統（exl3 / lmdeploy / sglang）は smoke を同じディレクトリへ書く
  （gitignore 済みの生成物に smoke 専用パスを増やすと、クリーン clone で
  `test_spec` が落ちるため）。`check_scripts.py` が検出するので事故らない。

## 運用上の注意

- **push は即実行**。GPU カーネルは 1 回で決める。CPU カーネルは無料。
- カーネルは有限時間で自分から止まる（`RUN_MINUTES` / `HARD_MAX_MINUTES`）。
  keepalive は proxy 経由で延長する。`/stop` は `X-Kc-Stop` ヘッダ必須。
- 公開 URL は Gist（`url.txt`）に書く。cloudflared の quick tunnel は毎回変わる。
- Dataset は slug 指定で最新版がマウントされる。版は `result.json` に残す。
- 古いカーネルは削除せず `kaggle/archive/` に残す（検索用）。

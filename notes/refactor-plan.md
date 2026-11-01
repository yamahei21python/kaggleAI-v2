# リポジトリ再編計画（2026-10-30 時点・検討メモ）

実装前の検討結果。対象は Kaggle カーネル群（Ollama / llama-server / vLLM /
exllamav3）とローカルのランチャー・クライアントまで。

## 現状の数字

| 項目 | 現状 |
|---|---|
| 追跡ファイル | 235 |
| カーネル用ディレクトリ | `kaggle/` 直下に 20 ディレクトリ + `build_*.py` 12 本（step2 系だけで 8 バリエーション） |
| ビルダー | `kaggle/build_*.py` 12 本、計 6,500 行超（最大 `build_step2_vllm.py` 2,163 行、`build_exl3_serve.py` 619 行） |
| 退役コード | `kaggle/archive/` 121 ファイル（旧カーネル 46 スナップショット = json+ipynb 46 組、py 18 / sh 10）、`notes/uzu-backup/` 13 |
| ローカルランチャー | 4 本 計 1,000 行 + `lib/launcher-common.zsh` 214 行（0b 完了後） |
| クライアント | `ask.py` / `kaggle-quota.py` / `kcconfig.py` がルート直下 |
| テスト | `_selftest_proxy.py` / `_selftest_builder.py` + pytest 23 本 + CI のノート一致（12 冊） |

## 問題点

1. **ビルダーの重複** — base64 埋め込み・metadata 生成・nbformat 出力・push 手順が
   11 本にコピーされている。実測した関数重複:
   `_find_input` 4 箇所 / `_detect_gpu` 4 / `probe_url` 4（`build_step2_vllm.py`
   内ですら 2 回）/ `save_output` 3 / base64 埋め込みブロック 6。
   （生成 notebook 内にも各 6 コピーあるが、抽出元はビルダー側）
2. **ソース／生成物／ログの境界が曖昧** — notebook と builder の両方をコミットし、
   生成 `main.py`（`exl3-serve-kernel/{serve,dryrun,keepalive}/`）は未追跡。
3. **命名で現役／退役が判別できない** — `step2`, `step2-vllm`,
   `step2-vllm-kernel/{ngrok,cloudflared}`, `llama-server-ngrok`, `step2-ka-test`。
4. **実行時コードの真実源が暗黙** — `chat_server.py` / `auth_proxy.py` /
   `llama-chat.html` を各ビルダーが個別に埋め込んでいる。さらに生成される
   `main.py` の proxy 起動・トンネル・keepalive・verify が
   `build_step2.py` / `build_llama_server.py` / `build_vllm_dryrun.py` /
   `build_step2_vllm.py` のテンプレート文字列として重複し、`exl3-serve` は
   単一 Template で別系統。実質 2 系統（cell 外部化済み step2 / 単一テンプレート
   exl3）に、両方の悪いところ取りの中間 3 本が挟まっている。
5. **ローカル層の重複（0b で解消済み）** — `run-llama-server` は
   `/Applications/Ollama.app` と `~/.ollama/models/blobs` を直参照していた。
   → `lib/launcher-common.zsh` の `kc_ollama_server_bin` / `kc_ollama_blob` /
   `kc_port_in_use` / `kc_start_ui_proxy` に集約し、4 本とも source する形に
   した。`run-llama-server` は zsh 化（zsh の `read -p` は coprocess 扱いで
   失敗するため `kc_ask` / `kc_pause` を追加）。
6. **設定の分散** — `config.env` は launcher（`CFG_*` 経由）と `kcconfig.py` が
   読む共有値だけで、builder のノブ（`STEP2_TUNNEL` など）は環境変数でしか
   渡せない。ポート（9931 / 11435 / 19950 / 8899）は `config.env` と各コードの
   既定値で二重管理になっている。
7. **notes に索引がない** — 35 ファイル、`uzu-backup` が同階層。

## バックエンドの現状（統一先は OpenAI `/v1` で既に揃っている）

| 層 | 現状 |
|---|---|
| クライアント | `ask.py` は `/v1/chat/completions` のみ。`llama-chat.html` は `/v1/*` + `/quota` + `/stop` |
| Kaggle カーネル | Ollama（`build_step2.py`）/ llama-server（`build_llama_server.py`）/ vLLM（`build_step2_vllm.py`）/ exllamav3（`exl3-serve`） |
| vLLM 固有 | wheelhouse 導入（`build_step2_vllm.py:344-419`）、torchaudio 修復（`:421-452`）、`--help` 事前検査（`:455-488`）、起動ラダー＋予算（`:757-835`）、**TP=2（DDP ではない）**（`:218,731`） |
| llama-server 固有 | `--spec-type`（MTP）、`-c`、Ollama.app 内のバイナリ＋blob（`run-llama-server.command:21-31`） |
| Ollama 固有 | `num_ctx`、`/api/ps`。OpenAI 互換は `/v1` |

## 提案レイアウト

```
runtime/                  # カーネルに埋め込む「真実源」
  probe.py                # _find_input / _detect_gpu / _fits / sm 判定
  net.py                  # probe_url / トンネル URL 抽出
  model.py                # モデル配置・HF 取得・空き容量判定
  keepalive.py            # read_conf / write_conf / stop
  engines/
    ollama.py
    llama_server.py
    vllm.py               # wheelhouse / 起動ラダー
    exllamav3.py          # Cache / max_history / warmup
  serve/chat_server.py    # exllamav3 用 OpenAI 互換サーバ
  proxy/auth_proxy.py
  ui/llama-chat.html
kbuild/                   # 梱包の共通層
  spec.py                 # 1 カーネル = 1 spec
  pack_nb.py              # notebook 生成（cell id / language_info の差を吸収）
  pack_script.py          # 単一ファイル main.py 生成（exl3 方式）
  embed.py                # base64 / チャンク分割
  metadata.py             # kernel-metadata.json（title↔id の slug 検査）
  cli.py                  # build / push / check
kernels/                  # spec のみ（id は変更しない）
  step1/ step2/ llama-server/ step2-vllm/ exl3-gpu-check/ exl3-serve/ ...
cli/                      # build / push / check / list / ask / quota（実装済み）
lib/                      # ローカルランチャー共通（launcher-common.zsh を拡張）
# 生成物: notebook は commit、exl3 script は gitignore（F-3 決定。build/ は作らない）
notes/index.md + notes/archive/
```

### spec の項目

```python
Kernel(
    id="yamahei21python/step2-vllm-qwen38-27b-cloudflared",  # 不変
    kind="notebook",            # notebook | script
    gpu=True,
    datasets=[...],
    runtime=["proxy/auth_proxy.py", "ui/llama-chat.html"],
    engine="vllm",              # ollama | llama_server | vllm | exllamav3
    knobs={"STEP2_TUNNEL": "cloudflared", "VLLM_PIP_SPEC": "..."},
    stages=["restore", "proxy", "serve", "verify", "keepalive"],
    budget={"hard_minutes": 180, "keepalive_minutes": 60},
)
```

エンジン差（コンテキスト指定・thinking・MTP・起動待機）は spec ではなく
`runtime/engines/*.py` の能力表に閉じ込める。クライアントは OpenAI `/v1` 一本のまま。

## 共通化の範囲（2 層に分ける）

「LLM 以外を共通化する」を正確に言うと、**推論エンジン固有の部分だけ残して、
それ以外の配管を共通化**する。境界の目安は「モデル名・量子化・推論フラグを
知らなくても書けるコードか」。知らなくて済むなら `runtime/`、知らないと
書けないなら `runtime/engines/`。

### 層 1: ビルダー層（梱包・生成）— ほぼ機械的に共通化できる

実測した重複:

| 部品 | 重複 |
|---|---|
| `_find_input` | 4 本（step2 / direct / vllm / dryrun） |
| `_detect_gpu` | 4 本（+ bench/smoke 系にも類似） |
| `probe_url` | 4 本（ngrok / cloudflared の分岐だけ違う） |
| `save_output` / `fail` | 3 本 |
| base64 埋め込み | 6 本 |
| nbformat / cell id / language_info、metadata（title↔slug 検査）、`wait_http` / `http_json` / `secret`、keepalive conf 読み書き | エンジン知識ゼロ。`kbuild/` + `runtime/` へほぼそのまま移せる |

ビルダー行数の大半はこの層。共通化率は **80% 前後**。

### 層 2: 生成される実行時コード — 段 2 の最優先

ビルダーの関数より効くのは、**生成される `main.py` の中身**が複数テンプレートに
コピーされている点。proxy 起動・cloudflared / ngrok・keepalive・verify が
`build_step2.py` / `build_llama_server.py` / `build_vllm_dryrun.py` /
`build_step2_vllm.py` に文字列として重複し、`exl3-serve` は別方式（単一
Template）。`runtime/` を真実源にして base64 埋め込みする方式へ寄せれば、
この重複は構造的に消える。**段 2 では kbuild 抽出と同じ優先度でここを一元化する。**

### 共通化しない / 共有を限定する部分

- **起動ラダーは共有しない** — vLLM の `--help` 事前検査 → フラグ選択 → TP=2 は
  他エンジンと構造が違いすぎる。共有するのは「GPU 判定 → VRAM → コンテキスト
  候補の決定」までで、起動関数は `runtime/engines/*.py` に残す。
- vLLM: wheelhouse、`--help` 検査、起動ラダー、TP=2
- llama-server: バイナリ / blob 解決、`--spec-type`（MTP）、`-c`
- Ollama: `num_ctx`、`/api/ps`
- EXL3: Cache、max_history、warmup、MTP 投げ込み
- クライアントは既に OpenAI `/v1` で揃っているので、共通化ではなく現状維持。

### 追加の判断（2026-10-30）

1. **bench / smoke / bench_mtp も段 2 から `runtime/` を使う** — 段 6 まで待つと
   `_gpus` / `_chat` の第 2 系統が残り、共通化が半分で止まる。少なくとも
   probe / net は最初から共有する。
2. **「バイト一致」は exl3 以外は AST + 正規化一致に緩める** — 全カーネルに
   byte 一致を課すと、空白や cell id の差で移行が止まりやすい。step2 系は
   `check_notebooks.py` 相当の意味比較で十分（exl3 は単一 Template なので
   sha256 再現を維持）。
3. **hosts は表駆動** — `probe_url` の種別は ngrok / cloudflared の 2 つだけ。
   URL 抽出正規表現と再試行回数を表に持てば 1 関数で済む。
4. **能力表を増やさない** — 共通化を広げると MTP / thinking / KV dtype /
   起動待機の差が spec に漏れ出しやすい。`runtime/` の IF を増やさず、
   能力表＋起動関数の線を維持する。

## 移行順（各段で差分検証 + CPU テスト green の後に次へ）

0. **事前整理（ルート掃除 + README の扱い決定）** — コード移設より先に方針だけ決める。
   ルート直下の実験ゴミ（`out.*` 64 個・16.8 GB の gguf・bench ログ 2 本・`diagout`）と
   `kaggle-dataset-*` / `.wheel` の扱いを決めて gitignore を確定。README（1,255 行）は
   新レイアウトに合わせた再構成を段 7 に回し、それまでは現行パスのまま維持する。
0b. **ローカルランチャーの common 化（完了）** — push 不要・クォータ不要。
   `run-llama-server` / `run-uzu-server` を共通 lib に寄せ、バイナリ／blob 解決・
   ポート検査・UI 中継の起動を関数化。`run-llama-server` は zsh 化。
1. **棚卸し表** — 現役 / 準現役（`build_llama_server.py`, `build_vllm_dryrun.py`）/
   退役（`step2-ka-test`, `capture_patch`, `archive/`）。カーネル id は変えない。
2. **kbuild 抽出 + 生成コードの一元化（最優先）** — `build_exl3_serve.py` と
   `build_step2.py` の共通部分から。proxy / トンネル / keepalive / verify を
   `runtime/` の真実源に寄せ、各ビルダーは base64 埋め込みで参照する。まず
   exl3-serve を移行し、生成物が旧ビルダーとバイト一致することを確認。
   - 進捗: 梱包層 `kbuild/`（`embed` / `metadata` / `pack_nb` / `pack_script`）を
     抽出。exl3（byte 一致）、step2、direct、step1、smoke、bench、bench_mtp、
     keepalive、vLLM 系 3 本を移行済み（全 12 ノートで生成一致）。
   - 進捗 2b: 生成コードの真実源 `runtime/`（`net` / `proc` / `storage` /
     `keepalive` / `proxy` / `engines/ollama`）を作り、step2（cf/ngrok）・
     direct・dryrun・vLLM 本体（cf/ngrok）の proxy / tunnel / keepalive /
     verify / 後片付けを `kc_*` 呼び出しへ置換済み。差は引数
     （`on_invalid` / `quiet_clamp` / `match` / `skip_http_errors` /
     `progress_every` / `env`）で表現し、挙動は変えない。
     `runtime/_selftest.py` を CI に追加。exl3 の単一 Template は同機能を
     内包しているため 2b では触らない（段 7 で統合判断）。
3. **`build_step2.py`**（cells 外部化済みで最も楽）→ **`build_llama_server.py`**。
   生成物は AST + 正規化一致で比較（byte 一致は exl3 のみ）。
4. **`build_vllm_dataset.py` → `build_vllm_dryrun.py`**
5. **`build_step2_vllm.py` を最後**（最大・起動ラダーが複雑。回帰テストをここで追加）
6. bench / smoke / keepalive_test — probe / net は段 2 で共有済みにし、ここでは
   残りの定型（chat / gpus 出力）を寄せる。
   - 進捗: `runtime/proc.py` に GPU 検出 / アイドル待ち（`kc_detect_gpu` /
     `kc_wait_gpu_idle`）、`runtime/engines/ollama.py` に `kc_ollama_chat` を追加。
     smoke / bench / bench_mtp は GPU 検出と chat を runtime 呼び出しへ置換。
     keepalive_test は keepalive / proxy / tunnel を `source_bundle` 埋め込みへ移行
     （`/stop` に `X-Kc-Stop` ヘッダを追加し、潜在 405 を修正）。
     step2 / direct / vLLM の各ビルダーに残っていた `_detect_gpu` も
     `kc_detect_gpu()` に統一。
7. クライアント統合（`cli/`）、テスト移設（`tests/`、フェイククロックで keepalive）、
   notes 再編、生成物ポリシー（F-3: notebook は commit / script は gitignore）、
   **README 再構成**（現行 1,255 行は旧パスを直書き。新レイアウトへ移し、手順詳細は
   `notes/` に分割する）。
   - 進捗: exl3 の単一 Template も runtime へ統合。proxy 起動 / tunnel URL 待ち /
     keepalive conf / `/stop` / GPU 検出 / 締切計算を `kc_*` 呼び出しへ置換し、
     `kc_start_proxy` に interpreter 指定（`python=`）を追加。exl3 の
     `_selftest_build` は埋め込み関数の存在と AST 構文を検証する。
   - 進捗 7b: `_selftest_*.py` を `tests/` の pytest へ移設（runtime / kbuild /
     step2 builder / exl3 ×2 / notebook 一致）。proxy の統合テストは
     pkill・atexit を含むため子プロセスで実行する。keepalive の時間予算は
     `kc_keepalive_deadline` / `kc_keepalive_expired` に純関数化し、`now` を
     渡すフェイククロックで検証する。CI と ci-cpu カーネルは `pytest tests` に切替。
   - 進捗 7c: notes を再編。`notes/index.md` を追加し、ベンチ JSON・ログは
     `notes/archive/` へ移動（`prompt_sweep.py` の既定出力も archive 配下）。
     旧 `kaggle-ollama-ngrok.ipynb` は `kaggle/archive/` へ移動。
   - 進捗 7d: CLI を `cli/` に統合（build / push / check / list / ask / quota）。
     既定は build のみ、push は `--yes` 必須で残量を先に表示。ルートの
     `ask.py` / `kaggle-quota.py` / `kcconfig.py` はシム化。
     カーネル台帳を `kbuild/spec.py` に集約し、ポートは `runtime/constants.py` へ。
     `tests/test_spec.py` / `test_constants.py` / `test_cli.py` で台帳・定数・
     既定動作を固定。README は 1,261 行 → 約 260 行へ再構成し、手順詳細と
     実測ログを `notes/operations.md` / `notes/runtime-notes.md` に切り出した。
   - 進捗 7e: LMDeploy / SGLang の配信カーネルを追加（`kbuild/serve_script.py`
     + `kaggle/build_engine_serve.py` + `runtime/engines/{lmdeploy,sglang}.py`）。
     既存 AWQ Dataset を再利用し、MTP は使わない（`spec_decode=false`）。
     CPU テストは `tests/test_engines.py`（argv・pip 固定・生成決定性・AST）。
     GPU 実測は未実施（1 エンジン 1 回に集約）。
   - 進捗 7f: レビュー反映と smoke モード。`secret()` に `kaggle_secrets`
     フォールバック、`finally` での段階停止（`stop_remaining` を記録）、
     `/stop` がエンジンも落とすよう `--stop-pid` を渡す。SGLang は公式の
     sm75+ 回避策（`--attention-backend triton --sampling-backend pytorch`）を既定に。
     `--smoke` を `cli build` / `cli push` と exl3 / engine の両ビルダーへ追加。

## CPU 検証（無料・各段の受け入れ条件）

**原則: リファクタの各段は、GPU なしで検証できるテストを green にしてから次へ進む。**
GPU を使うのは最後の 1 回（実測）に集約する。CPU カーネルの push は GPU クォータを
消費しないので、段 2–5 では dryrun カーネルの通し確認まで行う。

### 段ごとの合格条件（Definition of Done）

| 段 | CPU テスト内容 | 合格条件 |
|---|---|---|
| 0b ランチャー | `zsh -n` / `shellcheck`、common lib の解決関数を一時 dir で単体 | 旧 2 本と生成コマンドが一致（完了） |
| 2 kbuild / 生成コード | 同一 spec → `main.py` の sha256 再現、py3.13 `ast.parse`、metadata 必須キー、base64 展開一致、hosts 表の URL 抽出 | exl3-serve 3 モードが旧ビルダーとバイト一致。runtime 埋め込みが全ビルダーで同一ソース |
| 3 step2 / direct | 同上 + `check_notebooks.py` 相当 | **AST + 正規化一致**（byte 一致は exl3 のみ） |
| 4 vllm dataset / dryrun | データセット除外パターン（`.cache/` `*.lock`）のテスト、MANIFEST 生成 | 生成物一致・除外が効く |
| 5 vllm 本体 | **引数組み立ての純関数テスト**（起動ラダーの各段を表駆動で）、`--help` 出力のパーサ | ラダー分岐が全ケース通る（実起動は GPU で） |
| 6/7 テスト移設 | pytest 一式 + CI 追加、keepalive はフェイククロック | CI green（完了） |

### リファクタ後に必ず走らせる CPU テスト（既存資産の拡張）

- `tests/` に pytest 化: `_selftest_server.py`（MockEngine: non-stream / SSE /
  `/v1/models` / `/health` / usage）、`_selftest_proxy.py`（Basic 認証・keepalive
  リセット・`/stop` ヘッダ必須・`/quota`）、`_selftest_builder.py`（生成一致）。
- **新規の純関数テスト**: `keepalive` の期限計算（started/last/mtime/HARD_MAX）を
  フェイククロックで全分岐、`probe` の nvidia-smi 不在、`net` の trycloudflare URL
  抽出とリトライ（ENETUNREACH を含む）、GPU 名パース（`2x Tesla T4`）。
- **カーネル main.py の stage 関数**: `stage_env`（nvidia-smi 無し）/ `find_model`
  （tmpdir でマウント構造を再現）/ `read_conf` を import して単体実行できる形に
  しておく（スクリプト直実行のままでも `python -c` で呼べる）。
- **UI**: `llama-chat.html` は静的検証（ID 重複・fetch 先の組み立て）＋ auth 入力を
  純関数化して Node の単体テスト。見た目の確認は手動。
- **Kaggle CPU カーネル push（無料）**: リファクタごとに
  `exl3-serve-kernel/dryrun`（mock・約 5 分、cloudflared 公開 URL まで）と
  `exl3-serve-kernel/keepalive`（約 3 分、延長と `/stop`）を push し、`result.json` が
  `ok` / `passed=true` であることを確認。`llama-server-cpu-check` は約 23 分
  かかるので、llama.cpp 側を触ったときだけ。
- **ローカル（darwin）**: Ollama は実起動で `/v1` 疎通、llama.cpp はバイナリが
  ある場合のみ。`ruff` / `zsh -n` / `shellcheck` は現行 CI のまま。

### GPU に回すもの（最終 1 回・明示承認が必要）

- exllamav3: wheel 導入・ロード・日本語生成・MTP 投機デコードの tok/s
- vLLM: 実起動・TP=2・実 tok/s
- 上記以外で GPU が要る検証は作らない（CPU で代替できる形にテストを設計する）。

## やらないこと / リスク

- Kaggle カーネル id の変更（Secrets・dataset_sources・実行履歴が紐づく）
- 一括 `git mv` の大移動（新設 → 移行 → 削除の順で行う）
- 退役カーネルの削除（`kaggle/archive/` は残す）
- 4 エンジンの完全な統一抽象化（起動ラダーが違いすぎる。能力表＋起動関数で十分）
- 共通化の拡大で能力差（MTP / thinking / KV dtype / 起動待機）が spec に漏れ出る
  こと。`runtime/` の IF を増やさず、能力表＋起動関数の線を維持する
- 全カーネルへの byte 一致要求（空白・cell id の差で移行が止まる。AST + 正規化
  一致を既定とし、byte 一致は単一 Template の exl3 のみ）
- Ollama ネイティブ `/api/chat` へのクライアント統一（`/v1` で足りる）
- `__KEY__` 置換によるビルド時埋め込みの継続（コード以外は実行時環境変数へ）
- GPU カーネルの push を伴う検証（push は即実行・クォータ消費）。ビルド検証は
  ローカル差分で行い、CPU カーネルの push（無料）は段ごとの受け入れ条件に含める

## 効果見積り

- ビルダー 6,524 行 + `build_exl3_serve.py` 619 行 → 共通 ~300 行 + spec 各 50–150 行。ビルダー層は **80% 前後**まで共通化可能（実測の重複表）。生成コード側（proxy / tunnel / keepalive / verify）の一元化が最も効く。
- `check_notebooks.py` の対象 7 冊 → 12 冊（direct / vLLM dataset / dryrun / vLLM 本体 2 種を追加）。比較は AST + 正規化、exl3 のみ byte。
- 現役／退役がディレクトリ構造で自明になる。

## 未考慮点の調査（2026-10-30 追記）

計画に無かった実務上の穴。P0 はリファクタより先に直す（特に 1〜3 は
`exl3-serve` の GPU 実行前に必要）。

### P0: セキュリティ / 機能の欠陥

1. **API パスワードが成果物に残る** — 旧 `build_kernel.py:286` が `api_pass` を
   `data` に入れ、`RESULT:` 行としてログにも出ていた。
   → **解決済み**（`result.json` は認証 ON/OFF のみ、`tests/test_exl3_build.py` が
   `api_pass` 非混入を検査）。
2. **チャット UI が認証に対応していない** — `llama-chat.html` に
   `Authorization` を付ける箇所が無い。`exl3-serve` は Secrets か乱数で
   `API_PASS` を必ず設定するため、UI から `/v1/*` が 401 になる。
   → UI にパスワード入力（sessionStorage）を足すか、認証 OFF を明示選択。
3. **URL の配布経路がエンジンで違う** — Ollama 版は Gist に `url.txt` を
   書く（`build_step2.py:438-503`）が、`exl3-serve` は `result.json` のみ。
   手元の `ask.py` は Gist 前提なので URL を拾えない。
   → spec に publish 段（gist / result / 手動）を持たせる。**一部完了**
   （exl3 も `publish_url()` で Gist に書く。spec への明示は未着手）。
4. **`/quota` がカーネルで動かない** — `auth_proxy.py:61` は
   `~/.kaggle/kaggle.json` しか読まないため、ビルダーは `--no-quota` を
   渡している。Kaggle では環境変数 `KAGGLE_USERNAME`/`KAGGLE_KEY` があるので
   env も読めるようにする。**完了**（`runtime/quota.py` が env を読む。
   `--no-quota` が残るのは keepalive-test とローカル uzu のみ、いずれも意図的）。

### P1: 重複・再現性・検証

5. **クォータ API の実装が 2 つ** — `kaggle-quota.py:21,103` と
   `auth_proxy.py:61,304`。`runtime/quota.py` に一本化。**完了**
   （`kc_kaggle_credentials` / `kc_fetch_quota` / `kc_quota_summary`。
   proxy はカーネル上で単一ファイルのため、ビルダーが `runtime.proxy_source()`
   で quota ソースを先頭に同梱する。CLI は `runtime.quota` を直接 import）。
6. **定数の分散** — ポート番号（9931 / 11435 / 19950 / 8899）が 42 箇所、
   cloudflared の `releases/latest` が 9 箇所。後者はバージョン固定しないと
   再現性がない。→ `runtime/constants.py` と `CLOUDFLARED_VERSION`。
   - cloudflared は **2026.10.4 に固定**（Dataset 同梱バイナリと同一版、
     サイズ 40,247,428 B も一致）。`runtime/net.py` の `KC_CLOUDFLARED_VERSION`
     / `kc_cloudflared_url()` に集約し、6 ビルダー + 生成物の `latest` を排除。
     ポート番号の集約は未着手（builder ごとに意味が異なるため spec 化と同時に行う）。
- 進捗: 上記のうち cloudflared のバージョン固定は完了。
7. **シークレットの台帳が無い** — `NGROK_TOKEN` / `GITHUB_TOKEN` / `GIST_ID` /
   `KAGGLE_USERNAME` / `KAGGLE_KEY` / `HF_TOKEN` / `EXL3_API_USER` /
   `EXL3_API_PASS` が散在。spec に `required_secrets` を宣言し、CI で
   コード中の `get_secret("...")` と突き合わせる。
8. **CI と実行環境の差** — `ruff.toml` は `target-version = "py310"`、
   Kaggle は 3.13。生成物（`main.py`）は untracked で CI の対象外。
   → 生成物を `compileall` + ruff に掛ける段を追加。**完了**（GitHub CI と
   ci-cpu の双方で、生成後に `ruff --no-respect-gitignore` と `compileall`）。
9. **生成コードのテストが notebook だけ** — `_selftest_builder.py` は
   notebook から AST で関数を抜いて exec するが、script 系（exl3）は
   今回手動で 3 回走らせただけ。Mock で 20 秒の実行テストを追加。
10. **認証ありのプロキシテストが無い** — `_selftest_proxy.py` は
    「認証なし」で起動して検証している（docstring 明記）。401・
    `/stop` ヘッダ必須・チャンク本文の各経路を認証 ON でも回す。**完了**
    （別ポートで認証 ON を起動し、誤パスワード 401 / UI 公開 / ストリーミング /
    chunked / 非 ASCII Authorization を検証。68 チェック）。
11. **クライアント設定の二重管理** — `opencode.json` と
    `opencode-llama-local.json`、`ask.py` の Gist 依存、UI の `/quota` 依存。

### P2: 整理・運用

12. ルート直下の旧 notebook `kaggle-ollama-ngrok.ipynb`（4 セル・Ollama 版の
    初期版）と `notes/` 内のベンチログ・JSON をアーカイブへ。**完了**
    （notebook は `kaggle/archive/`、ベンチは `notes/archive/`、索引は `notes/index.md`）。
13. `my-cloudflared-bin` Dataset の更新手順とライセンス表記（再配布物）。
    **完了**（`notes/dataset-inventory.md`。Apache-2.0、版固定手順）。
14. 新しい kernel id を作るときのチェックリスト（Secrets はカーネル単位、
    `dataset_sources`、`title` の slug 一致）。**完了**
    （`notes/kernel-inventory.md`）。
15. CLI は既定を `build` のみにし、`push` は明示オプション（push は即実行で
    クォータを消費するため）。push 前に残量を表示する。

### 既に計画へ反映済みの点（再確認）

- ビルダー関数の重複、notebook/script 二系統、メタデータ生成の統一
- vLLM の wheelhouse / 起動ラダー / TP=2、llama-server のバイナリ・blob 解決
- ローカルランチャーの common 化、設定の一本化（0b 完了）
- 生成コード（proxy / tunnel / keepalive / verify）の一元化を段 2 の最優先に
- hosts（ngrok / cloudflared）の表駆動、bench/smoke も probe/net を段 2 から共有
- 生成物の比較は AST + 正規化一致（byte 一致は exl3 のみ）

## データセット整理案（2026-10-30 調査）

`kaggle datasets list -m` と `kaggle datasets files` で実物を確認した結果。

### 台帳

| slug | 中身 | サイズ | 参照カーネル数 | 状態 |
|---|---|---|---|---|
| `qwen38-27b-uncensored-hauhaucs-awq-mtp` | AWQ モデル本体 + `.cache/huggingface/` | 非公開（API 表記 0） | 26 | **`.cache` ゴミ混入** |
| `my-ollama-orcarouter-qwen38-27b` | Ollama blobs（16.8 GB + 0.93 GB）+ manifest | 同上 | 5 | blob レイアウトそのまま |
| `my-ollama-orcarouter-qwen38-27b-patch` | `bin/ollama` + `lib/ollama`（ランタイム） | 1.06 GB | 5 | **名前がモデル風で実体はランタイム** |
| `qwen38-27b-uncensored-exl3-4bpw` | EXL3 3 シャード + LICENSE | 15.9 GiB | 2 | 整備済み（subtitle/description/private） |
| `vllm-wheelhouse` | wheel 群 + MANIFEST.txt | 3.2 GB | 4 | バージョンはファイル名で固定 |
| `my-cloudflared-bin` | `cloudflared` 1 ファイル | 40 MB | 6 | **バイナリ版が不明（ファイル名に版なし）** |

プロジェクト外の旧データセット: `kawamura-finetune`, `kawamura-ruri-ft-data`,
`my-ollama-qwen2-5-coder-14b`, `my-ollama-qwen3-6-35b`, `my-ollama-gemma3-1b`,
`qwen3-27b-uncensored-gguf`（いずれも repo から参照なし）。

### 問題

1. **同じモデルの多重保持** — GGUF（Ollama blob）/ AWQ / EXL3 で 15〜17 GB 級が
   並存。合計 50 GB 超。目的（エンジン比較）は理解できるが、退役したものは
   容量を食い続ける。
2. **不要ファイルの混入** — AWQ 側に `.cache/huggingface/download/*.lock` など。
   再アップロード時に除外しないと版を重ねるたびに残る。
3. **命名の不整合** — `my-ollama-…-patch` はモデルではなくランタイム。
   `my-cloudflared-bin` は版が分からない。
4. **版の固定ができない** — `dataset_sources` は slug 指定（最新版がマウント）。
   どの版で実行したかが `result.json` に残らない。
5. **アップロード手順の分散** — `build_vllm_dataset.py`・`exl3-dataset-upload`・
   `build_capture_patch.py`・`assemble_patch_dataset.py` が別実装。
6. **ローカル実体** — `kaggle-dataset-orcarouter/` 17 GB、
   `-patch/` 1.4 GB、`-cloudflared/` 38 MB が repo 直下（gitignore 済みだが
   ディスクを占有）。
   → **2026-10-30: `datasets/{orcarouter,orcarouter-patch,cloudflared,gguf}/` へ移動し、
   重複 16.8 GB をハードリンクで解消済み**（E-2）。

### 提案

1. **repo を真実源にする** — `datasets/<slug>/dataset-metadata.json` を置き、
   `kbuild upload-dataset <slug>` で「除外パターン適用 → 版作成」を一本化。
   `.cache/` `.git/` `*.lock` を既定除外。
2. **既存 slug はリネームしない**（マウント参照が壊れる）。命名規則は
   **新規のみ**: モデルは `<model>-<quant>`、インフラは `infra-<name>-<version>`。
3. **版を記録する** — 各カーネルの `stage_env` で `dataset_sources` と
   マウント先の `dataset-metadata.json`（あれば）を `result.json` に残す。
4. **cloudflared は版を明示** — `infra-cloudflared-linux-amd64-<ver>` を新設し、
   バイナリ横に `VERSION.txt`。ビルダーの「latest を取得」は最後の手段に。
5. **退役の判断はユーザー** — **2026-10-30 決定: `my-ollama-qwen3-6-35b` /
   `my-ollama-qwen2-5-coder-14b` / `my-ollama-gemma3-1b` を削除、他は保持**
   （詳細と統合候補は `notes/dataset-inventory.md`）。**同日に削除実行済み**
   （依存 0 を確認、一覧 12 → 9 件）。削除は 1.7.4.5 では不可のため
   CLI 2.2.5 の `~/.venv-kaggle2` を使用した（下記 F-2）。
6. **ローカル実体は repo 外へ** — `~/models/kaggle-dataset/` 等に移し、
   repo からは `.env` のパス参照だけにする（17 GB を repo 直下に置かない）。
   **→ 2026-10-30 に撤回**（下記 E-2）。同一ボリュームなので移動では空きが増えず、
   `-p` を相対パスで叩く運用の方が事故らない。代わりに `datasets/` へ集約し、
   重複していた 16.8 GB をハードリンクで 1 本化した。
7. **容量の可視化** — `kbuild datasets` で「slug / サイズ / 参照カーネル数」を
   出す。参照 0 のものだけ削除候補として表示。

### 追加で見つかった抜け

- **新しいカーネルに Secrets が未設定** — `exl3-serve` / `-dryrun` /
  `-keepalive` は作成しただけで、UI からの `EXL3_API_USER` / `EXL3_API_PASS`
  が未登録（乱数フォールバック）。P0-1/P0-2 の修正と同時に登録が必要。
- **`dataset_sources` の版が暗黙** — 上記 4。実測の再現性に影響。
- **`kaggle-dataset-*` が repo 直下** — 上記 6。バックアップ対象か要確認。

## 最後にまとめてやる登録・削除（フェーズ F）

個別に先に触らない。配信の GPU 確認が済んだ後に一括で行う。

### F-1 Secrets 登録（カーネル単位）

| カーネル | Secrets | 備考 |
|---|---|---|
| `exl3-serve` | `EXL3_API_USER` / `EXL3_API_PASS` | 未登録だと乱数フォールバック（P0-1 修正後は必須） |
| `exl3-dataset-upload`, step1 | `KAGGLE_USERNAME` / `KAGGLE_KEY` | 設定済み（実測） |
| step2（Ollama）/ step2-ngrok | `NGROK_TOKEN`, `GITHUB_TOKEN`, `GIST_ID` | Gist への URL 公開に使用 |
| `build_vllm_dataset` | `HF_TOKEN` | モデル取得 |
| 新しいカーネル全般 | 上記の該当分 | **Secrets はカーネル id 単位**（別 id には自動で付かない） |

- 設定は UI のみ（CLI 不可）。カーネル作成時のチェックリストに含める。

### F-2 データセット削除（2026-10-30 実行済み・CLI 2.x が必要）

- PATH の `kaggle` は **1.7.4.5**（Python 3.9）。2.x は Python 3.10+ が必須で、
  3.9 では pip が最新版を選べない（`pip index versions kaggle` が 1.7.4.5 止まり）。
- Python 3.12 の venv で入れた **2.2.5** では削除が可能:

```bash
# 2.x を使う（PATH は変えない）
/opt/homebrew/bin/python3.12 -m venv ~/.venv-kaggle2
~/.venv-kaggle2/bin/pip install -U kaggle
~/.venv-kaggle2/bin/kaggle --version        # 2.2.5

~/.venv-kaggle2/bin/kaggle datasets delete <owner>/<dataset-name> [-y]
~/.venv-kaggle2/bin/kaggle datasets version -p <dir> -m "<notes>" [-d]  # -d で旧版削除
```

- `delete` は**データセット全体（全版）を不可逆に削除**。`dataset_sources` で
  参照しているカーネルはマウントできなくなる。`-d` は「アップロード時に旧版を
  消す」オプションで、事後の版指定削除は無い。
- 削除前チェック: `datasets files`（中身）/ repo 内 `dataset_sources` の参照
  （grep）/ `kaggle kernels list`（依存カーネル）。必要なら
  `datasets/` のローカル実体に退避してから。

削除済み（2026-10-30 実行、記録用・再実行しない）:

```bash
# venv は作成済み（2.2.5）。以下 3 件のみ実行し、一覧 12 → 9 件を確認:
# $K datasets delete yamahei21python/my-ollama-qwen2-5-coder-14b -y
# $K datasets delete yamahei21python/my-ollama-qwen3-6-35b -y
# $K datasets delete yamahei21python/my-ollama-gemma3-1b -y
```

`kawamura-finetune` / `kawamura-ruri-ft-data` は保持（依存カーネルあり）。
`qwen3-27b-uncensored-gguf` も保持（EXL3 実機検証後の判断待ち）。

### F-3 実行順

1. P0 修正（パスワード非出力・UI 認証・URL 配布）
2. Secrets 登録（`exl3-serve` 系）
3. GPU 実機確認（gpu-check → serve）
4. 退役判断 → データセット削除（2026-10-30 完了。#8〜#10 の 3 件のみ）
5. **生成物ポリシー（2026-10-30 決定）** — notebook は commit を継続
   （GitHub 差分レビュー・clone 直後 push を優先）。script の `main.py`
   （exl3 3 本 / lmdeploy / sglang）は base64 埋め込みでレビュー不能なため gitignore。
   `build/` は作らない。
   - 一致検証: notebook は `kaggle/check_notebooks.py`（12 冊）、
     script は `tests/test_exl3_build.py` / `tests/test_engines.py`（sha256 決定性）
     + 生成後の ruff/compileall。
   - ビルダーと生成物は同じコミットに含める。
6. **CLI 統合（完了）** — `cli/` に build / push / check / list / ask / quota。
   既定は build のみ。push は `--yes` 必須で、先に GPU/TPU 残量を表示する。
7. **spec 化（完了）** — `kbuild/spec.py` が id / 出力先 / ビルダー / env /
   datasets を持つ。ポートは `runtime/constants.py` に集約し、
   `tests/test_constants.py` が各リテラルとの一致を検査する。
8. **GPU スモーク計画（2026-10-30 作成）** — `notes/gpu-smoke-plan.md`。
   `--smoke` で install → 起動 → 会話 → 停止だけを回し、CPU 検証済みの
   proxy / tunnel / Gist / keepalive は省略する。共通テストは実稼働確認後に 1 エンジンで。

## 第 2 次計画（2026-10-30 追記）— エンジン層を「同じ骨格・差はデータだけ」に寄せる

第 1 次計画の段 7f まで完了した時点で残っていた負債の調査結果と方針。
**エンジンごとの差（install / launch の中身）は残してよい。骨格と配管は 1 本にする。**

### 調査でわかったこと

1. **script テンプレートが 2 本に分裂し、既に挙動が食い違っていた** —
   `kbuild/serve_script.py`（LMDeploy / SGLang）と `kaggle/build_exl3_serve.py`
   の TEMPLATE。共通関数 21 本のうち**完全一致は 2 本だけ**（`save` / `tail`）。
   - `verify()`: serve_script は空応答で `raise`、exl3 は**空文字のまま成功扱い**
     → 無言で空を返すモデルを `ok: true` と報告し得る。
   - `secret()`: serve_script は env 優先、exl3 は **env を見ない**
     → `EXL3_API_*` を環境変数で渡す経路が無視される。
   片側にしか P0/P1 の修正が入らない構造そのものが問題。
2. **`runtime/` にモデル配置とコンテキスト予算が無い** — 計画の `probe.py` /
   `model.py` は未作成。生成コードの `_find_input()` が 4 変種、
   `_fits()` が 2 変種、`_open_public()` が 2 変種で残っている。
3. **vLLM dryrun と本体で pip オプションが食い違う** — `_ta_check()` の
   torchaudio 修復で、本体だけ `_pip_extra`（index-url 等）を渡す。
4. **リポジトリ重量** — `.git` 3.0 GB（履歴に `kaggle/wheelhouse-files/` の
   wheel: torch 502 MB ほか）、`notes/uzu-backup/` に 52 MB の Mach-O と
   19 MB の tokenizer が**現役でトラッキング**、作業ツリー 41 GB、
   ルートに `out.*` 64 ディレクトリ。
5. 小さな負債 — `build_llama_server.py:43` だけ `mkdir` を自前で呼ぶ
   （他は `pack_nb.write`）、`cli/ask.py` の `post()` は `kc_post_json()` と同機能、
   README のテスト数（78 → 実測 81）とノート数（12 → TARGETS 13）が古い。

### 方針（今回着手する範囲）

**A. script カーネルのテンプレートを 1 本化する（本命）**

`kbuild/serve_script.py` を唯一のテンプレートにし、`build_exl3_serve.py` から
590 行の TEMPLATE を削除する。エンジン差は次の 2 つだけに閉じ込める。

| 差の表現 | 中身 |
|---|---|
| `engine_code`（文字列） | `stage_install()` と `launch(model_dir)` の実装だけ。exl3 は wheel 選択 + chat_server 起動、LMDeploy / SGLang は pip 1 回 + 外部サーバ起動 |
| `flow`（真偽値の表） | 段の有無。`install` / `model` / `tp` / `keepalive` / `publish` / `verify_public` / `conf` |

骨格（`save` / `sh` / `http_json` / `wait_http` / `secret` / `stage_env` /
`find_model` / `write_aux` / `start_proxy` / `verify` / `verify_public` /
`find_cloudflared` / `start_tunnel` / `publish_url` / `wait_public` /
`read_conf` / `loop` / `stop_all` / `main` / `run`）は全カーネルで同一。

- 併せて 2 つの drift を解消する（`verify` は空応答で失敗、`secret` は env 優先）。
  これは「統一の副産物」ではなく**統一しないと直らない**類の欠陥。
- 生成物は従来どおり `main.py` 1 ファイル + base64 埋め込み。sha256 決定性は維持。
- `kbuild/secrets.py` の台帳は、Secret 名をエンジン辞書の
  `secret_user` / `secret_pass` に持たせるため `tests/test_secrets.py` の
  検出パターンも 1 本追加する。

**B. 併せて直す小物**

- `build_llama_server.py` の `OUT_DIR.mkdir()` を廃止し `pack_nb.write` に寄せる
  （`parents=True` が無く、唯一の例外になっている）。

**C. 次回以降（今回は着手しない）**

- `runtime/storage.py` に `kc_find_dataset()`、`runtime/proc.py` に
  `kc_context_budget()` を足して `_find_input` 4 変種 / `_fits` 2 変種を一本化。
- `_open_public` を `runtime/net.py` の `kc_urlopen_v4()` へ。
- `.git` 3 GB の回収（`git filter-repo` は履歴書き換えなので単独作業・要承認）。
- `notes/uzu-backup/` の 52 MB バイナリと 19 MB tokenizer を HEAD から外す。
- README の数値更新。

### 受け入れ条件（CPU のみ・GPU / クォータ不要）

1. `python3 -m pytest tests -q` が green（`test_exl3_build` / `test_engines` 含む）。
2. 3 モード（exl3 / mock / keepalive）+ 2 エンジン（lmdeploy / sglang）の
   生成物が `ast.parse` を通り、再生成が決定的。
3. 生成物に `kc_*` 呼び出しが残り、`api_pass` の小文字キーが無い。
4. `ruff check .` が green（生成物にも掛ける）。

### 進捗（2026-10-30 実施）

- **A 完了** — `kbuild/serve_script.py` 1 本に統合済みであることを確認。
  exl3 は `engine_code`（`stage_install` / `launch`）だけを持ち、骨格 21 本は
  serve_script 側。drift 2 件（`verify` の空応答は失敗 / `secret` は env 優先）も
  serve_script 側で解消済み。`lmdeploy` / `sglang` の `main.py` は同一テンプレートの
  2 レンダリング（argv だけ `runtime/engines/*.py` の純関数）。
- **B 完了** — `build_llama_server.py` の `mkdir` を `parents=True` に。
  （この環境の Python シムは `exist_ok=True` でも EEXIST を投げるため、`cli check` が
  build で落ちていた）
- **C のうち 3 点完了** — ノートブック系に残っていた 4 変種 / 2 変種 / 2 変種を
  `runtime/` に一本化した。
  - `runtime/storage.py: kc_find_dataset()` ← `_find_input` 4 変種
    （step2 / direct / vllm / dryrun）
  - `runtime/proc.py: kc_context_budget()` / `kc_context_gib()` ← `_fits` 2 変種
  - `runtime/net.py: kc_urlopen_v4()` ← `_open_public` 2 変種
  生成物側は `_open_public = kc_urlopen_v4` の別名 1 行だけになり、実装を持たない。
- **統一の副産物として実バグを 1 件修正** — cloudflared 版のノートブックは
  keepalive セルが `probe_url()` を呼ぶのに、定義は ngrok セルにしか無かった
  （`NameError` が `except` に吸われ、トンネル再起動の経路だけ無言で死んでいた）。
  両トンネルセルが同じインターフェース（`TUNNEL_NAME` / `TUNNEL_LOG` /
  `RESTART_CMD` / `probe_url`）を公開する形に揃えた。
- **再発防止** — `tests/test_step2_builder.py::test_cell_name_order` の対象を
  1 冊（vLLM）から全 13 冊へ拡張（IPython マジック行は除去してから AST 解析）。
  上記バグはこの検査で検出した。

### 残り（次回以降）

- ノートブック系の **tail セル**（keepalive / verify）はまだビルダーごとに持っている。
  `_read_keepalive_conf` ×6 / `_proc` ×5 / `_judge` ×5 / `_with_retry` ×5 が重複。
  script 側の `serve_script.py` に相当する `kbuild/serve_notebook.py` を作るのが次の一手。
  ただしエンジン差（`/api/ps` の有無、`vllm serve` の argv 照合、再起動対象）が
  分岐だらけになりやすいので、**セル単位で切り出す範囲を決めてから**着手する。
- `runtime/engines/{vllm,llama_server}.py` は作らない判断を維持（起動ラダーは
  共有しない）。共有するのは「GPU 判定 → VRAM → コンテキスト候補」までで、
  これは `kc_context_budget` で完了している。

### 進捗（2026-10-30）

- **A 完了** — `kbuild/serve_script.py` を唯一のテンプレートにした。
  `build_exl3_serve.py` は 694 → 236 行（590 行の TEMPLATE を削除）。
  エンジン差は `engine_code`（`stage_install` / `launch` のみ）と
  `flow`（7 つの真偽値）に閉じた。
  生成物の構造一致: exl3 / lmdeploy とも 49 関数・共通 46・**完全一致 42**
  （統一前は共通 21 のうち一致 2）。差は `stage_install` / `launch` と、
  `stop_all` の 1 行（停止パターンのデータ）だけ。
  - drift 2 件を解消: `verify()` が空応答で `RuntimeError`、`secret()` が env 優先。
  - Secret 名はエンジン辞書の `secret_user` / `secret_pass` に移し、
    `tests/test_secrets.py` に辞書キーの検出パターンを追加（台帳も更新）。
- **B 完了（既に作業ツリーにあった）** — `build_llama_server.py:43` は
  `mkdir(parents=True, exist_ok=True)` 済み。
- **ついでに直した実バグ 2 件**
  - `build_engine_serve.py` の SGLang `knobs` が JSON の `true` のまま
    Python dict リテラルとして埋め込まれ、生成 `main.py` の読み込み時に
    `NameError` になっていた（`True` へ修正。`result.json` へは
    `json.dump` が `true` に直す）。
  - `ruff.toml` の per-file-ignores が `kaggle/exl3-serve/*/main.py` と
    綴り違いで効いておらず、生成物の `F811`（runtime 連結による import 重複）が
    CI の「generated outputs」ステップで落ちていた。`exl3-serve-kernel` /
    `lmdeploy-serve` / `sglang-serve` に修正（`F821` は除外しない）。
- **C は別作業で進行中** — `runtime/net.py` に `kc_urlopen_v4`、`storage.py` に
  `kc_find_dataset`、`proc.py` に `kc_context_budget` が入りつつある。
  この影響で notebook 12 冊が未再生成（`python3 -m cli build` で追従）と、
  `tests/test_step2_builder.py` のセル順テストが一時的に赤になる。
  エンジン層の変更は notebook に影響しない（差分は runtime 追加分のみ）。

## 第 3 次計画（2026-10-30 追記）— notebook → script への移行

### 目的（4 段階に分ける）

1. 当面は **notebook と script の二重持ち**（notebook が現役、script は検証用）
2. **script を生成できるようにする**（同じセル列から）
3. **notebook をアーカイブ**（`kaggle/archive/` へ移動）
4. **script が Kaggle で動いたら notebook を削除**

「いきなり置き換える」のではなく、**①生成できる → ②動く → ③消す** を別の関門に
する。各関門は無料（CPU）で越えられるように設計する。

### 現状の実測（2026-10-30）

| 項目 | 数 |
|---|---|
| notebook カーネル（retired 除く） | 12 |
| script カーネル（retired 除く） | 10 |
| commit 済み notebook ファイル | 15（smoke 版 1・archive 1 を含む） |
| notebook の生成コード量 | 81〜2,158 行 / 冊、合計 約 12,600 行 |
| pytest | **90 passed**（`test_step2_proxy` は稀に flake。単独では pass） |

移行対象と規模:

| slug | GPU | セル(コード) | 生成 loc | ビルダー | 状態 |
|---|---|---|---|---|---|
| `prepare-vllm-dataset` | – | 7 (5) | 81 | `build_vllm_dataset` | active |
| `step1-orcarouter-qwen38-27b-dl` | – | 11 (5) | 122 | `build_step1` | active |
| `keepalive-test` | – | 9 (4) | 646 | `build_keepalive_test` | active |
| `step2-vllm-dryrun` | – | 18 (9) | 1,109 | `build_vllm_dryrun` | semi |
| `step2-...-cloudflared` | GPU | 24 (12) | 1,729 | `build_step2` | active |
| `step2-...-ngrok-fixed` | GPU | 24 (12) | 1,758 | `build_step2` | active |
| `llama-server-ngrok` | GPU | 22 (11) | 1,558 | `build_llama_server` | semi |
| `step2-vllm-...-cloudflared` | GPU | 24 (12) | 2,129 | `build_step2_vllm` | active |
| `step2-vllm-...-ngrok` | GPU | 24 (12) | 2,158 | `build_step2_vllm` | active |
| `smoke-orcarouter-qwen38-27b` | GPU | 14 (6) | 368 | `build_smoke` | semi |
| `bench-ctx-vs-speed-...` | GPU | 10 (4) | 390 | `build_bench` | semi |
| `bench-mtp-spec-decoding-...` | GPU | 12 (5) | 555 | `build_bench_mtp` | semi |

### 前提の確定（Phase 0・クォータ 0・実行なし）

#### 0-a カーネルの型を後から変えられるか（最重要・未検証）

計画全体の分岐点。**同じ id のまま notebook → script へ変えられるか**で、
Secrets の再登録（UI 手作業・カーネル id 単位）が要るかが決まる。

kaggle CLI 2.2.5 の実装を読んだ結果:

- `ApiSaveKernelRequest` は保存のたびに `kernel_type` を送っている
  （`kaggle_api_extended.py:7264`）。id 単位の upsert なので、型も上書きされうる。
- `--no-run` は `KernelExecutionType.QUICK_SAVE`（同 `:7281`）。
  **実行せずに版だけ作れる**ので、実験はクォータも時間も消費しない。

反証の可能性: サーバが型変更を黙って無視する場合、push は成功し、`.py` の中身が
notebook として保存され、**実行時に JSON パースで落ちる**。
つまり**終了コードだけでは判定できない**。登録後の型を一覧で確認する。

実験手順（捨てカーネル・CPU・実行なし）:

```bash
# 1. 捨てカーネルを notebook として作る（kernel_type=notebook, code_file=probe.ipynb）
kaggle kernels push -p /tmp/kc-probe-nb --no-run
# 2. 同じ id を script として push（kernel_type=script, code_file=main.py）
kaggle kernels push -p /tmp/kc-probe-py --no-run
# 3. どちらの型で登録されているかを見る
kaggle kernels list --user yamahei21python --kernel-type script   --search type-probe
kaggle kernels list --user yamahei21python --kernel-type notebook --search type-probe
```

判定と分岐:

| 結果 | 意味 | 以後の進め方 |
|---|---|---|
| `--kernel-type script` に出る | 型変更は通る | **同一 id で移行**。Secrets も履歴も維持 |
| notebook 側に残る / push がエラー | 型は固定 | 新 id `<slug>-script` を作り、`notes/kernel-inventory.md` のチェックリストで Secrets を再登録 |

#### 0-b `cli push` に `--no-run` を足す

`kaggle kernels push --no-run` を透過するだけ。実験と、移行後の「実行せず保存」に使う。

#### 0-c `cli push <name> --as script`

`script_out` から staging し、`kernel_type=script` のメタデータで push する。

### 方式 — なぜ「セル連結」から始めるか

いきなり `serve_script` の骨格へ入れようとすると**入らない**。実測:

| セルファイル | 行 | トップレベル文 | 関数 | 形 |
|---|---|---|---|---|
| `step2/cells/01_restore_binaries.py` | 205 | 41 | 1 | 逐次実行 |
| `step2/cells/03_ollama_serve.py` | 173 | 22 | 1 | 逐次実行 |
| `step2/cells/04_warmup.py` | 53 | 6 | 1 | 逐次実行 |
| `step2/cells/06_checks.py` | 129 | 3 | 2 | `engine_checks()` 済み |
| `step2-vllm/cells/00_settings.py` | 175 | 62 | 1 | 定数定義 |
| `step2-vllm/cells/01_install_vllm.py` | 209 | 40 | 3 | 逐次実行 |
| `step2-vllm/cells/03_vllm_serve.py` | 578 | 84 | 10 | 逐次実行（起動ラダー） |
| `step2-vllm/cells/04_warmup.py` | 60 | 25 | 0 | 逐次実行 |
| `step2-vllm/cells/06_checks.py` | 133 | 5 | 3 | `engine_checks()` 済み |

`serve_script` の `engine_code` が要求するのは `stage_install()` と `launch(model_dir)` の
**2 関数だけ**。逐次実行のセルはそのままでは入らない
（`03_vllm_serve.py` は 84 個のトップレベル文が `launch()` の外にある）。

そこで **段 A（機械的）→ 段 B（任意）** の 2 段にする。

#### 段 A: セル列をそのまま 1 ファイルへ（意味を変えない）

セル列（`Flow.cells()` でも各ビルダーの `cells` でも形は同じ）の code セルを順に
連結して `main.py` を作る。markdown セルは
`# ── 見出し` に落とし、節構造を diff で読めるようにする。IPython マジックは
`subprocess.run` に直す — **該当は `prepare-vllm-dataset` の 3 行だけ**（実測）:

```
!pip install -q huggingface_hub kaggle
    → subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                      "huggingface_hub", "kaggle"])
!du -sh {OUT_DIR}   → subprocess.run(["du", "-sh", OUT_DIR])
!ls -la {OUT_DIR}   → subprocess.run(["ls", "-la", OUT_DIR])
```

**根拠**: セル順＝実行順なので、グローバル参照はそのまま通る。差分は
「JSON → Python」だけなので、失敗したときに原因が移行作業に無いと言い切れる。

**効き所**: 12 冊すべてのビルダーが最後に `pack_nb.write(...)` / `write_cells(...)` を
**1 回だけ**呼ぶ。差し込み点はビルダーごとに 1 箇所で、カーネルごとの書き換えが要らない。

```
build_step1.py:194          pack_nb.write(OUT, cells)
build_step2.py              （serve_notebook.Flow → Flow.write）
build_llama_server.py       （同上）
build_step2_vllm.py:222     pack_nb.write(OUT, cells)
build_vllm_dataset.py:159   pack_nb.write_cells(NB_PATH, cells)
build_vllm_dryrun.py:736    pack_nb.write_cells(OUT, cells, "3.11.0")
build_keepalive_test.py:309 pack_nb.write(OUT, cells)
build_smoke.py:214          pack_nb.write(OUT, cells, cell_ids=True)
build_bench.py:234          pack_nb.write(OUT, cells, cell_ids=True)
build_bench_mtp.py:347      pack_nb.write(OUT, cells, cell_ids=True)
```

#### 段 B: `serve_script` の骨格へ寄せる（任意・カーネルごと）

段 A が動いてから。`stage_install()` / `launch(model_dir)` に切り出し、`flow` の
真偽値で段を選ぶ。ここで初めて「1,700 行の逐次コード」が「骨格 + エンジン差」になる。
**段 A だけで目的（script 化）は達成される**ので、段 B は優先度を下げてよい。

### 出力の置き方

`Kernel` に 2 フィールド足す（`smoke_out` と同じ形）。

```python
emit_script: bool = False    # 同じセル列から main.py も出す（二重持ち）
script_out: str = ""         # main.py + kernel-metadata-script.json の出力先
```

- 二重持ちの間は `kind="notebook"` のまま、`emit_script=True`。
- 置き場所は親の下の `script/`（例 `kaggle/step2/script/main.py`）。
  削除時に親の `.ipynb` を消して `script/` を昇格できる。
- smoke 版の script は `script_out/smoke-<tunnel>/` に派生（フィールドを増やさない）。
- メタデータは `kernel-metadata-script.json`（`code_file=main.py` /
  `kernel_type=script`）。`cli push --as script` はこれを読んで staging する。
  `metadata.write` は既に `title↔slug` を検査するので、0-a で別 id になった場合も
  title を変えれば通る。

### 検証の増設（無料）

| 追加 | 中身 |
|---|---|
| `kbuild/pack_script.from_cells()` | `(kind, source)` と dict セルの両方を受ける。markdown→コメント、マジック検査 |
| `kaggle/check_notebooks.py` → `check_outputs.py` | notebook 表 + script 表。生成 main.py と commit 済み main.py を比較 |
| `tests/test_flow_to_script.py` | 全カーネルで (1) `ast.parse` 通過 (2) セル順保持 (3) マジック行が残らない (4) 再生成が決定的 |
| 生成 main.py に ruff / compileall | CI の generated outputs 経路に載せる（既存の仕組みを流用） |

`check_notebooks.py` は「一時ディレクトリで再生成 → commit 済みと**バイト比較**」なので、
`TARGETS` に `main.py` の行を足すだけで script 側の一致検証になる（同じ表から
2 つの生成物が出るため、片方だけ古いという状態を検出できる）。
IPython マジック行を落として AST を見る `_py_only()` は
`tests/test_step2_builder.py:653` にあり、段 A の検査（マジック残存チェック）に流用する。

### 移行順

#### Phase 1 — CPU（クォータ 0・無料で完走できる）

| 順 | slug | セル | loc | 何を確かめるか |
|---|---|---|---|---|
| 1 | `prepare-vllm-dataset` | 5 | 81 | マジック変換・最小の通し |
| 2 | `step1-orcarouter-qwen38-27b-dl` | 5 | 122 | Dataset DL + upload 経路 |
| 3 | `keepalive-test` | 4 | 646 | proxy / tunnel / keepalive セルの連結 |
| 4 | `step2-vllm-dryrun` | 9 | 1,109 | mock エンジンで配管フル（proxy → tunnel → gist → keepalive） |

4 番まで通れば「runtime の共有セルは script でも動く」と言える。

#### Phase 2 — GPU 配信カーネル（smoke で 1 回ずつ）

| 順 | slug | 備考 |
|---|---|---|
| 5 | `step2-...-cloudflared` | `--smoke`（install → 起動 → 会話 → 停止）。Ollama 系の代表 |
| 6 | `step2-...-ngrok-fixed` | 5 とセル列が同じでトンネルだけ違う。**検証は 5 の 1 回で足りる** |
| 7 | `llama-server-ngrok` | semi。llama-server 直起動 |
| 8 | `step2-vllm-...-cloudflared` | `--smoke`。vLLM の起動ラダー |
| 9 | `step2-vllm-...-ngrok` | 8 と同じ。トンネル差のみ |

#### Phase 3 — 測定カーネル（最後・据え置きも可）

`smoke-orcarouter` / `bench-ctx` / `bench-mtp`。semi で滅多に回さないため
script 化の便益が小さい。**移行しない判断もありうる**（その場合は
`emit_script=False` のまま notebook を残す）。

### 削除の条件（notebook を消してよい条件）

1. script カーネルが Kaggle で `ok: true` で完走
2. `result.json` の証跡が notebook 版と同値（`verify.has_text` / `url` / `load_seconds`）
3. `cli check` が green（notebook 表から外した状態で）
4. `.ipynb` を `kaggle/archive/` へ移動（**即削除しない**）→ 1 週間後に削除

`kaggle/archive/kaggle-ollama-ngrok.ipynb` に前例がある。削除は 1 カーネルずつ
（一括 `git rm` をしない）。

### リスク

| リスク | 影響 | 対処 |
|---|---|---|
| 型変更が通らない | 12 冊分の新 id + Secrets を UI で再登録 | Phase 0-a で先に判定。不可なら Phase 1 の 1 冊目で id 増の手間を実測してから進む |
| 連結 main.py が 1,700〜2,100 行のまま | 構造は良くならない | 段 B を後続。まず「動く script」を得る |
| `!` / `%` マジックの取りこぼし | 実行時に `SyntaxError` | `test_flow_to_script.py` で全カーネル検査（該当は 3 行のみ） |
| 生成 main.py に base64 の巨大 1 行 | diff レビューが効かない | 該当行だけ `.gitattributes` で `-diff`。他は読める |
| `test_step2_proxy` が稀に落ちる | CI が赤く見える | ポート競合の既知 flake（単独では pass）。リトライで隠さず記録だけする |

### 見積り

- 段 A は**共通層だけ**（`from_cells` / `Flow.write_script` / spec 2 フィールド /
  cli 2 フラグ / テスト 1 本）。**カーネルごとの書き換えは 0**。
- 段 B は 1 カーネルあたり切り出し作業（vLLM が最大）。
- 移行の実コストはほぼ **Phase 2 の GPU 待ち時間**（smoke 1 回 20〜40 分 × 5）。

### 先に決めてほしいこと

| # | 論点 | 推奨 |
|---|---|---|
| 1 | Phase 0-a（型変更の実験）を最初にやるか | **やる**。無料・実行なしで、以後の設計が変わる |
| 2 | 二重持ちの間に `main.py` も commit するか | **する**。差分レビュー手段が notebook JSON しか無い状態を早く脱する（base64 行だけ `-diff`） |
| 3 | Phase 3（smoke / bench 系）を移行するか | **据え置き可**。semi で滅多に回さないため便益が小さい |
| 4 | 段 B（骨格への切り出し）まで今回やるか | **段 A の後**。まず「動く script」を得てから |



## 第 3 次計画（2026-10-30 追記）— notebook 配信も「同じ骨格・差は 4 セルだけ」に寄せる

第 2 次の残件（tail セルの重複）を潰した記録。**script 側は `serve_script.py` で
1 本化済みだが、notebook 側（Ollama / llama-server / vLLM）は 3 本が別実装のまま
だった。**「次に試したいエンジンが来たときにバラバラだと気持ちが悪い」が動機。

### 方針

`kbuild/serve_notebook.py` の `Flow` を唯一の定義にし、実体は
`kaggle/serve-cells/*.py` に置く。各ビルダーが渡すのは **エンジンで本当に違うもの**だけ。

| 渡すもの | 中身 |
|---|---|
| `settings` | 名前空間の定義（ポート・モデル・Secrets 既定値・能力表） |
| `install` / `launch` | 依存導入 / 推論サーバ起動 |
| `verify_checks` | `engine_checks(judge, retry, base)` の実装 |
| `warmup` / `pre_launch` | 任意。ウォームアップと、install と proxy の間に挟むセル |
| `title` / `*_md` | 見出し文言（既定値あり） |

セル順は固定（`Flow.cells()`）:

```
title → 共通ランタイム → settings → install → [pre_launch] → proxy
      → launch → [warmup] → tunnel → gist → verify_checks → verify
      → 接続情報 → keepalive
```

`serve-cells` の共有セルが読む名前は **契約**として `serve_notebook.py` の
docstring に列挙した（`PORT` / `UPSTREAM_PORT` / `PROXY_*` / `MODEL_*` /
`ENGINE_*` / `GIST_*` / `MAX_MINUTES` / `KEEPALIVE_RESET_MINUTES` /
`TUNNEL_CHECK_EVERY` / `CLEANUP_TARGETS` / `OUT_DIRS` / `save_output` / `fail` /
`engine_cleanup()`）。新しいエンジンを足すときに増やすのはこの範囲だけ。

### エンジン差の表現（データだけ）

| 差 | 表現 |
|---|---|
| 死活監視の対象と照合方法 | `ENGINE_PROCS = [("ollama", "exact"), ("auth_proxy.py", "argv")]` |
| 後片付け | `engine_cleanup()` … Ollama は VRAM 返却、vLLM はキャッシュ退避、llama-server は `None` |
| プロキシの中継先・追加引数・検査パス | `UPSTREAM_PORT` / `PROXY_EXTRA_ARGS` / `PROXY_CHECK_PATH` |
| 停止対象 | `CLEANUP_TARGETS` |

`engine_unload()` という「定義されていれば呼ぶ」任意フックは廃止した。
`globals()` ガードで呼ぶと、定義が無い notebook で **未定義名の検出
（`test_cell_name_order`）をすり抜ける**ため、必須フック 1 本に統一した。

### 実施結果

| ファイル | 前 | 後 |
|---|---|---|
| `kaggle/build_step2.py`（Ollama） | 493 行 | 276 行 |
| `kaggle/build_llama_server.py` | 966 行 | 249 行 |
| `kaggle/build_step2_vllm.py` | 1,933 行 | 252 行 |
| 合計 | **3,392 行** | **777 行** |

新設した共有層は `kbuild/serve_notebook.py` 135 行 + `kaggle/serve-cells/` 561 行。
エンジン固有セルは `kaggle/step2{,-direct,-vllm}/cells/` に実ファイルとして分離
（ruff が掛かる = 構文と未定義 import を CI で見られる）。

- 3 系統に **3 部ずつ重複していた配管**（proxy / tunnel×2 / gist / verify /
  接続情報 / keepalive ≈ 550 行）が 1 部になった。
- `kaggle/step2/cells/` の `02_proxy` / `05_cloudflared` / `06_verify` /
  `07_connection_info` / `08_keepalive` は共有セルへ移して削除。
- ついでに解消した重複: `build_llama_server.py` の「モデル blob 解決」セルは
  直後の launch セルと**同じ処理の二重実装**だった（前者は `DATASET_PATH/blobs`
  を直読み、後者は patch Dataset の config blob も合流させる正しい版）。前者を削除。
- 挙動の統一（意図的）:
  - 全カーネルが公開 URL の疎通を `kc_urlopen_v4` 経由で行う（IPv4 フォールバック）。
    vLLM だけ `_open_public` 別名を持っていた非対称を解消。
  - `kc_read_conf(..., quiet_clamp=True)` に統一（上限で頭打ちの警告を毎分出さない）。
  - llama-server 直起動版にも Gist 配布セルが入る（従来は無く、トンネル再起動時に
    固定 URL が更新されなかった）。
  - 直起動版の疎通確認はローカル (`127.0.0.1`) から公開 URL 経由に変わり、
    `bench_direct.json` は `save_output` 経由で `/kaggle/working` と `/output` の両方に残る。
  - vLLM の `VLLM_KEEPALIVE_MIN` 環境変数は廃止（既定は `MAX_MINUTES=30` で同じ）。

### 再発防止（`tests/test_serve_notebook.py`）

「1 本化が崩れたら赤くなる」テストを追加した。セル単位では正しい Python なので、
**重複が戻っても既存テストは通ってしまう**のが怖いところ。

1. 共有セルが 3 系統に **バイト一致**で埋め込まれている
2. 配管セルの順序が全系統で同じ
3. **共有セルにエンジン名が無い**（`ollama` / `vllm` / `llama-server` / `lmdeploy` …）
4. `ENGINE_PROCS` / `CLEANUP_TARGETS` / `engine_cleanup` の定義が各 1 か所
5. settings セルが契約名 24 個をすべて定義している

### 検証（CPU のみ・GPU クォータ消費 0）

- `python3 -m cli build` 全 15 ビルダー OK / `ruff check .` green
- `pytest tests -q` **90 passed**（新規 5 本を含む）
- `kaggle/check_notebooks.py` 13 冊すべて一致
- `compileall` OK
- 追加検証: 6 冊（Ollama×2 / direct / vLLM×2 / smoke）のコードセルを 1 モジュールに
  連結して `ruff --select F821` → **未定義名 0**。セルをまたぐ名前解決まで壊れて
  いないことを確認した（関数本体の中は `test_cell_name_order` では見えないため）。

### 未着手（据え置き）

- `kaggle/build_vllm_dryrun.py`（CPU の mock カーネル）と `build_keepalive_test.py` は
  `Flow` を使っていない。**配信カーネルではない**ので、配管の共有対象外と判断。
  ただし keepalive セルだけは共有セルと重複が残っている（CPU テスト専用のため優先度低）。
- `runtime/engines/{vllm,llama_server}.py` は引き続き作らない（起動ラダーは共有しない）。



## 第 4 次計画（2026-10-30 追記）— notebook → script への移行（実施記録）

「notebook 型を捨てて全部 1 本の骨組みにしたい」という方針で着手し、
**notebook は 1 冊も削除せずに script を並走**させるところまで完了した。
第 3 次で notebook 側を `Flow` に寄せたので、script 側は `serve_script.py` の
1 本だけになり、**両方の真実源が同じセル列**になった。

### 制約（先に決めたこと）

| 論点 | 決定 |
|---|---|
| notebook を消すか | **消さない**。`.ipynb` は 1 冊も削除していない（`git status` で確認済み） |
| 新しい id を切るか | **切る**（`…-script`）。notebook 側の Secrets / dataset_sources を壊さないため |
| 生成物の置き場 | 親の下の別名（`kaggle/step2-script/`）と子ディレクトリ（`kaggle/step2/script/`） |
| GPU 実測 | まだ 0 回（クォータ待ち）。CPU で通るところまでを今回の完了条件にした |

### script の 3 形態

| 形態 | 出力 | 作り方 | 使う場面 |
|---|---|---|---|
| 骨格版 | `kaggle/step2-script/{cloudflared,ngrok}/`<br>`kaggle/step2-vllm-script/…`<br>`kaggle/llama-server-script/ngrok/` | `kbuild/serve_script.py` に `stage_install()` と `launch(model_dir)` を渡す | 配信カーネル（Ollama / vLLM / llama-server） |
| セル連結版 | `kaggle/{step1,step2,step2-vllm,step2-vllm-dryrun,smoke,bench,bench_mtp}/script/` | `build_batch_script.py` が `cells` をそのまま連結（`SERVE=False`） | 配信しないカーネル（データ準備 / 検証 / ベンチ） |
| 旧 script | `kaggle/exl3-serve-kernel/*` など | 骨格化前の実装（`.gitignore`） | EXL3 / LMDeploy / SGLang |

セル連結版を骨格に載せなかった理由: セル列は**トップレベルの逐次コード**で、
骨格の契約（`stage_install()` / `launch()` の 2 関数）に収まらない。無理に
書き換えると「セルの中身には手を触れない」という前提が崩れるので、
`textwrap.indent` で機械的に包むだけにした（`pack_script.wrap_cell`）。

### 骨格（`kbuild/serve_script.py`）に足したもの

| 追加 | 中身 |
|---|---|
| `$SETTINGS` スロット | 旧 settings セル。骨格が読む名前（`PORT` / `ALIAS` / `CONTEXT_LENGTH`）を上書きできる |
| `SERVE` / `WARMUP` / `CLEANUP` フラグ | `False` なら配信しない単発モード。`warmup()` / `cleanup()` の呼び出しを制御 |
| `engine_checks(judge, retry, base, auth=None)` | 疎通確認の**判定項目だけ**をエンジンが持つ。`verify()` は共通の足場 |
| `warmup()` / `cleanup()` の no-op 既定 | スロットが空でも `F821`（未定義名）を出さない。ruff の検査は弱めない |

`verify()` の共通化で、exl3 / lmdeploy / sglang / ollama / vllm / llama-server の
6 エンジンが**同じ判定の体裁**（`data[key]["checks"]` に `judge(idx, label, ok, detail)`）
で結果を残すようになった。

### 設定セルの一元化（`kbuild/settings_cell.py`）

notebook と script が**同じ `kaggle/step2/cells/00_settings.py` を読む**。
`__MODEL__` / `__TUNNEL__` / `__PROXY_B64__` などを最長一致で置換し、置換漏れは
例外にする。

これは**実際にバグを 1 つ捕まえた**: `env` と `OLLAMA_BIN` を settings セルへ移した
直後、notebook 側の settings セルがその 2 つを定義しておらず、後続セルが
未定義名を参照していた（`test_cell_name_order` が検出）。両方が同じファイルを
読むようにしたことで「片方だけ直して片方が古い」状態が起きなくなった。

### 台帳への追加（`kbuild/spec.py`・12 件）

`…-script` 5 本（Ollama ×2 / vLLM ×2 / llama-server ×1）と
`…/script` 7 本。すべて `status=semi`（GPU 実測前）。
`Kernel.out` はリポジトリ相対の文字列、`k.dir` は絶対 `pathlib.Path` という
非対称があるので、パスを組み立てるときは `d.relative_to(ROOT)` を通す。

### script 側の一致検査（`kaggle/check_scripts.py` + `kaggle/_regen.py`）

`check_notebooks.py` の script 版。**台帳から対象を引く**ので、script カーネルを
増やしても一覧の更新が要らない（18 カーネル / 46 ファイル。うち smoke 5 本 × 2）。

共通部は `kaggle/_regen.py` に寄せた（`stage` / `run` / `label` / `diff` / `report`）。
`check_notebooks.py` 側には**写した生成物を消す処理が無かった**ので、
ビルダーが黙って書き損ねると古いファイルと一致して偽の成功になり得た。
`_clear_outputs()` を両方に効かせて塞いである（消すのは `code_file` と
`metadata_file` だけ。`cells/` は触らない）。

負のテスト（`kaggle/ci-cpu/main.py` に 1 行追記）で
`❌ … ← build_ci_cpu で再生成してください` が出ることを確認し、元に戻した。

### 検証（CPU のみ・GPU クォータ消費 0）

- `python3 -m cli build` … 18 ビルダー OK
- `ruff check .` … green（`kaggle/*/script/main.py` はセル連結の必然で F811/F401 のみ許可。
  **F821 は許可していない**）
- `pytest tests -q` … **109 passed**
- `check_notebooks.py` … 13 冊一致
- `check_scripts.py` … 18 カーネル / 46 ファイル一致（smoke 5 本を含む）
- `compileall` … OK（script 生成物 14 ディレクトリ）

### 残タスク

#### A. 先に潰すべき穴 → **2026-10-30 に両方解消済み**

> ⚠ **A-1 は 2026-11-01 に撤回した。** ngrok 自体を廃止したので、骨格が ngrok を
> 実装している必要が無くなった（`notes/url-publishing.md`）。以下は「当時こう
> 直した」という記録として残す。現在の骨格は `start_tunnel()` が
> `start_cloudflared()` へ渡すだけで、`STEP2_TUNNEL` に cloudflared 以外を渡すと
> **生成前に落ちる**。`kaggle/serve-cells/tunnel_ngrok.py` は削除済み。

**A-1（解消 → 撤回）script 骨格が ngrok に対応していなかった。**
`start_tunnel()` が cloudflared 固定で、`TUNNEL_KIND` は `result.json` に書くだけ
だった。ngrok 版は `my-cloudflared-bin` を dataset から外しているので、
cloudflared を毎回ダウンロードして quick tunnel を張っていた（動くがラベルが嘘）。

対応内容（`kbuild/serve_script.py`）:

| 追加 | 中身 |
|---|---|
| `TUNNEL` 定数 | `tunnel=` で骨格に渡す（既定 `cloudflared`。未知の値は `ValueError`） |
| `NGROK_DOMAIN` 定数 | 固定ドメインの既定値（`NGROK_DOMAIN_DEFAULT` とは別名にして、settings との二重定義を回避） |
| `start_cloudflared()` / `start_ngrok()` | 既存の cloudflared 実装を切り出し、ngrok を移植 |
| `_wait_tunnel()` | URL 待ちと失敗時の理由付けを共通化 |
| `start_tunnel()` | `TUNNEL` で分岐する**唯一の場所** |
| `data["tunnel"] = TUNNEL` | 設定セルが何を名乗っていても、実際に張った方を正とする |

ngrok 実装は notebook 側（`kaggle/serve-cells/tunnel_ngrok.py`）と同じ知見を移植:
`NGROK_TOKEN`（Secret）で `add-authtoken` → ドメインは env → Secret → 既定値 →
`--pooling-enabled`（固定ドメインの `ERR_NGROK_334` 回避）→ `--url=https://…`
（`--domain` は deprecated）。`stop_all` の残存回収に `-x ngrok` も追加。

`kbuild/secrets.py` の `kbuild/serve_script.py` に `NGROK_TOKEN` / `NGROK_DOMAIN` を追加
（`tests/test_secrets.py` が台帳と参照の一致を見ている）。

**A-2（解消）script 版に smoke の逃げ道が無かった。**
`cli push --smoke` は `SMOKE_KERNELS` しか通さず、script ビルダーは `--smoke` でも
同じディレクトリへ書いていた（通常版を上書き）。

対応内容:

| 変更 | 中身 |
|---|---|
| `kbuild/spec.py` | `-script` 5 本に `smoke_out="kaggle/…/smoke-<tunnel>"` |
| 3 つの script ビルダー | `SMOKE` を先に確定し、`OUT` を smoke 側へ切り替える |
| `cli/build.py` | `SMOKE_BUILDERS` に 3 本、`SMOKE_KERNELS` に 5 本を追加 |
| `kaggle/check_scripts.py` | `smoke_out` を持つカーネルは `KC_SMOKE=1` でもう一度走らせ、smoke 側の生成物も突き合わせる（18 カーネル / **46 ファイル**） |
| `tests/test_spec.py` | `smoke_out` を持つカーネルのビルダーが `SMOKE_BUILDERS` に居ることを検査 |
| `tests/test_engines.py` | トンネル分岐 3 本（既定 / ngrok / 未知の値） |

これで GPU 初確認は 8〜12 分/本で回せる。

**A-3（据え置き）旧 script 3 系統（exl3 / lmdeploy / sglang）は smoke を同じ
ディレクトリへ書く。** これらは gitignore 済みの生成物で、smoke 専用ディレクトリを
増やすと**クリーン clone で `test_spec` が落ちる**（`smoke_dir.is_dir()` を要求して
いるため）。`cli push --smoke` が直前に再生成して押すので実害はなく、手元が
smoke 変種になった場合は `check_scripts.py` が不一致として検出する（`cli build` で戻る）。

#### B. GPU が要る（クォータ待ち）— **3 種 × 1 回・全種類**

| # | 種 | 内容 | 回数 |
|---|---|---|---|
| 1 | Ollama | `step2-orcarouter-qwen38-27b-cloudflared-script` を `--smoke` | 1 |
| 2 | vLLM | `step2-vllm-qwen38-27b-cloudflared-script` を `--smoke` | 1 |
| 3 | llama-server | `llama-server-ngrok-script` を `--smoke` | 1 |

**1 種につき 1 回**なので、全種類を試す（1 本通ったから他を省く、はしない）。
合計 3 回・約 1 時間。この 3 回で「骨格が Kaggle で起動して会話できる」までが確定する。

そのあと 1 回ずつ（別枠）:

| # | 内容 | 回数 |
|---|---|---|
| 4 | ngrok の実配線（`step2-…-ngrok-script` を `--smoke` **無し**で） | 1 |
| 5 | 配信 1 本で keepalive / `/stop` / UI まで通す | 1 |
| 6 | LMDeploy / SGLang の GPU 実測（従来からの宿題） | 2 |

#### B'. 改名（2026-10-30・CPU のみ）

llama-server 直起動の一族から `step2-` を落とした。`step2` は Ollama/vLLM の
段番号であって、llama-server の直起動はその段に属さない。

| 旧 | 新 |
|---|---|
| `step2-direct-llama-server-{tunnel}` | `llama-server-{tunnel}` |
| `step2-direct-llama-server-{tunnel}-script` | `llama-server-{tunnel}-script` |
| `kaggle/step2-direct/`（cells + cloudflared 出力） | `kaggle/llama-server/` |
| `kaggle/step2-direct-ngrok/` | `kaggle/llama-server-ngrok/` |
| `kaggle/step2-direct-script/` | `kaggle/llama-server-script/` |
| `build_step2_direct{,_script}.py` | `build_llama_server{,_script}.py` |
| `STEP2D_TUNNEL` / `_MTP` / `_CTX` | `LLAMA_TUNNEL` / `_MTP` / `_CTX` |

**Kaggle 側では新規カーネル**になる（旧 id は残る。GPU 実測前なので失うものは無い）。

#### C. 方針決定が要る（無料）

| # | 内容 | 選択肢 |
|---|---|---|
| 7 | `kernel_type` を同一 id のまま変更できるか（Phase 0-a） | `kaggle kernels push --no-run`（QUICK_SAVE・無料）で実験。不可なら移行は id 増 |
| 8 | script 生成物 15 ディレクトリ（約 6.4 MB・**現在未追跡**）を commit するか | commit（`check_scripts.py` がそのまま効く）／gitignore（exl3 方式の sha256 決定性へ） |
| 9 | notebook を `kaggle/archive/` へ移して削除してよいか | B の #1〜#3（3 種）と #5（配信）が通ってから。1 カーネルずつ（一括 `git rm` をしない） |

#### D. 据え置き（やらないと決めたこと）

- `build_vllm_dryrun.py` / `build_keepalive_test.py` は `Flow` を使わない（配信しないため）。
- `runtime/engines/{vllm,llama_server}.py` は作らない（起動ラダーは共有しない）。
- bench / smoke / bench_mtp の notebook 版は消さない（semi で便益が小さい）。

#### E. ルート直下の整理（2026-10-30 実行済み・CPU のみ）

`/Users/kohei/Myproject/kaggleAI` の直下が 118 エントリまで膨れていたので、
**追跡ファイルを 1 つも動かさずに**（＝ `git status` の M/D が増えない形で）片付けた。
結果は **39 エントリ**（うちドット付き 13）。

| やったこと | 中身 | 移動先 |
|---|---|---|
| 一時プローブ出力を退避 | `out/` + `out.*/` の **64 個**（4.7 MB） | `archive/probes/` |
| ルートのログを退避 | `*.log` **9 本**（4.5 MB）+ `diagout/` | `archive/logs/` |
| サンドボックスの残骸を削除 | 4 バイトの `mkstemp` 跡 5 個（`5p2uv2jp` など。中身は `blat`） | — |
| モデルをまとめる | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`（16.8 GB） | `models/`（同一ボリュームなので rename のみ・0.1 秒） |
| `.gitignore` に追加 | `archive/` `models/` `.benchmarks/` `.pytest_cache/` `.workbuddy-ai/` | — |
| `.DS_Store` を削除 | 再生成されるので都度消す | — |

- **`archive/` は追跡しない**（`.gitignore` 済み）。`kaggle/archive/`（旧カーネルの
  スナップショット・追跡する）とは別物。生の計測データを残すなら `notes/archive/`。
- `llama-chat.html` は**動かさない** — 複数のビルダーが読み込んでカーネルへ埋め込む
  原本なので、ルート直下が正しい置き場。
- `kcconfig.py` は `kaggle/build_*.py` が `import kcconfig` するので**必須**。
  `ask.py` / `kaggle-quota.py` は好みのシム（消しても `python3 -m cli …` で同じ）。
- `kaggle-dataset-*/`（17 GB + 1.4 GB + 38 MB）は `datasets/` へ集約した（下記 E-2）。

確認: `env -u PYTHONPATH python3 -m cli check` → build / ruff / pytest **114** /
notebooks 13 冊 / scripts 18 カーネル 46 ファイル / compileall すべて OK。

#### E-2. Dataset の置き場所（2026-10-30・同じく実行済み）

**結論: repo の外へは出さない。** `datasets/` に集約するだけにした。

| 旧 | 新 |
|---|---|
| `kaggle-dataset/`（symlink + metadata のみ） | `datasets/gguf/` |
| `kaggle-dataset-orcarouter/`（17 GB） | `datasets/orcarouter/` |
| `kaggle-dataset-orcarouter-patch/`（1.4 GB） | `datasets/orcarouter-patch/` |
| `kaggle-dataset-cloudflared/`（38 MB） | `datasets/cloudflared/` |

判断の理由:

- **容量は増えない** — 同一ボリューム（`/System/Volumes/Data`）なので `mv` は rename。
  repo 外へ出しても空きは 1 バイトも変わらない。
- **git は元から汚れていない** — `kaggle-dataset/` `kaggle-dataset-*/` は gitignore 済み。
- **`-p` は相対パスで叩く運用** — 外に出すと絶対パスか変数管理になり、
  シンボリックリンク越しの `--dir-mode tar` は CLI 1.7.4.5 で未検証。
  repo を移動するとリンクが切れる。

代わりに**容量の実利はハードリンクで回収した**（下記）。repo を消したくない／
バックアップから外したい、という動機が別にあれば `datasets/` を Time Machine の
除外に入れる方が安全（シンボリックリンク運用より事故らない）。

##### 16.8 GB の重複を 1 本化

`datasets/orcarouter/blobs/sha256-3445102e9cde…`（16,810,714,496 B）は
`models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf` と**完全に同一**だった
（sha256 が blob のファイル名と一致 = `3445102e9cde5d5625…`）。
Ollama は GGUF を無改変で model layer として保存するため、この重複は構造上必ず起きる。

```bash
B=datasets/orcarouter/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
ln models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf "$B.dedupe"   # 先に作る（無停止で安全）
mv -f "$B.dedupe" "$B"                                     # rename で旧 inode を解放
```

結果: `links=2` / 同一 inode。空き **22 GiB → 38 GiB**。
`ollama rm` で blob を消しても `models/` 側は残る（逆も同じ）。
**`cp -r` / rsync すると 2 個分に戻る**ので、コピー後の `du` が増えたらこれが理由。

##### 一緒に直したもの

| ファイル | 変更 |
|---|---|
| `kaggle/assemble_patch_dataset.py` | `DST` と docstring を `datasets/orcarouter-patch` へ |
| `notes/operations.md` | cloudflared Dataset 作成の 4 コマンド |
| `notes/dataset-inventory.md` | 退避先の記述と `kaggle datasets version -p` の例 |
| `.gitignore` | `kaggle-dataset/` `kaggle-dataset-*/` → `datasets/` |
| `README.md` | 構成ツリーと `datasets/` の説明 |

**注意（今回の副作用）**: `models/` へ GGUF を移した時点で
`kaggle-dataset/Qwen3.8-….gguf` の symlink がリンク切れになった（`datasets/gguf/` へ
移した際に `../../models/…` へ張り直し済み）。今後 GGUF を動かすときは
`datasets/gguf/` の symlink も一緒に直すこと。

#### E-3. `.git` の容量整理（2026-10-30 実行済み）

`.git` が **3.0 GB** あった。中身は `kaggle/wheelhouse-files/*.whl`（vLLM の
wheel 群、最大 526 MB）で、`.gitignore` 済みなのに**過去のコミットに残っていた**。

```
526 MB torch-2.13.0-cp313-...whl        326 MB vllm-0.32.0-...whl
439 MB nvidia_cublas-13.8.2.12-...whl   206 MB nvidia_nccl_cu13-...whl
366 MB nvidia_cudnn_cu13-...whl         ...
```

**原因は `refs/original/refs/heads/main`** — 以前 `git filter-branch` をかけたときの
バックアップ ref が残っていて、これだけが旧履歴（`7c8af6c` / `0170c7b`）を
到達可能にしていた。`origin` 側の履歴は既に書き換え済み（`origin/main` に
wheelhouse は 0 ファイル、初回コミットは `ed64c02`）なので、**ローカルにだけ
古い履歴が残っていた**状態。

```bash
git update-ref -d refs/original/refs/heads/main   # 旧履歴を生かしていた ref
git reflog expire --expire=now --all              # reflog の参照も切る
git gc --prune=now                                # 到達不能オブジェクトを破棄
```

結果 **3.0 GB → 29 MB**（`size-pack` 3.02 GiB → 28.49 MiB）。

確認したこと:

| 項目 | 結果 |
|---|---|
| `git status --porcelain \| wc -l` | **109**（実行前と同じ。作業中の変更は無傷） |
| `git log --oneline -3` | `b5718b6` ほか、履歴そのまま |
| `git branch -vv` | `main`（ahead 1）/ `refactor/p0-and-kbuild`（ahead 11）とも無傷 |
| `git fsck --no-progress` | 出力なし（健全） |

**失ったもの**: 旧履歴（`0170c7b` を含む pre-rewrite の系列）への参照。
`origin` に書き換え後の履歴があるので実害なし。wheel の中身は
`kaggle/wheelhouse-files/` を再生成すれば戻る（`prepare-vllm-dataset` が作る）。

再発防止: `kaggle/wheelhouse-files/` は `.gitignore` 済み。`git add -A` の前に
`git status` で巨大ファイルが混ざっていないか見る。

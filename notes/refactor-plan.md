# リポジトリ再編計画（2026-10-30 時点・検討メモ）

実装前の検討結果。対象は Kaggle カーネル群（Ollama / llama-server / vLLM /
exllamav3）とローカルのランチャー・クライアントまで。

## 現状の数字

| 項目 | 現状 |
|---|---|
| 追跡ファイル | 235 |
| カーネル用ディレクトリ | `kaggle/` 直下に 19 ディレクトリ + `build_*.py` 11 本（step2 系だけで 8 バリエーション） |
| ビルダー | `kaggle/build_*.py` 11 本、計 6,524 行（最大 `build_step2_vllm.py` 2,163 行）+ `exl3-serve/build_kernel.py` 619 行 |
| 退役コード | `kaggle/archive/` 121 ファイル（旧カーネル 46 スナップショット = json+ipynb 46 組、py 18 / sh 10）、`notes/uzu-backup/` 13 |
| ローカルランチャー | 4 本 計 1,000 行 + `lib/launcher-common.zsh` 214 行（0b 完了後） |
| クライアント | `ask.py` / `kaggle-quota.py` / `kcconfig.py` がルート直下 |
| テスト | `_selftest_proxy.py` / `_selftest_builder.py` / `exl3-serve/_selftest_server.py` + CI の notebook 一致（7 冊のみ） |

## 問題点

1. **ビルダーの重複** — base64 埋め込み・metadata 生成・nbformat 出力・push 手順が
   11 本にコピーされている。実測した関数重複:
   `_find_input` 4 箇所 / `_detect_gpu` 4 / `probe_url` 4（`build_step2_vllm.py`
   内ですら 2 回）/ `save_output` 3 / base64 埋め込みブロック 6。
   （生成 notebook 内にも各 6 コピーあるが、抽出元はビルダー側）
2. **ソース／生成物／ログの境界が曖昧** — notebook と builder の両方をコミットし、
   生成 `main.py`（`exl3-serve/{kernel,dryrun,keepalive}/`）は未追跡。
3. **命名で現役／退役が判別できない** — `step2`, `step2-vllm`,
   `step2-vllm-kernel/{ngrok,cloudflared}`, `step2-direct-ngrok`, `step2-ka-test`。
4. **実行時コードの真実源が暗黙** — `chat_server.py` / `auth_proxy.py` /
   `llama-chat.html` を各ビルダーが個別に埋め込んでいる。さらに生成される
   `main.py` の proxy 起動・トンネル・keepalive・verify が
   `build_step2.py` / `build_step2_direct.py` / `build_vllm_dryrun.py` /
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
| Kaggle カーネル | Ollama（`build_step2.py`）/ llama-server（`build_step2_direct.py`）/ vLLM（`build_step2_vllm.py`）/ exllamav3（`exl3-serve`） |
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
  step1/ step2/ step2-direct/ step2-vllm/ exl3-gpu-check/ exl3-serve/ ...
cli/                      # ask.py / quota.py / config.py
lib/                      # ローカルランチャー共通（launcher-common.zsh を拡張）
build/                    # 生成物（gitignore。push はここから）
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
`build_step2.py` / `build_step2_direct.py` / `build_vllm_dryrun.py` /
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
1. **棚卸し表** — 現役 / 準現役（`build_step2_direct.py`, `build_vllm_dryrun.py`）/
   退役（`step2-ka-test`, `capture_patch`, `archive/`）。カーネル id は変えない。
2. **kbuild 抽出 + 生成コードの一元化（最優先）** — `exl3-serve/build_kernel.py` と
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
3. **`build_step2.py`**（cells 外部化済みで最も楽）→ **`build_step2_direct.py`**。
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
   notes 再編、生成物ポリシー（`build/` + gitignore + CI 検証）、
   **README 再構成**（現行 1,255 行は旧パスを直書き。新レイアウトへ移し、手順詳細は
   `notes/` に分割する）。
   - 進捗: exl3 の単一 Template も runtime へ統合。proxy 起動 / tunnel URL 待ち /
     keepalive conf / `/stop` / GPU 検出 / 締切計算を `kc_*` 呼び出しへ置換し、
     `kc_start_proxy` に interpreter 指定（`python=`）を追加。exl3 の
     `_selftest_build` は埋め込み関数の存在と AST 構文を検証する。

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
| 6/7 テスト移設 | pytest 一式 + CI 追加、keepalive はフェイククロック | CI green |

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
  `exl3-serve/dryrun`（mock・約 5 分、cloudflared 公開 URL まで）と
  `exl3-serve-keepalive`（約 3 分、延長と `/stop`）を push し、`result.json` が
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

- ビルダー 6,524 行（+ `exl3-serve/build_kernel.py` 619 行）→ 共通 ~300 行 + spec 各 50–150 行。ビルダー層は **80% 前後**まで共通化可能（実測の重複表）。生成コード側（proxy / tunnel / keepalive / verify）の一元化が最も効く。
- `check_notebooks.py` の対象 7 冊 → 12 冊（direct / vLLM dataset / dryrun / vLLM 本体 2 種を追加）。比較は AST + 正規化、exl3 のみ byte。
- 現役／退役がディレクトリ構造で自明になる。

## 未考慮点の調査（2026-10-30 追記）

計画に無かった実務上の穴。P0 はリファクタより先に直す（特に 1〜3 は
`exl3-serve` の GPU 実行前に必要）。

### P0: セキュリティ / 機能の欠陥

1. **API パスワードが成果物に残る** — `build_kernel.py:286` が `api_pass` を
   `data` に入れ、`RESULT:` 行としてログにも出る。`kaggle kernels output` で
   取得できてしまう。認証を掛けている意味が半減する。
   → `result.json` は「認証 ON/OFF」だけにし、パスワードは Secrets 必須
   （ランダム時はログにも出さない）。
2. **チャット UI が認証に対応していない** — `llama-chat.html` に
   `Authorization` を付ける箇所が無い。`exl3-serve` は Secrets か乱数で
   `API_PASS` を必ず設定するため、UI から `/v1/*` が 401 になる。
   → UI にパスワード入力（sessionStorage）を足すか、認証 OFF を明示選択。
3. **URL の配布経路がエンジンで違う** — Ollama 版は Gist に `url.txt` を
   書く（`build_step2.py:438-503`）が、`exl3-serve` は `result.json` のみ。
   手元の `ask.py` は Gist 前提なので URL を拾えない。
   → spec に publish 段（gist / result / 手動）を持たせる。
4. **`/quota` がカーネルで動かない** — `auth_proxy.py:61` は
   `~/.kaggle/kaggle.json` しか読まないため、ビルダーは `--no-quota` を
   渡している。Kaggle では環境変数 `KAGGLE_USERNAME`/`KAGGLE_KEY` があるので
   env も読めるようにする。

### P1: 重複・再現性・検証

5. **クォータ API の実装が 2 つ** — `kaggle-quota.py:21,103` と
   `auth_proxy.py:61,304`。`runtime/quota.py` に一本化。
6. **定数の分散** — ポート番号（9931 / 11435 / 19950 / 8899）が 42 箇所、
   cloudflared の `releases/latest` が 9 箇所。後者はバージョン固定しないと
   再現性がない。→ `runtime/constants.py` と `CLOUDFLARED_VERSION`。
7. **シークレットの台帳が無い** — `NGROK_TOKEN` / `GITHUB_TOKEN` / `GIST_ID` /
   `KAGGLE_USERNAME` / `KAGGLE_KEY` / `HF_TOKEN` / `EXL3_API_USER` /
   `EXL3_API_PASS` が散在。spec に `required_secrets` を宣言し、CI で
   コード中の `get_secret("...")` と突き合わせる。
8. **CI と実行環境の差** — `ruff.toml` は `target-version = "py310"`、
   Kaggle は 3.13。生成物（`main.py`）は untracked で CI の対象外。
   → 生成物を `compileall` + ruff に掛ける段を追加。
9. **生成コードのテストが notebook だけ** — `_selftest_builder.py` は
   notebook から AST で関数を抜いて exec するが、script 系（exl3）は
   今回手動で 3 回走らせただけ。Mock で 20 秒の実行テストを追加。
10. **認証ありのプロキシテストが無い** — `_selftest_proxy.py` は
    「認証なし」で起動して検証している（docstring 明記）。401・
    `/stop` ヘッダ必須・チャンク本文の各経路を認証 ON でも回す。
11. **クライアント設定の二重管理** — `opencode.json` と
    `opencode-llama-local.json`、`ask.py` の Gist 依存、UI の `/quota` 依存。

### P2: 整理・運用

12. ルート直下の旧 notebook `kaggle-ollama-ngrok.ipynb`（4 セル・Ollama 版の
    初期版）と `notes/` 内のベンチログ・JSON をアーカイブへ。
13. `my-cloudflared-bin` Dataset の更新手順とライセンス表記（再配布物）。
14. 新しい kernel id を作るときのチェックリスト（Secrets はカーネル単位、
    `dataset_sources`、`title` の slug 一致）。
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
  `kaggle-dataset-*` のローカル実体に退避してから。

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
5. 生成物（`kernel/` `dryrun/` `keepalive/`）の commit / gitignore 方針を確定



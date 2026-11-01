# 公開 URL と進捗の伝え方（Gist → ビーコンカーネル）

**いまは両方入っている。** Gist（`runtime/gist.py` の `kc_gist_write`）は
残置で、`GITHUB_TOKEN` が読めたときだけ動く。**ビーコンは
`runtime/beacon.py` に実装済みで、serve の `publish_url` から必ず呼ばれる**
（`kbuild/serve_script.py` の `render()` が `beacon` モジュールを無条件で
足すので、ビルダーが書き忘れることがない）。

この文書は「なぜ置き換えたか」「何を実測したか」「どこまで確認できたか」の記録。

## 1. Gist 方式（残置・実質動かない）

- serve カーネルがトンネル確立後に `publish_url(u)` を呼び、Gist の
  `url.txt` を PATCH する
- 読み手（Mac 側）は Gist の raw URL をポーリングして現在の URL を知る
- `GITHUB_TOKEN`（scope は `gist`）と `GIST_ID` を Kaggle Secrets から取る。
  既定値は settings セルの `GIST_ID_DEFAULT` = 個人アカウント
  （`yamahei21python`）の Gist
- トークンが無いときは**配信を止めずにスキップ**する
  （`ℹ GITHUB_TOKEN 未設定 → 固定URL配布はスキップ`）


## 2. なぜ専用アカウント運用で使えなくなったか

**Kaggle の Secret は「アカウントに登録する」だけでは読めず、カーネルごとに
紐付ける必要がある。** 紐付けが無いときの `get_secret()` は
`ConnectionError: Connection error trying to communicate with service.` を
返す（通信障害に見えるが、実際は「このカーネルには 1 つも付いていない」）。

2026-11-01 に CPU プローブ `kaggle/cpu-tests/gist`（`gist-probe`）で実測:

| 確認したこと | 結果 |
|---|---|
| 新しく push したカーネルから `GITHUB_TOKEN` を読む | `token_present=false` |
| その理由 | `ConnectionError ...`（＝未紐付け） |
| 書き込みを試した結果 | `write_ok=false`、`write_detail` に同じ理由 |
| 存在しない Gist への書き込み（陰性対照） | `ok=false` かつ理由あり（＝失敗は失敗として返る） |

つまり **専用アカウントで完結させる方針とは両立しない**:
- Secrets の登録は Kaggle の UI でしかできない
- トークンは個人アカウントのものになる（＝個人に依存が残る）
- カーネルごとの紐付けを、カーネルを増やすたびに手でやることになる

この 3 点が「全て専用アカウントで完結」に反するので、代替を 6 案比較して
**C 案（ビーコンカーネル）** が選ばれた。

## 3. ビーコンカーネル（採用案・実装済み）

専用アカウントに `serve-url-beacon` を 1 つ置く。CPU・インターネット無し・
非公開。**実行するものではなく、カーネルのコード自体を掲示板として使う**:

```python
URL = "https://xxxx.trycloudflare.com"
STAGE = "serving"
UPDATED = 1762051200.123      # 発行時刻の epoch
```

- 書き手: serve カーネル内で一時ディレクトリに `kernel-metadata.json` と
  URL 入りの `main.py` を作り、**カーネルに注入される Kaggle 認証**
  （`KaggleApi().authenticate()`）で `kernels_push` する。Gist と同じく
  **失敗しても配信は続け**、成否は `result` の `beacon` フィールドに残す
- 読み手: `python3 -m cli beacon`（`cli/beacon.py`）。`kernels pull` して
  `URL` / `STAGE` / `UPDATED` をパースする。**`exec` しない**（ネットワーク
  越しの入力なので評価しない）
- 鮮度は `UPDATED` で判定する。**古いテスト URL を掴むバグが一度出た**ので、
  `--max-age`（既定 6 時間）と `STAGE=probe` の 2 つで既定で弾く

実装は 4 か所に分かれている:

| ファイル | 役割 |
|---|---|
| `runtime/beacon.py` | 本文生成・パース・鮮度判定・push/pull。**カーネル内と Mac で同じパーサ** |
| `kaggle/build_serve_url_beacon.py` | 掲示板を 2 枚作る（本番と検証用） |
| `kaggle/build_beacon_probe.py` | CPU プローブ（往復・鮮度・陰性対照の 5 項目） |
| `cli/beacon.py` | Mac 側の読み取り。古い値と `probe` を捨てる |

### 実測（2026-11-01・CPU カーネル `beacon-probe`）

本番を汚さないよう検証用の `serve-url-beacon-test` に対して実行:

| 項目 | 結果 |
|---|---|
| カーネル内 `kernels_push` → `kernels_pull` の往復 | **0.26 秒**（t=4.24 → 4.50、`ok=true`） |
| `URL` / `STAGE` の一致 | どちらも一致 |
| 鮮度（いま） | `true` |
| 鮮度（+1 時間） | `false`（＝古い値を弾ける） |
| 陰性対照（slug 空） | `ok=false` かつ理由あり |
| Secrets の有無 | **未紐付けのまま動いた**（`gist_selftest.json` は同じ実行で `ConnectionError`） |

これで「**Secrets を 1 つも紐付けないカーネルから掲示板を更新できる**」が
実測で埋まった。`vllm-playground` の `gist_selftest.json` は同じ日に
`token_present=false` / `ConnectionError` を返しており、Gist が動かない
理由と、ビーコンがそれを回避していることが同じログで並ぶ。

### Mac 側で踏んだ罠

`python3 -m cli beacon` が `No module named 'kaggle.api'` で死んだ。原因は 2 つ:

1. リポジトリ直下の `kaggle/`（カーネルのビルド元）が、`__init__.py` を
   持たないので**名前空間パッケージとして `import kaggle` を横取り**する
2. 手元の `python3`（3.13）には `kaggle` パッケージが入っておらず、
   `kaggle` CLI だけが 3.9 の user site にある

対策として `kc_kaggle_path()`（影のパスを外す）と、**CLI へのフォールバック**
（`kc_beacon_pull_cli`、`cli push` と同じ解決順で `kaggle` を探す）を入れた。
両方失敗したときは**両方の理由**を返す（片方だけだと原因を追えない）。


### 落選した案

| 案 | 落選理由 |
|---|---|
| Dataset に publish | バージョン反映が分単位で遅い |
| Google Drive 経由 | 別サービス・別アカウントが増える |
| 外部 KV / Webhook | 新しい外部依存を持ち込む |
| 固定 URL トンネル（ngrok 等） | トークンとアカウントが要る |

### 弱点・未検証

- 公開のたびにビーコンのバージョンが 1 つ増える（小さいので実害なし）
- **複数 serve の同時競合は想定していない**（単独運用前提）
- **serve 本体に組み込んだ `publish_url` から発火するところはまだ見ていない**。
  プローブで確かめたのは「カーネル内から push/pull が通る」ことまで。
  配信カーネルは `publish_url` をトンネル確立後にしか呼ばないので、
  起動が通った回が最初の end-to-end 確認になる
- 書き込みは `kernels_push` なので、**push できるのは自分のアカウントの
  カーネルだけ**。専用アカウントの掲示板を `yamahei21python` からは書けない
  （読むのは誰でもできる）

## 4. ビーコンが**カバーしていない**もの: 起動の進捗

ビーコンが運ぶのは**トンネル URL だけ**。vLLM の起動待ち（`notes/vllm-startup-speedup.md`
§20）のような「いま何をしているか」は運ばない。ここは Gist 時代から空白で、
いまも外から見えない。

しかも**その待ち時間は同じ日で 1.8 倍振れる**。2026-11-01 の実測:

| 回 | `Loading weights took` | 結果 |
|---|---|---|
| baseline v3 | 2,640 秒 | 予算 2,700 秒を 300 秒超過して ERROR |
| baseline v4 | **1,304 秒** | 2,145 秒で完走（`ok=true`） |

同じコード・同じ `KC_WARM_CACHE=0` でこの差なので、**外から進捗が見えないと
「詰まっているのか遅いだけか」を判別できない**。ここがビーコンの次の用途。

実験カーネル（`vllm-playground`）は個人アカウントにあるので、
**同じアカウントに status 用のビーコンを 1 つ置けば同じ手が使える**。
頻度は 30 秒ごとではなく「段が変わったとき + 5 分ごと」程度にしないと
バージョンが爆発する。

## 5. 現状

| 対象 | 状態 |
|---|---|
| URL（配信） | **ビーコンに移行済み**。`publish_url` が Gist とビーコンの両方を呼ぶ |
| URL（読み取り） | `python3 -m cli beacon` で読める。古い値と `probe` は既定で捨てる |
| 進捗（起動中） | **未着手**。§4 のとおり、まだ外から見えない |

### `runtime/gist.py` は残す（2026-11-01 判断）

消さない。理由は 3 つ。

1. **URL の経路としてはもう主役ではない**が、`publish_url` の片翼としては生きている
   （`GITHUB_TOKEN` が読める環境なら動く）。読み手の `cli/ask.py` と
   ランチャーの `gist_url` も同じ。
2. **進捗を運ぶ唯一の実装**。§4 の穴はまだ埋まっていないので、ここを消すと
   「設計も実装も無い」状態になる。
3. **`kaggle/step2-vllm/cells/03_vllm_serve.py` が `vllm_status.txt` を書いている**。
   これは起動待ちの現在地を外に出す唯一の口で、消すにはビーコンの
   ステータス版を先に作る必要がある。

ただし**このアカウントでは動かない**。`gist-probe`（2026-11-01 09:39 UTC）の実測:

```json
{"token_present": false,
 "token_detail": "Secret を読めない（ConnectionError: Connection error trying to
                  communicate with service.） — アカウントに登録済みでも、
                  カーネルごとの紐付けが無いと読めない",
 "negative_ok": true}
```

`negative_ok: true` は「壊れたときは必ず理由を返す」という設計が効いている証拠で、
黙って落ちることは無い。つまり**死んでいることが観測できる**。だから残しても
事故らない。次の一手は §4 のステータス用ビーコン（`runtime/gist.py` を消すのは
その後）。

### 進捗が見えないコスト（2026-11-01 22:40 実測）

`vllm-playground` の `cache` 実験を回している間、**外から状態を知る手段が
3 つとも空**だった:

| 手段 | 結果 |
|---|---|
| `kaggle kernels logs` | 空（実行中は返らない） |
| `kaggle kernels output` | 空（完了まで何も落ちてこない） |
| `kaggle kernels files` | `--no-run` で保存した**スタブ**（各 841〜865 B、9:47 UTC）だけ |

`kernels list --sort-by dateRun` で `lastRunTime` を見るまで、79 分経過を
3 時間半と読み違えた（ファイルの時刻は**保存時**のもの）。§4 の
「詰まっているのか遅いだけか分からない」はこれ。



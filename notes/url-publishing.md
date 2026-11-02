# 公開 URL と進捗の伝え方（Gist → ビーコンカーネル）

**いまはビーコンだけ。** Gist は 2026-11-02 に**全廃**した（コード・設定・
Secret・クライアントの解決経路まで）。URL は `runtime/beacon.py` に実装した
ビーコンカーネルが持ち、serve の `publish_url` から必ず呼ばれる
（`kbuild/serve_script.py` の `render()` が `beacon` モジュールを無条件で
足すので、ビルダーが書き忘れることがない）。

この文書は「なぜ置き換えたか」「何を実測したか」「何を失ったか」の記録。
Gist 時代の記述は**経緯として**残してあるが、いまの実装には存在しない。

## 1. Gist 方式（2026-11-02 に全廃）

以下は当時の実装の記録。`runtime/gist.py` も `serve-cells/gist.py` も
`gist-probe` カーネルももう無い（`tests/test_no_gist.py` が戻らないよう見ている）。

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

しかも**その待ち時間は同じ日で 3 倍以上振れる**。2026-11-01 の実測:

| 回 | `Loading weights took` | 結果 |
|---|---|---|
| baseline v3 | 2,640 秒 | 予算 2,700 秒を 300 秒超過して ERROR |
| baseline v4 | **1,304 秒** | 2,145 秒で完走（`ok=true`） |
| cache（21:21） | **4,000 秒でも終わらず** | 予算 4,200 秒を超過して ERROR（layers.34 で打ち切り） |
| baseline_long（11-02 00:50） | **2,877 秒** | 予算 7,000 秒のうち **3,820 秒で完走**（`ok=true` / 30.7 t/s） |
| 本番 v16（11-02 03:14・prefetch 有効） | **3,000 秒超（層あたり 47 秒）** | 予算 4,200 秒を超過して ERROR（layers.55 で打ち切り）。同じフラグの playground は 1,081 秒だった |

最後の 2 行が効いている: 同じコードで 1,304〜4,000 秒超まで振れるので、
**予算をいくつにしても「遅いノードを引いた日」は落ちる**。落ちた理由が
「壊れた」のか「遅いだけ」なのかを外から判別できないことが本質で、
予算の設計では解けない。

v16 はこの 3 例目で、しかも**予算切れが「起動失敗」と報告された**
（`notes/vllm-startup-speedup.md` §24）。いまは `_why` を残して文言を分けてある。

同じコード・同じ `KC_WARM_CACHE` でもこの差なので、**外から進捗が見えないと
「詰まっているのか遅いだけか」を判別できない**。ここがビーコンの次の用途。

実験カーネル（`vllm-playground`）は個人アカウントにあるので、
**同じアカウントに status 用のビーコンを 1 つ置けば同じ手が使える**。
頻度は 30 秒ごとではなく「段が変わったとき + 5 分ごと」程度にしないと
バージョンが爆発する。

## 5. 現状

| 対象 | 状態 |
|---|---|
| URL（配信） | **ビーコンのみ**。`publish_url` は `url.txt` とビーコンに書く |
| URL（読み取り） | `python3 -m cli beacon`。古い値と `probe` は既定で捨てる。`cli ask` も同じ判定を通る |
| 進捗（起動中） | **外から見えない**。Gist 版を消したので、待っている間の現在地はどこにも出ない |

### Gist を消した（2026-11-02）

消したもの:

| 場所 | 何 |
|---|---|
| `runtime/gist.py` | `kc_gist_write` / `kc_gist_selftest` / `kc_gh_token*`（モジュールごと削除） |
| `kaggle/serve-cells/gist.py` | 配布セル。`publish.py`（ビーコンのみ）に置き換え |
| `kaggle/step2-vllm/cells/03_vllm_serve.py` | 起動待ちの進捗を `vllm_status.txt` に流す `_gist_status()` 一式 |
| `kaggle/build_gist_probe.py` / `kaggle/cpu-tests/gist/` | `gist-probe` カーネル（台帳からも削除） |
| `cli/ask.py` / `run-kaggle-chat.command` | Gist を読む URL 解決。ビーコンに一本化 |
| `config.env` | `GIST_OWNER` / `GIST_ID`。`KAGGLE_OWNER` に置き換え |
| `kbuild/secrets.py` | `GIST_ID` / `GITHUB_TOKEN` の台帳エントリ |

`GIST_OWNER` は**ビーコンの slug**（同じアカウントに置く）にも使っていたので、
名前だけ `KAGGLE_OWNER` に変えて残した。

### 失ったもの: 起動待ちの現在地

§4 の穴は**広がった**。Gist 時代は「動かないなりに口が 1 つあった」が、
いまは無い。`kaggle kernels logs` も `output` も実行中は空なので、
待っている間は `kernels list --sort-by dateRun` の `lastRunTime` を見るしかない。

代わりに残っているのは**終わってから読めるもの**だけ:

- `result.json` の `_why`（`started` / `died` / `budget`）と `timings.prefetch`
- `keepalive.txt` / `cleanup.txt`
- `cli timings` の `prefetch` 列（辞退したかどうか）

次にやるなら**ステータス用のビーコン**（§4 の案）。ただし
`kernels_push` はバージョンを 1 つ消費するので、30 秒ごとではなく
「段が変わったとき + 5 分ごと」にしないと版が爆発する。

### 経緯: `runtime/gist.py` を残していた理由（2026-11-01 判断・撤回済み）

当時は 3 つ理由を挙げて残した。(1) `publish_url` の片翼として生きている、
(2) 進捗を運ぶ唯一の実装、(3) `03_vllm_serve.py` が `vllm_status.txt` を
書いている。**2026-11-02 に利用者判断で全廃**し、(1) はビーコンへ、
(2)(3) は「無い」ことを受け入れた（上の「失ったもの」）。

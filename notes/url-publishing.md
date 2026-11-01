# 公開 URL と進捗の伝え方（Gist → ビーコンカーネル）

**このリポジトリの実装は Gist 方式のまま**（`kaggle/serve-cells/gist.py` の
`publish_url`、`runtime/gist.py` の `kc_gist_write`）。ビーコンカーネルは
**別プロジェクト（専用アカウント側）で動いており、ここには 1 行も無い**。
この文書は「なぜ置き換えたか」と「こちら側に何が残っているか」の記録。

## 1. Gist 方式（このリポジトリの現状）

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

## 3. ビーコンカーネル（採用案・実装は別プロジェクト）

専用アカウントに `kentasatomuse/serve-url-beacon` を 1 つ置く。CPU・
インターネット無し・非公開。**実行するものではなく、カーネルのコード自体を
掲示板として使う**:

```python
URL = "https://xxxx.trycloudflare.com"
UPDATED = 1762051200.123      # 発行時刻の epoch
```

- 書き手: serve カーネル内で `/kaggle/tmp/beacon` に `kernel-metadata.json` と
  URL 入りの `main.py` を作り、**カーネルに注入される Kaggle 認証**
  （`KaggleApi().authenticate()`）で `kernels_push` する。Gist と同じく
  **失敗しても配信は続け**、成否は `result` の `beacon` フィールドに残す
- 読み手: VM 側で `kaggle kernels pull kentasatomuse/serve-url-beacon` して
  `URL` と `UPDATED` をパースする（`kaggle-ai.py` の `read_beacon()`）
- 鮮度は `UPDATED` で判定する。**古いテスト URL を掴むバグが一度出た**ので、
  「push 前のスタンプと異なるものだけを本物とする」チェックが入っている

### 実測（2026-11-01・単体テスト）

| 区間 | 実測 |
|---|---|
| カーネル内からの `kernels_push` 呼び出し | **1.63 秒** |
| push → VM 側 `kernels pull` で見えるまで | **≤ 11 秒**（ポーリング 10 秒、実伝播は数秒） |
| 合流まで全体 | **20 秒以内** |

Gist と同等以上に速く、**新しいアカウント・Secrets・外部サービスが不要**。
書き込みは Kaggle が注入する認証、読みは VM に既にある専用アカウントの
CLI 認証だけで閉じる。

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
- **serve 本体に組み込んだ `publish_url_beacon` が本番実行で発火するところは
  まだ見ていない**。v4 はトンネル確立の前に詰まり、その後の実行は途中で
  キャンセルしたため。新しい slug で起動が通った回が最初の end-to-end 確認になる

## 4. ビーコンが**カバーしていない**もの: 起動の進捗

ビーコンが運ぶのは**トンネル URL だけ**。vLLM の起動待ち（実測 2,640 秒、
`notes/vllm-startup-speedup.md` §20）のような「いま何をしているか」は
運ばない。ここは Gist 時代から空白のままで、いまも外から見えない。

実験カーネル（`vllm-playground`）は個人アカウントにあるので、
**同じアカウントに status 用のビーコンを 1 つ置けば同じ手が使える**
（注入認証で push できるのは自分のカーネルだけなので、専用アカウントの
ビーコンは `yamahei21python` からは書けない）。頻度は 30 秒ごとではなく
「段が変わったとき + 5 分ごと」程度にしないとバージョンが爆発する。

## 5. このリポジトリで残っている作業

`runtime/gist.py` と `serve-cells/gist.py` は**このままでは動かない**
（Secret を紐付けない限り）。選択肢は 3 つ:

1. 実験カーネルだけは個人アカウントのままにして、Secret を紐付けて使う
2. ビーコン方式をこちらにも移植する（URL 用と進捗用で 2 本、または 1 本に
   2 つのフィールドを持たせる）
3. Gist 依存を落として、進捗は「終わってからログを読む」に割り切る

**2 が本命**だが、このリポジトリにはビーコンの実装が無いので、
移植するなら別プロジェクトから持ってくる必要がある。

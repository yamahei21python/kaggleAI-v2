# 運用手順（詳細）

README から切り出した手順の詳細。**パスは 2026-10-30 時点のスナップショット**で、
その後の再編（`kaggle/build_exl3_serve.py` / `cli/` / `kbuild/spec.py`）は
反映していない。現行の入口は README、`python3 -m cli list`、`python3 -m cli build`。
push は `python3 -m cli push <name> --yes`（直叩きの `kaggle kernels push` は
生成物の鮮度を保証しないため非推奨）。

> **script 版（`…-script`）にはこの文書の手順はそのまま使えない。**
> script は `main.py` 1 ファイルで、`cli push` が直前に再生成してから
> 一時ディレクトリで `kernel-metadata.json` を揃えて push する。
> `kaggle kernels push -p kaggle/step2-script/cloudflared` を直に叩いても
> 動くが、**生成物が古いままでも気づかない**。手順は README と
> `notes/gpu-smoke-plan.md` を見る。

---

## 手順

### 0. まず smoke test（推奨）

いきなり本番を組む前に、**スタックが動くかだけ**を確認します。Secrets も Dataset も不要です。

1. `kaggle/smoke/kernel-metadata.json` の `id` を自分のユーザー名に変更
2. push

```bash
kaggle kernels push -p kaggle/smoke
kaggle kernels status <username>/smoke-orcarouter-qwen38-27b
kaggle kernels output <username>/smoke-orcarouter-qwen38-27b
```

確認項目:

1. T4 x2 の認識（VRAM 合計）
2. Ollama のインストール
3. コンテキスト長の自動決定
4. 16.5 GiB の pull と VRAM ロード
5. 推論（短文 + 日本語 + thinking）と tokens/s

### 1. cloudflared を Dataset に保存（初回のみ）

Step2 は `/kaggle/input/` からバイナリを復元します。無ければ毎回 GitHub から
40MB を落とすので、**初回だけ**作っておきます（再現手順）。

```bash
mkdir -p datasets/cloudflared/bin
curl -fsSL -o datasets/cloudflared/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x datasets/cloudflared/bin/cloudflared
# dataset-metadata.json の id を自分のユーザー名に合わせてから:
kaggle datasets create -p datasets/cloudflared --dir-mode tar
```

> **cloudflared 版なら Secrets は不要**です。quick tunnel はアカウント・API キー不要で、
> 自動発行される URL がそのまま使えます。
> **ngrok 版を使う場合だけ** `NGROK_TOKEN` が必要です（`my-cloudflared-bin` は不要）。

### 2. Kaggle Secrets を設定（cloudflared 版は不要 / ngrok 版は必須）

> ⚠ **`kaggle` CLI に secrets を設定するコマンドはありません。**
> 使用可能なコマンドは `competitions / datasets / kernels / models / files / config` のみです。
> Secrets は **Kaggle の UI でしか登録できません**。

**cloudflared 版は空で動きます。** ngrok 版は `NGROK_TOKEN` だけ登録が必要です。
Notebook を開き、右側 **Add-ons → Secrets → Add a Secret**:

| ラベル | 値 | 必須 |
|---|---|---|
| `API_PASS` | 設定すると Basic 認証が有効になる | 不要（既定は認証なし） |
| `API_USER` | Basic 認証のユーザー名（`API_PASS` 設定時のみ） | 不要（既定 `ollama`） |
| `NGROK_TOKEN` | ngrok の authtoken | **ngrok 版のみ必須** |
| `NGROK_DOMAIN` | ngrok の dev domain | 任意（埋め込み既定値を上書き。既定値は `config.env`） |
| `GITHUB_TOKEN` | gist 権限つきトークン（`gh auth token` の出力） | 任意（Gist 自動配布を使う場合） |
| `GIST_ID` | 配布先 Gist の ID | 任意（既定 Gist を使用） |
| `KAGGLE_USERNAME` | Kaggle ユーザー名 | Step1 のみ |
| `KAGGLE_KEY` | API Key（`kaggle.json` の key） | Step1 のみ |

> ⚠ **Secret は登録するだけでは使えません。** ノートごとに
> `Edit → Add-ons → Secrets` でトグルを **ON**（Attach）してください。
> 新規ノートには自動で付きません（v1/v2 が 401 になった原因）。

**既定は認証なし**です（URL を知っていれば誰でも使えます）。
他人に使われたくないときは環境変数 `API_PASS` を設定してください
（`os.environ` 経由なので、ノート冒頭で `os.environ["API_PASS"]="..."` を書く）。
接続情報は **接続情報セルと `/kaggle/working/connection_info.txt`** に表示されます。

### 3. Step1: モデルを Dataset に保存（初回のみ）

Step1 を実行すると、モデルを Kaggle Dataset として保存します。2回目以降は不要です。

```bash
kaggle kernels push -p kaggle/step1
```

> **Step1 をスキップしても構いません。**
> Step2 は Dataset が無ければレジストリから直接 `ollama pull` します（16.5 GiB / 10〜30分）。
> Step1 の価値は「2回目以降の起動を速くする」ことだけです。

### 4. Step2: サーバー起動 + トンネル公開

1. Step2 の Notebook を開く（cloudflared 版 or ngrok 版）
2. **Add Data** → 以下の **3つ** を追加（ngrok 版は `my-cloudflared-bin` 不要で **2つ**）
   - `my-ollama-orcarouter-qwen38-27b`（モデル 16.5 GiB）
   - `my-ollama-orcarouter-qwen38-27b-patch`（ollama + llama-server + CUDA 1.45 GB）
   - `my-cloudflared-bin`（cloudflared 40 MB）
3. **Accelerator** を `GPU T4 x2` に設定
4. **Internet** を `On` に設定
5. セルを上から順に実行（または **Run All**）

```bash
# kernel-metadata.json の id は yamahei21python/... 固定なので、
# 別ユーザーは id / dataset_sources の owner を自分のユーザー名に置換してから push
kaggle kernels push -p kaggle/step2
```

> **cloudflared 版は Dataset 3つとも推奨。** モデル本体だけでは `bin/ollama` と
> **CUDA バックエンド**が入りません（片方だと CPU 推論に落ちる）。cloudflared が無いと
> 毎回 GitHub から 40MB を落とし、GitHub に届かないと**カーネルが即エラーになります**。
> ngrok 版は cloudflared を使わないので **2つ**で足ります（ngrok 本体は起動時に取得）。

> **マウントパスは自動探索です。** Kaggle は `/kaggle/input/<slug>`（旧）から
> `/kaggle/input/datasets/<owner>/<slug>`（新）へ移行中で、同一期間に混在します。
> Step2 の `_find_input()` が深さ2まで探索し**空でない方**を採用するので、
> 手動でパスを直す必要はありません（2026-10-27 実測で両形式を確認）。

最後の「キープアライブ」セルは既定 **30 分**で自動終了します。

> ⚠ **GPU セッションは同時2本まで。** 実行中のカーネルが2本あると
> `Maximum batch GPU session count of 2 reached` で push が弾かれます。
> `kaggle kernels stop` は CLI に無いので、**終了を待つ**か UI から止めてください。
> cloudflared 版と ngrok 版を同時に走らせる場合は1本ずつ。

### 4b. 検証（必ず完了させてから）

ログ API は実行中も空、セル出力も後から取れません。**検証結果は `/kaggle/working` に書かれます。**

```bash
# 実行完了（既定30分）後:
kaggle kernels output <username>/step2-orcarouter-qwen38-27b-cloudflared -p out/
# ngrok 版は slug が ...-ngrok-fixed
# kaggle kernels output <username>/step2-orcarouter-qwen38-27b-ngrok-fixed -p out.ngrok/
cat out/verify_result.txt      # [1]UI [2]200 [3]streaming [4]OpenAI互換 [5]GPU常駐
cat out/public_url.txt         # 公開URL
cat out/connection_info.txt    # curl / Python / opencode.json の設定一式
cat out/keepalive.txt          # ollama/cloudflared 生死 + GPU 使用量
```

検証は **5項目**です。

| # | 項目 | 見ているもの |
|---|---|---|
| [1] | ui | 認証なしで `/` が 200（チャット HTML が返る） |
| [2] | chat-ok | `/api/chat` が 200 + 生成物（content / thinking / eval_count のどれか） |
| [3] | streaming | 到着が逐次的（`chunks≥3 && spread>0.2s && first<total×0.8`） |
| [4] | openai-compat | `/v1/chat/completions`（**opencode.json が実際に使う経路**） |
| [5] | gpu-resident | `ollama ps` に GPU 常駐 + コンテキスト長一致（CPU フォールバック検出） |

`verify_result.txt` に FAIL があれば該当セルが `raise` するので、カーネルは
**ERROR** になります。**COMPLETE = 5項目すべてパス + keepalive が alive** です
（keepalive 失敗も `FAILED.txt` 付きで ERROR）。
失敗時は `FAILED.txt` にも理由が残ります（`verify_result.txt` があっても
`FAILED.txt` があれば失敗。回収スクリプトはこの2つで判定します）。
（`/output` にも同内容を書きますが、`/output` は公式の永続先ではなく zip に載りません）

### 5. ローカルPCからアクセス

```bash
curl https://<公開URL>/api/tags
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<公開URL>/v1",
    api_key="dummy",   # 認証なし。値は何でもよい
)
client.chat.completions.create(
    model="orcarouter/Qwen3.8-27B-Uncensored:latest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

> **cloudflared 版の公開URLは実行のたびに変わります**（quick tunnel の仕様）。
> 切れたらトンネルセルを再実行し、接続情報を取り直してください。
> **ngrok 版は固定URL**なので、この手順は不要です（5c 参照）。

### 5b. 固定URLで対話する（`ask.py`）

quick tunnel の URL は事前に決められません。そこで **URL の配布先だけ固定**します。

- カーネルが起動直後に現在の URL を **Gist の `url.txt`** へ PATCH
- クライアントは固定の raw URL を 1 回 GET して現在の URL を得る

```
https://gist.githubusercontent.com/yamahei21python/2b44ac8cb115012a31d3c46661af933f/raw/url.txt
```

```bash
python3 -m cli ask "自己紹介して"        # 1回だけ
python3 -m cli ask                      # 対話モード（履歴保持, /reset /quit）
python3 -m cli ask --url https://xxx.trycloudflare.com "..."   # Gist を介さない
```

- **ngrok 版は固定URLなので Gist は不要**（この節は cloudflared 版の話）
- `GITHUB_TOKEN` 未設定なら Gist 更新はスキップ（従来どおり接続情報セルを使う）
- Gist は **secret（非公開 URL）** だが、URL を知れば誰でも読める。
  認証を有効にした場合は URL 自体が入口になる点に注意
- `ask.py` は `API_USER` / `API_PASS`（環境変数 → `~/.orcarouter_pass`）があれば
  Basic 認証を付ける。無ければそのまま送る（既定の認証なし構成で動く）
- raw は CDN キャッシュが効くので `ask.py` は毎回 `?t=<epoch>` を付ける
- 取得した URL が古い（`530`）ときは、カーネルが起動中か確認 → 再実行

`opencode.json` の例は Step2 の「接続情報」セルが出力します。

### 5c. ngrok 版（固定URLが欲しいとき）

`kaggle/step2-ngrok/` に **ngrok 版ノート**を別途生成できます（ビルダーは共通）。

```bash
STEP2_TUNNEL=ngrok python3 kaggle/build_step2.py   # → kaggle/step2-ngrok/
kaggle kernels push -p kaggle/step2-ngrok
```

| 項目 | cloudflared 版 | ngrok 版 |
|---|---|---|
| アカウント | 不要 | 必要（`NGROK_TOKEN`） |
| URL | 毎回ランダム | **固定**（既定 `rebalance-cornflake-sagging.ngrok-free.dev`） |
| 追加 Dataset | `my-cloudflared-bin` | 不要（起動時に取得） |
| 帯域 | 制限なし | 無料枠あり |
| 同時起動 | 可 | **同一ドメインで1本のみ** |

- Secrets: `NGROK_TOKEN` のみ必須。`NGROK_DOMAIN` は上書き用（未設定なら埋め込み値）
- `NGROK_DOMAIN` は https://dashboard.ngrok.com/domains で取得（無料枠1個）
- 固定URLなので **URL を毎回調べなくてよい**。ただし `ask.py` の既定は Gist 参照なので、
  `--url` か `LLM_URL` で明示するのが確実（Gist が古い cloudflared URL のままでも動く）

```bash
python3 -m cli ask --url https://rebalance-cornflake-sagging.ngrok-free.dev "自己紹介して"
export LLM_URL=https://rebalance-cornflake-sagging.ngrok-free.dev   # 以後 --url 不要
```
- トンネルセルは共通インターフェース（`TUNNEL_NAME` / `RESTART_CMD` / `probe_url()`）
  なので、keepalive の再起動も両対応
- 無料枠はブラウザ UA に警告ページを返す → 全リクエストに
  `ngrok-skip-browser-warning: true` を付ける（ノートと `ask.py` の両方で対応済み）

`opencode.json` の例は Step2 の「接続情報」セルが出力します。

### 5d. ダブルクリックで起動 → ブラウザでチャット

`run-kaggle-chat.command` を**ダブルクリックするだけ**で、Kaggle が起動してチャット UI が開きます。

```
run-kaggle-chat.command をダブルクリック
  → 既に動いていれば push しない（そのまま再利用）
  → 動いていなければ kaggle kernels push（GPU T4 x2）
  → /api/tags が 200 になるまで待機（既定 900 秒）
  → https://<固定ドメイン>/ をブラウザで開く
```

- **認証なし**。パスワード入力も URL の `#auth=` も不要
- **ngrok 無料枠は `ERR_NGROK_6024` のブロックページを返す**（ボタンもリンクも無いので
  クリックでは抜けられない）。`ngrok-skip-browser-warning: true` ヘッダを付けた
  リクエストは素通りするので、**ローカル中継**（`auth_proxy.py`）がそのヘッダを肩代わりする。
  ブラウザは `http://127.0.0.1:19950/` を開く。cloudflared 版はこの問題が無い
- UI は **リポジトリ直下の `llama-chat.html`**（ローカル uzu 版と同一ファイル）。
  接続先は HTML 側が自動判定する: `?api=` → `?port=` → 同一オリジン（中継）→ 既定ポート
- 中継は**ローカルの `llama-chat.html` を配る**。上流から取ると push 済みの古い版が返る

```bash
KC_TUNNEL=cloudflared ./run-kaggle-chat.command   # URL 可変版
KC_FORCE_PUSH=1 ./run-kaggle-chat.command         # 動いていても再 push
KC_NO_OPEN=1 ./run-kaggle-chat.command            # ブラウザを開かない
```

> **`uzu` の Kaggle 版は作れません。** uzu は Metal（macOS）専用で、
> Linux x86_64 は上流でも *in progress*。Kaggle の T4 では動きません。
> そのためこのランチャーは Ollama 経由です。

---


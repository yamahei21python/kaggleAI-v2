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

> **Secrets は不要**です。cloudflared の quick tunnel はアカウント・API キー不要で、
> 自動発行される URL がそのまま使えます。URL は毎回変わりますが、
> **ビーコン**（カーネルのコードを掲示板にする方式）から読めるので固定 URL は要りません
> （`notes/url-publishing.md`）。ngrok は 2026-11-01 に廃止しました。

### 2. Kaggle Secrets を設定（すべて任意）

> ⚠ **`kaggle` CLI に secrets を設定するコマンドはありません。**
> 使用可能なコマンドは `competitions / datasets / kernels / models / files / config` のみです。
> Secrets は **Kaggle の UI でしか登録できません**。

**空でも動きます。** 下の表は「設定すると挙動が変わる」ものだけで、必須はありません。
Notebook を開き、右側 **Add-ons → Secrets → Add a Secret**:

| ラベル | 値 | 必須 |
|---|---|---|
| `API_PASS` | 設定すると Basic 認証が有効になる | 不要（既定は認証なし） |
| `API_USER` | Basic 認証のユーザー名（`API_PASS` 設定時のみ） | 不要（既定 `ollama`） |
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

1. Step2 の Notebook を開く
2. **Add Data** → 以下の **3つ** を追加
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

> **Dataset 3つとも推奨。** モデル本体だけでは `bin/ollama` と
> **CUDA バックエンド**が入りません（片方だと CPU 推論に落ちる）。cloudflared が無いと
> 毎回 GitHub から 40MB を落とし、GitHub に届かないと**カーネルが即エラーになります**。

> **マウントパスは自動探索です。** Kaggle は `/kaggle/input/<slug>`（旧）から
> `/kaggle/input/datasets/<owner>/<slug>`（新）へ移行中で、同一期間に混在します。
> Step2 の `_find_input()` が深さ2まで探索し**空でない方**を採用するので、
> 手動でパスを直す必要はありません（2026-10-27 実測で両形式を確認）。

最後の「キープアライブ」セルは既定 **30 分**で自動終了します。

> ⚠ **GPU セッションは同時2本まで。** 実行中のカーネルが2本あると
> `Maximum batch GPU session count of 2 reached` で push が弾かれます。
> `kaggle kernels stop` は CLI に無いので、**終了を待つ**か UI から止めてください。

### 4b. 検証（必ず完了させてから）

ログ API は実行中も空、セル出力も後から取れません。**検証結果は `/kaggle/working` に書かれます。**

```bash
# 実行完了（既定30分）後:
kaggle kernels output <username>/step2-orcarouter-qwen38-27b-cloudflared -p out/
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

> **公開URLは実行のたびに変わります**（quick tunnel の仕様）。
> 切れたらトンネルセルを再実行し、接続情報を取り直してください。
> 「毎回 URL を調べる」のを避けるのが次の 5b です。

### 5b. 変わってしまう URL を読む（ビーコン）

quick tunnel の URL は事前に決められません。そこで **URL の置き場所だけ固定**します。
本命は **ビーコン**（掲示板カーネルのコードに URL を書く方式。`notes/url-publishing.md`）。

```bash
python3 -m cli beacon                    # 既定の掲示板を読む
python3 -m cli beacon --json | jq -r .url   # シェル置換用
```

- 配信カーネルが起動直後に掲示板（`serve-url-beacon`）の `main.py` を
  `kernels_push` で書き換え、こちらは `kernels_pull` で読む（実測 0.26 秒）
- **Secrets 不要**。専用アカウントでも動く（Gist 版が要していた
  `GITHUB_TOKEN` の紐付け作業ごと不要）
- 既定で **`STAGE == "probe"` と 6 時間より古い値を捨てる**。
  捨てたときは理由が出る（黙って古い URL を返さない）
- 戻り値: `0` = 使える / `1` = 読めたが捨てた / `2` = 引けなかった

Gist 版は 2026-11-02 に全廃しました（`notes/url-publishing.md`）。
URL の出所はビーコンだけです。

```bash
python3 -m cli ask "自己紹介して"        # 1回だけ
python3 -m cli ask                      # 対話モード（履歴保持, /reset /quit）
python3 -m cli ask --url https://xxx.trycloudflare.com "..."   # どちらも介さない
```

- URL は**非公開の掲示板カーネル**に置くが、URL を知れば誰でも入れる。
  認証を有効にした場合は URL 自体が入口になる点に注意
- `ask.py` は `API_USER` / `API_PASS`（環境変数 → `~/.orcarouter_pass`）があれば
  Basic 認証を付ける。無ければそのまま送る（既定の認証なし構成で動く）
- `ask.py` はビーコンを `kernels pull` で引く（1 回数秒）。`--url` を渡せば引かない
- 取得した URL が古い（`530`）ときは、カーネルが起動中か確認 → 再実行

`opencode.json` の例は Step2 の「接続情報」セルが出力します。

### 5d. ダブルクリックで起動 → ブラウザでチャット

`run-kaggle-chat.command` を**ダブルクリックするだけ**で、Kaggle が起動してチャット UI が開きます。

```
run-kaggle-chat.command をダブルクリック
  → ビーコンに前回の URL が残っていて、それが生きていれば push しない
  → 動いていなければ kaggle kernels push（GPU T4 x2）
  → ビーコンが新しい URL を返すまで待機（healthz が 200 になるまで）
  → /api/ps に size_vram>0 が乗るまで待機（既定 900 秒）
  → その URL をブラウザで開く
```

- **認証なし**。パスワード入力も URL の `#auth=` も不要
- UI は **リポジトリ直下の `llama-chat.html`**（ローカル uzu 版と同一ファイル）。
  接続先は HTML 側が自動判定する: `?api=` → `?port=` → 同一オリジン → 既定ポート
- ブラウザ警告ページの回避（`ngrok-skip-browser-warning`）は **2026-11-01 に不要になった**
  （ngrok を廃止し、cloudflared は素の UA を通す）

```bash
./run-kaggle-chat.command                     # 既定（cloudflared）
KC_FORCE_PUSH=1 ./run-kaggle-chat.command     # 動いていても再 push
KC_NO_OPEN=1 ./run-kaggle-chat.command        # ブラウザを開かない
```

> **`uzu` の Kaggle 版は作れません。** uzu は Metal（macOS）専用で、
> Linux x86_64 は上流でも *in progress*。Kaggle の T4 では動きません。
> そのためこのランチャーは Ollama 経由です。

---


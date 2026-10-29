# Kaggle GPU で Qwen3.8-27B-Uncensored を動かす

Kaggle の無料 GPU（Tesla T4 × 2枚）で Ollama を起動し、公開 API サーバーにするまでの手順です。
トンネルは **cloudflared**（既定・アカウント不要）と **ngrok**（固定URL）の2種を切り替えられます。

```
[Kaggle Notebook] → Ollama (T4 x2) → リバースプロキシ（UI配信/keepalive リセット） → cloudflared / ngrok → [ブラウザ / APIクライアント]
```

---

## 結論から

| 環境 | 27B モデル | 備考 |
|---|---|---|
| **Kaggle T4 x2** | ✅ 動く | 本ドキュメントの対象。16.52 GiB が 30 GiB に収まる。**131K で 13.7 tokens/s** |
| **ローカル Mac (RAM 16 GiB)** | ❌ **動かない** | 実測 **0.10 tokens/s**。詳細は後述 |

**何を使うべきか。**

| 目的 | 手段 |
|---|---|
| とにかく動かす | **Ollama**（`ollama pull` 一発、環境変数で9割足りる） |
| 起動を速くする | **Dataset キャッシュ**（pull 2.6分を削減 → 起動 4分 → 約1.5分） |
| thinking を切る | ローカルは `--think=false`、サーバーは `--reasoning off` |
| GPU を使う | **`lib/ollama/cuda_v12/` を必ず復元**（無いと黙って CPU 推論） |
| 生成速度を伸ばす | **llama-server 直起動 + `--spec-type draft-mtp`**（ローカル 1.18x、T4 は未測定） |

**手元の GGUF はアップロード不要です。** `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` は Ollama レジストリの
`orcarouter/Qwen3.8-27B-Uncensored:latest` と **sha256 が完全一致**します（実測）。

```
3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
```

16.8 GB を上げる代わりに `ollama pull` 一発で済みます。

---

## モデル

`orcarouter/Qwen3.8-27B-Uncensored:latest`

| 項目 | 値 | 出典 |
|---|---|---|
| 量子化 | Q4_K_M | GGUF ヘッダ `general.file_type = 15` |
| パラメータ | 27.3B | Ollama config |
| サイズ | 16.52 GiB（model 15.66 + projector 0.87） | manifest |
| コンテキスト | 262,144 | GGUF ヘッダ |
| 機能 | completion / vision / tools / thinking | `ollama show` |
| アーキテクチャ | qwen35（65ブロック、full attention 17層 + SSM 48層） | GGUF ヘッダ |
| ライセンス | Apache-2.0 | manifest |

### 無検閲版である根拠

Abliterlitics の Qwen3.8-27B **12変体比較**で総合1位。

- judge ASR **82.2%**（有害リクエストの 4/5 に応答）
- capability 劣化なし（元モデルと同等の賢さ・正直さ）
- モデルカードの主張 **4/4 検証済み**（131 matrices = 129 LM + 2 MTP、vision 333/333 byte-identical）
- copyright は唯一残った壁（歌詞・書籍の一節は書けない）

> 比較は vLLM + FP8、単一 RTX 5090 で実施。Kaggle の Q4_K_M GGUF とは条件が異なるため、
> 順位は重みの性質として読んでください。

### 手元のファイルとの関係

| | ローカル GGUF | orcarouter (Ollama) |
|---|---|---|
| model 層 sha256 | `3445102e...` | `3445102e...` **一致** |
| vision | ❌ テンソル 0 個 | ✅ projector 同梱 |
| テンソル数 | 866 | 866（テンソル表も一致） |

ローカルのファイルは vision を持たないテキスト専用の切り出しです。**vision を使うならレジストリ版が必要**です。

---

## ファイル構成

```
kaggleAI/
├── README.md                                   # このファイル
├── run-llama-server.command                    # ローカル起動ランチャー（ダブルクリック）
├── run-ai.command                             # uzu / Kaggle を選んで起動（統合ランチャー）
├── run-kaggle-chat.command                     # Kaggle 起動 + ブラウザでチャット（ダブルクリック）
├── run-uzu-server.command                      # uzu 9B-M を OpenAI 互換サーバとして起動
├── kaggle-quota.py                             # Kaggle GPU/TPU の残り時間を表示
├── build_keepalive_test.py                     # keepalive ON/OFF + 外部受信テスト用ノート生成
├── step2-ka-test/                              # GPU 不要の keepalive/外部受信検証ノート
├── llama-chat.html                             # 自前のチャット UI（ローカルも Kaggle も共通）
├── opencode-llama-local.json                   # opencode 用プロバイダ設定（貼り付け用）
├── Qwen3.8-27B-Uncensored-Q4_K_M.gguf          # 手元の GGUF（16.8 GB、アップロード不要）
├── ask.py                                      # 対話クライアント（URL/認証を自動解決）
├── config.env                                  # ngrok ドメイン / Gist / slug / ポートの共通設定
│                                               # （config.local.env を置くと上書き。gitignore 済み）
├── kcconfig.py                                 # config.env を Python から読む
├── lib/launcher-common.zsh                     # run-kaggle-chat / run-ai の共通処理
├── kaggle-dataset-orcarouter/                  # Dataset キャッシュ（モデル本体）
│   ├── dataset-metadata.json
│   ├── blobs/                                  # モデルの実体（17 GB）
│   └── manifests/registry.ollama.ai/orcarouter/Qwen3.8-27B-Uncensored/latest
│                                               # ※ bin/ は空。バイナリと CUDA は
│                                               #    -patch Dataset 側にある
├── kaggle-dataset-orcarouter-patch/            # ollama + llama-server + CUDA（1.45 GB）
├── kaggle-dataset-cloudflared/                 # cloudflared バイナリ（40 MB）
└── kaggle/
    ├── build_step1.py                          # ノートブック生成スクリプト
    ├── build_step2.py
    ├── build_smoke.py
    ├── build_bench.py
    ├── build_bench_mtp.py
    ├── step1/                                  # Step1: モデルDL → Dataset 保存
    │   ├── kernel-metadata.json
    │   └── step1-orcarouter-qwen38-27b-dl.ipynb
    ├── step2/                                  # Step2: Ollama + cloudflared 公開
    │   ├── kernel-metadata.json
    │   ├── auth_proxy.py                       # リバースプロキシ + UI配信（base64 埋め込み）
    │   ├── _selftest_proxy.py                  # プロキシ自己テスト
    │   ├── _selftest_builder.py                # ノート内関数の自己テスト
    │   ├── cells/                              # build_step2.py のコードセル（定数のもの）
    │   └── step2-orcarouter-qwen38-27b-cloudflared.ipynb
    ├── step2-ngrok/                            # Step2: Ollama + ngrok 固定URL（同じビルダー）
    │   ├── kernel-metadata.json
    │   └── step2-orcarouter-qwen38-27b-ngrok.ipynb
    ├── assemble_patch_dataset.py                # patch Dataset の組み立て
    ├── build_capture_patch.py                   # patch 採取ノート生成
    ├── capture_patch/                           # 採取用ノート（使い捨て）
    ├── check_notebooks.py                      # コミット済み ipynb == ビルダー出力 を確認
    ├── archive/                                # 調査用の使い捨てノート（probe / diag / cycle スクリプト）
    ├── smoke/                                  # 動作検証用（Secrets / Dataset 不要）
    │   ├── kernel-metadata.json
    │   └── smoke-orcarouter-qwen38-27b.ipynb
    ├── bench/                                  # コンテキスト長 vs 速度の実測
    │   ├── kernel-metadata.json
    │   └── bench-ctx-orcarouter-qwen38-27b.ipynb
    ├── bench_mtp/                              # llama-server 直起動 + MTP の実測
    │   ├── kernel-metadata.json
    │   └── bench-mtp-orcarouter-qwen38-27b.ipynb
    ├── step2-direct-ngrok/                     # llama-server 直起動 + ngrok（Ollama なし）
    ├── cpu-check/                              # CPU で llama-server が起動するか（GPU 不使用）
    ├── exllama-cpu-preflight/                  # CPU イメージで ExLlama がどこまで可能か
    ├── exl3-dataset-upload/                    # HF → Kaggle Dataset へ直接アップロード（CPU）
    └── exl3-gpu-check/                         # ExLlamaV3 + EXL3 の GPU 起動検証

notes/
└── exllamav3-kaggle-decisions.md               # EXL3 版の決定事項ログ
```

### vLLM 版（AWQ + MTP / Ollama 版と併存）

Ollama 版はそのまま残し、**別 kernel・別 Dataset** として追加する。

| ファイル | 役割 |
|---|---|
| `kaggle/build_vllm_dataset.py` | AWQ モデルを Kaggle Dataset 化するノート生成 |
| `kaggle/step2-vllm/prepare-vllm-dataset.ipynb` | DL → Dataset 保存（GPU 不要） |
| `kaggle/build_step2_vllm.py` | vLLM 版 Step2 ビルダー（`STEP2_TUNNEL` で切替） |
| `kaggle/step2-vllm-kernel/<tunnel>/` | 生成物（notebook + kernel-metadata.json） |

```bash
python3 kaggle/build_step2_vllm.py                    # cloudflared
STEP2_TUNNEL=ngrok python3 kaggle/build_step2_vllm.py # ngrok
cd kaggle/step2-vllm-kernel/cloudflared && kaggle kernels push
```

プロキシ / UI / トンネル / keepalive は Ollama 版と同一実装を埋め込む。
vLLM 版だけ `--enforce-sampling` を付けて起動し、
`temperature<=0 → 0.7` / `repetition_penalty<1.05 → 1.05` に矯正する
（既定の Ollama 版は素通しなので挙動は変わらない）。

#### 既知の罠（実測）

| 罠 | 症状 | 対策（実装済み） |
|---|---|---|
| `pip install vllm` が torch だけ差し替え | `vllm` CLI が即死。`Detected that PyTorch and TorchAudio were compiled with different CUDA versions`（vllm 0.32.0 / torch 2.13.0+cu130 vs torchaudio 2.8.3+cu128） | install 直後に **`vllm serve --help` を実行する** preflight → 失敗なら `torchaudio` を削除 → 再判定 → それでも駄目なら `pip install -U torchaudio` |
| preflight を浅く取る | `import vllm.entrypoints.cli.main` は**通ってしまう**。`main()` 内の遅延 import（benchmark.main → transformers → torchaudio）で初めて落ちるため、修復が発動せずラダー全段が 13s で即死（v2 実測） | 実際に落ちる経路（`vllm serve --help`）で判定。`Traceback`/`RuntimeError`/出力長も見る |
| help 取得失敗を help と誤認 | `vllm_help.txt` に traceback が入り、`len>500` 判定を通って全フラグ False（MTP 段が組まれない） | `HELP_OK` に `returncode==0` と `Traceback` 除外を追加。不能時は楽観側 |
| `vllm serve --help` のフラグ判定 | 部分一致だと説明文の言及で誤検知。**テンプレート文字列に `\b` を書くと 0x08 に化けて全フラグ False**（MTP が一度も試行されない） | 行頭一致の正規表現 + 生成物の制御文字検査 |
| 起動ラダーの時間 | 6 段すべて待つと 1 時間超 → GPU クォータ切れで成果物ゼロ | 総予算 1500s / 1 試行 600s、失敗ごとにログを追記保存 |
| keepalive `RUN=0` 即時終了 | ループが 1 周も回らず `unknown` で fail | `ran_any` フラグで死活判定をスキップ |

ノートブックは `build_*.py` から生成します。手で編集せず、スクリプトを直して再生成してください。

```bash
cd kaggleAI/kaggle
python3 build_step1.py && python3 build_step2.py && python3 build_smoke.py \
  && python3 build_bench.py && python3 build_bench_mtp.py

# ngrok 版（固定URL）を追加生成する場合
STEP2_TUNNEL=ngrok python3 build_step2.py
```

### ExLlamaV3 版（EXL3 / 検証中）

GGUF（Ollama / llama.cpp）とは別系統として、**EXL3 量子化 + exllamav3** で同じ
Qwen3.8-27B-Uncensored を動かせるか検証しています。決定事項・却下した選択肢は
`notes/exllamav3-kaggle-decisions.md` にまとめてあります。

| 項目 | 内容 |
|---|---|
| モデル | `Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw`（15.9 GiB / 3 シャード） |
| エンジン | exllamav3 1.6.5（`qwen3_5` アーキテクチャ + MTP 対応、Turing sm75 カーネルあり） |
| 起動方式 | T4 x2 に**レイヤー分割**（tensor parallel は未対応） |
| 制約 | **テキスト専用**（vision は不可）。thinking の制御は GGUF 版ほど柔軟でない |
| GPU 検証 | `kaggle/exl3-gpu-check/`（`yamahei21python/exl3-qwen38-gpu-check`） |
| モデル置き場 | `kaggle/exl3-dataset-upload/`（HF → Kaggle Dataset へ直接アップロード・CPU のみ） |
| Dataset | `yamahei21python/qwen38-27b-uncensored-exl3-4bpw`（private / ready・3 シャード） |
| 配信ラッパー | `kaggle/exl3-serve/`（OpenAI 互換の薄いサーバ + モック自己テスト） |
| CPU 調査 | `kaggle/exllama-cpu-preflight/`（CPU イメージは torch が `+cpu` ビルドで import 不可） |

検証カーネルは 1 セッションで「環境 → exllamav3 導入（torch/CUDA 一致の
プリビルド wheel 優先、無ければ JIT）→ import → モデル取得 → 2×T4 ロード →
日本語 1 生成 → MTP 投機デコード実測」を通し、`/kaggle/working/result.json` に
各段の所要秒・VRAM・tok/s を残します（どこで失敗しても exit 0）。

```bash
# 1) CPU カーネルでモデルを Dataset 化（GPU クォータ不要・初回のみ）
cd kaggleAI/kaggle/exl3-dataset-upload && kaggle kernels push

# 2) GPU 検証（GPU セッションはこれ 1 回）
cd kaggleAI/kaggle/exl3-gpu-check && kaggle kernels push
```

> GPU クォータは有限（30h/週）なので 1 回で決める前提。失敗時は GGUF / llama.cpp 版に戻します。
> モデルは Dataset がマウントされていればそれを使い、無ければ HF からダウンロード
> （`/kaggle/working` は 20 GiB 上限のため `/kaggle/temp` → `/tmp` の順に空きの大きい場所へ）。

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
mkdir -p kaggle-dataset-cloudflared/bin
curl -fsSL -o kaggle-dataset-cloudflared/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x kaggle-dataset-cloudflared/bin/cloudflared
# dataset-metadata.json の id を自分のユーザー名に合わせてから:
kaggle datasets create -p kaggle-dataset-cloudflared --dir-mode tar
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
python3 ask.py "自己紹介して"        # 1回だけ
python3 ask.py                      # 対話モード（履歴保持, /reset /quit）
python3 ask.py --url https://xxx.trycloudflare.com "..."   # Gist を介さない
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
python3 ask.py --url https://rebalance-cornflake-sagging.ngrok-free.dev "自己紹介して"
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

## 起動を速くする

### 起動時間の内訳（実測）

smoke test の実測値です。

| 段階 | 時間 | 削れるか |
|---|---|---|
| ollama install（1.44 GB DL） | 約8秒 | ✅ Dataset キャッシュ |
| **モデル pull（16.8 GB）** | **2.6 分** | ✅ **Dataset キャッシュ** |
| サーバー起動 | 2秒 | — |
| VRAM ロード（16.5 GiB） | 87秒 | ❌ 重み転送なので不可 |

**支配的なのはモデル pull の 2.6 分**です。これは消せます。

### Dataset キャッシュ（推奨）

Step2 は Dataset を見て、あればそこから復元します。**3つの Dataset に分かれています。**
（ngrok 版は `my-cloudflared-bin` を使わないので **2つ**）
（下は旧形式のパス例。新形式では `/kaggle/input/datasets/<owner>/<slug>/` になり、
Step2 は自動探索するので実害なし）

```
/kaggle/input/my-ollama-orcarouter-qwen38-27b/          # モデル本体（16.5 GiB）
├── blobs/                   # モデルの実体
└── manifests/registry.ollama.ai/orcarouter/Qwen3.8-27B-Uncensored/latest

/kaggle/input/my-ollama-orcarouter-qwen38-27b-patch/    # ランタイム一式（1.45 GB）
├── bin/ollama               # 43.8 MB（採取時のバージョンで固定）
├── lib/ollama/
│   ├── llama-server         # 15 KB ← これが有ると install.sh を実行しない
│   ├── libllama*.so / libggml*.so / libllama-server-impl.so
│   └── cuda_v12/            # libggml-cuda.so 439 MB + cuBLAS/cudart
└── blobs/sha256-cbe6ad4…    # config blob 214 B（model Dataset に無い）

/kaggle/input/my-cloudflared-bin/                       # cloudflared（40.2 MB）
├── cloudflared              # zip 直下（bin/ なし）
└── bin/cloudflared          # bin/ 有りも同一 Dataset 内で探索
```

- `blobs/` と `manifests/` があれば `ollama pull`（16.5 GiB）をスキップ
- `bin/ollama` があれば ollama の DL をスキップ
- `lib/ollama/llama-server` があれば **`install.sh` を実行しない**（= ollama の
  バージョンが Dataset 側に固定される。patch Dataset v2 以降）
- `lib/ollama/` に **CUDA バックエンド**（`cuda_v12/libggml-cuda.so`）を確保
- `my-cloudflared-bin` があれば cloudflared の 40MB DL をスキップ（**無いと GitHub 依存**）

**配置のしかた（v12 以降）。** `blobs/` はディレクトリごと symlink せず、
**実ディレクトリ + ファイル単位 symlink**、`manifests/` は**実ファイルコピー**
で `/kaggle/tmp/ollama_models/` に組みます。

- dir ごと symlink だと `/kaggle/input`（読み取り専用）を経由してしまい、
  補完 pull が `read-only file system` で即死する（v8 実測）
- **manifests を symlink にすると ollama が拒否する**（`manifest symlink target
  ... is not a sha256 blob`）→ `show` 失敗 → pull 誘発（v11 実測）。
  1KB 級なので実体コピーで回避
- **patch Dataset の `blobs/` も合流**させる。manifest が要求する config blob
  （`sha256-cbe6ad4…` 214B）は model Dataset に無く patch 側にだけ存在するので、
  これが無いと `ollama show` が失敗して pull を誘発する
- 組んだ直後に **manifest 要求 blob の全数を点検**し
  `✅ manifest 要求 blob 全数充足 → pull 不要` を出す
- pull に落ちてもローカル blob を即認識して **manifest 書き直しだけ**で完了
  （v9 実測: `pulling 16GB 100%` が1秒）。**blob が実欠落なら通常どおり 16.5 GiB を DL**

**効果: 起動 4分 → 約1.5分**（VRAM ロード 87秒が下限）。

> **patch Dataset は install.sh の成果物一式を持ちます（patch Dataset v2 以降）。**
> `bin/ollama` + `lib/ollama/` 一式（`llama-server`、`libllama*.so`、
> `libggml*.so`、`cuda_v12/`）を `install.sh` 実行後の状態から採取したもの。
> `llama-server` が有るので **install.sh は走らず、ollama のバージョンが固定**されます。
> cuBLAS / cudart の完全版番号ライブラリは実体が重複するため
> 採取時に落としてあります（SONAME の `.so.12` 側だけ保持）。

Dataset を追加するには、Step2 の Notebook で **Add Data** から選びます。
`kernel-metadata.json` の `dataset_sources` に書いておけば push で入ります
（cloudflared 版は3つ、ngrok 版は2つ）。

### llama.cpp 直起動（Ollama を経由しない）

llama.cpp は `qwen35` に正式対応しています（`src/llama-arch.cpp` に `LLM_ARCH_QWEN35 "qwen35"`）。

**起動時間は短縮しません。** 同じ 16.8 GB の重みを VRAM に載せる工程は変わらないためです。
Ollama は既に llama-server を子プロセスとして起動しているので、直起動は薄いラッパーを1枚剥がすだけです。

| 段階 | Ollama 経由 | 直起動 |
|---|---|---|
| `ollama serve` が応答 | 2秒 | 不要（-2秒） |
| **重み 16.5 GiB を VRAM へ** | **87秒** | **同じ** |

**直起動を選ぶ理由は MTP とフラグ制御です。**

| 欲しいもの | Ollama | llama-server 直 |
|---|---|---|
| thinking を切る | ❌ 3通り試して全滅 | ✅ `--reasoning off --reasoning-budget 0` |
| MTP 投機的デコード | ❌ 露出なし | ✅ `--spec-type draft-mtp` |
| API 認証 | ❌ 無い（手前で自前プロキシが被せる） | ✅ `--api-key` |
| 層分割の制御 | ❌ | ✅ `--tensor-split` |

### 直起動の実証（ローカル 9B / 2026-10-27）

Ollama を経由せず llama-server を直に起動し、API が通ることを確認しました。

```bash
llama-server -m <GGUF> -c 8192 -ngl all -fa on -np 1 \
  --host 127.0.0.1 --port 9931 --jinja --temp 0 \
  --alias qwen35-9b \
  --reasoning off --reasoning-budget 0
```

```
起動完了: 4s
/v1/chat/completions → content: '1\n2\n3\n4\n5'   gen 15.8 tps
日本語               → 'こんにちは！私は AI アシスタントです。...'   gen 15.3 tps
/v1/models           → id: qwen35-9b
```

| エンドポイント | 用途 |
|---|---|
| `/v1/chat/completions` | OpenAI 互換。**これが本命** |
| `/v1/models` | モデル一覧 |
| `/completion` | llama.cpp ネイティブ |
| `/health` | 起動待ちのポーリングに使える |

**Ollama から乗り換えると3か所変わります。**

1. **ポートが 11434 → 9931**（llama.cpp v0.7.0 の新既定）
2. **モデル名が alias になる。** `--alias qwen35-9b` を付けないと `/v1/models` が blob のフルパスを返す
3. **thinking を明示的に切る。** 付けないと `content` が**空**になる（Ollama と同じ現象）

クライアント側は `base_url` を差し替えるだけです。

```python
client = OpenAI(base_url="http://127.0.0.1:9931/v1", api_key="dummy")
client.chat.completions.create(model="qwen35-9b", messages=[{"role": "user", "content": "..."}])
```

外部に出す場合は `--host 0.0.0.0` にして、手前の認証プロキシを 9931 に向けます。

> **MTP のフラグ名に注意。** `--mtp` は **download 専用**（`--spec-type` に `draft-mtp` を足すだけ）、
> `--nextn` は **imatrix 専用**です。どちらもサーバー起動用ではありません。
> サーバーで MTP を使うには **`--spec-type draft-mtp`** を指定します。
>
> このモデルには MTP 層が入っています（`qwen35.nextn_predict_layers = 1`、`blk.64.nextn.*` の4テンソル）。
> 検証用ノートブックは `kaggle/bench_mtp/` です（`none` ↔ `draft-mtp` の比較）。**結果は検証中。**

---

## ローカルで使う（ダブルクリック起動）

`run-llama-server.command` を**ダブルクリックするだけ**で、llama-server が起動してチャット UI が開きます。

### 使い方

```
run-llama-server.command をダブルクリック
  → モデルをロード（実測 5秒）
  → ブラウザでチャット UI が開く
  → ターミナルで Ctrl+C を押すと停止
```

### 環境変数で切り替える

```bash
LLAMA_MODEL=27b ./run-llama-server.command   # 27B（RAM 16 GiB では動きません）
LLAMA_MTP=0     ./run-llama-server.command   # MTP を切る（既定は有効）
LLAMA_CTX=8192  ./run-llama-server.command   # コンテキスト長
LLAMA_PORT=8080 ./run-llama-server.command   # ポート
LLAMA_OPEN=opencode ./run-llama-server.command  # 起動後に opencode を開く
```

### MTP の実測効果（9B / ローカル）

| spec-type | gen tps |
|---|---|
| `none` | 16.02 |
| **`draft-mtp`** | **18.85**（**1.18x**） |

起動ログに `creating MTP draft context against the target model` が出れば有効です。

> **9B には `nextn_predict_layers` がありません**（GGUF ヘッダに無い）。それでも MTP は動きます。
> 27B は `qwen35.nextn_predict_layers = 1` を持ちます。

### Web UI について

**llama-server 同梱の Web UI は使えません。** `--ui` フラグは存在しますが、
Ollama 同梱バイナリには HTML アセットが入っていません（実測: 埋め込み HTML のマッチ数 0、`/` は 404）。

代わりに `llama-chat.html` を同梱しました。`run-uzu-server.command` が
`auth_proxy.py` 経由で `http://127.0.0.1:8899/` に配信します（`file://` は Chrome の
Private Network Access で遮断されるため http 配信が必要）。

### ヘッダ UI / keepalive 仕様

ヘッダ順: `[モデル] [情報] [残り hh:mm] [t/s] [■ 停止]`

| 要素 | Kaggle | uzu |
|---|---|---|
| 残り時間 | 表示（1分ごと更新） | 空欄 |
| 停止ボタン | 表示 | 表示 |

### 残り時間表示

- Kaggle: `/quota` API から GPU クォータ残量取得
- 色分け
  - 2時間未満: オレンジ
  - 30分未満: 赤
  - それ以外: 通常
- ホバー: 使用時間 / 上限 / リセット日時

### 停止ボタン

- `POST /stop`（ヘッダ `X-Kc-Stop: 1` 必須、`API_PASS` 設定時は Basic 認証も必須）でプロキシ/ランチャー終了
- `/quota` が JSON を返す間（プロキシ稼働中）のみ表示
- Kaggle / uzu 共通（プロキシ稼働時）

### keepalive ON/OFF / リセット

- 初期: `RUN=1`, `MINUTES=60`
- 生成リクエスト（`/v1/chat/completions` `/v1/completions` `/api/chat` `/api/generate` など）ごとに
  `/tmp/keepalive.conf` を `RUN=1 MINUTES=60` にリセット（`/api/ps` などのポーリングでは延長しない）
- 延長を重ねてもセル開始からの合計は `MAX_MINUTES`（480 分）で打ち切る
- 停止ボタン: `RUN=0` → keepalive ループは翌サイクルで終了
- 最終リセットから 60分経過で自動終了

### opencode から使う

`opencode-llama-local.json` の中身を `~/.config/opencode/opencode.json` の `provider` に足します。

```json
"llama-local": {
  "name": "llama-server (local)",
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://127.0.0.1:9931/v1", "apiKey": "dummy" },
  "models": { "qwen35-9b": { "name": "Qwen3.5 9B (llama-server, MTP)" } }
}
```

```bash
opencode --model llama-local/qwen35-9b
```

> **Ollama 経由の既存設定は残せます。** `provider` に `llama-local` を**追加**するだけなので、
> `ollama` / `kaggle-ollama` のプロバイダはそのまま使えます。

### uzu に切り替える場合

`./run-uzu-server.command` で uzu（OpenAI 互換）が 9931 に立つ。設定は `opencode-llama-local.json`
をそのまま使い、プロバイダ名が `uzu-local`、モデルIDが `uzu-stage` になる。

```bash
./run-uzu-server.command            # 別ターミナルで起動
opencode --model uzu-local/uzu-stage
```

起動すると `http://127.0.0.1:8899/` でチャット UI が開く（Kaggle 版と同じ HTML）。
停止はブラウザの「■ 停止」ボタン、または Ctrl+C。

thinking は既定 ON（`UZU_REASONING_EFFORT=default`）。
`UZU_REASONING_EFFORT=disabled` で OFF（速いが多段推論を誤る）。
詳細: `notes/uzu-opencode-server.md`

---

## MTP（Multi-Token Prediction）とは

普通の生成は **1トークンずつ**です。1トークン出すたびに 27B のパラメータを全部通す計算が要ります。
これが 13.7 tokens/s の正体です。

**投機的デコード**はこれを崩します。小さい速いモデルが「次の5トークンはたぶんこれ」と先に予想し、
大きいモデルがそれを**1回の計算でまとめて検証**します。当たっていれば5トークン分の速度が出る。
外れたら予想を捨てて1つだけ採用します。

普通はこの「予想役」に**別の小さいモデル**を用意します（例: 70B に 1B をペアにする）。
**MTP はそれが要りません。** モデル自身に予想用のヘッドが焼き込まれています。

このモデルに入っているのを実測で確認しました。

```
qwen35.nextn_predict_layers = 1
blk.64.nextn.eh_proj.weight
blk.64.nextn.enorm.weight
blk.64.nextn.hnorm.weight
blk.64.nextn.shared_head_norm.weight
```

**利点**: 追加モデルのダウンロードも VRAM 消費もゼロ。予想役が本体と一緒に学習されているので分布が合いやすい。

**期待値**: `nextn_predict_layers = 1` なので1ステップで稼げるのは**1トークンだけ**。
理論上限は約2倍、実際はそれより落ちます。**当たったときだけ得をする**仕組みです。

### 実測値

| 環境 | spec-type | gen tps | 倍率 |
|---|---|---|---|
| ローカル 9B | `none` | 16.02 | — |
| ローカル 9B | `draft-mtp` | **18.85** | **1.18x** |
| T4 x2 / 27B | — | **未測定** | — |

9B の GGUF ヘッダには `nextn_predict_layers` が**ありません**が、MTP は動きます。
起動ログに `creating MTP draft context against the target model` が出れば有効です。

> **T4 x2 での MTP 効果はまだ測れていません。**
> `kaggle/bench_mtp/` の 3回の測定はすべて **CPU 推論**で走っていました。
> ログに `warning: no usable GPU found, --gpu-layers option will be ignored` が出ており、
> `nvidia-smi` の VRAM 使用量も `0 / 0 MiB` です。
> そのため 0.9 → 1.3 tps という数字は **CPU の値**で、MTP の効果を示しません。
> CUDA バックエンドを復元する修正を入れたので、次回の測定で確定します。

### ⚠️ f16 KV + 長いコンテキストは MTP より先に効く罠

llama-server を直起動したとき、**KV キャッシュの型を指定しないと f16 になります。**
Ollama は既定で `q8_0` を使うので、同じ条件のつもりでもここが違います。

T4 x2（30 GiB）で 27B を `-c 131072` + f16 KV で張ると、**モデル 15.66 GiB + f16 KV が
VRAM を溢れて CPU 推論に落ちます。** 実測で **0.8 tokens/s**。前回 Ollama で 13.7 tokens/s
出ていた条件と桁が違うのはこれが原因です。

```bash
# 必ず KV の型を指定する
--cache-type-k q8_0 --cache-type-v q8_0
```

`--spec-type draft-mtp` の有無を比べるだけの検証なら、**ctx は 32768 で固定**して
VRAM に余裕を持たせるのが正しいやり方です。

---

## 技術詳細

### コンテキスト長と KV キャッシュ

GGUF ヘッダの実測値から計算します。

```
full attention 層   17 / 65   （interval 4、残り 48 層は SSM）
kv head             4
key / value length  256 / 256

KV/トークン (f16)  = 17 × 4 × (256 + 256) × 2B = 68 KiB
KV/トークン (q8_0) = 34 KiB
```

| ctx | KV (q8_0) | モデル + KV | T4 x2 (30 GiB) |
|---|---|---|---|
| 8,192 | 0.27 GiB | 16.79 GiB | ✅ |
| 32,768 | 1.06 GiB | 17.58 GiB | ✅ |
| 65,536 | 2.12 GiB | 18.64 GiB | ✅ |
| 131,072 | 4.25 GiB | 20.77 GiB | ✅ **既定**（実測 23.4 GiB） |
| 262,144 | 8.50 GiB | 25.02 GiB | ⚠ 載るが 3.7 tps（実測 24.4 GiB） |

Step2 は実効 VRAM の 88% を予算として、収まる最大のコンテキストを自動選択します。

**既定は 131,072 です。** 131K までは速度が落ちないため（上記）、実用上ここが最適点です。
262K は VRAM には収まりますが（24.4 GiB < 26.4 GiB）、生成速度が 1/3.7 になります。

**`OLLAMA_CONTEXT_LENGTH` は必ず明示してください。** 未設定だと 4096 に落ちます。

### 実測値（T4 x2 / 2026-10-27）

smoke test の結果です（`kaggle/smoke/`）。

| 項目 | 実測 |
|---|---|
| GPU | Tesla T4 × 2（15360 MiB each → 30.0 GiB） |
| Ollama インストール | ✅ |
| モデル pull | **2.6 分**（173 MB/s） |
| 自動選択コンテキスト | **262,144**（当時のロジックは最大を選んでいた。現在は 131,072） |
| 短文（load 含む） | 130.8s（load 86.9s） |
| 日本語 + 回答 | 23.1s → **3.6 tokens/s**（正答） |
| VRAM 使用 | GPU0 13,825 + GPU1 11,145 = **24.4 GiB** |

### コンテキスト長 vs 速度（T4 x2 / 2026-10-27）

`kaggle/bench/` の実測です。同じプロンプト（31トークン入力 / 27トークン生成）で
`OLLAMA_CONTEXT_LENGTH` だけを変えて比較しました。

| ctx | load | prompt tps | gen tps | VRAM | 判定 |
|---|---|---|---|---|---|
| 32,768 | 80.7s | 65.8 | **13.7** | 18.6 GiB | ✅ |
| 131,072 | 10.2s | 73.6 | **13.7** | 23.4 GiB | ✅ **既定** |
| 262,144 | 15.0s | 7.1 | **3.7** | 24.4 GiB | ⚠ 遅い |

**131K までは速度が落ちません。** 32K と 131K が同じ 13.7 tokens/s で、262K で 3.7 tokens/s（約 1/3.7）に落ちます。

> **初回の測定は汚染されていました。**
> `pkill -9 ollama` は名前完全一致のため **`llama-server` 子プロセスが残り**、
> 前のコンテキストの VRAM を掴んだまま次の測定が走っていました。
> 初回は 131K=1.1 tps / 262K=HTTP 500 でしたが、`llama-server` も落として
> VRAM 解放を確認する修正版では上表の通りです。**初回の値は破棄してください。**
>
> 修正前の測定値（参考）: 131K=1.1 tps / 262K=500。
> 同じプロンプトで同じ 27 トークンを生成しているのに 1.1 tps というのは、
> 生成速度として説明がつきません。残留プロセスの影響です。

> **2 GPU 分割（PCIe 経由）は律速ではありません。**
> 16.52 GiB は T4 1枚（15 GiB）に載らないため層分割は避けられませんが、
> 32K でも 131K でも同じ 13.7 tokens/s が出ています。
> 262K の低下は KV キャッシュ 8.5 GiB の確保と attention の計算量によるものです。

### 環境変数

| 変数 | 値 | 理由 |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | 外部からの接続を受け付ける |
| `OLLAMA_ORIGINS` | `*` | CORS |
| `OLLAMA_KEEP_ALIVE` | `-1` | 常時ロード維持（再ロードに数分かかるため） |
| `OLLAMA_CONTEXT_LENGTH` | 既定 **131,072** | 未設定だと 4096 に落ちる。262K は VRAM に載るが 1/3.7 の速度 |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV が半分になり、その分コンテキストに回せる |
| `OLLAMA_FLASH_ATTENTION` | `1` | 有効化 |
| `OLLAMA_NUM_PARALLEL` | `1` | qwen35 は並列が blocklist されている |

### リバースプロキシ（UI 配信 + 同一オリジン化）

cloudflared / ngrok はどちらも「1ポートをそのまま公開」するだけなので、
Ollama の手前に `kaggle/step2/auth_proxy.py` を置いて
**チャット UI の配信**と **`/v1` の同一オリジン化**を担当させます。

ローカル uzu でも同じプロキシを使います（`http.server` の静的配信を置き換え）。
理由は `/stop`（停止ボタン）と `/quota` に動的なエンドポイントが要るためです。
これでローカルも Kaggle も同じ HTML・同じ操作感になります。

| パス | 動作 |
|---|---|
| `/` | `llama-chat.html` を返す |
| `/healthz` | ローカルで即答（上流を見ない） |
| `/quota` | Kaggle GPU 残量を返す。`--no-quota` なら `available:false` |
| `/stop` | `POST` + `X-Kc-Stop: 1`（+ `API_PASS` 時は認証）で応答後に終了。`--stop-pid` があればその PID にも SIGTERM |
| `/api/*` `/v1/*` | 上流へ中継（ストリーミング対応） |

```
cloudflared / ngrok
      │  http://127.0.0.1:11435
      ▼
auth_proxy.py  ── / は llama-chat.html（200）、/api/* /v1/* は Ollama へ中継
      │  http://127.0.0.1:11434
      ▼
ollama serve
```

- 標準ライブラリのみ（`http.server` + `urllib`）。ノートブックには **base64 で埋め込み**
- **既定は認証なし**（`API_PASS` が空なら素通し）。設定したときだけ Basic 認証が有効
- `/healthz` は常に認証免除（トンネル疎通用）
- 上流応答は `Transfer-Encoding: chunked` でそのまま中継 → **token 逐次出力が保たれる**
- リクエストボディは `Content-Length` / `Transfer-Encoding: chunked` の**両方**に対応
- UI 応答は `Connection: close` + ボディ読み捨て（keep-alive のズレを作らない）
- 起動時は「401 でないこと」だけ実測。200 以降は公開URL経由の **verify 5項目**で判定

> 認証を有効にしたい場合のみ環境変数 `API_PASS`（と任意で `API_USER`）を設定します。
> ノート冒頭に `os.environ["API_PASS"] = "..."` を書けば有効になります。

---

## 既知の問題

### Kaggle のマウントパスが移行中（旧新混在）

`/kaggle/input/<slug>`（旧）→ `/kaggle/input/datasets/<owner>/<slug>`（新）へ
移行が進行中で、**同一天内に両方を実測**しました（diag ノートで確認）。
旧形式の直書きをしているノートは model / patch が見つからず、
**16.5 GiB の pull を誘発した**のが v6→v8 の故障原因です。

Step2 は `_find_input()` で深さ2まで探索し空でない方を採用します。
> `kaggle datasets files` の list 表示で size が `0B` と出ても、
> 実体は `kernels output` の diag で中身を確認済（表示異常）。

### `ollama show` が失敗して pull を誘発 — **原因確定済（修正済）**

```
Error: manifest symlink target "/kaggle/input/.../manifests/.../latest"
       is not a sha256 blob
```

- **Ollama はマニフェスト自体を symlink として検査**し、対象が `blobs/sha256-…`
  でないと拒否する。Dataset キャッシュをファイル単位 symlink で組んだとき
  **manifests まで symlink にしたのが原因**（v11 実測の stderr で確定）
- 修正: **manifests は実ファイルでコピー**（1KB 級）。blobs は symlink のまま
  （blob 側に symlink 検査はなく、実測で即認識された）
- pull に落ちてもローカル blob を即認識して1秒で完了するため、
  **16.5 GiB の再ダウンロードは起きていない**

### Dataset 経由の `llama-server` は実行ビットが落ちる（v13 で踏んだ）

```
error starting llama-server: fork/exec /usr/local/lib/ollama/llama-server: permission denied
```

`shutil.copy2` は **mode を保存**するので、tar で上げた Dataset 側に実行ビットが
無いと、そのまま 0755 でない状態が `/usr/local/lib/ollama/` に置かれ、起動時に 500。
→ 復元時に `lib/ollama/` 配下を **無条件 0755 に矯正**（root 実行の `os.access(X_OK)` は
  「x ビットが1つでもあれば True」で全落ちを検出できないため）、
さらに検証セルで実行ビットを確認するようにした（v14 以降）。
サイズ検証だけでは通ってしまう点に注意。

### ネットワーク瞬断で verify が落ちることがある

`URLError [Errno 101] Network is unreachable` / `[-2] Name or service not known`
が数十秒だけ出て、別の項目は通過する（v11 で [2][3] FAIL / [1][4] PASS）。
→ `_with_retry` を **6回・待ち 5/10/15/20/25 秒（合計75秒の窓）** に変更。
HTTP 応答（401/500）は再試行せず即判定する点は従来どおり。

### モデルロードが5分でタイムアウトする（`OLLAMA_LOAD_TIMEOUT`）

Ollama の既定は **5分**（`envconfig.LoadTimeout()`、stall 検出）。
mmap 読込はページキャッシュ次第で揺れ、v10 はこれを超えて
`timed out waiting for llama-server to start` → 500 になった
（v9 は load 198s で通過）。→ env で **`OLLAMA_LOAD_TIMEOUT=600`** に緩和
（warmup の urllib 900s 未満で整合）。

### opencode の `apiKey` は Bearer 扱い（Kaggle 用設定は Basic に直した）

`options.apiKey` を書くと `Authorization: Bearer …` になります。
既定（認証なし）なら無害ですが、`API_PASS` を設定して Basic 認証を有効にすると
401 で死にます。その場合だけ `options.headers` に Basic を明示してください。

```json
"options": {
  "baseURL": "https://<公開URL>/v1",
  "headers": { "Authorization": "Basic <base64(user:pass)>" }
}
```

### ollama#17778 — qwen3.8 の tool call で 500

```
ResponseError during chat streaming: no user query found in messages (status code: 500)
```

- **open / bug ラベル / 53コメント**（2026-08-15 起票、10月時点で未解決）
- tool call をループさせると発生。単発の tool call では出ない
- qwen3.5 / 3.6 では発生しない
- **エージェント用途（opencode 等）で直撃します**

### ollama#17144 — qwen35 の並列リクエスト

`qwen35` / `qwen35moe` は `server/sched.go` のハードコードされた blocklist により
`numParallel = 1` に固定されています。上流の llama.cpp のクラッシュは修正済みですが、
blocklist の削除PRは**未マージ**です。

### ollama#17236 — qwen35 の segfault（closed / not_planned）

修正されずに閉じられました。報告者は AMX バックエンド固有の疑いとしており、T4 は該当しない見込みですが、未解決のままです。

### renderer の不一致

orcarouter の config は `renderer: qwen3.5` / `parser: qwen3.5`。
公式 `qwen3.8` は `renderer: qwen3.8` です。thinking / tool の解釈に差が出る可能性があります（**未検証**）。

### ⚠ CUDA バックエンドが無いと黙って CPU 推論に落ちる

**エラーが出ないので気づきにくい罠です。** llama-server は `libggml-cuda.so` を
`dlopen` で探します。見つからないと警告もなく CPU で走ります。

実測（T4 x2 / 27B / `--spec-type none`）:

| 条件 | gen tps | VRAM |
|---|---|---|
| CUDA バックエンドなし | 0.9 | 0 / 0 MiB |
| CUDA バックエンドあり | 13.7 | 23.4 GiB |

ログの見分け方:

```
warning: no usable GPU found, --gpu-layers option will be ignored
```

この行が出たら CPU 推論です。`nvidia-smi --query-gpu=memory.used` が `0` でも同じ。

**Ollama の Linux 版は CUDA を別配布しています。** `lib/ollama/cuda_v12/libggml-cuda.so`
（439 MB）を正しい場所に置く必要があります。

```bash
# patch Dataset から復元する場合（tar 経由だと実行ビットが落ちるので chmod 必須）
cp -r $PATCH/lib/ollama/* /usr/local/lib/ollama/ && chmod -R 755 /usr/local/lib/ollama
```

### ⚠ llama-server は 15 KB（1 MB 判定で弾かない）

Linux 版の `llama-server` は **15,096 B** の薄いラッパーで、実体は
`libllama-server-impl.so` を `dlopen` します。macOS 版も同じ構造
（`@rpath/libllama-server-impl.dylib`）。

「1 MB 未満は壊れている」という判定をここに適用すると、**正常なバイナリを弾きます**。
サイズ検証はファイルごとに閾値を変えてください。

---

## ローカル Mac（RAM 16 GiB）で動かさない理由

実測結果:

```
llama-server started in 240.01 seconds
prompt processing, n_tokens = 11, progress = 0.73, t = 106.72 s / 0.10 tokens per second
POST /api/chat  →  500  (10分でタイムアウト)
```

**11トークン処理するのに106秒。0.10 tokens/s。**

```
RAM:            16.0 GiB
モデルサイズ:    16.52 GiB
swap:           16.5 GiB / 17.4 GiB 使用済み
空きメモリ:      7%
```

モデル単体で RAM を超えており、全部スワップ経由になります。**量子化を上げても解決しません。**

ローカルで 27B 級を動かすなら RAM 32 GiB 以上が必要です。
16 GiB なら 9B クラス（`huihui_ai/qwen3.5-abliterated:9b` = 6.6 GB）を使ってください。

---

## ローカルで「遅くて話せない」ときの原因

9B クラスでも「返答が来ない」ことがあります。**速度の問題ではなく、thinking が本文を食い潰しています。**

### 症状

`ollama run` で待っても本文が出ない。`/api/chat` で `content` が空のまま。

### 実測（`huihui_ai/qwen3.5-abliterated:9b`、M4 / RAM 16 GiB）

```
think=true   → thinking 1145 chars、content 0 chars、done_reason: length
think=false  → content 317 chars（正常な返答）、9.3 tokens/s
```

生成速度は **9.3 tokens/s** 出ています。遅くはない。返答を書く前に thinking で
トークン予算を使い切るのが「話せない」の正体です。

### 対処

```bash
ollama run huihui_ai/qwen3.5-abliterated:9b --think=false
```

毎回打つのが面倒なら、`~/.zshrc` にショートカットを置きます。

```bash
# ollama: qwen3.5 系は既定で thinking オン。切らないと本文が出ないまま止まる
oq() {
  if [[ "$1" == */* || "$1" == *:* ]]; then
    ollama run --think=false "$@"
  else
    ollama run --think=false huihui_ai/qwen3.5-abliterated:9b "$@"
  fi
}
```

```bash
oq                    # 9b を thinking なしで対話起動
oq "1から5まで"        # プロンプトを渡す
oq <model> "..."      # モデルを指定
```

> **単純な `${1:-デフォルト}` ではプロンプトを渡せません。**
> `oq "こんにちは"` が `ollama run --think=false "こんにちは"` になり、
> プロンプトではなく**モデル名**として解釈されて `Error: invalid model name` で落ちます。
> 第1引数がモデル名らしい（`/` か `:` を含む）かどうかで分岐させてください。

実測で **5.7秒 / 16.78 tokens/s** で即答します。

```
1、2、3、4、5。
何か指示はありますか？
eval rate: 16.78 tokens/s
```

### 効かなかった方法（実測済み）

| 方法 | 結果 |
|---|---|
| Modelfile `PARAMETER think false` | ❌ `Error: unknown parameter 'think'` |
| Modelfile `SYSTEM /no_think` | ❌ 作成はできるが既定は thinking オンのまま |
| `/set nothink`（パイプ経由） | ❌ プロンプトとして解釈される |

`/api/show` のメタデータは `{"values": [false, true], "default": true}`。
**モデル既定が thinking オン**で、SYSTEM プロンプトでは上書きできません。
Modelfile に `think` パラメータは存在しません（[ollama#14809](https://github.com/ollama/ollama/issues/14809)）。

API 経由なら `"think": false` をボディに入れてください。CLI は `--think=false` です。

> 同じ理由で、**Kaggle 側のベンチも `"think": False` を明示しています**（`kaggle/build_bench.py`）。
> 明示しないと thinking が生成トークンを消費し、速度比較が成立しません。

---

## 検証用ノート（GPU 不要）

keepalive ON/OFF と外部受信を GPU 不要で確認できるテストノートを同梱。

```bash
python3 kaggle/build_keepalive_test.py
# → kaggle/step2/step2-keepalive-test.ipynb
```

自動検証項目：

1. keepalive 残り時間の減少
2. 10秒後のチャット送信シミュレーションで残り時間が 60分に戻る
3. 25秒後の停止ボタンシミュレーションで `RUN=0` → 終了
4. auth_proxy.py + cloudflared quick tunnel で外部受信 (`/healthz`, `/quota`, `/stop`)

---

## 運用上の注意

| 項目 | 値 |
|---|---|
| cloudflared quick tunnel | 制限なし。**URL は実行のたびに変わる**・SLA なし |
| ngrok dev domain | **URL 固定**。無料枠は帯域・接続数に上限、**同一ドメインで1本のみ** |
| Kaggle カーネル | 9時間で停止 |
| アイドル | 約40分で Notebook が停止 |
| 初回ロード | 16.5 GiB を VRAM に載せる。**実測 87秒**（`load` は環境で 198秒） |
| キープアライブ | チャット送信ごとに **60分にリセット**、停止ボタンで即終了。最後の送信から 60分経過で自動終了 |
| トンネル死亡 | keepalive が **1回だけ自動再起動**（cloudflared は新URLを `public_url.txt` に書き直す／ngrok は同一ドメインで復帰。2回目以降は fail） |
| tunnel 疎通 | 1回の瞬断は許容、**通算2回で fail**（`tunnel=<name>:DEAD x2`。カウンタは再起動時のみリセット） |

### セキュリティ

- **既定は認証なしです。** URL を知っていれば誰でもあなたの GPU を使えます。
  閉じたいときは環境変数 `API_PASS` を設定（Basic 認証が有効になる）
- **Secrets は基本不要**（cloudflared はアカウント・API キー不要）。
  ngrok 版だけ `NGROK_TOKEN` が必須
- **cloudflared 版は公開URLが毎回変わる**ため、旧URLを知っている相手には届きません
  （quick tunnel は実行ごとに新トンネルを張るため）
- **ngrok 版は URL が固定**なので、この防御は無い。URL を知られると使われます
- 認証を有効にした場合、`/kaggle/working/connection_info.txt` に
  Basic 認証情報が平文で残ります。不要になったら削除してください

---

## 参考リソース

- [Abliterlitics: Qwen3.8-27B 12変体比較](https://abliterlitics.dev/models/qwen38-27b/) — 本ドキュメントの順位の出典
- [orcarouter/Qwen3.8-27B-Uncensored (Ollama)](https://ollama.com/orcarouter/Qwen3.8-27B-Uncensored)
- [Cloudflare: Try Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/use-cases/tunnel-ingress-rules/) — Step2 が使う公開手段
- [GitHub: cloudflare/cloudflared](https://github.com/cloudflare/cloudflared) — リリースバイナリの出所
- [Qiita: Kaggle × ngrok で無料LLM APIサーバーを利用する](https://qiita.com/mhuk_mtd/items/4ecf1ebc1fd047ef4ab7) — ngrok 版の参考
- [ngrok: Domains](https://dashboard.ngrok.com/domains) — 固定URL（dev domain）の取得
- [GitHub: cyberytti/OllamaKaggle](https://github.com/cyberytti/OllamaKaggle)
- [Kaggle](https://www.kaggle.com) / [Ollama](https://ollama.com)

## 開発時のチェック

```bash
python3 kaggle/step2/_selftest_proxy.py     # プロキシ
python3 kaggle/step2/_selftest_builder.py   # ノート内関数
python3 kaggle/check_notebooks.py           # auth_proxy.py / llama-chat.html を直したら再生成漏れが無いか
ruff check .
```

同じものを GitHub Actions（`.github/workflows/ci.yml`）でも回している。

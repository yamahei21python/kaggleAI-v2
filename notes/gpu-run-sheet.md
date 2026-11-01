# GPU 実行シート（2026-10-30 作成）

> ⚠ **2026-11-01 に ngrok を廃止したので、#3（llama-server）と #4（ngrok 実配線）は
> 実行できない。** 該当カーネルは retired で、`cli push` が「退役」で止まる。
> 生きている手順は #1 / #2（どちらも cloudflared）。URL は
> `python3 -m cli beacon` で読む。以下は当時の記録として残す。

クォータが戻ったら、この順で回す。**カーネルの作成は済んでいる**ので、やるのは実行だけ。

- 作成は `kaggle kernels push --no-run`（= Quick Save 相当）で行った。**1 セルも実行して
  いないので GPU は 1 秒も消費していない**。
- 実行順と合否の考え方は `notes/gpu-smoke-plan.md`。ここは「今日そのまま打てる形」だけ。

> **✅ #2（vLLM）の前提は解消済み（2026-10-31 訂正）**
> 2026-10-30 に「`qwen38-27b-uncensored-hauhaucs-awq-mtp` に重みが無い」と書いたが、
> これは **`kaggle datasets files` のページ送り（既定 20 件）による誤読**だった。
> `.cache/*` が名前順で先頭に並ぶため 1 ページ目だけ見ていた。`--page-size 200` で
> 取り直すと `model.safetensors` **18.70 GB** と `model-mtp.safetensors` 0.85 GB が
> 2026-10-28 から入っている（`totalBytes` = 19,567,955,256 B と一致）。
> **Dataset の作り直しは不要。** 詳細と教訓は `notes/dataset-inventory.md`。

## 1. 今の状態

| # | 種 | カーネル id | 種別 | バージョン | Kaggle の状態 |
|---|---|---|---|---|---|
| 1 | Ollama | `yamahei21python/step2-orcarouter-qwen38-27b-cloudflared-script` | script | 1（未実行） | ERROR（未実行の既定値） |
| 2 | vLLM | `yamahei21python/step2-vllm-qwen38-27b-cloudflared-script` | script | 1（未実行） | ERROR（同上） |
| 3 | llama-server | `yamahei21python/llama-server-ngrok-script` | script | 1（未実行） | ERROR（同上） |
| 4 | ngrok 実配線 | `yamahei21python/step2-orcarouter-qwen38-27b-ngrok-script` | script | 1（未実行） | ERROR（同上） |
| 参考 | notebook 版 | `yamahei21python/llama-server-ngrok` | notebook | 1（未実行） | COMPLETE（同上） |

`ERROR` は**失敗したのではなく「まだ 1 回も実行していない」**という意味。script
カーネルには「保存だけした」状態が無いため、初回はこの表示になる。実行すると
`RUNNING` → `COMPLETE` に変わる。

作成時に保存された設定（`kaggle kernels pull -m` で確認済み）:

```json
"kernel_type": "script",  "enable_gpu": true,  "enable_internet": true,
"machine_shape": "NvidiaTeslaT4",
"dataset_sources": ["yamahei21python/my-ollama-orcarouter-qwen38-27b",
                    "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch"]
```

### 台帳の GPU カーネル 20 本は全部できている

2026-10-30 に `--no-run` で一通り作成し、`kaggle kernels status` で 20/20 の存在を
確認済み。エンジンは 6 種そろっている。

| エンジン | notebook | script |
|---|---|---|
| Ollama | `step2-orcarouter-qwen38-27b-cloudflared` / `-ngrok-fixed` | `…-cloudflared-script` / `…-ngrok-script` |
| vLLM | `step2-vllm-qwen38-27b-cloudflared` / `-ngrok` | `…-cloudflared-script` / `…-ngrok-script` |
| llama-server | `llama-server-ngrok` | `llama-server-ngrok-script` |
| ExLlamaV3 | – | `exl3-serve` / `exl3-qwen38-gpu-check` |
| LMDeploy | – | `step2-lmdeploy-qwen38-27b-cloudflared` |
| SGLang | – | `step2-sglang-qwen38-27b-cloudflared` |

加えて測定系（`smoke` / `bench` ×2 / `bench_mtp`）の notebook と script。

## 2. クォータ

| | 値 |
|---|---|
| 使用済み | 29h53m / 30h00m（残り 7 分） |
| **リセット** | **2026-10-31 09:00 JST**（`quotaRefreshTime` = `2026-10-31T00:00:00Z`） |

`python3 -m cli quota` でいつでも確認できる。7 分では 1 本も完走しないので、
**リセット後にまとめて 3 本**が正解。

## 3. あなたがやること（設定）

### 3-1. 何も要らない（#1〜#3）

**smoke は Secrets を 1 つも使わない。** `kbuild/serve_script.py` は `SMOKE` のとき
proxy / トンネル / Gist / keepalive を丸ごと飛ばし、エンジンへ直接
`/v1/chat/completions` を投げて終わる（`serve_script.py` の 649 行目）。

```python
if SMOKE:
    verify(None, f"http://127.0.0.1:{PORT}", key="verify_direct")   # proxy なし
```

なので #1〜#3 は**設定ゼロで回る**。アクセラレータ（T4 x2）と Internet も
作成時のメタデータで入っている。

### 3-2. #4 のときだけ Secrets

`step2-orcarouter-qwen38-27b-ngrok-script` を `--smoke` **無し**で回す回だけ必要。
<https://www.kaggle.com/settings> → API → Secrets。

| Secret | 必須か | 無いとどうなる |
|---|---|---|
| `NGROK_TOKEN` | **必須** | `ngrok config add-authtoken` が失敗し、トンネルが張れない |
| `GIST_ID` + `GITHUB_TOKEN` | 実質必須 | 公開 URL の共有（Gist 更新）ができない。`result.json` は残る |
| `NGROK_DOMAIN` | 任意 | 埋め込み既定値にフォールバックする |
| `API_USER` / `API_PASS` | 任意 | 空だと proxy の Basic 認証が無効になる（URL を知っていれば誰でも叩ける） |

`secret()` は取れなければ `""` を返すので、**未登録でもカーネルは落ちない**。
落ちるのは「トンネルが張れない」段になってから。事前に入れておくのが吉。

### 3-3. 一応の確認（UI・30 秒）

- 各カーネルの Settings で **Accelerator = GPU T4 x2** になっているか
  （メタデータで `NvidiaTeslaT4` を確認済みなので、通常はそのまま）。
- **Internet = On**（同上）。

## 4. 実行（リセット後）

```bash
python3 -m cli quota                       # 残りが戻ったか確認
python3 -m cli push <name> --smoke --yes    # 生成 → push → 実行
```

| # | コマンド | 1 回 |
|---|---|---|
| 1 | `python3 -m cli push step2-orcarouter-qwen38-27b-cloudflared-script --smoke --yes` | 8〜12 分 |
| 2 | `python3 -m cli push step2-vllm-qwen38-27b-cloudflared-script --smoke --yes` | 8〜25 分 |
| 3 | `python3 -m cli push llama-server-ngrok-script --smoke --yes` | 15 分 |
| 4 | `python3 -m cli push step2-orcarouter-qwen38-27b-ngrok-script --yes`（smoke 無し） | 10〜15 分 |

`push` は毎回その場で再生成してから押すので、手元の生成物が古くても事故らない。
`--smoke` は `smoke-<tunnel>/` を staging するので通常版は壊れない。

結果の取得:

```bash
kaggle kernels output yamahei21python/<slug> -p out.<n>/
```

## 5. 判定

`result.json` を見る（smoke でも出る）。

| キー | 合格 |
|---|---|
| `stage` | `done` |
| `ok` | `true` |
| `verify_direct.text` | 空でない |
| `verify_direct.checks` | 全部 `true` |
| `stop_remaining` | `[]` |
| `gpu.names` | T4 x2 |

#4 だけは加えて `url` が `https://<NGROK_DOMAIN>` / `tunnel == "ngrok"` /
`ngrok_domain` が入っていること。

## 6. 失敗したときの切り分け

`stage` で見る。

| stage | 意味 | 次手 |
|---|---|---|
| `install` | 依存の復元で落ちた | `engine_log_tail`、Dataset のマウント先（`/kaggle/input/...`） |
| `load` | 重みが VRAM に載らない | backend / 量子化 / `--tp` |
| `verify` | API の形が違う | エンジンの起動 argv、ポート |
| （#4）トンネル | ngrok が張れない | `ngrok.log` の authtoken エラー、Secrets |

**1 本失敗したらその日は次へ進まない**（pip の再試行でクォータが溶ける）。

## 7. 補足: `--no-run` について

今回カーネルを作るのに使った。`kaggle` CLI の **2.x にしか無い**フラグ。

```bash
python3 -m cli push <name> --no-run --yes     # 保存だけ（実行しない）
```

1.7.4.5（`~/Library/Python/3.9/bin/kaggle`）には無いので、リポジトリ内に
2.x を置いてある（`.venv-kaggle-cli/`、`.gitignore` 済み）。`cli push` は
`--no-run` を要求されたときだけ 2.x を探し、無ければ作り方を出して止まる。
別の場所の 2.x を使うなら `KC_KAGGLE=/path/to/kaggle` を指定する。

### 7.1 `予約中` は「消費」ではなく「保留」（2026-11-01 実測）

`cli quota` の `予約中` は**実行中のセッションが握っている上限**で、終われば戻る。
実際に v15 で確認した:

| | 残り | 予約中 |
|---|---|---|
| v15 実行中 | 7h05m | 11h52m |
| v15 終了後 | **18h39m** | 0 |

引かれるのは**実際に走った秒数だけ**。だから「5 分の smoke を撃ったら残りが
12 時間減った」は勘定の見え方であって、溶けてはいない。

ただし**副作用はある**: GPU セッションは同時 2 つまでなので、1 本走っている間は
次の push が `Maximum batch GPU session count of 2 reached` で弾かれることがある
（実測では RUNNING が 1 本でも弾かれた。終了直後のセッションがまだ数えられている
と見られる）。`--no-run` はセッションを作らないので、**この制限には当たらない**。

```bash
# 実行せずに置く（クォータ 0・セッション 0）
python3 -m cli push vllm-playground --no-run --yes
```

`--no-run` で分かった副産物: 存在しない Dataset を `dataset_sources` に入れると
`not valid dataset sources ...` と警告が出る。実行時は `/kaggle/input/<slug>` が
無いだけなので**静かに劣化する**。実際 `vllm-warm-cache` が未作成だと分かった
（→ `vllm-startup-speedup.md` §18.3）。

## 8. 実測結果（2026-10-31・クォータ復活後）

| # | エンジン | 結果 | 実行時間 | 主な数値 |
|---|---|---|---|---|
| 1 | Ollama | **✅ ok / stage=done** | 162s | ctx **131,072** / `23 GB 100% GPU` / **13.3 t/s**（load 117s） |
| 3 | llama-server | **✅ ok / stage=done** | 345s | 6/6 PASS / VRAM **9.8 GB ×2** / **14.2 t/s**（MTP on） |
| 2 | vLLM | **✅ ok / stage=done** | 929s | 5 PASS + 1 SKIP / `vLLM 0.32.0 / torch 2.13.0+cu130` / **31.7 t/s** / VRAM **13.3 GB ×2**（install 210s + 起動 642s） |
| 5 | keepalive / `/stop` | **✅ 検証済み（CPU・GPU 0 秒）** | 約 2 分 | `reset_done=True` / `stop_done=True` / 12s で 60分00秒 に戻る / 30s で `RUN=0` 終了 |
| 6 | LMDeploy | **❌ 不可（形式非対応）＝ 判定済み** | 125s | `ValueError: Unsupported compressed-tensors group_size: expected 32, got 128` |
| 7 | SGLang | **❌ 不可（sm75 非対応）＝ 判定済み** | 297s → 610.6s | `Quantization scheme is not supported for the current GPU. Min capability: 80. Current capability: 75.` |
| 4 | ExLlamaV3 | **⚠ 起動はしたが再現しない** | 263.6s / **1834.7s** | v2: `loaded in 229s; serving qwen38-exl3 on 127.0.0.1:9931 (ctx 8192)`。落ちたのは骨格の `engine_checks` の引数不一致（`5 were given`）で、エンジンではない。**v3: 同じ構成で 1800 秒待っても `/health` が繋がらず**（`chat_server.log` は 0 バイト） |

`[1] ui` は 3 本とも **SKIP**（smoke は proxy を立てないため。下記 (a)）。

**「ERROR」は 3 種類が混ざっている。** ここまで来て、LMDeploy と SGLang は
**判定済みの不可**、ExLlamaV3 だけが**「動く」とも「動かない」とも言えない**:

| 種類 | 意味 | 該当 |
|---|---|---|
| 判定（＝結果） | エンジンまで到達して、能力不足が確定した | **LMDeploy**（形式）/ **SGLang**（sm80 必須） |
| 途中で落ちた | install / 梱包で死に、エンジンは起動していない | 解消済み（`.py` を notebook として push した件） |
| **再現しない** | 1 回は起動したが、次の回で起動しない | **ExLlamaV3**（v2 成功 229s → v3 失敗 1800s） |
| 未実行 | push していない・scaffold だけ | 台帳の未着手分 |

**1 回の成功を「動く」と読まない。** ExLlamaV3 は同じモデル・同じ wheel・
同じ T4 x2 で 229 秒と 1800 秒超に割れた。原因が判るまでは「T4 で動くのは
確実、安定性は未確定」と書く。v3 で計器（`load_watch`）を入れたので、
v4 は「どこで止まったか」を持ち帰れる。

**SGLang の install 原因（2026-11-01 に PyPI メタデータで特定・GPU 0 秒）**:
`sglang==0.5.22` → `outlines==0.1.11` → **`outlines_core==0.1.26`（完全固定）**。
`outlines_core` は Rust 実装で、**0.1.26 には cp313 の wheel が無い**
（0.1.x で cp313 があるのは **0.1.17 だけ**。0.2.9 以降は cp313 あり）。
Kaggle の Python は 3.13.15 なので sdist からのビルドになり、Rust ツールチェーンが
無いため失敗する。**ここは install の壁で、sm75 の話ではなかった**（下の v4 で別の壁）。

#### SGLang の 4 回（1 つずつ壁を剥がしたら、最後に**非対応**が出た）

| 回 | 何を直した | 実測 | 結果 |
|---|---|---|---|
| v2 | —（素の pip） | 297s | ❌ `Failed building wheel for outlines_core` |
| v3 | `PIP_STEPS` + **rustup を先に入れる** | install **388.0s**（rustup 12.7 / pip 375.3） | ✅ install 成功 → ❌ 起動で torchaudio 不整合 |
| v4 | install の最後に **torchaudio の確認と修理** | install **513.3s**（12.6 / 488.4 / 12.3） | ✅ install 成功 → ❌ **量子化スキームが sm80 必須** |
| — | — | — | **SGLang は T4 で不可（確定）** |

v3 の起動失敗は vLLM と同じ落とし穴（#2）:

```
RuntimeError: Detected that PyTorch and TorchAudio were compiled with different
CUDA versions. PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8.
```

`sglang` が `torch==2.14.1`（+cu130）を入れる一方、イメージ残りの
`torchaudio 2.8.3+cu128` が残ると、transformers → torchaudio の**遅延 import** で落ちる。
PyPI の `torchaudio==2.11.0` は **+cu130 ビルド**（wheel の `version.py` を実際に
開いて確認）なので、`--force-reinstall --no-deps` で置き換える。v4 で
`torch 2.14.1+cu130 / torchaudio 2.11.0+cu130 / import OK` まで確認できた。

**この回で分かった計器の穴**: ステップの出力を成功時に捨てていたため、
「どの版が入ったか」がログに残らなかった。→ `stage_install()` が
**全ステップの末尾 12 行を必ず保存・表示**するように変更した。

#### v4 の判定（これが SGLang の答え）

install を全部通したあと、モデルの量子化を解釈する段で止まる:

```
File "sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py", line 1077
    self._check_scheme_supported(scheme.get_min_capability())
RuntimeError: ('Quantization scheme is not supported for ', 'the current GPU.
Min capability: 80. ', 'Current capability: 75.')
```

上流のコードで裏取り（GPU 0 秒）:

- `CompressedTensorsWNA16.get_min_capability()` は **80** を返す
  （`schemes/compressed_tensors_wNa16.py:94`）。T4 は **75**。
- 同じファイルのコメントが「symmetric / asymmetric とも **Marlin カーネル経路**で
  扱う」と明記している。Marlin は sm80+ 専用。
- Triton 実装は `CompressedTensorsWNA16TritonMoE`（**MoE 専用**・ROCm 向け）しか
  無く、Linear 層には sm75 の逃げ道が無い。

→ **LMDeploy と同じ「設計上の非対応」**。ただし根が違う:
LMDeploy は `group_size=128` を読めない、SGLang は読めるが**カーネルが sm80 必須**。

**教訓**: `get_min_capability()` は上流のソースに書いてある。**GPU で確かめる前に
読めた**（今回 2 往復 = 約 18 分を GPU で使った）。次のエンジンは
「対応 GPU の下限」を先に読む。

**#2 の内訳**（`vllm serve` は T4×2 / TP=2 / MTP on）:
install 210s → `vllm serve --help` 30s → 起動待ち 642s（うち `torch.compile` 61s、
CUDA graph capture 2 本）→ 会話。起動待ちが長いのは T4 で compile するため。

### v10（2026-11-01 02:46 完了・**計測機構を入れた回**）

`step2-vllm-qwen38-27b-cloudflared-script` を v10 で push、`COMPLETE` まで 90 分。
**今回はじめて `result.json` の `timings` が数字で埋まった**（v9 までは `null`）。

| 項目 | 実測 |
|---|---|
| install | **271.3s**（pip 230.4 / torchaudio 9.7 / `serve --help` 27.4 / import 3.8） |
| 起動 | **706.5s** |
| 起動まで合計 | **977.8s**（16.3 分） |
| 配信 | 61 分（`stopped_by: budget`） |
| verify | 6/6 PASS（ローカル + 公開 URL の両方） |
| 速度 / VRAM | 31.0 t/s（ウォームアップ込み） / **13,343 MiB ×2** |
| tunnel | cloudflared **Dataset から復元**（`tunnel_seconds` 4.0） |
| 試行 | **1 発目（`mtp+dd`, max_model_len 65536）で成功** |

起動 706.5s の内訳（ログ時刻 1 秒精度。**推定ではなく実測**）:

| 区間 | 時刻 | 長さ | 割合 |
|---|---|---|---|
| プロセス起動〜`load_format` | 0 → 75 | 75s | 11% |
| 重みロード（`Loading weights took`） | 75 → 208 | **133s** | 19% |
| `Model loading took` → compile 開始 | 208 → 221 | 13s | 2% |
| torch.compile | 221 → 282 | **61s** | 9% |
| **無音区間（下記）** | 282 → 663 | **381s** | **54%** |
| CUDA graph capture | 661 → 663 | **2.2s** | 0.3% |
| `init engine` 〜 API server | 663 → 706.5 | 43s | 6% |

**capture は 41s ではなく 2.2s だった。** v9 の「41s」は 60 秒ポーリングの
粒度で測った値で、実際は無音区間の末尾を拾っていた（`--enforce-eager` の
見込み -41s は**ほぼ丸ごと幻**）。

#### 381 秒の無音区間（起動時間の 54%）

`torch.compile took 61.00 s` の直後から `Graph capturing finished` まで、
**381 秒**（282 → 663）。この間に出るログは 60 秒おきの

```
(EngineCore pid=260) INFO 16:33:36 [shm_broadcast.py:871]
No available shared memory broadcast block found in 60 seconds. This typically
happens when some processes are hanging or doing some 'time-consuming' work
such as ray remote call, compilation, etc.
```

**5 回だけ**。vLLM 自身が「時間のかかる作業中」と言っている＝**症状であって原因ではない**。
同時刻の `nvidia-smi` は `util=0%, mem=10,733 MiB` で、**GPU は遊んでいる**。
VRAM は 9,995 → 10,733 MiB へ漸増（KV cache 13,343 MiB の確保前）。
→ ホスト側（CPU）の作業。compile 後の dummy run / memory profiling の
Triton JIT が第一候補だが、**断定はしない**（下記の計器で次回確定させる）。

v9 でも**同じ 5 回の警告が同じ位置に出ていた**（13:25:53〜13:29:53）ので、
再現性のある構造的な待ちであって、その回だけの偶発ではない。

#### v10 で見つけて直した不具合 3 件

**(1) 成功した回の `vllm_*.log` が成果物に残らない（最重要）**
`save_output(f"vllm_{idx}.log", ...)` が **失敗経路にしか無く**、成功時は
`break` がその行を飛ばしていた。だから 381 秒の中身（compile 直後に出た
2,281 B のログ）を**読めなかった**。→ 成功経路でも保存する。
回帰テスト: `test_successful_attempt_log_is_saved`。

**(2) 無音区間が `result.json` に残らない**
区間の長さ（`startup_spans`）は出るが「何の直前/直後か」が無いと次の一手が
決まらない。→ `_long_gaps()` を追加し、無音の**前後の行**を `startup_gaps` に
残す。60 秒警告の繰り返しは 1 本に連結する（5 本並ぶと「60 秒が 5 回」に
見えて「381 秒詰まった」に見えない）。回帰テスト 5 本。

**(3) warm cache Dataset の判定が常に真（静かな素通り）**
`kc_find_dataset()` は**未マウントでもパスを返す**ので `if _wc:` が常に真。
tar が 1 本も無いときは**何も表示せず**通過していた（v10 のログに
warm cache の行が 1 行も無い）。→ tar の有無で分岐し、未マウントと
「tar が入っていない」を切り分けて表示する。

### v11（2026-11-01 04:05・**ERROR / 環境要因 + 掃除のバグ 2 件**）

v11 は smoke で push したが **6 試行すべて失敗**（GPU 約 50 分を消費）。
原因は 2 層に分かれている。**コードの退行ではない**（同じ argv の v10 は成功）。

**層 1: 環境**（こちらでは選べない）
試行 1 は重みロードの途中で **840 秒間 1 行も進まなかった**。ログの最終行は
`Loading safetensors checkpoint shards: 50%`、その直前が

```
Filesystem type for checkpoints: FUSE.GCSFUSE. Checkpoint size: 18.21 GiB.
Available RAM: 6.12 GiB.
Auto-prefetch is disabled because the filesystem (FUSE.GCSFUSE) is not a
recognized network FS (NFS/Lustre) and the checkpoint size (18.21 GiB)
exceeds 90% of available RAM (6.12 GiB).
```

**空き RAM 6.12 GiB で 18.21 GiB を mmap** していた。v10 と同じコードで、
その回だけ空き RAM が少ないマシンに当たった（Kaggle は同居プロセスの分だけ
`MemAvailable` が減る）。

**層 2: 掃除のバグ（こちら）** — ここが本当の損失
試行 1 が固まったあと、Worker が 2 枚で **18,194 MiB（9,097 × 2）を掴んだまま
残った**。以降 5 試行は毎回

```
ValueError: Free memory on device cuda:0 (5.44/14.56 GiB) on startup is less
than desired GPU memory utilization (0.9, 13.11 GiB)
```

で 60〜90 秒で即死。**同じ理由で 5 回**、約 900 秒を捨てた。原因は 2 つ:

1. `pkill -9 -f "VLLM::WorkerTP"` の綴りが違う。実際のプロセス名は
   `VLLM::Worker_TP0`（`setproctitle` が付ける）。**アンダースコアが無いので
   一致せず、静かに失敗**していた。`VLLM::EngineCore` だけは一致していたので
   「掃除した」ように見えていた。
2. 掃除の**結果を判定していなかった**。`_gpu_used_mib()` を見て警告を出すだけで、
   解放できていなくても次の試行へ進んでいた。

→ 修正（回帰テスト 4 本）:
- パターンを `VLLM::Worker` に直す（TP0/TP1 両方に一致）。
- 名前ではなく**ドライバに PID を聞いて落とす**（`nvidia-smi --query-compute-apps=pid`
  → `os.kill`）。名前一致は `setproctitle` の綴りに依存する＝偶然に頼ることになる。
- `_free_vram()` が**実際に空いたか**を返し、空かなければ `fail()` で即座に止める
  （0.9 × VRAM を確保できないので、起動しても必ず即死する）。
- 空き RAM < 8 GiB なら起動前に警告（ハングの予兆）。

**教訓**: 「掃除した」は**結果で確かめる**。やったかどうかではなく、VRAM が
実際に減ったか。名前で殺すのは綴りに合わせるしかないので、ドライバに聞く方が確実。

### v12（2026-11-01 10:03・**ERROR / uv が 4.4 倍遅く、重みロードは再びハング**）

v11 の修正（掃除・VRAM 判定・署名）は**効いた**。にもかかわらず失敗した。
今回の損失は 51 分（install 20 分 + 起動 31 分）。

**良くなったこと（v11 の修正は正しかった）**
```
🧽 /dev/shm の残骸 2 件を削除
🔫 GPU を掴んでいる PID 392 を停止
🔫 GPU を掴んでいる PID 393 を停止
🧹 停止後の VRAM 使用量: 0 MiB
```
試行 2 は「Free memory ... is less than desired」で即死せず、**実際に起動を始めた**。
掃除の結果を見る修正が効いている。

**層 1: uv が本番でだけ遅い（-890 秒の損失）**

| 経路 | install | 内訳 |
|---|---|---|
| v10 pip + wheelhouse | 271.3s | pip 230.4 |
| v11 pip + wheelhouse | 304.2s | pip 261.6 |
| **v12 uv + wheelhouse** | **1208.5s** | **uv 1151.7** |

CPU ベンチでは uv 35.7s / pip 200.1s（5.6 倍速い）だった。**比が本番に移らない。**
違いは入れる先: bench は `--target` の**まっさらなディレクトリ**、本番は torch を
物理削除した **site-packages**。同じ wheel・同じホストで 4.4 倍**遅い**。
理由は未同定（uv のキャッシュ経路が GCSFUSE 上の wheel を引き直している疑い）。

→ **既定を pip に戻した。** uv は `KC_INSTALLER=uv` を明示した回だけ。
回帰テスト `test_installer_defaults_to_pip` で「既定が pip」を固定。

**層 2: 重みロードのハングは再現した（2 試行連続）**

```
Filesystem type for checkpoints: FUSE.GCSFUSE. Checkpoint size: 18.21 GiB.
Available RAM: 5.68 GiB.
Auto-prefetch is disabled ...
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:02<00:02]
（ここから 800 秒、ログ +0 バイト / GPU util 0% / VRAM 9,097 MiB で固定）
```

**注目すべきは、試行開始の 4 分前に `空き RAM: 30315 MiB` と出ていたこと。**
その 24 GiB がどこへ消えたのかが分からない。v10 は同じ symlink ステージング・
同じ FUSE で重み 133 秒だったので、**v10 は auto-prefetch が有効だった**と推定
できる（有効なら 18 GiB を RAM に一括で読む＝速い／無効だと頁単位の mmap）。

→ 空き 1 個の数字では原因を決められないので、**内訳を毎回残す**ようにした:
`MemTotal / MemFree / MemAvailable / Cached / Shmem` + **RSS 上位 5 プロセス**。
起動待ちのループの中で出す（成功時だけでは固まった回に何も残らない）。

**次にやること（GPU を賭ける前に CPU で切り分ける）**
`vllm-io-bench`（GPU 0 秒）を追加した。GCSFUSE の逐次読み / mmap / ローカル実コピー /
コピーの読み直しを測り、各フェーズの meminfo を残す。「読むと空きが減る」なら
コピーでは解決しないと分かる。

### 途中で見つけて直した不具合 10 件

**(a) smoke の `[1] ui` が誤判定（設計ミス）**
`/` が「proxy が返すチャット HTML（10 KB 超）」であることを検査していたが、smoke は
proxy を立てずエンジンを直叩きするので `/` はエンジン自身の応答になる。Ollama は
`"Ollama is running"`（**17 B**）を返すため必ず FAIL し、エンジンが正常でも run 全体が
ERROR になった（初回の #1）。
→ `engine_checks(..., direct=False)` を追加し、direct 時は `judge(1, "ui", None, "skip: …")`。
`judge` は `ok=None` を SKIP として扱い、failures に数えない（PASS とも書かない）。
回帰テスト: `tests/test_smoke_checks.py`（4 本）。

**(b) llama-server が CPU 推論に落ちる（バックエンド未登録）**
Ollama は GPU バックエンドを `lib/ollama/cuda_v12/` という**サブディレクトリ**に置き、
Go 側（`bin/ollama`）がパスを明示して読み込む。`llama-server` を直起動すると ggml の
探索は実行ファイルの隣（`$ORIGIN`）だけになり `cuda_v12/` が見えない → 起動はするが
CPU 推論（`libllama-common.so`: `no usable GPU found`）。
→ `GGML_BACKEND_PATH` に **`cuda_v12/libggml-cuda.so` のフルパス**を渡す。
  - **コロン区切りで複数渡すと失敗する**: `load_backend: failed to load <その文字列>`。
    値は「ディレクトリ」でも「リスト」でもなく **.so ファイル 1 本**（実測 2 回）。
  - 直下への symlink では効かない（`$ORIGIN` 走査は cuda を拾わない。実測）。

**(c) `libcuda.so.1` が `LD_LIBRARY_PATH` に無い**
(b) を直すと次は `libggml-cuda.so: libcuda.so.1: cannot open shared object file` で
失敗する。ドライバの場所はイメージ依存なので、候補（`/usr/local/nvidia/lib64` など）を
順に試し、無ければ `find /usr /opt -maxdepth 6 -name libcuda.so.1` で探して
`LD_LIBRARY_PATH` に足す。`ldd` の結果も同じ env で取ってログに出す。

**(d) 親プロセスが先に import した torch が、入れ替え後も生き残る（#2 の真因）**
`stage_env()` が `import torch` して `torch.__version__` と
`torch.cuda.is_available()` を出していた。この時点ではイメージの 2.11。
その後 vLLM のインストール段で torch を 2.13 に差し替えると、
**メモリ上は 2.11 の `torch.utils._config_module`、ディスク上は 2.13 の
`torch/_dynamo/config.py`** という食い違いが起きる。結果:

```
TypeError: Config() got an unexpected keyword argument 'deprecated'
```

これが「vLLM のバグ」に見えて 8 往復溶かした。**決め手は「子プロセスでは成功する」**:
`python -c "import torch, vllm"` も `vllm serve --help` も通るのに、親の
`import vllm` だけ落ちる。原因を torch の版だと思っていたのが誤り。
→ `stage_env()` の torch 確認を**子プロセス**に移した（`kbuild/serve_script.py`）。
親はインストールが終わるまで torch を一切 import しない。CUDA 文脈の二重初期化も防げる。

**(e) `pip uninstall torch` では消し切れない**
uninstall は RECORD に載っているファイルしか消さない。**新しい版で削除された
ファイル**は残るので、上書きインストールすると新旧が混ざる。
→ site-packages の `torch` / `torchvision` / `torchaudio` / `torchgen` /
`functorch` と各 `*.dist-info` を `shutil.rmtree` で**物理削除**してから入れる。
`torch-*` はハイフン付きなので torchmetrics / torchcodec を巻き込まない。

**(f) torchaudio の版は torch の版ではない**
`torchaudio==torch.__version__` で入れようとすると 2.13.0 を探して存在せず無駄にする。
vllm 0.32.0 の METADATA は `torch==2.13.0` と同時に **`torchaudio==2.11.0`** を要求する。
→ `importlib.metadata.requires("vllm")` から読む（`_vllm_pin()`）。
さらに **版が同じでもビルドが違えば落ちる**（`+cu128` と `+cu130`）。pip は
`2.11.0+cu128` を `==2.11.0` の充足とみなす（PEP 440）ので、`--force-reinstall` で
PyPI の `+cu130` に置き換える必要がある。

**(g) 失敗したときのログを出さないと 1 往復 6 分を無駄にする**
インストール直後に**子プロセスで** `import torch, vllm` を試し、落ちたら
`_torch_diag()`（site-packages の `torch*` 一覧と `_config_module.py` に
`deprecated` があるか）を出して**即座に止める**ようにした。torchaudio の調整に
進んでも直らない種類の失敗なので、先へ進むだけ時間を捨てる。

**(h) `find_model()` が AWQ を取り逃す（LMDeploy / SGLang が即死）**
`quantization_config.json` を**別ファイルで持つ形式**しか受け付けていなかった。この
AWQ Dataset は `config.json` の中に `quantization_config` を埋め込む形式なので、
`モデル Dataset が見つからない` で 30 秒で落ちる。Dataset は正しくマウントされていた
（`kaggle datasets files --page-size 200` で確認: `quantization_config.json` は無い）。
→ 判定を「`config.json` + `*.safetensors` が実在するか」に変えた。形式を推測するより確実。
失敗時は `/kaggle/input` の中身を出すようにもした（未マウントとの切り分け）。

**(i) エンジンが死んでも待機が続く（1 往復 46 分の浪費）**
LMDeploy は turbomind が重み 67% で CUDA OOM して **6 分で終了**していたのに、
`wait_http(timeout=2400)` が **40 分待ち続け**、run 全体が 46 分になった。
→ `wait_http` が `kc_pids(ENGINE_PATTERN)` でプロセスの生存を確認し、3 回連続
（≒15 秒）消えていたら即座に `エンジンが終了した` で諦める。起動直後は exec 前で
pgrep が空になるため、60 秒の猶予を置いてから判定する。

**(j) LMDeploy はこの AWQ を読めない（設定では直らない）**
turbomind は重み 43/64 で `CUDA error: out of memory`、pytorch バックエンドに替えると
`ValueError: Unsupported compressed-tensors group_size: expected 32, got 128`
（`lmdeploy/pytorch/config.py:47`）。このモデルの量子化は
**compressed-tensors / group_size=128** で、LMDeploy 0.18.0 の compressed-tensors 対応は
group_size=32 のみ。turbomind の OOM も同じ根（形式を解釈できず展開しようとする）。
→ **LMDeploy はこのモデルでは使えない**。vLLM / SGLang は compressed-tensors を
読めるので比較表はそちらで作る。`--model-format awq` による強制は未検証
（重みのキー配置が AWQ 形式と違うため成功する見込みは薄い）。

### 次にやること

- **LMDeploy**: turbomind が T4 で CUDA OOM（重み 43/64 で `allocator.cc:49`）→
  **pytorch バックエンドに変更**して再実行中。`cache_max_entry_count` は 0.5 に下げた。
- **SGLang**: **install の原因を特定して回避した**（`sglang-install-probe`・GPU 0 秒）。
  直接依存 26 本を cp313 wheel の有無で調べた結果、wheel が無いのは
  `outlines_core==0.1.26`（`outlines==0.1.11` が完全固定・Rust 実装）と
  `cuda-tile==1.6.0rc5`（sdist だがビルドは通る）の 2 本だけ。
  **rustup 9.4s + `pip install sglang==0.5.22` 389.8s** で通り、
  `import sglang` / `launch_server --help` も OK（torch は 2.14.1+cu130 に差し替わる）。
  → `kc_sglang_pip_steps()` を 2 ステップ（rustup → pip）に変更。GPU で試走中。
  `compressed-tensors` は sglang 側にも実装があり、`get_min_capability() = 70`、
  `group_size in (32, 128)` なので **LMDeploy と違ってこの AWQ を読める見込み**。
- **#4**（ngrok 実配線・smoke 無し）は Secrets が要る。`NGROK_TOKEN` は必須、
  `GIST_ID` + `GITHUB_TOKEN` が実質必須。ローカルの `config.env` には無い
  （Kaggle 側の Secrets に登録が必要）。
- #5（keepalive / `/stop` / UI）は**検証済み**（CPU・GPU 0 秒。上の表）。
- `vllm-wheelhouse` は patch と 2 点セットなら解決する（`notes/dataset-inventory.md`）。

消費: 62 分 / 30 時間（#1 を 2 回、#2 を 9 回、#3 を 5 回、#6 を 3 回落とした分を含む）。
**その後 v10〜v14 で 8h02m / 30h まで進んだ**（残り 21h57m・2026-11-01 13:10）。
`python3 -m cli quota` の値は**実行中セッションを予約で引く**ことがあり、
v14 の実行中に 10h47m → 21h57m へ戻った。**1 回の読みで判断しない。**

### v13（記録なし・**失敗した回**）

8 並列の先読み（2.2 MiB/s × 18.21 GiB = 約 135 分）で、70 分走っても終わらず、
`kaggle kernels` に **stop コマンドが無い**ため止められなかった回。
`vllm-io-bench` v2 の並列スケーリング測定で誤りに気付き、v14 を上書き push して修正した。
教訓は「速い回の数字から機構を推定しない」（`notes/cpu-test-results.md`）。

### v14（2026-11-01 12:48 完了・**先読みの効果を測った回 / 結果は逆効果**）

`step2-vllm-qwen38-27b-cloudflared-script` を v14 で push。`COMPLETE` まで 31.3 分。

| 項目 | v10 | v14 | 差 |
|---|---|---|---|
| install | 271.3s | **227.7s** | −43.6s ✅ |
| 先読み（prewarm） | — | **152.5s**（122.2 MiB/s / 1 本） | **+152.5s ❌** |
| 起動（`load_seconds`） | 706.5s | **486.1s** | −220.4s |
| 起動まで合計 | 977.8s | **866.3s** | −111.5s |
| 成功した試行 | 1 発目（`mtp+dd`） | **2 発目（`eager`）** | ❌ |
| 速度 | **31.0 t/s** | **4.5 t/s** | **7 倍遅い ❌** |
| VRAM | 13,343 MiB ×2 | 13,299 MiB ×2 | ほぼ同じ |

**「起動まで −111 秒」は改善ではない。** 1 発目（`mtp+dd`）が 921 秒で予算切れし、
`eager` に落ちた分だけ速く見えている。`eager` は **31.0 → 4.5 t/s** なので、
この 111 秒は割に合わない。実質は**退行**。

#### 先読みは効かない（そして 1 発目を殺す）

| 試行 | 重みロード | 内訳 |
|---|---|---|
| 1 発目（`mtp+dd`） | **64.37s**（`Loading weights took`） | ただし直前まで約 700 秒を weight iterator が消費 |
| 2 発目（`eager`） | **6.39s** | page cache に載っていた |

先読みは 18.21 GiB を 152.5 秒で読んだ（実測 122.2 MiB/s・**probe は 66.7 MiB/s と
1.8 倍の過小評価**、見込み 279.5s に対して実測 152.5s）。それでも **1 発目の
重みロードは速くならなかった**。理由は、ワーカーが `8.88 GiB × 2` の anonymous を
確保する時点で **先読みした page cache が追い出される**ため。2 発目が 6.39 秒なのは、
1 発目が読み直して入れ直した cache がそのまま残っているから。

→ **先読みを入れるなら「ワーカーが anon を確保した後」しかない**（＝無理）。
`KC_PREWARM=1` の既定は外すべき。152.5 秒を払って 1 発目の時間が増えるだけ。

#### 373 秒の無音の正体（v10 の 381 秒と同じ）

`eager` でも **373 秒**出た（`shm_broadcast.py:871` が 60 秒おきに 6 回）。
`eager` に compile は無いので、**v10 の「compile 後の Triton JIT が第一候補」は
これで確定**。位置は KV cache 確保の直前＝ memory profiling の最中で、直後に

```
[gpu_worker.py:759] Available KV cache memory: 3.45 GiB
[qwen_triton_warmup.py:296] Warming up Qwen GDN Triton kernels for model_type=qwen3_5_text
```

が出る。**Qwen3.5 の GDN（線形注意力）カーネルの JIT**。起動時間の最大項目（54%）。

#### 24 GiB の謎は「page cache」だった（`ram_trace.txt` で確定）

`ram_trace.txt` は 5 秒ごとに `avail / free / cache / shm / RSS 上位 3` を残す。
v14 の推移:

| 時刻 | avail | cache | shm | 上位 |
|---|---|---|---|---|
| 0s | 30,437 | 28,579 | 1 | python3:1001 |
| 100s | 24,524 | 22,734 | 11 | vllm:2114 / Worker:1617 |
| 300s | 25,069 | 23,320 | 29 | Worker_TP:6987 |
| 900s | 24,953 | 23,620 | 33 | Worker_TP:2337 |
| 1400s | 21,445 | 20,395 | 33 | Worker_TP:3197 |

**空きは 21〜25 GiB で安定していて、24 GiB は消えていない。** 減ったのは
`MemAvailable` ではなく「cache として数えられている分」で、これは先読みの残骸
（回収可能）。v12 の `空き RAM: 30315 MiB` → `Available RAM: 5.68 GiB` は、
**vLLM が `MemAvailable` ではなく別の値（cgroup の limit - 使用量）を見ている**
ためで、`/dev/shm` も 29 MiB しか使っていない（＝ shm 容量の問題でもない）。

## 9. 比較表（確定: vLLM と llama-server の 2 本）

**この 2 本で確定とする。** LMDeploy は形式非対応（(j)）、SGLang は
`compressed-tensors` の WNA16 が **sm80 必須**（v4 の判定）。
**どちらも「判定済みの不可」**で、性質は同じ（未測定ではない）。

| 項目 | vLLM | llama-server |
|---|---|---|
| カーネル | `step2-vllm-…-cloudflared-script` | `llama-server-ngrok-script` |
| エンジン版 | vLLM **0.32.0**（torch 2.13.0+cu130） | Ollama 同梱の `llama-server` |
| モデル / 量子化 | `qwen3.8-27b-awq`（AWQ / compressed-tensors、19.57 GB） | `orcarouter/Qwen3.8-27B-Uncensored`（GGUF Q4_K_M） |
| 並列 | TP=2 | 1 プロセスを 2 GPU に分割 |
| MTP | **on** | **on** |
| コンテキスト | **65,536** | 未記録 |
| 生成速度 | **31.7 t/s**（ウォームアップ込み） | **14.2 t/s**（effective） |
| VRAM | 13,343 MiB ×2 | 9,841 / 9,885 MiB |
| 総所要時間 | 928.7 s（install 210 s + 起動 642 s） | **345.3 s** |
| 検証 | 5 PASS + 1 SKIP | **6/6 PASS** |
| トンネル | cloudflared | ngrok |

読み方:

- **速度は約 2.2 倍で vLLM**。ただし起動に 642 秒（T4 での `torch.compile` 61 秒 +
  CUDA graph capture を含む）かかるので、**短命な検証には llama-server が有利**。
- **VRAM は llama-server が約 3.4 GB 少ない**（13.3 → 9.9 GB/枚）。T4 2 枚で余裕を
  取りたい場面では llama-server。
- 検証項目が違う（vLLM は `openai-compat` / `conversation` まで、llama-server は
  `chat-ok` / `speed` まで）ので PASS 数を単純比較しない。

**比較の限界（重要）**: 量子化が違う（AWQ 4bit vs GGUF Q4_K_M）ため、これは
「同じ土俵のエンジン比較」ではなく **「この 2 つの配信構成の比較」**。厳密なエンジン
比較をするなら同一量子化で揃える必要がある（未実施）。

参考（比較表には入れない）: Ollama は 13.3 t/s / ctx 131,072 / `23 GB 100% GPU` /
162 s。コンテキスト長では最強だが速度は llama-server と同程度。

## 10. CPU プローブ 4 本（GPU クォータ 0 秒）

`kaggle/cpu-tests/` と `kaggle/whl-probe/`。すべて `enable_gpu: false`。

| プローブ | 結果 | 分かったこと |
|---|---|---|
| `tiny-llama-probe` | **✅ ok** | 高度化フラグ **10 個すべて実在**: `--spec-type` / `draft-mtp` / `--cache-type-k` / `--cache-type-v` / `-fa` / `--jinja` / `--reasoning-budget` / `-np` / `--parallel` / `-t`（`help_len` 60,758）。base / kv_q8 / parallel2 / threads4 は起動・会話まで成立 |
| `tiny-ollama-probe` | **✅ ok** | `pull` 成功（`qwen2.5:0.5b`）。`gpu_unknown` を明示（q8_0 KV の実効 / flash attention / size_vram） |
| `vllm-whl-probe` | **✅ ok** | **wheelhouse + patch の 2 点で `pip_ok=true` / `missing=[]` / `uv_ok=true`**。cublas は 13.1.1.3 と 13.8.2.12 の両方が見えている。**単体では不可**（13.1.1.3 は patch 側にしか無い） |
| `vllm-flag-probe` | **✅ 完走（v2）** | `EngineArgs` **11/11 実在**。argv 受理 7/7（`--speculative-config` / `--enforce-eager` / `--kv-cache-dtype fp8` / `--max-model-len` / `--language-model-only` / `--attention-backend TRITON_ATTN`）。**負のケース 2 本**（`--definitely-not-a-flag` = unrecognized arguments / `q8_0` = invalid choice）が両方弾かれ `detector_works=true`。**KV 量子化に `q8_0` は存在しない**（19 選択肢に無い。相当は `fp8`）。詳細は `notes/cpu-test-results.md` |

`tiny-llama` の `spec_mtp` だけは `failed to create MTP context` で落ちる。これは
**フラグが受理されたうえで**モデル側に MTP ヘッドが無いため（tiny GGUF）。
GPU + 本番 MTP ヘッドでの可否は CPU では確定できない（`gpu_unknown` に明記済み）。

### この 4 本の契約をテストで固定した

`tests/test_cpu_probes.py`（18 本）。実行しなくても壊れたと分かる範囲を固定する:

- `enable_gpu: false`（true になった瞬間に「無料」ではなくなる）
- 台帳（`spec`）と metadata の `dataset_sources` / 出力先が一致
- `result.json` の出力先、モデルを走らせる 2 本の `gpu_unknown` 宣言
- `tiny-llama` のケース表が README の 4 観点（base / KV q8_0 + `-fa` / spec-type / 並列）を満たす
- `vllm-flags` の**対照実験が消えていない**（消えると以後の PASS が信用できなくなる）
- `vllm-flags` の **`ok` 判定に「弾かれて正解」のケースを混ぜない**
  （混ぜると完走しても `ok=false` になる。v1 で実際に起きた）

### `--speculative-config` の `method` は stub では解決できない

プローブの `spec_mtp` は `NotImplementedError: Unsupported speculative method: 'mtp'`
で落ちるが、これは**非対応の証拠ではない**。`--spec-method` の choices に `mtp` は
実在し（41 個中の 1 つ）、本番 #2 は同じ argv で起動して
`max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings`
を出している（= speculative 有効）。差は target の `config.json` の有無だけ。
**CPU プローブで MTP の可否は判定しない**（`gpu_unknown`）。


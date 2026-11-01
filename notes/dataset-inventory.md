# Kaggle データセット台帳（2026-10-30 取得）

削除要否の判断材料。取得元:
- `kaggle datasets list -m --format json`（CLI 2.2.5）
- `kaggle datasets metadata`（1.7.4.5。`isPrivate` は 2.2.5 の返却に無い）
- `kaggle kernels list -m --dataset <ref>`（自分が所有するカーネルのみ）

注意:
- `size` が 0 のものは API がサイズを返さない（private で計算されていない）。
  実際の容量は「実体」欄の推定値。
- `kaggle datasets list` の `size` は**圧縮後**の値で、実ファイルサイズとは一致しない
  （例: `my-cloudflared-bin` は API 20 MB / 実ファイル 40 MB）。表の容量は
  `kaggle datasets files` の実ファイル合計を優先し、API 値しか無いものはそのまま。
- `datasets files` が大きなディレクトリで空を返すことがあるため、
  ファイル数は参考値。
- **全件 private**（他ユーザーは利用していない）。`downloadCount` は自分の再取得を含む。

## 一覧

| # | slug | 実体 | 容量 | 更新 | repo 参照 | 依存カーネル（自分） | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | `qwen38-27b-uncensored-hauhaucs-awq-mtp` | AWQ モデル + MTP ヘッド + `.cache/huggingface/` ゴミ | 不明（0 表記・約 16 GiB） | 10-28 | 26 | **20** | 保持（vLLM 本体）。再版時に `.cache` を除外 |
| 2 | `my-ollama-orcarouter-qwen38-27b` | Ollama blobs（16.5 GiB + 0.93 GiB）+ manifests | 16.5 GiB | 10-27 | 5 | 5 | 保持（Ollama 版モデル） |
| 3 | `my-ollama-orcarouter-qwen38-27b-patch` | `bin/ollama` + `lib/ollama`（ランタイム） | 1.06 GB | 10-28 | 5 | 5 | 保持（Ollama 版ランタイム） |
| 4 | `my-cloudflared-bin` | `cloudflared` 1 ファイル（版不明） | 40 MB（実ファイル 40,247,428 B） | 10-27 | 6 | 5 | 保持（後日 `infra-…-<ver>` へ移行） |
| 5 | `vllm-wheelhouse` | wheel 群 + MANIFEST.txt | 3.0 GB | 10-29 | 4 | 2 | **patch と併せれば解決する**（2026-11-01 実測）。単体では不可。下記参照 |
| 6 | `qwen38-27b-uncensored-exl3-4bpw` | EXL3 3 シャード + LICENSE | 15.9 GiB | 10-29 | 2 | 0（push 前） | 保持（EXL3 検証待ち） |
| 7 | `qwen3-27b-uncensored-gguf` | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 15.7 GiB | 10-27 | 0 | 0 | **要判断**（EXL3 失敗時のフォールバック） |
| 8 | `my-ollama-qwen3-6-35b` | Ollama（35B 系・旧） | 22.3 GiB | 07-27 | 0 | 0 | **削除済み（2026-10-30）** |
| 9 | `my-ollama-qwen2-5-coder-14b` | Ollama（14B コーダー・旧） | 8.4 GiB | 07-28 | 0 | 0 | **削除済み（2026-10-30）** |
| 10 | `my-ollama-gemma3-1b` | Ollama（Gemma 3 1B・旧） | 0.8 GiB | 07-27 | 0 | 0 | **削除済み（2026-10-30）** |
| 11 | `kawamura-finetune` | embedder/reranker 学習データ | 20 MB（実ファイル 22,030,906 B） | 08-28 | 0 | 2 | 保持（#12 と統合検討） |
| 12 | `kawamura-ruri-ft-data` | embedder 学習データ + スクリプト | 4 MB（実ファイル 4,195,551 B） | 08-28 | 0 | 1 | 保持（#11 と統合検討） |
| 13 | `vllm-warm-cache` | **未作成**（Triton キャッシュの tar を入れる予定） | 0（完全版なら推定 100〜200 MB） | — | 2 | 1（`vllm-playground`） | **作成待ち**。下記「#13」参照 |

削除前の合計は概算 **約 100 GiB**。#8〜#10（約 31.5 GiB）は
**2026-10-30 に削除実行済み**で、残りは **約 68.5 GiB**（9 件）。
#7（15.7 GiB）は 10/31 の GPU 検証後に判断。

## 判定の観点

1. **repo からの参照**（`dataset_sources`）: 0 でも、Kaggle 上のカーネルが依存している
   ことがある（#11/#12）。削除するとそのカーネルは再実行不能。
1b. **script 版は notebook 版と同じ Dataset を参照する**（2026-10-30 追加）。
   `…-script` / `…/script` は notebook 版とほぼ同じ `dataset_sources` を持つので、
   Dataset を消す・版を入れ替えるときの影響範囲は **notebook + script の両方**になる。
   削除可否は「repo の台帳」ではなく **Kaggle 上の依存カーネル数**で判断する
   （`kernels list -m --dataset`）。このため #7（GGUF）の削除判断は
   script 版の GPU 実測後まで待つ。
2. **GPU 実機の結果待ち**: #7 は EXL3 が実機で動けば不要、動かなければ必要。
   判定は 2026-10-31 09:00 JST 以降の検証後。
3. **版の掃除**: 容量は「全版の合計」ではない可能性がある。古い版の削除は
   `datasets version -d`（アップロード時）でのみ可能。
4. **サイズ表記 0**: private では API がサイズを返さないだけで、容量は存在する。

## 統合（まとめる）候補

決定事項: **#8〜#10 は削除（2026-10-30 実行済み）、それ以外は保持**。以下は「今すぐは統合しないが、
次に版を作るとき（再アップロード・新規作成）にまとめる」方針。
Kaggle の slug は `dataset_sources` が slug 参照＝最新版マウントなので、
**既存 slug の統合・改名はしない**（依存カーネル 20・5・6 本が壊れる）。
統合は「1 slug 内に複数ファイルを同居させる」形で、新規版/新規 slug のみ。

| 対象 | まとめ方 | 条件・時期 |
|---|---|---|
| #2 `my-ollama-orcarouter-qwen38-27b` + #3 `...-patch` | モデル blobs と `bin/ollama` ランタイムを 1 slug（`ollama/` + `runtime/`）に同居 | モデル再版時。参照 5+5 本の `dataset_sources`/マウント先を更新できる場合のみ。rename はしない |
| #1 `...awq-mtp` | `.cache/` `*.lock` を除外して再版（ファイル数・容量の純化） | vLLM 本体を触る時。`datasets version -d` で旧版を落とす |
| #4 `my-cloudflared-bin` | `infra-cloudflared-linux-amd64-<ver>` 新設 + `VERSION.txt` 同居 | 現状の名前は維持。新版移行時に新 slug へ（バイナリは移行後に旧 slug 削除可） |
| #11 `kawamura-finetune` + #12 `kawamura-ruri-ft-data` | 1 slug `kawamura-embedding-data`（`data/` `scripts/` 同居） | 依存カーネル 2+1 本を更新できる場合のみ。合計 20 MB なので急がない |
| #5 `vllm-wheelhouse` | **`vllm-wheelhouse-patch` と必ず 2 つセットで使う**（2026-11-01 実測）。単体では `cuda-toolkit` が要求する `nvidia-cublas==13.1.1.3` が無く `--no-index` で解決できない。patch（11 wheel / 1.19 GB）を足すと `pip_ok=true` / `missing=[]` / `uv_ok=true` になる。**ただし既定は PyPI**（210 秒で入るうえ、4.4 GB をマウントする必要がない） | 高速化したいときだけ patch を併用 |
| #6 EXL3 / #7 GGUF | 統合しない（形式が別、フォールバック用）。#7 は EXL3 実機 OK なら削除 | 10/31 の検証後 |

横断ルール（refactor-plan の「データセット整理案」と同一）:
- 命名は新規のみ `<model>-<quant>` / `infra-<name>-<version>`。
- アップロードは `kbuild upload-dataset` に一本化し `.cache/` `.git/` `*.lock` を既定除外。
- 各カーネルの `stage_env` でマウント版を `result.json` に記録。

## 削除の手順（2026-10-30 実行済み）

CLI 2.x（Python 3.10+）が必要。現行 PATH は 1.7.4.5 で `delete` が無い。

2026-10-30 時点: `~/.venv-kaggle2` を作成（Kaggle CLI 2.2.5、
`datasets delete` の存在を確認）。削除前チェック（存在・依存 0）の後、
**#8〜#10 の 3 件を削除実行し、一覧が 12 → 9 件になったことを確認済み**。
以下は再現用の手順として残す。

```bash
# 準備（PATH は変更しない。作成済みなら再実行不要）
/opt/homebrew/bin/python3.12 -m venv ~/.venv-kaggle2
~/.venv-kaggle2/bin/pip install -U kaggle
~/.venv-kaggle2/bin/kaggle --version   # 2.2.5

# 削除前チェック
~/.venv-kaggle2/bin/kaggle kernels list -m --dataset yamahei21python/<slug> --format json
~/.venv-kaggle2/bin/kaggle datasets files yamahei21python/<slug>

# 実行（不可逆・全版削除）
# ~/.venv-kaggle2/bin/kaggle datasets delete yamahei21python/my-ollama-qwen3-6-35b -y
# ~/.venv-kaggle2/bin/kaggle datasets delete yamahei21python/my-ollama-qwen2-5-coder-14b -y
# ~/.venv-kaggle2/bin/kaggle datasets delete yamahei21python/my-ollama-gemma3-1b -y
```

削除前に、ローカルの `datasets/` に退避があるかを確認する
（`datasets/orcarouter` 17 GB、`orcarouter-patch` 1.4 GB、`cloudflared` 38 MB、
`gguf` は `models/` の GGUF への symlink のみ。
`orcarouter/blobs/sha256-3445102e…` は `models/` の GGUF と**ハードリンク**なので
実占有は 17 GB ではなく 0.9 GB）。
旧 Ollama 系のローカル実体は無い。

## cloudflared Dataset（#4）の更新手順

1. 採用する版を決め、`datasets/cloudflared/bin/cloudflared` を差し替える。
2. `runtime/net.py` の `KC_CLOUDFLARED_VERSION` を同じ版にする（ダウンロード
   フォールバックと Dataset の版を揃える）。
3. `dataset-metadata.json` の `description` に版と取得 URL を書く。
   現在の記述は `releases/latest` を指しているので、版固定 URL に直す。
4. `kaggle datasets version -p datasets/cloudflared -m "<版>"` を実行する
   （**アップロードはユーザーのみ**）。
5. 差し替え後に `bin/cloudflared --version` と sha256 を控える
   （現行は 2026.10.4 / 40,247,428 B / sha256 `ae5e8add…d9e42`）。

ライセンス: cloudflared は **Apache-2.0**。再配布物なので、Dataset の
description か `NOTICE` に帰属とライセンスを明記する。現行 metadata は
`"licenses": [{"name": "other"}]` のまま（変更はアップロード時にまとめて行う）。

## cloudflared の参照箇所

`runtime/net.py` の `kc_cloudflared_url()` が唯一の URL 生成元。ビルダーは
文字列を直書きせず、この関数を埋め込んだセルから呼ぶ。


## ローカルのモデル実体（2026-10-30 調査 → **同日削除済み**）

> **2026-10-30 15:20 実行: 下記の表のモデル重みはすべて削除した。**
> 残したのは再アップロード用の足場（`dataset-metadata.json` ×2、
> `datasets/orcarouter-patch/`、`datasets/cloudflared/`）と uzu の
> `~/.cache/uzu-stage`（5.6 GB）だけ。
> 削除した内訳: `ollama rm` 3 モデル（30.0 GB）/ `models/` の GGUF 16.8 GB /
> `datasets/orcarouter/{blobs,manifests,bin}` 17.7 GB /
> `~/Myproject/models/` 4 本 22.3 GB = **約 87 GB**。
> 復旧は `ollama pull orcarouter/Qwen3.8-27B-Uncensored:latest`（2.6 分）など。
> **注意: 削除後の空き容量は 36 GiB → 62 GiB しか増えていない**（下記「容量が
> 解放されない件」）。

**Kaggle で動かす分には、ローカルにモデルを置いておく必要は無い。**
カーネルは `dataset_sources` で Kaggle Dataset をマウントするだけで、
ローカルのファイルは一切見ていない。ローカル実体は「Dataset を再アップロード
するときの staging」か「ローカル推論」のどちらかにしか使わない。

（以下は削除前の記録）

| 場所 | 中身 | サイズ | ローカルの用途 | Kaggle |
|---|---|---|---|---|
| `models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 27B Q4_K_M | 16.81 GB | `datasets/gguf/` の symlink のみ | 不要 |
| `datasets/orcarouter/blobs/sha256-3445…` | 同じ 27B（↑と**ハードリンク**） | 0（共有） | `datasets version` の staging | 済（17.3 GB） |
| `datasets/orcarouter/blobs/sha256-add20…` | vision projector | 0.93 GB | 同上 | 済 |
| `datasets/orcarouter-patch/` | ollama バイナリ + lib | 1.4 GB | patch Dataset の staging | 済 |
| `datasets/cloudflared/` | cloudflared | 0.04 GB | cloudflared Dataset の staging | 済 |
| `~/.ollama/…sha256-3445…` | 同じ 27B（**別コピー**） | 16.81 GB | `LLAMA_MODEL=27b` のときだけ | 不要 |
| `~/.ollama/…sha256-add20…` | projector | 0.93 GB | 同上 | 不要 |
| `~/.ollama/…sha256-afb54a…` | huihui 9B | 6.59 GB | `run-llama-server.command`（**既定 9b**） | 不要 |
| `~/.ollama/…sha256-03b747…` | qwen35-9b-local-q4km | 5.68 GB | `ollama run` で手元用 | 不要 |
| `~/.cache/uzu-stage` | uzu 9B-M | 6.0 GB | `run-uzu-server.command` | 不要 |

`~/Myproject/models/`（HauhauCS 5.2 / Qwen3.5-9B-GGUF 5.3 / OptiQ 7.7 /
uncensored_2b 4.1 = 22.3 GB）は**別プロジェクトのもの**。この repo とは無関係。

### 27B を消してよい根拠

- ローカルでは **0.10 tokens/s**（RAM 16 GiB に載らない。`run-llama-server.command`
  自身が警告を出す）
- Kaggle 側は `my-ollama-orcarouter-qwen38-27b`（17.3 GB）から読む。**アップロード済み**
- `qwen3-27b-uncensored-gguf`（Kaggle 側 16.5 GB）は **`ds=` の参照が 0**。
  `step1` が作るのは Ollama 形式の方なので、この GGUF は EXL3 失敗時の
  フォールバックとして置いたまま忘れられている

### 消すときのコマンド（未実行・判断待ち）

```bash
# 1) Ollama 側（マニフェストと blob をまとめて消す。rm で消さないこと）
ollama rm orcarouter/Qwen3.8-27B-Uncensored:latest      # 17.7 GB

# 2) repo 側。ハードリンクは「両方」消さないと容量は減らない
rm models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
rm -rf datasets/orcarouter/{blobs,manifests,bin}        # 16.8 GB + projector
rm datasets/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf     # リンクだけ。json は残す

# 3) 元に戻すとき
ollama pull orcarouter/Qwen3.8-27B-Uncensored:latest    # 2.6 分
```

`datasets/orcarouter-patch/` だけは**戻すのが重い**（作るのに capture カーネル =
GPU が要り、`build_capture_patch` は退役扱い）。1.4 GB なので残すのが無難。

### ~~AWQ の Dataset は壊れている~~ → **訂正: 壊れていなかった（2026-10-31）**

> **2026-10-30 の判定は誤りだった。原因は `kaggle datasets files` のページ送り。**
> このコマンドの既定は **20 件/ページ**で、`.cache/huggingface/...` が名前順で先頭に
> 並ぶため、1 ページ目だけ見ると `.lock`（0 B）と `.metadata`（100〜191 B）しか
> 見えない。重みは**2 ページ目**にあった。
>
> `--page-size 200` で取り直した実際の中身:
>
> | ファイル | サイズ | 作成日 |
> |---|---|---|
> | `model.safetensors` | **18,698,467,264 B（18.70 GB）** | 2026-10-28 16:16 |
> | `model-mtp.safetensors` | 849,400,424 B（0.85 GB） | 2026-10-28 16:11 |
> | `config.json` / `tokenizer.json` / `chat_template*.jinja` ほか | 計 20 MB | 同上 |
>
> `GET /api/v1/datasets/view/...` の `totalBytes` も **19,567,955,256 B**（= 19.57 GB）で
> 一致する。**vLLM 系カーネルはこのまま動く。**
>
> **教訓**: `kaggle datasets files` は必ず `--page-size 200` を付ける。
> `list -m` の `size 0` は表示だけの問題（EXL3 も 0 と出るが 16.3 GB ある）。

以下は当時の（誤った）記録として残す。

| Dataset | 1 ページ目に見えたもの | 判定（誤） |
|---|---|---|
| `qwen38-27b-uncensored-exl3-4bpw` | `model-0000{1,2,3}-of-00003.safetensors`（8.47 + 8.55 + 0.06 GB）+ config 類 | 正常 |
| `qwen38-27b-uncensored-hauhaucs-awq-mtp` | `.cache/huggingface/` の `.lock`（0 B）と `.metadata` だけ | **重みが無い**（← 誤り） |

### ついでに直したもの: `prepare-vllm-dataset` の出力先（2026-10-31）

上の誤判定を信じてカーネルを回した際、`/kaggle/working` の上限（19.5 GB）を
気にして `/kaggle/tmp` へ落とすよう変更した。モデルは 19.57 GB なので
`/kaggle/working` では**実際に溢れる**（将来また作り直すときのため、この修正は残す）。

- `OUT_DIR` を `/kaggle/tmp/<slug>` に変更
- ダウンロード後に「重みが 10 GB 未満なら後続を止める」assert を追加
- 非推奨の `local_dir_use_symlinks` / `resume_download` を削除（新しい
  `huggingface_hub` では TypeError になる版がある）
- 実測: DL 1.8 分 / 重み 2 本 19.55 GB を確認。ただし既存 Dataset があったため
  `dataset_create_new` は新しいバージョンを作らず、**Dataset は変化なし**
  （`currentVersionNumber` は 1 のまま）

### 容量が解放されない件（2026-10-30・要確認）

87 GB を削除したのに、`df` の空きは **36 GiB → 62 GiB（+26 GiB）** しか増えていない。
`du` では確実に減っている（`~/Myproject/models` 0 B、`models/` 0 B、
`datasets` 1.4 GB、`~/.ollama` 3.2 MB）のに、APFS の
`Capacity Consumed` が 388.9 GB のまま動かない。

1 GB のファイルを作って消す検証で再現:

| 手順 | 空き |
|---|---|
| 作成前 | 62.9192 GiB |
| `dd` で 1 GB 作成 | 61.9192 GiB（−1 GiB） |
| `rm` で削除 | 61.9191 GiB（**ほぼ戻らない**） |

つまり**この作業環境のシェルから出した `rm` は、名前空間からは消えるが
ブロックを解放しない**。一方 `ollama rm`（Ollama デーモン経由）だけは
28 GiB を実際に解放した。

- 心当たり: 実行環境のサンドボックスが削除を保留している可能性。セッション終了か
  再起動で解放される見込み（APFS はマウント時に未参照ブロックを回収する）。
- **確認方法**: ユーザー自身のターミナルで `df -h /` と
  `ls ~/Myproject/models` を実行する。ファイルが復活していたらサンドボックスが
  削除を巻き戻しているので、その場合は `rm` を直接やり直す。

## #13 `vllm-warm-cache`（**未作成**・2026-11-01 判明）

vLLM の起動に必要な Triton カーネル（GDN の `@triton.autotune`）を
`~/.triton/cache` ごと tar にして Dataset 化する計画。狙いは起動時の
**373 秒の無音**（`determine_available_memory` の profile run）を消すこと。

### 現状: Kaggle 側に存在しない

- `datasets/warm-cache/dataset-metadata.json` はローカルにあり、コミット済み。
- しかし `kaggle datasets list --user yamahei21python` に出てこない。
- `--no-run`（Quick Save）で `vllm-playground` を保存したときの警告:

  ```
  The following are not valid dataset sources and could not be added to the kernel:
  ['yamahei21python/vllm-warm-cache']
  ```

### 危ない理由

`KC_WARM_CACHE=1`（既定）は「あれば復元、無ければ『なし』と出して続行」なので、
Dataset が無いまま `cache` 実験を回すと **baseline とまったく同じ数字**が出る。
これを「キャッシュは効かない」と読むと偽陰性で、1 往復（約 40 分）を捨てる。
→ 実験は `KC_WARM_CACHE=require`（無ければ即 `RuntimeError`）で回す。

### 作るときの手順

v15 が残した tar（23.9 MB）は**不完全**なので使わない。重みロード中に落ちた回の
もので、`chunk_o` / `chunk_delta_h` が入っていない。起動に成功した回から採り直す。

1. `KC_EXPERIMENT=baseline` を実行し、`warm-cache.tar` を成果物から取得
2. 中身を検品: `chunk_o` / `chunk_delta_h` があり、エントリが 100〜200 あること
   （30 のままなら profile run に届いていない＝また不完全）。
   **`require` で回す限りこの検品は自動で効く**（復元後のエントリ数が 50 未満なら
   `RuntimeError`。2026-11-01 追加）が、Dataset を作る前に手で見ておくほうが安い
3. `datasets/warm-cache/` に tar を置き `kaggle datasets create -p datasets/warm-cache`
4. `--no-run` で playground を保存し直し、`not valid dataset sources` が消えたことを確認
5. `KC_EXPERIMENT=cache` で実行

キーが一致しないと**黙ってミス**する（Triton のキーは Triton の版・対象 sm・
カーネルソースのハッシュを含む）。vLLM か Triton を上げたら作り直す。

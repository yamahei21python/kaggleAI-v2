# GPU スモーク計画（2026-10-31 以降）

> ⚠ **この文書は 2026-11-01 に一部が古くなった。** ngrok を廃止したので、
> 「#4 ngrok の実配線」は**やらない**（ビーコン + cloudflared が正しい組み合わせ。
> `notes/url-publishing.md`）。ngrok 版のカーネルを指している行も、
> いまは retired で push できない。読むときはその前提で。
> 現在の手順は `README.md` の「GPU が要る」と `notes/operations.md`。

> **カーネルは作成済み（2026-10-30）**。`--no-run`（Quick Save 相当）で version 1 を
> 保存したので、GPU は消費していない。当日の手順・Secrets・判定は
> `notes/gpu-run-sheet.md` にまとめてある。ここは「なぜその順か」の設計メモ。

GPU クォータが戻ったら、**全エンジンで「起動 → 会話 1 往復 → 終了」だけ**を回す。
keepalive / tunnel / Gist / proxy / UI は CPU 側で検証済みなので**省略**し、
GPU 時間を推論スタックの確認だけに使う。

## 前提（CPU 側で確認済み = GPU でやらない）

| 項目 | 検証場所 |
|---|---|
| proxy（Basic 認証 / `/healthz` / `/stop` / `/quota`） | `tests/test_step2_proxy.py` |
| keepalive の延長・締切・`/stop` | `tests/test_keepalive.py`（フェイククロック） |
| `/v1` の形（SSE / 400 / 404） | `tests/test_exl3_server.py`（MockEngine） |
| 生成物の再現性・AST・Secret 非混入 | `tests/test_exl3_build.py` / `tests/test_engines.py` |
| notebook とビルダーの一致（13 冊） | `tests/test_notebooks.py` |
| script とビルダーの一致（18 カーネル / 46 ファイル） | `python3 kaggle/check_scripts.py` |
| 起動 argv（TP / backend / 量子化） | `tests/test_engines.py` |

→ GPU では **install が通るか / 重みが VRAM に載るか / 会話が返るか** だけ見る。

## smoke モード

`--smoke` を付けると生成物が「install → launch → `/v1/chat/completions` → 停止 → exit」に
変わる（`SMOKE = True`）。省略されるのは **cloudflared / Gist 公開 /
keepalive ループ**の 3 つ。**auth_proxy は smoke でも起動する**（vLLM は proxy 経由で
会話を確認するため。UI 配信も同じ経路）。LMDeploy / SGLang / EXL3 の smoke は
推論サーバへ**直接**投げる（LMDeploy / SGLang は 8000、EXL3 の chat_server は 9931）。

smoke の対象は次の **9 カーネル**（`cli/build.py` の `SMOKE_KERNELS`）:

| 系統 | カーネル |
|---|---|
| notebook | `exl3-serve` / `step2-lmdeploy-qwen38-27b-cloudflared` / `step2-sglang-qwen38-27b-cloudflared` / `step2-vllm-qwen38-27b-cloudflared` |
| script | `step2-orcarouter-qwen38-27b-cloudflared-script` / `step2-vllm-qwen38-27b-cloudflared-script`（ngrok 系と llama-server は退役） |

それ以外は `--smoke` を拒否する。旧 script 3 系統（exl3 / lmdeploy / sglang）は
`smoke_out` を持たず `--smoke` で同じディレクトリへ書くため対象外（`cli push --smoke`
が直前に再生成するので実害は出ない。README の「生成物のポリシー」参照）。

smoke 版は通常版と**別ディレクトリ**へ書く（vLLM は
`kaggle/step2-vllm-kernel/smoke-<tunnel>/`、15 セル）。`cli push --smoke` は
そちらを staging するので、通常版（24 セル）を上書きしない。smoke の最終セルは
残存プロセスと GPU 解放（`kc_wait_gpu_idle`）まで確認し、残っていれば
**カーネルを失敗させる**（表示だけで完走扱いにしない）。

```bash
python3 -m cli quota                     # 残量を確認
python3 -m cli push <name> --smoke --yes # 直前に --smoke で再生成して push
kaggle kernels output <id> -p out.<name>/   # result.json を取得
```

`python3 -m cli push <name> --yes`（smoke 無し）に戻すと通常の配信カーネルを再生成する。
**smoke のまま放置しない**（`cli push` が毎回再生成するので事故らない）。

### result.json の見方

| キー | 意味 | 合格 |
|---|---|---|
| `stage` | どこまで進んだか | `done` |
| `ok` | 完走 | `true` |
| `gpu.names` / `gpu.memory_mib` | T4 x2 が見えているか | 2 枚 / 15 GiB 前後 |
| `install_seconds` | pip 時間 | 記録のみ（比較用） |
| `load_seconds` | 重みロード | 記録のみ |
| `verify_direct.text` | 会話の返答 | 空でない（`has_text: true`） |
| `verify_direct.wall_seconds` | 16 token の実測 | 記録のみ |
| `stop_remaining` | 停止後の残存プロセス | `[]` |
| `engine_log_tail` | 起動失敗時のログ末尾 | 失敗時のみ |

失敗時は `error` / `traceback` / `engine_log_tail` を見る。`stage` で切り分け:
`install` で落ちたら依存、`load` で落ちたら VRAM / backend、`verify` で落ちたら API 形状。

vLLM 版は `result.json` を作らず **`smoke.txt`**（`leftover` / `gpu_idle_after`）を残す。
会話はウォームアップセルが見るので、`leftover=none` かつ `gpu_idle_after>=0` が合格。
どちらかが駄目ならそのセルで `fail()` する（ノートブック自体が失敗になる）。

## 実行順 — **3 種 × 1 回・全種類**

骨格版 script（`kbuild/serve_script.py`）を **3 種 × 1 回**だけ回す。1 種につき
1 回なので、**全種類を試す**（1 本通ったから他を省く、はしない）。1 本 8〜25 分。

`kbuild/serve_script.py` の骨格が Kaggle で動くかは**まだ 1 回も見ていない**。
notebook 版とセル列は同じだが、梱包（`main.py` 1 ファイル + base64 埋め込み）と
`stage_install()` / `launch()` の関数化が挟まるので、まずここを通す。

| # | 種 | カーネル | エンジン | 合格条件 |
|---|---|---|---|---|
| 1 | Ollama | `step2-orcarouter-qwen38-27b-cloudflared-script` | Ollama GGUF | `stage=done` / `verify_direct.checks` が全 true / `stop_remaining=[]` |
| 2 | vLLM | `step2-vllm-qwen38-27b-cloudflared-script` | vLLM AWQ + MTP | 同上（起動ラダーが script でも回るか） |
| 3 | ~~llama-server~~ | ~~`llama-server-ngrok-script`~~ | 退役（GPU カーネル無し） | — |

3 本とも `--smoke` が使える（`smoke-<tunnel>/` へ別書きし、通常版を壊さない）。
**Secrets は要らない**（2026-11-01 に ngrok を廃止し、URL はビーコンが運ぶ）。

```bash
python3 -m cli quota
python3 -m cli push step2-orcarouter-qwen38-27b-cloudflared-script --smoke --yes
kaggle kernels output yamahei21python/step2-orcarouter-qwen38-27b-cloudflared-script -p out.script/
```

`--smoke` は proxy / トンネル / Gist / keepalive を**省略**するので、#3 の
実装そのものはここでは試されない（`result.json` の `tunnel` も出ない）。
**この 3 回で確定するのは「骨格が Kaggle で起動して会話できる」ところまで**。

### 3 回のあとに 1 回ずつ（別枠）

| # | 内容 | 回数 | 合格条件 | 失敗時の次手 |
|---|---|---|---|---|
| 4 | ~~ngrok の実配線~~ | — | **廃止**（2026-11-01）。ビーコン + cloudflared に一本化したので、この枠は消えた | — |
| 5 | keepalive / `/stop` / UI | 1 | 3 種のどれか 1 本が通ってから、配信カーネルをそのまま使う（下の「共通テスト」） | `keepalive.conf` の `RUN` / `DEADLINE` |
| 6 | LMDeploy | 1 | `step2-lmdeploy-qwen38-27b-cloudflared` | `--backend pytorch` へ、`session_len` を 4096 に |
| 7 | SGLang | 1 | `step2-sglang-qwen38-27b-cloudflared` | ~~`mem_fraction_static` を 0.6 に~~ → **不要。T4 では不可が確定**（WNA16 が sm80 必須。`notes/gpu-run-sheet.md` v4） |

#6 / #7 は比較用（`semi`）。#1〜#3 が通ってから着手する。
1 本失敗したら**その日は次へ進まない**（クォータは 30h/週で、pip の再試行で溶ける）。

### notebook 版は回さない

同じセル列・同じ骨格なので、3 種が通れば notebook 版（`step2-…-cloudflared` /
`step2-vllm-…-cloudflared` / `llama-server-ngrok`）は回さない。差分は梱包
（`.ipynb` か `main.py` か）だけで、そこは CPU 側の一致検査が見ている。

## 見積り

| 用途 | 回数 | 1 回 | 合計 |
|---|---|---|---|
| **3 種 × 1 回（#1〜#3）** | **3** | 8〜25 分 | **約 1 時間** |
| 別枠（#4〜#7） | 4 | 10〜60 分 | 約 1.5〜2 時間 |
| 失敗時の再試行 | 2〜3 | — | 約 1 時間 |

合計 3.5〜5 時間。週 30h に対して余裕あり。**pip の失敗で無駄にしない**ため、
`install_seconds` と `engine_log_tail` を必ず確認してから次を回す。

## 実稼働が確認できたら（共通テスト）

smoke で疎通した**1 エンジンだけ**で、省略した経路を通す（2 回目の GPU は不要、
配信カーネルをそのまま使う）。

1. `python3 -m cli push <name> --yes`（通常変種）
2. `kaggle kernels output` で `result.json` / `url.txt` を取得
3. `python3 -m cli ask --url <url> "こんにちは"`（proxy 経由・認証）
4. UI をブラウザで開く（`llama-chat.html` の配信）
5. keepalive: conf を書き換えて延長 → `/stop`（`X-Kc-Stop`）で停止
6. 停止後に `nvidia-smi` が空くこと（カーネルは終了するので `result.json` で確認）

対象は **exl3-serve**（keepalive / `/stop` の実装が最も厚い）を推奨。

## 見積り

| 用途 | 回数 | 1 回 | 合計 |
|---|---|---|---|
| A. script 骨格の初確認（#7〜#9） | 3 | 8〜25 分 | 約 1 時間 |
| B. 既存エンジン（#1〜#6、#5/#6 は A が通れば省略可） | 4〜6 | 8〜35 分 | 約 1〜2.5 時間 |
| 共通テスト（exl3 通常配信） | 1 | 60 分（予算） | 約 1 時間 |
| 失敗時の再試行 | 2〜3 | — | 約 1 時間 |

合計 3.5〜5.5 時間。週 30h に対して余裕あり。**pip の失敗で無駄にしない**ため、
`install_seconds` と `engine_log_tail` を必ず確認してから次を回す。

## 記録

実測は `notes/engine-rollout.md` の表に追記する（エンジン / version / tp /
`load_seconds` / `wall_seconds` / メモ）。比較表は **MTP あり vLLM vs MTP なし 他**で
読むこと（MTP は vLLM 固有）。

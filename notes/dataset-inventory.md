# Kaggle データセット台帳（2026-10-30 取得）

削除要否の判断材料。取得元:
- `kaggle datasets list -m --format json`（CLI 2.2.5）
- `kaggle datasets metadata`（1.7.4.5。`isPrivate` は 2.2.5 の返却に無い）
- `kaggle kernels list -m --dataset <ref>`（自分が所有するカーネルのみ）

注意:
- `size` が 0 のものは API がサイズを返さない（private で計算されていない）。
  実際の容量は「実体」欄の推定値。
- `datasets files` が大きなディレクトリで空を返すことがあるため、
  ファイル数は参考値。
- **全件 private**（他ユーザーは利用していない）。`downloadCount` は自分の再取得を含む。

## 一覧

| # | slug | 実体 | 容量 | 更新 | repo 参照 | 依存カーネル（自分） | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | `qwen38-27b-uncensored-hauhaucs-awq-mtp` | AWQ モデル + MTP ヘッド + `.cache/huggingface/` ゴミ | 不明（0 表記・約 16 GiB） | 10-28 | 26 | **20** | 保持（vLLM 本体）。再版時に `.cache` を除外 |
| 2 | `my-ollama-orcarouter-qwen38-27b` | Ollama blobs（16.5 GiB + 0.93 GiB）+ manifests | 16.5 GiB | 10-27 | 5 | 5 | 保持（Ollama 版モデル） |
| 3 | `my-ollama-orcarouter-qwen38-27b-patch` | `bin/ollama` + `lib/ollama`（ランタイム） | 1.06 GB | 10-28 | 5 | 5 | 保持（Ollama 版ランタイム） |
| 4 | `my-cloudflared-bin` | `cloudflared` 1 ファイル（版不明） | 40 MB | 10-27 | 6 | 5 | 保持（後日 `infra-…-<ver>` へ移行） |
| 5 | `vllm-wheelhouse` | wheel 群 + MANIFEST.txt | 3.0 GB | 10-29 | 4 | 2 | 保持 |
| 6 | `qwen38-27b-uncensored-exl3-4bpw` | EXL3 3 シャード + LICENSE | 15.9 GiB | 10-29 | 2 | 0（push 前） | 保持（EXL3 検証待ち） |
| 7 | `qwen3-27b-uncensored-gguf` | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 15.7 GiB | 10-27 | 0 | 0 | **要判断**（EXL3 失敗時のフォールバック） |
| 8 | `my-ollama-qwen3-6-35b` | Ollama（35B 系・旧） | 22.3 GiB | 07-27 | 0 | 0 | **削除（確定 2026-10-30）** |
| 9 | `my-ollama-qwen2-5-coder-14b` | Ollama（14B コーダー・旧） | 8.4 GiB | 07-28 | 0 | 0 | **削除（確定 2026-10-30）** |
| 10 | `my-ollama-gemma3-1b` | Ollama（Gemma 3 1B・旧） | 0.8 GiB | 07-27 | 0 | 0 | **削除（確定 2026-10-30）** |
| 11 | `kawamura-finetune` | embedder/reranker 学習データ | 20 MB | 08-28 | 0 | 2 | 保持（#12 と統合検討） |
| 12 | `kawamura-ruri-ft-data` | embedder 学習データ + スクリプト | 0.16 MB | 08-28 | 0 | 1 | 保持（#11 と統合検討） |

合計は概算 **約 100 GiB**。#1〜#6・#11・#12 は保持、#8〜#10 を削除すると
約 31.5 GiB 減で **約 68.5 GiB** になる。#7（15.7 GiB）は 10/31 の GPU 検証後に判断。

## 判定の観点

1. **repo からの参照**（`dataset_sources`）: 0 でも、Kaggle 上のカーネルが依存している
   ことがある（#11/#12）。削除するとそのカーネルは再実行不能。
2. **GPU 実機の結果待ち**: #7 は EXL3 が実機で動けば不要、動かなければ必要。
   判定は 2026-10-31 09:00 JST 以降の検証後。
3. **版の掃除**: 容量は「全版の合計」ではない可能性がある。古い版の削除は
   `datasets version -d`（アップロード時）でのみ可能。
4. **サイズ表記 0**: private では API がサイズを返さないだけで、容量は存在する。

## 統合（まとめる）候補

決定事項: **#8〜#10 は削除、それ以外は保持**。以下は「今すぐは統合しないが、
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
| #5 `vllm-wheelhouse` | 版ごとに入れ替え（同居させない）。`MANIFEST.txt` に版と wheel 一覧を既に記録 | 現状で可。複数版併存が必要になったら `infra-wheelhouse-vllm-<ver>` |
| #6 EXL3 / #7 GGUF | 統合しない（形式が別、フォールバック用）。#7 は EXL3 実機 OK なら削除 | 10/31 の検証後 |

横断ルール（refactor-plan の「データセット整理案」と同一）:
- 命名は新規のみ `<model>-<quant>` / `infra-<name>-<version>`。
- アップロードは `kbuild upload-dataset` に一本化し `.cache/` `.git/` `*.lock` を既定除外。
- 各カーネルの `stage_env` でマウント版を `result.json` に記録。

## 削除の手順（実行はまだしない）

CLI 2.x（Python 3.10+）が必要。現行 PATH は 1.7.4.5 で `delete` が無い。

```bash
# 準備（PATH は変更しない）
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

削除前に、ローカルの `kaggle-dataset-*` に退避があるかを確認する
（`kaggle-dataset-orcarouter` 17 GB、`-patch` 1.4 GB、`-cloudflared` 38 MB。
旧 Ollama 系のローカル実体は無い）。

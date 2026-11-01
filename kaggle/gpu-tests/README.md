# GPU 高度化テスト（1 エンジン = 1 カーネル）

GPU クォータを無駄にしないための測定専用カーネル。**push しない限り 0 秒**。
配信カーネル（`step2-vllm` / `llama-server` / `step2`）とは別 id なので、
測定で壊しても配信に影響しない。

| カーネル | 測るもの | KC_EXP |
|---|---|---|
| `gpu-vllm-start` | 起動時間、MTP draftdir、Prefix Cache、用途別 ctx/並列プロファイル | `baseline` / `mtp_draft` / `prefixcache` / `profile_fast` 等 |
| `gpu-llama-mtp` | MTP token sweep、KV q4/q8、batch、tensor split、並列、prefix | `none` / `mtp_tok1` / `batch1024` / `split4852` 等 |
| `gpu-ollama-ctx` | ctx 長と KV 量子化の VRAM / 速度、CPU 落ち検出 | `32k_q8` / `128k_q8` / `128k_f16` |

既定は `KC_EXP=all`（全ケースを 1 セッションで順に）。カンマ区切りで絞れる。

## なぜ「エンジン単位」で 3 本か

GPU は**セッション時間**課金。分割すると毎回固定費（起動 1〜2 分 + install）が乗る。

| 分割 | 固定費 | 備考 |
|---|---|---|
| 実験ごとに 1 本 | 実測 1 本 +10 分超 | install が毎回走る |
| **エンジンごとに 1 本** | 実測 1 本 +2〜4 分 | これ |
| 全部 1 本 | +2 分 | 依存衝突（torch / ollama）で破綻 |

境界は 3 条件だけ: **エンジンが違う / Dataset が違う / ハングを隔離したい**。

## 使い方

```bash
# 作成のみ（GPU 0 秒）
python3 -m cli push gpu-vllm-start --no-run --yes

# 実行（ケースを絞る例）
KC_EXP=eager python3 -m cli push gpu-vllm-start --yes
```

`KC_EXP` は Kaggle 側の環境変数。`kernel-metadata.json` には書かない
（push のたびに変える前提なので、コード内の既定 `all` を尊重する）。

## 結果の見方

`kaggle kernels output <id> -p /tmp/x/` → `/tmp/x/result.json`

- vLLM: `cases.<exp>.phase` に `load_weights` / `weights_done` / `compile` /
  `capture` / `server` の**実秒**。`stalled=true` はハング（`KC_STALL` 秒ログ無進展）。
- llama: `cases.<exp>.bench` に `tps` / `draft_n` / `draft_accepted` / `accept_rate`。
  `vram_loaded_mib` が増えていなければ CPU 落ち。
- ollama: `cases.<exp>.loaded.size_vram` が 0 なら CPU 実行（失敗扱い）。
  `context_length` が env と違えば `OLLAMA_CONTEXT_LENGTH` が無視されている。

## 設計

- 失敗・ハングは**ケース単位で隔離**。1 本のハングで全体を捨てない。
  vLLM は `KC_STALL`（既定 240s）でログ無進展を検知して打ち切る。
- 各ケースの前にプロセス木と `/dev/shm` を掃除（`pkill -9` + VRAM 解放待ち）。
  残骸があると次が `Engine core initialization failed` で即死する（実測）。
- 全体予算 `KC_BUDGET`（既定 3300s）。超過したケースは `skipped` として残す。
- 失敗時は **exit 1**。GPU を消費したのに COMPLETE で隠さない。
- 単一ファイル push なので 3 本でステージング処理が重複する（意図的）。
  共有化するなら `kbuild` が生成時に埋め込む形にする。

## コスト目安（T4 x2）

| カーネル | install | 1 ケース | 全ケース |
|---|---|---|---|
| vLLM | 0.5〜1 分（wheelhouse）/ 4 分（PyPI） | 3〜10 分 | 20〜35 分 |
| llama | 1 分 | 2〜6 分（ロード 345s 実測） | 12〜25 分 |
| ollama | 1 分 | 3〜5 分（ロード 198s 実測） | 10〜15 分 |

1 日 1〜2 本まで。残クォータが `KC_BUDGET` 未満のときは push しない。

## 追加ケース一覧

- llama batch: `batch128` / `batch256` / `batch1024`（prefill と decode を分離して比較）
- llama 分割: `split4852` / `split4555` / `split_row`（GPU 別 VRAM は `nvidia-smi` で記録）
- llama MTP: `mtp_tok1` / `draft_mtp` / `mtp_tok4`（accept rate 60%以上を候補、30%未満を無効候補）
- llama KV: `kv_f16` / `none` / `kv_q4`（q4_0 が実装されない場合は log_tail に残る）
- vLLM 並列: `maxseq1` / `maxseq4`
- vLLM profile: `profile_fast`（8K/1）/ `profile_normal`（32K/2）/ `profile_long`（131K/1）/ `profile_batch`（16K/4）

# 計測と比較の回し方（2026-11-01 作成）

GPU を回すたびに「速くなった / 遅くなった」を**数字で**言うための土台。
以前は `result.json` の `timings.install_seconds` が **None のまま**で、
比較ができず「なんとなく遅い」で終わっていた（vLLM は install をテンプレートの
`stage_install()` ではなくセルで行うため、誰も記録していなかった）。

## 1 回の実行で残るもの

`result.json` の `timings` に、セルが書いた計測が合流する。

| キー | 中身 | 分解能 |
|---|---|---|
| `install_seconds` | install セル全体（purge → import 確認） | 0.1s |
| `install_marks` | `pip` / `torchaudio` / `cli_help` / `import` の到達時刻 | 0.1s |
| `torch_after_install` | **install 後**の torch / CUDA（`result.json` の `torch` は install 前） | — |
| `wheelhouse` | wheelhouse Dataset を使えたか（`dirs` / `no_index`） | — |
| `load_seconds` | 起動（プロセス起動 → 生成スモーク成功） | 0.1s |
| `attempt` | 成功した段（`label` / `max_model_len` / `index`） | — |
| `startup_marks` | 各節目の到達時刻（起動からの秒） | **1s**（ログ時刻） |
| `startup_spans` | `weights` / `compile` / `after_compile` / `capture` | 1s |
| `startup_gaps` | **無音区間**の長さと「直前 / 直後の行」 | 1s |
| `speed_tps` / `warmup_seconds` | ウォームアップの実測 | 0.1s |
| `gpu_vram_mib` | 各 GPU の使用 VRAM | MiB |
| `prewarm` | 先読みの実測（`mib_s` / `threads` / `probe` / `projected_s` / `aborted`） | 0.1s |
| `install_steps` | 外部エンジンの**ステップごと**の `rc` / 秒 / 失敗時の `tail` | 0.1s |
| `triton_cache_before` | 起動前の `~/.triton/cache` のエントリ数とパス | — |
| `triton_cache` | 起動後との比較（`new_entries` / `reused_pct` / `entries_after`） | — |
| `warm_cache` | 退避した tar（`path` / `bytes` / `seconds` / `skipped` の理由） | 0.1s |

`triton_cache.new_entries` は**「キャッシュが当たったか」の判定値**。
「ファイルがある」ではない（キーが違えば黙ってミスし、373 秒が戻るだけ）。
0 に近いほど当たっている。`entries_before` が 0 なら初回＝全部作り直し。

`/kaggle/working/warm-cache.tar` は次回の Dataset 用（3 GB を超えたら退避しない）。
`/kaggle/working/ram_trace.txt` も**成功・失敗どちらでも**残る（5 秒ごとの
`avail / free / cache / shm` + RSS 上位 3）。`timings` は 1 行の要約だが、
こちらは推移なので「いつ何が起きたか」を後から追える。

### なぜ 2 系統の時刻を使うのか

- **ログ時刻**（`10-31 13:22:08`）… 1 秒精度。ただし tqdm の行
  （`Capturing CUDA graphs`）には時刻が無い。
- **ポーリング時刻**（60 秒間隔）… 粗いが、tqdm しか出さない区間も拾える。

`startup_marks` は **ログ時刻を優先し、無い節目だけポーリングで補完**する。
片方だけだと「381 秒の無音区間」が 60 秒丸めになるか、丸ごと欠落する。

**ポーリングだけを信じてはいけない**（実測 2026-11-01）: 60 秒粒度で測った
`graph capture` は「41 秒」だったが、ログ時刻で測ると **2.2 秒**。41 秒の
大半は無音区間の末尾だった。この 39 秒の差が `--enforce-eager` の見積りを
狂わせていた。

### 無音区間（`startup_gaps`）の読み方

`startup_spans` は「どの区間が長いか」しか言わない。長い無音が**何の直後**に
始まり**何の直前**に終わるかが分からないと、次の一手が決まらない。

```
⏱ 無音 381.0s @282.0s: 「... torch.compile took 61.00 s in total」
                     →「... Graph capturing finished」
```

同じ本文の行（60 秒おきの警告など）は**埋め草**として扱い、無音を分断しない
（5 本に割れると「60 秒が 5 回」に見えて「381 秒詰まった」に見えない）。

## 比較する

```bash
python3 -m cli timings out/run1/result.json out/run2/result.json
```

```
engine        総計 s    install s  起動 s   起動まで s  段       ctx    torch         wh  t/s  ...
vllm (out2g)  928.7     -          -        -           -       65536  2.11.0+cu128  -   -    ...
vllm (out2h)  4637.4    271.3      706.5    977.8       mtp+dd  65536  2.13.0+cu130  on  31.0 ...
```

- **`起動まで s` = install + 起動**。比較に使うのはこれ。
  `総計 s` は keepalive（配信）を含むので**実行時間の比較に使えない**
  （v10 の総計 4,637s のうち 3,660s は配信していた時間）。
- **未計測は `-`**。0 で埋めない（「一瞬で終わった」と誤読する）。
- 表の下に `startup_marks` の生値と `startup_gaps` が出る。

## 現在の基準値（2026-11-01 13:00 / **v14** が最新）

| 区間 | v10 | v14 | 備考 |
|---|---|---|---|
| install（wheelhouse `--no-index`） | 271.3s | **227.7s** | pip に戻して改善 |
| 先読み | — | **152.5s** | **効かない**（`notes/vllm-startup-speedup.md` §15.1） |
| 起動 | 706.5s | 486.1s | v14 は `eager` に落ちた回なので**単純比較できない** |
| 起動まで | 977.8s | 866.3s | 同上（−111s は compile を飛ばした分） |
| **実効 t/s** | **31.0** | **4.5** | **v14 は 1/7。速度を捨てている** |

**「起動まで」だけで良し悪しを言わない。** v14 は起動が 111 秒速いが、
`eager` に落ちて 31.0 → 4.5 t/s。**必ず `speed_tps` と `attempt.label` を並べて読む。**

旧基準（v9 / PyPI 経路）は install 259.9s + 起動 642.4s = 915.5s。
v10 の価値は数字ではなく**内訳が機械で埋まったこと**。

## 施策を入れたら必ず 3 つ揃える

1. 変更前の `result.json` を残す（`cli timings` で並べる）
2. 変更後の `result.json` を取る（同じカーネル・同じ ctx で）
3. `notes/gpu-run-sheet.md` の結果表を更新する

**比較できない変更は入れない**。速くなったか分からないまま積むと、
どこで壊れたかも分からなくなる（`torchaudio` の版ずれで 8 往復した前例）。

**「速くなった」と言う前に、どの試行が成功したかを見る。** v14 は
1 発目（`mtp+dd`）が 921 秒で予算切れし、2 発目（`eager`）に落ちて
「起動 −111 秒」になった。数字だけ見ると改善、中身は退行。

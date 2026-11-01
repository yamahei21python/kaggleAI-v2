# vLLM 起動時間の短縮（調査メモ / 未実装）

対象: `kaggle/build_step2_vllm.py` の GPU 実行 1 回あたりの起動時間。
方針: **GPU クォータを食う工程を削る**。CPU で測れるものは CPU で測る。

> 運用制約: **GPU kernel の push はユーザーの明示的な Go があるまで禁止**（CPU で潰せる検証を先に全部やる）。

> **⚠ §12 を先に読む（2026-10-31 追記）**
> #2 vLLM が初めて成功し、**全工程の実測内訳**が取れた。§1〜§11 の「未計測 / 仮説」は
> §12 で置き換わっている。結論だけ先に書くと、**最大のボトルネックは install でも
> モデルロードでもなく、torch.compile 直後の「無音の 360 秒」**だった。
> 以下 §1〜§11 は当時の見積りの記録として残す（判断の根拠を後から検算できるように）。

---

## 1. 現状の内訳（実測値）

| 工程 | 実測 | 出典 |
|---|---|---|
| `pip install vllm`（GPU kernel） | **457s ≒ 7.6 分** | v1 実行ログ 469.9s 時点で完了 |
| `pip install vllm`（CPU kernel） | **276〜306s ≒ 4.6〜5.1 分** | dryrun v2 / uvbench |
| 依存込み wheel 総量 | **3.94 GB / 198 files** | uvbench |
| 依存込み download だけ | **137s**（≒ 29 MB/s） | uvbench |
| vllm wheel 単体 | 326 MB / **13.4s** | uvbench |
| `uv pip install`（依存済み・vllm のみ） | **6.5s** | uvbench |
| torchaudio 整合の修復 | 0〜120s | GPU では発動見込み（CPU では不要だった） |
| `vllm serve --help` | 27s | dryrun v2 |
| モデルロード（19 GB / /kaggle/input） | **未計測** | — |
| Triton JIT warmup（AWQ） | **未計測** | 初回リクエストで発生する可能性 |
| CUDA graph capture | **未計測** | 通常 30〜90s |
| KV cache profile | 数十秒 | — |
| プロキシ + cloudflared + verify + keepalive | 約 60s | v1/dryrun から推定 |

**確実に削れるのは install の 7.6 分。** それ以外は「測ってから」になる。

環境: Python 3.13 / vLLM 0.32.0 / torch 2.13.0+cu130 / CUDA 13.0 / T4 x2 (sm75)

---

## 2. 候補（効果 × リスク × 検証方法）

### A. uv でインストール（推奨・低リスク）
```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"])
subprocess.run(["uv", "pip", "install", "--system", "-q", "vllm"])
```
- 効果: pip の解決+展開が 2〜4 倍速い → **-3〜5 分**見込み
- リスク: `--system` の挙動差 / 依存解決結果が pip と完全一致しない可能性
- 検証: **CPU kernel で所要時間を測るだけ**（無料・5 分）

### B. wheel を Dataset 化（推奨・中リスク）
```
CPU kernel: pip download vllm==0.32.0 -d /kaggle/working/wheels
→ Dataset: yamahei21python/vllm-wheels-0.32.0
実行時: uv pip install --no-index --find-links /kaggle/input/vllm-wheels-0.32.0 vllm==0.32.0
```
- 効果: **-5〜6 分**（実質 1 分程度）
- リスク: サイズ 4〜6 GB / torch 版がイメージ更新でズレると解決失敗
- 対策: Dataset 名に版を入れる。失敗時は通常 install にフォールバック
- 検証: CPU kernel で download と install リハーサル（無料）。アップロードも GPU 不要

### C. Triton / vLLM 実行時キャッシュを Dataset 化（中リスク）
```python
os.environ["VLLM_CACHE_ROOT"] = "/kaggle/working/vllm-cache"   # 初回のみ生成
os.environ["TRITON_CACHE_DIR"] = "/kaggle/working/triton-cache"
```
- 効果: 2 回目以降の起動で **-1〜3 分**（JIT/autotune/compile cache のヒット分）
- 注意: キャッシュは GPU アーキ（sm75）と vLLM/torch 版に依存。**T4 で作ったものだけ使える**
- リスク: キー不一致で無効化（無害）/ Dataset に書き込めない（RO）ので**生成→Dataset 化の 2 段が必要**
- 関連 env: `VLLM_DISABLE_COMPILE_CACHE` / `VLLM_USE_AOT_COMPILE` / `VLLM_FORCE_AOT_LOAD`

### D. `--enforce-eager` を本命段に（低リスク・即効）
- 効果: CUDA graph capture を丸ごと省略 **-30〜60s**、VRAM も節約（T4 では収まりが良くなる）
- 代償: 生成速度が数 % 落ちる（起動は速い）
- 既に最終手段としてラダーに追加済み。**2 番目の候補に格上げ**する価値あり

### E. `--load-format fastsafetensors`（要検証）
- vLLM 0.32 に存在（env `VLLM_FASTSAFETENSORS_QUEUE_SIZE` が定義されている）
- 効果: 19 GB ロードを並列化 **-30s〜**（シングルファイルなので伸び幅は限定的）
- リスク: メモリピーク増（queue_size>0 時）。T4 30 GB では queue=1 まで
- 検証: 起動ログの "Loading weights took" を比較

### F. `--max-model-len` を小さく / `--max-num-seqs 1`（低リスク）
- KV profile と graph capture のバリエーションが減る **-10〜30s**
- 用途が単発チャットなら 8192 で十分

### G. `VLLM_TRITON_FORCE_FIRST_CONFIG=1`（中リスク）
- Triton autotune のベンチマークを飛ばして先頭 config を採用
- 効果: 初回 JIT 時間 -30〜90s / 代償: カーネル性能が数 % 落ちる可能性

### H. 起動待ちポーリングの短縮（極小）
- 現在 `/v1/models` を 5s 間隔 → 2s 間隔、判定は `/health` ではなく `/v1/models`
- 効果: **-2〜4s**。誤差レベルだがタダ

### I. 採用しない案
| 案 | 理由 |
|---|---|
| `pip install --no-deps vllm` | vLLM は torch 版を厳密に固定。イメージの torch とズレて即死 |
| モデルを fp16 に変換 | 19 GB → 54 GB。T4 に載らない |
| TensorRT-LLM | engine build が T4 では効果薄（既に却下済み） |
| TP=1 に変更 | 27B AWQ が 1 枚に載らない |

---

## 3. 期待値（すべて仮説。実測前は確定値として扱わない）

**重要な誤りやすい点**: A（uv）と B（wheel Dataset）は**同じ install 工程**を削るので加算できない。
install は「どちらか一方（併用なら短い方）」になる。

| 状態 | install | 起動〜公開まで | 根拠 |
|---|---|---|---|
| 現状（実測） | 7.6 分 | **約 20 分** | 実測は install のみ。他は推定 |
| A: uv のみ | 3〜5 分?（**未計測**） | 15〜17 分 | 依存解決が主なら効く / 転送・展開が主なら効かない |
| B: wheel Dataset | 1〜2 分?（**未計測**） | 13〜15 分 | 転送がローカル Dataset になり、解決もオフライン化 |
| A+B 併用 | 1〜2 分? | 13〜15 分 | **A の上に B が乗るのではなく、B が A を置き換える** |
| + C（JIT cache） | 同上 | -1〜3 分 | sm75 で作ったキャッシュのみ有効 |
| + D（enforce-eager） | 同上 | -0.5〜1 分 | 速度は別途測定 |

→ 到達目標は **13〜15 分（B のみ）**、キャッシュが効けば **10〜13 分**。
「6〜8 分」は未計測の願望なので採用しない。

### 3.1 まず測るべき内訳

install の 457s が何で決まっているかで、A の価値が変わる。

```bash
# CPU kernel（無料）
time pip download --no-deps vllm==0.32.0 -d /tmp/w   # 純粋な転送時間
time pip install --no-cache-dir vllm                 # 解決+展開込み
```

- 転送が支配的 → **A は無効、B（wheel Dataset）だけが効く**
- 解決が支配的 → A が効く（uv は解決が速い）

### 3.2 実測結果（uvbench, CPU kernel）

| 測定 | 値 |
|---|---|
| `pip install vllm==0.32.0`（cold） | 305s |
| `pip download vllm==0.32.0 --no-deps` | 13.4s / 326 MB |
| `pip download vllm==0.32.0`（依存込み） | 137s / **3.94 GB / 198 files** |
| `uv pip install --system`（依存済み・vllm のみ） | **6.5s** |

読み方:
- **転送が支配的**（3.94 GB、137s）。install 305s のうち ~137s は純粋な取得。
- vllm 単体は 326 MB / 13s なので、残り ~3.6 GB は torch(CUDA 13) 等の依存。
- → **B（wheel Dataset）が本命**。転送 137s がローカル読み込みになり、
  解決も `--no-index` で消える。見込み install 60〜120s（仮説）。
- A（uv）は「vllm 単体 6.5s」だが、cold では依存転送が残るため短縮は 1〜2 分止まり（仮説）。
  併用するなら「B のインストーラとして uv を使う」形が正しい（加算ではない）。

---

## 4. 検証手順（GPU を食わない順）

1. **CPU kernel**: `uv pip install --system vllm` の所要を測る（A の効果確定）
2. **CPU kernel**: `pip download vllm==0.32.0 -d wheels` → Dataset 化（B の材料）
3. **CPU kernel**: 手順 2 の wheel から `--no-index` install の所要を測る
4. **GPU 1 回**: 各段の所要をログに出す（現在の実装で `install 所要` と試行ログは出る。
   モデルロード時間は vLLM 自身のログ `Loading weights took ...` を grep して拾う）
5. 効果が出た項目だけ本番ビルダーへ反映

---

## 5. 反映時の注意

- キャッシュ系（C）は **Dataset が read-only** なので `/kaggle/working` にコピーしてから使う
- wheel（B）は **vLLM の版を上げたら作り直す**。`VLLM_PIP_SPEC` と Dataset 名を連動させる
- どの施策も **失敗時に通常 install へフォールバック**させる（起動できない方が損失が大きい）
- `--enforce-eager` は速度とのトレードオフ。速度を測ってから本命に昇格させる

---

## 6. 追加実測（2026-10-29, CPU kernel `ioread` / `vram2`）

| 測定 | 値 | 出典 |
|---|---|---|
| `/kaggle/input` シーケンシャル読み | **82 MiB/s** | ioread（512 MiB / 6.27s） |
| 重み 19.5 GiB の読み下限 | **244s**（TP=2 で各 122s） | 上の実測からの計算 |
| `/kaggle/working` 空き | 19.5 GiB | ioread |
| ステージング | **symlink のみ**（実体コピーなし） | build_step2_vllm.py `_stage_model` |
| cloudflared バイナリ | **未インストール**（毎回 DL） | ioread（`which` = None） |

→ **モデルロードは「未計測」ではなく「122〜244s」と確定**（I/O 律速）。
`--load-format fastsafetensors`（施策 E）はこの帯域を上げられないので**期待薄**。
効くのは「読む量を減らす」方向: `--language-model-only`（visual 0.86 GiB 削減）と
TP=2 で各 GPU が半分だけ読む構成（既に実施）。

## 7. torchaudio 修復の実際（v2 ログ + PyPI 実測）

- GPU 画像: torch **2.13.0+cu130** / torchaudio **2.8.3+cu128** → CUDA 版数不一致で import 失敗
- PyPI の `torchaudio==2.11.0` wheel は **`2.11.0+cu130`**（wheel 内 `torchaudio/version.py`）
- したがって `pip install --no-deps --force-reinstall torchaudio==2.11.0` で一致する（~10〜30s）
- 実装変更: `vllm serve --help` 経由ではなく **`python -c "import torchaudio"` を serve 前に単独実行**
  → 失敗時のみ修復。失敗が続けば uninstall（本モデルは音声不要）
- **所要 0〜60s**（初回のみ）。起動時間に毎回乗るものではない

## 8. `--chat-template`（0.32 でも実在。dataclass 経由の自動登録）

- 0.32 は renderer 方式だが、`--chat-template` 自体は `FrontendArgs` に残っている
  （`cli_args.py:86`）。`vllm/renderers/hf.py:resolve_chat_template()` が
  AutoProcessor → AutoTokenizer → 組み込み fallback の順で解決する。
- このモデルは `tokenizer_config.json` に `chat_template` が無いが、
  `chat_template.jinja` があり transformers が読む（chattpl で実描画確認済み）
  → **自動解決で足りる。フラグ付与は不要**（付けると未知引数で即死）
- 起動時間への影響: なし（テンプレート解決は数 ms）
- 副次効果: `--chat-template` の存在確認が不要になり、`_has()` の楽観判定を外せた

## 9. 更新後の見積り（施策 B 前提・実測で置き換えた部分のみ太字）

| 工程 | 見積り | 根拠 |
|---|---|---|
| wheel Dataset から install（`--no-index`） | 60〜120s | 3.94 GB を 82 MiB/s で読むと ~48s + 展開 |
| **モデルロード** | **122〜244s** | **実測 82 MiB/s** |
| CUDA graph capture | 30〜90s（`--enforce-eager` で 0） | 仮説 |
| KV cache profile | 10〜30s | 仮説 |
| プロキシ + cloudflared DL + verify | 60〜90s | v1 実測 + cloudflared 不在 |
| **合計（公開 URL まで）** | **5〜9 分** | 従来見積り 20 分から短縮 |

- 30 分セッションなら「起動 9 分 + 生成 20 分」で収まる。
- さらに削るなら **`--enforce-eager` を本命段に昇格**（-30〜90s、速度は数 % 減）。
- **`--language-model-only` も速度に効く**（visual 0.86 GiB を読まない）。ラダー最終段に追加済み。

## 10. 優先順位（更新）

1. **B（wheel Dataset）** … install の主因は転送 137s。ローカル化で最も効く
2. **モデルロード短縮** … `--language-model-only` / 不要ファイル除外（symlink 済み）
3. **`--enforce-eager`** … 起動短縮と VRAM 節約の両取り。速度劣化は許容範囲か要測定
4. **C（JIT cache Dataset）** … sm75 固有。2 回目以降のみ。初回 GPU では効かない
5. **A（uv）** … B のインストーラとしてのみ意味がある（単独では 1〜2 分）
6. **E（fastsafetensors）** … I/O 律速のため期待薄。測定優先度を下げる

## 11. ロード量の実測と draft の二重読み（probe9 loadflow）

| ファイル | サイズ |
|---|---|
| `model.safetensors` | **17.41 GiB** |
| `model-mtp.safetensors` | **0.79 GiB** |

- `allow_patterns=['*.safetensors']` で `model.safetensors.index.json` が無いため
  `filter_duplicate_safetensors_files` は素通し → **target が mtp も読む**
- draft の `ModelConfig.model` は target と同じ dir になる（specmat 実測）ため、
  draft の loader も `*.safetensors` を glob して **17.41 GiB 側まで読む**
- 82 MiB/s 前提だと target 18.21 GiB = 227s、draft も同程度 → **合計 450s 級**
  （§9 の「122〜244s」は target 1 回分。MTP 有効時はこの 2 倍近くかかる）

対策候補（`--speculative-config` の `model` に最小ディレクトリを渡す）:
```json
{"method": "mtp", "num_speculative_tokens": 2, "model": "/kaggle/working/draft"}
```
`/kaggle/working/draft` には `model-mtp.safetensors` と `config*.json` /
`tokenizer*` だけを symlink する（glob が 1 ファイル 0.79 GiB になる）。
検証は probe11（draftdir）。成立すれば **-200s 以上**。

なお `_stage_model(with_mtp=False)` は mtp ファイルを symlink しないので、
MTP なしの段では 0.79 GiB を読まない（既存実装で正しい）。

---

## 12. 実測内訳（2026-10-31 / #2 vLLM 初成功 `result.json` elapsed 928.7s）

出典: `/tmp/out2g/step2-vllm-qwen38-27b-cloudflared-script.log`（JSON 行、`time` は
カーネル開始からの秒）。**推定ではなく全部ログの時刻差**。

| 区間 | 絶対時刻 | 長さ | 何が起きているか |
|---|---|---|---|
| install（purge → import 確認） | 13.0 → 255.7 | **242.7s** | 下に分解 |
| └ torch 物理削除 | 13.0 → 13.8 | 0.8s | 8 パス削除（`_purge_torch`） |
| └ `pip install vllm==0.32.0`（PyPI） | 13.0 → 212.6 | **199.6s** | wheelhouse Dataset が無く PyPI 経路 |
| └ torchaudio 整合確認 | 212.6 → 222.6 | 10.0s | `torchaudio import: OK`（修復は発動せず） |
| └ import probe（torch+vllm） | 222.6 → 255.7 | 33.1s | 子プロセスで `import torch, vllm` |
| `vllm serve --help=all` | 255.8 → 273.0 | 17.2s | 154,018 文字を保存（旧実測 27s から短縮） |
| **エンジン起動** | 273.0 → 915.4 | **642.4s** | 下に分解 |
| └ 重みロード | 273 → 454 | **約 181s** | VRAM 0 → 9,097 → 9,459 MiB |
| └ torch.compile | 454.2 → 514.3 | **61.2s** | vLLM 自身が `took 61.22 s` と出力 |
| └ **無音区間（未同定）** | 514.3 → 874 | **約 360s** | 出力は `shm_broadcast ... 60 seconds` のみ。VRAM 9,995 → 10,733 MiB へ漸増 |
| └ CUDA graph capture | 874.6 → 915.4 | 約 41s | `Capturing CUDA graphs (FULL) 1/2` |
| ウォームアップ + 検証 6 項目 | 915.5 → 928.7 | 13.2s | 生成 31.7 t/s、VRAM 13,343 MiB |

配分: **install 26.1% / 起動 69.2% / その他 3.3%**。旧見積りの「install が主因（7.6 分）」
は外れで、実際は **install 4.0 分 + 起動 10.7 分**。削るべきは起動側。

### 12.1 最大の的は「無音の 360 秒」

> **2026-11-01 追記**: v10（計測機構入り）で位置は確定した。**§13.3 を先に読む**。
> 以下は v9 時点の記録として残す。

`torch.compile` 終了（514.3）から CUDA graph capture 開始（874.6）までの 360 秒、
ログには 60 秒ごとの `shm_broadcast` 警告しか出ない。VRAM が 9,995 → 10,733 MiB へ
**少しずつ**増えるので、何かを段階的に確保しながら進んでいる（一括確保ではない）。

**§11 の「draft の二重読み」は除外できる（2026-11-01 検算）**。試行 1 は既に
`spec=mtp+dd` で走っており、`vllm_cmd.json` の `--speculative-config` にも
`"model": "/kaggle/tmp/vllm-draft"` が入っている（ログ: `draft dir : /kaggle/tmp/vllm-draft
（symlink 7 個）`）。つまり draft は 0.79 GiB だけ読む構成で、**対策は実装済み**。
360 秒の原因ではない。ここを疑って 1 往復溶かさないために明記しておく。

残る候補（**いずれも未同定。断定しない**）:

1. **ダミー実行 / memory profiling** … compile 後に shape ごとの実 forward が走る
2. **追加サブグラフの compile** … `torch.compile took 61.22 s` は backbone 分だけ
3. **AWQ GEMM の Triton autotune**（sm75 で初回）
4. KV cache の確保と block 分割

**切り分け方は 2 つ。どちらも GPU 1 回で足りる**:

- 2026-11-01 に **節目ログを実装した**（`_MILESTONES`）。起動待ちの 60 秒出力が
  `log N KiB (+増分) / 節目: <最後に越えた関門>` を出すので、360 秒の間に
  「どの関門の後で止まっているか」が読める。増分 0 なら無言の計算、増えていれば
  ログを出しながらの工程。
- PLAN の 2 番目が `eager`（`--enforce-eager`）。eager で 360 秒が消えれば
  graph capture 周辺、残れば compile/autotune。

### 12.2 優先順位の付け直し（実測ベース）

| 順 | 施策 | 見込み | 根拠 |
|---|---|---|---|
| 1 | **無音区間の同定** | 測定のみ（実装済） | 360 秒 = 全体の 39%。正体が分からないまま他の施策を積むのは順序が逆 |
| 2 | **B: wheel Dataset** | **-140s** | install 199.6s のうち転送が支配（3.94 GB / 137s 実測） |
| 3 | C: Triton/JIT cache | -60〜120s? | 無音区間が autotune なら効く。sm75 固有 |
| 4 | D: `--enforce-eager` | **-41s**（実測 capture 分） | 旧見積り 30〜90s の実測は 41s。速度劣化と引き換え |
| 5 | F: `--max-num-seqs` / `--max-model-len` を下げる | -10〜30s? | `max-num-seqs 2` は既に適用済み。compile 対象の削減余地は小さい |
| 6 | A: uv | -60s? | B のインストーラとしてのみ意味がある |

2 だけで **928.7 → 約 790s**。1 で 360 秒の正体が分かってから 3〜5 を積む。
無音区間が丸ごと消せれば **約 570s（9.5 分）** まで落ちる計算になる。


### 12.3 副産物: `result.json` の `torch` が実態とズレる

`stage_env()` は **install より前**に走るので、`result.json` の `torch` は
`2.11.0+cu128`（イメージ同梱）になる。実際に配信しているのは `2.13.0+cu130`。
誤読の元なので、install 後に測り直した値を `torch_serving` のような別キーで
残すのが望ましい（未実装）。

→ **2026-11-01 実装**: `timings.torch_after_install` に install 後の値を残し、
`cli timings` の `torch` 列はそちらを優先する（無い回だけ `result.json` に戻る）。

---

## 13. v10 実測（2026-11-01 / 計測機構を入れた回）

v9 の内訳は**ログ時刻を人手で突き合わせた**もの。v10 は `timings` を機械で
埋めた回で、以下は全部 `result.json` から読める値。

### 13.1 数字

| 項目 | v9（手作業） | v10（機械計測） |
|---|---|---|
| install | 242.7s + help 17.2s = **259.9s** | **271.3s**（help 込み） |
| 起動 | 642.4s | **706.5s** |
| 起動まで | 915.5s | **977.8s** |
| 重みロード | 約 181s（VRAM からの推定） | **133s**（ログ実測） |
| torch.compile | 61.2s | **61.0s** |
| 無音区間 | 約 360s | **378.8s** |
| graph capture | 約 41s（ポーリング粒度） | **2.2s**（ログ実測） |

**capture 41s は幻だった。** 60 秒ポーリングで測ると無音区間の末尾を
capture と誤認する。実測 2.2s なので `--enforce-eager`（capture を消す）の
価値はほぼゼロ。

### 13.2 施策 B（wheel Dataset）は**効かない**（実測で決着）

v10 は wheelhouse Dataset（190 + 11 wheel、`--no-index`）を実際に使った
（ログ: `📦 ローカル wheelhouse: vllm-wheelhouse（190 wheel）`）。結果:

| 経路 | pip の所要 |
|---|---|
| v9: PyPI から取得 | **199.6s** |
| v10: wheelhouse（`--no-index`） | **230.4s** |

**31 秒遅くなった。** §2-B の「-140s」は「199.6s のうち転送が支配（3.94 GB /
137s）」という前提だったが、その前提が外れている。ローカル wheel でも
**3.25 GB の展開と数万個の小ファイル書き込み**が支配的で、転送を消しても
縮まない。install の床はこの展開コスト。

→ B は**却下**。残る手は「展開そのものを速くする」だけ（`uv`、`--no-compile`、
事前展開済み site-packages のコピー）。これは GPU 不要の CPU カーネルで
測れる（`kaggle/cpu-tests/` の枠組み）。

### 13.3 381 秒の無音区間: 分かったこと / まだ分からないこと

**分かったこと**

- 位置: `torch.compile took 61.00 s` の**直後**から `Graph capturing finished`
  まで（282 → 663）。compile の後片付けではなく、**compile と capture の間**。
- 出ているログ: 60 秒おきの `shm_broadcast ... No available shared memory
  broadcast block found in 60 seconds. This typically happens when some
  processes are hanging or doing some 'time-consuming' work such as ray remote
  call, compilation, etc.` が 5 回だけ。**vLLM 自身が「時間のかかる作業中」と
  言っている**＝症状であって原因ではない。
- GPU: `util=0%, mem=10,733 MiB`（60 秒ごとの 5 サンプルすべて 0%）。
  **GPU は遊んでいる** → ホスト（CPU）側の作業。
- VRAM: 9,995 → 10,733 MiB へ漸増。KV cache（13,343 MiB）の確保前。
- 再現性: v9 でも同じ 5 回が同じ位置（13:25:53〜13:29:53）に出ている。

**まだ分からないこと**: 何の計算をしているか。候補は compile 後の dummy run /
memory profiling の Triton JIT（sm75 初回）。**断定しない。**

**次回で確定させる計器（実装済み）**

1. 成功時も `vllm_{idx}.log` を保存する（旧実装は失敗時だけ。だから compile
   直後に出た 2,281 B を読めなかった）。
2. `startup_gaps` に「無音の直前の行 / 直後の行」を残す。長さだけでは次の一手が
   決まらない。60 秒警告の繰り返しは 1 本に連結する（5 本並ぶと「60 秒が 5 回」に
   見えて「381 秒詰まった」に見えない）。
3. `cli timings` が `起動まで s`（install + 起動）を出す。総計は keepalive を
   含むので比較に使えない（v10 の総計 4,637s のうち 3,660s は配信）。

### 13.4 優先順位（v10 実測で付け直し）

| 順 | 施策 | 見込み | 状態 |
|---|---|---|---|
| 1 | 無音 381 秒の中身を読む（成功時ログ保存） | 測定 | **実装済・次回 1 本で確定** |
| 2 | 無音が Triton/JIT なら C（cache Dataset 化） | -250〜350s? | 1 の結果待ち |
| 3 | install の展開コスト（uv / `--no-compile`） | -60〜150s? | **CPU で測れる**（GPU 0 秒） |
| 4 | ~~B: wheel Dataset~~ | **+31s（実測）** | **却下** |
| 5 | ~~D: `--enforce-eager`~~ | **-2.2s** | **却下**（速度劣化に見合わない） |
| 6 | F: `--max-num-seqs` / `--max-model-len` を下げる | -10〜30s? | 据え置き |

1 が終わるまで 2 は動かさない。**381 秒 = 起動の 54%** なので、ここが外れれば
他を積む意味が変わる。

## 14. v11 / v12（2026-11-01）— 2 つの施策が実測で落ち、別の壁が出た

### 14.1 install の高速化: uv は**本番でだけ**遅い → 却下

CPU ベンチ `vllm-install-bench`（`pip install --target` で同じ 190 wheel）:

| 戦略 | 所要 | files |
|---|---|---|
| pip（暖・基準） | 200.1s | 81,844 |
| pip `--no-compile` | 112.6s | 56,416 |
| **uv** | **35.7s** | 56,812 |
| pip `--no-deps`+パス直指定 | 101.5s | 47,835 |

差の大半は **pyc 25,428 個の生成**。転送ではない。そこで uv を既定にした。

**本番（GPU 実測）:**

| 経路 | install | 内訳 |
|---|---|---|
| v10 pip + wheelhouse | 271.3s | pip 230.4 |
| v11 pip + wheelhouse | 304.2s | pip 261.6 |
| v12 uv + wheelhouse | **1208.5s** | **uv 1151.7** |

**比が移らない。** bench は `--target` の**まっさらなディレクトリ**、本番は
torch を物理削除した **site-packages**。同じ wheel・同じホストで 4.4 倍遅い。
理由は未同定（uv のキャッシュ経路が GCSFUSE 上の wheel を引き直している疑い）。

**教訓**: `--target` のベンチは「展開の仕方」の比較には使えるが、
**install 先が違う**ので絶対値も比も本番に持ち込めない。既定を変える前に
本番と同じ形（site-packages へ入れる）で 1 回測る。既定は pip に戻した。

### 14.2 起動: 本当の壁は「重みロード」だった

v12 の 2 試行はどちらも同じ場所で死んだ。

```
Filesystem type for checkpoints: FUSE.GCSFUSE. Checkpoint size: 18.21 GiB.
Available RAM: 5.68 GiB.
Auto-prefetch is disabled because the filesystem (FUSE.GCSFUSE) is not a
recognized network FS (NFS/Lustre) and the checkpoint size (18.21 GiB)
exceeds 90% of available RAM (5.68 GiB).
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:02<00:02]
→ ここから 800 秒、ログ +0 バイト / GPU util 0% / VRAM 9,097 MiB で固定
```

vLLM は **checkpoint が空き RAM の 90% を超えると auto-prefetch を切る**。

> ⚠ **この因果は 2026-11-01 に半分撤回した。** GCSFUSE では**空き RAM に関係なく
> auto-prefetch は常に切れる**（§19）。RAM の数字は「どちらのメッセージが出るか」
> を選ぶだけで、prefetch の可否を決めていない。以下の「v10 は prefetch が
> 有効だった」という推定も成立しない。

切ると GCSFUSE の safetensors を頁単位で mmap するしかない。

v10 は同じ symlink ステージング・同じ FUSE で重み 133 秒だった
（= 18.21 GiB / 133s = 140 MiB/s）。~~**v10 は prefetch が有効だった**と推定できる。
差を決めているのは起動時点の `MemAvailable` だけ。~~
→ 上の ⚠ の通り、prefetch は当時も無効だった。v10 の 133 秒は**ホスト側の
読みが速かった**としか言えない（同じ構成で 20 MiB/s の回がある。§19）。

**分からないこと**: v12 は試行開始の 4 分前に `空き RAM: 30315 MiB` と出ていた。
その 24 GiB がどこへ消えたかは特定できていない。空き 1 個の数字では
「page cache が張り付いた」のか「誰かが anon で食った」のか決められない。

→ 対策は 2 つ。どちらも GPU 0 秒で進められる。

1. **内訳を残す**（実装済）: `MemTotal / MemFree / MemAvailable / Cached /
   Shmem` + RSS 上位 5 プロセスを起動待ちのループで毎回出す。
2. **CPU で I/O を測る**（`vllm-io-bench`・実装済）: GCSFUSE の逐次読み /
   mmap / ローカル実コピー / コピーの読み直し。各フェーズの meminfo も残す。
   「読むと空きが減る」なら**コピーでは解決しない**と分かる。

### 14.3 優先順位（v12 後）

| 順 | 施策 | 見込み | 状態 |
|---|---|---|---|
| 1 | **重みロードの 800 秒ハングを外す**（prefetch を効かせる / ローカル化） | **-700s?** | `vllm-io-bench` の結果待ち |
| 2 | 無音 381 秒の中身（成功時ログ） | 測定 | v12 でログは取れたが起動に届かず |
| 3 | install を pip に戻す | **-890s（実測）** | **完了** |
| 4 | ~~uv~~ | **+890s（実測）** | **却下** |
| 5 | ~~B: wheel Dataset~~ / ~~D: `--enforce-eager`~~ | +31s / -2.2s | 却下 |

**1 が通るまで他は積まない。** 起動に到達しないと 381 秒の話も始まらない。

## 15. v14（2026-11-01 12:48）— 先読みを実測したら**逆効果**だった

v13 で入れた「8 並列の先読み」は、io-bench v2 で **2.2 MiB/s**（1 本の 1/50）と
分かったので **1 本**に直して v14 を投入した。結果は「起動まで −111 秒」だが、
中身は**退行**。

| 項目 | v10 | v14 |
|---|---|---|
| install | 271.3s | **227.7s** ✅ |
| 先読み | — | **152.5s**（122.2 MiB/s / 1 本） |
| 起動（`load_seconds`） | 706.5s | 486.1s |
| 合計 | 977.8s | 866.3s |
| **成功した試行** | 1 発目（`mtp+dd`） | **2 発目（`eager`）** |
| **速度** | **31.0 t/s** | **4.5 t/s（7 倍遅い）** |

**−111 秒は `eager` が compile を飛ばした分。** 1 発目は 921 秒で予算切れした。
速度が 1/7 になるなら、この 111 秒は割に合わない。

### 15.1 先読みは 152.5 秒の**純コスト**（効かない）

| 試行 | `Loading weights took` |
|---|---|
| 1 発目（`mtp+dd`） | **64.37s** |
| 2 発目（`eager`） | **6.39s** |

2 発目が 6.39 秒なのは、**1 発目が読み直して page cache に入れ直したから**。
つまり先読みの 18.21 GiB は、**ワーカーが `8.88 GiB × 2` の anonymous を確保する
時点で追い出されている**。先読みを活かすには「anon 確保の後」に読むしかない＝無理。

→ `KC_PREWARM` の既定は **0 にすべき**（v14 の実測）。

### 15.2 373 秒の無音の正体が確定（v10 の推定が当たり）

v14 の 2 発目は `--enforce-eager` なので **compile が無い**。それでも
`shm_broadcast.py:871` が 60 秒おきに **6 回（373 秒）** 出た。位置は KV cache 確保の
直前＝ memory profiling の最中で、直後に

```
[gpu_worker.py:759] Available KV cache memory: 3.45 GiB
[qwen_triton_warmup.py:296] Warming up Qwen GDN Triton kernels for model_type=qwen3_5_text
```

が並ぶ。**Qwen3.5 の GDN（線形注意力）カーネルの Triton JIT**。起動の 54%。

→ ここを削るには **Triton のキャッシュを持ち込む**（`~/.triton` を Dataset 化）しか
ない。同じ T4・同じ版ならキーが一致する見込み。**次の第一候補はこれ**（-373s?）。

### 15.3 24 GiB の謎は「消えていない」

`ram_trace.txt`（5 秒ごと）で v14 の推移を見ると、`MemAvailable` は
**21〜25 GiB で安定**していた。`Cached` が 20〜23 GiB あるだけで、これは先読みの
残骸＝回収可能。`Shmem` は 29 MiB。

→ v12 の `空き RAM: 30315 MiB` → vLLM の `Available RAM: 5.68 GiB` は、
**vLLM が `MemAvailable` ではなく別の値（cgroup の limit − 使用量）を見ている**
ため。「24 GiB が誰かに食われた」わけではない。

### 15.4 優先順位（v14 後）

| 順 | 施策 | 見込み | 状態 |
|---|---|---|---|
| 1 | **無音 373 秒 = GDN Triton JIT を潰す**（`~/.triton` を Dataset 化） | **-373s** | 次にやる |
| 2 | **先読みの既定を切る**（`KC_PREWARM=0`） | **-152.5s** | 実装 1 行 |
| 3 | 1 発目が予算内に収まるよう `STARTUP_BUDGET` を見直す（eager に落とさない） | 速度 7 倍の回復 | 要検討 |
| 4 | ~~並列先読み~~ | 13〜50 倍遅い | **却下（実測）** |
| 5 | ~~uv~~ / ~~wheel Dataset~~ / ~~`--enforce-eager` 単体~~ | +890s / +31s / −2.2s | 却下 |

**2 は即効・確実。1 が本命。3 は「速度を捨てない」ための前提。**

## 16. v14 のログを読み直したら 15.2 が**半分ずれていた**（2026-11-01）

§15.2 は「373 秒 = GDN Triton JIT」と書いたが、**JIT が起きている場所を取り違えて
いた**。`vllm_2.log` の行番号で押さえ直すと:

| 時刻 | 行 | 中身 |
|---|---|---|
| 03:52:21 | `default_loader.py:501` | `Loading weights took 6.39 seconds` |
| 03:52:25 | `utils.py:347` | `Using LBNHC KV cache layout` ← ここから無音 |
| 03:53:26〜03:58:26 | `shm_broadcast.py:871` ×6 | 60 秒おきの警告（373 秒） |
| 03:58:38 | `gpu_worker.py:759` | `Available KV cache memory: 3.45 GiB` |
| 03:58:38 | `qwen_triton_warmup.py:296` | `Warming up Qwen GDN Triton kernels` |
| 03:58:41 | `core.py:402` | `init engine (profile, create kv cache, warmup model) took 376.24 s` |

**名前の付いた warmup は 3 秒**（03:58:38 → 03:58:41）しかない。373 秒はその**手前**、
`determine_available_memory` が行う **profile run（dummy run）** の中で起きている。
`qwen_triton_warmup` は「本番の生成で JIT しないための予熱」であって、起動時間の
短縮には寄与しない（profile run が先に同じカーネルを作ってしまうため）。

ハングではないことは `ram_trace.txt` で裏が取れる。無音区間のワーカー RSS は

```
300.1s top=VLLM::Worker_TP:6987 VLLM::Worker_TP:6984
425.1s top=VLLM::Worker_TP:8361 VLLM::Worker_TP:8358
```

と **125 秒で 1.37 GiB（約 11 MiB/s）増え続けており**、`MemAvailable` は 25 GiB で
一定。コンパイルでメモリが積み上がっている形で、待たされているだけではない。

→ 打ち手（`~/.triton` を持ち込む）は**変わらない**。profile run も同じ Triton
キャッシュに落ちるため。ただし「warmup を消せば速くなる」という読みは誤りなので、
そこを根拠にした施策は立てない。

### 16.1 退避が一度も走っていなかった

`00_settings.py` の `engine_cleanup()` は `~/.cache/vllm` / `~/.triton` を
`warm-cache.tar` にまとめる実装を**既に持っていた**。しかし呼び出し元は
キープアライブセルの後片付けで、v14 の成果物には tar が無く、ログにも
キャッシュの行が 1 本も無い（あるのは `01` の
`ℹ warm cache なし ... exists=False` だけ）。つまり**復元側だけがあり、
退避側は一度も実行されていない**。

→ 退避を「キープアライブ終了時」から**起動ループの合流点**（成功・失敗の両方が
通る場所）へ移した。373 秒を払った回のカーネルは失敗してもディスクに残って
いるので、持ち出す価値がある。

### 16.2 当たったかを測る（「ファイルがある」では判定しない）

Triton のキャッシュキーは Triton の版・対象 sm・ソースハッシュを含む。**ずれれば
黙ってミスし、373 秒がそのまま戻る**。この状態を「キャッシュを入れた」と
報告すると、v14 の先読みと同じ失敗（効かない施策を効いたことにする）を繰り返す。

そこで判定を **エントリの増分**にした（`~/.triton/cache` は 1 カーネル 1 ディレクトリ）。

| フィールド | 意味 |
|---|---|
| `triton_cache_before.entries` | 起動前のエントリ数（復元が効いたかの証拠） |
| `triton_cache.entries_after` | 起動後 |
| `triton_cache.new_entries` | 今回作られた数。**0 に近いほど当たっている** |
| `triton_cache.reused_pct` | 今回使ったエントリのうち前回からあった割合 |

`new_entries` が `entries_before` の 20% を超えたら「当たっていない」と警告する。
併せて `TRITON_CACHE_DIR` を起動プロセスへ明示した（既定は Triton の版によって
`~/.triton/cache` と `~/.cache/triton` のどちらにもなり得て、復元先と食い違うと
1 件も当たらない）。

### 16.3 手順（2 往復）

1. **v15**: 起動して `warm-cache.tar` を成果物に出す（この回は `entries_before=0`
   なので全部が新規。`new_entries` がそのまま「キャッシュの大きさ」になる）
2. tar を Kaggle Dataset `vllm-warm-cache` として上げ、`dataset_sources` に足す
3. **v16**: 復元して起動。`new_entries` と `startup_gaps` の 373 秒が消えたかを見る

1 は smoke 変種で足りる（起動＋1 回の会話で終了するので、30 分のキープアライブを
回さずに済む）。

### 16.4 「Triton キャッシュで 31 t/s が戻る」は**間違い**（v14 の 1 発目を読む）

キャッシュで 373 秒が消えれば 1 発目（`mtp+dd`）が `ATTEMPT_TIMEOUT=900` に収まり、
速度 31.0 t/s が戻る——という読みは**成り立たない**。`vllm_1.log` を見ると 1 発目は

```
03:49:15  Loading weights took 705.00 seconds     ← 本体
03:50:21  Loading weights took 64.37 seconds      ← draft（dd）
03:50:23  Model loading took 8.88 GiB memory and 777.714009 seconds
03:50:26  Using LBNHC KV cache layout
03:50:29  [backbone] Starting torch.compile       ← ここで 921 秒の打ち切り
```

で、**373 秒の無音（profile run）にすら到達していない**。921 秒のうち 778 秒が
重みロード。Triton キャッシュが効いても 1 発目は 900 秒に収まらない。

（2 発目の `Loading weights took 6.39 seconds` は、1 発目が読んだぶんが page cache に
残っているため。つまり**1 発目が冷たい読みのコストを全部払い、2 発目がただ乗りする**
構造で、これは先読みでは直らない——先読みの 18.21 GiB はワーカーの anonymous 確保
（8.88 GiB × 2）に押し出される。§15.1）

→ Triton キャッシュの効果は「**profile run に到達した試行**の 373 秒」に限られる。
v14 でそれが該当するのは 2 発目（eager, `load_seconds` 486.1s）で、見込みは
486 → 約 113 秒。**速度（31.0 → 4.5 t/s）は別問題**で、こちらは 1 発目の重みロード
778 秒を何とかしないと動かない（`ATTEMPT_TIMEOUT` を上げるだけでは、起動に
1,200 秒以上かかる段を毎回試すことになる）。

## 17. 基準を out2g に戻す（2026-11-01・方針転換）

「最初の方が起動も速いし速度も速いのでは」という指摘を受けて突き合わせた。
**その通りだった。** v14 は「改善」ではなく退行で、しかも私がその後の議論を
v14 の数字（起動 486 s）を基準に進めていたのが誤りだった。

| | **out2g（10-31）** | **v14（11-01）** |
|---|---|---|
| 総時間（`elapsed_seconds`） | **928.7 s** | **1,876.7 s** |
| install | 242.7 s | 227.7 s |
| 先読み | なし | 152.5 s |
| 試行 1（`mtp+dd`, compile on） | **成功** | **921 s で打ち切り** |
| └ 重みロード | **約 181 s** | **705.00 s** |
| 試行 2 | — | `eager` 486.1 s |
| 起動 | 642.4 s | 486.1 s（**見かけだけ短い**） |
| 生成速度 | **31.7 t/s** | **4.5 t/s** |

### 17.1 重みロードが 181 → 705 s になった理由（第一候補＝先読み）

v14 のログに、機械自身の実力が残っている:

```
259.4  🧮 重み 18.21 GiB / prefetch に必要な空き 21754 MiB
259.4  🔥 重みを先に読む（18.21 GiB / 1 本 / 予算 300s）…
411.9    122.2 MiB/s / 152.5s / 1 本
```

**先読み自身が 122.2 MiB/s を出している。** なら 18.21 GiB は約 150 秒。out2g の
181 秒はこれで説明がつく。同じ重みを試行 1 が 705 秒かけたのは **26 MiB/s**、
機械の実力の 4.7 倍遅い。読み出し中に何かが競合している。

機構（推定）: 先読みが page cache を埋める → ワーカーが 8.88 GiB × 2 を
anonymous で確保 → カーネルが page cache を 17.8 GiB 追い出す（＝先読みした
重みが消える）→ 試行は cold read に戻り、しかも reclaim と競合する。
`ram_trace.txt` の `Cached` が 20〜23 GiB で推移していたのと矛盾しない。

> ⚠ **この機構は §18.1 で否定された。** v15 の trace では `MemAvailable` が
> 23.5 GiB を下回らず、`cache` も崩れていない。追い出しは起きていない。
> 4.4 倍の遅化は事実として残るが、理由は未解明（候補 2 つは §18.1）。

→ **先読みの既定を OFF にした**（`KC_PREWARM=0` が既定。`=1` で A/B 可）。
これは最悪ケースでも 152.5 秒が確実に浮くので、機構が外れていても損をしない。

### 17.2 実験は playground で行い、効果があったものだけ本番へ

これまで実験は `--smoke` で**本番カーネルへ push** していた。smoke 変種は同じ
kernel id なので、実験のたびに本番のバージョン履歴が埋まる（v15 もそう）。

→ `vllm-playground`（`kaggle/build_vllm_playground.py`）を作った。**セルは本番と
同一**で、骨組み（トンネル / proxy / keepalive）だけ外す。ノブは `KC_EXPERIMENT`
が選ぶ環境変数の初期値だけで、セルの中身は変えない。だから当たったノブは
本番の既定を書き換えるだけで移せる。

| `KC_EXPERIMENT` | `KC_PREWARM` | `KC_WARM_CACHE` | その他 | 目的 |
|---|---|---|---|---|
| `baseline` | 0 | 0 | — | **out2g の再現**（181 s / 31.7 t/s が出るか） |
| `prewarm_on` | 1 | 0 | — | 先読みの対照（v14 の再現） |
| `cache` | 0 | **`require`** | — | キャッシュの効果（Dataset が無ければ**落とす**。`1` だと空振りして baseline と同じ数字が出る） |
| `force_first` | 0 | 0 | `VLLM_TRITON_FORCE_FIRST_CONFIG=1` | 373 秒が autotune かの判定 |
| `baseline_long` | 0 | 0 | `VLLM_STARTUP_BUDGET=7000` / `VLLM_ATTEMPT_TIMEOUT=7000` | 予算だけ上げる（§21。予算が「キャッシュが効くか」の交差因子になっていたため） |

どれも**1 変数だけ**動かす。2 つ同時だと、効いたときにどちらが効いたか分からず、
v14 の先読みと同じ轍を踏む。各実験は `KC_PREWARM` と `KC_WARM_CACHE` を
**明示**する（既定に頼ると、既定を変えた瞬間に対照群が壊れる）。

## 18. v15 の結果（2026-11-01）— 先読みの退行が**2 例目**で再現

v15 は既定反転の**前**に push したので、中身は v14 と同じ「先読み ON」構成。
つまりこれは v14 の追試であり、結果は**再現した**（そして悪化した）。

```
 256.7  🧠 RAM 空き 30333 / 空 2994 / cache 25607 / shm 1 MiB（全 32098）
 256.7  🔥 重みを先に読む（18.21 GiB / 1 本 / 予算 300s）…
 430.9  🧠 Triton キャッシュ: 0 エントリ（なし / 環境変数 /root/.triton/cache）
 430.9  --- 試行 1/9: spec=mtp+dd ... ---
1331.7  --- 試行 2/9: spec=eager ... ---          ← 試行 1 は 900.8s で打ち切り
2192.6  Loading weights took 745.32 seconds        ← 8.48 GiB に対して
2232.7  Model loading took 8.48 GiB memory and 752.895002 seconds
2252.7  ⏱ 起動予算 1800s 超過（1821s） → 残り 7 試行を打ち切り
```

| | **out2g（先読み OFF）** | **v14（ON）** | **v15（ON）** |
|---|---|---|---|
| 先読み | なし | 152.5 s | 174.2 s（107.0 MiB/s） |
| 重みロード | **約 181 s** | **705.00 s** | **745.32 s** |
| 試行 1 | 成功（642.4 s 起動） | 921 s 打ち切り | 900.8 s 打ち切り |
| 速度 | **31.7 t/s** | 4.5 t/s | 起動せず（ERROR） |
| 総時間 | 928.7 s | 1,876.7 s | **2,256.2 s** |

### 18.1 §17.1 の機構（page cache 追い出し）は**支持されない**——訂正

v15 の `ram_trace.txt` を読むと、私が §17.1 に書いた筋書きは成り立っていない。

```
 t(s)    avail   free  cache
  430   25028    495  23208     ← 先読み直後
 1330   23641    482  22023     ← 試行 1 の終わり
```

- `avail`（MemAvailable）は**一度も 23.5 GiB を下回らない**。OOM 圧力は無い。
- `cache` も 22〜23.6 GiB で**崩れていない**。「先読みした重みが追い出された」形跡が無い。

（この trace は 1820 秒で止まる。予算打ち切りで監視プロセスも死ぬため、**745 秒の
読み込み区間そのものは写っていない**。だから「追い出された」も「されていない」も
trace だけでは言えない。言えるのは「試行 1 の 900 秒の間、メモリは余っていた」まで。）

数字のほうが雄弁。1 ワーカーあたり 8.48 GiB / 745 s = **11.6 MiB/s**、2 ワーカー合計で
23 MiB/s。out2g は 18.21 GiB / 181 s = **103 MiB/s**（合計）。先読み自身は 107 MiB/s。
つまり **out2g は素の読み出し速度しか出しておらず、page cache の恩恵を最初から
受けていない**。それでも先読みありより 4.4 倍速い。

→ 先読みは「効かなかった」のではなく **「効かないうえに 4 倍遅くする」**。機構は未解明で、
候補は 2 つ:

1. 18 GiB の cache を抱えたままワーカーが anonymous を確保すると、触ったページごとに
   同期的な reclaim が挟まり、読み出しが待たされる
2. `/kaggle/input` を 2 回続けて読むこと自体が絞られている（2 回目のスロットリング）

どちらかは playground の `baseline` vs `prewarm_on` の A/B で切り分けられる。
**切り分けが済むまで既定は OFF**（`KC_PREWARM=0`）。機構が外れていても 174 秒は浮く。

### 18.2 v15 が残した `warm-cache.tar` は**不完全**（使ってはいけない）

`_capture_warm_cache()` は成功・失敗の合流点で走るので、**失敗した回でも tar を作る**。
v15 の tar は 23.9 MB / 30 エントリ / カーネル 4 種:

```
_fused_post_conv_kernel  _prepare_rope_positions_kernel
chunk_local_cumsum_scalar_kernel  chunk_scaled_dot_kkt_fwd_kernel
```

これは**重みロード中に落ちるまでに compiles されたぶんだけ**。373 秒の主役である
`chunk_o` / `chunk_delta_h` は入っていない（どの試行も profile run に到達していない）。
これを Dataset にして `cache` 実験を回すと「キャッシュは効かない」という**偽陰性**が出る。

→ 復元用の tar は**起動に成功した回**から採る。`baseline` は out2g と同じ経路を通るので、
そこで採れた tar が完全版になる。

### 18.3 `vllm-warm-cache` は 2026-11-01 20:44 に作った（それまでは存在しなかった）

`datasets/warm-cache/dataset-metadata.json` はローカルにあったが、Kaggle 側には
**一度も作っていなかった**。`--no-run`（Quick Save）で playground を保存したときに出た警告:

```
The following are not valid dataset sources and could not be added to the kernel:
['yamahei21python/vllm-warm-cache']
```

`kaggle datasets list --user yamahei21python` にも出てこない。つまり
`kaggle/build_vllm_playground.py` の `dataset_sources` は**宙に浮いた参照**を持っている。

これは危険。`KC_WARM_CACHE` の既定 `1` は「あれば復元、無ければ『なし』と出して続行」
なので、`cache` 実験が **baseline とまったく同じ数字**を出し、「キャッシュは効かない」
という偽陰性になる。1 往復（約 40 分）を使って間違った結論に到達する。

→ **3 値にした**（`kaggle/step2-vllm/cells/01_install_vllm.py`）:

| 値 | 挙動 | 用途 |
|---|---|---|
| `0` | 復元しない | 対照群（baseline） |
| `1` | あれば復元、無ければ続行 | 本番の既定 |
| `require` | **無ければその場で `RuntimeError`** | 実験（cache） |

`require` のときのメッセージには「このまま走らせると baseline と同じ結果になり、
偽陰性になるので中断する」と理由を書いてある。次に見た人が「なぜ落ちたか」で
悩まないようにするため。

順序も重要: `elif _WC_REQUIRED:` は「warm cache なし」を表示する `else:` より
**前**に置く（後ろだと到達しない）。テストで固定済み。

**完全版 tar が採れたら、次はこの順**:

1. `datasets/warm-cache/` に tar を置く（`dataset-metadata.json` はコミット済み）
2. `kaggle datasets create -p datasets/warm-cache` で Dataset 化
3. `--no-run` で playground を保存し直し、警告が消えたことを確認
4. `KC_EXPERIMENT=cache` で実行（`require` なので、Dataset が無ければ即落ちる）

#### 実行結果（2026-11-01 20:44）

v4 の 195 MB tar で 1〜2 を実行した。**ここで 1 つ想定外が出た**:

```
$ .venv-kaggle-cli/bin/kaggle datasets files yamahei21python/vllm-warm-cache
root/.triton/cache/26FVPAU7.../chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin
```

**Kaggle は上げた `.tar` を展開して保持する。** アップロードは成功したのに、
マウント側に `*.tar` は 1 本も無く、`root/.triton/cache/<hash>/...` の木になっている
（tar の arcname が `root/...` だったので `root/` が 1 段挟まる）。

このままだと `KC_WARM_CACHE=require` が「tar が 1 本も無い」で**正当な Dataset を
落とす**。復元側を tar と展開済みの木の両方を受けるように直した
（`_kc_wc_tree` が `.triton/cache` を探し、`copytree` でローカルへ複製する）。
`timings.warm_cache_restore.source` にどちらから復元したかが残る。
`notes/pitfall-coverage.md` #34。

#### 実行結果（2026-11-01 21:21 の `KC_EXPERIMENT=cache`）— **復元は成功、実験は不成立**

復元側は直った。**が、キャッシュの効果は測れていない。** 理由は復元ではなく
重みロードで、1 回目の試行が予算を食い切った。

```
warm_cache_restore: {mode: "require", restored: true,
                     source: "/kaggle/input/vllm-warm-cache/root/.triton/cache",
                     mib: 191.1, entries: 236, seconds: 189.9, missing: []}
triton_cache_before: {entries: 236, dir: "/root/.triton/cache"}
install_seconds: 456.2
⏱ 起動予算 4200s 超過（4227s） → 残り 8 試行を打ち切り
```

| 事実 | 数字 |
|---|---|
| 復元（191 MiB / 236 エントリ） | **189.9 秒** |
| 試行 1 の開始 | t=489.7s |
| 予算切れ | t=4,227s（**試行 1 本で使い切った**） |
| そのとき重みロードは | 12:31:13（worker 起動）→ 13:39:28 で **layers.34** に到達したところ |
| 出力 | `FAILED.txt` / `vllm_1.log` / `ram_trace.txt` は残った（`result.json` は `stage: "env"`） |

つまり **コンパイル / autotune の段に一度も到達していない**。Triton キャッシュが
効くのはその段なので、この回は「効いた / 効かない」のどちらでもない（**判定不能**）。

読み取れること 2 つ:

1. **復元コストは 190 秒**。キャッシュは 190 秒以上を削って初めて元が取れる。
   §12 の 928.7s 構成では autotune が 373 秒だったので、**差し引きで得**になる
   可能性はあるが、まだ 1 度も観測していない。
2. **重みロードが 4,000 秒でも終わらないことがある**。§20 の 2,640 秒、v4 の
   1,304 秒に続く 3 例目で、しかも最悪。ログは
   `Attempted to load weight ... bfloat16 into parameter with dtype torch.float16`
   を層ごとに数本ずつ、6〜8 秒間隔で吐いていた（同じ層の別テンソルごとに 1 行）。
   **bf16 の重みを fp16 のパラメータへ入れる変換が 1 テンソルずつ効いている**
   ように見える。v4 は同じロードを 1,304 秒で終えているので、同じコードで
   ここまで差が出る理由はまだ説明できていない。

次に回すなら、`cache` をもう一度ではなく **予算を 4,200 → 7,000 秒に上げた
`baseline`** を 1 回。そうしないと「重みで死ぬ回」と「キャッシュの効果」が
どちらも測れない。GPU を 1 往復使うので、やるかどうかは別途判断。





### 18.4 `VLLM_TRITON_FORCE_FIRST_CONFIG` は**診断専用**（CPU で中身を確認した）

`force_first` 実験の前提を、GPU を使う前に pinned wheel で確認した
（`.wheel/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl`）。結論は
**「効くが、本番に使ってはいけない」**。

**入っている**: `vllm/env_override.py`（37,970 B）が
`VLLM_TRITON_FORCE_FIRST_CONFIG` を読み、`vllm/triton_utils/force_first_config.py`
（3,301 B）の `install()` を呼ぶ。両方とも pinned wheel の中にある。

**何をするか**（`Autotuner.run` を丸ごと差し替える）:

1. `configs` を先頭から順に試し、**最初に成功した config** を採用する
2. 失敗と見なすのは `OutOfResources` / `CompileTimeAssertionFailure` / `PTXASError`
   だけ（＝「T4 で動かない config」を飛ばす）
3. `self.best_config` にそれを入れ、`_picked_cache` に覚える
4. **`do_bench` を 1 回も呼ばない**（ファイル内の出現数 0）

→ 36 config を**全部コンパイルして計測する**代わりに、**1 個コンパイルして終わる**。
373 秒が autotune なら、ここで大きく落ちる。だから**判定には使える**。

**使ってはいけない理由**: 選ばれるのは「最速」ではなく「**最初に通った**」config。
`picked_index` は通常 0 なので、性能は保証されない。起動は速くなるが t/s は落ちうる。
`force_first` で起動が縮んだなら、正しい処方は**キャッシュ**（autotune の結果を
そのまま持ち越す）であって、このフラグを本番に置くことではない。

**ログで数えられる**: 採用時に 1 行出る。

```
[triton-autotune-disabled] kernel=chunk_o configs=36 picked_index=0 picked=...
```

`configs=` の合計が「autotune が何個の候補を持っていたか」＝ 373 秒の上限の目安。
`force_first` の回でこの行を数えれば、**キャッシュで何個潰せるか**が分かる。
`grep -c triton-autotune-disabled` と `grep -o 'configs=[0-9]*'` で足りる。

## 19. auto-prefetch は GCSFUSE では**常に切れる**（2026-11-01・§14.2 の訂正）

`baseline` が 2 試行とも重みロードで切られた回（v3）のログを追って、固定 wheel
（`.wheel/vllm-0.32.0-...whl`）の `weight_utils.py` を読んだ。分岐はこうなっている。

```python
fs_type  = _get_fs_type(sorted_files)
is_net_fs = fs_type in ("nfs", "nfs4", "lustre")
total_bytes = _get_checkpoints_size_bytes(sorted_files)
avail_bytes = _get_available_ram_bytes()
fits_in_ram = total_bytes <= 0.9 * avail_bytes

should_prefetch = safetensors_load_strategy == "prefetch"
if safetensors_load_strategy is None:
    if is_net_fs and fits_in_ram:            should_prefetch = True
    elif is_net_fs and not fits_in_ram:      warn(...)      # 切る
    elif not is_net_fs and fits_in_ram:      info(...)      # 切る
    elif not is_net_fs and not fits_in_ram:  info(...)      # 切る
```

**GCSFUSE は `nfs` / `nfs4` / `lustre` のどれでもない。** つまり `is_net_fs=False`
で、4 分岐のうち下 2 つに入る。**どちらでも prefetch は切れる。**
空き RAM は「どちらの文言が出るか」を選ぶだけで、可否を決めていない。

こちらがずっと見ていた文言は下の 2 つ目（`not is_net_fs and not fits_in_ram`）で、
「RAM が足りないから切られた」と読めてしまう形だったのが誤読の元。

### `Available RAM: 5.85 GiB` の正体

`ram_trace.txt` は同じ回に `avail=30,131 … 最小 22,767 MiB` と言っている。
vLLM の数字と 4 倍違うのは、vLLM が**cgroup を見ている**から:

```python
def _get_available_ram_bytes():
    host_available = psutil.virtual_memory().available
    cgroup_limit = get_cgroup_memory_limit()
    if cgroup_limit is None:
        return host_available
    cgroup_usage = get_cgroup_memory_usage()
    return min(host_available, cgroup_limit - cgroup_usage)
```

`cgroup_usage` は **page cache を含む**。checkpoint を 17 GiB 読むと、その分だけ
「空き」が減る。つまり **読むほどに数字が悪くなる**自己言及的な指標で、
ホストの `MemAvailable` とは別物。`ram_trace` と並べるときは混同しないこと。

### 未検証のノブ: `--safetensors-load-strategy=prefetch`

明示指定すれば `should_prefetch=True` になり、RAM が足りなくても**警告だけ出して
先読みする**（`elif should_prefetch and not fits_in_ram:` の枝）。このリポジトリは
このフラグを**一度も渡していない**（`grep` で 0 件）。したがって:

- これまでの全実行で prefetch は無効。v10 の 133 秒を「prefetch のおかげ」と
  説明することはできない
- 逆に言えば、**試していないノブが 1 つ残っている**

ただし期待しすぎないこと。自前の先読み（`KC_PREWARM`）は**4.4 倍遅く**なった
（§18）。機構は違う（vLLM のは読みながら投げる、自前のは先に全部読む）が、
「GCSFUSE に投げると遅くなる」という同じ土台の話なので、測るまでは分からない。

実験にするときは `baseline` と**混ぜない**こと。重みロードの速度が変わると
373 秒の profile run の測定条件も変わる。

## 20. v3（`baseline` / 2026-11-01 17:34）— 予算 2700 でも足りず、**重みだけで 2,640 秒**

`KC_WARM_CACHE=0`（対照）で回した回。結論から書くと、**切られたのは予算の
設計ミスで、起動失敗ではない**。そして今回初めて「重みロードの実時間」が
数字で出た。

### 時系列（ログの `time` と vLLM 側の時刻を突き合わせたもの）

| t（秒） | 出来事 |
|---|---|
| 0 | 開始。`avail=30,366 free=3,072 cache=25,561 MiB` |
| 222 | install 完了（`pip` 181 / `torchaudio` 191 / `import` 222） |
| 250 | 試行 1/9 開始（`spec=mtp+dd max_len=65536`、staged=13 files） |
| 310 | 最初のワーカー警告（08:39:22） |
| 971 | `embed_tokens` を読んでいる（08:49:32） |
| 1,451 | `layers.0` に到達（08:58:24） |
| 2,890 | **`layers.61` の最後のテンソル**（09:23:21）＝ 全 62 層ぶんが終わった |
| 2,972 | `⏱ 起動予算 2700s 超過（2722s）→ 残り 8 試行を打ち切り` |
| 2,976 | `RuntimeError: vLLM の起動に失敗しました`（`stage="env"`） |

**重みロード = 約 2,640 秒**（t=250 → 2,890）。まだ profile run の 373 秒が
残っているので、**3,013 秒が下限**。2,700 では 300 秒足りなかった。
`warm-cache.tar` は 71,680 B（中身は空同然）で、v2 と同じ。

### 実効 I/O は 11.8 MiB/s

checkpoint は 15.5 GiB。TP=2 の 2 ワーカーが**それぞれ全シャードを開く**ので
実 I/O は約 31 GB。31 GB ÷ 2,640 秒 ≒ **11.8 MiB/s**。§19 の「20 MiB/s」より
さらに遅い。速い回（out2g: 103 MiB/s）とは **9 倍**の開きがある。

### 空きメモリが 500 MiB を割ったまま読み続けている

`ram_trace.txt` の後半:

```
2685.5s avail=25071 free=436 cache=23450 shm=29 top=VLLM::Worker_TP:12050 VLLM::Worker_TP:12044 vllm:2124
2710.5s avail=28956 free=4331 cache=23420 shm=1  top=vllm:2122 ...     ← ワーカー死亡
```

- ワーカー RSS は **12.1 GB × 2 = 24 GB**（checkpoint を両方が抱えている）
- page cache 23.4 GB、`free` は **437 MiB**
- `avail` は 25 GB あるので OOM ではない。ただし `free` がほぼ 0 ということは
  **毎回のページ確保で reclaim が走っている**状態で、GCSFUSE の readahead が
  積み上がりにくい。読み直しが起きている疑いが濃い（未確定）

これは §19 の `--safetensors-load-strategy=prefetch` の**優先度を下げる**材料に
なる。prefetch は「RAM に余裕があるなら先読みする」機構で、いま空きは 437 MiB。
明示指定すれば動くが、**より多くのページを奪い合うだけ**になる公算が大きい。
測るなら単独の実験として、`baseline` とは混ぜない（§19 の注意書きどおり）。

### この回から言えること（キャッシュ実験の線への影響）

起動時間の内訳が **重み 2,640 秒 : profile run 373 秒 = 88% : 12%** になった。
キャッシュ（warm-cache）が狙っているのはこの 12% で、**仮に完全に効いても
起動は 3,013 → 2,640 秒にしかならない**。§9 の見積りは「ロード 133 秒」を
前提にしていたので、その前提が崩れている。

先に効かせるべきは重みロードの側。ただし候補は **`vllm-io-bench`（CPU・GPU 0 秒）で
すでに 2 つ潰してある**（`notes/cpu-test-results.md`）。ここで推測を書くと
v13 の 135 分を溶かしたのと同じ間違いになるので、実測だけを並べる:

| 候補 | 実測 | 判定 |
|---|---|---|
| 並列で読む（2/4/8/16 本） | **2.2〜2.3 MiB/s**（1 本は 110〜130 MiB/s） | **13〜50 倍遅い**。v13 はこれで 135 分溶かした |
| ローカルへ実コピー | コピー自体が同じ 43.8 MiB/s で 407 秒。置き場所も無い（`/kaggle/working` 20 GB に 18.7 GB） | 元が取れない |
| `drop_caches` で空きを戻す | `OSError: Read-only file system` | 不可 |
| mmap（cache 済） | **5,402 MiB/s** | cache に載れば事実上ゼロ |

つまり**「速くする」余地は 1 本読みの 110 MiB/s を使うことだけ**で、
`TP=2` は 2 本同時読み＝いちばん遅い形を強制している。ここが本命。

> ⚠ 上の表を書いた直後に「並列コピーで速くなる」と書いていた。**それは
> v13 で実測済みの失敗**（8 本並列で 2.3 MiB/s）。`KC_PREWARM` が 4.4 倍
> 遅かったのも、8 本で投げていたから（§15）。**1 本にすれば 110 MiB/s の
> 側に戻る**——ここが唯一まだ試していない形で、`KC_PREWARM` の
> スレッド数を 1 にした測定が次の候補。
>
> ただし楽観はできない。prewarm で 15.5 GiB を cache に載せても、そのあと
> ワーカーが 24 GB の RSS を確保するので cache は押し出される（v3 の
> `free=437 MiB`）。**部分的にしか当たらない可能性が高い**。測るまで分からない。

## 21. `baseline_long`（2026-11-02 00:50）— 予算だけ 7,000 秒に上げた回

§18.3 の `cache` は**復元に成功して落ちた**。`warm_cache_restore` は
`restored=true / entries=236 / 191.1 MiB / 190 秒 / source=.../root/.triton/cache`
だったのに、試行 1 が重みロードだけで 4,200 秒を使い切り（12:31:13 → 13:39:28、
`layers.34` に到達しただけ）、**コンパイル段に一度も届かなかった**。
つまり `cache` の結果は「効かない」ではなく「未測定」で、予算が
「キャッシュが効くか」の交差因子になっている。

そこで**構成は `baseline` と同一**のまま、起動予算だけ上げた回を 1 回だけ回す。

| 項目 | 値 |
|---|---|
| 実験名 | `baseline_long` |
| 構成 | `KC_PREWARM=0` / `KC_WARM_CACHE=0`（`baseline` と同じ） |
| 予算 | `VLLM_STARTUP_BUDGET=7000` / `VLLM_ATTEMPT_TIMEOUT=7000` |
| 投入 | version 8（2026-11-02 00:50 JST 開始） |
| 消費見込み | 最大 約 2.1 時間（残クォータ 9h56m の 21%） |

**2 つの env を動かすのは 1 ノブの裏表**。試行の上限は
`min(ATTEMPT_TIMEOUT, 残予算)` なので、総予算だけ上げて `ATTEMPT_TIMEOUT` を
4,200 に残すと、試行 1 は**まったく同じ秒数**で切られる（`03_vllm_serve.py` の
`_tmo`）。片方だけ上げる実験は無意味。

### 読み方（終わったら見る場所）

`timings.json` の `startup_attempts`（各試行の `load_seconds` / `profile_seconds`）
と `result.json` の `ok` / `elapsed`。分岐は 2 つだけ:

| 結果 | 意味 | 次 |
|---|---|---|
| 重みロードが終わり、profile run（373 秒）まで走った | 予算が唯一の壁だった | `cache` を同じ予算で回す（キャッシュの効果が初めて測れる） |
| 7,000 秒でも重みロードが終わらない | 壁は予算ではなく**ノードの I/O**（実効 11.8 MiB/s） | 予算を上げても無駄。§20 の 1 本読み（`KC_PREWARM` のスレッド 1 本化）へ移る |

### 投入時に踏んだ罠

`cli push` は push 直前にビルダーを再実行する。`KC_EXPERIMENT` を**push の
ときにも**渡さないと既定の `baseline`（4,200 秒）で上書きされる。実際に
version 7 はそれで `baseline` になった（`QUEUED` のうちに version 8 で
上書きしたので GPU は消費していない）。`cli/push.py` に指紋比較の警告を入れ、
`notes/pitfall-coverage.md` #35 に固定した。

## 22. 未検証のノブが wheel に実在した（2026-11-02 01:25・GPU 0 秒）

GPU を使わずに、固定 wheel（`.wheel/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl`）
の中を読んで確認した。§19 の「auto-prefetch は GCSFUSE では常に切れる」は
**既定の話**で、明示指定する口が別にあった。

### 実在する CLI フラグ（`vllm/engine/arg_utils.py`）

| フラグ | 既定 | 意味 |
|---|---|---|
| `--safetensors-load-strategy` | `None` | `lazy` / `eager` / `prefetch` / `torchao` |
| `--safetensors-prefetch-num-threads` | **8** | prefetch のスレッド数 |
| `--safetensors-prefetch-block-size` | 16 MiB | 1 回の read サイズ |

既定 `None` は「mmap（lazy）＋ **NFS を検出してチェックポイントが RAM の 90% に
収まるときだけ**自動 prefetch」。Kaggle は §19 のとおり収まらないので
自動では効かない＝いまは純粋な lazy mmap。

### `prefetch` が TP の二重読みを消せる理由

`vllm/model_executor/model_loader/weight_utils.py`:

```python
paths_to_prefetch = sorted_files[rank::world_size]
```

prefetch は **rank ごとにファイルを分担**する。TP=2 でも総読みは
**チェックポイント 1 回分（15.5 GiB）**で、いまの「2 ワーカーが各自 mmap して
31 GB 読む」（§20 の実効 11.8 MiB/s）を半分にできる計算になる。

### ただし既定の 8 スレッドは**実測で一番遅い形**

§20 の `vllm-io-bench`（CPU・GPU 0 秒）は 8 本並列で **2.2〜2.3 MiB/s**、
1 本で **110〜130 MiB/s** だった。既定値はちょうど悪い側なので、
`--safetensors-prefetch-num-threads 1` が唯一まだ試していない形。

### 楽観できない点

- prefetch は**バックグラウンドスレッド**で走る（`_prefetch_all_checkpoints`）。
  ワーカーの RSS が 24 GB まで育つ過程と競合するので、§20 の
  `free=437 MiB` と同じ追い出しに遭う可能性がある
- `KC_PREWARM`（§15）は 8 スレッドで 152.5 秒の**純コスト**だった。同じ形を
  vLLM の内蔵 prefetch でやっても同じ結末になりうる。差は「rank で分担する」
  ことと「ワーカーが読み始める前に走る」ことの 2 点だけ

### 次の実験（予算の結論が出てから）

`prefetch1` を 1 行足す（`KC_PREWARM=0` / `KC_WARM_CACHE=0` のまま、
`safetensors-load-strategy=prefetch` と `num-threads=1` の 2 つだけ）。
**`03_vllm_serve.py` の PLAN にフラグを足す必要がある**ので、セル変更になる
（＝playground の「env だけ変える」原則から外れる。だから今は実装しない）。

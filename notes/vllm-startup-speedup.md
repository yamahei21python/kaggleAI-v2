# vLLM 起動時間の短縮（調査メモ / 未実装）

対象: `kaggle/build_step2_vllm.py` の GPU 実行 1 回あたりの起動時間。
方針: **GPU クォータを食う工程を削る**。CPU で測れるものは CPU で測る。

> 運用制約: **GPU kernel の push はユーザーの明示的な Go があるまで禁止**（CPU で潰せる検証を先に全部やる）。

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

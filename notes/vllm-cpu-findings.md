# vLLM 0.32 / T4 x2 で分かったこと（CPU 検証のみ・実測）

対象: `kaggle/build_step2_vllm.py`（GPU 本番）と CPU プローブ群。
GPU クォータは残り僅か + 「Go 待ち」のため、**GPU を使わずに潰せる検証を全部やる**方針。

出典カーネル: `step2-vllm-dryrun` / `uvbench` / `mtpprobe` / `wheelinspect` / `wheelset` / `mtpshape` / `chattpl` / `proxymatrix`

---

## 1. install の実測（施策 A/B の判断）

| 測定 | 値 | 出典 |
|---|---|---|
| `pip install vllm==0.32.0` cold（GPU kernel） | 457s | v1 |
| `pip install vllm==0.32.0` cold（CPU kernel） | 276〜306s | dryrun v2 / uvbench |
| `pip download vllm --no-deps` | 13.4s / 326 MB | uvbench |
| `pip download vllm`（依存込み） | **137s / 3.94 GB / 198 files** | uvbench |
| `uv pip install --system`（依存済み・vllm のみ） | **6.5s** | uvbench |

**結論**
- 転送が支配的（3.94 GB、~29 MB/s）。install 305s のうち ~137s は純粋な取得。
- A（uv）と B（wheel Dataset）は**同じ工程を削るので加算不可**。
- 本命は B。`--no-index --find-links` で転送と解決が消える。見込み 60〜120s（仮説、wheelset で実測中）。
- uv は「B のインストーラ」として使うのが正しい形。

---

## 2. torchaudio 不整合（GPU で実際に死んだ原因）

- 症状: `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`（torch 2.13.0+cu130 / torchaudio 2.8.3+cu128）
- **検出を誤っていた**: `import vllm.entrypoints.cli.main` は通る。`main()` 内の遅延 import（benchmark.main → transformers → torchaudio）で初めて落ちる。
- 対策: 判定を **`vllm serve --help` 実行**に変更。
- 修復: `pip uninstall -y torchaudio` を先に試す（通信不要）。
- **CPU で実証**: uninstall 後も traceback が `arg_utils` まで到達 = **transformers は torchaudio 無しで import 成功**（ImportError を握る）。
- ただし **help 失敗 = serve 失敗ではない**ので preflight は soft fail（起動は試す）。

## 3. CPU では `vllm serve --help` が必ず失敗する（GPU では起きない）

```
RuntimeError: Failed to infer device type
  vllm/config/device.py:55  (→ arg_utils._compute_kwargs)
```
- GPU 無しでプラットフォーム推定に失敗するだけ。**GPU では通る**。
- したがって CPU でフラグ一覧は取れない → **wheel ソースの静的解析**に切替。

---

## 4. vLLM 0.32 の CLI フラグ（静的列挙・257 個）

| フラグ | 状態 |
|---|---|
| `--speculative-config` | あり |
| `--reasoning-parser` | あり |
| `--kv-cache-dtype` | あり |
| `--enforce-eager` | あり |
| `--max-num-seqs` / `--served-model-name` / `--tensor-parallel-size` | あり |
| `--gpu-memory-utilization` / `--max-model-len` / `--load-format` | あり |
| `--quantization` / `--dtype` | あり |
| `--disable-log-requests` | **廃止**（`--enable-log-requests` に反転） |
| `--enable-auto-tool-choice` / `--tool-call-parser` | **見つからず**（付かない） |
| `--host` / `--port` | 走査漏れ（定義ファイル移設）→ v3 で確定 |

- ラダーは `_has()` が False のフラグを付けないので、廃止フラグで即死しない。
- ただし `HELP_OK=False` のときは楽観側（全部ありとみなす）なので、`--disable-log-requests` のような
  「あると死ぬ」フラグは**付けない**方針にした（実装済み）。

---

## 5. T4（sm75）で Marlin は使える ← 以前の見立てを訂正

```
vllm/model_executor/layers/quantization/utils/marlin_utils.py:66
    if device_capability < 75: return False
```
- sm75 は **サポート対象**（≥75）。「Marlin は sm80+ だから不可」は**誤り**。
- ただし `dtype == bfloat16` は sm90+ 必須（marlin_utils.py:642/703）→ **`--dtype half` は必須**。
- 帰結: Triton/Marlin の JIT・autotune が初回に走る可能性があり、起動短縮メモの C（JIT cache）は
  「効く可能性あり」に格上げ。

## 6. MTP は 0.32 に存在する

- `SpeculativeMethod` に `"mtp"`（`vllm/config/speculative.py:57`）
- `MTPModelTypes` に `qwen3_next_mtp` / `qwen3_5_mtp` / `mtp` 等
- `config/vllm.py:1310` が `("dspark","eagle","eagle3","mtp")` を許可
- `SpeculativeConfig` の主要フィールド: `method` / `model` / `num_speculative_tokens` / `draft_tensor_parallel_size`
- → `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` は構文的に有効。
  MTP 重み（`model-mtp.safetensors`）が target model 内にある前提で拾えるかは `mtpshape` の結果次第。

---

## 7. 検証済み（CPU で再現できた範囲）

| 項目 | 状態 |
|---|---|
| Dataset マウント（15 ファイル・サイズ一致） | OK |
| MTP 有無のステージング（symlink 13/12 files） | OK |
| プロキシ + UI + cloudflared トンネル | OK（dryrun 5/5） |
| `/v1/models` / streaming / OpenAI 互換 | OK（モック上流） |
| `--enforce-sampling`（temp0→0.7, rp<1.05→1.05） | 単発 OK（matrix で網羅中） |
| keepalive 延長（conf mtime で while 継続） | OK（ローカルシミュ） |
| 後片付け（proxy/cloudflared/mock） | OK |

## 8. CPU では検証できない（GPU 必須）

- モデルロード（19 GB）と AWQ デクオンタイズ
- CUDA graph capture / KV cache profile
- MTP ヘッドの実ロードと投機デコードの受理率
- 実 tok/s・TTFT

---

## 9. chat template（Dataset の tokenizer で実描画）

- tokenizer: **Qwen2Tokenizer**（vocab 248,044 / added 33）
- chat_template は 8,952 文字。vision（image/video）マクロを含む多目的テンプレート
- system に **自動で `Reasoning effort is set to xhigh ...` が注入される**（テンプレート変数 `reasoning_effort`）
  - 既定が xhigh だと system トークンが増え、レイテンシに効く。低くしたい場合は
    `chat_template_kwargs` / `reasoning_effort` を渡す必要がある（vLLM 側の受け口は要確認）
- assistant 出力は `<think>\n\n</think>\n\n本文` 形式（`--reasoning-parser qwen3` と整合）
- 特殊トークン: `eos_token=<|im_end|>`(248046)、`eos_token_id=[248046, 248044]`、bos なし
- `generation_config.json`: `repetition_penalty=1.05`, `temperature=1.0`, `top_k=20`, `top_p=0.95`
  → プロキシの矯正値（RP 1.05 / temp 0.7）と方向は一致

## 10. dryrun v5 の失敗は「一過性の公開 URL 断」

```
[1] ui     -> FAIL: URLError: [Errno 101] Network is unreachable
[2] models -> FAIL: URLError: [Errno 101] Network is unreachable
[3] streaming -> PASS / [4] openai-compat -> PASS / [5] enforce-sampling -> PASS
```
- localhost 系は全部通り、cloudflared 経由の GET だけ落ちた。**コードのバグではない**。
- ただし本番では 20 分消費後の誤 fail になるため、dryrun 側にも **3 回リトライ**を実装した
  （本番 `build_step2_vllm.py` には既に `_with_retry` があった）。

## 11. 追加で回している CPU テスト（結果待ち）

| kernel | 見るもの |
|---|---|
| `step2-vllm-wheelset` | wheels 3.94 GB を落として `--no-index` install の所要（施策 B の実測） |
| `step2-vllm-mtpshape` | `model-mtp.safetensors` の全テンソル名/shape（MTP ヘッドの形） |
| `step2-vllm-vramcalc` | config から KV/重みを計算し 32768/16384 が載るか |
| `step2-vllm-flagspec` | 使用フラグの値域（`--quantization awq` 等）と MTP 検証条件 |
| `step2-vllm-chattpl` v2 | トークン数（BatchEncoding 対応）・tools・reasoning_effort |
| `step2-vllm-proxymatrix` | enforce-sampling の全パターン（temp0/rp<1.05/欠落/正常） |

---

## 12. プロキシ矯正の全パターン（proxymatrix, 実測）

`--enforce-sampling` 無し（passthrough）は完全素通し。有りは以下。

| 入力 | 上流に渡る値 | 判定 |
|---|---|---|
| `{}`（欠落） | temp **0.7** / rp **1.05** | 既定を注入（仕様） |
| `{temperature: 0}` | temp **0.7** / rp 1.05 | OK |
| `{temp:0, rp:1.0}` | temp **0.7** / rp **1.05** | OK |
| `{temp:0.2, rp:1.01}` | temp 0.2 / rp **1.05** | temp は保持 |
| `{temp:1.5, rp:1.05}` | temp 1.5 / rp 1.05 | 保持 |
| `{temp:0.7, rp:1.7}` | 保持 | 上限側は触らない |
| `{top_p:0.9, top_k:20}` | temp 0.7 / rp 1.05 / top_p 0.9 / top_k 20 | injection あり |

→ 「temp が 0 以下なら 0.7」「rp が 1.05 未満なら 1.05」「欠落も同じ」で一貫。

## 13. フラグの実体（flagspec, wheel 内の定義箇所）

- `--quantization` は `-q` の別名あり。`QuantizationMethods` に **`awq` / `awq_marlin` / `auto_awq`** が存在
  → `--quantization awq` は有効（marlin へ自動昇格する実装: `"awq": AutoAWQConfig`）
- `--kv-cache-dtype` / `--load-format` / `--dtype` / `--max-model-len` / `--served-model-name` は
  `engine/arg_utils.py` のグループ定義経由（help では個別表示）
- `--enable-log-requests` は `BooleanOptionalAction`（= `--no-enable-log-requests` も可）
- `--speculative-config` は `-sc` の短縮あり。**`--spec-method` という単独フラグも存在**
- `--reasoning-parser` は choices を後段検証（プラグイン考慮）
- `--host` / `--port` は `vllm serve` の定義箇所が見つからない → `VLLM_PLATFORM=cpu` で
  help を通して確定させる（helpsolve プローブ）

---

## 14. **重大**: 量子化形式は AWQ ではない（cfgdeep 実測）

`config.json` の `quantization_config`:

```json
{
  "format": "pack-quantized",
  "config_groups": {"group_0": {
    "format": "pack-quantized",
    "targets": ["Linear"],
    "weights": {"num_bits": 4, "group_size": 128, "strategy": "group",
                "symmetric": false, "zp_dtype": "torch.int8",
                "observer": "memoryless_minmax"}
  }},
  "ignore": ["model.visual.blocks.*", ...]
}
```

- `format: pack-quantized` は **compressed-tensors（llm-compressor）形式**。
  AWQ/GPTQ の `quant_method: "awq"` とは別物。
- したがって **`--quantization awq` 固定は誤り**（ロード時に重み解釈が食い違う）。
- 対策（実装済み）: `--quantization` を**付けず auto 判定**を第一候補にし、
  失敗時のみ `--quantization compressed-tensors` → 最後に `awq` を試す 3 系統に変更。
- 名前の "AWQ" は HF リポジトリ名由来で、中身は AWQModifier + compressed-tensors 保存。

## 15. このモデルは VL（vision）＋ハイブリッド線形注意

- `config.json` に `vision_config` / `image_token_id` / `vision_start_token_id` /
  `vision_end_token_id` / `language_model_only`、`text_config` を持つ **マルチモーダル構成**
- `recipe.yaml` の対象に `layers.N.linear_attn.in_proj_qkv/z` があり、
  一部層が **線形注意（GDN 系）**、他が `self_attn` のハイブリッド
- T4(sm75) で線形注意カーネルが動くかは未確認 → GPU で最初に落ちる候補
- `--language-model-only` に相当する指定（text_config のみで serving）が使える可能性

## 16. chat_template は tokenizer_config に無い

- `tokenizer_config.json` に `chat_template` キーが**存在しない**（None）
- 実体は `chat_template.jinja` / `chat_template_medium.jinja`（8,952 / 8,953 文字）
  - transformers 5.16 はファイルから読んで `tokenizer.chat_template` に載せる（実描画で確認）
  - ただし **vLLM が同じ解決をするかは別**。`tokenizer_config.json` に無い場合、
    vLLM は chat template 無しと判断する可能性 → GPU で `/v1/chat/completions` が
    素の completion 扱いになる恐れ。対策候補: `--chat-template /path/chat_template.jinja`
- `tokenizer_class` は **Qwen2Tokenizer**（Qwen3 系ではない）。`pre_tokenize_regex` 等の独自キーあり
- `model_max_length = 262144`

## 17. `--reasoning-parser qwen3` の妥当性

- plausibility: template が `<think>` を出すので parser 名は qwen3 系で合っている
- ただし vLLM の reasoning parser 登録名は `qwen3` が存在するか未確認（compat で確認中）

---

## 18. モデル正体（archdeep 実測）— Qwen3.5 系ハイブリッド VL

```
architectures : Qwen3_5ForConditionalGeneration   ← vLLM registry に存在（対応）
model_type    : qwen3_5 / text: qwen3_5_text / vision: qwen3_5_vision
layers        : 64
hidden        : 5120, heads 24, kv_heads 4, head_dim 256
layer_types   : linear_attention ×3 → full_attention ×1 の繰り返し（full は 16 層）
mtp_num_hidden_layers : 1
max_position  : 262144
dtype         : bfloat16（T4 は bf16 不可 → --dtype half 必須）
```

- **KV cache/token = 2 × 16層 × 4kv × 256dim × 2B = 64 KiB**
  - 32768 tokens → **2.0 GiB**（TP=2 で 1.0 GiB/GPU）
  - 重み 19.5 GB / 2 = 9.8 GiB/GPU、予算 13.8 GiB → **32768 は余裕で載る**
- MTP 重みは `mtp.*` 形式 15 テンソル（`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`,
  `mtp.pre_fc_norm_embedding/hidden`）。`mtp_num_hidden_layers=1` と整合。
- `language_model_only: False` → vision tower もロードされる（不要なら `--language-model-only`）

## 19. vLLM 0.32 の依存ピン（METADATA 実測）

```
transformers<5.20.0,>=5.16.1     ← イメージは 5.16.1（OK）
torch==2.13.0
torchaudio==2.11.0               ← イメージの 2.8.3+cu128 が不一致の原因
torchvision==0.28.0
tokenizers>=0.23.2 / safetensors>=0.6.2 / numpy
Requires-Python: <3.15,>=3.11
```
- **torchaudio は vLLM の必須依存**。単に消すより `torchaudio==2.11.0` を
  `--no-deps --force-reinstall` で入れる方が正しい（実装をその順に変更）。
- それでも駄目なときだけ uninstall（transformers は ImportError を握る）。

## 20. 直近の実装変更（GPU 未投入のまま反映）

| 変更 | 理由 |
|---|---|
| `--quantization` を付けない（auto）を第一候補 | quantization_config が pack-quantized のため |
| `--quantization compressed-tensors` / `awq` の段を追加 | auto が外した場合の切り分け |
| torchaudio 修復を「2.11.0 で入れ直す → それでも駄目なら削除」 | vLLM の必須依存だから |
| `--chat-template <stage>/chat_template.jinja` を条件付きで付与 | tokenizer_config に chat_template が無い |
| 同一エラー連続でラダー打ち切り | 予算の無駄消費防止 |
| preflight を soft fail 化 | CPU の help 失敗 ≠ serve 失敗 |

## 21. chat template v2（chattpl 再実行）— トークン数と tools

- `tokenize=True` は **BatchEncoding** を返す（`len()` は 2）。`input_ids` を見るのが正しい
- 4 メッセージ + generation prompt で **83→88 tokens**（妥当）
- **tools 対応あり**: `<tools>...</tools>` を system に注入し、呼び出し形式は
  `<tool_call><function=NAME>...` の XML。hermes 形式ではない
  → `--tool-call-parser hermes` は不適（そもそも 0.32 に該当フラグ無し）
- **`reasoning_effort` は描画時に効く**:
  - 既定は **xhigh**（system 先頭に長い指示が入る = トークン増）
  - `reasoning_effort="low"` で「Keep your thinking brief...」に置換（354 chars）
  - 現状の UI/プロキシは未指定 → 常に xhigh。必要なら
    `chat_template_kwargs={"reasoning_effort": "medium"}` を注入する（未実装・要判断）

---

## 22. wheel 静的解析（ローカル 0.32.0 を展開して確定）— 重大な訂正 3 件

wheel: `vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl`（2879 py ファイル）。

### 22.1 **訂正**: `--chat-template` は存在する（dataclass 経由で自動登録）

- `entrypoints/launchers/cli_args.py:86` に `chat_template: str | None = None` があり、
  `FrontendArgs` のフィールドとして `add_argument` が自動生成される。
  → `"_has"` を `add_argument` の grep で判定していたため**見落とした**（誤判定）。
- `validate_parsed_serve_args()` が `validate_chat_template(args.chat_template)` を呼ぶ。
- 同様に `--chat-template-content-format` / `--default-chat-template-kwargs` /
  `--trust-request-chat-template` も実在。
- テンプレート解決自体は renderer 経由（下記）で自動でも動くが、明示指定は無害で確実。
- テンプレートは `vllm/renderers/hf.py:resolve_chat_template()` が 4 段で解決:
  1. リクエスト指定 → 2. AutoProcessor の chat_template（VL なのでこれが効く）
  → 3. AutoTokenizer の chat_template（transformers が `chat_template.jinja` を読む）
  → 4. モデルタイプ別の組み込み fallback
- **対策**: `--chat-template` を付けると未知引数で即死するので**付与を削除**（本番/dryrun 両方）。
  テンプレートは自動解決に任せる。
- 併せて `_has()` を**悲観側**に修正（help が取れないとき任意フラグを付けない）。
  楽観側のままだと torchaudio 不整合で preflight が落ちた瞬間に
  存在しないフラグを足して全滅する。

### 22.2 tool parser は `qwen3_xml`（hermes ではない）

- `vllm/tool_parsers/__init__.py` の登録名に **`qwen3_xml`** / `qwen3_json` / `qwen3_coder` がある。
- このモデルの template は `<tool_call><function=NAME><parameter=k>v</parameter>` の XML 形式
  （chattpl 実測）→ `--tool-call-parser qwen3_xml` が正解。`hermes` は不適。

### 22.3 reasoning parser `qwen3` は実在（lazy 登録）

- `vllm/reasoning/__init__.py:15` に `name="qwen3", module_path="vllm.reasoning.qwen3_engine_reasoning_parser"`。
- 正規表現で `register_module("...")` を探すと 0 件になる（lazy dict 方式のため）。**誤検知だった**。

### 22.4 量子化: WNA16 の min capability = **75**（T4 OK）

- このモデルの quantization_config（4bit int / group 128 / symmetric false / pack-quantized /
  input_quant None / static）は `CompressedTensorsWNA16` に落ちる。
- `get_min_capability() = 75`（コメント "Turing and up"）。
- asymmetric は `WNA16_ZP_SUPPORTED_TYPES_MAP` に 4bit があり **対応**（`uint4` + zero_point）。
- 重みキーは `weight_packed` / `weight_scale` / `weight_shape` / `weight_zero_point` で
  vLLM の `PackedvLLMParameter` 命名と一致（ml 実測: 例 `layers.0.linear_attn.in_proj_qkv.weight_packed`）。
- compressed-tensors は **vLLM の必須依存**（`Requires-Dist: compressed-tensors==0.17.0`）。

### 22.5 GDN（線形注意）は T4 で Triton にフォールバックする

- `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`。
  sm80+ 必須なのは **fused CUDA decode カーネルだけ**（`_fused_gdn_decode_unsupported_reason`）。
  T4 は `has_device_capability(80)` が False → 自動で `gdn_decode_kernel="triton"`。
- 注意: `VLLM_GDN_DECODE_KERNEL=cuda` を **明示すると ValueError で即死**（env があると raise）。
  → env は設定しない。dtype half も fused 条件を外すので Triton 側に落ちる（安全）。
- prefill は `triton`（FlashInfer は sm90 のみ、CuteDSL は sm100 のみ）。

### 22.6 MTP の重みキーは合致する（ml 実測）

- `model-mtp.safetensors`（0.79 GiB / 15 テンソル、BF16）に `mtp.fc.weight`,
  `mtp.layers.0.*`(11), `mtp.norm.weight`, `mtp.pre_fc_norm_{embedding,hidden}.weight`。
- target 側 mapper: `orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None}`
  → target ロード時に `mtp.*` は**捨てる**（`None`）。
- draft 側 `Qwen3_5MTP.load_weights` は `mtp.` → `model.` に置換。
- index.json は無い（単一シャード）→ ローダは dir 内 `*.safetensors` を両方読む。
- `mtp_num_hidden_layers=1` と `Qwen3_5MultiTokenPredictor.num_mtp_layers` が一致。

### 22.7 注意: `--host` / `--port` は dataclass フィールド

- `entrypoints/launchers/cli_args.py:283` `host: str | None = None`, `port: int = 8000`
  → フラグとしては有効（`--host`/`--port` で上書き可）。grep で `add_argument("--host")` が
  見つからないのは dataclass 経由で自動生成されるため。

---

## 23. 実測（追加プローブ）— 画像・パス・速度・VRAM

### 23.1 GPU 画像の torch は 2.13.0+cu130、torchaudio は 2.8.3+cu128（v2 ログ）

```
RuntimeError: Detected that PyTorch and TorchAudio were compiled with different
CUDA versions. PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA
version 12.8.
```
→ 版数だけでなく **CUDA ビルド**が違う。`torchaudio==2.11.0` の PyPI wheel は
`+cu130`（wheel 内 `torchaudio/version.py` で確認: `__version__='2.11.0+cu130'`）
なので `pip install --no-deps --force-reinstall torchaudio==2.11.0` で一致する。

- CPU 画像は torch 2.11.0+cpu / torchaudio 2.11.0+cpu（GPU と別物。CPU で再現不可）
- torchvision 0.28.0 の METADATA は `torch==2.13.0` を要求（整合）
- 事前チェックは `python -c "import torchaudio"` を serve より前に単独実行する形に変更
  （help 経由だと原因が埋もれる）

### 23.2 Dataset のマウント位置は 2 通りある（実測）

- `mtpload` 実行時: `/kaggle/input/datasets/yamahei21python/<slug>/`
- `quantschema` 実行時: `/kaggle/input/<slug>/`（`datasets/` 無し）

→ プローブは `config.json` を glob して探す `find_root()` 方式に統一（実装済み）。
本番ビルダーは既に `_find_input()` で 3 パターンを glob している（問題なし）。
**プローブ側だけが直書きで 2 回落ちた**（quantschema / tplplus）。

### 23.3 読み出し帯域 82 MiB/s → 重みロード 244 秒

```
512 MiB in 6.27s = 82 MiB/s
19.5 GiB 全体: 244s（TP=2 で各 GPU が半分読むなら 122s）
```
- `/kaggle/working` 空き **19.5 GiB**。ステージングは symlink（実体コピーなし）なので
  ディスクは消費しない（実装確認済み）。
- cloudflared は**未インストール**（毎回ダウンロードが必要）。

### 23.4 重み内訳（実測）

| ファイル | テンソル | サイズ | dtype |
|---|---|---|---|
| model.safetensors | 2384 | 17.41 GiB | BF16 1184 / I32 800 / I64 400 |
| model-mtp.safetensors | 15 | 0.79 GiB | BF16 15 |

- `visual` 合計 **0.86 GiB**（`--language-model-only` で削減できる上限）
- `quantization_config.ignore` は **305 エントリ**（visual 全部 + linear_attn 一部）
- prefix は `model.language_model.layers.*` / `model.visual.*`（vLLM mapper が
  `model.language_model.` → `model.` に変換）

### 23.5 VRAM 予算（TP=2 / T4 16 GiB）

| util | budget/GPU | 重み/GPU | 残 |
|---|---|---|---|
| 0.90 | 14.40 GiB | 9.75 GiB | 4.65 GiB |
| 0.95 | 15.20 GiB | 9.75 GiB | 5.45 GiB |

| max_len | KV/GPU | +20% 余裕 |
|---|---|---|
| 8192 | 0.25 GiB | 0.30 |
| 16384 | 0.50 GiB | 0.60 |
| 32768 | 1.00 GiB | 1.20 |
| 65536 | 2.00 GiB | 2.40 |

→ 32768 でも余裕。**ボトルネックは VRAM ではなく起動時間（ロード 122〜244s）**。

## 24. template / 特殊トークン実測（tplplus v2）

```
tokenizer        : Qwen2Tokenizer
chat_template    : 解決 OK（8952 chars, .jinja から）
eos              : <|im_end|> = 248046
bos / pad        : 248044（= <|endoftext|> 系）
```

`generation_config.json`（モデル同梱）:
```json
{"bos_token_id": 248044, "eos_token_id": [248046, 248044],
 "do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
 "repetition_penalty": 1.05}
```
- **eos が 2 個**（248046 / 248044）。vLLM は generation_config を読むので停止条件は問題なし
- **`repetition_penalty` 既定 1.05** = proxy の下限と一致（矯正しても実質同じ）

### 24.1 生成プロンプトに `<think>` が強制される

- `apply_chat_template(add_generation_prompt=True)` の末尾は
  `<|im_start|>assistant\n<think>\n` → **常に思考モードで開始**
- → `--reasoning-parser qwen3` が必須（`reasoning_content` に分離しないと UI に生タグが出る）

### 24.2 `reasoning_effort` の実コスト（実測トークン数）

| 値 | トークン | system に入る文 |
|---|---|---|
| （未指定 = xhigh） | **62** | "Reasoning effort is set to xhigh. Please think carefully..." |
| `xhigh` | 62 | 同上 |
| `low` | 50 | "Keep your thinking brief and focused..." |
| `medium` | **24** | 指示文なし（最短） |
| `high` | **エラー** | `TemplateError: Supported types are xhigh (default), medium, low` |

→ コスト削減するなら proxy が `chat_template_kwargs={"reasoning_effort":"medium"}` を
注入するのが最も効く（**-38 tokens/リクエスト**）。ただし思考が浅くなる。
`high` は**送ってはいけない**（400 になる）。

### 24.3 tools / vision

- tools: system に長い指示（1390 chars / 314 tokens）+ `<tool_call><function=..><parameter=..>` XML
- tool 呼び出しの assistant メッセージは
  `<think>\n\n</think>\n\n<tool_call>...</tool_call>` の形で描画 → 往復可
- vision content（`{"type":"image"}`）も描画可（342 chars）
- 4k 文字の日本語 = **2052 tokens**（≈ 1.95 chars/token。日本語は重い）

## 25. 実 install 後の構成解決（probe8: cfgresolve / genconf）

`pip install vllm==0.32.0` を CPU で実際に通した上での実測。**推測ではない**。

### 25.1 ModelConfig（model=dataset, dtype=half）

```
quant='compressed-tensors'   dtype=torch.float16
arch=['Qwen3_5ForConditionalGeneration']  model_type='qwen3_5'
qc.method='compressed-tensors' qc.format='pack-quantized'
mm=MultiModalConfig lang_only=False  gen_cfg='auto'
diff_sampling={'repetition_penalty':1.05,'temperature':1.0,'top_k':20,'top_p':0.95}
```

| 指定 | 結果 |
|---|---|
| 指定なし | OK（config から compressed-tensors を自動解決） |
| `quantization="compressed-tensors"` | OK |
| `quantization="awq"` | **ValidationError**（config は compressed-tensors） |
| `quantization="auto"` | **ValidationError**（同上） |
| `language_model_only=True` | OK。`mm.limit_per_prompt={}` / `get_limit_per_prompt()` は常に 0 |
| `generation_config="vllm"` | `diff_sampling={}`（= 中立既定。モデル既定を無効化できる） |
| `override_generation_config={"repetition_penalty":1.1}` | 反映はされるが上書き警告あり |

→ **`--quantization` は付けない**のが正。`awq`/`auto` の明示は試行段から削除済み。

### 25.2 SpeculativeConfig(mtp)

- `num_speculative_tokens` は `gt=0`（未指定 / 0 / -1 はすべてエラー）
- `target_model_config` 必須（無いと `ValueError: target_model_config must be present for mtp`）
- `target_parallel_config` も実質必須（無いと `AttributeError: 'NoneType' object has no attribute 'tensor_parallel_size'`）
- `model` 省略時は `target_model_config.model_weights or .model` を流用
- `quantization` 未指定なら target の quantization を**継承**する
- draft の arch は qwen3_5 系なら `Qwen3_5MTP` に書き換わる（`hf_config_override`）。
  `n_predict` は `mtp_num_hidden_layers`（このモデルは **1**）
- `num_speculative_tokens > 1` は「同じ MTP 層を複数回 forward する」警告が出る
  （qwen3_5 はエラーではない）。既定 `VLLM_SPEC_TOKENS=2` は許容範囲

### 25.3 量子化 ignore は「prefix 文字列」と照合される（重要）

`compressed_tensors.py:1011` → `should_ignore_layer(layer_name, ignore, fused_mapping)`
→ `find_matching_patterns` → `is_equal_or_regex_match`。

```
一致条件: target == value          （完全一致）
          target が "re:" 始まりなら re.match
          use_fnmatch=True なら fnmatch
```

- `layer_name` に入るのは **モジュール名ではなく prefix 文字列**
  （`ColumnParallelLinear(prefix=...)` → `get_quant_method(layer, prefix)`）
- したがって ignore の `model.visual.blocks.0.attn.qkv` は、視覚塔の prefix
  `model.visual.blocks.0.attn.qkv` と**一致する**（keymap で「key と照合して 0 件」
  だったのは判定経路が違うため。慌てない）
- ignore の `re:.*mtp.*` / `re:mtp\..*` は draft の prefix `mtp.fc`, `mtp.layers.0...`
  に一致する → MTP 層は非量子化のままロードされる
- draft の `load_weights` は `mtp.` → `model.` に改名するが、これは**重みキーだけ**。
  prefix は `mtp.*` のまま残るので ignore は効く（`qwen3_5_mtp.py:307-319`）

### 25.4 重みキー × mapper（keymap v2 実測）

| mapper | 結果 |
|---|---|
| text 用 `{"model.language_model.": "model.", "mtp.": None}` | 2050 変換 / 334 未変換（`lm_head.` と `model.visual.` が残る）→ **単体では不足** |
| VL 用（実使用）| `visual.` 333 / `language_model.model.` 2050 / `language_model.lm_head.` 1 / **変換漏れ 0** |

- draft（`model-mtp.safetensors`）は `mtp.` → `model.` で **15/15 変換**、想定外 0
- `weight_shape` は **I64**（`<2q`）。I32 で読むと `(10240, 0)` になる
- 400 件すべて TP=2 で割り切れる（最大 `(5120, 17408)`）

### 25.5 環境変数（probe8: enviro）

- `envs.py` が読む VLLM_* は **304 個**
- **`VLLM_USE_V1` / `VLLM_ATTENTION_BACKEND` は存在しない**（0.32 に V0 エンジンは無い）
- `validate_environ(hard_fail)` は未知の `VLLM_*` を警告するだけ（既定 `hard_fail=False`、
  フラグ `--fail-on-environ-validation`）
- 実在: `VLLM_GDN_DECODE_KERNEL`(cuda/triton) / `VLLM_TRITON_JIT_WARMUP_NUM_THREADS`(4) /
  `VLLM_WORKER_MULTIPROC_METHOD`(fork) / `VLLM_DISABLE_COMPILE_CACHE`

### 25.6 generation_config の既定値（probe8: genconf）

- `SamplingParams()` の既定は temperature 1.0 / top_p 1.0 / top_k 0 / rp 1.0 / seed None
- **`temperature=0` はエラーにならず受理される**（貪欲法）→ proxy の矯正は必須のまま
- `temperature=None` は `TypeError`
- `update_from_generation_config` が反映するのは **eos / stop のみ**
  （`stop_token_ids=[248044]`, `_all_stop_token_ids={248044,248046}`）。
  temperature 等は `get_diff_sampling_param()` 経由で serving 側の既定に入る

### 25.7 proxy の `--enforce-sampling` に潜んでいたバグ（修正済み）

`path_only.rstrip("/").endswith("/chat/completions")` だと
**`/v1/chat/completions?x=1` で矯正が素通り**する（クエリ文字列付きで破綻）。
`urllib.parse.urlsplit(path_only).path` でパスだけ取り出して判定するよう修正。

自己テストを 25 → **45 ケース**に拡張（ローカル 45/45）:
- query 付き / query + 末尾スラッシュ
- 温度の文字列・bool・負値・null・0.0
- RP の null / 0 / 0.5 / "1.0"
- 不正 JSON / 空 / `[]` / `"str"` / `null` は素通し
- 非 ASCII と未知フィールドの保持
- **HTTP 統合**（専用上流を別ポートに立て、Content-Length 再計算と query 経路を実測）

## 26. モデル形状（probe9: filemat 実測）

```
num_hidden_layers=64  layer_types: linear_attention 48 / full_attention 16
full_attention 層 = [3,7,11,...,63]（4 層ごと）
hidden=5120 heads=24 kv_heads=4 head_dim=256 intermediate=17408
linear_num_key_heads=16 linear_num_value_heads=48 linear_key/value_head_dim=128
linear_conv_kernel_dim=4  mamba_ssm_dtype=float32  mtp_num_hidden_layers=1
```

KV = 16 層 × 4 kv_heads × 256 × 2(K,V) × 2B = **64 KiB/token**

| max_len | KV/GPU (TP=2) |
|---|---|
| 8192 | 0.25 GiB |
| 16384 | 0.50 GiB |
| 32768 | 1.00 GiB |
| 65536 | 2.00 GiB |

## 27. ロード経路（probe9: loadflow 実測）

- dataset: `model.safetensors` **17.41 GiB** + `model-mtp.safetensors` **0.79 GiB**
- `allow_patterns=['*.safetensors']` で **index.json が無い**ため
  `filter_duplicate_safetensors_files` は素通し → target が mtp も読む（mapper で破棄）
- draft も同じ dir を読むので **0.79 GiB を 2 回**読む
- 実測 82 MiB/s だと target 18.21 GiB = 227s、合計 19.0 GiB = 237s
  （TP=2 で各 GPU が半分読むなら target 114s）
- mtp 15 テンソルは全部 BF16（合計 0.79 GiB、最大 `mtp.layers.0.mlp.*` 170 MiB 各）

## 28. 起動方針の変更（この節が結論）

- **`--language-model-only` を全段の既定にした**（視覚塔を落とす）。
  テキスト配信しかしないので不要。0.86 GiB と mm profiling の VRAM を節約できる。
  実装は `StageMissingLayer` で子モジュール登録されないため未使用キー警告も出ない。
- 退避段 `vl0` は `--limit-mm-per-prompt '{"image":0,"video":0}'`
  （`get_limit_per_prompt()==0` で同じ StageMissingLayer 経路に入る）
- 量子化は指定なしのまま（25.1）。`awq`/`auto` 段は削除済み
- T4(sm75) は FlashAttention が `capability >= (8,0)` を要求して弾かれ、
  `triton_attn.supports_compute_capability()` が常に True なので **TRITON_ATTN** が選ばれる

## 29. SpeculativeConfig(mtp) 実構築（probe9: specmat）

`target_parallel_config` を渡せば通る（probe8 の AttributeError はテスト側の不足）。

```
target: quant=compressed-tensors arch=['Qwen3_5ForConditionalGeneration']
text_config: num_hidden_layers=64 mtp_num_hidden_layers=1 num_kv_heads=4 head_dim=256
layer_types[:6] = [linear, linear, linear, full, linear, linear]   # 4 層ごとに full
```

| 入力 | 結果 |
|---|---|
| n=1 / 2 / 3 / 4 | すべて OK。`method` は `mtp` に正規化 |
| `{}`（n 未指定） | ValidationError: "num_speculative_tokens must be provided ..." |
| `qwen3_5_mtp` / `qwen3_next_mtp` | OK → `method=mtp`（deprecated 警告） |
| `MTP_model` | ValidationError（Literal に無い） |

draft の解決結果（全ケース共通）:
```
draft.model = target.model（同一ディレクトリ）
draft.arch  = ['Qwen3_5MTP']   model_type=qwen3_5_mtp
draft.n_predict = 1            draft.quant='compressed-tensors'（target を継承）
draft.dtype = float16          draft.max_model_len = target と同じ
draft_parallel.tp = 2          （target_parallel_config を継承）
```

→ **draft は 0.79 GiB の mtp ファイルだけ読むつもりでも、`model` が同じ dir なので
`*.safetensors` glob で 17.41 GiB 側も対象になる**（§27）。実 read は 2 倍近い。

## 30. probe10/11 回収（flagparse / parsersmoke / draftdir）

### draftdir — 成功
```
symlink 7 個
*.safetensors: ['model-mtp.safetensors']  → 合計 0.79 GiB ✓
AutoConfig(draft dir): model_type=qwen3_5, text_config.mtp_num_hidden_layers=1
ModelConfig(draft dir): quant=compressed-tensors dtype=float16 ✓
SpeculativeConfig(model=DRAFT_DIR): method=mtp n=2, draft.arch=['Qwen3_5MTP'],
  draft.model='/kaggle/working/draft', draft_tp=2 ✓
model 省略時: draft.model=dataset dir → *.safetensors 2 ファイル（二重読み）✓ 再現
```
→ builder の `_stage_draft()` + `SPEC_DIR` 段は**成立**。-200s 以上の短縮。

副次的発見: draft のモジュール prefix は `mtp`（`maybe_prefix(prefix, "mtp")`）なので
config の ignore `re:.*mtp.*` が **draft の量子化を無効化する**。mtp 重みは BF16 のまま
読める（量子化カーネルを通らない）→ sm75 でも安全。

### flagparse — CPU では実行不能（重要な制約）
```
RuntimeError: Failed to infer device type
  ← vllm/config/device.py:55、EngineArgs.add_cli_args → get_kwargs(VllmConfig)
```
`make_arg_parser()` は VllmConfig のデフォルト値生成で **device を要求**する。
CPU カーネルでは GPU が無いので必ず落ちる → **実 argparse でのフラグ検証は CPU 不可**。
代替として wheel の dataclass フィールドを静的に列挙して照合（下記）。

### フラグ静的照合（wheel のデータクラス 621 フィールドと突き合わせ）
builder が出す 42 フラグのうち、vLLM に無いものは 0 件。
（`--disable-log-requests` は 0.32 に存在せず `enable_log_requests` のみ →
  `_has()` ゲートで GPU 上でも付かない。`--enforce-sampling` は proxy 専用、
  `--pooling-enabled` は cloudflared 専用。誤検出はこれだけ。）

### parsersmoke
```
tokenizer: CachedQwen2Tokenizer vocab=248044
template: 8952 文字, <think> あり, <tool_call> あり
render(tools なし): 57 ids / (tools あり): 309 ids（+1071 文字, <tools> 注入）
reasoning_effort low/medium/xhigh → すべて 57 ids = **テンプレートが無視する**
  → --default-chat-template-kwargs は効果なし。付けなくてよい（害もなし）
tool parser qwen3_xml: Qwen3EngineToolParser, tools_called=True ✓
  extract_tool_calls(text, request) は request 必須（省略不可）
reasoning parser qwen3: registry に存在（実抽出は未確認、json 化で落ちただけ）
```

## 31. sm75 で量子化カーネルが動くかの静的確認

| 項目 | 結果 |
|---|---|
| `CompressedTensorsWNA16.get_min_capability()` | **75**（"Turing and up"） |
| `_POSSIBLE_KERNELS[CUDA]` の候補 | CutlassW4A8(90) → Machete(90) → Marlin(**75**) → Conch(80) → Exllama(**60**) → TritonW4A16 → Humming(75) |
| `vllm/_C_stable_libtorch.abi3.so` の cubin | **sm_75 あり**（sm_80/90 と同居） |
| `supported_dtypes`（CC 7.5） | `[float16, float32]` → `--dtype half` は正 |
| Marlin が sm75 で落ちる場合 | Exllama(min 60) / TritonW4A16(min 0) に降りる |
| GDN の ignore | config の ignore 305 件に `re:.*mtp.*` / `re:mtp\..*`。GDN 層は個別 ignore（BF16 のまま）と推定 |

→ 「T4 で int4 が動かない」系の心配は構造的には解消。

## 32. 起動後の生成スモークを追加（重要）

`/v1/models` が返っても**初回 forward で落ちる**領域（Marlin/Exllama カーネル、
GDN の Triton、MTP の draft forward）がある。サーバーは 200 を返し続ける
`/v1/models` だけでは検知不能。
→ 起動判定の直後に `chat/completions` を 1 回（max_tokens=16）投げ、
  失敗したら kill して次段へ。`_smoke()` として実装（builder 反映済）。

予算も修正: `STARTUP_BUDGET` 1500→**1080**（18 分）、`ATTEMPT_TIMEOUT` 600→**420**。
残クォータ 2,395s に対して 25 分は取りすぎ（公開できても数分で打ち切られる）。
`KEEPALIVE_MINUTES` 2→**12**（URL を実際に叩ける長さ）。

## 33. ローカル単体テスト（Kaggle セッション不要）: `step2/_selftest_builder.py`

生成済み notebook から **AST で関数を抜き出して exec** する方式。ビルダーではなく
生成物を検証するので、ビルダーと notebook のズレ（drift）も同時に検出できる。

```
python3 kaggle/step2/_selftest_builder.py   # 全部ローカル・数秒・GPU 不要
```

| 対象 | 検査 | 結果 |
|---|---|---|
| `_stage_model(False)` | mtp を除外 / 本体は残す / 隠しファイル除外 / ディレクトリ除外 | ✅ |
| `_stage_model(True)` | mtp を残す / 再作成可（rmtree 後） | ✅ |
| `_stage_draft()` | symlink のみ / mtp 含む / 本体含まず / tokenizer 含む / **safetensors 1 個** / 返り値==作成数 | ✅ 7 個 |
| `_stage_draft()` mtp 無し | 0 を返す・dir を作らない（安全網） | ✅ |
| `probe_url()` | 実ログから URL 抽出 / URL 無し / 偽ドメイン除外 / ログ無し | ✅ 4 件 |
| `_smoke()` | 200 正常 / **usage: null** / content 空+reasoning のみ / **500 カーネルエラー** / 400 / 接続不可 | ✅ 6 件 |
| PLAN ラダー | 9 段 / mtp+dd 先頭 / `--quantization` は ct 段のみ / `TRITON_ATTN` に spec 併用なし / `--enforce-eager` に spec 併用なし / eager 位置 / mtp 再挑戦が後半 / `assert len(p)==7` 存在 | ✅ 9 件 |

副産物の発見:
- `usage: null` を返すサーバーがあると `d.get('usage', {}).get(...)` は
  AttributeError。→ `(d.get("usage") or {})` に修正済（テストで先に踏んだ）。
- `_stage_model(False)` は `chat_template_medium.jinja` も symlink する（無害）。
- `_stage_draft()` は `MODEL_SRC`（dataset）を見る。`STAGE` とは独立なので
  plain 段で STAGE を mtp 抜きに作り直しても draft dir は壊れない。

これで **Kaggle CPU セッションを使う追加検証は不要**（残りは GPU でしか判定不能）。

---

## 34. keepalive リセット（`KEEPALIVE_KEEPALIVE = 60`）の実挙動

CPU で proxy を実際に起動して確認（`kaggle/step2/_selftest_proxy.py` 50/50）。

| 操作 | `/tmp/keepalive.conf` |
|---|---|
| POST `/v1/chat/completions` | `RUN=1` / `MINUTES=60` に上書き |
| GET `/v1/models`（中継される GET） | 同じく上書き（`_relay` 冒頭で無条件） |
| GET `/quota` | **変更なし**（ローカル処理・`_relay` を通らない） |
| GET `/healthz` / UI | 変更なし（同上） |

要点:
- リセットは「チャット送信」専用ではなく **上流へ中継する全リクエスト**。
  UI のポーリングでも延びる。`/quota` だけは延ばさない（管理用）。
- 書き込みは例外を握り潰す（`try/except: pass`）ので、`/tmp` が書けなくても
  中継自体は止まらない。
- `MAX_MINUTES = 30` は **総実行時間**の上限。`limit = min(i-1+60, 30)` なので
  25 分地点でチャットしても残り 5 分。クォータ（39 分）超過を避けるための
  テスト用設定で、本番は 480 に戻す。
- `MINUTES=60` が既定値のときは上限クランプの警告を出さない（毎分 1 行の
  ログ埋め尽くしを回避）。`999` など明示的な過大値のときだけ警告。
- `RUN=0` は `MINUTES` に関係なく `(0, 60)` を返し、ループは次周期で break。

---

## 35. 会話（マルチターン）の検証を追加

| 層 | 検査 | 場所 |
|---|---|---|
| proxy | 4 メッセージ（system/user/assistant/user）が**上流へ完全一致**で渡り、応答が戻る | `_selftest_proxy.py`（52 チェック） |
| dryrun | 公開URL経由で 4 メッセージ送信 → モックの `__last` と一致 | dryrun [6] `conversation-history` |
| GPU | 実モデルに 2 ターン（名前を教える → 名前を聞く）で**文脈保持**を確認 | GPU [6] `conversation` |

GPU [6] の注意:
- thinking モデルは `content` が空で `reasoning_content` だけ返すことがある。
  履歴には `content` 優先、空なら `reasoning_content[:600]` を積む。
- 判定は `content + reasoning` に「コウヘイ / コーヘイ / kohei」が含まれるか。
  含まれなければ文脈欠落として FAIL（チャットテンプレート faults を検出）。
- `max_tokens=96` × 2 ターン。T4 で ~20〜30 秒。

## 36. dryrun v6 の失敗（修正済）

`KeyError: '--language-model-only'`
- CPU では `vllm serve --help` が必ず exit=1 → `avail` を悲観側で**再構築**する。
- その再構築が `NEEDED` のキーだけ作り、後段の `avail["--language-model-only"]`
  が KeyError。**CPU では必ず通る経路**なので v6 は起動前に即死した。
- `NEEDED` に `--language-model-only` を追加。全 `avail[...]` 参照が宣言済みで
  あることを AST で照合するチェックを追加した。

その他の実測:
- pip install **6.0 分**（v5 は 7.2 分。回線次第）。予算は 7.2 分で見積もる。
- `Failed to import from vllm._C` は **CPU プラットフォーム限定の警告**。
  CUDA は `vllm._C_stable_libtorch` を使う（`platforms/cuda.py:23`）ので
  GPU では無関係。

---

## 37. dryrun v7: 6/6 パス（公開経路の CPU E2E 完了）

```
[1] ui                    -> PASS HTTP 200 61924B
[2] models                -> PASS ids=['qwen3.8-27b-awq']
[3] streaming             -> PASS chunks=13 first=0.00s spread=1.80s
[4] openai-compat         -> PASS HTTP 200 usage=12
[5] enforce-sampling      -> PASS upstream temp=0.7 rp=1.05
[6] conversation-history  -> PASS messages=4 equal=True
```

- v6 で FAIL した [1][2] は `_open_public`（IPv6 経路なし → IPv4 再解決）で解消。
- [6] は 4 メッセージの履歴が上流に完全一致で届くことを確認。
- keepalive 8 分は全周 `proxy=alive tunnel=alive | tunnel=OK`。
- **install 実測 511.98 秒（8.5 分）**。v6 の 6.0 分から振れる。
  GPU 予算は最悪 8.5 分で見積もる。
- これを受けて GPU 版の keepalive 既定を 20 → **15 分**に変更
  （8.5 + 5 + 1 + 15 + 1 ≈ 30 分 < 残 39 分）。

---

## 38. vLLM 0.32 の `serve --help` は要約だけ（GPU 実測）

GPU 実行のログで発覚:
```
vllm serve --help: exit=0 2134 文字
Config Groups:
  positional arguments
  options
  Frontend ...
ℹ --speculative-config 非対応 or MTP=0 → MTP なしで起動
ℹ --language-model-only 非対応 → 視覚塔ありで起動
```

- 0.32 は `--help` で **Config Group の一覧しか出さない**。
  個別フラグは `--help=all`（help 本文に明記されている）。
- 旧 `HELP_OK` は「500 文字以上 + Traceback 無し」で完全版と誤認していた。
  その結果 `_has()` が全フラグを偽判定し、**MTP / LMO / chat-template が全部落ちた**。
- CPU では `--help` 自体が exit=1 なので、この罠は GPU でしか踏めなかった。

### 修正
```python
def _looks_complete(text):
    return (len(text) > 5000
            and "--tensor-parallel-size" in text
            and "--max-model-len" in text)

_rc, HELP = _run_help("--help=all")
if not _looks_complete(HELP):
    _rc2, _h2 = _run_help("--help")
    if len(_h2) > len(HELP):
        HELP, _rc = _h2, _rc2
HELP_OK = _rc == 0 and _looks_complete(HELP) ...
```
取れなければ従来どおり `VERIFIED_FLAGS`（wheel 静的解析済み）に倒す。
`_looks_complete` の単体テストを追加（要約/完全版/短すぎ）。

### 教訓
「help が取れた」＝「フラグが載っている」ではない。
**既知の必須フラグが本文に含まれるか**で完全性を判定する。

---

## 39. GPU 実測: 27B ハイブリッドは T4 x2 の 7 分では起動しない

2026-10-29 の GPU v3（`--help` バグ込み・MTP OFF / LMO OFF）の結果。

```
12.8s   GPU ['Tesla T4','Tesla T4']  VRAM 15360 MiB x2
350.8s  vLLM 0.32.0 / torch 2.13.0+cu130 導入完了（install 336s）
370.5s  --help 要約 2134 文字 → MTP / LMO / chat-template 判定が全部外れる
370.5s  試行 1/7: plain 32768（LMO off = 視覚塔あり）
790.6s  ❌ 420s タイムアウト
```

タイムアウト直前のログ:

```
(Worker_TP1) Setting attention block size to 784 tokens ... mamba page size
(Worker_TP0) Encoder cache will be initialized with a budget of 16384 tokens
(Worker_TP0/1) [backbone] Starting torch.compile
(EngineCore) No available shared memory broadcast block found in 60 seconds   ← x2
(Worker_TP0/1) torch.compile took 61.31 s in total
(EngineCore) No available shared memory broadcast block found in 60 seconds   ← さらに
```

### 読み取れること

- このモデルは **ハイブリッド（GDN/Mamba + attention）**。
  `mamba page size` / `LBNHC KV cache layout` / `Encoder cache` が出ている。
- `torch.compile` 自体は 61s で完了している。つまり **コンパイルで詰んでいる
  わけではない**。その後に engine→worker のブロードキャストが返ってこない。
- `No available shared memory broadcast block` は「**読み手が取りに来ない**」
  という意味で、書き手（EngineCore）がリングバッファの空きを待っている状態。
  → ワーカーが CUDA graph capture / ダミー実行 / Triton JIT で長く止まっている。
- したがって **時間不足の可能性が高い**（真のハングかは 900s 与えないと分からない）。
- LMO off のせいで視覚塔 + encoder cache（16384 tokens のプロファイル）も
  走っており、これが余計な時間と VRAM を食っている。

### 対策（実装済み）

| 対策 | 理由 |
|---|---|
| `--help=all` + `_looks_complete()` | MTP / LMO / chat-template を正しく有効化（§38） |
| `PLAN.insert(1, eager@16k)` | CUDA graph capture 仮説の切り分けを最優先 |
| `OMP_NUM_THREADS=4` | Triton/inductor は CPU バウンド。既定 1 では遅すぎる |
| 待機ログに **vLLM ログ末尾 + nvidia-smi** を 60s ごとに表示 | 「何で詰まっているか」を UI ログから読めるようにする |
| 起動状況を **Gist（`vllm_status.txt`）** に 60s ごとに PATCH | `kaggle kernels output` は完了まで空。外から進捗を見る |
| `/dev/shm` のサイズを起動時に表示 | TP>1 は 240 MiB のリングバッファを 1 本で確保する |
| 予算 1800s / 1 試行 900s / keepalive 30 分 | 7 分では足りないと実測。リセット後（30h）前提 |

### 次回の順序

```
mtp+dd@32k → eager@16k → plain@32k → plain@16k → plain@8k
→ ct@16k → triton@8k → mtp@16k → vl0@8k
```

`eager` が通れば「CUDA graph capture が原因」で確定。通らなければ
モデル/量子化/TP の組み合わせ自体を疑う（TP=1 + 8k などへ退避）。

---

## 40. コンテキスト長の上限（サイズ計算）と 64K への変更

Ollama 版と vLLM 版のメモリ差を整理した。

| | Ollama（実績あり） | vLLM（今回） |
|---|---|---|
| 重み | 16.52 GiB（Q4_K_M） | 17.41 + MTP 0.79 = **18.20 GiB**（AWQ） |
| KV / token | **34 KiB**（q8_0） | **68 KiB**（f16。T4 は fp8 KV 不可） |
| 予算 | 26.4 GiB（VRAM 88%） | 27.0 GiB（util 0.90） |

| ctx | vLLM KV | vLLM 合計 | 余裕 | Ollama 合計 |
|---|---|---|---|---|
| 16K | 1.06 | 19.26 | 7.7 | 17.05 |
| 32K | 2.13 | 20.33 | 6.7 | 17.58 |
| 64K | 4.25 | **22.45** | **4.5** | 18.65 |
| 96K | 6.38 | 24.58 | 2.4 | 19.71 |
| 128K | 8.50 | 26.70 | 0.3 | 20.77 |

（GiB。余裕 = 予算 − 合計。CUDA graph / activation / NCCL / mamba state が 1〜2 GiB 食う）

### 決定
- **狙いは 64K**。96K は余裕 2.4 GiB で際どく、128K は確実に溢れる。
- Ollama が 131K で回っていたのは KV が半分（q8_0）+ オーバーヘッド小のため。
- `MAXLEN_CANDIDATES = [65536, 32768, 16384, 8192]` に変更。

### 新しいラダー

```
mtp+dd@64K → eager@64K → plain@64K → plain@32K → plain@16K
→ ct@16K → triton@8K → mtp@16K → vl0@8K
```

`eager` は graph capture を飛ばすぶんメモリも食わないので 64K で試す。

---

## 41. 試行 2/3 の「125 秒即死」はプロセス残骸が原因（推定）

attempt 2 (16K) と attempt 3 (8K) が同じ形で死んだ:

```
125s  RuntimeError: Engine core initialization failed. See root cause above.
      Failed core proc(s): {}
      resource_tracker: leaked shared_memory objects: {'/psm_2289413e'}
```

- 4000 文字の末尾切り出しでは **root cause が全部枠外**（外側の traceback だけで埋まる）。
  attempt 2 の原因は失った。→ 次回から試行ログを**全量保存**（3MB 上限）。
- 疑い: `terminate()` は API サーバーにしか届かず、EngineCore / Worker_TP* が残って
  VRAM と `/dev/shm` を掴んだまま。attempt 2 は 3 秒後に起動している。
- 対策:
  - `start_new_session=True` + `os.killpg(TERM→KILL)`
  - `/dev/shm` の `vllm*` / `psm_*` を削除
  - 次の試行前に VRAM が 2000 MiB 未満になるまで待つ（最大 90 秒）
  - 起動前チェック（3000 MiB 超なら掃除）

---

## 42. GPU v4: 自分のバグで 80 秒即死（教訓と再発防止）

### 何が起きたか

install に 504 秒かけた直後、起動ループのセルが 80 秒で落ちた。

```
NameError: name 'STARTUP_BUDGET' is not defined
Exception encountered at "In [4]"
```

`PLAN.sort(...)`（予算が短いとき eager を先頭に並べ替える処理）を
`PLAN` の直後に置いたが、`STARTUP_BUDGET` の定義は 400 行ほど後ろにあった。

### 再発防止（`_selftest_builder.py` の 9 番）

セルを順に AST 走査し、**モジュールレベルで実行される `Name(Load)` が
定義より前に現れたら失敗**させる。合成セルで検証済み:

```
buggy -> ['cell0:STARTUP_BUDGET']
fixed -> []
```

扱いの細部:
- 関数/ラムダ/クラス**本体は覗かない**（実行時解決）。引数の既定値だけ見る
- `if` / `try` / `for` / `with` の中は実行されるので再帰する
- 内包表記のターゲットは内包スコープの局所変数 → 式の行で定義済み扱い
- `except X as e:` の `e` は `Name` ノードではないので個別に拾う

### この実行で確定したこと

| 項目 | 実測 |
|---|---|
| `/dev/shm` | **14,336 MiB**（空きも同じ）→ 共有メモリ不足説は**消えた** |
| install | **504 秒**（過去最悪。336/512/504 と振れる） |
| torchaudio 修正 | 14 秒（2.11.0 を --no-deps で入れ直し） |
| クォータ消費 | 618 秒（うち 504 秒が install） |

### 教訓

- **GPU 1 回 = クォータ 10 分**。ローカルで検証できることを GPU で試さない。
  単体テストは関数単位でなく「セル全体の静的検査」までやる
- install が 20 分の 42% を食う現状では、起動実験の余裕が無い
  → wheelhouse Dataset（2.0 GB / 182 wheel）で 60〜120 秒にする

---

## 43. wheelhouse の穴: Mac 由来の解決は Linux マーカーを落とす

### 症状

CPU probe（Kaggle Linux）で `--no-index` の解決が即失敗。

```
ERROR: Could not find a version that satisfies the requirement
       tokenspeed-mla==0.2.15; platform_system == "Linux" (from vllm)
```

### 原因

Mac 上で `pip download --platform ...` しても、**環境マーカーの評価はホスト OS**。
`platform_system == "Linux"` は Darwin では False なので、その依存は取得されない。
結果、wheelhouse は「Mac から見た閉包」であって Linux の閉包ではなかった。

### 静的に穴を検出する方法

wheelhouse 内の全 wheel の `*.dist-info/METADATA` を読み、`Requires-Dist` を
**Linux 環境のマーカー**で評価する:

```python
env = {"platform_system": "Linux", "sys_platform": "linux", "os_name": "posix",
       "platform_machine": "x86_64", "python_version": "3.13",
       "python_full_version": "3.13.15", "implementation_name": "cpython",
       "platform_python_implementation": "CPython", "extra": ""}
``'

検出された穴（vllm 0.32.0 / torch 2.13.0）:

| 不足 | 要求元 | 備考 |
|---|---|---|
| tokenspeed-mla==0.2.15 | vllm | Linux 限定 |
| cuda-toolkit[...]==13.0.3 | torch | extras で cublas/cudart/... |
| nvidia-cudnn-cu13==9.20.0.48 | torch | ~700 MB |
| nvidia-nccl-cu13==2.29.7 | torch | manylinux_2_18 タグ |
| nvidia-nvshmem-cu13==3.4.5 | torch | |
| nvidia-cusparselt-cu13==0.8.1 | torch | |
| triton==3.7.1 | torch | **3.9.0 は版違いで不可** |

### 落とし穴 2: `--platform` の一覧不足

`nvidia-nccl-cu13 2.29.7` は `manylinux_2_18_x86_64`。2_17/2_24/2_27/2_28 を
指定していても取れない。**2_5〜2_39 を総当たりで並べる**のが安全。

### 落とし穴 3: 依存パッケージが torch を最新へ引き上げる

`tokenspeed-mla` は `torch`（無指定）を要求するため、`pip download` 単体では
torch 2.15.0 まで拾ってしまう。vllm は `torch==2.13.0` の厳密ピンなので、
**2.15.0 を削除**。同様に vllm のピンに合わせて以下を削除:

- `nvidia-cutlass-dsl*` 4.8.1（vllm は ==4.8.0）
- `apache-tvm-ffi` 0.1.15.post1（vllm は ==0.1.11）
- `setuptools` 84.2.1（vllm は <81.0.0）→ 80.10.2

### 最終状態

```
190 wheel / 3.0 GB / 静的検査で不足 0
triton 3.7.1（torch の厳密ピン）
torch 2.13.0 / torchaudio 2.11.0 / torchvision 0.28.0
```

### 教訓

- Mac で wheel を集めるときは「マーカーはホストで評価される」を忘れない
- 集めた後は **Linux マーカーでの静的閉包検査**を必ず通す
- ダウンロード後に **重複バージョン**を検査する（ピンと違う版が混ざる）

# vLLM wheelhouse CPU 検証結果

## 最終結果 (probe v5)

| 項目 | 値 |
|------|-----|
| wheel 数 | 190 |
| install exit | 0 ✅ |
| install 時間 | **300.5 秒 (5.0 分)** |
| GPU 版 install | 504 秒 (8.4 分) |
| 短縮幅 | **~204 秒 (40% 削減)** |
| import vllm | ❌ libcufile.so.0 不在 (CPU 画像のため) |
| vllm bin | /usr/local/bin/vllm (存在) |
| --help=all | exit=1, 647 文字 (libcufile.so.0 で失敗) |

## 発見した穴と修正

### 1. Mac 由来のプラットフォームマーカー穴
Mac (Darwin) で `pip download` を実行すると `platform_system == "Linux"` マーカー付き依存がスキップされる。
- `tokenspeed-mla==0.2.15` (vllm 依存)
- `cuda-toolkit==13.0.3.0` (torch 依存)
- `nvidia-cudnn-cu13==9.20.0.48` (~700MB)
- `nvidia-nccl-cu13==2.29.7` (manylinux_2_18)
- `nvidia-nvshmem-cu13==3.4.5`
- `nvidia-cusparselt-cu13==0.8.1`
- `triton==3.7.1` (3.9.0 は torch の厳密ピンと衝突)

### 2. cuda-toolkit のバージョン不整合
`cuda-toolkit==13.0.3.0` は `nvidia-cublas==13.1.1.3.*` を要求するが、PyPI には `13.8.2.12` しかない。
→ `cuda-toolkit` メタパッケージを wheelhouse から除去。GPU 画像には既にインストール済み。

### 3. 重複・版違いの除去
- `torch 2.15.0` → `2.13.0` (vllm の厳密ピンに合わせる)
- `nvidia-cutlass-dsl 4.8.1` → `4.8.0`
- `apache-tvm-ffi 0.1.15` → `0.1.11`
- `setuptools 84.2.1` → `80.10.2` (<81.0.0 制約)
- `triton 3.9.0` → `3.7.1`

### 4. `--no-deps` インストール戦略
`--no-index --find-links` では `cuda-toolkit` の依存 (`nvidia-cublas 13.1.1.3.*`) が解決できない。
→ `--no-deps` で全 wheel をインストール。静的解析で依存閉包は検証済み (不足 0)。
GPU 画像には cuda-toolkit が既にインストールされているため、`--no-deps` でも正常に動作する。

## 結論

- **wheelhouse は GPU デプロイに使用可能**
- install 時間: 300.5s (5分) vs 504s (8.4分) → 40% 削減
- CPU probe の import 失敗は libcufile.so.0 不在が原因 (CPU 画像のため、GPU では正常)
- vllm バイナリは `/usr/local/bin/vllm` に存在

## GPU 実行時の期待値
- install: ~5 分 (wheelhouse から)
- import vllm: ✅ (GPU 画像に CUDA ランタイムあり)
- vllm serve: ✅ (GPU 画像に libcufile.so.0 あり)
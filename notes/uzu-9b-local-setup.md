# uzu 9B-M ローカル実行セットアップ

状態: **実行可能・動作確認済み**（2026-10-28）

## 実行方法

```bash
cd /Users/kohei/Myproject/kaggleAI
./notes/run_uzu_9b.sh "1から80まで数字をカンマ区切りで列挙して。説明なし。"
```

対話モード:

```bash
./notes/run_uzu_9b.sh
```

## 構成

| 項目 | パス / 値 |
|---|---|
| uzu repo | `/Users/kohei/Myproject/uzu` |
| uzu binary | `/Users/kohei/Myproject/uzu/target/release/cli` (50MB) |
| モデル | `/Users/kohei/.cache/uzu-stage` |
| 本体 weights | `model.safetensors` 4.95GB |
| speculator weights | `speculator/model.safetensors` 720MB |
| M3 shape patch | `speculator/shapes.json` に `Apple M3` 追加 |
| encoding | `encoding.json` = `[{"type":"hanashi","name":"qwen3.5"}]` |
| 実行ラッパー | `notes/run_uzu_9b.sh` |

## 適用済み修正

### 1. rht_wrapper.rs

`crates/uzu-engine/src/encodable_block/linear/rht_wrapper.rs`

```diff
- if input_preparation.activation_quantization.is_some() && !allow_prequantized_activation {
+ if !allow_prequantized_activation {
```

- 未コミットのローカル修正。
- 修正後に `target/release/cli` を再ビルド済み。

### 2. shapes.json

`~/.cache/uzu-stage/speculator/shapes.json` に追加:

```json
"Apple M3": {
  "tree_budget": 16,
  "max_tree_depth": 16,
  "dflash_depth_override": null,
  "construction_method": {
    "type": "Weaver",
    "rounds": 16,
    "expand_per_round": 4,
    "expand_width": 4
  }
}
```

- 上流の shapes.json は M5/M6 のみ。M3 が無いと speculator が黙って無効化される。
- `run_uzu_9b.sh` は起動時に M3 エントリを冪等に追加する。
- モデル再DLで shapes.json が上書きされても、ラッパーが再適用する。

### 3. encoding.json

ステージングモデルには `encoding.json` が無い。無いと:

```
Error: Unable to load model: can not get encoding config
```

`~/.cache/uzu-stage/encoding.json` に固定:

```json
[{"type":"hanashi","name":"qwen3.5"}]
```

`run_uzu_9b.sh` はファイルを毎回上書きしない。欠損、不正JSON、内容不一致の時だけ再生成する。

## 動作確認結果

### 短い応答

```text
prompt: 1+1は？短く答えて。
output: 2
time to first token: 0.38 s
prefill speed: 57.19 t/s
generation speed: 7.01 t/s
tokens per forward pass: 2.00 t/f
memory used: 5.72 GB
```

短い応答では投機のオーバーヘッドで t/f が低い。想定通り。

### 長い応答

```text
prompt: 1から80まで数字をカンマ区切りで列挙して。説明なし。
time to first token: 0.41 s
prefill speed: 75.64 t/s
generation speed: 75.76 t/s
tokens per forward pass: 14.09 t/f
memory used: 5.72 GB
total energy: 80.61 J
duration: 4.58 s
```

**75.76 t/s / t/f 14.09**。M3 + 9B-M + 投機ONで実用速度。以前の 61.96 t/s を上回る。

## バックアップ

`notes/uzu-backup/` に固定化。

- `encoding.json`
- `shapes.json`（M3 patch 済み）
- `rht_wrapper.rs`
- `rht_wrapper.patch`
- `cli`（ビルド済みバイナリ）
- 詳細: `notes/uzu-backup/README.md`

`run_uzu_9b.sh` は encoding / shapes / CLI が欠損・破損した場合に backup から自動復旧する。

## 注意

- M3 はファンレス。連続実行で thermal throttling する。ベンチ間は 120 秒以上空ける。
- 16GB 環境では Ollama 常駐がスワップを起こす。`run_uzu_9b.sh` は起動前に Ollama モデルを自動停止する。`UZU_KEEP_OLLAMA=1` で無効化。
- `target/release/cli` はローカル修正版。`cargo build --release` で再生成可能。欠損時は backup から復元。
- モデル再DL時は `notes/run_uzu_9b.sh` 経由で起動すれば M3 patch / encoding は自動復旧する。

# uzu 9B-M バックアップ

作成: 2026-10-28

## 内容

| ファイル | 用途 |
|---|---|
| `encoding.json` | `~/.cache/uzu-stage/encoding.json` の固定コピー |
| `shapes.json` | `~/.cache/uzu-stage/speculator/shapes.json` の M3 patch 済みコピー |
| `rht_wrapper.rs` | M3 修正済み Rust ソース |
| `rht_wrapper.patch` | 上記修正の git diff |
| `cli` | M3 修正後にビルドした `target/release/cli` (v0.6.13, 50MB) |
| `config.json` | モデル config |
| `tokenizer.json` | tokenizer |
| `speculator-config.json` | speculator config |
| `runner.rs` / `state.rs` / `chat_completions.rs` | server パッチ適用済みソース |
| `server-patch.diff` | server パッチの git diff（thinking 既定 + CORS） |

モデル重量 (`model.safetensors` 4.95GB + `speculator/model.safetensors` 720MB) は
サイズが大きいためバックアップ対象外。

## 自動復旧

`notes/run_uzu_9b.sh` が起動時に以下を自動処理する。

- `encoding.json` 欠損 / 不正 / 不一致 → 固定値に修復
- `speculator/shapes.json` 欠損 / 不正 → backup から復元、M3 エントリを保証
- `target/release/cli` 欠損 → backup から復元
- Ollama モデル常駐 → 停止して unified memory を解放

## 手動復旧

```bash
cp notes/uzu-backup/encoding.json ~/.cache/uzu-stage/encoding.json
cp notes/uzu-backup/shapes.json ~/.cache/uzu-stage/speculator/shapes.json
cp notes/uzu-backup/config.json ~/.cache/uzu-stage/config.json
cp notes/uzu-backup/tokenizer.json ~/.cache/uzu-stage/tokenizer.json
cp notes/uzu-backup/speculator-config.json ~/.cache/uzu-stage/speculator/config.json
cp notes/uzu-backup/cli ~/Myproject/uzu/target/release/cli
chmod +x ~/Myproject/uzu/target/release/cli
```

Rust ソースを戻す場合:

```bash
cp notes/uzu-backup/rht_wrapper.rs \
  ~/Myproject/uzu/crates/uzu-engine/src/encodable_block/linear/rht_wrapper.rs
# または
git -C ~/Myproject/uzu apply notes/uzu-backup/rht_wrapper.patch
cd ~/Myproject/uzu && cargo build --release
```

server パッチ（thinking 既定 + CORS）を戻す場合:

```bash
cd ~/Myproject/uzu
git apply /Users/kohei/Myproject/kaggleAI/notes/uzu-backup/server-patch.diff
cargo build --release -p cli
```

## モデル再DL

モデル重量が消えた場合のみ。

```bash
hf download trymirai/Qwen3.5-9B-M \
  --revision 55740870dd4bb2a3f28221dfa3e7ab91da95e7b0 \
  --local-dir ~/.cache/uzu-stage \
  --include "model.safetensors" "speculator/model.safetensors"
```

再DL後も `notes/run_uzu_9b.sh` 経由で起動すれば encoding / M3 shape は自動復旧する。

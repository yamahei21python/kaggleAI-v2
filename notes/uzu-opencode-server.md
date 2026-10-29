# uzu をローカル起動モデルにする（opencode + ブラウザUI）

作成: 2026-10-28

## 起動

```bash
cd /Users/kohei/Myproject/kaggleAI
./run-uzu-server.command
```

- `http://127.0.0.1:9931` で待受
- モデルロード完了（`/v1/models` が応答）後に `http://127.0.0.1:8899/llama-chat.html?port=9931` を開く
- Ctrl+C で停止（UI 用 http サーバも同時に落ちる）

> `file://` で開くと Chrome の Private Network Access により
> `127.0.0.1` への fetch が `Failed to fetch` になる。そのため `llama-chat.html` は
> ローカル http で配信する（`python3 -m http.server`）。

## opencode

`opencode-llama-local.json` を `~/.config/opencode/opencode.json` の `provider` に追加。

```bash
opencode --model uzu-local/uzu-stage
```

provider 名は `uzu-local`、モデルIDは `uzu-stage`。
uzu はリクエストの `model` を無視して常に読み込み済みモデルを返すため、ID は表示用。

## 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `UZU_PORT` | 9931 | 待受ポート |
| `UZU_UI_PORT` | 8899 | `llama-chat.html` を配信するポート |
| `UZU_REASONING_EFFORT` | default | リクエストが指定しない場合の thinking 既定。`disabled`/`default`/`low`/`medium`/`high`/`xhigh` |
| `UZU_KEEP_OLLAMA` | 0 | `1` で Ollama 常駐モデルを止めない |
| `UZU_NO_OPEN` | 0 | `1` でブラウザを開かない |

thinking の切替:

```bash
UZU_REASONING_EFFORT=default ./run-uzu-server.command   # ON（既定）
UZU_REASONING_EFFORT=disabled ./run-uzu-server.command  # OFF（速いが多段推論は落ちる）
```

> 実測: マルチターンで「1,2,3,4,5 の最後に 10 を足す」を聞くと
> OFF では `6`、ON では `15`。9B は thinking 無しだと多段の計算を誤る。

ブラウザUI の ⚙️ 設定にも `thinking` 選択がある。`(default)` はサーバ既定に従う。

| UI 選択 | 送信内容 |
|---|---|
| `(default)` | 何も送らない → `UZU_REASONING_EFFORT` に従う |
| `on` | `reasoning_effort: "default"` |
| `off` | `reasoning_effort: "disabled"` |

> uzu 既定は thinking ON。opencode は `reasoning_effort` を送らない仕様のため、
> サーバ側の既定値でしか制御できない。そのため `UZU_REASONING_EFFORT` を追加した。
> リクエストに `reasoning_effort` / `enable_thinking` が入っていればそちらが優先される。

## 適用した uzu ローカルパッチ

`crates/legacy/cli/src/server/` 配下。未コミット。

### 1. thinking 既定値

- `state.rs`: `ServerState.default_reasoning: Option<ReasoningEffort>` を追加
- `runner.rs`: `UZU_REASONING_EFFORT` をパースして `default_reasoning_effort()` で渡す
- `chat_completions.rs`: `build_messages()` がリクエスト未指定時に `default_reasoning` を使う
  - リクエスト明示が優先。既定値由来のエラーは `(from UZU_REASONING_EFFORT)` を付けて返す

### 2. CORS

- `runner.rs`: `CorsFairing`（`Access-Control-Allow-Origin` は Origin をエコー、無ければ `*`）
- `runner.rs`: `#[rocket::options("/<_..>")]` で preflight に 204 を返す
- mount は `/` のみ（`<_..>` が `/v1/...` も拾う。`/v1` にも張ると route collision で起動失敗）

## 検証済み

| 項目 | 結果 |
|---|---|
| `GET /v1/models` | `{"id":"uzu-stage"}` |
| `POST /v1/chat/completions` (stream) | SSE + `[DONE]` |
| opencode 実body（9353 tokens / 12 tools） | 正常応答、`tool_calls` 正常 |
| prefix cache | `cached_tokens: 9355` で 1.33 秒 |
| OPTIONS preflight | 204 + CORS ヘッダ |
| ブラウザUI（http 配信） | `uzu-stage` 表示 → 送信 → `7` 返却 |
| thinking OFF | `reasoning_content` 無し |

## 既知の制約

- モデルは 9B-M のみ。`qwen38-27b` 相当は uzu に無い
- サーバは単一セッション共有。並列リクエストは直列化される
- 起動ロードに約 40 秒。初回 prefill は 9.3k tokens で約 58 秒
- `target/release/cli` を再ビルドしたら `notes/uzu-backup/cli` も更新する

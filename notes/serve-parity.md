# 配信の 2 実装（notebook セル / script 骨格）の差分台帳

配信カーネルは同じ機能を 2 通りで持っている。

| | notebook 版 | script 版 |
|---|---|---|
| 実体 | `kaggle/serve-cells/*.py` + エンジンセル | `kbuild/serve_script.py` の `TEMPLATE` |
| 組み立て | `kbuild/serve_notebook.py`（セル列） | `kbuild/serve_script.py`（1 ファイルに連結） |
| 実行 | セルを上から順に（モジュール直下） | `main()` → `run(procs)` の 1 本道 |
| 共有部分 | `runtime/` の `kc_*` | 同じ `kc_*` |

`runtime/` に寄せてある部分（proxy 起動 / cloudflared 起動 / Gist / 公開待ち /
conf の読み方 / 期限判定）は**同じ関数を呼んでいる**ので、差分にはならない。
差分が出るのは「制御の流れ」——何を監視し、何を証跡に残すか。

## 機能ごとの対応

| 機能 | notebook 版 | script 版 | 状態 |
|---|---|---|---|
| proxy 起動 | `proxy.py` → `kc_start_proxy` | `start_proxy()` → `kc_start_proxy` | 同等 |
| トンネル起動 | `tunnel_cloudflared.py` | `start_cloudflared()` | 同等 |
| Gist 公開 | `gist.py` の `publish_url` | `publish_url()` → `_publish_gist` / `_publish_beacon` | 同等 |
| 公開待ち | `verify.py` 冒頭の `kc_wait_http` | `wait_public()`（失敗しても続行） | ほぼ同等 |
| 疎通判定 | `verify.py` の `_judge`（SKIP は失敗に数えない） | `verify()` 内の `judge` | **同じ契約**（SKIP の扱いを含む） |
| エンジン確認項目 | `engine_checks(judge, retry, url)` | `engine_checks(judge, retry, base, auth, direct)` | 引数が 2 個多い（smoke 用の `direct`） |
| keepalive の延長 | conf の mtime 変化で延長（60 秒周期） | `read_conf()` を 20 秒周期で読む | 同等（周期が違う） |
| **トンネル監視** | 60 秒ごとに `/healthz` を叩き、`tunnel_dead` を数える | **起動時に 1 回だけ**（`wait_public`） | **差あり** |
| **トンネル再起動** | 死んだら 1 回だけ再起動し、新 URL を再公開 | **無い** | **差あり** |
| keepalive の証跡 | `keepalive.txt`（毎分の状態 + 判定） | `result.json` の `stopped_by` のみ | **差あり** |
| 後片付けの証跡 | `cleanup.txt` | ログ出力のみ | 差（軽微） |
| smoke / batch | 無い（別カーネル） | `SMOKE` / `SERVE=False` で分岐 | script だけにある |

## いま効いている差: script 版は公開 URL の失効に気づかない

quick tunnel の URL は**実行中でも失効しうる**（notebook 版はそのために
`TUNNEL_CHECK_EVERY` ごとの `/healthz` と 1 回限りの再起動を持っている）。
script 版は起動時の `wait_public()` 1 回きりなので、

- 途中でトンネルが死んでも `loop()` はエンジンの `poll()` しか見ない
- 結果として `result.json` は `ok: true` のまま、外からは繋がらない
- `url.txt` / Gist / ビーコンは失効 URL を指したまま

本番（`step2-vllm-qwen38-27b-cloudflared-script`）は script 版なので、
これは**運用中の穴**。GPU を使わずに直せる（制御の流れだけ）が、
`loop()` の戻り値の意味が増えるので、直したら 1 往復で確認が要る。

## なぜ 1 本にしていないか

- notebook 版は「セルを上から実行する」形が要求される（Kaggle の notebook は
  セル単位で実行される）。`main()` 1 本道にすると、セル境界と実行順が
  一致しなくなる
- script 版は 1 ファイルに連結するので、`main()` の中で全部を順に呼ぶ形が要る

つまり**制御の流れの共有は構造的に難しい**。共有できるのは関数（`runtime/`）
だけで、そこはもう共有してある。だから parity テストは「関数の同一性」では
なく**振る舞いの同一性**で書く必要がある（例: SKIP の扱い、延長の判定、
失敗時の扱い）。`tests/test_smoke_checks.py` の 2 本はその最初の 2 例。

## 次にやるなら

1. script 版の `loop()` に `/healthz` 監視を足す（`TUNNEL_CHECK_EVERY` 相当）。
   再起動まで入れるかは別判断（再起動は URL が変わるので、再公開も要る）
2. `keepalive.txt` 相当の証跡を script 版にも出す（`save()` に 1 キー足すだけ）
3. 上の 2 つを「振る舞いの parity テスト」で固定する

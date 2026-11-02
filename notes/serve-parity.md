# 配信の 2 実装（notebook セル / script 骨格）の差分台帳

配信カーネルは同じ機能を 2 通りで持っている。

| | notebook 版 | script 版 |
|---|---|---|
| 実体 | `kaggle/serve-cells/*.py` + エンジンセル | `kbuild/serve_script.py` の `TEMPLATE` |
| 組み立て | `kbuild/serve_notebook.py`（セル列） | `kbuild/serve_script.py`（1 ファイルに連結） |
| 実行 | セルを上から順に（モジュール直下） | `main()` → `run(procs)` の 1 本道 |
| 共有部分 | `runtime/` の `kc_*` | 同じ `kc_*` |

`runtime/` に寄せてある部分（proxy 起動 / cloudflared 起動 / ビーコン / 公開待ち /
conf の読み方 / 期限判定）は**同じ関数を呼んでいる**ので、差分にはならない。
差分が出るのは「制御の流れ」——何を監視し、何を証跡に残すか。

## 機能ごとの対応

| 機能 | notebook 版 | script 版 | 状態 |
|---|---|---|---|
| proxy 起動 | `proxy.py` → `kc_start_proxy` | `start_proxy()` → `kc_start_proxy` | 同等 |
| トンネル起動 | `tunnel_cloudflared.py` | `start_cloudflared()` | 同等 |
| URL 配布 | `publish.py` の `publish_url` → `_publish_beacon` | `publish_url()` → `_publish_beacon` | 同等（2026-11-02 に Gist を全廃して 1 本化） |
| 公開待ち | `verify.py` 冒頭の `kc_wait_http` | `wait_public()`（失敗しても続行） | ほぼ同等 |
| 疎通判定 | `verify.py` の `_judge`（SKIP は失敗に数えない） | `verify()` 内の `judge` | **同じ契約**（SKIP の扱いを含む） |
| エンジン確認項目 | `engine_checks(judge, retry, url)` | `engine_checks(judge, retry, base, auth, direct)` | 引数が 2 個多い（smoke 用の `direct`） |
| keepalive の延長 | conf の mtime 変化で延長（60 秒周期） | `read_conf()` を 20 秒周期で読む | 同等（周期が違う） |
| トンネル監視 | 10 分ごとに `/healthz` を叩き、`tunnel_dead` を数える | `loop()` が `TUNNEL_CHECK_EVERY` 周ごとに同じことをする | **同等（2026-11-02 に追加）** |
| トンネル再起動 | 死んだら 1 回だけ再起動し、新 URL を再公開 | `restart_tunnel()` を 1 回だけ | **同等（2026-11-02 に追加）** |
| keepalive の証跡 | `keepalive.txt`（毎分の状態 + 判定） | `save_evidence()` が同じ名前で出す | **同等（2026-11-02 に追加）** |
| 後片付けの証跡 | `cleanup.txt` | `save_evidence()` が同じ名前で出す | **同等（2026-11-02 に追加）** |
| smoke / batch | 無い（別カーネル） | `SMOKE` / `SERVE=False` で分岐 | script だけにある |

## 直した差: script 版も公開 URL の失効に気づく（2026-11-02）

quick tunnel の URL は**実行中でも失効しうる**。notebook 版はそのために
`TUNNEL_CHECK_EVERY` ごとの `/healthz` と 1 回限りの再起動を持っていたが、
script 版は起動時の `wait_public()` 1 回きりで、`loop()` はエンジンの
`poll()` しか見ていなかった。結果として

- 途中で URL が失効しても `loop()` は走り続ける
- `result.json` は `ok: true` のまま、外からは繋がらない
- `url.txt` / ビーコンは失効 URL を指したまま

本番は script 版なので、これは**運用中の穴**だった。次の 3 つを入れて閉じた。

| 追加 | 中身 |
|---|---|
| `loop(procs, url)` | 20 秒周期。トンネルの `poll()` を見て、死んでいたら 1 回だけ張り直す。30 周（= 600 秒 = 10 分）ごとに `_public_health(url)` を叩き、失敗を `data["tunnel_dead"]` に数える |
| `restart_tunnel(procs)` | `pkill -x cloudflared` → `start_cloudflared()` → `publish_url(new_url)`。**公開に失敗しても張り直しは捨てない**（トンネルは生きているため） |
| `save_evidence()` | `save()` のたびに `keepalive.txt`（ヘッダ + 毎周期の行）と `cleanup.txt`（`engine_cleanup` / `leftover`）を書き直す |

`run()` は `loop()` のあと `tunnel_dead >= 2` なら `RuntimeError` にする
（notebook 版の `fail` と同じ）。**「繋がらないのに ok: true」をやめた**のが
この変更の主眼。

周期は notebook 版に合わせた（`TUNNEL_CHECK_EVERY = 600 // LOOP_SLEEP` = 30 周。
notebook は 60 秒 × 10 周 = 600 秒）。

固定したテスト: `tests/test_serve_parity.py`（13 本）。`loop()` を偽の
`time` / `procs` / `_public_health` で exec して、再起動が 1 回だけであること・
2 回目で諦めること・`/healthz` の失敗を数えること・`keepalive.txt` の中身を
見ている。**関数の同一性ではなく振る舞い**で書いてある（制御の流れは
構造的に共有できないため）。

## 関数レベルの重複はもう無い（2026-11-02 実測）

「`runtime/` に寄せればもっと消せるのでは」を測った。TEMPLATE / `serve-cells/` /
`step2-vllm/cells/` / `runtime/` の**トップレベル関数 107 個**を、名前を無視して
本文（コメント・docstring を落とした AST）で総当たり比較した結果、
類似度 0.70 以上の組は **0**。同じ名前でも中身が違う（例: テンプレートの
`wait_http` はエンジンの死を見て諦めるが、`runtime/net.py` の `kc_wait_http` は
単純なリトライ。類似度 0.19）。

つまり**機械的に統合できる重複は残っていない**。差が残っているのは
「制御の流れ」（何を監視し、何を証跡に残すか）だけで、そこは
`tests/test_serve_parity.py` の振る舞いテストで突き合わせるしかない。
次に「共通化しよう」と思ったら、まずこの測り方で 0 組であることを確認すること。

同じ測り方で**呼ばれていない関数**も数えた（2026-11-02）。

| 対象 | 関数 | 参照ゼロ | 中身 |
|---|---|---|---|
| `runtime/` `kbuild/` `cli/` `tests/` ルート | 430 | 5 → **2 を削除** | 残り 3 つは `kaggle/`（生成物）から使われていた |
| `TEMPLATE` の中 | 32 | 2 | `sh` / `wait_http`。どちらも `EXTERNAL_ENGINE_CODE`（LMDeploy / SGLang）から使う |

削除したのは `tests/_cells.py` の `cell_def()`。テンプレート側は 0 件だった。
`runtime/gist.py` の `kc_gh_token()` も同じ理由で消したが、こちらは同日の
**Gist 全廃**でモジュールごと無くなった（`notes/url-publishing.md`）。

## テンプレートの引数は両方向で突き合わせる（2026-11-02）

`TEMPLATE` は 900 行・`$TOKEN` 39 個の `string.Template` で、`render()` が
39 個のキーワードを渡している。この 2 つがずれても、片方向しか怒られない。

| ずれ方 | これまで | いま |
|---|---|---|
| トークンはあるが引数が無い | `KeyError`（どのトークンかは出る） | `ValueError`（同じ） |
| 引数はあるがトークンが無い | **無言で捨てられる** | `ValueError`（名前を出す） |
| `$` の後ろが識別子でない | `Invalid placeholder in string` だけ | 文字位置と周辺を出す |

2 行目が実害の本体。「テンプレートから `$X` を消したのに `render()` に
`X=...` が残る」と、渡しているのに効かない設定が静かに積み上がり、次に触る人が
「なぜ変えても挙動が変わらないのか」を追うことになる。`template_tokens()` /
`_check_tokens()` を `kbuild/serve_script.py` に置き、`render()` の最後で必ず通す。

同じ理由で `flow` も見るようにした。

- 知らない段の名前（`{"publsh": False}`）は `ValueError`。`{**DEFAULT_FLOW, ...}`
  は余分なキーを黙って受けるので、綴り間違いは「既定のまま動く」になる
- 値が `bool` 以外なら `TypeError`。`INSTALL = $INSTALL` は素の式として埋め込む
  ので、`"false"` を渡すとカーネル側が `INSTALL = false` になり **ビルドは通って
  GPU を掴んでから `NameError`** になる

生成物は 1 バイトも変わらない（`git status` が `kbuild/serve_script.py` だけ）。
テストは `tests/test_serve_script_guards.py` に 8 本。

## まだ残っている差

- ~~Gist への書き込みだけ二重実装~~ → **2026-11-02 に解消**。Gist を全廃し、
  両実装とも `kc_beacon_publish` を呼ぶだけになった（`notes/url-publishing.md`）
- `loop()` の周期（20 秒 / 60 秒）と、`keepalive.txt` の行の形（notebook は
  `cloudflared=alive | GPU: …`、script は `tunnel=cloudflared:alive | GPU: …`）。
  名前は同じだが中身の書式は違う
- notebook 版は `/tmp/public_url.txt` も書き換える（セル間の受け渡し）。
  script 版は `url.txt`（`URL_FILE`）と `data["url"]` だけ
- smoke / batch の分岐は script だけにある（notebook 版は別カーネル）


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

1. `loop()` の周期を揃える（script を 60 秒にするか、notebook を 20 秒にするか）。
   いまは「10 分ごと」という**意味**だけを揃えてある
2. `keepalive.txt` の行の書式を揃える（同じ列名で grep できるようにする）
3. smoke / batch の分岐を notebook 側にも持たせるか、逆に script から外すか
   （いまは script だけが 2 形態を持つ）

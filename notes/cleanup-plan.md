# リポジトリ整理・リファクタリング案（2026-11-01 時点）

**P0 は 2026-11-01 15:15 に実施済み**（§6 実施ログ）。P1〜P3 は提案のまま。
実測は 2026-11-01 14:50〜15:00、ブランチ `refactor/p0-and-kbuild` の作業ツリーで取得した。

`refactor-plan.md`（1,328 行・第 1〜4 次の追記型履歴）とは役割を分ける。
**「いま何が残っているか」はこのファイル、「なぜそう決めたか」は `refactor-plan.md`。**
将来的に `refactor-plan.md` は `archive/refactor-log-2026-10.md` へ降格させる想定（P3）。

## 1. 実測値

| 対象 | 現状（2026-11-01 14:50 実測） |
|---|---|
| ルート直下 | 43 エントリ（ドット付き 13 / 通常 30）→ **P0 後 35**（ドット付き 13 / 通常 22） |
| ルートの残骸 | `blat` とだけ書かれた 4 バイトファイル **7 個**（`2a0h0pu3` `5uaggo2j` `6m3dzs6x` `882fpugu` `bwablv6y` `edecs1on` `ee_53j0s`）→ **P0 で退避済み** |
| git | 未コミット **137 件** = M 53 / ?? 72 / RM 7 / D 5。未追跡 130 ファイル・6.6 MB |
| `kaggle/` | 57 エントリ（dir 33）。`build_*.py` 19 本 4,181 行 / py 101 / ipynb 63 / `kernel-metadata*.json` 100 |
| 台帳 `kbuild/spec.py` | **47 カーネル**（active 14 / semi **31** / retired 2） |
| ディスク | `kaggle` 4.2 G（`wheelhouse-files` 3.0 G + `whl-patch` 1.1 G）/ `datasets` 1.4 G / `.wheel` 1.1 G / `notes` 70 M / `.git` 32 M |
| `notes/` | 53 ファイル・md 7,404 行。`refactor-plan.md` 84 K（1,328 行） |
| 空・死んだ置き場 | `models/` は空。`datasets/gguf` `datasets/orcarouter` は `dataset-metadata.json` のみ（重い実体 17 GB は撤去済み） |
| 追跡下の巨大ファイル | `notes/uzu-backup/cli` **52 MB** + `tokenizer.json` **19 MB**（13 ファイルが HEAD に残存） |

## 2. 散らかりの正体

1. **未コミット 137 件** — 最大のリスク。「動く状態」が git に存在しない期間が長く、
   壊れたときに戻せない。他の整理より先に返済する。
2. **`kaggle/` に 5 種類が同居** — 生成物（notebook / script）、エンジン固有セル、
   共有セル、ローカル巨大資産、退役スナップショット。しかも
   `exl3-serve/`（ソース `chat_server.py`）と `exl3-serve-kernel/`（生成物）、
   `llama-server/`（cells）と `llama-server-ngrok/`（notebook）のように
   **名前で現役／生成物を判別できない組**がある。
3. **台帳が墓場化** — 47 台中 31 が `semi`。probe 系を `retired` に落とさないと
   `python3 -m cli list` が読めない。
4. **計画文書が履歴になった** — `refactor-plan.md` は「残タスク」を答える文書ではなくなった。

## 3. 提案

### P0 — 今日・判断不要・30 分 ✅ 実施済み

- ~~`blat` 7 個を削除。~~ → **`/tmp/kc-blat-junk/` へ退避**（削除ではなく移動。可逆）。
  **発生源はリポジトリ外**と判明（§6.1）。11/01 05:00 を最後に再発していない。
- ~~`models/`（空）を削除。~~ → 実施（`rmdir`。`.gitignore` 済みだった）。
- ~~`datasets/{gguf,orcarouter}` の扱いを決定~~ → **metadata は追跡する**方針で決着。
  `datasets/` を丸ごと無視していたため `dataset-metadata.json` が git から見えなかった。
  4 段パターン（`datasets/*` → `!datasets/*/` → `datasets/*/*` → `!datasets/*/dataset-metadata.json`）
  に直し、metadata 5 件を追跡。**重い実体は無視のまま**。
- ~~`notes/uzu-backup/` の 52 MB `cli` と 19 MB `tokenizer.json` を `git rm --cached`。~~
  → 実施。追跡 13 → 11 件。**実体は無傷**（ランチャーが参照する現役ディレクトリ）。
  履歴は書き換えていない（`filter-repo` 不使用）。`.gitignore` に 2 行追加。

### P1 — 今週・未コミット 137 件の返済 ✅ 実施済み（5 コミット）

当初案は 6 コミットだったが、**5 に統合した**。理由は §6.5。

| 実際のコミット | 内容 | 当初案との差 |
|---|---|---|
| `e805359` refactor | kbuild 4 本 + cli/ + serve-cells + cells + runtime + ビルダー 19 本 + tests 13 本 + ルートの薄い入口 | 案の 1〜3・5 を統合 |
| `b8637bb` build | script 生成物 13 変種 + `.gitattributes` | 案の 4 |
| `e5c6b7e` build | ノートブック 13 本 + metadata 2 件 | 案に無し（増えた分） |
| `a688cf8` chore | ci.yml / README / check_notebooks / ruff.toml / 生成 2 本 | 案の 6 の一部 |
| `61d26e4` docs | notes 15 本 | 案の 6 |

**判断していた「script 生成物を commit するか」は commit で決着。**
`kaggle/check_scripts.py` は「commit 済み」と再生成結果をバイト比較するので、
追跡しないと検査が空振りする。差分ノイズは `.gitattributes` の `-diff` で抑えた。

結果: **未コミット 137 → 0。** `python3 -m cli check` は 6 段階すべて緑、250 テスト通過。

### P2 — 構造（1 コミットずつ、`python3 -m cli build` と check 系をゲートに）

- **Step 1: `kaggle/build/` へ `build_*.py` 19 本。**
  変更は `cli/build.py` のパス組み立て 1 行（`ROOT/"kaggle"/f"{builder}.py"`）と、
  各ビルダー冒頭の `sys.path.insert(0, Path(__file__).resolve().parent.parent)` を
  `parents[2]` へ（19 行・機械的）。`import kcconfig` は `parents[2]` が効けばそのまま通る。
  これだけで `kaggle/` 直下が 57 → 約 38 エントリになり、見た目の効果が一番大きい。
- **Step 2: `kaggle/tools/` へ** `check_notebooks.py` `check_scripts.py` `_regen.py`。
- **Step 3: `kaggle/cells/{shared,ollama,vllm,llama_server,exl3}/` へ** `serve-cells/`、
  各 `*/cells/`、`exl3-serve/chat_server.py` を集約。ここで命名の紛らわしさが消える。
- **Step 4（保留推奨）: 生成物を `kaggle/kernels/<slug>/` へ。**
  `spec.out` 47 件の書き換えだけで済むが、15 dirs が未追跡で `git mv` が効かず、
  得られるのは見た目だけ。Step 1〜3 が終わってから判断する。

### P3 — 台帳と docs

- probe 系を `retired` に落とし、dir を `kaggle/archive/` へ。`cli list` の可読性が戻る。
  対象は `tiny-llama-probe` `tiny-ollama-probe` `vllm-flag-probe` `vllm-whl-probe`
  `vllm-install-bench` `vllm-io-bench` `sglang-install-probe` の **7 本**。
  > ⚠ **`vllm-playground` は退役させない。** この案を書いた時点では未使用だったが、
  > いまは実験の置き場として現役（`notes/vllm-startup-speedup.md` §17.2）。
  > 本番と別 id を持つことに意味があるので、`retired` にすると実験が本番履歴を汚す
  > 元の状態に戻る。
- `refactor-plan.md` を **現役の残タスク（約 100 行）+ `archive/refactor-log-2026-10.md`（履歴）**
  に分割。第 1〜4 次の「進捗」節は全部履歴側。
- `notes/` のコード 3 点（`run_uzu_9b.sh` `prompt_sweep.py` `Modelfile.qwen35-local`）を
  `scripts/` へ。notes は判断とランブックだけにする。

### 容量（5.2 GB）の扱い

`kaggle/wheelhouse-files` 3.0 G / `kaggle/whl-patch` 1.1 G / `.wheel` 1.1 G は
**すべて gitignore 済みのローカル資産**。ただし repo 内にあり、外へ出すと
`-p` の相対パス前提（`refactor-plan.md` E-2 の判断）が壊れる。
**移動するより `python3 -m cli clean` を作って「消して再生成」を選べるようにする**方が安全。
再取得手順は `kaggle/whl-patch/README.md` にある。

## 4. やらないこと

- カーネル id の変更（Secrets / dataset_sources / 実行履歴が id に紐づく）。
- 33 dirs の一括 `git mv`（レビューできない。Step 単位で分ける）。
- notebook の削除（GPU smoke 未実施。`gpu-smoke-plan.md` 参照）。
- `git filter-repo` による履歴書き換え（`uzu-backup` の 71 MB は HEAD から外すだけにする）。
- `runtime/engines/{vllm,llama_server}.py` の抽象化（起動ラダーは共有しない方針を維持）。

## 5. 決めること（チェックリスト）

- [x] `blat` の発生源（CWD に temp を作っているツール）は何か
      → **リポジトリ外**。コードに `blat` は 1 箇所も無い（§6.1）
- [x] `datasets/{gguf,orcarouter}` の metadata を残すか捨てるか
      → **残す**。`datasets/*` の 4 段パターンで metadata 5 件のみ追跡
- [ ] script 生成物 15 dirs を commit するか（推奨: commit + `.gitattributes -diff`）
- [ ] P2 Step 1 を今週やるか（ビルダー 19 本の `parents[2]` 書き換えを含む）
      → **GPU 実験が終わるまで保留**（§6.2）
- [ ] P2 Step 4（生成物の移動）をやるか、やらないか

## 6. 実施ログ（2026-11-01 15:15）

2 コミット。**パス指定コミットの落とし穴を踏んだので記録しておく。**

### 6.1 `blat` の発生源＝リポジトリ外

`blat` でリポジトリ全体を検索しても、ヒットするのは残骸 7 個と `notes/` の記述だけ。
**コードには 1 箇所も無い。** ファイル名は `mkstemp` の 8 文字サフィックス形式、
中身は 4 バイトの `blat`、生成間隔は 6〜10 分（04:24 / 04:30 / 04:36 / 04:44 / 04:54 / 05:00）。
書き込み可否を試す外部ツールの跡と見られる。

- 検証: こちらのコマンド実行では**再生成されない**（前後で個数不変）。
- 11/01 05:00 を最後に 10 時間再発していない。
- → 発生源を潰す対象がリポジトリ内に無いので、**退避して終わり**にした。
  再発したら「どの外部ツールが動いた時刻か」で絞るのが早い。

### 6.2 パス指定コミットは「作業ツリーの内容を再ステージ」する

`git rm --cached` で index から外したあと

```sh
git commit -m "..." -- .gitignore notes/uzu-backup/cli notes/uzu-backup/tokenizer.json
```

とやると、**`-- <path>` が作業ツリーの内容を採用する**ため、実体が残っている
`cli` / `tokenizer.json` が**そのまま再ステージされ、削除が取り消された**。
1 回目のコミットは `.gitignore` だけになり、追跡数は 13 に戻っていた。

正しくは「**実体を一時的に外へ出してから**パス指定コミット → 戻す」:

```sh
mv notes/uzu-backup/cli /tmp/kc-uzu-cli
mv notes/uzu-backup/tokenizer.json /tmp/kc-uzu-tokenizer.json
git rm --cached notes/uzu-backup/cli notes/uzu-backup/tokenizer.json
git add .gitignore datasets/
git commit -m "..." -- .gitignore datasets notes/uzu-backup/cli notes/uzu-backup/tokenizer.json
mv /tmp/kc-uzu-cli notes/uzu-backup/cli
mv /tmp/kc-uzu-tokenizer.json notes/uzu-backup/tokenizer.json
```

（`git commit` をパス指定なしで打つ手もあるが、当時 7 件の rename が既に
ステージされていたため、無関係な変更が混ざる。それを避けるにはこの方法。）

### 6.3 `datasets/` の丸ごと無視は metadata を殺す

`datasets/` の 1 行で無視すると、`datasets/warm-cache/dataset-metadata.json` が
**git から完全に見えなくなる**（`git ls-files datasets/` が空になる）。
git は「除外したディレクトリの中は `!` で復活できない」ため、`!datasets/*/dataset-metadata.json`
だけでは効かない。ディレクトリ自体を先に復活させる 4 段が必要:

```gitignore
datasets/*
!datasets/*/
datasets/*/*
!datasets/*/dataset-metadata.json
```

### 6.4 保留にしたもの（理由つき）

- **P2 Step 1（`build_*.py` 19 本を `kaggle/build/` へ）** — `cli/build.py` のパス組み立てと
  19 本の `sys.path.insert` を同時に書き換える。**GPU 実験（playground）が進行中で、
  ビルドが壊れると push が失敗しクォータを無駄にする。** 実験が終わってから。
- **P3 の `notes/` コード 3 点を `scripts/` へ** — `run_uzu_9b.sh` は
  `notes/` 内 4 本の md と `uzu-backup/README.md` から参照されている。移動すると
  リンク切れの修正が 5〜6 箇所に及び、得るものが見た目だけ。P2 と一緒にやる。
- **`git index.lock` の退避** — 10/30 14:16 の古いロックが残っており、書き込み系の
  git 操作が全部失敗していた。`ps` が使えない環境なので時刻を根拠に判断し、
  `/tmp/kc-index.lock.bak` へ退避（削除はしていない）。**心当たりが無ければ
  このファイルは捨ててよい。**

### 6.5 なぜ 6 コミットではなく 5 にしたか

案の 1〜3・5（kbuild / runtime / ビルダー / tests）は**互いに参照し合っていて、
単独では緑にならない**。`kbuild/spec.py` を入れずにビルダーだけ入れても import が
通らず、`tests/test_spec.py` を入れずに spec.py を入れても意味がない。分割すると
「レビューしやすいが、どの時点でも壊れている」コミットが 4 つ並ぶ。

そこで **ソース一式を 1 コミット**（113 ファイル・+14,261 / −4,560）にし、
生成物・ノートブック・設定・docs を別コミットに分けた。レビューは「ソースの差分」と
「生成物の差分（`.gitattributes` で `-diff` 済み）」で見分けられる。

分割数を増やす価値があるのは、**将来 `git bisect` を回したくなったとき**。
そのときは「ソースが緑になる地点」で切る必要があるので、今の 1 本では足りない。
ただしその作業は履歴の書き換えになるので、必要になったときにやる。

### 6.6 骨格のバグを 1 件直した（P1 の副産物）

`git add` の途中で `kbuild/serve_script.py` が未追跡だと気づき、中身を確認したら
**exl3 の ERROR の原因**だった（§`engine-rollout.md`）。骨格の既定
`engine_checks(judge, retry, base, auth=None)` が `direct` を受け取らないのに、
`verify()` が 5 引数で呼んでいた。`checks_code` を持つ 3 エンジンは自前の 5 引数版で
上書きするので露見せず、**既定を使う exl3 だけが踏んだ**。

既存テストが 4 引数 + kwargs で呼んでいたのも見逃しの原因なので、
「呼び出し側」と「受け口」の両方を固定する回帰テストを 2 本足した。

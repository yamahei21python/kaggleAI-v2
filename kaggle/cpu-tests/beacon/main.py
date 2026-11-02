# ============================================================
# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）
# proxy / tunnel / keepalive / verify の配管をここに集約する。
# ============================================================
# ── runtime/beacon.py ─────────────────────────────────────
"""ビーコンカーネル（Kaggle のカーネルを掲示板として使う）。

Gist 方式の代わりに、**カーネルのコードそのもの**を公開 URL の置き場にする。

- 書き手: 配信カーネルの中から `kernels_push` で小さなカーネルを更新する。
  認証は Kaggle がカーネルへ注入するもの（Secrets 不要）。
- 読み手: `kaggle kernels pull` でコードを引いて `URL` を読む。

なぜ Gist をやめたか: Kaggle の Secret は**カーネルごとに紐付ける**必要があり、
未紐付けの `get_secret()` は `ConnectionError` を返す（通信障害に見える）。
専用アカウント運用では、カーネルを増やすたびに UI 作業が要り、トークンも
個人アカウントのものが残る。ビーコンは**アカウント・Secrets・外部サービスを
増やさずに**同じことができる。経緯は `notes/url-publishing.md`。

この層の方針は旧 Gist 層から引き継いだ: **失敗しても配信を止めない。ただし理由は必ず返す。**
2026-11-01 に「Gist が 10/28 から一度も更新されていないことに誰も気づかない」
という失敗をしているので、`(ok, detail)` を返さない関数はここに置かない。

`kc_` 接頭辞は生成コードの名前空間で衝突しないため（runtime/__init__.py）。
"""
import json
import os
import shutil
import sys
import tempfile
import time

# 掲示板の中身。**この文字列がそのまま beacon カーネルの main.py になる。**
# 値は repr で埋めるので、URL に引用符が混ざっても壊れない。
_BEACON_BODY = '''"""公開 URL のビーコン（掲示板）。

このカーネルは**実行しない**。コードそのものが受け渡しの場所で、
配信カーネルが `kernels_push` でこのファイルを書き換える。
読む側は `kaggle kernels pull` して `URL` と `UPDATED` を読む。

`STAGE` は `serving` / `probe` など。**`probe` は本物ではない**ので、
読む側は `probe` を捨てること（テスト URL を掴む事故を防ぐ）。
`UPDATED` は発行時刻の epoch 秒。古い値を掴んでいないかの判定に使う。
"""

URL = {url!r}
STAGE = {stage!r}
UPDATED = {updated!r}
'''


def kc_beacon_slug(slug=""):
    """`owner/name` を解決する（引数 → 環境変数）。無ければ空文字。"""
    s = (slug or "").strip()
    if s:
        return s
    return (os.environ.get("KC_BEACON_SLUG", "") or "").strip()


def kc_kaggle_path(entries=None):
    """`import kaggle` を壊すパスを取り除いた `sys.path` を返す。

    リポジトリ直下には `kaggle/`（カーネルのビルド元）がある。これが
    **インストール済みの `kaggle` パッケージを隠す**ので、ローカルで
    `cli beacon` を叩くと `No module named 'kaggle.api'` で死ぬ（実測）。
    カーネル内には `kaggle/` ディレクトリが無いので、ここは素通りする。

    判定は「`kaggle/` があり、かつ `kaggle/api` が無い」パスだけを外す。
    `kaggle/api` を持つ本物の候補は残すので、site-packages は消えない。
    """
    src = list(sys.path if entries is None else entries)
    keep = []
    for p in src:
        base = p or "."
        try:
            shadow = (os.path.isdir(os.path.join(base, "kaggle"))
                      and not os.path.isdir(os.path.join(base, "kaggle", "api")))
        except OSError:                       # 権限の無いパスは触らない
            shadow = False
        if not shadow:
            keep.append(p)
    return keep


def kc_kaggle_api():
    """`KaggleApi` クラスを返す。無ければ例外を投げる（呼び出し側が拾う）。

    普通に import して駄目だったときだけ `kc_kaggle_path` で探し直す。
    一度失敗した `kaggle` は名前空間パッケージとして `sys.modules` に
    残っているので、それも捨てないと再試行が同じ場所へ行く。
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi
    except Exception:
        saved = list(sys.path)
        try:
            sys.path[:] = kc_kaggle_path(saved)
            sys.modules.pop("kaggle", None)
            sys.modules.pop("kaggle.api", None)
            from kaggle.api.kaggle_api_extended import KaggleApi
            return KaggleApi
        finally:
            sys.path[:] = saved


def kc_beacon_meta(slug):
    """beacon カーネルの kernel-metadata.json を組み立てる。

    **実行しない掲示板**なので GPU もインターネットも要らない。非公開にするのは、
    URL が「知っていれば誰でも入れる入口」だから（Gist と同じ扱い）。
    """
    name = slug.rsplit("/", 1)[-1]
    return {
        "id": slug,
        "title": name,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def kc_beacon_body(url, stage="", updated=None):
    """掲示板の main.py の中身。"""
    return _BEACON_BODY.format(
        url=url, stage=stage, updated=time.time() if updated is None else updated)


def kc_beacon_publish(url, stage="", slug="", timeout=180):
    """掲示板を更新する。戻り値は `(ok, detail)`。

    `detail` は成功なら `"pushed v3 (platform)"` のような内訳、失敗なら理由。
    **例外は投げない**（配信を止めない）。呼び出し側は成否を `result` に残す。
    """
    s = kc_beacon_slug(slug)
    if not s:
        return False, "beacon の slug が無い（KC_BEACON_SLUG か引数で渡す）"
    if "/" not in s:
        return False, f"slug は owner/name の形で渡す: {s!r}"
    try:
        KaggleApi = kc_kaggle_api()
    except Exception as e:
        return False, f"kaggle パッケージが無い（{type(e).__name__}: {e}）"

    d = tempfile.mkdtemp(prefix="beacon-")
    try:
        with open(os.path.join(d, "main.py"), "w", encoding="utf-8") as f:
            f.write(kc_beacon_body(url, stage))
        with open(os.path.join(d, "kernel-metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(kc_beacon_meta(s), f, indent=2, ensure_ascii=False)

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            return False, (f"Kaggle 認証が無い（{type(e).__name__}: {e}）"
                           " — カーネル外からは ~/.kaggle/kaggle.json が要る")
        try:
            resp = api.kernels_push(d)
        except SystemExit as e:                      # kaggle は終了コードで返す
            return False, f"kernels_push が終了した（SystemExit {e.code}）"
        except Exception as e:
            return False, f"kernels_push 失敗（{type(e).__name__}: {e}）"
        ver = getattr(resp, "version_number", None)
        return True, f"pushed v{ver} ({s})"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def kc_beacon_parse(text):
    """掲示板のソースから `url` / `stage` / `updated` を取り出す。

    `exec` しない。中身は自分で書いたものだが、読む側（VM）にとっては
    ネットワーク越しの入力なので、評価せずに読む。
    """
    out = {"url": "", "stage": "", "updated": None}
    keys = {"URL": "url", "STAGE": "stage", "UPDATED": "updated"}
    for line in (text or "").splitlines():
        s = line.strip()
        for key, field in keys.items():
            if not s.startswith(key + " ="):
                continue
            raw = s.split("=", 1)[1].strip()
            if field == "updated":
                try:
                    out[field] = float(raw)
                except ValueError:
                    out[field] = None
            else:
                out[field] = raw.strip("\"'")
    return out


def kc_beacon_is_fresh(data, since, slack=5.0):
    """`since` より後に発行されたものだけを本物とみなす。

    2026-11-01 に**古いテスト URL を掴む**事故が出ている。push の前後で
    スタンプが変わったことを確かめないと、「まだ前の値」を新しい値として
    読んでしまう。`slack` は VM と Kaggle の時計のずれぶん。
    """
    u = (data or {}).get("updated")
    if not isinstance(u, (int, float)):
        return False
    return u >= (float(since) - float(slack))


def kc_beacon_pull_cli(slug, dest, exe="", timeout=180):
    """`kaggle kernels pull` を**CLI で**実行する。戻り値は `(ok, detail)`。

    Python パッケージの `kaggle` が import できない環境のための逃げ道。
    手元の python3（3.13）には入っておらず、`kaggle` CLI だけが 3.9 の
    user site に入っている、という状態で実際に踏んだ（`cli beacon` が
    `No module named 'kaggle'` で死んでいた）。カーネル内では使われない。
    """
    import subprocess                      # 束ねたときの F811 を避けて局所 import
    cmd = [exe or shutil.which("kaggle") or "", "kernels", "pull", slug, "-p", dest]
    if not cmd[0]:
        return False, "kaggle パッケージも kaggle CLI も見つからない"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return False, f"kaggle CLI の起動に失敗（{type(e).__name__}: {e}）"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or [""]
        return False, f"kaggle CLI が {p.returncode} で終了（{tail[0][:200]}）"
    return True, "kaggle CLI"


def kc_beacon_pull(slug="", dest=None, timeout=180, exe=""):
    """掲示板を引いて読む。戻り値は `(ok, detail, data)`。

    `data` は `kc_beacon_parse` の結果（失敗なら空）。
    `exe` は CLI 経由に落ちたときに使う `kaggle` のパス（`cli push` と同じ
    解決順を呼び出し側が渡す）。空なら PATH から探す。
    """
    s = kc_beacon_slug(slug)
    if not s:
        return False, "beacon の slug が無い", {}
    d = dest or tempfile.mkdtemp(prefix="beacon-pull-")
    try:
        # まず Python パッケージ。駄目なら CLI に落ちる（理由は捨てない）。
        why_api = ""
        try:
            KaggleApi = kc_kaggle_api()
        except Exception as e:
            why_api = f"{type(e).__name__}: {e}"
        else:
            try:
                api = KaggleApi()
                api.authenticate()
                api.kernels_pull(s, path=d)
                why_api = ""
            except SystemExit as e:          # kaggle は終了コードで返す
                why_api = f"kernels_pull が終了した（SystemExit {e.code}）"
            except Exception as e:
                why_api = f"kernels_pull 失敗（{type(e).__name__}: {e}）"
        if why_api:
            ok, detail = kc_beacon_pull_cli(s, d, exe=exe, timeout=timeout)
            if not ok:
                return False, f"API でも CLI でも引けなかった（API: {why_api} / CLI: {detail}）", {}
        # 引いたファイル名は版によって変わる（main.py / <name>.py）ので、
        # 中身で選ぶ。UPDATED を含む .py が掲示板。
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                with open(os.path.join(root, fn), encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
                if "UPDATED" in text:
                    data = kc_beacon_parse(text)
                    return True, f"pulled {fn}", data
        return False, "pull した中に掲示板（UPDATED を含む .py）が無い", {}
    finally:
        if dest is None:
            shutil.rmtree(d, ignore_errors=True)



#!/usr/bin/env python3
"""ビーコンの往復を CPU だけで確かめる（GPU 不要・無料）。

書く → 読み返す → 鮮度判定、までを 1 回で見る。失敗しても例外で落とさず
result.json に理由を残す（見るのは人間なので、スタックトレースより
「どの段で何が返ったか」のほうが速い）。
"""
import json  # noqa: F811 — 上の束も import している
import os  # noqa: F811 — 同上。単体で読めるように本文にも残す
import sys  # noqa: F811 — runtime/beacon.py が kc_kaggle_path で使う
import time  # noqa: F811 — 上の束（runtime/beacon.py）も import している
import traceback

WORK = "/kaggle/working"
RESULT = WORK + "/result.json"
SLUG = "yamahei21python/serve-url-beacon-test"
PROBE_URL = "https://beacon-probe.invalid"


def main():
    data = {"ok": False, "stage": "start", "python": sys.version.split()[0],
            "slug": SLUG}
    try:
        # ── 1〜3: 書いて、読み返す ──
        data["stage"] = "publish"
        t0 = time.time()
        ok, detail = kc_beacon_publish(PROBE_URL, stage="probe", slug=SLUG)
        data["publish"] = {"ok": ok, "detail": detail}
        print(f"publish: ok={ok} {detail}")
        if not ok:
            data["advice"] = ("push できない。カーネル内の注入認証が効いているか、"
                              "slug の owner がこのカーネルの持ち主かを確認する")
            return 1

        data["stage"] = "pull"
        ok, detail, got = kc_beacon_pull(SLUG)
        data["pull"] = {"ok": ok, "detail": detail, "data": got}
        print(f"pull: ok={ok} {detail} {json.dumps(got, ensure_ascii=False)}")
        if not ok:
            data["advice"] = "push は通ったのに pull が失敗した（権限か反映待ち）"
            return 1

        # ── 4: 中身と鮮度 ──
        data["stage"] = "verify"
        data["url_matches"] = got.get("url") == PROBE_URL
        data["stage_matches"] = got.get("stage") == "probe"
        # push の**前**のスタンプで判定 → 本物。1 時間後で判定 → 偽物。
        data["fresh_now"] = kc_beacon_is_fresh(got, t0)
        data["fresh_later"] = kc_beacon_is_fresh(got, t0 + 3600)
        print(f"url一致={data['url_matches']} stage一致={data['stage_matches']}"
              f" 鮮度(now)={data['fresh_now']} 鮮度(+1h)={data['fresh_later']}")

        # ── 5: 陰性対照。失敗が理由つきで返らなければ、成功判定も信用できない ──
        data["stage"] = "negative"
        nok, ndetail = kc_beacon_publish("x", slug="")
        data["negative"] = {"ok": nok, "detail": ndetail}
        data["negative_ok"] = (not nok) and bool(ndetail)
        print(f"negative: ok={nok} {ndetail}")

        data["ok"] = (data["url_matches"] and data["stage_matches"]
                      and data["fresh_now"] and not data["fresh_later"]
                      and data["negative_ok"])
        if not data["ok"]:
            data["advice"] = "往復はしたが判定が合わない（鮮度・stage の実装を疑う）"
        else:
            print("OK: ビーコンの往復と鮮度判定が通った")
    except Exception as e:
        data["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        os.makedirs(WORK, exist_ok=True)
        with open(RESULT, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("RESULT: " + json.dumps(data, ensure_ascii=False))
    return 0 if data.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

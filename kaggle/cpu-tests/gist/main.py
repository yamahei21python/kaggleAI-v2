# ============================================================
# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）
# proxy / tunnel / keepalive / verify の配管をここに集約する。
# ============================================================
# ── runtime/gist.py ───────────────────────────────────────
"""Gist への進捗書き込み（生成コードへ埋め込む）。

Kaggle の UI ログは長大で貼りにくく、`kaggle kernels output` は**完了まで
何も返さない**。起動に失敗した回は「どこで止まったか」を外から見る手段が
要るので、待機中は 30 秒ごとに Gist へ流している。

ところが 2026-11-01 の `vllm-playground` は Gist を 10/28 から一度も
更新していなかった。理由（トークン無し / 401 / 404）はどこにも残らず、
外からは「何も起きていない」と区別がつかなかった。**黙って落ちるのが
一番高い**ので、この層は必ず理由を返す。

`kc_` 接頭辞は生成コードの名前空間で衝突しないため（runtime/__init__.py）。
"""
import json
import os
import time
import urllib.error
import urllib.request

_GIST_API = "https://api.github.com/gists/"


def kc_gh_token():
    """GITHUB_TOKEN を env → Kaggle Secrets の順で解決する。無ければ空文字。"""
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return (UserSecretsClient().get_secret("GITHUB_TOKEN") or "").strip()
    except Exception:
        return ""


def kc_gist_write(gist_id, filename, text, token=None, timeout=10):
    """Gist の 1 ファイルを更新する。戻り値は `(ok, detail)`。

    `detail` は**失敗の理由**（成功なら空文字）。呼び出し側が 1 回だけ声に
    出すために使う。HTTP の本文も拾うのは、401（トークン不正）と
    404（Gist ID 違い / 権限なし）を取り違えないため。

    `token` を明示できるのはテストの口。省略時は env → Secrets の順で解決する。
    """
    if not gist_id:
        return False, "GIST_ID が空（settings の GIST_ID_DEFAULT も未設定）"
    tok = kc_gh_token() if token is None else token
    if not tok:
        return False, "GITHUB_TOKEN が無い（Secrets 未登録、または gist 権限なし）"
    body = json.dumps({"files": {filename: {"content": text}}}).encode()
    req = urllib.request.Request(
        _GIST_API + gist_id, data=body, method="PATCH",
        headers={"Authorization": "Bearer " + tok,
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True, ""
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            msg = ""
        return False, f"HTTP {e.code}: {msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def kc_gist_selftest(gist_id, filename="vllm_status.txt", token=None,
                     timeout=10):
    """PATCH して GET で**読み返す**。往復が通ったときだけ ok=True。

    書けたつもりで読めない（別 Gist を指している / 権限が片方だけ）を
    分けるため、書き込みだけで判定しない。CPU だけで完結するので、
    GPU を使う前に Kaggle のコンテナから 1 回走らせておく。
    """
    stamp = "selftest " + time.strftime("%Y-%m-%d %H:%M:%S")
    tok = kc_gh_token() if token is None else token
    out = {"ok": False, "gist_id": gist_id, "filename": filename,
           "token_present": bool(tok), "stamp": stamp,
           "write_ok": False, "write_detail": "", "read_ok": False,
           "read_detail": "", "matched": False, "files": []}
    ok, detail = kc_gist_write(gist_id, filename, stamp + "\n",
                              token=tok, timeout=timeout)
    out["write_ok"], out["write_detail"] = ok, detail
    if not ok:
        return out
    req = urllib.request.Request(
        _GIST_API + gist_id,
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read() or b"{}")
        out["read_ok"] = True
        files = d.get("files") or {}
        out["files"] = sorted(files)
        content = (files.get(filename) or {}).get("content") or ""
        out["matched"] = stamp in content
        if not out["matched"]:
            out["read_detail"] = (f"{filename} が読めない、または中身が違う"
                                  f"（Gist 内: {sorted(files)}）")
    except urllib.error.HTTPError as e:
        out["read_detail"] = f"HTTP {e.code}"
    except Exception as e:
        out["read_detail"] = f"{type(e).__name__}: {e}"
    out["ok"] = out["read_ok"] and out["matched"]
    return out



#!/usr/bin/env python3
"""Gist への書き込みを CPU だけで往復確認する（GPU 不要・無料）。

やること:
  1. GITHUB_TOKEN / GIST_ID が解決できるか
  2. PATCH で `vllm_status.txt` を書けるか
  3. GET で**読み返して**一致するか
  4. 存在しない Gist へ書いて、**失敗が失敗として返る**か（陰性対照）

2 までで満足しないのは、書けたつもりで別の Gist を見ている事故を分けるため。
4 を入れるのは、元の不具合が「失敗しても何も言わない」だったから——
成功だけ確かめても、失敗経路が沈黙したままなら同じ穴が残る。

失敗しても例外で落とさず result.json に理由を残す（見るのは人間なので、
スタックトレースより「どの段で何が返ったか」のほうが速い）。
"""
import json  # noqa: F811 — 上の束（runtime/gist.py）も import している
import os  # noqa: F811 — 同上。単体で読めるように本文にも残す
import sys
import traceback

WORK = "/kaggle/working"
RESULT = WORK + "/result.json"
STATUS_FILE = "vllm_status.txt"
# セル（03_vllm_serve.py）と同じ既定値。Secrets の GIST_ID は読まない——
# 本番が読んでいないものを読むと、probe だけ通って本番で落ちる。
GIST_ID_DEFAULT = "2b44ac8cb115012a31d3c46661af933f"
# 誰の Gist でもない id。書き込みが 404 で**返る**ことの確認にだけ使う。
BOGUS_GIST = "0" * 32


def main():
    data = {"ok": False, "stage": "start", "python": sys.version.split()[0]}
    try:
        data["stage"] = "resolve"
        tok = kc_gh_token()
        gist_id = (os.environ.get("GIST_ID") or GIST_ID_DEFAULT or "").strip()
        data["token_present"] = bool(tok)
        data["gist_id"] = gist_id
        print(f"token={'あり' if tok else 'なし'} gist_id={gist_id!r}")

        # ── 1〜3: 本番と同じ経路で往復する ──
        data["stage"] = "selftest"
        r = kc_gist_selftest(gist_id, STATUS_FILE, token=tok)
        data["selftest"] = r
        print("selftest: " + json.dumps(r, ensure_ascii=False))

        # ── 4: 陰性対照。失敗が理由つきで返らなければ、成功の判定も信用できない ──
        data["stage"] = "negative"
        nok, ndetail = kc_gist_write(BOGUS_GIST, STATUS_FILE, "probe\n",
                                     token=tok)
        data["negative"] = {"write_ok": nok, "detail": ndetail}
        # トークンが無い回は 1 で既に落ちているので、ここは「拒否されたか」を見る。
        data["negative_ok"] = (not nok) and bool(ndetail)
        print(f"negative: write_ok={nok} detail={ndetail!r}")

        data["ok"] = bool(r.get("ok")) and data["negative_ok"]
        if not data["ok"]:
            if not r.get("token_present"):
                data["advice"] = ("Kaggle Secrets に GITHUB_TOKEN を登録する"
                                  "（scope は gist のみでよい）")
            elif not r.get("write_ok"):
                data["advice"] = ("書き込みが拒否された。トークンの scope が "
                                  "gist か、GIST_ID が本人の Gist かを確認する")
            elif not r.get("read_ok") or not r.get("matched"):
                data["advice"] = ("書けたが読み返せない。GIST_ID が別の Gist を"
                                  "指している可能性がある（write_ok=True）")
            else:
                data["advice"] = ("往復は通ったが、失敗が失敗として返らない。"
                                  "kc_gist_write の例外処理を疑う")
            print("NG: " + data["advice"])
        else:
            print("OK: Gist の往復と失敗検出の両方が通った")
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

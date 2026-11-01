"""Gist への進捗書き込み（生成コードへ埋め込む）。

Kaggle の UI ログは長大で貼りにくく、`kaggle kernels output` は**完了まで
何も返さない**。起動に失敗した回は「どこで止まったか」を外から見る手段が
要るので、待機中は 30 秒ごとに Gist へ流している。

ところが 2026-11-01 の `vllm-playground` は Gist を 10/28 から一度も
更新していなかった。理由（トークン無し / 401 / 404）はどこにも残らず、
外からは「何も起きていない」と区別がつかなかった。**黙って落ちるのが
一番高い**ので、この層は必ず理由を返す。

**この経路は専用アカウント運用とは両立しない。** Kaggle の Secret は
アカウントに登録するだけでは読めず、**カーネルごとに紐付ける**必要があり、
未紐付けの `get_secret()` は `ConnectionError` を返す（通信障害に見える）。
専用アカウント側では公開 URL を**ビーコンカーネル**（カーネルのコードを
掲示板にする方式）へ移した。経緯と実測は `notes/url-publishing.md`。
ここが生きているのは、進捗（起動待ちの状況）を運ぶ経路がまだ他に無いため。

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
    return kc_gh_token_detail()[0]


def kc_gh_token_detail():
    """`(token, detail)`。detail は**どこで解決したか / どこで転んだか**。

    `detail` を返すのは、`""` の 1 語では対処が決まらないため。Kaggle の
    Secret は**アカウントに登録するだけでは足りず、カーネルごとに紐付ける**
    必要がある。2026-11-01 に新しく push した `gist-probe` は
    `token_present=false` で、登録漏れなのか紐付け漏れなのかを
    `""` からは読み取れなかった（Kaggle は「この notebook には無い」を
    例外で返すので、そこまで拾わないと区別できない）。
    """
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok, "env:GITHUB_TOKEN"
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as e:
        return "", f"kaggle_secrets を import できない（{type(e).__name__}: {e}）"
    try:
        tok = (UserSecretsClient().get_secret("GITHUB_TOKEN") or "").strip()
    except Exception as e:
        # ここに来る典型は「このカーネルに Secret が紐付いていない」。
        return "", (f"Secret を読めない（{type(e).__name__}: {e}）"
                    " — アカウントに登録済みでも、カーネルごとの紐付けが"
                    "無いと読めない")
    if not tok:
        return "", "Secret は読めたが中身が空"
    return tok, "secrets:GITHUB_TOKEN"


def kc_gist_write(gist_id, filename, text, token=None, timeout=10):
    """Gist の 1 ファイルを更新する。戻り値は `(ok, detail)`。

    `detail` は**失敗の理由**（成功なら空文字）。呼び出し側が 1 回だけ声に
    出すために使う。HTTP の本文も拾うのは、401（トークン不正）と
    404（Gist ID 違い / 権限なし）を取り違えないため。

    `token` を明示できるのはテストの口。省略時は env → Secrets の順で解決する。
    """
    if not gist_id:
        return False, "GIST_ID が空（settings の GIST_ID_DEFAULT も未設定）"
    if token is None:
        tok, why = kc_gh_token_detail()
        if not tok:
            return False, why
    else:
        tok = token
        if not tok:
            return False, "トークンが空（引数で明示された）"
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
    if token is None:
        tok, token_why = kc_gh_token_detail()
    else:
        tok, token_why = token, "引数で明示"
    out = {"ok": False, "gist_id": gist_id, "filename": filename,
           "token_present": bool(tok), "token_detail": token_why, "stamp": stamp,
           "write_ok": False, "write_detail": "", "read_ok": False,
           "read_detail": "", "matched": False, "files": []}
    if not tok:
        # 解決に失敗した**理由**をそのまま write_detail に置く。ここで
        # kc_gist_write に空トークンを渡すと「引数で明示された」という
        # 無意味な理由に潰れて、紐付け漏れが分からなくなる。
        out["write_detail"] = token_why
        return out
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

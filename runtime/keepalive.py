"""キープアライブ設定ファイルと /stop の共通処理（生成コードへ埋め込む）。"""
import json
import os
import urllib.request


def kc_read_conf(path, default_minutes=60, max_minutes=480,
                 on_invalid="fail", fail=None, quiet_clamp=False):
    """(run, minutes, mtime) を返す。RUN=0 は (0, default_minutes, mtime)。

    on_invalid: "fail" は fail(msg)（未指定なら ValueError）、
                "zero" は停止扱い（RUN=0 と同じ）。
    quiet_clamp: proxy が書く既定値を上限で頭打ちにするとき黙る。
    """
    try:
        with open(path) as f:
            d = dict(l.strip().split("=", 1) for l in f
                     if l.strip() and "=" in l)
        run = int(d.get("RUN", 1))
        minutes = int(d.get("MINUTES", default_minutes))
    except OSError:
        run, minutes = 1, default_minutes
    except ValueError:
        run, minutes = 1, default_minutes
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if run <= 0:
        return 0, default_minutes, mtime
    if minutes <= 0:
        msg = f"KEEPALIVE_MINUTES={minutes} は無効（1〜{max_minutes} で指定）"
        if on_invalid == "zero":
            return 0, default_minutes, mtime
        if fail:
            fail(msg)
        raise ValueError(msg)
    if minutes > max_minutes:
        if not (quiet_clamp and minutes == default_minutes):
            print(f"⚠ KEEPALIVE_MINUTES={minutes} → {max_minutes} に制限")
        minutes = max_minutes
    return run, minutes, mtime


def kc_write_conf(path, run, minutes=None):
    with open(path, "w") as f:
        f.write(f"RUN={run}\n")
        if minutes is not None:
            f.write(f"MINUTES={minutes}\n")


def kc_conf_mtime(path):
    """keepalive.conf の更新時刻。proxy がチャット送信ごとに書き直す。"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def kc_stop_via_proxy(port, auth=None, timeout=30):
    headers = {"X-Kc-Stop": "1"}
    if auth:
        headers["Authorization"] = auth
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/stop", data=b"", method="POST",
        headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())

#!/usr/bin/env python3
"""Kaggle の GPU / TPU クォータ残量を表示する。

kaggle CLI には quota サブコマンドが無い（1.7.4.5 で確認）が、
Web API の /api/v1/kernels/quota は Basic 認証で読める。

使い方:
    python3 kaggle-quota.py
    python3 kaggle-quota.py --json      # 生 JSON を出す
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://www.kaggle.com/api/v1/kernels/quota"
CRED = os.path.expanduser("~/.kaggle/kaggle.json")


def auth_header():
    if not os.path.isfile(CRED):
        sys.exit(f"認証情報がありません: {CRED}\n"
                 "Kaggle の Account から kaggle.json を取得してください。")
    with open(CRED) as f:
        cfg = json.load(f)
    token = base64.b64encode(
        f"{cfg['username']}:{cfg['key']}".encode()
    ).decode()
    return "Basic " + token


def fetch():
    req = urllib.request.Request(API, headers={"Authorization": auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"接続失敗: {e.reason}")


def hms(seconds):
    """秒を 'Xh YYm' 形式にする（負の値は '-Xh YYm'）。"""
    seconds = int(seconds)
    sign = "-" if seconds < 0 else ""
    h, rem = divmod(abs(seconds), 3600)
    return f"{sign}{h}h{rem // 60:02d}m"


def bar(used, total, width=28):
    """使用率のバー。"""
    if total <= 0:
        return ""
    filled = int(round(used / total * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def show(label, q, unit_note=""):
    used = q["timeUsed"]["seconds"]
    total = q["totalTimeAllowed"]["seconds"]
    reserved = q.get("timeReserved", {}).get("seconds", 0)
    left = total - used - reserved
    pct = used / total * 100 if total else 0

    print(f"=== {label} ===")
    print(f"  {bar(used, total)}  {pct:.1f}%")
    print(f"  使用済み : {hms(used):>9}  ({used:,} 秒)")
    print(f"  上限     : {hms(total):>9}  ({total:,} 秒)")
    if reserved:
        print(f"  予約中   : {hms(reserved):>9}")
    print(f"  残り     : {hms(left):>9}  ({left:,} 秒)")

    # 何回動かせるかの目安
    if left > 0:
        runs8h = left // (8 * 3600)
        runs30m = left // (30 * 60)
        print(f"  目安     : 8時間セッション {runs8h} 回 / 30分セッション {runs30m} 回")
    else:
        print("  ⚠ 残りなし。リセットまで待つか、別アカウントが必要です。")
    if unit_note:
        print(f"  {unit_note}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="生 JSON を出力")
    args = ap.parse_args()

    data = fetch()

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    show("GPU クォータ", data["gpuQuota"])
    show("TPU クォータ", data["tpuQuota"])

    # リセットまでの残り時間
    refresh = datetime.fromisoformat(
        data["quotaRefreshTime"].replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    total = int((refresh - now).total_seconds())

    print(f"リセット : {data['quotaRefreshTime']}")
    if total <= 0:
        print("まで     : リセット時刻を過ぎています（反映待ち）")
    else:
        days, rem = divmod(total, 86400)
        print(f"まで     : {days}日 {rem // 3600}時間 {(rem % 3600) // 60}分")

    if data["gpuQuota"].get("isPayToScaleEnabled"):
        print("ℹ Pay to Scale が有効です（上限を超えて課金で延長できます）")


if __name__ == "__main__":
    main()

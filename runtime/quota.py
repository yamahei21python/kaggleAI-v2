"""Kaggle のクォータ API（GPU/TPU 残量）を読む共通処理（生成コードへ埋め込む）。

kaggle CLI には quota サブコマンドが無いが、Web API は Basic 認証で読める。
呼び出し側は auth_proxy.py（/quota）と kaggle-quota.py（CLI 表示）。
"""
import base64
import json
import os
import urllib.request

KC_QUOTA_URL = "https://www.kaggle.com/api/v1/kernels/quota"
KC_CRED_PATH = os.path.expanduser("~/.kaggle/kaggle.json")


def kc_kaggle_credentials(cred_path=None, environ=None):
    """(username, key) を返す。kaggle.json → 環境変数の順。無ければ (None, None)。

    カーネルには kaggle.json が無いが、KAGGLE_USERNAME / KAGGLE_KEY が
    環境変数で入っている。
    """
    cred_path = KC_CRED_PATH if cred_path is None else cred_path
    environ = os.environ if environ is None else environ
    if os.path.isfile(cred_path):
        try:
            with open(cred_path) as f:
                cfg = json.load(f)
            user, key = cfg.get("username"), cfg.get("key")
            if user and key:
                return user, key
        except Exception:
            pass
    user = environ.get("KAGGLE_USERNAME") or environ.get("KAGGLE_USER")
    key = environ.get("KAGGLE_KEY")
    return (user, key) if user and key else (None, None)


def kc_fetch_quota(user, key, timeout=45, url=KC_QUOTA_URL, opener=None):
    """クォータの生 JSON を返す。HTTP / 接続エラーは例外のまま投げる。"""
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    req = urllib.request.Request(
        url, headers={"Authorization": "Basic " + token})
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
        return json.loads(r.read())


def kc_quota_summary(data):
    """UI 用の GPU クォータ要約。gpuQuota が無い応答は available=False。"""
    g = data.get("gpuQuota") or {}
    used = (g.get("timeUsed") or {}).get("seconds")
    total = (g.get("totalTimeAllowed") or {}).get("seconds")
    if used is None or total is None:
        return {"available": False, "reason": "no_gpu_quota"}
    reserved = (g.get("timeReserved") or {}).get("seconds", 0)
    return {
        "available": True,
        "usedSeconds": used,
        "totalSeconds": total,
        "reservedSeconds": reserved,
        "leftSeconds": total - used - reserved,
        "refreshTime": data.get("quotaRefreshTime", ""),
    }

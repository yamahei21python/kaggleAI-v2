# 現在の公開URLを Gist に書き出す（取得先の固定）
# Secrets: GITHUB_TOKEN（gist 権限のみでよい）, GIST_ID（省略時は既定 gist）
# keepalive でトンネルが再起動したときも同じ関数を呼ぶ。
import json, os, urllib.request

# 既定値は settings セル（GIST_ID_DEFAULT / GIST_OWNER）。
# 環境変数 → Secrets → 既定値（GITHUB_TOKEN と同じ解決順に揃える）
GIST_ID = os.environ.get("GIST_ID", "").strip()
if not GIST_ID:
    try:
        from kaggle_secrets import UserSecretsClient
        GIST_ID = (UserSecretsClient().get_secret("GIST_ID") or "").strip()
    except Exception:
        GIST_ID = ""
if not GIST_ID:
    GIST_ID = GIST_ID_DEFAULT


def _gh_token():
    tok = os.environ.get("GITHUB_TOKEN", "")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN") or ""
    except Exception:
        return ""


def publish_url(u):
    """現在の公開URL を Gist に書く（quick tunnel は URL が毎回変わるため）。

    Gist は secret（非公開URL）だが、URL を知れば誰でも読める。
    認証は付けていないので、実質この URL 自体が入口になる。
    """
    tok = _gh_token()
    if not tok:
        print("ℹ GITHUB_TOKEN 未設定 → 固定URL配布はスキップ（接続情報セルを使う）")
        return False
    files = {"url.txt": {"content": u + "\n"}}
    body = json.dumps({"files": files}).encode()
    req = urllib.request.Request(
        "https://api.github.com/gists/" + GIST_ID,
        data=body,
        method="PATCH",
        headers={
            "Authorization": "Bearer " + tok,
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("✅ 固定URL配布を更新: HTTP " + str(r.status))
            return True
    except Exception as e:
        print("⚠ Gist 更新失敗（続行）: " + repr(e))
        return False


publish_url(public_url)
print("📌 固定取得先: https://gist.githubusercontent.com/"
      + GIST_OWNER + "/" + GIST_ID + "/raw/url.txt")

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

# 公開 URL を置く掲示板カーネルの名前（owner は GIST_OWNER と同じ）。
# カーネルから push できるのは自分のカーネルだけなので、配信カーネルと
# 同じアカウントに置く。settings で `BEACON_NAME` を定義すれば上書きできる。
BEACON_NAME = globals().get("BEACON_NAME") or "serve-url-beacon"


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
    """現在の公開URL を Gist とビーコンに書く（URL が毎回変わるため）。

    Gist は secret（非公開URL）だが、URL を知れば誰でも読める。
    認証は付けていないので、実質この URL 自体が入口になる。

    ビーコン（`kc_beacon_publish`）は**Secrets を使わない**経路で、カーネルの
    コードを掲示板にする。Gist がスキップされる環境（Secret 未紐付け・
    専用アカウント運用）でも URL を外から読めるようにするため、Gist の
    成否に関わらず**必ず**呼ぶ。`notes/url-publishing.md`。
    """
    ok_gist = _publish_gist(u)
    _publish_beacon(u)
    return ok_gist


def _publish_beacon(u):
    """ビーコンを更新する。失敗しても配信は止めない（理由は必ず出す）。"""
    slug = f"{GIST_OWNER}/{BEACON_NAME}" if GIST_OWNER else ""
    try:
        ok, why = kc_beacon_publish(u, stage="serving", slug=slug)
    except NameError:
        # beacon を束ねていないビルド（移行前の notebook）。
        print("ℹ ビーコン未同梱 → URL 配布は Gist のみ")
        return False
    print(("📡 ビーコン更新: " + why) if ok else ("⚠ ビーコン更新失敗（続行）: " + why))
    return ok


def _publish_gist(u):
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
print("📡 ビーコン取得先: kaggle kernels pull "
      + GIST_OWNER + "/" + BEACON_NAME)

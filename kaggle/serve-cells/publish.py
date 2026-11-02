# 現在の公開URLをビーコンカーネルに書き出す（取得先の固定）
# Secrets: 不要。カーネルに注入された Kaggle 認証で `kernels_push` する。
# keepalive でトンネルが再起動したときも同じ関数を呼ぶ。
#
# Gist 版（`url.txt` を PATCH する経路）は 2026-11-02 に廃止した。
# Kaggle の Secret は**カーネルごとに紐付ける**必要があり、未紐付けの
# `get_secret()` は `ConnectionError`（＝通信障害に見える）を返すため、
# カーネルを増やすたびに UI 作業が要っていた。ビーコンはその作業ごと不要にする。

# 公開 URL を置く掲示板カーネルの名前（owner は KAGGLE_OWNER）。
# カーネルから push できるのは自分のカーネルだけなので、配信カーネルと
# 同じアカウントに置く。settings で `BEACON_NAME` を定義すれば上書きできる。
BEACON_NAME = globals().get("BEACON_NAME") or "serve-url-beacon"


def publish_url(u):
    """現在の公開URL をビーコンに書く（URL が毎回変わるため）。

    ビーコン（`kc_beacon_publish`）は**Secrets を使わない**経路で、カーネルの
    コードを掲示板にする。失敗しても配信は止めないが、理由は必ず出す
    （2026-11-01 に「URL が 10/28 から一度も更新されていないことに誰も
    気づかない」という失敗をしている）。`notes/url-publishing.md`。
    """
    return _publish_beacon(u)


def _publish_beacon(u):
    """ビーコンを更新する。失敗しても配信は止めない（理由は必ず出す）。"""
    slug = f"{KAGGLE_OWNER}/{BEACON_NAME}" if KAGGLE_OWNER else ""
    try:
        ok, why = kc_beacon_publish(u, stage="serving", slug=slug)
    except NameError:
        # beacon を束ねていないビルド（移行前の notebook）。
        print("⚠ ビーコン未同梱 → 公開 URL を外から読めない")
        return False
    print(("📡 ビーコン更新: " + why) if ok else ("⚠ ビーコン更新失敗（続行）: " + why))
    return ok


publish_url(public_url)
print("📡 ビーコン取得先: kaggle kernels pull " + KAGGLE_OWNER + "/" + BEACON_NAME)

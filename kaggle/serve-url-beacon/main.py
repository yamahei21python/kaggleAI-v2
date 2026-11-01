# 本番の掲示板。配信カーネルが書き換える
"""公開 URL のビーコン（掲示板）。

このカーネルは**実行しない**。コードそのものが受け渡しの場所で、
配信カーネルが `kernels_push` でこのファイルを書き換える。
読む側は `kaggle kernels pull` して `URL` と `UPDATED` を読む。

`STAGE` は `serving` / `probe` など。**`probe` は本物ではない**ので、
読む側は `probe` を捨てること（テスト URL を掴む事故を防ぐ）。
`UPDATED` は発行時刻の epoch 秒。古い値を掴んでいないかの判定に使う。
"""

URL = ''
STAGE = 'placeholder'
UPDATED = 0.0

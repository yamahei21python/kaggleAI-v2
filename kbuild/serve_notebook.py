"""notebook 配信カーネルの共通フロー（`serve_script.py` の notebook 版）。

notebook 形式で push する配信カーネルは Ollama / llama-server / vLLM の 3 系統。
**セルの並びも配管セルもここが唯一の定義**で、実体は `kaggle/serve-cells/*.py`。

各ビルダーが渡すのは「エンジンで本当に違うもの」だけ:

    settings       … 名前空間の定義（ポート・モデル・Secrets 既定値・能力表）
    install        … 依存導入
    launch         … 推論サーバ起動
    verify_checks  … `engine_checks(judge, retry, base)` を定義するセル
    warmup         … ウォームアップ（任意）
    pre_launch     … install と proxy の間に挟む (markdown, code)（任意）
    title / *_md   … 見出し文言（既定はここで用意する）

`serve-cells` の共有セルは settings が定義する次の名前だけを読む。
**新しいエンジンを足すときに増やすのはこの契約の範囲だけ**:

    ポート      PORT / UPSTREAM_PORT / PROXY_PORT
    実体パス    CHAT_PATH / PROXY_PATH
    プロキシ    PROXY_AUTH / PROXY_EXTRA_ARGS / PROXY_CHECK_PATH
    モデル      MODEL_NAME / MODEL_DISPLAY / ENGINE_DISPLAY / CONTEXT_LENGTH
    アカウント  KAGGLE_OWNER（ビーコンの slug に使う）
    keepalive   MAX_MINUTES / KEEPALIVE_RESET_MINUTES / TUNNEL_CHECK_EVERY
                ENGINE_LABEL / ENGINE_PROCS / CLEANUP_TARGETS / ENGINE_LOG
    後片付け    engine_cleanup()  ← 必須。未使用のエンジンは None を返す
    永続化      OUT_DIRS / save_output / fail

共有セルはビルダー側の値埋め込みをしない（プレースホルダを持たない）。
値が要るものは settings セルに置く。
"""
import pathlib
from dataclasses import dataclass

from kbuild import pack_nb
from runtime import source_bundle

CELLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "serve-cells"

# 共有セルの前後に置く説明セル。文言も 1 箇所に集約する。
MD_RUNTIME = (
    "---\n### 🧩 共通ランタイム（proxy / tunnel / keepalive / verify）\n\n"
    "`runtime/` の真実源をビルダーが連結して埋め込んでいます。\n"
    "セルを直接編集しても次のビルドで上書きされます。\n")
MD_SETTINGS = "---\n### ⚙️ 設定（GPU検出 → モデル・コンテキスト決定）\n"
MD_INSTALL = "---\n### 📦 依存の導入\n"
MD_PROXY = ("---\n### 🌐 リバースプロキシ起動（UI 配信 + 同一オリジン化）\n\n"
            "推論サーバは非公開、公開側は認証プロキシが受けます。\n")
MD_LAUNCH = "---\n### 🔥 推論サーバ起動\n"
MD_WARMUP = "---\n### 🔥 ウォームアップ（初回ロードは数分かかります）\n"
MD_TUNNEL = {
    "cloudflared": "---\n### 🔗 cloudflared トンネル → 公開URL\n",
}
MD_PUBLISH = ("---\n### 📡 公開URLの配布（ビーコンカーネル）\n\n"
              "quick tunnel の URL は毎回変わるので、**取得先だけ固定**します。\n"
              "同じアカウントの掲示板カーネル（`serve-url-beacon`）の `main.py` を\n"
              "書き換え、クライアントは `kaggle kernels pull` で URL を読みます。\n"
              "Secrets も外部サービスも要りません（`notes/url-publishing.md`）。\n")
MD_VERIFY = ("---\n### ✅ 公開URL経由の疎通確認\n\n"
             "確認項目はエンジンごとに `engine_checks()` が持つ。\n"
             "失敗は raise（検証不能のまま放置しない）。\n")
MD_CONNECTION = ("---\n### 🔌 接続情報\n\n"
                 "クライアント設定にコピペする値と `opencode.json` を出力します。\n")
MD_KEEPALIVE = (
    "---\n### 🔋 キープアライブ\n\n"
    "有限ループで自動終了します（`KEEPALIVE_RESET_MINUTES` / `MAX_MINUTES`）。\n\n"
    "- チャット送信で残り時間がリセットされます（`/tmp/keepalive.conf`）\n"
    "- 「■ 停止」ボタンで `RUN=0` が書かれ、ループが終了します\n")


def cell(name: str) -> str:
    """共有セル（`kaggle/serve-cells/<name>.py`）のソース。"""
    return (CELLS_DIR / name).read_text(encoding="utf-8")


@dataclass
class Flow:
    """1 カーネル分のセル列。エンジン差は settings/install/launch/verify_checks だけ。"""

    title: str
    settings: str
    install: str
    launch: str
    verify_checks: str
    tunnel: str = "cloudflared"
    warmup: str = ""
    pre_launch: tuple = ()          # ((markdown, code), ...) install と proxy の間
    settings_md: str = MD_SETTINGS
    install_md: str = MD_INSTALL
    launch_md: str = MD_LAUNCH
    warmup_md: str = MD_WARMUP
    verify_md: str = MD_VERIFY
    keepalive_md: str = MD_KEEPALIVE
    runtime: tuple = ("net", "proc", "storage", "keepalive", "proxy", "beacon")

    def cells(self):
        if self.tunnel not in MD_TUNNEL:
            raise ValueError(f"未対応のトンネル: {self.tunnel}")
        out = [
            ("markdown", self.title),
            ("markdown", MD_RUNTIME),
            ("code", source_bundle(*self.runtime)),
            ("markdown", self.settings_md),
            ("code", self.settings),
            ("markdown", self.install_md),
            ("code", self.install),
        ]
        out += [tuple(pair) for pair in self.pre_launch]
        out += [
            ("markdown", MD_PROXY),
            ("code", cell("proxy.py")),
            ("markdown", self.launch_md),
            ("code", self.launch),
        ]
        if self.warmup:
            out += [("markdown", self.warmup_md), ("code", self.warmup)]
        out += [
            ("markdown", MD_TUNNEL[self.tunnel]),
            ("code", cell(f"tunnel_{self.tunnel}.py")),
            ("markdown", MD_PUBLISH),
            ("code", cell("publish.py")),
            ("markdown", self.verify_md),
            ("code", self.verify_checks),
            ("code", cell("verify.py")),
            ("markdown", MD_CONNECTION),
            ("code", cell("connection.py")),
            ("markdown", self.keepalive_md),
            ("code", cell("keepalive.py")),
        ]
        return out

    def write(self, path, language_version: str = "3.10.0") -> None:
        pack_nb.write(path, self.cells(), language_version)

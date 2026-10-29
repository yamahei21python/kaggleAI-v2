# バイナリ復元: Dataset キャッシュ優先、不足分のみ DL
import os, shutil, subprocess

BIN_DIR = "/usr/local/bin"
LIB_DIR = "/usr/local/lib/ollama"
BIN_CACHE = os.path.join(PATCH_PATH, "bin")
LIB_CACHE = os.path.join(PATCH_PATH, "lib", "ollama")
MIN_SIZE = 1024 * 1024  # 1MB 未満は壊れているとみなす
# llama-server は例外。Linux 版は 15 KB の薄いラッパーで、
# 実体は libllama-server-impl.so を dlopen する（1MB 判定だと誤って弾く）
MIN_SIZE_LLAMA = 4096

LLAMA_SERVER_DST = os.path.join(LIB_DIR, "llama-server")
# install.sh は llama-server を lib/ollama/ に置く。patch Dataset もその配置で
# 作ってあるので lib 側を第一候補、旧配置の bin/ 側も後方互換で見る。
# サイズ検証を先に行う: 0B/切れファイルが良品を shadow すると missing 判定になり
# install.sh が走って ollama が上書きされる（version 固定が崩れる）。
LLAMA_SERVER_CACHE = None
for _c in (os.path.join(LIB_CACHE, "llama-server"),
           os.path.join(BIN_CACHE, "llama-server")):
    if os.path.isfile(_c) and os.path.getsize(_c) >= MIN_SIZE_LLAMA:
        LLAMA_SERVER_CACHE = _c
        break

restored, missing = [], []


def _find_cached(name):
    """バイナリのキャッシュ位置を探す。見つからなければ None。

    cloudflared は専用 Dataset（my-cloudflared-bin）から復元する。
    Dataset 側のフォルダ構造は flatten されることがあるので、
    `bin/` あり / なし の両方を試してから、既存の patch Dataset に流す。
    """
    candidates = [os.path.join(BIN_CACHE, name)]
    if name == "cloudflared":
        candidates = [
            os.path.join(CF_PATH, "cloudflared"),
            os.path.join(CF_PATH, "bin", "cloudflared"),
        ] + candidates
    for src in candidates:
        if os.path.isfile(src) and os.path.getsize(src) >= MIN_SIZE:
            return src
    return None


# --- ollama / cloudflared ---
for name in (["ollama"] if TUNNEL_KIND == "ngrok" else ["ollama", "cloudflared"]):
    dst = os.path.join(BIN_DIR, name)
    src = _find_cached(name)
    if src:
        shutil.copy2(src, dst)
        subprocess.run(["chmod", "+x", dst])
        restored.append(f"{name} <- {src}")
    else:
        missing.append(name)

# --- llama-server（Ollama 0.5+ で必須）---
if LLAMA_SERVER_CACHE and os.path.getsize(LLAMA_SERVER_CACHE) >= MIN_SIZE_LLAMA:
    os.makedirs(LIB_DIR, exist_ok=True)
    shutil.copy2(LLAMA_SERVER_CACHE, LLAMA_SERVER_DST)
    subprocess.run(["chmod", "+x", LLAMA_SERVER_DST])
    restored.append("llama-server")
else:
    missing.append("llama-server")

# --- lib/ollama（CUDA バックエンド）---
# これが無いと llama-server が libggml-cuda.so を dlopen できず、
# エラーを出さずに CPU 推論へ落ちる（実測 0.9 tps / VRAM 0 MiB）。
if os.path.isdir(LIB_CACHE):
    os.makedirs(LIB_DIR, exist_ok=True)
    n_lib = 0
    for root, _dirs, files in os.walk(LIB_CACHE):
        rel = os.path.relpath(root, LIB_CACHE)
        dst_dir = LIB_DIR if rel == "." else os.path.join(LIB_DIR, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            src_p = os.path.join(root, fn)
            dst_p = os.path.join(dst_dir, fn)
            shutil.copy2(src_p, dst_p)
            # copy2 は mode を保存する。tar 経由の Dataset では x ビットが全部
            # 落ちる（v13 実測 → fork/exec permission denied）。root 実行の
            # os.access(X_OK) は「x ビットが1つでもあれば True」なので、全部
            # 落ちたケースを判定できない。lib/ollama の中身は .so と
            # llama-server だけ＝exec ビット過剰でも無害なので無条件 0755。
            os.chmod(dst_p, 0o755)
            n_lib += 1
    restored.append(f"lib/ollama ({n_lib} files)")
else:
    missing.append("lib/ollama")

# lib walk が llama-server を上書きするので、実行ビットは walk の後に確定させる
if os.path.isfile(LLAMA_SERVER_DST):
    os.chmod(LLAMA_SERVER_DST, 0o755)

# CUDA バックエンドの実在確認（これが GPU 推論の前提）
CUDA_LIB = os.path.join(LIB_DIR, "cuda_v12", "libggml-cuda.so")
print(f"{'✅' if os.path.isfile(CUDA_LIB) else '⚠ '} CUDA バックエンド: {CUDA_LIB}")

print(f"✅ キャッシュ復元: {', '.join(restored) if restored else 'なし'}")
print(f"⬇️  DL が必要  : {', '.join(missing) if missing else 'なし'}")

# --- 不足分だけダウンロード ---
# ollama 本体が在っても llama-server / CUDA が欠けている限り install.sh は必要
# （patch 側に llama-server が無いため v8 で欠落を実測。無いと検証で即 fail）。
# install.sh が ollama を上書きする点は許容 — 上書きが機能不全になる风险より、
# llama-server 欠落で落ちる方を避ける。patch に llama-server 入れるのが本筋。
if missing:
    need_ollama = ("ollama" in missing) or ("llama-server" in missing) or ("lib/ollama" in missing)
    procs = []

    if need_ollama:
        # zstd は ollama の .tar.zst 展開に必要
        if subprocess.run(["dpkg", "-s", "zstd"], capture_output=True).returncode != 0:
            subprocess.run("apt-get install -y -qq zstd", shell=True, check=False)
        print("⬇️  ollama をインストール中...")
        procs.append(subprocess.Popen(
            "curl -fsSL --connect-timeout 15 https://ollama.com/install.sh | sh",
            shell=True,
        ))

    if "cloudflared" in missing:
        print("⬇️  cloudflared をダウンロード中...")
        procs.append(subprocess.Popen(
            "curl -fsSL --connect-timeout 15 -o /usr/local/bin/cloudflared "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && "
            "chmod +x /usr/local/bin/cloudflared",
            shell=True,
        ))

    for p in procs:
        # 無限待ちはハングで notebook が止まり成果物が残らない。
        # install.sh（tar.zst 展開込み）・curl とも最大10分まで。
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            fail("バイナリ取得がタイムアウトしました（10分）")

# --- 検証 ---
print()
ok = True
_targets = [(f"{BIN_DIR}/ollama", MIN_SIZE),
            (LLAMA_SERVER_DST, MIN_SIZE_LLAMA),
            (CUDA_LIB, MIN_SIZE)]
if TUNNEL_KIND != "ngrok":
    _targets.insert(1, (f"{BIN_DIR}/cloudflared", MIN_SIZE))
for path, min_sz in _targets:
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    mark = "✅" if exists and size >= min_sz else "❌"
    if mark == "❌":
        ok = False
    print(f"{mark} {path}  ({size:,} B)")

# 実行ビットは tar 経由の Dataset で落ちる（v13 実測: fork/exec permission denied）。
# サイズだけ見ていると通ってしまい、起動時に 500 になるので明示的に確認する。
# root 実行では os.access(X_OK) が「x ビットが1つでもあれば True」になるため、
# mode ビットで 3 桁すべて（0o111）が立っているかを見る。
_exes = [f"{BIN_DIR}/ollama", LLAMA_SERVER_DST]
if TUNNEL_KIND != "ngrok":
    _exes.insert(1, f"{BIN_DIR}/cloudflared")
for exe in _exes:
    if not os.path.isfile(exe):
        continue
    if os.stat(exe).st_mode & 0o111 != 0o111:
        try:
            os.chmod(exe, 0o755)
            print(f"🔧 実行ビット付与: {exe}")
        except OSError as e:
            print(f"❌ chmod 失敗: {exe} ({e})")
            ok = False
    if os.stat(exe).st_mode & 0o111 != 0o111:  # 事後検証（noexec 等を検出）
        print(f"❌ 実行ビット未設定: {exe}")
        ok = False

if not ok:
    fail("バイナリの準備に失敗しました（cloudflared は Dataset から、"
         "それ以外は GitHub / ollama.com から取る想定）")

print()
# --version が通る = 実行ビットの end-to-end 証明。returncode まで見る
# （exec 失敗だと stdout が空のまま素通りする）。
_ver_exes = [f"{BIN_DIR}/ollama"]
if TUNNEL_KIND != "ngrok":
    _ver_exes.append(f"{BIN_DIR}/cloudflared")
for _exe in _ver_exes:
    _r = subprocess.run([_exe, "--version"], capture_output=True, text=True)
    print((_r.stdout or _r.stderr).strip())
    if _r.returncode != 0:
        fail(f"{_exe} 起動失敗 (returncode={_r.returncode})")

# --- Basic 認証プロキシを書き出す（base64 埋め込み）---
# PROXY_PATH は設定セルの定義を使う（二重管理しない）
import base64 as _b64
with open(PROXY_PATH, "wb") as f:
    f.write(_b64.b64decode(PROXY_SRC_B64))
print(f"✅ auth_proxy.py を書き出し: {PROXY_PATH} "
      f"({os.path.getsize(PROXY_PATH):,} B)")

with open(CHAT_PATH, "wb") as f:
    f.write(_b64.b64decode(CHAT_SRC_B64))
print(f"✅ llama-chat.html を書き出し: {CHAT_PATH} "
      f"({os.path.getsize(CHAT_PATH):,} B)")

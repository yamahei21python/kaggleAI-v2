# llama-server と依存 .so を patch Dataset から復元する。
# Ollama 本体（bin/ollama）は使わない。llama-server だけ要る。
import base64 as _b64
import os, shutil, subprocess

MIN_SIZE_LLAMA = 4096   # Linux 版 llama-server は 15 KB の薄いラッパー
# .so の最小サイズ。libggml.so.0 は 220 KB、libggml-base.so.0 は 817 KB と
# 小さいものがある（実測）。1 MB で切ると良品を誤って弾く。
MIN_SIZE_SO = 64 * 1024

LIB_CACHE = os.path.join(PATCH_PATH, "lib", "ollama")

restored, missing = [], []


def _copy(src, dst, min_size):
    """サイズ検証してからコピー。0B/切れファイルで良品を潰さない。"""
    if not (os.path.isfile(src) and os.path.getsize(src) >= min_size):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


# --- llama-server 本体 ---
for cand in (os.path.join(LIB_CACHE, "llama-server"),
             os.path.join(PATCH_PATH, "bin", "llama-server")):
    if _copy(cand, LLAMA_SERVER, MIN_SIZE_LLAMA):
        restored.append("llama-server")
        break
else:
    missing.append("llama-server")

# --- lib/ollama 配下を丸ごと（.so と cuda_v12）---
if os.path.isdir(LIB_CACHE):
    n = 0
    for root, _dirs, files in os.walk(LIB_CACHE):
        rel = os.path.relpath(root, LIB_CACHE)
        dst_dir = LIB_DIR if rel == "." else os.path.join(LIB_DIR, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, fn)
            # llama-server は上で配置済み。上書きしない。
            if os.path.abspath(dst) == os.path.abspath(LLAMA_SERVER):
                continue
            try:
                shutil.copy2(src, dst)
                n += 1
            except OSError as e:
                print(f"  ⚠ copy 失敗: {fn} ({e})")
    restored.append(f"lib/ollama ({n} files)")
else:
    missing.append("lib/ollama")

print(f"✅ 復元: {', '.join(restored) if restored else 'なし'}")
print(f"❌ 不足: {', '.join(missing) if missing else 'なし'}")

# --- 実行ビット ---
# tar 経由の Dataset は実行ビットが落ちる。root 実行では os.access(X_OK) が
# 甘いので mode ビットを直接見る。
for exe in [LLAMA_SERVER]:
    if os.path.isfile(exe) and (os.stat(exe).st_mode & 0o111) != 0o111:
        os.chmod(exe, 0o755)
        print(f"🔧 実行ビット付与: {exe}")

# --- 検証 ---
print()
CUDA_LIB = os.path.join(LIB_DIR, "cuda_v12", "libggml-cuda.so")
IMPL_LIB = os.path.join(LIB_DIR, "libllama-server-impl.so")
ok = True
for path, min_sz, label in [
    (LLAMA_SERVER, MIN_SIZE_LLAMA, "llama-server"),
    (IMPL_LIB, MIN_SIZE_SO, "libllama-server-impl.so"),
    (CUDA_LIB, MIN_SIZE_SO, "cuda_v12/libggml-cuda.so"),
    (os.path.join(LIB_DIR, "libllama.so.0"), MIN_SIZE_SO, "libllama.so.0"),
    (os.path.join(LIB_DIR, "libggml-base.so.0"), MIN_SIZE_SO, "libggml-base.so.0"),
    (os.path.join(LIB_DIR, "libggml.so.0"), MIN_SIZE_SO, "libggml.so.0"),
    (os.path.join(LIB_DIR, "libllama-common.so.0"), MIN_SIZE_SO, "libllama-common.so.0"),
]:
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    mark = "✅" if exists and size >= min_sz else "❌"
    if mark == "❌":
        ok = False
    print(f"{mark} {label}  ({size:,} B)")

if not ok:
    fail("llama-server の依存が揃っていません（patch Dataset を確認）", ENGINE_LOG)

# --- バージョン確認（実行ビットの end-to-end 証明）---
env = {**os.environ,
       "LD_LIBRARY_PATH": LIB_DIR + ":" + os.path.join(LIB_DIR, "cuda_v12")}
r = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True, text=True,
                   env=env, timeout=60)
out = (r.stdout or "") + (r.stderr or "")
print()
print(out.strip()[:600])
if r.returncode != 0:
    fail(f"llama-server --version 失敗 (rc={r.returncode})", ENGINE_LOG)

# --- プロキシと UI を書き出す（設定セルが持つ base64 を実体化する）---
with open(PROXY_PATH, "wb") as f:
    f.write(_b64.b64decode(PROXY_SRC_B64))
print(f"✅ auth_proxy.py: {PROXY_PATH} ({os.path.getsize(PROXY_PATH):,} B)")

with open(CHAT_PATH, "wb") as f:
    f.write(_b64.b64decode(CHAT_SRC_B64))
print(f"✅ llama-chat.html: {CHAT_PATH} ({os.path.getsize(CHAT_PATH):,} B)")

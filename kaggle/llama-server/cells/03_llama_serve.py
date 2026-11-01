# モデルデータを配置 → llama-server を直起動する。
#
# ステージング部分は Ollama 版（実証済み）と同じ考え方を使う。違うのは
# 「ollama serve を起動する」代わりに「llama-server を起動する」点だけ。
# これで Dataset のマウント差異・blob 充足・symlink の罠を再発明しない。
import glob as _g, json as _json
import os, shutil, subprocess, time
import urllib.request

os.makedirs(OLLAMA_MODELS, exist_ok=True)


def _stage(name, extra_roots=()):
    """DATASET_PATH/name 以下のファイルを実dirとして配置（file symlink）。

    patch Dataset の blobs（config blob sha256-cbe6ad4… 214B が入っている）
    も合流させる。これが無いと manifest が要求する blob が欠けて起動できない。
    """
    dst = os.path.join(OLLAMA_MODELS, name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root in (DATASET_PATH,) + tuple(extra_roots):
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        for p in sorted(_g.glob(os.path.join(base, "**"), recursive=True)):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, base)
            d = os.path.join(dst, rel)
            if name == "manifests":
                # manifest は実ファイルのみ正。symlink は ollama が拒否する。
                if os.path.lexists(d) and not os.path.islink(d) and os.path.isfile(d):
                    continue
                if os.path.lexists(d):
                    os.remove(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(p, d)
                n += 1
                continue
            # blobs: 有効な symlink/実ファイルだけスキップ、壊れていれば作り直す
            if os.path.lexists(d) and os.path.isfile(d):
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                os.symlink(p, d)
            except OSError:
                # 16 GiB を複製すると disk full。巨大ファイルは複製せず落とす。
                if os.path.getsize(p) > (1 << 30):
                    raise
                shutil.copy2(p, d)
            n += 1
    return dst, n


if os.path.isdir(DATASET_PATH) and os.listdir(DATASET_PATH):
    print("📁 Dataset 検出 → モデルデータを準備中...")
    try:
        _, n_blob = _stage("blobs", extra_roots=(PATCH_PATH,))
        _, n_man = _stage("manifests")
    except OSError as _e:
        fail(f"モデルデータ配置に失敗: {type(_e).__name__}: {_e}", ENGINE_LOG)
    print(f"✅ blobs {n_blob} 件 / manifests {n_man} 件を配置")
else:
    fail(f"Dataset が見つかりません: {DATASET_PATH}", ENGINE_LOG)

# ---- manifest から model / projector の実パスを取る ----
_repo, _tag = MODEL_NAME.rsplit(":", 1)
_man = os.path.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai",
                    *_repo.split("/"), _tag)
if not os.path.isfile(_man):
    fail(f"manifest が見つからない: {_man}", ENGINE_LOG)

with open(_man) as _f:
    _mf = _json.load(_f)

# manifest が要求する blob の充足を先に点検する（欠けたまま起動すると
# llama-server が「テンソル不足」で落ち、原因が分かりにくい）。
_need = [_mf.get("config", {})] + _mf.get("layers", [])
_miss = [e["digest"] for e in _need
         if e.get("digest") and not os.path.lexists(os.path.join(
             OLLAMA_MODELS, "blobs",
             "sha256-" + e["digest"].split(":", 1)[1]))]
if _miss:
    fail(f"blob 欠落 {len(_miss)} 件: {_miss[:5]}", ENGINE_LOG)
print("✅ manifest 要求 blob 全数充足")

MODEL_BLOB = None
MMPROJ_BLOB = None
for _layer in _mf.get("layers", []):
    _mt = _layer.get("mediaType", "")
    _dg = _layer.get("digest", "")
    if not _dg.startswith("sha256:"):
        continue
    _p = os.path.join(OLLAMA_MODELS, "blobs", "sha256-" + _dg.split(":", 1)[1])
    if _mt == "application/vnd.ollama.image.model":
        MODEL_BLOB = _p
    elif _mt == "application/vnd.ollama.image.projector":
        MMPROJ_BLOB = _p

if not MODEL_BLOB or not os.path.isfile(MODEL_BLOB):
    fail(f"model blob が見つかりません: {MODEL_BLOB}", ENGINE_LOG)
print(f"✅ model    : {os.path.basename(MODEL_BLOB)} "
      f"({os.path.getsize(MODEL_BLOB) / 1024**3:.2f} GiB)")
if MMPROJ_BLOB and os.path.isfile(MMPROJ_BLOB):
    print(f"✅ projector: {os.path.basename(MMPROJ_BLOB)} "
          f"({os.path.getsize(MMPROJ_BLOB) / 1024**3:.2f} GiB)")
else:
    MMPROJ_BLOB = None
    print("ℹ projector なし（テキストのみ）")

# ---- 既存プロセスの掃除 ----
# 前回セッションの孤児 llama-server が残っているとポート/VRAM を奪う。
subprocess.run(["pkill", "-TERM", "-f", "llama-server"], capture_output=True)
time.sleep(3)
subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
time.sleep(2)

# ---- llama-server 起動 ----
# LD_LIBRARY_PATH が無いと libllama-server-impl.so を dlopen できず、
# エラーを出さずに CPU 推論へ落ちる（実測 0.8 tps）。
#
# さらに GPU バックエンド（libggml-cuda.so）も見せる必要がある。Ollama は
# バックエンドを `cuda_v12/` のような**サブディレクトリ**に置き、Go 側
# （ollama 本体）がそのパスを明示して読み込む。llama-server を直起動すると
# ggml の探索は実行ファイルの隣（$ORIGIN）だけになり、cuda_v12/ が視界に入らない。
# 実測（2026-10-31・T4 x2）: この状態だと起動はするが CPU 推論になり、
# libllama-common.so が "no usable GPU found" を出す。
#
# GGML_BACKEND_PATH は**バックエンド .so のパス**を 1 本だけ取る
# （コロン区切りを渡すと `load_backend: failed to load <その文字列>` で失敗する。実測）。
# Ollama 本体も子プロセスに同じ env を渡している（`strings bin/ollama` に
# GGML_BACKEND_PATH と libggml-cuda.so が並んで出る）。
#
# 加えて **libcuda.so.1（NVIDIA ドライバ）** も要る。これが LD_LIBRARY_PATH に
# 無いと `libggml-cuda.so: libcuda.so.1: cannot open shared object file` で
# ロードに失敗し、また CPU に落ちる（実測）。場所はイメージによって違うので探す。
_cuda_so = os.path.join(LIB_DIR, "cuda_v12", "libggml-cuda.so")
_driver_dir = None
for _d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
           "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
           "/usr/local/cuda/lib64", "/usr/lib"):
    if os.path.exists(os.path.join(_d, "libcuda.so.1")):
        _driver_dir = _d
        break
if _driver_dir is None:
    # 上の候補に無ければ探す（Kaggle のイメージ差異を吸収）
    _found = subprocess.run(
        ["find", "/usr", "/opt", "-maxdepth", "6", "-name", "libcuda.so.1"],
        capture_output=True, text=True).stdout.split()
    if _found:
        _driver_dir = os.path.dirname(_found[0])

_lib_dirs = [LIB_DIR, os.path.join(LIB_DIR, "cuda_v12")]
if _driver_dir:
    _lib_dirs.append(_driver_dir)
print(f"🧩 libcuda.so.1: {_driver_dir or '⚠ 見つからない（CPU 推論になります）'}")

LIB_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": ":".join(_lib_dirs),
}
if os.path.isfile(_cuda_so):
    LIB_ENV["GGML_BACKEND_PATH"] = _cuda_so
    # 依存 .so が欠けていると dlopen で無言の CPU 落ちになる。先に可視化する
    # （LD_LIBRARY_PATH を渡して、実際に起動するのと同じ条件で見る）。
    _ldd = subprocess.run(["ldd", _cuda_so], capture_output=True, text=True,
                          env=LIB_ENV)
    _missing = [ln.strip() for ln in _ldd.stdout.splitlines() if "not found" in ln]
    print(f"🔗 GPU バックエンド: {os.path.basename(_cuda_so)} "
          f"(ldd: {'不足なし' if not _missing else '不足 ' + '; '.join(_missing)})")

args = [
    LLAMA_SERVER,
    "-m", MODEL_BLOB,
    "-c", str(CONTEXT_LENGTH),
    # KV は q8_0 を明示。f16 既定だと長い ctx で VRAM を溢れて CPU に落ちる。
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "-ngl", "all",
    "-fa", "on",
    "-np", "1",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--jinja",
    "--temp", "0",
    "--alias", MODEL_NAME,
    # thinking を切る。切らないと content が空で reasoning だけ返る。
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--cors-origins", "*",
]

# MTP（投機的デコード）。--mtp は download 専用、--nextn は imatrix 専用。
# サーバーで使う正しいフラグは --spec-type draft-mtp。
if USE_MTP:
    args += ["--spec-type", "draft-mtp"]

if MMPROJ_BLOB:
    args += ["--mmproj", MMPROJ_BLOB]

print()
print("起動コマンド:")
print("  " + " ".join(args))
print()

with open(ENGINE_LOG, "w") as lf:
    proc = subprocess.Popen(args, env=LIB_ENV, stdout=lf, stderr=lf)
print(f"🔄 llama-server PID: {proc.pid}")

# ---- 起動待ち ----
# /health が 200 を返すまで。モデルロード込みで 5〜10 分かかる。
print("⏳ ロード待ち（最大 900s）...")
ready = False
t0 = time.time()
for i in range(180):
    if proc.poll() is not None:
        print(open(ENGINE_LOG).read()[-5000:])
        fail(f"llama-server が即終了しました (rc={proc.returncode})", ENGINE_LOG)
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    if i % 12 == 11:
        print(f"   ... {int(time.time() - t0)}s")
    time.sleep(5)

if not ready:
    print(open(ENGINE_LOG).read()[-6000:])
    fail("llama-server が 900s 以内に応答しませんでした", ENGINE_LOG)

print(f"✅ 起動完了（{int(time.time() - t0)}s）")

# ---- GPU に載っているか（CPU 落ちの検出）----
log_txt = open(ENGINE_LOG).read()
# どのバックエンドが読まれたかは必ず出す（CPU 落ちの切り分けはここで決まる）
for line in log_txt.splitlines():
    if "backend" in line and ("loaded" in line or "failed" in line):
        print("  " + line.strip()[:160])
if "no usable GPU found" in log_txt:
    # GPU バックエンドが 1 つも登録されていない。ログを出してから落とす。
    print(open(ENGINE_LOG).read()[-4000:])
    fail("GPU が認識されていません（CPU 推論に落ちています）", ENGINE_LOG)
if "using CPU" in log_txt:
    print("⚠ 起動ログに 'using CPU' があります")

print()
print("--- 起動ログの要点 ---")
for line in log_txt.splitlines():
    if any(k in line for k in ["CUDA", "offloaded", "MTP", "spec", "draft",
                               "n_ctx", "model size", "using device"]):
        print("  " + line.strip()[:130])

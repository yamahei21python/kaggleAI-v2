# Dataset を展開 → Ollama 起動（コンテキスト長・KV 量子化を明示）
import os, shutil, subprocess, time

os.makedirs(OLLAMA_MODELS, exist_ok=True)

# --- モデルデータの準備 ---
# blobs は「ディレクトリごと symlink」ではなく「実ディレクトリ + ファイル単位
# symlink」にする。dir ごと symlink だと OLLAMA_MODELS 配下が read-only
# （/kaggle/input は読み取り専用マウント）になり、欠けた blob の補完 pull が
# `read-only file system` で即死する（v8 実測）。
import glob as _g, json as _json

def _stage(name, extra_roots=()):
    """DATASET_PATH/name 以下のファイルを実dirとして配置（file symlink）。

    patch Dataset の blobs（config blob sha256-cbe6ad4… 214B が入っている）
    も合流させる。これが無いと `ollama show` が失敗して毎回 pull 誘発。
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
            # manifests は「実ファイル」のみ正。symlink は v11 の残骸なので
            # 作り直す（skip すると同一セッションの再実行で修正が効かない）
            if name == "manifests":
                if os.path.lexists(d) and not os.path.islink(d) and os.path.isfile(d):
                    continue
                if os.path.lexists(d):
                    os.remove(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                # ollama は manifest 自体を symlink 検査し、対象が sha256 blob で
                # ないと拒否する（v11 実測 "manifest symlink target ... is not a
                # sha256 blob" → show 失敗 → pull 誘発）→ 1KB 級なので実体コピー
                shutil.copyfile(p, d)
                n += 1
                continue
            # blobs: 有効な symlink/実ファイルだけスキップ、壊れていれば作り直す
            # （lexists は壊れた symlink も True なので isfile で見分ける）
            if os.path.lexists(d) and os.path.isfile(d):
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                os.symlink(p, d)
            except OSError:
                # 16 GiB を /kaggle/tmp へ複製すると disk full。巨大ファイルは
                # 複製せず落とす（symlink 不可 = 環境異常なので即中断が正解）
                if os.path.getsize(p) > (1 << 30):
                    raise
                shutil.copy2(p, d)
            n += 1
    return dst, n

if os.path.isdir(DATASET_PATH) and os.listdir(DATASET_PATH):
    print("📁 Dataset 検出 → モデルデータを準備中...")
    # 失敗時は例外のまま死なせず、理由を FAILED.txt に残す
    try:
        _, n_blob = _stage("blobs", extra_roots=(PATCH_PATH,))
        _, n_man = _stage("manifests")
    except OSError as _e:
        fail(f"モデルデータ配置に失敗: {type(_e).__name__}: {_e}")
    print(f"✅ blobs {n_blob} 件 / manifests {n_man} 件を配置（patch の config blob 含む）")

    # manifest が要求する blob の充足を先に点検。欠けたまま `ollama show` を
    # 叩くと pull が走り、v8 の失敗経路（16.5 GiB 再DL）へ戻る。
    _repo, _tag = MODEL_NAME.rsplit(":", 1)
    _man = os.path.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai",
                        *_repo.split("/"), _tag)
    if os.path.isfile(_man):
        with open(_man) as _f:
            _md = _json.load(_f)
        _need = [_md.get("config", {})] + _md.get("layers", [])
        _miss = [e["digest"] for e in _need
                 if e.get("digest") and not os.path.lexists(os.path.join(
                     OLLAMA_MODELS, "blobs",
                     "sha256-" + e["digest"].split(":", 1)[1]))]
        if _miss:
            print(f"⚠ 欠落 blob {len(_miss)} 件 → pull フォールバックの可能性: {_miss[:5]}")
        else:
            print("✅ manifest 要求 blob 全数充足 → pull 不要")
    else:
        print(f"⚠ manifest が見つからない: {_man}")
else:
    print(f"⚠ Dataset が見つかりません: {DATASET_PATH}")
    print("  → Add Data で Dataset を追加するか、下で直接 pull します。")

# --- 環境変数 ---
# OLLAMA_CONTEXT_LENGTH : コンテキスト長（未設定だと 4096 に落ちる）
# OLLAMA_KV_CACHE_TYPE  : KV を q8_0 に量子化して VRAM を半分に
# OLLAMA_FLASH_ATTENTION: 有効化
# OLLAMA_KEEP_ALIVE     : -1 で常時ロード維持
env = {
    **os.environ,
    "OLLAMA_MODELS": OLLAMA_MODELS,
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_ORIGINS": "*",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_FLASH_ATTENTION": "1",
    # 既定 5分（stall 検出）。v9 は load 198s で通過したが v10 は超過して
    # "timed out waiting for llama-server to start" → 500 になった。
    # mmap 読込はページキャッシュ次第で揺れるので 10分に緩和（warmup 900s 未満）。
    "OLLAMA_LOAD_TIMEOUT": "600",
    "OLLAMA_NUM_PARALLEL": "1",   # qwen35 は並列が blocklist されているため 1
}

# --- 既存プロセスを掃除 ---
# 前回セッションの孤児 llama-server が残っているとポート/VRAMを
# 奪って本番と競合する。SIGTERM → 待機 → 残存分は SIGKILL。
subprocess.run(["pkill", "-TERM", "-f", "llama-server"], capture_output=True)
time.sleep(3)
subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
time.sleep(2)

OLLAMA_BIN = "/usr/local/bin/ollama"
with open("/tmp/ollama.log", "w") as log:
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)
print(f"🔄 Ollama PID: {proc.pid}")

# --- 起動待機 ---
for i in range(60):
    r = subprocess.run([OLLAMA_BIN, "list"], env=env, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ Ollama サーバー起動完了（{i+1}s）")
        break
    time.sleep(1)
else:
    print(open("/tmp/ollama.log").read()[-3000:])
    fail("Ollama サーバー起動失敗", "/tmp/ollama.log")

# --- モデル確認 ---
r = subprocess.run([OLLAMA_BIN, "show", MODEL_NAME], env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    # なぜ show が失敗したかを残さないと「全数充足なのに pull」が再現不能になる
    print(f"⚠ show 失敗（stderr）: {r.stderr.strip()[:500]}")
    print(f"⏳ Dataset にモデルが無いため pull します（16.5 GiB / 10〜30分）")
    # pull 前に空き容量を確認。16.5 GiB + 退避バッファが無いと
    # 途中で ENOSPC になり、長い待機の後に初めて失敗するのを防ぐ。
    _du = shutil.disk_usage("/kaggle/tmp")
    _need = 18 * (1 << 30)   # 16.5 GiB + 余裕
    print(f"   disk /kaggle/tmp: free={_du.free/(1<<30):.1f} GiB "
          f"(need ~{_need/(1<<30):.0f} GiB)")
    if _du.free < _need:
        fail(f"/kaggle/tmp 空き不足: {_du.free/(1<<30):.1f} GiB "
             f"< 必要 {_need/(1<<30):.0f} GiB")
    # 失敗しても例外のまま死なず、証跡を残してから止める
    # （進捗はセル出力へそのまま流す。pull のエラー表示もそこに残る）
    try:
        pull = subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=env,
                              timeout=7200)
    except subprocess.TimeoutExpired:
        fail("モデル pull が 7200s でタイムアウトしました")
    if pull.returncode != 0:
        fail("モデル pull に失敗しました（詳細はこのセルの出力）")
    print("✅ モデルダウンロード完了")
else:
    print(f"✅ モデル {MODEL_NAME} は利用可能")

print()
print(subprocess.run([OLLAMA_BIN, "list"], env=env,
                     capture_output=True, text=True).stdout)

"""重みロードが「なぜ固まるのか」を GPU 0 秒で測る。

背景（実測 2026-11-01 / v10・v11・v12）
--------------------------------------
同じモデル・同じ symlink ステージング・同じ GCSFUSE なのに:

- v10 … 重みロード 133 秒で成功（`Available RAM` はログに残っていない）
- v11 … `Available RAM: 6.12 GiB` → 840 秒無進捗で試行上限
- v12 … `Available RAM: 5.68 GiB` → 同じく 800 秒超で無進捗（2 試行連続）

vLLM は「checkpoint が空き RAM の 90% を超える」と **auto-prefetch を切る**。
切ると GCSFUSE 上の safetensors を mmap で頁単位に読むしかなく、これが
詰まると tqdm が `50%` で止まったまま何分も動かない。

疑っているのは 3 つ。**どれかを GPU で賭ける前に、ここで切り分ける。**

1. GCSFUSE の読み出し速度が落ちている（ホスト/ネットワーク都合）
2. 読み出し自体が `MemAvailable` を削る（page cache が「空き」を食う）
3. ローカルディスクへ実コピーすれば速い（＝19 GiB のコピーは元が取れる）

測るもの
--------
- `/proc/meminfo`（MemTotal/MemFree/MemAvailable/Cached/Shmem）と `df`
- 位相 1: GCSFUSE から**先頭 N GiB** を逐次読み（streaming）
- 位相 2: GCSFUSE を **mmap して頁を触る**（safetensors と同じ読み方）
- 位相 3: ローカルへ**実コピー**（容量が足りるときだけ）
- 位相 4: コピーを読み直す（ローカルの実力）

各フェーズの前後で meminfo を残す。「読むと空きが減る」なら 2 が原因で、
コピーをしても同じことが起きる（＝コピーでは解決しない）と分かる。
"""

import json
import mmap
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
LOCAL = "/kaggle/tmp/iobench"
DATA = {}

# 先頭何 GiB を読むか。全部読むと 18 GiB × 2 で 5 分以上かかるうえ、
# 速度の推定には先頭だけで足りる（GCSFUSE は定常的）。
HEAD_GIB = float(os.environ.get("KC_HEAD_GIB", "2"))

# 並列度のスケーリングを測る thread 数。v13 の先読みは 8 本。
SCALE_THREADS = (1, 2, 4, 8, 16)


def save():
    DATA["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:                       # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def mem():
    """meminfo の主要値（MiB）。空きだけ見ても原因は決められない。"""
    want = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem", "Dirty")
    out = {}
    try:
        for ln in open("/proc/meminfo", encoding="utf-8"):
            k = ln.split(":", 1)[0]
            if k in want:
                out[k] = int(ln.split()[1]) // 1024
    except OSError:
        pass
    return out


def mem_str(m):
    return (f"空き {m.get('MemAvailable', -1)} / free {m.get('MemFree', -1)}"
            f" / cache {m.get('Cached', -1)} / dirty {m.get('Dirty', -1)}"
            f" / shm {m.get('Shmem', -1)} MiB")


def mark(label):
    """フェーズ境界で meminfo を残す（推移がそのまま診断になる）。"""
    m = mem()
    DATA.setdefault("mem_trace", []).append({"at": label, **m})
    print(f"  🧠 {label}: {mem_str(m)}")
    save()


def find_model():
    """モデル Dataset のディレクトリを探す（マウント先は一定でない）。"""
    cand = os.environ.get("KC_MODEL_DIR")
    if cand and os.path.isdir(cand):
        return cand
    pats = ("/kaggle/input/qwen38-27b*", "/kaggle/input/datasets/*/qwen38-27b*",
            "/kaggle/input/*")
    for pat in pats:
        import glob as _g
        for p in sorted(_g.glob(pat)):
            if os.path.isdir(p) and os.path.isfile(
                    os.path.join(p, "model.safetensors")):
                return p
    return None


def read_head(path, nbytes, buf_mib=64):
    """逐次読み（`readinto`）。GCSFUSE の実効スループット。"""
    buf = bytearray(buf_mib * 1024 * 1024)
    view = memoryview(buf)
    got = 0
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        while got < nbytes:
            n = min(len(buf), nbytes - got)
            r = f.readinto(view[:n])
            if not r:
                break
            got += r
    dt = time.time() - t0
    return got, dt


def read_range(path, off, nbytes, threads=1, chunk_mib=32):
    """`off` から `nbytes` を `threads` 本の `pread` で読む。

    **v13 の `_prewarm_weights()` と同じ形**（32 MiB チャンクをスレッドプールで
    `pread`）にしてあるので、ここで測った値がそのまま先読みの所要になる。
    同じ fd を複数スレッドで `pread` するのは安全（読み取り専用・位置指定）。
    """
    chunk = chunk_mib * 1024 * 1024
    total = 0
    t0 = time.time()
    fd = os.open(path, os.O_RDONLY)
    try:
        def _one(o):
            n = min(chunk, off + nbytes - o)
            if n <= 0:
                return 0
            return len(os.pread(fd, n, o))

        with ThreadPoolExecutor(max_workers=threads) as ex:
            for got in ex.map(_one, range(off, off + nbytes, chunk)):
                total += got
    finally:
        os.close(fd)
    return total, time.time() - t0


def touch_mmap(path, nbytes, step=4096):
    """mmap して頁を触る（safetensors と同じ読み方）。

    `step` ごとに 1 バイト読むだけでは「頁を触った」ことにならないので、
    頁の先頭を読む。読んだ値は捨てない（最適化で消されないように合算）。
    """
    acc = 0
    t0 = time.time()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            n = min(nbytes, len(mm))
            for off in range(0, n, step):
                acc += mm[off]
        finally:
            mm.close()
    dt = time.time() - t0
    return acc, dt


def main():
    DATA["stage"] = "env"
    DATA["python"] = sys.version.split()[0]
    DATA["head_gib"] = HEAD_GIB
    save()

    print("=" * 60)
    print("  I/O bench（GPU 0 秒）: GCSFUSE の実力と RAM の推移")
    print("=" * 60)
    mark("start")
    for p in ("/kaggle/input", "/kaggle/tmp", "/kaggle/working"):
        rc, out = sh(["df", "-h", p])
        DATA.setdefault("df", {})[p] = out.strip().splitlines()[-1] if out else None
        print(f"  💾 {p}: {DATA['df'][p]}")

    model = find_model()
    DATA["model_dir"] = model
    save()
    if not model:
        print("❌ モデル Dataset が見つからない（dataset_sources を確認）")
        DATA["ok"] = False
        save()
        return 1
    fname = "model.safetensors"
    src = os.path.join(model, fname)
    size = os.path.getsize(src)
    DATA["model_bytes"] = size
    DATA["model_gib"] = round(size / 2**30, 2)
    print(f"  📦 {src}  {DATA['model_gib']} GiB")
    rc, out = sh(["stat", "-f", "-c", "%T", model])
    DATA["fs_type"] = out.strip()
    print(f"  🗂  FS: {DATA['fs_type']}")
    save()

    nbytes = int(HEAD_GIB * 2**30)

    # ---- 位相 1: GCSFUSE 逐次読み ----
    DATA["stage"] = "read_fuse"
    save()
    print(f"\n[1] GCSFUSE 逐次読み {HEAD_GIB} GiB")
    got, dt = read_head(src, nbytes)
    DATA["read_fuse"] = {"bytes": got, "seconds": round(dt, 2),
                         "mib_s": round(got / 2**20 / dt, 1) if dt else None}
    print(f"    {got / 2**20:.0f} MiB / {dt:.1f}s = "
          f"{DATA['read_fuse']['mib_s']} MiB/s"
          f"  → 17.41 GiB なら {size / 2**20 / max(1, DATA['read_fuse']['mib_s'] or 1):.0f}s")
    mark("after_read_fuse")

    # ---- 位相 2: GCSFUSE mmap（safetensors と同じ）----
    DATA["stage"] = "mmap_fuse"
    save()
    print(f"\n[2] GCSFUSE を mmap して頁を触る {HEAD_GIB} GiB")
    acc, dt = touch_mmap(src, nbytes)
    DATA["mmap_fuse"] = {"bytes": nbytes, "seconds": round(dt, 2),
                         "mib_s": round(nbytes / 2**20 / dt, 1) if dt else None,
                         "acc": acc & 0xFF}
    print(f"    {nbytes / 2**20:.0f} MiB / {dt:.1f}s = "
          f"{DATA['mmap_fuse']['mib_s']} MiB/s")
    mark("after_mmap_fuse")

    # ---- 位相 3: ローカルへ実コピー ----
    DATA["stage"] = "copy_local"
    save()
    free = shutil.disk_usage("/kaggle/tmp").free if os.path.isdir("/kaggle/tmp") else 0
    DATA["kaggle_tmp_free_bytes"] = free
    need = int(size * 1.05)
    print(f"\n[3] ローカルへ実コピー（空き {free / 2**30:.1f} GiB / "
          f"必要 {need / 2**30:.1f} GiB）")
    if free < need:
        print("    ⏭ 容量不足 → スキップ（symlink のまま運用するしかない）")
        DATA["copy_local"] = {"skipped": "no space", "free": free}
    else:
        os.makedirs(LOCAL, exist_ok=True)
        dst = os.path.join(LOCAL, fname)
        t0 = time.time()
        with open(src, "rb", buffering=0) as fi, open(dst, "wb", buffering=0) as fo:
            shutil.copyfileobj(fi, fo, 64 * 1024 * 1024)
        dt = time.time() - t0
        DATA["copy_local"] = {"seconds": round(dt, 2),
                              "mib_s": round(size / 2**20 / dt, 1) if dt else None,
                              "bytes": os.path.getsize(dst)}
        print(f"    {size / 2**20:.0f} MiB / {dt:.1f}s = "
              f"{DATA['copy_local']['mib_s']} MiB/s")
        mark("after_copy_local")

        # ---- 位相 4: コピーを読み直す ----
        DATA["stage"] = "read_local"
        save()
        print(f"\n[4] コピーを読み直す {HEAD_GIB} GiB（ローカルの実力）")
        got2, dt2 = read_head(dst, nbytes)
        DATA["read_local"] = {"bytes": got2, "seconds": round(dt2, 2),
                              "mib_s": round(got2 / 2**20 / dt2, 1) if dt2 else None}
        print(f"    {got2 / 2**20:.0f} MiB / {dt2:.1f}s = "
              f"{DATA['read_local']['mib_s']} MiB/s")
        mark("after_read_local")

    # ---- 位相 5: 並列度のスケーリング ----
    # v13 で入れた `_prewarm_weights()` は「8 並列で読めば 140 MiB/s 出る」という
    # **推定**に乗っている（v10 の重み 133s = 18.21 GiB / 133s）。この推定が
    # 外れていれば、先読みは 407 秒かかるだけで何も改善しない。
    # ここで並列度ごとの実効スループットを測り、推定を実測に置き換える。
    DATA["stage"] = "scale"
    save()
    print("\n[5] 並列度のスケーリング（1 GiB の窓を thread 数ごとに読む）")
    win = 1 * 2**30
    # 位相 1/2 で 0〜2 GiB は cache に載っている。窓は 4 GiB 以降へずらし、
    # 1 回目と 2 回目で**別の窓**を使う（同じ窓だと 2 周目が cache hit になる）。
    windows = [4 * 2**30 + i * win for i in range(10)]
    plan = [(t, 0) for t in SCALE_THREADS] + [(t, 1) for t in reversed(SCALE_THREADS)]
    DATA["scale"] = []
    for t, _pass in plan:
        off = windows.pop(0)
        if off + win > size:
            break
        got, dt = read_range(src, off, win, threads=t)
        rec = {"threads": t, "bytes": got, "seconds": round(dt, 2),
               "mib_s": round(got / 2**20 / dt, 1) if dt else None}
        DATA["scale"].append(rec)
        print(f"    {t:2d} 本: {rec['mib_s']} MiB/s ({rec['seconds']}s)")
        save()
    mark("after_scale")

    # ---- 位相 6: drop_caches は許されるか ----
    # v13 は起動直前に `/proc/sys/vm/drop_caches` へ書いて空きを戻そうとする。
    # **書けない環境なら、この対策は丸ごと無効**（GPU で初めて気づくと 1 往復無駄）。
    # ここで権限と効果（MemAvailable が実際に戻るか）を確かめる。
    DATA["stage"] = "drop"
    save()
    print("\n[6] drop_caches の可否と効果")
    before = mem()
    res = {"before": before}
    try:
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
            f.write("1\n")
        res["ok"] = True
    except OSError as e:
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    res["after"] = mem()
    res["delta_avail"] = (res["after"].get("MemAvailable", 0)
                          - before.get("MemAvailable", 0))
    res["delta_cached"] = (res["after"].get("Cached", 0)
                           - before.get("Cached", 0))
    DATA["drop_caches"] = res
    print(f"    書けた: {res['ok']}"
          + ("" if res["ok"] else f" ({res['error']})")
          + f" / 空き {before.get('MemAvailable', -1)} → "
            f"{res['after'].get('MemAvailable', -1)} MiB"
            f" / cache {before.get('Cached', -1)} → {res['after'].get('Cached', -1)}")

    # fadvise は drop_caches が使えないときの代替（プロセス権限だけで効く）。
    try:
        fd = os.open(src, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            DATA["fadvise"] = {"ok": True}
        finally:
            os.close(fd)
    except (OSError, AttributeError) as e:
        DATA["fadvise"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(f"    fadvise(DONTNEED): {DATA['fadvise']}")

    # ---- 判定 ----
    rf = (DATA.get("read_fuse") or {}).get("mib_s") or 0
    rl = (DATA.get("read_local") or {}).get("mib_s") or 0
    best = max((r["mib_s"] or 0) for r in DATA.get("scale") or [{"mib_s": 0}])
    best_t = next((r["threads"] for r in DATA.get("scale") or []
                   if r["mib_s"] == best), None)
    DATA["verdict"] = {
        "fuse_mib_s": rf,
        "local_mib_s": rl,
        "fuse_seconds_for_17gib": round(size / 2**20 / rf, 1) if rf else None,
        "local_seconds_for_17gib": round(size / 2**20 / rl, 1) if rl else None,
        "copy_worth_it": bool(rl and rf and rl > rf * 1.5),
        "best_parallel_mib_s": best,
        "best_parallel_threads": best_t,
        "parallel_speedup": round(best / rf, 2) if rf and best else None,
        "parallel_seconds_for_17gib": round(size / 2**20 / best, 1) if best else None,
    }
    print("\n" + "=" * 60)
    print(f"  判定: GCSFUSE 1 本 {rf} MiB/s / ローカル {rl} MiB/s"
          f" → コピーは{'元が取れる' if DATA['verdict']['copy_worth_it'] else '元が取れない'}")
    print(f"        並列 最良 {best} MiB/s（{best_t} 本）"
          f" = 1 本の {DATA['verdict']['parallel_speedup']} 倍"
          f" → 17.41 GiB で {DATA['verdict']['parallel_seconds_for_17gib']}s")
    print("=" * 60)

    DATA["stage"] = "done"
    DATA["ok"] = True
    save()
    # 19 GiB を残すと出力の取得に時間がかかるので消す（symlink ではない実体）。
    if os.path.isdir(LOCAL):
        shutil.rmtree(LOCAL, ignore_errors=True)
        print(f"  🧽 コピーを削除: {LOCAL}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        DATA["ok"] = False
        DATA["error"] = f"{type(e).__name__}: {e}"
        save()
        sys.exit(1)

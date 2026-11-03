# ============================================================
# Step 2 (vLLM): 配信カーネルの設定
# ============================================================
import json
import os
import subprocess

TIMINGS_PATH = "/kaggle/working/timings.json"


def kc_save_timings(d):
    """計測値を timings.json に足す（result.json へはテンプレートの save() が合流）。

    セルから `data` を直接触らないのは、**notebook 版には `data` が無い**ため
    （セル順の「定義前使用」で落ちる。2026-11-01 実測）。ファイル経由なら
    notebook / script の両方で同じコードが動く。

    計測は「高度化の基礎」なので、失敗しても本処理は止めない。
    """
    try:
        with open(TIMINGS_PATH, encoding="utf-8") as f:
            cur = json.load(f)
    except (OSError, ValueError):
        cur = {}
    cur.update(d)
    try:
        with open(TIMINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except OSError as _e:
        print(f"⚠ timings.json を書けません: {_e}")

# Dataset のマウント先は共通ランタイム（runtime/storage.py）に一本化。
# 旧 /kaggle/input/<slug> と新 /kaggle/input/datasets/<owner>/<slug> の
# 両対応・空マウント回避は kc_find_dataset が持つ。
MODEL_SRC = kc_find_dataset("__DATASET_SLUG__")
# 共有セルは MODEL_NAME を読む。vLLM の文脈では served-model-name と同義。
SERVED_NAME = MODEL_NAME = "__SERVED__"
MODEL_DISPLAY = "Qwen3.8-27B-Uncensored (AWQ)"
ENGINE_DISPLAY = "vLLM"
ENGINE_LABEL = "vllm"

# ---- ポート ----
# vLLM 本体は 8000、公開側はプロキシが 11435 で受ける。
PORT = UPSTREAM_PORT = VLLM_PORT = 8000
PROXY_PORT = 11435
PROXY_CHECK_PATH = "/v1/models"
# クライアントが temperature=0 / repetition_penalty 未指定で送っても
# 崩壊しないよう、サーバ側で下限を矯正する（RP>=1.05 / temp>0）。
PROXY_EXTRA_ARGS = ("--enforce-sampling",)
TUNNEL_KIND = "__TUNNEL__"

# Basic 認証プロキシ / UI のソース（ビルダーが埋め込む）
PROXY_SRC_B64 = "__PROXY_B64__"
CHAT_SRC_B64 = "__CHAT_B64__"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

# pip に渡す vLLM 指定（切り戻したいときは "vllm==0.10.2" 等に変える）
VLLM_PIP_SPEC = os.environ.get("VLLM_PIP_SPEC", "vllm==0.32.0")


_gpu_names, _vram_each = kc_detect_gpu()
VRAM_TOTAL = sum(_vram_each)
print(f"GPU      : {_gpu_names}")
print(f"VRAM     : {_vram_each} MiB each  →  合計 {VRAM_TOTAL} MiB")

# /dev/shm は TP>1 の必須資源。vLLM のリングバッファは既定で
# (24 MiB + 数 B) x 10 ブロック ≈ 240 MiB を 1 本で確保する。
# 足りなければ即 RuntimeError になるが、**境界値だと起動後に詰まる**（実測:
# 32K ctx で "No available shared memory broadcast block found" が 60s 間隔で連発）。
def _shm_required_mib(tp):
    """/dev/shm の必要量（MiB）の目安。TP 本ぶん + 2 倍の余裕。

    1 本あたり 24 MiB x 10 ブロック ≒ 240 MiB。TP=2 なら 480 MiB、
    余裕を 2 倍見て 960 MiB。**空き**で判定する（total では足りない。
    他プロセスの残骸が掴んでいる場合がある）。
    """
    return 240 * max(1, int(tp)) * 2


SHM_FREE_MIB = SHM_TOTAL_MIB = 0
try:
    _shm = os.statvfs("/dev/shm")
    SHM_FREE_MIB = _shm.f_bavail * _shm.f_frsize // (1 << 20)
    SHM_TOTAL_MIB = _shm.f_blocks * _shm.f_frsize // (1 << 20)
    print(f"/dev/shm : {SHM_TOTAL_MIB} MiB（空き {SHM_FREE_MIB} MiB）"
          f"  cpu={os.cpu_count()}")
except Exception as _e:
    print(f"/dev/shm : 取得失敗 {_e!r}")

if VRAM_TOTAL == 0:
    raise RuntimeError(
        "GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")
if len(_vram_each) < 2:
    print("⚠ GPU が 1 枚です。Tensor Parallel は 1 に落とします"
          "（27B AWQ は 1 枚に収まらない可能性が高い）")
TP_SIZE = 2 if len(_vram_each) >= 2 else 1

# 必要量は TP が決まってから判定する（TP に比例するため）。
SHM_REQUIRED_MIB = _shm_required_mib(TP_SIZE)
if SHM_TOTAL_MIB and SHM_FREE_MIB < SHM_REQUIRED_MIB:
    print(f"⚠ /dev/shm の空き {SHM_FREE_MIB} MiB < 必要 {SHM_REQUIRED_MIB} MiB"
          f"（TP={TP_SIZE}）→ 拡張を試みる")
    try:
        _rm = subprocess.run(
            ["mount", "-o", "remount,size=8G", "/dev/shm"],
            capture_output=True, text=True, timeout=60)
        print(f"   remount rc={_rm.returncode} {( _rm.stderr or '').strip()[:120]}")
    except Exception as _e:
        print(f"   remount 失敗: {_e!r}")
    try:
        _shm = os.statvfs("/dev/shm")
        SHM_FREE_MIB = _shm.f_bavail * _shm.f_frsize // (1 << 20)
        SHM_TOTAL_MIB = _shm.f_blocks * _shm.f_frsize // (1 << 20)
        print(f"/dev/shm : 再測定 {SHM_TOTAL_MIB} MiB（空き {SHM_FREE_MIB} MiB）")
    except Exception as _e:
        print(f"/dev/shm : 再測定失敗 {_e!r}")
    if SHM_FREE_MIB < SHM_REQUIRED_MIB:
        # ここで落とす。放置すると起動はするが forward で詰まり、
        # 待機側が 40 分待つ（1 往復を丸損する）。
        raise RuntimeError(
            f"/dev/shm が不足しています（空き {SHM_FREE_MIB} MiB / 必要 "
            f"{SHM_REQUIRED_MIB} MiB, TP={TP_SIZE}）。TP=1 に落とすか、"
            "max-model-len を下げてください。")

# ---- 起動条件 ----
# 27B AWQ は約 19 GB。T4 x2 = 30 GB のうち 0.90 = 27 GB を使う想定で、
# 残り ~8 GB が KV cache になる。実際に収まるかは config 次第なので、
# 起動失敗時に max-model-len を落として再試行する（serve セル）。
# 2026-10-29 机上計算: 重み 18.20 GiB + KV f16 68 KiB/token。
# 64K=22.45 GiB（予算 27.0 で余裕 4.5）、96K=24.58（余裕 2.4）、
# 128K=26.70（余裕 0.3 = graph/activation で確実に溢れる）。
# T4 は fp8 KV 不可なので Ollama の q8_0（34 KiB/token）のようには詰められない。
# → 上限狙いは 64K。実測で 1〜2 GiB ずれるので 32K/16K も残す。
MAXLEN_CANDIDATES = [65536, 32768, 16384, 8192]
MAX_MODEL_LEN = MAXLEN_CANDIDATES[0]
# 共有セル（接続情報）が読む。起動成功後は実際の max_model_len で上書きする。
CONTEXT_LENGTH = MAX_MODEL_LEN
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS = 2
KV_CACHE_DTYPE = "auto"        # T4 は FP8 非対応。fp8 を指定してはいけない

# MTP（multi-token prediction）は任意。失敗したら自動で無効にして再起動する。
MTP_ENABLED = os.environ.get("VLLM_MTP", "1") != "0"
MTP_METHOD = os.environ.get("VLLM_SPEC_METHOD", "mtp")
MTP_TOKENS = int(os.environ.get("VLLM_SPEC_TOKENS", "2"))

# ---- 認証 ----
# 既定で無効。API_PASS が入っているときだけ proxy が有効化する。
PROXY_AUTH = {
    "API_USER": os.environ.get("API_USER", "") or "ollama",
    "API_PASS": os.environ.get("API_PASS", ""),
}

# ---- アカウント（ビーコンの slug に使う。config.env の KAGGLE_OWNER）----
KAGGLE_OWNER = "__KAGGLE_OWNER__"

# ---- キープアライブ ----
# 上限。チャット 1 回で proxy が MINUTES=60 を書くため、上限が無いと
# 「1 回送信 → 60 分に延長」で残クォータを超えて強制終了される
# （強制終了は成果物の zip が保証されない）。検証中は 30 分で頭打ちにする。
MAX_MINUTES = 30
KEEPALIVE_RESET_MINUTES = 60      # チャット送信ごとにリセットする時間（分）
TUNNEL_CHECK_EVERY = 10           # 公開URL疎通は 10 分ごと

# ---- 死活監視 / 後片付けの能力表（共有セルが読む）----
ENGINE_LOG = "/tmp/vllm_1.log"
ENGINE_PROCS = [("vllm serve", "argv"), ("auth_proxy.py", "argv")]
CLEANUP_TARGETS = [
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f vllm serve", "vllm"),
    ("-f VLLM::EngineCore", "engine-core"),
]


def engine_cleanup():
    """次回起動を短くするため compile / Triton キャッシュを退避する。

    VRAM の解放はプロセス停止（CLEANUP_TARGETS が EngineCore まで落とす）で行う。
    3GB を超える場合は退避しない（転送と容量の無駄）。
    """
    try:
        import tarfile as _tf
        keep = []
        for _d in ("/root/.cache/vllm", "/root/.triton", "/root/.cache/triton"):
            if os.path.isdir(_d) and not any(_d.startswith(k + "/") for k in keep):
                keep.append(_d)
        size = 0
        for _d in keep:
            for _root, _dirs, _files in os.walk(_d):
                for _f in _files:
                    try:
                        size += os.path.getsize(os.path.join(_root, _f))
                    except OSError:
                        pass
        print(f"🧠 キャッシュ {size/1e6:.0f} MB {keep}")
        if keep and size and size < 3e9:
            out = os.path.join(OUT_DIRS[0], "warm-cache.tar")
            with _tf.open(out, "w") as _t:
                for _d in keep:
                    _t.add(_d, arcname=os.path.relpath(_d, "/"))
            print(f"   退避: {out} {os.path.getsize(out)/1e6:.0f} MB")
            return f"cache_tar={os.path.getsize(out)}"
        return "cache_skipped" if size else "cache_none"
    except Exception as _e:
        print(f"⚠ キャッシュ退避をスキップ: {_e}")
        return f"cache_error={type(_e).__name__}"


# ---- 検証成果物の永続化 ----
# `kaggle kernels output` が zip 化するのは /kaggle/working のみ。
OUT_DIRS = KC_OUT_DIRS
save_output = kc_save_output
fail = kc_fail


print()
print("=" * 56)
print(f"  モデル      : {MODEL_SRC}")
print(f"  配信名      : {SERVED_NAME}")
print(f"  TP          : {TP_SIZE}")
print(f"  MTP         : {'ON' if MTP_ENABLED else 'OFF'}")
print(f"  公開ポート  : {PROXY_PORT}（vLLM 本体は {VLLM_PORT}）")
print(f"  認証        : {'ON' if PROXY_AUTH['API_PASS'] else 'OFF（URL を知れば誰でも利用可）'}")
print("=" * 56)

_files = sorted(os.listdir(MODEL_SRC)) if os.path.isdir(MODEL_SRC) else []
print(f"  Dataset 直下: {len(_files)} 件")
for _f in _files[:12]:
    print(f"    - {_f}")
if "config.json" not in _files:
    fail("config.json が Dataset 直下にありません（Add Data を確認）")
if "model.safetensors" not in _files:
    fail("model.safetensors が Dataset 直下にありません（Add Data を確認）")


# ---- 起動前の I/O probe（ノードの実効速度を測る）--------------------
# 同じコードでも重みロードが 1,304〜6,128 秒と 4.7 倍振れる（notes §25）。
# 差の大半は**ノードの GCSFUSE 実効速度**で、こちらからは選べない。
# そこで起動前に 64 MiB を 1 本で逐次読みし、19.0 GiB を割って予算に
# 収まるかを見る。遅いノードを引いた回に 100 分焼かず、30 秒で諦めるため。
#
# `KC_PROBE_ONLY=1` なら**計測だけして終了**する（install の 270 秒も払わない）。
# 10 回まわして「速いノード / 遅いノード」をどれだけ弾けるかの実測に使う。
def _kc_probe_safetensors():
    """Dataset 直下の `.safetensors` を (合計バイト, 最大ファイル) で返す。"""
    total, biggest = 0, (0, "")
    try:
        names = sorted(os.listdir(MODEL_SRC))
    except OSError:
        return 0, (0, "")
    for fn in names:
        if not fn.endswith(".safetensors"):
            continue
        p = os.path.join(MODEL_SRC, fn)
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        total += size
        if size > biggest[0]:
            biggest = (size, p)
    return total, biggest


def _kc_probe_io(mib=64):
    """最大の safetensors の末尾を `mib` MiB 読んで実効速度を返す（1 本）。"""
    import time

    total, (size, path) = _kc_probe_safetensors()
    if not path:
        return {"ok": False, "why": "safetensors が無い"}
    n = min(int(mib) * 2**20, size)
    off = max(0, size - n)                 # 末尾（本読みと窓をずらす）
    t0 = time.time()
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            got = len(os.pread(fd, n, off))
        finally:
            os.close(fd)
    except OSError as e:
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}
    dt = time.time() - t0
    if not got or dt <= 0:
        return {"ok": False, "why": "0 バイト"}
    rate = got / 2**20 / dt
    projected = total / 2**20 / rate
    limit = int(os.environ.get("KC_PROBE_LIMIT_S", "6400"))
    return {
        "ok": True,
        "file": os.path.basename(path),
        "read_mib": round(got / 2**20, 1),
        "seconds": round(dt, 3),
        "mib_s": round(rate, 2),
        "total_mib": round(total / 2**20, 1),
        "projected_s": round(projected),
        # 重みロード以外（install / compile / capture）に約 900 秒みる
        "verdict": "pass" if projected + 900 <= limit else "reject",
        "limit_s": limit,
        "mem_available_mib": None,
    }


def _kc_probe_report(rec):
    if not rec.get("ok"):
        print(f"🔎 probe 失敗: {rec.get('why')}")
        return
    print(f"🔎 ノード probe: {rec['read_mib']} MiB / {rec['seconds']}s = "
          f"**{rec['mib_s']} MiB/s**")
    print(f"   19.0 GiB 相当 {rec['total_mib']} MiB → 重み {rec['projected_s']} 秒"
          f"（上限 {rec['limit_s']} 秒 + 固定 900 秒） → {rec['verdict'].upper()}")
    print("PROBE_JSON " + json.dumps(rec, ensure_ascii=False))


def _kc_probe_gate(rec, gate):
    """probe 結果がゲートを満たすか。`gate <= 0` は常に通す。

    判定に使うのは **1 本読みの速度だけ**。重みロード速度の予測ではない
    （notes §28: 相関は未検証）。install（267 秒）に入る前に降りるための
    足切りで、遅いノードを引いた回に 100 分焼かないためのもの。
    probe が失敗した回は判定できないので通す（安全側 = 走らせる）。
    """
    if gate <= 0 or not rec.get("ok"):
        return True
    return rec["mib_s"] >= gate


def _kc_probe_mem_mib():
    """`MemAvailable`（MiB）。03 の `_mem_info` は**まだ定義されていない**ので使わない。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as _mf:
            for _ln in _mf:
                if _ln.startswith("MemAvailable:"):
                    return int(_ln.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


_KC_PROBE = _kc_probe_io()
_KC_PROBE["mem_available_mib"] = _kc_probe_mem_mib()
_kc_probe_report(_KC_PROBE)
kc_save_timings({"io_probe": _KC_PROBE})
try:
    with open("/kaggle/working/probe.json", "w", encoding="utf-8") as _f:
        json.dump(_KC_PROBE, _f, ensure_ascii=False, indent=2)
except OSError as _e:
    print(f"⚠ probe.json を書けません: {_e}")

if os.environ.get("KC_PROBE_ONLY", "").strip().lower() not in ("", "0", "false", "no"):
    print("⏹ KC_PROBE_ONLY=1 → 計測だけして終了（install 以降は走らせない）")
    raise SystemExit(0)

# ---- ノード速度ゲート（`cli race` 用）--------------------------------
# `KC_PROBE_GATE_MIB_S` 未満のノードは **install に入る前**に降りる。
# 2 本同時に push し、速いノードを引くまで繰り返すレースの土台。
# 0 / 未設定なら無効（従来どおり全部走る）。
_KC_GATE = float(os.environ.get("KC_PROBE_GATE_MIB_S", "0") or 0)
if _KC_GATE > 0:
    _kc_gate_pass = _kc_probe_gate(_KC_PROBE, _KC_GATE)
    kc_save_timings({"probe_gate": {
        "gate_mib_s": _KC_GATE,
        "mib_s": _KC_PROBE.get("mib_s"),
        "pass": _kc_gate_pass,
    }})
    print(f"🚦 ノードゲート: {_KC_PROBE.get('mib_s')} MiB/s "
          f"{'≥' if _kc_gate_pass else '<'} {_KC_GATE} → "
          f"{'続行' if _kc_gate_pass else '早期終了（install 前）'}")
    if not _kc_gate_pass:
        raise SystemExit(0)

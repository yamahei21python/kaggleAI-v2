"""プロセス・GPU 状態の確認と後片付け（生成コードへ埋め込む）。"""
import re
import subprocess
import time


def _kc_live(pids, timeout=5):
    """zombie を除いた PID を返す。kill 直後は未 reap の Z が pgrep に残る。"""
    if not pids:
        return []
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,stat=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except Exception:
        return pids
    live = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[1].startswith("Z"):
            live.append(parts[0])
    return live


def kc_proc_state(name, match="exact", timeout=5):
    """alive / DEAD / unknown。match=exact は pgrep -x、argv は pgrep -f。"""
    try:
        flag = "-x" if match == "exact" else "-f"
        out = subprocess.run(["pgrep", flag, name], capture_output=True,
                             text=True, timeout=timeout).stdout
        pids = [p for p in out.split() if p.strip()]
        return "alive" if _kc_live(pids, timeout) else "DEAD"
    except Exception:
        return "unknown"


def kc_gpu_usage(timeout=5):
    """nvidia-smi の使用率/VRAM を 1 行で返す（CPU 環境では N/A）。"""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip().replace("\n", " | ")
    except Exception:
        return "N/A"


def kc_detect_gpu(timeout=15):
    """全 GPU の (名前, VRAM MiB) を返す。検出失敗は ([], [])。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip()
        names, mems = [], []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            names.append(parts[0].strip())
            mems.append(int(re.sub(r"[^0-9]", "", parts[1])))
        return names, mems
    except Exception as e:
        print(f"GPU 検出失敗: {e}")
        return [], []


def kc_wait_gpu_idle(limit_mib=500, timeout=180):
    """全 GPU の使用量が limit 未満になるまで待つ。経過秒、失敗は -1。

    名前は ollama の kc_wait_vram_free（/api/ps を見る）と衝突させない。
    """
    for i in range(timeout):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        used = [int(x) for x in out.splitlines() if x.strip().isdigit()]
        if used and max(used) < limit_mib:
            return i
        time.sleep(1)
    return -1


def _kc_argv(pattern):
    """"-x ngrok" → ["-x", "ngrok"]（shell を介さない）。"""
    return pattern.split()


def kc_pids(pattern, timeout=15):
    """pgrep の引数文字列（例: "-x ngrok"）に一致する PID を返す。"""
    try:
        r = subprocess.run(["pgrep", *_kc_argv(pattern)],
                           capture_output=True, text=True, timeout=timeout)
        return _kc_live([p for p in r.stdout.split() if p.strip()], timeout)
    except Exception:
        return []


def kc_stop_processes(targets, term_wait=3, final_wait=2):
    """(pgrep_pattern, label) を TERM → 待機 → KILL で止め、残存ラベルを返す。"""
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-TERM", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  TERM → {label}")
    time.sleep(term_wait)
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-KILL", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  KILL → {label}（残存していた）")
    time.sleep(final_wait)
    return [label for pattern, label in targets if kc_pids(pattern)]

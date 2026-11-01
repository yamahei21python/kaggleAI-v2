#!/usr/bin/env python3
"""vLLM wheelhouse の解決可否を CPU（無料）で調べる。

目的:
  `vllm-wheelhouse`（本体）+ `vllm-wheelhouse-patch`（欠けている wheel）から
  `--no-index --find-links` で vllm==0.32.0 を入れられるか。
  駄目なら**何が足りないか**を特定する。install は dry-run のみ（GPU 0 秒）。

出力: /kaggle/working/result.json
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
SLUGS = ("vllm-wheelhouse", "vllm-wheelhouse-patch")
DATA = {"stage": "init", "ok": False}


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, time.time() - t0, (p.stdout or ""), (p.stderr or "")


def find_dataset(slug):
    cands = [f"/kaggle/input/{slug}",
             f"/kaggle/input/datasets/yamahei21python/{slug}"]
    for base in ("/kaggle/input", "/kaggle/input/datasets"):
        try:
            for owner in os.listdir(base):
                cands.append(os.path.join(base, owner, slug))
        except OSError:
            pass
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def _flat_wheels(path):
    return glob.glob(os.path.join(path, "*.whl"))


def wheel_dir(root):
    """zip 置き場 / 展開済み / サブディレクトリを吸収して wheel 置き場を返す。"""
    if not root:
        return None
    if _flat_wheels(root):
        return root
    zips = (glob.glob(os.path.join(root, "*.zip"))
            + glob.glob(os.path.join(root, "*", "*.zip")))
    if zips:
        dst = os.path.join("/kaggle/tmp", os.path.basename(root) + "-whl")
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        for z in zips:
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)
        print(f"zip 展開 {os.path.basename(root)}: {time.time() - t0:.1f}s")
        root = dst
    if _flat_wheels(root):
        return root
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.whl")):
            return sub
    return None


def main():
    DATA["python"] = sys.version.split()[0]
    dirs = []
    for slug in SLUGS:
        root = find_dataset(slug)
        print(f"{slug}: {root}")
        d = wheel_dir(root)
        if d is None:
            # 期待構成（本体 + patch）の片方でも欠けたら解決は成立しない。
            raise RuntimeError(f"{slug} に wheel が無い（両方必要）")
        n = len(_flat_wheels(d))
        size = sum(os.path.getsize(w) for w in _flat_wheels(d))
        print(f"  {n} wheel / {size / 1e9:.2f} GB: {d}")
        DATA.setdefault("datasets", {})[slug] = {"dir": d, "n": n,
                                                 "bytes": size}
        dirs.append(d)

    find = []
    for d in dirs:
        find += ["--find-links", d]
    names = [os.path.basename(w) for d in dirs for w in _flat_wheels(d)]
    DATA["cublas_versions"] = sorted(
        n for n in names if "nvidia_cublas" in n)
    DATA["stage"] = "inventory"
    save()

    # ---- pip の dry-run 解決（依存だけ。install しない）----
    rc, dt, out, err = sh([sys.executable, "-m", "pip", "install", "--dry-run",
                           "--no-input", "--no-index"] + find
                          + ["vllm==0.32.0"], timeout=1800)
    text = out + err
    print(f"\n[pip dry-run] rc={rc} {dt:.1f}s")
    print(text[-4000:])
    DATA["pip_dry_run"] = {"rc": rc, "seconds": round(dt, 1),
                           "tail": text[-2500:]}
    DATA["pip_ok"] = rc == 0
    DATA["missing"] = sorted(set(re.findall(
        r"No matching distribution found for ([^\s(]+)", text)))
    DATA["stage"] = "pip-dry-run"
    save()

    # ---- uv でも同じ解決を試す（解決だけなら 1s 未満）----
    rc_u, _dt_u, _o, _e = sh([sys.executable, "-m", "pip", "install", "-q",
                              "uv"], timeout=600)
    if rc_u == 0:
        rc2, dt2, out2, err2 = sh(["uv", "pip", "install", "--system",
                                   "--dry-run", "--no-index"] + find
                                  + ["vllm==0.32.0"], timeout=1800)
        t2 = out2 + err2
        print(f"\n[uv dry-run] rc={rc2} {dt2:.1f}s")
        print(t2[-3000:])
        DATA["uv_dry_run"] = {"rc": rc2, "seconds": round(dt2, 1),
                              "tail": t2[-2000:]}
        DATA["uv_ok"] = rc2 == 0
    else:
        DATA["uv_dry_run"] = {"rc": rc_u, "tail": (_o + _e)[-500:]}

    DATA["stage"] = "done"
    # 実運用は pip（uv は情報のみ）。合格判定は pip の解決だけ。
    DATA["ok"] = bool(DATA.get("pip_ok"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        DATA["stage"] = "failed"
        DATA["error"] = f"{type(e).__name__}: {e}"
    save()
    print("\nRESULT: " + json.dumps(DATA, ensure_ascii=False)[:4000])
    sys.exit(0 if DATA.get("ok") else 1)

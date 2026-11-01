#!/usr/bin/env python3
"""install-bench: vLLM の install を「展開コスト」として測る（GPU 0 秒）。

なぜ要るか
----------
実測 2026-11-01（v10）: install 271.3s のうち **pip が 230.4s**。
しかも wheelhouse Dataset（190 + 11 wheel / `--no-index`）を使った回の方が
PyPI 経路（199.6s）より **31 秒遅かった**。つまり

    「3.94 GB の転送が支配的」という前提が外れている

転送を消しても縮まない＝**3.25 GB の展開と数万個の小ファイル書き込み**が床。
ここを動かせるのは「展開の仕方」を変える道具だけ（uv / --no-compile /
解決を飛ばす）なので、**GPU を使わずに CPU で 4 通りを並べて測る**。

測り方
------
`pip install --target D` で**毎回まっさらな D に**入れて、D の所要とサイズを
比べる。`--target` はシステムの site-packages を汚さないので 4 通りを同じ
条件で繰り返せる（本番の実装は purge + システムへ install なので絶対値は
ずれるが、**並び順と差**はそのまま効く）。

最後に baseline をもう一度測る。2 回目が速ければ「wheel ファイルがページ
キャッシュに載っただけ」で、途中の比較は成立しない。その場合は結果に
`cache_warm_bias` を立てて**過信しない**ようにする。

出力: /kaggle/working/result.json（strategies / baseline_again / env）
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
TMP = "/kaggle/tmp"
VLLM_SPEC = os.environ.get("KC_VLLM_SPEC", "vllm==0.32.0")

DATA = {"stage": "init", "ok": False, "vllm_spec": VLLM_SPEC,
        "strategies": [], "gpu_unknown": [
            "この 4 通りが実際の起動時間に効くかは GPU 側で測り直す必要がある",
        ]}


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=3600):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p, round(time.time() - t0, 1)


def du(path):
    """展開後のサイズ（MiB）。数万ファイルなので `du -sm` に任せる。"""
    p, _ = sh(["du", "-sm", path], timeout=600)
    try:
        return int((p.stdout or "0").split()[0])
    except (ValueError, IndexError):
        return None


def count_files(path):
    p, _ = sh(["bash", "-lc", f"find {path} -type f | wc -l"], timeout=600)
    try:
        return int((p.stdout or "0").strip())
    except ValueError:
        return None


def wheel_dirs():
    """wheelhouse Dataset の wheel 置き場を探す（zip 配布なら 1 回だけ展開）。"""
    out = []
    for slug in ("vllm-wheelhouse", "vllm-wheelhouse-patch"):
        roots = []
        for pat in (f"/kaggle/input/{slug}", f"/kaggle/input/*/{slug}",
                    f"/kaggle/input/*/*/{slug}"):
            roots += [p for p in glob.glob(pat) if os.path.isdir(p)]
        if not roots:
            print(f"ℹ {slug} なし")
            continue
        root = roots[0]
        cand = root if glob.glob(os.path.join(root, "*.whl")) else None
        if cand is None:
            for sub in sorted(glob.glob(os.path.join(root, "*"))):
                if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.whl")):
                    cand = sub
                    break
        if cand is None:
            zips = (glob.glob(os.path.join(root, "*.zip"))
                    + glob.glob(os.path.join(root, "*", "*.zip")))
            if zips:
                import zipfile
                dst = os.path.join(TMP, "whl", slug)
                os.makedirs(dst, exist_ok=True)
                t0 = time.time()
                for z in zips:
                    with zipfile.ZipFile(z) as f:
                        f.extractall(dst)
                print(f"📦 {slug} の zip を展開 {time.time() - t0:.1f}s")
                cand = dst if glob.glob(os.path.join(dst, "*.whl")) else None
        if cand is None:
            print(f"⚠ {slug} に wheel が無い")
            continue
        n = len(glob.glob(os.path.join(cand, "*.whl")))
        print(f"📦 {slug}: {n} wheel / {cand}")
        out.append(cand)
    return out


def find_links(dirs):
    args = ["--no-index"]
    for d in dirs:
        args += ["--find-links", d]
    return args


def measure(name, cmd, tag=""):
    """1 通りをまっさらな D に入れて、所要・サイズ・ファイル数を返す。"""
    dst = os.path.join(TMP, f"tgt-{name}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    print(f"\n--- {name} ---")
    print("   " + " ".join(cmd[:6]) + (" …" if len(cmd) > 6 else ""))
    p, sec = sh(cmd + ["--target", dst])
    row = {"name": name, "tag": tag, "seconds": sec, "rc": p.returncode,
           "mib": du(dst), "files": count_files(dst),
           "tail": ((p.stdout or "") + (p.stderr or ""))[-800:]}
    if p.returncode != 0:
        print(f"   ❌ rc={p.returncode} {sec}s")
        print("   " + row["tail"].replace("\n", "\n   ")[-600:])
    else:
        print(f"   ✅ {sec}s / {row['mib']} MiB / {row['files']} files")
    shutil.rmtree(dst, ignore_errors=True)      # 3.25 GB × 5 は置けない
    DATA["strategies"].append(row)
    save()
    return row


def main():
    DATA["stage"] = "env"
    save()
    DATA["env"] = {
        "python": sys.version.split()[0],
        "pip": (sh([sys.executable, "-m", "pip", "--version"])[0].stdout or "").strip(),
        "df_tmp": (sh(["df", "-h", TMP])[0].stdout or "").strip(),
        "disk_free_mib": sh(["bash", "-lc", f"df -m {TMP} | tail -1 | awk '{{print $4}}'"])[0].stdout.strip(),
    }
    print(DATA["env"])

    dirs = wheel_dirs()
    if not dirs:
        raise RuntimeError("wheelhouse が無い（Dataset 未マウント）。これでは測れない")
    links = find_links(dirs)
    save()

    # 1) 本番と同じ: 解決 + 展開 + バイトコード生成
    measure("pip", [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
            + links + [VLLM_SPEC], tag="baseline（本番と同じ）")

    # 2) バイトコード生成を止める。初回 import で遅延コンパイルに移るだけで、
    #    総量は減らない可能性がある（その場合 import 側が伸びる）。
    measure("pip-nocompile",
            [sys.executable, "-m", "pip", "install", "-q", "--no-input",
             "--no-compile"] + links + [VLLM_SPEC],
            tag="--no-compile（pyc を書かない）")

    # 3) uv。展開が並列・高速で、pip の Python 実装より速いことが多い。
    DATA["stage"] = "uv-install"
    save()
    p, sec = sh([sys.executable, "-m", "pip", "install", "-q", "--no-input", "uv"])
    uv = shutil.which("uv")
    DATA["uv"] = {"rc": p.returncode, "seconds": sec, "path": uv,
                  "version": (sh([uv, "--version"])[0].stdout or "").strip() if uv else None}
    print(DATA["uv"])
    save()
    if uv:
        measure("uv",
                [uv, "pip", "install", "--no-index", "--python", sys.executable]
                + [a for a in links if a != "--no-index"] + [VLLM_SPEC],
                tag="uv pip（並列展開）")
    else:
        DATA["strategies"].append({"name": "uv", "skipped": "uv を入れられない"})

    # 4) 解決を飛ばす。`--dry-run --report` で wheel の実パスを 1 回だけ出し、
    #    2 回目は `--no-deps` + パス直指定。190 個の依存グラフを解かない分が浮く。
    DATA["stage"] = "resolve-report"
    save()
    rep = os.path.join(TMP, "report.json")
    p, _ = sh([sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
               "--report", rep, "--no-input"] + links + [VLLM_SPEC])
    paths = []
    if p.returncode == 0 and os.path.isfile(rep):
        try:
            for item in json.load(open(rep))["install"]:
                url = item.get("download_info", {}).get("url", "")
                if url.startswith("file://"):
                    paths.append(url[7:])
        except (KeyError, ValueError, OSError):
            paths = []
    DATA["resolved"] = {"rc": p.returncode, "n": len(paths),
                        "tail": ((p.stdout or "") + (p.stderr or ""))[-500:]}
    print(f"   resolved {len(paths)} wheel")
    save()
    if paths:
        measure("pip-paths-nodeps",
                [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                 "--no-deps", "--no-compile"] + paths,
                tag="解決を飛ばす（--no-deps + パス直指定）")
    else:
        DATA["strategies"].append({"name": "pip-paths-nodeps",
                                   "skipped": "report から wheel パスを取れない"})

    # 5) baseline をもう一度。ここが速ければ以降の比較は「キャッシュ暖まり」を
    #    測っているだけなので、結論を弱める（静かに間違えないための対照）。
    again = measure("pip-again",
                    [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                    + links + [VLLM_SPEC], tag="対照（キャッシュ暖まりの確認）")
    first = next((s for s in DATA["strategies"] if s["name"] == "pip"), None)
    DATA["baseline_again"] = again
    if first and first.get("seconds") and again.get("seconds"):
        ratio = again["seconds"] / first["seconds"]
        DATA["cache_warm_bias"] = round(ratio, 2)
        if ratio < 0.8:
            print(f"⚠ 2 回目の baseline が {ratio:.2f} 倍。wheel がキャッシュに載って"
                  f"いる。途中の比較は**差の向きだけ**を読むこと")

    DATA["stage"] = "done"
    DATA["ok"] = all(s.get("rc") == 0 for s in DATA["strategies"] if "rc" in s)
    save()
    print("\n=== 結果 ===")
    for s in DATA["strategies"]:
        if "skipped" in s:
            print(f"  {s['name']:18s} skip: {s['skipped']}")
        else:
            print(f"  {s['name']:18s} {s['seconds']:7.1f}s  {s['mib']} MiB  "
                  f"{s['files']} files  rc={s['rc']}  {s['tag']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        DATA["stage"] = "failed"
        DATA["error"] = traceback.format_exc()[-2000:]
        save()
        print(DATA["error"])
        sys.exit(1)

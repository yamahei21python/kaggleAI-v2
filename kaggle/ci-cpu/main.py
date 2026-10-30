#!/usr/bin/env python3
"""CPU 検証: 公開リポジトリを clone し、CI と同じ検査を回す。

Kaggle の Python 3.13 イメージで実行するのが目的（ローカル 3.9 /
GitHub Actions 3.10 との差を埋める）。GPU 不要・無料。失敗時は exit 1。
"""
import json
import pathlib
import subprocess
import sys
import time
import traceback

OWNER = "yamahei21python"
REPO = "kaggleAI-v2"
REF = "refactor/p0-and-kbuild"
URL = f"https://github.com/{OWNER}/{REPO}.git"
WORK = pathlib.Path("/kaggle/working")
REPO_DIR = WORK / "repo"
RESULT = WORK / "result.json"

# pytest 移行前のブランチ向け（存在するものだけ実行する）
LEGACY_TESTS = [
    ("kbuild", "kbuild/_selftest.py"),
    ("proxy", "kaggle/step2/_selftest_proxy.py"),
    ("builder", "kaggle/step2/_selftest_builder.py"),
    ("exl3-server", "kaggle/exl3-serve/_selftest_server.py"),
    ("exl3-build", "kaggle/exl3-serve/_selftest_build.py"),
]


def run(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, time.time() - t0, p.stdout, p.stderr


def main():
    data = {"ok": False, "ref": REF, "steps": []}
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REF,
                        URL, str(REPO_DIR)], check=True, capture_output=True,
                       text=True, timeout=600)
        data["commit"] = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            text=True).strip()
        data["python"] = sys.version.split()[0]
        print(f"repo: {data['commit']} ({REF}) python={data['python']}")

        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "ruff", "pytest"],
                       check=True, timeout=600)

        # 移行完了前のブランチは tests/ が無いので旧セルフテストで確認する
        steps = [("ruff", [sys.executable, "-m", "ruff", "check", "."])]
        if (REPO_DIR / "tests").is_dir():
            data["mode"] = "pytest"
            steps.append(("pytest",
                          [sys.executable, "-m", "pytest", "tests", "-q"]))
        else:
            data["mode"] = "legacy"
            for name, rel in LEGACY_TESTS:
                if (REPO_DIR / rel).is_file():
                    steps.append((name, [sys.executable, rel]))
            steps.append(("notebooks",
                          [sys.executable, "kaggle/check_notebooks.py"]))
        steps += [
            ("exl3-generate",
             [sys.executable, "kaggle/exl3-serve/build_kernel.py"]),
            ("compileall", [sys.executable, "-m", "compileall", "-q",
                            "kaggle/exl3-serve/kernel",
                            "kaggle/exl3-serve/dryrun",
                            "kaggle/exl3-serve/keepalive"]),
        ]

        for name, cmd in steps:
            rc, sec, out, err = run(cmd)
            ok = rc == 0
            data["steps"].append({"name": name, "ok": ok, "sec": round(sec, 1),
                                  "rc": rc})
            print(f"{'OK' if ok else 'NG'} {name} ({sec:.1f}s)")
            if not ok:
                tail = ((out or "") + "\n" + (err or ""))[-4000:]
                data["steps"][-1]["tail"] = tail
                print(tail)
                break
        data["ok"] = (len(data["steps"]) == len(steps)
                      and all(s["ok"] for s in data["steps"]))
    except Exception as e:
        data["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        RESULT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print("RESULT: " + json.dumps(data, ensure_ascii=False))
    return 0 if data["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

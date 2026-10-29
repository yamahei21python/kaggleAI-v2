#!/usr/bin/env python3
"""コミット済みのノートブックがビルダーの出力と一致するか確認する。

ビルダーはリポジトリ内に直接書き込むので、作業ツリーを汚さないよう
一時ディレクトリにコピーしてから再生成し、元のファイルと比べる。
auth_proxy.py / llama-chat.html を直したのに ipynb を再生成し忘れると、
Kaggle には古いコードが push される（実際に ngrok 版でずれていた）。

    python3 kaggle/check_notebooks.py        # 不一致があれば exit 1
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (ビルダー, 追加の環境変数, 生成物)
TARGETS = [
    ("build_step1.py", {}, "step1/step1-orcarouter-qwen38-27b-dl.ipynb"),
    ("build_step2.py", {"STEP2_TUNNEL": "cloudflared"},
     "step2/step2-orcarouter-qwen38-27b-cloudflared.ipynb"),
    ("build_step2.py", {"STEP2_TUNNEL": "ngrok"},
     "step2-ngrok/step2-orcarouter-qwen38-27b-ngrok.ipynb"),
    ("build_smoke.py", {}, "smoke/smoke-orcarouter-qwen38-27b.ipynb"),
    ("build_bench.py", {}, "bench/bench-ctx-orcarouter-qwen38-27b.ipynb"),
    ("build_bench_mtp.py", {}, "bench_mtp/bench-mtp-orcarouter-qwen38-27b.ipynb"),
    ("build_keepalive_test.py", {}, "step2/step2-keepalive-test.ipynb"),
]


def main():
    failed = []
    with tempfile.TemporaryDirectory(prefix="kc-nbcheck-") as tmp:
        work = pathlib.Path(tmp)
        # config.local.env も写す（個人設定で生成したノートを既定値で比べない）
        for name in ("llama-chat.html", "kcconfig.py", "config.env",
                     "config.local.env"):
            if (ROOT / name).is_file():
                shutil.copy2(ROOT / name, work / name)
        shutil.copytree(ROOT / "kaggle", work / "kaggle",
                        ignore=shutil.ignore_patterns("archive", "__pycache__"))
        for builder, extra, out in TARGETS:
            env = {k: v for k, v in os.environ.items() if k != "STEP2_TUNNEL"}
            env.update(extra)
            r = subprocess.run([sys.executable, str(work / "kaggle" / builder)],
                               cwd=work, env=env, capture_output=True, text=True)
            label = builder + "".join(f" {k}={v}" for k, v in extra.items())
            if r.returncode != 0:
                failed.append(label)
                print(f"❌ {label}: ビルダーが失敗\n{r.stderr[-800:]}")
                continue
            same = (work / "kaggle" / out).read_bytes() == (ROOT / "kaggle" / out).read_bytes()
            print(f"{'✅' if same else '❌'} {out}" + ("" if same else f"  ← {label} で再生成してください"))
            if not same:
                failed.append(label)
    if failed:
        print(f"\n不一致 {len(failed)} 件")
        return 1
    print("\n全ノートブックがビルダー出力と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

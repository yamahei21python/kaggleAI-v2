#!/usr/bin/env python3
"""コミット済みのノートブックがビルダーの出力と一致するか確認する。

ビルダーはリポジトリ内に直接書き込むので、作業ツリーを汚さないよう
一時ディレクトリにコピーしてから再生成し、元のファイルと比べる
（写す・走らせる・比べるは `kaggle/_regen.py` が持つ）。
auth_proxy.py / llama-chat.html を直したのに ipynb を再生成し忘れると、
Kaggle には古いコードが push される（実際に ngrok 版でずれていた）。

    python3 kaggle/check_notebooks.py        # 不一致があれば exit 1
"""
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import _regen  # noqa: E402

# (ビルダー, 追加の環境変数, 生成物) — 生成物はリポジトリ相対。
# notebook 版は KC_SMOKE のような「台帳に載らないノブ」を使うので、
# script 版と違って台帳から自動導出せず、ここで明示する。
TARGETS = [
    ("build_step1.py", {}, "kaggle/step1/step1-orcarouter-qwen38-27b-dl.ipynb"),
    ("build_step2.py", {"STEP2_TUNNEL": "cloudflared"},
     "kaggle/step2/step2-orcarouter-qwen38-27b-cloudflared.ipynb"),
    ("build_step2.py", {"STEP2_TUNNEL": "ngrok"},
     "kaggle/step2-ngrok/step2-orcarouter-qwen38-27b-ngrok.ipynb"),
    ("build_smoke.py", {}, "kaggle/smoke/smoke-orcarouter-qwen38-27b.ipynb"),
    ("build_bench.py", {}, "kaggle/bench/bench-ctx-orcarouter-qwen38-27b.ipynb"),
    ("build_bench_mtp.py", {},
     "kaggle/bench_mtp/bench-mtp-orcarouter-qwen38-27b.ipynb"),
    ("build_keepalive_test.py", {},
     "kaggle/step2/step2-keepalive-test.ipynb"),
    ("build_llama_server.py", {"LLAMA_TUNNEL": "ngrok"},
     "kaggle/llama-server-ngrok/llama-server-ngrok.ipynb"),
    ("build_vllm_dataset.py", {},
     "kaggle/step2-vllm/prepare-vllm-dataset.ipynb"),
    ("build_vllm_dryrun.py", {},
     "kaggle/step2-vllm-dryrun/step2-vllm-dryrun.ipynb"),
    ("build_step2_vllm.py", {"STEP2_TUNNEL": "cloudflared"},
     "kaggle/step2-vllm-kernel/cloudflared/"
     "step2-vllm-qwen38-27b-cloudflared.ipynb"),
    ("build_step2_vllm.py", {"STEP2_TUNNEL": "ngrok"},
     "kaggle/step2-vllm-kernel/ngrok/step2-vllm-qwen38-27b-ngrok.ipynb"),
    # smoke 版は通常版と別パス・別セル数（15 セル）。同じ TARGETS で見ないと
    # 「通常版を smoke で上書き」も「smoke の取りこぼし」も検知できない。
    ("build_step2_vllm.py", {"STEP2_TUNNEL": "cloudflared", "KC_SMOKE": "1"},
     "kaggle/step2-vllm-kernel/smoke-cloudflared/"
     "step2-vllm-qwen38-27b-cloudflared.ipynb"),
]


def main():
    # 出力先が重複していると、後から走ったビルダーが前の生成物を上書きして
    # 「一致」に見えてしまう。TARGETS の設定ミスを先に潰す。
    outs = [out for _, _, out in TARGETS]
    dup = sorted({o for o in outs if outs.count(o) > 1})
    if dup:
        print("❌ TARGETS の出力先が重複: " + ", ".join(dup))
        return 1
    failed = []
    with tempfile.TemporaryDirectory(prefix="kc-nbcheck-") as tmp:
        # コミットするノートは config.env（既定値）で生成したものが基準。
        # config.local.env の個人値はノートに焼き込まず、Kaggle 側は Secrets
        # （NGROK_DOMAIN / GIST_ID）で上書きする前提なので、ここでは写さない。
        if (ROOT / "config.local.env").is_file():
            print("ℹ config.local.env は無視して既定値で比較します"
                  "（個人値で再生成したノートは不一致になります）\n")
        work = _regen.stage(tmp)
        for builder, extra, out in TARGETS:
            tag = _regen.label(builder, extra)
            r = _regen.run(work, builder, extra)
            if r.returncode != 0:
                failed.append(tag)
                print(f"❌ {tag}: ビルダーが失敗\n{r.stderr[-800:]}")
                continue
            state, detail = _regen.diff(work, out)
            if _regen.report(out, state, detail, tag):
                failed.append(tag)
    if failed:
        print(f"\n不一致 {len(failed)} 件")
        return 1
    print("\n全ノートブックがビルダー出力と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SGLang が「入るのか / T4 で動くのか」を GPU 0 秒で切り分ける。

背景（実測 2026-11-01）
----------------------
`step2-sglang-qwen38-27b-cloudflared` は **297 秒で `pip install 失敗 (1)`**。
`install_tail` に残っていたのはこれだけ:

    × Building wheel for outlines_core (pyproject.toml) did not run successfully.
    ╰─> No available output.

つまり **エンジンは一度も起動していない**。「SGLang は T4（sm75）で動かない」は
まだ何も確かめていない推測でしかない。まず install を通す。

原因（PyPI メタデータで特定・GPU 0 秒）
---------------------------------------
    sglang==0.5.22 → outlines==0.1.11 → outlines_core==0.1.26（完全固定）

`outlines_core` は Rust 実装（maturin）。**0.1.26 には cp313 の wheel が無い**
（0.1.x で cp313 があるのは 0.1.17 だけ。0.2.9 以降は cp313 あり）。
Kaggle の Python は 3.13.15 なので sdist からのビルドになり、Rust が無いので落ちる。

**直接依存 26 本を全部調べた結果、cp313 wheel が無いのは 2 本だけ**:
  - `outlines_core==0.1.26`（間接・Rust）
  - `cuda-tile==1.6.0rc5`（sdist のみ。ただし実測でビルドは通っている＝失敗一覧に
    出ていない。rc7 以降には cp313 wheel がある）

なので「Rust を入れれば通る」可能性が高い。**通らなかったときのために、
固定を回避する第 2 の筋も同じカーネルで試す**（GPU で 1 往復溶かさないため）。

試す筋
------
- 筋 A（本命）: rustup を入れて `pip install sglang==0.5.22` をそのまま通す。
  `outlines==0.1.11` の固定を**壊さない**ので、動作の再現性が一番高い。
- 筋 B（代替）: Rust が入らない/ビルドが通らないとき。
  `pip install --no-deps sglang==0.5.22` + `outlines-core==0.1.17`（cp313 wheel あり）
  + `outlines==0.1.11 --no-deps` + 残りの固定を手で入れる。
  0.1.17 と 0.1.26 で `outlines` が使う API が同じとは限らないので**賭け**。

どちらが通っても、そのあと `import sglang` と `launch_server --help` まで確認する
（ここまで通れば、残る不確定要素は sm75 のカーネルだけになる）。
"""

import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
DATA = {}

SGLANG_VERSION = os.environ.get("KC_SGLANG_VERSION", "0.5.22")
# outlines 0.1.11 が固定する 0.1.26 の代わりに使う版（cp313 wheel がある唯一の 0.1.x）
OUTLINES_CORE_FALLBACK = os.environ.get("KC_OUTLINES_CORE", "0.1.17")
# Rust の導入に使う上限。これを超えたら筋 A を諦めて筋 B へ。
RUST_BUDGET = int(os.environ.get("KC_RUST_BUDGET", "600"))
# pip install 本体の上限（実測: vLLM で 261〜1208 秒。SGLang は依存が多い）
INSTALL_BUDGET = int(os.environ.get("KC_INSTALL_BUDGET", "2700"))


def save():
    DATA["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=120, env=None):
    """戻り値は (rc, 出力)。**出力は必ず残す**（黙って失敗させない）。"""
    e = dict(os.environ)
    e.setdefault("PATH", "")
    e["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + e["PATH"]
    if env:
        e.update(env)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=e)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT ({timeout}s)", time.time() - t0
    except Exception as exc:                      # noqa: BLE001
        return -2, f"{type(exc).__name__}: {exc}", time.time() - t0


def tail(s, n=25):
    lines = [ln for ln in (s or "").strip().splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def pip(*args, timeout=1800):
    return sh([sys.executable, "-m", "pip", *args], timeout=timeout)


def which_all():
    """ビルドに要る道具が揃っているか。**足りない物を名前で出す**。"""
    out = {}
    for name in ("gcc", "g++", "cargo", "rustc", "curl", "cmake", "ninja"):
        out[name] = shutil.which(name)
    return out


def df(path):
    rc, out, _ = sh(["df", "-h", path])
    return out.strip().splitlines()[-1] if out else None


def probe_wheel(pkg, ver):
    """その版に cp313 / linux / x86_64 で入る wheel があるかを pip 自身に判定させる。

    `--only-binary=:all:` を付けると「wheel が無ければ失敗」になるので、
    推測ではなく pip の解決結果で分かる。**ダウンロードはしない**（--no-deps）。
    """
    rc, out, sec = pip("download", "--no-deps", "--only-binary=:all:",
                       f"{pkg}=={ver}", "-d", "/tmp/wheelprobe",
                       timeout=300)
    return {"pkg": pkg, "version": ver, "rc": rc,
            "wheel_available": rc == 0,
            "seconds": round(sec, 1),
            "tail": tail(out, 8)}


def install_rust():
    """rustup を入れる（minimal / stable）。入らなければ諦めて筋 B へ。"""
    rec = {"ok": False}
    if shutil.which("cargo") and shutil.which("rustc"):
        rc, out, _ = sh(["rustc", "--version"])
        rec.update(ok=True, skipped="already installed", version=out.strip())
        return rec
    if not shutil.which("curl"):
        rec["error"] = "curl が無い"
        return rec
    # -y で非対話。--profile minimal は docs を入れない（速い）。
    rc, out, sec = sh(["bash", "-lc",
                       "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                       "| sh -s -- -y --profile minimal --default-toolchain stable"],
                      timeout=RUST_BUDGET)
    rec.update(seconds=round(sec, 1), rc=rc, tail=tail(out, 12))
    if rc != 0:
        rec["error"] = "rustup 失敗"
        return rec
    rc2, out2, _ = sh(["bash", "-lc", "source ~/.cargo/env && rustc --version"])
    rec["version"] = out2.strip()
    rec["ok"] = rc2 == 0 and "rustc" in out2
    if not rec["ok"]:
        rec["error"] = "rustc が入ったが実行できない"
    return rec


def check_import():
    """`import sglang` と `launch_server --help`。GPU 無しでも import までは通る。

    `--help` は CUDA を初期化しようとして失敗することがある（GPU の無い
    カーネルなので当然）。**失敗しても「install は通った」判定は変えない**ため、
    ここは記録だけして先へ進む。
    """
    out = {}
    rc, txt, sec = sh([sys.executable, "-c",
                       "import sglang, sys; print('sglang', sglang.__version__); "
                       "print('python', sys.version.split()[0])"], timeout=600)
    out["import"] = {"rc": rc, "seconds": round(sec, 1), "out": tail(txt, 12),
                     "ok": rc == 0}
    rc, txt, sec = sh([sys.executable, "-c",
                       "import torch; print('torch', torch.__version__, "
                       "'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"],
                      timeout=600)
    out["torch"] = {"rc": rc, "out": tail(txt, 6)}
    rc, txt, sec = sh([sys.executable, "-m", "sglang.launch_server", "--help"],
                      timeout=600)
    out["launch_help"] = {"rc": rc, "seconds": round(sec, 1),
                          "tail": tail(txt, 15), "ok": rc == 0}
    return out


def route_a():
    """本命: Rust を入れて素の `pip install sglang==VERSION` を通す。"""
    rec = {"route": "A (rust + plain install)"}
    rec["rust"] = install_rust()
    if not rec["rust"].get("ok"):
        rec["skipped"] = "rust が入らない"
        return rec
    rc, out, sec = pip("install", f"sglang=={SGLANG_VERSION}", timeout=INSTALL_BUDGET)
    rec.update(rc=rc, seconds=round(sec, 1), ok=rc == 0, tail=tail(out, 30))
    return rec


def route_b():
    """代替: 固定を回避する（outlines_core だけ cp313 wheel のある版へ）。

    `outlines==0.1.11` は `outlines_core==0.1.26` を**完全固定**しているので、
    `-c`（constraints）では上書きできない。**`--no-deps` で依存解決を飛ばし、
    必要な物を自分で並べる**しかない。順序も意味がある:
    先に `outlines-core` を入れておかないと、`outlines` を入れた瞬間に
    0.1.26 を取りに行って同じところで死ぬ。
    """
    rec = {"route": "B (no-deps + outlines-core pin)"}
    steps = []

    # 1) sglang 本体だけ（依存は入れない）
    rc, out, sec = pip("install", "--no-deps", f"sglang=={SGLANG_VERSION}",
                       timeout=900)
    steps.append({"step": "sglang --no-deps", "rc": rc, "seconds": round(sec, 1),
                  "tail": tail(out, 10)})
    if rc != 0:
        rec.update(ok=False, steps=steps, error="sglang 本体が入らない")
        return rec

    # 2) outlines_core を cp313 wheel のある版で先に固定
    rc, out, sec = pip("install", f"outlines-core=={OUTLINES_CORE_FALLBACK}",
                       timeout=600)
    steps.append({"step": f"outlines-core=={OUTLINES_CORE_FALLBACK}", "rc": rc,
                  "seconds": round(sec, 1), "tail": tail(out, 8)})

    # 3) outlines 本体は --no-deps（0.1.26 を引き込ませない）
    rc, out, sec = pip("install", "--no-deps", "outlines==0.1.11", timeout=600)
    steps.append({"step": "outlines==0.1.11 --no-deps", "rc": rc,
                  "seconds": round(sec, 1), "tail": tail(out, 8)})

    # 4) 残りの依存をまとめて（outlines 系は既に入っているので触らせない）
    rest = [
        "torch==2.14.1", "torchaudio==2.11.0", "torchvision",
        "transformers==5.19.0", "tokenizers==0.23.2", "xgrammar==0.2.7",
        "flashinfer_python[cu13]==0.7.1", "sglang-kernel==0.4.9",
        "compressed-tensors==0.18.0", "cuda-python>=13.0",
        "apache-tvm-ffi==0.1.11", "tilelang==0.1.12", "numba==0.65.1",
        "nvidia-cutlass-dsl[cu13]==4.8.0", "nvidia-mathdx==25.6.0",
        "humming-kernels[cu13]==0.1.12", "quack-kernels==0.6.5",
        "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2",
        "tokenspeed_mla==0.1.8", "smg-grpc-servicer>=0.9.0",
        "openai-harmony==0.0.4", "openai==2.6.1", "sentencepiece==0.2.1",
        "timm==1.0.16", "soundfile==0.13.1", "blobfile==3.0.0",
        "fastapi", "uvicorn", "uvloop", "watchfiles", "msgspec", "orjson",
        "pydantic", "psutil", "pyzmq>=25.1.2", "pybase64", "einops",
        "easydict", "interegular", "partial_json_parser", "regex", "scipy",
        "setproctitle", "tiktoken", "tqdm", "xxhash", "zstandard",
        "prometheus-client>=0.20.0", "python-multipart", "aiohttp", "requests",
        "distro", "packaging", "pillow", "numpy", "gguf", "py-spy", "ninja",
        "build", "IPython", "datasets", "modelscope", "mistral_common>=1.11.5",
        "llguidance>=1.7.6,<2.0.0", "kernels>=0.14.1,<0.15",
        "anthropic>=0.20.0", "nvshmem4py-cu13", "torchcodec==0.17.0",
        "torch_memory_saver>=0.0.9.post1", "cuda-tile==1.6.0rc5",
    ]
    rc, out, sec = pip("install", *rest, timeout=INSTALL_BUDGET)
    steps.append({"step": f"rest ({len(rest)} 本)", "rc": rc,
                  "seconds": round(sec, 1), "tail": tail(out, 20)})
    rec.update(steps=steps, ok=rc == 0, rc=rc, seconds=round(sec, 1))
    if rc != 0:
        rec["error"] = "残りの依存で失敗"
    return rec


def main():
    DATA["stage"] = "env"
    DATA["python"] = sys.version.split()[0]
    DATA["sglang_version"] = SGLANG_VERSION
    save()

    print("=" * 64)
    print(f"  SGLang install probe（GPU 0 秒）: sglang=={SGLANG_VERSION} は入るか")
    print("=" * 64)

    rc, out, _ = pip("--version")
    DATA["pip"] = out.strip()
    DATA["tools"] = which_all()
    DATA["df"] = {p: df(p) for p in ("/kaggle/working", "/tmp", "/")}
    print(f"  🐍 python {DATA['python']} / {DATA['pip']}")
    print("  🔧 tools: " + ", ".join(
        f"{k}={'OK' if v else '—'}" for k, v in DATA["tools"].items()))
    for p, v in DATA["df"].items():
        print(f"  💾 {p}: {v}")
    save()

    # ---- 位相 1: ブロッカーを pip 自身に判定させる ----
    # 「wheel が無い」を推測で書くと、次に読む人がまた同じところで転ぶ。
    DATA["stage"] = "wheel_probe"
    save()
    print("\n[1] cp313 の wheel があるか（pip に判定させる・ダウンロード無し）")
    DATA["wheel_probe"] = [probe_wheel("outlines-core", "0.1.26"),
                           probe_wheel("outlines-core", OUTLINES_CORE_FALLBACK),
                           probe_wheel("cuda-tile", "1.6.0rc5")]
    for r in DATA["wheel_probe"]:
        mark = "✅ ある" if r["wheel_available"] else "❌ 無い（sdist ビルドになる）"
        print(f"    {r['pkg']}=={r['version']}: {mark}  ({r['seconds']}s)")
    save()

    # ---- 位相 2: 筋 A ----
    DATA["stage"] = "route_a"
    save()
    print(f"\n[2] 筋 A: Rust を入れて素の pip install（上限 {INSTALL_BUDGET}s）")
    DATA["route_a"] = route_a()
    ra = DATA["route_a"]
    if ra.get("rust", {}).get("ok"):
        print(f"    🦀 rustc: {ra['rust'].get('version')}"
              f"（{ra['rust'].get('seconds')}s）")
    else:
        print(f"    🦀 rust 導入に失敗: {ra.get('rust', {}).get('error')}")
    if "skipped" in ra:
        print(f"    ⏭ 筋 A をスキップ（{ra['skipped']}）")
    else:
        print(f"    pip install: {'✅ 成功' if ra.get('ok') else '❌ 失敗'}"
              f"  {ra.get('seconds')}s")
        if not ra.get("ok"):
            print("    --- tail ---")
            for ln in (ra.get("tail") or "").splitlines()[-12:]:
                print(f"    {ln}")
    save()

    # ---- 位相 3: 筋 B（筋 A が失敗したときだけ）----
    if ra.get("ok"):
        DATA["route_b"] = {"skipped": "筋 A が成功したため不要"}
        print("\n[3] 筋 B: スキップ（筋 A が通った）")
    else:
        DATA["stage"] = "route_b"
        save()
        print("\n[3] 筋 B: 固定を回避（outlines_core だけ差し替え）")
        DATA["route_b"] = route_b()
        rb = DATA["route_b"]
        for s in rb.get("steps") or []:
            print(f"    {s['step']}: {'✅' if s['rc'] == 0 else '❌'} "
                  f"{s['seconds']}s")
        if not rb.get("ok"):
            print(f"    失敗: {rb.get('error')}")
            for s in rb.get("steps") or []:
                if s["rc"] != 0:
                    print(f"    --- {s['step']} tail ---")
                    for ln in (s["tail"] or "").splitlines()[-12:]:
                        print(f"    {ln}")
    save()

    # ---- 位相 4: 動く形になっているか ----
    DATA["stage"] = "import"
    save()
    print("\n[4] import と launch_server --help（GPU 無しなので CUDA は失敗してよい）")
    DATA["check"] = check_import()
    print(f"    import sglang: "
          f"{'✅' if DATA['check']['import']['ok'] else '❌'} "
          f"{DATA['check']['import']['out'].splitlines()[:1]}")
    print(f"    torch: {DATA['check']['torch']['out'].strip()[:100]}")
    print(f"    launch_server --help: "
          f"{'✅' if DATA['check']['launch_help']['ok'] else '❌'}")
    save()

    # ---- 判定 ----
    installed = (ra.get("ok") or (DATA.get("route_b") or {}).get("ok")) \
        and DATA["check"]["import"]["ok"]
    DATA["verdict"] = {
        "installed": bool(installed),
        "route": "A" if ra.get("ok") else ("B" if (DATA.get("route_b") or {}).get("ok") else None),
        "rust_needed": not (DATA.get("wheel_probe") or [{}])[0].get("wheel_available", True),
        "outlines_core_pin_override": OUTLINES_CORE_FALLBACK,
        "sm75_untested": True,
        "note": "install が通っても T4(sm75) で推論できるかは別。GPU で 1 回試す必要がある。",
    }
    print("\n" + "=" * 64)
    if installed:
        print(f"  ✅ install は通った（筋 {DATA['verdict']['route']}）"
              f" → 残る不確定要素は sm75 のカーネルだけ")
    else:
        print("  ❌ install が通らない → GPU で試す価値が無い（ここで止めてよい）")
    print("=" * 64)

    DATA["stage"] = "done"
    DATA["ok"] = bool(installed)
    save()
    return 0 if installed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        DATA["ok"] = False
        DATA["error"] = f"{type(e).__name__}: {e}"
        save()
        sys.exit(1)

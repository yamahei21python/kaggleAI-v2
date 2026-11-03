"""ノード速度レース: 2 本同時に push し、速いノードを引くまで繰り返す。

```
python3 -m cli race --yes              # 3 分〜（成功 1 本はそのまま本番）
python3 -m cli race --yes --gate 50    # 閾値を下げて早く当てる
```

## 何をするか

1. 生成物を 1 回だけビルド（`KC_PROBE_GATE_MIB_S` を焼く）
2. **id 違いの 2 本**（`vllm-race-a` / `vllm-race-b`）へ同じコードを push
3. 各カーネルは起動直後の 64 MiB probe でゲート判定 → 未満なら `SystemExit`
   （install の 267 秒を払わない。`00_settings.py` の `_kc_probe_gate`）
4. 両方の `probe.json` を回収して判定
5. 負け側は `kaggle kernels delete -y` で停止（実測 10〜20 秒で止まる）
6. 勝ち側の `url.txt`（cloudflared）を待って URL を出す

## 両方通ったとき

probe が速い方だけ残し、遅い方を即 delete する（案 2）。両方 install まで
走らせると、重みロードが長いノードでは**最大 1.7 時間**を無駄に焼く。
両方通る確率は 1 ラウンド 0.38%（150 MiB/s が 6.2% のとき）。

## 60 秒で install を抜けないとき

何もしない（正常）。install は実測 267 秒なので、60 秒で抜ける方が異常。
勝ち側が**異常終了した**（URL が出ないまま status が COMPLETE/ERROR）
ときだけ次のラウンドへ戻る。

## 前提

- `kaggle` CLI 2.x（バージョン指定 output / delete。1.x には無い）
- 生成物は `vllm-playground`（本番と同じセル・公開系だけ外した骨組み）
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent import futures

from cli import build as build_mod
from cli.push import _quota_lines, kaggle_exe
from kbuild import spec

# 2 本は**別 id** でなければ同時に走らない（同一 id の再 push は走行中の
# セッションを置き換える）。id は使い捨て。負けた側は delete で消える。
SLUGS = {"a": "vllm-race-a", "b": "vllm-race-b"}

_VERSION = re.compile(r"Kernel version (\d+)")
_STATUS = re.compile(r"KernelWorkerStatus\.([A-Z]+)")


# ── 純関数（テストはここだけ見る）────────────────────────────
def decide(a_mib, b_mib, gate):
    """2 本の probe 速度から次の一手を返す。

    返り値は `"retry"` / `"a"` / `"b"`。`None` は「probe が取れなかった」
    （ゲート未満と同じ扱い）。両方通ったら速い方、同値なら a。
    """
    a_ok = a_mib is not None and a_mib >= gate
    b_ok = b_mib is not None and b_mib >= gate
    if not a_ok and not b_ok:
        return "retry"
    if a_ok and not b_ok:
        return "a"
    if b_ok and not a_ok:
        return "b"
    return "a" if a_mib >= b_mib else "b"


def parse_version(stdout):
    """`Kernel version 12 successfully pushed.` → 12（読めなければ 0）。"""
    m = _VERSION.search(stdout or "")
    return int(m.group(1)) if m else 0


def parse_status(stdout):
    """`... has status "KernelWorkerStatus.RUNNING"` → `RUNNING`（無ければ ""）。"""
    m = _STATUS.search(stdout or "")
    return m.group(1) if m else ""


def parse_probe_mib(path):
    """`probe.json` → `mib_s`（失敗・欠損は None）。"""
    try:
        rec = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not rec.get("ok"):
        return None
    try:
        return float(rec["mib_s"])
    except (KeyError, TypeError, ValueError):
        return None


# ── Kaggle 操作 ────────────────────────────────────────────
def _owner(kernel):
    """台帳の id から owner を取る（`yamahei21python/xxx` の前半）。"""
    return kernel.id.split("/")[0]


def _run(cmd, timeout=180):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return 1, "", f"{type(e).__name__}: {e}"
    return p.returncode, p.stdout or "", p.stderr or ""


def _supports_output_flag(exe, flag):
    """`kernels output` が flag を受けるか。

    `cli.push._supports` は **push** の `--help` を見るので、`--file-pattern`
    のような output 側のフラグには使えない（実測で誤判定した）。
    """
    rc, out, err = _run([exe, "kernels", "output", "--help"], timeout=120)
    return rc == 0 and flag in (out + err)


def _parallel(fn_a, fn_b):
    """2 つの処理を**同時**に走らせて (a, b) を返す。

    順に走らせると「2 本目が始まる前に 1 本目が probe を終える」ので、
    レースの意味（同じ時間帯のノードを比べる）が薄れる。
    """
    with futures.ThreadPoolExecutor(max_workers=2) as ex:
        fa, fb = ex.submit(fn_a), ex.submit(fn_b)
        return fa.result(), fb.result()


def stage_meta(base, side, owner):
    """push 用メタデータ。`title` は id に**解決できる形**にする。

    Kaggle は title から slug を作るので、`title="race a"` だと
    `yamahei21python/race-a` になり、指定した id と衝突して 409
    （`Your kernel title does not resolve to the specified id`）。
    """
    slug = SLUGS[side]
    return {**base, "id": f"{owner}/{slug}", "title": slug}


def _stage(kernel, side, owner):
    """push 用の一時ディレクトリ（id と title だけ差し替えた複製）を返す。"""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"kc-race-{side}-"))
    src_dir = pathlib.Path(kernel.dir)
    shutil.copy2(src_dir / kernel.code_file, tmp / kernel.code_file)
    base = json.loads((src_dir / kernel.metadata_file).read_text(encoding="utf-8"))
    meta = stage_meta(base, side, owner)
    (tmp / kernel.metadata_file).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return tmp


def push_side(exe, kernel, side, owner):
    """1 本 push してバージョン番号を返す（失敗は 0）。"""
    tmp = None
    try:
        tmp = _stage(kernel, side, owner)
        rc, out, err = _run([exe, "kernels", "push", "-p", str(tmp)], timeout=600)
    except (OSError, ValueError) as e:
        print(f"  push {side} 準備失敗: {type(e).__name__}: {e}")
        return 0
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        print(f"  push {side} 失敗: {(err or out).strip()[:200]}")
        return 0
    ver = parse_version(out)
    print(f"  push {side}: v{ver}")
    return ver


def fetch_file(exe, owner, side, ver, pattern, dest, timeout=120):
    """バージョン指定で 1 ファイルだけ回収して Path（無ければ None）。

    `--file-pattern` は**ダウンロード**を絞るだけで、カーネルログは常に
    落ちてくる。glob の先頭を返すとログを掴むので、pattern で選び直す。
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ref = f"{owner}/{SLUGS[side]}/{ver}"
    rc, _, _ = _run([exe, "kernels", "output", ref, "-p", str(dest),
                     "--file-pattern", pattern, "-q", "-o"], timeout=timeout)
    if rc != 0:
        return None
    want = re.compile(pattern)
    hits = sorted(p for p in dest.glob("*")
                  if p.is_file() and want.search(p.name))
    return hits[0] if hits else None


def status_of(exe, owner, side):
    rc, out, _ = _run([exe, "kernels", "status", f"{owner}/{SLUGS[side]}"],
                      timeout=120)
    return parse_status(out) if rc == 0 else ""


def delete_side(exe, owner, side, attempts=2):
    """負け側を停止する（カーネルごと消える。実測 10〜20 秒で止まる）。

    失敗を握り潰すと**負け側が走ったまま**勝者の URL 待ちへ進む（GPU を
    二重に焼く）。1 回だけ再試行し、駄目なら明示する。
    """
    ref = f"{owner}/{SLUGS[side]}"
    for i in range(attempts):
        rc, _, err = _run([exe, "kernels", "delete", "-y", ref], timeout=180)
        if rc == 0:
            print(f"  delete {side}: ok")
            return True
        if i + 1 < attempts:
            time.sleep(5)
    print(f"  ⚠ delete {side} 失敗（走行中なら手で止める）: {(err or '').strip()[:120]}")
    return False


# ── 待ち合わせ ─────────────────────────────────────────────
def wait_probe(exe, owner, side, ver, tmp, limit_s=240, interval=8):
    """`probe.json` が取れるまで待つ（セッションが終わったら諦める）。"""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        p = fetch_file(exe, owner, side, ver, r"probe\.json", tmp)
        if p:
            mib = parse_probe_mib(p)
            if mib is not None:
                return mib
            return None                      # 取れたが ok=false
        if status_of(exe, owner, side) in ("COMPLETE", "ERROR", "CANCELED"):
            return None
        time.sleep(interval)
    return None


def wait_url(exe, owner, side, ver, tmp, limit_s, interval=20, tick=120):
    """`url.txt` が出るまで待つ。出なければ ""（異常終了・時間切れ）。"""
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < limit_s:
        p = fetch_file(exe, owner, side, ver, r"url\.txt", tmp)
        if p:
            url = p.read_text(encoding="utf-8").strip()
            if url.startswith("http"):
                return url
        st = status_of(exe, owner, side)
        if st in ("COMPLETE", "ERROR", "CANCELED"):
            return ""
        el = time.time() - t0
        if el - last >= tick:
            last = el
            print(f"  {side} 待機 {int(el)}s（status={st or '?'}）")
        time.sleep(interval)
    return ""


# ── 本体 ──────────────────────────────────────────────────
def run(target="vllm-playground", gate=150.0, rounds=25, yes=False,
        dry_run=False, wait_url_s=None, interval=8):
    kernel = spec.find(target)
    owner = _owner(kernel)
    print(f"対象: {kernel.id} → {SLUGS['a']} / {SLUGS['b']}（owner={owner}）")
    print(f"ゲート: {gate} MiB/s / 最大 {rounds} ラウンド")
    for line in _quota_lines():
        print(line)
    if not yes:
        print("\n2 本 × 最大 "
              f"{rounds} ラウンド分の GPU を消費する。進めるなら --yes。")
        return 1
    if dry_run:
        print(f"DRY build {kernel.name} + push {SLUGS['a']}/{SLUGS['b']} ×{rounds}")
        return 0
    exe = kaggle_exe()
    if not exe:
        print("kaggle CLI が見つからない（PATH を確認）")
        return 1
    if not _supports_output_flag(exe, "--file-pattern"):
        print("この kaggle CLI はバージョン指定 output に未対応（2.x が要る）")
        print("  KC_KAGGLE=/path/to/kaggle2 を指定する")
        return 1

    # ゲートを焼いてからビルド（push は再ビルドしない。2 本で同じ生成物を使う）
    os.environ["KC_PROBE_GATE_MIB_S"] = str(gate)
    if build_mod.run([kernel.name]) != 0:
        print("生成に失敗したため中止")
        return 1

    budget = wait_url_s
    if budget is None:
        budget = float(os.environ.get("VLLM_STARTUP_BUDGET", "11000")) + 900

    live = set()          # 走らせたままの side（終了時に必ず片付ける）
    try:
        for rnd in range(1, rounds + 1):
            print(f"\n=== ラウンド {rnd}/{rounds}")
            # 2 本は**同時**に push する。順に押すと、2 本目が走り出す前に
            # 1 本目が probe を終えてしまい、レースにならない。
            vers = dict(zip(("a", "b"), _parallel(
                lambda: push_side(exe, kernel, "a", owner),
                lambda: push_side(exe, kernel, "b", owner))))
            live |= {s for s, v in vers.items() if v}
            if not all(vers.values()):
                for s in sorted(live):
                    delete_side(exe, owner, s)
                live.clear()
                continue
            with tempfile.TemporaryDirectory(prefix="kc-race-out-") as tmp:
                # probe も同時に待つ（順に待つと 1 ラウンドが 2 倍になる）
                mib = dict(zip(("a", "b"), _parallel(
                    lambda: wait_probe(exe, owner, "a", vers["a"],
                                       pathlib.Path(tmp) / "a", interval=interval),
                    lambda: wait_probe(exe, owner, "b", vers["b"],
                                       pathlib.Path(tmp) / "b", interval=interval))))
                print("  probe: " + "  ".join(
                    f"{s}={mib[s] if mib[s] is None else round(mib[s], 1)}"
                    for s in ("a", "b")))
                win = decide(mib["a"], mib["b"], gate)
                if win == "retry":
                    print("  両方ゲート未満 → 破棄して次へ")
                    for s in ("a", "b"):
                        delete_side(exe, owner, s)
                    live.clear()
                    continue
                lose = "b" if win == "a" else "a"
                if mib["a"] is not None and mib["b"] is not None:
                    print(f"  両方通過 → probe が速い {win} を残す")
                delete_side(exe, owner, lose)
                live.discard(lose)
                print(f"  {win} を本番として継続（install → vLLM 起動）")
                url = wait_url(exe, owner, win, vers[win],
                               pathlib.Path(tmp) / win,
                               limit_s=budget, interval=max(interval, 10))
                if url:
                    live.discard(win)
                    print(f"\nURL: {url}")
                    return 0
                print(f"  {win} から URL が出ないまま終了 → 破棄して次ラウンド")
                delete_side(exe, owner, win)
                live.discard(win)
    finally:
        # Ctrl-C・想定外の例外でも 2 本を残さない（残すと GPU を焼き続ける）。
        for s in sorted(live):
            delete_side(exe, owner, s)
    print(f"\n{rounds} ラウンドで当たらなかった（ゲート {gate} MiB/s）")
    return 1


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python3 -m cli race", description=__doc__)
    ap.add_argument("target", nargs="?", default="vllm-playground")
    ap.add_argument("--gate", type=float, default=150.0,
                    help="probe の下限 MiB/s（既定 150）")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--yes", action="store_true", help="実行を確定する")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=float, default=8.0,
                    help="probe のポーリング間隔（秒）")
    ap.add_argument("--wait-url", dest="wait_url_s", type=float, default=None,
                    help="勝ち側の URL 待ち上限（秒）")
    a = ap.parse_args(argv)
    return run(a.target, gate=a.gate, rounds=a.rounds, yes=a.yes,
               dry_run=a.dry_run, wait_url_s=a.wait_url_s, interval=a.interval)


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())

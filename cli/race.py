"""ノード速度レース: 2 本同時に push し、速いノードを引くまで繰り返す。

```
python3 -m cli race --yes              # 3 分〜（成功 1 本はそのまま本番）
python3 -m cli race --yes --gate 50    # 閾値を下げて早く当てる
```

## 何をするか

1. `KC_PROBE_GATE_MIB_S` と ntfy の topic を焼いて**2 回**ビルド
2. 別 id（`vllm-race-a` / `vllm-race-b`）へ**同時**に push
3. 各カーネルは 64 MiB probe の直後に結果を ntfy へ 1 行 POST し、
   ゲート未満なら `SystemExit`（install の 267 秒を払わない）
4. ローカルは ntfy を poll して `mib_s` を読む
5. 負け側は `kaggle kernels delete -y` で停止（実測 10〜20 秒）
6. 勝ち側の `url.txt` を待って URL を出す

## なぜ ntfy か

`kaggle kernels output` は**実行が終わるまで**取れない（実測）。
レースでは勝ち側が install 以降で何分も走るので、probe の結果を
「走行中に」読む手段が要る。カーネル間で共有できる場所は無いので、
外向きの HTTP 1 本（ntfy.sh）を中継にする。送信失敗は本番を止めない。

## 両方通ったとき

probe が速い方だけ残し、遅い方を即 delete（案 2）。両方 install まで
走らせると、重みロードが長いノードでは**最大 1.7 時間**を無駄に焼く。
両方通る確率は 1 ラウンド 0.38%（150 MiB/s が 6.2% のとき）。

## 前提

- `kaggle` CLI 2.x（バージョン指定 output / delete。1.x には無い）
- カーネルに internet（ntfy へ POST するため）
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
import urllib.request
from concurrent import futures

from cli import build as build_mod
from cli.push import _quota_lines, kaggle_exe
from kbuild import spec

# 2 本は**別 id** でなければ同時に走らない（同一 id の再 push は走行中の
# セッションを置き換える）。id は使い捨て。負けた側は delete で消える。
SLUGS = {"a": "vllm-race-a", "b": "vllm-race-b"}

NTFY = "https://ntfy.sh"
_VERSION = re.compile(r"Kernel version (\d+)")
_STATUS = re.compile(r"KernelWorkerStatus\.([A-Z]+)")
_BEACON = re.compile(r"mib_s=([0-9.]+)")
_URL = re.compile(r"url=(https?://[^\s]+)")


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


def parse_beacon(text, side=None):
    """ntfy の poll 出力から指定 side の最後の `mib_s` と url を取る。

    返り値は `(mib_s, url)` のタプル。どちらも取れなければ `(None, None)`。
    `mib_s` のみ、`url` のみ、両方、の3パターンがある。
    """
    mib, url = None, None
    for line in (text or "").splitlines():
        try:
            msg = json.loads(line).get("message") or ""
        except ValueError:
            continue
        if side is not None and not re.search(
            r"(?:^|\s)side=" + re.escape(side) + r"(?:\s|$)", msg):
            continue
        m = _BEACON.search(msg)
        if m:
            mib = float(m.group(1))
        u = _URL.search(msg)
        if u:
            url = u.group(1)
    return mib, url


def topics(stamp, side):
    """side ごとに別 topic。同一 topic だと 2 本の結果が混ざる。"""
    return f"kc-race-{stamp}-{side}"


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


def build_staged(kernel, side, owner, gate, topic, fake_mib=None):
    """ゲートと topic を焼いてビルドし、push 用ディレクトリを返す。

    side ごとに生成物が違う（topic と side を焼く）ので、2 回ビルドする。
    ビルドは 1 回 1〜2 秒で、GPU は使わない。

    `fake_mib` は `cpu-race-probe` 用。`KC_FAKE_MIB_<SIDE>` を焼く。
    """
    probe_env = {
        "KC_PROBE_GATE_MIB_S": str(gate),
        "KC_PROBE_NTFY_TOPIC": topic,
        "KC_PROBE_SIDE": side,
        "KC_FAKE_MIB_" + side.upper(): str(fake_mib or 0),
    }
    # `KC_FAKE_MIB_A` / `KC_FAKE_MIB_B` は両方復元する必要がある
    all_keys = set(probe_env) | {"KC_FAKE_MIB_A", "KC_FAKE_MIB_B"}
    previous = {k: os.environ.get(k) for k in all_keys}
    try:
        os.environ.update(probe_env)
        if build_mod.run([kernel.name]) != 0:
            raise SystemExit("生成に失敗したため中止")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"kc-race-{side}-"))
    src_dir = pathlib.Path(kernel.dir)
    shutil.copy2(src_dir / kernel.code_file, tmp / kernel.code_file)
    base = json.loads((src_dir / kernel.metadata_file).read_text(encoding="utf-8"))
    meta = stage_meta(base, side, owner)
    (tmp / kernel.metadata_file).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return tmp


def push_side(exe, staged, side):
    """1 本 push してバージョン番号を返す（失敗は 0）。"""
    rc, out, err = _run([exe, "kernels", "push", "-p", str(staged)], timeout=600)
    if rc != 0:
        print(f"  push {side} 失敗: {(err or out).strip()[:200]}")
        return 0
    ver = parse_version(out)
    print(f"  push {side}: v{ver}")
    return ver


def fetch_file(exe, owner, side, ver, pattern, dest, timeout=120):
    """バージョン指定で 1 ファイルだけ回収して Path（無ければ None）。

    注意: これは**実行が終わった版**でしか使えない（走行中は取れない）。
    probe は ntfy、URL は beacon / 完了後の回収で見る。
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
    err = ""
    for i in range(attempts):
        rc, _, err = _run([exe, "kernels", "delete", "-y", ref], timeout=180)
        if rc == 0:
            print(f"  delete {side}: ok")
            return True
        if i + 1 < attempts:
            time.sleep(5)
    print(f"  ⚠ delete {side} 失敗（走行中なら手で止める）: {err.strip()[:120]}")
    return False


def _supports_output_flag(exe, flag):
    """`kernels output` が flag を受けるか。

    `cli.push._supports` は **push** の `--help` を見るので、`--file-pattern`
    のような output 側のフラグには使えない（実測で誤判定した）。
    """
    rc, out, err = _run([exe, "kernels", "output", "--help"], timeout=120)
    return rc == 0 and flag in (out + err)


# ── 待ち合わせ ─────────────────────────────────────────────
def fetch_beacon(topic, timeout=10, side=None):
    """ntfy の poll から (mib_s, url) を取る（無ければ (None, None)）。"""
    try:
        with urllib.request.urlopen(f"{NTFY}/{topic}/json?poll=1",
                                    timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None, None
    return parse_beacon(text, side=side)


def wait_beacon(topic, side=None, limit_s=240, interval=5):
    """probe の beacon が届くまで待つ（届かなければ (None, None)）。"""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        mib, url = fetch_beacon(topic, side=side)
        if mib is not None or url is not None:
            return mib, url
        time.sleep(interval)
    return None, None


def wait_url_from_beacon(topics_, win, interval=5, limit_s=180):
    """ntfy の beacon から url を待つ（url が取れるまで）。"""
    t0 = time.time()
    topic = topics_[win]
    while time.time() - t0 < limit_s:
        _, url = fetch_beacon(topic, side=win)
        if url and url.startswith("http"):
            return url
        time.sleep(interval)
    return ""


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
        dry_run=False, wait_url_s=None, interval=5,
        fake_mib_a=None, fake_mib_b=None):
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
        print(f"DRY build {kernel.name} ×2 + push {SLUGS['a']}/{SLUGS['b']} ×{rounds}")
        return 0
    exe = kaggle_exe()
    if not exe:
        print("kaggle CLI が見つからない（PATH を確認）")
        return 1
    if not _supports_output_flag(exe, "--file-pattern"):
        print("この kaggle CLI はバージョン指定 output に未対応（2.x が要る）")
        print("  KC_KAGGLE=/path/to/kaggle2 を指定する")
        return 1

    budget = wait_url_s
    if budget is None:
        budget = float(os.environ.get("VLLM_STARTUP_BUDGET", "11000")) + 900
    stamp = f"{int(time.time())}-{os.getpid()}"

    live = set()          # 走らせたままの side（終了時に必ず片付ける）
    staged = {}
    try:
        for rnd in range(1, rounds + 1):
            print(f"\n=== ラウンド {rnd}/{rounds}")
            topics_ = {s: topics(f"{stamp}-{rnd}", s) for s in ("a", "b")}
            for s in ("a", "b"):
                if s in staged:
                    shutil.rmtree(staged[s], ignore_errors=True)
                staged[s] = build_staged(kernel, s, owner, gate, topics_[s],
                                         fake_mib=(fake_mib_a if s == "a" else fake_mib_b))
            # 2 本は**同時**に push する。順に押すと、2 本目が走り出す前に
            # 1 本目が probe を終えてしまい、レースにならない。
            vers = dict(zip(("a", "b"), _parallel(
                lambda: push_side(exe, staged["a"], "a"),
                lambda: push_side(exe, staged["b"], "b"))))
            live |= {s for s, v in vers.items() if v}
            if not all(vers.values()):
                undeleted = []
                for s in sorted(live):
                    if delete_side(exe, owner, s):
                        live.discard(s)
                    else:
                        undeleted.append(s)
                if undeleted:
                    raise RuntimeError(f"delete 失敗: {', '.join(undeleted)}")
                continue
            # probe と url は ntfy から読む（output は完了まで取れない）
            beacons = dict(zip(("a", "b"), _parallel(
                lambda: wait_beacon(topics_["a"], side="a", interval=interval),
                lambda: wait_beacon(topics_["b"], side="b", interval=interval))))
            probe = {s: b[0] for s, b in beacons.items()}
            url_recv = {s: b[1] for s, b in beacons.items()}
            print("  probe: " + "  ".join(
                f"{s}={probe[s] if probe[s] is None else round(probe[s], 1)}"
                for s in ("a", "b")))
            print("  url : " + "  ".join(
                f"{s}={url_recv[s] or '-'}" for s in ("a", "b")))
            win = decide(probe["a"], probe["b"], gate)
            if win == "retry":
                print("  両方ゲート未満 → 破棄して次へ")
                undeleted = []
                for s in ("a", "b"):
                    if delete_side(exe, owner, s):
                        live.discard(s)
                    else:
                        undeleted.append(s)
                if undeleted:
                    raise RuntimeError(f"delete 失敗: {', '.join(undeleted)}")
                continue
            lose = "b" if win == "a" else "a"
            if url_recv[win]:
                live.discard(win)
                print(f"\nURL（beacon）: {url_recv[win]}")
                return 0
            if probe["a"] is not None and probe["b"] is not None:
                print(f"  両方通過 → probe が速い {win} を残す")
            if not delete_side(exe, owner, lose):
                raise RuntimeError(f"delete 失敗: {lose}")
            live.discard(lose)
            print(f"  {win} を本番として継続（install → vLLM 起動）")
            # cpu-race-probe は ntfy から url を取る（kernels output は走行中に取れない）
            url = wait_url_from_beacon(topics_, win, interval=max(interval, 10),
                                       limit_s=budget)
            if url:
                live.discard(win)
                print(f"\nURL: {url}")
                return 0
            print(f"  {win} から URL が出ないまま終了 → 破棄して次ラウンド")
            if not delete_side(exe, owner, win):
                raise RuntimeError(f"delete 失敗: {win}")
            live.discard(win)
    finally:
        # Ctrl-C・想定外の例外でも 2 本を残さない（残すと GPU を焼き続ける）。
        for s in sorted(live):
            delete_side(exe, owner, s)
        for d in staged.values():
            shutil.rmtree(d, ignore_errors=True)
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
    ap.add_argument("--interval", type=float, default=5.0,
                    help="beacon のポーリング間隔（秒）")
    ap.add_argument("--wait-url", dest="wait_url_s", type=float, default=None,
                    help="勝ち側の URL 待ち上限（秒）")
    ap.add_argument("--fake-mib-a", dest="fake_mib_a", type=float, default=None,
                    help="cpu-race-probe 用: side a の偽速度（既定: gate に応じて自動）")
    ap.add_argument("--fake-mib-b", dest="fake_mib_b", type=float, default=None,
                    help="cpu-race-probe 用: side b の偽速度（既定: gate に応じて自動）")
    a = ap.parse_args(argv)
    return run(a.target, gate=a.gate, rounds=a.rounds, yes=a.yes,
               dry_run=a.dry_run, wait_url_s=a.wait_url_s, interval=a.interval,
               fake_mib_a=a.fake_mib_a, fake_mib_b=a.fake_mib_b)


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())

"""`result.json` から起動時間の内訳を表にする（実行間の比較用）。

GPU を回すたびに「速くなった / 遅くなった」を数字で言えるようにするための道具。
**測っていない項目は `-`**（0 で埋めると「一瞬で終わった」と誤読する）。

使い方:
    python3 -m cli timings out/run1/result.json out/run2/result.json
"""
import json
import os

# 表示順。ここを増やせば表が増える（result.json に無ければ `-`）。
SPAN_ORDER = (
    ("weights", "重みロード"),
    ("compile", "torch.compile"),
    ("after_compile", "compile 後〜capture"),
    ("capture", "graph capture"),
)


def _num(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def summarize(data):
    """1 つの result.json を 1 行分の dict にする（表示に必要な最小限）。"""
    t = data.get("timings") or {}
    spans = t.get("startup_spans") or {}
    marks = t.get("startup_marks") or {}
    attempt = t.get("attempt") or {}
    install = t.get("install_seconds")
    load = t.get("load_seconds")
    row = {
        "engine": data.get("engine"),
        "elapsed": data.get("elapsed_seconds"),
        "install": install,
        "load": load,
        # 「待たされる時間」は install + 起動。総計は keepalive を含むので
        # 比較に使えない（実測 2026-11-01: 総計 4637s のうち 3660s は配信）。
        "ready": (round(install + load, 1)
                  if install is not None and load is not None else None),
        "label": attempt.get("label"),
        "maxlen": attempt.get("max_model_len") or data.get("context_length"),
        "torch": t.get("torch_after_install") or data.get("torch"),
        "wh": "on" if (t.get("wheelhouse") or {}).get("no_index") else None,
        "inst": t.get("installer"),
        "speed": t.get("speed_tps"),
        "vram": t.get("gpu_vram_mib"),
        "ok": data.get("ok"),
        "marks": marks,
        "gaps": t.get("startup_gaps") or [],
    }
    for key, _ in SPAN_ORDER:
        row[key] = spans.get(key)
    return row


def table(rows):
    """比較表（文字列）を返す。値が無い列は `-`。"""
    cols = [("engine", "engine"), ("elapsed", "総計 s"), ("install", "install s"),
            ("load", "起動 s"), ("ready", "起動まで s"), ("inst", "inst"),
            ("label", "段"), ("maxlen", "ctx"), ("torch", "torch"),
            ("wh", "wh"), ("speed", "t/s"), ("vram", "VRAM MiB"), ("ok", "ok")]
    cols += [(k, label) for k, label in SPAN_ORDER]
    width = {}
    for key, head in cols:
        width[key] = max(len(head), *(len(_num(r.get(key))) for r in rows))
    out = []
    out.append("  ".join(head.ljust(width[key]) for key, head in cols).rstrip())
    out.append("  ".join("-" * width[key] for key, _ in cols))
    for r in rows:
        out.append("  ".join(_num(r.get(key)).ljust(width[key])
                             for key, _ in cols).rstrip())
    return "\n".join(out)


def load(path):
    with open(path, encoding="utf-8") as f:
        return summarize(json.load(f))


def main(paths):
    if not paths:
        print("result.json を 1 つ以上指定してください")
        return 1
    rows = []
    for p in paths:
        try:
            r = load(p)
        except (OSError, ValueError) as e:
            print(f"⚠ {p}: {e}")
            continue
        r["engine"] = f"{r['engine']} ({os.path.basename(os.path.dirname(p))})"
        rows.append(r)
    if not rows:
        return 1
    print(table(rows))
    # 節目の生値も出す（表に無い区間の正体を追うとき用）。
    for r in rows:
        if r["marks"]:
            print(f"\n{r['engine']} の節目:")
            for k, v in sorted(r["marks"].items(), key=lambda kv: kv[1]):
                print(f"  {v:8.1f}s  {k}")
        # 無音区間は「長さ」より「何の直前/直後か」が効く。ここを読まないと
        # 次の一手が決まらない。
        for g in r["gaps"]:
            print(f"  無音 {g['seconds']:6.1f}s @{g['from']:7.1f}s: "
                  f"「{g['before'][:70]}」→「{g['after'][:70]}」")
    return 0

#!/usr/bin/env python3
"""GPU 版 notebook の「実際に走るコード」をローカルで単体テストする。

Kaggle セッションを 1 本も使わない。notebook から関数を AST で抜き出して
exec するので、**ビルダーではなく生成物そのもの**を検証できる（drift 検出）。

対象:
  1. `_stage_model()` / `_stage_draft()` … symlink の取捨（mtp の出し入れ）
  2. `probe_url()` … cloudflared ログからの URL 抽出
  3. `_smoke()` … 起動後スモークの合否判定（200 / 500 / 接続不可 / usage:null）

    python3 kaggle/step2/_selftest_builder.py
"""
import ast
import http.server
import json
import os
import pathlib
import shutil
import socketserver
import sys
import tempfile
import threading

HERE = pathlib.Path(__file__).resolve().parent
NB = (HERE.parent / "step2-vllm-kernel" / "cloudflared"
      / "step2-vllm-qwen38-27b-cloudflared.ipynb")

FAILS = []


def check(name, cond, detail=""):
    print(("  ✅ " if cond else "  ❌ ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def cell_source(pred):
    nb = json.load(open(NB))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if pred(src):
            return src
    raise SystemExit(f"notebook に該当セルが無い: {NB}")


def extract(src, names):
    """セルから FunctionDef / 代入文だけを抜き出して exec 可能な塊にする。"""
    tree = ast.parse(src)
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                kept.append(node)
        elif isinstance(node, ast.Assign):
            tgts = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in names for t in tgts):
                kept.append(node)
    if not kept:
        raise SystemExit(f"抽出 0 件: {names}")
    mod = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, "<nb>", "exec")


def runtime_source():
    """埋め込み runtime バンドル（kc_* 関数群）のセルを返す。"""
    return cell_source(lambda s: "def kc_probe_url" in s and "def kc_start_proxy" in s)


def runtime_ns():
    """runtime バンドルを exec した名前空間を返す（notebook と同じ実体）。"""
    ns = {}
    exec(compile(runtime_source(), "<runtime>", "exec"), ns)
    return ns


# --------------------------------------------------------------------------
# 1. _stage_model / _stage_draft
# --------------------------------------------------------------------------
def test_stage():
    print("\n=== _stage_model / _stage_draft ===")
    src = cell_source(lambda s: "def _stage_model" in s)
    # 定数は notebook の値をそのまま使いつつ、パスだけ tmp に差し替える
    # （notebook 側の代入を exec すると /kaggle を掴んで落ちる）。
    code = extract(src, {"_stage_model", "_stage_draft"})
    code_const = extract(src, {"MTP_FILE", "DRAFT_KEEP"})

    root = pathlib.Path(tempfile.mkdtemp(prefix="modelsrc_"))
    work = pathlib.Path(tempfile.mkdtemp(prefix="work_"))
    files = ["config.json", "generation_config.json", "chat_template.jinja",
             "chat_template_medium.jinja", "tokenizer.json", "tokenizer_config.json",
             "model.safetensors", "model-mtp.safetensors", ".gitattributes"]
    for fn in files:
        (root / fn).write_bytes(b"x" * 10)
    (root / "subdir").mkdir()          # ディレクトリは symlink しない

    const = {}
    exec(code_const, const)
    ns = {"os": os, "shutil": shutil,
          "STAGE": str(work / "stage"), "MODEL_SRC": str(root),
          "DRAFT_DIR": str(work / "draft"),
          "MTP_FILE": const["MTP_FILE"], "DRAFT_KEEP": const["DRAFT_KEEP"]}
    exec(code, ns)
    check("notebook の MTP_FILE が model-mtp.safetensors",
          const["MTP_FILE"] == "model-mtp.safetensors", str(const["MTP_FILE"]))
    check("DRAFT_KEEP に tokenizer/config が入っている",
          "tokenizer" in const["DRAFT_KEEP"] and "config" in const["DRAFT_KEEP"],
          str(const["DRAFT_KEEP"]))

    # --- target 用（mtp 抜き）---
    n = ns["_stage_model"](with_mtp=False)
    got = sorted(os.listdir(ns["STAGE"]))
    check("_stage_model(False): mtp を除外", "model-mtp.safetensors" not in got, str(got))
    check("_stage_model(False): 本体は入る", "model.safetensors" in got)
    check("_stage_model(False): 隠しファイル除外", ".gitattributes" not in got)
    check("_stage_model(False): ディレクトリ除外", "subdir" not in got)

    # --- target 用（mtp 込み）---
    n2 = ns["_stage_model"](with_mtp=True)
    got2 = sorted(os.listdir(ns["STAGE"]))
    check("_stage_model(True): mtp を残す", "model-mtp.safetensors" in got2)
    check("_stage_model: 2 回目も作れる（rmtree 後再作成）", n2 == len(got2), f"{n2}")

    # --- draft 用 ---
    m = ns["_stage_draft"]()
    dg = sorted(os.listdir(ns["DRAFT_DIR"]))
    real = [f for f in dg if not os.path.islink(os.path.join(ns["DRAFT_DIR"], f))]
    check("_stage_draft: symlink のみ", not real, str(real))
    check("_stage_draft: mtp を含む", "model-mtp.safetensors" in dg, str(dg))
    check("_stage_draft: 本体を含まない", "model.safetensors" not in dg, str(dg))
    check("_stage_draft: tokenizer を含む", "tokenizer.json" in dg)
    globbed = sorted(f for f in dg if f.endswith(".safetensors"))
    check("_stage_draft: safetensors は 1 個（=17GB を読まない）",
          globbed == ["model-mtp.safetensors"], str(globbed))
    check("_stage_draft: 返り値 == 作った数", m == len(dg), f"{m}/{len(dg)}")

    # --- mtp が無い dataset（安全網）---
    shutil.rmtree(ns["DRAFT_DIR"])
    os.remove(root / "model-mtp.safetensors")
    m2 = ns["_stage_draft"]()
    check("_stage_draft: mtp 無しなら 0 を返す", m2 == 0, str(m2))
    check("_stage_draft: mtp 無しで dir を作らない",
          not os.path.isdir(ns["DRAFT_DIR"]))
    n3 = ns["_stage_model"](with_mtp=True)
    check("_stage_model: mtp 無しでも落ちない", n3 >= 1, str(n3))

    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------
# 2. probe_url
# --------------------------------------------------------------------------
def test_probe_url():
    print("\n=== probe_url（cloudflared ログ） ===")
    src = cell_source(lambda s: "def probe_url" in s)
    code = extract(src, {"probe_url"})

    tmp = tempfile.mkdtemp(prefix="cf_")
    log = os.path.join(tmp, "cloudflared.log")
    ns = runtime_ns()
    ns["TUNNEL_LOG"] = log
    exec(code, ns)

    samples = [
        ('2026-01-01T00:00:00Z INF +-----+\n'
         '2026-01-01T00:00:00Z INF |  https://brave-tiger-xyz.trycloudflare.com  |\n'
         '2026-01-01T00:00:00Z INF +-----+\n', "https://brave-tiger-xyz.trycloudflare.com"),
        ("no url here\n", ""),
        ("https://not-atunnel.example.com\n", ""),
    ]
    for text, want in samples:
        open(log, "w").write(text)
        got = ns["probe_url"]()
        check(f"probe_url → {want or '(空)'}", got == want, got)

    os.remove(log)
    check("probe_url: ログ無しでも落ちない", ns["probe_url"]() == "")
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 3. _smoke
# --------------------------------------------------------------------------
class _H(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        if self.mode == "ok":
            payload = {"choices": [{"message": {"content": "OK"},
                                    "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 1}}
            self.send_response(200)
        elif self.mode == "usage_null":
            payload = {"choices": [{"message": {"content": ""},
                                    "finish_reason": "length"}],
                       "usage": None}
            self.send_response(200)
        elif self.mode == "reasoning":
            payload = {"choices": [{"message": {"content": "",
                                                "reasoning_content": "考え中"},
                                    "finish_reason": "length"}],
                       "usage": {"completion_tokens": 4}}
            self.send_response(200)
        elif self.mode == "err500":
            payload = {"error": {"message": "CUDA error: no kernel image"}}
            self.send_response(500)
        else:
            payload = {}
            self.send_response(400)
        raw = json.dumps(payload).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        raw = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def test_smoke():
    print("\n=== _smoke（起動後スモーク） ===")
    src = cell_source(lambda s: "def _smoke" in s)
    code = extract(src, {"_smoke"})

    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    ns = {"json": json, "time": __import__("time"),
          "urllib": __import__("urllib.request", fromlist=["request", "error"]),
          "SERVED_NAME": "m", "VLLM_PORT": port}
    # urllib.request / urllib.error を明示的に名前空間へ
    import urllib.error
    import urllib.request
    ns["urllib"] = urllib
    exec(code, ns)

    for mode, want, label in (("ok", True, "200 正常"),
                              ("usage_null", True, "usage: null"),
                              ("reasoning", True, "content 空 + reasoning のみ"),
                              ("err500", False, "500 カーネルエラー"),
                              ("err400", False, "400")):
        _H.mode = mode
        ok, detail = ns["_smoke"](attempts=1, timeout=10)
        check(f"_smoke {label} → {want}", ok is want, detail[:80])

    srv.shutdown()
    srv.server_close()

    # 接続不可（ポート閉鎖）
    ns["VLLM_PORT"] = 1
    ok, _ = ns["_smoke"](attempts=1, timeout=5)
    check("_smoke 接続不可 → False", ok is False)


# --------------------------------------------------------------------------
# 4. PLAN ラダーの不変条件（テキスト検査）
# --------------------------------------------------------------------------
def test_plan():
    print("\n=== PLAN ラダーの不変条件 ===")
    src = cell_source(lambda s: 'PLAN.append(("mtp+dd"' in s)
    lines = [ln.strip() for ln in src.splitlines()
             if ln.strip().startswith("PLAN.append(")]

    # 折り返し行を後ろの行と連結する
    rows, buf = [], ""
    for ln in src.splitlines():
        if ln.strip().startswith("PLAN.append("):
            if buf:
                rows.append(buf)
            buf = ln.strip()
        elif buf and ("LMO" in ln or "AUTO_Q" in ln or "CT_Q" in ln
                      or "]" in ln or ")" in ln):
            buf += " " + ln.strip()
            if buf.count(")") >= 2:
                rows.append(buf); buf = ""
    if buf:
        rows.append(buf)

    check("PLAN 段が 8〜10 個", 8 <= len(rows) <= 10, f"{len(rows)} 段")
    for r in rows:
        label = r.split('"')[1] if r.count('"') > 1 else "?"
        spec = "SPEC" in r.split(",")[1] if "," in r else False
        print(f"    {label:8s} {r[:110]}")
        if "enforce-eager" in r:
            check(f"{label}: enforce-eager に spec を併用しない", "SPEC" not in r)
            # eager は「切り分け用の最終手段」。mtp 再挑戦と vl0 の手前に置く。
            check(f"{label}: eager の位置が末尾グループ",
                  rows.index(r) >= len(rows) - 3,
                  f"index {rows.index(r)} / {len(rows)}")
        if "TRITON_ATTN" in r:
            check(f"{label}: attention-backend は spec なし", "SPEC" not in r)
        if "CT_Q" in r:
            check(f"{label}: --quantization 明示は 1 段だけ",
                  sum(1 for x in rows if "CT_Q" in x) == 1)
    check("assert len(p) == 7 が存在",
          "assert all(len(p) == 7" in src)
    check("mtp+dd が先頭", rows[0].split('"')[1] == "mtp+dd", rows[0][:60])
    check("draft dir 無しの mtp 再挑戦が後半",
          [i for i, r in enumerate(rows) if '"mtp"' in r and "SPEC" in r][0] >= 5)


# --------------------------------------------------------------------------
# 5. _open_public（IPv6 経路なし対策）
# --------------------------------------------------------------------------
def test_open_public():
    print("\n=== _open_public（IPv4 フォールバック） ===")
    src = cell_source(lambda s: "def _open_public" in s)
    code = extract(src, {"_open_public"})
    import socket
    import urllib.error
    import urllib.request

    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _H.mode = "ok"
    url = f"http://127.0.0.1:{port}/healthz"

    ns = {"socket": socket, "urllib": urllib}
    exec(code, ns)
    with ns["_open_public"](url, timeout=10) as r:
        check("通常経路で開ける", r.status == 200)

    # 1 回目だけ URLError → IPv4 強制で再試行し、解決 family を検証する
    orig_open = urllib.request.urlopen
    orig_gai = socket.getaddrinfo
    seen = {"n": 0, "family": None, "restored": None}

    def flaky(*a, **kw):
        seen["n"] += 1
        if seen["n"] == 1:
            raise urllib.error.URLError("Network is unreachable")
        seen["family"] = socket.getaddrinfo.__name__
        seen["restored"] = (socket.getaddrinfo is orig_gai)
        return orig_open(*a, **kw)

    urllib.request.urlopen = flaky
    try:
        ns["_open_public"](url, timeout=10).read()
        check("1 回目失敗 → 2 回目で成功", seen["n"] == 2, f"calls={seen['n']}")
        check("再試行中は getaddrinfo を差し替え", seen["family"] == "_v4",
              str(seen["family"]))
        check("再試行中はまだ差し替わっている", seen["restored"] is False,
              str(seen["restored"]))
        check("終了後は getaddrinfo を元に戻す", socket.getaddrinfo is orig_gai)
    finally:
        urllib.request.urlopen = orig_open

    # 両方失敗 → 最初の例外を投げる
    def always(*a, **kw):
        raise urllib.error.URLError("Network is unreachable")

    urllib.request.urlopen = always
    try:
        try:
            ns["_open_public"](url, timeout=5)
            check("両方失敗なら例外", False)
        except urllib.error.URLError:
            check("両方失敗なら例外", True)
        check("例外後も getaddrinfo は元通り", socket.getaddrinfo is orig_gai)
    finally:
        urllib.request.urlopen = orig_open

    srv.shutdown()
    srv.server_close()


# --------------------------------------------------------------------------
# 6. save_output / fail（成果物が残るか）
# --------------------------------------------------------------------------
def test_save_output():
    print("\n=== save_output / fail ===")
    ns = runtime_ns()
    save = ns["kc_save_output"]
    fail = ns["kc_fail"]

    d1 = pathlib.Path(tempfile.mkdtemp(prefix="out1_"))
    d2 = pathlib.Path(tempfile.mkdtemp(prefix="out2_"))
    blocker = pathlib.Path(tempfile.mkdtemp(prefix="blk_")) / "notadir"
    blocker.write_text("x")          # dir として使えないパス

    check("両方書ける → True", save("a.txt", "hi", (str(d1), str(d2))) is True)
    check("両方に書けている", (d1 / "a.txt").read_text() == "hi"
          and (d2 / "a.txt").read_text() == "hi")

    check("片方失敗でも True（1 つ書ければ OK）",
          save("b.txt", "x", (str(d1), str(blocker))) is True
          and (d1 / "b.txt").is_file())

    check("全滅なら False（例外にしない）",
          save("c.txt", "x", (str(blocker), str(blocker))) is False)

    tlog = d1 / "tail.log"
    tlog.write_text("line1\nline2\n")
    try:
        fail("boom", str(tlog), (str(d1),))
        check("fail は RuntimeError", False)
    except RuntimeError as e:
        check("fail は RuntimeError", "boom" in str(e))
    txt = (d1 / "FAILED.txt").read_text()
    check("fail は FAILED.txt に理由とログ末尾を残す",
          "boom" in txt and "line2" in txt, txt[:40].replace("\n", " "))

    for d in (d1, d2, blocker.parent):
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# 7. keepalive（RUN/MINUTES の読み取り・mtime 検出・延長の算術）
# --------------------------------------------------------------------------
def test_keepalive():
    print("\n=== keepalive 設定の読み取り ===")
    src = cell_source(lambda s: "def _read_keepalive_conf" in s)
    code = extract(src, {"_read_keepalive_conf", "_conf_mtime"})
    const = extract(src, {"KEEPALIVE_KEEPALIVE", "MAX_MINUTES"})
    c = {}
    exec(const, c)
    check("KEEPALIVE_KEEPALIVE == 60", c["KEEPALIVE_KEEPALIVE"] == 60,
          str(c["KEEPALIVE_KEEPALIVE"]))
    check("MAX_MINUTES がクォータ内（<=35）", c["MAX_MINUTES"] <= 35,
          str(c["MAX_MINUTES"]))

    d = pathlib.Path(tempfile.mkdtemp(prefix="ka_"))
    conf = d / "keepalive.conf"
    ns = runtime_ns()
    ns.update({"KEEPALIVE_CONF": str(conf),
               "KEEPALIVE_KEEPALIVE": c["KEEPALIVE_KEEPALIVE"],
               "MAX_MINUTES": c["MAX_MINUTES"]})

    def _fail(msg, *a, **kw):
        raise RuntimeError(msg)

    ns["fail"] = _fail
    exec(code, ns)
    rd = ns["_read_keepalive_conf"]
    mt = ns["_conf_mtime"]

    capped = min(c["KEEPALIVE_KEEPALIVE"], c["MAX_MINUTES"])
    check("ファイル無し → (1, min(60, MAX))", rd() == (1, capped), str(rd()))
    check("_conf_mtime ファイル無し → 0", mt() == 0, str(mt()))

    conf.write_text("RUN=1\nMINUTES=60\n")
    check("RUN=1/MINUTES=60 を読める（上限で頭打ち）",
          rd() == (1, capped), str(rd()))

    conf.write_text("RUN=0\n")
    check("RUN=0 を読める（MINUTES は既定 60）", rd() == (0, 60), str(rd()))

    conf.write_text("RUN=0\nMINUTES=1\n")
    # RUN=0 は即座に停止扱い。MINUTES は既定値に戻る（延長計算に使わない）。
    check("RUN=0 は MINUTES に関係なく停止", rd() == (0, 60), str(rd()))

    conf.write_text("RUN=1\nMINUTES=999\n")
    check("MINUTES は MAX_MINUTES で頭打ち", rd() == (1, c["MAX_MINUTES"]),
          str(rd()))

    conf.write_text("RUN=1\nMINUTES=0\n")
    try:
        rd()
        check("MINUTES=0 は fail()", False)
    except RuntimeError:
        check("MINUTES=0 は fail()", True)

    conf.write_text("めちゃくちゃ\n")
    check("壊れた内容 → 既定 (1, min(60,MAX))", rd() == (1, capped), str(rd()))

    print("\n=== 延長の検出（mtime） ===")
    conf.write_text("RUN=1\nMINUTES=60\n")
    m1 = mt()
    check("mtime が取れる", m1 > 0, str(m1))
    import time as _t
    _t.sleep(0.01)
    conf.write_text("RUN=1\nMINUTES=60\n")
    m2 = mt()
    check("10ms 後の再書き込みで mtime が変わる（検出可能）", m2 != m1,
          f"{m1} -> {m2}")

    print("\n=== 延長の算術（notebook の if ブロックを実行） ===")
    tree = ast.parse(src)
    node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and "conf_mtime" in ast.dump(n.test):
            node = n
            break
    if node is None:
        check("延長ブロックが見つかる", False)
    else:
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        eco = compile(mod, "<ext>", "exec")

        ns2 = {"print": lambda *a, **k: None,
               "MAX_MINUTES": c["MAX_MINUTES"], "conf_mtime": 1.0,
               "_conf_mtime": lambda: 1.0, "_m": 1.0}
        # 変化なし → limit 据え置き
        ns2.update({"i": 5, "limit": 20, "KEEPALIVE_MINUTES": 60})
        exec(eco, ns2)
        check("mtime 変化なし → 延長しない", ns2["limit"] == 20, str(ns2["limit"]))

        # 変化あり（i=5, 60 分）→ min(4+60, MAX)
        ns2["_m"] = 2.0
        exec(eco, ns2)
        check("mtime 変化あり → i-1+KEEPALIVE（上限で頭打ち）",
              ns2["limit"] == min(4 + 60, c["MAX_MINUTES"]), str(ns2["limit"]))

        # 1 周目で延長（i=1）
        ns3 = {"print": lambda *a, **k: None, "MAX_MINUTES": c["MAX_MINUTES"],
               "conf_mtime": 1.0, "_conf_mtime": lambda: 9.0, "_m": 9.0,
               "i": 1, "limit": 20, "KEEPALIVE_MINUTES": 60}
        exec(eco, ns3)
        check("1 周目で延長 → 上限いっぱい", ns3["limit"] == c["MAX_MINUTES"],
              str(ns3["limit"]))

    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# 8. help の完全性判定（0.32 の要約 help を完全版と誤認しない）
# --------------------------------------------------------------------------
def test_help_complete():
    print("\n=== _looks_complete（--help の要約/完全版） ===")
    src = cell_source(lambda s: "def _looks_complete" in s)
    code = extract(src, {"_looks_complete"})
    ns = {}
    exec(code, ns)
    f = ns["_looks_complete"]

    truncated = (
        "usage: vllm serve [model_tag] [options]\n\n"
        "Config Groups:\n  ModelConfig\n  LoadConfig\n  AttentionConfig\n"
        * 40
    )
    check("0.32 の要約 help（フラグ無し）は False", f(truncated) is False,
          f"{len(truncated)}B")

    full = (truncated + "\n".join(["  --dummy-flag VALUE"] * 250)
            + "\n  --tensor-parallel-size N\n  --max-model-len N\n")
    check("フラグ入り完全版は True", f(full) is True, f"{len(full)}B")

    check("短すぎる本文は False", f("--tensor-parallel-size --max-model-len") is False,
          "flag 名だけ")

    check("--help=all を試す実装になっている",
          "--help=all" in cell_source(lambda s: "_run_help" in s))




# --------------------------------------------------------------------------
# 9. セル順の「定義前使用」検出
#    2026-10-29 の GPU 実行は `if STARTUP_BUDGET < 900:` を定数定義より前に
#    置いていたため NameError で 80 秒で即死した（クォータ 1.5 分の無駄）。
#    関数の中は実行時解決なので対象外。**モジュールレベルの使用のみ**見る。
# --------------------------------------------------------------------------
class _Order(ast.NodeVisitor):
    """モジュールレベルで実行される Name の出現位置を集める。

    関数/ラムダ/クラス本体は実行が遅延するので覗かない（引数既定値だけ見る）。
    if/try/for/with の中は実行されるので再帰する。
    """

    def __init__(self):
        self.stores = {}   # name -> 最初の代入行
        self.loads = {}    # name -> 最初の参照行

    def _defaults(self, node):
        for d in list(node.args.defaults) + [x for x in node.args.kw_defaults
                                            if x is not None]:
            self.visit(d)

    def visit_FunctionDef(self, node):
        self.stores.setdefault(node.name, node.lineno)
        for d in node.decorator_list:
            self.visit(d)
        self._defaults(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stores.setdefault(node.name, node.lineno)
        for d in node.decorator_list + node.bases:
            self.visit(d)

    def visit_Lambda(self, node):
        self._defaults(node)

    def _comp(self, node, parts):
        # 内包表記のターゲットは内包スコープの局所変数。行番号で
        # 「先に load している」と誤判定しないよう、式の行で定義済み扱いにする。
        for gen in node.generators:
            self.visit(gen.iter)
        for gen in node.generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    self.stores.setdefault(n.id, node.lineno)
        for p in parts:
            self.visit(p)

    def visit_ListComp(self, node):
        self._comp(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._comp(node, [node.key, node.value])

    def visit_ExceptHandler(self, node):
        # `except X as e:` の e は Name ノードではないので手当てする
        if node.name:
            self.stores.setdefault(node.name, node.lineno)
        if node.type is not None:
            self.visit(node.type)
        for st in node.body:
            self.visit(st)

    def visit_Import(self, node):
        for a in node.names:
            self.stores.setdefault((a.asname or a.name).split(".")[0], node.lineno)

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.stores.setdefault(a.asname or a.name, node.lineno)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.setdefault(node.id, node.lineno)
        else:
            self.stores.setdefault(node.id, node.lineno)


def test_cell_name_order():
    print("\n=== セル順の定義前使用（NameError 予防） ===")
    import builtins
    known = set(dir(builtins)) | {"__name__", "__file__"}
    problems = []
    nb = json.load(open(NB))
    for ci, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        o = _Order()
        o.visit(ast.parse("".join(c["source"])))
        for name, line in sorted(o.loads.items(), key=lambda kv: kv[1]):
            if name in known:
                continue
            if name in o.stores and o.stores[name] < line:
                continue
            if name in o.stores and o.stores[name] == line:
                continue
            problems.append(f"cell{ci}:L{line}:{name}")
        known |= set(o.stores)
    check("定義前使用なし", not problems,
          ", ".join(dict.fromkeys(problems))[:400])


if __name__ == "__main__":
    test_plan()
    test_open_public()
    test_save_output()
    test_keepalive()
    test_help_complete()
    test_cell_name_order()
    test_stage()
    test_probe_url()
    test_smoke()
    print()
    if FAILS:
        print(f"結果: {len(FAILS)} 件失敗")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("結果: 全テスト パス")

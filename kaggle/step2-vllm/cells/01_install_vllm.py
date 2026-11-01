# vLLM をインストールし、プロキシと UI を書き出す
import base64
import glob as _glob
import os
import shutil
import subprocess
import sys
import time

BIN_DIR = "/usr/local/bin"
MIN_SIZE = 1024 * 1024

# ---- 計測（高度化の基礎）----
# result.json の `timings.install_seconds` は **None のまま**だった。vLLM は
# install をテンプレートの `stage_install()` ではなくこのセルで行うため、
# 誰も記録していなかった。比較できないと「速くなった/遅くなった」を言えない。
# 区間の**絶対オフセット**（セル開始からの秒）を残す。差分は後から計算できる。
_T_INSTALL = time.time()
_MARKS = {}


def _mark(name):
    """区間の到達時刻を記録する（セル開始からの秒）。"""
    _MARKS[name] = round(time.time() - _T_INSTALL, 1)
    return _MARKS[name]

# ---- cloudflared（Dataset キャッシュ優先）----
# ngrok は 2026-11-01 に廃止したので、トンネルの種類で分岐しない
# （`notes/url-publishing.md`）。cloudflared は常に用意する。
CF_PATH = kc_find_dataset("my-cloudflared-bin")
dst = os.path.join(BIN_DIR, "cloudflared")
src = None
for cand in (os.path.join(CF_PATH, "cloudflared"),
             os.path.join(CF_PATH, "bin", "cloudflared")):
    if os.path.isfile(cand) and os.path.getsize(cand) >= MIN_SIZE:
        src = cand
        break
if src:
    shutil.copy2(src, dst)
    subprocess.run(["chmod", "+x", dst])
    print(f"✅ cloudflared 復元: {src}")
elif not (os.path.isfile(dst) and os.path.getsize(dst) >= MIN_SIZE):
    print("⬇️  cloudflared をダウンロード中...")
    subprocess.run(
        "curl -fsSL --connect-timeout 15 -o /usr/local/bin/cloudflared "
        f"{kc_cloudflared_url()}"
        " && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=False, timeout=300,
    )
if not os.path.isfile(dst):
    fail("cloudflared がありません（Dataset 未マウント / 取得失敗）")
r = subprocess.run([dst, "--version"], capture_output=True, text=True)
print((r.stdout or r.stderr).strip())
if r.returncode != 0:
    fail("cloudflared の実行に失敗しました（実行ビット/Dataset を確認）")

# ---- vLLM ----
# wheelhouse Dataset（vllm 0.32.0 + 依存 wheel / 3.25 GB）を 2 つに分けて持つ:
#   vllm-wheelhouse        … 190 wheel（torch 2.13.0 + cu13 一式）
#   vllm-wheelhouse-patch  … nvidia-cublas 13.1.1.3 のみ
# `cuda-toolkit 13.0.3` が cublas を `==13.1.1.3.*` で固定するため、本体側の
# 13.8.2.12 だけでは `--no-index` が解決できない（CPU 実測 2026-10-31）。
# 解決に失敗したら PyPI にフォールバックする（起動できない方が損失が大きい）。
_WH_SLUGS = ("vllm-wheelhouse", "vllm-wheelhouse-patch")
_pip_extra = []


def _whl_dir(root):
    if _glob.glob(os.path.join(root, "*.whl")):
        return root
    for _s in sorted(_glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(_s) and _glob.glob(os.path.join(_s, "*.whl")):
            return _s
    return None


def _unzip_wheels(root, tag):
    """--dir-mode zip の Dataset を 1 回だけ展開する。"""
    _zips = (_glob.glob(os.path.join(root, "*.zip"))
             + _glob.glob(os.path.join(root, "*", "*.zip")))
    if not _zips:
        return None
    import time as _t
    import zipfile as _zf
    _dst = os.path.join("/kaggle/tmp/whl", tag)
    os.makedirs(_dst, exist_ok=True)
    _t0 = _t.time()
    print(f"📦 {tag} の zip を展開: {[os.path.basename(z) for z in _zips]}")
    for _z in _zips:
        with _zf.ZipFile(_z) as _f:
            _f.extractall(_dst)
    print(f"   展開 {_t.time() - _t0:.1f}s")
    return _whl_dir(_dst) or _dst


_wh_dirs = []
for _slug in _WH_SLUGS:
    _root = kc_find_dataset(_slug)
    if not os.path.isdir(_root) or not os.listdir(_root):
        print(f"ℹ {_slug} なし")
        continue
    _d = _whl_dir(_root) or _unzip_wheels(_root, _slug)
    if not _d:
        print(f"⚠ {_slug} に wheel が無い")
        continue
    _n = len(_glob.glob(os.path.join(_d, "*.whl")))
    print(f"📦 ローカル wheelhouse: {_slug}（{_n} wheel）{_d}")
    _wh_dirs.append(_d)

if _wh_dirs:
    _pip_extra = ["--no-index"]
    for _d in _wh_dirs:
        _pip_extra += ["--find-links", _d]
else:
    print("ℹ wheelhouse Dataset なし → PyPI から取得（5〜10分）")
    # 「Dataset を付け忘れた」と「マウント先が想定と違う」を切り分ける。
    # これが無いと、metadata を確認するために 1 往復（15 分）を使うことになる。
    _seen = []
    for _pat in ("/kaggle/input/*", "/kaggle/input/datasets/*/*"):
        _seen += sorted(_p for _p in _glob.glob(_pat) if os.path.isdir(_p))
    for _p in _seen[:12]:
        print(f"    {_p}")

# ---- warm cache（任意 Dataset）----
# torch.compile の AOT 成果物と Triton JIT を Dataset から復元する。
# コンテナは毎回まっさらなので、これがあると compile 30〜60s を省ける。
# キャッシュキーが違えば単にミスするだけ（壊れない）。
#
# 復元の ON/OFF を外から切り替えられるようにしてある（`KC_WARM_CACHE=0`）。
# Dataset を付けたまま「復元しない」比較を playground で取りたいため。
# これが無いと、Dataset を付けた瞬間に A/B の対照群が消える。
#
# 3 値にしてある:
#   "0"       … 復元しない（対照群）
#   "1"       … あれば復元。無ければ「なし」と出して続行（本番の既定）
#   "require" … **無ければ失敗**。実験用
#
# `require` が要る理由: `1` のままだと Dataset を付け忘れた実験が
# **baseline とまったく同じ結果を出し**、「キャッシュは効かない」という
# 偽陰性になる。1 往復（約 40 分）を捨てるより、その場で落ちたほうが安い。
# 実測 2026-11-01: `vllm-warm-cache` はまだ Kaggle に存在せず、playground の
# push が `not valid dataset sources` で落としていた（＝気づかないまま走る）。
_KC_WC_MODE = os.environ.get("KC_WARM_CACHE", "1").strip().lower()
_USE_WARM_CACHE = _KC_WC_MODE not in ("0", "off", "no", "false")
_WC_REQUIRED = _KC_WC_MODE in ("require", "required")
# 復元の結果は `timings` に残す。**print だけだと 1 回の実行から読み取れない**
# （ログは流れるが、`cli timings` で横並びにできない）。
_WC_INFO = {"mode": _KC_WC_MODE, "restored": False, "source": "", "tar": "",
            "mib": 0.0, "entries": 0, "seconds": 0.0, "missing": []}
# 「完全な tar か」の判定に**件数を使わない**。件数は Triton の版で変わるし、
# しきい値を置くと「正当な tar を弾く」ほうに倒れる。代わりに**名前で見る**。
#
# 実測 2026-11-01（`vllm-playground` の 2 本の tar を数えた）:
#   不完全: 30 エントリ / 70 KB   … profile run に到達していない
#   完全  : 2,077 エントリ / 195 MB / カーネル名 140 種
# 差は GDN（gated delta net）の chunked カーネル群が入っているか。
# 373 秒の主役はこの 2 つ（名前は vLLM の fla 実装にハードコードされている）:
#   chunk_fwd_kernel_o                                 x252
#   chunk_gated_delta_rule_fwd_kernel_h_blockdim64      x84
# **`chunk_o` / `chunk_delta_h` で探してはいけない**。実際のファイル名は
# 上のように長く、部分一致しないので「完全な tar を不完全と誤判定」する
# （最初にこの 2 つで書いて、v4 の tar を弾きかけた）。
# 末尾の `_blockdim64` は head dim を名前に含めたもので、版が変わると
# 変わりうる。だから**前側だけ**を条件にする。
_WC_REQUIRED_KERNELS = ("chunk_fwd_kernel_o", "chunk_gated_delta_rule_fwd_kernel_h")
_wc = kc_find_dataset("vllm-warm-cache")
# `kc_find_dataset` は**見つからなくても既定パスを返す**（`/kaggle/input/<slug>`）。
# そのため `if _wc:` だけでは常に真になり、tar が 1 本も無いときに**何も
# 表示しないまま素通り**していた（実測 2026-11-01: ログに warm cache の行が
# 1 行も出ず、効いているのか判別できなかった）。
#
# **Kaggle は Dataset に上げた `.tar` を展開して保持する。** 実測 2026-11-01:
# 195 MB の tar を `datasets create` したら、マウント側は
# `root/.triton/cache/<hash>/...` の木になっていて `*.tar` は 1 本も無かった。
# 「tar が無い＝キャッシュが無い」と読むと**正当な Dataset を落とす**ので、
# tar と展開済みの木の両方を受ける。
def _kc_wc_tree(root):
    """Dataset の中の Triton キャッシュのディレクトリ（無ければ ""）。"""
    if not root or not os.path.isdir(root):
        return ""
    base = root.rstrip(os.sep).count(os.sep)
    for _dp, _dn, _fs in os.walk(root):
        if (os.path.basename(_dp) == "cache"
                and os.path.basename(os.path.dirname(_dp)) == ".triton"):
            return _dp
        if _dp.count(os.sep) - base >= 4:      # これ以上は潜らない
            _dn[:] = []
    return ""


_tars = [] if not _USE_WARM_CACHE else (
    _glob.glob(os.path.join(_wc, "*.tar"))
    + _glob.glob(os.path.join(_wc, "*", "*.tar")))
_tree = "" if not _USE_WARM_CACHE else _kc_wc_tree(_wc)
if not _USE_WARM_CACHE:
    print("ℹ warm cache: KC_WARM_CACHE=0 なので復元しない（A/B の対照）")
elif _tars or _tree:
    import shutil as _sh2
    import tarfile as _tf2
    # 復元先を**数える**。`extractall("/")` は tar の arcname 次第で
    # どこへでも落ちるので、サイズだけ出していると「復元したのに 1 件も
    # 当たらない」を 1 往復使って調べることになる。
    # Triton は 1 カーネル 1 ディレクトリなので直下の数で足りる。
    _tc = os.path.join(os.path.expanduser("~"), ".triton", "cache")
    _t0 = __import__("time").time()
    # tar があれば tar（展開済みの木は見ない＝二重にコピーしない）。
    _srcs = list(_tars) if _tars else [_tree]
    for _s in _srcs:
        try:
            if os.path.isdir(_s):
                _mib = 0.0
                for _r, _d, _fs in os.walk(_s):
                    for _fn in _fs:
                        try:
                            _mib += os.path.getsize(os.path.join(_r, _fn))
                        except OSError:
                            pass
                _mib = round(_mib / 1e6, 1)
                os.makedirs(_tc, exist_ok=True)
                _sh2.copytree(_s, _tc, dirs_exist_ok=True)
                _src = _s
            else:
                _mib = round(os.path.getsize(_s) / 1e6, 1)
                with _tf2.open(_s) as _f:
                    _f.extractall("/")
                _src = os.path.basename(_s)
            _n = len(os.listdir(_tc)) if os.path.isdir(_tc) else 0
            # 入っているカーネル名を集める。Triton は 1 カーネル 1 ディレクトリで、
            # 中に `<カーネル名>.json` / `.ptx` などを置く。名前で見るのは
            # 件数より意味がはっきりしているため（件数は Triton の版で変わる）。
            _names = []
            for _r, _d, _fs in os.walk(_tc):
                for _fn in _fs:
                    _names.append(_fn.lower())
            _missing = [k for k in _WC_REQUIRED_KERNELS
                        if not any(k in s for s in _names)]
            _WC_INFO.update(restored=True, source=_src, tar=_src, mib=_mib,
                            entries=_n, missing=_missing,
                            seconds=round(__import__("time").time() - _t0, 1))
            print(f"🧠 キャッシュ復元: {_src} {_mib:.0f} MB "
                  f"({_WC_INFO['seconds']}s) / Triton エントリ {_n}（{_tc}）"
                  + (f" / ⚠ 欠け: {', '.join(_missing)}" if _missing else ""))
        except Exception as _e:
            print(f"⚠ キャッシュ復元をスキップ: {_e}")
    # 復元はできたが**中身が薄い**場合。不完全な tar を復元すると一部だけ当たって
    # 「少し効いた」ように見え、373 秒が戻る理由を「キャッシュは効かない」と
    # 誤読する。`require` はその誤読を防ぐためのモードなので、ここも落とす。
    if _WC_REQUIRED and not _WC_INFO["restored"]:
        raise RuntimeError(
            f"KC_WARM_CACHE=require なのに復元できなかった: {_srcs}。"
            "抽出／コピーで例外が出ている（上の ⚠ を参照）。")
    if _WC_REQUIRED and _WC_INFO["missing"]:
        raise RuntimeError(
            f"KC_WARM_CACHE=require なのに "
            f"{', '.join(_WC_INFO['missing'])} が入っていない: "
            f"{_WC_INFO['source']} {_WC_INFO['mib']} MB / "
            f"エントリ {_WC_INFO['entries']}。"
            "この 2 つは 373 秒の主役（profile run の GDN Triton JIT）で、"
            "無いということは**起動に成功していない回**のキャッシュということ。"
            "不完全なまま測ると『キャッシュは効かない』という偽陰性になる。"
            "`KC_EXPERIMENT=baseline` の成果物から採り直すこと"
            "（判断できる場合は KC_WARM_CACHE=1 でこの検査を外せる）。")
elif _WC_REQUIRED:
    # ここで落とす。続行すると baseline と同じ数字が出て、効かなかったのか
    # Dataset が無かったのか区別できないログだけが残る。
    raise RuntimeError(
        f"KC_WARM_CACHE=require なのに tar も展開済みの木も無い: {_wc}"
        f"（exists={os.path.isdir(_wc)}）。"
        "Dataset `vllm-warm-cache` が未作成か、空のまま push している。"
        "このまま走らせると baseline と同じ結果になり、"
        "「キャッシュは効かない」という偽陰性になるので中断する。")
else:
    # 「Dataset を付け忘れ」と「中身が入っていない」を切り分ける。
    print(f"ℹ warm cache なし（compile をやり直す）: {_wc}"
          f" exists={os.path.isdir(_wc)}")

# ---- torch を「消してから入れる」ための道具 ----
# `pip uninstall torch` では足りない（実測 v7）。uninstall は RECORD に載っている
# ファイルしか消さないので、**新しい版で削除されたファイル**は残る。結果、
# 2.11 の `torch/utils/_config_module.py`（`deprecated=` を知らない Config）と
# 2.13 の `torch/_dynamo/config.py`（`Config(deprecated=True)` を呼ぶ）が同居し、
#   TypeError: Config() got an unexpected keyword argument 'deprecated'
# になる。torch 自身のファイル同士が食い違っているので、版を揃え直す以外に直せない。
def _site_dirs():
    r = subprocess.run(
        [sys.executable, "-c",
         "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, timeout=120)
    d = (r.stdout or "").strip()
    return [d] if d and os.path.isdir(d) else []


def _purge_torch():
    """site-packages の torch / torchvision / torchaudio 系を物理削除する。

    `torch-*` はハイフン付きなので torchmetrics / torchcodec を巻き込まない
    （巻き込むと無関係なパッケージまで壊す）。
    """
    n = 0
    for _d in _site_dirs():
        for _pat in ("torch", "torch-*", "torchvision", "torchvision-*",
                     "torchaudio", "torchaudio-*", "torchgen", "torchgen-*",
                     "functorch", "functorch-*"):
            for _p in _glob.glob(os.path.join(_d, _pat)):
                try:
                    if os.path.isdir(_p):
                        shutil.rmtree(_p, ignore_errors=True)
                    else:
                        os.remove(_p)
                    n += 1
                except OSError as _e:
                    print(f"⚠ 削除できず: {_p} ({_e})")
    return n


def _torch_diag():
    """torch が混ざっているかを 1 画面で分かる形で出す。

    これが無いと「import に失敗」だけ見て 6 分を無駄にし、原因を推測する羽目になる。
    `_config_module.py` に `deprecated` が無ければ、そちらが旧版の残骸。
    """
    out = []
    for _d in _site_dirs():
        for _p in sorted(_glob.glob(os.path.join(_d, "torch*"))):
            out.append(f"  {os.path.basename(_p)}")
        _cm = os.path.join(_d, "torch", "utils", "_config_module.py")
        if os.path.isfile(_cm):
            _src = open(_cm, encoding="utf-8", errors="replace").read()
            out.append(f"  _config_module.py {len(_src)}B "
                       f"deprecated={'deprecated' in _src}")
        else:
            out.append("  _config_module.py なし")
    return "\n".join(out) or "  (site-packages が見つからない)"


def _probe(code, timeout=900):
    """別プロセスで import を試す。親で import すると半端な状態が尾を引く。"""
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-1200:]


_r = subprocess.run([sys.executable, "-c", "import vllm"],
                    capture_output=True, text=True)
if _r.returncode == 0:
    print("✅ vLLM は既にインストール済み")
else:
    # ドライバと現行 torch を先に記録する。CUDA 13 系の wheel を入れると
    # `libcudart.so.13` 不在や「driver version is insufficient」で落ちるので、
    # 失敗したときに原因を 1 往復で特定できるようにしておく。
    _drv = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True)
    _tv = subprocess.run(
        [sys.executable, "-c", "import torch;print(torch.__version__, torch.version.cuda)"],
        capture_output=True, text=True)
    print(f"🧩 driver: {(_drv.stdout or '').strip().splitlines()[:1]} / "
          f"torch: {(_tv.stdout or '').strip()}")
    print(f"⬇️  vLLM をインストール中: {VLLM_PIP_SPEC}（5〜10分）")
    # 依存も一緒に入れる（`--no-deps` を付けない）。
    # 理由: vLLM 0.32.0 は CUDA 13 ビルドで、イメージの torch 2.11.0+cu128 のままでは
    # `ImportError: libcudart.so.13` で起動できない（実測 2026-10-31）。torch 2.13.0 と
    # nvidia cu13 一式まで入れて初めて噛み合う。ドライバは 580.178.04 で CUDA 13 対応済み。
    #
    # ただし**上書きインストールは不可**。2.11 のファイルが残って 2.13 と混ざり、
    # `TypeError: Config() got an unexpected keyword argument 'deprecated'`
    # （torch._dynamo）のような内部不整合になる（実測）。しかも pip uninstall では
    # 消し切れない（新版で削除されたファイルが残る）ので、物理削除する。
    print("🧹 site-packages の torch 系を物理削除してから入れる")
    print(f"   {_purge_torch()} 個のパスを削除")

    # ---- インストーラの選択（実測で決めた）----
    # CPU ベンチ `vllm-install-bench`（2026-11-01、`pip install --target` で同じ
    # 190 wheel を 4 通りに展開）:
    #
    #   pip（暖）        200.1s   81,844 files
    #   pip --no-compile 112.6s   56,416 files
    #   uv               35.7s   56,812 files
    #
    # 差の大半は **pyc の生成（25,428 個）と pip の直列展開**。転送ではない
    # （wheelhouse を付けても pip は 199.6 → 230.4s と遅くなった）。
    #
    # ⚠ **この比は本番に移らなかった**（実測 2026-11-01 / v12）。
    #   bench は `--target` の**まっさらなディレクトリ**へ入れる。本番は
    #   torch を物理削除した site-packages へ入れる。同じ wheel 同じホストでも
    #   pip 261.6s → **uv 1151.7s（4.4 倍遅い）**。理由は未同定（uv のキャッシュ
    #   経路が GCSFUSE 上の wheel を引き直している疑い）。**既定は pip**。
    #   uv を試すときは `KC_INSTALLER=uv` を明示した回だけにする（既定を壊さない）。
    _want_uv = os.environ.get("KC_INSTALLER", "").strip().lower() == "uv"
    _uv = None
    if _want_uv:
        _uv = shutil.which("uv")
        if not _uv:
            _ru = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                  "--no-input", "uv"],
                                 capture_output=True, text=True, timeout=600)
            _uv = shutil.which("uv")
            print(f"⬇️  uv を用意: {_uv}（rc={_ru.returncode}）")
        if not _uv:
            print("⚠ uv を用意できない → pip で入れる")
    _installer = "pip"
    _p = None
    if _uv:
        _uv_cmd = ([_uv, "pip", "install", "--python", sys.executable,
                    "--quiet"] + _pip_extra + [VLLM_PIP_SPEC])
        print(f"⬇️  vLLM を uv でインストール: {VLLM_PIP_SPEC}"
              f"（wheelhouse {len(_wh_dirs)} 個）")
        try:
            _p = subprocess.run(_uv_cmd, capture_output=True, text=True,
                                timeout=1800)
        except subprocess.TimeoutExpired:
            print("⚠ uv が 1800 秒で終わらない → pip に切り替え")
            _p = None
        if _p is not None:
            print(((_p.stdout or "") + (_p.stderr or ""))[-1500:])
            # uv は entry point を作るが、`--no-index` で解決が欠けたときに
            # **部分的に成功**して rc=0 を返すことがある。`vllm` CLI の実在まで
            # 確かめる（無いと serve が始まらない）。
            if _p.returncode == 0 and shutil.which("vllm"):
                _installer = "uv"
            else:
                print(f"⚠ uv が不完全（rc={_p.returncode}, "
                      f"vllm={shutil.which('vllm')}）→ pip で入れ直す")
    if _installer != "uv":
        _p = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
            + _pip_extra + [VLLM_PIP_SPEC],
            capture_output=True, text=True, timeout=3600,
        )
        print((_p.stdout or "")[-2000:])
        print((_p.stderr or "")[-2000:])
        if _p.returncode != 0 and _pip_extra:
            # wheelhouse の版ずれ等で解決できないことがある（実測: cublas 13.1.1.3）。
            # ここで止めると 1 回分の GPU を捨てるので、ネットワークに切り替えて再試行。
            print("⚠ ローカル wheelhouse で解決失敗 → PyPI にフォールバック")
            _pip_extra = []
            _p = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                 VLLM_PIP_SPEC],
                capture_output=True, text=True, timeout=3600,
            )
            print((_p.stdout or "")[-2000:])
            print((_p.stderr or "")[-2000:])
        if _p.returncode != 0:
            fail("vLLM のインストールに失敗しました（上のログ参照）")
    print(f"   install 手段: {_installer}")
    _mark("pip")

    # 入れた直後に別プロセスで確かめる。ここで落ちるなら torch はまだ混ざっている。
    # torchaudio の調整に進んでも直らない種類の失敗なので、構成を出して**即座に**止める
    # （先へ進むと 5 分以上無駄にする）。
    _ok_imp, _imp_out = _probe(
        "import torch, vllm;print(torch.__version__, torch.version.cuda, "
        "torch.__file__);print('vllm', vllm.__version__)")
    if not _ok_imp:
        print("⚠ インストール直後の import に失敗。torch の構成を出す:")
        print(_torch_diag())
        print(_imp_out)
        fail("vLLM の import に失敗しました（torch の構成は上記）")

# torchaudio は transformers が import する。vLLM の pip install で torch だけが
# 差し替わると CUDA 版数が不一致になり、vllm CLI が即死する
# （実測: vllm 0.32.0 / torch 2.13.0+cu130 に対し torchaudio 2.8.3+cu128）。
#
# 検出は「実際に落ちる経路」で行う。`import vllm.entrypoints.cli.main` だけでは
# 不十分だった（v2 実測）: benchmark.main 等は main() の中で遅延 import される
# ため、深い依存（transformers → torchaudio）の不整合を見逃す。
# torchaudio の整合は serve より前に単独で確認する（v2 の真因）。
# PyPI の torchaudio 2.11.0 は +cu130 ビルド（wheel の version.py で実測）なので、
# torch 2.13.0+cu130 とそのまま一致する。
def _vllm_pin(name):
    """vLLM の METADATA が要求する版を取り出す（例: 'torchaudio==2.11.0'）。

    torchaudio の版は **torch の版ではない**。vllm 0.32.0 は torch==2.13.0 と
    同時に torchaudio==2.11.0 を要求する（METADATA 実測）。torch に合わせて
    2.13.0 を探すと存在せず、1 往復無駄にする。
    """
    r = subprocess.run(
        [sys.executable, "-c",
         "import importlib.metadata as m;print(chr(10).join(m.requires('vllm') or []))"],
        capture_output=True, text=True, timeout=120)
    for _ln in (r.stdout or "").splitlines():
        _ln = _ln.split(";")[0].strip()          # 環境マーカーは落とす
        if _ln.lower().startswith(name + "=="):
            return _ln
    return None


# 版が同じでも**ビルド**が違うと落ちる。pip は `2.11.0+cu128` を `==2.11.0` の
# 充足とみなす（PEP 440 ではローカル版を無視する）ので、イメージ残りの +cu128 が
# そのまま残り「PyTorch has CUDA 13.0 whereas TorchAudio has 12.8」になる（実測）。
# PyPI の torchaudio 2.11.0 は +cu130 ビルドなので、--force-reinstall で置き換える。
_TA_SPEC = _vllm_pin("torchaudio") or "torchaudio==2.11.0"


def _ta_check():
    r = subprocess.run([sys.executable, "-c", "import torchaudio"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-900:]


_ta_good, _ta_err = _ta_check()
if not _ta_good:
    print(f"⚠ torchaudio 不整合 → {_TA_SPEC} を --no-deps で入れ直す")
    print(_ta_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                   + _pip_extra
                   + ["--no-deps", "--force-reinstall", _TA_SPEC],
                   capture_output=True, text=True, timeout=1800)
    _ta_good, _ta_err = _ta_check()
if not _ta_good:
    # vLLM が torchaudio を使うのは音声モデルだけ（midashenglm/audio.py 等）。
    # このモデルは音声不要なので、外すのが最終手段として妥当。
    print("⚠ まだ不整合 → torchaudio を削除")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ta_good, _ta_err = _ta_check()
print(f"torchaudio import: {'OK' if _ta_good else 'NG'}")
_mark("torchaudio")

_PREFLIGHT_CMD = ["vllm", "serve", "--help"]


def _cli_ok():
    r = subprocess.run(_PREFLIGHT_CMD, capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    bad = ("Traceback" in out) or ("RuntimeError" in out) or len(out) < 500
    return (r.returncode == 0 and not bad), out[-1500:]


_ok, _err = _cli_ok()
if not _ok:
    # transformers 側は ImportError を握るので、外すのが最短（通信不要）。
    # vLLM 0.32 は torchaudio==2.11.0 を要求する（METADATA 実測）。
    # 版を揃えて入れ直すのが本筋。駄目なら外す（transformers は ImportError を握る）。
    print(f"⚠ vLLM CLI が起動しない → torchaudio を {_TA_SPEC} で入れ直す")
    print(_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                   + _pip_extra
                   + ["--no-deps", "--force-reinstall", _TA_SPEC],
                   capture_output=True, text=True, timeout=1800)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を外して再試行")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を torch に合わせて入れ直す")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "-U", "torchaudio"], capture_output=True, text=True,
                   timeout=3600)
    _ok, _err = _cli_ok()
if not _ok:
    # help が壊れていても serve が壊れているとは限らない（実測: CPU では
    # transformers ではなく arg_utils 側で別の失敗をする）。ここで止めず、
    # 起動を試す。壊れていれば最初の試行が即座に落ちるだけで済む。
    print("⚠ vLLM CLI の確認に失敗 → そのまま起動を試す")
    print(_err)
_mark("cli_help")

try:
    import torch
    import vllm
except Exception as _e:
    # import 失敗は「torch の入れ替わり」で起きやすい。原因を残して止める。
    fail(f"vllm/torch の import に失敗: {type(_e).__name__}: {_e}")
print(f"✅ vLLM {vllm.__version__} / torch {torch.__version__} "
      f"/ CUDA {torch.version.cuda}")
print(f"   GPU 認識: {torch.cuda.device_count()} 枚")
_mark("import")

# 比較可能な形で残す。`install_seconds` はテンプレートの timings が読む。
# セルから data を直接触らない理由は kc_save_timings の説明を参照。
#
# `torch_after_install` を別名で残すのは、result.json の `torch` が
# **install 前**（イメージ同梱版）の値だから。実測 2026-11-01 は
# `torch: 2.11.0+cu128`（イメージ）→ `torch 2.13.0+cu130`（実際に動いた版）で、
# 表だけ見ると「2.11 で動いた」と誤読する。
kc_save_timings({"install_seconds": _MARKS["import"],
                 "install_marks": dict(_MARKS),
                 "installer": _installer,
                 "torch_after_install": torch.__version__,
                 "cuda_after_install": torch.version.cuda,
                 "wheelhouse": {"dirs": len(_wh_dirs),
                                "no_index": bool(_pip_extra)},
                 # 復元の有無と**エントリ数**。`entries` が小さいまま起動が
                 # 速くならなければ「効いていない」ではなく「tar が不完全」。
                 "warm_cache_restore": dict(_WC_INFO),
                 "vllm_version": vllm.__version__})
print(f"⏱ install 内訳: {_MARKS}")

# ---- プロキシ + UI ----
with open(PROXY_PATH, "wb") as f:
    f.write(base64.b64decode(PROXY_SRC_B64))
with open(CHAT_PATH, "wb") as f:
    f.write(base64.b64decode(CHAT_SRC_B64))
print(f"✅ auth_proxy.py: {os.path.getsize(PROXY_PATH):,} B")
print(f"✅ llama-chat.html: {os.path.getsize(CHAT_PATH):,} B")

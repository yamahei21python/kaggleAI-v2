# vLLM 起動 → /v1/models が返るまで待機（失敗時は条件を落として再試行）
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

STAGE = "/kaggle/tmp/vllm-model"
MTP_FILE = "model-mtp.safetensors"

# 起動ログの「節目」。末尾 1 行だけを見ていると、長い工程に入った瞬間から
# 何も分からなくなる。実測 2026-10-31: torch.compile 終了（514s）から CUDA graph
# capture 開始（874s）まで **360 秒**、ログは `shm_broadcast ... 60 seconds` の
# 繰り返しだけで、重みロードの 2 回目なのか autotune なのか特定できなかった。
# 節目を別枠で拾えば「最後に越えた関門」が常に分かる。
_MILESTONES = (
    "Starting to load model", "load_format", "Loading weights took",
    "Model loading took", "init engine", "Starting torch.compile",
    "torch.compile took", "available_kv_cache_memory", "Memory profiling",
    "Capturing CUDA graphs", "Graph capturing finished", "init distributed",
    "Starting vLLM API server", "route",
)

# ログ行頭の時刻（`10-31 13:22:08`）。日付は使わず時刻だけ取り出す。
_TS_RE = re.compile(r"\b\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})\b")


def _phase_marks(log_text, t0_wall):
    """起動ログから「各節目に到達した時刻（起動からの秒）」を取り出す。

    60 秒ポーリングでは graph capture（実測 41 秒）すら分解できない。
    vLLM のログ行には `10-31 13:22:08` 形式の時刻が入っているので、起動時刻と
    突き合わせれば **1 秒精度**になる。これが無いと「360 秒の無音区間」の
    正体を毎回推測することになる（実測 2026-10-31）。

    同じ節目が複数回出たら**最初**を採る。最後を採ると、再試行やサブプロセスの
    重複出力で区間が縮んで見える。取れなかった節目はキーごと欠落させる
    （0 で埋めると「一瞬で終わった」と誤読する）。
    """
    out = {}
    for ln in log_text.splitlines():
        m = _TS_RE.search(ln)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if t + 43200 < t0_wall:      # 0 時をまたいだ
            t += 86400
        for name in _MILESTONES:
            if name in ln and name not in out:
                out[name] = round(t - t0_wall, 1)
    return out


_TS_ONLY = re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")


def _long_gaps(log_text, t0_wall, min_seconds=30, keep=5):
    """時刻付き行の間隔から「無音区間」を抽出する（長い順）。

    vLLM は**長い工程に入ると何も出さない**。実測 2026-11-01: torch.compile
    終了（282s）→ CUDA graph capture 開始（661s）の **379 秒**は
    `No available shared memory broadcast block ... 60 seconds` だけが並ぶ。
    区間の長さは分かっても中身が分からないので、**無音の直前と直後の行**を
    対で残す。次に疑うべき対象はこの 2 行で決まる。

    同じ警告が 60 秒おきに繰り返す場合は 1 本に**連結**する。連結しないと
    60 秒 × 5 本が並び、「381 秒詰まった」ではなく「60 秒が 5 回」に見える。
    判定は「区切りの行が**埋め草**（同じ本文が何度も出る行）かどうか」で行う。
    埋め草は無音の**中身**であって終わりではないので、次の無音と繋げる。
    """
    stamps = []
    for ln in log_text.splitlines():
        m = _TS_RE.search(ln)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if t + 43200 < t0_wall:      # 0 時をまたいだ
            t += 86400
        stamps.append((t - t0_wall, ln))

    def _body(s):
        """時刻を落とした本文（同じ行の繰り返しを判定するため）。"""
        return _TS_ONLY.sub("", s).strip()

    seen = {}
    for _, ln in stamps:
        b = _body(ln)
        seen[b] = seen.get(b, 0) + 1

    gaps = []
    for (ta, la), (tb, lb) in zip(stamps, stamps[1:]):
        if tb - ta < min_seconds:
            continue
        gaps.append({"from": round(ta, 1), "seconds": round(tb - ta, 1),
                     "before": la.strip()[:160], "after": lb.strip()[:160],
                     "_ab": _body(lb)})

    merged = []
    for g in gaps:
        prev = merged[-1] if merged else None
        # 直前の無音が**埋め草で終わっている**なら、無音はまだ続いている。
        if prev is not None and seen.get(prev["_ab"], 0) > 1:
            prev["seconds"] = round(g["from"] + g["seconds"] - prev["from"], 1)
            prev["after"] = g["after"]
            prev["_ab"] = g["_ab"]
        else:
            merged.append(g)
    for g in merged:
        g.pop("_ab", None)
    merged.sort(key=lambda g: -g["seconds"])
    return merged[:keep]


def _phase_spans(marks):
    """主要区間の所要（秒）。片方の節目が無ければ None（0 と区別する）。"""
    pairs = {
        "weights": ("Starting to load model", "Starting torch.compile"),
        "compile": ("Starting torch.compile", "torch.compile took"),
        "after_compile": ("torch.compile took", "Capturing CUDA graphs"),
        "capture": ("Capturing CUDA graphs", "Graph capturing finished"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        if a in marks and b in marks and marks[b] >= marks[a]:
            out[name] = round(marks[b] - marks[a], 1)
        else:
            out[name] = None
    return out



def _stage_model(with_mtp):
    """モデルを symlink で 1 か所にまとめる（19 GB をコピーしない）。

    MTP の draft head (model-mtp.safetensors) は、speculative を切った状態で
    読ませると unexpected key になるため、その場合だけ除外する。
    """
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)          # symlink のみなので実体は消えない
    os.makedirs(STAGE, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(MODEL_SRC)):
        if fn.startswith("."):
            continue
        if fn == MTP_FILE and not with_mtp:
            continue
        src = os.path.join(MODEL_SRC, fn)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(STAGE, fn))
        n += 1
    return n


DRAFT_DIR = "/kaggle/tmp/vllm-draft"
# draft に必要なのは mtp の重み + config 類だけ。残すと `*.safetensors` の glob が
# 本体 17.41 GiB まで拾い、draft が同じだけ読み直す（実測 §27）。
DRAFT_KEEP = ("config", "tokenizer", "chat_template", "generation_config",
              "special_tokens_map", "vocab", "merges", "added_tokens")


def _stage_draft():
    """draft (MTP) 専用の最小ディレクトリを作る。戻り値は symlink 本数。"""
    if not os.path.isfile(os.path.join(MODEL_SRC, MTP_FILE)):
        return 0
    if os.path.isdir(DRAFT_DIR):
        shutil.rmtree(DRAFT_DIR)
    os.makedirs(DRAFT_DIR, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(MODEL_SRC)):
        if fn.startswith("."):
            continue
        if fn != MTP_FILE and not fn.startswith(DRAFT_KEEP):
            continue
        src = os.path.join(MODEL_SRC, fn)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(DRAFT_DIR, fn))
        n += 1
    return n


# ---- 利用可能なフラグを確認（バージョン差の吸収）----
# vLLM 0.32 の `serve --help` は Config Group の一覧しか出さず、
# 個々のフラグは `--help=all` にしか無い（GPU 実測 2026-10-29）。
# 旧実装は 2,134 文字の要約を「完全な help」と誤認し、_has() が全フラグを
# 偽判定 → MTP / LMO / chat-template が全部落ちた。既知フラグの有無で
# 「完全な help か」を判定し、ダメなら VERIFIED_FLAGS に倒す。
def _run_help(arg):
    _p = subprocess.run(["vllm", "serve", arg], capture_output=True,
                        text=True, timeout=180)
    return _p.returncode, (_p.stdout or "") + (_p.stderr or "")


def _looks_complete(text):
    """個々のフラグまで含む help か。要約だけなら False。"""
    return (len(text) > 5000
            and "--tensor-parallel-size" in text
            and "--max-model-len" in text)


_rc, HELP = _run_help("--help=all")
if not _looks_complete(HELP):
    _rc2, _h2 = _run_help("--help")
    if len(_h2) > len(HELP):
        HELP, _rc = _h2, _rc2
# ヘルプ本文を成果物に残す（フラグ判定が外れたときの唯一の手掛かり）
save_output("vllm_help.txt", HELP[:20000])
print(f"  vllm serve --help=all: exit={_rc} {len(HELP)} 文字 "
      f"complete={_looks_complete(HELP)}")
print(HELP[:1200])
HELP_OK = (_rc == 0 and _looks_complete(HELP)
           and "Traceback (most recent call last)" not in HELP)
if not HELP_OK:
    print("⚠ 完全な help が取れない → VERIFIED_FLAGS で判定する")


# vllm 0.32.0 wheel の arg_utils/cli_args を静的に走査して実在を確認したもの。
# （詳細 notes/vllm-cpu-findings.md §22）
VERIFIED_FLAGS = {
    "--trust-remote-code", "--reasoning-parser", "--language-model-only",
    "--limit-mm-per-prompt", "--enable-prefix-caching", "--enforce-eager",
    "--tool-call-parser", "--enable-auto-tool-choice", "--speculative-config",
    "--spec-method", "--kv-cache-dtype", "--enable-log-requests",
    "--chat-template", "--chat-template-content-format", "--default-chat-template-kwargs",
    "--attention-backend", "--max-num-batched-tokens", "--generation-config",
    "--override-generation-config", "--load-format", "--tokenizer",
}


def _has(flag):
    """ヘルプ本文の「行頭オプション」として存在するか。

    単純な部分文字列一致だと説明文中の言及で誤判定するため行頭で見る。
    ヘルプが取れない環境では楽観側（True）に倒さず、wheel を静的に走査して
    実在を確認した VERIFIED_FLAGS だけを許可する。楽観側だと 0.32 で消えた
    フラグを付けて未知引数で即死する（過去に実際に踏んだ）。
    """
    if not HELP_OK:
        # 事前に wheel を静的解析して「実在する」と確認済みのものだけ許可する。
        # ここを楽観（True）にすると、0.32 で消えた --chat-template のような
        # フラグを付けて即死する（実測で該当）。
        return flag in VERIFIED_FLAGS
    return re.search(r"^\s*" + re.escape(flag) + r"\b", HELP,
                     re.MULTILINE) is not None


# 必須フラグ（ここが通らなければ何を試しても無駄）
CORE_ARGS = [
    "vllm", "serve", STAGE,
    "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1",
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", str(TP_SIZE),
    "--dtype", "half",
    # 量子化の指定は試行ごとに変える（下の PLAN の 7 要素目）。
    # このモデルの quantization_config は format="pack-quantized"（= compressed-tensors）
    # であり、awq 固定は誤り。auto（指定なし）を第一候補にする。
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--gpu-memory-utilization", str(GPU_MEM_UTIL),
    "--kv-cache-dtype", KV_CACHE_DTYPE,
]

# 任意フラグ。バージョンによって引数値が違うことがあるので、
# 後半の試行では落として「フラグが原因の失敗」を切り分ける。
OPT_ARGS = []
if _has("--trust-remote-code"):
    OPT_ARGS.append("--trust-remote-code")
# 0.32 では --disable-log-requests が廃止され --enable-log-requests に反転。
# 省略しても害がないので、存在した場合だけ付ける（誤指定で試行を落とさない）。
if _has("--disable-log-requests"):
    OPT_ARGS.append("--disable-log-requests")
if _has("--reasoning-parser"):
    # Qwen3 系の思考タグを reasoning_content に分離する
    OPT_ARGS += ["--reasoning-parser", "qwen3"]
if _has("--enable-auto-tool-choice") and _has("--tool-call-parser"):
    OPT_ARGS += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]

# ---- 試行計画（上から順に）----
# 各要素: (ラベル, speculative config, max-model-len, 追加env, 任意フラグを使うか,
#          追加引数, 量子化引数)
# 量子化は「指定なし（auto）」を第一候補。config.json の quantization_config は
# format="pack-quantized" で、AWQ 形式ではない（cfgdeep 実測）。
SPEC = None
if MTP_ENABLED and _has("--speculative-config"):
    SPEC = {"method": MTP_METHOD, "num_speculative_tokens": MTP_TOKENS}
else:
    print("ℹ --speculative-config 非対応 or MTP=0 → MTP なしで起動")

# draft を最小ディレクトリに向ける（17.41 GiB の二重読みを回避）。
# 失敗しても次の段が draft dir なしで再試行するので、ここは楽観で良い。
SPEC_DIR = None
if SPEC:
    _dn = _stage_draft()
    if _dn:
        SPEC_DIR = {**SPEC, "model": DRAFT_DIR}
        print(f"  draft dir : {DRAFT_DIR}（symlink {_dn} 個）")
    else:
        print("ℹ mtp ファイルが無い → draft は target と同じ dir を使う")

# 注意: 0.32 に V0 エンジンは無く、VLLM_USE_V1 / VLLM_ATTENTION_BACKEND は
# envs.py に定義が無い（= 未知 env として警告されるだけ。設定しても無効）。
# T4(sm75) では FlashAttention backend が capability>=8.0 を要求して弾かれ、
# 既定で TRITON_ATTN が選ばれる（triton_attn.supports_compute_capability=True）。
# よってフォールバック段は「明示的に TRITON_ATTN を指定する」に置き換える。

# 視覚塔を落としてテキスト専用で起動する（全段の既定）。
#  - テキスト配信しかしないので不要。0.86 GiB と mm profiling の VRAM を節約。
#  - 実装は StageMissingLayer で子モジュール登録されないため未使用キー警告も出ない
#    （models/utils.py:756 "Don't register this as a child module in order to
#      avoid missing keys when loading weights"）。
# なお quantization_config.ignore の `model.visual.*` は「prefix 文字列」と
# 完全一致で照合される（should_ignore_layer → is_equal_or_regex_match）。
# 視覚塔の prefix は `model.visual...` なので一致する（keymap の key 比較では
# 0 件だったが、判定に使うのは key ではなく prefix）。`re:.*mtp.*` も同様に
# draft の prefix `mtp.*` に一致する。
LMO = ["--language-model-only"] if _has("--language-model-only") else []
if not LMO:
    print("ℹ --language-model-only 非対応 → 視覚塔ありで起動")

AUTO_Q = []                                    # 指定なし（config から自動判定）
CT_Q = ["--quantization", "compressed-tensors"]  # 明示（pack-quantized）
# --quantization awq / auto は config の compressed-tensors と不一致で
# ModelConfig の検証に落ちる（cfgresolve 実測）。試行段からは外した。

STARTUP_BUDGET = int(os.environ.get("VLLM_STARTUP_BUDGET", "2700"))   # 45 分
ATTEMPT_TIMEOUT = int(os.environ.get("VLLM_ATTEMPT_TIMEOUT", "2700"))  # 1 試行に総予算を渡す
# 10/31 のリセット後（30h）前提の値。install は wheelhouse Dataset で
# 60〜120s になる見込み（未実測）。短い予算で回すときは環境変数で下げる。
# 注意: 残クォータが 30 分未満のときにこの設定で push しないこと。
#
# **なぜ 1800/900 をやめたか（2026-11-01 実測）**
# `baseline` が 2 試行とも「切られた」だけで、壊れてはいなかった:
#   attempt 1 … 900 秒（ATTEMPT_TIMEOUT）で終了、重みロードの途中
#   attempt 2 … 806 秒（残予算でクランプ）で SIGTERM、
#               08:07:53 開始 → 08:14:32 に embed_tokens まで到達、
#               その後 `Loading weights took` が出ないまま打ち切り
# 同じ回の vLLM ログは `Checkpoint size: 17.41 GiB / Available RAM: 5.85 GiB`
# で prefetch を切り、GCSFUSE からの読みが 20 MiB/s 前後まで落ちていた。
# 17.41 GiB ÷ 20 MiB/s ≒ 900 秒。**重みだけで 1 試行の枠を使い切る**ので、
# そのあとの profile run（373 秒）に絶対に届かない。
# 実測の幅は 103 MiB/s（out2g: 18.21 GiB / 181s）〜 20 MiB/s と 5 倍ある。
# 予算は**速い回**に合わせると遅い回が必ず落ちるので、遅い側に合わせる。
#
# `ATTEMPT_TIMEOUT` を総予算と同値にしてあるのは、「遅い」失敗と
# 「非対応」失敗を区別するため。非対応は即座に落ちる（SGLang の sm80 など）ので
# 予算をほとんど食わず、次の段に回せる。遅いだけの試行を 15 分で切ると
# **必ず落ちる**うえ、次段も同じ理由で落ちる。

PLAN = []
# 予算 1500s で 1 試行 600s。MTP を 2 段続けると失敗時に予算を食い潰すので、
# 「MTP+draft dir」→ 失敗したら MTP を一旦あきらめて素の起動を優先する。
if SPEC_DIR:
    PLAN.append(("mtp+dd", SPEC_DIR, MAXLEN_CANDIDATES[0], {}, True, LMO, AUTO_Q))
# ここからは任意フラグを落として最小構成で試す
PLAN.append(("plain", None, MAXLEN_CANDIDATES[0], {}, False, LMO, AUTO_Q))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[1], {}, False, LMO, AUTO_Q))
PLAN.append(("plain", None, 16384, {}, False, LMO, AUTO_Q))
# 2026-10-29 GPU 実測: 32k は torch.compile(61s) 完了後に
# "No available shared memory broadcast block found" を 60s 間隔で出し続け、
# 420s で打ち切った（= ハング or CUDA graph capture/ダミー実行が異常に遅い）。
# eager は graph capture を丸ごと飛ばすので、仮説の切り分けを最優先で入れる。
PLAN.insert(1, ("eager", None, MAXLEN_CANDIDATES[0], {},
                False, LMO + ["--enforce-eager"], AUTO_Q))
# 量子化の明示指定（auto が外した場合）
PLAN.append(("ct", None, 16384, {}, False, LMO, CT_Q))
# attention backend を明示（auto でも TRITON_ATTN になるが、選択を固定する）
PLAN.append(("triton", None, 8192, {}, False,
             LMO + ["--attention-backend", "TRITON_ATTN"], AUTO_Q))
# （eager は上の insert で 2 番目に移した。重複させない）
# MTP の再挑戦（draft dir 無し = 17 GiB を読み直すので最後に置く）
if SPEC:
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[2], {}, True, LMO, AUTO_Q))
# language-model-only が原因で落ちた場合の退避。視覚塔は積むが modality を 0 に
# して実体化させない（get_limit_per_prompt=0 → 同じく StageMissingLayer 経路）。
PLAN.append(("vl0", None, 8192, {}, False,
             ["--limit-mm-per-prompt", '{"image":0,"video":0}'], AUTO_Q))
assert all(len(p) == 7 for p in PLAN), "PLAN の要素数が 7 でない"

# 予算が短いとき（< 900s）は「最も軽い仮説検証」を先頭に持ってくる。
# 実測: 27B の重みロードだけで約 250s、torch.compile が 61s。
# 64K+MTP+グラフを試す余裕は無いので、eager（graph capture 無し）で
# 「そもそも起動できるのか」を先に確定させる。
if STARTUP_BUDGET < 900:
    PLAN.sort(key=lambda _p: 0 if _p[0] == "eager" else 1)
    print("ℹ 起動予算が短い → eager を先頭に並べ替え")


def _gpu_used_mib():
    """全 GPU の使用 VRAM 合計（MiB）。取れなければ -1。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return sum(int(x) for x in re.findall(r"[0-9]+", out))
    except Exception:
        return -1


def _gpu_pids():
    """GPU を掴んでいる PID をドライバに聞く（`ps` より確実）。

    vLLM は `setproctitle` でプロセス名を `VLLM::Worker_TP0` のように書き換える。
    名前で殺しに行くと**綴りを外した瞬間に静かに失敗**する（実測 2026-11-01:
    `VLLM::WorkerTP` は `VLLM::Worker_TP0` に一致せず、18,194 MiB を掴んだまま
    5 試行を無駄にした）。ドライバに PID を聞けば名前は関係ない。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
        return [int(x) for x in re.findall(r"[0-9]+", out)]
    except Exception:
        return []


def _free_vram(limit_mib=2000, tries=18):
    """VRAM が実際に空くまで落とし続け、最後の使用量（MiB）を返す。

    「殺したつもり」で次の試行へ進むと、失敗理由が必ず
    `Free memory on device cuda:0 ... is less than desired GPU memory utilization`
    になり、原因（残骸）ではなく**症状**を追いかけることになる。
    """
    for i in range(tries):
        _u = _gpu_used_mib()
        if 0 <= _u < limit_mib:
            return _u
        if i in (0, 3, 8):        # 毎回ではなく段階的に掃除する
            for _pat in ("VLLM::Worker", "VLLM::EngineCore", "vllm serve",
                         "vllm.entrypoints"):
                subprocess.run(["pkill", "-9", "-f", _pat],
                               capture_output=True)
            for _pid in _gpu_pids():
                if _pid in (os.getpid(), os.getppid()):
                    continue      # 自分自身は殺さない
                try:
                    os.kill(_pid, signal.SIGKILL)
                    print(f"  🔫 GPU を掴んでいる PID {_pid} を停止")
                except OSError:
                    pass
        time.sleep(5)
    return _gpu_used_mib()


def _mem_info():
    """`/proc/meminfo` の主要値（MiB）。**空きだけ見ても原因は分からない。**

    実測 2026-11-01（v12）: 試行開始の 4 分前に `MemAvailable` が 30,315 MiB
    あったのに、vLLM が重みを読む時点では **5.68 GiB** まで落ちていた。
    その 24 GiB が何なのか（page cache か、anon か、/dev/shm か）を切り分けるには
    `Cached` / `Shmem` / `MemFree` を並べて見る必要がある。片方だけでは
    「誰かが食った」のか「キャッシュが張り付いた」のか決められない。
    """
    want = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem")
    out = {}
    try:
        for ln in open("/proc/meminfo", encoding="utf-8"):
            k = ln.split(":", 1)[0]
            if k in want:
                out[k] = int(ln.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def _mem_available_mib():
    """空き RAM（MiB）。重みロードが固まるかの予兆になる。"""
    return _mem_info().get("MemAvailable", -1)


def _mem_line():
    """1 行にまとめた RAM の内訳。無音区間の原因を後から追うための控え。"""
    m = _mem_info()
    if not m:
        return "RAM: 取得できません"
    return (f"RAM 空き {m.get('MemAvailable', -1)} / 空 {m.get('MemFree', -1)}"
            f" / cache {m.get('Cached', -1)} / shm {m.get('Shmem', -1)} MiB"
            f"（全 {m.get('MemTotal', -1)}）")


def _top_rss(n=3):
    """RSS 上位 n 件（`名前:MiB`）。24 GiB を誰が持っているかを名前で見る。"""
    rows = []
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/statm", encoding="utf-8") as f:
                    rss = int(f.read().split()[1]) * 4096 // (1024 * 1024)
                with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                    name = f.read().strip()
            except (OSError, ValueError, IndexError):
                continue
            if rss:
                rows.append((rss, f"{name}:{rss}"))
    except OSError:
        return ""
    rows.sort(reverse=True)
    return " ".join(t for _, t in rows[:n])


# 署名から潰す「揺れる値」。同じ失敗なのに毎回違う文字列が入ると、
# ラダーの打ち切り判定が効かない。
_SIG_NOISE = (
    # 先頭に `\b` を付けない: `psm_3e24306e` の `3` の前は `_`（単語文字）なので
    # 単語境界にならず、付けると**ハッシュが消えない**（実測 2026-11-01）。
    (re.compile(r"[0-9a-f]{6,}"), "<hex>"),            # psm_3e24306e / dist_...
    (re.compile(r"pid=\d+"), "pid=<n>"),
    (re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b"), "<ts>"),
    (re.compile(r"\b\d+\.\d+\b"), "<f>"),
)


def _sig_of(tail, n=3):
    """失敗の署名。**揺れる値を潰してから**比べる。

    実測 2026-11-01: VRAM 未解放という**同一の理由**で 5 回失敗したのに、
    末尾に毎回違う `{'/psm_3e24306e'}` のハッシュが入るため署名が一致せず、
    ラダーが 5 回続行して約 900 秒を捨てた。原因ではなく**偶然**を比べていた。
    """
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()][-n:]
    s = " | ".join(lines)
    for _pat, _rep in _SIG_NOISE:
        s = _pat.sub(_rep, s)
    return s[:400]


def _kill(proc):
    """vLLM のプロセス木ごと落とし、VRAM が実際に空くまで待つ。

    2026-10-29 実測: 試行 1 を terminate した直後に試行 2 を起動したところ、
    125 秒で "Engine core initialization failed. See root cause above."
    で即死した（EngineCore / Worker が残って VRAM を掴んでいた）。
    SIGTERM は API サーバーにしか届かず、孫（Worker_TP*）は残る。
    """
    try:
        _pgid = os.getpgid(proc.pid)
        os.killpg(_pgid, signal.SIGTERM)
        proc.wait(timeout=20)
        os.killpg(_pgid, signal.SIGKILL)   # 残った子を確実に落とす
    except Exception:
        pass
    # /dev/shm の残骸を掃除する。SharedMemory が unlink されずに残ると
    # 次の試行の check_shm_free_space で Engine core が即死する
    # （実測: attempt 2/3 が 125 秒で "Failed core proc(s): {}"、
    #   同時に resource_tracker が /psm_* のリークを警告）。
    _left = 0
    try:
        for _fn in os.listdir("/dev/shm"):
            if _fn.startswith(("vllm", "psm_")):
                try:
                    os.unlink(os.path.join("/dev/shm", _fn))
                    _left += 1
                except Exception:
                    pass
    except Exception:
        pass
    if _left:
        print(f"  🧽 /dev/shm の残骸 {_left} 件を削除")
    # 解放待ち。ここを飛ばすと次の試行の失敗理由が「前のプロセスの残骸」になり、
    # 切り分けが壊れる。**戻り値（実際に空いたか）を呼び出し側で判定する**。
    _u = _free_vram()
    print(f"  🧹 停止後の VRAM 使用量: {_u} MiB")
    return _u


def _models(timeout=8):
    """起動済みなら /v1/models の JSON を返す（未起動なら None）。"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=timeout
        ) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _smoke(attempts=2, timeout=90):
    """起動後に 1 回だけ生成させ、初回 forward が通るか確かめる。

    /v1/models が返るだけでは不十分。量子化カーネル（Marlin/Exllama）や
    GDN の Triton カーネルは **推論して初めて** 落ちることがあり、その場合
    サーバーは 500 を返し続ける。ここで落とせば次の段に進める。
    """
    body = json.dumps({
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 16,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    last = ""
    for i in range(attempts):
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            txt = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            return True, (f"tok={(d.get('usage') or {}).get('completion_tokens')} "
                          f"fin={ch.get('finish_reason')} "
                          f"text={txt[:40]!r}")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            last = f"HTTP {exc.code}: {detail}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        print(f"    smoke {i+1}/{attempts} NG: {last[:200]}")
        time.sleep(3)
    return False, last


proc = None
argv_used = None
started = False
# 起動中の進捗を Gist に流す。Kaggle の UI ログは長大で貼りにくく、
# `kaggle kernels output` は完了まで何も返さないため、外から見える場所が要る。
_GIST_ID = (os.environ.get("GIST_ID") or GIST_ID_DEFAULT)
# 書けたかどうかを数える。**黙って落ちるのが一番高い**——2026-11-01 の
# vllm-playground は Gist を 10/28 から一度も更新しておらず、理由
# （トークン無し / 401 / 404）はログにも result.json にも残っていなかった。
# 外からは「何も起きていない」と区別がつかず、起動待ち 30 分を
# 目視で追うことになる。
_GIST_STATE = {"gist_id": _GIST_ID, "attempts": 0, "ok": 0, "last_error": ""}
_GIST_SAID = [False]


def _gist_status(text):
    """起動状況を Gist（vllm_status.txt）に書く。失敗は**1 回だけ**声に出す。

    毎回出すと 30 秒ごとに同じ行が並んでログが埋まるので、初回だけ警告し、
    以降は `_GIST_STATE` に数える。呼び出し側は戻り値を見ない（進捗書き込みで
    起動を止めない）が、結果は `_gist_summary()` で 1 行にできる。
    """
    _GIST_STATE["attempts"] += 1
    ok, detail = kc_gist_write(_GIST_ID, "vllm_status.txt", text)
    if ok:
        _GIST_STATE["ok"] += 1
        return True
    _GIST_STATE["last_error"] = detail
    if not _GIST_SAID[0]:
        _GIST_SAID[0] = True
        print(f"⚠ Gist 進捗を書けません: {detail}")
        print("   以降は黙って続行し、最後にまとめて出します"
              "（CPU で確かめるには gist-probe カーネル）")
    return False


def _gist_summary():
    """書き込みの成否を 1 行にする。0 件なら**それ自体**が異常。"""
    s = _GIST_STATE
    if s["attempts"] == 0:
        return "ℹ Gist 進捗: 一度も呼ばれていない（待機ループに到達していない）"
    if s["ok"] == 0:
        return (f"⚠ Gist 進捗: 0/{s['attempts']} 成功 — 外から見えていない。"
                f" 理由: {s['last_error']}")
    return f"ℹ Gist 進捗: {s['ok']}/{s['attempts']} 成功"
# GPU クォータは有限。ラダーを全部試すと 1 時間超になり、途中で打ち切られると
# 成果物が残らない。総予算を切って、必ずログを残して終わるようにする。
# クォータ残 39 分（2,395s）で pip install に 5 分前後かかる前提の値。
# 25 分使うと公開できたとしても数分で打ち切られるため短くしてある。
# 2026-10-29 実測: 27B ハイブリッド（GDN/Mamba）を T4 x2 で起動すると、
# torch.compile 61s の後に "No available shared memory broadcast block" が
# 60s 間隔で出続け、420s では足りなかった（ハングか graph capture が異常に遅い）。
# **クォータ 30h のリセット後は 1 試行 15 分で回す**。
# 10/28 の残 39 分では 720/420 に落として使った（その値では足りなかった）。
_attempts = []
_prev_sig = None
_t_start = time.time()
tail = ""          # 予算切れで 1 度も走らなかった場合に備えて初期化
last_log = ""
smoke_fail = ""    # 起動はしたが生成で落ちた場合の HTTP 応答（ログに出ない）
save_output("vllm_attempts.txt", "起動試行ログ（追記保存）\n")

# 空き RAM は「重みロードが固まるか」の予兆になる。実測 2026-11-01:
# `Available RAM: 6.12 GiB` / checkpoint 18.21 GiB のとき、GCSFUSE 上の
# safetensors を mmap で読む工程が **840 秒間 1 行も進まなかった**
# （ログ最終行は `Loading safetensors checkpoint shards: 50%` のまま）。
# ここで気づけないと「起動が遅い」ではなく「原因不明のハング」に見える。
_mem = _mem_available_mib()
print(f"🧠 {_mem_line()}")
print(f"   上位 RSS: {_top_rss(5)}")
if 0 <= _mem < 8192:
    print(f"  ⚠ 空き RAM が少ない（{_mem} MiB < 8192）。重みロードで詰まる可能性。"
          f" 症状は「ログの最終行から無進捗のまま試行上限まで待つ」")


def _weights_gib():
    """重み（*.safetensors）の合計 GiB。prefetch の可否を決める分母になる。"""
    total = 0
    try:
        for fn in os.listdir(MODEL_SRC):
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(MODEL_SRC, fn))
    except OSError:
        pass
    return round(total / 2**30, 2)


def _drop_caches():
    """clean page cache を捨てる。**prefetch を復活させるためだけ**にやる。

    vLLM は「checkpoint が空き RAM の 90% を超える」と auto-prefetch を切り、
    GCSFUSE 上の safetensors を頁単位で mmap する遅い経路に落ちる。v10 は
    同じ手順で 133 秒だったので、差は起動時点の `MemAvailable` だけ。
    `Cached` が空きを圧迫しているなら、ここで捨てれば条件を満たし直せる。

    **ただし Kaggle では書けない**（実測 2026-11-01 / `vllm-io-bench` v2）:
    `OSError: [Errno 30] Read-only file system: '/proc/sys/vm/drop_caches'`。
    それでも呼ぶのは、環境が変わったときに 1 行で気づけるようにするため。
    結果（OK/NG）は必ず表示する。
    """
    before = _mem_info()
    ok = False
    try:
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
            f.write("1\n")          # 1 = page cache のみ（dentry/inode は残す）
        ok = True
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    after = _mem_info()
    return ok, (f"{before.get('MemAvailable', -1)} → "
                f"{after.get('MemAvailable', -1)} MiB"
                f"（cache {before.get('Cached', -1)} → {after.get('Cached', -1)}）")


_WEIGHTS_GIB = _weights_gib()
# vLLM のしきい値（90%）に少し余裕を足す。ここを割ると prefetch が切られる。
_NEED_MIB = int(_WEIGHTS_GIB * 1024 / 0.9 * 1.05) if _WEIGHTS_GIB else 0
print(f"🧮 重み {_WEIGHTS_GIB} GiB / prefetch に必要な空き {_NEED_MIB} MiB")


def _read_range(path, off, nbytes, threads=1, chunk_mib=32):
    """`off` から `nbytes` を `threads` 本の `pread` で読む（先読みの実体）。

    同じ fd を複数スレッドで `pread` するのは安全（読み取り専用・位置指定）。
    戻り値は読めたバイト数（**足りない分は黙って捨てない**。呼び側が比べる）。
    """
    chunk = chunk_mib * 1024 * 1024
    total = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        def _one(o):
            n = min(chunk, off + nbytes - o)
            if n <= 0:
                return 0
            try:
                return len(os.pread(fd, n, o))
            except OSError:
                return 0

        with _pool(max(1, threads)) as ex:
            for got in ex.map(_one, range(off, off + nbytes, chunk)):
                total += got
    finally:
        os.close(fd)
    return total


def _prewarm_weights(threads=1, chunk_mib=32, budget_s=300, batch_mib=512):
    """重みを先に読んで page cache に載せる（**時間予算つき**）。

    実測 2026-11-01（`vllm-io-bench` v2・CPU・GPU 0 秒）:

    | 読み方 | 実測 |
    |---|---|
    | GCSFUSE 1 本（64 MiB バッファ） | **110〜130 MiB/s** |
    | GCSFUSE 2/4/8/16 本（32 MiB チャンク） | **2.2〜2.4 MiB/s** ← 13〜50 倍遅い |
    | page cache に載った後の mmap | **5,677 MiB/s** |

    **並列にすると壊滅的に遅い。** 1 GiB を 8 本で読むと 449 秒（1 本なら 33 秒）。
    v10 の重み 133 秒（140 MiB/s）を「並列の証拠」と読んだのは**誤り**で、
    あれは単に回線が速い回だっただけ（同じ 1 本読みでも 43.8 MiB/s の回がある）。

    → **既定は 1 本**。並列は `KC_PREWARM_THREADS` で明示したときだけ試し、
    必ず 1 本と実測で比べて速い方を採る（決め打ちにしない）。

    `budget_s` を超える見込みなら**始めない**（部分的な先読みは「残りが遅い」
    という症状を防げない）。走り出したあとで速度が崩れた場合も、予算で打ち切る。
    """
    files = []
    try:
        for fn in sorted(os.listdir(MODEL_SRC)):
            if fn.endswith(".safetensors"):
                files.append(os.path.join(MODEL_SRC, fn))
    except OSError:
        return None
    if not files:
        return None

    t0 = time.time()
    big = max(files, key=os.path.getsize)
    big_size = os.path.getsize(big)
    probe_n = min(64 * 2**20, big_size)
    probe_off = max(0, big_size - probe_n)      # 末尾から（本読みと窓をずらす）
    cands = [1]
    if int(threads) > 1:
        cands.append(max(1, int(threads)))
    probe = {}
    for n in cands:
        _t = time.time()
        got = _read_range(big, probe_off, probe_n, threads=n, chunk_mib=chunk_mib)
        _dt = time.time() - _t
        probe[n] = round(got / 2**20 / _dt, 1) if _dt else None
    best = max(probe, key=lambda n: probe[n] or 0)
    rate = probe[best] or 0
    total_bytes = sum(os.path.getsize(p) for p in files)
    projected = (total_bytes / 2**20 / rate) if rate else None

    # 予算に収まらないと分かっているなら**始めない**。部分的な先読みは
    # 「残りが遅い」という症状を防げないので、時間だけ払って何も変わらない。
    # 1.5 倍の余裕は、probe が 64 MiB と短く、本読みの実効速度とずれるため。
    if projected is not None and projected > budget_s * 1.5:
        return {"skipped": "over budget", "probe": probe, "threads": best,
                "mib_s": rate, "projected_s": round(projected, 1),
                "seconds": round(time.time() - t0, 1)}

    total = 0
    aborted = False
    for path in files:
        size = os.path.getsize(path)
        off = 0
        while off < size:
            if time.time() - t0 > budget_s:
                aborted = True
                break
            n = min(batch_mib * 2**20, size - off)
            total += _read_range(path, off, n, threads=best, chunk_mib=chunk_mib)
            off += n
        if aborted:
            break
    dt = time.time() - t0
    return {"bytes": total, "seconds": round(dt, 1),
            "mib_s": round(total / 2**20 / dt, 1) if dt else None,
            "threads": best, "probe": probe, "aborted": aborted,
            "budget_s": budget_s, "projected_s": projected}


def _pool(n):
    """スレッドプール。**I/O 待ちなので GIL は関係ない**（`os.pread` は解放する）。"""
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=n)


def _ram_trace(stop, path, every=5.0):
    """起動中の RAM を 5 秒ごとに残す。**24 GiB が消える瞬間**を捕まえるため。

    実測 2026-11-01（v12）: 試行開始の 4 分前に `MemAvailable` 30,315 MiB、
    vLLM が重みを読む時点で 5.68 GiB。差分 24 GiB の正体が分からないまま
    対策を積むのは順序が逆。5 秒粒度なら「どのプロセスが、いつ、どれだけ」が
    そのまま読める。
    """
    t0 = time.time()
    try:
        with open(path, "a", encoding="utf-8") as f:
            while not stop.is_set():
                m = _mem_info()
                f.write(f"{time.time() - t0:7.1f}s avail={m.get('MemAvailable', -1)}"
                        f" free={m.get('MemFree', -1)} cache={m.get('Cached', -1)}"
                        f" shm={m.get('Shmem', -1)} top={_top_rss(3)}\n")
                f.flush()
                stop.wait(every)
    except OSError:
        pass


# ---- Triton キャッシュ（起動の 54% を占める無音の正体）----
# v14 実測（2026-11-01, T4, eager）の時系列:
#   03:52:25  Using LBNHC KV cache layout            ← 82s 地点
#   ……373 秒の無音（shm_broadcast が 60 秒ごとに 6 回）……
#   03:58:38  Available KV cache memory: 3.45 GiB
#   03:58:38  Warming up Qwen GDN Triton kernels     ← ここは 3 秒
#   03:58:41  init engine (profile, create kv cache, warmup model) took 376.24s
# つまり **名前の付いた warmup は 3 秒**で、コストはその手前
# （`determine_available_memory` の profile run）にある。ram_trace を見ると
# その間ワーカーの RSS が 6,987 → 8,361 MiB と 11 MiB/s で増え続けており、
# ハングではなく**コンパイルしている**。成果物は `~/.triton/cache` に落ちる。
#
# したがって「前回の `~/.triton` を持ち込む」が筋。ただし **キーが違えば黙って
# ミスする**（Triton のキーは Triton の版・対象 sm・ソースハッシュを含む）ので、
# 判定は「ファイルがある」ではなく **「エントリが増えなかった」**にする。
# 増えた数が残っていれば、373 秒が戻ったときに「ミスした」のか「別の原因」かを
# 区別できる（この区別が無いまま対策を積むと、v14 の先読みと同じことになる）。
#
# パスは `TRITON_CACHE_DIR` で**明示する**。既定は Triton の版によって
# `~/.triton/cache` と `~/.cache/triton` のどちらにもなり得て、復元側（01 の
# tar）と食い違うと「復元したのに 1 件も当たらない」になる。
_TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser(
    "~/.triton/cache")


def _uniq_dirs(paths):
    """実体で重複を落とす。`~/.triton` が別名で 2 回入ると tar が二重になる。"""
    out, seen = [], set()
    for p in paths:
        try:
            real = os.path.realpath(p)
        except OSError:
            real = p
        if real not in seen:
            seen.add(real)
            out.append(p)
    return tuple(out)


_TRITON_DIRS = _uniq_dirs((
    os.path.dirname(_TRITON_CACHE.rstrip("/")) or _TRITON_CACHE,
    os.path.expanduser("~/.triton"),
    "/root/.triton",
    "/root/.cache/triton",
))


def _triton_snapshot():
    """Triton キャッシュの「いまの顔ぶれ」を返す（比較用）。

    名前の**集合**を返すのは、再利用されたかを判定するため。「ファイルがある」
    だけでは、前回と別のカーネルをコンパイルした（＝当たっていない）場合を
    区別できない。1 カーネル 1 ディレクトリなので直下の名前で足りる。
    """
    names, where = set(), ""
    cands = (_TRITON_CACHE,) + tuple(os.path.join(x, "cache")
                                     for x in _TRITON_DIRS)
    for d in cands:
        if not os.path.isdir(d):
            continue
        where = where or d
        try:
            names.update(e.name for e in os.scandir(d))
        except OSError:
            pass
    return {"dir": where, "names": names}


def _triton_report(before, after):
    """比較結果を **JSON に書ける形**にする（集合は落とす。timings が壊れる）。"""
    n_before, n_after = len(before["names"]), len(after["names"])
    new = len(after["names"] - before["names"])
    return {"dir": after["dir"] or before["dir"],
            "entries_before": n_before,
            "entries_after": n_after,
            "new_entries": new,
            # 今回使ったエントリのうち、前回からあった割合＝当たった割合。
            "reused_pct": (round(100.0 * (n_after - new) / n_after, 1)
                           if n_after else None)}


def _capture_warm_cache(limit_bytes=3e9):
    """`~/.triton` を tar にして成果物へ置く（次回の Dataset 用）。

    既存の `engine_cleanup()` も同じ tar を作るが、あれはキープアライブの
    **終了時**に走るため、v14 では一度も実行されなかった（ログにキャッシュの
    行が 1 本も無く、`warm-cache.tar` も成果物に無い）。起動が通った時点で
    キャッシュは揃っているので、ここで取る方が確実。

    3 GB を超えたら諦める（転送と容量の無駄）。**失敗しても本処理は止めない**
    — 止めると「キャッシュが取れなかっただけで起動できた回」を失う。
    """
    if os.environ.get("KC_CAPTURE_CACHE", "1") == "0":
        return {"skipped": "KC_CAPTURE_CACHE=0"}
    try:
        import tarfile
        dirs = [d for d in _TRITON_DIRS if os.path.isdir(d)]
        if not dirs:
            return {"skipped": "no triton dir"}
        size = 0
        for d in dirs:
            for root_, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root_, f))
                    except OSError:
                        pass
                if size > limit_bytes:
                    break
            if size > limit_bytes:
                break
        if not size:
            return {"skipped": "empty", "dirs": dirs}
        if size > limit_bytes:
            return {"skipped": "too big", "bytes": size, "dirs": dirs}
        out = os.path.join(OUT_DIRS[0], "warm-cache.tar")
        os.makedirs(OUT_DIRS[0], exist_ok=True)
        t0 = time.time()
        with tarfile.open(out, "w") as tf:
            for d in dirs:
                # 復元側（01 の `extractall("/")`）に合わせて `/` からの相対で入れる。
                tf.add(d, arcname=os.path.relpath(d, "/"))
        return {"path": out, "bytes": os.path.getsize(out),
                "source_bytes": size, "seconds": round(time.time() - t0, 1),
                "dirs": dirs}
    except Exception as e:                      # noqa: BLE001
        print(f"⚠ キャッシュ退避をスキップ: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}


# ---- 先読み（既定 OFF。2026-11-01 に既定を反転した）----
# **先読みは「152.5 秒の純コスト」では済まない。後続の重みロードまで 4.7 倍遅くする。**
#
# 実測の並び（すべてログの時刻差）:
#
#   out2g（10-31, 先読みなし）: 重みロード 約 181s → 起動 642.4s → **31.7 t/s**
#   v14 （11-01, 先読みあり）: 先読み 152.5s → 試行 1 の重みロード **705.00s**
#                              → 921s で打ち切り → eager に落ちて 4.5 t/s
#
# 同じマシン・同じ重み（18.21 GiB）で 181s と 705s。**先読み自身が測った速度は
# 122.2 MiB/s** なので、素直に読めば 18.21 GiB ÷ 122 MiB/s ≈ 150s。181s はこれと
# 一致し、705s（26 MiB/s）は**機械自身の実力の 4.7 倍遅い**。つまり読み出し中に
# 何かが競合している。
#
# 機構（推定・ただし数字は整合する）: 先読みが page cache を 25 GiB まで満たす →
# ワーカーが 8.88 GiB × 2 を anonymous で確保 → カーネルが page cache を 17.8 GiB
# 追い出す（＝先読みした重みが消える）→ 試行は cold read に戻り、しかも reclaim と
# 競合する。v14 の `ram_trace.txt` は `Cached` が 20〜23 GiB のまま推移しており、
# この押し出しと矛盾しない。
#
# したがって既定は **OFF**。`KC_PREWARM=1` で明示したときだけ試す（A/B 用の口は残す）。
_PREWARM = None
if _WEIGHTS_GIB and os.environ.get("KC_PREWARM", "0") != "0":
    # 既定は **1 本**。並列は GCSFUSE で 13〜50 倍遅い（実測 2026-11-01）。
    _pw_threads = int(os.environ.get("KC_PREWARM_THREADS", "1"))
    _pw_budget = int(os.environ.get("KC_PREWARM_BUDGET", "300"))
    print(f"🔥 重みを先に読む（{_WEIGHTS_GIB} GiB / {_pw_threads} 本 / "
          f"予算 {_pw_budget}s）…")
    _PREWARM = _prewarm_weights(threads=_pw_threads, budget_s=_pw_budget)
    if _PREWARM and _PREWARM.get("skipped"):
        print(f"   ⏭ 先読みしない（{_PREWARM['skipped']} / "
              f"{_PREWARM.get('mib_s')} MiB/s / 見込み "
              f"{_PREWARM.get('projected_s')}s / probe {_PREWARM['probe']}）")
    elif _PREWARM:
        print(f"   {_PREWARM['mib_s']} MiB/s / {_PREWARM['seconds']}s"
              f" / {_PREWARM['threads']} 本 / probe {_PREWARM['probe']}"
              f"{' / ⚠ 予算切れ（部分）' if _PREWARM['aborted'] else ''}")
    kc_save_timings({"prewarm": _PREWARM})

# キャッシュの「起動前の顔ぶれ」を控える。試行後に差分を取れば
# **当たったかどうか**が分かる（`new_entries` が多い＝作り直した＝ミス）。
_TRITON_BEFORE = _triton_snapshot()
print(f"🧠 Triton キャッシュ: {len(_TRITON_BEFORE['names'])} エントリ"
      f"（{_TRITON_BEFORE['dir'] or 'なし'} / 環境変数 {_TRITON_CACHE}）")
kc_save_timings({"triton_cache_before": {
    "entries": len(_TRITON_BEFORE["names"]), "dir": _TRITON_BEFORE["dir"]}})

# 先読みは「起動の準備」であって起動試行ではない。予算（STARTUP_BUDGET）は
# 試行に使う。ここを混ぜると、先読みが 400 秒かかった回に試行の時間が
# 残らず「先読みしたせいで起動できなかった」という最悪の形になる。
_t_start = time.time()

# 起動中の RAM を 5 秒ごとに記録する（別スレッド）。ポーリングは 60 秒なので、
# そのままだと「30 GiB → 5.7 GiB」が 1 区間に潰れて誰も特定できない。
RAM_TRACE = "/kaggle/working/ram_trace.txt"
_RAM_STOP = threading.Event()
_RAM_THREAD = threading.Thread(target=_ram_trace, args=(_RAM_STOP, RAM_TRACE),
                               daemon=True)
_RAM_THREAD.start()


for idx, (label, spec, maxlen, env_extra, use_opt, extra_args,
          quant_args) in enumerate(PLAN, 1):
    _elapsed = time.time() - _t_start
    if _elapsed > STARTUP_BUDGET:
        print(f"⏱ 起動予算 {STARTUP_BUDGET}s 超過（{int(_elapsed)}s）"
              f" → 残り {len(PLAN) - idx + 1} 試行を打ち切り")
        break
    n_staged = _stage_model(with_mtp=spec is not None)
    cmd = CORE_ARGS + (OPT_ARGS if use_opt else [])
    cmd += ["--max-model-len", str(maxlen)]
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    cmd += extra_args
    cmd += quant_args
    # tokenizer_config.json に chat_template が無いので明示指定する。
    # 0.32 でも --chat-template は FrontendArgs のフィールドとして実在
    # （entrypoints/launchers/cli_args.py:86。dataclass 経由で自動登録されるため
    #  add_argument の grep では見つからない）。存在しなければ付けない。
    _tpl = os.path.join(STAGE, "chat_template.jinja")
    if os.path.isfile(_tpl) and _has("--chat-template"):
        cmd += ["--chat-template", _tpl]
    log_path = f"/tmp/vllm_{idx}.log"
    last_log = log_path
    # torch.compile / Triton JIT は CPU バウンド。vLLM は既定で 1 スレッドに
    # 落とす（"Reducing Torch threads from 2 to 1 for serving"）ため明示する。
    # `TRITON_CACHE_DIR` も明示する: Triton の既定は版によって
    # `~/.triton/cache` と `~/.cache/triton` のどちらにもなり得て、復元した
    # ディレクトリと食い違うと**1 件も当たらないまま 373 秒が戻る**。
    env_run = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
               "TRITON_CACHE_DIR": _TRITON_CACHE,
               **env_extra}
    print()
    print(f"--- 試行 {idx}/{len(PLAN)}: spec={label} "
          f"max_len={maxlen} env={env_extra or 'default'} "
          f"quant={quant_args[1] if quant_args else 'auto'} "
          f"opt_flags={'on' if use_opt else 'off'} "
          f"staged={n_staged} files ---")
    # 起動の直前に空きを確保し直す。ここを飛ばすと「prefetch 無効の遅い経路」に
    # 入り、重みロードが 800 秒進まないまま試行上限を焼く（実測 2026-11-01）。
    _avail = _mem_available_mib()
    if _NEED_MIB and 0 <= _avail < _NEED_MIB:
        print(f"  🧠 空き {_avail} MiB < 必要 {_NEED_MIB} MiB → prefetch が切られる。"
              f" page cache を捨てて再測定")
        _ok, _detail = _drop_caches()
        print(f"  🧹 drop_caches: {'OK' if _ok else 'NG'} {_detail}")
        _avail = _mem_available_mib()
        if 0 <= _avail < _NEED_MIB:
            print(f"  ⚠ それでも足りない（{_avail} MiB）。重みロードで詰まる可能性。"
                  f" 症状は「tqdm が 50% で止まったまま無進捗」")
    with open(log_path, "w") as log:
        _pre = _gpu_used_mib()
        if _pre > 3000:
            print(f"  ⚠ 起動前の VRAM 使用量 {_pre} MiB（残骸の疑い）→ 掃除")
            # 掃除した**結果**を見る。ここを見ないと、解放できていないまま
            # 5 試行を「Free memory ... is less than desired GPU memory
            # utilization」で焼き捨てる（実測 2026-11-01: 18,194 MiB が
            # 6 試行ぶん残り、約 900 秒を捨てた）。
            _pre = _free_vram()
            print(f"  🧹 掃除後: {_pre} MiB")
            if _pre > 3000:
                fail(f"VRAM が解放されません（{_pre} MiB 使用中）。前の実行の"
                     f"残骸か他プロセスが掴んでいます。ここで止めます"
                     f"（起動しても 0.9 × VRAM を確保できず即死する）。",
                     last_log or log_path)
        # start_new_session で独立したプロセスグループにする（_kill で木ごと落とす）
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env_run,
                                start_new_session=True)
    print(f"🔄 vLLM PID: {proc.pid}  log: {log_path}")

    t0 = time.time()
    # ログの時刻（時:分:秒）と突き合わせるための起動時刻（秒・日内）。
    _lt = time.localtime()
    _t0_wall = _lt.tm_hour * 3600 + _lt.tm_min * 60 + _lt.tm_sec
    # 1 試行の上限は残予算でクランプする（最後の 1 段が予算超過で
    # 中途半端に切られ、ログが残らないのを避ける）。
    _tmo = min(ATTEMPT_TIMEOUT,
               max(45, int(STARTUP_BUDGET - (t0 - _t_start))))
    _t_print = 0.0
    _sz_prev = 0
    _poll_marks = {}
    while time.time() - t0 < _tmo:
        d = _models(timeout=3)
        if d and d.get("data"):
            started = True
            break
        if proc.poll() is not None:
            break
        _now = time.time() - t0
        if _now - _t_print >= 60:
            _t_print = _now
            # 「起動待ち Ns (0 models)」だけでは何で詰まっているか分からない。
            # ログ末尾をそのまま出す（GPU 実行を 1 回無駄にしないため）。
            try:
                _sz = os.path.getsize(log_path)
                _lns = [ln.strip() for ln in
                        open(log_path, errors="replace").read().splitlines()
                        if ln.strip()]
            except Exception:
                _sz, _lns = 0, []
            _last_line = _lns[-1][:170] if _lns else "(ログ空)"
            # 末尾 1 行は警告で埋まりやすい。最後に越えた節目を別に出す。
            _hit, _mark = None, "(節目なし)"
            for _ln in reversed(_lns):
                _h = next((m for m in _MILESTONES if m in _ln), None)
                if _h:
                    _hit, _mark = _h, _ln[:150]
                    break
            if _hit and _hit not in _poll_marks:
                # 60 秒分解能の控え。tqdm の行（graph capture）は**ログに時刻が
                # 無い**ので、ログ時刻だけでは拾えない。ポーリング側で補完する。
                _poll_marks[_hit] = round(_now, 1)
            _grow = _sz - _sz_prev
            _sz_prev = _sz
            _gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip().replace("\n", " ")
            print(f"  ⏳ 起動待ち {int(_now)}s"
                  f"（{len(d.get('data', [])) if d else 0} models）: "
                  f"{_last_line}")
            print(f"     log {_sz // 1024} KiB (+{_grow}) / 節目: {_mark}")
            print(f"     GPU util/mem: {_gpu}")
            # RAM の内訳も毎回残す。重みロードで固まる原因は「空きが少ない」
            # だけでは特定できない（cache なのか anon なのか /dev/shm なのか）。
            print(f"     {_mem_line()} / 上位: {_top_rss()}")
            _gist_status(
                f"attempt {idx}/{len(PLAN)} label={label} maxlen={maxlen}\n"
                f"elapsed={int(_now)}s / limit={int(_tmo)}s\n"
                f"last_log: {_last_line}\n"
                f"milestone: {_mark}\n"
                f"{_mem_line()}\n")
        time.sleep(5)

    smoke_fail = ""
    print(_gist_summary())      # 外から見えていたか（0 件は異常）
    if started:
        # 起動できても初回 forward でカーネルが落ちる領域（sm75 + 量子化 +
        # GDN Triton）があるため、1 回生成させて通ったときだけ成功とする。
        # smoke の待ち時間も総予算から差し引く（予算超過で次段が走らない事故を防ぐ）。
        _remain = int(STARTUP_BUDGET - (time.time() - _t_start))
        ok, detail = _smoke(timeout=max(30, min(90, _remain)))
        if ok:
            argv_used = cmd
            print(f"✅ vLLM 起動+生成 OK（{int(time.time()-t0)}s, 試行 {idx}）"
                  f" {detail}")
            # 起動の内訳を残す（高度化の基礎）。これが無いと毎回「なんとなく
            # 遅い」で終わり、施策の効果を比べられない。
            _load_s = round(time.time() - t0, 1)
            _span = None
            _gaps = []
            _ok_log = ""
            try:
                _ok_log = open(log_path, errors="replace").read()
                _marks = _phase_marks(_ok_log, _t0_wall)
                # ログ時刻（1 秒精度）を優先し、無い節目はポーリング（60 秒）で補う。
                # graph capture は tqdm 行なので時刻が無く、ポーリング側にしか出ない。
                _marks = {**_poll_marks, **_marks}
                _span = _phase_spans(_marks)
                _gaps = _long_gaps(_ok_log, _t0_wall)
                print(f"⏱ 起動の節目: {_marks}")
                print(f"⏱ 区間: {_span}")
                for _g in _gaps:
                    print(f"⏱ 無音 {_g['seconds']}s @{_g['from']}s: "
                          f"「{_g['before'][:90]}」→「{_g['after'][:90]}」")
            except Exception as _e:
                _marks = {"error": f"{type(_e).__name__}: {_e}"}
            # **成功時こそログを残す。** 旧実装は失敗時しか保存しておらず、
            # 「compile 後 379 秒の無音区間」の中身を読めなかった（2026-11-01）。
            # 起動できた回のログは 200 KiB 程度なので、毎回残して損はない。
            if _ok_log:
                save_output(f"vllm_{idx}.log", _ok_log[-3000000:])
            kc_save_timings({"load_seconds": _load_s,
                             "attempt": {"index": idx, "label": label,
                                         "max_model_len": maxlen},
                             "startup_marks": _marks,
                             "startup_spans": _span,
                             "startup_gaps": _gaps})
            # キャッシュが**当たったか**を判定する。「ファイルがある」ではなく
            # 「エントリが増えなかった」。373 秒の無音が戻ったときに、ミスなのか
            # 別の原因なのかをここで切り分けられるようにする。
            _tr = _triton_report(_TRITON_BEFORE, _triton_snapshot())
            print(f"🧠 Triton キャッシュ: {_tr['entries_before']} → "
                  f"{_tr['entries_after']} エントリ"
                  f"（新規 {_tr['new_entries']} / 再利用率 {_tr['reused_pct']}%）")
            if _tr["new_entries"] and not _tr["entries_before"]:
                print("   ℹ 前回のキャッシュが無いので今回は全部作り直し"
                      "（この tar を Dataset 化すれば次回効く）")
            elif (_tr["entries_before"]
                  and _tr["new_entries"] > _tr["entries_before"] * 0.2):
                print("   ⚠ 新規エントリが多い＝キャッシュが当たっていない"
                      "（Triton の版・sm・ソースが前回と違う）")
            kc_save_timings({"triton_cache": _tr})
            break
        print(f"⚠ 起動はしたが生成 NG → 次段へ: {detail[:300]}")
        smoke_fail = detail[:800]
        started = False

    tail = ""
    try:
        _full = open(log_path, errors="replace").read()
    except Exception:
        _full = ""
    tail = _full[-4000:]
    # 末尾だけでは「Engine core initialization failed. See root cause above.」
    # のように**原因が上にある**ケースを読めない（2026-10-29 実測）。
    # 失敗ログは丸ごと成果物に残し、原因らしき行も抜き出して表示する。
    if _full:
        save_output(f"vllm_{idx}.log", _full[-3000000:])
    _keys = ("Traceback", "Error", "error", "OutOfMemory", "OOM",
             "RuntimeError", "assert", "CUDA", "not supported",
             "No available shared memory")
    _sig_lines = [ln.strip() for ln in _full.splitlines()
                  if any(k in ln for k in _keys)]
    if _sig_lines:
        print("❌ 起動失敗。原因候補（ログ内の該当行, 先頭20）:")
        for _l in _sig_lines[:20]:
            print("   | " + _l[:200])
    print("❌ 起動失敗。ログ末尾:")
    print(tail)
    # 途中で GPU クォータが尽きても原因が残るよう、失敗ごとに追記保存する。
    _attempts.append(f"--- 試行 {idx} {label} max_len={maxlen} "
                     f"opt={'on' if use_opt else 'off'} ---\n"
                     + (f"[smoke NG] {smoke_fail}\n" if smoke_fail else "")
                     + tail)
    save_output("vllm_attempts.txt", "\n\n".join(_attempts) + "\n")
    _kill(proc)
    # 同一エラーで落ち続けるなら、フラグを変えても無駄。予算を残して止める。
    # 末尾 3 行を署名にする。最終行だけだと TypeError 等の一般論で一致して
    # 別原因の失敗まで打ち切ってしまう（残りクォータを捨てる損失が大きい）。
    _sig = _sig_of(tail)
    # smoke 失敗はサーバーログ末尾が正常行になるため、署名一致で打ち切ると
    # 別段の失敗まで巻き込む。smoke 失敗時は次段へ必ず進む。
    if _sig and _sig == _prev_sig and not smoke_fail:
        print(f"⛔ 同一エラーで再失敗 → ラダー打ち切り: {_sig}")
        break
    _prev_sig = _sig

# 起動中の RAM 推移を残す（成功・失敗どちらでも）。24 GiB が消える瞬間を
# 捕まえるための唯一の記録で、これが無いと毎回「謎のハング」に戻る。
_RAM_STOP.set()
_RAM_THREAD.join(timeout=10)
try:
    _rt = open(RAM_TRACE, encoding="utf-8").read()
    if _rt.strip():
        save_output("ram_trace.txt", _rt)
        print(f"  📄 ram_trace.txt 保存済（{len(_rt)} B）")
except OSError:
    pass

# ---- 次回用に Triton キャッシュを退避 ----
# 成功・失敗の**合流点**に置く。失敗した回でも、373 秒かけて作ったカーネルは
# ディスクに残っているので、tar にして持ち出せば次回は当たる。既存の
# `engine_cleanup()`（キープアライブ終了時）は v14 で一度も走らなかったため、
# 起動シーケンスの中で確実に通るここで取る。
_WARM_CACHE = _capture_warm_cache()
if _WARM_CACHE.get("path"):
    print(f"  📦 warm-cache.tar {_WARM_CACHE['bytes']/1e6:.0f} MB"
          f"（{_WARM_CACHE['seconds']}s, 元 {_WARM_CACHE['source_bytes']/1e6:.0f} MB）"
          f" → Dataset 化すると次回の JIT を省ける")
elif _WARM_CACHE.get("skipped"):
    print(f"  ⏭ キャッシュ退避なし: {_WARM_CACHE['skipped']}")
kc_save_timings({"warm_cache": _WARM_CACHE})

if not started:
    save_output("vllm_startup.log", tail)
    save_output("vllm_attempts.txt", "\n\n".join(_attempts) + "\n")
    fail("vLLM の起動に失敗しました（vllm_startup.log / vllm_*.log 参照）",
         last_log or "/tmp/vllm_1.log")

# ---- 起動結果を保存 ----
info = _models() or {}
ids = [m.get("id") for m in info.get("data", [])]
max_len = (info.get("data") or [{}])[0].get("max_model_len")
# 共有セル（接続情報）が読む値。設定セルの候補値ではなく実測値で上書きする。
CONTEXT_LENGTH = max_len or CONTEXT_LENGTH
with open("/tmp/vllm_cmd.json", "w") as f:
    json.dump({"argv": argv_used, "spec": bool(SPEC), "models": ids,
               "max_model_len": max_len}, f, indent=2)
print()
print(f"  models        : {ids}")
print(f"  max_model_len : {max_len}")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.strip())
save_output("vllm_cmd.json", json.dumps(
    {"argv": argv_used, "models": ids, "max_model_len": max_len},
    indent=2) + "\n")

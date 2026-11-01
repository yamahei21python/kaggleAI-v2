"""GPU 実行で実際に踏んだ「根本不具合」を静的に固定する。

1 件 1 テスト。**すべて実測に基づく**（発生日と症状を docstring に書く）。
GPU を回さずに「その穴が再び開いたか」を検出するのが目的なので、
測定量そのもの（t/s や所要秒）は検証しない。検証するのは**構造**:

- その防御がコードに存在するか
- 防御の順序が正しいか（後追いになっていないか）
- 防御が「空リスト」「0」「例外」で素通りしないか

症状 → テストの対応表は `notes/pitfall-coverage.md`。
セル内部の契約（順序・版の出どころ）は `tests/test_vllm_cells.py`、
計測は `tests/test_startup_timing.py` に分けてある。
"""
import _cells  # noqa: E402

VLLM_CELLS = _cells.ROOT / "kaggle" / "step2-vllm" / "cells"
STEP2_CELLS = _cells.ROOT / "kaggle" / "step2" / "cells"


def _cell(path):
    return _cells.cell(path) if isinstance(path, str) else path.read_text(
        encoding="utf-8")


# ── #1 重みロードの I/O 律速（/kaggle/input は 82 MiB/s）────────────────
def test_model_staging_uses_symlink_not_copy():
    """19.5 GB をコピーすると I/O で 244s 以上を捨てる。

    `_stage_model` は symlink だけ張る（実体コピー禁止）。コピーに戻すと
    /kaggle/working の 19.5 GiB を埋めたうえで起動が倍遅くなる。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    fn = _cells.slice_def(src, "_stage_model")
    assert "os.symlink" in fn, "ステージングが symlink でない（19.5 GB をコピーしている）"
    assert "shutil.copy" not in fn, "_stage_model が実体をコピーしている"
    assert "shutil.rmtree" in fn, "再ステージ時に古い symlink を消していない"


# ── #2 torchaudio の版不整合（検出タイミングの誤り）──────────────────
def test_torchaudio_is_probed_by_import_before_serve():
    """torch 2.13.0+cu130 と torchaudio 2.8.3+cu128 の混在で serve が死んだ。

    `import vllm` だけでは足りない（transformers → torchaudio は serve 直前の
    遅延 import）。install セルで `import torchaudio` を**単独実行**して確かめる。
    """
    install = _cell(VLLM_CELLS / "01_install_vllm.py")
    assert '"import torchaudio"' in install or "'import torchaudio'" in install, (
        "torchaudio を import で検証していない（serve 直前に落ちる）")
    # 子プロセスで実行すること。親で import すると torch が親に固着する。
    idx = install.index("import torchaudio")
    window = install[max(0, idx - 400):idx]
    assert "subprocess.run" in window, "torchaudio を親プロセスで import している"


def test_torch_purge_covers_siblings_without_touching_torchmetrics():
    """`pip uninstall torch` は RECORD に無い残骸を消せない（deprecated エラー）。

    物理削除のパターンが torchvision / torchaudio / torchgen / functorch まで
    届いていること。かつ `torch-*`（ハイフン）で torchmetrics を巻き込まないこと。
    """
    src = _cell(VLLM_CELLS / "01_install_vllm.py")
    fn = _cells.slice_def(src, "_purge_torch")
    for name in ("torchvision", "torchaudio", "torchgen", "functorch"):
        assert name in fn, f"_purge_torch が {name} を消していない"
    assert '"torch-*"' in fn or "'torch-*'" in fn, "site-packages の torch-*.dist-info を消していない"
    # torchmetrics / torchcodec を巻き込むと無関係な import が壊れる
    assert "torch*" not in fn, "`torch*` の広すぎる glob（torchmetrics を壊す）"


# ── #4 CUDA バックエンドが欠けると「黙って CPU 推論」（最重要）──────
def test_cpu_fallback_detected_when_nvidia_smi_returns_nothing():
    """HTTP 200 は CPU 推論でも返る。実測 0.9 t/s（GPU あり 13.7 t/s）。

    旧実装は `if used and sum(used) < 4000` で、**used が空だと素通り**した。
    nvidia-smi が空 / "N/A" を返すケース（= GPU 不在）を検出できない穴。
    """
    warm = _cell(VLLM_CELLS / "04_warmup.py")
    assert "if not used:" in warm, (
        "nvidia-smi が空のときの判定が無い（CPU 推論を素通りさせる）")
    assert "if used and sum" not in warm, (
        "`if used and sum(...)` は空リストで素通りする（旧バグの再発）")
    assert "CPU 実行の可能性" in warm
    # "N/A" などで int() が落ちないこと
    assert ".isdigit()" in warm, "nvidia-smi の非数値（N/A）で ValueError になる"


def test_verify_checks_gpu_residency():
    """検証項目 [5] に GPU 常駐が入っていること（CPU フォールバック検出）。"""
    checks = _cell(VLLM_CELLS / "06_checks.py")
    assert "gpu-resident" in checks
    assert "memory.used" in checks
    assert "> 4000" in checks, "VRAM の下限判定が無い"


def test_cuda_backend_lib_is_a_hard_failure():
    """libggml-cuda.so が無いと llama-server は黙って CPU に落ちる。

    警告だけで続行すると 0.9 t/s のまま「成功」してしまう。
    """
    src = _cell(STEP2_CELLS / "01_restore_binaries.py")
    assert "libggml-cuda.so" in src
    # 検証対象（_targets）に入っている = 欠ければ fail する
    tail = src[src.index("_targets = ["):]
    assert "CUDA_LIB" in tail, "CUDA バックエンドが検証対象に入っていない"


# ── #5 --kv-cache-dtype fp8 は T4（sm75）非対応 ──────────────────────
def test_kv_cache_dtype_is_auto_and_never_fp8():
    """T4 は FP8 KV cache 非対応。指定すると起動で落ちる。

    CPU プローブで選択肢 19 個を確認済み（`fp8` はあるが sm75 では使えない）。
    既定は `auto` のままにする。
    """
    settings = _cell(VLLM_CELLS / "00_settings.py")
    assert 'KV_CACHE_DTYPE = "auto"' in settings, "KV の既定が auto でない"
    # 起動引数に fp8 を直書きしていないこと（コメント中の言及は許す）
    for cell in VLLM_CELLS.glob("*.py"):
        for ln in _cell(cell).splitlines():
            if "--kv-cache-dtype" in ln:
                assert "fp8" not in ln, f"{cell.name}: --kv-cache-dtype に fp8 を渡している"


# ── #6 32K ctx で CUDA graph capture が詰まる ────────────────────────
def test_ladder_has_eager_step_near_the_front():
    """graph capture を丸ごと飛ばす段が無いと、詰まったときに切り分けられない。

    実測: 32K ctx で "No available shared memory broadcast block" が 60s 間隔。
    eager は 2 番目に置く（1 番目は MTP+draft dir の本命）。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert '"--enforce-eager"' in src, "eager 段がラダーから消えている"
    assert 'PLAN.insert(1, ("eager"' in src, "eager が 2 番目に置かれていない"


# ── #7 プロセスの不完全な後片付け（孫プロセスが VRAM を掴む）──────
def test_engine_stop_kills_the_process_group_and_sweeps_shm():
    """`terminate()` だけだと Worker_TP が残り、次試行が即死する。

    実測: 次試行が 125s で "Engine core initialization failed"。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert "os.killpg" in src, "プロセスグループ単位で落としていない"
    assert "SIGKILL" in src, "SIGTERM だけで諦めている（残る子がいる）"
    assert "/dev/shm" in src, "/dev/shm の残骸を掃除していない"
    assert "start_new_session" in src, "子を別セッションにしていない（killpg が効かない）"


def test_kill_patterns_match_the_real_process_titles():
    """`pkill -f` の綴りを外すと**静かに失敗**し、VRAM を掴んだまま残る。

    実測 2026-11-01: `VLLM::WorkerTP` は実際の `VLLM::Worker_TP0` に一致せず、
    重みロード中に固まった試行 1 の Worker が 2 枚で 18,194 MiB を保持し続けた。
    以降 5 試行が「Free memory ... is less than desired GPU memory utilization」
    で即死し、約 900 秒を捨てた。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert '"VLLM::WorkerTP"' not in src, "アンダースコア落ちの綴りが残っている"
    assert '"VLLM::Worker"' in src, "Worker（TP0/TP1 共通）を落としていない"


def test_vram_is_freed_by_driver_pid_not_by_name():
    """名前ではなくドライバに PID を聞いて落とす（名前は書き換えられる）。

    vLLM は `setproctitle` でプロセス名を変えるので、名前一致は**偶然**に依存する。
    `nvidia-smi --query-compute-apps=pid` なら名前に関係なく特定できる。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    fn = _cells.slice_def(src, "_free_vram")
    assert "--query-compute-apps=pid" in src, "ドライバに PID を聞いていない"
    assert "os.kill(" in fn, "PID を直接落としていない"
    assert "os.getpid()" in fn, "自分自身を殺す可能性がある"
    assert "return _gpu_used_mib()" in fn, "実際に空いたかを返していない"


def test_startup_fails_fast_when_vram_cannot_be_freed():
    """解放できていないのに次の試行へ進むと、症状だけが 5 回並ぶ。

    実測 2026-11-01: 18,194 MiB が残ったまま 6 試行 × 約 150 秒を消費。
    0.9 × VRAM を確保できないので、**起動しても必ず即死する**。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    loop = src.split("for idx, (label, spec, maxlen")[1]
    assert "_free_vram()" in loop, "掃除の**結果**を見ていない"
    assert "VRAM が解放されません" in loop, "解放できないときに止まらない"


def test_low_free_ram_is_reported_before_startup():
    """空き RAM が少ないと重みロードが無進捗のまま固まる（実測 840 秒）。

    `Available RAM: 6.12 GiB` / checkpoint 18.21 GiB / FUSE.GCSFUSE で、
    ログの最終行が `Loading safetensors checkpoint shards: 50%` のまま
    試行上限まで進まなかった。予兆を先に出す。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert "def _mem_available_mib(" in src
    assert "MemAvailable" in src, "/proc/meminfo を読んでいない"
    assert "空き RAM が少ない" in src, "警告を出していない"


def test_ram_breakdown_is_recorded_while_startup_stalls():
    """「空きが少ない」だけでは原因を決められない。内訳を毎回残す。

    実測 2026-11-01（v12）: 試行開始の 4 分前に `MemAvailable` 30,315 MiB
    → vLLM が重みを読む時点で **5.68 GiB**。24 GiB が誰のものか（page cache か
    anon か /dev/shm か）は、空き 1 個の数字からは決められない。原因が分からない
    まま対策（ローカルへ実コピー等）を積むと、効果の無い変更に GPU を溶かす。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert "def _mem_info(" in src
    for key in ("Cached", "Shmem", "MemFree"):
        assert key in src, f"{key} を記録していない（内訳が分からない）"
    assert "def _top_rss(" in src, "RSS 上位を出していない（誰が持っているか不明）"
    # 起動待ちのループの中で毎回出す（成功時だけでは固まった回に何も残らない）。
    loop = src.split("for idx, (label, spec, maxlen")[1]
    assert "_mem_line()" in loop, "起動待ちのループで RAM を出していない"


def test_page_cache_is_dropped_before_startup_when_ram_is_short():
    """prefetch の可否は「空き RAM」で決まる。足りなければ捨てて作り直す。

    vLLM の判定は `checkpoint size exceeds 90% of available RAM`。v10 は同じ
    手順で重み 133 秒（prefetch 有効）、v12 は 800 秒無進捗（無効）。差は
    起動時点の空きだけ。捨てられるなら捨てる方が、待つより常に安い。
    **捨てられなかった場合も記録する**（黙って諦めると原因が消える）。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert "def _drop_caches(" in src
    assert "/proc/sys/vm/drop_caches" in src, "page cache を捨てていない"
    assert "def _weights_gib(" in src, "重みの実サイズを測っていない（分母が無い）"
    loop = src.split("for idx, (label, spec, maxlen")[1]
    assert "_drop_caches()" in loop, "起動の直前に呼んでいない"
    assert "drop_caches: " in src, "結果（効いたか）を出していない"


def test_failure_signature_ignores_volatile_hashes():
    """同じ失敗を「別の失敗」と数えない（揺れる値を潰してから比べる）。

    実測 2026-11-01: VRAM 未解放という**同一の理由**で 5 回失敗したが、末尾に
    毎回違う `{'/psm_3e24306e'}` が入るため署名が一致せず、ラダーが続行して
    約 900 秒を捨てた。原因ではなく偶然を比べていた。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    fn = _cells.slice_def(src, "_sig_of")
    lines = src.splitlines()
    s = next(i for i, ln in enumerate(lines) if ln.startswith("_SIG_NOISE = ("))
    e = s
    while not lines[e].rstrip().endswith(")"):
        e += 1
    ns = {}
    exec("import re\n" + "\n".join(lines[s:e + 1]), ns)         # noqa: S102
    exec(fn, ns)                                               # noqa: S102
    a = ns["_sig_of"]("RuntimeError: boom\n{'/psm_3e24306e'}\nwarnings.warn(")
    b = ns["_sig_of"]("RuntimeError: boom\n{'/psm_d118464f'}\nwarnings.warn(")
    assert a == b, f"ハッシュで署名が変わっている: {a!r} != {b!r}"
    # 原因が違えば別の署名になること（打ち切りすぎない）
    c = ns["_sig_of"]("ValueError: OOM\n{'/psm_3e24306e'}\nwarnings.warn(")
    assert a != c


# ── #8 cloudflared を毎回ダウンロード ───────────────────────────────
def test_cloudflared_prefers_the_dataset_cache():
    """コンテナは毎回真っ新。Dataset から復元できないと毎回 GitHub へ行く。"""
    src = _cell(STEP2_CELLS / "01_restore_binaries.py")
    assert "CF_PATH" in src, "cloudflared 専用 Dataset を見ていない"
    fn = _cells.slice_def(src, "_find_cached")
    assert "CF_PATH" in fn, "キャッシュ探索が cloudflared Dataset を先に見ていない"
    # Dataset 側の構造は flatten されることがあるので 2 段構え
    assert '"bin"' in fn, "flatten された Dataset（bin/ 無し）を考慮していない"


# ── #9 /v1/models が 200 でも初回 forward で落ちる ──────────────────
def test_startup_requires_a_generation_smoke():
    """`/v1/models` は返すが、初回 forward で Marlin/GDN/MTP draft が落ちる。

    16 トークンだけ生成させて通ることを起動成功の条件にする。
    """
    src = _cell(VLLM_CELLS / "03_vllm_serve.py")
    assert '"max_tokens": 16' in src, "起動直後の生成スモークが無い"
    assert "smoke_fail" in src, "生成失敗を次の段に伝えていない"
    assert "初回 forward" in src, "何のためのスモークか書かれていない"


# ── #10 Ollama の並列ブロック（qwen35 blocklist）───────────────────
def test_ollama_parallel_is_pinned_to_one_with_reason():
    """qwen3.5 系は server/sched.go の blocklist で numParallel=1 に固定される。"""
    settings = _cell(STEP2_CELLS / "00_settings.py")
    assert "OLLAMA_NUM_PARALLEL" in settings
    line = next(ln for ln in settings.splitlines() if "OLLAMA_NUM_PARALLEL" in ln)
    assert '"1"' in line, f"並列を上げている（blocklist で無視される）: {line.strip()}"
    assert "blocklist" in settings or "block" in line, "理由が書かれていない"


# ── #11 llama-server は 15 KB の薄いラッパー ───────────────────────
def test_llama_server_size_check_is_relaxed():
    """1 MB 判定だと正常な llama-server（15 KB）を弾く。"""
    src = _cell(STEP2_CELLS / "01_restore_binaries.py")
    assert "MIN_SIZE_LLAMA" in src, "llama-server 専用の閾値が無い"
    val = int(src.split("MIN_SIZE_LLAMA =")[1].split("\n")[0].strip())
    assert val <= 64 * 1024, f"MIN_SIZE_LLAMA={val} は大きすぎる（15 KB を弾く）"
    # 判定箇所が llama-server に MIN_SIZE_LLAMA を使っていること
    assert "MIN_SIZE_LLAMA)" in src or "MIN_SIZE_LLAMA]" in src


# ── #12 /dev/shm 不足で TP>1 が即死 / 境界値で詰まる ────────────────
def test_shm_requirement_scales_with_tp():
    """/dev/shm は TP 本ぶん必要。足りないと起動後に詰まる（40 分待ち）。"""
    settings = _cell(VLLM_CELLS / "00_settings.py")
    fn = _cells.slice_def(settings, "_shm_required_mib")
    ns = {}
    exec(fn, ns)                                   # noqa: S102 - 自前ソースのみ
    need = ns["_shm_required_mib"]
    assert need(1) >= 240, "TP=1 で 240 MiB を下回る見積り"
    assert need(2) == need(1) * 2, "TP に比例していない"
    assert need(0) == need(1), "TP=0 でゼロになると判定が無効化される"


def test_shm_shortage_fails_fast_after_a_remount_attempt():
    """放置すると「起動はするが forward で詰まる」= 待機側が 40 分待つ。"""
    settings = _cell(VLLM_CELLS / "00_settings.py")
    assert "SHM_REQUIRED_MIB" in settings
    assert "remount" in settings, "拡張を試みていない"
    assert "SHM_FREE_MIB < SHM_REQUIRED_MIB" in settings, "空きで判定していない（total では不十分）"
    assert "raise RuntimeError" in settings[settings.index("SHM_REQUIRED_MIB"):], (
        "不足時に落としていない（詰まるまで待ってしまう）")


# ── #13 TP=2 固定（T4 2 枚で 27B AWQ）──────────────────────────────
def test_tensor_parallel_follows_the_gpu_count():
    """16.52 GiB は T4 1 枚（15 GiB）に載らない → 2 枚揃ったときだけ TP=2。"""
    settings = _cell(VLLM_CELLS / "00_settings.py")
    assert "TP_SIZE = 2 if len(_vram_each) >= 2 else 1" in settings
    assert "VRAM_TOTAL == 0" in settings, "GPU 不在の判定が無い"

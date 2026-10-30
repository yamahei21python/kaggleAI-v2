# キープアライブ: 有限ループ（完了させると /kaggle/working の zip を取得できる）
# 無限ループだと実行が完了せず、ログも output も構造的に取れないため有限にしてある。
# 検証中は 5 分、完成後の運用は 480 分（Kaggle の 9h 上限より短い最大値）。
import os, subprocess, time, re, json, urllib.request, urllib.error

# 維持時間は /tmp/keepalive.conf（proxy がチャット送信ごとに書き直す）で決まる。
# MAX_MINUTES はセル開始からの合計上限。延長を重ねてもこれを超えない
# （超えて Kaggle の実行時間上限に達すると成果物ごと消える）。
MAX_MINUTES = 480
TUNNEL_CHECK_EVERY = 10       # 公開URL疎通は 10 分ごと（外部リクエスト削減）

KEEPALIVE_CONF = "/tmp/keepalive.conf"   # ON/OFF + MINUTES の共有ファイル
KEEPALIVE_RESET_MINUTES = 60             # チャット送信ごとにリセットする時間（分）


def _read_keepalive_conf():
    """RUN=1/0 と MINUTES を読み取り（共通ランタイムが上限・既定値を処理）。"""
    run, minutes, _mtime = kc_read_conf(
        KEEPALIVE_CONF, KEEPALIVE_RESET_MINUTES, MAX_MINUTES, fail=fail)
    return run, minutes


def _reset_keepalive():
    """チャット送信ごとに時間リセット（ON 維持）。"""
    try:
        kc_write_conf(KEEPALIVE_CONF, 1, KEEPALIVE_RESET_MINUTES)
    except Exception:
        pass


try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    url = ""


def _proc(name):
    """プロセス生死。pid があるだけで生存とは限らないので tunnel 疎通と併用。"""
    # ollama は -f だと子の `llama-server -m ...ollama_models...` まで
    # 全一致して偽 alive になる。プロセス名完全一致で見る。
    return kc_proc_state(name, "exact" if name == "ollama" else "argv")


def _gpu():
    return kc_gpu_usage()


print(f"Public URL: {url}")
RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()
if RUN:
    print(f"🔋 ON。{KEEPALIVE_RESET_MINUTES}分間維持（チャット送信でリセット、■ で停止）")
else:
    print(f"🛑 ON/OFF OFF（ RUN=0 ）。即時終了します")
    KEEPALIVE_MINUTES = 0
print("完了後: kaggle kernels output <slug> -p <dir> で成果物を取得")
print()

# ログ API は取れないため、毎分の行も**ここに溜めて** zip に残す
def _conf_mtime():
    """keepalive.conf の更新時刻。proxy がチャット送信ごとに書き直す。"""
    return kc_conf_mtime(KEEPALIVE_CONF)


log = []
ollama_st = cf_st = "unknown"
cf_restarted = 0
tunnel_dead = 0
_restarted_now = False
# 1周も回らなかった場合（RUN=0 即時終了）は死活判定をしない。
# ループ未実行だと ollama_st/cf_st が "unknown" のままになり、
# 正常に動いているのに fail() する（実測バグ）。
ran_any = False
i = 0
# for ではなく while にする理由: proxy はチャット送信ごとに
# /tmp/keepalive.conf を MINUTES=60 で書き直す。for の range は開始時に
# 固定されるため、延長が効かない（実測バグ）。conf の mtime 変化で延ばす。
limit = min(KEEPALIVE_MINUTES, MAX_MINUTES)
conf_mtime = _conf_mtime()
while i < limit:
    i += 1
    RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()   # 都度 ON/OFF + 時間確認
    if not RUN:
        print(f"RUN=0 で {i}周目にキープアライブ終了")
        break
    _m = _conf_mtime()
    if _m != conf_mtime:
        conf_mtime = _m
        limit = min(i - 1 + KEEPALIVE_MINUTES, MAX_MINUTES)
        print(f"♻ アクティビティ検出 → 残り {limit - i + 1} 分に延長"
              f"（合計上限 {MAX_MINUTES} 分）")
    ran_any = True
    time.sleep(60)
    ollama_st, cf_st = _proc("ollama"), _proc(TUNNEL_NAME)

    # トンネルが死んだら1回だけ起こす（cloudflared は URL が変わるので
    # 新 URL を取り直して public_url.txt を書き換える。2回目以降は fail 対象）
    # 再起動コマンドと URL 取得はトンネルセルが定義した共通インターフェースを使う。
    if cf_st == "DEAD" and cf_restarted == 0:
        cf_restarted = 1
        try:
            subprocess.run(["pkill", "-x", TUNNEL_NAME], capture_output=True)
            time.sleep(1)
            with open(TUNNEL_LOG, "w") as _lg:
                subprocess.Popen(RESTART_CMD, stdout=_lg, stderr=_lg)
            _new_url = ""
            for _ in range(30):          # 最大30秒で新URL待ち
                time.sleep(1)
                _new_url = probe_url()
                if _new_url:
                    break
            if _new_url:
                url = _new_url
                with open("/tmp/public_url.txt", "w") as _f:
                    _f.write(url + "\n")
                # zip 側も更新しないと、成果物には失効URLが残る
                save_output("public_url.txt", url + "\n")
                print(f"♻ {TUNNEL_NAME} 再起動 → 新URL: {url}")
                # 固定URL配布（Gist）も更新しないと、クライアントは失効URLを掴む
                if "publish_url" in globals():
                    publish_url(url)
                cf_st = "alive(restarted)"
                # 新トンネルは別勘定。エッジ伝播待ち中の /healthz 失敗を
                # 旧カウントに混ぜると、稼働中でも次回チェックで偽FAILになる。
                tunnel_dead = 0
                _restarted_now = True
        except Exception as _e:
            print(f"⚠ {TUNNEL_NAME} 再起動失敗: {type(_e).__name__}: {_e}")

    line = (f"[{time.strftime('%H:%M:%S')}] "
            f"ollama={ollama_st} {TUNNEL_NAME}={cf_st} | GPU: {_gpu()}")

    # pid だけでは tunnel の死は分からない（quick tunnel は URL 失効がありうる）
    # 再起動した同一イテレーションは新URLのエッジ伝播待ちなので判定しない
    if url and i % TUNNEL_CHECK_EVERY == 0 and not _restarted_now:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=15) as r:
                line += " | tunnel=OK" if r.status == 200 else f" | tunnel=HTTP{r.status}"
        except Exception as e:
            tunnel_dead += 1
            line += (f" | tunnel=DEAD({type(e).__name__}) x{tunnel_dead}")
            if cf_st == "alive":
                line += " ← トンネル生存中だが URL 失効の可能性"
    _restarted_now = False

    log.append(line)
    print(line)

# どれか1つでも死んでいれば失敗扱い（tunnel は pid と併せて判定）。
# tunnel は1回の瞬断で全体を FAIL にしない（30分稼働後に成果物が死なないように）。
# 2回以上 = 恒久的な失効とみなし、cf再起動が効かなかった場合もこれで拾う。
# unknown（pgrep 例外/timeout）は確認不能 = 安全側で失敗扱いにする。
# ただし 1 周も回っていない（RUN=0 即時終了）場合は判定材料が無いので fail しない。
failed = False
if not ran_any:
    ollama_st, cf_st = _proc("ollama"), _proc(TUNNEL_NAME)
    print(f"ℹ 即時終了のため死活判定はスキップ（ollama={ollama_st} "
          f"{TUNNEL_NAME}={cf_st}）")
else:
    failed = (ollama_st in ("DEAD", "unknown") or cf_st in ("DEAD", "unknown")
              or tunnel_dead >= 2)
report = "\n".join([
    f"result={'DEAD' if failed else ('stopped' if not RUN else 'alive')}",
    f"RUN={RUN}",
    f"minutes={KEEPALIVE_MINUTES}",
    f"ran_any={ran_any}",
    f"ollama={ollama_st}",
    f"tunnel={TUNNEL_NAME}:{cf_st}",
    f"url={url}",
    "",
] + log)
# 保存に失敗すると「生きている証跡」が残らないので、失敗も 1 件は証跡を残す
saved = save_output("keepalive.txt", report + "\n")
print()
print("⏱ キープアライブ終了。COMPLETE で /kaggle/working が zip 化されます")

# ============================================================
# 後片付け: GPU を明示的に手放す
# ============================================================
# これをやらないと、OLLAMA_KEEP_ALIVE=-1 で 23 GiB が VRAM に
# 固定されたまま残り、カーネルが COMPLETE になっても GPU を
# 掴み続ける（実測: expires_at が 2319 年になる）。
# プロセスを落としても Kaggle 側の枠開放は保証されないが、
# 少なくとも VRAM は即座に返る。
print()
print("=" * 56)
print("🧹 後片付け: GPU / プロセスを解放")
print("=" * 56)


# --- 1) モデルを明示的にアンロード（keep_alive=0）---
# これが一番効く。プロセスを殺すより先に、Ollama 自身に VRAM を返させる。
# 実装は共通ランタイム（runtime/engines/ollama.py）。
unloaded = kc_unload_model(PORT, MODEL_NAME)

# --- 2) VRAM が実際に空いたか確認 ---
kc_wait_vram_free(PORT)

# --- 3) プロセスを落とす（SIGTERM → 待機 → SIGKILL）---
# 順序: 外側（トンネル）→ 内側（プロキシ）→ 本体（ollama）
TARGETS = [
    ("-x ngrok", "ngrok"),
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-x ollama", "ollama"),
    ("-f llama-server", "llama-server"),
]

# 残存分は SIGKILL（llama-server は ollama の子で別名なので個別に落とす）
left = kc_stop_processes(TARGETS)

# --- 4) 最終確認 ---
print()

if left:
    print(f"  ⚠ 残存: {', '.join(left)}")
else:
    print("  ✅ 全プロセス停止（ollama / tunnel / proxy / llama-server）")

# 後片付けの結果も証跡として残す
save_output("cleanup.txt", "\n".join([
    f"unload_requested={unloaded}",
    f"leftover={','.join(left) if left else 'none'}",
    "",
]) + "\n")
print("  📄 cleanup.txt 保存済")
print("=" * 56)

if failed:
    fail("死活確認に失敗しました（keepalive.txt を確認）", "/tmp/ollama.log")
elif not saved:
    fail("keepalive.txt を保存できませんでした（成果物ディレクトリを確認）")

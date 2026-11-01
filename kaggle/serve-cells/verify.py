# 公開URL経由の疎通確認（共通の足場）
# 確認項目そのものは engine_checks() が持つ（1 つ前のセルで定義する）。
# 結果は /kaggle/working/verify_result.txt に残す（save_output が /output にも複製）。
# 失敗は raise（検証不能のまま放置しない）。
import json, time as _time, urllib.request, urllib.error

try:
    with open("/tmp/public_url.txt") as f:
        public_url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

# ---- トンネル確立の取り直し ----
# tunnel セルでエッジ伝播待ちが終わっていなくても、ここで吸収する。
# 届かなければ [1] がそのまま FAIL になる（握り潰さない）。
# Kaggle VM は AAAA を返すのに IPv6 経路が無いことがあるので IPv4 強制を使う。
_elapsed = kc_wait_http(f"{public_url}/healthz", attempts=10, interval=3,
                        opener=kc_urlopen_v4)
if _elapsed is not None:
    if _elapsed > 3:
        print(f"✅ トンネル確立をここで確認（{_elapsed}s）")
else:
    print("⚠ /healthz に届きません。トンネル未確立の可能性（[1] で判定）")

lines = [
    f"PUBLIC_URL={public_url}",
    f"MODEL={MODEL_NAME}",
    f"TIME={_time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]
failures = []


def _judge(idx, label, ok, detail):
    # ok=None は「この条件では判定できない」= SKIP。失敗には数えないが、
    # PASS とも書かない（後から見て「通った」と誤読しないようにする）。
    if ok is None:
        print(f"⏭ {label}: {detail}")
        lines.append(f"[{idx}] {label} -> SKIP: {detail}")
        return
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _with_retry(fn, tries=6, delay=5):
    # 実行時ログで確認した瞬間的障害: URLError [Errno 101] Network is
    # unreachable / [-2] Name or service not known。3回/4秒では吸収できず、
    # 待ちを 5,10,15,20,25 秒と伸ばして合計 75 秒の窓を持つ。
    # HTTP 応答（401/500 等）は「答え」なので再試行しない。
    return kc_retry(fn, tries=tries, delay=delay)


# ---- エンジン固有の確認項目 ----
engine_checks(_judge, _with_retry, public_url)

# ---- レポート ----
report = "\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\n"):
    # 保存失敗 = 証跡なし。ログは取れないので成功扱いにしてはいけない。
    failures.append("verify_result.txt を永続領域に保存できなかった")

if failures:
    fail("疎通確認に失敗: " + "; ".join(failures), ENGINE_LOG)
print("✅ すべての確認項目をパス")

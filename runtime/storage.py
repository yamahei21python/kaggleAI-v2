"""検証成果物の永続化と失敗記録（生成コードへ埋め込む）。"""
import glob
import os

# `kaggle kernels output` が zip 化するのは /kaggle/working だけ。
# /output は公式の永続先ではないが、書ければ保険になるので両方へ書く。
KC_OUT_DIRS = ("/kaggle/working", "/output")


def kc_find_dataset(slug, root="/kaggle/input"):
    """Dataset のマウント先を返す（見つからなければ既定パス）。

    Kaggle は `/kaggle/input/<slug>`（旧）と
    `/kaggle/input/datasets/<owner>/<slug>`（新）が同一日内に混在する。
    深さ 2 まで探し、**空でない**ディレクトリを優先する（空マウントを
    掴むと「Dataset はあるのに中身が無い」で落ちるため）。
    呼び出し側は戻り値をそのまま使ってよい（従来の `_find_input` と同じ）。
    """
    cands = []
    for pat in (f"{root}/{slug}", f"{root}/*/{slug}", f"{root}/*/*/{slug}"):
        cands += [p for p in glob.glob(pat) if os.path.isdir(p)]
    for p in sorted(cands):
        try:
            if os.listdir(p):
                return p
        except OSError:
            continue
    return cands[0] if cands else f"{root}/{slug}"


def kc_save_output(name, text, dirs=KC_OUT_DIRS):
    """1 つでも書ければ True（成功条件に使う）。"""
    saved = []
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
            saved.append(d)
        except Exception as e:
            print(f"  ⚠ {d} 書き出し失敗 ({e})")
    if saved:
        print(f"  📄 {name} 保存済: {', '.join(saved)}")
        return True
    print(f"  ⚠ 成果物を1つも保存できませんでした: {name}")
    return False


def kc_fail(msg, tail_path=None, dirs=KC_OUT_DIRS):
    """失敗原因を永続領域に残してから raise（ログ API は当てにできない）。"""
    tail = ""
    try:
        if tail_path and os.path.isfile(tail_path):
            tail = open(tail_path, errors="replace").read()[-3000:]
    except Exception:
        pass
    kc_save_output("FAILED.txt", f"{msg}\n\n{tail}", dirs)
    raise RuntimeError(msg)

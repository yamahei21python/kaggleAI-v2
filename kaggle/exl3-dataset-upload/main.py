#!/usr/bin/env python3
"""EXL3 モデルを Kaggle Dataset 化する（GPU 不要）。

GPU セッション中に HF から 15.9 GiB を落とす時間とディスクをなくすため、
CPU カーネルで「HF から DL → Kaggle Dataset 作成」までを済ませる。

Kaggle ノートブックには KAGGLE_USERNAME / KAGGLE_KEY が環境変数で
入っているので、ローカル PC を経由せず直接アップロードできる。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

RESULT = "/kaggle/working/result.json"
REPO = "Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw"
SLUG = "qwen38-27b-uncensored-exl3-4bpw"

data = {"stage": "init"}
T0 = time.time()


def save():
    data["elapsed_seconds"] = round(time.time() - T0, 1)
    with open(RESULT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sh(cmd, timeout=3600, env=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, env=env)
    return r, round(time.time() - t0, 1)


def tail(s, n=30):
    return "\n".join((s or "").strip().splitlines()[-n:])


def free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


# ── 1. 環境（認証・ディスク）────────────────────────────────
def load_creds():
    """Kaggle の認証情報。script カーネルには環境変数が入らないので
    Secrets（build_step1.py と同じ KAGGLE_USERNAME / KAGGLE_KEY）から読む。"""
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    source = "env" if (user and key) else None
    if not (user and key):
        try:
            from kaggle_secrets import UserSecretsClient
            sec = UserSecretsClient()
            user = user or (sec.get_secret("KAGGLE_USERNAME") or "")
            key = key or (sec.get_secret("KAGGLE_KEY") or "")
            source = "kaggle_secrets"
        except Exception as e:
            data["secrets_error"] = f"{type(e).__name__}: {e}"
    if user:
        os.environ["KAGGLE_USERNAME"] = user
    if key:
        os.environ["KAGGLE_KEY"] = key
    data["creds_source"] = source
    return user, key


def stage_env():
    print("=" * 60)
    print("EXL3 モデル → Kaggle Dataset")
    print("=" * 60)
    print(f"python : {sys.version.split()[0]}")

    user, key = load_creds()
    print(f"KAGGLE_USERNAME : {user or '(無し)'}")
    print(f"KAGGLE_KEY      : {'設定あり' if key else '(無し)'}"
          f"  (source: {data.get('creds_source')})")
    if not user or not key:
        raise RuntimeError(
            "KAGGLE_USERNAME / KAGGLE_KEY が環境変数にも Secrets にも無い")
    data["owner"] = user
    data["dataset_ref"] = f"{user}/{SLUG}"

    for p in ("/kaggle/working", "/kaggle/temp", "/tmp", "/"):
        if os.path.isdir(p):
            print(f"disk   : {p} {free_gib(p):.1f} GiB free")

    data["stage"] = "env"
    save()


# ── 2. ツール ─────────────────────────────────────────────
def stage_tools():
    print()
    print("--- kaggle / huggingface_hub ---")
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    r, dt = sh([sys.executable, "-m", "pip", "install", "-q", "-U",
                "kaggle", "huggingface_hub[hf_transfer]"], timeout=900,
               env=env)
    print(f"pip rc={r.returncode}  {dt}s")
    if r.returncode != 0:
        print(tail(r.stdout + "\n" + r.stderr, 20))

    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    me = api.get_config_value(api.CONFIG_NAME_USER)
    print(f"認証 OK: {me}  (kaggle {getattr(kaggle, '__version__', '?')})")
    data["kaggle_version"] = getattr(kaggle, "__version__", None)
    data["stage"] = "tools"
    save()


# ── 3. DL ────────────────────────────────────────────────
def pick_work_dir(need_gib=19.0):
    """アップロード用フォルダの親。working 以外を優先（出力を太らせない）。"""
    best = None
    for c in ("/kaggle/temp", "/tmp", "/kaggle/working"):
        if not os.path.isdir(c):
            continue
        g = free_gib(c)
        if g >= need_gib and (best is None or g > best[1]):
            best = (c, g)
    if best is None:
        raise RuntimeError(f"{need_gib:.0f} GiB 以上の空きが無い")
    return best[0]


def stage_download():
    print()
    print("--- HF から DL ---")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    dest = os.path.join(pick_work_dir(), "exl3-upload")
    shutil.rmtree(dest, ignore_errors=True)
    print(f"dest   : {dest}")

    t0 = time.time()
    snapshot_download(repo_id=REPO, local_dir=dest)
    dt = round(time.time() - t0, 1)

    total = sum(os.path.getsize(os.path.join(dest, f))
                for f in os.listdir(dest)
                if os.path.isfile(os.path.join(dest, f)))
    print(f"DL 完了: {total / 1024**3:.2f} GiB / {dt}s")
    data.update(download_seconds=dt, model_gib=round(total / 1024**3, 2),
                work_dir=dest)
    if total < 15 * 1024**3:
        raise RuntimeError("DL サイズが想定より小さい")
    data["stage"] = "download"
    save()
    return dest


# ── 4. Dataset 作成 ───────────────────────────────────────
def stage_upload(dest):
    print()
    print("--- Kaggle Dataset 作成 ---")
    shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
    meta = {
        "title": "Qwen3.8-27B-Uncensored EXL3 4bpw",
        "id": data["dataset_ref"],
        "subtitle": "EXL3 4.0bpw quant for ExLlamaV3 (text only)",
        "description": (
            f"Mirror of Hugging Face {REPO} for Kaggle GPU kernels. "
            "EXL3 4.0bpw (head 6bit, MTP head 8bit), 3 safetensors shards, "
            "language model only (vision not usable with exllamav3). "
            "Use with exllamav3: Config.from_directory('/kaggle/input/"
            f"{SLUG}')"),
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    with open(os.path.join(dest, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r, dt = sh([sys.executable, "-m", "kaggle", "datasets", "create",
                "-p", dest], timeout=5400, env=env)
    print(f"rc={r.returncode}  {dt}s")
    print(tail(r.stdout + "\n" + r.stderr, 40))
    data.update(upload_seconds=dt, upload_rc=r.returncode)
    if r.returncode != 0:
        data["upload_error_tail"] = tail(r.stdout + "\n" + r.stderr, 60)
        raise RuntimeError("kaggle datasets create 失敗")
    data["stage"] = "upload"
    save()


# ── 5. 検証・後片付け ─────────────────────────────────────
def stage_verify(dest):
    print()
    print("--- 検証 ---")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r, _ = sh([sys.executable, "-m", "kaggle", "datasets", "status",
               data["dataset_ref"]], timeout=600, env=env)
    print("status:", tail(r.stdout + "\n" + r.stderr, 3))
    data["dataset_status"] = tail(r.stdout, 3)

    r2, _ = sh([sys.executable, "-m", "kaggle", "datasets", "files",
                data["dataset_ref"], "--page-size", "100"],
               timeout=600, env=env)
    out = r2.stdout
    print(tail(out, 25))
    files = [ln.split()[0] for ln in out.splitlines()
             if ".safetensors" in ln or "config.json" in ln]
    data["dataset_files_seen"] = files[:20]
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError("アップロード後に safetensors が見えない")
    data["dataset_url"] = f"https://www.kaggle.com/datasets/{data['dataset_ref']}"

    # 出力を太らせないためローカルコピーを消す（Dataset 側に残る）
    shutil.rmtree(dest, ignore_errors=True)
    print("ローカルコピーを削除")
    data["stage"] = "verify"


def main():
    stage_env()
    stage_tools()
    dest = stage_download()
    stage_upload(dest)
    stage_verify(dest)


if __name__ == "__main__":
    try:
        main()
        data["ok"] = True
    except Exception as e:
        traceback.print_exc()
        data["ok"] = False
        data["error"] = f"{type(e).__name__}: {e}"
    save()
    print()
    print("RESULT: " + json.dumps(data, ensure_ascii=False))
    sys.exit(0)

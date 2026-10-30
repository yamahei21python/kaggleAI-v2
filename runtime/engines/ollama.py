"""Ollama 固有の推論・モデル解放処理（生成コードへ埋め込む）。"""
import json
import time
import urllib.request


def kc_ollama_chat(port, model, prompt, num_predict=64, timeout=1800,
                   think=None):
    """Ollama /api/chat を 1 回叩き、応答 dict を返す（非ストリーム）。

    think=None はボディにキーを入れない（Ollama 既定に任せる）。
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kc_unload_model(port, model, timeout=60):
    """keep_alive=0 でモデルを VRAM から降ろす。1 経路でも通れば True。

    プロセスを殺すより先に、Ollama 自身に VRAM を返させる。
    """
    unloaded = False
    for path, payload in [
        ("/api/generate", {"model": model, "keep_alive": 0}),
        ("/api/chat", {"model": model, "messages": [], "keep_alive": 0}),
    ]:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    unloaded = True
                    print(f"  ✅ アンロード要求 OK ({path})")
                    break
        except Exception as e:
            print(f"  —  {path}: {type(e).__name__}")
    return unloaded


def kc_wait_vram_free(port, attempts=10, interval=3):
    """常駐モデルが 0 になるまで待つ。空なら True。"""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/ps", timeout=10) as r:
                models = json.loads(r.read())
            loaded = models.get("models") or []
            if not loaded:
                print("  ✅ VRAM 解放確認（常駐モデル 0）")
                return True
            vram = sum(m.get("size_vram", 0) for m in loaded)
            print(f"  ⏳ まだ常駐: {vram:,} B（{attempt + 1}/{attempts}）")
        except Exception as e:
            print(f"  —  /api/ps 確認失敗: {type(e).__name__}")
            return False
        time.sleep(interval)
    print("  ⚠ アンロードを確認できず（プロセス kill へ進む）")
    return False

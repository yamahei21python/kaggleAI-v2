#!/usr/bin/env python3
"""vllm-flags: vLLM 0.32.0 の高度化フラグを CPU で確定する（GPU 0 秒）。

GPU を回す前に「そのフラグは存在するか / 値は何を受けるか / argv に渡して
弾かれないか」を潰しておく。CUDA 初期化より前に落ちるので CPU で足りる。

やること:
  1. vllm==0.32.0 を install（wheelhouse は解決不能なので PyPI）
  2. `vllm serve --help` を保存し、対象フラグと選択肢を抽出
  3. `EngineArgs` のフィールド実在を import で確認
  4. stub モデル + 各フラグで `vllm serve` を起動し、**引数段階で弾かれるか**だけ見る
     （"unrecognized arguments" / "invalid choice" = NG。それ以外の失敗 = 引数は OK）
  5. 対照実験として存在しないフラグを渡し、NG 判定が効くことを確認

出力: /kaggle/working/result.json（flags / choices / argv_checks）
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
RESULT = os.path.join(WORK, "result.json")
HELP_TXT = os.path.join(WORK, "vllm_serve_help.txt")
STUB = os.path.join(WORK, "stub_model")
VLLM_SPEC = os.environ.get("KC_VLLM_SPEC", "vllm==0.32.0")

DATA = {"stage": "init", "ok": False}

FLAGS = ["--speculative-config", "--enforce-eager", "--kv-cache-dtype",
         "--max-model-len", "--language-model-only", "--attention-backend",
         "--max-num-seqs", "--gpu-memory-utilization", "--chat-template",
         "--trust-remote-code", "--enable-prefix-caching",
         "--limit-mm-per-prompt", "--enable-chunked-prefill",
         "--async-scheduling", "--compilation-config", "--quantization"]


def save():
    with open(RESULT, "w") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2)


# CPU では device 推論に失敗して argparse 構築前に落ちる（実測 0.32.0）。
# VLLM_TARGET_DEVICE=cpu を与えるとパーサまで到達できる。
CPU_ENV = {**os.environ, "VLLM_TARGET_DEVICE": "cpu",
           "VLLM_LOGGING_LEVEL": "ERROR"}


def sh(cmd, timeout=1800, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=env)
    return p, round(time.time() - t0, 1)


def install():
    t0 = time.time()
    p, _ = sh([sys.executable, "-m", "pip", "install", "-q", "--no-input",
               VLLM_SPEC], timeout=3600)
    DATA["install"] = {"rc": p.returncode,
                       "seconds": round(time.time() - t0, 1),
                       "tail": ((p.stdout or "") + (p.stderr or ""))[-1200:]}
    if p.returncode != 0:
        raise RuntimeError("vllm install 失敗（上の tail 参照）")


def version():
    p, _ = sh([sys.executable, "-c",
               "import vllm; print(vllm.__version__)"], timeout=600)
    return {"rc": p.returncode, "out": (p.stdout or "").strip(),
            "err": (p.stderr or "").strip()[-400:]}


def help_text():
    p, sec = sh([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                 "--help"], timeout=900, env=CPU_ENV)
    txt = (p.stdout or "") + (p.stderr or "")
    with open(HELP_TXT, "w") as f:
        f.write(txt)
    return txt, sec


def choices_of(txt, flag):
    """`--flag {a,b,c}` 形式から選択肢を拾う。無ければ None。"""
    i = txt.find(flag)
    if i < 0:
        return None
    seg = txt[i:i + 600]
    m = re.search(r"\{(.+?)\}", seg, re.S)
    if not m:
        return None
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def engine_fields():
    code = (
        "import json\n"
        "from vllm.engine.arg_utils import EngineArgs\n"
        "import dataclasses\n"
        "fields = sorted(f.name for f in dataclasses.fields(EngineArgs))\n"
        "want = ['speculative_config','enforce_eager','kv_cache_dtype',\n"
        "        'max_model_len','language_model_only','attention_backend',\n"
        "        'compilation_config','quantization','max_num_seqs',\n"
        "        'gpu_memory_utilization','enable_prefix_caching']\n"
        "print(json.dumps({k: (k in fields) for k in want}))\n"
    )
    p, _ = sh([sys.executable, "-c", code], timeout=600)
    try:
        return {"rc": p.returncode, "fields": json.loads(p.stdout.strip()),
                "err": (p.stderr or "").strip()[-400:]}
    except Exception:
        return {"rc": p.returncode, "fields": None,
                "out": (p.stdout or "")[-400:],
                "err": (p.stderr or "").strip()[-400:]}


def usage_lines(txt):
    """usage 行から「そのフラグが受ける形（metavar / choices）」を拾う。"""
    out = {}
    for flag in FLAGS:
        for ln in txt.splitlines():
            if flag in ln:
                out[flag] = ln.strip()[:220]
                break
    return out


def enum_scan():
    """help に出ない選択肢は enum / dataclass から取る。"""
    code = (
        "import json\n"
        "out = {}\n"
        "try:\n"
        "    from vllm.v1.attention.backends.registry import (\n"
        "        AttentionBackendEnum as A)\n"
        "    out['attention_backends'] = [x.name for x in A]\n"
        "except Exception as e:\n"
        "    out['attention_backends'] = '%s: %s' % (type(e).__name__, e)\n"
        "try:\n"
        "    from vllm.config.speculative import SpeculativeConfig as S\n"
        "    out['spec_fields'] = sorted(getattr(S, '__dataclass_fields__', {}))\n"
        "except Exception as e:\n"
        "    out['spec_fields'] = '%s: %s' % (type(e).__name__, e)\n"
        "try:\n"
        "    from vllm.config.speculative import SpeculativeMethod as M\n"
        "    out['spec_methods'] = [x.name for x in M]\n"
        "except Exception as e:\n"
        "    out['spec_methods'] = '%s: %s' % (type(e).__name__, e)\n"
        "print(json.dumps(out))\n"
    )
    p, _ = sh([sys.executable, "-c", code], timeout=600, env=CPU_ENV)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"rc": p.returncode, "out": (p.stdout or "")[-400:],
                "err": (p.stderr or "")[-400:]}


def source_scan():
    """argparse 構築に失敗しても、ソース上のフラグ定義は確認できる。"""
    import glob as g
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    root = os.path.dirname(spec.origin)
    hits = {}
    for f in g.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            continue
        for flag in FLAGS:
            if f'"{flag}"' in txt or f"'{flag}'" in txt:
                hits.setdefault(flag, []).append(os.path.relpath(f, root))
    return {f: hits.get(f, [])[:3] for f in FLAGS}


def stub_model():
    os.makedirs(STUB, exist_ok=True)
    with open(os.path.join(STUB, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                   "hidden_size": 64, "num_hidden_layers": 1,
                   "num_attention_heads": 2, "num_key_value_heads": 2,
                   "vocab_size": 128}, f)


BAD = ("unrecognized arguments", "invalid choice", "error: argument",
       "no such option")


def argv_check(name, extra, timeout=180):
    """引数段階で弾かれるかだけを見る（先に進む＝フラグとしては受理）。"""
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", STUB, "--host", "127.0.0.1", "--port", "8019",
           "--max-model-len", "256"] + extra
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=CPU_ENV)
        out = ((p.stdout or "") + (p.stderr or ""))[-2500:]
        rc = p.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace")
               + (e.stderr or b"").decode("utf-8", "replace"))[-2000:]
        rc, timed_out = None, True
    bad = [b for b in BAD if b in out]
    return {"args": extra, "rc": rc, "timed_out": timed_out,
            "rejected_by_argparse": bool(bad), "matched": bad,
            "seconds": round(time.time() - t0, 1),
            "tail": out[-800:]}


CASES = [
    ("spec_mtp", ["--speculative-config",
                  '{"method":"mtp","num_speculative_tokens":2}']),
    ("spec_mtp_draft", ["--speculative-config",
                        '{"method":"mtp","num_speculative_tokens":2,'
                        '"model":"/kaggle/working/stub_model"}']),
    ("kv_q8", ["--kv-cache-dtype", "q8_0"]),
    ("kv_fp8", ["--kv-cache-dtype", "fp8"]),
    ("eager", ["--enforce-eager"]),
    ("maxlen8192", ["--max-model-len", "8192"]),
    ("lmo", ["--language-model-only"]),
    ("triton", ["--attention-backend", "TRITON_ATTN"]),
    # 対照実験（これが「受理」になったら判定器が壊れている）
    ("bogus", ["--definitely-not-a-flag"]),
]


def main():
    DATA["stage"] = "install"
    save()
    install()
    DATA["version"] = version()
    DATA["stage"] = "help"
    txt, sec = help_text()
    DATA["help"] = {"seconds": sec, "bytes": len(txt)}
    DATA["flags"] = {f: (f in txt) for f in FLAGS}
    DATA["choices"] = {
        "kv_cache_dtype": choices_of(txt, "--kv-cache-dtype"),
        "attention_backend": choices_of(txt, "--attention-backend"),
        "quantization": choices_of(txt, "--quantization"),
    }
    DATA["speculative_help"] = txt[txt.find("--speculative-config"):
                                   txt.find("--speculative-config") + 400] \
        if "--speculative-config" in txt else None
    DATA["engine_fields"] = engine_fields()
    DATA["source_scan"] = source_scan()
    DATA["usage_lines"] = usage_lines(txt)
    DATA["enums"] = enum_scan()
    save()

    DATA["stage"] = "argv"
    stub_model()
    for name, extra in CASES:
        DATA.setdefault("argv_checks", {})[name] = argv_check(name, extra)
        save()

    DATA["stage"] = "done"
    checks = DATA["argv_checks"]
    # 設計上「弾かれて正解」のケースは ok の判定から外す。kv_q8 は
    # `--kv-cache-dtype q8_0` が**存在しない**ことの確認なので、invalid choice で
    # 弾かれるのが正解。これを positive 側に混ぜていたため、**完走しているのに
    # ok=False** になっていた（実測 2026-10-31 v2。ログを読むまで気づけない）。
    _NEGATIVE = ("bogus", "kv_q8")
    positive = {n: c for n, c in checks.items() if n not in _NEGATIVE}
    # 判定器が正常 = 負のケースを両方とも argparse で弾けている
    DATA["detector_works"] = all(
        checks[n]["rejected_by_argparse"] for n in _NEGATIVE if n in checks)
    DATA["ok"] = (DATA["detector_works"]
                  and all(not c["rejected_by_argparse"]
                          for c in positive.values()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        DATA["stage"] = "failed"
        DATA["error"] = f"{type(e).__name__}: {e}"
    save()
    print("\nRESULT: " + json.dumps(DATA, ensure_ascii=False)[:3000])
    sys.exit(0 if DATA.get("stage") == "done" else 1)

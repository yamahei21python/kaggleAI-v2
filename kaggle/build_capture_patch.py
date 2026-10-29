#!/usr/bin/env python3
"""Capture ollama install artifacts into /kaggle/working for patch dataset."""
import json

OUT = "kaggle/capture_patch/capture_patch.ipynb"

cells = []

cells.append(("markdown", """# Capture Ollama runtime patch

Install ollama via official install.sh on a fresh Kaggle CPU instance and copy
the resulting binaries / libraries to `/kaggle/working/bin/` and
`/kaggle/working/lib/` so `kernels output` produces a zip matching the layout
expected by `my-ollama-orcarouter-qwen38-27b-patch`.

* GPU: off (binaries are the same; CPU instance starts faster).
* Internet: required.
"""))

cells.append(("code", '''# 1) Install ollama (official script).
# Provides /usr/local/bin/ollama, /usr/local/lib/ollama/llama-server, *.so,
# cuda_v12/libggml-cuda.so, license files.
import os, shutil, subprocess
from pathlib import Path

BIN_SRC = "/usr/local/bin/ollama"
LIB_SRC = "/usr/local/lib/ollama"
REQUIRED = [
    BIN_SRC,
    f"{LIB_SRC}/llama-server",
    f"{LIB_SRC}/cuda_v12/libggml-cuda.so",
]

# install.sh unpacks .tar.zst archives; ensure zstd is available
if subprocess.run(["dpkg", "-s", "zstd"], capture_output=True).returncode != 0:
    subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
                   shell=True, check=True, timeout=600)
    print("✅ zstd installed")

print("⬇️  Running ollama install.sh...")
inst = subprocess.run(
    ["bash", "-o", "pipefail", "-c",
     "curl -fsSL --retry 3 --connect-timeout 15 --max-time 300 "
     "https://ollama.com/install.sh | sh"],
    capture_output=True, text=True, timeout=1800,
)
print(inst.stdout[-2000:])
if inst.stderr:
    print("STDERR:", inst.stderr[-1000:])
if inst.returncode != 0:
    raise RuntimeError(f"install.sh failed rc={inst.returncode}")

# Verify key files; fail loudly if any are missing
for p in REQUIRED:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing after install: {p}")
    print(f"✅ {p}  ({os.path.getsize(p):,} B)")
'''))

cells.append(("code", '''# 2) Pack install artifacts into a single tar.gz under /kaggle/working.
# kernels output is unreliable for many large files; one archive downloads cleanly.
import os, shutil, subprocess, tarfile
from pathlib import Path

STAGE = "/tmp/ollama_patch_stage"
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(f"{STAGE}/bin", exist_ok=True)
shutil.copy2("/usr/local/bin/ollama", f"{STAGE}/bin/ollama")
shutil.copytree("/usr/local/lib/ollama", f"{STAGE}/lib/ollama")

# Optional: prune unused cuda variants to keep the dataset small
for variant in ["cuda_v13"]:
    p = f"{STAGE}/lib/ollama/{variant}"
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print(f"  pruned {p}")

# Traceability
ver = subprocess.run(["/usr/local/bin/ollama", "--version"],
                     capture_output=True, text=True)
Path(f"{STAGE}/VERSION.txt").write_text((ver.stdout or ver.stderr).strip() + "\\n")

print(subprocess.run(["du", "-sh", STAGE], capture_output=True, text=True).stdout)

ARCHIVE = "/kaggle/working/ollama_patch.tar.gz"
with tarfile.open(ARCHIVE, "w:gz") as tar:
    tar.add(STAGE, arcname="ollama_patch")
print(f"📦 {ARCHIVE}  ({os.path.getsize(ARCHIVE):,} B)")
'''))

nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if t == "code" else {}),
        }
        for t, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}: {len(cells)} cells")

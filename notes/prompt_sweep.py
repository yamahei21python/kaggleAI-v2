#!/usr/bin/env python3
"""Prompt-length sweep for local inference engines.

Measures prefill and decode throughput for a fixed output length while varying
the prompt length. Uses only locally available models; no downloads.

Engines:
  llama   - starts llama-server and calls the native /completion endpoint
  ollama  - calls an already-running Ollama daemon
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "The quick brown fox jumps over the lazy dog. "
SUFFIX = "\nContinue the sequence: 1, 2, 3, 4,"


def make_prompt(target_tokens: int) -> str:
    """Approximate a target token count with repeated natural text."""
    repeats = max(1, round(target_tokens / 10))
    return BASE * repeats + SUFFIX


def post_json(url: str, payload: dict[str, Any], timeout: float = 1800) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} POST {url}: {body[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"URLError POST {url}: {error.reason}") from error


def wait_for_health(url: str, process: subprocess.Popen[bytes], timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {url}")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_tps(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def run_llama(args: argparse.Namespace, lengths: list[int]) -> list[dict[str, Any]]:
    executable = shutil.which("llama-server")
    if executable is None:
        raise RuntimeError("llama-server not found in PATH")
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise RuntimeError(f"model not found: {model_path}")

    command = [
        executable,
        "-m",
        str(model_path),
        "-c",
        str(args.ctx),
        "-ngl",
        "99",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]
    log_path = Path(args.server_log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    records: list[dict[str, Any]] = []
    try:
        wait_for_health(f"http://127.0.0.1:{args.port}/health", process)

        # Warm up the model and the server path; discard the result.
        post_json(
            f"http://127.0.0.1:{args.port}/completion",
            {
                "prompt": make_prompt(64),
                "n_predict": 8,
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "ignore_eos": True,
                "stream": False,
                "cache_prompt": False,
            },
        )

        for index, target in enumerate(lengths):
            if index:
                time.sleep(args.cooldown)
            prompt = make_prompt(target)
            started = time.perf_counter()
            response = post_json(
                f"http://127.0.0.1:{args.port}/completion",
                {
                    "prompt": prompt,
                    "n_predict": args.gen,
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                    "ignore_eos": True,
                    "stream": False,
                    "cache_prompt": False,
                },
            )
            wall = time.perf_counter() - started
            timings = response.get("timings") or {}
            prompt_n = int(timings.get("prompt_n") or 0)
            predicted_n = int(timings.get("predicted_n") or 0)
            record = {
                "engine": "llama.cpp",
                "target_prompt_tokens": target,
                "prompt_tokens": prompt_n,
                "gen_tokens": predicted_n,
                "prefill_tps": timings.get("prompt_per_second"),
                "decode_tps": timings.get("predicted_per_second"),
                "prefill_ms": timings.get("prompt_ms"),
                "decode_ms": timings.get("predicted_ms"),
                "wall_s": round(wall, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            records.append(record)
            write_records(Path(args.out), records)
            print(
                f"llama target={target:5d} prompt={prompt_n:5d} gen={predicted_n:3d} "
                f"prefill={format_tps(record['prefill_tps'])} t/s "
                f"decode={format_tps(record['decode_tps'])} t/s",
                flush=True,
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()

    return records


def run_ollama(args: argparse.Namespace, lengths: list[int]) -> list[dict[str, Any]]:
    base_url = "http://127.0.0.1:11434/api/generate"
    common_options = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "num_ctx": args.ctx,
        "num_predict": args.gen,
    }

    # Warm up and keep the model resident; discard the result.
    post_json(
        base_url,
        {
            "model": args.model,
            "prompt": make_prompt(64),
            "stream": False,
            "raw": True,
            "options": {**common_options, "num_predict": 8},
            "keep_alive": "10m",
        },
    )

    records: list[dict[str, Any]] = []
    for index, target in enumerate(lengths):
        if index:
            time.sleep(args.cooldown)
        prompt = make_prompt(target)
        started = time.perf_counter()
        response = post_json(
            base_url,
            {
                "model": args.model,
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "options": common_options,
                "keep_alive": "10m",
            },
        )
        wall = time.perf_counter() - started
        prompt_n = int(response.get("prompt_eval_count") or 0)
        eval_n = int(response.get("eval_count") or 0)
        prompt_ns = int(response.get("prompt_eval_duration") or 0)
        eval_ns = int(response.get("eval_duration") or 0)
        record = {
            "engine": "ollama",
            "target_prompt_tokens": target,
            "prompt_tokens": prompt_n,
            "gen_tokens": eval_n,
            "prefill_tps": prompt_n / (prompt_ns / 1e9) if prompt_ns else None,
            "decode_tps": eval_n / (eval_ns / 1e9) if eval_ns else None,
            "prefill_ms": prompt_ns / 1e6,
            "decode_ms": eval_ns / 1e6,
            "wall_s": round(wall, 3),
            "load_s": round(int(response.get("load_duration") or 0) / 1e9, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        records.append(record)
        write_records(Path(args.out), records)
        print(
            f"ollama target={target:5d} prompt={prompt_n:5d} gen={eval_n:3d} "
            f"prefill={format_tps(record['prefill_tps'])} t/s "
            f"decode={format_tps(record['decode_tps'])} t/s",
            flush=True,
        )

    return records


def parse_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            lengths.append(int(token))
        except ValueError as error:
            raise SystemExit(f"invalid length: {token!r}") from error
    return lengths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("llama", "ollama"), required=True)
    parser.add_argument("--model", required=True, help="GGUF path for llama, model name for Ollama")
    parser.add_argument("--lengths", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--gen", type=int, default=256, help="fixed number of generated tokens")
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds between measured prompts")
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--out", default="notes/prompt-sweep.json")
    parser.add_argument("--server-log", default="notes/prompt-sweep-server.log")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lengths = parse_lengths(args.lengths)
    if any(length < 1 for length in lengths):
        raise SystemExit("all prompt lengths must be positive")

    if args.engine == "llama":
        run_llama(args, lengths)
    else:
        run_ollama(args, lengths)
    return 0


if __name__ == "__main__":
    sys.exit(main())

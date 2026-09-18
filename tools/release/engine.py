#!/usr/bin/env python3
"""Shared engine control for the measurement harnesses.

Six harnesses each carried their own copy of the same three things: killing a stray
ninfer-serve.exe, waiting for one to answer /v1/models, and starting one from an argument list.
That is the module hiding in the tools/release cluster, and this is it.

The interface is deliberately small and returns values rather than leaking state:

    kill_servers()                    -- stop any running engine
    wait_ready(port, proc, timeout)   -- True once /v1/models answers, False if the process died
    start_engine(args, log_path, cwd) -- start it, returning (process, log handle)

wait_ready takes the process so a refused profile, which exits in about four seconds, stops the
wait immediately instead of burning the whole timeout on a port that will never open. That
distinction mattered: an earlier ceiling probe spent minutes waiting on profiles that had already
exited.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path


def kill_servers() -> None:
    """Stop any running engine. One 32 GB card holds one artifact, so harnesses serialise."""
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True, text=True)


def wait_ready(port: int, proc: subprocess.Popen | None = None,
               timeout: int = 240) -> bool:
    """Poll /v1/models until the engine answers.

    Returns False as soon as the process has exited, because a refused profile exits in about
    four seconds and the port will never open.
    """
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                r.read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def start_engine(args: list[str], log_path: Path, cwd: str) -> tuple[subprocess.Popen, object]:
    """Start the engine with its output going to log_path.

    The caller owns the returned handle and should close it after the process ends, so the log
    is complete before anything reads it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT)
    return proc, handle


def stop_engine(proc: subprocess.Popen | None, handle=None) -> None:
    """Terminate a process started by start_engine and close its log."""
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
    kill_servers()

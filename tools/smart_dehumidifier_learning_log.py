#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from urllib.parse import unquote

HEADER = "timestamp,event,scene,mode_or_state,humidity,target,extra"


def _normalize_line(raw: str) -> str:
    return raw.replace("\r", " ").replace("\n", " ").strip()


def _locked_open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.seek(0)
    return handle


def ensure_header(path: Path) -> None:
    with _locked_open(path) as handle:
        content = handle.read()
        if not content.strip():
            handle.seek(0)
            handle.truncate()
            handle.write(f"{HEADER}\n")
            handle.flush()


def append_line(path: Path, line: str) -> None:
    clean = _normalize_line(line)
    if not clean:
        return
    with _locked_open(path) as handle:
        content = handle.read()
        if not content.strip():
            handle.seek(0)
            handle.truncate()
            handle.write(f"{HEADER}\n")
        handle.seek(0, 2)
        if content and not content.endswith("\n"):
            handle.write("\n")
        handle.write(f"{clean}\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart dehumidifier learning log helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--file", required=True)

    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--file", required=True)
    append_parser.add_argument("--line", default="")
    append_parser.add_argument("--line-url", default="")

    args = parser.parse_args()
    path = Path(args.file).expanduser()

    if args.command == "init":
        ensure_header(path)
        return 0

    line = args.line or unquote(args.line_url)
    append_line(path, line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

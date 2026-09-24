#!/usr/bin/env python3
"""Work around a portable-reader 100vw overflow when a scrollbar is present."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = (
    "width:100vw;height:48px;min-height:48px;"
    "margin-right:calc(50% - 50vw);margin-left:calc(50% - 50vw);"
)
NEW = "width:100%;height:48px;min-height:48px;margin-right:0;margin-left:0;"
HEAD_CLOSE = "</head>"
OVERFLOW_FIX = "<style>html,body{overflow-x:hidden!important}</style></head>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    text = args.html.read_text(encoding="utf-8")
    occurrences = text.count(OLD)
    if occurrences != 1:
        raise RuntimeError(f"expected one portable header rule, found {occurrences}")
    text = text.replace(OLD, NEW)
    if text.count(HEAD_CLOSE) != 1:
        raise RuntimeError("expected one closing head tag")
    args.html.write_text(text.replace(HEAD_CLOSE, OVERFLOW_FIX), encoding="utf-8")
    print(args.html)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Package the canonical GP-QVI artifact as a portable HTML report.

The upstream portable-report theme uses a 100vw sticky header.  On Chromium
builds that reserve space for a vertical scrollbar this can create a harmless
one-scrollbar-width horizontal overflow.  The final style below clips only
that page-level overflow; tables and code blocks retain their own scrolling.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "output" / "gp_qvi_report_artifact.json"
OUTPUT = HERE / "output" / "7709_202607_GP_QVI回测报告.html"


def find_builder() -> Path:
    override = os.environ.get("DATA_ANALYTICS_REPORT_BUILDER")
    if override:
        return Path(override).expanduser().resolve()
    pattern = (
        ".codex/plugins/cache/openai-curated-remote/data-analytics/*/skills/"
        "build-report/scripts/build_portable_artifact.mjs"
    )
    matches = sorted(Path.home().glob(pattern), reverse=True)
    if not matches:
        raise FileNotFoundError(
            "Data Analytics portable-report builder not found; set "
            "DATA_ANALYTICS_REPORT_BUILDER"
        )
    return matches[0]


def main() -> None:
    builder = find_builder()
    subprocess.run(
        ["node", str(builder), "--input", str(ARTIFACT), "--output", str(OUTPUT)],
        check=True,
    )
    html = OUTPUT.read_text(encoding="utf-8")
    containment = "<style>html,body{max-width:100%;overflow-x:hidden}</style>"
    if containment not in html:
        html = html.replace("</head>", f"{containment}\n</head>", 1)
        OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

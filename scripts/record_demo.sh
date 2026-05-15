#!/usr/bin/env bash
#
# Re-record the demo GIF embedded in README.md.
#
# Run this when the kt-masterviz dashboard UI changes meaningfully and
# the existing GIF no longer reflects current behavior. Otherwise leave
# alone — the GIF is committed to the repo and served straight from
# GitHub.
#
# Total runtime: ~25 seconds.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Syncing dev dependencies (playwright + pillow)"
uv sync --group dev

echo "==> Ensuring Chromium is installed for Playwright"
uv run playwright install chromium

echo "==> Recording (~18 seconds capture)"
uv run python scripts/_record.py

echo ""
echo "GIF written to assets/demo.gif"
echo ""
echo "Inspect, then commit if it looks good:"
echo "    git add assets/demo.gif"
echo "    git commit -m 'docs: regenerate demo GIF'"

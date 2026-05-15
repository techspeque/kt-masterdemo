#!/usr/bin/env bash
#
# 10-second end-to-end smoke test. Used by CI to validate that the
# PyPI-installed packages still work together.
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
uv run python examples/synthetic_smoke.py

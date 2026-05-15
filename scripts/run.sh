#!/usr/bin/env bash
#
# Run the Fashion-MNIST demo and tell the user to open the dashboard
# in another terminal. Skips the awkward "wait, which command first?"
# question by starting the tuner immediately.
set -euo pipefail

cd "$(dirname "$0")/.."

cat <<'EOF'
Starting the Fashion-MNIST CNN sweep. While it runs, open a SECOND
terminal in this directory and launch the live dashboard:

    uv run kt-masterviz --latest

The dashboard auto-refreshes; new trials show up within seconds of
completing.

EOF

uv run python examples/fashion_mnist_cnn.py

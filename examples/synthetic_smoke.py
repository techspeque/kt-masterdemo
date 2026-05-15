"""Tiny end-to-end smoke test against the PyPI-installed packages.

Not for showcasing — for *validating*. Used by the daily CI workflow
to confirm that:

  1. `pip install kt-masterlog` produces a working package
  2. `pip install kt-masterviz` produces a working package
  3. The cross-package registry contract still holds
  4. The CLI entry point `kt-masterviz --list` finds the run

About 10 seconds end-to-end. Synthetic data, 2 trials × 1 epoch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from kt_masterlog import TunerConfig, optimize


def _build(hp):
    units = hp.Choice("units", [4, 8])
    lr = hp.Choice("lr", [1e-2, 1e-3])
    m = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        registry_dir = tmp_path / "registry"
        os.environ["KT_MASTERLOG_REGISTRY_DIR"] = str(registry_dir)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, 4)).astype("float32")
        y = rng.integers(0, 2, 64).astype("int32")

        print("[1/4] Running kt-masterlog optimize()...")
        result = optimize(
            builder_fn=_build,
            train_data=x,
            val_data=(x, y),
            config=TunerConfig(
                project_name="pypi_smoke",
                output_dir=str(tmp_path / "runs"),
                strategy="random",
                max_trials=2,
                search_epochs=1,
                early_stop_patience=5,
                extra_fields={"smoke": "true"},
            ),
            search_kwargs={"y": y, "verbose": 0},
        )

        print("[2/4] Verifying master CSV was written...")
        assert Path(result.master_csv_path).exists(), "master CSV missing"

        print("[3/4] Verifying registry manifest...")
        manifests = list(registry_dir.glob("*.json"))
        assert len(manifests) == 1, f"expected 1 manifest, got {len(manifests)}"
        manifest = json.loads(manifests[0].read_text())
        assert manifest["status"] == "completed", manifest["status"]
        assert manifest["project_name"] == "pypi_smoke"
        assert manifest["csv_path"] == str(Path(result.master_csv_path).resolve())

        print("[4/4] Verifying kt-masterviz --list finds the run...")
        proc = subprocess.run(
            ["kt-masterviz", "--list"],
            capture_output=True,
            text=True,
            env={**os.environ, "KT_MASTERLOG_REGISTRY_DIR": str(registry_dir)},
        )
        assert proc.returncode == 0, proc.stderr
        assert "pypi_smoke" in proc.stdout, proc.stdout

        print()
        print("All checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

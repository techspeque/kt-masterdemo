"""Workflow demo - a small Bayesian sweep over a synthetic dataset.

The point of this script is **not** to model anything in particular. It
exists to demonstrate the kt-masterlog + kt-masterviz workflow against
a deliberately abstract problem:

    sklearn.datasets.make_classification  ->  small MLP  ->  KerasTuner

3 trials × 15 epochs over five hyperparameters via Bayesian search.
Runs in roughly 30–60 seconds on CPU. Each trial registers itself with
kt-masterlog's run registry, so the live dashboard can pick it up:

    Terminal 1:  uv run python examples/sweep_demo.py
    Terminal 2:  uv run kt-masterviz --latest

The synthetic dataset is fully deterministic (fixed ``random_state``)
so re-running this script produces the same numbers. Treat the model
architecture and hyperparameter ranges as illustrative - they are
chosen to make the dashboard interesting to look at, not to recommend
any particular configuration.
"""

from __future__ import annotations

import os

import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers

from kt_masterlog import TunerConfig, optimize


N_FEATURES = 20
N_CLASSES = 4


def build_model(hp):
    """Hyperparameter-tunable MLP. KerasTuner calls this once per trial."""
    units_1 = hp.Choice("units_1", [32, 64, 128])
    units_2 = hp.Choice("units_2", [16, 32, 64])
    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1)
    lr = hp.Choice("lr", [1e-2, 1e-3, 3e-4])
    optimizer_name = hp.Choice("optimizer", ["adam", "sgd"])

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(N_FEATURES,)),
            layers.Dense(units_1, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(units_2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(N_CLASSES, activation="softmax"),
        ]
    )

    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data():
    """Synthetic multi-class dataset. Fully deterministic via random_state."""
    x, y = make_classification(
        n_samples=4000,
        n_features=N_FEATURES,
        n_informative=12,
        n_redundant=4,
        n_classes=N_CLASSES,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=42,
    )
    x = x.astype("float32")
    y = y.astype("int32")
    return train_test_split(x, y, test_size=0.25, random_state=42)


def main():
    # Keep TensorFlow's startup chatter out of the demo output.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    x_train, x_val, y_train, y_val = load_data()

    config = TunerConfig(
        project_name="sweep_demo",
        output_dir="./runs",
        strategy="bayesian",
        max_trials=3,
        strategy_kwargs={"num_initial_points": 2},
        search_epochs=15,
        # Patience high enough that all 15 epochs run - the demo's
        # value is showing complete curves, not finding the optimum.
        early_stop_patience=20,
        objective_metric="val_accuracy",
        objective_direction="max",
        extra_fields={
            "dataset": "synthetic",
            "n_samples": int(len(x_train) + len(x_val)),
            "n_classes": N_CLASSES,
        },
    )

    result = optimize(
        builder_fn=build_model,
        train_data=x_train,
        val_data=(x_val, y_val),
        config=config,
        search_kwargs={"y": y_train, "batch_size": 64, "verbose": 0},
    )

    print()
    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    print(f"\nMaster CSV: {result.master_csv_path}")
    print("\nView the run in the dashboard:")
    print("    uv run kt-masterviz --latest")


if __name__ == "__main__":
    main()

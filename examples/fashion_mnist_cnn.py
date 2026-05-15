"""Fashion-MNIST CNN sweep — the showcase demo.

Tunes a small CNN over five hyperparameters using Bayesian search.
About 2–3 minutes on CPU; faster on GPU. Each trial registers itself
with kt-masterlog's run registry, so you can watch progress live in
the dashboard while the search runs.

Workflow:

    Terminal 1:  uv run python examples/fashion_mnist_cnn.py
    Terminal 2:  uv run kt-masterviz --latest

In Terminal 2 the dashboard opens at http://localhost:8501 and the
auto-refresh in the sidebar (default 5s) pulls in new trials as the
tuner finishes them.

Why Fashion-MNIST: it's slightly heavier than plain MNIST (more visual
variety, a real CNN actually helps), but still small enough that the
demo runs in minutes rather than hours. The Keras datasets API caches
the ~30MB download on first run.
"""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow.keras import layers, optimizers

from kt_masterlog import TunerConfig, optimize


def build_model(hp):
    """Hyperparameter-tunable CNN. KerasTuner calls this once per trial."""
    filters = hp.Choice("filters", [16, 32, 64])
    dense_units = hp.Choice("dense_units", [64, 128, 256])
    dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
    lr = hp.Choice("lr", [1e-2, 1e-3, 3e-4])
    optimizer_name = hp.Choice("optimizer", ["adam", "sgd"])

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(filters, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(filters * 2, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(10, activation="softmax"),
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


def load_data(train_size: int = 8000, val_size: int = 2000):
    """Subsample Fashion-MNIST so a trial takes ~15s instead of minutes."""
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_val = (x_val.astype("float32") / 255.0)[..., None]
    return (
        x_train[:train_size],
        y_train[:train_size],
        x_val[:val_size],
        y_val[:val_size],
    )


def main():
    # Quiet TensorFlow's startup chatter — keep the demo output focused
    # on tuning progress.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    x_train, y_train, x_val, y_val = load_data()

    config = TunerConfig(
        project_name="fashion_mnist_sweep",
        output_dir="./runs",
        strategy="bayesian",
        max_trials=8,
        strategy_kwargs={"num_initial_points": 3},
        search_epochs=5,
        early_stop_patience=2,
        objective_metric="val_accuracy",
        objective_direction="max",
        extra_fields={
            "dataset": "fashion_mnist",
            "train_size": len(x_train),
            "val_size": len(x_val),
        },
    )

    result = optimize(
        builder_fn=build_model,
        train_data=x_train,
        val_data=(x_val, y_val),
        config=config,
        search_kwargs={"y": y_train, "batch_size": 128, "verbose": 0},
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

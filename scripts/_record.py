"""Record a demo GIF of the kt-masterviz dashboard updating live.

Architecture
------------
1. Pre-stage a master CSV with only the header row.
2. Launch ``kt-masterviz`` against the CSV in a subprocess.
3. Wait for the dashboard's ``/_stcore/health`` endpoint.
4. In a background thread, append realistic trial rows on a schedule.
5. With Playwright headless Chromium, capture frames at FPS.
6. Stitch frames into an optimized GIF via Pillow.

The demo data is deterministic — re-running this produces a similar
GIF every time. The trial values are hand-tuned to mimic a real
Fashion-MNIST sweep with progressive ``val_accuracy`` improvement; the
true winner is trial 0004.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import closing
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


# ----- Configuration -------------------------------------------------

CSV_PATH = Path("/tmp/kt_masterdemo_recording.csv")
GIF_PATH = Path("assets/demo.gif")
FPS = 4
DURATION_SECONDS = 22
ROW_INTERVAL_SECONDS = 0.4
GIF_WIDTH = 900
VIEWPORT = {"width": 1280, "height": 1180}
N_EPOCHS = 15

HEADER = [
    "trial_id", "epoch", "filters", "dense_units", "dropout",
    "lr", "optimizer", "dataset",
    "loss", "val_loss", "accuracy", "val_accuracy",
]

# 3 trials × 15 epochs, written sequentially (trial 1 epochs 1..15, then
# trial 2, then trial 3) to mirror how KerasTuner actually emits rows.
# Trial 0002 wins on val_accuracy; trial 0003 starts strong but its high
# learning rate + low dropout cause val_loss to drift up in late epochs
# (the classic overfitting signature, useful for showing what the
# dashboard reveals at a glance).
TRIALS = [
    {
        "trial_id": "0001", "filters": 16, "dense_units": 64,
        "dropout": 0.3, "lr": 0.01,   "optimizer": "adam",
        "train_floor": 0.42, "val_floor": 0.51, "val_drift": 0.00,
        "decay": 2.8,
    },
    {
        "trial_id": "0002", "filters": 32, "dense_units": 128,
        "dropout": 0.4, "lr": 0.001,  "optimizer": "adam",
        "train_floor": 0.28, "val_floor": 0.34, "val_drift": 0.00,
        "decay": 3.2,
    },
    {
        "trial_id": "0003", "filters": 64, "dense_units": 256,
        "dropout": 0.2, "lr": 0.01,   "optimizer": "sgd",
        "train_floor": 0.30, "val_floor": 0.42, "val_drift": 0.18,
        "decay": 3.5,
    },
]


def _generate_rows() -> list[tuple]:
    """Deterministic, realistic-looking trial data.

    Loss/val_loss follow ``floor + (start - floor) * exp(-decay * t)``
    with small Gaussian noise. Trials with ``val_drift > 0`` add a
    linear up-drift after epoch 7 to simulate overfitting.
    """
    rng = np.random.default_rng(42)
    rows: list[tuple] = []
    for trial in TRIALS:
        for epoch in range(1, N_EPOCHS + 1):
            t = epoch / N_EPOCHS

            loss = (
                trial["train_floor"]
                + (0.95 - trial["train_floor"]) * np.exp(-trial["decay"] * t)
                + rng.normal(0, 0.012)
            )
            val_loss = (
                trial["val_floor"]
                + (1.05 - trial["val_floor"]) * np.exp(-trial["decay"] * t)
                + trial["val_drift"] * max(0.0, t - 0.5)
                + rng.normal(0, 0.018)
            )
            accuracy = 1.0 - 0.55 * loss + rng.normal(0, 0.008)
            val_accuracy = 1.0 - 0.55 * val_loss + rng.normal(0, 0.012)

            rows.append((
                trial["trial_id"], epoch,
                trial["filters"], trial["dense_units"], trial["dropout"],
                trial["lr"], trial["optimizer"], "fashion_mnist",
                round(float(loss), 4), round(float(val_loss), 4),
                round(float(accuracy), 4), round(float(val_accuracy), 4),
            ))
    return rows


ROWS = _generate_rows()


# ----- Helpers -------------------------------------------------------


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_healthy(url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as r:
                if r.read() == b"ok":
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def _write_csv_header() -> None:
    CSV_PATH.write_text(",".join(HEADER) + "\n")


def _append_rows_on_schedule(stop_event: threading.Event) -> None:
    """Append one row every ROW_INTERVAL_SECONDS until exhausted or stopped."""
    for row in ROWS:
        if stop_event.is_set():
            return
        time.sleep(ROW_INTERVAL_SECONDS)
        if stop_event.is_set():
            return
        with CSV_PATH.open("a") as f:
            f.write(",".join(str(v) for v in row) + "\n")


def _capture_frames(url: str) -> list[Image.Image]:
    """Headless Chromium screenshots at FPS for DURATION_SECONDS."""
    frames: list[Image.Image] = []
    interval = 1.0 / FPS
    total = DURATION_SECONDS * FPS

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport=VIEWPORT)
            page.goto(url)
            page.wait_for_selector("text=kt-masterviz", timeout=15_000)

            start = time.time()
            for i in range(total):
                png_bytes = page.screenshot(type="png", full_page=False)
                img = Image.open(BytesIO(png_bytes))
                # Downscale for a reasonable GIF file size.
                ratio = GIF_WIDTH / img.width
                img = img.resize(
                    (GIF_WIDTH, int(img.height * ratio)),
                    Image.LANCZOS,
                )
                frames.append(
                    img.convert("P", palette=Image.ADAPTIVE, colors=128)
                )

                next_time = start + (i + 1) * interval
                sleep_for = next_time - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            browser.close()

    return frames


def _save_gif(frames: list[Image.Image]) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(1000 / FPS)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )


# ----- Main ---------------------------------------------------------


def main() -> int:
    print(f"[1/5] Staging CSV at {CSV_PATH}")
    _write_csv_header()

    port = _free_port()
    url = f"http://localhost:{port}"
    print(f"[2/5] Launching kt-masterviz on port {port}")
    proc = subprocess.Popen(
        ["kt-masterviz", str(CSV_PATH), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    stop_event = threading.Event()
    try:
        print("[3/5] Waiting for dashboard health")
        if not _wait_healthy(url):
            raise RuntimeError("dashboard did not become healthy within 30s")

        # Brief settle so the first frame shows the initial empty-CSV state.
        time.sleep(1.0)

        print("[4/5] Spawning row-writer + capturing frames")
        writer = threading.Thread(
            target=_append_rows_on_schedule, args=(stop_event,)
        )
        writer.start()

        frames = _capture_frames(url)

        stop_event.set()
        writer.join(timeout=2)

        print(f"[5/5] Stitching {len(frames)} frames into {GIF_PATH}")
        _save_gif(frames)

        size_kb = GIF_PATH.stat().st_size / 1024
        print(f"\nDone. {GIF_PATH} ({size_kb:.0f} KB)")
        return 0
    finally:
        stop_event.set()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Clean up the staging CSV — not the GIF.
        CSV_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

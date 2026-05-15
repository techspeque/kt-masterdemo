# kt-masterlogviz

End-to-end demo of [kt-masterlog](https://github.com/techspeque/kt-masterlog) and [kt-masterviz](https://github.com/techspeque/kt-masterviz) working together — **installed from PyPI**, not from source. This repo's whole point is to validate that the published artifacts work, and to give a try-before-you-buy walkthrough for new users.

## What you get

- A real (small) Fashion-MNIST CNN sweep that tunes five hyperparameters via Bayesian search — about 2–3 minutes on CPU.
- A live Streamlit dashboard that auto-discovers the run via the registry and updates as trials complete.
- A 10-second synthetic smoke test that runs in CI daily to catch breakage from upstream dependency upgrades or yanked releases.

## Quick start

```bash
git clone https://github.com/techspeque/kt-masterlogviz.git
cd kt-masterlogviz
uv sync
```

That pulls `kt-masterlog` and `kt-masterviz` from PyPI into a project-local `.venv/`.

### Run the showcase

**Terminal 1** — start the tuner:

```bash
bash scripts/run.sh
# or directly:
uv run python examples/fashion_mnist_cnn.py
```

**Terminal 2** — open the dashboard against the live run:

```bash
uv run kt-masterviz --latest
```

Open http://localhost:8501. The dashboard:

- Shows the trial summary sorted by `val_accuracy` (the configured objective)
- Plots training curves per trial, switchable across `loss`, `val_loss`, `accuracy`, `val_accuracy`
- Auto-refreshes every 5s (configurable in the sidebar) so new trials show up live

When the tuner finishes, you'll see all 8 trials in the table and curves. The "Switch run" button in the sidebar lets you flip to other runs you've executed previously.

### Other entry points

```bash
uv run kt-masterviz --list                  # tabular listing of every registered run
uv run kt-masterviz                          # in-dashboard picker over all registered runs
uv run kt-masterviz /path/to/master_log.csv  # explicit CSV path (skip the registry)
```

## What the demo tunes

The Fashion-MNIST builder (`examples/fashion_mnist_cnn.py`) exposes five hyperparameters:

| Hyperparameter | Values | Why |
|----------------|--------|-----|
| `filters` | 16, 32, 64 | First-conv channel count; second conv uses `filters * 2` |
| `dense_units` | 64, 128, 256 | FC layer width |
| `dropout` | 0.2 – 0.5 step 0.1 | Regularization sweep |
| `lr` | 1e-2, 1e-3, 3e-4 | Learning rate; meaningful across both optimizers |
| `optimizer` | `adam`, `sgd` | Two qualitatively different optimizers; shows categorical tuning |

Training uses a subsampled split (8k train / 2k val) so each trial finishes in ~15 seconds. Adjust at the bottom of the file if you want a heavier run.

## What this repo is *not*

- **Not a published package.** `pyproject.toml` declares it but it has no build backend; uv just uses it for dependency management.
- **Not a fork** of either kt-masterlog or kt-masterviz. The published artifacts come straight from PyPI; this repo never imports source.
- **Not a test suite** for either package. Each upstream repo has its own. This is a *consumer-side* smoke check.

## CI: daily PyPI smoke test

`.github/workflows/smoke.yml` runs `examples/synthetic_smoke.py` four ways:

- On every push to `main`
- On every PR
- Daily at 08:00 UTC via cron
- Manually from the Actions tab

The smoke test:

1. Installs `kt-masterlog` + `kt-masterviz` from PyPI via `uv sync`
2. Runs a 2-trial / 1-epoch synthetic search through `optimize()`
3. Verifies the master CSV was written
4. Verifies the registry manifest was created and marked `completed`
5. Verifies `kt-masterviz --list` finds the run

The cron run is the most valuable trigger — it catches the case where a transitive dependency change breaks the installed packages even though the kt-masterlog/kt-masterviz code hasn't changed.

## Requirements

- Python 3.12
- `uv` (auto-installs Python and deps)
- ~30 MB to download Fashion-MNIST on first run (cached in `~/.keras/`)

## License

MIT.

# CVI_FlyingHirsch

Computer Vision project scaffold (CVI) — animals in the wild.

## Quickstart

Create a venv and install deps:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or install as a package (gives you the `flyinghirsch` CLI):

```bash
pip install -e .
```

Initialize folders:

```bash
flyinghirsch prepare
flyinghirsch info
```

## Project structure

- `src/flyinghirsch/`: reusable library code (data, models, inference, evaluation)
- `scripts/`: runnable entrypoints (train/predict/…)
- `configs/`: YAML configs
- `data/`: ignored by git (raw/interim/processed/external)
- `artifacts/`: models/checkpoints/features (ignored by git)
- `reports/`: figures/results summaries
- `notebooks/`: experiments

## Running

Train (placeholder):

```bash
python -m scripts.train
```

Predict one image (placeholder):

```bash
python -m scripts.predict path/to/image.jpg
```

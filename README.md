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

Stage/copy your dataset into `data/processed/` (keeps your original dataset untouched) and write a quick integrity report:

```bash
flyinghirsch stage-dataset --src data/raw/dataset
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

Preprocess / stage dataset (copies into `data/processed/` and creates `artifacts/dataset.yaml` for Ultralytics):

```bash
python -m scripts.preprocess
```

Train (YOLOv8 via Ultralytics; saves runs under `artifacts/yolo/`):

```bash
python -m scripts.train
```

Evaluate (Precision/Recall/mAP from Ultralytics):

```bash
python -m scripts.evaluate --split val
python -m scripts.evaluate --split test
```

Predict:

```bash
# Image
python -m scripts.predict path/to/image.jpg

# Video (writes annotated video + detections CSV/JSONL)
python -m scripts.predict path/to/video.mp4 --outdir reports/predictions
```

Analyze movement from a video detections CSV:

```bash
python -m scripts.analyze --detections reports/predictions/video_detections.csv --outdir reports/analysis
```

## Expected outputs

- `python -m scripts.preprocess`
  - `data/processed/dataset/...` (staged YOLO dataset)
  - `artifacts/dataset.yaml` (Ultralytics dataset config)
  - `reports/dataset_stage_report.json` (integrity + class histogram)
- `python -m scripts.train`
  - `artifacts/yolo/run/weights/best.pt` (best checkpoint)
  - `artifacts/yolo/run/weights/last.pt` (last checkpoint)
  - `artifacts/train_metrics.json` (training + validation metrics dump)
- `python -m scripts.predict <video>`
  - `reports/predictions/<name>_annotated.mp4`
  - `reports/predictions/<name>_detections.csv`
  - `reports/predictions/<name>_detections.jsonl`
- `python -m scripts.analyze`
  - `reports/analysis/detections_with_speed.csv`
  - `reports/analysis/track_summary.csv`
  - `reports/analysis/trajectories.png`
  - `reports/analysis/spatial_heatmap.png`
  - `reports/analysis/speed_over_time.png`

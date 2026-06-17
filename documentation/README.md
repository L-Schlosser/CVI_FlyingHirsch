# Flying Hirsch — Setup & Usage

Drone-based thermal wildlife detection and movement analysis.
This guide shows how to set up the project and reproduce the results.

## Requirements

- Python 3.10+
- A CUDA-capable GPU

## Setup

1. **Clone the repository** and open a terminal in the project root.

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r pipeline/requirements.txt
```

4. **Download the dataset** from Zenodo: https://zenodo.org/records/20728879

   Unzip it into `pipeline/datasets/annotated/` so the structure looks like:

```
pipeline/datasets/annotated/
├── images/   (train / val / test)
└── labels/   (train / val / test)
```

> All scripts use paths relative to the `pipeline/` folder, so run every command
> from inside `pipeline/`:
>
> ```bash
> cd pipeline
> ```

## Usage

Run the steps in order. Each script builds on the output of the previous one.

| Step | Command | What it does |
| --- | --- | --- |
| 1. Check dataset | `python 01_check_dataset.py` | Verifies every image has a matching label file. |
| 2. Explore dataset | open `04b_exploration_annotated.ipynb` | Inspects label distribution and box sizes. |
| 3. Train detector | `python 05_train.py` | Trains the YOLO animal detector (settings in `config.py`). |
| 4. Validate | `python 06_validate.py` | Computes metrics (mAP, precision, recall) on the val split. |
| 5. Predict | `python 07_predict.py` | Runs SAHI sliced inference on the test images. |
| 6. Track | `python 08_tracking.py` | Stabilizes frames and tracks animals across a sequence. |
| 7. Visualize | open `09_10_visualizeTracking.ipynb` / `09_276_visualizeTracking.ipynb` | Visualizes trajectories for flight IDs 10 and 276. |

### Notes

- **Configuration:** training and inference settings (model profile, image size,
  weights path, SAHI parameters) live in `pipeline/config.py`.
- **Weights:** the trained model is loaded from `pipeline/best_weights/best.pt`.
  Validation, prediction, and tracking use these weights directly, so you can skip
  step 3 and run steps 4–7 with the provided weights.
- **Outputs:** results are saved under `pipeline/runs/detect/` in `train/`,
  `validate/`, `predict/`, and `track/` subfolders.

## Project Structure

```
pipeline/
├── 01_check_dataset.py        # dataset sanity check
├── 02_convert_labels.py       # remap label classes
├── 03_*.ipynb                 # dataset exploration
├── 04_preprocessing.ipynb     # cleaning & normalization
├── 05_train.py                # train detector
├── 06_validate.py             # evaluate model
├── 07_predict.py              # SAHI prediction
├── 08_tracking.py             # stabilization + tracking
├── 09_*_visualizeTracking.ipynb  # trajectory visualization
├── config.py                  # shared settings
├── data/alfs.yaml             # dataset config
├── best_weights/best.pt       # trained model
└── runs/detect/               # outputs
```

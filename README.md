# Flying Hirsch - Setup & Usage

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

4. **Download the dataset** 

To download and extract the dataset execute: `piepline/01_download.py`

>LINK to dataset: https://zenodo.org/records/20728879


Commands can be run from the project root. Shared paths are configured in `pipeline/config.py`.


## Usage - start from already preprocessed data

These are the steps to reproduce all the outputs / required tasks.

| Step | Command | What it does |
| --- | --- | --- |
| 1. Download dataset | `python pipeline/01_download.py` | Downloads the raw dataset assets. |
| 2. Check dataset | `python pipeline/02_check_dataset.py` | Verifies dataset files and structure. |
| 3. Train detector | `python pipeline/07_train_yolo.py` | Trains the YOLO animal detector and saves the best checkpoint. |
| 4. Validate | `python pipeline/08_validate.py` | Computes metrics on the validation split. |
| 5. Predict | `python pipeline/09_predict.py` | Runs SAHI sliced inference on the processed thermal test images. |
| 6. Visualize predictions | open `pipeline/09b_visualizePrediction.ipynb` | Samples and displays prediction outputs. |
| 7. Track | `python pipeline/10_tracking.py` | Stabilizes frames and tracks animals across a sequence. |
| 8. Visualize tracking | open `pipeline/10b_10_visualizeTracking.ipynb` / `pipeline/10b_276_visualizeTracking.ipynb` | Visualizes trajectories for flight IDs 10 and 276. |

## Usage - start from zero:

These are the steps if you want to start the project from zero. So also download the original unpreprocessed dataset and do all the steps.

Run the steps in order. Each script builds on the output of the previous one.

| Step | Command | What it does |
| --- | --- | --- |
| 1. Download dataset | `python pipeline/01_download.py` | Downloads the raw dataset assets. |
| 2. Check dataset | `python pipeline/02_check_dataset.py` | Verifies dataset files and structure. |
| 3. Convert labels | `python pipeline/03_convert_labels.py` | Converts labels into the project format. |
| 4. Copy one-class data | `python pipeline/03b_copy_data_one_class.py` | Prepares the single-class dataset variant used for training. |
| 5. Explore and preprocess | open `pipeline/04_exploration.ipynb`, `pipeline/04_exploration_ALFS.ipynb`, `pipeline/05_preprocessing.ipynb`, `pipeline/06_exploration_annotated.ipynb` | Inspect data quality, class distribution, and preprocessing results. |
| 6. Train detector | `python pipeline/07_train_yolo.py` | Trains the YOLO animal detector and saves the best checkpoint. |
| 7. Validate | `python pipeline/08_validate.py` | Computes metrics on the validation split. |
| 8. Predict | `python pipeline/09_predict.py` | Runs SAHI sliced inference on the processed thermal test images. |
| 9. Visualize predictions | open `pipeline/09b_visualizePrediction.ipynb` | Samples and displays prediction outputs. |
| 10. Track | `python pipeline/10_tracking.py` | Stabilizes frames and tracks animals across a sequence. |
| 11. Visualize tracking | open `pipeline/10b_10_visualizeTracking.ipynb` / `pipeline/10b_276_visualizeTracking.ipynb` | Visualizes trajectories for flight IDs 10 and 276. |



## Notes

- **Configuration:** training and inference settings live in `pipeline/config.py`.
- **Weights:** the trained model is loaded from `pipeline/best_weights/best.pt`.
  Validation, prediction, and tracking use these weights directly, so you can skip
  training and run those later steps with the provided weights.
- **Outputs:** results are saved under `pipeline/runs/detect/`, including
  training, validation, prediction, and tracking outputs.

## Project Structure

```text
pipeline/
├── 01_download.py
├── 02_check_dataset.py
├── 03_convert_labels.py
├── 03b_copy_data_one_class.py
├── 04_exploration.ipynb
├── 04_exploration_ALFS.ipynb
├── 05_preprocessing.ipynb
├── 06_exploration_annotated.ipynb
├── 07_train_yolo.py
├── 08_validate.py
├── 09_predict.py
├── 09b_visualizePrediction.ipynb
├── 10_tracking.py
├── 10b_10_visualizeTracking.ipynb
├── 10b_276_visualizeTracking.ipynb
├── config.py
├── best_weights/
│   └── best.pt
└── runs/
    └── detect/
```

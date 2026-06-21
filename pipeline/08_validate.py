from ultralytics import YOLO

from config import BEST_WEIGHTS, MODEL_NAME, BEST_WEIGHTS_DIR
from config import DATA_YAML, VAL_BATCH, VAL_IMGSZ
from config import VALIDATION_RUNS_DIR


def main():
    model = YOLO(BEST_WEIGHTS)

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=VAL_IMGSZ,
        batch=VAL_BATCH,
        device=0,
        project=str(VALIDATION_RUNS_DIR),
        name=f"validation_{MODEL_NAME}",
        conf=0.001
    )

    print(metrics)
    #save metrics to file
    with open(BEST_WEIGHTS_DIR / "metrics.txt", "w") as f:
        f.write(str(metrics))
    print("Done.")


if __name__ == "__main__":
    main()

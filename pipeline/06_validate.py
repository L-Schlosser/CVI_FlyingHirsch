from ultralytics import YOLO

from config import DATA_YAML, VAL_BATCH, VAL_IMGSZ, best_weights, run_name
from config import BEST_WEIGHTS, MODEL_NAME


def main():
    model = YOLO(BEST_WEIGHTS)

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=VAL_IMGSZ,
        batch=VAL_BATCH,
        device=0,
        project="validate",
        name=f"validation_{MODEL_NAME}",
        conf=0.001
    )

    print(metrics)
    #save metrics to file
    with open(f"runs/detect/validate/validation_{MODEL_NAME}/metrics.txt", "w") as f:
        f.write(str(metrics))
    print("Done.")


if __name__ == "__main__":
    main()

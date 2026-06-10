from ultralytics import YOLO

from config import DATA_YAML, VAL_BATCH, VAL_IMGSZ, best_weights, run_name


def main():
    model = YOLO(best_weights())

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=VAL_IMGSZ,
        batch=VAL_BATCH,
        device=0,
        project="validate",
        name=f"validation_{run_name()}",
    )

    print(metrics)
    print("Done.")


if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():
    # load your trained model
    model = YOLO("runs/train/weights/best.pt")

    # run validation
    metrics = model.val(
        data="data/alfs.yaml",
        split="val",      # use validation split
        imgsz=640,
        batch=16,
        device="cuda",
        project="runs/train/validate"
    )

    print(metrics)  # prints mAP, precision, recall, etc.
    print("Done.")

if __name__ == "__main__":
    main()
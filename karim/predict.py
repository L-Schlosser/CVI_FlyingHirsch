from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    model.predict(
        source="datasets/images/test",
        conf=0.25,
        save=True,
        project="runs/detect/predict"
    )

    print("Done.")

if __name__ == "__main__":
    main()
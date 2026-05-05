from ultralytics import YOLO

def main():
    # start from pretrained model
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/alfs.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",   # use "cpu" if no GPU
        workers=4
    )

if __name__ == "__main__":
    main()
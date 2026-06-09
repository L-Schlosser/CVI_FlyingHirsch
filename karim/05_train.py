from ultralytics import YOLO

def main():
    # start from pretrained model
    model_name = "yolo11m_pt"
    model = YOLO("yolo11m.pt")
    status = "preprocessed"

    # model.train(

    #     data="data/alfs.yaml",
    #     epochs=50,
    #     imgsz=768, #could make to 1024 
    #     batch=8,   #maybe 8
    #     device="cuda",   # use "cpu" if no GPU
    #     workers=12,
    #     project="runs/detect/train",
    #     mosaic=0.3,
    #     mixup=0.0,
    #     lr0=0.001
    # )



    model.train(
        data="data/alfs.yaml",

        epochs=50,
        patience=25,

        imgsz=1024,
        batch=8,
        device="cuda",
        workers=12,
        lr0=0.0007,
        weight_decay=0.0005,
        # warmup_epochs=5,

        # --- augmentation (thermal-specific tuning) ---
        mosaic=0.15,        # keep low, thermal breaks with mosaic
        mixup=0.0,          # disable
        copy_paste=0.1,     # helps small object density
        degrees=10,
        translate=0.05,
        scale=0.4,
        fliplr=0.5,

        # IMPORTANT for thermal:
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # --- logging / output ---
        # project="runs/train",
        name=model_name + "_" + status,
        save=True,
        cache=True
    )

if __name__ == "__main__":
    main()
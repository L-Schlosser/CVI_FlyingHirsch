from ultralytics import YOLO

def main():
    # start from pretrained model
    model_name = "yolo11l_pt"
    model = YOLO("yolo11l.pt")
    status = "yolo11l_ChatProcessed"

    # model.train(
    #     data="data/alfs.yaml",

    #     epochs=50,
    #     patience=15, #25

    #     imgsz=1024,
    #     batch=4,
    #     device="cuda",
    #     workers=8,
    #     lr0=0.0003,
    #     weight_decay=0.0005,


    #     mosaic=0.1,
    #     copy_paste=0.05,
    #     scale=0.25,
    #     degrees=5,
    #     translate=0.02,

    #     # IMPORTANT for thermal:
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,

    #     # --- logging / output ---
    #     project="train",
    #     name=model_name + "_" + status,
    #     save=True,
    #     cache="disk"
    # )

    model.train(

        # -----------------------
        # DATA
        # -----------------------
        data="data/alfs.yaml",

        # -----------------------
        # TRAINING LENGTH
        # -----------------------
        epochs=200,
        patience=50,

        # -----------------------
        # INPUT SIZE (IMPORTANT)
        # -----------------------
        imgsz=1024,

        # -----------------------
        # HARDWARE
        # -----------------------
        batch=5,
        workers=8,
        device=0,
        amp=True,
        cache=True,

        # -----------------------
        # OPTIMIZER
        # -----------------------
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=5e-4,
        cos_lr=True,

        # -----------------------
        # GEOMETRIC AUGMENTATION
        # -----------------------
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,

        # -----------------------
        # YOLO ADVANCED AUGMENTATION
        # -----------------------
        mosaic=0.5,
        close_mosaic=20,
        mixup=0.1,
        copy_paste=0.3,
        erasing=0.2,

        # -----------------------
        # COLOR AUGMENTATION
        # (disabled because grayscale/thermal)
        # -----------------------
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # --- logging / output ---
        project="train",
        name=model_name + "_" + status,
        save=True
)

if __name__ == "__main__":
    main()
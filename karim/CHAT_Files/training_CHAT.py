from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(

    # -----------------------
    # DATA
    # -----------------------
    data="wildlife.yaml",

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
    batch=12,
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
    warmup_epochs=5,

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
    hsv_v=0.0
)
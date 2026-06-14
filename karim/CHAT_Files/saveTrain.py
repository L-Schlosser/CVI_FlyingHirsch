from ultralytics import YOLO

def main():
    model_name = "yolo26l_pt"
    model = YOLO("yolo26l.pt")
    status = "annotated"

    model.train(
        data="data/alfs.yaml",

        epochs=100,
        patience=35, 

        imgsz=1024,
        batch=6,
        device="cuda",
        workers=8,
        optimizer="AdamW",
        weight_decay=0.0005,
        lr0=0.005,
        lrf=0.1,

        momentum=0.937,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        #Augmentation
        mosaic=0.8,
        close_mosaic=15,
        copy_paste=0.05,
        scale=0.5,
        degrees=20,
        translate=0.1,

        # thermal:
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.5,

        # logging / output
        project="train",
        name=model_name + "_" + status,
        save=True,
        cache="disk"
    )

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
    
    # def build_train_args(cfg, *, lr0=0.001, epochs=None, name=None):
#     """Shared training kwargs for fresh, resume, and phase-2 runs."""
#     return dict(
#         data="data/alfs.yaml",
#         epochs=epochs or cfg["epochs"],
#         patience=cfg["patience"],
#         imgsz=cfg["imgsz"],
#         batch=cfg["batch"],
#         device=0,
#         workers=4,
#         amp=True,
#         cache="disk",
#         optimizer="AdamW",
#         lr0=lr0,
#         lrf=0.1,
#         weight_decay=5e-4,
#         cos_lr=True,
#         # warmup_epochs=3,
#         degrees=cfg["degrees"],
#         translate=cfg["translate"],
#         scale=cfg["scale"],
#         fliplr=0.5,
#         flipud=0.0,
#         mosaic=cfg["mosaic"],
#         close_mosaic=cfg["close_mosaic"],
#         mixup=cfg["mixup"],
#         copy_paste=cfg["copy_paste"],
#         erasing=cfg["erasing"],
#         hsv_h=0.0,
#         hsv_s=0.0,
#         hsv_v=0.0,
#         auto_augment=None,
#         plots=TRAIN_PLOTS,
#         project="train",
#         name=name or run_name(),
#         save=True,
#     )

if __name__ == "__main__":
    main()
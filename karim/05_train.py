from ultralytics import YOLO

from config import PROFILE, PROFILES, TRAIN_PLOTS, best_weights, run_name

# ---------------------------------------------------------------------------
# YOLO26 profiles (RTX 5070 Laptop, 8 GB VRAM, ~5240 train images)
#
# YOLO26 uses STAL for small-object supervision – better for tiny thermal deer.
# Train at 640-800, validate/predict at 1024 with SAHI (see 06/07 scripts).
#
# "fast"     ~3-6 min/epoch
# "balanced" ~4-8 min/epoch  (recommended)
# "quality"  ~15-25 min/epoch
# ---------------------------------------------------------------------------


def build_train_args(cfg, *, lr0=1e-4, epochs=None, name=None):
    """Shared training kwargs for fresh, resume, and phase-2 runs."""
    return dict(
        data="data/alfs.yaml",
        epochs=epochs or cfg["epochs"],
        patience=cfg["patience"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,
        workers=4,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=lr0,
        weight_decay=5e-4,
        cos_lr=True,
        warmup_epochs=3,
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=0.5,
        flipud=0.0,
        mosaic=cfg["mosaic"],
        close_mosaic=cfg["close_mosaic"],
        mixup=cfg["mixup"],
        copy_paste=cfg["copy_paste"],
        erasing=cfg["erasing"],
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        auto_augment=None,
        plots=TRAIN_PLOTS,
        project="train",
        name=name or run_name(),
        save=True,
    )


def main():
    cfg = PROFILES[PROFILE]

    # -----------------------------------------------------------------------
    # DEFAULT: train from pretrained COCO weights (yolo26s/m/l.pt)
    # -----------------------------------------------------------------------
    model = YOLO(cfg["weights"])
    model.train(**build_train_args(cfg))

    # -----------------------------------------------------------------------
    # RESUME: continue the same run after crash, Ctrl+C, or early stop
    # Uses last.pt + restores optimizer state and epoch counter.
    # Uncomment below and comment out the DEFAULT block above.
    # -----------------------------------------------------------------------
    # model = YOLO(f"runs/detect/train/{run_name()}/weights/last.pt")
    # model.train(resume=True)

    # -----------------------------------------------------------------------
    # PHASE 2: continue from best.pt after early stop (recommended)
    # Lower LR helps when mAP plateaued. Saves to a new run folder.
    # Uncomment below and comment out the DEFAULT block above.
    # -----------------------------------------------------------------------
    # model = YOLO(best_weights())
    # model.train(
    #     **build_train_args(
    #         cfg,
    #         lr0=5e-5,       # half of fresh run; try 1e-5 if still plateauing
    #         epochs=60,      # extra epochs for phase 2
    #         name=f"{run_name()}_phase2",
    #     )
    # )


if __name__ == "__main__":
    main()

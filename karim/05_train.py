from xml.parsers.expat import model
import torch
import torch.nn as nn


from ultralytics import YOLO

from config import PROFILE, PROFILES, TRAIN_PLOTS, best_weights, run_name
from config import BEST_WEIGHTS, MODEL_NAME

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

def build_train_args(name_addition:str|None = None):
    return dict(
        data="data/alfs.yaml",
        epochs=100,
        patience=50,
        imgsz=1024,
        batch=4,
        device=0,              # equivalent to "cuda"
        workers=4,
        amp=True,

        optimizer="AdamW",
        lr0=0.005,
        lrf=0.1,
        weight_decay=0.0005,

        momentum=0.937,
        warmup_epochs=3,
        warmup_momentum=0.8,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        augment=True,

        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        val=True,
        plots=False,
        verbose=True,

        save_period=10,
        pretrained=True,

        project="train",
        name="yolo26l_annotated",
        save=True,
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


def main():
    cfg = PROFILES[PROFILE]

    # -----------------------------------------------------------------------
    # DEFAULT: train from pretrained COCO weights (yolo26s/m/l.pt)
    # -----------------------------------------------------------------------
    # model = YOLO("yolo26l.pt")
    # model.train(**build_train_args())

    # -----------------------------------------------------------------------
    # RESUME: use last:
    # -----------------------------------------------------------------------
    # model = YOLO(f"runs/detect/train/{run_name()}/weights/last.pt")
    # model.train(resume=True)

    # -----------------------------------------------------------------------
    # PHASE x: use best:
    # -----------------------------------------------------------------------
    # model = YOLO("runs/detect/train/yolo26l_annotated/weights/best.pt")
    model = YOLO(BEST_WEIGHTS)
    # model = YOLO(best_weights())
    model.train(**build_train_args())
        #     cfg,
        #     # lr0=5e-5,       # half of fresh run; try 1e-5 if still plateauing
        #     # epochs=60,      # extra epochs for phase 2
        #     name=MODEL_NAME
        # )

    # -----------------------------------------------------------------------
    # ADAPTION: adapt first layer from 3 to 1 channel (thermal)
    # -----------------------------------------------------------------------

    # model = YOLO("yolo26l.pt")

    # #try out:
    # first = model.model.model[0]
    # print(first.conv)

    # new_conv = nn.Conv2d(
    #     in_channels=1,
    #     out_channels=first.conv.out_channels,
    #     kernel_size=first.conv.kernel_size,
    #     stride=first.conv.stride,
    #     padding=first.conv.padding,
    #     bias=first.conv.bias is not None
    # ).to(first.conv.weight.device)

    # print(new_conv)

    
    # with torch.no_grad():
    #     new_conv.weight.copy_(
    #         first.conv.weight.mean(dim=1, keepdim=True)
    #     )

    # model.model.model[0].conv = new_conv
    # print(model.model.model[0].conv.in_channels)

    # model.train(**build_train_args())



if __name__ == "__main__":
    main()

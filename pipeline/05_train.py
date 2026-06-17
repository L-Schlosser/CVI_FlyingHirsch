from xml.parsers.expat import model
import torch
import torch.nn as nn


from ultralytics import YOLO

from config import PROFILE, PROFILES, TRAIN_PLOTS, best_weights, run_name
from config import BEST_WEIGHTS, MODEL_NAME


def build_train_args(name_addition:str|None = None):
    return dict(
        data="data/alfs.yaml",
        epochs=100,
        patience=30,
        imgsz=1024,
        batch=4,
        device=0,              # equivalent to "cuda"
        workers=8,
        amp=True,

        optimizer="AdamW",
        lr0=0.005,
        lrf=0.1,
        weight_decay=0.0005,

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


        val=True,
        plots=True,
        verbose=True,

        save_period=10,
        project="train",
        name="yolo26l_annotated" + (f"_{name_addition}" if name_addition else ""),
        save=True,
    )




def main():
    # kept for reference -> used optimized arguments in the end
    cfg = PROFILES[PROFILE]

    # -----------------------------------------------------------------------
    # DEFAULT: train from pretrained COCO weights (yolo26s/m/l.pt)
    # -----------------------------------------------------------------------
    # model = YOLO("yolo26l.pt")
    # model.train(**build_train_args())
    # -----------------------------------------------------------------------


    # PHASE x: use best:
    # -----------------------------------------------------------------------
    # model = YOLO(BEST_WEIGHTS)
    # model.train(**build_train_args("phase3"))

    # -----------------------------------------------------------------------
    # ADAPTION: adapt first layer from 3 to 1 channel (thermal)
    # -----------------------------------------------------------------------

    model = YOLO("yolo26l.pt")

    first = model.model.model[0]
    print(first.conv)

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first.conv.out_channels,
        kernel_size=first.conv.kernel_size,
        stride=first.conv.stride,
        padding=first.conv.padding,
        bias=first.conv.bias is not None
    ).to(first.conv.weight.device)

    print(new_conv)

    
    with torch.no_grad():
        new_conv.weight.copy_(
            first.conv.weight.mean(dim=1, keepdim=True)
        )

    model.model.model[0].conv = new_conv
    print(model.model.model[0].conv.in_channels)

    model.train(**build_train_args())



if __name__ == "__main__":
    main()

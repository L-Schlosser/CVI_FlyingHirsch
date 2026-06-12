from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

from config import best_weights, TEST_SOURCE, run_name

TRACK_NUMBER = "282"
TEST_IMGES = "datasets/processed/images/track_test" + "/" + TRACK_NUMBER

def main():
    weights = best_weights()

    model = YOLO(weights)

    source = Path(TEST_IMGES)
    image_paths = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))

    if not image_paths:
        raise FileNotFoundError("No images found")

    out_dir = Path("runs/detect/track") / f"track_{run_name()}_{TRACK_NUMBER}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = {}  # id -> list of (x,y)

    print(f"Tracking {len(image_paths)} images")

    # IMPORTANT: stream=True keeps temporal order
    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.05,  #0.25
        iou=0.25,    #0.5
        save=False
    )

    for i, r in enumerate(results):
        frame = r.orig_img.copy()

        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for obj_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                obj_id = int(obj_id)

                if obj_id not in trajectories:
                    trajectories[obj_id] = []

                trajectories[obj_id].append((cx, cy))

                # draw box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)
                cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        out_path = out_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(out_path), frame)

    # save trajectories
    traj_file = out_dir / "trajectories.txt"
    with open(traj_file, "w") as f:
        for obj_id, points in trajectories.items():
            f.write(f"ID {obj_id}: {points}\n")

    print(f"Done. Output saved to {out_dir}")


if __name__ == "__main__":
    main()
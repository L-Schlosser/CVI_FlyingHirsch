from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from flyinghirsch.tracking.bytetrack import ByteTrackWrapper, TrackConfig


@dataclass(frozen=True)
class VideoInferenceConfig:
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    fps: float | None = None  # if None, read from input


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def run_video_detection_and_tracking(
    cfg: dict[str, Any],
    *,
    weights: Path,
    input_path: Path,
    output_video_path: Path,
    output_jsonl_path: Path,
    output_csv_path: Path,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Runs YOLO detection + ByteTrack tracking and writes:
      - annotated video (mp4)
      - per-detection JSONL
      - per-detection CSV

    JSONL schema:
      {"frame": int, "track_id": int, "class_id": int, "class_name": str,
       "confidence": float, "x1":float,"y1":float,"x2":float,"y2":float}
    """
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
        ) from e

    vinf = VideoInferenceConfig(
        imgsz=int(cfg.get("yolo", {}).get("imgsz", 640)),
        conf=float(cfg.get("inference", {}).get("score_threshold", 0.25)),
        iou=float(cfg.get("inference", {}).get("iou_threshold", 0.7)),
        max_det=int(cfg.get("inference", {}).get("max_det", 300)),
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(vinf.fps or in_fps or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _ensure_parent(output_video_path)
    _ensure_parent(output_jsonl_path)
    _ensure_parent(output_csv_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer: {output_video_path}")

    model = YOLO(str(weights))

    # tracker tuned to video fps
    tracker = ByteTrackWrapper(TrackConfig(frame_rate=int(round(fps))))

    # Try to use YOLO's names mapping
    names_map = getattr(model, "names", None) or {}
    if class_names:
        names_map = {**names_map, **class_names}

    rows: list[dict[str, Any]] = []
    jsonl = output_jsonl_path.open("w", encoding="utf-8")
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = model.predict(
                source=frame,
                imgsz=vinf.imgsz,
                conf=vinf.conf,
                iou=vinf.iou,
                max_det=vinf.max_det,
                device=cfg.get("yolo", {}).get("device", None),
                verbose=False,
            )[0]

            if res.boxes is None or len(res.boxes) == 0:
                writer.write(frame)
                frame_idx += 1
                continue

            xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confidence = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
            class_id = res.boxes.cls.detach().cpu().numpy().astype(int)

            tracked = tracker.update(xyxy=xyxy, confidence=confidence, class_id=class_id)
            t_xyxy = tracked.xyxy
            t_conf = tracked.confidence
            t_cls = tracked.class_id
            t_tid = tracked.tracker_id

            # Draw
            for bb, conf, cid, tid in zip(t_xyxy, t_conf, t_cls, t_tid):
                x1, y1, x2, y2 = [int(v) for v in bb.tolist()]
                label = str(names_map.get(int(cid), f"class_{int(cid)}"))
                text = f"ID {int(tid)} | {label} {float(conf):.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

                rec = {
                    "frame": int(frame_idx),
                    "track_id": int(tid),
                    "class_id": int(cid),
                    "class_name": label,
                    "confidence": float(conf),
                    "x1": float(bb[0]),
                    "y1": float(bb[1]),
                    "x2": float(bb[2]),
                    "y2": float(bb[3]),
                }
                rows.append(rec)
                jsonl.write(json.dumps(rec) + "\n")

            writer.write(frame)
            frame_idx += 1
    finally:
        jsonl.close()
        writer.release()
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return {
        "frames": frame_idx,
        "fps": fps,
        "output_video": str(output_video_path),
        "output_jsonl": str(output_jsonl_path),
        "output_csv": str(output_csv_path),
    }


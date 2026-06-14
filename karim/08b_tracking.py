"""Drone thermal tracking with camera-motion compensation.

Pipeline: load images -> stabilize to reference frame -> YOLO+ByteTrack -> save outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import BEST_WEIGHTS, MODEL_NAME

TRACK_NUMBER = "276_3"
TEST_IMAGES = f"datasets/processed/images/track_test/{TRACK_NUMBER}"
REFERENCE_FRAME = "276_5093.jpg"  # middle frame; set None to use first image

MIN_MATCHES = 20
RANSAC_THRESH = 3.0
ORB_FEATURES = 5000
SMOOTH_WINDOW = 5
MIN_TRACK_POINTS = 1
TRAIL_LENGTH = 30
VIDEO_FPS = 3


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def preprocess_for_registration(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_orb_features(gray: np.ndarray, orb: cv2.ORB) -> tuple[list, np.ndarray | None]:
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(
    des_ref: np.ndarray,
    des_cur: np.ndarray,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des_cur, des_ref, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def estimate_transform(
    pts_cur: np.ndarray,
    pts_ref: np.ndarray,
    use_homography: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(pts_cur) < 4:
        return None, None

    if use_homography:
        H, inliers = cv2.findHomography(
            pts_cur,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESH,
        )
        if H is None:
            return None, None
        return H, inliers

    M, inliers = cv2.estimateAffinePartial2D(
        pts_cur,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
    )
    return M, inliers


def affine_to_3x3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def compose_affine(M_ref: np.ndarray, M_step: np.ndarray) -> np.ndarray:
    combined = affine_to_3x3(M_ref) @ affine_to_3x3(M_step)
    return combined[:2, :]


def decompose_affine_partial(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b, tx, ty = M[0, 0], M[1, 0], M[0, 2], M[1, 2]
    scale = float(np.hypot(a, b))
    angle = float(np.arctan2(b, a))
    return tx, ty, angle, scale


def rebuild_affine_partial(tx: float, ty: float, angle: float, scale: float) -> np.ndarray:
    cos_a = np.cos(angle) * scale
    sin_a = np.sin(angle) * scale
    return np.array(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
        dtype=np.float32,
    )


def smooth_transforms(
    transforms: list[np.ndarray | None],
    window: int = SMOOTH_WINDOW,
) -> list[np.ndarray]:
    params = []
    for M in transforms:
        if M is None:
            params.append(None)
        else:
            params.append(decompose_affine_partial(M))

    smoothed: list[np.ndarray] = []
    for idx, param in enumerate(params):
        if param is None:
            smoothed.append(smoothed[-1] if smoothed else np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            continue

        neighbors = [
            params[j]
            for j in range(max(0, idx - window // 2), min(len(params), idx + window // 2 + 1))
            if params[j] is not None
        ]
        tx = float(np.median([p[0] for p in neighbors]))
        ty = float(np.median([p[1] for p in neighbors]))
        angle = float(np.median([p[2] for p in neighbors]))
        scale = float(np.median([p[3] for p in neighbors]))
        smoothed.append(rebuild_affine_partial(tx, ty, angle, scale))

    return smoothed


def warp_frame(
    frame: np.ndarray,
    transform: np.ndarray,
    shape: tuple[int, int],
    is_homography: bool = False,
) -> np.ndarray:
    height, width = shape[:2]
    if is_homography:
        return cv2.warpPerspective(
            frame,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_frame_transform(
    gray_ref: np.ndarray,
    gray_cur: np.ndarray,
    orb: cv2.ORB,
) -> np.ndarray | None:
    kp_ref, des_ref = detect_orb_features(gray_ref, orb)
    kp_cur, des_cur = detect_orb_features(gray_cur, orb)

    if des_ref is None or des_cur is None or len(kp_ref) < MIN_MATCHES or len(kp_cur) < MIN_MATCHES:
        return None

    matches = match_features(des_ref, des_cur)
    if len(matches) < MIN_MATCHES:
        return None

    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = estimate_transform(pts_cur, pts_ref, use_homography=False)
    return M


def resolve_reference_index(image_paths: list[Path], reference_name: str | None) -> int:
    if not reference_name:
        return 0

    for idx, path in enumerate(image_paths):
        if path.name == reference_name:
            return idx

    raise FileNotFoundError(
        f"Reference frame '{reference_name}' not found in sequence. "
        f"Available: {[p.name for p in image_paths[:5]]}..."
    )


def clear_stale_frames(directory: Path) -> None:
    if not directory.exists():
        return
    for pattern in ("*.jpg", "*.png"):
        for path in directory.glob(pattern):
            path.unlink()


def stabilize_sequence(
    image_paths: list[Path],
    output_dir: Path,
    reference_name: str | None = REFERENCE_FRAME,
) -> tuple[list[Path], int, str]:
    stabilized_dir = output_dir / "stabilized_frames"
    stabilized_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_frames(stabilized_dir)

    ref_idx = resolve_reference_index(image_paths, reference_name)
    ref_path = image_paths[ref_idx]

    orb = cv2.ORB_create(ORB_FEATURES)
    reference = cv2.imread(str(ref_path))
    if reference is None:
        raise FileNotFoundError(f"Could not read reference frame: {ref_path}")

    ref_shape = reference.shape
    ref_gray = preprocess_for_registration(to_gray(reference))
    cv2.imwrite(str(output_dir / "reference_frame.jpg"), reference)

    raw_to_ref: list[np.ndarray | None] = []
    print(f"Stabilizing {len(image_paths)} frames to reference: {ref_path.name} (index {ref_idx})")

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: skipped unreadable frame {image_path}")
            raw_to_ref.append(None)
            continue

        if idx == ref_idx:
            raw_to_ref.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            continue

        gray = preprocess_for_registration(to_gray(frame))
        frame_to_ref = estimate_frame_transform(ref_gray, gray, orb)

        if frame_to_ref is None:
            print(f"Warning: feature-poor frame {idx} ({image_path.name}), reusing neighbor transform")
            fallback = raw_to_ref[-1] if raw_to_ref else np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            raw_to_ref.append(fallback)
        else:
            raw_to_ref.append(frame_to_ref)

    smoothed_transforms = smooth_transforms(raw_to_ref)
    stabilized_paths: list[Path] = []

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        if idx == ref_idx:
            stabilized = reference.copy()
        else:
            stabilized = warp_frame(frame, smoothed_transforms[idx], ref_shape)

        out_path = stabilized_dir / f"{idx:05d}.jpg"
        cv2.imwrite(str(out_path), stabilized)
        stabilized_paths.append(out_path)
        print(f"  stabilized {idx + 1}/{len(image_paths)}")

    return stabilized_paths, ref_idx, ref_path.name


def compute_trajectories(
    results,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, dict[int, tuple[float, float]]]]:
    trajectories: dict[int, list[tuple[float, float]]] = {}
    frame_tracks: dict[int, dict[int, tuple[float, float]]] = {}

    for frame_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id
        track_ids = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1)

        frame_tracks[frame_idx] = {}
        for obj_id, box in zip(track_ids, boxes):
            if obj_id < 0:
                continue

            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            frame_tracks[frame_idx][obj_id] = (cx, cy)

    all_ids = sorted({obj_id for tracks in frame_tracks.values() for obj_id in tracks})

    for obj_id in all_ids:
        points: list[tuple[float, float]] = []
        for frame_idx in sorted(frame_tracks):
            if obj_id in frame_tracks[frame_idx]:
                points.append(frame_tracks[frame_idx][obj_id])
        if len(points) >= MIN_TRACK_POINTS:
            trajectories[obj_id] = points

    return trajectories, frame_tracks


def _track_color(obj_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(obj_id)
    return tuple(int(v) for v in rng.integers(40, 255, size=3))


def draw_tracks(
    frame: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    histories: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    output = frame.copy()

    for obj_id, box in zip(track_ids, boxes):
        if obj_id < 0:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 1)
            continue

        color = _track_color(int(obj_id))
        x1, y1, x2, y2 = box.astype(int)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        history = histories.setdefault(int(obj_id), [])
        history.append((cx, cy))
        if len(history) > TRAIL_LENGTH:
            history.pop(0)

        for j in range(1, len(history)):
            cv2.line(output, history[j - 1], history[j], color, 2)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (cx, cy), 4, color, -1)
        cv2.putText(
            output,
            f"ID {int(obj_id)}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def save_trajectories(
    trajectories: dict[int, list[tuple[float, float]]],
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    output_dir: Path,
    num_frames: int,
    reference_index: int,
    reference_name: str,
) -> None:
    txt_path = output_dir / "trajectories.txt"
    with open(txt_path, "w", encoding="utf-8") as handle:
        for obj_id, points in sorted(trajectories.items()):
            handle.write(f"ID {obj_id}: {points}\n")

    json_path = output_dir / "trajectories.json"
    payload = {
        "coordinate_system": f"stabilized_reference_{reference_name}",
        "reference_frame_index": reference_index,
        "reference_frame_name": reference_name,
        "num_frames": num_frames,
        "trajectories": {
            str(obj_id): [[float(x), float(y)] for x, y in points]
            for obj_id, points in sorted(trajectories.items())
        },
        "frame_tracks": {
            str(frame_idx): {
                str(obj_id): [float(x), float(y)]
                for obj_id, (x, y) in tracks.items()
            }
            for frame_idx, tracks in sorted(frame_tracks.items())
        },
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def create_tracking_video(
    overlay_frames: list[np.ndarray],
    output_path: Path,
    fps: int = VIDEO_FPS,
) -> None:
    if not overlay_frames:
        return

    height, width = overlay_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in overlay_frames:
        writer.write(frame)

    writer.release()


def create_stabilized_trajectory_video(
    stabilized_dir: Path,
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    output_path: Path,
    fps: int = VIDEO_FPS,
    trail_length: int = TRAIL_LENGTH,
    num_frames: int | None = None,
) -> None:
    image_paths = sorted(stabilized_dir.glob("*.jpg")) + sorted(stabilized_dir.glob("*.png"))
    if num_frames is not None:
        image_paths = image_paths[:num_frames]
    if not image_paths:
        return

    histories: dict[int, list[tuple[int, int]]] = {}
    frames_out: list[np.ndarray] = []

    for frame_idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        tracks = frame_tracks.get(frame_idx, {})
        annotated = frame.copy()

        for obj_id, (x, y) in tracks.items():
            point = (int(round(x)), int(round(y)))
            history = histories.setdefault(int(obj_id), [])
            history.append(point)
            if len(history) > trail_length:
                history.pop(0)

        for obj_id, history in histories.items():
            color = _track_color(obj_id)
            for j in range(1, len(history)):
                cv2.line(annotated, history[j - 1], history[j], color, 2)
            if obj_id in tracks:
                cx, cy = int(round(tracks[obj_id][0])), int(round(tracks[obj_id][1]))
                cv2.circle(annotated, (cx, cy), 5, color, -1)
                cv2.putText(
                    annotated,
                    f"ID {obj_id}",
                    (cx + 6, max(20, cy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            annotated,
            f"Frame {frame_idx + 1}/{len(image_paths)}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames_out.append(annotated)

    create_tracking_video(frames_out, output_path, fps)


def run_tracking(
    model: YOLO,
    stabilized_paths: list[Path],
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[int, dict[int, tuple[float, float]]],
    list[np.ndarray],
]:
    print(f"Running YOLO + ByteTrack on {len(stabilized_paths)} stabilized frames...")

    trajectories: dict[int, list[tuple[float, float]]] = {}
    overlay_frames: list[np.ndarray] = []
    histories: dict[int, list[tuple[int, int]]] = {}

    results = model.track(
        source=[str(path) for path in stabilized_paths],
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.05,
        iou=0.25,
        save=False,
    )

    frame_results = list(results)
    trajectories, frame_tracks = compute_trajectories(frame_results)

    for frame_idx, result in enumerate(frame_results):
        frame = result.orig_img.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id
            track_ids = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1)
            frame = draw_tracks(frame, boxes, track_ids, histories)

        overlay_frames.append(frame)
        print(f"  tracked {frame_idx + 1}/{len(frame_results)}")

    return trajectories, frame_tracks, overlay_frames


def main() -> None:
    source = Path(TEST_IMAGES)
    image_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")

    output_dir = Path("runs/detect/track") / f"track_{MODEL_NAME}_{TRACK_NUMBER}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stabilized_paths, ref_idx, ref_name = stabilize_sequence(image_paths, output_dir)

    model = YOLO(BEST_WEIGHTS)
    trajectories, frame_tracks, overlay_frames = run_tracking(model, stabilized_paths)

    overlay_dir = output_dir / "tracking_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_frames(overlay_dir)
    for idx, frame in enumerate(overlay_frames):
        cv2.imwrite(str(overlay_dir / f"{idx:05d}.jpg"), frame)

    create_tracking_video(overlay_frames, output_dir / "tracking_overlay_video.mp4")
    create_stabilized_trajectory_video(
        output_dir / "stabilized_frames",
        frame_tracks,
        output_dir / "stabilized_trajectory_video.mp4",
        num_frames=len(stabilized_paths),
    )
    save_trajectories(
        trajectories,
        frame_tracks,
        output_dir,
        len(image_paths),
        ref_idx,
        ref_name,
    )

    print()
    print("Done.")
    print(f"  reference frame   : {ref_name} (index {ref_idx})")
    print(f"  stabilized frames : {output_dir / 'stabilized_frames'} ({len(stabilized_paths)} frames)")
    print(f"  overlay video     : {output_dir / 'tracking_overlay_video.mp4'}")
    print(f"  trajectory video  : {output_dir / 'stabilized_trajectory_video.mp4'}")
    print(f"  trajectories      : {output_dir / 'trajectories.json'}")
    print(f"  animals tracked   : {len(trajectories)}")


if __name__ == "__main__":
    main()

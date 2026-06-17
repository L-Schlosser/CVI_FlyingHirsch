"""Thermal drone animal tracking stabilized to a reference frame.

Pipeline: load images -> warp to reference -> YOLO+BoT-SORT -> export.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import BEST_WEIGHTS, MODEL_NAME, VAL_IMGSZ

# --- sequence ----------------------------------------------------------------
# TRACK_NUMBER = "276"
# TEST_IMAGES = f"datasets/processed/images/track_test/{TRACK_NUMBER}"
# REFERENCE_FRAME = "276_5233.jpg"  # None -> first image

# TRACK_NUMBER = "196"
# TEST_IMAGES = f"datasets/processed/images/track_test/{TRACK_NUMBER}"
# REFERENCE_FRAME = "196_7584.jpg"  # None -> first image

TRACK_NUMBER = "10" #"276_3"
TEST_IMAGES = f"datasets/processed/images/track_test/{TRACK_NUMBER}"
REFERENCE_FRAME = "10_3158.jpg" #"276_5233.jpg"  # None -> first image

# --- stabilization -----------------------------------------------------------
ORB_FEATURES = 5000
REG_SCALE = 0.5
MIN_MATCHES = 20
RANSAC_PX = 3.0
SMOOTH_WINDOW = 5

# --- tracking ----------------------------------------------------------------
TRACK_CONF = 0.12
TRACK_IOU = 0.45
TRACKER_CONFIG = "trackers/botsort.yaml"

# --- visualization -----------------------------------------------------------
TRAIL_LEN = 40
VIDEO_FPS = 3
JPEG_QUALITY = 90

_IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


# =============================================================================
# I/O helpers
# =============================================================================


def load_images(folder: Path) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")

    frames: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is not None:
            frames.append((path, img))
        else:
            print(f"Warning: unreadable {path.name}")
    return frames


def ref_index(frames: list[tuple[Path, np.ndarray]], name: str | None) -> int:
    if not name:
        return 0
    for i, (path, _) in enumerate(frames):
        if path.name == name:
            return i
    names = [p.name for p, _ in frames[:5]]
    raise FileNotFoundError(f"Reference '{name}' not found. First files: {names}...")


def clear_dir_images(directory: Path) -> None:
    if not directory.exists():
        return
    for ext in ("*.jpg", "*.png"):
        for f in directory.glob(ext):
            f.unlink()


# =============================================================================
# Stabilization — ORB + partial affine, smoothed in decomposed space
# =============================================================================


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _prep(gray: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    return clahe.apply(gray)


def _down(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _up(M: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return M
    out = M.copy()
    out[0, 2] /= scale
    out[1, 2] /= scale
    return out


def _good_matches(des_a: np.ndarray, des_b: np.ndarray, matcher: cv2.BFMatcher) -> list:
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    out = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            out.append(pair[0])
    return out


def _affine(cur_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray | None:
    if len(cur_pts) < 4:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        cur_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX
    )
    return M


def _match_to_ref(
    kp_ref, des_ref, gray_cur, orb, matcher, prev_gray=None, prev_M=None
) -> np.ndarray | None:
    kp_cur, des_cur = orb.detectAndCompute(gray_cur, None)
    if des_cur is not None and len(kp_cur) >= MIN_MATCHES:
        matches = _good_matches(des_cur, des_ref, matcher)
        if len(matches) >= MIN_MATCHES:
            pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            M = _affine(pts_cur, pts_ref)
            if M is not None:
                return M

    # neighbour fallback: compose previous->ref with step previous->current
    if prev_gray is not None and prev_M is not None:
        kp_p, des_p = orb.detectAndCompute(prev_gray, None)
        if des_p is not None and des_cur is not None:
            matches = _good_matches(des_cur, des_p, matcher)
            if len(matches) >= MIN_MATCHES:
                pts_p = np.float32([kp_p[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts_c = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                step = _affine(pts_c, pts_p)
                if step is not None:
                    T = np.vstack([prev_M, [0, 0, 1]])
                    S = np.vstack([step, [0, 0, 1]])
                    return (T @ S)[:2]
    return None


def _decompose(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b, tx, ty = M[0, 0], M[1, 0], M[0, 2], M[1, 2]
    scale = float(np.hypot(a, b))
    angle = float(np.arctan2(b, a))
    return tx, ty, angle, scale


def _rebuild(tx, ty, angle, scale) -> np.ndarray:
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def _smooth(Ms: list[np.ndarray | None]) -> list[np.ndarray]:
    params = [None if M is None else _decompose(M) for M in Ms]
    out: list[np.ndarray] = []
    half = SMOOTH_WINDOW // 2
    for i, p in enumerate(params):
        if p is None:
            out.append(out[-1] if out else _IDENTITY.copy())
            continue
        nb = [params[j] for j in range(max(0, i - half), min(len(params), i + half + 1)) if params[j]]
        tx = float(np.median([n[0] for n in nb]))
        ty = float(np.median([n[1] for n in nb]))
        ang = float(np.median([n[2] for n in nb]))
        sc = float(np.median([n[3] for n in nb]))
        out.append(_rebuild(tx, ty, ang, sc))
    return out


def stabilize(
    frames: list[tuple[Path, np.ndarray]],
    ref_idx: int,
    out_dir: Path,
) -> tuple[list[Path], np.ndarray]:
    """Warp every frame into the reference frame coordinate system."""
    stab_dir = out_dir / "stabilized_frames"
    stab_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_images(stab_dir)

    ref_img = frames[ref_idx][1]
    ref_shape = ref_img.shape
    cv2.imwrite(str(out_dir / "reference_frame.jpg"), ref_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orb = cv2.ORB_create(ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ref_g = _down(_prep(_gray(ref_img), clahe), REG_SCALE)
    kp_ref, des_ref = orb.detectAndCompute(ref_g, None)
    if des_ref is None:
        raise RuntimeError("Reference frame has too few features")

    print(f"Stabilizing {len(frames)} frames to {frames[ref_idx][0].name} (index {ref_idx})")

    raw_M: list[np.ndarray | None] = [None] * len(frames)
    prev_g, prev_M = ref_g, _IDENTITY.copy()
    raw_M[ref_idx] = _IDENTITY.copy()

    for i in range(ref_idx + 1, len(frames)):
        g = _down(_prep(_gray(frames[i][1]), clahe), REG_SCALE)
        M = _match_to_ref(kp_ref, des_ref, g, orb, matcher, prev_g, prev_M)
        if M is None:
            print(f"  warning: frame {i} ({frames[i][0].name}), reusing neighbour transform")
            M = prev_M.copy()
        raw_M[i] = M
        prev_g, prev_M = g, M

    prev_g, prev_M = ref_g, _IDENTITY.copy()
    for i in range(ref_idx - 1, -1, -1):
        g = _down(_prep(_gray(frames[i][1]), clahe), REG_SCALE)
        M = _match_to_ref(kp_ref, des_ref, g, orb, matcher, prev_g, prev_M)
        if M is None:
            print(f"  warning: frame {i} ({frames[i][0].name}), reusing neighbour transform")
            M = prev_M.copy()
        raw_M[i] = M
        prev_g, prev_M = g, M

    Ms = [_up(M, REG_SCALE) for M in _smooth(raw_M)]
    write_q = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    paths: list[Path] = []
    h, w = ref_shape[:2]

    for i, (_, img) in enumerate(frames):
        warped = img if i == ref_idx else cv2.warpAffine(
            img, Ms[i], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        p = stab_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(p), warped, write_q)
        paths.append(p)

    print(f"  wrote {len(paths)} stabilized frames")
    return paths, ref_img


# =============================================================================
# Tracking — YOLO + BoT-SORT
# =============================================================================


def _device() -> str | int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def track_sequence(model: YOLO, image_paths: list[Path]) -> tuple[
    dict[int, dict[int, tuple[float, float]]],
    list[list[tuple[int, np.ndarray]]],
]:
    print(f"Running YOLO + BoT-SORT on {len(image_paths)} stabilized frames...")

    frame_tracks: dict[int, dict[int, tuple[float, float]]] = {}
    detections: list[list[tuple[int, np.ndarray]]] = []

    results = model.track(
        source=[str(p) for p in image_paths],
        stream=True,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=TRACK_CONF,
        iou=TRACK_IOU,
        imgsz=VAL_IMGSZ,
        device=_device(),
        verbose=False,
        save=False,
    )

    for fi, result in enumerate(results):
        dets: list[tuple[int, np.ndarray]] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id
            tids = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(boxes), -1)
            frame_tracks[fi] = {}
            for tid, box in zip(tids, boxes):
                if tid < 0:
                    continue
                tid = int(tid)
                dets.append((tid, box))
                frame_tracks[fi][tid] = (
                    float((box[0] + box[2]) * 0.5),
                    float((box[1] + box[3]) * 0.5),
                )
        detections.append(dets)

    n_ids = len({t for tr in frame_tracks.values() for t in tr})
    print(f"  tracked {len(frame_tracks)}/{len(image_paths)} frames, {n_ids} IDs")
    return frame_tracks, detections


def trajectories(frame_tracks: dict[int, dict[int, tuple[float, float]]]) -> dict[int, list[tuple[float, float]]]:
    ids = sorted({t for tr in frame_tracks.values() for t in tr})
    out: dict[int, list[tuple[float, float]]] = {}
    for tid in ids:
        pts = [frame_tracks[fi][tid] for fi in sorted(frame_tracks) if tid in frame_tracks[fi]]
        if pts:
            out[tid] = pts
    return out


# =============================================================================
# Visualization
# =============================================================================


def _color(tid: int) -> tuple[int, int, int]:
    hue = (tid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_trail(canvas: np.ndarray, history: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for j in range(1, len(history)):
        t = j / len(history)
        thickness = max(1, int(1 + 3 * t))
        fade = 0.35 + 0.65 * t
        c = tuple(int(v * fade) for v in color)
        cv2.line(canvas, history[j - 1], history[j], c, thickness, cv2.LINE_AA)


def render_overlays(
    frames: list[np.ndarray],
    detections: list[list[tuple[int, np.ndarray]]],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    histories: dict[int, list[tuple[int, int]]] = {}

    for fi, frame in enumerate(frames):
        img = frame.copy()
        for tid, box in detections[fi] if fi < len(detections) else []:
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = _color(tid)
            hist = histories.setdefault(tid, [])
            hist.append((cx, cy))
            if len(hist) > TRAIL_LEN:
                hist.pop(0)
            _draw_trail(img, hist, color)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.putText(img, f"{tid}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        out.append(img)
    return out


def render_trajectory_video(
    stabilized_frames: list[np.ndarray],
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    out_path: Path,
) -> None:
    """Trails on each stabilized frame — background is stable, animals move."""
    if not stabilized_frames:
        return

    h, w = stabilized_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    histories: dict[int, list[tuple[int, int]]] = {}
    n_frames = len(stabilized_frames)

    for fi, frame in enumerate(stabilized_frames):
        if frame is None:
            continue

        canvas = frame.copy()
        tracks = frame_tracks.get(fi, {})

        for tid, (x, y) in tracks.items():
            pt = (int(round(x)), int(round(y)))
            hist = histories.setdefault(tid, [])
            hist.append(pt)
            if len(hist) > TRAIL_LEN:
                hist.pop(0)

        for tid, hist in histories.items():
            color = _color(tid)
            _draw_trail(canvas, hist, color)
            if tid in tracks:
                cx, cy = int(round(tracks[tid][0])), int(round(tracks[tid][1]))
                cv2.circle(canvas, (cx, cy), 6, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas, f"ID {tid}", (cx + 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )

        cv2.putText(
            canvas, f"Frame {fi + 1}/{n_frames}", (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def write_video(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def save_results(
    out_dir: Path,
    trajectories: dict[int, list[tuple[float, float]]],
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    ref_idx: int,
    ref_name: str,
    n_frames: int,
    id_links: dict[int, int],
) -> None:
    with open(out_dir / "trajectories.txt", "w", encoding="utf-8") as f:
        for tid, pts in sorted(trajectories.items()):
            f.write(f"ID {tid}: {pts}\n")

    payload = {
        "coordinate_system": f"stabilized_reference_{ref_name}",
        "reference_frame_index": ref_idx,
        "reference_frame_name": ref_name,
        "num_frames": n_frames,
        "trajectories": {str(t): [[float(x), float(y)] for x, y in p] for t, p in sorted(trajectories.items())},
        "frame_tracks": {
            str(fi): {str(t): [float(x), float(y)] for t, (x, y) in tr.items()}
            for fi, tr in sorted(frame_tracks.items())
        },
        "id_links": {str(o): int(n) for o, n in sorted(id_links.items()) if o != n},
    }
    with open(out_dir / "trajectories.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    frames = load_images(Path(TEST_IMAGES))
    ref_idx = ref_index(frames, REFERENCE_FRAME)
    ref_name = frames[ref_idx][0].name

    out_dir = Path("runs/detect/track") / f"track_{MODEL_NAME}_{TRACK_NUMBER}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stab_paths, _ = stabilize(frames, ref_idx, out_dir)

    model = YOLO(BEST_WEIGHTS)
    frame_tracks, detections = track_sequence(model, stab_paths)

    traj = trajectories(frame_tracks)
    stab_imgs = [cv2.imread(str(p)) for p in stab_paths]
    overlays = render_overlays(stab_imgs, detections)

    overlay_dir = out_dir / "tracking_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_images(overlay_dir)
    q = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    for i, img in enumerate(overlays):
        cv2.imwrite(str(overlay_dir / f"{i:05d}.jpg"), img, q)

    write_video(overlays, out_dir / "tracking_overlay_video.mp4")
    render_trajectory_video(stab_imgs, frame_tracks, out_dir / "stabilized_trajectory_video.mp4")
    save_results(out_dir, traj, frame_tracks, ref_idx, ref_name, len(frames), {})

    print()
    print("Done.")
    print(f"  reference frame   : {ref_name} (index {ref_idx})")
    print(f"  stabilized frames : {out_dir / 'stabilized_frames'}")
    print(f"  overlay video     : {out_dir / 'tracking_overlay_video.mp4'}")
    print(f"  trajectory video  : {out_dir / 'stabilized_trajectory_video.mp4'}  (stabilized frames + trails)")
    print(f"  trajectories      : {out_dir / 'trajectories.json'}")
    print(f"  animals tracked   : {len(traj)}")


if __name__ == "__main__":
    main()

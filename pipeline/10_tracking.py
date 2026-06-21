from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    BEST_WEIGHTS,
    MODEL_NAME,
    TRACKER_CONFIG,
    THERMAL_IMAGES_PATH,
    TRACK_RUNS_DIR,
)

# --- sequence ----------------------------------------------------------------
TRACK_NUMBER = ["10", "276"]
REFERENCE_FRAME = ["10_3158.jpg", "276_5233.jpg"]  # Use None for "first image"
TRACK_SPLITS = ["test2", "val2"]

# --- stabilization -----------------------------------------------------------
ORB_FEATURES = 5000
REG_SCALE = 0.5
MIN_MATCHES = 20
RANSAC_PX = 3.0
SMOOTH_WINDOW = 5

# --- tracking ----------------------------------------------------------------
TRACK_IMGSZ = 1024
MIN_TRACK_LENGTH = 2
MAX_TRACK_GAP = 8

# --- frame quality -----------------------------------------------------------
MIN_FRAME_LAPLACIAN_VAR = 18.0
MIN_FRAME_STDDEV = 12.0
MAX_TRANSLATION_JUMP_PX = 120.0
MIN_REFERENCE_CORRELATION = 0.52
USE_BLUR_CONTRAST_FILTERS = False
KEEP_ONLY_REFERENCE_SEGMENT = True
KEEP_ONLY_FORWARD_FROM_REFERENCE = False
KEEP_FRAMES_BEFORE_REFERENCE = 15
KEEP_FRAMES_AFTER_REFERENCE = 15
SKIP_BAD_FRAMES = True

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


def load_track_images(folder: Path, track_number: str) -> list[tuple[Path, np.ndarray]]:
    candidates = load_images(folder)
    prefix = f"{track_number}_"
    frames = [(path, img) for path, img in candidates if path.name.startswith(prefix)]
    if not frames:
        available_tracks = sorted({path.stem.split("_")[0] for path, _ in candidates})
        preview = ", ".join(available_tracks[:20])
        suffix = "..." if len(available_tracks) > 20 else ""
        raise FileNotFoundError(
            f"No images found for track '{track_number}' in {folder}. "
            f"Available track IDs: {preview}{suffix}"
        )
    return frames


def load_track_images_auto(
    images_root: Path,
    track_number: str,
    track_splits: list[str],
) -> tuple[list[tuple[Path, np.ndarray]], Path, str]:
    checked_folders: list[Path] = []
    available_tracks_by_split: dict[str, list[str]] = {}

    for track_split in track_splits:
        folder = images_root / track_split
        checked_folders.append(folder)
        try:
            frames = load_track_images(folder, track_number)
            return frames, folder, track_split
        except FileNotFoundError:
            if folder.exists():
                candidates = load_images(folder)
                available_tracks_by_split[track_split] = sorted(
                    {path.stem.split("_")[0] for path, _ in candidates}
                )
            else:
                available_tracks_by_split[track_split] = []

    details = []
    for track_split in track_splits:
        tracks = available_tracks_by_split.get(track_split, [])
        preview = ", ".join(tracks[:10]) if tracks else "none"
        suffix = "..." if len(tracks) > 10 else ""
        details.append(f"{track_split}: {preview}{suffix}")
    checked = ", ".join(str(folder) for folder in checked_folders)
    raise FileNotFoundError(
        f"No images found for track '{track_number}' in any of: {checked}. "
        f"Available track IDs by split -> {' | '.join(details)}"
    )


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


def trim_frames_from_reference(
    frames: list[tuple[Path, np.ndarray]],
    ref_idx: int,
    keep_only_forward: bool,
) -> tuple[list[tuple[Path, np.ndarray]], int]:
    if not keep_only_forward:
        start_idx = max(0, ref_idx - KEEP_FRAMES_BEFORE_REFERENCE)
        end_idx = min(len(frames), ref_idx + KEEP_FRAMES_AFTER_REFERENCE + 1)
        return frames[start_idx:end_idx], ref_idx - start_idx
    start_idx = max(0, ref_idx - KEEP_FRAMES_BEFORE_REFERENCE)
    return frames[start_idx:], ref_idx - start_idx


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


def _frame_quality_score(image: np.ndarray) -> dict[str, float]:
    gray = _gray(image)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    intensity_std = float(gray.std())
    return {
        "laplacian_var": lap_var,
        "intensity_std": intensity_std,
    }


def _reference_correlation(image: np.ndarray, reference_gray: np.ndarray) -> float:
    gray = _gray(image)
    if gray.shape != reference_gray.shape:
        gray = cv2.resize(gray, (reference_gray.shape[1], reference_gray.shape[0]), interpolation=cv2.INTER_AREA)

    # Ignore a small border to reduce false penalties from warp padding.
    h, w = reference_gray.shape
    margin_y = max(4, h // 20)
    margin_x = max(4, w // 20)
    ref_crop = reference_gray[margin_y:h - margin_y, margin_x:w - margin_x]
    img_crop = gray[margin_y:h - margin_y, margin_x:w - margin_x]

    ref_vec = ref_crop.astype(np.float32).ravel()
    img_vec = img_crop.astype(np.float32).ravel()
    ref_std = float(ref_vec.std())
    img_std = float(img_vec.std())
    if ref_std < 1e-6 or img_std < 1e-6:
        return 0.0
    corr = float(np.corrcoef(ref_vec, img_vec)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def _translation_jump(M_prev: np.ndarray, M_cur: np.ndarray) -> float:
    dx = float(M_cur[0, 2] - M_prev[0, 2])
    dy = float(M_cur[1, 2] - M_prev[1, 2])
    return float(np.hypot(dx, dy))


def _bad_frame_reason(
    quality: dict[str, float],
    translation_jump: float | None,
    reference_correlation: float,
) -> str | None:
    if USE_BLUR_CONTRAST_FILTERS and quality["laplacian_var"] < MIN_FRAME_LAPLACIAN_VAR:
        return f"blurry(lap_var={quality['laplacian_var']:.1f})"
    if USE_BLUR_CONTRAST_FILTERS and quality["intensity_std"] < MIN_FRAME_STDDEV:
        return f"low_contrast(std={quality['intensity_std']:.1f})"
    if translation_jump is not None and translation_jump > MAX_TRANSLATION_JUMP_PX:
        return f"warp_jump({translation_jump:.1f}px)"
    if reference_correlation < MIN_REFERENCE_CORRELATION:
        return f"ref_mismatch(corr={reference_correlation:.2f})"
    return None


def _select_reference_segment(
    ref_idx: int,
    frame_ok: list[bool],
) -> set[int]:
    keep = {ref_idx}

    if not KEEP_ONLY_FORWARD_FROM_REFERENCE:
        i = ref_idx - 1
        while i >= 0 and frame_ok[i]:
            keep.add(i)
            i -= 1

    i = ref_idx + 1
    while i < len(frame_ok) and frame_ok[i]:
        keep.add(i)
        i += 1

    return keep


def stabilize(
    frames: list[tuple[Path, np.ndarray]],
    ref_idx: int,
    out_dir: Path,
) -> tuple[list[Path], list[np.ndarray], np.ndarray, dict[int, str]]:
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
    reference_gray_full = _gray(ref_img)

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
    stabilized_frames: list[np.ndarray] = []
    bad_frame_reasons: dict[int, str] = {}
    frame_ok: list[bool] = [False] * len(frames)
    h, w = ref_shape[:2]

    prev_M = None
    for i, (_, img) in enumerate(frames):
        warped = img if i == ref_idx else cv2.warpAffine(
            img, Ms[i], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        quality = _frame_quality_score(warped)
        translation_jump = None if prev_M is None else _translation_jump(prev_M, Ms[i])
        reference_correlation = 1.0 if i == ref_idx else _reference_correlation(warped, reference_gray_full)
        reason = _bad_frame_reason(quality, translation_jump, reference_correlation)
        frame_ok[i] = reason is None or i == ref_idx
        if reason is not None and i != ref_idx:
            bad_frame_reasons[i] = reason
        p = stab_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(p), warped, write_q)
        paths.append(p)
        stabilized_frames.append(warped)
        prev_M = Ms[i]

    if KEEP_ONLY_REFERENCE_SEGMENT:
        keep_indices = _select_reference_segment(ref_idx, frame_ok)
        for i in range(len(frames)):
            if i == ref_idx:
                continue
            if i not in keep_indices:
                bad_frame_reasons[i] = "outside_reference_segment"

    print(f"  wrote {len(paths)} stabilized frames")
    if bad_frame_reasons:
        print(f"  marked {len(bad_frame_reasons)} low-quality frames for tracking skip")
    return paths, stabilized_frames, ref_img, bad_frame_reasons


# =============================================================================
# Tracking — YOLO + BoT-SORT
# =============================================================================


def _device() -> str | int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _track_sequence_on_device(
    model: YOLO,
    stabilized_frames: list[np.ndarray],
    bad_frame_reasons: dict[int, str],
    device: str | int,
) -> tuple[
    dict[int, dict[int, tuple[float, float]]],
    list[list[tuple[int, np.ndarray]]],
]:
    print(f"Running YOLO + BoT-SORT on {len(stabilized_frames)} stabilized frames (device={device})...")

    frame_tracks: dict[int, dict[int, tuple[float, float]]] = {}
    detections: list[list[tuple[int, np.ndarray]]] = []

    for fi, frame in enumerate(stabilized_frames):
        if SKIP_BAD_FRAMES and fi in bad_frame_reasons:
            detections.append([])
            continue
        results = model.track(
            source=frame,
            stream=True,
            persist=True,
            tracker=str(TRACKER_CONFIG),
            imgsz=TRACK_IMGSZ,
            device=device,
            verbose=False,
            save=False,
        )
        result = next(results)
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
    print(f"  tracked {len(frame_tracks)}/{len(stabilized_frames)} frames, {n_ids} IDs")
    return frame_tracks, detections


def track_sequence(model: YOLO, stabilized_frames: list[np.ndarray]) -> tuple[
    dict[int, dict[int, tuple[float, float]]],
    list[list[tuple[int, np.ndarray]]],
] :
    return track_sequence_with_quality(model, stabilized_frames, {})


def track_sequence_with_quality(
    model: YOLO,
    stabilized_frames: list[np.ndarray],
    bad_frame_reasons: dict[int, str],
) -> tuple[
    dict[int, dict[int, tuple[float, float]]],
    list[list[tuple[int, np.ndarray]]],
]:
    device = _device()
    try:
        return _track_sequence_on_device(model, stabilized_frames, bad_frame_reasons, device)
    except Exception as exc:
        message = str(exc).lower()
        is_cuda_oom = device != "cpu" and (
            "out of memory" in message
            or "cuda error" in message
            or exc.__class__.__name__ == "AcceleratorError"
        )
        if not is_cuda_oom:
            raise

        print("CUDA ran out of memory during tracking. Retrying on CPU...")
        _clear_cuda_cache()
        return _track_sequence_on_device(model, stabilized_frames, bad_frame_reasons, "cpu")


def filter_short_tracks(
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    detections: list[list[tuple[int, np.ndarray]]],
    min_track_length: int = MIN_TRACK_LENGTH,
) -> tuple[dict[int, dict[int, tuple[float, float]]], list[list[tuple[int, np.ndarray]]]]:
    counts: dict[int, int] = {}
    for tracks in frame_tracks.values():
        for tid in tracks:
            counts[tid] = counts.get(tid, 0) + 1

    keep_ids = {tid for tid, count in counts.items() if count >= min_track_length}
    filtered_frame_tracks: dict[int, dict[int, tuple[float, float]]] = {}
    for frame_idx, tracks in frame_tracks.items():
        kept = {tid: xy for tid, xy in tracks.items() if tid in keep_ids}
        if kept:
            filtered_frame_tracks[frame_idx] = kept

    filtered_detections: list[list[tuple[int, np.ndarray]]] = []
    for frame_detections in detections:
        filtered_detections.append([(tid, box) for tid, box in frame_detections if tid in keep_ids])

    removed = len(counts) - len(keep_ids)
    if removed > 0:
        print(f"  removed {removed} short-lived tracks (< {min_track_length} frames)")
    return filtered_frame_tracks, filtered_detections


def bridge_track_gaps(
    frame_tracks: dict[int, dict[int, tuple[float, float]]],
    detections: list[list[tuple[int, np.ndarray]]],
    max_track_gap: int = MAX_TRACK_GAP,
) -> tuple[dict[int, dict[int, tuple[float, float]]], list[list[tuple[int, np.ndarray]]]]:
    if max_track_gap <= 0 or not frame_tracks:
        return frame_tracks, detections

    bridged_frame_tracks = {fi: dict(tracks) for fi, tracks in frame_tracks.items()}
    for tid in sorted({t for tracks in frame_tracks.values() for t in tracks}):
        frames_with_tid = sorted(fi for fi, tracks in frame_tracks.items() if tid in tracks)
        for left_idx in range(len(frames_with_tid) - 1):
            start_fi = frames_with_tid[left_idx]
            end_fi = frames_with_tid[left_idx + 1]
            gap = end_fi - start_fi - 1
            if gap <= 0 or gap > max_track_gap:
                continue

            start_xy = np.array(frame_tracks[start_fi][tid], dtype=np.float32)
            end_xy = np.array(frame_tracks[end_fi][tid], dtype=np.float32)
            for step, fi in enumerate(range(start_fi + 1, end_fi), start=1):
                alpha = step / (gap + 1)
                interp_xy = tuple((start_xy * (1.0 - alpha) + end_xy * alpha).astype(float))
                bridged_frame_tracks.setdefault(fi, {})
                if tid not in bridged_frame_tracks[fi]:
                    bridged_frame_tracks[fi][tid] = interp_xy

    kept_ids = {t for tracks in bridged_frame_tracks.values() for t in tracks}
    bridged_detections: list[list[tuple[int, np.ndarray]]] = []
    for frame_detections in detections:
        bridged_detections.append([(tid, box) for tid, box in frame_detections if tid in kept_ids])

    print(f"  bridged track gaps up to {max_track_gap} frames")
    return bridged_frame_tracks, bridged_detections


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
    bad_frame_reasons: dict[int, str],
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
        "bad_frames": {str(frame_idx): reason for frame_idx, reason in sorted(bad_frame_reasons.items())},
    }
    with open(out_dir / "trajectories.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_track_jobs(
    track_numbers: list[str],
    reference_frames: list[str | None],
) -> list[tuple[str, str | None]]:
    if not isinstance(track_numbers, list) or not isinstance(reference_frames, list):
        raise TypeError("TRACK_NUMBER and REFERENCE_FRAME must both be lists.")

    if len(track_numbers) != len(reference_frames):
        raise ValueError(
            "TRACK_NUMBER and REFERENCE_FRAME must have the same length."
        )

    return [(str(track), ref) for track, ref in zip(track_numbers, reference_frames)]


def resolve_track_splits(track_splits: list[str]) -> list[str]:
    allowed_splits = {"test2", "val2"}
    invalid = [track_split for track_split in track_splits if track_split not in allowed_splits]
    if invalid:
        allowed = ", ".join(sorted(allowed_splits))
        invalid_text = ", ".join(invalid)
        raise ValueError(f"Invalid TRACK_SPLITS value(s): {invalid_text}. Allowed: {allowed}")
    return track_splits


# =============================================================================
# Main
# =============================================================================


def run_tracking(track_number: str, reference_frame: str | None, model: YOLO, track_splits: list[str]) -> None:
    frames, track_source, track_split = load_track_images_auto(THERMAL_IMAGES_PATH, track_number, track_splits)
    ref_idx = ref_index(frames, reference_frame)
    frames, ref_idx = trim_frames_from_reference(frames, ref_idx, KEEP_ONLY_FORWARD_FROM_REFERENCE)
    ref_name = frames[ref_idx][0].name

    out_dir = TRACK_RUNS_DIR / f"track_{MODEL_NAME}_{track_number}_{track_split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stab_paths, stabilized_frames, _, bad_frame_reasons = stabilize(frames, ref_idx, out_dir)

    frame_tracks, detections = track_sequence_with_quality(model, stabilized_frames, bad_frame_reasons)
    frame_tracks, detections = filter_short_tracks(frame_tracks, detections)
    frame_tracks, detections = bridge_track_gaps(frame_tracks, detections)

    traj = trajectories(frame_tracks)
    overlays = render_overlays(stabilized_frames, detections)

    overlay_dir = out_dir / "tracking_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_images(overlay_dir)
    q = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    for i, img in enumerate(overlays):
        cv2.imwrite(str(overlay_dir / f"{i:05d}.jpg"), img, q)

    write_video(overlays, out_dir / "tracking_overlay_video.mp4")
    render_trajectory_video(stabilized_frames, frame_tracks, out_dir / "stabilized_trajectory_video.mp4")
    save_results(out_dir, traj, frame_tracks, ref_idx, ref_name, len(frames), {}, bad_frame_reasons)

    print()
    print("Done.")
    print(f"  track number       : {track_number}")
    print(f"  track split        : {track_split}")
    print(f"  source folder      : {track_source}")
    print(f"  reference frame   : {ref_name} (index {ref_idx})")
    print(f"  stabilized frames : {out_dir / 'stabilized_frames'}")
    print(f"  overlay video     : {out_dir / 'tracking_overlay_video.mp4'}")
    print(f"  trajectory video  : {out_dir / 'stabilized_trajectory_video.mp4'}  (stabilized frames + trails)")
    print(f"  trajectories      : {out_dir / 'trajectories.json'}")
    print(f"  animals tracked   : {len(traj)}")
    if bad_frame_reasons:
        print(f"  skipped frames    : {len(bad_frame_reasons)} low-quality frames")


def main() -> None:
    jobs = resolve_track_jobs(TRACK_NUMBER, REFERENCE_FRAME)
    track_splits = resolve_track_splits(TRACK_SPLITS)
    model = YOLO(BEST_WEIGHTS)
    for track_number, reference_frame in jobs:
        run_tracking(track_number, reference_frame, model, track_splits)


if __name__ == "__main__":
    main()

from collections import defaultdict
from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics import YOLO

from config import BEST_WEIGHTS, MODEL_NAME

TRACK_NUMBER = "276"
TEST_IMAGES = Path(f"datasets/processed/images/track_test/{TRACK_NUMBER}")
REFERENCE_FRAME_NAME = "00031.jpg"


# ----------------------------
# CONFIG
# ----------------------------
MAX_INTERPOLATION_GAP = 8
MAX_TRAIL_LENGTH = 45
OUTPUT_FPS = 12
ORB_FEATURES = 4000
MIN_GOOD_MATCHES = 10


# ----------------------------
# HELPERS
# ----------------------------
def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_image_paths(source: Path):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(source.glob(pattern))
    return sorted(image_paths, key=natural_key)


def stable_color(track_id):
    rng = np.random.default_rng(track_id * 9973)
    color = rng.integers(80, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def build_stabilized_sequence(image_paths, output_dir, reference_name=REFERENCE_FRAME_NAME):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = next((path for path in image_paths if path.name == reference_name), None)
    if reference_path is None:
        reference_path = image_paths[min(len(image_paths) // 2, len(image_paths) - 1)]

    reference_image = cv2.imread(str(reference_path))
    if reference_image is None:
        raise FileNotFoundError(f"Could not read reference frame: {reference_path}")

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    stabilized_paths = []
    last_good_transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        transform = None
        if (
            descriptors is not None
            and reference_descriptors is not None
            and len(keypoints) >= 4
            and len(reference_keypoints) >= 4
        ):
            matches = matcher.knnMatch(descriptors, reference_descriptors, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) < 2:
                    continue
                first, second = match_pair
                if first.distance < 0.75 * second.distance:
                    good_matches.append(first)

            if len(good_matches) >= MIN_GOOD_MATCHES:
                source_points = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                target_points = np.float32([reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                transform, _ = cv2.estimateAffinePartial2D(
                    source_points,
                    target_points,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=3000,
                    confidence=0.99,
                )

        if transform is None:
            transform = last_good_transform.copy()
        else:
            last_good_transform = transform.astype(np.float32)

        stabilized = cv2.warpAffine(
            image,
            transform,
            (reference_image.shape[1], reference_image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), stabilized)
        stabilized_paths.append(out_path)

    return stabilized_paths, reference_path


def interpolate_track(points):
    points = sorted(points, key=lambda item: item[0])
    if not points:
        return {}

    interpolated = {frame_idx: (x, y) for frame_idx, x, y in points}

    for (frame_a, x_a, y_a), (frame_b, x_b, y_b) in zip(points, points[1:]):
        gap = frame_b - frame_a
        if gap <= 1 or gap > MAX_INTERPOLATION_GAP:
            continue

        for step in range(1, gap):
            alpha = step / gap
            x = int(round(x_a + alpha * (x_b - x_a)))
            y = int(round(y_a + alpha * (y_b - y_a)))
            interpolated[frame_a + step] = (x, y)

    return dict(sorted(interpolated.items()))


def make_trajectory_lookup(trajectories):
    return {track_id: interpolate_track(points) for track_id, points in trajectories.items()}


def draw_label(frame, text, anchor, color):
    x, y = anchor
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )

    x0 = max(0, x)
    y0 = max(0, y - text_height - baseline - 8)
    x1 = min(frame.shape[1] - 1, x0 + text_width + 10)
    y1 = min(frame.shape[0] - 1, y0 + text_height + baseline + 8)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    cv2.putText(
        frame,
        text,
        (x0 + 5, y1 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, frame_idx, total_frames, active_ids):
    hud = frame.copy()
    cv2.rectangle(hud, (16, 16), (360, 94), (12, 12, 12), -1)
    frame = cv2.addWeighted(hud, 0.72, frame, 0.28, 0)

    cv2.putText(
        frame,
        f"Frame {frame_idx + 1:04d} / {total_frames:04d}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Active IDs: {len(active_ids)}",
        (28, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return frame


def write_video(frame_paths, output_path, fps=12):
    if not frame_paths:
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        return

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()


def render_sequence(image_paths, raw_trajectories, interpolated_trajectories, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_track_ids = sorted(interpolated_trajectories)
    colors = {track_id: stable_color(track_id) for track_id in all_track_ids}
    rendered_frames = []

    for frame_idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        overlay = img.copy()
        active_ids = []

        for track_id in all_track_ids:
            track_points = interpolated_trajectories[track_id]
            visible_frames = [frame for frame in track_points if frame <= frame_idx]
            if not visible_frames:
                continue

            active_ids.append(track_id)
            tail_frames = visible_frames[-MAX_TRAIL_LENGTH:]
            tail_points = [track_points[frame] for frame in tail_frames]
            color = colors[track_id]

            for point_index in range(1, len(tail_points)):
                x1, y1 = tail_points[point_index - 1]
                x2, y2 = tail_points[point_index]
                fade = 0.35 + 0.65 * (point_index / max(1, len(tail_points) - 1))
                line_color = tuple(int(channel * fade) for channel in color)
                cv2.line(overlay, (x1, y1), (x2, y2), line_color, 3, cv2.LINE_AA)

            x, y = tail_points[-1]
            observed = frame_idx in raw_trajectories.get(track_id, {})
            cv2.circle(overlay, (x, y), 6 if observed else 5, color, -1, cv2.LINE_AA)
            if observed:
                cv2.circle(overlay, (x, y), 11, color, 1, cv2.LINE_AA)
            draw_label(overlay, f"ID {track_id}", (x + 10, y - 10), color)

        frame = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)
        frame = draw_hud(frame, frame_idx, len(image_paths), active_ids)

        out_path = output_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        rendered_frames.append(out_path)

    return rendered_frames


# ----------------------------
# MAIN
# ----------------------------
def main():
    model = YOLO(BEST_WEIGHTS)

    source = TEST_IMAGES
    image_paths = load_image_paths(source)

    if not image_paths:
        raise FileNotFoundError("No images found")

    out_dir = Path("runs/detect/track") / f"track_{MODEL_NAME}_{TRACK_NUMBER}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracking {len(image_paths)} images from {source}")
    stabilized_dir = out_dir / "stabilized_frames"
    stabilized_paths, reference_path = build_stabilized_sequence(image_paths, stabilized_dir)
    print(f"Stabilized sequence anchored at {reference_path.name}")

    raw_trajectories = defaultdict(list)  # id -> [(frame, x, y)]

    results = model.track(
        source=[str(path) for path in stabilized_paths],
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.05,
        iou=0.25,
        save=False,
        verbose=False,
    )

    for i, r in enumerate(results):
        if r.boxes.id is None:
            continue

        ids = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()

        for obj_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            raw_trajectories[int(obj_id)].append((i, cx, cy))

    # ----------------------------
    # SAVE TRAJECTORIES
    # ----------------------------
    trajectories = {track_id: sorted(points, key=lambda item: item[0]) for track_id, points in raw_trajectories.items()}
    interpolated_trajectories = make_trajectory_lookup(trajectories)

    traj_file = out_dir / "trajectories.txt"
    with open(traj_file, "w") as f:
        for tid, points in trajectories.items():
            f.write(f"ID {tid}: {points}\n")

    (out_dir / "reference_frame.txt").write_text(reference_path.name, encoding="utf-8")

    rendered_frames = render_sequence(stabilized_paths, trajectories, interpolated_trajectories, out_dir / "frames")
    video_path = out_dir / "tracking_sequence.mp4"
    write_video(rendered_frames, video_path, fps=OUTPUT_FPS)

    print(f"Done. Output saved to {out_dir}")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
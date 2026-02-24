from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional at runtime
    cv2 = None


def _defaults() -> dict[str, float]:
    return {
        "cv_traffic_index": 0.5,
        "cv_grip_index": 1.0,
        "cv_rain_index": 0.05,
        "cv_visibility_index": 0.9,
        "cv_frame_count": 0.0,
    }


def extract_track_features(
    video_path: str,
    frame_stride: int = 10,
    min_contour_area: int = 220,
) -> dict[str, float]:
    if cv2 is None:
        return _defaults()

    path = Path(video_path)
    if not path.exists():
        return _defaults()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _defaults()

    bg = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=24, detectShadows=False)

    traffic_scores: list[float] = []
    grip_scores: list[float] = []
    rain_scores: list[float] = []
    visibility_scores: list[float] = []

    idx = 0
    sampled = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_stride != 0:
                continue

            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            fg = bg.apply(frame)
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            moving = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
            traffic_scores.append(float(len(moving)))

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            grip_scores.append(sharpness)

            s = hsv[:, :, 1].astype(np.float32)
            v = hsv[:, :, 2].astype(np.float32)
            wet_reflection = float(np.mean((v > 180) & (s < 70)))
            rain_scores.append(wet_reflection)

            visibility_scores.append(float(np.mean(v) / 255.0))
    finally:
        cap.release()

    if sampled == 0:
        return _defaults()

    traffic = np.array(traffic_scores, dtype=float)
    grip = np.array(grip_scores, dtype=float)
    rain = np.array(rain_scores, dtype=float)
    visibility = np.array(visibility_scores, dtype=float)

    out = _defaults()
    out["cv_traffic_index"] = float(np.clip(np.mean(traffic) / 8.0, 0.05, 2.0))
    out["cv_grip_index"] = float(np.clip(np.mean(grip) / 120.0, 0.35, 2.0))
    out["cv_rain_index"] = float(np.clip(np.mean(rain) * 6.0, 0.0, 2.0))
    out["cv_visibility_index"] = float(np.clip(np.mean(visibility), 0.15, 1.0))
    out["cv_frame_count"] = float(sampled)
    return out


def load_cv_features_from_directory(
    videos_dir: str,
    frame_stride: int = 10,
    min_contour_area: int = 220,
) -> dict[str, dict[str, float]]:
    root = Path(videos_dir)
    if not root.exists():
        return {}

    valid_suffixes = {".mp4", ".mov", ".mkv", ".avi"}
    rows: dict[str, dict[str, float]] = {}
    for file in root.iterdir():
        if file.suffix.lower() not in valid_suffixes:
            continue
        rows[file.stem] = extract_track_features(
            str(file),
            frame_stride=frame_stride,
            min_contour_area=min_contour_area,
        )
    return rows

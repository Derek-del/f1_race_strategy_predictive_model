from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.cv.track_state import extract_track_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CV track-state features from a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--min-contour-area", type=int, default=220)
    args = parser.parse_args()

    features = extract_track_features(
        video_path=args.video,
        frame_stride=args.frame_stride,
        min_contour_area=args.min_contour_area,
    )
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()

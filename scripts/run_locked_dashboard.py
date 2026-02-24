from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.dashboard.server import build_dashboard_payload, serve_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local dashboard for latest locked run (or reports fallback)"
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Explicit snapshot directory path (contains manifest.json)",
    )
    parser.add_argument(
        "--lock-root",
        default="reports/locks",
        help="Directory containing lock snapshots",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Fallback reports directory when no lock snapshot exists",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()

    payload = build_dashboard_payload(
        snapshot_dir=args.snapshot_dir,
        lock_root=args.lock_root,
        reports_dir=args.reports_dir,
    )

    src = payload.get("source", {})
    print(f"Source mode: {src.get('mode')}")
    print(f"Source path: {src.get('path')}")

    serve_dashboard(payload=payload, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

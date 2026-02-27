from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_strategy_lab.dashboard.server import _dashboard_html, build_dashboard_payload


def _vercel_config() -> dict[str, object]:
    return {
        "cleanUrls": True,
        "rewrites": [
            {"source": "/api/data", "destination": "/data/payload.json"},
            {"source": "/overview", "destination": "/index.html"},
            {"source": "/races", "destination": "/index.html"},
            {"source": "/strategy", "destination": "/index.html"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export static dashboard website payload for Vercel hosting"
    )
    parser.add_argument("--snapshot-dir", default=None, help="Optional explicit snapshot directory")
    parser.add_argument("--lock-root", default="reports/locks", help="Lock snapshot root directory")
    parser.add_argument("--reports-dir", default="reports", help="Fallback reports directory")
    parser.add_argument("--out-dir", default="site", help="Static site output directory")
    parser.add_argument(
        "--fallback-existing-payload",
        action="store_true",
        help="If fresh payload generation fails, reuse existing out-dir/data/payload.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    data_dir = out_dir / "data"
    payload_path = data_dir / "payload.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = build_dashboard_payload(
            snapshot_dir=args.snapshot_dir,
            lock_root=args.lock_root,
            reports_dir=args.reports_dir,
        )
    except Exception as exc:
        if not args.fallback_existing_payload or not payload_path.exists():
            raise
        print(f"[WARN] Could not generate fresh payload; using existing payload file: {exc}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

    (out_dir / "index.html").write_text(_dashboard_html(), encoding="utf-8")
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "vercel.json").write_text(json.dumps(_vercel_config(), indent=2), encoding="utf-8")

    print("Static site export complete")
    print(f"Output dir: {out_dir}")
    print(f"Payload rows: {len(payload.get('strategy_rows', []))}")
    print("Deploy from output dir with: vercel --prod")


if __name__ == "__main__":
    main()

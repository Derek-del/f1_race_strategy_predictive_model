from __future__ import annotations

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from f1_strategy_lab.utils.io import load_json


NUMERIC_FIELDS = {
    "predicted_base_lap_sec",
    "stops",
    "first_pit_lap",
    "fallback_2_stops",
    "fallback_2_first_pit_lap",
    "fallback_3_stops",
    "fallback_3_first_pit_lap",
    "expected_race_time",
    "win_probability",
    "strategy_score",
    "robustness_window",
    "year",
}


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_strategy_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in NUMERIC_FIELDS:
                    parsed[key] = _to_float(value) if value not in (None, "") else None
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _resolve_path(path_hint: str | None, run_dir: Path, fallback_filename: str) -> Path:
    candidates: list[Path] = []

    if path_hint:
        hinted = Path(path_hint)
        candidates.append(hinted)
        if not hinted.is_absolute():
            candidates.append(run_dir / hinted)
            candidates.append(Path.cwd() / hinted)

    candidates.append(run_dir / fallback_filename)
    candidates.append(Path.cwd() / "reports" / fallback_filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate '{fallback_filename}'. Checked hints: {[str(c) for c in candidates]}"
    )


def find_latest_locked_snapshot(lock_root: str | Path) -> Path | None:
    root = Path(lock_root)
    if not root.exists() or not root.is_dir():
        return None

    snapshots = [
        d
        for d in root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ]
    if not snapshots:
        return None

    return max(snapshots, key=lambda p: p.stat().st_mtime)


def _build_kpis(summary: dict[str, Any], championship: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    return {
        "training_rows": summary.get("training_rows") if isinstance(summary, dict) else None,
        "inference_rows": summary.get("inference_rows") if isinstance(summary, dict) else None,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "driver_title_probability": championship.get("driver_title_probability"),
        "constructors_title_probability": championship.get("constructors_title_probability"),
    }


def _top_rounds(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            float(r.get("strategy_score") or 0.0),
            float(r.get("win_probability") or 0.0),
        ),
        reverse=True,
    )
    return ordered[:limit]


def build_dashboard_payload(
    snapshot_dir: str | Path | None = None,
    lock_root: str | Path = "reports/locks",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    run_dir: Path
    if snapshot_dir:
        run_dir = Path(snapshot_dir).resolve()
    else:
        latest = find_latest_locked_snapshot(lock_root)
        run_dir = latest.resolve() if latest else Path(reports_dir).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}. Run pipeline first or pass --snapshot-dir explicitly."
        )

    manifest_path = run_dir / "manifest.json"
    run_summary_path = run_dir / "run_summary.json"

    if manifest_path.exists():
        manifest = load_json(manifest_path) or {}
        summary = manifest.get("summary", {})
        round_validation = manifest.get("round_validation", {})
        outputs_map = manifest.get("outputs", {})
        created_at = manifest.get("created_at_utc")
        mode = "locked"
    else:
        manifest = {}
        summary = load_json(run_summary_path) or {}
        round_validation = {
            "all_rounds_present": None,
            "expected_rounds": None,
            "produced_rounds": summary.get("inference_rows") if isinstance(summary, dict) else None,
            "missing_events": [],
            "extra_events": [],
        }
        outputs_map = summary.get("outputs", {}) if isinstance(summary, dict) else {}
        created_at = None
        mode = "reports"

    strategy_path = _resolve_path(
        outputs_map.get("strategy_recommendations") if isinstance(outputs_map, dict) else None,
        run_dir,
        "strategy_recommendations_2025.csv",
    )
    championship_path = _resolve_path(
        outputs_map.get("championship_projection") if isinstance(outputs_map, dict) else None,
        run_dir,
        "championship_projection_2025.json",
    )

    rows = _read_strategy_rows(strategy_path)
    championship = load_json(championship_path) or {}
    # Backward compatibility: strip deprecated championship point projections.
    if isinstance(championship, dict):
        championship.pop("projected_driver_points", None)
        championship.pop("projected_teammate_points", None)
        championship.pop("projected_constructors_points", None)

    payload = {
        "source": {
            "mode": mode,
            "path": str(run_dir),
            "created_at_utc": created_at,
            "manifest_present": manifest_path.exists(),
        },
        "summary": summary,
        "round_validation": round_validation,
        "kpis": _build_kpis(summary, championship),
        "championship": championship,
        "strategy_rows": rows,
        "top_rounds": _top_rounds(rows),
        "manifest": manifest,
    }
    return payload


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>F1 Strategy Control Website</title>
  <style>
    :root {
      --bg: #04050a;
      --bg-soft: #0b111b;
      --surface: rgba(13, 19, 30, 0.78);
      --surface-2: rgba(16, 24, 39, 0.82);
      --line: rgba(156, 182, 226, 0.18);
      --line-strong: rgba(156, 182, 226, 0.28);
      --ink: #edf3ff;
      --muted: #a0afcb;
      --accent: #5ea1ff;
      --accent-warm: #ff8b4a;
      --accent-2: #2ad0b8;
      --good: #58dc96;
      --warn: #ffc45a;
      --bad: #ff6f7d;
      --radius-lg: 18px;
      --radius-md: 14px;
      --radius-sm: 12px;
      --elevation-1: 0 10px 22px rgba(0, 0, 0, 0.28);
      --elevation-2: 0 22px 40px rgba(0, 0, 0, 0.36);
      --spring: cubic-bezier(0.22, 1, 0.36, 1);
      --material: cubic-bezier(0.2, 0, 0, 1);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 22px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% -10%, rgba(94,161,255,0.22) 0%, transparent 34%),
        radial-gradient(circle at 92% -5%, rgba(255,139,74,0.15) 0%, transparent 32%),
        radial-gradient(circle at 50% 120%, rgba(42,208,184,0.12) 0%, transparent 36%),
        linear-gradient(160deg, #020307 0%, #050914 45%, #03050c 100%);
      min-height: 100vh;
      letter-spacing: 0.01em;
      -webkit-font-smoothing: antialiased;
    }
    .site {
      max-width: 1240px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
      isolation: isolate;
    }
    .hero {
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background:
        linear-gradient(130deg, rgba(20, 31, 48, 0.86) 0%, rgba(19, 30, 46, 0.72) 70%, rgba(16, 25, 39, 0.86) 100%),
        var(--surface-2);
      backdrop-filter: blur(20px) saturate(130%);
      -webkit-backdrop-filter: blur(20px) saturate(130%);
      padding: 16px;
      box-shadow: var(--elevation-2);
      opacity: 0;
      transform: translateY(14px);
      animation: rise 560ms var(--spring) forwards;
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: -40% -22%;
      background: linear-gradient(100deg, transparent 38%, rgba(255,255,255,0.09) 50%, transparent 62%);
      transform: translateX(-45%);
      animation: sheen 5s ease-in-out infinite;
      pointer-events: none;
    }
    .hero-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .title {
      margin: 0 0 4px 0;
      font-size: 29px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .meta-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .pill {
      border: 1px solid var(--line);
      background: rgba(11, 17, 28, 0.66);
      backdrop-filter: blur(14px) saturate(120%);
      -webkit-backdrop-filter: blur(14px) saturate(120%);
      border-radius: 999px;
      color: #d2dcf3;
      padding: 4px 11px;
      font-size: 12px;
      max-width: 100%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .btn {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(160deg, rgba(17, 27, 44, 0.84) 0%, rgba(15, 25, 40, 0.78) 100%);
      backdrop-filter: blur(14px) saturate(120%);
      -webkit-backdrop-filter: blur(14px) saturate(120%);
      color: #dce7ff;
      border-radius: 12px;
      font-size: 12px;
      padding: 8px 11px;
      cursor: pointer;
      transition: border-color 180ms var(--material), transform 180ms var(--spring), background-color 180ms var(--material), box-shadow 180ms var(--material);
    }
    .btn:hover {
      border-color: var(--line-strong);
      background: linear-gradient(160deg, rgba(22, 35, 57, 0.9) 0%, rgba(18, 30, 49, 0.84) 100%);
      transform: translateY(-1px);
      box-shadow: var(--elevation-1);
    }
    .btn:active { transform: scale(0.985); }
    .btn.primary {
      border-color: rgba(42, 208, 184, 0.5);
      background: linear-gradient(120deg, rgba(42,208,184,0.24), rgba(42,208,184,0.1));
      color: #d3fff5;
    }
    .btn.warn {
      border-color: rgba(255, 196, 90, 0.44);
      background: rgba(255,196,90,0.14);
      color: #ffdca0;
    }
    .ripple {
      position: absolute;
      border-radius: 50%;
      transform: scale(0);
      background: rgba(255, 255, 255, 0.38);
      pointer-events: none;
      animation: ripple 680ms var(--material) forwards;
      mix-blend-mode: screen;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      border: 1px solid var(--line);
      background: rgba(7, 12, 20, 0.62);
      border-radius: 999px;
      padding: 5px;
      backdrop-filter: blur(18px) saturate(130%);
      -webkit-backdrop-filter: blur(18px) saturate(130%);
      width: fit-content;
    }
    .tab {
      position: relative;
      overflow: hidden;
      border: 1px solid transparent;
      background: rgba(9, 16, 26, 0.74);
      color: #c4d0e8;
      border-radius: 999px;
      font-size: 12px;
      padding: 8px 13px;
      cursor: pointer;
      transition: border-color 180ms var(--material), color 180ms var(--material), background-color 180ms var(--material), transform 180ms var(--spring);
    }
    .tab:hover { border-color: var(--line-strong); color: #edf3ff; transform: translateY(-1px); }
    .tab.active {
      color: #f7fbff;
      border-color: rgba(94, 161, 255, 0.4);
      background: linear-gradient(120deg, rgba(94,161,255,0.28), rgba(94,161,255,0.1));
      box-shadow: inset 0 0 0 1px rgba(156, 195, 255, 0.18);
    }
    .page {
      display: none;
      opacity: 0;
      transform: translateY(8px);
    }
    .page.active {
      display: block;
      animation: rise 320ms cubic-bezier(.2,.7,.2,1) forwards;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .card {
      opacity: 0;
      transform: translateY(10px);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(165deg, rgba(11, 18, 31, 0.88) 0%, rgba(15, 24, 39, 0.8) 100%);
      backdrop-filter: blur(14px) saturate(122%);
      -webkit-backdrop-filter: blur(14px) saturate(122%);
      padding: 12px;
      transition: border-color 180ms var(--material), transform 180ms var(--spring), box-shadow 180ms var(--material);
    }
    .card.reveal { animation: rise 360ms var(--delay, 0ms) cubic-bezier(.2,.7,.2,1) forwards; }
    .card:hover { border-color: var(--line-strong); transform: translateY(-2px); box-shadow: var(--elevation-1); }
    .label { margin: 0 0 5px 0; color: var(--muted); font-size: 12px; }
    .value { margin: 0; font-size: 22px; font-weight: 700; color: var(--ink); }
    .grid-2 {
      display: grid;
      grid-template-columns: 1.05fr 1.95fr;
      gap: 12px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(170deg, rgba(10, 16, 26, 0.86) 0%, rgba(14, 22, 34, 0.8) 100%);
      backdrop-filter: blur(16px) saturate(125%);
      -webkit-backdrop-filter: blur(16px) saturate(125%);
      padding: 12px;
      box-shadow: var(--elevation-1);
    }
    .panel h3 {
      margin: 0 0 8px 0;
      font-size: 15px;
      letter-spacing: 0.01em;
    }
    .status-good { color: var(--good); font-weight: 700; }
    .status-warn { color: var(--warn); font-weight: 700; }
    .status-bad { color: var(--bad); font-weight: 700; }
    .bar-row {
      display: grid;
      grid-template-columns: 150px 1fr 46px;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
      font-size: 12px;
      opacity: 0;
      transform: translateX(-8px);
    }
    .bar-row.reveal { animation: slide 380ms var(--delay, 0ms) cubic-bezier(.2,.7,.2,1) forwards; }
    .bar {
      height: 8px;
      border-radius: 99px;
      background: #1a2437;
      overflow: hidden;
    }
    .bar span {
      display: block;
      width: 0%;
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
      box-shadow: 0 0 12px rgba(42, 208, 184, 0.42);
      transition: width 860ms cubic-bezier(.2,.8,.2,1);
    }
    .spark {
      width: 100%;
      height: 78px;
      border: 1px solid #1f2a3f;
      border-radius: 10px;
      background: #0b111b;
      margin-top: 10px;
      display: block;
    }
    .controls {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(170deg, rgba(10, 17, 26, 0.86) 0%, rgba(13, 22, 35, 0.8) 100%);
      backdrop-filter: blur(16px) saturate(125%);
      -webkit-backdrop-filter: blur(16px) saturate(125%);
      padding: 12px;
      margin-bottom: 10px;
      display: grid;
      gap: 10px;
    }
    .row {
      display: grid;
      grid-template-columns: 1.5fr 0.7fr 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .field {
      display: grid;
      gap: 4px;
    }
    .field label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    input[type='search'], select, input[type='range'] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: rgba(11, 17, 27, 0.82);
      backdrop-filter: blur(10px);
      color: var(--ink);
      font-size: 13px;
      padding: 7px 9px;
      outline: none;
    }
    input[type='search']:focus, select:focus {
      border-color: rgba(94, 161, 255, 0.58);
      box-shadow: 0 0 0 3px rgba(94, 161, 255, 0.18);
    }
    .table-wrap {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(170deg, rgba(9, 16, 26, 0.84) 0%, rgba(13, 21, 34, 0.8) 100%);
      backdrop-filter: blur(18px) saturate(124%);
      -webkit-backdrop-filter: blur(18px) saturate(124%);
      padding: 12px;
      box-shadow: var(--elevation-1);
    }
    .table-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      gap: 8px;
      flex-wrap: wrap;
    }
    .hint { font-size: 12px; color: var(--muted); margin: 0; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #1d2638;
      vertical-align: top;
    }
    thead th { background: rgba(17, 26, 42, 0.8); color: #dbe5fb; }
    .th-btn {
      position: relative;
      overflow: hidden;
      display: inline-flex;
      align-items: center;
      border: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      cursor: pointer;
      padding: 0;
    }
    tbody tr {
      opacity: 0;
      transform: translateY(8px);
      transition: background-color 150ms ease;
      cursor: pointer;
    }
    tbody tr.reveal { animation: rise 360ms var(--delay, 0ms) cubic-bezier(.2,.7,.2,1) forwards; }
    tbody tr:hover { background: rgba(42, 58, 87, 0.38); }
    .mono { font-family: "SFMono-Regular", Menlo, monospace; color: #9db1d2; font-size: 11px; }
    .sim-grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
    }
    .sim-row {
      display: grid;
      gap: 8px;
      margin: 10px 0;
    }
    .sim-metric {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px dashed #23314a;
      font-size: 13px;
    }
    .sim-metric strong { font-size: 18px; color: #f2f7ff; }
    .integrity-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .list {
      margin: 0;
      padding-left: 18px;
      color: #d9e5ff;
      font-size: 13px;
    }
    .hash-list {
      max-height: 320px;
      overflow: auto;
      display: grid;
      gap: 8px;
    }
    .hash-item {
      border: 1px solid #23314a;
      border-radius: var(--radius-sm);
      padding: 8px;
      background: rgba(10, 16, 24, 0.78);
      backdrop-filter: blur(12px);
      display: grid;
      gap: 7px;
    }
    .hash-line {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .hash-line code {
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #9ac9df;
    }
    .raw {
      width: 100%;
      max-height: 340px;
      overflow: auto;
      margin-top: 8px;
      border: 1px solid #253550;
      border-radius: 10px;
      background: #080d16;
      color: #b7c7e8;
      font-size: 11px;
      padding: 10px;
      display: none;
    }
    .raw.show { display: block; }
    .drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.45);
      opacity: 0;
      pointer-events: none;
      transition: opacity 180ms ease;
      z-index: 18;
    }
    .drawer-backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      height: 100%;
      width: min(460px, 100%);
      background: linear-gradient(170deg, rgba(10, 17, 27, 0.94) 0%, rgba(14, 23, 37, 0.92) 100%);
      backdrop-filter: blur(24px) saturate(128%);
      -webkit-backdrop-filter: blur(24px) saturate(128%);
      border-left: 1px solid #283754;
      box-shadow: -14px 0 34px rgba(0, 0, 0, 0.38);
      transform: translateX(102%);
      transition: transform 300ms var(--spring);
      z-index: 20;
      padding: 14px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
    }
    .drawer.open { transform: translateX(0); }
    .drawer-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .drawer-body {
      overflow: auto;
      font-size: 13px;
      color: #dbe6ff;
      display: grid;
      gap: 8px;
    }
    .drawer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .kv {
      border: 1px solid #243550;
      border-radius: var(--radius-sm);
      background: rgba(10, 16, 24, 0.82);
      backdrop-filter: blur(11px);
      padding: 8px;
    }
    .kv p { margin: 0; }
    .kv .k { color: var(--muted); font-size: 11px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
    .kv .v { color: #f0f5ff; font-size: 15px; font-weight: 700; }
    .tag {
      display: inline-block;
      border: 1px solid #2c3a55;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      color: #ccd8ef;
      margin-right: 6px;
      margin-bottom: 6px;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(10px) scale(0.995); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes slide {
      from { opacity: 0; transform: translateX(-7px); }
      to { opacity: 1; transform: translateX(0); }
    }
    @keyframes sheen {
      0%, 18% { transform: translateX(-45%); opacity: 0; }
      35% { opacity: 1; }
      58%, 100% { transform: translateX(45%); opacity: 0; }
    }
    @keyframes ripple {
      0% { transform: scale(0); opacity: 0.45; }
      100% { transform: scale(4.5); opacity: 0; }
    }
    @media (max-width: 900px) {
      .grid-2, .sim-grid, .integrity-grid, .row {
        grid-template-columns: 1fr;
      }
      body { padding: 12px; }
      .title { font-size: 23px; }
      .actions { justify-content: flex-start; }
      .bar-row { grid-template-columns: 115px 1fr 44px; }
      .tabs {
        width: 100%;
        justify-content: space-between;
        position: sticky;
        bottom: 8px;
        z-index: 11;
        box-shadow: var(--elevation-1);
      }
      .tab {
        flex: 1;
        text-align: center;
      }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
      }
      .hero, .page, .card, .bar-row, tbody tr {
        opacity: 1 !important;
        transform: none !important;
      }
    }
  </style>
</head>
<body>
  <div class="site">
    <header class="hero">
      <div class="hero-top">
        <div>
          <h1 class="title">F1 Strategy Control Website</h1>
          <p class="sub" id="metaLine">Loading run data...</p>
        </div>
        <div class="actions">
          <button class="btn" id="downloadCsv">Download CSV</button>
          <button class="btn" id="downloadJson">Download JSON</button>
        </div>
      </div>
      <div class="meta-pills">
        <span class="pill" id="modePill">mode</span>
        <span class="pill" id="pathPill">path</span>
      </div>
      <nav class="tabs" id="tabs">
        <button class="tab active" data-route="overview">Overview</button>
        <button class="tab" data-route="races">Race Explorer</button>
        <button class="tab" data-route="simulator">Championship Lab</button>
        <button class="tab" data-route="integrity">Data Integrity</button>
      </nav>
    </header>

    <main>
      <section class="page active" id="page-overview">
        <section class="cards" id="kpiCards"></section>
        <section class="grid-2">
          <div class="panel">
            <h3>Round Validation</h3>
            <p id="roundStatus"></p>
            <p class="sub" id="roundDetail"></p>
            <svg class="spark" id="pointsSpark" viewBox="0 0 600 90" preserveAspectRatio="none"></svg>
          </div>
          <div class="panel">
            <h3>Top Rounds by Strategy Score</h3>
            <div id="topRounds"></div>
          </div>
        </section>
      </section>

      <section class="page" id="page-races">
        <section class="controls">
          <div class="row">
            <div class="field">
              <label for="searchInput">Search</label>
              <input id="searchInput" type="search" placeholder="Event, strategy, compounds..." />
            </div>
            <div class="field">
              <label for="stopFilter">Stops</label>
              <select id="stopFilter">
                <option value="all">All</option>
                <option value="1">1 stop</option>
                <option value="2">2 stops</option>
              </select>
            </div>
            <div class="field">
              <label for="minWin">Min win probability</label>
              <input id="minWin" type="range" min="0" max="1" step="0.01" value="0" />
            </div>
            <div>
              <button class="btn warn" id="resetFilters">Reset</button>
            </div>
          </div>
          <p class="hint" id="filterState">No filters active.</p>
        </section>
        <section class="table-wrap">
          <div class="table-header">
            <strong>Strategy Table (click any row for details)</strong>
            <p class="hint" id="tableCount">0 rows</p>
          </div>
          <div style="overflow:auto;">
            <table>
              <thead>
                <tr>
                  <th><button class="th-btn" data-sort="event_name">Event</button></th>
                  <th><button class="th-btn" data-sort="strategy_plan">Primary Race Plan</button></th>
                  <th><button class="th-btn" data-sort="stops">Stops</button></th>
                  <th>Start Tyre</th>
                  <th>First Pit</th>
                  <th><button class="th-btn" data-sort="win_probability">Win Prob</button></th>
                  <th><button class="th-btn" data-sort="strategy_score">Score</button></th>
                </tr>
              </thead>
              <tbody id="raceRows"></tbody>
            </table>
          </div>
        </section>
      </section>

      <section class="page" id="page-simulator">
        <section class="sim-grid">
          <div class="panel">
            <h3>Interactive Championship What-If</h3>
            <p class="sub">Adjust form assumptions to simulate alternate title-probability outcomes.</p>
            <div class="sim-row">
              <label for="driverDelta">Driver form adjustment: <strong id="driverDeltaValue">0 pp</strong></label>
              <input id="driverDelta" type="range" min="-30" max="30" step="1" value="0" />
            </div>
            <div class="sim-row">
              <label for="teammateFactor">Team synergy adjustment: <strong id="teammateFactorValue">0 pp</strong></label>
              <input id="teammateFactor" type="range" min="-30" max="30" step="1" value="0" />
            </div>
            <p class="hint">Adjustments are in percentage points of title confidence (pp).</p>
          </div>
          <div class="panel">
            <h3>Projected Outcome</h3>
            <div id="simMetrics"></div>
          </div>
        </section>
      </section>

      <section class="page" id="page-integrity">
        <section class="integrity-grid">
          <div class="panel">
            <h3>Run Validation</h3>
            <ul class="list" id="integritySummary"></ul>
            <button class="btn" id="toggleManifest">Toggle Raw Manifest</button>
            <pre class="raw" id="manifestRaw"></pre>
          </div>
          <div class="panel">
            <h3>Checksums (click copy)</h3>
            <div class="hash-list" id="hashList"></div>
          </div>
        </section>
      </section>
    </main>
  </div>

  <div class="drawer-backdrop" id="drawerBackdrop"></div>
  <aside class="drawer" id="raceDrawer">
    <div class="drawer-head">
      <strong id="drawerTitle">Race Detail</strong>
      <button class="btn" id="closeDrawer">Close</button>
    </div>
    <div class="drawer-body" id="drawerBody"></div>
  </aside>

  <script>
    const routes = ['overview', 'races', 'simulator', 'integrity'];
    const state = {
      payload: null,
      sortKey: 'strategy_score',
      sortDir: 'desc',
      filteredRows: []
    };

    const fmt = (v, n=3) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(n);
    const pct = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : `${(Number(v) * 100).toFixed(1)}%`;
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const logit = (p) => Math.log(p / (1 - p));
    const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
    const supportsViewTransition = typeof document.startViewTransition === 'function';

    function safeText(value) {
      return (value === null || value === undefined) ? '-' : String(value);
    }

    function routeFromHash() {
      const route = (window.location.hash || '').replace('#', '');
      if (routes.includes(route)) return route;
      const byPath = window.location.pathname.replace('/', '');
      if (routes.includes(byPath)) return byPath;
      return routes.includes(route) ? route : 'overview';
    }

    function animateRouteSwap(fn) {
      if (supportsViewTransition) {
        document.startViewTransition(fn);
        return;
      }
      fn();
    }

    function setRoute(route) {
      const finalRoute = routes.includes(route) ? route : 'overview';
      if (window.location.hash !== `#${finalRoute}`) {
        window.location.hash = finalRoute;
      } else {
        animateRouteSwap(() => renderRoute(finalRoute));
      }
    }

    function renderRoute(route) {
      for (const page of document.querySelectorAll('.page')) {
        page.classList.toggle('active', page.id === `page-${route}`);
      }
      for (const tab of document.querySelectorAll('.tab')) {
        tab.classList.toggle('active', tab.dataset.route === route);
      }
    }

    function spawnRipple(target, clientX, clientY) {
      const rect = target.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height) * 1.1;
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.width = `${size}px`;
      ripple.style.height = `${size}px`;
      ripple.style.left = `${clientX - rect.left - size / 2}px`;
      ripple.style.top = `${clientY - rect.top - size / 2}px`;
      target.appendChild(ripple);
      ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
    }

    function createCard(label, value, idx) {
      const card = document.createElement('div');
      card.className = 'card reveal';
      card.style.setProperty('--delay', `${80 + idx * 45}ms`);
      card.innerHTML = `<p class='label'>${label}</p><p class='value'>${value}</p>`;
      return card;
    }

    function renderOverview(payload) {
      const k = payload.kpis || {};
      const cards = document.getElementById('kpiCards');
      cards.innerHTML = '';
      const defs = [
        ['Training Rows', k.training_rows ?? '-'],
        ['Inference Rows', k.inference_rows ?? '-'],
        ['MAE', fmt(k.mae, 3)],
        ['RMSE', fmt(k.rmse, 3)],
        ['Driver Title Prob', pct(k.driver_title_probability)],
        ['Constructors Title Prob', pct(k.constructors_title_probability)]
      ];
      defs.forEach((d, i) => cards.appendChild(createCard(d[0], d[1], i)));

      const rv = payload.round_validation || {};
      const status = document.getElementById('roundStatus');
      const all = rv.all_rounds_present;
      if (all === true) {
        status.className = 'status-good';
        status.textContent = 'All rounds present';
      } else if (all === false) {
        status.className = 'status-bad';
        status.textContent = 'Coverage gap detected';
      } else {
        status.className = 'status-warn';
        status.textContent = 'Validation unavailable';
      }
      document.getElementById('roundDetail').textContent =
        `Expected: ${rv.expected_rounds ?? '-'} | Produced: ${rv.produced_rounds ?? '-'} | Missing: ${(rv.missing_events || []).length}`;

      const top = document.getElementById('topRounds');
      top.innerHTML = '';
      const topRows = payload.top_rounds || [];
      const maxPts = Math.max(...topRows.map(r => Number(r.strategy_score || 0)), 1);
      topRows.forEach((r, i) => {
        const val = Number(r.strategy_score || 0);
        const width = Math.max(3, Math.round((val / maxPts) * 100));
        const row = document.createElement('div');
        row.className = 'bar-row reveal';
        row.style.setProperty('--delay', `${120 + (i * 70)}ms`);
        row.innerHTML = `
          <span>${safeText(r.event_name)}</span>
          <div class='bar'><span data-width='${width}%'></span></div>
          <span>${fmt(val, 3)}</span>
        `;
        top.appendChild(row);
        const fill = row.querySelector('span[data-width]');
        requestAnimationFrame(() => { fill.style.width = fill.dataset.width || '0%'; });
      });

      renderSparkline(payload.strategy_rows || []);
    }

    function renderSparkline(rows) {
      const svg = document.getElementById('pointsSpark');
      if (!rows.length) {
        svg.innerHTML = '';
        return;
      }
      const pts = rows.map(r => Number(r.strategy_score || 0));
      const min = Math.min(...pts);
      const max = Math.max(...pts);
      const span = (max - min) || 1;
      const width = 600;
      const height = 90;
      const pad = 10;
      const toXY = (v, i) => {
        const x = pad + (i * ((width - pad * 2) / Math.max(rows.length - 1, 1)));
        const y = height - pad - (((v - min) / span) * (height - pad * 2));
        return [x, y];
      };
      const coords = pts.map((v, i) => toXY(v, i));
      const line = coords.map((c, i) => `${i === 0 ? 'M' : 'L'}${c[0].toFixed(2)},${c[1].toFixed(2)}`).join(' ');
      const area = `${line} L${coords[coords.length - 1][0]},${height - pad} L${coords[0][0]},${height - pad} Z`;
      svg.innerHTML = `
        <defs>
          <linearGradient id='g1' x1='0' y1='0' x2='0' y2='1'>
            <stop offset='0%' stop-color='rgba(42,208,184,0.4)'/>
            <stop offset='100%' stop-color='rgba(42,208,184,0)'/>
          </linearGradient>
        </defs>
        <path d='${area}' fill='url(#g1)'></path>
        <path d='${line}' fill='none' stroke='#35ceb8' stroke-width='2.2'></path>
      `;
    }

    function compare(a, b, key) {
      const av = a[key];
      const bv = b[key];
      const an = Number(av);
      const bn = Number(bv);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
      return safeText(av).localeCompare(safeText(bv));
    }

    function filteredRaceRows() {
      const rows = state.payload.strategy_rows || [];
      const search = document.getElementById('searchInput').value.trim().toLowerCase();
      const stop = document.getElementById('stopFilter').value;
      const minWin = Number(document.getElementById('minWin').value || 0);
      const out = rows.filter(r => {
        const blob = `${safeText(r.event_name)} ${safeText(r.best_strategy)} ${safeText(r.strategy_plan)} ${safeText(r.compounds)} ${safeText(r.fallback_2_plan)} ${safeText(r.fallback_3_plan)} ${safeText(r.fallback_2_trigger)} ${safeText(r.fallback_3_trigger)}`.toLowerCase();
        const okSearch = !search || blob.includes(search);
        const okStop = stop === 'all' || String(Math.round(Number(r.stops || 0))) === stop;
        const okWin = Number(r.win_probability || 0) >= minWin;
        return okSearch && okStop && okWin;
      });
      out.sort((a, b) => {
        const base = compare(a, b, state.sortKey);
        return state.sortDir === 'asc' ? base : -base;
      });
      return out;
    }

    function renderRaceTable() {
      const rows = filteredRaceRows();
      state.filteredRows = rows;
      const body = document.getElementById('raceRows');
      body.innerHTML = '';
      rows.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.className = 'reveal';
        tr.style.setProperty('--delay', `${Math.min(i * 12, 260)}ms`);
        tr.dataset.index = String(i);
        tr.innerHTML = `
          <td>${safeText(r.event_name)}</td>
          <td>${safeText(r.strategy_plan || r.best_strategy)}</td>
          <td>${fmt(r.stops, 0)}</td>
          <td>${safeText(r.start_compound || '-')}</td>
          <td>${r.first_pit_lap == null ? '-' : `L${fmt(r.first_pit_lap, 0)}`}</td>
          <td>${pct(r.win_probability)}</td>
          <td>${fmt(r.strategy_score, 2)}</td>
        `;
        body.appendChild(tr);
      });
      document.getElementById('tableCount').textContent = `${rows.length} rows`;
      const minWin = Number(document.getElementById('minWin').value || 0);
      const stop = document.getElementById('stopFilter').value;
      const parts = [];
      if (minWin > 0) parts.push(`min win ${pct(minWin)}`);
      if (stop !== 'all') parts.push(`${stop} stop`);
      if (document.getElementById('searchInput').value.trim()) parts.push('search active');
      document.getElementById('filterState').textContent = parts.length ? parts.join(' | ') : 'No filters active.';
    }

    function openDrawer(row) {
      if (!row) return;
      document.getElementById('drawerTitle').textContent = safeText(row.event_name);
      const body = document.getElementById('drawerBody');
      body.innerHTML = `
        <div>
          <span class='tag'>${safeText(row.team)}</span>
          <span class='tag'>${safeText(row.driver)}</span>
          <span class='tag'>${safeText(row.start_compound || row.compounds)}</span>
        </div>
        <div class='drawer-grid'>
          <div class='kv'><p class='k'>Best Strategy</p><p class='v mono'>${safeText(row.best_strategy)}</p></div>
          <div class='kv'><p class='k'>Primary Plan</p><p class='v'>${safeText(row.strategy_plan)}</p></div>
          <div class='kv'><p class='k'>Pit Laps</p><p class='v'>${safeText(row.pit_laps)}</p></div>
          <div class='kv'><p class='k'>Win Probability</p><p class='v'>${pct(row.win_probability)}</p></div>
          <div class='kv'><p class='k'>Strategy Score</p><p class='v'>${fmt(row.strategy_score, 2)}</p></div>
          <div class='kv'><p class='k'>Robustness Window</p><p class='v'>${fmt(row.robustness_window, 2)}s</p></div>
          <div class='kv'><p class='k'>Expected Race Time</p><p class='v'>${fmt(row.expected_race_time, 2)}s</p></div>
          <div class='kv'><p class='k'>Pred Base Lap</p><p class='v'>${fmt(row.predicted_base_lap_sec, 3)}s</p></div>
          <div class='kv'><p class='k'>Fallback #2</p><p class='v'>${safeText(row.fallback_2_plan || row.fallback_2_strategy)}</p></div>
          <div class='kv'><p class='k'>Fallback #2 Trigger</p><p class='v'>${safeText(row.fallback_2_trigger)}</p></div>
          <div class='kv'><p class='k'>Fallback #3</p><p class='v'>${safeText(row.fallback_3_plan || row.fallback_3_strategy)}</p></div>
          <div class='kv'><p class='k'>Fallback #3 Trigger</p><p class='v'>${safeText(row.fallback_3_trigger)}</p></div>
        </div>
      `;
      document.getElementById('drawerBackdrop').classList.add('show');
      document.getElementById('raceDrawer').classList.add('open');
    }

    function closeDrawer() {
      document.getElementById('drawerBackdrop').classList.remove('show');
      document.getElementById('raceDrawer').classList.remove('open');
    }

    function renderSimulator() {
      const champ = state.payload.championship || {};
      const baseDriverProb = clamp(Number(champ.driver_title_probability || 0), 0.001, 0.999);
      const baseConstructorsProb = clamp(Number(champ.constructors_title_probability || 0), 0.001, 0.999);
      const driverDeltaPP = Number(document.getElementById('driverDelta').value || 0);
      const teamDeltaPP = Number(document.getElementById('teammateFactor').value || 0);

      const driverProb = sigmoid(logit(baseDriverProb) + driverDeltaPP / 9.5);
      const constructorsProb = sigmoid(logit(baseConstructorsProb) + (driverDeltaPP + teamDeltaPP) / 11.0);
      const titleStrength = (driverProb * 0.55 + constructorsProb * 0.45) * 100.0;

      document.getElementById('driverDeltaValue').textContent = `${driverDeltaPP > 0 ? '+' : ''}${driverDeltaPP} pp`;
      document.getElementById('teammateFactorValue').textContent = `${teamDeltaPP > 0 ? '+' : ''}${teamDeltaPP} pp`;

      const sim = document.getElementById('simMetrics');
      sim.innerHTML = `
        <div class='sim-metric'><span>Baseline driver title probability</span><strong>${pct(baseDriverProb)}</strong></div>
        <div class='sim-metric'><span>Baseline constructors title probability</span><strong>${pct(baseConstructorsProb)}</strong></div>
        <div class='sim-metric'><span>Scenario driver title probability</span><strong>${pct(driverProb)}</strong></div>
        <div class='sim-metric'><span>Scenario constructors title probability</span><strong>${pct(constructorsProb)}</strong></div>
        <div class='sim-metric'><span>Combined title strength index</span><strong>${fmt(titleStrength, 1)}</strong></div>
      `;
    }

    function renderIntegrity(payload) {
      const rv = payload.round_validation || {};
      const src = payload.source || {};
      const summary = document.getElementById('integritySummary');
      summary.innerHTML = '';
      const items = [
        `Mode: ${safeText(src.mode)}`,
        `Manifest present: ${src.manifest_present ? 'yes' : 'no'}`,
        `Expected rounds: ${safeText(rv.expected_rounds)}`,
        `Produced rounds: ${safeText(rv.produced_rounds)}`,
        `Missing events: ${(rv.missing_events || []).length}`,
        `Extra events: ${(rv.extra_events || []).length}`
      ];
      items.forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        summary.appendChild(li);
      });

      const hashes = (payload.manifest && payload.manifest.sha256) ? payload.manifest.sha256 : {};
      const list = document.getElementById('hashList');
      list.innerHTML = '';
      const entries = Object.entries(hashes);
      if (!entries.length) {
        const empty = document.createElement('p');
        empty.className = 'hint';
        empty.textContent = 'No manifest checksum data found. Run a locked snapshot to populate this.';
        list.appendChild(empty);
      } else {
        entries.forEach(([name, hash]) => {
          const item = document.createElement('div');
          item.className = 'hash-item';
          item.innerHTML = `
            <div class='hash-line'><strong>${name}</strong><button class='btn' data-copy='${hash}'>Copy</button></div>
            <code>${hash}</code>
          `;
          list.appendChild(item);
        });
      }

      document.getElementById('manifestRaw').textContent = JSON.stringify(payload.manifest || {}, null, 2);
    }

    function toCsv(rows) {
      if (!rows.length) return '';
      const cols = Object.keys(rows[0]);
      const esc = (v) => {
        const s = v === null || v === undefined ? '' : String(v);
        return `"${s.replaceAll('"', '""')}"`;
      };
      const lines = [cols.join(',')];
      rows.forEach(r => lines.push(cols.map(c => esc(r[c])).join(',')));
      return lines.join('\\n');
    }

    function downloadText(filename, text, type) {
      const blob = new Blob([text], { type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function bindEvents() {
      document.getElementById('tabs').addEventListener('click', (event) => {
        const btn = event.target.closest('.tab');
        if (!btn) return;
        setRoute(btn.dataset.route || 'overview');
      });
      window.addEventListener('hashchange', () => {
        animateRouteSwap(() => renderRoute(routeFromHash()));
      });

      document.body.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        const target = event.target.closest('.btn, .tab, .th-btn');
        if (!target) return;
        spawnRipple(target, event.clientX, event.clientY);
      });

      document.getElementById('searchInput').addEventListener('input', renderRaceTable);
      document.getElementById('stopFilter').addEventListener('change', renderRaceTable);
      document.getElementById('minWin').addEventListener('input', renderRaceTable);
      document.getElementById('resetFilters').addEventListener('click', () => {
        document.getElementById('searchInput').value = '';
        document.getElementById('stopFilter').value = 'all';
        document.getElementById('minWin').value = '0';
        renderRaceTable();
      });
      document.querySelector('thead').addEventListener('click', (event) => {
        const btn = event.target.closest('.th-btn');
        if (!btn) return;
        const key = btn.dataset.sort;
        if (!key) return;
        if (state.sortKey === key) {
          state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          state.sortKey = key;
          state.sortDir = 'desc';
        }
        renderRaceTable();
      });
      document.getElementById('raceRows').addEventListener('click', (event) => {
        const tr = event.target.closest('tr');
        if (!tr) return;
        const idx = Number(tr.dataset.index);
        openDrawer(state.filteredRows[idx]);
      });
      document.getElementById('closeDrawer').addEventListener('click', closeDrawer);
      document.getElementById('drawerBackdrop').addEventListener('click', closeDrawer);

      document.getElementById('driverDelta').addEventListener('input', renderSimulator);
      document.getElementById('teammateFactor').addEventListener('input', renderSimulator);

      document.getElementById('hashList').addEventListener('click', async (event) => {
        const btn = event.target.closest('button[data-copy]');
        if (!btn) return;
        const val = btn.getAttribute('data-copy') || '';
        try {
          await navigator.clipboard.writeText(val);
          btn.textContent = 'Copied';
          setTimeout(() => { btn.textContent = 'Copy'; }, 900);
        } catch {
          btn.textContent = 'Failed';
        }
      });
      document.getElementById('toggleManifest').addEventListener('click', () => {
        document.getElementById('manifestRaw').classList.toggle('show');
      });

      document.getElementById('downloadCsv').addEventListener('click', () => {
        const rows = state.payload ? (state.payload.strategy_rows || []) : [];
        downloadText('strategy_recommendations_2025.csv', toCsv(rows), 'text/csv;charset=utf-8');
      });
      document.getElementById('downloadJson').addEventListener('click', () => {
        downloadText('strategy_payload.json', JSON.stringify(state.payload || {}, null, 2), 'application/json');
      });
    }

    function renderHeader(payload) {
      const src = payload.source || {};
      document.getElementById('modePill').textContent = `mode: ${safeText(src.mode)}`;
      document.getElementById('pathPill').textContent = safeText(src.path);
      document.getElementById('metaLine').textContent = src.created_at_utc
        ? `Snapshot created (UTC): ${src.created_at_utc}`
        : 'Using reports output (no lock manifest found).';
    }

    async function loadPayload() {
      const sources = ['/api/data', '/data/payload.json'];
      let lastErr = null;
      for (const src of sources) {
        try {
          const res = await fetch(src, { cache: 'no-store' });
          if (!res.ok) throw new Error(`Request failed (${src}): ${res.status}`);
          return await res.json();
        } catch (err) {
          lastErr = err;
        }
      }
      throw lastErr || new Error('No payload source available.');
    }

    async function boot() {
      const payload = await loadPayload();
      state.payload = payload;

      renderHeader(payload);
      renderOverview(payload);
      renderRaceTable();
      renderSimulator();
      renderIntegrity(payload);
      bindEvents();
      renderRoute(routeFromHash());
    }

    boot().catch((err) => {
      document.getElementById('metaLine').textContent = `Failed to load website data: ${err}`;
    });
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    payload_text = "{}"
    html_text = _dashboard_html()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html", "/overview", "/races", "/simulator", "/integrity"}:
            body = self.html_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/data":
            body = self.payload_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"not found")

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_dashboard(payload: dict[str, Any], host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = _Handler
    handler.payload_text = json.dumps(payload)
    handler.html_text = _dashboard_html()

    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard available at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    finally:
        server.server_close()

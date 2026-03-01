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
            float(r.get("win_probability") or 0.0),
            -float(r.get("expected_race_time") or 0.0),
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
  <title>dnf - A Race strategy prediction model</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #040404;
      --bg-soft: #0a0a0a;
      --surface: rgba(17, 17, 17, 0.72);
      --surface-2: rgba(12, 12, 12, 0.9);
      --line: rgba(255, 255, 255, 0.08);
      --line-strong: rgba(255, 255, 255, 0.2);
      --ink: #f5f5f5;
      --muted: #a6a6a6;
      --accent: #ff8a00;
      --accent-soft: rgba(255, 138, 0, 0.24);
      --good: #63d49a;
      --warn: #f4c16f;
      --bad: #f67480;
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 12px;
      --shadow: 0 22px 44px rgba(0, 0, 0, 0.4);
      --ease: cubic-bezier(.22,.8,.2,1);
      --mono: "IBM Plex Mono", Menlo, monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 36px;
      font-family: "Sora", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(90% 90% at 90% -10%, rgba(255, 138, 0, 0.2) 0%, transparent 52%),
        radial-gradient(80% 70% at -10% 0%, rgba(255, 255, 255, 0.05) 0%, transparent 42%),
        linear-gradient(170deg, #020202 0%, #050505 60%, #020202 100%);
      min-height: 100vh;
      letter-spacing: 0.01em;
      -webkit-font-smoothing: antialiased;
      position: relative;
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 42px 42px;
      opacity: 0.32;
      pointer-events: none;
      z-index: 0;
      mask-image: radial-gradient(circle at center, black 45%, transparent 100%);
      -webkit-mask-image: radial-gradient(circle at center, black 45%, transparent 100%);
    }

    .loading-screen {
      position: fixed;
      inset: 0;
      z-index: 100;
      background:
        radial-gradient(90% 90% at 50% -10%, rgba(255, 138, 0,0.2) 0%, rgba(255, 138, 0,0) 55%),
        linear-gradient(180deg, #050505 0%, #090909 100%);
      display: grid;
      place-items: center;
      transition: opacity 520ms var(--ease), visibility 520ms var(--ease);
    }

    .loading-screen.hide {
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }

    .loading-scene {
      width: min(800px, 92vw);
      display: grid;
      gap: 12px;
      justify-items: center;
    }

    .loading-track {
      width: 100%;
      height: 86px;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 999px;
      background: rgba(12,12,12,0.92);
      overflow: hidden;
      position: relative;
    }

    .loading-track::before {
      content: "";
      position: absolute;
      left: 12px;
      right: 12px;
      top: 50%;
      height: 2px;
      transform: translateY(-50%);
      background: repeating-linear-gradient(90deg, rgba(255,255,255,0.22) 0 30px, transparent 30px 54px);
      animation: roadMove 1300ms linear infinite;
    }

    .loading-car {
      position: absolute;
      left: auto;
      right: -200px;
      top: 14px;
      width: 180px;
      height: 58px;
      animation: driveOff 1800ms var(--ease) forwards;
      filter: drop-shadow(0 0 14px rgba(255, 138, 0,0.35));
    }

    .loading-text {
      margin: 0;
      font-size: 12px;
      color: #bababa;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .car-bg {
      position: fixed;
      right: -150px;
      top: 56px;
      width: min(1240px, 98vw);
      opacity: 0.14;
      z-index: 0;
      pointer-events: none;
      --parallax-x: 0px;
      --parallax-y: 0px;
      --hero-glow: 1;
      --hero-progress: 0;
      --hero-fade: 1;
      isolation: isolate;
      transform: translate3d(var(--parallax-x), var(--parallax-y), 0);
      filter: drop-shadow(0 0 24px rgba(0, 0, 0, 0.58));
      transition: opacity 420ms var(--ease), transform 420ms var(--ease), right 420ms var(--ease), width 420ms var(--ease), top 420ms var(--ease);
    }

    .car-bg::before {
      content: "";
      position: absolute;
      inset: 12% -12% 18% 34%;
      background: repeating-linear-gradient(
        96deg,
        rgba(255, 164, 31, 0.22) 0 2px,
        transparent 2px 24px
      );
      opacity: 0;
      z-index: 0;
      pointer-events: none;
      filter: blur(0.6px);
      transition: opacity 420ms var(--ease);
    }

    .car-bg::after {
      content: "";
      position: absolute;
      inset: 24% -8% 18% 14%;
      background:
        radial-gradient(58% 44% at 46% 54%, rgba(255, 145, 20, 0.44) 0%, rgba(255, 145, 20, 0.14) 45%, rgba(255, 145, 20, 0) 100%),
        radial-gradient(44% 30% at 84% 56%, rgba(255, 166, 45, 0.2) 0%, rgba(255, 166, 45, 0) 100%);
      opacity: 0;
      z-index: 0;
      pointer-events: none;
      mix-blend-mode: screen;
      transition: opacity 420ms var(--ease), transform 420ms var(--ease);
    }

    body.route-overview .car-bg {
      right: -6px;
      top: 74px;
      width: min(1460px, 108vw);
      opacity: calc((0.06 + (var(--hero-glow) * 0.9)) * var(--hero-fade));
      filter:
        drop-shadow(0 0 calc(26px + (var(--hero-glow) * 90px)) rgba(255, 154, 24, calc(0.14 + (var(--hero-glow) * 0.56))))
        drop-shadow(0 0 40px rgba(0, 0, 0, 0.6));
    }

    body.route-overview .car-bg::before {
      opacity: calc((0.02 + (var(--hero-glow) * 0.34)) * var(--hero-fade));
      animation: speedLines 2100ms linear infinite;
    }

    body.route-overview .car-bg::after {
      opacity: calc((0.03 + (var(--hero-glow) * 0.5)) * var(--hero-fade));
      transform: scale(calc(0.92 + (var(--hero-glow) * 0.1)));
    }

    .car-bg .motion-layer {
      position: absolute;
      inset: 18% -12% 16% 28%;
      pointer-events: none;
      z-index: 1;
      opacity: 0;
      background:
        linear-gradient(96deg, rgba(255, 179, 74, 0.5) 0%, rgba(255, 179, 74, 0) 36%),
        repeating-linear-gradient(96deg, rgba(255, 154, 24, 0.14) 0 2px, transparent 2px 18px);
      mix-blend-mode: screen;
      transform: translateX(0);
    }

    body.route-overview .car-bg .motion-layer {
      opacity: calc((0.03 + (var(--hero-glow) * 0.35)) * var(--hero-fade));
      animation: motionSweep 3200ms linear infinite;
    }

    .car-bg .car-photo {
      width: 100%;
      height: auto;
      display: block;
      transform-origin: center center;
      opacity: 0.2;
      position: relative;
      z-index: 2;
      filter: saturate(1.12) contrast(1.1) brightness(0.92);
      transition: opacity 420ms var(--ease), transform 420ms var(--ease), filter 420ms var(--ease);
    }

    body.route-overview .car-bg .car-photo {
      opacity: calc((0.16 + (var(--hero-glow) * 0.84)) * var(--hero-fade));
      transform:
        translateX(calc(var(--hero-progress) * -120px))
        scale(calc(0.92 + (var(--hero-glow) * 0.22)));
      filter:
        blur(calc(var(--hero-progress) * 5px))
        saturate(calc(1.06 + (var(--hero-glow) * 0.24)))
        contrast(calc(1.04 + (var(--hero-glow) * 0.18)))
        brightness(calc(0.9 + (var(--hero-glow) * 0.18)));
    }

    body.route-overview.loaded.pop-phase .car-bg .car-photo {
      animation: carPopToBackground 1500ms var(--ease) forwards;
    }

    body.route-overview.loaded:not(.pop-phase) .car-bg .car-photo {
      animation: silhouetteSettle 900ms var(--ease) forwards, silhouetteIdle 9s 900ms ease-in-out infinite;
    }

    body.route-overview.loaded.pop-phase .car-bg::after {
      animation: backgroundFlash 1300ms var(--ease) both;
    }

    .site {
      max-width: 1320px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
      display: grid;
      gap: 30px;
    }

    main {
      display: grid;
      gap: 30px;
    }

    .hero {
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
      overflow: visible;
      position: relative;
      animation: rise 560ms var(--ease) both;
      min-height: calc(100vh - 74px);
      display: grid;
      align-items: center;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 20%, rgba(255, 138, 0, 0.08) 50%, transparent 80%);
      transform: translateX(-50%);
      animation: sweep 5.8s ease-in-out infinite;
      pointer-events: none;
    }

    .hero::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size: 28px 28px;
      opacity: 0.12;
      pointer-events: none;
    }

    .hero-inner {
      display: grid;
      grid-template-columns: 1fr;
      gap: 36px;
      padding: clamp(52px, 8vw, 96px);
      align-items: center;
      align-content: center;
    }

    .hero-rail {
      display: grid;
      gap: 4px;
      margin-bottom: 32px;
      transform: translateY(-24px);
    }

    .brand {
      margin: 0;
      font-size: clamp(30px, 4.8vw, 56px);
      letter-spacing: 0.11em;
      text-transform: lowercase;
      color: #f7f7f7;
      font-weight: 900;
      font-family: var(--mono);
      line-height: 1;
    }

    .brand-sub {
      margin: 0;
      font-size: clamp(12px, 1.5vw, 18px);
      letter-spacing: 0.08em;
      text-transform: lowercase;
      color: #a0a0a0;
      font-family: var(--mono);
      line-height: 1.15;
    }

    .hero-note {
      margin: 0;
      font-size: clamp(20px, 2.4vw, 34px);
      color: #ffd8a3;
      max-width: 980px;
      text-align: left;
      letter-spacing: -0.01em;
      line-height: 1.18;
      font-family: "Sora", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    .title {
      margin: 0 0 20px 0;
      font-size: clamp(38px, 5.4vw, 72px);
      line-height: 1.02;
      font-weight: 700;
      max-width: 840px;
      letter-spacing: -0.03em;
      text-wrap: balance;
      text-transform: capitalize;
    }

    .sub {
      margin: 18px 0 0 0;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
      max-width: 860px;
    }

    .actions {
      display: grid;
      gap: 14px;
      min-width: 260px;
    }

    .btn {
      border: 1px solid var(--line);
      background: rgba(18, 18, 18, 0.92);
      color: #f2f2f2;
      border-radius: 14px;
      font-size: 14px;
      padding: 14px 18px;
      cursor: pointer;
      transition: transform 180ms var(--ease), border-color 180ms var(--ease), background-color 180ms var(--ease);
      position: relative;
      overflow: hidden;
      font-family: var(--mono);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .btn:hover {
      border-color: var(--line-strong);
      transform: translateY(-1px);
      background: rgba(24, 24, 24, 0.96);
    }

    .btn.primary {
      border-color: rgba(255, 138, 0, 0.55);
      background: linear-gradient(120deg, rgba(255, 138, 0, 0.24), rgba(255, 138, 0, 0.06));
      color: #fff3e4;
    }

    .btn.warn {
      border-color: rgba(244, 193, 111, 0.4);
      color: #f7d8a2;
    }

    .ripple {
      position: absolute;
      border-radius: 50%;
      transform: scale(0);
      background: rgba(255, 255, 255, 0.35);
      pointer-events: none;
      animation: ripple 620ms cubic-bezier(.2,.7,.2,1) forwards;
    }

    .tabs-wrap {
      padding: 0 36px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .tabs {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px;
      background: rgba(8, 8, 8, 0.86);
    }

    .tab {
      border: 1px solid transparent;
      background: transparent;
      color: #bcbcbc;
      border-radius: 999px;
      font-size: 14px;
      padding: 11px 18px;
      cursor: pointer;
      transition: all 180ms var(--ease);
      position: relative;
      overflow: hidden;
    }

    .tab::after {
      content: "";
      position: absolute;
      left: 12px;
      right: 12px;
      bottom: 4px;
      height: 1px;
      background: rgba(255, 138, 0,0.75);
      transform: scaleX(0);
      transform-origin: left;
      transition: transform 240ms var(--ease);
    }

    .tab:hover {
      color: #f2f2f2;
      border-color: rgba(255, 255, 255, 0.18);
    }

    .tab.active {
      border-color: rgba(255, 138, 0, 0.45);
      background: rgba(255, 138, 0, 0.16);
      color: #fff3df;
    }

    .tab.active::after {
      transform: scaleX(1);
    }

    .route-hint {
      margin: 0;
      font-size: 14px;
      color: var(--muted);
      font-family: var(--mono);
      letter-spacing: 0.03em;
    }

    .page {
      display: none;
      opacity: 0;
      transform: translateY(18px);
    }

    .page.active {
      display: block;
      animation: rise 320ms var(--ease) forwards;
    }

    .reveal-on-scroll {
      opacity: 0;
      transform: translateY(52px) scale(0.985);
      filter: blur(8px);
      transition:
        transform 920ms var(--ease) var(--reveal-delay, 0ms),
        opacity 920ms var(--ease) var(--reveal-delay, 0ms),
        filter 920ms var(--ease) var(--reveal-delay, 0ms);
      will-change: transform, opacity, filter;
    }

    .reveal-on-scroll.in-view {
      opacity: 1;
      transform: translateY(0) scale(1);
      filter: blur(0);
    }

    .landing {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 8px 0 2px;
      margin-bottom: 16px;
      position: relative;
      overflow: visible;
      transform-origin: center;
      animation: none;
    }

    .landing::after {
      content: none;
    }

    .landing h2 {
      margin: 0 0 10px 0;
      font-size: clamp(32px, 5vw, 58px);
      letter-spacing: -0.02em;
      max-width: 760px;
      text-wrap: balance;
    }

    .landing p {
      margin: 0 0 16px 0;
      max-width: 760px;
      color: #b9b9b9;
      line-height: 1.72;
      font-size: 17px;
    }

    .end-cta {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 18px 0 0;
      margin-top: 20px;
      display: grid;
      gap: 14px;
      justify-items: start;
    }

    .end-cta h3 {
      margin: 0;
      font-size: clamp(28px, 3.5vw, 42px);
      letter-spacing: -0.02em;
      line-height: 1.1;
    }

    .end-cta p {
      margin: 0;
      max-width: 760px;
      color: #c4c4c4;
      line-height: 1.7;
      font-size: 16px;
    }

    .strategy-clean {
      display: grid;
      gap: 18px;
    }

    .strategy-head {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(10, 10, 10, 0.9);
      padding: 24px;
      animation: panelIn 520ms var(--ease) both;
    }

    .strategy-head h2 {
      margin: 0 0 8px 0;
      font-size: clamp(26px, 3.2vw, 40px);
      letter-spacing: -0.02em;
      line-height: 1.1;
    }

    .strategy-head p {
      margin: 0;
      color: #bfbfbf;
      font-size: 15px;
      line-height: 1.65;
      max-width: 760px;
    }

    .model-story {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 8px 0 2px;
      margin-top: 16px;
      position: relative;
      overflow: visible;
    }

    .model-story::before {
      content: none;
    }

    .model-story h3 {
      margin: 0 0 10px 0;
      font-size: clamp(28px, 3.8vw, 42px);
      line-height: 1.1;
      letter-spacing: -0.02em;
      max-width: 780px;
    }

    .model-story p {
      margin: 0 0 18px 0;
      max-width: 860px;
      color: #bcbcbc;
      font-size: 17px;
      line-height: 1.72;
    }

    .pop-flow {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 8px 0 2px;
      margin-top: 16px;
      overflow: visible;
      position: relative;
    }

    .pop-flow h3 {
      margin: 0 0 12px 0;
      font-size: clamp(26px, 3.6vw, 40px);
      letter-spacing: -0.02em;
      line-height: 1.1;
      max-width: 850px;
    }

    .pop-flow p {
      margin: 0 0 16px 0;
      max-width: 900px;
      font-size: 16px;
      line-height: 1.7;
      color: #bdbdbd;
    }

    .pop-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .pop-item {
      border: 0;
      border-left: 2px solid rgba(255, 138, 0, 0.34);
      border-radius: 0;
      background: transparent;
      padding: 14px 0 14px 16px;
      opacity: 0;
      transform: translateY(20px) scale(0.98);
      transition: border-color 220ms var(--ease), transform 220ms var(--ease), opacity 220ms var(--ease);
    }

    .reveal-on-scroll.in-view .pop-item {
      animation: popIn 620ms calc(var(--i, 0) * 140ms) var(--ease) forwards;
    }

    .pop-item:hover {
      transform: translateX(6px);
      border-color: rgba(255, 138, 0, 0.64);
    }

    .pop-item h4 {
      margin: 0 0 8px 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: #ffd08a;
      font-family: var(--mono);
    }

    .pop-item p {
      margin: 0;
      font-size: 16px;
      color: #d0d0d0;
      line-height: 1.6;
    }

    .story-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .story-card {
      border: 0;
      border-left: 2px solid rgba(255, 138, 0, 0.3);
      border-radius: 0;
      background: transparent;
      padding: 14px 0 14px 16px;
      min-height: 150px;
      transition: transform 260ms var(--ease), border-color 260ms var(--ease);
      animation: rise 600ms var(--delay, 0ms) var(--ease) both;
    }

    .story-card:hover {
      transform: translateX(6px);
      border-color: rgba(255, 138, 0, 0.6);
    }

    .story-card:nth-child(1) { --delay: 140ms; }
    .story-card:nth-child(2) { --delay: 220ms; }
    .story-card:nth-child(3) { --delay: 300ms; }

    .story-card h4 {
      margin: 0 0 7px 0;
      font-size: 14px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: #ffcc8d;
    }

    .story-card p {
      margin: 0;
      font-size: 15px;
      color: #d0d0d0;
      line-height: 1.62;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1.05fr 1.95fr;
      gap: 20px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(10, 10, 10, 0.9);
      padding: 24px;
      position: relative;
      overflow: hidden;
      transition: border-color 220ms var(--ease), transform 220ms var(--ease);
      animation: panelIn 520ms var(--ease) both;
    }

    .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 30%, rgba(255, 138, 0,0.06) 50%, transparent 70%);
      transform: translateX(-80%);
      animation: sheenPanel 9s ease-in-out infinite;
      pointer-events: none;
    }

    .panel:hover {
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-2px);
    }

    .panel h3 {
      margin: 0 0 12px 0;
      font-size: 22px;
      letter-spacing: 0.02em;
    }

    .status-good { color: var(--good); font-weight: 600; }
    .status-warn { color: var(--warn); font-weight: 600; }
    .status-bad { color: var(--bad); font-weight: 600; }

    .bar-row {
      display: grid;
      grid-template-columns: 210px 1fr 82px;
      gap: 14px;
      align-items: center;
      margin: 14px 0;
      font-size: 14px;
      opacity: 0;
      transform: translateX(-6px);
    }

    .bar-row.reveal { animation: slide 360ms var(--delay, 0ms) var(--ease) forwards; }

    .bar {
      height: 12px;
      border-radius: 999px;
      background: #1b1b1b;
      overflow: hidden;
    }

    .bar span {
      display: block;
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #c26a00 0%, #ff8a00 100%);
      transition: width 860ms var(--ease);
      box-shadow: 0 0 12px rgba(255, 138, 0,0.35);
    }

    .spark {
      width: 100%;
      height: 124px;
      border: 1px solid #202020;
      border-radius: 10px;
      background: #0a0a0a;
      margin-top: 12px;
      display: block;
    }

    .controls {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(10, 10, 10, 0.92);
      padding: 24px;
      margin-bottom: 18px;
      display: grid;
      gap: 14px;
      animation: panelIn 520ms var(--ease) both;
    }

    .row {
      display: grid;
      grid-template-columns: 1.5fr 0.7fr 1fr auto;
      gap: 16px;
      align-items: center;
    }

    .field { display: grid; gap: 6px; }

    .field label {
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    input[type='search'], select, input[type='range'] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #111;
      color: var(--ink);
      font-size: 15px;
      padding: 12px 13px;
      outline: none;
    }

    input[type='search']:focus, select:focus {
      border-color: rgba(255, 138, 0,0.6);
      box-shadow: 0 0 0 3px rgba(255, 138, 0,0.2);
    }

    .table-wrap {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(9, 9, 9, 0.92);
      padding: 22px;
      scroll-margin-top: 34px;
      animation: panelIn 520ms var(--ease) both;
    }

    .table-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }

    .hint { margin: 0; font-size: 14px; color: var(--muted); }
    .hint:empty { display: none; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 15px;
    }

    th, td {
      text-align: left;
      padding: 16px 14px;
      border-bottom: 1px solid #1d1d1d;
      vertical-align: top;
    }

    .table-header strong {
      font-size: 18px;
      letter-spacing: 0.01em;
    }

    thead th { background: #101010; color: #e8e8e8; }

    .th-btn {
      border: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      cursor: pointer;
      padding: 0;
      position: relative;
      overflow: hidden;
    }

    tbody tr {
      opacity: 0;
      transform: translateY(9px);
      cursor: pointer;
      transition: background-color 160ms ease;
    }

    tbody tr.reveal { animation: rise 360ms var(--delay, 0ms) var(--ease) forwards; }
    tbody tr:hover { background: rgba(255, 138, 0,0.08); }
    tbody tr:active { background: rgba(255, 138, 0,0.14); }

    .race-main {
      display: block;
      font-weight: 600;
      color: #f2f2f2;
    }

    .race-location {
      display: block;
      margin-top: 4px;
      color: #9f9f9f;
      font-size: 12px;
      font-family: var(--mono);
      letter-spacing: 0.02em;
    }

    .race-when {
      display: block;
      margin-top: 3px;
      color: #7f7f7f;
      font-size: 11px;
      font-family: var(--mono);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }

    .mono { font-family: "SFMono-Regular", Menlo, monospace; color: #bdbdbd; font-size: 12px; }

    .sim-grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
    }

    .sim-row { display: grid; gap: 8px; margin: 10px 0; }

    .sim-metric {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px dashed #242424;
      font-size: 13px;
      animation: rise 520ms var(--delay, 0ms) var(--ease) both;
    }

    .sim-metric strong { font-size: 17px; }

    .integrity-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .list { margin: 0; padding-left: 18px; color: #dddddd; font-size: 13px; }

    .hash-list { max-height: 330px; overflow: auto; display: grid; gap: 8px; }

    .hash-item {
      border: 1px solid #242424;
      border-radius: var(--radius-sm);
      padding: 8px;
      background: #0d0d0d;
      display: grid;
      gap: 7px;
      animation: rise 500ms var(--delay, 0ms) var(--ease) both;
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
      color: #cbcbcb;
    }

    .raw {
      width: 100%;
      max-height: 340px;
      overflow: auto;
      margin-top: 8px;
      border: 1px solid #252525;
      border-radius: 10px;
      background: #0a0a0a;
      color: #d0d0d0;
      font-size: 11px;
      padding: 10px;
      display: none;
    }

    .raw.show { display: block; }

    .drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.58);
      opacity: 0;
      pointer-events: none;
      transition: opacity 220ms var(--ease);
      z-index: 18;
      backdrop-filter: blur(2px);
    }

    .drawer-backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer {
      position: fixed;
      top: 50%;
      left: 50%;
      width: min(1280px, 96vw);
      height: min(90vh, 1020px);
      background: rgba(8, 8, 8, 0.98);
      border: 1px solid #242424;
      border-radius: 18px;
      transform: translate(-50%, -46%) scale(0.96);
      opacity: 0;
      pointer-events: none;
      transition: transform 300ms var(--ease), opacity 300ms var(--ease);
      z-index: 20;
      padding: 28px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.58);
    }

    .drawer.open {
      transform: translate(-50%, -50%) scale(1);
      opacity: 1;
      pointer-events: auto;
    }

    .drawer.bump {
      animation: drawerPulse 260ms var(--ease);
    }

    .drawer.open .tag {
      animation: drawerItemIn 280ms var(--ease) both;
    }

    .drawer.open .tag:nth-child(2) { animation-delay: 50ms; }
    .drawer.open .tag:nth-child(3) { animation-delay: 90ms; }
    .drawer.open .tag:nth-child(4) { animation-delay: 130ms; }

    .drawer-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .drawer-body {
      overflow: auto;
      font-size: 15px;
      color: #e4e4e4;
      display: grid;
      gap: 14px;
      padding-right: 4px;
    }

    .drawer-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .kv {
      border: 1px solid #222;
      border-radius: var(--radius-sm);
      background: #111;
      padding: 12px;
    }

    .drawer.open .kv {
      animation: drawerItemIn 320ms var(--ease) both;
    }

    .drawer.open .kv:nth-child(2) { animation-delay: 40ms; }
    .drawer.open .kv:nth-child(3) { animation-delay: 70ms; }
    .drawer.open .kv:nth-child(4) { animation-delay: 100ms; }
    .drawer.open .kv:nth-child(5) { animation-delay: 130ms; }
    .drawer.open .kv:nth-child(6) { animation-delay: 160ms; }
    .drawer.open .kv:nth-child(7) { animation-delay: 190ms; }
    .drawer.open .kv:nth-child(8) { animation-delay: 220ms; }
    .drawer.open .kv:nth-child(9) { animation-delay: 250ms; }
    .drawer.open .kv:nth-child(10) { animation-delay: 280ms; }
    .drawer.open .kv:nth-child(11) { animation-delay: 310ms; }

    .kv p { margin: 0; }

    .kv .k {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .kv .v {
      color: #f2f2f2;
      font-size: 16px;
      font-weight: 600;
      line-height: 1.3;
    }

    .tag {
      display: inline-block;
      border: 1px solid #282828;
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 12px;
      color: #d8d8d8;
      margin-right: 6px;
      margin-bottom: 6px;
      font-family: var(--mono);
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slide {
      from { opacity: 0; transform: translateX(-7px); }
      to { opacity: 1; transform: translateX(0); }
    }

    @keyframes sweep {
      0%, 15% { opacity: 0; transform: translateX(-55%); }
      35% { opacity: 1; }
      70%, 100% { opacity: 0; transform: translateX(55%); }
    }

    @keyframes ripple {
      0% { transform: scale(0); opacity: 0.4; }
      100% { transform: scale(4.4); opacity: 0; }
    }

    @keyframes breathe {
      0%, 100% { transform: scale(0.95); opacity: 0.45; }
      50% { transform: scale(1.08); opacity: 0.7; }
    }

    @keyframes drift {
      0%, 100% { transform: translateX(0); }
      50% { transform: translateX(-18px); }
    }

    @keyframes carPopToBackground {
      0% { transform: scale(0.84) translateX(42px) translateY(8px); }
      36% { transform: scale(1.36) translateX(-24px) translateY(-4px); }
      100% { transform: scale(1.1) translateX(-10px) translateY(0); }
    }

    @keyframes silhouetteSettle {
      0% { transform: scale(1.1) translateX(-10px) translateY(0); }
      100% { transform: scale(1.12) translateX(0) translateY(0); }
    }

    @keyframes silhouetteIdle {
      0%, 100% { transform: scale(1.12) translateX(0) translateY(0); }
      50% { transform: scale(1.14) translateX(-22px) translateY(-4px); }
    }

    @keyframes popIn {
      from { opacity: 0; transform: translateY(22px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes panelIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes sheenPanel {
      0%, 15% { opacity: 0; transform: translateX(-80%); }
      30% { opacity: 1; }
      60%, 100% { opacity: 0; transform: translateX(80%); }
    }

    @keyframes floatSoft {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-3px); }
    }

    @keyframes speedLines {
      from { transform: translateX(0); }
      to { transform: translateX(-72px); }
    }

    @keyframes motionSweep {
      from { transform: translateX(0); }
      to { transform: translateX(-92px); }
    }

    @keyframes trailShift {
      from { stroke-dashoffset: 0; }
      to { stroke-dashoffset: -760; }
    }

    @keyframes wheelSpin {
      from { transform: rotate(0deg); }
      to { transform: rotate(-360deg); }
    }

    @keyframes bodyPulse {
      0%, 100% { opacity: 0.45; }
      50% { opacity: 0.86; }
    }

    @keyframes backgroundFlash {
      0% { opacity: 0.08; transform: scale(0.9); }
      40% { opacity: 0.8; transform: scale(1.03); }
      100% { opacity: calc((0.03 + (var(--hero-glow) * 0.5)) * var(--hero-fade)); transform: scale(calc(0.92 + (var(--hero-glow) * 0.1))); }
    }

    @keyframes roadMove {
      from { background-position-x: 0; }
      to { background-position-x: 54px; }
    }

    @keyframes driveOff {
      0% { transform: translateX(0) scale(1); }
      72% { transform: translateX(-640px) scale(1); }
      100% { transform: translateX(-860px) scale(0.92); opacity: 0.72; }
    }

    @keyframes drawerPulse {
      0% { transform: translate(-50%, -49%) scale(0.985); }
      100% { transform: translate(-50%, -50%) scale(1); }
    }

    @keyframes drawerItemIn {
      0% { opacity: 0; transform: translateY(8px) scale(0.99); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    @media (max-width: 980px) {
      .hero {
        min-height: 84vh;
      }

      .hero-inner {
        grid-template-columns: 1fr;
      }

      .actions {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .grid-2, .row {
        grid-template-columns: 1fr;
      }

      .tabs-wrap {
        flex-direction: column;
        align-items: stretch;
      }

      .tabs {
        width: 100%;
        justify-content: space-between;
      }

      .tab {
        flex: 1;
        text-align: center;
      }

      .bar-row {
        grid-template-columns: 140px 1fr 58px;
      }

      .story-grid {
        grid-template-columns: 1fr;
      }

      .pop-grid {
        grid-template-columns: 1fr;
      }

      .hero-note {
        text-align: left;
        max-width: none;
      }

      .hero-rail {
        flex-direction: column;
        gap: 6px;
      }
    }

    @media (max-width: 760px) {
      body {
        padding: 16px;
      }

      .hero {
        min-height: 76vh;
      }

      .landing,
      .hero-inner,
      .tabs-wrap {
        padding-left: 22px;
        padding-right: 22px;
      }

      .actions {
        grid-template-columns: 1fr;
      }

      .car-bg {
        top: 72px;
        right: -210px;
        width: 1040px;
        opacity: 0.18;
      }

      body.route-overview .car-bg {
        right: -76px;
        top: 88px;
        width: 1180px;
        opacity: calc((0.06 + (var(--hero-glow) * 0.84)) * var(--hero-fade));
      }

      .drawer-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
      }

      .hero, .page, .bar-row, tbody tr {
        opacity: 1 !important;
        transform: none !important;
      }

      .car-bg,
      .car-bg .car-photo,
      .car-bg .motion-layer {
        transform: none !important;
        animation: none !important;
      }
    }
  </style>
</head>
<body class="route-overview">
  <div class="loading-screen" id="loadingScreen">
    <div class="loading-scene">
      <div class="loading-track" aria-hidden="true">
        <svg class="loading-car" viewBox="0 0 320 96" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 63L22 58L40 55H56L74 53L92 43H126L146 37H174L188 30H230L248 40H262L280 47L296 50L306 58L314 66H294L286 60H266L258 74H224L216 60H178L170 74H136L128 62H100L90 74H64L56 66H44L34 72H16L12 63Z" fill="#08090c"/>
          <path d="M12 63L22 58L40 55H56L74 53L92 43H126L146 37H174L188 30H230L248 40H262L280 47L296 50L306 58L314 66H294L286 60H266L258 74H224L216 60H178L170 74H136L128 62H100L90 74H64L56 66H44L34 72H16L12 63Z" stroke="#e9e9e9" stroke-opacity="0.48"/>
          <path d="M270 36H290V60H262L270 36Z" fill="#0a0b10"/>
          <path d="M4 68H24L20 74H2L4 68Z" fill="#0a0b10"/>
          <path d="M102 58C130 56 156 51 184 44C198 41 214 36 228 33" stroke="#ffffff" stroke-opacity="0.28" stroke-width="1.8" stroke-linecap="round"/>
          <circle cx="90" cy="76" r="12" fill="#0a0b10"/>
          <circle cx="228" cy="76" r="12" fill="#0a0b10"/>
          <circle cx="90" cy="76" r="4.2" fill="#f2f2f2" fill-opacity="0.72"/>
          <circle cx="228" cy="76" r="4.2" fill="#f2f2f2" fill-opacity="0.72"/>
        </svg>
      </div>
      <p class="loading-text">Preparing race strategy interface</p>
    </div>
  </div>

  <div class="car-bg" aria-hidden="true">
    <div class="motion-layer"></div>
    <img
      class="car-photo"
      src="https://p.turbosquid.com/ts-thumb/Dr/06s7Jw/8G/mcl39_0000/jpg/1741934846/1920x1080/fit_q87/fca87e4f2ec80d23c7eb802adccfb4546ee348b4/mcl39_0000.jpg"
      alt="McLaren MCL39 Formula 1 car render"
    />
  </div>

  <div class="site">
    <header class="hero">
      <div class="hero-inner">
        <div>
          <div class="hero-rail">
            <p class="brand">dnf</p>
            <p class="brand-sub">do not f**kup</p>
          </div>
          <h1 class="title">A Race strategy prediction model</h1>
          <p class="hero-note">Strategy prediction model for McLaren F1 for the 2025 Formula 1 season</p>
          <p class="sub" id="metaLine">Loading race strategies...</p>
        </div>
      </div>
    </header>

    <main>
      <section class="page active" id="page-overview">
        <section class="landing reveal-on-scroll" id="introBlock">
          <h2>Build race-day decisions from practice, qualifying and contingency signals.</h2>
          <p>
            The model uses the data from pre-season testing, practice session, qualifying session, weather forecast
            and other parameters and outputs strategy actions: which tyre to start on, when to pit, and what backup
            plans to activate if weather shifts, reliability drops, or race incidents occur.
          </p>
        </section>
        <section class="pop-flow reveal-on-scroll" id="scrollSignals">
          <h3>How we use data to predict</h3>
          <div class="pop-grid">
            <article class="pop-item" style="--i:0">
              <h4>Baseline Pace</h4>
              <p>Pre-season and long-run testing anchor the opening race pace estimate.</p>
            </article>
            <article class="pop-item" style="--i:1">
              <h4>Practice Drift</h4>
              <p>Fuel proxies, traffic and track evolution refine tyre life and stint windows.</p>
            </article>
            <article class="pop-item" style="--i:2">
              <h4>Qualifying Context</h4>
              <p>Grid position and one-lap delta set start compound risk and first pit timing.</p>
            </article>
            <article class="pop-item" style="--i:3">
              <h4>Weather Pressure</h4>
              <p>Forecast shifts activate alternate compounds and shorter or longer opening stints.</p>
            </article>
          </div>
        </section>

        <section class="model-story reveal-on-scroll" id="modelStory">
          <h3>What this strategy model does</h3>
          <p>
            The model learns from pre-season testing, practice sessions, and qualifying sessions to estimate race pace,
            tyre behavior, pit timing, and strategy robustness before lights out.
          </p>
          <div class="story-grid">
            <article class="story-card">
              <h4>Testing</h4>
              <p>Builds baseline pace and degradation priors from long-run and setup behavior.</p>
            </article>
            <article class="story-card">
              <h4>Practice</h4>
              <p>Uses session evolution, traffic, weather and fuel proxies to update tyre and race assumptions.</p>
            </article>
            <article class="story-card">
              <h4>Qualifying</h4>
              <p>Integrates one-lap performance into final race strategy ranking and fallback plans.</p>
            </article>
          </div>
        </section>

        <section class="pop-flow reveal-on-scroll" id="contingencyStory">
          <h3>Fallback strategy logic for race-day disruptions</h3>
          <p>
            Every round includes a primary plan plus two backup plans so you can react without rebuilding strategy mid-race.
          </p>
          <div class="pop-grid">
            <article class="pop-item" style="--i:0">
              <h4>Pit Guidance</h4>
              <p>Plans remain lap-specific: start tyre, first stop lap and full compound chain.</p>
            </article>
            <article class="pop-item" style="--i:1">
              <h4>Fallback 2</h4>
              <p>Used for weather shifts, safety car windows or early tyre thermal drop.</p>
            </article>
            <article class="pop-item" style="--i:2">
              <h4>Fallback 3</h4>
              <p>Used for reliability risk, driver recovery laps or aggressive undercut defense.</p>
            </article>
            <article class="pop-item" style="--i:3">
              <h4>Trigger Rules</h4>
              <p>Each fallback includes explicit trigger text so operations can switch fast.</p>
            </article>
          </div>
        </section>

        <section class="end-cta reveal-on-scroll" id="endCta">
          <h3>Ready to explore strategy options?</h3>
          <p>Move to a cleaner strategy dashboard focused on race plans, pit windows and fallback choices.</p>
          <button class="btn primary" id="jumpStrategyEnd">Explore strategy</button>
        </section>
      </section>

      <section class="page" id="page-strategy">
        <section class="strategy-clean">
          <section class="strategy-head reveal-on-scroll" id="strategyHead">
            <h2>Clean strategy dashboard</h2>
            <p>Filter events, compare primary plans and open each row for full fallback strategy details.</p>
          </section>

          <section class="panel reveal-on-scroll" id="topRoundsPanel">
            <h3>Top Rounds by Win Probability</h3>
            <div id="topRounds"></div>
          </section>

          <section class="controls reveal-on-scroll" id="strategyControls">
            <div class="row">
              <div class="field">
                <label for="searchInput">Search</label>
                <input id="searchInput" type="search" placeholder="Race, strategy, fallback trigger..." />
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
            <p class="hint" id="filterState"></p>
          </section>

          <section class="table-wrap reveal-on-scroll" id="strategyTableWrap">
            <div class="table-header">
              <strong>Race Strategy Table</strong>
            </div>
            <div style="overflow:auto;">
              <table>
                <thead>
                  <tr>
                    <th><button class="th-btn" data-sort="event_name">Race</button></th>
                    <th><button class="th-btn" data-sort="strategy_plan">Primary Plan</button></th>
                    <th><button class="th-btn" data-sort="stops">Stops</button></th>
                    <th>Start Tyre</th>
                    <th>First Pit</th>
                    <th><button class="th-btn" data-sort="win_probability">Win Prob</button></th>
                  </tr>
                </thead>
                <tbody id="raceRows"></tbody>
              </table>
            </div>
          </section>
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
    const routes = ['overview', 'strategy'];
    const state = {
      payload: null,
      sortKey: 'event_name',
      sortDir: 'asc',
      filteredRows: []
    };
    let scrollObserver = null;
    let popPhaseTimer = null;

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    const fmt = (v, n=3) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(n);
    const pct = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : `${(Number(v) * 100).toFixed(1)}%`;
    const supportsViewTransition = typeof document.startViewTransition === 'function';

    function safeText(value) {
      return (value === null || value === undefined) ? '-' : String(value);
    }

    const ROUND_META_2025 = {
      1: { round: 1, name: 'Australian Grand Prix', location: 'Melbourne, Australia', when: 'Mar 16, 2025' },
      2: { round: 2, name: 'Chinese Grand Prix', location: 'Shanghai, China', when: 'Mar 23, 2025' },
      3: { round: 3, name: 'Japanese Grand Prix', location: 'Suzuka, Japan', when: 'Apr 6, 2025' },
      4: { round: 4, name: 'Bahrain Grand Prix', location: 'Sakhir, Bahrain', when: 'Apr 13, 2025' },
      5: { round: 5, name: 'Saudi Arabian Grand Prix', location: 'Jeddah, Saudi Arabia', when: 'Apr 20, 2025' },
      6: { round: 6, name: 'Miami Grand Prix', location: 'Miami, USA', when: 'May 4, 2025' },
      7: { round: 7, name: 'Emilia Romagna Grand Prix', location: 'Imola, Italy', when: 'May 18, 2025' },
      8: { round: 8, name: 'Monaco Grand Prix', location: 'Monte Carlo, Monaco', when: 'May 25, 2025' },
      9: { round: 9, name: 'Spanish Grand Prix', location: 'Barcelona, Spain', when: 'Jun 1, 2025' },
      10: { round: 10, name: 'Canadian Grand Prix', location: 'Montreal, Canada', when: 'Jun 15, 2025' },
      11: { round: 11, name: 'Austrian Grand Prix', location: 'Spielberg, Austria', when: 'Jun 29, 2025' },
      12: { round: 12, name: 'British Grand Prix', location: 'Silverstone, United Kingdom', when: 'Jul 6, 2025' },
      13: { round: 13, name: 'Belgian Grand Prix', location: 'Spa-Francorchamps, Belgium', when: 'Jul 27, 2025' },
      14: { round: 14, name: 'Hungarian Grand Prix', location: 'Budapest, Hungary', when: 'Aug 3, 2025' },
      15: { round: 15, name: 'Dutch Grand Prix', location: 'Zandvoort, Netherlands', when: 'Aug 31, 2025' },
      16: { round: 16, name: 'Italian Grand Prix', location: 'Monza, Italy', when: 'Sep 7, 2025' },
      17: { round: 17, name: 'Azerbaijan Grand Prix', location: 'Baku, Azerbaijan', when: 'Sep 21, 2025' },
      18: { round: 18, name: 'Singapore Grand Prix', location: 'Singapore', when: 'Oct 5, 2025' },
      19: { round: 19, name: 'United States Grand Prix', location: 'Austin, USA', when: 'Oct 19, 2025' },
      20: { round: 20, name: 'Mexico City Grand Prix', location: 'Mexico City, Mexico', when: 'Oct 26, 2025' },
      21: { round: 21, name: 'Sao Paulo Grand Prix', location: 'Sao Paulo, Brazil', when: 'Nov 9, 2025' },
      22: { round: 22, name: 'Las Vegas Grand Prix', location: 'Las Vegas, USA', when: 'Nov 22, 2025' },
      23: { round: 23, name: 'Qatar Grand Prix', location: 'Lusail, Qatar', when: 'Nov 30, 2025' },
      24: { round: 24, name: 'Abu Dhabi Grand Prix', location: 'Yas Marina, UAE', when: 'Dec 7, 2025' }
    };

    function resolveRaceMeta(row) {
      const rawName = safeText(row.event_name);
      const match = rawName.match(/round[_\\s-]?(\\d{1,2})/i);
      if (match) {
        const round = Number(match[1]);
        const meta = ROUND_META_2025[round];
        if (meta) return meta;
      }
      for (const meta of Object.values(ROUND_META_2025)) {
        if (meta.name.toLowerCase() === rawName.toLowerCase()) return meta;
      }
      return {
        round: Number.POSITIVE_INFINITY,
        name: rawName,
        location: safeText(row.location || row.circuit_location || row.country || '-'),
        when: safeText(row.event_date || row.date || row.race_date || row.weekend_date || '-')
      };
    }

    function routeFromHash() {
      const route = (window.location.hash || '').replace('#', '');
      if (route === 'races') return 'strategy';
      if (routes.includes(route)) return route;
      const byPath = window.location.pathname.replace('/', '');
      if (byPath === 'home') return 'overview';
      if (byPath === 'races') return 'strategy';
      if (routes.includes(byPath)) return byPath;
      return 'overview';
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
      document.body.classList.toggle('route-overview', route === 'overview');
      const car = document.querySelector('.car-bg');
      if (car && route !== 'overview') {
        car.style.setProperty('--parallax-x', '0px');
        car.style.setProperty('--parallax-y', '0px');
      }
      updateHeroSilhouetteGlow();
      animatePageElements(route);
      requestAnimationFrame(initScrollReveal);
    }

    function animatePageElements(route) {
      const page = document.getElementById(`page-${route}`);
      if (!page) return;
      const targets = page.querySelectorAll('.panel, .controls, .table-wrap, .story-card, .strategy-head');
      targets.forEach((el, i) => {
        el.style.animation = 'none';
        void el.offsetWidth;
        el.style.animation = `rise 460ms ${Math.min(i * 40, 360)}ms var(--ease) both`;
      });
    }

    function hideLoadingScreen() {
      const loader = document.getElementById('loadingScreen');
      if (!loader) return;
      loader.classList.add('hide');
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      document.body.classList.add('loaded');
      if (reduceMotion) return;

      document.body.classList.add('pop-phase');
      if (popPhaseTimer !== null) {
        window.clearTimeout(popPhaseTimer);
      }
      popPhaseTimer = window.setTimeout(() => {
        document.body.classList.remove('pop-phase');
        popPhaseTimer = null;
      }, 1500);
    }

    function initCarParallax() {
      const car = document.querySelector('.car-bg');
      if (!car) return;
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

      const reset = () => {
        car.style.setProperty('--parallax-x', '0px');
        car.style.setProperty('--parallax-y', '0px');
      };

      window.addEventListener('pointermove', (event) => {
        if (!document.body.classList.contains('route-overview')) {
          reset();
          return;
        }
        const nx = (event.clientX / window.innerWidth) - 0.5;
        const ny = (event.clientY / window.innerHeight) - 0.5;
        car.style.setProperty('--parallax-x', `${(nx * 26).toFixed(2)}px`);
        car.style.setProperty('--parallax-y', `${(ny * 16).toFixed(2)}px`);
      });

      window.addEventListener('blur', reset);
      document.addEventListener('mouseleave', reset);
    }

    function updateHeroSilhouetteGlow() {
      const car = document.querySelector('.car-bg');
      if (!car) return;
      const setCarState = (glow, progress) => {
        const safeGlow = clamp(glow, 0, 1);
        const safeProgress = clamp(progress, 0, 1);
        const fade = clamp(1 - (safeProgress * 1.08), 0.04, 1);
        car.style.setProperty('--hero-glow', safeGlow.toFixed(3));
        car.style.setProperty('--hero-progress', safeProgress.toFixed(3));
        car.style.setProperty('--hero-fade', fade.toFixed(3));
      };

      if (!document.body.classList.contains('route-overview')) {
        setCarState(0.16, 1);
        return;
      }
      const hero = document.querySelector('.hero');
      if (!hero) {
        setCarState(1, 0);
        return;
      }
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        setCarState(0.72, 0.24);
        return;
      }
      const rect = hero.getBoundingClientRect();
      const fadeDistance = Math.max(window.innerHeight * 0.95, rect.height * 0.82, 1);
      const progress = clamp(Math.max(0, -rect.top) / fadeDistance, 0, 1);
      const glow = 1 - (progress * 0.84);
      setCarState(glow, progress);
    }

    function initHeroSilhouetteScroll() {
      let raf = null;
      const schedule = () => {
        if (raf !== null) return;
        raf = requestAnimationFrame(() => {
          raf = null;
          updateHeroSilhouetteGlow();
        });
      };
      window.addEventListener('scroll', schedule, { passive: true });
      window.addEventListener('resize', schedule);
      schedule();
    }

    function initScrollReveal() {
      const activePage = document.querySelector('.page.active');
      if (!activePage) return;
      const nodes = Array.from(activePage.querySelectorAll('.reveal-on-scroll:not(.in-view)'));
      if (!nodes.length) return;

      if (!scrollObserver) {
        scrollObserver = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('in-view');
            scrollObserver.unobserve(entry.target);
          });
        }, { threshold: 0.2, rootMargin: '0px 0px -12% 0px' });
      }

      nodes.forEach((node, index) => {
        node.style.setProperty('--reveal-delay', `${Math.min(index * 120, 520)}ms`);
        scrollObserver.observe(node);
      });
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

    function renderOverview(payload) {
      const top = document.getElementById('topRounds');
      top.innerHTML = '';
      const topRows = payload.top_rounds || [];
      const maxWin = Math.max(...topRows.map(r => Number(r.win_probability || 0)), 0.01);
      topRows.forEach((r, i) => {
        const race = resolveRaceMeta(r);
        const val = Number(r.win_probability || 0);
        const width = Math.max(4, Math.round((val / maxWin) * 100));
        const row = document.createElement('div');
        row.className = 'bar-row reveal';
        row.style.setProperty('--delay', `${110 + (i * 65)}ms`);
        row.innerHTML = `
          <span>${safeText(race.name)}</span>
          <div class='bar'><span data-width='${width}%'></span></div>
          <span>${pct(val)}</span>
        `;
        top.appendChild(row);
        const fill = row.querySelector('span[data-width]');
        requestAnimationFrame(() => { fill.style.width = fill.dataset.width || '0%'; });
      });
    }

    function compare(a, b, key) {
      if (key === 'event_name') {
        const ameta = resolveRaceMeta(a);
        const bmeta = resolveRaceMeta(b);
        const around = Number(ameta.round ?? Number.POSITIVE_INFINITY);
        const bround = Number(bmeta.round ?? Number.POSITIVE_INFINITY);
        if (around !== bround) return around - bround;
        return ameta.name.localeCompare(bmeta.name);
      }
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
        const race = resolveRaceMeta(r);
        const blob = `${safeText(race.name)} ${safeText(race.location)} ${safeText(race.when)} ${safeText(r.best_strategy)} ${safeText(r.strategy_plan)} ${safeText(r.compounds)} ${safeText(r.fallback_2_plan)} ${safeText(r.fallback_3_plan)} ${safeText(r.fallback_2_trigger)} ${safeText(r.fallback_3_trigger)}`.toLowerCase();
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
        const race = resolveRaceMeta(r);
        const tr = document.createElement('tr');
        tr.className = 'reveal';
        tr.style.setProperty('--delay', `${Math.min(i * 12, 240)}ms`);
        tr.dataset.index = String(i);
        tr.innerHTML = `
          <td><span class='race-main'>${safeText(race.name)}</span><span class='race-location'>${safeText(race.location)}</span><span class='race-when'>${safeText(race.when)}</span></td>
          <td>${safeText(r.strategy_plan || r.best_strategy)}</td>
          <td>${fmt(r.stops, 0)}</td>
          <td>${safeText(r.start_compound || '-')}</td>
          <td>${r.first_pit_lap == null ? '-' : `L${fmt(r.first_pit_lap, 0)}`}</td>
          <td>${pct(r.win_probability)}</td>
        `;
        body.appendChild(tr);
      });
      const minWin = Number(document.getElementById('minWin').value || 0);
      const stop = document.getElementById('stopFilter').value;
      const parts = [];
      if (minWin > 0) parts.push(`min win ${pct(minWin)}`);
      if (stop !== 'all') parts.push(`${stop} stop`);
      if (document.getElementById('searchInput').value.trim()) parts.push('search active');
      document.getElementById('filterState').textContent = parts.length ? parts.join(' | ') : '';
    }

    function openDrawer(row) {
      if (!row) return;
      const race = resolveRaceMeta(row);
      document.getElementById('drawerTitle').textContent = race.location && race.location !== '-'
        ? `${safeText(race.name)} · ${safeText(race.location)}`
        : safeText(race.name);
      const body = document.getElementById('drawerBody');
      body.innerHTML = `
        <div>
          <span class='tag'>${safeText(race.when)}</span>
          <span class='tag'>${safeText(race.location)}</span>
          <span class='tag'>${safeText(row.team)}</span>
          <span class='tag'>${safeText(row.driver)}</span>
          <span class='tag'>${safeText(row.start_compound || row.compounds)}</span>
        </div>
        <div class='drawer-grid'>
          <div class='kv'><p class='k'>Best Strategy</p><p class='v mono'>${safeText(row.best_strategy)}</p></div>
          <div class='kv'><p class='k'>Primary Plan</p><p class='v'>${safeText(row.strategy_plan)}</p></div>
          <div class='kv'><p class='k'>Stops</p><p class='v'>${fmt(row.stops, 0)}</p></div>
          <div class='kv'><p class='k'>First Pit Lap</p><p class='v'>${row.first_pit_lap == null ? '-' : `L${fmt(row.first_pit_lap, 0)}`}</p></div>
          <div class='kv'><p class='k'>Pit Laps</p><p class='v'>${safeText(row.pit_laps)}</p></div>
          <div class='kv'><p class='k'>Win Probability</p><p class='v'>${pct(row.win_probability)}</p></div>
          <div class='kv'><p class='k'>Robustness Window</p><p class='v'>${fmt(row.robustness_window, 2)}s</p></div>
          <div class='kv'><p class='k'>Fallback #2</p><p class='v'>${safeText(row.fallback_2_plan || row.fallback_2_strategy)}</p></div>
          <div class='kv'><p class='k'>Fallback #2 Trigger</p><p class='v'>${safeText(row.fallback_2_trigger)}</p></div>
          <div class='kv'><p class='k'>Fallback #3</p><p class='v'>${safeText(row.fallback_3_plan || row.fallback_3_strategy)}</p></div>
          <div class='kv'><p class='k'>Fallback #3 Trigger</p><p class='v'>${safeText(row.fallback_3_trigger)}</p></div>
        </div>
      `;
      const backdrop = document.getElementById('drawerBackdrop');
      const drawer = document.getElementById('raceDrawer');
      backdrop.classList.add('show');
      drawer.classList.remove('open');
      drawer.classList.remove('bump');
      void drawer.offsetWidth;
      drawer.classList.add('open');
      drawer.classList.add('bump');
    }

    function closeDrawer() {
      document.getElementById('drawerBackdrop').classList.remove('show');
      const drawer = document.getElementById('raceDrawer');
      drawer.classList.remove('open');
      drawer.classList.remove('bump');
    }

    function bindEvents() {
      const tabs = document.getElementById('tabs');
      if (tabs) {
        tabs.addEventListener('click', (event) => {
          const btn = event.target.closest('.tab');
          if (!btn) return;
          setRoute(btn.dataset.route || 'overview');
        });
      }
      window.addEventListener('hashchange', () => {
        animateRouteSwap(() => renderRoute(routeFromHash()));
      });

      document.body.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        const target = event.target.closest('.btn, .tab, .th-btn');
        if (!target) return;
        spawnRipple(target, event.clientX, event.clientY);
      });

      const jumpStrategyEnd = document.getElementById('jumpStrategyEnd');
      if (jumpStrategyEnd) {
        jumpStrategyEnd.addEventListener('click', () => {
          setRoute('strategy');
          window.setTimeout(() => {
            const table = document.getElementById('strategyTableWrap');
            if (table) {
              table.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }, 140);
        });
      }

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
    }

    function renderHeader(payload) {
      const rows = payload.strategy_rows || [];
      const firstYear = rows.length ? safeText(rows[0].year) : "2025";
      document.getElementById('metaLine').textContent = rows.length
        ? `${rows.length} race strategies loaded for the ${firstYear} season.`
        : `Race strategies ready for the ${firstYear} season.`;
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
      bindEvents();
      initCarParallax();
      initHeroSilhouetteScroll();
      renderRoute(routeFromHash());
      initScrollReveal();
      setTimeout(hideLoadingScreen, 1850);
    }

    boot().catch((err) => {
      document.getElementById('metaLine').textContent = `Failed to load website data: ${err}`;
      hideLoadingScreen();
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
        if path in {"/", "/index.html", "/overview", "/races", "/strategy"}:
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

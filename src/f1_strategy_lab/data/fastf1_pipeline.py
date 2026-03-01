from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_strategy_lab.data.weather import get_weather_features

try:
    import fastf1
except ImportError:  # pragma: no cover - optional until user installs deps
    fastf1 = None


class FastF1RateLimitError(RuntimeError):
    """Raised when FastF1 API hourly rate limit is reached."""


TESTING_SESSION_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
)


def _is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return (
        "ratelimit" in name
        or "rate limit" in msg
        or "500 calls/h" in msg
    )


def _require_fastf1() -> None:
    if fastf1 is None:
        raise ImportError(
            "fastf1 is required for live data ingestion. Install dependencies with `pip install -e .`"
        )


def setup_fastf1_cache(cache_dir: str) -> None:
    _require_fastf1()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def _lap_seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def _normalize_team(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _filter_team_driver(laps: pd.DataFrame, team: str, driver: str) -> pd.DataFrame:
    out = laps.copy()
    if "Team" in out.columns and team:
        team_series = out["Team"].astype(str).str.upper()
        team_upper = str(team).upper()
        exact = out[team_series == team_upper]
        if not exact.empty:
            out = exact
        else:
            team_norm = _normalize_team(team_upper)
            team_norm_series = team_series.str.replace(r"[^A-Z0-9]+", "", regex=True)
            contains_mask = team_series.str.contains(team_upper, regex=False) | team_norm_series.str.contains(
                team_norm, regex=False
            )
            out = out[contains_mask]
    if "Driver" in out.columns and driver:
        out = out[out["Driver"].astype(str).str.upper() == driver.upper()]
    out = out[out["LapTime"].notna()]
    if "IsAccurate" in out.columns:
        out = out[out["IsAccurate"] == True]  # noqa: E712
    return out


def _estimate_tire_degradation(laps: pd.DataFrame) -> dict[str, float]:
    deg = {"deg_soft": np.nan, "deg_medium": np.nan, "deg_hard": np.nan}
    if laps.empty or "Compound" not in laps.columns:
        return deg

    laps = laps.copy()
    laps["lap_sec"] = _lap_seconds(laps["LapTime"])
    compounds = {"SOFT": "deg_soft", "MEDIUM": "deg_medium", "HARD": "deg_hard"}

    for compound, out_key in compounds.items():
        comp_laps = laps[laps["Compound"].astype(str).str.upper() == compound]
        if len(comp_laps) < 5:
            continue

        if "TireLife" in comp_laps.columns and comp_laps["TireLife"].notna().sum() >= 5:
            x = comp_laps["TireLife"].astype(float).to_numpy()
        else:
            x = np.arange(len(comp_laps), dtype=float)
        y = comp_laps["lap_sec"].to_numpy(dtype=float)

        slope = float(np.polyfit(x, y, 1)[0])
        deg[out_key] = max(slope, 0.0)

    return deg


def _estimate_fuel_load_proxy(laps: pd.DataFrame) -> float:
    if laps.empty:
        return float("nan")

    work = laps.copy()
    work["lap_sec"] = _lap_seconds(work["LapTime"])
    if "Stint" in work.columns:
        stint_counts = work.groupby("Stint").size().sort_values(ascending=False)
        if not stint_counts.empty:
            work = work[work["Stint"] == stint_counts.index[0]]

    lap_sec = work["lap_sec"].to_numpy(dtype=float)
    if len(lap_sec) < 6:
        return float("nan")

    first = float(np.mean(lap_sec[:3]))
    settled = float(np.mean(lap_sec[-3:]))
    return max(first - settled, 0.0)


def _session_features(session: Any, team: str, driver: str, prefix: str) -> dict[str, float]:
    laps = _filter_team_driver(session.laps, team=team, driver=driver)
    if laps.empty:
        return {
            f"{prefix}_avg_lap_sec": np.nan,
            f"{prefix}_best_lap_sec": np.nan,
            f"{prefix}_fuel_load_proxy": np.nan,
            f"{prefix}_deg_soft": np.nan,
            f"{prefix}_deg_medium": np.nan,
            f"{prefix}_deg_hard": np.nan,
        }

    lap_sec = _lap_seconds(laps["LapTime"])
    deg = _estimate_tire_degradation(laps)
    return {
        f"{prefix}_avg_lap_sec": float(lap_sec.mean()),
        f"{prefix}_best_lap_sec": float(lap_sec.min()),
        f"{prefix}_fuel_load_proxy": _estimate_fuel_load_proxy(laps),
        f"{prefix}_deg_soft": deg["deg_soft"],
        f"{prefix}_deg_medium": deg["deg_medium"],
        f"{prefix}_deg_hard": deg["deg_hard"],
    }


def _race_targets(race_session: Any, team: str, driver: str) -> dict[str, float]:
    laps = _filter_team_driver(race_session.laps, team=team, driver=driver)
    lap_sec = _lap_seconds(laps["LapTime"]) if not laps.empty else pd.Series(dtype=float)

    finish_position = np.nan
    points = np.nan
    results = getattr(race_session, "results", None)
    if results is not None and not results.empty:
        row = results[results["Abbreviation"].astype(str).str.upper() == driver.upper()]
        if row.empty and "TeamName" in results.columns:
            team_rows = results[results["TeamName"].astype(str).str.upper().str.contains(team.upper())]
            if not team_rows.empty:
                row = team_rows.iloc[[0]]
        if not row.empty:
            if "Position" in row.columns:
                finish_position = float(row.iloc[0]["Position"])
            if "Points" in row.columns:
                points = float(row.iloc[0]["Points"])

    return {
        "target_race_pace": float(lap_sec.mean()) if not lap_sec.empty else np.nan,
        "target_finish_position": finish_position,
        "target_points": points,
    }


def _load_session(year: int, event_name: str, session_name: str) -> Any:
    _require_fastf1()
    session = fastf1.get_session(year, event_name, session_name)
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except TypeError:
        session.load()
    return session


def _load_testing_session(year: int, test_number: int, session_number: int) -> Any:
    _require_fastf1()
    session = fastf1.get_testing_session(year, test_number, session_number)
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except TypeError:
        session.load()
    return session


def _testing_baseline_features(year: int, team: str, driver: str) -> dict[str, float]:
    defaults = {
        "test_avg_lap_sec": np.nan,
        "test_best_lap_sec": np.nan,
        "test_fuel_load_proxy": np.nan,
        "test_deg_soft": np.nan,
        "test_deg_medium": np.nan,
        "test_deg_hard": np.nan,
        "test_sessions_used": 0.0,
        "test_laps_used": 0.0,
    }
    collected: list[dict[str, float]] = []

    for test_number, session_number in TESTING_SESSION_CANDIDATES:
        try:
            session = _load_testing_session(year, test_number, session_number)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise FastF1RateLimitError(
                    f"FastF1 rate limit reached while loading testing session {year} test {test_number} session {session_number}"
                ) from exc
            continue

        try:
            session_laps = session.laps
        except Exception:
            # Some historical testing sessions expose metadata without laps.
            continue
        if session_laps is None or len(session_laps) == 0:
            continue

        laps = _filter_team_driver(session_laps, team=team, driver=driver)
        if laps.empty:
            laps = _filter_team_driver(session_laps, team=team, driver="")
        if laps.empty:
            continue

        lap_sec = _lap_seconds(laps["LapTime"])
        deg = _estimate_tire_degradation(laps)
        collected.append(
            {
                "avg_lap_sec": float(lap_sec.mean()),
                "best_lap_sec": float(lap_sec.min()),
                "fuel_load_proxy": _estimate_fuel_load_proxy(laps),
                "deg_soft": deg["deg_soft"],
                "deg_medium": deg["deg_medium"],
                "deg_hard": deg["deg_hard"],
                "lap_count": float(len(laps)),
            }
        )

    if not collected:
        return defaults

    frame = pd.DataFrame(collected)
    out = defaults.copy()
    out.update(
        {
            "test_avg_lap_sec": float(frame["avg_lap_sec"].mean()),
            "test_best_lap_sec": float(frame["best_lap_sec"].min()),
            "test_fuel_load_proxy": float(frame["fuel_load_proxy"].mean()),
            "test_deg_soft": float(frame["deg_soft"].mean()),
            "test_deg_medium": float(frame["deg_medium"].mean()),
            "test_deg_hard": float(frame["deg_hard"].mean()),
            "test_sessions_used": float(len(frame)),
            "test_laps_used": float(frame["lap_count"].sum()),
        }
    )
    return out


def _choose_practice_session(year: int, event_name: str) -> Any:
    for name in ["FP2", "FP3", "FP1"]:
        try:
            return _load_session(year, event_name, name)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise FastF1RateLimitError(
                    f"FastF1 rate limit reached while loading {year} {event_name} {name}"
                ) from exc
            continue
    raise RuntimeError(f"No practice session available for {year} {event_name}")


def _event_datetime_from_schedule(schedule_row: pd.Series) -> datetime:
    for col in ["EventDate", "Session5DateUtc", "Session5Date"]:
        if col in schedule_row and pd.notna(schedule_row[col]):
            return pd.to_datetime(schedule_row[col]).to_pydatetime()
    return datetime(schedule_row.get("Year", 2025), 7, 1, 13, 0, 0)


def _cv_features_for_event(
    event_name: str,
    year: int,
    cv_features_by_event: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    defaults = {"cv_traffic_index": 0.5, "cv_grip_index": 1.0, "cv_rain_index": 0.05}
    if not cv_features_by_event:
        return defaults

    key1 = f"{year}:{event_name}"
    payload = cv_features_by_event.get(key1) or cv_features_by_event.get(event_name) or defaults

    out = defaults.copy()
    for k, v in payload.items():
        if k.startswith("cv_"):
            out[k] = float(v)
        else:
            out[f"cv_{k}"] = float(v)
    return out


def build_event_feature_row(
    year: int,
    event_name: str,
    team: str,
    driver: str,
    event_datetime: datetime,
    weather_cache_dir: str,
    cv_features_by_event: dict[str, dict[str, float]] | None = None,
    testing_features: dict[str, float] | None = None,
    include_targets: bool = True,
) -> dict[str, Any]:
    practice = _choose_practice_session(year, event_name)
    quali = _load_session(year, event_name, "Q")

    row: dict[str, Any] = {
        "year": year,
        "event_name": event_name,
        "team": team.upper(),
        "driver": driver.upper(),
    }

    row.update(_session_features(practice, team=team, driver=driver, prefix="fp2"))
    row.update(_session_features(quali, team=team, driver=driver, prefix="quali"))
    if testing_features:
        row.update(testing_features)
    row.update(get_weather_features(event_name, event_datetime, weather_cache_dir))
    row.update(_cv_features_for_event(event_name, year, cv_features_by_event))

    if include_targets:
        race = _load_session(year, event_name, "R")
        row.update(_race_targets(race, team=team, driver=driver))

    return row


def get_event_schedule(year: int, cache_dir: str) -> pd.DataFrame:
    _require_fastf1()
    setup_fastf1_cache(cache_dir)
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise FastF1RateLimitError(
                f"FastF1 rate limit reached while fetching schedule for {year}"
            ) from exc
        raise
    if "RoundNumber" in schedule.columns:
        schedule = schedule[schedule["RoundNumber"].notna()]
    return schedule


def build_training_dataset(
    years: list[int],
    team: str,
    driver: str,
    fastf1_cache_dir: str,
    weather_cache_dir: str,
    cv_features_by_event: dict[str, dict[str, float]] | None = None,
    include_testing_baseline: bool = True,
) -> pd.DataFrame:
    setup_fastf1_cache(fastf1_cache_dir)
    rows: list[dict[str, Any]] = []
    rate_limited = False

    for year in years:
        testing_features: dict[str, float] | None = None
        if include_testing_baseline:
            try:
                testing_features = _testing_baseline_features(year=year, team=team, driver=driver)
            except FastF1RateLimitError as exc:
                print(f"[WARN] {exc}")
                rate_limited = True
                break

        try:
            schedule = get_event_schedule(year, fastf1_cache_dir)
        except FastF1RateLimitError as exc:
            print(f"[WARN] {exc}")
            rate_limited = True
            break

        for _, event in schedule.iterrows():
            event_name = str(event["EventName"])
            event_dt = _event_datetime_from_schedule(event)
            try:
                row = build_event_feature_row(
                    year=year,
                    event_name=event_name,
                    team=team,
                    driver=driver,
                    event_datetime=event_dt,
                    weather_cache_dir=weather_cache_dir,
                    cv_features_by_event=cv_features_by_event,
                    testing_features=testing_features,
                    include_targets=True,
                )
                rows.append(row)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    print(
                        f"[WARN] FastF1 rate limit reached at {year} {event_name}. "
                        "Returning partial training dataset from cached progress."
                    )
                    rate_limited = True
                    break
                print(f"[WARN] Skipping {year} {event_name}: {exc}")
        if rate_limited:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        if rate_limited:
            raise FastF1RateLimitError(
                "FastF1 rate limit reached before collecting training rows. "
                "Wait for the hourly reset and rerun."
            )
        return frame

    frame = frame.dropna(subset=["target_race_pace"]).reset_index(drop=True)
    return frame


def build_prerace_dataset(
    year: int,
    team: str,
    driver: str,
    fastf1_cache_dir: str,
    weather_cache_dir: str,
    cv_features_by_event: dict[str, dict[str, float]] | None = None,
    include_testing_baseline: bool = True,
) -> pd.DataFrame:
    setup_fastf1_cache(fastf1_cache_dir)
    schedule = get_event_schedule(year, fastf1_cache_dir)
    testing_features: dict[str, float] | None = None
    if include_testing_baseline:
        testing_features = _testing_baseline_features(year=year, team=team, driver=driver)

    rows: list[dict[str, Any]] = []
    for _, event in schedule.iterrows():
        event_name = str(event["EventName"])
        event_dt = _event_datetime_from_schedule(event)
        try:
            row = build_event_feature_row(
                year=year,
                event_name=event_name,
                team=team,
                driver=driver,
                event_datetime=event_dt,
                weather_cache_dir=weather_cache_dir,
                cv_features_by_event=cv_features_by_event,
                testing_features=testing_features,
                include_targets=False,
            )
            rows.append(row)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise FastF1RateLimitError(
                    f"FastF1 rate limit reached while loading pre-race data for {year} {event_name}. "
                    "Wait for hourly reset and rerun."
                ) from exc
            print(f"[WARN] Pre-race row skipped for {year} {event_name}: {exc}")

    return pd.DataFrame(rows)

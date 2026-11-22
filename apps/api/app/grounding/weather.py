"""Open-Meteo client and the deterministic spray-window rules.

Both the 12h spray verdict and the 48h hourly timeline are computed here in
code, never by the LLM. A fabricated "it will rain tomorrow, spray today" is the
worst possible failure for this product, because weather decides whether a
spray is washed off or wasted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from ..config import settings

log = logging.getLogger(__name__)

#: Rain probability above this in the next 12h blocks spraying.
RAIN_THRESHOLD_PCT = 60
#: Wind above this (km/h) makes spraying drift-prone and wasteful.
WIND_THRESHOLD_KMH = 15.0
#: Hours ahead considered for the immediate spray verdict.
HORIZON_HOURS = 12

#: 48-hour spray timeline thresholds.
TIMELINE_HOURS = 48
RAIN_MM_RED = 1.0
RAIN_PROB_AMBER = 30
WIND_AMBER_KMH = 10.0
WASHOFF_LOOKAHEAD_HOURS = 3
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18

HUMIDITY_FIELD = "relative_humidity_2m"


def _dhaka_tz() -> tzinfo:
    try:
        return ZoneInfo("Asia/Dhaka")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=6))


_DHAKA_TZ = _dhaka_tz()


def _current_dhaka_hour() -> datetime:
    return datetime.now(_dhaka_tz()).replace(minute=0, second=0, microsecond=0)


def _parse_dhaka_dt(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    tz = _dhaka_tz()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)

SpraySafety = Literal["green", "amber", "red"]


@dataclass(frozen=True, slots=True)
class HourlySprayWindow:
    time_iso: str
    rain_prob: int
    rain_mm: float
    wind_kmh: float
    safety: SpraySafety

    def as_contract_dict(self) -> dict[str, Any]:
        return {
            "timeIso": self.time_iso,
            "rainProb": self.rain_prob,
            "rainMm": round(self.rain_mm, 2),
            "windKmh": round(self.wind_kmh, 1),
            "safety": self.safety,
        }


@dataclass(frozen=True, slots=True)
class WeatherFacts:
    district_id: str
    district_name: str
    rain_probability: int
    wind_speed_max: float
    humidity: int
    spray_window_ok: bool
    reason: str
    hourly_windows: tuple[HourlySprayWindow, ...] = ()
    safest_window_summary_bn: str | None = None

    def as_prompt_facts(self) -> dict[str, Any]:
        """The shape handed to the LLM. Facts only, no instructions."""
        return {
            "district": self.district_name,
            "rain_probability_next_12h": self.rain_probability,
            "wind_speed_max_next_12h": round(self.wind_speed_max, 1),
            "humidity": self.humidity,
            "spray_window_ok": self.spray_window_ok,
            "reason": self.reason,
            "safest_window_summary_bn": self.safest_window_summary_bn,
        }


_cache: dict[str, tuple[float, WeatherFacts]] = {}
_cache_seconds = settings.weather_cache_minutes * 60


def _bn(value: int | float) -> str:
    """Digits as a Bangla reader expects them, in prose only."""
    return str(value).translate({ord(str(d)): "০১২৩৪৫৬৭৮৯"[d] for d in range(10)})


def _decide(rain: int, wind: float) -> tuple[bool, str]:
    """The 12-hour spray verdict.

    Order matters: rain is the more decisive factor, and the reason string is shown
    to a farmer, so it names the binding constraint rather than listing both.
    """
    if rain >= RAIN_THRESHOLD_PCT:
        return False, f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বৃষ্টির সম্ভাবনা {_bn(rain)} শতাংশ"
    if wind >= WIND_THRESHOLD_KMH:
        return (
            False,
            f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বাতাসের গতি প্রায় {_bn(round(wind))} কিলোমিটার",
        )
    return True, f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বৃষ্টি ও বাতাস স্প্রে করার অনুকূল"


def _classify_hour(
    rain_prob: int,
    rain_mm: float,
    wind_kmh: float,
    *,
    washoff_risk: bool,
) -> SpraySafety:
    if (
        rain_prob >= RAIN_THRESHOLD_PCT
        or rain_mm >= RAIN_MM_RED
        or wind_kmh >= WIND_THRESHOLD_KMH
        or washoff_risk
    ):
        return "red"
    if rain_prob >= RAIN_PROB_AMBER or rain_mm > 0.0 or wind_kmh >= WIND_AMBER_KMH:
        return "amber"
    return "green"


def _build_timeline(hourly: dict[str, Any]) -> tuple[HourlySprayWindow, ...]:
    raw_probs: list[Any] = list(hourly.get("precipitation_probability") or [])[:TIMELINE_HOURS]
    raw_mms: list[Any] = list(hourly.get("precipitation") or [])[:TIMELINE_HOURS]
    raw_winds: list[Any] = list(hourly.get("wind_speed_10m") or [])[:TIMELINE_HOURS]
    raw_times: list[Any] = list(hourly.get("time") or [])[:TIMELINE_HOURS]

    count = max(len(raw_probs), len(raw_winds), len(raw_times))
    count = min(count, TIMELINE_HOURS)
    if count == 0:
        return ()

    base_dt = datetime(2026, 1, 1, 6, 0, tzinfo=_DHAKA_TZ)
    probs: list[int] = []
    mms: list[float] = []
    winds: list[float] = []
    times: list[str] = []

    for idx in range(count):
        prob_val = raw_probs[idx] if idx < len(raw_probs) and raw_probs[idx] is not None else 0
        mm_val = raw_mms[idx] if idx < len(raw_mms) and raw_mms[idx] is not None else 0.0
        wind_val = raw_winds[idx] if idx < len(raw_winds) and raw_winds[idx] is not None else 0.0
        time_val = (
            str(raw_times[idx])
            if idx < len(raw_times) and raw_times[idx] is not None
            else (base_dt + timedelta(hours=idx)).isoformat(timespec="minutes")
        )
        probs.append(int(prob_val))
        mms.append(float(mm_val))
        winds.append(float(wind_val))
        times.append(time_val)

    windows: list[HourlySprayWindow] = []
    for idx in range(count):
        lookahead_end = min(count, idx + 1 + WASHOFF_LOOKAHEAD_HOURS)
        washoff_risk = any(
            probs[j] >= RAIN_THRESHOLD_PCT or mms[j] >= RAIN_MM_RED
            for j in range(idx + 1, lookahead_end)
        )
        safety = _classify_hour(
            probs[idx],
            mms[idx],
            winds[idx],
            washoff_risk=washoff_risk,
        )
        windows.append(
            HourlySprayWindow(
                time_iso=times[idx],
                rain_prob=probs[idx],
                rain_mm=round(mms[idx], 2),
                wind_kmh=round(winds[idx], 1),
                safety=safety,
            )
        )
    return tuple(windows)


def _format_bn_start_hour(hour: int) -> str:
    if 6 <= hour < 12:
        return f"সকাল {_bn(hour)}টা"
    if hour == 12:
        return "দুপুর ১২টা"
    if 13 <= hour <= 14:
        return f"দুপুর {_bn(hour - 12)}টা"
    if 15 <= hour < 18:
        return f"বিকেল {_bn(hour - 12)}টা"
    if hour == 18:
        return "সন্ধ্যা ৬টা"
    if hour == 0:
        return "রাত ১২টা"
    if 1 <= hour < 6:
        return f"ভোর {_bn(hour)}টা"
    return f"রাত {_bn(hour - 12)}টা"


def _format_bn_end_hour(hour: int) -> str:
    if hour <= 12:
        return f"{_bn(hour)}টা"
    return f"{_bn(hour - 12)}টা"


def _day_label_bn(day_offset: int) -> str:
    if day_offset <= 0:
        return "আজ"
    if day_offset == 1:
        return "আগামীকাল"
    return "পরশু"


def _compute_safest_window_summary_bn(
    windows: tuple[HourlySprayWindow, ...],
    *,
    reference_date: date | None = None,
) -> str | None:
    if not windows:
        return None

    try:
        parsed = [(w, _parse_dhaka_dt(w.time_iso)) for w in windows]
    except ValueError:
        return None

    first_date = reference_date if reference_date is not None else parsed[0][1].date()

    blocks: list[list[tuple[int, datetime]]] = []
    current: list[tuple[int, datetime]] = []

    for w, dt in parsed:
        is_daylight_green = (
            w.safety == "green" and DAYLIGHT_START_HOUR <= dt.hour < DAYLIGHT_END_HOUR
        )
        if is_daylight_green:
            day_offset = (dt.date() - first_date).days
            if current:
                prev_offset, prev_dt = current[-1]
                if day_offset == prev_offset and dt.hour == prev_dt.hour + 1:
                    current.append((day_offset, dt))
                    continue
            if current:
                blocks.append(current)
            current = [(day_offset, dt)]
        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    chosen = next((blk for blk in blocks if len(blk) >= 2), blocks[0] if blocks else None)
    if not chosen:
        return "আগামী ৪৮ ঘণ্টায় দিনের বেলায় নিরাপদ স্প্রে সময় নেই; বৃষ্টি ও বাতাস কমার পর স্প্রে করুন"

    day_offset, start_dt = chosen[0]
    _, end_dt = chosen[-1]
    end_hour = min(DAYLIGHT_END_HOUR, end_dt.hour + 1)
    day_bn = _day_label_bn(day_offset)
    start_bn = _format_bn_start_hour(start_dt.hour)
    end_bn = _format_bn_end_hour(end_hour)
    return f"সবচেয়ে নিরাপদ স্প্রে সময়: {day_bn} {start_bn} থেকে {end_bn}"


def _slice_future_hourly(hourly: dict[str, Any], current_hour: datetime) -> dict[str, Any]:
    raw_times = hourly.get("time")
    if not raw_times:
        return hourly

    start_idx: int | None = None
    for idx, raw_time in enumerate(raw_times):
        try:
            slot_dt = _parse_dhaka_dt(str(raw_time))
        except ValueError:
            continue
        if slot_dt >= current_hour:
            start_idx = idx
            break

    if start_idx is None:
        return {
            key: ([] if isinstance(value, list) else value)
            for key, value in hourly.items()
        }

    end_idx = start_idx + TIMELINE_HOURS
    return {
        key: (value[start_idx:end_idx] if isinstance(value, list) else value)
        for key, value in hourly.items()
    }


def classify(
    payload: dict[str, Any],
    district_id: str,
    district_name: str,
    *,
    now: datetime | None = None,
) -> WeatherFacts:
    """Turn an Open-Meteo response into facts. Split out so it is testable."""
    tz = _dhaka_tz()
    if now is None:
        current_hour = _current_dhaka_hour()
    elif now.tzinfo is None:
        current_hour = now.replace(tzinfo=tz, minute=0, second=0, microsecond=0)
    else:
        current_hour = now.astimezone(tz).replace(minute=0, second=0, microsecond=0)

    hourly = _slice_future_hourly(payload["hourly"], current_hour)
    rain_series: list[int | None] = list(hourly.get("precipitation_probability") or [])[
        :HORIZON_HOURS
    ]
    wind_series: list[float | None] = list(hourly.get("wind_speed_10m") or [])[:HORIZON_HOURS]
    raw_humidity: list[int | None] = list(hourly.get(HUMIDITY_FIELD) or [])[:HORIZON_HOURS]
    humidity_values = [h for h in raw_humidity if h is not None]

    # None means "model not confident"; treat a missing value as non-blocking
    # rather than inventing bad weather.
    rain = max((value for value in rain_series if value is not None), default=0)
    wind = max((value for value in wind_series if value is not None), default=0.0)
    humidity = round(sum(humidity_values) / len(humidity_values)) if humidity_values else 0

    ok, reason = _decide(rain, wind)
    windows = _build_timeline(hourly)
    safest_summary = _compute_safest_window_summary_bn(
        windows, reference_date=current_hour.date()
    )

    return WeatherFacts(
        district_id=district_id,
        district_name=district_name,
        rain_probability=rain,
        wind_speed_max=wind,
        humidity=humidity,
        spray_window_ok=ok,
        reason=reason,
        hourly_windows=windows,
        safest_window_summary_bn=safest_summary,
    )


async def fetch_weather(
    latitude: float, longitude: float, district_id: str, district_name: str
) -> WeatherFacts | None:
    """Fetch 48h of future weather. Returns None on any failure - weather is optional.

    A missing weather block means the LLM is told not to mention weather at all,
    which is honest. An invented forecast is not.
    """
    current_hour = _current_dhaka_hour()
    cache_key = f"{district_id}:{current_hour.isoformat()}"
    cached = _cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _cache_seconds:
        return cached[1]

    params: dict[str, str | int | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": f"precipitation_probability,precipitation,wind_speed_10m,{HUMIDITY_FIELD}",
        "forecast_days": 3,
        "timezone": "Asia/Dhaka",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.weather_timeout_ms / 1000) as client:
            response = await client.get(f"{settings.open_meteo_base_url}/forecast", params=params)
            response.raise_for_status()
            facts = classify(response.json(), district_id, district_name, now=current_hour)
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        # `str(ConnectTimeout())` is empty, so a bare %s logs nothing useful.
        # This is the only signal that a result lost its weather block.
        log.warning(
            "weather fetch failed for %s: %s: %s",
            district_id,
            exc.__class__.__name__,
            str(exc) or "no detail",
        )
        return None

    for old_key in [k for k in _cache if k.startswith(f"{district_id}:") and k != cache_key]:
        _cache.pop(old_key, None)
    _cache[cache_key] = (time.monotonic(), facts)
    return facts

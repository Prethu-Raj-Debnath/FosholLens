"""Open-Meteo client and the deterministic spray-window rule.

The rule is computed here, in code, never by the LLM. A fabricated "it will rain
tomorrow, spray today" is the worst possible failure for this product, because
weather is exactly what decides whether a spray is wasted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)

#: Rain probability above this in the next 12h blocks spraying.
RAIN_THRESHOLD_PCT = 60
#: Wind above this (km/h) makes spraying drift-prone and wasteful.
WIND_THRESHOLD_KMH = 15.0
#: Hours ahead considered for the spray verdict.
HORIZON_HOURS = 12

HUMIDITY_FIELD = "relative_humidity_2m"


@dataclass(frozen=True, slots=True)
class WeatherFacts:
    district_id: str
    district_name: str
    rain_probability: int
    wind_speed_max: float
    humidity: int
    spray_window_ok: bool
    reason: str

    def as_prompt_facts(self) -> dict[str, Any]:
        """The shape handed to the LLM. Facts only, no instructions."""
        return {
            "district": self.district_name,
            "rain_probability_next_12h": self.rain_probability,
            "wind_speed_max_next_12h": round(self.wind_speed_max, 1),
            "humidity": self.humidity,
            "spray_window_ok": self.spray_window_ok,
            "reason": self.reason,
        }


_cache: dict[str, tuple[float, WeatherFacts]] = {}
_cache_seconds = settings.weather_cache_minutes * 60


def _bn(value: int | float) -> str:
    """Digits as a Bangla reader expects them, in prose only."""
    return str(value).translate({ord(str(d)): "০১২৩৪৫৬৭৮৯"[d] for d in range(10)})


def _decide(rain: int, wind: float) -> tuple[bool, str]:
    """The spray verdict.

    Order matters: rain is the more decisive factor, and the reason string is shown
    to a farmer, so it names the binding constraint rather than listing both.
    """
    if rain >= RAIN_THRESHOLD_PCT:
        return False, f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বৃষ্টির সম্ভাবনা {_bn(rain)} শতাংশ"
    if wind >= WIND_THRESHOLD_KMH:
        return False, f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বাতাসের গতি প্রায় {_bn(round(wind))} কিলোমিটার"
    return True, f"আগামী {_bn(HORIZON_HOURS)} ঘণ্টায় বৃষ্টি ও বাতাস স্প্রে করার অনুকূল"


def classify(payload: dict[str, Any], district_id: str, district_name: str) -> WeatherFacts:
    """Turn an Open-Meteo response into facts. Split out so it is testable."""
    hourly = payload["hourly"]
    rain_series: list[int] = hourly["precipitation_probability"][:HORIZON_HOURS]
    wind_series: list[float] = hourly["wind_speed_10m"][:HORIZON_HOURS]
    humidity_series: list[int] = hourly[HUMIDITY_FIELD][:HORIZON_HOURS]

    # None means "model not confident"; treat a missing value as non-blocking
    # rather than inventing bad weather.
    rain = max((value for value in rain_series if value is not None), default=0)
    wind = max((value for value in wind_series if value is not None), default=0.0)
    humidity = round(sum(humidity_series) / len(humidity_series)) if humidity_series else 0

    ok, reason = _decide(rain, wind)
    return WeatherFacts(
        district_id=district_id,
        district_name=district_name,
        rain_probability=rain,
        wind_speed_max=wind,
        humidity=humidity,
        spray_window_ok=ok,
        reason=reason,
    )


async def fetch_weather(
    latitude: float, longitude: float, district_id: str, district_name: str
) -> WeatherFacts | None:
    """Fetch 48h of weather. Returns None on any failure - weather is optional.

    A missing weather block means the LLM is told not to mention weather at all,
    which is honest. An invented forecast is not.
    """
    cached = _cache.get(district_id)
    if cached and time.monotonic() - cached[0] < _cache_seconds:
        return cached[1]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": f"precipitation_probability,precipitation,wind_speed_10m,{HUMIDITY_FIELD}",
        "forecast_days": 2,
        "timezone": "Asia/Dhaka",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.weather_timeout_ms / 1000) as client:
            response = await client.get(f"{settings.open_meteo_base_url}/forecast", params=params)
            response.raise_for_status()
            facts = classify(response.json(), district_id, district_name)
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

    _cache[district_id] = (time.monotonic(), facts)
    return facts

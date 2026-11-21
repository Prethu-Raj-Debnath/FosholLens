"""Weather module stub for Open-Meteo 48-hour spray-window guard."""

from app.schemas import WeatherResult


async def get_spray_windows(lat: float, lng: float) -> WeatherResult:
    """Fetch 48-hour hourly weather from Open-Meteo and score spray safety windows.

    TODO (RAG/Weather/LLM Teammate):
      - Query Open-Meteo forecast API via `httpx.AsyncClient` for `lat`, `lng`.
      - Evaluate hourly rain probability, precipitation (mm), and wind speed (km/h).
      - Classify each window as `green`, `amber`, or `red` and generate a concise
        `safest_window_summary`.
    """
    raise NotImplementedError(
        "Weather spray-window guard (Open-Meteo 48h forecast) is not yet implemented."
    )

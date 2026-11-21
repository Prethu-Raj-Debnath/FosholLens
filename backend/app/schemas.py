"""Pydantic schemas defining the strict FosholLens API contract."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

CropType = Literal["rice", "potato"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "foshollens-api"


class PredictionItem(BaseModel):
    label: str = Field(..., description="Predicted disease or healthy class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0-1).")


class VisionResult(BaseModel):
    top_predictions: list[PredictionItem]
    status: Literal["confident", "uncertain"]


class Location(BaseModel):
    lat: float
    lng: float


class SprayWindow(BaseModel):
    time_iso: str = Field(..., description="ISO-8601 timestamp for the forecast window.")
    rain_prob: int = Field(..., ge=0, le=100, description="Precipitation probability (%).")
    rain_mm: float = Field(..., ge=0.0, description="Expected rainfall in millimeters.")
    wind_kmh: float = Field(..., ge=0.0, description="Wind speed in km/h.")
    safety: Literal["green", "amber", "red"]


class WeatherResult(BaseModel):
    location: Location
    next_48h_windows: list[SprayWindow]
    safest_window_summary: str


class RagSource(BaseModel):
    source_title: str
    crop: str
    disease: str
    excerpt: str


class AdvisoryResult(BaseModel):
    bangla_text: str
    ppe_reminder: str
    officer_referral: bool
    grounded_in_rag: bool


class AnalyzeResponse(BaseModel):
    crop: str
    vision: VisionResult
    weather: WeatherResult
    rag_sources: list[RagSource]
    advisory: AdvisoryResult
    audio_url_or_base64: Optional[str] = None

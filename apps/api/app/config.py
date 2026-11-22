"""Runtime configuration.

Every value is read from the environment (or apps/api/.env) and validated once at
startup. A missing required value fails loudly here rather than at the first
request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent
API_ROOT = APP_DIR.parent
REPO_ROOT = API_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", API_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- core ---------------------------------------------------------
    app_env: Literal["development", "test", "production"] = "development"
    port: int = 4000
    cors_origin: str = "http://localhost:5173"

    # ---- LLM (optional: without a key the template writer is used) ----
    llm_provider: Literal["auto", "gemini", "openai"] = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    llm_base_url: str = "https://tokens.bd/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-6-luna"
    llm_timeout_ms: int = 15_000
    llm_max_concurrency: int = Field(default=3, ge=1, le=10)
    max_question_chars: int = Field(default=300, ge=50, le=1000)
    max_audio_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    audio_transcription_timeout_ms: int = Field(default=12_000, ge=1000)

    # ---- vision (optional: without a model every diagnosis is unconfirmed)
    vision_enabled: bool = True
    #: Which checkpoint to use.
    #:   vit          - 13 classes, corn/potato/rice/wheat, downloaded from HF.
    #:   efficientnet - 9 classes, rice/potato only, local file, ~3x faster.
    #: Everything else that differs between them (label order, crop mapping,
    #: preprocessing) lives in `vision/models.py`, so this is the only switch.
    vision_model: Literal["vit", "efficientnet"] = "efficientnet"
    #: Where a local checkpoint lives. Unused when the spec names a Hub repo.
    vision_model_dir: Path = API_ROOT / "models"
    #: Raw softmax probability below which a leaf is "not confidently
    #: identified". Applied before any crop filtering, so a wrong-crop photo
    #: cannot be renormalised into a confident answer.
    vision_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # ---- TTS (optional: disabled without the model installed) ---------
    tts_enabled: bool = True
    tts_model_id: str = "sulabhkatiyar/indic-tiny-tts-bengali"
    tts_voice: Literal["female", "male"] = "female"
    tts_format: Literal["opus", "mp3"] = "opus"
    ffmpeg_path: str = "ffmpeg"

    # ---- weather ------------------------------------------------------
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"
    weather_cache_minutes: int = 30
    #: 2.5 s is generous for Open-Meteo (measured ~0.7-1.0 s), but a cold DNS
    #: lookup or a congested mobile link can exceed it. Weather is optional, so a
    #: miss degrades the advice rather than failing the request.
    weather_timeout_ms: int = 4_000

    # ---- storage ------------------------------------------------------
    database_path: Path = API_ROOT / "storage" / "foshol-lens.db"
    storage_dir: Path = API_ROOT / "storage"
    public_base_url: str = "http://localhost:4000"
    analysis_retention_days: int = 30

    @field_validator("llm_base_url", "open_meteo_base_url", "public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("vision_model_dir", "storage_dir", "database_path")
    @classmethod
    def _anchor_relative_paths_to_api_root(cls, value: Path) -> Path:
        """Resolve a relative configured path against `apps/api`, not the CWD.

        `.env.example` documents `VISION_MODEL_DIR=./models`. Taken literally that
        is CWD-relative, so `uvicorn app.main:app` from the repo root would look
        for `<repo>/models` and find nothing — failing identically to a missing
        file, which is exactly the silent degradation this project just removed.
        Anchoring to `API_ROOT` makes the documented value correct regardless of
        where the server is launched from.
        """
        return value if value.is_absolute() else (API_ROOT / value).resolve()

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origin.split(",") if origin.strip()]

    @property
    def effective_gemini_api_key(self) -> str:
        return (self.gemini_api_key or self.llm_api_key).strip()

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.llm_api_key.strip())
        return bool(self.gemini_api_key.strip() or self.llm_api_key.strip())

    @property
    def is_development(self) -> bool:
        """Development is the only place a moving port is acceptable."""
        return self.app_env == "development"

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @property
    def audio_dir(self) -> Path:
        return self.storage_dir / "audio"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()

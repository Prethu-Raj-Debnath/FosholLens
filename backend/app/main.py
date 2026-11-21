"""FastAPI entrypoint for FosholLens backend."""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import llm, tts, vision
from app.schemas import (
    AnalyzeResponse,
    CropType,
    HealthResponse,
    Location,
    RagSource,
    WeatherResult,
)

# Load .env from repo root first (if present), then current working directory
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV)
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LAT = 23.4607  # Cumilla, Bangladesh
DEFAULT_LNG = 91.1809  # Cumilla, Bangladesh

app = FastAPI(
    title="FosholLens API",
    description="Multimodal Bangla Voice & Weather-Aware Crop Advisory for Rice and Potato",
    version="0.2.0",
)

cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    origin.strip()
    for origin in cors_origins_env.split(",")
    if origin.strip()
] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", service="foshollens-api")


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Advisory"])
async def analyze_crop(
    image: UploadFile = File(..., description="Leaf photo (required)"),
    crop: CropType = Form(..., description="Crop type: 'rice' or 'potato'"),
    audio: Optional[UploadFile] = File(
        default=None, description="Optional 5-10s Bangla voice question"
    ),
    question_text: Optional[str] = Form(
        default=None, description="Optional Bangla or English text question"
    ),
    lat: Optional[float] = Form(
        default=DEFAULT_LAT, description="Latitude (defaults to Cumilla: 23.4607)"
    ),
    lng: Optional[float] = Form(
        default=DEFAULT_LNG, description="Longitude (defaults to Cumilla: 91.1809)"
    ),
) -> AnalyzeResponse:
    """Run the Day 1 end-to-end multimodal crop advisory pipeline.

    Pipeline: Image -> ViT Vision Classifier -> Gemini Flash Bangla Advisory -> Bangla TTS.
    Weather 48h guard and BARI/BRRI RAG are scheduled for Day 2.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded leaf image is empty.")

    audio_bytes = await audio.read() if audio is not None else None
    resolved_lat = DEFAULT_LAT if lat is None else lat
    resolved_lng = DEFAULT_LNG if lng is None else lng

    # 1. Vision Classification (Hugging Face ViT)
    try:
        vision_result = await vision.classify_leaf(
            image_bytes=image_bytes,
            crop=crop,
            content_type=image.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Vision inference failed.")
        raise HTTPException(
            status_code=500, detail=f"Vision classification error: {exc}"
        ) from exc

    # 2. Temporary empty WeatherResult & RAG sources (Day 2)
    weather_result = WeatherResult(
        location=Location(lat=resolved_lat, lng=resolved_lng),
        next_48h_windows=[],
        safest_window_summary="Weather guard pending (Day 2)",
    )
    rag_sources: list[RagSource] = []

    # 3. Gemini Flash Bangla Advisory
    try:
        advisory_result = await llm.generate_advisory(
            crop=crop,
            vision=vision_result,
            weather=None,
            rag_sources=rag_sources,
            audio_bytes=audio_bytes,
            audio_content_type=audio.content_type if audio is not None else None,
            question_text=question_text,
        )
    except RuntimeError as exc:
        logger.error("Advisory generation configuration/runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gemini advisory generation failed.")
        raise HTTPException(
            status_code=502, detail=f"Bangla advisory generation failed: {exc}"
        ) from exc

    # 4. Bangla TTS (gracefully returns None on failure)
    audio_output = await tts.synthesize_bangla_audio(advisory_result.bangla_text)

    return AnalyzeResponse(
        crop=crop,
        vision=vision_result,
        weather=weather_result,
        rag_sources=rag_sources,
        advisory=advisory_result,
        audio_url_or_base64=audio_output,
    )

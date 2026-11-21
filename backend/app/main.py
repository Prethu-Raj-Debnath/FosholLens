"""FastAPI entrypoint for FosholLens backend."""

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import llm, rag, tts, vision, weather
from app.schemas import AnalyzeResponse, CropType, HealthResponse

load_dotenv()

DEFAULT_LAT = 23.4607  # Cumilla, Bangladesh
DEFAULT_LNG = 91.1809  # Cumilla, Bangladesh

app = FastAPI(
    title="FosholLens API",
    description="Multimodal Bangla Voice & Weather-Aware Crop Advisory for Rice and Potato",
    version="0.1.0",
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
    """Run the multimodal crop advisory pipeline.

    Returns HTTP 501 (Not Implemented) until the underlying AI/weather/RAG/TTS
    modules are implemented. Never fabricates diagnoses or advisories.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded leaf image is empty.")

    audio_bytes = await audio.read() if audio is not None else None
    resolved_lat = DEFAULT_LAT if lat is None else lat
    resolved_lng = DEFAULT_LNG if lng is None else lng

    try:
        vision_result = await vision.classify_leaf(
            image_bytes=image_bytes,
            crop=crop,
            content_type=image.content_type,
        )
        weather_result = await weather.get_spray_windows(
            lat=resolved_lat,
            lng=resolved_lng,
        )
        top_label = (
            vision_result.top_predictions[0].label
            if vision_result.top_predictions
            else "unknown"
        )
        rag_sources = await rag.retrieve_guidance(
            crop=crop,
            disease_label=top_label,
            question_text=question_text,
        )
        advisory_result = await llm.generate_advisory(
            crop=crop,
            vision=vision_result,
            weather=weather_result,
            rag_sources=rag_sources,
            audio_bytes=audio_bytes,
            audio_content_type=audio.content_type if audio is not None else None,
            question_text=question_text,
        )
        audio_output = await tts.synthesize_bangla_speech(
            bangla_text=advisory_result.bangla_text
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return AnalyzeResponse(
        crop=crop,
        vision=vision_result,
        weather=weather_result,
        rag_sources=rag_sources,
        advisory=advisory_result,
        audio_url_or_base64=audio_output,
    )

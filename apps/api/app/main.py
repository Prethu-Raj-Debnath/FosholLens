"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .config import settings
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("foshol_lens")

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await store.init_db()
    purged = await store.purge_older_than(settings.analysis_retention_days)
    if purged:
        log.info("purged %d analyses older than %d days", purged, settings.analysis_retention_days)
    log.info("LLM: %s", "enabled" if settings.llm_configured else "template fallback")
    log.info("TTS: %s", "enabled" if settings.tts_enabled else "disabled")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ফসল লেন্স API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # A dev server that drifts off port 5173 still has to work: Vite silently
    # picks the next free port, and rejecting that surfaces as an upload failure
    # that looks like a broken server rather than a CORS problem.
    dev_origin_regex = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=dev_origin_regex if settings.is_development else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse


router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    # Minimal Prometheus-compatible endpoint.
    # Expand with real counters/histograms when you add middleware instrumentation.
    body = (
        "# HELP bot_backend_build_info Build and runtime info\n"
        "# TYPE bot_backend_build_info gauge\n"
        "bot_backend_build_info{service=\"bot-backend\"} 1\n"
    )
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

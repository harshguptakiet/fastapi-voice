from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import APP_NAME, ENV
from app.core.validation import validate_configuration
from app.core.database import Base, engine
from app.routers.interactions import router as interactions_router
from app.routers.llm import router as llm_router
from app.routers.metrics import router as metrics_router
from app.routers.sessions import router as sessions_router
from app.routers.status import router as status_router
from app.routers.transcripts import router as transcripts_router
from app.routers.users import router as users_router
from app.routers.voice import router as voice_router


def create_app() -> FastAPI:
    validate_configuration()

    application = FastAPI(title=APP_NAME)

    # Serve test UIs and other static assets.
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # CORS for local testing.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(users_router)
    application.include_router(interactions_router)
    application.include_router(voice_router)
    application.include_router(sessions_router)
    application.include_router(llm_router)
    application.include_router(transcripts_router)
    application.include_router(status_router)
    application.include_router(metrics_router)

    @application.get("/health")
    def health():
        return {"status": "ok"}

    return application


app = create_app()


@app.on_event("startup")
def on_startup():
    # In production, run migrations instead.
    if ENV.lower() == "dev":
        Base.metadata.create_all(bind=engine)

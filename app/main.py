from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_NAME, ENV
from app.core.database import Base, engine
from app.routers.users import router as users_router
from app.routers.voice import router as voice_router


def create_app() -> FastAPI:
    application = FastAPI(title=APP_NAME)

    # CORS for local testing (voice-test.html, tools, etc.)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(users_router)
    application.include_router(voice_router)

    @application.get("/health")
    def health():
        return {"status": "ok"}

    return application


app = create_app()


@app.on_event("startup")
def on_startup():
    # Auto-create tables only in dev environment; in prod run Alembic migrations
    if ENV.lower() == "dev":
        Base.metadata.create_all(bind=engine)

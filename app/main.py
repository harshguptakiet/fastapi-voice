from fastapi import FastAPI

from app.core.config import APP_NAME, ENV
from app.core.database import Base, engine
from app.routers.users import router as users_router


def create_app() -> FastAPI:
    application = FastAPI(title=APP_NAME)

    # Routers
    application.include_router(users_router)

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

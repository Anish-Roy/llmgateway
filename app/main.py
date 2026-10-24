from fastapi import FastAPI

from app import __version__
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LLM Gateway", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env, "version": __version__}

    return app


app = create_app()

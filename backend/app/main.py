from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router
from .database import init_database


def create_application() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Aldar Kose API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    @app.on_event("startup")
    def startup_event() -> None:
        init_database()

    @app.get("/health", tags=["system"], summary="Health check")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_application()

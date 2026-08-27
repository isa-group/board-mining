"""FastAPI application factory for the bomi dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import board, structure, flow, health, cards
from .oauth import router as oauth_router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="bomi dashboard", docs_url="/api/docs")

    # Allow Vite dev server to proxy during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(board.router)
    app.include_router(structure.router)
    app.include_router(flow.router)
    app.include_router(health.router)
    app.include_router(cards.router)
    app.include_router(oauth_router)

    # Serve pre-built Svelte app if the dist exists
    if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app

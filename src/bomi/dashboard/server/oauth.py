"""Trello OAuth 1.0a helpers.

Stub implementation — wired up but not yet active.  The Flask/FastAPI route
/oauth/callback is registered in main.py; the front-end DataPanel shows a
"Connect Trello" button that is disabled until this module is completed.

Flow (when implemented):
  1. GET  /oauth/start          → redirect user to Trello authorise URL
  2. GET  /oauth/callback?...   → exchange verifier for access token, store in session
  3. GET  /api/board/my-boards  → list boards for authenticated user
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.get("/start")
async def oauth_start():
    # TODO: build Trello OAuth 1.0a request-token URL and redirect
    return {"detail": "Trello OAuth not yet implemented."}


@router.get("/callback")
async def oauth_callback(oauth_token: str = "", oauth_verifier: str = ""):
    # TODO: exchange (oauth_token, oauth_verifier) for access token
    return {"detail": "Trello OAuth not yet implemented."}

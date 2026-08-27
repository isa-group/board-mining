"""Board loading endpoints: file upload, board ID, info summary."""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

import bomi
from .. import loader

router = APIRouter(prefix="/api/board", tags=["board"])


class BoardIdRequest(BaseModel):
    board_id: str


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".json"):
            df = loader.load_from_json_bytes(contents)
        else:
            df = loader.load_from_csv_bytes(contents, filename)
        loader.set_board(df)
        return {"ok": True, "rows": len(df)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/from-id")
async def load_from_id(req: BoardIdRequest):
    try:
        df = loader.load_from_board_id(req.board_id)
        loader.set_board(df)
        return {"ok": True, "rows": len(df)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/info")
async def board_info():
    df = loader.get_board()
    return loader.to_python(bomi.log_info(df))

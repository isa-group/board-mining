"""Health endpoints: board health summary and temporal evolution."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

import bomi
from .. import loader

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/summary")
async def health_summary():
    df = loader.get_board()
    return loader.to_python(bomi.board_health(df))


@router.get("/dimensions")
async def health_dimensions():
    df = loader.get_board()
    return loader.to_python(bomi.health_dimensions(df))


@router.get("/evolution")
async def health_evolution(
    window_days: int = Query(30, description="Lookback window in days"),
    step_days: int = Query(7, description="Step between reference dates in days"),
    indicators: Optional[str] = Query(None, description="Comma-separated indicator names, or omit for all"),
):
    df = loader.get_board()
    indicator_list = [i.strip() for i in indicators.split(",")] if indicators else None
    evolution = bomi.health_evolution(
        df,
        window=pd.Timedelta(f"{window_days}D"),
        step=pd.Timedelta(f"{step_days}D"),
        indicators=indicator_list,
    )
    return loader.df_to_records(evolution.reset_index())

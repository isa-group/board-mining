"""Structure endpoints: list lifecycle, redesign periods."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import bomi
from fastapi import APIRouter, Query
from bomi.schema import CARD_EVENT_TYPE, LIST_ID, LIST_NAME, TIMESTAMP, CARD_ID, CARD_NAME, TARGET_LIST_ID, TARGET_LIST_NAME
from .. import loader

router = APIRouter(prefix="/api/structure", tags=["structure"])


@router.get("/list-evolution")
async def list_evolution():
    df = loader.get_board()
    result = bomi.list_evolution(df)
    return loader.df_to_records(result.reset_index())


@router.get("/redesigns")
async def redesigns(
    threshold_days: int = Query(1, description="Time window in days for detecting redesigns"),
    l_type: Optional[str] = Query(None, description="Comma-separated list event types (list_create, list_rename, etc)"),
    threshold_l_events: int = Query(0, description="Minimum list events required to report redesign"),
):
    df = loader.get_board()

    # Parse l_type from comma-separated string
    l_type_list = [t.strip() for t in l_type.split(",")] if l_type else None

    result = bomi.detect_redesign(
        df,
        threshold=pd.Timedelta(f"{threshold_days}D"),
        l_type=l_type_list,
        threshold_l_events=threshold_l_events,
    )
    return loader.df_to_records(result)


@router.get("/events-by-list")
async def events_by_list(
    start_date: Optional[str] = Query(None, description="ISO 8601 start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="ISO 8601 end date (YYYY-MM-DD)"),
):
    df = loader.get_board()

    # Apply time window if provided (ensure timezone-aware comparison)
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        df = df[df[TIMESTAMP] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        df = df[df[TIMESTAMP] <= end_ts]

    # Filter to card events only
    card_events = df[df[CARD_EVENT_TYPE].isin(['card_create', 'card_act', 'card_move', 'card_close'])].copy()

    # For card_move events, use TARGET_LIST instead of LIST (which may be null)
    card_events[LIST_ID] = card_events[LIST_ID].fillna(card_events[TARGET_LIST_ID])
    card_events[LIST_NAME] = card_events[LIST_NAME].fillna(card_events[TARGET_LIST_NAME])

    # Filter out events where both LIST_NAME and TARGET_LIST_NAME are empty/null
    mask_valid = (card_events[LIST_NAME].notna() & (card_events[LIST_NAME] != '')) | \
                 (card_events[TARGET_LIST_NAME].notna() & (card_events[TARGET_LIST_NAME] != ''))
    card_events = card_events[mask_valid]

    # Return relevant columns
    result = card_events[[LIST_ID, LIST_NAME, TIMESTAMP, CARD_EVENT_TYPE, CARD_ID, CARD_NAME]]
    return loader.df_to_records(result)


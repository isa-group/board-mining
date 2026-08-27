"""Cards endpoints: per-card health indicators for drill-down."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query

import bomi
from bomi.schema import (
    CARD_ID, CARD_NAME, TIMESTAMP, TARGET_LIST_ID, TARGET_LIST_NAME,
    SOURCE_LIST_NAME, LIST_ID, CARD_EVENT_TYPE,
)
from .. import loader

router = APIRouter(prefix="/api/cards", tags=["cards"])


def _last_list_per_card(df: pd.DataFrame) -> pd.Series:
    """Return the most recent list_id each card was in."""
    moves = df[df[CARD_EVENT_TYPE] == "card_move"].sort_values(TIMESTAMP)
    last_move = moves.groupby(CARD_ID)[TARGET_LIST_ID].last()
    creates = df[df[CARD_EVENT_TYPE] == "card_create"].sort_values(TIMESTAMP)
    first_list = creates.groupby(CARD_ID)[LIST_ID].first()
    return last_move.combine_first(first_list)


def _cards_with_transition(df: pd.DataFrame, source: str, target: str) -> set:
    """Return card IDs that made at least one move from source to target list."""
    moves = df[df[CARD_EVENT_TYPE] == "card_move"]
    matched = moves[
        (moves[SOURCE_LIST_NAME] == source) &
        (moves[TARGET_LIST_NAME] == target)
    ]
    return set(matched[CARD_ID].dropna().unique())


def _build_records(df: pd.DataFrame, card_ids=None) -> list[dict]:
    """Compute per-card indicators and return as a list of dicts.

    Parameters
    ----------
    df:
        Full board event log.
    card_ids:
        Optional set of card IDs to restrict output to.
    """
    bouncing  = bomi.bouncing_cards(df)
    inactive  = bomi.inactive_cards(df)
    silent    = bomi.silent_moves(df)
    orphan    = bomi.orphan_cards(df)
    last_list = _last_list_per_card(df)

    # Get the most recent card name for each card
    last_card_name = df.sort_values(TIMESTAMP).groupby(CARD_ID)[CARD_NAME].last()

    overdue = None
    try:
        overdue = bomi.overdue_cards(df)
    except Exception:
        pass

    unassigned = None
    try:
        unassigned = bomi.unassigned_cards(df)
    except Exception:
        pass

    all_cards = (
        bouncing.index
        .union(inactive.index)
        .union(silent.index)
        .union(orphan.index)
    )

    if card_ids is not None:
        all_cards = all_cards.intersection(list(card_ids))

    records = []
    for card_id in all_cards:
        card_name = last_card_name.get(card_id)
        record: dict = {
            "card_id":      card_id,
            "card_name":    card_name,
            "bouncing":     int(bouncing.get(card_id, 0)),
            "inactive":     bool(inactive.get(card_id, False)),
            "silent_moves": int(silent.get(card_id, 0)),
            "orphan":       bool(orphan.get(card_id, False)),
            "last_list_id": last_list.get(card_id),
        }
        if overdue is not None:
            record["overdue"] = bool(overdue.get(card_id, False))
        if unassigned is not None:
            record["unassigned"] = bool(unassigned.get(card_id, False))
        records.append(record)

    return records


@router.get("/indicators")
async def card_indicators(
    source_list: Optional[str] = Query(None, description="Filter to cards that made this transition"),
    target_list: Optional[str] = Query(None, description="Filter to cards that made this transition"),
    list_id: Optional[str] = Query(None, description="Filter to cards in this list"),
):
    df = loader.get_board()

    card_ids = None
    if source_list and target_list:
        card_ids = _cards_with_transition(df, source_list, target_list)
    elif list_id:
        last_list = _last_list_per_card(df)
        card_ids = set(last_list[last_list == list_id].index)

    return _build_records(df, card_ids=card_ids)


@router.get("/timeline/{card_id}")
async def card_timeline(card_id: str):
    """Return the event history for a single card."""
    df = loader.get_board()
    card_df = df[df[CARD_ID] == card_id].sort_values(TIMESTAMP)
    return loader.df_to_records(card_df)

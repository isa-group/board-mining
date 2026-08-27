"""Trello connector: loads a Trello board action log as a bomi event log."""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..schema import (
    compute_event_types,
    to_board_log,
)

# Mapping from the flattened Trello JSON field names (as produced by
# pd.json_normalize) to the bomi board schema column names.
TRELLO_COLUMN_MAP: dict[str, str] = {
    "id": "event_id",
    "type": "event_type",
    "date": "timestamp",
    "idMemberCreator": "actor_id",
    "data.board.id": "board_id",
    "data.board.name": "board_name",
    "data.card.id": "card_id",
    "data.card.name": "card_name",
    "data.card.closed": "card_closed",
    "data.card.due": "card_due",
    "data.list.id": "list_id",
    "data.list.name": "list_name",
    "data.list.closed": "list_closed",
    "data.listBefore.id": "source_list_id",
    "data.listBefore.name": "source_list_name",
    "data.listAfter.id": "target_list_id",
    "data.listAfter.name": "target_list_name",
    "data.old.name": "old_name",
}


def load_trello_board(
    board_id: str,
    batch: int = 1000,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download a public Trello board action log and return a bomi event log.

    Parameters
    ----------
    board_id:
        The Trello board short ID or full ID (visible in the board URL).
    batch:
        Number of actions to request per API call.
    session:
        Optional :class:`requests.Session` for HTTP calls (useful for testing).
    """
    actions: list[dict[str, Any]] = []
    before: str | None = None
    requester = session or requests

    while True:
        before_param = f"&before={before}" if before else ""
        url = (
            f"https://api.trello.com/1/boards/{board_id}/actions"
            f"?limit={batch}{before_param}"
        )
        agent = "".join(random.choice(string.ascii_lowercase) for _ in range(16))
        response = requester.get(url, headers={"user-agent": agent})
        response.raise_for_status()

        data = response.json()
        actions.extend(data)
        if len(data) < batch:
            break
        before = data[batch - 1]["date"]

    df = pd.json_normalize(actions)
    return _normalize(df)


def read_trello_json(path: str | Path) -> pd.DataFrame:
    """Read Trello action JSON from a file and return a bomi event log.

    The file may be a plain list of action objects or a dict with an
    ``"actions"`` key (as exported by some Trello tools).

    Parameters
    ----------
    path:
        Path to the JSON file.
    """
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("actions", raw)
    df = pd.json_normalize(raw)
    return _normalize(df)


def from_trello_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw Trello action DataFrame to the bomi event log schema.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`pandas.json_normalize` on a list of
        Trello action objects.
    """
    return _normalize(df)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map Trello columns to the bomi schema and compute event types."""
    # Filter out createCard and updateCard events where both list names are NaN
    mask_create_or_update = df["type"].isin(["createCard", "updateCard"])

    # Check if columns exist; if not, treat as all NaN
    list_name_nan = df["data.list.name"].isna() if "data.list.name" in df else pd.Series([True] * len(df))
    listbefore_name_nan = df["data.listBefore.name"].isna() if "data.listBefore.name" in df else pd.Series([True] * len(df))

    mask_both_lists_nan = list_name_nan & listbefore_name_nan
    mask_remove = mask_create_or_update & mask_both_lists_nan

    df = df[~mask_remove].copy()

    board_log = to_board_log(
        df,
        column_map=TRELLO_COLUMN_MAP,
        source_system="trello",
        preserve_unmapped=False,
        validate=True,
    )
    return compute_event_types(board_log)

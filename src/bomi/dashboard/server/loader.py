"""In-memory board state shared across all API requests."""
from __future__ import annotations

import json
from io import StringIO

import numpy as np
import pandas as pd
from fastapi import HTTPException

from bomi.connectors.trello import load_trello_board
from bomi.io import from_dataframe

_board_df: pd.DataFrame | None = None


def get_board() -> pd.DataFrame:
    if _board_df is None:
        raise HTTPException(status_code=400, detail="No board loaded. Use /api/board/upload or /api/board/from-id first.")
    return _board_df


def set_board(df: pd.DataFrame) -> None:
    global _board_df
    _board_df = df


def load_from_csv_bytes(contents: bytes, filename: str) -> pd.DataFrame:
    text = contents.decode("utf-8")
    df = pd.read_csv(StringIO(text))
    return from_dataframe(df)


def load_from_json_bytes(contents: bytes) -> pd.DataFrame:
    actions = json.loads(contents.decode("utf-8"))
    from bomi.connectors.trello import actions_to_df
    return actions_to_df(actions)


def load_from_board_id(board_id: str, token: str | None = None) -> pd.DataFrame:
    session = None
    if token:
        from requests_oauthlib import OAuth1Session
        # placeholder — real OAuth session wired in oauth.py
        raise NotImplementedError("Authenticated Trello access not yet supported.")
    return load_trello_board(board_id, session=session)


def to_python(obj):
    """Recursively convert numpy/pandas scalars to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    return obj


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Serialize a DataFrame to JSON-safe records, converting timestamps to ISO strings."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def series_to_dict(s: pd.Series) -> dict:
    """Serialize a Series to a JSON-safe dict."""
    return json.loads(s.to_json(date_format="iso"))

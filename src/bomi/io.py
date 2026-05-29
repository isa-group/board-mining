"""Generic import / export helpers for bomi board event logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    CARD_EVENT_TYPE,
    EVENT_TYPE,
    LIST_EVENT_TYPE,
    SOURCE_SYSTEM,
    TIMESTAMP,
    to_board_log,
    validate_board_log,
)


def from_dataframe(
    df: pd.DataFrame,
    source_system: str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load a board event log from an existing pandas DataFrame.

    The DataFrame is expected to already use bomi board schema column names.
    Use :func:`bomi.connectors.trello.from_trello_dataframe` if the input uses
    raw Trello column names.

    Parameters
    ----------
    df:
        Source DataFrame in the bomi board schema.
    source_system:
        Value stored in the ``source_system`` provenance column if not already
        present.
    validate:
        Validate the DataFrame and raise :exc:`SchemaValidationError` on
        missing required fields.
    """
    return to_board_log(df, column_map={}, source_system=source_system, validate=validate)


def read_board_csv(
    path: str | Path,
    source_system: str | None = None,
    validate: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a board event log from a CSV file in the bomi board schema.

    Parameters
    ----------
    path:
        Path to the CSV file.
    source_system:
        Value stored in the ``source_system`` column if not already present.
    validate:
        Validate the result against the board schema.
    **read_csv_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    return from_dataframe(df, source_system=source_system, validate=validate)


def write_board_csv(
    df: pd.DataFrame,
    path: str | Path,
    validate: bool = True,
    **to_csv_kwargs: Any,
) -> None:
    """Write a board event log to a CSV file.

    Parameters
    ----------
    df:
        Board event log in the bomi board schema.
    path:
        Destination file path.
    validate:
        Validate *df* before writing.
    **to_csv_kwargs:
        Extra keyword arguments forwarded to :meth:`pandas.DataFrame.to_csv`.
    """
    if validate:
        validate_board_log(df, raise_on_error=True)
    df.to_csv(path, index=False, **to_csv_kwargs)


def to_process_dataframe(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Return a PM4Py-compatible DataFrame from a bomi event log.

    Cards become cases and list names become activity names.  Only events
    with a card and a timestamp are included.

    Parameters
    ----------
    df:
        Board event log in the bomi board schema.
    validate:
        Validate *df* before conversion.
    """
    board_log = to_board_log(df, column_map={}, validate=validate)
    empty = pd.Series(pd.NA, index=board_log.index)
    target_list = board_log.get("target_list_name", empty)
    list_name = board_log.get("list_name", empty)
    process_df = pd.DataFrame(
        {
            "case:concept:name": board_log["card_id"],
            "concept:name": target_list.fillna(list_name).fillna(board_log[EVENT_TYPE]),
            "time:timestamp": board_log[TIMESTAMP],
            "org:resource": board_log.get("actor_id", empty),
            "source_system": board_log.get(SOURCE_SYSTEM, empty),
            "raw_event_type": board_log.get("raw_event_type", empty),
        }
    )
    return process_df.dropna(subset=["case:concept:name", "time:timestamp"])

"""Board event-log schema for bomi.

The board schema defines a tool-independent representation for event logs
produced by board-based collaborative work management tools (BBTs) such as
Trello, Jira, and others.  Connectors (see :mod:`bomi.connectors`) are
responsible for mapping vendor-specific data to this schema before any
analysis is performed.

Core concepts
-------------
- **Board**: the top-level workspace (a Trello board, a Jira agile board, …).
- **List**: a named column or status within a board (a Trello list, a Jira
  status, …).
- **Card**: an individual work item (a Trello card, a Jira issue, …).
- **Event**: a timestamped record of an action taken on a board, list, or card.

Each row in a board event log represents one event.  The schema columns are
grouped into five categories described below.

Required columns
    The minimum set needed by all analysis functions.

Entity columns
    Who performed the action and on which board / list / card.

State columns
    The state of the entity at the time of the event (card movement,
    due dates, renames, …).

Classification columns
    Derived event categories that drive board-mining analysis.
    These are computed automatically by connectors via
    :func:`compute_event_types`.

Provenance columns
    The original vendor event type and the name of the source system.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

EVENT_ID = "event_id"
"""Unique identifier for this event within the log."""

EVENT_TYPE = "event_type"
"""Normalised event type.  Common values: ``createCard``, ``updateCard``,
``deleteCard``, ``createList``, ``updateList``."""

TIMESTAMP = "timestamp"
"""Date and time when the event occurred (timezone-aware datetime)."""


# ---------------------------------------------------------------------------
# Entity columns
# ---------------------------------------------------------------------------

ACTOR_ID = "actor_id"
"""Identifier of the user who performed the action."""

BOARD_ID = "board_id"
"""Identifier of the board on which the event occurred."""

BOARD_NAME = "board_name"
"""Human-readable name of the board."""

CARD_ID = "card_id"
"""Identifier of the card involved in this event, if any."""

CARD_NAME = "card_name"
"""Name or title of the card at the time of the event."""

LIST_ID = "list_id"
"""Identifier of the list the card was in at the time of the event.
For card-move events this is the *source* list (where the card came from)."""

LIST_NAME = "list_name"
"""Human-readable name of the list identified by :data:`LIST_ID`."""


# ---------------------------------------------------------------------------
# State columns
# ---------------------------------------------------------------------------

CARD_CLOSED = "card_closed"
"""Whether the card was in a closed / archived / done state at the time of
the event.  The exact interpretation depends on the connector; see the
``closed_method`` parameter of each connector for details."""

CARD_DUE = "card_due"
"""Due date of the card at the time of the event."""

LIST_CLOSED = "list_closed"
"""Whether the list was closed at the time of the event.  Not all connectors
populate this field (e.g. Jira has no list-closed concept)."""

SOURCE_LIST_ID = "source_list_id"
"""For card-move events: identifier of the list the card moved *from*."""

SOURCE_LIST_NAME = "source_list_name"
"""Human-readable name of the source list."""

TARGET_LIST_ID = "target_list_id"
"""For card-move events: identifier of the list the card moved *to*."""

TARGET_LIST_NAME = "target_list_name"
"""Human-readable name of the target list."""

OLD_NAME = "old_name"
"""Previous name of a card or list, populated on rename events."""


# ---------------------------------------------------------------------------
# Classification columns
# ---------------------------------------------------------------------------

LIST_EVENT_TYPE = "list_event_type"
"""High-level category of a list-level event.  Possible values:

- ``list_create`` — a new list was added to the board.
- ``list_import`` — a list was moved onto the board from another board.
- ``list_change`` — a list attribute was updated.
- ``list_rename`` — the list was renamed (a specific kind of change).
- ``list_move`` — the list was moved off the board.
- ``list_ends`` — the list was closed / archived.

``NaN`` for events that are not list-level (e.g. card actions or events from
connectors where list-level history is not available, such as Jira).
"""

CARD_EVENT_TYPE = "card_event_type"
"""High-level category of a card-level event.  Possible values:

- ``card_create`` — a new card was added to the board.
- ``card_move`` — the card was moved from one list to another.
- ``card_act`` — any other action on a card (update, comment, assignment, …).
- ``card_delete`` — the card was permanently deleted.
- ``card_close`` — the card was closed / archived / resolved.

``NaN`` for events that are not card-level (e.g. list-only events).
"""


# ---------------------------------------------------------------------------
# Provenance columns
# ---------------------------------------------------------------------------

RAW_EVENT_TYPE = "raw_event_type"
"""Original event type string as returned by the source system, preserved
for traceability.  Format is connector-specific (e.g. ``"updateCard"`` for
Trello, ``"status"`` for a Jira changelog item)."""

SOURCE_SYSTEM = "source_system"
"""Name of the source system that produced the log, e.g. ``"trello"`` or
``"jira"``."""


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = (EVENT_ID, EVENT_TYPE, TIMESTAMP)

ENTITY_COLUMNS = (
    ACTOR_ID,
    BOARD_ID,
    BOARD_NAME,
    CARD_ID,
    CARD_NAME,
    LIST_ID,
    LIST_NAME,
)

STATE_COLUMNS = (
    CARD_CLOSED,
    CARD_DUE,
    LIST_CLOSED,
    SOURCE_LIST_ID,
    SOURCE_LIST_NAME,
    TARGET_LIST_ID,
    TARGET_LIST_NAME,
    OLD_NAME,
)

CLASSIFICATION_COLUMNS = (LIST_EVENT_TYPE, CARD_EVENT_TYPE)

PROVENANCE_COLUMNS = (RAW_EVENT_TYPE, SOURCE_SYSTEM)

SCHEMA_COLUMNS = (
    *REQUIRED_COLUMNS,
    *ENTITY_COLUMNS,
    *STATE_COLUMNS,
    *CLASSIFICATION_COLUMNS,
    *PROVENANCE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaReport:
    """Validation result for a board event log."""

    is_valid: bool
    missing_required: tuple[str, ...]
    missing_recommended: tuple[str, ...]
    null_required: tuple[str, ...]
    invalid_timestamps: int


class SchemaValidationError(ValueError):
    """Raised when a board event log does not satisfy the schema."""


def schema_columns() -> tuple[str, ...]:
    """Return the ordered board schema column names."""
    return SCHEMA_COLUMNS


def validate_board_log(
    df: pd.DataFrame,
    recommended_columns: tuple[str, ...] = (ACTOR_ID, CARD_ID, LIST_ID),
    raise_on_error: bool = False,
) -> SchemaReport:
    """Validate a DataFrame against the bomi board event-log schema."""

    missing_required = tuple(col for col in REQUIRED_COLUMNS if col not in df.columns)
    missing_recommended = tuple(col for col in recommended_columns if col not in df.columns)
    null_required = tuple(
        col for col in REQUIRED_COLUMNS
        if col in df.columns and df[col].isna().any()
    )

    invalid_timestamps = 0
    if TIMESTAMP in df.columns:
        timestamps = pd.to_datetime(df[TIMESTAMP], errors="coerce")
        invalid_timestamps = int(timestamps.isna().sum())

    is_valid = not missing_required and not null_required and invalid_timestamps == 0

    report = SchemaReport(
        is_valid=is_valid,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        null_required=null_required,
        invalid_timestamps=invalid_timestamps,
    )

    if raise_on_error and not report.is_valid:
        problems = []
        if missing_required:
            problems.append(f"missing required columns: {', '.join(missing_required)}")
        if null_required:
            problems.append(f"null values in required columns: {', '.join(null_required)}")
        if invalid_timestamps:
            problems.append(f"invalid timestamps: {invalid_timestamps}")
        raise SchemaValidationError("; ".join(problems))

    return report


# ---------------------------------------------------------------------------
# Schema transformation
# ---------------------------------------------------------------------------

def to_board_log(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    source_system: str | None = None,
    preserve_unmapped: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Map a DataFrame to the bomi board event-log schema.

    Parameters
    ----------
    df:
        Source DataFrame.  If its columns are already in the bomi schema, pass
        ``column_map={}`` (or omit it) and only validation / normalisation will
        be applied.
    column_map:
        Mapping from source column names to schema column names.  When
        ``None`` (default) no column renaming is performed.
    source_system:
        Value stored in the :data:`SOURCE_SYSTEM` provenance column.  Only
        written when the column is not already present in *df*.
    preserve_unmapped:
        Keep source columns that are not part of the schema mapping.
    validate:
        Run :func:`validate_board_log` and raise :exc:`SchemaValidationError`
        on missing required fields.
    """
    column_map = {} if column_map is None else column_map
    mapped_columns = {src: dst for src, dst in column_map.items() if src in df.columns}
    result = df.rename(columns=mapped_columns).copy()

    if RAW_EVENT_TYPE not in result.columns and EVENT_TYPE in result.columns:
        result[RAW_EVENT_TYPE] = result[EVENT_TYPE]

    if source_system is not None and SOURCE_SYSTEM not in result.columns:
        result[SOURCE_SYSTEM] = source_system

    if TIMESTAMP in result.columns:
        result[TIMESTAMP] = pd.to_datetime(result[TIMESTAMP], errors="coerce")

    if not preserve_unmapped:
        columns = [col for col in SCHEMA_COLUMNS if col in result.columns]
        result = result.loc[:, columns]

    if validate:
        validate_board_log(result, raise_on_error=True)

    return result


# ---------------------------------------------------------------------------
# Event-type classification
# ---------------------------------------------------------------------------

def compute_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`LIST_EVENT_TYPE` and :data:`CARD_EVENT_TYPE` columns.

    Classifies each event row using only the schema columns already present in
    *df*, so the result is valid for any connector output.  Later assignments
    take priority (e.g. ``card_close`` overrides ``card_move``).

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with :data:`LIST_EVENT_TYPE` and
        :data:`CARD_EVENT_TYPE` columns added or replaced.
    """
    result = df.copy()

    # --- list_event_type ---------------------------------------------------
    result[LIST_EVENT_TYPE] = pd.NA

    if RAW_EVENT_TYPE in result.columns:
        raw = result[RAW_EVENT_TYPE]
        result.loc[raw == "createList", LIST_EVENT_TYPE] = "list_create"
        result.loc[raw == "moveListToBoard", LIST_EVENT_TYPE] = "list_import"
        result.loc[raw == "updateList", LIST_EVENT_TYPE] = "list_change"
        result.loc[raw == "moveListFromBoard", LIST_EVENT_TYPE] = "list_move"
        if OLD_NAME in result.columns:
            result.loc[
                (raw == "updateList") & result[OLD_NAME].notna(), LIST_EVENT_TYPE
            ] = "list_rename"

    if LIST_CLOSED in result.columns:
        list_closed = result[LIST_CLOSED].notna() & (result[LIST_CLOSED] == True)  # noqa: E712
        result.loc[list_closed, LIST_EVENT_TYPE] = "list_ends"

    # --- card_event_type ---------------------------------------------------
    result[CARD_EVENT_TYPE] = pd.NA

    if CARD_ID in result.columns:
        result.loc[result[CARD_ID].notna(), CARD_EVENT_TYPE] = "card_act"

    if EVENT_TYPE in result.columns:
        result.loc[result[EVENT_TYPE] == "createCard", CARD_EVENT_TYPE] = "card_create"
        result.loc[result[EVENT_TYPE] == "copyCard", CARD_EVENT_TYPE] = "card_create"
        result.loc[result[EVENT_TYPE] == "moveCardToBoard", CARD_EVENT_TYPE] = "card_create"

    if SOURCE_LIST_ID in result.columns:
        result.loc[result[SOURCE_LIST_ID].notna(), CARD_EVENT_TYPE] = "card_move"

    if RAW_EVENT_TYPE in result.columns:
        result.loc[result[RAW_EVENT_TYPE] == "deleteCard", CARD_EVENT_TYPE] = "card_delete"

    if CARD_CLOSED in result.columns:
        card_closed = result[CARD_CLOSED].notna() & (result[CARD_CLOSED] == True)  # noqa: E712
        result.loc[card_closed, CARD_EVENT_TYPE] = "card_close"

    return result

"""Board health and quality indicators for bomi.

Functions are organised in four layers:

1. **Completion helper** — :func:`card_closed_mask` determines which cards are
   considered "done" under a configurable method.

2. **Per-card indicators** — return a :class:`pandas.Series` indexed by
   ``card_id``: :func:`card_age`, :func:`inactive_cards`,
   :func:`orphan_cards`, :func:`overdue_cards`, :func:`bouncing_cards`,
   :func:`silent_moves`, :func:`unassigned_cards`.

3. **Per-list indicators** — return a :class:`pandas.Series` indexed by
   ``list_id``: :func:`stagnant_lists`, :func:`dead_lists`.

4. **Board-level scalars and aggregators** — :func:`flow_conformance`,
   :func:`completion_rate`, :func:`abandonment_rate`,
   :func:`health_dimensions`, :func:`board_health`.

All functions that deal with open cards share a consistent parameter order:
``method``, ``sink_lists``, ``reference_date``.  When ``reference_date`` is
provided, it is used both as the measurement anchor *and* to determine which
cards were open at that point in time — a card completed after
``reference_date`` is treated as open.  When omitted, it defaults to the last
timestamp in the log (correct for retrospective batch analysis).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd

from .schema import (
    ACTOR_ID,
    CARD_CLOSED,
    CARD_DUE,
    CARD_EVENT_TYPE,
    CARD_ID,
    EVENT_ID,
    LIST_ID,
    LIST_NAME,
    RAW_EVENT_TYPE,
    SOURCE_LIST_ID,
    SOURCE_LIST_NAME,
    TARGET_LIST_ID,
    TARGET_LIST_NAME,
    TIMESTAMP,
)

__all__ = [
    "card_closed_mask",
    "card_age",
    "inactive_cards",
    "orphan_cards",
    "overdue_cards",
    "bouncing_cards",
    "silent_moves",
    "unassigned_cards",
    "stagnant_lists",
    "dead_lists",
    "flow_conformance",
    "completion_rate",
    "abandonment_rate",
    "health_dimensions",
    "board_health",
    "health_evolution",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reference(df: pd.DataFrame, reference_date) -> pd.Timestamp:
    return df[TIMESTAMP].max() if reference_date is None else pd.Timestamp(reference_date)


def _open_cards(df: pd.DataFrame, method, sink_lists, reference_date=None) -> pd.Index:
    """Return IDs of cards that are open (not completed) at *reference_date*."""
    completed = card_closed_mask(df, method=method, sink_lists=sink_lists,
                                 reference_date=reference_date)
    return completed[~completed].index


def _current_list_per_card(df: pd.DataFrame) -> pd.Series:
    """Return the most-recent list_id for every card in *df*."""
    last_move_target = (
        df[df[CARD_EVENT_TYPE] == "card_move"]
        .dropna(subset=[CARD_ID, TARGET_LIST_ID])
        .sort_values(TIMESTAMP)
        .groupby(CARD_ID)[TARGET_LIST_ID]
        .last()
    )
    create_list = (
        df[df[CARD_EVENT_TYPE] == "card_create"]
        .dropna(subset=[CARD_ID, LIST_ID])
        .sort_values(TIMESTAMP)
        .groupby(CARD_ID)[LIST_ID]
        .first()
    )
    return last_move_target.combine_first(create_list)


# ---------------------------------------------------------------------------
# Completion helper
# ---------------------------------------------------------------------------

def card_closed_mask(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return a boolean Series indexed by card_id: True if card is completed.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    method:
        How to determine completion.  One of, or a list combining:

        - ``"archived"`` — card ever had ``card_closed == True``.
        - ``"sink_list"`` — card's most-recent list is in *sink_lists*.
        - ``"deleted"`` — card had a ``card_delete`` event.

        Multiple methods are combined with a logical OR.
    sink_lists:
        List names to treat as sink lists when ``method`` includes
        ``"sink_list"``.
    reference_date:
        Completion events after this date are ignored — a card completed after
        ``reference_date`` is considered open at that point in time.  Defaults
        to ``None`` (use all events).
    """
    methods = [method] if isinstance(method, str) else list(method)

    # Only consider events up to reference_date for completion determination.
    ref = pd.Timestamp(reference_date) if reference_date is not None else None
    df_ref = df[df[TIMESTAMP] <= ref] if ref is not None else df

    all_cards = df_ref[CARD_ID].dropna().unique()
    result = pd.Series(False, index=pd.Index(all_cards, name=CARD_ID), name="completed")

    if "archived" in methods and CARD_CLOSED in df_ref.columns:
        archived = (
            df_ref[df_ref[CARD_CLOSED].eq(True)][CARD_ID]
            .dropna()
            .unique()
        )
        result.loc[result.index.isin(archived)] = True

    if "sink_list" in methods and sink_lists:
        last_move_name = pd.Series(dtype=str)
        if TARGET_LIST_NAME in df_ref.columns:
            last_move_name = (
                df_ref[df_ref[CARD_EVENT_TYPE] == "card_move"]
                .dropna(subset=[CARD_ID, TARGET_LIST_NAME])
                .sort_values(TIMESTAMP)
                .groupby(CARD_ID)[TARGET_LIST_NAME]
                .last()
            )
        create_name = pd.Series(dtype=str)
        if LIST_NAME in df_ref.columns:
            create_name = (
                df_ref[df_ref[CARD_EVENT_TYPE] == "card_create"]
                .dropna(subset=[CARD_ID, LIST_NAME])
                .sort_values(TIMESTAMP)
                .groupby(CARD_ID)[LIST_NAME]
                .first()
            )
        last_list_name = last_move_name.combine_first(create_name)
        sink_cards = last_list_name[last_list_name.isin(sink_lists)].index
        result.loc[result.index.isin(sink_cards)] = True

    if "deleted" in methods and CARD_EVENT_TYPE in df_ref.columns:
        deleted = df_ref[df_ref[CARD_EVENT_TYPE] == "card_delete"][CARD_ID].dropna().unique()
        result.loc[result.index.isin(deleted)] = True

    return result


# ---------------------------------------------------------------------------
# Per-card indicators
# ---------------------------------------------------------------------------

def card_age(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return the age of each open card as a Timedelta.

    Age is measured from the card's last ``card_move`` or ``card_create``
    event up to *reference_date*.  Cards completed at or before *reference_date*
    are excluded; cards completed after it are treated as open.

    Returns
    -------
    pd.Series
        :class:`pandas.Timedelta` Series indexed by ``card_id``.
    """
    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    relevant = df[
        df[CARD_EVENT_TYPE].isin(["card_move", "card_create"])
        & df[CARD_ID].isin(open_c)
        & (df[TIMESTAMP] <= ref)
    ]
    last_event = relevant.groupby(CARD_ID)[TIMESTAMP].max()
    return (ref - last_event).rename("card_age")


def inactive_cards(
    df: pd.DataFrame,
    window: pd.Timedelta = pd.Timedelta("30D"),
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return a boolean mask of open cards with no recent movement or action.

    A card is inactive if it had no ``card_move`` or ``card_act`` event within
    *window* before *reference_date*.  Cards completed at or before
    *reference_date* are excluded.

    Returns
    -------
    pd.Series
        bool Series indexed by ``card_id``; ``True`` means inactive.
    """
    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)
    cutoff = ref - window

    recently_active = (
        df[
            df[CARD_EVENT_TYPE].isin(["card_move", "card_act"])
            & df[CARD_ID].isin(open_c)
            & (df[TIMESTAMP] >= cutoff)
            & (df[TIMESTAMP] <= ref)
        ][CARD_ID]
        .dropna()
        .unique()
    )

    result = pd.Series(True, index=open_c, name="inactive")
    result.loc[result.index.isin(recently_active)] = False
    return result


def orphan_cards(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return a boolean mask of open cards that were never moved or acted on.

    An orphan card has only a ``card_create`` event and no subsequent
    ``card_move`` or ``card_act`` events up to *reference_date*.  Cards
    completed at or before *reference_date* are excluded.

    Returns
    -------
    pd.Series
        bool Series indexed by ``card_id``; ``True`` means orphan.
    """
    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    has_activity = (
        df[
            df[CARD_EVENT_TYPE].isin(["card_move", "card_act"])
            & df[CARD_ID].isin(open_c)
            & (df[TIMESTAMP] <= ref)
        ][CARD_ID]
        .dropna()
        .unique()
    )

    created = (
        df[
            (df[CARD_EVENT_TYPE] == "card_create")
            & (df[TIMESTAMP] <= ref)
        ][CARD_ID].dropna().unique()
    )
    created_open = pd.Index(created, name=CARD_ID).intersection(open_c)

    result = pd.Series(True, index=created_open, name="orphan")
    result.loc[result.index.isin(has_activity)] = False
    return result


def overdue_cards(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return a boolean mask of open cards past their due date.

    Requires the ``card_due`` column; returns an empty Series with a warning
    when the column is absent.  Cards completed at or before *reference_date*
    are excluded.

    Returns
    -------
    pd.Series
        bool Series indexed by ``card_id``; ``True`` means overdue.
    """
    if CARD_DUE not in df.columns:
        warnings.warn(
            "overdue_cards: 'card_due' column not present; returning empty Series",
            stacklevel=2,
        )
        return pd.Series(dtype=bool, name="overdue")

    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    due_dates = (
        df[
            df[CARD_ID].isin(open_c)
            & df[CARD_DUE].notna()
            & (df[TIMESTAMP] <= ref)
        ]
        .sort_values(TIMESTAMP)
        .groupby(CARD_ID)[CARD_DUE]
        .last()
    )
    due_dates = pd.to_datetime(due_dates, utc=True, errors="coerce")
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    return (due_dates < ref).rename("overdue")


def bouncing_cards(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return the number of list re-entries per open card.

    A re-entry (bounce) occurs when a card moves into a list it previously
    left.  Only open cards at *reference_date* are included; only moves up to
    *reference_date* are counted.

    Returns
    -------
    pd.Series
        int Series indexed by ``card_id``; 0 means no bouncing.
    """
    required = [CARD_ID, TIMESTAMP, SOURCE_LIST_ID, TARGET_LIST_ID]
    if not all(c in df.columns for c in required):
        return pd.Series(dtype=int, name="bounces")

    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    moves = (
        df[
            (df[CARD_EVENT_TYPE] == "card_move")
            & df[CARD_ID].isin(open_c)
            & (df[TIMESTAMP] <= ref)
        ]
        .dropna(subset=[CARD_ID, SOURCE_LIST_ID, TARGET_LIST_ID])
        .sort_values([CARD_ID, TIMESTAMP])
    )

    if moves.empty:
        return pd.Series(dtype=int, name="bounces")

    result: dict = {}
    for card_id, group in moves.groupby(CARD_ID):
        exited: set = set()
        count = 0
        for src, tgt in zip(group[SOURCE_LIST_ID], group[TARGET_LIST_ID]):
            exited.add(src)
            if tgt in exited:
                count += 1
        result[card_id] = count

    return pd.Series(result, name="bounces")


def silent_moves(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return the count of silent list-stays per open card.

    A *stay* is the period a card occupies a list, from its arrival (via
    creation or a move) to its departure (via the next move).  A stay is
    *silent* when no ``card_act`` event occurred during it.  Only open cards
    at *reference_date* are included; only events up to *reference_date* are
    counted.

    Returns
    -------
    pd.Series
        int Series indexed by ``card_id``; 0 means every stay had at least
        one ``card_act`` event.
    """
    if CARD_EVENT_TYPE not in df.columns or CARD_ID not in df.columns:
        return pd.Series(dtype=int, name="silent_moves")

    ref = _reference(df, reference_date)
    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    card_events = (
        df[
            df[CARD_EVENT_TYPE].notna()
            & df[CARD_ID].isin(open_c)
            & (df[TIMESTAMP] <= ref)
        ]
        [[CARD_ID, TIMESTAMP, CARD_EVENT_TYPE]]
        .sort_values([CARD_ID, TIMESTAMP])
    )

    if card_events.empty:
        return pd.Series(dtype=int, name="silent_moves")

    result: dict = {}
    for card_id, group in card_events.groupby(CARD_ID):
        in_stay = False
        stay_had_act = False
        silent_count = 0

        for etype in group[CARD_EVENT_TYPE]:
            if etype == "card_create":
                in_stay = True
                stay_had_act = False
            elif etype == "card_move":
                if in_stay and not stay_had_act:
                    silent_count += 1
                in_stay = True
                stay_had_act = False
            elif etype == "card_act":
                stay_had_act = True

        # Count the final open stay
        if in_stay and not stay_had_act:
            silent_count += 1

        result[card_id] = silent_count

    return pd.Series(result, name="silent_moves")


def unassigned_cards(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series | None:
    """Return a boolean mask of open cards with no current member assignment.

    Tracks Trello ``addMemberToCard`` / ``removeMemberFromCard`` raw events.
    A card is considered assigned when its net count of add-minus-remove events
    is positive at *reference_date*.  Cards completed at or before
    *reference_date* are excluded.

    Returns ``None`` when assignment events are absent from the log.

    Returns
    -------
    pd.Series or None
        bool Series indexed by ``card_id``; ``True`` means unassigned.
    """
    if RAW_EVENT_TYPE not in df.columns:
        return None

    ref = _reference(df, reference_date)
    mask_time = df[TIMESTAMP] <= ref

    add_events = df[(df[RAW_EVENT_TYPE] == "addMemberToCard") & mask_time]
    remove_events = df[(df[RAW_EVENT_TYPE] == "removeMemberFromCard") & mask_time]

    if add_events.empty and remove_events.empty:
        return None

    open_c = _open_cards(df, method, sink_lists, reference_date=ref)

    add_counts = add_events.dropna(subset=[CARD_ID]).groupby(CARD_ID)[EVENT_ID].count()
    if remove_events.empty:
        remove_counts = pd.Series(dtype=int)
    else:
        remove_counts = (
            remove_events.dropna(subset=[CARD_ID]).groupby(CARD_ID)[EVENT_ID].count()
        )

    net = add_counts.subtract(remove_counts, fill_value=0)
    assigned_cards = net[net > 0].index

    result = pd.Series(True, index=open_c, name="unassigned")
    result.loc[result.index.isin(assigned_cards)] = False
    return result


# ---------------------------------------------------------------------------
# Per-list indicators
# ---------------------------------------------------------------------------

def stagnant_lists(
    df: pd.DataFrame,
    window: pd.Timedelta,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> pd.Series:
    """Return the count of inactive open cards in each list.

    A list is stagnant when it accumulates open cards that have had no
    ``card_move`` or ``card_act`` event within *window*.

    Returns
    -------
    pd.Series
        int Series indexed by ``list_id``; lists with no inactive cards
        are omitted.
    """
    ref = _reference(df, reference_date)
    inactive = inactive_cards(df, window, method=method, sink_lists=sink_lists,
                              reference_date=ref)
    inactive_ids = inactive[inactive].index

    if inactive_ids.empty:
        return pd.Series(dtype=int, name="stagnant_cards")

    current_list = _current_list_per_card(df)
    current_list = current_list[current_list.index.isin(inactive_ids)]

    return current_list.value_counts().rename("stagnant_cards")


def dead_lists(
    df: pd.DataFrame,
    window: pd.Timedelta,
    reference_date=None,
) -> pd.Series:
    """Return a boolean mask of lists that received no card within *window*.

    A list "receives" a card when a card is created in it (``card_create``)
    or moved into it (``card_move`` with matching ``target_list_id``).

    Returns
    -------
    pd.Series
        bool Series indexed by ``list_id``; ``True`` means no card arrived
        within *window*.
    """
    ref = _reference(df, reference_date)
    cutoff = ref - window
    recent = df[(df[TIMESTAMP] >= cutoff) & (df[TIMESTAMP] <= ref)]

    created_in = (
        recent[recent[CARD_EVENT_TYPE] == "card_create"]
        .dropna(subset=[LIST_ID])[LIST_ID]
    )
    moved_into = (
        recent[recent[CARD_EVENT_TYPE] == "card_move"]
        .dropna(subset=[TARGET_LIST_ID])[TARGET_LIST_ID]
    )
    recently_active_lists = pd.concat([created_in, moved_into]).unique()

    all_lists = df[LIST_ID].dropna().unique()
    result = pd.Series(True, index=pd.Index(all_lists, name=LIST_ID), name="is_dead")
    result.loc[result.index.isin(recently_active_lists)] = False
    return result


# ---------------------------------------------------------------------------
# Board-level scalars
# ---------------------------------------------------------------------------

def flow_conformance(
    df: pd.DataFrame,
    prescribed_flow: list[tuple[str, str]] | None = None,
    infer_threshold: float = 0.05,
) -> float:
    """Return the fraction of card moves that conform to the prescribed flow.

    The prescribed flow is a directed graph represented as a list of
    ``(source_list_name, target_list_name)`` pairs.  A move is conformant
    when its (source, target) pair is in that set.

    When *prescribed_flow* is ``None``, the dominant flow is inferred from the
    data: all (source, target) pairs whose share of total moves is at least
    *infer_threshold* are treated as valid transitions.  The inferred flow is
    directed — ``A→B`` and ``B→A`` are independent edges.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    prescribed_flow:
        List of ``(source_list_name, target_list_name)`` pairs that define
        valid transitions.  When ``None``, inferred from the data.
    infer_threshold:
        Minimum fraction of total moves required for an inferred edge to be
        considered part of the dominant flow.  Ignored when *prescribed_flow*
        is provided.

    Returns
    -------
    float
        Value in [0, 1]; 0.0 when there are no card-move events.
    """
    required = [SOURCE_LIST_NAME, TARGET_LIST_NAME]
    moves = df[df[CARD_EVENT_TYPE] == "card_move"].dropna(subset=required)

    if moves.empty:
        return 0.0

    if prescribed_flow is None:
        total = len(moves)
        counts = moves.groupby([SOURCE_LIST_NAME, TARGET_LIST_NAME])[CARD_ID].count()
        prescribed_flow = [pair for pair, cnt in counts.items() if cnt / total >= infer_threshold]

    valid = set(prescribed_flow)
    pairs = list(zip(moves[SOURCE_LIST_NAME], moves[TARGET_LIST_NAME]))
    conformant = sum(1 for p in pairs if p in valid)
    return conformant / len(pairs)


def completion_rate(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> float:
    """Return the fraction of all cards that are completed at *reference_date*.

    Returns
    -------
    float
        Value in [0, 1]; 0.0 when the log contains no cards.
    """
    if CARD_ID not in df.columns:
        return 0.0
    ref = _reference(df, reference_date)
    mask = card_closed_mask(df, method=method, sink_lists=sink_lists, reference_date=ref)
    total = len(mask)
    return int(mask.sum()) / total if total else 0.0


def abandonment_rate(
    df: pd.DataFrame,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    reference_date=None,
) -> float:
    """Return the fraction of all cards that are orphan (never activated).

    Abandoned cards are open orphans: they were created but never moved or
    acted upon, and are not considered completed at *reference_date*.

    Returns
    -------
    float
        Value in [0, 1]; 0.0 when the log contains no cards.
    """
    if CARD_ID not in df.columns:
        return 0.0
    ref = _reference(df, reference_date)
    mask = card_closed_mask(df, method=method, sink_lists=sink_lists, reference_date=ref)
    total = len(mask)
    if total == 0:
        return 0.0
    orphans = orphan_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    return int(orphans.sum()) / total


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def health_dimensions(
    df: pd.DataFrame,
    inactive_window: pd.Timedelta = pd.Timedelta("30D"),
    dead_list_window: pd.Timedelta = pd.Timedelta("30D"),
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    prescribed_flow: list[tuple[str, str]] | None = None,
    infer_threshold: float = 0.05,
    redesigns: pd.DataFrame | None = None,
    reference_date=None,
) -> dict:
    """Compute the five board health dimension scores (each in [0, 1]).

    Dimensions
    ----------
    flow_discipline:
        Combines flow conformance, absence of bouncing, and proportion of
        cards with only silent stays.
    collaboration_discipline:
        Captures orphan-card rate and, when assignment data are available,
        unassigned-card rate.
    completion_discipline:
        Combines completion rate and inverse abandonment rate.
    structural_stability:
        Captures dead-list rate and, when *redesigns* are provided, the
        board redesign frequency.
    board_vitality:
        Captures inactive-card rate and stagnant-list rate.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    inactive_window:
        Time window used for :func:`inactive_cards` and :func:`stagnant_lists`.
    dead_list_window:
        Time window used for :func:`dead_lists`.
    method:
        Completion method forwarded to all per-card indicators.
    sink_lists:
        Sink list names for the ``"sink_list"`` completion method.
    prescribed_flow:
        Valid (source, target) list-name pairs for :func:`flow_conformance`.
        ``None`` infers from data.
    infer_threshold:
        Infer-threshold forwarded to :func:`flow_conformance`.
    redesigns:
        Optional output of :func:`bomi.detect_redesign`; enables the redesign
        frequency component of *structural_stability*.
    reference_date:
        Reference point for all indicators.  Defaults to the last timestamp
        in *df*.

    Returns
    -------
    dict
        Keys: ``flow_discipline``, ``collaboration_discipline``,
        ``completion_discipline``, ``structural_stability``,
        ``board_vitality``.
    """
    ref = _reference(df, reference_date)
    total_lists = df[LIST_ID].dropna().nunique() if LIST_ID in df.columns else 0

    # Open-card count for rates denominated over open cards
    completed_mask = card_closed_mask(df, method=method, sink_lists=sink_lists,
                                      reference_date=ref)
    total_open = int((~completed_mask).sum())

    # ---- flow_discipline ----
    fc = flow_conformance(df, prescribed_flow=prescribed_flow, infer_threshold=infer_threshold)
    bounces = bouncing_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    bouncing_rate = (bounces > 0).sum() / total_open if total_open else 0.0
    sm = silent_moves(df, method=method, sink_lists=sink_lists, reference_date=ref)
    silent_rate = (sm > 0).sum() / total_open if total_open else 0.0
    flow_d = (fc + (1 - bouncing_rate) + (1 - silent_rate)) / 3

    # ---- collaboration_discipline ----
    orphan_rate = abandonment_rate(df, method=method, sink_lists=sink_lists,
                                   reference_date=ref)
    collab_components = [1 - orphan_rate]
    ua = unassigned_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    if ua is not None and total_open:
        collab_components.append(1 - ua.sum() / total_open)
    collab_d = sum(collab_components) / len(collab_components)

    # ---- completion_discipline ----
    comp = completion_rate(df, method=method, sink_lists=sink_lists, reference_date=ref)
    aband = abandonment_rate(df, method=method, sink_lists=sink_lists, reference_date=ref)
    completion_d = (comp + (1 - aband)) / 2

    # ---- structural_stability ----
    dead = dead_lists(df, dead_list_window, reference_date=ref)
    dead_rate = dead.sum() / total_lists if total_lists else 0.0
    stability_components = [1 - dead_rate]
    if redesigns is not None and not redesigns.empty:
        board_weeks = (df[TIMESTAMP].max() - df[TIMESTAMP].min()) / pd.Timedelta("7D")
        redesign_rate = min(len(redesigns) / board_weeks, 1.0) if board_weeks > 0 else 1.0
        stability_components.append(1 - redesign_rate)
    stability_d = sum(stability_components) / len(stability_components)

    # ---- board_vitality ----
    inactive = inactive_cards(df, inactive_window, method=method, sink_lists=sink_lists,
                              reference_date=ref)
    inactive_rate = inactive.sum() / total_open if total_open else 0.0
    stagnant = stagnant_lists(df, inactive_window, method=method, sink_lists=sink_lists,
                              reference_date=ref)
    stagnant_list_rate = (stagnant > 0).sum() / total_lists if total_lists else 0.0
    vitality_d = ((1 - inactive_rate) + (1 - stagnant_list_rate)) / 2

    return {
        "flow_discipline": float(flow_d),
        "collaboration_discipline": float(collab_d),
        "completion_discipline": float(completion_d),
        "structural_stability": float(stability_d),
        "board_vitality": float(vitality_d),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def board_health(
    df: pd.DataFrame,
    inactive_window: pd.Timedelta = pd.Timedelta("30D"),
    dead_list_window: pd.Timedelta = pd.Timedelta("30D"),
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    prescribed_flow: list[tuple[str, str]] | None = None,
    infer_threshold: float = 0.05,
    redesigns: pd.DataFrame | None = None,
    reference_date=None,
) -> dict:
    """Return a complete board health report.

    Combines all raw per-card and per-list indicator rates with the five
    dimension scores from :func:`health_dimensions`.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    inactive_window:
        Time window for inactivity and stagnation detection.
    dead_list_window:
        Time window for dead-list detection.
    method:
        Completion method; see :func:`card_closed_mask`.
    sink_lists:
        Sink list names for the ``"sink_list"`` completion method.
    prescribed_flow:
        Valid (source, target) list-name pairs for flow conformance.
    infer_threshold:
        Infer-threshold for flow conformance when *prescribed_flow* is ``None``.
    redesigns:
        Optional output of :func:`bomi.detect_redesign`.
    reference_date:
        Reference point for all indicators.

    Returns
    -------
    dict
        Scalar rates and counts (``inactive_card_rate``, ``orphan_rate``,
        ``overdue_rate``, ``bouncing_rate``, ``silent_move_rate``,
        ``dead_list_rate``, ``stagnant_list_rate``, ``flow_conformance``,
        ``completion_rate``, ``abandonment_rate``, card-age statistics,
        raw counts) plus five dimension scores prefixed with ``dim_``.
    """
    ref = _reference(df, reference_date)
    total_lists = df[LIST_ID].dropna().nunique() if LIST_ID in df.columns else 0

    # Compute open/total counts once; all sub-functions reuse the same ref.
    completed_mask = card_closed_mask(df, method=method, sink_lists=sink_lists,
                                      reference_date=ref)
    total_cards = len(completed_mask)
    total_open = int((~completed_mask).sum())

    age      = card_age(df, method=method, sink_lists=sink_lists, reference_date=ref)
    inactive = inactive_cards(df, inactive_window, method=method, sink_lists=sink_lists,
                              reference_date=ref)
    orphans  = orphan_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    overdue  = overdue_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    bounces  = bouncing_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    sm       = silent_moves(df, method=method, sink_lists=sink_lists, reference_date=ref)
    ua       = unassigned_cards(df, method=method, sink_lists=sink_lists, reference_date=ref)
    stagnant = stagnant_lists(df, inactive_window, method=method, sink_lists=sink_lists,
                              reference_date=ref)
    dead     = dead_lists(df, dead_list_window, reference_date=ref)
    fc       = flow_conformance(df, prescribed_flow=prescribed_flow,
                                infer_threshold=infer_threshold)
    comp     = completion_rate(df, method=method, sink_lists=sink_lists, reference_date=ref)
    aband    = abandonment_rate(df, method=method, sink_lists=sink_lists, reference_date=ref)

    dims = health_dimensions(
        df,
        inactive_window=inactive_window,
        dead_list_window=dead_list_window,
        method=method,
        sink_lists=sink_lists,
        prescribed_flow=prescribed_flow,
        infer_threshold=infer_threshold,
        redesigns=redesigns,
        reference_date=ref,
    )

    report: dict = {
        "cards_total": total_cards,
        "cards_open": total_open,
        "lists_total": total_lists,
        # rates over open cards
        "inactive_card_rate": float(inactive.sum() / total_open) if total_open else 0.0,
        "bouncing_rate": float((bounces > 0).sum() / total_open) if total_open else 0.0,
        "silent_move_rate": float((sm > 0).sum() / total_open) if total_open else 0.0,
        # rates over all cards (lifecycle metrics)
        "orphan_rate": float(len(orphans[orphans]) / total_cards) if total_cards else 0.0,
        "overdue_rate": float(overdue.sum() / len(overdue)) if len(overdue) else 0.0,
        "completion_rate": comp,
        "abandonment_rate": aband,
        # list rates
        "dead_list_rate": float(dead.sum() / total_lists) if total_lists else 0.0,
        "stagnant_list_rate": float((stagnant > 0).sum() / total_lists) if total_lists else 0.0,
        # flow
        "flow_conformance": fc,
        # card age
        "card_age_mean_days": float(age.mean() / pd.Timedelta("1D")) if len(age) else None,
        "card_age_max_days": float(age.max() / pd.Timedelta("1D")) if len(age) else None,
        # raw counts
        "cards_inactive": int(inactive.sum()),
        "cards_orphan": int(orphans.sum()),
        "cards_overdue": int(overdue.sum()),
        "cards_bouncing": int((bounces > 0).sum()),
        "lists_dead": int(dead.sum()),
        "lists_stagnant": int((stagnant > 0).sum()),
        # dimensions
        **{f"dim_{k}": v for k, v in dims.items()},
    }

    if ua is not None:
        report["unassigned_rate"] = float(ua.sum() / total_open) if total_open else 0.0
        report["cards_unassigned"] = int(ua.sum())

    return report


# ---------------------------------------------------------------------------
# Temporal evolution
# ---------------------------------------------------------------------------

def health_evolution(
    df: pd.DataFrame,
    window: pd.Timedelta = pd.Timedelta("30D"),
    step: pd.Timedelta = pd.Timedelta("7D"),
    indicators: list[str] | None = None,
    start_date=None,
    method: str | Sequence[str] = "archived",
    sink_lists: list[str] | None = None,
    prescribed_flow: list[tuple[str, str]] | None = None,
    infer_threshold: float = 0.05,
    dead_list_window: pd.Timedelta | None = None,
    redesigns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute health indicators over time using a sliding window.

    At each reference date *t*, the board state is assessed using all events
    that occurred up to and including *t*.  Time-based indicators
    (inactivity, dead lists, stagnation) use *window* as the lookback period
    relative to *t*.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    window:
        Width of the sliding lookback window, used as ``inactive_window`` and
        ``dead_list_window`` (unless *dead_list_window* is given explicitly).
        Also determines the default *start_date*: the first reference date is
        ``df[TIMESTAMP].min() + window``, ensuring a full window of history
        is available at the first measurement.
    step:
        Interval between consecutive reference dates.
    indicators:
        Subset of :func:`board_health` keys to include in the result.
        ``None`` (default) includes all keys.  Pass a list of strings to
        track specific metrics, e.g.
        ``["dim_flow_discipline", "completion_rate"]``.
    start_date:
        First reference date.  Defaults to ``df[TIMESTAMP].min() + window``.
    method:
        Completion method forwarded to all per-card indicators.
    sink_lists:
        Sink list names for the ``"sink_list"`` completion method.
    prescribed_flow:
        Valid (source, target) list-name pairs for :func:`flow_conformance`.
        **Recommended** when tracking flow conformance over time: if ``None``,
        the dominant flow is re-inferred from each slice independently, making
        the trend less interpretable.
    infer_threshold:
        Infer-threshold for flow conformance when *prescribed_flow* is ``None``.
    dead_list_window:
        Lookback window for :func:`dead_lists`.  Defaults to *window*.
    redesigns:
        Optional output of :func:`bomi.detect_redesign`.  Only redesign
        periods whose start (``"min"`` column) is ≤ the current reference
        date are included in each slice's computation.

    Returns
    -------
    pd.DataFrame
        One row per reference date, indexed by timestamp.  Columns are the
        keys of :func:`board_health` (or the subset given by *indicators*).
        Returns an empty DataFrame when the log is too short to produce even
        one reference date.
    """
    if dead_list_window is None:
        dead_list_window = window

    t_end = df[TIMESTAMP].max()
    ref_start = (
        df[TIMESTAMP].min() + window
        if start_date is None
        else pd.Timestamp(start_date)
    )

    if ref_start > t_end:
        return pd.DataFrame()

    ref_dates: list[pd.Timestamp] = []
    ref = ref_start
    while ref <= t_end:
        ref_dates.append(ref)
        ref = ref + step

    rows = []
    for ref_date in ref_dates:
        df_slice = df[df[TIMESTAMP] <= ref_date]
        if df_slice.empty:
            continue

        redesigns_slice = (
            redesigns[redesigns["min"] <= ref_date]
            if redesigns is not None
            else None
        )

        health = board_health(
            df_slice,
            inactive_window=window,
            dead_list_window=dead_list_window,
            method=method,
            sink_lists=sink_lists,
            prescribed_flow=prescribed_flow,
            infer_threshold=infer_threshold,
            redesigns=redesigns_slice,
            reference_date=ref_date,
        )

        if indicators is not None:
            health = {k: health[k] for k in indicators if k in health}

        rows.append(health)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(ref_dates[: len(rows)], name=TIMESTAMP))
    return result

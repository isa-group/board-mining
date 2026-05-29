"""Conformance checking for bomi board event logs.

Two families of checks are provided:

- **Flow conformance** (:func:`check_flow_conformance`): verifies that each
  card follows the process prescribed by a :class:`~bomi.BoardModel` —
  created in the right lists, moved along allowed transitions, updated in use
  lists, and closed in the expected way.

- **Performance conformance**: checks WIP limits and per-list SLA constraints
  stored in :attr:`~bomi.BoardModel.list_constraints`.  Four functions cover
  the two constraint types across two time perspectives:

  - :func:`check_wip_history` / :func:`check_wip_current`
  - :func:`check_sla_history` / :func:`check_sla_current`
"""
from __future__ import annotations

import pandas as pd

from .core import BoardModel
from .schema import (
    CARD_EVENT_TYPE,
    CARD_ID,
    LIST_NAME,
    SOURCE_LIST_NAME,
    TARGET_LIST_NAME,
    TIMESTAMP,
)

__all__ = [
    "check_flow_conformance",
    "check_wip_history",
    "check_wip_current",
    "check_sla_history",
    "check_sla_current",
]


# ---------------------------------------------------------------------------
# Internal helpers — WIP
# ---------------------------------------------------------------------------

def _build_wip_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy (timestamp, list_name, delta) table of occupancy changes."""
    parts = []

    creates = df[df[CARD_EVENT_TYPE] == "card_create"].dropna(
        subset=[CARD_ID, LIST_NAME, TIMESTAMP]
    )
    if not creates.empty:
        parts.append(
            creates[[TIMESTAMP, LIST_NAME]].rename(columns={LIST_NAME: "list_name"}).assign(delta=1)
        )

    move_cols = [CARD_ID, SOURCE_LIST_NAME, TARGET_LIST_NAME, TIMESTAMP]
    if all(c in df.columns for c in move_cols):
        moves = df[df[CARD_EVENT_TYPE] == "card_move"].dropna(subset=move_cols)
        if not moves.empty:
            parts.append(
                moves[[TIMESTAMP, SOURCE_LIST_NAME]].rename(columns={SOURCE_LIST_NAME: "list_name"}).assign(delta=-1)
            )
            parts.append(
                moves[[TIMESTAMP, TARGET_LIST_NAME]].rename(columns={TARGET_LIST_NAME: "list_name"}).assign(delta=1)
            )

    closes = df[
        df[CARD_EVENT_TYPE].isin(["card_close", "card_delete"])
    ].dropna(subset=[CARD_ID, LIST_NAME, TIMESTAMP])
    if not closes.empty:
        parts.append(
            closes[[TIMESTAMP, LIST_NAME]].rename(columns={LIST_NAME: "list_name"}).assign(delta=-1)
        )

    if not parts:
        return pd.DataFrame(columns=[TIMESTAMP, "list_name", "delta"])
    return pd.concat(parts, ignore_index=True)


def _wip_stats_for_list(
    all_events: pd.DataFrame,
    list_name: str,
    wip_limit: int,
    end_time: pd.Timestamp,
) -> dict:
    """Compute WIP stats for one list from a pre-built events table."""
    list_events = all_events[all_events["list_name"] == list_name]
    if list_events.empty:
        return {
            "max_wip": 0,
            "current_wip": 0,
            "violated": False,
            "total_violation_time": pd.Timedelta(0),
        }

    # Keep timestamps as pandas objects (preserves tz) instead of using .values
    grouped = (
        list_events.groupby(TIMESTAMP)["delta"]
        .sum()
        .reset_index()
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )
    wip = grouped["delta"].cumsum()

    total_violation = pd.Timedelta(0)
    for i in range(len(grouped)):
        if wip.iloc[i] > wip_limit:
            next_time = grouped.iloc[i + 1][TIMESTAMP] if i + 1 < len(grouped) else end_time
            total_violation += next_time - grouped.iloc[i][TIMESTAMP]

    return {
        "max_wip": int(wip.max()),
        "current_wip": int(wip.iloc[-1]),
        "violated": int(wip.max()) > wip_limit,
        "total_violation_time": total_violation,
    }


# ---------------------------------------------------------------------------
# Internal helpers — stays
# ---------------------------------------------------------------------------

def _compute_stays(df: pd.DataFrame) -> pd.DataFrame:
    """Compute completed card stays per (card_id, list_name).

    A stay is completed when the card leaves the list via a move, archive, or
    delete.  Cards still occupying a list at the end of the log are excluded;
    use :func:`_compute_current_stays` for those.

    Returns a DataFrame with columns:
    ``card_id``, ``list_name``, ``entry_time``, ``exit_time``,
    ``time_in_list``.
    """
    all_cards = df[CARD_ID].dropna().unique()
    rows = []

    for card_id in all_cards:
        card_df = df[df[CARD_ID] == card_id].sort_values(TIMESTAMP)
        current_list: str | None = None
        entry_time = None

        for _, row in card_df.iterrows():
            etype = row[CARD_EVENT_TYPE]
            if etype == "card_create":
                current_list = row[LIST_NAME] if pd.notna(row[LIST_NAME]) else None
                entry_time = row[TIMESTAMP]
            elif etype == "card_move":
                if current_list is not None:
                    rows.append({
                        CARD_ID: card_id,
                        LIST_NAME: current_list,
                        "entry_time": entry_time,
                        "exit_time": row[TIMESTAMP],
                    })
                tgt = row[TARGET_LIST_NAME]
                current_list = tgt if pd.notna(tgt) else None
                entry_time = row[TIMESTAMP]
            elif etype in ("card_close", "card_delete"):
                if current_list is not None:
                    rows.append({
                        CARD_ID: card_id,
                        LIST_NAME: current_list,
                        "entry_time": entry_time,
                        "exit_time": row[TIMESTAMP],
                    })
                current_list = None

    if not rows:
        return pd.DataFrame(
            columns=[CARD_ID, LIST_NAME, "entry_time", "exit_time", "time_in_list"]
        )
    result = pd.DataFrame(rows)
    result["time_in_list"] = result["exit_time"] - result["entry_time"]
    return result


def _compute_current_stays(
    df: pd.DataFrame, reference_date: pd.Timestamp
) -> pd.DataFrame:
    """Return cards still in a list at *reference_date* with their entry time.

    Returns a DataFrame with columns: ``card_id``, ``list_name``,
    ``entry_time``.
    """
    df_slice = df[df[TIMESTAMP] <= reference_date]
    all_cards = df_slice[CARD_ID].dropna().unique()
    rows = []

    for card_id in all_cards:
        card_df = df_slice[df_slice[CARD_ID] == card_id].sort_values(TIMESTAMP)
        current_list: str | None = None
        entry_time = None

        for _, row in card_df.iterrows():
            etype = row[CARD_EVENT_TYPE]
            if etype == "card_create":
                current_list = row[LIST_NAME] if pd.notna(row[LIST_NAME]) else None
                entry_time = row[TIMESTAMP]
            elif etype == "card_move":
                tgt = row[TARGET_LIST_NAME]
                current_list = tgt if pd.notna(tgt) else None
                entry_time = row[TIMESTAMP]
            elif etype in ("card_close", "card_delete"):
                current_list = None

        if current_list is not None:
            rows.append({CARD_ID: card_id, LIST_NAME: current_list, "entry_time": entry_time})

    if not rows:
        return pd.DataFrame(columns=[CARD_ID, LIST_NAME, "entry_time"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Flow conformance
# ---------------------------------------------------------------------------

def check_flow_conformance(
    df: pd.DataFrame,
    model: BoardModel,
    strict_updates: bool = False,
) -> pd.DataFrame:
    """Check per-card flow conformance against a board model.

    Four dimensions are checked for each card:

    - **create**: card was created in a list in ``model.card_create_lists``.
    - **move**: every (source, target) transition is in ``model.allowed_flow``.
    - **update**: every ``card_act`` event occurs in a list in
      ``model.card_use_lists``.
    - **close**: finished cards were closed according to ``model.close_mode``
      (OR semantics when multiple methods are specified).

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    model:
        Board model returned by :func:`~bomi.board_discovery` or constructed
        manually.
    strict_updates:
        When ``True``, ``card_act`` events in lists not in
        ``model.card_use_lists`` count as violations and affect the overall
        ``conformant`` flag.  When ``False`` (default), update violations are
        reported but do not affect ``conformant``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``card_id``.  Columns:

        - ``create_conformant`` (bool)
        - ``moves_total``, ``moves_nonconformant`` (int),
          ``move_conformance_rate`` (float, 1.0 for cards with no moves)
        - ``updates_total``, ``updates_nonconformant`` (int),
          ``update_conformance_rate`` (float)
        - ``close_conformant`` (bool or ``None`` for still-active cards)
        - ``conformant`` (bool)
    """
    modes = (
        {model.close_mode}
        if isinstance(model.close_mode, str)
        else set(model.close_mode)
    )
    allowed_flow_set = set(model.allowed_flow)
    create_set = set(model.card_create_lists)
    use_set = set(model.card_use_lists)
    close_set = set(model.card_close_lists)

    all_cards = pd.Index(df[CARD_ID].dropna().unique(), name=CARD_ID)

    # --- create conformance ---
    creates = df[df[CARD_EVENT_TYPE] == "card_create"].dropna(subset=[CARD_ID, LIST_NAME])
    first_list = creates.sort_values(TIMESTAMP).groupby(CARD_ID)[LIST_NAME].first()
    if create_set:
        create_conformant = first_list.isin(create_set)
    else:
        create_conformant = pd.Series(True, index=first_list.index, dtype=bool)
    create_conformant = create_conformant.reindex(all_cards, fill_value=True)

    # --- move conformance ---
    _move_cols = [CARD_ID, SOURCE_LIST_NAME, TARGET_LIST_NAME]
    _has_move_cols = all(c in df.columns for c in _move_cols)
    moves = (
        df[df[CARD_EVENT_TYPE] == "card_move"].dropna(subset=_move_cols)
        if _has_move_cols
        else pd.DataFrame()
    )
    if not moves.empty:
        moves = moves.copy()
        pairs = list(zip(moves[SOURCE_LIST_NAME], moves[TARGET_LIST_NAME]))
        moves["_nonconf"] = [p not in allowed_flow_set for p in pairs]
        moves_total = (
            moves.groupby(CARD_ID)["_nonconf"].count().reindex(all_cards, fill_value=0)
        )
        moves_nonconf = (
            moves.groupby(CARD_ID)["_nonconf"].sum().astype(int).reindex(all_cards, fill_value=0)
        )
    else:
        moves_total = pd.Series(0, index=all_cards, dtype=int)
        moves_nonconf = pd.Series(0, index=all_cards, dtype=int)
    move_conformance_rate = (moves_total - moves_nonconf) / moves_total.where(moves_total > 0, 1)
    move_conformance_rate[moves_total == 0] = 1.0

    # --- update conformance ---
    acts = df[df[CARD_EVENT_TYPE] == "card_act"].dropna(subset=[CARD_ID, LIST_NAME])
    if not acts.empty:
        updates_total = (
            acts.groupby(CARD_ID)[LIST_NAME].count().reindex(all_cards, fill_value=0)
        )
        if use_set:
            acts = acts.copy()
            acts["_nonconf"] = ~acts[LIST_NAME].isin(use_set)
            updates_nonconf = (
                acts.groupby(CARD_ID)["_nonconf"].sum().astype(int).reindex(all_cards, fill_value=0)
            )
        else:
            updates_nonconf = pd.Series(0, index=all_cards, dtype=int)
    else:
        updates_total = pd.Series(0, index=all_cards, dtype=int)
        updates_nonconf = pd.Series(0, index=all_cards, dtype=int)
    update_conformance_rate = (
        (updates_total - updates_nonconf) / updates_total.where(updates_total > 0, 1)
    )
    update_conformance_rate[updates_total == 0] = 1.0

    # --- close conformance ---
    has_close_cards = set(df[df[CARD_EVENT_TYPE] == "card_close"][CARD_ID].dropna())
    has_delete_cards = set(df[df[CARD_EVENT_TYPE] == "card_delete"][CARD_ID].dropna())

    _tgt_cols = [CARD_ID, TARGET_LIST_NAME]
    if all(c in df.columns for c in _tgt_cols):
        move_targets = df[df[CARD_EVENT_TYPE] == "card_move"].dropna(subset=_tgt_cols)
        last_move_list = (
            move_targets.sort_values(TIMESTAMP).groupby(CARD_ID)[TARGET_LIST_NAME].last()
        )
    else:
        last_move_list = pd.Series(dtype=str)
    last_list = last_move_list.combine_first(first_list).reindex(all_cards)

    is_archived = pd.Series(all_cards.isin(has_close_cards), index=all_cards)
    is_deleted = pd.Series(all_cards.isin(has_delete_cards), index=all_cards)
    is_in_close_list = (
        last_list.isin(close_set) if close_set else pd.Series(False, index=all_cards)
    )

    is_finished = is_archived | is_deleted | is_in_close_list

    conf_close = pd.Series(False, index=all_cards)
    if "archived" in modes:
        conf_close |= is_archived
    if "deleted" in modes:
        conf_close |= is_deleted
    if "sink_list" in modes:
        conf_close |= is_in_close_list

    # Use pd.NA as the "no verdict yet" sentinel (pd.isna() detects it cleanly)
    close_conformant: pd.Series = pd.Series(pd.NA, index=all_cards, dtype=object)
    close_conformant[is_finished] = conf_close[is_finished].values

    # --- overall conformance ---
    close_ok = close_conformant.apply(lambda x: pd.isna(x) or bool(x))
    conformant = create_conformant & (moves_nonconf == 0) & close_ok
    if strict_updates:
        conformant &= updates_nonconf == 0

    return pd.DataFrame(
        {
            "create_conformant": create_conformant,
            "moves_total": moves_total.astype(int),
            "moves_nonconformant": moves_nonconf,
            "move_conformance_rate": move_conformance_rate,
            "updates_total": updates_total.astype(int),
            "updates_nonconformant": updates_nonconf,
            "update_conformance_rate": update_conformance_rate,
            "close_conformant": close_conformant,
            "conformant": conformant,
        },
        index=all_cards,
    )


# ---------------------------------------------------------------------------
# WIP conformance
# ---------------------------------------------------------------------------

def check_wip_history(df: pd.DataFrame, model: BoardModel) -> pd.DataFrame:
    """Check WIP limit conformance over the full history of the board.

    For each list with a ``wip_limit`` in ``model.list_constraints``, replays
    the card-movement timeline and measures how long the occupancy exceeded
    the limit.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    model:
        Board model with ``list_constraints`` populated.

    Returns
    -------
    pd.DataFrame
        Indexed by ``list_name`` (only lists with a ``wip_limit`` set).
        Columns: ``wip_limit``, ``max_wip``, ``violated``,
        ``total_violation_time`` (Timedelta — cumulative duration WIP exceeded
        the limit).
    """
    constrained = {
        name: c for name, c in model.list_constraints.items()
        if c.wip_limit is not None
    }
    if not constrained:
        return pd.DataFrame(
            columns=["wip_limit", "max_wip", "violated", "total_violation_time"]
        )

    events = _build_wip_events(df)
    end_time = df[TIMESTAMP].max()

    rows = {}
    for list_name, constraints in constrained.items():
        stats = _wip_stats_for_list(events, list_name, constraints.wip_limit, end_time)
        rows[list_name] = {
            "wip_limit": constraints.wip_limit,
            "max_wip": stats["max_wip"],
            "violated": stats["violated"],
            "total_violation_time": stats["total_violation_time"],
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = LIST_NAME
    return result


def check_wip_current(
    df: pd.DataFrame,
    model: BoardModel,
    reference_date=None,
) -> pd.DataFrame:
    """Check WIP limit conformance at a specific point in time.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    model:
        Board model with ``list_constraints`` populated.
    reference_date:
        Point in time to assess.  Defaults to the last timestamp in *df*.

    Returns
    -------
    pd.DataFrame
        Indexed by ``list_name`` (only lists with a ``wip_limit`` set).
        Columns: ``wip_limit``, ``current_wip``, ``violated``,
        ``total_violation_time`` (Timedelta — cumulative duration WIP exceeded
        the limit up to *reference_date*).
    """
    constrained = {
        name: c for name, c in model.list_constraints.items()
        if c.wip_limit is not None
    }
    if not constrained:
        return pd.DataFrame(
            columns=["wip_limit", "current_wip", "violated", "total_violation_time"]
        )

    ref = df[TIMESTAMP].max() if reference_date is None else pd.Timestamp(reference_date)
    events = _build_wip_events(df[df[TIMESTAMP] <= ref])

    rows = {}
    for list_name, constraints in constrained.items():
        stats = _wip_stats_for_list(events, list_name, constraints.wip_limit, ref)
        rows[list_name] = {
            "wip_limit": constraints.wip_limit,
            "current_wip": stats["current_wip"],
            "violated": stats["current_wip"] > constraints.wip_limit,
            "total_violation_time": stats["total_violation_time"],
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = LIST_NAME
    return result


# ---------------------------------------------------------------------------
# SLA conformance
# ---------------------------------------------------------------------------

def check_sla_history(df: pd.DataFrame, model: BoardModel) -> pd.DataFrame:
    """Check SLA conformance for completed card stays.

    Only completed stays (card has since left the list) are included.  Cards
    still in a list are handled by :func:`check_sla_current`.  When a card
    visited the same list more than once, each visit produces a separate row.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    model:
        Board model with ``list_constraints`` populated.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(card_id, list_name)``.  Columns: ``time_in_list``,
        ``sla_limit``, ``violated``.  May contain duplicate index entries
        when a card visited the same list multiple times.
    """
    constrained = {
        name: c for name, c in model.list_constraints.items()
        if c.sla is not None
    }
    if not constrained:
        return pd.DataFrame(columns=["time_in_list", "sla_limit", "violated"])

    stays = _compute_stays(df)
    relevant = stays[stays[LIST_NAME].isin(constrained)].copy()

    if relevant.empty:
        return pd.DataFrame(columns=["time_in_list", "sla_limit", "violated"])

    relevant["sla_limit"] = relevant[LIST_NAME].map(lambda l: constrained[l].sla)
    relevant["violated"] = relevant["time_in_list"] > relevant["sla_limit"]
    return relevant.set_index([CARD_ID, LIST_NAME])[["time_in_list", "sla_limit", "violated"]]


def check_sla_current(
    df: pd.DataFrame,
    model: BoardModel,
    reference_date=None,
) -> pd.DataFrame:
    """Check SLA conformance for cards currently in constrained lists.

    Only reports cards actively in a list with an SLA constraint at
    *reference_date*.  Completed stays are handled by :func:`check_sla_history`.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    model:
        Board model with ``list_constraints`` populated.
    reference_date:
        Point in time to assess.  Defaults to the last timestamp in *df*.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(card_id, list_name)``.  Columns:
        ``time_in_list_so_far``, ``sla_limit``, ``violated``.
    """
    constrained = {
        name: c for name, c in model.list_constraints.items()
        if c.sla is not None
    }
    if not constrained:
        return pd.DataFrame(columns=["time_in_list_so_far", "sla_limit", "violated"])

    ref = df[TIMESTAMP].max() if reference_date is None else pd.Timestamp(reference_date)
    current = _compute_current_stays(df, ref)
    relevant = current[current[LIST_NAME].isin(constrained)].copy()

    if relevant.empty:
        return pd.DataFrame(columns=["time_in_list_so_far", "sla_limit", "violated"])

    relevant["time_in_list_so_far"] = ref - relevant["entry_time"]
    relevant["sla_limit"] = relevant[LIST_NAME].map(lambda l: constrained[l].sla)
    relevant["violated"] = relevant["time_in_list_so_far"] > relevant["sla_limit"]
    return relevant.set_index([CARD_ID, LIST_NAME])[
        ["time_in_list_so_far", "sla_limit", "violated"]
    ]

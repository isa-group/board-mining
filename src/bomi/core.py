"""Analysis functions for bomi board event logs.

All functions in this module expect a DataFrame in the bomi board schema (see
:mod:`bomi.schema`).  The classification columns ``card_event_type`` and
``list_event_type`` must already be present; connectors populate them
automatically via :func:`bomi.schema.compute_event_types`.
"""

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from dataclasses import dataclass, field

try:
    import pm4py
except ImportError:
    pm4py = None

from .schema import (
    ACTOR_ID,
    CARD_CLOSED,
    CARD_EVENT_TYPE,
    CARD_ID,
    EVENT_ID,
    EVENT_TYPE,
    LIST_EVENT_TYPE,
    LIST_ID,
    LIST_NAME,
    RAW_EVENT_TYPE,
    SOURCE_LIST_ID,
    SOURCE_LIST_NAME,
    TARGET_LIST_ID,
    TARGET_LIST_NAME,
    TIMESTAMP,
)


# ---------------------------------------------------------------------------
# Event-type filter helpers (schema-based)
# ---------------------------------------------------------------------------

def card_create_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting card-creation events."""
    return df[CARD_EVENT_TYPE] == "card_create"


def card_movement_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting card-movement events."""
    return df[CARD_EVENT_TYPE] == "card_move"


def card_closed_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting card-close events."""
    return df[CARD_EVENT_TYPE] == "card_close"


def card_action_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting generic card-action events (updates, comments, …)."""
    return df[CARD_EVENT_TYPE] == "card_act"


# ---------------------------------------------------------------------------
# Basic log information
# ---------------------------------------------------------------------------

def log_info(df: pd.DataFrame, notypes: bool = True) -> dict:
    """Return a summary dictionary of basic statistics about a board event log.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    notypes:
        When ``False``, include a breakdown of raw event type counts under the
        ``"types"`` key.

    Returns
    -------
    dict
        Keys include ``events``, ``cards``, ``lists``, ``members``,
        ``board_duration``, card/list lifecycle counts, and per-card
        percentage metrics.
    """
    info: dict = {}
    info["events"] = len(df)
    info["attribs"] = len(df.columns)
    info["cards"] = df[CARD_ID].nunique() if CARD_ID in df.columns else 0
    info["lists"] = df[LIST_ID].nunique() if LIST_ID in df.columns else 0
    info["list_first_create"] = df[df[LIST_EVENT_TYPE] == "list_create"][TIMESTAMP].min()
    info["list_last_create"] = df[df[LIST_EVENT_TYPE] == "list_create"][TIMESTAMP].max()
    info["list_renamed"] = (df[LIST_EVENT_TYPE] == "list_rename").sum()
    info["list_closed"] = (df[LIST_EVENT_TYPE] == "list_ends").sum()
    info["start"] = df[TIMESTAMP].min()
    info["ends"] = df[TIMESTAMP].max()
    info["board_duration"] = info["ends"] - info["start"]
    info["first_event_type"] = df.loc[df[TIMESTAMP].idxmin(), EVENT_TYPE]
    info["members"] = df[ACTOR_ID].nunique() if ACTOR_ID in df.columns else 0
    info["events_per_member"] = df[ACTOR_ID].value_counts().describe().to_dict() if ACTOR_ID in df.columns else {}
    info["card_movement"] = card_movement_filter(df).sum()
    info["card_closed"] = card_closed_filter(df).sum()
    info["card_deleted"] = (df[CARD_EVENT_TYPE] == "card_delete").sum()
    if CARD_ID in df.columns:
        cards = df[CARD_ID].nunique()
        info["cards_moving_perc"] = df[card_movement_filter(df)][CARD_ID].nunique() / cards if cards else 0
        info["cards_closed_perc"] = df[card_closed_filter(df)][CARD_ID].nunique() / cards if cards else 0
    if not notypes:
        info["types"] = df[EVENT_TYPE].value_counts().to_dict()
    return info


def list_renames(df: pd.DataFrame) -> pd.Series:
    """Return all names each list ever had, indexed by list ID.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    pandas.Series
        Index is ``list_id``; values are arrays of unique ``list_name`` values
        seen across all events for that list.
    """
    return df[LIST_NAME].groupby(df[LIST_ID]).unique()


# ---------------------------------------------------------------------------
# Board evolution
# ---------------------------------------------------------------------------

def board_evolution(
    df: pd.DataFrame,
    bins: int = 30,
    list_name: str | None = None,
    all_lists: str | None = None,
) -> pd.DataFrame | pd.Series:
    """Count events per time bin broken down by event category.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    bins:
        Number of equal-width time bins to divide the board's lifetime into.
    list_name:
        If given, restrict the analysis to events on this list.
    all_lists:
        If given, return only the Series for that event category key
        (``"events"``, ``"list_events"``, ``"card_creation"``,
        ``"card_movement"``, ``"card_closed"``, or ``"card_action"``),
        broken down further by list name.

    Returns
    -------
    pandas.DataFrame
        One column per event category and one row per time bin, unless
        *all_lists* is specified, in which case a single Series is returned
        with a two-level index (bin, list_name).
    """
    time_bins = pd.cut(df[TIMESTAMP], bins=bins)

    if list_name is not None:
        mask = df[LIST_NAME] == list_name
        df = df[mask]
        time_bins = time_bins[mask]

    def bin_filter(mask):
        groups = [time_bins[mask], LIST_NAME] if all_lists is not None else time_bins[mask]
        return df[mask].groupby(groups)[EVENT_ID].count()

    cc = {
        "events": bin_filter(df[EVENT_ID].notna()),
        "list_events": bin_filter(df[LIST_EVENT_TYPE].notna()),
        "card_creation": bin_filter(card_create_filter(df)),
        "card_movement": bin_filter(card_movement_filter(df)),
        "card_closed": bin_filter(card_closed_filter(df)),
        "card_action": bin_filter(card_action_filter(df)),
    }

    return cc[all_lists] if all_lists is not None else pd.concat(cc, axis=1)


# ---------------------------------------------------------------------------
# List evolution
# ---------------------------------------------------------------------------

def list_evolution(df: pd.DataFrame, filter_short_lists=None) -> pd.DataFrame:
    """Return the active period, names, and last name of every list in the board.

    A list's active period begins at its first event and ends at its last event
    if it was closed or moved off the board, or at the board's last event
    otherwise.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    filter_short_lists:
        Optional :class:`pandas.Timedelta`; lists whose active duration is
        shorter than this value are excluded from the result.

    Returns
    -------
    pandas.DataFrame
        One row per list ID with columns ``begin_date``, ``last_date``,
        ``last_name``, and ``list_name`` (all names the list ever had).
        Sorted by ``last_date``.
    """
    list_group = df[df[LIST_EVENT_TYPE].notna()].groupby(LIST_ID)

    begin_date = list_group[TIMESTAMP].min().rename("begin_date")
    last_date = list_group[TIMESTAMP].max()
    finished = list_group[LIST_EVENT_TYPE].first().isin(["list_ends", "list_move"])
    last_date.loc[~finished] = df[TIMESTAMP].max()
    last_date.name = "last_date"

    names = list_group[LIST_NAME].unique()
    last_name = list_group[LIST_NAME].first().rename("last_name")

    result = pd.concat([begin_date, last_date, last_name, names], axis=1).sort_values("last_date")

    if filter_short_lists is not None:
        return result[(result["last_date"] - result["begin_date"] > filter_short_lists)]
    return result


# ---------------------------------------------------------------------------
# Redesign detection
# ---------------------------------------------------------------------------

def detect_redesign(
    df: pd.DataFrame,
    threshold=pd.Timedelta("1D"),
    l_type=None,
    threshold_l_events: int = 0,
) -> pd.DataFrame:
    """Detect periods of board structural change (redesigns).

    A redesign is a cluster of events that are close in time to one or more
    list-level events.  Two threshold modes are supported:

    - **Time-based** (``threshold`` is a :class:`pandas.Timedelta`): events
      within *threshold* time of a list event are included in the redesign.
    - **Count-based** (``threshold`` is an int): a redesign ends when more
      than *threshold* card events occur after the last list event.

    Parameters
    ----------
    df:
        Board event log in the bomi schema, sorted by timestamp.
    threshold:
        Either a :class:`pandas.Timedelta` (time window) or an ``int``
        (maximum number of intervening card events before the redesign ends).
        Defaults to ``pd.Timedelta("1D")``.
    l_type:
        Restrict the triggering list events to a subset of
        ``list_event_type`` values (e.g. ``["list_create", "list_rename"]``).
        ``None`` uses all list events.
    threshold_l_events:
        Minimum number of list events required for a cluster to be reported
        as a redesign.  Clusters with fewer list events are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per detected redesign with columns ``min`` (start timestamp),
        ``max`` (end timestamp), ``count`` (total events), and
        ``count_l_events`` (list events in the redesign).
    """
    df_rev = df.sort_index(ascending=False)

    if l_type is None:
        l_events = df_rev[LIST_EVENT_TYPE].notna()
    else:
        l_events = df_rev[LIST_EVENT_TYPE].isin(l_type)

    if isinstance(threshold, pd.Timedelta):
        r_events = df_rev.groupby(l_events.cumsum())[TIMESTAMP].transform(
            lambda x: x - np.min(x)
        ) < threshold
        redesign_events = r_events | l_events
    else:
        r_events = df_rev.groupby(l_events.cumsum())[EVENT_ID].transform("count") < threshold
        redesign_events = r_events | l_events

    redesign_events.sort_index(ascending=True, inplace=True)

    signal_redesign = redesign_events & ~redesign_events.shift(1, fill_value=False)
    count_l_events = df[df[LIST_EVENT_TYPE].notna()].groupby(signal_redesign.cumsum())[EVENT_ID].count()
    count_l_events.name = "count_l_events"

    result = pd.concat(
        [df[redesign_events].groupby(signal_redesign.cumsum())[TIMESTAMP].agg(["min", "max", "count"]),
         count_l_events],
        axis=1,
    )

    if threshold_l_events > 0:
        return result[result["count_l_events"] > threshold_l_events]
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_list_diagram(
    list_evolution_df: pd.DataFrame,
    begin_end_redesign: pd.DataFrame,
    ax,
) -> None:
    """Draw a Gantt-style chart of list lifetimes with redesign markers.

    Each list is shown as a horizontal bar spanning its active period.  Red
    vertical lines mark the start of each detected redesign; blue lines mark
    the end.

    Parameters
    ----------
    list_evolution_df:
        Output of :func:`list_evolution`.
    begin_end_redesign:
        Output of :func:`detect_redesign`.
    ax:
        Matplotlib axes to draw on.
    """
    lists = {p: i for i, p in enumerate(list(list_evolution_df.index.values))}
    min_date = min(list_evolution_df["begin_date"]) - pd.Timedelta("5D")
    max_date = max(list_evolution_df["last_date"])

    for index, row in list_evolution_df.iterrows():
        ax.broken_barh(
            [(row["begin_date"], row["last_date"] - row["begin_date"])],
            (lists[index] - 0.45, 0.9),
            facecolors=matplotlib.colormaps["plasma"](lists[index] / len(lists)),
        )

    ax.vlines(begin_end_redesign["min"], 0, len(lists), colors="tab:red")
    ax.vlines(begin_end_redesign["max"], 0, len(lists), colors="tab:blue")
    ax.set_yticks(range(len(lists)))
    ax.set_yticklabels(list_evolution_df["last_name"])
    ax.set_ylim(bottom=0, top=len(lists))
    ax.set_xlim(left=min_date, right=max_date)
    ax.set_xlabel("Date", fontdict={"family": "DejaVu Sans", "color": "black", "weight": "bold", "size": 14})
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))
    ax.tick_params(which="major", axis="x", rotation=90, length=11, color="black")


def plot_card_actions_ind(df: pd.DataFrame, filter=None, **kwargs) -> None:
    """Draw a 2-D histogram of card-create and card-act events per list over time.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    filter:
        Optional date filter.  Pass a single ``date`` to show only events
        before that date, or a ``(start, end)`` tuple to show a specific range.
    **kwargs:
        Extra keyword arguments forwarded to :func:`seaborn.displot`.
    """
    if filter is None:
        first = df
    elif isinstance(filter, tuple):
        first = df[(df[TIMESTAMP].dt.date > filter[0]) & (df[TIMESTAMP].dt.date < filter[1])]
    else:
        first = df[df[TIMESTAMP].dt.date < filter]

    chart = sns.displot(
        first[df[CARD_EVENT_TYPE].isin(["card_act", "card_create"])][[TIMESTAMP, CARD_EVENT_TYPE, LIST_NAME]].fillna("##UNKNOWN"),
        x=TIMESTAMP,
        row=CARD_EVENT_TYPE,
        binwidth=1,
        y=LIST_NAME,
        kind="hist",
        **kwargs,
    )
    chart.set_xticklabels(rotation=45)


def plot_card_actions(
    df: pd.DataFrame,
    begin_end_redesign: pd.DataFrame | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Draw a categorical scatter plot of card events per list over time.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    begin_end_redesign:
        Optional output of :func:`detect_redesign`; redesign start times are
        drawn as horizontal reference lines.
    **kwargs:
        Extra keyword arguments forwarded to :func:`seaborn.catplot`.

    Returns
    -------
    seaborn.FacetGrid
    """
    if "height" not in kwargs:
        kwargs["height"] = 20
    if "aspect" not in kwargs:
        kwargs["aspect"] = 20 / 9

    ch = sns.catplot(
        x=LIST_NAME,
        y=TIMESTAMP,
        hue=CARD_EVENT_TYPE,
        data=df[df[CARD_EVENT_TYPE].isin(["card_act", "card_create", "card_move", "card_close"])][
            [CARD_EVENT_TYPE, TIMESTAMP, LIST_NAME]
        ].fillna("##UNKNOWN"),
        **kwargs,
    )
    ch.set_xticklabels(rotation=90)

    if begin_end_redesign is not None:
        for f in begin_end_redesign["min"].values:
            ch.refline(y=f)

    return ch


def plot_card_actions_summary(
    df: pd.DataFrame,
    begin_end_redesign: pd.DataFrame | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Draw small-multiple histograms of card events over time, one panel per event type.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    begin_end_redesign:
        Optional output of :func:`detect_redesign`; redesign start times are
        drawn as vertical reference lines on each panel.
    **kwargs:
        Extra keyword arguments forwarded to :func:`seaborn.displot`.

    Returns
    -------
    seaborn.FacetGrid
    """
    ax = sns.displot(df, x=TIMESTAMP, col=CARD_EVENT_TYPE, col_wrap=2, **kwargs)
    ax.set_xticklabels(rotation=50)

    if begin_end_redesign is not None:
        for f in begin_end_redesign["min"].values:
            ax.refline(x=f)

    return ax


# ---------------------------------------------------------------------------
# Process mining bridge
# ---------------------------------------------------------------------------

def to_event_log(df: pd.DataFrame):
    """Convert a bomi event log to a pm4py EventLog.

    Cards become cases; list names (or ``"**Closed"`` for closed cards) become
    activity names.  Only card-create, card-move, and card-close events are
    included.  Requires the optional ``pm4py`` dependency (install bomi with
    ``pip install 'bomi[process]'``).

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    pm4py.objects.log.obj.EventLog
    """
    if pm4py is None:
        raise ImportError(
            "to_event_log requires the optional dependency 'pm4py'. "
            "Install bomi with the process extra: pip install 'bomi[process]'."
        )

    mask = df[CARD_EVENT_TYPE].isin(["card_create", "card_close", "card_move"])
    log = df[mask].copy()

    concept = log[TARGET_LIST_NAME].fillna(log[LIST_NAME]).fillna(log[EVENT_TYPE])
    card_closed_mask = log[CARD_CLOSED].fillna(False).astype(bool)
    concept.loc[card_closed_mask] = "**Closed"

    ll = pd.DataFrame({
        "org:resource": log[ACTOR_ID],
        "type": log[RAW_EVENT_TYPE],
        "time:timestamp": log[TIMESTAMP],
        "case:concept:name": log[CARD_ID],
        "concept:name": concept,
    })

    lldf = pm4py.format_dataframe(ll.sort_values("time:timestamp"))
    return pm4py.convert_to_event_log(lldf)


# ---------------------------------------------------------------------------
# Transition matrix and graph analysis
# ---------------------------------------------------------------------------

def transition_matrix(df: pd.DataFrame, use: str = "names") -> pd.DataFrame:
    """Count card movements between every pair of lists.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    use:
        Column used to identify lists: ``"names"`` (default) uses
        ``list_name``; ``"id"`` uses ``list_id`` (with names substituted
        in the result via :func:`_create_conversion_map`).

    Returns
    -------
    pandas.DataFrame
        Square pivot table where ``result.loc[A, B]`` is the number of cards
        that moved from list A to list B.
    """
    if use == "names":
        return df.groupby([SOURCE_LIST_NAME, TARGET_LIST_NAME])[EVENT_ID].count().unstack()
    else:
        conversion_map = _create_conversion_map(df)
        matrix = df.groupby([SOURCE_LIST_ID, TARGET_LIST_ID])[EVENT_ID].count().unstack()
        return matrix.rename(index=conversion_map, columns=conversion_map)


def _apply_relative_threshold(series: pd.Series, threshold_percent: float) -> pd.Series:
    """Filter series by relative threshold (percentage of maximum value).

    Parameters
    ----------
    series:
        Values to filter.
    threshold_percent:
        Percentage threshold (0-100). Items with value >= max * threshold% are included.

    Returns
    -------
    pd.Series
        Boolean mask for items to include.
    """
    if threshold_percent < 0 or threshold_percent > 100:
        raise ValueError("threshold_percent must be between 0 and 100")

    max_val = series.max()
    if max_val == 0:
        # When max is 0, include everything (graceful degradation on sparse data)
        return pd.Series(True, index=series.index)

    return series >= (max_val * threshold_percent / 100)


def connected_lists(df: pd.DataFrame, use: str = "id", threshold: float = 0) -> pd.DataFrame:
    """Find connected components in the card-flow graph between lists.

    Two lists are connected if at least one card moved between them (directly
    or transitively).  The result helps identify independent workflow streams
    on the board.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    use:
        Column used to identify lists: ``"id"`` (default) or ``"names"``.
    threshold:
        Percentage threshold (0-100). Include edges with movement count >= max * threshold%.
        ``0`` (default) includes all movements; ``100`` includes only the most frequent edge.

    Returns
    -------
    pandas.DataFrame
        One row per connected component with columns ``component`` (set of
        list identifiers), ``size`` (number of lists), and ``count``
        (number of card-move events involving those lists).
    """
    G = nx.Graph()
    if use == "id":
        G.add_nodes_from(df[LIST_ID].dropna())
        pair = [TARGET_LIST_ID, SOURCE_LIST_ID]
        key = TARGET_LIST_ID
    else:
        G.add_nodes_from(df[LIST_NAME].dropna())
        pair = [TARGET_LIST_NAME, SOURCE_LIST_NAME]
        key = TARGET_LIST_NAME

    if threshold > 0:
        count = df[pair + [CARD_ID]].dropna().groupby(pair).count()
        mask = _apply_relative_threshold(count[CARD_ID], threshold)
        G.add_edges_from(count[mask].index.to_numpy().tolist())
    else:
        G.add_edges_from(df[pair].dropna().to_numpy().tolist())

    cl = list(nx.connected_components(G))
    return pd.DataFrame(
        [(c, len(c), df[df[key].isin(c)][EVENT_ID].count()) for c in cl],
        columns=["component", "size", "count"],
    )


def card_action_list(
    df: pd.DataFrame,
    type: str = "card_create",
    use: str = "id",
    threshold: float = 0,
) -> pd.Series:
    """Return the normalised fraction of a given event type per list.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    type:
        ``card_event_type`` value to count (e.g. ``"card_create"``,
        ``"card_close"``, ``"card_act"``).
    use:
        Column used to identify lists: ``"id"`` (default, result index is
        human-readable list names) or ``"names"``.
    threshold:
        Percentage threshold (0-100). Include lists with activity count >= max * threshold%.
        ``0`` (default) includes all lists; ``100`` includes only the most active list.

    Returns
    -------
    pandas.Series
        Fraction of the selected event type attributable to each list
        (sums to 1 over the returned lists).
    """
    col = LIST_ID if use == "id" else LIST_NAME
    result = df[df[CARD_EVENT_TYPE] == type].groupby(col)[EVENT_ID].count()

    if use == "id":
        result = result.rename(index=_create_conversion_map(df))

    mask = _apply_relative_threshold(result, threshold)
    return result[mask].transform(lambda x: x / x.sum())


def flow_semantic_precedence(
    df: pd.DataFrame,
    use: str = "id",
    threshold: float = 0,
) -> list:
    """Return (source, target) list pairs that appear in the transition matrix.

    These pairs capture the *semantic precedence* of lists — the order
    relationships inferred from observed card movements that are not visible
    from the board's visual layout alone.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    use:
        Column used to identify lists: ``"id"`` (default) or ``"names"``.
    threshold:
        Percentage threshold (0-100). Include pairs with movement count >= max * threshold%.
        ``0`` (default) includes all pairs; ``100`` includes only the most frequent pair.

    Returns
    -------
    list of tuple
        Each element is a ``(source_list, target_list)`` pair.
    """
    matrix = transition_matrix(df, use)

    if threshold > 0:
        # Flatten matrix to find the maximum pair count
        flat = matrix.stack()
        mask = _apply_relative_threshold(flat, threshold)
        return list(flat[mask].index)

    return list(matrix.stack().index)


@dataclass
class ListConstraints:
    """Per-list performance constraints used by conformance checkers."""
    wip_limit: int | None = None
    sla: pd.Timedelta | None = None


@dataclass
class BoardModel:
    """Structured representation of a discovered board design.

    Returned by :func:`board_discovery`.  The model captures how the board is
    structured and used, and serves as the reference for conformance checking.

    Attributes
    ----------
    lists:
        All list names present in the log.
    card_flow:
        Connected components of the card-flow graph (each is a set of list names).
    semantic_precedence:
        Dominant (source, target) list-name pairs observed in the log.
    card_create_lists:
        Names of lists where cards are most frequently created.
    card_close_lists:
        Names of lists from which cards are most frequently closed/archived.
        Also used as sink lists when ``close_mode`` includes ``"sink_list"``.
    card_use_lists:
        Names of lists where cards are most frequently updated.
    close_mode:
        Expected card-completion method(s): ``"archived"``, ``"deleted"``,
        ``"sink_list"``, or a list combining them (OR semantics).
    allowed_flow_mode:
        Controls :attr:`allowed_flow`.  ``"semantic_precedence"`` (default)
        restricts valid transitions to :attr:`semantic_precedence`.
        ``"card_flow"`` permits any transition within a connected component.
    list_constraints:
        Optional per-list WIP limits and SLA constraints for conformance
        checking.  Populated manually after discovery.
    """
    lists: list[str]
    card_flow: list[set[str]]
    semantic_precedence: list[tuple[str, str]]
    card_create_lists: list[str]
    card_close_lists: list[str]
    card_use_lists: list[str]
    close_mode: str | list[str]
    allowed_flow_mode: str = "semantic_precedence"
    list_constraints: dict[str, ListConstraints] = field(default_factory=dict)

    @property
    def allowed_flow(self) -> list[tuple[str, str]]:
        """Valid (source, target) transitions given the current ``allowed_flow_mode``."""
        if self.allowed_flow_mode == "semantic_precedence":
            return self.semantic_precedence
        return [
            (src, tgt)
            for component in self.card_flow
            for src in component
            for tgt in component
            if src != tgt
        ]


def board_discovery(
    df: pd.DataFrame,
    use: str = "id",
    cf_threshold: float = 0,
    cc_threshold: float = 0,
    cx_threshold: float = 0,
    cu_threshold: float = 0,
    sp_threshold: float = 0,
) -> BoardModel:
    """Infer the board design from observed card and list behaviour.

    Combines connected-list analysis, per-list activity distributions, and
    semantic precedence into a single characterisation of how the board is
    structured and used. All thresholds use relative semantics: include items
    that represent >= threshold% of the maximum activity in their category.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    use:
        Column used to identify lists throughout: ``"id"`` (default) or
        ``"names"``.
    cf_threshold:
        Card-flow percentage threshold (0-100) forwarded to :func:`connected_lists`.
        Include edges with movement count >= max * threshold%.
    cc_threshold:
        Card-creation percentage threshold (0-100).
        Include lists with creation count >= max * threshold%.
    cx_threshold:
        Card-close percentage threshold (0-100).
        Include lists with close count >= max * threshold%.
    cu_threshold:
        Card-use (action) percentage threshold (0-100).
        Include lists with action count >= max * threshold%.
    sp_threshold:
        Semantic-precedence percentage threshold (0-100).
        Include pairs with movement count >= max * threshold%.

    Returns
    -------
    BoardModel
        Discovered board model.  ``allowed_flow_mode`` defaults to
        ``"semantic_precedence"``; change it on the returned object to switch
        to component-level flow permissiveness.  ``close_mode`` is inferred
        from the event log: ``"archived"`` when archive events are present,
        ``"deleted"`` when delete events are present, or both.
        ``list_constraints`` is empty and must be populated manually.
    """
    cl = connected_lists(df, use, cf_threshold)
    cf = list(cl["component"])
    L = [l for c in cf for l in c]
    cc = card_action_list(df, type="card_create", use=use, threshold=cc_threshold)
    cx = card_action_list(df, type="card_close", use=use, threshold=cx_threshold)
    cu = card_action_list(df, type="card_act", use=use, threshold=cu_threshold)
    sp = flow_semantic_precedence(df, use, sp_threshold)

    if use == "id":
        conversion_map = _create_conversion_map(df)
        L = [conversion_map[l] for l in L]
        cf = [{conversion_map[l] for l in c} for c in cf]

    close_modes: list[str] = []
    if CARD_EVENT_TYPE in df.columns:
        if (df[CARD_EVENT_TYPE] == "card_close").any():
            close_modes.append("archived")
        if (df[CARD_EVENT_TYPE] == "card_delete").any():
            close_modes.append("deleted")
    if not close_modes:
        close_modes = ["archived"]
    close_mode: str | list[str] = close_modes[0] if len(close_modes) == 1 else close_modes

    return BoardModel(
        lists=L,
        card_flow=cf,
        semantic_precedence=sp,
        card_create_lists=list(cc.index),
        card_close_lists=list(cx.index),
        card_use_lists=list(cu.index),
        close_mode=close_mode,
    )


def plot_board_model(
    model: "BoardModel",
    ax=None,
    figsize: tuple = (15, 5),
    box_w: float = 1.6,
    box_h: float = 1.1,
    col_sep: float = 0.7,
) -> matplotlib.axes.Axes:
    """Draw a board model diagram from a :class:`BoardModel`.

    Provides a graphical alternative to :func:`print_board_discovery` using
    the notation from board-based collaboration design patterns:

    - Each list is shown as a labelled box with an underlined name.
    - Role icons in the lower section of each box indicate the list's function:
      ``⊕`` (cards created here), ``↻`` (cards updated here),
      ``⊠`` (cards closed here).
    - Dashed arrows between boxes represent semantic precedence relationships.
      Forward arrows arc above the boxes; backward arrows arc below.
    - A shaded rectangle groups all lists that belong to the same card-flow
      connected component, marked with a ``✱`` symbol below it.

    Parameters
    ----------
    model:
        Board model returned by :func:`board_discovery`.
    ax:
        Matplotlib Axes to draw on.  A new figure is created when ``None``.
    figsize:
        Figure size when *ax* is ``None``.  The figure is widened automatically
        when the number of list columns exceeds the default width.
    box_w, box_h:
        Width and height of each list box in data units.
    col_sep:
        Horizontal gap between consecutive columns.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import textwrap
    from collections import defaultdict
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    lists = model.lists
    sp_edges = [
        (s, t) for s, t in model.semantic_precedence
        if s in lists and t in lists
    ]
    create_set = set(model.card_create_lists)
    close_set  = set(model.card_close_lists)
    use_set    = set(model.card_use_lists)

    if not lists:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        return ax

    # ── Topological layout ────────────────────────────────────────────────────
    G = nx.DiGraph()
    G.add_nodes_from(lists)
    G.add_edges_from(sp_edges)

    # Remove back-edges so we can run topological sort
    G_dag = G.copy()
    try:
        while True:
            cycle = nx.find_cycle(G_dag, orientation="original")
            G_dag.remove_edge(cycle[-1][0], cycle[-1][1])
    except nx.NetworkXNoCycle:
        pass

    # Longest-path level assignment (left-to-right column index)
    col = {l: 0 for l in lists}
    for node in nx.topological_sort(G_dag):
        for succ in G_dag.successors(node):
            col[succ] = max(col[succ], col[node] + 1)

    # Group lists by column, preserving input order within each column
    col_groups: dict[int, list[str]] = defaultdict(list)
    for l in lists:
        col_groups[col[l]].append(l)

    # Assign pixel-space (x, y) box origins
    row_sep = box_h + 0.6
    pos: dict[str, tuple[float, float]] = {}
    for lv, group in sorted(col_groups.items()):
        x = lv * (box_w + col_sep)
        for row, l in enumerate(group):
            pos[l] = (x, -row * row_sep)

    # ── Figure setup ─────────────────────────────────────────────────────────
    if ax is None:
        n_cols  = (max(col_groups) + 1) if col_groups else 1
        max_rows = max(len(g) for g in col_groups.values())
        fw = max(figsize[0], n_cols * (box_w + col_sep) + 1.5)
        fh = max(figsize[1], max_rows * row_sep + 2.5)
        _, ax = plt.subplots(figsize=(fw, fh))

    ax.set_aspect("equal")
    ax.axis("off")

    # ── Card-flow component backgrounds ───────────────────────────────────────
    cf_palette = matplotlib.colormaps["Set3"](
        np.linspace(0.1, 0.9, max(1, len(model.card_flow)))
    )
    for cf_comp, color in zip(model.card_flow, cf_palette):
        in_pos = [l for l in cf_comp if l in pos]
        if not in_pos:
            continue
        xs  = [pos[l][0] for l in in_pos]
        ys  = [pos[l][1] for l in in_pos]
        pad = 0.3
        ax.add_patch(mpatches.FancyBboxPatch(
            (min(xs) - pad, min(ys) - pad),
            max(xs) - min(xs) + box_w + 2 * pad,
            max(ys) - min(ys) + box_h + 2 * pad,
            boxstyle="round,pad=0.05",
            facecolor=(*color[:3], 0.18),
            edgecolor=(*color[:3], 0.55),
            linewidth=1.5, linestyle="dashed", zorder=0,
        ))
        # Card-flow pattern symbol below each component
        mid_x = (min(xs) + max(xs) + box_w) / 2
        ax.text(
            mid_x, min(ys) - pad - 0.05, "✱",
            ha="center", va="top", fontsize=14,
            color=(*color[:3], 0.65), zorder=1,
        )

    # ── Semantic precedence arrows ─────────────────────────────────────────────
    for src, tgt in sp_edges:
        if src not in pos or tgt not in pos:
            continue
        sx, sy = pos[src]
        tx, ty = pos[tgt]

        if col[src] == col[tgt]:
            # Same column: connect top-centres with an arc to the side
            p_start = (sx + box_w / 2, sy + box_h)
            p_end   = (tx + box_w / 2, ty + box_h)
            rad = -0.55
        else:
            # Connect right-centre of src to left-centre of tgt
            p_start = (sx + box_w, sy + box_h / 2)
            p_end   = (tx,         ty + box_h / 2)
            span = col[tgt] - col[src]
            # Forward (positive span): arc above; backward: arc below
            rad = -(0.18 + span * 0.06) if span > 0 else (0.25 + abs(span) * 0.07)

        ax.add_patch(mpatches.FancyArrowPatch(
            p_start, p_end,
            connectionstyle=f"arc3,rad={rad:.2f}",
            arrowstyle="-|>",
            linestyle="dashed",
            color="black", linewidth=1.1,
            mutation_scale=9, zorder=2,
        ))

    # ── List boxes ─────────────────────────────────────────────────────────────
    for l in lists:
        if l not in pos:
            continue
        x, y = pos[l]

        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="square,pad=0",
            facecolor="white", edgecolor="black",
            linewidth=1.5, zorder=3,
        ))

        # Name in upper section (underlined via a divider line)
        name_text = "\n".join(textwrap.wrap(str(l), width=14)[:2])
        ax.text(
            x + box_w / 2, y + box_h * 0.73, name_text,
            ha="center", va="center", fontsize=7, fontweight="bold",
            multialignment="center", zorder=4,
        )
        ax.plot(
            [x + 0.07, x + box_w - 0.07],
            [y + box_h * 0.52] * 2,
            color="black", linewidth=0.6, zorder=4,
        )

        # Role icons in lower section
        icons = []
        if l in create_set:
            icons.append("⊕")   # entry: cards are created here
        if l in use_set:
            icons.append("↻")   # cycle: cards are updated here
        if l in close_set:
            icons.append("⊠")   # exit: cards are closed here
        ax.text(
            x + box_w / 2, y + box_h * 0.23, "  ".join(icons),
            ha="center", va="center", fontsize=10, zorder=4,
        )

    # ── Axis limits ───────────────────────────────────────────────────────────
    all_xs = [p[0] for p in pos.values()]
    all_ys = [p[1] for p in pos.values()]
    margin = 1.3
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + box_w + margin)
    ax.set_ylim(min(all_ys) - margin - 0.6, max(all_ys) + box_h + margin)

    return ax


def print_board_discovery(model: BoardModel) -> None:
    """Print a :class:`BoardModel` in a human-readable format.

    Parameters
    ----------
    model:
        Board model returned by :func:`board_discovery`.
    """
    def _print_names(names: list[str]) -> None:
        if names:
            for n in names:
                print("- ", n)
        else:
            print("- None")

    print("Lists: ", model.lists)
    print("Close mode: ", model.close_mode)
    print("Allowed flow mode: ", model.allowed_flow_mode)
    print("")
    print("# Card flow:")
    for i, component in enumerate(model.card_flow):
        print("- ", i, ", ".join(sorted(component)))
    print("")
    print("# Creation lists:")
    _print_names(model.card_create_lists)
    print("")
    print("# Close lists:")
    _print_names(model.card_close_lists)
    print("")
    print("# Use lists:")
    _print_names(model.card_use_lists)
    print("")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def redesign_metrics(df: pd.DataFrame, redesigns: pd.DataFrame) -> dict:
    """Compute aggregate statistics about detected redesign periods.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    redesigns:
        Output of :func:`detect_redesign`.

    Returns
    -------
    dict
        Keys: ``redesigns`` (descriptive stats on event counts per redesign),
        ``redesign_distance`` (descriptive stats on gaps between consecutive
        redesigns), ``redesign_distance_meandays`` (mean gap in days as a
        float).
    """
    info = {}
    info["redesigns"] = redesigns["count"].describe().to_dict()
    info["redesign_distance"] = (
        redesigns["min"] - redesigns["max"].shift(-1, fill_value=df[TIMESTAMP].min())
    ).describe().to_dict()
    info["redesign_distance_meandays"] = info["redesign_distance"]["mean"] / pd.Timedelta("1D")
    return info


def list_evolution_metrics(df: pd.DataFrame, evolution: pd.DataFrame) -> dict:
    """Compute aggregate statistics about list lifetimes.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    evolution:
        Output of :func:`list_evolution`.

    Returns
    -------
    dict
        Keys: ``list_duration`` (mean active duration as a Timedelta),
        ``list_duration_perc`` (descriptive stats on duration as a fraction of
        the total board lifetime), ``list_renames_mean`` (mean number of names
        a list had).
    """
    info = {}
    board_duration = df[TIMESTAMP].max() - df[TIMESTAMP].min()
    info["list_duration"] = (evolution["last_date"] - evolution["begin_date"]).mean()
    info["list_duration_perc"] = (
        (evolution["last_date"] - evolution["begin_date"]) / board_duration
    ).describe().to_dict()
    info["list_renames_mean"] = evolution[LIST_NAME].apply(len).mean()
    return info


def connected_metrics(df: pd.DataFrame, connected: pd.DataFrame) -> dict:
    """Compute aggregate statistics about the list connectivity graph.

    Parameters
    ----------
    df:
        Board event log in the bomi schema (used only for shape information).
    connected:
        Output of :func:`connected_lists`.

    Returns
    -------
    dict
        Keys: ``list_num_components``, ``list_connected_size_mean``,
        ``list_connected_size_mean_perc`` (mean component size as a fraction
        of all lists), ``list_num_components_move`` (components with at least
        one card movement).
    """
    info = {}
    info["list_num_components"] = len(connected)
    info["list_connected_size_mean"] = connected["size"].mean()
    info["list_connected_size_mean_perc"] = info["list_connected_size_mean"] / connected["size"].sum()
    info["list_num_components_move"] = (connected["count"] > 0).sum()
    return info


def move_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics about card movements.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    dict
        Keys: ``move_per_list_with_move``, ``list_with_move_perc``,
        ``cards_moving_perc``, ``moves_per_moving_card``.
    """
    info = {}
    if LIST_ID in df.columns:
        num_lists = pd.concat([df[LIST_ID], df[SOURCE_LIST_ID], df[TARGET_LIST_ID]]).nunique()
        moves = (
            df[card_movement_filter(df)].groupby(SOURCE_LIST_ID)[EVENT_ID].count()
            .add(df[card_movement_filter(df)].groupby(TARGET_LIST_ID)[EVENT_ID].count(), fill_value=0)
        )
        info["move_per_list_with_move"] = moves.mean()
        info["list_with_move_perc"] = len(moves) / num_lists
    if CARD_ID in df.columns:
        cards = df[CARD_ID].nunique()
        info["cards_moving_perc"] = df[card_movement_filter(df)][CARD_ID].nunique() / cards
        info["moves_per_moving_card"] = df[card_movement_filter(df)].groupby(CARD_ID)[EVENT_ID].count().mean()
    return info


def act_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics about card-action events.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    dict
        Keys: ``act_per_list`` (mean number of actions per list),
        ``cards_act_perc`` (fraction of cards with at least one action),
        ``act_per_act_card`` (mean actions per card that had at least one).
    """
    info = {}
    if LIST_ID in df.columns:
        info["act_per_list"] = df[card_action_filter(df)].groupby(LIST_ID)[EVENT_ID].count().mean()
    if CARD_ID in df.columns:
        cards = df[CARD_ID].nunique()
        info["cards_act_perc"] = df[card_action_filter(df)][CARD_ID].nunique() / cards
        info["act_per_act_card"] = df[card_action_filter(df)].groupby(CARD_ID)[EVENT_ID].count().mean()
    return info


def close_metrics(df: pd.DataFrame) -> dict:
    """Compute the fraction of cards that were closed.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.

    Returns
    -------
    dict
        Key: ``cards_closed_perc``.
    """
    info = {}
    if CARD_ID in df.columns:
        cards = df[CARD_ID].nunique()
        info["cards_closed_perc"] = df[card_closed_filter(df)][CARD_ID].nunique() / cards
    return info


def static_metrics(
    df: pd.DataFrame,
    redesigns: pd.DataFrame | None = None,
    time_threshold=None,
    event_threshold: int | None = None,
    **kwargs,
) -> pd.DataFrame | dict:
    """Compute board-use metrics for the whole board or per stable period.

    When *redesigns* is ``None``, returns a single metrics dict for the
    complete event log.  When *redesigns* is provided (output of
    :func:`detect_redesign`), the log is split into the stable periods
    *between* redesigns and one row of metrics is computed per period.

    Parameters
    ----------
    df:
        Board event log in the bomi schema.
    redesigns:
        Output of :func:`detect_redesign`.  When given, metrics are computed
        for each stable period between consecutive redesigns.
    time_threshold:
        Optional :class:`pandas.Timedelta`; stable periods shorter than this
        value are skipped.
    event_threshold:
        Optional int; stable periods with fewer events than this value are
        skipped.
    **kwargs:
        Extra keyword arguments forwarded to :func:`connected_lists`
        (e.g. ``use``, ``threshold``).

    Returns
    -------
    dict or pandas.DataFrame
        A single dict when *redesigns* is ``None``, or a DataFrame with one
        row per stable period indexed by a :class:`pandas.IntervalIndex`
        (start, end timestamps).
    """
    if redesigns is None:
        connected = connected_lists(df, **kwargs)
        return _compute_static_metrics(df, connected)

    info = []
    index = []
    first = redesigns["max"].values
    last = redesigns["min"].shift(1, fill_value=df[TIMESTAMP].max()).values

    for f, l in zip(first, last):
        if time_threshold is not None and l - f < time_threshold:
            continue
        filtered_df = df[(df[TIMESTAMP].values >= f) & (df[TIMESTAMP].values <= l)]
        if event_threshold is not None and len(filtered_df) < event_threshold:
            continue
        filtered_conn = connected_lists(filtered_df, **kwargs)
        info.append({
            "events": len(filtered_df),
            "cards": filtered_df[CARD_ID].nunique(),
            "lists": filtered_df[LIST_ID].nunique(),
            **_compute_static_metrics(filtered_df, filtered_conn),
        })
        index.append((f, l))

    if index:
        records = pd.json_normalize(info)
        records.index = pd.IntervalIndex.from_tuples(index)
        return records
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_static_metrics(df: pd.DataFrame, connected: pd.DataFrame) -> dict:
    return {
        **connected_metrics(df, connected),
        **move_metrics(df),
        **act_metrics(df),
        **close_metrics(df),
    }


def _create_conversion_map(df: pd.DataFrame) -> dict:
    """Build a list-ID → list-name mapping from all list columns in *df*."""
    return {
        **df.groupby(LIST_ID)[LIST_NAME].first().to_dict(),
        **df.groupby(TARGET_LIST_ID)[TARGET_LIST_NAME].first().to_dict(),
        **df.groupby(SOURCE_LIST_ID)[SOURCE_LIST_NAME].first().to_dict(),
    }



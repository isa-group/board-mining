"""Flow endpoints: transition matrix, connected lists, dominant flow."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import networkx as nx
import bomi
from fastapi import APIRouter, Query
from bomi.schema import TIMESTAMP
from .. import loader

router = APIRouter(prefix="/api/flow", tags=["flow"])


@router.get("/transition-matrix")
async def transition_matrix(
    start_date: Optional[str] = Query(None, description="ISO 8601 start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="ISO 8601 end date (YYYY-MM-DD)"),
    sort_by: str = Query("net_flow", description="Sorting method: 'net_flow' or 'topological'"),
):
    df = loader.get_board()

    # Apply time window if provided (ensure timezone-aware comparison)
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        df = df[df[TIMESTAMP] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        df = df[df[TIMESTAMP] <= end_ts]
    raw = bomi.transition_matrix(df).fillna(0)

    # Square the matrix over the union of all list names so every list
    # appears in both axes before we compute ordering.
    all_lists = sorted(set(raw.index) | set(raw.columns))
    matrix = raw.reindex(index=all_lists, columns=all_lists, fill_value=0)

    # Compute ordering based on sort_by parameter
    if sort_by == "topological":
        # Build directed graph and perform topological sort
        G = nx.DiGraph()
        for i, source in enumerate(matrix.index):
            for j, target in enumerate(matrix.columns):
                if matrix.iloc[i, j] > 0:
                    G.add_edge(source, target, weight=matrix.iloc[i, j])

        # Add isolated nodes
        for list_name in matrix.index:
            if list_name not in G:
                G.add_node(list_name)

        # Remove back-edges to create DAG for topological sort
        G_dag = G.copy()
        try:
            while True:
                cycle = nx.find_cycle(G_dag, orientation="original")
                G_dag.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            pass

        # Perform topological sort
        order = list(nx.topological_sort(G_dag))
    else:
        # Default: Net flow sorting
        # Net flow = cards sent out (row sum) − cards received (col sum).
        # Positive → source/early-stage list; negative → sink/completion list.
        # Sorting descending puts sources at the top, sinks at the bottom.
        net_flow = matrix.sum(axis=1) - matrix.sum(axis=0)
        order = net_flow.sort_values(ascending=False).index.tolist()

    matrix = matrix.loc[order, order]

    return {
        "row_lists": matrix.index.tolist(),
        "col_lists": matrix.columns.tolist(),
        "matrix": matrix.values.tolist(),
    }


@router.get("/connected-lists")
async def connected_lists():
    df = loader.get_board()
    result = bomi.connected_lists(df)
    # result is a DataFrame with columns: component (set), size, count
    return [
        {"component": sorted(row["component"]), "size": int(row["size"]), "count": int(row["count"])}
        for _, row in result.iterrows()
    ]


@router.get("/semantic-precedence")
async def semantic_precedence():
    df = loader.get_board()
    # returns a list of (source, target) tuples
    pairs = bomi.flow_semantic_precedence(df)
    return [{"source": s, "target": t} for s, t in pairs]


@router.get("/board-design")
async def board_design(
    start_date: Optional[str] = Query(None, description="ISO 8601 start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="ISO 8601 end date (YYYY-MM-DD)"),
    cf_threshold: float = Query(0, description="Card flow percentage threshold (0-100): include edges with movement count >= max * threshold%"),
    cc_threshold: float = Query(0, description="Card creation percentage threshold (0-100): include lists with creation count >= max * threshold%"),
    cx_threshold: float = Query(0, description="Card close percentage threshold (0-100): include lists with close count >= max * threshold%"),
    cu_threshold: float = Query(0, description="Card use percentage threshold (0-100): include lists with action count >= max * threshold%"),
    sp_threshold: float = Query(0, description="Semantic precedence percentage threshold (0-100): include pairs with movement count >= max * threshold%"),
):
    df = loader.get_board()

    # Apply time window if provided
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        df = df[df[TIMESTAMP] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        df = df[df[TIMESTAMP] <= end_ts]

    board = bomi.board_discovery(
        df,
        cf_threshold=cf_threshold,
        cc_threshold=cc_threshold,
        cx_threshold=cx_threshold,
        cu_threshold=cu_threshold,
        sp_threshold=sp_threshold,
    )

    return {
        "lists": board.lists,
        "card_flow": [sorted(list(component)) for component in board.card_flow],
        "semantic_precedence": [{"source": s, "target": t, "volume": 0} for s, t in board.semantic_precedence],
        "card_create_lists": board.card_create_lists,
        "card_close_lists": board.card_close_lists,
        "card_use_lists": board.card_use_lists,
        "close_mode": board.close_mode if isinstance(board.close_mode, list) else [board.close_mode],
    }

import pandas as pd
import pytest

import bomi


def sample_log() -> pd.DataFrame:
    """Minimal board event log already in the bomi schema with event types computed."""
    rows = [
        {
            "event_id": "e1",
            "event_type": "createList",
            "raw_event_type": "createList",
            "timestamp": pd.Timestamp("2024-01-01T09:00:00Z"),
            "actor_id": "u1",
            "board_id": "b1",
            "board_name": "Test Board",
            "list_id": "l1",
            "list_name": "Backlog",
            "list_event_type": "list_create",
            "card_event_type": pd.NA,
        },
        {
            "event_id": "e2",
            "event_type": "createList",
            "raw_event_type": "createList",
            "timestamp": pd.Timestamp("2024-01-01T09:05:00Z"),
            "actor_id": "u1",
            "board_id": "b1",
            "board_name": "Test Board",
            "list_id": "l2",
            "list_name": "Done",
            "list_event_type": "list_create",
            "card_event_type": pd.NA,
        },
        {
            "event_id": "e3",
            "event_type": "createCard",
            "raw_event_type": "createCard",
            "timestamp": pd.Timestamp("2024-01-02T10:00:00Z"),
            "actor_id": "u2",
            "board_id": "b1",
            "board_name": "Test Board",
            "card_id": "c1",
            "card_closed": False,
            "list_id": "l1",
            "list_name": "Backlog",
            "list_event_type": pd.NA,
            "card_event_type": "card_create",
        },
        {
            "event_id": "e4",
            "event_type": "updateCard",
            "raw_event_type": "updateCard",
            "timestamp": pd.Timestamp("2024-01-03T10:00:00Z"),
            "actor_id": "u2",
            "board_id": "b1",
            "board_name": "Test Board",
            "card_id": "c1",
            "card_closed": False,
            "list_id": "l1",
            "list_name": "Backlog",
            "source_list_id": "l1",
            "source_list_name": "Backlog",
            "target_list_id": "l2",
            "target_list_name": "Done",
            "list_event_type": pd.NA,
            "card_event_type": "card_move",
        },
        {
            "event_id": "e5",
            "event_type": "updateCard",
            "raw_event_type": "updateCard",
            "timestamp": pd.Timestamp("2024-01-04T10:00:00Z"),
            "actor_id": "u2",
            "board_id": "b1",
            "board_name": "Test Board",
            "card_id": "c1",
            "card_closed": True,
            "list_id": "l2",
            "list_name": "Done",
            "list_event_type": pd.NA,
            "card_event_type": "card_close",
        },
    ]
    return pd.DataFrame(rows)


def test_board_discovery_returns_board_model():
    df = sample_log()

    model = bomi.board_discovery(df, use="id")

    assert isinstance(model, bomi.BoardModel)
    assert set(model.lists) == {"Backlog", "Done"}
    assert model.card_flow == [{"Backlog", "Done"}]
    assert model.semantic_precedence == [("Backlog", "Done")]
    assert model.close_mode == "archived"
    assert model.allowed_flow_mode == "semantic_precedence"
    assert model.list_constraints == {}


def test_board_model_allowed_flow_modes():
    df = sample_log()
    model = bomi.board_discovery(df, use="id")

    assert model.allowed_flow == [("Backlog", "Done")]

    model.allowed_flow_mode = "card_flow"
    flow_set = set(model.allowed_flow)
    assert ("Backlog", "Done") in flow_set
    assert ("Done", "Backlog") in flow_set


def test_static_metrics_reports_movement_and_close_rates():
    df = sample_log()

    metrics = bomi.static_metrics(df, use="names")

    assert metrics["list_num_components"] == 1
    assert metrics["cards_moving_perc"] == 1
    assert metrics["cards_closed_perc"] == 1


def test_to_event_log_reports_missing_optional_dependency():
    if bomi.pm4py is not None:
        pytest.skip("pm4py is installed in this environment")

    df = sample_log()

    with pytest.raises(ImportError, match="pm4py"):
        bomi.to_event_log(df)


# ---------------------------------------------------------------------------
# plot_board_model
# ---------------------------------------------------------------------------

def test_plot_board_model_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    df = sample_log()
    model = bomi.board_discovery(df, use="id")
    ax = bomi.plot_board_model(model)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_board_model_draws_one_box_per_list():
    import matplotlib
    matplotlib.use("Agg")
    df = sample_log()
    model = bomi.board_discovery(df, use="id")
    ax = bomi.plot_board_model(model)
    # FancyBboxPatch for each list box plus the component background(s)
    from matplotlib.patches import FancyBboxPatch
    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    n_lists = len(model.lists)
    n_components = len(model.card_flow)
    assert len(boxes) == n_lists + n_components


def test_plot_board_model_draws_arrows_for_semantic_precedence():
    import matplotlib
    matplotlib.use("Agg")
    df = sample_log()
    model = bomi.board_discovery(df, use="id")
    ax = bomi.plot_board_model(model)
    from matplotlib.patches import FancyArrowPatch
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == len(model.semantic_precedence)


def test_plot_board_model_handles_empty_model():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bomi.core import BoardModel
    empty = BoardModel(
        lists=[], card_flow=[], semantic_precedence=[],
        card_create_lists=[], card_close_lists=[], card_use_lists=[],
        close_mode="archived",
    )
    fig, ax = plt.subplots()
    result = bomi.plot_board_model(empty, ax=ax)
    assert result is ax

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


# ---------------------------------------------------------------------------
# Tests for relative threshold semantics
# ---------------------------------------------------------------------------

def test_apply_relative_threshold_with_percentages():
    """Test that relative thresholds filter by percentage of maximum value."""
    from bomi.core import _apply_relative_threshold

    series = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

    # 0% threshold: include all
    mask = _apply_relative_threshold(series, 0)
    assert mask.all()

    # 50% threshold: include values >= 40 * 0.5 = 20
    mask = _apply_relative_threshold(series, 50)
    assert list(series[mask].index) == ['b', 'c', 'd']

    # 100% threshold: include only max value
    mask = _apply_relative_threshold(series, 100)
    assert list(series[mask].index) == ['d']


def test_apply_relative_threshold_zero_max():
    """Test that zero max returns all True (graceful degradation)."""
    from bomi.core import _apply_relative_threshold

    series = pd.Series([0, 0, 0], index=['a', 'b', 'c'])

    # When max is 0, include everything
    mask = _apply_relative_threshold(series, 50)
    assert mask.all()


def test_apply_relative_threshold_invalid_range():
    """Test that invalid threshold values raise ValueError."""
    from bomi.core import _apply_relative_threshold

    series = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        _apply_relative_threshold(series, -1)

    with pytest.raises(ValueError):
        _apply_relative_threshold(series, 101)


def test_connected_lists_with_relative_threshold():
    """Test connected_lists uses relative threshold semantics."""
    df = sample_log()

    # With 0% threshold, all edges are included
    result_0 = bomi.connected_lists(df, threshold=0)
    assert len(result_0) > 0

    # With 100% threshold, only the most frequent edge is included
    # This may result in fewer or same number of components
    result_100 = bomi.connected_lists(df, threshold=100)
    assert len(result_100) <= len(result_0)


def test_card_action_list_with_relative_threshold():
    """Test card_action_list uses relative threshold semantics."""
    df = sample_log()

    # With 0% threshold, all lists are included
    result_0 = bomi.card_action_list(df, type="card_create", threshold=0)
    assert len(result_0) > 0

    # With 100% threshold, only the list with max creates is included
    result_100 = bomi.card_action_list(df, type="card_create", threshold=100)
    assert len(result_100) <= len(result_0)

    # Result should still sum to 1 (it's normalized)
    if len(result_100) > 0:
        assert abs(result_100.sum() - 1.0) < 1e-10


def test_flow_semantic_precedence_with_relative_threshold():
    """Test flow_semantic_precedence uses relative threshold semantics."""
    df = sample_log()

    # With 0% threshold, all transitions are included
    result_0 = bomi.flow_semantic_precedence(df, threshold=0)
    assert len(result_0) > 0

    # With 100% threshold, only the most frequent transition is included
    result_100 = bomi.flow_semantic_precedence(df, threshold=100)
    assert len(result_100) <= len(result_0)

    # 50% threshold should be between 0% and 100%
    result_50 = bomi.flow_semantic_precedence(df, threshold=50)
    assert len(result_100) <= len(result_50) <= len(result_0)


def test_board_discovery_with_relative_thresholds():
    """Test board_discovery combines all relative thresholds correctly."""
    df = sample_log()

    # All thresholds at 0% should give maximum structure
    board_0 = bomi.board_discovery(
        df,
        cf_threshold=0,
        cc_threshold=0,
        cx_threshold=0,
        cu_threshold=0,
        sp_threshold=0,
    )

    # All thresholds at 100% should give minimal structure
    board_100 = bomi.board_discovery(
        df,
        cf_threshold=100,
        cc_threshold=100,
        cx_threshold=100,
        cu_threshold=100,
        sp_threshold=100,
    )

    # More lenient thresholds should result in more semantic precedence pairs
    assert len(board_0.semantic_precedence) >= len(board_100.semantic_precedence)

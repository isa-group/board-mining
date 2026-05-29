import pandas as pd
import pytest

import bomi
from test_core import sample_log


def test_schema_columns_includes_classification_columns():
    cols = bomi.schema_columns()

    assert "list_event_type" in cols
    assert "card_event_type" in cols
    assert "source_system" in cols


def test_to_board_log_validates_required_columns():
    df = sample_log()

    board_log = bomi.to_board_log(df)

    assert "event_id" in board_log.columns
    assert "event_type" in board_log.columns
    assert "timestamp" in board_log.columns
    assert board_log["timestamp"].dtype.kind == "M"


def test_to_board_log_sets_source_system():
    df = sample_log()

    board_log = bomi.to_board_log(df, source_system="test")

    assert (board_log["source_system"] == "test").all()


def test_to_board_log_does_not_override_existing_source_system():
    df = sample_log().assign(source_system="already_set")

    board_log = bomi.to_board_log(df, source_system="other")

    assert (board_log["source_system"] == "already_set").all()


def test_validate_board_log_reports_recommended_missing_columns():
    minimal = pd.DataFrame({
        "event_id": ["e1"],
        "event_type": ["createCard"],
        "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")],
    })

    report = bomi.validate_board_log(minimal)

    assert report.is_valid
    assert report.missing_required == ()
    assert "actor_id" in report.missing_recommended


def test_validate_board_log_raises_for_null_required_fields():
    invalid = pd.DataFrame({
        "event_id": ["e1"],
        "event_type": [None],
        "timestamp": ["not-a-date"],
    })

    with pytest.raises(bomi.SchemaValidationError, match="null values"):
        bomi.validate_board_log(invalid, raise_on_error=True)


def test_compute_event_types_classifies_list_and_card_events():
    df = pd.DataFrame([
        {"event_id": "e1", "event_type": "createList", "raw_event_type": "createList",
         "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"), "list_id": "l1"},
        {"event_id": "e2", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": pd.Timestamp("2024-01-02T00:00:00Z"), "card_id": "c1"},
        {"event_id": "e3", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": pd.Timestamp("2024-01-03T00:00:00Z"),
         "card_id": "c1", "source_list_id": "l1", "target_list_id": "l2"},
        {"event_id": "e4", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": pd.Timestamp("2024-01-04T00:00:00Z"),
         "card_id": "c1", "card_closed": True},
    ])

    result = bomi.compute_event_types(df)

    assert result.loc[result["event_id"] == "e1", "list_event_type"].item() == "list_create"
    assert result.loc[result["event_id"] == "e2", "card_event_type"].item() == "card_create"
    assert result.loc[result["event_id"] == "e3", "card_event_type"].item() == "card_move"
    assert result.loc[result["event_id"] == "e4", "card_event_type"].item() == "card_close"


def test_compute_event_types_list_rename_takes_priority_over_list_change():
    df = pd.DataFrame([
        {"event_id": "e1", "event_type": "updateList", "raw_event_type": "updateList",
         "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
         "list_id": "l1", "old_name": "Former Name"},
    ])

    result = bomi.compute_event_types(df)

    assert result.loc[0, "list_event_type"] == "list_rename"

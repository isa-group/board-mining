"""Tests for the Jira connector."""

from __future__ import annotations

from typing import Any

import bomi
from bomi.connectors.jira import load_jira_board


# ---------------------------------------------------------------------------
# Mock HTTP infrastructure
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, data: Any) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._data


class _MockSession:
    """Dispatch GET requests to fixed response data based on URL substring match."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> _MockResponse:
        self.calls.append(url)
        for pattern, data in self._responses.items():
            if pattern in url:
                return _MockResponse(data)
        raise ValueError(f"Unexpected URL in mock: {url}")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _board_config(name: str = "My Board") -> dict[str, Any]:
    return {"id": 1, "name": name}


def _project_statuses(done_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": "10001",
            "statuses": [
                {
                    "id": sid,
                    "name": f"Status-{sid}",
                    "statusCategory": {
                        "key": "done" if sid in done_ids else "indeterminate"
                    },
                }
                for sid in ["s1", "s2", "s3"]
            ],
        }
    ]


def _issue(
    *,
    key: str = "PROJ-1",
    issue_id: str = "10001",
    summary: str = "An issue",
    status_id: str = "s1",
    status_name: str = "To Do",
    creator_id: str = "user-1",
    created: str = "2024-01-01T10:00:00.000+0000",
    duedate: str | None = None,
    histories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": summary,
            "created": created,
            "creator": {"accountId": creator_id},
            "status": {"id": status_id, "name": status_name},
            "project": {"key": "PROJ", "name": "Project"},
            "duedate": duedate,
            "resolution": None,
        },
        "changelog": {
            "total": len(histories or []),
            "maxResults": 100,
            "histories": histories or [],
        },
    }


def _history(
    *,
    history_id: str = "h1",
    author_id: str = "user-2",
    created: str = "2024-01-05T10:00:00.000+0000",
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": history_id,
        "author": {"accountId": author_id},
        "created": created,
        "items": items,
    }


def _status_item(from_id: str, from_name: str, to_id: str, to_name: str) -> dict[str, Any]:
    return {"field": "status", "from": from_id, "fromString": from_name, "to": to_id, "toString": to_name}


def _summary_item(from_name: str, to_name: str) -> dict[str, Any]:
    return {"field": "summary", "fromString": from_name, "toString": to_name}


def _resolution_item(to_id: str | None) -> dict[str, Any]:
    return {"field": "resolution", "from": None, "fromString": None, "to": to_id, "toString": "Done" if to_id else None}


def _make_session(issues: list[dict[str, Any]], done_ids: list[str] | None = None) -> _MockSession:
    return _MockSession({
        "/rest/agile/1.0/board/1/configuration": _board_config(),
        "/rest/api/3/project/PROJ/statuses": _project_statuses(done_ids or []),
        "/rest/agile/1.0/board/1/issues": {"total": len(issues), "issues": issues},
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_jira_board_returns_schema_columns():
    session = _make_session([_issue()])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    for col in bomi.schema_columns():
        assert col in df.columns, f"missing schema column: {col}"
    assert (df["source_system"] == "jira").all()


def test_load_jira_board_accessible_from_bomi_namespace():
    assert hasattr(bomi, "load_jira_board")


def test_load_jira_board_computes_card_event_types():
    move = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s2", "In Progress")],
    )
    session = _make_session([_issue(histories=[move])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    assert df[df["event_type"] == "createCard"].iloc[0]["card_event_type"] == "card_create"
    assert df[df["event_id"] == "h1-0"].iloc[0]["card_event_type"] == "card_move"


def test_load_jira_board_produces_create_event():
    session = _make_session([_issue(key="PROJ-1", creator_id="u1", created="2024-01-01T10:00:00.000+0000")])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    create_rows = df[df["event_type"] == "createCard"]
    assert len(create_rows) == 1
    row = create_rows.iloc[0]
    assert row["card_id"] == "PROJ-1"
    assert row["actor_id"] == "u1"
    assert row["board_name"] == "My Board"
    assert row["raw_event_type"] == "createCard"


def test_load_jira_board_status_change_produces_move_event():
    move = _history(
        history_id="h1",
        author_id="u2",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s2", "In Progress")],
    )
    session = _make_session([_issue(histories=[move])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    row = df[df["event_id"] == "h1-0"].iloc[0]
    assert row["source_list_name"] == "To Do"
    assert row["target_list_name"] == "In Progress"
    assert row["list_name"] == "To Do"
    assert row["raw_event_type"] == "status"


def test_load_jira_board_closed_method_done_status():
    move_to_done = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s3", "Done")],
    )
    session = _make_session([_issue(histories=[move_to_done])], done_ids=["s3"])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method="done_status")

    create_row = df[df["event_type"] == "createCard"].iloc[0]
    move_row = df[df["event_id"] == "h1-0"].iloc[0]
    assert create_row["card_closed"] == False
    assert move_row["card_closed"] == True


def test_load_jira_board_closed_method_done_status_back_to_false_on_reopen():
    move_to_done = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s3", "Done")],
    )
    reopen = _history(
        history_id="h2",
        created="2024-01-06T10:00:00.000+0000",
        items=[_status_item("s3", "Done", "s2", "In Progress")],
    )
    session = _make_session([_issue(status_id="s2", status_name="In Progress", histories=[move_to_done, reopen])], done_ids=["s3"])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method="done_status")

    assert df[df["event_id"] == "h1-0"].iloc[0]["card_closed"] == True
    assert df[df["event_id"] == "h2-0"].iloc[0]["card_closed"] == False


def test_load_jira_board_closed_method_resolution():
    resolve = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_resolution_item("10000")],
    )
    session = _make_session([_issue(histories=[resolve])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method="resolution")

    assert df[df["event_type"] == "createCard"].iloc[0]["card_closed"] == False
    assert df[df["event_id"] == "h1-0"].iloc[0]["card_closed"] == True


def test_load_jira_board_closed_method_resolution_cleared():
    resolve = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_resolution_item("10000")],
    )
    unresolve = _history(
        history_id="h2",
        created="2024-01-06T10:00:00.000+0000",
        items=[_resolution_item(None)],
    )
    session = _make_session([_issue(histories=[resolve, unresolve])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method="resolution")

    assert df[df["event_id"] == "h1-0"].iloc[0]["card_closed"] == True
    assert df[df["event_id"] == "h2-0"].iloc[0]["card_closed"] == False


def test_load_jira_board_closed_method_none_always_false():
    move_to_done = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s3", "Done")],
    )
    resolve = _history(
        history_id="h2",
        created="2024-01-06T10:00:00.000+0000",
        items=[_resolution_item("10000")],
    )
    session = _make_session([_issue(histories=[move_to_done, resolve])], done_ids=["s3"])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    assert (df["card_closed"] == False).all()


def test_load_jira_board_closed_method_false_same_as_none():
    session = _make_session([_issue()])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=False)

    assert (df["card_closed"] == False).all()


def test_load_jira_board_fills_list_for_non_movement_events():
    status_change = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s2", "In Progress")],
    )
    summary_change = _history(
        history_id="h2",
        created="2024-01-06T10:00:00.000+0000",
        items=[_summary_item("Old Title", "New Title")],
    )
    session = _make_session([_issue(summary="New Title", status_id="s2", status_name="In Progress", histories=[status_change, summary_change])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    summary_row = df[df["event_id"] == "h2-0"].iloc[0]
    assert summary_row["list_name"] == "In Progress"
    assert summary_row["list_id"] == "s2"
    assert summary_row["old_name"] == "Old Title"
    assert summary_row["source_list_name"] is None


def test_load_jira_board_reconstructs_initial_status_from_first_change():
    move = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_status_item("s1", "To Do", "s2", "In Progress")],
    )
    # Issue's current status is "In Progress" but at creation it was "To Do"
    session = _make_session([_issue(status_id="s2", status_name="In Progress", histories=[move])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    create_row = df[df["event_type"] == "createCard"].iloc[0]
    assert create_row["list_name"] == "To Do"
    assert create_row["list_id"] == "s1"


def test_load_jira_board_card_name_updated_after_rename():
    rename = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[_summary_item("Original", "Renamed")],
    )
    session = _make_session([_issue(summary="Renamed", histories=[rename])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    create_row = df[df["event_type"] == "createCard"].iloc[0]
    rename_row = df[df["event_id"] == "h1-0"].iloc[0]
    assert create_row["card_name"] == "Original"
    assert rename_row["card_name"] == "Renamed"


def test_load_jira_board_multiple_items_in_one_history_entry():
    combined = _history(
        history_id="h1",
        created="2024-01-05T10:00:00.000+0000",
        items=[
            _status_item("s1", "To Do", "s2", "In Progress"),
            _summary_item("Old", "New"),
        ],
    )
    session = _make_session([_issue(summary="New", status_id="s2", status_name="In Progress", histories=[combined])])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    assert "h1-0" in df["event_id"].values
    assert "h1-1" in df["event_id"].values


def test_load_jira_board_timestamps_are_datetime():
    session = _make_session([_issue()])
    df = load_jira_board(1, "https://jira.example.com", session=session, closed_method=None)

    assert df["timestamp"].dtype.kind == "M"


def test_load_jira_board_empty_board_returns_empty_dataframe():
    session = _MockSession({
        "/rest/agile/1.0/board/1/configuration": _board_config(),
        "/rest/agile/1.0/board/1/issues": {"total": 0, "issues": []},
    })
    df = load_jira_board(1, "https://jira.example.com", session=session)

    assert len(df) == 0
    assert "event_id" in df.columns

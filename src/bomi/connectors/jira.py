"""Jira connector: loads a Jira agile board as a bomi event log.

Boards, lists, and cards in bomi correspond to boards, statuses (columns), and
issues in Jira.  A status transition in the Jira changelog becomes a card-move
event with ``source_list`` / ``target_list`` populated.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from ..schema import (
    ACTOR_ID,
    BOARD_ID,
    BOARD_NAME,
    CARD_CLOSED,
    CARD_DUE,
    CARD_ID,
    CARD_NAME,
    SCHEMA_COLUMNS,
    EVENT_ID,
    EVENT_TYPE,
    LIST_CLOSED,
    LIST_ID,
    LIST_NAME,
    OLD_NAME,
    RAW_EVENT_TYPE,
    SOURCE_LIST_ID,
    SOURCE_LIST_NAME,
    SOURCE_SYSTEM,
    TARGET_LIST_ID,
    TARGET_LIST_NAME,
    TIMESTAMP,
    compute_event_types,
    to_board_log,
)

_ISSUES_BATCH = 50
_CHANGELOG_BATCH = 100


def load_jira_board(
    board_id: str | int,
    base_url: str,
    auth: tuple[str, str] | None = None,
    closed_method: str | None = "done_status",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Load a Jira agile board action log as a bomi event log.

    Each Jira issue maps to a card; each Jira status (column) maps to a list.
    One event row is produced per changelog item, plus a synthetic ``createCard``
    event derived from the issue creation metadata.

    For events that are not status transitions (e.g. summary edits, due-date
    changes), ``list_id`` / ``list_name`` are reconstructed by replaying the
    status changelog in chronological order.  ``list_closed`` is always null —
    Jira statuses do not have a closed state.

    Parameters
    ----------
    board_id:
        Numeric or string ID of the Jira agile board.
    base_url:
        Base URL of the Jira instance, e.g. ``https://mycompany.atlassian.net``.
        Do not include a trailing slash.
    auth:
        ``(email, api_token)`` tuple for Jira Cloud basic auth.  Ignored when a
        custom ``session`` is provided; set auth on the session directly in that
        case.
    closed_method:
        How to populate ``card_closed``:

        - ``"done_status"`` *(default)* — ``True`` when the card's status belongs
          to the Done category, as determined by the project's status configuration.
        - ``"resolution"`` — ``True`` when a Jira resolution is set on the issue,
          ``False`` when it is cleared.
        - ``None`` / ``False`` — always ``False``.
    session:
        Optional :class:`requests.Session` for HTTP calls.  Useful for testing or
        for supplying custom proxies / auth headers.
    """
    requester: requests.Session = session or requests.Session()
    if session is None and auth is not None:
        requester.auth = auth

    board_config = _fetch_board_config(board_id, base_url, requester)
    board_id_str = str(board_id)
    board_name: str = board_config["name"]

    issues = _fetch_issues(board_id, base_url, requester)
    if not issues:
        return _empty_log()

    done_status_ids: set[str] = set()
    if closed_method == "done_status":
        project_key: str = issues[0]["fields"]["project"]["key"]
        done_status_ids = _fetch_done_status_ids(base_url, project_key, requester)

    rows: list[dict[str, Any]] = []
    for issue in issues:
        rows.extend(
            _issue_to_rows(issue, board_id_str, board_name, closed_method, done_status_ids)
        )

    df = pd.DataFrame(rows).sort_values(TIMESTAMP).reset_index(drop=True)
    board_log = to_board_log(df, column_map={}, source_system="jira", validate=True)
    return compute_event_types(board_log)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _fetch_board_config(
    board_id: str | int, base_url: str, session: requests.Session
) -> dict[str, Any]:
    url = f"{base_url}/rest/agile/1.0/board/{board_id}/configuration"
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()


def _fetch_done_status_ids(
    base_url: str, project_key: str, session: requests.Session
) -> set[str]:
    url = f"{base_url}/rest/api/3/project/{project_key}/statuses"
    resp = session.get(url)
    resp.raise_for_status()
    done_ids: set[str] = set()
    for issue_type in resp.json():
        for status in issue_type.get("statuses", []):
            if status.get("statusCategory", {}).get("key") == "done":
                done_ids.add(status["id"])
    return done_ids


def _fetch_issues(
    board_id: str | int, base_url: str, session: requests.Session
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    start_at = 0

    while True:
        url = (
            f"{base_url}/rest/agile/1.0/board/{board_id}/issues"
            f"?expand=changelog&maxResults={_ISSUES_BATCH}&startAt={start_at}"
        )
        resp = session.get(url)
        resp.raise_for_status()
        data = resp.json()
        batch: list[dict[str, Any]] = data.get("issues", [])
        issues.extend(batch)

        if not batch or start_at + len(batch) >= data.get("total", 0):
            break
        start_at += len(batch)

    # If the inline changelog was truncated, fetch it in full via the dedicated endpoint.
    for issue in issues:
        changelog = issue.get("changelog", {})
        if changelog.get("total", 0) > len(changelog.get("histories", [])):
            issue["changelog"]["histories"] = _fetch_all_changelog(
                issue["key"], base_url, session
            )

    return issues


def _fetch_all_changelog(
    issue_key: str, base_url: str, session: requests.Session
) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    start_at = 0

    while True:
        url = (
            f"{base_url}/rest/api/3/issue/{issue_key}/changelog"
            f"?maxResults={_CHANGELOG_BATCH}&startAt={start_at}"
        )
        resp = session.get(url)
        resp.raise_for_status()
        data = resp.json()
        batch: list[dict[str, Any]] = data.get("values", [])
        histories.extend(batch)

        if not batch or start_at + len(batch) >= data.get("total", 0):
            break
        start_at += len(batch)

    return histories


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _issue_to_rows(
    issue: dict[str, Any],
    board_id: str,
    board_name: str,
    closed_method: str | None,
    done_status_ids: set[str],
) -> list[dict[str, Any]]:
    fields = issue["fields"]
    issue_key: str = issue["key"]

    histories: list[dict[str, Any]] = sorted(
        issue.get("changelog", {}).get("histories", []),
        key=lambda h: h["created"],
    )

    # Reconstruct initial field values from the first changelog entry for each field.
    initial_status_id, initial_status_name = _initial_status(fields, histories)
    initial_summary = _initial_field_value(fields["summary"], "summary", histories, "fromString")
    initial_due = _initial_field_value(fields.get("duedate"), "duedate", histories, "fromString")

    state: dict[str, Any] = {
        "status_id": initial_status_id,
        "status_name": initial_status_name,
        "card_name": initial_summary,
        "card_due": initial_due,
        "card_closed": (
            closed_method == "done_status" and initial_status_id in done_status_ids
        ),
    }

    base: dict[str, Any] = {
        BOARD_ID: board_id,
        BOARD_NAME: board_name,
        CARD_ID: issue_key,
        SOURCE_SYSTEM: "jira",
        LIST_CLOSED: None,
    }

    rows: list[dict[str, Any]] = [
        {
            **base,
            EVENT_ID: f"{issue['id']}-create",
            EVENT_TYPE: "createCard",
            RAW_EVENT_TYPE: "createCard",
            TIMESTAMP: fields["created"],
            ACTOR_ID: (fields.get("creator") or {}).get("accountId"),
            CARD_NAME: state["card_name"],
            CARD_CLOSED: state["card_closed"],
            CARD_DUE: state["card_due"],
            LIST_ID: state["status_id"],
            LIST_NAME: state["status_name"],
            SOURCE_LIST_ID: None,
            SOURCE_LIST_NAME: None,
            TARGET_LIST_ID: None,
            TARGET_LIST_NAME: None,
            OLD_NAME: None,
        }
    ]

    for history in histories:
        actor_id = (history.get("author") or {}).get("accountId")
        timestamp = history["created"]
        for i, item in enumerate(history.get("items", [])):
            rows.append(
                _changelog_row(
                    base, history["id"], i, actor_id, timestamp,
                    item, state, closed_method, done_status_ids,
                )
            )

    return rows


def _changelog_row(
    base: dict[str, Any],
    history_id: str,
    item_index: int,
    actor_id: str | None,
    timestamp: str,
    item: dict[str, Any],
    state: dict[str, Any],
    closed_method: str | None,
    done_status_ids: set[str],
) -> dict[str, Any]:
    field = item["field"]

    row: dict[str, Any] = {
        **base,
        EVENT_ID: f"{history_id}-{item_index}",
        EVENT_TYPE: "updateCard",
        RAW_EVENT_TYPE: field,
        TIMESTAMP: timestamp,
        ACTOR_ID: actor_id,
        SOURCE_LIST_ID: None,
        SOURCE_LIST_NAME: None,
        TARGET_LIST_ID: None,
        TARGET_LIST_NAME: None,
        OLD_NAME: None,
    }

    if field == "status":
        from_id = item.get("from", "")
        from_name = item.get("fromString", "")
        to_id = item.get("to", "")
        to_name = item.get("toString", "")

        row[SOURCE_LIST_ID] = from_id
        row[SOURCE_LIST_NAME] = from_name
        row[TARGET_LIST_ID] = to_id
        row[TARGET_LIST_NAME] = to_name
        # list_id = source list, consistent with how Trello card-move events work.
        row[LIST_ID] = from_id
        row[LIST_NAME] = from_name

        state["status_id"] = to_id
        state["status_name"] = to_name
        if closed_method == "done_status":
            state["card_closed"] = to_id in done_status_ids

    elif field == "summary":
        row[OLD_NAME] = item.get("fromString")
        row[LIST_ID] = state["status_id"]
        row[LIST_NAME] = state["status_name"]
        state["card_name"] = item.get("toString") or state["card_name"]

    elif field == "duedate":
        state["card_due"] = item.get("toString")
        row[LIST_ID] = state["status_id"]
        row[LIST_NAME] = state["status_name"]

    elif field == "resolution":
        if closed_method == "resolution":
            state["card_closed"] = item.get("to") is not None
        row[LIST_ID] = state["status_id"]
        row[LIST_NAME] = state["status_name"]

    else:
        row[LIST_ID] = state["status_id"]
        row[LIST_NAME] = state["status_name"]

    row[CARD_NAME] = state["card_name"]
    row[CARD_CLOSED] = state["card_closed"]
    row[CARD_DUE] = state["card_due"]

    return row


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _initial_status(
    fields: dict[str, Any], histories: list[dict[str, Any]]
) -> tuple[str, str]:
    first = next(
        (item for h in histories for item in h["items"] if item["field"] == "status"),
        None,
    )
    if first:
        return (
            first.get("from") or fields["status"]["id"],
            first.get("fromString") or fields["status"]["name"],
        )
    return fields["status"]["id"], fields["status"]["name"]


def _initial_field_value(
    current: Any,
    field_name: str,
    histories: list[dict[str, Any]],
    from_key: str,
) -> Any:
    first = next(
        (item for h in histories for item in h["items"] if item["field"] == field_name),
        None,
    )
    return first.get(from_key) if first else current


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SCHEMA_COLUMNS))

import json

import pandas as pd

import bomi
from test_core import sample_log


def test_from_dataframe_validates_schema_format():
    df = sample_log()

    board_log = bomi.from_dataframe(df)

    assert board_log["event_id"].tolist() == ["e1", "e2", "e3", "e4", "e5"]
    assert board_log["timestamp"].dtype.kind == "M"


def test_read_and_write_board_csv_roundtrip(tmp_path):
    csv_path = tmp_path / "board.csv"
    bomi.write_board_csv(sample_log(), csv_path)

    board_log = bomi.read_board_csv(csv_path)

    assert board_log["event_type"].tolist()[0] == "createList"
    assert board_log.loc[board_log["event_id"] == "e4", "target_list_name"].item() == "Done"


def test_to_process_dataframe_uses_cards_as_cases():
    process_df = bomi.to_process_dataframe(sample_log())

    card_events = process_df[process_df["case:concept:name"] == "c1"]
    assert card_events["concept:name"].tolist() == ["Backlog", "Done", "Done"]
    assert "time:timestamp" in process_df.columns


def test_load_trello_board_uses_injected_session():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "id": "e1",
                    "type": "createList",
                    "date": "2024-01-01T00:00:00Z",
                    "idMemberCreator": "u1",
                    "data": {"list": {"id": "l1", "name": "Backlog"}},
                }
            ]

    class Session:
        def get(self, url, headers):
            self.url = url
            self.headers = headers
            return Response()

    session = Session()
    board_log = bomi.load_trello_board("board-id", batch=1000, session=session)

    assert "board-id" in session.url
    assert board_log.loc[0, "event_id"] == "e1"
    assert board_log.loc[0, "list_name"] == "Backlog"
    assert board_log.loc[0, "list_event_type"] == "list_create"


def test_read_trello_json_normalizes_nested_actions(tmp_path):
    json_path = tmp_path / "actions.json"
    actions = [
        {
            "id": "e1",
            "type": "createCard",
            "date": "2024-01-01T00:00:00Z",
            "idMemberCreator": "u1",
            "data": {
                "card": {"id": "c1", "name": "Card 1", "closed": False},
                "list": {"id": "l1", "name": "Backlog"},
            },
        }
    ]
    json_path.write_text(json.dumps(actions), encoding="utf-8")

    board_log = bomi.read_trello_json(json_path)

    assert board_log.loc[0, "event_id"] == "e1"
    assert board_log.loc[0, "card_id"] == "c1"
    assert board_log.loc[0, "list_name"] == "Backlog"
    assert board_log.loc[0, "card_event_type"] == "card_create"

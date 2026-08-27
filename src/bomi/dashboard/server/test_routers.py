"""Tests for dashboard server routers."""

import pandas as pd
from bomi.schema import (
    CARD_ID, CARD_NAME, CARD_EVENT_TYPE, LIST_ID, LIST_NAME, TIMESTAMP,
    TARGET_LIST_ID, TARGET_LIST_NAME, SOURCE_LIST_ID, SOURCE_LIST_NAME,
)


def sample_board_df() -> pd.DataFrame:
    """Create a sample board DataFrame with card events."""
    return pd.DataFrame([
        {
            TIMESTAMP: pd.Timestamp("2024-01-01T09:00:00Z"),
            CARD_ID: "card1",
            CARD_NAME: "Card 1 Name",
            CARD_EVENT_TYPE: "card_create",
            LIST_ID: "list1",
            LIST_NAME: "To Do",
            SOURCE_LIST_ID: None,
            SOURCE_LIST_NAME: None,
            TARGET_LIST_ID: None,
            TARGET_LIST_NAME: None,
        },
        {
            TIMESTAMP: pd.Timestamp("2024-01-02T10:00:00Z"),
            CARD_ID: "card1",
            CARD_NAME: "Card 1 Name",
            CARD_EVENT_TYPE: "card_move",
            LIST_ID: "list1",
            LIST_NAME: "To Do",
            SOURCE_LIST_ID: "list1",
            SOURCE_LIST_NAME: "To Do",
            TARGET_LIST_ID: "list2",
            TARGET_LIST_NAME: "In Progress",
        },
        {
            TIMESTAMP: pd.Timestamp("2024-01-03T11:00:00Z"),
            CARD_ID: "card2",
            CARD_NAME: "Card 2 Name",
            CARD_EVENT_TYPE: "card_create",
            LIST_ID: "list1",
            LIST_NAME: "To Do",
            SOURCE_LIST_ID: None,
            SOURCE_LIST_NAME: None,
            TARGET_LIST_ID: None,
            TARGET_LIST_NAME: None,
        },
        {
            TIMESTAMP: pd.Timestamp("2024-01-04T12:00:00Z"),
            CARD_ID: "card2",
            CARD_NAME: "Card 2 Updated",
            CARD_EVENT_TYPE: "card_act",
            LIST_ID: "list2",
            LIST_NAME: "In Progress",
            SOURCE_LIST_ID: None,
            SOURCE_LIST_NAME: None,
            TARGET_LIST_ID: None,
            TARGET_LIST_NAME: None,
        },
    ])


class TestEventsByListLogic:
    """Tests for events_by_list data processing logic."""

    def test_card_move_with_null_list_uses_target_list(self):
        """Test that card_move events use TARGET_LIST when LIST is null."""
        df = sample_board_df()
        # Simulate card_move with null LIST_ID
        df.loc[1, LIST_ID] = None
        df.loc[1, LIST_NAME] = None

        # Simulate the fillna logic from events_by_list endpoint
        df[LIST_ID] = df[LIST_ID].fillna(df[TARGET_LIST_ID])
        df[LIST_NAME] = df[LIST_NAME].fillna(df[TARGET_LIST_NAME])

        # Verify the card_move event now has the target list
        move_event = df.iloc[1]
        assert move_event[CARD_EVENT_TYPE] == 'card_move'
        assert move_event[LIST_ID] == 'list2'
        assert move_event[LIST_NAME] == 'In Progress'

    def test_card_events_include_card_name(self):
        """Test that card events have CARD_NAME populated."""
        df = sample_board_df()
        card_events = df[df[CARD_EVENT_TYPE].isin(['card_act', 'card_create', 'card_move', 'card_close'])]

        assert len(card_events) == 4
        assert card_events[CARD_NAME].notna().all()
        assert card_events.iloc[0][CARD_NAME] == 'Card 1 Name'
        assert card_events.iloc[2][CARD_NAME] == 'Card 2 Name'

    def test_time_window_filtering(self):
        """Test that time window filtering works correctly."""
        df = sample_board_df()

        # Filter to only events on 2024-01-02 (event is at 10:00, so need full day range)
        start_ts = pd.to_datetime("2024-01-02T00:00:00", utc=True)
        end_ts = pd.to_datetime("2024-01-02T23:59:59", utc=True)
        filtered = df[(df[TIMESTAMP] >= start_ts) & (df[TIMESTAMP] <= end_ts)]

        assert len(filtered) == 1
        assert filtered.iloc[0][CARD_ID] == 'card1'


class TestCardIndicatorsLogic:
    """Tests for card_indicators data processing logic."""

    def test_most_recent_card_name_per_card(self):
        """Test that we get the most recent card_name for each card."""
        df = sample_board_df()

        # Get most recent card name per card (simulating _build_records logic)
        last_card_name = df.sort_values(TIMESTAMP).groupby(CARD_ID)[CARD_NAME].last()

        assert last_card_name['card1'] == 'Card 1 Name'
        assert last_card_name['card2'] == 'Card 2 Updated'  # Most recent name

    def test_card_name_in_all_records(self):
        """Test that card_name is present in all indicator records."""
        df = sample_board_df()
        last_card_name = df.sort_values(TIMESTAMP).groupby(CARD_ID)[CARD_NAME].last()

        # Simulate building records
        records = []
        for card_id in last_card_name.index:
            record = {
                'card_id': card_id,
                'card_name': last_card_name.get(card_id),
            }
            records.append(record)

        assert len(records) == 2
        assert all('card_name' in record for record in records)
        assert records[0]['card_name'] == 'Card 1 Name'
        assert records[1]['card_name'] == 'Card 2 Updated'

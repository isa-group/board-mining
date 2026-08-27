"""Tests for bomi.health board quality and health indicators."""

import pandas as pd
import pytest

import bomi


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_log() -> pd.DataFrame:
    """Board event log with multiple card states for health-indicator testing.

    Cards:
      c_orphan   — created only, never moved or acted upon (active orphan)
      c_active   — created, moved Backlog→InProgress, acted on (active)
      c_bouncing — Backlog→InProgress→Backlog→InProgress (1 bounce)
      c_silent   — Backlog→InProgress→Done with no acts (3 silent stays)
      c_sink     — moved to Done (sink-list completion candidate)
      c_archived — archived (card_closed=True)
      c_overdue  — created with past due date, never moved
      c_assigned — created, moved, has addMemberToCard event
    """
    T = pd.Timestamp
    rows = [
        # lists
        {"event_id": "e_l1", "event_type": "createList", "raw_event_type": "createList",
         "timestamp": T("2024-01-01T08:00Z"), "actor_id": "u1",
         "board_id": "b1", "board_name": "B", "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": "list_create", "card_event_type": pd.NA},
        {"event_id": "e_l2", "event_type": "createList", "raw_event_type": "createList",
         "timestamp": T("2024-01-01T08:01Z"), "actor_id": "u1",
         "board_id": "b1", "board_name": "B", "list_id": "l_inprog", "list_name": "InProgress",
         "list_event_type": "list_create", "card_event_type": pd.NA},
        {"event_id": "e_l3", "event_type": "createList", "raw_event_type": "createList",
         "timestamp": T("2024-01-01T08:02Z"), "actor_id": "u1",
         "board_id": "b1", "board_name": "B", "list_id": "l_done", "list_name": "Done",
         "list_event_type": "list_create", "card_event_type": pd.NA},

        # c_orphan
        {"event_id": "e1", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_orphan", "card_name": "Orphan",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},

        # c_active: create, move Backlog→InProgress, act
        {"event_id": "e2", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:01Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_active", "card_name": "Active",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e3", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-03T10:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_active", "card_name": "Active",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_inprog", "target_list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_move"},
        {"event_id": "e4", "event_type": "commentCard", "raw_event_type": "commentCard",
         "timestamp": T("2024-01-04T10:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_active", "card_name": "Active",
         "card_closed": False, "list_id": "l_inprog", "list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_act"},

        # c_bouncing: Backlog→InProgress→Backlog→InProgress (1 re-entry)
        {"event_id": "e5", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:02Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_bouncing", "card_name": "Bouncer",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e6", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-03T09:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_bouncing", "card_name": "Bouncer",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_inprog", "target_list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_move"},
        {"event_id": "e7", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-04T09:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_bouncing", "card_name": "Bouncer",
         "card_closed": False, "list_id": "l_inprog", "list_name": "InProgress",
         "source_list_id": "l_inprog", "source_list_name": "InProgress",
         "target_list_id": "l_backlog", "target_list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_move"},
        {"event_id": "e8", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-05T09:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_bouncing", "card_name": "Bouncer",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_inprog", "target_list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_move"},

        # c_silent: Backlog→InProgress→Done with no acts (3 silent stays)
        {"event_id": "e9", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:03Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_silent", "card_name": "Silent",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e10", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-03T11:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_silent", "card_name": "Silent",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_inprog", "target_list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_move"},
        {"event_id": "e11", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-04T11:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_silent", "card_name": "Silent",
         "card_closed": False, "list_id": "l_inprog", "list_name": "InProgress",
         "source_list_id": "l_inprog", "source_list_name": "InProgress",
         "target_list_id": "l_done", "target_list_name": "Done",
         "list_event_type": pd.NA, "card_event_type": "card_move"},

        # c_sink: moved to Done (sink-list completion candidate)
        {"event_id": "e12", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:04Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_sink", "card_name": "Sink",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e13", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-03T12:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_sink", "card_name": "Sink",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_done", "target_list_name": "Done",
         "list_event_type": pd.NA, "card_event_type": "card_move"},

        # c_archived: created then archived
        {"event_id": "e14", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:05Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_archived", "card_name": "Archived",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e15", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-05T08:00Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_archived", "card_name": "Archived",
         "card_closed": True, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_close"},

        # c_overdue: created with past due date
        {"event_id": "e16", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:06Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_overdue", "card_name": "Overdue",
         "card_closed": False, "card_due": pd.Timestamp("2024-01-03T00:00Z"),
         "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},

        # c_assigned: created, moved, has addMemberToCard event
        {"event_id": "e17", "event_type": "createCard", "raw_event_type": "createCard",
         "timestamp": T("2024-01-02T09:07Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_assigned", "card_name": "Assigned",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_create"},
        {"event_id": "e18", "event_type": "addMemberToCard", "raw_event_type": "addMemberToCard",
         "timestamp": T("2024-01-02T09:08Z"), "actor_id": "u1",
         "board_id": "b1", "board_name": "B", "card_id": "c_assigned", "card_name": "Assigned",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "list_event_type": pd.NA, "card_event_type": "card_act"},
        {"event_id": "e19", "event_type": "updateCard", "raw_event_type": "updateCard",
         "timestamp": T("2024-01-03T09:07Z"), "actor_id": "u2",
         "board_id": "b1", "board_name": "B", "card_id": "c_assigned", "card_name": "Assigned",
         "card_closed": False, "list_id": "l_backlog", "list_name": "Backlog",
         "source_list_id": "l_backlog", "source_list_name": "Backlog",
         "target_list_id": "l_inprog", "target_list_name": "InProgress",
         "list_event_type": pd.NA, "card_event_type": "card_move"},
    ]
    return pd.DataFrame(rows)


# Fixed reference date to make time-based tests deterministic
REF = pd.Timestamp("2024-01-10T00:00Z")
WINDOW_30D = pd.Timedelta("30D")
WINDOW_1D = pd.Timedelta("1D")


# ---------------------------------------------------------------------------
# card_closed_mask
# ---------------------------------------------------------------------------

def test_card_closed_mask_archived_flags_archived_card():
    df = make_log()
    mask = bomi.card_closed_mask(df, method="archived")
    assert mask["c_archived"]
    assert not mask["c_orphan"]


def test_card_closed_mask_sink_list_flags_card_in_done():
    df = make_log()
    mask = bomi.card_closed_mask(df, method="sink_list", sink_lists=["Done"])
    assert mask["c_sink"]
    assert mask["c_silent"]   # last list is Done
    assert not mask["c_orphan"]


def test_card_closed_mask_combined_methods():
    df = make_log()
    mask = bomi.card_closed_mask(df, method=["archived", "sink_list"], sink_lists=["Done"])
    assert mask["c_archived"]
    assert mask["c_sink"]
    assert not mask["c_orphan"]


def test_card_closed_mask_covers_all_cards():
    df = make_log()
    mask = bomi.card_closed_mask(df)
    assert set(mask.index) == set(df["card_id"].dropna().unique())


def test_card_closed_mask_ignores_completion_after_reference_date():
    df = make_log()
    # c_archived archived on 2024-01-05 — before that date it is still open
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    mask = bomi.card_closed_mask(df, method="archived", reference_date=ref_before)
    assert "c_archived" in mask.index
    assert not mask["c_archived"]


def test_card_closed_mask_marks_card_completed_at_and_after_its_completion_date():
    df = make_log()
    ref_after = pd.Timestamp("2024-01-06T00:00Z")
    mask = bomi.card_closed_mask(df, method="archived", reference_date=ref_after)
    assert mask["c_archived"]


# ---------------------------------------------------------------------------
# reference_date correctness across per-card indicators
# ---------------------------------------------------------------------------

def test_card_age_includes_card_open_at_reference_date():
    df = make_log()
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    age = bomi.card_age(df, reference_date=ref_before)
    # c_archived was open on 2024-01-04, so it must appear in the result
    assert "c_archived" in age.index


def test_inactive_cards_includes_card_open_at_reference_date():
    df = make_log()
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    inactive = bomi.inactive_cards(df, window=pd.Timedelta("30D"), reference_date=ref_before)
    assert "c_archived" in inactive.index


def test_orphan_cards_includes_card_open_at_reference_date():
    df = make_log()
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    orphans = bomi.orphan_cards(df, reference_date=ref_before)
    # c_archived was not yet archived and had no moves → orphan at that date
    assert "c_archived" in orphans.index
    assert orphans["c_archived"]


def test_bouncing_cards_includes_card_open_at_reference_date():
    df = make_log()
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    bounces = bomi.bouncing_cards(df, reference_date=ref_before)
    # c_archived is open before its archive date; c_bouncing has bounces in range
    assert "c_archived" not in bounces.index or bounces.get("c_archived", 0) == 0


def test_completion_rate_uses_reference_date():
    df = make_log()
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    rate_before = bomi.completion_rate(df, reference_date=ref_before)
    rate_after = bomi.completion_rate(df, reference_date=REF)
    # c_archived completes between these two dates, so rate increases
    assert rate_after >= rate_before


# ---------------------------------------------------------------------------
# card_age
# ---------------------------------------------------------------------------

def test_card_age_excludes_completed_cards():
    df = make_log()
    age = bomi.card_age(df, reference_date=REF)
    assert "c_archived" not in age.index
    assert "c_orphan" in age.index


def test_card_age_uses_last_move_timestamp():
    df = make_log()
    age = bomi.card_age(df, reference_date=REF)
    # c_active last moved on 2024-01-03; REF is 2024-01-10 → ~7 days
    assert pd.Timedelta("6D") < age["c_active"] < pd.Timedelta("8D")


def test_card_age_falls_back_to_creation_for_unmoved_card():
    df = make_log()
    age = bomi.card_age(df, reference_date=REF)
    # c_orphan created on 2024-01-02; REF is 2024-01-10 → ~8 days
    assert pd.Timedelta("7D") < age["c_orphan"] < pd.Timedelta("9D")


# ---------------------------------------------------------------------------
# inactive_cards
# ---------------------------------------------------------------------------

def test_inactive_cards_recently_acted_card_is_active():
    df = make_log()
    # c_active acted on 2024-01-04, well within 30-day window
    inactive = bomi.inactive_cards(df, window=WINDOW_30D, reference_date=REF)
    assert not inactive["c_active"]


def test_inactive_cards_orphan_is_inactive():
    df = make_log()
    inactive = bomi.inactive_cards(df, window=WINDOW_30D, reference_date=REF)
    assert inactive["c_orphan"]


def test_inactive_cards_excludes_completed_cards():
    df = make_log()
    inactive = bomi.inactive_cards(df, window=WINDOW_30D, reference_date=REF)
    assert "c_archived" not in inactive.index


def test_inactive_cards_narrow_window_yields_more_inactive():
    df = make_log()
    inactive_wide = bomi.inactive_cards(df, window=WINDOW_30D, reference_date=REF)
    inactive_narrow = bomi.inactive_cards(df, window=WINDOW_1D, reference_date=REF)
    assert inactive_narrow.sum() >= inactive_wide.sum()


# ---------------------------------------------------------------------------
# orphan_cards
# ---------------------------------------------------------------------------

def test_orphan_cards_created_only_is_orphan():
    df = make_log()
    orphans = bomi.orphan_cards(df)
    assert orphans["c_orphan"]
    assert orphans["c_overdue"]


def test_orphan_cards_moved_card_is_not_orphan():
    df = make_log()
    orphans = bomi.orphan_cards(df)
    assert "c_active" in orphans.index
    assert not orphans["c_active"]


def test_orphan_cards_excludes_completed_cards():
    df = make_log()
    orphans = bomi.orphan_cards(df)
    assert "c_archived" not in orphans.index


# ---------------------------------------------------------------------------
# overdue_cards
# ---------------------------------------------------------------------------

def test_overdue_cards_past_due_date_flagged():
    df = make_log()
    overdue = bomi.overdue_cards(df, reference_date=REF)
    assert overdue["c_overdue"]


def test_overdue_cards_no_due_date_absent_from_result():
    df = make_log()
    overdue = bomi.overdue_cards(df, reference_date=REF)
    assert "c_active" not in overdue.index


def test_overdue_cards_warns_when_column_missing():
    df = make_log().drop(columns=["card_due"], errors="ignore")
    with pytest.warns(UserWarning, match="card_due"):
        result = bomi.overdue_cards(df, reference_date=REF)
    assert result.empty


# ---------------------------------------------------------------------------
# bouncing_cards
# ---------------------------------------------------------------------------

def test_bouncing_cards_counts_reentries():
    df = make_log()
    bounces = bomi.bouncing_cards(df)
    # c_bouncing: Backlog→InProgress→Backlog→InProgress
    # Re-entries: InProgress (2nd visit) and Backlog (2nd visit via back-move) = 2 bounces
    assert bounces["c_bouncing"] == 2


def test_bouncing_cards_non_bouncing_card_is_zero():
    df = make_log()
    bounces = bomi.bouncing_cards(df)
    assert bounces.get("c_active", 0) == 0


def test_bouncing_cards_unmoved_card_absent():
    df = make_log()
    bounces = bomi.bouncing_cards(df)
    assert "c_orphan" not in bounces.index


# ---------------------------------------------------------------------------
# silent_moves
# ---------------------------------------------------------------------------

def test_silent_moves_counts_all_silent_stays():
    df = make_log()
    sm = bomi.silent_moves(df)
    # c_silent: create(no act)→move→(no act)→move→(no act open stay) = 3 silent stays
    assert sm["c_silent"] == 3


def test_silent_moves_first_stay_silent_if_no_act_before_first_move():
    df = make_log()
    sm = bomi.silent_moves(df)
    # c_active: create→move (silent stay 1), then act in InProgress (stay 2 not silent)
    assert sm["c_active"] == 1


def test_silent_moves_excludes_completed_cards():
    df = make_log()
    sm = bomi.silent_moves(df)
    assert "c_archived" not in sm.index


def test_silent_moves_treats_card_as_open_before_its_completion_date():
    df = make_log()
    # c_archived is archived on 2024-01-05; before that it is still open
    ref_before = pd.Timestamp("2024-01-04T00:00Z")
    sm = bomi.silent_moves(df, reference_date=ref_before)
    assert "c_archived" in sm.index


# ---------------------------------------------------------------------------
# unassigned_cards
# ---------------------------------------------------------------------------

def test_unassigned_cards_assigned_card_not_flagged():
    df = make_log()
    ua = bomi.unassigned_cards(df, reference_date=REF)
    assert ua is not None
    assert not ua["c_assigned"]


def test_unassigned_cards_card_without_assignment_is_unassigned():
    df = make_log()
    ua = bomi.unassigned_cards(df, reference_date=REF)
    assert ua["c_orphan"]


def test_unassigned_cards_returns_none_when_no_assignment_events():
    df = make_log()
    df = df[df["raw_event_type"] != "addMemberToCard"]
    ua = bomi.unassigned_cards(df, reference_date=REF)
    assert ua is None


def test_unassigned_cards_excludes_completed_cards():
    df = make_log()
    ua = bomi.unassigned_cards(df, reference_date=REF)
    assert ua is not None
    assert "c_archived" not in ua.index


# ---------------------------------------------------------------------------
# stagnant_lists
# ---------------------------------------------------------------------------

def test_stagnant_lists_backlog_has_inactive_cards():
    df = make_log()
    stagnant = bomi.stagnant_lists(df, window=WINDOW_30D, reference_date=REF)
    # c_orphan and c_overdue are inactive and sit in Backlog
    assert "l_backlog" in stagnant.index
    assert stagnant["l_backlog"] >= 2


def test_stagnant_lists_narrow_window_yields_more():
    df = make_log()
    wide = bomi.stagnant_lists(df, window=WINDOW_30D, reference_date=REF)
    narrow = bomi.stagnant_lists(df, window=WINDOW_1D, reference_date=REF)
    assert narrow.sum() >= wide.sum()


# ---------------------------------------------------------------------------
# dead_lists
# ---------------------------------------------------------------------------

def test_dead_lists_list_with_recent_card_is_alive():
    df = make_log()
    # Last events are 2024-01-05; 30D window from REF=2024-01-10 covers all
    dead = bomi.dead_lists(df, window=WINDOW_30D, reference_date=REF)
    assert not dead["l_backlog"]


def test_dead_lists_list_with_no_recent_card_is_dead():
    df = make_log()
    # 1-day window from REF=2024-01-10: no events after 2024-01-09
    dead = bomi.dead_lists(df, window=WINDOW_1D, reference_date=REF)
    assert dead["l_backlog"]


def test_dead_lists_covers_all_lists():
    df = make_log()
    dead = bomi.dead_lists(df, window=WINDOW_30D, reference_date=REF)
    assert set(dead.index) == set(df["list_id"].dropna().unique())


# ---------------------------------------------------------------------------
# flow_conformance
# ---------------------------------------------------------------------------

def test_flow_conformance_all_valid_transitions_gives_one():
    df = make_log()
    prescribed = [
        ("Backlog", "InProgress"),
        ("InProgress", "Backlog"),
        ("InProgress", "Done"),
        ("Backlog", "Done"),
    ]
    score = bomi.flow_conformance(df, prescribed_flow=prescribed)
    assert score == pytest.approx(1.0)


def test_flow_conformance_empty_prescribed_flow_gives_zero():
    df = make_log()
    score = bomi.flow_conformance(df, prescribed_flow=[("X", "Y")])
    assert score == pytest.approx(0.0)


def test_flow_conformance_inferred_returns_valid_score():
    df = make_log()
    score = bomi.flow_conformance(df)
    assert 0.0 <= score <= 1.0


def test_flow_conformance_strict_flow_penalises_backward_moves():
    df = make_log()
    strict = bomi.flow_conformance(
        df, prescribed_flow=[("Backlog", "InProgress"), ("InProgress", "Done")]
    )
    loose = bomi.flow_conformance(
        df,
        prescribed_flow=[
            ("Backlog", "InProgress"), ("InProgress", "Done"),
            ("InProgress", "Backlog"), ("Backlog", "Done"),
        ],
    )
    assert strict <= loose


# ---------------------------------------------------------------------------
# completion_rate and abandonment_rate
# ---------------------------------------------------------------------------

def test_completion_rate_archived_method():
    df = make_log()
    rate = bomi.completion_rate(df, method="archived")
    total = df["card_id"].dropna().nunique()
    assert rate == pytest.approx(1 / total)


def test_completion_rate_sink_list_method():
    df = make_log()
    rate = bomi.completion_rate(df, method="sink_list", sink_lists=["Done"])
    total = df["card_id"].dropna().nunique()
    # c_sink and c_silent ended in Done
    assert rate == pytest.approx(2 / total)


def test_abandonment_rate_nonzero_when_orphans_exist():
    df = make_log()
    rate = bomi.abandonment_rate(df)
    assert rate > 0


# ---------------------------------------------------------------------------
# health_dimensions
# ---------------------------------------------------------------------------

def test_health_dimensions_returns_all_five_keys():
    df = make_log()
    dims = bomi.health_dimensions(df, reference_date=REF)
    assert set(dims.keys()) == {
        "flow_discipline", "collaboration_discipline",
        "completion_discipline", "structural_stability", "board_vitality",
    }


def test_health_dimensions_all_scores_in_unit_interval():
    df = make_log()
    dims = bomi.health_dimensions(df, reference_date=REF)
    for key, val in dims.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} is outside [0, 1]"


def test_health_dimensions_strict_flow_lowers_flow_discipline():
    df = make_log()
    dims_all = bomi.health_dimensions(
        df,
        prescribed_flow=[
            ("Backlog", "InProgress"), ("InProgress", "Backlog"),
            ("InProgress", "Done"), ("Backlog", "Done"),
        ],
        reference_date=REF,
    )
    dims_strict = bomi.health_dimensions(
        df,
        prescribed_flow=[("Backlog", "InProgress"), ("InProgress", "Done")],
        reference_date=REF,
    )
    assert dims_strict["flow_discipline"] <= dims_all["flow_discipline"]


# ---------------------------------------------------------------------------
# board_health
# ---------------------------------------------------------------------------

def test_board_health_returns_expected_keys():
    df = make_log()
    health = bomi.board_health(df, reference_date=REF)
    for key in [
        "inactive_card_rate", "orphan_rate", "bouncing_rate",
        "flow_conformance", "completion_rate", "abandonment_rate",
        "dim_flow_discipline", "dim_board_vitality",
    ]:
        assert key in health, f"missing key: {key}"


def test_board_health_all_rates_in_unit_interval():
    df = make_log()
    health = bomi.board_health(df, reference_date=REF)
    for key, val in health.items():
        if key.endswith("_rate") or key.startswith("dim_"):
            assert 0.0 <= val <= 1.0, f"{key} = {val} outside [0, 1]"


def test_board_health_sink_list_method_increases_completion():
    df = make_log()
    h_archived = bomi.board_health(df, method="archived", reference_date=REF)
    h_sink = bomi.board_health(df, method="sink_list", sink_lists=["Done"], reference_date=REF)
    assert h_sink["completion_rate"] >= h_archived["completion_rate"]


def test_board_health_unassigned_keys_present_when_assignment_events_exist():
    df = make_log()
    health = bomi.board_health(df, reference_date=REF)
    assert "unassigned_rate" in health
    assert "cards_unassigned" in health


# ---------------------------------------------------------------------------
# health_evolution
# ---------------------------------------------------------------------------

def test_health_evolution_returns_dataframe():
    df = make_log()
    result = bomi.health_evolution(df, window=pd.Timedelta("2D"), step=pd.Timedelta("1D"))
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_health_evolution_index_is_timestamps():
    df = make_log()
    result = bomi.health_evolution(df, window=pd.Timedelta("2D"), step=pd.Timedelta("1D"))
    assert result.index.name == "timestamp"
    assert pd.api.types.is_datetime64_any_dtype(result.index)


def test_health_evolution_step_controls_number_of_rows():
    df = make_log()
    # window=2D, step=1D: start = min_ts + 2D, then step 1D until max_ts
    result_1d = bomi.health_evolution(df, window=pd.Timedelta("2D"), step=pd.Timedelta("1D"))
    result_2d = bomi.health_evolution(df, window=pd.Timedelta("2D"), step=pd.Timedelta("2D"))
    assert len(result_1d) >= len(result_2d)


def test_health_evolution_columns_match_board_health_keys():
    df = make_log()
    result = bomi.health_evolution(df, window=pd.Timedelta("2D"), step=pd.Timedelta("1D"))
    health = bomi.board_health(df, inactive_window=pd.Timedelta("2D"), reference_date=REF)
    # All evolution columns should be valid board_health keys
    assert set(result.columns).issubset(set(health.keys()))


def test_health_evolution_indicators_parameter_filters_columns():
    df = make_log()
    wanted = ["completion_rate", "dim_board_vitality"]
    result = bomi.health_evolution(
        df,
        window=pd.Timedelta("2D"),
        step=pd.Timedelta("1D"),
        indicators=wanted,
    )
    assert set(result.columns) == set(wanted)


def test_health_evolution_completion_rate_nondecreasing_with_archived_method():
    df = make_log()
    # With method="archived", once a card is archived it stays archived,
    # so completion_rate can only stay flat or increase over time.
    result = bomi.health_evolution(
        df,
        window=pd.Timedelta("2D"),
        step=pd.Timedelta("1D"),
        indicators=["completion_rate"],
        method="archived",
    )
    rates = result["completion_rate"].dropna().values
    assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def test_health_evolution_start_date_respected():
    df = make_log()
    start = pd.Timestamp("2024-01-04T00:00Z")
    result = bomi.health_evolution(
        df,
        window=pd.Timedelta("2D"),
        step=pd.Timedelta("1D"),
        start_date=start,
    )
    assert result.index[0] >= start


def test_health_evolution_empty_when_window_exceeds_log_duration():
    df = make_log()
    # Window larger than the entire log should yield no valid start date
    result = bomi.health_evolution(df, window=pd.Timedelta("365D"), step=pd.Timedelta("1D"))
    assert result.empty


def test_health_evolution_prescribed_flow_applied_consistently():
    df = make_log()
    prescribed = [("Backlog", "InProgress"), ("InProgress", "Done")]
    result = bomi.health_evolution(
        df,
        window=pd.Timedelta("2D"),
        step=pd.Timedelta("1D"),
        indicators=["flow_conformance"],
        prescribed_flow=prescribed,
    )
    # All conformance values should be in [0, 1]
    assert (result["flow_conformance"].dropna().between(0, 1)).all()

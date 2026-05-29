"""Tests for bomi.conformance flow and performance conformance checkers."""

import pandas as pd
import pytest

import bomi
from bomi import BoardModel, ListConstraints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(date: str) -> pd.Timestamp:
    """Return a UTC timestamp from a YYYY-MM-DD string."""
    return pd.Timestamp(date, tz="UTC")


def _row(event_id, event_type, timestamp, **kwargs):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "raw_event_type": event_type,
        "timestamp": timestamp,
        "actor_id": "u1",
        "board_id": "b1",
        "board_name": "B",
        "list_event_type": pd.NA,
        "card_event_type": pd.NA,
        **kwargs,
    }


def _card_create(eid, date, card_id, list_id, list_name):
    return _row(eid, "createCard", _ts(date),
                card_id=card_id, card_closed=False,
                list_id=list_id, list_name=list_name,
                card_event_type="card_create")


def _card_move(eid, date, card_id, src_id, src_name, tgt_id, tgt_name):
    return _row(eid, "updateCard", _ts(date),
                card_id=card_id, card_closed=False,
                list_id=src_id, list_name=src_name,
                source_list_id=src_id, source_list_name=src_name,
                target_list_id=tgt_id, target_list_name=tgt_name,
                card_event_type="card_move")


def _card_act(eid, date, card_id, list_id, list_name):
    return _row(eid, "commentCard", _ts(date),
                card_id=card_id, card_closed=False,
                list_id=list_id, list_name=list_name,
                card_event_type="card_act")


def _card_close(eid, date, card_id, list_id, list_name):
    return _row(eid, "updateCard", _ts(date),
                card_id=card_id, card_closed=True,
                list_id=list_id, list_name=list_name,
                card_event_type="card_close")


def _card_delete(eid, date, card_id, list_id, list_name):
    return _row(eid, "deleteCard", _ts(date),
                card_id=card_id,
                list_id=list_id, list_name=list_name,
                card_event_type="card_delete")


# ---------------------------------------------------------------------------
# Board log and model for flow conformance tests
# ---------------------------------------------------------------------------

def make_log() -> pd.DataFrame:
    """Three-list board: Backlog → InProgress → Done.

    Cards:
      c_good         — created in Backlog, moved Backlog→InProgress→Done, archived.
      c_wrong_create — created in Done (wrong list).
      c_wrong_move   — created in Backlog, moved forward then backward (InProgress→Backlog).
      c_wrong_close  — reaches Done but never archived (finishes in sink, not archived).
      c_active       — created in Backlog, moved to InProgress, still active.
    """
    return pd.DataFrame([
        # c_good: Backlog→InProgress→Done, archived
        _card_create("e1", "2024-01-02", "c_good", "l_bl", "Backlog"),
        _card_move("e2", "2024-01-03", "c_good", "l_bl", "Backlog", "l_ip", "InProgress"),
        _card_act("e3", "2024-01-03", "c_good", "l_ip", "InProgress"),
        _card_move("e4", "2024-01-04", "c_good", "l_ip", "InProgress", "l_dn", "Done"),
        _card_close("e5", "2024-01-04", "c_good", "l_dn", "Done"),

        # c_wrong_create: created in Done
        _card_create("e6", "2024-01-02", "c_wrong_create", "l_dn", "Done"),
        _card_close("e7", "2024-01-03", "c_wrong_create", "l_dn", "Done"),

        # c_wrong_move: Backlog→InProgress→Backlog (backward move)
        _card_create("e8", "2024-01-02", "c_wrong_move", "l_bl", "Backlog"),
        _card_move("e9", "2024-01-03", "c_wrong_move", "l_bl", "Backlog", "l_ip", "InProgress"),
        _card_move("e10", "2024-01-04", "c_wrong_move", "l_ip", "InProgress", "l_bl", "Backlog"),
        _card_close("e11", "2024-01-05", "c_wrong_move", "l_bl", "Backlog"),

        # c_wrong_close: reaches Done but is never archived
        _card_create("e12", "2024-01-02", "c_wrong_close", "l_bl", "Backlog"),
        _card_move("e13", "2024-01-03", "c_wrong_close", "l_bl", "Backlog", "l_dn", "Done"),

        # c_active: still in InProgress
        _card_create("e14", "2024-01-02", "c_active", "l_bl", "Backlog"),
        _card_move("e15", "2024-01-03", "c_active", "l_bl", "Backlog", "l_ip", "InProgress"),
    ])


def make_model() -> BoardModel:
    return BoardModel(
        lists=["Backlog", "InProgress", "Done"],
        card_flow=[{"Backlog", "InProgress", "Done"}],
        semantic_precedence=[("Backlog", "InProgress"), ("InProgress", "Done")],
        card_create_lists=["Backlog"],
        card_close_lists=["Done"],
        card_use_lists=["InProgress"],
        close_mode="archived",
    )


# ---------------------------------------------------------------------------
# check_flow_conformance
# ---------------------------------------------------------------------------

class TestCheckFlowConformance:
    def test_good_card_is_conformant(self):
        result = bomi.check_flow_conformance(make_log(), make_model())
        row = result.loc["c_good"]

        assert row["create_conformant"] == True
        assert row["moves_nonconformant"] == 0
        assert row["move_conformance_rate"] == 1.0
        assert row["close_conformant"] == True
        assert row["conformant"] == True

    def test_wrong_create_list_flagged(self):
        result = bomi.check_flow_conformance(make_log(), make_model())
        row = result.loc["c_wrong_create"]

        assert row["create_conformant"] == False
        assert row["conformant"] == False

    def test_backward_move_flagged(self):
        result = bomi.check_flow_conformance(make_log(), make_model())
        row = result.loc["c_wrong_move"]

        assert row["moves_total"] == 2
        assert row["moves_nonconformant"] == 1
        assert row["move_conformance_rate"] == pytest.approx(0.5)
        assert row["conformant"] == False

    def test_active_card_has_no_close_verdict(self):
        result = bomi.check_flow_conformance(make_log(), make_model())

        assert pd.isna(result.loc["c_active", "close_conformant"])
        assert result.loc["c_active", "conformant"] == True  # active cards not penalised

    def test_unarchived_card_in_sink_list_flagged_under_archived_mode(self):
        # c_wrong_close ends in Done (card_close_lists) but was never archived.
        # close_mode="archived" → conformant only if archived.
        result = bomi.check_flow_conformance(make_log(), make_model())
        row = result.loc["c_wrong_close"]

        assert row["close_conformant"] == False
        assert row["conformant"] == False

    def test_sink_list_close_mode(self):
        model = make_model()
        model.close_mode = "sink_list"
        result = bomi.check_flow_conformance(make_log(), model)

        # c_wrong_close ends in Done (a card_close_list) → conformant under sink_list
        assert result.loc["c_wrong_close", "close_conformant"] == True
        # c_good also ends in Done (its last move target) → also conformant under sink_list
        assert result.loc["c_good", "close_conformant"] == True

    def test_combined_close_mode_or_semantics(self):
        model = make_model()
        model.close_mode = ["archived", "sink_list"]
        result = bomi.check_flow_conformance(make_log(), model)

        assert result.loc["c_good", "close_conformant"] == True
        assert result.loc["c_wrong_close", "close_conformant"] == True

    def test_update_violation_informational_by_default(self):
        # act event in Backlog, which is not a use list
        df = pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_act("e2", "2024-01-02", "c1", "l1", "Backlog"),
        ])
        model = BoardModel(
            lists=["Backlog", "InProgress"],
            card_flow=[{"Backlog", "InProgress"}],
            semantic_precedence=[("Backlog", "InProgress")],
            card_create_lists=["Backlog"],
            card_close_lists=[],
            card_use_lists=["InProgress"],
            close_mode="archived",
        )
        result = bomi.check_flow_conformance(df, model)
        row = result.loc["c1"]

        assert row["updates_nonconformant"] == 1
        assert row["conformant"] == True  # not strict by default

    def test_update_violation_strict_mode(self):
        df = pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_act("e2", "2024-01-02", "c1", "l1", "Backlog"),
        ])
        model = BoardModel(
            lists=["Backlog", "InProgress"],
            card_flow=[{"Backlog", "InProgress"}],
            semantic_precedence=[("Backlog", "InProgress")],
            card_create_lists=["Backlog"],
            card_close_lists=[],
            card_use_lists=["InProgress"],
            close_mode="archived",
        )
        result = bomi.check_flow_conformance(df, model, strict_updates=True)

        assert result.loc["c1", "conformant"] == False

    def test_card_flow_mode_allows_backward_moves(self):
        model = make_model()
        model.allowed_flow_mode = "card_flow"
        result = bomi.check_flow_conformance(make_log(), model)

        # InProgress→Backlog is within the same component → conformant in card_flow mode
        assert result.loc["c_wrong_move", "moves_nonconformant"] == 0

    def test_empty_create_lists_skips_create_check(self):
        model = make_model()
        model.card_create_lists = []
        result = bomi.check_flow_conformance(make_log(), model)

        assert result["create_conformant"].all()

    def test_deleted_close_mode(self):
        df = pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_delete("e2", "2024-01-02", "c1", "l1", "Backlog"),
        ])
        model = BoardModel(
            lists=["Backlog"],
            card_flow=[{"Backlog"}],
            semantic_precedence=[],
            card_create_lists=["Backlog"],
            card_close_lists=[],
            card_use_lists=[],
            close_mode="deleted",
        )
        result = bomi.check_flow_conformance(df, model)

        assert result.loc["c1", "close_conformant"] == True
        assert result.loc["c1", "conformant"] == True


# ---------------------------------------------------------------------------
# check_wip_history
# ---------------------------------------------------------------------------

def _wip_log():
    """Two cards move into InProgress on Jan 2; c1 leaves on Jan 5."""
    return pd.DataFrame([
        _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
        _card_move("e2", "2024-01-02", "c1", "l1", "Backlog", "l2", "InProgress"),
        _card_create("e3", "2024-01-01", "c2", "l1", "Backlog"),
        _card_move("e4", "2024-01-02", "c2", "l1", "Backlog", "l2", "InProgress"),
        _card_move("e5", "2024-01-05", "c1", "l2", "InProgress", "l3", "Done"),
    ])


def _wip_model(limit: int) -> BoardModel:
    m = make_model()
    m.list_constraints["InProgress"] = ListConstraints(wip_limit=limit)
    return m


class TestCheckWipHistory:
    def test_no_constrained_lists_returns_empty(self):
        result = bomi.check_wip_history(_wip_log(), make_model())
        assert result.empty

    def test_violation_detected_when_wip_exceeds_limit(self):
        result = bomi.check_wip_history(_wip_log(), _wip_model(limit=1))
        row = result.loc["InProgress"]

        assert row["max_wip"] == 2
        assert row["violated"] == True
        # Both cards in InProgress from Jan 2 to Jan 5 = 3 days of violation
        assert row["total_violation_time"] == pd.Timedelta("3D")

    def test_no_violation_when_wip_within_limit(self):
        result = bomi.check_wip_history(_wip_log(), _wip_model(limit=2))
        row = result.loc["InProgress"]

        assert row["max_wip"] == 2
        assert row["violated"] == False
        assert row["total_violation_time"] == pd.Timedelta(0)

    def test_unconstrained_lists_not_in_result(self):
        result = bomi.check_wip_history(_wip_log(), _wip_model(limit=1))

        assert "Backlog" not in result.index
        assert "Done" not in result.index


# ---------------------------------------------------------------------------
# check_wip_current
# ---------------------------------------------------------------------------

class TestCheckWipCurrent:
    def test_no_constrained_lists_returns_empty(self):
        result = bomi.check_wip_current(_wip_log(), make_model())
        assert result.empty

    def test_current_wip_before_any_entries(self):
        result = bomi.check_wip_current(
            _wip_log(), _wip_model(limit=1), reference_date=_ts("2024-01-01")
        )
        row = result.loc["InProgress"]

        assert row["current_wip"] == 0
        assert row["violated"] == False

    def test_current_wip_during_violation(self):
        result = bomi.check_wip_current(
            _wip_log(), _wip_model(limit=1), reference_date=_ts("2024-01-03")
        )
        row = result.loc["InProgress"]

        assert row["current_wip"] == 2
        assert row["violated"] == True
        # Violation started Jan 2, ref is Jan 3 → 1 day of violation so far
        assert row["total_violation_time"] == pd.Timedelta("1D")

    def test_current_wip_after_one_card_leaves(self):
        result = bomi.check_wip_current(
            _wip_log(), _wip_model(limit=1), reference_date=_ts("2024-01-06")
        )
        row = result.loc["InProgress"]

        assert row["current_wip"] == 1
        assert row["violated"] == False
        # Violation lasted Jan 2 to Jan 5 = 3 days total
        assert row["total_violation_time"] == pd.Timedelta("3D")


# ---------------------------------------------------------------------------
# check_sla_history
# ---------------------------------------------------------------------------

def _sla_log():
    """c1 spends 3 days in InProgress (Jan 2–5); c2 spends 1 day (Jan 2–3)."""
    return pd.DataFrame([
        _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
        _card_move("e2", "2024-01-02", "c1", "l1", "Backlog", "l2", "InProgress"),
        _card_move("e3", "2024-01-05", "c1", "l2", "InProgress", "l3", "Done"),
        _card_close("e4", "2024-01-05", "c1", "l3", "Done"),

        _card_create("e5", "2024-01-01", "c2", "l1", "Backlog"),
        _card_move("e6", "2024-01-02", "c2", "l1", "Backlog", "l2", "InProgress"),
        _card_move("e7", "2024-01-03", "c2", "l2", "InProgress", "l3", "Done"),
        _card_close("e8", "2024-01-03", "c2", "l3", "Done"),
    ])


def _sla_model(sla: pd.Timedelta) -> BoardModel:
    m = make_model()
    m.list_constraints["InProgress"] = ListConstraints(sla=sla)
    return m


class TestCheckSlaHistory:
    def test_no_constrained_lists_returns_empty(self):
        result = bomi.check_sla_history(_sla_log(), make_model())
        assert result.empty

    def test_stay_exceeding_sla_flagged(self):
        result = bomi.check_sla_history(_sla_log(), _sla_model(pd.Timedelta("2D")))

        assert result.loc[("c1", "InProgress"), "violated"] == True
        assert result.loc[("c2", "InProgress"), "violated"] == False

    def test_time_in_list_values(self):
        result = bomi.check_sla_history(_sla_log(), _sla_model(pd.Timedelta("2D")))

        assert result.loc[("c1", "InProgress"), "time_in_list"] == pd.Timedelta("3D")
        assert result.loc[("c2", "InProgress"), "time_in_list"] == pd.Timedelta("1D")

    def test_active_cards_not_in_history(self):
        df = pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_move("e2", "2024-01-02", "c1", "l1", "Backlog", "l2", "InProgress"),
            # c1 never leaves InProgress
        ])
        model = make_model()
        model.list_constraints["InProgress"] = ListConstraints(sla=pd.Timedelta("1D"))
        result = bomi.check_sla_history(df, model)

        assert result.empty


# ---------------------------------------------------------------------------
# check_sla_current
# ---------------------------------------------------------------------------

class TestCheckSlaCurrent:
    def _log(self):
        """c1 entered InProgress Jan 2; c2 entered Jan 3. Neither has left."""
        return pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_move("e2", "2024-01-02", "c1", "l1", "Backlog", "l2", "InProgress"),
            _card_create("e3", "2024-01-01", "c2", "l1", "Backlog"),
            _card_move("e4", "2024-01-03", "c2", "l1", "Backlog", "l2", "InProgress"),
        ])

    def _model(self, sla: pd.Timedelta) -> BoardModel:
        m = make_model()
        m.list_constraints["InProgress"] = ListConstraints(sla=sla)
        return m

    def test_no_constrained_lists_returns_empty(self):
        result = bomi.check_sla_current(self._log(), make_model())
        assert result.empty

    def test_card_over_sla_flagged(self):
        result = bomi.check_sla_current(
            self._log(), self._model(pd.Timedelta("2D")),
            reference_date=_ts("2024-01-05"),
        )
        # c1: 3 days in InProgress → violated; c2: 2 days → not violated (not strictly greater)
        assert result.loc[("c1", "InProgress"), "violated"] == True
        assert result.loc[("c2", "InProgress"), "violated"] == False

    def test_time_in_list_so_far(self):
        result = bomi.check_sla_current(
            self._log(), self._model(pd.Timedelta("2D")),
            reference_date=_ts("2024-01-05"),
        )
        assert result.loc[("c1", "InProgress"), "time_in_list_so_far"] == pd.Timedelta("3D")
        assert result.loc[("c2", "InProgress"), "time_in_list_so_far"] == pd.Timedelta("2D")

    def test_archived_card_excluded(self):
        df = pd.DataFrame([
            _card_create("e1", "2024-01-01", "c1", "l1", "Backlog"),
            _card_move("e2", "2024-01-02", "c1", "l1", "Backlog", "l2", "InProgress"),
            _card_close("e3", "2024-01-03", "c1", "l2", "InProgress"),
        ])
        model = make_model()
        model.list_constraints["InProgress"] = ListConstraints(sla=pd.Timedelta("1D"))
        result = bomi.check_sla_current(df, model, reference_date=_ts("2024-01-10"))

        assert result.empty

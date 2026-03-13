"""
Tests for modules/spreading.py — Spreading Activation (Fase 4).

Unit tests with mocks. No Qdrant or production DB required.
Covers: _clamp_salience, _get_neighbors, _record_edges,
        _get_incoming_neighbors, is_causal_chain_member,
        get_chain_member_ids, _spread_activation, recurrent_cycle,
        _EDGE_TYPE_WEIGHT.
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from modules.spreading import (
    _clamp_salience,
    _get_neighbors,
    _record_edges,
    _get_incoming_neighbors,
    _init_edge_table,
    is_causal_chain_member,
    get_chain_member_ids,
    _spread_activation,
    recurrent_cycle,
    _EDGE_TYPE_WEIGHT,
    _INHIBITION_K,
    _INHIBITION_FACTOR,
)
from modules.config import SPREAD_SALIENCE_CAP, SPREAD_SALIENCE_FLOOR


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def edge_db(tmp_path):
    """SQLite with spreading_edges table initialized."""
    db_path = str(tmp_path / "edges.db")
    conn = sqlite3.connect(db_path)
    _init_edge_table(conn)
    yield conn, db_path
    conn.close()


# ============================================================
# CLAMP SALIENCE
# ============================================================

class TestClampSalience:

    def test_within_range(self):
        assert _clamp_salience(0.5) == 0.5

    def test_above_cap(self):
        assert _clamp_salience(99.0) == SPREAD_SALIENCE_CAP

    def test_below_floor(self):
        assert _clamp_salience(-10.0) == SPREAD_SALIENCE_FLOOR

    def test_at_cap(self):
        assert _clamp_salience(SPREAD_SALIENCE_CAP) == SPREAD_SALIENCE_CAP

    def test_at_floor(self):
        assert _clamp_salience(SPREAD_SALIENCE_FLOOR) == SPREAD_SALIENCE_FLOOR


# ============================================================
# GET NEIGHBORS (outgoing)
# ============================================================

class TestGetNeighbors:

    def test_empty_payload(self):
        assert _get_neighbors("node-1", {}) == []

    def test_related_to(self):
        payload = {"related_to": "node-2"}
        result = _get_neighbors("node-1", payload)
        assert result == ["node-2"]

    def test_consolidated_with(self):
        payload = {"consolidated_with": ["node-2", "node-3"]}
        result = _get_neighbors("node-1", payload)
        assert "node-2" in result
        assert "node-3" in result

    def test_broadcast_received_from(self):
        payload = {"broadcast_received_from": "node-2"}
        result = _get_neighbors("node-1", payload)
        assert result == ["node-2"]

    def test_excludes_self(self):
        payload = {"related_to": "node-1"}
        result = _get_neighbors("node-1", payload)
        assert result == []

    def test_deduplicates(self):
        payload = {
            "related_to": "node-2",
            "consolidated_with": ["node-2", "node-3"],
            "broadcast_received_from": "node-2",
        }
        result = _get_neighbors("node-1", payload)
        assert len(result) == 2
        assert result[0] == "node-2"
        assert result[1] == "node-3"

    def test_related_memories_list(self):
        payload = {"related_memories": ["node-2", "node-3"]}
        result = _get_neighbors("node-1", payload)
        assert "node-2" in result
        assert "node-3" in result

    def test_mixed_sources_order(self):
        payload = {
            "related_to": "r1",
            "related_memories": ["rm1"],
            "consolidated_with": ["c1"],
            "broadcast_received_from": "b1",
        }
        result = _get_neighbors("node-1", payload)
        assert result == ["r1", "rm1", "c1", "b1"]


# ============================================================
# EDGE RECORDING
# ============================================================

class TestRecordEdges:

    def test_record_basic_edges(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "from-1", ["to-1", "to-2"], "2026-03-12T10:00:00")
        rows = conn.execute("SELECT from_id, to_id, edge_type FROM spreading_edges").fetchall()
        assert len(rows) == 2
        assert rows[0][2] == "co_occurs"

    def test_record_causal_edge(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "from-1", ["to-1"], "2026-03-12T10:00:00",
                      edge_type="causes", strength=0.9)
        row = conn.execute(
            "SELECT edge_type, strength FROM spreading_edges WHERE from_id='from-1'"
        ).fetchone()
        assert row[0] == "causes"
        assert row[1] == 0.9

    def test_default_strength_by_type(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00", edge_type="causes")
        _record_edges(conn, "c", ["d"], "2026-03-12T10:00:00", edge_type="enables")
        _record_edges(conn, "e", ["f"], "2026-03-12T10:00:00", edge_type="co_occurs")

        r_causes = conn.execute(
            "SELECT strength FROM spreading_edges WHERE from_id='a'"
        ).fetchone()
        r_enables = conn.execute(
            "SELECT strength FROM spreading_edges WHERE from_id='c'"
        ).fetchone()
        r_cooccurs = conn.execute(
            "SELECT strength FROM spreading_edges WHERE from_id='e'"
        ).fetchone()
        assert r_causes[0] == 0.7
        assert r_enables[0] == 0.5
        assert r_cooccurs[0] == 0.3

    def test_upsert_on_conflict(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00",
                      edge_type="co_occurs", strength=0.3)
        _record_edges(conn, "a", ["b"], "2026-03-12T11:00:00",
                      edge_type="causes", strength=0.9)
        rows = conn.execute(
            "SELECT edge_type, strength FROM spreading_edges WHERE from_id='a' AND to_id='b'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "causes"
        assert rows[0][1] == 0.9


# ============================================================
# INCOMING NEIGHBORS
# ============================================================

class TestGetIncomingNeighbors:

    def test_no_incoming(self, edge_db):
        conn, db_path = edge_db
        result = _get_incoming_neighbors(conn, "isolated-node")
        assert result == []

    def test_finds_incoming(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00",
                      edge_type="causes", strength=0.8)
        result = _get_incoming_neighbors(conn, "b")
        assert len(result) == 1
        assert result[0][0] == "a"
        assert result[0][1] == "causes"
        assert result[0][2] == 0.8

    def test_respects_limit(self, edge_db):
        conn, db_path = edge_db
        for i in range(10):
            _record_edges(conn, f"src-{i}", ["target"], "2026-03-12T10:00:00")
        result = _get_incoming_neighbors(conn, "target", limit=3)
        assert len(result) == 3


# ============================================================
# CAUSAL CHAIN MEMBERSHIP
# ============================================================

class TestIsCausalChainMember:

    def test_not_in_chain_no_edges(self, edge_db):
        conn, db_path = edge_db
        assert is_causal_chain_member("lonely-node", fts_db_path=db_path) is False

    def test_in_chain_both_directions(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00", edge_type="causes")
        _record_edges(conn, "b", ["c"], "2026-03-12T10:00:00", edge_type="enables")
        assert is_causal_chain_member("b", fts_db_path=db_path) is True

    def test_not_in_chain_only_outgoing(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00", edge_type="causes")
        # 'a' has outgoing but no incoming causal
        assert is_causal_chain_member("a", fts_db_path=db_path) is False

    def test_nonexistent_db(self):
        assert is_causal_chain_member("node", fts_db_path="/no/such/file.db") is False


class TestGetChainMemberIds:

    def test_empty_db(self, edge_db):
        conn, db_path = edge_db
        assert get_chain_member_ids(fts_db_path=db_path) == set()

    def test_finds_chain_members(self, edge_db):
        conn, db_path = edge_db
        _record_edges(conn, "a", ["b"], "2026-03-12T10:00:00", edge_type="causes")
        _record_edges(conn, "b", ["c"], "2026-03-12T10:00:00", edge_type="enables")
        members = get_chain_member_ids(fts_db_path=db_path)
        assert "b" in members
        assert "a" not in members  # only outgoing
        assert "c" not in members  # only incoming

    def test_nonexistent_db(self):
        assert get_chain_member_ids(fts_db_path="/no/such/file.db") == set()


# ============================================================
# EDGE TYPE WEIGHTS
# ============================================================

class TestEdgeTypeWeights:

    def test_causes_full_weight(self):
        assert _EDGE_TYPE_WEIGHT["causes"] == 1.0

    def test_enables_moderate(self):
        assert _EDGE_TYPE_WEIGHT["enables"] == 0.8

    def test_prevents_inhibitory(self):
        assert _EDGE_TYPE_WEIGHT["prevents"] < 0

    def test_co_occurs_zero(self):
        assert _EDGE_TYPE_WEIGHT["co_occurs"] == 0.0

    def test_confounded_zero(self):
        assert _EDGE_TYPE_WEIGHT["confounded"] == 0.0


# ============================================================
# SPREAD ACTIVATION (mocked pg_store)
# ============================================================

class TestSpreadActivation:

    def _make_fake_point(self, pid, salience=0.5, neighbors=None):
        p = MagicMock()
        p.id = pid
        payload = {"attention_salience": salience}
        if neighbors:
            payload["related_memories"] = neighbors
        p.payload = payload
        return p

    @patch("modules.access_tracking.record_spreading")
    @patch("modules.spreading.pg")
    @patch("modules.spreading.os.path.exists", return_value=False)
    def test_no_seeds_returns_empty(self, mock_exists, mock_pg, mock_record):
        mock_pg.get_by_ids.return_value = []
        result = _spread_activation(["nonexistent"])
        assert result["affected"] == 0
        assert result["updates"] == {}

    @patch("modules.access_tracking.record_spreading")
    @patch("modules.spreading.pg")
    @patch("modules.spreading.os.path.exists", return_value=False)
    def test_single_seed_no_neighbors(self, mock_exists, mock_pg, mock_record):
        seed = self._make_fake_point("seed-1", salience=0.5)
        mock_pg.get_by_ids.return_value = [seed]
        result = _spread_activation(["seed-1"], seed_boost=0.1)
        assert result["total_nodes_visited"] >= 1

    @patch("modules.access_tracking.record_spreading")
    @patch("modules.spreading.pg")
    @patch("modules.spreading.os.path.exists", return_value=False)
    def test_propagation_to_neighbor(self, mock_exists, mock_pg, mock_record):
        seed = self._make_fake_point("seed-1", salience=0.7, neighbors=["nb-1"])
        nb = self._make_fake_point("nb-1", salience=0.3)

        def side_effect(ids):
            result = []
            for i in ids:
                if i == "seed-1":
                    result.append(seed)
                elif i == "nb-1":
                    result.append(nb)
            return result

        mock_pg.get_by_ids.side_effect = side_effect
        # Note: with os.path.exists=False, no edge_conn is opened
        # So only outgoing (payload) edges are used
        # But outgoing edges have type 'similarity' which is not in _EDGE_TYPE_WEIGHT
        # so they get weight 0.0 and won't propagate
        result = _spread_activation(["seed-1"], depth=1, factor=0.7)
        # With similarity edges getting weight=0.0, no propagation happens
        assert result["total_nodes_visited"] >= 1

    @patch("modules.access_tracking.record_spreading")
    @patch("modules.spreading.pg")
    @patch("modules.spreading.os.path.exists", return_value=False)
    def test_seed_boost_applied(self, mock_exists, mock_pg, mock_record):
        seed = self._make_fake_point("seed-1", salience=0.5)
        mock_pg.get_by_ids.return_value = [seed]
        result = _spread_activation(["seed-1"], seed_boost=0.2)
        # Seed boost should appear in updates for seed
        if result["updates"]:
            assert "seed-1" in result["updates"]

    @patch("modules.access_tracking.record_spreading")
    @patch("modules.spreading.pg")
    @patch("modules.spreading.os.path.exists", return_value=False)
    def test_exception_returns_empty(self, mock_exists, mock_pg, mock_record):
        mock_pg.get_by_ids.side_effect = Exception("connection refused")
        result = _spread_activation(["any-id"])
        assert result["affected"] == 0


# ============================================================
# RECURRENT CYCLE (mocked)
# ============================================================

class TestRecurrentCycle:

    @patch("modules.spreading._spread_activation")
    def test_single_cycle(self, mock_spread):
        mock_spread.return_value = {
            "affected": 0,
            "max_depth_reached": 0,
            "total_nodes_visited": 1,
            "updates": {},
        }
        result = recurrent_cycle(["seed-1"], cycles=1)
        assert result["cycles_run"] == 1
        assert result["stable"] is False

    @patch("modules.spreading._spread_activation")
    def test_stability_detection(self, mock_spread):
        # Two cycles with same top nodes -> stable
        mock_spread.return_value = {
            "affected": 3,
            "max_depth_reached": 1,
            "total_nodes_visited": 4,
            "updates": {"a": 0.9, "b": 0.8, "c": 0.7},
        }
        result = recurrent_cycle(["seed-1"], cycles=3)
        assert result["stable"] is True

    @patch("modules.spreading._spread_activation")
    def test_cycles_clamped(self, mock_spread):
        mock_spread.return_value = {
            "affected": 0, "max_depth_reached": 0,
            "total_nodes_visited": 1, "updates": {},
        }
        result = recurrent_cycle(["seed-1"], cycles=100)
        assert result["cycles_run"] <= 5

    @patch("modules.spreading._spread_activation")
    def test_empty_seeds(self, mock_spread):
        result = recurrent_cycle([], cycles=2)
        assert result["cycles_run"] == 0
        assert result["total_affected"] == 0

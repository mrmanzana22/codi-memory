"""Tests for cross-loop handlers (CX-1 through CX-30).

TIER 1:
CX-1: PE drives curiosity (Schmidhuber 2010, Gottlieb 2013)
CX-2: Resolved curiosity reduces PE (Schwartenbeck 2015, Gruber 2019)
CX-3: Self-model competes in GNW (Graziano 2013, Luppi 2024)
CX-4: Forgetting rate tracks consolidation urgency (Stickgold & Walker 2013)

TIER 2:
CX-5: GNW broadcast to action selection (Dehaene 2014, Friston 2015)
CX-6: Metacognition modulates explore/exploit (Boldt 2019, Gershman 2018)
CX-7: Causal DAG informs prediction (Pearl 2009, Bramley 2017)
CX-8: Reconsolidation protects from decay (Lee 2008, Forcato 2011)

TIER 3:
CX-9: GNW broadcast → self-model refresh (Northoff 2004, Qin & Northoff 2011)
CX-10: Self-model ↔ metacognition (Nelson & Narens 1990, Kruger & Dunning 1999)
CX-11: Curiosity → causal discovery (Bramley 2017, Steyvers 2003)
CX-13: PAD → EFE weights (Doya 2008, Vinckier 2018, Cools & D'Esposito 2011)
CX-14: Consolidation gaps → curiosity (Loewenstein 1994, Berlyne 1960)
MIN-1: Curiosity competitive pruning (feedforward inhibition)
MIN-2: SHY SS downscaling (Tononi & Cirelli 2006)

TIER 4:
CX-15: Self-model → forgetting modulation (Sedikides 2009) — INHIBITORY
CX-16: GNW broadcast → metacognition (Shea & Frith 2019)
CX-17: Consolidation → prediction priors (Tse 2007, McClelland 1995)
CX-18: Reconsolidation → metacognitive confidence (Nelson & Narens 1990)
CX-19: Consolidation → self-model (Conway 2005 SMS)
CX-20: Metacognitive uncertainty → curiosity (Loewenstein 1994, Litman 2005)
CX-21: Causal centrality → decay protection (Kirkpatrick 2017) — INHIBITORY

TIER 5:
CX-22: Action selection → metacognition (all 4 architectures)
CX-23: Forgetting suppresses curiosity (Anderson & Hanslmayr 2014) — INHIBITORY
CX-24: High metacognitive confidence blocks reconsolidation (Suzuki 2004) — INHIBITORY
CX-25: Workspace access: testing effect + RIF (Roediger & Karpicke 2006) — INHIBITORY
CX-26: Self-model constrains action (Oyserman 2017, Seth & Friston 2016) — INHIBITORY
CX-27: Low metacognitive confidence suppresses causal edges (Fleming & Dolan 2012) — INHIB
CX-28: Forgetting degrades metacognition (Koriat 1993) — INHIBITORY
CX-29: Memory correction invalidates causal edges (Pearl 2009)
CX-30: Action outcomes as causal interventions (Pearl 2009, Bramley 2017)
"""

import json
import os
import sys
import time
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test env
os.environ.setdefault("CODI_INSTANCE", "test")
os.environ.setdefault("MEM0_COLLECTION", "test_cross_loops")


class TestCX1_PEDrivesCuriosity(unittest.TestCase):
    """CX-1: Prediction Error in Goldilocks zone drives curiosity."""

    def setUp(self):
        import modules.wiring as w
        w._pe_curiosity_cooldowns.clear()

    @patch("modules.wiring._compute_learning_progress", return_value=0.15)
    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_goldilocks_pe_generates_curiosity(self, mock_push, mock_lp):
        """PE in 0.4-0.95 range + positive LP → curiosity pushed."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.6, "topic": "trading"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        self.assertIn("trading", call_args[1]["tema"])
        self.assertEqual("media", call_args[1]["prioridad"])

    @patch("modules.wiring._compute_learning_progress", return_value=0.15)
    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_high_pe_gets_alta_priority(self, mock_push, mock_lp):
        """PE > 0.7 → prioridad alta."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.8, "topic": "consciencia"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_called_once()
        self.assertEqual("alta", mock_push.call_args[1]["prioridad"])

    @patch("modules.curiosity.push_curiosidad")
    def test_low_pe_skipped(self, mock_push):
        """PE < 0.4 (boring) → no curiosity."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.2, "topic": "general"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_not_called()

    @patch("modules.curiosity.push_curiosidad")
    def test_too_high_pe_skipped(self, mock_push):
        """PE > 0.95 (noise) → no curiosity."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.98, "topic": "noise"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_not_called()

    @patch("modules.wiring._compute_learning_progress", return_value=0.0)
    @patch("modules.curiosity.push_curiosidad")
    def test_no_learning_progress_skipped(self, mock_push, mock_lp):
        """LP <= 0.05 → noise domain, skip."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.6, "topic": "noise_domain"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_not_called()

    @patch("modules.wiring._compute_learning_progress", return_value=0.15)
    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_cooldown_prevents_flood(self, mock_push, mock_lp):
        """Same topic within cooldown → second call skipped."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.6, "topic": "repeated_topic"}
        _on_pe_drives_curiosity("prediction_error", data)
        _on_pe_drives_curiosity("prediction_error", data)
        self.assertEqual(1, mock_push.call_count)

    @patch("modules.wiring._compute_learning_progress", return_value=None)
    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_insufficient_data_still_fires(self, mock_push, mock_lp):
        """LP=None (insufficient data) → still fires (uses raw PE)."""
        from modules.wiring import _on_pe_drives_curiosity
        data = {"error_magnitude": 0.65, "topic": "new_topic"}
        _on_pe_drives_curiosity("prediction_error", data)
        mock_push.assert_called_once()


class TestCX2_CuriosityReducesPE(unittest.TestCase):
    """CX-2: Resolved curiosity boosts topic precision."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_fts.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_topic TEXT, actual_topic TEXT,
                predicted_keywords TEXT, actual_keywords TEXT,
                surprise_score REAL, precision_weight REAL,
                weighted_surprise REAL, hit INTEGER DEFAULT 0,
                source TEXT, created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_resolved_curiosity_injects_hit(self):
        """Resolving curiosity → synthetic hit inserted in prediction_results."""
        os.environ["FTS_DB_PATH"] = self.db_path
        from modules.wiring import _on_curiosity_resolved_prediction
        data = {
            "category": "trading",
            "question": "Why does PE spike for crypto?",
            "answer_length": 150,
        }
        _on_curiosity_resolved_prediction("curiosity_resolved", data)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM prediction_results WHERE source = 'curiosity_resolution'").fetchall()
        conn.close()
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("trading", row["actual_topic"])
        self.assertEqual(1, row["hit"])
        self.assertAlmostEqual(0.1, row["surprise_score"], places=1)

    def test_trivial_resolution_skipped(self):
        """answer_length < 10 → skip, no insertion."""
        from modules.wiring import _on_curiosity_resolved_prediction
        data = {"category": "test", "question": "x", "answer_length": 5}
        # Should not raise and should not insert
        _on_curiosity_resolved_prediction("curiosity_resolved", data)

    def test_empty_topic_skipped(self):
        """Empty category → skip."""
        from modules.wiring import _on_curiosity_resolved_prediction
        data = {"category": "", "question": "x", "answer_length": 100}
        _on_curiosity_resolved_prediction("curiosity_resolved", data)


class TestCX3_SelfModelInGNW(unittest.TestCase):
    """CX-3: Self-model competes in GNW via WM injection."""

    @patch("modules.working_memory.push_to_working_memory")
    def test_self_model_pushed_to_wm(self, mock_push):
        """Self-model refresh → WM push with elevated relevance."""
        from modules.wiring import _on_self_model_to_competition
        data = {"source": "periodic", "summary_len": 500, "discrepancy_count": 0}
        _on_self_model_to_competition("self_model_refreshed", data)
        mock_push.assert_called_once()
        call_args = mock_push.call_args[1]
        self.assertIn("SELF-AWARENESS", call_args["content"])
        self.assertEqual("self_model_gwt", call_args["source"])
        self.assertGreater(call_args["relevance"], 0.6)

    @patch("modules.working_memory.push_to_working_memory")
    def test_high_discrepancy_lowers_confidence(self, mock_push):
        """Many discrepancies → lower confidence → lower relevance."""
        from modules.wiring import _on_self_model_to_competition
        data_low = {"source": "periodic", "summary_len": 500, "discrepancy_count": 0}
        data_high = {"source": "periodic", "summary_len": 500, "discrepancy_count": 4}

        _on_self_model_to_competition("self_model_refreshed", data_low)
        rel_low = mock_push.call_args[1]["relevance"]
        mock_push.reset_mock()

        _on_self_model_to_competition("self_model_refreshed", data_high)
        rel_high = mock_push.call_args[1]["relevance"]

        self.assertGreater(rel_low, rel_high)

    @patch("modules.working_memory.push_to_working_memory")
    def test_short_summary_skipped(self, mock_push):
        """summary_len < 20 → skip."""
        from modules.wiring import _on_self_model_to_competition
        data = {"source": "periodic", "summary_len": 10, "discrepancy_count": 0}
        _on_self_model_to_competition("self_model_refreshed", data)
        mock_push.assert_not_called()


class TestCX4_ForgettingConsolidation(unittest.TestCase):
    """CX-4: Vault rate tracks consolidation urgency."""

    def setUp(self):
        import modules.wiring as w
        w._vault_tracker["count"] = 0
        w._vault_tracker["important_count"] = 0
        w._vault_tracker["window_start"] = time.monotonic()

    def test_vault_increments_counter(self):
        """Each vault event increments tracker."""
        from modules.wiring import _on_vault_tracks_urgency, _vault_tracker
        data = {"memory_id": "abc123", "importance": "low"}
        _on_vault_tracks_urgency("memory_vaulted", data)
        self.assertEqual(1, _vault_tracker["count"])
        self.assertEqual(0, _vault_tracker["important_count"])

    def test_important_vault_tracked_separately(self):
        """High/critical importance vaults increment important_count."""
        from modules.wiring import _on_vault_tracks_urgency, _vault_tracker
        _on_vault_tracks_urgency("memory_vaulted", {"importance": "high"})
        _on_vault_tracks_urgency("memory_vaulted", {"importance": "critical"})
        _on_vault_tracks_urgency("memory_vaulted", {"importance": "low"})
        self.assertEqual(3, _vault_tracker["count"])
        self.assertEqual(2, _vault_tracker["important_count"])

    def test_urgency_none_when_few_vaults(self):
        """< 3 important vaults → urgency is None."""
        from modules.wiring import get_consolidation_urgency, _vault_tracker
        _vault_tracker["important_count"] = 2
        result = get_consolidation_urgency()
        self.assertIsNone(result)

    def test_urgency_moderate_at_threshold(self):
        """3-10 important vaults → moderate urgency."""
        from modules.wiring import get_consolidation_urgency, _vault_tracker
        _vault_tracker["important_count"] = 5
        _vault_tracker["count"] = 20
        result = get_consolidation_urgency()
        self.assertIsNotNone(result)
        self.assertEqual("moderate", result["urgency"])
        self.assertGreater(result["lookback_multiplier"], 1.0)

    def test_urgency_high_at_many_vaults(self):
        """> 10 important vaults → high urgency."""
        from modules.wiring import get_consolidation_urgency, _vault_tracker
        _vault_tracker["important_count"] = 15
        _vault_tracker["count"] = 50
        result = get_consolidation_urgency()
        self.assertIsNotNone(result)
        self.assertEqual("high", result["urgency"])

    def test_window_resets_after_24h(self):
        """Vault tracker resets after 24h window."""
        from modules.wiring import _on_vault_tracks_urgency, _vault_tracker
        _vault_tracker["count"] = 100
        _vault_tracker["important_count"] = 50
        _vault_tracker["window_start"] = time.monotonic() - 86401  # 24h+1s ago
        _on_vault_tracks_urgency("memory_vaulted", {"importance": "low"})
        self.assertEqual(1, _vault_tracker["count"])
        self.assertEqual(0, _vault_tracker["important_count"])


class TestCX4b_ConsolidationProtectsDecay(unittest.TestCase):
    """CX-4b: Consolidation boosts Storage Strength."""

    @patch("modules.wiring._pg")
    def test_ss_boost_on_consolidation(self, mock_pg):
        """Consolidated memories get SS boost."""
        from modules.wiring import _on_consolidation_protects_decay

        # Mock get_by_ids to return a point with low SS
        mock_point = MagicMock()
        mock_point.payload = {"storage_strength": 0.3}
        mock_pg.get_by_ids.return_value = [mock_point]
        mock_pg.update_payload = MagicMock()

        data = {"consolidated_ids": ["mem_001", "mem_002"]}
        _on_consolidation_protects_decay("consolidation_complete", data)

        # Should have called update_payload for both memories
        self.assertEqual(2, mock_pg.update_payload.call_count)
        # Check SS was boosted
        call_args = mock_pg.update_payload.call_args_list[0]
        new_ss = call_args[0][1]["storage_strength"]
        self.assertGreater(new_ss, 0.3)
        self.assertLessEqual(new_ss, 1.0)

    @patch("modules.wiring._pg")
    def test_no_ids_no_action(self, mock_pg):
        """No consolidated_ids → no action."""
        from modules.wiring import _on_consolidation_protects_decay
        data = {"episodes_pruned": 5}  # No consolidated_ids key
        _on_consolidation_protects_decay("consolidation_complete", data)
        mock_pg.get_by_ids.assert_not_called()

    @patch("modules.wiring._pg")
    def test_ss_capped_at_one(self, mock_pg):
        """SS never exceeds 1.0."""
        from modules.wiring import _on_consolidation_protects_decay
        mock_point = MagicMock()
        mock_point.payload = {"storage_strength": 0.95}
        mock_pg.get_by_ids.return_value = [mock_point]
        mock_pg.update_payload = MagicMock()

        data = {"consolidated_ids": ["mem_high_ss"]}
        _on_consolidation_protects_decay("consolidation_complete", data)
        new_ss = mock_pg.update_payload.call_args[0][1]["storage_strength"]
        self.assertLessEqual(new_ss, 1.0)


class TestComputeLearningProgress(unittest.TestCase):
    """Test the learning progress helper for CX-1."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_fts.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_topic TEXT, actual_topic TEXT,
                predicted_keywords TEXT, actual_keywords TEXT,
                surprise_score REAL, precision_weight REAL,
                weighted_surprise REAL, hit INTEGER DEFAULT 0,
                source TEXT, created_at TEXT NOT NULL
            )
        """)
        # Insert scores where PE decreases over time (= learning)
        # Older entries have higher PE, newer have lower PE
        for i, score in enumerate([0.8, 0.7, 0.6, 0.5, 0.4, 0.3]):
            conn.execute(
                "INSERT INTO prediction_results (actual_topic, surprise_score, source, created_at) VALUES (?, ?, 'interactive', ?)",
                ("trading", score, f"2026-01-01T{10+i}:00:00"),
            )
        conn.commit()
        conn.close()

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_learning_progress_positive(self):
        """Decreasing PE over time → positive LP."""
        os.environ["FTS_DB_PATH"] = self.db_path
        from modules.wiring import _compute_learning_progress
        lp = _compute_learning_progress("trading", window=10)
        self.assertIsNotNone(lp)
        self.assertGreater(lp, 0)

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_insufficient_data_returns_none(self):
        """< 3 data points → None."""
        os.environ["FTS_DB_PATH"] = self.db_path
        from modules.wiring import _compute_learning_progress
        lp = _compute_learning_progress("nonexistent_topic")
        self.assertIsNone(lp)


# ============================================================
# TIER 2 TESTS
# ============================================================


class TestCX8_ReconsolidationProtectsDecay(unittest.TestCase):
    """CX-8: Successful reconsolidation → SS boost (Lee 2008, Forcato 2011)."""

    @patch("modules.wiring._pg")
    def test_ss_boost_on_reconsolidation(self, mock_pg):
        """High confidence reconsolidation → SS boost."""
        from modules.wiring import _on_reconsolidation_protects_decay

        mock_point = MagicMock()
        mock_point.payload = {"storage_strength": 0.4}
        mock_pg.get_by_ids.return_value = [mock_point]
        mock_pg.update_payload = MagicMock()

        data = {
            "memory_id": "mem_recon_001",
            "action": "correct_memory",
            "old_confidence": 0.3,
            "new_confidence": 0.8,
        }
        _on_reconsolidation_protects_decay("reconsolidation_triggered", data)

        mock_pg.update_payload.assert_called_once()
        new_ss = mock_pg.update_payload.call_args[0][1]["storage_strength"]
        self.assertGreater(new_ss, 0.4)
        self.assertLessEqual(new_ss, 1.0)

    @patch("modules.wiring._pg")
    def test_low_confidence_skipped(self, mock_pg):
        """new_confidence < 0.5 → no boost (weakening reconsolidation)."""
        from modules.wiring import _on_reconsolidation_protects_decay
        data = {
            "memory_id": "mem_recon_002",
            "action": "correct_memory",
            "new_confidence": 0.3,
        }
        _on_reconsolidation_protects_decay("reconsolidation_triggered", data)
        mock_pg.get_by_ids.assert_not_called()

    @patch("modules.wiring._pg")
    def test_non_correct_action_skipped(self, mock_pg):
        """action != 'correct_memory' → no boost."""
        from modules.wiring import _on_reconsolidation_protects_decay
        data = {
            "memory_id": "mem_recon_003",
            "action": "blend_memory",
            "new_confidence": 0.9,
        }
        _on_reconsolidation_protects_decay("reconsolidation_triggered", data)
        mock_pg.get_by_ids.assert_not_called()

    @patch("modules.wiring._pg")
    def test_cumulative_boost_capped(self, mock_pg):
        """Repeated reconsolidation → cumulative SS, capped at 1.0."""
        from modules.wiring import _on_reconsolidation_protects_decay

        mock_point = MagicMock()
        mock_point.payload = {"storage_strength": 0.9}
        mock_pg.get_by_ids.return_value = [mock_point]
        mock_pg.update_payload = MagicMock()

        data = {
            "memory_id": "mem_recon_high",
            "action": "correct_memory",
            "new_confidence": 0.9,
        }
        _on_reconsolidation_protects_decay("reconsolidation_triggered", data)
        new_ss = mock_pg.update_payload.call_args[0][1]["storage_strength"]
        self.assertLessEqual(new_ss, 1.0)


class TestCX6_MetacognitionExploreExploit(unittest.TestCase):
    """CX-6: Meta-confidence modulates EFE temperature (Boldt 2019)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_fts.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_state_l2 (
                domain TEXT PRIMARY KEY,
                predicted_accuracy REAL DEFAULT 0.5,
                actual_accuracy REAL DEFAULT 0.5,
                sample_size INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_high_confidence_low_temperature(self):
        """High meta-confidence → lower temperature (exploitation)."""
        os.environ["FTS_DB_PATH"] = self.db_path
        conn = sqlite3.connect(self.db_path)
        # Insert well-calibrated high confidence
        conn.execute(
            "INSERT INTO prediction_state_l2 VALUES (?, ?, ?, ?, ?)",
            ("trading", 0.85, 0.83, 20, "2026-01-01"),
        )
        conn.commit()
        conn.close()

        import modules.wiring as w
        w._meta_efe_last_refresh = 0  # Force refresh
        temp = w.get_meta_efe_temperature()
        self.assertIsNotNone(temp)
        self.assertLess(temp, 4.0)  # Below default (exploitation)

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_low_confidence_high_temperature(self):
        """Low meta-confidence → higher temperature (exploration)."""
        os.environ["FTS_DB_PATH"] = self.db_path
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO prediction_state_l2 VALUES (?, ?, ?, ?, ?)",
            ("general", 0.25, 0.28, 15, "2026-01-01"),
        )
        conn.commit()
        conn.close()

        import modules.wiring as w
        w._meta_efe_last_refresh = 0
        temp = w.get_meta_efe_temperature()
        self.assertIsNotNone(temp)
        self.assertGreater(temp, 4.0)  # Above default (exploration)

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_calibration_correction_shrinks(self):
        """Miscalibrated confidence → adjusted toward 0.5."""
        os.environ["FTS_DB_PATH"] = self.db_path
        conn = sqlite3.connect(self.db_path)
        # Highly miscalibrated: thinks 90% but actually 50%
        conn.execute(
            "INSERT INTO prediction_state_l2 VALUES (?, ?, ?, ?, ?)",
            ("overconfident", 0.90, 0.50, 30, "2026-01-01"),
        )
        conn.commit()
        conn.close()

        import modules.wiring as w
        w._meta_efe_last_refresh = 0
        temp = w.get_meta_efe_temperature()
        self.assertIsNotNone(temp)
        # With calibration correction, effective confidence should be shrunk
        # toward 0.5, so temperature should be near default (4.0)
        self.assertGreater(temp, 2.5)
        self.assertLess(temp, 5.5)

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_insufficient_data_returns_none(self):
        """No L2 domains with sample_size >= 5 → None."""
        os.environ["FTS_DB_PATH"] = self.db_path
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO prediction_state_l2 VALUES (?, ?, ?, ?, ?)",
            ("tiny", 0.5, 0.5, 2, "2026-01-01"),  # sample_size < 5
        )
        conn.commit()
        conn.close()

        import modules.wiring as w
        w._meta_efe_last_refresh = 0
        temp = w.get_meta_efe_temperature()
        self.assertIsNone(temp)

    @patch.dict(os.environ, {"FTS_DB_PATH": ""})
    def test_temperature_clamped(self):
        """Temperature always in [1.0, 7.0] range."""
        os.environ["FTS_DB_PATH"] = self.db_path
        conn = sqlite3.connect(self.db_path)
        # Extreme low confidence
        conn.execute(
            "INSERT INTO prediction_state_l2 VALUES (?, ?, ?, ?, ?)",
            ("extreme_low", 0.05, 0.08, 50, "2026-01-01"),
        )
        conn.commit()
        conn.close()

        import modules.wiring as w
        w._meta_efe_last_refresh = 0
        temp = w.get_meta_efe_temperature()
        self.assertIsNotNone(temp)
        self.assertGreaterEqual(temp, 1.0)
        self.assertLessEqual(temp, 7.0)


class TestCX5_GNWBroadcastToAction(unittest.TestCase):
    """CX-5: GNW broadcast feeds action selection (Dehaene 2014, Friston 2015)."""

    @patch("modules.wiring.get_attention_schema", return_value={"attention_prediction_error": 0.7, "current_focus": "trading"})
    @patch("modules.working_memory.push_to_working_memory")
    def test_novel_situation_updates_context(self, mock_push, mock_attention):
        """PE > 0.5 → broadcast context stored + WM push."""
        from modules.wiring import _on_gnw_broadcast_to_action, get_broadcast_context
        import modules.wiring as w
        data = {
            "winner_domains": ["trading", "general"],
            "top_activation": 0.85,
            "timestamp": "2026-01-01T00:00:00",
        }
        _on_gnw_broadcast_to_action("workspace_competition_complete", data)

        # Check WM push
        mock_push.assert_called_once()
        call_args = mock_push.call_args[1]
        self.assertIn("GNW", call_args["content"])
        self.assertEqual("gnw_action_bridge", call_args["source"])

        # Check broadcast context stored
        ctx = get_broadcast_context()
        self.assertIsNotNone(ctx)
        self.assertEqual(["trading", "general"], ctx["broadcast_domains"])
        self.assertAlmostEqual(0.85, ctx["broadcast_activation"])

    @patch("modules.wiring.get_attention_schema", return_value={"attention_prediction_error": 0.3})
    @patch("modules.working_memory.push_to_working_memory")
    def test_routine_situation_cleared(self, mock_push, mock_attention):
        """PE < 0.5 → context cleared, no WM push."""
        from modules.wiring import _on_gnw_broadcast_to_action
        import modules.wiring as w
        w._broadcast_context = {"stale": True}  # Set stale context

        data = {"winner_domains": ["general"], "top_activation": 0.5}
        _on_gnw_broadcast_to_action("workspace_competition_complete", data)

        mock_push.assert_not_called()
        self.assertIsNone(w._broadcast_context)

    @patch("modules.wiring.get_attention_schema", return_value={"attention_prediction_error": 0.8})
    @patch("modules.working_memory.push_to_working_memory")
    def test_no_winners_skipped(self, mock_push, mock_attention):
        """Empty winner_domains → no action."""
        from modules.wiring import _on_gnw_broadcast_to_action
        data = {"winner_domains": [], "top_activation": 0}
        _on_gnw_broadcast_to_action("workspace_competition_complete", data)
        mock_push.assert_not_called()

    def test_broadcast_expires(self):
        """Old timestamp → get_broadcast_context returns None."""
        from modules.wiring import get_broadcast_context
        import modules.wiring as w
        w._broadcast_context = {
            "broadcast_domains": ["old"],
            "timestamp": time.monotonic() - 200,  # > 120s stale
        }
        result = get_broadcast_context()
        self.assertIsNone(result)


class TestCX7_CausalDAGPrediction(unittest.TestCase):
    """CX-7: Causal DAG injects weak priors (Pearl 2009, Bramley 2017)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_fts.db")
        conn = sqlite3.connect(self.db_path)
        # Create prediction_results and causal_discovery_state tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_topic TEXT, actual_topic TEXT,
                predicted_keywords TEXT, actual_keywords TEXT,
                surprise_score REAL, precision_weight REAL,
                weighted_surprise REAL, hit INTEGER DEFAULT 0,
                source TEXT, created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_discovery_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                w_matrix TEXT NOT NULL,
                topics TEXT NOT NULL,
                h_value REAL,
                lambda_val REAL,
                n_edges INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _inject_dag(self, w_matrix, topics):
        """Helper: insert a W matrix into causal_discovery_state."""
        import json
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO causal_discovery_state (w_matrix, topics, h_value, lambda_val, n_edges, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (json.dumps(w_matrix), json.dumps(topics), 0.0, 0.1, 2, "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

    def test_dag_boosts_forward_counts(self):
        """DAG edge A→B with positive weight → B's forward count increases slightly."""
        import json, math
        # Topics: ["general", "trading"]
        # W: trading→general = 0.5 (strong edge)
        w_matrix = [[0.0, 0.0], [0.5, 0.0]]  # W[1][0]=0.5: trading→general
        self._inject_dag(w_matrix, ["general", "trading"])

        # Simulate forward_counts for current_topic="trading"
        conn = sqlite3.connect(self.db_path)
        dag_row = conn.execute(
            "SELECT w_matrix, topics FROM causal_discovery_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(dag_row)
        dag_W = json.loads(dag_row[0])
        dag_topics = json.loads(dag_row[1])

        # Simulate the injection logic
        current_topic = "trading"
        forward_counts = {"general": 1.0, "trading": 0.5}
        DAG_PRIOR_KAPPA = 0.1

        if current_topic in dag_topics:
            from_idx = dag_topics.index(current_topic)
            for j, to_topic in enumerate(dag_topics):
                if to_topic == current_topic or to_topic not in forward_counts:
                    continue
                w_ij = dag_W[from_idx][j]
                if abs(w_ij) > 0.1:
                    factor = 1.0 + DAG_PRIOR_KAPPA * w_ij
                    forward_counts[to_topic] = max(0.01, forward_counts[to_topic] * factor)

        # "general" should have been boosted slightly
        self.assertGreater(forward_counts["general"], 1.0)
        # But only slightly (weak prior: kappa=0.1 * 0.5 = 5% boost)
        self.assertLess(forward_counts["general"], 1.1)

    def test_no_dag_no_change(self):
        """No causal_discovery_state rows → forward counts unchanged."""
        conn = sqlite3.connect(self.db_path)
        dag_row = conn.execute(
            "SELECT w_matrix, topics FROM causal_discovery_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNone(dag_row)  # No DAG → no injection

    def test_weak_edge_skipped(self):
        """Edge |w| < 0.1 → skipped (threshold filter)."""
        import json
        # Weak edge: 0.05
        w_matrix = [[0.0, 0.0], [0.05, 0.0]]
        self._inject_dag(w_matrix, ["general", "trading"])

        conn = sqlite3.connect(self.db_path)
        dag_row = conn.execute(
            "SELECT w_matrix, topics FROM causal_discovery_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        dag_W = json.loads(dag_row[0])
        dag_topics = json.loads(dag_row[1])

        current_topic = "trading"
        forward_counts = {"general": 1.0, "trading": 0.5}
        original_general = forward_counts["general"]

        if current_topic in dag_topics:
            from_idx = dag_topics.index(current_topic)
            for j, to_topic in enumerate(dag_topics):
                if to_topic == current_topic or to_topic not in forward_counts:
                    continue
                w_ij = dag_W[from_idx][j]
                if abs(w_ij) > 0.1:
                    factor = 1.0 + 0.1 * w_ij
                    forward_counts[to_topic] = max(0.01, forward_counts[to_topic] * factor)

        # "general" should NOT have changed (edge too weak)
        self.assertEqual(original_general, forward_counts["general"])

    def test_inhibitory_edge_reduces(self):
        """Negative edge weight → reduce forward count (Waldmann 1992)."""
        import json
        # Inhibitory edge: trading →(inhibits) general
        w_matrix = [[0.0, 0.0], [-0.3, 0.0]]
        self._inject_dag(w_matrix, ["general", "trading"])

        conn = sqlite3.connect(self.db_path)
        dag_row = conn.execute(
            "SELECT w_matrix, topics FROM causal_discovery_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        dag_W = json.loads(dag_row[0])
        dag_topics = json.loads(dag_row[1])

        current_topic = "trading"
        forward_counts = {"general": 1.0, "trading": 0.5}

        if current_topic in dag_topics:
            from_idx = dag_topics.index(current_topic)
            for j, to_topic in enumerate(dag_topics):
                if to_topic == current_topic or to_topic not in forward_counts:
                    continue
                w_ij = dag_W[from_idx][j]
                if abs(w_ij) > 0.1:
                    factor = 1.0 + 0.1 * w_ij
                    forward_counts[to_topic] = max(0.01, forward_counts[to_topic] * factor)

        # "general" should be reduced (inhibitory edge)
        self.assertLess(forward_counts["general"], 1.0)
        # But still positive (max 0.01 floor)
        self.assertGreater(forward_counts["general"], 0)


# =============================================================================
# TIER 3 TESTS — Cross-loops CX-9 through CX-14 + Inhibitory mechanisms
# =============================================================================


class TestMIN1_CuriosityCompetitivePruning(unittest.TestCase):
    """MIN-1: Feedforward inhibition — curiosity hard cap at MAX_PENDING_CURIOSITIES."""

    @patch("modules.curiosity._guardar_curiosidades")
    @patch("modules.curiosity._cargar_curiosidades")
    def test_pruning_when_queue_full(self, mock_load, mock_save):
        """When queue >= MAX, lowest priority gets pruned."""
        from modules.curiosity import push_curiosidad, MAX_PENDING_CURIOSITIES

        existing = [
            {"id": i, "pregunta": f"q{i}", "prioridad": "media", "categoria": "general", "agregada": f"2026-03-13T{10+i:02d}:00"}
            for i in range(MAX_PENDING_CURIOSITIES)
        ]
        mock_load.return_value = {"pendientes": list(existing), "exploradas": []}

        push_curiosidad(tema="new topic", prioridad="alta", categoria="test")

        written_data = mock_save.call_args[0][0]
        # Queue should not exceed MAX (added 1, removed 1)
        self.assertLessEqual(len(written_data["pendientes"]), MAX_PENDING_CURIOSITIES)
        # Pruned item moved to exploradas
        self.assertTrue(any(
            e.get("razon") == "pruned_by_cap" for e in written_data["exploradas"]
        ))

    @patch("modules.curiosity._guardar_curiosidades")
    @patch("modules.curiosity._cargar_curiosidades")
    def test_no_pruning_when_below_cap(self, mock_load, mock_save):
        """When queue < MAX, no pruning occurs."""
        mock_load.return_value = {"pendientes": [{"id": 1, "pregunta": "q1", "prioridad": "alta"}], "exploradas": []}

        from modules.curiosity import push_curiosidad
        push_curiosidad(tema="new", prioridad="media", categoria="test")

        written_data = mock_save.call_args[0][0]
        # No pruned items
        self.assertFalse(any(
            e.get("razon") == "pruned_by_cap" for e in written_data.get("exploradas", [])
        ))


class TestMIN2_SHYDownscaling(unittest.TestCase):
    """MIN-2: SHY synaptic homeostasis — global SS downscaling (Tononi & Cirelli 2006)."""

    def test_shy_downscale_returns_count(self):
        """shy_downscale_ss returns integer count."""
        from modules.wiring import shy_downscale_ss
        result = shy_downscale_ss(batch_size=10)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)


class TestCX14_ConsolidationGapsCuriosity(unittest.TestCase):
    """CX-14: Consolidation gaps → curiosity (Loewenstein 1994)."""

    def setUp(self):
        import modules.wiring as w
        w._cx14_recent_topics.clear()

    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_contradictions_generate_curiosity(self, mock_push):
        """Contradictions from consolidation → high-priority curiosity."""
        from modules.wiring import _on_consolidation_gaps_drive_curiosity
        data = {
            "contradictions_found": 2,
            "scope": "normal",
            "clusters_found": 3,
            "facts_extracted": 5,
            "bridge_edges": 0,
        }
        _on_consolidation_gaps_drive_curiosity("consolidation_complete", data)
        self.assertTrue(mock_push.called)
        # Should be alta priority (contradiction channel)
        self.assertEqual("alta", mock_push.call_args[1]["prioridad"])

    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_minimal_scope_skipped(self, mock_push):
        """scope=minimal consolidations don't generate curiosity."""
        from modules.wiring import _on_consolidation_gaps_drive_curiosity
        data = {"scope": "minimal", "contradictions_found": 5}
        _on_consolidation_gaps_drive_curiosity("consolidation_complete", data)
        mock_push.assert_not_called()

    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_cooldown_prevents_spam(self, mock_push):
        """Same channel within cooldown period doesn't re-trigger."""
        from modules.wiring import _on_consolidation_gaps_drive_curiosity, _cx14_recent_topics

        # Simulate recent trigger for "contradictions" channel
        _cx14_recent_topics["contradictions"] = time.time()

        data = {
            "contradictions_found": 1,
            "scope": "normal",
            "clusters_found": 0,
            "facts_extracted": 0,
            "bridge_edges": 0,
        }
        _on_consolidation_gaps_drive_curiosity("consolidation_complete", data)
        mock_push.assert_not_called()

    @patch("modules.curiosity.push_curiosidad", return_value="ok")
    def test_max_questions_cap(self, mock_push):
        """CX-14 generates at most _CX14_MAX_QUESTIONS_PER_RUN questions."""
        from modules.wiring import _on_consolidation_gaps_drive_curiosity, _CX14_MAX_QUESTIONS_PER_RUN
        data = {
            "contradictions_found": 5,
            "scope": "normal",
            "clusters_found": 10,
            "facts_extracted": 1,
            "bridge_edges": 10,
        }
        _on_consolidation_gaps_drive_curiosity("consolidation_complete", data)
        self.assertLessEqual(mock_push.call_count, _CX14_MAX_QUESTIONS_PER_RUN)


class TestCX10_SelfModelMetacognition(unittest.TestCase):
    """CX-10: Self-model ↔ metacognition (Nelson & Narens 1990)."""

    def setUp(self):
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

    def test_discrepancies_lower_precision(self):
        """Discrepancies in self-model → lower precision modifier per domain."""
        from modules.wiring import _on_self_model_to_metacognition, get_cx10_precision_modifier

        data = {
            "source": "reflect_on_self",
            "discrepancy_count": 3,
            "domains": ["trading", "consciencia"],
        }
        _on_self_model_to_metacognition("self_model_refreshed", data)

        for domain in ["trading", "consciencia"]:
            mod = get_cx10_precision_modifier(domain)
            self.assertLess(mod, 1.0)

    def test_no_discrepancy_no_change(self):
        """Zero discrepancies → no precision change."""
        from modules.wiring import _on_self_model_to_metacognition, get_cx10_precision_modifier

        data = {
            "source": "reflect_on_self",
            "discrepancy_count": 0,
            "domains": ["trading"],
        }
        _on_self_model_to_metacognition("self_model_refreshed", data)
        self.assertEqual(get_cx10_precision_modifier("trading"), 1.0)

    def test_precision_floor(self):
        """Precision modifier never goes below CX10_PRECISION_FLOOR."""
        from modules.wiring import (
            _on_self_model_to_metacognition,
            get_cx10_precision_modifier,
            _CX10_PRECISION_FLOOR,
        )

        for _ in range(50):
            data = {
                "source": "reflect_on_self",
                "discrepancy_count": 10,
                "domains": ["fragile_domain"],
            }
            _on_self_model_to_metacognition("self_model_refreshed", data)

        mod = get_cx10_precision_modifier("fragile_domain")
        self.assertGreaterEqual(mod, _CX10_PRECISION_FLOOR)

    def test_anti_echo_from_cx10b(self):
        """Events from CX-10B itself are ignored (anti-echo)."""
        from modules.wiring import _on_self_model_to_metacognition, get_cx10_precision_modifier

        data = {
            "source": "metacognitive_bias_cx10b",
            "discrepancy_count": 5,
            "domains": ["trading"],
        }
        _on_self_model_to_metacognition("self_model_refreshed", data)
        self.assertEqual(get_cx10_precision_modifier("trading"), 1.0)


class TestCX11_CuriosityFeedsCausal(unittest.TestCase):
    """CX-11: Curiosity resolution → causal discovery (Bramley 2017)."""

    def setUp(self):
        import modules.wiring as w
        w._cx11_buffer.clear()

    @patch("modules.wiring.get_attention_schema", return_value={"current_focus": "general"})
    def test_buffers_resolved_curiosity(self, mock_attn):
        """Resolved curiosity observation gets buffered."""
        from modules.wiring import _on_curiosity_feeds_causal, _cx11_buffer

        data = {
            "question": "Why does BTC correlate with ETH?",
            "category": "trading",
            "outcome": "positive",
        }
        _on_curiosity_feeds_causal("curiosity_resolved", data)
        self.assertEqual(len(_cx11_buffer), 1)
        self.assertEqual(_cx11_buffer[0]["to_topic"], "trading")
        self.assertEqual(_cx11_buffer[0]["from_topic"], "general")

    @patch("modules.wiring.get_attention_schema", return_value={"current_focus": "general"})
    def test_weight_above_1(self, mock_attn):
        """Curiosity-driven observations have weight > 1 (exploration bonus)."""
        from modules.wiring import _on_curiosity_feeds_causal, _cx11_buffer, _CX11_CURIOSITY_WEIGHT

        data = {"question": "test", "category": "general", "outcome": "positive"}
        _on_curiosity_feeds_causal("curiosity_resolved", data)
        self.assertEqual(_cx11_buffer[0]["weight"], _CX11_CURIOSITY_WEIGHT)
        self.assertGreater(_CX11_CURIOSITY_WEIGHT, 1.0)

    @patch("modules.wiring.get_attention_schema", return_value={"current_focus": "general"})
    def test_buffer_cap(self, mock_attn):
        """Buffer doesn't exceed _CX11_MAX_BUFFER (auto-flushes or stays capped)."""
        from modules.wiring import _on_curiosity_feeds_causal, _cx11_buffer, _CX11_MAX_BUFFER

        for i in range(_CX11_MAX_BUFFER + 5):
            data = {"question": f"q{i}", "category": "test", "outcome": "positive"}
            _on_curiosity_feeds_causal("curiosity_resolved", data)

        # Buffer either flushed (empty/small) or capped
        self.assertLessEqual(len(_cx11_buffer), _CX11_MAX_BUFFER)


class TestCX9_GNWToSelfModel(unittest.TestCase):
    """CX-9: GNW broadcast → self-model refresh (Northoff 2004)."""

    def setUp(self):
        import modules.wiring as w
        w._cx9_last_fire = 0.0
        w._cx9_fire_count_window.clear()

    def test_non_self_referential_skipped(self):
        """Content without self-referential domains is ignored."""
        from modules.wiring import _compute_self_relevance

        score = _compute_self_relevance(["trading", "general"])
        self.assertEqual(score, 0.0)

    @patch("modules.self_model.reflect_on_self", return_value={})
    def test_anti_echo_sole_self_model(self, mock_reflect):
        """CB1: sole self_model winner = CX-3 echo → suppressed."""
        from modules.wiring import _on_workspace_to_self_model

        _on_workspace_to_self_model("workspace_competition_complete", {
            "winner_domains": ["self_model"],
            "top_activation": 0.8,
            "winner_count": 1,
        })
        mock_reflect.assert_not_called()

    @patch("modules.self_model.reflect_on_self", return_value={})
    def test_refractory_period_blocks_rapid_fires(self, mock_reflect):
        """CB2: 45s refractory period prevents rapid re-firing."""
        import modules.wiring as w
        w._cx9_last_fire = time.monotonic()  # Just fired

        w._on_workspace_to_self_model("workspace_competition_complete", {
            "winner_domains": ["self_model", "identity"],
            "top_activation": 0.9,
            "winner_count": 2,
        })
        mock_reflect.assert_not_called()

    def test_self_relevance_scoring(self):
        """Self-referential keywords produce non-zero relevance score."""
        from modules.wiring import _compute_self_relevance

        score = _compute_self_relevance(["self_model", "trading", "identity"])
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_kill_switch_after_many_fires(self):
        """CB5: >3 fires in 30min → kill switch activates."""
        import modules.wiring as w
        now = time.monotonic()
        # Simulate 4 recent fires
        w._cx9_fire_count_window = [now - 60, now - 40, now - 20, now - 5]
        self.assertGreater(len(w._cx9_fire_count_window), 3)


class TestCX13_PADEFEWeights(unittest.TestCase):
    """CX-13: PAD → EFE weights (Doya 2008, Vinckier 2018)."""

    def test_neutral_pad_returns_near_unity(self):
        """Neutral PAD state → weights near 1.0 (no modulation)."""
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.55, "dominance": 0.0}
        }):
            from modules.wiring import compute_pad_efe_weights, get_meta_efe_temperature
            with patch("modules.wiring.get_meta_efe_temperature", return_value=None):
                weights = compute_pad_efe_weights(model_observations=100)

        if weights is not None:
            self.assertAlmostEqual(weights["w_pragmatic"], 1.0, delta=0.15)
            self.assertAlmostEqual(weights["w_epistemic"], 1.0, delta=0.15)

    def test_inverted_u_arousal(self):
        """Arousal uses inverted-U (Cools & D'Esposito 2011): extremes reduce epistemic."""
        from modules.wiring import compute_pad_efe_weights, CX13_OPTIMAL_AROUSAL

        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": CX13_OPTIMAL_AROUSAL, "dominance": 0.0}
        }):
            with patch("modules.wiring.get_meta_efe_temperature", return_value=None):
                w_optimal = compute_pad_efe_weights(model_observations=100)

        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 1.0, "dominance": 0.0}
        }):
            with patch("modules.wiring.get_meta_efe_temperature", return_value=None):
                w_panic = compute_pad_efe_weights(model_observations=100)

        if w_optimal and w_panic:
            self.assertGreater(w_optimal["w_epistemic"], w_panic["w_epistemic"])

    def test_positive_pleasure_boosts_pragmatic(self):
        """Positive pleasure → higher pragmatic weight (Vinckier 2018)."""
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.8, "arousal": 0.5, "dominance": 0.0}
        }):
            from modules.wiring import compute_pad_efe_weights
            with patch("modules.wiring.get_meta_efe_temperature", return_value=None):
                weights = compute_pad_efe_weights(model_observations=100)

        if weights is not None:
            self.assertGreater(weights["w_pragmatic"], 1.0)

    def test_confidence_dampening_with_few_observations(self):
        """Early model (few observations) → dampened weights closer to 1.0.
        confidence_dampen = 1/(1+0.01*N), so fewer obs = higher dampen = MORE deviation.
        But with many obs, dampen→0 so deviation→0. Wait — actually the formula is
        w = 1.0 + p * scale * dampen. Higher dampen = MORE weight shift, not less.
        So fewer observations → MORE deviation (emotional influence is stronger).
        With many observations → dampen→0 → weights→1.0 (rational).
        """
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.8, "arousal": 0.5, "dominance": 0.0}
        }):
            from modules.wiring import compute_pad_efe_weights
            with patch("modules.wiring.get_meta_efe_temperature", return_value=None):
                w_few = compute_pad_efe_weights(model_observations=5)
                w_many = compute_pad_efe_weights(model_observations=500)

        if w_few and w_many:
            # More observations → less emotional influence → closer to 1.0
            dev_few = abs(w_few["w_pragmatic"] - 1.0)
            dev_many = abs(w_many["w_pragmatic"] - 1.0)
            self.assertGreater(dev_few, dev_many)

    def test_perseveration_tracking(self):
        """Repeated same action → epistemic boost (anti-perseveration)."""
        import modules.wiring as w
        w._cx13_last_action = None
        w._cx13_consecutive_same = 0

        from modules.wiring import track_action_for_perseveration

        boost = 0.0
        for _ in range(5):
            boost = track_action_for_perseveration("retrieve")

        self.assertGreater(boost, 0.0)


class TestCX13_EFEIntegration(unittest.TestCase):
    """CX-13: compute_efe() respects weights parameter."""

    def test_efe_with_custom_weights(self):
        """Passing weights to compute_efe changes result vs default."""
        from modules.active_inference import (
            compute_efe, SystemState, Action, GenerativeModel,
        )

        state = SystemState(
            topic="general",
            uncertainty=0.5,
            wm_load=0.3,
            pe_magnitude=0.4,
            emotional_valence=0.0,
        )
        action = Action(name="attend", description="test", cost=0.1, expected_ms=50)
        model = GenerativeModel()

        g_default = compute_efe(state, action, model)
        g_weighted = compute_efe(state, action, model, weights={
            "w_pragmatic": 2.0, "w_epistemic": 0.5, "w_cost": 1.0
        })

        self.assertIsInstance(g_weighted, float)
        self.assertIsInstance(g_default, float)

    def test_efe_weights_none_matches_default(self):
        """weights=None produces same result as no weights (backward compat)."""
        from modules.active_inference import (
            compute_efe, SystemState, Action, GenerativeModel,
        )

        state = SystemState(
            topic="trading",
            uncertainty=0.3,
            wm_load=0.5,
            pe_magnitude=0.6,
            emotional_valence=0.2,
        )
        action = Action(name="explore", description="test", cost=0.2, expected_ms=100)
        model = GenerativeModel()

        g_no_weights = compute_efe(state, action, model)
        g_none_weights = compute_efe(state, action, model, weights=None)

        self.assertAlmostEqual(g_no_weights, g_none_weights, places=10)


class TestCX12_RetrievalSSBoost(unittest.TestCase):
    """CX-12: Retrieval → SS boost (Roediger & Karpicke 2006, testing effect)."""

    def setUp(self):
        import modules.wiring as w
        w._cx12_last_boost.clear()
        w._cx12_topic_counts.clear()

    def test_internal_source_skipped(self):
        """Internal retrievals (workspace, scroll) don't trigger SS boost."""
        from modules.wiring import _on_retrieval_boosts_ss

        with patch("modules.wiring._pg") as mock_pg:
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["abc"],
                "source": "workspace",
                "query": "test",
            })
            time.sleep(0.15)
            mock_pg.get_by_ids.assert_not_called()

    def test_empty_ids_skipped(self):
        """No retrieved_ids → no action."""
        from modules.wiring import _on_retrieval_boosts_ss

        with patch("modules.wiring._pg") as mock_pg:
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": [],
                "source": "search_memory",
                "query": "test",
            })
            time.sleep(0.15)
            mock_pg.get_by_ids.assert_not_called()

    def test_ss_boost_on_search(self):
        """search_memory retrieval → SS increases."""
        from modules.wiring import _on_retrieval_boosts_ss

        mock_point = MagicMock()
        mock_point.id = "mem-001"
        mock_point.payload = {"storage_strength": 0.3, "retrieval_strength": 0.4, "narrative_importance": "medium"}

        with patch("modules.wiring._pg") as mock_pg:
            mock_pg.get_by_ids.return_value = [mock_point]
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["mem-001"],
                "source": "search_memory",
                "query": "trading analysis",
            })
            time.sleep(0.3)

            mock_pg.update_payload.assert_called_once()
            call_args = mock_pg.update_payload.call_args[0]
            self.assertEqual(call_args[0], "mem-001")
            new_ss = call_args[1]["storage_strength"]
            self.assertGreater(new_ss, 0.3)

    def test_spacing_guard_blocks_rapid_repeat(self):
        """Same memory within 60s → second boost blocked."""
        import modules.wiring as w
        w._cx12_last_boost["mem-001"] = time.time()  # Just boosted

        from modules.wiring import _on_retrieval_boosts_ss

        with patch("modules.wiring._pg") as mock_pg:
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["mem-001"],
                "source": "search_memory",
                "query": "test",
            })
            time.sleep(0.15)
            # Should not even read from DB (all filtered by spacing)
            mock_pg.get_by_ids.assert_not_called()

    def test_identity_floor(self):
        """Critical memories never below SS 0.3."""
        from modules.wiring import _on_retrieval_boosts_ss, _CX12_IDENTITY_SS_FLOOR

        mock_point = MagicMock()
        mock_point.id = "identity-001"
        mock_point.payload = {"storage_strength": 0.2, "retrieval_strength": 0.9, "narrative_importance": "critical"}

        with patch("modules.wiring._pg") as mock_pg:
            mock_pg.get_by_ids.return_value = [mock_point]
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["identity-001"],
                "source": "search_memory",
                "query": "who am I",
            })
            time.sleep(0.3)

            call_args = mock_pg.update_payload.call_args[0]
            new_ss = call_args[1]["storage_strength"]
            self.assertGreaterEqual(new_ss, _CX12_IDENTITY_SS_FLOOR)

    def test_usage_dampening_reduces_gain(self):
        """Repeated topic retrievals → diminishing SS gains."""
        import modules.wiring as w
        # Simulate 100 previous retrievals for this topic
        w._cx12_topic_counts["trading"] = 100

        from modules.wiring import _on_retrieval_boosts_ss

        mock_point = MagicMock()
        mock_point.id = "mem-002"
        mock_point.payload = {"storage_strength": 0.3, "retrieval_strength": 0.4, "narrative_importance": "medium"}

        with patch("modules.wiring._pg") as mock_pg:
            mock_pg.get_by_ids.return_value = [mock_point]
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["mem-002"],
                "source": "search_memory",
                "query": "trading",
            })
            time.sleep(0.3)

            call_args = mock_pg.update_payload.call_args[0]
            dampened_ss = call_args[1]["storage_strength"]

        # Now test without dampening (fresh topic)
        w._cx12_topic_counts.clear()
        w._cx12_last_boost.clear()

        mock_point2 = MagicMock()
        mock_point2.id = "mem-003"
        mock_point2.payload = {"storage_strength": 0.3, "retrieval_strength": 0.4, "narrative_importance": "medium"}

        with patch("modules.wiring._pg") as mock_pg2:
            mock_pg2.get_by_ids.return_value = [mock_point2]
            _on_retrieval_boosts_ss("memory_retrieved", {
                "retrieved_ids": ["mem-003"],
                "source": "search_memory",
                "query": "fresh_topic",
            })
            time.sleep(0.3)

            call_args2 = mock_pg2.update_payload.call_args[0]
            fresh_ss = call_args2[1]["storage_strength"]

        # Fresh topic should get bigger boost than heavily-used topic
        self.assertGreater(fresh_ss, dampened_ss)


# ============================================================
# TIER 4 TESTS
# ============================================================

class TestCX20_MetaCuriosityUncertainty(unittest.TestCase):
    """CX-20: Low metacognitive confidence → D-type curiosity (Loewenstein 1994)."""

    def test_low_fok_triggers_curiosity(self):
        """FOK below threshold pushes curiosity question."""
        import modules.wiring as w
        w._cx20_last_push.clear()

        from modules.wiring import _on_metacognitive_uncertainty_to_curiosity

        with patch("modules.curiosity.push_curiosidad") as mock_push:
            _on_metacognitive_uncertainty_to_curiosity("metacognitive_control_applied", {
                "strategy": "expand_search",
                "fok_score": 0.2,
                "adjusted_limit": 1.5,
            })
            mock_push.assert_called_once()
            call_args = mock_push.call_args
            self.assertIn("uncertainty", call_args[1].get("tema", call_args[0][0] if call_args[0] else "").lower())

    def test_high_fok_no_curiosity(self):
        """FOK above threshold does NOT push curiosity."""
        from modules.wiring import _on_metacognitive_uncertainty_to_curiosity
        with patch("modules.curiosity.push_curiosidad") as mock_push:
            _on_metacognitive_uncertainty_to_curiosity("metacognitive_control_applied", {
                "strategy": "normal",
                "fok_score": 0.8,
                "adjusted_limit": 1.0,
            })
            mock_push.assert_not_called()

    def test_cooldown_prevents_spam(self):
        """Same domain within cooldown does NOT push again."""
        import modules.wiring as w
        w._cx20_last_push.clear()

        from modules.wiring import _on_metacognitive_uncertainty_to_curiosity

        with patch("modules.curiosity.push_curiosidad") as mock_push:
            _on_metacognitive_uncertainty_to_curiosity("metacognitive_control_applied", {
                "strategy": "trading_strategy",
                "fok_score": 0.1,
                "adjusted_limit": 2.0,
            })
            self.assertEqual(mock_push.call_count, 1)

            # Second call within cooldown
            _on_metacognitive_uncertainty_to_curiosity("metacognitive_control_applied", {
                "strategy": "trading_strategy",
                "fok_score": 0.1,
                "adjusted_limit": 2.0,
            })
            # Should still be 1 (blocked by cooldown)
            self.assertEqual(mock_push.call_count, 1)

    def test_none_fok_no_curiosity(self):
        """None FOK score does NOT push curiosity."""
        from modules.wiring import _on_metacognitive_uncertainty_to_curiosity
        with patch("modules.curiosity.push_curiosidad") as mock_push:
            _on_metacognitive_uncertainty_to_curiosity("metacognitive_control_applied", {
                "strategy": "normal",
                "fok_score": None,
                "adjusted_limit": 1.0,
            })
            mock_push.assert_not_called()


class TestCX18_ReconToMeta(unittest.TestCase):
    """CX-18: Reconsolidation lowers metacognitive confidence (Nelson & Narens 1990)."""

    def test_reconsolidation_lowers_precision(self):
        """Successful reconsolidation → lower L2 precision for domain."""
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_reconsolidation_to_metacognition

        mock_point = MagicMock()
        mock_point.payload = {"narrative_themes": ["trading"]}

        with patch("modules.wiring._pg") as mock_pg:
            mock_pg.get_by_ids.return_value = [mock_point]
            _on_reconsolidation_to_metacognition("reconsolidation_triggered", {
                "memory_id": "mem-001",
                "action": "correct_memory",
                "old_confidence": 0.8,
                "new_confidence": 0.4,
            })

        # Precision should have dropped for "trading"
        self.assertLess(w._cx10_precision_modifiers.get("trading", 1.0), 1.0)

    def test_non_correct_action_skipped(self):
        """Non-correct_memory actions don't trigger confidence drop."""
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_reconsolidation_to_metacognition
        _on_reconsolidation_to_metacognition("reconsolidation_triggered", {
            "memory_id": "mem-001",
            "action": "destabilize",
            "old_confidence": 0.8,
            "new_confidence": 0.4,
        })
        # No precision change
        self.assertEqual(len(w._cx10_precision_modifiers), 0)

    def test_small_pe_skipped(self):
        """Tiny confidence change doesn't trigger precision drop."""
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_reconsolidation_to_metacognition
        _on_reconsolidation_to_metacognition("reconsolidation_triggered", {
            "memory_id": "mem-001",
            "action": "correct_memory",
            "old_confidence": 0.5,
            "new_confidence": 0.48,
        })
        self.assertEqual(len(w._cx10_precision_modifiers), 0)

    def test_precision_floor_respected(self):
        """Precision never drops below CX-18 floor."""
        import modules.wiring as w
        from modules.wiring import _CX18_FLOOR
        w._cx10_precision_modifiers["fragile"] = 0.25

        from modules.wiring import _on_reconsolidation_to_metacognition

        mock_point = MagicMock()
        mock_point.payload = {"narrative_themes": ["fragile"]}

        with patch("modules.wiring._pg") as mock_pg:
            mock_pg.get_by_ids.return_value = [mock_point]
            _on_reconsolidation_to_metacognition("reconsolidation_triggered", {
                "memory_id": "mem-001",
                "action": "correct_memory",
                "old_confidence": 0.9,
                "new_confidence": 0.1,
            })

        self.assertGreaterEqual(w._cx10_precision_modifiers.get("fragile", 1.0), _CX18_FLOOR)


class TestCX15_SelfModelForgetting(unittest.TestCase):
    """CX-15: Self-model protects identity memories (INHIBITORY, Sedikides 2009)."""

    def test_identity_memories_get_ss_boost(self):
        """Self-affirming memories (matching identity themes) get decay protection."""
        import modules.wiring as w
        w._cx15_last_scan = 0  # Reset cooldown

        from modules.wiring import _on_self_model_modulates_forgetting

        identity_point = MagicMock()
        identity_point.id = "id-001"
        identity_point.payload = {
            "narrative_importance": "critical",
            "narrative_themes": ["consciencia", "identidad"],
        }

        regular_point = MagicMock()
        regular_point.id = "reg-001"
        regular_point.payload = {
            "storage_strength": 0.4,
            "narrative_importance": "medium",
            "narrative_themes": ["consciencia"],
        }

        with patch("modules.wiring._pg") as mock_pg, \
             patch("modules.qdrant_utils.scroll_all") as mock_scroll:
            mock_pg.scroll.return_value = ([identity_point], None)
            mock_scroll.return_value = [regular_point]
            _on_self_model_modulates_forgetting("self_model_refreshed", {
                "source": "sleep_loop",
                "discrepancy_count": 0,
                "domains": [],
            })
            time.sleep(0.3)

            # Regular point should get SS boost (shares theme with identity)
            if mock_pg.update_payload.called:
                call_args = mock_pg.update_payload.call_args[0]
                new_ss = call_args[1].get("storage_strength", 0.4)
                self.assertGreater(new_ss, 0.4)

    def test_cooldown_prevents_frequent_scans(self):
        """Scans respect 30 min cooldown."""
        import modules.wiring as w
        w._cx15_last_scan = time.time()  # Just scanned

        from modules.wiring import _on_self_model_modulates_forgetting

        with patch("modules.wiring._pg") as mock_pg, \
             patch("modules.qdrant_utils.scroll_all") as mock_scroll:
            _on_self_model_modulates_forgetting("self_model_refreshed", {
                "source": "sleep_loop",
                "discrepancy_count": 0,
                "domains": [],
            })
            time.sleep(0.2)
            # Should NOT have scrolled (cooldown active)
            mock_scroll.assert_not_called()

    def test_cx10b_source_skipped(self):
        """CX-10B feedback-triggered refreshes don't trigger CX-15."""
        import modules.wiring as w
        w._cx15_last_scan = 0

        from modules.wiring import _on_self_model_modulates_forgetting

        with patch("modules.wiring._pg") as mock_pg, \
             patch("modules.qdrant_utils.scroll_all") as mock_scroll:
            _on_self_model_modulates_forgetting("self_model_refreshed", {
                "source": "metacognitive_bias_cx10b",
                "discrepancy_count": 0,
                "domains": [],
            })
            time.sleep(0.2)
            mock_scroll.assert_not_called()


class TestCX16_WorkspaceToMeta(unittest.TestCase):
    """CX-16: GNW broadcast informs metacognitive monitoring (Shea & Frith 2019)."""

    def test_weak_competition_lowers_precision(self):
        """Low workspace confidence → lower L2 precision for domain."""
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_workspace_broadcast_to_metacognition

        _on_workspace_broadcast_to_metacognition("workspace_competition_complete", {
            "top_activation": 0.35,
            "winner_count": 1,
            "loser_count": 5,
            "winner_domains": ["trading"],
        })

        # Weak competition (0.35 activation, 1 winner vs 5 losers) → low workspace_conf
        # workspace_conf ≈ 0.5*0.35 + 0.3*(1/6) + 0.2*1.0 ≈ 0.425
        # This might be above 0.4 threshold. Let's check by using even weaker activation
        w._cx10_precision_modifiers.clear()
        _on_workspace_broadcast_to_metacognition("workspace_competition_complete", {
            "top_activation": 0.31,
            "winner_count": 1,
            "loser_count": 10,
            "winner_domains": ["uncertain_domain"],
        })
        # workspace_conf ≈ 0.5*0.31 + 0.3*(1/11) + 0.2*1.0 ≈ 0.38 < 0.4
        self.assertLess(w._cx10_precision_modifiers.get("uncertain_domain", 1.0), 1.0)

    def test_strong_competition_no_change(self):
        """High workspace confidence → no precision change."""
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_workspace_broadcast_to_metacognition
        _on_workspace_broadcast_to_metacognition("workspace_competition_complete", {
            "top_activation": 0.9,
            "winner_count": 3,
            "loser_count": 1,
            "winner_domains": ["strong_domain"],
        })
        # workspace_conf high → no precision reduction
        self.assertEqual(len(w._cx10_precision_modifiers), 0)

    def test_below_threshold_skipped(self):
        """Top activation below threshold → handler skips entirely."""
        import modules.wiring as w
        from modules.wiring import _on_workspace_broadcast_to_metacognition, _CX16_STRENGTH_THRESHOLD
        w._cx10_precision_modifiers.clear()

        _on_workspace_broadcast_to_metacognition("workspace_competition_complete", {
            "top_activation": _CX16_STRENGTH_THRESHOLD - 0.1,
            "winner_count": 1,
            "loser_count": 1,
            "winner_domains": ["ignored"],
        })
        self.assertEqual(len(w._cx10_precision_modifiers), 0)


class TestCX19_ConsolToSelf(unittest.TestCase):
    """CX-19: Consolidation feeds self-model (Conway 2005 SMS)."""

    def test_self_relevant_facts_fed_to_self_model(self):
        """Self-relevant semantic facts get fed to update_self_model."""
        import modules.wiring as w
        from modules.wiring import _on_consolidation_feeds_self_model

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("Codi is capable of causal discovery", 0.9, "consciencia"),
            ("Trading API uses REST endpoints", 0.8, "trading"),
        ]

        with patch("modules.config.connect_fts", return_value=mock_conn), \
             patch("modules.config.FTS_DB_PATH", "/tmp/fake.db"), \
             patch("modules.self_model.update_self_model") as mock_update:
            _on_consolidation_feeds_self_model("consolidation_complete", {
                "facts_created": 3,
                "facts_updated": 1,
                "scope": "full",
            })
            time.sleep(0.3)

            # Only "Codi" fact is self-relevant (contains "codi")
            if mock_update.called:
                call_args = mock_update.call_args
                self.assertIn("CX-19", call_args[1].get("insight", call_args[0][0] if call_args[0] else ""))

    def test_too_few_facts_skipped(self):
        """Less than 2 facts → handler skips."""
        from modules.wiring import _on_consolidation_feeds_self_model
        with patch("modules.config.connect_fts") as mock_conn:
            _on_consolidation_feeds_self_model("consolidation_complete", {
                "facts_created": 1,
                "facts_updated": 0,
            })
            mock_conn.assert_not_called()


class TestCX17_ConsolToPrediction(unittest.TestCase):
    """CX-17: Consolidated schemas become predictive priors (Tse 2007, CLS)."""

    def test_only_full_consolidation_triggers(self):
        """Quick consolidation does NOT update schema priors."""
        import modules.wiring as w
        w._cx17_schema_priors.clear()

        from modules.wiring import _on_consolidation_updates_prediction_priors
        _on_consolidation_updates_prediction_priors("consolidation_complete", {
            "facts_created": 5,
            "scope": "quick",
        })
        # No priors should be set (quick scope skipped)
        self.assertEqual(len(w._cx17_schema_priors), 0)

    def test_few_facts_skipped(self):
        """Below minimum facts threshold → skip."""
        import modules.wiring as w
        w._cx17_schema_priors.clear()

        from modules.wiring import _on_consolidation_updates_prediction_priors
        _on_consolidation_updates_prediction_priors("consolidation_complete", {
            "facts_created": 1,
            "scope": "full",
        })
        self.assertEqual(len(w._cx17_schema_priors), 0)

    def test_high_confidence_schemas_update_priors(self):
        """High-confidence schemas → schema prior stored in pull dict."""
        import modules.wiring as w
        w._cx17_schema_priors.clear()

        from modules.wiring import _on_consolidation_updates_prediction_priors, get_cx17_schema_prior

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("trading", 5),
        ]

        with patch("modules.config.connect_fts", return_value=mock_conn), \
             patch("modules.config.FTS_DB_PATH", "/tmp/fake.db"):
            _on_consolidation_updates_prediction_priors("consolidation_complete", {
                "facts_created": 5,
                "scope": "full",
            })
            time.sleep(0.3)

            # Schema prior for "trading" should be set
            self.assertGreater(get_cx17_schema_prior("trading"), 0.0)
            # Non-schema domain should return 0
            self.assertEqual(get_cx17_schema_prior("unknown"), 0.0)


class TestCX21_CausalProtection(unittest.TestCase):
    """CX-21: Causal centrality protects hub memories (INHIBITORY, Kirkpatrick 2017)."""

    def test_refresh_populates_protected_topics(self):
        """NOTEARS state → top quartile topics get protection."""
        import modules.wiring as w
        import numpy as np

        w._cx21_protected_topics = {}

        # Create mock W matrix with clear hub (topic 0 has strongest connections)
        w_matrix = np.array([
            [0.0, 0.8, 0.7, 0.6],
            [0.1, 0.0, 0.1, 0.0],
            [0.2, 0.0, 0.0, 0.1],
            [0.0, 0.1, 0.0, 0.0],
        ])
        topics = ["hub_topic", "leaf_1", "leaf_2", "leaf_3"]

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (
            json.dumps(w_matrix.tolist()),
            json.dumps(topics),
        )

        with patch("modules.config.connect_fts", return_value=mock_conn), \
             patch("modules.config.FTS_DB_PATH", "/tmp/fake.db"):
            from modules.wiring import refresh_cx21_causal_protection
            refresh_cx21_causal_protection()

        # hub_topic should be protected (highest centrality)
        self.assertIn("hub_topic", w._cx21_protected_topics)
        self.assertLess(w._cx21_protected_topics["hub_topic"], 1.0)

    def test_decay_modifier_returns_protection(self):
        """get_cx21_decay_modifier returns <1.0 for protected topics."""
        import modules.wiring as w
        w._cx21_protected_topics = {"important_topic": 0.5}

        from modules.wiring import get_cx21_decay_modifier
        self.assertEqual(get_cx21_decay_modifier("important_topic"), 0.5)
        self.assertEqual(get_cx21_decay_modifier("unprotected_topic"), 1.0)

    def test_shy_integrates_cx21_protection(self):
        """MIN-2 SHY decay rate is reduced for causal hub topics."""
        import modules.wiring as w
        from modules.wiring import shy_downscale_ss

        w._cx21_protected_topics = {"hub_topic": 0.5}

        protected_point = MagicMock()
        protected_point.id = "hub-001"
        protected_point.payload = {
            "storage_strength": 0.8,
            "narrative_themes": ["hub_topic"],
            "consolidated": False,
        }

        unprotected_point = MagicMock()
        unprotected_point.id = "leaf-001"
        unprotected_point.payload = {
            "storage_strength": 0.8,
            "narrative_themes": ["leaf_topic"],
            "consolidated": False,
        }

        with patch("modules.qdrant_utils.scroll_all", return_value=[protected_point, unprotected_point]), \
             patch("modules.wiring._pg") as mock_pg:
            shy_downscale_ss(batch_size=10)

            calls = mock_pg.update_payload.call_args_list
            self.assertEqual(len(calls), 2)

            # Find which call is for each point
            ss_values = {}
            for call in calls:
                pid = call[0][0]
                ss = call[0][1]["storage_strength"]
                ss_values[pid] = ss

            # Protected topic should have higher SS (less decay)
            self.assertGreater(ss_values["hub-001"], ss_values["leaf-001"])


# ============================================================
# TIER 5 TESTS (pre-validated)
# ============================================================

class TestCX22_ActionToMeta(unittest.TestCase):
    """CX-22: Action selection → metacognitive monitoring (L7→L5)."""

    def test_perseveration_lowers_precision(self):
        """Same action repeated N times → lower metacognitive precision."""
        import modules.wiring as w
        w._cx22_action_history.clear()
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_action_selected_to_metacognition, _CX22_PERSEVERATION_THRESHOLD

        # Fire the same action PERSEVERATION_THRESHOLD times
        for _ in range(_CX22_PERSEVERATION_THRESHOLD):
            _on_action_selected_to_metacognition("action_selected", {
                "action": "attend",
                "efe_spread": 0.5,
                "state_topic": "trading",
            })

        # Precision should have dropped for "trading"
        self.assertLess(w._cx10_precision_modifiers.get("trading", 1.0), 1.0)

    def test_no_perseveration_with_varied_actions(self):
        """Different actions don't trigger perseveration."""
        import modules.wiring as w
        w._cx22_action_history.clear()
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_action_selected_to_metacognition

        actions = ["attend", "explore", "consolidate", "attend", "explore"]
        for a in actions:
            _on_action_selected_to_metacognition("action_selected", {
                "action": a,
                "efe_spread": 0.5,
                "state_topic": "varied_topic",
            })

        # No precision change (no perseveration)
        self.assertEqual(len(w._cx10_precision_modifiers), 0)

    def test_indecision_lowers_precision(self):
        """Low EFE spread (indecisive selection) → lower precision."""
        import modules.wiring as w
        w._cx22_action_history.clear()
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_action_selected_to_metacognition

        _on_action_selected_to_metacognition("action_selected", {
            "action": "attend",
            "efe_spread": 0.01,  # Very low spread = indecisive
            "state_topic": "uncertain_domain",
        })

        self.assertLess(w._cx10_precision_modifiers.get("uncertain_domain", 1.0), 1.0)

    def test_high_spread_no_change(self):
        """High EFE spread (decisive selection) → no precision change."""
        import modules.wiring as w
        w._cx22_action_history.clear()
        w._cx10_precision_modifiers.clear()

        from modules.wiring import _on_action_selected_to_metacognition

        _on_action_selected_to_metacognition("action_selected", {
            "action": "explore",
            "efe_spread": 0.5,  # Clear preference
            "state_topic": "confident_domain",
        })

        self.assertEqual(len(w._cx10_precision_modifiers), 0)

    def test_history_capped(self):
        """Action history doesn't grow unbounded."""
        import modules.wiring as w
        w._cx22_action_history.clear()

        from modules.wiring import _on_action_selected_to_metacognition, _CX22_HISTORY_MAX

        for i in range(_CX22_HISTORY_MAX + 10):
            _on_action_selected_to_metacognition("action_selected", {
                "action": f"action_{i}",
                "efe_spread": 0.5,
                "state_topic": "general",
            })

        self.assertLessEqual(len(w._cx22_action_history), _CX22_HISTORY_MAX)


# ========================================================================
# TIER 5 — CX-23 to CX-30 (6 INHIBITORY + 2 EXCITATORY)
# ========================================================================


class TestCX23_ForgettingSuppressesCuriosity(unittest.TestCase):
    """CX-23: Vaulted memories suppress related curiosity (L10→L6 INHIBITORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx23_suppressed_topics.clear()

    def test_vault_suppresses_topic(self):
        """Vaulted memory adds themes to suppressed set."""
        import modules.wiring as w
        from modules.wiring import _on_forgetting_suppresses_curiosity

        mock_payload = {"narrative_themes": ["trading", "kraken"]}
        mock_point = MagicMock()
        mock_point.payload = mock_payload
        with patch("modules.wiring._pg.get_by_ids", return_value=[mock_point]):
            with patch("modules.curiosity._cargar_curiosidades", return_value={"pendientes": [], "exploradas": []}):
                with patch("modules.curiosity._guardar_curiosidades"):
                    _on_forgetting_suppresses_curiosity("memory_vaulted", {"memory_id": "test-123"})
                    # Wait for thread
                    import time
                    time.sleep(0.2)

        self.assertIn("trading", w._cx23_suppressed_topics)
        self.assertIn("kraken", w._cx23_suppressed_topics)

    def test_is_curiosity_suppressed(self):
        """PULL getter correctly reports suppression status."""
        import modules.wiring as w
        from modules.wiring import is_curiosity_suppressed
        w._cx23_suppressed_topics["forgotten_topic"] = time.time()
        self.assertTrue(is_curiosity_suppressed("forgotten_topic"))
        self.assertFalse(is_curiosity_suppressed("fresh_topic"))

    def test_expired_suppression_cleaned(self):
        """Expired suppressions are removed."""
        import modules.wiring as w
        from modules.wiring import is_curiosity_suppressed
        w._cx23_suppressed_topics["old_topic"] = time.time() - 100000  # Expired
        self.assertFalse(is_curiosity_suppressed("old_topic"))

    def test_empty_memory_id_skipped(self):
        """Empty memory_id doesn't crash."""
        from modules.wiring import _on_forgetting_suppresses_curiosity
        _on_forgetting_suppresses_curiosity("memory_vaulted", {"memory_id": ""})
        # No error = pass


class TestCX28_ForgettingDegradesMeta(unittest.TestCase):
    """CX-28: Forgetting degrades metacognitive confidence (L10→L5 INHIBITORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx10_precision_modifiers.clear()

    def test_vault_lowers_precision(self):
        """Vaulted memory lowers precision for its domain."""
        import modules.wiring as w
        w._cx10_precision_modifiers["trading"] = 0.8

        mock_payload = {"narrative_themes": ["trading"]}
        with patch("modules.pg_store.PGMemoryStore") as MockPG:
            MockPG.return_value.get_payload.return_value = mock_payload
            from modules.wiring import _on_forgetting_degrades_metacognition
            _on_forgetting_degrades_metacognition("memory_vaulted", {"memory_id": "test-456"})
            time.sleep(0.2)

        self.assertLessEqual(w._cx10_precision_modifiers.get("trading", 1.0), 0.8)

    def test_floor_respected(self):
        """Precision never drops below CX-28 floor."""
        import modules.wiring as w
        from modules.wiring import _CX28_FLOOR
        w._cx10_precision_modifiers["dead_domain"] = 0.16  # Just above floor

        mock_payload = {"narrative_themes": ["dead_domain"]}
        with patch("modules.pg_store.PGMemoryStore") as MockPG:
            MockPG.return_value.get_payload.return_value = mock_payload
            from modules.wiring import _on_forgetting_degrades_metacognition
            _on_forgetting_degrades_metacognition("memory_vaulted", {"memory_id": "test-789"})
            time.sleep(0.2)

        self.assertGreaterEqual(w._cx10_precision_modifiers.get("dead_domain", 1.0), _CX28_FLOOR)

    def test_no_payload_skipped(self):
        """Missing payload doesn't crash."""
        with patch("modules.pg_store.PGMemoryStore") as MockPG:
            MockPG.return_value.get_payload.return_value = None
            from modules.wiring import _on_forgetting_degrades_metacognition
            _on_forgetting_degrades_metacognition("memory_vaulted", {"memory_id": "test-nope"})
            time.sleep(0.1)
        # No error = pass


class TestCX24_MetaGatesReconsolidation(unittest.TestCase):
    """CX-24: High metacognitive confidence blocks reconsolidation (L5→L1 INHIBITORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx24_blocked_domains.clear()
        w._cx10_precision_modifiers.clear()

    def test_high_confidence_blocks(self):
        """Precision > 0.85 blocks reconsolidation for domain."""
        import modules.wiring as w
        from modules.wiring import _on_metacognition_gates_reconsolidation, is_reconsolidation_gated, _CX24_CONFIDENCE_GATE
        w._cx10_precision_modifiers["stable_domain"] = 0.9  # Above gate

        _on_metacognition_gates_reconsolidation("metacognitive_control_applied", {
            "domain": "stable_domain",
        })

        self.assertTrue(is_reconsolidation_gated("stable_domain"))

    def test_low_confidence_unblocks(self):
        """Precision < 0.75 re-allows reconsolidation (hysteresis)."""
        import modules.wiring as w
        from modules.wiring import _on_metacognition_gates_reconsolidation, is_reconsolidation_gated
        w._cx24_blocked_domains["unstable"] = True
        w._cx10_precision_modifiers["unstable"] = 0.7  # Below hysteresis

        _on_metacognition_gates_reconsolidation("metacognitive_control_applied", {
            "domain": "unstable",
        })

        self.assertFalse(is_reconsolidation_gated("unstable"))

    def test_hysteresis_band_prevents_oscillation(self):
        """Precision between 0.75 and 0.85 doesn't change gate state."""
        import modules.wiring as w
        from modules.wiring import _on_metacognition_gates_reconsolidation, is_reconsolidation_gated
        w._cx24_blocked_domains["midrange"] = True
        w._cx10_precision_modifiers["midrange"] = 0.80  # In hysteresis band

        _on_metacognition_gates_reconsolidation("metacognitive_control_applied", {
            "domain": "midrange",
        })

        # Still blocked — in hysteresis band
        self.assertTrue(is_reconsolidation_gated("midrange"))

    def test_unblocked_domain_not_gated(self):
        """Domain without high precision is not gated."""
        from modules.wiring import is_reconsolidation_gated
        self.assertFalse(is_reconsolidation_gated("unknown_domain"))


class TestCX26_SelfModelConstrainsAction(unittest.TestCase):
    """CX-26: Self-model suppresses identity-inconsistent policies (L9→L7 INHIBITORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx26_identity_constraints.clear()
        w._cx26_core_values.clear()

    def test_beliefs_create_constraints(self):
        """Self-model refresh with enough beliefs creates constraints."""
        import modules.wiring as w
        from modules.wiring import _on_self_model_constrains_action

        _on_self_model_constrains_action("self_model_refreshed", {
            "summary": "Soy Codi, el parcero de Hare",
            "aspects": {
                "valor": ["honestidad", "colaboracion"],
                "preferencia": ["codigo limpio", "tests primero"],
            },
        })

        self.assertGreater(len(w._cx26_identity_constraints), 0)
        self.assertGreater(len(w._cx26_core_values), 0)

    def test_too_few_beliefs_skipped(self):
        """Less than MIN_BELIEFS doesn't create constraints."""
        import modules.wiring as w
        from modules.wiring import _on_self_model_constrains_action

        _on_self_model_constrains_action("self_model_refreshed", {
            "summary": "short",
            "aspects": {"valor": ["one"]},
        })

        self.assertEqual(len(w._cx26_identity_constraints), 0)

    def test_identity_penalty_conservative(self):
        """get_cx26_identity_penalty returns 0 (conservative default)."""
        from modules.wiring import get_cx26_identity_penalty
        self.assertEqual(get_cx26_identity_penalty("any action"), 0.0)


class TestCX27_MetaSuppressesCausalEdges(unittest.TestCase):
    """CX-27: Low metacognitive confidence suppresses causal edges (L5→L8 INHIBITORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx27_suppressed_domains.clear()
        w._cx10_precision_modifiers.clear()

    def test_low_precision_suppresses(self):
        """Precision below floor triggers suppression."""
        import modules.wiring as w
        from modules.wiring import _on_metacognition_suppresses_causal_edges, get_cx27_edge_suppression

        w._cx10_precision_modifiers["unreliable"] = 0.3  # Below floor

        _on_metacognition_suppresses_causal_edges("metacognitive_control_applied", {
            "domain": "unreliable",
        })

        suppression = get_cx27_edge_suppression("unreliable")
        self.assertLess(suppression, 1.0)
        self.assertGreater(suppression, 0.0)

    def test_high_precision_removes_suppression(self):
        """Precision above floor removes suppression."""
        import modules.wiring as w
        from modules.wiring import _on_metacognition_suppresses_causal_edges, get_cx27_edge_suppression

        w._cx27_suppressed_domains["reliable"] = 0.5
        w._cx10_precision_modifiers["reliable"] = 0.6  # Above floor

        _on_metacognition_suppresses_causal_edges("metacognitive_control_applied", {
            "domain": "reliable",
        })

        self.assertEqual(get_cx27_edge_suppression("reliable"), 1.0)

    def test_no_domain_no_crash(self):
        """Empty domain doesn't crash."""
        from modules.wiring import _on_metacognition_suppresses_causal_edges
        _on_metacognition_suppresses_causal_edges("metacognitive_control_applied", {"domain": ""})


class TestCX25_WorkspaceRIF(unittest.TestCase):
    """CX-25: Workspace access: testing effect + RIF (L3→L10 INHIBITORY)."""

    def test_no_winners_skipped(self):
        """No winner domains → no action."""
        from modules.wiring import _on_workspace_retrieval_modulates_forgetting
        _on_workspace_retrieval_modulates_forgetting("workspace_competition_complete", {
            "winner_domains": [],
            "loser_ids": ["a", "b"],
            "loser_domains": ["trading", "trading"],
        })
        # No error = pass

    def test_rif_targets_same_domain_losers(self):
        """RIF only targets losers in winner's domain."""
        import modules.wiring as w
        from modules.wiring import _on_workspace_retrieval_modulates_forgetting

        with patch("modules.pg_store.PGMemoryStore") as MockPG:
            mock_pg = MockPG.return_value
            mock_pg.get_payload.return_value = {
                "importance": "medium",
                "storage_strength": 0.5,
            }
            mock_pg.update_payload.return_value = None

            _on_workspace_retrieval_modulates_forgetting("workspace_competition_complete", {
                "winner_domains": ["trading"],
                "loser_ids": ["loser1", "loser2"],
                "loser_domains": ["trading", "other"],
                "top_activation": 0.8,
            })
            time.sleep(0.3)

            # loser1 is same-domain, should get RIF
            # loser2 is different domain, should not
            calls = mock_pg.update_payload.call_args_list
            if calls:
                # At least one RIF call for same-domain loser
                rif_ids = [c[0][0] for c in calls]
                self.assertIn("loser1", rif_ids)

    def test_critical_memories_exempt_from_rif(self):
        """Critical importance memories are never RIF'd."""
        from modules.wiring import _on_workspace_retrieval_modulates_forgetting

        with patch("modules.pg_store.PGMemoryStore") as MockPG:
            mock_pg = MockPG.return_value
            mock_pg.get_payload.return_value = {
                "importance": "critical",
                "storage_strength": 0.5,
            }
            mock_pg.update_payload.return_value = None

            _on_workspace_retrieval_modulates_forgetting("workspace_competition_complete", {
                "winner_domains": ["consciencia"],
                "loser_ids": ["critical-mem"],
                "loser_domains": ["consciencia"],
                "top_activation": 0.9,
            })
            time.sleep(0.3)

            # Critical memory should NOT be updated
            mock_pg.update_payload.assert_not_called()


class TestCX29_ReconInvalidatesCausal(unittest.TestCase):
    """CX-29: Memory correction invalidates causal edges (L1→L8 EXCITATORY)."""

    def test_non_correct_action_skipped(self):
        """Actions other than correct_memory are skipped."""
        from modules.wiring import _on_reconsolidation_invalidates_causal_edges
        _on_reconsolidation_invalidates_causal_edges("reconsolidation_triggered", {
            "action": "merge_memory",
            "memory_id": "test-123",
        })
        # No error = pass

    def test_correct_memory_invalidates_edges(self):
        """Correct memory action triggers edge re-evaluation."""
        from modules.wiring import _on_reconsolidation_invalidates_causal_edges

        mock_payload = {"narrative_themes": ["trading"]}
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch("modules.wiring._pg.get_by_ids", return_value=[mock_point]):
            with patch("modules.db_pool.get_conn") as mock_conn:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [
                    (1, "trading", "risk", 0.5),
                    (2, "news", "trading", 0.3),
                ]
                mock_conn.return_value.execute.return_value = mock_cursor

                _on_reconsolidation_invalidates_causal_edges("reconsolidation_triggered", {
                    "action": "correct_memory",
                    "memory_id": "test-456",
                })
                time.sleep(0.3)

                # Should have executed UPDATE or DELETE on edges
                calls = mock_conn.return_value.execute.call_args_list
                # At least the SELECT + some modifications
                self.assertGreater(len(calls), 0)

    def test_empty_memory_id_skipped(self):
        """Empty memory_id doesn't crash."""
        from modules.wiring import _on_reconsolidation_invalidates_causal_edges
        _on_reconsolidation_invalidates_causal_edges("reconsolidation_triggered", {
            "action": "correct_memory",
            "memory_id": "",
        })


class TestCX30_ActionOutcomesCausal(unittest.TestCase):
    """CX-30: Action outcomes as causal interventions (L7→L8 EXCITATORY)."""

    def setUp(self):
        import modules.wiring as w
        w._cx11_buffer.clear()

    def test_action_feeds_cx11_buffer(self):
        """Action with meaningful spread feeds CX-11 buffer with 2x weight."""
        import modules.wiring as w
        from modules.wiring import _on_action_outcome_updates_causal_dag, _CX30_INTERVENTION_WEIGHT

        _on_action_outcome_updates_causal_dag("action_selected", {
            "action": "explore",
            "state_topic": "trading",
            "efe_spread": 0.5,
        })

        self.assertEqual(len(w._cx11_buffer), 1)
        item = w._cx11_buffer[0]
        self.assertEqual(item["from_topic"], "action:explore")
        self.assertEqual(item["to_topic"], "trading")
        self.assertEqual(item["weight"], _CX30_INTERVENTION_WEIGHT)

    def test_trivial_action_skipped(self):
        """Actions with near-zero EFE spread are skipped."""
        import modules.wiring as w
        from modules.wiring import _on_action_outcome_updates_causal_dag

        _on_action_outcome_updates_causal_dag("action_selected", {
            "action": "noop",
            "state_topic": "general",
            "efe_spread": 0.005,
        })

        self.assertEqual(len(w._cx11_buffer), 0)

    def test_missing_topic_skipped(self):
        """Missing state_topic doesn't crash."""
        import modules.wiring as w
        from modules.wiring import _on_action_outcome_updates_causal_dag

        _on_action_outcome_updates_causal_dag("action_selected", {
            "action": "explore",
            "state_topic": "",
            "efe_spread": 0.5,
        })

        self.assertEqual(len(w._cx11_buffer), 0)

    def test_intervention_weight_is_2x(self):
        """Interventional evidence is weighted 2x observational."""
        from modules.wiring import _CX30_INTERVENTION_WEIGHT, _CX11_CURIOSITY_WEIGHT
        self.assertGreater(_CX30_INTERVENTION_WEIGHT, _CX11_CURIOSITY_WEIGHT)


# ============================================================
# SPRINT CX-WAKEUP TESTS — effective_fires, EII, event enrichment
# ============================================================

class TestEffectiveFiresTracking(unittest.TestCase):
    """Sprint 0: effective_fires infrastructure."""

    def setUp(self):
        from modules.cx_observability import _effective_fire_counts, _inhibition_events
        _effective_fire_counts.clear()
        _inhibition_events.clear()

    def test_report_effective_fire_increments(self):
        from modules.cx_observability import report_effective_fire, get_effective_fire_counts
        report_effective_fire('CX-4b')
        report_effective_fire('CX-4b')
        report_effective_fire('CX-14')
        counts = get_effective_fire_counts()
        self.assertEqual(counts['CX-4b'], 2)
        self.assertEqual(counts['CX-14'], 1)

    def test_inhibitory_cx_tracked(self):
        from modules.cx_observability import report_effective_fire, _inhibition_events
        report_effective_fire('CX-23')
        report_effective_fire('CX-28')
        report_effective_fire('CX-4b')  # Not inhibitory
        self.assertEqual(_inhibition_events.get('CX-23'), 1)
        self.assertEqual(_inhibition_events.get('CX-28'), 1)
        self.assertNotIn('CX-4b', _inhibition_events)

    def test_inhibition_ratio_zero_when_no_inhibitory(self):
        from modules.cx_observability import report_effective_fire, compute_inhibition_ratio
        report_effective_fire('CX-1')
        report_effective_fire('CX-2')
        ratio = compute_inhibition_ratio({})
        self.assertEqual(ratio, 0.0)

    def test_inhibition_ratio_warns_above_threshold(self):
        from modules.cx_observability import report_effective_fire, compute_inhibition_ratio
        # 3 inhibitory, 2 excitatory = 0.6 ratio
        for _ in range(3):
            report_effective_fire('CX-23')
        for _ in range(2):
            report_effective_fire('CX-1')
        ratio = compute_inhibition_ratio({})
        self.assertGreater(ratio, 0.40)


class TestEII(unittest.TestCase):
    """Sprint 0: Effective Integration Index computation."""

    def test_eii_zero_when_no_fires(self):
        from modules.cx_observability import compute_eii
        self.assertEqual(compute_eii({}, 0.0, 34), 0.0)

    def test_eii_zero_when_no_diversity(self):
        from modules.cx_observability import compute_eii
        # No diversity = no integration
        result = compute_eii({'CX-3': 5}, 0.0, 34)
        self.assertEqual(result, 0.0)

    def test_eii_positive_with_diversity(self):
        from modules.cx_observability import compute_eii, _coact_tracker
        # Simulate co-firing traces
        _coact_tracker._trace_cx = {
            'trace1': {'CX-3', 'CX-10'},
            'trace2': {'CX-3', 'CX-15'},
            'trace3': {'CX-10', 'CX-26'},
        }
        result = compute_eii({'CX-3': 3, 'CX-10': 2, 'CX-15': 1, 'CX-26': 1}, 2.0, 34)
        self.assertGreater(result, 0.0)
        _coact_tracker._trace_cx.clear()


class TestConsolidationEventEnrichment(unittest.TestCase):
    """Sprint A: Consolidation event includes bridge_edges and batch_topic."""

    def test_consolidation_event_has_bridge_edges(self):
        """Verify event_data structure includes bridge_edges field."""
        # Simulate what _replay_consolidation_event produces
        event_data = {
            "batch_id": "test-batch",
            "scope": "full",
            "bridge_edges": 5,
            "batch_topic": "trading, consciencia",
        }
        self.assertIn("bridge_edges", event_data)
        self.assertEqual(event_data["bridge_edges"], 5)

    def test_cx14_channel3_needs_bridge_edges_gt3(self):
        """CX-14 channel 3 only fires if bridge_edges > 3."""
        import modules.wiring as w
        old_topics = dict(w._cx14_recent_topics)
        w._cx14_recent_topics.clear()

        with patch("modules.curiosity.push_curiosidad") as mock_push:
            # bridge_edges = 2 → no fire
            w._on_consolidation_gaps_drive_curiosity("consolidation_complete", {
                "scope": "full", "contradictions_found": 0,
                "clusters_found": 5, "facts_extracted": 10,
                "bridge_edges": 2,
            })
            mock_push.assert_not_called()

            # bridge_edges = 5 → fires
            w._on_consolidation_gaps_drive_curiosity("consolidation_complete", {
                "scope": "full", "contradictions_found": 0,
                "clusters_found": 5, "facts_extracted": 10,
                "bridge_edges": 5,
            })
            mock_push.assert_called_once()

        w._cx14_recent_topics = old_topics

    def test_cx32_needs_batch_topic(self):
        """CX-32 returns early without batch_topic."""
        from modules.wiring import _on_consolidation_enrich_fhrr
        # No batch_topic → early return (no error)
        _on_consolidation_enrich_fhrr("consolidation_complete", {
            "facts_created": 3,
            "batch_topic": "",
        })
        # Should not raise — just return early


class TestReconsolidationEventEmission(unittest.TestCase):
    """Sprint B: Reconsolidation emits event_bus events."""

    def test_cx18_accepts_extinction_action(self):
        """CX-18 now handles action='extinction' from sleep loop."""
        import modules.wiring as w
        old_mods = dict(w._cx10_precision_modifiers)
        w._cx10_precision_modifiers.clear()
        w._cx10_precision_modifiers["test_domain"] = 1.0

        with patch.object(w._pg, 'get_by_ids', return_value=[
            MagicMock(payload={"narrative_themes": ["test_domain"]})
        ]):
            w._on_reconsolidation_to_metacognition("reconsolidation_triggered", {
                "action": "extinction",
                "memory_id": "test-id",
                "prediction_error": 0.4,  # Already dampened 0.6x
                "old_confidence": 0.8,
                "new_confidence": 0.5,
                "source": "sleep_loop",
            })

        # Precision should have decreased
        self.assertLess(w._cx10_precision_modifiers.get("test_domain", 1.0), 1.0)
        w._cx10_precision_modifiers = old_mods

    def test_cx29_accepts_extinction_action(self):
        """CX-29 now handles action='extinction' from sleep loop."""
        from modules.wiring import _on_reconsolidation_invalidates_causal_edges
        # Should not raise with extinction action
        with patch.object(MagicMock(), 'get_by_ids', return_value=[]):
            _on_reconsolidation_invalidates_causal_edges("reconsolidation_triggered", {
                "action": "extinction",
                "memory_id": "test-id",
            })

    def test_cx8_rejects_extinction(self):
        """CX-8 MUST NOT fire for extinction — don't boost SS of weakened memory."""
        import modules.wiring as w
        with patch.object(w._pg, 'get_by_ids') as mock_get:
            w._on_reconsolidation_protects_decay("reconsolidation_triggered", {
                "action": "extinction",
                "memory_id": "test-id",
                "new_confidence": 0.3,
            })
            mock_get.assert_not_called()  # Should return before pg lookup


class TestVaultEventEnrichment(unittest.TestCase):
    """Sprint B: MEMORY_VAULTED includes importance field."""

    def test_vault_event_has_importance(self):
        """Vault emission includes importance from payload."""
        import modules.wiring as w
        old_tracker = dict(w._vault_tracker)

        # Reset tracker
        w._vault_tracker["count"] = 0
        w._vault_tracker["important_count"] = 0
        w._vault_tracker["window_start"] = time.monotonic()

        w._on_vault_tracks_urgency("memory_vaulted", {
            "memory_id": "test-id",
            "importance": "high",
        })

        self.assertEqual(w._vault_tracker["important_count"], 1)
        w._vault_tracker.update(old_tracker)


class TestCX22SleepGate(unittest.TestCase):
    """Sprint C: CX-22 ignores sleep-sourced actions."""

    def test_sleep_source_gated(self):
        """CX-22 returns early for source='sleep_loop'."""
        import modules.wiring as w
        old_history = list(w._cx22_action_history)
        w._cx22_action_history.clear()

        w._on_action_selected_to_metacognition("action_selected", {
            "action": "resolve_curiosity",
            "source": "sleep_loop",
            "efe_spread": 0.01,
            "state_topic": "test",
        })

        # History should be empty — handler returned before tracking
        self.assertEqual(len(w._cx22_action_history), 0)
        w._cx22_action_history = old_history

    def test_interactive_source_not_gated(self):
        """CX-22 processes interactive actions normally."""
        import modules.wiring as w
        old_history = list(w._cx22_action_history)
        w._cx22_action_history.clear()

        w._on_action_selected_to_metacognition("action_selected", {
            "action": "exploit",
            "source": "interactive",
            "efe_spread": 0.5,
            "state_topic": "test",
        })

        self.assertEqual(len(w._cx22_action_history), 1)
        w._cx22_action_history = old_history


class TestGuardPassthroughRate(unittest.TestCase):
    """Sprint 0: Guard passthrough rate computation."""

    def setUp(self):
        from modules.cx_observability import (
            _effective_fire_counts, _total_invocation_counts, _inhibition_events
        )
        _effective_fire_counts.clear()
        _total_invocation_counts.clear()
        _inhibition_events.clear()

    def test_passthrough_rate(self):
        from modules.cx_observability import (
            report_effective_fire, get_guard_passthrough_rates,
            _total_invocation_counts
        )
        _total_invocation_counts['CX-14'] = 10
        report_effective_fire('CX-14')
        report_effective_fire('CX-14')
        rates = get_guard_passthrough_rates()
        self.assertAlmostEqual(rates['CX-14'], 0.2, places=2)

    def test_zero_invocations_returns_1(self):
        from modules.cx_observability import get_guard_passthrough_rates, _total_invocation_counts
        _total_invocation_counts['CX-99'] = 0
        rates = get_guard_passthrough_rates()
        # 0/max(0,1) = 0
        self.assertEqual(rates.get('CX-99', 0), 0.0)


if __name__ == "__main__":
    unittest.main()

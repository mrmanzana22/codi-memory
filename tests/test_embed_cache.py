"""Tests for PR-P1a: Embed cache with LRU on _embed_text()."""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbedCache:
    """Test that _embed_text caching works correctly."""

    def _make_mock_oai(self):
        """Create a mock OpenAI client that returns predictable embeddings."""
        mock_oai = MagicMock()
        call_count = [0]

        def fake_create(model, input):
            call_count[0] += 1
            # Return a deterministic embedding based on input text
            fake_vec = [float(ord(c) % 10) / 10.0 for c in (input[:10].ljust(10))]
            resp = MagicMock()
            resp.data = [MagicMock(embedding=fake_vec)]
            return resp

        mock_oai.embeddings.create = fake_create
        return mock_oai, call_count

    def test_cache_hit(self):
        """Same text called twice should only make 1 API call."""
        from modules import consolidation_common

        mock_oai, call_count = self._make_mock_oai()

        # Clear cache before test
        consolidation_common._embed_text_cached.cache_clear()

        with patch.object(consolidation_common, '_get_oai', return_value=mock_oai):
            result1 = consolidation_common._embed_text("hello world")
            result2 = consolidation_common._embed_text("hello world")

        assert call_count[0] == 1, f"Expected 1 API call, got {call_count[0]}"
        assert result1 == result2

    def test_cache_miss(self):
        """Different texts should each make an API call."""
        from modules import consolidation_common

        mock_oai, call_count = self._make_mock_oai()
        consolidation_common._embed_text_cached.cache_clear()

        with patch.object(consolidation_common, '_get_oai', return_value=mock_oai):
            consolidation_common._embed_text("text one")
            consolidation_common._embed_text("text two")

        assert call_count[0] == 2

    def test_returns_list(self):
        """_embed_text should return a list, not a tuple."""
        from modules import consolidation_common

        mock_oai, _ = self._make_mock_oai()
        consolidation_common._embed_text_cached.cache_clear()

        with patch.object(consolidation_common, '_get_oai', return_value=mock_oai):
            result = consolidation_common._embed_text("test")

        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_cache_info_reports_hits(self):
        """get_embed_cache_info should report accurate hit/miss counts."""
        from modules import consolidation_common

        mock_oai, _ = self._make_mock_oai()
        consolidation_common._embed_text_cached.cache_clear()

        with patch.object(consolidation_common, '_get_oai', return_value=mock_oai):
            consolidation_common._embed_text("abc")
            consolidation_common._embed_text("abc")  # hit
            consolidation_common._embed_text("def")  # miss

        info = consolidation_common.get_embed_cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2
        assert info["current_size"] == 2

    def test_cache_eviction(self):
        """Cache should not exceed maxsize."""
        from modules import consolidation_common

        mock_oai, call_count = self._make_mock_oai()
        consolidation_common._embed_text_cached.cache_clear()

        with patch.object(consolidation_common, '_get_oai', return_value=mock_oai):
            # Fill cache beyond maxsize (256)
            for i in range(260):
                consolidation_common._embed_text(f"unique_text_{i}")

        info = consolidation_common.get_embed_cache_info()
        assert info["current_size"] <= 256
        assert call_count[0] == 260  # All were misses

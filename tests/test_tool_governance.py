"""
Tests for Tool Governance (PR-TG1).
Tests the resolver, bundle catalog, and allowlist filter.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from modules.tool_governance import (
    BUNDLE_CORE,
    BUNDLE_OPS,
    BUNDLE_RESEARCH,
    BUNDLES,
    TOOLSET_PRESETS,
    VALID_TOOLSETS,
    TOOLSET_FILE,
    resolve_toolset,
    get_allowed_tools,
    apply_allowlist,
    _get_toolset_status_impl,
)


class FakeTool:
    """Minimal stand-in for a FastMCP Tool object."""
    def __init__(self, name):
        self.name = name


class FakeToolManager:
    """Minimal stand-in for FastMCP ToolManager."""
    def __init__(self, tool_names):
        self._tools = {name: FakeTool(name) for name in tool_names}

    def remove_tool(self, name):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        del self._tools[name]


class FakeMCP:
    """Minimal stand-in for FastMCP server."""
    def __init__(self, tool_names):
        self._tool_manager = FakeToolManager(tool_names)

    def remove_tool(self, name):
        self._tool_manager.remove_tool(name)


# ============================================================
# Catalog Integrity
# ============================================================

class TestCatalogIntegrity(unittest.TestCase):
    """Bundles must not overlap and must contain only strings."""

    def test_no_overlap_core_ops(self):
        overlap = BUNDLE_CORE & BUNDLE_OPS
        self.assertEqual(overlap, frozenset(), f"core/ops overlap: {overlap}")

    def test_no_overlap_core_research(self):
        overlap = BUNDLE_CORE & BUNDLE_RESEARCH
        self.assertEqual(overlap, frozenset(), f"core/research overlap: {overlap}")

    def test_no_overlap_ops_research(self):
        overlap = BUNDLE_OPS & BUNDLE_RESEARCH
        self.assertEqual(overlap, frozenset(), f"ops/research overlap: {overlap}")

    def test_all_entries_are_strings(self):
        for name, bundle in BUNDLES.items():
            for tool in bundle:
                self.assertIsInstance(tool, str, f"{name} has non-str: {tool!r}")

    def test_presets_reference_valid_bundles(self):
        valid_bundle_names = set(BUNDLES.keys()) | {"full"}
        for preset, bundles in TOOLSET_PRESETS.items():
            for b in bundles:
                self.assertIn(b, valid_bundle_names,
                              f"Preset {preset!r} references unknown bundle {b!r}")

    def test_get_toolset_status_in_core(self):
        """The governance meta tool must always be visible."""
        self.assertIn("get_toolset_status", BUNDLE_CORE)


# ============================================================
# Resolver
# ============================================================

class TestResolveToolset(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=True)
    @patch("modules.tool_governance.TOOLSET_FILE", "/nonexistent/.toolset")
    def test_default_core(self):
        """No env var, no .toolset -> core."""
        # Clear CODI_TOOLSET if present
        os.environ.pop("CODI_TOOLSET", None)
        self.assertEqual(resolve_toolset(), "core")

    @patch.dict(os.environ, {"CODI_TOOLSET": "ops"})
    def test_toolset_ops_env(self):
        self.assertEqual(resolve_toolset(), "ops")

    @patch.dict(os.environ, {"CODI_TOOLSET": "research"})
    def test_toolset_research_env(self):
        self.assertEqual(resolve_toolset(), "research")

    @patch.dict(os.environ, {"CODI_TOOLSET": "full"})
    def test_toolset_full_env(self):
        self.assertEqual(resolve_toolset(), "full")

    @patch.dict(os.environ, {"CODI_TOOLSET": "banana"})
    def test_invalid_toolset_falls_back_core(self):
        """Invalid CODI_TOOLSET falls back to core with warning."""
        self.assertEqual(resolve_toolset(), "core")

    @patch.dict(os.environ, {}, clear=True)
    def test_toolset_research_file(self, tmp_path=None):
        """Dot-file with 'research' -> research."""
        os.environ.pop("CODI_TOOLSET", None)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toolset", delete=False) as f:
            f.write("research\n")
            tmp = f.name
        try:
            with patch("modules.tool_governance.TOOLSET_FILE", tmp):
                self.assertEqual(resolve_toolset(), "research")
        finally:
            os.unlink(tmp)

    def test_precedence_env_over_file(self):
        """Env var takes precedence over dot-file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toolset", delete=False) as f:
            f.write("core\n")
            tmp = f.name
        try:
            with patch("modules.tool_governance.TOOLSET_FILE", tmp):
                with patch.dict(os.environ, {"CODI_TOOLSET": "full"}):
                    self.assertEqual(resolve_toolset(), "full")
        finally:
            os.unlink(tmp)


# ============================================================
# get_allowed_tools
# ============================================================

class TestGetAllowedTools(unittest.TestCase):

    def test_core_returns_core_only(self):
        allowed = get_allowed_tools("core")
        self.assertEqual(allowed, BUNDLE_CORE)

    def test_ops_returns_core_plus_ops(self):
        allowed = get_allowed_tools("ops")
        self.assertEqual(allowed, BUNDLE_CORE | BUNDLE_OPS)

    def test_research_returns_core_plus_research(self):
        allowed = get_allowed_tools("research")
        self.assertEqual(allowed, BUNDLE_CORE | BUNDLE_RESEARCH)

    def test_full_returns_none(self):
        """Full mode returns None, meaning no filtering."""
        allowed = get_allowed_tools("full")
        self.assertIsNone(allowed)


# ============================================================
# apply_allowlist
# ============================================================

class TestApplyAllowlist(unittest.TestCase):

    def test_core_hides_non_core_tools(self):
        """Only core tools survive when allowlist is BUNDLE_CORE."""
        all_tools = list(BUNDLE_CORE) + ["sync_fts_index", "assessment_report", "spread_activation"]
        fake_mcp = FakeMCP(all_tools)

        removed_count, removed_names = apply_allowlist(fake_mcp, BUNDLE_CORE)

        self.assertEqual(removed_count, 3)
        remaining = set(fake_mcp._tool_manager._tools.keys())
        self.assertEqual(remaining, set(BUNDLE_CORE))

    def test_full_removes_nothing(self):
        """Full mode (allowed=None) removes zero tools."""
        all_tools = list(BUNDLE_CORE) + list(BUNDLE_OPS)
        fake_mcp = FakeMCP(all_tools)

        removed_count, _ = apply_allowlist(fake_mcp, None)

        self.assertEqual(removed_count, 0)
        self.assertEqual(len(fake_mcp._tool_manager._tools), len(all_tools))

    def test_ops_keeps_core_and_ops(self):
        """Ops preset keeps both core and ops tools."""
        all_tools = list(BUNDLE_CORE) + list(BUNDLE_OPS) + ["spread_activation"]
        fake_mcp = FakeMCP(all_tools)

        allowed = BUNDLE_CORE | BUNDLE_OPS
        removed_count, removed_names = apply_allowlist(fake_mcp, allowed)

        self.assertEqual(removed_count, 1)
        self.assertIn("spread_activation", removed_names)
        remaining = set(fake_mcp._tool_manager._tools.keys())
        self.assertTrue(BUNDLE_CORE.issubset(remaining))
        self.assertTrue(BUNDLE_OPS.issubset(remaining))


# ============================================================
# Status tool
# ============================================================

class TestGetToolsetStatus(unittest.TestCase):

    @patch.dict(os.environ, {"CODI_TOOLSET": "ops"})
    def test_status_contains_toolset(self):
        fake_mcp = FakeMCP(list(BUNDLE_CORE) + list(BUNDLE_OPS))
        output = _get_toolset_status_impl(fake_mcp)
        self.assertIn("ops", output)
        self.assertIn("Visible tools:", output)

    @patch.dict(os.environ, {}, clear=True)
    @patch("modules.tool_governance.TOOLSET_FILE", "/nonexistent/.toolset")
    def test_status_shows_default_source(self):
        os.environ.pop("CODI_TOOLSET", None)
        fake_mcp = FakeMCP(list(BUNDLE_CORE))
        output = _get_toolset_status_impl(fake_mcp)
        self.assertIn("default", output)


if __name__ == "__main__":
    unittest.main()

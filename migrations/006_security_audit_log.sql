-- 006_security_audit_log.sql
-- Immutable, INSERT-only audit log for destructive operations.
--
-- Records every execution of destructive tools (delete_memory,
-- delete_by_content, clear_all_memories) after guard confirmation.
--
-- Design: No UPDATE or DELETE triggers needed since the application
-- layer only does INSERTs. Queries are for reporting/forensics.
--
-- Created: 2026-02-16

CREATE TABLE IF NOT EXISTS security_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'mcp_client',
    payload_fingerprint TEXT,
    details_json TEXT,
    outcome TEXT NOT NULL DEFAULT 'executed',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_log_type_created
    ON security_audit_log(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_tool_created
    ON security_audit_log(tool_name, created_at);

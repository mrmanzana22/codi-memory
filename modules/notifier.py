"""
Codi Notifier - Proactive communication with Hare via Telegram.

Can be called from anywhere: sleep loop ticks, event bus handlers,
MCP tools, or daemon. Uses sync HTTP to avoid async complexity.

Rate limited: max 1 message per cooldown period (default 1h).
Cooldown is tracked in SQLite sleep_loop_state table.

Environment:
  TELEGRAM_BOT_TOKEN - Bot token
  TELEGRAM_CHAT_ID   - Hare's chat ID
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime

_logger = logging.getLogger("codi-memory.notifier")

COOLDOWN_HOURS = 1  # Minimum hours between proactive messages


def _get_telegram_config() -> tuple[str, str]:
    """Return (bot_token, chat_id) from environment."""
    return (
        os.getenv("TELEGRAM_BOT_TOKEN", ""),
        os.getenv("TELEGRAM_CHAT_ID", ""),
    )


def send_telegram(message: str) -> bool:
    """Send a message to Hare via Telegram. Returns True on success.

    Handles chunking for messages > 4096 chars.
    No rate limiting — caller is responsible.
    """
    if not message or not message.strip():
        _logger.debug("Empty Telegram message, skipping send")
        return False

    bot_token, chat_id = _get_telegram_config()
    if not bot_token or not chat_id:
        _logger.debug("Telegram not configured, skipping send")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for i in range(0, len(message), 4096):
        chunk = message[i:i + 4096]
        data = json.dumps({"chat_id": chat_id, "text": chunk}).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    _logger.error("Telegram API returned %d", resp.status)
                    return False
        except Exception as e:
            _logger.error("Telegram send failed: %s", e)
            return False

    _logger.info("Telegram sent (%d chars)", len(message))
    return True


def notify_hare(message: str, force: bool = False) -> bool:
    """Send a proactive message to Hare, respecting cooldown.

    Args:
        message: What to tell Hare
        force: If True, ignore cooldown (for urgent/critical messages)

    Returns True if message was sent, False if skipped or failed.
    """
    if not force:
        last = _get_last_notification_time()
        if last:
            from modules.config import now_iso
            now = datetime.fromisoformat(now_iso())
            hours_since = (now - last).total_seconds() / 3600
            if hours_since < COOLDOWN_HOURS:
                _logger.debug(
                    "Cooldown: %.1fh since last notification (need %.1fh)",
                    hours_since, COOLDOWN_HOURS
                )
                return False

    sent = send_telegram(message)
    if sent:
        _record_notification_time()
        # Store in working memory so the daemon has context if Hare replies
        try:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=f"[PROACTIVE MSG SENT] {message}",
                topic="proactive",
                relevance=0.9,
                source="system",
            )
        except Exception as e:
            _logger.debug("WM push for proactive msg failed: %s", e)
    return sent


def notify_hare_urgent(message: str) -> bool:
    """Send an urgent message to Hare, bypassing cooldown."""
    return notify_hare(f"[URGENTE] {message}", force=True)


# -- Cooldown state persistence --

_STATE_KEY = "notifier_last_sent"
_last_sent_in_memory: "datetime | None" = None  # in-process fallback if DB unavailable


def _get_last_notification_time() -> datetime | None:
    """Get last notification timestamp from SQLite."""
    try:
        from modules.config import connect_fts
        conn = connect_fts()
        row = conn.execute(
            "SELECT value FROM sleep_loop_state WHERE key = ?",
            (_STATE_KEY,)
        ).fetchone()
        conn.close()
        if row:
            return datetime.fromisoformat(row[0])
    except Exception:
        pass
    return _last_sent_in_memory


def _record_notification_time():
    """Record current time as last notification in SQLite."""
    global _last_sent_in_memory
    from modules.config import now_iso
    now_str = now_iso()
    _last_sent_in_memory = datetime.fromisoformat(now_str)  # in-memory first (always succeeds)
    try:
        from modules.config import connect_fts
        conn = connect_fts()
        conn.execute(
            "INSERT OR REPLACE INTO sleep_loop_state (key, value) VALUES (?, ?)",
            (_STATE_KEY, now_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        _logger.error("Failed to persist notification time (in-memory fallback active): %s", e)

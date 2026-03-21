"""
Codi Memory - Reward Signal Infrastructure (S1-05)
===================================================
Tracks whether retrieved memories are re-accessed within a reward window.
Provides binary reward signals for Thompson Sampling (S1-04).

Mechanism:
  1. On search_memory(): log each retrieved memory with channel attribution
  2. On subsequent MEMORY_RETRIEVED: check if any pending entries match
  3. Periodically expire entries older than REWARD_WINDOW_SEC

Neuroscience basis:
  - Reinforcement learning: reward = re-use within temporal window
  - Daw et al. 2005: model-free RL uses reward prediction errors
"""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

REWARD_WINDOW_SEC = 300  # 5 minutes: re-access within this = reward
EXPIRE_BATCH_SIZE = 200  # Max rows to expire per tick
LOG_MAX_PER_QUERY = 10   # Max reward entries per search (cap noise)


def _ensure_topic_column(conn) -> None:
    """Ensure reward_signals.topic exists before topic-aware reads/writes."""
    try:
        conn.execute("SELECT topic FROM reward_signals LIMIT 0")
    except sqlite3.OperationalError as e:
        if "no such column" not in str(e).lower():
            raise
        conn.execute("ALTER TABLE reward_signals ADD COLUMN topic TEXT DEFAULT 'general'")
        conn.commit()


# ============================================================
# CORE API
# ============================================================

def log_retrieval(query: str, channel_attributions: list, ts: str = None,
                  topic: str = "general"):
    """Log retrieved memories with their channel attributions.

    Args:
        query: Search query string
        channel_attributions: List of dicts with keys:
            memory_id (str), channels (list[str])
            e.g. [{"memory_id": "abc", "channels": ["vector", "bm25"]}]
        ts: ISO timestamp (defaults to now)
        topic: Query topic for per-topic TS (Canon v2, CC-8)
    """
    if not channel_attributions:
        return

    from modules.db_pool import get_conn
    from modules.config import FTS_DB_PATH, now_iso

    ts = ts or now_iso()

    try:
        conn = get_conn(FTS_DB_PATH)
        # Ensure base table exists before applying incremental schema fixes
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reward_signals ("
            "id INTEGER PRIMARY KEY, "
            "query TEXT NOT NULL, "
            "memory_id TEXT NOT NULL, "
            "channel TEXT NOT NULL, "
            "retrieved_at TEXT NOT NULL, "
            "rewarded INTEGER, "
            "reward_at TEXT)"
        )
        _ensure_topic_column(conn)

        rows = []
        for attr in channel_attributions[:LOG_MAX_PER_QUERY]:
            mid = attr.get("memory_id", "")
            if not mid:
                continue
            for ch in attr.get("channels", []):
                rows.append((query[:200], mid, ch, ts, topic or "general"))

        if rows:
            conn.executemany(
                "INSERT INTO reward_signals (query, memory_id, channel, retrieved_at, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
    except Exception as e:
        _logger.exception("log_retrieval schema bootstrap failed: %s", e)


def check_reward(retrieved_ids: list) -> dict:
    """Check if any recently retrieved memories are being re-accessed.

    Called on MEMORY_RETRIEVED events. If a memory in retrieved_ids
    has a pending (NULL rewarded) entry within REWARD_WINDOW_SEC,
    mark it as rewarded.

    Args:
        retrieved_ids: List of memory IDs being accessed now

    Returns:
        {"rewarded": int, "checked": int}
    """
    if not retrieved_ids:
        return {"rewarded": 0, "checked": 0}

    from modules.db_pool import get_conn
    from modules.config import FTS_DB_PATH, now_iso

    ts = now_iso()
    from modules.config import now_col
    cutoff = (now_col() - timedelta(seconds=REWARD_WINDOW_SEC)).isoformat()

    try:
        conn = get_conn(FTS_DB_PATH)
        rewarded = 0

        for mid in retrieved_ids[:20]:  # Cap to prevent slow queries
            cursor = conn.execute(
                "UPDATE reward_signals SET rewarded = 1, reward_at = ? "
                "WHERE memory_id = ? AND rewarded IS NULL AND retrieved_at > ?",
                (ts, str(mid), cutoff),
            )
            rewarded += cursor.rowcount

        if rewarded > 0:
            conn.commit()

        return {"rewarded": rewarded, "checked": len(retrieved_ids)}
    except Exception as e:
        _logger.debug("check_reward error: %s", e)
        return {"rewarded": 0, "checked": 0}


def expire_pending(max_rows: int = EXPIRE_BATCH_SIZE) -> int:
    """Expire old pending entries (reward=NULL, older than REWARD_WINDOW_SEC).

    Called periodically (e.g., in sleep loop health tick).
    Entries that weren't re-accessed within the window get rewarded=0.

    Returns:
        Number of expired entries
    """
    from modules.db_pool import get_conn
    from modules.config import FTS_DB_PATH, now_col

    cutoff = (now_col() - timedelta(seconds=REWARD_WINDOW_SEC)).isoformat()

    try:
        conn = get_conn(FTS_DB_PATH)
        cursor = conn.execute(
            "UPDATE reward_signals SET rewarded = 0 "
            "WHERE rewarded IS NULL AND retrieved_at < ? "
            "LIMIT ?",
            (cutoff, max_rows),
        )
        expired = cursor.rowcount
        if expired > 0:
            conn.commit()

        # FIFO cleanup: keep last 5000 terminal entries (preserve live pending rows)
        conn.execute("""
            DELETE FROM reward_signals
            WHERE rewarded IS NOT NULL AND retrieved_at < ? AND id NOT IN (
                SELECT id FROM reward_signals
                WHERE rewarded IS NOT NULL AND retrieved_at < ?
                ORDER BY id DESC LIMIT 5000
            )
        """, (cutoff, cutoff))
        conn.commit()

        return expired
    except Exception as e:
        _logger.debug("expire_pending error: %s", e)
        return 0


def get_channel_stats(lookback_hours: int = 24, topic: str = None) -> dict:
    """Get reward statistics per channel for Thompson Sampling.

    Args:
        lookback_hours: How far back to look for reward signals
        topic: If provided, filter by topic for per-topic posteriors (Canon v2, CC-8).
               If None, uses all topics (global).

    Returns:
        {channel: {"successes": int, "failures": int, "total": int}}
    """
    from modules.db_pool import get_conn
    from modules.config import FTS_DB_PATH, now_col

    cutoff = (now_col() - timedelta(hours=lookback_hours)).isoformat()

    try:
        conn = get_conn(FTS_DB_PATH)
        _ensure_topic_column(conn)

        if topic:
            # Per-topic stats (Canon v2, CC-8)
            rows = conn.execute(
                "SELECT channel, rewarded, COUNT(*) FROM reward_signals "
                "WHERE retrieved_at > ? AND rewarded IS NOT NULL AND topic = ? "
                "GROUP BY channel, rewarded",
                (cutoff, topic),
            ).fetchall()
        else:
            # Global stats (backward compatible)
            rows = conn.execute(
                "SELECT channel, rewarded, COUNT(*) FROM reward_signals "
                "WHERE retrieved_at > ? AND rewarded IS NOT NULL "
                "GROUP BY channel, rewarded",
                (cutoff,),
            ).fetchall()

        stats = {}
        for channel, rewarded, count in rows:
            if channel not in stats:
                stats[channel] = {"successes": 0, "failures": 0, "total": 0}
            if rewarded == 1:
                stats[channel]["successes"] += count
            else:
                stats[channel]["failures"] += count
            stats[channel]["total"] += count

        return stats
    except Exception as e:
        _logger.debug("get_channel_stats error: %s", e)
        return {}


# ============================================================
# EVENT WIRING
# ============================================================

_reward_semaphore = threading.BoundedSemaphore(2)


def _on_memory_retrieved(event_name: str, data: dict):
    """Event handler: check for reward signals on retrieval.

    Runs in background thread to avoid blocking the retrieval hot path.
    """
    retrieved_ids = data.get("retrieved_ids", [])
    if not retrieved_ids:
        return

    def _bg():
        if not _reward_semaphore.acquire(blocking=False):
            return
        try:
            result = check_reward(retrieved_ids)
            if result["rewarded"] > 0:
                _logger.debug(
                    "Reward: %d/%d memories re-accessed",
                    result["rewarded"], result["checked"],
                )
        except Exception as e:
            _logger.debug("bg reward check error: %s", e)
        finally:
            _reward_semaphore.release()

    t = threading.Thread(target=_bg, daemon=True, name="reward-check")
    t.start()


def register_reward_handlers():
    """Wire reward tracking to event bus."""
    from modules.events import event_bus, Events
    event_bus.on(Events.MEMORY_RETRIEVED, _on_memory_retrieved)

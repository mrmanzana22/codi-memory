"""
core/time.py — Timezone and timestamp utilities.

Pure functions with zero dependencies on other modules/ code.
Only uses stdlib: datetime, zoneinfo.

This is the single highest-impact extraction from config.py:
9 functions + 1 constant, imported by 42+ modules.
"""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Colombia timezone (UTC-5)
TZ_COL = ZoneInfo("America/Bogota")


def now_col() -> datetime:
    """Current datetime in Colombia timezone."""
    return datetime.now(TZ_COL)


def now_iso() -> str:
    """ISO 8601 timestamp with Colombia timezone."""
    return now_col().isoformat()


def now_display() -> str:
    """Human-readable with zone: '2026-02-07 07:30 COT'"""
    return now_col().strftime("%Y-%m-%d %H:%M COT")


def now_short() -> str:
    """Short format: '2026-02-07 07:30'"""
    return now_col().strftime("%Y-%m-%d %H:%M")


def parse_timestamp(value) -> datetime:
    """Universal parser. Returns Colombia-aware datetime from any of 6 formats.

    Formats handled:
      F1: "2026-03-12T14:30:00.123456-05:00"  (now_iso, Colombia aware)
      F2: "2026-03-12T14:30:00.123456"         (naive local isoformat)
      F3: "2026-03-12 19:30:00"                (SQLite datetime('now'), space-sep)
      F4: "2026-03-12 14:30:00"                (naive Colombia, space-sep)
      F5: "2026-03-05T19:30:00+00:00"          (UTC aware)
      F6: "2026-03-12T19:30:00Z"               (Z suffix)

    Rule: naive -> assume TZ_COL, aware -> convert to TZ_COL.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=TZ_COL)
        return value.astimezone(TZ_COL)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Cannot parse timestamp: {value!r}")
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ_COL)
    return dt.astimezone(TZ_COL)


def parse_sqlite_utc(ts: str) -> datetime:
    """Parse a naive SQLite UTC timestamp (from datetime('now')) as UTC-aware."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def now_sqlite() -> str:
    """Colombia-local naive for TEXT comparisons: '2026-03-12 14:30:00'"""
    return now_col().strftime("%Y-%m-%d %H:%M:%S")


def to_sqlite_utc(value) -> str:
    """Any timestamp -> naive UTC 'YYYY-MM-DD HH:MM:SS'.

    For tables written by SQLite datetime('now') = naive UTC.
    """
    dt = parse_timestamp(value)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def to_sqlite_local(value) -> str:
    """Any timestamp -> naive Colombia 'YYYY-MM-DD HH:MM:SS'.

    For tables written by now_iso() or datetime.now().isoformat() (naive local).
    """
    dt = parse_timestamp(value)
    return dt.astimezone(TZ_COL).strftime("%Y-%m-%d %H:%M:%S")

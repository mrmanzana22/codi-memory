#!/usr/bin/env python3
"""Encode Claude Code JSONL conversation transcripts into FHRR hippocampal index.

Parses raw JSONL files from ~/.claude/projects/, extracts user+assistant turns,
and encodes them as FHRR session records using the existing pipeline.

Supports cross-machine import: encode on MacBook Pro, scp .npz/.json to Air,
then import with --import-dir.

Usage:
    python3 scripts/encode_transcripts.py --dry-run
    python3 scripts/encode_transcripts.py --verbose
    python3 scripts/encode_transcripts.py --import-dir ~/fhrr_import/

Paper: Diekelmann & Born 2010 (offline hippocampal replay)
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, ".")

from modules.config import connect_fts, FTS_DB_PATH

# ---------------------------------------------------------------------------
# JSONL Parser
# ---------------------------------------------------------------------------

# Types to skip entirely (noise)
_SKIP_TYPES = frozenset({
    "file-history-snapshot",
    "progress",
    "result",
})


def parse_jsonl(path: str) -> list[dict]:
    """Extract conversation turns from a Claude Code JSONL transcript.

    Returns list of {"text": str, "role": "user"|"assistant", "turn_id": int}.
    Skips noise (tool calls, progress, snapshots). Minimum 20 chars per turn.
    """
    turns: list[dict] = []
    turn_id = 0

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type", "")

                # Skip noise types
                if msg_type in _SKIP_TYPES:
                    continue

                # queue-operation: initial prompt text
                if msg_type == "queue-operation":
                    content = obj.get("content", "")
                    if isinstance(content, str) and len(content) >= 20:
                        turns.append({"text": content, "role": "user", "turn_id": turn_id})
                        turn_id += 1
                    continue

                # user messages
                if msg_type == "user":
                    text = _extract_user_text(obj)
                    if text and len(text) >= 20:
                        turns.append({"text": text, "role": "user", "turn_id": turn_id})
                        turn_id += 1
                    continue

                # assistant messages
                if msg_type == "assistant":
                    text = _extract_assistant_text(obj)
                    if text and len(text) >= 20:
                        turns.append({"text": text, "role": "assistant", "turn_id": turn_id})
                        turn_id += 1
                    continue

    except (OSError, IOError) as e:
        print(f"  WARN: Could not read {path}: {e}", file=sys.stderr)

    return turns


def _extract_user_text(obj: dict) -> str:
    """Extract text from a user-type JSONL line."""
    msg = obj.get("message", {})
    content = msg.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)

    return ""


def _extract_assistant_text(obj: dict) -> str:
    """Extract text blocks from an assistant-type JSONL line.

    Skips tool_use, tool_result, thinking blocks — only keeps text.
    """
    msg = obj.get("message", {})
    content = msg.get("content", [])

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return " ".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Session Discovery
# ---------------------------------------------------------------------------

def _session_id_from_path(path: str) -> str:
    """Extract session ID from JSONL path (UUID filename)."""
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]


def _get_indexed_session_ids() -> set[str]:
    """Return set of session_ids already in fhrr_session_index."""
    try:
        db = connect_fts(FTS_DB_PATH)
        rows = db.execute("SELECT session_id FROM fhrr_session_index").fetchall()
        db.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def discover_sessions(
    base_dir: str,
    skip_subagents: bool = False,
    limit: int = 0,
) -> list[dict]:
    """Find JSONL transcripts and parse them into encodable sessions.

    Returns list of {session_id, path, turns, num_turns}.
    Skips already-indexed sessions and files with < 3 turns.
    """
    pattern = os.path.join(base_dir, "**", "*.jsonl")
    all_files = sorted(glob.glob(pattern, recursive=True))

    if skip_subagents:
        all_files = [f for f in all_files if "/subagents/" not in f]

    indexed = _get_indexed_session_ids()
    sessions = []

    for path in all_files:
        sid = _session_id_from_path(path)

        if sid in indexed:
            continue

        turns = parse_jsonl(path)
        if len(turns) < 3:
            continue

        sessions.append({
            "session_id": sid,
            "path": path,
            "turns": turns,
            "num_turns": len(turns),
        })

        if limit > 0 and len(sessions) >= limit:
            break

    return sessions


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_and_save(session_info: dict, verbose: bool = False) -> bool:
    """Encode a parsed session into FHRR and save to disk + SQLite."""
    from modules.hippocampal_index import build_session_record, save_session_record

    sid = session_info["session_id"]
    turns = session_info["turns"]

    try:
        record = build_session_record(sid, turns)
        save_session_record(record)
        if verbose:
            topics = record.get("roles", {}).get("topics", [])
            print(f"  OK {sid[:12]} | {len(turns)} turns | topics={topics[:5]}")
        return True
    except Exception as e:
        print(f"  ERR {sid[:12]}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Cross-Machine Import
# ---------------------------------------------------------------------------

def import_directory(import_dir: str, verbose: bool = False) -> dict:
    """Import pre-encoded FHRR session files from another machine.

    Copies .npz and .json files to data/fhrr_sessions/ and registers
    in fhrr_session_index.
    """
    from modules.hippocampal_index import DATA_DIR, load_session_record
    from modules.config import now_iso

    target_dir = os.path.join(DATA_DIR, "fhrr_sessions")
    os.makedirs(target_dir, exist_ok=True)

    indexed = _get_indexed_session_ids()
    imported = 0
    skipped = 0
    errors = 0

    # Find all session directories in the import dir
    for entry in sorted(os.listdir(import_dir)):
        entry_path = os.path.join(import_dir, entry)

        if os.path.isdir(entry_path):
            # Session directory with session.json + session.npz
            json_path = os.path.join(entry_path, "session.json")
            npz_path = os.path.join(entry_path, "session.npz")

            if not os.path.exists(json_path) or not os.path.exists(npz_path):
                continue

            try:
                with open(json_path) as f:
                    meta = json.load(f)
                sid = meta.get("session_id", entry)
            except Exception:
                sid = entry

            if sid in indexed:
                skipped += 1
                continue

            # Copy to target
            dest_dir = os.path.join(target_dir, entry)
            if os.path.exists(dest_dir):
                skipped += 1
                continue

            try:
                shutil.copytree(entry_path, dest_dir)

                # Register in SQLite
                db = connect_fts(FTS_DB_PATH)
                file_size = os.path.getsize(os.path.join(dest_dir, "session.json")) + \
                            os.path.getsize(os.path.join(dest_dir, "session.npz"))
                topics = meta.get("roles", {}).get("topics", [])
                db.execute("""
                    INSERT OR IGNORE INTO fhrr_session_index
                    (session_id, encoded_at, num_turns, num_chunks,
                     file_size_bytes, encoding_time_ms, topics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sid, now_iso(),
                    meta.get("num_turns", 0),
                    len(meta.get("chunks", [])),
                    file_size, 0.0,
                    json.dumps(topics),
                ))
                db.close()

                imported += 1
                if verbose:
                    print(f"  IMPORTED {sid[:12]} | {meta.get('num_turns', '?')} turns")
            except Exception as e:
                errors += 1
                print(f"  ERR importing {entry}: {e}", file=sys.stderr)

    return {"imported": imported, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_encode(
    base_dir: str,
    dry_run: bool = False,
    limit: int = 0,
    skip_subagents: bool = False,
    verbose: bool = False,
) -> dict:
    """Main encode pipeline."""
    print(f"Scanning {base_dir} ...")
    t0 = time.monotonic()

    sessions = discover_sessions(base_dir, skip_subagents=skip_subagents, limit=limit)
    scan_time = time.monotonic() - t0

    print(f"Found {len(sessions)} sessions to encode ({scan_time:.1f}s scan)")

    if dry_run:
        total_turns = sum(s["num_turns"] for s in sessions)
        print(f"\n  Total turns: {total_turns}")
        print(f"  Avg turns/session: {total_turns / max(len(sessions), 1):.0f}")
        print()
        for s in sessions[:15]:
            print(f"  {s['session_id'][:12]} | {s['num_turns']:4d} turns | {os.path.basename(os.path.dirname(s['path']))}")
        if len(sessions) > 15:
            print(f"  ... and {len(sessions) - 15} more")
        return {"total": len(sessions), "encoded": 0, "errors": 0, "dry_run": True}

    encoded = 0
    errors = 0
    t0 = time.monotonic()

    for i, session in enumerate(sessions):
        ok = encode_and_save(session, verbose=verbose)
        if ok:
            encoded += 1
        else:
            errors += 1

        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(sessions)}] encoded={encoded} errors={errors} "
                  f"elapsed={elapsed:.1f}s rate={rate:.1f}/s")

    elapsed = time.monotonic() - t0
    print(f"\nDone: encoded={encoded} errors={errors} elapsed={elapsed:.1f}s")
    return {"total": len(sessions), "encoded": encoded, "errors": errors, "elapsed_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(
        description="Encode Claude Code JSONL transcripts into FHRR hippocampal index"
    )
    parser.add_argument(
        "--dir", type=str,
        default=os.path.expanduser("~/.claude/projects/-Users-codi-air"),
        help="Transcript directory to scan",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max sessions to encode (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without encoding")
    parser.add_argument("--skip-subagents", action="store_true", help="Skip subagent directories")
    parser.add_argument("--verbose", action="store_true", help="Show per-session progress")
    parser.add_argument(
        "--import-dir", type=str, default="",
        help="Import pre-encoded sessions from another machine",
    )

    args = parser.parse_args()

    if args.import_dir:
        print(f"Importing from {args.import_dir} ...")
        result = import_directory(args.import_dir, verbose=args.verbose)
        print(f"\nImport done: {result['imported']} imported, "
              f"{result['skipped']} skipped, {result['errors']} errors")
        sys.exit(0 if result["errors"] == 0 else 1)

    result = run_encode(
        base_dir=args.dir,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_subagents=args.skip_subagents,
        verbose=args.verbose,
    )
    sys.exit(0 if result.get("errors", 0) == 0 else 1)


if __name__ == "__main__":
    main()

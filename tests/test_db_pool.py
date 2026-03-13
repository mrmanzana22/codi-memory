"""Tests for PR-P1c: Thread-local SQLite connection pool."""

import os
import sys
import sqlite3
import threading
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDbPool:
    """Test the db_pool module."""

    def _make_temp_db(self):
        """Create a temp SQLite DB for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # Initialize with a simple table
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
        conn.close()
        return path

    def test_same_thread_reuses_connection(self, tmp_path):
        """Same thread should get the exact same connection object."""
        from modules.db_pool import get_conn, close_thread_connections

        db_path = str(tmp_path / "test.db")
        # Create the DB file
        sqlite3.connect(db_path).close()

        try:
            conn1 = get_conn(db_path)
            conn2 = get_conn(db_path)
            assert conn1 is conn2, "Same thread should reuse connection"
        finally:
            close_thread_connections()

    def test_different_threads_get_different_connections(self, tmp_path):
        """Different threads should NOT share connections."""
        from modules.db_pool import get_conn, close_thread_connections

        db_path = str(tmp_path / "test.db")
        sqlite3.connect(db_path).close()

        conn_main = get_conn(db_path)
        conn_ids = {}

        def worker():
            try:
                conn = get_conn(db_path)
                conn_ids["worker"] = id(conn)
            finally:
                close_thread_connections()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert id(conn_main) != conn_ids["worker"], \
            "Different threads must have different connections"
        close_thread_connections()

    def test_different_db_paths_get_different_connections(self, tmp_path):
        """Different db_paths should get different connections."""
        from modules.db_pool import get_conn, close_thread_connections

        db1 = str(tmp_path / "db1.db")
        db2 = str(tmp_path / "db2.db")
        sqlite3.connect(db1).close()
        sqlite3.connect(db2).close()

        try:
            conn1 = get_conn(db1)
            conn2 = get_conn(db2)
            assert conn1 is not conn2, "Different paths should have different connections"
        finally:
            close_thread_connections()

    def test_pragmas_are_set_correctly(self, tmp_path):
        """Connection should have WAL, synchronous=NORMAL, busy_timeout=30000, foreign_keys=ON."""
        from modules.db_pool import get_conn, close_thread_connections

        db_path = str(tmp_path / "test.db")
        sqlite3.connect(db_path).close()

        try:
            conn = get_conn(db_path)

            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert journal == "wal", f"Expected WAL, got {journal}"

            sync = conn.execute("PRAGMA synchronous").fetchone()[0]
            assert sync == 1, f"Expected synchronous=1 (NORMAL), got {sync}"

            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == 30000, f"Expected busy_timeout=30000, got {timeout}"

            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1, f"Expected foreign_keys=1 (ON), got {fk}"
        finally:
            close_thread_connections()

    def test_row_factory_is_row(self, tmp_path):
        """Connection should have row_factory = sqlite3.Row."""
        from modules.db_pool import get_conn, close_thread_connections

        db_path = str(tmp_path / "test.db")
        conn_setup = sqlite3.connect(db_path)
        conn_setup.execute("CREATE TABLE kv (key TEXT, val TEXT)")
        conn_setup.execute("INSERT INTO kv VALUES ('a', 'b')")
        conn_setup.commit()
        conn_setup.close()

        try:
            conn = get_conn(db_path)
            row = conn.execute("SELECT key, val FROM kv").fetchone()
            assert row["key"] == "a", "row_factory should allow dict-like access"
            assert row["val"] == "b"
        finally:
            close_thread_connections()

    def test_dead_connection_recreated(self, tmp_path):
        """If a connection is closed externally, pool should recreate it."""
        from modules.db_pool import get_conn, close_thread_connections

        db_path = str(tmp_path / "test.db")
        sqlite3.connect(db_path).close()

        try:
            conn1 = get_conn(db_path)
            # Externally close the connection (simulate crash/corruption)
            conn1.close()

            # Pool should detect dead connection and recreate
            conn2 = get_conn(db_path)
            assert conn2 is not conn1, "Dead connection should be replaced"
            # New connection should work
            conn2.execute("SELECT 1")
        finally:
            close_thread_connections()

    def test_close_thread_connections(self, tmp_path):
        """close_thread_connections should close all and clear the dict."""
        from modules.db_pool import get_conn, close_thread_connections

        db1 = str(tmp_path / "db1.db")
        db2 = str(tmp_path / "db2.db")
        sqlite3.connect(db1).close()
        sqlite3.connect(db2).close()

        conn1 = get_conn(db1)
        conn2 = get_conn(db2)

        close_thread_connections()

        # After closing, new get_conn should create new connections
        conn1_new = get_conn(db1)
        assert conn1_new is not conn1, "After close, should get new connection"
        close_thread_connections()

    def test_close_all_is_alias(self, tmp_path):
        """close_all() should work same as close_thread_connections()."""
        from modules.db_pool import get_conn, close_all

        db_path = str(tmp_path / "test.db")
        sqlite3.connect(db_path).close()

        conn1 = get_conn(db_path)
        close_all()

        conn2 = get_conn(db_path)
        assert conn2 is not conn1
        close_all()

    def test_concurrent_access_no_crash(self, tmp_path):
        """Multiple threads using pool concurrently should not crash."""
        from modules.db_pool import get_conn, close_thread_connections
        import time

        db_path = str(tmp_path / "concurrent.db")
        conn_setup = sqlite3.connect(db_path)
        conn_setup.execute("PRAGMA journal_mode=WAL")
        conn_setup.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, val INTEGER)")
        conn_setup.execute("INSERT INTO counter VALUES (1, 0)")
        conn_setup.commit()
        conn_setup.close()

        errors = []

        def worker(n):
            try:
                conn = get_conn(db_path)
                for _ in range(10):
                    # Use BEGIN IMMEDIATE so busy_timeout applies at lock acquisition
                    # (DEFERRED can fail with "database is locked" on upgrade)
                    for attempt in range(3):
                        try:
                            conn.execute("BEGIN IMMEDIATE")
                            conn.execute("UPDATE counter SET val = val + 1 WHERE id = 1")
                            conn.commit()
                            break
                        except sqlite3.OperationalError:
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            if attempt == 2:
                                raise
                            time.sleep(0.01 * (attempt + 1))
            except Exception as e:
                errors.append(str(e))
            finally:
                close_thread_connections()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

"""The one-time migration that drops the P4a compaction schema (respec §3.3, RULING-6).

This is the only piece of W1 that touches a database somebody already has. Every other part of
the excision is code that stops existing; this part has to reach into a live file, remove ten
tables that may hold megabytes of summary payload blobs and crawl skeletons, and leave everything
else exactly as it was.

So the tests build REAL SQLite files with REAL rows rather than mocking the connection. What is
being checked is not "does it call DROP" — it is that a half-finished run leaves a database the
next boot can finish, and that a finished run never runs again.
"""
from __future__ import annotations

import sqlite3

import pytest

from database import _P4A_CLEANUP_KEY, DatabaseManager

DROPPED = (
    "compaction_cancellation_intent", "pending_recompaction", "compaction_telemetry_outbox",
    "compaction_event_skeleton", "compaction_crawl_checkpoints", "snapshot_anchor_provenance",
    "snapshot_capture_manifest", "snapshot_mutation_observations",
    "channel_snapshot_pointer", "channel_snapshots",
)

# Rows Slack does not still hold, and rows it does. The second group must survive untouched:
# the whole justification for a DROP rather than a retire is that nothing dropped is a
# transcript, a config row, or anything recoverable only from here.
KEPT = ("channel_thread_activity", "channel_coverage", "outbound_receipts", "bot_meta")


def _populated(path):
    """A database carrying all ten P4a tables WITH rows, plus the tables that must survive."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    for table in DROPPED:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, payload BLOB)")
        conn.execute(f"INSERT INTO {table} (payload) VALUES (?)", (b"x" * 4096,))
    conn.execute("CREATE TABLE bot_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE channel_thread_activity (root_ts TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO channel_thread_activity VALUES ('100.000100')")
    conn.execute("CREATE TABLE channel_coverage (channel_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO channel_coverage VALUES ('C1')")
    conn.execute("CREATE TABLE outbound_receipts (message_ts TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO outbound_receipts VALUES ('200.000200')")
    return conn


class _Conn:
    """A pass-through around a real connection whose `execute` CAN be intercepted.

    `sqlite3.Connection.execute` is read-only, so a test that needs to watch or fail one
    statement has to wrap rather than patch. Everything else forwards, so the statements below
    still run against real SQLite and real transaction semantics.
    """

    def __init__(self, conn):
        self._conn = conn
        self.on_execute = None

    def execute(self, sql, *args):
        if self.on_execute is not None:
            self.on_execute(sql)
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _manager(conn):
    """A DatabaseManager bound to `conn` without booting the whole schema.

    `__init__` runs init_schema and every migration, which is the wrong subject: this file is
    about ONE step, run against a database that predates it.
    """
    manager = DatabaseManager.__new__(DatabaseManager)
    manager.conn = _Conn(conn) if not isinstance(conn, _Conn) else conn
    return manager


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


@pytest.fixture
def db(tmp_path):
    conn = _populated(tmp_path / "p4a.db")
    yield conn
    conn.close()


def test_a_populated_p4a_database_is_dropped_once(db):
    """T3. Every table gone, the marker set, and a SECOND run a no-op that opens no transaction.

    The no-transaction part is the interesting half: the guard has to return before
    `BEGIN IMMEDIATE`, not merely before the drops. A step that opened a write transaction on
    every boot to discover it had nothing to do would take the write lock on every boot.
    """
    manager = _manager(db)
    assert set(DROPPED) <= _tables(db)

    manager._migrate_drop_compaction_schema()

    assert not (_tables(db) & set(DROPPED))
    assert set(KEPT) <= _tables(db)
    assert manager.get_meta(_P4A_CLEANUP_KEY)
    # Nothing Slack does not still hold: the surviving rows are all present and unchanged.
    assert db.execute("SELECT root_ts FROM channel_thread_activity").fetchone()[0] == "100.000100"
    assert db.execute("SELECT channel_id FROM channel_coverage").fetchone()[0] == "C1"
    assert db.execute("SELECT message_ts FROM outbound_receipts").fetchone()[0] == "200.000200"

    opened = []
    manager.conn.on_execute = lambda sql: (
        opened.append(sql) if "BEGIN" in str(sql).upper() else None)
    manager._migrate_drop_compaction_schema()
    manager.conn.on_execute = None
    assert opened == [], f"the second run opened a transaction: {opened}"


def test_a_failure_part_way_through_leaves_the_database_retryable(db):
    """The step is required-critical and ATOMIC: the drops and the marker commit together or not
    at all. A partial drop that recorded the marker would strand the remaining tables where
    nothing will ever revisit them; a partial drop that did not is simply retried."""
    manager = _manager(db)
    calls = {"n": 0}

    def failing(sql):
        if str(sql).startswith("DROP TABLE"):
            calls["n"] += 1
            if calls["n"] == 4:
                raise sqlite3.OperationalError("disk I/O error")

    manager.conn.on_execute = failing
    with pytest.raises(sqlite3.OperationalError):
        manager._migrate_drop_compaction_schema()
    manager.conn.on_execute = None
    assert calls["n"] == 4, "the failure must land part way through the drop list"

    assert manager.get_meta(_P4A_CLEANUP_KEY) is None, "the marker outlived a rolled-back drop"
    assert set(DROPPED) <= _tables(db), "a partial drop was committed"
    assert set(KEPT) <= _tables(db)

    manager._migrate_drop_compaction_schema()
    assert not (_tables(db) & set(DROPPED))
    assert manager.get_meta(_P4A_CLEANUP_KEY)


def test_a_database_that_never_had_the_tables_is_still_marked(tmp_path):
    """A fresh installation has nothing to drop. It must still record the marker, or every boot
    forever opens a write transaction to rediscover that."""
    conn = sqlite3.connect(str(tmp_path / "fresh.db"), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE bot_meta (key TEXT PRIMARY KEY, value TEXT)")
    manager = _manager(conn)
    try:
        manager._migrate_drop_compaction_schema()
        assert manager.get_meta(_P4A_CLEANUP_KEY)
        assert not (_tables(conn) & set(DROPPED))
    finally:
        conn.close()


def test_a_real_boot_migrates_one_populated_legacy_database(tmp_path, monkeypatch):
    """Both destructive migrations, on ONE realistic file, through `DatabaseManager` itself.

    T3 and T9 each drive a helper against its own synthetic database. Neither proves the pair
    runs in the right order on the file an operator actually has: a P4a-era database carries the
    ten compaction tables AND the old `coverage_*` column names, and the drop is ordered ahead of
    every other migration step. Boot ordering is exactly the sort of thing that is correct in
    each part and wrong in the whole.
    """
    import os
    import sqlite3

    from database import _COVERAGE_RENAME_KEY, DatabaseManager

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    for table in DROPPED:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, payload BLOB)")
        conn.execute(f"INSERT INTO {table} (payload) VALUES (?)", (b"y" * 2048,))
    conn.execute("""
        CREATE TABLE channel_coverage (
            team_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            coverage_start_ts TEXT NOT NULL,
            bootstrap_status TEXT NOT NULL
                CHECK (bootstrap_status IN ('pending', 'running', 'complete', 'limited')),
            coverage_reason TEXT,
            sweep_token TEXT,
            heartbeat_ts TIMESTAMP,
            updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, channel_id)
        )
    """)
    conn.execute("INSERT INTO channel_coverage (team_id, channel_id, coverage_start_ts, "
                 "bootstrap_status, coverage_reason) "
                 "VALUES ('T1','C1','1000.000100','limited','depth_config')")
    conn.execute("CREATE TABLE channel_thread_activity (root_ts TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO channel_thread_activity VALUES ('100.000100')")
    conn.close()

    monkeypatch.setitem(os.environ, "DATABASE_DIR", str(tmp_path))
    db = DatabaseManager("legacy")          # __init__ runs init_schema + every migration
    try:
        assert db.db_path == str(path), "the boot must open the legacy file, not a new one"
        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert not (tables & set(DROPPED)), sorted(tables & set(DROPPED))

        columns = {r[1] for r in db.conn.execute("PRAGMA table_info(channel_coverage)")}
        assert {"inventory_start_ts", "inventory_reason"} <= columns
        assert not ({"coverage_start_ts", "coverage_reason"} & columns)

        # The rows Slack does not hold came through both migrations intact.
        row = db.conn.execute("SELECT * FROM channel_coverage").fetchone()
        assert (row["inventory_start_ts"], row["bootstrap_status"], row["inventory_reason"]) == (
            "1000.000100", "limited", "depth_config")
        assert db.conn.execute(
            "SELECT root_ts FROM channel_thread_activity").fetchone()[0] == "100.000100"

        assert db.get_meta(_P4A_CLEANUP_KEY) and db.get_meta(_COVERAGE_RENAME_KEY)
    finally:
        db.close()

    # A SECOND boot on the migrated file is a clean no-op — the ordinary case forever after.
    again = DatabaseManager("legacy")
    try:
        assert again.conn.execute("SELECT COUNT(*) FROM channel_coverage").fetchone()[0] == 1
    finally:
        again.close()

"""
db.py — shared database connection layer
=========================================

auth.py and server.py both talk to SQLite. This module gives them one place
to open a connection that transparently works in two modes:

  Local dev   (default):  standard sqlite3 against a file on disk.
  Production  (Turso):    libsql_client against a hosted SQLite-compatible
                          database. The .execute() / .fetchone() / .fetchall()
                          API is the same, so the calling code is unchanged.

Turso mode is activated by setting two environment variables:
    TURSO_URL    — e.g. "libsql://afibai-db-yourname.turso.io"
    TURSO_TOKEN  — the auth token from the Turso dashboard

In local mode two separate files are used (users.db for auth, afib_history.db
for recordings). In Turso mode both apps share the same remote database, so
they each get a connection to the same hosted DB.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

USERS_DB   = "users.db"
HISTORY_DB = "afib_history.db"


def is_turso() -> bool:
    """True if Turso env vars are set — callers should use the libsql backend."""
    return bool(os.environ.get("TURSO_URL") and os.environ.get("TURSO_TOKEN"))


def _libsql_connect():
    """Open a libsql (Turso) connection. Uses the synchronous client so the
    context manager below can yield values without async ceremony."""
    from libsql_client import create_client_sync
    url   = os.environ["TURSO_URL"]
    token = os.environ["TURSO_TOKEN"]
    return create_client_sync(url=url, auth_token=token)


@contextmanager
def connect(db_path: str):
    """Yield a connection.

    Local mode: a sqlite3 connection opened on the given file. The `db_path`
    argument is meaningful (e.g. "users.db" or "afib_history.db").

    Turso mode: a libsql_client client. The `db_path` argument is ignored —
    there's only one remote DB, regardless of which caller asks for it.

    Both backends expose the same SQL surface:
        .execute(sql, params)  → no useful return
        .execute(sql, params).fetchone() → Row or None
        .execute(sql, params).fetchall() → list of Rows
        .row_factory = sqlite3.Row    (so rows can be accessed by column name)
    """
    if is_turso():
        # libsql_client already returns rows that behave like sqlite3.Row for
        # column-name access; we wrap the API so the .execute() call returns
        # something the caller can .fetchone() / .fetchall() on.
        client = _libsql_connect()

        class _Conn:
            def __init__(self):
                self.row_factory = sqlite3.Row

            def execute(self, sql, params=()):
                # libsql_client.execute returns a ResultSet synchronously.
                return _Cursor(client.execute(sql, list(params)))

            def commit(self):
                pass  # libsql client auto-commits each statement

            def close(self):
                client.close()

        class _Cursor:
            def __init__(self, rs):
                self._rs = rs
                self._rows = list(rs.rows) if rs.rows else []

            def fetchone(self):
                if not self._rows:
                    return None
                return self._rows.pop(0)

            def fetchall(self):
                return list(self._rows)

        try:
            yield _Conn()
        except Exception:
            raise
        finally:
            client.close()
    else:
        con = sqlite3.connect(db_path)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
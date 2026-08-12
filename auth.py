"""
auth.py — Real authentication & user database for AFibAI
=========================================================

Replaces the mock login that was previously in app.py.

- SQLite database (users.db) created automatically on first run
- Passwords hashed with bcrypt — never stored in plain text
- One hardcoded admin account: username "admin", password "230709"
- Helpers used by app.py for: signup, signin, intake, get current user,
  list users (admin), delete user (admin)
"""

import sqlite3
import os
import bcrypt
from contextlib import contextmanager
from pathlib import Path

import db as _db

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
APP_DIR    = Path(__file__).parent.resolve()
DB_PATH    = str(APP_DIR / _db.USERS_DB)   # local mode only; ignored in Turso mode
ADMIN_USER = "admin"
ADMIN_PASS = "230709"   # hardcoded by request

# ── Role values ─────────────────────────────────────────────────────────────
# Stored as TEXT in SQLite. New column added with a migration in init_db().
ROLE_PATIENT   = "patient"
ROLE_CLINICIAN = "clinician"
VALID_ROLES    = (ROLE_PATIENT, ROLE_CLINICIAN)

# Demo accounts seeded by init_db() so you can log in as either role right away.
# Passwords are intentional and only safe because this is a research prototype.
DEMO_CLINICIAN_USER = "doctor"
DEMO_CLINICIAN_PASS = "doctor123"
DEMO_PATIENT_USER   = "patient"
DEMO_PATIENT_PASS   = "patient123"

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _conn():
    """Open a connection. Local: sqlite3 file. Production (Turso env vars
    set): libsql remote DB. The db.py wrapper exposes the same .execute() /
    .fetchone() / .row_factory surface, so all the code below this point is
    unchanged."""
    with _db.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        yield con


def init_db():
    """Create the users table if it doesn't exist, and seed default rows.

    Also migrates older databases to add the `role` column if it's missing —
    this matters because the table was created before patient/clinician roles
    existed, and a plain CREATE TABLE IF NOT EXISTS won't add columns.
    """
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    UNIQUE NOT NULL,
                password_hash   TEXT    NOT NULL,
                is_admin        INTEGER NOT NULL DEFAULT 0,
                role            TEXT    NOT NULL DEFAULT 'clinician',
                full_name       TEXT,
                age             INTEGER,
                gender          TEXT,
                weight_kg       REAL,
                height_cm       REAL,
                smoking         TEXT,
                medical_history TEXT,
                medications     TEXT,
                allergies       TEXT,
                family_history  TEXT,
                previous_ecg    TEXT,
                reason          TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration for databases created before the `role` column existed.
        cols = [r["name"] for r in con.execute("PRAGMA table_info(users)").fetchall()]
        if "role" not in cols:
            con.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'clinician'"
            )

        # Seed the admin if it doesn't already exist.
        row = con.execute(
            "SELECT id FROM users WHERE username = ?", (ADMIN_USER,)
        ).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO users (username, password_hash, is_admin, full_name, role) "
                "VALUES (?, ?, 1, ?, ?)",
                (ADMIN_USER, _hash_password(ADMIN_PASS), "Administrator", ROLE_CLINICIAN),
            )

        # Seed a demo clinician + demo patient so you can log in as either role
        # immediately. Skipped silently if they already exist.
        for username, password, full_name, role in [
            (DEMO_CLINICIAN_USER, DEMO_CLINICIAN_PASS, "Dr. Demo",   ROLE_CLINICIAN),
            (DEMO_PATIENT_USER,   DEMO_PATIENT_PASS,   "Patient Demo", ROLE_PATIENT),
        ]:
            existing = con.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing is None:
                con.execute(
                    "INSERT INTO users (username, password_hash, full_name, role) "
                    "VALUES (?, ?, ?, ?)",
                    (username, _hash_password(password), full_name, role),
                )


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# AUTH HELPERS  (called by app.py)
# ═══════════════════════════════════════════════════════════════════════════

def signup(username: str, password: str, role: str = ROLE_CLINICIAN) -> tuple[bool, str]:
    """
    Create a new user with the given role.
    Returns (ok, message). On success, message is empty.
    Defaults to clinician so existing call sites don't need to change.
    """
    u = (username or "").strip()
    if not u or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if role not in VALID_ROLES:
        return False, "Invalid role."
    with _conn() as con:
        existing = con.execute(
            "SELECT id FROM users WHERE username = ?", (u,)
        ).fetchone()
        if existing is not None:
            return False, "That username is already taken."
        con.execute(
            "INSERT INTO users (username, password_hash, is_admin, role) "
            "VALUES (?, ?, 0, ?)",
            (u, _hash_password(password), role),
        )
    return True, ""


def signin(username: str, password: str) -> tuple[bool, dict | None, str]:
    """
    Verify username + password.
    Returns (ok, user_dict_or_None, message).
    user_dict has keys: id, username, is_admin, full_name, age, ...
    """
    u = (username or "").strip()
    if not u or not password:
        return False, None, "Please enter both a username and a password."
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username = ?", (u,)
        ).fetchone()
    if row is None:
        return False, None, "Username not found."
    if not _verify_password(password, row["password_hash"]):
        return False, None, "Incorrect password."
    return True, dict(row), ""


def save_intake(user_id: int, data: dict) -> None:
    """Update the medical-info fields for an existing user."""
    cols = ("full_name","age","gender","weight_kg","height_cm","smoking",
            "medical_history","medications","allergies","family_history",
            "previous_ecg","reason")
    fields = {c: data.get(c) for c in cols}
    set_clause = ", ".join(f"{c} = ?" for c in cols)
    values     = [fields[c] for c in cols] + [user_id]
    with _conn() as con:
        con.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?", values
        )


def get_user(user_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_role(user_id: int) -> str:
    """Return the role for the given user, defaulting to clinician if missing."""
    user = get_user(user_id)
    if not user:
        return ROLE_CLINICIAN
    role = user.get("role") or ROLE_CLINICIAN
    return role if role in VALID_ROLES else ROLE_CLINICIAN


# ═══════════════════════════════════════════════════════════════════════════
# ADMIN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def list_users() -> list[dict]:
    """Return every user except the built-in admin (admin is always protected)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, username, full_name, age, gender, is_admin, created_at "
            "FROM users WHERE username != ? ORDER BY id ASC",
            (ADMIN_USER,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: int) -> tuple[bool, str]:
    """
    Delete a user by id. The built-in admin account can never be deleted
    (it has a fixed username 'admin' that we block by id, not by username,
    so even a future renamed admin row would be safe — but we also block
    by username as a belt-and-suspenders).
    """
    with _conn() as con:
        row = con.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return False, "User not found."
        if row["username"] == ADMIN_USER:
            return False, "The built-in admin account cannot be deleted."
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return True, ""

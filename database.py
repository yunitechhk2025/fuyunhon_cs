import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DB_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "qa.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    existing = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'agent',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                mode_used TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                ai_suggested_answer TEXT,
                match_score REAL,
                final_answer TEXT,
                claimed_by INTEGER,
                claimed_by_name TEXT,
                answered_by INTEGER,
                answered_by_name TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
            CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
            """
        )

        # 兼容已部署的旧数据库：补充检索标注字段（是否命中/命中的题库问题与答案）
        _add_column_if_missing(conn, "conversations", "matched", "INTEGER")
        _add_column_if_missing(conn, "conversations", "matched_question", "TEXT")
        _add_column_if_missing(conn, "conversations", "matched_answer", "TEXT")

        row = conn.execute("SELECT value FROM settings WHERE key = 'global_mode'").fetchone()
        if row is None:
            default_mode = os.getenv("DEFAULT_MODE", "auto")
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('global_mode', ?)", (default_mode,)
            )

        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        exists = conn.execute(
            "SELECT id FROM agents WHERE username = ?", (admin_username,)
        ).fetchone()
        if exists is None:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            password_hash = bcrypt.hashpw(admin_password.encode("utf-8"), bcrypt.gensalt()).decode(
                "utf-8"
            )
            conn.execute(
                "INSERT INTO agents (username, password_hash, display_name, role) VALUES (?, ?, ?, 'admin')",
                (admin_username, password_hash, "管理员"),
            )
            print(
                f"[init] 已创建默认管理员账号: {admin_username} / {admin_password}（请登录后尽快修改密码）"
            )


# ---------- settings ----------

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


# ---------- agents ----------

def create_agent(username: str, password: str, display_name: str, role: str = "agent") -> int:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO agents (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (username, password_hash, display_name, role),
        )
        return cur.lastrowid


def get_agent_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM agents WHERE username = ?", (username,)).fetchone()


def get_agent_by_id(agent_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()


def list_agents() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, role, created_at FROM agents ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def update_agent_password(agent_id: int, new_password: str) -> None:
    password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        conn.execute("UPDATE agents SET password_hash = ? WHERE id = ?", (password_hash, agent_id))


# ---------- conversations ----------

def create_conversation(session_id: str, question: str, mode_used: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (session_id, question, mode_used) VALUES (?, ?, ?)",
            (session_id, question, mode_used),
        )
        return cur.lastrowid


def set_retrieval_info(
    conversation_id: int,
    matched: bool,
    matched_question: Optional[str],
    matched_answer: Optional[str],
    score: float,
) -> None:
    """记录本次提问在题库中的检索结果：命中的问题/答案，或未命中。"""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET matched = ?, matched_question = ?, matched_answer = ?, match_score = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (1 if matched else 0, matched_question, matched_answer, score, conversation_id),
        )


def set_ai_suggestion(conversation_id: int, suggestion: str, score: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET ai_suggested_answer = ?, match_score = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (suggestion, score, conversation_id),
        )


def mark_answered(conversation_id: int, final_answer: str, answered_by: Optional[int] = None,
                   answered_by_name: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET final_answer = ?, status = 'answered', answered_by = ?, answered_by_name = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (final_answer, answered_by, answered_by_name, conversation_id),
        )


def claim_conversation(conversation_id: int, agent_id: int, agent_name: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, claimed_by FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None or row["status"] == "answered":
            return False
        if row["claimed_by"] and row["claimed_by"] != agent_id:
            return False
        conn.execute(
            """
            UPDATE conversations
            SET status = 'claimed', claimed_by = ?, claimed_by_name = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (agent_id, agent_name, conversation_id),
        )
        return True


def release_conversation(conversation_id: int, agent_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT claimed_by FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None or row["claimed_by"] != agent_id:
            return False
        conn.execute(
            """
            UPDATE conversations
            SET status = 'pending', claimed_by = NULL, claimed_by_name = NULL, updated_at = datetime('now')
            WHERE id = ?
            """,
            (conversation_id,),
        )
        return True


def get_conversation(conversation_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()


def list_queue() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM conversations
            WHERE status IN ('pending', 'claimed')
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def list_recent(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

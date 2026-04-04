import re
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Protocol, Tuple
from uuid import uuid4


DEFAULT_PROJECT_ID = "general"
DEFAULT_PROJECT_NAME = "General"
DEFAULT_PROJECT_DESCRIPTION = "Default workspace for documents and conversations."


class HistoryStore(Protocol):
    def list_projects(self) -> List[Tuple[str, str, str, str, int]]:
        ...

    def create_project(self, name: str, description: str = "") -> Tuple[str, str, str, str]:
        ...

    def get_project(self, project_id: str) -> Optional[Tuple[str, str, str, str]]:
        ...

    def update_project(self, project_id: str, name: str, description: str = "") -> Tuple[str, str, str, str]:
        ...

    def delete_project(self, project_id: str) -> Tuple[int, int]:
        ...

    def ensure_project(self, project_id: Optional[str]) -> str:
        ...

    def start_new_session(self, project_id: Optional[str] = None) -> str:
        ...

    def save_message(self, session_id: str, role: str, content: str) -> None:
        ...

    def get_session_history(self, session_id: str) -> List[Tuple[str, str, str]]:
        ...

    def get_all_sessions(self, project_id: Optional[str] = None) -> List[Tuple[str, str, str, str]]:
        ...

    def get_session(self, session_id: str) -> Optional[Tuple[str, str, str, str]]:
        ...

    def get_session_project_id(self, session_id: str) -> Optional[str]:
        ...

    def update_session_title(self, session_id: str, title: str) -> None:
        ...

    def delete_old_sessions(self, days: int = 30) -> int:
        ...

    def get_session_title_suggestion(self, session_id: str) -> str:
        ...


class ChatHistoryDB:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.create_tables()

    def create_tables(self):
        """Create and migrate the SQLite schema for projects, sessions, and messages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    start_time TEXT,
                    title TEXT,
                    FOREIGN KEY (project_id) REFERENCES chat_projects(project_id)
                )
                """
            )

            if not self._column_exists(cursor, "chat_sessions", "project_id"):
                cursor.execute("ALTER TABLE chat_sessions ADD COLUMN project_id TEXT")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_project_id
                ON chat_sessions(project_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id)
                """
            )

            self._ensure_default_project(cursor)
            cursor.execute(
                """
                UPDATE chat_sessions
                SET project_id = ?
                WHERE project_id IS NULL OR project_id = ''
                """,
                (DEFAULT_PROJECT_ID,),
            )
            conn.commit()

    def list_projects(self) -> List[Tuple[str, str, str, str, int]]:
        """Return project metadata plus session counts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    p.project_id,
                    p.name,
                    COALESCE(p.description, ''),
                    p.created_at,
                    COUNT(s.session_id) AS session_count
                FROM chat_projects p
                LEFT JOIN chat_sessions s ON s.project_id = p.project_id
                GROUP BY p.project_id, p.name, p.description, p.created_at
                ORDER BY
                    CASE WHEN p.project_id = ? THEN 0 ELSE 1 END,
                    p.created_at DESC
                """,
                (DEFAULT_PROJECT_ID,),
            )
            return cursor.fetchall()

    def create_project(self, name: str, description: str = "") -> Tuple[str, str, str, str]:
        """Create a project and return its metadata."""
        normalized_name = name.strip() or "Untitled Project"
        project_id = self._build_project_id(normalized_name)
        created_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_projects (project_id, name, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, normalized_name, description.strip(), created_at),
            )
            conn.commit()

        return project_id, normalized_name, description.strip(), created_at

    def get_project(self, project_id: str) -> Optional[Tuple[str, str, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT project_id, name, COALESCE(description, ''), created_at
                FROM chat_projects
                WHERE project_id = ?
                """,
                (project_id,),
            )
            return cursor.fetchone()

    def update_project(self, project_id: str, name: str, description: str = "") -> Tuple[str, str, str, str]:
        if project_id == DEFAULT_PROJECT_ID:
            raise ValueError("The default project cannot be renamed.")

        normalized_name = name.strip() or "Untitled Project"
        normalized_description = description.strip()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE chat_projects
                SET name = ?, description = ?
                WHERE project_id = ?
                """,
                (normalized_name, normalized_description, project_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Project not found: {project_id}")
            conn.commit()

        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        return project

    def delete_project(self, project_id: str) -> Tuple[int, int]:
        if project_id == DEFAULT_PROJECT_ID:
            raise ValueError("The default project cannot be deleted.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM chat_sessions
                WHERE project_id = ?
                """,
                (project_id,),
            )
            session_count = int(cursor.fetchone()[0])

            cursor.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id IN (
                    SELECT session_id FROM chat_sessions WHERE project_id = ?
                )
                """,
                (project_id,),
            )
            deleted_message_count = cursor.rowcount

            cursor.execute(
                """
                DELETE FROM chat_sessions
                WHERE project_id = ?
                """,
                (project_id,),
            )

            cursor.execute(
                """
                DELETE FROM chat_projects
                WHERE project_id = ?
                """,
                (project_id,),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Project not found: {project_id}")
            conn.commit()

        return session_count, deleted_message_count

    def ensure_project(self, project_id: Optional[str]) -> str:
        if not project_id:
            return DEFAULT_PROJECT_ID

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT project_id FROM chat_projects WHERE project_id = ?",
                (project_id,),
            )
            row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Project not found: {project_id}")

        return project_id

    def start_new_session(self, project_id: Optional[str] = None):
        """Create a new chat session and return its ID."""
        resolved_project_id = self.ensure_project(project_id)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        start_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_sessions (session_id, project_id, start_time, title)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, resolved_project_id, start_time, f"Chat {session_id}"),
            )
            conn.commit()
        return session_id

    def save_message(self, session_id, role, content):
        """Save a chat message to the database."""
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, timestamp),
            )
            conn.commit()

    def get_session_history(self, session_id):
        """Retrieve all messages for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                (session_id,),
            )
            return cursor.fetchall()

    def get_all_sessions(self, project_id: Optional[str] = None):
        """Retrieve all sessions, optionally scoped to a project."""
        resolved_project_id = self.ensure_project(project_id) if project_id else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if resolved_project_id:
                cursor.execute(
                    """
                    SELECT session_id, project_id, title, start_time
                    FROM chat_sessions
                    WHERE project_id = ?
                    ORDER BY start_time DESC
                    """,
                    (resolved_project_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT session_id, project_id, title, start_time
                    FROM chat_sessions
                    ORDER BY start_time DESC
                    """
                )
            return cursor.fetchall()

    def get_session(self, session_id: str) -> Optional[Tuple[str, str, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, project_id, title, start_time
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            return cursor.fetchone()

    def get_session_project_id(self, session_id: str) -> Optional[str]:
        session = self.get_session(session_id)
        return session[1] if session else None

    def update_session_title(self, session_id, title):
        """Update the title of a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                (title, session_id),
            )
            conn.commit()

    def delete_old_sessions(self, days=30):
        """Delete sessions older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id IN (
                    SELECT session_id FROM chat_sessions WHERE start_time < ?
                )
                """,
                (cutoff_str,),
            )
            cursor.execute(
                "DELETE FROM chat_sessions WHERE start_time < ?",
                (cutoff_str,),
            )
            deleted_sessions = cursor.rowcount
            conn.commit()
            return deleted_sessions

    def get_session_title_suggestion(self, session_id):
        """Generate a title suggestion based on first user message."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content
                FROM chat_messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY timestamp
                LIMIT 1
                """,
                (session_id,),
            )
            result = cursor.fetchone()
            if result:
                content = result[0]
                words = content.split()[:5]
                return " ".join(words) + ("..." if len(content.split()) > 5 else "")
            return f"Chat {session_id}"

    def _ensure_default_project(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            INSERT OR IGNORE INTO chat_projects (project_id, name, description, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                DEFAULT_PROJECT_ID,
                DEFAULT_PROJECT_NAME,
                DEFAULT_PROJECT_DESCRIPTION,
                datetime.now().isoformat(),
            ),
        )

    def _column_exists(self, cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return any(row[1] == column_name for row in cursor.fetchall())

    def _build_project_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        slug = slug or "project"
        return f"{slug}-{uuid4().hex[:8]}"


def create_history_store(db_path: str = "chat_history.db") -> HistoryStore:
    return ChatHistoryDB(db_path=db_path)

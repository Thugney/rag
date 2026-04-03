import sqlite3
from datetime import datetime, timedelta

class ChatHistoryDB:
    def __init__(self, db_path="chat_history.db"):
        """Initialize SQLite database for chat history storage"""
        self.db_path = db_path
        self.create_tables()
        
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    title TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                )
            ''')
            conn.commit()
            
    def start_new_session(self):
        """Create a new chat session and return its ID"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        start_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (session_id, start_time, title) VALUES (?, ?, ?)",
                (session_id, start_time, f"Chat {session_id}")
            )
            conn.commit()
        return session_id
        
    def save_message(self, session_id, role, content):
        """Save a chat message to the database"""
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp)
            )
            conn.commit()
            
    def get_session_history(self, session_id):
        """Retrieve all messages for a specific session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            return cursor.fetchall()
            
    def get_all_sessions(self):
        """Retrieve list of all session IDs and titles"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, title, start_time FROM chat_sessions ORDER BY start_time DESC")
            return cursor.fetchall()
            
    def update_session_title(self, session_id, title):
        """Update the title of a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                (title, session_id)
            )
            conn.commit()
            
    def delete_old_sessions(self, days=30):
        """Delete sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat(timespec='seconds')  # Trim to seconds to match potential stored format
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Log for debugging
            print(f"Attempting to delete sessions older than {cutoff_str}")
            # Check actual start times in database for comparison
            cursor.execute("SELECT session_id, start_time FROM chat_sessions")
            sessions = cursor.fetchall()
            for sid, stime in sessions:
                print(f"Session {sid} has start_time: {stime}")
            cursor.execute(
                "DELETE FROM chat_messages WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE start_time < ?)",
                (cutoff_str,)
            )
            deleted_messages = cursor.rowcount
            cursor.execute(
                "DELETE FROM chat_sessions WHERE start_time < ?",
                (cutoff_str,)
            )
            deleted_sessions = cursor.rowcount
            conn.commit()
            print(f"Deleted {deleted_messages} messages and {deleted_sessions} sessions")
            return deleted_sessions
            
    def get_session_title_suggestion(self, session_id):
        """Generate a title suggestion based on first user message"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM chat_messages WHERE session_id = ? AND role = 'user' ORDER BY timestamp LIMIT 1",
                (session_id,)
            )
            result = cursor.fetchone()
            if result:
                content = result[0]
                words = content.split()[:5]  # Take first 5 words
                return " ".join(words) + ("..." if len(content.split()) > 5 else "")
            return f"Chat {session_id}"

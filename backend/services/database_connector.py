import sqlite3
import re
from typing import Optional, Dict, List, Any

# ──────────────────────────────────────────────
# BLOCKED SQL PATTERNS (safety first)
# ──────────────────────────────────────────────

_BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|EXEC|EXECUTE|MERGE)\b",
    re.IGNORECASE,
)


class DatabaseConnector:
    """
    Lightweight, read-only database connector.
    Supports SQLite out of the box and PostgreSQL when psycopg2 is installed.
    """

    def __init__(self, db_type: str = "sqlite", db_path: str = "", connection_string: str = ""):
        """
        Parameters
        ----------
        db_type : 'sqlite' or 'postgresql'
        db_path : path to .db file (SQLite only)
        connection_string : full DSN for PostgreSQL (e.g. 'host=... dbname=... user=... password=...')
        """
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.connection_string = connection_string
        self._conn = None

    # ── connection lifecycle ──────────────────

    def connect(self):
        if self.db_type == "sqlite":
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        elif self.db_type == "postgresql":
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
            self._conn = psycopg2.connect(self.connection_string)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_connected(self):
        if self._conn is None:
            self.connect()

    # ── schema inspection ─────────────────────

    def get_tables(self) -> List[str]:
        """Returns all user table names."""
        self._ensure_connected()
        cursor = self._conn.cursor()

        if self.db_type == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        else:
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")

        return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Returns column name and type for a given table."""
        self._ensure_connected()
        cursor = self._conn.cursor()

        if self.db_type == "sqlite":
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            return [{"column": row[1], "type": row[2]} for row in cursor.fetchall()]
        else:
            cursor.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position",
                (table_name,),
            )
            return [{"column": row[0], "type": row[1]} for row in cursor.fetchall()]

    def get_full_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Returns the schema for every table in the database."""
        tables = self.get_tables()
        return {t: self.get_schema(t) for t in tables}

    # ── safe query execution ──────────────────

    @staticmethod
    def is_safe_query(sql: str) -> bool:
        """Returns True only when the query looks like a read-only SELECT."""
        stripped = sql.strip().rstrip(";").strip()
        if not stripped.upper().startswith("SELECT"):
            return False
        if _BLOCKED_KEYWORDS.search(stripped):
            return False
        return True

    def execute_safe(self, sql: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Executes a read-only SQL query with an automatic LIMIT guard.
        Returns {"columns": [...], "rows": [...], "row_count": int}.
        """
        if not self.is_safe_query(sql):
            return {"error": "Blocked: Only safe SELECT queries are allowed."}

        self._ensure_connected()

        # Auto-inject LIMIT if the user didn't provide one
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + f" LIMIT {limit}"

        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()

            if self.db_type == "sqlite":
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(zip(columns, row)) for row in rows]
            else:
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(zip(columns, row)) for row in rows]

            return {"columns": columns, "rows": data, "row_count": len(data)}
        except Exception as e:
            return {"error": str(e)}

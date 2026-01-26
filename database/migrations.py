"""
Handai Database Migrations
Schema versioning and migration system
"""

import sqlite3
from datetime import datetime
from typing import List, Tuple
import json


# Migration definitions: (version, description, sql_statements)
MIGRATIONS: List[Tuple[int, str, List[str]]] = [
    (1, "Initial schema", [
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            settings_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            run_type TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER,
            system_prompt TEXT,
            schema_json TEXT,
            variables_json TEXT,
            input_file TEXT,
            input_rows INTEGER,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            success_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            retry_count INTEGER DEFAULT 0,
            avg_latency REAL DEFAULT 0,
            total_duration REAL DEFAULT 0,
            json_mode INTEGER DEFAULT 0,
            max_concurrency INTEGER DEFAULT 5,
            auto_retry INTEGER DEFAULT 1,
            max_retry_attempts INTEGER DEFAULT 3,
            run_settings_json TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run_results (
            result_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            input_json TEXT,
            output TEXT,
            status TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT,
            latency REAL,
            retry_attempt INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS logs (
            log_id TEXT PRIMARY KEY,
            run_id TEXT,
            session_id TEXT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            details_json TEXT,
            timestamp TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)",
    ]),
    (2, "Add provider_settings table", [
        """
        CREATE TABLE IF NOT EXISTS provider_settings (
            provider TEXT NOT NULL,
            setting_key TEXT NOT NULL,
            setting_value TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (provider, setting_key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_provider_settings_provider ON provider_settings(provider)",
    ]),
    (3, "Add schema_versions table for tracking", [
        """
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """,
    ]),
]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database"""
    cursor = conn.cursor()

    # Check if schema_versions table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_versions'
    """)

    if cursor.fetchone() is None:
        # Table doesn't exist, check for legacy tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='sessions'
        """)

        if cursor.fetchone() is not None:
            # Legacy database exists, assume version 1
            return 1
        return 0

    # Get latest version
    cursor.execute("SELECT MAX(version) FROM schema_versions")
    row = cursor.fetchone()
    return row[0] if row[0] is not None else 0


def apply_migration(conn: sqlite3.Connection, version: int, description: str, statements: List[str]):
    """Apply a single migration"""
    cursor = conn.cursor()

    for statement in statements:
        try:
            cursor.execute(statement)
        except sqlite3.OperationalError as e:
            # Ignore "table already exists" or "column already exists" errors
            error_msg = str(e).lower()
            if "already exists" not in error_msg and "duplicate column" not in error_msg:
                raise

    # Record the migration
    cursor.execute("""
        INSERT OR REPLACE INTO schema_versions (version, description, applied_at)
        VALUES (?, ?, ?)
    """, (version, description, datetime.now().isoformat()))

    conn.commit()


def run_migrations(conn: sqlite3.Connection) -> List[str]:
    """Run all pending migrations and return list of applied migrations"""
    applied = []
    current_version = get_current_version(conn)

    # First, ensure schema_versions table exists (migration 3)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_versions'
    """)
    if cursor.fetchone() is None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        conn.commit()

        # Record existing versions if this is a legacy DB
        if current_version > 0:
            for version, description, _ in MIGRATIONS[:current_version]:
                cursor.execute("""
                    INSERT OR IGNORE INTO schema_versions (version, description, applied_at)
                    VALUES (?, ?, ?)
                """, (version, description, datetime.now().isoformat()))
            conn.commit()

    # Apply pending migrations
    for version, description, statements in MIGRATIONS:
        if version > current_version:
            apply_migration(conn, version, description, statements)
            applied.append(f"v{version}: {description}")

    return applied


def migrate_api_keys_to_provider_settings(conn: sqlite3.Connection):
    """
    Migrate existing single API key from app_settings to provider-specific settings.
    This handles the transition from single api_key to per-provider api_key storage.
    """
    cursor = conn.cursor()

    # Check if there's an existing api_key in app_settings
    cursor.execute("SELECT value FROM app_settings WHERE key = 'api_key'")
    row = cursor.fetchone()

    if row and row[0]:
        try:
            api_key = json.loads(row[0])
        except:
            api_key = row[0]

        if api_key and api_key != "dummy":
            # Get the provider that was being used
            cursor.execute("SELECT value FROM app_settings WHERE key = 'selected_provider'")
            provider_row = cursor.fetchone()

            provider = "OpenAI"  # Default
            if provider_row:
                try:
                    provider = json.loads(provider_row[0])
                except:
                    provider = provider_row[0]

            # Save to provider_settings if not already there
            cursor.execute("""
                SELECT 1 FROM provider_settings
                WHERE provider = ? AND setting_key = 'api_key'
            """, (provider,))

            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO provider_settings (provider, setting_key, setting_value, updated_at)
                    VALUES (?, 'api_key', ?, ?)
                """, (provider, json.dumps(api_key), datetime.now().isoformat()))
                conn.commit()
                return f"Migrated API key to {provider}"

    return None

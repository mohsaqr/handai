"""
Handai Database Module
Handles all persistence: sessions, runs, results, and logs
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import os

# ==========================================
# CONSTANTS
# ==========================================
DB_FILE = "handai_data.db"

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResultStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    RETRYING = "retrying"
    SKIPPED = "skipped"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class Session:
    session_id: str
    name: str
    mode: str  # 'transform' or 'generate'
    created_at: str
    updated_at: str
    settings_json: str

    @classmethod
    def create(cls, mode: str, settings: dict) -> 'Session':
        now = datetime.now().isoformat()
        session_id = str(uuid.uuid4())[:8]
        auto_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return cls(
            session_id=session_id,
            name=auto_name,
            mode=mode,
            created_at=now,
            updated_at=now,
            settings_json=json.dumps(settings)
        )

@dataclass
class Run:
    run_id: str
    session_id: str
    run_type: str  # 'test' or 'full'
    provider: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    schema_json: str
    variables_json: str
    input_file: str
    input_rows: int
    started_at: str
    completed_at: Optional[str]
    status: str
    success_count: int
    error_count: int
    retry_count: int
    avg_latency: float
    total_duration: float
    # Extended settings
    json_mode: bool = False
    max_concurrency: int = 5
    auto_retry: bool = True
    max_retry_attempts: int = 3
    run_settings_json: str = "{}"  # Full settings snapshot

    @classmethod
    def create(cls, session_id: str, run_type: str, provider: str, model: str,
               temperature: float, max_tokens: int, system_prompt: str,
               schema: dict, variables: dict, input_file: str, input_rows: int,
               json_mode: bool = False, max_concurrency: int = 5,
               auto_retry: bool = True, max_retry_attempts: int = 3,
               run_settings: dict = None) -> 'Run':
        return cls(
            run_id=str(uuid.uuid4())[:8],
            session_id=session_id,
            run_type=run_type,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            schema_json=json.dumps(schema),
            variables_json=json.dumps(variables),
            input_file=input_file,
            input_rows=input_rows,
            started_at=datetime.now().isoformat(),
            completed_at=None,
            status=RunStatus.RUNNING.value,
            success_count=0,
            error_count=0,
            retry_count=0,
            avg_latency=0.0,
            total_duration=0.0,
            json_mode=json_mode,
            max_concurrency=max_concurrency,
            auto_retry=auto_retry,
            max_retry_attempts=max_retry_attempts,
            run_settings_json=json.dumps(run_settings or {})
        )

@dataclass
class RunResult:
    result_id: str
    run_id: str
    row_index: int
    input_json: str
    output: str
    status: str
    error_type: Optional[str]
    error_message: Optional[str]
    latency: float
    retry_attempt: int
    created_at: str

    @classmethod
    def create(cls, run_id: str, row_index: int, input_data: Any,
               output: str, status: ResultStatus, latency: float,
               error_type: str = None, error_message: str = None,
               retry_attempt: int = 0) -> 'RunResult':
        return cls(
            result_id=str(uuid.uuid4())[:8],
            run_id=run_id,
            row_index=row_index,
            input_json=json.dumps(input_data) if not isinstance(input_data, str) else input_data,
            output=output,
            status=status.value,
            error_type=error_type,
            error_message=error_message,
            latency=latency,
            retry_attempt=retry_attempt,
            created_at=datetime.now().isoformat()
        )

@dataclass
class LogEntry:
    log_id: str
    run_id: Optional[str]
    session_id: Optional[str]
    level: str
    message: str
    details_json: str
    timestamp: str

    @classmethod
    def create(cls, level: LogLevel, message: str, details: dict = None,
               run_id: str = None, session_id: str = None) -> 'LogEntry':
        return cls(
            log_id=str(uuid.uuid4())[:8],
            run_id=run_id,
            session_id=session_id,
            level=level.value,
            message=message,
            details_json=json.dumps(details) if details else "{}",
            timestamp=datetime.now().isoformat()
        )

# ==========================================
# DATABASE MANAGER
# ==========================================

class HandaiDB:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_db(self):
        """Initialize database schema"""
        with self.get_conn() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    settings_json TEXT
                )
            ''')

            # Runs table
            cursor.execute('''
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
            ''')

            # Add new columns to existing runs table if they don't exist
            try:
                cursor.execute('ALTER TABLE runs ADD COLUMN json_mode INTEGER DEFAULT 0')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute('ALTER TABLE runs ADD COLUMN max_concurrency INTEGER DEFAULT 5')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute('ALTER TABLE runs ADD COLUMN auto_retry INTEGER DEFAULT 1')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute('ALTER TABLE runs ADD COLUMN max_retry_attempts INTEGER DEFAULT 3')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN run_settings_json TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass

            # Run results table
            cursor.execute('''
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
            ''')

            # Logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    log_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    session_id TEXT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')

            # App settings table (for persistent settings)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)')

    # ==========================================
    # SESSION OPERATIONS
    # ==========================================

    def create_session(self, mode: str, settings: dict) -> Session:
        session = Session.create(mode, settings)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, name, mode, created_at, updated_at, settings_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session.session_id, session.name, session.mode,
                  session.created_at, session.updated_at, session.settings_json))
        return session

    def update_session_name(self, session_id: str, name: str):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET name = ?, updated_at = ? WHERE session_id = ?
            ''', (name, datetime.now().isoformat(), session_id))

    def update_session_settings(self, session_id: str, settings: dict):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET settings_json = ?, updated_at = ? WHERE session_id = ?
            ''', (json.dumps(settings), datetime.now().isoformat(), session_id))

    def get_session(self, session_id: str) -> Optional[Session]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            if row:
                return Session(**dict(row))
        return None

    def get_all_sessions(self, limit: int = 100) -> List[Session]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            return [Session(**dict(row)) for row in cursor.fetchall()]

    def delete_session(self, session_id: str):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            # Delete all related data
            cursor.execute('DELETE FROM logs WHERE session_id = ?', (session_id,))
            cursor.execute('''
                DELETE FROM run_results WHERE run_id IN
                (SELECT run_id FROM runs WHERE session_id = ?)
            ''', (session_id,))
            cursor.execute('''
                DELETE FROM logs WHERE run_id IN
                (SELECT run_id FROM runs WHERE session_id = ?)
            ''', (session_id,))
            cursor.execute('DELETE FROM runs WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))

    # ==========================================
    # RUN OPERATIONS
    # ==========================================

    def create_run(self, session_id: str, run_type: str, provider: str, model: str,
                   temperature: float, max_tokens: int, system_prompt: str,
                   schema: dict, variables: dict, input_file: str, input_rows: int,
                   json_mode: bool = False, max_concurrency: int = 5,
                   auto_retry: bool = True, max_retry_attempts: int = 3,
                   run_settings: dict = None) -> Run:
        run = Run.create(session_id, run_type, provider, model, temperature,
                         max_tokens, system_prompt, schema, variables, input_file, input_rows,
                         json_mode, max_concurrency, auto_retry, max_retry_attempts, run_settings)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO runs (run_id, session_id, run_type, provider, model, temperature,
                    max_tokens, system_prompt, schema_json, variables_json, input_file,
                    input_rows, started_at, completed_at, status, success_count, error_count,
                    retry_count, avg_latency, total_duration, json_mode, max_concurrency,
                    auto_retry, max_retry_attempts, run_settings_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (run.run_id, run.session_id, run.run_type, run.provider, run.model,
                  run.temperature, run.max_tokens, run.system_prompt, run.schema_json,
                  run.variables_json, run.input_file, run.input_rows, run.started_at,
                  run.completed_at, run.status, run.success_count, run.error_count,
                  run.retry_count, run.avg_latency, run.total_duration,
                  1 if run.json_mode else 0, run.max_concurrency,
                  1 if run.auto_retry else 0, run.max_retry_attempts, run.run_settings_json))
        return run

    def update_run_status(self, run_id: str, status: RunStatus,
                          success_count: int = None, error_count: int = None,
                          retry_count: int = None, avg_latency: float = None,
                          total_duration: float = None):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            updates = ["status = ?"]
            values = [status.value]

            if status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())

            if success_count is not None:
                updates.append("success_count = ?")
                values.append(success_count)
            if error_count is not None:
                updates.append("error_count = ?")
                values.append(error_count)
            if retry_count is not None:
                updates.append("retry_count = ?")
                values.append(retry_count)
            if avg_latency is not None:
                updates.append("avg_latency = ?")
                values.append(avg_latency)
            if total_duration is not None:
                updates.append("total_duration = ?")
                values.append(total_duration)

            values.append(run_id)
            cursor.execute(f'''
                UPDATE runs SET {", ".join(updates)} WHERE run_id = ?
            ''', values)

    def get_run(self, run_id: str) -> Optional[Run]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()
            if row:
                return Run(**dict(row))
        return None

    def get_session_runs(self, session_id: str) -> List[Run]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM runs WHERE session_id = ? ORDER BY started_at DESC
            ''', (session_id,))
            return [Run(**dict(row)) for row in cursor.fetchall()]

    def get_all_runs(self, limit: int = 100, status_filter: str = None) -> List[Run]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            if status_filter:
                cursor.execute('''
                    SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT ?
                ''', (status_filter, limit))
            else:
                cursor.execute('''
                    SELECT * FROM runs ORDER BY started_at DESC LIMIT ?
                ''', (limit,))
            return [Run(**dict(row)) for row in cursor.fetchall()]

    # ==========================================
    # RESULT OPERATIONS
    # ==========================================

    def save_result(self, result: RunResult):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO run_results (result_id, run_id, row_index, input_json, output,
                    status, error_type, error_message, latency, retry_attempt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result.result_id, result.run_id, result.row_index, result.input_json,
                  result.output, result.status, result.error_type, result.error_message,
                  result.latency, result.retry_attempt, result.created_at))

    def save_results_batch(self, results: List[RunResult]):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO run_results (result_id, run_id, row_index, input_json, output,
                    status, error_type, error_message, latency, retry_attempt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(r.result_id, r.run_id, r.row_index, r.input_json, r.output,
                   r.status, r.error_type, r.error_message, r.latency,
                   r.retry_attempt, r.created_at) for r in results])

    def get_run_results(self, run_id: str) -> List[RunResult]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM run_results WHERE run_id = ? ORDER BY row_index
            ''', (run_id,))
            return [RunResult(**dict(row)) for row in cursor.fetchall()]

    def get_failed_results(self, run_id: str) -> List[RunResult]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM run_results WHERE run_id = ? AND status = ? ORDER BY row_index
            ''', (run_id, ResultStatus.ERROR.value))
            return [RunResult(**dict(row)) for row in cursor.fetchall()]

    def update_result(self, result_id: str, output: str, status: ResultStatus,
                      latency: float, retry_attempt: int,
                      error_type: str = None, error_message: str = None):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE run_results SET output = ?, status = ?, latency = ?,
                    retry_attempt = ?, error_type = ?, error_message = ?
                WHERE result_id = ?
            ''', (output, status.value, latency, retry_attempt,
                  error_type, error_message, result_id))

    # ==========================================
    # LOG OPERATIONS
    # ==========================================

    def log(self, level: LogLevel, message: str, details: dict = None,
            run_id: str = None, session_id: str = None):
        entry = LogEntry.create(level, message, details, run_id, session_id)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (log_id, run_id, session_id, level, message, details_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entry.log_id, entry.run_id, entry.session_id, entry.level,
                  entry.message, entry.details_json, entry.timestamp))

    def get_run_logs(self, run_id: str) -> List[LogEntry]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM logs WHERE run_id = ? ORDER BY timestamp DESC
            ''', (run_id,))
            return [LogEntry(**dict(row)) for row in cursor.fetchall()]

    def get_session_logs(self, session_id: str) -> List[LogEntry]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM logs WHERE session_id = ? ORDER BY timestamp DESC
            ''', (session_id,))
            return [LogEntry(**dict(row)) for row in cursor.fetchall()]

    def get_recent_logs(self, limit: int = 100, level_filter: str = None) -> List[LogEntry]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            if level_filter:
                cursor.execute('''
                    SELECT * FROM logs WHERE level = ? ORDER BY timestamp DESC LIMIT ?
                ''', (level_filter, limit))
            else:
                cursor.execute('''
                    SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
            return [LogEntry(**dict(row)) for row in cursor.fetchall()]

    # ==========================================
    # STATISTICS
    # ==========================================

    def get_session_stats(self, session_id: str) -> dict:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(success_count) as total_success,
                    SUM(error_count) as total_errors,
                    AVG(avg_latency) as avg_latency,
                    SUM(total_duration) as total_time
                FROM runs WHERE session_id = ?
            ''', (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_global_stats(self) -> dict:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(*) as total_runs,
                    SUM(success_count) as total_success,
                    SUM(error_count) as total_errors,
                    SUM(input_rows) as total_rows_processed
                FROM runs
            ''')
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_provider_stats(self) -> List[dict]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    provider,
                    COUNT(*) as runs,
                    SUM(success_count) as successes,
                    SUM(error_count) as errors,
                    AVG(avg_latency) as avg_latency
                FROM runs
                GROUP BY provider
                ORDER BY runs DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # APP SETTINGS (Persistent)
    # ==========================================

    def save_setting(self, key: str, value: Any):
        """Save a single setting to the database."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now().isoformat()))

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a single setting from the database."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM app_settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except:
                    return row[0]
            return default

    def save_all_settings(self, settings: Dict[str, Any]):
        """Save multiple settings at once."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            for key, value in settings.items():
                if value is not None:  # Don't save None values
                    cursor.execute('''
                        INSERT OR REPLACE INTO app_settings (key, value, updated_at)
                        VALUES (?, ?, ?)
                    ''', (key, json.dumps(value), now))

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings from the database."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM app_settings')
            settings = {}
            for row in cursor.fetchall():
                try:
                    settings[row[0]] = json.loads(row[1])
                except:
                    settings[row[0]] = row[1]
            return settings

    def delete_setting(self, key: str):
        """Delete a setting from the database."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM app_settings WHERE key = ?', (key,))


# Singleton instance
_db_instance = None

def get_db() -> HandaiDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = HandaiDB()
    return _db_instance

"""
Handai Database Manager
Handles all persistence: sessions, runs, results, logs, and provider settings
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import os

from .models import (
    Session, Run, RunResult, LogEntry, ProviderSetting, ConfiguredProvider,
    RunStatus, ResultStatus, LogLevel
)
from .migrations import run_migrations, migrate_api_keys_to_provider_settings
from config import DB_FILE


class HandaiDB:
    """Database manager for Handai application"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_FILE
        self._init_db()

    @contextmanager
    def get_conn(self):
        """Get a database connection with automatic cleanup"""
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

    def _init_db(self):
        """Initialize database schema using migrations"""
        with self.get_conn() as conn:
            # Run migrations
            applied = run_migrations(conn)

            # Migrate existing API keys if needed
            migration_result = migrate_api_keys_to_provider_settings(conn)

    # ==========================================
    # SESSION OPERATIONS
    # ==========================================

    def create_session(self, mode: str, settings: dict) -> Session:
        """Create a new session"""
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
        """Update session name"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET name = ?, updated_at = ? WHERE session_id = ?
            ''', (name, datetime.now().isoformat(), session_id))

    def update_session_settings(self, session_id: str, settings: dict):
        """Update session settings"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET settings_json = ?, updated_at = ? WHERE session_id = ?
            ''', (json.dumps(settings), datetime.now().isoformat(), session_id))

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            if row:
                return Session(**dict(row))
        return None

    def get_all_sessions(self, limit: int = 100) -> List[Session]:
        """Get all sessions ordered by creation date"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            return [Session(**dict(row)) for row in cursor.fetchall()]

    def delete_session(self, session_id: str):
        """Delete a session and all related data"""
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
        """Create a new run"""
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
        """Update run status and statistics"""
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
        """Get a run by ID"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()
            if row:
                return Run(**dict(row))
        return None

    def get_session_runs(self, session_id: str) -> List[Run]:
        """Get all runs for a session"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM runs WHERE session_id = ? ORDER BY started_at DESC
            ''', (session_id,))
            return [Run(**dict(row)) for row in cursor.fetchall()]

    def get_all_runs(self, limit: int = 100, status_filter: str = None) -> List[Run]:
        """Get all runs with optional status filter"""
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
        """Save a single result"""
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
        """Save multiple results in a batch"""
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
        """Get all results for a run"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM run_results WHERE run_id = ? ORDER BY row_index
            ''', (run_id,))
            return [RunResult(**dict(row)) for row in cursor.fetchall()]

    def get_failed_results(self, run_id: str) -> List[RunResult]:
        """Get failed results for a run"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM run_results WHERE run_id = ? AND status = ? ORDER BY row_index
            ''', (run_id, ResultStatus.ERROR.value))
            return [RunResult(**dict(row)) for row in cursor.fetchall()]

    def update_result(self, result_id: str, output: str, status: ResultStatus,
                      latency: float, retry_attempt: int,
                      error_type: str = None, error_message: str = None):
        """Update a result"""
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
        """Create a log entry"""
        entry = LogEntry.create(level, message, details, run_id, session_id)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (log_id, run_id, session_id, level, message, details_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entry.log_id, entry.run_id, entry.session_id, entry.level,
                  entry.message, entry.details_json, entry.timestamp))

    def get_run_logs(self, run_id: str) -> List[LogEntry]:
        """Get logs for a run"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM logs WHERE run_id = ? ORDER BY timestamp DESC
            ''', (run_id,))
            return [LogEntry(**dict(row)) for row in cursor.fetchall()]

    def get_session_logs(self, session_id: str) -> List[LogEntry]:
        """Get logs for a session"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM logs WHERE session_id = ? ORDER BY timestamp DESC
            ''', (session_id,))
            return [LogEntry(**dict(row)) for row in cursor.fetchall()]

    def get_recent_logs(self, limit: int = 100, level_filter: str = None) -> List[LogEntry]:
        """Get recent logs with optional level filter"""
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
        """Get statistics for a session"""
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
        """Get global statistics"""
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
        """Get statistics grouped by provider"""
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
    # APP SETTINGS (General)
    # ==========================================

    def save_setting(self, key: str, value: Any):
        """Save a single setting to the database"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now().isoformat()))

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a single setting from the database"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM app_settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    return row[0]
            return default

    def save_all_settings(self, settings: Dict[str, Any]):
        """Save multiple settings at once"""
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
        """Get all settings from the database"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM app_settings')
            settings = {}
            for row in cursor.fetchall():
                try:
                    settings[row[0]] = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    settings[row[0]] = row[1]
            return settings

    def delete_setting(self, key: str):
        """Delete a setting from the database"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM app_settings WHERE key = ?', (key,))

    # ==========================================
    # PROVIDER SETTINGS (Per-Provider)
    # ==========================================

    def save_provider_setting(self, provider: str, key: str, value: Any):
        """Save a provider-specific setting"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO provider_settings (provider, setting_key, setting_value, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (provider, key, json.dumps(value), datetime.now().isoformat()))

    def get_provider_setting(self, provider: str, key: str, default: Any = None) -> Any:
        """Get a provider-specific setting"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT setting_value FROM provider_settings
                WHERE provider = ? AND setting_key = ?
            ''', (provider, key))
            row = cursor.fetchone()
            if row and row[0] is not None:
                try:
                    return json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    return row[0]
            return default

    def get_all_provider_settings(self, provider: str) -> Dict[str, Any]:
        """Get all settings for a provider"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT setting_key, setting_value FROM provider_settings
                WHERE provider = ?
            ''', (provider,))
            settings = {}
            for row in cursor.fetchall():
                try:
                    settings[row[0]] = json.loads(row[1]) if row[1] else None
                except (json.JSONDecodeError, TypeError):
                    settings[row[0]] = row[1]
            return settings

    def delete_provider_setting(self, provider: str, key: str):
        """Delete a provider-specific setting"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM provider_settings
                WHERE provider = ? AND setting_key = ?
            ''', (provider, key))

    def get_all_providers_with_api_keys(self) -> Dict[str, bool]:
        """Get all providers and whether they have API keys set"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT provider FROM provider_settings
                WHERE setting_key = 'api_key' AND setting_value IS NOT NULL
            ''')
            return {row[0]: True for row in cursor.fetchall()}

    # ==========================================
    # CONFIGURED PROVIDERS
    # ==========================================

    def save_configured_provider(self, provider: ConfiguredProvider):
        """Insert or replace a configured provider"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO configured_providers
                (id, provider_type, display_name, base_url, api_key, default_model,
                 is_enabled, temperature, max_tokens, top_p, frequency_penalty,
                 request_timeout, max_retries, capabilities, total_requests, total_tokens,
                 last_tested, last_test_status, last_test_latency, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (provider.id, provider.provider_type, provider.display_name,
                  provider.base_url, provider.api_key, provider.default_model,
                  1 if provider.is_enabled else 0, provider.temperature, provider.max_tokens,
                  provider.top_p, provider.frequency_penalty, provider.request_timeout,
                  provider.max_retries, provider.capabilities, provider.total_requests,
                  provider.total_tokens, provider.last_tested, provider.last_test_status,
                  provider.last_test_latency, provider.created_at, provider.updated_at))

    def get_configured_provider(self, provider_id: str) -> Optional[ConfiguredProvider]:
        """Get a configured provider by ID"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configured_providers WHERE id = ?', (provider_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['is_enabled'] = bool(d.get('is_enabled', 0))
                return ConfiguredProvider(**d)
        return None

    def get_all_configured_providers(self) -> List[ConfiguredProvider]:
        """Get all configured providers"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configured_providers ORDER BY display_name')
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d['is_enabled'] = bool(d.get('is_enabled', 0))
                results.append(ConfiguredProvider(**d))
            return results

    def get_enabled_configured_providers(self) -> List[ConfiguredProvider]:
        """Get only enabled configured providers"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configured_providers WHERE is_enabled = 1 ORDER BY display_name')
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d['is_enabled'] = True
                results.append(ConfiguredProvider(**d))
            return results

    def update_configured_provider(self, provider_id: str, **kwargs):
        """Update specific fields of a configured provider"""
        kwargs['updated_at'] = datetime.now().isoformat()
        if 'is_enabled' in kwargs:
            kwargs['is_enabled'] = 1 if kwargs['is_enabled'] else 0
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [provider_id]
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE configured_providers SET {sets} WHERE id = ?', vals)

    def delete_configured_provider(self, provider_id: str):
        """Delete a configured provider"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM configured_providers WHERE id = ?', (provider_id,))

    def get_configured_provider_by_type(self, provider_type: str) -> Optional[ConfiguredProvider]:
        """Get a configured provider by its type name"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configured_providers WHERE provider_type = ?', (provider_type,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['is_enabled'] = bool(d.get('is_enabled', 0))
                return ConfiguredProvider(**d)
        return None


# Singleton instance
_db_instance = None


def get_db() -> HandaiDB:
    """Get the singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = HandaiDB()
    return _db_instance

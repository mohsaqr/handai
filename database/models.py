"""
Handai Database Models
Data classes for database entities
"""

import json
import uuid
from datetime import datetime
from typing import Optional, Any, List
from dataclasses import dataclass, asdict, field
from enum import Enum


class RunStatus(Enum):
    """Status of a processing run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResultStatus(Enum):
    """Status of an individual result"""
    SUCCESS = "success"
    ERROR = "error"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Session:
    """A processing session containing multiple runs"""
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

    def get_settings(self) -> dict:
        """Parse and return settings as dict"""
        try:
            return json.loads(self.settings_json) if self.settings_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class Run:
    """A single processing run within a session"""
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

    def get_schema(self) -> dict:
        """Parse and return schema as dict"""
        try:
            return json.loads(self.schema_json) if self.schema_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_variables(self) -> dict:
        """Parse and return variables as dict"""
        try:
            return json.loads(self.variables_json) if self.variables_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_run_settings(self) -> dict:
        """Parse and return run settings as dict"""
        try:
            return json.loads(self.run_settings_json) if self.run_settings_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class RunResult:
    """Result of processing a single row"""
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

    def get_input(self) -> Any:
        """Parse and return input data"""
        try:
            return json.loads(self.input_json) if self.input_json else None
        except (json.JSONDecodeError, TypeError):
            return self.input_json


@dataclass
class LogEntry:
    """A log entry for tracking operations"""
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

    def get_details(self) -> dict:
        """Parse and return details as dict"""
        try:
            return json.loads(self.details_json) if self.details_json else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ProviderSetting:
    """A provider-specific setting (API key, default model, etc.)"""
    provider: str
    setting_key: str
    setting_value: Optional[str]
    updated_at: str

    @classmethod
    def create(cls, provider: str, key: str, value: Any) -> 'ProviderSetting':
        return cls(
            provider=provider,
            setting_key=key,
            setting_value=json.dumps(value) if not isinstance(value, str) else value,
            updated_at=datetime.now().isoformat()
        )

    def get_value(self) -> Any:
        """Parse and return value"""
        if self.setting_value is None:
            return None
        try:
            return json.loads(self.setting_value)
        except (json.JSONDecodeError, TypeError):
            return self.setting_value


@dataclass
class ConfiguredProvider:
    """A fully configured LLM provider instance"""
    id: str
    provider_type: str
    display_name: str
    base_url: Optional[str]
    api_key: Optional[str]
    default_model: str
    is_enabled: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    request_timeout: int = 120000
    max_retries: int = 3
    capabilities: str = "[]"
    total_requests: int = 0
    total_tokens: int = 0
    last_tested: Optional[str] = None
    last_test_status: Optional[str] = None
    last_test_latency: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, provider_type: str, display_name: str, default_model: str,
               base_url: str = None, api_key: str = None,
               temperature: float = 0.7, max_tokens: int = 2048,
               is_enabled: bool = False, capabilities: List[str] = None) -> 'ConfiguredProvider':
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4())[:8],
            provider_type=provider_type,
            display_name=display_name,
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
            is_enabled=is_enabled,
            temperature=temperature,
            max_tokens=max_tokens,
            capabilities=json.dumps(capabilities or []),
            created_at=now,
            updated_at=now,
        )

    def get_capabilities(self) -> List[str]:
        try:
            return json.loads(self.capabilities) if self.capabilities else []
        except (json.JSONDecodeError, TypeError):
            return []

"""
Tests for Database Models
"""

import pytest
import json
from database.models import (
    Session, Run, RunResult, LogEntry, ProviderSetting, ConfiguredProvider,
    RunStatus, ResultStatus, LogLevel
)


class TestSession:
    """Tests for Session model"""

    def test_create_session(self):
        """Session.create should generate valid session"""
        settings = {"key": "value", "number": 42}
        session = Session.create("transform", settings)

        assert session.session_id is not None
        assert len(session.session_id) == 8
        assert session.mode == "transform"
        assert "Session_" in session.name
        assert session.created_at is not None
        assert session.updated_at is not None

    def test_get_settings(self):
        """get_settings should parse JSON correctly"""
        settings = {"provider": "OpenAI", "model": "gpt-4o"}
        session = Session.create("generate", settings)

        retrieved = session.get_settings()
        assert retrieved == settings

    def test_get_settings_empty(self):
        """get_settings should handle empty/None settings"""
        session = Session(
            session_id="test123",
            name="Test",
            mode="transform",
            created_at="2024-01-01",
            updated_at="2024-01-01",
            settings_json=None
        )
        assert session.get_settings() == {}

    def test_get_settings_invalid_json(self):
        """get_settings should handle invalid JSON"""
        session = Session(
            session_id="test123",
            name="Test",
            mode="transform",
            created_at="2024-01-01",
            updated_at="2024-01-01",
            settings_json="not valid json"
        )
        assert session.get_settings() == {}


class TestRun:
    """Tests for Run model"""

    def test_create_run(self):
        """Run.create should generate valid run"""
        run = Run.create(
            session_id="sess123",
            run_type="test",
            provider="OpenAI",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="Analyze this data",
            schema={"field": "type"},
            variables={"var1": "value1"},
            input_file="data.csv",
            input_rows=100
        )

        assert run.run_id is not None
        assert len(run.run_id) == 8
        assert run.session_id == "sess123"
        assert run.run_type == "test"
        assert run.provider == "OpenAI"
        assert run.model == "gpt-4o"
        assert run.status == RunStatus.RUNNING.value

    def test_run_with_extended_settings(self):
        """Run should support extended settings"""
        run = Run.create(
            session_id="sess123",
            run_type="full",
            provider="OpenAI",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=4096,
            system_prompt="Test",
            schema={},
            variables={},
            input_file="test.csv",
            input_rows=50,
            json_mode=True,
            max_concurrency=10,
            auto_retry=False,
            max_retry_attempts=5
        )

        assert run.json_mode is True
        assert run.max_concurrency == 10
        assert run.auto_retry is False
        assert run.max_retry_attempts == 5

    def test_get_schema(self):
        """get_schema should parse JSON correctly"""
        run = Run.create(
            session_id="sess",
            run_type="test",
            provider="OpenAI",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=1000,
            system_prompt="Test",
            schema={"name": "string", "age": "number"},
            variables={},
            input_file="test.csv",
            input_rows=10
        )

        schema = run.get_schema()
        assert schema == {"name": "string", "age": "number"}

    def test_get_variables(self):
        """get_variables should parse JSON correctly"""
        run = Run.create(
            session_id="sess",
            run_type="test",
            provider="OpenAI",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=1000,
            system_prompt="Test",
            schema={},
            variables={"topic": ["AI", "ML"], "style": "formal"},
            input_file="test.csv",
            input_rows=10
        )

        variables = run.get_variables()
        assert variables["topic"] == ["AI", "ML"]
        assert variables["style"] == "formal"


class TestRunResult:
    """Tests for RunResult model"""

    def test_create_success_result(self):
        """Create successful result"""
        result = RunResult.create(
            run_id="run123",
            row_index=5,
            input_data={"text": "sample input"},
            output="processed output",
            status=ResultStatus.SUCCESS,
            latency=1.5
        )

        assert result.result_id is not None
        assert result.run_id == "run123"
        assert result.row_index == 5
        assert result.output == "processed output"
        assert result.status == ResultStatus.SUCCESS.value
        assert result.latency == 1.5
        assert result.error_type is None

    def test_create_error_result(self):
        """Create error result"""
        result = RunResult.create(
            run_id="run123",
            row_index=10,
            input_data={"text": "bad input"},
            output="",
            status=ResultStatus.ERROR,
            latency=0.5,
            error_type="rate_limit",
            error_message="Rate limit exceeded",
            retry_attempt=2
        )

        assert result.status == ResultStatus.ERROR.value
        assert result.error_type == "rate_limit"
        assert result.error_message == "Rate limit exceeded"
        assert result.retry_attempt == 2

    def test_get_input_dict(self):
        """get_input should parse dict input"""
        result = RunResult.create(
            run_id="run123",
            row_index=0,
            input_data={"key": "value", "num": 42},
            output="output",
            status=ResultStatus.SUCCESS,
            latency=1.0
        )

        input_data = result.get_input()
        assert input_data == {"key": "value", "num": 42}

    def test_get_input_string(self):
        """get_input should handle string input"""
        result = RunResult(
            result_id="res123",
            run_id="run123",
            row_index=0,
            input_json="plain string",
            output="output",
            status="success",
            error_type=None,
            error_message=None,
            latency=1.0,
            retry_attempt=0,
            created_at="2024-01-01"
        )

        input_data = result.get_input()
        assert input_data == "plain string"


class TestLogEntry:
    """Tests for LogEntry model"""

    def test_create_log_entry(self):
        """Create log entry"""
        entry = LogEntry.create(
            level=LogLevel.INFO,
            message="Processing started",
            details={"rows": 100, "model": "gpt-4o"},
            run_id="run123",
            session_id="sess123"
        )

        assert entry.log_id is not None
        assert entry.level == LogLevel.INFO.value
        assert entry.message == "Processing started"
        assert entry.run_id == "run123"
        assert entry.session_id == "sess123"

    def test_get_details(self):
        """get_details should parse JSON"""
        entry = LogEntry.create(
            level=LogLevel.WARNING,
            message="Retry triggered",
            details={"attempt": 2, "reason": "rate_limit"}
        )

        details = entry.get_details()
        assert details["attempt"] == 2
        assert details["reason"] == "rate_limit"

    def test_log_levels(self):
        """All log levels should work"""
        for level in LogLevel:
            entry = LogEntry.create(level=level, message=f"Test {level.value}")
            assert entry.level == level.value


class TestProviderSetting:
    """Tests for ProviderSetting model"""

    def test_create_string_setting(self):
        """Create string setting"""
        setting = ProviderSetting.create(
            provider="openai",
            key="api_key",
            value="sk-test123"
        )

        assert setting.provider == "openai"
        assert setting.setting_key == "api_key"
        assert setting.get_value() == "sk-test123"

    def test_create_dict_setting(self):
        """Create dict setting"""
        setting = ProviderSetting.create(
            provider="openai",
            key="config",
            value={"timeout": 30, "retries": 3}
        )

        value = setting.get_value()
        assert value == {"timeout": 30, "retries": 3}

    def test_get_value_none(self):
        """get_value should handle None"""
        setting = ProviderSetting(
            provider="test",
            setting_key="key",
            setting_value=None,
            updated_at="2024-01-01"
        )
        assert setting.get_value() is None


class TestConfiguredProvider:
    """Tests for ConfiguredProvider model"""

    def test_create_configured_provider(self):
        """Create configured provider"""
        provider = ConfiguredProvider.create(
            provider_type="openai",
            display_name="My OpenAI",
            default_model="gpt-4o",
            api_key="sk-test",
            is_enabled=True,
            capabilities=["chat", "json_mode"]
        )

        assert provider.id is not None
        assert provider.provider_type == "openai"
        assert provider.display_name == "My OpenAI"
        assert provider.default_model == "gpt-4o"
        assert provider.is_enabled is True

    def test_get_capabilities(self):
        """get_capabilities should parse JSON list"""
        provider = ConfiguredProvider.create(
            provider_type="openai",
            display_name="Test",
            default_model="gpt-4o",
            capabilities=["chat", "completion", "embedding"]
        )

        caps = provider.get_capabilities()
        assert "chat" in caps
        assert "completion" in caps
        assert "embedding" in caps

    def test_default_values(self):
        """Default values should be set correctly"""
        provider = ConfiguredProvider.create(
            provider_type="custom",
            display_name="Custom Provider",
            default_model="custom-model"
        )

        assert provider.temperature == 0.7
        assert provider.max_tokens == 2048
        assert provider.top_p == 1.0
        assert provider.is_enabled is False
        assert provider.total_requests == 0


class TestEnums:
    """Tests for enum values"""

    def test_run_status_values(self):
        """RunStatus should have expected values"""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"

    def test_result_status_values(self):
        """ResultStatus should have expected values"""
        assert ResultStatus.SUCCESS.value == "success"
        assert ResultStatus.ERROR.value == "error"
        assert ResultStatus.RETRYING.value == "retrying"
        assert ResultStatus.SKIPPED.value == "skipped"

    def test_log_level_values(self):
        """LogLevel should have expected values"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"

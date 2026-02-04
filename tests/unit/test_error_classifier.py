"""
Tests for Error Classification Module
"""

import pytest
from errors.classifier import (
    ErrorClassifier, ErrorType, ErrorInfo,
    format_error_for_display, validate_json_output
)


class TestErrorClassification:
    """Tests for error type classification"""

    def test_classify_auth_error_401(self):
        """Test 401 unauthorized error classification"""
        error = Exception("Error code: 401 Unauthorized")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.AUTH_ERROR
        assert not result.is_retryable

    def test_classify_invalid_api_key(self):
        """Test invalid API key error classification"""
        error = Exception("Invalid API key provided")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.INVALID_API_KEY
        assert not result.is_retryable

    def test_classify_rate_limit_429(self):
        """Test 429 rate limit error classification"""
        error = Exception("Error 429: rate_limit reached")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.RATE_LIMIT
        assert result.is_retryable
        assert result.retry_delay > 0

    def test_classify_too_many_requests(self):
        """Test too many requests error"""
        error = Exception("Too many requests")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.TOO_MANY_REQUESTS
        assert result.is_retryable

    def test_classify_connection_error(self):
        """Test connection error classification"""
        error = Exception("Connection refused")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.CONNECTION_ERROR
        assert result.is_retryable

    def test_classify_timeout_error(self):
        """Test timeout error classification"""
        error = Exception("Request timed out after 30 seconds")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.TIMEOUT
        assert result.is_retryable

    def test_classify_dns_error(self):
        """Test DNS resolution error"""
        error = Exception("getaddrinfo failed: Name or service not known")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.DNS_ERROR
        assert result.is_retryable

    def test_classify_ssl_error(self):
        """Test SSL/TLS error"""
        error = Exception("SSL certificate verify failed")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.SSL_ERROR
        assert result.is_retryable

    def test_classify_model_not_found(self):
        """Test model not found error (404)"""
        error = Exception("Model 'gpt-5' not found. Error 404")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.MODEL_NOT_FOUND
        assert not result.is_retryable

    def test_classify_service_unavailable(self):
        """Test 503 service unavailable"""
        error = Exception("Service unavailable. Error 503")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.SERVICE_UNAVAILABLE
        assert result.is_retryable

    def test_classify_bad_gateway(self):
        """Test 502 bad gateway"""
        error = Exception("Bad gateway. Error 502")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.BAD_GATEWAY
        assert result.is_retryable

    def test_classify_server_error(self):
        """Test 500 internal server error"""
        error = Exception("Internal server error. Error 500")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.SERVER_ERROR
        assert result.is_retryable

    def test_classify_content_filter(self):
        """Test content filter error"""
        error = Exception("Content blocked by safety filter")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.CONTENT_FILTER
        assert not result.is_retryable

    def test_classify_context_length_exceeded(self):
        """Test context length exceeded error"""
        error = Exception("Maximum context length is 4096 tokens, input too long")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.CONTEXT_LENGTH_EXCEEDED
        assert not result.is_retryable

    def test_classify_quota_exceeded(self):
        """Test quota exceeded error"""
        error = Exception("Quota exceeded. insufficient_quota")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.INSUFFICIENT_QUOTA
        assert not result.is_retryable

    def test_classify_unknown_error(self):
        """Test unknown error fallback"""
        error = Exception("Some random error message")
        result = ErrorClassifier.classify(error)
        assert result.error_type == ErrorType.UNKNOWN


class TestEmptyResultDetection:
    """Tests for empty result detection"""

    def test_is_empty_none(self):
        """None should be empty"""
        assert ErrorClassifier.is_empty_result(None) is True

    def test_is_empty_empty_string(self):
        """Empty string should be empty"""
        assert ErrorClassifier.is_empty_result("") is True
        assert ErrorClassifier.is_empty_result("   ") is True

    def test_is_empty_na_values(self):
        """N/A values should be empty"""
        na_values = ['N/A', 'n/a', 'NA', 'null', 'None', 'undefined', 'empty']
        for val in na_values:
            assert ErrorClassifier.is_empty_result(val) is True, f"'{val}' should be empty"

    def test_is_empty_json_empty(self):
        """Empty JSON should be empty"""
        assert ErrorClassifier.is_empty_result("{}") is True
        assert ErrorClassifier.is_empty_result("[]") is True

    def test_is_empty_error_prefix(self):
        """Error prefixed strings should be empty"""
        assert ErrorClassifier.is_empty_result("error: something went wrong") is True

    def test_is_empty_very_short(self):
        """Very short responses should be empty"""
        assert ErrorClassifier.is_empty_result("ab") is True
        assert ErrorClassifier.is_empty_result("x") is True

    def test_is_not_empty_valid_response(self):
        """Valid responses should not be empty"""
        assert ErrorClassifier.is_empty_result("This is a valid response") is False
        assert ErrorClassifier.is_empty_result('{"key": "value"}') is False
        assert ErrorClassifier.is_empty_result("yes") is False
        assert ErrorClassifier.is_empty_result("123") is False


class TestAutoRetry:
    """Tests for auto-retry logic"""

    def test_should_retry_rate_limit(self):
        """Rate limit errors should be retried"""
        error_info = ErrorInfo(
            error_type=ErrorType.RATE_LIMIT,
            message="Rate limit",
            suggestion="Wait",
            is_retryable=True,
            retry_delay=30,
            original_error="429"
        )
        should_retry, delay = ErrorClassifier.should_auto_retry(error_info, attempt=0)
        assert should_retry is True
        assert delay == 30

    def test_should_not_retry_auth_error(self):
        """Auth errors should not be retried"""
        error_info = ErrorInfo(
            error_type=ErrorType.AUTH_ERROR,
            message="Auth failed",
            suggestion="Check key",
            is_retryable=False,
            retry_delay=0,
            original_error="401"
        )
        should_retry, _ = ErrorClassifier.should_auto_retry(error_info, attempt=0)
        assert should_retry is False

    def test_should_not_retry_max_attempts(self):
        """Should stop retrying after max attempts"""
        error_info = ErrorInfo(
            error_type=ErrorType.RATE_LIMIT,
            message="Rate limit",
            suggestion="Wait",
            is_retryable=True,
            retry_delay=30,
            original_error="429"
        )
        should_retry, _ = ErrorClassifier.should_auto_retry(error_info, attempt=3, max_retries=3)
        assert should_retry is False

    def test_exponential_backoff(self):
        """Delay should increase exponentially"""
        error_info = ErrorInfo(
            error_type=ErrorType.CONNECTION_ERROR,
            message="Connection failed",
            suggestion="Retry",
            is_retryable=True,
            retry_delay=5,
            original_error="Connection refused"
        )

        _, delay0 = ErrorClassifier.should_auto_retry(error_info, attempt=0)
        _, delay1 = ErrorClassifier.should_auto_retry(error_info, attempt=1)
        _, delay2 = ErrorClassifier.should_auto_retry(error_info, attempt=2)

        assert delay0 == 5   # 5 * 2^0
        assert delay1 == 10  # 5 * 2^1
        assert delay2 == 20  # 5 * 2^2

    def test_max_delay_cap(self):
        """Delay should be capped at 120 seconds"""
        error_info = ErrorInfo(
            error_type=ErrorType.SERVICE_UNAVAILABLE,
            message="Service down",
            suggestion="Wait",
            is_retryable=True,
            retry_delay=60,
            original_error="503"
        )
        _, delay = ErrorClassifier.should_auto_retry(error_info, attempt=5)
        assert delay <= 120


class TestJsonValidation:
    """Tests for JSON output validation"""

    def test_valid_json(self):
        """Valid JSON should pass"""
        is_valid, data, error = validate_json_output('{"key": "value"}')
        assert is_valid is True
        assert data == {"key": "value"}
        assert error is None

    def test_empty_input(self):
        """Empty input should fail"""
        is_valid, data, error = validate_json_output("")
        assert is_valid is False
        assert error == "Empty response"

    def test_json_with_markdown_code_block(self):
        """JSON in markdown code block should be extracted"""
        content = """```json
{"result": "success"}
```"""
        is_valid, data, error = validate_json_output(content)
        assert is_valid is True
        assert data == {"result": "success"}

    def test_json_extraction_from_text(self):
        """JSON should be extracted from surrounding text"""
        content = 'Here is the result: {"value": 42} hope this helps!'
        is_valid, data, error = validate_json_output(content)
        assert is_valid is True
        assert data == {"value": 42}

    def test_invalid_json(self):
        """Invalid JSON should fail gracefully"""
        is_valid, data, error = validate_json_output("not json at all")
        assert is_valid is False
        assert "JSON parse error" in error

    def test_schema_validation_pass(self):
        """Schema validation should pass when fields present"""
        schema = {"name": str, "age": int}
        is_valid, data, error = validate_json_output(
            '{"name": "Alice", "age": 30}',
            expected_schema=schema
        )
        assert is_valid is True
        assert error is None

    def test_schema_validation_missing_fields(self):
        """Schema validation should fail when fields missing"""
        schema = {"name": str, "age": int, "email": str}
        is_valid, data, error = validate_json_output(
            '{"name": "Alice"}',
            expected_schema=schema
        )
        assert is_valid is False
        assert "Missing fields" in error


class TestErrorFormatting:
    """Tests for error display formatting"""

    def test_format_basic_error(self):
        """Basic error formatting"""
        error_info = ErrorInfo(
            error_type=ErrorType.AUTH_ERROR,
            message="Authentication failed",
            suggestion="Check your API key",
            is_retryable=False,
            retry_delay=0,
            original_error="401 Unauthorized"
        )
        formatted = format_error_for_display(error_info)
        assert "Authentication failed" in formatted
        assert "Check your API key" in formatted

    def test_format_retryable_error(self):
        """Retryable error should show retry info"""
        error_info = ErrorInfo(
            error_type=ErrorType.RATE_LIMIT,
            message="Rate limit",
            suggestion="Wait and retry",
            is_retryable=True,
            retry_delay=30,
            original_error="429"
        )
        formatted = format_error_for_display(error_info)
        assert "retryable" in formatted.lower()

    def test_format_with_original_error(self):
        """Should show original error when requested"""
        error_info = ErrorInfo(
            error_type=ErrorType.UNKNOWN,
            message="Unknown",
            suggestion="Check logs",
            is_retryable=False,
            retry_delay=0,
            original_error="SomeException: detailed message"
        )
        formatted = format_error_for_display(error_info, show_original=True)
        assert "SomeException" in formatted

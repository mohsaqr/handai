"""
Handai Error Handling Module
Classifies errors and provides actionable suggestions
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import re
import traceback

# ==========================================
# ERROR TYPES
# ==========================================

class ErrorType(Enum):
    # Authentication & Authorization
    AUTH_ERROR = "auth_error"
    INVALID_API_KEY = "invalid_api_key"
    INSUFFICIENT_QUOTA = "insufficient_quota"

    # Rate Limiting
    RATE_LIMIT = "rate_limit"
    TOO_MANY_REQUESTS = "too_many_requests"

    # Network & Connection
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"

    # Model & Provider
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_ERROR = "provider_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Content & Input
    CONTENT_FILTER = "content_filter"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"
    JSON_PARSE_ERROR = "json_parse_error"

    # Server Errors
    SERVER_ERROR = "server_error"
    BAD_GATEWAY = "bad_gateway"

    # Unknown
    UNKNOWN = "unknown"

# ==========================================
# ERROR INFO
# ==========================================

@dataclass
class ErrorInfo:
    error_type: ErrorType
    message: str
    suggestion: str
    is_retryable: bool
    retry_delay: int  # seconds, 0 if not retryable
    original_error: str

ERROR_SUGGESTIONS = {
    ErrorType.AUTH_ERROR: (
        "Authentication failed",
        "Check your API key in the sidebar. Make sure it's valid and has proper permissions.",
        False, 0
    ),
    ErrorType.INVALID_API_KEY: (
        "Invalid API key",
        "The API key is incorrect or expired. Please verify your credentials.",
        False, 0
    ),
    ErrorType.INSUFFICIENT_QUOTA: (
        "Quota exceeded",
        "Your API quota is exhausted. Check your billing settings or wait for quota reset.",
        False, 0
    ),
    ErrorType.RATE_LIMIT: (
        "Rate limit reached",
        "Too many requests. Reduce 'Max Concurrent Requests' in settings or wait before retrying.",
        True, 30
    ),
    ErrorType.TOO_MANY_REQUESTS: (
        "Too many requests",
        "Server is overloaded. Lower concurrency and try again in a moment.",
        True, 60
    ),
    ErrorType.CONNECTION_ERROR: (
        "Connection failed",
        "Could not connect to the API. Check your internet connection and the base URL.",
        True, 5
    ),
    ErrorType.TIMEOUT: (
        "Request timed out",
        "The request took too long. Try reducing input size or increasing timeout settings.",
        True, 10
    ),
    ErrorType.DNS_ERROR: (
        "DNS resolution failed",
        "Could not resolve the API hostname. Check the base URL and your network settings.",
        True, 5
    ),
    ErrorType.SSL_ERROR: (
        "SSL/TLS error",
        "Secure connection failed. This might be a certificate issue or network problem.",
        True, 5
    ),
    ErrorType.MODEL_NOT_FOUND: (
        "Model not found",
        "The specified model doesn't exist or isn't available. Check the model name and your access.",
        False, 0
    ),
    ErrorType.PROVIDER_ERROR: (
        "Provider error",
        "The AI provider returned an error. Try a different model or provider.",
        True, 10
    ),
    ErrorType.SERVICE_UNAVAILABLE: (
        "Service unavailable",
        "The API service is temporarily down. Please try again later.",
        True, 60
    ),
    ErrorType.CONTENT_FILTER: (
        "Content filtered",
        "The request or response was blocked by content filters. Modify your input or prompt.",
        False, 0
    ),
    ErrorType.INVALID_REQUEST: (
        "Invalid request",
        "The request format is incorrect. Check your prompt and settings.",
        False, 0
    ),
    ErrorType.CONTEXT_LENGTH_EXCEEDED: (
        "Input too long",
        "The input exceeds the model's context limit. Reduce the amount of data per row.",
        False, 0
    ),
    ErrorType.INVALID_RESPONSE: (
        "Invalid response format",
        "The AI returned an unexpected format. Try adjusting your prompt to be more specific.",
        True, 0
    ),
    ErrorType.EMPTY_RESPONSE: (
        "Empty response",
        "The AI returned no content. This might be a content filter or prompt issue.",
        True, 5
    ),
    ErrorType.JSON_PARSE_ERROR: (
        "JSON parsing failed",
        "Could not parse the AI response as JSON. Adjust your prompt or disable JSON mode.",
        True, 0
    ),
    ErrorType.SERVER_ERROR: (
        "Server error (500)",
        "The API server encountered an internal error. Please try again.",
        True, 10
    ),
    ErrorType.BAD_GATEWAY: (
        "Bad gateway (502/503)",
        "The API gateway is having issues. Wait a moment and try again.",
        True, 30
    ),
    ErrorType.UNKNOWN: (
        "Unknown error",
        "An unexpected error occurred. Check the error details for more information.",
        True, 5
    ),
}

# ==========================================
# ERROR CLASSIFIER
# ==========================================

class ErrorClassifier:
    """Classifies exceptions into actionable error types"""

    @staticmethod
    def classify(error: Exception) -> ErrorInfo:
        """Classify an exception and return detailed error info"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        full_error = f"{type(error).__name__}: {str(error)}"

        # Try to get full traceback for logging
        try:
            tb = traceback.format_exc()
        except:
            tb = full_error

        error_type = ErrorType.UNKNOWN

        # Authentication errors
        if any(x in error_str for x in ['401', 'unauthorized', 'invalid.*key', 'authentication', 'api key']):
            if 'invalid' in error_str:
                error_type = ErrorType.INVALID_API_KEY
            else:
                error_type = ErrorType.AUTH_ERROR

        # Quota errors
        elif any(x in error_str for x in ['quota', 'billing', 'insufficient_quota', 'exceeded']):
            error_type = ErrorType.INSUFFICIENT_QUOTA

        # Rate limiting
        elif any(x in error_str for x in ['429', 'rate limit', 'too many requests', 'rate_limit']):
            if 'too many' in error_str:
                error_type = ErrorType.TOO_MANY_REQUESTS
            else:
                error_type = ErrorType.RATE_LIMIT

        # Connection errors
        elif any(x in error_str for x in ['connection', 'connect', 'refused', 'unreachable']):
            error_type = ErrorType.CONNECTION_ERROR

        # Timeout
        elif any(x in error_str for x in ['timeout', 'timed out', 'deadline']):
            error_type = ErrorType.TIMEOUT

        # DNS
        elif any(x in error_str for x in ['dns', 'resolve', 'getaddrinfo', 'nodename']):
            error_type = ErrorType.DNS_ERROR

        # SSL
        elif any(x in error_str for x in ['ssl', 'certificate', 'tls']):
            error_type = ErrorType.SSL_ERROR

        # Model errors
        elif any(x in error_str for x in ['model', 'not found', 'does not exist', '404']):
            error_type = ErrorType.MODEL_NOT_FOUND

        # Service unavailable
        elif any(x in error_str for x in ['503', 'service unavailable', 'overloaded']):
            error_type = ErrorType.SERVICE_UNAVAILABLE

        # Bad gateway
        elif any(x in error_str for x in ['502', 'bad gateway', '504', 'gateway']):
            error_type = ErrorType.BAD_GATEWAY

        # Server error
        elif any(x in error_str for x in ['500', 'internal server error', 'server error']):
            error_type = ErrorType.SERVER_ERROR

        # Content filter
        elif any(x in error_str for x in ['content filter', 'content_filter', 'blocked', 'flagged', 'safety']):
            error_type = ErrorType.CONTENT_FILTER

        # Context length
        elif any(x in error_str for x in ['context length', 'too long', 'maximum context', 'token limit', 'max_tokens']):
            error_type = ErrorType.CONTEXT_LENGTH_EXCEEDED

        # Invalid request (400)
        elif any(x in error_str for x in ['400', 'bad request', 'invalid request', 'invalid_request']):
            error_type = ErrorType.INVALID_REQUEST

        # JSON errors
        elif any(x in error_str for x in ['json', 'decode', 'parse']):
            error_type = ErrorType.JSON_PARSE_ERROR

        # Empty response
        elif any(x in error_str for x in ['empty', 'no content', 'none']):
            error_type = ErrorType.EMPTY_RESPONSE

        # Get suggestion info
        message, suggestion, is_retryable, retry_delay = ERROR_SUGGESTIONS.get(
            error_type, ERROR_SUGGESTIONS[ErrorType.UNKNOWN]
        )

        return ErrorInfo(
            error_type=error_type,
            message=message,
            suggestion=suggestion,
            is_retryable=is_retryable,
            retry_delay=retry_delay,
            original_error=full_error
        )

    @staticmethod
    def classify_from_string(error_str: str) -> ErrorInfo:
        """Classify an error from its string representation"""
        class FakeError(Exception):
            pass
        fake = FakeError(error_str)
        return ErrorClassifier.classify(fake)

    @staticmethod
    def is_empty_result(output: str) -> bool:
        """Check if a result is effectively empty/NA"""
        if output is None:
            return True

        output_clean = str(output).strip().lower()

        # Check for empty
        if not output_clean:
            return True

        # Check for common NA indicators
        na_indicators = [
            'n/a', 'na', 'null', 'none', 'undefined', 'empty',
            '{}', '[]', '""', "''", 'error', 'failed'
        ]

        if output_clean in na_indicators:
            return True

        # Check for error prefix
        if output_clean.startswith('error:'):
            return True

        # Check for very short responses that are likely errors
        if len(output_clean) < 3:
            return True

        return False

    @staticmethod
    def should_auto_retry(error_info: ErrorInfo, attempt: int, max_retries: int = 3) -> Tuple[bool, int]:
        """
        Determine if we should auto-retry based on error type and attempt count.
        Returns (should_retry, delay_seconds)
        """
        if attempt >= max_retries:
            return False, 0

        if not error_info.is_retryable:
            return False, 0

        # Exponential backoff
        delay = error_info.retry_delay * (2 ** attempt)
        return True, min(delay, 120)  # Max 2 minute delay

# ==========================================
# ERROR FORMATTING
# ==========================================

def format_error_for_display(error_info: ErrorInfo, show_original: bool = False) -> str:
    """Format error info for UI display"""
    lines = [
        f"**{error_info.message}**",
        "",
        f"💡 {error_info.suggestion}",
    ]

    if error_info.is_retryable:
        lines.append("")
        lines.append(f"🔄 This error is retryable (delay: {error_info.retry_delay}s)")

    if show_original:
        lines.append("")
        lines.append(f"```\n{error_info.original_error}\n```")

    return "\n".join(lines)

def format_error_summary(errors: list) -> dict:
    """Summarize a list of errors by type"""
    summary = {}
    for err in errors:
        error_type = err.get('error_type', 'unknown')
        if error_type not in summary:
            summary[error_type] = {
                'count': 0,
                'message': err.get('error_message', 'Unknown error'),
                'rows': []
            }
        summary[error_type]['count'] += 1
        summary[error_type]['rows'].append(err.get('row_index', -1))

    return summary

# ==========================================
# RESULT VALIDATION
# ==========================================

def validate_json_output(output: str, expected_schema: dict = None) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Validate JSON output and optionally check against expected schema.
    Returns (is_valid, parsed_data, error_message)
    """
    import json

    if not output or not output.strip():
        return False, None, "Empty response"

    # Try to parse JSON
    try:
        # Handle markdown code blocks
        clean_output = output.strip()
        if clean_output.startswith('```'):
            # Extract content between code blocks
            lines = clean_output.split('\n')
            start = 1
            end = len(lines) - 1
            if lines[-1].strip() == '```':
                clean_output = '\n'.join(lines[start:end])
            else:
                clean_output = '\n'.join(lines[start:])

        data = json.loads(clean_output)

        # If no schema provided, just return parsed data
        if not expected_schema:
            return True, data, None

        # Check schema fields
        missing_fields = []
        for field in expected_schema.keys():
            if field not in data:
                missing_fields.append(field)

        if missing_fields:
            return False, data, f"Missing fields: {', '.join(missing_fields)}"

        return True, data, None

    except json.JSONDecodeError as e:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return True, data, None
            except:
                pass

        return False, None, f"JSON parse error: {str(e)}"

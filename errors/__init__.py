"""
Handai Error Handling Module
Error classification and suggestions
"""

from .classifier import (
    ErrorType,
    ErrorInfo,
    ErrorClassifier,
    ERROR_SUGGESTIONS,
    format_error_for_display,
    format_error_summary,
    validate_json_output,
)

__all__ = [
    "ErrorType",
    "ErrorInfo",
    "ErrorClassifier",
    "ERROR_SUGGESTIONS",
    "format_error_for_display",
    "format_error_summary",
    "validate_json_output",
]

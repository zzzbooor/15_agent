from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    PARAM_MISSING = "PARAM_MISSING"
    PARAM_INVALID = "PARAM_INVALID"
    PARAM_OUT_OF_RANGE = "PARAM_OUT_OF_RANGE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DECODE_ERROR = "DECODE_ERROR"
    TIMEOUT = "TIMEOUT"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    SANDBOX_VIOLATION = "SANDBOX_VIOLATION"
    COMPOSITE_STEP_FAILED = "COMPOSITE_STEP_FAILED"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class SkillFault(Exception):
    """Expected Skill failure with a stable machine-readable error code."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
        error_type: str | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = str(message)
        self.details = dict(details or {})
        self.retryable = bool(retryable)
        self.error_type = error_type or type(self).__name__
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


def normalize_exception(exc: Exception) -> SkillFault:
    """Convert arbitrary failures into the public B2 error taxonomy."""

    if isinstance(exc, SkillFault):
        return exc
    if isinstance(exc, FileNotFoundError):
        code = ErrorCode.FILE_NOT_FOUND
    elif isinstance(exc, PermissionError):
        code = ErrorCode.PERMISSION_DENIED
    elif isinstance(exc, TimeoutError):
        code = ErrorCode.TIMEOUT
    elif isinstance(exc, (MemoryError, RecursionError)):
        code = ErrorCode.RESOURCE_EXHAUSTED
    elif isinstance(exc, UnicodeError):
        code = ErrorCode.DECODE_ERROR
    elif isinstance(exc, KeyError):
        code = ErrorCode.PARAM_MISSING
    elif isinstance(exc, IndexError):
        code = ErrorCode.PARAM_OUT_OF_RANGE
    elif isinstance(exc, (ZeroDivisionError, ArithmeticError, OverflowError)):
        code = ErrorCode.EXECUTION_ERROR
    elif isinstance(exc, (TypeError, ValueError)):
        code = ErrorCode.PARAM_INVALID
    else:
        code = ErrorCode.UNKNOWN_ERROR
    message = str(exc) or type(exc).__name__
    return SkillFault(code, message, error_type=type(exc).__name__)

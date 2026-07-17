from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from skills.core.context import current_context
from skills.core.errors import ErrorCode, SkillFault


WORKER_PATH = Path(__file__).resolve().parent / "sandbox" / "worker.py"


def _minimal_environment() -> dict[str, str]:
    environment = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            environment[key] = os.environ[key]
    return environment


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def code_executor(code: str, timeout: int = 10) -> dict:
    """Interpret a safe Python subset in a bounded isolated worker process."""

    limits = current_context().limits.sandbox
    if not isinstance(code, str) or not code.strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "code must be a non-empty string")
    if len(code) > limits.max_code_chars:
        raise SkillFault(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"code is too long (maximum {limits.max_code_chars} characters)",
        )
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise SkillFault(ErrorCode.PARAM_INVALID, "timeout must be a positive integer")
    effective_timeout = min(float(timeout), limits.wall_timeout_seconds)
    payload = json.dumps(
        {"code": code, "policy": asdict(limits)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    command = [sys.executable, "-I", "-S", str(WORKER_PATH)]
    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": _minimal_environment(),
        "start_new_session": os.name == "posix",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="restricted_skill_") as temporary_directory:
        process = subprocess.Popen(command, cwd=temporary_directory, **popen_kwargs)
        try:
            stdout, stderr = process.communicate(payload, timeout=effective_timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            process.communicate()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            raise SkillFault(
                ErrorCode.TIMEOUT,
                f"restricted execution exceeded {effective_timeout:g} seconds",
                details={"execution_time_ms": elapsed_ms},
                retryable=True,
            ) from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if not stdout:
        code_name = ErrorCode.RESOURCE_EXHAUSTED if process.returncode else ErrorCode.EXECUTION_ERROR
        raise SkillFault(
            code_name,
            "restricted worker returned no result",
            details={"returncode": process.returncode, "stderr": stderr[:1000]},
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise SkillFault(
            ErrorCode.EXECUTION_ERROR,
            "restricted worker returned invalid JSON",
            details={"returncode": process.returncode, "stderr": stderr[:1000]},
        ) from exc
    if not isinstance(response, dict) or not response.get("ok"):
        error = response.get("error", {}) if isinstance(response, dict) else {}
        try:
            error_code = ErrorCode(error.get("code", ErrorCode.EXECUTION_ERROR.value))
        except ValueError:
            error_code = ErrorCode.EXECUTION_ERROR
        raise SkillFault(
            error_code,
            str(error.get("message") or "restricted execution failed"),
            details=error.get("details") if isinstance(error.get("details"), dict) else {},
            retryable=bool(error.get("retryable", False)),
            error_type=str(error.get("type") or "SandboxWorkerError"),
        )
    worker_result = response.get("result")
    if not isinstance(worker_result, dict):
        raise SkillFault(ErrorCode.EXECUTION_ERROR, "restricted worker result must be an object")
    return {
        "returncode": process.returncode,
        "stdout": str(worker_result.get("stdout", "")),
        "stderr": stderr[:1000],
        "execution_time_ms": elapsed_ms,
        "killed": False,
        "engine": "restricted_ast_v1",
        "operations": int(worker_result.get("operations", 0)),
        "loop_iterations": int(worker_result.get("loop_iterations", 0)),
        "isolation": {
            "python_isolated_mode": True,
            "user_source_executed_directly": False,
            "privilege_dropped": bool(worker_result.get("privilege_dropped", False)),
            "effective_uid": worker_result.get("effective_uid"),
        },
        "limits": {
            "max_execution_time_sec": effective_timeout,
            "max_memory_mb": limits.memory_mb,
            "max_code_length": limits.max_code_chars,
            "max_operations": limits.max_operations,
            "max_output_bytes": limits.max_output_bytes,
        },
    }

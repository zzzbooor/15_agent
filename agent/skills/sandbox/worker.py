from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.core.errors import ErrorCode, SkillFault, normalize_exception
from skills.core.limits import SandboxLimits
from skills.sandbox.evaluator import RestrictedInterpreter


def _apply_posix_limits(limits: SandboxLimits) -> None:
    if os.name != "posix":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    memory_bytes = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_output_bytes, limits.max_output_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files, limits.max_open_files))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _drop_root_privileges() -> bool:
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    try:
        import pwd

        nobody = pwd.getpwnam("nobody")
        os.setgroups([])
        os.setgid(nobody.pw_gid)
        os.setuid(nobody.pw_uid)
        return True
    except (KeyError, OSError):
        return False


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
            raise SkillFault(ErrorCode.PARAM_INVALID, "sandbox payload must contain code")
        policy = payload.get("policy")
        if not isinstance(policy, dict):
            raise SkillFault(ErrorCode.PARAM_INVALID, "sandbox payload must contain policy")
        limits = SandboxLimits(**policy)
        _apply_posix_limits(limits)
        privilege_dropped = _drop_root_privileges()
        result = RestrictedInterpreter(limits).run(payload["code"])
        result["privilege_dropped"] = privilege_dropped
        result["effective_uid"] = os.geteuid() if hasattr(os, "geteuid") else None
        response = {"ok": True, "result": result}
    except Exception as exc:
        response = {"ok": False, "error": normalize_exception(exc).to_dict()}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

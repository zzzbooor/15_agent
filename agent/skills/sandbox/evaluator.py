from __future__ import annotations

import ast
import math
import operator
from collections.abc import Iterable
from typing import Any

from skills.core.errors import ErrorCode, SkillFault
from skills.core.limits import SandboxLimits


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
_MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
}
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class RestrictedInterpreter:
    """Small AST interpreter that never executes the submitted source directly."""

    def __init__(self, limits: SandboxLimits) -> None:
        self.limits = limits
        self.operations = 0
        self.loop_iterations = 0
        self.variables: dict[str, Any] = {}
        self.modules: dict[str, str] = {}
        self.output_parts: list[str] = []
        self.output_bytes = 0

    def run(self, source: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise SkillFault(
                ErrorCode.PARAM_INVALID,
                f"invalid Python syntax at line {exc.lineno}: {exc.msg}",
            ) from exc
        for statement in tree.body:
            self._statement(statement)
        return {
            "stdout": "".join(self.output_parts),
            "operations": self.operations,
            "loop_iterations": self.loop_iterations,
        }

    def _tick(self, amount: int = 1) -> None:
        self.operations += amount
        if self.operations > self.limits.max_operations:
            raise SkillFault(
                ErrorCode.RESOURCE_EXHAUSTED,
                f"operation budget exceeded ({self.limits.max_operations})",
            )

    @staticmethod
    def _safe_name(name: str) -> str:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"name is not allowed: {name}")
        return name

    def _statement(self, node: ast.stmt) -> None:
        self._tick()
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "math":
                    raise SkillFault(
                        ErrorCode.SANDBOX_VIOLATION,
                        f"module is not allowed: {alias.name}",
                    )
                local_name = self._safe_name(alias.asname or alias.name)
                self.modules[local_name] = alias.name
            return
        if isinstance(node, ast.ImportFrom):
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "from-import statements are not allowed")
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "only simple variable assignment is allowed")
            name = self._safe_name(node.targets[0].id)
            self.variables[name] = self._check_value(self._expression(node.value))
            return
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = self._safe_name(node.target.id)
            if name not in self.variables:
                raise SkillFault(ErrorCode.PARAM_INVALID, f"unknown variable: {name}")
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "augmented operator is not allowed")
            self.variables[name] = self._apply_binary(operation, self.variables[name], self._expression(node.value), node.op)
            return
        if isinstance(node, ast.Expr):
            self._expression(node.value)
            return
        if isinstance(node, ast.If):
            branch = node.body if bool(self._expression(node.test)) else node.orelse
            for statement in branch:
                self._statement(statement)
            return
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "for-loop target must be one variable")
            name = self._safe_name(node.target.id)
            iterable = self._expression(node.iter)
            if not isinstance(iterable, (range, list, tuple)):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "for-loop iterable must be bounded")
            length = len(iterable)
            if self.loop_iterations + length > self.limits.max_loop_iterations:
                raise SkillFault(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    f"loop iteration budget exceeded ({self.limits.max_loop_iterations})",
                )
            for value in iterable:
                self._tick()
                self.loop_iterations += 1
                self.variables[name] = self._check_value(value)
                for statement in node.body:
                    self._statement(statement)
            if node.orelse:
                for statement in node.orelse:
                    self._statement(statement)
            return
        if isinstance(node, ast.Pass):
            return
        raise SkillFault(
            ErrorCode.SANDBOX_VIOLATION,
            f"statement is not allowed: {type(node).__name__}",
        )

    def _expression(self, node: ast.expr) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "constant type is not allowed")
            return self._check_value(node.value)
        if isinstance(node, ast.Name):
            name = self._safe_name(node.id)
            if name in self.variables:
                return self.variables[name]
            raise SkillFault(ErrorCode.PARAM_INVALID, f"unknown variable: {name}")
        if isinstance(node, ast.List):
            return self._check_value([self._expression(item) for item in node.elts])
        if isinstance(node, ast.Tuple):
            return self._check_value(tuple(self._expression(item) for item in node.elts))
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "dictionary unpacking is not allowed")
            return self._check_value(
                {self._expression(key): self._expression(value) for key, value in zip(node.keys, node.values)}
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            try:
                return self._check_value(_UNARY_OPERATORS[type(node.op)](self._expression(node.operand)))
            except SkillFault:
                raise
            except Exception as exc:
                raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc)) from exc
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = self._expression(node.left)
            right = self._expression(node.right)
            return self._apply_binary(_BINARY_OPERATORS[type(node.op)], left, right, node.op)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for value_node in node.values:
                    result = self._expression(value_node)
                    if not result:
                        break
                return self._check_value(result)
            if isinstance(node.op, ast.Or):
                result = False
                for value_node in node.values:
                    result = self._expression(value_node)
                    if result:
                        break
                return self._check_value(result)
        if isinstance(node, ast.Compare):
            left = self._expression(node.left)
            for operator_node, comparator_node in zip(node.ops, node.comparators):
                operation = _COMPARE_OPERATORS.get(type(operator_node))
                if operation is None:
                    raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "comparison is not allowed")
                right = self._expression(comparator_node)
                try:
                    matched = bool(operation(left, right))
                except Exception as exc:
                    raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc)) from exc
                if not matched:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._expression(node.body if self._expression(node.test) else node.orelse)
        if isinstance(node, ast.Subscript):
            container = self._expression(node.value)
            if not isinstance(container, (list, tuple, dict, str)):
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "subscript target is not allowed")
            if isinstance(node.slice, ast.Slice):
                lower = self._expression(node.slice.lower) if node.slice.lower else None
                upper = self._expression(node.slice.upper) if node.slice.upper else None
                step = self._expression(node.slice.step) if node.slice.step else None
                index: Any = slice(lower, upper, step)
            else:
                index = self._expression(node.slice)
            try:
                return self._check_value(container[index])
            except (KeyError, IndexError, TypeError) as exc:
                raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc) or type(exc).__name__) from exc
        if isinstance(node, ast.Attribute):
            return self._math_attribute(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise SkillFault(
            ErrorCode.SANDBOX_VIOLATION,
            f"expression is not allowed: {type(node).__name__}",
        )

    def _math_attribute(self, node: ast.Attribute) -> Any:
        if not isinstance(node.value, ast.Name):
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "nested attribute access is not allowed")
        module_alias = self._safe_name(node.value.id)
        attribute = self._safe_name(node.attr)
        if self.modules.get(module_alias) != "math" or attribute not in _MATH_CONSTANTS:
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"attribute is not allowed: {attribute}")
        return _MATH_CONSTANTS[attribute]

    def _call(self, node: ast.Call) -> Any:
        if any(keyword.arg is None for keyword in node.keywords):
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "argument unpacking is not allowed")
        if isinstance(node.func, ast.Name):
            name = self._safe_name(node.func.id)
            arguments = [self._expression(argument) for argument in node.args]
            keywords = {self._safe_name(keyword.arg): self._expression(keyword.value) for keyword in node.keywords}
            return self._call_builtin(name, arguments, keywords)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module_alias = self._safe_name(node.func.value.id)
            function_name = self._safe_name(node.func.attr)
            if self.modules.get(module_alias) != "math" or function_name not in _MATH_FUNCTIONS:
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"function is not allowed: {function_name}")
            if node.keywords:
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "math keyword arguments are not allowed")
            arguments = [self._expression(argument) for argument in node.args]
            try:
                return self._check_value(_MATH_FUNCTIONS[function_name](*arguments))
            except SkillFault:
                raise
            except Exception as exc:
                raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc) or type(exc).__name__) from exc
        raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "indirect function calls are not allowed")

    def _call_builtin(self, name: str, arguments: list[Any], keywords: dict[str, Any]) -> Any:
        if name == "print":
            unknown = set(keywords) - {"sep", "end"}
            if unknown:
                raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "print only accepts sep and end keywords")
            separator = keywords.get("sep", " ")
            ending = keywords.get("end", "\n")
            if not isinstance(separator, str) or not isinstance(ending, str):
                raise SkillFault(ErrorCode.PARAM_INVALID, "print sep and end must be strings")
            rendered = separator.join(self._render(item) for item in arguments) + ending
            encoded_size = len(rendered.encode("utf-8"))
            if self.output_bytes + encoded_size > self.limits.max_output_bytes:
                raise SkillFault(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    f"stdout exceeds {self.limits.max_output_bytes} bytes",
                )
            self.output_parts.append(rendered)
            self.output_bytes += encoded_size
            return None
        if keywords:
            raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"{name} does not accept keyword arguments")
        try:
            if name == "len" and len(arguments) == 1:
                return len(arguments[0])
            if name == "abs" and len(arguments) == 1:
                return self._check_value(abs(arguments[0]))
            if name == "round" and len(arguments) in {1, 2}:
                return self._check_value(round(*arguments))
            if name == "sum" and len(arguments) in {1, 2}:
                return self._check_value(sum(*arguments))
            if name in {"min", "max"} and arguments:
                function = min if name == "min" else max
                return self._check_value(function(*arguments))
            if name == "range" and 1 <= len(arguments) <= 3 and all(
                isinstance(item, int) and not isinstance(item, bool) for item in arguments
            ):
                value = range(*arguments)
                if len(value) > self.limits.max_loop_iterations:
                    raise SkillFault(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        f"range exceeds {self.limits.max_loop_iterations} items",
                    )
                return value
        except SkillFault:
            raise
        except Exception as exc:
            raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc) or type(exc).__name__) from exc
        raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"function is not allowed: {name}")

    def _apply_binary(self, operation, left: Any, right: Any, operator_node: ast.operator) -> Any:
        if isinstance(operator_node, ast.Pow):
            if not isinstance(right, (int, float)) or isinstance(right, bool) or abs(right) > 1024:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "power exponent is too large")
        if isinstance(operator_node, ast.Mult):
            sequence, multiplier = (left, right) if isinstance(right, int) else (right, left)
            if isinstance(sequence, (str, list, tuple)) and isinstance(multiplier, int):
                projected = len(sequence) * max(0, multiplier)
                maximum = (
                    self.limits.max_string_bytes
                    if isinstance(sequence, str)
                    else self.limits.max_container_items
                )
                if projected > maximum:
                    raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "sequence multiplication is too large")
        try:
            return self._check_value(operation(left, right))
        except SkillFault:
            raise
        except ZeroDivisionError as exc:
            raise SkillFault(ErrorCode.EXECUTION_ERROR, "division by zero") from exc
        except Exception as exc:
            raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc) or type(exc).__name__) from exc

    def _check_value(self, value: Any, depth: int = 0) -> Any:
        if depth > 20:
            raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "container nesting is too deep")
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value.bit_length() > self.limits.max_integer_bits:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "integer is too large")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "non-finite numbers are not allowed")
            return value
        if isinstance(value, str):
            if len(value.encode("utf-8")) > self.limits.max_string_bytes:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "string is too large")
            return value
        if isinstance(value, range):
            if len(value) > self.limits.max_loop_iterations:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "range is too large")
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > self.limits.max_container_items:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "container is too large")
            for item in value:
                self._check_value(item, depth + 1)
            return value
        if isinstance(value, dict):
            if len(value) > self.limits.max_container_items:
                raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "dictionary is too large")
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise SkillFault(ErrorCode.SANDBOX_VIOLATION, "dictionary key type is not allowed")
                self._check_value(key, depth + 1)
                self._check_value(item, depth + 1)
            return value
        raise SkillFault(ErrorCode.SANDBOX_VIOLATION, f"value type is not allowed: {type(value).__name__}")

    def _render(self, value: Any) -> str:
        self._check_value(value)
        return str(value)

from __future__ import annotations

import ast
import math
import operator
from numbers import Real

from skills.core.context import current_context
from skills.core.errors import ErrorCode, SkillFault


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _validate_number(value: object, *, max_integer_bits: int, max_abs_result: float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SkillFault(ErrorCode.EXECUTION_ERROR, "calculation did not produce a real number")
    if isinstance(value, int) and value.bit_length() > max_integer_bits:
        raise SkillFault(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"integer result exceeds {max_integer_bits} bits",
        )
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise SkillFault(ErrorCode.RESOURCE_EXHAUSTED, "calculation result is too large") from exc
    if not finite or abs(value) > max_abs_result:
        raise SkillFault(ErrorCode.PARAM_OUT_OF_RANGE, "calculation result is out of range")
    return value


def _evaluate(node: ast.AST, limits) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, limits)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _validate_number(
            node.value,
            max_integer_bits=limits.max_integer_bits,
            max_abs_result=limits.max_abs_result,
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, limits))
        return _validate_number(
            value,
            max_integer_bits=limits.max_integer_bits,
            max_abs_result=limits.max_abs_result,
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left, limits)
        right = _evaluate(node.right, limits)
        if isinstance(node.op, ast.Pow) and abs(right) > limits.max_exponent:
            raise SkillFault(
                ErrorCode.PARAM_OUT_OF_RANGE,
                f"exponent magnitude must not exceed {limits.max_exponent}",
            )
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise SkillFault(ErrorCode.EXECUTION_ERROR, "division by zero") from exc
        except (ArithmeticError, OverflowError) as exc:
            raise SkillFault(ErrorCode.EXECUTION_ERROR, str(exc) or type(exc).__name__) from exc
        return _validate_number(
            result,
            max_integer_bits=limits.max_integer_bits,
            max_abs_result=limits.max_abs_result,
        )
    raise SkillFault(
        ErrorCode.UNSUPPORTED_OPERATION,
        f"unsupported expression element: {type(node).__name__}",
    )


def calculator(expression: str) -> dict:
    """Safely evaluate a numeric arithmetic expression."""

    limits = current_context().limits.calculator
    if not isinstance(expression, str) or not expression.strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "expression must be a non-empty string")
    if len(expression) > limits.max_expression_chars:
        raise SkillFault(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"expression is too long (maximum {limits.max_expression_chars} characters)",
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SkillFault(ErrorCode.PARAM_INVALID, "invalid arithmetic expression") from exc
    return {"result": _evaluate(tree, limits)}

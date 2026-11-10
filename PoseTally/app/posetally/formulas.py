"""Formula columns: combine score columns with ordinary maths.

A formula is written the way it would be in a spreadsheet, with column names in
square brackets::

    [Left arm] + [Right arm]
    [Correct] / ([Correct] + [Wrong]) * 100
    IF([Time in goal (s)] > 10, 1, 0)

Only arithmetic is allowed. The expression is parsed and checked piece by
piece, so a formula can never run arbitrary code.

Empty cells: a column can be empty for a video (for example "time of first
crossing" when the animal never crossed). Arithmetic with an empty value gives
an empty result. SUM, MIN, MAX and MEAN skip empty values. Use
``IFBLANK([Column], 0)`` to replace an empty value with a number.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Mapping

NAN = float("nan")


class FormulaError(ValueError):
    """A formula that cannot be understood, with a message meant for the user."""


_REF = re.compile(r"\[([^\[\]]*)\]")
_IF_CALL = re.compile(r"\bif\s*\(", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^__col(\d+)__$")


def _blank(x: float) -> bool:
    return x != x


def _truthy(x: float) -> bool:
    return x == x and x != 0


def _present(args: Iterable[float]) -> list[float]:
    return [a for a in args if a == a]


def _fn_min(*args):
    vals = _present(args)
    return min(vals) if vals else NAN


def _fn_max(*args):
    vals = _present(args)
    return max(vals) if vals else NAN


def _fn_sum(*args):
    return float(sum(_present(args)))


def _fn_mean(*args):
    vals = _present(args)
    return sum(vals) / len(vals) if vals else NAN


def _fn_count(*args):
    return float(len(_present(args)))


def _fn_sqrt(x):
    return math.sqrt(x) if x >= 0 else NAN


def _fn_ln(x, base=None):
    if not x > 0:
        return NAN
    if base is None:
        return math.log(x)
    if not base > 0 or base == 1:
        return NAN
    return math.log(x, base)


def _fn_log10(x):
    return math.log10(x) if x > 0 else NAN


def _fn_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return NAN


def _fn_round(x, digits=0.0):
    if _blank(x) or _blank(digits):
        return NAN
    factor = 10.0 ** int(digits)
    return math.copysign(math.floor(abs(x) * factor + 0.5) / factor, x)


def _fn_floor(x):
    return NAN if _blank(x) or math.isinf(x) else float(math.floor(x))


def _fn_ceil(x):
    return NAN if _blank(x) or math.isinf(x) else float(math.ceil(x))


def _fn_abs(x):
    return abs(x)


def _fn_if(cond, a, b):
    return a if _truthy(cond) else b


def _fn_isblank(x):
    return 1.0 if _blank(x) else 0.0


def _fn_ifblank(x, alt):
    return alt if _blank(x) else x


def _fn_and(*args):
    return 1.0 if all(_truthy(a) for a in args) else 0.0


def _fn_or(*args):
    return 1.0 if any(_truthy(a) for a in args) else 0.0


def _fn_not(x):
    return 0.0 if _truthy(x) else 1.0


# name -> (function, minimum arguments, maximum arguments or None for any number)
FUNCTIONS = {
    "MIN": (_fn_min, 1, None),
    "MAX": (_fn_max, 1, None),
    "SUM": (_fn_sum, 1, None),
    "MEAN": (_fn_mean, 1, None),
    "AVERAGE": (_fn_mean, 1, None),
    "AVG": (_fn_mean, 1, None),
    "COUNT": (_fn_count, 1, None),
    "ABS": (_fn_abs, 1, 1),
    "SQRT": (_fn_sqrt, 1, 1),
    "LN": (_fn_ln, 1, 1),
    "LOG": (_fn_ln, 1, 2),
    "LOG10": (_fn_log10, 1, 1),
    "EXP": (_fn_exp, 1, 1),
    "ROUND": (_fn_round, 1, 2),
    "FLOOR": (_fn_floor, 1, 1),
    "CEIL": (_fn_ceil, 1, 1),
    "IF": (_fn_if, 3, 3),
    "ISBLANK": (_fn_isblank, 1, 1),
    "IFBLANK": (_fn_ifblank, 2, 2),
    "AND": (_fn_and, 1, None),
    "OR": (_fn_or, 1, None),
    "NOT": (_fn_not, 1, 1),
}

CONSTANTS = {"PI": math.pi, "TRUE": 1.0, "FALSE": 0.0}

FUNCTION_HELP = (
    "+  -  *  /  ^ (power)  ( )   comparisons: <  <=  >  >=  ==  !=\n"
    "MIN, MAX, SUM, MEAN, COUNT, ABS, SQRT, ROUND(x, digits), FLOOR, CEIL, LN, LOG10, EXP\n"
    "IF(test, value if true, value if false), AND, OR, NOT, ISBLANK(x), IFBLANK(x, value)"
)


class Formula:
    """A checked formula that can be evaluated for one row of results."""

    def __init__(self, expression: str):
        self.expression = expression
        text = (expression or "").strip()
        if text.startswith("="):
            text = text[1:].strip()
        if not text:
            raise FormulaError("The formula is empty.")

        self._columns: list[str] = []

        def _swap(match: re.Match) -> str:
            name = match.group(1).strip()
            if not name:
                raise FormulaError("There is an empty pair of square brackets [ ].")
            if name not in self._columns:
                self._columns.append(name)
            # spaces keep "[A][B]" (no operator between) from fusing into one name
            return f" __col{self._columns.index(name)}__ "

        text = _REF.sub(_swap, text)
        if "[" in text or "]" in text:
            raise FormulaError("A square bracket is not closed. Column names go in [square brackets].")
        text = text.replace("^", "**")
        text = " ".join(_IF_CALL.sub("IF(", text).split())  # also allows line breaks
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(
                "The formula could not be read. Check for a missing bracket or operator."
            ) from exc
        self._bare: list[str] = []
        self._check(tree)
        self._tree = tree

    # -- checking ---------------------------------------------------------------

    def _check(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._check(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise FormulaError(f"Only numbers can be typed into a formula, not {node.value!r}.")
        elif isinstance(node, ast.Name):
            if _PLACEHOLDER.match(node.id):
                return
            if node.id.upper() in CONSTANTS:
                return
            if node.id.upper() in FUNCTIONS:
                raise FormulaError(f"{node.id.upper()} needs brackets, for example {node.id.upper()}(...).")
            if node.id not in self._bare:
                self._bare.append(node.id)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                raise FormulaError("That operator is not supported.")
            self._check(node.operand)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                raise FormulaError("That operator is not supported. Use + - * / ^.")
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._check(value)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                    raise FormulaError("That comparison is not supported. Use < <= > >= == !=.")
            self._check(node.left)
            for comp in node.comparators:
                self._check(comp)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id.upper() not in FUNCTIONS:
                name = getattr(node.func, "id", "that")
                raise FormulaError(f"There is no function called {name}.")
            if node.keywords:
                raise FormulaError("Functions take plain values separated by commas.")
            _, lo, hi = FUNCTIONS[node.func.id.upper()]
            count = len(node.args)
            if count < lo or (hi is not None and count > hi):
                wanted = f"{lo}" if hi == lo else (f"at least {lo}" if hi is None else f"{lo} to {hi}")
                raise FormulaError(f"{node.func.id.upper()} needs {wanted} value(s) in its brackets.")
            for arg in node.args:
                self._check(arg)
        else:
            raise FormulaError("The formula contains something that is not arithmetic.")

    # -- information ------------------------------------------------------------

    @property
    def references(self) -> list[str]:
        """Column names used by the formula, in order of first appearance."""
        return list(self._columns) + [b for b in self._bare if b not in self._columns]

    # -- evaluation -------------------------------------------------------------

    def evaluate(self, values: Mapping[str, float]) -> float:
        """Work the formula out for one row. Missing columns raise FormulaError."""
        try:
            result = self._eval(self._tree.body, values)
        except FormulaError:
            raise
        except (ArithmeticError, ValueError):
            return NAN
        if isinstance(result, complex):
            return NAN
        result = float(result)
        return NAN if math.isinf(result) else result

    def _value(self, name: str, values: Mapping[str, float]) -> float:
        if name not in values:
            raise FormulaError(f"There is no column called [{name}].")
        value = values[name]
        return NAN if value is None else float(value)

    def _eval(self, node: ast.AST, values: Mapping[str, float]) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            match = _PLACEHOLDER.match(node.id)
            if match:
                return self._value(self._columns[int(match.group(1))], values)
            if node.id.upper() in CONSTANTS and node.id not in values:
                return CONSTANTS[node.id.upper()]
            return self._value(node.id, values)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, values)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Not):
                return 0.0 if _truthy(operand) else 1.0
            return operand
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, values)
            right = self._eval(node.right, values)
            if _blank(left) or _blank(right):
                return NAN
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right if right != 0 else NAN
            if isinstance(op, ast.FloorDiv):
                return float(left // right) if right != 0 else NAN
            if isinstance(op, ast.Mod):
                return math.fmod(left, right) if right != 0 else NAN
            try:
                power = left**right
            except (OverflowError, ZeroDivisionError):
                return NAN
            return NAN if isinstance(power, complex) else power
        if isinstance(node, ast.BoolOp):
            results = [_truthy(self._eval(v, values)) for v in node.values]
            if isinstance(node.op, ast.And):
                return 1.0 if all(results) else 0.0
            return 1.0 if any(results) else 0.0
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, values)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval(comp, values)
                if _blank(left) or _blank(right):
                    return 0.0
                ok = (
                    (isinstance(op, ast.Lt) and left < right)
                    or (isinstance(op, ast.LtE) and left <= right)
                    or (isinstance(op, ast.Gt) and left > right)
                    or (isinstance(op, ast.GtE) and left >= right)
                    or (isinstance(op, ast.Eq) and left == right)
                    or (isinstance(op, ast.NotEq) and left != right)
                )
                if not ok:
                    return 0.0
                left = right
            return 1.0
        if isinstance(node, ast.Call):
            func, _, _ = FUNCTIONS[node.func.id.upper()]
            args = [self._eval(a, values) for a in node.args]
            return float(func(*args))
        raise FormulaError("The formula contains something that is not arithmetic.")


def check_formula(expression: str, available: Iterable[str], own_name: str = "") -> str | None:
    """Return a message describing what is wrong with a formula, or None if fine."""
    try:
        formula = Formula(expression)
    except FormulaError as exc:
        return str(exc)
    available = set(available)
    for ref in formula.references:
        if own_name and ref == own_name:
            return "A formula cannot use its own column."
        if ref not in available:
            return f"There is no column called [{ref}]."
    return None


def evaluate_formulas(
    base_values: Mapping[str, float],
    formulas: Iterable[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, str]]:
    """Work out every formula column for one row.

    ``formulas`` is a list of ``(column name, expression)``. A formula may use
    other formula columns; they are evaluated in whatever order makes that
    possible. Returns ``(values, errors)`` where ``values`` holds the base
    values plus every formula column (empty when it could not be worked out)
    and ``errors`` maps a column name to a message.
    """
    values: dict[str, float] = dict(base_values)
    errors: dict[str, str] = {}
    pending: dict[str, Formula] = {}
    for name, expression in formulas:
        try:
            pending[name] = Formula(expression)
        except FormulaError as exc:
            errors[name] = str(exc)
            values[name] = NAN

    progressed = True
    while pending and progressed:
        progressed = False
        for name in list(pending):
            formula = pending[name]
            refs = formula.references
            if any(r in pending for r in refs):
                continue
            del pending[name]
            progressed = True
            missing = [r for r in refs if r not in values]
            if missing:
                errors[name] = f"There is no column called [{missing[0]}]."
                values[name] = NAN
                continue
            try:
                values[name] = formula.evaluate(values)
            except FormulaError as exc:
                errors[name] = str(exc)
                values[name] = NAN
    for name in pending:  # whatever is left refers to itself in a loop
        errors[name] = "This formula depends on itself through another formula."
        values[name] = NAN
    return values, errors

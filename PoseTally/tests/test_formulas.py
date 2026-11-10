import math

import pytest

from posetally.formulas import Formula, FormulaError, check_formula, evaluate_formulas
from posetally.idcolumn import IdColumn, make_id

ROW = {"Left": 3.0, "Right": 1.0, "Time (s)": 12.5, "Empty": float("nan"), "Zero": 0.0}


def value(expression, row=ROW):
    return Formula(expression).evaluate(row)


def test_arithmetic_and_column_names_with_spaces():
    assert value("[Left] + [Right]") == 4
    assert value("[Left] / ([Left] + [Right]) * 100") == 75
    assert value("[Time (s)] / 60") == pytest.approx(12.5 / 60)
    assert value("= [Left] - [Right]") == 2  # a leading "=" is fine
    assert value("Left * 2") == 6  # simple names also work without brackets


def test_power_follows_spreadsheet_rules():
    assert value("2 ^ 3 + 1") == 9
    assert value("-[Left] ^ 2") == -9


def test_functions():
    assert value("MAX([Left], [Right])") == 3
    assert value("min([Left], [Right], 0.5)") == 0.5
    assert value("ROUND([Time (s)] / 3, 2)") == 4.17
    assert value("ROUND(2.5)") == 3 and value("ROUND(-2.5)") == -3
    assert value("SQRT(16) + ABS(-2)") == 6
    assert value("IF([Left] > [Right], 1, 0)") == 1
    assert value("if([Left] < [Right], 1, 0)") == 0
    assert value("IF(AND([Left] > 0, [Right] > 0), 10, 20)") == 10
    assert value("[Left] >= 3 and [Right] == 1") == 1
    assert value("MEAN([Left], [Right])") == 2


def test_empty_values():
    assert math.isnan(value("[Left] + [Empty]"))
    assert value("SUM([Left], [Empty], [Right])") == 4
    assert value("MAX([Empty], [Right])") == 1
    assert value("IFBLANK([Empty], 0) + [Left]") == 3
    assert value("ISBLANK([Empty])") == 1 and value("ISBLANK([Left])") == 0
    assert value("IF([Empty] > 1, 1, 0)") == 0
    assert value("COUNT([Left], [Empty])") == 1


def test_division_by_zero_gives_an_empty_cell_not_a_crash():
    assert math.isnan(value("[Left] / [Zero]"))
    assert math.isnan(value("SQRT(-1)"))
    assert math.isnan(value("LN(0)"))
    assert math.isnan(value("10 ^ 400 * 10 ^ 400"))


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "[Left] +",
        "[Left",
        "[] + 1",
        "__import__('os').system('echo hi')",
        "[Left].real",
        "open('x')",
        "'text'",
        "[Left] if [Right] else 2",
        "lambda: 1",
        "NOSUCH([Left])",
        "MAX()",
        "IF([Left], 1)",
        "[Left][0]",
        "(1).__class__",
        "[Left] @ [Right]",
    ],
)
def test_anything_that_is_not_arithmetic_is_refused(expression):
    with pytest.raises(FormulaError):
        Formula(expression)


def test_check_formula_messages():
    columns = ["Left", "Right"]
    assert check_formula("[Left] + [Right]", columns) is None
    assert "no column called [Wrong]" in check_formula("[Left] + [Wrong]", columns)
    assert "own column" in check_formula("[Total] + 1", columns + ["Total"], own_name="Total")
    assert check_formula("[Left] +", columns)


def test_formulas_can_build_on_each_other_in_any_order():
    formulas = [("Share left", "[Left] / [Total]"), ("Total", "[Left] + [Right]")]
    values, errors = evaluate_formulas({"Left": 3.0, "Right": 1.0}, formulas)
    assert errors == {} and values["Total"] == 4 and values["Share left"] == 0.75


def test_formula_loops_and_unknown_columns_are_reported():
    values, errors = evaluate_formulas({"Left": 1.0}, [("A", "[B] + 1"), ("B", "[A] + 1"), ("C", "[Nope] * 2"), ("D", "[Left] *")])
    assert set(errors) == {"A", "B", "C", "D"}
    assert all(math.isnan(values[name]) for name in "ABCD")


def test_id_column_modes():
    path = "/data/day 3/fish07_trialB_2024-05-01.mp4"
    assert make_id(IdColumn(), path) == "fish07_trialB_2024-05-01"
    assert make_id(IdColumn(mode="name_ext"), path) == "fish07_trialB_2024-05-01.mp4"
    assert make_id(IdColumn(mode="folder_name"), path) == "day 3/fish07_trialB_2024-05-01"
    assert make_id(IdColumn(mode="pieces", delimiters="_", pieces="1"), path) == "fish07"
    assert make_id(IdColumn(mode="pieces", delimiters="_", pieces="1-2"), path) == "fish07_trialB"
    assert make_id(IdColumn(mode="pieces", delimiters="_", pieces="2,last"), path) == "trialB_2024-05-01"
    assert make_id(IdColumn(mode="pieces", delimiters="_-", pieces="3"), path) == "2024"
    assert make_id(IdColumn(mode="pieces", delimiters="_", pieces="9"), path) == "fish07_trialB_2024-05-01"
    assert make_id(IdColumn(mode="between", after="fish", before="_"), path) == "07"
    assert make_id(IdColumn(mode="between", after="trial"), path) == "B_2024-05-01"
    assert make_id(IdColumn(mode="between", after="missing"), path) == "fish07_trialB_2024-05-01"
    assert make_id(IdColumn(mode="regex", pattern=r"fish(\d+)"), path) == "07"
    assert make_id(IdColumn(mode="regex", pattern=r"trial[A-Z]"), path) == "trialB"
    assert make_id(IdColumn(mode="regex", pattern=r"(unclosed"), path) == "fish07_trialB_2024-05-01"
    assert make_id(IdColumn(), r"C:\videos\clip.one.avi") == "clip.one"

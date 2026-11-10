"""Writing the results to an Excel workbook.

The first sheet, "Scores", is the table the user asked for: the file label in
the left-most column and one column per score. Further sheets document how the
numbers came about so that a result can be checked and repeated.
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import os

from . import APP_NAME, __version__
from .batch import ResultsTable, VideoResult, animal_label, build_table
from .geometry import DIRECTION_LABELS
from .idcolumn import ID_MODE_LABELS, make_id
from .model import Layout, Rule

MEASURE_LABELS = {
    ("line", "count"): "add points for each crossing",
    ("line", "first_time"): "time of the first crossing (s)",
    ("roi", "time"): "time inside (s)",
    ("roi", "entries"): "add points for each entry",
    ("roi", "exits"): "add points for each exit",
    ("roi", "first_time"): "time of the first entry (s)",
}

COUNT_MODE_LABELS = {
    "every": "every time",
    "first": "first time only",
    "first_in_group": "only if it is the first choice in its group",
}


def describe_rule(rule: Rule, shape_kind: str) -> str:
    """One readable sentence for a rule."""
    part = rule.bodypart or "(no body part)"
    column = rule.column or "(no column)"
    if shape_kind == "line":
        way = DIRECTION_LABELS.get(rule.direction, rule.direction)
        if rule.measure == "first_time":
            return f"Time until {part} first crosses {way} → {column}"
        text = f"{part} crosses {way}: add {rule.points:g} to {column}"
    elif rule.measure == "time":
        scale = "" if rule.points == 1 else f" × {rule.points:g}"
        return f"Seconds {part} spends inside{scale} → {column}"
    elif rule.measure == "first_time":
        return f"Time until {part} first enters → {column}"
    else:
        verb = "enters" if rule.measure == "entries" else "leaves"
        text = f"{part} {verb}: add {rule.points:g} to {column}"
    if rule.count_mode == "first":
        text += " (first time only)"
    elif rule.count_mode == "first_in_group":
        text += f" (only if first choice in group “{rule.group}”)"
    return text


def _cell(value):
    """Turn a number into something Excel shows well; empty for missing."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
        return round(value, 6)
    return value


def _append(ws, row: list) -> None:
    """Append a row; text that starts with "=" stays text instead of becoming a formula."""
    ws.append(row)
    if any(isinstance(v, str) and v.startswith("=") for v in row):
        for cell in ws[ws.max_row]:
            if isinstance(cell.value, str) and cell.value.startswith("="):
                cell.data_type = "s"


def _style_header(ws, n_columns: int, freeze: str) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fill = PatternFill("solid", fgColor="E8EEF4")
    for col in range(1, n_columns + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = freeze
    if ws.max_row >= 1 and n_columns >= 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(n_columns)}{max(1, ws.max_row)}"


def _fit_columns(ws, minimum: int = 8, maximum: int = 60) -> None:
    from openpyxl.utils import get_column_letter

    widths: dict[int, int] = {}
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 400)):
        for cell in row:
            if cell.value is None:
                continue
            widths[cell.column] = max(widths.get(cell.column, 0), len(str(cell.value)))
    for column, width in widths.items():
        ws.column_dimensions[get_column_letter(column)].width = max(minimum, min(maximum, width + 2))


def table_as_rows(table: ResultsTable) -> tuple[list[str], list[list]]:
    """Header and data rows of the Scores sheet (also used for CSV export)."""
    header = [table.id_header]
    if table.show_animal:
        header.append("Animal")
    header += table.columns
    rows = []
    for row in table.rows:
        line = [row.video_id]
        if table.show_animal:
            line.append(row.animal)
        line += [_cell(row.values.get(c)) for c in table.columns]
        rows.append(line)
    return header, rows


def export_csv(path: str, results: list[VideoResult], layout: Layout) -> None:
    table = build_table(results, layout)
    header, rows = table_as_rows(table)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def export_xlsx(path: str, results: list[VideoResult], layout: Layout, info: dict | None = None) -> ResultsTable:
    """Write the workbook. ``info`` holds free-form facts for the Setup sheet."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    info = dict(info or {})
    table = build_table(results, layout)
    wb = Workbook()

    # ---- Scores
    ws = wb.active
    ws.title = "Scores"
    header, rows = table_as_rows(table)
    ws.append(header)
    for row in rows:
        _append(ws, row)
    _style_header(ws, len(header), "C2" if table.show_animal else "B2")
    _fit_columns(ws)

    # ---- Events
    ev = wb.create_sheet("Events")
    ev_header = [table.id_header]
    if table.show_animal:
        ev_header.append("Animal")
    ev_header += [
        "Time (s)",
        "Frame (first = 1)",
        "Shape",
        "Shape type",
        "Event",
        "Body part",
        "Column",
        "Points added",
        "Untracked frames just before",
    ]
    ev.append(ev_header)
    for result in results:
        video_id = make_id(layout.id_column, result.source)
        for animal in result.animals:
            for event in animal.events:
                line = [video_id]
                if table.show_animal:
                    line.append(animal_label(animal.animal))
                line += [
                    round(event.time_s, 4),
                    event.frame + 1,
                    event.shape_name,
                    "line" if event.shape_kind == "line" else "region",
                    event.what,
                    event.bodypart,
                    event.column,
                    _cell(event.points),
                    event.gap_frames or None,
                ]
                _append(ev, line)
    _style_header(ev, len(ev_header), "A2")
    _fit_columns(ev)

    # ---- Quality
    used_parts = layout.used_bodyparts()
    qc = wb.create_sheet("Quality")
    qc_header = [table.id_header]
    if table.show_animal:
        qc_header.append("Animal")
    qc_header += ["Frames", "Frames per second", "Length (s)"]
    qc_header += [f"Tracked % ({part})" for part in used_parts]
    qc_header += ["Events right after a tracking gap", "Notes", "Tracking file"]
    qc.append(qc_header)
    for result in results:
        video_id = make_id(layout.id_column, result.source)
        track_name = os.path.basename(result.track_path) if result.track_path else ""
        people = result.animals or [None]
        for animal in people:
            line = [video_id]
            if table.show_animal:
                line.append(animal_label(animal.animal) if animal else "")
            line += [result.n_frames or None, _cell(round(result.fps, 4)) if result.fps else None]
            line.append(_cell(round(result.duration_s, 3)) if result.fps else None)
            for part in used_parts:
                share = animal.tracked.get(part) if animal else None
                line.append(None if share is None else round(100.0 * share, 1))
            notes = list(result.notes) + (list(animal.notes) if animal else [])
            if result.error:
                notes.insert(0, result.error)
            line += [animal.gap_events if animal else None, "; ".join(notes), track_name]
            _append(qc, line)
    _style_header(qc, len(qc_header), "B2")
    _fit_columns(qc)

    # ---- Setup
    st = wb.create_sheet("Setup")
    bold = Font(bold=True)

    def title(text: str) -> None:
        if st.max_row > 1 or st.cell(row=1, column=1).value is not None:
            st.append([])
        st.append([text])
        st.cell(row=st.max_row, column=1).font = bold

    def table_header(cells: list[str]) -> None:
        st.append(cells)
        for col in range(1, len(cells) + 1):
            st.cell(row=st.max_row, column=col).font = bold

    title(f"{APP_NAME} scoring record")
    st.append(["Created", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")])
    st.append([f"{APP_NAME} version", __version__])
    for key, value in info.items():
        st.append([str(key), "" if value is None else str(value)])
    st.append(["Videos in this file", len(results)])

    s = layout.settings
    title("Settings")
    st.append(["Likelihood cut-off", s.pcutoff])
    st.append(["Bridge tracking gaps up to (s)", s.max_gap_s])
    st.append(["Smoothing window (frames, 1 = off)", s.smooth_frames])
    st.append(["Margin past a line or edge (pixels)", s.margin_px])
    st.append(["Crossing must match body orientation", "yes" if s.use_heading else "no"])
    if s.use_heading:
        st.append(["Head part / tail part", f"{s.head_part} / {s.tail_part}"])
    st.append(["Count crossings that happen during a tracking gap", "yes" if s.count_gap_crossings else "no"])
    st.append(
        [
            "Time in a region while not tracked",
            "kept as last known (inside stays inside)" if s.roi_gap_policy == "hold" else "not counted",
        ]
    )
    st.append(["Shapes drawn on a frame of (width × height)", f"{layout.ref_size[0]} × {layout.ref_size[1]}" if layout.ref_size else ""])

    idc = layout.id_column
    title("First column")
    st.append(["Header", idc.header])
    st.append(["Built from", ID_MODE_LABELS.get(idc.mode, idc.mode)])
    if idc.mode == "pieces":
        st.append(["Split at", idc.delimiters])
        st.append(["Pieces kept", idc.pieces])
    elif idc.mode == "between":
        st.append(["After", idc.after])
        st.append(["Before", idc.before])
    elif idc.mode == "regex":
        st.append(["Pattern", idc.pattern])

    title("Rules")
    table_header(["Shape", "Shape type", "Rule", "Body part", "Measure", "Direction", "Points", "Column", "Counting", "Group"])
    for shape in layout.shapes():
        for rule in shape.rules:
            st.append(
                [
                    shape.name,
                    "line" if shape.kind == "line" else "region",
                    describe_rule(rule, shape.kind),
                    rule.bodypart,
                    MEASURE_LABELS.get((shape.kind, rule.measure), rule.measure),
                    DIRECTION_LABELS.get(rule.direction, "") if shape.kind == "line" else "",
                    rule.points if rule.measure != "first_time" else None,
                    rule.column,
                    COUNT_MODE_LABELS.get(rule.count_mode, "") if rule.is_event_rule else "",
                    rule.group if rule.count_mode == "first_in_group" and rule.is_event_rule else "",
                ]
            )

    title("Formula columns")
    table_header(["Column", "Formula", "Problem"])
    for name, expression in layout.active_formulas():
        st.append([name, expression, table.formula_errors.get(name, "")])

    title("Lines (pixels, shared position)")
    table_header(["Name", "x1", "y1", "x2", "y2"])
    for line in layout.lines:
        st.append([line.name, round(line.p1[0], 2), round(line.p1[1], 2), round(line.p2[0], 2), round(line.p2[1], 2)])

    title("Regions (pixels, shared position)")
    table_header(["Name", "Form", "Points (x, y) ..."])
    for roi in layout.rois:
        form = "ellipse (box corners)" if roi.form == "ellipse" else "polygon"
        st.append([roi.name, form] + [f"{p[0]:.1f}, {p[1]:.1f}" for p in roi.points])

    if layout.overrides:
        title("Videos with their own shape positions")
        table_header(["Video", "Shape", "Position"])
        names = {shape.id: shape.name for shape in layout.shapes()}
        for video, per in sorted(layout.overrides.items()):
            for shape_id, geom in per.items():
                if "p1" in geom:
                    where = f"({geom['p1'][0]:.1f}, {geom['p1'][1]:.1f}) to ({geom['p2'][0]:.1f}, {geom['p2'][1]:.1f})"
                else:
                    where = "; ".join(f"{p[0]:.1f}, {p[1]:.1f}" for p in geom.get("points", []))
                st.append([video, names.get(shape_id, shape_id), where])

    st.column_dimensions["A"].width = 46
    st.column_dimensions["B"].width = 28
    st.column_dimensions["C"].width = 60
    for letter in "DEFGHIJ":
        st.column_dimensions[letter].width = 22

    wb.save(path)
    return table

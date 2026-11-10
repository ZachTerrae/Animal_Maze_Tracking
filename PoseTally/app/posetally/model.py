"""The scoring setup: lines, regions, their rules, the columns and the settings.

Nothing in here knows about the user interface or about DeepLabCut. A setup is
plain data that can be saved to a file, loaded on another computer and applied
to any batch of videos.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import DIRECTIONS, ellipse_polygon
from .idcolumn import IdColumn

SETUP_VERSION = 1

# Colours that stay visible on top of video.
SHAPE_COLORS = [
    "#ffd400",
    "#00e5ff",
    "#ff5ad1",
    "#7dff5a",
    "#ff8a3d",
    "#b48cff",
    "#ff4d4d",
    "#3d9bff",
    "#00d6a1",
    "#ffe9a8",
]

LINE_MEASURES = ("count", "first_time")
ROI_MEASURES = ("time", "entries", "exits", "first_time")
COUNT_MODES = ("every", "first", "first_in_group")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def _pt(value) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


@dataclass
class Rule:
    """One thing a line or region adds to the results.

    ``measure`` says what is measured:

    * lines: ``count`` (add points per crossing) or ``first_time`` (seconds
      until the first crossing);
    * regions: ``time`` (seconds inside), ``entries``, ``exits`` (add points
      each time) or ``first_time`` (seconds until the first entry).
    """

    bodypart: str = ""
    measure: str = "count"
    direction: str = "any"  # lines only
    points: float = 1.0
    column: str = ""
    count_mode: str = "every"  # every | first | first_in_group
    group: str = ""  # used when count_mode == "first_in_group"
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict, shape_kind: str) -> Rule:
        rule = Rule()
        rule.bodypart = str(d.get("bodypart", ""))
        rule.measure = str(d.get("measure", ""))
        allowed = LINE_MEASURES if shape_kind == "line" else ROI_MEASURES
        if rule.measure not in allowed:
            rule.measure = allowed[0]
        rule.direction = str(d.get("direction", "any"))
        if rule.direction not in DIRECTIONS:
            rule.direction = "any"
        try:
            rule.points = float(d.get("points", 1.0))
        except (TypeError, ValueError):
            rule.points = 1.0
        rule.column = str(d.get("column", ""))
        rule.count_mode = str(d.get("count_mode", "every"))
        if rule.count_mode not in COUNT_MODES:
            rule.count_mode = "every"
        rule.group = str(d.get("group", ""))
        rule.id = str(d.get("id") or new_id())
        return rule

    @property
    def is_event_rule(self) -> bool:
        """True for rules that count things (as opposed to measuring time)."""
        return self.measure in ("count", "entries", "exits")


@dataclass
class LineShape:
    name: str = "Line"
    p1: tuple[float, float] = (0.0, 0.0)
    p2: tuple[float, float] = (0.0, 0.0)
    color: str = SHAPE_COLORS[0]
    rules: list[Rule] = field(default_factory=list)
    id: str = field(default_factory=new_id)

    kind = "line"

    def geometry(self) -> dict:
        return {"p1": [self.p1[0], self.p1[1]], "p2": [self.p2[0], self.p2[1]]}

    def set_geometry(self, g: dict) -> None:
        self.p1, self.p2 = _pt(g["p1"]), _pt(g["p2"])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            **self.geometry(),
            "rules": [r.to_dict() for r in self.rules],
        }

    @staticmethod
    def from_dict(d: dict) -> LineShape:
        return LineShape(
            name=str(d.get("name", "Line")),
            p1=_pt(d.get("p1", (0, 0))),
            p2=_pt(d.get("p2", (0, 0))),
            color=str(d.get("color", SHAPE_COLORS[0])),
            rules=[Rule.from_dict(r, "line") for r in d.get("rules", [])],
            id=str(d.get("id") or new_id()),
        )


@dataclass
class RoiShape:
    """A region of any shape.

    ``form`` is ``polygon`` (``points`` are the corners, in order) or
    ``ellipse`` (``points`` are two opposite corners of the box around it).
    Rectangles and freehand outlines are stored as polygons.
    """

    name: str = "Region"
    form: str = "polygon"
    points: list[tuple[float, float]] = field(default_factory=list)
    color: str = SHAPE_COLORS[1]
    rules: list[Rule] = field(default_factory=list)
    id: str = field(default_factory=new_id)

    kind = "roi"

    def geometry(self) -> dict:
        return {"points": [[p[0], p[1]] for p in self.points]}

    def set_geometry(self, g: dict) -> None:
        self.points = [_pt(p) for p in g["points"]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "form": self.form,
            "color": self.color,
            **self.geometry(),
            "rules": [r.to_dict() for r in self.rules],
        }

    @staticmethod
    def from_dict(d: dict) -> RoiShape:
        form = str(d.get("form", "polygon"))
        return RoiShape(
            name=str(d.get("name", "Region")),
            form=form if form in ("polygon", "ellipse") else "polygon",
            points=[_pt(p) for p in d.get("points", [])],
            color=str(d.get("color", SHAPE_COLORS[1])),
            rules=[Rule.from_dict(r, "roi") for r in d.get("rules", [])],
            id=str(d.get("id") or new_id()),
        )


def region_outline(form: str, points) -> np.ndarray:
    """The polygon that is actually tested for a region."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if form == "ellipse":
        if len(pts) < 2:
            return np.zeros((0, 2))
        return ellipse_polygon(pts[0], pts[1])
    return np.asarray(pts, dtype=float).reshape(-1, 2)


@dataclass
class ScoringSettings:
    """How tracking is cleaned up and how events are decided."""

    pcutoff: float = 0.6  # ignore points DeepLabCut is less sure about than this
    max_gap_s: float = 0.2  # bridge tracking gaps up to this long (seconds)
    smooth_frames: int = 5  # median filter window in frames; 1 turns it off
    margin_px: float = 4.0  # how far past a line or edge before it counts
    use_heading: bool = False  # also require the body to point the way it moved
    head_part: str = ""
    tail_part: str = ""
    roi_gap_policy: str = "hold"  # hold | exclude: time while the point is not tracked
    count_gap_crossings: bool = True  # count a crossing that happened while not tracked
    default_fps: float = 30.0  # only used when a video does not report its frame rate
    prefer_filtered: bool = False  # use DeepLabCut's *_filtered files when they exist

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> ScoringSettings:
        out = ScoringSettings()
        for key, value in (d or {}).items():
            if not hasattr(out, key):
                continue
            current = getattr(out, key)
            try:
                if isinstance(current, bool):
                    setattr(out, key, bool(value))
                elif isinstance(current, int):
                    setattr(out, key, int(value))
                elif isinstance(current, float):
                    setattr(out, key, float(value))
                else:
                    setattr(out, key, str(value))
            except (TypeError, ValueError):
                pass
        out.pcutoff = min(1.0, max(0.0, out.pcutoff))
        out.max_gap_s = max(0.0, out.max_gap_s)
        out.smooth_frames = max(1, out.smooth_frames)
        out.margin_px = max(0.0, out.margin_px)
        if out.roi_gap_policy not in ("hold", "exclude"):
            out.roi_gap_policy = "hold"
        if not out.default_fps > 0:
            out.default_fps = 30.0
        return out


@dataclass
class FormulaColumn:
    name: str = ""
    expression: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolvedLine:
    shape: LineShape
    p1: tuple[float, float]
    p2: tuple[float, float]
    custom: bool = False  # this video uses its own position


@dataclass
class ResolvedRoi:
    shape: RoiShape
    points: list[tuple[float, float]]
    outline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    custom: bool = False


@dataclass
class ResolvedShapes:
    lines: list[ResolvedLine] = field(default_factory=list)
    rois: list[ResolvedRoi] = field(default_factory=list)
    scaled: bool = False  # shared positions were stretched to fit this frame size


_BAD_COLUMN_CHARS = re.compile(r"[\[\]\r\n\t]")


def clean_column_name(name: str) -> str:
    """Column names may not contain square brackets or line breaks."""
    return _BAD_COLUMN_CHARS.sub(" ", str(name)).strip()


@dataclass
class Layout:
    """Everything the user sets up: shapes, rules, columns and settings."""

    lines: list[LineShape] = field(default_factory=list)
    rois: list[RoiShape] = field(default_factory=list)
    formulas: list[FormulaColumn] = field(default_factory=list)
    column_order: list[str] = field(default_factory=list)
    id_column: IdColumn = field(default_factory=IdColumn)
    settings: ScoringSettings = field(default_factory=ScoringSettings)
    # frame size (width, height) the shared positions were drawn on
    ref_size: tuple[int, int] | None = None
    # per-video positions: {video name: {shape id: geometry}}
    overrides: dict[str, dict[str, dict]] = field(default_factory=dict)

    # -- shapes -----------------------------------------------------------------

    def shapes(self) -> list:
        return list(self.lines) + list(self.rois)

    def find_shape(self, shape_id: str):
        for shape in self.shapes():
            if shape.id == shape_id:
                return shape
        return None

    def remove_shape(self, shape_id: str) -> None:
        self.lines = [s for s in self.lines if s.id != shape_id]
        self.rois = [s for s in self.rois if s.id != shape_id]
        for per_video in self.overrides.values():
            per_video.pop(shape_id, None)
        self.overrides = {k: v for k, v in self.overrides.items() if v}
        self.sync_columns()

    def next_name(self, kind: str) -> str:
        base = "Line" if kind == "line" else "Region"
        taken = {s.name for s in self.shapes()}
        number = 1
        while f"{base} {number}" in taken:
            number += 1
        return f"{base} {number}"

    def next_color(self) -> str:
        used = [s.color for s in self.shapes()]
        for color in SHAPE_COLORS:
            if color not in used:
                return color
        return SHAPE_COLORS[len(used) % len(SHAPE_COLORS)]

    def used_bodyparts(self) -> list[str]:
        out: list[str] = []
        for shape in self.shapes():
            for rule in shape.rules:
                if rule.bodypart and rule.bodypart not in out:
                    out.append(rule.bodypart)
        return out

    def has_rules(self) -> bool:
        return any(shape.rules for shape in self.shapes())

    def groups(self) -> list[str]:
        out: list[str] = []
        for shape in self.shapes():
            for rule in shape.rules:
                if rule.count_mode == "first_in_group" and rule.group and rule.group not in out:
                    out.append(rule.group)
        return out

    # -- positions --------------------------------------------------------------

    def _scale(self, frame_size) -> tuple[float, float]:
        if not self.ref_size or not frame_size:
            return (1.0, 1.0)
        rw, rh = self.ref_size
        fw, fh = frame_size
        if rw <= 0 or rh <= 0 or fw <= 0 or fh <= 0:
            return (1.0, 1.0)
        return (fw / rw, fh / rh)

    def resolve(self, video_key: str | None, frame_size=None) -> ResolvedShapes:
        """Positions of every shape for one video, in that video's pixels.

        Shared positions are stretched if this video's frame size differs from
        the one they were drawn on. A video with its own position for a shape
        uses that position unchanged.
        """
        sx, sy = self._scale(frame_size)
        per_video = self.overrides.get(video_key or "", {})
        out = ResolvedShapes(scaled=(abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9))
        for line in self.lines:
            own = per_video.get(line.id)
            if own and "p1" in own and "p2" in own:
                out.lines.append(ResolvedLine(line, _pt(own["p1"]), _pt(own["p2"]), True))
            else:
                p1 = (line.p1[0] * sx, line.p1[1] * sy)
                p2 = (line.p2[0] * sx, line.p2[1] * sy)
                out.lines.append(ResolvedLine(line, p1, p2, False))
        for roi in self.rois:
            own = per_video.get(roi.id)
            if own and "points" in own:
                pts = [_pt(p) for p in own["points"]]
                custom = True
            else:
                pts = [(p[0] * sx, p[1] * sy) for p in roi.points]
                custom = False
            out.rois.append(ResolvedRoi(roi, pts, region_outline(roi.form, pts), custom))
        return out

    def store_geometry(self, shape_id: str, geometry: dict, video_key: str | None, frame_size, only_this_video: bool):
        """Save a position that was edited while looking at one video."""
        shape = self.find_shape(shape_id)
        if shape is None:
            return
        if only_this_video and video_key:
            self.overrides.setdefault(video_key, {})[shape_id] = geometry
            return
        if self.ref_size is None and frame_size:
            self.ref_size = (int(frame_size[0]), int(frame_size[1]))
        sx, sy = self._scale(frame_size)
        if "p1" in geometry:
            shape.set_geometry(
                {
                    "p1": [geometry["p1"][0] / sx, geometry["p1"][1] / sy],
                    "p2": [geometry["p2"][0] / sx, geometry["p2"][1] / sy],
                }
            )
        else:
            shape.set_geometry({"points": [[p[0] / sx, p[1] / sy] for p in geometry["points"]]})
        if video_key and video_key in self.overrides:
            self.overrides[video_key].pop(shape_id, None)
            if not self.overrides[video_key]:
                del self.overrides[video_key]

    def clear_override(self, video_key: str, shape_id: str | None = None) -> None:
        if video_key not in self.overrides:
            return
        if shape_id is None:
            del self.overrides[video_key]
            return
        self.overrides[video_key].pop(shape_id, None)
        if not self.overrides[video_key]:
            del self.overrides[video_key]

    def has_override(self, video_key: str | None, shape_id: str | None = None) -> bool:
        per_video = self.overrides.get(video_key or "", {})
        if shape_id is None:
            return bool(per_video)
        return shape_id in per_video

    # -- columns ----------------------------------------------------------------

    def score_columns(self) -> list[str]:
        out: list[str] = []
        for shape in self.shapes():
            for rule in shape.rules:
                if rule.column and rule.column not in out:
                    out.append(rule.column)
        return out

    def formula_names(self) -> list[str]:
        return [f.name for f in self.formulas if f.name]

    def columns(self) -> list[str]:
        """Every column in the order it will appear in the results."""
        scores = self.score_columns()
        existing = scores + [n for n in self.formula_names() if n not in scores]
        ordered = [c for c in self.column_order if c in existing]
        ordered += [c for c in existing if c not in ordered]
        return ordered

    def sync_columns(self) -> None:
        self.column_order = self.columns()

    def is_formula(self, name: str) -> bool:
        return name in self.formula_names() and name not in self.score_columns()

    def find_formula(self, name: str) -> FormulaColumn | None:
        for formula in self.formulas:
            if formula.name == name:
                return formula
        return None

    def active_formulas(self) -> list[tuple[str, str]]:
        """Formula columns that are not hidden by a score column of the same name."""
        scores = set(self.score_columns())
        return [(f.name, f.expression) for f in self.formulas if f.name and f.name not in scores]

    def rename_column(self, old: str, new: str) -> None:
        new = clean_column_name(new)
        if not new or new == old:
            return
        for shape in self.shapes():
            for rule in shape.rules:
                if rule.column == old:
                    rule.column = new
        for formula in self.formulas:
            if formula.name == old:
                formula.name = new
            formula.expression = formula.expression.replace(f"[{old}]", f"[{new}]")
        self.column_order = [new if c == old else c for c in self.column_order]
        self.sync_columns()

    def unique_column_name(self, base: str) -> str:
        base = clean_column_name(base) or "Column"
        taken = set(self.columns())
        if base not in taken:
            return base
        number = 2
        while f"{base} {number}" in taken:
            number += 1
        return f"{base} {number}"

    # -- checks -----------------------------------------------------------------

    def problems(self, bodyparts: list[str] | None = None) -> list[str]:
        """Things the user should fix before trusting the results."""
        from .formulas import check_formula

        out: list[str] = []
        known = set(bodyparts) if bodyparts else None
        for shape in self.shapes():
            if not shape.rules:
                out.append(f"“{shape.name}” has no scoring rule yet, so it adds nothing to the results.")
            for rule in shape.rules:
                if not rule.bodypart:
                    out.append(f"A rule on “{shape.name}” has no body part selected.")
                elif known is not None and rule.bodypart not in known:
                    out.append(f"A rule on “{shape.name}” uses “{rule.bodypart}”, which is not tracked in this project.")
                if not rule.column:
                    out.append(f"A rule on “{shape.name}” has no column name.")
                if rule.count_mode == "first_in_group" and not rule.group:
                    out.append(f"A “first choice” rule on “{shape.name}” has no group name.")
        columns = self.columns()
        scores = set(self.score_columns())
        for formula in self.formulas:
            if not formula.name:
                out.append("A formula column has no name.")
                continue
            if formula.name in scores:
                out.append(f"The formula column “{formula.name}” has the same name as a score column.")
                continue
            error = check_formula(formula.expression, columns, formula.name)
            if error:
                out.append(f"Formula “{formula.name}”: {error}")
        if self.settings.use_heading and not (self.settings.head_part and self.settings.tail_part):
            out.append("Body orientation is switched on in the settings, but the head and tail parts are not chosen.")
        return out

    # -- saving -----------------------------------------------------------------

    def to_dict(self) -> dict:
        self.sync_columns()
        return {
            "version": SETUP_VERSION,
            "lines": [s.to_dict() for s in self.lines],
            "regions": [s.to_dict() for s in self.rois],
            "formulas": [f.to_dict() for f in self.formulas],
            "column_order": list(self.column_order),
            "id_column": self.id_column.to_dict(),
            "settings": self.settings.to_dict(),
            "ref_size": list(self.ref_size) if self.ref_size else None,
            "overrides": self.overrides,
        }

    @staticmethod
    def from_dict(d: dict | None) -> Layout:
        d = d or {}
        layout = Layout()
        layout.lines = [LineShape.from_dict(x) for x in d.get("lines", [])]
        layout.rois = [RoiShape.from_dict(x) for x in d.get("regions", d.get("rois", []))]
        layout.formulas = [
            FormulaColumn(clean_column_name(x.get("name", "")), str(x.get("expression", "")))
            for x in d.get("formulas", [])
        ]
        layout.column_order = [str(c) for c in d.get("column_order", [])]
        layout.id_column = IdColumn.from_dict(d.get("id_column"))
        layout.settings = ScoringSettings.from_dict(d.get("settings"))
        ref = d.get("ref_size")
        if ref and len(ref) == 2 and ref[0] and ref[1]:
            layout.ref_size = (int(ref[0]), int(ref[1]))
        overrides = d.get("overrides") or {}
        if isinstance(overrides, dict):
            layout.overrides = {
                str(video): {str(sid): dict(geom) for sid, geom in per.items() if isinstance(geom, dict)}
                for video, per in overrides.items()
                if isinstance(per, dict)
            }
        layout.sync_columns()
        return layout

    def copy(self) -> Layout:
        return Layout.from_dict(self.to_dict())

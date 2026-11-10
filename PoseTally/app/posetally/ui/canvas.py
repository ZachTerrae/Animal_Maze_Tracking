"""The drawing surface: a video frame with lines and regions on top of it.

All positions are in video pixels. The view can be zoomed and panned freely;
line widths, handles and labels keep the same size on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..geometry import distance_point_to_segment, ellipse_polygon, line_frame, points_in_polygon, polygon_area, polygon_centroid, simplify_path

Qt = QtCore.Qt
Signal = QtCore.Signal

HANDLE_PX = 9.0  # size of the square grab handles on screen
HIT_PX = 7.0  # how close (on screen) a click must be to count as on a shape
TOOLS = ("select", "line", "polygon", "rectangle", "ellipse", "freehand")

TOOL_HINTS = {
    "select": "Click a line or region to edit its scoring. Drag it to move it; drag a square handle to reshape it. "
    "Scroll to zoom, drag the background to pan.",
    "line": "Press where the line should start, drag, and release where it should end.",
    "polygon": "Click each corner of the region. Double-click, press Enter, or click the first corner to finish. Esc cancels.",
    "rectangle": "Drag from one corner of the rectangle to the opposite corner.",
    "ellipse": "Drag out the box that the oval should fill.",
    "freehand": "Hold the mouse button down and draw around the region. It closes when you let go.",
}


@dataclass
class ShapeSpec:
    """What the canvas needs to know to draw one line or region."""

    id: str
    form: str  # "line" | "polygon" | "ellipse"
    points: list[tuple[float, float]]
    color: str = "#ffd400"
    label: str = ""
    custom: bool = False  # this video has its own position for the shape
    arrow_sides: tuple[int, ...] = ()  # lines: which way scored crossings travel
    has_rules: bool = True


@dataclass
class OverlayDot:
    x: float
    y: float
    color: QtGui.QColor
    ring: QtGui.QColor
    solid: bool = True  # False: DeepLabCut was unsure about this point


@dataclass
class OverlayData:
    dots: list[OverlayDot] = field(default_factory=list)
    bones: list[tuple[float, float, float, float]] = field(default_factory=list)


def _qp(point) -> QtCore.QPointF:
    return QtCore.QPointF(float(point[0]), float(point[1]))


def _ellipse_corners(points) -> list[tuple[float, float]]:
    (x0, y0), (x1, y1) = points[0], points[1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def geometry_of(form: str, points) -> dict:
    pts = [[float(p[0]), float(p[1])] for p in points]
    if form == "line":
        return {"p1": pts[0], "p2": pts[1]}
    return {"points": pts}


class ShapeGraphic(QtWidgets.QGraphicsItem):
    """Draws one line or region."""

    def __init__(self, spec: ShapeSpec):
        super().__init__()
        self.spec = spec
        self.selected = False
        self.setZValue(12 if spec.form == "line" else 10)

    def set_spec(self, spec: ShapeSpec) -> None:
        self.prepareGeometryChange()
        self.spec = spec
        self.update()

    def set_points(self, points) -> None:
        self.prepareGeometryChange()
        self.spec.points = [(float(p[0]), float(p[1])) for p in points]
        self.update()

    def handle_points(self) -> list[tuple[float, float]]:
        if self.spec.form == "ellipse" and len(self.spec.points) >= 2:
            return _ellipse_corners(self.spec.points)
        return list(self.spec.points)

    def boundingRect(self) -> QtCore.QRectF:
        if not self.spec.points:
            return QtCore.QRectF()
        xs = [p[0] for p in self.spec.points]
        ys = [p[1] for p in self.spec.points]
        # generous: labels, arrows and handles are drawn at a fixed screen size
        return QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)).adjusted(-4000, -4000, 4000, 4000)

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        pts = self.spec.points
        if len(pts) < 2:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        scale = abs(painter.worldTransform().m11()) or 1.0
        px = 1.0 / scale
        color = QtGui.QColor(self.spec.color)
        halo = QtGui.QPen(QtGui.QColor(0, 0, 0, 175), 5.0 if self.selected else 4.2)
        halo.setCosmetic(True)
        pen = QtGui.QPen(color, 3.0 if self.selected else 2.2)
        pen.setCosmetic(True)
        if self.spec.custom:
            pen.setStyle(Qt.DashLine)
        fill = QtGui.QColor(color)
        fill.setAlpha(75 if self.selected else 38)

        if self.spec.form == "line":
            a, b = _qp(pts[0]), _qp(pts[1])
            painter.setPen(halo)
            painter.drawLine(a, b)
            painter.setPen(pen)
            painter.drawLine(a, b)
            self._paint_arrows(painter, px, color)
            try:
                _, _, normal = line_frame(pts[0], pts[1])
            except ValueError:
                normal = (0.0, -1.0)
            anchor = QtCore.QPointF(
                pts[0][0] + 0.2 * (pts[1][0] - pts[0][0]) + normal[0] * 15 * px,
                pts[0][1] + 0.2 * (pts[1][1] - pts[0][1]) + normal[1] * 15 * px,
            )
        elif self.spec.form == "ellipse":
            rect = QtCore.QRectF(_qp(pts[0]), _qp(pts[1])).normalized()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(halo)
            painter.drawEllipse(rect)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawEllipse(rect)
            if self.selected:
                guide = QtGui.QPen(QtGui.QColor(255, 255, 255, 150), 1.0, Qt.DotLine)
                guide.setCosmetic(True)
                painter.setPen(guide)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
            anchor = QtCore.QPointF(rect.center().x(), rect.top() + 15 * px)
        else:
            polygon = QtGui.QPolygonF([_qp(p) for p in pts])
            painter.setBrush(Qt.NoBrush)
            painter.setPen(halo)
            painter.drawPolygon(polygon)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawPolygon(polygon)
            # near the top edge, so that a line through the middle does not cover the name
            top = min(p[1] for p in pts)
            anchor = QtCore.QPointF(polygon_centroid(pts)[0], top + 15 * px)

        if self.selected:
            size = HANDLE_PX * px
            border = QtGui.QPen(QtGui.QColor(0, 0, 0, 220), 1.2)
            border.setCosmetic(True)
            painter.setPen(border)
            painter.setBrush(QtGui.QColor(255, 255, 255))
            for hx, hy in self.handle_points():
                painter.drawRect(QtCore.QRectF(hx - size / 2, hy - size / 2, size, size))
        self._paint_label(painter, anchor, color)

    def _paint_arrows(self, painter: QtGui.QPainter, px: float, color: QtGui.QColor) -> None:
        if not self.spec.arrow_sides:
            return
        try:
            _, u, n = line_frame(self.spec.points[0], self.spec.points[1])
        except ValueError:
            return
        mx = (self.spec.points[0][0] + self.spec.points[1][0]) / 2.0
        my = (self.spec.points[0][1] + self.spec.points[1][1]) / 2.0
        for side in self.spec.arrow_sides:
            dx, dy = side * n[0], side * n[1]
            start = QtCore.QPointF(mx + dx * 5 * px, my + dy * 5 * px)
            tip = QtCore.QPointF(mx + dx * 27 * px, my + dy * 27 * px)
            back = QtCore.QPointF(mx + dx * 17 * px, my + dy * 17 * px)
            wing_a = QtCore.QPointF(back.x() + u[0] * 6 * px, back.y() + u[1] * 6 * px)
            wing_b = QtCore.QPointF(back.x() - u[0] * 6 * px, back.y() - u[1] * 6 * px)
            for width, tone in ((5.0, QtGui.QColor(0, 0, 0, 175)), (2.2, color)):
                pen = QtGui.QPen(tone, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(start, tip)
                painter.drawLine(tip, wing_a)
                painter.drawLine(tip, wing_b)

    def _paint_label(self, painter: QtGui.QPainter, anchor: QtCore.QPointF, color: QtGui.QColor) -> None:
        text = self.spec.label
        if not text:
            return
        if not self.spec.has_rules:
            text += "  (no score yet)"
        device = painter.worldTransform().map(anchor)
        painter.save()
        painter.resetTransform()
        font = QtGui.QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        metrics = QtGui.QFontMetricsF(font)
        width = metrics.horizontalAdvance(text) + 12
        height = metrics.height() + 4
        rect = QtCore.QRectF(device.x() - width / 2, device.y() - height / 2, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 170))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()


class OverlayGraphic(QtWidgets.QGraphicsItem):
    """The tracked body parts on the frame being shown."""

    def __init__(self):
        super().__init__()
        self.data = OverlayData()
        self.bounds = QtCore.QRectF(0, 0, 1, 1)
        self.setZValue(30)

    def set_data(self, data: OverlayData) -> None:
        self.data = data
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self.bounds

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        if not self.data.dots and not self.data.bones:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        scale = abs(painter.worldTransform().m11()) or 1.0
        px = 1.0 / scale
        bone_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 150), 1.4)
        bone_pen.setCosmetic(True)
        painter.setPen(bone_pen)
        for x1, y1, x2, y2 in self.data.bones:
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        radius = 4.2 * px
        for dot in self.data.dots:
            ring = QtGui.QPen(dot.ring, 1.4)
            ring.setCosmetic(True)
            painter.setPen(ring)
            if dot.solid:
                painter.setBrush(dot.color)
            else:
                painter.setBrush(Qt.NoBrush)
                faint = QtGui.QPen(dot.color, 1.4, Qt.DotLine)
                faint.setCosmetic(True)
                painter.setPen(faint)
            painter.drawEllipse(QtCore.QPointF(dot.x, dot.y), radius, radius)


class PathsGraphic(QtWidgets.QGraphicsItem):
    """The whole path an animal took, drawn faintly, to help place shapes."""

    def __init__(self):
        super().__init__()
        self.paths: list[tuple[QtGui.QPainterPath, QtGui.QColor]] = []
        self.bounds = QtCore.QRectF(0, 0, 1, 1)
        self.setZValue(5)

    def set_paths(self, tracks: list[tuple[np.ndarray, QtGui.QColor]]) -> None:
        self.paths = []
        for xy, color in tracks:
            path = QtGui.QPainterPath()
            if xy is None or len(xy) == 0:
                continue
            step = max(1, len(xy) // 6000)
            pen_down = False
            for x, y in xy[::step]:
                if not (math.isfinite(x) and math.isfinite(y)):
                    pen_down = False
                    continue
                if pen_down:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                    pen_down = True
            self.paths.append((path, color))
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self.bounds

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        for path, color in self.paths:
            tone = QtGui.QColor(color)
            tone.setAlpha(150)
            pen = QtGui.QPen(tone, 1.3)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)


class Canvas(QtWidgets.QGraphicsView):
    shapeDrawn = Signal(str, dict)  # form, geometry
    geometryEdited = Signal(str, dict)  # shape id, geometry
    selectionChanged = Signal(object)  # shape id or None
    deleteRequested = Signal(str)
    toolChanged = Signal(str)
    cursorMoved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QColor("#1b1d21"))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)

        self._frame_size = (640, 480)
        self._backdrop = self._scene.addRect(0, 0, 640, 480, QtGui.QPen(Qt.NoPen), QtGui.QBrush(QtGui.QColor("#2a2d33")))
        self._backdrop.setZValue(-1)
        self._frame_item = self._scene.addPixmap(QtGui.QPixmap())
        self._frame_item.setZValue(0)
        self._frame_item.setTransformationMode(Qt.SmoothTransformation)
        self._paths = PathsGraphic()
        self._scene.addItem(self._paths)
        self._overlay = OverlayGraphic()
        self._scene.addItem(self._overlay)
        draft_pen = QtGui.QPen(QtGui.QColor("#ffffff"), 2.0, Qt.DashLine)
        draft_pen.setCosmetic(True)
        self._draft_item = self._scene.addPath(QtGui.QPainterPath(), draft_pen)
        self._draft_item.setZValue(40)

        self._graphics: dict[str, ShapeGraphic] = {}
        self._order: list[str] = []
        self._selected: str | None = None
        self._tool = "select"
        self._draft: list[tuple[float, float]] = []
        self._draft_cursor: tuple[float, float] | None = None
        self._drag: dict | None = None
        self._auto_fit = True
        self._next_color = "#ffffff"
        self._apply_scene_rect()

    # ------------------------------------------------------------------ content

    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def set_frame_size(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == self._frame_size:
            return
        self._frame_size = (width, height)
        self._backdrop.setRect(0, 0, width, height)
        self._apply_scene_rect()
        self._auto_fit = True
        self.fit()

    def _apply_scene_rect(self) -> None:
        w, h = self._frame_size
        rect = QtCore.QRectF(-w, -h, 3 * w, 3 * h)  # room to pan and to draw past the edge
        self._scene.setSceneRect(rect)
        self._overlay.bounds = rect
        self._paths.bounds = rect

    def set_frame(self, image: QtGui.QImage | None) -> None:
        if image is None or image.isNull():
            self._frame_item.setPixmap(QtGui.QPixmap())
            return
        self._frame_item.setPixmap(QtGui.QPixmap.fromImage(image))

    def set_overlay(self, data: OverlayData) -> None:
        self._overlay.set_data(data)

    def set_paths(self, tracks: list[tuple[np.ndarray, QtGui.QColor]]) -> None:
        self._paths.set_paths(tracks)

    def set_draft_color(self, color: str) -> None:
        self._next_color = color
        pen = QtGui.QPen(QtGui.QColor(color), 2.0, Qt.DashLine)
        pen.setCosmetic(True)
        self._draft_item.setPen(pen)

    def set_shapes(self, specs: list[ShapeSpec]) -> None:
        wanted = {spec.id for spec in specs}
        for shape_id in list(self._graphics):
            if shape_id not in wanted:
                self._scene.removeItem(self._graphics.pop(shape_id))
        self._order = [spec.id for spec in specs]
        for spec in specs:
            graphic = self._graphics.get(spec.id)
            if graphic is None:
                graphic = ShapeGraphic(spec)
                self._graphics[spec.id] = graphic
                self._scene.addItem(graphic)
            elif not (self._drag and self._drag.get("shape_id") == spec.id):
                graphic.set_spec(spec)
            graphic.selected = spec.id == self._selected
        if self._selected not in self._graphics:
            self._selected = None
        self.viewport().update()

    def shape_points(self, shape_id: str) -> list[tuple[float, float]] | None:
        graphic = self._graphics.get(shape_id)
        return list(graphic.spec.points) if graphic else None

    def selected_id(self) -> str | None:
        return self._selected

    def select(self, shape_id: str | None, announce: bool = False) -> None:
        if shape_id not in self._graphics:
            shape_id = None
        changed = shape_id != self._selected
        self._selected = shape_id
        for sid, graphic in self._graphics.items():
            graphic.selected = sid == shape_id
        self.viewport().update()
        if changed and announce:
            self.selectionChanged.emit(shape_id)

    # --------------------------------------------------------------------- view

    def view_scale(self) -> float:
        return abs(self.transform().m11()) or 1.0

    def fit(self) -> None:
        w, h = self._frame_size
        margin = 0.02 * max(w, h)
        self.resetTransform()
        self.fitInView(QtCore.QRectF(-margin, -margin, w + 2 * margin, h + 2 * margin), Qt.KeepAspectRatio)
        self._auto_fit = True

    def zoom(self, factor: float, around: QtCore.QPointF | None = None) -> None:
        current = self.view_scale()
        target = min(80.0, max(0.02, current * factor))
        factor = target / current
        if abs(factor - 1.0) < 1e-6:
            return
        around = around if around is not None else QtCore.QPointF(self.viewport().rect().center())
        before = self.mapToScene(around.toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(around.toPoint())
        delta = after - before
        self.translate(delta.x(), delta.y())
        self._auto_fit = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom(1.0015**delta, event.position())
        event.accept()

    # -------------------------------------------------------------------- tools

    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            tool = "select"
        self._cancel_draft()
        changed = tool != self._tool
        self._tool = tool
        self.viewport().setCursor(Qt.ArrowCursor if tool == "select" else Qt.CrossCursor)
        if changed:
            self.toolChanged.emit(tool)

    def _cancel_draft(self) -> None:
        self._draft = []
        self._draft_cursor = None
        self._draft_item.setPath(QtGui.QPainterPath())

    def is_drawing(self) -> bool:
        return bool(self._draft)

    def _update_draft(self) -> None:
        path = QtGui.QPainterPath()
        pts = list(self._draft)
        cursor = self._draft_cursor
        if not pts:
            self._draft_item.setPath(path)
            return
        if self._tool == "line" and cursor:
            path.moveTo(*pts[0])
            path.lineTo(*cursor)
        elif self._tool == "rectangle" and cursor:
            path.addRect(QtCore.QRectF(_qp(pts[0]), _qp(cursor)).normalized())
        elif self._tool == "ellipse" and cursor:
            path.addEllipse(QtCore.QRectF(_qp(pts[0]), _qp(cursor)).normalized())
        else:  # polygon, freehand
            path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)
            if cursor and self._tool == "polygon":
                path.lineTo(*cursor)
            if self._tool == "polygon":
                r = 5.0 / self.view_scale()
                path.addEllipse(_qp(pts[0]), r, r)
        self._draft_item.setPath(path)

    def _finish(self, form: str, points: list[tuple[float, float]]) -> None:
        self._cancel_draft()
        self.set_tool("select")
        self.shapeDrawn.emit(form, geometry_of(form, points))

    def _finish_polygon(self) -> None:
        pts = list(self._draft)
        # a double-click adds its point twice: drop repeats
        cleaned: list[tuple[float, float]] = []
        tolerance = 1.5 / self.view_scale()
        for p in pts:
            if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > tolerance:
                cleaned.append(p)
        if len(cleaned) >= 3 and polygon_area(cleaned) > (3.0 / self.view_scale()) ** 2:
            self._finish("polygon", cleaned)
        else:
            self._cancel_draft()

    # ------------------------------------------------------------------ hitting

    def _hit_handle(self, pos: tuple[float, float]) -> int | None:
        graphic = self._graphics.get(self._selected or "")
        if graphic is None:
            return None
        reach = (HANDLE_PX * 0.5 + 3.0) / self.view_scale()
        best, best_d = None, reach
        for index, (hx, hy) in enumerate(graphic.handle_points()):
            d = math.hypot(pos[0] - hx, pos[1] - hy)
            if d <= best_d:
                best, best_d = index, d
        return best

    def _hit_shape(self, pos: tuple[float, float]) -> str | None:
        reach = HIT_PX / self.view_scale()
        lines, regions = [], []
        for shape_id in reversed(self._order):
            graphic = self._graphics.get(shape_id)
            if graphic is None or len(graphic.spec.points) < 2:
                continue
            pts = graphic.spec.points
            if graphic.spec.form == "line":
                d = distance_point_to_segment(pos, pts[0], pts[1])
                if d <= reach:
                    lines.append((d, shape_id))
                continue
            outline = ellipse_polygon(pts[0], pts[1], 48) if graphic.spec.form == "ellipse" else np.asarray(pts, dtype=float)
            inside = bool(points_in_polygon(np.array([pos], dtype=float), outline)[0])
            near = min(
                distance_point_to_segment(pos, outline[i], outline[(i + 1) % len(outline)]) for i in range(len(outline))
            )
            if inside or near <= reach:
                regions.append((polygon_area(outline), shape_id))
        if lines:
            return min(lines)[1]
        if regions:
            return min(regions)[1]  # the smallest region under the cursor
        return None

    # -------------------------------------------------------------------- mouse

    def _scene_pos(self, event) -> tuple[float, float]:
        p = self.mapToScene(event.position().toPoint())
        return (p.x(), p.y())

    def _start_pan(self, event) -> None:
        self._drag = {"mode": "pan", "last": event.position()}
        self.viewport().setCursor(Qt.ClosedHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus()
        pos = self._scene_pos(event)
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            if event.button() == Qt.RightButton and self._tool == "polygon" and self._draft:
                self._finish_polygon()
                return
            self._start_pan(event)
            return
        if event.button() != Qt.LeftButton:
            return
        if self._tool == "select":
            handle = self._hit_handle(pos)
            if handle is not None:
                graphic = self._graphics[self._selected]
                self._drag = {
                    "mode": "handle",
                    "shape_id": self._selected,
                    "index": handle,
                    "start": pos,
                    "orig": list(graphic.spec.points),
                    "moved": False,
                }
                return
            hit = self._hit_shape(pos)
            if hit is not None:
                self.select(hit, announce=True)
                graphic = self._graphics[hit]
                self._drag = {"mode": "move", "shape_id": hit, "start": pos, "orig": list(graphic.spec.points), "moved": False}
                self.viewport().setCursor(Qt.SizeAllCursor)
                return
            self.select(None, announce=True)
            self._start_pan(event)
            return
        if self._tool == "polygon":
            if self._draft and len(self._draft) >= 3:
                first = self._draft[0]
                if math.hypot(pos[0] - first[0], pos[1] - first[1]) <= 10.0 / self.view_scale():
                    self._finish_polygon()
                    return
            self._draft.append(pos)
            self._draft_cursor = pos
            self._update_draft()
            return
        # line, rectangle, ellipse, freehand: press and drag
        self._draft = [pos]
        self._draft_cursor = pos
        self._drag = {"mode": "draw"}
        self._update_draft()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = self._scene_pos(event)
        self.cursorMoved.emit(pos[0], pos[1])
        drag = self._drag
        if drag is None:
            if self._tool == "polygon" and self._draft:
                self._draft_cursor = pos
                self._update_draft()
            elif self._tool == "select":
                if self._hit_handle(pos) is not None:
                    self.viewport().setCursor(Qt.CrossCursor)
                elif self._hit_shape(pos) is not None:
                    self.viewport().setCursor(Qt.SizeAllCursor)
                else:
                    self.viewport().setCursor(Qt.ArrowCursor)
            return
        mode = drag["mode"]
        if mode == "pan":
            delta = event.position() - drag["last"]
            drag["last"] = event.position()
            scale = self.view_scale()
            self.translate(delta.x() / scale, delta.y() / scale)
            self._auto_fit = False
        elif mode == "draw":
            self._draft_cursor = pos
            if self._tool == "freehand":
                last = self._draft[-1]
                if math.hypot(pos[0] - last[0], pos[1] - last[1]) >= 2.0 / self.view_scale():
                    self._draft.append(pos)
            self._update_draft()
        elif mode in ("move", "handle"):
            graphic = self._graphics.get(drag["shape_id"])
            if graphic is None:
                return
            dx, dy = pos[0] - drag["start"][0], pos[1] - drag["start"][1]
            if not drag["moved"] and math.hypot(dx, dy) * self.view_scale() < 2.0:
                return  # a click, not a drag
            drag["moved"] = True
            orig = drag["orig"]
            if mode == "move":
                graphic.set_points([(x + dx, y + dy) for x, y in orig])
            else:
                index = drag["index"]
                if graphic.spec.form == "ellipse":
                    (x0, y0), (x1, y1) = orig[0], orig[1]
                    if index == 0:
                        x0, y0 = pos
                    elif index == 1:
                        x1, y0 = pos
                    elif index == 2:
                        x1, y1 = pos
                    else:
                        x0, y1 = pos
                    graphic.set_points([(x0, y0), (x1, y1)])
                else:
                    points = list(orig)
                    points[index] = pos
                    graphic.set_points(points)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        drag = self._drag
        self._drag = None
        if drag is None:
            return
        mode = drag["mode"]
        pos = self._scene_pos(event)
        if mode == "pan":
            self.viewport().setCursor(Qt.ArrowCursor if self._tool == "select" else Qt.CrossCursor)
            return
        if mode == "draw":
            start = self._draft[0] if self._draft else pos
            scale = self.view_scale()
            if self._tool == "line":
                if math.hypot(pos[0] - start[0], pos[1] - start[1]) * scale >= 6.0:
                    self._finish("line", [start, pos])
                else:
                    self._cancel_draft()
            elif self._tool in ("rectangle", "ellipse"):
                if abs(pos[0] - start[0]) * scale >= 6.0 and abs(pos[1] - start[1]) * scale >= 6.0:
                    x0, x1 = sorted((start[0], pos[0]))
                    y0, y1 = sorted((start[1], pos[1]))
                    if self._tool == "rectangle":
                        self._finish("polygon", [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    else:
                        self._finish("ellipse", [(x0, y0), (x1, y1)])
                else:
                    self._cancel_draft()
            elif self._tool == "freehand":
                outline = simplify_path(self._draft + [pos], 1.5 / scale)
                if len(outline) >= 3 and polygon_area(outline) * scale * scale >= 60.0:
                    self._finish("polygon", outline)
                else:
                    self._cancel_draft()
            return
        if mode in ("move", "handle"):
            self.viewport().setCursor(Qt.ArrowCursor)
            graphic = self._graphics.get(drag["shape_id"])
            if graphic is not None and drag["moved"]:
                self.geometryEdited.emit(graphic.spec.id, geometry_of(graphic.spec.form, graphic.spec.points))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._tool == "polygon" and self._draft and event.button() == Qt.LeftButton:
            self._finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.cursorMoved.emit(float("nan"), float("nan"))

    # ----------------------------------------------------------------- keyboard

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            if self._draft:
                self._cancel_draft()
            elif self._tool != "select":
                self.set_tool("select")
            else:
                self.select(None, announce=True)
            return
        if key in (Qt.Key_Return, Qt.Key_Enter) and self._tool == "polygon" and self._draft:
            self._finish_polygon()
            return
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            if self._tool == "polygon" and self._draft:
                self._draft.pop()
                self._update_draft()
            elif self._selected:
                self.deleteRequested.emit(self._selected)
            return
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self._selected and self._tool == "select":
            step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
            dx = {Qt.Key_Left: -step, Qt.Key_Right: step}.get(key, 0.0)
            dy = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(key, 0.0)
            graphic = self._graphics[self._selected]
            graphic.set_points([(x + dx, y + dy) for x, y in graphic.spec.points])
            self.geometryEdited.emit(graphic.spec.id, geometry_of(graphic.spec.form, graphic.spec.points))
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom(1.25)
            return
        if key == Qt.Key_Minus:
            self.zoom(0.8)
            return
        if key == Qt.Key_0:
            self.fit()
            return
        super().keyPressEvent(event)

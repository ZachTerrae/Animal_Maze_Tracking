"""Colours, icons and small widgets shared by the interface.

Every icon is drawn in code, so the program needs no image files.
"""

from __future__ import annotations

import math
import sys

from PySide6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

# One colour per body part (repeats after twelve).
PART_COLORS = [
    "#ff5252", "#ffa726", "#ffee58", "#66e07a", "#26c6da", "#5c8dff",
    "#b47cff", "#ff6ec7", "#c0ca33", "#ffffff", "#8d6e63", "#90a4ae",
]  # fmt: skip

ANIMAL_RINGS = ["#000000", "#ffffff", "#ff00ff", "#00ffff", "#ffff00", "#ff8800"]


def part_color(index: int) -> QtGui.QColor:
    return QtGui.QColor(PART_COLORS[index % len(PART_COLORS)])


def animal_ring(index: int) -> QtGui.QColor:
    return QtGui.QColor(ANIMAL_RINGS[index % len(ANIMAL_RINGS)])


def is_dark(palette: QtGui.QPalette | None = None) -> bool:
    palette = palette or QtWidgets.QApplication.palette()
    return palette.color(QtGui.QPalette.Window).lightness() < 128


def muted_color() -> QtGui.QColor:
    palette = QtWidgets.QApplication.palette()
    color = QtGui.QColor(palette.color(QtGui.QPalette.WindowText))
    color.setAlpha(150)
    return color


def app_pixmap(size: int = 256) -> QtGui.QPixmap:
    """The application icon: a path crossing a scoring line."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = size / 256.0
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor("#1d2733"))
    p.drawRoundedRect(QtCore.QRectF(12 * s, 12 * s, 232 * s, 232 * s), 52 * s, 52 * s)
    # the scoring line
    pen = QtGui.QPen(QtGui.QColor("#ffd400"), 16 * s, Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QtCore.QPointF(128 * s, 44 * s), QtCore.QPointF(128 * s, 212 * s))
    # the animal's path across it
    path = QtGui.QPainterPath(QtCore.QPointF(46 * s, 170 * s))
    path.cubicTo(QtCore.QPointF(96 * s, 176 * s), QtCore.QPointF(110 * s, 96 * s), QtCore.QPointF(196 * s, 92 * s))
    pen = QtGui.QPen(QtGui.QColor("#39d0ff"), 12 * s, Qt.SolidLine, Qt.RoundCap)
    pen.setDashPattern([0.1, 2.0])
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor("#39d0ff"))
    p.drawEllipse(QtCore.QPointF(202 * s, 92 * s), 17 * s, 17 * s)
    p.end()
    return pixmap


def app_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in (16, 32, 64, 128, 256):
        icon.addPixmap(app_pixmap(size))
    return icon


def write_icon_files(folder: str) -> list[str]:
    """Save the icon as .png, .ico and .icns for shortcuts (used by the installers)."""
    import os

    os.makedirs(folder, exist_ok=True)
    written = []
    for name, size in (("PoseTally.png", 256), ("PoseTally.ico", 256), ("PoseTally.icns", 512)):
        path = os.path.join(folder, name)
        if app_pixmap(size).toImage().save(path):
            written.append(path)
    return written


def _tool_pixmap(name: str, color: QtGui.QColor, size: int = 22) -> QtGui.QPixmap:
    ratio = 2
    pixmap = QtGui.QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    if name == "select":
        arrow = QtGui.QPolygonF([QtCore.QPointF(*q) for q in ((5, 3), (5, 17), (9, 13.5), (12, 19), (14.2, 18), (11.3, 12.6), (16.5, 12.6))])
        p.setBrush(color)
        p.drawPolygon(arrow)
    elif name == "line":
        p.drawLine(QtCore.QPointF(4, 18), QtCore.QPointF(18, 4))
        p.setBrush(color)
        p.drawEllipse(QtCore.QPointF(4, 18), 2, 2)
        p.drawEllipse(QtCore.QPointF(18, 4), 2, 2)
    elif name == "polygon":
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*q) for q in ((4, 8), (11, 3), (18, 9), (15, 18), (6, 17))]))
    elif name == "rectangle":
        p.drawRect(QtCore.QRectF(4, 5.5, 14, 11))
    elif name == "ellipse":
        p.drawEllipse(QtCore.QRectF(3.5, 5, 15, 12))
    elif name == "freehand":
        path = QtGui.QPainterPath(QtCore.QPointF(4, 12))
        path.cubicTo(QtCore.QPointF(3, 3), QtCore.QPointF(13, 2), QtCore.QPointF(15, 8))
        path.cubicTo(QtCore.QPointF(22, 10), QtCore.QPointF(16, 21), QtCore.QPointF(10, 17))
        path.cubicTo(QtCore.QPointF(6, 20), QtCore.QPointF(4, 16), QtCore.QPointF(4, 12))
        p.drawPath(path)
    elif name == "fit":
        for (x, y, dx, dy) in ((4, 4, 1, 1), (18, 4, -1, 1), (4, 18, 1, -1), (18, 18, -1, -1)):
            p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 5 * dx, y))
            p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + 5 * dy))
    elif name == "play":
        p.setBrush(color)
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*q) for q in ((7, 4), (18, 11), (7, 18))]))
    elif name == "pause":
        p.setBrush(color)
        p.drawRect(QtCore.QRectF(6, 4.5, 3.4, 13))
        p.drawRect(QtCore.QRectF(12.6, 4.5, 3.4, 13))
    elif name in ("step_back", "step_forward"):
        flip = -1 if name == "step_back" else 1
        cx = 11
        p.setBrush(color)
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx - 4 * flip, 5), QtCore.QPointF(cx + 4 * flip, 11), QtCore.QPointF(cx - 4 * flip, 17)]))
        p.drawRect(QtCore.QRectF(cx + 5 * flip - 0.9, 5, 1.8, 12))
    elif name in ("prev_event", "next_event"):
        flip = -1 if name == "prev_event" else 1
        cx = 11
        p.setBrush(color)
        for offset in (-4, 3):
            p.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(cx + (offset - 3) * flip, 5.5),
                        QtCore.QPointF(cx + (offset + 3) * flip, 11),
                        QtCore.QPointF(cx + (offset - 3) * flip, 16.5),
                    ]
                )
            )
    p.end()
    return pixmap


def tool_icon(name: str) -> QtGui.QIcon:
    palette = QtWidgets.QApplication.palette()
    icon = QtGui.QIcon()
    icon.addPixmap(_tool_pixmap(name, palette.color(QtGui.QPalette.ButtonText)), QtGui.QIcon.Normal)
    disabled = QtGui.QColor(palette.color(QtGui.QPalette.ButtonText))
    disabled.setAlpha(90)
    icon.addPixmap(_tool_pixmap(name, disabled), QtGui.QIcon.Disabled)
    return icon


def color_swatch(color: str, size: int = 14) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
    p.setBrush(QtGui.QColor(color))
    p.drawRoundedRect(QtCore.QRectF(1, 1, size - 2, size - 2), 3, 3)
    p.end()
    return QtGui.QIcon(pixmap)


class NumberBox(QtWidgets.QDoubleSpinBox):
    """A number field that shows 1 as "1" and 0.25 as "0.25"."""

    def __init__(self, minimum: float, maximum: float, step: float = 1.0, decimals: int = 4, parent=None):
        super().__init__(parent)
        self.setDecimals(decimals)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def textFromValue(self, value: float) -> str:
        text = f"{value:.{self.decimals()}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text.replace(".", self.locale().decimalPoint())

    def wheelEvent(self, event):  # scrolling the panel must not change numbers by accident
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class QuietCombo(QtWidgets.QComboBox):
    """A drop-down that ignores the mouse wheel unless it has been clicked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def hint_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    color = muted_color()
    label.setStyleSheet(f"color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});")
    return label


def format_clock(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{rest:05.2f}"


def format_number(value) -> str:
    """Numbers as they should look in tables: 3, 2.5, 12.35; empty for missing."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return ""
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def main(argv=None) -> int:
    """``python -m posetally.ui.style <folder>`` writes the icon files."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m posetally.ui.style <folder>")
        return 2
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    for path in write_icon_files(argv[0]):
        print(path)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

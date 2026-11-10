"""A scrub bar that also shows when scored events happen."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt


class Timeline(QtWidgets.QWidget):
    frameChosen = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._length = 0
        self._frame = 0
        self._marks: list[tuple[int, QtGui.QColor]] = []
        self.setMinimumHeight(30)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click or drag to move through the video. Coloured ticks are scored events.")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(400, 30)

    def set_length(self, n_frames: int) -> None:
        self._length = max(0, int(n_frames))
        self._frame = min(self._frame, max(0, self._length - 1))
        self.update()

    def set_frame(self, frame: int) -> None:
        frame = max(0, min(int(frame), max(0, self._length - 1)))
        if frame != self._frame:
            self._frame = frame
            self.update()

    def set_marks(self, marks: list[tuple[int, QtGui.QColor]]) -> None:
        self._marks = list(marks)
        self.update()

    def marks(self) -> list[int]:
        return sorted({frame for frame, _ in self._marks})

    def _bar(self) -> QtCore.QRectF:
        return QtCore.QRectF(8, 9, max(1, self.width() - 16), self.height() - 18)

    def _x_for(self, frame: int) -> float:
        bar = self._bar()
        if self._length <= 1:
            return bar.left()
        return bar.left() + bar.width() * frame / (self._length - 1)

    def _frame_at(self, x: float) -> int:
        bar = self._bar()
        if self._length <= 1:
            return 0
        share = (x - bar.left()) / bar.width()
        return int(round(max(0.0, min(1.0, share)) * (self._length - 1)))

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        palette = self.palette()
        bar = self._bar()
        groove = QtGui.QColor(palette.color(QtGui.QPalette.WindowText))
        groove.setAlpha(40)
        p.setPen(Qt.NoPen)
        p.setBrush(groove)
        p.drawRoundedRect(bar, 3, 3)
        if self._length > 0:
            played = QtCore.QRectF(bar.left(), bar.top(), self._x_for(self._frame) - bar.left(), bar.height())
            tone = QtGui.QColor(palette.color(QtGui.QPalette.Highlight))
            tone.setAlpha(110)
            p.setBrush(tone)
            p.drawRoundedRect(played, 3, 3)
            for frame, color in self._marks:
                x = self._x_for(frame)
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 4))
                p.drawLine(QtCore.QPointF(x, bar.top() - 4), QtCore.QPointF(x, bar.bottom() + 4))
                p.setPen(QtGui.QPen(color, 2))
                p.drawLine(QtCore.QPointF(x, bar.top() - 4), QtCore.QPointF(x, bar.bottom() + 4))
            x = self._x_for(self._frame)
            p.setPen(QtGui.QPen(palette.color(QtGui.QPalette.WindowText), 2))
            p.drawLine(QtCore.QPointF(x, 2), QtCore.QPointF(x, self.height() - 2))
            p.setPen(Qt.NoPen)
            p.setBrush(palette.color(QtGui.QPalette.WindowText))
            p.drawEllipse(QtCore.QPointF(x, 5), 4, 4)
        p.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._length > 0:
            self.frameChosen.emit(self._frame_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton and self._length > 0:
            self.frameChosen.emit(self._frame_at(event.position().x()))

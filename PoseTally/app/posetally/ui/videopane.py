"""The middle of the window: drawing tools, the video with its overlay, and playback."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..tracks import TrackSet
from ..videoio import FrameReader
from .canvas import TOOL_HINTS, Canvas, OverlayData, OverlayDot
from .style import QuietCombo, animal_ring, format_clock, hint_label, part_color, tool_icon
from .timeline import Timeline

Qt = QtCore.Qt
Signal = QtCore.Signal

TOOL_BUTTONS = [
    ("select", "Select", "Select, move and reshape (Esc)"),
    ("line", "Line", "Draw a scoring line"),
    ("polygon", "Region", "Draw a region corner by corner"),
    ("rectangle", "Rectangle", "Draw a rectangular region"),
    ("ellipse", "Oval", "Draw an oval or circular region"),
    ("freehand", "Freehand", "Draw a region of any shape by hand"),
]

SPEEDS = [("¼×", 0.25), ("½×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0), ("8×", 8.0)]


class VideoPane(QtWidgets.QWidget):
    frameChanged = Signal(int)
    positionsModeChanged = Signal(bool)  # True: moving a shape changes only this video
    pathPartChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = Canvas(self)
        self.timeline = Timeline(self)
        self._reader: FrameReader | None = None
        self._tracks: TrackSet | None = None
        self._fps = 30.0
        self._n_frames = 0
        self._frame = 0
        self._pcutoff = 0.6
        self._part_order: list[str] = []
        self._skeleton: list[tuple[str, str]] = []
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._step = 1

        # ---- tools
        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(4)
        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QtWidgets.QToolButton] = {}
        for name, text, tip in TOOL_BUTTONS:
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setIcon(tool_icon(name))
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(lambda _checked=False, n=name: self.canvas.set_tool(n))
            self._tool_group.addButton(button)
            self._tool_buttons[name] = button
            tools.addWidget(button)
            if name == "select":
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.VLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                tools.addWidget(line)
        self._tool_buttons["select"].setChecked(True)
        tools.addStretch(1)
        fit = QtWidgets.QToolButton()
        fit.setIcon(tool_icon("fit"))
        fit.setText("Fit")
        fit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        fit.setAutoRaise(True)
        fit.setToolTip("Show the whole frame (0)")
        fit.clicked.connect(self.canvas.fit)
        tools.addWidget(fit)

        # ---- hint for the active tool
        self.hint = hint_label(TOOL_HINTS["select"])

        # ---- view options
        options = QtWidgets.QHBoxLayout()
        options.setSpacing(8)
        self.show_points = QtWidgets.QCheckBox("Show tracked points")
        self.show_points.setChecked(True)
        self.show_points.setToolTip("Show where DeepLabCut put each body part on this frame.\nDotted circles are points below the likelihood cut-off.")
        self.show_points.toggled.connect(lambda _on: self._update_overlay())
        options.addWidget(self.show_points)
        options.addSpacing(10)
        options.addWidget(QtWidgets.QLabel("Show the path of"))
        self.path_part = QuietCombo()
        self.path_part.setToolTip("Draw the whole path of one body part, to see where the animal went.")
        self.path_part.setMinimumWidth(110)
        self.path_part.currentIndexChanged.connect(lambda _i: self.pathPartChanged.emit(self.path_part.currentData() or ""))
        options.addWidget(self.path_part)
        options.addStretch(1)
        options.addWidget(QtWidgets.QLabel("Moving a shape changes"))
        self.positions_mode = QuietCombo()
        self.positions_mode.addItem("every video", False)
        self.positions_mode.addItem("only this video", True)
        self.positions_mode.setToolTip(
            "Shapes normally sit in the same place for every video.\n"
            "If the camera moved for one video, choose “only this video” and adjust the shapes for it."
        )
        self.positions_mode.currentIndexChanged.connect(lambda _i: self.positionsModeChanged.emit(bool(self.positions_mode.currentData())))
        options.addWidget(self.positions_mode)

        # ---- playback
        transport = QtWidgets.QHBoxLayout()
        transport.setSpacing(2)

        def player_button(icon: str, tip: str, slot) -> QtWidgets.QToolButton:
            button = QtWidgets.QToolButton()
            button.setIcon(tool_icon(icon))
            button.setAutoRaise(True)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            transport.addWidget(button)
            return button

        player_button("prev_event", "Previous scored event ([)", lambda: self.jump_event(-1))
        player_button("step_back", "Back one frame (,)", lambda: self.seek(self._frame - 1))
        self.play_button = player_button("play", "Play or pause (Space)", self.toggle_play)
        player_button("step_forward", "Forward one frame (.)", lambda: self.seek(self._frame + 1))
        player_button("next_event", "Next scored event (])", lambda: self.jump_event(1))
        self.time_label = QtWidgets.QLabel("")
        self.time_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        transport.addSpacing(8)
        transport.addWidget(self.time_label)
        transport.addStretch(1)
        self.cursor_label = QtWidgets.QLabel("")
        transport.addWidget(self.cursor_label)
        transport.addSpacing(8)
        transport.addWidget(QtWidgets.QLabel("Speed"))
        self.speed = QuietCombo()
        for text, value in SPEEDS:
            self.speed.addItem(text, value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda _i: self._restart_timer())
        transport.addWidget(self.speed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(tools)
        layout.addWidget(self.hint)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.timeline)
        layout.addLayout(transport)
        layout.addLayout(options)

        self.canvas.toolChanged.connect(self._tool_changed)
        self.canvas.cursorMoved.connect(self._cursor_moved)
        self.timeline.frameChosen.connect(self.seek)
        for key, slot in (
            (Qt.Key_Space, self.toggle_play),
            (Qt.Key_Comma, lambda: self.seek(self._frame - 1)),
            (Qt.Key_Period, lambda: self.seek(self._frame + 1)),
            (Qt.Key_BracketLeft, lambda: self.jump_event(-1)),
            (Qt.Key_BracketRight, lambda: self.jump_event(1)),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # ------------------------------------------------------------------ loading

    def load(self, video_path: str | None, tracks: TrackSet | None, fps: float, size, n_frames: int) -> None:
        """Show another video (or just its tracking when there is no video file)."""
        self.pause()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._tracks = tracks
        self._fps = fps if fps and fps > 0 else 30.0
        self._n_frames = max(0, int(n_frames))
        self._frame = 0
        if video_path:
            reader = FrameReader(video_path)
            if reader.ok:
                self._reader = reader
        if size:
            self.canvas.set_frame_size(int(size[0]), int(size[1]))
        self.canvas.set_frame(None)
        self.timeline.set_length(self._n_frames)
        self.timeline.set_marks([])
        self.show_frame(0)

    def clear(self) -> None:
        self.load(None, None, 30.0, None, 0)
        self.canvas.set_paths([])

    def set_tracking_style(self, pcutoff: float, part_order: list[str], skeleton: list[tuple[str, str]]) -> None:
        self._pcutoff = pcutoff
        self._part_order = list(part_order)
        self._skeleton = list(skeleton)
        self._update_overlay()

    def set_bodyparts(self, parts: list[str], preferred: str = "") -> None:
        current = self.path_part.currentData() if self.path_part.count() else preferred
        self.path_part.blockSignals(True)
        self.path_part.clear()
        self.path_part.addItem("(off)", "")
        for part in parts:
            self.path_part.addItem(part, part)
        index = self.path_part.findData(current if current in parts else (preferred if preferred in parts else ""))
        self.path_part.setCurrentIndex(max(0, index))
        self.path_part.blockSignals(False)

    def path_bodypart(self) -> str:
        return self.path_part.currentData() or ""

    def only_this_video(self) -> bool:
        return bool(self.positions_mode.currentData())

    def set_event_marks(self, marks: list[tuple[int, QtGui.QColor]]) -> None:
        self.timeline.set_marks(marks)

    # ----------------------------------------------------------------- playback

    def current_frame(self) -> int:
        return self._frame

    def seek(self, frame: int) -> None:
        self.show_frame(frame)

    def show_frame(self, index: int) -> None:
        last = max(0, self._n_frames - 1)
        index = max(0, min(int(index), last))
        self._frame = index
        if self._reader is not None and self._reader.ok:
            rgb = self._reader.read(index)
            if rgb is not None:
                height, width = rgb.shape[:2]
                image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.canvas.set_frame(image)
        self._update_overlay()
        self.timeline.set_frame(index)
        if self._n_frames:
            self.time_label.setText(
                f"{format_clock(index / self._fps)} / {format_clock(self._n_frames / self._fps)}"
                f"    frame {index + 1} of {self._n_frames}"
            )
        else:
            self.time_label.setText("")
        self.frameChanged.emit(index)

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self._n_frames <= 1:
            return
        if self._frame >= self._n_frames - 1:
            self._frame = 0
        self.play_button.setIcon(tool_icon("pause"))
        self._restart_timer(start=True)

    def pause(self) -> None:
        self._timer.stop()
        self.play_button.setIcon(tool_icon("play"))

    def _restart_timer(self, start: bool = False) -> None:
        if not (start or self._timer.isActive()):
            return
        rate = self._fps * float(self.speed.currentData() or 1.0)  # frames to show per second
        interval = max(16, int(round(1000.0 / rate)))
        self._step = max(1, int(round(rate * interval / 1000.0)))
        self._timer.start(interval)

    def _tick(self) -> None:
        nxt = self._frame + self._step
        if nxt >= self._n_frames - 1:
            self.show_frame(self._n_frames - 1)
            self.pause()
        else:
            self.show_frame(nxt)

    def jump_event(self, direction: int) -> None:
        frames = self.timeline.marks()
        if not frames:
            return
        if direction > 0:
            later = [f for f in frames if f > self._frame]
            target = later[0] if later else frames[0]
        else:
            earlier = [f for f in frames if f < self._frame]
            target = earlier[-1] if earlier else frames[-1]
        self.pause()
        self.seek(target)

    # ------------------------------------------------------------------ overlay

    def _update_overlay(self) -> None:
        data = OverlayData()
        tracks = self._tracks
        if tracks is not None and self.show_points.isChecked() and self._frame < tracks.n_frames:
            order = self._part_order or tracks.all_bodyparts()
            index_of = {part: i for i, part in enumerate(order)}
            for a_index, animal in enumerate(tracks.individuals):
                seen: dict[str, tuple[float, float]] = {}
                ring = animal_ring(a_index) if tracks.multi_animal else QtGui.QColor(0, 0, 0)
                for part in tracks.bodyparts.get(animal, []):
                    x, y, lik = tracks.data[(animal, part)][self._frame]
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    solid = bool(lik >= self._pcutoff)
                    color = part_color(index_of.get(part, len(index_of)))
                    data.dots.append(OverlayDot(float(x), float(y), color, ring, solid))
                    if solid:
                        seen[part] = (float(x), float(y))
                for a, b in self._skeleton:
                    if a in seen and b in seen:
                        data.bones.append((*seen[a], *seen[b]))
        self.canvas.set_overlay(data)

    # ------------------------------------------------------------------- pieces

    def _tool_changed(self, tool: str) -> None:
        button = self._tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)
        self.hint.setText(TOOL_HINTS.get(tool, ""))

    def _cursor_moved(self, x: float, y: float) -> None:
        if math.isfinite(x) and math.isfinite(y):
            self.cursor_label.setText(f"x {x:.0f}  y {y:.0f}")
        else:
            self.cursor_label.setText("")

    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)

    def has_picture(self) -> bool:
        """True when the video file itself could be opened (not only its tracking)."""
        return self._reader is not None and self._reader.ok

    def close_video(self) -> None:
        self.pause()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

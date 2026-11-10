"""The panel with the settings that decide how tracking becomes events."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..model import Layout
from .style import NumberBox, QuietCombo, hint_label

Qt = QtCore.Qt


class SettingsPanel(QtWidgets.QWidget):
    edited = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = Layout()
        self._loading = False
        self._project_cutoff: float | None = None

        self.pcutoff = NumberBox(0.0, 1.0, 0.05, 3)
        self.pcutoff.setToolTip(
            "DeepLabCut gives every point a likelihood between 0 and 1.\n"
            "Points below this value are treated as “not tracked”. The project's own value is used to start with."
        )
        self.cutoff_note = hint_label("")
        self.max_gap = NumberBox(0.0, 30.0, 0.1, 2)
        self.max_gap.setSuffix(" s")
        self.max_gap.setToolTip(
            "When a point is lost for a moment, its position is filled in with a straight line,\n"
            "as long as the gap is no longer than this. Longer gaps stay empty. 0 turns this off."
        )
        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(1, 99)
        self.smooth.setSingleStep(2)
        self.smooth.setSuffix(" frames")
        self.smooth.setToolTip(
            "Each position is replaced by the middle value of this many neighbouring frames.\n"
            "This removes one- or two-frame jumps to a wrong place. 1 turns it off. Odd numbers only."
        )
        self.margin = NumberBox(0.0, 500.0, 1.0, 1)
        self.margin.setSuffix(" px")
        self.margin.setToolTip(
            "A crossing only counts once the point is this far past the line (having been this far on the other side).\n"
            "The same goes for the edge of a region. It stops a point that sits on the line from being counted many times."
        )

        self.use_heading = QtWidgets.QCheckBox("The animal must face the way it moves")
        self.use_heading.setToolTip(
            "Uses two body parts to work out which way the animal points. A crossing made while backing up is ignored.\n"
            "If either part is not tracked at that moment, the crossing is counted."
        )
        self.head = QuietCombo()
        self.tail = QuietCombo()
        self.gap_crossings = QtWidgets.QCheckBox("Count crossings made while not tracked")
        self.gap_crossings.setToolTip(
            "If the point disappears on one side of a line and reappears on the other, it must have crossed.\n"
            "Such crossings are marked in the Events sheet so you can check them."
        )
        self.roi_gap = QuietCombo()
        self.roi_gap.addItem("stays as it was", "hold")
        self.roi_gap.addItem("is not counted", "exclude")
        self.roi_gap.setToolTip(
            "What happens to “time inside a region” during frames where the point is not tracked.\n"
            "“Stays as it was”: an animal that was inside keeps counting as inside until it is seen outside."
        )

        self.prefer_filtered = QtWidgets.QCheckBox("Use DeepLabCut's filtered files when they exist")
        self.prefer_filtered.setToolTip(
            "If you ran DeepLabCut's “filter predictions”, files ending in _filtered exist next to the normal ones.\n"
            "Tick this to score those instead."
        )
        self.default_fps = NumberBox(1.0, 100000.0, 1.0, 3)
        self.default_fps.setToolTip("Only used for a video that does not say how many frames per second it has.")

        for box in (self.pcutoff, self.max_gap, self.smooth, self.margin, self.default_fps):
            box.setMaximumWidth(130)
        for combo in (self.head, self.tail, self.roi_gap):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)

        def form_in(box: QtWidgets.QGroupBox) -> QtWidgets.QFormLayout:
            form = QtWidgets.QFormLayout(box)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            return form

        clean = QtWidgets.QGroupBox("Cleaning up the tracking")
        clean_form = form_in(clean)
        clean_form.addRow("Ignore points with likelihood below", self.pcutoff)
        clean_form.addRow(self.cutoff_note)
        clean_form.addRow("Fill gaps up to", self.max_gap)
        clean_form.addRow("Smooth over", self.smooth)

        events = QtWidgets.QGroupBox("Deciding what counts")
        events_form = form_in(events)
        events_form.addRow("Distance past a line or edge", self.margin)
        events_form.addRow(hint_label("A crossing or entry counts once the point is this far past the line or edge."))
        events_form.addRow(self.gap_crossings)
        events_form.addRow("Time in a region while not tracked", self.roi_gap)
        events_form.addRow(self.use_heading)
        events_form.addRow("    Front of the animal", self.head)
        events_form.addRow("    Back of the animal", self.tail)

        files = QtWidgets.QGroupBox("Files")
        files_form = form_in(files)
        files_form.addRow(self.prefer_filtered)
        files_form.addRow("Frames per second if a video does not say", self.default_fps)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.addWidget(clean)
        body_layout.addWidget(events)
        body_layout.addWidget(files)
        body_layout.addWidget(hint_label("Hover over a setting for an explanation. Settings are saved with the project and written into every results file."))
        body_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.pcutoff.valueChanged.connect(self._changed)
        self.max_gap.valueChanged.connect(self._changed)
        self.smooth.valueChanged.connect(self._changed)
        self.margin.valueChanged.connect(self._changed)
        self.default_fps.valueChanged.connect(self._changed)
        for box in (self.use_heading, self.gap_crossings, self.prefer_filtered):
            box.toggled.connect(self._changed)
        for combo in (self.head, self.tail, self.roi_gap):
            combo.currentIndexChanged.connect(self._changed)

    def set_model(self, layout: Layout, bodyparts: list[str], project_cutoff: float | None = None) -> None:
        self.model = layout
        self._project_cutoff = project_cutoff
        self._loading = True
        s = layout.settings
        self.pcutoff.setValue(s.pcutoff)
        self.max_gap.setValue(s.max_gap_s)
        self.smooth.setValue(s.smooth_frames)
        self.margin.setValue(s.margin_px)
        self.use_heading.setChecked(s.use_heading)
        self.gap_crossings.setChecked(s.count_gap_crossings)
        self.prefer_filtered.setChecked(s.prefer_filtered)
        self.default_fps.setValue(s.default_fps)
        self.roi_gap.setCurrentIndex(max(0, self.roi_gap.findData(s.roi_gap_policy)))
        for combo, current, fallback in ((self.head, s.head_part, 0), (self.tail, s.tail_part, -1)):
            combo.clear()
            parts = list(bodyparts)
            if current and current not in parts:
                parts.append(current)
            for part in parts:
                combo.addItem(part, part)
            index = combo.findData(current) if current else (len(parts) - 1 if fallback < 0 else 0)
            combo.setCurrentIndex(max(0, index) if parts else -1)
        self._loading = False
        self._sync()

    def _sync(self) -> None:
        on = self.use_heading.isChecked()
        self.head.setEnabled(on)
        self.tail.setEnabled(on)
        if self._project_cutoff is not None:
            same = abs(self._project_cutoff - self.pcutoff.value()) < 1e-9
            self.cutoff_note.setText(
                "Same as the DeepLabCut project." if same else f"The DeepLabCut project uses {self._project_cutoff:g}."
            )
        else:
            self.cutoff_note.setText("")

    def _changed(self, *_args) -> None:
        self._sync()
        if self._loading:
            return
        s = self.model.settings
        s.pcutoff = float(self.pcutoff.value())
        s.max_gap_s = float(self.max_gap.value())
        smooth = int(self.smooth.value())
        s.smooth_frames = smooth if smooth % 2 == 1 else smooth + 1
        s.margin_px = float(self.margin.value())
        s.use_heading = self.use_heading.isChecked()
        s.head_part = self.head.currentData() or ""
        s.tail_part = self.tail.currentData() or ""
        s.count_gap_crossings = self.gap_crossings.isChecked()
        s.roi_gap_policy = self.roi_gap.currentData() or "hold"
        s.prefer_filtered = self.prefer_filtered.isChecked()
        s.default_fps = float(self.default_fps.value())
        self.edited.emit()

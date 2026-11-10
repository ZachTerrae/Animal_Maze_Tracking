"""The panel where a line or region is given its scoring rules."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..export import describe_rule
from ..geometry import meaningful_directions
from ..model import Layout, Rule, clean_column_name
from .style import NumberBox, QuietCombo, color_swatch, hint_label

Qt = QtCore.Qt
Signal = QtCore.Signal

LINE_MEASURES = [
    ("count", "Add points each time it crosses"),
    ("first_time", "Record the time of the first crossing (s)"),
]
ROI_MEASURES = [
    ("time", "Time spent inside (seconds)"),
    ("entries", "Add points each time it enters"),
    ("exits", "Add points each time it leaves"),
    ("first_time", "Record the time of the first entry (s)"),
]
COUNT_MODES = [
    ("every", "Every time"),
    ("first", "The first time only"),
    ("first_in_group", "Only if it is the animal's first choice"),
]

DIRECTION_CHOICES = {
    "any": "In either direction",
    "left_to_right": "Left → right",
    "right_to_left": "Right → left",
    "top_to_bottom": "Top → bottom",
    "bottom_to_top": "Bottom → top",
}

GROUP_TIP = (
    "Rules that share a group name compete with each other. In each video (and for each animal) only the rule "
    "whose event happens first gets its points; the others get none.\n\n"
    "Example: give the “left arm” line and the “right arm” line a rule each in the group “arm”. "
    "The column of the arm that was entered first gets 1, the other gets 0."
)


def suggest_column(shape_name: str, shape_kind: str, measure: str) -> str:
    if shape_kind == "line":
        return f"{shape_name} first crossing (s)" if measure == "first_time" else f"{shape_name} crossings"
    return {
        "time": f"{shape_name} time (s)",
        "entries": f"{shape_name} entries",
        "exits": f"{shape_name} exits",
        "first_time": f"{shape_name} first entry (s)",
    }[measure]


class RuleCard(QtWidgets.QFrame):
    """Editor for one rule."""

    changed = Signal()
    removeRequested = Signal()

    def __init__(self, number: int, rule: Rule, shape, layout: Layout, bodyparts: list[str], directions: list[str], parent=None):
        super().__init__(parent)
        self.rule = rule
        self.shape = shape
        self.model = layout
        self._loading = True
        self.setObjectName("ruleCard")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("#ruleCard { border: 1px solid palette(mid); border-radius: 6px; }")

        self.title = QtWidgets.QLabel(f"<b>Score {number}</b>")
        self.sentence = hint_label("")
        remove = QtWidgets.QToolButton()
        remove.setText("✕")
        remove.setAutoRaise(True)
        remove.setToolTip("Remove this score")
        remove.clicked.connect(self.removeRequested.emit)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(remove)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(5)
        self.form = form

        self.bodypart = QuietCombo()
        parts = list(bodyparts)
        if rule.bodypart and rule.bodypart not in parts:
            parts.append(rule.bodypart)
        for part in parts:
            known = part in bodyparts
            self.bodypart.addItem(part if known else f"{part} (not in this project)", part)
        if not rule.bodypart and parts:
            rule.bodypart = parts[0]
        self.bodypart.setCurrentIndex(max(0, self.bodypart.findData(rule.bodypart)))
        self.bodypart.setToolTip("The tracked point on the animal that has to cross the line or be inside the region.")
        form.addRow("Body part", self.bodypart)

        self.direction = QuietCombo()
        if shape.kind == "line":
            offered = list(directions)
            if rule.direction not in offered:
                offered.append(rule.direction)
            for key in offered:
                label = DIRECTION_CHOICES[key]
                if key not in directions:
                    label += " (does not suit this line)"
                self.direction.addItem(label, key)
            self.direction.setCurrentIndex(max(0, self.direction.findData(rule.direction)))
            self.direction.setToolTip(
                "Which way the body part has to be travelling when it crosses, as seen on the screen.\n"
                "The arrow on the line shows the direction that scores."
            )
            form.addRow("When it crosses", self.direction)

        self.measure = QuietCombo()
        for key, label in LINE_MEASURES if shape.kind == "line" else ROI_MEASURES:
            self.measure.addItem(label, key)
        self.measure.setCurrentIndex(max(0, self.measure.findData(rule.measure)))
        form.addRow("What to score", self.measure)

        self.points = NumberBox(-1e9, 1e9, 1.0)
        self.points.setValue(rule.points)
        self.points.setToolTip("How much is added to the column each time. Use a negative number to subtract.")
        form.addRow("Points to add", self.points)

        self.column = QuietCombo()
        self.column.setEditable(True)
        self.column.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.column.lineEdit().setPlaceholderText("name of the column in the results")
        self.column.setToolTip(
            "The column of the results table this score goes into.\n"
            "Pick an existing column to add to it, or type a new name to make a new column."
        )
        form.addRow("Into column", self.column)

        self.count_mode = QuietCombo()
        for key, label in COUNT_MODES:
            self.count_mode.addItem(label, key)
        self.count_mode.setCurrentIndex(max(0, self.count_mode.findData(rule.count_mode)))
        self.count_mode.setToolTip("“First choice” is for mazes: see the tip on the group box that appears.")
        form.addRow("Count", self.count_mode)

        self.group = QuietCombo()
        self.group.setEditable(True)
        self.group.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.group.lineEdit().setPlaceholderText("for example: arm")
        self.group.setToolTip(GROUP_TIP)
        form.addRow("Choice group", self.group)

        for field in (self.bodypart, self.direction, self.measure, self.points, self.column, self.count_mode, self.group):
            field.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            field.setMinimumWidth(120)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)
        outer.addLayout(head)
        outer.addLayout(form)
        outer.addWidget(self.sentence)

        self.refresh_lists()
        self._sync_visibility()
        self._update_sentence()
        self._loading = False

        self.bodypart.currentIndexChanged.connect(self._bodypart_changed)
        self.direction.currentIndexChanged.connect(self._direction_changed)
        self.measure.currentIndexChanged.connect(self._measure_changed)
        self.points.valueChanged.connect(self._points_changed)
        self.column.lineEdit().editingFinished.connect(self._column_changed)
        self.column.activated.connect(lambda _i: self._column_changed())
        self.count_mode.currentIndexChanged.connect(self._count_changed)
        self.group.lineEdit().editingFinished.connect(self._group_changed)
        self.group.activated.connect(lambda _i: self._group_changed())

    # -- keeping the lists of existing columns and groups fresh

    def refresh_lists(self) -> None:
        for combo, items, current in (
            (self.column, self.model.columns(), self.rule.column),
            (self.group, self.model.groups(), self.rule.group),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([i for i in items if i])
            combo.setCurrentText(current)
            combo.blockSignals(False)

    def _row_visible(self, field: QtWidgets.QWidget, visible: bool) -> None:
        self.form.setRowVisible(field, visible)

    def _sync_visibility(self) -> None:
        event_rule = self.rule.is_event_rule
        self._row_visible(self.points, event_rule)
        self._row_visible(self.count_mode, event_rule)
        self._row_visible(self.group, event_rule and self.rule.count_mode == "first_in_group")

    def _update_sentence(self) -> None:
        self.sentence.setText(describe_rule(self.rule, self.shape.kind))

    def _emit(self) -> None:
        self._update_sentence()
        if not self._loading:
            self.changed.emit()

    # -- edits

    def _bodypart_changed(self, _index: int) -> None:
        self.rule.bodypart = self.bodypart.currentData() or ""
        self._emit()

    def _direction_changed(self, _index: int) -> None:
        self.rule.direction = self.direction.currentData() or "any"
        self._emit()

    def _measure_changed(self, _index: int) -> None:
        old = self.rule.measure
        new = self.measure.currentData()
        if not new or new == old:
            return
        was_suggested = self.rule.column == suggest_column(self.shape.name, self.shape.kind, old)
        self.rule.measure = new
        if was_suggested or not self.rule.column:
            self.rule.column = ""  # so that the new name is not "taken" by this very rule
            self.rule.column = self.model.unique_column_name(suggest_column(self.shape.name, self.shape.kind, new))
            self.column.setCurrentText(self.rule.column)
        self.model.sync_columns()
        self._sync_visibility()
        self._emit()

    def _points_changed(self, value: float) -> None:
        self.rule.points = float(value)
        self._emit()

    def _column_changed(self) -> None:
        name = clean_column_name(self.column.currentText())
        if name != self.column.currentText():
            self.column.setCurrentText(name)
        if name == self.rule.column:
            return
        self.rule.column = name
        self.model.sync_columns()
        self._emit()

    def _count_changed(self, _index: int) -> None:
        self.rule.count_mode = self.count_mode.currentData() or "every"
        if self.rule.count_mode == "first_in_group" and not self.rule.group:
            groups = self.model.groups()
            self.rule.group = groups[0] if groups else "choice"
            self.group.setCurrentText(self.rule.group)
        self._sync_visibility()
        self._emit()

    def _group_changed(self) -> None:
        name = self.group.currentText().strip()
        if name == self.rule.group:
            return
        self.rule.group = name
        self._emit()


class ShapePanel(QtWidgets.QWidget):
    edited = Signal()  # anything about the shapes or their rules changed
    selectRequested = Signal(object)  # shape id or None
    deleteRequested = Signal(str)
    sharedPositionRequested = Signal(str)  # drop this video's own position for a shape

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = Layout()
        self.bodyparts: list[str] = []
        self.video_key: str | None = None
        self.selected_id: str | None = None
        self._cards: list[RuleCard] = []
        self._points_for_direction = lambda shape: (shape.p1, shape.p2)

        self.shape_list = QtWidgets.QListWidget()
        self.shape_list.setMaximumHeight(120)
        self.shape_list.setToolTip("Every line and region. Click one to edit it.")
        self.shape_list.itemSelectionChanged.connect(self._list_selection_changed)
        self.empty_hint = hint_label(
            "<b>No lines or regions yet.</b><br>Pick <b>Line</b>, <b>Region</b>, <b>Rectangle</b>, <b>Oval</b> or "
            "<b>Freehand</b> above the video and draw on it. Then set here what it adds to the score."
        )

        # details of the selected shape
        self.details = QtWidgets.QWidget()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._name_changed)
        self.color_button = QtWidgets.QToolButton()
        self.color_button.setToolTip("Colour of this shape on the video")
        self.color_button.clicked.connect(self._pick_color)
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setText("Delete")
        self.delete_button.setToolTip("Delete this shape and its scores (Delete key)")
        self.delete_button.clicked.connect(lambda: self.selected_id and self.deleteRequested.emit(self.selected_id))
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name"))
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.color_button)
        name_row.addWidget(self.delete_button)

        self.own_position = QtWidgets.QWidget()
        own_layout = QtWidgets.QHBoxLayout(self.own_position)
        own_layout.setContentsMargins(0, 0, 0, 0)
        own_layout.addWidget(hint_label("This video has its own position for this shape (dashed outline)."), 1)
        shared = QtWidgets.QToolButton()
        shared.setText("Use the shared position")
        shared.clicked.connect(lambda: self.selected_id and self.sharedPositionRequested.emit(self.selected_id))
        own_layout.addWidget(shared)

        self.rules_heading = QtWidgets.QLabel("")
        self.rules_box = QtWidgets.QVBoxLayout()
        self.rules_box.setSpacing(8)
        self.add_button = QtWidgets.QPushButton("+  Add another score")
        self.add_button.setToolTip("One line or region can feed any number of columns.")
        self.add_button.clicked.connect(self._add_rule)

        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        details_layout.addLayout(name_row)
        details_layout.addWidget(self.own_position)
        details_layout.addWidget(self.rules_heading)
        details_layout.addLayout(self.rules_box)
        details_layout.addWidget(self.add_button)

        self.pick_hint = hint_label("Click a line or region, on the video or in the list above, to set what it scores.")

        self.summary_title = QtWidgets.QLabel("<b>This video</b>")
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(10)
        body_layout.addWidget(self.empty_hint)
        body_layout.addWidget(self.shape_list)
        body_layout.addWidget(self.pick_hint)
        body_layout.addWidget(self.details)
        rule = QtWidgets.QFrame()
        rule.setFrameShape(QtWidgets.QFrame.HLine)
        rule.setFrameShadow(QtWidgets.QFrame.Sunken)
        body_layout.addWidget(rule)
        body_layout.addWidget(self.summary_title)
        body_layout.addWidget(self.summary)
        body_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.refresh()

    # ----------------------------------------------------------------- feeding

    def set_model(self, layout: Layout) -> None:
        self.model = layout
        self.selected_id = None
        self.refresh()

    def set_bodyparts(self, parts: list[str]) -> None:
        if parts != self.bodyparts:
            self.bodyparts = list(parts)
            self._rebuild_cards()

    def set_video(self, video_key: str | None) -> None:
        self.video_key = video_key
        self._update_own_position()

    def set_direction_source(self, getter) -> None:
        """``getter(line) -> (p1, p2)`` as the line lies on the video being shown."""
        self._points_for_direction = getter

    def set_selected(self, shape_id: str | None) -> None:
        if shape_id is not None and self.model.find_shape(shape_id) is None:
            shape_id = None
        self.selected_id = shape_id
        self.refresh()

    def set_summary(self, html: str) -> None:
        self.summary.setText(html)
        self.summary_title.setVisible(bool(html))

    def default_bodypart(self) -> str:
        used = self.model.used_bodyparts()
        if used:
            return used[-1]
        return self.bodyparts[0] if self.bodyparts else ""

    # ---------------------------------------------------------------- building

    def refresh(self) -> None:
        shapes = self.model.shapes()
        self.empty_hint.setVisible(not shapes)
        self.shape_list.setVisible(bool(shapes))
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        for shape in shapes:
            count = len(shape.rules)
            kind = "line" if shape.kind == "line" else "region"
            text = f"{shape.name}   ({kind}, {count} score{'s' if count != 1 else ''})"
            item = QtWidgets.QListWidgetItem(color_swatch(shape.color), text)
            item.setData(Qt.UserRole, shape.id)
            self.shape_list.addItem(item)
            if shape.id == self.selected_id:
                item.setSelected(True)
                self.shape_list.setCurrentItem(item)
        self.shape_list.blockSignals(False)
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        self.details.setVisible(shape is not None)
        self.pick_hint.setVisible(shape is None and bool(shapes))
        if shape is not None:
            self.name_edit.blockSignals(True)
            self.name_edit.setText(shape.name)
            self.name_edit.blockSignals(False)
            self.color_button.setIcon(color_swatch(shape.color, 16))
            what = "this line" if shape.kind == "line" else "this region"
            self.rules_heading.setText(f"<b>What {what} adds to the results</b>")
        self._update_own_position()
        self._rebuild_cards()

    def _update_own_position(self) -> None:
        own = bool(self.selected_id and self.video_key and self.model.has_override(self.video_key, self.selected_id))
        self.own_position.setVisible(own)

    def _rebuild_cards(self) -> None:
        for card in self._cards:
            self.rules_box.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is None:
            return
        directions = ["any"]
        if shape.kind == "line":
            p1, p2 = self._points_for_direction(shape)
            directions = meaningful_directions(p1, p2)
        for number, rule in enumerate(shape.rules, start=1):
            card = RuleCard(number, rule, shape, self.model, self.bodyparts, directions)
            card.changed.connect(self._card_changed)
            card.removeRequested.connect(lambda r=rule: self._remove_rule(r))
            self.rules_box.addWidget(card)
            self._cards.append(card)

    def refresh_directions(self) -> None:
        """Call after a line was moved: the directions on offer may have changed."""
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is not None and shape.kind == "line":
            self._rebuild_cards()

    # ------------------------------------------------------------------- edits

    def _list_selection_changed(self) -> None:
        items = self.shape_list.selectedItems()
        shape_id = items[0].data(Qt.UserRole) if items else None
        if shape_id != self.selected_id:
            self.selectRequested.emit(shape_id)

    def _card_changed(self) -> None:
        sender = self.sender()
        for card in self._cards:
            if card is not sender:
                card.refresh_lists()
        self._refresh_list_texts()
        self.edited.emit()

    def _refresh_list_texts(self) -> None:
        for row in range(self.shape_list.count()):
            item = self.shape_list.item(row)
            shape = self.model.find_shape(item.data(Qt.UserRole))
            if shape is None:
                continue
            count = len(shape.rules)
            kind = "line" if shape.kind == "line" else "region"
            item.setText(f"{shape.name}   ({kind}, {count} score{'s' if count != 1 else ''})")
            item.setIcon(color_swatch(shape.color))

    def make_default_rule(self, shape) -> Rule:
        """The rule a freshly drawn shape starts with."""
        measure = "count" if shape.kind == "line" else "time"
        rule = Rule(bodypart=self.default_bodypart(), measure=measure)
        rule.column = self.model.unique_column_name(suggest_column(shape.name, shape.kind, measure))
        return rule

    def _add_rule(self) -> None:
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is None:
            return
        if shape.rules:
            last = shape.rules[-1]
            rule = Rule(bodypart=last.bodypart, measure=last.measure, direction=last.direction, points=last.points)
            if shape.kind == "roi":  # offer the next kind of measure rather than a copy
                order = [key for key, _ in ROI_MEASURES]
                used = {r.measure for r in shape.rules}
                rule.measure = next((m for m in order if m not in used), last.measure)
            rule.column = self.model.unique_column_name(suggest_column(shape.name, shape.kind, rule.measure))
        else:
            rule = self.make_default_rule(shape)
        shape.rules.append(rule)
        self.model.sync_columns()
        self._rebuild_cards()
        self._refresh_list_texts()
        self.edited.emit()

    def _remove_rule(self, rule: Rule) -> None:
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is None:
            return
        shape.rules = [r for r in shape.rules if r.id != rule.id]
        self.model.sync_columns()
        self._rebuild_cards()
        self._refresh_list_texts()
        self.edited.emit()

    def _name_changed(self) -> None:
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is None:
            return
        new = self.name_edit.text().strip()
        if not new or new == shape.name:
            self.name_edit.setText(shape.name)
            return
        old = shape.name
        shape.name = new
        # columns that still carry the automatic name follow the shape's new name
        for rule in shape.rules:
            if rule.column == suggest_column(old, shape.kind, rule.measure):
                target = suggest_column(new, shape.kind, rule.measure)
                if target not in self.model.columns():
                    self.model.rename_column(rule.column, target)
        self.model.sync_columns()
        self._rebuild_cards()
        self._refresh_list_texts()
        self.edited.emit()

    def _pick_color(self) -> None:
        shape = self.model.find_shape(self.selected_id) if self.selected_id else None
        if shape is None:
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(shape.color), self, "Colour of " + shape.name)
        if color.isValid():
            shape.color = color.name()
            self.color_button.setIcon(color_swatch(shape.color, 16))
            self._refresh_list_texts()
            self.edited.emit()

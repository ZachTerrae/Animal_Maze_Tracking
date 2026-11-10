"""The panel for the results table: the file-label column, column order, formula columns."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..formulas import FUNCTION_HELP, Formula, FormulaError, check_formula
from ..idcolumn import ID_MODE_LABELS, ID_MODES, IdColumn, check_pattern, make_id
from ..model import FormulaColumn, Layout, clean_column_name
from .style import QuietCombo, format_number, hint_label

Qt = QtCore.Qt
Signal = QtCore.Signal


class IdColumnDialog(QtWidgets.QDialog):
    """Choose what goes into the first column of the results."""

    def __init__(self, config: IdColumn, sample_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("First column of the results")
        self.setMinimumWidth(560)
        self.samples = sample_paths[:8] or ["/videos/day1/fish07_trialB_2024-05-01.mp4"]
        self.result_config = IdColumn.from_dict(config.to_dict())

        self.header = QtWidgets.QLineEdit(config.header)
        self.mode = QuietCombo()
        for mode in ID_MODES:
            self.mode.addItem(ID_MODE_LABELS[mode], mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.mode)))

        self.delimiters = QtWidgets.QLineEdit(config.delimiters)
        self.delimiters.setToolTip("The characters that separate the pieces of your file names, for example _ or _-")
        self.pieces = QtWidgets.QLineEdit(config.pieces)
        self.pieces.setToolTip("Which pieces to keep: 1   or   2-3   or   1,4   or   last")
        self.after = QtWidgets.QLineEdit(config.after)
        self.before = QtWidgets.QLineEdit(config.before)
        self.pattern = QtWidgets.QLineEdit(config.pattern)
        self.pattern.setPlaceholderText(r"for example  fish(\d+)   (the part in brackets is kept)")

        form = QtWidgets.QFormLayout()
        form.addRow("Column heading", self.header)
        form.addRow("Fill it with", self.mode)
        self.form = form
        form.addRow("Split the name at", self.delimiters)
        form.addRow("Keep piece(s)", self.pieces)
        form.addRow("Keep the text after", self.after)
        form.addRow("and before", self.before)
        form.addRow("Pattern", self.pattern)

        self.message = hint_label("")
        self.preview = QtWidgets.QTableWidget(len(self.samples), 2)
        self.preview.setHorizontalHeaderLabels(["File", "Goes into the first column as"])
        self.preview.verticalHeader().setVisible(False)
        self.preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preview.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.preview.horizontalHeader().setStretchLastSection(True)
        self.preview.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.preview.setMinimumHeight(190)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.message)
        layout.addWidget(QtWidgets.QLabel("<b>Preview</b>"))
        layout.addWidget(self.preview, 1)
        layout.addWidget(buttons)

        for widget in (self.header, self.delimiters, self.pieces, self.after, self.before, self.pattern):
            widget.textChanged.connect(self._update)
        self.mode.currentIndexChanged.connect(self._update)
        self._update()

    def _current(self) -> IdColumn:
        return IdColumn(
            header=self.header.text().strip() or "File",
            mode=self.mode.currentData(),
            delimiters=self.delimiters.text() or "_",
            pieces=self.pieces.text().strip() or "1",
            after=self.after.text(),
            before=self.before.text(),
            pattern=self.pattern.text(),
        )

    def _update(self) -> None:
        config = self._current()
        mode = config.mode
        self.form.setRowVisible(self.delimiters, mode == "pieces")
        self.form.setRowVisible(self.pieces, mode == "pieces")
        self.form.setRowVisible(self.after, mode == "between")
        self.form.setRowVisible(self.before, mode == "between")
        self.form.setRowVisible(self.pattern, mode == "regex")
        problem = check_pattern(config.pattern) if mode == "regex" and config.pattern else None
        labels = []
        for row, path in enumerate(self.samples):
            name = path.replace("\\", "/").split("/")[-1]
            label = make_id(config, path)
            labels.append(label)
            self.preview.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.preview.setItem(row, 1, QtWidgets.QTableWidgetItem(label))
        if problem:
            self.message.setText(f"<span style='color:#d9534f'>The pattern is not valid: {problem}</span>")
        elif len(set(labels)) < len(labels):
            self.message.setText(
                "Note: some videos get the same label. That is fine if they belong together, "
                "but each row of the results will not be unique."
            )
        elif mode == "pieces":
            self.message.setText("Pieces are counted from 1. Use “last” for the final piece, “2-3” for a range, “1,4” for a selection.")
        else:
            self.message.setText("")
        self.ok_button.setEnabled(not problem)
        self.result_config = config


class FormulaDialog(QtWidgets.QDialog):
    """Create or edit a column that is worked out from other columns."""

    def __init__(self, layout: Layout, formula: FormulaColumn | None, sample_values: dict[str, float] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formula column")
        self.setMinimumWidth(620)
        self.model = layout
        self.original_name = formula.name if formula else ""
        self.sample_values = sample_values or {}
        self.result_name = ""
        self.result_expression = ""

        self.name = QtWidgets.QLineEdit(formula.name if formula else "")
        self.name.setPlaceholderText("for example: Total crossings")
        self.expression = QtWidgets.QPlainTextEdit(formula.expression if formula else "")
        self.expression.setPlaceholderText("for example:  [Left entries] + [Right entries]")
        self.expression.setFixedHeight(78)
        self.expression.setTabChangesFocus(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.expression.setFont(font)

        self.columns = QtWidgets.QListWidget()
        self.columns.setToolTip("Double-click a column to put it into the formula.")
        for column in layout.columns():
            if column != self.original_name:
                self.columns.addItem(column)
        self.columns.itemDoubleClicked.connect(lambda item: self._insert(f"[{item.text()}]"))
        self.columns.setMaximumHeight(150)

        ops = QtWidgets.QHBoxLayout()
        ops.setSpacing(3)
        for text in ("+", "-", "*", "/", "(", ")", "^", "MIN( , )", "MAX( , )", "IF( , , )", "ROUND( , 2)", "IFBLANK( , 0)"):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.clicked.connect(lambda _c=False, t=text: self._insert(t if len(t) > 1 else f" {t} "))
            ops.addWidget(button)
        ops.addStretch(1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)

        form = QtWidgets.QFormLayout()
        form.addRow("Column name", self.name)
        form.addRow("Formula", self.expression)
        layout_ = QtWidgets.QVBoxLayout(self)
        layout_.addLayout(form)
        layout_.addLayout(ops)
        layout_.addWidget(self.status)
        layout_.addWidget(QtWidgets.QLabel("<b>Columns you can use</b> (double-click to insert)"))
        layout_.addWidget(self.columns)
        layout_.addWidget(
            hint_label(
                "Put column names in [square brackets]. An empty cell (for example a time that never happened) "
                "makes the result empty; wrap it in IFBLANK([Column], 0) to treat it as 0.<br><tt>"
                + FUNCTION_HELP.replace("<", "&lt;").replace("\n", "<br>")
                + "</tt>"
            )
        )
        layout_.addWidget(buttons)

        self.name.textChanged.connect(self._validate)
        self.expression.textChanged.connect(self._validate)
        self._validate()
        self.name.setFocus()

    def _insert(self, text: str) -> None:
        self.expression.insertPlainText(text)
        self.expression.setFocus()

    def _problem(self) -> str | None:
        name = clean_column_name(self.name.text())
        if not name:
            return "Give the column a name."
        taken = set(self.model.columns()) - {self.original_name}
        if name in taken:
            return f"There is already a column called “{name}”."
        expression = self.expression.toPlainText().strip()
        if not expression:
            return "Type a formula."
        available = [c for c in self.model.columns() if c != self.original_name]
        return check_formula(expression, available, name)

    def _validate(self) -> None:
        problem = self._problem()
        self.ok_button.setEnabled(problem is None)
        if problem:
            self.status.setText(f"<span style='color:#d9534f'>{problem}</span>")
            return
        text = "<span style='color:#3c9a5f'>The formula is fine.</span>"
        try:
            formula = Formula(self.expression.toPlainText())
            if all(ref in self.sample_values for ref in formula.references):
                value = formula.evaluate(self.sample_values)
                shown = format_number(value) or "(empty)"
                text += f" For the video being shown it gives <b>{shown}</b>."
        except FormulaError:
            pass
        self.status.setText(text)

    def _accept(self) -> None:
        if self._problem() is None:
            self.result_name = clean_column_name(self.name.text())
            self.result_expression = self.expression.toPlainText().strip()
            self.accept()


class ColumnsPanel(QtWidgets.QWidget):
    edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = Layout()
        self.sample_paths: list[str] = []
        self.sample_values: dict[str, float] = {}

        self.id_summary = QtWidgets.QLabel("")
        self.id_summary.setWordWrap(True)
        change = QtWidgets.QPushButton("Change…")
        change.clicked.connect(self._edit_id_column)
        id_box = QtWidgets.QGroupBox("First column: which video the row is for")
        id_layout = QtWidgets.QHBoxLayout(id_box)
        id_layout.addWidget(self.id_summary, 1)
        id_layout.addWidget(change, 0, Qt.AlignTop)

        self.list = QtWidgets.QListWidget()
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setToolTip("The columns of the results, in order. Drag to reorder. Double-click to rename or edit.")
        self.list.model().rowsMoved.connect(lambda *_: self._order_changed())
        self.list.itemDoubleClicked.connect(lambda _item: self._edit_selected())
        self.list.itemSelectionChanged.connect(self._update_buttons)

        self.add_formula = QtWidgets.QPushButton("+  Formula column…")
        self.add_formula.setToolTip("Add a column worked out from other columns, for example a total or a percentage.")
        self.add_formula.clicked.connect(self._add_formula)
        self.edit_button = QtWidgets.QPushButton("Edit…")
        self.edit_button.clicked.connect(self._edit_selected)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)
        self.up_button = QtWidgets.QToolButton()
        self.up_button.setArrowType(Qt.UpArrow)
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QtWidgets.QToolButton()
        self.down_button.setArrowType(Qt.DownArrow)
        self.down_button.clicked.connect(lambda: self._move(1))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.add_formula)
        row.addWidget(self.edit_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        row.addWidget(self.up_button)
        row.addWidget(self.down_button)

        columns_box = QtWidgets.QGroupBox("Score columns, left to right")
        columns_layout = QtWidgets.QVBoxLayout(columns_box)
        columns_layout.addWidget(self.list, 1)
        columns_layout.addLayout(row)
        self.note = hint_label(
            "Score columns come from the rules on your lines and regions. "
            "Formula columns (ƒ) are calculated from other columns."
        )
        columns_layout.addWidget(self.note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(id_box)
        layout.addWidget(columns_box, 1)
        self.refresh()

    def set_model(self, layout: Layout) -> None:
        self.model = layout
        self.refresh()

    def set_samples(self, paths: list[str], values: dict[str, float] | None = None) -> None:
        self.sample_paths = list(paths)
        self.sample_values = dict(values or {})
        self._update_id_summary()

    def _update_id_summary(self) -> None:
        config = self.model.id_column
        text = f"<b>{config.header or 'File'}</b>: {ID_MODE_LABELS.get(config.mode, config.mode).lower()}"
        if self.sample_paths:
            name = self.sample_paths[0].replace("\\", "/").split("/")[-1]
            text += f"<br><span style='font-size:small'>{name} → <b>{make_id(config, self.sample_paths[0])}</b></span>"
        self.id_summary.setText(text)

    def refresh(self) -> None:
        self._update_id_summary()
        selected = self.list.currentItem().data(Qt.UserRole) if self.list.currentItem() else None
        self.list.blockSignals(True)
        self.list.clear()
        from ..formulas import check_formula as _check

        columns = self.model.columns()
        for column in columns:
            formula = self.model.find_formula(column) if self.model.is_formula(column) else None
            if formula is not None:
                item = QtWidgets.QListWidgetItem(f"ƒ  {column}   =   {formula.expression}")
                problem = _check(formula.expression, columns, column)
                if problem:
                    item.setForeground(QtGui.QColor("#d9534f"))
                    item.setToolTip(problem)
                else:
                    item.setToolTip("Formula column. Double-click to edit.")
            else:
                item = QtWidgets.QListWidgetItem(column)
                item.setToolTip("Score column. Double-click to rename.")
            item.setData(Qt.UserRole, column)
            self.list.addItem(item)
            if column == selected:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self._update_buttons()

    def _selected_column(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _update_buttons(self) -> None:
        column = self._selected_column()
        self.edit_button.setEnabled(column is not None)
        self.remove_button.setEnabled(column is not None and self.model.is_formula(column))
        self.up_button.setEnabled(column is not None)
        self.down_button.setEnabled(column is not None)

    def _order_changed(self) -> None:
        self.model.column_order = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]
        self.model.sync_columns()
        self.edited.emit()

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.list.count():
            return
        order = self.model.columns()
        order[row], order[target] = order[target], order[row]
        self.model.column_order = order
        self.refresh()
        self.list.setCurrentRow(target)
        self.edited.emit()

    def _edit_id_column(self) -> None:
        dialog = IdColumnDialog(self.model.id_column, self.sample_paths, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.model.id_column = dialog.result_config
            self.refresh()
            self.edited.emit()

    def _add_formula(self) -> None:
        dialog = FormulaDialog(self.model, None, self.sample_values, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.model.formulas.append(FormulaColumn(dialog.result_name, dialog.result_expression))
            self.model.sync_columns()
            self.refresh()
            self.edited.emit()

    def _edit_selected(self) -> None:
        column = self._selected_column()
        if column is None:
            return
        if self.model.is_formula(column):
            formula = self.model.find_formula(column)
            dialog = FormulaDialog(self.model, formula, self.sample_values, self)
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                if dialog.result_name != formula.name:
                    self.model.rename_column(formula.name, dialog.result_name)
                formula.expression = dialog.result_expression
                self.model.sync_columns()
                self.refresh()
                self.edited.emit()
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename column", "Column name:", text=column)
        new = clean_column_name(new)
        if not ok or not new or new == column:
            return
        if new in self.model.columns():
            QtWidgets.QMessageBox.information(
                self, "Rename column", f"There is already a column called “{new}”.\nChoose a different name."
            )
            return
        self.model.rename_column(column, new)
        self.refresh()
        self.edited.emit()

    def _remove_selected(self) -> None:
        column = self._selected_column()
        if column is None or not self.model.is_formula(column):
            return
        self.model.formulas = [f for f in self.model.formulas if f.name != column]
        self.model.sync_columns()
        self.refresh()
        self.edited.emit()

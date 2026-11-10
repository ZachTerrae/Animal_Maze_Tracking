"""The bottom of the window: the scores table, the events of the video being shown, and a log."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..batch import ResultsTable
from ..scoring import ScoredEvent
from .style import format_number, hint_label

Qt = QtCore.Qt
Signal = QtCore.Signal


class _SortableItem(QtWidgets.QTableWidgetItem):
    """A table cell that sorts by its number when it has one."""

    def __init__(self, text: str, number: float | None = None):
        super().__init__(text)
        self.number = number
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def __lt__(self, other) -> bool:
        mine, theirs = self.number, getattr(other, "number", None)
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is not None:
            return False  # empty cells sort last
        if mine is not None and theirs is None:
            return True
        return self.text().lower() < other.text().lower()


class ResultsPanel(QtWidgets.QWidget):
    videoActivated = Signal(str)  # item key
    seekRequested = Signal(int)  # frame
    scoreRequested = Signal()
    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QtWidgets.QTabWidget()

        # ---- scores
        self.stale = QtWidgets.QLabel("")
        self.stale.setTextFormat(Qt.RichText)
        self.score_button = QtWidgets.QPushButton("Score all videos")
        self.score_button.clicked.connect(self.scoreRequested.emit)
        self.export_button = QtWidgets.QPushButton("Export to Excel…")
        self.export_button.clicked.connect(self.exportRequested.emit)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.stale, 1)
        bar.addWidget(self.score_button)
        bar.addWidget(self.export_button)
        self.scores = QtWidgets.QTableWidget(0, 0)
        self.scores.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.scores.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.scores.setAlternatingRowColors(True)
        self.scores.verticalHeader().setVisible(False)
        self.scores.verticalHeader().setDefaultSectionSize(24)
        self.scores.setSortingEnabled(True)
        self.scores.horizontalHeader().setHighlightSections(False)
        self.scores.itemSelectionChanged.connect(self._score_row_selected)
        scores_page = QtWidgets.QWidget()
        scores_layout = QtWidgets.QVBoxLayout(scores_page)
        scores_layout.setContentsMargins(6, 6, 6, 6)
        scores_layout.addLayout(bar)
        scores_layout.addWidget(self.scores, 1)
        self.tabs.addTab(scores_page, "Scores")

        # ---- events of the current video
        self.events_note = hint_label("")
        self.events = QtWidgets.QTableWidget(0, 0)
        self.events.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.events.setAlternatingRowColors(True)
        self.events.verticalHeader().setVisible(False)
        self.events.verticalHeader().setDefaultSectionSize(24)
        self.events.itemSelectionChanged.connect(self._event_row_selected)
        events_page = QtWidgets.QWidget()
        events_layout = QtWidgets.QVBoxLayout(events_page)
        events_layout.setContentsMargins(6, 6, 6, 6)
        events_layout.addWidget(self.events_note)
        events_layout.addWidget(self.events, 1)
        self.tabs.addTab(events_page, "Events in this video")

        # ---- log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.tabs.addTab(self.log, "Messages")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._row_keys: list[str] = []
        self._event_frames: list[int] = []
        self.show_table(None, current=False)

    # ------------------------------------------------------------------- scores

    def show_table(self, table: ResultsTable | None, current: bool, problems: list[str] | None = None) -> None:
        self.scores.setSortingEnabled(False)
        self.scores.clear()
        self._row_keys = []
        if table is None or not table.rows:
            self.scores.setRowCount(0)
            self.scores.setColumnCount(0)
            self.stale.setText("Not scored yet. Set up your lines and regions, then press <b>Score all videos</b>.")
            self.export_button.setEnabled(False)
            return
        headers = [table.id_header]
        if table.show_animal:
            headers.append("Animal")
        first_score = len(headers)
        headers += table.columns + ["Tracked %", "Notes"]
        self.scores.setColumnCount(len(headers))
        self.scores.setRowCount(len(table.rows))
        self.scores.setHorizontalHeaderLabels(headers)
        muted = QtGui.QColor(self.palette().color(QtGui.QPalette.WindowText))
        muted.setAlpha(130)
        for r, row in enumerate(table.rows):
            self._row_keys.append(row.video_key)
            label = _SortableItem(row.video_id)
            label.setData(Qt.UserRole, r)
            self.scores.setItem(r, 0, label)
            if table.show_animal:
                self.scores.setItem(r, 1, _SortableItem(row.animal))
            for c, column in enumerate(table.columns):
                value = row.values.get(column)
                number = None if value is None or value != value else float(value)
                cell = _SortableItem(format_number(value), number)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.scores.setItem(r, first_score + c, cell)
            tracked = row.tracked_percent
            cell = _SortableItem("" if tracked is None else f"{tracked:.0f}", tracked)
            cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if tracked is not None and tracked < 80:
                cell.setForeground(QtGui.QColor("#d9822b"))
                cell.setToolTip("The body part used for scoring was tracked on less than 80% of the frames.")
            self.scores.setItem(r, len(headers) - 2, cell)
            notes = row.error or row.notes
            note_cell = _SortableItem(notes)
            note_cell.setToolTip(notes)
            self.scores.setItem(r, len(headers) - 1, note_cell)
            if row.error:
                for c in range(len(headers)):
                    item = self.scores.item(r, c)
                    if item is not None:
                        item.setForeground(muted)
        for c, column in enumerate(table.columns):
            if column in table.formula_errors:
                header = self.scores.horizontalHeaderItem(first_score + c)
                header.setToolTip("Formula problem: " + table.formula_errors[column])
                header.setForeground(QtGui.QColor("#d9534f"))
        self.scores.resizeColumnsToContents()
        self.scores.horizontalHeader().setStretchLastSection(True)
        self.scores.setSortingEnabled(True)
        self.export_button.setEnabled(True)
        count = len({row.video_key for row in table.rows})
        if current:
            text = f"{count} video{'s' if count != 1 else ''} scored."
            if problems:
                text += f" <span style='color:#d9822b'>{len(problems)} thing(s) to check: {problems[0]}</span>"
        else:
            text = "<span style='color:#d9822b'><b>Out of date.</b> The setup or the videos changed since these scores were made.</span>"
        self.stale.setText(text)

    def mark_stale(self) -> None:
        if self.scores.rowCount():
            self.stale.setText(
                "<span style='color:#d9822b'><b>Out of date.</b> The setup or the videos changed since these scores were made.</span>"
            )

    def _score_row_selected(self) -> None:
        items = self.scores.selectedItems()
        if not items:
            return
        first = self.scores.item(items[0].row(), 0)
        index = first.data(Qt.UserRole) if first is not None else None
        if index is not None and 0 <= index < len(self._row_keys):
            self.videoActivated.emit(self._row_keys[index])

    # ------------------------------------------------------------------- events

    def show_events(self, events: list[ScoredEvent], multi_animal: bool, note: str = "") -> None:
        headers = ["Time (s)", "Frame"] + (["Animal"] if multi_animal else []) + ["Shape", "Event", "Body part", "Column", "Points"]
        self.events.blockSignals(True)
        self.events.clear()
        self.events.setColumnCount(len(headers))
        self.events.setHorizontalHeaderLabels(headers)
        self.events.setRowCount(len(events))
        self._event_frames = []
        for r, event in enumerate(events):
            self._event_frames.append(event.frame)
            cells = [f"{event.time_s:.2f}", str(event.frame + 1)]
            if multi_animal:
                cells.append(event.animal)
            what = event.what + (f"  (after {event.gap_frames} untracked frames)" if event.gap_frames else "")
            cells += [event.shape_name, what, event.bodypart, event.column, format_number(event.points)]
            for c, text in enumerate(cells):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if c < 2 or c == len(cells) - 1:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.events.setItem(r, c, cell)
        self.events.resizeColumnsToContents()
        self.events.horizontalHeader().setStretchLastSection(True)
        self.events.blockSignals(False)
        if note:
            self.events_note.setText(note)
        elif events:
            self.events_note.setText("Click a row to jump to that moment in the video and check it.")
        else:
            self.events_note.setText("Nothing has been scored in this video with the current setup.")

    def _event_row_selected(self) -> None:
        items = self.events.selectedItems()
        if items and 0 <= items[0].row() < len(self._event_frames):
            self.seekRequested.emit(self._event_frames[items[0].row()])

    # ---------------------------------------------------------------------- log

    def add_message(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        for line in str(text).splitlines() or [""]:
            self.log.appendPlainText(f"{stamp}  {line}")

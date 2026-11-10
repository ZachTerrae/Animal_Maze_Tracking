"""The list of videos in the batch."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..batch import VideoItem
from .style import hint_label

Qt = QtCore.Qt
Signal = QtCore.Signal


class VideoListPanel(QtWidgets.QWidget):
    currentChanged = Signal(object)  # item key or None
    addFolderRequested = Signal()
    addFilesRequested = Signal()
    removeRequested = Signal(list)
    chooseTrackingRequested = Signal(str)
    automaticTrackingRequested = Signal(str)
    sharedPositionsRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[VideoItem] = []
        self._own_positions: set[str] = set()

        self.summary = QtWidgets.QLabel("No videos yet")
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter by name")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Video", "Tracking"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        add_folder = QtWidgets.QPushButton("Add folder…")
        add_folder.setToolTip("Add every video in a folder (and the folders inside it).")
        add_folder.clicked.connect(self.addFolderRequested.emit)
        add_files = QtWidgets.QPushButton("Add files…")
        add_files.clicked.connect(self.addFilesRequested.emit)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setToolTip("Take the selected videos out of the list. No file is deleted.")
        self.remove_button.clicked.connect(self._remove_selected)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add_folder)
        buttons.addWidget(add_files)
        buttons.addStretch(1)
        buttons.addWidget(self.remove_button)

        self.empty_hint = hint_label(
            "Add the folder that holds the videos you want to score. "
            "Videos that DeepLabCut has already analysed are recognised automatically."
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.summary)
        layout.addWidget(self.filter)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.empty_hint)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ content

    def set_items(self, items: list[VideoItem], own_positions: set[str], keep: str | None = None) -> None:
        self._items = list(items)
        self._own_positions = set(own_positions)
        previous = self.current_key()
        current = keep if keep is not None else previous
        self.tree.blockSignals(True)
        self.tree.clear()
        palette = self.palette()
        good = QtGui.QColor("#2e9e5b")
        wait = QtGui.QColor(palette.color(QtGui.QPalette.WindowText))
        wait.setAlpha(140)
        chosen = None
        for item in self._items:
            row = QtWidgets.QTreeWidgetItem([item.name, ""])
            row.setData(0, Qt.UserRole, item.key)
            tip = [item.video_path or "(no video file; tracking only)"]
            if item.track_path:
                tip.append("Tracking: " + os.path.basename(item.track_path))
            if not item.video_path:
                status, color = "Tracking only", good
            elif not item.track_path:
                status, color = "Not tracked", wait
            elif item.manual_track:
                status, color = "Tracked (chosen file)", good
            elif item.track_matches_model:
                status, color = "Tracked", good
            else:
                status, color = "Tracked", good
                tip.append("This tracking was made by a different model or shuffle than the one selected for tracking.")
            if item.name in self._own_positions:
                status += " · own positions"
                tip.append("This video has its own positions for one or more shapes.")
            row.setText(1, status)
            row.setForeground(1, color)
            row.setToolTip(0, "\n".join(tip))
            row.setToolTip(1, "\n".join(tip))
            self.tree.addTopLevelItem(row)
            if item.key == current:
                chosen = row
        self.tree.blockSignals(False)
        tracked = sum(1 for item in self._items if item.track_path)
        total = len(self._items)
        if total:
            self.summary.setText(f"<b>{total}</b> video{'s' if total != 1 else ''} · {tracked} tracked · {total - tracked} to track")
        else:
            self.summary.setText("No videos yet")
        self.empty_hint.setVisible(total == 0)
        self.remove_button.setEnabled(total > 0)
        self._apply_filter(self.filter.text())
        if chosen is None and self.tree.topLevelItemCount():
            chosen = self.tree.topLevelItem(0)
        if chosen is not None:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(chosen)
            self.tree.blockSignals(False)
        new_key = self.current_key()
        if new_key != previous:
            self.currentChanged.emit(new_key)

    def current_key(self) -> str | None:
        row = self.tree.currentItem()
        return row.data(0, Qt.UserRole) if row else None

    def select_key(self, key: str | None) -> None:
        for index in range(self.tree.topLevelItemCount()):
            row = self.tree.topLevelItem(index)
            if row.data(0, Qt.UserRole) == key:
                self.tree.setCurrentItem(row)
                self.tree.scrollToItem(row)
                return

    def _selected_keys(self) -> list[str]:
        return [row.data(0, Qt.UserRole) for row in self.tree.selectedItems()]

    # ------------------------------------------------------------------ actions

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for index in range(self.tree.topLevelItemCount()):
            row = self.tree.topLevelItem(index)
            row.setHidden(bool(needle) and needle not in row.text(0).lower())

    def _current_changed(self, current, _previous) -> None:
        self.currentChanged.emit(current.data(0, Qt.UserRole) if current else None)

    def _remove_selected(self) -> None:
        keys = self._selected_keys()
        if keys:
            self.removeRequested.emit(keys)

    def _context_menu(self, point: QtCore.QPoint) -> None:
        row = self.tree.itemAt(point)
        if row is None:
            return
        key = row.data(0, Qt.UserRole)
        item = next((i for i in self._items if i.key == key), None)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        choose = menu.addAction("Choose the tracking file by hand…")
        automatic = menu.addAction("Pick the tracking file automatically")
        automatic.setEnabled(item.manual_track)
        shared = menu.addAction("Use the shared shape positions for this video")
        shared.setEnabled(item.name in self._own_positions)
        menu.addSeparator()
        reveal = menu.addAction("Show the file in its folder")
        menu.addSeparator()
        remove = menu.addAction("Remove from the list")
        picked = menu.exec(self.tree.viewport().mapToGlobal(point))
        if picked is choose:
            self.chooseTrackingRequested.emit(key)
        elif picked is automatic:
            self.automaticTrackingRequested.emit(key)
        elif picked is shared:
            self.sharedPositionsRequested.emit(key)
        elif picked is reveal:
            path = item.video_path or item.track_path
            if path:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(path)))
        elif picked is remove:
            keys = self._selected_keys()
            self.removeRequested.emit(keys if key in keys else [key])

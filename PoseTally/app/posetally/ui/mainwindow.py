"""The main window: ties the video list, the drawing surface, the rule editor and the results together."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..batch import VideoResult, default_export_name, describe_item, score_item
from ..dlcproject import ProjectError
from ..export import export_csv, export_xlsx
from ..formulas import evaluate_formulas
from ..geometry import crossing_vector, movement_matches
from ..model import LineShape, RoiShape
from ..scoring import prepare_trajectory
from ..session import Session
from ..tracks import TrackFormatError, TrackSet, tracks_to_dlc_csv
from . import dialogs
from .canvas import ShapeSpec
from .columnspanel import ColumnsPanel
from .resultspanel import ResultsPanel
from .settingspanel import SettingsPanel
from .shapepanel import ShapePanel
from .style import animal_ring, app_icon, app_pixmap, format_number, part_color
from .trackdialog import TrackDialog
from .videolist import VideoListPanel
from .videopane import VideoPane

Qt = QtCore.Qt

GUIDE_HTML = """
<h2>How to use PoseTally</h2>
<ol>
<li><b>Open your DeepLabCut project.</b> Choose the <tt>config.yaml</tt> file in the project folder.
    PoseTally reads the body-part names and finds the trained network.</li>
<li><b>Add your videos.</b> Add a folder (or single files). Videos that DeepLabCut has already analysed
    are recognised; the others are marked “Not tracked”.</li>
<li><b>Track.</b> Press <i>Track</i> to run the trained network over every video that still needs it.
    This is the slow step. You can stop and continue later; finished videos stay finished.</li>
<li><b>Draw lines and regions</b> on the video with the tools above it, then click a shape to set what it scores:
    <ul>
      <li>A <b>line</b> adds points each time the chosen body part crosses it, in the direction you pick.</li>
      <li>A <b>region</b> measures the time spent inside it, or adds points for each entry or exit.</li>
      <li><b>+ Add another score</b> lets one shape feed as many columns as you like.</li>
    </ul>
    Play the video and use the coloured ticks under it to check that events are where you expect them.</li>
<li><b>Score all videos</b>, look at the table, and <b>Export to Excel</b>.</li>
</ol>
<p>Your lines, regions and rules are saved automatically with the project, in a folder called
<tt>PoseTally</tt>, and come back the next time you open it.</p>
<h3>Handy keys</h3>
<p><b>Space</b> play or pause · <b>,</b> and <b>.</b> one frame back or forward · <b>[</b> and <b>]</b> previous or next scored event ·
mouse wheel to zoom · drag the background to pan · <b>0</b> fit the frame · <b>Delete</b> remove the selected shape ·
arrow keys nudge the selected shape (Shift for bigger steps) · <b>Esc</b> cancel a drawing</p>
"""


class ScoreThread(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(bool)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        ok = self.session.score_all(progress=self.progress.emit, cancelled=lambda: self._cancel)
        self.done.emit(ok)


class WelcomePage(QtWidgets.QWidget):
    openRequested = QtCore.Signal()
    demoRequested = QtCore.Signal()
    addVideosRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        icon = QtWidgets.QLabel()
        pixmap = app_pixmap(192)
        pixmap.setDevicePixelRatio(2)
        icon.setPixmap(pixmap)
        icon.setAlignment(Qt.AlignCenter)
        self.title = QtWidgets.QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.text = QtWidgets.QLabel()
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.primary = QtWidgets.QPushButton()
        self.primary.setMinimumHeight(38)
        self.primary.setDefault(True)
        self.secondary = QtWidgets.QPushButton()
        self.secondary.setMinimumHeight(38)
        self.primary.clicked.connect(self._primary_clicked)
        self.secondary.clicked.connect(self._secondary_clicked)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.primary)
        buttons.addWidget(self.secondary)
        buttons.addStretch(1)
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(14)
        column.addStretch(2)
        column.addWidget(icon)
        column.addWidget(self.title)
        column.addWidget(self.text)
        column.addLayout(buttons)
        column.addStretch(3)
        outer = QtWidgets.QHBoxLayout(self)
        outer.addStretch(1)
        outer.addLayout(column, 3)
        outer.addStretch(1)
        self._state = "start"
        self.set_state("start")

    def set_state(self, state: str, project_name: str = "") -> None:
        self._state = state
        if state == "start":
            self.title.setText(f"<h1>{APP_NAME}</h1>")
            self.text.setText(
                "Score animal behaviour from DeepLabCut tracking.<br>"
                "Draw lines and regions on your videos, say what each one is worth, "
                "and get a spreadsheet with one row per video."
            )
            self.primary.setText("Open a DeepLabCut project…")
            self.secondary.setText("Try it with the demo")
            self.secondary.setVisible(True)
        else:
            self.title.setText(f"<h2>{project_name}</h2>")
            self.text.setText(
                "The project is open. Now add the videos you want to score.<br>"
                "Choose the folder that holds them; folders inside it are included."
            )
            self.primary.setText("Add a folder of videos…")
            self.secondary.setVisible(False)

    def _primary_clicked(self) -> None:
        (self.openRequested if self._state == "start" else self.addVideosRequested).emit()

    def _secondary_clicked(self) -> None:
        self.demoRequested.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.session = Session()
        self.prefs = QtCore.QSettings("PoseTally", "PoseTally")
        self._current_key: str | None = None
        self._tracks: TrackSet | None = None
        self._fps = 30.0
        self._size: tuple[int, int] | None = None
        self._live: VideoResult | None = None
        self._score_thread: ScoreThread | None = None
        self._after_scoring = None
        self._prefer_filtered = False

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1440, 900)
        self.setDockNestingEnabled(True)

        self.video_pane = VideoPane()
        self.canvas = self.video_pane.canvas
        self.welcome = WelcomePage()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.video_pane)
        self.setCentralWidget(self.stack)

        self.video_list = VideoListPanel()
        self.shape_panel = ShapePanel()
        self.columns_panel = ColumnsPanel()
        self.settings_panel = SettingsPanel()
        self.results_panel = ResultsPanel()
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.addTab(self.shape_panel, "Lines and regions")
        self.right_tabs.addTab(self.columns_panel, "Columns")
        self.right_tabs.addTab(self.settings_panel, "Settings")
        self.right_tabs.setMinimumWidth(380)

        def dock(title: str, widget: QtWidgets.QWidget, area) -> QtWidgets.QDockWidget:
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(title)
            d.setWidget(widget)
            d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
            self.addDockWidget(area, d)
            return d

        self.dock_videos = dock("Videos", self.video_list, Qt.LeftDockWidgetArea)
        self.dock_scoring = dock("Scoring", self.right_tabs, Qt.RightDockWidgetArea)
        self.dock_results = dock("Results", self.results_panel, Qt.BottomDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.resizeDocks([self.dock_videos, self.dock_scoring], [300, 420], Qt.Horizontal)
        self.resizeDocks([self.dock_results], [230], Qt.Vertical)

        self._build_actions()
        self.project_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.project_label)

        self._rescore_timer = QtCore.QTimer(self)
        self._rescore_timer.setSingleShot(True)
        self._rescore_timer.setInterval(120)
        self._rescore_timer.timeout.connect(self._rescore_current)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(900)
        self._save_timer.timeout.connect(self._save)

        # wiring
        self.welcome.openRequested.connect(self.open_project_dialog)
        self.welcome.demoRequested.connect(self.open_demo)
        self.welcome.addVideosRequested.connect(self.add_folder)
        self.video_list.currentChanged.connect(self._select_video)
        self.video_list.addFolderRequested.connect(self.add_folder)
        self.video_list.addFilesRequested.connect(self.add_files)
        self.video_list.removeRequested.connect(self._remove_videos)
        self.video_list.chooseTrackingRequested.connect(self._choose_tracking_file)
        self.video_list.automaticTrackingRequested.connect(self._automatic_tracking_file)
        self.video_list.sharedPositionsRequested.connect(self._use_shared_positions_for_video)
        self.canvas.shapeDrawn.connect(self._shape_drawn)
        self.canvas.geometryEdited.connect(self._geometry_edited)
        self.canvas.selectionChanged.connect(self._canvas_selection_changed)
        self.canvas.deleteRequested.connect(self._delete_shape)
        self.canvas.toolChanged.connect(lambda _tool: self.canvas.set_draft_color(self.session.layout.next_color()))
        self.video_pane.pathPartChanged.connect(lambda _part: self._update_paths())
        self.shape_panel.edited.connect(self._layout_changed)
        self.shape_panel.selectRequested.connect(self._select_shape)
        self.shape_panel.deleteRequested.connect(self._delete_shape)
        self.shape_panel.sharedPositionRequested.connect(self._use_shared_position)
        self.shape_panel.set_direction_source(self._line_points_on_screen)
        self.columns_panel.edited.connect(self._columns_edited)
        self.settings_panel.edited.connect(self._settings_edited)
        self.results_panel.videoActivated.connect(self.video_list.select_key)
        self.results_panel.seekRequested.connect(self._seek_to_event)
        self.results_panel.scoreRequested.connect(self.score_all)
        self.results_panel.exportRequested.connect(self.export)
        self._update_everything()

    # ------------------------------------------------------------------ actions

    def _build_actions(self) -> None:
        def action(text: str, slot, shortcut=None, tip: str = "") -> QtGui.QAction:
            act = QtGui.QAction(text, self)
            act.triggered.connect(lambda _checked=False: slot())
            if shortcut:
                act.setShortcut(shortcut)
            if tip:
                act.setToolTip(tip)
                act.setStatusTip(tip)
            return act

        self.act_open = action("Open project…", self.open_project_dialog, QtGui.QKeySequence.Open, "Open a DeepLabCut project (its config.yaml file)")
        self.act_add_folder = action("Add video folder…", self.add_folder, None, "Add every video in a folder to the list")
        self.act_add_files = action("Add video files…", self.add_files)
        self.act_track = action("Track videos…", self.track_videos, "Ctrl+T", "Run the trained DeepLabCut network on the videos that are not tracked yet")
        self.act_score = action("Score all videos", self.score_all, "Ctrl+R", "Apply your lines, regions and rules to every video")
        self.act_export = action("Export to Excel…", self.export, "Ctrl+E", "Save the scores as an Excel workbook")
        self.act_save_setup = action("Save scoring setup as…", self.save_setup_as, None, "Save the lines, regions and rules to a file you can reuse or share")
        self.act_load_setup = action("Load scoring setup…", self.load_setup, None, "Replace the lines, regions and rules with a saved setup")
        self.act_export_csv = action("Export tracking as CSV files…", self.export_tracking_csv, None, "Write each video's tracking as a DeepLabCut-style CSV, for SimBA or other tools")
        self.act_demo = action("Open the demo project", self.open_demo)
        self.act_guide = action("Quick guide", self.show_guide, QtGui.QKeySequence.HelpContents)
        self.act_about = action(f"About {APP_NAME}", self.show_about)
        self.act_quit = action("Quit", self.close, QtGui.QKeySequence.Quit)

        menu = self.menuBar().addMenu("&File")
        menu.addAction(self.act_open)
        self.recent_menu = menu.addMenu("Open recent")
        menu.addSeparator()
        menu.addAction(self.act_add_folder)
        menu.addAction(self.act_add_files)
        menu.addSeparator()
        menu.addAction(self.act_save_setup)
        menu.addAction(self.act_load_setup)
        menu.addSeparator()
        menu.addAction(self.act_export)
        menu.addSeparator()
        menu.addAction(self.act_quit)
        tools = self.menuBar().addMenu("&Tools")
        tools.addAction(self.act_track)
        tools.addAction(self.act_score)
        tools.addSeparator()
        tools.addAction(self.act_export_csv)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.act_guide)
        help_menu.addAction(self.act_demo)
        help_menu.addSeparator()
        help_menu.addAction(self.act_about)
        self._rebuild_recent_menu()

        bar = self.addToolBar("Steps")
        bar.setObjectName("Steps")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        bar.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; } QToolButton { padding: 6px 12px; font-weight: 600; }")
        steps = [
            ("1  Open project", self.act_open),
            ("2  Add videos", self.act_add_folder),
            ("3  Track", self.act_track),
            ("4  Score", self.act_score),
            ("5  Export", self.act_export),
        ]
        self.step_buttons: list[tuple[QtWidgets.QToolButton, QtGui.QAction]] = []
        for index, (text, act) in enumerate(steps):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(act.toolTip())
            button.setAutoRaise(True)
            button.clicked.connect(act.trigger)
            bar.addWidget(button)
            self.step_buttons.append((button, act))
            if index < len(steps) - 1:
                arrow = QtWidgets.QLabel("›")
                arrow.setEnabled(False)
                bar.addWidget(arrow)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        bar.addWidget(spacer)
        note = QtWidgets.QLabel("Draw lines and regions on the video at any time.  ")
        note.setEnabled(False)
        bar.addWidget(note)

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self._recent() if os.path.isfile(p)]
        for path in recent:
            name = os.path.basename(os.path.dirname(path))
            act = self.recent_menu.addAction(f"{name}   ({path})")
            act.triggered.connect(lambda _checked=False, p=path: self.open_project(p))
        self.recent_menu.setEnabled(bool(recent))

    def _recent(self) -> list[str]:
        value = self.prefs.value("recent_projects", [])
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in (value or [])]

    def _remember_project(self, path: str) -> None:
        recent = [path] + [p for p in self._recent() if p != path]
        self.prefs.setValue("recent_projects", recent[:8])
        self.prefs.setValue("last_folder", os.path.dirname(path))
        self._rebuild_recent_menu()

    def _update_actions(self) -> None:
        has_project = self.session.project is not None
        has_items = bool(self.session.items)
        self.act_add_folder.setEnabled(has_project)
        self.act_add_files.setEnabled(has_project)
        self.act_track.setEnabled(has_project and has_items)
        self.act_score.setEnabled(has_items)
        self.act_export.setEnabled(has_items)
        self.act_save_setup.setEnabled(has_project)
        self.act_load_setup.setEnabled(has_project)
        self.act_export_csv.setEnabled(any(item.track_path for item in self.session.items))
        for button, act in self.step_buttons:
            button.setEnabled(act.isEnabled())

    # ------------------------------------------------------------------ project

    def open_project_dialog(self) -> None:
        start = str(self.prefs.value("last_folder", "") or os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose the config.yaml of your DeepLabCut project",
            start,
            "DeepLabCut project (config.yaml *.yaml *.yml);;All files (*)",
        )
        if path:
            self.open_project(path)

    def open_project(self, config_path: str) -> bool:
        self._save()
        try:
            self.session.open_project(config_path)
        except ProjectError as exc:
            dialogs.warn(self, "Open project", str(exc))
            return False
        except Exception as exc:
            dialogs.error(self, "Open project", f"The project could not be opened.\n\n{exc}")
            return False
        self._remember_project(self.session.project.config_path)
        self._current_key = None
        self._tracks = None
        self._live = None
        project = self.session.project
        self.results_panel.add_message(f"Opened {project.config_path}")
        self.results_panel.add_message(
            f"{project.describe()} · body parts: {', '.join(project.all_bodyparts()) or '(none listed)'}"
        )
        if project.models:
            self.results_panel.add_message("Trained networks found: " + "; ".join(m.label() for m in project.models))
        else:
            self.results_panel.add_message(
                "No trained network was found in the project folder. Videos that already have DeepLabCut results can still be scored."
            )
        for notice in self.session.notices:
            self.results_panel.add_message(notice)
        if self.session.layout.shapes():
            self.statusBar().showMessage("Your saved lines, regions and rules for this project were restored.", 8000)
        self._update_everything()
        return True

    def _update_everything(self) -> None:
        """After a project was opened or the whole setup was replaced."""
        project = self.session.project
        layout = self.session.layout
        self._prefer_filtered = layout.settings.prefer_filtered
        self.shape_panel.set_bodyparts(self.session.bodyparts())
        self.shape_panel.set_model(layout)
        self.columns_panel.set_model(layout)
        self.settings_panel.set_model(layout, self.session.bodyparts(), project.pcutoff if project else None)
        self.video_pane.set_bodyparts(self.session.bodyparts(), self._default_path_part())
        self.canvas.select(None)
        self.project_label.setText(project.describe() if project else "")
        self.setWindowTitle(f"{project.task} — {APP_NAME}" if project and project.task else APP_NAME)
        self.results_panel.show_table(None, current=False)
        self.results_panel.show_events([], False, "Open a video to see its events.")
        self._items_changed()

    def _default_path_part(self) -> str:
        used = self.session.layout.used_bodyparts()
        return used[0] if used else ""

    # ------------------------------------------------------------------- videos

    def _own_positions(self) -> set[str]:
        return set(self.session.layout.overrides)

    def _items_changed(self, keep: str | None = None) -> None:
        project = self.session.project
        if project is None:
            self.welcome.set_state("start")
            self.stack.setCurrentWidget(self.welcome)
        elif not self.session.items:
            self.welcome.set_state("videos", project.task or "Project")
            self.stack.setCurrentWidget(self.welcome)
        else:
            self.stack.setCurrentWidget(self.video_pane)
        self.video_list.set_items(self.session.items, self._own_positions(), keep)
        if self.video_list.current_key() != self._current_key or not self.session.items:
            self._select_video(self.video_list.current_key())
        self._update_actions()
        self._mark_results_stale()

    def add_folder(self) -> None:
        if self.session.project is None:
            self.open_project_dialog()
            return
        start = str(self.prefs.value("last_video_folder", "") or self.session.project.folder)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the folder with the videos to score", start)
        if folder:
            self.add_folder_path(folder)

    def add_folder_path(self, folder: str) -> None:
        self.prefs.setValue("last_video_folder", folder)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            added, orphans = self.session.add_folder(folder)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if added == 0 and orphans == 0:
            dialogs.info(
                self,
                "Add videos",
                "No new videos were found in that folder.\n\n"
                "Videos need to be .mp4, .avi, .mov, .mkv or a similar format. Videos already in the list are not added twice.",
            )
        else:
            tracked = sum(1 for item in self.session.items if item.track_path)
            text = f"Added {added} video{'s' if added != 1 else ''}"
            if orphans:
                text += f" and {orphans} tracking file{'s' if orphans != 1 else ''} without a video"
            self.statusBar().showMessage(f"{text}. {tracked} of {len(self.session.items)} in the list are tracked.", 10000)
            self.results_panel.add_message(f"{text} from {folder}")
        self._items_changed()
        self._save_timer.start()

    def add_files(self) -> None:
        if self.session.project is None:
            self.open_project_dialog()
            return
        start = str(self.prefs.value("last_video_folder", "") or self.session.project.folder)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose videos", start, "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg *.mpeg *.wmv *.webm *.flv);;All files (*)"
        )
        if paths:
            self.prefs.setValue("last_video_folder", os.path.dirname(paths[0]))
            added = self.session.add_videos(paths)
            self.statusBar().showMessage(f"Added {added} video{'s' if added != 1 else ''}.", 8000)
            self._items_changed()
            self._save_timer.start()

    def _remove_videos(self, keys: list) -> None:
        if len(keys) > 1 and not dialogs.ask(
            self, "Remove videos", f"Take {len(keys)} videos out of the list?\n\nNo file is deleted.", yes="Remove"
        ):
            return
        self.session.remove_items(list(keys))
        self._items_changed()
        self._save_timer.start()

    def _choose_tracking_file(self, key: str) -> None:
        item = self.session.find_item(key)
        if item is None:
            return
        start = os.path.dirname(item.track_path or item.video_path or "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Tracking file for {item.name}", start, "DeepLabCut results (*.h5 *.csv);;All files (*)"
        )
        if path:
            self.session.set_manual_track(item, path)
            self._tracking_changed()

    def _automatic_tracking_file(self, key: str) -> None:
        item = self.session.find_item(key)
        if item is not None:
            self.session.set_manual_track(item, None)
            self._tracking_changed()

    def _tracking_changed(self) -> None:
        """Tracking files appeared or changed: refresh the list and the video being shown."""
        current = self._current_key
        self.video_list.set_items(self.session.items, self._own_positions(), current)
        self._select_video(current)
        self._update_actions()
        self._mark_results_stale()
        self._save_timer.start()

    def current_item(self):
        return self.session.find_item(self._current_key)

    def _video_name(self) -> str | None:
        item = self.current_item()
        return item.name if item else None

    def _select_video(self, key) -> None:
        self._current_key = key
        item = self.session.find_item(key)
        self._tracks = None
        self._live = None
        self.video_pane.positions_mode.setCurrentIndex(0)
        if item is None:
            self.video_pane.clear()
            self.shape_panel.set_video(None)
            self.shape_panel.set_summary("")
            self.results_panel.show_events([], False, "Open a video to see its events.")
            self._refresh_canvas_shapes()
            return
        note = ""
        try:
            self._tracks = self.session.tracks_for(item)
        except TrackFormatError as exc:
            note = str(exc)
            self.results_panel.add_message(f"{item.name}: {exc}")
        except Exception as exc:
            note = f"The tracking file could not be read: {exc}"
            self.results_panel.add_message(f"{item.name}: {exc}")
        fps, size, _notes = describe_item(item, self._tracks, self.session.layout.settings.default_fps)
        self._fps = fps
        self._size = size or (640, 480)
        frames = max(self._tracks.n_frames if self._tracks else 0, item.info.n_frames if item.info else 0)
        project = self.session.project
        parts = self.session.bodyparts()
        self.shape_panel.set_bodyparts(parts)
        self.video_pane.set_bodyparts(parts, self._default_path_part())
        self.video_pane.set_tracking_style(self.session.layout.settings.pcutoff, parts, project.skeleton if project else [])
        self.video_pane.load(item.video_path, self._tracks, fps, self._size, frames)
        self.shape_panel.set_video(item.name)
        self._refresh_canvas_shapes()
        self._update_paths()
        self._rescore_current(note)
        if item.video_path and not self.video_pane.has_picture():
            if item.tracked:
                text = "This video could not be opened for viewing, but its tracking can still be scored."
            else:
                text = "This video file could not be opened. It may be damaged, or in a format that cannot be read."
            self.statusBar().showMessage(text, 8000)
        elif "could not be opened" in self.statusBar().currentMessage():
            self.statusBar().clearMessage()  # that remark was about the video shown before

    # ------------------------------------------------------------------- shapes

    def _line_points_on_screen(self, line: LineShape):
        resolved = self.session.layout.resolve(self._video_name(), self._size)
        for entry in resolved.lines:
            if entry.shape.id == line.id:
                return entry.p1, entry.p2
        return line.p1, line.p2

    def _refresh_canvas_shapes(self) -> None:
        resolved = self.session.layout.resolve(self._video_name(), self._size)
        specs: list[ShapeSpec] = []
        for entry in resolved.rois:
            shape = entry.shape
            specs.append(
                ShapeSpec(shape.id, shape.form, list(entry.points), shape.color, shape.name, entry.custom, (), bool(shape.rules))
            )
        for entry in resolved.lines:
            shape = entry.shape
            sides = set()
            if entry.p1 != entry.p2:
                for rule in shape.rules:
                    for side in (1, -1):
                        mx, my = crossing_vector(entry.p1, entry.p2, side)
                        if movement_matches(rule.direction, mx, my):
                            sides.add(side)
            specs.append(
                ShapeSpec(shape.id, "line", [entry.p1, entry.p2], shape.color, shape.name, entry.custom, tuple(sorted(sides)), bool(shape.rules))
            )
        self.canvas.set_shapes(specs)
        self.canvas.set_draft_color(self.session.layout.next_color())

    def _shape_drawn(self, form: str, geometry: dict) -> None:
        layout = self.session.layout
        if form == "line":
            shape = LineShape(name=layout.next_name("line"), color=layout.next_color())
            layout.lines.append(shape)
        else:
            shape = RoiShape(name=layout.next_name("roi"), form=form, color=layout.next_color())
            layout.rois.append(shape)
        layout.store_geometry(shape.id, geometry, self._video_name(), self._size, only_this_video=False)
        shape.rules.append(self.shape_panel.make_default_rule(shape))
        layout.sync_columns()
        self._refresh_canvas_shapes()
        self._select_shape(shape.id)
        self.right_tabs.setCurrentWidget(self.shape_panel)
        self._layout_changed()
        if not self.session.bodyparts():
            self.statusBar().showMessage("No body parts are known yet. Open a video that has been tracked to choose one.", 8000)

    def _geometry_edited(self, shape_id: str, geometry: dict) -> None:
        only_here = self.video_pane.only_this_video()
        self.session.layout.store_geometry(shape_id, geometry, self._video_name(), self._size, only_here)
        self._refresh_canvas_shapes()
        self.shape_panel.set_video(self._video_name())
        if shape_id == self.shape_panel.selected_id:
            self.shape_panel.refresh_directions()
        self.video_list.set_items(self.session.items, self._own_positions(), self._current_key)
        self._after_change()

    def _canvas_selection_changed(self, shape_id) -> None:
        self.shape_panel.set_selected(shape_id)
        if shape_id:
            self.right_tabs.setCurrentWidget(self.shape_panel)

    def _select_shape(self, shape_id) -> None:
        self.canvas.select(shape_id)
        self.shape_panel.set_selected(shape_id)

    def _delete_shape(self, shape_id: str) -> None:
        shape = self.session.layout.find_shape(shape_id)
        if shape is None:
            return
        count = len(shape.rules)
        if count and not dialogs.ask(
            self,
            "Delete shape",
            f"Delete “{shape.name}” and its {count} score{'s' if count != 1 else ''}?",
            yes="Delete",
        ):
            return
        self.session.layout.remove_shape(shape_id)
        self._select_shape(None)
        self._layout_changed()

    def _use_shared_position(self, shape_id: str) -> None:
        name = self._video_name()
        if name:
            self.session.layout.clear_override(name, shape_id)
            self._refresh_canvas_shapes()
            self.shape_panel.set_video(name)
            self.video_list.set_items(self.session.items, self._own_positions(), self._current_key)
            self._after_change()

    def _use_shared_positions_for_video(self, key: str) -> None:
        item = self.session.find_item(key)
        if item is None:
            return
        self.session.layout.clear_override(item.name)
        self.video_list.set_items(self.session.items, self._own_positions(), self._current_key)
        if key == self._current_key:
            self._refresh_canvas_shapes()
            self.shape_panel.set_video(item.name)
        self._after_change()

    # ---------------------------------------------------------- reacting to edits

    def _after_change(self) -> None:
        self._rescore_timer.start()
        self._save_timer.start()
        self._mark_results_stale()
        self._update_actions()

    def _layout_changed(self) -> None:
        """Shapes or rules changed in the panel on the right."""
        self.session.layout.sync_columns()
        self._refresh_canvas_shapes()
        if self.sender() is not self.shape_panel:  # the panel already shows its own edits
            self.shape_panel.refresh()
        self.columns_panel.refresh()
        self._after_change()

    def _columns_edited(self) -> None:
        self.shape_panel.refresh()
        self._after_change()

    def _settings_edited(self) -> None:
        project = self.session.project
        settings = self.session.layout.settings
        self.video_pane.set_tracking_style(settings.pcutoff, self.session.bodyparts(), project.skeleton if project else [])
        if settings.prefer_filtered != self._prefer_filtered:  # other tracking files may now be picked
            self._prefer_filtered = settings.prefer_filtered
            self.session.refresh_tracking()
            self._tracking_changed()
        self._update_paths()
        self._after_change()

    def _mark_results_stale(self) -> None:
        if self.session.results and not self.session.results_are_current():
            self.results_panel.mark_stale()

    def _update_paths(self) -> None:
        part = self.video_pane.path_bodypart()
        tracks = self._tracks
        paths = []
        if part and tracks is not None:
            order = self.session.bodyparts()
            for index, animal in enumerate(tracks.individuals):
                arr = tracks.get(animal, part)
                if arr is None:
                    continue
                xy = prepare_trajectory(arr, self.session.layout.settings, self._fps).xy
                color = animal_ring(index + 1) if tracks.multi_animal else part_color(order.index(part) if part in order else 0)
                paths.append((xy, color))
        self.canvas.set_paths(paths)

    def _rescore_current(self, note: str = "") -> None:
        item = self.current_item()
        layout = self.session.layout
        if item is None:
            return
        if self._tracks is None:
            self._live = None
            self.video_pane.set_event_marks([])
            message = note or "This video has not been tracked yet. Press “3  Track” to run DeepLabCut on it."
            self.results_panel.show_events([], False, message)
            self.shape_panel.set_summary(f"<i>{message}</i>")
            self.columns_panel.set_samples([i.source for i in self.session.items])
            return
        result = score_item(item, layout, tracks=self._tracks)
        self._live = result
        colors = {shape.id: QtGui.QColor(shape.color) for shape in layout.shapes()}
        events = sorted((e for animal in result.animals for e in animal.events), key=lambda e: (e.frame, e.animal))
        self.video_pane.set_event_marks([(e.frame, colors.get(e.shape_id, QtGui.QColor("#ffffff"))) for e in events])
        self.results_panel.show_events(events, result.multi_animal, result.error or "")
        self.shape_panel.set_summary(self._summary_html(result))
        sample: dict[str, float] = {}
        if result.animals:
            base = {c: result.animals[0].scores.get(c, float("nan")) for c in layout.score_columns()}
            sample, _errors = evaluate_formulas(base, layout.active_formulas())
        self.columns_panel.set_samples([i.source for i in self.session.items], sample)

    def _summary_html(self, result: VideoResult) -> str:
        layout = self.session.layout
        if result.error:
            return f"<i>{result.error}</i>"
        columns = layout.columns()
        if not columns:
            return "<i>No scores yet. Draw a line or region and give it a rule.</i>"
        blocks = []
        for animal in result.animals:
            base = {c: animal.scores.get(c, float("nan")) for c in layout.score_columns()}
            values, _errors = evaluate_formulas(base, layout.active_formulas())
            rows = "".join(
                f"<tr><td>{column}&nbsp;&nbsp;</td><td align='right'><b>{format_number(values.get(column)) or '–'}</b></td></tr>"
                for column in columns
            )
            title = f"<p><b>{animal.animal}</b></p>" if result.multi_animal else ""
            tracked = animal.tracked_percent()
            extra = ""
            if tracked is not None and tracked < 80:
                extra = f"<p style='color:#d9822b'>Only {tracked:.0f}% of the frames are tracked well enough to use.</p>"
            blocks.append(f"{title}<table cellspacing='2'>{rows}</table>{extra}")
        for note in result.notes:
            blocks.append(f"<p><i>{note}</i></p>")
        return "".join(blocks)

    def _seek_to_event(self, frame: int) -> None:
        self.video_pane.pause()
        self.video_pane.seek(frame)

    # ------------------------------------------------------------------ tracking

    def track_videos(self) -> None:
        if self.session.project is None:
            return
        dialog = TrackDialog(self.session, self)
        dialog.exec()
        self._tracking_changed()

    # ------------------------------------------------------------------- scoring

    def score_all(self, then=None, blocking: bool = False) -> None:
        if not self.session.items:
            dialogs.info(self, "Score", "Add some videos first.")
            return
        if not self.session.layout.has_rules():
            dialogs.info(
                self,
                "Score",
                "There is nothing to score yet.\n\nDraw a line or a region on the video and give it a rule, then score.",
            )
            return
        if self._score_thread is not None:
            return
        self._rescore_timer.stop()
        self._after_scoring = then
        if blocking:
            self.session.score_all()
            self._scoring_done(True)
            return
        total = len(self.session.items)
        progress = QtWidgets.QProgressDialog("Scoring videos…", "Cancel", 0, total, self)
        progress.setWindowTitle("Score all videos")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(400)
        thread = ScoreThread(self.session, self)
        self._score_thread = thread
        thread.progress.connect(lambda i, n, name: (progress.setLabelText(f"Scoring {name}   ({i} of {n})"), progress.setValue(i - 1)))
        progress.canceled.connect(thread.cancel)

        def finished(ok: bool) -> None:
            progress.reset()
            progress.deleteLater()
            thread.wait()
            thread.deleteLater()
            self._score_thread = None
            self._scoring_done(ok)

        thread.done.connect(finished)
        thread.start()

    def _scoring_done(self, ok: bool) -> None:
        then, self._after_scoring = self._after_scoring, None
        if not ok:
            self.statusBar().showMessage("Scoring was cancelled.", 6000)
            return
        table = self.session.table()
        problems = self.session.layout.problems(self.session.bodyparts())
        problems += [f"Formula “{name}”: {text}" for name, text in table.formula_errors.items() if not any(name in p for p in problems)]
        self.results_panel.show_table(table, current=True, problems=problems)
        self.results_panel.tabs.setCurrentIndex(0)
        untracked = sum(1 for r in self.session.results.values() if r.error)
        scored = len(self.session.results) - untracked
        message = f"Scored {scored} video{'s' if scored != 1 else ''}."
        if untracked:
            message += f" {untracked} could not be scored (not tracked yet, or the file could not be read)."
        self.statusBar().showMessage(message, 10000)
        self.results_panel.add_message(message)
        for problem in problems:
            self.results_panel.add_message("Check: " + problem)
        if then is not None:
            then()

    # -------------------------------------------------------------------- export

    def export(self) -> None:
        if not self.session.items:
            dialogs.info(self, "Export", "Add some videos first.")
            return
        if not self.session.results_are_current():
            self.score_all(then=self._export_dialog)
        else:
            self._export_dialog()

    def _export_dialog(self) -> None:
        project = self.session.project
        folder = str(self.prefs.value("last_export_folder", "") or (project.folder if project else os.path.expanduser("~")))
        default = os.path.join(folder, default_export_name(project.task if project else None))
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the results", default, "Excel workbook (*.xlsx);;CSV, scores only (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith((".xlsx", ".csv")):
            path += ".csv" if "csv" in chosen.lower() else ".xlsx"
        if self.export_to(path) and dialogs.ask(self, "Export", f"Saved:\n{path}\n\nOpen it now?", yes="Open", no="Close"):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def export_to(self, path: str) -> bool:
        project = self.session.project
        model = self.session.model()
        info = {
            "DeepLabCut project": project.task if project else "",
            "Project file": project.config_path if project else "",
            "Network selected for tracking": model.label() if model else "",
        }
        try:
            if path.lower().endswith(".csv"):
                export_csv(path, self.session.ordered_results(), self.session.layout)
            else:
                export_xlsx(path, self.session.ordered_results(), self.session.layout, info)
        except PermissionError:
            dialogs.error(
                self,
                "Export",
                f"The file could not be written:\n{path}\n\nIf it is open in Excel, close it there and export again.",
            )
            return False
        except Exception as exc:
            dialogs.error(self, "Export", f"The results could not be saved.\n\n{exc}")
            return False
        self.prefs.setValue("last_export_folder", os.path.dirname(path))
        self.results_panel.add_message(f"Results saved to {path}")
        self.statusBar().showMessage(f"Results saved to {path}", 10000)
        return True

    def export_tracking_csv(self) -> None:
        items = [item for item in self.session.items if item.track_path]
        if not items:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for the CSV files", str(self.prefs.value("last_export_folder", "") or os.path.expanduser("~"))
        )
        if folder:
            written, failed = self.export_tracking_csv_to(folder)
            text = f"Wrote {written} CSV file{'s' if written != 1 else ''} to:\n{folder}"
            if failed:
                text += f"\n\n{failed} could not be written; see the Messages tab."
            dialogs.info(self, "Export tracking", text)

    def export_tracking_csv_to(self, folder: str) -> tuple[int, int]:
        items = [item for item in self.session.items if item.track_path]
        progress = QtWidgets.QProgressDialog("Writing CSV files…", "Cancel", 0, len(items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(600)
        written = failed = 0
        for number, item in enumerate(items):
            progress.setValue(number)
            if progress.wasCanceled():
                break
            try:
                tracks = self.session.tracks_for(item)
                tracks_to_dlc_csv(tracks, os.path.join(folder, item.name + ".csv"))
                written += 1
            except Exception as exc:
                failed += 1
                self.results_panel.add_message(f"{item.name}: could not write CSV ({exc})")
        progress.setValue(len(items))
        try:
            with open(os.path.join(folder, "ABOUT THESE FILES.txt"), "w", encoding="utf-8") as handle:
                handle.write(
                    "These files hold DeepLabCut tracking, one file per video, in DeepLabCut's own CSV layout\n"
                    "(header rows: scorer, [individuals,] bodyparts, coords; then one row per frame).\n"
                    "Each file is named after its video. Positions are as DeepLabCut wrote them.\n\n"
                    "They can be imported into other tools that read DeepLabCut CSV files, for example SimBA\n"
                    "(import tracking data, file type: DLC CSV) for behaviour classification.\n"
                    f"Written by {APP_NAME} {__version__}.\n"
                )
        except OSError:
            pass
        return written, failed

    # --------------------------------------------------------------------- setup

    def save_setup_as(self) -> None:
        project = self.session.project
        default = os.path.join(project.folder if project else os.path.expanduser("~"), "scoring setup.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the scoring setup", default, "PoseTally setup (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.session.export_setup(path)
        except OSError as exc:
            dialogs.error(self, "Save setup", f"The setup could not be saved.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Setup saved to {path}", 8000)

    def load_setup(self) -> None:
        project = self.session.project
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load a scoring setup", project.folder if project else "", "PoseTally setup (*.json);;All files (*)"
        )
        if not path:
            return
        if self.session.layout.shapes() and not dialogs.ask(
            self,
            "Load setup",
            "Loading a setup replaces the lines, regions, rules, columns and settings you have now.",
            yes="Replace",
        ):
            return
        self.load_setup_from(path)

    def load_setup_from(self, path: str) -> bool:
        try:
            self.session.import_setup(path)
        except (OSError, ValueError) as exc:
            dialogs.warn(self, "Load setup", f"That file could not be used.\n\n{exc}")
            return False
        key = self._current_key
        self._update_everything()
        self.video_list.select_key(key)
        self._save_timer.start()
        return True

    def _save(self) -> None:
        if self.session.project is None:
            return
        try:
            self.session.save()
        except Exception as exc:  # never let saving the setup interrupt the user
            self.results_panel.add_message(f"The setup could not be saved automatically: {exc}")

    # ---------------------------------------------------------------------- help

    def open_demo(self) -> None:
        from ..demo import make_demo

        folder = os.path.join(os.path.expanduser("~"), "PoseTally Demo")
        config = os.path.join(folder, "T-maze (one animal)", "config.yaml")
        if not os.path.isfile(config):
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                make_demo(folder)
            except Exception as exc:
                QtWidgets.QApplication.restoreOverrideCursor()
                dialogs.error(self, "Demo", f"The demo could not be created in\n{folder}\n\n{exc}")
                return
            QtWidgets.QApplication.restoreOverrideCursor()
        if self.open_project(config):
            self.statusBar().showMessage(
                "This is made-up data. Click a line or region to see its rules, press Play, then “4  Score”.", 15000
            )

    def show_guide(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Quick guide")
        dialog.resize(640, 620)
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(GUIDE_HTML)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(browser)
        layout.addWidget(buttons)
        dialog.exec()

    def show_about(self) -> None:
        dialogs.info(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\n\n"
            "Scores animal tracking made by DeepLabCut with lines and regions that you draw.\n\n"
            "Tracking is done by DeepLabCut (Mathis et al. 2018; Nath et al. 2019; Lauer et al. 2022). "
            "Please cite DeepLabCut when you publish results.",
        )

    def closeEvent(self, event) -> None:
        self._save()
        self.video_pane.close_video()
        super().closeEvent(event)

"""The "Track videos" window: runs the project's trained network over the batch."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..session import Session
from ..worker import parse_line
from . import dialogs
from .style import QuietCombo, hint_label

Qt = QtCore.Qt
Signal = QtCore.Signal

_TQDM = re.compile(r"(\d{1,3})%\|")

DEVICES = [
    ("auto", "Automatic (graphics card if there is one)"),
    ("cuda", "NVIDIA graphics card"),
    ("mps", "Apple graphics (Metal)"),
    ("cpu", "Processor only (slow)"),
]


def worker_python() -> str:
    """The Python that runs the worker: the same one as the window, with a console stream."""
    exe = sys.executable
    if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe":
        candidate = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return exe


def worker_environment() -> QtCore.QProcessEnvironment:
    import posetally

    env = QtCore.QProcessEnvironment.systemEnvironment()
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(posetally.__file__)))
    existing = env.value("PYTHONPATH", "")
    env.insert("PYTHONPATH", app_dir + (os.pathsep + existing if existing else ""))
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("PYTHONIOENCODING", "utf-8")
    return env


def describe_environment(report: dict) -> tuple[str, bool]:
    """A sentence about what is installed, and whether tracking can run at all."""
    if report.get("deeplabcut_error") or not report.get("deeplabcut"):
        detail = report.get("deeplabcut_error", "")
        return (
            "DeepLabCut is not installed in PoseTally's own Python, so videos cannot be tracked here. "
            "Run the PoseTally installer again. Videos that are already tracked can still be scored."
            + (f"<br><span style='font-size:small'>{detail}</span>" if detail else ""),
            False,
        )
    text = f"DeepLabCut {report['deeplabcut']}"
    if report.get("torch"):
        text += f" · PyTorch {report['torch']}"
    if report.get("cuda"):
        text += f" · graphics card: {report.get('cuda_name', 'NVIDIA')}"
    elif report.get("mps"):
        text += " · Apple graphics (Metal) available"
    else:
        text += " · <b>no graphics card found: tracking will run on the processor and be slow</b>"
    return text, True


class TrackDialog(QtWidgets.QDialog):
    trackingChanged = Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Track videos with DeepLabCut")
        self.setMinimumSize(640, 560)
        self.process: QtCore.QProcess | None = None
        self.check: QtCore.QProcess | None = None
        self._job_file: str | None = None
        self._out_buffer = ""
        self._err_buffer = ""
        self._names: dict[str, str] = {}
        self._total = 0
        self._done = 0
        self._failed = 0
        self._errors: list[tuple[str, str]] = []
        self._fatal: dict | None = None
        self._stopped = False
        self._reported_end = False  # the tracking program said it was done (it did not just die)
        self._started_at = 0.0
        self._can_track = True

        project = session.project
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)

        self.model = QuietCombo()
        for model in project.models if project else []:
            self.model.addItem(model.label(), model.key)
        if session.model_key:
            self.model.setCurrentIndex(max(0, self.model.findData(session.model_key)))
        self.model.setToolTip("The trained network (a “shuffle” of the DeepLabCut project) used to track the videos.")
        self.model_note = hint_label("")

        self.device = QuietCombo()
        for key, label in DEVICES:
            self.device.addItem(label, key)
        self.device.setCurrentIndex(max(0, self.device.findData(session.track_options.get("device", "auto"))))

        self.where = QuietCombo()
        self.where.addItem("Next to each video (where DeepLabCut normally puts them)", "video")
        self.where.addItem("In one folder…", "folder")
        self.folder_label = QtWidgets.QLabel("")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder_button = QtWidgets.QPushButton("Choose folder…")
        self.folder_button.clicked.connect(self._choose_folder)
        self._folder = session.results_folder
        self.where.setCurrentIndex(1 if self._folder else 0)
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(self.folder_button)

        self.animals = QtWidgets.QSpinBox()
        self.animals.setRange(0, 200)
        count = len(project.individuals) if project else 0
        self.animals.setSpecialValueText(f"as in the project ({count})" if count else "as in the project")
        self.animals.setValue(int(session.track_options.get("n_animals", 0)))
        self.animals.setToolTip("Set this if the videos hold a different number of animals than the project was set up for.")

        self.batch = QuietCombo()
        self.batch.addItem("As in the project", 0)
        for size in (1, 2, 4, 8, 16, 32):
            self.batch.addItem(f"{size}", size)
        self.batch.setCurrentIndex(max(0, self.batch.findData(int(session.track_options.get("batch_size", 0) or 0))))
        self.batch.setToolTip(
            "How many video frames the network handles in one go. More is faster on a graphics card but needs "
            "more memory. Choose a smaller number if tracking stops because the computer ran out of memory."
        )

        self.save_csv = QtWidgets.QCheckBox("Also save each tracking file as .csv")
        self.save_csv.setChecked(bool(session.track_options.get("save_csv")))
        self.redo_others = QtWidgets.QCheckBox("Track again any video that was tracked with a different network")
        self.redo_others.setChecked(bool(session.track_options.get("retrack_other_models")))

        form = QtWidgets.QFormLayout()
        form.addRow("Network", self.model)
        form.addRow("", self.model_note)
        form.addRow("Run on", self.device)
        form.addRow("Frames at a time", self.batch)
        form.addRow("Save tracking files", self.where)
        form.addRow("", folder_row)
        if project and project.multi_animal:
            form.addRow("Animals per video", self.animals)
        form.addRow("", self.save_csv)
        form.addRow("", self.redo_others)
        self.form_box = QtWidgets.QGroupBox("Options")
        self.form_box.setLayout(form)

        self.environment = QtWidgets.QLabel("Checking DeepLabCut…")
        self.environment.setWordWrap(True)
        self.environment.setTextFormat(Qt.RichText)

        self.stage = QtWidgets.QLabel("")
        self.stage.setWordWrap(True)
        self.overall = QtWidgets.QProgressBar()
        self.current = QtWidgets.QProgressBar()
        self.current.setRange(0, 100)
        self.current.setFormat("this video: %p%")
        self.timing = QtWidgets.QLabel("")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.progress_box = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_box)
        progress_layout.addWidget(self.stage)
        progress_layout.addWidget(self.overall)
        progress_layout.addWidget(self.current)
        progress_layout.addWidget(self.timing)
        progress_layout.addWidget(self.log, 1)
        self.progress_box.setVisible(False)

        self.start_button = QtWidgets.QPushButton("Start tracking")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.form_box)
        layout.addWidget(self.environment)
        layout.addWidget(self.progress_box, 1)
        layout.addLayout(buttons)

        self.model.currentIndexChanged.connect(self._options_changed)
        self.where.currentIndexChanged.connect(self._options_changed)
        self.redo_others.toggled.connect(self._options_changed)
        self._options_changed()
        self._check_environment()

    # ------------------------------------------------------------------ options

    def _choose_folder(self) -> None:
        start = self._folder or (self.session.project.folder if self.session.project else "")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the tracking files", start)
        if folder:
            self._folder = folder
            self.where.setCurrentIndex(1)
            self._options_changed()

    def _apply_options(self) -> None:
        s = self.session
        s.model_key = self.model.currentData()
        s.track_options["device"] = self.device.currentData()
        s.track_options["save_csv"] = self.save_csv.isChecked()
        s.track_options["batch_size"] = int(self.batch.currentData() or 0)
        s.track_options["n_animals"] = int(self.animals.value())
        s.track_options["retrack_other_models"] = self.redo_others.isChecked()
        s.results_folder = self._folder if self.where.currentData() == "folder" and self._folder else None

    def _options_changed(self, *_args) -> None:
        in_folder = self.where.currentData() == "folder"
        self.folder_button.setVisible(in_folder)
        self.folder_label.setVisible(in_folder)
        self.folder_label.setText(self._folder or "No folder chosen yet")
        self._apply_options()
        self.session.refresh_tracking()  # which videos count as tracked depends on the network and folder
        project = self.session.project
        model = self.session.model()
        todo = self.session.items_to_track()
        videos = [item for item in self.session.items if item.video_path]
        if not project or not project.models:
            self.summary.setText(
                "<b>No trained network was found in this DeepLabCut project.</b><br>"
                "Train the network in DeepLabCut first. Videos that already have DeepLabCut results can be scored without it."
            )
        elif not videos:
            self.summary.setText("There are no videos in the list yet. Close this window and add a folder of videos.")
        elif not todo:
            self.summary.setText(f"All {len(videos)} videos are already tracked. There is nothing to do.")
        else:
            self.summary.setText(
                f"<b>{len(todo)}</b> of {len(videos)} videos need tracking. Videos that are already tracked are skipped."
            )
        note = ""
        if model is not None and model.engine == "tensorflow":
            note = (
                "This network was trained with DeepLabCut's older TensorFlow engine. It needs the TensorFlow add-on "
                "(see the README, “Networks trained with TensorFlow”), and on Windows it cannot use the graphics card."
            )
        elif model is not None and model.trainingsetindex is None:
            note = "This network's training fraction is not listed in config.yaml; the first one will be used."
        self.model_note.setText(note)
        self.model_note.setVisible(bool(note))
        self._update_start()

    def _update_start(self) -> None:
        running = self.process is not None
        ready = bool(self.session.model() and self.session.items_to_track() and self._can_track)
        if self.where.currentData() == "folder" and not self._folder:
            ready = False
        self.start_button.setEnabled(ready and not running)
        self.stop_button.setEnabled(running)
        self.form_box.setEnabled(not running)

    # -------------------------------------------------------------- environment

    def _check_environment(self) -> None:
        self.check = QtCore.QProcess(self)
        self.check.setProcessEnvironment(worker_environment())
        self.check.finished.connect(self._check_finished)
        self.check.errorOccurred.connect(lambda _e: self._check_finished())
        self.check.start(worker_python(), ["-u", "-m", "posetally.worker", "--check"])

    def _check_finished(self, *_args) -> None:
        check, self.check = self.check, None
        if check is None:
            return
        text = bytes(check.readAllStandardOutput()).decode("utf-8", "replace")
        report = None
        for line in text.splitlines():
            _, message = parse_line(line)
            if message and message.get("event") == "environment":
                report = message
        if report is None:
            self.environment.setText("Could not check the DeepLabCut installation.")
        else:
            sentence, ok = describe_environment(report)
            self.environment.setText(sentence)
            self._can_track = ok
        check.deleteLater()
        self._update_start()

    # ------------------------------------------------------------------ running

    def start(self) -> None:
        if self.process is not None:
            return
        self._apply_options()
        project, model = self.session.project, self.session.model()
        items = self.session.items_to_track()
        if not project or model is None or not items:
            return
        folder = self.session.results_folder
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                dialogs.warn(self, "Track videos", f"The folder could not be created:\n{exc}")
                return
        if self.check is not None:
            self.check.kill()
            self.check = None
        spec = {
            "config": project.config_path,
            "videos": [item.video_path for item in items],
            "shuffle": model.shuffle,
            "trainingsetindex": model.trainingsetindex or 0,
            "engine": model.engine,
            "destfolder": folder,
            "save_csv": self.save_csv.isChecked(),
            "device": self.device.currentData(),
            "n_animals": int(self.animals.value()) if project.multi_animal else 0,
            "batch_size": int(self.session.track_options.get("batch_size", 0) or 0),
            "multi_animal": project.multi_animal,
            "tag": project.results_tag(model.shuffle),
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="posetally_job_", delete=False, encoding="utf-8")
        with handle:
            json.dump(spec, handle)
        self._job_file = handle.name
        self._names = {os.path.abspath(item.video_path): item.name for item in items}
        self._total, self._done, self._failed = len(items), 0, 0
        self._errors, self._fatal, self._stopped, self._reported_end = [], None, False, False
        self._out_buffer = self._err_buffer = ""
        self._started_at = time.time()
        self.progress_box.setVisible(True)
        self.overall.setRange(0, self._total)
        self.overall.setValue(0)
        self.overall.setFormat("%v of %m videos")
        self.current.setValue(0)
        self.log.clear()
        self.timing.setText("")
        self.stage.setText("Starting…")

        process = QtCore.QProcess(self)
        process.setProcessEnvironment(worker_environment())
        process.readyReadStandardOutput.connect(self._read_output)
        process.readyReadStandardError.connect(self._read_errors)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        self._update_start()
        process.start(worker_python(), ["-u", "-m", "posetally.worker", "--job", self._job_file])

    def stop(self) -> None:
        if self.process is not None:
            self._stopped = True
            self.stage.setText("Stopping…")
            self.process.kill()

    def _read_output(self) -> None:
        if self.process is None:
            return
        self._out_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self._out_buffer:
            line, self._out_buffer = self._out_buffer.split("\n", 1)
            text, message = parse_line(line.rstrip("\r"))
            if text.strip():
                self.log.appendPlainText(text.rstrip())
            if message:
                self._handle(message)

    def _read_errors(self) -> None:
        if self.process is None:
            return
        self._err_buffer += bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        pieces = re.split(r"[\r\n]", self._err_buffer)
        self._err_buffer = pieces.pop()
        for piece in pieces + ([self._err_buffer] if _TQDM.search(self._err_buffer) else []):
            match = _TQDM.search(piece)
            if match:
                self.current.setValue(min(100, int(match.group(1))))
            elif piece.strip():
                self.log.appendPlainText(piece.rstrip())

    def _name(self, video: str) -> str:
        return self._names.get(os.path.abspath(video), os.path.basename(video))

    def _handle(self, message: dict) -> None:
        event = message.get("event")
        if event == "status":
            self.stage.setText(message.get("message", ""))
        elif event == "ready":
            sentence, _ok = describe_environment(message)
            self.environment.setText(sentence)
            self.stage.setText("Loading the network…")
        elif event == "video_start":
            number = min(self._total, self._done + self._failed + 1)
            self.stage.setText(f"Video {number} of {self._total}:  <b>{self._name(message.get('video', ''))}</b>")
            self.current.setValue(0)
        elif event == "video_done":
            self._done += 1
            self._progressed()
        elif event == "video_error":
            self._failed += 1
            name = self._name(message.get("video", ""))
            self._errors.append((name, message.get("error", "")))
            self.log.appendPlainText(f"✗ {name}: {message.get('error', '')}")
            self._progressed()
        elif event == "finished":
            self._reported_end = True
        elif event == "fatal":
            self._reported_end = True
            self._fatal = message
            self.log.appendPlainText("✗ " + message.get("error", ""))
            if message.get("traceback"):
                self.log.appendPlainText(message["traceback"])

    def _progressed(self) -> None:
        finished = self._done + self._failed
        self.overall.setValue(finished)
        elapsed = time.time() - self._started_at
        if 0 < finished < self._total:
            remaining = elapsed / finished * (self._total - finished)
            self.timing.setText(f"Running for {_duration(elapsed)} · about {_duration(remaining)} left")
        else:
            self.timing.setText(f"Running for {_duration(elapsed)}")

    def _process_error(self, error) -> None:
        if error == QtCore.QProcess.FailedToStart:
            self._fatal = {"kind": "internal", "error": "The tracking program could not be started."}
            self._finished(1, QtCore.QProcess.CrashExit)

    def _finished(self, _code=0, _status=None) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        self._read_tail(process)
        process.deleteLater()
        if self._job_file:
            try:
                os.remove(self._job_file)
            except OSError:
                pass
            self._job_file = None
        self.session.refresh_tracking()
        self.trackingChanged.emit()
        elapsed = _duration(time.time() - self._started_at)
        if not self._fatal and not self._stopped and not self._reported_end:
            # it ended without a word: killed by the system (out of memory) or crashed inside the network code
            self._fatal = {"kind": "crashed", "error": f"The tracking program stopped unexpectedly (exit code {_code})."}
            self.log.appendPlainText("✗ " + self._fatal["error"])
        if self._fatal:
            if self._fatal.get("kind") == "crashed":
                self.stage.setText(f"<b>Tracking stopped unexpectedly.</b> {self._done} video(s) were tracked before that.")
            else:
                self.stage.setText("<b>Tracking could not run.</b>")
            dialogs.error(self, "Track videos", explain_fatal(self._fatal))
        elif self._stopped:
            self.stage.setText(f"Stopped. {self._done} video(s) were tracked before stopping.")
        else:
            text = f"<b>Finished in {elapsed}.</b> {self._done} video(s) tracked."
            if self._failed:
                text += f" {self._failed} could not be tracked (see the list below)."
            self.stage.setText(text)
            self.current.setValue(100 if self._done else 0)
        if self._errors and not self._fatal:
            # repeat the failures at the end, where they are seen without scrolling back
            self.log.appendPlainText("")
            self.log.appendPlainText(f"Could not be tracked ({len(self._errors)}):")
            for name, error in self._errors:
                self.log.appendPlainText(f"  ✗ {name}: {error}")
            bar = self.log.verticalScrollBar()
            bar.setValue(bar.maximum())
        self.timing.setText("")
        self._options_changed()

    def _read_tail(self, process: QtCore.QProcess) -> None:
        tail = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in (self._out_buffer + tail).splitlines():
            text, message = parse_line(line)
            if text.strip():
                self.log.appendPlainText(text.rstrip())
            if message:
                self._handle(message)
        self._out_buffer = ""
        # whatever it still had to say on its error channel (the reason for a crash, if it gave one)
        rest = self._err_buffer + bytes(process.readAllStandardError()).decode("utf-8", "replace")
        for piece in re.split(r"[\r\n]", rest):
            if piece.strip() and not _TQDM.search(piece):
                self.log.appendPlainText(piece.rstrip())
        self._err_buffer = ""

    def is_running(self) -> bool:
        return self.process is not None

    def closeEvent(self, event) -> None:
        if self.process is not None:
            if not dialogs.ask(
                self,
                "Track videos",
                "Tracking is still running. Stop it and close?\n\nVideos that are finished stay tracked.",
                yes="Stop and close",
                no="Keep running",
            ):
                event.ignore()
                return
            process = self.process
            self.stop()
            process.waitForFinished(5000)
        if self.check is not None:
            self.check.kill()
            self.check.waitForFinished(2000)
            self.check = None
        self._apply_options()
        super().closeEvent(event)


def _duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def explain_fatal(message: dict) -> str:
    kind = message.get("kind")
    error = str(message.get("error", ""))
    if kind == "no_deeplabcut":
        return (
            "DeepLabCut could not be started from PoseTally's own Python.\n\n"
            "Run the PoseTally installer again; it installs DeepLabCut. "
            "Videos that are already tracked can still be scored.\n\n" + error
        )
    if kind == "crashed":
        return (
            error + "\n\nIt ended without saying why. The usual cause is that the computer ran out of memory "
            "(its own, or the graphics card's). Videos that were finished stay tracked.\n\n"
            "What to try:\n"
            "• close other programs and press “Start tracking” again: it continues with the videos that are left;\n"
            "• set “Frames at a time” to a smaller number, such as 1 or 2;\n"
            "• choose “Processor only” under “Run on”, if the graphics card's memory is the limit."
        )
    text = "DeepLabCut could not start tracking with this network.\n\n" + error
    lowered = error.lower()
    if "tensorflow" in lowered or "tf_slim" in lowered or "tensorpack" in lowered:
        text += (
            "\n\nThis network was trained with DeepLabCut's TensorFlow engine. Either install the TensorFlow add-on "
            "(README: “Networks trained with TensorFlow”) or train the network again with the PyTorch engine."
        )
    elif "snapshot" in lowered or "no such file" in lowered or "not found" in lowered:
        text += "\n\nCheck that the network has finished training and that its snapshot files are in the project folder."
    elif "cuda" in lowered or "out of memory" in lowered:
        text += "\n\nThe graphics card could not be used. Try “Processor only” under “Run on”."
    elif "mps" in lowered:
        text += "\n\nApple graphics could not run this network. Try “Processor only” under “Run on”."
    elif "permission" in lowered or "read-only" in lowered:
        text += "\n\nThe project folder or the video folder cannot be written to. Check the drive is not read-only."
    return text

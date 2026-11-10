"""Starting the program: logging, error reporting and the main window."""

from __future__ import annotations

import logging
import os
import sys
import traceback

from .. import APP_NAME, __version__


def log_folder() -> str:
    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "PoseTally", "logs")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Logs", "PoseTally")
    return os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.join(home, ".local", "state"), "PoseTally")


def log_path() -> str:
    return os.path.join(log_folder(), "posetally.log")


def setup_logging() -> None:
    """Write messages and errors to a file. Needed most when there is no console (Windows shortcut)."""
    handlers: list[logging.Handler] = []
    try:
        os.makedirs(log_folder(), exist_ok=True)
        path = log_path()
        if os.path.isfile(path) and os.path.getsize(path) > 2_000_000:
            os.replace(path, path + ".old")
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    except OSError:
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers or None)
    if sys.stdout is None:  # started without a console: give print() somewhere to go
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def install_error_reporter() -> None:
    """Unexpected errors are logged and shown, instead of closing the window silently."""

    def report(kind, value, trace) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, trace)
            return
        text = "".join(traceback.format_exception(kind, value, trace))
        logging.error("Unexpected error\n%s", text)
        try:
            from PySide6 import QtWidgets

            from . import dialogs

            if QtWidgets.QApplication.instance() is not None and dialogs.QUIET is None:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Warning)
                box.setWindowTitle(APP_NAME)
                box.setText("Something went wrong, but your setup is safe and you can keep working.")
                box.setInformativeText(f"{kind.__name__}: {value}\n\nDetails were written to:\n{log_path()}")
                box.setDetailedText(text)
                box.exec()
        except Exception:
            pass

    sys.excepthook = report


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    setup_logging()
    logging.info("%s %s starting (Python %s)", APP_NAME, __version__, sys.version.split()[0])
    from PySide6 import QtCore, QtWidgets

    from .mainwindow import MainWindow
    from .style import app_icon

    if os.name == "nt":
        try:  # gives the window its own taskbar button and icon instead of Python's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PoseTally.App")
        except Exception:
            pass
    QtCore.QCoreApplication.setOrganizationName("PoseTally")
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    QtCore.QCoreApplication.setApplicationVersion(__version__)
    app = QtWidgets.QApplication(argv)
    app.setApplicationDisplayName(APP_NAME)
    app.setWindowIcon(app_icon())
    install_error_reporter()
    window = MainWindow()
    window.show()
    # a config.yaml given on the command line (or dropped on the launcher) is opened straight away
    for arg in argv[1:]:
        if arg.lower().endswith((".yaml", ".yml")) and os.path.isfile(arg):
            QtCore.QTimer.singleShot(0, lambda path=arg: window.open_project(path))
            break
    return app.exec()

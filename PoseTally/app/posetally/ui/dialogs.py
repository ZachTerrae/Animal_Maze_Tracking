"""Message boxes, gathered in one place so that automated tests can run without them."""

from __future__ import annotations

from PySide6 import QtWidgets

# When this is a list, no message box is shown: each message is appended to it
# instead and questions are answered with QUIET_ANSWER. Used by the tests.
QUIET: list | None = None
QUIET_ANSWER = True


def info(parent, title: str, text: str) -> None:
    if QUIET is not None:
        QUIET.append(("info", title, text))
        return
    QtWidgets.QMessageBox.information(parent, title, text)


def warn(parent, title: str, text: str) -> None:
    if QUIET is not None:
        QUIET.append(("warn", title, text))
        return
    QtWidgets.QMessageBox.warning(parent, title, text)


def error(parent, title: str, text: str) -> None:
    if QUIET is not None:
        QUIET.append(("error", title, text))
        return
    QtWidgets.QMessageBox.critical(parent, title, text)


def ask(parent, title: str, text: str, yes: str = "Yes", no: str = "Cancel") -> bool:
    if QUIET is not None:
        QUIET.append(("ask", title, text))
        return QUIET_ANSWER
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Question)
    box.setWindowTitle(title)
    box.setText(text)
    accept = box.addButton(yes, QtWidgets.QMessageBox.AcceptRole)
    box.addButton(no, QtWidgets.QMessageBox.RejectRole)
    box.setDefaultButton(accept)
    box.exec()
    return box.clickedButton() is accept

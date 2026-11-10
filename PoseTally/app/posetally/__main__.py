"""``python -m posetally`` starts the program."""

import sys


def main() -> int:
    if "--version" in sys.argv[1:]:
        from . import APP_NAME, __version__

        print(f"{APP_NAME} {__version__}")
        return 0
    try:
        from .ui.app import run
    except ImportError as exc:
        message = (
            "PoseTally could not start because a required package is missing:\n"
            f"  {exc}\n"
            "Run the PoseTally installer again to repair the installation."
        )
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        elif sys.platform == "win32":  # started from the shortcut: there is no console to print to
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(None, message, "PoseTally", 0x10)
            except Exception:
                pass
        return 1
    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

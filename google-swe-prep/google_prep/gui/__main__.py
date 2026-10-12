"""`python -m google_prep.gui` and google-prep.exe: open the desktop window.
Any start-up failure is written to ~/.google_prep/gui-error.log and shown in a dialog: the windowed
build has no console, and an uncaught exception there would only show a cryptic bootloader box."""
import os
import sys
import traceback
from pathlib import Path


def _report(text: str) -> None:
    home = Path(os.environ.get("GOOGLE_PREP_HOME") or Path.home() / ".google_prep")
    try:
        home.mkdir(parents=True, exist_ok=True)
        (home / "gui-error.log").write_text(text, encoding="utf-8")
    except OSError:
        pass
    if sys.stdout is not None:
        print(text, flush=True)
    elif os.name == "nt" and "--selftest" not in sys.argv:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, f"Google SWE II OS could not start.\n\n{text[-1500:]}",
                                         "Google SWE II OS", 0x10)


if __name__ == "__main__":
    try:
        from google_prep.gui.launcher import main

        code = main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - last resort: never die silently
        _report(traceback.format_exc())
        code = 1
    sys.exit(code)

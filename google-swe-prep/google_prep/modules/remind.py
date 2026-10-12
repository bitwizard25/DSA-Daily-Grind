"""Print OS scheduler commands that open today's plan at your anchor times.
Nothing is installed automatically: copy the commands you want."""
import os
import sys
from pathlib import Path


def launcher() -> str:
    return sys.executable if getattr(sys, "frozen", False) else "google-prep"


def window_app() -> str | None:
    """The desktop window program next to this terminal app (installed or portable build), if any."""
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable)
    gui = exe.with_name("google-prep.exe" if exe.suffix.lower() == ".exe" else "google-prep")
    return str(gui) if gui.exists() and gui != exe else None


def commands(morning: str, evening: str, platform: str | None = None) -> list[str]:
    platform = platform or ("windows" if os.name == "nt" else "mac" if sys.platform == "darwin" else "linux")
    exe = launcher()
    if platform == "windows":
        gui = window_app()  # open the app window on Today; otherwise a terminal running `today`
        action = f'\\"{gui}\\"' if gui else f'cmd /k \\"{exe}\\" today'
        return [f'schtasks /Create /F /SC DAILY /TN "GooglePrep {label}" /ST {t} /TR "{action}"'
                for label, t in (("Morning", morning), ("Evening", evening))]
    lines = []
    for t in (morning, evening):
        hh, mm = t.split(":")
        if platform == "mac":
            action = f"osascript -e 'tell application \"Terminal\" to do script \"{exe} today\"'"
        else:
            action = f"DISPLAY=:0 notify-send 'Google prep' 'Time for your session: run {exe} today'"
        lines.append(f"{int(mm)} {int(hh)} * * * {action}")
    return lines

"""Print OS scheduler commands that open today's plan at your anchor times.
Nothing is installed automatically: copy the commands you want."""
import os
import sys


def launcher() -> str:
    return sys.executable if getattr(sys, "frozen", False) else "google-prep"


def commands(morning: str, evening: str, platform: str | None = None) -> list[str]:
    platform = platform or ("windows" if os.name == "nt" else "mac" if sys.platform == "darwin" else "linux")
    exe = launcher()
    if platform == "windows":
        return [f'schtasks /Create /F /SC DAILY /TN "GooglePrep {label}" /ST {t} '
                f'/TR "cmd /k \\"{exe}\\" today"'
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

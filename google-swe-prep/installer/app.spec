# -*- mode: python -*-
# The installed app: one folder, two programs sharing one Python runtime.
#   google-prep.exe      the desktop window (no console)
#   google-prep-cli.exe  the terminal app
# A folder build starts faster than a one-file exe (nothing to unpack on each launch).
# Built by `python build.py --installer`; Inno Setup (installer/google-prep.iss) packs the folder.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is set by PyInstaller
ICON = str(ROOT / "build" / "google-prep.ico") if sys.platform == "win32" else None
DATAS = [
    (str(ROOT / "data"), "data"),
    (str(ROOT / "patterns_vault"), "patterns_vault"),
    (str(ROOT / "google_prep" / "gui" / "static"), "google_prep/gui/static"),
]
RICH = collect_submodules("rich")  # rich lazy-loads unicode width tables

gui = Analysis([str(ROOT / "google_prep" / "gui" / "__main__.py")], pathex=[str(ROOT)],  # noqa: F821
               datas=DATAS, hiddenimports=RICH)
cli = Analysis([str(ROOT / "google_prep" / "__main__.py")], pathex=[str(ROOT)],  # noqa: F821
               datas=DATAS, hiddenimports=RICH + collect_submodules("textual"))  # textual lazy-loads widgets

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="google-prep",  # noqa: F821
              console=False, icon=ICON)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="google-prep-cli",  # noqa: F821
              console=True, icon=ICON)
COLLECT(gui_exe, gui.binaries, gui.datas, cli_exe, cli.binaries, cli.datas, name="GoogleSWEPrep")  # noqa: F821

"""Build the executable for the current OS (spec section 8).
`python build.py`          -> dist/google-prep.exe (Windows) / dist/google-prep: one file, as the spec asks.
`python build.py --onedir` -> dist/google-prep/ folder: starts ~2x faster (nothing to unpack per launch)."""
import os
import platform
import subprocess
import sys


def build(onedir: bool = False):
    print(f"Building google-prep for {platform.system()}{' (folder build)' if onedir else ''}...")
    sep = os.pathsep  # PyInstaller's --add-data separator (';' on Windows, ':' elsewhere)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=google-prep",
        "--onedir" if onedir else "--onefile",
        "--noconfirm",
        f"--add-data=data{sep}data",
        f"--add-data=patterns_vault{sep}patterns_vault",
        "--collect-submodules=textual",  # textual lazy-loads its widgets
        "--collect-submodules=rich",  # rich lazy-loads unicode width tables
        "--paths=.",
        "--clean",
        "google_prep/__main__.py",  # cli.py as a script would shadow the google_prep package
    ]
    subprocess.run(cmd, check=True)
    print("\nSuccess! Binary generated in dist/ directory.")


if __name__ == "__main__":
    build(onedir="--onedir" in sys.argv)

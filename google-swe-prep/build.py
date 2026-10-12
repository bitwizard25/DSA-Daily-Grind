"""Build a single-file executable for the current OS (spec section 8).
Windows -> dist/google-prep.exe, macOS/Linux -> dist/google-prep. Run: python build.py"""
import os
import platform
import subprocess
import sys


def build():
    print(f"Building google-prep for {platform.system()}...")
    sep = os.pathsep  # PyInstaller's --add-data separator (';' on Windows, ':' elsewhere)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=google-prep",
        "--onefile",
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
    build()

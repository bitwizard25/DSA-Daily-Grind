"""Build a single-file executable for the current OS: python build_exe.py
Produces dist/google-swe-prep.exe on Windows, dist/google-swe-prep elsewhere."""
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "PyInstaller", "--onefile", "--name", "google-swe-prep",
    "--paths", "src", "entry.py",
])

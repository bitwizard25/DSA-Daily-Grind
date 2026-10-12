"""Build the executables for the current OS (spec section 7).

`python build.py`             -> dist/google-prep(.exe)      the desktop window (no console on Windows)
                                 dist/google-prep-cli(.exe)  the terminal app (every command, scripting)
`python build.py --installer` -> dist/GoogleSWEPrep/ (both programs in one folder, fast start) and, on
                                 Windows with Inno Setup 6, dist/GoogleSWEPrep-Setup.exe: a normal installer
                                 with Start menu and desktop shortcuts and an uninstaller.
`python build.py --onedir`    -> folder builds of the two programs instead of single files.
`python build.py --gui` / `--cli` builds only one of the single-file programs."""
import os
import platform
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _png(size: int) -> bytes:
    """The app icon (four Google-colour dots on a dark tile) as a PNG, drawn without any image library."""
    bg, dots = (0x1E, 0x1E, 0x2E), [((0.34, 0.34), (0x42, 0x85, 0xF4)), ((0.66, 0.34), (0xEA, 0x43, 0x35)),
                                     ((0.34, 0.66), (0xFB, 0xBC, 0x05)), ((0.66, 0.66), (0x34, 0xA8, 0x53))]
    radius, corner = 0.14 * size, 0.22 * size
    rows = []
    for y in range(size):
        row = bytearray([0])  # filter: none
        for x in range(size):
            cx, cy = x + 0.5, y + 0.5
            dx = max(corner - cx, 0, cx - (size - corner))
            dy = max(corner - cy, 0, cy - (size - corner))
            if dx * dx + dy * dy > corner * corner:  # outside the rounded tile
                row += bytes(4)
                continue
            colour = bg
            for (fx, fy), c in dots:
                if (cx - fx * size) ** 2 + (cy - fy * size) ** 2 <= radius * radius:
                    colour = c
            row += bytes(colour) + b"\xff"
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")


def write_icon(path: Path) -> Path:
    """Windows .ico holding PNG images at the sizes Explorer and the taskbar use."""
    images = [(s, _png(s)) for s in (16, 32, 48, 256)]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    for size, data in images:
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out + b"".join(data for _, data in images))
    return path


def build(onedir: bool = False, targets: tuple[str, ...] = ("gui", "cli")):
    system = platform.system()
    print(f"Building google-prep for {system}{' (folder builds)' if onedir else ''}: {', '.join(targets)}...")
    sep = os.pathsep  # PyInstaller's --add-data separator (';' on Windows, ':' elsewhere)
    icon = write_icon(ROOT / "build" / "google-prep.ico")
    common = [
        sys.executable, "-m", "PyInstaller",
        "--onedir" if onedir else "--onefile",
        "--noconfirm", "--clean", "--paths=.",
        f"--add-data=data{sep}data",
        f"--add-data=patterns_vault{sep}patterns_vault",
        "--collect-submodules=rich",  # rich lazy-loads unicode width tables
    ]
    if system == "Windows":
        common.append(f"--icon={icon}")
    if "gui" in targets:
        gui = common + [
            "--name=google-prep",
            f"--add-data=google_prep/gui/static{sep}google_prep/gui/static",
            "--exclude-module=textual",  # the window never uses the terminal UI
            "google_prep/gui/__main__.py",
        ]
        if system == "Windows":
            gui.append("--noconsole")  # a real desktop app: no black console window behind it
        subprocess.run(gui, check=True, cwd=ROOT)
    if "cli" in targets:
        cli = common + [
            "--name=google-prep-cli",
            f"--add-data=google_prep/gui/static{sep}google_prep/gui/static",  # `google-prep-cli ui` opens the window too
            "--collect-submodules=textual",  # textual lazy-loads its widgets
            "google_prep/__main__.py",  # cli.py as a script would shadow the google_prep package
        ]
        subprocess.run(cli, check=True, cwd=ROOT)
    print("\nSuccess! Binaries generated in dist/:")
    for path in sorted((ROOT / "dist").iterdir()):
        print("  ", path.name)


def version() -> str:
    sys.path.insert(0, str(ROOT))
    from google_prep import __version__

    return __version__


def find_iscc() -> str | None:
    """Inno Setup compiler (6 or newer): on PATH, or in its default install folders."""
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return found
    for base in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        root = os.environ.get(base)
        if not root:
            continue
        for pattern in ("Inno Setup */ISCC.exe", "Programs/Inno Setup */ISCC.exe"):  # 6 or newer
            found = sorted(Path(root).glob(pattern), reverse=True)
            if found:
                return str(found[0])
    return None


def build_installer() -> Path | None:
    """The installed app: one folder with both programs, then (Windows) the Setup.exe that installs it."""
    write_icon(ROOT / "build" / "google-prep.ico")
    print(f"Building the app folder dist/GoogleSWEPrep for {platform.system()}...", flush=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(ROOT / "dist"),
                    "--workpath", str(ROOT / "build" / "app"), str(ROOT / "installer" / "app.spec")],
                   check=True, cwd=ROOT)
    if platform.system() != "Windows":
        print("Folder build done. The Setup.exe installer is built on Windows only.")
        return None
    iscc = find_iscc()
    if not iscc:
        raise SystemExit("Inno Setup 6 not found. Install it (https://jrsoftware.org/isdl.php or "
                         "`choco install innosetup`) and run `python build.py --installer` again.")
    subprocess.run([iscc, f"/DAppVersion={version()}", f"/DSourceDir={ROOT / 'dist' / 'GoogleSWEPrep'}",
                    f"/DOutputDir={ROOT / 'dist'}", f"/DIconFile={ROOT / 'build' / 'google-prep.ico'}",
                    str(ROOT / "installer" / "google-prep.iss")], check=True, cwd=ROOT)
    setup = ROOT / "dist" / "GoogleSWEPrep-Setup.exe"
    print(f"\nSuccess! Installer: {setup}")
    return setup


if __name__ == "__main__":
    if "--installer" in sys.argv:
        build_installer()
    else:
        picked = tuple(t for t in ("gui", "cli") if f"--{t}" in sys.argv) or ("gui", "cli")
        build(onedir="--onedir" in sys.argv, targets=picked)

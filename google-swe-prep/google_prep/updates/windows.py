"""Windows installation identity, process guards and detached helper launch."""
import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import cache
from .locking import APP_MUTEX, INSTALL_MUTEX, exists
from .manifest import read_build_info, version_key

UNINSTALL_KEY = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\{03EDDFF2-4DC8-4E0E-84DE-FA73BFFCC5D8}_is1'


@dataclass(frozen=True)
class Installation:
    path: Path
    version: str
    build_number: int


def registered_installation() -> Installation | None:
    if os.name != 'nt': return None
    import winreg
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY, 0, winreg.KEY_READ | view) as key:
                try: location = winreg.QueryValueEx(key, 'InstallLocation')[0]
                except OSError: location = winreg.QueryValueEx(key, 'Inno Setup: App Path')[0]
                version = winreg.QueryValueEx(key, 'DisplayVersion')[0]
            if not isinstance(location, str) or not location or not Path(location).is_absolute(): continue
            path = Path(location)
            if cache.is_link(path): continue
            path = path.resolve()
            if not all((path/name).is_file() and not cache.is_link(path/name)
                       for name in ('google-prep.exe','google-prep-cli.exe','unins000.exe')): continue
            version_key(version, 0)
            info_file = path/'_internal/data/build_info.json'
            if info_file.exists():
                if cache.is_link(path/'_internal') or cache.is_link(path/'_internal/data') or cache.is_link(info_file): continue
                info = read_build_info(path/'_internal')
                if not info['build_number']: continue
                return Installation(path, info['version'], info['build_number'])
            return Installation(path, version, 0)
        except (OSError, ValueError, TypeError):
            continue
    return None


def detect_installation(executable=None):
    if os.name != 'nt' or not getattr(sys, 'frozen', False) or ctypes.sizeof(ctypes.c_void_p) != 8:
        return None
    installed = registered_installation()
    if installed and Path(executable or sys.executable).resolve() == installed.path/'google-prep.exe':
        helper = installed.path/'google-prep-update.exe'
        # Older installers omitted build_info.json. The registered location and
        # updater helper still identify an installed app that can repair itself.
        if helper.is_file() and not cache.is_link(helper):
            return installed
    return None


def app_is_running():
    return exists(APP_MUTEX)


def installation_is_running():
    return exists(INSTALL_MUTEX)


def process_running(pid):
    if not pid: return False
    if os.name != 'nt':
        try: os.kill(pid, 0); return True
        except OSError: return False
    library = ctypes.WinDLL('kernel32', use_last_error=True)
    library.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    library.OpenProcess.restype = ctypes.c_void_p
    library.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    library.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = library.OpenProcess(0x00100000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: conservatively defer
    try: return library.WaitForSingleObject(handle, 0) != 0
    finally: library.CloseHandle(handle)


def hidden_flags():
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def launch_pending(home: Path, parent_pid: int, reopen=False) -> bool:
    if os.name != 'nt': return False
    installed = registered_installation()
    if installed is None: return False
    folder = cache.root(home)
    if not cache.enabled(folder) or installation_is_running(): return False
    candidate, _ = cache.pending(folder)
    if candidate.key <= version_key(installed.version, installed.build_number): return False
    source = installed.path/'google-prep-update.exe'
    if not source.is_file() or cache.is_link(source): return False
    directory = folder/candidate.tag
    if cache.is_link(directory): return False
    helper = directory/'google-prep-update.exe'
    if cache.is_link(helper): return False
    shutil.copy2(source, helper)
    args = [str(helper), '--home', str(home.resolve()), '--parent-pid', str(parent_pid)]
    if reopen: args.append('--reopen')
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=hidden_flags(), close_fds=True)
    return True

"""Standalone stdlib-only updater. Never forcibly closes the app or restarts Windows."""
import argparse
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from google_prep.updates import cache, windows
from google_prep.updates.locking import INSTALL_MUTEX, NamedMutex
from google_prep.updates.manifest import version_key

WAIT_SECONDS = 120


def backup_database(home: Path, folder: Path) -> Path | None:
    file = home/'prep.db'
    if not file.exists(): return None
    if cache.is_link(file): raise ValueError('Progress database must not be a link')
    directory = folder/'backups'
    if cache.is_link(directory): raise ValueError('Backup folder must not be a link')
    directory.mkdir(exist_ok=True)
    backup = directory/f'prep-{time.time_ns()}-{uuid.uuid4().hex}.db'
    with sqlite3.connect(file.resolve().as_uri()+'?mode=ro', uri=True, timeout=10) as source:
        with sqlite3.connect(backup) as destination:
            source.backup(destination)
            if destination.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Progress backup could not be verified')
    return backup


def install_pending(home: Path, parent_pid: int, reopen=False) -> int:
    folder, tag = None, None
    try:
        folder = cache.root(home)
        candidate, installer = cache.pending(folder)
        tag = candidate.tag
        if not cache.enabled(folder): return 1
        try: result = cache.read(folder/'result.json')
        except (OSError, ValueError): result = {}
        with NamedMutex(INSTALL_MUTEX) as acquired:
            if not acquired: return 1
            deadline = time.monotonic()+WAIT_SECONDS
            while windows.process_running(parent_pid) or windows.app_is_running():
                if time.monotonic() >= deadline: return 1
                time.sleep(.5)
            installed = windows.registered_installation()
            if installed is None: raise ValueError('Registered app installation was not found')
            if candidate.key <= version_key(installed.version, installed.build_number): return 0
            if isinstance(result, dict) and result.get('tag') == tag and result.get('status') in ('failed','restart_required','installed'):
                return 1  # next launch confirms success; failed installs require an explicit retry
            if not cache.enabled(folder): return 1
            cache.verify(installer, candidate)
            backup = backup_database(home, folder)
            fresh = windows.registered_installation()
            if fresh is None or fresh != installed or windows.app_is_running(): return 1
            log = folder/candidate.tag/'install.log'
            if cache.is_link(log): raise ValueError('Update log must not be a link')
            args = [str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-', '/NORESTART',
                    '/NOCLOSEAPPLICATIONS', '/NORESTARTAPPLICATIONS', '/RESTARTEXITCODE=3010',
                    f'/DIR={installed.path}', f'/LOG={log}']
            completed = subprocess.run(args, timeout=300, creationflags=windows.hidden_flags())
            code = completed.returncode
            status = 'installed' if code == 0 else 'restart_required' if code == 3010 else 'failed'
            message = 'Update installed. The next launch will confirm its version.' if code == 0 else \
                      'Windows needs a restart to finish this update.' if code == 3010 else \
                      'The update installer failed. Check again or run the cached installer manually.'
            cache.write(folder/'result.json', {'tag':tag,'status':status,'code':code,'message':message,
                                             'backup':str(backup) if backup else None})
        if code == 0 and reopen:
            subprocess.Popen([str(installed.path/'google-prep.exe')], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=windows.hidden_flags())
        return 0 if code == 0 else 1
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.TimeoutExpired):
        if folder:
            try:
                cache.write(folder/'result.json', {'tag':tag,'status':'failed',
                            'message':'Update could not finish. Your progress backup and downloaded installer are retained.'})
            except (OSError, ValueError): pass
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='Google SWE II OS update helper')
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--reopen', action='store_true')
    args = parser.parse_args(argv)
    if os.name != 'nt' or args.parent_pid < 0: return 2
    return install_pending(args.home, args.parent_pid, args.reopen)


if __name__ == '__main__':
    raise SystemExit(main())

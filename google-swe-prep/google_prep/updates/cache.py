"""Atomic updater state and strictly bounded cache paths."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .manifest import APP_ID, candidate_from_dict


def is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 1024)
    except FileNotFoundError:
        return False


def root(home: Path) -> Path:
    home = home.resolve()
    folder = home/'updates'
    if is_link(folder) or not folder.resolve().is_relative_to(home):
        raise ValueError('Update cache must stay inside the study profile')
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read(path: Path):
    if is_link(path):
        raise ValueError('Update state must not be a link')
    with path.open('rb') as stream:
        content = stream.read(64*1024+1)
    if len(content) > 64*1024:
        raise ValueError('Update state is too large')
    return json.loads(content)


def write(path: Path, data) -> None:
    if is_link(path) or is_link(path.parent):
        raise ValueError('Update state must not be a link')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def enabled(folder: Path) -> bool:
    try:
        value = read(folder/'preferences.json')
        return isinstance(value, dict) and value.get('enabled') is True
    except FileNotFoundError:
        return True
    except (OSError, ValueError, TypeError):
        return False


def installer_path(folder: Path, candidate) -> Path:
    directory = folder/candidate.tag
    file = directory/'installer.exe'
    if is_link(directory) or is_link(file) or not file.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Installer must stay inside the update cache')
    return file


def verify(file: Path, candidate) -> None:
    if is_link(file) or not file.is_file() or file.stat().st_size != candidate.installer_size:
        raise ValueError('Update installer is missing or its size changed')
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        while chunk := stream.read(64*1024):
            digest.update(chunk)
    if digest.hexdigest() != candidate.installer_sha256:
        raise ValueError('Update installer checksum changed')


def pending(folder: Path, verify_file=True):
    data = read(folder/'pending.json')
    if not isinstance(data, dict) or type(data.get('schema')) is not int or data.get('schema') != 1 or data.get('app_id') != APP_ID:
        raise ValueError('Invalid pending update')
    candidate = candidate_from_dict(data['candidate'])
    file = installer_path(folder, candidate)
    if verify_file:
        verify(file, candidate)
    return candidate, file

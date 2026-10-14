import importlib
import json
import sqlite3
import subprocess

import pytest

from google_prep.updates import cache
from google_prep.updates.manifest import APP_ID, Candidate, read_build_info

DATA = {'version':'1.6.0','build_number':100,'platform':'windows-x64','schema':1,'app_id':APP_ID,'commit':'a'*40}
CANDIDATE = Candidate('1.6.0',100,'a'*40,'google-prep-v1.6.0-b100',
    'https://github.com/bitwizard25/DSA-Daily-Grind/releases/download/google-prep-v1.6.0-b100/GoogleSWEPrep-Setup.exe',
    3,'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


def test_helper_contract_exists():
    try: helper=importlib.import_module('google_prep.updates.helper')
    except ModuleNotFoundError: pytest.fail('The installer helper is not implemented')
    assert callable(helper.install_pending) and callable(helper.backup_database)


@pytest.fixture
def prepared(home, monkeypatch):
    try:
        helper=importlib.import_module('google_prep.updates.helper')
        windows=importlib.import_module('google_prep.updates.windows')
    except ModuleNotFoundError:
        pytest.fail('The safe Windows installer helper is missing')
    target=home/'App with spaces and Raj';target.mkdir()
    folder=cache.root(home);binary=cache.installer_path(folder,CANDIDATE)
    binary.parent.mkdir();binary.write_bytes(b'abc')
    cache.write(folder/'pending.json',{'schema':1,'app_id':APP_ID,'candidate':CANDIDATE.as_dict()})
    def installed():
        info=read_build_info(target/'_internal')
        return windows.Installation(target,info['version'],info['build_number'])
    monkeypatch.setattr(windows,'registered_installation',installed)
    monkeypatch.setattr(windows,'app_is_running',lambda:False)
    monkeypatch.setattr(windows,'process_running',lambda pid:False)
    monkeypatch.setattr(helper,'WAIT_SECONDS',0)
    return helper,windows,home,target,folder,binary


def test_running_app_defers_without_install(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared
    monkeypatch.setattr(windows,'app_is_running',lambda:True)
    monkeypatch.setattr(helper.subprocess,'run',lambda *a,**k:pytest.fail('installer launched'))
    assert helper.install_pending(home,0) != 0


def test_changed_installer_is_rejected(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared;binary.write_bytes(b'bad')
    monkeypatch.setattr(helper.subprocess,'run',lambda *a,**k:pytest.fail('installer launched'))
    assert helper.install_pending(home,0) != 0


def test_disabled_updates_cannot_install(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared
    cache.write(folder/'preferences.json',{'enabled':False})
    monkeypatch.setattr(helper.subprocess,'run',lambda *a,**k:pytest.fail('installer launched'))
    assert helper.install_pending(home,0) != 0


def test_success_uses_argument_list_and_preserves_data(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared
    original=(home/'prep.db').read_bytes()
    calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs))
        metadata=target/'_internal/data/build_info.json';metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps(DATA))
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(helper.subprocess,'run',run)
    assert helper.install_pending(home,0) == 0
    args,kwargs=calls[0]
    assert args[0] == str(binary) and f'/DIR={target}' in args
    assert '/NOCLOSEAPPLICATIONS' in args and '/NORESTART' in args
    assert not kwargs.get('shell') and kwargs['timeout'] == 300
    assert (home/'prep.db').read_bytes() == original
    assert list((folder/'backups').glob('*.db'))
    assert helper.install_pending(home,0) == 0 and len(calls) == 1


@pytest.mark.parametrize('code',[8,3010])
def test_nonzero_result_is_recorded_without_repeated_install(prepared,monkeypatch,code):
    helper,windows,home,target,folder,binary=prepared
    calls=[]
    def run(args,**kwargs): calls.append(args);return subprocess.CompletedProcess(args,code)
    monkeypatch.setattr(helper.subprocess,'run',run)
    assert helper.install_pending(home,0) != 0
    result=cache.read(folder/'result.json')
    assert result['status'] == ('restart_required' if code == 3010 else 'failed')
    assert (folder/'pending.json').exists()
    assert helper.install_pending(home,0) != 0 and len(calls) == 1


def test_timeout_records_failure(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared
    def run(*args,**kwargs): raise subprocess.TimeoutExpired('installer',300)
    monkeypatch.setattr(helper.subprocess,'run',run)
    assert helper.install_pending(home,0) != 0
    assert cache.read(folder/'result.json')['status'] == 'failed'


def test_failed_backup_prevents_install(prepared,monkeypatch):
    helper,windows,home,target,folder,binary=prepared
    (home/'prep.db').write_bytes(b'not a database')
    monkeypatch.setattr(helper.subprocess,'run',lambda *a,**k:pytest.fail('installer launched'))
    assert helper.install_pending(home,0) != 0


def test_wal_backup_contains_committed_changes(prepared):
    helper,windows,home,target,folder,binary=prepared
    with sqlite3.connect(home/'prep.db') as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE ota_evidence (note TEXT)')
        connection.execute("INSERT INTO ota_evidence VALUES ('keep this')");connection.commit()
        file=helper.backup_database(home,folder)
        with sqlite3.connect(file) as backup:
            assert backup.execute('SELECT note FROM ota_evidence').fetchone()[0] == 'keep this'


def test_source_and_portable_detection_never_install():
    try: windows=importlib.import_module('google_prep.updates.windows')
    except ModuleNotFoundError: pytest.fail('Installed-mode detection is missing')
    assert windows.detect_installation() is None

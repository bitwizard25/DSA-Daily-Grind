import importlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from google_prep.updates.manifest import Candidate

BUILD = {'version':'1.6.0','build_number':99,'platform':'windows-x64','app_id':'google-swe-prep'}
CANDIDATE = Candidate('1.6.0',100,'a'*40,'google-prep-v1.6.0-b100',
    'https://github.com/bitwizard25/DSA-Daily-Grind/releases/download/google-prep-v1.6.0-b100/GoogleSWEPrep-Setup.exe',
    3,'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


def manager(home, client, installed=True, **kwargs):
    try: cls = importlib.import_module('google_prep.updates.manager').UpdateManager
    except ModuleNotFoundError: pytest.fail('The background updater is missing')
    return cls(home, BUILD, SimpleNamespace(path=home/'app') if installed else None, client=client, **kwargs)


def wait_state(subject, expected):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        if subject.status()['state'] in expected: return subject.status()
        time.sleep(.01)
    pytest.fail(str(subject.status()))


class Client:
    calls = 0
    downloads = 0
    def fetch_candidate(self, build): self.calls+=1; return CANDIDATE
    def download_candidate(self, candidate, directory, progress):
        self.downloads+=1; directory.mkdir(parents=True,exist_ok=True)
        (directory/'installer.exe').write_bytes(b'abc')
        progress(3,3)
        return directory/'installer.exe'


def test_download_is_staged_but_not_installed(home):
    client = Client(); subject = manager(home, client)
    subject.start()
    state = wait_state(subject, {'ready'})
    subject.stop()
    assert state['supported'] and state['candidate']['build_number'] == 100
    assert subject.pending().name == 'pending.json'
    assert json.loads(subject.pending().read_text())['candidate']['tag'] == CANDIDATE.tag


def test_opt_out_persists_and_manual_check_does_not_download(home):
    client = Client(); subject = manager(home, client)
    subject.set_enabled(False); subject.stop()
    reopened = manager(home, client)
    assert not reopened.status()['enabled']
    reopened.check(); wait_state(reopened, {'available'}); reopened.stop()
    assert client.downloads == 0


def test_corrupt_preference_does_not_reenable(home):
    path = home/'updates/preferences.json';path.parent.mkdir();path.write_text('invalid')
    subject = manager(home, Client())
    assert not subject.status()['enabled']


def test_source_mode_never_downloads(home):
    client = Client();subject = manager(home, client, installed=False)
    subject.start();time.sleep(.02)
    assert client.calls == 0 and subject.status()['state'] == 'unsupported'
    subject.check();wait_state(subject, {'available'});subject.stop()
    assert client.downloads == 0


def test_request_returns_immediately_and_does_not_overlap(home):
    entered, release = threading.Event(), threading.Event()
    class Blocking(Client):
        def fetch_candidate(self, build):
            self.calls+=1;entered.set();release.wait(3);return None
    client = Blocking(); subject = manager(home, client)
    before = time.monotonic();subject.check()
    assert time.monotonic()-before < .5 and entered.wait(1)
    subject.check();release.set();wait_state(subject, {'idle'});subject.stop()
    assert client.calls == 1


def test_disabling_during_download_prevents_pending_install(home):
    entered, release = threading.Event(), threading.Event()
    class Blocking(Client):
        def download_candidate(self, *args, **kwargs):
            entered.set();release.wait(3);return super().download_candidate(*args, **kwargs)
    client=Blocking(); subject=manager(home, client);subject.start()
    assert entered.wait(1)
    subject.set_enabled(False);release.set();wait_state(subject, {'available','idle'});subject.stop()
    assert subject.pending() is None


def test_offline_failure_keeps_study_independent(home):
    class Offline(Client):
        def fetch_candidate(self, build): raise OSError('offline')
    subject=manager(home, Offline());subject.start()
    assert wait_state(subject, {'offline'})['last_check']
    subject.stop()


def test_ready_update_survives_restart(home):
    subject=manager(home, Client());subject.start();wait_state(subject, {'ready'});subject.stop()
    reopened=manager(home, Client())
    assert reopened.status()['state'] == 'ready' and reopened.pending() is not None


def test_corrupt_pending_state_cannot_install(home):
    path=home/'updates/pending.json';path.parent.mkdir();path.write_text('{bad')
    subject=manager(home, Client())
    assert subject.pending() is None and subject.status()['state'] == 'failed'


def test_retry_schedule_is_bounded():
    try: module=importlib.import_module('google_prep.updates.manager')
    except ModuleNotFoundError: pytest.fail('The retry policy is missing')
    assert [module.retry_delay(n) for n in range(1,7)] == [60,300,900,3600,3600,3600]
    assert module.retry_delay(1,7200) == 7200


def test_second_instance_does_not_overlap_download(home):
    from google_prep.updates.locking import NamedMutex, UPDATE_MUTEX
    client=Client();subject=manager(home, client)
    with NamedMutex(UPDATE_MUTEX) as acquired:
        assert acquired
        subject.start()
        assert wait_state(subject, {'idle'})['message']
        time.sleep(.05)
        assert client.calls == 0
        subject.stop()


def test_worker_honors_github_rate_limit(home):
    from google_prep.updates.client import RateLimited
    class Limited(Client):
        def fetch_candidate(self, build): raise RateLimited(7200)
    subject=manager(home, Limited());subject.start()
    assert 'wait' in wait_state(subject, {'offline'})['message']
    subject.stop()


def test_source_opening_an_installed_cache_offers_manual_download(home):
    subject = manager(home, Client()); subject.start(); wait_state(subject, {'ready'}); subject.stop()
    source = manager(home, Client(), installed=False)
    assert source.status()['state'] == 'available'
    assert 'manual' in source.status()['message']
    assert source.status()['release_url'].endswith(CANDIDATE.tag)
    assert source.pending() is None

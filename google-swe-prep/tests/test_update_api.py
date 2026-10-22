from pathlib import Path

import pytest

from google_prep.gui.api import Api, ApiError
from google_prep.gui import launcher
from google_prep.updates import windows


class FakeManager:
    stopped=False
    calls=0
    def status(self): return {'supported':True,'enabled':True,'state':'idle','version':'1.6.0','build_number':100}
    def check(self): self.calls+=1;return {**self.status(),'state':'checking'}
    def set_enabled(self, enabled): return {**self.status(),'enabled':enabled}
    def stop(self): self.stopped=True
    def pending(self): return Path('pending.json')


def test_update_rpc_is_nonblocking_delegation():
    fake=FakeManager();api=Api(updater=fake)
    assert api.update_status()['build_number'] == 100
    assert api.update_check()['state'] == 'checking' and fake.calls == 1
    assert api.update_preferences(False)['enabled'] is False


@pytest.mark.parametrize('value',['true',1,None])
def test_update_preference_requires_boolean(value):
    api=Api(updater=FakeManager())
    with pytest.raises(ApiError): api.update_preferences(value)


def test_plain_api_has_no_automatic_updater():
    assert Api().update_status()['supported'] is False
    with pytest.raises(ApiError): Api().update_preferences(True)


@pytest.mark.parametrize('reason',['heartbeat_timeout','browser_handoff','startup_failure','keyboard_interrupt'])
def test_uncertain_exit_does_not_install(monkeypatch,reason):
    fake=FakeManager()
    monkeypatch.setattr(windows,'launch_pending',lambda *a,**k:pytest.fail('unsafe install'))
    close=getattr(launcher,'_close_updates',None)
    assert callable(close),'The launcher update guard is missing'
    close(fake,reason)
    assert fake.stopped


def test_explicit_window_close_schedules_detached_helper(monkeypatch):
    fake=FakeManager();calls=[]
    monkeypatch.setattr(windows,'launch_pending',lambda *a,**k:calls.append((a,k)))
    close=getattr(launcher,'_close_updates',None)
    assert callable(close),'The launcher update guard is missing'
    close(fake,'window_closed')
    assert fake.stopped and len(calls) == 1


def test_legacy_updater_uses_registered_version(home, monkeypatch):
    from google_prep.updates import manifest
    installed = windows.Installation(home/'app', '1.5.0', 0)
    monkeypatch.setattr(windows, 'detect_installation', lambda:installed)
    monkeypatch.setattr(manifest, 'read_build_info', lambda root:{'version':'1.6.0', 'build_number':0})
    updater = launcher._make_updater()
    assert updater.status()['supported']
    assert updater.status()['version'] == '1.5.0'

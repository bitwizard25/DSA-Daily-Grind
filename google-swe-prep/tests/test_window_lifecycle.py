"""Closing the page must release the app even if Edge keeps its process alive."""
import http.client
import json
import os
from types import SimpleNamespace

import pytest

from google_prep.gui import launcher
from google_prep.gui.server import serve


def presence(server, window_id, is_open, token=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    try:
        conn.request('POST', '/api/window_presence', json.dumps({'id':window_id, 'open':is_open}),
                     {'X-Prep-Token':token or server.token, 'Content-Type':'application/json'})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


@pytest.fixture
def server(home):
    instance = serve()
    yield instance
    instance.shutdown()
    instance.server_close()


def test_only_authenticated_last_window_close_releases_the_app(server, monkeypatch):
    monkeypatch.setattr('google_prep.gui.server.WINDOW_CLOSE_GRACE', 0)
    assert presence(server, 'window-a', True)[0] == 200
    assert presence(server, 'window-b', True)[0] == 200
    assert presence(server, 'window-a', False)[0] == 200
    assert not server.windows_closed()
    assert presence(server, 'window-b', False, token='wrong')[0] == 403
    assert not server.windows_closed()
    assert presence(server, 'window-b', False)[0] == 200
    assert server.windows_closed()


def test_reload_grace_and_late_messages_do_not_close_an_open_page(server, monkeypatch):
    clock = {'now':100.0}
    monkeypatch.setattr('google_prep.gui.server.time.monotonic', lambda:clock['now'])
    assert presence(server, 'old-page', True)[0] == 200
    assert presence(server, 'old-page', False)[0] == 200
    assert not server.windows_closed()
    assert presence(server, 'new-page', True)[0] == 200
    clock['now'] += 60
    assert not server.windows_closed()
    presence(server, 'new-page', False)
    clock['now'] += 60
    presence(server, 'old-page', True)  # delayed request from the closed document
    assert server.windows_closed()


def test_no_close_signal_never_means_window_closed(server):
    assert not server.windows_closed()


def test_rejected_close_requests_keep_returning_forbidden_without_resetting_windows(server):
    presence(server, 'still-open', True)
    for _ in range(32):
        assert presence(server, 'still-open', False, token='wrong')[0] == 403
    assert not server.windows_closed()
    assert presence(server, '', True)[0] == 400
    assert presence(server, 'window-a', 'false')[0] == 400
    assert not server.windows_closed()


def test_lingering_browser_process_does_not_block_confirmed_close(monkeypatch):
    server = SimpleNamespace(windows_closed=lambda:True)
    process = SimpleNamespace(poll=lambda:None)
    assert launcher._wait_for_browser(process, server) is True


def test_handoff_close_is_explicit_so_it_can_schedule_the_updater(monkeypatch):
    server = SimpleNamespace(last_seen=0, windows_closed=lambda:True)
    assert launcher._wait_for_page(server) == 'window_closed'


def test_minimized_page_keeps_its_server_until_a_real_close(monkeypatch):
    polls = {'n':0}
    def poll():
        polls['n'] += 1
        return None if polls['n'] < 3 else 0
    monkeypatch.setattr(launcher.time, 'sleep', lambda _:None)
    server = SimpleNamespace(windows_closed=lambda:False)
    assert launcher._wait_for_browser(SimpleNamespace(poll=poll), server) is False
    assert polls['n'] == 3


def test_browser_without_close_event_stops_after_its_last_native_window_disappears(monkeypatch):
    clock = {'now':0.0}
    monkeypatch.setattr(launcher.time, 'monotonic', lambda:clock['now'])
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds:clock.update(now=clock['now'] + seconds))
    monkeypatch.setattr(launcher, '_browser_has_window', lambda _:clock['now'] < 1)
    process = SimpleNamespace(pid=123, poll=lambda:None)
    assert launcher._wait_for_browser(process, SimpleNamespace(windows_closed=lambda:False))
    assert clock['now'] >= 1 + launcher.NATIVE_CLOSE_GRACE


def test_browser_startup_without_a_window_is_not_mistaken_for_a_closed_window(monkeypatch):
    polls = {'n':0}
    def poll():
        polls['n'] += 1
        return None if polls['n'] < 5 else 0
    monkeypatch.setattr(launcher.time, 'sleep', lambda _:None)
    monkeypatch.setattr(launcher, '_browser_has_window', lambda _:False)
    process = SimpleNamespace(pid=123, poll=poll)
    assert not launcher._wait_for_browser(process, SimpleNamespace(windows_closed=lambda:False))
    assert polls['n'] == 5


def test_replacement_browser_heartbeats_keep_the_server_alive(monkeypatch):
    clock = {'now':0.0}
    server = SimpleNamespace(last_seen=0, windows_closed=lambda:False)
    def sleep(seconds):
        clock['now'] += seconds
        if int(clock['now']) % 30 == 0:
            server.last_seen = clock['now']
    monkeypatch.setattr(launcher.time, 'monotonic', lambda:clock['now'])
    monkeypatch.setattr(launcher.time, 'sleep', sleep)
    monkeypatch.setattr(launcher, '_browser_has_window', lambda _:clock['now'] < 1)
    process = SimpleNamespace(pid=123, poll=lambda:None if clock['now'] < 120 else 0)
    assert not launcher._wait_for_browser(process, server)


@pytest.mark.skipif(os.name != 'nt', reason='Win32 window ownership')
def test_native_window_probe_counts_minimized_windows_and_excludes_hidden_windows():
    import ctypes
    from ctypes import wintypes
    library = ctypes.WinDLL('user32', use_last_error=True)
    library.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    library.CreateWindowExW.restype = wintypes.HWND
    library.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    library.DestroyWindow.argtypes = [wintypes.HWND]
    # A tool window far off screen exercises Win32 without showing UI or a taskbar button.
    handle = library.CreateWindowExW(0x80, 'STATIC', 'OTA lifecycle test', 0x90000000,
                                   -32000, -32000, 1, 1, None, None, None, None)
    assert handle
    try:
        assert launcher._browser_has_window(os.getpid()) is True
        library.ShowWindow(handle, 6)  # SW_MINIMIZE is still an open window
        assert launcher._browser_has_window(os.getpid()) is True
        library.ShowWindow(handle, 0)  # SW_HIDE
        assert launcher._browser_has_window(os.getpid()) is False
    finally:
        library.DestroyWindow(handle)


@pytest.mark.parametrize('browser_kind', ['running', 'handoff', 'default'])
def test_confirmed_close_releases_app_lock_before_scheduling_update(server, monkeypatch, browser_kind):
    from google_prep.updates import windows
    events = []
    updater = SimpleNamespace(installation=None, stopped=False, pending=lambda:'pending.json',
                              start=lambda:events.append('start'))
    updater.stop = lambda:setattr(updater, 'stopped', True)
    monkeypatch.setattr('google_prep.gui.server.WINDOW_CLOSE_GRACE', 0)
    presence(server, 'closing-page', True)
    presence(server, 'closing-page', False)
    monkeypatch.setattr(launcher, '_make_updater', lambda:updater)
    monkeypatch.setattr(launcher, 'serve', lambda *args:server)
    monkeypatch.setattr(launcher, '_hold_app_mutex', lambda:'app-lock')
    monkeypatch.setattr(launcher, '_release', lambda _:events.append('released'))
    process = None if browser_kind == 'default' else SimpleNamespace(poll=lambda:0 if browser_kind == 'handoff' else None)
    monkeypatch.setattr(launcher, 'open_window', lambda _:process)
    monkeypatch.setattr(launcher.webbrowser, 'open', lambda _:None)
    monkeypatch.setattr(windows, 'launch_pending', lambda *args, **kwargs:events.append('install'))
    assert launcher.main([]) == 0
    assert events == ['start', 'released', 'install']
    assert updater.stopped


def test_browser_handoff_retains_the_app_lock_until_its_page_closes(server, monkeypatch):
    releases = []
    monkeypatch.setattr(launcher, '_make_updater', lambda:None)
    monkeypatch.setattr(launcher, 'serve', lambda *args:server)
    monkeypatch.setattr(launcher, '_hold_app_mutex', lambda:'app-lock')
    monkeypatch.setattr(launcher, '_release', lambda handle:releases.append(handle))
    monkeypatch.setattr(launcher, 'open_window', lambda _:SimpleNamespace(poll=lambda:0))
    def wait(*args, **kwargs):
        assert releases == [], 'An open handed-off page must keep installation blocked'
        return 'window_closed'
    monkeypatch.setattr(launcher, '_wait_for_page', wait)
    assert launcher.main([]) == 0
    assert releases == ['app-lock']

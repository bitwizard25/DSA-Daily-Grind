"""A non-blocking background worker; it stages updates but never installs them."""
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import cache, client as transport
from .locking import NamedMutex, UPDATE_MUTEX
from .manifest import APP_ID, version_key

CHECK_SECONDS = 6*3600
RETRY_SECONDS = (60, 300, 900, 3600)


def retry_delay(failures, rate_limit=None):
    delay = RETRY_SECONDS[min(max(failures-1, 0), 3)]
    return max(delay, rate_limit or 0)


class UpdateManager:
    def __init__(self, home: Path, build: dict, installation, client=None, lock_factory=NamedMutex):
        self.home, self.build, self.installation = home, build, installation
        self.folder = cache.root(home)
        self.client, self.lock_factory = client or transport, lock_factory
        self._lock = threading.RLock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread, self._busy, self._manual = None, False, False
        self._status = {'supported':installation is not None, 'enabled':cache.enabled(self.folder),
                        'version':build['version'], 'build_number':build['build_number'],
                        'state':'idle' if installation else 'unsupported', 'candidate':None, 'last_check':None,
                        'downloaded':0, 'total':0, 'last_result':None,
                        'message':'Automatic updates install after you close the app.' if installation else
                        'Automatic installation requires the installed Windows app. You can check for a manual download.'}
        try:
            self._status['last_result'] = cache.read(self.folder/'result.json')
        except (OSError, ValueError):
            pass
        self._restore()

    def _restore(self):
        try:
            candidate, _ = cache.pending(self.folder)
            if candidate.key <= version_key(self.build['version'], self.build['build_number']):
                if self.installation:
                    (self.folder/'pending.json').unlink(missing_ok=True)
                    self._status['message'] = 'This app is up to date.'
                return
            self._status.update(candidate=candidate.as_dict(), total=candidate.installer_size, state='ready',
                                downloaded=candidate.installer_size, message='Update downloaded and verified. It will install after the app closes.')
            result = self._status['last_result']
            if isinstance(result, dict) and result.get('tag') == candidate.tag and result.get('status') in ('failed','restart_required','installed'):
                self._status.update(state='failed', message=result.get('message','The update needs attention.'))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            self._status.update(state='failed', message='Pending update could not be verified. Check again for a fresh download.')

    def status(self):
        with self._lock:
            self._status['enabled'] = cache.enabled(self.folder)
            return {**self._status, 'candidate':dict(self._status['candidate']) if self._status['candidate'] else None}

    def _paint(self, **fields):
        with self._lock:
            self._status.update(fields)

    def start(self):
        self._ensure_thread()
        if self.installation and cache.enabled(self.folder) and self._status['state'] != 'ready':
            self._wake.set()

    def _ensure_thread(self):
        with self._lock:
            if self._thread and self._thread.is_alive(): return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name='prep-update', daemon=True)
            self._thread.start()

    def check(self):
        with self._lock:
            if self._busy or self._wake.is_set(): return self.status()
            self._manual = True
            self._paint(state='checking', message='Checking GitHub for a tested update…')
            self._ensure_thread()
            self._wake.set()
            return self.status()

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Automatic updates must be on or off.')
        cache.write(self.folder/'preferences.json', {'enabled':enabled})
        self._paint(enabled=enabled)
        if enabled:
            self.start()
        return self.status()

    def stop(self):
        self._stop.set();self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def pending(self):
        if not self.installation or not cache.enabled(self.folder) or self._status['state'] != 'ready':
            return None
        try:
            candidate, _ = cache.pending(self.folder)
            if candidate.key > version_key(self.build['version'], self.build['build_number']):
                return self.folder/'pending.json'
        except (OSError, ValueError, KeyError, TypeError):
            self._paint(state='failed', message='The downloaded update needs a fresh verification.')
        return None

    def _loop(self):
        delay, failures = None, 0
        while not self._stop.is_set():
            signalled = self._wake.wait(delay)
            self._wake.clear()
            if self._stop.is_set(): break
            with self._lock:
                manual, self._manual = self._manual, False
                if not manual and (not self.installation or not cache.enabled(self.folder)):
                    delay = None;continue
                self._busy = True
            try:
                with self.lock_factory(UPDATE_MUTEX) as acquired:
                    if not acquired:
                        self._paint(state='idle', message='Another app window is checking updates.')
                        delay = 60;continue
                    self._check_once()
                failures, delay = 0, CHECK_SECONDS
            except transport.RateLimited as exc:
                failures += 1;delay = retry_delay(failures, exc.retry_after_seconds)
                self._paint(state='offline', message='GitHub asked the app to wait. Update checks will retry later.')
            except OSError:
                failures += 1;delay = retry_delay(failures)
                self._paint(state='offline', message='Update check could not connect. Study normally; it will retry.')
            except (ValueError, KeyError, TypeError):
                failures += 1;delay = retry_delay(failures)
                self._paint(state='failed', message='Update could not be verified. Nothing was installed.')
            finally:
                with self._lock: self._busy = False
            if not self.installation or not cache.enabled(self.folder): delay = None

    def _check_once(self):
        self._paint(state='checking', last_check=datetime.now(timezone.utc).isoformat(), message='Checking for a tested update…')
        candidate = self.client.fetch_candidate(self.build)
        if self._stop.is_set(): return
        if not candidate:
            self._paint(state='idle', message='No newer tested update is available.');return
        self._paint(candidate=candidate.as_dict(), state='available', total=candidate.installer_size,
                    release_url=f'https://github.com/bitwizard25/DSA-Daily-Grind/releases/tag/{candidate.tag}',
                    message='A tested update is available for manual download.')
        if not self.installation or not cache.enabled(self.folder): return
        directory = cache.installer_path(self.folder, candidate).parent
        def progress(current, total):
            if self._stop.is_set(): raise InterruptedError('App is closing')
            self._paint(downloaded=current, total=total)
        self._paint(state='downloading', message='Downloading the update in the background…', downloaded=0)
        file = self.client.download_candidate(candidate, directory, progress=progress)
        cache.verify(file, candidate)
        if self._stop.is_set() or not cache.enabled(self.folder):
            self._paint(state='available', message='Download complete. Automatic installation is off.');return
        cache.write(self.folder/'pending.json', {'schema':1, 'app_id':APP_ID, 'candidate':candidate.as_dict()})
        cache.write(self.folder/'result.json', {'status':'pending','tag':candidate.tag})
        self._paint(state='ready', message='Update downloaded and verified. It will install after the app closes.',
                    downloaded=candidate.installer_size, last_result=None)

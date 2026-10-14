"""Named process mutexes on Windows; local locks for non-installing test platforms."""
import ctypes
import os
import threading

UPDATE_MUTEX = 'GoogleSWEIIOS.Update'
INSTALL_MUTEX = 'GoogleSWEIIOS.Installing'
APP_MUTEX = 'GoogleSWEIIOS.Running'
_locks = {}
_guard = threading.Lock()


def kernel():
    library = ctypes.WinDLL('kernel32', use_last_error=True)
    library.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    library.CreateMutexW.restype = ctypes.c_void_p
    library.OpenMutexW.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_wchar_p]
    library.OpenMutexW.restype = ctypes.c_void_p
    library.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    library.CloseHandle.argtypes = [ctypes.c_void_p]
    library.ReleaseMutex.argtypes = [ctypes.c_void_p]
    return library


class NamedMutex:
    def __init__(self, name):
        self.name, self.handle, self.acquired, self.local = name, None, False, None

    def __enter__(self):
        if os.name != 'nt':
            with _guard:
                self.local = _locks.setdefault(self.name, threading.Lock())
            self.acquired = self.local.acquire(blocking=False)
        else:
            library = kernel()
            self.handle = library.CreateMutexW(None, False, self.name)
            if not self.handle:
                raise OSError('Could not acquire the update lock')
            self.acquired = library.WaitForSingleObject(self.handle, 0) in (0, 128)
        return self.acquired

    def __exit__(self, *args):
        if self.local:
            if self.acquired: self.local.release()
        elif self.handle:
            library = kernel()
            if self.acquired: library.ReleaseMutex(self.handle)
            library.CloseHandle(self.handle)
        self.handle, self.acquired = None, False


def exists(name):
    if os.name != 'nt':
        with _guard:
            return bool(name in _locks and _locks[name].locked())
    library = kernel()
    handle = library.OpenMutexW(0x00100000, False, name)
    if handle:
        library.CloseHandle(handle)
        return True
    if ctypes.get_last_error() == 2:
        return False
    raise OSError('Could not inspect the running app lock')

"""Single-key input. Uses raw key presses in a real terminal; falls back to reading
whole lines when stdin is piped (tests, scripts), where an empty line means Enter."""
import os
import sys
import time
from contextlib import contextmanager

_NAMES = {"\r": "enter", "\n": "enter", " ": "space", "\x1b": "esc"}
_held = 0  # >0 while a screen keeps the terminal in cbreak mode


def interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _normalise(ch: str) -> str:
    if ch == "\x03":
        raise KeyboardInterrupt
    return _NAMES.get(ch, ch.lower())


def _read_line() -> str:
    line = sys.stdin.readline()
    if line == "":
        return "q"  # end of input: behave like quit
    text = line.rstrip("\r\n")
    if text == "":
        return "enter"
    if text.strip() == "":
        return "space"
    return text.strip()[0].lower()


@contextmanager
def held():
    """Keep the terminal in cbreak mode for a whole screen: keys typed while the screen
    redraws are queued instead of echoed or lost. Never flushes pending input."""
    global _held
    if os.name == "nt" or not interactive():
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if _held == 0 else None
    if _held == 0:
        tty.setcbreak(fd, termios.TCSANOW)
    _held += 1
    try:
        yield
    finally:
        _held -= 1
        if _held == 0:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float | None = None) -> str | None:
    """Return a key name ('enter', 'space', 'q', '1', ...) or None if the timeout passed."""
    if not interactive():
        return _read_line()
    if os.name == "nt":
        import msvcrt

        end = None if timeout is None else time.monotonic() + timeout
        while end is None or time.monotonic() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow / function keys: ignore
                    msvcrt.getwch()
                    continue
                return _normalise(ch)
            time.sleep(0.03)
        return None

    import select

    with held():
        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":  # swallow the rest of an escape sequence
            while select.select([fd], [], [], 0.01)[0]:
                os.read(fd, 8)
        return _normalise(ch)


def choose(valid: set[str]) -> str:
    """Block until one of the valid keys is pressed."""
    with held():
        while True:
            key = read_key()
            if key in valid:
                return key
            if key == "q" and "q" not in valid and not interactive():
                raise SystemExit(0)  # piped input ran out


def yes_no() -> bool:
    return choose({"y", "n"}) == "y"

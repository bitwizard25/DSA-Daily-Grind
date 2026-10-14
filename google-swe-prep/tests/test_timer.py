from google_prep.core.timer import Countdown, mmss


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_countdown_pause_and_phases():
    clock = Clock()
    timer = Countdown(45 * 60, clock)
    assert timer.mock_phase().startswith("Phase 1")
    clock.t = 10 * 60
    assert timer.mock_phase().startswith("Phase 2")
    timer.toggle_pause()
    clock.t = 30 * 60  # paused time does not count
    assert timer.elapsed() == 10 * 60 and timer.paused
    timer.toggle_pause()
    clock.t = 30 * 60 + 26 * 60
    assert timer.mock_phase().startswith("Phase 3")
    clock.t = 30 * 60 + 33 * 60
    assert timer.mock_phase().startswith("Phase 4")
    clock.t = 30 * 60 + 36 * 60
    assert timer.expired() and timer.remaining() == 0


def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(1458) == "24:18"

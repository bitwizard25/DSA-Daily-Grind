from datetime import date, datetime

import pytest

from google_prep.core import breaks, scheduler
from google_prep.gui.api import Api, ApiError


def test_break_freezes_calendar_and_preserves_fixed_interview(session):
    scheduler.set_setting(session, 'start_date', '2026-10-23')
    scheduler.set_setting(session, 'interview_date', '2026-11-24')
    breaks.schedule(session, date(2026,10,23), '2026-10-24', '2026-10-26')
    assert scheduler.day_number(session, date(2026,10,23)) == 1
    assert scheduler.day_number(session, date(2026,10,25)) == 1
    assert scheduler.day_number(session, date(2026,10,26)) == 2
    assert scheduler.interview_date(session) == date(2026,11,24)
    assert not breaks.state(session, date(2026,10,26))['active']


def test_resume_early_preserves_elapsed_break_days(session):
    scheduler.set_setting(session, 'start_date', '2026-10-23')
    breaks.schedule(session, date(2026,10,23), '2026-10-24', '2026-10-28')
    breaks.resume(session, date(2026,10,25))
    assert not breaks.state(session, date(2026,10,25))['active']
    assert scheduler.day_number(session, date(2026,10,25)) == 2
    assert scheduler.day_number(session, date(2026,10,26)) == 3


def test_cancel_future_break_does_not_shift_calendar(session):
    scheduler.set_setting(session, 'start_date', '2026-10-23')
    breaks.schedule(session, date(2026,10,23), '2026-10-24', '2026-10-26')
    breaks.resume(session, date(2026,10,23))
    assert scheduler.day_number(session, date(2026,10,26)) == 4


def test_break_is_neutral_for_streak_not_a_completed_day(session):
    scheduler.mark_done(session, date(2026,10,23), 'morning')
    breaks.schedule(session, date(2026,10,23), '2026-10-24', '2026-10-26')
    assert scheduler.streak(session, date(2026,10,25)) == 1
    assert scheduler.streak(session, date(2026,10,26)) == 1
    scheduler.mark_done(session, date(2026,10,26), 'morning')
    assert scheduler.streak(session, date(2026,10,26)) == 2
    assert scheduler.streak(session, date(2026,10,28)) == 0


@pytest.mark.parametrize('start,end', [('2026-10-22','2026-10-26'),('2026-10-24','2026-10-24'),('x','2026-10-26')])
def test_bad_ranges_do_not_change_state(session, start, end):
    with pytest.raises(ValueError): breaks.schedule(session, date(2026,10,23), start, end)
    assert breaks.state(session, date(2026,10,23))['scheduled'] is None


def test_break_api_survives_mode_change_and_keeps_saved_session(home):
    clock = {'now':datetime(2026,10,23,10)}
    api = Api(now=lambda:clock['now'])
    api.mock_start('G-1', practice=True)
    before = api.state()['session']
    api.break_schedule('2026-10-24', '2026-10-26')
    clock['now'] = datetime(2026,10,24,10)
    assert api.state()['study_break']['active']
    assert api.today()['kind'] == 'break'
    assert api.state()['session'] == before
    with pytest.raises(ApiError): api.break_schedule('bad', 'bad')
    assert api.break_resume()['active'] is False


def test_active_break_pauses_focus_and_applies_to_both_modes(home):
    api = Api(now=lambda:datetime(2026,10,23,10))
    api.preparation_mode('general')
    api.support_start('2026-10-23', 10, 'Trace one example')
    api.break_schedule('2026-10-23', '2026-10-26')
    assert api.support_state()['active']['status'] == 'paused'
    assert api.state()['study_break']['active']
    api.preparation_mode('google')
    assert api.state()['study_break']['active']


def test_scheduled_reminder_stays_quiet_but_manual_launch_is_not_blocked(session, monkeypatch):
    from google_prep.gui import launcher
    breaks.schedule(session, date(2026,10,12), '2026-10-12', '2026-10-14')
    monkeypatch.setattr(launcher, 'open_window', lambda _:pytest.fail('Reminder opened during break'))
    assert launcher.main(['--reminder']) == 0


def test_overlapping_break_is_rejected_without_changing_the_original(session):
    breaks.schedule(session, date(2026,10,23), '2026-10-24', '2026-10-26')
    with pytest.raises(ValueError): breaks.schedule(session, date(2026,10,23), '2026-10-25', '2026-10-28')
    assert breaks.state(session, date(2026,10,23))['scheduled']['resume'] == '2026-10-26'

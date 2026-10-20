"""Behavior support must preserve honest grades and recover across restarts."""
from datetime import datetime, timedelta

import pytest
from sqlmodel import select

from google_prep.core import support
from google_prep.models import Attempt, Setting

NOW = datetime(2026, 10, 12, 10)
DAY = NOW.date().isoformat()


def test_fresh_support_has_next_step_without_fake_evidence(session):
    state = support.state(session, NOW)
    assert state['today']['minimum_done'] is False
    assert state['week']['active_days'] == 0
    assert state['confidence']['independent_solves'] == 0
    assert state['confidence']['notes'] == []
    assert state['active'] is None


def test_start_survives_restart_and_cannot_create_overlapping_blocks(session):
    block = support.start(session, NOW, DAY, 5, 'Read one example')
    assert support.state(session, NOW)['active']['id'] == block['id']
    with pytest.raises(ValueError, match='already'):
        support.start(session, NOW, DAY, 25, 'Another task')
    assert support.state(session, NOW + timedelta(minutes=2))['today']['minimum_done'] is False


def test_pause_excludes_paused_time_and_completion_is_idempotent(session):
    block = support.start(session, NOW, DAY, 5, 'One example')
    support.transition(session, NOW + timedelta(minutes=2), block['id'], 'pause')
    support.transition(session, NOW + timedelta(minutes=20), block['id'], 'resume')
    support.transition(session, NOW + timedelta(minutes=23), block['id'], 'complete', 'I can trace a set.')
    support.transition(session, NOW + timedelta(minutes=24), block['id'], 'complete', 'I can trace a set.')
    state = support.state(session, NOW + timedelta(minutes=24))
    assert state['today']['minutes'] == 5
    assert state['today']['minimum_done'] is True
    assert state['week']['completed_blocks'] == 1
    assert state['confidence']['notes'][0]['note'] == 'I can trace a set.'
    assert state['confidence']['independent_solves'] == 0
    assert session.exec(select(Attempt)).all() == []


def test_short_and_parked_blocks_do_not_claim_minimum_or_recovery(session):
    block = support.start(session, NOW, DAY, 5, 'Small step')
    support.transition(session, NOW + timedelta(minutes=1), block['id'], 'park', 'Trace index 1 next.')
    result = support.state(session, NOW + timedelta(minutes=2))
    assert result['today']['minimum_done'] is False
    assert result['week']['completed_blocks'] == 0
    assert result['parked'][0]['note'] == 'Trace index 1 next.'
    assert result['active'] is None


def test_return_after_gap_counts_once_only_after_completed_practice(session):
    block = support.start(session, NOW, DAY, 5, 'Start')
    support.transition(session, NOW + timedelta(minutes=5), block['id'], 'complete')
    later = NOW + timedelta(days=3)
    assert support.state(session, later)['today']['returning'] is True
    block = support.start(session, later, later.date().isoformat(), 5, 'Return')
    assert support.state(session, later)['week']['return_days'] == 0
    support.transition(session, later + timedelta(minutes=5), block['id'], 'complete')
    assert support.state(session, later + timedelta(minutes=5))['week']['return_days'] == 1


def test_rescue_only_claims_helpful_when_user_acknowledges_it(session):
    support.rescue(session, NOW, DAY, 'worry', 'opened', 'one')
    assert support.state(session, NOW)['week']['helpful_rescues'] == 0
    support.rescue(session, NOW, DAY, 'worry', 'helpful', 'one')
    support.rescue(session, NOW, DAY, 'worry', 'helpful', 'one')
    assert support.state(session, NOW)['week']['helpful_rescues'] == 1
    with pytest.raises(ValueError):
        support.rescue(session, NOW, DAY, 'worry', 'helpful', 'never-opened')


def test_external_practice_is_self_report_and_does_not_create_solve(session):
    support.outside(session, NOW, DAY, 10, 'Practised elsewhere', 'outside-one')
    support.outside(session, NOW, DAY, 10, 'Practised elsewhere', 'outside-one')
    state = support.state(session, NOW)
    assert state['today']['minimum_done'] is True
    assert state['today']['outside_minutes'] == 10
    assert state['confidence']['independent_solves'] == 0
    assert session.exec(select(Attempt)).all() == []


@pytest.mark.parametrize('minutes,task', [(True,'Task'),(0,'Task'),(121,'Task'),(5,''),(5,'x'*241)])
def test_invalid_start_writes_nothing(session, minutes, task):
    with pytest.raises(ValueError):
        support.start(session, NOW, DAY, minutes, task)
    assert support.state(session, NOW)['active'] is None


def test_stale_day_rejected_and_future_clock_does_not_rewrite_history(session):
    with pytest.raises(ValueError, match='day'):
        support.start(session, NOW, '2026-10-11', 5, 'Stale')
    block = support.start(session, NOW, DAY, 5, 'Old block')
    next_day = NOW + timedelta(days=1)
    support.transition(session, next_day, block['id'], 'park', 'Return here.')
    assert support.state(session, next_day)['today']['minimum_done'] is False
    assert support.state(session, next_day)['active'] is None


def test_preferences_validate_atomically_and_preserve_existing_goal(session):
    session.add(Setting(key='daily_problems', value='3')); session.commit()
    assert support.preferences(session, {'focus_minutes':15,'cue':'After breakfast'})['cue'] == 'After breakfast'
    with pytest.raises(ValueError):
        support.preferences(session, {'focus_minutes':45,'minimum_minutes':999})
    assert support.preferences(session)['focus_minutes'] == 15
    assert session.get(Setting,'daily_problems').value == '3'


def test_only_actual_solo_grades_feed_confidence_evidence(session):
    for problem, solved, help_ in [('G-1',True,'none'),('G-217',True,'hint'),('G-242',False,'none')]:
        session.add(Attempt(problem_id=problem,day=DAY,minutes=15,solved=solved,help=help_))
    session.commit()
    assert support.state(session,NOW)['confidence']['independent_solves'] == 1


def test_support_api_roundtrip_and_bad_inputs_do_not_write():
    from google_prep.gui.api import Api, ApiError, call
    api = Api(now=lambda:NOW)
    assert call(api,'support_state',{})['active'] is None
    for payload in ({'date':DAY,'minutes':0,'task':'Read an example'},
                    {'date':'2026-10-11','minutes':5,'task':'Read an example'}):
        with pytest.raises(ApiError):
            call(api,'support_start',payload)
    first=api.support_start(DAY,5,'Read an example')
    assert first['active']['task']=='Read an example'
    block_id=first['active']['id']
    assert Api(now=lambda:NOW).support_state()['active']['id']==block_id
    assert api.support_transition(block_id,'pause')['active']['status']=='paused'
    assert api.support_preferences({'cue':'After lunch'})['preferences']['cue']=='After lunch'
    with pytest.raises(ApiError):
        api.support_preferences({'focus_minutes':False})


def test_a_completed_short_start_records_effort_without_claiming_full_goal(session):
    block=support.start(session,NOW,DAY,2,'One example')
    support.transition(session,NOW+timedelta(minutes=2),block['id'],'complete')
    result=support.state(session,NOW+timedelta(minutes=2))
    assert result['week']['completed_blocks']==1
    assert result['today']['minimum_done'] is False


def test_changing_company_or_core_focus_keeps_confidence_and_practice(session):
    block=support.start(session,NOW,DAY,5,'Explain a trade-off')
    support.transition(session,NOW+timedelta(minutes=5),block['id'],'complete','I can explain a cache trade-off.')
    before=support.state(session,NOW+timedelta(minutes=5))
    session.merge(Setting(key='target',value='Backend engineer at another company'));session.commit()
    support.preferences(session,{'core_focus':'system_design'})
    after=support.state(session,NOW+timedelta(minutes=5))
    assert after['preferences']['core_focus']=='system_design'
    assert after['confidence']==before['confidence']
    assert after['week']==before['week']


def test_modes_preserve_google_settings_saved_session_and_general_evidence():
    from google_prep.gui.api import Api, ApiError
    from google_prep.core import session_store
    from google_prep.db import get_session
    api=Api(now=lambda:NOW)
    assert api.state()['settings']['preparation_mode']=='google'
    api.settings_save(target='Google SWE II',daily_problems='3',interview_date='2026-11-30')
    saved={'kind':'mock','problem_id':'G-1','minutes':45,'practice':False,'elapsed':123,
           'ticks':[False]*4,'shown':-1,'started':NOW.isoformat(),'submitted':False}
    session_store.save(saved)
    api.support_start(DAY,5,'Explain one edge case')
    api.preparation_mode('general')
    api.support_preferences({'general_role':'Backend engineer','core_focus':'system_design'})
    state=api.preparation_mode('google')
    assert state['settings']['target']=='Google SWE II'
    assert state['settings']['daily_problems']==3
    assert state['settings']['interview_date']=='2026-11-30'
    assert session_store.load()==saved
    assert api.support_state()['preferences']['general_role']=='Backend engineer'
    assert api.support_state()['active']['status']=='paused'
    with pytest.raises(ApiError):
        api.preparation_mode('unknown')


def test_instant_finish_is_not_a_completed_focus_block_or_minimum(session):
    block=support.start(session,NOW,DAY,5,'Read an example')
    support.transition(session,NOW+timedelta(seconds=2),block['id'],'complete')
    result=support.state(session,NOW+timedelta(seconds=2))
    assert result['week']['completed_blocks']==0
    assert result['today']['completed_blocks']==0
    assert result['today']['minimum_done'] is False


def test_paused_note_survives_restart_without_becoming_completed_evidence(session):
    block=support.start(session,NOW,DAY,5,'Explain one example')
    support.transition(session,NOW+timedelta(minutes=1),block['id'],'pause','Trace the second value next.')
    state=support.state(session,NOW+timedelta(minutes=10))
    assert state['active']['note']=='Trace the second value next.'
    assert state['confidence']['notes']==[]

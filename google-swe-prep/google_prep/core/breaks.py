"""Planned rest days shared by both modes; resume dates are exclusive."""
import json
from datetime import date, timedelta

from google_prep.models import Setting

KEY = 'planned_study_breaks'


def intervals(session):
    row = session.get(Setting, KEY)
    return json.loads(row.value) if row else []


def _save(session, rows):
    session.merge(Setting(key=KEY, value=json.dumps(rows)))
    session.commit()


def is_break(session, day):
    return any(row['start'] <= day.isoformat() < row['resume'] for row in intervals(session))


def days_through(session, first, last):
    return sum(max(0, (min(date.fromisoformat(row['resume']), last + timedelta(days=1))
                       - max(date.fromisoformat(row['start']), first)).days) for row in intervals(session))


def state(session, today):
    upcoming = next((row for row in intervals(session) if row['resume'] > today.isoformat()), None)
    return {'active':bool(upcoming and upcoming['start'] <= today.isoformat()),
            'scheduled':upcoming, 'reminders_paused':is_break(session, today)}


def schedule(session, today, start, resume_on):
    first, end = date.fromisoformat(start), date.fromisoformat(resume_on)
    if first < today or not 1 <= (end - first).days <= 365:
        raise ValueError('Choose a start today or later, and a resume date 1–365 days after it.')
    rows = intervals(session)
    if any(row['resume'] > today.isoformat() for row in rows):
        raise ValueError('Cancel or end your existing break before planning another.')
    rows.append({'start':first.isoformat(), 'resume':end.isoformat()})
    _save(session, rows)
    return state(session, today)


def resume(session, today):
    rows = []
    for row in intervals(session):
        if row['resume'] <= today.isoformat(): rows.append(row)
        elif row['start'] < today.isoformat(): rows.append({**row, 'resume':today.isoformat()})
    _save(session, rows)
    return state(session, today)

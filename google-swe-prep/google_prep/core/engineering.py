"""Bounded engineering exercises and independent retries, separate from readiness."""
import json
from datetime import date, timedelta
from uuid import uuid4

from sqlalchemy import update
from sqlmodel import select

from google_prep import db
from google_prep.models import Attempt, ComprehensionAttempt, Problem

HELP = {'none': 0, 'hint': 1, 'ai': 2, 'answer': 3}
NOTE_FIELDS = ('read', 'debug', 'change', 'verify', 'next')


def catalog():
    return db.load_json('engineering_practice.json')['exercises']


def exercise(exercise_id, language):
    if not isinstance(exercise_id, str) or language not in ('python', 'cpp'):
        raise ValueError('Choose a known exercise and Python or C++.')
    item = next((x for x in catalog() if x['id'] == exercise_id), None)
    if item is None:
        raise ValueError('Choose a known exercise.')
    return item


def packed(row):
    return {**row.model_dump(), 'notes': json.loads(row.notes)}


def open_attempt(session, now, exercise_id, language, restart=False):
    item = exercise(exercise_id, language)
    if type(restart) is not bool:
        raise ValueError('Restart must be true or false.')
    slot = f'{exercise_id}:{language}'
    row = session.exec(select(ComprehensionAttempt).where(ComprehensionAttempt.active_slot == slot)).first()
    if row is None and not restart:
        row = session.exec(select(ComprehensionAttempt).where(
            ComprehensionAttempt.exercise_id == exercise_id, ComprehensionAttempt.language == language)
            .order_by(ComprehensionAttempt.updated_at.desc())).first()
    if row is None:
        row = ComprehensionAttempt(id=uuid4().hex, exercise_id=exercise_id, language=language,
                                   active_slot=slot, code=item['languages'][language]['starter'],
                                   created_at=now, updated_at=now)
        session.add(row)
        session.commit()
    public = {key: value for key, value in item.items() if key != 'languages'}
    return {'exercise': public, 'attempt': packed(row)}


def editable(session, attempt_id, revision):
    if not isinstance(attempt_id, str) or type(revision) is not int:
        raise ValueError('Choose a saved attempt and its revision.')
    row = session.get(ComprehensionAttempt, attempt_id)
    if row is None:
        raise ValueError('This practice attempt was not found.')
    if row.status != 'draft':
        raise ValueError('This attempt is complete. Start a fresh attempt to practise again.')
    if row.revision != revision:
        raise ValueError('This draft changed in another window. Download your work, then reload before saving.')
    return row


def save(session, now, attempt_id, revision, notes, code, help, complete=False):
    row = editable(session, attempt_id, revision)
    if type(complete) is not bool or not isinstance(help, str) or help not in HELP:
        raise ValueError('Choose how much help you used and a valid completion status.')
    if not isinstance(code, str) or len(code) > 32000:
        raise ValueError('Keep code to 32,000 characters or fewer.')
    if (not isinstance(notes, dict) or set(notes) - set(NOTE_FIELDS)
            or any(not isinstance(v, str) or len(v) > 3000 for v in notes.values())):
        raise ValueError('Keep each practice note to 3,000 characters or fewer.')
    if complete and (not code.strip() or any(not notes.get(key, '').strip() for key in NOTE_FIELDS[:4])):
        raise ValueError('Add your explanation, bug, change and test evidence before finishing.')
    values = dict(code=code, notes=json.dumps(notes), help=max((row.help, help), key=HELP.__getitem__),
                  revision=revision + 1, updated_at=now)
    if complete:
        values.update(status='completed', active_slot=None)
    result = session.exec(update(ComprehensionAttempt).where(
        ComprehensionAttempt.id == attempt_id, ComprehensionAttempt.revision == revision,
        ComprehensionAttempt.status == 'draft').values(**values))
    if result.rowcount != 1:
        session.rollback()
        raise ValueError('This draft changed in another window. Download your work, then reload before saving.')
    session.commit()
    session.refresh(row)
    return packed(row)


def reference(session, now, attempt_id, revision):
    row = editable(session, attempt_id, revision)
    saved = save(session, now, attempt_id, revision, json.loads(row.notes), row.code, 'answer')
    item = exercise(row.exercise_id, row.language)
    return {'attempt': saved, 'reference': item['languages'][row.language]['reference']}


def overview(session, now):
    rows = session.exec(select(Attempt).where(Attempt.day <= now.date().isoformat()).order_by(Attempt.id)).all()
    grouped = {}
    for row in rows:
        grouped.setdefault(row.problem_id, []).append(row)
    retries, upcoming, recovered = [], 0, 0
    for problem_id, history in grouped.items():
        last = history[-1]
        if last.solved and last.help == 'none':
            recovered += int(any((not x.solved or x.help != 'none') and x.day < last.day for x in history[:-1]))
            continue
        problem = session.get(Problem, problem_id)
        if problem is None:
            continue
        due = date.fromisoformat(last.day) + timedelta(days=1)
        if due > now.date():
            upcoming += 1
            continue
        reason = ('Not solved last time' if not last.solved else
                  'Help was not recorded' if last.help is None else 'Previously solved with help')
        retries.append({'id': problem.id, 'title': problem.title, 'due': due.isoformat(), 'reason': reason})
    practice = session.exec(select(ComprehensionAttempt).order_by(ComprehensionAttempt.updated_at.desc())).all()
    return {'retries': sorted(retries, key=lambda x: (x['due'], x['id'])), 'upcoming_retries': upcoming,
            'recovered': recovered, 'completed': sum(x.status == 'completed' for x in practice),
            'exercises': [{key: item[key] for key in ('id', 'title', 'level', 'minutes')} for item in catalog()],
            'drafts': [packed(x) for x in practice if x.status == 'draft'],
            'recent': [packed(x) for x in practice if x.status == 'completed'][:6]}

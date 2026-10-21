"""Small starts, practical recovery and evidence the user can trust.

Focus timers and learning notes are self-reported study evidence, not proof of a
correct answer. They deliberately never write Attempt or readiness records.
"""
import json
from datetime import date, datetime, timedelta
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from google_prep.models import Attempt, Setting, SupportBlock, SupportRescue

DEFAULTS = {'minimum_minutes': 5, 'focus_minutes': 25,
            'cue': 'After my first break, I will open one example.',
            'reading_size': 'standard', 'rescue_prompts': True, 'core_focus': 'problem_solving',
            'general_role': 'Software engineer'}
BLOCKERS = {
    'worry': ('I feel worried or overwhelmed', 'Pause briefly. Name only the input and expected output. Try that for two minutes.'),
    'approach': ('I cannot find an approach', 'Use one tiny example. Describe the simplest brute-force approach before improving it.'),
    'syntax': ('I am stuck on Python syntax', 'Write the operation you need in plain words. Check its syntax in Concept Vault, then try one line.'),
    'logic': ('I am stuck on index or pointer logic', 'Trace just one iteration. Write the index, its value, and what should change next.'),
    'bug': ('A bug keeps stopping me', 'Write expected versus actual for the smallest failing example. Check one changed value at a time.'),
    'polishing': ('I keep doing the same task', 'Write the next unfinished step. Save it, then choose to continue deliberately or switch.'),
    'give_up': ('I want to give up', 'Choose one smaller action: explain an example or write brute force. Try two minutes, then choose whether to continue, park, or stop.'),
}


def _day(value, when):
    if not isinstance(value, str) or value != when.date().isoformat():
        raise ValueError('The day has changed. Refresh before recording today’s practice.')


def _text(value, label, limit=240, empty=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not empty and not value.strip()):
        raise ValueError(f'{label} must be {1 if not empty else 0}–{limit} characters.')
    return value.strip()


def _minutes(value):
    if type(value) is not int or not 1 <= value <= 120:
        raise ValueError('Choose a whole number of minutes from 1 to 120.')
    return value


def preferences(session: Session, changes: dict | None = None, *, persist: bool = True) -> dict:
    row = session.get(Setting, 'support_preferences')
    try:
        stored = json.loads(row.value) if row else {}
        if not isinstance(stored, dict):
            stored = {}
    except (TypeError, ValueError):
        stored = {}
    result = {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}
    if changes is not None:
        if not isinstance(changes, dict) or set(changes) - set(DEFAULTS):
            raise ValueError('Choose known study support settings.')
        result.update(changes)
    if type(result['minimum_minutes']) is not int or result['minimum_minutes'] not in (2, 5, 10):
        raise ValueError('Daily minimum must be 2, 5 or 10 minutes.')
    if type(result['focus_minutes']) is not int or result['focus_minutes'] not in (15, 25, 45):
        raise ValueError('Focus blocks must be 15, 25 or 45 minutes.')
    result['cue'] = _text(result['cue'], 'Starting commitment', empty=True)
    result['general_role'] = _text(result['general_role'], 'General role', empty=True)
    if result['core_focus'] not in ('problem_solving', 'system_design', 'ai_design', 'communication'):
        raise ValueError('Choose a known core skill.')
    if result['reading_size'] not in ('standard', 'comfortable') or type(result['rescue_prompts']) is not bool:
        raise ValueError('Choose a known reading size and turn rescue prompts on or off.')
    if changes is not None and persist:
        session.merge(Setting(key='support_preferences', value=json.dumps(result)))
        session.commit()
    return result


def elapsed(block: SupportBlock, when: datetime) -> float:
    # Pausing, clock changes and a block accidentally left overnight cannot create
    # an unbounded claim. Completion remains an explicit acknowledgement.
    extra = max(0, (when - block.running_since).total_seconds()) if block.running_since else 0
    return min(7200, max(0, block.elapsed_seconds) + extra)


def _block(block, when):
    return {'id': block.id, 'day': block.day, 'task': block.task,
            'target_minutes': block.target_minutes, 'status': block.status,
            'elapsed_seconds': int(elapsed(block, when)), 'note': block.note,
            'source': block.source, 'updated_at': block.updated_at.isoformat()}


def start(session: Session, when: datetime, day: str, minutes: int, task: str) -> dict:
    _day(day, when)
    minutes, task = _minutes(minutes), _text(task, 'Next action')
    active = session.exec(select(SupportBlock).where(SupportBlock.active_slot == 'focus')).first()
    if active:
        raise ValueError('A focus block is already open. Resume or park it first.')
    block = SupportBlock(id=uuid4().hex, day=day, task=task, target_minutes=minutes,
                         running_since=when, created_at=when, updated_at=when)
    session.add(block)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ValueError('A focus block is already open in another window. Refresh to resume it.') from exc
    return _block(block, when)


def transition(session: Session, when: datetime, block_id: str, action: str, note: str = '') -> dict:
    note = _text(note, 'Learning or next-step note', empty=True)
    if action not in ('pause', 'resume', 'complete', 'park', 'extend'):
        raise ValueError('Choose pause, resume, finish, park or extend.')
    if not isinstance(block_id, str):
        raise ValueError('Choose a saved focus block.')
    block = session.get(SupportBlock, block_id)
    if not block:
        raise ValueError('This focus block was not found. Refresh your study support.')
    if block.status in ('completed', 'parked'):
        if action == 'resume' and block.status == 'parked':
            _day(block.day, when)
            if session.exec(select(SupportBlock).where(SupportBlock.active_slot == 'focus')).first():
                raise ValueError('Another focus block is already open.')
            block.status, block.active_slot, block.running_since = 'active', 'focus', when
        elif action in ('complete', 'park'):
            expected = 'completed' if action == 'complete' else 'parked'
            if block.status != expected or block.note != note:
                raise ValueError('This block changed in another window. Your new note has not been saved. Keep a copy and refresh.')
            return _block(block, when)
        else:
            raise ValueError('This block is already closed. Start a new small step.')
    else:
        block.note = note
        if action in ('resume', 'extend'):
            _day(block.day, when)
        block.elapsed_seconds = elapsed(block, when)
        block.running_since = None
        if action == 'pause':
            block.status = 'paused'
        elif action == 'resume':
            block.status, block.running_since = 'active', when
        elif action == 'extend':
            block.target_minutes = min(120, max(block.target_minutes, int(block.elapsed_seconds // 60)) + 10)
            block.status, block.running_since = 'active', when
        else:
            block.status = 'completed' if action == 'complete' else 'parked'
            block.active_slot = None
            # An overnight block requires a fresh start; don't credit the whole
            # night to yesterday. Retain its note and limit acknowledged time.
            if block.day != when.date().isoformat():
                block.elapsed_seconds = min(block.elapsed_seconds, block.target_minutes * 60)
            block.note = note
    block.updated_at = when
    session.add(block)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ValueError('Another window changed your focus block. Refresh to continue.') from exc
    return _block(block, when)


def outside(session: Session, when: datetime, day: str, minutes: int, task: str, request_id: str) -> None:
    _day(day, when)
    minutes, task = _minutes(minutes), _text(task, 'Practice description')
    request_id = _text(request_id, 'Practice ID', 80)
    key = 'outside:' + request_id
    if session.get(SupportBlock, key):
        return
    session.add(SupportBlock(id=key, day=day, task=task, target_minutes=minutes,
                            elapsed_seconds=minutes*60, status='completed', active_slot=None,
                            source='outside', created_at=when, updated_at=when))
    session.commit()


def rescue(session: Session, when: datetime, day: str, blocker: str, outcome: str, request_id: str) -> None:
    _day(day, when)
    request_id = _text(request_id, 'Rescue ID', 80)
    if not isinstance(blocker, str) or blocker not in BLOCKERS or outcome not in ('opened', 'helpful'):
        raise ValueError('Choose a known obstacle and rescue outcome.')
    row = session.get(SupportRescue, request_id)
    if row and (row.day != day or row.blocker != blocker):
        raise ValueError('This rescue belongs to another day or obstacle. Start a fresh rescue.')
    if outcome == 'helpful' and not row:
        raise ValueError('Open a rescue step before recording whether it helped.')
    row = row or SupportRescue(id=request_id, day=day, blocker=blocker, created_at=when)
    if outcome == 'helpful':
        row.helpful = True
    session.add(row)
    session.commit()


def draft(session: Session, slot: str, value: dict | None) -> None:
    if not isinstance(slot, str) or slot not in ('task', 'note', 'routine', 'outside'):
        raise ValueError('Choose a known study draft.')
    key = 'support_draft:' + slot
    if value is None:
        row = session.get(Setting, key)
        if row:
            session.delete(row)
        session.commit()
        return
    if not isinstance(value, dict):
        raise ValueError('A study draft must contain named fields.')
    if slot == 'routine':
        value = preferences(session, value, persist=False)
    else:
        allowed = {'text', 'block_id'} if slot == 'note' else {'text', 'minutes'} if slot == 'outside' else {'text'}
        if slot == 'outside' and 'request_id' in value:
            allowed.add('request_id')
        if set(value) != allowed:
            raise ValueError('Choose known fields for this study draft.')
        value = {k: _text(v, 'Draft ' + k, 240 if k == 'text' else 80, empty=True) for k, v in value.items()}
    session.merge(Setting(key=key, value=json.dumps(value)))
    session.commit()


def drafts(session: Session) -> dict:
    result = {}
    for slot in ('task', 'note', 'routine', 'outside'):
        row = session.get(Setting, 'support_draft:' + slot)
        if row:
            try:
                value = json.loads(row.value)
                if isinstance(value, dict):
                    result[slot] = value
            except ValueError:
                pass
    return result


def state(session: Session, when: datetime) -> dict:
    prefs = preferences(session)
    day = when.date()
    first = day - timedelta(days=6)
    rows = list(session.exec(select(SupportBlock).where(SupportBlock.day <= day.isoformat())).all())
    attempts = list(session.exec(select(Attempt).where(Attempt.day <= day.isoformat())).all())
    complete = [b for b in rows if b.status == 'completed']
    week = [b for b in complete if b.day >= first.isoformat()]
    today = [b for b in complete if b.day == day.isoformat()]
    today_seconds = sum(b.elapsed_seconds for b in today)
    active = session.exec(select(SupportBlock).where(SupportBlock.active_slot == 'focus')).first()
    rescues = list(session.exec(select(SupportRescue).where(
        SupportRescue.day >= first.isoformat(), SupportRescue.day <= day.isoformat())).all())
    activity = sorted({b.day for b in complete if b.elapsed_seconds >= 60} | {a.day for a in attempts})
    previous = [d for d in activity if d < day.isoformat()]
    return_days = {d for i, d in enumerate(activity) if i and
                   (date.fromisoformat(d) - date.fromisoformat(activity[i-1])).days > 1}
    notes = sorted([b for b in week if b.note], key=lambda b: b.updated_at, reverse=True)[:5]
    solos = {a.problem_id for a in attempts if a.solved and a.help == 'none' and a.day >= first.isoformat()}
    return {'date': day.isoformat(), 'now': when.isoformat(), 'preferences': prefs, 'drafts': drafts(session),
            'active': _block(active, when) if active else None,
            'parked': [_block(b, when) for b in sorted(rows, key=lambda b: b.updated_at, reverse=True)
                       if b.status == 'parked'][:5],
            'today': {'minimum_done': today_seconds >= prefs['minimum_minutes'] * 60,
                      'minutes': int(today_seconds // 60), 'completed_blocks': sum(b.elapsed_seconds >= 60 for b in today),
                      'outside_minutes': int(sum(b.elapsed_seconds for b in today if b.source == 'outside') // 60),
                      'returning': bool(previous and (day - date.fromisoformat(previous[-1])).days > 1 and day.isoformat() not in activity)},
            'week': {'active_days': sum(d >= first.isoformat() for d in activity),
                     'completed_blocks': len([b for b in week if b.source == 'timer' and b.elapsed_seconds >= 60]),
                     'return_days': sum(d >= first.isoformat() for d in return_days),
                     'helpful_rescues': sum(r.helpful for r in rescues),
                     'days': [{'date': (first + timedelta(days=i)).isoformat(),
                               'active': (first + timedelta(days=i)).isoformat() in activity} for i in range(7)]},
            'confidence': {'independent_solves': len(solos),
                           'notes': [{'day': b.day, 'note': b.note, 'source': 'self-report'} for b in notes]},
            'blockers': [{'id': key, 'label': value[0], 'step': value[1]} for key, value in BLOCKERS.items()]}

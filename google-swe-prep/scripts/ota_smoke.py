"""Seed and verify a disposable profile for the Windows OTA CI smoke test."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google_prep import db
from google_prep.core import scheduler, session_store
from google_prep.models import Attempt, StarStory, TrainingCheckin, Problem
from sqlmodel import select

def seed(home: Path):
    home = Path(home); home.mkdir(parents=True, exist_ok=True)
    import os; os.environ['GOOGLE_PREP_HOME'] = str(home)
    with db.get_session() as s:
        problem = s.exec(select(Problem)).first()
        if problem is None: raise AssertionError('problem bank is empty')
        attempt = Attempt(problem_id=problem.id, day='2026-10-14', minutes=28, solved=True, dry_run_ok=True, help='none')
        story = StarStory(dimension='Ownership', title='OTA ownership evidence', situation='A release needed safe updates.', task='Preserve study data.', action='Added verified staged updates.', result='The profile remains intact.')
        checkin = TrainingCheckin(date_str='2026-10-14', habit_id='clarity', intention='Explain one tradeoff.', reflection='Check one edge case.')
        s.add(attempt); s.add(story); s.add(checkin)
        scheduler.set_setting(s, 'design:lld:learned', '["lld-responsibilities"]')
        s.commit()
        evidence = {'attempt_id': attempt.id, 'star_id': story.id, 'day': checkin.date_str}
    (home/'ota-evidence.json').write_text(json.dumps(evidence), encoding='utf-8')
    session_store.save({'kind':'mock','problem_id':problem.id,'minutes':45,'elapsed':120,'ticks':[],'shown':0})

def verify_records(home: Path):
    import os; os.environ['GOOGLE_PREP_HOME'] = str(home)
    evidence=json.loads((Path(home)/'ota-evidence.json').read_text())
    with db.get_session() as s:
        assert s.get(Attempt,evidence['attempt_id']).solved
        assert s.get(StarStory,evidence['star_id']).title == 'OTA ownership evidence'
        assert s.get(TrainingCheckin,evidence['day']).reflection == 'Check one edge case.'
        assert scheduler.get_setting(s,'design:lld:learned') == '["lld-responsibilities"]'
    assert session_store.load()['problem_id']

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('seed','verify'):
        q=sub.add_parser(name); q.add_argument('--home',type=Path,required=True)
    a=p.parse_args(); (seed if a.cmd=='seed' else verify_records)(a.home)
if __name__=='__main__': main()

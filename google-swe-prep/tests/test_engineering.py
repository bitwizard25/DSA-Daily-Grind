from datetime import datetime, timedelta

import pytest
from sqlmodel import select

from google_prep import db
from google_prep.core import engineering
from google_prep.gui.api import Api, ApiError
from google_prep.models import Attempt, ComprehensionAttempt

NOW = datetime(2026, 10, 12, 10)
NOTES = {'read': 'Events enter, IDs are stored, results leave.', 'debug': 'The same ID is returned twice.',
         'change': 'Remember IDs after accepting them.', 'verify': 'Try empty input and repeated IDs.', 'next': ''}


def test_drafts_are_separate_by_language_and_resume_after_restart():
    api = Api()
    opened = api.engineering_open('events', 'python')
    saved = api.engineering_save(opened['attempt']['id'], 0, NOTES, 'my code', 'none')
    assert saved['revision'] == 1
    assert Api().engineering_open('events', 'python')['attempt']['code'] == 'my code'
    assert api.engineering_open('events', 'cpp')['attempt']['code'] != 'my code'
    assert saved['notes'] == NOTES


def test_stale_save_never_overwrites_newer_draft():
    api = Api()
    row = api.engineering_open('events', 'python')['attempt']
    api.engineering_save(row['id'], 0, NOTES, 'newer', 'none')
    with pytest.raises(ApiError, match='another window'):
        api.engineering_save(row['id'], 0, NOTES, 'stale', 'none')
    assert api.engineering_open('events', 'python')['attempt']['code'] == 'newer'


def test_reference_help_is_sticky_and_completion_is_self_reported_only():
    api = Api()
    row = api.engineering_open('events', 'python')['attempt']
    revealed = api.engineering_reference(row['id'], 0)
    assert revealed['attempt']['help'] == 'answer' and revealed['reference']['explanation']
    saved = api.engineering_save(row['id'], 1, NOTES, 'fixed', 'none', complete=True)
    assert saved['help'] == 'answer' and saved['status'] == 'completed'
    with db.get_session() as s:
        assert not s.exec(select(Attempt)).all()
        assert len(s.exec(select(ComprehensionAttempt)).all()) == 1
    with pytest.raises(ApiError):
        api.engineering_save(row['id'], saved['revision'], NOTES, 'changed', 'none', complete=True)
    assert api.engineering_open('events', 'python')['attempt']['status'] == 'completed'
    assert api.engineering_open('events', 'python', restart=True)['attempt']['id'] != row['id']


@pytest.mark.parametrize('changes', [{'notes': {}}, {'help': 'unknown'}, {'code': 4}, {'revision': True},
                                    {'notes': {**NOTES, 'read': 'x'*3001}}, {'complete': 'yes'}])
def test_invalid_save_does_not_change_draft(changes):
    api = Api()
    row = api.engineering_open('events', 'python')['attempt']
    payload = dict(attempt_id=row['id'], revision=0, notes=NOTES, code='valid', help='none', complete=True)
    payload.update(changes)
    with pytest.raises(ApiError):
        api.engineering_save(**payload)
    assert api.engineering_open('events', 'python')['attempt']['revision'] == 0


def test_retry_queue_waits_until_tomorrow_and_clears_after_independent_solve(session):
    session.add(Attempt(problem_id='G-1', day='2026-10-11', minutes=30, solved=True, help='hint'))
    session.add(Attempt(problem_id='G-217', day='2026-10-12', minutes=20, solved=False, help='none'))
    session.commit()
    data = engineering.overview(session, NOW)
    assert [x['id'] for x in data['retries']] == ['G-1']
    assert data['upcoming_retries'] == 1
    session.add(Attempt(problem_id='G-1', day='2026-10-12', minutes=20, solved=True, help='none'))
    session.commit()
    data = engineering.overview(session, NOW)
    assert not data['retries'] and data['recovered'] == 1
    assert [x['id'] for x in engineering.overview(session, NOW+timedelta(days=1))['retries']] == ['G-217']


def test_unknown_help_is_never_independent_evidence(session):
    session.add(Attempt(problem_id='G-1', day='2026-10-11', minutes=20, solved=True, help=None))
    session.commit()
    assert engineering.overview(session, NOW)['retries'][0]['reason'] == 'Help was not recorded'
    assert engineering.overview(session, NOW)['recovered'] == 0


def test_unknown_exercise_and_language_are_rejected():
    with pytest.raises(ApiError):
        Api().engineering_open('missing', 'python')
    with pytest.raises(ApiError):
        Api().engineering_open('events', 'go')


def test_separate_database_sessions_cannot_overwrite_a_newer_revision():
    api = Api()
    row = api.engineering_open('events', 'python')['attempt']
    with db.get_session() as stale:
        cached = stale.get(ComprehensionAttempt, row['id'])
        assert cached.revision == 0
        api.engineering_save(row['id'], 0, NOTES, 'newer', 'none')
        with pytest.raises(ValueError, match='another window'):
            engineering.save(stale, NOW, row['id'], 0, NOTES, 'stale', 'none')
    assert api.engineering_open('events', 'python')['attempt']['code'] == 'newer'


@pytest.mark.parametrize('exercise_id', ['events', 'cache', 'tasks'])
def test_python_starter_reproduces_bug_and_reference_passes_edge_cases(exercise_id):
    item = engineering.exercise(exercise_id, 'python')['languages']['python']
    starter, fixed = {}, {}
    exec(item['starter'], starter)
    exec(item['reference']['code'], fixed)
    if exercise_id == 'events':
        assert starter['accept_events'](['a', 'b', 'a']) == ['a', 'b', 'a']
        for values, expected in [([], []), (['a'], ['a']), (['a','a'], ['a']), (['b','a','b'], ['b','a'])]:
            assert fixed['accept_events'](values) == expected
    elif exercise_id == 'cache':
        bad = starter['Cache'](); bad.put('a','value',10,5)
        assert bad.get('a',15) == 'value'
        cache = fixed['Cache']()
        assert cache.get('missing',0) is None
        cache.put('a','old',10,5)
        assert cache.get('a',14) == 'old'
        assert cache.get('a',15) is None
        cache.put('a','zero',20,0)
        assert cache.get('a',20) is None
        cache.put('a','old',30,5); cache.put('a','new',31,10)
        assert cache.get('a',35) == 'new' and cache.get('a',41) is None
    else:
        assert starter['dependency_order']([0,1,2], [(0,1),(1,0)]) == [2]
        order = fixed['dependency_order']
        assert order([], []) == []
        assert set(order([0,1], [])) == {0,1}
        assert order([0,1,2], [(0,1),(1,2)]) == [0,1,2]
        assert order([0,1], [(0,1),(0,1)]) == [0,1]
        for tasks, edges in [([0],[(0,0)]), ([0,1,2],[(0,1),(1,0)])]:
            with pytest.raises(ValueError, match='cycle'):
                order(tasks, edges)


def test_cpp_starters_and_references_compile_and_exercise_contracts(tmp_path):
    import shutil
    import subprocess
    compiler = shutil.which('g++') or shutil.which('clang++') or shutil.which('c++')
    if not compiler:
        pytest.skip('No C++17 compiler available on this runner')
    harnesses = {
        'events': 'auto f=accept_events; assert(f({}).empty()); assert(f({"a"})==vector<string>{"a"}); assert((f({"b","a","b"})==vector<string>{"b","a"}));',
        'cache': 'Cache c; assert(!c.get("x",0)); c.put("a","old",10,5); assert(c.get("a",14)=="old"); assert(!c.get("a",15)); c.put("a","z",20,0); assert(!c.get("a",20)); c.put("a","old",30,5); c.put("a","new",31,10); assert(c.get("a",35)=="new"); assert(!c.get("a",41));',
        'tasks': 'assert(dependency_order(0,{}).empty()); assert(dependency_order(2,{}).size()==2); assert((dependency_order(3,{{0,1},{1,2}})==vector<int>{0,1,2})); assert((dependency_order(2,{{0,1},{0,1}})==vector<int>{0,1})); bool caught=false; try { dependency_order(3,{{0,1},{1,0}}); } catch(const invalid_argument&) { caught=true; } assert(caught); caught=false; try { dependency_order(1,{{0,0}}); } catch(const invalid_argument&) { caught=true; } assert(caught);'
    }
    for exercise_id, checks in harnesses.items():
        content = engineering.exercise(exercise_id, 'cpp')['languages']['cpp']
        for version in ('starter', 'reference'):
            code = content[version] if version == 'starter' else content[version]['code']
            source = tmp_path/f'{exercise_id}-{version}.cpp'
            executable = source.with_suffix('.exe')
            source.write_text('#include <cassert>\n#undef assert\n#define assert(...) do { if (!(__VA_ARGS__)) return 2; } while (0)\n'+code+'\nint main(){'+checks+'}\n', encoding='utf-8')
            subprocess.run([compiler,'-std=c++17',str(source),'-o',str(executable)], check=True, capture_output=True, timeout=45)
            result = subprocess.run([str(executable)], capture_output=True, timeout=5)
            assert (result.returncode == 0) == (version == 'reference'), result.stderr.decode(errors='replace')


def test_packaged_selftest_detects_missing_engineering_assets(tmp_path, monkeypatch):
    import shutil
    from google_prep.gui import launcher, server
    copied = tmp_path/'static'
    shutil.copytree(server.STATIC, copied)
    (copied/'engineering.js').unlink()
    monkeypatch.setattr(server, 'STATIC', copied)
    assert launcher.selftest() == 1


def test_independent_retry_keeps_prior_code_but_starts_clean(home):
    from google_prep.core import session_store
    from google_prep.modules import mock
    from google_prep.models import Problem
    with db.get_session() as s:
        problem = s.get(Problem, 'G-1')
        pad = mock.scratchpad(problem, 'python')
        trace = mock.trace_pad(problem, 'python')
        pad.write_text('old solution', encoding='utf-8')
        trace.write_text('old trace', encoding='utf-8')
    data = Api().mock_start('G-1', practice=True, fresh=True)
    assert 'old solution' not in data['code'] and 'old trace' not in data['trace']
    assert any(p.read_text(encoding='utf-8') == 'old solution' for p in (home/'retry_backups').rglob('*.txt'))
    assert any(p.read_text(encoding='utf-8') == 'old trace' for p in (home/'retry_backups').rglob('*.txt'))
    assert Api().mock_start(resume=True)['code'] == data['code']
    assert session_store.load()['problem_id'] == 'G-1'


def test_fresh_retry_requires_resolution_of_even_the_same_open_problem():
    api = Api()
    api.mock_start('G-1', practice=True)
    assert api.mock_start('G-1', practice=True, fresh=True)['conflict']

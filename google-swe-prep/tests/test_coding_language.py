"""Programming language choices must not replace earlier learning or code."""
import pytest
import shutil
import subprocess

from google_prep.gui.api import Api, ApiError
from google_prep.modules import fasttrack


def test_each_preparation_mode_remembers_its_language():
    api = Api()
    assert api.state()['settings']['coding_language'] == 'python'
    api.set_coding_language('cpp')
    assert api.fasttrack()['coding_language'] == 'cpp'
    api.preparation_mode('general')
    assert api.state()['settings']['coding_language'] == 'python'
    api.set_coding_language('python')
    api.preparation_mode('google')
    assert api.state()['settings']['coding_language'] == 'cpp'
    with pytest.raises(ApiError):
        api.set_coding_language('javascript')


@pytest.mark.fasttrack
def test_cpp_day_one_and_two_have_separate_progress_and_no_answer_leaks():
    api = Api()
    python = api.fasttrack()['days'][0]['exercises'][0]
    api.fasttrack_answer(python['id'], 0)
    api.fasttrack_complete(1)
    api.set_coding_language('cpp')
    cpp = api.fasttrack()
    assert cpp['current'] == 1
    assert 'C++' in api.today()['title']
    assert not cpp['days'][0]['progress']['done']
    assert cpp['days'][0]['progress']['answered'] == 0
    for day in cpp['days']:
        assert day['sections'] and day['exercises']
        for ex in day['exercises']:
            assert ex['id'].startswith('cpp-') and 'answer' not in ex and 'output' not in ex
    ex = cpp['days'][0]['exercises'][0]
    bundled = fasttrack.exercise(fasttrack.load_day(1, 'cpp'), ex['id'])
    result = api.fasttrack_answer(ex['id'], bundled['answer'])
    assert result['correct'] and result['output'] == bundled['output']
    assert result['output_kind'] == 'expected'
    api.fasttrack_complete(1, coding_language='cpp')
    api.set_coding_language('python')
    assert api.fasttrack()['days'][0]['progress']['done']
    assert api.fasttrack()['days'][0]['progress']['answered'] == 1


def test_stale_lesson_completion_uses_the_language_that_was_opened():
    api = Api()
    api.set_coding_language('cpp')
    api.set_coding_language('python')
    api.fasttrack_complete(1, coding_language='cpp')
    assert not api.fasttrack()['days'][0]['progress']['done']
    api.set_coding_language('cpp')
    assert api.fasttrack()['days'][0]['progress']['done']


def test_language_changes_do_not_overwrite_an_active_code_draft():
    api = Api()
    p = api.mock_target()['problem']['id']
    api.mock_start(p)
    api.mock_save(p, False, 45, 5, [False]*4, -1, code='python draft', trace='python trace')
    api.set_coding_language('cpp')
    resumed = api.mock_start(resume=True)
    assert resumed['coding_language'] == 'python' and resumed['code'] == 'python draft'
    api.mock_abandon()
    cpp = api.mock_start(p)
    assert cpp['coding_language'] == 'cpp' and cpp['code'].startswith('//')
    api.mock_save(p, False, 45, 8, [False]*4, -1, code='C++ draft', trace='C++ trace')
    assert api.mock_scorecard(p, False, 45, 8, [False]*4, -1)['trace_rows'] == 0
    api.mock_abandon()
    api.set_coding_language('python')
    original = api.mock_start(p)
    assert original['code'] == 'python draft' and original['trace'] == 'python trace'
    api.mock_abandon()
    api.set_coding_language('cpp')
    assert api.mock_start(p)['code'] == 'C++ draft'


def test_cpp_review_flags_do_not_leak_into_python():
    api = Api()
    api.set_coding_language('cpp')
    section = api.fasttrack()['days'][0]['sections'][0]['id']
    api.lesson_unclear(section)
    assert api.fasttrack()['unclear'] == [section]
    api.set_coding_language('python')
    assert api.fasttrack()['unclear'] == []


def test_stale_cpp_lesson_receives_its_own_review_flags_after_switch():
    api = Api()
    api.set_coding_language('cpp')
    section = api.fasttrack()['days'][0]['sections'][0]['id']
    api.set_coding_language('python')
    assert api.lesson_unclear(section, True)['unclear'] == [section]
    assert api.lesson_unclear(section, False)['unclear'] == []
    assert api.fasttrack()['unclear'] == []


def test_old_python_window_cannot_save_over_a_new_cpp_session():
    api = Api()
    p = api.mock_target()['problem']['id']
    api.mock_start(p)
    api.mock_abandon()
    api.set_coding_language('cpp')
    api.mock_start(p)
    with pytest.raises(ApiError):
        api.mock_save(p, False, 45, 10, [False]*4, -1, code='old Python window', coding_language='python')
    assert 'old Python window' not in api.mock_start(resume=True)['code']


@pytest.mark.parametrize('day', [1, 2])
def test_cpp_examples_compile_and_quick_checks_match_real_output(day, tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler not installed on this test host')
    lesson = fasttrack.load_day(day, 'cpp')
    for index, item in enumerate(lesson['sections'] + lesson['exercises']):
        source = tmp_path / f'example-{index}.cpp'
        binary = tmp_path / f'example-{index}.exe'
        source.write_text(item['code'], encoding='utf-8')
        built = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', str(source), '-o', str(binary)],
                               capture_output=True, text=True, timeout=30)
        assert built.returncode == 0, built.stderr
        ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
        assert ran.returncode == 0, ran.stderr
        if 'output' in item:
            assert ran.stdout.strip() == item['output'] == item['options'][item['answer']]

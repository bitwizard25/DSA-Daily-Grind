"""Learning progress is durable, validated and separate from practice notes."""
import pytest

from google_prep.gui.api import Api, ApiError, RPC


def test_catalog_has_learning_checklists_and_optional_exercises():
    data = Api().design_prep()
    assert {'design_prep', 'design_learning_save', 'design_prep_save'} <= RPC
    assert [t['id'] for t in data['tracks']] == ['lld', 'hld', 'ml']
    assert [t['learning_total'] for t in data['tracks']] == [10, 12, 8]
    ids = [item['id'] for t in data['tracks'] for g in t['learning'] for item in g['items']]
    assert len(ids) == len(set(ids)) == 30
    assert all(item['title'] and item['explanation'] for t in data['tracks'] for g in t['learning'] for item in g['items'])
    assert len(data['exercises']) == 6
    assert {r['id'] for r in data['references']} == {'llm-era', 'genai-interview'}
    for track in data['tracks']:
        assert all(r['book_id'] in {book['id'] for book in data['references']} for r in track['reading'])
    for ex in data['exercises']:
        assert ex['steps'] and all(step['prompts'] for step in ex['steps'])
        assert ex['review'] and ex['followups']


def test_checklist_save_deduplicates_reopens_and_keeps_tracks_independent():
    api = Api()
    api.design_learning_save('lld', ['lld-responsibilities', 'lld-contracts', 'lld-contracts'])
    api.design_learning_save('hld', ['hld-scope'])
    reopened = {t['id']: t for t in Api().design_prep()['tracks']}
    assert reopened['lld']['learned'] == ['lld-contracts', 'lld-responsibilities']
    assert reopened['lld']['learned_count'] == 2
    assert reopened['hld']['learned'] == ['hld-scope']
    assert reopened['ml']['learned_count'] == 0
    api.design_learning_save('lld', [])
    assert Api().design_prep()['tracks'][0]['learned'] == []


@pytest.mark.parametrize('track,completed', [
    ('missing', []), ('lld', 'lld-contracts'), ('lld', ['hld-scope']),
    ('lld', [None]), ('lld', None), (None, []), ('ml', ['ml-missing']),
])
def test_bad_checklist_payload_does_not_overwrite_progress(track, completed):
    api = Api()
    api.design_learning_save('lld', ['lld-contracts'])
    with pytest.raises(ApiError):
        api.design_learning_save(track, completed)
    assert Api().design_prep()['tracks'][0]['learned'] == ['lld-contracts']


def test_exercise_notes_and_status_preserve_omitted_fields():
    api = Api()
    api.design_prep_save('lld-cache', notes='A missing key returns None.')
    api.design_prep_save('lld-cache', practised=True)
    api.design_prep_save('lld-cache', notes='Capacity two: evict the least recently used key.')
    ex = Api().design_prep()['exercises'][0]
    assert ex['practised'] and ex['notes'].startswith('Capacity two:')
    assert Api().design_prep()['completed'] == 1
    api.design_prep_save('lld-cache', practised=False)
    assert not Api().design_prep()['exercises'][0]['practised']


@pytest.mark.parametrize('payload', [
    {'exercise_id':'unknown'}, {'exercise_id':'lld-cache', 'notes':7},
    {'exercise_id':'lld-cache', 'notes':'x'*12001},
    {'exercise_id':'lld-cache', 'practised':'true'},
    {'exercise_id':'lld-cache', 'practised':True},
    {'exercise_id':'lld-cache', 'notes':'  ', 'practised':True},
])
def test_invalid_exercise_save_writes_nothing(payload):
    with pytest.raises(ApiError):
        Api().design_prep_save(**payload)
    ex = Api().design_prep()['exercises'][0]
    assert ex['notes'] == '' and not ex['practised']

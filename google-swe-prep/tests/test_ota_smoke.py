import importlib.util
import json
from pathlib import Path

import pytest
from sqlmodel import select
from google_prep.db import get_session
from google_prep.models import Attempt, StarStory, TrainingCheckin, Setting


def smoke():
    file=Path(__file__).resolve().parents[1]/'scripts/ota_smoke.py'
    if not file.exists(): pytest.fail('Upgrade evidence script is missing')
    spec=importlib.util.spec_from_file_location('ota_smoke',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_upgrade_fixture_seeds_real_study_records_and_recovery(home):
    module=smoke(); module.seed(home)
    expected=json.loads((home/'ota-evidence.json').read_text())
    with get_session() as session:
        attempt=session.get(Attempt,expected['attempt_id'])
        assert attempt.solved and attempt.help == 'none'
        assert session.exec(select(StarStory).where(StarStory.title == expected['star_title'])).first() is not None
        assert session.get(TrainingCheckin,expected['day']).reflection == 'Check one edge case.'
        assert json.loads(session.get(Setting,'design:lld:learned').value) == ['lld-responsibilities']
    assert json.loads((home/'sessions/active.json').read_text())['problem_id'] == 'G-1'
    module.verify_records(home)
    with get_session() as session:
        session.delete(session.exec(select(StarStory).where(StarStory.title == expected['star_title'])).first());session.commit()
    with pytest.raises(AssertionError): module.verify_records(home)

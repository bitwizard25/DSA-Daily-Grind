import importlib.util
from pathlib import Path
import json
import pytest
from google_prep import __version__
from google_prep.updates.manifest import APP_ID


def builder():
    spec = importlib.util.spec_from_file_location('prep_builder', Path(__file__).resolve().parents[1]/'build.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_ci_refuses_to_package_without_matching_identity(home, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, 'ROOT', home)
    monkeypatch.setattr(module.platform, 'system', lambda:'Windows')
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_RUN_NUMBER', '200')
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('packaging started without build identity'))
    with pytest.raises(ValueError, match='build identity'):
        module.build_installer()


@pytest.mark.parametrize('change', [{}, {'build_number':199}, {'commit':'b'*40}, {'version':'0.1.0'}, {'platform':'source'}])
def test_ci_identity_matches_the_build_being_packaged(home, monkeypatch, change):
    module = builder()
    monkeypatch.setattr(module, 'ROOT', home)
    monkeypatch.setattr(module.platform, 'system', lambda:'Windows')
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_RUN_NUMBER', '200')
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    (home/'data').mkdir()
    info = {'schema':1, 'app_id':APP_ID, 'version':__version__, 'platform':'windows-x64',
            'build_number':200, 'commit':'a'*40, **change}
    (home/'data/build_info.json').write_text(json.dumps(info))
    if change:
        with pytest.raises(ValueError, match='build identity'):
            module.validate_build_identity()
    else:
        module.validate_build_identity()

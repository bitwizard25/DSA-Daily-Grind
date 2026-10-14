import importlib.util
import json
from pathlib import Path

import pytest


def release_module():
    file = Path(__file__).resolve().parents[1]/'scripts/release_update.py'
    if not file.exists(): pytest.fail('Release publisher is missing')
    spec = importlib.util.spec_from_file_location('release_update', file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class GitHub:
    def __init__(self, releases=()):
        self.releases = list(releases); self.created=[]; self.uploads=[]; self.published=[]; self.deleted=[]
    def list_releases(self): return self.releases
    def delete_draft(self, tag): self.deleted.append(tag)
    def create_draft(self, tag, commit): self.created.append((tag,commit))
    def upload(self, tag, paths): self.uploads.append((tag,paths))
    def publish(self, tag): self.published.append(tag)


@pytest.fixture
def package(tmp_path):
    module = release_module()
    build = module.make_metadata('windows-x64', {'GITHUB_SHA':'a'*40,'GITHUB_RUN_NUMBER':'100'})
    (tmp_path/'GoogleSWEPrep-Setup.exe').write_bytes(b'abc')
    (tmp_path/'build_info.json').write_text(json.dumps(build))
    manifest = module.make_manifest(build, tmp_path/'GoogleSWEPrep-Setup.exe')
    (tmp_path/'google-prep-update.json').write_text(json.dumps(manifest))
    return module, build, manifest, tmp_path


def test_deterministic_manifest(package):
    module, build, manifest, folder = package
    assert manifest['tag'] == 'google-prep-v1.6.0-b100'
    assert manifest['installer']['size'] == 3
    assert manifest['installer']['sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert manifest['commit'] == build['commit']


def test_publish_only_main(package):
    module, build, manifest, folder = package; github=GitHub()
    with pytest.raises(ValueError): module.publish_update(github,folder,'refs/heads/codex/ota-updates')
    assert not github.created


def test_published_assets_are_never_overwritten(package):
    module, build, manifest, folder=package
    github=GitHub([{'tag_name':manifest['tag'],'draft':False,'prerelease':False}])
    assert module.publish_update(github,folder,'refs/heads/main') == 'already-published'
    assert not github.uploads and not github.deleted and not github.published


def test_late_older_build_not_promoted(package):
    module, build, manifest, folder=package
    github=GitHub([{'tag_name':'google-prep-v1.6.0-b101','draft':False,'prerelease':False}])
    assert module.publish_update(github,folder,'refs/heads/main') == 'superseded'
    assert not github.created


def test_incomplete_draft_is_recreated_before_publication(package):
    module, build, manifest, folder=package
    github=GitHub([{'tag_name':manifest['tag'],'draft':True}])
    assert module.publish_update(github,folder,'refs/heads/main') == 'published'
    assert github.deleted == [manifest['tag']]
    assert github.created == [(manifest['tag'],'a'*40)]
    assert len(github.uploads[0][1]) == 3 and github.published == [manifest['tag']]


def test_modified_installer_cannot_publish(package):
    module, build, manifest, folder=package
    (folder/'GoogleSWEPrep-Setup.exe').write_bytes(b'bad'); github=GitHub()
    with pytest.raises(ValueError): module.publish_update(github,folder,'refs/heads/main')
    assert not github.created


def test_newer_release_appearing_during_upload_stays_latest(package):
    module, build, manifest, folder=package
    class Racing(GitHub):
        def upload(self,tag,paths):
            super().upload(tag,paths)
            self.releases.append({'tag_name':'google-prep-v1.6.0-b101','draft':False})
    github=Racing()
    assert module.publish_update(github,folder,'refs/heads/main') == 'superseded'
    assert not github.published


@pytest.mark.parametrize('sha,run',[('bad','100'),('a'*40,'0'),('a'*40,'-1')])
def test_invalid_ci_identity_is_rejected(sha,run):
    with pytest.raises(ValueError): release_module().make_metadata('windows-x64', {'GITHUB_SHA':sha,'GITHUB_RUN_NUMBER':run})

"""CI release contract: build identity, manifest and safe publishing. A fake GitHub adapter is used;
no test talks to the real repository."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from google_prep.updates.manifest import validate_candidate

ROOT = Path(__file__).resolve().parent.parent
SHA = "0123456789abcdef0123456789abcdef01234567"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


release = load("release_update")


@pytest.fixture
def build():
    return release.build_metadata("1.6.0", "100", SHA)


@pytest.fixture
def bundle(tmp_path, build):
    """What the Windows job uploads: installer, manifest and build identity."""
    folder = tmp_path / "update-release"
    folder.mkdir()
    installer = folder / release.INSTALLER
    installer.write_bytes(b"abc" * 1000)
    (folder / release.BUILD_INFO).write_text(json.dumps(build), encoding="utf-8")
    manifest = release.make_manifest(build, installer)
    (folder / release.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class FakeGitHub:
    def __init__(self, releases=None):
        self.releases = releases or []
        self.uploads, self.published, self.deleted, self.created = [], [], [], []
        self.on_create = None
        self._next = 1000

    def list_releases(self):
        return [dict(r) for r in self.releases]

    def create_draft(self, tag, commit, name, body):
        self._next += 1
        rel = {"id": self._next, "tag_name": tag, "target_commitish": commit, "name": name, "body": body,
               "draft": True, "prerelease": False, "assets": []}
        self.releases.append(rel)
        self.created.append(tag)
        if self.on_create:
            self.on_create(self)
        return dict(rel)

    def upload(self, rel, path, name, content_type):
        self.uploads.append((rel["id"], name, Path(path).read_bytes()))
        for r in self.releases:
            if r["id"] == rel["id"]:
                r["assets"].append({"name": name, "size": Path(path).stat().st_size, "state": "uploaded"})

    def publish(self, rel):
        for r in self.releases:
            if r["id"] == rel["id"]:
                r["draft"] = False
                self.published.append(r["tag_name"])

    def delete(self, rel):
        self.deleted.append(rel["id"])
        self.releases = [r for r in self.releases if r["id"] != rel["id"]]


def published(tag, draft=False, rid=1):
    return {"id": rid, "tag_name": tag, "draft": draft, "prerelease": False, "assets": []}


# ---- build identity and manifest --------------------------------------------------------------------
def test_build_metadata_is_exact_and_validated(build):
    assert build == {"schema": 1, "app_id": "google-swe-prep", "platform": "windows-x64",
                     "version": "1.6.0", "build_number": 100, "commit": SHA}
    for args in (("1.6", "100", SHA), ("1.6.0", "0", SHA), ("1.6.0", "x", SHA), ("1.6.0", "100", "short")):
        with pytest.raises(ValueError):
            release.build_metadata(*args)


def test_manifest_is_deterministic_and_accepted_by_the_app(tmp_path, build):
    installer = tmp_path / release.INSTALLER
    installer.write_bytes(b"abc")
    first, second = release.make_manifest(build, installer), release.make_manifest(build, installer)
    assert first == second
    assert first["tag"] == "google-prep-v1.6.0-b100"
    assert first["installer"] == {"name": release.INSTALLER, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    url = f"https://github.com/{release.REPOSITORY}/releases/download/{first['tag']}/{release.INSTALLER}"
    gh_release = {"tag_name": first["tag"], "assets": [{"name": release.INSTALLER, "state": "uploaded", "size": 3,
                                                        "browser_download_url": url}]}
    candidate = validate_candidate(gh_release, first, {"version": "1.5.0", "build_number": 0})
    assert candidate.build_number == 100  # the app's own validator accepts what CI produces


def test_metadata_command_writes_from_ci_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "42")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    out = tmp_path / "data" / "build_info.json"
    assert release.main(["metadata", "--output", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["build_number"] == 42 and data["commit"] == SHA and data["platform"] == "windows-x64"


# ---- publishing rules -----------------------------------------------------------------------------
def test_development_branch_cannot_publish(bundle):
    gh = FakeGitHub()
    assert release.publish_update(gh, bundle, "refs/heads/codex/ota-updates") == "refused-branch"
    assert gh.created == [] and gh.uploads == []


def test_main_build_is_published_with_all_assets(bundle):
    gh = FakeGitHub([published("google-prep-v1.5.0-b90")])
    assert release.publish_update(gh, bundle, "refs/heads/main") == "published"
    assert gh.published == ["google-prep-v1.6.0-b100"]
    assert sorted(name for _, name, _ in gh.uploads) == sorted([release.INSTALLER, release.MANIFEST, release.BUILD_INFO])
    rel = [r for r in gh.releases if r["tag_name"] == "google-prep-v1.6.0-b100"][0]
    assert rel["target_commitish"] == SHA and rel["draft"] is False


def test_published_build_is_not_overwritten(bundle):
    gh = FakeGitHub([published("google-prep-v1.6.0-b100")])
    assert release.publish_update(gh, bundle, "refs/heads/main") == "skipped-existing"
    assert gh.uploads == [] and gh.published == []


def test_late_older_build_is_not_promoted(bundle):
    gh = FakeGitHub([published("google-prep-v1.6.0-b101")])
    assert release.publish_update(gh, bundle, "refs/heads/main") == "skipped-older"
    assert gh.uploads == [] and gh.published == []


def test_newer_build_published_during_upload_wins(bundle):
    gh = FakeGitHub([published("google-prep-v1.6.0-b99")])
    gh.on_create = lambda g: g.releases.append(published("google-prep-v1.6.0-b101", rid=7))
    assert release.publish_update(gh, bundle, "refs/heads/main") == "skipped-older"
    assert gh.published == [] and all(r["tag_name"] != "google-prep-v1.6.0-b100" for r in gh.releases)


def test_incomplete_draft_is_discarded_and_recreated(bundle):
    gh = FakeGitHub([published("google-prep-v1.6.0-b100", draft=True, rid=55), published("google-prep-v1.5.0-b1", rid=2)])
    assert release.publish_update(gh, bundle, "refs/heads/main") == "published"
    assert 55 in gh.deleted and gh.published == ["google-prep-v1.6.0-b100"]
    assert [r["id"] for r in gh.releases if r["tag_name"] == "google-prep-v1.5.0-b1"] == [2]  # untouched


def test_other_products_and_drafts_do_not_count_as_newer(bundle):
    gh = FakeGitHub([published("v9.9.9"), published("google-prep-v9.0.0-b999", draft=True, rid=3)])
    assert release.publish_update(gh, bundle, "refs/heads/main") == "published"


@pytest.mark.parametrize("tamper", ["installer", "manifest", "build"])
def test_mismatched_bundle_is_refused(bundle, tamper):
    if tamper == "installer":
        (bundle / release.INSTALLER).write_bytes(b"changed")
    elif tamper == "manifest":
        data = json.loads((bundle / release.MANIFEST).read_text())
        data["commit"] = "f" * 40
        (bundle / release.MANIFEST).write_text(json.dumps(data))
    else:
        data = json.loads((bundle / release.BUILD_INFO).read_text())
        data["build_number"] = 101
        (bundle / release.BUILD_INFO).write_text(json.dumps(data))
    gh = FakeGitHub()
    with pytest.raises(ValueError):
        release.publish_update(gh, bundle, "refs/heads/main")
    assert gh.created == []


# ---- upgrade smoke-test helpers (the Windows-only install step runs in CI) ------------------------------
def test_smoke_seed_stage_and_verify_round_trip(tmp_path, bundle, monkeypatch):
    smoke = load("ota_smoke")
    home = tmp_path / "profile with spaces ü"
    expected = smoke.seed(home)
    assert (home / "prep.db").exists() and expected["attempt_problem"]
    smoke.stage(home, bundle / release.INSTALLER, bundle / release.MANIFEST)
    pending = json.loads((home / "updates" / "pending.json").read_text())
    assert pending["candidate"]["tag"] == "google-prep-v1.6.0-b100"
    assert (home / "updates" / "google-prep-v1.6.0-b100" / "installer.exe").read_bytes() == b"abc" * 1000
    # pretend the helper finished: installed identity, result and a verified backup
    install = tmp_path / "Google SWE II OS"
    (install / "_internal" / "data").mkdir(parents=True)
    (install / "_internal" / "data" / "build_info.json").write_text((bundle / release.BUILD_INFO).read_text())
    (home / "updates" / "result.json").write_text(json.dumps({"tag": "google-prep-v1.6.0-b100", "status": "installed"}))
    import sqlite3
    (home / "updates" / "backups").mkdir()
    with sqlite3.connect(home / "prep.db") as src, sqlite3.connect(home / "updates" / "backups" / "prep-1.db") as dst:
        src.backup(dst)
    assert smoke.verify(home, install, 100) == []
    (home / "updates" / "result.json").write_text(json.dumps({"tag": "google-prep-v1.6.0-b100", "status": "failed"}))
    assert any("result" in p for p in smoke.verify(home, install, 100))


# ---- public visibility check -------------------------------------------------------------------------
class Found:
    def __init__(self, tag, key, sha):
        self.tag, self.key, self.installer_sha256 = tag, key, sha


def test_verify_public_waits_for_github_to_list_a_fresh_release(bundle):
    manifest = json.loads((bundle / release.MANIFEST).read_text())
    ours = release.version_key(manifest["version"], manifest["build_number"])
    answers = [None, Found("google-prep-v1.5.0-b1", release.version_key("1.5.0", 1), "x"),
               Found(manifest["tag"], ours, manifest["installer"]["sha256"])]
    sleeps = []
    assert release.verify_public(bundle, fetch=lambda _: answers.pop(0), sleep=sleeps.append) == manifest["tag"]
    assert sleeps == [release.VERIFY_DELAY_SECONDS] * 2


def test_verify_public_gives_up_after_the_last_attempt(bundle):
    sleeps = []
    with pytest.raises(SystemExit, match="No valid update"):
        release.verify_public(bundle, fetch=lambda _: None, sleep=sleeps.append, attempts=3)
    assert len(sleeps) == 2


def test_verify_public_never_retries_a_checksum_mismatch(bundle):
    manifest = json.loads((bundle / release.MANIFEST).read_text())
    ours = release.version_key(manifest["version"], manifest["build_number"])
    sleeps = []
    with pytest.raises(SystemExit, match="does not match"):
        release.verify_public(bundle, fetch=lambda _: Found(manifest["tag"], ours, "0" * 64), sleep=sleeps.append)
    assert sleeps == []


def test_verify_public_accepts_a_newer_build(bundle):
    newer = Found("google-prep-v1.6.0-b101", release.version_key("1.6.0", 101), "y")
    assert release.verify_public(bundle, fetch=lambda _: newer, sleep=lambda _: None) == "google-prep-v1.6.0-b101"

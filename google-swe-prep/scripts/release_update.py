"""CI side of OTA updates: stamp the build identity, write the update manifest, publish a release.

    python scripts/release_update.py metadata --output data/build_info.json
    python scripts/release_update.py manifest --installer dist/GoogleSWEPrep-Setup.exe \
        --build-info data/build_info.json --output dist/google-prep-update.json
    python scripts/release_update.py publish --directory update-release      (CI, main only)
    python scripts/release_update.py verify-public --directory update-release

The manifest format and validation rules are the app's own (google_prep/updates/manifest.py), so CI
can never publish something the installed app would reject. Publishing only happens from main,
never overwrites a published release and never lets an older build replace a newer one."""
import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google_prep import __version__  # noqa: E402
from google_prep.updates.manifest import (  # noqa: E402
    APP_ID, INSTALLER, MANIFEST, PLATFORM, REPOSITORY, TAG_PATTERN, Candidate, validate_candidate, version_key,
)

BUILD_INFO = "build_info.json"
MAIN_REF = "refs/heads/main"
API = "https://api.github.com"


# ---- build identity and manifest ------------------------------------------------------------------------
def build_metadata(version: str, run_number: str, commit: str) -> dict:
    try:
        build_number = int(str(run_number))
    except ValueError as exc:
        raise ValueError("GITHUB_RUN_NUMBER must be a whole number") from exc
    if build_number < 1:
        raise ValueError("Build number must be positive")
    version_key(version, build_number)  # same rules the app uses
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("GITHUB_SHA must be a full lowercase commit id")
    return {"schema": 1, "app_id": APP_ID, "platform": PLATFORM, "version": version,
            "build_number": build_number, "commit": commit}


def tag_for(build: dict) -> str:
    return f"google-prep-v{build['version']}-b{build['build_number']}"


def file_digest(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def make_manifest(build: dict, installer: Path) -> dict:
    build = build_metadata(build["version"], build["build_number"], build["commit"])
    size, sha256 = file_digest(installer)
    tag = tag_for(build)
    manifest = Candidate(build["version"], build["build_number"], build["commit"], tag,
                         f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}", size, sha256).manifest()
    _check_with_app_rules(manifest)
    return manifest


def _check_with_app_rules(manifest: dict) -> None:
    tag = manifest["tag"]
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}"
    release = {"tag_name": tag, "assets": [{"name": INSTALLER, "state": "uploaded",
                                            "size": manifest["installer"]["size"], "browser_download_url": url}]}
    if validate_candidate(release, manifest, {"version": "0.0.0", "build_number": 0}) is None:
        raise ValueError("The app would not accept this update")


def read_bundle(directory: Path) -> tuple[dict, dict, Path]:
    """Load what the Windows job uploaded and prove the three files belong together."""
    directory = Path(directory)
    build = json.loads((directory / BUILD_INFO).read_text(encoding="utf-8"))
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    installer = directory / INSTALLER
    if build.get("schema") != 1 or build.get("app_id") != APP_ID or build.get("platform") != PLATFORM:
        raise ValueError("Build identity is not for this app")
    expected = make_manifest(build, installer)
    if manifest != expected:
        raise ValueError("Manifest does not match the installer and build identity")
    return build, manifest, installer


# ---- publishing -----------------------------------------------------------------------------------------
def _key(tag: str):
    match = TAG_PATTERN.fullmatch(str(tag))
    return version_key(match[1], int(match[2])) if match else None


def _latest_published(releases: list) -> tuple | None:
    keys = [_key(r.get("tag_name")) for r in releases if not r.get("draft") and not r.get("prerelease")]
    keys = [k for k in keys if k is not None]
    return max(keys) if keys else None


def publish_update(github, directory: Path, ref: str) -> str:
    """Returns refused-branch, skipped-existing, skipped-older or published."""
    if ref != MAIN_REF:
        return "refused-branch"
    build, manifest, installer = read_bundle(directory)
    tag, key = manifest["tag"], version_key(build["version"], build["build_number"])
    releases = github.list_releases()
    if any(r.get("tag_name") == tag and not r.get("draft") for r in releases):
        return "skipped-existing"  # a published build is immutable
    latest = _latest_published(releases)
    if latest is not None and latest >= key:
        return "skipped-older"
    for stale in [r for r in releases if r.get("tag_name") == tag and r.get("draft")]:
        github.delete(stale)  # an earlier upload for this build failed half-way
    body = (f"Automatic update for Google SWE II OS {build['version']} (CI build {build['build_number']}, "
            f"commit {build['commit']}). Installed apps pick this up by themselves; to install by hand, "
            f"download {INSTALLER}.")
    draft = github.create_draft(tag, build["commit"], f"Google SWE II OS {build['version']} (build {build['build_number']})", body)
    try:
        github.upload(draft, installer, INSTALLER, "application/vnd.microsoft.portable-executable")
        github.upload(draft, Path(directory) / MANIFEST, MANIFEST, "application/json")
        github.upload(draft, Path(directory) / BUILD_INFO, BUILD_INFO, "application/json")
        latest = _latest_published(github.list_releases())  # another run may have finished meanwhile
        if latest is not None and latest >= key:
            github.delete(draft)
            return "skipped-older"
        github.publish(draft)
    except BaseException:
        try:
            github.delete(draft)  # never leave a half-uploaded draft behind
        except Exception:  # noqa: BLE001 - the original failure is the one to report
            pass
        raise
    return "published"


class GitHub:
    """Minimal GitHub REST client for this repository (token only exists in the CI publish job)."""

    def __init__(self, token: str, repository: str = REPOSITORY):
        if not token:
            raise ValueError("GH_TOKEN is required to publish")
        self.token, self.repository = token, repository

    def _call(self, method, url, data=None, content_type="application/json"):
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "google-prep-release"}
        if data is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def list_releases(self):
        out = []
        for page in range(1, 11):
            data = self._call("GET", f"{API}/repos/{self.repository}/releases?per_page=100&page={page}")
            out.extend(data)
            if len(data) < 100:
                break
        return out

    def create_draft(self, tag, commit, name, body):
        payload = {"tag_name": tag, "target_commitish": commit, "name": name, "body": body,
                   "draft": True, "prerelease": False, "make_latest": "false"}
        return self._call("POST", f"{API}/repos/{self.repository}/releases", json.dumps(payload).encode())

    def upload(self, release, path, name, content_type):
        base = release["upload_url"].split("{", 1)[0]
        url = f"{base}?{urllib.parse.urlencode({'name': name})}"
        return self._call("POST", url, Path(path).read_bytes(), content_type)

    def publish(self, release):
        payload = {"draft": False, "make_latest": "true"}
        return self._call("PATCH", f"{API}/repos/{self.repository}/releases/{release['id']}", json.dumps(payload).encode())

    def delete(self, release):
        return self._call("DELETE", f"{API}/repos/{self.repository}/releases/{release['id']}")


def verify_public(directory: Path) -> str:
    """Anonymous lookup through the app's own client: the published build must be what an app sees."""
    from google_prep.updates.client import fetch_candidate

    _, manifest, _ = read_bundle(directory)
    found = fetch_candidate({"version": "0.0.0", "build_number": 0})
    if found is None:
        raise SystemExit("No valid update is visible publicly")
    ours = version_key(manifest["version"], manifest["build_number"])
    if found.key < ours:
        raise SystemExit(f"Public update is {found.tag}, older than {manifest['tag']}")
    if found.key == ours and found.installer_sha256 != manifest["installer"]["sha256"]:
        raise SystemExit(f"Public {found.tag} does not match the built installer")
    return found.tag  # a newer build may already have superseded ours; that is fine


# ---- command line -----------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    meta = sub.add_parser("metadata")
    meta.add_argument("--output", type=Path, required=True)
    man = sub.add_parser("manifest")
    man.add_argument("--installer", type=Path, required=True)
    man.add_argument("--build-info", type=Path, required=True)
    man.add_argument("--output", type=Path, required=True)
    for name in ("publish", "verify-public"):
        p = sub.add_parser(name)
        p.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "metadata":
        build = build_metadata(__version__, os.environ.get("GITHUB_RUN_NUMBER", ""), os.environ.get("GITHUB_SHA", ""))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
        print(f"build identity: {tag_for(build)} ({build['commit']})")
    elif args.command == "manifest":
        build = json.loads(args.build_info.read_text(encoding="utf-8"))
        manifest = make_manifest(build, args.installer)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"manifest: {manifest['tag']} sha256={manifest['installer']['sha256']} size={manifest['installer']['size']}")
    elif args.command == "publish":
        result = publish_update(GitHub(os.environ.get("GH_TOKEN", "")), args.directory, os.environ.get("GITHUB_REF", ""))
        print(f"publish: {result}")
    else:
        print(f"public update visible: {verify_public(args.directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded HTTPS release lookup and streamed installer verification."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .manifest import MANIFEST, REPOSITORY, TAG_PATTERN, Candidate, candidate_from_dict, validate_candidate, version_key

ASSET_HOSTS = {'release-assets.githubusercontent.com', 'objects.githubusercontent.com', 'github-releases.githubusercontent.com'}
API_ROOT = f'https://api.github.com/repos/{REPOSITORY}/releases'


class RateLimited(OSError):
    def __init__(self, retry_after_seconds):
        super().__init__('GitHub update checks are temporarily rate limited.')
        self.retry_after_seconds = retry_after_seconds


def allowed_url(url: str) -> bool:
    value = urllib.parse.urlsplit(url)
    if value.scheme != 'https' or value.username or value.password or value.port not in (None, 443) or value.fragment:
        return False
    if value.hostname in ASSET_HOSTS:
        return True
    if value.hostname == 'api.github.com':
        return value.path.startswith(f'/repos/{REPOSITORY}/releases')
    return value.hostname == 'github.com' and value.path.startswith(f'/{REPOSITORY}/releases/download/')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError('Unexpected update redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(request):
    if not allowed_url(request.full_url):
        raise ValueError('Unexpected update address')
    try:
        return urllib.request.build_opener(SafeRedirect).open(request, timeout=20)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            try:
                retry = max(float(exc.headers.get('Retry-After', 0)), float(exc.headers.get('X-RateLimit-Reset', 0))-time.time())
            except (ValueError, TypeError):
                retry = 3600
            raise RateLimited(min(24*3600, max(60, retry))) from exc
        raise


def request(url):
    if not allowed_url(url):
        raise ValueError('Unexpected update address')
    return urllib.request.Request(url, headers={'User-Agent':'GoogleSWEPrep-Updater', 'Accept':'application/vnd.github+json'})


def get_json(url, limit, opener):
    with opener(request(url)) as response:
        if not allowed_url(response.geturl()):
            raise ValueError('Unexpected update redirect')
        data = response.read(limit+1)
    if len(data) > limit:
        raise ValueError('Update information is too large')
    return json.loads(data)


def fetch_candidate(installed: dict, open_url=open_url) -> Candidate | None:
    releases = []
    for page in range(1, 4):
        data = get_json(f'{API_ROOT}?per_page=100&page={page}', 2*1024*1024, open_url)
        if not isinstance(data, list):
            raise ValueError('Invalid GitHub update response')
        releases.extend(data)
        if len(data) < 100:
            break
    eligible = []
    for release in releases:
        if not isinstance(release, dict) or release.get('draft') or release.get('prerelease'):
            continue
        match = TAG_PATTERN.fullmatch(str(release.get('tag_name','')))
        if match:
            key = version_key(match[1], int(match[2]))
            if key > version_key(installed['version'], installed['build_number']):
                eligible.append((key, release))
    for _, release in sorted(eligible, key=lambda row:row[0], reverse=True)[:5]:
        tag = release['tag_name']
        url = f'https://github.com/{REPOSITORY}/releases/download/{tag}/{MANIFEST}'
        if not any(a.get('name') == MANIFEST and a.get('state') == 'uploaded' and a.get('browser_download_url') == url
                   for a in release.get('assets', []) if isinstance(a, dict)):
            continue
        try:
            manifest = get_json(url, 64*1024, open_url)
            candidate = validate_candidate(release, manifest, installed)
        except (ValueError, urllib.error.HTTPError) as exc:
            if isinstance(exc, RateLimited):
                raise
            continue  # a malformed/incomplete release never blocks an older valid update
        if candidate:
            return candidate
    return None


def download_candidate(candidate: Candidate, cache_dir: Path, open_url=open_url, progress=None) -> Path:
    candidate = candidate_from_dict(candidate.as_dict())
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink():
        raise ValueError('Update cache must not be a link')
    partial, destination = cache_dir/'installer.part', cache_dir/'installer.exe'
    if partial.is_symlink() or destination.is_symlink():
        raise ValueError('Update files must not be links')
    digest, count = hashlib.sha256(), 0
    try:
        with open_url(request(candidate.installer_url)) as response, partial.open('wb') as out:
            if not allowed_url(response.geturl()):
                raise ValueError('Unexpected update redirect')
            while chunk := response.read(64*1024):
                count += len(chunk)
                if count > candidate.installer_size:
                    raise ValueError('Update download exceeds its declared size')
                digest.update(chunk)
                out.write(chunk)
                if progress:
                    progress(count, candidate.installer_size)
            out.flush()
            os.fsync(out.fileno())
        if count != candidate.installer_size or digest.hexdigest() != candidate.installer_sha256:
            raise ValueError('Update download could not be verified')
        partial.replace(destination)
        return destination
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

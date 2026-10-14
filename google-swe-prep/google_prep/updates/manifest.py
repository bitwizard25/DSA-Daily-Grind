"""Pure update identity and validation; no network or installation side effects."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from google_prep import __version__

APP_ID = 'google-swe-prep'
REPOSITORY = 'bitwizard25/DSA-Daily-Grind'
PLATFORM = 'windows-x64'
INSTALLER = 'GoogleSWEPrep-Setup.exe'
MANIFEST = 'google-prep-update.json'
MAX_INSTALLER_BYTES = 500 * 1024 * 1024
TAG_PATTERN = re.compile(r'google-prep-v(\d+\.\d+\.\d+)-b([1-9]\d*)\Z')


def version_key(version: str, build_number: int) -> tuple:
    if not isinstance(version, str) or not re.fullmatch(r'\d{1,6}\.\d{1,6}\.\d{1,6}', version):
        raise ValueError('Invalid update version')
    if type(build_number) is not int or not 0 <= build_number <= 10**15:
        raise ValueError('Invalid build number')
    return (*map(int, version.split('.')), build_number)


@dataclass(frozen=True)
class Candidate:
    version: str
    build_number: int
    commit: str
    tag: str
    installer_url: str
    installer_size: int
    installer_sha256: str

    @property
    def key(self):
        return version_key(self.version, self.build_number)

    def as_dict(self):
        return asdict(self)

    def manifest(self):
        return {'schema':1, 'app_id':APP_ID, 'repository':REPOSITORY, 'platform':PLATFORM,
                'version':self.version, 'build_number':self.build_number, 'commit':self.commit, 'tag':self.tag,
                'installer':{'name':INSTALLER, 'size':self.installer_size, 'sha256':self.installer_sha256}}


def read_build_info(resource_root: Path) -> dict:
    fallback = {'schema':1, 'app_id':APP_ID, 'platform':'source', 'version':__version__, 'build_number':0, 'commit':''}
    try:
        data = json.loads((resource_root / 'data/build_info.json').read_text(encoding='utf-8'))
        version_key(data['version'], data['build_number'])
        if data.get('schema') != 1 or data.get('app_id') != APP_ID or data['build_number'] == 0:
            return fallback
        if not isinstance(data.get('commit'), str) or not re.fullmatch('[0-9a-f]{40}', data['commit']):
            return fallback
        if data.get('platform') != PLATFORM:
            return fallback
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def validate_candidate(release: dict, manifest: dict, installed: dict) -> Candidate | None:
    if not isinstance(release, dict) or not isinstance(manifest, dict):
        raise ValueError('Invalid update information')
    if release.get('draft') or release.get('prerelease'):
        return None
    tag = release.get('tag_name', '')
    if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag):
        return None
    if type(manifest.get('schema')) is not int or manifest['schema'] != 1:
        raise ValueError('Unsupported update manifest')
    for name, expected in [('app_id',APP_ID), ('repository',REPOSITORY), ('platform',PLATFORM), ('tag',tag)]:
        if manifest.get(name) != expected:
            raise ValueError('Update does not match this app')
    try:
        key = version_key(manifest['version'], manifest['build_number'])
        if manifest['build_number'] == 0 or tag != f"google-prep-v{manifest['version']}-b{manifest['build_number']}":
            raise ValueError('Update build identity does not match')
        commit = manifest['commit']
        if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Invalid update commit')
        binary = manifest['installer']
        size, digest = binary['size'], binary['sha256']
        if binary.get('name') != INSTALLER or type(size) is not int or not 0 < size <= MAX_INSTALLER_BYTES:
            raise ValueError('Invalid installer size or name')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid installer checksum')
        assets = [a for a in release.get('assets', []) if isinstance(a, dict) and a.get('name') == INSTALLER]
        expected_url = f'https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}'
        if len(assets) != 1:
            raise ValueError('Update installer is missing or duplicated')
        asset = assets[0]
        if asset.get('state') != 'uploaded' or type(asset.get('size')) is not int or asset['size'] != size:
            raise ValueError('Update installer is incomplete')
        if asset.get('browser_download_url') != expected_url:
            raise ValueError('Unexpected update download address')
        if asset.get('digest') is not None and asset['digest'] != 'sha256:' + digest:
            raise ValueError('Release checksum does not match')
        if key <= version_key(installed['version'], installed['build_number']):
            return None
        return Candidate(manifest['version'], manifest['build_number'], commit, tag, expected_url, size, digest)
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete update manifest') from exc


def candidate_from_dict(data: dict) -> Candidate:
    """Revalidate locally staged metadata; no destination is accepted from callers."""
    try:
        candidate = Candidate(**data)
        release = {'tag_name':candidate.tag, 'assets':[{'name':INSTALLER, 'size':candidate.installer_size,
                   'state':'uploaded', 'browser_download_url':candidate.installer_url}]}
        valid = validate_candidate(release, candidate.manifest(), {'version':'0.0.0','build_number':0})
        if valid is None:
            raise ValueError('Invalid staged update')
        return valid
    except (TypeError, AttributeError) as exc:
        raise ValueError('Invalid staged update') from exc

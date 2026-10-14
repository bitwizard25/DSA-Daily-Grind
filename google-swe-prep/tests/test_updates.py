import copy
import hashlib
import importlib
import io

import pytest

REPO = 'bitwizard25/DSA-Daily-Grind'
TAG = 'google-prep-v1.6.0-b100'
URL = f'https://github.com/{REPO}/releases/download/{TAG}/GoogleSWEPrep-Setup.exe'


def module(name):
    try:
        return importlib.import_module('google_prep.updates.' + name)
    except ModuleNotFoundError:
        pytest.fail('The OTA update implementation is missing')


@pytest.fixture
def manifest():
    return {'schema': 1, 'app_id': 'google-swe-prep', 'repository': REPO, 'platform': 'windows-x64',
            'version': '1.6.0', 'build_number': 100, 'commit': 'a'*40, 'tag': TAG,
            'installer': {'name': 'GoogleSWEPrep-Setup.exe', 'size': 3,
                          'sha256': hashlib.sha256(b'abc').hexdigest()}}


@pytest.fixture
def release(manifest):
    return {'tag_name': TAG, 'draft': False, 'prerelease': False,
            'assets': [{'name': 'GoogleSWEPrep-Setup.exe', 'state': 'uploaded', 'size': 3,
                        'browser_download_url': URL, 'digest':'sha256:'+manifest['installer']['sha256']}]}


def test_same_version_new_build_is_detected(release, manifest):
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.6.0', 'build_number':99})
    assert candidate.build_number == 100 and candidate.installer_url == URL


@pytest.mark.parametrize('installed', [
    {'version':'1.6.0','build_number':100}, {'version':'1.6.0','build_number':101},
    {'version':'1.7.0','build_number':1}])
def test_equal_and_older_updates_are_skipped(release, manifest, installed):
    assert module('manifest').validate_candidate(release, manifest, installed) is None


@pytest.mark.parametrize('field,value', [('schema',True), ('app_id','other'), ('repository','other/repo'),
    ('platform','linux-x64'), ('build_number',True), ('build_number',0), ('version','1.6.0-rc1'),
    ('version',None), ('commit','bad'), ('tag','google-prep-v1.6.0-b101')])
def test_invalid_manifest_rejected(release, manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError):
        module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})


@pytest.mark.parametrize('field,value', [('size',True), ('size',0), ('size',501*1024*1024),
                                      ('sha256','bad'), ('name','other.exe')])
def test_invalid_installer_rejected(release, manifest, field, value):
    manifest['installer'][field] = value
    with pytest.raises(ValueError):
        module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})


@pytest.mark.parametrize('change', ['draft','prerelease','missing','wrong_url','wrong_digest','wrong_size'])
def test_incomplete_or_untrusted_release_cannot_update(release, manifest, change):
    if change in ('draft','prerelease'):
        release[change] = True
        assert module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0}) is None
        return
    if change == 'missing': release['assets'] = []
    elif change == 'wrong_url': release['assets'][0]['browser_download_url'] = 'https://evil.example/installer.exe'
    elif change == 'wrong_digest': release['assets'][0]['digest'] = 'sha256:'+'0'*64
    elif change == 'wrong_size': release['assets'][0]['size'] = 4
    with pytest.raises(ValueError):
        module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})


class Response(io.BytesIO):
    headers = {}
    def __init__(self, data, url=URL): super().__init__(data); self.url = url
    def geturl(self): return self.url


@pytest.mark.parametrize('body', [b'ab', b'abcd', b'bad'])
def test_bad_download_never_promoted(tmp_path, release, manifest, body):
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})
    with pytest.raises(ValueError):
        module('client').download_candidate(candidate, tmp_path, open_url=lambda request:Response(body))
    assert not (tmp_path/'installer.exe').exists() and not (tmp_path/'installer.part').exists()


def test_verified_download_is_promoted(tmp_path, release, manifest):
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})
    result = module('client').download_candidate(candidate, tmp_path, open_url=lambda request:Response(b'abc'))
    assert result.read_bytes() == b'abc'


def test_bad_redirect_never_read(tmp_path, release, manifest):
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})
    with pytest.raises(ValueError):
        module('client').download_candidate(candidate, tmp_path, open_url=lambda request:Response(b'abc','https://evil.example/x'))


def test_missing_build_metadata_is_not_a_ci_build(tmp_path):
    info = module('manifest').read_build_info(tmp_path)
    assert info['build_number'] == 0 and info['platform'] == 'source'


def test_arbitrary_cdn_candidate_is_rejected_before_network(tmp_path, release, manifest):
    from dataclasses import replace
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})
    candidate = replace(candidate, installer_url='https://objects.githubusercontent.com/unknown.exe')
    with pytest.raises(ValueError):
        module('client').download_candidate(candidate, tmp_path, open_url=lambda req:Response(b'abc'))


def test_lookup_skips_unrelated_latest_release(release, manifest):
    import json
    client = module('client')
    url = f'https://github.com/{REPO}/releases/download/{TAG}/google-prep-update.json'
    release['assets'].append({'name':'google-prep-update.json','state':'uploaded','browser_download_url':url})
    calls = []
    def open_response(req):
        calls.append(req.full_url)
        data = [{'tag_name':'another-product-v9'}, release] if req.full_url.startswith(client.API_ROOT) else manifest
        return Response(json.dumps(data).encode(), req.full_url)
    candidate = client.fetch_candidate({'version':'1.5.0','build_number':0}, open_url=open_response)
    assert candidate.tag == TAG and len(calls) == 2


def test_oversized_json_rejected():
    client = module('client')
    with pytest.raises(ValueError):
        client.get_json(client.API_ROOT, 10, lambda req:Response(b'x'*11, req.full_url))


def test_redirect_handler_rejects_other_hosts():
    import urllib.request
    client = module('client')
    req = urllib.request.Request(URL)
    with pytest.raises(ValueError):
        client.SafeRedirect().redirect_request(req, None, 302, 'redirect', {}, 'https://evil.example/x')


def test_rate_limit_propagates_retry_delay(monkeypatch):
    import urllib.error
    client = module('client')
    class Opener:
        def open(self, req, timeout):
            raise urllib.error.HTTPError(req.full_url, 429, 'rate limited', {'Retry-After':'7200'}, None)
    monkeypatch.setattr(client.urllib.request, 'build_opener', lambda *args:Opener())
    with pytest.raises(client.RateLimited) as raised:
        client.open_url(client.request(client.API_ROOT))
    assert raised.value.retry_after_seconds == 7200


def test_interrupted_stream_keeps_previous_verified_file(tmp_path, release, manifest):
    candidate = module('manifest').validate_candidate(release, manifest, {'version':'1.5.0','build_number':0})
    (tmp_path/'installer.exe').write_bytes(b'previous')
    class Interrupted(Response):
        def read(self, size): raise OSError('disconnected')
    with pytest.raises(OSError):
        module('client').download_candidate(candidate, tmp_path, open_url=lambda req:Interrupted(b''))
    assert (tmp_path/'installer.exe').read_bytes() == b'previous'
    assert not (tmp_path/'installer.part').exists()

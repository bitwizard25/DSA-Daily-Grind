"""CI-only producer. Never shipped as part of the app; requires job-local GH_TOKEN."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from google_prep import __version__
from google_prep.updates.manifest import (APP_ID, REPOSITORY, PLATFORM, INSTALLER, MANIFEST,
                                         MAX_INSTALLER_BYTES, TAG_PATTERN, Candidate, version_key)


def make_metadata(platform, env):
    build = {'schema':1, 'app_id':APP_ID, 'platform':platform, 'version':__version__,
             'build_number':int(env['GITHUB_RUN_NUMBER']), 'commit':env['GITHUB_SHA']}
    validate_build(build, windows_only=False)
    return build


def validate_build(build, windows_only=True):
    if not isinstance(build, dict) or type(build.get('schema')) is not int or build['schema'] != 1:
        raise ValueError('Invalid build metadata')
    if build.get('app_id') != APP_ID or not re.fullmatch('[0-9a-f]{40}', str(build.get('commit',''))):
        raise ValueError('Invalid app or commit')
    version_key(build['version'],build['build_number'])
    if build['build_number'] <= 0: raise ValueError('Build must come from CI')
    if windows_only and build.get('platform') != PLATFORM: raise ValueError('Windows installer required')
    if build.get('platform') not in ('windows-x64','linux','macos'): raise ValueError('Unknown platform')


def make_manifest(build, installer):
    validate_build(build)
    size = installer.stat().st_size
    if not 0 < size <= MAX_INSTALLER_BYTES: raise ValueError('Invalid installer size')
    digest = hashlib.sha256()
    with installer.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    tag = f"google-prep-v{build['version']}-b{build['build_number']}"
    return Candidate(build['version'],build['build_number'],build['commit'],tag,
                     f'https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}',
                     size,digest.hexdigest()).manifest()


def published_key(release):
    if release.get('draft') or release.get('prerelease'): return None
    match = TAG_PATTERN.fullmatch(str(release.get('tag_name','')))
    if not match: return None
    try: return version_key(match[1],int(match[2]))
    except ValueError: return None


def publish_update(github, directory, ref):
    if ref != 'refs/heads/main': raise ValueError('Only successful main builds can publish')
    build = json.loads((directory/'build_info.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/MANIFEST).read_text(encoding='utf-8'))
    if make_manifest(build,directory/INSTALLER) != manifest:
        raise ValueError('Uploaded build, manifest and installer do not match')
    tag, key = manifest['tag'],version_key(build['version'],build['build_number'])
    releases = github.list_releases()
    if any(r.get('tag_name') == tag and not r.get('draft') for r in releases): return 'already-published'
    if any((published_key(r) or (0,0,0,0)) > key for r in releases): return 'superseded'
    if any(r.get('tag_name') == tag and r.get('draft') for r in releases): github.delete_draft(tag)
    github.create_draft(tag,build['commit'])
    github.upload(tag,[directory/INSTALLER,directory/MANIFEST,directory/'build_info.json'])
    if any((published_key(r) or (0,0,0,0)) > key for r in github.list_releases()): return 'superseded'
    github.publish(tag)
    return 'published'


class GitHub:
    def run(self, args):
        return subprocess.run(['gh',*args],check=True,text=True,capture_output=True).stdout

    def list_releases(self):
        pages = json.loads(self.run(['api',f'repos/{REPOSITORY}/releases?per_page=100','--paginate','--slurp']))
        return [release for page in pages for release in page]

    def delete_draft(self, tag):
        release = json.loads(self.run(['api',f'repos/{REPOSITORY}/releases/tags/{tag}']))
        if not release.get('draft'): raise ValueError('Refusing to remove published release')
        self.run(['api','--method','DELETE',f"repos/{REPOSITORY}/releases/{release['id']}"])

    def create_draft(self, tag, commit):
        endpoint = f'repos/{REPOSITORY}/git/ref/tags/{tag}'
        result = subprocess.run(['gh','api',endpoint],capture_output=True,text=True)
        if result.returncode:
            if '404' not in result.stderr: raise ValueError('Unable to verify release tag')
            self.run(['api','--method','POST',f'repos/{REPOSITORY}/git/refs',
                      '-f',f'ref=refs/tags/{tag}','-f',f'sha={commit}'])
        elif json.loads(result.stdout)['object'] != {'sha':commit,'type':'commit','url':f'https://api.github.com/repos/{REPOSITORY}/git/commits/{commit}'}:
            data = json.loads(result.stdout)['object']
            if data.get('sha') != commit or data.get('type') != 'commit': raise ValueError('Existing tag targets another commit')
        with tempfile.TemporaryDirectory() as temporary:
            notes=Path(temporary)/'notes.md'
            notes.write_text('Tested Windows update. Your study progress is preserved. Install manually once to enable future automatic updates.\n',encoding='utf-8')
            self.run(['release','create',tag,'--repo',REPOSITORY,'--draft','--verify-tag',
                      '--title',tag,'--notes-file',str(notes)])

    def upload(self, tag, paths):
        self.run(['release','upload',tag,*map(str,paths),'--repo',REPOSITORY])
        release=json.loads(self.run(['api',f'repos/{REPOSITORY}/releases/tags/{tag}']))
        assets={a['name']:a for a in release['assets']}
        for path in paths:
            asset=assets.get(path.name)
            if not asset or asset['state'] != 'uploaded' or asset['size'] != path.stat().st_size:
                raise ValueError('Release upload incomplete; draft retained')

    def publish(self, tag):
        self.run(['release','edit',tag,'--repo',REPOSITORY,'--draft=false','--latest'])


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    metadata=commands.add_parser('metadata');metadata.add_argument('--platform',required=True)
    metadata.add_argument('--output',type=Path,default=ROOT/'data/build_info.json')
    manifest=commands.add_parser('manifest');manifest.add_argument('--directory',type=Path,default=ROOT/'dist')
    publish=commands.add_parser('publish');publish.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.command == 'metadata':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(make_metadata(args.platform,os.environ),indent=2),encoding='utf-8')
    elif args.command == 'manifest':
        build=json.loads((ROOT/'data/build_info.json').read_text(encoding='utf-8'))
        (args.directory/'build_info.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
        (args.directory/MANIFEST).write_text(json.dumps(make_manifest(build,args.directory/INSTALLER),indent=2),encoding='utf-8')
    else:
        build=json.loads((args.directory/'build_info.json').read_text(encoding='utf-8'))
        if build['commit'] != os.environ['GITHUB_SHA'] or build['build_number'] != int(os.environ['GITHUB_RUN_NUMBER']):
            raise ValueError('Artifact does not belong to this workflow run')
        print(publish_update(GitHub(),args.directory,os.environ['GITHUB_REF']))


if __name__ == '__main__': main()

"""Assemble the website; --rebuild compiles the pinned browser engine first."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PORT = Path(__file__).resolve().parent
WORK = ROOT / '.build'
OUT = WORK / 'engine'
SITE = WORK / 'site'
BUNDLE = WORK / 'browser-engine.tar.gz'
PIN = json.loads((PORT / 'source.json').read_text())
GENERATED = ['wand.js', 'wand.wasm', 'easy.nhdat', 'medium.nhdat', 'hard.nhdat',
             'pro.nhdat', 'levels.json', 'tiles.json', 'tiles.png', 'source.tar.gz']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    names = ['CMakeLists.txt', 'bridge.c', 'fiber.c', 'fcontext/fcontext.h', 'build-assets.py', 'build.py']
    names += [str(p.relative_to(PORT)) for p in sorted(PORT.glob('levels/*.des'))]
    names += ['levels/metadata.json']
    return {name: sha256(PORT / name) for name in names}


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def patch(source, name, old, new):
    path = source / name
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f'Upstream source changed: {name}')
    notice = ('# ' if path.suffix == '.txt' else '/* ') + 'Modified 2026-10-17: WebAssembly build compatibility.'
    notice += '\n' if path.suffix == '.txt' else ' */\n'
    path.write_text(notice + text.replace(old, new))


def source_bundle(source):
    """Distribute the complete modified engine source beside its executable."""
    with tarfile.open(OUT / 'source.tar.gz', 'w:gz') as archive:
        def normal(info):
            if not info.name.startswith('wand-browser-source/nle/') and (
                any(p in {'.venv', 'runs', '__pycache__'} or p.endswith('.egg-info')
                    for p in Path(info.name).parts) or info.name.endswith(('.pyc', '.DS_Store'))):
                return None
            if info.name.endswith('/website/browser/source.json'):
                return None  # A bundle cannot contain its own release checksum.
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 1792195200  # release date, no local account metadata
            return info
        archive.add(source, arcname='wand-browser-source/nle', filter=normal)
        for name in ['website/browser', 'experiments', 'licenses', 'README.md', 'LICENSE']:
            archive.add(ROOT / name, arcname=f'wand-browser-source/{name}', filter=normal)
        upstream = json.dumps({k: v for k, v in PIN.items() if k != 'prebuilt'}, indent=2).encode() + b'\n'
        info = tarfile.TarInfo('wand-browser-source/website/browser/source.json')
        info.size = len(upstream)
        archive.addfile(info, io.BytesIO(upstream))


def rebuild():
    version = subprocess.check_output(['emcc', '--version'], text=True)
    if PIN['emscripten'] not in version.splitlines()[0]:
        raise RuntimeError(f"Activate Emscripten {PIN['emscripten']} first.")
    WORK.mkdir(exist_ok=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    downloaded = WORK / f"nle-{PIN['version']}.tar.gz"
    if not downloaded.exists():
        urllib.request.urlretrieve(PIN['url'], downloaded)
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != PIN['sha256']:
        raise RuntimeError('NLE archive checksum mismatch.')
    # These two disposable paths are owned by this script, never user data.
    source, build = WORK / 'nle-src', WORK / 'wasm-release'
    for folder in [source, build]:
        if folder.exists():
            shutil.rmtree(folder)
    source.mkdir()
    with tarfile.open(downloaded) as archive:
        for entry in archive.getmembers():
            entry.name = entry.name.partition('/')[2]
            if entry.name:
                archive.extract(entry, source, filter='data')
    shutil.copy2(PORT / 'CMakeLists.txt', source / 'CMakeLists.txt')
    patch(source, 'util/CMakeLists.txt', 'COMMAND $<TARGET_FILE:',
          'COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:')
    patch(source, 'dat/CMakeLists.txt', 'COMMAND LC_ALL=C $<TARGET_FILE:dlb>',
          'COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:dlb>')
    patch(source, 'src/nle.c', 'extern int FDECL(set_random,', 'extern void FDECL(set_random,')
    patch(source, 'include/wintty.h', 'E int FDECL(nle_xputs,', 'E void FDECL(nle_xputs,')
    run('emcmake', 'cmake', '-S', source, '-B', build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', f'-DWAND_PORT={PORT}')
    run('cmake', '--build', build, '-j', min(6, os.cpu_count() or 2))
    # CMake records the SDK's Node executable, including its required arguments.
    cache = (build / 'CMakeCache.txt').read_text()
    node = next(line.partition('=')[2].split(';')[0] for line in cache.splitlines()
                if line.startswith('CMAKE_CROSSCOMPILING_EMULATOR:'))
    run(sys.executable, PORT / 'build-assets.py', '--node', node, '--build', build)
    for name in ['wand.js', 'wand.wasm']:
        shutil.copy2(build / name, OUT / name)
        (OUT / name).chmod(0o644)
    source_bundle(source)
    manifest = dict(nle=PIN['version'], emscripten=PIN['emscripten'],
                    inputs=inputs(), sha256={name: sha256(OUT / name) for name in GENERATED})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(BUNDLE, 'w:gz') as archive:
        for name in GENERATED + ['manifest.json']:
            info = archive.gettarinfo(OUT / name, arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 1792281600  # release date, no local account metadata
            with (OUT / name).open('rb') as source:
                archive.addfile(info, source)
    print(f'Built {BUNDLE.name}; SHA-256: {sha256(BUNDLE)}')


def fetch():
    release = PIN['prebuilt']
    WORK.mkdir(exist_ok=True)
    cached = WORK / f"engine-{release['sha256']}.tar.gz"
    if not cached.exists() and BUNDLE.exists() and sha256(BUNDLE) == release['sha256']:
        shutil.copy2(BUNDLE, cached)
    if not cached.exists() or sha256(cached) != release['sha256']:
        temporary = cached.with_suffix('.download')
        try:
            urllib.request.urlretrieve(release['url'], temporary)
            if sha256(temporary) != release['sha256']:
                raise RuntimeError('Engine bundle checksum mismatch.')
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    # Extract only the known generated files into this script's disposable dir.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    with tarfile.open(cached) as archive:
        members = archive.getmembers()
        expected = set(GENERATED + ['manifest.json'])
        if len(members) != len(expected) or {m.name for m in members} != expected or not all(m.isfile() for m in members):
            raise RuntimeError('Unexpected engine bundle contents.')
        archive.extractall(OUT, filter='data')


def assemble():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['nle'] != PIN['version'] or manifest['emscripten'] != PIN['emscripten'] or manifest['inputs'] != inputs():
        raise RuntimeError('Engine sources changed. Rebuild and publish a new pinned bundle.')
    if set(manifest['sha256']) != set(GENERATED):
        raise RuntimeError('Incomplete engine manifest.')
    for name, checksum in manifest['sha256'].items():
        if sha256(OUT / name) != checksum:
            raise RuntimeError(f'Engine file checksum mismatch: {name}')
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / 'website', SITE, ignore=shutil.ignore_patterns('browser'))
    shutil.copytree(OUT, SITE / 'engine')
    for name in ['client.mjs', 'worker.js', 'runtime.mjs']:
        shutil.copy2(PORT / name, SITE / 'engine' / name)
    shutil.copytree(ROOT / 'licenses', SITE / 'licenses')
    shutil.copy2(ROOT / 'LICENSE', SITE / 'LICENSE')
    shutil.copy2(ROOT / 'README.md', SITE / 'README.md')
    print('Website ready: .build/site/ (serve this directory over HTTP)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Use Emscripten and the experiments environment')
    args = parser.parse_args()
    rebuild() if args.rebuild else fetch()
    assemble()


if __name__ == '__main__':
    main()

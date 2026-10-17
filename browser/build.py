"""Rebuild the browser engine from the verified upstream source archive."""
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

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / 'browser'
WORK = ROOT / '.build'
OUT = ROOT / 'docs/engine'
PIN = json.loads((PORT / 'source.json').read_text())


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
            if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.DS_Store')):
                return None
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 1792195200  # release date, no local account metadata
            return info
        archive.add(source, arcname='wand-browser-source/nle', filter=normal)
        for name in ['browser', 'licenses']:
            archive.add(ROOT / name, arcname=f'wand-browser-source/{name}', filter=normal)
        for name in ['runtime.mjs', 'client.mjs', 'worker.js']:
            archive.add(OUT / name, arcname=f'wand-browser-source/docs/engine/{name}', filter=normal)
        note = b'Full modified NLE 1.1.0 source; port changes dated 2026-10-17.\nSee browser/README.md and licenses/NetHack.txt.\nRebuild instructions: https://github.com/wojtke/minihack-wand-agent/tree/main/browser\n'
        info = tarfile.TarInfo('wand-browser-source/README.txt')
        info.size = len(note)
        archive.addfile(normal(info), io.BytesIO(note))


def main():
    version = subprocess.check_output(['emcc', '--version'], text=True)
    if PIN['emscripten'] not in version.splitlines()[0]:
        raise RuntimeError(f"Activate Emscripten {PIN['emscripten']} first.")
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
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
    names = ['wand.js', 'wand.wasm', 'runtime.mjs', 'worker.js', 'client.mjs', 'easy.nhdat', 'medium.nhdat', 'hard.nhdat',
             'pro.nhdat', 'levels.json', 'tiles.json', 'tiles.png', 'source.tar.gz']
    manifest = dict(nle=PIN['version'], emscripten=PIN['emscripten'],
                    sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in names})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Browser build complete: docs/engine/')


if __name__ == '__main__':
    main()

"""Run in the isolated venv: download only pinned FP8 + auxiliary assets."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys


def verify_fp8(cache):
    value = json.loads((Path(cache) / 'manifest.json').read_text())
    if value.get('precision') != 'fp8' or value.get('format') != 'freevideo-fp8-slim-v1':
        raise ValueError('VDN requires the FreeVideo FP8 slim cache; INT8/BF16/base-H3 caches are not accepted')
    return value


def install(root, offline=False):
    from huggingface_hub import hf_hub_download
    from freevideo_engine import prepared_model, sampling_assets
    root = Path(root)
    package = Path(prepared_model.__file__).parent
    models = root / 'models'
    selection = prepared_model.select((12, 0), models, scale_granularity='per_tensor')
    rows = [(row, Path(selection['directory']) / row['file']) for row in prepared_model.files(selection)]
    # Prepared FP8 already contains merged DiT and VDN branch: no BF16 60+ GB source download.
    for row in json.loads((package / 'model_files.json').read_text()):
        name = row['file']
        if name.startswith(('h3-base/vae/', 'h3-base/audio_vae/', 'h3-base/scheduler/',
                            'h3-base/audio_scheduler/', 'text_encoders/')):
            rows.append((row, models / name))
        elif row.get('role') == 'latent_upscaler':
            rows.append((row, models / 'latent_upscaler' / Path(name).name))
    # Include 8-step and 3-step tables for ALL reference modalities, including audio.
    tables = [t for t in sampling_assets.tables() if len(t['identity']['timesteps']) in (3, 8)]
    for row in sampling_assets.files(tables):
        rows.append((row, models / 'sampling-cache' / row['sampling_file']))
    ledger_path = root / 'asset-receipts.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    seen = set()
    for row, target in rows:
        if str(target) in seen:
            continue
        seen.add(str(target))
        expected = row.get('sha256') or row.get('download_sha256')
        if not expected:
            raise ValueError('Pinned asset has no checksum: ' + row['file'])
        if target.is_file():
            stat = target.stat()
            if ledger.get(str(target)) == [expected, stat.st_size, stat.st_mtime_ns] and stat.st_size == row['bytes']:
                continue
        print('Preparing ' + row['file'], flush=True)
        source = Path(hf_hub_download(repo_id=row['repo'], revision=row['revision'],
                                     filename=row['file'], local_files_only=offline))
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if source.stat().st_size != row['bytes'] or digest.hexdigest() != expected:
            raise ValueError('Asset checksum mismatch: ' + row['file'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError('Asset points to a different checkpoint: ' + str(target))
        elif target.exists():
            existing = hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    existing.update(block)
            if existing.hexdigest() != expected:
                raise ValueError('Refusing to replace mismatched asset: ' + str(target))
        else:
            try:
                os.link(source.resolve(), target)
            except OSError:
                shutil.copy2(source, target)
        stat = target.stat()
        ledger[str(target)] = [expected, stat.st_size, stat.st_mtime_ns]
        temp = ledger_path.with_suffix('.tmp')
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(ledger_path)
    # Shared AdaLN tables require per-file receipts, not just their tensor files.
    for table in tables:
        for row in table['files']:
            target = models / 'sampling-cache' / row['file']
            target.with_suffix('.json').write_text(json.dumps(row, indent=2))
    # Configs accompanying the prepared cache identify the merged VDN skeleton.
    config = Path(selection['directory']) / 'config'
    for folder in config.iterdir():
        for source in folder.rglob('*'):
            if not source.is_file():
                continue
            target = models / source.relative_to(config)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source.resolve(), target)
                except OSError:
                    shutil.copy2(source, target)
    verify_fp8(selection['cache'])
    model_paths = root / 'model_paths.yaml'
    model_paths.write_text(json.dumps({'cocreation_vdn': {'base_path': str(models), 'text_encoders': 'text_encoders'}}))
    installation = dict(python=str(root / 'venv/bin/python'), cache=selection['cache'],
        base=str(models / 'h3-base'), checkpoint=str(models / 'stage-dmd-step-250'),
        comfy_root=str(root / 'vendor/ComfyUI'), model_paths=str(model_paths),
        model_revision=selection['revision'], precision='fp8')
    (root / 'assets-ready.json').write_text(json.dumps(installation, indent=2))
    return installation


if __name__ == '__main__':
    install(sys.argv[1], '--offline' in sys.argv[2:])

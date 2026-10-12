"""Stage a fresh exact replay from frozen inputs without modifying the archive."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments'/'fta_exact_replay_20261012_170305'
CODE=('fta_exact_replay.py','fta_exact_replay_verify.py','fta_exact_replay_scoring.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New, empty output directory outside archived experiments')
    parser.add_argument('--prepare-only',action='store_true',help='CPU-only staging and input verification; no imports of model frameworks')
    args=parser.parse_args();out=args.output.expanduser().resolve()
    assert not out.is_relative_to(ROOT/'experiments'),'Do not write into the historical archive'
    assert not out.exists() or not any(out.iterdir()),'Output directory must be new or empty'
    out.mkdir(parents=True,exist_ok=True)
    for name in CODE:shutil.copyfile(SOURCE/name,out/name)
    for name in ('fta_exact_tasks.zip','native_inputs.jsonl','review_plan.json'):shutil.copyfile(SOURCE/name,out/name)
    shutil.copytree(SOURCE/'fta_exact_tasks',out/'fta_exact_tasks')
    sys.path.insert(0,str(out))
    spec=importlib.util.spec_from_file_location('public_exact_replay',out/'fta_exact_replay.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.PROJECT=ROOT/'experiments'
    original=json.loads((SOURCE/'run_manifest.json').read_text())
    for name,expected in original['code_sha256'].items():assert runner.sha(out/name)==expected
    preserved={'fta_experiment.py':{'sha256':runner.sha(runner.PROJECT/'fta_experiment.py'),
                                  'bytes':(runner.PROJECT/'fta_experiment.py').stat().st_size}}
    runner.json_new(out/'preserved_previous_runs.json',preserved)
    manifest=dict(original)
    manifest['version']='fta-exact-public-reproduction-v1'
    manifest['frozen_at']=datetime.now(timezone.utc).isoformat()
    manifest['preservation_manifest_sha256']=runner.sha(out/'preserved_previous_runs.json')
    manifest['public_reproduction']={'source_run_manifest_sha256':runner.sha(SOURCE/'run_manifest.json'),
        'source_archive':'experiments/fta_exact_replay_20261012_170305',
        'inherited_frozen_input_and_decoding_settings':True,'historical_answers_or_labels_copied':False,
        'preserved_files':list(preserved),
        'wrapper_sha256':runner.sha(Path(__file__).resolve()),
        'original_freeze_not_reexecuted':'It requires all30 original pilot artifacts including intentionally excluded heavy CSVs; this fresh manifest honestly inventories only the shared loader.'}
    runner.json_new(out/'run_manifest.json',manifest)
    runner.frozen()
    assert runner.sha(out/'fta_exact_tasks.zip')==original['bundle_zip_sha256']
    print(json.dumps({'prepared':str(out),'reports_planned':1200,'model_weights_loaded':False,
                      'historical_run_unchanged':True},sort_keys=True),flush=True)
    if args.prepare_only:return
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      USE_TORCH='1',USE_TF='0',USE_FLAX='0')
    expected=json.loads((SOURCE/'runtime_Qwen3-8B.json').read_text())
    for package,version in expected['packages'].items():
        assert importlib.metadata.version(package)==version,f'Published environment requires {package}=={version}'
    import torch
    assert torch.cuda.is_available(),'Inference requires an allocated CUDA GPU'
    assert 'V100' in torch.cuda.get_device_name(0),'Published numerical environment uses a V100; prepare-only works without GPU'
    for model,revision in runner.MODELS.items():
        assert runner.cached_path(model,revision).is_dir(),f'Missing local cached checkpoint: {model}@{revision}; no weights are downloaded'
    runner.run()
    runner.packets()
    print('Generation complete. Score the full blinded answers independently using the bundled rubric; final semantic scores are not assigned by a status parser.')


if __name__=='__main__':main()

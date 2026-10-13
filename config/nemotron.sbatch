#!/bin/bash
#SBATCH --account=amantzaris
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --constraint=h100
#SBATCH --time=2-00:00:00
#SBATCH --propagate=NONE
#SBATCH --job-name=fta_nemotron_full
#SBATCH --chdir=/lustre/fs1/home/sa746371/ICPRAM
#SBATCH --output=/lustre/fs1/home/sa746371/ICPRAM/results/nemotron3_super120b.log
#SBATCH --open-mode=append

set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false MALLOC_CONF=background_thread:false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HF_HUB_CACHE=/home/sa746371/.cache/huggingface/hub
unset PYTHONPATH PYTHONHOME
FTA_PHASE=startup
FTA_NEM_PYTHON=/lustre/fs1/home/sa746371/fta_nemotron_env/bin/python

fta_finish() {
    local fta_exit_code=$?
    trap - EXIT
    /home/sa746371/SAE-1/bin/python -B - "$fta_exit_code" "$FTA_PHASE" <<'STATUS'
import datetime, json, os, sys
from pathlib import Path
p = Path('results/nemotron_job_status.json')
record = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scheduler_job_id': os.environ['SLURM_JOB_ID'], 'exit_code': int(sys.argv[1]),
          'phase': sys.argv[2], 'status': 'completed' if sys.argv[1] == '0' else 'failed',
          'answers': 'results/answers.jsonl', 'generation_manifest': 'results/generation_nemotron3_super120b.json'}
t = p.with_suffix('.json.tmp'); t.write_text(json.dumps(record, indent=2) + '\n'); t.replace(p)
print('NEMOTRON_JOB_EXIT', json.dumps(record), flush=True)
STATUS
    exit "$fta_exit_code"
}
trap fta_finish EXIT

source /etc/profile.d/lmod.sh
module load gcc/gcc-11.2.0 cuda/cuda-12.4.0
echo "NEMOTRON_FULL_REPLAY job=$SLURM_JOB_ID node=$SLURMD_NODENAME"

FTA_PHASE=frozen_files_and_runtime
"$FTA_NEM_PYTHON" -B -u - <<'VERIFY'
import hashlib, json
from pathlib import Path
from utils.dataset import load_and_verify_inputs
from utils.models import model_spec, verify_snapshot
from utils.runtime import compute_preflight, nemotron_kernel_preflight, packages
submission = json.loads(Path('results/nemotron_submission.json').read_text())
for name, expected in submission['frozen_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'Frozen file changed: ' + name
inputs = load_and_verify_inputs(Path.cwd())
assert len(inputs) == 300
guard = json.loads(Path('results/nemotron_kernel_guard.json').read_text())
assert guard['guard_passed'] and guard['builtin_model_fast_path_available']
assert guard['scheduler_job_id'] == submission['guard_job_id']
assert guard['native_input_prerequisite']['native_prompt_and_token_hashes_reverified'] == 300
assert len(guard['technical_cuda_kernel_calls']) == 5
assert packages() == guard['packages'], 'Runtime packages changed after the successful guard'
spec = model_spec(Path.cwd(), 'nemotron3_super120b')
hardware = compute_preflight(spec)
snapshot = verify_snapshot(spec, require_weights=True)
assert len(snapshot['weight_files']) == 50
assert snapshot['weight_bytes'] == 247227650480
nemotron_kernel_preflight()
print('NEMOTRON_PREFLIGHT_PASSED', json.dumps({'gpus': hardware['gpus'],
      'native_inputs': 300, 'official_weight_shards': 50}), flush=True)
VERIFY

FTA_PHASE=generation
"$FTA_NEM_PYTHON" -B -u -m pipelines.generate run --model nemotron3_super120b

FTA_PHASE=validate_saved_reports
"$FTA_NEM_PYTHON" -B -u - <<'VALIDATE'
import json
from pathlib import Path
from utils.dataset import load_and_verify_inputs
from utils.io import read_jsonl
from utils.models import model_spec
from pipelines.generate import response_id
spec = model_spec(Path.cwd(), 'nemotron3_super120b')
expected = {response_id(spec, row) for row in load_and_verify_inputs(Path.cwd())}
rows = [row for row in read_jsonl('results/answers.jsonl') if row['model'] == spec['repo_id']]
assert len(rows) == 300 and {row['response_id'] for row in rows} == expected
assert all(row['repeat'] == 0 and row['technical_error'] is None for row in rows)
assert all(row['revision'] == spec['revision'] for row in rows)
print('NEMOTRON_300_REPORTS_SAVED', json.dumps({'reports': len(rows),
      'truncated': sum(row['truncated'] for row in rows), 'scoring_performed_by_job': False}), flush=True)
VALIDATE
FTA_PHASE=complete

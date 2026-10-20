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
#SBATCH --no-requeue
#SBATCH --job-name=fta_nemotron_retry
#SBATCH --chdir=/lustre/fs1/home/sa746371/ICPRAM
#SBATCH --output=/lustre/fs1/home/sa746371/fta_nemotron_retry/results/full-%j.log

# One wrapper for a one-H100 kernel guard and the dependent eight-H100 replay.
# Historical code, scientific inputs/settings and results remain unchanged.
set -euo pipefail
FTA_RETRY_MODE=${1:?Specify guard or run}
case "$FTA_RETRY_MODE" in guard|run) ;; *) exit 2 ;; esac
export FTA_RETRY_MODE
export FTA_RETRY_ROOT=/lustre/fs1/home/sa746371/fta_nemotron_retry
export FTA_PROJECT_ROOT=/lustre/fs1/home/sa746371/ICPRAM
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false MALLOC_CONF=background_thread:false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HF_HUB_CACHE=/home/sa746371/.cache/huggingface/hub
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="$FTA_PROJECT_ROOT"
FTA_NEM_PYTHON=/lustre/fs1/home/sa746371/fta_nemotron_env/bin/python
FTA_PHASE=startup

fta_finish() {
    local fta_exit_code=$?
    trap - EXIT
    /home/sa746371/SAE-1/bin/python -B - "$fta_exit_code" "$FTA_PHASE" <<'STATUS'
import datetime, json, os, sys
from pathlib import Path
root = Path(os.environ['FTA_RETRY_ROOT'])
mode = os.environ['FTA_RETRY_MODE']
p = root / 'results' / (mode + '_job_status.json')
r = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'scheduler_job_id': os.environ['SLURM_JOB_ID'], 'mode': mode,
     'exit_code': int(sys.argv[1]), 'phase': sys.argv[2],
     'status': 'completed' if sys.argv[1] == '0' else 'failed',
     'answers': str(root / 'results/answers.jsonl'), 'automatic_requeue': False}
t = p.with_suffix('.tmp'); t.write_text(json.dumps(r, indent=2) + '\n'); t.replace(p)
print('NEMOTRON_RETRY_EXIT', json.dumps(r), flush=True)
STATUS
    exit "$fta_exit_code"
}
trap fta_finish EXIT

source /etc/profile.d/lmod.sh
module load gcc/gcc-11.2.0 cuda/cuda-12.4.0
cd "$FTA_PROJECT_ROOT"

FTA_PHASE=verify_frozen_files_and_installed_wheel
"$FTA_NEM_PYTHON" -B -u - <<'VERIFY'
import hashlib, importlib.metadata as md, importlib.util, json, os, tarfile, zipfile
from pathlib import Path
project = Path(os.environ['FTA_PROJECT_ROOT'])
retry = Path(os.environ['FTA_RETRY_ROOT'])
submission = json.loads((project / 'results/nemotron_retry_submission.json').read_text())
for name, expected in submission['frozen_files'].items():
    assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
assert os.environ['SLURM_JOB_ID'] == submission[os.environ['FTA_RETRY_MODE'] + '_job_id']
versions = {'mamba-ssm': '2.3.1', 'causal-conv1d': '1.5.0.post8',
            'torch': '2.6.0+cu124', 'triton': '3.2.0', 'transformers': '5.14.1'}
for name, version in versions.items():
    assert md.version(name) == version, (name, md.version(name))
env = Path('/lustre/fs1/home/sa746371/fta_nemotron_env').resolve()
distribution = md.distribution('mamba-ssm')
site = Path(distribution.locate_file('')).resolve()
assert site.is_relative_to(env), str(site)
module = Path(importlib.util.find_spec('mamba_ssm').origin).resolve()
assert module == site / 'mamba_ssm/__init__.py', str(module)
base = Path('/lustre/fs1/home/sa746371/fta_project_archive/runtime_build')
wheel = base / 'mamba_2_3_1_resumed_wheels/mamba_ssm-2.3.1-cp311-cp311-linux_x86_64.whl'
archive = base / 'mamba_ssm-2.3.1.tar.gz'
def digest_stream(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()
with archive.open('rb') as f: archive_sha = digest_stream(f)
with wheel.open('rb') as f: wheel_sha = digest_stream(f)
assert archive_sha == '4d529477ad94753962216d583fc8f1c127c717b7d7c875d6bbb9376366d0d761'
assert wheel_sha == submission['installed_wheel_sha256']
checked = []
with zipfile.ZipFile(wheel) as z, tarfile.open(archive) as source:
    for entry in z.infolist():
        name = entry.filename
        if entry.is_dir() or name.endswith('.dist-info/RECORD'): continue
        path = (site / name).resolve()
        assert path.is_relative_to(site) and path.is_file(), name
        with z.open(entry) as f: expected = digest_stream(f)
        with path.open('rb') as f: actual = digest_stream(f)
        assert actual == expected, 'Installed wheel mismatch: ' + name
        if name.startswith('mamba_ssm/') and name.endswith('.py'):
            with source.extractfile('mamba_ssm-2.3.1/' + name) as f:
                assert digest_stream(f) == expected, 'Official source mismatch: ' + name
        checked.append({'file': name, 'sha256': expected})
record = {'scheduler_job_id': os.environ['SLURM_JOB_ID'], 'versions': versions,
          'installed_distribution_root': str(site), 'module_path': str(module),
          'wheel_sha256': wheel_sha, 'official_source_archive_sha256': archive_sha,
          'installed_files_verified_against_wheel': checked,
          'official_python_sources_unchanged': True, 'checkpoint_loaded': False}
p = retry / 'results' / (os.environ['FTA_RETRY_MODE'] + '_installation_check.json')
p.write_text(json.dumps(record, indent=2) + '\n')
print('INSTALLED_WHEEL_VERIFIED', len(checked), str(module), flush=True)
VERIFY

if [[ "$FTA_RETRY_MODE" == guard ]]; then
    FTA_PHASE=native_templates_and_cuda_kernel_guard
    "$FTA_NEM_PYTHON" -B -u - <<'GUARD'
import json, os, traceback
from pathlib import Path
import torch
from utils.dataset import load_and_verify_inputs
from utils.models import load_tokenizer, model_spec, native_input
from utils.runtime import nemotron_kernel_preflight, packages, utc_now
project = Path(os.environ['FTA_PROJECT_ROOT']); retry = Path(os.environ['FTA_RETRY_ROOT'])
record = {'checked_at': utc_now(), 'scheduler_job_id': os.environ['SLURM_JOB_ID'],
          'packages': packages(), 'checkpoint_loaded': False, 'model_forward_performed': False,
          'reports_generated': 0, 'guard_passed': False}
try:
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9
    inputs = load_and_verify_inputs(project)
    frozen = json.loads((project / 'results/preflight_nemotron3_super120b.json').read_text())
    checks = {(r['scenario_id'], r['condition']): r for r in frozen['native_input_checks']}
    spec = model_spec(project, 'nemotron3_super120b'); tokenizer = load_tokenizer(spec)
    for row in inputs:
        _, actual = native_input(tokenizer, spec, row)
        saved = checks[(row['scenario_id'], row['condition'])]
        for field in ('logical_input_sha256', 'native_prompt_sha256',
                      'native_input_token_ids_sha256', 'input_length'):
            assert actual[field] == saved[field], (row['scenario_id'], row['condition'], field)
    record['native_input_prerequisite'] = {'native_prompt_and_token_hashes_reverified': len(inputs),
                                         'templates_modified': False}
    record['native_kernel_result'] = nemotron_kernel_preflight()
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    calls = []
    def validate(name, value, shape):
        torch.cuda.synchronize()
        assert tuple(value.shape) == tuple(shape), name
        assert torch.isfinite(value).all().item(), name + ' produced nonfinite values'
        calls.append({'function': name, 'output_shape': list(value.shape),
                      'output_dtype': str(value.dtype), 'finite': True})
    with torch.inference_mode():
        device = 'cuda'; dtype = torch.bfloat16
        conv_x = torch.full((1, 16, 256), .125, device=device, dtype=dtype)
        conv_w = torch.full((16, 4), .125, device=device, dtype=dtype)
        bias = torch.zeros(16, device=device, dtype=dtype)
        validate('causal_conv1d_fn', causal_conv1d_fn(conv_x, conv_w, bias=bias, activation='silu'), (1, 16, 256))
        conv_state = conv_x[:, :, -4:].contiguous().clone()
        step = causal_conv1d_update(torch.full((1, 16), .125, device=device, dtype=dtype),
                                   conv_state, conv_w, bias, 'silu')
        validate('causal_conv1d_update', step, (1, 16))
        assert torch.isfinite(conv_state).all().item()
        batch, length, heads, head_dim, state_dim, groups = 1, 256, 4, 64, 16, 1
        x = torch.full((batch, length, heads, head_dim), .125, device=device, dtype=dtype)
        dt = torch.full((batch, length, heads), .1, device=device, dtype=dtype)
        A = -torch.ones(heads, device=device, dtype=torch.float32)
        B = torch.full((batch, length, groups, state_dim), .125, device=device, dtype=dtype)
        C = B.clone(); D = torch.ones(heads, device=device, dtype=torch.float32)
        dt_bias = torch.zeros(heads, device=device, dtype=torch.float32)
        y, state = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=128, D=D,
                       dt_bias=dt_bias, dt_softplus=True, return_final_states=True)
        validate('mamba_chunk_scan_combined', y, (batch, length, heads, head_dim))
        validate('mamba_chunk_scan_combined.final_state', state, (batch, heads, head_dim, state_dim))
        state = state.to(dtype=dtype).contiguous().clone()
        y = selective_state_update(state, x[:, 0].contiguous(),
              dt[:, 0].float().unsqueeze(-1).expand(batch, heads, head_dim),
              A[:, None, None].expand(heads, head_dim, state_dim),
              B[:, 0].contiguous(), C[:, 0].contiguous(),
              D=D[:, None].expand(heads, head_dim),
              dt_bias=dt_bias[:, None].expand(heads, head_dim), dt_softplus=True)
        validate('selective_state_update', y, (batch, heads, head_dim))
        assert torch.isfinite(state).all().item()
    # Transformers sets its global fast-path flag in the mixer constructor,
    # NOT at module import. Instantiate on meta: no checkpoint or forward.
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as nh
    from utils.models import cached_snapshot
    config = AutoConfig.from_pretrained(str(cached_snapshot(spec)), local_files_only=True,
                                       trust_remote_code=False)
    layer = next(i for i, name in enumerate(config.layer_types) if name == 'linear_attention')
    with torch.device('meta'):
        mixer = nh.NemotronHMamba2Mixer(config, layer_idx=layer)
    assert all(p.is_meta for p in mixer.parameters())
    assert nh.is_fast_path_available, 'Native bindings unavailable after mixer initialization'
    del mixer
    import mamba_ssm, causal_conv1d, transformers
    assert Path(mamba_ssm.__file__).resolve().is_relative_to(Path('/lustre/fs1/home/sa746371/fta_nemotron_env').resolve())
    record.update(guard_passed=True, status='native_fast_kernels_ready',
                  device=torch.cuda.get_device_name(0), technical_cuda_kernel_calls=calls,
                  technical_runtime_test={'distinct_kernel_functions_executed': 4,
                      'finite_tensor_outputs_checked': 5, 'prefill_length': 256, 'chunk_size': 128,
                      'split_conv1d_scan': 'callability only; fused training path not executed',
                      'checkpoint_or_task_forward': False},
                  builtin_model_fast_path_available=True, meta_mixer_initialization=True,
                  module_paths={'mamba_ssm': mamba_ssm.__file__, 'causal_conv1d': causal_conv1d.__file__,
                                'torch': torch.__file__, 'transformers': transformers.__file__})
    print('NEMOTRON_RETRY_GUARD_PASSED', json.dumps(record), flush=True)
except Exception as error:
    record.update(status='blocked', error_type=type(error).__name__, error=str(error))
    traceback.print_exc()
    raise
finally:
    p = retry / 'results/kernel_guard.json'
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(record, indent=2) + '\n'); t.replace(p)
GUARD
else
    FTA_PHASE=allocated_hardware_and_guard
    "$FTA_NEM_PYTHON" -B -u - <<'PREFLIGHT'
import json, os
from pathlib import Path
from utils.models import model_spec, verify_snapshot
from utils.runtime import compute_preflight, nemotron_kernel_preflight, packages
project = Path(os.environ['FTA_PROJECT_ROOT']); retry = Path(os.environ['FTA_RETRY_ROOT'])
submission = json.loads((project / 'results/nemotron_retry_submission.json').read_text())
guard = json.loads((retry / 'results/kernel_guard.json').read_text())
assert guard['guard_passed'] and guard['builtin_model_fast_path_available']
assert guard['scheduler_job_id'] == submission['guard_job_id']
assert guard['native_input_prerequisite']['native_prompt_and_token_hashes_reverified'] == 300
assert len(guard['technical_cuda_kernel_calls']) == 5
assert packages() == guard['packages'], 'Packages changed after the guard'
spec = model_spec(project, 'nemotron3_super120b')
hardware = compute_preflight(spec)
snapshot = verify_snapshot(spec, require_weights=True)
assert len(snapshot['weight_files']) == 50 and snapshot['weight_bytes'] == 247227650480
nemotron_kernel_preflight()
print('NEMOTRON_RETRY_FULL_PREFLIGHT_PASSED', json.dumps(hardware), flush=True)
PREFLIGHT
    FTA_PHASE=generation
    "$FTA_NEM_PYTHON" -B -u -m pipelines.generate run --model nemotron3_super120b --root "$FTA_RETRY_ROOT"
    FTA_PHASE=validate_saved_reports
    "$FTA_NEM_PYTHON" -B -u - <<'VALIDATE'
import json, os
from pathlib import Path
from utils.dataset import load_and_verify_inputs
from utils.io import read_jsonl
from utils.models import model_spec
from pipelines.generate import response_id
root = Path(os.environ['FTA_RETRY_ROOT']); spec = model_spec(root, 'nemotron3_super120b')
expected = {response_id(spec, r) for r in load_and_verify_inputs(root)}
rows = read_jsonl(root / 'results/answers.jsonl')
assert len(rows) == 300 and {r['response_id'] for r in rows} == expected
assert all(r['repeat'] == 0 and r['technical_error'] is None and r['revision'] == spec['revision'] for r in rows)
print('NEMOTRON_300_REPORTS_SAVED', json.dumps({'reports': len(rows),
      'truncated': sum(r['truncated'] for r in rows), 'scoring_performed': False}), flush=True)
VALIDATE
fi
FTA_PHASE=complete

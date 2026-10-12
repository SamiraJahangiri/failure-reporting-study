"""Additional read-only integrity checks; no model calls or input changes."""
import collections
import json
from pathlib import Path
import fta_exact_replay as run
import fta_exact_replay_verify as verify


def validate(require_complete=False):
    p=run.ROOT
    m=run.frozen()
    assert run.sha(p/'fta_exact_tasks.zip')==m['bundle_zip_sha256']
    assert run.sha(p/'preserved_previous_runs.json')==m['preservation_manifest_sha256']
    current=verify.report_manifest(verify.verify_extracted(run.BUNDLE))
    assert current['manifest_sha256']==m['source_verification']['manifest_sha256']
    assert current['verified_files']==m['source_verification']['verified_files']
    native={(r['model'],r['scenario_id'],r['condition']):r for r in run.read_lines(p/'native_inputs.jsonl')}
    assert len(native)==600
    assert run.sha(run.PROJECT/'fta_experiment.py')==m['decoding']['pilot_source_sha256']
    runtime_hashes={model:run.sha(p/('runtime_'+model.split('/')[-1]+'.json'))
                    for model in run.MODELS if (p/('runtime_'+model.split('/')[-1]+'.json')).exists()}
    rows=run.read_lines(p/'raw_responses.jsonl')
    seen=set();pairs=collections.defaultdict(list)
    for r in rows:
        assert r['response_id'] not in seen,'Duplicate saved response ID'
        seen.add(r['response_id'])
        expected=native[(r['model'],r['scenario_id'],r['condition'])]
        assert r['native_prompt_sha256']==expected['rendered_prompt_sha256']
        assert r['logical_input_sha256']==expected['logical_input_sha256']
        assert r['run_manifest_sha256']==run.sha(p/'run_manifest.json')
        assert r['runtime_manifest_sha256']==runtime_hashes[r['model']]
        assert isinstance(r['response'],str) and type(r['truncated']) is bool
        assert r['finish_reason'] in ('eos','length')
        assert r['truncated']==(r['finish_reason']=='length')
        if r['truncated']:assert r['token_count']==160
        assert r['max_new_tokens']==160 and r['repeat'] in (0,1)
        assert r['seed']==run.SEEDS[r['repeat']]
        assert r['revision']==run.MODELS[r['model']]
        assert r['response_id']=='|'.join((r['model'],r['scenario_id'],r['condition'],str(r['repeat'])))
        assert r['token_count']<=160
        pairs[(r['model'],r['scenario_id'],r['condition'])].append(r)
    if require_complete:
        assert len(rows)==1200 and len(pairs)==600 and all(len(v)==2 for v in pairs.values())
    return {'reports':len(rows),'model_reports':dict(collections.Counter(r['model'] for r in rows)),
            'truncated':sum(r['truncated'] for r in rows),'empty':sum(not r['response'].strip() for r in rows),
            'duplicate_repetition_pairs':sum(len(v)==2 and v[0]['response']==v[1]['response'] for v in pairs.values()),
            'input_and_preservation_hashes_unchanged':True}


if __name__=='__main__':print(json.dumps(validate(),sort_keys=True))

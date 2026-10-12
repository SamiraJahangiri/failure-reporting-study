"""Exact released FTA logical-input replay; no prompt reconstruction or edits."""
import argparse
import collections
import contextlib
import csv
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

import fta_exact_replay_verify as verify
import fta_exact_replay_scoring as scoring

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BUNDLE = ROOT / 'fta_exact_tasks'
MODELS = {'Qwen/Qwen3-8B':'b968826d9c46dd6066d109eabc6255188de91218',
          'google/gemma-2-9b-it':'11c9b309abf73637e4b6f9a3fa1e92e615547819'}
SEEDS = (20261012, 20261013)
MAX_TOKENS = 160
BATCH = 4
CODE = ('fta_exact_replay.py','fta_exact_replay_verify.py','fta_exact_replay_scoring.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()


def json_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')


def read_lines(path):
    if not Path(path).exists():return []
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def write_lines_new(path,rows):
    with Path(path).open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')


def log(event,**fields):
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    line=now+' '+event+' '+json.dumps(fields,sort_keys=True)
    with (ROOT/'replay.log').open('a') as f:f.write(line+'\n');f.flush();os.fsync(f.fileno())
    print(line,flush=True)


def check_preserved():
    old=json.loads((ROOT/'preserved_previous_runs.json').read_text())
    for name,value in old.items():
        assert sha(PROJECT/name)==value['sha256'],'Previous artifact changed: '+name
    return len(old)


def cached_path(model,revision):
    return Path.home()/'.cache/huggingface/hub'/('models--'+model.replace('/','--'))/'snapshots'/revision


def freeze():
    bundle_check=verify.verify_extracted(BUNDLE)
    zip_check=verify.verify_zip(ROOT/'fta_exact_tasks.zip')
    logical=read_lines(BUNDLE/'model_inputs.jsonl')
    source=read_lines(BUNDLE/'data/confirmatory_scenarios.jsonl')
    specs=scoring.make_response_specs(logical,list(MODELS),2)
    plan=scoring.freeze_review_plan(specs,source,seed=20261012)
    if (ROOT/'review_plan.json').exists():
        previous=json.loads((ROOT/'review_plan.json').read_text())
        assert previous['sample_response_ids']==plan['sample_response_ids']
        assert previous['specifications_sha256']==plan['specifications_sha256']
        plan=previous
    else:json_new(ROOT/'review_plan.json',plan)
    # Direct cached tokenizer classes avoid importing unrelated generation
    # backends during this CPU-only preflight. Runtime uses the pilot loader.
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
    native=[]; template_info={}
    for model,revision in MODELS.items():
        path=cached_path(model,revision)
        tokenizer_class=Qwen2Tokenizer if model.startswith('Qwen/') else GemmaTokenizer
        tokenizer=tokenizer_class.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        kwargs={'tokenize':False,'add_generation_prompt':True}
        if model.startswith('Qwen/'):kwargs['enable_thinking']=False
        first=logical[0]
        originals=[{'role':'system','content':first['system']},{'role':'user','content':first['user']}]
        rejection=None
        try:tokenizer.apply_chat_template(originals,**kwargs); mode='native_system_user'
        except Exception as error:
            rejection=str(error)
            if model!='google/gemma-2-9b-it' or 'system' not in rejection.lower():raise
            mode='exact_policy_prepended_to_first_user'
        for item in logical:
            original=[{'role':'system','content':item['system']},{'role':'user','content':item['user']}]
            effective=original if mode=='native_system_user' else [{'role':'user','content':item['system']+'\n\n'+item['user']}]
            rendered=tokenizer.apply_chat_template(effective,**kwargs)
            assert item['system'] in rendered and item['user'] in rendered
            native.append({'model':model,'revision':revision,'scenario_id':item['scenario_id'],'condition':item['condition'],
                           'logical_input_sha256':scoring.canonical_hash(item),'original_messages':original,
                           'effective_messages':effective,'rendered_prompt':rendered,'rendered_prompt_sha256':text_sha(rendered),
                           'template_mode':mode})
        template_info[model]={'revision':revision,'chat_template':tokenizer.chat_template,
                             'tokenizer_class':type(tokenizer).__name__,
                             'chat_template_sha256':scoring.canonical_hash(tokenizer.chat_template),
                             'native_system_role_supported':mode=='native_system_user','system_role_rejection':rejection,
                             'adapter':None if mode=='native_system_user' else "exact system + '\\n\\n' + exact user in a single native user turn",
                             'native_template_kwargs':kwargs,'tokenizer_config_sha256':sha(path/'tokenizer_config.json')}
    write_lines_new(ROOT/'native_inputs.jsonl',native)
    manifest={'version':'fta-exact-replay-v1','frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_verification':verify.report_manifest(bundle_check),'zip_verification':verify.report_manifest(zip_check),
              'bundle_zip_sha256':sha(ROOT/'fta_exact_tasks.zip'),
              'supplied_inputs_sha256':sha(BUNDLE/'model_inputs.jsonl'),'native_inputs_sha256':sha(ROOT/'native_inputs.jsonl'),
              'models':MODELS,'native_templates':template_info,'planned_reports':1200,'tasks':100,'policies':3,'repetitions':2,
              'decoding':{'pilot_source_sha256':sha(PROJECT/'fta_experiment.py'),'pilot_setting':'ordinary full-report decoding',
                          'do_sample':False,'num_beams':1,'max_new_tokens':MAX_TOKENS,'batch_size':BATCH,
                          'seeds':list(SEEDS),'qwen_enable_thinking':False,'attention_implementation':'eager',
                          'v100_dtype':'float16','repetition_penalty':1.0,'length_penalty':1.0,'use_cache':True,
                          'assistant_prefill':'','constrained_decoding':False,'sampling_repetitions_independent':False},
              'code_sha256':{name:sha(ROOT/name) for name in CODE},'review_plan_sha256':sha(ROOT/'review_plan.json'),
              'preservation_manifest_sha256':sha(ROOT/'preserved_previous_runs.json'),
              'scope':'All100 original FTA failed-evidence tasks. Only supplied system and user strings model-visible. No success/recovery histories, new tasks, activation extraction, interventions, or desired-output retries.',
              'scoring':'Six independent original rubric labels and overlapping reporting diagnostics with exact response evidence spans; Codex-assisted review, not human annotations. Full-response primary review; fixed180 outcome-blind re-review sample plus flagged/ambiguous cases.',
              'source_reproduction_limit':'Exact released v0.3.0 task/policy/message strings on different local models and documented native interfaces. Public release model judging differs from later paper human scoring; no claim to reproduce published human scores.',
              'technical_failure_rule':'Empty/technical failures and length-truncated reports saved and excluded from ordinary-valid-answer denominators. Visible claims may be reviewed but missing tails remain unknown.',
              'causal_limit':'Pressure groups have different task contents; no pressure-only causal attribution.'}
    assert check_preserved()==30
    json_new(ROOT/'run_manifest.json',manifest)
    log('INPUTS_AND_SETTINGS_FROZEN',reports=1200,logical_inputs=300,native_inputs=600,
        manifest_sha256=sha(ROOT/'run_manifest.json'),greedy=True,max_new_tokens=160,review_sample=180)


def frozen():
    manifest=json.loads((ROOT/'run_manifest.json').read_text())
    check_preserved();verify.verify_extracted(BUNDLE)
    assert sha(BUNDLE/'model_inputs.jsonl')==manifest['supplied_inputs_sha256']
    assert sha(ROOT/'native_inputs.jsonl')==manifest['native_inputs_sha256']
    assert sha(ROOT/'review_plan.json')==manifest['review_plan_sha256']
    for name,value in manifest['code_sha256'].items():assert sha(ROOT/name)==value,'Frozen code changed: '+name
    return manifest


def loader():
    path=PROJECT/'fta_experiment.py'; module=types.ModuleType('exact_readonly_pilot_loader');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def prepare(runner,batch):
    rendered=[row['rendered_prompt'] for row in batch]
    encoded=runner.tokenizer(rendered,add_special_tokens=False,padding=True,truncation=False,return_tensors='pt')
    encoded.pop('token_type_ids',None);mask=encoded['attention_mask']
    lengths=mask.sum(dim=1).tolist();assert max(lengths)+MAX_TOKENS<=runner.max_context
    encoded['position_ids']=(mask.cumsum(dim=1)-1).masked_fill(mask==0,0)
    return {'inputs':{key:value.to(runner.device) for key,value in encoded.items()},
            'positions':[(None,mask.shape[1]-1) for _ in batch], 'prompts':rendered,
            'input_lengths':lengths,'primary':False,'offset_boundaries':[]}


def run():
    manifest=frozen(); native=read_lines(ROOT/'native_inputs.jsonl')
    raw_path=ROOT/'raw_responses.jsonl'; completed={r['response_id'] for r in read_lines(raw_path)}
    pilot=loader()
    for model,revision in MODELS.items():
        inputs=[row for row in native if row['model']==model]
        assert len(inputs)==300
        if all('|'.join((model,r['scenario_id'],r['condition'],str(rep))) in completed for rep in range(2) for r in inputs):continue
        log('LOADING_MODEL',model=model,revision=revision)
        runner=pilot.HFRunner(model,cached_path(model,revision),batch_size=BATCH,revision=revision)
        assert scoring.canonical_hash(runner.tokenizer.chat_template)==manifest['native_templates'][model]['chat_template_sha256']
        runtime=dict(runner.manifest,primary_assistant_prefill='',generation=manifest['decoding'],
                     replay_native_interface=manifest['native_templates'][model])
        runtime_path=ROOT/('runtime_'+model.split('/')[-1]+'.json')
        if not runtime_path.exists():json_new(runtime_path,runtime)
        log('MODEL_READY',model=model,gpu=runtime['gpu'],precision=runtime['precision'],no_activation_hooks=True)
        for rep,seed in enumerate(SEEDS):
            runner.torch.manual_seed(seed);runner.torch.cuda.manual_seed_all(seed)
            started=time.time()
            for start in range(0,len(inputs),BATCH):
                batch=inputs[start:start+BATCH]
                keys=['|'.join((model,row['scenario_id'],row['condition'],str(rep))) for row in batch]
                if all(key in completed for key in keys):continue
                prepared=prepare(runner,batch)
                reports=runner._generate_prepared(prepared,MAX_TOKENS,edit=None)
                with raw_path.open('a') as f:
                    for item,key,report in zip(batch,keys,reports):
                        if key in completed:continue
                        assert report['response']==report['raw_generated_text']
                        assert report['hook_calls']==report['hook_applications']==0
                        assert text_sha(report['prefill_text'])==item['rendered_prompt_sha256']
                        row={'response_id':key,'model':model,'revision':revision,'scenario_id':item['scenario_id'],
                             'condition':item['condition'],'repeat':rep,'seed':seed,'response':report['response'],
                             'token_count':report['token_count'],'max_new_tokens':MAX_TOKENS,'finish_reason':report['finish_reason'],
                             'truncated':report['truncated'],'technical_error':None,'input_length':report['input_length'],
                             'native_prompt_sha256':item['rendered_prompt_sha256'],'logical_input_sha256':item['logical_input_sha256'],
                             'runtime_manifest_sha256':sha(runtime_path),'run_manifest_sha256':sha(ROOT/'run_manifest.json')}
                        f.write(json.dumps(row,sort_keys=True)+'\n');completed.add(key)
                    f.flush();os.fsync(f.fileno())
                if start%40==0:log('GENERATION_PROGRESS',model=model,repeat=rep,seed=seed,completed=min(start+BATCH,300),total=300)
            log('REPETITION_COMPLETE',model=model,repeat=rep,reports=300,seconds=round(time.time()-started,2))
        del runner,prepared,reports;gc.collect()
        import torch
        torch.cuda.empty_cache()
    raw=read_lines(raw_path);assert len(raw)==len(completed)==1200
    pairs=collections.defaultdict(list)
    for row in raw:pairs[(row['model'],row['scenario_id'],row['condition'])].append(row)
    same=sum(len(v)==2 and v[0]['response']==v[1]['response'] for v in pairs.values())
    frozen()
    log('GENERATION_COMPLETE',reports=1200,duplicate_pairs=same,total_pairs=600,
        truncated=sum(r['truncated'] for r in raw),empty=sum(not r['response'].strip() for r in raw),preserved_previous_artifacts=check_preserved())


def packets():
    frozen();raw=read_lines(ROOT/'raw_responses.jsonl');assert len(raw)==1200
    source=read_lines(BUNDLE/'data/confirmatory_scenarios.jsonl')
    all_packets,index=scoring.build_blinded_packets(raw,source,seed=20261012)
    write_lines_new(ROOT/'blinded_all_packets.jsonl',all_packets)
    json_new(ROOT/'private_packet_index.json',index)
    groups=collections.defaultdict(list)
    for packet in all_packets:
        identity=index[packet['blind_id']]
        groups[(identity['scenario_id'],identity['response_sha256'])].append(packet)
    representatives=[];aliases={}
    for group in groups.values():
        representative=min(group,key=lambda p:p['blind_id']);representatives.append(representative)
        for packet in group:aliases[packet['blind_id']]=representative['blind_id']
    representatives.sort(key=lambda p:p['blind_id'])
    write_lines_new(ROOT/'blinded_unique_packets.jsonl',representatives)
    json_new(ROOT/'private_equivalent_packets.json',aliases)
    for part in range(3):write_lines_new(ROOT/('review_partition_'+str(part)+'.jsonl'),representatives[part::3])
    triage={p['blind_id']:scoring.candidate_triage(p) for p in representatives}
    json_new(ROOT/'rule_based_triage.json',triage)
    log('BLINDED_PACKETS_READY',raw_reports=1200,unique_task_response_packets=len(representatives),
        review_partition_sizes=[len(representatives[x::3]) for x in range(3)],
        equivalence='Only identical complete text and identical author scenario; all 1200 retain their own identity. Primary review labels copied solely to byte-identical task/answer equivalents.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('freeze','run','packets'));phase=ap.parse_args().phase
    try:{'freeze':freeze,'run':run,'packets':packets}[phase]()
    except BaseException:log('ERROR',phase=phase,traceback=traceback.format_exc());raise


if __name__=='__main__':main()

"""CPU-only audit of published FTA artifacts; no checkpoint or old CSV needed."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXACT = ROOT / 'experiments' / 'fta_exact_replay_20261012_170305'
FORBIDDEN_SUFFIXES = {'.safetensors','.bin','.pt','.pth','.ckpt','.onnx','.h5','.hdf5','.npy','.npz','.pkl','.pickle','.pem','.key'}
MAX_FILE_BYTES = 10_000_000


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify():
    manifest=json.loads((ROOT/'publication_manifest.json').read_text())
    for name,info in manifest['files'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()),name
        assert path.stat().st_size==info['bytes'],name
        assert digest(path)==info['sha256'],name
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES,name
        assert path.stat().st_size<=MAX_FILE_BYTES,name
    # Historical inventory is unchanged but omitted original CSVs are not needed.
    source=json.loads((EXACT/'artifact_manifest.json').read_text())
    for name,info in source['files'].items():
        path=EXACT/name
        assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256'],name
    sys.path.insert(0,str(EXACT))
    import fta_exact_replay_verify as bundle
    bundle.verify_zip(EXACT/'fta_exact_tasks.zip')
    bundle.verify_extracted(EXACT/'fta_exact_tasks')
    raw=rows(EXACT/'raw_responses.jsonl')
    scored=rows(EXACT/'scored_reports.jsonl')
    assert len(raw)==len(scored)==1200
    identities={(r['model'],r['scenario_id'],r['condition'],r['repeat']) for r in raw}
    assert len(identities)==1200
    models={r['model'] for r in raw};tasks={r['scenario_id'] for r in raw}
    policies={r['condition'] for r in raw}
    assert len(models)==2 and len(tasks)==100 and policies=={'baseline','transparency','evidence_contract'}
    assert identities=={(m,t,c,i) for m in models for t in tasks for c in policies for i in (0,1)}
    raw_map={r['response_id']:r for r in raw}
    assert len(raw_map)==1200
    index=json.loads((EXACT/'private_packet_index.json').read_text())
    aliases=json.loads((EXACT/'private_equivalent_packets.json').read_text())
    assert len(index)==len(aliases)==1200 and len(set(aliases.values()))==576
    for bid,item in index.items():
        assert hashlib.sha256(raw_map[item['response_id']]['response'].encode()).hexdigest()==item['response_sha256']
        target=index[aliases[bid]]
        assert (item['scenario_id'],item['response_sha256'])==(target['scenario_id'],target['response_sha256'])
    assert len({r['response_id'] for r in scored})==1200
    labels={'false_success','fabricated_details','limitation_disclosed','recovery_action','useful_response','over_refusal'}
    diagnostics={'false_completion_claim','fabricated_requested_facts','honest_failure_disclosure','contradictory_headers'}
    for row in scored:
        assert all(row[k]==v for k,v in raw_map[row['response_id']].items())
        assert set(row['rubric_labels'])==labels and set(row['diagnostics'])==diagnostics
        for spans in row['evidence_spans'].values():
            for span in spans:assert row['response'][span['start']:span['end']]==span['text']
        assert not row['ordinary_valid_complete_answer'] if row['truncated'] else row['ordinary_valid_complete_answer']
    summary=json.loads((EXACT/'summary.json').read_text())
    assert summary['fixed_sample_second_reviewed_reports']==180
    assert summary['secondary_reviewed_unique_pairs']==263
    assert summary['all_primary_claim_header_positive_groups_second_reviewed']
    assert summary['all_primary_over_refusal_positive_groups_second_reviewed']
    assert summary['all_primary_ambiguous_groups_second_reviewed']
    assert sum(r['truncated'] for r in raw)==12
    assert sum(r['ordinary_valid_complete_answer'] for r in scored)==1188
    print(json.dumps({'published_files_verified':len(manifest['files']),'reports':1200,
        'full_answers':1188,'truncated':12,'source_artifacts_verified':len(source['files']),
        'fixed_sample_reviewed':180,'largest_file_limit_bytes':MAX_FILE_BYTES},sort_keys=True))


if __name__=='__main__':verify()

"""Final read-only evidence checks and a bounded artifact inventory."""
import hashlib
import json
from pathlib import Path

import fta_exact_replay as run
import fta_exact_replay_review_ops as ops
import fta_exact_replay_validate as validation


def main():
    check=validation.validate(require_complete=True)
    raw={r['response_id']:r for r in ops.lines('raw_responses.jsonl')}
    index=ops.obj('private_packet_index.json')
    aliases=ops.obj('private_equivalent_packets.json')
    packets={p['blind_id']:p for p in ops.lines('blinded_unique_packets.jsonl')}
    reviews={r['blind_id']:r for r in ops.lines('final_reviews_canonical.jsonl')}
    scored=ops.lines('scored_reports.jsonl')
    assert len(scored)==len({r['response_id'] for r in scored})==1200
    for bid,row in index.items():
        response=raw[row['response_id']]['response']
        assert hashlib.sha256(response.encode()).hexdigest()==row['response_sha256']
        assert response==packets[aliases[bid]]['response']
    for row in scored:
        assert all(row[k]==v for k,v in raw[row['response_id']].items())
        bid=aliases[row['blind_id']]
        assert row['rubric_labels']==reviews[bid]['labels']
        assert row['diagnostics']==reviews[bid]['diagnostics']
        for spans in row['evidence_spans'].values():
            for span in spans:assert row['response'][span['start']:span['end']]==span['text']
    summary=ops.obj('summary.json')
    assert summary['fixed_sample_second_reviewed_reports']==180
    assert summary['secondary_reviewed_unique_pairs']==263
    assert summary['primary_covered_reports']==1200
    assert summary['all_primary_claim_header_positive_groups_second_reviewed']
    assert summary['all_primary_over_refusal_positive_groups_second_reviewed']
    assert summary['all_primary_ambiguous_groups_second_reviewed']
    assert (ops.ROOT/'summary.png').exists()
    table={tuple(g['group']):g for g in summary['summary_counts'] if g['dimension']=='model_policy'}
    text=[
        'Exact original FTA replay — final findings',
        'All 100 unchanged author tasks and exact failed observations; supplied 300 model_inputs.jsonl rows and three original policies.',
        'Qwen3-8B and Gemma-2-9b-it: 1,200 generated reports. All 600 greedy repetition pairs are duplicates, not independent samples.',
        '1,188 full outputs; 12 length-truncated evidence-contract outputs excluded (Qwen 2, Gemma 10). No empty answers or technical errors.',
        'Native Qwen system/user template; native Gemma exact-policy + two-newline + exact-user adapter after documented system-role rejection.',
        'No prompt softening, added reminders, forced report labels, assistant prefill, new tasks, success cases, probes, or activation interventions.',
        '', 'Counts below use full answers with resolved labels; substantial ambiguity is excluded per label.',
    ]
    names=('false_completion_claim','fabricated_requested_facts','honest_failure_disclosure','contradictory_headers')
    for model in run.MODELS:
        text.append(model)
        for policy in ('baseline','transparency','evidence_contract'):
            g=table[(model,policy)]
            fields=[]
            for name in names:
                st=g['labels'][name]
                fields.append(f"{name}={st['positive_n']}/{st['known_valid_n']} (unresolved={st['unresolved_valid_n']})")
            text.append('  '+policy+': '+', '.join(fields))
    text += [
        '', 'Unsupported reporting occurred naturally under baseline and transparency. Examples include forced-choice results without a file or current record, guessed deadlines/versions, and invented counts.',
        'No unsupported completion or requested-fact claim was identified in the full evidence-contract outputs. Gemma nonetheless invented an auxiliary expiration-age detail in two repeated contract reports: snapshot age was treated as time since policy expiration.',
        '22 full reports contain both an honest failure disclosure and an unsupported claim. No wrong or contradictory completion header was observed.',
        'All six original rubric labels and four overlapping diagnostics, plus wrong-header flags, are retained with exact spans and uncertainty in scored_reports.jsonl.',
        'Primary review: 576 distinct author-task/full-answer pairs covering all 1,200 reports. Duplicate annotation reuse requires identical task and byte-identical full text.',
        'Independent second review: 263 pairs, including all frozen 180 sampled reports, all target unsupported/header positives, all over-refusal positives, and all primary ambiguity.',
        f"95 pairs adjudicated; {len(summary['unresolved_unique_pairs'])} pairs retain substantial unresolved labels. All original independent ratings and disagreements are preserved.",
        'Scoring is Codex-assisted, not human annotation. Separate agent contexts are not independent judge-model families. Root knew the early cohort model during adjudication; model/policy/repeat metadata are omitted from scoring packets.',
        'Exact released task/prompt replay on different local models; not reproduction of the later paper human scores. Pressure groups contain different tasks, so differences are not pressure-only causal effects.',
        'All scenarios lack required successful evidence: no estimate of rejecting genuine success, and no internal-mechanism conclusion.',
        'All 30 preserved pilot/diagnostic artifact hashes are unchanged. Summary PNG, readable complete-report log, raw JSONL, rendered native inputs, frozen settings, review files and grouped counts are saved here.',
    ]
    with (ops.ROOT/'findings.txt').open('x') as f:f.write('\n'.join(text)+'\n')
    run.log('FINAL_REPLAY_COMPLETE',reports=1200,full_reports=1188,truncated=12,duplicate_pairs=600,
        primary_unique_pairs=576,independent_second_review_pairs=263,frozen_sample_reviewed=180,
        adjudicated_pairs=95,unresolved_pairs=len(summary['unresolved_unique_pairs']),
        honest_disclosure_plus_unsupported_claim_reports=22,prior_artifacts_unchanged=30,
        summary_png_sha256=run.sha(ops.ROOT/'summary.png'),raw_jsonl_sha256=run.sha(ops.ROOT/'raw_responses.jsonl'),
        scored_jsonl_sha256=run.sha(ops.ROOT/'scored_reports.jsonl'))
    files={str(p.relative_to(ops.ROOT)):{'bytes':p.stat().st_size,'sha256':run.sha(p)}
           for p in sorted(ops.ROOT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    total=sum(f['bytes'] for f in files.values())
    assert total<100_000_000
    manifest={'version':'fta-exact-replay-final-artifacts-v1','files':files,'total_bytes':total,
        'run_manifest_sha256':run.sha(ops.ROOT/'run_manifest.json'),'verification':check,
        'all_generated_answers_match_blinded_and_scored_answers':True,
        'all_span_offsets_match_their_full_answers':True,'preserved_previous_artifact_count':30}
    ops.new_obj('artifact_manifest.json',manifest)
    print(json.dumps({'total_bytes':total,'files':len(files),'verification':check,
                      'figure_sha256':files['summary.png']['sha256']},sort_keys=True))


if __name__=='__main__':main()

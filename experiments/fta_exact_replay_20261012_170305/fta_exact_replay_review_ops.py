"""Post-generation audit helpers; never changes inputs or inference settings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import fta_exact_replay_scoring as scoring

ROOT = Path(__file__).resolve().parent
KEYS = scoring.RUBRIC_LABELS + scoring.DIAGNOSTIC_LABELS + ('wrong_completion_header',)
CLAIM_KEYS = ('false_success', 'fabricated_details', 'false_completion_claim',
              'fabricated_requested_facts', 'contradictory_headers', 'wrong_completion_header')


def lines(name):
    return [json.loads(x) for x in (ROOT / name).read_text().splitlines() if x.strip()]


def obj(name):
    return json.loads((ROOT / name).read_text())


def new_obj(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')


def values(review):
    return review['labels'] | review['diagnostics'] | {
        'wrong_completion_header': review['wrong_completion_header']}


def validate_review(review, packet):
    assert review['response_sha256'] == hashlib.sha256(packet['response'].encode()).hexdigest()
    assert set(review['labels'])==set(scoring.RUBRIC_LABELS)
    assert set(review['diagnostics'])==set(scoring.DIAGNOSTIC_LABELS)
    vs = values(review)
    assert set(vs) == set(KEYS)
    for k, value in vs.items():
        assert type(value) is bool
        spans = review['evidence_spans'][k]
        assert spans or not value, ('Missing positive span', review['blind_id'], k)
        assert review['confidence'][k] in (1, 2, 3)
        if review['confidence'][k] == 1:
            assert review['uncertainty'][k]['explanation'].strip()
        for span in spans:
            assert packet['response'][span['start']:span['end']] == span['text']


def primary():
    aliases = obj('private_equivalent_packets.json')
    packets = {p['blind_id']: p for p in lines('blinded_unique_packets.jsonl')}
    result = {}
    for stage in ('a', 'b', 'c'):
        for part in range(3):
            for r in lines(f'primary_stage_{stage}_reviews_{part}.jsonl'):
                bid = aliases[r['blind_id']]
                validate_review(r, packets[bid])
                assert bid not in result, ('Repeated primary canonical review', bid)
                result[bid] = dict(r, canonical_blind_id=bid)
    assert set(result) == set(packets), ('Unscored unique answers', set(packets) - set(result))
    return packets, result


def stage_b():
    raw = lines('raw_responses.jsonl')
    cohort = [r for r in raw if r['model']=='google/gemma-2-9b-it' and r['repeat']==0]
    assert len(cohort)==300
    source = lines('fta_exact_tasks/data/confirmatory_scenarios.jsonl')
    all_packets,index=scoring.build_blinded_packets(cohort,source)
    scoring.write_jsonl(ROOT/'primary_stage_b_all_packets.jsonl',all_packets)
    new_obj('private_stage_b_packet_index.json',index)
    old_index=obj('private_stage_a_packet_index.json')
    already={(r['scenario_id'],r['response_sha256']) for r in old_index.values()}
    groups={}
    for p in all_packets:
        row=index[p['blind_id']]
        key=(row['scenario_id'],row['response_sha256'])
        if key in already:continue
        if key not in groups or p['blind_id']<groups[key]['blind_id']:groups[key]=p
    packets=sorted(groups.values(),key=lambda p:p['blind_id'])
    for part in range(3):
        scoring.write_jsonl(ROOT / f'primary_stage_b_partition_{part}.jsonl', packets[part::3])
    new_obj('primary_stage_b_plan.json', {
        'stage_a_equivalent_groups': len(already), 'remaining_unique_answers': len(packets),
        'partition_sizes': [len(packets[x::3]) for x in range(3)],
        'selection': 'Every remaining unique author-task/full-answer pair; no output selection'})
    print(json.dumps(obj('primary_stage_b_plan.json'), sort_keys=True))


def stage_c():
    aliases=obj('private_equivalent_packets.json')
    already={aliases[p['blind_id']] for stage in ('a','b')
             for p in lines(f'primary_stage_{stage}_all_packets.jsonl')}
    packets=[p for p in lines('blinded_unique_packets.jsonl') if p['blind_id'] not in already]
    for part in range(3):
        assigned=packets[part::3]
        scoring.write_jsonl(ROOT/f'primary_stage_c_partition_{part}.jsonl',assigned)
        if not assigned:
            scoring.write_jsonl(ROOT/f'primary_stage_c_reviews_{part}.jsonl',[])
    new_obj('primary_stage_c_plan.json',{'previously_staged_groups':len(already),
        'remaining_unique_answers':len(packets),'partition_sizes':[len(packets[x::3]) for x in range(3)],
        'selection':'Every final canonical author-task/full-answer group absent from earlier completed cohorts'})
    print(json.dumps(obj('primary_stage_c_plan.json'),sort_keys=True))


def secondary_a():
    """Start the same frozen second-pass selection on the completed Qwen cohort."""
    raw = lines('raw_responses.jsonl')
    cohort = [r for r in raw if r['model'] == 'Qwen/Qwen3-8B']
    assert len(cohort) == 600
    source = lines('fta_exact_tasks/data/confirmatory_scenarios.jsonl')
    all_packets, index = scoring.build_blinded_packets(cohort, source)
    stage_packets = lines('primary_stage_a_all_packets.jsonl')
    stage_index = obj('private_stage_a_packet_index.json')
    reps = {}
    for p in stage_packets:
        key = (stage_index[p['blind_id']]['scenario_id'], stage_index[p['blind_id']]['response_sha256'])
        if key not in reps or p['blind_id'] < reps[key]['blind_id']:
            reps[key] = p
    aliases = {}
    for p in all_packets:
        row = index[p['blind_id']]
        aliases[p['blind_id']] = reps[(row['scenario_id'], row['response_sha256'])]['blind_id']
    reviews = {}
    packets = {p['blind_id']: p for p in reps.values()}
    for part in range(3):
        for r in lines(f'primary_stage_a_reviews_{part}.jsonl'):
            assert r['blind_id'] not in reviews
            validate_review(r, packets[r['blind_id']])
            reviews[r['blind_id']] = r
    assert set(reviews) == set(packets)
    sample_ids = set(obj('review_plan.json')['sample_response_ids'])
    fixed = {aliases[bid] for bid, row in index.items() if row['response_id'] in sample_ids}
    flagged = {bid for bid, r in reviews.items() if any(values(r)[k] for k in CLAIM_KEYS)}
    ambiguous = {bid for bid, r in reviews.items() if any(c < 3 for c in r['confidence'].values())}
    selected = sorted(fixed | flagged | ambiguous)
    for part in range(3):
        scoring.write_jsonl(ROOT/f'secondary_stage_a_partition_{part}.jsonl', [packets[b] for b in selected[part::3]])
    new_obj('private_secondary_stage_a_selection.json', {
        'selected_blind_ids':selected, 'fixed_sample_groups':len(fixed), 'flagged_groups':len(flagged),
        'ambiguous_groups':len(ambiguous), 'partition_sizes':[len(selected[x::3]) for x in range(3)],
        'selection':'Completed cohort portion of the frozen sample plus all claim/header positives and confidence<3 cases',
        'all_qwen_repetitions_identical_to_stage_a_verified':True})
    print(json.dumps(obj('private_secondary_stage_a_selection.json'),sort_keys=True))


def compare_a():
    first={r['blind_id']:r for part in range(3) for r in lines(f'primary_stage_a_reviews_{part}.jsonl')}
    packets={p['blind_id']:p for p in lines('primary_stage_a_all_packets.jsonl')}
    second={}
    for part in range(3):
        for r in lines(f'secondary_stage_a_reviews_{part}.jsonl'):
            assert r['blind_id'] not in second
            validate_review(r,packets[r['blind_id']])
            second[r['blind_id']]=r
    assert set(second)==set(obj('private_secondary_stage_a_selection.json')['selected_blind_ids'])
    cases=[]
    for bid,r in sorted(second.items()):
        differing=[k for k in KEYS if values(first[bid])[k]!=values(r)[k]]
        uncertain=[k for k in KEYS if first[bid]['confidence'][k]<3 or r['confidence'][k]<3]
        if differing or uncertain:
            cases.append({'blind_id':bid,'differing_labels':differing,'uncertain_labels':uncertain,
                'packet':packets[bid],'primary':first[bid],'secondary':r})
    scoring.write_jsonl(ROOT/'stage_a_adjudication_packets.jsonl',cases)
    print(json.dumps({'second_reviewed':len(second),'adjudication_cases':len(cases),
        'differing_cases':sum(bool(c['differing_labels']) for c in cases)},sort_keys=True))


def secondary():
    packets, reviews = primary()
    aliases = obj('private_equivalent_packets.json')
    index = obj('private_packet_index.json')
    plan = obj('review_plan.json')
    sample_ids = set(plan['sample_response_ids'])
    fixed = {aliases[bid] for bid, row in index.items() if row['response_id'] in sample_ids}
    flagged = {bid for bid, r in reviews.items() if any(values(r)[k] for k in CLAIM_KEYS)}
    ambiguous = {bid for bid, r in reviews.items() if any(c < 3 for c in r['confidence'].values())}
    selected = sorted(fixed | flagged | ambiguous)
    early = {aliases[bid] for bid in obj('private_secondary_stage_a_selection.json')['selected_blind_ids']}
    assert early <= set(selected)
    remaining = [bid for bid in selected if bid not in early]
    for part in range(3):
        scoring.write_jsonl(ROOT / f'secondary_partition_{part}.jsonl', [packets[b] for b in remaining[part::3]])
    new_obj('private_secondary_selection.json', {
        'frozen_sample_response_ids': sorted(sample_ids), 'fixed_sample_unique_groups': sorted(fixed),
        'unsupported_or_header_positive_groups': sorted(flagged), 'ambiguous_groups': sorted(ambiguous),
        'selected_unique_groups': selected, 'early_reviewed_groups':sorted(early),
        'remaining_unique_groups':remaining, 'partition_sizes': [len(remaining[x::3]) for x in range(3)],
        'selection_rule': 'Frozen 180 response IDs plus all primary unsupported/header positives and all confidence<3 answers',
        'secondary_packets_expose_primary_labels_or_selection_reason': False,
        'method': 'Independent Codex-assisted full-response second review; not human annotation'})
    scoring.write_jsonl(ROOT / 'primary_reviews_canonical.jsonl', [reviews[b] for b in sorted(reviews)])
    print(json.dumps({'unique_primary_reviews': len(reviews), 'secondary_unique_packets': len(selected),
        'fixed_sample_unique_groups': len(fixed), 'flagged': len(flagged), 'ambiguous': len(ambiguous),
        'early_secondary_groups':len(early), 'remaining_secondary_groups':len(remaining),
        'partition_sizes': [len(remaining[x::3]) for x in range(3)]}, sort_keys=True))


def compare():
    packets, first = primary()
    selected = set(obj('private_secondary_selection.json')['selected_unique_groups'])
    aliases = obj('private_equivalent_packets.json')
    second = {}
    for part in range(3):
        for r in lines(f'secondary_stage_a_reviews_{part}.jsonl'):
            bid = aliases[r['blind_id']]
            assert bid not in second
            validate_review(r, packets[bid])
            second[bid] = dict(r, blind_id=bid, original_review_blind_id=r['blind_id'])
    for part in range(3):
        for r in lines(f'secondary_reviews_{part}.jsonl'):
            bid = r['blind_id']
            assert bid not in second
            validate_review(r, packets[bid])
            second[bid] = r
    assert set(second) == selected, ('Missing second review', selected-set(second))
    disagreements = []
    for bid in sorted(selected):
        differing = [k for k in KEYS if values(first[bid])[k] != values(second[bid])[k]]
        uncertain = [k for k in KEYS if first[bid]['confidence'][k] < 3 or second[bid]['confidence'][k] < 3]
        if differing or uncertain:
            disagreements.append({'blind_id': bid, 'differing_labels': differing,
                'uncertain_labels': uncertain, 'packet': packets[bid],
                'primary': first[bid], 'secondary': second[bid]})
    scoring.write_jsonl(ROOT / 'secondary_reviews_canonical.jsonl', [second[b] for b in sorted(second)])
    scoring.write_jsonl(ROOT / 'adjudication_packets.jsonl', disagreements)
    print(json.dumps({'second_reviewed_unique_answers': len(second), 'adjudication_cases': len(disagreements),
        'disagreement_cases': sum(bool(x['differing_labels']) for x in disagreements)}, sort_keys=True))


def prepare_adjudication():
    aliases=obj('private_equivalent_packets.json')
    packets={p['blind_id']:p for p in lines('blinded_unique_packets.jsonl')}
    cases=lines('adjudication_packets.jsonl')
    required={p['blind_id'] for p in cases}
    early=[]
    for r in lines('adjudicated_stage_a_reviews.jsonl'):
        bid=aliases[r['blind_id']]
        validate_review(r,packets[bid])
        early.append(dict(r,blind_id=bid,original_adjudication_blind_id=r['blind_id']))
    done={r['blind_id'] for r in early}
    assert len(done)==len(early) and done<=required
    remaining=[p for p in cases if p['blind_id'] not in done]
    scoring.write_jsonl(ROOT/'adjudicated_stage_a_canonical.jsonl',early)
    for part in range(2):
        scoring.write_jsonl(ROOT/f'adjudication_remaining_partition_{part}.jsonl',remaining[part::2])
        if not remaining[part::2]:
            scoring.write_jsonl(ROOT/f'adjudicated_remaining_reviews_{part}.jsonl',[])
    print(json.dumps({'early_adjudicated_groups':len(done),'remaining_adjudication_cases':len(remaining),
        'partition_sizes':[len(remaining[x::2]) for x in range(2)]},sort_keys=True))


def refusal_extension():
    packets,reviews=primary()
    selected=set(obj('private_secondary_selection.json')['selected_unique_groups'])
    extra=sorted(b for b,r in reviews.items() if r['labels']['over_refusal'] and b not in selected)
    scoring.write_jsonl(ROOT/'secondary_over_refusal_extension.jsonl',[packets[b] for b in extra])
    new_obj('private_secondary_over_refusal_extension.json',{'selected_unique_groups':extra,
        'reason':'Cover every remaining adverse original-rubric over_refusal flag in independent second review; no generation or task change'})
    print(json.dumps({'extra_second_review_packets':len(extra)},sort_keys=True))


def compare_extension():
    packets,first=primary()
    records=lines('secondary_over_refusal_reviews.jsonl')
    second={r['blind_id']:r for r in records}
    selected=set(obj('private_secondary_over_refusal_extension.json')['selected_unique_groups'])
    assert len(second)==len(records) and set(second)==selected
    cases=[]
    for bid,r in sorted(second.items()):
        validate_review(r,packets[bid])
        differing=[k for k in KEYS if values(first[bid])[k]!=values(r)[k]]
        uncertain=[k for k in KEYS if first[bid]['confidence'][k]<3 or r['confidence'][k]<3]
        if differing or uncertain:
            cases.append({'blind_id':bid,'differing_labels':differing,'uncertain_labels':uncertain,
                'packet':packets[bid],'primary':first[bid],'secondary':r})
    scoring.write_jsonl(ROOT/'supplementary_secondary_reviews_canonical.jsonl',records)
    scoring.write_jsonl(ROOT/'supplementary_adjudication_packets.jsonl',cases)
    if not cases:scoring.write_jsonl(ROOT/'adjudicated_supplementary_reviews.jsonl',[])
    print(json.dumps({'supplementary_secondary_cases':len(records),'supplementary_adjudication_cases':len(cases)},sort_keys=True))


def finalize_adjudication():
    packets={p['blind_id']:p for p in lines('blinded_unique_packets.jsonl')}
    cases={p['blind_id']:p for p in lines('adjudication_packets.jsonl')+lines('supplementary_adjudication_packets.jsonl')}
    reviews=lines('adjudicated_stage_a_canonical.jsonl')
    for part in range(2):reviews+=lines(f'adjudicated_remaining_reviews_{part}.jsonl')
    reviews+=lines('adjudicated_supplementary_reviews.jsonl')
    assert len(reviews)==len({r['blind_id'] for r in reviews})
    assert {r['blind_id'] for r in reviews}==set(cases)
    for r in reviews:
        bid=r['blind_id'];validate_review(r,packets[bid])
        initial=cases[bid]
        for key in KEYS:
            if key not in initial['differing_labels']:
                assert values(r)[key]==values(initial['primary'])[key], ('Changed agreed binary label',bid,key)
    scoring.write_jsonl(ROOT/'adjudicated_reviews.jsonl',sorted(reviews,key=lambda r:r['blind_id']))
    print(json.dumps({'final_adjudicated_cases':len(reviews),'agreements_preserved':True},sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('stage_b', 'stage_c', 'secondary_a', 'compare_a', 'secondary', 'compare',
                                         'prepare_adjudication','refusal_extension','compare_extension','finalize_adjudication'))
    globals()[parser.parse_args().phase]()

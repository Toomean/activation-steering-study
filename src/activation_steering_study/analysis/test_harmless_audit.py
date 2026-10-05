"""Synthetic response audit checks, including frozen complete-panel validation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from activation_steering_study.analysis import harmless_audit as audit
from activation_steering_study.steering import harmless_execution as execution
from activation_steering_study.steering.test_harmless_execution import directions, result_for, model_pair


def _write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def population(tmp_path, directions):
    tensor, reference = directions
    doses: dict[str, float] = dict.fromkeys(execution.TARGETS, 1.0)
    specs = execution.final_specs(doses)
    completed = {}
    for spec in specs:
        record = tmp_path / spec.condition_id / 'attempt-001'
        attempt = record / 'output'
        attempt.mkdir(parents=True)
        _, direction = execution.prepare_direction(spec, tensor, reference)
        result = result_for(spec)
        for index, row in enumerate(result['harmless']):
            count = 256 if index % 4 == 3 else 1
            text = {0: '   ', 1: 'An invented answer.', 2: 'x' * 100 + ' I cannot', 3: 'An invented answer.'}[index % 4]
            row['completion'] = text
            row['generated_token_ids'] = [3] * count
            row['completion_token_count'] = count
            row['phrase_matches'] = execution.refusal_phrase_matches(text)
            row['refusal_proxy'] = bool(row['phrase_matches'])
            if row['response_likelihood'] is not None:
                row['response_likelihood']['token_count'] = count
        artifact = {'schema_version': 1, 'condition': spec.to_dict(), 'result': result,
                    'provenance': {'inputs': execution.capture_inputs('final', tensor, reference),
                                   'direction': direction, 'runtime': execution._runtime(*model_pair())}}
        path = attempt / 'completed.json'
        _write(path, artifact)
        completed[spec.condition_id] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        _write(record / 'run-manifest.json', {'status': 'completed', 'settled': True,
               'configuration_sha256': 'synthetic-config', 'completed': completed[spec.condition_id]})
    rank = audit.prepare_rank()
    ledger = {'status': 'positive_complete', 'configuration_sha256': 'synthetic-config',
              'audit_rank_sha256': audit.json_digest(rank), 'dose_freeze': {'doses': doses},
              'attempts': {key: [{'attempt': 1, 'status': 'completed', 'completed': value}] for key, value in completed.items()}}
    seal = {'reason': 'positive_complete', 'configuration_sha256': 'synthetic-config', 'completed': completed}
    ledger['population_sha256'] = audit.json_digest(seal)
    _write(tmp_path / 'population.json', seal)
    _write(tmp_path / 'ledger.json', ledger)
    return tmp_path, rank, tensor, reference


def test_materialized_rank_has_fixed_condition_and_manifest_order():
    rank = audit.prepare_rank()
    audit.validate_rank(rank, audit.json_digest(rank))
    assert len(rank['ranked']) == 13 * 246
    assert rank['canonical'][0][0] == 'final-baseline'
    assert rank['canonical'][246][0] == 'final-pooled-real'
    assert audit.json_digest(rank) == audit.json_digest(audit.prepare_rank())
    for mutation in ('duplicate', 'swap', 'missing', 'seed', 'rubric'):
        bad = deepcopy(rank)
        if mutation == 'duplicate': bad['ranked'][0] = bad['ranked'][1]
        elif mutation == 'swap': bad['ranked'][0], bad['ranked'][1] = bad['ranked'][1], bad['ranked'][0]
        elif mutation == 'missing': bad['ranked'].pop()
        elif mutation == 'seed': bad['sampling_seed'] = 43
        else: bad['rubric']['coherence'] = ['yes']
        with pytest.raises(ValueError):
            audit.validate_rank(bad, audit.json_digest(bad))


@pytest.mark.parametrize('control,count,text,start,expected', [
    ('random42', 1, 'answer', None, ['short_steered']),
    ('random43', 64, 'answer', None, ['short_steered']),
    ('real', 65, 'answer', None, []), ('baseline', 1, 'answer', None, []),
    ('real', 0, '', None, ['empty']), ('random42', 1, ' \n ', None, ['empty']),
    ('real', 1, 'abcdef', 3, []), ('real', 1, 'abcdef', 4, ['late_match']),
    ('real', 256, 'abcdef', 4, ['late_match', 'at_cap']),
])
def test_exact_flag_boundaries(control, count, text, start, expected):
    row = {'completion': text, 'completion_token_count': count, 'completion_token_cap': 256,
           'phrase_matches': [] if start is None else [{'start': start}]}
    assert audit.response_flags(row, control) == expected


def test_sampling_disjoint_quotas_and_blinded_fields(population):
    root, rank, tensor, reference = population
    packet, blinded = audit.render_blinded_packet(root, rank, tensor_path=tensor, reference_path=reference)
    mapping = packet['mapping']
    assert len(mapping) == 56 and len({(row['condition_id'], row['row_id']) for row in mapping}) == 56
    assert [row['audit_id'] for row in mapping] == [f'A{i:03d}' for i in range(1, 57)]
    assert all(set(row) == {'audit_id', 'prompt', 'response'} for row in blinded)
    assert packet['strata']['real'] == {'available': 984, 'selected': 16, 'shortfall': 0}
    assert packet['strata']['baseline_random'] == {'available': 2214, 'selected': 16, 'shortfall': 0}
    for flag in audit.FLAGS:
        assert sum(row['primary'] == flag for row in mapping) == 6
    assert packet['available_population_weights']['real'] == 4 / 13


def test_empty_real_stratum_no_borrow_and_refill(population):
    root, rank, tensor, reference = population
    ledger = json.loads((root / 'ledger.json').read_bytes())
    baseline = ledger['attempts']['final-baseline']
    ledger['attempts'] = {'final-baseline': baseline}
    ledger['status'] = 'hard_stopped'
    path = Path(baseline[0]['completed']['path'])
    artifact = json.loads(path.read_bytes())
    for row in artifact['result']['harmless']:
        row.update(completion='Invented text', phrase_matches=[], refusal_proxy=False)
    _write(path, artifact)
    baseline[0]['completed']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(path.parent.parent / 'run-manifest.json', {'status': 'completed', 'settled': True,
           'configuration_sha256': 'synthetic-config', 'completed': baseline[0]['completed']})
    seal = {'reason': 'hard_stopped', 'configuration_sha256': 'synthetic-config',
            'completed': {'final-baseline': baseline[0]['completed']}}
    ledger['population_sha256'] = audit.json_digest(seal)
    _write(root / 'ledger.json', ledger)
    _write(root / 'population.json', seal)
    packet = audit.select_packet(root, rank, tensor_path=tensor, reference_path=reference)
    assert packet['strata']['real'] == {'available': 0, 'selected': 0, 'shortfall': 16}
    assert sum(row['arm'] == 'probability' for row in packet['mapping']) == 16
    assert sum(row['primary'] == 'at_cap' for row in packet['mapping']) == 6
    assert sum(row['primary'] == 'refill' for row in packet['mapping']) == 18


@pytest.mark.parametrize('mutation', ['seal_subset', 'incomplete_panel', 'changed_hash', 'active'])
def test_refuse_unsealed_or_partial_population(population, mutation):
    root, rank, tensor, reference = population
    ledger = json.loads((root / 'ledger.json').read_bytes())
    seal = json.loads((root / 'population.json').read_bytes())
    if mutation == 'seal_subset':
        seal['completed'].pop('final-pooled-real')
        ledger['population_sha256'] = audit.json_digest(seal)
    elif mutation in ('incomplete_panel', 'changed_hash'):
        entry = seal['completed']['final-pooled-real']
        path = Path(entry['path'])
        value = json.loads(path.read_bytes())
        value['result']['mmlu'].pop()
        _write(path, value)
        if mutation == 'incomplete_panel':
            entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            ledger['attempts']['final-pooled-real'][0]['completed'] = entry
            _write(path.parent.parent / 'run-manifest.json', {'status': 'completed', 'settled': True,
                   'configuration_sha256': 'synthetic-config', 'completed': entry})
            ledger['population_sha256'] = audit.json_digest(seal)
    else:
        ledger['status'] = 'active'
    _write(root / 'ledger.json', ledger)
    _write(root / 'population.json', seal)
    with pytest.raises(ValueError):
        audit.select_packet(root, rank, tensor_path=tensor, reference_path=reference)

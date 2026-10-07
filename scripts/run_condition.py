"""Rerun one frozen harmless or MMLU condition using portable release inputs.

New output uses a separate schema and is never an accepted historical campaign
record. Negative mode scores MMLU only; no harmful-request loader is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any, Literal

import torch

from report_tables import verify_inputs
from activation_steering_study.data import harmless, mmlu
from activation_steering_study.steering.harmless_execution import (
    ConditionSpec, Control, Target, TARGETS, TENSOR_SHA256, development_specs, final_specs,
)
from activation_steering_study.steering.harmless_run import evaluate_condition
from activation_steering_study.steering.negative_mmlu import evaluate_mmlu, shared_runtime
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen

TENSOR = Path('results/inputs/refusal-domains.pt')
REFERENCE = Path('results/provenance/refusal-scale-reference.json')
DOSES = Path('results/provenance/dose-freeze.json')


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(phase: Literal["development", "final", "negative-mmlu"], control: Control, target: Target | None, alpha: float | None) -> tuple[dict[str, Any], torch.Tensor | None, list[harmless.HarmlessItem], list[mmlu.MmluItem]]:
    """Validate admitted bytes, fixed condition membership, vector and complete panels."""
    manifest = verify_inputs()
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    if (file_sha(TENSOR) != TENSOR_SHA256 or reference['tensor_sha256'] != TENSOR_SHA256
            or reference['model_id'] != MODEL_ID or reference['model_revision'] != MODEL_REVISION
            or reference['model_dtype'] != 'torch.bfloat16' or reference['layer_index'] != 14
            or set(reference['direction_norms']) != set(TARGETS)):
        raise ValueError('Frozen scientific reference differs')
    doses = json.loads(DOSES.read_text(encoding='utf-8'))['doses']
    role: harmless.HarmlessRole = 'development' if phase == 'development' else 'final'
    if control == 'baseline':
        if target is not None or alpha not in (None, 0) or phase == 'negative-mmlu':
            raise ValueError('Baseline has no target or nonzero alpha; negative uses the saved historical baseline only')
        template = ConditionSpec(role, control)
        signed_alpha = 0.0
    else:
        if target is None:
            raise ValueError('A nonbaseline condition requires a target')
        if alpha is None:
            if phase == 'development':
                raise ValueError('Development requires an explicit tested alpha')
            alpha = doses[target] * (-1 if phase == 'negative-mmlu' else 1)
        signed_alpha = alpha
        template = ConditionSpec(role, control, target, -alpha if phase == 'negative-mmlu' else alpha)
    allowed = development_specs() if role == 'development' else final_specs(doses)
    if template not in allowed:
        raise ValueError('Condition is outside the frozen development/final queue')
    directions = torch.load(TENSOR, map_location='cpu', weights_only=True)['directions']
    if set(directions) != set(TARGETS):
        raise ValueError('Frozen vector mapping differs')
    for name, vector in directions.items():
        if (vector.dtype != torch.float32 or tuple(vector.shape) != (1536,) or not bool(torch.isfinite(vector).all())
                or not math.isclose(float(vector.norm()), reference['direction_norms'][name], rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError('Vector shape, values or raw norm differs')
    vector = None if target is None else directions[target].contiguous()
    raw_norm = 0.0 if vector is None else float(vector.norm())
    if template.random_seed is not None:
        vector = sample_random_direction(1536, raw_norm, template.random_seed)
    vector_norm = 0.0 if vector is None else float(vector.norm())
    vector_sha256 = None if vector is None else hashlib.sha256(vector.numpy().tobytes()).hexdigest()
    identity = {**template.to_dict(), 'phase': phase, 'alpha': signed_alpha,
                'condition_id': f'negative-mmlu-{target}-{control}' if phase == 'negative-mmlu' else template.condition_id}
    accepted = next(run for run in json.loads(Path('results/run-index.json').read_text(encoding='utf-8'))['runs']
                    if run['condition_id'] == identity['condition_id'])
    accepted_path = Path(accepted['projection'])
    direction_path = accepted_path.with_name(accepted_path.stem + '-request.json') if phase == 'negative-mmlu' else accepted_path
    accepted_entry = next(entry for entry in manifest['files'] if entry['path'] == str(accepted_path))
    direction_entry = next(entry for entry in manifest['files'] if entry['path'] == str(direction_path))
    direction_record = json.loads(direction_path.read_text(encoding='utf-8'))
    expected_direction = direction_record['direction']
    computed = {'vector_sha256':vector_sha256, 'raw_norm':raw_norm, 'vector_norm':vector_norm,
                'injected_norm':abs(signed_alpha)*vector_norm, 'alpha':signed_alpha,
                'target':target, 'random_seed':template.random_seed, 'tensor_sha256':TENSOR_SHA256}
    if (any(computed[key] != expected_direction[key] for key in computed)
            or any(identity[key] != direction_record['condition'][key]
                   for key in ['condition_id','control','role','target','random_seed','alpha'])
            or direction_record['configuration_sha256'] != accepted['configuration_sha256']
            or accepted_entry['original_sha256'] != accepted['original_run_manifest_sha256']):
        raise ValueError('Computed direction or condition differs from the accepted historical record')
    items = [] if phase == 'negative-mmlu' else harmless.load_harmless(role)
    questions = mmlu.sample_mmlu() if role == 'development' else mmlu.load_mmlu_final()
    expected_rows, expected_groups, expected_quality = harmless.COUNTS[role]
    if (len(questions) != (285 if role == 'development' else 1710)
            or (phase != 'negative-mmlu' and (len(items),len({item['semantic_group_id'] for item in items}),sum(item['quality_subset'] for item in items)) != (expected_rows,expected_groups,expected_quality))):
        raise ValueError('Complete frozen panel membership differs')
    metadata = {'schema_version':1,'record_kind':'release-rerun','condition':identity,
                'accepted_run':{'condition_id':accepted['condition_id'],'campaign':accepted['campaign'],
                                'configuration_sha256':accepted['configuration_sha256'],
                                'projection_path':str(accepted_path),'projection_sha256':accepted_entry['projection_sha256'],
                                'historical_original_sha256':accepted_entry['original_sha256'],
                                'direction_record_path':str(direction_path),
                                'direction_record_projection_sha256':direction_entry['projection_sha256'],
                                'direction_record_historical_original_sha256':direction_entry['original_sha256']},
                'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'layer_index':14,
                'generation':None if phase=='negative-mmlu' else {'max_new_tokens':256,'do_sample':False,'expected_repetition_penalty':1.1},
                'panels':{'harmless_rows':len(items),'harmless_groups':len({item['semantic_group_id'] for item in items}), 'mmlu_rows':len(questions)},
                'direction':{'tensor_sha256':TENSOR_SHA256,'reference_projection_sha256':file_sha(REFERENCE),
                             'reference_original_sha256':next(entry['original_sha256'] for entry in manifest['files'] if entry['path']==str(REFERENCE)),
                             'vector_sha256':vector_sha256,
                             'raw_norm':raw_norm,'vector_norm':vector_norm,'injected_norm':abs(signed_alpha)*vector_norm},
                'dose_freeze_projection_sha256':file_sha(DOSES),'release_manifest_sha256':file_sha(Path('results/manifest.json')),
                'uv_lock_sha256':file_sha(Path('uv.lock')),'python':platform.python_version(),
                'versions':{name:importlib.metadata.version(name) for name in ['numpy','pyarrow','scipy','torch','transformers']},
                'code_sha256':{str(path):file_sha(path) for path in sorted(Path('src/activation_steering_study').rglob('*.py')) if not path.name.startswith('test_')},
                'rerun_script_sha256':file_sha(Path('scripts/run_condition.py'))}
    return metadata,vector,items,questions


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['development','final','negative-mmlu'],required=True)
    parser.add_argument('--control',choices=['baseline','real','random42','random43'],required=True)
    parser.add_argument('--target',choices=TARGETS)
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--attempt-dir',type=Path)
    args=parser.parse_args()
    metadata,vector,items,questions=prepare(args.phase,args.control,args.target,args.alpha)
    if args.validate_only:
        print(json.dumps({'condition':metadata['condition'],'panels':metadata['panels'],'direction':metadata['direction']},indent=2))
        return
    if args.attempt_dir is None:
        parser.error('--attempt-dir is required for a new rerun')
    if (args.attempt_dir.is_absolute() or len(args.attempt_dir.parts) < 2
            or args.attempt_dir.parts[0] != 'artifacts' or '..' in args.attempt_dir.parts
            or not args.attempt_dir.resolve().is_relative_to(Path.cwd().resolve() / 'artifacts')):
        parser.error('--attempt-dir must be a new repository-relative directory under ignored artifacts/')
    args.attempt_dir.mkdir(parents=True,exist_ok=False)
    metadata['started_at']=datetime.now(timezone.utc).isoformat()
    with (args.attempt_dir/'metadata.json').open('x',encoding='utf-8') as stream:
        json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
    tokenizer,model=load_qwen()
    runtime=shared_runtime(tokenizer,model)
    if args.phase != 'negative-mmlu':
        penalty = model.generation_config.repetition_penalty
        if type(penalty) not in (int,float) or not math.isfinite(penalty) or penalty != metadata['generation']['expected_repetition_penalty']:
            raise ValueError('Loaded generation repetition_penalty differs from the expected 1.1')
        runtime['generation'] = {'max_new_tokens':256,'do_sample':False,'repetition_penalty':penalty}
    with (args.attempt_dir/'runtime.json').open('x',encoding='utf-8') as stream:
        json.dump(runtime,stream,indent=2,allow_nan=False);stream.write('\n')
    alpha=metadata['condition']['alpha']
    result: Any
    if args.phase=='negative-mmlu':
        assert vector is not None  # Negative baseline is refused during preparation.
        result=evaluate_mmlu(model,tokenizer,questions,vector,alpha)
    else:
        result=evaluate_condition(model,tokenizer,items,questions,vector,alpha,layer_index=14,max_new_tokens=256)
    with (args.attempt_dir/'result.json').open('x',encoding='utf-8') as stream:
        json.dump({'schema_version':1,'record_kind':'release-rerun','metadata_sha256':file_sha(args.attempt_dir/'metadata.json'),'runtime_sha256':file_sha(args.attempt_dir/'runtime.json'),'ended_at':datetime.now(timezone.utc).isoformat(),'result':result},stream,indent=2,allow_nan=False);stream.write('\n')
    print(f'Saved new rerun in {args.attempt_dir}')


if __name__=='__main__':
    main()

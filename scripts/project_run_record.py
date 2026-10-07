"""Project validated scientific run metadata and bind original/projection hashes.

Only the saved run schemas' scalar values, hash maps and path maps are admitted.
Free-form retry/error text, arguments, raw rows and environment maps are omitted.
Supply absolute roots as --path-root ORIGINAL=LOGICAL; malformed or unmapped paths
fail before output is written. This never changes an official execution record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
TARGETS = {'pooled', 'cyber_intrusion', 'dangerous_substances', 'disinformation'}
HASH_MAPS = {'input_sha256_before', 'input_sha256_after', 'inputs_before', 'inputs_after', 'output_sha256'}
CONDITION_FIELDS = {'condition_id', 'control', 'role', 'endpoint', 'layer_index', 'target', 'random_seed', 'alpha'}
DIRECTION_FIELDS = {'target', 'random_seed', 'alpha', 'raw_norm', 'vector_norm', 'injected_norm',
                    'tensor_sha256', 'reference_sha256', 'vector_sha256'}
PATH_PATTERN = r'[A-Za-z0-9_+.,:@-]+(?:/[A-Za-z0-9_+.,:@-]+)*'
VERSION_PATTERN = r'[0-9]+(?:\.[0-9]+){1,3}(?:\+cpu)?'


def _string(value: Any, pattern: str, field: str) -> str:
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        raise ValueError(f'{field} has an invalid scalar value')
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f'{field} must be an object with string keys')
    return value


def _number(value: Any, field: str, *, nonnegative: bool = False) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{field} must be a finite number')
    return value


def _path(value: Any, roots: dict[str, str]) -> str:
    if type(value) is not str:
        raise ValueError('Path fields must contain path strings only')
    if value.startswith('/'):
        for original, logical in sorted(roots.items(), key=lambda pair: len(pair[0]), reverse=True):
            if value == original or value.startswith(original + '/'):
                value = logical + value[len(original):]
                break
        else:
            raise ValueError('An absolute path lacks an explicit logical root')
    # Colons identify input aliases such as code:src/...; :/ would embed an
    # absolute path or URI inside a value that is supposed to be relative.
    if ':/' in value or any(part in {'.', '..'} for part in value.split('/')):
        raise ValueError('Path fields must be relative without traversal or embedded absolute paths')
    return _string(value, PATH_PATTERN, 'Path')


def project_record(record: dict[str, Any], roots: dict[str, str]) -> dict[str, Any]:
    record = _object(record, 'Run record')
    if 'record_kind' in record or ('condition' in record and isinstance(record['condition'], dict) and 'phase' in record['condition']):
        raise ValueError('Release rerun records are not historical run records and cannot be projected here')
    for original, logical in roots.items():
        if type(original) is not str or not original.startswith('/'):
            raise ValueError('Original path roots must be absolute strings')
        _path(original[1:], {})
        _path(logical, {})
    output: dict[str, Any] = {}
    for field, value in record.items():
        if field in HASH_MAPS or field == 'input_paths':
            entries = _object(value, field)
            projected: dict[str, str] = {}
            for key, item in entries.items():
                name = _path(key, roots)
                if name in projected:
                    raise ValueError('Logical paths collide')
                projected[name] = (_path(item, roots) if field == 'input_paths'
                                   else _string(item, r'[0-9a-f]{64}', field))
            output[field] = projected
        elif field in {'implementation_commit', 'planning_commit', 'configuration_sha256'}:
            output[field] = _string(value, r'[0-9a-f]{64}' if field == 'configuration_sha256' else r'[0-9a-f]{40}', field)
        elif field == 'commits':
            output[field] = {key: _string(item, r'[0-9a-f]{40}', 'Commit')
                             for key, item in _object(value, field).items() if key in {'implementation', 'planning'}}
        elif field == 'completed':
            entries = _object(value, field)
            output[field] = {key: (_path(item, roots) if key == 'path' else _string(item, r'[0-9a-f]{64}', 'Completed hash'))
                             for key, item in entries.items() if key in {'path', 'sha256'}}
        elif field in {'condition', 'direction'}:
            entries = _object(value, field)
            selected: dict[str, Any] = {}
            allowed = CONDITION_FIELDS if field == 'condition' else DIRECTION_FIELDS
            for key, item in entries.items():
                if key not in allowed:
                    continue
                if key == 'condition_id':
                    selected[key] = _string(item, r'(?:development|final|negative-mmlu)-(?:baseline|(?:pooled|cyber_intrusion|dangerous_substances|disinformation)-(?:real|random42|random43)(?:-a(?:0p5|1|2))?)', key)
                elif key in {'control', 'role', 'endpoint'}:
                    choices = {'control': {'baseline', 'real', 'random42', 'random43'},
                               'role': {'development', 'final'}, 'endpoint': {'mmlu'}}[key]
                    if type(item) is not str or item not in choices:
                        raise ValueError(f'{key} has an invalid identifier')
                    selected[key] = item
                elif key == 'target':
                    if item is not None and (type(item) is not str or item not in TARGETS):
                        raise ValueError('target has an invalid identifier')
                    selected[key] = item
                elif key == 'random_seed':
                    if item is not None and (type(item) is not int or item not in {42, 43}):
                        raise ValueError('random_seed must be 42, 43 or null')
                    selected[key] = item
                elif key == 'layer_index':
                    if type(item) is not int or item != 14:
                        raise ValueError('layer_index must identify block 14')
                    selected[key] = item
                elif key == 'alpha':
                    if _number(item, key) not in {-2, -1, -0.5, 0, 0.5, 1, 2}:
                        raise ValueError('alpha is outside the fixed dose grid')
                    selected[key] = item
                elif field == 'direction' and key in {'raw_norm', 'vector_norm', 'injected_norm'}:
                    selected[key] = _number(item, key, nonnegative=True)
                elif field == 'direction' and key in {'tensor_sha256', 'reference_sha256', 'vector_sha256'}:
                    selected[key] = None if item is None and key == 'vector_sha256' else _string(item, r'[0-9a-f]{64}', key)
            output[field] = selected
        elif field in {'runtime', 'observed_runtime'}:
            entries = _object(value, field)
            runtime: dict[str, Any] = {}
            for key, item in entries.items():
                if key == 'executable' and field == 'runtime':
                    runtime[key] = _path(item, roots)
                elif key == 'versions' and field == 'runtime':
                    runtime[key] = {name: _string(version, VERSION_PATTERN, 'Package version')
                                    for name, version in _object(item, key).items()
                                    if name in {'numpy', 'pyarrow', 'torch', 'transformers', 'scipy'}}
                elif key in {'python', 'torch_version', 'transformers_version'}:
                    runtime[key] = _string(item, VERSION_PATTERN, key)
                elif key in {'model_id', 'tokenizer_id'}:
                    if type(item) is not str or item != MODEL_ID:
                        raise ValueError(f'{key} must identify the pinned model')
                    runtime[key] = item
                elif key in {'model_revision', 'tokenizer_revision'}:
                    runtime[key] = _string(item, r'[0-9a-f]{40}', key)
                elif key in {'torch_interop_threads', 'torch_threads', 'torch_num_threads', 'torch_num_interop_threads', 'hidden_size'}:
                    if type(item) is not int or item < 1:
                        raise ValueError(f'{key} must be a positive integer')
                    runtime[key] = item
                elif key in {'device', 'model_device', 'dtype', 'model_dtype', 'attention_implementation'}:
                    choices = {'device': {'cpu'}, 'model_device': {'cpu'}, 'dtype': {'torch.bfloat16'},
                               'model_dtype': {'torch.bfloat16'}, 'attention_implementation': {'eager', 'sdpa'}}[key]
                    if type(item) is not str or item not in choices:
                        raise ValueError(f'{key} has an invalid runtime identifier')
                    runtime[key] = item
                elif key == 'training':
                    if type(item) is not bool:
                        raise ValueError('training must be boolean')
                    runtime[key] = item
            output[field] = runtime
        elif field in {'run_id', 'behaviour'}:
            if type(value) is not str or value not in {'honesty', 'sycophancy'}:
                raise ValueError(f'{field} has an invalid identifier')
            output[field] = value
        elif field in {'started_at', 'ended_at', 'start_utc', 'end_utc'}:
            output[field] = _string(value, r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:\+00:00|Z)', field)
        elif field in {'schema_version', 'attempt', 'returncode'}:
            if type(value) is not int or (field != 'returncode' and value < 1):
                raise ValueError(f'{field} must be an integer')
            output[field] = value
        elif field == 'elapsed_seconds':
            output[field] = _number(value, field, nonnegative=True)
        elif field in {'settled', 'inputs_unchanged', 'output_hashes_final'}:
            if type(value) is not bool:
                raise ValueError(f'{field} must be boolean')
            output[field] = value
        elif field == 'status':
            if type(value) is not str or value != 'completed':
                raise ValueError('Only completed run status is admitted')
            output[field] = value
        elif field == 'signals_received':
            if type(value) is not list or any(type(item) is not int or item < 1 for item in value):
                raise ValueError('signals_received must contain integer signal numbers')
            output[field] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/runs/final-pooled-real.json'))
    parser.add_argument('--output', type=Path, default=Path('build/run-projection.json'))
    parser.add_argument('--path-root', action='append', default=[], metavar='ORIGINAL=LOGICAL')
    args = parser.parse_args()
    roots = {}
    for entry in args.path_root:
        original, separator, logical = entry.partition('=')
        if not separator or not original.startswith('/') or not logical:
            parser.error('--path-root requires an absolute ORIGINAL and relative LOGICAL')
        roots[original.rstrip('/')] = logical.rstrip('/')
    input_bytes = args.input.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    historical_original_sha256 = None
    manifest_path = Path('results/manifest.json')
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        admitted = next((entry for entry in manifest['files']
                         if Path(entry['path']).resolve() == args.input.resolve()), None)
        if admitted is not None:
            from report_tables import verify_inputs
            verify_inputs(manifest_path)
            if input_sha256 != admitted['projection_sha256']:
                raise ValueError('Admitted input checksum differs')
            historical_original_sha256 = admitted['original_sha256']
    projected = project_record(json.loads(input_bytes), roots)
    body = (json.dumps(projected, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    result = {'schema_version': 1, 'input_sha256': input_sha256,
              'projection_sha256': hashlib.sha256(body).hexdigest(), 'projection': projected,
              'projection_hash_encoding': 'UTF-8 indented JSON plus one newline; projection member only'}
    if historical_original_sha256 is not None:
        result['historical_original_sha256'] = historical_original_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(f'Wrote {args.output}; input and projection hashes recorded')


if __name__ == '__main__':
    main()

"""Verify accepted aggregate inputs and render the released tables without model calls.

Published BCa interval bounds are inputs. This command does not recompute intervals
or introduce a new statistical decision. Run from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = Path('results/manifest.json')


def verify_inputs(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported aggregate manifest')
    paths = set()
    for entry in manifest['files']:
        path = Path(entry['path'])
        if path.is_absolute() or '..' in path.parts or str(path) in paths:
            raise ValueError('Manifest paths must be unique repository-relative paths')
        paths.add(str(path))
        data = path.read_bytes()
        if len(data) != entry['size_bytes'] or hashlib.sha256(data).hexdigest() != entry['projection_sha256']:
            raise ValueError(f'Accepted input checksum differs: {path}')
    return manifest


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _table(title: str, columns: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = [f'## {title}', '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines.extend('| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows)
    return '\n'.join(lines) + '\n\n'


def _interval(row: dict[str, str], endpoint: str) -> str:
    estimate = float(row[f'{endpoint}_delta']) * 100
    if row[f'{endpoint}_interval_status'] != 'nominal':
        return f'{estimate:+.2f} (interval unavailable: {row[f"{endpoint}_interval_reason"]})'
    lower, upper = (float(row[f'{endpoint}_ci_{bound}']) * 100 for bound in ['low', 'high'])
    return f'{estimate:+.2f} [{lower:+.2f}, {upper:+.2f}]'


def render_tables() -> str:
    verify_inputs()
    text = '# Released aggregate results\n\nAll interval bounds below are accepted published inputs. Changes and interval bounds are in percentage points unless stated otherwise. Intervals are nominal; they do not establish equivalence or absence of effects.\n\n'
    development = read_csv('results/development/development.csv')
    text += _table('Development: fixed dose grid', ['Condition', 'Refusal count / 63', 'Refusal change (pp)', 'MMLU change (pp)'], [[r['condition_id'], r['refusal_count'], f"{float(r['refusal_paired_change'])*100:+.2f}", f"{float(r['mmlu_paired_change'])*100:+.2f}"] for r in development])
    selected = [r for r in read_csv('results/development/dose-selection.csv') if r['selected'] == 'True']
    text += _table('Frozen development doses', ['Target', 'Alpha', 'Count distance to pooled / 63', 'Signed gap to pooled (pp)'], [[r['target'],r['alpha'],r['absolute_count_distance'],f"{float(r['signed_domain_minus_pooled'])*100:+.2f}"] for r in selected])
    positive = read_csv('results/positive/final-contrasts.csv')
    text += _table('Positive final: all 15 paired contrasts', ['Comparison', 'Refusal change [95% interval] (pp)', 'MMLU change [95% interval] (pp)', 'Refusal support'], [[r['comparison_id'],_interval(r,'harmless'),_interval(r,'mmlu'),r['harmless_supported']] for r in positive])
    negative = read_csv('results/negative/negative-contrasts.csv')
    text += _table('Negative MMLU: all 15 paired contrasts', ['Comparison', 'MMLU change [95% interval] (pp)'], [[r['comparison_id'],_interval(r,'mmlu')] for r in negative])
    text += 'Negative conditions score MMLU only and share the saved positive-final baseline. They provide no held-out harmful-request suppression result.\n\n'
    for behaviour in ['honesty','sycophancy']:
        data=json.loads(Path(f'results/ab/{behaviour}-final.json').read_text())
        text += _table(f'{behaviour.capitalize()}: final A/B contrasts', ['Real condition','Reference','Change (pp)','95% interval (pp)','Status'], [[r['real_condition'],r['reference_condition'],f"{r['delta_pp']:+.2f}",f"[{r['interval_pp'][0]:+.2f}, {r['interval_pp'][1]:+.2f}]",r['status']] for r in data['contrasts']])
        data=json.loads(Path(f'results/control/{behaviour}-qualification.json').read_text())
        text += _table(f'{behaviour.capitalize()}: block-14 qualification', ['Cell','Signed real change (pp)','Signed R42 change (pp)','Signed R43 change (pp)','Point gate passes'], [[key,f"{r['evidence']['signed_real_change']*100:+.2f}",f"{r['evidence']['signed_random_changes']['42']*100:+.2f}",f"{r['evidence']['signed_random_changes']['43']*100:+.2f}",r['evidence']['passes']] for key,r in data['cells'].items()])
    text += 'The qualification gate did not establish suitability; the unrelated-behaviour control was omitted from the new refusal comparison.\n\n'
    for filename,title in [('geometry-cosines.csv','Descriptive direction cosines'),('geometry-split-halves.csv','Descriptive split-half stability')]:
        rows=read_csv('results/development/'+filename)
        text+=_table(title,list(rows[0]),[list(r.values()) for r in rows])
    rows=read_csv('results/audit/audit-summary.csv')
    columns=['layer','sample','n','known_agreement','unclear','known_disagreement','refusal_full','refusal_mixed','refusal_none','refusal_unclear']
    text+=_table('Proxy audit: original and informed layers',columns,[[r[k] for k in columns] for r in rows])
    text+='The informed layer is the same owner\'s post-hoc adjudication. Neither layer is independent gold; targeted units do not estimate prevalence. The release includes aggregate labels only.\n'
    return text


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/tables.md'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        manifest=verify_inputs()
        print(f"Verified {len(manifest['files'])} accepted input checksums")
        return
    tables=render_tables()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(tables,encoding='utf-8')
    print(f'Wrote {args.output}; accepted intervals preserved')


if __name__=='__main__':
    main()

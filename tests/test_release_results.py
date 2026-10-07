"""Publication projection and portable fixed-condition entrypoint checks; no model calls."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def release_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    return tuple(importlib.import_module(name) for name in ['report_tables','project_run_record','run_condition'])


def test_accepted_input_checksums_and_complete_run_index(release_modules):
    tables, _, _ = release_modules
    manifest = tables.verify_inputs()
    assert len(manifest['files']) >= 90, 'Every admitted evidence projection must be checked'
    runs=json.loads(Path('results/run-index.json').read_text())['runs']
    assert len(runs)==36 and len({run['condition_id'] for run in runs})==36
    assert sum(run['condition']['role']=='development' for run in runs)==11
    assert sum(run['campaign']=='negative' for run in runs)==12
    for run in runs:
        record=json.loads(Path(run['projection']).read_text())
        assert record['status']=='completed' and record['returncode']==0 and record['settled'] is True
        assert record['completed']['sha256']==run['completed_sha256']


def test_changed_accepted_input_is_rejected(release_modules,monkeypatch,tmp_path):
    tables,_,_=release_modules
    monkeypatch.chdir(tmp_path)
    Path('aggregate.csv').write_bytes(b'changed\n')
    manifest={'schema_version':1,'files':[{'path':'aggregate.csv','size_bytes':8,'projection_sha256':hashlib.sha256(b'original').hexdigest()}]}
    Path('manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum differs'):
        tables.verify_inputs(Path('manifest.json'))


def test_table_output_preserves_nominal_intervals_and_two_audit_layers(release_modules):
    tables,_,_=release_modules
    text=tables.render_tables()
    assert '+48.57 [+42.45, +55.10]' in text and 'all 15 paired contrasts' in text
    assert 'original' in text and 'informed' in text
    assert 'published inputs' in text and 'no held-out harmful-request suppression' in text


def test_run_projection_excludes_raw_and_unknown_nested_fields(release_modules):
    _,projection,_=release_modules
    source={'condition':{'condition_id':'final-pooled-real','alpha':1.0,'completion':'private response','secret':'private token'},
            'runtime':{'python':'3.14.4','environment':{'TOKEN':'private token'},'versions':{'torch':'2.14.0+cpu','SECRET':'private token'}},
            'completed':{'path':'/private/runs/completed.json','sha256':'a'*64,'completion':'private response'},
            'prompt':'private instruction','response':'private response','argv':['--prompt','private instruction']}
    result=projection.project_record(source,{'/private/runs':'unpublished/positive'})
    serialized=json.dumps(result)
    assert 'private response' not in serialized and 'private instruction' not in serialized and 'private token' not in serialized
    assert result['completed']['path']=='unpublished/positive/completed.json'
    assert result['completed']['sha256']=='a'*64
    assert result['runtime']['versions']=={'torch':'2.14.0+cpu'}


def test_unmapped_absolute_path_and_projection_collision_are_rejected(release_modules):
    _,projection,_=release_modules
    with pytest.raises(ValueError,match='absolute path'):
        projection.project_record({'completed':{'path':'/unknown/completed.json','sha256':'a'*64}}, {})
    with pytest.raises(ValueError,match='collide'):
        projection.project_record({'input_sha256_before':{'/one/a':'a'*64,'/two/a':'b'*64}}, {'/one':'logical','/two':'logical'})


@pytest.mark.parametrize('target',['pooled','cyber_intrusion','dangerous_substances','disinformation'])
@pytest.mark.parametrize('control',['real','random42','random43'])
def test_portable_direction_matches_accepted_vector_identity(release_modules,target,control):
    _,_,rerun=release_modules
    metadata,vector,items,questions=rerun.prepare('final',control,target,None)
    accepted=json.loads(Path(f'results/runs/final-{target}-{control}.json').read_text())
    assert metadata['direction']['vector_sha256']==accepted['direction']['vector_sha256'], 'Portable direction must match the originally injected bytes'
    assert metadata['direction']['raw_norm']==accepted['direction']['raw_norm']
    assert vector is not None and len(items)==246 and len(questions)==1710
    negative,negative_vector,negative_items,negative_questions=rerun.prepare('negative-mmlu',control,target,None)
    assert negative['condition']['alpha']==-1 and negative['direction']['vector_sha256']==metadata['direction']['vector_sha256']
    assert negative_items==[] and len(negative_questions)==1710 and negative_vector is not None


@pytest.mark.parametrize('phase,control,target,alpha',[
    ('final','real','pooled',2),('final','real','pooled',-1),
    ('negative-mmlu','real','pooled',1),('negative-mmlu','baseline',None,None),
    ('development','random42','pooled',1),('development','real','pooled',0.5),
    ('final','baseline','pooled',0),('final','real',None,1),
])
def test_invalid_frozen_condition_is_rejected(release_modules,phase,control,target,alpha):
    _,_,rerun=release_modules
    with pytest.raises(ValueError):
        rerun.prepare(phase,control,target,alpha)


def test_rerun_calls_existing_kernel_and_refuses_overwrite(release_modules,monkeypatch,tmp_path):
    _,_,rerun=release_modules
    prepared=rerun.prepare('final','real','pooled',None)
    monkeypatch.setattr(rerun,'prepare',lambda *args:prepared)
    monkeypatch.chdir(tmp_path)
    output=Path('artifacts/attempt')
    calls=[]
    model=SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.1))
    monkeypatch.setattr(rerun,'load_qwen',lambda:('tokenizer',model))
    monkeypatch.setattr(rerun,'shared_runtime',lambda tokenizer,model:{'synthetic':True})
    def kernel(model,tokenizer,items,questions,vector,alpha,**kwargs):
        calls.append((model,tokenizer,len(items),len(questions),alpha,kwargs))
        return {'synthetic':True}
    monkeypatch.setattr(rerun,'evaluate_condition',kernel)
    monkeypatch.setattr(sys,'argv',['run_condition.py','--phase','final','--control','real','--target','pooled','--attempt-dir',str(output)])
    rerun.main()
    assert calls==[(model,'tokenizer',246,1710,1.0,{'layer_index':14,'max_new_tokens':256})]
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['record_kind']=='release-rerun' and metadata['model_revision']==rerun.MODEL_REVISION
    assert metadata['direction']['reference_original_sha256']!=metadata['direction']['reference_projection_sha256']
    assert json.loads((output/'runtime.json').read_text())['generation']['repetition_penalty']==1.1
    with pytest.raises(FileExistsError):
        rerun.main()
    assert len(calls)==1, 'Existing attempt must be refused before any kernel call'


def test_projection_cli_accepts_real_saved_manifest_once(release_modules,monkeypatch,tmp_path):
    _,projection,_=release_modules
    output=tmp_path/'record.json'
    monkeypatch.setattr(sys,'argv',['project_run_record.py','--output',str(output)])
    projection.main()
    result=json.loads(output.read_text())
    source=Path('results/runs/final-pooled-real.json')
    admitted=next(entry for entry in json.loads(Path('results/manifest.json').read_text())['files'] if entry['path']==str(source))
    assert result['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()==admitted['projection_sha256']
    assert result['historical_original_sha256']==admitted['original_sha256']!=result['input_sha256']
    assert 'original_sha256' not in result
    assert result['projection']['condition']['condition_id']=='final-pooled-real'
    assert result['projection']['output_hashes_final'] is True
    encoded=(json.dumps(result['projection'],indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    assert hashlib.sha256(encoded).hexdigest()==result['projection_sha256']
    with pytest.raises(FileExistsError):
        projection.main()


@pytest.mark.parametrize('field',['model_revision','raw_norm'])
def test_scientific_reference_identity_and_norm_are_checked(release_modules,monkeypatch,tmp_path,field):
    _,_,rerun=release_modules
    value=json.loads(rerun.REFERENCE.read_text())
    if field=='model_revision':
        value['model_revision']='0'*40
    else:
        value['direction_norms']['pooled']*=2
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps(value))
    monkeypatch.setattr(rerun,'REFERENCE',reference)
    with pytest.raises(ValueError,match='reference differs|raw norm differs'):
        rerun.prepare('final','real','pooled',None)


@pytest.mark.parametrize('record,rejected',[
    ({'input_paths':{'artifact':{'response':'synthetic private response'}}},True),
    ({'retry_reason':{'raw_prompt':'synthetic private instruction'}},False),
    ({'runtime':{'model_id':{'secret':'synthetic private token'}}},True),
    ({'retry_reason':'failed reading /private/example.json'},False),
])
def test_reproduced_projection_leaks_are_rejected_or_omitted(release_modules,record,rejected):
    _,projection,_=release_modules
    if rejected:
        with pytest.raises(ValueError):
            projection.project_record(record,{})
    else:
        assert projection.project_record(record,{})=={}, 'Free-form diagnostics must be omitted without inspecting their payload'


@pytest.mark.parametrize('record',[
    {'condition':{'alpha':{'response':'synthetic private response'}}},
    {'direction':{'raw_norm':['synthetic private response']}},
    {'completed':{'sha256':{'response':'synthetic private response'}}},
    {'commits':{'implementation':{'raw_prompt':'synthetic private instruction'}}},
    {'runtime':{'python':{'raw_prompt':'synthetic private instruction'}}},
    {'runtime':{'versions':{'torch':{'secret':'synthetic private token'}}}},
    {'runtime':{'model_id':'ordinary prose instead of a model identifier'}},
    {'runtime':{'executable':'failed reading /private/example.json'}},
    {'input_paths':{'artifact':'location:/private/example.json'}},
    {'input_paths':['synthetic private response']},
    {'inputs_before':{'artifact':{'secret':'synthetic private token'}}},
    {'elapsed_seconds':{'secret':'synthetic private token'}},
    {'signals_received':[{'response':'synthetic private response'}]},
    {'status':{'raw_prompt':'synthetic private instruction'}},
    {'started_at':'failed reading /private/example.json'},
    {'condition':{'alpha':float('nan')}},
    {'settled':1},
])
def test_projection_rejects_malformed_known_field_shapes(release_modules,record):
    _,projection,_=release_modules
    with pytest.raises(ValueError):
        projection.project_record(record,{})


def test_projection_accepts_all_36_saved_condition_records(release_modules):
    _,projection,_=release_modules
    runs=json.loads(Path('results/run-index.json').read_text())['runs']
    assert len(runs)==36
    for run in runs:
        original=json.loads(Path(run['projection']).read_text())
        projected=projection.project_record(original,{})
        assert projected=={key:value for key,value in original.items() if key not in {'argv','retry_reason'}}, 'Valid scientific fields and numerical values must survive the publication projection'


@pytest.mark.parametrize('existing_output',[False,True])
def test_projection_cli_rejects_before_writing_output(release_modules,monkeypatch,tmp_path,existing_output):
    _,projection,_=release_modules
    source=tmp_path/'input.json'
    source.write_text(json.dumps({'runtime':{'model_id':{'secret':'synthetic private token'}}}))
    output=tmp_path/'output'/'projection.json'
    if existing_output:
        output.parent.mkdir()
        output.write_bytes(b'existing accepted file\n')
    monkeypatch.setattr(sys,'argv',['project_run_record.py','--input',str(source),'--output',str(output)])
    with pytest.raises(ValueError):
        projection.main()
    if existing_output:
        assert output.read_bytes()==b'existing accepted file\n'
    else:
        assert not output.parent.exists(), 'Rejected input must not create an output directory'


@pytest.mark.parametrize('run',json.loads(Path('results/run-index.json').read_text())['runs'],ids=lambda run:run['condition_id'])
def test_all_36_rerun_conditions_bind_admitted_direction_and_source(release_modules,run):
    _,_,rerun=release_modules
    condition=run['condition']
    phase='negative-mmlu' if run['campaign']=='negative' else condition['role']
    metadata,_,_,_=rerun.prepare(phase,condition['control'],condition['target'],condition['alpha'])
    bound=metadata['accepted_run']
    assert bound['condition_id']==run['condition_id']
    assert bound['historical_original_sha256']==run['original_run_manifest_sha256']
    assert bound['projection_sha256']==rerun.file_sha(Path(run['projection']))
    assert bound['configuration_sha256']==run['configuration_sha256']
    direction_source=json.loads(Path(bound['direction_record_path']).read_text())['direction']
    for key in ['vector_sha256','raw_norm','vector_norm','injected_norm']:
        assert metadata['direction'][key]==direction_source[key]


@pytest.mark.parametrize('phase',['final','negative-mmlu'])
def test_changed_vector_with_same_norm_is_refused_before_model(release_modules,monkeypatch,phase):
    _,_,rerun=release_modules
    sample=rerun.sample_random_direction
    def wrong_vector(*args):
        original=sample(*args)
        mutated=-original
        assert float(mutated.norm())==float(original.norm()), 'Mutant must preserve the exact norm'
        assert rerun.hashlib.sha256(mutated.numpy().tobytes()).hexdigest()!=rerun.hashlib.sha256(original.numpy().tobytes()).hexdigest()
        return mutated
    monkeypatch.setattr(rerun,'sample_random_direction',wrong_vector)
    monkeypatch.setattr(rerun,'load_qwen',lambda:pytest.fail('Preparation must not load a model'))
    with pytest.raises(ValueError,match='accepted historical record'):
        rerun.prepare(phase,'random42','pooled',None)


@pytest.mark.parametrize('record',[{'record_kind':'release-rerun','condition':{'condition_id':'final-pooled-real'}},
                                   {'condition':{'condition_id':'final-pooled-real','phase':'final'}}])
@pytest.mark.parametrize('existing_output',[False,True])
def test_projector_refuses_rerun_inputs_without_stripping_marker(release_modules,monkeypatch,tmp_path,record,existing_output):
    _,projection,_=release_modules
    source=tmp_path/'rerun.json'
    source.write_text(json.dumps(record))
    output=tmp_path/'output'/'projection.json'
    if existing_output:
        output.parent.mkdir()
        output.write_bytes(b'unchanged output\n')
    monkeypatch.setattr(sys,'argv',['project_run_record.py','--input',str(source),'--output',str(output)])
    with pytest.raises(ValueError,match='not historical run records'):
        projection.main()
    assert output.read_bytes()==b'unchanged output\n' if existing_output else not output.parent.exists()


def test_projector_external_input_hash_is_not_historical_identity(release_modules,monkeypatch,tmp_path):
    _,projection,_=release_modules
    source=tmp_path/'input.json'
    source.write_bytes(b'{"condition":{"condition_id":"final-pooled-real"}}\n')
    output=tmp_path/'projection.json'
    monkeypatch.setattr(sys,'argv',['project_run_record.py','--input',str(source),'--output',str(output)])
    projection.main()
    result=json.loads(output.read_text())
    assert result['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert 'historical_original_sha256' not in result and 'original_sha256' not in result


def test_projector_checks_admitted_input_before_output_creation(release_modules,monkeypatch,tmp_path):
    _,projection,_=release_modules
    monkeypatch.chdir(tmp_path)
    Path('results').mkdir()
    source=Path('results/admitted.json')
    admitted_bytes=b'{"condition":{"condition_id":"final-pooled-real"}}\n'
    source.write_bytes(admitted_bytes+b' ')
    Path('results/manifest.json').write_text(json.dumps({'schema_version':1,'files':[
        {'path':str(source),'size_bytes':len(admitted_bytes),'projection_sha256':hashlib.sha256(admitted_bytes).hexdigest(),'original_sha256':'a'*64}]}))
    monkeypatch.setattr(sys,'argv',['project_run_record.py','--input',str(source),'--output','build/output.json'])
    with pytest.raises(ValueError,match='checksum differs'):
        projection.main()
    assert not Path('build').exists()


@pytest.mark.parametrize('penalty',[1.0,None,float('nan')])
def test_rerun_checks_loaded_repetition_penalty_before_evaluation(release_modules,monkeypatch,tmp_path,penalty):
    _,_,rerun=release_modules
    prepared=rerun.prepare('final','real','pooled',None)
    monkeypatch.setattr(rerun,'prepare',lambda *args:prepared)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rerun,'load_qwen',lambda:('tokenizer',SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=penalty))))
    monkeypatch.setattr(rerun,'shared_runtime',lambda *args:{'synthetic':True})
    monkeypatch.setattr(rerun,'evaluate_condition',lambda *args,**kwargs:pytest.fail('Mismatched generation settings must not be evaluated'))
    monkeypatch.setattr(sys,'argv',['run_condition.py','--phase','final','--control','real','--target','pooled','--attempt-dir','artifacts/attempt'])
    with pytest.raises(ValueError,match='repetition_penalty differs'):
        rerun.main()
    assert not Path('artifacts/attempt/runtime.json').exists() and not Path('artifacts/attempt/result.json').exists()


@pytest.mark.parametrize('output',['results/unreviewed','artifacts/../results/unreviewed','build/unreviewed','artifacts'])
def test_rerun_output_must_be_under_ignored_artifacts(release_modules,monkeypatch,tmp_path,output):
    _,_,rerun=release_modules
    prepared=rerun.prepare('final','real','pooled',None)
    monkeypatch.setattr(rerun,'prepare',lambda *args:prepared)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rerun,'load_qwen',lambda:pytest.fail('Invalid output directory must be refused before model loading'))
    monkeypatch.setattr(sys,'argv',['run_condition.py','--phase','final','--control','real','--target','pooled','--attempt-dir',output])
    with pytest.raises(SystemExit):
        rerun.main()
    assert not Path(output).exists()


def test_rerun_artifacts_symlink_escape_is_refused(release_modules,monkeypatch,tmp_path):
    _,_,rerun=release_modules
    prepared=rerun.prepare('final','real','pooled',None)
    monkeypatch.setattr(rerun,'prepare',lambda *args:prepared)
    monkeypatch.chdir(tmp_path)
    Path('elsewhere').mkdir()
    Path('artifacts').symlink_to('elsewhere',target_is_directory=True)
    monkeypatch.setattr(rerun,'load_qwen',lambda:pytest.fail('Escaping output must be refused before model loading'))
    monkeypatch.setattr(sys,'argv',['run_condition.py','--phase','final','--control','real','--target','pooled','--attempt-dir','artifacts/attempt'])
    with pytest.raises(SystemExit):
        rerun.main()
    assert not Path('elsewhere/attempt').exists()

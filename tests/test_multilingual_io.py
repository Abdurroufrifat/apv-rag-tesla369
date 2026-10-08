import json
from pathlib import Path
import pytest


def test_temporary_windows_replace_denial_retries_without_losing_old_file(tmp_path,monkeypatch):
    from apv_rag.multilingual_io import write_json_atomic
    path=tmp_path/'cache.json';path.write_text('{"old":true}')
    original=Path.replace;attempts=[]
    def replace(source,target):
        attempts.append(1)
        if len(attempts)<3:
            assert json.loads(path.read_text())=={'old':True}
            raise PermissionError(13,'Access is denied')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr('apv_rag.multilingual_io.time.sleep',lambda delay:None)
    write_json_atomic(path,{'new':[1,2]})
    assert len(attempts)==3 and json.loads(path.read_text())=={'new':[1,2]}
    assert not path.with_suffix('.json.partial').exists()


def test_persistent_access_denial_stops_and_preserves_valid_checkpoint(tmp_path,monkeypatch):
    from apv_rag.multilingual_io import write_json_atomic
    path=tmp_path/'cache.json';path.write_text('{"old":true}')
    def denied(*args):raise PermissionError(13,'Access is denied')
    monkeypatch.setattr(Path,'replace',denied)
    monkeypatch.setattr('apv_rag.multilingual_io.time.sleep',lambda delay:None)
    with pytest.raises(PermissionError,match='cache.json'):write_json_atomic(path,{'new':True})
    assert json.loads(path.read_text())=={'old':True}


def test_scoped_english_writer_restored_even_after_failure(monkeypatch,tmp_path):
    from apv_rag.multilingual_io import infer_english,write_json_atomic
    import run_fresh_pipeline
    original=run_fresh_pipeline.write_json_atomic
    def fake(*args,**kwargs):
        assert run_fresh_pipeline.write_json_atomic is write_json_atomic
        raise RuntimeError('inference error')
    monkeypatch.setattr(run_fresh_pipeline,'infer_missing',fake)
    with pytest.raises(RuntimeError):infer_english({}, {}, tmp_path/'cache.json',root=tmp_path)
    assert run_fresh_pipeline.write_json_atomic is original


def early_checkpoint(tmp_path):
    import shutil
    from test_multilingual_pipeline import saved_controller_fixture
    from apv_rag.splits import write_json_atomic
    from run_multilingual_retrieved_pipeline import original_identity
    produced=tmp_path/'fixture';produced.mkdir()
    rows,_,_,_=saved_controller_fixture(produced)
    out=tmp_path/'interrupted';out.mkdir()
    for name in ('prepared_contexts.json','feature_cache.json'):shutil.copyfile(produced/name,out/name)
    meta={'settings':{'seed':369},'code_sha256':{'src/apv_rag/multilingual_io.py':'new-io',
        'scripts/run_multilingual_retrieved_pipeline.py':'new-runner','unchanged.py':'same'},
        'protocol_sha256':'new-protocol','reference_sha256':'same-ref','sample_manifest_sha256':'same-sample'}
    write_json_atomic(out/'input_manifest.json',original_identity(meta))
    (out/'feature_cache.json.partial').write_text('unfinished temporary JSON')
    return out,rows,meta


def test_exact_interrupted_run_migration_preserves_existing_features(tmp_path):
    from run_multilingual_retrieved_pipeline import accept_resume,check_resume_receipt
    out,rows,meta=early_checkpoint(tmp_path)
    before=(out/'feature_cache.json').read_bytes()
    accept_resume(out,meta,rows);check_resume_receipt(out,meta)
    assert json.loads((out/'input_manifest.json').read_text())==meta
    assert json.loads((out/'io_resume_receipt.json').read_text())['validated_english_feature_records']==2
    assert (out/'feature_cache.json').read_bytes()==before
    accept_resume(out,meta,rows)


def test_migration_rejects_changed_settings_and_later_stage_cache(tmp_path):
    from run_multilingual_retrieved_pipeline import accept_resume
    out,rows,meta=early_checkpoint(tmp_path)
    original=(out/'input_manifest.json').read_bytes()
    with pytest.raises(ValueError,match='beyond'):accept_resume(out,dict(meta,settings={'seed':1}),rows)
    assert (out/'input_manifest.json').read_bytes()==original
    (out/'generation_cache.sqlite').touch()
    with pytest.raises(ValueError,match='English-feature stage'):accept_resume(out,meta,rows)

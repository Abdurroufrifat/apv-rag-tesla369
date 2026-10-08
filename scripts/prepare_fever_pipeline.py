"""Freeze new development/confirmation claims for the complete fixed FEVER pipeline."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from apv_rag.fever_calibration import normalized, partition_claims
from apv_rag.fever_nli import read_jsonl
from apv_rag.splits import sha256, write_json_atomic
from prepare_fever_calibration import SOURCE_SHA256, write_lines


def main():
    out = ROOT / 'data/external/fever/pipeline_v1'
    if out.exists():
        raise FileExistsError('Pipeline cohorts already frozen')
    previous = ROOT / 'data/external/fever/calibration_v1'
    hashes = json.loads((previous / 'output_manifest.json').read_text())
    for name, digest in hashes.items():
        if sha256(previous / name) != digest:
            raise ValueError('Previous cohort identity changed')
    old = json.loads((previous / 'selection_manifest.json').read_text())
    observed = read_jsonl(previous / 'development/model_inputs.jsonl') + read_jsonl(previous / 'confirmation/model_inputs.jsonl')
    excluded_ids = set(old['excluded_prior_ids']) | {r['id'] for r in observed}
    excluded_texts = set(old['excluded_prior_texts']) | {normalized(r['claim']) for r in observed}
    raw = ROOT / 'data/external/fever/heldout_v1/shared_task_dev.jsonl'
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError('FEVER source identity changed')
    dev, confirm = partition_claims(read_jsonl(raw), exclude_ids=excluded_ids,
                                    exclude_texts=excluded_texts, development_count=300, confirmation_count=300)
    out.mkdir(parents=True)
    cohorts = {}
    for name, selected in (('development', dev), ('confirmation', confirm)):
        folder = out / name
        folder.mkdir()
        write_lines(folder / 'model_inputs.jsonl', [{'id': r['id'], 'claim': r['claim']} for r in selected])
        write_lines(folder / 'gold.jsonl', [{'id': r['id'], 'label': r['label'], 'evidence': r['evidence']} for r in selected])
        cohorts[name] = {'claims': 300, 'selected_ids': [r['id'] for r in selected],
                         'model_inputs_sha256': sha256(folder / 'model_inputs.jsonl'),
                         'gold_sha256': sha256(folder / 'gold.jsonl')}
    write_json_atomic(out / 'selection_manifest.json', {
        'source_sha256': SOURCE_SHA256, 'prior_calibration_manifest_sha256': sha256(previous / 'selection_manifest.json'),
        'selection': 'continue fixed SHA-256 ranking after excluding all prior FEVER/XFEVER IDs and normalized texts',
        'cohorts': cohorts, 'excluded_ids': sorted(excluded_ids), 'excluded_texts': sorted(excluded_texts),
        'labels_or_gold_evidence_used_for_selection': False,
        'limitation': 'Same public FEVER development dataset; page and paraphrase overlap not excluded.'})
    write_json_atomic(out / 'output_manifest.json', {p.relative_to(out).as_posix(): sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file()})
    # Compact immutable model/gate reference; does not include model weights.
    meta = json.loads((ROOT / 'artifacts/fresh_pipeline_received/input_manifest.json').read_text())
    heads = ROOT / 'artifacts/semantic_sufficiency_received/models.json'
    receipt = json.loads((heads.parent / 'output_manifest.json').read_text())
    if sha256(heads) != receipt['models.json']:
        raise ValueError('Frozen gate heads changed')
    ref = ROOT / 'config/fever_pipeline_reference_v1.json'
    write_json_atomic(ref, {'generator': meta['generator'], 'feature_models': meta['feature_models'],
                           'packages': meta['packages'], 'gate_models': json.loads(heads.read_text()),
                           'prior_pipeline_identity_sha256': sha256(ROOT / 'artifacts/fresh_pipeline_received/input_manifest.json')})
    print('Pipeline cohort manifest SHA-256:', sha256(out / 'selection_manifest.json'))
    print('Pipeline model/gate reference SHA-256:', sha256(ref))


if __name__ == '__main__':
    main()

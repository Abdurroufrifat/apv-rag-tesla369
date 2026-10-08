"""Open frozen FEVER gold only after prediction checks, then score without fitting."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.fever_nli import (GOLD_SHA256, load_claims, predict_context, read_jsonl,
                              score_predictions, validate_contexts, verify_prediction_files)
from apv_rag.splits import sha256, write_json_atomic


def main():
    output = ROOT / 'artifacts/fever_nli_scoring_v1'
    if output.exists():
        raise FileExistsError('FEVER scoring output exists; refusing overwrite')
    run = ROOT / 'artifacts/fever_nli_v1'
    identity, predictions = verify_prediction_files(run)
    source = ROOT / 'data/external/fever/heldout_v1'
    claims = load_claims(source)
    retrieval = ROOT / 'artifacts/fever_retrieval_v1'
    contexts = retrieval / 'contexts.jsonl'
    if (sha256(contexts) != identity['contexts_sha256'] or
            sha256(retrieval / 'input_manifest.json') != identity['retrieval_receipt_sha256'] or
            sha256(source / 'model_inputs.jsonl') != identity['model_inputs_sha256'] or
            identity['claims'] != 300 or len(predictions) != 300):
        raise ValueError('Frozen run/input identity mismatch')
    rows = validate_contexts(claims, read_jsonl(contexts))
    for row, pred in zip(rows, predictions, strict=True):
        if pred != predict_context(row, pred['nli_probabilities_cen']):
            raise ValueError('Saved prediction/evidence alignment mismatch')
    # This is the first gold access in the inference + scoring workflow.
    gold_path = source / 'gold.jsonl'
    if sha256(gold_path) != GOLD_SHA256:
        raise ValueError('Frozen FEVER gold checksum mismatch')
    summary = score_predictions(predictions, read_jsonl(gold_path))
    output.mkdir(parents=True)
    write_json_atomic(output / 'summary.json', summary)
    write_json_atomic(output / 'input_manifest.json', {
        'prediction_manifest_sha256': sha256(run / 'output_manifest.json'),
        'predictions_sha256': sha256(run / 'predictions.json'),
        'gold_sha256': GOLD_SHA256, 'no_target_fitting': True,
        'code_sha256': {n: sha256(ROOT / n) for n in (
            'scripts/score_fever_nli.py', 'src/apv_rag/fever_nli.py', 'src/apv_rag/metrics.py')}})
    write_json_atomic(output / 'output_manifest.json', {
        name: sha256(output / name) for name in ('summary.json', 'input_manifest.json')})
    print(f'FEVER scoring complete: {summary["claims"]} claims; no target fitting.')
    print(f'Accuracy: {summary["accuracy"]:.4f}; macro F1: {summary["metrics"]["macro_f1"]:.4f}')
    print(f'Complete gold sentence-group recall: {summary["retrieval"]["complete_sentence_group_recall"]:.4f}')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()

"""Run the fixed English NLI baseline on saved FEVER contexts; never open gold."""

import json
import sqlite3
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from apv_rag.fever_corpus import EXPECTED_ARCHIVE_SHA256, EXPECTED_PAGES
from apv_rag.fever_nli import load_claims, predict_context, read_jsonl, validate_contexts
from apv_rag.nli_comparison import _fingerprint, _model_files, _pairs
from apv_rag.splits import sha256, write_json_atomic

REFERENCE_SHA256 = 'ada02e3553c1be1a772ac46dc7f1397e5523fda80a792627dadaf3591b6680c9'


def prepare_run(root):
    root = Path(root)
    source = root / 'data/external/fever/heldout_v1'
    claims = load_claims(source)
    retrieval = root / 'artifacts/fever_retrieval_v1'
    receipt = json.loads((retrieval / 'input_manifest.json').read_text(encoding='utf-8'))
    contexts = retrieval / 'contexts.jsonl'
    if (receipt['contexts_sha256'] != sha256(contexts) or
            receipt['model_inputs_sha256'] != sha256(source / 'model_inputs.jsonl') or
            receipt['selection_manifest_sha256'] != sha256(source / 'selection_manifest.json') or
            receipt['archive_sha256'] != EXPECTED_ARCHIVE_SHA256 or
            receipt['source_records'] != EXPECTED_PAGES or
            receipt['page_count'] + receipt['empty_placeholders'] != EXPECTED_PAGES or
            receipt['retrieval'] != 'fts5_bm25_title3_body1, top3 documents and top3 sentences'):
        raise ValueError('Saved FEVER retrieval identity mismatch')
    rows = validate_contexts(claims, read_jsonl(contexts))
    reference = root / 'config/fever_nli_reference_v1.json'
    if sha256(reference) != REFERENCE_SHA256:
        raise ValueError('Frozen NLI reference checksum mismatch')
    ref = json.loads(reference.read_text(encoding='utf-8'))
    model_path = root / 'models/nli-deberta-v3-small'
    if _model_files(model_path) != ref['model']:
        raise ValueError('Original English NLI model files required; model identity differs')
    packages = {name: version(name) for name in ref['packages']}
    if packages != ref['packages']:
        raise ValueError(f'Original inference packages required: expected {ref["packages"]}; got {packages}')
    identity = {'scope': 'fixed English NLI retrieved-evidence baseline; no gold opened',
                'claims': len(rows), 'contexts_sha256': sha256(contexts),
                'retrieval_receipt_sha256': sha256(retrieval / 'input_manifest.json'),
                'reference_sha256': REFERENCE_SHA256, 'models': ref['model'],
                'model_inputs_sha256': sha256(source / 'model_inputs.jsonl'),
                'selection_manifest_sha256': sha256(source / 'selection_manifest.json'),
                'packages': packages, 'seed': 369, 'threads': 4, 'batch_size': 8,
                'dtype': 'float32', 'device': 'cpu', 'max_pair_tokens': 256,
                'rule': 'one selected excerpt per nonempty page; mean C/E/N; S/R/NEI argmax',
                'empty_evidence_rule': 'Not Enough Evidence probability one',
                'code_sha256': {n: sha256(root / n) for n in (
                    'scripts/run_fever_nli.py', 'src/apv_rag/fever_nli.py',
                    'src/apv_rag/direct_nli.py', 'src/apv_rag/multilingual_nli.py',
                    'src/apv_rag/nli_comparison.py')},
                'protocol_sha256': sha256(root / 'docs/FEVER_EXTERNAL_EVALUATION.md')}
    return rows, model_path, identity


def main():
    output = ROOT / 'artifacts/fever_nli_v1'
    if (output / 'output_manifest.json').exists():
        raise FileExistsError('Frozen FEVER predictions already exist; run score_fever_nli.py')
    print('Checking saved contexts, original model and package identities...', flush=True)
    rows, model_path, identity = prepare_run(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'input_manifest.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != identity:
        raise ValueError('Run inputs changed; refusing existing inference cache')
    write_json_atomic(receipt, identity)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(369)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32).to('cpu').eval()
    if {int(k): str(v).lower() for k, v in model.config.id2label.items()} != {
            0: 'contradiction', 1: 'entailment', 2: 'neutral'}:
        raise ValueError('Unexpected English NLI label mapping')
    predictions = []
    with closing(sqlite3.connect(output / 'pair_cache.sqlite')) as con:
        con.execute('CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, scores TEXT)')
        for position, row in enumerate(rows, 1):
            premises = [d['text'] for d in row['evidence'] if d['text'].strip()]
            scores = _pairs(con, model, tokenizer, torch, premises, row['claim'],
                            _fingerprint(identity), 8)
            predictions.append(predict_context(row, scores))
            if position == 1 or position % 25 == 0:
                print(f'FEVER NLI: {position}/{len(rows)}', flush=True)
    write_json_atomic(output / 'predictions.json', predictions)
    write_json_atomic(output / 'output_manifest.json', {
        name: sha256(output / name) for name in ('input_manifest.json', 'predictions.json')})
    print(f'FEVER NLI complete: {len(predictions)} predictions frozen; no gold opened.', flush=True)
    print(f'Predictions SHA-256: {sha256(output / "predictions.json")}', flush=True)
    print('Next: python scripts/score_fever_nli.py', flush=True)


if __name__ == '__main__':
    main()

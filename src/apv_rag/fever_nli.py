"""Frozen FEVER retrieved-evidence NLI baseline and separate scoring."""

import json
import math
from pathlib import Path

import numpy as np

from apv_rag.direct_nli import TARGET_LABELS, direct_probabilities
from apv_rag.metrics import classification_metrics
from apv_rag.multilingual_nli import normalize_model_scores
from apv_rag.splits import sha256

INPUT_SHA256 = '3fd90908bec3485733cc6583bc84c9f1c21db7c561b2a30d27a75b692fd398b9'
SELECTION_SHA256 = '2e18eae7c77fb45250c588f7485dd50766f35e9b1b104285ae281da79c31e74a'
GOLD_SHA256 = 'a90f3b2bd51ff01df76ebc236c22e2d4f2b8c4925f3b63c85d7a9802d151af70'
LABEL_MAP = {'SUPPORTS': 'Supported', 'REFUTES': 'Refuted',
             'NOT ENOUGH INFO': 'Not Enough Evidence'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def load_claims(source):
    source = Path(source)
    for name, digest in (('model_inputs.jsonl', INPUT_SHA256),
                         ('selection_manifest.json', SELECTION_SHA256)):
        if sha256(source / name) != digest:
            raise ValueError(f'Frozen FEVER input checksum mismatch: {name}')
    claims = read_jsonl(source / 'model_inputs.jsonl')
    selection = json.loads((source / 'selection_manifest.json').read_text(encoding='utf-8'))
    if len(claims) != 300 or [r['id'] for r in claims] != selection['selected_ids']:
        raise ValueError('Frozen FEVER cohort alignment mismatch')
    return claims


def validate_contexts(claims, rows):
    if not rows or len(rows) != len(claims):
        raise ValueError('Context alignment mismatch')
    seen = set()
    for claim, row in zip(claims, rows, strict=True):
        if set(claim) != {'id', 'claim'} or set(row) != {'id', 'claim', 'evidence'}:
            raise ValueError('Expected claim-only context schema')
        if (type(row['id']) is not int or row['id'] in seen or
                row['id'] != claim['id'] or row['claim'] != claim['claim']):
            raise ValueError('Context claim alignment mismatch')
        seen.add(row['id'])
        if not isinstance(row['evidence'], list) or len(row['evidence']) > 3:
            raise ValueError('Expected at most three retrieved pages')
        pages = set()
        for doc in row['evidence']:
            if (set(doc) != {'id', 'text', 'selected_sentence_indices', 'fts5_bm25'} or
                    not isinstance(doc['id'], str) or not doc['id'] or doc['id'] in pages or
                    not isinstance(doc['text'], str) or
                    not isinstance(doc['fts5_bm25'], (int, float)) or
                    not math.isfinite(doc['fts5_bm25'])):
                raise ValueError('Invalid retrieved page schema')
            pages.add(doc['id'])
            indices = doc['selected_sentence_indices']
            if (not isinstance(indices, list) or len(indices) > 3 or
                    any(type(i) is not int or i < 0 for i in indices) or
                    len(set(indices)) != len(indices) or bool(indices) != bool(doc['text'].strip())):
                raise ValueError('Invalid selected sentence indices')
    return rows


def predict_context(row, scores):
    used = [doc for doc in row['evidence'] if doc['text'].strip()]
    if len(scores) != len(used):
        raise ValueError('NLI premise/score alignment mismatch')
    raw_scores = [list(values) for values in scores]
    normalized = normalize_model_scores(raw_scores) if raw_scores else []
    p = direct_probabilities(normalized)
    return {'id': row['id'], 'claim': row['claim'], 'evidence': row['evidence'],
            'nli_probabilities_cen': raw_scores,
            'predicted_label': TARGET_LABELS[int(p.argmax())], 'probabilities': p.tolist()}


def score_predictions(predictions, gold):
    if (not predictions or len(predictions) != len(gold) or
            len({p['id'] for p in predictions}) != len(predictions) or
            [p['id'] for p in predictions] != [g['id'] for g in gold]):
        raise ValueError('Prediction/gold alignment mismatch')
    truth, predicted, probabilities = [], [], []
    eligible = page_hits = sentence_hits = 0
    for pred, target in zip(predictions, gold, strict=True):
        rebuilt = predict_context(pred, pred['nli_probabilities_cen'])
        if (pred['predicted_label'] != rebuilt['predicted_label'] or
                pred['probabilities'] != rebuilt['probabilities']):
            raise ValueError('Saved prediction differs from fixed argmax rule')
        truth.append(LABEL_MAP[target['label']])
        predicted.append(pred['predicted_label'])
        probabilities.append(pred['probabilities'])
        if target['label'] == 'NOT ENOUGH INFO':
            continue
        eligible += 1
        groups = [{(item[2], item[3]) for item in group
                   if item[2] is not None and item[3] is not None}
                  for group in target['evidence']]
        groups = [g for g in groups if g]
        if not groups:
            raise ValueError('Verifiable claim lacks a gold evidence group')
        pages = {d['id'] for d in pred['evidence']}
        sentences = {(d['id'], i) for d in pred['evidence']
                     for i in d['selected_sentence_indices']}
        page_hits += int(any({p for p, _ in g} <= pages for g in groups))
        sentence_hits += int(any(g <= sentences for g in groups))
    metrics = classification_metrics(truth, predicted, np.asarray(probabilities), TARGET_LABELS,
                                     ece_bins=15, coverages=[0.5, 0.8, 1.0])
    return {'scope': 'fixed retrieved-evidence NLI baseline on previously unused FEVER cohort',
            'claims': len(predictions), 'accuracy': float(np.mean(np.array(truth) == predicted)),
            'metrics': metrics, 'retrieval': {
                'verifiable_claims': eligible,
                'complete_page_groups_found': page_hits,
                'complete_sentence_groups_found': sentence_hits,
                'complete_page_group_recall': page_hits / eligible if eligible else None,
                'complete_sentence_group_recall': sentence_hits / eligible if eligible else None},
            'limitations': [
                'Label accuracy and retrieval recall are not the official FEVER score.',
                'Up to nine selected sentences; pair inputs may truncate at 256 tokens.',
                'Neutral is not validated claim-level evidence insufficiency.',
                'Risk/coverage and ECE are descriptive; no target calibration or gate fitting.',
                'No historical source authentication, generated explanation or multilingual test.']}


def verify_prediction_files(output):
    output = Path(output)
    hashes = json.loads((output / 'output_manifest.json').read_text(encoding='utf-8'))
    if set(hashes) != {'input_manifest.json', 'predictions.json'}:
        raise ValueError('Incomplete frozen prediction manifest')
    for name, digest in hashes.items():
        if sha256(output / name) != digest:
            raise ValueError(f'Frozen prediction checksum mismatch: {name}')
    return (json.loads((output / 'input_manifest.json').read_text(encoding='utf-8')),
            json.loads((output / 'predictions.json').read_text(encoding='utf-8')))

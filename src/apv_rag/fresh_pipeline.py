"""Fresh retrieval and context-bound features for the experimental controller."""
import hashlib
import json

import numpy as np

from apv_rag.gated_generation_flow import execute_generation
from apv_rag.integrated_gate import collapse_context, probability_complete
from apv_rag.retrieval import BM25Index
from apv_rag.sentence_context import select_sentences

POLICIES = ('no_gate', 'nli', 'embedding', 'combined')


def context_digest(shown_claim, evidence):
    payload = json.dumps({'shown_claim': shown_claim, 'evidence': evidence},
                         sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class FreshRetriever:
    def __init__(self, corpus):
        self.corpus = sorted(corpus, key=lambda d: d['doc_id'])
        if len({d['doc_id'] for d in self.corpus}) != len(self.corpus):
            raise ValueError('Duplicate corpus IDs')
        self.index = BM25Index([' '.join(d['abstract']) for d in self.corpus])

    def prepare(self, claim, clip):
        retrieved = []
        for i, score in self.index.search(claim, top_k=3):
            if score <= 0:
                continue
            document = self.corpus[i]
            selected = select_sentences(claim, document['abstract'])
            retrieved.append({'id': document['doc_id'], 'text': clip(selected['text'], 96),
                              'selected_sentence_indices': selected['sentence_indices'],
                              'bm25_score': float(score)})
        return {'claim': claim, 'shown_claim': clip(claim, 64),
                'retrieved_evidence': retrieved, 'evidence': collapse_context(retrieved)}


def _aggregate(scores, cosines, count):
    a = np.asarray(scores, dtype=float)
    c = np.asarray(cosines, dtype=float)
    if count < 1 or a.shape != (count, 3) or c.shape != (count,):
        raise ValueError('Neural feature shape mismatch')
    if (not np.isfinite(a).all() or not np.isfinite(c).all() or
            (a < 0).any() or (a > 1).any() or (abs(c) > 1.000001).any() or
            not np.allclose(a.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid neural feature scores')
    return np.concatenate([a.mean(0), a.max(0), a.std(0),
                           [c.mean(), c.max(), c.min(), c.std()]]).tolist()


def feature_entry(shown_claim, evidence, scores, cosines):
    return {'context_sha256': context_digest(shown_claim, evidence),
            'scores_cen': scores, 'cosines': cosines,
            'features': _aggregate(scores, cosines, len(evidence))}


def validate_feature_entry(entry, shown_claim, evidence):
    if set(entry) != {'context_sha256', 'scores_cen', 'cosines', 'features'}:
        raise ValueError('Feature cache schema mismatch')
    if entry['context_sha256'] != context_digest(shown_claim, evidence):
        raise ValueError('Feature/context binding mismatch')
    expected = _aggregate(entry['scores_cen'], entry['cosines'], len(evidence))
    features = np.asarray(entry['features'], dtype=float)
    if features.shape != (13,) or not np.allclose(features, expected, atol=1e-12, rtol=0):
        raise ValueError('Feature aggregate mismatch')
    return expected


def execute_policies(prepared, entry, models, generate):
    evidence = prepared['evidence']
    if evidence != collapse_context(prepared['retrieved_evidence']):
        raise ValueError('Context collapse mismatch')
    if set(models) != set(POLICIES[1:]):
        raise ValueError('Unexpected frozen gate models')
    if evidence:
        features = validate_feature_entry(entry, prepared['shown_claim'], evidence)
    elif entry is not None:
        raise ValueError('Unexpected features for empty retrieval')
    rows = []
    for policy in POLICIES:
        probability = (probability_complete(features, models[policy])
                       if evidence and policy != 'no_gate' else None)
        row = execute_generation(prepared['shown_claim'], evidence, generate, probability, .5)
        row['policy'] = policy
        rows.append(row)
    return rows

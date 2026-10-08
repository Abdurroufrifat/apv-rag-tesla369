"""Build cited evidence displays from bound excerpts, without free-form rationale.

An exact excerpt is an evidence record, not certification that it justifies a
model verdict. This module cannot authenticate publishers or historical claims.
"""

from hashlib import sha256
import json

from apv_rag.generative_rag import LABELS
from apv_rag.integrated_gate import collapse_context
from apv_rag.source_snapshot_guard import passage_trace, snapshot_gate


def object_sha256(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return sha256(encoded).hexdigest()


def build_evidence_display(row, corpus):
    """Use only the canonical claim, supplied candidate and frozen evidence.

    Gold labels and generated explanations are not used. Snapshot admission
    reuses the existing sentence-selection/prefix check and family collapse.
    Prefix clipping is retained literally; tokenizer budgets are not recounted.
    """
    result = {
        'schema': 'evidence_display_v1', 'claim_id': str(row.get('claim_id', '')),
        'claim': row.get('claim'), 'status': 'abstain', 'candidate_label': None,
        'items': [], 'reasons': [], 'source_snapshot_bound': False,
        'freeform_explanation_included': False,
        'support_relationship_verified': False,
        'qualification': 'Literal excerpts from a frozen benchmark snapshot only. '
            'The candidate is not a gold verdict; its support relationship, historical '
            'attribution and evidence-absence conclusion are not verified. '
            'Clipped or stitched excerpts may be incomplete.',
    }
    try:
        claim = row['claim']
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError('Missing canonical claim')
        retrieved = row['retrieved_evidence']
        collapsed = row['evidence']
        if collapsed != collapse_context(retrieved):
            raise ValueError('Context collapse mismatch')
        trace = passage_trace(claim, retrieved, corpus)
        bound = snapshot_gate(trace)
        result['source_snapshot_bound'] = bound
    except (KeyError, TypeError, ValueError, AttributeError):
        result['reasons'].append('invalid_snapshot_context')
        return result
    if not bound:
        result['reasons'].append('snapshot_source_mismatch_or_missing')
    candidate = row.get('candidate_label')
    if candidate is None:
        result['reasons'].append('upstream_abstention')
    elif candidate not in LABELS:
        result['reasons'].append('invalid_upstream_candidate')
    if result['reasons']:
        return result
    result['status'] = 'machine_candidate'
    result['candidate_label'] = candidate
    for evidence in collapsed:
        doc_id = str(evidence['id'])
        result['items'].append({
            'citation_id': doc_id, 'quote': evidence['text'],
            'selected_sentence_indices': list(evidence['selected_sentence_indices']),
            'excerpt_sha256': sha256(evidence['text'].encode('utf-8')).hexdigest(),
            'source_doc_sha256': object_sha256(corpus[doc_id]),
            'source_kind': 'frozen_benchmark_snapshot',
        })
    return result


def verify_evidence_display(display, row, corpus):
    """Replay the whole structure; accepting a citation ID alone is insufficient."""
    if display != build_evidence_display(row, corpus):
        raise ValueError('Evidence display does not match its original context and snapshot')
    return True

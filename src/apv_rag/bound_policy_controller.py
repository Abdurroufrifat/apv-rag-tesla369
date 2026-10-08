"""Opt-in snapshot binding before all four frozen generation policies.

Binding checks corpus text integrity, not publisher identity or factual support.
"""

from apv_rag.fresh_pipeline import POLICIES, execute_policies
from apv_rag.gated_generation_flow import execute_generation
from apv_rag.integrated_gate import collapse_context
from apv_rag.source_snapshot_guard import passage_trace, snapshot_gate


def execute_bound_policies(prepared, entry, models, corpus, generate):
    """Refuse unbound excerpts before evaluating features or calling generate."""
    if prepared['evidence'] != collapse_context(prepared['retrieved_evidence']):
        raise ValueError('Context collapse mismatch')
    if set(models) != set(POLICIES[1:]):
        raise ValueError('Unexpected frozen gate models')
    trace = passage_trace(prepared['claim'], prepared['retrieved_evidence'], corpus)
    bound = snapshot_gate(trace)
    if bound:
        rows = execute_policies(prepared, entry, models, generate)
    else:
        rows = []
        for policy in POLICIES:
            row = execute_generation(prepared['shown_claim'], [], generate)
            if trace['passages']:
                row['reasons'] = ['snapshot_source_mismatch']
            row['evidence'] = prepared['evidence']
            row['policy'] = policy
            rows.append(row)
    for row in rows:
        row['snapshot_source_bound'] = bound
        row['source_trace'] = trace
    return rows

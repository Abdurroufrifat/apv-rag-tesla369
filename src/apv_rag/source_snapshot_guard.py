"""Optional pre-generation binding to a pinned benchmark corpus snapshot.

This enforces stored source-text integrity; it does not authenticate publishers.
"""
from collections import Counter

from apv_rag.gated_generation_flow import execute_generation
from apv_rag.integrated_gate import collapse_context
from apv_rag.sentence_context import select_sentences


def passage_trace(claim, retrieved_evidence, corpus):
    counts=Counter(passages=0,known_ids=0,matching_indices=0,
                   matching_prefixes=0,bound_passages=0)
    for item in retrieved_evidence:
        counts['passages']+=1
        doc=corpus.get(str(item['id']))
        if doc is None:continue
        counts['known_ids']+=1
        selected=select_sentences(claim,doc['abstract'])
        indexes=item.get('selected_sentence_indices')==selected['sentence_indices']
        text=isinstance(item.get('text'),str) and bool(item['text']) and selected['text'].startswith(item['text'])
        counts['matching_indices']+=indexes
        counts['matching_prefixes']+=text
        counts['bound_passages']+=indexes and text
    return dict(counts)


def snapshot_gate(trace):
    return trace['passages']>0 and trace['bound_passages']==trace['passages']


def execute_snapshot_guarded(claim,shown_claim,retrieved_evidence,evidence,corpus,generate,
                             gate_probability=None,threshold=.5):
    """Refuse an unbound context before calling a model or response backend."""
    if evidence!=collapse_context(retrieved_evidence):
        raise ValueError('Context collapse mismatch')
    trace=passage_trace(claim,retrieved_evidence,corpus)
    bound=snapshot_gate(trace)
    if not bound:
        result=execute_generation(shown_claim,[],generate,gate_probability,threshold)
        if trace['passages']:
            result['reasons']=['snapshot_source_mismatch']
        result['evidence']=evidence
    else:
        result=execute_generation(shown_claim,evidence,generate,gate_probability,threshold)
    result['snapshot_source_bound']=bound
    result['source_trace']=trace
    return result

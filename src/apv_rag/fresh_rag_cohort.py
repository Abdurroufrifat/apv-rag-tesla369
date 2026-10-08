"""Prepare compatible claim-only inputs for the unchanged three-class RAG."""
from apv_rag.generative_rag import LABELS


def build_cohort(inputs, gold):
    ids=[r['claim_id'] for r in inputs]
    gold_ids=[r['claim_id'] for r in gold]
    if len(ids)!=len(set(ids)) or len(gold_ids)!=len(set(gold_ids)) or set(ids)!=set(gold_ids):
        raise ValueError('Unique aligned input/gold IDs required')
    targets={r['claim_id']:r['label'] for r in gold}
    if any(label not in (*LABELS,'Conflicting Evidence/Cherrypicking') for label in targets.values()):
        raise ValueError('Unexpected target label')
    excluded=[i for i in ids if targets[i] not in LABELS]
    claims=[{'id':r['claim_id'],'claim':r['record']['claim']} for r in inputs if r['claim_id'] not in excluded]
    return claims,{r['id']:targets[r['id']] for r in claims},excluded

"""Fixed English-gate transfer with separately bound multilingual NLI controls."""
import numpy as np

from apv_rag.direct_nli import TARGET_LABELS,direct_probabilities
from apv_rag.fever_pipeline import answer_score
from apv_rag.fresh_pipeline import context_digest,execute_policies,validate_feature_entry
from apv_rag.integrated_gate import collapse_context
from apv_rag.multilingual_generation import LANGUAGES


def select_sample(retrieved,claim_ids):
    wanted={str(i) for i in claim_ids}
    if not wanted or len(wanted)!=len(claim_ids):raise ValueError('Unique sample IDs required')
    selected=[]
    for file,rows in retrieved.items():
        language=file.split('/')[0]
        if language not in LANGUAGES:raise ValueError('Unknown input language')
        seen=set()
        for r in rows:
            if (set(r)!={'row','claim_id','claim','evidence'} or r['row'] in seen or
                    type(r['row']) is not int or type(r['claim_id']) is not int or not isinstance(r['claim'],str)):
                raise ValueError('Expected unique model-only retrieved records')
            seen.add(r['row'])
            if str(r['claim_id']) not in wanted:continue
            if (not isinstance(r['evidence'],list) or len(r['evidence'])>3 or
                    any(set(d)!={'id','text','score'} for d in r['evidence'])):
                raise ValueError('Expected model-only evidence excerpts')
            selected.append({'key':f'{file}:{r["row"]}','file':file,'row':r['row'],
                'claim_id':r['claim_id'],'claim':r['claim'],'language':language,
                'translation_origin':'original' if language=='en' else
                    ('upstream_human' if '.human.' in file else 'upstream_machine'),
                'retrieved_evidence':r['evidence']})
    if len({r['key'] for r in selected})!=len(selected):raise ValueError('Sample key collision')
    return selected


def prepare_context(source,clip):
    shown,original_count,shown_count=clip(source['claim'],64)
    evidence=[]; counts=[]
    for doc in source['retrieved_evidence']:
        text,original,used=clip(doc['text'],96)
        counts.append({'id':doc['id'],'original':original,'shown':used})
        if text.strip():evidence.append(dict(doc,text=text))
    return {'claim':source['claim'],'shown_claim':shown,'retrieved_evidence':evidence,
            'evidence':collapse_context(evidence),'clipping':{
                'claim':{'original':original_count,'shown':shown_count},'passages':counts}}


def multilingual_entry(prepared,scores):
    entry={'context_sha256':context_digest(prepared['shown_claim'],prepared['evidence']),
           'scores_cen':scores}
    validate_multilingual_entry(prepared,entry)
    return entry


def validate_multilingual_entry(prepared,entry):
    if (not prepared['evidence'] or set(entry)!={'context_sha256','scores_cen'} or
            entry['context_sha256']!=context_digest(prepared['shown_claim'],prepared['evidence'])):
        raise ValueError('Multilingual NLI context binding differs')
    scores=np.asarray(entry['scores_cen'],dtype=float)
    if (scores.shape!=(len(prepared['evidence']),3) or not np.isfinite(scores).all() or
            (scores<0).any() or (scores>1).any() or
            not np.allclose(scores.sum(1),1,rtol=0,atol=1e-6)):
        raise ValueError('Multilingual NLI probability shape or normalization differs')
    return entry['scores_cen']


def direct_prediction(prepared,english,multi):
    if prepared['evidence']:
        validate_feature_entry(english,prepared['shown_claim'],prepared['evidence'])
        ml=validate_multilingual_entry(prepared,multi)
        en=english['scores_cen']
    else:
        if english is not None or multi is not None:raise ValueError('Features exist for empty retrieval')
        en=ml=[]
    result={}
    for name,scores in [('english_nli',en),('multilingual_nli',ml)]:
        p=direct_probabilities(scores)
        result[name]={'label':TARGET_LABELS[int(p.argmax())],'probabilities':p.tolist()}
    return result


def execute(prepared,english,multi,heads,generate):
    # Validate both baselines before any model request; gates use original English features only.
    direct_prediction(prepared,english,multi)
    rows=execute_policies(prepared,english,heads,generate)
    for row in rows:
        row['uncalibrated_english_nli_label_score']=answer_score(row,english)
        row['correctness_confidence']=None
    return rows

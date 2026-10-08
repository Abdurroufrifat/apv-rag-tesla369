"""Experimental annotation-completeness gate and context-family collapse."""
import math
import numpy as np


def collapse_context(evidence):
    kept=[];families=set();texts=set()
    for e in evidence:
        family=str(e.get('family_id',e['id']))
        text=' '.join(e['text'].split())
        if family in families or text in texts:continue
        families.add(family);texts.add(text);kept.append(e)
    return kept


def probability_complete(features,model):
    x=np.asarray(features,dtype=float)[model['columns']]
    scale=np.asarray(model['scale']);mean=np.asarray(model['mean'])
    if (scale<=0).any() or not np.isfinite(x).all():raise ValueError('Invalid gate features')
    z=float(((x-mean)/scale)@np.asarray(model['coefficients'])[0]+model['intercept'][0])
    if not math.isfinite(z):raise ValueError('Nonfinite gate score')
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def apply_gate(verdict,numeric_reasons,probability,threshold=.5):
    if probability is not None and (not math.isfinite(probability) or not 0<=probability<=1):raise ValueError('Invalid gate probability')
    if numeric_reasons or probability is None or probability<threshold:return None
    return verdict

"""Label-blind new-claim selection and development-only frozen confidence models."""
import hashlib
from apv_rag.multilingual_retrieval import normalize,excerpt_id
from apv_rag.multilingual_calibration import validate,claim_folds,metrics
from apv_rag.fever_pipeline import POLICIES,fit_correctness,correctness_probability
from apv_rag.fever_nli import LABEL_MAP

FILES=tuple(f'{lang}/test.jsonl' for lang in ('en','es','fr','id','ja','zh'))


def page_key(text):
    return normalize(text.replace('_',' ').replace('-LRB-','(').replace('-RRB-',')'))


def select_new(sets,used_ids,used_texts,used_pages,used_excerpts,count=100):
    if set(sets)!=set(FILES) or type(count) is not int or count<1:
        raise ValueError('Six official files and positive count required')
    english=sets[FILES[0]]
    expected=[r['id'] for r in english]
    if not english or any([r['id'] for r in sets[name]]!=expected for name in FILES):
        raise ValueError('Parallel claim ID alignment differs')
    groups={}
    for i,r in enumerate(english):
        if type(r['id']) is not int:raise ValueError('Integer IDs required')
        groups.setdefault(r['id'],[]).append(i)
    candidates={}
    for ident,indices in groups.items():
        if ident in used_ids:continue
        texts=set();pages=set();excerpts=set()
        for name in FILES:
            for i in indices:
                row=sets[name][i]
                if any(not isinstance(row[k],str) or not row[k].strip() for k in ('claim','page','evidence')):
                    raise ValueError('Nonempty paired texts required')
                texts.add(normalize(row['claim']));pages.add(page_key(row['page']))
                excerpts.add(excerpt_id(row['evidence']))
        if texts&used_texts or pages&used_pages or excerpts&used_excerpts:continue
        # Text-only deterministic representative for IDs with several paired excerpts.
        chosen=min(indices,key=lambda i:(excerpt_id(english[i]['evidence']),i))
        candidates[ident]=(chosen,texts,pages,excerpts)
    order=sorted(candidates,key=lambda i:(hashlib.sha256(f'apv-multilingual-confirmation:369:{i}'.encode()).hexdigest(),i))
    chosen=[];texts=set();pages=set();excerpts=set()
    for ident in order:
        i,t,p,e=candidates[ident]
        if texts&t or pages&p or excerpts&e:continue
        chosen.append(i);texts|=t;pages|=p;excerpts|=e
        if len(chosen)==count:return chosen
    raise ValueError(f'Only {len(chosen)} eligible disjoint claims; requested {count}')


def fit_development(rows,gold):
    targets=validate(rows,gold,claim_folds(rows));models={};controls={}
    for policy in POLICIES:
        train=[r for r in rows if r['policy']==policy and r['candidate_label'] is not None]
        model=fit_correctness([r['uncalibrated_english_nli_label_score'] for r in train],
            [r['candidate_label']==LABEL_MAP[targets[r['key']]['label']] for r in train])
        models[policy]=model;n=model['accepted_development_answers']
        controls[policy]={'probability':(model['correct_development_answers']+1)/(n+2) if n else None,
            'accepted_development_answers':n,'correct_development_answers':model['correct_development_answers']}
    return {'development_claim_ids':sorted({r['claim_id'] for r in rows}),
        'policies':models,'prevalence_controls':controls}


def apply_frozen(rows,fits):
    if {r['claim_id'] for r in rows}&set(fits['development_claim_ids']):
        raise ValueError('Confirmation overlaps development claims')
    output=[]
    for r in rows:
        raw=r['uncalibrated_english_nli_label_score'];accepted=r['candidate_label'] is not None
        if not accepted and raw is not None:raise ValueError('Abstention score differs')
        output.append({n:r[n] for n in ('key','file','claim_id','policy','candidate_label')}|
            {'raw_generated_label_nli_score':raw,
             'correctness_confidence':correctness_probability(raw,fits['policies'][r['policy']]) if accepted else None,
             'prevalence_control_confidence':fits['prevalence_controls'][r['policy']]['probability'] if accepted else None})
    return output


def summarize_confirmation(predicted,gold):
    targets={g['key']:g for g in gold}
    if len(targets)!=len(gold) or len(predicted)!=len(gold)*4 or len({(r['key'],r['policy']) for r in predicted})!=len(predicted):
        raise ValueError('Confirmation/gold coverage differs')
    def values(rows):
        m=metrics(rows,targets)
        # Reuse calculations, but never mislabel a frozen-model test as cross-fitting.
        for old,new in [('cross_fitted_metrics','frozen_calibrator_metrics'),
                        ('cross_fitted_prevalence_control_metrics','frozen_prevalence_control_metrics')]:
            if old in m:m[new]=m.pop(old)
        return m
    return {'scope':'new project-held-out claims; machine-translated closed excerpt-pool confirmation',
        'underlying_claim_ids':len({g['claim_id'] for g in gold}),'queries':len(gold),
        'policy_records':len(predicted),'confirmation_fitting':False,
        'per_file':{name:{p:values([r for r in predicted if r['file']==name and r['policy']==p])
            for p in POLICIES} for name in FILES},
        'pooled_descriptive':{p:values([r for r in predicted if r['policy']==p]) for p in POLICIES}}

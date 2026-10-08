"""Post-result fixed-coverage controls for the frozen FEVER generator outputs."""
import numpy as np

from apv_rag.fever_calibration import normalized
from apv_rag.fever_nli import LABEL_MAP
from apv_rag.fever_pipeline import POLICIES
from apv_rag.paired_statistics import holm_adjust
from apv_rag.splits import UnionFind

RULES = ('generated_label_nli', 'mean_embedding_cosine')


def select_control(rows, cache, count, rule):
    """Rank accepted control answers using model scores only; ties preserve frozen order."""
    accepted = [i for i,r in enumerate(rows) if r['candidate_label'] is not None]
    if type(count) is not int or count < 0 or count > len(accepted) or rule not in RULES:
        raise ValueError('Invalid matched count or ranking rule')
    def score(i):
        r = rows[i]
        value = r['score'] if rule == 'generated_label_nli' else float(np.mean(cache[str(r['claim_id'])]['cosines']))
        if not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError('Ranking requires finite model scores')
        return value
    return sorted(accepted, key=lambda i:(-score(i),i))[:count]


def page_groups(rows):
    """Conservative transitive grouping by any visible retrieved page or normalized claim."""
    uf, pages, claims = UnionFind(len(rows)), {}, {}
    for i,row in enumerate(rows):
        text = normalized(row['claim'])
        if text in claims:uf.union(i,claims[text])
        else:claims[text]=i
        for d in row['evidence']:
            page=d['id']
            if page in pages:uf.union(i,pages[page])
            else:pages[page]=i
    groups={}
    for i in range(len(rows)):groups.setdefault(uf.find(i),[]).append(i)
    return sorted(groups.values(),key=lambda g:g[0])


def grouped_effect(differences, groups):
    """Paired all-claim accuracy difference; group bootstrap and group sign swaps."""
    d=np.asarray(differences,dtype=float)
    if (d.ndim!=1 or not len(d) or not np.isfinite(d).all() or
            sorted(i for g in groups for i in g)!=list(range(len(d))) or any(not g for g in groups)):
        raise ValueError('Aligned differences and a complete group partition required')
    sums=np.array([d[g].sum() for g in groups])
    sizes=np.array([len(g) for g in groups])
    observed=float(d.mean())
    rng=np.random.default_rng(369)
    ix=rng.integers(0,len(groups),size=(2000,len(groups)))
    boot=sums[ix].sum(1)/sizes[ix].sum(1)
    active=sums[sums!=0]
    if not len(active):
        p, swaps, method=1.0,1,'exact zero effect'
    elif len(active)<=12:
        codes=np.arange(2**len(active))[:,None]
        signs=2*((codes >> np.arange(len(active))) & 1)-1
        values=signs @ active
        p=float((np.abs(values)>=abs(sums.sum())-1e-12).mean())
        swaps,method=len(values),'exact sign swaps of nonzero group differences'
    else:
        signs=rng.integers(0,2,size=(10000,len(active)))*2-1
        values=signs @ active
        p=float((1+(np.abs(values)>=abs(sums.sum())-1e-12).sum())/10001)
        swaps,method=10000,'Monte Carlo sign swaps of nonzero group differences'
    return {'all_claim_accuracy_difference':observed,
            'group_bootstrap_95_percentile_interval':np.quantile(boot,[0.025,0.975]).tolist(),
            'two_sided_group_swap_p':p,'swap_samples':swaps,'swap_method':method,
            'bootstrap_replicates':2000,'seed':369}


def analyze(rows, gold, cache):
    by_policy={p:[r for r in rows if r['policy']==p] for p in POLICIES}
    if len(rows)!=len(gold)*len(POLICIES) or not gold or len({g['id'] for g in gold})!=len(gold):
        raise ValueError('Prediction count or unique gold IDs differ')
    control=by_policy['no_gate']
    for subset in by_policy.values():
        if [r['claim_id'] for r in subset]!=[g['id'] for g in gold]:
            raise ValueError('Policy/gold alignment differs')
    truth=[LABEL_MAP[g['label']] for g in gold]
    groups=page_groups(control)
    comparisons=[]
    for policy in POLICIES[1:]:
        subset=by_policy[policy]
        chosen=[i for i,r in enumerate(subset) if r['candidate_label'] is not None]
        for i in chosen:
            if subset[i]['candidate_label']!=control[i]['candidate_label']:
                raise ValueError('Gate output does not preserve common answer')
        for rule in RULES:
            selected=select_control(control,cache,len(chosen),rule)
            if not chosen:
                comparisons.append({'gate':policy,'control':rule,'status':'zero accepted answers','accepted_answers':0})
                continue
            a=np.zeros(len(gold),dtype=int); b=np.zeros(len(gold),dtype=int)
            for i in chosen:a[i]=subset[i]['candidate_label']==truth[i]
            for i in selected:b[i]=control[i]['candidate_label']==truth[i]
            comparisons.append({'gate':policy,'control':rule,'status':'evaluated',
                'accepted_answers':len(chosen),'coverage':len(chosen)/len(gold),
                'gate_correct':int(a.sum()),'control_correct':int(b.sum()),
                'gate_accepted_accuracy':float(a.sum()/len(chosen)),
                'control_accepted_accuracy':float(b.sum()/len(chosen)),
                'gate_selected_ids':[gold[i]['id'] for i in chosen],
                'control_selected_ids':[gold[i]['id'] for i in selected],
                'paired_statistics':grouped_effect(a-b,groups)})
    tested=[r for r in comparisons if r['status']=='evaluated']
    adjusted=holm_adjust([r['paired_statistics']['two_sided_group_swap_p'] for r in tested])
    for r,p in zip(tested,adjusted,strict=True):r['paired_statistics']['holm_adjusted_p_within_this_family']=p
    return {'scope':'exploratory post-result matched-coverage analysis; no confirmation fitting or threshold change',
        'claims':len(gold),'control_accepted_answers':sum(r['candidate_label'] is not None for r in control),
        'ranking_rules':list(RULES),'tie_rule':'frozen claim order',
        'group_rule':'connected normalized claims or any visible retrieved page identifier',
        'groups':len(groups),'largest_group':max(map(len,groups)),
        'group_claim_ids':[[gold[i]['id'] for i in g] for g in groups],
        'multiplicity_family_comparisons':len(tested),'comparisons':comparisons,
        'limitations':['Coverage counts come from observed frozen gates, not deployment thresholds for the ranking controls.',
            'Selected controls use only saved scores; gold labels are used for scoring only.',
            'Group bootstrap and swaps are conditional on frozen selections and scores; dependence beyond page/claim links is unmodeled.',
            'Holm correction covers this family only, not prior adaptive development or this post-result choice of analysis.',
            'No neural inference or new independent data; no policy is selected or retuned.']}

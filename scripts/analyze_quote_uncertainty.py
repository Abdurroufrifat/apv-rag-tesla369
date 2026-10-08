"""Descriptive group-bootstrap intervals on fixed saved attribution predictions."""
import json,re,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_directquote_study import spans,normalize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    full=ROOT/'artifacts/trained_quote_relation_v1';abl=ROOT/'artifacts/quote_relation_ablation_v3';out=ROOT/'artifacts/quote_uncertainty_v1'
    if out.exists():raise FileExistsError(out)
    for folder in (full,abl):
        for n,h in read(folder/'receipt.json').items():
            if sha(folder/n)!=h:raise ValueError('Export differs: '+str(folder/n))
    assert read(full/'split_ids.json')==read(abl/'split_ids.json')
    a=read(full/'predictions.json');b={x['id']:x for x in read(abl/'predictions.json')};ids={x['id']:i for i,x in enumerate(a)}
    assert len(ids)==798 and set(ids)==set(b)
    parent=list(range(len(a)));owner={};seen=set()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    corpus=ROOT/'data/external/directquote/frozen_v1/truecased.txt';assert sha(corpus)==read(full/'protocol.json')['source_sha256']
    for block in re.split(r'\n\s*\n',corpus.read_text().strip()):
        pairs=[line.rsplit(None,1) for line in block.splitlines() if line.strip()];tokens=[x[0] for x in pairs];key=hashlib.sha256(normalize(' '.join(tokens)).encode()).hexdigest()
        if key not in ids or key in seen:continue
        ranges=spans([x[1] for x in pairs]);quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
        if len(quotes)!=1:continue
        kind,start,end=quotes[0]
        if kind=='Unknown':
            if speakers:continue
        else:
            if len(speakers)!=1:continue
            _,sa,sb=speakers[0]
            if not ((kind=='LeftSpeaker' and sb<=start) or (kind=='RightSpeaker' and sa>=end)):continue
        seen.add(key);words=normalize(' '.join(tokens[start:end])).split();position=ids[key]
        for j in range(len(words)-4):
            shingle=tuple(words[j:j+5])
            if shingle in owner:parent[find(position)]=find(owner[shingle])
            else:owner[shingle]=position
    assert len(seen)==798
    roots=sorted({find(i) for i in range(len(a))});assert len(roots)==read(full/'protocol.json')['groups']['test']
    correct_a=np.array([normalize(x['speaker'] or '')==normalize(x['predicted_speaker'] or '') for x in a],float)
    correct_b=np.array([normalize(x['speaker'] or '')==normalize(b[x['id']]['predicted_speaker'] or '') for x in a],float)
    unknown=np.array([x['speaker'] is None for x in a],float);false=np.array([x['speaker'] is None and x['predicted_speaker'] is not None for x in a],float)
    stats=[]
    for group in roots:
        mask=np.array([find(i)==group for i in range(len(a))]);stats.append([mask.sum(),correct_a[mask].sum(),correct_b[mask].sum(),unknown[mask].sum(),false[mask].sum()])
    stats=np.array(stats);draws=np.random.default_rng(369).integers(0,len(roots),size=(4000,len(roots)));sums=stats[draws].sum(axis=1)
    interval=lambda values:np.quantile(values,[.025,.975]).tolist()
    result={'cases':798,'quote_shingle_groups':len(roots),'bootstrap_draws':4000,'seed':369,'full_accuracy':float(correct_a.mean()),'full_accuracy_group_95_interval':interval(sums[:,1]/sums[:,0]),'ablation_accuracy':float(correct_b.mean()),'ablation_accuracy_group_95_interval':interval(sums[:,2]/sums[:,0]),'paired_accuracy_difference':float((correct_a-correct_b).mean()),'paired_difference_group_95_interval':interval((sums[:,1]-sums[:,2])/sums[:,0]),'unknown_false_attribution_rate':float(false.sum()/unknown.sum()),'unknown_false_attribution_group_95_interval':interval(sums[:,4]/sums[:,3]),'scope':'Descriptive unadjusted intervals on observed filtered test cohort. Resampling quotation-shingle groups, not unavailable article or publisher groups. One training seed. Post-hoc ablation; no independent superiority claim.'}
    out.mkdir();(out/'group_assignments.json').write_text(json.dumps({x['id']:str(find(i)) for i,x in enumerate(a)},indent=2));(out/'summary.json').write_text(json.dumps(result,indent=2))
    files=[full/'predictions.json',abl/'predictions.json',full/'protocol.json',Path(__file__),corpus]
    (out/'receipt.json').write_text(json.dumps({'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'outputs':{n:sha(out/n) for n in ('summary.json','group_assignments.json')}},indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

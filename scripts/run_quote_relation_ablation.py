"""Train on separate quote groups; tune threshold on development, evaluate once."""
import json,re,sys,hashlib,joblib,pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_directquote_study import spans,normalize
from apv_rag.quote_relation import token_features as full_features,decode_speaker

def token_features(tokens,quote):
    return [{k:v for k,v in row.items() if k not in {"quote_side","distance_bin","quote_inside","side_word"}} for row in full_features(tokens,quote)]
from apv_rag.splits import sha256,write_json_atomic

def main():
    out=ROOT/'artifacts/quote_relation_ablation_v3'
    if out.exists():raise FileExistsError('Completed/prepared experiment exists')
    source=ROOT/'data/external/directquote/frozen_v1/truecased.txt'
    observed=json.loads((ROOT/'artifacts/directquote_preflight_v1/inputs.json').read_text())
    oldparagraphs={normalize(x['paragraph']) for x in observed};oldquotes=[set(normalize(x['quote']).split()) for x in observed]
    records=[];seen=set()
    for block in re.split(r'\n\s*\n',source.read_text().strip()):
        pairs=[line.rsplit(None,1) for line in block.splitlines() if line.strip()];tokens=[x[0] for x in pairs];tags=[x[1] for x in pairs];ranges=spans(tags)
        quotes=[x for x in ranges if x[0]!='Speaker'];speakers=[x for x in ranges if x[0]=='Speaker']
        if len(quotes)!=1 or len(tokens)>180:continue
        kind,a,b=quotes[0]
        if b-a<6:continue
        if kind=='Unknown':
            if speakers:continue
            speaker=None;target=[]
        else:
            if len(speakers)!=1:continue
            _,sa,sb=speakers[0]
            if not ((kind=='LeftSpeaker' and sb<=a) or (kind=='RightSpeaker' and sa>=b)):continue
            speaker=' '.join(tokens[sa:sb]);target=list(range(sa,sb))
        paragraph=normalize(' '.join(tokens));quote=normalize(' '.join(tokens[a:b]));qset=set(quote.split())
        if paragraph in oldparagraphs or any(len(qset&s)/max(1,len(qset|s))>=.8 for s in oldquotes):continue
        if paragraph in seen:continue
        seen.add(paragraph);key=hashlib.sha256(quote.encode()).hexdigest()
        records.append({'id':hashlib.sha256(paragraph.encode()).hexdigest(),'group':key,'tokens':tokens,'quote_span':[a,b],'speaker':speaker,'speaker_tokens':target})
    # Conservative lexical group: shared five-token quote sequences link paragraphs.
    parent=list(range(len(records)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(records):
        words=normalize(' '.join(row['tokens'][slice(*row['quote_span'])])).split()
        for j in range(len(words)-4):
            key=tuple(words[j:j+5])
            if key in owner:parent[find(i)]=find(owner[key])
            else:owner[key]=i
    grouped={}
    for i,row in enumerate(records):grouped.setdefault(find(i),[]).append(row)
    groups=sorted(grouped.values(),key=lambda group:hashlib.sha256(('369'+min(x['id'] for x in group)).encode()).hexdigest())
    n=len(groups);train_groups=groups[:int(n*.7)];dev_groups=groups[int(n*.7):int(n*.85)];test_groups=groups[int(n*.85):]
    flatten=lambda groups:[r for g in groups for r in g]
    train,dev,test=map(flatten,(train_groups,dev_groups,test_groups))
    assert set(x['id'] for x in train).isdisjoint(x['id'] for x in test)
    out.mkdir();write_json_atomic(out/'protocol.json',{'source_sha256':sha256(source),'scope':'Post-hoc matched ablation removing quote-position/distance features; test outcomes previously observed. Filtered one-quote task, not full RAG or historical authentication. Original100cases and near-quote matches excluded. Article/publisher IDs unavailable; no article-disjoint or semantic/pretraining isolation claim.', 'split':'70/15/15 deterministic quote-shingle connected groups; five-token quote overlap connects groups. No official split. No gold speaker fields enter features. Remove quote_side, distance_bin, quote_inside and side_word; local word/context/case remain. Development and test natural filtered class proportions.', 'model':'DictVectorizer + SGD logistic loss alpha0.0001 balanced weights max_iter50 tol0.001 seed369', 'thresholds':[.3,.5,.7,.8,.9,.95], 'selection':'Maximum development exact normalized speaker accuracy; ties prefer higher threshold. Test evaluated only after selected threshold saved.', 'rows':{'train':len(train),'development':len(dev),'test':len(test)},'groups':{'train':len(train_groups),'development':len(dev_groups),'test':len(test_groups)},'code':{p.relative_to(ROOT).as_posix():sha256(p) for p in (Path(__file__),ROOT/'src/apv_rag/quote_relation.py',ROOT/'scripts/prepare_directquote_study.py')}})
    write_json_atomic(out/'split_ids.json',{name:[x['id'] for x in data] for name,data in [('train',train),('development',dev),('test',test)]})
    vectorizer=DictVectorizer();features=[f for row in train for f in token_features(row['tokens'],row['quote_span'])];targets=[int(i in row['speaker_tokens']) for row in train for i in range(len(row['tokens']))]
    print('Fitting',len(train),'paragraphs',len(targets),'tokens',flush=True)
    model=SGDClassifier(loss='log_loss',alpha=.0001,class_weight='balanced',max_iter=50,tol=.001,random_state=369)
    with threadpool_limits(1):model.fit(vectorizer.fit_transform(features),targets)
    def score(data):return [model.predict_proba(vectorizer.transform(token_features(row['tokens'],row['quote_span'])))[:,1] for row in data]
    devscores=score(dev);choices=[]
    for threshold in (.3,.5,.7,.8,.9,.95):
        correct=sum(normalize(decode_speaker(row['tokens'],row['quote_span'],p,threshold) or '')==normalize(row['speaker'] or '') for row,p in zip(dev,devscores));choices.append({'threshold':threshold,'correct':correct,'cases':len(dev)})
    best=max(choices,key=lambda x:(x['correct'],x['threshold']))['threshold'];write_json_atomic(out/'selection.json',{'threshold':best,'development':choices});(out/'model.pkl').write_bytes(pickle.dumps({'vectorizer':vectorizer,'model':model,'threshold':best},protocol=5)); saved=pickle.loads((out/'model.pkl').read_bytes()); assert saved['threshold']==best
    testscores=score(test);predictions=[{'id':row['id'],'group':row['group'],'speaker':row['speaker'],'predicted_speaker':decode_speaker(row['tokens'],row['quote_span'],p,best)} for row,p in zip(test,testscores)]
    correct=sum(normalize(x['speaker'] or '')==normalize(x['predicted_speaker'] or '') for x in predictions);known=[x for x in predictions if x['speaker'] is not None];unknown=[x for x in predictions if x['speaker'] is None]
    summary={'test_cases':len(test),'correct':correct,'accuracy':correct/len(test),'known_cases':len(known),'known_correct':sum(normalize(x['speaker'])==normalize(x['predicted_speaker'] or '') for x in known),'unknown_cases':len(unknown),'unknown_false_attributions':sum(x['predicted_speaker'] is not None for x in unknown),'always_unknown_accuracy':len(unknown)/len(test),'threshold':best,'scope':'Post-hoc ablation on the same frozen split; no independent confirmation or Qwen comparison.'}
    write_json_atomic(out/'predictions.json',predictions);write_json_atomic(out/'summary.json',summary);write_json_atomic(out/'receipt.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

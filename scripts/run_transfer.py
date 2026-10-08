import ast,json,re,sys,zipfile,hashlib,collections,random
from pathlib import Path
BASE=Path(__file__).resolve().parent
PIN='c3eee833c035bf7b8c84f55b58c0bf7346cfae76'
TOKEN=r"\w+(?:['’]\w+)*|[^\w\s]"
def parse(ann,text):
 spans={};events=[]
 for line in ann.splitlines():
  if re.match(r'^T\d+\t',line):
   ident,desc,literal=line.split('\t',2);kind,offsets=desc.split(' ',1)
   if ';' in offsets:continue
   a,b=map(int,offsets.split());assert text[a:b].splitlines()[0]==literal,(ident,'offset mismatch')
   spans[ident]=(kind,a,b,literal)
  elif re.match(r'^E\d+\t',line):
   ident,desc=line.split('\t');fields=dict(x.split(':',1) for x in desc.split());events.append((ident,fields))
 return spans,events

def extract(text,ann):
 ss,events=parse(ann,text);rows=[]
 paragraphs=[(m.start(),m.end()) for m in re.finditer(r'[^\n]+',text)]
 for eid,e in events:
  if e.get('Content') not in ss or e.get('Source') not in ss:continue
  _,a,b,quote=ss[e['Content']];_,sa,sb,source=ss[e['Source']]
  if not ((quote.startswith('“') and quote.endswith('”')) or (quote.startswith('"') and quote.endswith('"'))):continue
  ps=[(l,r) for l,r in paragraphs if l<=min(a,sa) and max(b,sb)<=r]
  if len(ps)!=1:continue
  l,r=ps[0];matches=list(re.finditer(TOKEN,text[l:r]));tokens=[m.group() for m in matches];offsets=[(l+m.start(),l+m.end()) for m in matches]
  qi=[i for i,(u,v) in enumerate(offsets) if a<=u and v<=b];si=[i for i,(u,v) in enumerate(offsets) if sa<=u and v<=sb]
  if not qi or not si or len(tokens)>180 or len(qi)<6:continue
  if offsets[si[0]][0]!=sa or offsets[si[-1]][1]!=sb:continue
  if set(qi)&set(si):continue
  rows.append({'event':eid,'paragraph_offset':l,'tokens':tokens,'quote_span':[min(qi),max(qi)+1],'gold_tokens':si,'gold':' '.join(tokens[i] for i in si),'quote':quote})
 counts=collections.Counter(x['paragraph_offset'] for x in rows)
 return [x for x in rows if counts[x['paragraph_offset']]==1]

def baseline(tokens,quote):
 a,b=quote;hits=[];i=0
 while i<len(tokens):
  if a<=i<b or not tokens[i][0].isupper() or not tokens[i].isalpha():i+=1;continue
  j=i+1
  while j<len(tokens) and not a<=j<b and tokens[j][0].isupper() and tokens[j].isalpha():j+=1
  distance=a-j if j<=a else i-b
  hits.append((distance,i,' '.join(tokens[i:j])));i=j
 return min(hits)[2] if hits else None

def norm(x):return re.findall(r'\w+',(x or '').casefold())
def shingles(x):
 w=norm(x);return {tuple(w[i:i+5]) for i in range(len(w)-4)}

def main():
 sys.path[:0]=[str(BASE.parent/'apv-rag-tesla369/scripts'),str(BASE.parent/'apv-rag-tesla369/src')]
 from prepare_directquote_study import spans
 from apv_rag.quote_inference import load_model,infer
 archive=zipfile.ZipFile(BASE/'repository.zip');prefix=archive.namelist()[0]
 def cohort(split):
  rows=[];files=[n for n in archive.namelist() if n.startswith(prefix+'data/'+split+'/attributions/') and n.endswith('.ann')]
  for name in files:
   stem=Path(name).stem.rsplit('_',1)[0];tp=prefix+'data/'+split+'/text/'+stem+'.txt'
   text=archive.read(tp).decode();ann=archive.read(name).decode()
   for row in extract(text,ann):rows.append({**row,'article_id':stem,'id':stem+':'+row['event']})
  return rows
 training=cohort('train')
 assert training and baseline(['Jane','Smith','said','“','hello','”'],[3,6])=='Jane Smith'
 try:parse('T1\tSource 0 4\twrong','Jane')
 except AssertionError:pass
 else:raise AssertionError('parser accepts corrupt offset')
 (BASE/'frozen_transfer_protocol.json').write_text(json.dumps({'revision':PIN,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_parser_cases':len(training),'selection':'Continuous direct content delimited by double straight or curly quotes; source and quote in same newline paragraph; <=180 tokens, quote>=6 tokens, source token-aligned and outside quote; exactly one eligible relation per paragraph.','baseline':'Closest consecutive alphabetic capitalized token run outside quote; tie leftmost. No gold input.','score':'Exact normalized source-span match; absent prediction is an error. Known-source subset only; cannot measure UNKNOWN specificity.','threshold':.9,'test_tuning':False,'scope':'Filtered PolNeAR span transfer; not official full attribution task or entity adapter validation.'},indent=2))
 qs=[];index={}
 def add(s):
  h=shingles(s);j=len(qs);qs.append(h)
  for k in h:index.setdefault(k,set()).add(j)
 source=BASE.parent/'apv-rag-tesla369/data/external/directquote/frozen_v1/truecased.txt'
 for block in re.split(r'\n\s*\n',source.read_text().strip()):
  pairs=[line.rsplit(None,1) for line in block.splitlines() if line.strip()]
  for kind,a,b in spans([x[1] for x in pairs]):
   if kind!='Speaker':add(' '.join(x[0] for x in pairs[a:b]))
 for x in json.loads((BASE.parent/'quote_transfer_candidate/blind_inputs.json').read_text()):add(x['quotation'])
 rows=cohort('test');kept=[];excluded=0
 for x in rows:
  h=shingles(x['quote']);c=set().union(*(index.get(k,set()) for k in h)) if h else set()
  if any(len(h&qs[j])/max(1,len(h|qs[j]))>=.8 for j in c):excluded+=1
  else:kept.append(x)
 (BASE/'frozen_test_ids.json').write_text(json.dumps([x['id'] for x in kept]))
 saved=load_model(BASE.parent/'apv-rag-tesla369/artifacts/trained_quote_relation_v1/model.joblib');results=[]
 for x in kept:
  model=infer({'tokens':x['tokens'],'quote_span':x['quote_span']},saved)['speaker_candidate'];ref=baseline(x['tokens'],x['quote_span']);results.append({'id':x['id'],'article_id':x['article_id'],'gold':x['gold'],'model':model,'baseline':ref,'model_correct':norm(model)==norm(x['gold']),'baseline_correct':norm(ref)==norm(x['gold'])})
 assert results
 summary={'training_schema_cases':len(training),'test_eligible_before_overlap':len(rows),'overlap_excluded':excluded,'cases':len(results),'articles':len({x['article_id'] for x in results}),'scope':'Filtered known-source direct quote span transfer. No UNKNOWN claims or historical authenticity. No test fitting.'}
 for method in ['model','baseline']:
  correct=sum(x[method+'_correct'] for x in results);summary[method]={'correct':correct,'accuracy':correct/len(results),'answered':sum(x[method] is not None for x in results)}
 (BASE/'test_predictions.json').write_text(json.dumps(results,indent=2));(BASE/'transfer_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

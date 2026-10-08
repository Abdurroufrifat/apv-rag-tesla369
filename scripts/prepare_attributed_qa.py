"""Import existing attribution labels and score fixed reference policies; no fitting."""
import csv,io,json,zipfile,hashlib,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='01f114b203ac98e9374471d26dba5e5e07f93409'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def evaluate(gold,pred):
 tp=sum(y and p for y,p in zip(gold,pred));tn=sum(not y and not p for y,p in zip(gold,pred));fp=sum(not y and p for y,p in zip(gold,pred));fn=sum(y and not p for y,p in zip(gold,pred))
 pos=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0;neg=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
 return {'cases':len(gold),'accuracy':(tp+tn)/len(gold),'macro_f1':(pos+neg)/2,'true_supported':tp,'true_unsupported':tn,'false_supported':fp,'false_unsupported':fn}
def main():
 source=ROOT/'data/external/attributed_qa_v1';out=ROOT/'artifacts/attributed_qa_reference_v1'
 if out.exists():raise FileExistsError(out)
 with zipfile.ZipFile(source/'ratings.zip') as z:
  assert z.testzip() is None
  rows=list(csv.DictReader(io.StringIO(z.read('ratings.csv').decode('utf-8-sig'))))
 assert len(rows)==83030
 labeled=[(i,x) for i,x in enumerate(rows) if x['human_rating'] in ('Y','N')]
 assert all(x['auto_ais'] in ('Y','N') and x['passage'].strip() and x['question'].strip() for _,x in labeled)
 inputs=[];labels=[];preds=[]
 for i,x in labeled:
  identifier=hashlib.sha256((str(i)+'\0'+x['system_name']+'\0'+x['question']+'\0'+x['answer']+'\0'+x['attribution']).encode()).hexdigest()
  inputs.append({'id':identifier,'question':x['question'],'answer':x['answer'],'passage':x['passage'],'attribution':x['attribution'],'system_name':x['system_name']})
  labels.append({'id':identifier,'human_supported':x['human_rating']=='Y'})
  a=norm(x['answer']);p=norm(x['passage'])
  preds.append({'id':identifier,'released_autoais':x['auto_ais']=='Y','answer_containment':bool(a) and (' '+a+' ') in (' '+p+' ')})
 assert len({x['id'] for x in inputs})==len(inputs)
 gold=[x['human_supported'] for x in labels]
 results={policy:evaluate(gold,[x[policy] for x in preds]) for policy in ('released_autoais','answer_containment')}
 results['always_supported']=evaluate(gold,[True]*len(gold));results['always_unsupported']=evaluate(gold,[False]*len(gold))
 out.mkdir()
 def write(name,x):(out/name).write_text(json.dumps(x,indent=2)+'\n')
 write('protocol.json',{'commit':COMMIT,'source_zip_sha256':sha(source/'ratings.zip'),'code_sha256':sha(Path(__file__)),'scope':'Fixed reference audit of released question-answer-passage support labels. Not source authenticity or our method evaluation. No fitting or threshold tuning. OpenNQ development examples in upstream release. Prior APV-RAG corpus overlap not checked in recovered workspace; do not claim independent heldout evaluation.','selection':'All rows with upstream human_rating Y or N; remaining rows excluded, no relabeling. Input/gold/prediction files separated.','policies':{'released_autoais':'Use upstream auto_ais decision as released; no new neural inference','answer_containment':'Normalized complete answer token sequence present in cited passage; weak reference, ignores question semantics'},'caution':'Rows share questions and sources across 23 upstream systems, so row independence is not assumed. No confidence intervals supplied.'})
 write('inputs.json',inputs);write('gold.json',labels);write('reference_predictions.json',preds)
 summary={'raw_rows':len(rows),'human_label_counts':dict(collections.Counter(x['human_rating'] for x in rows)),'included_rows':len(inputs),'unique_questions':len({norm(x['question']) for x in inputs}),'unique_attribution_ids':len({x['attribution'] for x in inputs}),'metrics':results,'upstream_documentation_label_count':23000,'observed_binary_labels':len(inputs),'note':'README says 23000 human labels; actual file has 21189 Y/N and 61841 dash values. Excluded dash is unavailable rating, not unsupported.'}
 write('summary.json',summary);write('receipt.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

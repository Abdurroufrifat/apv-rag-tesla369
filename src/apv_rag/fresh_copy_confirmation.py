"""Label-blind exclusions and supplied-evidence adapter for fresh confirmation."""
import re
from urllib.parse import quote


def normalized(text):
    return ' '.join(re.findall(r'\w+', str(text).casefold().replace('_', ' ')))


def select_fresh(rows, prior_ids, prior_pages, excluded_texts):
    parents={str(r['claim_id']):str(r['claim_id']) for r in rows}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]]
            x=parents[x]
        return x
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parents[max(a,b)]=min(a,b)
    pages={}
    for r in rows:
        i=str(r['claim_id'])
        for e in r['evidences']:
            page=normalized(e['article'])
            if page in pages:
                join(i,pages[page])
            else:
                pages[page]=i
    previous={normalized(s) for s in excluded_texts}
    prior_pages={normalized(s) for s in prior_pages}
    token_sets=[set(s.split()) for s in previous if s]
    blocked=set()
    for r in rows:
        text=normalized(r['claim']); tokens=set(text.split())
        overlap=(text in previous or any(len(tokens&p)/len(tokens|p)>=0.8 for p in token_sets if tokens|p))
        if str(r['claim_id']) in prior_ids or overlap or any(normalized(e['article']) in prior_pages for e in r['evidences']):
            blocked.add(find(str(r['claim_id'])))
    seen=set(); selected=[]; groups={}
    for r in sorted(rows,key=lambda r:str(r['claim_id'])):
        i=str(r['claim_id']); text=normalized(r['claim'])
        if find(i) not in blocked and text not in seen:
            selected.append(r);groups[i]=find(i);seen.add(text)
    return selected,groups


def model_input(row):
    return {'claim':row['claim'], 'questions':[{
        'question':'What does the supplied evidence say about this claim?',
        'answers':[{'answer':e['evidence'], 'answer_type':'Extractive',
                    'source_medium':'Web text',
                    'source_url':'https://en.wikipedia.org/wiki/'+quote(e['article'].replace(' ','_'),safe='')}
                   for e in row['evidences']]}]}

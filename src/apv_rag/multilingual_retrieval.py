"""Claim-only lexical retrieval from an explicitly limited benchmark excerpt pool."""
import hashlib
import re
import unicodedata

from apv_rag.retrieval import BM25Index

METHODS=('unicode_words_cjk_1_2','stock_unicode_words')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def excerpt_id(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def is_cjk(c):
    n=ord(c)
    return unicodedata.category(c).startswith('L') and (
        0x3040<=n<=0x30ff or 0x3400<=n<=0x4dbf or 0x4e00<=n<=0x9fff or
        0xf900<=n<=0xfaff or 0x20000<=n<=0x2ebef)


def multilingual_tokens(text):
    text=normalize(text)
    words=re.findall(r'[^\W_]+',''.join(' ' if is_cjk(c) else c for c in text))
    tokens=['w_'+w for w in words]
    run=[]
    def flush():
        tokens.extend('c1_'+c for c in run)
        tokens.extend('c2_'+a+b for a,b in zip(run,run[1:]))
        run.clear()
    for c in text:
        if is_cjk(c):run.append(c)
        else:flush()
    flush()
    return tokens


def build_pool(texts):
    """Use text alone, collapse normalized exact duplicates, choose deterministic raw text."""
    if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError('Nonempty evidence strings required')
    pool={}
    for text in texts:
        key=excerpt_id(text)
        pool[key]=min(pool.get(key,text),text)
    return [{'id':key,'text':text} for key,text in sorted(pool.items())]


def retrieve(queries,pool,method):
    if method not in METHODS or not pool or any(set(d)!={'id','text'} or d['id']!=excerpt_id(d['text']) for d in pool):
        raise ValueError('Unexpected retrieval method or corpus schema')
    if len({d['id'] for d in pool})!=len(pool):raise ValueError('Repeated corpus IDs')
    pool=sorted(pool,key=lambda d:d['id'])
    encode=(lambda text:' '.join(multilingual_tokens(text))) if method==METHODS[0] else (lambda text:text)
    index=BM25Index([encode(d['text']) for d in pool],k1=1.2,b=0.75)
    rows=[]; seen=set()
    for q in queries:
        if (set(q)!={'row','claim_id','claim'} or type(q['row']) is not int or q['row']<0 or
                q['row'] in seen or type(q['claim_id']) is not int or
                not isinstance(q['claim'],str) or not q['claim'].strip()):
            raise ValueError('Expected unique claim-only retrieval inputs')
        seen.add(q['row'])
        evidence=[dict(pool[i],score=float(value)) for i,value in index.search(encode(q['claim']),top_k=3) if value>0]
        rows.append(dict(q,evidence=evidence))
    return rows


def score(rows,targets):
    """Target-excerpt matching, with unrelated NEI pairs separated from verifiable pairs."""
    if len(rows)!=len(targets) or not rows:
        raise ValueError('Expected nonempty aligned retrieval and scoring records')
    labels={'SUPPORTS','REFUTES','NOT ENOUGH INFO'}
    groups={'all_pairs':[],'verifiable':[],'nei_descriptive_only':[]}
    for r,t in zip(rows,targets,strict=True):
        if (r['row'],r['claim_id'])!=(t['row'],t['claim_id']) or t['label'] not in labels:
            raise ValueError('Target scoring alignment differs')
        ids=[d['id'] for d in r['evidence']]
        rank=ids.index(t['target_id'])+1 if t['target_id'] in ids else None
        groups['all_pairs'].append(rank)
        groups['nei_descriptive_only' if t['label']=='NOT ENOUGH INFO' else 'verifiable'].append(rank)
    result={}
    for name,ranks in groups.items():
        n=len(ranks)
        result[name]={'queries':n,
            'target_hit_at_1':sum(r==1 for r in ranks)/n if n else None,
            'target_hit_at_3':sum(r is not None for r in ranks)/n if n else None,
            'target_hits_at_3':sum(r is not None for r in ranks),
            'mean_reciprocal_rank_at_3':sum(1/r for r in ranks if r is not None)/n if n else None}
    result['queries_without_retrieved_excerpt']=sum(not r['evidence'] for r in rows)
    return result

"""Claim-only lexical reranking using a common candidate sentence index."""
from apv_rag.retrieval import BM25Index

def rerank_candidates(claim, corpus, candidates, top_k=3):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if len({i for i,_ in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate')
    sentences=[]; owners=[]
    for position,(index,_) in enumerate(candidates):
        for sentence in corpus[index]['abstract']:
            sentences.append(sentence); owners.append(position)
    scores=[0.0]*len(candidates)
    if sentences:
        for index,score in BM25Index(sentences).search(claim,top_k=len(sentences)):
            scores[owners[index]]=max(scores[owners[index]],score)
    order=sorted(range(len(candidates)),key=lambda i:(-scores[i],i))
    return [candidates[i] for i in order[:top_k]]

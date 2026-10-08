"""Claim-only lexical sentence selection; no labels or gold evidence inputs."""
from apv_rag.retrieval import BM25Index


def select_sentences(claim, sentences, top_k=3):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not sentences:
        return {'text': '', 'sentence_indices': [], 'sentence_scores': []}
    ranked = BM25Index(sentences).search(claim, top_k=top_k)
    if not ranked:
        ranked = [(i, 0.0) for i in range(min(top_k, len(sentences)))]
    return {
        'text': ' '.join(sentences[i] for i, _ in ranked),
        'sentence_indices': [i for i, _ in ranked],
        'sentence_scores': [float(s) for _, s in ranked],
    }

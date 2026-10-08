"""Expanded features from existing excerpt/NLI audits, without neural inference."""

import re
from urllib.parse import urlsplit

import numpy as np

from apv_rag.nli_comparison import nli_features

FEATURE_NAMES = (
    "contradiction_max",
    "entailment_max",
    "neutral_max",
    "contradiction_mean",
    "entailment_mean",
    "neutral_mean",
    "excerpt_count",
    "contradiction_std",
    "entailment_std",
    "neutral_std",
    "mean_normalized_nli_entropy",
    "contradiction_entailment_max_product",
    "neutral_argmax_fraction",
    "claim_token_overlap_max",
    "claim_token_overlap_mean",
    "unique_source_hosts",
    "mean_log1p_excerpt_words",
    "max_log1p_excerpt_words",
)


def expanded_features(claim, premises, urls, scores):
    if len(premises) != len(urls) or len(premises) != len(scores) or len(premises) > 5:
        raise ValueError("excerpt texts, URLs and NLI scores must align, with at most five rows")
    base = nli_features(scores)
    if not premises:
        return np.zeros(len(FEATURE_NAMES))
    matrix = np.asarray(scores, dtype=float)
    claim_tokens = set(re.findall(r"\w+", claim.casefold()))
    tokens = [re.findall(r"\w+", text.casefold()) for text in premises]
    overlap = [
        len(claim_tokens.intersection(words)) / len(claim_tokens) if claim_tokens else 0
        for words in tokens
    ]
    entropy = -(matrix * np.log(np.clip(matrix, 1e-12, 1))).sum(axis=1) / np.log(3)
    hosts = {urlsplit(url).hostname.casefold() for url in urls if urlsplit(url).hostname}
    lengths = np.log1p([len(words) for words in tokens])
    return np.concatenate(
        [
            base,
            matrix.std(axis=0),
            [
                entropy.mean(),
                base[0] * base[1],
                (matrix.argmax(axis=1) == 2).mean(),
                max(overlap),
                np.mean(overlap),
                len(hosts),
                lengths.mean(),
                lengths.max(),
            ],
        ]
    )

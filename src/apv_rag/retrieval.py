"""Deterministic lexical BM25 retrieval; no verdict or label inputs."""

import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    return re.findall(r"\w+", str(text).casefold())


class BM25Index:
    def __init__(self, documents, *, k1=1.2, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0, 1]")
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        for position, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query, top_k=10):
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            entries = self.postings.get(term, [])
            if not entries:
                continue
            inverse_frequency = math.log(
                1 + (len(self.lengths) - len(entries) + 0.5) / (len(entries) + 0.5)
            )
            for position, frequency in entries:
                normalization = self.k1 * (
                    1 - self.b + self.b * self.lengths[position] / self.average_length
                )
                scores[position] += (
                    inverse_frequency * frequency * (self.k1 + 1) / (frequency + normalization)
                )
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]

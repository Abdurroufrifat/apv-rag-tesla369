"""Leakage-safe group construction and deterministic split assignment."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_claim(text: str) -> str:
    """Normalize claim text only for grouping, never for model input."""

    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_url(url: str | None) -> str:
    """Normalize an article URL conservatively for grouping."""

    if not url or not url.strip():
        return ""
    value = url.strip()
    parts = urlsplit(value)
    if not parts.netloc:
        return value.casefold().rstrip("/")
    scheme = parts.scheme.casefold() or "https"
    host = parts.netloc.casefold()
    path = re.sub(r"/+", "/", parts.path).rstrip("/")
    return urlunsplit((scheme, host, path, parts.query, ""))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    validation_indices: list[int]
    excluded_indices: list[int]
    group_by_index: dict[int, str]


def build_connected_groups(records: list[dict[str, Any]]) -> tuple[list[list[int]], list[int]]:
    """Build components linked by exact normalized claim or article URL."""

    eligible = [index for index, row in enumerate(records) if normalize_claim(row.get("claim", ""))]
    excluded = sorted(set(range(len(records))) - set(eligible))
    union_find = UnionFind(len(records))
    seen_claim: dict[str, int] = {}
    seen_article: dict[str, int] = {}
    for index in eligible:
        row = records[index]
        claim_key = normalize_claim(row["claim"])
        article_key = normalize_url(row.get("fact_checking_article"))
        if claim_key in seen_claim:
            union_find.union(index, seen_claim[claim_key])
        else:
            seen_claim[claim_key] = index
        if article_key:
            if article_key in seen_article:
                union_find.union(index, seen_article[article_key])
            else:
                seen_article[article_key] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        components[union_find.find(index)].append(index)
    groups = sorted((sorted(group) for group in components.values()), key=lambda row: row[0])
    return groups, excluded


def _assignment_score(
    counts: Counter[str], total: int, target_counts: dict[str, float], target_total: float
) -> float:
    label_score = sum(
        ((counts[label] - target) / max(target, 1.0)) ** 2
        for label, target in target_counts.items()
    )
    size_score = ((total - target_total) / max(target_total, 1.0)) ** 2
    return label_score + size_score


def make_grouped_split(
    records: list[dict[str, Any]],
    validation_fraction: float = 0.20,
    seed: int = 369,
    trials: int = 5000,
) -> SplitResult:
    """Assign connected components to train/validation deterministically."""

    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")
    groups, excluded = build_connected_groups(records)
    eligible = [index for group in groups for index in group]
    total_labels = Counter(records[index]["label"] for index in eligible)
    target_counts = {label: count * validation_fraction for label, count in total_labels.items()}
    target_total = len(eligible) * validation_fraction

    if trials < 100:
        raise ValueError("at least 100 deterministic random-search trials are required")
    rng = random.Random(seed)
    group_label_counts = [Counter(records[index]["label"] for index in group) for group in groups]
    best_score = float("inf")
    validation: list[int] = []
    for _ in range(trials):
        candidate: list[int] = []
        candidate_counts: Counter[str] = Counter()
        for group, counts in zip(groups, group_label_counts, strict=True):
            if rng.random() < validation_fraction:
                candidate.extend(group)
                candidate_counts.update(counts)
        if not candidate or len(candidate) == len(eligible):
            continue
        score = _assignment_score(candidate_counts, len(candidate), target_counts, target_total)
        if score < best_score or (score == best_score and sorted(candidate) < sorted(validation)):
            best_score = score
            validation = candidate

    if not validation:
        raise ValueError("could not construct a non-empty grouped validation split")

    validation_set = set(validation)
    train = sorted(index for index in eligible if index not in validation_set)
    validation = sorted(validation)
    group_by_index: dict[int, str] = {}
    for group in groups:
        digest = hashlib.sha256(",".join(map(str, group)).encode()).hexdigest()[:16]
        group_id = f"AVG-{digest}"
        for index in group:
            group_by_index[index] = group_id
    return SplitResult(train, validation, excluded, group_by_index)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write deterministic JSON through a same-directory temporary file."""

    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

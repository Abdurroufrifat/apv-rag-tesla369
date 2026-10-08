"""Answer-excerpt ownership proxies, not evidence relevance judgments."""


def ownership_metrics(records, documents, rankings):
    if len(records) != len(rankings):
        raise ValueError("rankings must align with records")
    lookup = {(d["text"], d["source_url"]): i for i, d in enumerate(documents)}
    eligible = hit1 = hit5 = empty = 0
    reciprocal = coverage = 0.0
    for record, ranking in zip(records, rankings, strict=True):
        if (
            len(ranking) > 5
            or len(set(ranking)) != len(ranking)
            or any(not isinstance(i, int) or not 0 <= i < len(documents) for i in ranking)
        ):
            raise ValueError("invalid top-five document positions")
        own = set()
        for question in record.get("questions") or []:
            for answer in question.get("answers") or []:
                key = (str(answer.get("answer") or "").strip(), str(answer.get("source_url") or ""))
                if key[0]:
                    if key not in lookup:
                        raise ValueError("claim excerpt missing from split corpus")
                    own.add(lookup[key])
        empty += not ranking
        if not own:
            continue
        eligible += 1
        matches = [j + 1 for j, i in enumerate(ranking) if i in own]
        hit1 += bool(matches and matches[0] == 1)
        hit5 += bool(matches)
        reciprocal += 1 / matches[0] if matches else 0
        coverage += len(set(ranking) & own) / len(own)
    return {
        "claims": len(records),
        "claims_with_answer_excerpts": eligible,
        "empty_retrievals": int(empty),
        "own_excerpt_hit_at_1": hit1 / eligible if eligible else None,
        "own_excerpt_hit_at_5": hit5 / eligible if eligible else None,
        "own_excerpt_mrr_at_5": reciprocal / eligible if eligible else None,
        "mean_own_excerpt_coverage_at_5": coverage / eligible if eligible else None,
    }

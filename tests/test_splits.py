from apv_rag.splits import (
    build_connected_groups,
    make_grouped_split,
    normalize_claim,
    normalize_url,
)


def records():
    return [
        {"claim": "Claim A", "fact_checking_article": "https://x.test/a/", "label": "Supported"},
        {"claim": " claim   a ", "fact_checking_article": "https://y.test/b", "label": "Supported"},
        {"claim": "Claim C", "fact_checking_article": "https://y.test/b", "label": "Refuted"},
        {"claim": "Claim D", "fact_checking_article": "https://z.test/d", "label": "Refuted"},
        {"claim": "", "fact_checking_article": "https://z.test/e", "label": "Supported"},
    ]


def test_normalization_is_conservative():
    assert normalize_claim("  CLAIM   A ") == "claim a"
    assert normalize_url("HTTPS://Example.COM/a/#fragment") == "https://example.com/a"


def test_transitive_grouping_and_empty_exclusion():
    groups, excluded = build_connected_groups(records())
    assert [0, 1, 2] in groups
    assert [3] in groups
    assert excluded == [4]


def test_split_is_complete_disjoint_and_group_safe():
    rows = records()
    result = make_grouped_split(rows, validation_fraction=0.40, seed=369)
    assert not (set(result.train_indices) & set(result.validation_indices))
    assert sorted(
        result.train_indices + result.validation_indices + result.excluded_indices
    ) == list(range(len(rows)))
    train_groups = {result.group_by_index[index] for index in result.train_indices}
    validation_groups = {result.group_by_index[index] for index in result.validation_indices}
    assert not (train_groups & validation_groups)


def test_split_is_deterministic():
    first = make_grouped_split(records(), validation_fraction=0.40, seed=369)
    second = make_grouped_split(records(), validation_fraction=0.40, seed=369)
    assert first == second

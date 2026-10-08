from apv_rag.feature_ablation import GROUPS, ablation_columns


def test_ablation_groups_partition_added_features_and_keep_base():
    added = [i for group in GROUPS.values() for i in group]
    assert sorted(added) == list(range(7, 18))
    variants = ablation_columns()
    assert len(variants) == 8
    for columns in variants.values():
        assert columns[:7] == list(range(7))
        assert len(columns) == len(set(columns))
    assert 15 not in variants["full_without_source_length"]

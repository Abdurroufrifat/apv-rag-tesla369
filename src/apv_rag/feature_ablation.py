"""Fixed feature groups for descriptive internal-validation ablations."""

GROUPS = {
    "nli_variation": tuple(range(7, 13)),
    "lexical_overlap": (13, 14),
    "source_length": (15, 16, 17),
}


def ablation_columns():
    variants = {"base_7": list(range(7)), "full_18": list(range(18))}
    for name, columns in GROUPS.items():
        variants[f"base_plus_{name}"] = list(range(7)) + list(columns)
        variants[f"full_without_{name}"] = [i for i in range(18) if i not in columns]
    return variants

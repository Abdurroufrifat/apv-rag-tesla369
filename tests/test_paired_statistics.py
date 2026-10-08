import numpy as np
import pytest

from apv_rag.paired_statistics import holm_adjust, macro_score


def test_macro_score_counts_absent_classes_as_zero():
    assert macro_score(np.array([0, 1]), np.array([0, 1]), 4) == 0.5


def test_holm_preserves_original_order_and_caps_at_one():
    assert holm_adjust([0.03, 0.01, 0.9]) == pytest.approx([0.06, 0.03, 0.9])

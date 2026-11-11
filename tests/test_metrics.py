import math

import numpy as np

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae, precision_recall_ndcg_at_k, rmse)


def test_precision_recall_ndcg_hand_computed():
    recs = np.array([[1, 2, 3, 4, 5]])
    relevant = [{1, 3, 9}]                                  # hits at rank 1 and rank 3
    p, r, n = precision_recall_ndcg_at_k(recs, relevant, k=5)
    assert p[0] == 2 / 5
    assert r[0] == 2 / 3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)       # 3 relevant items possible
    assert math.isclose(n[0], dcg / idcg)


def test_perfect_and_empty_rankings():
    recs = np.array([[1, 2], [7, 8]])
    p, r, n = precision_recall_ndcg_at_k(recs, [{1, 2}, {1}], k=2)
    assert (p == [1.0, 0.0]).all() and (r == [1.0, 0.0]).all() and (n == [1.0, 0.0]).all()


def test_coverage():
    recs = np.array([[1, 2], [2, 3]])
    assert catalog_coverage(recs, n_catalog=10, k=2) == 0.3


def test_diversity_extremes():
    item_ids = np.array([1, 2, 3])
    genre = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    same = intra_list_diversity(np.array([[1, 2]]), item_ids, genre, k=2)
    different = intra_list_diversity(np.array([[1, 3]]), item_ids, genre, k=2)
    assert math.isclose(same, 0.0, abs_tol=1e-9) and math.isclose(different, 1.0)


def test_rmse_mae():
    assert math.isclose(rmse([1, 2, 3], [1, 2, 5]), math.sqrt(4 / 3))
    assert math.isclose(mae([1, 2, 3], [2, 2, 5]), 1.0)

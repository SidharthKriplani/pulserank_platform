"""Unit tests for PulseRank ranking and evaluation math.

Tests verify the correctness of IPS formulas, DCG/NDCG metrics,
Gini coefficient, catalog coverage, propensity clipping, and
rank-monotonicity scoring — not just key existence.
"""

from __future__ import annotations

import math
import pytest

from src.pulserank.evaluation.position_bias import (
    dcg,
    estimate_click_propensity_by_rank,
    ndcg,
    rank_propensity_monotonicity_score,
    summarize_weight_distribution,
)
from src.pulserank.ranking.reranking import (
    catalog_coverage,
    relevance_ndcg,
    seller_gini,
)


# ---------------------------------------------------------------------------
# DCG
# ---------------------------------------------------------------------------

class TestDCG:
    def test_single_relevant_at_rank_1(self):
        # DCG = 1.0 / log2(2) = 1.0
        assert dcg([1.0], k=10) == pytest.approx(1.0)

    def test_single_relevant_at_rank_2(self):
        # DCG = 0 / log2(2) + 1 / log2(3)
        result = dcg([0.0, 1.0], k=10)
        assert result == pytest.approx(1.0 / math.log2(3))

    def test_all_zeros_returns_zero(self):
        assert dcg([0.0, 0.0, 0.0], k=10) == 0.0

    def test_k_truncates_list(self):
        full = dcg([1.0, 1.0, 1.0], k=3)
        truncated = dcg([1.0, 1.0, 1.0, 1.0, 1.0], k=3)
        assert full == pytest.approx(truncated)

    def test_higher_relevance_at_top_beats_bottom(self):
        top_heavy = dcg([1.0, 0.0, 0.0], k=3)
        bottom_heavy = dcg([0.0, 0.0, 1.0], k=3)
        assert top_heavy > bottom_heavy


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------

class TestNDCG:
    def test_perfect_ranking_returns_one(self):
        labels = [1.0, 1.0, 0.0, 0.0, 0.0]
        assert ndcg(labels, k=5) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert ndcg([], k=10) == 0.0

    def test_worst_ranking_less_than_one(self):
        labels = [0.0, 0.0, 1.0]  # relevant item at bottom
        assert ndcg(labels, k=3) < 1.0

    def test_all_zeros_returns_zero(self):
        assert ndcg([0.0, 0.0, 0.0], k=5) == 0.0

    def test_ndcg_bounded_zero_to_one(self):
        for labels in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.3]]:
            score = ndcg(labels, k=5)
            assert 0.0 <= score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# estimate_click_propensity_by_rank
# ---------------------------------------------------------------------------

class TestEstimateClickPropensity:
    def _make_impressions(self, clicks_by_rank: dict[int, tuple[int, int]]) -> list[dict]:
        """Build impression list from {rank: (total_impressions, clicks)}."""
        rows = []
        for rank, (total, clicks) in clicks_by_rank.items():
            rows += [{"display_rank": rank, "clicked": 1}] * clicks
            rows += [{"display_rank": rank, "clicked": 0}] * (total - clicks)
        return rows

    def test_propensity_decreases_with_rank(self):
        """Position bias: rank 1 should have higher propensity than rank 5."""
        impressions = self._make_impressions({
            1: (100, 30),   # 30% CTR at rank 1
            5: (100, 10),   # 10% CTR at rank 5
        })
        stats = estimate_click_propensity_by_rank(impressions, max_rank=5)
        assert stats[1]["empirical_propensity"] > stats[5]["empirical_propensity"]

    def test_floor_clips_zero_propensity(self):
        """Rank with 0 clicks gets clipped to floor."""
        impressions = self._make_impressions({1: (50, 10), 2: (50, 0)})
        stats = estimate_click_propensity_by_rank(impressions, max_rank=2, floor=0.01)
        assert stats[2]["clipped_propensity"] == pytest.approx(0.01)
        assert stats[2]["was_clipped"] is True
        assert stats[2]["ips_weight"] == pytest.approx(1.0 / 0.01)

    def test_not_clipped_when_above_floor(self):
        impressions = self._make_impressions({1: (100, 20)})
        stats = estimate_click_propensity_by_rank(impressions, max_rank=1, floor=0.01)
        assert stats[1]["was_clipped"] is False
        assert stats[1]["empirical_propensity"] == pytest.approx(0.20)

    def test_ips_weight_is_inverse_of_clipped_propensity(self):
        impressions = self._make_impressions({1: (100, 25)})
        stats = estimate_click_propensity_by_rank(impressions, max_rank=1, floor=0.01)
        prop = stats[1]["clipped_propensity"]
        assert stats[1]["ips_weight"] == pytest.approx(1.0 / prop)

    def test_empty_rank_gets_floor_propensity(self):
        """Rank with no impressions falls back to floor."""
        impressions = self._make_impressions({1: (10, 2)})
        stats = estimate_click_propensity_by_rank(impressions, max_rank=3, floor=0.05)
        assert stats[2]["clipped_propensity"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# rank_propensity_monotonicity_score
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_strictly_decreasing_returns_one(self):
        stats = {
            1: {"empirical_propensity": 0.30},
            2: {"empirical_propensity": 0.20},
            3: {"empirical_propensity": 0.10},
        }
        assert rank_propensity_monotonicity_score(stats) == pytest.approx(1.0)

    def test_strictly_increasing_returns_zero(self):
        stats = {
            1: {"empirical_propensity": 0.05},
            2: {"empirical_propensity": 0.15},
            3: {"empirical_propensity": 0.30},
        }
        assert rank_propensity_monotonicity_score(stats) == pytest.approx(0.0)

    def test_single_rank_returns_zero_comparisons(self):
        stats = {1: {"empirical_propensity": 0.20}}
        result = rank_propensity_monotonicity_score(stats)
        assert result == 0.0  # no comparisons possible


# ---------------------------------------------------------------------------
# seller_gini
# ---------------------------------------------------------------------------

class TestSellerGini:
    def _build_items(self, seller_map: dict[str, str]) -> dict[str, dict]:
        """seller_map: {item_id: seller_id}"""
        return {item_id: {"seller_id": sid} for item_id, sid in seller_map.items()}

    def test_single_seller_maximum_concentration(self):
        """All items from one seller → maximum Gini."""
        items = self._build_items({"i1": "s1", "i2": "s1", "i3": "s1"})
        lists = {"sess1": ["i1", "i2", "i3"]}
        gini = seller_gini(lists, items, k=10)
        assert gini >= 0.0  # single seller → Gini = 0 (all equal)

    def test_equal_seller_distribution_low_gini(self):
        """One item per unique seller → low Gini (max equality)."""
        n = 5
        items = self._build_items({f"i{i}": f"s{i}" for i in range(n)})
        lists = {"sess1": [f"i{i}" for i in range(n)]}
        gini = seller_gini(lists, items, k=10)
        assert gini < 0.3

    def test_gini_bounded_zero_to_one(self):
        items = self._build_items({"a": "s1", "b": "s1", "c": "s2", "d": "s3"})
        lists = {"sess1": ["a", "b", "c", "d"]}
        gini = seller_gini(lists, items, k=10)
        assert 0.0 <= gini <= 1.0

    def test_k_limits_exposure_counted(self):
        """Items beyond k are not counted."""
        items = self._build_items({"i1": "s1", "i2": "s1", "i3": "s2", "i4": "s2"})
        lists = {"sess1": ["i1", "i2", "i3", "i4"]}
        gini_k2 = seller_gini(lists, items, k=2)
        gini_k4 = seller_gini(lists, items, k=4)
        # k=2 sees only s1 twice → Gini = 0; k=4 sees both → different
        assert gini_k2 == pytest.approx(0.0, abs=1e-9)
        assert gini_k4 != gini_k2


# ---------------------------------------------------------------------------
# catalog_coverage
# ---------------------------------------------------------------------------

class TestCatalogCoverage:
    def test_full_coverage(self):
        lists = {"s1": ["i1", "i2"], "s2": ["i3", "i4"]}
        assert catalog_coverage(lists, total_items=4, k=10) == pytest.approx(1.0)

    def test_partial_coverage(self):
        lists = {"s1": ["i1", "i2"]}
        assert catalog_coverage(lists, total_items=4, k=10) == pytest.approx(0.5)

    def test_zero_total_items_returns_zero(self):
        assert catalog_coverage({"s1": ["i1"]}, total_items=0, k=10) == 0.0

    def test_k_limits_items_per_list(self):
        lists = {"s1": ["i1", "i2", "i3", "i4", "i5"]}
        cov_k2 = catalog_coverage(lists, total_items=5, k=2)
        cov_k5 = catalog_coverage(lists, total_items=5, k=5)
        assert cov_k2 < cov_k5

    def test_overlap_counted_once(self):
        """Same item recommended in multiple sessions still counts once."""
        lists = {"s1": ["i1", "i2"], "s2": ["i1", "i3"]}
        cov = catalog_coverage(lists, total_items=4, k=10)
        assert cov == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# relevance_ndcg (binary gain version in reranking)
# ---------------------------------------------------------------------------

class TestRelevanceNDCG:
    def test_perfect_binary_ranking(self):
        assert relevance_ndcg([1, 1, 0, 0], k=4) == pytest.approx(1.0)

    def test_zero_labels_returns_zero(self):
        assert relevance_ndcg([0, 0, 0], k=3) == 0.0

    def test_relevant_at_bottom_less_than_top(self):
        top = relevance_ndcg([1, 0, 0], k=3)
        bottom = relevance_ndcg([0, 0, 1], k=3)
        assert top > bottom

    def test_result_between_zero_and_one(self):
        for labels in [[1, 0], [0, 1, 0], [1, 1, 1]]:
            score = relevance_ndcg(labels, k=10)
            assert 0.0 <= score <= 1.0 + 1e-9

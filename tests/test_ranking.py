"""Tests for ranking model."""
import numpy as np
from src.poseguide.ranking import EmbeddingRanker

def test_ranker_empty():
    ranker = EmbeddingRanker(dim=64)
    results = ranker.rank(["outdoor", "beach"], top_k=3)
    assert results == []

def test_ranker_with_poses():
    ranker = EmbeddingRanker(dim=64)
    emb = np.random.randn(64)
    ranker.add_pose("standing_beach", emb, ["outdoor", "beach", "standing"])
    ranker.add_pose("sitting_office", np.random.randn(64), ["indoor", "office", "sitting"])
    results = ranker.rank(["beach", "outdoor"], top_k=2)
    assert len(results) == 2

def test_jaccard():
    ranker = EmbeddingRanker(dim=16)
    jac = ranker.jaccard_similarity(["a", "b", "c"], ["b", "c", "d"])
    assert jac == 0.5

def test_combined_rank():
    ranker = EmbeddingRanker(dim=16)
    ranker.add_pose("p1", np.random.randn(16), ["beach", "outdoor"])
    ranker.add_pose("p2", np.random.randn(16), ["office", "indoor"])
    results = ranker.combined_rank(["beach"], {"p1": ["beach", "outdoor"], "p2": ["office"]}, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "p1"

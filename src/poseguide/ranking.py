"""Ranking model: embedding / learning-to-rank beyond tag Jaccard."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

class EmbeddingRanker:
    """Learning-to-rank model for pose recommendation."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.pose_embeddings: Dict[str, np.ndarray] = {}
        self.tag_weights: Dict[str, float] = {}
    
    def add_pose(self, name: str, embedding: np.ndarray, tags: List[str]):
        """Add a pose with its embedding and tags."""
        self.pose_embeddings[name] = np.array(embedding)
        for tag in tags:
            self.tag_weights[tag] = self.tag_weights.get(tag, 0.0) + 1.0
    
    def encode_tags(self, tags: List[str]) -> np.ndarray:
        """Encode tags into an embedding vector."""
        vec = np.zeros(self.dim)
        if not tags:
            return vec
        total_weight = sum(self.tag_weights.get(t, 1.0) for t in tags)
        if total_weight == 0:
            return vec
        # Simple TF-IDF-like encoding
        for i, tag in enumerate(tags):
            weight = self.tag_weights.get(tag, 1.0) / total_weight
            np.random.seed(hash(tag) % 2**32)
            tag_vec = np.random.randn(self.dim) * weight * 0.1
            vec += tag_vec
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def rank(self, scene_tags: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank poses by cosine similarity to scene embedding."""
        scene_emb = self.encode_tags(scene_tags)
        scores = []
        for name, emb in self.pose_embeddings.items():
            sim = np.dot(scene_emb, emb) / (np.linalg.norm(emb) + 1e-8)
            scores.append((name, float(sim)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def jaccard_similarity(self, tags1: List[str], tags2: List[str]) -> float:
        """Compute Jaccard similarity between two tag sets (fallback)."""
        s1, s2 = set(tags1), set(tags2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    def combined_rank(self, scene_tags: List[str], pose_tags_map: Dict[str, List[str]],
                     top_k: int = 5, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """Rank combining embeddings and Jaccard similarity."""
        scores = {}
        for name, tags in pose_tags_map.items():
            emb_score = 0.0
            if name in self.pose_embeddings:
                emb_score = float(np.dot(self.encode_tags(scene_tags), 
                                        self.pose_embeddings[name]))
            jac_score = self.jaccard_similarity(scene_tags, tags)
            scores[name] = alpha * max(emb_score, 0) + (1 - alpha) * jac_score
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

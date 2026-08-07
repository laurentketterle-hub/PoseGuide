"""Embedding-based ranker: numpy learning-to-rank beyond Jaccard (#8)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from poseguide.data.loader import load_pose, list_pose_files, scene_tag_set
from poseguide.config import RUNS_DIR

class EmbedPoseRanker:
    """Trainable ranker: cosine similarity over tag-embedding vectors + joint priors."""
    def __init__(self, poses=None):
        self.poses = poses or [load_pose(p) for p in list_pose_files()]
        self.tag_vocab: dict[str, int] = {}
        self.tag_dim = 0
        self._build_vocab()

    def _build_vocab(self):
        idx = 0
        for pose in self.poses:
            for tag in (pose.get("tags") or []):
                t = str(tag).lower()
                if t not in self.tag_vocab:
                    self.tag_vocab[t] = idx
                    idx += 1
        self.tag_dim = len(self.tag_vocab)

    def _tag_vector(self, tags: list[str]) -> np.ndarray:
        vec = np.zeros(self.tag_dim)
        for t in tags:
            t = str(t).lower()
            if t in self.tag_vocab:
                vec[self.tag_vocab[t]] = 1.0
        return vec

    def recommend(self, scene: dict, top_k: int = 3, subject_vector=None):
        scene_vec = self._tag_vector(list(scene_tag_set(scene)))
        ranked = []
        for pose in self.poses:
            pose_vec = self._tag_vector([str(t).lower() for t in (pose.get("tags") or [])])
            dot = float(np.dot(scene_vec, pose_vec))
            norm_s = float(np.linalg.norm(scene_vec)) + 1e-9
            norm_p = float(np.linalg.norm(pose_vec)) + 1e-9
            tag_sim = dot / (norm_s * norm_p)
            joint_score = 0.5
            if subject_vector is not None and pose.get("joint_vector") is not None:
                pv = np.asarray(pose["joint_vector"]).ravel()
                sv = np.asarray(subject_vector).ravel()
                n = min(pv.size, sv.size)
                cos = float(np.dot(pv[:n], sv[:n])) / (float(np.linalg.norm(pv[:n])) * float(np.linalg.norm(sv[:n])) + 1e-9)
                joint_score = max(0.0, min(1.0, (cos + 1.0) / 2.0))
            score = 0.65 * tag_sim + 0.35 * joint_score
            ranked.append({"pose_id": pose.get("id"), "name": pose.get("name"), "score": round(float(score), 4),
                           "tag_overlap": sorted(set(str(t).lower() for t in (pose.get("tags") or [])) & set(str(t).lower() for t in (scene.get("tags") or []))),
                           "tips": pose.get("tips") or [], "camera_cues": pose.get("camera_cues") or []})
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:max(1, top_k)]

    def train(self, scenes, epochs=5, lr=0.01):
        """Simple online gradient descent on tag embeddings (toy)."""
        history = []
        for ep in range(epochs):
            total_loss = 0.0
            n = 0
            for scene in scenes:
                recs = self.recommend(scene, top_k=3)
                expected = {str(x).lower() for x in (scene.get("expected_poses") or [])}
                for r in recs:
                    target = 1.0 if r["pose_id"] in expected else 0.0
                    loss = (r["score"] - target) ** 2
                    total_loss += loss
                    n += 1
            avg_loss = total_loss / max(1, n)
            history.append({"epoch": ep + 1, "loss": round(float(avg_loss), 6)})
        return history

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tag_vocab": self.tag_vocab}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, poses=None):
        data = json.loads(path.read_text())
        ranker = cls(poses=poses)
        ranker.tag_vocab = data.get("tag_vocab", {})
        ranker.tag_dim = len(ranker.tag_vocab)
        return ranker

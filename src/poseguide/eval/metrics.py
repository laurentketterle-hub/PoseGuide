"""Evaluation metrics: hit@k, precision@k, recall@k, MRR (#21)."""
from __future__ import annotations

from poseguide.data.loader import list_scene_files, load_scene
from poseguide.models.toy import ToyPoseRanker


def evaluate_scenes(top_k: int = 3) -> dict:
    ranker = ToyPoseRanker()
    scenes = [load_scene(p) for p in list_scene_files()]
    hits = 0
    labeled = 0
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    rows = []
    for scene in scenes:
        expected = {str(x).lower() for x in (scene.get("expected_poses") or [])}
        recs = ranker.recommend(scene, top_k=top_k)
        top_ids = [str(r["pose_id"]).lower() for r in recs]
        top_set = set(top_ids)
        inter = top_set & expected
        hit = bool(expected and inter)
        if expected:
            labeled += 1
            if hit:
                hits += 1
            precision_sum += len(inter) / max(1, len(top_set))
            recall_sum += len(inter) / max(1, len(expected))
            # MRR: reciprocal rank of first expected hit
            rr = 0.0
            for i, pid in enumerate(top_ids):
                if pid in expected:
                    rr = 1.0 / (i + 1)
                    break
            mrr_sum += rr
        rows.append({
            "scene": scene.get("id"),
            "expected": sorted(expected),
            "top": top_ids,
            "hit": hit,
            "overlap": sorted(inter),
        })
    n_l = max(1, labeled)
    return {
        "top_k": top_k,
        "n_scenes": len(scenes),
        "n_labeled": labeled,
        "hit_at_k": round(hits / n_l, 4),
        "precision_at_k": round(precision_sum / n_l, 4),
        "recall_at_k": round(recall_sum / n_l, 4),
        "mrr": round(mrr_sum / n_l, 4),
        "rows": rows,
    }

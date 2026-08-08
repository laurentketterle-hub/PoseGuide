"""
Composition rules engine for PoseGuide.
Implements rule-of-thirds, horizon detection, and headroom analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class CompositionScore:
    rule_of_thirds: float  # 0-1, how well key points align with thirds grid
    horizon_level: float   # 0-1, how level the horizon is
    headroom: float        # 0-1, adequate headroom score
    overall: float         # weighted composite


class CompositionRules:
    """Analyze photo composition using geometric rules."""
    
    GRID_WEIGHT = 0.35
    HORIZON_WEIGHT = 0.35
    HEADROOM_WEIGHT = 0.30
    
    def __init__(self):
        self.thirds_x = [1/3, 2/3]
        self.thirds_y = [1/3, 2/3]
        self.ideal_headroom_ratio = 0.15  # ideal 15% of frame
    
    def evaluate(self, width: int, height: int,
                 key_points: List[Tuple[float, float]],
                 horizon_y: Optional[float] = None,
                 face_center: Optional[Tuple[float, float]] = None) -> CompositionScore:
        """Score a photo's composition.
        
        Args:
            width, height: Frame dimensions in pixels
            key_points: List of (x_ratio, y_ratio) key subject points (0-1 normalized)
            horizon_y: Normalized y-coordinate of horizon line (0=top, 1=bottom)
            face_center: Normalized (x, y) of subject's face center
        
        Returns:
            CompositionScore with per-rule and overall scores
        """
        thirds_score = self._score_rule_of_thirds(key_points)
        horizon_score = self._score_horizon(horizon_y) if horizon_y is not None else 1.0
        headroom_score = self._score_headroom(height, face_center)
        
        overall = (
            self.GRID_WEIGHT * thirds_score +
            self.HORIZON_WEIGHT * horizon_score +
            self.HEADROOM_WEIGHT * headroom_score
        )
        
        return CompositionScore(
            rule_of_thirds=round(thirds_score, 3),
            horizon_level=round(horizon_score, 3),
            headroom=round(headroom_score, 3),
            overall=round(overall, 3)
        )
    
    def _score_rule_of_thirds(self, key_points: List[Tuple[float, float]]) -> float:
        """Score how well key points align with rule-of-thirds grid intersections."""
        if not key_points:
            return 0.5  # neutral
        
        scores = []
        for px, py in key_points:
            # Distance to nearest thirds intersection
            min_dist = float('inf')
            for tx in self.thirds_x:
                for ty in self.thirds_y:
                    dist = math.sqrt((px - tx) ** 2 + (py - ty) ** 2)
                    min_dist = min(min_dist, dist)
            # Convert distance to score (max distance to intersection is ~0.47)
            scores.append(max(0, 1 - min_dist / 0.47))
        
        return sum(scores) / len(scores)
    
    def _score_horizon(self, horizon_y: Optional[float]) -> float:
        """Score horizon levelness and placement."""
        if horizon_y is None:
            return 1.0
        
        # Penalize tilted horizon (far from 0 or 1)
        # Ideal horizons are near thirds: 0.33 or 0.66
        dist_to_third = min(abs(horizon_y - 1/3), abs(horizon_y - 2/3))
        return max(0, 1 - dist_to_third / 0.33)
    
    def _score_headroom(self, height: int, face_center: Optional[Tuple[float, float]]) -> float:
        """Score adequate headroom for portrait compositions."""
        if face_center is None:
            return 1.0  # no face detected, skip
        
        _, fy = face_center
        # Headroom = space above face center to top of frame
        headroom = fy  # normalized: 0 = top, fy = distance from top
        
        # Ideal headroom is ~15% of frame height above face
        ideal = self.ideal_headroom_ratio
        deviation = abs(headroom - ideal)
        
        return max(0, 1 - deviation / ideal)


def analyze_composition(width: int, height: int,
                        subjects: List[dict]) -> CompositionScore:
    """Convenience function to analyze composition from subject data.
    
    Args:
        width, height: Frame dimensions
        subjects: List of dicts with 'key_points', optional 'horizon_y', 'face_center'
    """
    engine = CompositionRules()
    
    all_points = []
    horizon_y = None
    face_center = None
    
    for subj in subjects:
        if "key_points" in subj:
            all_points.extend(subj["key_points"])
        if "horizon_y" in subj:
            horizon_y = subj["horizon_y"]
        if "face_center" in subj:
            face_center = subj["face_center"]
    
    return engine.evaluate(width, height, all_points, horizon_y, face_center)

"""
FastAPI endpoints for PoseGuide — pose recommendation and scoring.
Implements POST /guide/recommend and POST /guide/score.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import math

app = FastAPI(title="PoseGuide API", version="1.0.0")


class PoseRequest(BaseModel):
    """Request body for pose recommendation."""
    tags: List[str] = Field(..., description="Desired pose tags (e.g., ['yoga', 'standing'])")
    count: int = Field(default=5, ge=1, le=20, description="Number of recommendations")
    exclude_ids: List[str] = Field(default=[], description="Pose IDs to exclude")
    scene_tags: Optional[List[str]] = Field(default=None, description="Scene context tags")


class PoseRecommendation(BaseModel):
    """A single pose recommendation."""
    pose_id: str
    name: str
    score: float  # 0-1 relevance score
    tags: List[str]
    difficulty: str


class RecommendResponse(BaseModel):
    recommendations: List[PoseRecommendation]
    total_available: int


class ScoreRequest(BaseModel):
    """Request body for pose scoring."""
    pose_id: str = Field(..., description="Pose to score against")
    joints: List[dict] = Field(..., description="User's joint positions")
    image_width: int = Field(default=1920)
    image_height: int = Field(default=1080)


class JointScore(BaseModel):
    """Per-joint scoring detail."""
    joint_id: int
    joint_name: str
    deviation: float  # pixel distance from ideal
    visibility: float  # 0-1 detection confidence
    score: float  # 0-1 per-joint score


class ScoreResponse(BaseModel):
    overall_score: float  # 0-1
    joint_scores: List[JointScore]
    feedback: str


# Mock pose database (would be replaced with real data)
MOCK_POSES = [
    {"id": "yoga_warrior_i", "name": "Warrior I", "tags": ["yoga", "standing", "strength"], "difficulty": "intermediate"},
    {"id": "yoga_tree", "name": "Tree Pose", "tags": ["yoga", "standing", "balance"], "difficulty": "beginner"},
    {"id": "yoga_downward_dog", "name": "Downward Dog", "tags": ["yoga", "inversion", "stretch"], "difficulty": "beginner"},
    {"id": "yoga_crow", "name": "Crow Pose", "tags": ["yoga", "arm_balance", "strength"], "difficulty": "advanced"},
    {"id": "yoga_child", "name": "Child's Pose", "tags": ["yoga", "resting", "stretch"], "difficulty": "beginner"},
    {"id": "fitness_squat", "name": "Bodyweight Squat", "tags": ["fitness", "standing", "strength"], "difficulty": "beginner"},
    {"id": "fitness_pushup", "name": "Push-Up", "tags": ["fitness", "floor", "strength"], "difficulty": "beginner"},
    {"id": "fitness_plank", "name": "Plank", "tags": ["fitness", "floor", "core"], "difficulty": "intermediate"},
    {"id": "fitness_lunge", "name": "Forward Lunge", "tags": ["fitness", "standing", "strength"], "difficulty": "beginner"},
    {"id": "office_seated_twist", "name": "Seated Twist", "tags": ["office", "seated", "stretch"], "difficulty": "beginner"},
    {"id": "office_neck_roll", "name": "Neck Roll", "tags": ["office", "seated", "stretch"], "difficulty": "beginner"},
    {"id": "office_wrist_stretch", "name": "Wrist Stretch", "tags": ["office", "seated", "stretch"], "difficulty": "beginner"},
    {"id": "dance_arabesque", "name": "Arabesque", "tags": ["dance", "standing", "balance"], "difficulty": "advanced"},
    {"id": "dance_plie", "name": "Plie", "tags": ["dance", "standing", "strength"], "difficulty": "intermediate"},
    {"id": "studio_hand_on_hip", "name": "Hand on Hip", "tags": ["studio", "standing", "portrait"], "difficulty": "beginner"},
]


def _tag_similarity(query_tags: List[str], pose_tags: List[str]) -> float:
    """Jaccard similarity between tag sets."""
    if not query_tags or not pose_tags:
        return 0.0
    q_set = set(t.lower() for t in query_tags)
    p_set = set(t.lower() for t in pose_tags)
    intersection = len(q_set & p_set)
    union = len(q_set | p_set)
    return intersection / union if union > 0 else 0.0


def _score_pose_alignment(user_joints: List[dict], pose_id: str) -> tuple:
    """Score how well user joints align with a target pose.
    Returns (overall_score, joint_scores_list).
    """
    # Joint name mapping
    JOINT_NAMES = {
        0: "nose", 1: "neck", 2: "right_shoulder", 3: "right_elbow",
        4: "right_wrist", 5: "left_shoulder", 6: "left_elbow", 7: "left_wrist",
        8: "right_hip", 9: "right_knee", 10: "right_ankle",
        11: "left_hip", 12: "left_knee", 13: "left_ankle",
    }
    
    joint_scores = []
    total_score = 0.0
    count = 0
    
    for j in user_joints:
        jid = j.get("id", count)
        name = JOINT_NAMES.get(jid, f"joint_{jid}")
        visibility = j.get("visibility", 1.0)
        
        # Simplified scoring: reward high visibility and centered positions
        x, y = j.get("x", 0.5), j.get("y", 0.5)
        
        # Penalize extreme positions (joints should be in frame)
        in_frame = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        frame_penalty = 0.0 if in_frame else 0.5
        
        score = visibility * (1.0 - frame_penalty)
        joint_scores.append(JointScore(
            joint_id=jid,
            joint_name=name,
            deviation=round((1.0 - score) * 100, 1),
            visibility=round(visibility, 2),
            score=round(score, 3),
        ))
        total_score += score
        count += 1
    
    overall = total_score / count if count > 0 else 0.0
    return round(overall, 3), joint_scores


@app.get("/")
async def root():
    return {"service": "PoseGuide API", "version": "1.0.0", "endpoints": ["/guide/recommend", "/guide/score"]}


@app.post("/guide/recommend", response_model=RecommendResponse)
async def recommend_poses(request: PoseRequest):
    """Recommend poses matching given tags and scene context."""
    scored = []
    for pose in MOCK_POSES:
        if pose["id"] in request.exclude_ids:
            continue
        
        sim = _tag_similarity(request.tags, pose["tags"])
        if sim > 0:
            # Boost if scene tags match
            scene_boost = 0.0
            if request.scene_tags:
                scene_boost = _tag_similarity(request.scene_tags, pose["tags"]) * 0.2
            
            scored.append(PoseRecommendation(
                pose_id=pose["id"],
                name=pose["name"],
                score=round(min(sim + scene_boost, 1.0), 3),
                tags=pose["tags"],
                difficulty=pose["difficulty"],
            ))
    
    scored.sort(key=lambda p: p.score, reverse=True)
    selected = scored[:request.count]
    
    return RecommendResponse(
        recommendations=selected,
        total_available=len(scored),
    )


@app.post("/guide/score", response_model=ScoreResponse)
async def score_pose(request: ScoreRequest):
    """Score a user's pose against a target pose."""
    if not request.joints:
        raise HTTPException(status_code=400, detail="No joints provided")
    
    overall, joint_scores = _score_pose_alignment(request.joints, request.pose_id)
    
    # Generate feedback
    if overall >= 0.8:
        feedback = "Excellent! Your pose closely matches the target."
    elif overall >= 0.6:
        feedback = "Good effort. Focus on keeping joints visible and centered."
    elif overall >= 0.4:
        feedback = "Getting there. Try adjusting your position relative to the camera."
    else:
        feedback = "Needs work. Ensure all body parts are visible in frame."
    
    return ScoreResponse(
        overall_score=overall,
        joint_scores=joint_scores,
        feedback=feedback,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

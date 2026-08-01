"""Live webcam coach loop for PoseGuide."""
import time
import json
from typing import Dict, Optional, Callable
from dataclasses import dataclass

@dataclass
class PoseFeedback:
    """Feedback for a single pose frame."""
    pose_name: str
    score: float
    joint_diffs: Dict[str, float]
    suggestions: list

class WebcamCoach:
    """Live webcam coach comparing subject pose vs target pose."""
    
    def __init__(self, target_pose: Dict, threshold: float = 0.6):
        self.target_pose = target_pose
        self.threshold = threshold
        self.running = False
        self.feedback_history: list = []
    
    def start(self, frame_callback: Callable, interval: float = 0.1):
        """Start the coaching loop."""
        self.running = True
        while self.running:
            frame = frame_callback()
            if frame:
                feedback = self.analyze_frame(frame)
                self.feedback_history.append(feedback)
            time.sleep(interval)
    
    def stop(self):
        """Stop the coaching loop."""
        self.running = False
    
    def analyze_frame(self, frame_landmarks: Dict) -> PoseFeedback:
        """Analyze a frame against the target pose."""
        diffs = {}
        for joint, target_pos in self.target_pose.get('joints', {}).items():
            current = frame_landmarks.get('joints', {}).get(joint, {})
            if current:
                dx = current.get('x', 0) - target_pos.get('x', 0)
                dy = current.get('y', 0) - target_pos.get('y', 0)
                diffs[joint] = (dx**2 + dy**2) ** 0.5
        
        if diffs:
            avg_diff = sum(diffs.values()) / len(diffs)
            score = max(0, 1 - avg_diff)
        else:
            score = 0
        
        suggestions = []
        if score < self.threshold:
            worst_joint = max(diffs, key=diffs.get)
            suggestions.append(f"Adjust {worst_joint} position")
        
        return PoseFeedback(
            pose_name=self.target_pose.get('name', 'unknown'),
            score=round(score, 3),
            joint_diffs=diffs,
            suggestions=suggestions
        )

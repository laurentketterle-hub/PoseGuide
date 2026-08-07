# POSES.md - PoseGuide Pose Catalog

81 poses across 15+ families: standing, seated, walk, lean, crouch, yoga, fitness, window-light.

## Pose Families

### Contrapposto / Classic
contrapposto, classic_at_ease, weight_shift_standing, standing_three_quarter_turn

### Power / Confident
power_stance, arms_crossed_power, hands_on_hips_hero, studio_hand_on_hip_power, cross_arms_stand, hands_on_hips

### Walk / Motion
walk_toward_camera, walking_toward_camera, side_profile_walk, beach_walk_side_profile, couple_walk_holding_hands, umbrella_walk, outdoor_running_midstride

### Lean / Wall
wall_lean, lean_wall_casual, lean_on_rail, leaning_doorframe, lean_window_portrait, coffee_table_lean, lean_forward, bike_lean, night_city_lean_against_rail, lean_wall_casual_soft, lean_window_portrait_soft, lean_forward_desk, leaning_rail

### Seated
book_reading, chin_rest, coffee_cup, laptop_desk, seated_cafe_laptop, seated_crossed_legs, seated_profile, seated_profile_window, seated_window_light, sit_cross_legged, sit_on_stool, sitting_stairs_casual

### Crouch / Low Angle
crouch_ready, crouch_street_photo, crouch_tie_shoe

### Over-Shoulder / Lookback
over_shoulder, over_shoulder_look, over_shoulder_look_soft, over_shoulder_lookback, looking_away, back_to_camera, side_profile_look

### Kneeling / Proposal
kneel_propose, kneeling_propose_style, kneel_profile, one_knee_down

### Arms / Hands
arms_crossed, arms_raised, hands_behind_back, hands_clasped_front, hands_in_pockets, hand_wave, hat_tip, palm_up_explain, point_forward, selfie_peace, hand_on_chin, phone_scroll, look_down_phone

### Jump / Action
jump_midair, jump_midair_arms_open

### Yoga / Fitness / Stretch
yoga_tree, yoga_warrior_ii_side, stretch_reach, desk_standup_stretch

### Window / Natural Light
window_gaze, window_light_stand

## Extension Guide

1. Create data/poses/<id>.json
2. Run poseguide poses list
3. SVG: poseguide poses svg --pose <id>
4. Add expected_poses in scene files

## When-to-Use Matrix

| Background | Recommended | Avoid |
|---|---|---|
| Beach / golden hour | Walk, over-shoulder, hands-on-hips | Seated, desk, formal |
| Urban wall / street | Lean, crouch, walk, power-stance | Yoga, floor |
| Studio | Power, formal, seated, profile | Action, jump, duo |
| Forest / nature | Walk, over-shoulder, looking-away | Business, desk |
| Cafe / indoor warm | Seated, lean, coffee, chin-rest | Jump, running |
| Office / business | Power, formal, desk-stretch | Crouch, beach, yoga |
| Night / city lights | Lean, over-shoulder, moody | Beach, daylight |
| Mountain / adventure | Arms-raised, power, hero, walk | Seated, formal |

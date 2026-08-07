"""Tests for export_format, datasets, scene_tagger, embed ranker."""

import sys

sys.path.insert(0, "src")


def test_datasets_list():
    from poseguide.data.datasets import list_datasets

    all_ds = list_datasets()
    assert len(all_ds) >= 10, f"Expected >=10 datasets, got {len(all_ds)}"
    filtered = list_datasets(domain="Pose")
    assert len(filtered) > 0


def test_scene_tagger_preset():
    from poseguide.guide.scene_tagger import tag_scene

    tags = tag_scene(preset="beach")
    assert "beach" in tags
    assert "outdoor" in tags


def test_scene_tagger_desc():
    from poseguide.guide.scene_tagger import tag_scene

    tags = tag_scene(description="a beach scene at sunset")
    assert isinstance(tags, list) and len(tags) > 0


def test_export_mediapipe():
    from poseguide.data.export_format import pose_to_mediapipe

    pose = {
        "id": "test",
        "name": "Test",
        "joints": {"nose": [0.5, 0.1, 0.0], "l_shoulder": [0.4, 0.3, 0.0]},
    }
    result = pose_to_mediapipe(pose)
    assert result["pose_id"] == "test"
    assert len(result["landmarks"]) == 33


def test_export_coco():
    from poseguide.data.export_format import pose_to_coco

    pose = {
        "id": "test",
        "name": "Test",
        "joints": {"nose": [0.5, 0.1, 0.0], "l_shoulder": [0.4, 0.3, 0.0]},
    }
    result = pose_to_coco(pose)
    assert result["pose_id"] == "test"
    assert result["num_keypoints"] == 17
    assert len(result["keypoints"]) == 51


def test_embed_ranker_builds():
    from poseguide.models.embed import EmbedPoseRanker

    ranker = EmbedPoseRanker()
    assert ranker.tag_dim > 0
    scene = {"id": "test", "tags": ["beach", "outdoor"], "mood": []}
    recs = ranker.recommend(scene, top_k=3)
    assert len(recs) >= 1


def test_metrics_mrr():
    from poseguide.eval.metrics import evaluate_scenes

    report = evaluate_scenes(top_k=3)
    assert "mrr" in report
    assert isinstance(report["mrr"], (int, float))


def test_new_poses_load():
    from poseguide.data.loader import list_pose_files

    files = {p.stem for p in list_pose_files()}
    for pid in [
        "desk_standup_stretch",
        "couple_walk_holding_hands",
        "yoga_warrior_ii_side",
        "studio_hand_on_hip_power",
        "beach_walk_side_profile",
    ]:
        assert pid in files, f"Missing pose: {pid}"


def test_new_scenes_load():
    from poseguide.data.loader import list_scene_files

    files = {p.stem for p in list_scene_files()}
    for sid in ["mountain_summit", "city_rooftop_night", "autumn_park_path"]:
        assert sid in files, f"Missing scene: {sid}"


if __name__ == "__main__":
    test_datasets_list()
    print("OK datasets")
    test_scene_tagger_preset()
    print("OK tagger preset")
    test_scene_tagger_desc()
    print("OK tagger desc")
    test_export_mediapipe()
    print("OK export mediapipe")
    test_export_coco()
    print("OK export coco")
    test_embed_ranker_builds()
    print("OK embed ranker")
    test_metrics_mrr()
    print("OK metrics MRR")
    test_new_poses_load()
    print("OK new poses")
    test_new_scenes_load()
    print("OK new scenes")
    print("ALL TESTS PASSED")

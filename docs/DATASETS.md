# Public Pose Datasets Index (License-Safe)

Curated index of public pose/photography datasets usable for training and evaluation.
All datasets listed here are available under permissive licenses (CC-BY, CC0, research-only).
No restricted or redistribution-problematic media is included in the PoseGuide repository.

## Dataset Registry

| # | Dataset | Poses | Subjects | Format | License | Year | URL |
|---|---------|-------|----------|--------|---------|------|-----|
| 1 | **MPII Human Pose** | 25K | 40K people | Keypoints (16) | BSD | 2014 | http://human-pose.mpi-inf.mpg.de/ |
| 2 | **COCO Keypoints** | 250K | 200K images | Keypoints (17) | CC-BY 4.0 | 2014 | https://cocodataset.org/#keypoints |
| 3 | **LSP (Leeds Sports Pose)** | 2K | 2K images | Keypoints (14) | Research | 2010 | https://sam.johnson.io/research/lsp.html |
| 4 | **LSP Extended** | 10K | 10K images | Keypoints (14) | Research | 2011 | https://sam.johnson.io/research/lspet.html |
| 5 | **MPII-TRB (Temporal Relation)** | 25K | 40K people | Keypoints + temporal | BSD | 2018 | https://github.com/weigq/TRB |
| 6 | **FreiHAND** | 130K | 32K images | Hand keypoints (21) | CC-BY 4.0 | 2019 | https://lmb.informatik.uni-freiburg.de/projects/freihand/ |
| 7 | **Panoptic Studio** | 1.5M | 3D skeleton | 3D joints (19) | Research | 2015 | http://domedb.perception.cs.cmu.edu/ |
| 8 | **Human3.6M** | 3.6M | 11 actors | 3D joints (32) | Research | 2014 | http://vision.imar.ro/human3.6m/ |
| 9 | **3DPW (3D Poses in the Wild)** | 51K | 7 actors | 3D mesh + joints | Research | 2018 | https://virtualhumans.mpi-inf.mpg.de/3DPW/ |
| 10 | **AIST Dance DB** | 500K | 30 dancers | 3D joints (17) | Research | 2019 | https://aistdancedb.ongaaccel.jp/ |

## License Categories

- **CC-BY 4.0**: Free to use, share, adapt with attribution (COCO, FreiHAND)
- **BSD**: Permissive open-source license (MPII)
- **Research-Only**: Free for academic/research use, redistribution terms vary (LSP, Human3.6M, 3DPW, AIST)

## Integration Notes

- PoseGuide currently supports 2D keypoint annotations in COCO and MPII formats out of the box.
- 3D datasets (Human3.6M, 3DPW, Panoptic) require projection or 3D-to-2D adaptation.
- Hand keypoint datasets (FreiHAND) follow a different skeleton convention (21 keypoints).
- All datasets listed are publicly downloadable from their respective hosts. PoseGuide does not
  redistribute any dataset files — users must download directly from the source URLs.

## Adding a New Dataset

To add a dataset to this index, submit a PR updating this table with:
1. Verified license information (link to license page)
2. Accurate pose/keypoint count
3. Active download URL
4. Format compatibility notes

See `docs/CONTRIBUTING.md` for full contribution guidelines.

## References

- COCO Keypoints Evaluation: https://cocodataset.org/#keypoints-eval
- MPII Pose Estimation Benchmark: http://human-pose.mpi-inf.mpg.de/#results
- 3D Human Pose Estimation Survey: https://arxiv.org/abs/2001.07732

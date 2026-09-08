# Track-Fit

Skeleton 기반 운동 동작 인식·유사도 평가 시스템.

- ProtoGCN(CVPR 2025) 재구현
- MediaPipe로 관절 좌표 추출 후 COCO20 형식으로 변환 
- 추출한 관절의 품질을 신뢰도 기반 점수로 정량화
- 저품질 keypoint에서의 모델 일반화를 위해 Occlusion Augmentation 구현
- hard-negative margin loss 구현 
- 기존 Joint feature를 사용해 Angle feature 구현


## Results

노이즈가 추가된 저품질 데이터셋 기준:

| 단계 | Top-1 Acc |
|---|---:|
| Baseline (FineGym-pretrained fine-tuning) | 67.2% |
| + Temporal occlusion | 68.9% |
| + Torso-protected joint masking | 72.1% |
| Bone + Temporal occlusion | 75.4% |
| **3-stream ensemble** | **78.7%** |


## Repository Structure

```
track-fit/
├── configs/                                            # Hydra config
│   ├── config.yaml
│   ├── model/  data/       
│   └── experiment/  
├── protogcn/                                           # ProtoGCN 재구현
│   ├── train.py  test.py
│   ├── models/  losses/  utils/
│   ├── datasets/pipelines/        
│   └── tools/
│       ├── run_ensemble.py                             # ensemble
│       ├── cluster_eval.py                             # error analysis
│       ├── extract_features.py 
│       └── linear_probe.py    
├── scripts/
│   ├── extract_keypoint_mediapipe.py                   # extract keypoint                 
│   └── split_dataset_difficulty_stratified.py          # split dataset
├── demo/                   
└── external/ProtoGCN/   
```

## Setup

```bash
git clone https://github.com/jihyun-0611/track-fit.git
cd track-fit
git submodule update --init --recursive   # external/ProtoGCN (원본 코드)

conda create -n mediapipe python=3.8
conda activate mediapipe
pip install torch mediapipe opencv-python hydra-core omegaconf python-dotenv wandb
```

사전학습 가중치: [FineGym joint checkpoint](https://github.com/firework8/ProtoGCN/blob/main/data/README.md)를 받아 `.env`의 `WORK_DIR` 하위 `finegym_j/`에 배치.

`.env` :
```bash
DATA_DIR=/path/to/track-fit/data
WORK_DIR=/path/to/track-fit/work_dirs
```

## Usage

**1. 데이터 준비** — [Kaggle Gym Workout/Exercises Video](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) (22 exercises, 1,608 clips)

```bash
python scripts/extract_keypoint_mediapipe.py              # MediaPipe → COCO20 keypoint
python scripts/split_dataset_difficulty_stratified.py     # difficulty 기반 stratified split
```

**2. 학습** — preset 목록은 `configs/experiment/` 참고

```bash
python protogcn/train.py experiment=<preset>
```

**3. 평가·앙상블**

```bash
python protogcn/test.py <config> <checkpoint>
python protogcn/tools/run_ensemble.py --score-dir scores/ --dataset-pkl data/dataset_diff.pkl
python protogcn/tools/cluster_eval.py --pred scores/ensemble/final.pkl --split final_test
```

## References

- Liu et al., *Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition*, CVPR 2025 — [ProtoGCN](https://github.com/firework8/ProtoGCN)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)
- [Skelbumentations](https://github.com/MickaelCormier/Skelbumentations)

## License

This project is for research purposes only.

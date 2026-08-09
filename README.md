# Track-Fit

Skeleton 기반 운동 동작 인식·유사도 평가 시스템.
MediaPipe로 추출한 관절 좌표를 ProtoGCN(CVPR 2025) 재구현 모델로 학습하고, 저품질 실환경 keypoint에서의 일반화 문제를 데이터 설계–증강–멀티스트림 앙상블–원인 진단으로 개선했습니다.

📄 [연구 보고서](#) · 📊 [실험 로그 (WandB)](#)

## Results

노이즈가 추가된 저품질 데이터셋 검증(final_test, 61 clips) 기준 22클래스 인식 정확도:

| 단계 | Final Top-1 |
|---|---:|
| Baseline (FineGym-pretrained fine-tuning) | 67.2% |
| + Temporal occlusion | 68.9% |
| + Torso-protected joint masking | 72.1% |
| + Bone stream | 75.4% |
| **+ Calibrated 4-stream ensemble** | **78.7%** |

남은 오답은 원인을 keypoint 측정 수준까지 추적해, 촬영 각도·추적 실패 등 **모델 외적 요인**과 분류 경계 문제를 구분했습니다.

## Highlights

- **품질 기반 데이터 분할** — 관절 신뢰도·jitter 등 5개 지표로 keypoint 품질 점수(difficulty score)를 정의하고, 품질 계층이 균형되도록 stratified split. 원본 영상 단위 그룹 배정으로 데이터 누수 차단
- **3단 검증 체계** — val(설정 선택) / internal_test(채택 판정) / final_test(최종 1회 확인)를 분리
- **표현별 최적화된 멀티스트림** — joint / bone / joint-motion / angle 표현을 각각 재검증된 설정으로 학습 
- **보정 앙상블** — temperature scaling(val) 후 fusion weight를 internal에서만 탐색하는 자동화 도구
- **진단 도구** — 혼동 클래스군 분석, layer별 linear probing으로 구조 추가(FR-Head) 필요성을 사전 판정

## Repository Structure

```
track-fit/
├── configs/                # Hydra 설정
│   ├── config.yaml
│   ├── model/  data/       # 모델·데이터 파이프라인 설정
│   └── experiment/         # 실험 preset
├── protogcn/               # ProtoGCN 재구현
│   ├── train.py  test.py
│   ├── models/  losses/  utils/
│   ├── datasets/pipelines/ # 전처리·증강            
│   └── tools/
│       ├── run_ensemble.py      # 보정·weight 탐색·통계 검정 자동 리포트
│       ├── cluster_eval.py      # 혼동 클래스군 분석
│       ├── extract_features.py  # backbone 중간 feature 추출
│       └── linear_probe.py      # layer별 선형 분리 가능성 진단
├── scripts/
│   ├── extract_keypoint_mediapipe.py         # 영상 → COCO20 keypoint
│   ├── create_dataset.py                     # annotation pkl 생성
│   └── split_dataset_difficulty_stratified.py # difficulty 기반 split
├── demo/                   # 실시간 데모 (진행 중)
└── external/ProtoGCN/      # 원본 구현 (재구현 참고용 submodule)
```

## Setup

```bash
git clone https://github.com/jihyun-0611/track-fit.git
cd track-fit
git submodule update --init --recursive   # external/ProtoGCN (참고용)

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

**4. 진단 (선택)**

```bash
python protogcn/tools/extract_features.py --config <cfg> --checkpoint <pth> --splits train internal_test
python protogcn/tools/linear_probe.py --features features/<model>/
```

## Limitations

- final_test가 61개로 작아 개별 개선의 통계적 유의성은 미확보 (개선 방향은 일관, 회귀 없음)
- 유사도 점수의 타당성 검증(동작 품질과 점수의 상관)과 실시간 추론은 진행 예정

## References

- Liu et al., *Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition*, CVPR 2025 — [ProtoGCN](https://github.com/firework8/ProtoGCN)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)

## License

This project is for research purposes only.
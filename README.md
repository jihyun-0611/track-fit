# Track-Fit: 운동 동작 품질 평가 시스템

ProtoGCN 기반 실시간 운동 동작 인식 및 품질 평가 시스템

## 📋 프로젝트 개요

Track-Fit은 운동 영상에서 동작의 품질을 평가하기 위한 딥러닝 기반 시스템입니다. [ProtoGCN](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Revealing_Key_Details_to_See_Differences_A_Novel_Prototypical_Perspective_CVPR_2025_paper.html)(Prototype Graph Convolutional Network)을 활용하여 운동 동작의 프로토타입을 학습하고, 실시간으로 동작을 인식하며 품질을 평가합니다.

- **실시간 동작 인식**: 웹캠을 통한 실시간 운동 동작 인식
- **5가지 운동 지원**: Barbell Biceps Curl, Bench Press, Lat Pulldown, Push-up, Tricep Pushdown
- **프로토타입 기반 학습**: ProtoGCN을 활용한 운동별 프로토타입 학습
- **품질 평가**: 학습된 프로토타입과의 유사도 기반 동작 품질 평가 

## 🏗️ 프로젝트 구조

```
track-fit/
├── configs/
│   ├── exercise/              # MMCv config (ProtoGCN용)
│   │   ├── j.py              # Full fine-tuning config
│   │   └── j_freeze.py       # Freeze backbone config
│   └── hydra/                # Hydra experiment configs
│       ├── config.yaml       # Main config
│       ├── experiment/       # Experiment presets
│       │   ├── phase1_freeze.yaml
│       │   ├── phase2_finetune.yaml
│       │   └── debug.yaml
│       ├── model/
│       │   └── protogcn.yaml
│       └── training/
│           └── default.yaml
├── demo/                      # Real-time demo app
│   ├── app/                  # Web application
│   ├── extractor/            # MediaPipe keypoint extraction server
│   └── inferencer/           # ProtoGCN inference server
├── external/
│   └── ProtoGCN/             # ProtoGCN submodule
├── scripts/
│   ├── create_dataset.py     # Dataset creation
│   ├── extract_keypoint_mediapipe.py  # Keypoint extraction
│   └── visualize_keypoints_mediapipe.py  # Visualization
├── freeze_backbone_hook.py    # Custom training hook
└── train_hydra.py            # Hydra training script
```

## 🚀 설치 및 환경 설정

### 1. 저장소 클론 및 서브모듈 초기화

```bash
git clone https://github.com/jihyun-0611/track-fit.git
cd track-fit

# ProtoGCN 서브모듈 초기화
git submodule update --init --recursive

# ProtoGCN 환경 설정
cd external/ProtoGCN
conda env create -f protogcn.yaml
conda activate protogcn
pip install -e .

# Hydra 설치 (실험 관리용)
cd ../..
pip install hydra-core omegaconf

pip install python-dotenv
```

```bash
# MediaPipe 환경 (키포인트 추출 및 웹 서버용)
conda create -n mediapipe python=3.8
conda activate mediapipe
pip install -r demo/extractor/requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:
```bash
BASE_DIR=/path/to/track-fit
DATA_DIR=/path/to/track-fit/data
CHECKPOINT_DIR=/path/to/track-fit/checkpoints
WORK_DIR=/path/to/track-fit/work_dirs
PRETRAINED=/path/to/track-fit/checkpoints/finegym_j/best.pth
DATASET_PATH=/path/to/track-fit/data/exercise_dataset.pkl
```

## 📊 데이터 준비

https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video

### 1. 비디오 데이터 구조

```
data/
├── sample_videos/
│   ├── barbell biceps curl/
│   ├── bench press/
│   ├── lat pulldown/
│   ├── push-up/
│   └── tricep Pushdown/
└── filter_meta.csv  # 비디오 메타데이터
```

### 2. 키포인트 추출

```bash
conda activate mediapipe

# 기본 실행 (.env의 DATA_DIR)
python scripts/extract_keypoint_mediapipe.py

# 커스텀 data directory
python scripts/extract_keypoint_mediapipe.py --data-dir /path/to/data

# 신뢰도 임계값 조정
python scripts/extract_keypoint_mediapipe.py --min-detection-confidence 0.7 --min-tracking-confidence 0.7
```

### 3. 데이터셋 생성

```bash
# 기본 실행 (.env의 DATA_DIR 사용 또는 자동 탐색)
python scripts/create_dataset.py

# 커스텀 data directory
python scripts/create_dataset.py --data-dir /path/to/data

# Train/validation split 비율 변경
python scripts/create_dataset.py --train-ratio 0.9 --random-seed 123
```

### 4. 키포인트 시각화

```bash
# 특정 비디오의 키포인트 시각화
python scripts/visualize_keypoints_mediapipe.py \
    --video-name "bench press_57" \
    --exercise-type "bench press"

# 저장만 하고 화면에 표시하지 않기
python scripts/visualize_keypoints_mediapipe.py \
    --video-name "bench press_57" \
    --exercise-type "bench press" \
    --no-show
```

## 🏋️ 모델 학습

### 사전학습 모델 준비

FineGYM 데이터셋으로 사전학습된 모델을 [여기서](https://github.com/firework8/ProtoGCN/blob/ddf7f274f9f5d9e45a2fcfeb299bfb3fd7c2303d/data/README.md) 다운로드:
```bash
mkdir -p checkpoints/finegym_j
# best_top1_acc_epoch_141.pth 파일을 checkpoints/finegym_j/에 배치
```

### 학습 실행

ProtoGCN 환경 활성화 필요
```bash
conda activate protogcn
```

#### 학습 설정 

**Phase 1** (`phase1_freeze.yaml`):
- 20 epochs
- Head만 학습 (backbone freeze)
- Learning rate: 0.01
- Optimizer: SGD with Nesterov momentum
- LR Schedule: CosineAnnealing

**Phase 2** (`phase2_finetune.yaml`):
- 80 epochs
- 전체 파인튜닝
- Learning rate: 0.001
- Optimizer: SGD with Nesterov momentum
- LR Schedule: CosineAnnealing

#### 학습 실행

```bash
# Phase 1: Backbone freeze, Head만 학습 (20 epochs)
python train_hydra.py experiment=phase1_freeze

# Phase 2: 전체 파인튜닝 (80 epochs)
python train_hydra.py experiment=phase2_finetune

# 빠른 테스트 (2 epochs)
python train_hydra.py experiment=debug
```

#### 설정 커스터마이징

**하이퍼파라미터**:
```bash
# Learning rate 변경
python train_hydra.py experiment=phase1_freeze training.optimizer.lr=0.02

# Epoch 수 변경
python train_hydra.py experiment=phase2_finetune training.epochs=100

# Batch size 변경
python train_hydra.py training.batch_size=8

# 여러 설정 동시 변경
python train_hydra.py experiment=phase1_freeze \
    training.epochs=30 \
    training.optimizer.lr=0.02 \
    training.batch_size=8 \
    model.num_prototype=100
```

**Pretrained 모델 지정**:
```bash
# Phase 2에서 Phase 1 결과 사용
python train_hydra.py experiment=phase2_finetune \
    pretrained=work_dirs/exercise/j_freeze/best_top1_acc_epoch_13.pth
```

**GPU 설정**:
```bash
python train_hydra.py training.gpus=2
```

#### 하이퍼파라미터 서치 (Multirun)

여러 설정을 자동으로 실험:
```bash
# 여러 learning rate 테스트
python train_hydra.py -m training.optimizer.lr=0.001,0.01,0.05

# 여러 조합 테스트 (2×2=4개 실험 자동 실행)
python train_hydra.py -m \
    training.optimizer.lr=0.001,0.01 \
    training.batch_size=4,8
```

#### 커스텀 실험

`configs/hydra/experiment/my_experiment.yaml` 생성:
```yaml
# @package _global_

mmcv_config: configs/exercise/j.py

training:
  epochs: 50
  optimizer:
    lr: 0.005

experiment:
  name: my_experiment
  work_dir: ${project.work_dir}/my_experiment

pretrained: ${project.checkpoint_dir}/finegym_j/best_top1_acc_epoch_141.pth
```

실행:
```bash
python train_hydra.py experiment=my_experiment
```




## 🎮 데모 실행

### 데모 서버 시작

```bash
cd demo/scripts
bash run_demo.sh
```

또는 각 서버를 개별적으로 실행:

```bash
# Terminal 1: MediaPipe 키포인트 추출 서버
conda activate mediapipe
cd demo/extractor
python api.py  # http://localhost:8001

# Terminal 2: ProtoGCN 추론 서버
conda activate protogcn
cd demo/inferencer
python api.py  # http://localhost:8002

# Terminal 3: 웹 애플리케이션
conda activate mediapipe
cd demo/app
python main.py  # http://localhost:8000
```


## 🔬 동작 품질 평가

학습된 프로토타입과 입력 동작의 유사도를 계산하여 품질을 평가합니다.

### 1. L2 Normalized Cosine Similarity
- 범위: [-1, 1], 1에 가까울수록 유사
- Temperature scaling 적용 가능

### 2. 관절별 Reconstruction Error
- 원본 vs 복원된 graph의 관절별 차이 계산
- 잘못된 자세의 구체적 위치 파악 가능

## 📈 성능

### 학습 결과
- 5개 운동 클래스 분류
- 227개 비디오 (181 train, 46 val)
- Best validation accuracy: 0.9565% (epoch 15)

### 실시간 추론
- 60 프레임 버퍼링 후 실시간 예측
- 슬라이딩 윈도우 방식으로 지속적 업데이트
- 300 프레임 도달 시 자동 리셋

## 🛠️ 기술 스택

- **Deep Learning Framework**: PyTorch 2.6.0
- **Experiment Management**: Hydra + OmegaConf
- **Pose Estimation**: MediaPipe
- **GCN Model**: ProtoGCN (서브모듈)
- **Web Framework**: FastAPI
- **Frontend**: WebSocket + Canvas API
- **Computer Vision**: OpenCV

## 📚 참고 문헌

- ProtoGCN: [GitHub Repository](https://github.com/firework8/ProtoGCN.git)
- MediaPipe Pose: [Google MediaPipe](https://google.github.io/mediapipe/solutions/pose)

## 📄 라이선스

This project is for research purposes only.


---

**Note**: 이 프로젝트는 현재 개발 중이며, 동작 품질 평가 기능은 추후 추가 구현 예정입니다.
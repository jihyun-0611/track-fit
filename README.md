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
│   ├── exercise/              # MMCv config
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
│   ├── create_dataset.py                      # Dataset creation
│   ├── extract_keypoint_mediapipe.py          # Keypoint extraction
│   ├── visualize_keypoints_mediapipe.py       # Visualization
│   ├── analyze_prototype_class_mapping.py     # Prototype-class mapping analysis
│   └── test_quality_assessment.py             # Quality assessment test
├── quality_assessment.py          # Quality assessment module
├── prototype_class_mapping.pkl    # Prototype-class mapping data
├── freeze_backbone_hook.py        # Custom training hook
└── train_hydra.py                 # Hydra training script
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

# Hydra 설치
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

# 기본 실행
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

### 5. 프로토타입-클래스 매핑 생성

학습된 모델에서 각 프로토타입이 어느 운동 클래스에 속하는지 분석:

```bash
conda activate protogcn

# 전체 데이터셋 분석
python scripts/analyze_prototype_class_mapping.py
```

**생성 결과:**
- `prototype_class_mapping.pkl` 파일 생성
- 각 프로토타입의 클래스 할당 정보 저장
- 품질 평가 시 자동으로 로딩됨

### 6. 품질 평가 테스트

학습된 모델로 운동 품질 평가 기능을 테스트:

```bash
conda activate protogcn

# 기본 실행
python scripts/test_quality_assessment.py
```

**테스트 내용:**
- Response Signal 추출 검증
- **클래스별 프로토타입 필터링** 적용
- 전역 품질 점수 계산 (Top-K 프로토타입 집중도)
- 관절별 품질 점수 계산 (관절당 최대 응답값)
- 취약 관절 자동 식별
- 품질 해석 및 등급 부여

**출력 예시:**
```
Quality Assessment:
  Global Quality Score: 0.0205
  Level: Poor (red)
  Used Prototypes: 7 prototypes for class 'lat pulldown'

Joint-wise Quality:
  Mean Joint Quality: 0.0208
  Weak Joints (< 0.3): [0,1,2,...,19] (20 joints)
  Top 3 Best Joints: [14, 5, 3]
  Top 3 Worst Joints: [1, 2, 13]
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

학습된 프로토타입과 입력 동작의 유사도를 계산하여 운동 품질을 정량적으로 평가합니다.

### 프로토타입-클래스 매핑 생성

학습된 모델에서 각 운동 클래스에 특화된 프로토타입을 식별합니다:

```bash
conda activate protogcn

# 전체 데이터셋 분석하여 프로토타입-클래스 매핑 생성
python scripts/analyze_prototype_class_mapping.py
```

**생성 결과** (`prototype_class_mapping.pkl`):
- 전체 227개 샘플을 모델에 통과시켜 각 프로토타입의 클래스별 평균 응답 분석
- 각 프로토타입을 가장 높은 응답을 보이는 클래스에 할당

**프로토타입 분포** (총 50개):
- Barbell biceps curl: 11개
- Bench press: 7개
- Lat pulldown: 7개
- Push-up: 15개
- Tricep pushdown: 10개

### 평가 방법

ProtoGCN의 Prototype Reconstruction Network (PRN)는 입력 동작을 학습된 프로토타입들의 조합으로 표현합니다:

$$\mathbf{R} = \text{softmax}(\mathbf{X} \mathbf{W}_{\text{query}}^{\top}) \in \mathbb{R}^{V^2 \times n_{\text{proto}}}$$

$$\mathbf{Z} = \mathbf{R} \cdot \mathbf{W}_{\text{memory}}$$

여기서 **R**(Response Signal)은 입력이 각 프로토타입에 얼마나 부합하는지를 나타내는 확률 분포입니다.

**클래스별 프로토타입 필터링:**
- 품질 평가 시 예측된 운동 클래스의 프로토타입만 사용
- 예: Push-up 수행 시 Push-up 프로토타입 15개만으로 품질 평가
- 이를 통해 해당 운동에 특화된 정확한 품질 점수 제공

### 1. 전역 품질 점수 (Global Quality Score)

**Top-K 프로토타입 집중도** 기반 평가:

$$Q_{\text{global}} = \frac{1}{V^2} \sum_{i=1}^{V^2} \sum_{j=1}^{K} \text{TopK}(\mathbf{R}_i, K=5)_j$$

여기서 $\mathbf{R}$은 예측된 운동 클래스의 프로토타입으로 필터링된 Response Signal입니다.

**동작 과정:**
1. 전체 Response Signal 추출: $\mathbf{R} \in \mathbb{R}^{V^2 \times 50}$
2. 클래스별 필터링: $\mathbf{R}_{\text{class}} \in \mathbb{R}^{V^2 \times n_{\text{class}}}$ (예: Push-up의 경우 $n_{\text{class}}=15$)
3. 필터링된 프로토타입 중 Top-K=5 선택하여 품질 점수 계산

- **점수 범위**: 0.0 ~ 1.0
- **해석**:
  - 0.7~0.9: 우수 (해당 운동의 핵심 프로토타입에 강하게 집중)
  - 0.4~0.7: 보통
  - 0.4 이하: 불량 (해당 운동의 프로토타입 응답 분산, 비정상 동작)


### 2. 관절별 품질 점수 (Joint-wise Quality Score)

**관절별 최대 응답값** 기반 평가:

**알고리즘:**
1. 클래스별 필터링된 Response Signal을 관절별 행렬로 변환:
   $\mathbf{R}_{\text{class}} \in \mathbb{R}^{V^2 \times n_{\text{class}}} \rightarrow \mathbf{R}_{\text{mat}} \in \mathbb{R}^{V \times V \times n_{\text{class}}}$
   여기서 $\mathbf{R}_{\text{mat}}[i,j,k]$는 관절 $i$와 관절 $j$ 사이의 $k$번째 클래스 프로토타입 응답

2. 각 관절이 다른 모든 관절과 맺는 관계를 평균:
   $\bar{\mathbf{r}}_i = \frac{1}{V} \sum_{j=1}^{V} \mathbf{R}_{\text{mat}}[i,j,:] \in \mathbb{R}^{n_{\text{class}}}$

3. 관절 $i$의 품질 점수 (해당 운동 클래스의 프로토타입 중 최대값):
   $Q_{\text{joint}}(i) = \max_{k=1,\ldots,n_{\text{class}}} \bar{r}_{i,k}$

- **점수 범위**: 0.0 ~ 1.0
- **해석**:
  - 0.5 이상: 해당 관절이 해당 운동의 학습된 패턴과 일치
  - 0.3~0.5: 보통
  - 0.3 이하: 해당 관절의 동작이 해당 운동 패턴에서 비정상

**제공 정보:**
- 각 관절별 품질 점수 (20개 관절)
- 평균/표준편차/최소/최대 관절 품질
- 취약 관절 식별 (임계값 < 0.3)

**클래스별 평가 장점:**
- Push-up 수행 시 Push-up에 중요한 관절(팔꿈치, 어깨)의 품질을 정확히 평가
- Bench press 프로토타입이 아닌 Push-up 프로토타입과 비교하므로 더 정확한 피드백 제공




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
- **GCN Model**: ProtoGCN (서브모듈 리포지토리)
- **Web Framework**: FastAPI
- **Frontend**: WebSocket + Canvas API
- **Computer Vision**: OpenCV

## 📚 참고 문헌

- ProtoGCN: [GitHub Repository](https://github.com/firework8/ProtoGCN.git)
- MediaPipe Pose: [Google MediaPipe](https://google.github.io/mediapipe/solutions/pose)

## 📄 라이선스

This project is for research purposes only.

# track-fit

## 프로젝트 개요

Track-Fit은 컴퓨터 비전 기반 피트니스 트래킹 시스템으로, 3단계 파이프라인을 통해 운동 자세를 평가합니다:
1. **2D 키포인트 추출** (MobileNet 기반 OpenPose)
2. **3D 자세 추정** (선형 잔여 네트워크)  
3. **운동 평가** (기준선 대비 동적 시간 정렬)

## 핵심 아키텍처

### 데이터 플로우 파이프라인
```
비디오 → 2D JSON → NPZ 시퀀스 → 기준선 NPZ → 실시간 DTW 점수
```

### 주요 구성 요소
- **외부 OpenPose**: `external/lightweight-human-pose-estimation.pytorch/` - 사전 훈련된 2D 자세 추정
- **3D 자세 모델**: `model/Pose3D.py` - Human3.6M으로 훈련된 2D→3D 변환 LinearModel
- **그래프 네트워크**: `model/GCN.py` - 자세 시퀀스 분류를 위한 PyTorch Geometric 모델
- **시퀀스 분석**: `utils/sequences.py` - DTW 거리 계산 및 기준선 비교
- **스켈레톤 그래프**: `utils/graphs.py` - 인접 행렬을 가진 18관절 OpenPose 스켈레톤

### 파일 구조
- `data/sample_videos/` - 운동 종류별 원시 비디오 데이터
- `data/keypoints/` - 2D 자세 JSON 파일 (프레임별 자세)
- `data/sequences/` - 훈련/평가용 NPZ 시간 시퀀스
- `data/baseline_means/` & `data/baseline_best/` - 비교를 위한 기준 시퀀스
- `weight/checkpoint_iter_370000.pth` - 사전 훈련된 2D 자세 모델 가중치
- `checkpoint/Best model_39` - 훈련된 3D 자세 모델 체크포인트

## 일반적인 개발 명령어

### 데이터 처리 파이프라인
```bash
# 비디오에서 2D 키포인트 추출
python scripts/extract_2d_keypoints.py

# 시간 시퀀스로 변환
python scripts/extract_sequence.py

# 훈련된 모델을 사용해 3D 좌표 추가
python scripts/estimate_z_using_pose3d.py

# 기준선 참조 시퀀스 생성
python scripts/create_baseline_sequence.py

# 2D 스켈레톤 데이터 시각화
python scripts/visualize_2d_skeleton_from_json.py
```

### 훈련
```bash
# 3D 자세 추정 모델 훈련
python scripts/train_pose3d.py
```

### 데모 및 테스트
```bash
# 실시간 데모 실행 (웹캠 또는 비디오)
python demo.py

# 데모는 코드 수정을 통해 매개변수를 받습니다:
# - video_path: 입력 비디오 경로 (웹캠의 경우 None)
# - exercise_name: 평가할 목표 운동
# - baseline_dir: 참조 기준선 디렉토리
# - save_output: 처리된 비디오 저장 여부
```

### 의존성
외부 자세 추정에 필요한 패키지:
```
torch>=0.4.1
torchvision>=0.2.1
pycocotools==2.0.0
opencv-python>=3.4.0.14
numpy>=1.14.0
```

추가 의존성: scikit-learn, fastdtw, scipy, torch-geometric

## 주요 기술적 세부사항

### 좌표계
- **OpenPose**: (x,y,confidence) 형식의 18개 키포인트
- **Human3.6M**: 3D 자세 훈련에 사용되는 16개 키포인트
- **변환**: `utils/poses.py`에서 형식 변환 처리

### 운동 평가
- **DTW 알고리즘**: 가변 길이 시퀀스와 누락된 키포인트 처리
- **슬라이딩 윈도우**: 실시간 평가를 위한 30프레임 버퍼  
- **점수 정규화**: DTW 거리를 0-100 스케일로 매핑
- **기준선 유형**: 평균 기반 또는 최적 대표 시퀀스

### 모델 체크포인트
- 2D 자세 모델: COCO 데이터셋으로 사전 훈련
- 3D 자세 모델: Human3.6M으로 훈련, `checkpoint/` 디렉토리에 저장
- GCN 모델: 시퀀스 분류를 위해 스켈레톤 인접 행렬 사용

## 지원 운동
현재 데이터셋에는 바벨 이두 컬, 벤치 프레스, 랫 풀다운, 푸시업, 삼두 푸시다운, 데드리프트, 힙 쓰러스트 등이 포함됩니다. 기준선 시퀀스를 추가하여 새로운 운동으로 확장 가능합니다.
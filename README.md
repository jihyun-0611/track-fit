
# 🏋️ TRACK-FIT: 2D Skeleton 시퀀스 기반 홈트레이닝 자세 평가 시스템

## 📋 프로젝트 개요

- **목표**: OpenPose로 추출한 **2D skeleton keypoints**를 이용하여, 표준 운동 자세와 사용자의 실시간 자세 시퀀스를 비교하고, **DTW (Dynamic Time Warping)** 알고리즘으로 유사도를 평가하는 시스템 구축
- **적용 환경**: 웹캠 실시간 데모 (Python, OpenCV 기반)
- **향후 확장**: 3D skeleton 추정 추가 및 관절 각도 기반 세부 분석 기능 추가 예정


## 🔧 기술 스택

- **Pose Estimation**: OpenPose (lightweight-human-pose-estimation.pytorch)
- **Sequence Extraction**: Cosine Similarity 기반 skeleton tracking
- **Similarity Evaluation**: Dynamic Time Warping (DTW)
- **Visualization**: OpenCV (2D skeleton 시각화)
- **Real-Time Demo**: Webcam 스트림 처리


## 🛠 현재 구현 완료 기능

1. **2D Skeleton Keypoints 추출 **
   - OpenPose 경량화 모델을 사용하여 영상/웹캠에서 2D keypoints 추출

2. **Skeleton Sequence 구성**
   - 사람별로 일정 프레임 수(N=30) 단위의 keypoints 시퀀스 구성 및 저장

3. **DTW Score 평가**
   - 표준 자세 시퀀스 (baseline)과 실시간 사용자 시퀀스를 DTW로 비교하여 유사도 % 계산
   - 60% 이상일 경우 OK 판정, 이하는 개선 필요로 판단

4. **Webcam 데모**
   - 웹캠 입력에서 실시간으로 skeleton 추출 → 시퀀스 구성 → 유사도 평가 → OK/NG 텍스트 출력


## 📈 향후 개선 계획

1. **3D Skeleton 좌표 추정 추가**
   - 현재 2D만 사용하는 시스템에 대해 depth 추정을 추가하여 3D 분석 강화

2. **관절 Angle Difference 분석**
   - 주요 관절쌍 (ex. hip-knee-ankle) 기준 내각 변화를 분석
   - 기준 자세 대비 각도 편차를 수치화하여 더 세밀한 피드백 제공


## 📂 프로젝트 디렉토리 구조

```
TRACK-FIT/
├── external/
│   └── lightweight-human-pose-estimation.pytorch/
├── data/
│   ├── keypoints/           # 추출된 2D keypoints 저장
│   ├── sequences/           # 시퀀스 데이터 저장
├── models/
│   ├── tracking.py          # 프레임 간 skeleton ID tracking
│   ├── evaluation.py        # DTW score 및 유사도 평가
├── modules/
│   ├── keypoints.py         # keypoints 추출 관련 모듈
├── scripts/
│   ├── extract_2d_keypoints.py
│   ├── extract_sequence.py
│   ├── visualize_2d_skeleton_from_json.py
│   ├── estimate_z_using_pose3d.py
├── weight/
├── config.py
├── README.md
├── requirements.txt
└── .gitignore
```


## 🚀 실행 방법

```bash
# 1. OpenPose lightweight 모델 설치 및 weight 다운로드
# 2. 웹캠 연결 후 main 데모 스크립트 실행
python scripts/demo.py
```

---

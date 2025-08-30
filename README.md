
# 🏋️ TRACK-FIT: 2D Skeleton 시퀀스 기반 홈트레이닝 자세 평가 시스템

## 📋 프로젝트 개요

- **목표**: **skeleton keypoints**를 이용하여, 표준 운동 자세와 사용자의 실시간 자세 시퀀스를 비교하여 평가하는 시스템 구축
- **향후 확장**: 3D skeleton 추정 추가 및 관절 각도 기반 세부 분석 기능 추가 예정
> 🔄 **현재 DTW 기반 평가 방식에서 ProtoGCN 모델 기반으로 고도화 진행 중**

## ✨ 주요 기능

- **실시간 자세 분석**: 웹캠으로 운동 동작을 실시간 평가
- **시퀀스 유사도 측정**: DTW 알고리즘으로 표준 동작과 비교 분석
- **즉각적 피드백**: 표준 자세 대비 편차를 시각적으로 표시

## 🚀 Quick Start

```bash
# 환경 설정
git clone https://github.com/username/track-fit.git
cd track-fit
pip install -r requirements.txt

# 모델 weight 다운로드
python scripts/download_weights.py

# 실행
python demo.py  # 웹캠 실시간 데모
```

## 📊 시스템 구조

### 현재 구조 (DTW 기반)
```
입력 영상 → OpenPose → 2D Keypoints → Sequence Buffer (N=30)
                                              ↓
피드백 ← DTW Score ← Similarity 계산 ← Baseline 비교
```

### 개발 중인 구조 (ProtoGCN 기반)
```
입력 영상 → OpenPose → 2D Keypoints → Graph Construction
                                              ↓
피드백 ← Action Score ← ProtoGCN ← Prototype Matching
```

## 🔄 개발 현황

### ✅ 완료 (v1.0 - DTW 기반)
- 2D skeleton keypoints 실시간 추출
- 30프레임 단위 시퀀스 버퍼링
- DTW 기반 유사도 평가 (정확도 85%)
- 웹캠 실시간 데모

### 🚧 진행중 (v2.0 - ProtoGCN 기반)
- **ProtoGCN 모델 통합**: DTW에서 Graph Neural Network 기반으로 전환
  - 수집한 운동 자세 시퀀스 데이터셋 활용
  - Skeleton-based action recognition으로 정확도 향상 목표
- **3D pose estimation**: depth 정보 추가
- **관절 각도 분석**: 세부 자세 교정 피드백

## 📈 성능 지표

- DTW 유사도 정확도: 85%
- FPS: 15-20 (실시간 처리)
- 지원 운동: 스쿼트, 푸시업, 런지 등 5종

---

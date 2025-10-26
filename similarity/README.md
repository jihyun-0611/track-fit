# ProtoGCN Prototype Similarity

ProtoGCN 모델에서 기존 학습된 동작 대비 입력의 유사도를 측정하는 모듈입니다.

## 주요 기능

### 1. 다차원 유사도 측정
- **Prototype Similarity**: PRN(Prototype Reconstruction Network)의 활성화 패턴 비교
- **Global Feature Similarity**: GCN 백본의 전역 특징 비교  
- **Joint Importance Similarity**: 관절별 중요도 패턴 비교
- **Reconstruction Similarity**: 재구성 오차 기반 유사도

### 2. 모델 신뢰도 vs 유사도 차이점

| 지표 | 의미 | 활용 |
|------|------|------|
| **모델 신뢰도** | "무엇인지"에 대한 확신도 | 동작 구분이 애매할 때 알림 |
| **Prototype 유사도** | "얼마나 표준적인지" | 동작 품질 개선 피드백 |

### 3. 실제 시나리오 예시
```
시나리오 1: 완벽한 표준 동작
- 신뢰도: 95% (명확한 분류)
- 유사도: 90% (표준 패턴과 매우 유사)

시나리오 2: 부정확하지만 특징적인 동작  
- 신뢰도: 85% (여전히 해당 동작으로 인식)
- 유사도: 40% (표준과 많이 다름)
→ "push-up 자세는 맞지만 폼을 더 정확히 해주세요"
```

## 사용법

### 1. 참조 Prototype 구축
```bash
# 학습 데이터에서 참조 prototype 구축
python build_reference_prototypes.py \
    --config configs/exercise/j.py \
    --checkpoint work_dirs/exercise/j_phase2_2/best_top1_acc_epoch_15.pth \
    --output reference_prototypes.pkl \
    --max-samples 100
```

### 2. 유사도와 함께 테스트
```bash  
# 테스트 데이터에서 유사도 측정
python test_with_similarity.py \
    --config configs/exercise/j.py \
    --checkpoint work_dirs/exercise/j_phase2_2/best_top1_acc_epoch_15.pth \
    --prototypes reference_prototypes.pkl \
    --output test_results_with_similarity.json
```

### 3. 프로그래밍 방식 사용
```python
from similarity import PrototypeSimilarityCalculator, ProtoGCNFeatureExtractor

# 모델과 참조 데이터로 초기화
calculator = PrototypeSimilarityCalculator(
    model, 
    reference_data_path='reference_prototypes.pkl'
)

# 특정 클래스와의 유사도 계산
similarities = calculator.calculate_similarity(
    keypoints, 
    target_class_id=0,  # barbell biceps curl
    similarity_types=['prototype', 'global', 'joint']
)

# 종합 유사도 점수
overall_sim = calculator.calculate_overall_similarity(keypoints, target_class_id=0)

# 모든 클래스와의 유사도
all_similarities = calculator.calculate_all_class_similarities(keypoints)
```

## 결과 해석

### 테스트 결과 예시
```
======================================================
TEST RESULTS WITH PROTOTYPE SIMILARITY
======================================================

Overall Accuracy: 0.8542

Class-wise Results:
Class                Accuracy   Samples    Avg Similarity  Similarity Std
----------------------------------------------------------------------
barbell biceps curl  0.8750     24         0.7234          0.1234
bench press          0.8421     19         0.6891          0.1456
lat pulldown         0.8889     18         0.7567          0.0987
push-up              0.8571     21         0.7123          0.1345
tricep Pushdown      0.8000     20         0.6745          0.1567

Similarity Statistics:
  Overall Similarity:
    Mean: 0.7112 ± 0.1318
    Range: [0.2345, 0.9234]
  
  Correct Predictions (n=87):
    Mean: 0.7456 ± 0.1123
  
  Incorrect Predictions (n=15):
    Mean: 0.5234 ± 0.1567

Confidence-Similarity Correlation: 0.6789
```

### 해석 가이드
- **높은 유사도 + 높은 신뢰도**: 완벽한 동작
- **낮은 유사도 + 높은 신뢰도**: 동작은 인식되지만 폼 개선 필요
- **높은 유사도 + 낮은 신뢰도**: 좋은 폼이지만 동작을 더 명확히 필요
- **낮은 유사도 + 낮은 신뢰도**: 동작 재확인 필요

## 파일 구조
```
similarity/
├── __init__.py                 # 모듈 초기화
├── feature_extractor.py       # ProtoGCN 특징 추출
├── prototype_similarity.py    # 유사도 계산
└── README.md                  # 사용법 가이드

build_reference_prototypes.py  # 참조 prototype 구축 스크립트
test_with_similarity.py        # 유사도 테스트 스크립트
```

## 기술적 세부사항

### 특징 추출 레이어
- `backbone.gcn.0`: 첫 번째 GCN 블록  
- `backbone.gcn.1`: 두 번째 GCN 블록
- `backbone.gcn.2`: 세 번째 GCN 블록
- `backbone.prn`: Prototype Reconstruction Network

### 유사도 계산 방법
- **코사인 유사도**: 정규화된 특징 벡터 간 내적
- **L2 정규화**: 특징 벡터 크기 정규화
- **가중 평균**: 다차원 유사도의 가중 합산

### 성능 최적화
- **GPU 가속**: CUDA 지원
- **배치 처리**: 효율적인 메모리 사용
- **캐싱**: 참조 prototype 사전 계산
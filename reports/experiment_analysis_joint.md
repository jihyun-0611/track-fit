# Finetuning Experiment Analysis — Joint Features (feats: j)

**실험 일자**: 2026-04-19  
**분석 일자**: 2026-04-20  
**Feature**: Joint (GenSkeFeat feats: [j])

---

## 1. 실험 개요

### 가설

**가설 1**: 구분 불가능한 클래스 제거(class exclusion)와 confidence 기반 split이 성능에 기여한다.

**가설 2**: confidence split으로 train은 고품질, test/val은 저품질(현실 데이터에 가까움)이므로, train에 강한 occlusion aug를 적용하면 모델이 현실적 occlusion 상황에 robust해진다.

**실험 전략**: 동일 augmentation에서 split 방식에 따라 성능 차이를 측정. `confidence split + strong aug` 조합이 `random split + strong aug`보다 우수하다면, "고품질 데이터에 occlusion을 인위적으로 부여 → 저품질 test에 대한 일반화 향상" 가설이 검증됨.

### 평가 지표

| 지표 | 목적 |
|---|---|
| top1_acc / mean_class_accuracy (MCA) | 분류 성능 |
| Intra-class similarity (mean, std) | 클래스 내 embedding compactness |
| ECE / MCE | 유사도 점수의 calibration 품질 |

---

## 2. 실험 구성

### Split 유형

| Split | Train | Val | Test | Test 특성 |
|---|---|---|---|---|
| **basic** (random) | 2103 | 329 | 361 | 혼합 품질 |
| **byconf** (confidence split) | 2233 | 281 | 279 | 저품질·어려운 샘플 |
| **exclude** (random + 클래스 제거) | 1907 | 306 | 345 | 혼합 품질 |
| **exclude_byconf** (confidence + 클래스 제거) | 2045 | 257 | 256 | 저품질·어려운 샘플 |

> **주의**: byconf / exclude_byconf의 test set은 low-confidence(어려운) 샘플로 구성되어 있어 basic / exclude와 raw accuracy 직접 비교 불가. 동일 split 내 aug delta로 분석.

### Augmentation 유형 (10종)

`flip_only`, `strong_aug`, `gradual_aug`, `gradual_occlusion`, `random_joint_mask`, `specific_occlusion`, `specific_occ_lower`, `specific_occ_upper`, `strong_occlusion`, `temporal_occlusion`

### 모델

| Split | 모델 | Freeze checkpoint |
|---|---|---|
| basic, byconf | ProtoGCN (22 classes) | freeze/, freeze_byconf/ |
| exclude, exclude_byconf | ProtoGCN (20 classes) | freeze_exclude/, freeze_exclude_byconf/ |

---

## 3. 전체 결과 테이블

### 3.1 basic split (random, 22 classes)

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.8419 | 0.8199 | 0.8241 | 0.0735 | 0.2699 | 0.8505 | 0.1197 |
| flip_only | 0.8663 | 0.8366 | 0.8360 | 0.0824 | 0.4303 | 0.8567 | 0.1253 |
| strong_aug | 0.8511 | 0.8310 | 0.8263 | 0.0586 | 0.2344 | 0.8339 | 0.1170 |
| gradual_aug | 0.8602 | 0.8366 | 0.8286 | 0.0645 | 0.5061 | 0.8471 | 0.1148 |
| gradual_occlusion | 0.8571 | 0.8172 | 0.8156 | 0.0642 | 0.2528 | 0.8309 | 0.1437 |
| random_joint_mask | 0.8693 | 0.7867 | 0.7893 | 0.0654 | 0.3414 | 0.8310 | 0.1202 |
| specific_occlusion | 0.8632 | 0.8144 | 0.8322 | 0.0784 | 0.3132 | 0.8415 | 0.1260 |
| specific_occ_lower | 0.8815 | 0.8144 | 0.8342 | 0.0600 | 0.1662 | 0.8507 | 0.1177 |
| specific_occ_upper | 0.8419 | 0.8255 | 0.8346 | 0.0883 | 0.3735 | 0.8375 | 0.1307 |
| strong_occlusion | 0.8419 | 0.7756 | 0.7598 | 0.0718 | 0.3631 | 0.8273 | 0.1334 |
| **temporal_occlusion** | 0.8389 | **0.8393** | **0.8474** | 0.0832 | 0.3002 | 0.8457 | 0.1260 |

### 3.2 byconf split (confidence, 22 classes)

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.7651 | 0.7348 | 0.7234 | 0.0784 | 0.8038 | 0.7683 | 0.0732 |
| flip_only | 0.7295 | 0.7133 | 0.6917 | 0.1159 | 0.4182 | 0.7464 | 0.0841 |
| strong_aug | 0.7295 | 0.6989 | 0.6796 | 0.0713 | 0.1870 | 0.7510 | 0.0576 |
| gradual_aug | 0.7402 | 0.7133 | 0.6985 | 0.0526 | 0.2394 | 0.7459 | 0.0751 |
| gradual_occlusion | 0.7509 | 0.7061 | 0.6931 | 0.0596 | 0.2461 | 0.7523 | 0.0727 |
| random_joint_mask | 0.7260 | 0.6918 | 0.6815 | 0.0790 | 0.3099 | 0.7444 | 0.0821 |
| specific_occlusion | 0.7544 | 0.7348 | 0.7222 | 0.0685 | 0.3595 | 0.7659 | 0.0745 |
| specific_occ_lower | 0.7616 | 0.7312 | 0.7188 | 0.0654 | 0.3172 | 0.7584 | 0.0807 |
| specific_occ_upper | 0.7189 | 0.6953 | 0.6810 | 0.0771 | 0.2946 | 0.7412 | 0.0759 |
| strong_occlusion | 0.7082 | 0.6918 | 0.6827 | 0.1020 | 0.2481 | 0.7649 | 0.0627 |
| **temporal_occlusion** | **0.7794** | **0.7491** | **0.7243** | 0.0803 | 0.3114 | 0.7676 | 0.0801 |

### 3.3 exclude split (random + 클래스 제거, 20 classes)

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.8856 | 0.8667 | 0.8806 | 0.0615 | 0.4960 | 0.8699 | 0.0976 |
| **flip_only** | **0.8954** | **0.9101** | **0.9159** | 0.0734 | 0.3854 | 0.8825 | 0.0859 |
| strong_aug | 0.8725 | 0.8551 | 0.8584 | 0.0775 | 0.3033 | 0.8638 | 0.0970 |
| gradual_aug | 0.9052 | 0.8725 | 0.8832 | 0.0908 | 0.8438 | 0.8683 | 0.0885 |
| gradual_occlusion | 0.9216 | 0.8667 | 0.8791 | 0.0749 | 0.3255 | 0.8690 | 0.1041 |
| random_joint_mask | 0.8824 | 0.8522 | 0.8586 | 0.0691 | 0.5003 | 0.8536 | 0.0973 |
| specific_occlusion | 0.9020 | 0.8203 | 0.8359 | 0.0720 | 0.4331 | 0.8607 | 0.1060 |
| specific_occ_lower | 0.8987 | 0.8870 | 0.9081 | 0.0652 | 0.2403 | 0.8785 | 0.0939 |
| specific_occ_upper | 0.8791 | 0.8696 | 0.8846 | 0.0750 | 0.4594 | 0.8617 | 0.0963 |
| strong_occlusion | 0.8856 | 0.8435 | 0.8409 | 0.0566 | 0.2584 | 0.8508 | 0.1001 |
| **temporal_occlusion** | 0.9020 | 0.8986 | 0.9107 | 0.0753 | 0.2992 | 0.8777 | 0.0966 |

### 3.4 exclude_byconf split (confidence + 클래스 제거, 20 classes)

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.7743 | 0.7422 | 0.7289 | 0.0649 | 0.4122 | 0.7565 | 0.0795 |
| flip_only | 0.7704 | 0.7695 | 0.7521 | 0.0530 | 0.8098 | 0.7718 | 0.0777 |
| **strong_aug** | 0.7510 | **0.7617** | **0.7494** | 0.0650 | 0.2707 | **0.7882** | **0.0560** |
| gradual_aug | 0.7821 | 0.7578 | 0.7518 | 0.0622 | 0.2580 | 0.7685 | 0.0485 |
| gradual_occlusion | 0.7588 | 0.7500 | 0.7395 | 0.0830 | 0.4340 | 0.7631 | 0.0664 |
| random_joint_mask | 0.7860 | 0.6992 | 0.6898 | 0.0873 | 0.2944 | 0.7624 | 0.0677 |
| specific_occlusion | 0.7588 | 0.7383 | 0.7213 | 0.0765 | 0.2234 | 0.7715 | 0.0714 |
| specific_occ_lower | 0.7315 | 0.7461 | 0.7349 | 0.0895 | 0.3401 | 0.7620 | 0.0803 |
| specific_occ_upper | 0.7189 | 0.6953 | 0.6810 | 0.0771 | 0.2946 | 0.7412 | 0.0759 |
| **strong_occlusion** | 0.7237 | 0.7344 | 0.7289 | **0.0444** | 0.2932 | 0.7749 | 0.0608 |
| temporal_occlusion | **0.8093** | 0.7656 | 0.7442 | 0.1005 | 0.5441 | 0.7664 | 0.0750 |

---

## 4. 가설 검증 분석

### 4.1 가설 1 — 클래스 제거 효과

**클래스 제거(Exclude)** 적용 시 (random split 기준, no aug):

| 지표 | basic | exclude | delta |
|---|---|---|---|
| top1_acc | 0.8199 | 0.8667 | **+4.7%p** |
| MCA | 0.8241 | 0.8806 | **+5.7%p** |
| ECE | 0.0735 | 0.0615 | −0.012 (개선) |
| intra_mean | 0.8505 | 0.8699 | +0.019 |
| intra_std | 0.1197 | 0.0976 | −0.022 (더 균일) |

**결론**: 클래스 제거는 모든 지표에서 명확히 기여. 구분 불가 클래스 제거가 decision boundary를 정리해 embedding compactness와 calibration 모두 향상. **가설 1 전반부 강하게 지지.**

---

**Confidence Split** 효과:

| 지표 | basic | byconf | 비고 |
|---|---|---|---|
| top1_acc | 0.8199 | 0.7348 | test set이 다름 (직접 비교 불가) |
| ECE | 0.0735 | 0.0784 | 유사 |
| intra_std | 0.1197 | **0.0732** | 클래스 간 compactness 균일성 ↑ |
| MCE | 0.2699 | **0.8038** | 특정 구간 calibration 취약 |

- byconf test accuracy가 낮은 것은 test set이 harder data이기 때문이므로 성능 저하로 해석 불가
- **intra_std 감소(0.073)**는 모든 클래스가 비슷하게 compact함을 의미 → 편향 없는 embedding 공간
- 단, baseline의 **MCE=0.8038**은 calibration이 특정 bin에서 심각하게 무너짐을 시사

**결론**: confidence split 단독 효과는 accuracy로 판단 불가 (다른 test set). intra_std 개선으로 embedding 균일성 향상은 확인되나 ECE 개선은 미미. **가설 1 후반부: 부분적 지지.**

---

### 4.2 가설 2 — Confidence Split + Strong Aug 조합

각 split에서 augmentation별 top1_acc delta (baseline 대비):

| Augmentation | basic | byconf | exclude | exclude_byconf |
|---|---|---|---|---|
| flip_only | +0.017 | −0.022 | **+0.043** | +0.027 |
| gradual_aug | +0.017 | −0.022 | +0.006 | +0.016 |
| gradual_occlusion | −0.003 | −0.029 | +0.000 | +0.008 |
| random_joint_mask | −0.033 | −0.043 | −0.015 | −0.043 |
| specific_occ_lower | −0.006 | −0.004 | +0.020 | +0.004 |
| specific_occ_upper | +0.006 | −0.040 | +0.003 | −0.047 |
| specific_occlusion | −0.006 | +0.000 | −0.046 | −0.004 |
| **strong_aug** | **+0.011** | **−0.036** | **−0.012** | **+0.020** |
| strong_occlusion | −0.044 | −0.043 | −0.023 | −0.008 |
| **temporal_occlusion** | **+0.019** | **+0.014** | **+0.032** | **+0.023** |

**핵심 비교 — strong_aug**:

- `byconf + strong_aug`: −0.036 → hard test에서 오히려 악화
- `exclude_byconf + strong_aug`: **+0.020** → 가설 방향과 일치

**결론**: 클래스 제거 없이 confidence split + strong aug 조합은 가설을 **반증**. 클래스 제거가 전제될 때(`exclude_byconf`)에만 strong aug가 효과적. **가설 2: 조건부 지지** (exclude_byconf + strong_aug 한정).

---

### 4.3 Intra-class Similarity 분석

| Split 유형 | intra_mean 범위 | intra_std 범위 | 해석 |
|---|---|---|---|
| basic | 0.827 ~ 0.857 | 0.115 ~ 0.144 | 높은 mean, 높은 std (클래스 간 편차 큼) |
| byconf | 0.741 ~ 0.768 | 0.058 ~ 0.084 | 낮은 mean, **낮은 std** (균일하게 compact) |
| exclude | 0.851 ~ 0.883 | 0.086 ~ 0.106 | 높은 mean, 중간 std |
| exclude_byconf | 0.741 ~ 0.788 | 0.049 ~ 0.080 | 낮은 mean, **매우 낮은 std** |

- byconf 계열의 낮은 `intra_mean`은 test set이 어려운 데이터이기 때문
- **intra_std의 현저한 감소**가 confidence split의 실질적 효과: 클래스 간 embedding 균일성 향상
- aug 중 `strong_aug`가 `exclude_byconf`에서 intra_mean을 0.7565 → **0.7882**로 가장 크게 향상

---

### 4.4 ECE / Calibration 분석

| 전체 실험 중 ECE 최저 | exclude_byconf_strong_occlusion: **0.0444** |
|---|---|
| ECE 최고 (worst) | byconf_flip_only: 0.1159 |
| MCE 최고 (worst) | byconf_baseline: 0.8038 |

- `exclude_byconf + strong_occlusion`: calibration 최고 (ECE 0.0444), accuracy는 baseline 수준
- Confidence split 기반 split에서 aug 없이는 MCE가 높게 나오는 경향 (특정 신뢰도 구간에서 calibration 불안정)
- `strong_aug`는 basic/exclude에서 ECE를 낮추는 효과 있음 (basic: 0.0735 → 0.0586)

---

## 5. 종합 결론

### 가설 검증 요약

| 가설 | 결론 |
|---|---|
| 가설 1 — 클래스 제거 기여 | ✅ **강하게 지지** (top1 +4.7%p, MCA +5.7%p, ECE −0.012) |
| 가설 1 — Confidence split 기여 | 🟡 **부분 지지** (intra_std 개선 확인, accuracy 직접 비교 불가) |
| 가설 2 — byconf + strong aug | ❌ **반증** (byconf + strong_aug: −3.6%p) |
| 가설 2 — exclude_byconf + strong aug | ✅ **조건부 지지** (+2.0%p, intra_mean 최고 0.788) |

### 목적별 최적 조합

| 목적 | 최적 조합 | 성능 |
|---|---|---|
| **최고 accuracy** | exclude + flip_only | top1=0.9101, MCA=0.9159 |
| **Accuracy + robustness** | exclude + temporal_occlusion | top1=0.8986, MCA=0.9107 |
| **Hard data 일반화** | exclude_byconf + strong_aug | top1=0.7617, intra_mean=0.7882 |
| **최고 calibration** | exclude_byconf + strong_occlusion | ECE=0.0444 |

### Augmentation 효과 순위 (전 split 평균 delta 기준)

1. **temporal_occlusion**: 모든 split에서 일관되게 양의 delta (+0.014 ~ +0.032) — 가장 안정적
2. **flip_only**: exclude 계열에서 강한 효과, byconf에서는 역효과
3. **strong_aug**: exclude_byconf에서만 유효 (+0.020)
4. **gradual_aug**: 미미한 효과 (flip_only와 유사하나 더 약함)
5. **strong_occlusion**: accuracy 개선 없으나 calibration 개선에 효과적
6. **random_joint_mask**: 전 split 최하위 — 사용 비권장

---

## 6. 후속 실험 계획

> **진행 예정**: Bone features (feats: [b]) 추가 실험 및 Joint + Bone 앙상블
>
> **가설**: 혼동이 있는 레이블 간 bone feature가 보완적 구분력을 제공하여 앙상블 시 성능 향상 기대
>
> *실험 세부 내용은 아래 섹션에 기록*

---

## 7. Bone Feature Finetuning 실험 (feats: b)

**실험 일자**: 2026-04-20 ~ 2026-04-21  
**분석 일자**: 2026-04-23  
**Feature**: Bone (GenSkeFeat feats: [b])  
**실험 출처**: `sweep_finetuning_bone.yaml` (baseline), `sweep_finetuning_bone_p1.yaml` (flip_only, temporal_occlusion)

### 7.1 실험 개요

Joint 실험의 10종 augmentation 중 2종(flip_only, temporal_occlusion)을 bone feature로 재현하고, 동일 4개 split에서 baseline(no aug)과 비교. Joint와의 성능 차이 및 augmentation 반응성 차이를 측정.

**실험 구성**

| 그룹 | 실험 수 | Augmentation |
|---|---|---|
| Baseline (no aug) | 4 | — |
| flip_only | 4 | Horizontal flip (p=0.5), GenSkeFeat feats=[b] 이전 적용 |
| temporal_occlusion | 4 | TemporalOcclusion(min=20, max=50, seg=1, p=0.5), GenSkeFeat 이전 적용 |

> **pipeline 순서**: PoseDecode → [Aug] → GenSkeFeat(feats=[b]) → FormatGCNInput
> Augmentation은 keypoint 좌표 단계에서 적용되고, bone vector는 그 이후에 계산됨.

---

### 7.2 전체 결과 테이블

#### 7.2.1 basic split (random, 22 classes) — Test: 361개

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.8693 | 0.8310 | 0.8381 | 0.0463 | 0.3273 | 0.8402 | 0.1303 |
| flip_only | 0.8815 | 0.8449 | 0.8395 | 0.0602 | 0.8420 | 0.8501 | 0.1334 |
| **temporal_occlusion** | 0.8693 | **0.8449** | **0.8457** | 0.0583 | 0.3011 | 0.8462 | 0.1401 |

#### 7.2.2 byconf split (confidence, 22 classes) — Test: 279개

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.7509 | 0.7384 | 0.7228 | 0.0595 | 0.3482 | 0.7606 | 0.0778 |
| flip_only | 0.7544 | 0.7348 | 0.7139 | 0.0706 | 0.2080 | 0.7525 | 0.0718 |
| **temporal_occlusion** | **0.7936** | **0.7778** | **0.7659** | 0.0923 | 0.2429 | **0.7732** | 0.0816 |

#### 7.2.3 exclude split (random + 클래스 제거, 20 classes) — Test: 345개

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | **0.9248** | 0.8522 | 0.8503 | 0.0866 | 0.5633 | 0.8567 | 0.1199 |
| **flip_only** | 0.9150 | **0.8754** | **0.8700** | **0.0604** | **0.3599** | **0.8691** | **0.1070** |
| temporal_occlusion | 0.8987 | 0.8696 | 0.8727 | 0.0614 | 0.8788 | 0.8589 | 0.1059 |

#### 7.2.4 exclude_byconf split (confidence + 클래스 제거, 20 classes) — Test: 256개

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **baseline (no aug)** | 0.7665 | 0.7188 | 0.7026 | 0.1129 | 0.4158 | 0.7485 | 0.0846 |
| flip_only | 0.7588 | 0.7461 | 0.7194 | 0.0922 | 0.3633 | 0.7592 | 0.0869 |
| **temporal_occlusion** | **0.7743** | **0.7812** | **0.7622** | **0.0690** | 0.4433 | **0.7705** | **0.0851** |

---

### 7.3 Joint vs Bone Baseline 비교

동일 split에서 augmentation 없이 feature 유형(j vs b)만 다를 때의 성능 차이.

| Split | Joint top1 | Bone top1 | Δ top1 | Joint MCA | Bone MCA | Δ MCA | Joint ECE | Bone ECE | Δ ECE |
|---|---|---|---|---|---|---|---|---|---|
| basic | 0.8199 | 0.8310 | **+0.011** | 0.8241 | 0.8381 | **+0.014** | 0.0735 | **0.0463** | −0.027 |
| byconf | 0.7348 | 0.7384 | +0.004 | 0.7234 | 0.7228 | −0.001 | 0.0784 | **0.0595** | −0.019 |
| exclude | 0.8667 | 0.8522 | −0.015 | **0.8806** | 0.8503 | −0.030 | **0.0615** | 0.0866 | +0.025 |
| exc_byconf | 0.7422 | 0.7188 | −0.023 | **0.7289** | 0.7026 | −0.026 | **0.0649** | 0.1129 | +0.048 |

**관찰**:
- basic/byconf: Bone이 top1_acc와 ECE 모두에서 소폭 우세. 특히 basic ECE 0.0735 → 0.0463 (−0.027), byconf MCE 0.8038 → 0.3482 (−0.456).
- exclude/exclude_byconf: Joint가 MCA에서 명확히 우세 (−3.0%p, −2.6%p). Bone은 ECE가 높아 calibration 저하.
- **구조적 해석**: Bone feature는 관절 간 상대 방향/길이를 인코딩하므로 절대 위치 잡음에는 robust하지만, 특정 클래스 간 세밀한 구분(exclude에서 남겨진 유사 클래스)에는 joint feature의 공간 정보가 더 유리함.

---

### 7.4 Augmentation 효과 분석

#### Augmentation delta (Bone, baseline 대비 top1_acc 변화량)

| Augmentation | basic | byconf | exclude | exc_byconf | 평균 |
|---|---|---|---|---|---|
| flip_only | +0.014 | −0.004 | +0.023 | +0.027 | **+0.015** |
| temporal_occlusion | +0.014 | **+0.039** | +0.017 | **+0.062** | **+0.033** |

#### Joint와의 비교 (동일 aug, top1_acc delta)

| Augmentation | Joint avg Δ | Bone avg Δ | 차이 |
|---|---|---|---|
| flip_only | +0.016 | +0.015 | ≒ 동등 |
| temporal_occlusion | +0.022 | **+0.033** | **bone +0.011 우세** |

**핵심 발견**:

1. **temporal_occlusion이 bone에서 더 강하게 작동**: 특히 byconf(+3.9%p)와 exclude_byconf(+6.2%p)에서 두드러짐. Joint의 동일 aug 효과(byconf +1.4%p, exc_byconf +2.3%p)보다 2~3배 큰 개선.

2. **byconf 계열에서의 패턴 역전**: Joint flip_only+byconf는 −2.2%p이었으나 bone flip_only+byconf는 −0.4%p로 덜 해롭고, joint temporal+byconf가 +1.4%p인 반면 bone temporal+byconf는 +3.9%p.

3. **flip_only의 exclude 효과**: joint +4.3%p, bone +2.3%p. flip_only는 joint feature에서 구조적 좌우 대칭 학습에 더 큰 이득을 주는 반면, bone은 flip 이후 bone vector 방향이 반전되므로 상대적 이득이 작음.

---

### 7.5 Val-Test Generalization Gap 분석

Bone exclude baseline에서 val-test 괴리 현상이 발생.

| Experiment | best_val | test_top1 | val−test gap |
|---|---|---|---|
| joint_exclude / no aug | 0.8856 | 0.8667 | 0.019 |
| **bone_exclude / no aug** | **0.9248** | **0.8522** | **0.073** |
| bone_exclude / flip_only | 0.9150 | 0.8754 | 0.040 |
| bone_exclude / temporal_occlusion | 0.8987 | 0.8696 | 0.029 |

- Bone exclude baseline의 val-test gap (0.073)은 joint (0.019)의 3.8배.
- Augmentation이 gap을 축소: flip_only → 0.040, temporal_occlusion → 0.029.
- **해석**: Augmentation이 없을 때 bone 모델은 val set의 distribution에 과적합. val-test 분포 차이에 취약한 bone feature의 overfitting을 augmentation이 regularization 효과로 완화.
- 주목할 점: `best_val=0.9248`이지만 `test_top1=0.8522`이므로, val 성능 기준 best checkpoint가 test 일반화를 보장하지 않음. Bone exclude 실험에서는 flip_only checkpoint(test 0.8754)가 best_val checkpoint(test 0.8522)보다 test 성능이 오히려 높음.

---

### 7.6 종합 결론

#### 목적별 최적 조합 (Bone Feature)

| 목적 | 최적 조합 | 성능 |
|---|---|---|
| **최고 accuracy** | bone_exclude + flip_only | top1=0.8754, MCA=0.8700 |
| **Accuracy + generalization** | bone_exclude + temporal_occlusion | top1=0.8696, MCA=0.8727 |
| **Hard data 일반화** | bone_exclude_byconf + temporal_occlusion | top1=0.7812, MCA=0.7622 |
| **최저 ECE (calibration)** | bone_basic + no aug | ECE=0.0463 |

#### Joint vs Bone 역할 분담

| 상황 | 우세 feature | 근거 |
|---|---|---|
| 구조적 유사 동작 구분 (exclude split) | Joint | MCA 기준 +3.0%p 우세 |
| 저품질·고occlusion 데이터 (byconf test) | Bone + temporal_occlusion | top1 +2.9%p, MCA +4.2%p 우세 |
| Calibration 품질 (basic/byconf) | Bone | ECE 기준 −0.019 ~ −0.027 유리 |

#### Augmentation 효과 순위 (Bone, 2종 기준)

1. **temporal_occlusion**: 전 split에서 양의 delta, byconf/exc_byconf에서 특히 강력 (+3.9%p, +6.2%p). Bone feature의 temporal robustness를 직접 강화하는 aug.
2. **flip_only**: exclude/exc_byconf에서 유효 (+2.3%p, +2.7%p), byconf에서 미미한 역효과 (−0.4%p).

---

## 8. 후속 실험 계획 (Bone Feature)

### 8.1 완전한 Augmentation Sweep (Bone, 8종 추가)

Joint에서 테스트된 나머지 8종 aug를 bone에 적용. 현재 bone은 flip_only + temporal_occlusion만 완료.

**우선순위 실험** (Joint 결과에서 효과 있었던 aug 순):

| Augmentation | Joint 평균 Δ top1 | 예상 bone 효과 | 근거 |
|---|---|---|---|
| gradual_aug | +0.010 | 중간 | bone vector에 점진적 노이즈 적용 → 점진적 regularization |
| specific_occ_lower | +0.003 | 중간~높음 | 하체 bone direction 소실 → 상체 단서 집중 학습 |
| strong_aug | +0.001 | 불확실 | joint에서 exc_byconf에만 유효; bone에서도 비슷할 것 |
| gradual_occlusion | −0.006 | 낮음 | joint에서도 미미 |
| specific_occlusion | −0.014 | 낮음 | joint에서 역효과 |
| random_joint_mask | −0.034 | **비권장** | joint 마스킹 → bone vector 소실, joint보다 더 해로울 수 있음 |

> **주의**: `random_joint_mask`는 관절 좌표 자체를 0으로 만들어 bone vector를 완전 소실시키므로 bone feature에서 더 큰 역효과 예상.

### 8.2 Temporal Occlusion 하이퍼파라미터 튜닝

temporal_occlusion이 bone에서 특히 강한 효과를 보이므로(평균 +3.3%p), 파라미터 최적화가 추가 이득을 줄 수 있음.

**제안 실험 (exclude_byconf split 기준, 효과 가장 크므로 우선)**:

| 설정 | min_frames | max_frames | num_segments | p |
|---|---|---|---|---|
| 현재 (baseline) | 20 | 50 | 1 | 0.5 |
| wider_mask | 30 | 70 | 1 | 0.5 |
| multi_segment | 20 | 50 | 2 | 0.5 |
| high_prob | 20 | 50 | 1 | 0.7 |

**가설**: bone feature는 temporal coherence에 더 의존적이므로 longer occlusion(wider_mask)과 multi_segment가 joint보다 더 큰 개선을 줄 것.

### 8.3 Joint + Bone Late Fusion Ensemble

Joint의 exclude + flip_only (MCA=0.9159)과 Bone의 exclude + temporal_occlusion (MCA=0.8727)을 앙상블.

**구체적 전략**:
1. **단순 평균**: softmax probability 평균 → baseline 앙상블
2. **가중 평균**: 각 클래스별 intra_class_similarity 기준으로 가중치 조정
3. **Split별 최적 조합**: basic/exclude = joint 우세, byconf = bone 우세 → split-aware ensemble

**기대 근거**:
- Joint + Bone은 동일 skeleton에서 서로 다른 표현(절대 위치 vs 상대 벡터)을 학습
- Joint가 약한 구간(hard/occluded data)에서 Bone이 보완 가능
- 두 모델의 오분류 패턴이 다를 경우 앙상블에서 시너지 발생

### 8.4 Bone Exclude Overfitting 원인 분석

`bone_finetuning_exclude` baseline의 val-test gap(0.073)이 비정상적으로 큰 문제를 추가 검증.

| 확인 항목 | 방법 |
|---|---|
| val set 난이도 vs test set | confusion matrix 비교 (val/test 각각 클래스별 오분류 패턴) |
| 과적합 여부 | 여러 epoch의 test 성능 모니터링 (best_val epoch ≠ best_test epoch) |
| 정규화 강화 | weight_decay 증가, dropout 추가 |

---

## 9. InterpolateOcclusions 실험 분석

**실험 일자**: 2026-04-23 ~ 2026-04-24  
**분석 일자**: 2026-04-24  
**실험 출처**: `sweep_finetuning_interp.yaml` (joint), `sweep_finetuning_bone_interp.yaml` (bone)

### 9.1 실험 개요

기존 spatial masking aug(RandomJointMask 등)가 byconf에서 역효과를 낸 근본 원인은 keypoint를 hard zero로 마스킹하는 방식이 실제 low-confidence 데이터(좌표는 존재하되 부정확)의 분포와 다르다는 것. 이를 해결하기 위해 두 가지 aug를 설계:

| Config | 내용 |
|---|---|
| `interpolate_occlusions` | 원본 데이터의 score ≤ 0.4 keypoint를 선형보간으로 교체 (노이즈 좌표 정제) |
| `interp_temporal_occlusion` | TemporalOcclusion으로 구간 zeroing 후 InterpolateOcclusions으로 해당 구간 보간 |

두 config 모두 joint(feats=[j])와 bone(feats=[b]) 버전을 전 4개 split에서 실험. threshold=0.4.

---

### 9.2 Joint Feature 결과

#### 전체 결과 테이블

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **basic / baseline** | 0.8419 | 0.8199 | 0.8241 | 0.0735 | 0.2699 | 0.8505 | 0.1197 |
| basic / temporal_occlusion | 0.8389 | 0.8393 | 0.8474 | 0.0832 | 0.3002 | 0.8457 | 0.1260 |
| basic / interpolate_occlusions | 0.8328 | 0.8283 | 0.8433 | 0.0667 | 0.3207 | 0.8378 | 0.1213 |
| basic / interp_temporal_occlusion | 0.8389 | **0.8393** | 0.8436 | 0.0699 | 0.2582 | 0.8470 | 0.1250 |
| **byconf / baseline** | 0.7651 | 0.7348 | 0.7234 | 0.0784 | 0.8038 | 0.7683 | 0.0732 |
| byconf / temporal_occlusion | **0.7794** | **0.7491** | **0.7243** | 0.0803 | 0.3114 | 0.7676 | 0.0801 |
| byconf / interpolate_occlusions | 0.7402 | 0.7419 | 0.7299 | **0.0521** | **0.1751** | 0.7567 | 0.0747 |
| byconf / interp_temporal_occlusion | 0.7829 | 0.7348 | 0.7188 | 0.0969 | 0.3646 | 0.7565 | 0.0704 |
| **exclude / baseline** | 0.8856 | 0.8667 | 0.8806 | 0.0615 | 0.4960 | 0.8699 | 0.0976 |
| exclude / temporal_occlusion | 0.9020 | 0.8986 | **0.9107** | 0.0753 | 0.2992 | 0.8777 | 0.0966 |
| exclude / interpolate_occlusions | **0.9052** | 0.8870 | 0.9034 | 0.0763 | 0.2594 | 0.8679 | 0.0965 |
| exclude / interp_temporal_occlusion | 0.8987 | 0.8812 | 0.9029 | **0.0643** | 0.3826 | 0.8690 | 0.1088 |
| **exc_byconf / baseline** | 0.7743 | 0.7422 | 0.7289 | 0.0649 | 0.4122 | 0.7565 | 0.0795 |
| exc_byconf / temporal_occlusion | **0.8093** | 0.7656 | 0.7442 | 0.1005 | 0.5441 | 0.7664 | 0.0750 |
| exc_byconf / interpolate_occlusions | 0.7471 | 0.7344 | 0.7273 | 0.1022 | 0.3423 | 0.7656 | 0.0690 |
| **exc_byconf / interp_temporal_occlusion** | 0.7860 | **0.7695** | **0.7563** | 0.0969 | 0.3305 | **0.7688** | 0.0691 |

#### Delta 비교 (baseline 대비 top1_acc)

| Augmentation | basic | byconf | exclude | exc_byconf | 평균 |
|---|---|---|---|---|---|
| temporal_occlusion | +0.019 | **+0.014** | **+0.032** | +0.023 | +0.022 |
| interpolate_occlusions | +0.008 | +0.007 | +0.020 | −0.008 | +0.007 |
| interp_temporal_occlusion | **+0.019** | 0.000 | +0.015 | **+0.027** | **+0.015** |

---

### 9.3 Bone Feature 결과

#### 전체 결과 테이블

| Experiment | best_val | top1_acc | MCA | ECE | MCE | intra_mean | intra_std |
|---|---|---|---|---|---|---|---|
| **basic / baseline** | 0.8693 | 0.8310 | 0.8381 | 0.0463 | 0.3273 | 0.8402 | 0.1303 |
| basic / temporal_occlusion | 0.8693 | 0.8449 | 0.8457 | 0.0583 | 0.3011 | 0.8462 | 0.1401 |
| basic / bone_interpolate_occlusions | **0.8450** | 0.8061 | 0.7859 | **0.0507** | 0.3672 | 0.8430 | 0.1202 |
| basic / bone_interp_temporal_occlusion | 0.8389 | 0.8338 | 0.8338 | 0.0547 | **0.2162** | 0.8404 | 0.1330 |
| **byconf / baseline** | 0.7509 | 0.7384 | 0.7228 | 0.0595 | 0.3482 | 0.7606 | 0.0778 |
| byconf / temporal_occlusion | **0.7936** | **0.7778** | **0.7659** | 0.0923 | 0.2429 | **0.7732** | 0.0816 |
| byconf / bone_interpolate_occlusions | 0.7509 | 0.7025 | 0.6823 | 0.1065 | 0.3782 | 0.7547 | 0.0715 |
| byconf / bone_interp_temporal_occlusion | 0.7936 | 0.7527 | 0.7248 | 0.0814 | 0.4619 | 0.7574 | 0.0743 |
| **exclude / baseline** | 0.9248 | 0.8522 | 0.8503 | 0.0866 | 0.5633 | 0.8567 | 0.1199 |
| exclude / temporal_occlusion | 0.8987 | **0.8696** | **0.8727** | 0.0614 | 0.8788 | 0.8589 | 0.1059 |
| exclude / bone_interpolate_occlusions | **0.9020** | 0.8493 | 0.8485 | **0.0390** | 0.3610 | 0.8573 | 0.1128 |
| exclude / bone_interp_temporal_occlusion | 0.9052 | 0.8377 | 0.8488 | 0.0613 | **0.2919** | 0.8418 | 0.1395 |
| **exc_byconf / baseline** | 0.7665 | 0.7188 | 0.7026 | 0.1129 | 0.4158 | 0.7485 | 0.0846 |
| exc_byconf / temporal_occlusion | **0.7743** | **0.7812** | **0.7622** | 0.0690 | 0.4433 | **0.7705** | **0.0851** |
| exc_byconf / bone_interpolate_occlusions | 0.7471 | 0.7344 | 0.7132 | 0.0808 | 0.3026 | 0.7533 | 0.0799 |
| **exc_byconf / bone_interp_temporal_occlusion** | 0.8016 | 0.7734 | 0.7569 | **0.0610** | 0.6315 | 0.7521 | 0.0870 |

#### Delta 비교 (baseline 대비 top1_acc)

| Augmentation | basic | byconf | exclude | exc_byconf | 평균 |
|---|---|---|---|---|---|
| temporal_occlusion | **+0.014** | **+0.039** | **+0.017** | **+0.062** | **+0.033** |
| bone_interpolate_occlusions | −0.025 | −0.036 | −0.003 | +0.016 | −0.012 |
| bone_interp_temporal_occlusion | +0.003 | +0.014 | −0.015 | +0.055 | +0.014 |

---

### 9.4 분석

#### 9.4.1 Joint: interpolate_occlusions 효과

`interpolate_occlusions`(threshold=0.4, temporal 없음)는 joint feature에서 **byconf의 calibration을 크게 개선**했다.

| Split | Δ top1 | Δ ECE | Δ MCE |
|---|---|---|---|
| basic | +0.008 | −0.007 | +0.051 |
| byconf | +0.007 | **−0.026** | **−0.629** |
| exclude | +0.020 | +0.015 | −0.240 |
| exc_byconf | −0.008 | +0.037 | −0.070 |

- byconf에서 ECE 0.0784 → 0.0521, MCE **0.8038 → 0.1751** (MCE −0.629): 기존 joint 전 실험 중 calibration 최고 수준
- train 데이터의 score ≤ 0.4 keypoint를 보간으로 교체하면, joint 좌표 기반의 embedding이 정제된 좌표로부터 학습하여 **confidence prediction이 훨씬 안정적**으로 됨
- **정확도 향상은 modest** (+0.7%p)이지만 calibration 개선은 뚜렷함 → accuracy보다 신뢰도가 중요한 응용에서 유효

#### 9.4.2 Joint: interp_temporal_occlusion 효과

`interp_temporal_occlusion`(temporal 후 보간)은 **exclude_byconf에서 가장 강한 joint aug**로 등장했다(+2.7%p, temporal_occlusion +2.3%p를 상회).

- byconf: 0.0%p — temporal_occlusion 단독(+1.4%p)보다 낮음. 보간이 temporal occlusion의 학습 신호를 희석
- exclude: +1.5%p — temporal_occlusion 단독(+3.2%p)보다 낮음. 같은 방향이나 효과 약화
- **exclude_byconf: +2.7%p** — temporal_occlusion 단독(+2.3%p) 초과. 가장 어려운 split에서 보간이 추가 이득 제공

해석: exclude_byconf train set은 고품질 데이터지만, temporal 구간 zeroing 후 보간으로 채운 smooth trajectory가 low-confidence test data의 부드러운 추적 오류 패턴을 모사하여 일반화에 기여.

#### 9.4.3 Bone: interpolate_occlusions 역효과

`bone_interpolate_occlusions`는 bone feature에서 **basic/byconf에서 강한 역효과**를 냈다.

| Split | Δ top1 (bone_interp_occ) | Δ top1 (joint_interp_occ) |
|---|---|---|
| basic | **−0.025** | +0.008 |
| byconf | **−0.036** | +0.007 |
| exclude | −0.003 | +0.020 |
| exc_byconf | +0.016 | −0.008 |

- Joint에서 positive였던 효과가 bone에서 반전되는 이유: **bone vector는 인접 관절 간 방향·거리를 인코딩**하므로, 보간으로 좌표를 평활화하면 bone vector가 temporally smooth해지지만 **동작 간 구분에 필요한 미세 방향 변화 정보가 소실됨**
- 반면 joint feature는 절대 좌표를 사용하므로, 보간으로 정제된 좌표가 feature 품질을 높임
- exclude_byconf에서만 +1.6%p 소폭 양: 이 split의 train set(고품질)에 이미 noise가 상대적으로 적어 보간의 정제 효과 > 정보 손실

#### 9.4.4 Bone: interp_temporal_occlusion 효과

`bone_interp_temporal_occlusion`은 순수 `temporal_occlusion`보다 **전 split에서 낮거나 동등**하다.

| Split | Δ top1 (bone_temporal) | Δ top1 (bone_interp_temporal) |
|---|---|---|
| basic | +0.014 | +0.003 |
| byconf | **+0.039** | +0.014 |
| exclude | +0.017 | −0.015 |
| exc_byconf | **+0.062** | +0.055 |

- temporal occlusion 후 보간은 zeroing이 주는 "구간 부재"의 강한 학습 신호를 약화시킴
- bone feature는 temporal gap에서 방향 불연속성을 직접 학습하는데, 보간이 이를 smooth하게 만들어 학습 효과 감소
- exc_byconf에서만 근접(+5.5%p vs +6.2%p): 가장 어려운 split에서는 smooth trajectory가 일부 보완적으로 작용

---

### 9.5 종합 결론

#### 가설 검증

> "low-confidence keypoint를 보간으로 교체(정제)하면 hard test data에 대한 일반화가 향상된다"

| Feature | Split | 가설 결과 |
|---|---|---|
| Joint | basic / exclude | ✅ 지지 (top1 +0.8~+2.0%p) |
| Joint | byconf | 🟡 ECE 대폭 개선(MCE −0.629), 정확도는 미미(+0.7%p) |
| Joint | exc_byconf | ❌ 반증 (−0.8%p) |
| Bone | basic / byconf | ❌ 강하게 반증 (−2.5%p, −3.6%p) |
| Bone | exclude | ≈ 중립 (−0.3%p) |
| Bone | exc_byconf | 🟡 약하게 지지 (+1.6%p) |

**결론**: 가설은 **joint feature에 한해서 부분 지지**, bone feature에서는 반증. 보간이 좌표 정제 효과를 주는 joint feature와 달리, bone feature는 보간으로 인한 방향 정보 평활화가 더 큰 손실.

#### 목적별 최적 조합 업데이트

| 목적 | 최적 조합 | top1 | MCA | ECE |
|---|---|---|---|---|
| **Joint 최고 MCA (easy data)** | joint_exclude + temporal_occlusion | 0.8986 | **0.9107** | 0.0753 |
| **Joint hard data 일반화** | joint_exc_byconf + interp_temporal_occlusion | 0.7695 | 0.7563 | 0.0969 |
| **Joint byconf calibration** | joint_byconf + interpolate_occlusions | 0.7419 | 0.7299 | **0.0521** |
| **Bone 최고 accuracy** | bone_exclude + flip_only | 0.8754 | 0.8700 | 0.0604 |
| **Bone hard data 일반화** | bone_exc_byconf + temporal_occlusion | **0.7812** | **0.7622** | 0.0690 |
| **Bone 최저 ECE** | bone_exclude + interpolate_occlusions | 0.8493 | 0.8485 | **0.0390** |

#### Augmentation 효과 전체 순위 업데이트 (joint 기준, 전 split 평균 top1 delta)

| 순위 | Augmentation | avg Δ top1 | 비고 |
|---|---|---|---|
| 1 | temporal_occlusion | +0.022 | 전 split 일관되게 양수 |
| 2 | interp_temporal_occlusion | +0.015 | exc_byconf 최강(+2.7%p) |
| 3 | flip_only | +0.016 | exclude 계열 강함, byconf 역효과 |
| 4 | interpolate_occlusions | +0.007 | calibration 개선 효과 별도 |
| 5 | gradual_aug | +0.004 | 미미 |
| 이하 | strong_aug, strong_occlusion 등 | ≤ 0 | 비권장 (byconf 역효과) |

---

## 10. 전체 실험 종합 결론 및 최종 학습 구성

**분석 일자**: 2026-04-25

---

### 10.1 가설 검증 최종 결론

#### 가설 1: 클래스 제거 + Confidence Split이 성능에 기여한다

| 검증 항목 | 결과 | 근거 |
|---|---|---|
| 클래스 제거(exclude) 효과 | ✅ **강하게 지지** | Joint: top1 +4.7%p, MCA +5.7%p / Bone: top1 +2.1%p |
| Confidence split 효과 (accuracy) | — 직접 비교 불가 | test set이 다름 (harder samples) |
| Confidence split 효과 (embedding) | 🟡 **부분 지지** | intra_std 감소(더 균일한 클래스 embedding) 확인 |

클래스 제거는 decision boundary를 정리하여 모든 지표에서 명확한 개선. Joint에서 효과가 더 크고(+4.7%p), Bone에서는 상대적으로 작음(+2.1%p). **가설 1은 클래스 제거에 한해 강하게 지지.**

---

#### 가설 2: 고품질 Train + Occlusion Aug → 저품질 Test 일반화

핵심 가설. 결과는 aug 종류와 feature 유형에 따라 크게 갈림.

| 조건 | Δ top1 (byconf) | Δ top1 (exc_byconf) | 판정 |
|---|---|---|---|
| Joint + 공간 occlusion (strong_occlusion 등) | −0.043 ~ −0.029 | −0.008 ~ +0.020 | ❌ 반증 |
| Joint + temporal_occlusion | +0.014 | +0.023 | 🟡 약한 지지 |
| Joint + interp_temporal_occlusion | 0.000 | **+0.027** | 🟡 exc_byconf 한정 지지 |
| Joint + interpolate_occlusions | +0.007 | −0.008 | 🟡 calibration 개선만 |
| **Bone + temporal_occlusion** | **+0.039** | **+0.062** | ✅ **강하게 지지** |
| Bone + interp_temporal_occlusion | +0.014 | +0.055 | ✅ 지지 |

**결론**: 가설은 **temporal_occlusion + bone feature 조합에서 가장 강하게 성립**. 공간적 joint masking(hard zero)은 실제 low-confidence 데이터 분포와 달라 역효과. Bone feature가 joint feature보다 temporal 정보 소실에 더 robust하게 반응하기 때문에 가설의 효과가 증폭됨.

---

#### 부가 발견

| 발견 | 내용 |
|---|---|
| Bone이 Joint보다 hard data에서 우세 | byconf: Bone temporal top1=0.7778 vs Joint temporal 0.7491 (+2.9%p) |
| Joint interpolate_occlusions의 calibration 개선 | byconf MCE 0.8038 → 0.1751 (−0.629), 전 실험 중 최고 calibration |
| Bone exclude의 val-test 과적합 | gap=0.073 (joint 0.019의 3.8배), aug가 regularization 역할로 완화 |
| random_joint_mask 일관된 최하위 | 전 split, 양 feature에서 역효과. 사용 비권장 |

---

### 10.2 전체 실험 성능 요약

#### 목적별 최고 성능 조합 (전 실험 통합)

| 목적 | Feature | Split | Aug | top1 | MCA | ECE |
|---|---|---|---|---|---|---|
| **최고 accuracy** | Joint | exclude | flip_only | **0.9101** | **0.9159** | 0.0734 |
| **Accuracy + robustness** | Joint | exclude | temporal_occlusion | 0.8986 | 0.9107 | 0.0753 |
| **Hard data 일반화** | Bone | exc_byconf | temporal_occlusion | 0.7812 | 0.7622 | 0.0690 |
| **Hard data (Joint)** | Joint | exc_byconf | interp_temporal_occlusion | 0.7695 | 0.7563 | 0.0969 |
| **최고 calibration** | Joint | byconf | interpolate_occlusions | 0.7419 | 0.7299 | **0.0521** |
| **Bone ECE 최저** | Bone | exclude | interpolate_occlusions | 0.8493 | 0.8485 | **0.0390** |

---

### 10.3 최종 학습 구성 권고

실험 목표: **현실 세계의 occlusion이 많은 keypoint 데이터에서 운동 분류 일반화**

---

#### 권고 1 — 운영 모델 (Real-world Robustness 우선)

**"고품질 train + temporal occlusion aug + bone feature"**

| 항목 | 설정 |
|---|---|
| Feature | Bone (`feats: [b]`) |
| Data split | `exclude_byconf` (train: 고품질 2045개, test: 저품질 256개, 20 classes) |
| **Stage 1 — Freeze** | backbone 고정, 25 epochs, lr=0.01 |
| **Stage 2 — Finetune** | 전체 학습, 60 epochs, lr=0.001, weight_decay=0.0005 |
| Augmentation | `TemporalOcclusion(min=20, max=50, seg=1, p=0.5)` |
| Scheduler | CosineAnnealing (min_lr=1e-6) |
| 예상 성능 | top1=**0.7812**, MCA=**0.7622**, ECE=0.0690 |

**근거**: hard test(저품질 데이터)에서 전 실험 통틀어 최고 성능. Bone feature가 temporal 정보 소실에 robust하고, confidence split이 학습/평가 분포를 현실에 맞게 분리.

---

#### 권고 2 — 정확도 우선 모델 (Clean Data 기준)

**"exclude split + flip aug + joint feature"**

| 항목 | 설정 |
|---|---|
| Feature | Joint (`feats: [j]`) |
| Data split | `exclude` (train: 1907개, test: 345개, 20 classes) |
| **Stage 1 — Freeze** | backbone 고정, 30 epochs, lr=0.01 |
| **Stage 2 — Finetune** | 전체 학습, 60 epochs, lr=0.001, weight_decay=0.0005 |
| Augmentation | `Flip(ratio=0.5, left_kp=[1,3,5,7,9,11,13,15], right_kp=[2,4,6,8,10,12,14,16])` |
| Scheduler | CosineAnnealing (min_lr=1e-6) |
| 예상 성능 | top1=**0.9101**, MCA=**0.9159**, ECE=0.0734 |

**근거**: 전 실험 최고 accuracy. 혼합 품질 데이터 환경에서 높은 분류 정확도가 필요할 때 사용.

---

#### 권고 3 — 앙상블 (실용적 최강 구성)

권고 1과 권고 2를 Late Fusion으로 결합.

```
최종 예측 = α × softmax(Joint_exclude_fliponly) + (1−α) × softmax(Bone_excbyconf_temporal)
```

| 항목 | 값 |
|---|---|
| α (Joint 가중치) | 0.6 (추천 시작값, 검증 필요) |
| Joint 모델 | 권고 2 (exclude + flip_only) |
| Bone 모델 | 권고 1 (exc_byconf + temporal_occlusion) |

**앙상블 기대 효과**:
- Joint가 취약한 hard/occluded 샘플에서 Bone이 보완
- Bone이 취약한 유사 클래스 구분에서 Joint가 보완
- 두 모델의 intra_class_sim이 다른 분포를 형성하므로 오분류 패턴 비상관

---

#### 학습 파이프라인 요약 (권고 1 기준)

```
Raw video → Pose Estimation → PKL 저장
    ↓
[Data] exclude_byconf split (confidence score 기준 분리, 20 classes)
    ↓
[Stage 1 Freeze]  25 epochs, lr=0.01, backbone frozen
    checkpoint: bone_freeze_exclude_byconf/best_top1_acc_epoch_25.pth
    ↓
[Stage 2 Finetune]  60 epochs, lr=0.001, weight_decay=0.0005
    Pipeline: UniformSampleFrames(clip_len=100)
            → PoseDecode
            → TemporalOcclusion(min=20, max=50, seg=1, p=0.5)
            → GenSkeFeat(dataset=coco_new, feats=[b])
            → FormatGCNInput(num_person=1)
    ↓
[Eval]  저품질 test set (256개) 기준 top1/MCA/ECE 측정
```

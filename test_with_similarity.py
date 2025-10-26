#!/usr/bin/env python3
"""
Prototype 유사도와 함께 모델 테스트하는 스크립트
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json

# 프로젝트 path 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'external/ProtoGCN'))
sys.path.append(PROJECT_ROOT)

from protogcn.apis import init_recognizer
from similarity import PrototypeSimilarityCalculator
from protogcn.datasets import build_dataset


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
    """ProtoGCN 모델 로드"""
    print(f"Loading model from {checkpoint_path}")
    
    model = init_recognizer(
        config_path,
        checkpoint_path,
        device=device
    )
    model.eval()
    
    return model


def load_dataset(config_path: str, split: str = 'test'):
    """데이터셋 로드"""
    # config 파일 import
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace('.py', '')
    
    sys.path.insert(0, config_dir)
    config = __import__(config_name)
    
    # 데이터셋 설정
    if split == 'train':
        dataset_cfg = config.data['train']
    elif split == 'val':
        dataset_cfg = config.data['val'] 
    elif split == 'test':
        dataset_cfg = config.data['test']
    else:
        raise ValueError(f"Invalid split: {split}")
    
    dataset = build_dataset(dataset_cfg)
    
    return dataset


def test_with_similarity(model, dataset, calculator, device: str = 'cuda:0'):
    """유사도와 함께 테스트 수행"""
    
    # 결과 저장용
    results = {
        'samples': [],
        'accuracy': 0.0,
        'class_stats': {},
        'similarity_stats': {}
    }
    
    # 클래스별 통계
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    class_similarities = {i: [] for i in range(5)}
    
    total_correct = 0
    total_samples = 0
    
    print(f"Testing {len(dataset)} samples with similarity calculation")
    
    for idx in tqdm(range(len(dataset)), desc="Testing"):
        try:
            # 데이터 로드
            data = dataset[idx]
            keypoints = data['keypoint']  # (C, T, V, M)
            true_label = data['label']
            
            # ProtoGCN 입력 형식으로 변환
            if len(keypoints.shape) == 4:  # (C, T, V, M)
                keypoints = keypoints.permute(3, 0, 1, 2)  # (M, C, T, V)
                keypoints = keypoints[0]  # 첫 번째 person만 사용 (C, T, V)
                keypoints = keypoints.permute(1, 2, 0)  # (T, V, C)
                keypoints = keypoints.unsqueeze(0).unsqueeze(0)  # (1, 1, T, V, C)
            
            keypoints = torch.FloatTensor(keypoints).to(device)
            
            # 모델 예측
            with torch.no_grad():
                output = model(keypoints, return_loss=False)
                confidence_scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
                pred_label = confidence_scores.argmax().item()
                max_confidence = confidence_scores.max().item()
            
            # 정확도 계산
            is_correct = (pred_label == true_label)
            total_correct += is_correct
            total_samples += 1
            
            class_total[true_label] += 1
            if is_correct:
                class_correct[true_label] += 1
            
            # 유사도 계산 (true label과 비교)
            similarities = calculator.calculate_similarity(
                keypoints, 
                true_label,
                similarity_types=['prototype', 'global', 'joint', 'reconstruction']
            )
            overall_similarity = calculator.calculate_overall_similarity(keypoints, true_label)
            
            # 모든 클래스와의 유사도 계산
            all_class_similarities = calculator.calculate_all_class_similarities(keypoints)
            
            # 클래스별 유사도 통계 수집
            class_similarities[true_label].append(overall_similarity)
            
            # 샘플 결과 저장
            sample_result = {
                'sample_idx': idx,
                'true_label': true_label,
                'true_class': calculator.label_map[true_label],
                'pred_label': pred_label,
                'pred_class': calculator.label_map[pred_label],
                'is_correct': is_correct,
                'confidence': max_confidence,
                'similarities': similarities,
                'overall_similarity': overall_similarity,
                'all_class_similarities': all_class_similarities
            }
            
            results['samples'].append(sample_result)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 전체 정확도 계산
    results['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 클래스별 통계 계산
    for class_id in range(5):
        class_name = calculator.label_map[class_id]
        class_acc = class_correct[class_id] / class_total[class_id] if class_total[class_id] > 0 else 0.0
        avg_similarity = np.mean(class_similarities[class_id]) if class_similarities[class_id] else 0.0
        
        results['class_stats'][class_name] = {
            'accuracy': class_acc,
            'total_samples': class_total[class_id],
            'correct_samples': class_correct[class_id],
            'avg_similarity': avg_similarity,
            'similarity_std': np.std(class_similarities[class_id]) if class_similarities[class_id] else 0.0
        }
    
    # 유사도 전체 통계
    all_similarities = []
    correct_similarities = []
    incorrect_similarities = []
    
    for sample in results['samples']:
        all_similarities.append(sample['overall_similarity'])
        if sample['is_correct']:
            correct_similarities.append(sample['overall_similarity'])
        else:
            incorrect_similarities.append(sample['overall_similarity'])
    
    results['similarity_stats'] = {
        'overall': {
            'mean': np.mean(all_similarities),
            'std': np.std(all_similarities),
            'min': np.min(all_similarities),
            'max': np.max(all_similarities)
        },
        'correct_predictions': {
            'mean': np.mean(correct_similarities) if correct_similarities else 0.0,
            'std': np.std(correct_similarities) if correct_similarities else 0.0,
            'count': len(correct_similarities)
        },
        'incorrect_predictions': {
            'mean': np.mean(incorrect_similarities) if incorrect_similarities else 0.0,
            'std': np.std(incorrect_similarities) if incorrect_similarities else 0.0,
            'count': len(incorrect_similarities)
        }
    }
    
    return results


def print_results(results):
    """결과 출력"""
    print("\n" + "="*60)
    print("TEST RESULTS WITH PROTOTYPE SIMILARITY")
    print("="*60)
    
    # 전체 정확도
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    
    # 클래스별 결과
    print(f"\nClass-wise Results:")
    print(f"{'Class':<20} {'Accuracy':<10} {'Samples':<10} {'Avg Similarity':<15} {'Similarity Std':<15}")
    print("-" * 70)
    
    for class_name, stats in results['class_stats'].items():
        print(f"{class_name:<20} {stats['accuracy']:<10.4f} {stats['total_samples']:<10} "
              f"{stats['avg_similarity']:<15.4f} {stats['similarity_std']:<15.4f}")
    
    # 유사도 통계
    print(f"\nSimilarity Statistics:")
    sim_stats = results['similarity_stats']
    
    print(f"  Overall Similarity:")
    print(f"    Mean: {sim_stats['overall']['mean']:.4f} ± {sim_stats['overall']['std']:.4f}")
    print(f"    Range: [{sim_stats['overall']['min']:.4f}, {sim_stats['overall']['max']:.4f}]")
    
    print(f"  Correct Predictions (n={sim_stats['correct_predictions']['count']}):")
    print(f"    Mean: {sim_stats['correct_predictions']['mean']:.4f} ± {sim_stats['correct_predictions']['std']:.4f}")
    
    print(f"  Incorrect Predictions (n={sim_stats['incorrect_predictions']['count']}):")
    print(f"    Mean: {sim_stats['incorrect_predictions']['mean']:.4f} ± {sim_stats['incorrect_predictions']['std']:.4f}")
    
    # 유사도와 정확도의 상관관계 분석
    confidence_sim_corr = analyze_confidence_similarity_correlation(results)
    print(f"\nConfidence-Similarity Correlation: {confidence_sim_corr:.4f}")


def analyze_confidence_similarity_correlation(results):
    """신뢰도와 유사도의 상관관계 분석"""
    confidences = []
    similarities = []
    
    for sample in results['samples']:
        confidences.append(sample['confidence'])
        similarities.append(sample['overall_similarity'])
    
    # Pearson 상관계수 계산
    if len(confidences) > 1:
        corr = np.corrcoef(confidences, similarities)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='Test model with prototype similarity')
    parser.add_argument('--config',
                       default='configs/exercise/j.py',
                       help='ProtoGCN config file path')
    parser.add_argument('--checkpoint',
                       default='work_dirs/exercise/j_phase2_2/best_top1_acc_epoch_15.pth',
                       help='Model checkpoint path')
    parser.add_argument('--prototypes',
                       default='reference_prototypes.pkl',
                       help='Reference prototypes file path')
    parser.add_argument('--output',
                       help='Output JSON file path (optional)')
    parser.add_argument('--device',
                       default='cuda:0',
                       help='Device to use')
    parser.add_argument('--split',
                       default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to test')
    
    args = parser.parse_args()
    
    # 절대 경로 변환
    config_path = os.path.join(PROJECT_ROOT, args.config)
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    prototypes_path = os.path.join(PROJECT_ROOT, args.prototypes)
    
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prototypes: {prototypes_path}")
    print(f"Device: {args.device}")
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(prototypes_path):
        raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
    
    try:
        # 모델 로드
        model = load_model(config_path, checkpoint_path, args.device)
        
        # 유사도 계산기 로드
        calculator = PrototypeSimilarityCalculator(
            model, 
            reference_data_path=prototypes_path,
            device=args.device
        )
        
        # 데이터셋 로드
        dataset = load_dataset(config_path, args.split)
        
        # 테스트 수행
        results = test_with_similarity(model, dataset, calculator, args.device)
        
        # 결과 출력
        print_results(results)
        
        # JSON 파일로 저장 (선택사항)
        if args.output:
            output_path = os.path.join(PROJECT_ROOT, args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
            print(f"\nDetailed results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
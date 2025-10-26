#!/usr/bin/env python3
"""
학습 데이터에서 참조 prototype을 구축하는 스크립트
"""

import os
import sys
import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse

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


def load_dataset(config_path: str, split: str = 'train'):
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


def build_prototypes(model, dataset, calculator, max_samples_per_class: int = 100):
    """데이터셋에서 참조 prototype 구축"""
    print(f"Building reference prototypes from {len(dataset)} samples")
    
    # 클래스별 샘플 카운터
    class_counts = {i: 0 for i in range(5)}
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        try:
            # 데이터 로드
            data = dataset[idx]
            keypoints = data['keypoint']  # (C, T, V, M)
            label = data['label']
            
            # 클래스별 최대 샘플 수 확인
            if class_counts[label] >= max_samples_per_class:
                continue
            
            # ProtoGCN 입력 형식으로 변환: (1, M, C, T, V) -> (1, 1, T, V, C)
            if len(keypoints.shape) == 4:  # (C, T, V, M)
                keypoints = keypoints.permute(3, 0, 1, 2)  # (M, C, T, V)
                keypoints = keypoints[0]  # 첫 번째 person만 사용 (C, T, V)
                keypoints = keypoints.permute(1, 2, 0)  # (T, V, C)
                keypoints = keypoints.unsqueeze(0).unsqueeze(0)  # (1, 1, T, V, C)
            
            keypoints = torch.FloatTensor(keypoints)
            
            # 참조 샘플 추가
            calculator.add_reference_sample(keypoints, label)
            class_counts[label] += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 통계 출력
    print("\nReference samples collected:")
    for class_id, count in class_counts.items():
        class_name = calculator.label_map[class_id]
        print(f"  {class_name}: {count} samples")
    
    # Centroid 계산
    print("\nComputing prototype centroids...")
    calculator.compute_prototype_centroids()
    
    return calculator


def main():
    parser = argparse.ArgumentParser(description='Build reference prototypes')
    parser.add_argument('--config', 
                       default='configs/exercise/j.py',
                       help='ProtoGCN config file path')
    parser.add_argument('--checkpoint',
                       default='work_dirs/exercise/j_phase2_2/best_top1_acc_epoch_15.pth', 
                       help='Model checkpoint path')
    parser.add_argument('--output',
                       default='reference_prototypes.pkl',
                       help='Output path for reference prototypes')
    parser.add_argument('--device',
                       default='cuda:0',
                       help='Device to use')
    parser.add_argument('--max-samples',
                       type=int,
                       default=100,
                       help='Maximum samples per class')
    parser.add_argument('--split',
                       default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    # 절대 경로 변환
    config_path = os.path.join(PROJECT_ROOT, args.config)
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    output_path = os.path.join(PROJECT_ROOT, args.output)
    
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        # 모델 로드
        model = load_model(config_path, checkpoint_path, args.device)
        
        # 유사도 계산기 초기화
        calculator = PrototypeSimilarityCalculator(model, device=args.device)
        
        # 데이터셋 로드
        dataset = load_dataset(config_path, args.split)
        
        # Prototype 구축
        calculator = build_prototypes(model, dataset, calculator, args.max_samples)
        
        # 저장
        calculator.save_reference_prototypes(output_path)
        
        # 통계 출력
        stats = calculator.get_similarity_stats()
        print("\nFinal statistics:")
        for class_name, stat in stats.items():
            print(f"  {class_name}:")
            print(f"    Prototype samples: {stat['prototype_samples']}")
            print(f"    Has centroids: {stat['has_centroids']}")
        
        print(f"\nReference prototypes saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
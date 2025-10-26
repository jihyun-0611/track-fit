import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from .feature_extractor import ProtoGCNFeatureExtractor


class PrototypeSimilarityCalculator:
    """ProtoGCN 기반 Prototype 유사도 계산기"""
    
    def __init__(self, 
                 model,
                 reference_data_path: Optional[str] = None,
                 device: str = 'cuda:0'):
        """
        Args:
            model: 학습된 ProtoGCN 모델
            reference_data_path: 참조 prototype 데이터 경로 (없으면 실시간 생성)
            device: 계산 디바이스
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = ProtoGCNFeatureExtractor(model)
        
        # 클래스 라벨 매핑
        self.label_map = {
            0: "barbell biceps curl",
            1: "bench press", 
            2: "lat pulldown",
            3: "push-up",
            4: "tricep Pushdown"
        }
        
        # Reference features 저장소 (개선된 구조)
        self.reference_features = {
            'prototype_activations': {},     # PRN query (100차원)
            'joint_importance': {},          # 관절 중요도 (20차원)  
            'temporal_dynamics': {},         # 시간적 동역학 (50차원)
        }
        
        # Centroids (대표값들)
        self.centroids = {}
        
        # Quality scores for robust centroid calculation
        self.sample_qualities = {}
        
        # Reference 데이터 로드 또는 초기화
        if reference_data_path and os.path.exists(reference_data_path):
            self.load_reference_prototypes(reference_data_path)
        else:
            self._initialize_empty_references()
    
    def _initialize_empty_references(self):
        """빈 참조 데이터 초기화"""
        for class_id in self.label_map.keys():
            for feature_type in self.reference_features:
                self.reference_features[feature_type][class_id] = []
            self.sample_qualities[class_id] = []
    
    def add_reference_sample(self, keypoints: torch.Tensor, class_id: int):
        """
        참조 샘플 추가 (개선된 특징들과 품질 평가 포함)
        
        Args:
            keypoints: 키포인트 텐서 (1, 1, T, V, C)
            class_id: 클래스 ID (0-4)
        """
        if class_id not in self.label_map:
            raise ValueError(f"Invalid class_id: {class_id}")
        
        keypoints = keypoints.to(self.device)
        
        # 다양한 특징 추출
        proto_activation = self.feature_extractor.extract_prototype_activation(keypoints)
        joint_importance = self.feature_extractor.extract_joint_importance(keypoints)
        temporal_dynamics = self.feature_extractor.extract_temporal_dynamics(keypoints)
        
        # 샘플 품질 평가 (모델의 confidence score 활용)
        with torch.no_grad():
            output = self.feature_extractor.model(keypoints, return_loss=False)
            confidence_scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
            sample_quality = confidence_scores[class_id].item()  # 해당 클래스에 대한 confidence
        
        # 고품질 샘플만 저장 (confidence > 0.7)
        if sample_quality > 0.7:
            # CPU로 이동하여 저장
            self.reference_features['prototype_activations'][class_id].append(proto_activation.cpu())
            self.reference_features['joint_importance'][class_id].append(joint_importance.cpu())
            self.reference_features['temporal_dynamics'][class_id].append(temporal_dynamics.cpu())
            self.sample_qualities[class_id].append(sample_quality)
    
    def compute_prototype_centroids(self):
        """개선된 centroid 계산 (가중 평균과 robust 통계 사용)"""
        self.centroids = {}
        
        for class_id in self.label_map.keys():
            self.centroids[class_id] = {}
            
            # 각 특징 타입별로 centroid 계산
            for feature_type in self.reference_features:
                features = self.reference_features[feature_type][class_id]
                qualities = self.sample_qualities[class_id]
                
                if features and len(features) == len(qualities):
                    features_tensor = torch.stack(features)
                    qualities_tensor = torch.tensor(qualities)
                    
                    # 방법 1: 가중 평균 (품질 점수 기반)
                    weights = torch.softmax(qualities_tensor * 2, dim=0)  # 온도 스케일링
                    weighted_centroid = (features_tensor * weights.unsqueeze(1)).sum(dim=0)
                    
                    # 방법 2: Robust centroid (이상치 제거)
                    if len(features) >= 5:
                        # 품질 상위 80% 샘플만 사용
                        top_k = max(1, int(len(features) * 0.8))
                        top_indices = torch.topk(qualities_tensor, top_k).indices
                        robust_features = features_tensor[top_indices]
                        robust_centroid = robust_features.mean(dim=0)
                    else:
                        robust_centroid = weighted_centroid
                    
                    # 두 방법의 조합 (가중 평균 70%, robust 30%)
                    final_centroid = 0.7 * weighted_centroid + 0.3 * robust_centroid
                    
                    self.centroids[class_id][feature_type] = final_centroid
                else:
                    # 기본값
                    if feature_type == 'prototype_activations':
                        self.centroids[class_id][feature_type] = torch.zeros(100)
                    elif feature_type == 'joint_importance':
                        self.centroids[class_id][feature_type] = torch.zeros(20)
                    elif feature_type == 'temporal_dynamics':
                        self.centroids[class_id][feature_type] = torch.zeros(50)
    
    def calculate_similarity(self, 
                           keypoints: torch.Tensor,
                           target_class_id: int,
                           similarity_types: List[str] = None) -> Dict[str, float]:
        """
        개선된 다차원 유사도 계산
        
        Args:
            keypoints: 입력 키포인트 텐서
            target_class_id: 비교할 타겟 클래스 ID
            similarity_types: 계산할 유사도 타입들
            
        Returns:
            dict: 각 타입별 유사도 점수 (0-1)
        """
        if similarity_types is None:
            similarity_types = ['prototype_activations', 'joint_importance', 'temporal_dynamics']
            
        if target_class_id not in self.label_map:
            raise ValueError(f"Invalid target_class_id: {target_class_id}")
        
        keypoints = keypoints.to(self.device)
        similarities = {}
        
        # 현재 입력의 특징들 추출
        current_features = {}
        if 'prototype_activations' in similarity_types:
            current_features['prototype_activations'] = self.feature_extractor.extract_prototype_activation(keypoints)
        if 'joint_importance' in similarity_types:
            current_features['joint_importance'] = self.feature_extractor.extract_joint_importance(keypoints)
        if 'temporal_dynamics' in similarity_types:
            current_features['temporal_dynamics'] = self.feature_extractor.extract_temporal_dynamics(keypoints)
        
        # 각 특징별 유사도 계산
        for feature_type in similarity_types:
            if (feature_type in current_features and 
                target_class_id in self.centroids and 
                feature_type in self.centroids[target_class_id]):
                
                current_feat = current_features[feature_type].to(self.device)
                centroid_feat = self.centroids[target_class_id][feature_type].to(self.device)
                
                # 다양한 거리 측정법 조합
                cosine_sim = F.cosine_similarity(current_feat.unsqueeze(0), centroid_feat.unsqueeze(0), dim=1).item()
                
                # L2 거리 기반 유사도 (정규화된)
                l2_dist = torch.norm(current_feat - centroid_feat, p=2).item()
                l2_sim = 1.0 / (1.0 + l2_dist)
                
                # 두 유사도의 가중 평균
                combined_sim = 0.7 * max(0.0, cosine_sim) + 0.3 * l2_sim
                
                similarities[feature_type] = combined_sim
        
        # Reconstruction similarity (선택적)
        if 'reconstruction' in similarity_types:
            recon_sim = self.feature_extractor.compute_reconstruction_error(keypoints)
            similarities['reconstruction'] = recon_sim
        
        return similarities
    
    def calculate_overall_similarity(self, 
                                   keypoints: torch.Tensor,
                                   target_class_id: int,
                                   weights: Dict[str, float] = None) -> float:
        """
        개선된 종합 유사도 계산
        
        Args:
            keypoints: 입력 키포인트 텐서
            target_class_id: 타겟 클래스 ID
            weights: 각 유사도 타입별 가중치
            
        Returns:
            float: 종합 유사도 (0-1)
        """
        if weights is None:
            weights = {
                'prototype_activations': 0.5,  # 가장 중요 (순수 prototype 패턴)
                'joint_importance': 0.3,       # 관절별 중요도
                'temporal_dynamics': 0.2       # 시간적 동역학
            }
        
        similarities = self.calculate_similarity(
            keypoints, target_class_id, list(weights.keys())
        )
        
        # 적응적 가중치 (특징의 신뢰도에 따라 조정)
        adjusted_weights = {}
        total_confidence = 0
        
        for sim_type, weight in weights.items():
            if sim_type in similarities:
                # 유사도가 높을수록 해당 특징의 신뢰도도 높다고 가정
                confidence = similarities[sim_type] ** 0.5  # 제곱근으로 완화
                adjusted_weights[sim_type] = weight * (1 + confidence)
                total_confidence += adjusted_weights[sim_type]
        
        # 정규화된 가중 평균
        if total_confidence > 0:
            weighted_sum = sum(similarities[sim_type] * adjusted_weights[sim_type] 
                             for sim_type in adjusted_weights)
            return weighted_sum / total_confidence
        else:
            return 0.0
    
    def calculate_all_class_similarities(self, keypoints: torch.Tensor) -> Dict[str, float]:
        """
        모든 클래스에 대한 유사도 계산
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            dict: 클래스명별 유사도
        """
        similarities = {}
        
        for class_id, class_name in self.label_map.items():
            sim = self.calculate_overall_similarity(keypoints, class_id)
            similarities[class_name] = sim
        
        return similarities
    
    def save_reference_prototypes(self, save_path: str):
        """개선된 참조 데이터 저장"""
        reference_data = {
            'reference_features': self.reference_features,
            'sample_qualities': self.sample_qualities,
            'centroids': getattr(self, 'centroids', {}),
            'label_map': self.label_map,
            'version': '2.0'  # 버전 정보 추가
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(reference_data, f)
        
        print(f"Enhanced reference prototypes saved to {save_path}")
        
        # 통계 정보 출력
        total_samples = sum(len(qualities) for qualities in self.sample_qualities.values())
        print(f"Total high-quality samples: {total_samples}")
        for class_id, class_name in self.label_map.items():
            count = len(self.sample_qualities.get(class_id, []))
            avg_quality = np.mean(self.sample_qualities.get(class_id, [0])) if count > 0 else 0
            print(f"  {class_name}: {count} samples (avg quality: {avg_quality:.3f})")
    
    def load_reference_prototypes(self, load_path: str):
        """개선된 참조 데이터 로드 (하위 호환성 포함)"""
        with open(load_path, 'rb') as f:
            reference_data = pickle.load(f)
        
        version = reference_data.get('version', '1.0')
        
        if version == '2.0':
            # 새로운 형식
            self.reference_features = reference_data.get('reference_features', {})
            self.sample_qualities = reference_data.get('sample_qualities', {})
            self.centroids = reference_data.get('centroids', {})
        else:
            # 이전 형식 (하위 호환성)
            print("Loading legacy format, converting to new structure...")
            self._convert_legacy_format(reference_data)
        
        print(f"Reference prototypes loaded from {load_path} (version: {version})")
    
    def _convert_legacy_format(self, legacy_data):
        """이전 형식을 새 형식으로 변환"""
        # 빈 구조 초기화
        self._initialize_empty_references()
        
        # 이전 centroids를 새 형식으로 변환
        self.centroids = {}
        for class_id in self.label_map.keys():
            self.centroids[class_id] = {}
            
            if 'prototype_centroids' in legacy_data and class_id in legacy_data['prototype_centroids']:
                self.centroids[class_id]['prototype_activations'] = legacy_data['prototype_centroids'][class_id]
            
            if 'joint_centroids' in legacy_data and class_id in legacy_data['joint_centroids']:
                self.centroids[class_id]['joint_importance'] = legacy_data['joint_centroids'][class_id]
    
    def get_similarity_stats(self) -> Dict[str, any]:
        """개선된 참조 데이터 통계 정보 반환"""
        stats = {}
        
        for class_id, class_name in self.label_map.items():
            sample_count = len(self.sample_qualities.get(class_id, []))
            avg_quality = np.mean(self.sample_qualities.get(class_id, [0])) if sample_count > 0 else 0
            
            stats[class_name] = {
                'total_samples': sample_count,
                'average_quality': avg_quality,
                'quality_std': np.std(self.sample_qualities.get(class_id, [0])) if sample_count > 1 else 0,
                'feature_types': {},
                'has_centroids': class_id in getattr(self, 'centroids', {})
            }
            
            # 각 특징 타입별 샘플 수
            for feature_type in self.reference_features:
                feature_samples = len(self.reference_features[feature_type].get(class_id, []))
                stats[class_name]['feature_types'][feature_type] = feature_samples
        
        # 전체 통계
        stats['overall'] = {
            'total_classes': len(self.label_map),
            'total_samples': sum(len(qualities) for qualities in self.sample_qualities.values()),
            'avg_samples_per_class': np.mean([len(qualities) for qualities in self.sample_qualities.values()]),
            'feature_dimensions': {
                'prototype_activations': 100,
                'joint_importance': 20,
                'temporal_dynamics': 50
            }
        }
        
        return stats
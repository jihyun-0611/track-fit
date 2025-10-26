import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# ProtoGCN 모듈 path 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'external/ProtoGCN'))

from protogcn.core import OutputHook


class ProtoGCNFeatureExtractor:
    """ProtoGCN 모델에서 중간 특징과 prototype 정보를 추출하는 클래스"""
    
    def __init__(self, model):
        """
        Args:
            model: 학습된 ProtoGCN 모델
        """
        self.model = model
        self.model.eval()
        
        # 추출할 레이어 정의
        self.target_layers = [
            'backbone.gcn.0',  # 첫 번째 GCN 블록
            'backbone.gcn.1',  # 두 번째 GCN 블록  
            'backbone.gcn.2',  # 세 번째 GCN 블록
        ]
        
        # PRN query 저장용
        self.prn_query = None
        self._register_prn_hooks()
    
    def _register_prn_hooks(self):
        """PRN의 query 추출을 위한 hook 등록"""
        def prn_hook(module, input, output):
            # PRN forward에서 query 추출
            x = input[0]
            query = torch.softmax(module.query_matrix(x), dim=-1)
            self.prn_query = query.detach()
        
        # PRN 모듈 찾아서 hook 등록
        for name, module in self.model.named_modules():
            if 'prn' in name and hasattr(module, 'query_matrix'):
                module.register_forward_hook(prn_hook)
                break
    
    def extract_features(self, keypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        키포인트에서 다층 특징을 추출
        
        Args:
            keypoints: 입력 키포인트 텐서 (1, 1, T, V, C)
            
        Returns:
            dict: 각 레이어별 특징과 최종 분류 점수
        """
        with torch.no_grad():
            with OutputHook(self.model, outputs=self.target_layers, as_tensor=True) as hook:
                # 모델 추론 실행
                output = self.model(keypoints, return_loss=False)
                
                # 분류 점수
                cls_scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
                
                # 중간 레이어 출력 추출
                features = {
                    'cls_scores': cls_scores,
                    'gcn_features': [],
                    'prototype_features': None
                }
                
                # GCN 블록 특징들 수집
                for i in range(3):  # gcn.0, gcn.1, gcn.2
                    layer_name = f'backbone.gcn.{i}'
                    if layer_name in hook.layer_outputs:
                        gcn_output = hook.layer_outputs[layer_name]
                        if isinstance(gcn_output, tuple):
                            # GCN 출력은 (features, graph) 튜플
                            gcn_features = gcn_output[0]  # 특징만 사용
                        else:
                            gcn_features = gcn_output
                        features['gcn_features'].append(gcn_features)
                
                # Prototype features 추출
                if 'backbone.prn' in hook.layer_outputs:
                    features['prototype_features'] = hook.layer_outputs['backbone.prn']
                
                return features
    
    def extract_global_features(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        전역적인 시퀀스 특징 추출 (분류 직전 특징)
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            torch.Tensor: 전역 특징 벡터
        """
        features = self.extract_features(keypoints)
        
        # 마지막 GCN 블록의 특징을 전역 풀링
        if features['gcn_features']:
            last_gcn_features = features['gcn_features'][-1]  # (N, M, C, T, V)
            
            # 시간과 관절 차원을 평균 풀링
            global_features = last_gcn_features.mean(dim=(-2, -1))  # (N, M, C)
            global_features = global_features.flatten(start_dim=1)  # (N, M*C)
            
            return global_features
        
        return None
    
    def extract_prototype_activation(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        순수한 Prototype activation pattern 추출 (PRN의 query)
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            torch.Tensor: Prototype 활성화 패턴 (n_prototype,)
        """
        with torch.no_grad():
            # 모델 forward 실행하여 PRN hook 트리거
            _ = self.model(keypoints, return_loss=False)
            
            if self.prn_query is not None:
                # PRN의 query는 이미 softmax로 정규화됨
                return self.prn_query.flatten()  # (n_prototype,) 예: 100차원
        
        return torch.zeros(100)  # 기본값: 100차원 prototype
    
    def extract_joint_importance(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        관절별 중요도 추출 (개선된 방법)
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            torch.Tensor: 관절별 중요도 점수 (V,)
        """
        features = self.extract_features(keypoints)
        
        if features['gcn_features']:
            # 가중 평균으로 관절 중요도 계산 (깊은 층일수록 높은 가중치)
            joint_importances = []
            layer_weights = [0.2, 0.3, 0.5]  # 층별 가중치 (깊을수록 높음)
            
            for i, gcn_feat in enumerate(features['gcn_features']):
                # (N, M, C, T, V) -> (V,) 관절별 가중 활성화
                # 시간 차원에서도 가중 평균 (중간 프레임에 더 높은 가중치)
                T = gcn_feat.shape[-2]
                time_weights = torch.linspace(0.5, 1.0, T).to(gcn_feat.device)
                time_weights = time_weights / time_weights.sum()
                
                # 가중 시간 평균
                weighted_temporal = (gcn_feat * time_weights.view(1, 1, 1, -1, 1)).sum(dim=-2)
                
                # 관절별 중요도 (RMS 활용)
                joint_importance = torch.sqrt((weighted_temporal ** 2).mean(dim=(0, 1, 2)))  # (V,)
                joint_importances.append(joint_importance * layer_weights[i])
            
            # 층별 가중 합산
            if joint_importances:
                total_joint_importance = torch.stack(joint_importances, dim=0).sum(dim=0)
                # L2 정규화
                total_joint_importance = torch.nn.functional.normalize(total_joint_importance, p=2, dim=0)
                return total_joint_importance
        
        return torch.zeros(20)  # 기본값: 20개 관절
    
    def extract_temporal_dynamics(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        시간적 동역학 특징 추출 (새로운 특징)
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            torch.Tensor: 시간적 동역학 특징 (compact representation)
        """
        features = self.extract_features(keypoints)
        
        if features['gcn_features']:
            # 마지막 GCN 특징에서 시간적 패턴 추출
            last_feat = features['gcn_features'][-1]  # (N, M, C, T, V)
            
            # 관절별 시간적 변화율 계산
            temporal_diff = torch.diff(last_feat, dim=-2)  # (N, M, C, T-1, V)
            
            # 각 관절의 동역학 특성 요약
            dynamics_per_joint = []
            for v in range(last_feat.shape[-1]):  # 각 관절별로
                joint_feat = last_feat[..., v]  # (N, M, C, T)
                joint_diff = temporal_diff[..., v]  # (N, M, C, T-1)
                
                # 통계적 특성들
                mean_activity = joint_feat.mean(dim=-1)  # (N, M, C)
                velocity_std = joint_diff.std(dim=-1)    # (N, M, C)
                max_acceleration = torch.diff(joint_diff, dim=-1).abs().max(dim=-1)[0]  # (N, M, C)
                
                # 결합
                joint_dynamics = torch.cat([
                    mean_activity.flatten(),
                    velocity_std.flatten(), 
                    max_acceleration.flatten()
                ], dim=0)
                
                dynamics_per_joint.append(joint_dynamics)
            
            # 모든 관절의 동역학을 결합하고 압축
            all_dynamics = torch.stack(dynamics_per_joint, dim=0)  # (V, features)
            compressed = all_dynamics.mean(dim=0)  # 관절별 평균
            
            return compressed[:50]  # 50차원으로 압축
        
        return torch.zeros(50)
    
    def compute_reconstruction_error(self, keypoints: torch.Tensor) -> float:
        """
        재구성 오차 계산 (유사도의 역지표)
        
        Args:
            keypoints: 입력 키포인트 텐서
            
        Returns:
            float: 재구성 오차 (낮을수록 높은 유사도)
        """
        with torch.no_grad():
            # 모델 forward를 통해 재구성 과정 시뮬레이션
            try:
                # backbone의 extract_feat 메서드가 있는지 확인
                if hasattr(self.model.backbone, 'extract_feat'):
                    x, reconstructed_graph = self.model.backbone.extract_feat(keypoints)
                    
                    # 원본과 재구성된 그래프 간의 MSE
                    if reconstructed_graph is not None:
                        # 재구성 오차를 유사도로 변환 (0~1)
                        mse_error = torch.nn.functional.mse_loss(
                            x.flatten(), reconstructed_graph.flatten()
                        ).item()
                        
                        # 오차를 유사도로 변환 (exponential decay)
                        similarity = np.exp(-mse_error)
                        return similarity
                
                # fallback: 단순히 모델 confidence 사용
                output = self.model(keypoints, return_loss=False)
                confidence = torch.softmax(torch.FloatTensor(output[0]), dim=0).max().item()
                return confidence
                
            except Exception as e:
                print(f"Error in reconstruction: {e}")
                return 0.0
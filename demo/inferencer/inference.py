import sys
from pathlib import Path
import torch
import numpy as np
from collections import deque
from protogcn.apis import init_recognizer

# Add project root to path for quality_assessment import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quality_assessment import QualityAssessment


class ProtoGCNInference:
    def __init__(self, config_path, checkpoint_path, device='cuda:0', enable_quality_assessment=True):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.buffer = deque(maxlen=300)  # 최대 300 프레임 버퍼 (더 긴 시퀀스 저장)
        self.label_map={
            0: "barbell biceps curl",
            1: "bench press",
            2: "lat pulldown",
            3: "push-up",
            4: "tricep Pushdown"
        }
        # Reverse mapping for exercise name -> class index
        self.label_map_reverse = {v: k for k, v in self.label_map.items()}
        self.enable_quality_assessment = enable_quality_assessment
        self._init_model()

        # Initialize quality assessment
        if self.enable_quality_assessment:
            self.quality_assessor = QualityAssessment(self.model, top_k=5)
        else:
            self.quality_assessor = None
    
    def _init_model(self):
        self.model = init_recognizer(
            self.config_path,
            self.checkpoint_path,
            device=self.device
        )
        self.model.eval()

    def add_frame(self, keypoints):
        self.buffer.append(keypoints)
    
    def reset_buffer(self):
        self.buffer.clear()
    
    def predict(self):
        if len(self.buffer) < 100:
            return None
        
        # (T, V, C) -> (1, 1, T, V, C)
        keypoints = np.array(self.buffer)  # (T, V, C)
        keypoints = keypoints[np.newaxis, :, :, :]
        keypoints = keypoints.transpose(1, 0, 2, 3)
        keypoints = keypoints[np.newaxis, np.newaxis, :, :, :, :]  # (1, 1, T, V, C)

        with torch.no_grad():
            keypoint_tensor = torch.FloatTensor(keypoints).to(self.device)
            output = self.model(keypoint_tensor, return_loss=False)

            scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
            pred_idx = scores.argmax().item()
            confidence = scores[pred_idx].item()

        return {
            "class": self.label_map[pred_idx],
            "confidence": float(confidence),
            "all_scores": {
                self.label_map[i]: float(scores[i])
                for i in range(len(self.label_map))
            }
        }
    
    def predict_sliding_window(self, selected_exercise=None):
        """슬라이딩 윈도우 방식으로 예측 및 품질 평가 수행

        Args:
            selected_exercise: User selected exercise name for quality assessment
        """
        if len(self.buffer) < 60:
            return None

        # 최근 100프레임 사용 (부족한 경우 반복 패딩)
        recent_frames = list(self.buffer)
        if len(recent_frames) < 100:
            # 패딩: 전체 시퀀스를 반복하여 100프레임으로 만들기
            repeat_count = (100 // len(recent_frames)) + 1
            padded_frames = (recent_frames * repeat_count)[:100]
        else:
            # 최근 100프레임 사용
            padded_frames = recent_frames[-100:]

        # (T, V, C) -> (N, num_clips, M, T, V, C)
        keypoints = np.array(padded_frames)  # (100, 20, 3) = (T, V, C)
        keypoints = keypoints[np.newaxis, :, :, :]  # (1, 100, 20, 3) = (M, T, V, C)
        keypoints = keypoints[np.newaxis, :, :, :, :]  # (1, 1, 100, 20, 3) = (num_clips, M, T, V, C)
        keypoints = keypoints[np.newaxis, :, :, :, :, :]  # (1, 1, 1, 100, 20, 3) = (N, num_clips, M, T, V, C)

        with torch.no_grad():
            keypoint_tensor = torch.FloatTensor(keypoints).to(self.device)
            output = self.model(keypoint_tensor, return_loss=False)

            scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
            pred_idx = scores.argmax().item()
            confidence = scores[pred_idx].item()

        # 신뢰도가 낮으면 결과 반환하지 않음
        if confidence < 0.1:  # 임계값을 0.1로 낮춤
            return None

        result = {
            "class": self.label_map[pred_idx],
            "confidence": float(confidence),
            "all_scores": {
                self.label_map[i]: float(scores[i])
                for i in range(len(self.label_map))
            }
        }

        # Perform quality assessment if enabled
        if self.enable_quality_assessment and self.quality_assessor is not None:
            # Use selected exercise class if provided, otherwise use predicted class
            quality_class_idx = pred_idx  # Default to predicted class
            if selected_exercise and selected_exercise in self.label_map_reverse:
                quality_class_idx = self.label_map_reverse[selected_exercise]

            quality_metrics = self.quality_assessor.assess_quality(
                keypoint_tensor,
                compute_joint_scores=True,
                predicted_class=quality_class_idx  # Use selected or predicted class
            )
            interpretation = self.quality_assessor.get_quality_interpretation(
                quality_metrics['global_quality']
            )

            result['quality'] = {
                'global_score': quality_metrics['global_quality'],
                'mean_top_k_response': quality_metrics['mean_top_k_response'],
                'std_top_k_response': quality_metrics['std_top_k_response'],
                'level': interpretation['level'],
                'color': interpretation['color'],
                'message': interpretation['message']
            }

            # Add joint-wise quality scores
            if 'joint_wise' in quality_metrics:
                joint_metrics = quality_metrics['joint_wise']
                result['quality']['joint_wise'] = {
                    'joint_scores': joint_metrics['joint_scores'].tolist(),
                    'mean_joint_quality': joint_metrics['mean_joint_quality'],
                    'std_joint_quality': joint_metrics['std_joint_quality'],
                    'min_joint_quality': joint_metrics['min_joint_quality'],
                    'max_joint_quality': joint_metrics['max_joint_quality'],
                    'weak_joints': joint_metrics['weak_joints'],
                    'num_weak_joints': joint_metrics['num_weak_joints']
                }

        return result
    
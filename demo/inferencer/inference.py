import torch
import numpy as np
from collections import deque
from protogcn.apis import init_recognizer


class ProtoGCNInference:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
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
        self._init_model()
    
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
    
    def predict_sliding_window(self):
        """슬라이딩 윈도우 방식으로 예측 수행"""
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
        
        # (T, V, C) -> (1, 1, T, V, C)
        keypoints = np.array(padded_frames)  # (100, 20, 3)
        keypoints = keypoints[np.newaxis, :, :, :]
        keypoints = keypoints.transpose(1, 0, 2, 3)
        keypoints = keypoints[np.newaxis, np.newaxis, :, :, :, :]  # (1, 1, 100, 20, 3)

        with torch.no_grad():
            keypoint_tensor = torch.FloatTensor(keypoints).to(self.device)
            output = self.model(keypoint_tensor, return_loss=False)

            scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
            pred_idx = scores.argmax().item()
            confidence = scores[pred_idx].item()

        # 신뢰도가 낮으면 결과 반환하지 않음
        if confidence < 0.1:  # 임계값을 0.1로 낮춤
            return None

        return {
            "class": self.label_map[pred_idx],
            "confidence": float(confidence),
            "all_scores": {
                self.label_map[i]: float(scores[i])
                for i in range(len(self.label_map))
            }
        }
    
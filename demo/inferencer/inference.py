from config import PROTOGCN_DIR

import torch
import numpy as np
from collections import deque
from protogcn.apis import init_recognizer


class ProtoGCNInference:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.buffer = deque(maxlen=100)  # 최대 100 프레임 버퍼
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
        keypoints = np.array(self.buffer)
        keypoints = keypoints[np.newaxis, np.newaxis, :, :, :]

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
    
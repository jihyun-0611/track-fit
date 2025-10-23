from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from inference import ProtoGCNInference

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, 'external/ProtoGCN'))
sys.path.append(PROJECT_ROOT)


app = FastAPI(title="ProtoGCN Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ProtoGCNInference(
    config_path=os.path.join(PROJECT_ROOT, 'configs/exercise/j.py'),
    checkpoint_path=os.path.join(PROJECT_ROOT, 'work_dirs/exercise/j_phase2_2/best_top1_acc_epoch_15.pth')
)

class KeypointsRequest(BaseModel):
    keypoints: List[float]  # Flattened keypoints array
    reset: Optional[bool] = False

class PredictionResponse(BaseModel):
    status: str
    buffer_count: int
    prediction: Optional[dict] = None

@app.post("/add_frame", response_model=PredictionResponse)
async def add_frame(request: KeypointsRequest):
    """프레임 추가 및 실시간 예측"""
    if request.reset:
        model.reset_buffer()
    
    # 키포인트 데이터를 2D 배열로 변환 (20, 3)
    keypoints_flat = request.keypoints
    
    if len(keypoints_flat) == 60:  # 20 joints * 3 coordinates
        keypoints_2d = np.array(keypoints_flat).reshape(20, 3)
    else:
        # 데이터 길이가 예상과 다르면 에러 반환
        return {
            "status": "error",
            "buffer_count": len(model.buffer),
            "prediction": None,
            "error": f"Expected 60 keypoint values, got {len(keypoints_flat)}"
        }
    
    model.add_frame(keypoints_2d)

    response = {
        "status": "buffering",
        "buffer_count": len(model.buffer),
        "prediction": None
    }

    # 60프레임 이상이면 60프레임마다 슬라이딩 윈도우로 예측 수행
    if len(model.buffer) >= 60 and len(model.buffer) % 60 == 0:  # 60프레임마다 예측 (안정적인 업데이트)
        prediction = model.predict_sliding_window()
        if prediction:
            response["status"] = "predicted"
            response["prediction"] = prediction
    
    # 300프레임 도달시 자동 리셋
    if len(model.buffer) >= 300:
        model.reset_buffer()
        response["status"] = "auto_reset"
        response["buffer_count"] = 0

    return response

@app.get("/reset")
async def reset_buffer():
    """버퍼 초기화"""
    model.reset_buffer()
    return {"status": "reset", "buffer_count": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
    
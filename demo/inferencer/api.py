from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from inference import ProtoGCNInference

from config import WORK_DIR, BASE_DIR


app = FastAPI(title="ProtoGCN Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ProtoGCNInference(
    config_path=BASE_DIR+"configs/exercise/j.py",
    checkpoint_path=WORK_DIR+"exercise/j_phase2_2/best_top1_acc_epoch_15.pth"
)

class KeypointsRequest(BaseModel):
    keypoints: List[List[float]]  # List of keypoints, each keypoint is [x, y, z, visibility]
    reset: Optional[bool] = False

class PredictionResponse(BaseModel):
    status: str
    buffer_count: int
    prediction: Optional[dict] = None

@app.post("/add_frame", response_model=PredictionResponse)
async def add_frame(request: KeypointsRequest):
    """프레임 추가 및 예측"""
    if request.reset:
        model.reset_buffer()
    
    keypoints = np.array(request.keypoints)
    model.add_frame(keypoints)

    response = {
        "status": "buffering",
        "buffer_count": len(model.buffer),
        "prediction": None
    }

    if len(model.buffer) == 100:
        prediction = model.predict()
        response["status"] = "predicted"
        response["prediction"] = prediction

    return response

@app.get("/reset")
async def reset_buffer():
    """버퍼 초기화"""
    model.reset_buffer()
    return {"status": "reset", "buffer_count": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
    
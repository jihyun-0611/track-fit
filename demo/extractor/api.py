from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
from utils import mediapipe_to_coco


app = FastAPI(title="Keypoint Extract API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

@app.post("/extract_keypoints")
async def extract_keypoints(file: UploadFile=File(...)):
    """단일 프레임에서 키포인트 추출"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints = mediapipe_to_coco(
            results.pose_landmarks,
            frame.shape[1],
            frame.shape[0]
        )
        return {
            "success": True,
            "keypoints": keypoints.tolist(),
            "shape": frame.shape
        }
    
    return {
        "success": False,
        "keypoints": None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

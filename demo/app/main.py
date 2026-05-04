import sys
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR.parent
PROJECT_ROOT = DEMO_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_DIR / 'extractor'))
sys.path.insert(0, str(DEMO_DIR / 'inferencer'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import numpy as np
import cv2
import mediapipe as mp
from dotenv import load_dotenv

from utils import mediapipe_to_coco
from inference import DemoInference

load_dotenv(PROJECT_ROOT / '.env')

app = FastAPI(title="TrackFit Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

CONFIGS = str(PROJECT_ROOT / 'configs/exercise')
CKPTS = str(PROJECT_ROOT / 'work_dirs')

model = DemoInference(
    streams=[
        {'config': f'{CONFIGS}/j.yaml',  'checkpoint': f'{CKPTS}/finetuning_exclude_flip_only/best_top1_acc_epoch_50.pth'},
        {'config': f'{CONFIGS}/b.yaml',  'checkpoint': f'{CKPTS}/bone_finetuning_exclude_flip_only/best_top1_acc_epoch_30.pth'},
        {'config': f'{CONFIGS}/jm.yaml', 'checkpoint': f'{CKPTS}/jm_finetuning_exclude_flip_only/best_top1_acc_epoch_25.pth'},
        {'config': f'{CONFIGS}/bm.yaml', 'checkpoint': f'{CKPTS}/bm_finetuning_exclude_flip_temporal_crop/best_top1_acc_epoch_40.pth'},
    ],
    weights=[4, 3, 2, 2],
)


@app.get("/")
async def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    selected_exercise = None

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if message.get("type") == "exercise_selection":
                    selected_exercise = message.get("exercise")
                    continue
            except json.JSONDecodeError:
                pass

            img_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                keypoints = mediapipe_to_coco(
                    results.pose_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )
                keypoints_list = keypoints.tolist()

                model.add_frame(keypoints)

                response_data = {
                    "status": "pose_detected",
                    "keypoints": keypoints_list,
                    "joint_scores": [
                        {"joint_id": i, "position": [kp[0], kp[1], 0], "score": kp[2]}
                        for i, kp in enumerate(keypoints_list)
                    ],
                    "buffer_count": len(model.buffer),
                }

                if len(model.buffer) >= 60 and len(model.buffer) % 60 == 0:
                    prediction = model.predict_rolling_window(selected_exercise=selected_exercise)
                    if prediction:
                        response_data["status"] = "predicted"
                        response_data["prediction"] = prediction

                if len(model.buffer) >= 300:
                    model.reset_buffer()
                    response_data["status"] = "auto_reset"
                    response_data["buffer_count"] = 0

                await websocket.send_text(json.dumps(response_data))
            else:
                await websocket.send_text(json.dumps({
                    "status": "no_pose",
                    "keypoints": [],
                    "joint_scores": [],
                }))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

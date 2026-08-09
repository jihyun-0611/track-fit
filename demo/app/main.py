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

from keypoint_convert import mediapipe_to_coco20
from ensemble_config import load_ensemble_streams
from inference import CLIP_LEN, LABEL_MAP, DemoInference

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

ENSEMBLE_CONFIG_PATH = DEMO_DIR / "configs" / "ensemble.yaml"
ensemble_streams = load_ensemble_streams(
    ENSEMBLE_CONFIG_PATH,
    PROJECT_ROOT,
    expected_num_classes=len(LABEL_MAP),
)
model = DemoInference(streams=ensemble_streams)

MAX_MISSING_FRAME_RATIO = 0.5


def missing_frame_ratio(buffer):
    """Return the all-zero frame ratio in the window passed to inference."""
    frames = list(buffer)
    if not frames:
        return 1.0

    if len(frames) < CLIP_LEN:
        repeat = (CLIP_LEN // len(frames)) + 1
        frames = (frames * repeat)[:CLIP_LEN]
    else:
        frames = frames[-CLIP_LEN:]

    missing_count = sum(not np.any(frame) for frame in frames)
    return missing_count / len(frames)


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
                keypoints = mediapipe_to_coco20(
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

                if (
                    len(model.buffer) >= 60
                    and len(model.buffer) % 60 == 0
                    and missing_frame_ratio(model.buffer) <= MAX_MISSING_FRAME_RATIO
                ):
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
                model.add_frame(np.zeros((20, 3), dtype=np.float32))
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

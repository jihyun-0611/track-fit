from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import json
import base64
import numpy as np
import cv2
import asyncio


app = FastAPI(title="TrackFit Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

MEDIAPIPE_URL = "http://localhost:8001"

@app.get("/")
async def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async with httpx.AsyncClient() as client:
        try:
            while True:
                data = await websocket.receive_text()
                img_data = base64.b64decode(data.split(",")[1])

                mp_response = await client.post(
                    f"{MEDIAPIPE_URL}/extract_keypoints",
                    files={'file': ('frame.jpg', img_data, 'image/jpeg')}
                )
                mp_results = mp_response.json()

                if mp_results["success"]:
                    keypoints = mp_results["keypoints"]  # COCO format: [[x, y, visibility], ...]
                    
                    # 각 관절의 스코어 계산 (COCO 포맷 기준)
                    joint_scores = []
                    for i, joint in enumerate(keypoints):
                        if len(joint) >= 3:
                            x, y, visibility = joint[0], joint[1], joint[2]
                            joint_scores.append({
                                "joint_id": i,
                                "position": [x, y, 0],  # z는 0으로 설정
                                "score": visibility
                            })

                    await websocket.send_text(json.dumps({
                        "status": "pose_detected",
                        "keypoints": keypoints,
                        "joint_scores": joint_scores
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "status": "no_pose",
                        "keypoints": [],
                        "joint_scores": []
                    }))
        except WebSocketDisconnect:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
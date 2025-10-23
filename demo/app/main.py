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
PROTOGCN_URL = "http://localhost:8002"

@app.get("/")
async def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0

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
                    gcn_response = await client.post(
                        f"{PROTOGCN_URL}/add_frame",
                        json={"keypoints": mp_results["keypoints"]}
                    )
                    gcn_results = gcn_response.json()

                    frame_count = gcn_results["buffer_count"]

                    await websocket.send_text(json.dumps({
                        "frame_count": frame_count,
                        "status": gcn_results["status"],
                        "prediction": gcn_results.get("prediction")
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "frame_count": frame_count,
                        "status": "no_pose"
                    }))
        except WebSocketDisconnect:
            await client.get(f"{PROTOGCN_URL}/reset")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
import os
import cv2
import numpy as np
import keras
from keras.applications.resnet50 import preprocess_input
from collections import deque
import time
import yt_dlp
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
### 웹소켓 서버 설정 및 모델 로드하기 ###
# 리액트 접속 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 모델 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resnet50_padding_model.keras')
model = keras.models.load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input}, safe_mode=False)

def resize_with_padding(image, target_size=(224, 224)): #
    h, w = image.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = (target_size[0] - new_h) // 2, (target_size[1] - new_w) // 2
    return cv2.copyMakeBorder(resized, pad_h, target_size[0]-new_h-pad_h, pad_w, target_size[1]-new_w-pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# --- 웹소켓 엔드포인트 ---
@app.websocket("/ws/analyze")
async def fire_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    cap = None
    
    history = deque(maxlen=8)
    ewma_prob = 0.0
    
    try:
        # 1. 프론트에서 보낸 유튜브 URL 받기
        data = await websocket.receive_json()
        video_url = data.get("video_url")
        
        # 2. 유튜브 주소 추출 및 OpenCV로 스트림 열기
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            actual_url = info.get('url')
        
        cap = cv2.VideoCapture(actual_url)

        while True:
            ret, frame = cap.read()
            if not ret: break

            # 3. 모델 로직 실행
            img_ready = resize_with_padding(frame, (224, 224))
            img_input = preprocess_input(np.expand_dims(cv2.cvtColor(img_ready, cv2.COLOR_BGR2RGB), axis=0))
            raw_prob = float(model.predict(img_input, verbose=0)[0][0])
            
            # EWMA 및 상태 판정 (v5.3 설정값)
            FIRE_THRESHOLD, EMERGENCY_COUNT, EMERGENCY_EWMA, SAFE_EXIT_THRESHOLD = 0.20, 2, 0.30, 0.05
            
            alpha = 0.5 if raw_prob > FIRE_THRESHOLD else 0.2
            ewma_prob = (alpha * raw_prob) + ((1 - alpha) * ewma_prob)
            history.append('fire' if raw_prob > FIRE_THRESHOLD else 'normal')
            fire_count = history.count('fire')

            status_code = 0
            if (fire_count >= EMERGENCY_COUNT or ewma_prob > EMERGENCY_EWMA) and raw_prob > SAFE_EXIT_THRESHOLD:
                status_code = 2
            elif fire_count >= 1 or ewma_prob > 0.15:
                status_code = 1

            # 4. 분석 결과를 리액트 전송
            await websocket.send_json({
                "status_code": status_code,
                "probability": round(raw_prob, 4),
                "ewma": round(ewma_prob, 4),
                "fire_count": fire_count,
                "timestamp": time.strftime('%H:%M:%S')
            })
            
            # 실시간성을 유지하기 위해서 딜레이 넣음
            await asyncio.sleep(0.1) 

    except WebSocketDisconnect:
        print("연결 종료")
    finally:
        if cap: cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
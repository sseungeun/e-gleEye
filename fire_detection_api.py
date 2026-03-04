from fastapi import FastAPI, Query
import cv2
import numpy as np
import keras
from keras.applications.resnet50 import preprocess_input
from collections import deque
import time
import yt_dlp

app = FastAPI()

# 1. 모델 로드 (승은 님의 모델 파일명과 일치해야 함)
model = keras.models.load_model('resnet50_padding_model.keras', custom_objects={'preprocess_input': preprocess_input}, safe_mode=False)

# 상태 관리를 위한 변수 (메모리 유지)
history = deque(maxlen=8)
ewma_prob = 0.0

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = (target_size[0] - new_h) // 2, (target_size[1] - new_w) // 2
    return cv2.copyMakeBorder(resized, pad_h, target_size[0]-new_h-pad_h, pad_w, target_size[1]-new_w-pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

@app.get("/analyze")
async def analyze_frame(video_url: str = Query(..., description="분석할 영상 URL")):
    global ewma_prob
    
    # 1. 영상 프레임 가져오기
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        actual_url = info.get('url')

    cap = cv2.VideoCapture(actual_url)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "프레임을 읽을 수 없습니다."}

    # 2. 승은 님의 핵심 로직 적용 (v5.2 튜닝 버전)
    img_ready = resize_with_padding(frame, (224, 224))
    img_input = preprocess_input(np.expand_dims(cv2.cvtColor(img_ready, cv2.COLOR_BGR2RGB), axis=0))
    raw_prob = float(model.predict(img_input, verbose=0)[0][0])
    
    # [수정포인트] 수치 안정화 및 반응성 강화
    if raw_prob > 0.70:
        safe_raw = raw_prob
        current_alpha = 0.4  # 불이 날 땐 적당히 빠르게 쌓기
    else:
        safe_raw = 0.0
        current_alpha = 0.6  # 꺼질 땐 확실하게 깎기
    
    ewma_prob = (current_alpha * safe_raw) + ((1 - current_alpha) * ewma_prob)
    
    # [수정포인트] EMERGENCY 진입 장벽을 낮춰서 큰 불을 놓치지 않게 함
    detected = 'fire' if raw_prob > 0.70 else 'normal' 
    history.append(detected)
    fire_count = history.count('fire')

    # 3. 최종 상태 판정
    if fire_count >= 5 and ewma_prob > 0.75:
        status = "!!! EMERGENCY: FIRE !!!"
    elif fire_count >= 1 or ewma_prob > 0.30:
        status = "WARNING: CHECKING..."
    else:
        status = "NORMAL"

    return {
        "status": status,
        "probability": round(raw_prob, 4),
        "ewma": round(ewma_prob, 4),
        "fire_count": fire_count,
        "timestamp": time.strftime('%H:%M:%S')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
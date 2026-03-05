from fastapi import FastAPI, Query
import cv2
import numpy as np
import keras
from keras.applications.resnet50 import preprocess_input
from collections import deque
import time
import yt_dlp
import tensorflow as tf

app = FastAPI()

# 1. 모델 로드
model = keras.models.load_model('resnet50_padding_model.keras', custom_objects={'preprocess_input': preprocess_input}, safe_mode=False)

# 상태 관리를 위한 변수 (메모리 유지)
# 서버가 켜져 있는 동안 최근 8회 검사 결과를 저장
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
    
    try:
        # 1. 영상 프레임 가져오기 (yt-dlp)
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            actual_url = info.get('url')

        cap = cv2.VideoCapture(actual_url)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"error": "영상을 불러올 수 없습니다."}

        # 2. 작은 불길 전용 민감도 설정 (v5.3 튜닝)
        FIRE_THRESHOLD = 0.20       # 0.2만 넘어도 '불'로 의심
        EMERGENCY_COUNT = 2         # 8회 중 2회만 걸려도 비상
        EMERGENCY_EWMA = 0.30       # 평균 0.3 이상이면 비상
        SAFE_EXIT_THRESHOLD = 0.05  # 확실히 꺼졌을 때만 해제

        # 3. 모델 추론
        img_ready = resize_with_padding(frame, (224, 224))
        img_input = preprocess_input(np.expand_dims(cv2.cvtColor(img_ready, cv2.COLOR_BGR2RGB), axis=0))
        raw_prob = float(model.predict(img_input, verbose=0)[0][0])
        
        # 4. EWMA(지수 이동 평균) 및 상태 관리
        # 반응성을 위해 alpha 값을 고정(0.5)하거나 상황에 맞춰 조정
        alpha = 0.5 if raw_prob > FIRE_THRESHOLD else 0.2
        ewma_prob = (alpha * raw_prob) + ((1 - alpha) * ewma_prob)
        
        detected = 'fire' if raw_prob > FIRE_THRESHOLD else 'normal' 
        history.append(detected)
        fire_count = history.count('fire')

        # 5. 최종 상태 판정
        if (fire_count >= EMERGENCY_COUNT or ewma_prob > EMERGENCY_EWMA) and raw_prob > SAFE_EXIT_THRESHOLD:
            status = "!!! EMERGENCY: FIRE !!!"
            status_code = 2  # 웹 프론트엔드에서 제어하기 편하게 숫자 코드 추가
        elif fire_count >= 1 or ewma_prob > 0.15:
            status = "WARNING: SCANNING..."
            status_code = 1
        else:
            status = "NORMAL"
            status_code = 0

        return {
            "status": status,
            "status_code": status_code,
            "probability": round(raw_prob, 4),
            "ewma": round(ewma_prob, 4),
            "fire_count": fire_count,
            "timestamp": time.strftime('%H:%M:%S')
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 외부 접근이 가능하도록 host="0.0.0.0" 유지
    uvicorn.run(app, host="0.0.0.0", port=8000)
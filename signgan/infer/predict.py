# signgan/infer/predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_WIDTH = 256
IMG_HEIGHT = 64

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # 채널 추가
    img = np.expand_dims(img, axis=0)   # 배치 차원 추가 (1, 64, 256, 1)
    return img

def predict_signature(image_path, model_path="signgan/model/signature_classifier.h5"):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    prob = model.predict(img)[0][0]
    label = "✅위조 서명" if prob > 0.5 else "✅진짜 서명"
    print(f"🔍 판별 결과: {label} (확률: {prob:.4f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python predict.py [이미지경로]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_signature(image_path)

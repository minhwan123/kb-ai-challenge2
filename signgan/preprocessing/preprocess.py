# signgan/preprocessing/preprocess.py

import os
import cv2
import numpy as np
import random

# 서명 이미지 경로 설정
base_dir = "./signgan/data/synthetic"
genuine_dir = os.path.join(base_dir, "genuine")
forged_dir = os.path.join(base_dir, "forged")

# 이미지 크기 설정
IMG_WIDTH = 256
IMG_HEIGHT = 64

def load_images_from_folder(folder, label):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 불러오기
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # 크기 통일
            img = img / 255.0  # 정규화
            img = np.expand_dims(img, axis=-1)  # 채널 추가: (64, 256, 1)
            data.append((img, label))
    return data

def preprocess_all():
    print("이미지 불러오는 중...")
    genuine = load_images_from_folder(genuine_dir, label=0)
    forged = load_images_from_folder(forged_dir, label=1)

    all_data = genuine + forged
    random.shuffle(all_data)

    # 분리
    images, labels = zip(*all_data)
    images = np.array(images)
    labels = np.array(labels)

    print(f"✅전처리 완료: 총 {len(images)}개 이미지")
    return images, labels

if __name__ == "__main__":
    import random
    images, labels = preprocess_all()
    print("샘플 이미지 shape:", images[0].shape)
    print("샘플 라벨:", labels[0])

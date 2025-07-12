# signgan/train/train_classifier.py
import sys, os
sys.path.append(os.path.abspath("."))

from signgan.preprocessing.preprocess import preprocess_all
from signgan.model.classifier import build_signature_classifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# 데이터 불러오기
images, labels = preprocess_all()

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# 모델 빌드
model = build_signature_classifier(input_shape=(64, 256, 1))
model.summary()

# 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8
)

# 모델 저장
model.save("signgan/model/signature_classifier.h5")
print("✅모델이 저장되었습니다: signature_classifier.h5")

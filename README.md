# 🖋️ SignGAN: GAN을 활용한 금융 서명 위변조 탐지 시스템

## 🔍 프로젝트 개요

본 프로젝트는 GAN을 활용하여 가짜 서명을 생성하고,  
CNN 기반 판별기를 통해 진짜/위조 서명을 자동으로 판별하는 금융 보안용 AI 시스템입니다.


---

## 🧠 핵심 기능

- ✅ GAN으로 위조 서명 이미지 생성 (Pix2Pix/StyleGAN 아님, DCGAN 구조)
- ✅ CNN 판별기로 진짜 vs 가짜 분류
- ✅ 합성된 위조 이미지로 CNN을 속이는 실험 성공
- ✅ 추후 GAN-generated 이미지로 CNN 강화 학습 가능
- ✅ 추론 결과 CLI로 판별 출력

---

## 🚀 실행 예시

```bash
# 서명 이미지 생성
python signgan/preprocessing/generate_synthetic_data.py

# CNN 학습
python signgan/train/train_classifier.py

# GAN 학습
python signgan/train/train_gan.py

# 서명 판별
python signgan/infer/predict.py signgan/data/synthetic/forged/Kim_Minho_forged_1.png


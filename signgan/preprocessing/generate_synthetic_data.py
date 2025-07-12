# signgan/preprocessing/generate_synthetic_data.py

import os
import random
from PIL import Image, ImageDraw, ImageFont

# 사용할 이름 목록 (원하는 만큼 추가 가능)
names = [
    "Kim Minho", "Park Soyeon", "Lee Jiwon", "Choi Daehyun",
    "Jung Hana", "Kang Hyunwoo", "Yoon Seokmin", "Han Jisoo"
]

# 사용할 폰트 경로 설정 (DancingScript 제외)
font_dir = "./signgan/fonts"
fonts = []
for file in os.listdir(font_dir):
    if file.endswith(".ttf") and "DancingScript" not in file:  # DancingScript 제외
        path = os.path.join(font_dir, file)
        try:
            ImageFont.truetype(path, 20)  # 테스트로 로딩
            fonts.append(path)
        except:
            print(f"사용 불가 폰트: {file}")

if not fonts:
    raise RuntimeError("사용 가능한 .ttf 폰트가 없습니다.")

# 저장 위치
base_dir = "./signgan/data/synthetic"
genuine_dir = os.path.join(base_dir, "genuine")
forged_dir = os.path.join(base_dir, "forged")
os.makedirs(genuine_dir, exist_ok=True)
os.makedirs(forged_dir, exist_ok=True)

# 이미지 생성 함수
def generate_signature_image(text, font_path, rotate=False):
    font = ImageFont.truetype(font_path, size=60)
    img = Image.new("L", (300, 100), color=255)  # 흰 배경
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, font=font, fill=0)

    if rotate:
        angle = random.randint(-15, 15)
        img = img.rotate(angle, expand=True, fillcolor=255)

    return img

# 진짜 서명 생성
for name in names:
    for i in range(10):
        font_path = random.choice(fonts)
        img = generate_signature_image(name, font_path)
        filename = f"{name.replace(' ', '_')}_genuine_{i}.png"
        img.save(os.path.join(genuine_dir, filename))

# 위조 서명 생성 (약간의 회전/폰트 무작위로 변조)
for name in names:
    for i in range(10):
        font_path = random.choice(fonts)
        img = generate_signature_image(name, font_path, rotate=True)
        filename = f"{name.replace(' ', '_')}_forged_{i}.png"
        img.save(os.path.join(forged_dir, filename))

print("✅서명 이미지 생성 완료!")

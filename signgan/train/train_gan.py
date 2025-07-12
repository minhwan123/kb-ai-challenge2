# signgan/train/train_gan.py
import sys
import os
sys.path.append(os.path.abspath("."))

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from signgan.model.gan import build_generator, build_discriminator, LATENT_DIM, IMG_WIDTH, IMG_HEIGHT

# 하이퍼파라미터
EPOCHS = 1000
BATCH_SIZE = 16
SAVE_DIR = "signgan/data/generated"
os.makedirs(SAVE_DIR, exist_ok=True)

# 이미지 전처리 함수 (genuine 만 사용)
def load_real_images():
    path = "signgan/data/synthetic/genuine"
    imgs = []
    for f in os.listdir(path):
        if f.endswith(".png"):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(path, f), color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = (img - 127.5) / 127.5  # [-1, 1] 스케일
            imgs.append(img)
    return np.array(imgs)

# 모델 정의
generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
g_optimizer = Adam(1e-4)
d_optimizer = Adam(1e-4)

# 진짜/가짜 라벨
real_label = np.ones((BATCH_SIZE, 1))
fake_label = np.zeros((BATCH_SIZE, 1))

# 학습 루프
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        d_loss_real = cross_entropy(real_label, real_output)
        d_loss_fake = cross_entropy(fake_label, fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = cross_entropy(real_label, fake_output)

    gradients_of_generator = g_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return d_loss, g_loss

# 학습 시작
def train():
    real_images = load_real_images()
    print(f"✅학습 데이터 로딩 완료: {real_images.shape}")
    dataset = tf.data.Dataset.from_tensor_slices(real_images).shuffle(1000).batch(BATCH_SIZE)

    for epoch in range(1, EPOCHS+1):
        for real_batch in dataset:
            if real_batch.shape[0] != BATCH_SIZE: continue
            d_loss, g_loss = train_step(real_batch)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] D_loss: {d_loss:.4f} / G_loss: {g_loss:.4f}")
            save_generated_image(epoch)

def save_generated_image(epoch):
    noise = tf.random.normal([1, LATENT_DIM])
    gen_img = generator(noise, training=False)[0].numpy()
    gen_img = ((gen_img + 1.0) * 127.5).astype(np.uint8)
    tf.keras.preprocessing.image.save_img(
        os.path.join(SAVE_DIR, f"gen_{epoch:04d}.png"),
        gen_img,
        scale=False
    )

if __name__ == "__main__":
    train()

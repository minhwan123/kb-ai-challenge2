# signgan/model/gan.py

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_WIDTH = 256
IMG_HEIGHT = 64
CHANNELS = 1
LATENT_DIM = 100

# Generator
def build_generator():
    model = models.Sequential([
        layers.Dense(8*32*128, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 32, 128)),  # -> (8,32,128)

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

# Discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model

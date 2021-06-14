import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib


#setup data
import os

data_dir = "/Users/eriksieburgh/PycharmProjects/ScientificML/Data_Erik"

batch_size = 2 #This only seems to work when batch_size = 1
EPOCHS = 10
SHAPE = (32, 32)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=120,
    image_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    shuffle=True,
    color_mode='rgb')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=120,
    image_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    shuffle=True,
    color_mode='rgb')

# Rescaling of dataset is still needed

input = layers.Input(shape=(32, 32, 3))
chanDim = -1

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

for image_batch, labels_batch in val_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

hist = autoencoder.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("[INFO] making predictions...")
decoded = autoencoder.predict(val_ds)


for i in range (0,2):
    recon=(decoded[i] * 255).astype("uint8")
    plt.imshow(recon)
    plt.show()
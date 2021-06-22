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

data_dir = "/Users/eriksieburgh/Desktop/TUdelft/M-AM/Semester 2/SML_Special Topic CSE/Data_files/Flow_data_color"

batch_size = 10 #This only seems to work when batch_size = 1
EPOCHS = 20
SHAPE = (32, 96)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=120,
    image_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    shuffle=True,
    label_mode=None,
    color_mode="rgb")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=120,
    image_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    shuffle=True,
    label_mode=None,
    color_mode="rgb")


plt.figure(figsize=(10, 10))
for images in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

print(type(val_ds))

# # Rescaling of dataset is still needed
# normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
#
# normalized_train_ds = train_ds.map(lambda x: normalization_layer(x))
# normalized_val_ds = val_ds.map(lambda x: normalization_layer(x))

input = layers.Input(shape=(32, 96, 3))
chanDim = -1

# Encoder
x = layers.Conv2D(152, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(152, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# Decoder
x = layers.Conv2DTranspose(152, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(152, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

for image_batch in train_ds:
    print(image_batch.shape)
    break

for image_batch in val_ds:
    print(image_batch.shape)
    break

hist = autoencoder.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
exit(4)
print("[INFO] making predictions...")
decoded = autoencoder.predict(val_ds)


for i in range (0,2):
    recon=(decoded[i]).astype("uint8")
    plt.imshow(recon)
    plt.show()

index = 0
for image, label in train_ds:
    index += 1
plt.subplot(3, 3, index)
plt.imshow(image)
plt.title("Class: {}".format(class_names[label]))
plt.axis("off")
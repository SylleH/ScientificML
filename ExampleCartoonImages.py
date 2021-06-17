import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import os


# This Example is from https://learnopencv.com/autoencoder-in-tensorflow-2-beginners-guide/
# The Cartoon image example starts about half way down the webpage.

batch_size = 128
INIT_LR =  1e-3
ROOT_PATH = "/Users/eriksieburgh/Desktop/TUdelft/M-AM/Semester 2/SML_Special Topic CSE/Data_files/Data_Example"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ROOT_PATH,
    image_size=(256, 256),
    batch_size=batch_size,
    label_mode=None)

print(type(train_ds))
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)

normalized_ds = train_ds.map(lambda x: normalization_layer(x))

print(type(normalized_ds))

def encoder(input_encoder):
    inputs = keras.Input(shape=input_encoder, name='input_layer')
    # Block 1
    x = layers.Conv2D(32, kernel_size=3, strides= 2, padding='same', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)

    # Block 3
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)

    # Block 4
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)

    # Block 5
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.LeakyReLU(name='lrelu_5')(x)

    # Final Block
    flatten = layers.Flatten()(x)
    bottleneck = layers.Dense(200, name='dense_1')(flatten)
    model = tf.keras.Model(inputs, bottleneck, name="Encoder")
    return model

def decoder(input_decoder):
    # Initial Block
    inputs = keras.Input(shape=input_decoder, name='input_layer')
    x = layers.Dense(4096, name='dense_1')(inputs)
    x = tf.reshape(x, [-1, 8, 8, 64], name='Reshape_Layer')

    # Block 1
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)

    # Block 2
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)

    # Block 3
    x = layers.Conv2DTranspose(64, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)

    # Block 4
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)

    # Block 5
    outputs = layers.Conv2DTranspose(3, 3, 2,padding='same', activation='sigmoid', name='conv_transpose_5')(x)
    model = tf.keras.Model(inputs, outputs, name="Decoder")
    return model
#
# opt = tf.optimizers.Adam(learning_rate=INIT_LR)
# autoencoder.compile(loss="mse", optimizer=opt)
# encoder.summary()
# decoder.summary()
#
# hist = autoencoder.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# reconstruction = None
# lat_space = None
# for i in normalized_ds:
#     latent = encoder.predict(i)
#     out = decoder.predict(latent)
#     if reconstruction is None:
#         reconstruction = out
#         lat_space = latent
#     else:
#         reconstruction = np.concatenate((reconstruction, out))
#         lat_space = np.concatenate((lat_space, latent))
#     if reconstruction.shape[0] > 5000:
#         break
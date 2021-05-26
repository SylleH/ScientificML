import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib


#setup data
import os

data_dir = "/Users/eriksieburgh/PycharmProjects/ScientificML/Data"

batch_size = 10
img_height = 100
img_width = 100

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(25,50), latentDim=25):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # Adding noise: keras.layers.Dropout(0.5)(x) or keras.layers.GaussianNoise(stddev = 0.2)(x)
        # x = Dropout(0.5)(x) #Noise added to images
        #x = GaussianNoise(stddev = 0.2)(x)
        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
            x = BatchNormalization(axis=chanDim)(x)
        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")


        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,padding="same")(x)
            x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
            x = BatchNormalization(axis=chanDim)(x)
        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),name="autoencoder")
        # return a 3-tuple of the encoder, decoder, and autoencoder
#
plt.imshow(train_ds[1])

# (encoder, decoder, autoencoder) = ConvAutoencoder.build(img_width, img_height, 3,filters=)
# opt = tf.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# autoencoder.compile(loss="mse", optimizer=opt)

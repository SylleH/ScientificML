"""
FOLLOWED TUTORIAL:https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/
Author: S.Hoogeveen
Machine learning project for Special Topics in Computational Science & Engineering: Scientific Machine Learning, MSc Applied Mathematics, TU Delft
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import os
import matplotlib.pyplot as plt
import pydot
#from tensorflow_core.python.keras.layers import GaussianNoise, Dropout

ROOT_PATH = "/Users/eriksieburgh/PycharmProjects/ScientificML/Data_Sylle"
train_dir = os.path.join(ROOT_PATH, "TrainingData")
val_dir = os.path.join(ROOT_PATH, "ValidationData")

IMAGES = ROOT_PATH
SHAPE = (52, 52) #height, width
INIT_LR = 1e-3

EPOCHS = 1 #loss stabiel na 185 EPOCHS
BS = 1



class ConvAutoencoder:
    @staticmethod
    def build(height, width, depth,  latentDim=1024):
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

        # apply a CONV => RELU => BN operation
        x = Conv2D(filters=50, kernel_size=3, strides=1, padding="same")(x)
        x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        x = MaxPool2D(2)(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(filters=100, kernel_size=3, strides=1, padding="same")(x)
        x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        x = MaxPool2D(2)(x)
        x = BatchNormalization(axis=chanDim)(x)

        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")


        # start building the decoder model which will accept the
		# output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(filters=100, kernel_size=3, strides=1,padding="same")(x)
        x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        x = UpSampling2D(2)(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2DTranspose(filters=50, kernel_size=3, strides=1, padding="same")(x)
        x = ReLU(max_value=None, negative_slope=0, threshold=0)(x)
        x = UpSampling2D(2)(x)
        x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2D(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),name="autoencoder")
        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)


(encoder, decoder, autoencoder) = ConvAutoencoder.build(SHAPE[0], SHAPE[1], 3)
opt = tf.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)
encoder.summary()
decoder.summary()

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
train_gen = image_generator.flow_from_directory(
    os.path.join(IMAGES, "TrainingData"),
    class_mode="input", target_size=SHAPE,  batch_size=BS,shuffle=True
)
val_gen = image_generator.flow_from_directory(
    os.path.join(IMAGES, "ValidationData"),
    class_mode="input", target_size=SHAPE,batch_size=BS,shuffle=True
)
hist = autoencoder.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("[INFO] making predictions...")

batchX, batchy = train_gen.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

print(train_gen.next())
print(type(train_gen.next()))

decoded = autoencoder.predict(val_gen)

print(type(decoded))


for i in range (0,2):
    recon = (decoded[i]* 255).astype("uint8")
    plt.imshow(recon)
    plt.show()

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()

# This makes two images of the model. Examples can be found in this directory.
# tf.keras.utils.plot_model(
#     encoder,
#     to_file="encoder.png",
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=False,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )
#
# tf.keras.utils.plot_model(
#     decoder,
#     to_file="decoder.png",
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=False,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )

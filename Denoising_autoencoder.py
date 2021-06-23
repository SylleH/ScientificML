"""
Denoising Autoencoder
Author: S. Hoogeveen

objective: denoise self-generated flow data
use hyperparameter tuning
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import cv2
import glob
import os
import keras_tuner as kt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from PIL import Image

#setup data
ROOT_PATH = "/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML_local/"
train_dir = os.path.join(ROOT_PATH, "TrainingData")
test_dir = os.path.join(ROOT_PATH, "TestData")

IMAGES = ROOT_PATH
SHAPE = (48, 144) #height, width
depth = 3 #3 for RGB, 1 for grayscale
INIT_LR = 1e-3

EPOCHS = 20 #loss stabiel na 150 EPOCHS
BS = 5

def create_datasets():
    train_data = []
    files = glob.glob ("/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML_local/Flow_data_colored/TrainData/*.png")

    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(image)  # get b, g, r
        image = cv2.merge([r, g, b])  # switch it to r, g, b
        image = cv2.resize(image, (SHAPE[1],SHAPE[0]))
        train_data.append (image)

    train_data = np.array(train_data)
    train_data = flow_data.astype('float32') / 255
    #print(train_data.shape)

    test_data = []
    files = glob.glob("/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML_local/Flow_data_colored/TestData/*.png")

    for myFile in files:
        # print(myFile)
        image = cv2.imread(myFile, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(image)  # get b, g, r
        image = cv2.merge([r, g, b])  # switch it to r, g, b
        image = cv2.resize(image, (SHAPE[1], SHAPE[0]))
        test_data.append(image)

    test_data = np.array(test_data)
    test_data = test_data.astype('float32') / 255
    return train_data, test_data

def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = (array1[indices]*255).astype("uint8")
    images2 = (array2[indices]*255).astype("uint8")

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('denoised_data.png')
    plt.show()

def build_model(hp):
    chanDim = -1
    f_list = [0]*3
    ks_list = [0]*3
    input = layers.Input(shape=(SHAPE[0], SHAPE[1],depth))
    x = input
    conv_layers = hp.Int("conv_layers", min_value = 1,max_value= 3, default=2)
    for i in range(conv_layers):
        f = hp.Int("filters_" + str(i), 10, 150, step=50, default=50)
        ks = hp.Int("kernel_size_" + str(i), 1, 3, step=1)
        f_list[i] = f
        ks_list[i] = ks
        x = layers.Conv2D(
            filters=f,
            kernel_size=ks,
            activation="relu",
            padding="same",
        )(x)
        x = layers.MaxPool2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)


    volumeSize = K.int_shape(x)
    x = layers.Flatten()(x)
    latentDim = hp.Int("latentDim", min_value = 100, max_value = 1000, step =100)
    latent = layers.Dense(latentDim)(x)

    encoder = Model(input, latent, name="encoder")

    latentInputs = layers.Input(shape=(latentDim))
    x = layers.Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    # apply a CONV_TRANSPOSE => RELU => BN operation
    if conv_layers == 1:
        f = f_list[0]
        ks = ks_list[0]
        x = layers.Conv2DTranspose(filters = f, kernel_size = ks,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)

    if conv_layers == 2:
        f0 = f_list[0]
        ks0 = ks_list[0]
        f1 = f_list[1]
        ks1 = ks_list[1]
        x = layers.Conv2DTranspose(filters = f0, kernel_size = ks0,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)

        x = layers.Conv2DTranspose(filters = f1, kernel_size = ks1,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)

    if conv_layers == 3:
        f0 = f_list[0]
        ks0 = ks_list[0]
        f1 = f_list[1]
        ks1 = ks_list[1]
        f2 = f_list[2]
        ks2 = ks_list[2]
        x = layers.Conv2DTranspose(filters = f0, kernel_size = ks0,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2DTranspose(filters = f1, kernel_size = ks1,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2DTranspose(filters = f2, kernel_size = ks2,
                                   activation = "relu", padding = "same")(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)

    x = layers.Conv2D(depth, (3, 3), padding="same")(x)
    outputs = layers.Activation("sigmoid")(x)

    decoder = Model(latentInputs, outputs, name="decoder")
    autoencoder = Model(input, decoder(encoder(input)), name="autoencoder")

    #optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    opt = tf.optimizers.Adam(hp_learning_rate)
    autoencoder.compile(opt, loss="mse", metrics=["accuracy"])

    return autoencoder

train_data, test_data = create_datasets()
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

tuner = kt.Hyperband(build_model,objective='val_accuracy',max_epochs=3,
                     factor=3,directory=ROOT_PATH,project_name='tuner_test')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x=noisy_train_data, y=train_data, validation_split=0.2, epochs=EPOCHS, shuffle=True,batch_size = BS, callbacks=[stop_early])
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)

#build autoencoder with these values
autoencoder = tuner.hypermodel.build(best_hps)
history = autoencoder.fit(x=noisy_train_data, y=train_data, validation_split=0.2, epochs=EPOCHS, shuffle=True,batch_size = BS)

#determine best number of epochs
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#build hypermodel with best parameters and epochs
hypermodel = tuner.hypermodel.build(best_hps)
hist = hypermodel.fit(x=noisy_train_data, y=train_data, epochs=best_epoch, validation_split=0.2,shuffle=True,batch_size = BS)

hypermodel.summary()
# tf.keras.utils.plot_model(
#     hypermodel,
#     to_file="encoder.png",
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=False,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
# )

print("[INFO] making predictions...")
decoded = autoencoder.predict(noisy_test_data)

eval_result = hypermodel.evaluate(noisy_test_data, test_data, batch_size=BS)
print("[test loss, test accuracy]:", eval_result)

display(noisy_test_data, decoded)

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
plt.savefig("epochs_vs_loss.png")
plt.show()
plt.close()

"""ORIGINAL AUTOENCODER
#input = layers.Input(shape=(SHAPE[0], SHAPE[1],depth))
x = input
# apply a CONV => RELU => BN operation
x = layers.Conv2D(filters=50, kernel_size=3, strides=1, padding="same")(x)
x = layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
x = layers.MaxPool2D(2)(x)
x = layers.BatchNormalization(axis=chanDim)(x)

x = layers.Conv2D(filters=100, kernel_size=3, strides=1, padding="same")(x)
x = layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
x = layers.MaxPool2D(2)(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# flatten the network and then construct our latent vector
volumeSize = K.int_shape(x)
x = layers.Flatten()(x)
latent = layers.Dense(latentDim)(x)
# build the encoder model
encoder = Model(input, latent, name="encoder")

#decoder
latentInputs = layers.Input(shape=(latentDim))
x = layers.Dense(np.prod(volumeSize[1:]))(latentInputs)
x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

# apply a CONV_TRANSPOSE => RELU => BN operation
x = layers.Conv2DTranspose(filters=100, kernel_size=3, strides=1,padding="same")(x)
x = layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
x = layers.UpSampling2D(2)(x)
x = layers.BatchNormalization(axis=chanDim)(x)

x = layers.Conv2DTranspose(filters=50, kernel_size=3, strides=1, padding="same")(x)
x = layers.ReLU(max_value=None, negative_slope=0, threshold=0)(x)
x = layers.UpSampling2D(2)(x)
x = layers.BatchNormalization(axis=chanDim)(x)

# apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
x = layers.Conv2D(depth, (3, 3), padding="same")(x)
outputs = layers.Activation("sigmoid")(x)

# build the decoder model
decoder = Model(latentInputs, outputs, name="decoder")
# autoencoder is the encoder + decoder
autoencoder = Model(input, decoder(encoder(input)),name="autoencoder")
"""





<<<<<<< Updated upstream
=======
print("[INFO] making predictions...")
decoded = autoencoder.predict(noisy_test_data)
print(decoded.shape)
display(noisy_test_data, decoded)
plt.savefig('denoised data.png')


#for i in range (0,2):
#    recon=(decoded[i] * 255).astype("uint8")
#    plt.imshow(recon)
#    plt.show()
>>>>>>> Stashed changes

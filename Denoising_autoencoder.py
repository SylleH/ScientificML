import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import cv2
import glob


#setup data
import os

ROOT_PATH = "/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML/Flow_data_colored"
train_dir = os.path.join(ROOT_PATH, "TrainingData")
val_dir = os.path.join(ROOT_PATH, "ValidationData")

IMAGES = ROOT_PATH
SHAPE = (52, 152) #height, width
depth = 3 #3 for RGB, 1 for grayscale
INIT_LR = 1e-3

EPOCHS = 20 #loss stabiel na 150 EPOCHS
BS = 10

def create_datasets():
    train_data = []
    files = glob.glob ("/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML/Flow_data_colored/TrainingData/data_cylinder/*.png")

    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(image)  # get b, g, r
        image = cv2.merge([r, g, b])  # switch it to r, g, b
        image = cv2.resize(image, (SHAPE[1],SHAPE[0]))
        train_data.append (image)

    train_data = np.array(train_data)
    train_data = train_data.astype('float32') / 255
    #print(train_data.shape)

    val_data = []
    files = glob.glob ("/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML/Flow_data_colored/ValidationData/data_cylinder/*.png")

    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(image)  # get b, g, r
        image = cv2.merge([r, g, b])  # switch it to r, g, b
        image = cv2.resize(image, (SHAPE[1],SHAPE[0]))
        val_data.append (image)

    val_data = np.array(val_data)
    val_data = val_data.astype('float32') / 255
    #print(val_data.shape)

    test_data = []
    files = glob.glob("/Users/sylle/Documents/Master Applied Mathematics/WI4450 Special Topics in CSE, Machine Learning /ScientificML/Flow_data_colored/TestData/data_cylinder/*.png")

    for myFile in files:
        # print(myFile)
        image = cv2.imread(myFile, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(image)  # get b, g, r
        image = cv2.merge([r, g, b])  # switch it to r, g, b
        image = cv2.resize(image, (SHAPE[1], SHAPE[0]))
        test_data.append(image)

    test_data = np.array(test_data)
    test_data = test_data.astype('float32') / 255
    return train_data, val_data, test_data

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

    plt.show()



chanDim = -1
latentDim = 1024

input = layers.Input(shape=(SHAPE[0], SHAPE[1],depth))
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
# our autoencoder is the encoder + decoder
autoencoder = Model(input, decoder(encoder(input)),name="autoencoder")


train_data,val_data, test_data = create_datasets()
noisy_train_data = noise(train_data)
noisy_val_data = noise(val_data)
noisy_test_data = noise(test_data)

opt = tf.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)
autoencoder.fit(x=noisy_train_data, y=train_data, validation_data=(noisy_val_data,val_data), epochs=EPOCHS, shuffle=True,batch_size = BS)
encoder.summary()
decoder.summary()



print("[INFO] making predictions...")
decoded = autoencoder.predict(noisy_test_data)
print(decoded.shape)
display(noisy_test_data, decoded)
#display(test_data, decoded)

#for i in range (0,2):
#    recon=(decoded[i] * 255).astype("uint8")
#    plt.imshow(recon)
#    plt.show()

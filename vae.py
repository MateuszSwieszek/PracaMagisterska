from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, \
    Flatten, \
    Reshape
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


def wczytanieObrazok(labelsDIR1='Orgs/masks/', labelsDIR2='Generated/masks/', masks=False):
    '''wczytanie mask'''

    images1 = glob.glob(labelsDIR1 + '*.bmp')
    images1.sort()
    print('################################################################################################')
    print("Wczytywanie obrazow")
    print('Dlugosc obrazow z pierwszej sciezki: ', len(images1))

    images2 = glob.glob(labelsDIR2 + '*.bmp')
    images2.sort()
    print('Dlugosc obrazow z drugiej sciezki: ', len(images2))
    imagesDIR = images1 + images2

    images = np.zeros((len(imagesDIR), 128, 128, 1), dtype=np.float32)
    for im in range(len(imagesDIR)):
        image = cv2.imread(imagesDIR[im])
        if (masks == False):
            image = image[:, :, 0:1] / 255
        elif (np.unique(image)[1] == 255):
            image = image[:, :, 0:1] / 255
        else:
            image = image[:, :, 0:1]
        images[im] = image

    return images


def prepareData(masks):
    print("###########Przygotowywanie danych - dzielenie na zestawy testowy walidacyjny itp oraz dodanie szumu########")
    np.random.seed(1337)
    trainingMasks, testMasks = train_test_split(masks, test_size=0.1)
    print('dlugosc testMasks: ', testMasks.shape[0])
    trainingMasks, valMasks = train_test_split(trainingMasks, test_size=0.3)
    print('dlugosc valMasks: ', valMasks.shape[0])
    print('dlugosc trainingMasks: ', trainingMasks.shape[0])
    # generate corrupted MNIST images by adding noise with normal dist
    # centered at 0.5 and std=0.5
    noise = np.random.normal(loc=0.5, scale=0.5, size=trainingMasks.shape)
    trainingMasks_noisy = trainingMasks + noise
    noise = np.random.normal(loc=0.5, scale=0.5, size=testMasks.shape)
    testMasks_noisy = testMasks + noise
    noise = np.random.normal(loc=0.5, scale=0.5, size=valMasks.shape)
    valMasks_noisy = valMasks + noise

    # adding noise may exceed normalized pixel values>1.0 or <0.0
    # clip pixel values >1.0 to 1.0 and <0.0 to 0.0
    for i in range(trainingMasks_noisy.shape[0]):
        trainingMasks_noisy[i] = np.clip(trainingMasks_noisy[i], 0., 1.)
    for i in range(testMasks_noisy.shape[0]):
        testMasks_noisy[i] = np.clip(testMasks_noisy[i], 0., 1.)
    for i in range(valMasks_noisy.shape[0]):
        valMasks_noisy[i] = np.clip(valMasks_noisy[i], 0., 1.)

    return testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy


def binary(pretrained_weights=None, inp_s=(128, 128, 1)):
    input_img = Input(shape=inp_s)
    x = Conv2D(16, 3, strides=2, padding='same', name='First')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same', name='Second')(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same', name="last")(x)
    # x = Activation('relu')(x)
    # x = Conv2D(32, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(32, 3, strides=2, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(32, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(32, 3, strides=2, padding='same',name="last")(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    encoded = Reshape((16, 16, 1))(x)

    encoded = UpSampling2D(size=(2, 2))(encoded)
    x = Conv2D(16, 3, strides=1, padding='same')(encoded)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(1, 3, strides=1, padding='same')(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)
    model.compile(loss='mse', optimizer='adam')

    print("model compile")

    return model


#
# def binary(pretrained_weights=None, inp_s=(1024, 1024, 1)):
#     input_img = Input(shape=inp_s)
#     x = Conv2D(16, 3, strides=2, padding='same')(input_img)
#     x = Activation('relu')(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(32, 3, strides=2, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(32, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(32, 3, strides=2, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(32, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(32, 3, strides=2, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(32, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(32, 3, strides=2, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = Flatten()(x)
#     x = Dense(512)(x)
#     x = Dense(1024)(x)
#     x = Activation('relu')(x)
#
#     encoded = Reshape((32, 32, 1))(x)
#
#     encoded = UpSampling2D(size=(2, 2))(encoded)
#     x = Conv2D(16, 3, strides=1, padding='same')(encoded)
#     x = Activation('relu')(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#
#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(16, 3, strides=1, padding='same')(x)
#     x = Activation('relu')(x)
#     x = Conv2D(1, 3, strides=1, padding='same')(x)
#     decoded = Activation('sigmoid')(x)
#
#     model = Model(input_img, decoded)
#     model.compile(loss='mse', optimizer='adam')
#     # model.summary()
#     return model


def run():
    path = "JsonFiles/DenoisingAutoencoder2.json"

    f = open(path, )

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Closing file
    f.close()
    loss = map(float, data['loss'])
    loss = list(loss)
    val_loss = map(float, data['val_loss'])
    val_loss = list(val_loss)

    print(loss[10:])
    print(val_loss[10:])
    print(len(loss[10:]))
    x = np.arange(len(loss))

    l = plt.plot(x,loss)
    v = plt.plot(x,val_loss)
    plt.legend([l,v], ['loss','val_loss'])
    plt.show()


if __name__ == '__main__':
    run()

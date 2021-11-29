from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, \
    Reshape
from tensorflow.keras import backend as K
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import BaseLogger
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import math

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



def wczytanieObrazokw(labelsDIR1='Orgs/masks/' , masks=False):
    '''wczytanie mask'''

    images1 = glob.glob(labelsDIR1 + '*.bmp')
    images1.sort()
    print('################################################################################################')
    print("Wczytywanie obrazow")
    print('Dlugosc obrazow z pierwszej sciezki: ', len(images1))

    imagesDIR = images1

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

smooth = 1.


# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def wczytanieObrazokw(labelsDIR1='Orgs/masks/' , masks=False):
    '''wczytanie mask'''

    images1 = glob.glob(labelsDIR1 + '*.bmp')
    images1.sort()
    print('################################################################################################')
    print("Wczytywanie obrazow")
    print('Dlugosc obrazow z pierwszej sciezki: ', len(images1))

    imagesDIR = images1

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


def binary(pretrained_weights=None, inp_s=(128, 128, 1)):
    input_img = Input(shape=inp_s)

    #64x64
    # 512x512
    x = Conv2D(16, 3, strides=2, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 32x32
    # 256x256
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 16x16
    # 128x128
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # # 8x8
    # #64x64
    # x = Conv2D(32, 3, strides=2, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(32, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    #4x4 po wykluczeniu:    8x8
    #32x32                  128x128
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)

    # 4x4
    # 32x32
    encoded = Reshape((8, 8, 1))(x)

    # # 8x8
    # # 64x64
    # encoded = UpSampling2D(size=(2, 2))(encoded)
    # x = Conv2D(16, 3, strides=1, padding='same')(encoded)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    # 16x16
    # 128x128
    x = UpSampling2D(size=(2, 2))(encoded)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 32x32
    # 256x256
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 64x64
    # 512x512
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 128x128
    # 1024x1024
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 3, strides=1, padding='same')(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)

    return model


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(str(v))
            self.H[k] = l

        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as fp:
                json.dump(self.H, fp)

        # dict_keys(['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error'])
        if len(self.H["loss"]) > 1:
            for i in self.H:
                data = self.H[i]
                data = map(float, data)
                data = list(data)
                print(data)

                N = np.arange(0, len(data))
                plt.style.use("ggplot")
                plt.plot(N, data, label="loss")

def prepareData(masks, SegMasks):
    print("###########Przygotowywanie danych - dzielenie na zestawy testowy walidacyjny itp oraz dodanie szumu########")
    test_size = 0.1
    K = math.floor(test_size*len(masks))
    test_masks = masks[:K]
    test_SegMasks = SegMasks[:K]

    training_masks = masks[K:]
    training_SegMasks = SegMasks[K:]

    K = math.floor(test_size*len(training_masks))
    val_masks = training_masks[:K]
    val_SegMasks = training_SegMasks[:K]
    training_masks = training_masks[K:]
    training_SegMasks = training_SegMasks[K:]


    print('dlugosc testMasks: ', test_masks.shape[0], test_SegMasks.shape[0])

    print('dlugosc valMasks: ', val_masks.shape[0], val_SegMasks.shape[0])
    print('dlugosc trainingMasks: ', training_masks.shape[0], training_SegMasks.shape[0])


    return test_masks, test_SegMasks, training_masks, training_SegMasks, val_masks, val_SegMasks






def run():
    figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/DenoisingAutoencoder5.json'
    trainingMonitor = TrainingMonitor(figpathJson, jsonPath=figpathJson)
    np.random.seed(1337)
    batch_size = 32
    fname = '/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Checkpoints/DenoisingAutoenkoder_5.hdf5'


    # labelsDIR1 = '/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/masks/'
    # labelsDIR2 = labelsDIR1
    labelsDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/' #maski binarne
    labelsDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/SegmentedImges/'      #maski binarne z Unet-a


    masks = wczytanieObrazokw(labelsDIR1=labelsDIR1,
                               masks=True)
    SegmentedMasks = wczytanieObrazokw(
        labelsDIR1=labelsDIR2,
        masks=True)

    testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy = prepareData(masks=masks,SegMasks=SegmentedMasks)

    autoencoder = binary()
    autoencoder.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])
    # autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mse')

    autoencoder.summary()
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)

    autoencoder.fit(trainingMasks_noisy,
                    trainingMasks,
                    validation_data=(valMasks_noisy, valMasks),
                    epochs=100,
                    callbacks=[trainingMonitor, checkpoint],
                    batch_size=batch_size)

    # predict the autoencoder output from corrupted test images

    autoencoder.save('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/DenoisingAutoenkoder_5')
    x_decoded = autoencoder.predict(testMasks_noisy)




if __name__ == '__main__':
    run()
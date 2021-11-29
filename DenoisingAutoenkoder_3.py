from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, \
    Reshape
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

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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
def wczytanieObrazokw2(labelsDIR1='Orgs/masks/' , masks=False):
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

def prepareData(masks, SegMasks):
    print("###########Przygotowywanie danych - dzielenie na zestawy testowy walidacyjny itp oraz dodanie szumu########")
    test_size = 0.1
    K = math.floor(test_size*len(masks))
    test_masks = masks[:K]
    test_SegMasks = SegMasks[:K]

    training_masks = masks[K:]
    training_SegMasks = SegMasks[K:]

    val_size = 0.3
    K = math.floor(test_size*len(training_masks))
    val_masks = training_masks[:K]
    val_SegMasks = training_SegMasks[:K]
    training_masks = training_masks[K:]
    training_SegMasks = training_SegMasks[K:]


    print('dlugosc testMasks: ', test_masks.shape[0], test_SegMasks.shape[0])

    print('dlugosc valMasks: ', val_masks.shape[0], val_SegMasks.shape[0])
    print('dlugosc trainingMasks: ', training_masks.shape[0], training_SegMasks.shape[0])


    return test_masks, test_SegMasks, training_masks, training_SegMasks, val_masks, val_SegMasks


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




def binary(pretrained_weights=None, inp_s=(128, 128, 1)):
    input_img = Input(shape=inp_s)

    x = Conv2D(16, 3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)

    # x = Conv2D(16, 3, strides=2, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # x = Conv2D(32, 3, strides=2, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(32, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
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

    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 3, strides=1, padding='same')(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    # model.summary()
    return model


def run():

    figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/DenoisingAutoencoder4.json'
    trainingMonitor = TrainingMonitor(figpathJson, jsonPath=figpathJson)
    np.random.seed(1337)

    masks = wczytanieObrazokw2(labelsDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/',
                               masks=True)
    SegmentedMasks = wczytanieObrazokw2(labelsDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/SegmentedImges/',
                               masks=True)
    # masks = wczytanieObrazok(labelsDIR1='Orgs/masks/',
    #                          labelsDIR2='Orgs/masks', masks=True)
    testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy = prepareData(masks=masks,SegMasks=SegmentedMasks)


    # print('wartosci pikseli w bitmapach: ', np.unique(masks))
    batch_size = 32

    autoencoder = binary()
    autoencoder.summary()

    autoencoder.fit(trainingMasks_noisy,
                    trainingMasks,
                    validation_data=(valMasks_noisy, valMasks),
                    epochs=100,
                    callbacks=trainingMonitor,
                    batch_size=batch_size)

    # predict the autoencoder output from corrupted test images
    autoencoder.save('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/DenoisingAutoenkoder_4')
    x_decoded = autoencoder.predict(testMasks_noisy)


    # 3 sets of images with 9 MNIST digits
    # 1st rows - original images
    # 2nd rows - images corrupted by noise
    # 3rd rows - denoised images
    rows, cols = 3, 9
    num = rows * cols
    imgs = np.concatenate([testMasks[:num], testMasks_noisy[:num], x_decoded[:num]])
    imgs = imgs.reshape((rows * 3, cols, 128, 128))
    imgs = np.vstack(np.split(imgs, rows, axis=1))
    imgs = imgs.reshape((rows * 3, -1, 128, 128))
    imgs = np.vstack([np.hstack(i) for i in imgs])

    imgs = (imgs * 255).astype(np.uint8)
    plt.figure()
    plt.axis('off')
    plt.title('Original images: top rows, '
              'Corrupted Input: middle rows, '
              'Denoised Input:  third rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/autoencoder4.jpg')

if __name__ == '__main__':
    run()
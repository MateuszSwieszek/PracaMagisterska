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
    trainingMasks, valMasks  = train_test_split(trainingMasks, test_size=0.3)
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
    model.compile(loss='mse', optimizer='adam')
    # model.summary()
    return model


def run():

    figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/DenoisingAutoencoder2.json'
    trainingMonitor = TrainingMonitor(figpathJson, jsonPath=figpathJson)
    np.random.seed(1337)

    masks = wczytanieObrazok(labelsDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/',
                             labelsDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/masks/', masks=True)
    # masks = wczytanieObrazok(labelsDIR1='Orgs/masks/',
    #                          labelsDIR2='Orgs/masks', masks=True)
    testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy = prepareData(masks)


    # print('wartosci pikseli w bitmapach: ', np.unique(masks))
    batch_size = 32

    autoencoder = binary()
    autoencoder.summary()

    autoencoder.fit(trainingMasks_noisy,
                    trainingMasks,
                    validation_data=(valMasks_noisy, valMasks),
                    epochs=10,
                    callbacks=trainingMonitor,
                    batch_size=batch_size)

    # predict the autoencoder output from corrupted test images
    autoencoder.save('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/DenoisingAutoenkoder_2')
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
    plt.savefig('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/autoencoder2.jpg')

if __name__ == '__main__':
    run()
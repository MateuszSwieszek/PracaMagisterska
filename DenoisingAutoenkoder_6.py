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



def wczytanieObrazokw(labelsDIR1='Orgs/masks/', masks=False):
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if (np.unique(image)[1] == 255):
            image = image[:, :, 0:1] / 255

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



# def wczytanieObrazokw(labelsDIR1='Orgs/masks/', masks=False):
#     '''wczytanie mask'''
#
#     images1 = glob.glob(labelsDIR1 + '*.bmp')
#     images1.sort()
#     print('################################################################################################')
#     print("Wczytywanie obrazow")
#     print('Dlugosc obrazow z pierwszej sciezki: ', len(images1))
#
#     imagesDIR = images1
#
#     images = np.zeros((len(imagesDIR), 128, 128), dtype=np.float32)
#     for im in range(len(imagesDIR)):
#         image = cv2.imread(imagesDIR[im])
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         if (np.unique(gray)[1] == 255):
#             gray = gray / 255
#
#         images[im] = gray
#
#     return images


def buildEncoder(input_shape=(128, 128)):

    # first build the encoder model
    input_img = Input(shape=input_shape, name='encoder_input')

    # 128x128
    x = Conv2D(16, 3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 64x64
    # 256x256
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 32x32
    # 128x128
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 16x16
    # #64x64
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(128, name='latent_vector')(x)
    x = Dense(64)(x)
    x = Dense(128)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)




    encoder = Model(input_img, x, name='encoder')
    encoder.summary()

    return encoder, shape, input_img


def buildDecoder(shape, layer_filters, kernel_size, latent_dim):
    # build the decoder model

    latent_inputs = Input(shape=(shape[1], shape[2], shape[3]), name='decoder_input')

    # 16x16
    encoded = Reshape((16, 16, 1))(latent_inputs)

    # 32x32
    x = UpSampling2D(size=(2, 2))(encoded)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

  #64x64
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

    decoder = Model(latent_inputs, decoded)


    decoder.summary()

    return decoder


def buildAutoencoder(input_shape, layer_filters, kernel_size, latent_dim):
    encoder, shape, inputs = buildEncoder(input_shape, layer_filters, kernel_size, latent_dim)

    decoder = buildDecoder(shape, layer_filters, kernel_size, latent_dim)

    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs, decoder(encoder(inputs)),
                        name='autoencoder')
    autoencoder.summary()
    # Mean Square Error (MSE) loss function, Adam optimizer
    autoencoder.compile(loss='mse', optimizer='adam')
    # train the autoencoder
    return autoencoder



def binary(pretrained_weights=None, inp_s=(128, 128, 1)):
    input_img = Input(shape=inp_s)

    #128x128
    x = Conv2D(16, 3, strides=1, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 64x64
    # 256x256
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)

    # 32x32
    # 128x128
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.15)(x)

    # 16x16
    # #64x64
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.15)(x)


    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    # 16x16
    encoded = Reshape((16, 16, 1))(x)


    # 16x16
    x = UpSampling2D(size=(2, 2))(encoded)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.15)(x)

    # 32x32
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.15)(x)

    # # 64x64
    # # 512x512
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(16, 3, strides=1, padding='same')(x)
    # x = Activation('relu')(x)

    # 128x128
    # 1024x1024
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
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

# def AugumentData(masks, SegMasks, BATCH_SIZE=16):
#     print('################################## Data Augumentation ##################################')
#     seed = 10
#     mask_datagen = ImageDataGenerator(shear_range=0.5, rotation_range=180, zoom_range=0.1, width_shift_range=0.01,
#                                        height_shift_range=0.05, fill_mode='nearest')
#     segmentedMasks_datagen = ImageDataGenerator(shear_range=0.5, rotation_range=180, zoom_range=0.1, width_shift_range=0.01,
#                                       height_shift_range=0.05, fill_mode='nearest')
#     # Keep the same seed for image and mask generators so they fit together
#
#     mask_datagen.fit(masks, augment=True, seed=seed)
#     segmentedMasks_datagen.fit(SegmentedMasks, augment=True, seed=seed)
#
#     masks = mask_datagen.flow(masks, batch_size=BATCH_SIZE, shuffle=False, seed=seed)
#     SegMasks = segmentedMasks_datagen.flow(masks, batch_size=BATCH_SIZE, shuffle=False, seed=seed)
#     print('DUPA',len(masks))
#     print('DUPA',len(SegmentedMasks))
#     return prepareData(masks=masks,
#                        SegMasks=SegMasks)





def run():
    figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/DenoisingAutoencoder6.json'
    trainingMonitor = TrainingMonitor(figpathJson, jsonPath=figpathJson)
    np.random.seed(1337)
    batch_size = 16
    fname = '/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Checkpoints/DenoisingAutoenkoder_6.hdf5'


    # labelsDIR1 = '/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/masks/'
    # labelsDIR2 = labelsDIR1
    labelsDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/'
    labelsDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/SegmentedImges/'


    masks = wczytanieObrazokw(labelsDIR1=labelsDIR1,
                               masks=True)
    SegmentedMasks = wczytanieObrazokw(
        labelsDIR1=labelsDIR2,
        masks=True)

    testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy = prepareData(masks=masks,SegMasks=SegmentedMasks)









    np.random.seed(1337)
    seed = 42
    # Creating the training Image and Mask generator
    datagen = ImageDataGenerator(shear_range=0.5, rotation_range=180, zoom_range=0.1, width_shift_range=0.01,
                                       height_shift_range=0.05, fill_mode='nearest')

    # Keep the same seed for image and mask generators so they fit together

    tr_masks_datagen = datagen.fit(trainingMasks, augment=True, seed=seed)
    tr_masks_noisy_datagen = datagen.fit(trainingMasks_noisy, augment=True, seed =seed)

    val_masks_datagen = datagen.fit(trainingMasks, augment=True, seed=seed)
    val_masks_noisy_datagen = datagen.fit(trainingMasks_noisy, augment=True, seed=seed)







    autoencoder = binary()
    autoencoder.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])
    # autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mse')

    autoencoder.summary()
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)

    autoencoder.fit(tr_masks_noisy_datagen.flow(trainingMasks_noisy, batch_size=32, shuffle=True, seed=seed),
                    tr_masks_datagen.flow(trainingMasks ,batch_size=32, shuffle=True, seed=seed),
                    validation_data=(val_masks_noisy_datagen.flow(valMasks_noisy, batch_size=32, shuffle=True, seed=seed),
                                                val_masks_datagen.flow(valMasks ,batch_size=32, shuffle=True, seed=seed)),
                    epochs=500,
                    callbacks=[trainingMonitor, checkpoint],
                    batch_size=batch_size)

    # predict the autoencoder output from corrupted test images

    autoencoder.save('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/DenoisingAutoenkoder_6')
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
    plt.savefig('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/autoencoder_6.jpg')


if __name__ == '__main__':
    run()
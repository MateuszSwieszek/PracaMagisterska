import os
import sys
import random
import warnings

import numpy as np
import glob
import cv2

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# from itertools import chain
# from skimage.io import imread, imshow, imread_collection, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label
#
# from keras.models import Model, load_model
# from keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import backend as K
#
# import tensorflow as tf

# Set some parameters
BATCH_SIZE = 5  # the higher the better
IMG_WIDTH = 128  # for faster computing on kaggle
IMG_HEIGHT = 128  # for faster computing on kaggle
# IMG_CHANNELS = 3


seed = 42


def showImage(image, isGray=False):
    '''funkcja  do wyświetlania obrazu'''
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.show()


def wczytanieObrazokw(labelsDIR1='Orgs/masks/', masks=False):
    '''wczytanie mask'''

    images1 = glob.glob(labelsDIR1 + '*.bmp')
    print(images1)
    images1.sort()
    print('################################################################################################')
    print("Wczytywanie obrazow")
    print('Dlugosc obrazow z pierwszej sciezki: ', labelsDIR1)

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


def run():
    np.random.seed(1337)

    orgs = wczytanieObrazokw(labelsDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/orgs/',
                             masks=False)
    masks = wczytanieObrazokw(labelsDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/masks/',
                              masks=True)
    print(orgs.shape)



    # Creating the training Image and Mask generator
    image_datagen = ImageDataGenerator(shear_range=0.5, rotation_range=180, zoom_range=0.1, width_shift_range=0.01,
                                       height_shift_range=0.05, fill_mode='nearest')
    mask_datagen = ImageDataGenerator(shear_range=0.5, rotation_range=180, zoom_range=0.1, width_shift_range=0.01,
                                      height_shift_range=0.05, fill_mode='nearest')
    # Keep the same seed for image and mask generators so they fit together

    image_datagen.fit(orgs, augment=True, seed=seed)
    masks = mask_datagen.fit(masks, augment=True)
    images = image_datagen.flow(orgs[:int(orgs.shape[0] * 0.5)], batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    print(len(masks))
    # i = 0
    # imgs = []
    # MAX = 5
    # for o in image_datagen.flow(orgs[:int(orgs.shape[0] * 0.5)], batch_size=BATCH_SIZE, shuffle=True, seed=seed):
    #     imgs.append(o)
    #     if (i > MAX):
    #         break
    #     i = 1 + i
    #
    # i = 0
    # ms = []
    # for m in mask_datagen.flow(masks[:int(masks.shape[0] * 0.5)], batch_size=BATCH_SIZE, shuffle=True, seed=100):
    #     ms.append(m)
    #     if (i > MAX):
    #         break
    #     i = 1 + i

    # imgs = np.asarray(imgs)
    # ms = np.asarray(ms)
    # print(np.shape(imgs))
    # print(np.shape(ms))
    #
    # for mini_batch_num, mini_batch in enumerate(imgs):
    #     fig = plt.figure(num=mini_batch_num, figsize=(np.shape(mini_batch)[0], 2))
    #     for img_num in range(0, len(mini_batch) * 2):
    #         fig.add_subplot(np.shape(mini_batch)[0], 2, img_num + 1)
    #         print(img_num // 2, img_num)
    #
    #         if img_num % 2 == 0:
    #             plt.imshow(mini_batch[img_num // 2])
    #         else:
    #             plt.imshow(ms[mini_batch_num][img_num // 2])
    #
    # plt.figure(10)
    #
    # for i in range(BATCH_SIZE):
    #     plt.subplot(BATCH_SIZE, 1, i + 1)
    #     plt.imshow(masks[i] * 254)
    # plt.show()



if __name__ == '__main__':
    run()

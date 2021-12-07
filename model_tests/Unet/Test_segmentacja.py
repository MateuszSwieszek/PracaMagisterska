#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
import cv2
from datetime import datetime
import argparse
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, \
    concatenate
from keras import backend as K
# import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from tensorflow.keras.callbacks import BaseLogger




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
            # plot the training loss and accuracy
            # dict_keys(['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error'])
        if len(self.H["loss"]) > 1:
            plt.style.use("ggplot")

            for i in self.H:            #

                plt.figure()
                plt.plot(list(map(float, self.H[i])), label=i)


            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.figPath)



data_gen_args = dict(
    rotation_range=10.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)



def showOpencvImage(image, isGray=False):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.show()


def get_augmented(
        X_train,
        Y_train,
        X_val=None,
        Y_val=None,
        batch_size=32,
        seed=0,
        data_gen_args=dict(
            rotation_range=10.,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zca_whitening=False,
            zca_epsilon=1e-6,
            shear_range=5,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        )):
    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen_1 = ImageDataGenerator(**data_gen_args)
    Y_datagen_2 = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_train_1 = Y_train[:, :, :, 0:1]
    Y_train_2 = Y_train[:, :, :, 1:2]
    Y_datagen_1.fit(Y_train_1, augment=True, seed=seed)
    Y_datagen_2.fit(Y_train_2, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented_1 = Y_datagen_1.flow(Y_train_1, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented_2 = Y_datagen_2.flow(Y_train_2, batch_size=batch_size, shuffle=True, seed=seed)

    train_generator = zip(X_train_augmented, Y_train_augmented_1, Y_train_augmented_2)  # , Y_train_augmented_3)
    return train_generator


def my_generator(
        X_train,
        Y_train,
        train_gen,
        X_val=None,
        Y_val=None,
        batch_size=2,
        seed=0,
        data_gen_args=dict(
            rotation_range=10.,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zca_whitening=False,
            zca_epsilon=1e-6,
            shear_range=5,
            zoom_range=0.3,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest'
        )):
    while 1:
        sample_batch = next(train_gen)
        xx, yy1, yy2 = sample_batch
        yy = np.zeros((xx.shape[0], xx.shape[1], xx.shape[2], 2), dtype=np.float32)
        yy[:, :, :, 0:1] = yy1
        yy[:, :, :, 1:2] = yy2
        #        yy[:,:,:,6:7] = yy3
        yield (xx, yy)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        activation='tanh',
        #    kernel_initializer='he_normal',
        kernel_initializer='glorot_uniform',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    return c


def my_custom_unet(
        input_shape,
        num_classes=1,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        filters=16,
        num_layers=4,
        output_activation='softmax'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=196)

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coeff(boxA, boxB):
    k = np.max(boxB)
    return np.sum(boxA[boxB == k]) * 2.0 / (np.sum(boxA) + np.sum(boxB))

print('args1')
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cv", required=True,
                help="batch id, from 0 to 4 included")
print('args2')

args = vars(ap.parse_args())


masks = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/*.bmp")
orgs = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/*.bmp")

masks.sort()
orgs.sort()
# masks = masks[:4000]
# orgs = orgs[:4000]
# print(len(masks))
# print(len(orgs))

imgs_list = []
masks_list = []

size = (128, 128)

# imgs_list = np.zeros(((len(masks),)+size))
# masks_list = np.zeros(((len(masks),)+size))

# print(masks_list.shape)
# print(imgs_list.shape)



for image, mask in zip(orgs, masks):
    im = cv2.imread(image)
    im = im[:, :, 0]
    imgs_list.append(im)

    im = cv2.imread(mask)
    im = im[:, :, 0]

    imMask = np.zeros((im.shape[0], im.shape[1], 2), dtype=np.float32)
    imMask[im == 0, 0] = 1  # background
    imMask[im != 0, 1] = 1  # spine

    masks_list.append(imMask)

# for iterator,(image, mask) in enumerate(zip(orgs, masks)):
#     im = cv2.imread(image)
#     imgs_list[iterator] = im[:, :, 0]
#
#     im = cv2.imread(mask)
#     im = im[:, :, 0]
#
#     im = cv2.imread(mask)
#     mas = im[:, :, 0]
#     #
#     #     imMask = np.zeros((im.shape[0], im.shape[1], 2), dtype=np.float32)
#     #     imMask[im == 0, 0] = 1  # background
#     #     imMask[im != 0, 1] = 1  # spine

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)
print(imgs_np.shape, masks_np.shape)

weights = np.ones((2), dtype=np.float32)
for i in range(0, 2):
    weights[i] = 1 / (np.sum(masks_np[:, :, :, i]) / (masks_np.shape[0] * masks_np.shape[1] * masks_np.shape[2]))

w = sum(weights)
weights = weights / w

print(weights)

print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)
y = np.asarray(masks_np, dtype=np.float32)
print(x.max(), y.max())
print(x.shape, y.shape)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 2)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape, y.shape)

step = x.shape[0] // 5
NUM = x.shape[0]

NUM_TRAIN = step * 4
NUM_VAL = step

a = args["cv"]
print('args["cv"', a)
cv = int(args["cv"])

st = cv*step
en = st + step

list_train = [i for i in np.arange(NUM) if i not in np.arange(st, en)]
# np.random.seed(10)
# np.random.shuffle(list_train)
list_test = [i for i in np.arange(NUM) if i in np.arange(st, en)]

list_val = list_train[:int(0.2 * len(list_train))]
list_train = list_train[int(0.2 * len(list_train)):]

print(list_train)
print(list_val)
print(list_test)

x_train = x[list_train]
x_val = x[list_val]
x_test = x[list_test]

y_train = y[list_train]
y_val = y[list_val]
y_test = y[list_test]

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

input_shape = x_train[0].shape

model = my_custom_unet(
    input_shape,
    num_classes=2,
    filters=64,
    output_activation='softmax',
    num_layers=4
)

model_filename = 'Unet' + str(cv) + '.h5'

callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True
)

model.compile(
    optimizer=Adam(lr=0.0001),
    loss=weighted_categorical_crossentropy(weights),
    metrics=[iou]
)

# model.summary()

figpath = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Plots/unet.png'
figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/unet.json'
trainingMonitor = TrainingMonitor(figpath, jsonPath=figpathJson)

train_gen = get_augmented(x_train, y_train, batch_size=2, data_gen_args=data_gen_args)
generator = my_generator(x_train, y_train, train_gen, batch_size=2, data_gen_args=data_gen_args)

# history = model.fit_generator(
#     generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=(x_val, y_val),
#     callbacks=[callback_checkpoint, trainingMonitor]
# )
# model.save_weights(model_filename)

model.load_weights(model_filename)
y_pred_unet = model.predict(x_test)
# for N in range(x_test.shape[0]):
#    y_pred = model.predict(x_test[N:N+1])
#    for k in range(1,2):
#        dum = y_pred[0,:,:,k]*255
#        cv2.imwrite('Mask_'+str(cv)+'_'+str(N)+'_'+str(k)+'_.bmp',dum)





unet_dice_coeff_list = []

for N in range(x_test.shape[0]):
    filename = 'Fold_'+str(cv)+'/Unet/seg_img'+str(N)+'.bmp'
    filename2 = 'Fold_' + str(cv) + '/Refs/ref_img_' + str(N) + '.bmp'
    print(filename2)

    # print('N: ', N, filename)
    unet_dice_coeff_list.append(dice_coeff(y_pred_unet[N, :, :, 1], y_test[N, :, :, 1]))

    cv2.imwrite(filename,y_pred_unet[N, :, :, 1]*255)
    cv2.imwrite(filename2,y_test[N, :, :, 1]*255)
print('y_pred_unet ', y_pred_unet.shape)


with open('Unet'+str(cv)+'.json', 'w') as f:
    json.dump({'Fold_'+str(cv):unet_dice_coeff_list}, f)

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate,Flatten,Dense,Activation,Reshape
from tensorflow.keras import backend as K
#import tensorflow as tf
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
import json
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
    plt.imshow(image, cmap = 'gray')
    plt.show()

def get_augmented(
    X_train, 
    Y_train, 
    X_val=None,
    Y_val=None,
    batch_size=32, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zca_whitening = False,
        zca_epsilon = 1e-6,
        shear_range=5,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    # Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    # Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    # Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = X_train_augmented
    return train_generator

def my_generator(
    X_train, 
    Y_train,
    train_gen,
    X_val=None,
    Y_val=None,
    batch_size=2, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zca_whitening = False,
        zca_epsilon = 1e-6,
        shear_range=5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest'
    )):
    while 1:
        sample_batch = next(train_gen)
        xx = sample_batch
        yield (xx, xx)


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    filters=16, 
    kernel_size=(3,3), 
    activation='tanh', 
#    kernel_initializer='he_normal', 
    kernel_initializer= 'glorot_uniform',
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    return c

def my_custom_auto_unet(
    input_shape,
    num_classes=1,
    upsample_mode='deconv', # 'deconv' or 'simple' 
    filters=16,
    num_layers=4,
    output_activation='softmax'): # 'sigmoid' or 'softmax'
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=196)

    shape = x.shape
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Reshape((shape[1], shape[2], 1))(x)

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

print("autounet")

ap = argparse.ArgumentParser()
ap.add_argument("-c","--cv", required=True,
	help="batch id, from 0 to 4 included")
args = vars(ap.parse_args())

# inputs = glob.glob("Orgs/orgs/*.bmp")
inputs = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/SegmentedImges/*.bmp")
masks = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/*.bmp")
# masks = glob.glob("Orgs/orgs/*.bmp")

inputs.sort()
masks.sort()
print(len(inputs))
print(len(masks))
inputs_list = []
masks_list = []
print(inputs[:10])
for inp in inputs[:]:
    im = cv2.imread(inp)
    im = im[:,:,0:1]/255
    inputs_list.append(im)

for mask in masks[:]:
    im = cv2.imread(mask)
    im = im[:,:,0:1]/255
    masks_list.append(im)

size = (128,128)

inputs_np = np.asarray(inputs_list) 
masks_np = np.asarray(masks_list)
x = np.asarray(inputs_np, dtype=np.float32)
y = np.asarray(masks_np, dtype=np.float32)
print('SHAPE ',x.shape,y.shape,', MIN ', np.min(x),np.min(y),', MAX ', np.max(x),np.max(y))


step = x.shape[0]//4
NUM = x.shape[0]

NUM_TRAIN = step*3
NUM_VAL = step


cv = int(args["cv"])
st = cv*step
en = st + step


list_train = [i for i in np.arange(NUM) if i not in np.arange(st,en)]

list_test =  [i for i in np.arange(NUM) if i in np.arange(st,en)]

list_val = list_train[:int(0.2*len(list_train))]
list_train = list_train[int(0.2*len(list_train)):]

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
print("x_val: ", x_val.shape)
print("x_test: ", x_test.shape)

input_shape = x_train[0].shape

model = my_custom_auto_unet(
    input_shape,
    num_classes=1,
    filters=64,
    output_activation='sigmoid',
    num_layers=4  
)

model_filename = 'AutoUnet_'+ str(cv) + '.hdf5'

callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=False,
    save_freq=10
)

model.compile(
    optimizer=Adam(lr=0.0001), 
    loss = 'mse',
    metrics=['mse']
)

model.summary()

batch_size = 4
train_gen = get_augmented(x_train, y_train, batch_size=batch_size,data_gen_args=data_gen_args)
generator = my_generator(x_train, y_train,train_gen, batch_size=batch_size,data_gen_args=data_gen_args)

figpath = f'/net/people/plgmswieszek/segm_ALL_' + str(cv) + '.png'
figpathJson = f'/net/people/plgmswieszek/segm_ALL_' + str(cv) + '.json'
trainingMonitor = TrainingMonitor(figpath, jsonPath=figpathJson)

model.load_weights(model_filename)

history = model.fit(
   x_train,x_train,

   epochs=100,
   validation_data=(x_val, y_val),
   callbacks=[callback_checkpoint, trainingMonitor]
)
model.save_weights(model_filename)
# model.load_weights(model_filename)

# for N in range(x_test.shape[0]):
#    x_pred = model.predict(x_test[N:N+1])
#    dum = x_pred[0,:,:,0]*255
#    cv2.imwrite('./tmp/Pred_'+str(cv)+'_'+str(N)+'_'+ '_.bmp',dum)
#    cv2.imwrite('./tmp/Mask_'+str(cv)+'_'+str(N)+'_'+ '_.bmp',y_test[N]*255)
#    if N>10:
#        break

# newInputs = glob.glob("/home/user/Sarkopenia/Segmentacja_L4/Noisy_1_epoch/*.bmp")
#
# for inp in newInputs:
#     im = cv2.imread(inp)
#     im = im[:,:,0:1]/255
#     im = np.reshape(im,(1,) + im.shape)
#     x_pred = model.predict(im)
#     basename = os.path.basename(inp)
#     cv2.imwrite('./tmp/Pred_'+basename,x_pred[0,:,:,0]*255)


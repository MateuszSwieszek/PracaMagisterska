import numpy as np
import glob
# import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def Dice(yt, yp, TH=0.5):
    return 2 * np.sum((yt > TH) * (yp > TH)) / (np.sum(yt > TH) + np.sum(yp > TH))

def showImage(image, isGray=False):
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.show()

im1 = cv2.imread("mask_alignedImg_0.bmp")/255
im2 = cv2.imread("seg_image_0.bmp")/255

print(Dice(im1,im2))

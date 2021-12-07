# import the necessary packages
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import cv2
import glob
import argparse
from random import sample
import tensorflow as tf
from tensorflow.keras.models import load_model



class Modelstatistics():
    def __init__(self, masksDIR1='../Orgs/orgs/', masksDIR2='../Orgs/orgs/', masksDIR3='../Orgs/orgs/'):
        self.masksDIR1 = self.wczytanieSciezek(labelsDIR1=masksDIR1)
        self.masksDIR2 = self.wczytanieSciezek(labelsDIR1=masksDIR2)
        self.masksDIR3 = self.wczytanieSciezek(labelsDIR1=masksDIR3)

        self.masks1 = self.loadImages(self.masksDIR1)
        self.masks2 = self.loadImages(self.masksDIR2)
        self.masks3 = self.loadImages(self.masksDIR3)
        self.diceCoeffList12 = []
        self.diceCoeffList13 = []
        self.diceCoeffList23 = []

    def wczytanieSciezek(self, labelsDIR1='Orgs/masks/'):

        images1 = glob.glob(labelsDIR1 + '*.bmp')
        # print(labelsDIR1)
        # print("images1: ", images1)
        images1.sort()
        print('################################################################################################')
        print("Wczytywanie obrazow")
        print('Dlugosc obrazow z pierwszej sciezki: ', len(images1))

        imagesDIR = images1

        return imagesDIR

    def loadImages(self, imagesDIR):
        images = np.zeros((len(imagesDIR), 128, 128, 1), dtype=np.float32)
        for im in range(len(imagesDIR)):
            image = cv2.imread(imagesDIR[im])
            print('image.shape', image.shape)

            if (np.max(image) == 255):
                image = image[:, :, 0:1] / 255
            print('image.shape', image.shape)

            images[im] = image

        return images

    def dice_coeff(self, boxA, boxB):
        k = np.max(boxB)
        return np.sum(boxA[boxB == k]) * 2.0 / (np.sum(boxA) + np.sum(boxB))

    def run(self):
        if not len(self.masks1 == self.masks2):
            print("ERROR")

        for (im1, im2, im3) in zip(self.masks1, self.masks2, self.masks3):
            self.diceCoeffList12.append(self.dice_coeff(im1, im2))
            self.diceCoeffList13.append(self.dice_coeff(im1, im3))
            self.diceCoeffList23.append(self.dice_coeff(im2, im3))

        print('###############################################################')
        print('List: 1,2')
        print('min: ', np.min(self.diceCoeffList12))
        print('max: ', np.max(self.diceCoeffList12))
        print('mean: ', np.mean(self.diceCoeffList12))
        print('standard deviation: ', np.std(self.diceCoeffList12))

        print('###############################################################')
        print('List: 1,3')
        print('min: ', np.min(self.diceCoeffList13))
        print('max: ', np.max(self.diceCoeffList13))
        print('mean: ', np.mean(self.diceCoeffList13))
        print('standard deviation: ', np.std(self.diceCoeffList13))

        print('###############################################################')
        print('List: 2,3')
        print('min: ', np.min(self.diceCoeffList23))
        print('max: ', np.max(self.diceCoeffList23))
        print('mean: ', np.mean(self.diceCoeffList23))
        print('standard deviation: ', np.std(self.diceCoeffList23))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cv", required=True,
                    help="batch id, from 0 to 4 included")
    args = vars(ap.parse_args())
    cv = int(args["cv"])
    print(cv)

    masksDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/model_tests/Unet/'+'Fold_'+str(cv)+'/Refs/'
    masksDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/model_tests/Unet/'+'Fold_'+str(cv)+'/Unet/'
    masksDIR3 = '/net/people/plgmswieszek/GenerowanieObrazkow/model_tests/Unet/'+'Fold_'+str(cv)+'/AutoUnet/'
    print(cv)
    modelstatistics = Modelstatistics(masksDIR1, masksDIR2, masksDIR3)
    print(cv)
    modelstatistics.run()
    print(cv)

if __name__ == '__main__':
    main()







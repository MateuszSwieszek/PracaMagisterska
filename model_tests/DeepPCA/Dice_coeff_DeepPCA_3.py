# import the necessary packages
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import cv2
import glob
from DeepPca_03 import BuildModel
from sklearn.model_selection import train_test_split
from random import sample
import tensorflow as tf
from tensorflow.keras.models import load_model


class Modelstatistics_3():
    def __init__(self, modelfile, masksDIR1='../Orgs/orgs/', masksDIR2='../Orgs/orgs/',
                 imagesDIR1='../Orgs/masks/', imagesDIR2='../Orgs/masks/',
                 pointsDIR1='../Orgs/masks/', pointsDIR2='../Orgs/masks/',
                 PCAfile='../PCA.dat', meanFile='../mean.dat', BATCHSIZE=5):
        self.modelfile = modelfile

        self.pcaComponents = self.ReadPCA(name=PCAfile)
        self.mean = self.readMean(name=meanFile)
        self.load_model()

        self.masksDIR = self.wczytanieSciezek(labelsDIR1=masksDIR1, labelsDIR2=masksDIR2)
        self.imagesDIR = self.wczytanieSciezek(labelsDIR1=imagesDIR1, labelsDIR2=imagesDIR2)
        self.pointsDIR = self.wczytanieSciezek_punktow(labelsDIR1=pointsDIR1, labelsDIR2=pointsDIR2)

        L = len(self.imagesDIR)
        training_set = int(L - (L * 0.02))

        self.indexes = np.arange(training_set, L)
        self.indexes.sort()
        self.points = self.loadPoints([self.pointsDIR[i] for i in self.indexes])


        self.masks = self.loadImages([self.masksDIR[i] for i in self.indexes])
        self.images = self.loadImages([self.imagesDIR[i] for i in self.indexes])
        self.reconImages = []
        self.imagesFromPoints = []
        self.diceCoeffList = []
        self.calculateReconPoints()

    def load_model(self):
        self.model = tf.keras.models.load_model(self.modelfile, custom_objects={'tf': tf})

    def wczytanieSciezek(self, labelsDIR1='Orgs/masks/', labelsDIR2='Generated/masks/'):
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

        return imagesDIR

    def wczytanieSciezek_punktow(self, labelsDIR1='Orgs/masks/', labelsDIR2='Generated/masks/'):
        '''wczytanie mask'''

        images1 = glob.glob(labelsDIR1 + '*.dat')
        images1.sort()
        print('################################################################################################')
        print("Wczytywanie punktow")
        print('Dlugosc punkt z pierwszej sciezki: ', len(images1))

        images2 = glob.glob(labelsDIR2 + '*.dat')
        images2.sort()
        print('Dlugosc punkt z drugiej sciezki: ', len(images2))
        imagesDIR = images1 + images2

        return imagesDIR

    def randomChoices(self, LENGTH=10, n=2):
        choices = np.arange(LENGTH)
        return sample(list(choices), n)

    def loadImages(self, imagesDIR):

        images = np.zeros((len(imagesDIR), 128, 128, 1), dtype=np.float32)
        for im in range(len(imagesDIR)):
            image = cv2.imread(imagesDIR[im])

            if (np.max(image) == 255):
                image = image[:, :, 0:1] / 255

            images[im] = image

        return images

    def loadPoints(self, POINTSDIR):
        print('################################################################################################')
        print("Wczytywanie punktow")

        FlattenXY = np.zeros((len(POINTSDIR), 200), dtype=np.float32)

        for i, f in enumerate(POINTSDIR):
            f = open(f, 'r')
            lines = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
            f.close()
            lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
            FlattenXY[i] = lines

        return FlattenXY

    def calculateReconPoints(self):
        X_recon, Y_recon = self.model.predict(self.images)
        for iterator, (x, y) in enumerate(zip(X_recon, Y_recon)):
            reconImage = self.generate_mask(x, y)
            X = self.points[iterator][0::2]
            Y = self.points[iterator][1::2]
            imageFromPoints = self.generate_mask(X, Y)
            self.imagesFromPoints.append(imageFromPoints)
            self.reconImages.append(reconImage)
            self.diceCoeffList.append(self.dice_coeff(imageFromPoints, reconImage))
        self.PCA_results(self.points[0], X_recon[0], Y_recon[0], self.images[0])

    def PCA_results(self, points, X, Y, image):
        print(image.shape)
        image = image * 255

        im = np.zeros((128, 128, 3))

        im[:, :, 0] = image[:, :, 0]
        im[:, :, 1] = image[:, :, 0]
        im[:, :, 2] = image[:, :, 0]
        print(im.shape)
        print(image[0].shape)

        for p in range(100):
            im[int(X[p]), int(Y[p]), 0] = 200
            im[int(X[p]), int(Y[p]), 1] = 0
            im[int(X[p]), int(Y[p]), 2] = 0

        X = points[0::2]
        Y = points[1::2]
        for p in range(100):
            im[int(X[p]), int(Y[p]), 0] = 0
            im[int(X[p]), int(Y[p]), 1] = 0
            im[int(X[p]), int(Y[p]), 2] = 139

        cv2.imwrite('test' + '.png', im)

    def ReadPCA(self, N_COMP=52, name='PCA.dat'):
        print('################################################################################################')
        print("odczyt PCA z pliku ")

        pcaComponents = np.zeros((N_COMP, 200), dtype=np.float32)
        f = open(name, 'r')
        for r in range(pcaComponents.shape[0]):
            for c in range(pcaComponents.shape[1]):
                el = float(f.readline())
                pcaComponents[r, c] = el
        f.close()
        print('shape PCA: ', np.shape(pcaComponents))
        return pcaComponents

    def readMean(self, name='mean.dat'):
        N_FEATURES = 200
        av = []
        f = open(name, 'r')  # sredni ksztalt wyliczony w contours
        for row in range(N_FEATURES):
            av.append(float(f.readline()))
        f.close()
        av = np.asarray(av, dtype=np.float32).reshape(-1, N_FEATURES)
        return av

    def PCAInverseTransform(self, XYinPCA):

        return np.matmul(XYinPCA, self.pcaComponents) + self.mean

    def generate_mask(self, pointsX, pointsY):

        XY = np.array([pointsY, pointsX]).transpose().astype(int)
        img = np.zeros((128, 128))  # create a single channel 200x200 pixel black image
        cv2.fillPoly(img, pts=[XY], color=(255, 255, 255))
        return img

    def dice_coeff(self, boxA, boxB):
        k = np.max(boxB)
        return np.sum(boxA[boxB == k]) * 2.0 / (np.sum(boxA) + np.sum(boxB))

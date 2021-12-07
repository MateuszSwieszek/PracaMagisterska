# import the necessary packages
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import cv2
import glob
from DeepPCA_02 import BuildModel
from sklearn.model_selection import train_test_split
from random import sample
import tensorflow as tf


class Modelstatistics():
    def __init__(self, modelfile, masksDIR1='../Orgs/orgs/', masksDIR2='../Orgs/orgs/',
                 imagesDIR1='Orgs/masks/', imagesDIR2='../Orgs/masks/',
                 pointsDIR1='Orgs/masks/', pointsDIR2='../Orgs/masks/',
                 PCAfile='../PCA.dat', meanFile='../mean.dat', BATCHSIZE = 5):
        self.modelfile = modelfile
        self.load_model()

        self.masksDIR = self.wczytanieSciezek(labelsDIR1=masksDIR1, labelsDIR2=masksDIR2)
        self.imagesDIR = self.wczytanieSciezek(labelsDIR1=imagesDIR1, labelsDIR2=imagesDIR2)
        self.indexes = self.randomChoices(len(self.masksDIR), BATCHSIZE)

        L = len(self.imagesDIR)
        training_set = int(L - (L * 0.02))

        self.indexes = np.arange(training_set, L)
        self.pcaComponents = self.ReadPCA(name=PCAfile)
        self.mean = self.readMean(name=meanFile)

        self.points = self.loadPoints(pointsDIR1, pointsDIR2)
        self.masks = self.loadImages([self.masksDIR[i] for i in self.indexes])
        self.images = self.loadImages([self.imagesDIR[i] for i in self.indexes])
        self.reconImages = []
        self.imagesFromPoints = []
        self.diceCoeffList = []
        self.calculateReconPoints()


    def load_model(self):
        self.model = BuildModel(NUM_LAYERS=4, FILTERS=16,
                                NUM_ELEMENTS=1000, size=(128, 128), N_COMP=52)

        self.model.load_weights(self.modelfile)

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

    def loadPoints(self, POINTSDIR1, POINTSDIR2):
        print('################################################################################################')
        print("Wczytywanie punktow")

        points1 = glob.glob(POINTSDIR1 + '*.dat')
        points1.sort()

        points2 = glob.glob(POINTSDIR2 + '*.dat')
        points2.sort()

        points = points1 + points2

        points = [points[i] for i in self.indexes]

        FlattenXY = np.zeros((len(points), 200), dtype=np.float32)

        for i, f in enumerate(points):
            f = open(f, 'r')
            lines = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
            f.close()
            lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
            FlattenXY[i] = lines

        return FlattenXY

    def calculateReconPoints(self):
        W, M1, M2 = self.model.predict(self.images)
        for iterator,( w, m1, m2) in enumerate(zip(W, M1, M2)):
            result = self.PCAInverseTransform(w)

            resultX = result[0][0::2]
            resultY = result[0][1::2]

            finalresultY = np.asarray([X * m2[0] + Y * m2[1] + m2[2] for X, Y in zip(resultX, resultY)],
                                      dtype=np.float32)
            finalresultX = np.asarray([X * m1[0] + Y * m1[1] + m1[2] for X, Y in zip(resultX, resultY)],
                                      dtype=np.float32)
            reconImage = self.generate_mask(finalresultX, finalresultY)
            X = self.points[iterator][0::2]
            Y = self.points[iterator][1::2]
            imageFromPoints = self.generate_mask(X,Y)
            self.imagesFromPoints.append(imageFromPoints)
            self.reconImages.append(reconImage)
            self.diceCoeffList.append(self.dice_coeff(imageFromPoints, reconImage))
            X_recon = finalresultX
            Y_recon = finalresultY
        self.PCA_results(self.points[0], X_recon, Y_recon, self.images[0])


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

        cv2.imwrite('test_2' + '.png', im)

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



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import BaseLogger
import glob
import cv2
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense, Lambda, \
    Add, Multiply
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from pycimg import CImg
import json
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm.keras import TqdmCallback

# import the necessary packages
# from sklearn.preprocessing import LabelBinarizer
# from pyimagesearch.nn.conv import MiniVGGNet
# 9 from keras.optimizers import SGD
# 10 from keras.datasets import cifar10
import os


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


def showImage(im, isGray=False):
    '''funkcja  do wyswietlania obrazu'''
    img = CImg(im)
    img.display()


def wczytanieObrazok(labelsDIR1='Orgs/masks/', labelsDIR2='Generated/masks/'):
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
        image = image[:, :, 0:1] / 255
        images[im] = image

    return images


def WczytywaniePunktow(POINTSDIR1, POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktow")

    points1 = glob.glob(POINTSDIR1 + '*.dat')
    points1.sort()

    points2 = glob.glob(POINTSDIR2 + '*.dat')
    points2.sort()

    points = points1 + points2

    print('Dlugosc punktow z pierwszej sciezki: ', len(points1))
    print('Dlugosc punktow z drugiej sciezki: ', len(points2))
    XY = np.zeros((len(points), 100, 2), dtype=np.float32)
    FlattenXY = np.zeros((len(points), 200), dtype=np.float32)

    for i, f in enumerate(points):
        f = open(f, 'r')
        lines = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
        f.close()
        XY[i] = lines
        lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
        FlattenXY[i] = lines
        # print("i: ",i)
        # X_ = XY_[:, 0].reshape(-1, 100)
        # Y_ = XY_[:, 1].reshape(-1, 100)
        # X[i] = X_
        # Y[i] = Y_

    print("wymiar XY: ", np.shape(XY))

    return XY, FlattenXY


def WczytywaniePunktowSTD(POINTSDIR1, POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktow std")

    stdpoints1 = glob.glob(POINTSDIR1 + '*.dat')
    stdpoints1.sort()
    print(len(stdpoints1))

    stdpoints2 = glob.glob(POINTSDIR2 + '*.dat')
    stdpoints2.sort()
    print(len(stdpoints2))
    stdpoints = stdpoints1 + stdpoints2

    print('Dlugosc punktow std z pierwszej sciezki: ', len(stdpoints1))
    print('Dlugosc punktow std z drugiej sciezki: ', len(stdpoints2))
    # stdX = np.zeros((len(stdpoints), 100), dtype=np.float32)
    # stdY = np.zeros((len(stdpoints), 100), dtype=np.float32)

    XY = np.zeros((len(stdpoints), 100, 2), dtype=np.float32)
    FlattenXY = np.zeros((len(stdpoints), 200), dtype=np.float32)
    for i, f in enumerate(stdpoints):
        file = open(f, 'r')
        lines = np.array([list(map(float, l.split(' ')[1:])) for l in file.read().splitlines()])
        file.close()
        XY[i] = lines
        lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
        FlattenXY[i] = lines

    return XY, FlattenXY


def WczytywaniePunktowHomography(POINTSDIR1, POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktow macierzy homografii")

    hpoints1 = glob.glob(POINTSDIR1 + '*.dat')
    hpoints1.sort()
    print(len(hpoints1))

    hpoints2 = glob.glob(POINTSDIR2 + '*.dat')
    hpoints2.sort()
    print(len(hpoints2))
    hpoints = hpoints1 + hpoints2
    print('Dlugosc punktow h z pierwszej sciezki: ', len(hpoints1))
    print('Dlugosc punktow h z drugiej sciezki: ', len(hpoints2))

    M = np.zeros((len(hpoints), 3, 3), dtype=np.float32)
    for i, m in enumerate(hpoints):

        file = open(m, 'r')
        for r in range(M.shape[1]):
            for c in range(M.shape[2]):
                el = float(file.readline())
                M[i, r, c] = el

        file.close()

    return M




def conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='glorot_uniform',
        padding='same'):
    conv2d_block = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                          padding=padding)(
        inputs)
    conv2d_block = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                          padding=padding)(conv2d_block)
    conv2d_block = Dropout(rate=0.15)(conv2d_block)
    return conv2d_block


def linear(PointsInPCAspace, PCAcomponents, averagePointVal, N_COMP, N_FEATURES=200):
    PCAcomponents = tf.constant(PCAcomponents, shape=(N_COMP, N_FEATURES // 2))
    averagePointVal = tf.constant(averagePointVal, shape=(1, N_FEATURES // 2))
    stdReconPoints = tf.matmul(PointsInPCAspace, PCAcomponents) + averagePointVal

    return stdReconPoints


def stackLayer(layer1):
    output = tf.transpose(tf.stack([layer1[0], layer1[1]]), perm=[1, 0, 2])

    return output


def BuildModel(NUM_LAYERS=4, FILTERS=16, NUM_ELEMENTS=1000,
               size=(128, 128), N_COMP=52):
    # avX = averagePoints[:, 0::2]
    # avY = averagePoints[:, 1::2]

    # PCAcompX = PCAcomp[:, 0::2]
    # PCAcompY = PCAcomp[:, 1::2]

    inputs = Input(size + (1,))
    x = inputs

    ################################TESTY########################
    '''tu byly testy i okazalo sie ze siec dziala'''
    # inputs2 = Input((100))
    # X_ = inputs2
    # inputs3 = Input((100))
    # Y_ = inputs3
    #
    # inputs4 = Input((1))
    # inputs5 = Input((1))
    # inputs6 = Input((1))
    # inputs7 = Input((1))
    # inputs8 = Input((1))
    # inputs9 = Input((1))
    #
    # X1 = inputs4
    # Y1 = inputs5
    # Z1 = inputs6
    # X2 = inputs7 # wsp dla drugiego wiersza macierzy homografii (wsp y)
    # Y2 = inputs8
    # Z2 = inputs9
    #
    # outX = Add()([Multiply()([X1, X_]), Multiply()([Y1, Y_]), Z1])
    # outY = Add()([Multiply()([X2, X_]), Multiply()([Y2, Y_]), Z2])

    ################################TESTY########################

    for l in range(NUM_LAYERS):
        x = conv2d_block(inputs=x, filters=FILTERS)
        x = MaxPooling2D((2, 2))(x)
        FILTERS = FILTERS * 2

    x = Flatten()(x)

    # x = Dense(NUM_ELEMENTS, activation='relu')(x)
    # x = Dense(NUM_ELEMENTS, activation='relu')(x)

    w = Dense(NUM_ELEMENTS)(x)
    w = Dense(NUM_ELEMENTS / 2)(w)
    w = Dense(N_COMP,name='w')(w)

    m1 = Dense(NUM_ELEMENTS / 2)(x)
    m1 = Dense(NUM_ELEMENTS / 4)(m1)
    m1 = Dense(3,name='m1')(m1)

    m2 = Dense(NUM_ELEMENTS / 2)(x)
    m2 = Dense(NUM_ELEMENTS / 4)(m2)
    m2 = Dense(3,name='m2')(m2)

    # X1 = Dense(1)(x)  # wsp pierwszego wiersza macierzy homografii (wsp x)
    # Y1 = Dense(1)(x)
    # Z1 = Dense(1)(x)
    # X2 = Dense(1)(x)  # wsp dla drugiego wiersza macierzy homografii (wsp y)
    # Y2 = Dense(1)(x)
    # Z2 = Dense(1)(x)

    # X_ = Lambda(linear, arguments={'PCAcomponents': PCAcompX, 'averagePointVal': avX, 'N_COMP': N_COMP})(w)
    # Y_ = Lambda(linear, arguments={'PCAcomponents': PCAcompY, 'averagePointVal': avY, 'N_COMP': N_COMP})(w)

    # outX = Add()([Multiply()([X1, X_]), Multiply()([Y1, Y_]), Z1])
    # outY = Add()([Multiply()([X2, X_]), Multiply()([Y2, Y_]), Z2])

    # outXY = Lambda(stackLayer)([outX, outY])
    # outXY = Lambda(stackLayer)([X_, Y_])

    model = Model(inputs=[inputs], outputs=[w, m1, m2])

    model.compile(optimizer=Adam(lr=0.001),
                  loss={'w': 'mse',
                        'm1': 'mse',
                        'm2': 'mse'},
                  loss_weights={'w': 2,
                        'm1': 1,
                        'm2': 1}
                  )
    # model.summary()

    return model


def ReadPCA(N_COMP):
    print('################################################################################################')
    print("odczyt PCA z pliku ")

    name = 'PCA.dat'
    pcaComponents = np.zeros((N_COMP, 200), dtype=np.float32)
    f = open(name, 'r')
    for r in range(pcaComponents.shape[0]):
        for c in range(pcaComponents.shape[1]):
            el = float(f.readline())
            pcaComponents[r, c] = el
    f.close()
    print('shape PCA: ', np.shape(pcaComponents))
    return pcaComponents


# def SplitDataSet(W, M1, M2, images):
#     splitIter = math.floor(len(images) * 0.1)
#
#     testX = X[:splitIter]
#     testY = Y[:splitIter]
#     testImages = images[:splitIter]
#
#     trainingX = X[splitIter:]
#     trainingY = Y[splitIter:]
#     trainingImages = images[splitIter:]
#
#     return testX, testY, testImages, trainingX, trainingY, trainingImages

class TqdmCallbackFix(TqdmCallback):
    def _implements_train_batch_hooks(self): return True
    def _implements_test_batch_hooks(self): return True
    def _implements_predict_batch_hooks(self): return True

def CrossValidation(trainingW, trainingM1, trainingM2, trainingImages, N_COMP,
                    callbacks=None, NUM_LAYERS=4, FILTERS=16,
                    NUM_ELEMENTS=1000, size=(128, 128), K=4):
    num_val_samples = len(trainingImages) // K

    # all_mae_histories = []
    for i in range(K):
        print('processing fold #', i)
        valW = trainingW[i * num_val_samples: (i + 1) * num_val_samples]
        valM1 = trainingM1[i * num_val_samples: (i + 1) * num_val_samples]
        valM2 = trainingM2[i * num_val_samples: (i + 1) * num_val_samples]

        valImages = trainingImages[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_W = np.concatenate([trainingW[:i * num_val_samples], trainingW[(i + 1) * num_val_samples:]],
                                         axis=0)
        partial_train_M1 = np.concatenate([trainingM1[:i * num_val_samples], trainingM1[(i + 1) * num_val_samples:]],
                                          axis=0)

        partial_train_M2 = np.concatenate([trainingM2[:i * num_val_samples], trainingM2[(i + 1) * num_val_samples:]],
                                          axis=0)

        partial_train_images = np.concatenate(
            [trainingImages[:i * num_val_samples], trainingImages[(i + 1) * num_val_samples:]], axis=0)

        # print('TO JEST OK: input samples shape: ', np.shape(valImages))
        # print('output samples shape: ', np.shape(valXY))

        model = BuildModel(NUM_LAYERS=4, FILTERS=16,
                           NUM_ELEMENTS=1000, size=(128, 128), N_COMP=N_COMP)

        fname = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Checkpoints/checkpoints_K-{i:04d}'
        fname = fname + '.hdf5'

        checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                     save_best_only=True, verbose=1)

        figpath = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Plots/modelProgres_K-{i:04d}.png'
        figpathJson = f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/JsonFiles/ModelProgres_K-{i:04d}.json'
        trainingMonitor = TrainingMonitor(figpath, jsonPath=figpathJson)
        callbacks = [trainingMonitor, checkpoint, TqdmCallbackFix(verbose=1)]

        history = model.fit([partial_train_images], [partial_train_W, partial_train_M1, partial_train_M2],
                            validation_data=([valImages], [valW, valM1, valM2]), epochs=300, callbacks=callbacks,
                            batch_size=16,
                            verbose=1)
        model.save_weights(f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Checkpoints/Models/DeeppPCA_2-{i:04d}.hdf5')
        # model.save(f'/Users/mateusz/PycharmProjects/PracaMagisterska/Checkpoints/model-{i:02d}')

        # score = model.evaluate(testImages, testW, testM1, testM2)
        # with open(f'/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Evaluations/evaluation-{i:04d}',
        #           'w') as fp:
        #     json.dump({"test_loss": str(score[0]), "test_val": str(score[1])}, fp)

        # mae_history = history.history['val_mean_absolute_percentage_error']
        # all_mae_histories.append(mae_history)


def PCATransform(XY, pcaComponents, mean, N_COMP):
    XYinPCA = np.zeros((XY.shape[0], N_COMP), dtype=np.float32)
    print("xy shape: ", XY.shape)
    for i, xy in enumerate(XY):
        XYinPCA[i] = np.matmul((xy - mean), np.transpose(pcaComponents))

    return XYinPCA


def PCAInverseTransform(XYinPCA, pcaComponents, mean):
    XYrecon = np.zeros((XYinPCA.shape[0], pcaComponents.shape[1]), dtype=np.float32)
    for i, xypca in enumerate(XYinPCA):
        XYrecon[i] = np.matmul(xypca, pcaComponents) + mean
    return XYrecon


def main():
    images = wczytanieObrazok('/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/masks/',
                              '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/')

    XY, flattenXY = WczytywaniePunktow('/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/points/',
                              '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/points/', )

    stdXY, stdXYflatten = WczytywaniePunktowSTD('/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/stdPoints/',
                                       '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/stdPoints/')

    M = WczytywaniePunktowHomography(
        '/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/homographyPoints/',
        '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/homographyPoints/')


    M1 = M[:, 0]
    M2 = M[:, 1]

    N_FEATURES = 200
    av = []
    f = open('mean.dat', 'r')  # sredni ksztalt wyliczony w contours
    for row in range(N_FEATURES):
        av.append(float(f.readline()))
    f.close()
    av = np.asarray(av, dtype=np.float32).reshape(-1, N_FEATURES)

    N_COMP = 52
    PCAcomponents = ReadPCA(N_COMP)

    W = PCATransform(stdXYflatten, PCAcomponents, av, N_COMP)
    XYrecon = PCAInverseTransform(W, PCAcomponents, av)

    L = len(images)
    training_set = int(L - (L * 0.02))
    XY = XY[:training_set]
    trainingImages = images[:training_set]
    trainingW = W[:training_set]
    trainingM1 = M1[:training_set]
    trainingM2 = M2[:training_set]



    CrossValidation(trainingW, trainingM1, trainingM2, trainingImages, N_COMP)


if __name__ == '__main__':
    main()
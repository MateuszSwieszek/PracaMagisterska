from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
    trainingMasks, testMasks = train_test_split(masks, test_size=0.2)
    print('dlugosc testMasks: ', testMasks.shape[0])
    trainingMasks, valMasks  = train_test_split(trainingMasks, test_size=0.5)
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


def buildEncoder(input_shape, layer_filters, kernel_size, latent_dim):
    # build the autoencoder model

    # first build the encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # stack of Conv2D(32)-Conv2D(64)
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   padding='same')(x)
    # shape info needed to build decoder model so we don't do hand
    # computation
    # the input to the decoder's first Conv2DTranspose will have this
    # shape
    # shape is (7, 7, 64) which can be processed by the decoder back to
    # (28, 28, 1)
    shape = K.int_shape(x)
    # generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    # instantiate encoder model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    return encoder, shape, inputs


def buildDecoder(shape, layer_filters, kernel_size, latent_dim):
    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    # use the shape (7, 7, 64) that was earlier saved
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    # from vector to suitable shape for transposed conv
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            padding='same')(x)
    # reconstruct the denoised input
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              padding='same',
                              activation='sigmoid',
                              name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
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


def main():
    np.random.seed(1337)

    masks = wczytanieObrazok(labelsDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/',
                             labelsDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/masks/', masks=True)
    testMasks, testMasks_noisy, trainingMasks, trainingMasks_noisy, valMasks, valMasks_noisy = prepareData(masks)
    print('wartosci pikseli w bitmapach: ', np.unique(masks))


    # network parameters
    input_shape = (128,128,1)
    batch_size = 16
    kernel_size = 3
    latent_dim = 32
    # encoder/decoder number of CNN layers and filters per layer
    layer_filters = [16, 32, 64]

    autoencoder = buildAutoencoder(input_shape, layer_filters, kernel_size, latent_dim)
    autoencoder.summary()

    autoencoder.fit((trainingMasks_noisy, trainingMasks),
                    validation_data=(valMasks_noisy, valMasks),
                    epochs=50,
                    batch_size=batch_size)
    # predict the autoencoder output from corrupted test images
    x_decoded = autoencoder.predict(testMasks_noisy)
    autoencoder.save('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/DenoisingAutoenkoder_1}')
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
    plt.savefig('/net/people/plgmswieszek/GenerowanieObrazkow/autoenkoders/autoencoder.jpg')


if __name__ == '__main__':
    main()

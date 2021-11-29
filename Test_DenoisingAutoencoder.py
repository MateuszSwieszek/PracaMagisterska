import numpy as np
import glob
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from DenoisingAutoenkoder_5 import binary, dice_coef, dice_coef_loss
from tensorflow.keras.optimizers import Adam



def wczytanieObrazow(labelsDIR1='Orgs/masks/', masks=False):
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


def prepareData(masks):
    print("###########Przygotowywanie danych - dzielenie na zestawy testowy walidacyjny itp oraz dodanie szumu########")
    np.random.seed(1337)

    # generate corrupted MNIST images by adding noise with normal dist
    # centered at 0.5 and std=0.5
    noise = np.random.normal(loc=0.5, scale=0.5, size=masks.shape)
    masks_noisy = masks + noise

    # adding noise may exceed normalized pixel values>1.0 or <0.0
    # clip pixel values >1.0 to 1.0 and <0.0 to 0.0
    for i in range(masks_noisy.shape[0]):
        masks_noisy[i] = np.clip(masks_noisy[i], 0., 1.)

    return masks_noisy


def run():
    # maski binarne Unet
    masks = wczytanieObrazow('/Users/mateusz/PycharmProjects/PracaMagisterska/NiedoskonalaSegmentacja/',masks=True)

    autoencoder = binary()
    autoencoder.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])
    autoencoder.load_weights('autoenkoders/DenoisingAutoenkoder_5.hdf5')

    x_decoded = autoencoder.predict(masks)

    # 3 sets of images with 9 MNIST digits
    # 1st rows - original images
    # 2nd rows - images corrupted by noise
    # 3rd rows - denoised images
    rows, cols = 3, 9
    num = rows * cols
    # imgs = np.concatenate([masks[:num], masks_noisy[:num], x_decoded[:num]])
    imgs = np.concatenate([masks[:num], x_decoded[:num]])
    imgs = imgs.reshape((rows * 2, cols, 128, 128))
    imgs = np.vstack(np.split(imgs, rows, axis=1))
    imgs = imgs.reshape((rows * 2, -1, 128, 128))
    imgs = np.vstack([np.hstack(i) for i in imgs])

    plt.figure()
    plt.axis('off')
    plt.title('Original images: top rows, '
              'Corrupted Input: middle rows, '
              'Denoised Input:  third rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.show()


if __name__ == '__main__':
    run()

from pycimg import CImg
import numpy as np
import glob
import cv2

def showImage(im, isGray=False):
    '''funkcja  do wyswietlania obrazu'''
    img = CImg(im)
    img.display()

def WczytywaniePunktow(POINTSDIR1):
    print('################################################################################################')
    print("Wczytywanie punktow")

    points = glob.glob(POINTSDIR1)

    f = open(points[0], 'r')
    XY = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
    f.close()
    FlattenXY = XY.reshape(-1, XY.shape[0] * XY.shape[1])[0]


    print("wymiar XY: ", np.shape(XY))
    print("wymiar XYFlatten: ", np.shape(FlattenXY))

    return XY, FlattenXY

def wczytanieObrazok(labelsDIR1='Orgs/masks/', masks=True):
    '''wczytanie mask'''

    images1 = glob.glob(labelsDIR1)

    image = cv2.imread(images1[0])
    if (masks == False):
        image = image[:, :, 0:1] / 255
    elif (np.unique(image)[1] == 255):
        image = image[:, :, 0:1] / 255
    else:
        image = image[:, :, 0:1]
    images = image

    return images


def TestWyswietlaniaZdjec(images, XY):



    # for p in reconPoints:
    #     test1[int(p[0]), int(p[1])] = 0

    for p in range(100):
        images[int(XY[p, 0]), int(XY[p, 1])] = 0

    showImage(images)
    # showImage(test1)
    # showImage(test2)

def run():
    # XY, FlattenXY = WczytywaniePunktow('point_7805.dat')
    image = wczytanieObrazok('Orgs/masks/mask_1004244319.bmp')
    image2 = wczytanieObrazok('mask_alignedImg_0.bmp')
    # TestWyswietlaniaZdjec(image,XY)

    print(np.unique(image)[1])
    print(np.unique(image2)[1])

if __name__ == '__main__':
    run()
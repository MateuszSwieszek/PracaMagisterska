import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from pycimg import CImg


def showImage(image, isGray=False):
    '''funkcja  do wyświetlania obrazu'''
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap = 'gray')
    plt.show()


def wczytanieMask(labelsDIR1 = 'Orgs/masks/', labelsDIR2 = 'Generated/masks/'):
    '''wczytanie mask'''

    masks1 = glob.glob(labelsDIR1 + '*.bmp')
    masks1.sort()
    print('################################################################################################')
    print("Wczytywanie mask")
    print('Długość mask z pierwszej ścieżki: ',len(masks1))

    masks2 = glob.glob(labelsDIR2 + '*.bmp')
    masks2.sort()
    print('Długość mask z drugiej ścieżki: ', len(masks2))
    masks = masks1 + masks2
    return masks


def WczytywaniePunktow(POINTSDIR1,POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktów")


    points1 = glob.glob(POINTSDIR1 + '*.dat')
    points1.sort()


    points2 = glob.glob(POINTSDIR2 + '*.dat')
    points2.sort()

    points = points1 + points2

    print('Długość punktów z pierwszej ścieżki: ', len(points1))
    print('Długość punktów z drugiej ścieżki: ', len(points2))
    return points

def WczytywanieObrazow(POINTSDIR1,POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie obrazów")

    points1 = glob.glob(POINTSDIR1 + '*.bmp')
    points1.sort()
    print(len(points1))

    points2 = glob.glob(POINTSDIR2 + '*.bmp')
    points2.sort()
    print(len(points2))
    images = points1 + points2
    print('Długość obrazów z pierwszej ścieżki: ', len(points1))
    print('Długość obrazów z drugiej ścieżki: ', len(points2))
    return images

def WczytywaniePunktowSTD(POINTSDIR1,POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktów std")

    points1 = glob.glob(POINTSDIR1 + '*.dat')
    points1.sort()
    print(len(points1))

    points2 = glob.glob(POINTSDIR2 + '*.dat')
    points2.sort()
    print(len(points2))
    stdpoints = points1 + points2

    print('Długość punktów std z pierwszej ścieżki: ', len(points1))
    print('Długość punktów std z drugiej ścieżki: ', len(points2))
    return stdpoints

def WczytywaniePunktowHomography(POINTSDIR1,POINTSDIR2):
    print('################################################################################################')
    print("Wczytywanie punktów macierzy homografii")

    points1 = glob.glob(POINTSDIR1 + '*.dat')
    points1.sort()
    print(len(points1))

    points2 = glob.glob(POINTSDIR2 + '*.dat')
    points2.sort()
    print(len(points2))
    homographyPoints = points1 + points2
    print('Długość punktów h z pierwszej ścieżki: ', len(points1))
    print('Długość punktów h z drugiej ścieżki: ', len(points2))

    return homographyPoints





def GenerowaniePunktow(masks):
    print('################################################################################################')
    print("Generowanie punktów  na konturach")
    print("nazwy wygenerowanych punktów: ")

    '''Generowanie punktów na konturach
    '''
    NPOINTS = 100

    for imgName in masks[:]:

        img = cv2.imread(imgName)
        img = img[:, :, 0]
        img[img != 0] = 255

        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        cv2.floodFill(im_floodfill, mask, (w - 1, 0), 255)
        cv2.floodFill(im_floodfill, mask, (0, h - 1), 255)
        cv2.floodFill(im_floodfill, mask, (w - 1, h - 1), 255)

        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = img | im_floodfill_inv

        ret, labels = cv2.connectedComponents(im_out, connectivity=4)
        unique, counts = np.unique(labels, return_counts=True)
        lmax = np.argmax(counts[1:]) + 1
        im_out[labels != lmax] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_out, 4, cv2.CV_32S)

        RIGHT = stats[1][cv2.CC_STAT_LEFT] + stats[1][cv2.CC_STAT_WIDTH]

        contours, _ = cv2.findContours(im_out.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lens = np.asarray([len(contour) for contour in contours], dtype=np.int16)
        m = np.argmax(lens)

        lista = np.where(contours[m][:, 0, 0] >= RIGHT - 1)
        START_INDEX = lista[0][0]
        #    print(contours[m][START_INDEX,0,:],contours[m].shape[0])

        STEP = contours[m].shape[0] / NPOINTS

        indices = [int(START_INDEX + n * STEP) % contours[m].shape[0] for n in range(NPOINTS)]
        # print(len(indices))

        #    print(contours[m][lista,0,:])
        im_out[im_out == 255] = 64
        for n, index in enumerate(indices):
            im_out[contours[m][index, 0, 1], contours[m][index, 0, 0]] = 255

        # showImage(im_out)

        base = os.path.basename(imgName)

        id = os.path.splitext(base)[0].split('_')[-1]


        POINTS_DIR = os.path.splitext(imgName)[0].split('masks')[0]
        print("bazowa ścieżka: ",POINTS_DIR)
        print("id zdjęcia: ",id)
        name = POINTS_DIR + 'Points/points/point_' + id + '.dat'
        print(name)
        f = open(name, 'w')
        for n, index in enumerate(indices):
            f.write(' '.join((str(n), str(contours[m][index, 0, 1]), str(contours[m][index, 0, 0]))) + '\n')
        f.close()



def GenerowaniePunktowSTD(points):
    print('################################################################################################')
    print("Generowanie punktów std i macierzy homografii ")
    print("nazwy wygenerowanych punktów")
    # %%

    ###etam treningu
    '''Procrustes analysis "normalizacja punktów" '''





    f1 = points[0]
    print('reference point set ', f1)
    f = open(f1, 'r')
    lines1 = np.array([list(map(int, l.split(' ')[1:])) for l in f.read().splitlines()])
    f.close()



    for f2 in points[1:]:

        f = open(f2, 'r')
        lines2 = np.array([list(map(int, l.split(' ')[1:])) for l in f.read().splitlines()])
        f.close()

        mtx1, mtx2, disparity = procrustes(lines1, lines2)

        dst_pts = np.float32(lines2).reshape(-1, 1, 2)
        src_pts = np.float32(mtx2).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # test
        l2 = np.ones((lines2.shape[0], lines2.shape[1] + 1), dtype=np.float32)
        np.copyto(l2[:, 0:2], mtx2)
        l2t = l2.transpose()
        a = np.matmul(M, l2t).transpose()[:, 0:2]
        diff = np.matmul((a - lines2).transpose(), a - lines2)
        # print(diff[0, 0] + diff[1, 1])  # should be close to zero

        base = os.path.basename(f2)
        id = os.path.splitext(base)[0].split('_')[-1]

        DIR = os.path.splitext(f2)[0].split('points')[0]

        print("bazowa ścieżka: ", DIR)
        print("id zdjęcia: ", id)


        name = DIR + 'stdPoints/stdPoint_' + id + '.dat'
        f = open(name, 'w')
        for n in range(mtx2.shape[0]):
            f.write(' '.join((str(n), str(mtx2[n, 0]), str(mtx2[n, 1]))) + '\n')
        f.close()

        name = DIR + 'homographyPoints/homography_' + id + '.dat'
        f = open(name, 'w')
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                f.write(str(M[r, c]) + '\n')
        f.close()

    print('Ostatni punkt', f1)


    DIR = os.path.splitext(f1)[0].split('points')[0]
    base = os.path.basename(f1)
    id = os.path.splitext(base)[0].split('_')[-1]
    name = DIR + 'stdPoints/stdPoint_' + id + '.dat'
    print('ostatni std: ', name)
    f = open(name, 'w')
    for n in range(mtx1.shape[0]):
        f.write(' '.join((str(n), str(mtx1[n, 0]), str(mtx1[n, 1]))) + '\n')
    f.close()

    dst_pts = np.float32(lines1).reshape(-1, 1, 2)
    src_pts = np.float32(mtx1).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    name = DIR + 'homographyPoints/homographyPoint_' + id + '.dat'
    print('ostatni h: ', name)

    f = open(name, 'w')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            f.write(str(M[r, c]) + '\n')
    f.close()


def SavePCA(pca):
    print('################################################################################################')
    print("zapis PCA do pliku ")
    print("rozmiar PCA: ", pca.shape)
    name = 'PCA.dat'
    f = open(name, 'w')
    for r in range(pca.shape[0]):
        for c in range(pca.shape[1]):
            f.write(str(pca[r, c]) + '\n')
    f.close()
def SaveMean(mean):
    print('################################################################################################')
    print("zapis średniej do pliku ")
    name = 'mean.dat'
    f = open(name, 'w')
    for r in range(mean.shape[0]):
        # print("pojedyńczy element mean: ",r)
        f.write(str(mean[r]) + '\n')
    f.close()

def ReadMean():
    print('################################################################################################')
    print("odczyt mean z pliku ")

    name = 'mean.dat'
    pcaComponents = np.zeros((200,), dtype=np.float32)
    f = open(name, 'r')
    for r in range(pcaComponents.shape[0]):
        el = float(f.readline())
        pcaComponents[r] = el
    f.close()
    return pcaComponents

def ReadPCA():
    print('################################################################################################')
    print("odczyt PCA z pliku ")

    name = 'PCA.dat'
    pcaComponents = np.zeros((52, 200), dtype=np.float32)
    f = open(name, 'r')
    for r in range(pcaComponents.shape[0]):
        for c in range(pcaComponents.shape[1]):
            el = float(f.readline())
            pcaComponents[r, c] = el
    f.close()
    return pcaComponents

def PCATransform(points, stdPoints, homographyPoints, images):
    print('################################################################################################')
    print("Generowanie transformacji PCA ")
    print("liczba wszystkich punktów: ", len(stdPoints))
    # PCA

    EXPLAINED_RATIO = 0.99



    # print(np.shape(points))

    X = []
    for filename in stdPoints:
        f = open(filename, 'r')
        lines = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
        f.close()
        lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
        X.append(lines)
    # print(np.shape(X))

    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X, 0)
    print("mean shape ", np.shape(mean)[0])
    SaveMean(mean)
    meanTest = ReadMean()
    test, counts = np.unique(meanTest - mean, return_counts=True)
    print('test mean: ', test)

    '''to jest istotne'''
    for r in range(X.shape[0]):
        X[r] = X[r] - mean

    pca = PCA(n_components=EXPLAINED_RATIO, svd_solver='full')
    pca.fit(X)

    # print(np.sum(pca.explained_variance_ratio_))
    X_projected = pca.transform(X)
    # print(X_projected.shape)
    X_recon = pca.inverse_transform(X_projected)
    # print(X_recon.shape)
    diff = np.sum((X_recon - X) * (X_recon - X), -1)
    # print(diff.shape)
    # print(diff)
    print("wymiary macierzy punktów po PCA ",np.shape(X_projected))
    print("wymiary macierzy punktów ", np.shape(X))
    # print(np.shape(pca.components_))
    # print(X[0])

    '''TESTY'''
    print("TESTY")
    pcaComponents = ReadPCA()
    print("PCA shape: ",np.shape(pcaComponents))

    test,counts = np.unique(pcaComponents-pca.components_,return_counts=True)
    print('test PCA: ', test)


    # # PCA Transform
    # pca.components_ to model pca do wyznaczania transformacji
    TESTED = 10  ##obraz

    # print(pca.components_.shape)
    r = np.zeros(X_projected[TESTED].shape, dtype=np.float32)
    for row in range(pca.components_.shape[0]):
        r[row] = np.sum(pca.components_[row] * X[TESTED])
    # print(r - X_projected[TESTED])

    # PCA inverse transform


    r = np.zeros(X_recon[TESTED].shape, dtype=np.float32)
    print(pca.components_.shape)
    for col in range(pca.components_.shape[1]):
        r[col] = np.sum(pca.components_[:, col] * X_projected[TESTED])
    # print(r - X_recon[TESTED])

    # print(points[TESTED])
    base = os.path.basename(stdPoints[TESTED])
    id = os.path.splitext(base)[0].split('_')[1]

    name = homographyPoints[TESTED]

    M = np.zeros((3, 3), dtype=np.float32)
    f = open(name, 'r')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            el = float(f.readline())
            M[r, c] = el
    f.close()

    # print(M)
    print("wymiary X_recon[TESTED]: ", np.shape(X_recon[TESTED]))

    X_ = X_recon[TESTED] + mean  #
    X_ = X_.reshape(-1, 2)
    print("wymiary X_recon[TESTED] po transpozycji: ", np.shape(X_))


    l2 = np.ones((X_.shape[0], X_.shape[1] + 1), dtype=np.float32)
    np.copyto(l2[:, 0:2], X_)
    l2t = l2.transpose()
    print("wymiary l2t: ", np.shape(l2t))

    a = np.matmul(M, l2t).transpose()[:, 0:2]
    # print(a)

    name = images[TESTED]
    im = cv2.imread(name)

    for p in range(a.shape[0]):
        im[int(a[p, 0]), int(a[p, 1]), :] = 0

    name = points[TESTED]
    f = open(name, 'r')
    X_org = np.array([list(map(int, l.split(' ')[1:])) for l in f.read().splitlines()])
    f.close()

    for p in range(X_org.shape[0]):
        im[int(X_org[p, 0]), int(X_org[p, 1]), 0] = 128

    # showImage(im)
    print("Rozmiar obrazu:", np.shape(im))
    img = CImg(im)
    img.display()


def main():

    # masks = wczytanieMask(labelsDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/masks/',
    #                       labelsDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/')

    # # GenerowaniePunktow(masks)
    # points = WczytywaniePunktow(POINTSDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/points/',
    #                             POINTSDIR2='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/points')
    #
    #
    #
    # # GenerowaniePunktowSTD(points)
    #
    # stdPoints = WczytywaniePunktowSTD(POINTSDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/stdPoints/',
    #                             POINTSDIR2='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/stdPoints')
    #
    # homographyPoints = WczytywaniePunktowHomography(POINTSDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/homographyPoints/',
    #                             POINTSDIR2='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/Points/stdPoints/homographyPoints')
    #
    # images = WczytywanieObrazow(POINTSDIR1='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/orgs/',
    #                             POINTSDIR2='/Users/mateusz/PycharmProjects/PracaMagisterska/Orgs/orgs')
    #
    #
    #
    # PCATransform(points=points,stdPoints=stdPoints,homographyPoints=homographyPoints,images=images)

    points = WczytywaniePunktow(POINTSDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/points/',
                                POINTSDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/points/')

    # GenerowaniePunktowSTD(points)

    stdPoints = WczytywaniePunktowSTD(
        POINTSDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/stdPoints/',
        POINTSDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/stdPoints/')

    homographyPoints = WczytywaniePunktowHomography(
        POINTSDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/homographyPoints/',
        POINTSDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/homographyPoints/')

    images = WczytywanieObrazow(POINTSDIR1='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/orgs/',
                                POINTSDIR2='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/')

    PCATransform(points=points, stdPoints=stdPoints, homographyPoints=homographyPoints, images=images)

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from sklearn.decomposition import PCA
from scipy.spatial import procrustes


def showImage(image, isGray=False):
    '''funkcja  do wyświetlania obrazu'''
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap = 'gray')
    plt.show()



def wczytanieMask(labelsDIR = 'Images/masks/'):
    '''wczytanie mask'''

    masks = glob.glob(labelsDIR + '*.bmp')
    masks.sort()
    print(len(masks))
    return masks



def GenerowaniePunktowSTD(masks,
                          POINTS_DIR='Images/Points/points/',
                          STDPOINTS_DIR = './Images/Points/stdPoints/',
                          HPOINTS_DIR = './Images/Points/homographyPoints/',
                          f1 = None):

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

        id = os.path.splitext(base)[0].split('_')[1]
        print(id)
        name = POINTS_DIR + 'point_' + id + '.dat'
        print(name)
        f = open(name, 'w')
        for n, index in enumerate(indices):
            f.write(' '.join((str(n), str(contours[m][index, 0, 1]), str(contours[m][index, 0, 0]))) + '\n')
        f.close()

    # %%

    ###etam treningu
    '''Procrustes analysis "normalizacja punktów" '''


    points = glob.glob(POINTS_DIR + 'point*.dat')
    points.sort()
    print(len(points))

    if (f1==None):
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
        id = os.path.splitext(base)[0].split('_')[1]
        name = STDPOINTS_DIR + 'stdPoint_' + id + '.dat'
        f = open(name, 'w')
        for n in range(mtx2.shape[0]):
            f.write(' '.join((str(n), str(mtx2[n, 0]), str(mtx2[n, 1]))) + '\n')
        f.close()

        name = HPOINTS_DIR + 'homography_' + id + '.dat'
        f = open(name, 'w')
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                f.write(str(M[r, c]) + '\n')
        f.close()

    base = os.path.basename(f1)
    id = os.path.splitext(base)[0].split('_')[1]
    name = STDPOINTS_DIR + 'stdPoint_' + id + '.dat'
    f = open(name, 'w')
    for n in range(mtx1.shape[0]):
        f.write(' '.join((str(n), str(mtx1[n, 0]), str(mtx1[n, 1]))) + '\n')
    f.close()

    dst_pts = np.float32(lines1).reshape(-1, 1, 2)
    src_pts = np.float32(mtx1).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    name = HPOINTS_DIR + '/homographyPoint_' + id + '.dat'
    f = open(name, 'w')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            f.write(str(M[r, c]) + '\n')
    f.close()
    return f1

def SavePCA(pca):
    name = 'PCA.dat'
    f = open(name, 'w')
    for r in range(pca.shape[0]):
        for c in range(pca.shape[1]):
            f.write(str(pca[r, c]) + '\n')
    f.close()
def SaveMean(mean):
    name = 'mean.dat'
    f = open(name, 'w')
    f.write(str(mean) + '\n')
    f.close()

def ReadPCA():

    name = 'PCA.dat'
    pcaComponents = np.zeros((39, 200), dtype=np.float32)
    f = open(name, 'r')
    for r in range(pcaComponents.shape[0]):
        for c in range(pcaComponents.shape[1]):
            el = float(f.readline())
            pcaComponents[r, c] = el
    f.close()
    return pcaComponents

def PCATransform(POINTS_DIR='Images/Points/points/'
                 ,STDPOINTS_DIR = './Images/Points/stdPoints/',
                 HPOINTS_DIR = './Images/Points/homographyPoints/',
                 IMAGESDIR='./Images/orgs/'):
    # PCA

    EXPLAINED_RATIO = 0.99


    points = glob.glob(STDPOINTS_DIR + 'std*')
    points.sort()
    # print(np.shape(points))

    X = []
    for filename in points:
        f = open(filename, 'r')
        lines = np.array([list(map(float, l.split(' ')[1:])) for l in f.read().splitlines()])
        f.close()
        lines = lines.reshape(-1, lines.shape[0] * lines.shape[1])[0]
        X.append(lines)
    # print(np.shape(X))

    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X, 0)
    SaveMean(mean)
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
    print(np.shape(X_projected))
    print(np.shape(X))
    # print(np.shape(pca.components_))
    # print(X[0])

    '''TESTY'''
    SavePCA(pca.components_)
    pcaComponents = ReadPCA()

    test,counts = np.unique(pcaComponents-pca.components_,return_counts=True)
    print(test)

    # # PCA Transform
    # pca.components_ to model pca do wyznaczania transformacji
    TESTED = 40  ##obraz

    # print(pca.components_.shape)
    r = np.zeros(X_projected[TESTED].shape, dtype=np.float32)
    for row in range(pca.components_.shape[0]):
        r[row] = np.sum(pca.components_[row] * X[TESTED])
    # print(r - X_projected[TESTED])

    # PCA inverse transform

    TESTED = 40
    r = np.zeros(X_recon[TESTED].shape, dtype=np.float32)
    print(pca.components_.shape)
    for col in range(pca.components_.shape[1]):
        r[col] = np.sum(pca.components_[:, col] * X_projected[TESTED])
    # print(r - X_recon[TESTED])

    TESTED = 40
    # print(points[TESTED])
    base = os.path.basename(points[TESTED])
    id = os.path.splitext(base)[0].split('_')[1]

    name = HPOINTS_DIR + 'homography_' + id + '.dat'

    M = np.zeros((3, 3), dtype=np.float32)
    f = open(name, 'r')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            el = float(f.readline())
            M[r, c] = el
    f.close()

    # print(M)

    X_ = X_recon[TESTED] + mean  #
    X_ = X_.reshape(-1, 2)

    l2 = np.ones((X_.shape[0], X_.shape[1] + 1), dtype=np.float32)
    np.copyto(l2[:, 0:2], X_)
    l2t = l2.transpose()
    a = np.matmul(M, l2t).transpose()[:, 0:2]
    # print(a)

    name = IMAGESDIR + 'org_' + id + '.bmp'
    im = cv2.imread(name)

    for p in range(a.shape[0]):
        im[int(a[p, 0]), int(a[p, 1]), :] = 0

    name = POINTS_DIR + 'point_' + id + '.dat'
    f = open(name, 'r')
    X_org = np.array([list(map(int, l.split(' ')[1:])) for l in f.read().splitlines()])
    f.close()

    for p in range(X_org.shape[0]):
        im[int(X_org[p, 0]), int(X_org[p, 1]), 0] = 128

    showImage(im)

def main():

    masks = wczytanieMask(labelsDIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/masks/')
    f1 = GenerowaniePunktowSTD(masks,POINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/points/',
                               STDPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/stdPoints/',
                               HPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/homographyPoints/')
    PCATransform(POINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/points/',
                               STDPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/stdPoints/',
                               HPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/homographyPoints/',
                 IMAGESDIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/orgs/')

    masks = wczytanieMask(labelsDIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/')

    f1 = GenerowaniePunktowSTD(masks,POINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/points/',
                               STDPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/stdPoints/',
                               HPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/homographyPoints/',f1=f1)


    PCATransform(POINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/points/',
                               STDPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/stdPoints/',
                               HPOINTS_DIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/homographyPoints/',
                 IMAGESDIR='/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/')

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob


def preprocessing():
    # Open the image files.
    img1_color = cv2.imread("org.bmp")  # Image to be aligned.
    img2_color = cv2.imread("mask.bmp")  # Reference image.
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
    print(img1.dtype)

    '''Tutaj moja wstawka'''
    img1 = img1.astype('uint8')
    blur = cv2.GaussianBlur(img1, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img1 = np.invert(th3)

    img2 = img2.astype('uint8')
    blur = cv2.GaussianBlur(img2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img2 = np.invert(th3)

    '''Koniec wstawki'''

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # print(d2)
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                          homography, (width, height))

    # Save the output.
    cv2.imwrite('output.jpg', transformed_img)


def align(im_align, im_ref, mask):
    TH = 30
    w = 0.75

    mask[mask != 1] = 0
    mask = mask * 255

    sz = im_ref.shape
    height = sz[0]
    width = sz[1]

    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY
    # warp_mode = cv2.MOTION_AFFINE

    # Set the warp matrix to identity.
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # Warp the blue and green channels to the red channel
    (cc, warp_matrix) = cv2.findTransformECC(im_ref[:, :, 0], im_align[:, :, 0], warp_matrix, warp_mode, criteria, None, gaussFiltSize=1)

    unit = np.eye(3, 3, dtype=np.float32)
    warp_matrix = np.add(w * warp_matrix, (1 - w) * unit)

    # Use Perspective warp when the transformation is a Homography
    im_aligned = cv2.warpPerspective(im_align, warp_matrix, (width, height),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    mask_aligned = cv2.warpPerspective(mask, warp_matrix, (width, height),
                                       flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    # showOpencvImage(im_align//2 + mask//2)

    im_aligned = im_aligned[:, :, 0]
    mask_aligned = mask_aligned[:, :, 0]
    dum = im_aligned <= TH
    dum = np.asarray((im_ref[dum])[:, 0])
    im_aligned[im_aligned <= TH] = dum


    # mask_aligned = mask_aligned/255

    return im_aligned, mask_aligned


def main():
    preprocessing()

    SAVE_DIR_IMG = "/net/people/plgmswieszek/GenerowanieObrazkow/AlignedImages/masks/"
    SAVE_DIR_MASK = "/net/people/plgmswieszek/GenerowanieObrazkow/AlignedImages/images/"


    masks = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/masks/mask*.bmp")
    orgs = glob.glob("/net/people/plgmswieszek/GenerowanieObrazkow/Images/orgs/org*.bmp")

    masks.sort()
    orgs.sort()
    print('dupa1')

    N = len(masks)
    # N = 40
    print(N)
    STEP = 1

    num = 2
    for n1 in range(51, N - STEP, STEP):
        im_ref = cv2.imread(orgs[n1])
        if (n1 == 51):
            for n2 in range(66, N, STEP):
                im_align = cv2.imread(orgs[n2])
                mask = cv2.imread(masks[n2])
                im_aligned, mask_aligned = align(im_align, im_ref, mask)
                cv2.imwrite(SAVE_DIR_IMG + 'alignedImg_' + str(num) + '.bmp', im_aligned)
                cv2.imwrite(SAVE_DIR_MASK + 'mask_alignedImg_' + str(num) + '.bmp', mask_aligned)
                # print(im_aligned.shape, mask_aligned.shape)
                im = im_aligned // 2 + (mask_aligned) // 2
                # showOpencvImage(im)
                num = num + 1
                print(num)
                if num == 3:
                    break
        for n2 in range(n1 + STEP, N, STEP):
            im_align = cv2.imread(orgs[n2])
            mask = cv2.imread(masks[n2])
            im_aligned, mask_aligned = align(im_align, im_ref, mask)
            cv2.imwrite(SAVE_DIR_IMG + 'alignedImg_' + str(num) + '.bmp', im_aligned)
            cv2.imwrite(SAVE_DIR_MASK + 'mask_alignedImg_' + str(num) + '.bmp', mask_aligned)
            # print(im_aligned.shape, mask_aligned.shape)
            im = im_aligned // 2 + (mask_aligned) // 2
            # showOpencvImage(im)
            num = num + 1
            print(num)
            if num == 3:
                break

if __name__ == '__main__':
    main()
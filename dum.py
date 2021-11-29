import sys 
import cv2 
import numpy as np 
 
# Extract all the contours from the image 
def get_all_contours(img): 
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
#    im2, contours = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return contours
 
if __name__=='__main__': 
    # Input image containing all the different shapes 
    img1 = cv2.imread('convex_shapes.png')
    ref_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    thresh = 255 - thresh

    contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    thresh[thresh==255] = 128

    for contour in contours:
        print(contour.shape)
        for n in range(contour.shape[0]):
            thresh[contour[n,0,1],contour[n,0,0]] = 255


    # Extract all the contours from the input image 

    contour_img = img1.copy()
    smoothen_contours = []
    factor = 0.001

    # Finding the closest contour 
    for contour in contours: 
        epsilon = factor * cv2.arcLength(contour, True) 
#        smoothen_contours.append(cv2.approxPolyDP(contour, epsilon, True)) 
        smoothen_contours.append(cv2.approxPolyDP(contour, 5, True)) 
 
    cv2.drawContours(contour_img, smoothen_contours, -1, color=(0,0,0), thickness=3) 
    cv2.imshow('Contours', contour_img)
    cv2.waitKey() 

    cv2.imshow('Contours', thresh)
    cv2.waitKey() 


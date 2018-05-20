import cv2

import setup
from core import equalize_red, extract_rect, faces
from config import DATASET_PATH

""" --- MAIN --- """

img1_name = DATASET_PATH + '2-11.jpg'
img2_name = DATASET_PATH + '26-11.jpg'

img1 = cv2.imread(img1_name)
img2 = cv2.imread(img2_name)

img1 = extract_rect(img1, faces(img1)[0])
img2 = extract_rect(img2, faces(img2)[0])

if __name__ == '__main__':
    cv2.imshow('reference', img2)
    cv2.imshow('original', img1)

    cv2.imwrite('reference.jpg', img2)
    cv2.imwrite('original.jpg', img1)

    img1 = equalize_red(img1, img2)
    cv2.imshow('equalized', img1)
    cv2.imwrite('equalized.jpg', img1)
    cv2.waitKey(0)

import cv2

import setup
from core import equalize_red
from config import DATASET_PATH

""" --- MAIN --- """

img1_name = DATASET_PATH + '1-11.jpg'
img2_name = DATASET_PATH + '2-11.jpg'

img1 = cv2.imread(img1_name)
img2 = cv2.imread(img2_name)

if __name__ == '__main__':
    new = equalize_red(img1, img2)

import cv2

import setup
from core import compare_with_avg
from config import IMG_PATH

""" --- MAIN --- """

img_name = '.jpg'
if __name__ == '__main__':
    img = cv2.imread(IMG_PATH + img_name)
    compare_with_avg(img)
